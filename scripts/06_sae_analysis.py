#!/usr/bin/env python3
"""
Script 06: SAE Feature Analysis

SAE decomposition, delta scoring, interpretation, causal validation.
Skips automatically for models without a pre-trained SAE (e.g., Aya).

Usage:
    python scripts/06_sae_analysis.py \
        --model llama \
        --critical-layers results/attribution/critical_layers.json \
        --output-dir results/sae_features/
"""

import argparse
import getpass
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.sae.delta_scores import compute_delta_scores, rank_features, feature_analysis_table
from src.sae.feature_extract import load_sae, encode_activations
from src.sae.interpret import load_feature_labels
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


AVAILABLE_SAE_LAYERS_BY_MODEL = {
    "llama": [23, 29],
    "qwen": [3, 7, 11, 15, 19, 23, 27],
}


def choose_sae_layer(critical_layers, model_key):
    available = AVAILABLE_SAE_LAYERS_BY_MODEL.get(model_key, [23])
    if not critical_layers:
        return available[0]
    best = min(available, key=lambda sl: min(abs(sl - cl) for cl in critical_layers))
    return best


def _activation_cache_component(hook_component: str) -> str:
    comp = (hook_component or "").lower().strip()
    if comp in {"mlp", "mlp_out"}:
        return "mlp_out"
    if comp in {"attn", "attn_out"}:
        return "attn_out"
    if comp in {"resid", "residual", "resid_post", "hidden", "hidden_state"}:
        return "residual"
    raise ValueError(f"Unsupported SAE hook component: {hook_component}")


def _result_hookpoint(model_key: str, layer: int, hook_component: str) -> str:
    comp = (hook_component or "").lower().strip()
    if model_key == "qwen" and comp in {"resid", "residual", "resid_post", "hidden", "hidden_state"}:
        return f"resid_post_layer_{layer}"
    if comp in {"mlp", "mlp_out"}:
        return f"layers.{layer}.mlp"
    if comp in {"attn", "attn_out"}:
        return f"layers.{layer}.attn"
    return f"layers.{layer}.residual"


def parse_args():
    parser = argparse.ArgumentParser(description="SAE feature analysis.")
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "aya"])
    parser.add_argument("--critical-layers", required=True,
                        help="Path to critical_layers.json from script 05.")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--output-dir", default="results/sae_features/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--comparison-language", default="ar")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("sae_analysis")
    setup_reproducibility(seed=42)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gate: skip models without a pre-trained SAE
    if not model_cfg.get("has_sae", False):
        logger.info(
            f"Model {args.model} ({model_name}) has no pre-trained SAE. Skipping SAE analysis."
        )
        with open(output_dir / "sae_skipped.json", "w") as f:
            json.dump({
                "model": args.model,
                "model_name": model_name,
                "reason": "no_pretrained_sae",
            }, f, indent=2)
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        hf_token = getpass.getpass("Enter your Hugging Face token: ").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    with open(args.critical_layers) as f:
        critical_data = json.load(f)
    critical_layers = sorted(set(critical_data.get("critical_layers", [])))
    logger.info(f"Critical layers: {critical_layers}")

    if args.dry_run:
        logger.info("[DRY RUN] Would run SAE feature analysis.")
        return

    from src.activations.cache import load_activations, get_activation_path

    primary_layer = choose_sae_layer(critical_layers, args.model)
    logger.info(f"Primary SAE layer: {primary_layer}")
    activation_component = model_cfg.get("sae_hook_component", "mlp") or "mlp"
    activation_cache_component = _activation_cache_component(activation_component)

    def _load_layer(model_key, lang, layer):
        path = get_activation_path(
            model_key, lang, args.perturbation, args.token_position,
            activation_cache_component, args.activations_dir
        )
        if not Path(path).exists():
            logger.error(f"{lang} activations not found: {path}")
            sys.exit(1)
        acts = load_activations(path)
        if acts.ndim == 3:
            return acts[:, layer, :].to(torch.float32)
        return acts.to(torch.float32)

    en_layer = _load_layer(args.model, "en", primary_layer)
    lang_layer = _load_layer(args.model, args.comparison_language, primary_layer)

    # Filter to harmful-only so delta scores reflect the harmful population
    try:
        full_df = load_dataset(dataset_dir=args.dataset_dir, perturbations=[args.perturbation])
        en_df = full_df[full_df["language"] == "en"].reset_index(drop=True)
        lang_df = full_df[full_df["language"] == args.comparison_language].reset_index(drop=True)
        if len(en_df) == len(en_layer):
            en_harm_mask = en_df["is_harmful"].values if "is_harmful" in en_df.columns else (en_df.index < len(en_df) // 2)
            en_layer = en_layer[torch.from_numpy(en_harm_mask).bool()]
        if len(lang_df) == len(lang_layer):
            lang_harm_mask = lang_df["is_harmful"].values if "is_harmful" in lang_df.columns else (lang_df.index < len(lang_df) // 2)
            lang_layer = lang_layer[torch.from_numpy(lang_harm_mask).bool()]
        logger.info(f"Filtered to harmful-only: en={len(en_layer)}, {args.comparison_language}={len(lang_layer)}")
    except Exception as e:
        logger.warning(f"Could not filter to harmful-only: {e}. Using all activations.")

    logger.info(f"Loading SAE for layer {primary_layer}...")
    try:
        sae = load_sae(model_name, primary_layer, hook_component=activation_component)
    except Exception as e:
        logger.error(f"Could not load SAE: {e}")
        sys.exit(1)

    en_features = encode_activations(sae, en_layer)
    lang_features = encode_activations(sae, lang_layer)

    delta = compute_delta_scores(en_features, lang_features)
    lang = args.comparison_language
    np.save(str(output_dir / f"delta_scores_{lang}_layer{primary_layer}.npy"), delta)

    ranked_idx = rank_features(delta, args.top_k)
    labels = load_feature_labels(
        model_name=model_name,
        layer=primary_layer,
        feature_indices=ranked_idx,
        cache_dir=str(output_dir / "neuronpedia_cache"),
        use_neuronpedia=True,
    )

    table_df = feature_analysis_table(delta, args.top_k, labels)
    table_df.to_csv(output_dir / f"top_features_{lang}_layer{primary_layer}.csv", index=False)
    logger.info(f"Top {args.top_k} features identified.")
    logger.info(table_df.head(10).to_string(index=False))

    with open(output_dir / "ranked_features.json", "w") as f:
        json.dump({
            "ranked_features": [int(x) for x in ranked_idx],
            "layer": int(primary_layer),
            "comparison_language": lang,
            "critical_layers_sorted": [int(x) for x in critical_layers],
            "sae_repo": model_cfg.get("sae_repo", ""),
            "hookpoint": _result_hookpoint(args.model, primary_layer, activation_component),
            "activation_component": activation_component,
        }, f, indent=2)

    en_feat_means = en_features.float().mean(dim=0).cpu().numpy()
    np.save(str(output_dir / "en_feature_means.npy"), en_feat_means)

    logger.info(f"SAE analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
