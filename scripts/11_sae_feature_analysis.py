#!/usr/bin/env python3
"""
Script 11: SAE Feature Analysis

SAE decomposition, delta scoring, interpretation, causal validation.
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

from src.sae.delta_scores import compute_delta_scores, rank_features, feature_analysis_table
from src.sae.feature_extract import load_sae, encode_activations
from src.sae.interpret import load_feature_labels
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


AVAILABLE_SAE_LAYERS = [23, 29]


def choose_sae_layer(critical_layers):
    if not critical_layers:
        return 23

    # Pick the critical layer closest to an available SAE layer.
    best = None
    best_dist = None
    for sae_layer in AVAILABLE_SAE_LAYERS:
        dist = min(abs(sae_layer - cl) for cl in critical_layers)
        if best is None or dist < best_dist:
            best = sae_layer
            best_dist = dist
    return best


def parse_args():
    parser = argparse.ArgumentParser(description="SAE feature analysis.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--critical-layers", required=True,
                        help="Path to critical_layers.json.")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--output-dir", default="results/sae_features/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--comparison-language", default="ar")
    parser.add_argument("--hf-token", default=None,
                        help="Optional Hugging Face token. If omitted, the script will prompt for it.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("sae_feature_analysis")
    setup_reproducibility(seed=42)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        hf_token = getpass.getpass("Enter your Hugging Face token: ").strip()

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        logger.info("HF token detected and exported for SAE loading.")
    else:
        logger.warning("No HF token provided. SAE loading may fail.")

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    with open(args.critical_layers) as f:
        critical_data = json.load(f)

    critical_layers = sorted(set(critical_data.get("critical_layers", [])))
    logger.info(f"Critical layers (sorted): {critical_layers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Would run SAE feature analysis.")
        return

    from src.activations.cache import load_activations, get_activation_path

    primary_layer = choose_sae_layer(critical_layers)
    logger.info(f"Primary SAE layer selected from available published SAEs: {primary_layer}")

    activation_component = "mlp_out"

    en_act_path = get_activation_path(
        args.model, "en", args.perturbation, args.token_position, activation_component,
        args.activations_dir
    )
    if not Path(en_act_path).exists():
        logger.error(f"English MLP activations not found: {en_act_path}")
        sys.exit(1)

    en_acts_all = load_activations(en_act_path)
    if en_acts_all.ndim == 3:
        if primary_layer >= en_acts_all.shape[1]:
            logger.error(
                f"Requested layer {primary_layer}, but English activations only have "
                f"{en_acts_all.shape[1]} layers in shape {tuple(en_acts_all.shape)}"
            )
            sys.exit(1)
        en_layer = en_acts_all[:, primary_layer, :]
    else:
        en_layer = en_acts_all
    en_layer = en_layer.to(torch.float32)

    lang = args.comparison_language
    lang_act_path = get_activation_path(
        args.model, lang, args.perturbation, args.token_position, activation_component,
        args.activations_dir
    )
    if not Path(lang_act_path).exists():
        logger.error(f"{lang} MLP activations not found: {lang_act_path}")
        sys.exit(1)

    lang_acts_all = load_activations(lang_act_path)
    if lang_acts_all.ndim == 3:
        if primary_layer >= lang_acts_all.shape[1]:
            logger.error(
                f"Requested layer {primary_layer}, but {lang} activations only have "
                f"{lang_acts_all.shape[1]} layers in shape {tuple(lang_acts_all.shape)}"
            )
            sys.exit(1)
        lang_layer = lang_acts_all[:, primary_layer, :]
    else:
        lang_layer = lang_acts_all
    lang_layer = lang_layer.to(torch.float32)

    logger.info(f"Loading SAE for layer {primary_layer}...")
    try:
        sae = load_sae(model_name, primary_layer, hook_component="mlp")
    except Exception as e:
        logger.error(f"Could not load SAE: {e}")
        sys.exit(1)

    logger.info("Encoding activations through SAE...")
    en_features = encode_activations(sae, en_layer)
    lang_features = encode_activations(sae, lang_layer)

    logger.info("Computing delta scores...")
    delta = compute_delta_scores(en_features, lang_features)
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

    logger.info(f"Top {args.top_k} features identified. Table saved.")
    logger.info(table_df.head(10).to_string(index=False))

    with open(output_dir / "ranked_features.json", "w") as f:
        json.dump({
            "ranked_features": [int(x) for x in ranked_idx],
            "layer": int(primary_layer),
            "comparison_language": str(lang),
            "critical_layers_sorted": [int(x) for x in critical_layers],
            "published_sae_layers": [int(x) for x in AVAILABLE_SAE_LAYERS],
            "sae_repo": "EleutherAI/sae-llama-3.1-8b-64x",
            "hookpoint": f"layers.{int(primary_layer)}.mlp",
            "activation_component": str(activation_component),
        }, f, indent=2)

    en_feat_means = en_features.float().mean(dim=0).cpu().numpy()
    np.save(str(output_dir / "en_feature_means.npy"), en_feat_means)

    logger.info(f"SAE feature analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()