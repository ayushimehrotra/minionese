#!/usr/bin/env python3
"""
Script 11: SAE Feature Analysis

SAE decomposition, delta scoring, interpretation, causal validation.
For Qwen, checks for trained SAEs and triggers training if not found.

Usage:
    python scripts/11_sae_feature_analysis.py \
        --model llama \
        --critical-layers results/attribution/critical_layers.json \
        --activations-dir data/activations/ \
        --output-dir results/sae_features/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sae.delta_scores import compute_delta_scores, rank_features, feature_analysis_table
from src.sae.feature_extract import load_sae, encode_activations
from src.sae.interpret import load_feature_labels
from src.sae.train_sae import check_sae_availability, train_all_critical_layers
from src.utils.config import load_config, load_yaml, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="SAE feature analysis.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--critical-layers", required=True,
                        help="Path to critical_layers.json.")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--output-dir", default="results/sae_features/")
    parser.add_argument("--trained-saes-dir", default="trained_saes/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--comparison-language", default="ar",
                        help="Primary LangX for delta score computation.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("sae_feature_analysis")
    setup_reproducibility(seed=42)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    with open(args.critical_layers) as f:
        critical_data = json.load(f)
    critical_layers = critical_data.get("critical_layers", [])
    logger.info(f"Critical layers: {critical_layers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check SAE availability; train if needed (Qwen)
    avail = check_sae_availability(model_name)
    if not avail["available"]:
        logger.info(f"No pre-trained SAEs for {model_name}. Checking for custom-trained SAEs...")
        from src.sae.train_sae import _model_short
        model_short = _model_short(model_name)
        missing_layers = [
            l for l in critical_layers
            if not (Path(args.trained_saes_dir) / model_short / f"layer_{l}" / "sae").exists()
        ]
        if missing_layers and not args.dry_run:
            logger.info(f"Training SAEs for layers: {missing_layers}")
            train_all_critical_layers(model_name, missing_layers, args.trained_saes_dir)

    if args.dry_run:
        logger.info("[DRY RUN] Would run SAE feature analysis.")
        return

    from src.activations.cache import load_activations, get_activation_path

    # Primary layer for analysis
    primary_layer = critical_layers[len(critical_layers) // 2] if critical_layers else 15

    # Load EN harmful activations
    en_act_path = get_activation_path(
        args.model, "en", args.perturbation, args.token_position, "residual",
        args.activations_dir
    )
    if not Path(en_act_path).exists():
        logger.error(f"English activations not found: {en_act_path}")
        sys.exit(1)

    en_acts_all = load_activations(en_act_path)
    if en_acts_all.ndim == 3:
        en_layer = en_acts_all[:, primary_layer, :]
    else:
        en_layer = en_acts_all
    en_layer = en_layer.to(torch.float32)

    # Load LangX harmful activations
    lang = args.comparison_language
    lang_act_path = get_activation_path(
        args.model, lang, args.perturbation, args.token_position, "residual",
        args.activations_dir
    )
    if not Path(lang_act_path).exists():
        logger.error(f"{lang} activations not found: {lang_act_path}")
        sys.exit(1)

    lang_acts_all = load_activations(lang_act_path)
    if lang_acts_all.ndim == 3:
        lang_layer = lang_acts_all[:, primary_layer, :]
    else:
        lang_layer = lang_acts_all
    lang_layer = lang_layer.to(torch.float32)

    # Load SAE
    logger.info(f"Loading SAE for layer {primary_layer}...")
    try:
        sae = load_sae(model_name, primary_layer, args.trained_saes_dir)
    except Exception as e:
        logger.error(f"Could not load SAE: {e}")
        sys.exit(1)

    # Encode activations
    logger.info("Encoding activations through SAE...")
    en_features = encode_activations(sae, en_layer)
    lang_features = encode_activations(sae, lang_layer)

    # Compute delta scores
    logger.info("Computing delta scores...")
    delta = compute_delta_scores(en_features, lang_features)
    np.save(str(output_dir / f"delta_scores_{lang}_layer{primary_layer}.npy"), delta)

    # Rank and create analysis table
    ranked_idx = rank_features(delta, args.top_k)

    # Look up feature labels
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

    # Save ranked feature indices for interventions
    with open(output_dir / "ranked_features.json", "w") as f:
        json.dump({
            "ranked_features": ranked_idx,
            "layer": primary_layer,
            "comparison_language": lang,
        }, f, indent=2)

    # Save EN mean activation values for clamping
    en_feat_means = en_features.float().mean(dim=0).cpu().numpy()
    np.save(str(output_dir / "en_feature_means.npy"), en_feat_means)

    logger.info(f"SAE feature analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
