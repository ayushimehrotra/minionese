#!/usr/bin/env python3
"""
Script 09: Attention Head Tracing

Head-level causal tracing at critical layers.

Usage:
    python scripts/09_attention_head_tracing.py \
        --model llama \
        --critical-layers results/attribution/critical_layers.json \
        --output-dir results/head_tracing/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.attention_heads import trace_attention_heads, extract_attention_patterns
from src.dataset.loader import load_dataset, format_for_model
from src.utils.config import load_config, load_yaml, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.visualization.attribution_maps import plot_head_level_map


def parse_args():
    parser = argparse.ArgumentParser(description="Attention head-level causal tracing.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--critical-layers", required=True,
                        help="Path to critical_layers.json from script 08.")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--refusal-dir", default="results/disentangle/")
    parser.add_argument("--output-dir", default="results/head_tracing/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("attention_head_tracing")
    setup_reproducibility(seed=42)

    config = load_config()
    lang_cfg = load_yaml("configs/languages.yaml")
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    # Load critical layers
    with open(args.critical_layers) as f:
        critical_data = json.load(f)
    critical_layers = critical_data.get("critical_layers", [])
    logger.info(f"Critical layers: {critical_layers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load refusal direction
    refusal_path = Path(args.refusal_dir) / f"refusal_direction_{args.model}.npy"
    refusal_direction = np.load(str(refusal_path)) if refusal_path.exists() else None

    if refusal_direction is None:
        logger.error("Refusal direction not found. Run script 07 first.")
        sys.exit(1)

    # Load dataset
    df = load_dataset(dataset_dir=args.dataset_dir, perturbations=[args.perturbation])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    df = format_for_model(df, model_name, tokenizer)

    en_harmful = df[(df["language"] == "en") & df["is_harmful"]]["formatted_prompt"].tolist()
    en_harmful = en_harmful[:args.n_prompts]

    if args.dry_run:
        logger.info(f"[DRY RUN] Would trace {len(critical_layers)} layers x {model_cfg['num_heads']} heads.")
        return

    lang_to_tier = {}
    for tier_name, td in lang_cfg.get("tiers", {}).items():
        for lang in td.get("languages", []):
            lang_to_tier[lang] = tier_name

    all_results = []

    for lang in list(lang_to_tier.keys())[:5]:  # Limit for speed; extend as needed
        if lang == "en":
            continue

        langx = df[(df["language"] == lang) & df["is_harmful"]]["formatted_prompt"].tolist()
        langx = langx[:args.n_prompts]
        if not langx:
            continue

        n = min(len(en_harmful), len(langx))
        logger.info(f"Tracing heads: en vs {lang} ({n} pairs)...")

        try:
            trace_df = trace_attention_heads(
                model_name=model_name,
                en_prompts=en_harmful[:n],
                langx_prompts=langx[:n],
                critical_layers=critical_layers,
                refusal_direction=refusal_direction,
                num_heads=model_cfg.get("num_heads"),
                batch_size=args.batch_size,
                language=lang,
            )
            all_results.append(trace_df)
        except Exception as e:
            logger.error(f"Head tracing failed for {lang}: {e}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(output_dir / "head_tracing_results.csv", index=False)

        # Identify critical heads (top by restoration score)
        top_heads = (
            combined_df.groupby(["layer", "head_idx"])["restoration_score"]
            .mean()
            .nlargest(20)
            .reset_index()
        )
        top_heads.to_csv(output_dir / "critical_heads.csv", index=False)

        # Plot
        plot_head_level_map(combined_df, str(output_dir / "fig_head_level_map"))
        logger.info(f"Head tracing complete. Results in {output_dir}.")
    else:
        logger.warning("No head tracing results produced.")


if __name__ == "__main__":
    main()
