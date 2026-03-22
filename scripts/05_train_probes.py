#!/usr/bin/env python3
"""
Script 05: Train Probes

Train linear probes per (language, layer, harm_category).

Usage:
    python scripts/05_train_probes.py \
        --activations-dir data/activations/ \
        --output-dir results/probes/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.probing.linear_probe import train_all_probes
from src.utils.config import load_config, load_yaml
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Train linear probes.")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--output-dir", default="results/probes/")
    parser.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices. Default: all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("train_probes")
    setup_reproducibility(seed=42)

    config = load_config()

    # Determine languages
    if args.languages:
        languages = args.languages
    else:
        lang_cfg = load_yaml("configs/languages.yaml")
        languages = []
        for tier_data in lang_cfg.get("tiers", {}).values():
            languages.extend(tier_data.get("languages", []))

    # Determine layers
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        model_cfg = config["models"]["target_models"].get(args.model, {})
        num_layers = model_cfg.get("num_layers", 32)
        layers = list(range(num_layers))

    logger.info(f"Languages: {languages}")
    logger.info(f"Layers: {layers[:5]}...{layers[-3:]} ({len(layers)} total)")

    # Load dataset for metadata
    logger.info("Loading dataset...")
    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=[args.perturbation],
        languages=languages,
    )

    if df.empty:
        logger.error("No dataset loaded.")
        sys.exit(1)

    harm_categories = sorted(df["category"].dropna().unique().tolist())
    logger.info(f"Harm categories: {harm_categories}")

    if args.dry_run:
        logger.info(
            f"[DRY RUN] Would train {len(languages)} x {len(layers)} x "
            f"{len(harm_categories) + 1} probes."
        )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training probes...")
    summary_df = train_all_probes(
        activations_dir=args.activations_dir,
        dataset=df,
        languages=languages,
        layers=layers,
        harm_categories=harm_categories,
        output_dir=str(output_dir),
        model_short=args.model,
        perturbation=args.perturbation,
        token_position=args.token_position,
    )

    logger.info(f"Trained {len(summary_df)} probes. Summary saved to {output_dir}/probe_summary.csv")


if __name__ == "__main__":
    main()
