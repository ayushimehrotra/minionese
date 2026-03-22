#!/usr/bin/env python3
"""
Script 10: English-Pivot Test

Tests whether safety routes through internal English representations.

Usage:
    python scripts/10_english_pivot_test.py \
        --activations-dir data/activations/ \
        --output-dir results/english_pivot/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.english_pivot import (
    train_language_probe,
    english_pivot_correlation,
)
from src.utils.config import load_config, load_yaml
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="English-pivot hypothesis test.")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--probes-dir", default="results/probes/")
    parser.add_argument("--output-dir", default="results/english_pivot/")
    parser.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("english_pivot_test")
    setup_reproducibility(seed=42)

    config = load_config()
    lang_cfg = load_yaml("configs/languages.yaml")

    languages = []
    for td in lang_cfg.get("tiers", {}).values():
        languages.extend(td.get("languages", []))

    model_cfg = config["models"]["target_models"].get(args.model, {})
    num_layers = model_cfg.get("num_layers", 32)
    layers = list(range(num_layers))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Would run English-pivot analysis.")
        return

    from src.activations.cache import load_activations, get_activation_path

    # Load activations per language
    activations_by_lang = {}
    for lang in languages:
        act_path = get_activation_path(
            args.model, lang, args.perturbation, args.token_position, "residual",
            args.activations_dir
        )
        if Path(act_path).exists():
            try:
                acts = load_activations(act_path)  # (n, n_layers, hidden) or (n, hidden)
                activations_by_lang[lang] = acts.numpy()
                logger.info(f"Loaded {lang}: shape={acts.shape}")
            except Exception as e:
                logger.warning(f"Could not load activations for {lang}: {e}")

    if not activations_by_lang:
        logger.error("No activations loaded.")
        sys.exit(1)

    # Train language probe
    logger.info("Training language identity probes...")
    lang_probe_results = train_language_probe(activations_by_lang, layers)

    # Save per-layer accuracy
    per_layer_acc = {
        layer: result["accuracy"]
        for layer, result in lang_probe_results["per_layer"].items()
    }
    acc_df = pd.DataFrame(
        [{"layer": l, "language_probe_accuracy": a} for l, a in per_layer_acc.items()]
    )
    acc_df.to_csv(output_dir / "language_probe_accuracy.csv", index=False)

    # Load harm probe results (per-layer)
    harm_probe_results = {}
    probe_summary_path = Path(args.probes_dir) / "probe_summary.csv"
    if probe_summary_path.exists():
        probe_summary = pd.read_csv(probe_summary_path)
        en_probes = probe_summary[
            (probe_summary["language"] == "en") & (probe_summary["category"] == "all")
        ]
        for _, row in en_probes.iterrows():
            harm_probe_results[int(row["layer"])] = {"cv_accuracy": row["cv_accuracy"]}

    # Compute correlation
    corr_df = english_pivot_correlation(lang_probe_results, harm_probe_results, layers)
    corr_df.to_csv(output_dir / "pivot_correlation.csv", index=False)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.75, 3.5))
        ax.plot(corr_df["layer"], corr_df["lang_probe_accuracy"], label="Language probe acc")
        ax.plot(corr_df["layer"], corr_df["harm_probe_accuracy"], label="Harm probe acc (EN)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title("English-Pivot: Language vs. Harm Probe Accuracy")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(output_dir / "fig_english_pivot.pdf"), dpi=300, bbox_inches="tight")
        plt.savefig(str(output_dir / "fig_english_pivot.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save pivot plot: {e}")

    logger.info(f"English-pivot analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
