#!/usr/bin/env python3
"""
Script 08: Attribution Patching

Layer-level and component-level patching to identify critical layers.

Usage:
    python scripts/08_attribution_patching.py \
        --model llama \
        --dataset-dir dataset/ \
        --refusal-dir results/disentangle/ \
        --output-dir results/attribution/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.attribution_patch import run_attribution_patching, aggregate_by_tier
from src.dataset.loader import load_dataset, format_for_model
from src.utils.config import load_config, load_yaml, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.visualization.attribution_maps import plot_attribution_map


def parse_args():
    parser = argparse.ArgumentParser(description="Run attribution patching.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--refusal-dir", default="results/disentangle/")
    parser.add_argument("--output-dir", default="results/attribution/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Number of prompt pairs to use for patching (keep small for speed).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("attribution_patching")
    setup_reproducibility(seed=42)

    config = load_config()
    lang_cfg = load_yaml("configs/languages.yaml")
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load refusal direction
    refusal_path = Path(args.refusal_dir) / f"refusal_direction_{args.model}.npy"
    if not refusal_path.exists():
        logger.error(f"Refusal direction not found: {refusal_path}. Run script 07 first.")
        sys.exit(1)
    refusal_direction = np.load(str(refusal_path))
    logger.info(f"Loaded refusal direction: shape={refusal_direction.shape}")

    # Load dataset
    logger.info("Loading dataset...")
    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=[args.perturbation],
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    df = format_for_model(df, model_name, tokenizer)

    # Get English harmful prompts
    en_harmful = df[(df["language"] == "en") & df["is_harmful"]]["formatted_prompt"].tolist()
    en_harmful = en_harmful[:args.n_prompts]

    if not en_harmful:
        logger.error("No English harmful prompts found.")
        sys.exit(1)

    if args.dry_run:
        logger.info(f"[DRY RUN] Would patch {len(en_harmful)} prompt pairs per language.")
        return

    all_results = []

    # Build language -> tier mapping
    lang_to_tier = {}
    for tier_name, tier_data in lang_cfg.get("tiers", {}).items():
        for lang in tier_data.get("languages", []):
            lang_to_tier[lang] = tier_name

    languages = [l for l in lang_to_tier.keys() if l != "en"]

    for lang in languages:
        tier = lang_to_tier.get(lang, "unknown")
        langx_harmful = df[
            (df["language"] == lang) & df["is_harmful"]
        ]["formatted_prompt"].tolist()
        langx_harmful = langx_harmful[:args.n_prompts]

        if not langx_harmful:
            logger.warning(f"No {lang} harmful prompts found. Skipping.")
            continue

        # Align lengths
        n = min(len(en_harmful), len(langx_harmful))
        en_batch = en_harmful[:n]
        lang_batch = langx_harmful[:n]

        logger.info(f"Attribution patching: en vs {lang} ({n} pairs)...")
        try:
            results_df = run_attribution_patching(
                model_name=model_name,
                en_prompts=en_batch,
                langx_prompts=lang_batch,
                refusal_direction=refusal_direction,
                components=["residual", "attn_out", "mlp_out"],
                batch_size=args.batch_size,
                language=lang,
                tier=tier,
            )
            all_results.append(results_df)
        except Exception as e:
            logger.error(f"Attribution patching failed for {lang}: {e}")

    if not all_results:
        logger.error("No attribution patching results.")
        sys.exit(1)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / "attribution_results.csv", index=False)

    # Aggregate by tier
    tier_df = aggregate_by_tier(combined_df)
    tier_df.to_csv(output_dir / "attribution_by_tier.csv", index=False)

    # Identify critical layers (top 5 by mean restoration score, residual only)
    residual_df = combined_df[combined_df["component"] == "residual"]
    if not residual_df.empty:
        critical_layers = (
            residual_df.groupby("layer")["mean_restoration"]
            .mean()
            .nlargest(5)
            .index.tolist()
        )
        with open(output_dir / "critical_layers.json", "w") as f:
            json.dump({"critical_layers": critical_layers}, f, indent=2)
        logger.info(f"Critical layers identified: {critical_layers}")

    # Plot
    plot_attribution_map(tier_df, str(output_dir / "fig_attribution_map"))

    logger.info(f"Attribution patching complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
