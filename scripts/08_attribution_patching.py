#!/usr/bin/env python3
"""
Script 08: Attribution Patching

Runs two complementary analyses (per the paper design):

  Analysis A  (normalize=False):
    For every (EN, LangX) harmful pair, compute the raw shift in the
    refusal-direction projection when EN activations at t_inst are patched
    into layer L of the corrupted (LangX) run.  No division by
    baseline_diff, so near-zero baseline is never a problem.
    Used for critical layer identification.

  Analysis B  (normalize=True):
    Filters to behaviorally-contrastive pairs (EN refused, LangX complied)
    and computes the normalized restoration score.  Reports how many pairs
    had sufficient contrast per language.

Both analyses are run per perturbation type so that standard_translation,
transliteration, and code_switching can be compared.

Usage:
    python scripts/08_attribution_patching.py \\
        --model llama \\
        --dataset-dir dataset/ \\
        --refusal-dir results/disentangle/ \\
        --output-dir results/attribution/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.attribution_patch import (
    run_attribution_patching,
    aggregate_by_tier,
    identify_critical_layers,
)
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
    parser.add_argument("--perturbations", nargs="+",
                        default=["standard_translation", "transliteration", "code_switching"],
                        help="Perturbation types to run. Each is analysed separately.")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Languages to include. Default: all non-English.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Max prompt pairs per (language, perturbation).")
    parser.add_argument("--min-baseline-diff", type=float, default=0.01,
                        help="Minimum |baseline_diff| for Analysis B pairs.")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token (or set HF_TOKEN env var).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_hf_token(cli_token):
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("attribution_patching")
    setup_reproducibility(seed=42)

    hf_token = resolve_hf_token(args.hf_token)

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

    # Build lang -> tier map
    lang_to_tier = {}
    for tier_name, tier_data in lang_cfg.get("tiers", {}).items():
        for lang in tier_data.get("languages", []):
            lang_to_tier[lang] = tier_name

    languages_to_run = args.languages or [l for l in lang_to_tier if l != "en"]

    all_a_results = []
    all_b_results = []

    for perturbation in args.perturbations:
        logger.info(f"=== Perturbation: {perturbation} ===")

        df = load_dataset(
            dataset_dir=args.dataset_dir,
            perturbations=[perturbation],
        )
        if df.empty:
            logger.warning(f"No data for perturbation={perturbation}. Skipping.")
            continue

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token
        )
        df = format_for_model(df, model_name, tokenizer)

        # English harmful prompts (clean signal)
        en_harmful = df[
            (df["language"] == "en") & df["is_harmful"]
        ]["formatted_prompt"].tolist()[:args.n_prompts]

        if not en_harmful:
            logger.warning(f"No English harmful prompts for {perturbation}. Skipping.")
            continue

        if args.dry_run:
            logger.info(
                f"[DRY RUN] Would patch {len(en_harmful)} pairs × "
                f"{len(languages_to_run)} languages for {perturbation}."
            )
            continue

        for lang in languages_to_run:
            tier = lang_to_tier.get(lang, "unknown")

            langx_harmful = df[
                (df["language"] == lang) & df["is_harmful"]
            ]["formatted_prompt"].tolist()[:args.n_prompts]

            if not langx_harmful:
                logger.warning(f"  No {lang} harmful prompts. Skipping.")
                continue

            n = min(len(en_harmful), len(langx_harmful))
            en_batch = en_harmful[:n]
            lang_batch = langx_harmful[:n]

            logger.info(
                f"  Analysis A (unnormalized): en vs {lang} "
                f"({perturbation}, {n} pairs)..."
            )
            try:
                df_a = run_attribution_patching(
                    model_name=model_name,
                    en_prompts=en_batch,
                    langx_prompts=lang_batch,
                    refusal_direction=refusal_direction,
                    components=["residual", "attn_out", "mlp_out"],
                    batch_size=args.batch_size,
                    language=lang,
                    tier=tier,
                    perturbation=perturbation,
                    normalize=False,
                    hf_token=hf_token,
                )
                all_a_results.append(df_a)
            except Exception as e:
                logger.error(f"  Analysis A failed for {lang}/{perturbation}: {e}")

            logger.info(
                f"  Analysis B (normalized, filtered): en vs {lang} "
                f"({perturbation})..."
            )
            try:
                df_b = run_attribution_patching(
                    model_name=model_name,
                    en_prompts=en_batch,
                    langx_prompts=lang_batch,
                    refusal_direction=refusal_direction,
                    components=["residual", "attn_out", "mlp_out"],
                    batch_size=args.batch_size,
                    language=lang,
                    tier=tier,
                    perturbation=perturbation,
                    normalize=True,
                    min_baseline_diff=args.min_baseline_diff,
                    hf_token=hf_token,
                )
                all_b_results.append(df_b)
                if not df_b.empty:
                    n_used = df_b["n_pairs_used"].iloc[0]
                    n_skip = df_b["n_pairs_skipped"].iloc[0]
                    logger.info(
                        f"    Analysis B: {n_used} pairs used, "
                        f"{n_skip} skipped (no behavioral contrast)"
                    )
            except Exception as e:
                logger.error(f"  Analysis B failed for {lang}/{perturbation}: {e}")

    if not all_a_results and not all_b_results:
        logger.error("No attribution patching results produced.")
        sys.exit(1)

    # ── Save Analysis A ──
    if all_a_results:
        df_a_all = pd.concat(all_a_results, ignore_index=True)
        df_a_all.to_csv(output_dir / "attribution_a_results.csv", index=False)
        tier_a = aggregate_by_tier(df_a_all)
        tier_a.to_csv(output_dir / "attribution_a_by_tier.csv", index=False)

        # Critical layers from Analysis A residual scores (averaged across languages)
        critical = identify_critical_layers(df_a_all, top_k=5, component="residual")
        with open(output_dir / "critical_layers.json", "w") as f:
            json.dump({"critical_layers": critical}, f, indent=2)
        logger.info(f"Critical layers (Analysis A, residual): {critical}")

        try:
            plot_attribution_map(tier_a, str(output_dir / "fig_attribution_a"))
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

    # ── Save Analysis B ──
    if all_b_results:
        df_b_all = pd.concat(all_b_results, ignore_index=True)
        df_b_all.to_csv(output_dir / "attribution_b_results.csv", index=False)
        tier_b = aggregate_by_tier(df_b_all)
        tier_b.to_csv(output_dir / "attribution_b_by_tier.csv", index=False)

        # Pair coverage summary: how many pairs had behavioral contrast per language
        if "n_pairs_used" in df_b_all.columns:
            coverage = (
                df_b_all.groupby(["language", "perturbation"])["n_pairs_used"]
                .first()
                .reset_index()
            )
            coverage.to_csv(output_dir / "analysis_b_pair_coverage.csv", index=False)
            logger.info("Analysis B pair coverage:\n" + coverage.to_string(index=False))

        try:
            plot_attribution_map(tier_b, str(output_dir / "fig_attribution_b"))
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

    logger.info(f"Attribution patching complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
