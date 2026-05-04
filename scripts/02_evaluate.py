#!/usr/bin/env python3
"""
Script 02: Evaluate Safety + Coherence

Run WildGuard on all generated responses. Compute ASR, coherence,
and regime classification. Produces Figures 1 and 2.

Usage:
    python scripts/02_evaluate.py \
        --generations-dir results/generations/ \
        --output-dir results/evaluation/ \
        --use-vllm
"""

import argparse
import json
import logging
import os
import sys
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import login
from src.evaluation.asr import compute_asr, asr_by_tier, asr_delta_from_english
from src.evaluation.coherence import compute_coherence_table
from src.evaluation.generation import load_responses
from src.evaluation.safety_judge import score_wildguard
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate safety and coherence of generated responses.")
    parser.add_argument("--generations-dir", default="results/generations/")
    parser.add_argument("--output-dir", default="results/evaluation/")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for faster WildGuard inference.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token.")
    parser.add_argument("--no-hf-login", action="store_true")
    return parser.parse_args()


def ensure_hf_token(args, logger):
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        if sys.stdin.isatty():
            token = getpass("Enter your Hugging Face token (hf_...): ").strip()
        else:
            raise RuntimeError(
                "No Hugging Face token found. Pass --hf-token or set HF_TOKEN."
            )
    if not token or not token.startswith("hf_"):
        raise RuntimeError("Invalid Hugging Face token format.")
    os.environ["HF_TOKEN"] = token
    if not args.no_hf_login:
        try:
            login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
            logger.info("Logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning(f"HF login() failed, continuing with in-process token: {e}")
    return token


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("evaluate")
    setup_reproducibility(seed=42)

    gen_dir = Path(args.generations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_files = sorted(gen_dir.glob("*.jsonl"))
    if not gen_files:
        logger.error(f"No .jsonl files found in {gen_dir}.")
        sys.exit(1)

    all_responses = []
    for fpath in gen_files:
        try:
            responses = load_responses(str(fpath))
            stem_parts = fpath.stem.split("_", 2)
            for r in responses:
                if "model" not in r:
                    r["model"] = stem_parts[0] if stem_parts else "unknown"
                if "language" not in r and len(stem_parts) >= 2:
                    r["language"] = stem_parts[1]
                if "perturbation" not in r and len(stem_parts) >= 3:
                    r["perturbation"] = stem_parts[2]
                if "is_harmful" not in r and "prompt_id" in r:
                    r["is_harmful"] = str(r.get("prompt_id", "")).startswith("harmful_")
            all_responses.extend(responses)
            logger.info(f"Loaded {len(responses)} responses from {fpath.name}.")
        except Exception as e:
            logger.error(f"Error loading {fpath}: {e}")

    logger.info(f"Total responses: {len(all_responses)}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would evaluate {len(all_responses)} responses.")
        return

    hf_token = ensure_hf_token(args, logger)

    # WildGuard scoring
    logger.info("Running WildGuard...")
    cache_path = str(output_dir / "wildguard_cache.jsonl")
    scored = score_wildguard(
        all_responses,
        batch_size=args.batch_size,
        cache_path=cache_path,
        hf_token=hf_token,
        use_vllm=args.use_vllm,
    )
    all_scored_path = output_dir / "all_scored.jsonl"
    write_jsonl(scored, all_scored_path)
    logger.info(f"Raw WildGuard labels saved to {all_scored_path}.")

    # Compute ASR
    logger.info("Computing ASR...")
    asr_df = compute_asr(scored)
    asr_df.to_csv(output_dir / "asr_detailed.csv", index=False)

    tier_asr = asr_by_tier(asr_df)
    tier_asr.to_csv(output_dir / "asr_by_tier.csv", index=False)

    delta_asr = asr_delta_from_english(asr_df)
    delta_asr.to_csv(output_dir / "asr_delta_from_english.csv", index=False)

    # Coherence and regime classification
    logger.info("Computing coherence and regime classification...")
    config = load_config()
    coherence_cfg = config.get("experiment", {}).get("coherence", {})
    coherence_df = compute_coherence_table(
        scored,
        langid_threshold=coherence_cfg.get("langid_confidence_threshold", 0.3),
        unicode_threshold=coherence_cfg.get("unicode_validity_threshold", 0.8),
        min_length=coherence_cfg.get("min_response_length", 20),
        min_coherence_for_asr=coherence_cfg.get("min_coherence_for_asr", 0.5),
    )
    coherence_df.to_csv(output_dir / "coherence_table.csv", index=False)
    logger.info("Coherence table saved.")

    if "model" in coherence_df.columns and "regime" in coherence_df.columns:
        regime_summary = coherence_df.groupby(["model", "regime"]).size().reset_index(name="count")
        logger.info(f"Regime summary:\n{regime_summary.to_string(index=False)}")

    # Produce figures
    try:
        from src.visualization.heatmaps import (
            plot_coherence_heatmap,
            plot_regime_comparison,
            plot_asr_heatmap,
        )
        figures_dir = Path("figures/")
        figures_dir.mkdir(parents=True, exist_ok=True)

        if not coherence_df.empty:
            plot_coherence_heatmap(coherence_df, str(figures_dir / "fig1_coherence_heatmap"))
            plot_regime_comparison(coherence_df, str(figures_dir / "fig1b_regime_comparison"))
            logger.info("Figure 1 (coherence heatmap + regime comparison) saved.")

        if not asr_df.empty:
            plot_asr_heatmap(
                asr_df,
                str(figures_dir / "fig2_asr_heatmap"),
                coherence_data=coherence_df,
            )
            logger.info("Figure 2 (ASR heatmap) saved.")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    logger.info(f"Evaluation complete. Results saved to {output_dir}.")


if __name__ == "__main__":
    main()
