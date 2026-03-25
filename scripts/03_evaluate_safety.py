#!/usr/bin/env python3
"""
Script 03: Evaluate Safety

Run WildGuard and LlamaGuard on all generated responses. Compute ASR.

Usage:
    python scripts/03_evaluate_safety.py \
        --generations-dir results/generations/ \
        --output-dir results/safety_scores/
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
from src.evaluation.generation import load_responses
from src.evaluation.safety_judge import score_wildguard, score_llamaguard, compute_agreement
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate safety of generated responses.")
    parser.add_argument("--generations-dir", default="results/generations/")
    parser.add_argument("--output-dir", default="results/safety_scores/")
    parser.add_argument("--skip-llamaguard", action="store_true",
                        help="Skip LlamaGuard (use WildGuard only).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    # NEW
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN env var or prompts interactively."
    )
    parser.add_argument(
        "--no-hf-login",
        action="store_true",
        help="Do not persist token via huggingface_hub.login(); only use it for this process."
    )
    return parser.parse_args()


def ensure_hf_token(args, logger):
    """
    Get an HF token from:
      1) --hf-token
      2) HF_TOKEN environment variable
      3) interactive prompt
    Then expose it to downstream code and optionally persist it.
    """
    token = args.hf_token or os.environ.get("HF_TOKEN")

    if not token:
        if sys.stdin.isatty():
            token = getpass("Enter your Hugging Face token (hf_...): ").strip()
        else:
            raise RuntimeError(
                "No Hugging Face token found. "
                "Pass --hf-token, set HF_TOKEN, or run interactively."
            )

    if not token or not token.startswith("hf_"):
        raise RuntimeError("Invalid Hugging Face token format. Expected something starting with 'hf_'.")

    # Make it visible to any library that reads HF_TOKEN.
    os.environ["HF_TOKEN"] = token

    # Optional: cache the token for future runs.
    if not args.no_hf_login:
        try:
            login(token=token, add_to_git_credential=False, skip_if_logged_in=False)
            logger.info("Logged into Hugging Face Hub for this machine.")
        except Exception as e:
            # Not fatal if direct token passing still works
            logger.warning(f"HF login() failed, but continuing with in-process token: {e}")

    return token


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("evaluate_safety")
    setup_reproducibility(seed=42)

    gen_dir = Path(args.generations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all generation files
    gen_files = sorted(gen_dir.glob("*.jsonl"))
    if not gen_files:
        logger.error(f"No .jsonl files found in {gen_dir}.")
        sys.exit(1)

    all_responses = []
    for fpath in gen_files:
        try:
            responses = load_responses(str(fpath))
            # Infer metadata from filename: {model}_{lang}_{pert}.jsonl
            stem_parts = fpath.stem.split("_", 2)
            for r in responses:
                if "language" not in r and len(stem_parts) >= 2:
                    r["language"] = stem_parts[1]
                if "perturbation" not in r and len(stem_parts) >= 3:
                    r["perturbation"] = stem_parts[2]
            all_responses.extend(responses)
            logger.info(f"Loaded {len(responses)} responses from {fpath.name}.")
        except Exception as e:
            logger.error(f"Error loading {fpath}: {e}")

    logger.info(f"Total responses: {len(all_responses)}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would evaluate {len(all_responses)} responses.")
        return

    # NEW: only ask once, before the first gated model load
    hf_token = ensure_hf_token(args, logger)

    # WildGuard scoring
    logger.info("Running WildGuard...")
    cache_path = str(output_dir / "wildguard_cache.jsonl")
    scored = score_wildguard(
        all_responses,
        batch_size=args.batch_size,
        cache_path=cache_path,
        hf_token=hf_token,   # NEW
    )

    # LlamaGuard scoring (optional)
    if not args.skip_llamaguard:
        logger.info("Running LlamaGuard...")
        lg_cache = str(output_dir / "llamaguard_cache.jsonl")
        scored = score_llamaguard(
            scored,
            batch_size=args.batch_size,
            cache_path=lg_cache,
            hf_token=hf_token,   # NEW
        )

        agreement = compute_agreement(scored)
        with open(output_dir / "judge_agreement.json", "w") as f:
            json.dump(agreement, f, indent=2, default=str)
        logger.info(f"Judge agreement (kappa): {agreement.get('kappa')}")

    # Compute ASR
    logger.info("Computing ASR...")
    asr_df = compute_asr(scored)
    asr_df.to_csv(output_dir / "asr_detailed.csv", index=False)

    tier_asr = asr_by_tier(asr_df)
    tier_asr.to_csv(output_dir / "asr_by_tier.csv", index=False)

    delta_asr = asr_delta_from_english(asr_df)
    delta_asr.to_csv(output_dir / "asr_delta_from_english.csv", index=False)

    # Save all scored results
    with open(output_dir / "all_scored.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Safety evaluation complete. Results saved to {output_dir}.")


if __name__ == "__main__":
    main()