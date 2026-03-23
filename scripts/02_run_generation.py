#!/usr/bin/env python3
"""
Script 02: Run Generation

Generate model responses for all languages x perturbations.
"""

import argparse
import logging
import os
import sys
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset, format_for_model
from src.evaluation.generation import generate_responses
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Run model generation.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"],
                        help="Model key (llama, gemma, qwen).")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--output-dir", default="results/generations/")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--perturbations", nargs="+", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token. If omitted, uses HF_TOKEN env var or prompts securely.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load dataset and validate without running inference.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_hf_token(cli_token: str | None) -> str | None:
    # Priority: CLI arg -> environment variable -> interactive prompt
    token = cli_token or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    print("\nThis model may require Hugging Face authentication.")
    print("Paste your HF token below. Input will be hidden.")
    token = getpass("HF token: ").strip()

    if not token:
        return None
    return token


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("run_generation")
    setup_reproducibility(seed=42)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]

    logger.info(f"Model: {model_name}")
    logger.info(f"Loading dataset from {args.dataset_dir}...")

    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=args.perturbations,
        languages=args.languages,
    )

    if df.empty:
        logger.error("No data loaded. Check dataset directory and filters.")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} rows.")

    if args.dry_run:
        logger.info("[DRY RUN] Would generate responses for:")
        for (pert, lang), grp in df.groupby(["perturbation", "language"]):
            logger.info(f"  {pert}/{lang}: {len(grp)} prompts")
        logger.info("[DRY RUN] Exiting.")
        return

    hf_token = resolve_hf_token(args.hf_token)

    # Format prompts with chat template
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    logger.info("Applying chat template...")
    df = format_for_model(df, model_name, tokenizer)

    # Generate per (perturbation, language) group
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (pert, lang), grp in df.groupby(["perturbation", "language"]):
        output_file = output_dir / f"{args.model}_{lang}_{pert}.jsonl"

        prompts = grp["formatted_prompt"].tolist()
        prompt_ids = [
            f"{'harmful' if row['is_harmful'] else 'harmless'}_{i:04d}_{lang}_{pert}"
            for i, row in enumerate(grp.to_dict("records"))
        ]

        logger.info(f"Generating: {pert}/{lang} ({len(prompts)} prompts) -> {output_file}")

        results = generate_responses(
            model_name=model_name,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            batch_size=args.batch_size,
            output_path=str(output_file),
            prompt_ids=prompt_ids,
            hf_token=hf_token,   # <-- ADD THIS LINE
        )

        logger.info(f"Done: {pert}/{lang} -> {len(results)} responses saved.")

    logger.info("Generation complete.")


if __name__ == "__main__":
    main()