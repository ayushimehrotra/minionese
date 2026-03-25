#!/usr/bin/env python3
"""
Script 04: Extract Activations

Extract and cache model activations for all languages x perturbations.
This is the most compute-intensive step.

Usage:
    python scripts/04_extract_activations.py \
        --model llama \
        --dataset-dir dataset/ \
        --output-dir data/activations/ \
        --positions last_instruction last_post_instruction last \
        --components residual attn_out mlp_out
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.cache import activation_exists
from src.activations.extract import extract_activations, _model_short_name
from src.dataset.loader import load_dataset, format_for_model
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Extract model activations.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--output-dir", default="data/activations/")
    parser.add_argument("--positions", nargs="+",
                        default=["last_instruction", "last_post_instruction", "last"])
    parser.add_argument("--components", nargs="+",
                        default=["residual", "attn_out", "mlp_out"])
    parser.add_argument("--layers", default="all",
                        help="Comma-separated layer indices or 'all'.")
    parser.add_argument("--perturbations", nargs="+", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip if cached file already exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("extract_activations")
    setup_reproducibility(seed=42)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]
    model_short = _model_short_name(model_name)

    # Parse layers
    if args.layers == "all":
        layers = "all"
    else:
        layers = [int(l) for l in args.layers.split(",")]

    logger.info(f"Loading dataset from {args.dataset_dir}...")
    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=args.perturbations,
        languages=args.languages,
    )

    if df.empty:
        logger.error("No data loaded.")
        sys.exit(1)

    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    df = format_for_model(df, model_name, tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process one (perturbation, language) at a time
    groups = df.groupby(["perturbation", "language"])
    total = len(groups)

    for idx, ((pert, lang), grp) in enumerate(groups):
        logger.info(f"[{idx + 1}/{total}] Processing: {pert}/{lang} ({len(grp)} prompts)")

        # Check if all cached files already exist
        if args.skip_existing:
            all_exist = all(
                activation_exists(model_short, lang, pert, pos, comp, str(output_dir))
                for pos in args.positions
                for comp in args.components
            )
            if all_exist:
                logger.info(f"  Skipping {pert}/{lang}: all activations already cached.")
                continue

        if args.dry_run:
            logger.info(f"  [DRY RUN] Would extract activations for {pert}/{lang}.")
            continue

        prompts = grp["formatted_prompt"].tolist()
        metadata = [{"language": lang, "perturbation": pert}] * len(prompts)

        try:
            extract_activations(
                model_name=model_name,
                prompts=prompts,
                token_positions=args.positions,
                layers=layers,
                output_dir=str(output_dir),
                batch_size=args.batch_size,
                dtype=args.dtype,
                components=args.components,
                prompt_metadata=metadata,
            )
        except Exception:
            logger.exception(f"Extraction failed for {pert}/{lang}")
            continue

    logger.info("Activation extraction pipeline complete.")


if __name__ == "__main__":
    main()
