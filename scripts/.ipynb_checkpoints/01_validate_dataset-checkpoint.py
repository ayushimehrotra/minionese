#!/usr/bin/env python3
"""
Script 01: Validate Dataset

Validates the pre-built dataset structure and contrastive pair integrity.
Does NOT regenerate data.

Usage:
    python scripts/01_validate_dataset.py --dataset-dir dataset/ --config configs/experiment.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import (
    TIERED_PERTURBATIONS, FLAT_PERTURBATIONS,
    discover_languages, load_dataset, validate_contrastive_pairs,
)
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Validate pre-built dataset.")
    parser.add_argument("--dataset-dir", default="dataset/", help="Root dataset directory.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Experiment config.")
    parser.add_argument("--output", default="results/dataset_validation.json", help="Validation report output path.")
    parser.add_argument("--dry-run", action="store_true", help="Check structure only; skip pair validation.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("validate_dataset")
    setup_reproducibility(seed=42)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    report = {
        "dataset_dir": str(dataset_dir),
        "perturbation_checks": {},
        "missing_files": [],
        "row_count_summary": {},
        "validation_flags": [],
    }

    # --- Check directory structure ---
    logger.info("Checking directory structure...")
    discovered = discover_languages(str(dataset_dir))
    logger.info(f"Discovered languages: {discovered}")
    report["discovered_languages"] = discovered

    # Expected structure from languages config
    from src.utils.config import load_yaml
    try:
        lang_cfg = load_yaml("configs/languages.yaml")
        tiers = lang_cfg.get("tiers", {})
    except Exception:
        tiers = {}

    missing_files = []
    row_counts = {}

    for pert in TIERED_PERTURBATIONS:
        pert_dir = dataset_dir / pert
        if not pert_dir.exists():
            logger.warning(f"Missing perturbation directory: {pert_dir}")
            missing_files.append(str(pert_dir))
            continue

        for tier_name, tier_data in tiers.items():
            for lang in tier_data.get("languages", []):
                for label in ["harmful", "harmless"]:
                    csv_path = pert_dir / tier_name / lang / f"{label}.csv"
                    if not csv_path.exists():
                        logger.warning(f"Missing: {csv_path}")
                        missing_files.append(str(csv_path))
                    else:
                        import pandas as pd
                        try:
                            df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
                            key = f"{pert}/{tier_name}/{lang}/{label}"
                            row_counts[key] = len(df)
                        except Exception as e:
                            logger.error(f"Error reading {csv_path}: {e}")
                            missing_files.append(f"UNREADABLE: {csv_path}")

    # Minionese: flat structure
    for label in ["harmful", "harmless"]:
        csv_path = dataset_dir / "minionese" / f"{label}.csv"
        if not csv_path.exists():
            logger.warning(f"Missing minionese: {csv_path}")
            missing_files.append(str(csv_path))
        else:
            import pandas as pd
            try:
                df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
                row_counts[f"minionese/{label}"] = len(df)
            except Exception as e:
                logger.error(f"Error reading {csv_path}: {e}")

    report["missing_files"] = missing_files
    report["row_count_summary"] = row_counts

    logger.info(f"Structure check: {len(missing_files)} missing files/directories.")

    if not args.dry_run:
        # --- Validate contrastive pairs ---
        logger.info("Loading dataset for contrastive pair validation...")
        try:
            df = load_dataset(str(dataset_dir))
            if not df.empty:
                flags_df = validate_contrastive_pairs(df)
                report["validation_flags"] = flags_df.to_dict(orient="records")
                logger.info(f"Contrastive pair validation: {len(flags_df)} issues found.")
            else:
                logger.warning("Dataset is empty; skipping pair validation.")
        except Exception as e:
            logger.error(f"Error during dataset loading: {e}")
            report["validation_error"] = str(e)

    # --- Save report ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Validation report saved to {output_path}.")

    if missing_files:
        logger.warning(f"ATTENTION: {len(missing_files)} files/dirs missing. Check the report.")
        sys.exit(0 if args.dry_run else 0)  # Exit 0 even with warnings; report captures details


if __name__ == "__main__":
    main()
