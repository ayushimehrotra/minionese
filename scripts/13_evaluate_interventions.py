#!/usr/bin/env python3
"""
Script 13: Evaluate Interventions

Computes ASR, over-refusal, MMLU, LangID for all intervention results.

Usage:
    python scripts/13_evaluate_interventions.py \
        --interventions-dir results/interventions/ \
        --output-dir results/intervention_eval/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interventions.sweep import compute_pareto_frontier
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.visualization.pareto import plot_pareto_frontier
from src.visualization.tables import generate_intervention_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate interventions.")
    parser.add_argument("--interventions-dir", default="results/interventions/")
    parser.add_argument("--output-dir", default="results/intervention_eval/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("evaluate_interventions")
    setup_reproducibility(seed=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = Path(args.interventions_dir) / "sweep_results.csv"
    if not sweep_path.exists():
        logger.error(f"Sweep results not found: {sweep_path}. Run script 12 first.")
        sys.exit(1)

    results_df = pd.read_csv(sweep_path)
    logger.info(f"Loaded {len(results_df)} sweep results.")

    if args.dry_run:
        logger.info("[DRY RUN] Would evaluate intervention results.")
        return

    # Pareto frontier
    logger.info("Computing Pareto frontier...")
    pareto_df = compute_pareto_frontier(results_df)
    pareto_df.to_csv(output_dir / "pareto_frontier.csv", index=False)
    logger.info(f"Pareto-optimal points: {len(pareto_df)}")

    # Merge pareto flag into results
    results_with_pareto = results_df.copy()
    results_with_pareto["is_pareto_optimal"] = False
    pareto_indices = pareto_df.index
    results_with_pareto.loc[pareto_indices, "is_pareto_optimal"] = True

    # Plot Pareto frontier
    logger.info("Generating Pareto plot...")
    plot_pareto_frontier(results_with_pareto, str(output_dir / "fig_pareto_frontier"))

    # LaTeX table
    logger.info("Generating LaTeX intervention table...")
    latex_table = generate_intervention_table(
        results_df.sort_values(["intervention", "param_value"]),
        caption="Intervention Comparison: Safety vs. Utility Tradeoff",
        label="tab:interventions",
    )
    with open(output_dir / "intervention_table.tex", "w") as f:
        f.write(latex_table)

    # Summary statistics
    summary = (
        results_df.groupby("intervention")
        .agg(
            asr_mean=("asr", "mean"),
            asr_min=("asr", "min"),
            over_refusal_mean=("over_refusal", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "intervention_summary.csv", index=False)

    logger.info(f"Intervention evaluation complete. Results in {output_dir}.")
    logger.info("\nSummary:")
    logger.info(summary.to_string(index=False))


if __name__ == "__main__":
    main()
