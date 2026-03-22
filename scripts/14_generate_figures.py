#!/usr/bin/env python3
"""
Script 14: Generate Figures

Generates all paper figures and LaTeX tables.

Usage:
    python scripts/14_generate_figures.py \
        --results-dir results/ \
        --output-dir figures/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.attribution_maps import plot_attribution_map, plot_head_level_map
from src.visualization.heatmaps import plot_silhouette_heatmap, plot_asr_heatmap, plot_effective_rank
from src.visualization.pareto import plot_pareto_frontier
from src.visualization.tables import generate_asr_table, generate_intervention_table, generate_feature_table
from src.utils.logging_setup import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-dir", default="figures/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_csv_if_exists(path: str, name: str, logger) -> pd.DataFrame:
    """Load a CSV if it exists, otherwise return empty DataFrame."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(str(p))
    else:
        logger.warning(f"{name} not found: {path}")
        return pd.DataFrame()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("generate_figures")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Would generate figures.")
        return

    # --- Figure 1: ASR heatmap ---
    asr_df = _load_csv_if_exists(
        results_dir / "safety_scores" / "asr_detailed.csv", "ASR data", logger
    )
    if not asr_df.empty:
        plot_asr_heatmap(asr_df, str(output_dir / "fig1_asr_heatmap"))
        logger.info("Fig 1: ASR heatmap generated.")

    # --- Figure 2: Silhouette heatmap ---
    sil_df = _load_csv_if_exists(
        results_dir / "cross_lingual" / "silhouette_scores.csv", "Silhouette data", logger
    )
    if not sil_df.empty:
        plot_silhouette_heatmap(sil_df, str(output_dir / "fig2_silhouette_heatmap"))
        logger.info("Fig 2: Silhouette heatmap generated.")

    # --- Figure 3: Effective rank vs layer ---
    rank_df = _load_csv_if_exists(
        results_dir / "cross_lingual" / "effective_rank.csv", "Effective rank data", logger
    )
    if not rank_df.empty:
        plot_effective_rank(rank_df, str(output_dir / "fig3_effective_rank"))
        logger.info("Fig 3: Effective rank plot generated.")

    # --- Figure 4: Principal angles ---
    pa_df = _load_csv_if_exists(
        results_dir / "cross_lingual" / "principal_angles.csv", "Principal angles data", logger
    )
    if not pa_df.empty:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6.75, 3.5))
            for lang in pa_df["language"].unique():
                lang_data = pa_df[(pa_df["language"] == lang) & (pa_df["angle_idx"] == 0)]
                ax.plot(lang_data["layer"], lang_data["angle_deg"], label=lang, marker=".")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Principal Angle (degrees)")
            ax.set_title("EN vs. Other Languages: First Principal Angle")
            ax.legend(fontsize=8, ncol=3)
            plt.tight_layout()
            plt.savefig(str(output_dir / "fig4_principal_angles.pdf"), dpi=300, bbox_inches="tight")
            plt.savefig(str(output_dir / "fig4_principal_angles.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Fig 4: Principal angles plot generated.")
        except Exception as e:
            logger.warning(f"Could not generate principal angles plot: {e}")

    # --- Figure 5: Harmfulness vs refusal signal ---
    disen_df = _load_csv_if_exists(
        results_dir / "disentangle" / "disentangle_results.csv", "Disentangle data", logger
    )
    if not disen_df.empty:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0))
            for ax, col, label in [
                (axes[0], "harm_component_norm", "Harm Component Norm"),
                (axes[1], "t_post_inst_refusal_signal", "Refusal Signal Strength"),
            ]:
                for lang in disen_df["language"].unique():
                    ld = disen_df[disen_df["language"] == lang].sort_values("layer")
                    ax.plot(ld["layer"], ld[col], label=lang, alpha=0.7)
                ax.set_xlabel("Layer")
                ax.set_ylabel(label)
                ax.legend(fontsize=7, ncol=2)
            plt.suptitle("Harmfulness vs. Refusal Signal by Language")
            plt.tight_layout()
            plt.savefig(str(output_dir / "fig5_harm_refusal_signal.pdf"), dpi=300, bbox_inches="tight")
            plt.savefig(str(output_dir / "fig5_harm_refusal_signal.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Fig 5: Harm/refusal signal plot generated.")
        except Exception as e:
            logger.warning(f"Could not generate harm/refusal plot: {e}")

    # --- Figure 6: Attribution patching map ---
    attr_df = _load_csv_if_exists(
        results_dir / "attribution" / "attribution_by_tier.csv", "Attribution data", logger
    )
    if not attr_df.empty:
        plot_attribution_map(attr_df, str(output_dir / "fig6_attribution_map"))
        logger.info("Fig 6: Attribution map generated.")

    # --- Figure 7: Head-level map ---
    head_df = _load_csv_if_exists(
        results_dir / "head_tracing" / "head_tracing_results.csv", "Head tracing data", logger
    )
    if not head_df.empty:
        plot_head_level_map(head_df, str(output_dir / "fig7_head_level_map"))
        logger.info("Fig 7: Head-level map generated.")

    # --- Figure 8: English-pivot ---
    pivot_df = _load_csv_if_exists(
        results_dir / "english_pivot" / "pivot_correlation.csv", "Pivot data", logger
    )
    if not pivot_df.empty:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.25, 2.5))
            ax.plot(pivot_df["layer"], pivot_df["lang_probe_accuracy"], label="Language probe")
            ax.plot(pivot_df["layer"], pivot_df["harm_probe_accuracy"], label="Harm probe (EN)")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=8)
            ax.set_title("English-Pivot Hypothesis")
            plt.tight_layout()
            plt.savefig(str(output_dir / "fig8_english_pivot.pdf"), dpi=300, bbox_inches="tight")
            plt.savefig(str(output_dir / "fig8_english_pivot.png"), dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Fig 8: English-pivot plot generated.")
        except Exception as e:
            logger.warning(f"Could not generate English-pivot plot: {e}")

    # --- Figure 9: SAE features ---
    sae_df = _load_csv_if_exists(
        results_dir / "sae_features" / "top_features_ar_layer15.csv", "SAE feature data", logger
    )
    if sae_df.empty:
        # Try other naming conventions
        sae_files = list((results_dir / "sae_features").glob("top_features_*.csv"))
        if sae_files:
            sae_df = pd.read_csv(str(sae_files[0]))

    if not sae_df.empty:
        logger.info("Fig 9: SAE features table would be generated (see LaTeX tables below).")

    # --- Figure 10: Pareto frontier ---
    interv_eval_df = _load_csv_if_exists(
        results_dir / "intervention_eval" / "pareto_frontier.csv", "Pareto data", logger
    )
    if not interv_eval_df.empty:
        plot_pareto_frontier(interv_eval_df, str(output_dir / "fig10_pareto_frontier"))
        logger.info("Fig 10: Pareto frontier generated.")

    # --- LaTeX Tables ---
    logger.info("Generating LaTeX tables...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    if not asr_df.empty:
        latex = generate_asr_table(asr_df)
        (tables_dir / "tab_asr.tex").write_text(latex)

    interv_df = _load_csv_if_exists(
        results_dir / "interventions" / "sweep_results.csv", "Intervention results", logger
    )
    if not interv_df.empty:
        latex = generate_intervention_table(interv_df)
        (tables_dir / "tab_interventions.tex").write_text(latex)

    if not sae_df.empty:
        latex = generate_feature_table(sae_df)
        (tables_dir / "tab_sae_features.tex").write_text(latex)

    logger.info(f"All figures and tables generated. Output: {output_dir}")


if __name__ == "__main__":
    main()
