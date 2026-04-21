"""
Visualization: Safety/Utility Pareto Plots
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DOUBLE_COL = (6.75, 3.5)
FONT_SIZE = 10
TIER_LABELS = ["tier1", "tier2", "tier3", "tier4"]

INTERVENTION_COLORS = {
    "baseline": "gray",
    "caa": "#0072B2",
    "sae_clamp": "#D55E00",
    "subspace_projection": "#009E73",
}

INTERVENTION_MARKERS = {
    "baseline": "s",
    "caa": "o",
    "sae_clamp": "^",
    "subspace_projection": "D",
}


def plot_pareto_frontier(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Safety vs. Utility Pareto Frontier",
    figsize: Optional[tuple] = None,
) -> None:
    """
    Safety vs. Utility scatter plot with Pareto front highlighted.
    One subplot per language tier, all interventions overlaid.

    Args:
        data: DataFrame with columns: intervention, tier, asr, over_refusal,
              mmlu_accuracy, and optionally is_pareto_optimal.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size. Defaults to double-column width.
    """
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")

    if data.empty:
        logger.warning("No data to plot Pareto frontier.")
        return

    tiers = sorted(data.get("tier", pd.Series(["all"])).unique())
    n_tiers = max(len(tiers), 1)

    if figsize is None:
        figsize = (DOUBLE_COL[0] * min(n_tiers, 2), DOUBLE_COL[1] * ((n_tiers + 1) // 2))

    fig, axes = plt.subplots(
        (n_tiers + 1) // 2, min(n_tiers, 2),
        figsize=figsize,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax_idx, tier in enumerate(tiers):
        ax = axes_flat[ax_idx]

        if "tier" in data.columns:
            tier_data = data[data["tier"] == tier]
        else:
            tier_data = data

        interventions = tier_data["intervention"].unique()

        for interv in sorted(interventions):
            idata = tier_data[tier_data["intervention"] == interv]

            # Safety = 1 - ASR, Utility = 1 - over_refusal
            safety = 1.0 - idata["asr"].fillna(1.0)
            utility = 1.0 - idata["over_refusal"].fillna(0.0)

            color = INTERVENTION_COLORS.get(interv, "black")
            marker = INTERVENTION_MARKERS.get(interv, "o")

            ax.scatter(
                utility,
                safety,
                label=interv,
                color=color,
                marker=marker,
                s=40,
                alpha=0.8,
                zorder=3,
            )

            # Connect points in sweep order
            if len(idata) > 1:
                param_vals = idata["param_value"].fillna(0).values
                sort_idx = np.argsort(param_vals)
                ax.plot(
                    utility.values[sort_idx],
                    safety.values[sort_idx],
                    color=color,
                    alpha=0.4,
                    linewidth=0.8,
                    zorder=2,
                )

            # Highlight Pareto-optimal points
            if "is_pareto_optimal" in idata.columns:
                pareto = idata[idata["is_pareto_optimal"]]
                ax.scatter(
                    1.0 - pareto["over_refusal"].fillna(0.0),
                    1.0 - pareto["asr"].fillna(1.0),
                    color=color,
                    marker=marker,
                    s=80,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=4,
                )

        ax.set_xlabel("Utility (1 - over-refusal)", fontsize=FONT_SIZE)
        ax.set_ylabel("Safety (1 - ASR)", fontsize=FONT_SIZE)
        ax.set_title(f"Tier: {tier}", fontsize=FONT_SIZE)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=FONT_SIZE - 2, loc="lower left")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax_idx in range(n_tiers, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=FONT_SIZE + 1)
    plt.tight_layout()

    base = Path(output_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Pareto plot saved: {base}.pdf / {base}.png")
