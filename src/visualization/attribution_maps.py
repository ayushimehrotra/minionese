"""
Visualization: Attribution Patching Maps
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
SINGLE_COL = (3.25, 2.5)
FONT_SIZE = 10

COMPONENT_COLORS = {
    "residual": "#0072B2",
    "attn_out": "#D55E00",
    "mlp_out": "#009E73",
}

TIER_COLORS = {
    "tier1": "#332288",
    "tier2": "#117733",
    "tier3": "#CC6677",
    "tier4": "#DDCC77",
}


def _apply_style():
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
    })


def _save_figure(fig, output_path: str) -> None:
    base = Path(output_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {base}.pdf / {base}.png")


def plot_attribution_map(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Attribution Patching: Refusal Restoration",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Layer (x) vs restoration score (y), separate lines per component.
    One subplot per language tier.

    Args:
        data: DataFrame with columns: layer, component, tier, mean_restoration.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot attribution map.")
        return

    tiers = sorted(data["tier"].unique()) if "tier" in data.columns else ["all"]
    n_tiers = len(tiers)

    fig, axes = plt.subplots(
        1, n_tiers,
        figsize=(figsize[0] * n_tiers / 2, figsize[1]),
        squeeze=False,
    )

    for t_idx, tier in enumerate(tiers):
        ax = axes[0, t_idx]

        if "tier" in data.columns:
            tier_data = data[data["tier"] == tier]
        else:
            tier_data = data

        components = tier_data["component"].unique() if "component" in tier_data.columns else ["residual"]

        for comp in sorted(components):
            comp_data = tier_data[tier_data["component"] == comp].sort_values("layer")
            color = COMPONENT_COLORS.get(comp, "black")

            ax.plot(
                comp_data["layer"],
                comp_data["mean_restoration"],
                label=comp,
                color=color,
                linewidth=1.2,
                marker=".",
                markersize=3,
            )

            # Shade std if available
            if "std_restoration" in comp_data.columns:
                ax.fill_between(
                    comp_data["layer"],
                    comp_data["mean_restoration"] - comp_data["std_restoration"],
                    comp_data["mean_restoration"] + comp_data["std_restoration"],
                    color=color,
                    alpha=0.15,
                )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(y=1, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Layer", fontsize=FONT_SIZE)
        ax.set_ylabel("Restoration Score", fontsize=FONT_SIZE)
        ax.set_title(f"Tier: {tier}", fontsize=FONT_SIZE)
        ax.legend(fontsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=FONT_SIZE + 1)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_head_level_map(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Head-Level Restoration Scores",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Heatmap of (layer, head) with restoration score.

    Args:
        data: DataFrame with columns: layer, head_idx, restoration_score.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot head-level map.")
        return

    pivot = data.pivot_table(
        index="head_idx", columns="layer", values="restoration_score", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu",
        center=0,
        annot=False,
        cbar_kws={"label": "Restoration Score"},
    )
    ax.set_xlabel("Layer", fontsize=FONT_SIZE)
    ax.set_ylabel("Head Index", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    plt.tight_layout()
    _save_figure(fig, output_path)
