"""
Visualization: Layer x Language Heatmaps
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# NeurIPS-compatible style settings
SINGLE_COL = (3.25, 2.5)
DOUBLE_COL = (6.75, 3.5)
FONT_SIZE = 10

# Language tier order for consistent x-axis labeling
TIER_ORDER = {
    "tier1": ["en", "de", "fr", "zh", "es"],
    "tier2": ["ar", "ru", "ko", "ja"],
    "tier3": ["tr", "id", "hi", "sw"],
    "tier4": ["yo", "zu", "gd", "gn", "jv"],
}


def _apply_style():
    """Apply consistent plot style."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-paper")
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
    })


def plot_silhouette_heatmap(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Harmful/Harmless Silhouette Score",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Layer x Language heatmap colored by silhouette score.

    Args:
        data: DataFrame with columns: layer, language, silhouette_score.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot silhouette heatmap.")
        return

    pivot = data.pivot(index="layer", columns="language", values="silhouette_score")

    # Order languages by tier
    ordered_langs = []
    for tier_langs in TIER_ORDER.values():
        for lang in tier_langs:
            if lang in pivot.columns:
                ordered_langs.append(lang)
    remaining = [c for c in pivot.columns if c not in ordered_langs]
    ordered_langs += remaining
    pivot = pivot[ordered_langs]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        annot=False,
        cbar_kws={"label": "Silhouette Score"},
    )
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Language", fontsize=FONT_SIZE)
    ax.set_ylabel("Layer", fontsize=FONT_SIZE)

    # Annotate tier boundaries
    _annotate_tier_boundaries(ax, ordered_langs)

    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_asr_heatmap(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Attack Success Rate",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Language x Perturbation heatmap colored by ASR.

    Args:
        data: DataFrame with columns: language, perturbation, asr_wildguard.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot ASR heatmap.")
        return

    if "asr_wildguard" not in data.columns:
        logger.warning("'asr_wildguard' column missing from data.")
        return

    pivot = data.pivot_table(
        index="language", columns="perturbation", values="asr_wildguard", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": FONT_SIZE - 2},
        cbar_kws={"label": "ASR"},
    )
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Perturbation Type", fontsize=FONT_SIZE)
    ax.set_ylabel("Language", fontsize=FONT_SIZE)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_effective_rank(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Effective Rank vs Layer",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Line plot of effective rank per layer, one line per language.

    Args:
        data: DataFrame with columns: layer, language, effective_rank.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot effective rank.")
        return

    palette = sns.color_palette("colorblind", n_colors=data["language"].nunique())

    fig, ax = plt.subplots(figsize=figsize)
    for i, lang in enumerate(sorted(data["language"].unique())):
        lang_data = data[data["language"] == lang].sort_values("layer")
        ax.plot(
            lang_data["layer"],
            lang_data["effective_rank"],
            label=lang,
            color=palette[i],
            marker=".",
            markersize=3,
            linewidth=1.0,
        )

    ax.set_xlabel("Layer", fontsize=FONT_SIZE)
    ax.set_ylabel("Effective Rank (95% energy)", fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE - 2, ncol=3, loc="upper right")
    plt.tight_layout()
    _save_figure(fig, output_path)


def _annotate_tier_boundaries(ax, ordered_langs: list) -> None:
    """Add vertical lines separating tier boundaries."""
    tier_positions = []
    pos = 0
    for tier_langs in TIER_ORDER.values():
        count = sum(1 for l in tier_langs if l in ordered_langs)
        if count > 0:
            tier_positions.append(pos + count)
        pos += count

    for boundary in tier_positions[:-1]:
        ax.axvline(x=boundary, color="black", linewidth=1.0, linestyle="--", alpha=0.5)


def _save_figure(fig, output_path: str) -> None:
    """Save figure as both PDF and PNG."""
    base = Path(output_path).with_suffix("")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figures: {base}.pdf, {base}.png")
