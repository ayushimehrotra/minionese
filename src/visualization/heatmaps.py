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

    # Clip negatives — negative silhouette means misclassified overlap, shown as 0
    pivot = pivot.clip(lower=0.0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
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
    coherence_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Language x Perturbation heatmap colored by ASR.

    Args:
        data: DataFrame with columns: language, perturbation, asr_wildguard.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
        coherence_data: Optional coherence table; collapse cells are grayed out.
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot ASR heatmap.")
        return

    if "asr_wildguard" not in data.columns:
        logger.warning("'asr_wildguard' column missing from data.")
        return

    # Drop minionese — it is a special case plotted separately
    data = data[data["language"] != "minionese"].copy()

    pivot = data.pivot_table(
        index="language", columns="perturbation", values="asr_wildguard", aggfunc="mean"
    )

    # Order languages by tier, dropping minionese
    ordered_langs = []
    for tier_langs in TIER_ORDER.values():
        for lang in tier_langs:
            if lang in pivot.index:
                ordered_langs.append(lang)
    remaining = [l for l in pivot.index if l not in ordered_langs]
    ordered_langs += remaining
    pivot = pivot.reindex(ordered_langs)

    # Scale to percentage for readability; vmax = data max (min 10%) for contrast
    pivot_pct = pivot * 100
    vmax = max(pivot_pct.max().max(), 10.0)

    # Annotate all cells with their ASR value — coherence is already baked into the
    # per-generation ASR numerator, so no cells need to be masked as N/A
    annot = pivot_pct.copy().map(lambda v: f"{v:.1f}%" if not np.isnan(v) else "")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_pct,
        ax=ax,
        cmap="Reds",
        vmin=0.0,
        vmax=vmax,
        annot=annot,
        fmt="",
        annot_kws={"size": FONT_SIZE - 2},
        cbar_kws={"label": "ASR (%)"},
    )

    # Tier boundary horizontal lines
    _annotate_tier_boundaries_h(ax, ordered_langs)

    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("Perturbation Type", fontsize=FONT_SIZE)
    ax.set_ylabel("Language", fontsize=FONT_SIZE)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_coherence_heatmap(
    coherence_df: pd.DataFrame,
    output_path: str,
    title: str = "Generation Coherence Rate",
    figsize: tuple = (10, 4),
) -> None:
    """
    One subplot per model: language (y) x perturbation (x), colored by coherence_rate.
    Tier boundaries are annotated with horizontal lines.

    Args:
        coherence_df: DataFrame with columns: model, language, perturbation, coherence_rate.
        output_path: Output file path (without extension).
        title: Figure title.
        figsize: Overall figure size.
    """
    _apply_style()

    if coherence_df.empty:
        logger.warning("No data to plot coherence heatmap.")
        return

    coherence_df = coherence_df[coherence_df["language"] != "minionese"].copy()

    models = sorted(coherence_df["model"].unique()) if "model" in coherence_df.columns else ["all"]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, squeeze=False)

    # Build ordered language list from tier definitions (no minionese)
    ordered_langs = []
    for tier_langs in TIER_ORDER.values():
        ordered_langs.extend(tier_langs)

    for col_idx, model in enumerate(models):
        ax = axes[0][col_idx]
        subset = coherence_df[coherence_df["model"] == model] if "model" in coherence_df.columns else coherence_df
        if subset.empty:
            ax.set_visible(False)
            continue

        pivot = subset.pivot_table(
            index="language", columns="perturbation", values="coherence_rate", aggfunc="mean"
        )

        # Reorder rows by tier
        present = [l for l in ordered_langs if l in pivot.index]
        rest = [l for l in pivot.index if l not in present]
        pivot = pivot.reindex(present + rest)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            annot=False,
            cbar=col_idx == n_models - 1,
            cbar_kws={"label": "Coherence Rate"} if col_idx == n_models - 1 else {},
        )
        ax.set_title(model, fontsize=FONT_SIZE)
        ax.set_xlabel("Perturbation" if col_idx == n_models // 2 else "", fontsize=FONT_SIZE)
        ax.set_ylabel("Language" if col_idx == 0 else "", fontsize=FONT_SIZE)

        # Annotate tier boundaries (horizontal lines)
        lang_order = list(pivot.index)
        pos = 0
        for tier_langs in TIER_ORDER.values():
            count = sum(1 for l in tier_langs if l in lang_order)
            pos += count
            if 0 < pos < len(lang_order):
                ax.axhline(y=pos, color="white", linewidth=1.5, linestyle="--")

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=FONT_SIZE - 2)
        plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE - 2)

    fig.suptitle(title, fontsize=FONT_SIZE + 1)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_regime_comparison(
    coherence_df: pd.DataFrame,
    output_path: str,
    title: str = "Regime Distribution by Model and Tier",
    figsize: tuple = DOUBLE_COL,
) -> None:
    """
    Grouped stacked bar chart: fraction of languages in refuse/comply/collapse per (model, tier).

    Args:
        coherence_df: DataFrame with columns: model, tier, regime.
        output_path: Output file path (without extension).
        title: Plot title.
        figsize: Figure size.
    """
    _apply_style()

    if coherence_df.empty:
        logger.warning("No data to plot regime comparison.")
        return

    required = {"regime"}
    if not required.issubset(coherence_df.columns):
        logger.warning(f"Missing columns for regime comparison: {required - set(coherence_df.columns)}")
        return

    group_cols = [c for c in ["model", "tier"] if c in coherence_df.columns]
    if not group_cols:
        logger.warning("Need at least one of 'model', 'tier' columns.")
        return

    counts = coherence_df.groupby(group_cols + ["regime"]).size().reset_index(name="count")
    totals = counts.groupby(group_cols)["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals

    regimes = ["refuse", "comply", "collapse"]
    colors = {"refuse": "#2ecc71", "comply": "#e74c3c", "collapse": "#95a5a6"}

    pivot = counts.pivot_table(
        index=group_cols, columns="regime", values="fraction", aggfunc="sum"
    ).fillna(0.0)
    for r in regimes:
        if r not in pivot.columns:
            pivot[r] = 0.0
    pivot = pivot[regimes]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for regime in regimes:
        vals = pivot[regime].values
        ax.bar(x, vals, bottom=bottom, label=regime, color=colors[regime], width=0.6)
        bottom += vals

    labels = [" / ".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,))) for idx in pivot.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=FONT_SIZE - 2)
    ax.set_ylabel("Fraction of Languages", fontsize=FONT_SIZE)
    ax.set_ylim(0, 1)
    ax.legend(title="Regime", fontsize=FONT_SIZE - 2, loc="upper right")
    ax.set_title(title, fontsize=FONT_SIZE)
    plt.tight_layout()
    _save_figure(fig, output_path)


def plot_principal_angles_by_tier(
    data: pd.DataFrame,
    output_path: str,
    title: str = "Principal Angles vs English Harm Subspace",
    figsize: tuple = (10, 7),
) -> None:
    """
    4-panel figure (one per tier) showing principal angle vs layer per language.

    Args:
        data: DataFrame with columns: language, layer, angle_deg.
        output_path: Output file path (without extension).
    """
    _apply_style()

    if data.empty:
        logger.warning("No data to plot principal angles.")
        return

    # Average across angle indices if multiple
    data = data.groupby(["language", "layer"])["angle_deg"].mean().reset_index()

    tier_labels = {
        "tier1": "Tier 1 — High-resource",
        "tier2": "Tier 2 — Mid-resource",
        "tier3": "Tier 3 — Low-resource",
        "tier4": "Tier 4 — Very-low-resource",
    }
    tiers = list(TIER_ORDER.keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax_idx, tier_key in enumerate(tiers):
        ax = axes[ax_idx]
        tier_langs = [l for l in TIER_ORDER[tier_key] if l in data["language"].values]

        palette = sns.color_palette("colorblind", n_colors=max(len(tier_langs), 1))
        for i, lang in enumerate(tier_langs):
            lang_data = data[data["language"] == lang].sort_values("layer")
            ax.plot(
                lang_data["layer"],
                lang_data["angle_deg"],
                label=lang,
                color=palette[i],
                marker="o",
                markersize=3,
                linewidth=1.2,
            )

        ax.axhline(y=90, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title(tier_labels[tier_key], fontsize=FONT_SIZE)
        ax.set_ylim(0, 95)
        ax.set_ylabel("Angle (°)", fontsize=FONT_SIZE)
        ax.set_xlabel("Layer", fontsize=FONT_SIZE)
        if tier_langs:
            ax.legend(fontsize=FONT_SIZE - 2, loc="upper left", ncol=2)

    fig.suptitle(title, fontsize=FONT_SIZE + 1)
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


def _annotate_tier_boundaries_h(ax, ordered_langs: list) -> None:
    """Add horizontal lines separating tier boundaries (for language-indexed rows)."""
    pos = 0
    for tier_langs in list(TIER_ORDER.values())[:-1]:
        count = sum(1 for l in tier_langs if l in ordered_langs)
        pos += count
        if 0 < pos < len(ordered_langs):
            ax.axhline(y=pos, color="black", linewidth=1.0, linestyle="--", alpha=0.5)


def _save_figure(fig, output_path: str) -> None:
    """Save figure as both PDF and PNG."""
    base = Path(output_path).with_suffix("")
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figures: {base}.pdf, {base}.png")
