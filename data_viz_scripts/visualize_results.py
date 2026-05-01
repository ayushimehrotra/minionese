#!/usr/bin/env python3
"""
visualize_results.py
====================
Produce publication-quality figures from the evaluation CSVs output by
02_evaluate.py.  Works with any number of models – just concatenate the
per-model CSVs before running, or point --results-dir at a folder that
contains the four files already merged across models.

Usage
-----
    # Quick start – supply the four CSV files directly
    python visualize_results.py \
        --asr-detailed      results/evaluation/asr_detailed.csv \
        --asr-by-tier       results/evaluation/asr_by_tier.csv \
        --asr-delta         results/evaluation/asr_delta_from_english.csv \
        --coherence         results/evaluation/coherence_table.csv \
        --output-dir        figures/

    # Or point at a directory that contains all four files
    python visualize_results.py --results-dir results/evaluation/ --output-dir figures/

Figures produced
----------------
  fig1_asr_heatmap_<model>.png        Per-model language × perturbation ASR heatmap
  fig2_asr_by_tier.png                Grouped bar: ASR by tier × perturbation, one bar per model
  fig3_asr_delta_heatmap_<model>.png  Per-model delta-from-English heatmap
  fig4_regime_distribution.png        Stacked bar: comply/refuse ratio per model × tier
  fig5_model_comparison.png           Radar / grouped bar comparing models overall
  fig6_transliteration_spike.png      Transliteration ASR vs other perturbations across models
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── aesthetics ────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10")
PERTURBATION_ORDER = ["standard_translation", "translationese", "code_switching", "transliteration"]
PERTURBATION_LABELS = {
    "standard_translation": "Std. Translation",
    "translationese":       "Translationese",
    "code_switching":       "Code Switching",
    "transliteration":      "Transliteration",
}
TIER_ORDER = ["tier1", "tier2", "tier3", "tier4"]
TIER_LABELS = {
    "tier1": "Tier 1\n(High-resource)",
    "tier2": "Tier 2\n(Mid-resource)",
    "tier3": "Tier 3\n(Low-resource)",
    "tier4": "Tier 4\n(Very Low-resource)",
}

LANG_FULL = {
    "ar": "Arabic",  "de": "German",  "en": "English",
    "es": "Spanish", "fr": "French",  "gd": "Scottish Gaelic",
    "gn": "Guaraní", "hi": "Hindi",   "id": "Indonesian",
    "ja": "Japanese","jw": "Javanese","ko": "Korean",
    "ru": "Russian", "sw": "Swahili", "tr": "Turkish",
    "yo": "Yoruba",  "zh": "Chinese", "zu": "Zulu",
    "minionese": "Minionese",
}

ASR_CMAP    = "YlOrRd"
DELTA_CMAP  = "RdBu_r"


def short_model(name: str) -> str:
    """Strip org prefix for display, e.g. 'CohereForAI/aya-expanse-8b' → 'aya-expanse-8b'."""
    return name.split("/")[-1]


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] {label} file not found: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {len(df)} rows, models: {df['model'].unique().tolist()}")
    return df


def load_all(args) -> dict[str, pd.DataFrame]:
    if args.results_dir:
        d = Path(args.results_dir)
        paths = dict(
            detailed  = d / "asr_detailed.csv",
            by_tier   = d / "asr_by_tier.csv",
            delta     = d / "asr_delta_from_english.csv",
            coherence = d / "coherence_table.csv",
        )
    else:
        paths = dict(
            detailed  = Path(args.asr_detailed),
            by_tier   = Path(args.asr_by_tier),
            delta     = Path(args.asr_delta),
            coherence = Path(args.coherence),
        )
    return {k: load_csv(v, k) for k, v in paths.items()}


# ── figure helpers ─────────────────────────────────────────────────────────────

def savefig(fig, out_dir: Path, name: str, dpi: int = 150):
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Per-model ASR heatmap  (language × perturbation)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_asr_heatmap(df: pd.DataFrame, out_dir: Path):
    """One heatmap per model showing ASR for every language × perturbation cell."""
    models = df["model"].unique()
    # filter out minionese rows (separate control)
    df = df[df["tier"] != "minionese"].copy()
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    df = df.sort_values(["tier", "language"])

    # derive ordered language list from first model (same for all)
    lang_order = (
        df[df["model"] == models[0]]
        .drop_duplicates("language")
        .sort_values(["tier", "language"])["language"]
        .tolist()
    )
    lang_labels = [LANG_FULL.get(l, l) for l in lang_order]

    sub = df[df["model"] == "CohereForAI/aya-expanse-8b"]
    tier_sizes = (
            sub.drop_duplicates("language")
            .sort_values(["tier", "language"])
            .groupby("tier", sort=False)["language"]
            .count()
        )
    for model in models:
        sub = df[df["model"] == model]
        pivot = (
            sub
            .pivot(index="language", columns="perturbation", values="asr_wildguard")
            .reindex(index=lang_order, columns=PERTURBATION_ORDER)
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pivot, ax=ax,
            cmap=ASR_CMAP, vmin=0, vmax=1,
            annot=True, fmt=".0%", annot_kws={"size": 7},
            linewidths=0.4, linecolor="#e0e0e0",
            cbar_kws={"label": "Attack Success Rate (ASR)", "shrink": 0.75, "format": mticker.PercentFormatter(xmax=1)},
        )
        ax.set_title(f"ASR Heatmap — {short_model(model)}", fontsize=13, pad=12)
        ax.set_xlabel("Perturbation Type", fontsize=10)
        ax.set_ylabel("Language", fontsize=10)
        ax.set_yticklabels(lang_labels, rotation=0, fontsize=8)
        ax.set_xticklabels(
            [PERTURBATION_LABELS.get(c, c) for c in pivot.columns],
            rotation=20, ha="right", fontsize=9,
        )

        # draw tier separators
        print("Tier sizes:\n", tier_sizes)
        cumulative = 0
        for tier in TIER_ORDER:
            if tier in tier_sizes.index:
                print("cumulative:", cumulative)
                cumulative += tier_sizes[tier]
                ax.axhline(cumulative, color="black", linewidth=1.5, linestyle='--')

        savefig(fig, out_dir, f"fig1_asr_heatmap_{safe_filename(short_model(model))}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – ASR by tier × perturbation, grouped bars per model
# ══════════════════════════════════════════════════════════════════════════════

def fig2_asr_by_tier(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: tier on x-axis, one group per perturbation, bars per model."""
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    df["perturbation"] = pd.Categorical(
        df["perturbation"], categories=PERTURBATION_ORDER, ordered=True
    )
    df = df.sort_values(["tier", "perturbation"])
    
    # print("unique models raw:", df["model"].unique())
    # print("nunique:", df["model"].nunique())
    # print(df[["model"]].drop_duplicates().to_list()[:10])

    models = sorted(df["model"].unique(), key=short_model)
    n_models = len(models)
    n_perturb = len(PERTURBATION_ORDER)

    fig, axes = plt.subplots(
        1, n_perturb,
        figsize=(4.5 * n_perturb, 5),
        sharey=True,
    )
    if n_perturb == 1:
        axes = [axes]

    colors = {m: PALETTE[i] for i, m in enumerate(models)}
    bar_width = 0.7 / n_models
    x = np.arange(len(TIER_ORDER))

    for ax, pert in zip(axes, PERTURBATION_ORDER):
        sub = df[df["perturbation"] == pert]
        for i, model in enumerate(models):
            mdf = sub[sub["model"] == model].set_index("tier")
            vals = [mdf.loc[t, "asr_wildguard"] if t in mdf.index else 0.0 for t in TIER_ORDER]
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset, vals,
                width=bar_width * 0.9,
                color=colors[model],
                label=short_model(model),
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(PERTURBATION_LABELS.get(pert, pert), fontsize=10, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TIER_LABELS.get(t, t) for t in TIER_ORDER],
            fontsize=8, rotation=15, ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        # tier_sizes = (
        #     sub.drop_duplicates("language")
        #     .sort_values(["tier", "language"])
        #     .groupby("tier", sort=False)["language"]
        #     .count()
        # )
        # cumulative = 0
        # for tier in TIER_ORDER:
        #     if tier in tier_sizes.index:
        #         cumulative += tier_sizes[tier]
        #         ax.axhline(cumulative, color="black", linewidth=1.5, linestyle='--')

    axes[0].set_ylabel("ASR (WildGuard)", fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in models
    ]
    
    fig.legend(
        handles, [short_model(m) for m in models],
        loc="lower center", ncol=n_models,
        fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle("ASR by Tier and Perturbation Type", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig2_asr_by_tier")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – Per-model delta-from-English heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig3_delta_heatmap(df: pd.DataFrame, out_dir: Path):
    """Shows how much each language raises ASR compared to English baseline."""
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)
    df = df.sort_values(["tier", "language"])

    models = df["model"].unique()
    lang_order = (
        df[df["model"] == models[0]]
        .drop_duplicates("language")
        .sort_values(["tier", "language"])["language"]
        .tolist()
    )
    lang_labels = [LANG_FULL.get(l, l) for l in lang_order]

    for model in models:
        sub = df[df["model"] == model]
        pivot = (
            sub
            .pivot(index="language", columns="perturbation", values="asr_delta_from_en")
            .reindex(index=lang_order, columns=PERTURBATION_ORDER)
        )

        abs_max = pivot.abs().max().max()
        abs_max = max(abs_max, 0.1)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pivot, ax=ax,
            cmap=DELTA_CMAP, vmin=-abs_max, vmax=abs_max,
            center=0,
            annot=True, fmt="+.1%", annot_kws={"size": 7},
            linewidths=0.4, linecolor="#e0e0e0",
            cbar_kws={"label": "ΔASR vs. English", "shrink": 0.75, "format": mticker.PercentFormatter(xmax=1)},
        )
        ax.set_title(
            f"ΔASR from English Baseline — {short_model(model)}",
            fontsize=13, pad=12,
        )
        ax.set_xlabel("Perturbation Type", fontsize=10)
        ax.set_ylabel("Language", fontsize=10)
        ax.set_yticklabels(lang_labels, rotation=0, fontsize=8)
        ax.set_xticklabels(
            [PERTURBATION_LABELS.get(c, c) for c in pivot.columns],
            rotation=20, ha="right", fontsize=9,
        )

        tier_sizes = (
            sub.drop_duplicates("language")
            .sort_values(["tier", "language"])
            .groupby("tier", sort=False)["language"]
            .count()
        )
        cumulative = 0
        for tier in TIER_ORDER:
            if tier in tier_sizes.index:
                cumulative += tier_sizes[tier]
                ax.axhline(cumulative, color="black", linewidth=1.5, linestyle='--')

        savefig(fig, out_dir, f"fig3_asr_delta_{safe_filename(short_model(model))}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Regime distribution (comply / refuse) stacked bar per model × tier
# ══════════════════════════════════════════════════════════════════════════════

def fig4_regime_distribution(df: pd.DataFrame, out_dir: Path):
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)

    models = sorted(df["model"].unique(), key=short_model)
    # print(f"  Models found: {models}")
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.5 * n_models, 5),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        summary = (
            sub.groupby(["tier", "regime"])
            .size()
            .unstack(fill_value=0)
            .reindex(TIER_ORDER)
        )
        # Normalise to fractions
        summary_pct = summary.div(summary.sum(axis=1), axis=0)

        regime_colors = {"comply": "#e05252", "refuse": "#5b9bd5", "mixed": "#f0a050"}
        bottom = np.zeros(len(TIER_ORDER))
        for regime in ["comply", "mixed", "refuse"]:
            if regime not in summary_pct.columns:
                continue
            vals = summary_pct[regime].values
            ax.bar(
                range(len(TIER_ORDER)), vals,
                bottom=bottom,
                color=regime_colors.get(regime, "#aaa"),
                label=regime.capitalize(),
                width=0.55,
                edgecolor="white",
                linewidth=0.6,
            )
            # label if big enough
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.07:
                    ax.text(
                        j, b + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                    )
            bottom += vals

        ax.set_title(short_model(model), fontsize=10, pad=8)
        ax.set_xticks(range(len(TIER_ORDER)))
        ax.set_xticklabels(
            [TIER_LABELS.get(t, t) for t in TIER_ORDER],
            fontsize=8, rotation=15, ha="right",
        )
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Proportion of language–perturbation cells", fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in [regime_colors["comply"], regime_colors.get("mixed", "#aaa"), regime_colors["refuse"]]
    ]
    fig.legend(
        handles, ["Comply", "Mixed", "Refuse"],
        loc="lower center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Safety Regime Distribution by Tier", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig4_regime_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – Cross-model comparison: mean ASR per perturbation, bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig5_model_comparison(df: pd.DataFrame, out_dir: Path):
    """
    Two panels:
      Left  – mean ASR per model × perturbation (excluding minionese)
      Right – mean ASR per model × tier
    """
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    models = sorted(df["model"].unique(), key=short_model)
    n_models = len(models)
    colors = {m: PALETTE[i] for i, m in enumerate(models)}

    fig, (ax_pert, ax_tier) = plt.subplots(1, 2, figsize=(13, 5))

    # — Left: by perturbation —
    mean_pert = (
        df.groupby(["model", "perturbation"])["asr_wildguard"].mean().reset_index()
    )
    mean_pert["perturbation"] = pd.Categorical(
        mean_pert["perturbation"], categories=PERTURBATION_ORDER, ordered=True
    )
    x = np.arange(len(PERTURBATION_ORDER))
    bw = 0.7 / n_models
    for i, model in enumerate(models):
        sub = mean_pert[mean_pert["model"] == model].set_index("perturbation")
        vals = [sub.loc[p, "asr_wildguard"] if p in sub.index else 0.0 for p in PERTURBATION_ORDER]
        offset = (i - (n_models - 1) / 2) * bw
        ax_pert.bar(
            x + offset, vals, width=bw * 0.9,
            color=colors[model], label=short_model(model),
            alpha=0.88, edgecolor="white", linewidth=0.4,
        )
    ax_pert.set_xticks(x)
    ax_pert.set_xticklabels(
        [PERTURBATION_LABELS.get(p, p) for p in PERTURBATION_ORDER],
        rotation=15, ha="right", fontsize=9,
    )
    ax_pert.set_ylim(0, 1.05)
    ax_pert.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_pert.set_ylabel("Mean ASR (WildGuard)", fontsize=10)
    ax_pert.set_title("Mean ASR by Perturbation Type", fontsize=11, pad=10)
    ax_pert.grid(axis="y", linestyle="--", alpha=0.4)
    ax_pert.spines[["top", "right"]].set_visible(False)

    # — Right: by tier —
    mean_tier = (
        df.groupby(["model", "tier"])["asr_wildguard"].mean().reset_index()
    )
    mean_tier["tier"] = pd.Categorical(mean_tier["tier"], categories=TIER_ORDER, ordered=True)
    x2 = np.arange(len(TIER_ORDER))
    for i, model in enumerate(models):
        sub = mean_tier[mean_tier["model"] == model].set_index("tier")
        vals = [sub.loc[t, "asr_wildguard"] if t in sub.index else 0.0 for t in TIER_ORDER]
        offset = (i - (n_models - 1) / 2) * bw
        ax_tier.bar(
            x2 + offset, vals, width=bw * 0.9,
            color=colors[model], label=short_model(model),
            alpha=0.88, edgecolor="white", linewidth=0.4,
        )
    ax_tier.set_xticks(x2)
    ax_tier.set_xticklabels(
        [TIER_LABELS.get(t, t) for t in TIER_ORDER],
        rotation=15, ha="right", fontsize=9,
    )
    ax_tier.set_ylim(0, 1.05)
    ax_tier.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_tier.set_title("Mean ASR by Language Tier", fontsize=11, pad=10)
    ax_tier.grid(axis="y", linestyle="--", alpha=0.4)
    ax_tier.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in models]
    fig.legend(
        handles, [short_model(m) for m in models],
        loc="lower center", ncol=n_models, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Cross-Model ASR Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig5_model_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Transliteration spike: line plot across languages per model
# ══════════════════════════════════════════════════════════════════════════════

def fig6_transliteration_spike(df: pd.DataFrame, out_dir: Path):
    """
    Highlights the 'transliteration loophole' by plotting per-language ASR
    for transliteration vs the mean of the other three perturbations.
    """
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    df["tier"] = pd.Categorical(df["tier"], categories=TIER_ORDER, ordered=True)

    models = sorted(df["model"].unique(), key=short_model)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].copy()
        other_mean = (
            sub[sub["perturbation"] != "transliteration"]
            .groupby("language")["asr_wildguard"]
            .mean()
            .rename("other_mean")
        )
        translit = (
            sub[sub["perturbation"] == "transliteration"]
            .set_index("language")["asr_wildguard"]
            .rename("transliteration")
        )
        compare = pd.concat([translit, other_mean], axis=1).dropna()
        # order by tier then language
        tier_map = sub.drop_duplicates("language").set_index("language")["tier"]
        compare["tier"] = compare.index.map(tier_map)
        compare["tier"] = pd.Categorical(compare["tier"], categories=TIER_ORDER, ordered=True)
        compare = compare.sort_values(["tier", "transliteration"], ascending=[True, False])

        x = np.arange(len(compare))
        lang_labels = [LANG_FULL.get(l, l) for l in compare.index]

        ax.plot(x, compare["transliteration"].values, "o-", color="#e05252",
                label="Transliteration", linewidth=1.8, markersize=5)
        ax.plot(x, compare["other_mean"].values,      "s--", color="#5b9bd5",
                label="Mean (other 3)", linewidth=1.4, markersize=4, alpha=0.85)
        ax.fill_between(
            x,
            compare["other_mean"].values,
            compare["transliteration"].values,
            where=compare["transliteration"].values > compare["other_mean"].values,
            alpha=0.12, color="#e05252",
        )

        # shade tier regions
        tier_sizes = compare.groupby("tier", sort=False).size()
        cumulative = 0
        tier_colors = ["#f0f4ff", "#e8f5e8", "#fff8e8", "#fce8e8"]
        for k, tier in enumerate(TIER_ORDER):
            if tier in tier_sizes.index:
                n = tier_sizes[tier]
                ax.axvspan(cumulative - 0.5, cumulative + n - 0.5,
                           alpha=0.25, color=tier_colors[k], zorder=0)
                ax.text(cumulative + n / 2 - 0.5, 1.02,
                        f"T{k+1}", ha="center", fontsize=7, color="#666")
                cumulative += n

        ax.set_xticks(x)
        ax.set_xticklabels(lang_labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(-0.05, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(short_model(model), fontsize=10, pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("ASR (WildGuard)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Transliteration ASR vs. Mean of Other Perturbations\n"
        "(shaded = transliteration exceeds mean)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig6_transliteration_spike")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – Scatter: coherence rate vs ASR (if coherent), coloured by regime
# ══════════════════════════════════════════════════════════════════════════════

def fig7_coherence_vs_asr(df: pd.DataFrame, out_dir: Path):
    """Coherence rate (x) vs ASR-if-coherent (y) scatter, one subplot per model."""
    df = df[df["tier"].isin(TIER_ORDER)].copy()
    models = sorted(df["model"].unique(), key=short_model)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    regime_color = {"comply": "#e05252", "refuse": "#5b9bd5", "mixed": "#f0a050"}

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for _, row in sub.iterrows():
            color = regime_color.get(str(row.get("regime", "")).lower(), "#aaa")
            ax.scatter(
                row["coherence_rate"], row["asr_if_coherent"],
                c=color, alpha=0.7, s=40, edgecolors="none",
            )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Coherence Rate", fontsize=9)
        ax.set_title(short_model(model), fontsize=10, pad=8)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=0.8)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("ASR if Coherent", fontsize=10)
    handles = [plt.Circle((0, 0), 0.5, color=c) for c in [regime_color["comply"], regime_color.get("mixed","#aaa"), regime_color["refuse"]]]
    fig.legend(
        handles, ["Comply", "Mixed", "Refuse"],
        loc="lower center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle("Coherence Rate vs. ASR (when coherent)", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "fig7_coherence_vs_asr")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--results-dir", help="Directory containing the four evaluation CSVs.")
    grp.add_argument("--asr-detailed",  help="Path to asr_detailed.csv")
    p.add_argument("--asr-by-tier",  default=None, help="Path to asr_by_tier.csv")
    p.add_argument("--asr-delta",    default=None, help="Path to asr_delta_from_english.csv")
    p.add_argument("--coherence",    default=None, help="Path to coherence_table.csv")
    p.add_argument("--output-dir",   default="figures/", help="Where to save the figures.")
    p.add_argument("--dpi",          type=int, default=150)
    p.add_argument(
        "--figures", nargs="+",
        default=["1", "2", "3", "4", "5", "6", "7"],
        help="Which figure numbers to produce (default: all).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Fall back to results-dir discovery if individual paths not given
    if args.asr_detailed and not args.results_dir:
        base = Path(args.asr_detailed).parent
        if not args.asr_by_tier:
            args.asr_by_tier = str(base / "asr_by_tier.csv")
        if not args.asr_delta:
            args.asr_delta   = str(base / "asr_delta_from_english.csv")
        if not args.coherence:
            args.coherence   = str(base / "coherence_table.csv")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs…")
    data = load_all(args)

    figures_to_run = set(args.figures)
    print(f"\nProducing figures: {sorted(figures_to_run)}")

    if "1" in figures_to_run:
        print("\n[Fig 1] ASR heatmaps (per model)…")
        fig1_asr_heatmap(data["detailed"], out_dir)

    if "2" in figures_to_run:
        print("\n[Fig 2] ASR by tier grouped bars…")
        fig2_asr_by_tier(data["by_tier"], out_dir)

    if "3" in figures_to_run:
        print("\n[Fig 3] ΔASR from English heatmaps…")
        fig3_delta_heatmap(data["delta"], out_dir)

    if "4" in figures_to_run:
        print("\n[Fig 4] Regime distribution…")
        fig4_regime_distribution(data["coherence"], out_dir)

    if "5" in figures_to_run:
        print("\n[Fig 5] Cross-model comparison…")
        fig5_model_comparison(data["detailed"], out_dir)

    if "6" in figures_to_run:
        print("\n[Fig 6] Transliteration spike…")
        fig6_transliteration_spike(data["detailed"], out_dir)

    if "7" in figures_to_run:
        print("\n[Fig 7] Coherence vs ASR scatter…")
        fig7_coherence_vs_asr(data["coherence"], out_dir)

    print(f"\nAll done. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
