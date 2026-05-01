"""
visualize_geometry.py

Produces six figures for the "Geometric Analysis of Refusal" section,
using representation analysis outputs from Aya, Llama, and Qwen.

Expected directory structure:
    data/
        aya/
            silhouette_scores.csv
            probe_summary.csv
            principal_angles.csv
            effective_rank.csv
            disentangle_results.csv
            refusal_direction_aya.npy      # optional — used for Fig 6
        llama/   (same files, refusal_direction_llama.npy)
        qwen/    (same files, refusal_direction_qwen.npy)

Figures produced:
    fig1_silhouette_heatmap.pdf        -- Harmful-harmless separation (lang x layer)
    fig2_probe_accuracy_heatmap.pdf    -- Linear probe AUC (lang x layer)
    fig3_principal_angles.pdf          -- Safety subspace angle vs English by tier
    fig4_effective_rank.pdf            -- Representation effective rank by tier & layer
    fig5_disentanglement.pdf           -- Harm vs refusal component norms + failure types
    fig6_cross_model_silhouette.pdf    -- Silhouette score comparison across all models

Usage:
    python visualize_geometry.py --data-dir data/ --output-dir figures/

    # Per-model overrides:
    python visualize_geometry.py \\
        --aya-dir   results/representation/aya/ \\
        --llama-dir results/representation/llama/ \\
        --qwen-dir  results/representation/qwen/ \\
        --output-dir figures/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# ── Language metadata ─────────────────────────────────────────────────────────

TIER_MAP = {
    "en": "tier_1", "de": "tier_1", "fr": "tier_1", "zh": "tier_1", "es": "tier_1",
    "ar": "tier_2", "ru": "tier_2", "ko": "tier_2", "ja": "tier_2",
    "tr": "tier_3", "id": "tier_3", "hi": "tier_3", "sw": "tier_3",
    "yo": "tier_4", "zu": "tier_4", "gd": "tier_4", "gn": "tier_4", "jw": "tier_4",
}
TIER_ORDER  = ["tier_1", "tier_2", "tier_3", "tier_4"]
TIER_COLORS = {
    "tier_1": "#2166AC",
    "tier_2": "#74ADD1",
    "tier_3": "#F4A582",
    "tier_4": "#D6604D",
}
TIER_LABELS = {
    "tier_1": "Tier 1 (High)",
    "tier_2": "Tier 2",
    "tier_3": "Tier 3",
    "tier_4": "Tier 4 (Low)",
}

# Language display order: tier-sorted then alpha within tier
LANG_ORDER = ["en", "de", "es", "fr", "zh",
              "ar", "ja", "ko", "ru",
              "hi", "id", "sw", "tr",
              "gd", "gn", "yo", "zu"]

MODEL_COLORS  = {"aya": "#E07B39", "llama": "#4A90D9", "qwen": "#5DBB8A"}
MODEL_LABELS  = {
    "aya":   "Aya-Expanse-8B",
    "llama": "Llama-3.1-8B-Instruct",
    "qwen":  "Qwen2.5-7B-Instruct",
}
MODEL_MARKERS = {"aya": "o", "llama": "s", "qwen": "^"}

FAILURE_COLORS = {
    "upstream":   "#9B59B6",
    "subthreshold": "#E74C3C",
    "mixed":      "#F39C12",
    "none":       "#27AE60",
    "unknown":    "#95A5A6",
}
FAILURE_LABELS = {
    "upstream":     "Upstream (semantic recovery failure)",
    "subthreshold": "Subthreshold activation",
    "mixed":        "Mixed signal (benign-feature confusion)",
    "none":         "No failure (refusal intact)",
    "unknown":      "Unclassified",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_model_data(model_dir: Path, model: str) -> dict | None:
    files = {
        "silhouette":   "silhouette_scores.csv",
        "probe":        "probe_summary.csv",
        "angles":       "principal_angles.csv",
        "rank":         "effective_rank.csv",
        "disentangle":  "disentangle_results.csv",
    }
    data = {"model": model}
    found_any = False
    for key, fname in files.items():
        path = model_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model
            # Attach tier
            if "language" in df.columns:
                df["tier"] = df["language"].map(TIER_MAP).fillna("tier_4")
            data[key] = df
            found_any = True
        else:
            print(f"  [WARN] Missing {path}")
            data[key] = None

    # Optional: refusal direction vector
    rd_path = model_dir / f"refusal_direction_{model}.npy"
    if rd_path.exists():
        data["refusal_direction"] = np.load(str(rd_path))
    else:
        data["refusal_direction"] = None

    return data if found_any else None


def _ordered_langs(df: pd.DataFrame) -> list:
    """Return languages present in df, sorted by LANG_ORDER then alpha."""
    present = set(df["language"].unique())
    ordered = [l for l in LANG_ORDER if l in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _tier_dividers(lang_list: list) -> list[float]:
    """Return heatmap row indices where tier changes (for dashed lines)."""
    dividers = []
    prev_tier = TIER_MAP.get(lang_list[0], "tier_4")
    for i, lang in enumerate(lang_list[1:], 1):
        t = TIER_MAP.get(lang, "tier_4")
        if t != prev_tier:
            dividers.append(i)
            prev_tier = t
    return dividers


def _norm_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Add layer_norm column (0-1) based on max layer in dataframe."""
    df = df.copy()
    mx = df["layer"].max()
    df["layer_norm"] = df["layer"] / mx if mx > 0 else df["layer"]
    return df


# ── Figure 1: Silhouette score heatmap ───────────────────────────────────────

def plot_fig1(all_data: dict, output_path: Path):
    """
    Heatmap: language (y, tier-sorted) × layer (x).
    Colour = silhouette score of harmful vs harmless in activation space.
    One panel per model.
    """
    models = list(all_data.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6.5), squeeze=False)
    axes = axes[0]

    vmin, vmax = -0.3, 0.9

    for ax, model in zip(axes, models):
        sil = all_data[model].get("silhouette")
        if sil is None:
            ax.set_visible(False)
            continue

        langs = _ordered_langs(sil)
        pivot = (sil.pivot_table(index="language", columns="layer",
                                  values="silhouette_score", aggfunc="mean")
                    .reindex(langs))

        im = sns.heatmap(
            pivot, ax=ax,
            cmap="RdYlGn", vmin=vmin, vmax=vmax,
            linewidths=0,
            cbar=True,
            cbar_kws={"shrink": 0.7, "label": "Silhouette score"},
            yticklabels=langs,
        )

        # Tier boundary dashed lines
        for div in _tier_dividers(langs):
            ax.axhline(div, color="white", linewidth=1.8, linestyle="--")

        # Tier labels on left margin
        tier_starts = {}
        for i, lang in enumerate(langs):
            t = TIER_MAP.get(lang, "tier_4")
            if t not in tier_starts:
                tier_starts[t] = i
        for tier, start in tier_starts.items():
            end = next((tier_starts[t2] for t2 in TIER_ORDER
                        if TIER_ORDER.index(t2) > TIER_ORDER.index(tier)
                        and t2 in tier_starts), len(langs))
            mid = (start + end) / 2
            ax.text(-1.5, mid, TIER_LABELS[tier].split()[0] + "\n" +
                    TIER_LABELS[tier].split()[1],
                    va="center", ha="right", fontsize=7,
                    color=TIER_COLORS[tier], fontweight="bold")

        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Layer")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7.5)

        # Tick every 4 layers
        n_layers = pivot.shape[1]
        ticks = list(range(0, n_layers, 4))
        ax.set_xticks([t + 0.5 for t in ticks])
        ax.set_xticklabels([str(pivot.columns[t]) for t in ticks
                            if t < len(pivot.columns)], rotation=0)

    fig.suptitle(
        "Harmful–Harmless Separation in Activation Space\n"
        "(Silhouette Score; higher = cleaner separation)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 2: Probe accuracy heatmap ─────────────────────────────────────────

def plot_fig2(all_data: dict, output_path: Path):
    """
    Heatmap: language × layer, colour = probe cv_auc (category='all').
    One panel per model.  Diverging colourmap centred at 0.5 (chance).
    """
    models = list(all_data.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6.5), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        probe = all_data[model].get("probe")
        if probe is None:
            ax.set_visible(False)
            continue

        sub = probe[probe["category"] == "all"]
        langs = _ordered_langs(sub)
        pivot = (sub.pivot_table(index="language", columns="layer",
                                  values="cv_auc", aggfunc="mean")
                    .reindex(langs))

        sns.heatmap(
            pivot, ax=ax,
            cmap="RdYlGn", vmin=0.4, vmax=1.0,
            linewidths=0,
            cbar=True,
            cbar_kws={"shrink": 0.7, "label": "Probe AUC"},
            yticklabels=langs,
        )

        for div in _tier_dividers(langs):
            ax.axhline(div, color="white", linewidth=1.8, linestyle="--")

        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Layer")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=7.5)

        n_layers = pivot.shape[1]
        ticks = list(range(0, n_layers, 4))
        ax.set_xticks([t + 0.5 for t in ticks])
        ax.set_xticklabels([str(pivot.columns[t]) for t in ticks
                            if t < len(pivot.columns)], rotation=0)

        # Chance-level annotation
        ax.text(0.98, 0.01, "Chance = 0.5", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="dimgray")

    fig.suptitle(
        "Linear Probe AUC for Harmfulness Detection\n"
        "(category='all'; chance = 0.5)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 3: Principal angles vs English ────────────────────────────────────

def plot_fig3(all_data: dict, output_path: Path):
    """
    For each model: per-language principal angle (first angle only) between
    that language's safety subspace and English, plotted vs layer.
    Lines coloured by tier.  Shaded band = ±1 std across languages in tier.
    Bottom panel: tier-averaged angle across all models.
    """
    models = list(all_data.keys())
    n_models = len(models)

    fig = plt.figure(figsize=(5.5 * n_models, 9))
    gs  = gridspec.GridSpec(2, n_models, hspace=0.45, wspace=0.3,
                            height_ratios=[1.6, 1])

    # ── Row 1: per-model per-language lines ───────────────────────────────────
    for col, model in enumerate(models):
        ax = fig.add_subplot(gs[0, col])
        pa = all_data[model].get("angles")
        if pa is None:
            ax.set_visible(False)
            continue

        sub = pa[pa["angle_idx"] == 0].copy()
        sub = _norm_layer(sub)

        for tier in TIER_ORDER:
            tier_langs = [l for l in sub["language"].unique()
                         if TIER_MAP.get(l, "tier_4") == tier]
            if not tier_langs:
                continue
            tier_sub = sub[sub["language"].isin(tier_langs)]
            agg = (tier_sub.groupby("layer_norm")["angle_deg"]
                   .agg(["mean", "std"]).reset_index())

            ax.plot(agg["layer_norm"], agg["mean"],
                    color=TIER_COLORS[tier], linewidth=2,
                    label=TIER_LABELS[tier])
            ax.fill_between(agg["layer_norm"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color=TIER_COLORS[tier], alpha=0.15)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 95)
        ax.set_xlabel("Normalised layer depth")
        ax.set_ylabel("Principal angle vs English (°)" if col == 0 else "")
        ax.set_title(MODEL_LABELS.get(model, model))
        if col == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.7)

    # ── Row 2: tier-averaged across all models ────────────────────────────────
    ax_bot = fig.add_subplot(gs[1, :])

    for model in models:
        pa = all_data[model].get("angles")
        if pa is None:
            continue
        sub = pa[pa["angle_idx"] == 0].copy()
        sub = _norm_layer(sub)
        agg = sub.groupby("layer_norm")["angle_deg"].mean().reset_index()
        ax_bot.plot(agg["layer_norm"], agg["angle_deg"],
                    color=MODEL_COLORS.get(model, "#333"),
                    linewidth=2.2,
                    label=MODEL_LABELS.get(model, model),
                    marker=MODEL_MARKERS.get(model, "o"),
                    markersize=4)

    ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_bot.set_ylim(0, 95)
    ax_bot.set_xlabel("Normalised layer depth")
    ax_bot.set_ylabel("Mean principal angle (°)\naveraged over all non-EN languages")
    ax_bot.set_title("All Models — Mean Angle vs English")
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.8)

    fig.suptitle(
        "Principal Angle Between Non-English and English Safety Subspaces",
        fontsize=12, y=1.01,
    )
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 4: Effective rank by tier and layer ────────────────────────────────

def plot_fig4(all_data: dict, output_path: Path):
    """
    Two panels:
      Left:  effective rank vs layer for each tier (per model, one subplot per model).
      Right: effective rank at the critical layer, boxplot per tier, all models combined.
    """
    models = list(all_data.keys())
    n_models = len(models)

    fig = plt.figure(figsize=(5 * n_models + 4, 4.5))
    gs  = gridspec.GridSpec(1, n_models + 1,
                             width_ratios=[1] * n_models + [1.1],
                             wspace=0.35)

    # ── Left panels: rank vs layer per model ─────────────────────────────────
    for col, model in enumerate(models):
        ax = fig.add_subplot(gs[0, col])
        rank = all_data[model].get("rank")
        if rank is None:
            ax.set_visible(False)
            continue

        rank_n = _norm_layer(rank)
        for tier in TIER_ORDER:
            tier_langs = [l for l in rank_n["language"].unique()
                         if TIER_MAP.get(l, "tier_4") == tier]
            if not tier_langs:
                continue
            tier_sub = rank_n[rank_n["language"].isin(tier_langs)]
            agg = (tier_sub.groupby("layer_norm")["effective_rank"]
                   .agg(["mean", "std"]).reset_index())
            ax.plot(agg["layer_norm"], agg["mean"],
                    color=TIER_COLORS[tier], linewidth=2,
                    label=TIER_LABELS[tier])
            ax.fill_between(agg["layer_norm"],
                            (agg["mean"] - agg["std"]).clip(0),
                            agg["mean"] + agg["std"],
                            color=TIER_COLORS[tier], alpha=0.15)

        ax.set_xlabel("Normalised layer depth")
        ax.set_ylabel("Effective rank" if col == 0 else "")
        ax.set_title(MODEL_LABELS.get(model, model))
        if col == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.7)

    # ── Right panel: boxplot at critical layer, all models ───────────────────
    ax_box = fig.add_subplot(gs[0, n_models])
    box_data = []
    for model in models:
        rank = all_data[model].get("rank")
        if rank is None:
            continue
        # Critical layer = median layer in data
        crit = int(rank["layer"].median())
        sub = rank[rank["layer"] == crit].copy()
        for tier in TIER_ORDER:
            tier_langs = [l for l in sub["language"].unique()
                         if TIER_MAP.get(l, "tier_4") == tier]
            vals = sub[sub["language"].isin(tier_langs)]["effective_rank"].dropna()
            for v in vals:
                box_data.append({"tier": tier, "model": model, "effective_rank": v})

    if box_data:
        bdf = pd.DataFrame(box_data)
        positions = {t: i for i, t in enumerate(TIER_ORDER)}
        offset = {"aya": -0.22, "llama": 0.0, "qwen": 0.22}
        bp_width = 0.18
        for model in models:
            mdf = bdf[bdf["model"] == model]
            for tier in TIER_ORDER:
                tvals = mdf[mdf["tier"] == tier]["effective_rank"].values
                if len(tvals) == 0:
                    continue
                x = positions[tier] + offset.get(model, 0)
                bp = ax_box.boxplot(
                    tvals, positions=[x], widths=bp_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=MODEL_COLORS.get(model, "#AAA"),
                                  alpha=0.7),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color=MODEL_COLORS.get(model, "#AAA")),
                    capprops=dict(color=MODEL_COLORS.get(model, "#AAA")),
                    flierprops=dict(marker=".", markersize=4,
                                   color=MODEL_COLORS.get(model, "#AAA"),
                                   alpha=0.5),
                    manage_ticks=False,
                )

        ax_box.set_xticks(list(positions.values()))
        ax_box.set_xticklabels([TIER_LABELS[t].split()[0] + "\n" +
                                 TIER_LABELS[t].split()[1]
                                 for t in TIER_ORDER], fontsize=8)
        ax_box.set_ylabel("Effective rank at critical layer")
        ax_box.set_title("All Models\n(at critical layer)")

        model_patches = [mpatches.Patch(
            color=MODEL_COLORS.get(m, "#AAA"), alpha=0.7,
            label=MODEL_LABELS.get(m, m)) for m in models]
        ax_box.legend(handles=model_patches, fontsize=7, loc="upper right",
                      framealpha=0.7)

    fig.suptitle(
        "Effective Rank of Harmful Activations by Language Tier and Layer",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 5: Disentanglement ─────────────────────────────────────────────────

def plot_fig5(all_data: dict, output_path: Path):
    """
    Four panels per model:
      A: harm_component_norm vs refusal_component_norm scatter (coloured by tier)
      B: t_inst_harm_signal vs t_post_inst_refusal_signal scatter
      C: failure type breakdown stacked bar per language
      D: harm & refusal signal by tier (line, averaged over critical layers)
    """
    models = list(all_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(
        4, n_models,
        figsize=(5.5 * n_models, 16),
        squeeze=False,
    )

    for col, model in enumerate(models):
        dis = all_data[model].get("disentangle")
        if dis is None:
            for row in range(4):
                axes[row][col].set_visible(False)
            continue

        # Average over layers for per-language summaries
        lang_agg = (dis.groupby(["language", "tier"])
                       .agg(
                           harm_norm   =("harm_component_norm", "mean"),
                           refusal_norm=("refusal_component_norm", "mean"),
                           t_harm      =("t_inst_harm_signal", "mean"),
                           t_refusal   =("t_post_inst_refusal_signal", "mean"),
                       )
                       .reset_index())

        # ── Panel A: harm vs refusal norm scatter ─────────────────────────────
        ax_a = axes[0][col]
        for tier in TIER_ORDER:
            sub = lang_agg[lang_agg["tier"] == tier]
            ax_a.scatter(sub["harm_norm"], sub["refusal_norm"],
                         color=TIER_COLORS[tier], s=60, alpha=0.85,
                         label=TIER_LABELS[tier], zorder=3)
            for _, r in sub.iterrows():
                ax_a.annotate(r["language"],
                              (r["harm_norm"], r["refusal_norm"]),
                              fontsize=6.5, xytext=(3, 3),
                              textcoords="offset points")

        ax_a.set_xlabel("Harm component norm")
        ax_a.set_ylabel("Refusal component norm")
        ax_a.set_title(f"{MODEL_LABELS.get(model, model)}\nHarm vs Refusal Component Norms")
        if col == 0:
            ax_a.legend(fontsize=7, framealpha=0.7)

        # ── Panel B: t_inst vs t_post scatter ─────────────────────────────────
        ax_b = axes[1][col]
        for tier in TIER_ORDER:
            sub = lang_agg[lang_agg["tier"] == tier]
            ax_b.scatter(sub["t_harm"], sub["t_refusal"],
                         color=TIER_COLORS[tier], s=60, alpha=0.85,
                         label=TIER_LABELS[tier], zorder=3)
            for _, r in sub.iterrows():
                ax_b.annotate(r["language"],
                              (r["t_harm"], r["t_refusal"]),
                              fontsize=6.5, xytext=(3, 3),
                              textcoords="offset points")

        # Pearson r
        if len(lang_agg) > 3:
            r_val, _ = pearsonr(lang_agg["t_harm"], lang_agg["t_refusal"])
            ax_b.text(0.05, 0.93, f"r = {r_val:.2f}",
                      transform=ax_b.transAxes, fontsize=8, color="dimgray")

        ax_b.set_xlabel("Harm signal at t_inst")
        ax_b.set_ylabel("Refusal signal at t_post_inst")
        ax_b.set_title("Harm–Refusal Signal Decoupling\n(per language, averaged over critical layers)")

        # ── Panel C: failure type stacked bar ─────────────────────────────────
        ax_c = axes[2][col]
        # Count dominant failure type per language
        fail_counts = (dis.groupby(["language", "failure_type"])
                          .size().unstack(fill_value=0))
        fail_types = [f for f in FAILURE_COLORS if f in fail_counts.columns]
        langs = _ordered_langs(dis)
        fail_counts = fail_counts.reindex(langs, fill_value=0)

        bottom = np.zeros(len(langs))
        x = np.arange(len(langs))
        for ft in fail_types:
            vals = fail_counts[ft].values if ft in fail_counts.columns \
                   else np.zeros(len(langs))
            ax_c.bar(x, vals, bottom=bottom,
                     color=FAILURE_COLORS[ft],
                     label=FAILURE_LABELS.get(ft, ft) if col == 0 else "")
            bottom += vals

        # Tier dividers
        for div in _tier_dividers(langs):
            ax_c.axvline(div - 0.5, color="gray", linewidth=1.2,
                         linestyle="--", alpha=0.7)

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(langs, rotation=45, ha="right", fontsize=7.5)
        ax_c.set_ylabel("Layer-count")
        ax_c.set_title("Failure Type Distribution per Language")
        if col == 0:
            ax_c.legend(fontsize=6.5, loc="upper right", framealpha=0.7)

        # ── Panel D: harm & refusal signal by tier across layers ──────────────
        ax_d = axes[3][col]
        dis_n = _norm_layer(dis)
        tier_layer = (dis_n.groupby(["tier", "layer_norm"])
                           .agg(harm=("t_inst_harm_signal", "mean"),
                                refusal=("t_post_inst_refusal_signal", "mean"))
                           .reset_index())

        for tier in TIER_ORDER:
            tsub = tier_layer[tier_layer["tier"] == tier]
            if tsub.empty:
                continue
            color = TIER_COLORS[tier]
            ax_d.plot(tsub["layer_norm"], tsub["harm"],
                      color=color, linewidth=2,
                      linestyle="-",
                      label=f"{TIER_LABELS[tier]} — harm" if col == 0 else "")
            ax_d.plot(tsub["layer_norm"], tsub["refusal"],
                      color=color, linewidth=2,
                      linestyle="--",
                      label=f"{TIER_LABELS[tier]} — refusal" if col == 0 else "")

        ax_d.set_xlabel("Normalised layer depth")
        ax_d.set_ylabel("Signal magnitude")
        ax_d.set_title("Harm & Refusal Signal by Tier Across Layers\n(solid = harm, dashed = refusal)")
        if col == 0:
            ax_d.legend(fontsize=6.5, ncol=2, framealpha=0.7)

    fig.suptitle("Disentanglement of Harmfulness and Refusal Representations",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 6: Cross-model silhouette comparison ───────────────────────────────

def plot_fig6(all_data: dict, output_path: Path):
    """
    Two panels:
      Left:  per-language silhouette at critical layer, one line per model,
             x-axis ordered by tier.
      Right: tier-averaged silhouette vs normalised layer depth,
             one line per model × tier.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: per-language at critical layer ─────────────────────────────────
    lang_scores = {}
    for model in all_data:
        sil = all_data[model].get("silhouette")
        if sil is None:
            continue
        crit = int(sil["layer"].median())
        sub = sil[sil["layer"] == crit]
        lang_scores[model] = sub.set_index("language")["silhouette_score"]

    all_langs = _ordered_langs(
        pd.concat([s.reset_index() for s in lang_scores.values()],
                  ignore_index=True).rename(columns={"index": "language"})
        if lang_scores else pd.DataFrame(columns=["language"])
    )
    # Recalculate using union of available langs
    available_langs = sorted(
        set().union(*[set(v.index) for v in lang_scores.values()]),
        key=lambda l: LANG_ORDER.index(l) if l in LANG_ORDER else 999
    )

    x = np.arange(len(available_langs))
    for model in all_data:
        if model not in lang_scores:
            continue
        scores = [lang_scores[model].get(l, np.nan) for l in available_langs]
        ax_left.plot(x, scores,
                     color=MODEL_COLORS.get(model, "#333"),
                     marker=MODEL_MARKERS.get(model, "o"),
                     markersize=5, linewidth=1.8,
                     label=MODEL_LABELS.get(model, model))

    # Tier shading
    tier_regions = []
    current_tier = TIER_MAP.get(available_langs[0], "tier_4") if available_langs else None
    start_idx = 0
    for i, lang in enumerate(available_langs):
        t = TIER_MAP.get(lang, "tier_4")
        if t != current_tier or i == len(available_langs) - 1:
            end_idx = i if t != current_tier else i + 1
            ax_left.axvspan(start_idx - 0.5, end_idx - 0.5,
                            alpha=0.07, color=TIER_COLORS.get(current_tier, "#AAA"))
            current_tier = t
            start_idx = i

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(available_langs, rotation=45, ha="right", fontsize=8)
    ax_left.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_left.set_ylim(-0.35, 1.0)
    ax_left.set_ylabel("Silhouette score at critical layer")
    ax_left.set_title("Per-Language Separation at Critical Layer")
    ax_left.legend(fontsize=8, framealpha=0.8)

    # Tier labels on x-axis
    tier_label_x = {}
    for i, lang in enumerate(available_langs):
        t = TIER_MAP.get(lang, "tier_4")
        if t not in tier_label_x:
            tier_label_x[t] = []
        tier_label_x[t].append(i)
    for tier, idxs in tier_label_x.items():
        mid = np.mean(idxs)
        ax_left.text(mid, -0.42, TIER_LABELS[tier].split()[0],
                     ha="center", fontsize=7,
                     color=TIER_COLORS[tier], fontweight="bold",
                     transform=ax_left.get_xaxis_transform())

    # ── Right: tier × model across layers ────────────────────────────────────
    for model in all_data:
        sil = all_data[model].get("silhouette")
        if sil is None:
            continue
        sil_n = _norm_layer(sil)
        for tier in TIER_ORDER:
            tier_langs = [l for l in sil_n["language"].unique()
                         if TIER_MAP.get(l, "tier_4") == tier]
            tier_sub = sil_n[sil_n["language"].isin(tier_langs)]
            agg = tier_sub.groupby("layer_norm")["silhouette_score"].mean()
            linestyle = "-" if model == "aya" else ("--" if model == "llama" else ":")
            ax_right.plot(agg.index, agg.values,
                          color=TIER_COLORS[tier],
                          linewidth=1.8, linestyle=linestyle,
                          alpha=0.9)

    ax_right.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_right.set_ylim(-0.35, 1.0)
    ax_right.set_xlabel("Normalised layer depth")
    ax_right.set_ylabel("Mean silhouette score")
    ax_right.set_title("Tier-Averaged Separation Across Layers\n(all models)")

    # Custom legend: tier colours + model linestyles
    tier_patches = [mpatches.Patch(color=TIER_COLORS[t],
                                   label=TIER_LABELS[t]) for t in TIER_ORDER]
    model_lines  = [mlines.Line2D([], [],
                                  color="gray",
                                  linestyle="-" if m == "aya" else ("--" if m == "llama" else ":"),
                                  linewidth=1.8,
                                  label=MODEL_LABELS.get(m, m))
                    for m in all_data]
    ax_right.legend(handles=tier_patches + model_lines,
                    fontsize=7, loc="lower right", framealpha=0.8,
                    ncol=2)

    fig.suptitle(
        "Cross-Model Harmful–Harmless Separation by Language Tier",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise geometric analysis of refusal across Aya, Llama, and Qwen."
    )
    parser.add_argument("--data-dir",   default=None,
                        help="Root dir with aya/, llama/, qwen/ subdirectories.")
    parser.add_argument("--aya-dir",    default=None)
    parser.add_argument("--llama-dir",  default=None)
    parser.add_argument("--qwen-dir",   default=None)
    parser.add_argument("--output-dir", default="figures/")
    parser.add_argument("--figures",    nargs="+", type=int, default=[1,2,3,4,5,6],
                        help="Which figures to produce, e.g. --figures 1 3 5")
    return parser.parse_args()


def resolve_model_dirs(args) -> dict:
    dirs = {}
    if args.data_dir:
        root = Path(args.data_dir)
        for model in ["aya", "llama", "qwen"]:
            candidate = root / model
            if candidate.is_dir():
                dirs[model] = candidate
    for model, attr in [("aya", "aya_dir"), ("llama", "llama_dir"), ("qwen", "qwen_dir")]:
        val = getattr(args, attr)
        if val is not None:
            dirs[model] = Path(val)
    return dirs


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dirs = resolve_model_dirs(args)
    if not model_dirs:
        print("ERROR: no model directories found. Use --data-dir or "
              "--aya-dir / --llama-dir / --qwen-dir.")
        sys.exit(1)

    print(f"Loading data for: {list(model_dirs.keys())}")
    all_data = {}
    for model, path in model_dirs.items():
        d = load_model_data(path, model)
        if d is not None:
            all_data[model] = d
            print(f"  Loaded '{model}' from {path}")

    if not all_data:
        print("ERROR: no valid data loaded.")
        sys.exit(1)

    figs = set(args.figures)
    dispatch = {
        1: ("fig1_silhouette_heatmap.pdf",      plot_fig1),
        2: ("fig2_probe_accuracy_heatmap.pdf",  plot_fig2),
        3: ("fig3_principal_angles.pdf",        plot_fig3),
        4: ("fig4_effective_rank.pdf",          plot_fig4),
        5: ("fig5_disentanglement.pdf",         plot_fig5),
        6: ("fig6_cross_model_silhouette.pdf",  plot_fig6),
    }
    for fig_num, (fname, fn) in dispatch.items():
        if fig_num in figs:
            fn(all_data, outdir / fname)

    print(f"\nDone. All figures written to {outdir}/")


if __name__ == "__main__":
    main()
