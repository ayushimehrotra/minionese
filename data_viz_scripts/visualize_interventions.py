"""
visualize_interventions.py

Produces four figures from intervention sweep results across Aya, Llama, and Qwen.

Expected directory structure:
    results/
        aya/
            sweep_results.csv
            pareto_frontier.csv
        llama/
            sweep_results.csv
            pareto_frontier.csv
        qwen/
            sweep_results.csv
            pareto_frontier.csv

CSV columns expected:
    sweep_results.csv  : intervention, param_value, language, tier,
                         asr, over_refusal, langid_consistency, mmlu_accuracy
    pareto_frontier.csv: all of the above + safety, utility, is_pareto_optimal

Usage:
    python visualize_interventions.py --data-dir results/ --output-dir figures/

    # Per-model overrides:
    python visualize_interventions.py \\
        --aya-dir   results/interventions/aya/ \\
        --llama-dir results/interventions/llama/ \\
        --qwen-dir  results/interventions/qwen/ \\
        --output-dir figures/

# Standard usage
python visualize_interventions.py --data-dir results/ --output-dir figures/

# Per-model overrides
python3 visualize_interventions.py \
  --aya-dir   aya/interventions/ \
  --llama-dir llama/interventions/ \
  --qwen-dir  qwen/interventions/ \
  --output-dir figures/

# Specific figures only
python visualize_interventions.py --data-dir results/ --figures 1 3 4
Figures produced:
    fig1_pareto_per_model.pdf      -- Safety vs utility Pareto per model + combined overlay
    fig2_param_sweep_curves.pdf    -- ASR & over-refusal vs param value, per intervention x model
    fig3_pareto_overlay.pdf        -- All models' Pareto frontiers on one canvas
    fig4_best_param_comparison.pdf -- Bar chart at optimal param setting per intervention x model
    fig5_tier_breakdown.pdf        -- Per-tier ASR at best param (if tier-level data exists)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Aesthetics ────────────────────────────────────────────────────────────────

MODEL_COLORS  = {"aya": "#E07B39", "llama": "#4A90D9", "qwen": "#5DBB8A"}
MODEL_LABELS  = {
    "aya":   "Aya-Expanse-8B",
    "llama": "Llama-3.1-8B-Instruct",
    "qwen":  "Qwen2.5-7B-Instruct",
}
MODEL_MARKERS = {"aya": "o", "llama": "s", "qwen": "^"}

INTERV_COLORS = {
    "baseline":            "#AAAAAA",
    "caa":                 "#E74C3C",
    "sae_clamp":           "#9B59B6",
    "subspace_projection": "#2980B9",
}
INTERV_LABELS = {
    "baseline":            "Baseline",
    "caa":                 "CAA",
    "sae_clamp":           "SAE Clamp",
    "subspace_projection": "Subspace Proj.",
}
INTERV_MARKERS = {
    "baseline":            "D",
    "caa":                 "o",
    "sae_clamp":           "^",
    "subspace_projection": "s",
}

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

# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_model_data(model_dir: Path, model: str) -> dict | None:
    sweep_path  = model_dir / "sweep_results.csv"
    pareto_path = model_dir / "pareto_frontier.csv"

    if not sweep_path.exists() and not pareto_path.exists():
        print(f"[WARN] No CSVs found in {model_dir} — skipping '{model}'.")
        return None

    data = {}

    if sweep_path.exists():
        df = pd.read_csv(sweep_path)
        df["model"] = model
        df = _add_derived_cols(df)
        data["sweep"] = df
    else:
        data["sweep"] = None

    if pareto_path.exists():
        df = pd.read_csv(pareto_path)
        df["model"] = model
        df = _add_derived_cols(df)
        data["pareto"] = df
    else:
        # Compute Pareto from sweep if pareto CSV is absent
        if data["sweep"] is not None:
            data["pareto"] = _compute_pareto(data["sweep"].copy())
        else:
            data["pareto"] = None

    return data


def _add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure safety and utility columns exist; fill from asr/over_refusal."""
    df = df.copy()
    if "safety" not in df.columns:
        df["safety"] = 1.0 - df["asr"].clip(0, 1)
    if "utility" not in df.columns:
        df["utility"] = 1.0 - df["over_refusal"].clip(0, 1)
    if "is_pareto_optimal" not in df.columns:
        df["is_pareto_optimal"] = False
    # Normalise param label for display
    df["param_str"] = df["param_value"].apply(
        lambda x: "—" if pd.isna(x) else str(x)
    )
    return df


def _compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Mark Pareto-optimal rows (maximise safety and utility simultaneously)."""
    agg = df[df["language"] == "all"].copy()
    dominated = []
    for i, row_i in agg.iterrows():
        for j, row_j in agg.iterrows():
            if i == j:
                continue
            if (row_j["safety"] >= row_i["safety"] and
                    row_j["utility"] >= row_i["utility"] and
                    (row_j["safety"] > row_i["safety"] or
                     row_j["utility"] > row_i["utility"])):
                dominated.append(i)
                break
    agg["is_pareto_optimal"] = ~agg.index.isin(dominated)
    return agg


def _best_param(df: pd.DataFrame, intervention: str) -> float | None:
    """Return the param value closest to the ideal point (safety=1, utility=1)."""
    sub = df[(df["intervention"] == intervention) &
             (df["language"] == "all") &
             (~df["param_value"].isna())]
    if sub.empty:
        return None
    dist = np.sqrt((1 - sub["safety"]) ** 2 + (1 - sub["utility"]) ** 2)
    return sub.loc[dist.idxmin(), "param_value"]


def _pareto_frontier_sorted(df: pd.DataFrame):
    """Extract Pareto-optimal rows sorted by utility for step-function plotting."""
    pts = df[df["is_pareto_optimal"] & (df["language"] == "all")].copy()
    if pts.empty:
        return pts
    return pts.sort_values("utility")



def _pick_src(data: dict):
    """Return pareto DataFrame if present, else sweep, else None."""
    p = data.get("pareto")
    if p is not None:
        return p
    return data.get("sweep")

# ── Figure 1: Per-model Pareto + combined overlay ─────────────────────────────

def plot_fig1(all_data: dict, output_path: Path):
    """
    Row 1: one Pareto scatter per model.
    Row 2: combined overlay of all models' Pareto frontiers.
    """
    models = list(all_data.keys())
    n = len(models)

    fig = plt.figure(figsize=(5 * n, 9))
    gs  = fig.add_gridspec(2, n, hspace=0.45, wspace=0.35)

    # ── Row 1: per-model scatter ──────────────────────────────────────────────
    for col, model in enumerate(models):
        ax = fig.add_subplot(gs[0, col])
        data = all_data[model]
        pareto = data.get("pareto")
        sweep  = data.get("sweep")
        src = pareto if pareto is not None else sweep
        if src is None:
            ax.set_visible(False)
            continue

        src_agg = src[src["language"] == "all"]

        for interv, grp in src_agg.groupby("intervention"):
            color  = INTERV_COLORS.get(interv, "#555555")
            marker = INTERV_MARKERS.get(interv, "o")
            label  = INTERV_LABELS.get(interv, interv)
            ax.scatter(grp["utility"], grp["safety"],
                       color=color, marker=marker, s=60,
                       alpha=0.85, label=label, zorder=3)

            # Annotate param values (skip baseline)
            if interv != "baseline":
                for _, r in grp.iterrows():
                    ax.annotate(r["param_str"],
                                (r["utility"], r["safety"]),
                                fontsize=6, color=color,
                                xytext=(4, 3), textcoords="offset points")

        # Draw Pareto step-function
        pf = _pareto_frontier_sorted(src_agg)
        if not pf.empty:
            ax.step(pf["utility"], pf["safety"],
                    where="post", color="black",
                    linewidth=1.5, linestyle="--",
                    label="Pareto frontier", zorder=4)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Utility  (1 − over-refusal)")
        ax.set_ylabel("Safety  (1 − ASR)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.axhline(0.5, color="lightgray", linewidth=0.7, linestyle=":")
        ax.axvline(0.5, color="lightgray", linewidth=0.7, linestyle=":")
        if col == 0:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.7)

    # ── Row 2: combined overlay ───────────────────────────────────────────────
    ax_combined = fig.add_subplot(gs[1, :])

    for model in models:
        data = all_data[model]
        src = _pick_src(data)
        if src is None:
            continue
        pf = _pareto_frontier_sorted(src[src["language"] == "all"])
        if pf.empty:
            continue
        color = MODEL_COLORS.get(model, "#333333")
        ax_combined.step(pf["utility"], pf["safety"],
                         where="post", color=color,
                         linewidth=2.2,
                         label=MODEL_LABELS.get(model, model))
        ax_combined.scatter(pf["utility"], pf["safety"],
                            color=color,
                            marker=MODEL_MARKERS.get(model, "o"),
                            s=55, zorder=4)

    ax_combined.set_xlim(-0.05, 1.05)
    ax_combined.set_ylim(-0.05, 1.05)
    ax_combined.set_xlabel("Utility  (1 − over-refusal)")
    ax_combined.set_ylabel("Safety  (1 − ASR)")
    ax_combined.set_title("Combined — All Models' Pareto Frontiers")
    ax_combined.axhline(0.5, color="lightgray", linewidth=0.7, linestyle=":")
    ax_combined.axvline(0.5, color="lightgray", linewidth=0.7, linestyle=":")
    ax_combined.legend(loc="lower left", framealpha=0.8)

    fig.suptitle("Fig 1 — Safety–Utility Pareto Analysis", fontsize=13, y=1.01)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 2: Parameter sweep curves ─────────────────────────────────────────

def plot_fig2(all_data: dict, output_path: Path):
    """
    Two metric rows (ASR, over-refusal) × one column per model.
    One line per non-baseline intervention. x = param_value.
    """
    models = list(all_data.keys())
    metrics = [("asr",          "ASR (↓ better)"),
               ("over_refusal", "Over-refusal rate (↓ better)")]
    n_models  = len(models)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_models,
                             figsize=(4.5 * n_models, 3.5 * n_metrics),
                             squeeze=False, sharey="row")

    all_interventions = sorted({
        interv
        for d in all_data.values()
        for src in [d.get("sweep"), d.get("pareto")]
        if src is not None
        for interv in src["intervention"].unique()
        if interv != "baseline"
    })

    for col, model in enumerate(models):
        data = all_data[model]
        src  = _pick_src(data)
        if src is None:
            for row in range(n_metrics):
                axes[row][col].set_visible(False)
            continue

        src_agg = src[(src["language"] == "all") &
                      (src["intervention"] != "baseline")]

        # Baseline reference lines
        baseline = src[(src["language"] == "all") &
                       (src["intervention"] == "baseline")]

        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][col]

            if not baseline.empty:
                bl_val = baseline[metric].mean()
                ax.axhline(bl_val, color=INTERV_COLORS["baseline"],
                           linewidth=1.2, linestyle="--",
                           label="Baseline" if row == 0 and col == 0 else "")

            for interv in all_interventions:
                grp = src_agg[src_agg["intervention"] == interv].sort_values("param_value")
                if grp.empty:
                    continue
                color = INTERV_COLORS.get(interv, "#555555")
                label = INTERV_LABELS.get(interv, interv) if row == 0 else None
                ax.plot(grp["param_value"], grp[metric],
                        color=color, linewidth=2,
                        marker="o", markersize=5,
                        label=label)
                # Shade std if available
                std_col = metric.replace("asr", "std_asr") \
                                .replace("over_refusal", "std_over_refusal")
                if std_col in grp.columns:
                    ax.fill_between(grp["param_value"],
                                    grp[metric] - grp[std_col],
                                    grp[metric] + grp[std_col],
                                    color=color, alpha=0.12)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Parameter value" if row == n_metrics - 1 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model, model))

    # Shared legend
    handles = ([mlines.Line2D([], [], color=INTERV_COLORS["baseline"],
                              linestyle="--", label="Baseline")] +
               [mlines.Line2D([], [], color=INTERV_COLORS.get(i, "#555"),
                              marker="o", label=INTERV_LABELS.get(i, i))
                for i in all_interventions])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle("Fig 2 — Parameter Sweep: ASR and Over-Refusal", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 3: Cross-model Pareto overlay ─────────────────────────────────────

def plot_fig3(all_data: dict, output_path: Path):
    """
    One panel per intervention type.
    Each panel overlays all models' ASR vs over-refusal sweep curves.
    """
    all_interventions = sorted({
        interv
        for d in all_data.values()
        for src in [d.get("sweep"), d.get("pareto")]
        if src is not None
        for interv in src["intervention"].unique()
        if interv != "baseline"
    })

    n = len(all_interventions)
    if n == 0:
        print("[SKIP Fig 3] No non-baseline interventions found.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, interv in zip(axes, all_interventions):
        for model in all_data:
            data = all_data[model]
            src  = _pick_src(data)
            if src is None:
                continue
            grp = src[(src["intervention"] == interv) &
                      (src["language"] == "all")].sort_values("param_value")
            if grp.empty:
                continue

            color  = MODEL_COLORS.get(model, "#333")
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(grp["utility"], grp["safety"],
                    color=color, linewidth=2,
                    marker=marker, markersize=6,
                    label=MODEL_LABELS.get(model, model))

            # Label param values along the curve
            for _, r in grp.iterrows():
                ax.annotate(r["param_str"],
                            (r["utility"], r["safety"]),
                            fontsize=6, color=color,
                            xytext=(3, 3), textcoords="offset points")

        # Baseline scatter per model
        for model in all_data:
            src = _pick_src(all_data[model])
            if src is None:
                continue
            bl = src[(src["intervention"] == "baseline") &
                     (src["language"] == "all")]
            if not bl.empty:
                ax.scatter(bl["utility"], bl["safety"],
                           color=MODEL_COLORS.get(model, "#333"),
                           marker="x", s=80, zorder=5, linewidths=1.5)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Utility  (1 − over-refusal)")
        ax.set_ylabel("Safety  (1 − ASR)" if interv == all_interventions[0] else "")
        ax.set_title(INTERV_LABELS.get(interv, interv))
        ax.axhline(0.5, color="lightgray", linewidth=0.7, linestyle=":")
        ax.axvline(0.5, color="lightgray", linewidth=0.7, linestyle=":")

    handles = [mlines.Line2D([], [], color=MODEL_COLORS[m],
                             marker=MODEL_MARKERS[m],
                             label=MODEL_LABELS[m])
               for m in all_data if m in MODEL_COLORS]
    handles += [mlines.Line2D([], [], color="gray", marker="x",
                              linestyle="None", markersize=7,
                              label="Baseline")]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.08),
               frameon=False)

    fig.suptitle("Fig 3 — Cross-Model Safety–Utility Trajectories per Intervention",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 4: Best-parameter comparison bar chart ────────────────────────────

def plot_fig4(all_data: dict, output_path: Path):
    """
    For each intervention, select each model's optimal param (closest to ideal).
    Side-by-side bars: ASR (left) and over-refusal (right) per model.
    Horizontal dashed lines show per-model baseline.
    """
    all_interventions = sorted({
        interv
        for d in all_data.values()
        for src in [d.get("sweep"), d.get("pareto")]
        if src is not None
        for interv in src["intervention"].unique()
        if interv != "baseline"
    })
    models = list(all_data.keys())
    metrics = [("asr", "ASR (↓ better)"),
               ("over_refusal", "Over-refusal (↓ better)")]

    n_interv  = len(all_interventions)
    n_metrics = len(metrics)
    if n_interv == 0:
        print("[SKIP Fig 4] No non-baseline interventions found.")
        return

    fig, axes = plt.subplots(n_metrics, n_interv,
                             figsize=(4.5 * n_interv, 3.5 * n_metrics),
                             squeeze=False, sharey="row")

    x = np.arange(len(models))
    width = 0.55

    for col, interv in enumerate(all_interventions):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            bar_vals, bar_errs = [], []
            bl_vals = []

            for model in models:
                data = all_data[model]
                src  = _pick_src(data)
                if src is None:
                    bar_vals.append(np.nan)
                    bar_errs.append(0)
                    bl_vals.append(np.nan)
                    continue

                src_agg = src[src["language"] == "all"]
                best_p  = _best_param(src_agg, interv)
                if best_p is None:
                    bar_vals.append(np.nan)
                    bar_errs.append(0)
                else:
                    row_data = src_agg[(src_agg["intervention"] == interv) &
                                       (src_agg["param_value"] == best_p)]
                    bar_vals.append(row_data[metric].mean() if not row_data.empty
                                    else np.nan)
                    std_col = f"std_{metric}"
                    bar_errs.append(row_data[std_col].mean()
                                    if std_col in row_data.columns and not row_data.empty
                                    else 0)

                bl = src_agg[src_agg["intervention"] == "baseline"]
                bl_vals.append(bl[metric].mean() if not bl.empty else np.nan)

            colors = [MODEL_COLORS.get(m, "#AAA") for m in models]
            bars = ax.bar(x, bar_vals, width=width,
                          color=colors, yerr=bar_errs,
                          error_kw={"elinewidth": 1.2, "capsize": 4},
                          zorder=3)

            # Baseline reference per model
            for i, (bl, color) in enumerate(zip(bl_vals, colors)):
                if not np.isnan(bl):
                    ax.hlines(bl, i - width / 2, i + width / 2,
                              colors=color, linewidths=1.8,
                              linestyles="--", zorder=4)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                               rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(INTERV_LABELS.get(interv, interv))
            ax.axhline(0, color="black", linewidth=0.7)

    # Legend: model colours + baseline marker
    model_patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#AAA"),
                                    label=MODEL_LABELS.get(m, m))
                     for m in models]
    baseline_line = mlines.Line2D([], [], color="gray", linestyle="--",
                                  linewidth=1.8, label="Model baseline")
    fig.legend(handles=model_patches + [baseline_line],
               loc="lower center", ncol=len(models) + 1,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    fig.suptitle("Fig 4 — Best-Parameter Performance per Intervention × Model",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 5: Per-tier ASR breakdown ─────────────────────────────────────────

def plot_fig5(all_data: dict, output_path: Path):
    """
    For each model, show ASR by language tier at the best param per intervention.
    Falls back gracefully if only aggregate ('all') tier data is available.
    """
    tier_order = ["tier_1", "tier_2", "tier_3", "tier_4"]
    models     = list(all_data.keys())

    all_interventions = sorted({
        interv
        for d in all_data.values()
        for src in [d.get("sweep"), d.get("pareto")]
        if src is not None
        for interv in src["intervention"].unique()
        if interv != "baseline"
    })

    # Check whether any model has real tier data
    has_tier_data = any(
        (all_data[m].get("sweep") is not None and
         set(all_data[m]["sweep"]["tier"].unique()) - {"all"})
        for m in models
    )

    if not has_tier_data:
        print("[INFO] No per-tier data found (all rows have tier='all'). "
              "Fig 5 will show per-tier placeholder — re-run after collecting "
              "language-level sweep results.")
        # Produce a clear informational placeholder rather than an empty file
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5,
                "Per-tier breakdown not yet available.\n"
                "Re-run the intervention script with per-language evaluation\n"
                "to populate this figure.",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes, color="gray",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                          edgecolor="lightgray"))
        ax.axis("off")
        fig.suptitle("Fig 5 — Per-Tier ASR at Best Parameter (data pending)",
                     fontsize=12)
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
        return

    n_interv = len(all_interventions)
    n_models  = len(models)
    x = np.arange(len(tier_order))
    width = 0.2

    fig, axes = plt.subplots(n_models, n_interv,
                             figsize=(4.5 * n_interv, 3.5 * n_models),
                             squeeze=False, sharey=True)

    for row, model in enumerate(models):
        data = all_data[model]
        src  = _pick_src(data)

        for col, interv in enumerate(all_interventions):
            ax = axes[row][col]
            if src is None:
                ax.set_visible(False)
                continue

            best_p = _best_param(src[src["language"] == "all"], interv)
            sub = src[(src["intervention"] == interv) &
                      (src["tier"] != "all")]
            if best_p is not None:
                sub = sub[sub["param_value"] == best_p]

            # Baseline tier values
            bl = src[(src["intervention"] == "baseline") &
                     (src["tier"] != "all")]

            # Plot bars per tier
            for ti, tier in enumerate(tier_order):
                tier_row = sub[sub["tier"] == tier]
                asr_val  = tier_row["asr"].mean() if not tier_row.empty else np.nan
                ax.bar(ti, asr_val, color=TIER_COLORS.get(tier, "#AAA"),
                       width=0.6, zorder=3,
                       label=TIER_LABELS.get(tier, tier) if col == 0 and row == 0 else "")

                bl_row = bl[bl["tier"] == tier]
                if not bl_row.empty:
                    bl_val = bl_row["asr"].mean()
                    ax.hlines(bl_val, ti - 0.3, ti + 0.3,
                              colors=TIER_COLORS.get(tier, "#AAA"),
                              linewidths=1.5, linestyles="--", zorder=4)

            ax.set_xticks(range(len(tier_order)))
            ax.set_xticklabels([TIER_LABELS.get(t, t) for t in tier_order],
                               rotation=25, ha="right", fontsize=7)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("ASR" if col == 0 else "")
            ax.axhline(0, color="black", linewidth=0.7)
            if row == 0:
                ax.set_title(INTERV_LABELS.get(interv, interv))
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS.get(model, model)}\nASR", fontsize=8)

    tier_patches = [mpatches.Patch(color=TIER_COLORS[t],
                                   label=TIER_LABELS[t]) for t in tier_order]
    bl_line = mlines.Line2D([], [], color="gray", linestyle="--",
                             linewidth=1.5, label="Baseline (dashed)")
    fig.legend(handles=tier_patches + [bl_line],
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle("Fig 5 — Per-Tier ASR at Best Parameter per Intervention × Model",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise intervention sweep results across Aya, Llama, and Qwen."
    )
    parser.add_argument("--data-dir",   default=None,
                        help="Root dir containing aya/, llama/, qwen/ subdirectories.")
    parser.add_argument("--aya-dir",    default=None)
    parser.add_argument("--llama-dir",  default=None)
    parser.add_argument("--qwen-dir",   default=None)
    parser.add_argument("--output-dir", default="figures/")
    parser.add_argument("--figures",    nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which figures to produce e.g. --figures 1 3 4")
    return parser.parse_args()


def resolve_model_dirs(args) -> dict:
    dirs = {}
    overrides = {"aya": args.aya_dir, "llama": args.llama_dir, "qwen": args.qwen_dir}
    if args.data_dir:
        root = Path(args.data_dir)
        for model in ["aya", "llama", "qwen"]:
            candidate = root / model
            if candidate.is_dir():
                dirs[model] = candidate
    for model, path_str in overrides.items():
        if path_str is not None:
            dirs[model] = Path(path_str)
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

    if not all_data:
        print("ERROR: no valid data loaded.")
        sys.exit(1)

    figs = set(args.figures)

    if 1 in figs:
        plot_fig1(all_data, outdir / "fig1_pareto_per_model.pdf")
    if 2 in figs:
        plot_fig2(all_data, outdir / "fig2_param_sweep_curves.pdf")
    if 3 in figs:
        plot_fig3(all_data, outdir / "fig3_pareto_overlay.pdf")
    if 4 in figs:
        plot_fig4(all_data, outdir / "fig4_best_param_comparison.pdf")
    if 5 in figs:
        plot_fig5(all_data, outdir / "fig5_tier_breakdown.pdf")

    print(f"\nDone. All figures written to {outdir}/")


if __name__ == "__main__":
    main()
