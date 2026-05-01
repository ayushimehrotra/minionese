"""
visualize_attribution.py

Produces four figures from attribution patching results across Aya, Llama, and Qwen.

Expected directory structure (one folder per model, each containing four CSVs):
    data/
        aya/
            attribution_a_results.csv
            attribution_a_by_tier.csv
            attribution_b_results.csv
            attribution_b_by_tier.csv
        llama/
            attribution_a_results.csv
            ...
        qwen/
            attribution_a_results.csv
            ...

Usage:
    python visualize_attribution.py --data-dir data/ --output-dir figures/

    # Override individual model paths:
    python visualize_attribution.py \
        --aya-dir   results/attribution/aya/ \
        --llama-dir results/attribution/llama/ \
        --qwen-dir  results/attribution/qwen/ \
        --output-dir figures/

Figures produced:
    fig1_critical_layer_profiles.pdf   -- Cross-model restoration vs. normalized layer depth
    fig2_tier_layer_heatmaps.pdf       -- Tier x layer restoration heatmaps (3 models x 3 perturbations)
    fig3_component_dissection.pdf      -- attn_out vs mlp_out vs residual at critical layers
    fig4_universality_scatter.pdf      -- Per-language restoration: Aya vs Llama / Aya vs Qwen
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# ── Aesthetics ────────────────────────────────────────────────────────────────

MODEL_COLORS   = {"aya": "#E07B39", "llama": "#4A90D9", "qwen": "#5DBB8A"}
MODEL_LABELS   = {"aya": "Aya-Expanse-8B", "llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}
TIER_COLORS    = {"tier_1": "#2166AC", "tier_2": "#74ADD1", "tier_3": "#F4A582", "tier_4": "#D6604D"}
TIER_LABELS    = {"tier_1": "Tier 1 (High)", "tier_2": "Tier 2", "tier_3": "Tier 3", "tier_4": "Tier 4 (Low)"}
COMP_COLORS    = {"residual": "#333333", "attn_out": "#9B59B6", "mlp_out": "#E67E22"}
COMP_LABELS    = {"residual": "Residual", "attn_out": "Attention Out", "mlp_out": "MLP Out"}
PERT_LABELS    = {
    "standard_translation": "Standard Translation",
    "transliteration":      "Transliteration",
    "code_switching":       "Code-Switching",
    "translationese":       "Translationese",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_model_data(model_dir: Path, model: str) -> dict:
    """Load all four CSVs for one model. Returns dict with keys a_results,
    a_by_tier, b_results, b_by_tier."""
    files = {
        "a_results":  "attribution_a_results.csv",
        "a_by_tier":  "attribution_a_by_tier.csv",
        "b_results":  "attribution_b_results.csv",
        "b_by_tier":  "attribution_b_by_tier.csv",
    }
    data = {}
    for key, fname in files.items():
        path = model_dir / fname
        if not path.exists():
            print(f"[WARN] Missing {path} — skipping model '{model}' for plots that need {key}.")
            data[key] = None
        else:
            df = pd.read_csv(path)
            df["model"] = model
            data[key] = df
    return data


def normalize_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'layer_norm' column in [0, 1] based on the max layer in that df."""
    df = df.copy()
    max_layer = df["layer"].max()
    df["layer_norm"] = df["layer"] / max_layer
    return df


def identify_critical_layers(b_results: pd.DataFrame, component: str = "residual",
                              top_k: int = 3) -> dict:
    """Return {perturbation: [critical_layer, ...]} using peak mean_restoration
    in the residual stream, averaged across tiers and languages."""
    sub = b_results[b_results["component"] == component]
    grouped = (
        sub.groupby(["perturbation", "layer"])["mean_restoration"]
        .mean()
        .reset_index()
    )
    critical = {}
    for pert, grp in grouped.groupby("perturbation"):
        top = grp.nlargest(top_k, "mean_restoration")["layer"].tolist()
        critical[pert] = sorted(top)
    return critical


# ── Figure 1: Cross-model critical layer profiles ────────────────────────────

def plot_fig1(all_data: dict, perturbations: list, output_path: Path):
    """
    For each perturbation, plot mean_restoration (residual) vs normalised layer
    depth, one line per model, averaged across all tiers.
    """
    n_perts = len(perturbations)
    fig, axes = plt.subplots(1, n_perts, figsize=(5 * n_perts, 4), sharey=False)
    if n_perts == 1:
        axes = [axes]

    for ax, pert in zip(axes, perturbations):
        for model, data in all_data.items():
            b = data.get("b_results")
            if b is None:
                continue
            sub = b[(b["perturbation"] == pert) & (b["component"] == "residual")]
            if sub.empty:
                continue
            sub = normalize_layers(sub)
            agg = (
                sub.groupby("layer_norm")
                .agg(mean=("mean_restoration", "mean"), sem=("mean_restoration", "sem"))
                .reset_index()
            )
            ax.plot(
                agg["layer_norm"], agg["mean"],
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                linewidth=2, marker="o", markersize=3,
            )
            ax.fill_between(
                agg["layer_norm"],
                agg["mean"] - agg["sem"],
                agg["mean"] + agg["sem"],
                color=MODEL_COLORS[model], alpha=0.15,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(PERT_LABELS.get(pert, pert))
        ax.set_xlabel("Normalised Layer Depth")
        ax.set_ylabel("Mean Restoration Score" if ax == axes[0] else "")
        ax.set_xlim(0, 1)

    handles = [
        Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2, label=MODEL_LABELS[m])
        for m in all_data
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(all_data),
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Cross-Model Refusal Restoration by Layer Depth (Residual Stream)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 2: Tier × layer heatmaps ──────────────────────────────────────────

def plot_fig2(all_data: dict, perturbations: list, output_path: Path):
    """
    3 models × n_perturbations grid of heatmaps.
    Each cell: tier (y) × layer (x), colour = mean_restoration.
    Uses by_tier data for cleaner signal.
    """
    models = list(all_data.keys())
    n_models = len(models)
    n_perts  = len(perturbations)

    tier_order = ["tier_1", "tier_2", "tier_3", "tier_4"]
    tier_tick_labels = [TIER_LABELS[t] for t in tier_order]

    fig, axes = plt.subplots(
        n_models, n_perts,
        figsize=(4.5 * n_perts, 3.2 * n_models),
        squeeze=False,
    )

    vmin, vmax = -0.5, 1.0          # shared colour scale across all panels

    for row, model in enumerate(models):
        bt = all_data[model].get("b_by_tier")
        for col, pert in enumerate(perturbations):
            ax = axes[row][col]
            if bt is None:
                ax.set_visible(False)
                continue

            sub = bt[(bt["perturbation"] == pert) & (bt["component"] == "residual")]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Pivot: rows = tier, cols = layer
            pivot = sub.pivot_table(
                index="tier", columns="layer", values="mean_restoration", aggfunc="mean"
            ).reindex(tier_order)

            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdYlGn",
                vmin=vmin, vmax=vmax,
                linewidths=0,
                cbar=(col == n_perts - 1),          # only rightmost column gets colourbar
                cbar_kws={"shrink": 0.8, "label": "Restoration"},
                yticklabels=tier_tick_labels if col == 0 else False,
                xticklabels=False,
            )

            # Tick every 4 layers
            n_layers = pivot.shape[1]
            tick_positions = list(range(0, n_layers, 4))
            ax.set_xticks([p + 0.5 for p in tick_positions])
            ax.set_xticklabels([str(p) for p in tick_positions], fontsize=7)

            if row == 0:
                ax.set_title(PERT_LABELS.get(pert, pert), fontsize=10)
            if col == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Layer" if row == n_models - 1 else "")

    fig.suptitle("Tier × Layer Restoration Heatmaps (Residual Stream)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 3: Component dissection at critical layers ─────────────────────────

def plot_fig3(all_data: dict, perturbations: list, output_path: Path):
    """
    For each perturbation × model, bar chart of mean_delta (Analysis A)
    for each component (residual, attn_out, mlp_out), evaluated at the
    critical layers identified per model.
    """
    components = ["residual", "attn_out", "mlp_out"]
    n_perts   = len(perturbations)
    models    = list(all_data.keys())
    n_models  = len(models)

    fig, axes = plt.subplots(
        n_models, n_perts,
        figsize=(4.5 * n_perts, 3 * n_models),
        squeeze=False,
        sharey="row",
    )

    for row, model in enumerate(models):
        a_res = all_data[model].get("a_results")
        b_res = all_data[model].get("b_results")
        critical = {}
        if b_res is not None:
            critical = identify_critical_layers(b_res, component="residual", top_k=3)

        for col, pert in enumerate(perturbations):
            ax = axes[row][col]
            if a_res is None:
                ax.set_visible(False)
                continue

            crit_layers = critical.get(pert, None)
            sub = a_res[a_res["perturbation"] == pert]
            if crit_layers is not None:
                sub = sub[sub["layer"].isin(crit_layers)]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Average over critical layers, languages, tiers
            comp_means = (
                sub.groupby("component")["mean_delta"]
                .mean()
                .reindex(components)
            )
            comp_sems = (
                sub.groupby("component")["mean_delta"]
                .sem()
                .reindex(components)
            )

            x = np.arange(len(components))
            bars = ax.bar(
                x,
                comp_means.values,
                yerr=comp_sems.values,
                color=[COMP_COLORS[c] for c in components],
                width=0.55,
                error_kw={"elinewidth": 1.2, "capsize": 4},
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [COMP_LABELS[c] for c in components],
                rotation=25, ha="right", fontsize=8,
            )

            label_str = (f"layers {crit_layers}" if crit_layers
                         else "all layers")
            if row == 0:
                ax.set_title(PERT_LABELS.get(pert, pert), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nMean Δ (refusal proj.)", fontsize=8)
            ax.set_xlabel(label_str, fontsize=7)

    fig.suptitle("Component Dissection at Critical Layers (Analysis A)",
                 fontsize=12, y=1.01)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMP_COLORS[c],
                              label=COMP_LABELS[c]) for c in components]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 4: Cross-model universality scatter ────────────────────────────────

def plot_fig4(all_data: dict, perturbations: list, output_path: Path):
    """
    Per-language mean_restoration at the critical layer (residual).
    Scatter: Aya (x) vs Llama/Qwen (y), coloured by tier.
    One row per comparison pair, one column per perturbation.
    """
    anchor = "aya"
    comparators = [m for m in all_data if m != anchor]
    if anchor not in all_data or not comparators:
        print("[SKIP Fig 4] Need Aya plus at least one other model.")
        return

    n_comp  = len(comparators)
    n_perts = len(perturbations)

    fig, axes = plt.subplots(
        n_comp, n_perts,
        figsize=(4 * n_perts, 4 * n_comp),
        squeeze=False,
    )

    for row, comp_model in enumerate(comparators):
        b_anchor = all_data[anchor].get("b_results")
        b_comp   = all_data[comp_model].get("b_results")
        if b_anchor is None or b_comp is None:
            for col in range(n_perts):
                axes[row][col].set_visible(False)
            continue

        # Find critical layers per model per perturbation
        crit_anchor = identify_critical_layers(b_anchor, "residual", top_k=3)
        crit_comp   = identify_critical_layers(b_comp,   "residual", top_k=3)

        for col, pert in enumerate(perturbations):
            ax = axes[row][col]

            def lang_restoration(b_df, crit_dict):
                layers = crit_dict.get(pert, None)
                sub = b_df[
                    (b_df["perturbation"] == pert) &
                    (b_df["component"]    == "residual")
                ]
                if layers:
                    sub = sub[sub["layer"].isin(layers)]
                return (
                    sub.groupby(["language", "tier"])["mean_restoration"]
                    .mean()
                    .reset_index()
                )

            df_a = lang_restoration(b_anchor, crit_anchor).rename(columns={"mean_restoration": "restoration_aya"})
            df_c = lang_restoration(b_comp,   crit_comp  ).rename(columns={"mean_restoration": f"restoration_{comp_model}"})

            merged = pd.merge(df_a, df_c, on=["language", "tier"], how="inner")
            if merged.empty:
                ax.text(0.5, 0.5, "No overlap", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            for tier, grp in merged.groupby("tier"):
                ax.scatter(
                    grp["restoration_aya"],
                    grp[f"restoration_{comp_model}"],
                    color=TIER_COLORS.get(tier, "gray"),
                    label=TIER_LABELS.get(tier, tier),
                    s=55, alpha=0.85, zorder=3,
                )
                # Annotate language codes
                for _, r in grp.iterrows():
                    ax.annotate(
                        r["language"],
                        (r["restoration_aya"], r[f"restoration_{comp_model}"]),
                        fontsize=6.5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points",
                    )

            # Diagonal reference
            all_vals = pd.concat([merged["restoration_aya"],
                                   merged[f"restoration_{comp_model}"]])
            lo, hi = all_vals.min(), all_vals.max()
            pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color="gray", linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)

            if row == 0:
                ax.set_title(PERT_LABELS.get(pert, pert), fontsize=10)
            ax.set_xlabel(f"{MODEL_LABELS[anchor]} restoration", fontsize=8)
            ax.set_ylabel(f"{MODEL_LABELS[comp_model]} restoration", fontsize=8)

            # Pearson r annotation
            r_val = merged["restoration_aya"].corr(merged[f"restoration_{comp_model}"])
            ax.text(0.05, 0.93, f"r = {r_val:.2f}", transform=ax.transAxes,
                    fontsize=8, color="dimgray")

    # Shared tier legend
    handles = [
        plt.scatter([], [], color=TIER_COLORS[t], s=50, label=TIER_LABELS[t])
        for t in ["tier_1", "tier_2", "tier_3", "tier_4"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False, title="Language Tier")
    fig.suptitle("Cross-Model Universality: Per-Language Restoration at Critical Layers",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise attribution patching results across Aya, Llama, and Qwen."
    )
    # Convenience: single root dir with model subdirectories
    parser.add_argument("--data-dir", default=None,
                        help="Root dir containing aya/, llama/, qwen/ subdirectories.")
    # Per-model overrides
    parser.add_argument("--aya-dir",   default=None)
    parser.add_argument("--llama-dir", default=None)
    parser.add_argument("--qwen-dir",  default=None)

    parser.add_argument("--output-dir", default="figures/",
                        help="Directory to write figures into.")
    parser.add_argument("--perturbations", nargs="+",
                        default=["standard_translation", "transliteration", "code_switching"],
                        help="Perturbation types to include (must match CSV values).")
    parser.add_argument("--figures", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which figures to produce, e.g. --figures 1 3")
    return parser.parse_args()


def resolve_model_dirs(args) -> dict:
    """Return {model_name: Path} for each model that has a directory specified."""
    dirs = {}
    name_map = {"aya": args.aya_dir, "llama": args.llama_dir, "qwen": args.qwen_dir}

    if args.data_dir:
        root = Path(args.data_dir)
        for model in ["aya", "llama", "qwen"]:
            candidate = root / model
            if candidate.is_dir():
                dirs[model] = candidate

    # Per-model overrides take precedence
    for model, path_str in name_map.items():
        if path_str is not None:
            dirs[model] = Path(path_str)

    return dirs


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dirs = resolve_model_dirs(args)
    if not model_dirs:
        print("ERROR: No model directories found. Use --data-dir or --aya-dir / "
              "--llama-dir / --qwen-dir.")
        sys.exit(1)

    print(f"Loading data for models: {list(model_dirs.keys())}")
    all_data = {model: load_model_data(path, model)
                for model, path in model_dirs.items()}

    perts = args.perturbations
    figs  = set(args.figures)

    if 1 in figs:
        plot_fig1(all_data, perts,
                  outdir / "fig1_critical_layer_profiles.pdf")
    if 2 in figs:
        plot_fig2(all_data, perts,
                  outdir / "fig2_tier_layer_heatmaps.pdf")
    if 3 in figs:
        plot_fig3(all_data, perts,
                  outdir / "fig3_component_dissection.pdf")
    if 4 in figs:
        plot_fig4(all_data, perts,
                  outdir / "fig4_universality_scatter.pdf")

    print(f"\nDone. All figures written to {outdir}/")


if __name__ == "__main__":
    main()
