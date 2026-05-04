#!/usr/bin/env python3
"""
Visualize Step 4b cross-lingual refusal cone outputs.

The script reads the artifacts produced by scripts/04b_refusal_cones.py:

  - refusal_cone_{model}.npy
  - refusal_cone_metrics.csv
  - refusal_cone_per_language.csv
  - refusal_cone_repind.csv
  - refusal_cone_summary.json

It produces:

  - fig_refusal_cone_metrics_{model}.{pdf,png}
  - fig_refusal_cone_language_margins_{model}.{pdf,png}
  - fig_refusal_cone_sections_{model}.{pdf,png}
  - fig_refusal_cone_activation_sections_{model}.{pdf,png}  (optional)

The "sections" figure shows 2D slices through the positive cone: for each pair
of basis directions, language margin vectors are plotted in that 2D coordinate
system. Points in the shaded first quadrant are directions where both basis
directions have positive harmful-vs-harmless margin.

Usage:
    python data_viz_scripts/visualize_refusal_cones.py \
        --representation-dir results_aya_new/representation/aya \
        --output-dir results_aya_new/figures

Optional activation-space sections:
    python data_viz_scripts/visualize_refusal_cones.py \
        --representation-dir results_aya_new/representation/aya \
        --activations-dir data/activations \
        --dataset-dir dataset \
        --output-dir results_aya_new/figures
"""

import argparse
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TIER_MAP = {
    "en": "tier_1", "de": "tier_1", "fr": "tier_1", "zh": "tier_1", "es": "tier_1",
    "ar": "tier_2", "ru": "tier_2", "ko": "tier_2", "ja": "tier_2",
    "tr": "tier_3", "id": "tier_3", "hi": "tier_3", "sw": "tier_3",
    "yo": "tier_4", "zu": "tier_4", "gd": "tier_4", "gn": "tier_4", "jw": "tier_4",
}
TIER_ORDER = ["tier_1", "tier_2", "tier_3", "tier_4"]
TIER_LABELS = {
    "tier_1": "Tier 1",
    "tier_2": "Tier 2",
    "tier_3": "Tier 3",
    "tier_4": "Tier 4",
}
TIER_COLORS = {
    "tier_1": "#2166AC",
    "tier_2": "#4393C3",
    "tier_3": "#F4A582",
    "tier_4": "#D6604D",
}
LANG_ORDER = [
    "en", "de", "es", "fr", "zh",
    "ar", "ja", "ko", "ru",
    "hi", "id", "sw", "tr",
    "gd", "gn", "yo", "zu",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Step 4b refusal cone outputs.")
    parser.add_argument("--representation-dir", required=True,
                        help="Directory containing refusal_cone_* outputs for one model.")
    parser.add_argument("--output-dir", default="figures/",
                        help="Directory where figures are written.")
    parser.add_argument("--model", default=None,
                        help="Model key. Defaults to summary JSON or refusal_cone filename.")
    parser.add_argument("--activations-dir", default=None,
                        help="Optional cached activations dir for activation-space sections.")
    parser.add_argument("--dataset-dir", default="dataset/",
                        help="Dataset dir used only with --activations-dir.")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Languages to use in activation-space sections.")
    parser.add_argument("--max-samples", type=int, default=160,
                        help="Max harmful/harmless samples per language in activation scatter.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"],
                        choices=["pdf", "png", "svg"],
                        help="Figure formats to write.")
    return parser.parse_args()


def save_figure(fig, output_dir: Path, stem: str, formats: list[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def resolve_artifact(repr_dir: Path, filename: str) -> Path | None:
    path = repr_dir / filename
    if path.exists():
        return path
    zipped = repr_dir / f"{filename}.zip"
    if zipped.exists():
        return zipped
    return None


def read_text_artifact(path: Path) -> str:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = useful_zip_names(zf)
            if not names:
                raise ValueError(f"Zip has no files: {path}")
            with zf.open(names[0]) as f:
                return f.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def read_csv_artifact(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [name for name in useful_zip_names(zf) if name.endswith(".csv")]
            if not names:
                names = useful_zip_names(zf)
            if not names:
                raise ValueError(f"Zip has no CSV files: {path}")
            with zf.open(names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def read_npy_artifact(path: Path) -> np.ndarray:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [name for name in useful_zip_names(zf) if name.endswith(".npy")]
            if not names:
                names = useful_zip_names(zf)
            if not names:
                raise ValueError(f"Zip has no files: {path}")
            data = zf.read(names[0])
            return np.load(io.BytesIO(data))
    return np.load(str(path))


def useful_zip_names(zf: zipfile.ZipFile) -> list[str]:
    names = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        parts = Path(name).parts
        if "__MACOSX" in parts:
            continue
        if Path(name).name.startswith("._"):
            continue
        names.append(name)
    return names


def infer_model(repr_dir: Path, summary: dict | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if summary and summary.get("model"):
        return str(summary["model"])
    for path in repr_dir.glob("refusal_cone_*.npy"):
        name = path.stem.removeprefix("refusal_cone_")
        if name:
            return name
    for path in repr_dir.glob("refusal_cone_*.npy.zip"):
        name = path.name.removeprefix("refusal_cone_").removesuffix(".npy.zip")
        if name:
            return name
    raise SystemExit("Could not infer model. Pass --model.")


def load_cone_outputs(repr_dir: Path, model: str | None):
    summary_path = resolve_artifact(repr_dir, "refusal_cone_summary.json")
    summary = json.loads(read_text_artifact(summary_path)) if summary_path else None
    model = infer_model(repr_dir, summary, model)

    basis_path = resolve_artifact(repr_dir, f"refusal_cone_{model}.npy")
    if not basis_path:
        candidates = sorted(repr_dir.glob("refusal_cone_*.npy"))
        candidates += sorted(repr_dir.glob("refusal_cone_*.npy.zip"))
        if not candidates:
            raise SystemExit(f"No refusal_cone_*.npy found in {repr_dir}")
        basis_path = candidates[0]
        if basis_path.name.endswith(".npy.zip"):
            model = basis_path.name.removeprefix("refusal_cone_").removesuffix(".npy.zip")
        else:
            model = basis_path.stem.removeprefix("refusal_cone_")

    required = {
        "metrics": resolve_artifact(repr_dir, "refusal_cone_metrics.csv"),
        "per_language": resolve_artifact(repr_dir, "refusal_cone_per_language.csv"),
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise SystemExit("Missing required Step 4b outputs:\n  " + "\n  ".join(missing))

    data = {
        "model": model,
        "basis": read_npy_artifact(basis_path),
        "metrics": read_csv_artifact(required["metrics"]),
        "per_language": read_csv_artifact(required["per_language"]),
        "summary": summary or {},
    }

    repind_path = resolve_artifact(repr_dir, "refusal_cone_repind.csv")
    data["repind"] = read_csv_artifact(repind_path) if repind_path else None
    return data


def prepare_representation_dir(path: Path, model: str | None):
    """Return a directory containing Step 4b artifacts, extracting zip inputs."""
    if path.suffix != ".zip":
        return path, None

    tmp = tempfile.TemporaryDirectory(prefix="refusal_cone_repr_")
    tmp_path = Path(tmp.name)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp_path)

    candidates = []
    for directory in [tmp_path, *[p for p in tmp_path.rglob("*") if p.is_dir()]]:
        has_cone = any(directory.glob("refusal_cone_*.npy")) or any(directory.glob("refusal_cone_*.npy.zip"))
        if has_cone:
            candidates.append(directory)

    if not candidates:
        raise SystemExit(f"No refusal_cone artifacts found inside {path}")

    if model:
        model_matches = [p for p in candidates if p.name == model]
        if model_matches:
            return model_matches[0], tmp

    # Prefer the deepest candidate; top-level archives often contain
    # representation_new/{model}/...
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    return candidates[0], tmp


def canonical_dim(metrics: pd.DataFrame, basis: np.ndarray) -> int:
    if "cone_dim" in metrics.columns and not metrics.empty:
        return int(metrics["cone_dim"].max())
    return int(basis.shape[1])


def ordered_languages(langs: list[str]) -> list[str]:
    present = set(langs)
    ordered = [lang for lang in LANG_ORDER if lang in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def add_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["language"].map(TIER_MAP).fillna("tier_4")
    return df


def canonical_margin_frame(data: dict) -> pd.DataFrame:
    metrics = data["metrics"]
    per_lang = data["per_language"].copy()
    dim = canonical_dim(metrics, data["basis"])
    per_lang = per_lang[per_lang["cone_dim"] == dim].copy()

    en_rows = []
    canon_metrics = metrics[metrics["cone_dim"] == dim]
    for _, row in canon_metrics.iterrows():
        en_rows.append({
            "cone_dim": dim,
            "basis_idx": int(row["basis_idx"]),
            "language": "en",
            "margin": float(row["en_margin"]),
        })
    if en_rows:
        per_lang = pd.concat([pd.DataFrame(en_rows), per_lang], ignore_index=True)

    return add_tier(per_lang)


def plot_cone_metrics(data: dict, output_dir: Path, formats: list[str]):
    model = data["model"]
    metrics = data["metrics"].copy()
    dim = canonical_dim(metrics, data["basis"])
    canon = metrics[metrics["cone_dim"] == dim].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    cone_summary = (metrics.groupby("cone_dim")
                    .agg(en_margin=("en_margin", "mean"),
                         xling_margin=("xling_margin", "mean"),
                         min_cone_margin=("cone_min_margin_en", "mean"),
                         frac_positive=("cone_frac_positive_en", "mean"))
                    .reset_index())
    axes[0].plot(cone_summary["cone_dim"], cone_summary["en_margin"],
                 marker="o", label="English basis margin", color="#2166AC")
    axes[0].plot(cone_summary["cone_dim"], cone_summary["xling_margin"],
                 marker="s", label="Cross-lingual basis margin", color="#D6604D")
    axes[0].plot(cone_summary["cone_dim"], cone_summary["min_cone_margin"],
                 marker="^", label="Min sampled EN cone margin", color="#4D9221")
    axes[0].set_title("Cone Quality vs Dimensionality")
    axes[0].set_xlabel("Cone dimension")
    axes[0].set_ylabel("Mean margin")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False)

    x = np.arange(len(canon))
    width = 0.36
    axes[1].bar(x - width / 2, canon["en_margin"], width=width,
                color="#2166AC", alpha=0.85, label="English")
    axes[1].bar(x + width / 2, canon["xling_margin"], width=width,
                color="#D6604D", alpha=0.85, label="Cross-lingual")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title(f"Canonical Cone Basis Margins (dim={dim})")
    axes[1].set_xlabel("Basis direction")
    axes[1].set_ylabel("Harmful - harmless margin")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"b{int(i)}" for i in canon["basis_idx"]])
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False)

    fig.suptitle(f"Refusal Cone Metrics: {model}", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_refusal_cone_metrics_{model}", formats)


def plot_language_margins(data: dict, output_dir: Path, formats: list[str]):
    model = data["model"]
    margins = canonical_margin_frame(data)
    if margins.empty:
        return

    langs = ordered_languages(margins["language"].unique().tolist())
    pivot = (margins.pivot_table(index="language", columns="basis_idx", values="margin", aggfunc="mean")
             .reindex(langs))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="#eeeeee",
        cbar_kws={"label": "Harmful - harmless margin"},
    )
    ax.set_title(f"Per-Language Refusal Cone Margins: {model}")
    ax.set_xlabel("Basis direction")
    ax.set_ylabel("Language")
    ax.set_xticklabels([f"b{int(float(t.get_text()))}" for t in ax.get_xticklabels()], rotation=0)

    # Draw tier boundaries.
    tiers = [TIER_MAP.get(lang, "tier_4") for lang in langs]
    for i in range(1, len(langs)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i, color="black", linewidth=1.0, linestyle="--")

    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_refusal_cone_language_margins_{model}", formats)


def basis_pairs(n_basis: int) -> list[tuple[int, int]]:
    pairs = []
    for i in range(n_basis):
        for j in range(i + 1, n_basis):
            pairs.append((i, j))
    return pairs[:6]


def plot_margin_sections(data: dict, output_dir: Path, formats: list[str]):
    model = data["model"]
    margins = canonical_margin_frame(data)
    dim = canonical_dim(data["metrics"], data["basis"])
    pairs = basis_pairs(min(dim, data["basis"].shape[1]))
    if not pairs:
        return

    wide = margins.pivot_table(index=["language", "tier"], columns="basis_idx", values="margin", aggfunc="mean")
    wide = wide.reset_index()

    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (i, j) in zip(axes_flat, pairs):
        if i not in wide.columns or j not in wide.columns:
            ax.set_visible(False)
            continue
        x = wide[i].to_numpy(dtype=float)
        y = wide[j].to_numpy(dtype=float)
        xmin, xmax = min(x.min(), 0.0), max(x.max(), 0.0)
        ymin, ymax = min(y.min(), 0.0), max(y.max(), 0.0)
        pad_x = 0.08 * (xmax - xmin + 1e-6)
        pad_y = 0.08 * (ymax - ymin + 1e-6)

        # Shaded positive quadrant: the 2D cross-section of positive cone coords.
        ax.axvspan(0, xmax + pad_x, ymin=0, ymax=1, color="#F5E6A1", alpha=0.18)
        ax.axhspan(0, ymax + pad_y, xmin=0, xmax=1, color="#F5E6A1", alpha=0.18)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.axvline(0, color="#333333", linewidth=0.8)

        for tier in TIER_ORDER:
            sub = wide[wide["tier"] == tier]
            if sub.empty:
                continue
            ax.scatter(sub[i], sub[j], s=42, label=TIER_LABELS[tier],
                       color=TIER_COLORS[tier], edgecolor="white", linewidth=0.5, alpha=0.9)

        for _, row in wide.iterrows():
            ax.text(row[i], row[j], str(row["language"]), fontsize=7, ha="left", va="bottom")

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_title(f"Cone Section: b{i} vs b{j}")
        ax.set_xlabel(f"Margin on b{i}")
        ax.set_ylabel(f"Margin on b{j}")
        ax.grid(linestyle="--", alpha=0.25)

    for ax in axes_flat[len(pairs):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"2D Cross-Sections of the Refusal Cone: {model}", y=1.08)
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_refusal_cone_sections_{model}", formats)


def plot_repind(data: dict, output_dir: Path, formats: list[str]):
    repind = data.get("repind")
    if repind is None or repind.empty:
        print("Skipping RepInd heatmap: refusal_cone_repind.csv missing or empty.")
        return

    model = data["model"]
    n = int(max(repind["i"].max(), repind["j"].max())) + 1
    M = np.eye(n)
    for _, row in repind.iterrows():
        i, j = int(row["i"]), int(row["j"])
        M[i, j] = float(row["cl_repind"])

    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    sns.heatmap(
        M,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        square=True,
        cbar_kws={"label": "CL-RepInd"},
        xticklabels=[f"b{i}" for i in range(n)],
        yticklabels=[f"b{i}" for i in range(n)],
    )
    ax.set_title(f"Cross-Lingual RepInd Between Cone Directions: {model}")
    ax.set_xlabel("Basis direction")
    ax.set_ylabel("Basis direction")
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_refusal_cone_repind_{model}", formats)


def select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    if acts.ndim == 3:
        return acts[:, layer, :]
    if acts.ndim == 2:
        return acts
    raise ValueError(f"Unexpected activation shape: {acts.shape}")


def activation_section_languages(requested: list[str] | None, available: list[str]) -> list[str]:
    if requested:
        return [lang for lang in requested if lang in available]
    picks = []
    for lang in ["en", "ar", "hi", "yo"]:
        if lang in available:
            picks.append(lang)
    if len(picks) < 4:
        for lang in ordered_languages(available):
            if lang not in picks:
                picks.append(lang)
            if len(picks) >= 4:
                break
    return picks


def load_activation_projection_data(data: dict, args) -> dict[str, dict[str, np.ndarray]]:
    if not args.activations_dir:
        return {}

    act_dir = Path(args.activations_dir)
    if not act_dir.exists():
        print(f"Skipping activation sections: activations dir not found: {act_dir}")
        return {}

    from src.activations.cache import get_activation_path, load_activations
    from src.dataset.loader import load_dataset
    from src.utils.config import load_yaml

    summary = data["summary"]
    model = data["model"]
    layer = int(summary.get("layer", 0))
    perturbation = summary.get("perturbation", "standard_translation")
    token_position = summary.get("token_position", "last_post_instruction")

    lang_cfg = load_yaml("configs/languages.yaml")
    config_langs = []
    for tier_data in lang_cfg.get("tiers", {}).values():
        config_langs.extend(tier_data.get("languages", []))
    config_langs = list(dict.fromkeys(config_langs))

    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=[perturbation],
        languages=config_langs,
    )
    if df.empty:
        print("Skipping activation sections: dataset loader returned no rows.")
        return {}

    available = []
    for lang in config_langs:
        path = get_activation_path(model, lang, perturbation, token_position, "residual", str(act_dir))
        if Path(path).exists():
            available.append(lang)

    langs = activation_section_languages(args.languages, available)
    if not langs:
        print("Skipping activation sections: no matching activation files found.")
        return {}

    basis = data["basis"].astype(np.float32)
    rng = np.random.RandomState(42)
    out = {}
    for lang in langs:
        path = get_activation_path(model, lang, perturbation, token_position, "residual", str(act_dir))
        try:
            acts = select_layer(load_activations(path).numpy(), layer).astype(np.float32)
        except Exception as exc:
            print(f"  [WARN] Could not load activations for {lang}: {exc}")
            continue

        lang_df = df[df["language"] == lang].reset_index(drop=True)
        if len(lang_df) != len(acts):
            print(f"  [WARN] Size mismatch for {lang}: dataset={len(lang_df)}, acts={len(acts)}")
            continue

        is_harm = lang_df["is_harmful"].astype(bool).to_numpy()
        projections = acts @ basis[:, :min(3, basis.shape[1])]
        harm = projections[is_harm]
        safe = projections[~is_harm]
        if len(harm) > args.max_samples:
            harm = harm[rng.choice(len(harm), size=args.max_samples, replace=False)]
        if len(safe) > args.max_samples:
            safe = safe[rng.choice(len(safe), size=args.max_samples, replace=False)]
        out[lang] = {"harmful": harm, "harmless": safe}

    return out


def plot_activation_sections(data: dict, args, output_dir: Path, formats: list[str]):
    projection_data = load_activation_projection_data(data, args)
    if not projection_data:
        return

    model = data["model"]
    n_basis = data["basis"].shape[1]
    pairs = basis_pairs(min(3, n_basis))
    if not pairs:
        return

    langs = list(projection_data.keys())
    nrows = len(langs)
    ncols = len(pairs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)

    for row_idx, lang in enumerate(langs):
        harm = projection_data[lang]["harmful"]
        safe = projection_data[lang]["harmless"]
        for col_idx, (i, j) in enumerate(pairs):
            ax = axes[row_idx][col_idx]
            x_all = np.concatenate([harm[:, i], safe[:, i]])
            y_all = np.concatenate([harm[:, j], safe[:, j]])
            xmin, xmax = min(x_all.min(), 0.0), max(x_all.max(), 0.0)
            ymin, ymax = min(y_all.min(), 0.0), max(y_all.max(), 0.0)
            pad_x = 0.08 * (xmax - xmin + 1e-6)
            pad_y = 0.08 * (ymax - ymin + 1e-6)
            ax.axvspan(0, xmax + pad_x, color="#F5E6A1", alpha=0.14)
            ax.axhspan(0, ymax + pad_y, color="#F5E6A1", alpha=0.14)
            ax.scatter(safe[:, i], safe[:, j], s=12, alpha=0.35, color="#4A90D9", label="Harmless")
            ax.scatter(harm[:, i], harm[:, j], s=12, alpha=0.35, color="#D6604D", label="Harmful")
            ax.axhline(0, color="#333333", linewidth=0.7)
            ax.axvline(0, color="#333333", linewidth=0.7)
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
            ax.grid(linestyle="--", alpha=0.22)
            if row_idx == 0:
                ax.set_title(f"b{i} vs b{j}")
            if col_idx == 0:
                ax.set_ylabel(f"{lang}\nprojection b{j}")
            else:
                ax.set_ylabel(f"projection b{j}")
            ax.set_xlabel(f"projection b{i}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Activation-Space Cone Sections: {model}", y=1.05)
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig_refusal_cone_activation_sections_{model}", formats)


def main():
    args = parse_args()
    repr_dir, tmp_dir = prepare_representation_dir(Path(args.representation_dir), args.model)
    output_dir = Path(args.output_dir)
    data = load_cone_outputs(repr_dir, args.model)

    print(f"Loaded Step 4b outputs for {data['model']} from {repr_dir}")
    print(f"Basis shape: {data['basis'].shape}")
    if data["summary"]:
        print(
            "Summary: "
            f"layer={data['summary'].get('layer')}, "
            f"perturbation={data['summary'].get('perturbation')}, "
            f"position={data['summary'].get('token_position')}"
        )

    plot_cone_metrics(data, output_dir, args.formats)
    plot_language_margins(data, output_dir, args.formats)
    plot_margin_sections(data, output_dir, args.formats)
    plot_repind(data, output_dir, args.formats)
    plot_activation_sections(data, args, output_dir, args.formats)

    if tmp_dir is not None:
        tmp_dir.cleanup()


if __name__ == "__main__":
    main()
