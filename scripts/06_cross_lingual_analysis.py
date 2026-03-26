#!/usr/bin/env python3
"""
Script 06: Cross-Lingual Analysis

Compute principal angles, silhouette scores, and effective rank.
Generates heatmaps.

Usage:
    python scripts/06_cross_lingual_analysis.py \
        --probes-dir results/probes/ \
        --activations-dir data/activations/ \
        --output-dir results/cross_lingual/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.probing.cross_lingual import compute_principal_angles
from src.probing.effective_rank import compute_effective_rank_table
from src.probing.subspace import build_subspace_from_probes
from src.utils.config import load_config, load_yaml
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.visualization.heatmaps import plot_silhouette_heatmap, plot_effective_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-lingual subspace analysis.")
    parser.add_argument("--probes-dir", default="results/probes/")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--output-dir", default="results/cross_lingual/")
    parser.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument(
        "--critical-layers",
        default=None,
        help="Comma-separated layer indices for principal angle analysis.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _normalize_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _infer_harm_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Infer a boolean harmful mask from common dataset column conventions.
    Returns a boolean numpy array where True means harmful.
    """
    candidate_cols = [
        "is_harmful",
        "harmful",
        "label",
        "binary_label",
        "class",
        "target",
        "is_safe",
        "safe",
        "harmless",
        "benign",
        "safety_label",
    ]

    harmful_tokens = {
        "1", "true", "t", "yes", "y",
        "harmful", "unsafe", "malicious", "attack", "jailbreak", "toxic",
    }
    harmless_tokens = {
        "0", "false", "f", "no", "n",
        "harmless", "safe", "benign", "clean",
    }

    for col in candidate_cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Boolean columns
        if pd.api.types.is_bool_dtype(series):
            vals = series.fillna(False).astype(bool).to_numpy()
            if col in {"is_safe", "safe", "harmless", "benign"}:
                return ~vals
            return vals

        # Numeric binary columns
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            unique_vals = set(non_null.astype(int).unique().tolist())
            if unique_vals.issubset({0, 1}):
                vals = series.fillna(0).astype(int).astype(bool).to_numpy()
                if col in {"is_safe", "safe", "harmless", "benign"}:
                    return ~vals
                return vals

        # String-like binary columns
        norm = _normalize_string_series(series.dropna())
        unique_vals = set(norm.unique().tolist())
        if unique_vals and unique_vals.issubset(harmful_tokens | harmless_tokens):
            full_norm = _normalize_string_series(series.fillna("0"))
            vals = full_norm.isin(harmful_tokens).to_numpy()
            if col in {"is_safe", "safe", "harmless", "benign"}:
                return ~vals
            return vals

    raise ValueError(
        "Could not infer harmful/harmless labels from dataset. "
        f"Available columns: {list(df.columns)}"
    )


def _select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    """
    Convert activations to a 2D matrix for one layer.
    Expected input shape is (n_prompts, n_layers, hidden) or (n_prompts, hidden).
    """
    acts = np.asarray(acts)

    if acts.ndim == 3:
        if layer >= acts.shape[1]:
            raise IndexError(f"Layer {layer} out of bounds for activations with shape {acts.shape}")
        return acts[:, layer, :]

    if acts.ndim == 2:
        if layer != 0:
            raise IndexError(
                f"Received 2D activations with shape {acts.shape}; only layer 0 is valid, got layer={layer}"
            )
        return acts

    raise ValueError(f"Expected 2D or 3D activations, got shape {acts.shape}")


def _compute_silhouette_df(
    activations: dict,
    subspace_projections: dict,
    critical_layers: list[int],
    languages: list[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []

    for lang in languages:
        harm_acts = activations.get((lang, "harmful"))
        harmless_acts = activations.get((lang, "harmless"))

        if harm_acts is None or harmless_acts is None:
            logger.warning(f"Missing harmful/harmless activations for {lang}; skipping silhouette.")
            continue

        for layer in critical_layers:
            try:
                harm_layer = _select_layer(harm_acts, layer)
                harmless_layer = _select_layer(harmless_acts, layer)

                if harm_layer.shape[0] < 2 or harmless_layer.shape[0] < 2:
                    logger.warning(
                        f"Skipping silhouette ({lang}, layer={layer}): "
                        f"need at least 2 samples per class, got "
                        f"{harm_layer.shape[0]} harmful and {harmless_layer.shape[0]} harmless."
                    )
                    continue

                P = subspace_projections.get((lang, layer))
                if P is not None:
                    if harm_layer.shape[1] != P.shape[0]:
                        logger.warning(
                            f"Projection mismatch ({lang}, layer={layer}): "
                            f"acts dim={harm_layer.shape[1]}, projection shape={P.shape}. "
                            f"Using raw activations instead."
                        )
                        harm_proj = harm_layer
                        harmless_proj = harmless_layer
                    else:
                        harm_proj = harm_layer @ P
                        harmless_proj = harmless_layer @ P
                else:
                    harm_proj = harm_layer
                    harmless_proj = harmless_layer

                X = np.vstack([harm_proj, harmless_proj])
                y = np.array([1] * len(harm_proj) + [0] * len(harmless_proj))

                if X.ndim != 2:
                    raise ValueError(f"Silhouette input must be 2D, got shape {X.shape}")

                score = silhouette_score(X, y)
                rows.append(
                    {
                        "language": lang,
                        "layer": layer,
                        "silhouette_score": float(score),
                        "n_harmful": int(len(harm_proj)),
                        "n_harmless": int(len(harmless_proj)),
                        "projected": bool(P is not None),
                    }
                )

            except Exception as e:
                logger.warning(f"Silhouette score failed ({lang}, layer={layer}): {e}")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("cross_lingual_analysis")
    setup_reproducibility(seed=42)

    config = load_config()
    lang_cfg = load_yaml("configs/languages.yaml")

    languages = []
    for tier_data in lang_cfg.get("tiers", {}).values():
        languages.extend(tier_data.get("languages", []))
    languages = list(dict.fromkeys(languages))

    model_cfg = config["models"]["target_models"].get(args.model, {})
    num_layers = model_cfg.get("num_layers", 32)
    layers = list(range(num_layers))

    if args.critical_layers:
        critical_layers = [int(l) for l in args.critical_layers.split(",")]
    else:
        cr = model_cfg.get("critical_layer_range", [num_layers // 3, 2 * num_layers // 3])
        critical_layers = list(range(cr[0], cr[1]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Would run cross-lingual analysis.")
        return

    logger.info("Loading dataset for harmful/harmless split...")
    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=[args.perturbation],
        languages=languages,
    )
    if df.empty:
        logger.error("No dataset loaded.")
        sys.exit(1)

    harm_mask = _infer_harm_mask(df)
    df = df.copy()
    df["__is_harmful__"] = harm_mask

    # --- Load activations and split harmful vs harmless ---
    logger.info("Loading activations for silhouette and rank analysis...")
    from src.activations.cache import load_activations, get_activation_path

    activations: dict = {}
    subspace_projections: dict = {}

    for lang in languages:
        lang_df = df[df["language"] == lang].reset_index(drop=True)

        if lang_df.empty:
            logger.warning(f"No dataset rows for language={lang}")
            continue

        act_path = get_activation_path(
            model=args.model,
            language=lang,
            perturbation=args.perturbation,
            position=args.token_position,
            component="residual",
            cache_dir=args.activations_dir,
        )

        if not Path(act_path).exists():
            logger.warning(f"Missing activations: {act_path}")
            continue

        try:
            act_tensor = load_activations(act_path).numpy()

            if len(lang_df) != act_tensor.shape[0]:
                logger.warning(
                    f"Dataset/activation size mismatch for {lang}: "
                    f"dataset={len(lang_df)}, activations={act_tensor.shape[0]}. Skipping."
                )
                continue

            lang_harm_mask = lang_df["__is_harmful__"].to_numpy().astype(bool)
            harmful_acts = act_tensor[lang_harm_mask]
            harmless_acts = act_tensor[~lang_harm_mask]

            if len(harmful_acts) == 0 or len(harmless_acts) == 0:
                logger.warning(
                    f"Language {lang} has an empty class after split: "
                    f"{len(harmful_acts)} harmful, {len(harmless_acts)} harmless. Skipping."
                )
                continue

            activations[(lang, "harmful")] = harmful_acts
            activations[(lang, "harmless")] = harmless_acts

            logger.info(
                f"Loaded {lang}: harmful={len(harmful_acts)}, harmless={len(harmless_acts)}, "
                f"shape={act_tensor.shape}"
            )

        except Exception as e:
            logger.error(f"Error loading {act_path}: {e}")

    # --- Load subspace projections from probes ---
    harm_categories = []
    probe_summary_path = Path(args.probes_dir) / "probe_summary.csv"
    if probe_summary_path.exists():
        probe_summary = pd.read_csv(probe_summary_path)
        if "category" in probe_summary.columns:
            harm_categories = [
                c for c in probe_summary["category"].dropna().unique().tolist()
                if c != "all"
            ]

    for lang in languages:
        for layer in critical_layers:
            try:
                subspace = build_subspace_from_probes(
                    probes_dir=args.probes_dir,
                    language=lang,
                    layer=layer,
                    categories=harm_categories,
                )
                subspace_projections[(lang, layer)] = subspace["projection_matrix"]
            except Exception as e:
                logger.debug(f"Could not build subspace for ({lang}, layer={layer}): {e}")

    # --- Silhouette score heatmap ---
    logger.info("Computing silhouette scores...")
    sil_df = _compute_silhouette_df(
        activations=activations,
        subspace_projections=subspace_projections,
        critical_layers=critical_layers,
        languages=languages,
        logger=logger,
    )
    sil_df.to_csv(output_dir / "silhouette_scores.csv", index=False)

    if not sil_df.empty:
        plot_silhouette_heatmap(sil_df, str(output_dir / "fig_silhouette_heatmap"))
        logger.info("Silhouette heatmap saved.")
    else:
        logger.warning("Silhouette dataframe is empty; skipping heatmap.")

    # --- Principal angles ---
    logger.info("Computing principal angles between English and other languages...")
    pa_rows = []
    en_subspaces = {
        layer: subspace_projections.get(("en", layer))
        for layer in critical_layers
        if ("en", layer) in subspace_projections
    }

    for lang in languages:
        if lang == "en":
            continue
        for layer in critical_layers:
            if layer not in en_subspaces or en_subspaces[layer] is None:
                continue

            lang_proj = subspace_projections.get((lang, layer))
            if lang_proj is None:
                continue

            try:
                # Use top singular vectors as bases
                U_en = np.linalg.svd(en_subspaces[layer], full_matrices=False)[2][:5].T
                U_lang = np.linalg.svd(lang_proj, full_matrices=False)[2][:5].T
                angles = compute_principal_angles(U_en, U_lang)

                for i, angle in enumerate(angles):
                    pa_rows.append(
                        {
                            "language": lang,
                            "layer": layer,
                            "angle_idx": i,
                            "angle_rad": float(angle),
                            "angle_deg": float(np.degrees(angle)),
                        }
                    )
            except Exception as e:
                logger.warning(f"Principal angle computation failed ({lang}, layer={layer}): {e}")

    pa_df = pd.DataFrame(pa_rows)
    pa_df.to_csv(output_dir / "principal_angles.csv", index=False)
    logger.info("Principal angles computed and saved.")

    # --- Effective rank analysis ---
    logger.info("Computing effective rank profiles...")
    rank_dfs = []
    for lang in languages:
        act_key = (lang, "harmful")
        if act_key not in activations:
            continue

        acts = activations[act_key]
        if acts.ndim == 3:
            acts_by_layer = {l: acts[:, l, :] for l in range(min(acts.shape[1], len(layers)))}
        elif acts.ndim == 2:
            acts_by_layer = {0: acts}
        else:
            logger.warning(f"Unexpected activation rank for {lang}: shape={acts.shape}")
            continue

        try:
            rank_df = compute_effective_rank_table(acts_by_layer, lang)
            rank_dfs.append(rank_df)
        except Exception as e:
            logger.warning(f"Effective rank failed for {lang}: {e}")

    if rank_dfs:
        all_rank_df = pd.concat(rank_dfs, ignore_index=True)
        all_rank_df.to_csv(output_dir / "effective_rank.csv", index=False)
        plot_effective_rank(all_rank_df, str(output_dir / "fig_effective_rank"))
        logger.info("Effective rank analysis saved.")
    else:
        logger.warning("No effective rank results produced.")

    logger.info(f"Cross-lingual analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()