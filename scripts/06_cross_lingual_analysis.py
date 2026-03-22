#!/usr/bin/env python3
"""
Script 06: Cross-Lingual Analysis

Compute principal angles, silhouette scores, effective rank.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.cross_lingual import compute_principal_angles, compute_silhouette_map
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
    parser.add_argument("--output-dir", default="results/cross_lingual/")
    parser.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--critical-layers", default=None,
                        help="Comma-separated layer indices for principal angle analysis.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


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

    # --- Load activations for silhouette analysis ---
    logger.info("Loading activations for silhouette analysis...")
    from src.activations.cache import load_activations, get_activation_path

    activations: dict = {}
    subspace_projections: dict = {}

    for lang in languages:
        for label, is_harm in [("harmful", True), ("harmless", False)]:
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
                act_tensor = load_activations(act_path)
                # Assume harmful/harmless are interleaved -- need dataset for correct split
                # For now use all activations (will be split properly in full implementation)
                activations[(lang, label)] = act_tensor.numpy()
            except Exception as e:
                logger.error(f"Error loading {act_path}: {e}")

    # Load subspace projections from probes
    harm_categories = []
    probe_summary_path = Path(args.probes_dir) / "probe_summary.csv"
    if probe_summary_path.exists():
        probe_summary = pd.read_csv(probe_summary_path)
        harm_categories = [c for c in probe_summary["category"].unique() if c != "all"]

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
    sil_df = compute_silhouette_map(activations, subspace_projections, critical_layers, languages)
    sil_df.to_csv(output_dir / "silhouette_scores.csv", index=False)

    plot_silhouette_heatmap(sil_df, str(output_dir / "fig_silhouette_heatmap"))
    logger.info("Silhouette heatmap saved.")

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
            # Use singular vectors as bases
            U_en = np.linalg.svd(en_subspaces[layer], full_matrices=False)[2][:5].T
            U_lang = np.linalg.svd(lang_proj, full_matrices=False)[2][:5].T
            angles = compute_principal_angles(U_en, U_lang)
            for i, angle in enumerate(angles):
                pa_rows.append({
                    "language": lang,
                    "layer": layer,
                    "angle_idx": i,
                    "angle_rad": float(angle),
                    "angle_deg": float(np.degrees(angle)),
                })

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
        # acts shape: (n_prompts, n_layers, hidden) or (n_prompts, hidden)
        if acts.ndim == 3:
            acts_by_layer = {l: acts[:, l, :] for l in range(acts.shape[1])}
        else:
            acts_by_layer = {0: acts}
        rank_df = compute_effective_rank_table(acts_by_layer, lang)
        rank_dfs.append(rank_df)

    if rank_dfs:
        all_rank_df = pd.concat(rank_dfs, ignore_index=True)
        all_rank_df.to_csv(output_dir / "effective_rank.csv", index=False)
        plot_effective_rank(all_rank_df, str(output_dir / "fig_effective_rank"))
        logger.info("Effective rank analysis saved.")

    logger.info(f"Cross-lingual analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
