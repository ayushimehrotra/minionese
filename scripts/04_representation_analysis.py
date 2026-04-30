#!/usr/bin/env python3
"""
Script 04: Representation Analysis

Trains linear probes, computes cross-lingual subspace metrics
(silhouette, principal angles, effective rank), and runs
harm/refusal disentanglement. All in one pass over all languages.

No coherence filtering: we analyze incoherent tiers too, because
subspace degeneration at those tiers is a core finding.

Produces:
  - results/representation/{model}/probe_summary.csv
  - results/representation/{model}/probes/probe_{lang}_layer{L}_all.npz
  - results/representation/{model}/silhouette_scores.csv
  - results/representation/{model}/principal_angles.csv
  - results/representation/{model}/effective_rank.csv
  - results/representation/{model}/disentangle_results.csv
  - results/representation/{model}/refusal_direction_{model}.npy
  - figures/fig3_silhouette_{model}.{pdf,png}
  - figures/fig4_principal_angles_{model}.{pdf,png}
  - figures/fig5_harm_refusal_{model}.{pdf,png}

Usage:
    python scripts/04_representation_analysis.py --model llama
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset
from src.probing.cross_lingual import compute_principal_angles
from src.probing.disentangle import extract_refusal_direction, disentangle_analysis
from src.probing.effective_rank import compute_effective_rank_table
from src.probing.linear_probe import train_all_probes
from src.probing.subspace import build_subspace_from_probes
from src.utils.config import load_config, load_yaml, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Representation analysis (probes + cross-lingual + disentangle).")
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "aya"])
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--output-dir", default="results/representation/")
    parser.add_argument("--figures-dir", default="figures/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _infer_harm_mask(df: pd.DataFrame) -> np.ndarray:
    """Return boolean array where True = harmful row."""
    if "is_harmful" in df.columns:
        return df["is_harmful"].fillna(False).astype(bool).to_numpy()
    raise ValueError(f"Cannot infer harm labels. Columns: {list(df.columns)}")


def _select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    acts = np.asarray(acts)
    if acts.ndim == 3:
        return acts[:, layer, :]
    if acts.ndim == 2 and layer == 0:
        return acts
    raise ValueError(f"Cannot select layer {layer} from shape {acts.shape}")


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("representation_analysis")
    setup_reproducibility(seed=42)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    num_layers = model_cfg.get("num_layers", 32)
    cr = model_cfg.get("critical_layer_range", [num_layers // 3, 2 * num_layers // 3])
    critical_layers = list(range(cr[0], cr[1]))
    layers = list(range(num_layers))

    lang_cfg = load_yaml("configs/languages.yaml")
    languages = []
    for tier_data in lang_cfg.get("tiers", {}).values():
        languages.extend(tier_data.get("languages", []))
    languages = list(dict.fromkeys(languages))

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model} | Languages: {len(languages)} | Critical layers: {critical_layers}")

    if args.dry_run:
        logger.info("[DRY RUN] Would run full representation analysis.")
        return

    # Load dataset
    logger.info("Loading dataset...")
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
    harm_categories = sorted(df["category"].dropna().unique().tolist())

    from src.activations.cache import load_activations, get_activation_path

    # Load activations for all languages
    activations = {}
    for lang in languages:
        act_path = get_activation_path(
            args.model, lang, args.perturbation, args.token_position,
            "residual", args.activations_dir
        )
        if not Path(act_path).exists():
            logger.warning(f"Missing activations: {act_path}")
            continue
        try:
            act_tensor = load_activations(act_path).numpy()
            lang_df = df[df["language"] == lang].reset_index(drop=True)
            if len(lang_df) != act_tensor.shape[0]:
                logger.warning(f"Size mismatch {lang}: dataset={len(lang_df)}, acts={act_tensor.shape[0]}")
                continue
            lang_harm = lang_df["__is_harmful__"].to_numpy().astype(bool)
            activations[(lang, "harmful")] = act_tensor[lang_harm]
            activations[(lang, "harmless")] = act_tensor[~lang_harm]
            logger.debug(f"Loaded {lang}: {lang_harm.sum()} harmful, {(~lang_harm).sum()} harmless")
        except Exception as e:
            logger.error(f"Error loading {act_path}: {e}")

    # === PROBING PHASE ===
    logger.info("Training probes...")
    probe_summary_df = train_all_probes(
        activations_dir=args.activations_dir,
        dataset=df,
        languages=languages,
        layers=layers,
        harm_categories=harm_categories,
        output_dir=str(probes_dir),
        model_short=args.model,
        perturbation=args.perturbation,
        token_position=args.token_position,
    )
    probe_summary_df.to_csv(output_dir / "probe_summary.csv", index=False)
    logger.info(f"Trained {len(probe_summary_df)} probes.")

    # === CROSS-LINGUAL PHASE ===
    logger.info("Building subspace projections...")
    subspace_projections = {}
    subspace_bases = {}  # (lang, layer) -> Vt_top (rank, hidden) — used for principal angles
    for lang in languages:
        for layer in critical_layers:
            try:
                subspace = build_subspace_from_probes(
                    str(probes_dir), lang, layer, harm_categories
                )
                subspace_projections[(lang, layer)] = subspace["projection_matrix"]
                # Vt[:effective_rank] are the basis vectors (rank, hidden_dim)
                subspace_bases[(lang, layer)] = subspace["Vt"][:subspace["effective_rank"]].T
            except Exception as e:
                logger.debug(f"No subspace for ({lang}, layer={layer}): {e}")

    # Silhouette scores
    sil_out = output_dir / "silhouette_scores.csv"
    if sil_out.exists() and sil_out.stat().st_size > 10:
        logger.info("Silhouette scores already computed, skipping.")
        sil_df = pd.read_csv(sil_out)
    else:
        logger.info("Computing silhouette scores...")
        sil_rows = []
        for lang in languages:
            harm_acts = activations.get((lang, "harmful"))
            harmless_acts = activations.get((lang, "harmless"))
            if harm_acts is None or harmless_acts is None:
                continue
            for layer in critical_layers:
                try:
                    hl = _select_layer(harm_acts, layer)
                    hsl = _select_layer(harmless_acts, layer)
                    if hl.shape[0] < 2 or hsl.shape[0] < 2:
                        continue
                    P = subspace_projections.get((lang, layer))
                    if P is not None and hl.shape[1] == P.shape[0]:
                        hl = hl @ P
                        hsl = hsl @ P
                    X = np.vstack([hl, hsl])
                    y = np.array([1] * len(hl) + [0] * len(hsl))
                    score = silhouette_score(X, y)
                    sil_rows.append({"language": lang, "layer": layer, "silhouette_score": float(score)})
                except Exception as e:
                    logger.debug(f"Silhouette failed ({lang}, {layer}): {e}")
        sil_df = pd.DataFrame(sil_rows)
        sil_df.to_csv(sil_out, index=False)

    # Principal angles
    logger.info("Computing principal angles...")
    pa_rows = []
    for lang in languages:
        if lang == "en":
            continue
        for layer in critical_layers:
            en_proj = subspace_projections.get(("en", layer))
            lang_proj = subspace_projections.get((lang, layer))
            if en_proj is None or lang_proj is None:
                continue
            try:
                U_en = subspace_bases.get(("en", layer))
                U_lang = subspace_bases.get((lang, layer))
                if U_en is None or U_lang is None:
                    continue
                angles = compute_principal_angles(U_en, U_lang)
                for i, angle in enumerate(angles):
                    pa_rows.append({"language": lang, "layer": layer,
                                    "angle_idx": i, "angle_rad": float(angle),
                                    "angle_deg": float(np.degrees(angle))})
            except Exception as e:
                logger.debug(f"Principal angles failed ({lang}, {layer}): {e}")
    pa_df = pd.DataFrame(pa_rows)
    pa_df.to_csv(output_dir / "principal_angles.csv", index=False)

    # Effective rank
    logger.info("Computing effective rank...")
    rank_dfs = []
    for lang in languages:
        acts = activations.get((lang, "harmful"))
        if acts is None:
            continue
        if acts.ndim == 3:
            acts_by_layer = {l: acts[:, l, :] for l in range(acts.shape[1])}
        elif acts.ndim == 2:
            acts_by_layer = {0: acts}
        else:
            continue
        try:
            rank_df = compute_effective_rank_table(acts_by_layer, lang)
            rank_dfs.append(rank_df)
        except Exception as e:
            logger.warning(f"Effective rank failed for {lang}: {e}")
    if rank_dfs:
        all_rank_df = pd.concat(rank_dfs, ignore_index=True)
        all_rank_df.to_csv(output_dir / "effective_rank.csv", index=False)

    # === DISENTANGLEMENT PHASE ===
    logger.info("Running disentanglement analysis...")
    crit_layer = critical_layers[len(critical_layers) // 2]

    en_acts_all = activations.get(("en", "harmful"))
    if en_acts_all is not None:
        if en_acts_all.ndim == 3:
            en_layer_acts = en_acts_all[:, crit_layer, :]
        else:
            en_layer_acts = en_acts_all

        # Load WildGuard labels for English harmful prompts (better than naive split)
        scored_path = Path("results/evaluation/all_scored.jsonl")
        en_harmful_labels = {}
        if scored_path.exists():
            with open(scored_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("language") == "en" and rec.get("is_harmful"):
                        en_harmful_labels[rec.get("prompt_id", "")] = rec.get("wildguard_label", "safe")

        # Align labels to activation row order using prompt_ids from dataset
        en_df = df[(df["language"] == "en") & (df["__is_harmful__"])].reset_index(drop=True)
        if en_harmful_labels and len(en_df) == len(en_layer_acts):
            en_prompt_ids = [
                f"harmful_{i:04d}_en_{args.perturbation}"
                for i in range(len(en_df))
            ]
            refused_mask = np.array([
                en_harmful_labels.get(pid, "safe") in ("safe", "refusal")
                for pid in en_prompt_ids
            ])
            complied_mask = ~refused_mask
            if refused_mask.sum() >= 5 and complied_mask.sum() >= 5:
                refused_acts = en_layer_acts[refused_mask]
                complied_acts = en_layer_acts[complied_mask]
                logger.info(f"Refusal direction: {refused_mask.sum()} refused, {complied_mask.sum()} complied.")
            else:
                logger.warning(
                    f"Insufficient behavioral contrast: {refused_mask.sum()} refused, "
                    f"{complied_mask.sum()} complied. Falling back to naive split."
                )
                n = len(en_layer_acts)
                refused_acts = en_layer_acts[:n // 2]
                complied_acts = en_layer_acts[n // 2:]
        else:
            logger.warning("No WildGuard labels found. Using naive split for refusal direction.")
            n = len(en_layer_acts)
            refused_acts = en_layer_acts[:n // 2]
            complied_acts = en_layer_acts[n // 2:]

        refusal_direction = extract_refusal_direction(args.model, refused_acts, complied_acts)
        np.save(str(output_dir / f"refusal_direction_{args.model}.npy"), refusal_direction)
        logger.info(f"Refusal direction saved: norm={np.linalg.norm(refusal_direction):.4f}")

        # Load activations for disentanglement
        activations_t_inst = {}
        activations_t_post = {}
        harm_subspaces = {}

        for lang in languages:
            for layer in critical_layers:
                try:
                    subspace = build_subspace_from_probes(str(probes_dir), lang, layer, harm_categories)
                    harm_subspaces[(lang, layer)] = subspace["W_l"]
                except Exception:
                    pass

            for position, acts_dict in [
                ("last_instruction", activations_t_inst),
                ("last_post_instruction", activations_t_post),
            ]:
                act_path = get_activation_path(
                    args.model, lang, args.perturbation, position, "residual", args.activations_dir
                )
                if Path(act_path).exists():
                    try:
                        acts = load_activations(act_path)
                        val = acts[:, crit_layer, :].numpy() if acts.ndim == 3 else acts.numpy()
                        lang_df = df[df["language"] == lang].reset_index(drop=True)
                        if len(lang_df) == len(val):
                            harm_mask_lang = lang_df["__is_harmful__"].values
                            acts_dict[(lang, "harmful")] = val[harm_mask_lang]
                            acts_dict[(lang, "harmless")] = val[~harm_mask_lang]
                        else:
                            acts_dict[(lang, "harmful")] = val
                    except Exception as e:
                        logger.debug(f"Error loading {act_path}: {e}")

        results_df = disentangle_analysis(
            harm_subspaces=harm_subspaces,
            refusal_direction=refusal_direction,
            activations_t_inst=activations_t_inst,
            activations_t_post_inst=activations_t_post,
            layers=critical_layers,
        )
        results_df.to_csv(output_dir / "disentangle_results.csv", index=False)
        logger.info("Disentanglement analysis complete.")
    else:
        logger.warning("No English harmful activations found; skipping disentanglement.")

    # === FIGURES ===
    try:
        from src.visualization.heatmaps import plot_silhouette_heatmap, plot_effective_rank

        if not sil_df.empty:
            plot_silhouette_heatmap(sil_df, str(figures_dir / f"fig3_silhouette_{args.model}"))
            logger.info("Figure 3 (silhouette heatmap) saved.")

        if rank_dfs:
            plot_effective_rank(all_rank_df, str(figures_dir / f"fig4_effective_rank_{args.model}"))
            logger.info("Figure 4 (effective rank) saved.")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    logger.info(f"Representation analysis complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
