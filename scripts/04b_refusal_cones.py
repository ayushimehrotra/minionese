#!/usr/bin/env python3
"""
Script 04b: Cross-Lingual Refusal Cone Discovery

Discovers a multi-dimensional cone of refusal directions and analyzes their
cross-lingual representational independence. Inspired by Wollschlager et al.
(2025) "The Geometry of Refusal in Large Language Models", with two original
twists tailored to this multilingual project:

  1. Activation-domain cone optimization. The paper's RCO requires gradient-
     based interventions through the model; we instead optimize on cached
     residual activations, which we already have for every (model, language,
     perturbation, position).

  2. Cross-lingual representational independence (CL-RepInd). The paper's
     RepInd is monolingual. We measure independence across languages: do
     two refusal directions act through different cross-lingual circuits?

Dependencies (must run first):
  - 03_extract_activations.py   (cached residual activations)
  - 04_representation_analysis.py   (refusal_direction_{model}.npy + probes)

Outputs:
  - results/representation/{model}/refusal_cone_{model}.npy            (d, N) basis
  - results/representation/{model}/refusal_cone_metrics.csv            per-dim margins
  - results/representation/{model}/refusal_cone_repind.csv             pairwise CL-RepInd
  - results/representation/{model}/refusal_cone_summary.json           run metadata

Usage:
    python scripts/04b_refusal_cones.py --model aya
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activations.cache import get_activation_path, load_activations
from src.dataset.loader import load_dataset
from src.probing.refusal_cone import (
    CrossLingualActivations,
    cone_attack_proxy,
    optimize_cross_lingual_cone,
    repind_matrix,
)
from src.utils.config import get_model_config, load_config, load_yaml
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover cross-lingual refusal cones from cached activations."
    )
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "aya"])
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--representation-dir", default=None,
                        help="Defaults to results/representation/{model}/.")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--token-position", default="last_post_instruction")
    parser.add_argument("--cone-dims", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Cone dimensionalities to fit; one basis per dim is saved "
                             "but only the largest is written as the canonical basis.")
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--cone-weight", type=float, default=5.0)
    parser.add_argument("--xling-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    if acts.ndim == 3:
        return acts[:, layer, :]
    if acts.ndim == 2:
        return acts
    raise ValueError(f"Unexpected activation shape: {acts.shape}")


def _load_cross_lingual(
    model: str, perturbation: str, position: str, layer: int,
    activations_dir: str, dataset_dir: str, languages: list, logger: logging.Logger,
) -> CrossLingualActivations:
    """Load and align cached residual activations for every language."""
    df = load_dataset(dataset_dir=dataset_dir, perturbations=[perturbation], languages=languages)
    if df.empty:
        raise RuntimeError("Dataset load returned empty frame.")

    en_h = en_s = None
    nonen_h = {}
    nonen_s = {}

    for lang in languages:
        path = get_activation_path(model, lang, perturbation, position, "residual", activations_dir)
        if not Path(path).exists():
            logger.warning(f"Missing activations for {lang}: {path}")
            continue
        try:
            tensor = load_activations(path)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            continue

        lang_df = df[df["language"] == lang].reset_index(drop=True)
        if len(lang_df) != tensor.shape[0]:
            logger.warning(
                f"Size mismatch for {lang}: dataset={len(lang_df)} vs acts={tensor.shape[0]}; skipping."
            )
            continue

        layer_acts = _select_layer(tensor.numpy(), layer)
        is_harm = lang_df["is_harmful"].astype(bool).to_numpy()
        h_arr = torch.tensor(layer_acts[is_harm], dtype=torch.float32)
        s_arr = torch.tensor(layer_acts[~is_harm], dtype=torch.float32)

        if lang == "en":
            en_h, en_s = h_arr, s_arr
        else:
            nonen_h[lang] = h_arr
            nonen_s[lang] = s_arr

    if en_h is None or en_s is None:
        raise RuntimeError("English activations are required for cone discovery but were not found.")

    return CrossLingualActivations(en_h, en_s, nonen_h, nonen_s)


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("refusal_cones")
    setup_reproducibility(seed=args.seed)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    num_layers = model_cfg.get("num_layers", 32)
    cr = model_cfg.get("critical_layer_range", [num_layers // 3, 2 * num_layers // 3])

    # Match 07_interventions.py: prefer the attribution-patching critical layer
    # if it exists, otherwise fall back to the start of the range. This keeps
    # the cone basis on the same layer that the intervention sweep will use.
    critical_layers_path = Path("results/attribution/critical_layers.json")
    if critical_layers_path.exists():
        try:
            with open(critical_layers_path) as f:
                critical_data = json.load(f)
            cl_list = critical_data.get("critical_layers", [])
            critical_layer = int(cl_list[0]) if cl_list else int(cr[0])
        except Exception:
            critical_layer = int(cr[0])
    else:
        critical_layer = int(cr[0])

    lang_cfg = load_yaml("configs/languages.yaml")
    languages = []
    for tier_data in lang_cfg.get("tiers", {}).values():
        languages.extend(tier_data.get("languages", []))
    languages = list(dict.fromkeys(languages))

    repr_dir = Path(args.representation_dir) if args.representation_dir else \
        Path(f"results/representation/{args.model}")
    repr_dir.mkdir(parents=True, exist_ok=True)

    seed_path = repr_dir / f"refusal_direction_{args.model}.npy"
    if not seed_path.exists():
        logger.error(
            f"Missing seed refusal direction at {seed_path}. "
            "Run scripts/04_representation_analysis.py first."
        )
        sys.exit(1)
    seed_direction = np.load(str(seed_path)).astype(np.float32)
    logger.info(
        f"Loaded seed DIM direction from {seed_path}: norm={np.linalg.norm(seed_direction):.4f}"
    )

    if args.dry_run:
        logger.info(
            f"[DRY RUN] Would optimize cones of dims {args.cone_dims} at layer {critical_layer} "
            f"using {len(languages)} languages."
        )
        return

    cl_acts = _load_cross_lingual(
        args.model, args.perturbation, args.token_position, critical_layer,
        args.activations_dir, args.dataset_dir, languages, logger,
    )
    logger.info(
        f"Loaded activations: EN harmful={tuple(cl_acts.en_harmful.shape)}, "
        f"EN harmless={tuple(cl_acts.en_harmless.shape)}, "
        f"non-EN langs={cl_acts.nonen_languages}"
    )

    if cl_acts.hidden_dim != seed_direction.shape[0]:
        raise RuntimeError(
            f"Hidden dim mismatch: activations={cl_acts.hidden_dim} vs "
            f"seed direction={seed_direction.shape[0]}"
        )

    metric_rows = []
    per_lang_rows = []
    largest_basis = None
    largest_dim = -1

    for dim in sorted(set(args.cone_dims)):
        logger.info(f"Optimizing cross-lingual cone of dimension {dim}...")
        result = optimize_cross_lingual_cone(
            activations=cl_acts,
            seed_direction=seed_direction,
            cone_dim=dim,
            n_steps=args.n_steps,
            lr=args.lr,
            xling_weight=args.xling_weight,
            cone_weight=args.cone_weight,
            seed=args.seed,
        )

        proxy_en = cone_attack_proxy(
            result["basis"],
            cl_acts.en_harmful.numpy(),
            cl_acts.en_harmless.numpy(),
            seed=args.seed,
        )

        for i in range(dim):
            metric_rows.append({
                "cone_dim": dim,
                "basis_idx": i,
                "en_margin": float(result["en_margins"][i]),
                "xling_margin": float(result["xling_margins"][i]),
                "seed_alignment": float(result["seed_alignment"][i]),
                "cone_mean_margin_en": proxy_en["mean_margin"],
                "cone_min_margin_en": proxy_en["min_margin"],
                "cone_frac_positive_en": proxy_en["frac_positive"],
            })
            for lang, margins in result.get("per_language_margins", {}).items():
                per_lang_rows.append({
                    "cone_dim": dim,
                    "basis_idx": i,
                    "language": lang,
                    "margin": float(margins[i]),
                })

        logger.info(
            f"  dim={dim}: en_margin_mean={result['en_margins'].mean():.4f}, "
            f"xling_margin_mean={result['xling_margins'].mean():.4f}, "
            f"cone_frac_positive={proxy_en['frac_positive']:.3f}"
        )

        if dim > largest_dim:
            largest_dim = dim
            largest_basis = result

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = repr_dir / "refusal_cone_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Wrote per-dimension metrics: {metrics_path}")

    if per_lang_rows:
        per_lang_df = pd.DataFrame(per_lang_rows)
        per_lang_path = repr_dir / "refusal_cone_per_language.csv"
        per_lang_df.to_csv(per_lang_path, index=False)
        logger.info(f"Wrote per-language margins: {per_lang_path}")

    basis_path = repr_dir / f"refusal_cone_{args.model}.npy"
    np.save(str(basis_path), largest_basis["basis"])
    logger.info(f"Wrote canonical basis (dim={largest_dim}): {basis_path}")

    if largest_basis["basis"].shape[1] >= 2:
        nonen_arrays = {lang: t.numpy() for lang, t in cl_acts.nonen_harmful.items()}
        M = repind_matrix(largest_basis["basis"], nonen_arrays)
        repind_rows = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i == j:
                    continue
                repind_rows.append({"i": i, "j": j, "cl_repind": float(M[i, j])})
        repind_df = pd.DataFrame(repind_rows)
        repind_path = repr_dir / "refusal_cone_repind.csv"
        repind_df.to_csv(repind_path, index=False)
        logger.info(f"Wrote pairwise CL-RepInd matrix: {repind_path}")
    else:
        logger.info("Skipping CL-RepInd matrix (only one basis vector).")

    summary = {
        "model": args.model,
        "layer": critical_layer,
        "perturbation": args.perturbation,
        "token_position": args.token_position,
        "cone_dims_tried": sorted(set(args.cone_dims)),
        "canonical_dim": int(largest_dim),
        "nonen_languages": cl_acts.nonen_languages,
        "n_en_harmful": int(cl_acts.en_harmful.shape[0]),
        "n_en_harmless": int(cl_acts.en_harmless.shape[0]),
        "n_steps": args.n_steps,
        "lr": args.lr,
        "cone_weight": args.cone_weight,
        "xling_weight": args.xling_weight,
        "seed": args.seed,
    }
    summary_path = repr_dir / "refusal_cone_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote run summary: {summary_path}")
    logger.info("Refusal cone discovery complete.")


if __name__ == "__main__":
    main()
