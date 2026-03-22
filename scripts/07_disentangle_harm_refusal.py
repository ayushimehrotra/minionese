#!/usr/bin/env python3
"""
Script 07: Disentangle Harm/Refusal

Extends Zhao et al. cross-lingually. Separates harm detection from refusal gate failure.

Usage:
    python scripts/07_disentangle_harm_refusal.py \
        --probes-dir results/probes/ \
        --activations-dir data/activations/ \
        --output-dir results/disentangle/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.disentangle import (
    extract_refusal_direction,
    disentangle_analysis,
)
from src.utils.config import load_config, load_yaml
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Disentangle harmfulness vs. refusal directions.")
    parser.add_argument("--probes-dir", default="results/probes/")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--output-dir", default="results/disentangle/")
    parser.add_argument("--model", default="llama", choices=["llama", "gemma", "qwen"])
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("disentangle_harm_refusal")
    setup_reproducibility(seed=42)

    config = load_config()
    lang_cfg = load_yaml("configs/languages.yaml")

    languages = []
    for tier_data in lang_cfg.get("tiers", {}).values():
        languages.extend(tier_data.get("languages", []))

    model_cfg = config["models"]["target_models"].get(args.model, {})
    num_layers = model_cfg.get("num_layers", 32)
    cr = model_cfg.get("critical_layer_range", [num_layers // 3, 2 * num_layers // 3])
    critical_layers = list(range(cr[0], cr[1]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Would run disentanglement analysis.")
        return

    from src.activations.cache import load_activations, get_activation_path

    # Load English activations for refusal direction extraction
    logger.info("Loading English activations for refusal direction...")
    en_t_post_path = get_activation_path(
        args.model, "en", args.perturbation, "last_post_instruction", "residual",
        args.activations_dir
    )

    if not Path(en_t_post_path).exists():
        logger.error(f"English activations not found: {en_t_post_path}")
        logger.error("Run script 04 first to extract activations.")
        sys.exit(1)

    en_acts = load_activations(en_t_post_path)  # (n_prompts, n_layers, hidden)
    # Use the critical layer for refusal direction
    crit_layer = critical_layers[len(critical_layers) // 2]
    if en_acts.ndim == 3:
        en_layer_acts = en_acts[:, crit_layer, :].numpy()
    else:
        en_layer_acts = en_acts.numpy()

    # Split into refused/complied -- use simple heuristic or load from safety scores
    # For now use first half as "refused" proxy (English harmful -> refused)
    n = len(en_layer_acts)
    refused_acts = en_layer_acts[:n // 2]
    complied_acts = en_layer_acts[n // 2:]

    refusal_direction = extract_refusal_direction(args.model, refused_acts, complied_acts)

    # Save refusal direction
    np.save(str(output_dir / f"refusal_direction_{args.model}.npy"), refusal_direction)
    logger.info(f"Refusal direction saved: norm={np.linalg.norm(refusal_direction):.4f}")

    # Load harm subspaces and activations for all languages
    harm_subspaces = {}
    activations_t_inst = {}
    activations_t_post_inst = {}

    probe_summary_path = Path(args.probes_dir) / "probe_summary.csv"
    harm_categories = []
    if probe_summary_path.exists():
        probe_summary = pd.read_csv(probe_summary_path)
        harm_categories = [c for c in probe_summary["category"].unique() if c != "all"]

    from src.probing.subspace import build_subspace_from_probes

    for lang in languages:
        for layer in critical_layers:
            try:
                subspace = build_subspace_from_probes(
                    args.probes_dir, lang, layer, harm_categories
                )
                harm_subspaces[(lang, layer)] = subspace["W_l"]
            except Exception as e:
                logger.debug(f"No subspace for ({lang}, {layer}): {e}")

        for position, acts_dict in [
            ("last_instruction", activations_t_inst),
            ("last_post_instruction", activations_t_post_inst),
        ]:
            for label in ["harmful", "harmless"]:
                act_path = get_activation_path(
                    args.model, lang, args.perturbation, position, "residual",
                    args.activations_dir
                )
                if Path(act_path).exists():
                    try:
                        acts = load_activations(act_path)
                        if acts.ndim == 3:
                            acts_dict[(lang, label)] = acts[:, crit_layer, :].numpy()
                        else:
                            acts_dict[(lang, label)] = acts.numpy()
                    except Exception as e:
                        logger.warning(f"Error loading {act_path}: {e}")

    # Run disentanglement analysis
    logger.info("Running disentanglement analysis...")
    results_df = disentangle_analysis(
        harm_subspaces=harm_subspaces,
        refusal_direction=refusal_direction,
        activations_t_inst=activations_t_inst,
        activations_t_post_inst=activations_t_post_inst,
        layers=critical_layers,
    )

    results_df.to_csv(output_dir / "disentangle_results.csv", index=False)
    logger.info(f"Disentanglement analysis complete. Results saved to {output_dir}.")


if __name__ == "__main__":
    main()
