#!/usr/bin/env python3
"""
Script 12: Run Interventions

Apply all three interventions with parameter sweeps.

Usage:
    python scripts/12_run_interventions.py \
        --model llama \
        --config configs/experiment.yaml \
        --output-dir results/interventions/
"""

import argparse
import getpass
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.loader import load_dataset, format_for_model, get_split
from src.interventions.caa import compute_steering_vector
from src.interventions.sweep import run_full_sweep
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility


def parse_args():
    parser = argparse.ArgumentParser(description="Run all interventions.")
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "qwen"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/interventions/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--probes-dir", default="results/probes/")
    parser.add_argument("--sae-features-dir", default="results/sae_features/")
    parser.add_argument("--disentangle-dir", default="results/disentangle/")
    parser.add_argument("--trained-saes-dir", default="trained_saes/")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--hf-token", default=None,
                        help="Optional Hugging Face token. If omitted, the script will prompt for it.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("run_interventions")
    setup_reproducibility(seed=42)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        try:
            hf_token = getpass.getpass("Enter your Hugging Face token: ").strip()
        except Exception:
            hf_token = None

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        logger.info("HF token detected and exported.")
    else:
        logger.warning("No HF token provided. Gated model or dataset loading may fail.")

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]
    interv_cfg = config["experiment"].get("interventions", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset test split
    logger.info("Loading dataset test split...")
    df = load_dataset(
        dataset_dir=args.dataset_dir,
        perturbations=[args.perturbation],
    )
    test_df = get_split(df, split="test", seed=42, test_ratio=0.2)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_df = format_for_model(test_df, model_name, tokenizer)

    test_harmful = test_df[test_df["is_harmful"]].to_dict("records")
    test_benign = test_df[~test_df["is_harmful"]].to_dict("records")

    if args.dry_run:
        logger.info(f"[DRY RUN] Test harmful: {len(test_harmful)}, benign: {len(test_benign)}")
        return

    # Determine critical layer
    critical_layers_path = Path("results/attribution/critical_layers.json")
    if critical_layers_path.exists():
        with open(critical_layers_path) as f:
            critical_data = json.load(f)
        critical_layers = critical_data.get("critical_layers", [15])
        critical_layer = int(critical_layers[0]) if critical_layers else 15
    else:
        critical_layer = int(model_cfg.get("critical_layer_range", [12, 22])[0])

    logger.info(f"Using critical layer: {critical_layer}")

    # Load activations for steering vector
    from src.activations.cache import load_activations, get_activation_path
    en_act_path = get_activation_path(
        args.model, "en", args.perturbation, "last_post_instruction", "residual",
        args.activations_dir
    )

    interventions = {}

    # --- CAA ---
    en_layer = None
    if Path(en_act_path).exists():
        en_acts = load_activations(en_act_path)
        if en_acts.ndim == 3:
            if critical_layer >= en_acts.shape[1]:
                logger.warning(
                    f"Critical layer {critical_layer} out of bounds for English activations "
                    f"with shape {tuple(en_acts.shape)}. Skipping CAA."
                )
            else:
                en_layer = en_acts[:, critical_layer, :].float()
        else:
            en_layer = en_acts.float()

        if en_layer is not None:
            n = len(en_layer)
            steering_vec = compute_steering_vector(
                en_layer[:n // 2], en_layer[n // 2:], layer=critical_layer
            )

            interventions["caa"] = {
                "steering_vector": steering_vec,
                "alphas": interv_cfg.get("caa", {}).get("alpha_range", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                "layer": critical_layer,
            }
            logger.info("CAA steering vector prepared.")
    else:
        logger.warning(f"English activations not found: {en_act_path}. Skipping CAA.")

    # --- SAE Clamping ---
    sae_features_path = Path(args.sae_features_dir) / "ranked_features.json"
    en_feat_means_path = Path(args.sae_features_dir) / "en_feature_means.npy"
    if sae_features_path.exists() and en_feat_means_path.exists():
        with open(sae_features_path) as f:
            sae_data = json.load(f)

        ranked_features = [int(x) for x in sae_data.get("ranked_features", [])]
        sae_layer = int(sae_data.get("layer", critical_layer))
        hookpoint = sae_data.get("hookpoint", f"layers.{sae_layer}.mlp")

        en_feat_means = np.load(str(en_feat_means_path))
        clamp_values = {int(idx): float(en_feat_means[int(idx)]) for idx in ranked_features}

        if ".mlp" in hookpoint:
            hook_component = "mlp"
        elif ".attn" in hookpoint:
            hook_component = "attn"
        else:
            hook_component = "resid"

        try:
            from src.sae.feature_extract import load_sae
            sae = load_sae(model_name, sae_layer, hook_component=hook_component)

            interventions["sae_clamp"] = {
                "sae": sae,
                "ranked_features": ranked_features,
                "clamp_values": clamp_values,
                "layer": sae_layer,
                "counts": interv_cfg.get("sae_clamp", {}).get("top_features", [5, 10, 20, 50]),
            }
            logger.info(
                f"SAE clamping prepared (layer={sae_layer}, hook_component={hook_component})."
            )
        except Exception as e:
            logger.warning(f"Could not load SAE for clamping: {e}. Skipping SAE clamp.")
    else:
        logger.warning("SAE feature data not found. Skipping SAE clamping.")

    # --- Subspace Projection ---
    probe_summary_path = Path(args.probes_dir) / "probe_summary.csv"
    if probe_summary_path.exists() and en_layer is not None:
        try:
            from src.probing.subspace import build_subspace_from_probes
            from src.interventions.subspace_project import learn_subspace_map

            probe_summary = pd.read_csv(probe_summary_path)
            harm_cats = [c for c in probe_summary["category"].unique() if c != "all"]

            subspace = build_subspace_from_probes(
                args.probes_dir, "en", critical_layer, harm_cats
            )
            P = subspace["projection_matrix"]

            M_tiers = {}
            for lam in interv_cfg.get("subspace_projection", {}).get("regularization", [0.01]):
                en_np = en_layer.cpu().numpy() if hasattr(en_layer, "cpu") else np.array(en_layer)
                M = learn_subspace_map(en_np, en_np, P, regularization=lam)
                M_tiers[str(lam)] = M

            interventions["subspace_projection"] = {
                "M_tiers": M_tiers,
                "projection_matrix": P,
                "layer": critical_layer,
            }
            logger.info("Subspace projection maps prepared.")
        except Exception as e:
            logger.warning(f"Could not prepare subspace projection: {e}")

    # Load MMLU data for utility evaluation
    from src.evaluation.mmlu import load_multilingual_mmlu
    mmlu_data = load_multilingual_mmlu(languages=["EN-US"], n_samples=50)

    logger.info(f"Running sweep with {len(interventions)} interventions...")
    results_df = run_full_sweep(
        model_name=model_name,
        interventions=interventions,
        test_harmful=test_harmful[:40],
        test_benign=test_benign[:40],
        mmlu_data=mmlu_data,
        output_dir=str(output_dir),
    )

    logger.info(f"Intervention sweep complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()