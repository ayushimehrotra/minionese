#!/usr/bin/env python3
"""
Script 07: Interventions + Evaluation

Runs all applicable interventions with parameter sweeps, evaluates
safety (ASR on harmful) and utility (over-refusal on harmless),
computes Pareto frontier, and produces figures.

COHERENCE FILTERING: Only evaluates on languages where the model
generates coherently. There is no point applying steering vectors
to a model that outputs gibberish in the target language.

Interventions:
  - CAA (all models, coherent languages only)
  - SAE clamping (Llama, Qwen only; skipped if has_sae=false)
  - Subspace projection (all models, coherent languages only)

Produces:
  - results/interventions/{model}/sweep_results.csv
  - results/interventions/{model}/pareto_frontier.csv
  - figures/fig7_pareto_{model}.{pdf,png}

Usage:
    python scripts/07_interventions.py --model llama
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
from src.interventions.sweep import run_full_sweep, compute_pareto_frontier
from src.utils.coherence_filter import filter_languages
from src.utils.config import load_config, get_model_config
from src.utils.logging_setup import setup_logging
from src.utils.reproducibility import setup_reproducibility
from src.visualization.pareto import plot_pareto_frontier


def parse_args():
    parser = argparse.ArgumentParser(description="Run interventions (coherent languages only).")
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "aya"])
    parser.add_argument("--output-dir", default="results/interventions/")
    parser.add_argument("--dataset-dir", default="dataset/")
    parser.add_argument("--activations-dir", default="data/activations/")
    parser.add_argument("--probes-dir", default=None,
                        help="Probes directory. Defaults to results/representation/{model}/probes/.")
    parser.add_argument("--sae-features-dir", default="results/sae_features/")
    parser.add_argument("--representation-dir", default=None,
                        help="Dir with refusal_direction_{model}.npy. "
                             "Defaults to results/representation/{model}/.")
    parser.add_argument("--perturbation", default="standard_translation")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_hf_token(cli_token):
    token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        try:
            token = getpass.getpass("Enter your Hugging Face token: ").strip()
        except Exception:
            token = None
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("interventions")
    setup_reproducibility(seed=42)

    hf_token = resolve_hf_token(args.hf_token)

    config = load_config()
    model_cfg = get_model_config(config, args.model)
    model_name = model_cfg["name"]
    interv_cfg = config.get("experiment", {}).get("interventions", {})

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    probes_dir = Path(args.probes_dir) if args.probes_dir else Path(f"results/representation/{args.model}/probes")
    repr_dir = Path(args.representation_dir) if args.representation_dir else Path(f"results/representation/{args.model}")

    # Load dataset test split
    logger.info("Loading dataset test split...")
    df = load_dataset(dataset_dir=args.dataset_dir, perturbations=[args.perturbation])
    test_df = get_split(df, split="test", seed=42, test_ratio=0.2)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    test_df = format_for_model(test_df, model_name, tokenizer)

    # Coherence filtering: only intervene on languages where model generates coherently
    all_test_langs = test_df["language"].unique().tolist()
    coherent_langs = filter_languages(all_test_langs, args.model)
    test_df = test_df[test_df["language"].isin(coherent_langs)].reset_index(drop=True)
    logger.info(
        f"Intervention evaluation on {len(coherent_langs)} coherent languages "
        f"for {args.model}: {coherent_langs}"
    )

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
        cr = model_cfg.get("critical_layer_range", [12, 22])
        critical_layer = int(cr[0])
    logger.info(f"Using critical layer: {critical_layer}")

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
        if en_acts.ndim == 3 and critical_layer < en_acts.shape[1]:
            en_layer = en_acts[:, critical_layer, :].float()
        elif en_acts.ndim == 2:
            en_layer = en_acts.float()
    else:
        logger.warning(f"English activations not found: {en_act_path}. Computing on-the-fly...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer as AT

            en_train_df = load_dataset(dataset_dir=args.dataset_dir, perturbations=[args.perturbation])
            en_train_df = get_split(en_train_df, split="train", seed=42, test_ratio=0.2)
            en_train_df = en_train_df[en_train_df["language"] == "en"].reset_index(drop=True)
            en_train_df = format_for_model(en_train_df, model_name, tokenizer)

            en_harmful_prompts = en_train_df[en_train_df["is_harmful"]]["prompt"].tolist()[:80]
            en_benign_prompts = en_train_df[~en_train_df["is_harmful"]]["prompt"].tolist()[:80]
            all_en_prompts = en_harmful_prompts + en_benign_prompts

            logger.info(f"Extracting activations for {len(all_en_prompts)} English prompts at layer {critical_layer}...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, token=hf_token
            )
            _tok = AT.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
            if _tok.pad_token is None:
                _tok.pad_token = _tok.eos_token
            _tok.padding_side = "left"
            _model.eval()

            collected = []
            bs = 8
            for i in range(0, len(all_en_prompts), bs):
                batch = all_en_prompts[i:i + bs]
                enc = _tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(_model.device)
                hooks, handles = [], []
                def _hook(mod, inp, out, _store=hooks):
                    h = out[0] if isinstance(out, tuple) else out
                    _store.append(h[:, -1, :].float().detach().cpu())
                layer_mod = _model.model.layers[critical_layer]
                handles.append(layer_mod.register_forward_hook(_hook))
                with torch.no_grad():
                    _model(**enc)
                for h in handles:
                    h.remove()
                if hooks:
                    collected.append(hooks[0])

            if collected:
                en_layer = torch.cat(collected, dim=0)
                logger.info(f"On-the-fly activations: shape={en_layer.shape}")

            del _model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"On-the-fly activation extraction failed: {e}. Skipping CAA.")

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

    # --- SAE Clamping (only if model has SAE) ---
    if model_cfg.get("has_sae", False):
        sae_features_path = Path(args.sae_features_dir) / "ranked_features.json"
        en_feat_means_path = Path(args.sae_features_dir) / "en_feature_means.npy"
        if sae_features_path.exists() and en_feat_means_path.exists():
            with open(sae_features_path) as f:
                sae_data = json.load(f)
            ranked_features = [int(x) for x in sae_data.get("ranked_features", [])]
            sae_layer = int(sae_data.get("layer", critical_layer))
            hookpoint = sae_data.get("hookpoint", f"layers.{sae_layer}.mlp")
            hook_component = "mlp" if ".mlp" in hookpoint else ("attn" if ".attn" in hookpoint else "resid")
            en_feat_means = np.load(str(en_feat_means_path))
            clamp_values = {int(idx): float(en_feat_means[int(idx)]) for idx in ranked_features}
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
                logger.info(f"SAE clamping prepared (layer={sae_layer}).")
            except Exception as e:
                logger.warning(f"Could not load SAE for clamping: {e}. Skipping SAE clamp.")
        else:
            logger.warning("SAE feature data not found. Skipping SAE clamping.")
    else:
        logger.info(f"Skipping SAE clamping for {args.model} (no SAE).")

    # --- Subspace Projection ---
    probe_summary_path = probes_dir / "probe_summary.csv"
    if probe_summary_path.exists() and en_layer is not None:
        try:
            from src.probing.subspace import build_subspace_from_probes
            from src.interventions.subspace_project import learn_subspace_map

            probe_summary = pd.read_csv(probe_summary_path)
            harm_cats = [c for c in probe_summary["category"].unique() if c != "all"]
            subspace = build_subspace_from_probes(str(probes_dir), "en", critical_layer, harm_cats)
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

    logger.info(f"Running sweep with {len(interventions)} interventions...")
    results_df = run_full_sweep(
        model_name=model_name,
        interventions=interventions,
        test_harmful=test_harmful[:40],
        test_benign=test_benign[:40],
        output_dir=str(output_dir),
    )

    # Pareto frontier + figure (inline — no separate evaluation script)
    logger.info("Computing Pareto frontier...")
    pareto_df = compute_pareto_frontier(results_df)
    pareto_df.to_csv(output_dir / "pareto_frontier.csv", index=False)

    results_df["is_pareto_optimal"] = results_df.index.isin(pareto_df.index)

    try:
        figures_dir = Path("figures/")
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_pareto_frontier(results_df, str(figures_dir / f"fig7_pareto_{args.model}"))
        logger.info("Figure 7 (Pareto frontier) saved.")
    except Exception as e:
        logger.warning(f"Pareto plot failed: {e}")

    logger.info(f"Interventions complete. Results in {output_dir}.")


if __name__ == "__main__":
    main()
