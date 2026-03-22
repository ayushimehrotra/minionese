"""
Attribution Patching: Cross-Lingual Refusal Failure Localization

Identify which layers mediate cross-lingual refusal failure using
clean (English) vs. corrupted (LangX) patching.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _compute_refusal_activation(hidden: torch.Tensor, refusal_direction: np.ndarray) -> float:
    """Compute dot product of hidden state with refusal direction."""
    r = torch.tensor(refusal_direction, dtype=hidden.dtype, device=hidden.device)
    r = r / (r.norm() + 1e-12)
    return float((hidden @ r).mean().item())


def run_attribution_patching(
    model_name: str,
    en_prompts: List[str],
    langx_prompts: List[str],
    refusal_direction: np.ndarray,
    components: List[str] = None,
    batch_size: int = 4,
    language: str = "unknown",
    tier: str = "unknown",
) -> pd.DataFrame:
    """
    Layer-level attribution patching.

    For each layer, patch the corrupted (LangX) activation with the clean (EN)
    activation and measure refusal restoration.

    Args:
        model_name: HuggingFace model name.
        en_prompts: English (clean) prompts.
        langx_prompts: Matched LangX (corrupted) prompts.
        refusal_direction: Refusal direction vector (hidden_dim,).
        components: Components to patch. Default: ["residual", "attn_out", "mlp_out"].
        batch_size: Prompts per batch.
        language: Language code (for output labeling).
        tier: Language tier (for output labeling).

    Returns:
        DataFrame with columns: layer, component, language, tier,
                                mean_restoration, std_restoration.
    """
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    from src.utils.gpu import clear_gpu_memory, log_gpu_memory

    if components is None:
        components = ["residual", "attn_out", "mlp_out"]

    assert len(en_prompts) == len(langx_prompts), "Prompt lists must be aligned."

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lm = LanguageModel(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    try:
        num_layers = lm.config.num_hidden_layers
    except AttributeError:
        num_layers = len(lm.model.layers)

    log_gpu_memory("attribution patching: model loaded")

    restoration_scores: Dict = {
        (layer, comp): [] for layer in range(num_layers) for comp in components
    }

    for i in range(0, len(en_prompts), batch_size):
        en_batch = en_prompts[i : i + batch_size]
        langx_batch = langx_prompts[i : i + batch_size]

        en_inputs = tokenizer(
            en_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        langx_inputs = tokenizer(
            langx_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )

        # Step 1: Clean forward pass -- cache all activations
        clean_residuals = {}
        clean_attn_outs = {}
        clean_mlp_outs = {}

        with torch.no_grad():
            with lm.trace(en_inputs["input_ids"].to(lm.device)) as tracer:
                for l in range(num_layers):
                    try:
                        layer_obj = lm.model.layers[l]
                    except AttributeError:
                        layer_obj = lm.model.model.layers[l]

                    if "residual" in components:
                        clean_residuals[l] = layer_obj.output[0].save()
                    if "attn_out" in components:
                        try:
                            clean_attn_outs[l] = layer_obj.self_attn.output[0].save()
                        except AttributeError:
                            pass
                    if "mlp_out" in components:
                        try:
                            mlp_out = layer_obj.mlp.output
                            if isinstance(mlp_out, tuple):
                                mlp_out = mlp_out[0]
                            clean_mlp_outs[l] = mlp_out.save()
                        except AttributeError:
                            pass

        # Measure metric on clean run (metric_clean)
        with torch.no_grad():
            with lm.trace(en_inputs["input_ids"].to(lm.device)) as tracer:
                try:
                    final_h_clean = lm.model.layers[-1].output[0][:, -1, :].save()
                except AttributeError:
                    final_h_clean = lm.model.model.layers[-1].output[0][:, -1, :].save()

        metric_clean = _compute_refusal_activation(final_h_clean.value, refusal_direction)

        # Measure metric on corrupted run (metric_corrupted, no patching)
        with torch.no_grad():
            with lm.trace(langx_inputs["input_ids"].to(lm.device)) as tracer:
                try:
                    final_h_corrupted = lm.model.layers[-1].output[0][:, -1, :].save()
                except AttributeError:
                    final_h_corrupted = lm.model.model.layers[-1].output[0][:, -1, :].save()

        metric_corrupted = _compute_refusal_activation(final_h_corrupted.value, refusal_direction)
        baseline_diff = metric_clean - metric_corrupted

        # Step 2: Patch each layer
        for patch_layer in range(num_layers):
            for comp in components:
                if comp == "residual" and patch_layer not in clean_residuals:
                    continue
                if comp == "attn_out" and patch_layer not in clean_attn_outs:
                    continue
                if comp == "mlp_out" and patch_layer not in clean_mlp_outs:
                    continue

                with torch.no_grad():
                    with lm.trace(langx_inputs["input_ids"].to(lm.device)) as tracer:
                        try:
                            layer_obj = lm.model.layers[patch_layer]
                        except AttributeError:
                            layer_obj = lm.model.model.layers[patch_layer]

                        if comp == "residual":
                            layer_obj.output[0][:] = clean_residuals[patch_layer].value
                        elif comp == "attn_out":
                            layer_obj.self_attn.output[0][:] = clean_attn_outs[patch_layer].value
                        elif comp == "mlp_out":
                            mlp_out = layer_obj.mlp.output
                            if isinstance(mlp_out, tuple):
                                layer_obj.mlp.output[0][:] = clean_mlp_outs[patch_layer].value
                            else:
                                layer_obj.mlp.output[:] = clean_mlp_outs[patch_layer].value

                        try:
                            final_h_patched = lm.model.layers[-1].output[0][:, -1, :].save()
                        except AttributeError:
                            final_h_patched = lm.model.model.layers[-1].output[0][:, -1, :].save()

                metric_patched = _compute_refusal_activation(final_h_patched.value, refusal_direction)
                if abs(baseline_diff) > 1e-6:
                    restoration = (metric_patched - metric_corrupted) / baseline_diff
                else:
                    restoration = 0.0

                restoration_scores[(patch_layer, comp)].append(restoration)

        logger.info(
            f"Batch {i // batch_size + 1}/{(len(en_prompts) + batch_size - 1) // batch_size} done."
        )

    del lm
    clear_gpu_memory()

    rows = []
    for (layer, comp), scores in restoration_scores.items():
        if not scores:
            continue
        rows.append({
            "layer": layer,
            "component": comp,
            "language": language,
            "tier": tier,
            "mean_restoration": round(float(np.mean(scores)), 4),
            "std_restoration": round(float(np.std(scores)), 4),
        })

    return pd.DataFrame(rows)


def aggregate_by_tier(results: pd.DataFrame) -> pd.DataFrame:
    """
    Average restoration scores within each language tier.

    Args:
        results: DataFrame from run_attribution_patching().

    Returns:
        Aggregated DataFrame per tier.
    """
    group_cols = [c for c in ["tier", "layer", "component"] if c in results.columns]
    if not group_cols:
        return results

    agg = (
        results.groupby(group_cols)
        .agg(
            mean_restoration=("mean_restoration", "mean"),
            std_restoration=("std_restoration", "mean"),
        )
        .reset_index()
    )
    return agg
