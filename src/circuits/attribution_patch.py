"""
Attribution Patching: Cross-Lingual Refusal Failure Localization

Identify which layers mediate cross-lingual refusal failure using
clean (English) vs. corrupted (LangX) patching.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _patch_suffix_inplace(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Patch the shared right-aligned suffix from source into target.
    Assumes tensors have shape [batch, seq_len, hidden].
    """
    if target.ndim != 3 or source.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got {target.shape} and {source.shape}")

    if target.shape[0] != source.shape[0] or target.shape[2] != source.shape[2]:
        raise ValueError(f"Incompatible shapes for patching: {target.shape} vs {source.shape}")

    shared_len = min(target.shape[1], source.shape[1])
    target[:, -shared_len:, :] = source[:, -shared_len:, :]

def _unwrap_saved(x):
    return x.value if hasattr(x, "value") else x

def _compute_refusal_activation(hidden: torch.Tensor, refusal_direction: np.ndarray) -> float:
    """Compute mean dot product of hidden states with refusal direction."""
    r = torch.tensor(refusal_direction, dtype=hidden.dtype, device=hidden.device)
    r = r / (r.norm() + 1e-12)
    return float((hidden @ r).mean().item())


def _get_layers(lm):
    """Return the transformer block list for different model wrappers."""
    try:
        return lm.model.layers
    except AttributeError:
        return lm.model.model.layers


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Do not call lm.to(device) afterward; that can trigger meta-tensor issues.
    lm = LanguageModel(
        model_name,
        device_map=device,
        dtype=torch.float16,
        dispatch=True,
    )

    layers = _get_layers(lm)
    num_layers = len(layers)

    log_gpu_memory("attribution patching: model loaded")

    restoration_scores: Dict = {
        (layer, comp): [] for layer in range(num_layers) for comp in components
    }

    for i in range(0, len(en_prompts), batch_size):
        en_batch = en_prompts[i : i + batch_size]
        langx_batch = langx_prompts[i : i + batch_size]

        en_inputs = tokenizer(
            en_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        langx_inputs = tokenizer(
            langx_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        en_inputs = {k: v.to(device) for k, v in en_inputs.items()}
        langx_inputs = {k: v.to(device) for k, v in langx_inputs.items()}

        clean_residuals = {}
        clean_attn_outs = {}
        clean_mlp_outs = {}

        # Step 1: clean forward pass, cache activations
        with torch.no_grad():
            with lm.trace(**en_inputs):
                layers = _get_layers(lm)
                for l in range(num_layers):
                    layer_obj = layers[l]

                    # IMPORTANT: save in forward order
                    # self_attn -> mlp -> layer output
                    if "attn_out" in components:
                        try:
                            attn_out = layer_obj.self_attn.output
                            if isinstance(attn_out, tuple):
                                attn_out = attn_out[0]
                            clean_attn_outs[l] = attn_out.save()
                        except Exception:
                            pass

                    if "mlp_out" in components:
                        try:
                            mlp_out = layer_obj.mlp.output
                            if isinstance(mlp_out, tuple):
                                mlp_out = mlp_out[0]
                            clean_mlp_outs[l] = mlp_out.save()
                        except Exception:
                            pass

                    if "residual" in components:
                        try:
                            residual_out = layer_obj.output
                            if isinstance(residual_out, tuple):
                                residual_out = residual_out[0]
                            clean_residuals[l] = residual_out.save()
                        except Exception:
                            pass

        # Clean metric
        with torch.no_grad():
            with lm.trace(**en_inputs):
                layers = _get_layers(lm)
                final_out = layers[-1].output
                if isinstance(final_out, tuple):
                    final_out = final_out[0]
                final_h_clean = final_out[:, -1, :].save()

        metric_clean = _compute_refusal_activation(_unwrap_saved(final_h_clean), refusal_direction)

        # Corrupted metric
        with torch.no_grad():
            with lm.trace(**langx_inputs):
                layers = _get_layers(lm)
                final_out = layers[-1].output
                if isinstance(final_out, tuple):
                    final_out = final_out[0]
                final_h_corrupted = final_out[:, -1, :].save()

        metric_corrupted = _compute_refusal_activation(_unwrap_saved(final_h_corrupted), refusal_direction)
        baseline_diff = metric_clean - metric_corrupted

        # Step 2: patch each layer/component into corrupted run
        for patch_layer in range(num_layers):
            for comp in components:
                if comp == "residual" and patch_layer not in clean_residuals:
                    continue
                if comp == "attn_out" and patch_layer not in clean_attn_outs:
                    continue
                if comp == "mlp_out" and patch_layer not in clean_mlp_outs:
                    continue

                with torch.no_grad():
                    with lm.trace(**langx_inputs):
                        layers = _get_layers(lm)
                        layer_obj = layers[patch_layer]

                        if comp == "residual":
                            residual_out = layer_obj.output
                            if isinstance(residual_out, tuple):
                                residual_out = residual_out[0]
                            _patch_suffix_inplace(residual_out, _unwrap_saved(clean_residuals[patch_layer]))

                        elif comp == "attn_out":
                            attn_out = layer_obj.self_attn.output
                            if isinstance(attn_out, tuple):
                                attn_out = attn_out[0]
                            _patch_suffix_inplace(attn_out, _unwrap_saved(clean_attn_outs[patch_layer]))

                        elif comp == "mlp_out":
                            mlp_out = layer_obj.mlp.output
                            if isinstance(mlp_out, tuple):
                                mlp_out = mlp_out[0]
                            _patch_suffix_inplace(mlp_out, _unwrap_saved(clean_mlp_outs[patch_layer]))

                        final_out = layers[-1].output
                        if isinstance(final_out, tuple):
                            final_out = final_out[0]
                        final_h_patched = final_out[:, -1, :].save()

                metric_patched = _compute_refusal_activation(_unwrap_saved(final_h_patched), refusal_direction)

                if abs(baseline_diff) > 1e-6:
                    restoration = (metric_patched - metric_corrupted) / baseline_diff
                else:
                    restoration = 0.0

                restoration_scores[(patch_layer, comp)].append(float(restoration))

        logger.info(
            f"Batch {i // batch_size + 1}/{(len(en_prompts) + batch_size - 1) // batch_size} done."
        )

    del lm
    clear_gpu_memory()

    rows = []
    for (layer, comp), scores in restoration_scores.items():
        if not scores:
            continue
        rows.append(
            {
                "layer": layer,
                "component": comp,
                "language": language,
                "tier": tier,
                "mean_restoration": round(float(np.mean(scores)), 4),
                "std_restoration": round(float(np.std(scores)), 4),
            }
        )

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