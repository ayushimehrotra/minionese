"""
Attention Head-Level Causal Tracing

Extends attribution patching to individual attention heads at critical layers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _unwrap_saved(x):
    """Return the underlying tensor for either an NNsight saved object or a raw tensor."""
    return x.value if hasattr(x, "value") else x


def _patch_suffix_inplace(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Patch the shared right-aligned suffix from source into target.

    Assumes tensors have shape [batch, seq_len, hidden_or_head_dim].
    This is needed because English and translated prompts often tokenize
    to different sequence lengths.
    """
    if target.ndim != 3 or source.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got {target.shape} and {source.shape}")

    if target.shape[0] != source.shape[0] or target.shape[2] != source.shape[2]:
        raise ValueError(f"Incompatible shapes for patching: {target.shape} vs {source.shape}")

    shared_len = min(target.shape[1], source.shape[1])
    target[:, -shared_len:, :] = source[:, -shared_len:, :]


def _get_layers(lm):
    """Return transformer layers for different wrapper layouts."""
    try:
        return lm.model.layers
    except AttributeError:
        return lm.model.model.layers


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _refusal_metric(hidden: torch.Tensor, refusal_direction: np.ndarray) -> float:
    r = torch.tensor(refusal_direction, dtype=hidden.dtype, device=hidden.device)
    r = r / (r.norm() + 1e-12)
    return float((hidden @ r).mean().item())


def trace_attention_heads(
    model_name: str,
    en_prompts: List[str],
    langx_prompts: List[str],
    critical_layers: List[int],
    refusal_direction: np.ndarray,
    num_heads: Optional[int] = None,
    batch_size: int = 4,
    language: str = "unknown",
) -> pd.DataFrame:
    """
    Head-level causal tracing at critical layers.

    For each (critical_layer, head_idx), patch the clean head output
    into the corrupted run and measure refusal restoration.

    Args:
        model_name: HuggingFace model name.
        en_prompts: English (clean) prompts.
        langx_prompts: Matched LangX (corrupted) prompts.
        critical_layers: Layer indices to trace.
        refusal_direction: Refusal direction vector.
        num_heads: Number of attention heads (auto-detected if None).
        batch_size: Prompts per batch.
        language: Language code for labeling.

    Returns:
        DataFrame with columns: layer, head_idx, language, restoration_score.
    """
    from nnsight import LanguageModel
    from transformers import AutoConfig, AutoTokenizer
    from src.utils.gpu import clear_gpu_memory, log_gpu_memory

    assert len(en_prompts) == len(langx_prompts), "Prompt lists must be aligned."

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if num_heads is None:
        num_heads = getattr(config, "num_attention_heads", 32)
    hidden_dim = getattr(config, "hidden_size", 4096)
    head_dim = hidden_dim // num_heads

    device = _get_device()

    # Important:
    # Avoid device_map="auto" and avoid lm.to(device) afterward, since that can
    # trigger meta-tensor issues with NNsight.
    lm = LanguageModel(
        model_name,
        device_map=device,
        dtype=torch.float16,
        dispatch=True,
    )

    layers = _get_layers(lm)
    num_model_layers = len(layers)
    valid_critical_layers = sorted({l for l in critical_layers if 0 <= l < num_model_layers})

    log_gpu_memory("attention head tracing: model loaded")

    def _get_final_hidden(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            with lm.trace(**inputs):
                layers_local = _get_layers(lm)
                final_out = layers_local[-1].output
                if isinstance(final_out, tuple):
                    final_out = final_out[0]
                final_h = final_out[:, -1, :].save()
        return _unwrap_saved(final_h)

    rows = []

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

        # Cache clean attention outputs at critical layers
        clean_attn: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            with lm.trace(**en_inputs):
                layers_local = _get_layers(lm)
                for l in valid_critical_layers:
                    try:
                        attn_out = layers_local[l].self_attn.output
                        if isinstance(attn_out, tuple):
                            attn_out = attn_out[0]
                        clean_attn[l] = attn_out.save()
                    except Exception as e:
                        logger.warning(f"Skipping layer {l} clean attention cache: {e}")

        # Baseline metrics
        h_clean = _get_final_hidden(en_inputs)
        h_corrupted = _get_final_hidden(langx_inputs)

        metric_clean = _refusal_metric(h_clean, refusal_direction)
        metric_corrupted = _refusal_metric(h_corrupted, refusal_direction)
        baseline_diff = metric_clean - metric_corrupted

        # Head-level patching
        for layer in valid_critical_layers:
            if layer not in clean_attn:
                continue

            clean_attn_val = _unwrap_saved(clean_attn[layer])  # [batch, seq, hidden]

            for head_idx in range(num_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                with torch.no_grad():
                    with lm.trace(**langx_inputs):
                        layers_local = _get_layers(lm)
                        attn_out = layers_local[layer].self_attn.output
                        if isinstance(attn_out, tuple):
                            attn_out = attn_out[0]

                        target_slice = attn_out[:, :, start:end]
                        source_slice = clean_attn_val[:, :, start:end]

                        _patch_suffix_inplace(target_slice, source_slice)

                        final_out = layers_local[-1].output
                        if isinstance(final_out, tuple):
                            final_out = final_out[0]
                        final = final_out[:, -1, :].save()

                metric_patched = _refusal_metric(_unwrap_saved(final), refusal_direction)

                if abs(baseline_diff) > 1e-6:
                    restoration = (metric_patched - metric_corrupted) / baseline_diff
                else:
                    restoration = 0.0

                rows.append(
                    {
                        "layer": layer,
                        "head_idx": head_idx,
                        "language": language,
                        "restoration_score": round(float(restoration), 4),
                    }
                )

        logger.info(
            f"Head tracing batch {i // batch_size + 1}/"
            f"{(len(en_prompts) + batch_size - 1) // batch_size} done."
        )

    del lm
    clear_gpu_memory()
    return pd.DataFrame(rows)


def extract_attention_patterns(
    model_name: str,
    prompts: List[str],
    critical_layers: List[int],
    critical_heads: List[Tuple[int, int]],
    batch_size: int = 4,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extract attention weight matrices for identified critical heads.

    Args:
        model_name: HuggingFace model name.
        prompts: List of formatted prompt strings.
        critical_layers: Layer indices.
        critical_heads: List of (layer, head_idx) tuples.
        batch_size: Prompts per batch.

    Returns:
        Dict mapping (layer, head_idx) -> mean attention weights (seq, seq).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.utils.gpu import clear_gpu_memory

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = _get_device()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        output_attentions=True,
        trust_remote_code=True,
    )
    model.eval()

    attn_accum: Dict[Tuple[int, int], List[np.ndarray]] = {k: [] for k in critical_heads}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # tuple: (layer -> [batch, heads, seq, seq])

        for layer, head_idx in critical_heads:
            if layer >= len(attentions):
                continue

            layer_attn = attentions[layer]
            if head_idx >= layer_attn.shape[1]:
                continue

            attn_matrix = layer_attn[:, head_idx, :, :].mean(0).float().cpu().numpy()
            attn_accum[(layer, head_idx)].append(attn_matrix)

    del model
    clear_gpu_memory()

    result: Dict[Tuple[int, int], np.ndarray] = {}
    for key, matrices in attn_accum.items():
        if matrices:
            result[key] = np.mean(matrices, axis=0)

    return result