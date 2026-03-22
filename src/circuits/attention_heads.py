"""
Attention Head-Level Causal Tracing (NEW)

Extends attribution patching to individual attention heads at critical layers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


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
    from transformers import AutoTokenizer, AutoConfig
    from src.utils.gpu import clear_gpu_memory

    assert len(en_prompts) == len(langx_prompts)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if num_heads is None:
        num_heads = getattr(config, "num_attention_heads", 32)
    hidden_dim = getattr(config, "hidden_size", 4096)
    head_dim = hidden_dim // num_heads

    lm = LanguageModel(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    try:
        num_model_layers = lm.config.num_hidden_layers
    except AttributeError:
        num_model_layers = len(lm.model.layers)

    def _get_final_hidden(model, input_ids):
        with torch.no_grad():
            with model.trace(input_ids) as tracer:
                try:
                    h = model.model.layers[-1].output[0][:, -1, :].save()
                except AttributeError:
                    h = model.model.model.layers[-1].output[0][:, -1, :].save()
        return h.value

    def _refusal_metric(h: torch.Tensor) -> float:
        r = torch.tensor(refusal_direction, dtype=h.dtype, device=h.device)
        r = r / (r.norm() + 1e-12)
        return float((h @ r).mean().item())

    rows = []

    for i in range(0, len(en_prompts), batch_size):
        en_batch = en_prompts[i : i + batch_size]
        langx_batch = langx_prompts[i : i + batch_size]

        en_inputs = tokenizer(
            en_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )["input_ids"].to(lm.device)
        langx_inputs = tokenizer(
            langx_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )["input_ids"].to(lm.device)

        # Cache clean attention outputs at critical layers
        clean_attn: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            with lm.trace(en_inputs) as tracer:
                for l in critical_layers:
                    try:
                        layer_obj = lm.model.layers[l]
                    except AttributeError:
                        layer_obj = lm.model.model.layers[l]
                    clean_attn[l] = layer_obj.self_attn.output[0].save()

        # Baseline metrics
        h_clean = _get_final_hidden(lm, en_inputs)
        h_corrupted = _get_final_hidden(lm, langx_inputs)
        metric_clean = _refusal_metric(h_clean)
        metric_corrupted = _refusal_metric(h_corrupted)
        baseline_diff = metric_clean - metric_corrupted

        # Head-level patching
        for layer in critical_layers:
            if layer not in clean_attn:
                continue
            clean_attn_val = clean_attn[layer].value  # (batch, seq, hidden)

            for head_idx in range(num_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim

                with torch.no_grad():
                    with lm.trace(langx_inputs) as tracer:
                        try:
                            layer_obj = lm.model.layers[layer]
                        except AttributeError:
                            layer_obj = lm.model.model.layers[layer]

                        attn_out = layer_obj.self_attn.output[0]
                        # Patch only this head's slice
                        attn_out[:, :, start:end] = clean_attn_val[:, :, start:end]

                        try:
                            final = lm.model.layers[-1].output[0][:, -1, :].save()
                        except AttributeError:
                            final = lm.model.model.layers[-1].output[0][:, -1, :].save()

                metric_patched = _refusal_metric(final.value)
                if abs(baseline_diff) > 1e-6:
                    restoration = (metric_patched - metric_corrupted) / baseline_diff
                else:
                    restoration = 0.0

                rows.append({
                    "layer": layer,
                    "head_idx": head_idx,
                    "language": language,
                    "restoration_score": round(restoration, 4),
                })

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_attentions=True,
        trust_remote_code=True,
    )
    model.eval()

    attn_accum: Dict[Tuple[int, int], List[np.ndarray]] = {k: [] for k in critical_heads}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer

        for layer, head_idx in critical_heads:
            if layer < len(attentions):
                attn_matrix = attentions[layer][:, head_idx, :, :].mean(0).cpu().numpy()
                attn_accum[(layer, head_idx)].append(attn_matrix)

    del model
    clear_gpu_memory()

    result = {}
    for key, matrices in attn_accum.items():
        if matrices:
            result[key] = np.mean(matrices, axis=0)
    return result
