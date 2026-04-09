"""
Attribution Patching: Cross-Lingual Refusal Failure Localization

Two analysis modes (per the paper design):

  Analysis A (normalize=False, default):
    For every (EN, LangX) pair, compute the RAW change in refusal-direction
    projection when the EN activation at t_inst is patched into layer L of
    the corrupted (LangX) run:

        delta_L = metric_patched - metric_corrupted

    Aggregated across all pairs per language.  Large positive delta => layer L
    matters for routing the refusal pathway.  No division by baseline_diff,
    so near-zero baseline is harmless.

  Analysis B (normalize=True):
    First filter to behaviorally-contrastive pairs where EN is refused AND
    LangX complies (|baseline_diff| >= min_baseline_diff).  For these pairs
    compute the normalized restoration score:

        restoration_L = (metric_patched - metric_corrupted) / baseline_diff

    Reports n_pairs_used and n_pairs_skipped so pair counts are visible.

Patching position:
    Activations are patched at the t_inst token (last token of the user
    instruction, i.e. the token just before the final <|eot_id|>).  The
    refusal metric is measured at the LAST token before generation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_saved(x):
    return x.value if hasattr(x, "value") else x


def _get_layers(lm):
    try:
        return lm.model.layers
    except AttributeError:
        return lm.model.model.layers


def _find_positions_in_batch(
    input_ids: torch.Tensor,
    pad_id: int,
    model_name: str,
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """
    For each sample in a left-padded batch, return:
        t_inst_indices  – position of the last token of the user instruction
                          (token just before the final end-of-user-turn token)
        last_indices    – position of the last real token (before generation)

    Works directly on token IDs; no character-level text scanning.
    """
    seq_len = input_ids.shape[1]
    model_lower = model_name.lower()

    if "gemma" in model_lower:
        eot_token_ids = set(tokenizer.convert_tokens_to_ids(["<end_of_turn>"]))
    elif "qwen" in model_lower:
        eot_token_ids = set(tokenizer.convert_tokens_to_ids(["<|im_end|>"]))
    else:
        eot_token_ids = set(tokenizer.convert_tokens_to_ids(["<|eot_id|>"]))
    # Remove unk
    eot_token_ids = {i for i in eot_token_ids if i != getattr(tokenizer, "unk_token_id", -1)}

    t_inst_list = []
    last_list = []

    for sample_idx in range(input_ids.shape[0]):
        ids = input_ids[sample_idx]
        pad_count = int((ids == pad_id).sum().item())
        first_real = pad_count
        last_real = seq_len - 1

        # t_inst: token just before the LAST end-of-user-turn token
        last_eot = -1
        for t in range(last_real, first_real - 1, -1):
            if ids[t].item() in eot_token_ids:
                last_eot = t
                break
        t_inst = max(first_real, last_eot - 1) if last_eot > first_real else last_real
        t_inst_list.append(t_inst)
        last_list.append(last_real)

    return t_inst_list, last_list


def _gather_position(tensor: torch.Tensor, positions: List[int]) -> torch.Tensor:
    """
    Extract one vector per sample at the given per-sample position index.

    tensor: (batch, seq, hidden)
    positions: list of length batch
    returns: (batch, hidden)
    """
    idx = torch.tensor(positions, device=tensor.device).view(-1, 1, 1)
    idx = idx.expand(-1, 1, tensor.shape[-1])   # (batch, 1, hidden)
    return tensor.gather(dim=1, index=idx).squeeze(1)   # (batch, hidden)


def _compute_refusal_metric(
    hidden: torch.Tensor,
    refusal_direction: np.ndarray,
) -> float:
    """Mean dot-product of (batch, hidden) hidden states with the refusal direction."""
    r = torch.tensor(refusal_direction, dtype=hidden.dtype, device=hidden.device)
    r = r / (r.norm() + 1e-12)
    return float((hidden @ r).mean().item())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_attribution_patching(
    model_name: str,
    en_prompts: List[str],
    langx_prompts: List[str],
    refusal_direction: np.ndarray,
    components: Optional[List[str]] = None,
    batch_size: int = 4,
    language: str = "unknown",
    tier: str = "unknown",
    perturbation: str = "unknown",
    normalize: bool = False,
    min_baseline_diff: float = 0.01,
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Layer-level attribution patching with position-specific t_inst patching.

    Args:
        model_name: HuggingFace model name.
        en_prompts: English (clean) prompts.
        langx_prompts: Matched LangX (corrupted) prompts.
        refusal_direction: Refusal direction vector (hidden_dim,).
        components: Components to patch. Default: ["residual", "attn_out", "mlp_out"].
        batch_size: Prompts per batch.
        language: Language code (for output labeling).
        tier: Language tier (for output labeling).
        perturbation: Perturbation type (for output labeling).
        normalize: If False (Analysis A), report raw delta across all pairs.
                   If True (Analysis B), report normalized restoration score for
                   behaviorally-contrastive pairs only.
        min_baseline_diff: Minimum |baseline_diff| to include a pair in
                           Analysis B (ignored when normalize=False).
        hf_token: Optional Hugging Face token for gated model access.

    Returns:
        DataFrame with columns: layer, component, language, tier, perturbation,
            mean_delta (Analysis A) or mean_restoration (Analysis B),
            std_delta / std_restoration, n_pairs_used, n_pairs_skipped.
    """
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    from src.utils.gpu import clear_gpu_memory, log_gpu_memory

    if components is None:
        components = ["residual", "attn_out", "mlp_out"]

    assert len(en_prompts) == len(langx_prompts), "Prompt lists must be aligned."

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lm = LanguageModel(
        model_name,
        device_map=device,
        dtype=torch.float16,
        dispatch=True,
        token=hf_token,
    )

    layer_list = _get_layers(lm)
    num_layers = len(layer_list)

    log_gpu_memory("attribution patching: model loaded")

    # Accumulators: (layer, component) -> list of per-pair scores
    scores: Dict[Tuple[int, str], List[float]] = {
        (l, c): [] for l in range(num_layers) for c in components
    }
    n_pairs_used = 0
    n_pairs_skipped = 0

    for batch_start in range(0, len(en_prompts), batch_size):
        en_batch = en_prompts[batch_start : batch_start + batch_size]
        langx_batch = langx_prompts[batch_start : batch_start + batch_size]

        en_inputs = tokenizer(
            en_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        langx_inputs = tokenizer(
            langx_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        en_ids = en_inputs["input_ids"].to(device)
        langx_ids = langx_inputs["input_ids"].to(device)

        pad_id = tokenizer.pad_token_id
        en_t_inst, en_last = _find_positions_in_batch(en_ids, pad_id, model_name, tokenizer)
        langx_t_inst, langx_last = _find_positions_in_batch(langx_ids, pad_id, model_name, tokenizer)

        # ── Clean (EN) forward pass: cache t_inst activations at every layer ──
        en_acts: Dict[Tuple[int, str], torch.Tensor] = {}  # key: (layer, comp) -> (batch, hidden)

        # Declare containers OUTSIDE the trace block — nnsight 0.6.x trace context
        # creates a new execution scope; variables assigned inside are not visible outside.
        saved_en: Dict = {}
        en_last_h_container: List = [None]

        with torch.no_grad():
            with lm.trace(en_ids):
                _layers = _get_layers(lm)
                for l in range(num_layers):
                    lo = _layers[l]
                    if "attn_out" in components:
                        try:
                            ao = lo.self_attn.o_proj.output
                            saved_en[(l, "attn_out")] = ao.save()
                        except Exception:
                            pass
                    if "mlp_out" in components:
                        try:
                            mo = lo.mlp.output
                            saved_en[(l, "mlp_out")] = mo.save()
                        except Exception:
                            pass
                    if "residual" in components:
                        try:
                            ro = lo.output
                            if isinstance(ro, tuple):
                                ro = ro[0]
                            saved_en[(l, "residual")] = ro.save()
                        except Exception:
                            pass
                # Also save last-token hidden state for clean metric
                last_layer_out = _layers[-1].output
                if isinstance(last_layer_out, tuple):
                    last_layer_out = last_layer_out[0]
                en_last_h_container[0] = last_layer_out.save()

        # Extract t_inst slice (batch, hidden) per layer/component
        for key, saved in saved_en.items():
            t = _unwrap_saved(saved)
            if t is None or t.ndim < 3:
                continue
            en_acts[key] = _gather_position(t.float(), en_t_inst).to(device)

        metric_clean = _compute_refusal_metric(
            _gather_position(_unwrap_saved(en_last_h_container[0]).float(), en_last), refusal_direction
        )

        # ── Corrupted (LangX) baseline pass ──
        langx_last_h_container: List = [None]
        with torch.no_grad():
            with lm.trace(langx_ids):
                _layers = _get_layers(lm)
                last_out_corrupted = _layers[-1].output
                if isinstance(last_out_corrupted, tuple):
                    last_out_corrupted = last_out_corrupted[0]
                langx_last_h_container[0] = last_out_corrupted.save()

        metric_corrupted = _compute_refusal_metric(
            _gather_position(_unwrap_saved(langx_last_h_container[0]).float(), langx_last), refusal_direction
        )
        baseline_diff = metric_clean - metric_corrupted

        # Analysis B: skip pairs without behavioral contrast
        if normalize and abs(baseline_diff) < min_baseline_diff:
            n_pairs_skipped += len(en_batch)
            logger.debug(
                f"Batch {batch_start // batch_size}: skipped "
                f"(baseline_diff={baseline_diff:.4f} < {min_baseline_diff})"
            )
            continue
        n_pairs_used += len(en_batch)

        # ── Patched passes: one per (layer, component) ──
        for patch_layer in range(num_layers):
            for comp in components:
                key = (patch_layer, comp)
                if key not in en_acts:
                    continue

                en_t_inst_act = en_acts[key]   # (batch, hidden)

                # Determine which submodule to patch (avoid 'continue' inside trace)
                if comp == "attn_out":
                    def _get_act_proxy(lo):
                        return lo.self_attn.o_proj.output
                elif comp == "mlp_out":
                    def _get_act_proxy(lo):
                        return lo.mlp.output
                else:  # residual
                    def _get_act_proxy(lo):
                        out = lo.output
                        return out[0] if isinstance(out, tuple) else out

                patched_last_h_container: List = [None]
                try:
                    with torch.no_grad():
                        with lm.trace(langx_ids):
                            _layers = _get_layers(lm)
                            lo = _layers[patch_layer]
                            act_proxy = _get_act_proxy(lo)

                            # Position-specific patch at each sample's t_inst
                            for k in range(len(langx_batch)):
                                act_proxy[k, langx_t_inst[k], :] = en_t_inst_act[k].to(act_proxy.dtype)

                            final_out = _layers[-1].output
                            if isinstance(final_out, tuple):
                                final_out = final_out[0]
                            patched_last_h_container[0] = final_out.save()
                except Exception as e:
                    logger.debug(f"Patch trace failed for layer={patch_layer} comp={comp}: {e}")
                    continue

                metric_patched = _compute_refusal_metric(
                    _gather_position(_unwrap_saved(patched_last_h_container[0]).float(), langx_last),
                    refusal_direction,
                )

                raw_delta = metric_patched - metric_corrupted

                if normalize:
                    score = raw_delta / baseline_diff
                else:
                    score = raw_delta

                scores[key].append(float(score))

        logger.info(
            f"Batch {batch_start // batch_size + 1}/"
            f"{(len(en_prompts) + batch_size - 1) // batch_size} done "
            f"(baseline_diff={baseline_diff:.4f})."
        )

    del lm
    clear_gpu_memory()

    score_col = "mean_restoration" if normalize else "mean_delta"
    std_col = "std_restoration" if normalize else "std_delta"

    rows = []
    for (layer, comp), vals in scores.items():
        if not vals:
            continue
        rows.append(
            {
                "layer": layer,
                "component": comp,
                "language": language,
                "tier": tier,
                "perturbation": perturbation,
                score_col: round(float(np.mean(vals)), 4),
                std_col: round(float(np.std(vals)), 4),
                "n_pairs_used": n_pairs_used,
                "n_pairs_skipped": n_pairs_skipped,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_tier(results: pd.DataFrame) -> pd.DataFrame:
    """Average scores within each (tier, layer, component) group."""
    score_col = "mean_restoration" if "mean_restoration" in results.columns else "mean_delta"
    std_col = "std_restoration" if "std_restoration" in results.columns else "std_delta"

    group_cols = [c for c in ["tier", "layer", "component", "perturbation"] if c in results.columns]
    if not group_cols:
        return results

    return (
        results.groupby(group_cols)
        .agg(**{score_col: (score_col, "mean"), std_col: (std_col, "mean")})
        .reset_index()
    )


def identify_critical_layers(
    results: pd.DataFrame,
    top_k: int = 5,
    component: str = "residual",
    score_col: Optional[str] = None,
) -> List[int]:
    """
    Return the top-k layers sorted by mean score (Analysis A: delta, B: restoration).

    Args:
        results: DataFrame from run_attribution_patching().
        top_k: Number of critical layers to return.
        component: Which component to rank by.
        score_col: Column name to rank by. Auto-detected if None.
    """
    if score_col is None:
        score_col = "mean_restoration" if "mean_restoration" in results.columns else "mean_delta"

    sub = results[results["component"] == component].copy()
    if sub.empty:
        return []

    by_layer = sub.groupby("layer")[score_col].mean().sort_values(ascending=False)
    return by_layer.head(top_k).index.tolist()
