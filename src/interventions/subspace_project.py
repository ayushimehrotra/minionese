"""
Intervention 3: Subspace Projection Sharpening

Learn constrained linear maps to align non-English harmfulness subspaces
to English, while leaving the complement space unchanged.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def learn_subspace_map(
    langx_activations: np.ndarray,
    en_activations: np.ndarray,
    projection_matrix: np.ndarray,
    regularization: float = 0.01,
) -> np.ndarray:
    """
    Learn a constrained linear map M_tier that aligns LangX harmfulness
    representations to the English cluster within the harmfulness subspace.

    Optimization:
        min ||M @ P @ H_langx - P @ H_en||_F^2 + lambda * ||M - I||_F^2

    Args:
        langx_activations: (n_train, hidden_dim) LangX activations.
        en_activations: (n_train, hidden_dim) English activations.
        projection_matrix: (hidden_dim, hidden_dim) projection P onto subspace.
        regularization: Lambda for identity regularization.

    Returns:
        M of shape (hidden_dim, hidden_dim) or (subspace_dim, subspace_dim).
    """
    # Project activations into subspace
    H_langx_proj = langx_activations @ projection_matrix  # (n, hidden)
    H_en_proj = en_activations @ projection_matrix  # (n, hidden)

    # Reduce to subspace dimensions (columns with non-negligible variance)
    P = projection_matrix
    # Use only the non-zero columns of P (the subspace basis)
    col_norms = np.linalg.norm(P, axis=0)
    active_cols = col_norms > 1e-6
    P_reduced = P[:, active_cols]  # (hidden, k)

    X = H_langx_proj @ P_reduced  # (n, k) - LangX in subspace coords
    Y = H_en_proj @ P_reduced  # (n, k) - EN in subspace coords

    # Solve: M @ X.T = Y.T  (least squares with regularization)
    # Equivalent to: (X.T @ X + lambda * I) M.T = X.T @ Y
    k = X.shape[1]
    A = X.T @ X + regularization * np.eye(k)
    B = X.T @ Y

    try:
        M_sub = np.linalg.solve(A, B).T  # (k, k)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix; using least-squares fallback.")
        M_sub, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        M_sub = M_sub.T

    # Lift M_sub back to full hidden space
    # M_full: (hidden, hidden) = P_reduced @ M_sub @ P_reduced.T + (I - P)
    M_full = P_reduced @ M_sub @ P_reduced.T + (np.eye(projection_matrix.shape[0]) - projection_matrix)

    logger.info(
        f"Subspace map learned: shape={M_full.shape}, "
        f"|M - I|_F = {np.linalg.norm(M_full - np.eye(M_full.shape[0])):.4f}"
    )
    return M_full


def apply_subspace_projection(
    model_name: str,
    prompts: List[str],
    M_tier: np.ndarray,
    projection_matrix: np.ndarray,
    layer: int,
    max_new_tokens: int = 512,
    batch_size: int = 4,
) -> List[str]:
    """
    Apply subspace projection intervention at inference time.

    For each forward pass at `layer`:
        h' = (I - P) @ h + M_tier @ (P @ h)

    Args:
        model_name: HuggingFace model name.
        prompts: Formatted prompt strings.
        M_tier: (hidden_dim, hidden_dim) learned linear map.
        projection_matrix: (hidden_dim, hidden_dim) projection P.
        layer: Layer to intervene at.
        max_new_tokens: Max generation tokens.
        batch_size: Batch size.

    Returns:
        List of generated response strings.
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
        trust_remote_code=True,
    )
    model.eval()

    M_t = torch.tensor(M_tier, dtype=torch.bfloat16).to(model.device)
    P_t = torch.tensor(projection_matrix, dtype=torch.bfloat16).to(model.device)
    I_minus_P = torch.eye(P_t.shape[0], dtype=torch.bfloat16, device=model.device) - P_t

    def intervention_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
        else:
            h = output.clone()

        # Apply: h' = (I-P)h + M*(P*h) at last token position
        h_last = h[:, -1, :].float()
        h_proj = (h_last @ P_t.float())
        h_complement = (h_last @ I_minus_P.float())
        h_new = h_complement + (h_proj @ M_t.T.float())
        h[:, -1, :] = h_new.to(h.dtype)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    try:
        hook_layer = model.model.layers[layer]
    except AttributeError:
        hook_layer = model.model.model.layers[layer]

    handle = hook_layer.register_forward_hook(intervention_hook)

    all_responses = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for j in range(len(batch)):
                resp = tokenizer.decode(out[j][input_len:], skip_special_tokens=True)
                all_responses.append(resp)

    finally:
        handle.remove()
        del model
        clear_gpu_memory()

    return all_responses
