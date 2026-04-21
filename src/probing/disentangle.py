"""
Harmfulness vs. Refusal Disentanglement (Cross-Lingual Extension of Zhao et al. 2025)

Separates the harmfulness subspace from the refusal direction to determine
whether cross-lingual failure is upstream (harm detection) or downstream
(refusal gate).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_refusal_direction(
    model_name: str,
    refused_activations: np.ndarray,
    complied_activations: np.ndarray,
) -> np.ndarray:
    """
    Extract the refusal direction using the Arditi et al. mean-difference method.

    Args:
        model_name: Model name (for logging).
        refused_activations: (n_refused, hidden_dim) activations for refused prompts.
        complied_activations: (n_complied, hidden_dim) activations for complied prompts.

    Returns:
        Normalized refusal direction vector of shape (hidden_dim,).
    """
    r = refused_activations.mean(axis=0) - complied_activations.mean(axis=0)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        logger.warning(f"Refusal direction has near-zero norm for {model_name}.")
        return r
    r_hat = r / norm
    logger.info(f"Refusal direction extracted for {model_name}, norm={norm:.4f}.")
    return r_hat


def project_orthogonal_to_refusal(
    harm_directions: np.ndarray,
    refusal_direction: np.ndarray,
) -> np.ndarray:
    """
    Project harm direction(s) onto the complement of the refusal direction.

    For each direction w in harm_directions:
        w' = w - (w . r_hat) * r_hat

    Args:
        harm_directions: (n_categories, hidden_dim) or (hidden_dim,) array.
        refusal_direction: Normalized refusal direction (hidden_dim,).

    Returns:
        Projected directions of same shape as input.
    """
    r = refusal_direction / (np.linalg.norm(refusal_direction) + 1e-12)

    if harm_directions.ndim == 1:
        proj = np.dot(harm_directions, r) * r
        return harm_directions - proj
    else:
        # (n_categories, hidden)
        projs = (harm_directions @ r)[:, np.newaxis] * r[np.newaxis, :]
        return harm_directions - projs


def disentangle_analysis(
    harm_subspaces: Dict[Tuple, np.ndarray],
    refusal_direction: np.ndarray,
    activations_t_inst: Dict[Tuple, np.ndarray],
    activations_t_post_inst: Dict[Tuple, np.ndarray],
    layers: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full disentanglement analysis across languages and layers.

    Measures:
    - harm_component_norm: norm of W' (harm component orthogonal to refusal)
    - refusal_component_norm: norm of W projected onto refusal
    - t_inst_harm_signal: harmfulness signal at t_inst position
    - t_post_inst_refusal_signal: refusal signal at t_post_inst position
    - failure_type: "upstream", "downstream", or "mixed"

    Args:
        harm_subspaces: Dict mapping (lang, layer) -> W_l (n_cat, hidden).
        refusal_direction: Refusal direction vector (hidden_dim,).
        activations_t_inst: Dict mapping (lang, "harmful"/"benign") -> (n, hidden).
        activations_t_post_inst: Dict mapping (lang, "harmful"/"benign") -> (n, hidden).
        layers: Layer indices to analyze (optional filter).

    Returns:
        DataFrame with per-language-layer disentanglement metrics.
    """
    r_hat = refusal_direction / (np.linalg.norm(refusal_direction) + 1e-12)

    rows = []

    all_keys = list(harm_subspaces.keys())
    if layers is not None:
        all_keys = [(lang, layer) for lang, layer in all_keys if layer in layers]

    for lang, layer in sorted(all_keys):
        W_l = harm_subspaces[(lang, layer)]  # (n_cat, hidden)
        if W_l.ndim == 1:
            W_l = W_l.reshape(1, -1)

        # Refusal-aligned component of each probe direction
        refusal_projs = W_l @ r_hat  # (n_cat,)
        refusal_component_norm = float(np.linalg.norm(refusal_projs))

        # Orthogonal (pure harm) component
        W_prime = project_orthogonal_to_refusal(W_l, r_hat)
        harm_component_norm = float(np.linalg.norm(W_prime))

        # t_inst harm signal: mean cosine sim of harmful activations with W_prime direction
        t_inst_harm_signal = _compute_signal_strength(
            activations_t_inst, lang, W_prime
        )

        # t_post_inst refusal signal: mean dot product with refusal direction
        t_post_refusal_signal = _compute_refusal_signal(
            activations_t_post_inst, lang, r_hat
        )

        # Classify failure type
        # Compare to English baseline
        en_key = ("en", layer)
        if en_key in harm_subspaces:
            W_en = harm_subspaces[en_key]
            if W_en.ndim == 1:
                W_en = W_en.reshape(1, -1)
            W_prime_en = project_orthogonal_to_refusal(W_en, r_hat)
            harm_en = float(np.linalg.norm(W_prime_en))
        else:
            harm_en = harm_component_norm

        # Simple heuristic classification
        harm_ratio = harm_component_norm / (harm_en + 1e-9)
        if harm_ratio < 0.5:
            failure_type = "upstream"
        elif t_post_refusal_signal < 0.0:
            failure_type = "downstream"
        else:
            failure_type = "mixed"

        rows.append({
            "language": lang,
            "layer": layer,
            "harm_component_norm": round(harm_component_norm, 4),
            "refusal_component_norm": round(refusal_component_norm, 4),
            "t_inst_harm_signal": round(t_inst_harm_signal, 4),
            "t_post_inst_refusal_signal": round(t_post_refusal_signal, 4),
            "failure_type": failure_type,
        })

    return pd.DataFrame(rows)


def _compute_signal_strength(
    activations: Dict[Tuple, np.ndarray],
    lang: str,
    direction: np.ndarray,
) -> float:
    """Mean projection of harmful activations onto a direction."""
    harm_key = (lang, "harmful")
    harmless_key = (lang, "harmless")
    if harm_key not in activations:
        return 0.0

    harm_acts = activations[harm_key]  # (n, hidden)
    dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    dir_unit = direction / (dir_norm + 1e-12)

    if dir_unit.ndim == 2:
        # Use first direction if matrix
        dir_unit = dir_unit[0]

    harm_proj = harm_acts @ dir_unit  # (n,)

    if harmless_key in activations:
        harmless_acts = activations[harmless_key]
        harmless_proj = harmless_acts @ dir_unit
        return float(harm_proj.mean() - harmless_proj.mean())
    return float(harm_proj.mean())


def _compute_refusal_signal(
    activations: Dict[Tuple, np.ndarray],
    lang: str,
    refusal_direction: np.ndarray,
) -> float:
    """Mean projection of harmful activations onto refusal direction."""
    harm_key = (lang, "harmful")
    harmless_key = (lang, "harmless")
    if harm_key not in activations:
        return 0.0

    harm_acts = activations[harm_key]
    harm_proj = harm_acts @ refusal_direction

    if harmless_key in activations:
        harmless_acts = activations[harmless_key]
        harmless_proj = harmless_acts @ refusal_direction
        return float(harm_proj.mean() - harmless_proj.mean())
    return float(harm_proj.mean())
