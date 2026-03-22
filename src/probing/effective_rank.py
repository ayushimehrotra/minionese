"""
Effective Rank Utilities

Compute and analyze the effective rank of representation matrices
at each layer across languages.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def effective_rank_at_threshold(S: np.ndarray, threshold: float = 0.95) -> int:
    """
    Compute the minimum number of singular values to capture `threshold`
    fraction of total energy (sum of squares).

    Args:
        S: 1-D array of singular values (in descending order).
        threshold: Energy fraction in (0, 1].

    Returns:
        Integer effective rank.
    """
    if S.size == 0:
        return 0
    cumulative = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    idx = np.searchsorted(cumulative, threshold)
    return int(idx) + 1


def compute_rank_profile(
    activations: np.ndarray,
    threshold: float = 0.95,
) -> dict:
    """
    Compute the rank profile of an activation matrix.

    Args:
        activations: (n_samples, hidden_dim) array.
        threshold: Energy threshold.

    Returns:
        Dict with keys: singular_values, effective_rank, energy_spectrum.
    """
    # Center the activations
    centered = activations - activations.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    rank = effective_rank_at_threshold(S, threshold)
    energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)

    return {
        "singular_values": S,
        "effective_rank": rank,
        "energy_spectrum": energy,
    }


def compute_effective_rank_table(
    activations_by_layer: Dict[int, np.ndarray],
    language: str,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Compute effective rank at each layer for a given language.

    Args:
        activations_by_layer: Dict mapping layer_idx -> (n_prompts, hidden) array.
        language: Language code (for labeling).
        threshold: Energy threshold.

    Returns:
        DataFrame with columns: layer, language, effective_rank, top_sv.
    """
    rows = []
    for layer_idx in sorted(activations_by_layer.keys()):
        acts = activations_by_layer[layer_idx]
        profile = compute_rank_profile(acts, threshold)
        rows.append({
            "layer": layer_idx,
            "language": language,
            "effective_rank": profile["effective_rank"],
            "top_sv": float(profile["singular_values"][0]) if len(profile["singular_values"]) > 0 else 0.0,
        })
    return pd.DataFrame(rows)
