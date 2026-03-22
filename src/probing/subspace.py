"""
Harmfulness Subspace Construction

Construct the harmfulness subspace W_l per layer from probe weight vectors.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def compute_effective_rank(S: np.ndarray, threshold: float = 0.95) -> int:
    """
    Compute effective rank at a given energy threshold.

    Args:
        S: Singular values (1-D array).
        threshold: Cumulative energy fraction to retain.

    Returns:
        Minimum rank to capture `threshold` of total energy.
    """
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(cumulative_energy, threshold)) + 1


def construct_subspace(
    probe_weights: Dict[str, np.ndarray],
    energy_threshold: float = 0.95,
) -> dict:
    """
    Construct the harmfulness subspace from probe weight vectors.

    Args:
        probe_weights: Dict mapping category_name -> weight_vector (hidden_dim,).
        energy_threshold: Fraction of energy to retain for effective rank.

    Returns:
        Dict with keys:
            W_l: (n_categories, hidden_dim) matrix of probe weights
            U:   Left singular vectors (n_categories, k)
            S:   Singular values (k,)
            Vt:  Right singular vectors (k, hidden_dim)
            effective_rank: int
            energy_spectrum: np.ndarray (cumulative)
            projection_matrix: (hidden_dim, hidden_dim) = Vt.T @ Vt
    """
    categories = sorted(probe_weights.keys())
    if not categories:
        raise ValueError("probe_weights is empty.")

    W_l = np.stack([probe_weights[cat] for cat in categories], axis=0)  # (n_cat, hidden)

    # SVD
    U, S, Vt = np.linalg.svd(W_l, full_matrices=False)

    rank = compute_effective_rank(S, energy_threshold)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Projection matrix onto the top-rank right singular vectors
    # P = Vt[:rank].T @ Vt[:rank]  shape: (hidden, hidden)
    Vt_top = Vt[:rank]  # (rank, hidden)
    projection_matrix = Vt_top.T @ Vt_top  # (hidden, hidden)

    return {
        "categories": categories,
        "W_l": W_l,
        "U": U,
        "S": S,
        "Vt": Vt,
        "effective_rank": rank,
        "energy_spectrum": energy,
        "projection_matrix": projection_matrix,
    }


def build_subspace_from_probes(
    probes_dir: str,
    language: str,
    layer: int,
    categories: list,
    energy_threshold: float = 0.95,
) -> dict:
    """
    Load probe weights from disk and construct the subspace.

    Args:
        probes_dir: Directory containing .npz probe files.
        language: Language code.
        layer: Layer index.
        categories: List of harm categories (excluding "all").
        energy_threshold: Energy threshold for effective rank.

    Returns:
        Subspace dict from construct_subspace().
    """
    from pathlib import Path
    from src.probing.linear_probe import load_probe

    probe_weights = {}
    for cat in categories:
        probe_path = Path(probes_dir) / f"probe_{language}_layer{layer}_{cat}.npz"
        if probe_path.exists():
            probe = load_probe(str(probe_path))
            probe_weights[cat] = probe["weights"]
        else:
            logger.warning(f"Probe not found: {probe_path}. Skipping category '{cat}'.")

    if not probe_weights:
        raise FileNotFoundError(
            f"No probe weights found for ({language}, layer={layer})."
        )

    return construct_subspace(probe_weights, energy_threshold)
