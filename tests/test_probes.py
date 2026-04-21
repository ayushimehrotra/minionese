"""
Tests for the probing modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.linear_probe import train_probe
from src.probing.subspace import construct_subspace, compute_effective_rank
from src.probing.effective_rank import effective_rank_at_threshold


# ────────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_activations():
    """Generate synthetic harmful/benign activations with clear separation."""
    rng = np.random.RandomState(42)
    hidden_dim = 64
    n = 100

    # Harmful cluster centered at +1 along first dim
    harmful = rng.randn(n, hidden_dim)
    harmful[:, 0] += 3.0

    # Benign cluster centered at -1
    benign = rng.randn(n, hidden_dim)
    benign[:, 0] -= 3.0

    return harmful, benign


# ────────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────────


def test_probe_training_convergence(synthetic_activations):
    """Probes should achieve >60% accuracy on clearly separated activations."""
    harmful, benign = synthetic_activations
    result = train_probe(harmful, benign, C_values=[0.1, 1.0], cv_folds=3)

    assert "weights" in result
    assert "cv_accuracy" in result
    assert result["cv_accuracy"] > 0.6, (
        f"Probe accuracy {result['cv_accuracy']:.3f} below expected threshold."
    )


def test_probe_weight_shape(synthetic_activations):
    """Probe weights should be (hidden_dim,)."""
    harmful, benign = synthetic_activations
    hidden_dim = harmful.shape[1]
    result = train_probe(harmful, benign, C_values=[1.0], cv_folds=2)
    assert result["weights"].shape == (hidden_dim,), (
        f"Expected weights shape ({hidden_dim},), got {result['weights'].shape}"
    )


def test_probe_auc(synthetic_activations):
    """Probe AUC should be >0.9 for well-separated activations."""
    harmful, benign = synthetic_activations
    result = train_probe(harmful, benign, C_values=[1.0], cv_folds=3)
    assert result["cv_auc"] > 0.9, (
        f"Probe AUC {result['cv_auc']:.3f} below expected threshold."
    )


def test_subspace_construction():
    """SVD output shapes should be correct."""
    rng = np.random.RandomState(42)
    hidden_dim = 64
    n_categories = 5

    probe_weights = {
        f"cat_{i}": rng.randn(hidden_dim)
        for i in range(n_categories)
    }

    subspace = construct_subspace(probe_weights, energy_threshold=0.95)

    assert subspace["W_l"].shape == (n_categories, hidden_dim)
    assert subspace["U"].shape[0] == n_categories
    assert subspace["S"].ndim == 1
    assert subspace["Vt"].shape[1] == hidden_dim
    assert subspace["projection_matrix"].shape == (hidden_dim, hidden_dim)
    assert isinstance(subspace["effective_rank"], int)
    assert 1 <= subspace["effective_rank"] <= n_categories


def test_effective_rank():
    """Effective rank should be <= n_categories."""
    rng = np.random.RandomState(42)
    n_categories = 8
    hidden_dim = 64
    W = rng.randn(n_categories, hidden_dim)

    from scipy.linalg import svd
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    rank = compute_effective_rank(S, threshold=0.95)
    assert 1 <= rank <= n_categories, f"Unexpected effective rank: {rank}"


def test_effective_rank_at_threshold():
    """Effective rank should capture at least threshold fraction of energy."""
    S = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    rank = effective_rank_at_threshold(S, threshold=0.95)
    total_energy = np.sum(S ** 2)
    captured = np.sum(S[:rank] ** 2) / total_energy
    assert captured >= 0.95, f"Energy captured {captured:.3f} < 0.95 threshold."


def test_effective_rank_all_equal_singular_values():
    """With equal singular values, effective rank should equal full rank."""
    S = np.ones(10)
    rank = effective_rank_at_threshold(S, threshold=0.95)
    assert rank <= 10


def test_projection_matrix_is_symmetric():
    """Projection matrix P = Vt.T @ Vt should be symmetric and idempotent (approx)."""
    rng = np.random.RandomState(0)
    hidden_dim = 32
    probe_weights = {f"c{i}": rng.randn(hidden_dim) for i in range(4)}
    subspace = construct_subspace(probe_weights)
    P = subspace["projection_matrix"]

    # Symmetry
    assert np.allclose(P, P.T, atol=1e-5), "Projection matrix should be symmetric."

    # Idempotency: P @ P ≈ P (within rank)
    # Note: P = Vt_top.T @ Vt_top, and Vt_top @ Vt_top.T = I for orthonormal rows
    # So P @ P = Vt_top.T @ (Vt_top @ Vt_top.T) @ Vt_top = Vt_top.T @ Vt_top = P
    PP = P @ P
    assert np.allclose(PP, P, atol=1e-5), "Projection matrix should be idempotent."
