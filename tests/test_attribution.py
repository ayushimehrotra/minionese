"""
Tests for the attribution patching module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.attribution_patch import aggregate_by_tier
from src.probing.disentangle import (
    extract_refusal_direction,
    project_orthogonal_to_refusal,
)


# ────────────────────────────────────────────────────────────────────────────────
# Tests that do not require GPU / model loading
# ────────────────────────────────────────────────────────────────────────────────


def test_refusal_direction_extraction():
    """Refusal direction should be normalized."""
    rng = np.random.RandomState(42)
    hidden_dim = 64
    refused = rng.randn(20, hidden_dim) + 1.0  # offset for separation
    complied = rng.randn(20, hidden_dim) - 1.0

    r_hat = extract_refusal_direction("test_model", refused, complied)
    assert r_hat.shape == (hidden_dim,)
    norm = np.linalg.norm(r_hat)
    assert abs(norm - 1.0) < 1e-5, f"Refusal direction should be unit norm, got {norm:.4f}"


def test_project_orthogonal_single_vector():
    """Orthogonal projection should be perpendicular to refusal direction."""
    rng = np.random.RandomState(0)
    hidden_dim = 32
    r_hat = rng.randn(hidden_dim)
    r_hat /= np.linalg.norm(r_hat)

    harm_dir = rng.randn(hidden_dim)
    projected = project_orthogonal_to_refusal(harm_dir, r_hat)

    # Should be orthogonal to r_hat
    dot = np.dot(projected, r_hat)
    assert abs(dot) < 1e-5, f"Projection should be orthogonal to r_hat, dot={dot:.6f}"


def test_project_orthogonal_matrix():
    """Projection of a matrix of directions should all be orthogonal to refusal."""
    rng = np.random.RandomState(1)
    hidden_dim = 32
    n_cats = 6
    r_hat = rng.randn(hidden_dim)
    r_hat /= np.linalg.norm(r_hat)

    harm_dirs = rng.randn(n_cats, hidden_dim)
    projected = project_orthogonal_to_refusal(harm_dirs, r_hat)

    assert projected.shape == (n_cats, hidden_dim)
    dots = projected @ r_hat
    assert np.allclose(dots, 0.0, atol=1e-5), f"Not all projected directions are orthogonal: {dots}"


def test_restoration_metric_range():
    """Restoration scores in aggregate_by_tier should be reasonable floats."""
    import pandas as pd

    # Synthetic restoration data
    data = pd.DataFrame([
        {"layer": i, "component": "residual", "language": "ar", "tier": "tier2",
         "mean_restoration": float(np.clip(np.random.randn(), -0.5, 1.5)),
         "std_restoration": abs(float(np.random.randn() * 0.1))}
        for i in range(10)
    ] + [
        {"layer": i, "component": "residual", "language": "ko", "tier": "tier2",
         "mean_restoration": float(np.clip(np.random.randn(), -0.5, 1.5)),
         "std_restoration": abs(float(np.random.randn() * 0.1))}
        for i in range(10)
    ])

    agg = aggregate_by_tier(data)
    assert "tier" in agg.columns
    assert "mean_restoration" in agg.columns
    # Restoration scores can be outside [0,1] due to normalization, but should be finite
    assert agg["mean_restoration"].notna().all()


def test_aggregate_by_tier_groups():
    """aggregate_by_tier should produce one row per (tier, layer, component)."""
    import pandas as pd

    data = pd.DataFrame([
        {"layer": l, "component": "residual", "tier": "tier1",
         "mean_restoration": 0.5, "std_restoration": 0.1}
        for l in range(5)
    ] + [
        {"layer": l, "component": "residual", "tier": "tier2",
         "mean_restoration": 0.3, "std_restoration": 0.1}
        for l in range(5)
    ])

    agg = aggregate_by_tier(data)
    assert len(agg) == 10  # 5 layers x 2 tiers
    assert set(agg["tier"].unique()) == {"tier1", "tier2"}


def test_refusal_direction_nonzero():
    """Refusal direction should not be all zeros unless activations are identical."""
    hidden_dim = 64
    refused = np.ones((5, hidden_dim))
    complied = -np.ones((5, hidden_dim))

    r_hat = extract_refusal_direction("test", refused, complied)
    assert np.linalg.norm(r_hat) > 0.5, "Refusal direction should be non-zero."
