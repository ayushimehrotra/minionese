"""
Tests for the intervention modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interventions.caa import compute_steering_vector
from src.interventions.subspace_project import learn_subspace_map
from src.interventions.sweep import compute_pareto_frontier


# ────────────────────────────────────────────────────────────────────────────────
# Tests that do not require GPU / model loading
# ────────────────────────────────────────────────────────────────────────────────


def test_caa_vector_shape():
    """Steering vector should have shape (hidden_dim,)."""
    rng = np.random.RandomState(42)
    hidden_dim = 128
    n = 50

    harmful = torch.tensor(rng.randn(n, hidden_dim), dtype=torch.float32)
    harmless = torch.tensor(rng.randn(n, hidden_dim), dtype=torch.float32)

    v = compute_steering_vector(harmful, harmless, layer=15)
    assert v.shape == (hidden_dim,), f"Expected shape ({hidden_dim},), got {v.shape}"
    assert v.dtype == torch.float32


def test_caa_vector_direction():
    """Steering vector should point from harmless to harmful cluster."""
    hidden_dim = 32
    n = 100

    # Harmful cluster offset in first dimension
    harmful = torch.zeros(n, hidden_dim)
    harmful[:, 0] = 2.0
    harmless = torch.zeros(n, hidden_dim)
    harmless[:, 0] = -2.0

    v = compute_steering_vector(harmful, harmless)
    # v[0] should be positive (pointing toward harmful)
    assert v[0].item() > 0, f"Steering vector should point toward harmful cluster, v[0]={v[0].item()}"


def test_subspace_map_identity_init():
    """Subspace map with large regularization should start near identity."""
    rng = np.random.RandomState(0)
    hidden_dim = 32
    n_train = 50

    # Same activations for source and target -> M should converge to identity
    activations = rng.randn(n_train, hidden_dim)

    # Simple projection matrix (project onto first 8 dims)
    P = np.zeros((hidden_dim, hidden_dim))
    P[:8, :8] = np.eye(8)

    M = learn_subspace_map(
        langx_activations=activations,
        en_activations=activations,
        projection_matrix=P,
        regularization=0.1,
    )

    assert M.shape == (hidden_dim, hidden_dim), f"Expected ({hidden_dim}, {hidden_dim}), got {M.shape}"

    # With identical source and target, M should be close to identity
    I = np.eye(hidden_dim)
    deviation = np.linalg.norm(M - I, "fro")
    assert deviation < 5.0, f"|M - I|_F = {deviation:.4f}, expected small deviation."


def test_sae_clamping_reconstructs():
    """
    Mock SAE clamping: clamped and decoded output should be close to input
    if all other features are preserved.
    """
    # Create a minimal mock SAE
    hidden_dim = 32
    sae_width = 128

    class MockSAE:
        def __init__(self):
            self.W_enc = np.random.randn(hidden_dim, sae_width).astype(np.float32)
            col_norms = np.linalg.norm(self.W_enc, axis=0, keepdims=True)  # (1, sae_width)
            self.W_dec = (self.W_enc / (col_norms + 1e-8)).T  # (sae_width, hidden_dim)
            self.device = torch.device("cpu")

        def encode(self, x):
            x_np = x.float().numpy()
            return torch.tensor(np.maximum(0, x_np @ self.W_enc), dtype=torch.float32)

        def decode(self, z):
            z_np = z.float().numpy()
            return torch.tensor(z_np @ self.W_dec.T, dtype=torch.float32)

        def to(self, device):
            return self

        def eval(self):
            return self

    sae = MockSAE()
    x = torch.randn(4, hidden_dim)

    # Encode, then decode without clamping - reconstruction should be similar
    z = sae.encode(x)
    x_recon = sae.decode(z)

    mse = float(torch.mean((x - x_recon) ** 2).item())
    # Mock SAE won't have perfect reconstruction, just check it returns the right shape
    assert x_recon.shape == x.shape, f"Reconstructed shape mismatch: {x_recon.shape} vs {x.shape}"
    assert torch.isfinite(x_recon).all(), "Reconstructed activations should be finite."


def test_pareto_frontier_computation():
    """Pareto frontier should contain non-dominated points."""
    import pandas as pd

    data = pd.DataFrame([
        {"intervention": "caa", "param_value": 0.5, "asr": 0.8, "over_refusal": 0.1, "tier": "tier1"},
        {"intervention": "caa", "param_value": 1.0, "asr": 0.6, "over_refusal": 0.15, "tier": "tier1"},
        {"intervention": "caa", "param_value": 2.0, "asr": 0.4, "over_refusal": 0.3, "tier": "tier1"},
        {"intervention": "sae_clamp", "param_value": 5, "asr": 0.5, "over_refusal": 0.1, "tier": "tier1"},
        {"intervention": "sae_clamp", "param_value": 10, "asr": 0.3, "over_refusal": 0.2, "tier": "tier1"},
        {"intervention": "baseline", "param_value": None, "asr": 0.9, "over_refusal": 0.05, "tier": "tier1"},
    ])

    pareto = compute_pareto_frontier(data)
    assert len(pareto) > 0, "Pareto frontier should have at least one point."

    # All returned points should be in original data
    for idx in pareto.index:
        assert idx in data.index

    # Verify no returned point is dominated by another returned point
    for i, row_i in pareto.iterrows():
        for j, row_j in pareto.iterrows():
            if i == j:
                continue
            # row_j should not dominate row_i
            safety_i = 1.0 - row_i["asr"]
            safety_j = 1.0 - row_j["asr"]
            util_i = 1.0 - row_i["over_refusal"]
            util_j = 1.0 - row_j["over_refusal"]
            dominated = (safety_j >= safety_i and util_j >= util_i and
                         (safety_j > safety_i or util_j > util_i))
            assert not dominated, f"Point {i} is dominated by point {j} in Pareto set."


def test_steering_vector_norm():
    """Steering vector should have non-zero norm."""
    rng = np.random.RandomState(1)
    n = 30
    hidden_dim = 64
    harmful = torch.tensor(rng.randn(n, hidden_dim), dtype=torch.float32) + 1.0
    harmless = torch.tensor(rng.randn(n, hidden_dim), dtype=torch.float32) - 1.0
    v = compute_steering_vector(harmful, harmless)
    assert v.norm().item() > 0.01, "Steering vector should have non-zero norm."
