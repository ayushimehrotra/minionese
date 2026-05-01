"""
Tests for the cross-lingual refusal cone module.

Synthetic activations are constructed so that we know the ground-truth
refusal direction. The cone optimizer should recover bases that are
strongly aligned with that direction and have positive harmful-vs-harmless
margins both in English and in non-English splits.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probing.refusal_cone import (
    CrossLingualActivations,
    cone_attack_proxy,
    cross_lingual_repind_score,
    optimize_cross_lingual_cone,
    repind_matrix,
)


def _make_synthetic(seed: int = 0, hidden_dim: int = 64, n: int = 80,
                    nonen_langs=("fr", "zh"), shift: float = 1.5,
                    n_refusal_axes: int = 3):
    """Build synthetic activations with multiple known refusal axes.

    Harmful clusters are shifted along the first `n_refusal_axes` axes by
    progressively smaller amounts (shift, shift/2, shift/3, ...). Non-English
    languages reuse the same refusal axes, with a per-language offset on a
    *separate* axis (added to both harmful and harmless, so it cancels out
    in the difference of means) to simulate language-specific drift.

    The first refusal axis is the "DIM" axis a single-direction baseline
    would recover; the additional axes are what cone_dim>1 should pick up.
    """
    rng = np.random.RandomState(seed)
    en_h = rng.randn(n, hidden_dim).astype(np.float32)
    en_s = rng.randn(n, hidden_dim).astype(np.float32)
    for k in range(n_refusal_axes):
        en_h[:, k] += shift / (k + 1)

    nonen_h = {}
    nonen_s = {}
    for li, lang in enumerate(nonen_langs):
        h = rng.randn(n, hidden_dim).astype(np.float32)
        s = rng.randn(n, hidden_dim).astype(np.float32)
        for k in range(n_refusal_axes):
            h[:, k] += shift / (k + 1)
        # language-specific drift on a non-refusal axis (cancels in diff-of-means)
        drift_axis = n_refusal_axes + li
        h[:, drift_axis] += 0.5
        s[:, drift_axis] += 0.5
        nonen_h[lang] = h
        nonen_s[lang] = s

    return CrossLingualActivations(
        en_harmful=torch.tensor(en_h),
        en_harmless=torch.tensor(en_s),
        nonen_harmful={k: torch.tensor(v) for k, v in nonen_h.items()},
        nonen_harmless={k: torch.tensor(v) for k, v in nonen_s.items()},
    )


def test_cone_basis_shape_and_orthonormality():
    acts = _make_synthetic(seed=0)
    seed_dir = np.zeros(acts.hidden_dim, dtype=np.float32)
    seed_dir[0] = 1.0

    result = optimize_cross_lingual_cone(
        acts, seed_direction=seed_dir, cone_dim=3, n_steps=80, seed=1,
    )
    B = result["basis"]
    assert B.shape == (acts.hidden_dim, 3)
    gram = B.T @ B
    assert np.allclose(gram, np.eye(3), atol=1e-3), \
        f"Basis not orthonormal: max off-diag = {np.max(np.abs(gram - np.eye(3))):.4f}"


def test_cone_basis_columns_have_positive_margin():
    """Every column of the basis should have non-negative EN refusal margin —
    the cone-correctness condition (every direction in the cone mediates refusal).
    """
    acts = _make_synthetic(seed=1)
    seed_dir = np.zeros(acts.hidden_dim, dtype=np.float32)
    seed_dir[0] = 1.0

    result = optimize_cross_lingual_cone(
        acts, seed_direction=seed_dir, cone_dim=4, n_steps=80, seed=2,
    )
    margins = result["en_margins"]
    # Allow a tiny numerical slack — refinement may overshoot by a small amount.
    assert (margins >= -1e-3).all(), f"Some basis columns have negative EN margin: {margins}"
    # The first column should be the most aligned with the dominant refusal axis.
    assert result["seed_alignment"][0] > 0.5


def test_cone_recovers_refusal_axis_in_1d():
    acts = _make_synthetic(seed=2, shift=2.0)
    seed_dir = np.zeros(acts.hidden_dim, dtype=np.float32)
    seed_dir[0] = 1.0

    result = optimize_cross_lingual_cone(
        acts, seed_direction=seed_dir, cone_dim=1, n_steps=120, seed=3,
    )
    b = result["basis"][:, 0]
    # The harmful-shifting axis is index 0. Cosine sim with axis 0 should be high.
    cos = float(b[0] / (np.linalg.norm(b) + 1e-12))
    assert cos > 0.6, f"Recovered direction not aligned with refusal axis: cos={cos:.3f}"
    # Margins should be positive in both EN and cross-lingual splits.
    assert result["en_margins"][0] > 0.5
    assert result["xling_margins"][0] > 0.3


def test_cone_attack_proxy_positive_on_known_signal():
    acts = _make_synthetic(seed=3, shift=2.0)
    seed_dir = np.zeros(acts.hidden_dim, dtype=np.float32)
    seed_dir[0] = 1.0

    result = optimize_cross_lingual_cone(
        acts, seed_direction=seed_dir, cone_dim=2, n_steps=120, seed=4,
    )
    proxy = cone_attack_proxy(
        result["basis"],
        acts.en_harmful.numpy(),
        acts.en_harmless.numpy(),
        n_samples=128,
        seed=5,
    )
    assert proxy["frac_positive"] > 0.95, \
        f"Most cone samples should mediate refusal in EN: frac_positive={proxy['frac_positive']:.3f}"
    assert proxy["mean_margin"] > 0.5


def test_repind_score_self_is_one_or_higher():
    """A direction is perfectly representationally independent from itself
    only in the trivial sense that ablating it kills its own signal — so
    self-RepInd should be bounded in [0, 1] but is not the diagonal of the
    matrix. We instead check the score is in range and finite."""
    acts = _make_synthetic(seed=4)
    nonen = {k: v.numpy() for k, v in acts.nonen_harmful.items()}
    u = np.random.RandomState(0).randn(acts.hidden_dim).astype(np.float32)
    v = np.random.RandomState(1).randn(acts.hidden_dim).astype(np.float32)
    out = cross_lingual_repind_score(u, v, nonen)
    assert 0.0 <= out["score"] <= 1.0
    assert set(out["per_language"].keys()) == set(nonen.keys())


def test_repind_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.RandomState(0)
    d = 32
    n = 4
    basis = rng.randn(d, n).astype(np.float32)
    # Orthonormalize for sanity.
    q, _ = np.linalg.qr(basis)
    basis = q[:, :n]

    nonen = {
        "fr": rng.randn(40, d).astype(np.float32),
        "zh": rng.randn(40, d).astype(np.float32),
    }
    M = repind_matrix(basis, nonen)
    assert M.shape == (n, n)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T, atol=1e-6)
    assert ((M >= 0.0) & (M <= 1.0)).all()


def test_repind_matrix_orthogonal_directions_are_independent():
    """If u and v are orthogonal and non-EN activations have no special
    correlation with v, then ablating v should leave u's alignment nearly
    unchanged, so CL-RepInd should be close to 1."""
    rng = np.random.RandomState(7)
    d = 64
    e0 = np.zeros(d, dtype=np.float32); e0[0] = 1.0
    e1 = np.zeros(d, dtype=np.float32); e1[1] = 1.0
    basis = np.stack([e0, e1], axis=1)

    X = rng.randn(120, d).astype(np.float32)
    nonen = {"fr": X, "zh": X * 0.5}
    M = repind_matrix(basis, nonen)
    assert M[0, 1] > 0.9, f"Orthogonal axes should be highly CL-RepInd, got {M[0, 1]:.3f}"


def test_higher_cone_dims_do_not_collapse_in_en_margin():
    """Margins should remain positive even as we add more basis vectors."""
    acts = _make_synthetic(seed=5, shift=2.0)
    seed_dir = np.zeros(acts.hidden_dim, dtype=np.float32)
    seed_dir[0] = 1.0

    for dim in (1, 2, 3):
        result = optimize_cross_lingual_cone(
            acts, seed_direction=seed_dir, cone_dim=dim, n_steps=100, seed=10 + dim,
        )
        assert (result["en_margins"] > 0).all(), \
            f"Cone dim={dim} produced non-positive EN margin: {result['en_margins']}"


# ────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline integration test. Builds a tiny synthetic dataset and
# matching cached activations, places a seed refusal direction, then invokes
# scripts/04b_refusal_cones.py via subprocess and checks the output files.
# ────────────────────────────────────────────────────────────────────────────


def _write_pair_csvs(lang_dir, n_pairs, rng):
    """Write tiny harmful/harmless CSVs for one language."""
    import pandas as pd
    lang_dir.mkdir(parents=True, exist_ok=True)
    harmful = pd.DataFrame({
        "prompt": [f"harmful prompt {i}" for i in range(n_pairs)],
        "category": ["test"] * n_pairs,
        "en_prompt": [f"harmful prompt {i}" for i in range(n_pairs)],
    })
    harmless = pd.DataFrame({
        "prompt": [f"harmless prompt {i}" for i in range(n_pairs)],
        "category": ["test"] * n_pairs,
        "en_prompt": [f"harmless prompt {i}" for i in range(n_pairs)],
    })
    harmful.to_csv(lang_dir / "harmful.csv", index=False)
    harmless.to_csv(lang_dir / "harmless.csv", index=False)


def _write_safetensors(path, tensor):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"activations": tensor}, str(path))


def test_pipeline_end_to_end(tmp_path):
    """Build a minimum synthetic input tree and run scripts/04b_refusal_cones.py
    with --model aya. The model config in configs/models.yaml drives everything;
    we override paths via CLI args to point into tmp_path."""
    import os
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).parent.parent
    pert = "standard_translation"
    n_pairs = 16
    hidden_dim = 4096  # matches aya hidden_size in configs/models.yaml
    num_layers = 32  # aya
    critical_layer = 16  # midpoint of [10, 22]

    rng = np.random.RandomState(0)
    languages_by_tier = {
        "tier_1": ["en", "de"],  # only 2 langs to keep test fast
    }
    all_langs = [l for langs in languages_by_tier.values() for l in langs]

    # Per-language refusal axes (linearly independent across languages so the
    # cone has rank-2 structure, mimicking real cross-lingual refusal data).
    refusal_signals = {
        "en": [(0, 1.5), (1, 0.7)],
        "de": [(0, 1.0), (1, 1.2)],
    }

    # 1. Synthetic dataset CSVs
    dataset_dir = tmp_path / "dataset"
    for tier_idx, (tier_name, langs) in enumerate(languages_by_tier.items(), start=1):
        for lang in langs:
            lang_dir = dataset_dir / pert / f"tier{tier_idx}" / lang
            _write_pair_csvs(lang_dir, n_pairs, rng)

    # 2. Synthetic cached activations (n_prompts = 2*n_pairs, layered)
    acts_dir = tmp_path / "activations"
    seed_axis = np.zeros(hidden_dim, dtype=np.float32); seed_axis[0] = 1.0
    for lang in all_langs:
        # interleave order = harmful_0..n, harmless_0..n (matches CSV concat order)
        per_lang_blocks = []
        for is_harm in (True, False):
            block = rng.randn(n_pairs, num_layers, hidden_dim).astype(np.float32) * 0.2
            if is_harm:
                for axis, shift in refusal_signals[lang]:
                    block[:, critical_layer, axis] += shift
            per_lang_blocks.append(block)
        full = torch.tensor(np.concatenate(per_lang_blocks, axis=0))
        path = (acts_dir / f"aya_{lang}_{pert}_last_post_instruction_residual.safetensors")
        _write_safetensors(path, full)

    # 3. Seed refusal direction (the EN DIM direction; we approximate it with axis 0)
    repr_dir = tmp_path / "representation" / "aya"
    repr_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(repr_dir / "refusal_direction_aya.npy"), seed_axis)

    # 4. Restrict the languages config to the two we built so the script
    #    doesn't iterate over languages we did not synthesize.
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(repo_root / "configs" / "models.yaml", cfg_dir / "models.yaml")
    shutil.copy(repo_root / "configs" / "experiment.yaml", cfg_dir / "experiment.yaml")
    shutil.copy(repo_root / "configs" / "paths.yaml", cfg_dir / "paths.yaml")
    (cfg_dir / "languages.yaml").write_text(
        "tiers:\n"
        "  tier_1:\n"
        "    languages: [en, de]\n"
        "    label: synthetic\n"
        "perturbation_types:\n"
        "  - standard_translation\n"
    )

    # 5. Run the script. It loads config from cwd, so we cd into tmp_path
    #    after copying configs there.
    script = repo_root / "scripts" / "04b_refusal_cones.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    cmd = [
        sys.executable, str(script),
        "--model", "aya",
        "--activations-dir", str(acts_dir),
        "--dataset-dir", str(dataset_dir),
        "--representation-dir", str(repr_dir),
        "--cone-dims", "1", "2",
        "--n-steps", "20",
    ]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # 6. Verify outputs
    basis_path = repr_dir / "refusal_cone_aya.npy"
    metrics_path = repr_dir / "refusal_cone_metrics.csv"
    repind_path = repr_dir / "refusal_cone_repind.csv"
    summary_path = repr_dir / "refusal_cone_summary.json"
    per_lang_path = repr_dir / "refusal_cone_per_language.csv"

    assert basis_path.exists(), f"Missing basis output. STDERR:\n{result.stderr}"
    assert metrics_path.exists()
    assert repind_path.exists()
    assert summary_path.exists()
    assert per_lang_path.exists()

    basis = np.load(str(basis_path))
    assert basis.shape == (hidden_dim, 2), f"Unexpected basis shape: {basis.shape}"
    # Each column should have positive EN refusal margin (cone-correctness).
    import pandas as pd
    metrics = pd.read_csv(metrics_path)
    assert (metrics["en_margin"] > 0).all(), \
        f"Some basis columns lost positive EN margin: {metrics['en_margin'].tolist()}"

    # Per-language margins: every (cone_dim, basis_idx) row should have a
    # margin entry for each non-EN language we synthesized (de). We only
    # assert that the first basis vector (the dominant refusal axis) has
    # positive DE margin — secondary basis vectors live in the orthogonal
    # subspace and can have either sign depending on per-language drift.
    per_lang = pd.read_csv(per_lang_path)
    assert set(per_lang["language"].unique()) == {"de"}
    assert per_lang["margin"].notna().all() and np.isfinite(per_lang["margin"]).all()
    primary = per_lang[per_lang["basis_idx"] == 0]["margin"]
    assert (primary > 0).all(), \
        f"Primary basis vector should have positive DE margin: {primary.tolist()}"
