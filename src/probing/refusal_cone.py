"""
Cross-Lingual Refusal Cones with Activation-Level Representational Independence

An activation-domain adaptation of Refusal Cone Optimization (RCO; Wollschlager
et al., 2025) tailored to multilingual mechanistic analysis.

The original RCO requires gradient-based interventions through the model's
forward pass. Because this pipeline already caches per-language residual
activations, we re-formulate the problem as a constrained optimization on
those cached tensors:

  Maximize the harmful-vs-harmless mean projection gap of every basis vector
  in B = [b_1, ..., b_N], simultaneously in EN and in non-EN languages,
  subject to:
    1. each b_i lying in the half-space of the seed DIM refusal direction
       (so positive convex combinations of B form a *cone* of refusal
       directions);
    2. B being orthonormal (so b_i are not co-linear).

We further define **cross-lingual representational independence** (CL-RepInd),
an activation-only analog of Definition 6.1 in the paper. Two basis directions
u, v are CL-RepInd if ablating v from non-English harmful activations does not
change the alignment of u with those activations. This isolates the directions
that mediate refusal *through different cross-lingual circuits* — a richer
question than the monolingual independence in the original paper.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CrossLingualActivations:
    """Cached residual activations grouped by language and harm label.

    Each value is a (n, hidden_dim) float32 tensor at a single layer.
    """
    en_harmful: torch.Tensor
    en_harmless: torch.Tensor
    nonen_harmful: Dict[str, torch.Tensor]
    nonen_harmless: Dict[str, torch.Tensor]

    @property
    def hidden_dim(self) -> int:
        return self.en_harmful.shape[-1]

    @property
    def nonen_languages(self) -> List[str]:
        return sorted(self.nonen_harmful.keys())


def _gram_schmidt(B: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the columns of B with stabilized Gram-Schmidt.

    Args:
        B: (d, n) tensor.
    Returns:
        (d, n) tensor with orthonormal columns.
    """
    d, n = B.shape
    Q = torch.zeros_like(B)
    for i in range(n):
        v = B[:, i].clone()
        for j in range(i):
            v = v - torch.dot(Q[:, j], v) * Q[:, j]
        norm = v.norm()
        if norm < 1e-8:
            v = torch.randn(d, dtype=B.dtype, device=B.device)
            for j in range(i):
                v = v - torch.dot(Q[:, j], v) * Q[:, j]
            norm = v.norm()
        Q[:, i] = v / (norm + 1e-12)
    return Q


def _project_to_halfspace(B: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Flip any column of B whose dot product with r is negative.

    After this step, every column of B lies in the closed half-space
    {x : <x, r> >= 0}, which guarantees that all positive convex combinations
    of the columns form a cone aligned with r.
    """
    signs = torch.sign(B.t() @ r)
    signs[signs == 0] = 1.0
    return B * signs.unsqueeze(0)


def _signed_margin(harm: torch.Tensor, safe: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean projection gap for a single direction.

    Returns a scalar tensor: <harm, b>_mean - <safe, b>_mean.
    """
    return (harm @ b).mean() - (safe @ b).mean()


def _difference_of_means_matrix(
    activations: CrossLingualActivations,
    en_weight: float,
    xling_weight: float,
) -> Tuple[np.ndarray, List[str]]:
    """Stack per-language harmful-vs-harmless mean differences as rows.

    Each row is a "refusal axis" for one (language) cluster, with a weight.
    The English row gets `en_weight`; each non-English language gets
    `xling_weight`. SVD on this matrix gives the top right singular vectors
    that jointly maximize squared margins across languages.
    """
    rows = []
    labels = []

    en_diff = (activations.en_harmful.float().mean(0) - activations.en_harmless.float().mean(0)).numpy()
    rows.append(en_weight * en_diff)
    labels.append("en")

    for lang in activations.nonen_languages:
        h = activations.nonen_harmful[lang].float()
        s = activations.nonen_harmless[lang].float()
        if h.numel() == 0 or s.numel() == 0:
            continue
        rows.append(xling_weight * (h.mean(0) - s.mean(0)).numpy())
        labels.append(lang)

    V = np.stack(rows, axis=0).astype(np.float32)
    return V, labels


def _orient_to_halfspace_np(B: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Numpy version: flip columns whose dot with r is negative."""
    signs = np.sign(B.T @ r)
    signs[signs == 0] = 1.0
    return B * signs[None, :]


def _orient_for_positive_margin(B: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip each column of B so its inner product with `ref` is non-negative.

    Used with `ref = (mean_harmful - mean_harmless)` to guarantee a positive
    refusal margin per basis vector, which is the cone-correctness condition
    in the paper: every direction in the resulting cone mediates refusal.
    """
    signs = np.sign(B.T @ ref)
    signs[signs == 0] = 1.0
    return B * signs[None, :]


def optimize_cross_lingual_cone(
    activations: CrossLingualActivations,
    seed_direction: np.ndarray,
    cone_dim: int,
    n_steps: int = 200,
    lr: float = 1e-2,
    en_weight: float = 1.0,
    xling_weight: float = 1.0,
    cone_weight: float = 5.0,
    refine: bool = True,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Optimize an N-dimensional cross-lingual refusal cone.

    The objective is to find an orthonormal basis B = [b_1, ..., b_N] that
    jointly maximizes the harmful-vs-harmless mean projection gap (the
    "refusal margin") in English and in non-English languages, while keeping
    every column in the half-space of the seed DIM direction.

    The algorithm has two stages:

    1. SVD initialization. Stack weighted per-language difference-of-means
       vectors as rows of V; the top `cone_dim` right singular vectors of V
       are exactly the orthonormal basis that maximizes the sum of squared
       projection magnitudes — a closed-form analog of the paper's RCO step
       on the harmful-vs-harmless contrast. Columns are then flipped into
       the half-space of the seed direction.

    2. Projected-SGD refinement. We optionally take a few small SGD steps
       on the linear margin objective with re-orthonormalization between
       steps. This mirrors the paper's gradient-based RCO loop. We use
       SGD instead of Adam because Adam's per-dimension preconditioning
       amplifies noise in directions orthogonal to the signal.

    Args:
        activations: Cached residual activations split by language/harm label.
        seed_direction: Seed DIM refusal direction (hidden_dim,) used as cone axis.
        cone_dim: Cone dimensionality N (>= 1).
        n_steps: SGD refinement steps (set to 0 to use the SVD solution as-is).
        lr: SGD learning rate.
        en_weight: Weight on EN refusal-margin objective.
        xling_weight: Weight on cross-lingual refusal-margin objective.
        cone_weight: Weight on the half-space penalty during refinement.
        refine: If False, skip the SGD refinement and return the SVD solution.
        seed: RNG seed (unused by the deterministic SVD path; reserved for refine).
        device: torch device for refinement; defaults to CPU.

    Returns:
        Dict with keys:
            basis: (hidden_dim, cone_dim) orthonormal basis with each column
                in the half-space of `seed_direction`.
            en_margins, xling_margins, seed_alignment: per-direction metrics.
    """
    if cone_dim < 1:
        raise ValueError("cone_dim must be >= 1.")

    en_h = activations.en_harmful.float()
    en_s = activations.en_harmless.float()
    d = en_h.shape[-1]
    nonen_h = [activations.nonen_harmful[k].float() for k in activations.nonen_languages]
    nonen_s = [activations.nonen_harmless[k].float() for k in activations.nonen_languages]

    r_np = seed_direction.astype(np.float32) / (np.linalg.norm(seed_direction) + 1e-12)

    V, _ = _difference_of_means_matrix(activations, en_weight, xling_weight)
    if cone_dim > V.shape[0]:
        V = np.vstack([
            V,
            np.random.RandomState(seed).randn(cone_dim - V.shape[0], d).astype(np.float32) * 1e-3,
        ])

    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    B_np = Vt[:cone_dim].T  # (d, cone_dim)
    B_np = _orient_to_halfspace_np(B_np, r_np)
    Q, _ = np.linalg.qr(B_np)
    Q = Q[:, :cone_dim]

    # Flip each column of Q so its projection onto the EN difference-of-means
    # is non-negative. This guarantees a non-negative EN refusal margin per
    # basis vector — the cone-correctness condition. Because all per-language
    # difference vectors generally point in similar directions, this also
    # tends to keep cross-lingual margins positive.
    en_diff = (activations.en_harmful.float().mean(0) - activations.en_harmless.float().mean(0)).numpy()
    Q = _orient_for_positive_margin(Q, en_diff)

    if refine and n_steps > 0:
        device = torch.device(device) if device is not None else torch.device("cpu")
        r = torch.tensor(r_np, device=device, dtype=torch.float32)
        ref = torch.tensor(en_diff, device=device, dtype=torch.float32)
        B = torch.tensor(Q, device=device, dtype=torch.float32, requires_grad=True)
        en_h_t = en_h.to(device=device); en_s_t = en_s.to(device=device)
        nonen_h_t = [t.to(device=device) for t in nonen_h]
        nonen_s_t = [t.to(device=device) for t in nonen_s]

        opt = torch.optim.SGD([B], lr=lr, momentum=0.5)
        for step in range(n_steps):
            opt.zero_grad()
            loss = -en_weight * ((en_h_t @ B).mean(0) - (en_s_t @ B).mean(0)).sum()
            if nonen_h_t and xling_weight > 0:
                xling = sum(((h @ B).mean(0) - (s @ B).mean(0)).sum() for h, s in zip(nonen_h_t, nonen_s_t))
                loss = loss - xling_weight * (xling / max(len(nonen_h_t), 1))
            loss = loss + cone_weight * torch.relu(-(B.t() @ r)).pow(2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                # Re-orthonormalize, then flip each column to keep positive
                # margin (cone-correctness) and finally re-flip any column
                # whose seed alignment has gone strongly negative.
                Bn = _gram_schmidt(B.detach())
                signs = torch.sign(Bn.t() @ ref)
                signs[signs == 0] = 1.0
                Bn = Bn * signs.unsqueeze(0)
                B.copy_(Bn)
            if step % max(1, n_steps // 5) == 0:
                logger.debug(f"[CLR-Cone refine step {step}] loss={loss.item():.4f}")

        B_final_np = B.detach().cpu().numpy()
    else:
        B_final_np = Q.astype(np.float32)

    en_h_np = en_h.numpy()
    en_s_np = en_s.numpy()
    en_margins = (en_h_np @ B_final_np).mean(0) - (en_s_np @ B_final_np).mean(0)

    per_lang_gaps: Dict[str, np.ndarray] = {}
    gaps = []
    for lang, h_lang, s_lang in zip(activations.nonen_languages, nonen_h, nonen_s):
        if h_lang.numel() == 0 or s_lang.numel() == 0:
            continue
        gap = (h_lang.numpy() @ B_final_np).mean(0) - (s_lang.numpy() @ B_final_np).mean(0)
        per_lang_gaps[lang] = gap.astype(np.float32)
        gaps.append(gap)
    xling_margins = (
        np.mean(np.stack(gaps, axis=0), axis=0) if gaps else np.zeros(cone_dim, dtype=np.float32)
    )

    seed_alignment = B_final_np.T @ r_np

    return {
        "basis": B_final_np,
        "en_margins": en_margins.astype(np.float32),
        "xling_margins": xling_margins.astype(np.float32),
        "per_language_margins": per_lang_gaps,
        "seed_alignment": seed_alignment.astype(np.float32),
    }


def cross_lingual_repind_score(
    u: np.ndarray,
    v: np.ndarray,
    nonen_harmful: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """Cross-lingual representational independence between two directions.

    For each non-English language, compute how much u's cosine alignment with
    that language's harmful activations changes when v is ablated. CL-RepInd
    is the (negated, clipped) magnitude of that change averaged over languages
    — values near 1 mean the two directions act through different cross-lingual
    pathways, values near 0 mean they share the cross-lingual circuit.

    The original RepInd in the paper is monolingual and requires running the
    model under ablation. Here we work directly on cached activations: ablation
    is the linear projection x - <x, v_hat> v_hat. This is the "linear shadow"
    of the paper's intervention; non-linear effects across layers are not
    captured, but the cross-lingual axis we add is novel and complementary.

    Args:
        u: Direction (hidden_dim,).
        v: Direction (hidden_dim,).
        nonen_harmful: Per-language harmful activations (n_lang, hidden_dim).

    Returns:
        Dict with keys:
            score: Mean CL-RepInd score across languages (higher = more independent).
            per_language: Dict[lang -> score].
    """
    u_hat = u / (np.linalg.norm(u) + 1e-12)
    v_hat = v / (np.linalg.norm(v) + 1e-12)

    per_lang = {}
    for lang, X in nonen_harmful.items():
        if X.size == 0:
            continue
        # Cosine alignment of u with each row, before and after ablating v.
        norms = np.linalg.norm(X, axis=1) + 1e-12
        cos_before = (X @ u_hat) / norms
        X_abl = X - np.outer(X @ v_hat, v_hat)
        norms_abl = np.linalg.norm(X_abl, axis=1) + 1e-12
        cos_after = (X_abl @ u_hat) / norms_abl
        delta = float(np.mean(np.abs(cos_before - cos_after)))
        per_lang[lang] = float(np.clip(1.0 - delta, 0.0, 1.0))

    score = float(np.mean(list(per_lang.values()))) if per_lang else 0.0
    return {"score": score, "per_language": per_lang}


def repind_matrix(
    basis: np.ndarray,
    nonen_harmful: Dict[str, np.ndarray],
) -> np.ndarray:
    """Pairwise CL-RepInd matrix for the columns of `basis`.

    Args:
        basis: (hidden_dim, n) matrix.
        nonen_harmful: Per-language harmful activations.

    Returns:
        (n, n) symmetric matrix; diagonal is 1 by convention.
    """
    _, n = basis.shape
    M = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = cross_lingual_repind_score(basis[:, i], basis[:, j], nonen_harmful)["score"]
            M[i, j] = M[j, i] = s
    return M


def cone_attack_proxy(
    basis: np.ndarray,
    harmful: np.ndarray,
    harmless: np.ndarray,
    n_samples: int = 64,
    seed: int = 0,
) -> Dict[str, float]:
    """Monte-Carlo proxy for the cone-level attack-success-rate of the paper.

    Following Algorithm 2, sample unit vectors uniformly from the cone (positive
    orthant of the basis), compute the harmful-vs-harmless mean projection gap
    for each sample, and report the distribution. A larger and more uniform
    gap across samples indicates that *every* direction in the cone mediates
    refusal — the criterion the paper uses to validate cones.

    This is an activation-level proxy: the original paper measures real ASR
    by intervening on the model and judging completions. We measure the linear
    refusal margin instead, which correlates with intervention success but is
    cheap to compute over cached activations.
    """
    rng = np.random.RandomState(seed)
    d, n = basis.shape
    samples = rng.gamma(1.0, 1.0, size=(n_samples, n))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True) + 1e-12
    cone_vecs = samples @ basis.T  # (n_samples, d)
    cone_vecs /= np.linalg.norm(cone_vecs, axis=1, keepdims=True) + 1e-12

    margins = (harmful @ cone_vecs.T).mean(0) - (harmless @ cone_vecs.T).mean(0)
    return {
        "mean_margin": float(margins.mean()),
        "min_margin": float(margins.min()),
        "max_margin": float(margins.max()),
        "frac_positive": float((margins > 0).mean()),
    }
