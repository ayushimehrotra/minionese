"""
Cross-Lingual SAE Feature Scoring

Identify cross-lingual failure features using mean-difference delta scores.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def compute_delta_scores(
    en_features: torch.Tensor,
    langx_features: torch.Tensor,
) -> np.ndarray:
    """
    Compute mean-difference delta scores per SAE feature.

    delta_i = mean(z_i | EN_harmful) - mean(z_i | LangX_harmful)

    Features with large |delta_i| are "cross-lingual failure features":
    they activate strongly for English harmful prompts but not for LangX.

    Args:
        en_features: (n_en_harmful, sae_width) tensor.
        langx_features: (n_langx_harmful, sae_width) tensor.

    Returns:
        (sae_width,) numpy array of delta scores.
    """
    en_mean = en_features.float().mean(dim=0).cpu().numpy()
    langx_mean = langx_features.float().mean(dim=0).cpu().numpy()
    delta = en_mean - langx_mean
    return delta


def rank_features(delta_scores: np.ndarray, top_k: int = 50) -> List[int]:
    """
    Return indices of top-k features by |delta|.

    Args:
        delta_scores: (sae_width,) array of delta scores.
        top_k: Number of top features to return.

    Returns:
        List of feature indices sorted by |delta| descending.
    """
    abs_delta = np.abs(delta_scores)
    top_k = min(top_k, len(abs_delta))
    return list(np.argsort(abs_delta)[::-1][:top_k])


def feature_analysis_table(
    delta_scores: np.ndarray,
    top_k: int = 50,
    feature_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Create a ranked feature analysis table.

    Args:
        delta_scores: (sae_width,) array of delta scores.
        top_k: Number of top features to include.
        feature_labels: Optional dict mapping feature_idx -> human-readable label.

    Returns:
        DataFrame with columns: feature_idx, delta_score, abs_delta, rank, label.
    """
    ranked_indices = rank_features(delta_scores, top_k)
    rows = []
    for rank, idx in enumerate(ranked_indices):
        label = ""
        if feature_labels:
            label = feature_labels.get(idx, "")
        rows.append({
            "rank": rank + 1,
            "feature_idx": int(idx),
            "delta_score": round(float(delta_scores[idx]), 4),
            "abs_delta": round(float(abs(delta_scores[idx])), 4),
            "label": label,
        })
    return pd.DataFrame(rows)


def compute_multi_language_delta_scores(
    en_features: torch.Tensor,
    language_features: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    Compute delta scores for multiple languages vs. English.

    Args:
        en_features: (n_en, sae_width) English harmful feature activations.
        language_features: Dict mapping lang_code -> (n_lang, sae_width) tensor.

    Returns:
        Dict mapping lang_code -> delta_scores array.
    """
    results = {}
    for lang, langx_features in language_features.items():
        delta = compute_delta_scores(en_features, langx_features)
        results[lang] = delta
    return results
