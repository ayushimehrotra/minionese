"""
Coherence-based language filtering.

Used ONLY for intervention evaluation (script 07). All other scripts
(activation extraction, probing, attribution patching, SAE analysis)
run on ALL languages including incoherent ones, because we need those
representations to explain WHY collapse happens.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_COHERENCE_PATH = "results/evaluation/coherence_table.csv"


def get_coherent_languages(
    model_key: str,
    coherence_path: str = DEFAULT_COHERENCE_PATH,
    min_coherence: float = 0.5,
) -> Optional[list]:
    """
    Return languages where model produces coherent output.
    Returns None if coherence table not found (caller should proceed unfiltered).
    """
    p = Path(coherence_path)
    if not p.exists():
        logger.warning(f"Coherence table not found: {p}. No filtering applied.")
        return None

    df = pd.read_csv(str(p))

    model_col = "model_key" if "model_key" in df.columns else "model"
    if model_col not in df.columns:
        logger.warning("Coherence table has no model column. No filtering applied.")
        return None

    # Match exact key or as a substring of the model name (e.g. "aya" matches "CohereForAI/aya-expanse-8b")
    exact = df[model_col] == model_key
    partial = df[model_col].str.contains(model_key, case=False, na=False)
    mask = (exact | partial) & (df["coherence_rate"] >= min_coherence)
    langs = sorted(df[mask]["language"].unique().tolist())
    logger.info(f"Coherent languages for {model_key} (n={len(langs)}): {langs}")
    return langs if langs else None


def filter_languages(languages: list, model_key: str, **kwargs) -> list:
    """
    Intersect a language list with coherent languages.
    If coherence table is missing, returns the original list unchanged.
    """
    coherent = get_coherent_languages(model_key, **kwargs)
    if coherent is None:
        return languages
    filtered = [lang for lang in languages if lang in coherent]
    n_dropped = len(languages) - len(filtered)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} incoherent languages for {model_key} interventions.")
    return filtered
