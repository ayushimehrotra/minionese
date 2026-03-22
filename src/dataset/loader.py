"""
Dataset Loader

The dataset is PRE-BUILT and lives in the `dataset/` directory at the repo root.

Directory layout:
    dataset/
    ├── standard_translation/tier1/en/harmful.csv
    ├── standard_translation/tier1/en/harmless.csv
    ├── standard_translation/tier1/de/harmful.csv
    ├── ...                (same for all tiers and languages)
    ├── translationese/tier1/en/harmful.csv
    ├── ...                (same structure)
    ├── transliteration/...
    ├── code_switching/...
    └── minionese/harmful.csv          # Flat -- no tier/language subfolders
        minionese/harmless.csv

Each CSV has columns (verify and adapt on first run -- column names may vary):
    - prompt: the translated/perturbed prompt text
    - category: HarmBench harm subcategory (e.g. "cybercrime", "violence")
    - en_prompt: the original English prompt this was derived from
    (possibly additional metadata columns)

Harmful and harmless CSVs are contrastive pairs: row i of harmful.csv and
row i of harmless.csv should differ by roughly one token (e.g. "pipe bomb"
vs "pipe long"). THIS HAS NOT BEEN VERIFIED -- the loader must include a
validation step (see validate_contrastive_pairs below).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names after normalization
PROMPT_COL = "prompt"
CATEGORY_COL = "category"
EN_PROMPT_COL = "en_prompt"

# Alternative column names that map to canonical names
PROMPT_ALIASES = ["prompt", "translated_prompt", "text", "input", "sentence", "question"]
CATEGORY_ALIASES = ["category", "harm_category", "label", "type", "harm_type"]
EN_PROMPT_ALIASES = ["en_prompt", "english_prompt", "original_prompt", "source_prompt", "original"]

TIERED_PERTURBATIONS = [
    "standard_translation",
    "translationese",
    "code_switching",
    "transliteration",
]
FLAT_PERTURBATIONS = ["minionese"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical form."""
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for alias in PROMPT_ALIASES:
        if alias in df.columns and PROMPT_COL not in rename_map.values():
            rename_map[alias] = PROMPT_COL
            break
    for alias in CATEGORY_ALIASES:
        if alias in df.columns and CATEGORY_COL not in rename_map.values():
            rename_map[alias] = CATEGORY_COL
            break
    for alias in EN_PROMPT_ALIASES:
        if alias in df.columns and EN_PROMPT_COL not in rename_map.values():
            rename_map[alias] = EN_PROMPT_COL
            break

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    if PROMPT_COL not in df.columns:
        # Try using the first string column
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            logger.warning(
                f"No recognized prompt column found; using '{str_cols[0]}' as prompt."
            )
            df = df.rename(columns={str_cols[0]: PROMPT_COL})
        else:
            raise ValueError(f"Cannot find prompt column in CSV. Columns: {df.columns.tolist()}")

    if CATEGORY_COL not in df.columns:
        df[CATEGORY_COL] = "unknown"
    if EN_PROMPT_COL not in df.columns:
        df[EN_PROMPT_COL] = None

    return df


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV with UTF-8 (with BOM fallback)."""
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            return _normalize_columns(df)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not load CSV with any known encoding: {path}")


def discover_languages(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Auto-discover which languages exist under each perturbation/tier
    by walking the directory tree.

    Args:
        dataset_dir: Root dataset directory.

    Returns:
        Dict mapping perturbation -> list of language codes found.
    """
    dataset_dir = Path(dataset_dir)
    result: Dict[str, List[str]] = {}

    for pert in TIERED_PERTURBATIONS:
        pert_dir = dataset_dir / pert
        if not pert_dir.exists():
            continue
        langs = set()
        for tier_dir in sorted(pert_dir.iterdir()):
            if not tier_dir.is_dir():
                continue
            for lang_dir in sorted(tier_dir.iterdir()):
                if lang_dir.is_dir():
                    langs.add(lang_dir.name)
        result[pert] = sorted(langs)

    for pert in FLAT_PERTURBATIONS:
        pert_dir = dataset_dir / pert
        if pert_dir.exists():
            result[pert] = [pert]  # treat the perturbation itself as the language key

    return result


def load_dataset(
    dataset_dir: str = "dataset/",
    perturbations: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Walk the directory tree and load all matching CSVs into a single DataFrame.

    Adds columns: perturbation, tier, language, is_harmful.
    For minionese (no tier/lang subfolders), tier="none" and language="minionese".

    Args:
        dataset_dir: Root dataset directory.
        perturbations: List of perturbation types to include (None = all).
        languages: List of language codes to include (None = all).
        tiers: List of tier names to include (None = all).

    Returns:
        Concatenated DataFrame.
    """
    dataset_dir = Path(dataset_dir)
    all_dfs = []

    all_perturbations = TIERED_PERTURBATIONS + FLAT_PERTURBATIONS
    if perturbations is not None:
        all_perturbations = [p for p in all_perturbations if p in perturbations]

    for pert in all_perturbations:
        pert_dir = dataset_dir / pert
        if not pert_dir.exists():
            logger.warning(f"Perturbation directory not found: {pert_dir}")
            continue

        if pert in FLAT_PERTURBATIONS:
            # Minionese: flat structure
            if languages is not None and pert not in languages and "minionese" not in languages:
                continue
            if tiers is not None and "none" not in tiers:
                continue
            for label, is_harmful in [("harmful", True), ("harmless", False)]:
                csv_path = pert_dir / f"{label}.csv"
                if not csv_path.exists():
                    logger.warning(f"Missing: {csv_path}")
                    continue
                try:
                    df = _load_csv(str(csv_path))
                    df["perturbation"] = pert
                    df["tier"] = "none"
                    df["language"] = "minionese"
                    df["is_harmful"] = is_harmful
                    all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {csv_path}: {e}")
        else:
            # Tiered structure
            for tier_dir in sorted(pert_dir.iterdir()):
                if not tier_dir.is_dir():
                    continue
                tier = tier_dir.name
                if tiers is not None and tier not in tiers:
                    continue
                for lang_dir in sorted(tier_dir.iterdir()):
                    if not lang_dir.is_dir():
                        continue
                    lang = lang_dir.name
                    if languages is not None and lang not in languages:
                        continue
                    for label, is_harmful in [("harmful", True), ("harmless", False)]:
                        csv_path = lang_dir / f"{label}.csv"
                        if not csv_path.exists():
                            logger.warning(f"Missing: {csv_path}")
                            continue
                        try:
                            df = _load_csv(str(csv_path))
                            df["perturbation"] = pert
                            df["tier"] = tier
                            df["language"] = lang
                            df["is_harmful"] = is_harmful
                            all_dfs.append(df)
                        except Exception as e:
                            logger.error(f"Error loading {csv_path}: {e}")

    if not all_dfs:
        logger.warning("No CSV files were loaded.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Loaded {len(combined)} rows from {len(all_dfs)} CSV files "
        f"({combined['perturbation'].nunique()} perturbations, "
        f"{combined['language'].nunique()} languages)."
    )
    return combined


def validate_contrastive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that harmful/harmless pairs differ by approximately one token.

    For each (perturbation, tier, language) group:
        1. Check that harmful and harmless have the same number of rows.
        2. For each row pair, compute token-level edit distance.
        3. Flag pairs where edit distance > 3 tokens.

    Args:
        df: Dataset DataFrame with is_harmful column.

    Returns:
        Report DataFrame with flagged pairs.
    """
    from difflib import SequenceMatcher

    def token_edit_distance(s1: str, s2: str) -> int:
        tokens1 = str(s1).split()
        tokens2 = str(s2).split()
        sm = SequenceMatcher(None, tokens1, tokens2)
        # Edit distance = len(tokens1) + len(tokens2) - 2 * matching_blocks
        matching = sum(block.size for block in sm.get_matching_blocks())
        return len(tokens1) + len(tokens2) - 2 * matching

    report_rows = []
    groups = df.groupby(["perturbation", "tier", "language"])

    total_pairs = 0
    flagged_pairs = 0

    for (pert, tier, lang), group in groups:
        harmful = group[group["is_harmful"]].reset_index(drop=True)
        harmless = group[~group["is_harmful"]].reset_index(drop=True)

        if len(harmful) != len(harmless):
            logger.warning(
                f"Row count mismatch for ({pert}, {tier}, {lang}): "
                f"{len(harmful)} harmful vs {len(harmless)} harmless."
            )
            report_rows.append({
                "perturbation": pert,
                "tier": tier,
                "language": lang,
                "row_idx": -1,
                "issue": "row_count_mismatch",
                "edit_distance": None,
                "harmful_prompt": None,
                "harmless_prompt": None,
            })
            continue

        n = min(len(harmful), len(harmless))
        total_pairs += n

        for i in range(n):
            h_text = str(harmful.iloc[i][PROMPT_COL])
            hl_text = str(harmless.iloc[i][PROMPT_COL])
            dist = token_edit_distance(h_text, hl_text)
            if dist > 3:
                flagged_pairs += 1
                report_rows.append({
                    "perturbation": pert,
                    "tier": tier,
                    "language": lang,
                    "row_idx": i,
                    "issue": "high_edit_distance",
                    "edit_distance": dist,
                    "harmful_prompt": h_text[:200],
                    "harmless_prompt": hl_text[:200],
                })

    print(
        f"Contrastive pair validation: "
        f"{total_pairs - flagged_pairs} of {total_pairs} pairs validated "
        f"({flagged_pairs} flagged for review)."
    )
    return pd.DataFrame(report_rows)


def format_for_model(
    dataset: pd.DataFrame,
    model_name: str,
    tokenizer,
) -> pd.DataFrame:
    """
    Add a "formatted_prompt" column by applying the model's chat template.

    Args:
        dataset: DataFrame with a 'prompt' column.
        model_name: HuggingFace model name (used to select template style).
        tokenizer: HuggingFace tokenizer with apply_chat_template support.

    Returns:
        DataFrame with added 'formatted_prompt' column.
    """
    dataset = dataset.copy()

    def _format_single(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Chat template error for prompt '{prompt[:50]}...': {e}")
            return prompt

    dataset["formatted_prompt"] = dataset[PROMPT_COL].apply(_format_single)
    return dataset


def get_contrastive_pairs(
    df: pd.DataFrame,
    language: str,
    perturbation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (harmful_df, harmless_df) with aligned row indices.

    Args:
        df: Full dataset DataFrame.
        language: Language code.
        perturbation: Perturbation type.

    Returns:
        Tuple of (harmful_df, harmless_df) both reset to integer indices.
    """
    mask = (df["language"] == language) & (df["perturbation"] == perturbation)
    group = df[mask]
    harmful = group[group["is_harmful"]].reset_index(drop=True)
    harmless = group[~group["is_harmful"]].reset_index(drop=True)
    return harmful, harmless


def get_split(
    df: pd.DataFrame,
    split: str,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    Deterministic train/test split by English source prompt.

    Splits by unique en_prompt value so the same English source is never
    in both train and test.

    Args:
        df: Dataset DataFrame.
        split: "train" or "test".
        seed: Random seed for reproducibility.
        test_ratio: Fraction of unique source prompts to put in test.

    Returns:
        Subset DataFrame for the requested split.
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    # Use en_prompt if available, else fall back to the prompt itself
    if EN_PROMPT_COL in df.columns and df[EN_PROMPT_COL].notna().any():
        unique_sources = df[EN_PROMPT_COL].dropna().unique()
    else:
        unique_sources = df[PROMPT_COL].unique()

    n_test = max(1, int(len(unique_sources) * test_ratio))
    perm = rng.permutation(len(unique_sources))
    test_sources = set(unique_sources[perm[:n_test]])

    if EN_PROMPT_COL in df.columns and df[EN_PROMPT_COL].notna().any():
        is_test = df[EN_PROMPT_COL].isin(test_sources)
    else:
        is_test = df[PROMPT_COL].isin(test_sources)

    if split == "test":
        return df[is_test].reset_index(drop=True)
    elif split == "train":
        return df[~is_test].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split: '{split}'. Use 'train' or 'test'.")
