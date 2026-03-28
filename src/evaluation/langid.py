"""
Output Language Consistency

Check if interventions cause the model to switch to English responses
even when prompted in another language.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _sanitize_text_for_langid(text: str) -> str:
    """
    ftlangdetect expects one line at a time, so remove newlines and collapse
    repeated whitespace.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of a text string.

    Args:
        text: Input text.

    Returns:
        Tuple of (language_code, confidence_score).
    """
    cleaned = _sanitize_text_for_langid(text)
    if not cleaned:
        return "unknown", 0.0

    try:
        from ftlangdetect import detect
        result = detect(cleaned, low_memory=False)
        return result["lang"], result["score"]
    except ImportError:
        logger.warning("ftlangdetect not available. Returning 'unknown'.")
        return "unknown", 0.0
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "unknown", 0.0


def compute_langid_consistency(
    prompts: List[dict],
    responses: List[str],
) -> Dict[str, float]:
    """
    For each target language, compute the fraction of responses in that language.

    Args:
        prompts: List of dicts with a 'language' field (target language code).
        responses: List of model response strings corresponding to prompts.

    Returns:
        Dict mapping language_code -> consistency_rate (fraction in target lang).
    """
    if len(prompts) != len(responses):
        raise ValueError(
            f"Mismatch: {len(prompts)} prompts vs {len(responses)} responses."
        )

    from collections import defaultdict
    total: Dict[str, int] = defaultdict(int)
    consistent: Dict[str, int] = defaultdict(int)

    for prompt_dict, response in zip(prompts, responses):
        target_lang = prompt_dict.get("language", "unknown")
        total[target_lang] += 1

        if not response or not str(response).strip():
            continue

        detected_lang, score = detect_language(response)

        detected_norm = _normalize_lang_code(detected_lang)
        target_norm = _normalize_lang_code(target_lang)

        if detected_norm == target_norm:
            consistent[target_lang] += 1

    result = {}
    for lang, count in total.items():
        result[lang] = round(consistent[lang] / count, 4) if count > 0 else 0.0

    logger.info(f"LangID consistency: {result}")
    return result


def _normalize_lang_code(code: str) -> str:
    """Normalize language codes (e.g., 'zh-cn' -> 'zh')."""
    code = str(code).lower().strip()
    if "-" in code:
        code = code.split("-")[0]
    if "_" in code:
        code = code.split("_")[0]
    return code


def compute_english_switch_rate(
    prompts: List[dict],
    responses: List[str],
) -> float:
    """
    Compute the fraction of non-English prompts that received English responses.

    Args:
        prompts: List of prompt dicts with 'language' field.
        responses: Corresponding model responses.

    Returns:
        English switch rate in [0, 1].
    """
    non_en = [
        (p, r)
        for p, r in zip(prompts, responses)
        if _normalize_lang_code(p.get("language", "en")) != "en"
    ]
    if not non_en:
        return 0.0

    switched = 0
    valid = 0

    for _, response in non_en:
        if not response or not str(response).strip():
            continue

        valid += 1
        detected_lang, _ = detect_language(response)
        if _normalize_lang_code(detected_lang) == "en":
            switched += 1

    if valid == 0:
        return 0.0

    return switched / valid