"""
Back-Translation + WildGuard Re-scoring

Translates non-English model responses to English using NLLB-200,
then re-scores with WildGuard to get accurate ASR across all language tiers.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# NLLB-200 language code mapping from ISO 639-1 to NLLB flores codes
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "tr": "tur_Latn",
    "id": "ind_Latn",
    "hi": "hin_Deva",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "zu": "zul_Latn",
    "gd": "gla_Latn",
    "gn": "grn_Latn",
    "jw": "jav_Latn",
}


def _load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info(f"Loading NLLB-200 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Main branch only has pytorch_model.bin; PR #45 adds safetensors weights
    # which avoid the torch.load CVE-2025-32434 restriction in transformers.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        revision="refs/pr/45",
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    logger.info("NLLB-200 loaded.")
    return tokenizer, model


def translate_to_english(
    texts: List[str],
    source_lang: str,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[str]:
    """
    Translate a list of texts from source_lang to English using NLLB-200.
    Returns the translated strings in the same order.
    If source_lang is already English, returns texts unchanged.
    """
    import torch

    nllb_src = NLLB_LANG_MAP.get(source_lang)
    if nllb_src is None or source_lang == "en":
        return texts

    tgt_lang = "eng_Latn"
    translations = []

    tokenizer.src_lang = nllb_src
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_length,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def backtranslate_and_rescore(
    scored_responses: List[dict],
    cache_path: str = "results/evaluation/backtranslated_cache.jsonl",
    batch_size: int = 16,
    nllb_model: str = "facebook/nllb-200-distilled-600M",
    wildguard_batch_size: int = 8,
    hf_token: Optional[str] = None,
) -> List[dict]:
    """
    For each response that is coherent and in a non-English language,
    translate to English and re-score with WildGuard.

    Skips English responses (already scored correctly) and incoherent responses
    (they don't count as attack successes regardless).

    Returns a copy of scored_responses with an added 'wildguard_label_en' field.
    """
    from src.evaluation.safety_judge import score_wildguard

    cache_path = Path(cache_path)

    # Load existing cache
    cache: dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["prompt_id"]] = rec
        logger.info(f"Loaded {len(cache)} cached back-translations.")

    # Collect records that need translation (non-English, coherent, harmful)
    to_translate: dict = {}  # lang -> list of (idx, record)
    for idx, rec in enumerate(scored_responses):
        lang = rec.get("language", "en")
        is_harmful = rec.get("is_harmful", False)
        is_coherent = rec.get("is_coherent", True)
        pid = rec.get("prompt_id", "")

        if lang == "en" or not is_harmful or not is_coherent:
            continue
        if pid in cache:
            continue
        to_translate.setdefault(lang, []).append((idx, rec))

    # Translate language by language to keep source language consistent
    if to_translate:
        tokenizer, nllb_model_obj = _load_nllb(nllb_model)

        for lang, items in to_translate.items():
            logger.info(f"Translating {len(items)} {lang} responses to English...")
            texts = [r["response"] for _, r in items]
            translated = translate_to_english(
                texts, lang, tokenizer, nllb_model_obj, batch_size=batch_size
            )
            for (idx, rec), en_text in zip(items, translated):
                entry = {**rec, "response_en": en_text}
                cache[rec["prompt_id"]] = entry
                with open(cache_path, "a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        del tokenizer, nllb_model_obj
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Translation complete.")

    # Build list of records to re-score with WildGuard
    to_score = []
    for idx, rec in enumerate(scored_responses):
        lang = rec.get("language", "en")
        pid = rec.get("prompt_id", "")
        if lang == "en":
            continue
        cached = cache.get(pid)
        if cached and "response_en" in cached:
            to_score.append({**rec, "response": cached["response_en"], "_orig_idx": idx})

    # Re-score translated responses with WildGuard
    wg_cache_path = str(cache_path.parent / "wildguard_cache_en.jsonl")
    if to_score:
        logger.info(f"Re-scoring {len(to_score)} translated responses with WildGuard...")
        rescored = score_wildguard(
            to_score,
            batch_size=wildguard_batch_size,
            cache_path=wg_cache_path,
            hf_token=hf_token,
            use_vllm=True,
        )
        label_by_pid = {r["prompt_id"]: r.get("wildguard_label") for r in rescored}
    else:
        label_by_pid = {}

    # Merge back into original records
    result = []
    for rec in scored_responses:
        rec = dict(rec)
        lang = rec.get("language", "en")
        pid = rec.get("prompt_id", "")
        if lang == "en":
            rec["wildguard_label_en"] = rec.get("wildguard_label")
        elif pid in label_by_pid:
            rec["wildguard_label_en"] = label_by_pid[pid]
        else:
            rec["wildguard_label_en"] = rec.get("wildguard_label")
        result.append(rec)

    return result
