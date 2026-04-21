"""
Multilingual MMLU Evaluation

Measure general capability preservation across languages.
"""

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MMLU_DATASET_ID = "openai/MMMLU"
CHOICES = ["A", "B", "C", "D"]


def load_multilingual_mmlu(
    languages: Optional[List[str]] = None,
    n_samples: int = 200,
    seed: int = 42,
) -> List[dict]:
    """
    Load multilingual MMLU samples.

    Args:
        languages: Language codes to include (MMMLU uses ISO codes).
                   If None, defaults to English.
        n_samples: Number of samples per language.
        seed: Random seed for sampling.

    Returns:
        List of dicts with 'prompt', 'answer', 'language' keys.
    """
    if languages is None:
        languages = ["EN-US"]

    try:
        from datasets import load_dataset as hf_load

        records = []
        for lang in languages:
            try:
                # MMMLU uses language names as config names
                ds = hf_load(MMLU_DATASET_ID, lang, split="test", trust_remote_code=True)
            except Exception:
                # Try the default config
                try:
                    ds = hf_load(MMLU_DATASET_ID, split="test", trust_remote_code=True)
                except Exception as e2:
                    logger.warning(f"Could not load MMLU for language '{lang}': {e2}")
                    continue

            # Sample
            import random
            rng = random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            sample_indices = indices[:n_samples]

            for idx in sample_indices:
                item = ds[idx]
                question = item.get("question", "")
                choices = [item.get(c, "") for c in ["A", "B", "C", "D"]]
                answer = item.get("answer", "A")

                # Format as multiple-choice prompt
                choices_str = "\n".join(
                    f"{letter}. {text}"
                    for letter, text in zip(CHOICES, choices)
                    if text
                )
                prompt = f"{question}\n{choices_str}\nAnswer:"

                records.append({
                    "prompt": prompt,
                    "answer": answer,
                    "language": lang,
                    "question": question,
                    "choices": choices,
                    "source": "MMMLU",
                })

        logger.info(f"Loaded {len(records)} MMLU examples across {len(languages)} languages.")
        return records

    except Exception as e:
        logger.warning(f"Could not load MMMLU: {e}. Returning empty list.")
        return []


def evaluate_mmlu(
    model_name: str,
    mmlu_data: List[dict],
    intervention_fn: Optional[Callable] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Evaluate model accuracy on MMLU data, optionally with an intervention.

    Args:
        model_name: HuggingFace model identifier.
        mmlu_data: List of dicts from load_multilingual_mmlu().
        intervention_fn: Optional callable that wraps model generation
                         (e.g., applies CAA or subspace projection).
        batch_size: Batch size for generation.

    Returns:
        Dict mapping language -> accuracy.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.gpu import clear_gpu_memory

    logger.info(f"Loading model for MMLU eval: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Group by language
    from collections import defaultdict
    lang_groups: Dict[str, List[dict]] = defaultdict(list)
    for item in mmlu_data:
        lang_groups[item.get("language", "unknown")].append(item)

    results: Dict[str, float] = {}

    for lang, items in lang_groups.items():
        correct = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            prompts = [item["prompt"] for item in batch]
            answers = [item["answer"] for item in batch]

            if intervention_fn is not None:
                responses = intervention_fn(prompts)
            else:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                responses = [
                    tokenizer.decode(o[input_len:], skip_special_tokens=True).strip()
                    for o in out
                ]

            for resp, gold in zip(responses, answers):
                # Check if the first character of response matches the answer letter
                pred = resp.strip()[:1].upper() if resp.strip() else ""
                if pred == gold.upper():
                    correct += 1

        acc = correct / len(items) if items else 0.0
        results[lang] = round(acc, 4)
        logger.info(f"MMLU [{lang}]: accuracy = {acc:.3f} ({correct}/{len(items)})")

    del model
    clear_gpu_memory()
    return results
