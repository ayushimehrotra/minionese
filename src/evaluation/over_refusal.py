"""
Over-Refusal Evaluation

Measure how often the model incorrectly refuses benign prompts.
Uses OR-Bench and XSTest datasets.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_or_bench(languages: Optional[List[str]] = None) -> List[dict]:
    """
    Load OR-Bench dataset (or subset for target languages).

    Args:
        languages: Language codes to include. If None, returns English only.

    Returns:
        List of dicts with 'prompt', 'language', 'source' keys.
    """
    try:
        from datasets import load_dataset as hf_load

        dataset = hf_load("bench-llm/or-bench", split="test", trust_remote_code=True)
        records = []
        for item in dataset:
            records.append({
                "prompt": item.get("prompt", item.get("text", "")),
                "language": "en",
                "source": "or_bench",
            })
        logger.info(f"Loaded {len(records)} OR-Bench examples.")
        return records
    except Exception as e:
        logger.warning(f"Could not load OR-Bench: {e}. Returning empty list.")
        return []


def load_xstest(languages: Optional[List[str]] = None) -> List[dict]:
    """
    Load XSTest dataset.

    Args:
        languages: Language codes to include. If None, returns English only.

    Returns:
        List of dicts with 'prompt', 'language', 'source' keys.
    """
    try:
        from datasets import load_dataset as hf_load

        dataset = hf_load("paul-rottger/xstest-v2-prompts", split="test", trust_remote_code=True)
        records = []
        for item in dataset:
            prompt = item.get("prompt", item.get("text", ""))
            if item.get("type", "") == "safe":
                records.append({
                    "prompt": prompt,
                    "language": "en",
                    "source": "xstest",
                    "xstest_type": item.get("type", ""),
                })
        logger.info(f"Loaded {len(records)} XSTest examples.")
        return records
    except Exception as e:
        logger.warning(f"Could not load XSTest: {e}. Returning empty list.")
        return []


def compute_over_refusal_rate(
    model_name: str,
    benign_prompts: List[str],
    responses: List[str],
    judge: str = "wildguard",
) -> float:
    """
    Compute over-refusal rate: fraction of benign prompts that were refused.

    Args:
        model_name: Model that generated the responses (for logging).
        benign_prompts: List of benign prompt strings.
        responses: List of model responses corresponding to benign_prompts.
        judge: Which judge to use ("wildguard" or "llamaguard").

    Returns:
        Over-refusal rate in [0, 1].
    """
    if len(benign_prompts) != len(responses):
        raise ValueError(
            f"Mismatch: {len(benign_prompts)} prompts vs {len(responses)} responses."
        )

    records = [
        {
            "prompt_id": f"benign_{i}",
            "prompt": p,
            "response": r,
            "is_harmful": False,
        }
        for i, (p, r) in enumerate(zip(benign_prompts, responses))
    ]

    if judge == "wildguard":
        from src.evaluation.safety_judge import score_wildguard
        scored = score_wildguard(records)
        refusals = sum(1 for s in scored if s.get("wildguard_label") == "refusal")
    elif judge == "llamaguard":
        from src.evaluation.safety_judge import score_llamaguard
        scored = score_llamaguard(records)
        # LlamaGuard: classify as refusal if labeled "safe" (model refused the harmful request)
        refusals = sum(1 for s in scored if s.get("llamaguard_label") == "safe")
    else:
        raise ValueError(f"Unknown judge: {judge}")

    rate = refusals / len(records) if records else 0.0
    logger.info(
        f"Over-refusal rate [{model_name}, judge={judge}]: "
        f"{refusals}/{len(records)} = {rate:.3f}"
    )
    return rate
