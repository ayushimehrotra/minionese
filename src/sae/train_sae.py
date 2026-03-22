"""
SAE Availability Registry

All three target models have pre-trained SAE suites available on HuggingFace.
No custom SAE training is required.

  - Llama-3.1-8B-Instruct: LlamaScope (fnlp/Llama-Scope-8B-Base-LXR-32x-TopK)
                             Fallback: EleutherAI/sae-llama-3.1-8b-32x
  - Gemma-2-9B-IT:          GemmaScope IT (google/gemma-scope-9b-it-res)
                             Fallback: google/gemma-scope-9b-pt-res
  - Qwen2.5-7B-Instruct:    andyrdt/saes-qwen2.5-7b-instruct
"""

import logging

logger = logging.getLogger(__name__)

SAE_AVAILABILITY = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "available": True,
        "source": "saelens",
        "hf_repo": "fnlp/Llama-Scope-8B-Base-LXR-32x-TopK",
        "fallback": "EleutherAI/sae-llama-3.1-8b-32x",
    },
    "google/gemma-2-9b-it": {
        "available": True,
        "source": "saelens",
        "hf_repo": "google/gemma-scope-9b-it-res",
        "fallback": "google/gemma-scope-9b-pt-res",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "available": True,
        "source": "saelens",
        "hf_repo": "andyrdt/saes-qwen2.5-7b-instruct",
        "fallback": None,
    },
}


def check_sae_availability(model_name: str) -> dict:
    """
    Return SAE availability info for the given model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Dict with keys: available (bool), source (str), hf_repo (str or None).
    """
    info = SAE_AVAILABILITY.get(model_name)
    if info is None:
        for key, val in SAE_AVAILABILITY.items():
            if any(part in model_name.lower() for part in key.lower().split("/")):
                return val
        logger.warning(f"SAE availability unknown for {model_name}.")
        return {"available": False, "source": "unknown", "hf_repo": None}
    return info


def _model_short(model_name: str) -> str:
    """Extract short model identifier."""
    name = model_name.split("/")[-1].lower()
    if "llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    return name[:20]
