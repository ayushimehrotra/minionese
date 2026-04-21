"""
Feature Interpretation via Neuronpedia / Auto-Interp

Look up human-interpretable labels for top SAE features.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api/feature"
REQUEST_DELAY = 1.0  # seconds between API calls


def lookup_neuronpedia(
    model_id: str,
    layer: int,
    feature_indices: List[int],
    cache_dir: Optional[str] = None,
) -> Dict[int, str]:
    """
    Look up feature labels from the Neuronpedia API.

    Args:
        model_id: Neuronpedia model identifier (e.g. "gemma-2-9b").
        layer: Layer index.
        feature_indices: List of feature indices to look up.
        cache_dir: Optional directory to cache API responses.

    Returns:
        Dict mapping feature_idx -> label_string.
    """
    import json
    import os

    labels = {}

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"neuronpedia_{model_id}_layer{layer}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cached = json.load(f)
            labels.update({int(k): v for k, v in cached.items()})
            feature_indices = [i for i in feature_indices if i not in labels]

    if not feature_indices:
        return labels

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available. Cannot query Neuronpedia.")
        return labels

    for idx in feature_indices:
        url = f"{NEURONPEDIA_BASE_URL}/{model_id}/{layer}/{idx}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Extract label from response
                label = data.get("label", "") or data.get("description", "")
                if not label and "explanations" in data:
                    explanations = data["explanations"]
                    if explanations:
                        label = explanations[0].get("description", "")
                labels[idx] = label
            else:
                logger.debug(f"Neuronpedia returned {resp.status_code} for feature {idx}.")
                labels[idx] = ""
        except Exception as e:
            logger.warning(f"Neuronpedia lookup failed for feature {idx}: {e}")
            labels[idx] = ""

        time.sleep(REQUEST_DELAY)  # Rate limiting

    # Update cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        all_labels = {}
        if cache_path.exists():
            with open(cache_path, "r") as f:
                all_labels = json.load(f)
        all_labels.update({str(k): v for k, v in labels.items()})
        with open(cache_path, "w") as f:
            json.dump(all_labels, f)

    return labels


def auto_interpret_features(
    sae,
    feature_indices: List[int],
    dataset: List[str],
    top_k_examples: int = 20,
    interp_model: Optional[str] = None,
) -> Dict[int, str]:
    """
    Auto-interpret features by finding top-activating examples.

    Args:
        sae: SAELens SAE object.
        feature_indices: Features to interpret.
        dataset: List of text examples to find top activations on.
        top_k_examples: Number of top-activating examples per feature.
        interp_model: Optional LLM to generate descriptions (e.g., "gpt-4").

    Returns:
        Dict mapping feature_idx -> description string.
    """
    import torch

    logger.info(f"Auto-interpreting {len(feature_indices)} features...")

    # Encode all dataset examples
    sae.eval()
    all_feature_acts = []
    batch_size = 32

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        # Tokenize and encode -- requires model activations
        # This is a simplified placeholder; in practice, you'd run the
        # base model to get residual stream activations first
        logger.debug(f"Processing dataset batch {i // batch_size + 1}")

    descriptions = {}
    for idx in feature_indices:
        # Placeholder: return empty description
        # In a full implementation, this would:
        # 1. Sort examples by feature activation value
        # 2. Send top examples to an LLM
        # 3. Ask the LLM to describe what the feature detects
        descriptions[idx] = f"Feature {idx} (auto-interp pending)"

    return descriptions


def load_feature_labels(
    model_name: str,
    layer: int,
    feature_indices: List[int],
    cache_dir: str = "results/sae_features/",
    use_neuronpedia: bool = True,
) -> Dict[int, str]:
    """
    Load feature labels using Neuronpedia (if available) or auto-interp fallback.

    Args:
        model_name: HuggingFace model name.
        layer: Layer index.
        feature_indices: Feature indices to look up.
        cache_dir: Cache directory for API results.
        use_neuronpedia: Whether to query Neuronpedia API.

    Returns:
        Dict mapping feature_idx -> label string.
    """
    # Map model name to Neuronpedia model ID
    neuronpedia_model_map = {
        "google/gemma-2-9b-it": "gemma-2-9b-it",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    }

    model_id = neuronpedia_model_map.get(model_name)
    if model_id and use_neuronpedia:
        try:
            labels = lookup_neuronpedia(model_id, layer, feature_indices, cache_dir)
            return labels
        except Exception as e:
            logger.warning(f"Neuronpedia lookup failed: {e}")

    # Return empty labels if no source available
    return {idx: "" for idx in feature_indices}
