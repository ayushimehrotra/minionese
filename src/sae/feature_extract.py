"""
SAE Feature Extraction via SAELens

Decompose activations into SAE features at critical layers.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# SAE suite IDs per model (SAELens conventions)
SAE_SUITE_IDS = {
    "llama": {
        "primary": "fnlp/Llama-Scope-8B-Base-LXR-32x-TopK",
        "fallback": "EleutherAI/sae-llama-3.1-8b-32x",
    },
    "gemma": {
        "primary": "google/gemma-scope-9b-it-res",
        "fallback": "google/gemma-scope-9b-pt-res",
    },
}


def load_sae(
    model_name: str,
    layer: int,
    trained_saes_dir: str = "trained_saes/",
):
    """
    Load the appropriate SAE for a given model and layer.

    For models with pre-trained SAEs (Llama, Gemma), loads from HuggingFace.
    For models without (Qwen), loads custom-trained SAE from disk.

    Args:
        model_name: HuggingFace model name.
        layer: Layer index.
        trained_saes_dir: Base directory for custom-trained SAEs.

    Returns:
        SAELens SAE object.
    """
    try:
        from sae_lens import SAE
    except ImportError:
        raise ImportError("SAELens is required. Install with: pip install sae-lens")

    from src.sae.train_sae import check_sae_availability, _model_short

    avail = check_sae_availability(model_name)
    model_short = _model_short(model_name)

    if avail["available"]:
        suite = avail["hf_repo"]
        # Try to construct the correct sae_id for the suite
        sae_id = _build_sae_id(model_short, layer)
        try:
            logger.info(f"Loading SAE from {suite}, sae_id={sae_id}, layer={layer}")
            sae, cfg_dict, feature_sparsity = SAE.from_pretrained(
                release=suite,
                sae_id=sae_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            return sae
        except Exception as e:
            logger.warning(f"Could not load from primary suite ({suite}): {e}")
            # Try fallback
            fallback = avail.get("fallback")
            if fallback:
                try:
                    logger.info(f"Trying fallback SAE suite: {fallback}")
                    sae, _, _ = SAE.from_pretrained(
                        release=fallback,
                        sae_id=sae_id,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    return sae
                except Exception as e2:
                    logger.error(f"Fallback SAE also failed: {e2}")
            raise
    else:
        # Load custom-trained SAE from disk
        sae_path = Path(trained_saes_dir) / model_short / f"layer_{layer}" / "sae"
        if not sae_path.exists():
            raise FileNotFoundError(
                f"No custom SAE found at {sae_path}. "
                "Run script 11 (SAE feature analysis) which calls train_sae first."
            )
        logger.info(f"Loading custom-trained SAE from {sae_path}")
        sae = SAE.load_from_disk(str(sae_path))
        return sae


def _build_sae_id(model_short: str, layer: int) -> str:
    """Build SAE ID string for a given model and layer."""
    if model_short == "gemma":
        return f"layer_{layer}/width_16k/average_l0_100"
    elif model_short == "llama":
        return f"layer_{layer}"
    return f"layer_{layer}"


def encode_activations(sae, activations: torch.Tensor) -> torch.Tensor:
    """
    Encode activations through the SAE encoder.

    Args:
        sae: SAELens SAE object.
        activations: (n_prompts, hidden_dim) tensor.

    Returns:
        Feature activations of shape (n_prompts, sae_width).
    """
    sae.eval()
    with torch.no_grad():
        feature_acts = sae.encode(activations.to(sae.device))
    return feature_acts


def decode_features(sae, feature_activations: torch.Tensor) -> torch.Tensor:
    """
    Decode SAE feature activations back to residual stream space.

    Args:
        sae: SAELens SAE object.
        feature_activations: (n_prompts, sae_width) tensor.

    Returns:
        Reconstructed activations (n_prompts, hidden_dim).
    """
    sae.eval()
    with torch.no_grad():
        reconstructed = sae.decode(feature_activations)
    return reconstructed


def get_top_features(feature_activations: torch.Tensor, k: int = 50) -> List[int]:
    """
    Get indices of top-k features by mean activation magnitude.

    Args:
        feature_activations: (n_prompts, sae_width) tensor.
        k: Number of top features to return.

    Returns:
        List of feature indices sorted by mean activation (descending).
    """
    mean_acts = feature_activations.abs().mean(dim=0)  # (sae_width,)
    top_k = torch.topk(mean_acts, min(k, len(mean_acts))).indices
    return top_k.cpu().tolist()
