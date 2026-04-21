"""
Activation Caching Utilities

Save and load activation tensors from disk using safetensors format.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def save_activations(activations: Dict[str, torch.Tensor], path: str) -> None:
    """
    Save activations dict to a safetensors file with metadata.

    Args:
        activations: Dict mapping tensor names to tensors.
        path: Output file path (should end with .safetensors).
    """
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(activations, path)
    logger.debug(f"Saved activations to {path}.")


def load_activations(
    path: str,
    layers: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Load activations from a safetensors file.

    Args:
        path: Path to the .safetensors file.
        layers: Optional list of layer indices to load (selects along dim 1).
                If None, load all layers.

    Returns:
        Tensor of shape (n_prompts, n_layers, hidden_dim) or
        (n_prompts, hidden_dim) if single-layer.
    """
    from safetensors.torch import load_file

    data = load_file(path)
    # Expect a single "activations" key by convention
    if "activations" in data:
        tensor = data["activations"]
    else:
        # Fall back: take the first tensor
        tensor = next(iter(data.values()))

    if layers is not None and tensor.dim() >= 2:
        # Shape: (n_prompts, n_layers, hidden_dim) -> select layers
        indices = torch.tensor(layers, dtype=torch.long)
        tensor = tensor[:, indices]

    return tensor


def get_activation_path(
    model: str,
    language: str,
    perturbation: str,
    position: str,
    component: str,
    cache_dir: str = "data/activations/",
) -> str:
    """
    Construct the deterministic file path for a cached activation file.

    Args:
        model: Short model name (e.g., "llama", "gemma", "qwen").
        language: Language code.
        perturbation: Perturbation type.
        position: Token position name.
        component: Component name ("residual", "attn_out", "mlp_out").
        cache_dir: Root cache directory.

    Returns:
        Full path string.
    """
    fname = f"{model}_{language}_{perturbation}_{position}_{component}.safetensors"
    return str(Path(cache_dir) / fname)


def activation_exists(
    model: str,
    language: str,
    perturbation: str,
    position: str,
    component: str,
    cache_dir: str = "data/activations/",
) -> bool:
    """
    Check whether a cached activation file exists.

    Args:
        model: Short model name.
        language: Language code.
        perturbation: Perturbation type.
        position: Token position name.
        component: Component name.
        cache_dir: Root cache directory.

    Returns:
        True if the file exists on disk.
    """
    path = get_activation_path(model, language, perturbation, position, component, cache_dir)
    return Path(path).exists()


def list_cached_activations(cache_dir: str = "data/activations/") -> List[str]:
    """
    List all cached activation files in a directory.

    Args:
        cache_dir: Root cache directory.

    Returns:
        List of file paths.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(str(p) for p in cache_dir.glob("*.safetensors"))
