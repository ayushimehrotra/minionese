#!/usr/bin/env python3
"""
Utilities for loading sparse autoencoders (SAEs) and encoding activations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def _hf_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def _normalize_model_name(model_name: str) -> str:
    if not model_name:
        return ""
    name = model_name.lower().strip()
    if any(x in name for x in [
        "meta-llama/llama-3.1-8b",
        "llama-3.1-8b",
        "llama3.1-8b",
        "llama_3.1_8b",
        "llama",
    ]):
        return "llama-3.1-8b"
    return name


def _get_hookpoint(layer: int, hook_component: str = "mlp") -> str:
    comp = hook_component.lower().strip()
    if comp in {"mlp", "mlp_out"}:
        return f"layers.{layer}.mlp"
    if comp in {"resid", "residual", "resid_post", "hidden", "hidden_state"}:
        return f"layers.{layer}"
    if comp in {"attn", "attn_out"}:
        return f"layers.{layer}.attn"
    raise ValueError(f"Unsupported hook_component: {hook_component}")


def _to_device_dtype(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(dtype=torch.float32)


def load_sae(model_name: str, layer: int, hook_component: str = "mlp") -> Any:
    """
    Load an SAE for a given model/layer/component.

    This version uses the public 64x Llama 3.1 8B SAE repo, which is documented
    to load via the `sparsify` library.
    """
    norm_model = _normalize_model_name(model_name)
    logger.info(
        "Requested SAE load: model_name=%s normalized=%s layer=%s hook_component=%s",
        model_name,
        norm_model,
        layer,
        hook_component,
    )

    if "llama-3.1-8b" not in norm_model and norm_model != "llama":
        logger.warning(
            "This loader is primarily configured for Llama 3.1 8B SAEs. "
            "Received model_name=%s",
            model_name,
        )

    hookpoint = _get_hookpoint(layer, hook_component=hook_component)

    # Use 64x because the public model card explicitly documents sparsify loading.
    repo_id = "EleutherAI/sae-llama-3.1-8b-64x"

    try:
        from sparsify import Sae
    except ImportError as e:
        raise RuntimeError(
            "Missing SAE loader package. Install it with:\n"
            "pip install -U eai-sparsify"
        ) from e

    kwargs = {}
    token = _hf_token()
    if token:
        kwargs["token"] = token

    logger.info(
        "Loading EleutherAI SAE via sparsify.Sae.load_from_hub(repo=%s, hookpoint=%s)",
        repo_id,
        hookpoint,
    )

    try:
        sae = Sae.load_from_hub(repo_id, hookpoint=hookpoint, **kwargs)
    except TypeError:
        sae = Sae.load_from_hub(repo_id, hookpoint=hookpoint)

    if hasattr(sae, "eval"):
        sae.eval()

    return sae


def _extract_feature_tensor(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, torch.Tensor):
        return encoded

    if hasattr(encoded, "features"):
        feats = getattr(encoded, "features")
        if isinstance(feats, torch.Tensor):
            return feats

    if hasattr(encoded, "acts"):
        feats = getattr(encoded, "acts")
        if isinstance(feats, torch.Tensor):
            return feats

    if hasattr(encoded, "latents"):
        feats = getattr(encoded, "latents")
        if isinstance(feats, torch.Tensor):
            return feats

    if isinstance(encoded, dict):
        for key in ("features", "acts", "codes", "latent_acts", "z", "latents"):
            if key in encoded and isinstance(encoded[key], torch.Tensor):
                return encoded[key]

    if isinstance(encoded, (tuple, list)):
        for item in encoded:
            if isinstance(item, torch.Tensor):
                return item

    raise TypeError(
        f"Could not extract feature tensor from SAE encode output of type {type(encoded)}"
    )


def _encode_once(sae: Any, batch: torch.Tensor) -> torch.Tensor:
    if hasattr(sae, "encode"):
        out = sae.encode(batch)
        return _extract_feature_tensor(out)

    if hasattr(sae, "encoder"):
        out = sae.encoder(batch)
        return _extract_feature_tensor(out)

    if callable(sae):
        out = sae(batch)
        return _extract_feature_tensor(out)

    raise AttributeError(
        "SAE object has neither .encode(), .encoder(), nor a callable forward path."
    )


@torch.no_grad()
def encode_activations(
    sae: Any,
    activations: torch.Tensor,
    batch_size: int = 1024,
) -> torch.Tensor:
    x = _to_device_dtype(activations)

    original_shape = tuple(x.shape)
    if x.ndim < 2:
        raise ValueError(f"Expected activations with ndim >= 2, got shape {original_shape}")

    hidden_dim = x.shape[-1]
    x = x.reshape(-1, hidden_dim)

    sae_device = None
    if hasattr(sae, "parameters"):
        try:
            sae_device = next(sae.parameters()).device
        except Exception:
            sae_device = None

    if sae_device is None:
        sae_device = x.device

    outputs = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start:start + batch_size].to(sae_device)
        feats = _encode_once(sae, batch)
        feats = feats.detach().to("cpu")
        outputs.append(feats)

    result = torch.cat(outputs, dim=0)

    logger.info(
        "Encoded activations: input_shape=%s flat_input=%s output_shape=%s",
        original_shape,
        tuple(x.shape),
        tuple(result.shape),
    )
    return result