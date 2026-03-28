"""
SAE feature clamping intervention.

Clamp selected SAE features to reference values during generation.
"""

import logging
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


def _extract_latent_tensor(encoded):
    """
    Normalize SAE encoder outputs into a tensor of latent activations.
    Works across different SAE libraries / wrapper return types.
    """
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

    if hasattr(encoded, "z"):
        feats = getattr(encoded, "z")
        if isinstance(feats, torch.Tensor):
            return feats

    if isinstance(encoded, dict):
        for key in ("features", "acts", "latents", "codes", "latent_acts", "z"):
            if key in encoded and isinstance(encoded[key], torch.Tensor):
                return encoded[key]

    if isinstance(encoded, (tuple, list)):
        for item in encoded:
            if isinstance(item, torch.Tensor):
                return item

    raise TypeError(
        f"Could not extract latent tensor from SAE encode output of type {type(encoded)}"
    )


def _decode_with_sae(sae, z):
    """
    Decode latent features back into activation space.
    """
    if hasattr(sae, "decode"):
        return sae.decode(z)
    if hasattr(sae, "decoder"):
        return sae.decoder(z)
    raise AttributeError("SAE object has neither .decode() nor .decoder().")


def apply_sae_clamping(
    model,
    tokenizer,
    prompts: List[dict],
    sae,
    ranked_features: List[int],
    clamp_values: Dict[int, float],
    layer: int,
    n_features: int = 10,
    max_new_tokens: int = 128,
    batch_size: int = 8,
):
    """
    Apply SAE clamping during generation by intercepting one transformer layer,
    encoding activations into SAE space, clamping top features, and decoding back.
    """
    device = next(model.parameters()).device
    selected_features = [int(f) for f in ranked_features[:n_features]]

    logger.info(
        "Applying SAE clamp at layer=%s with %s features.",
        layer,
        len(selected_features),
    )

    # Pre-build clamp tensors on device for speed / dtype safety
    clamp_tensors = {
        int(fidx): torch.tensor(float(clamp_values[int(fidx)]), device=device)
        for fidx in selected_features
        if int(fidx) in clamp_values
    }

    def hook_fn(module, inputs, output):
        """
        output is usually hidden_states with shape [batch, seq, hidden].
        We encode the last-token activations, clamp chosen SAE latents,
        decode them, and write them back into the last token position.
        """
        hidden = output[0] if isinstance(output, tuple) else output

        if not isinstance(hidden, torch.Tensor):
            return output

        # Operate on final token only
        last_hidden = hidden[:, -1, :]  # [B, H]

        try:
            encoded = sae.encode(last_hidden)
            z = _extract_latent_tensor(encoded)

            if z.ndim != 2:
                logger.warning(f"Unexpected SAE latent shape: {tuple(z.shape)}")
                return output

            z = z.clone()

            for fidx in selected_features:
                if fidx >= z.shape[1]:
                    continue
                if fidx not in clamp_tensors:
                    continue
                z[:, fidx] = clamp_tensors[fidx].to(dtype=z.dtype)

            recon = _decode_with_sae(sae, z)

            if not isinstance(recon, torch.Tensor):
                logger.warning(f"Unexpected SAE decode output type: {type(recon)}")
                return output

            if recon.shape != last_hidden.shape:
                logger.warning(
                    f"Decoded SAE reconstruction shape mismatch: "
                    f"{tuple(recon.shape)} vs {tuple(last_hidden.shape)}"
                )
                return output

            hidden = hidden.clone()
            hidden[:, -1, :] = recon.to(hidden.dtype)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        except Exception as e:
            logger.warning(f"SAE clamp hook failed: {e}")
            return output

    # Locate transformer block list robustly
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        target_layer = model.model.layers[layer]
    elif hasattr(model, "layers"):
        target_layer = model.layers[layer]
    else:
        raise AttributeError("Could not find transformer layers on model.")

    handle = target_layer.register_forward_hook(hook_fn)

    responses = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            texts = [p["prompt"] if isinstance(p, dict) else str(p) for p in batch_prompts]

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            responses.extend(decoded)

    finally:
        handle.remove()

    return responses