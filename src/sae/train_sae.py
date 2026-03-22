"""
SAE Training for Models Without Public Suites

Currently applies to Qwen2.5-7B-Instruct.
Gemma-2-9B and Llama-3.1-8B have pre-trained SAEs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Known SAE availability mapping
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
        "available": False,
        "source": "custom",
        "hf_repo": None,
        "fallback": None,
    },
}


def check_sae_availability(model_name: str) -> dict:
    """
    Check whether a pre-trained SAE suite exists for the given model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Dict with keys: available (bool), source (str), hf_repo (str or None).
    """
    info = SAE_AVAILABILITY.get(model_name)
    if info is None:
        # Try fuzzy match
        for key, val in SAE_AVAILABILITY.items():
            if any(part in model_name.lower() for part in key.lower().split("/")):
                return val
        logger.warning(f"SAE availability unknown for {model_name}. Assuming not available.")
        return {"available": False, "source": "unknown", "hf_repo": None}
    return info


def train_sae_for_layer(
    model_name: str,
    layer: int,
    output_dir: str,
    config_overrides: Optional[dict] = None,
) -> str:
    """
    Train a TopK SAE for a single layer using SAELens.

    Args:
        model_name: HuggingFace model name.
        layer: Layer index to train SAE for.
        output_dir: Directory to save the trained SAE.
        config_overrides: Optional dict of SAELens config overrides.

    Returns:
        Path to the saved SAE directory.
    """
    try:
        from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
    except ImportError:
        raise ImportError(
            "SAELens is required for SAE training. "
            "Install with: pip install sae-lens"
        )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hidden_dim = getattr(config, "hidden_size", 3584)

    model_short = _model_short(model_name)
    layer_output_dir = str(Path(output_dir) / f"{model_short}" / f"layer_{layer}")
    Path(layer_output_dir).mkdir(parents=True, exist_ok=True)

    default_config = dict(
        model_name=model_name,
        hook_name=f"blocks.{layer}.hook_resid_post",
        hook_layer=layer,
        d_in=hidden_dim,
        dataset_path="HuggingFaceFW/fineweb-edu",
        streaming=True,
        context_size=1024,
        is_dataset_tokenized=False,
        architecture="topk",
        activation_fn="topk",
        k=64,
        d_sae=hidden_dim * 8,
        lr=3e-4,
        l1_coefficient=0,
        train_batch_size_tokens=4096,
        training_tokens=500_000_000,
        log_to_wandb=True,
        wandb_project="multilingual-jailbreak-sae",
        checkpoint_path=layer_output_dir,
    )

    if config_overrides:
        default_config.update(config_overrides)

    logger.info(f"Starting SAE training for {model_name} layer {layer}...")
    sae_config = LanguageModelSAERunnerConfig(**default_config)
    runner = SAETrainingRunner(sae_config)
    sae = runner.run()

    save_path = str(Path(layer_output_dir) / "sae")
    sae.save_model(save_path)
    logger.info(f"SAE saved to {save_path}.")
    return save_path


def train_all_critical_layers(
    model_name: str,
    critical_layers: List[int],
    output_dir: str,
    config_overrides: Optional[dict] = None,
) -> Dict[int, str]:
    """
    Train SAEs for all critical layers.

    Args:
        model_name: HuggingFace model name.
        critical_layers: List of layer indices.
        output_dir: Base output directory.
        config_overrides: Optional config overrides for each layer.

    Returns:
        Dict mapping layer_idx -> saved SAE path.
    """
    avail = check_sae_availability(model_name)
    if avail["available"]:
        logger.info(
            f"Pre-trained SAEs available for {model_name} at {avail['hf_repo']}. "
            "Skipping training."
        )
        return {l: avail["hf_repo"] for l in critical_layers}

    logger.info(
        f"No pre-trained SAEs for {model_name}. "
        f"Training at layers: {critical_layers}"
    )

    paths = {}
    for layer in critical_layers:
        try:
            path = train_sae_for_layer(
                model_name, layer, output_dir, config_overrides
            )
            paths[layer] = path
        except Exception as e:
            logger.error(f"SAE training failed for layer {layer}: {e}")
            paths[layer] = None

    return paths


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
