"""
Activation Extraction via NNsight

Extract residual stream, attention output, and MLP output activations
at every layer for every prompt.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_activations(
    model_name: str,
    prompts: List[str],
    token_positions: List[str],
    layers: Union[List[int], str],
    output_dir: str,
    batch_size: int = 4,
    dtype: str = "float32",
    components: List[str] = None,
    prompt_metadata: Optional[List[dict]] = None,
) -> None:
    """
    Extract activations using NNsight and save to disk.

    Saves as .safetensors files:
      {output_dir}/{model_short}_{lang}_{pert}_{position}_{component}.safetensors

    Args:
        model_name: HuggingFace model identifier.
        prompts: List of formatted prompt strings.
        token_positions: List of position names to extract at.
        layers: List of layer indices or "all".
        output_dir: Directory to save activation files.
        batch_size: Number of prompts per forward pass.
        dtype: Storage dtype ("float32" or "float16").
        components: Which components to extract. Default: ["residual", "attn_out", "mlp_out"].
        prompt_metadata: Optional list of dicts with 'language', 'perturbation' keys
                         used for file naming.
    """
    from nnsight import LanguageModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from src.activations.positions import find_token_positions
    from src.utils.gpu import log_gpu_memory, clear_gpu_memory

    if components is None:
        components = ["residual", "attn_out", "mlp_out"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float16

    logger.info(f"Loading model via NNsight: {model_name}")
    lm = LanguageModel(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        dispatch=True,
    )
    logger.info(f"NNsight model device: {lm.device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Determine number of layers
    try:
        num_layers = lm.config.num_hidden_layers
    except AttributeError:
        num_layers = len(lm.model.layers)

    if layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = list(layers)

    log_gpu_memory("after model load")

    # Accumulators: {(position, component, layer): list of tensors}
    accumulators: Dict = {
        (pos, comp, layer): []
        for pos in token_positions
        for comp in components
        for layer in layer_indices
    }

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs["input_ids"]

        # Find token positions directly from the actual padded token IDs so that
        # the positions are always consistent with what the model sees.
        pos_indices: Dict[str, List[int]] = {pos: [] for pos in token_positions}
        seq_len = input_ids.shape[1]
        pad_id = tokenizer.pad_token_id
        for sample_idx in range(len(batch_prompts)):
            ids = input_ids[sample_idx]   # (seq_len,) — includes left-pad
            pad_count = int((ids == pad_id).sum().item())
            # Indices of actual (non-pad) tokens in the padded sequence
            first_real = pad_count          # index of first real token
            last_real = seq_len - 1         # index of last token (always real after pad)

            pos_map: Dict[str, int] = {}

            # last_instruction: last token of user message, i.e., token just before
            # the LAST <|eot_id|> in the sequence.  We search the padded IDs directly.
            model_lower = model_name.lower()
            if "gemma" in model_lower:
                end_inst_ids = tokenizer.convert_tokens_to_ids(["<end_of_turn>"])
            elif "qwen" in model_lower:
                end_inst_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>"])
            elif "aya" in model_lower or "cohere" in model_lower:
                end_inst_ids = tokenizer.convert_tokens_to_ids(["<|END_OF_TURN_TOKEN|>"])
            else:
                end_inst_ids = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
            end_inst_ids = [i for i in end_inst_ids if i != tokenizer.unk_token_id]

            last_eot_pos = -1
            for t in range(last_real, first_real - 1, -1):
                if ids[t].item() in end_inst_ids:
                    last_eot_pos = t
                    break
            pos_map["last_instruction"] = max(first_real, last_eot_pos - 1) if last_eot_pos > first_real else last_real

            # last_post_instruction: last token of the assistant-turn header
            # (<|end_header_id|> for Llama, <start_of_turn>model for Gemma, etc.)
            if "gemma" in model_lower:
                asst_end_ids = tokenizer.convert_tokens_to_ids(["<start_of_turn>"])
            elif "qwen" in model_lower:
                asst_end_ids = tokenizer.convert_tokens_to_ids(["<|im_start|>"])
            elif "aya" in model_lower or "cohere" in model_lower:
                asst_end_ids = tokenizer.convert_tokens_to_ids(["<|START_OF_TURN_TOKEN|>"])
            else:
                asst_end_ids = tokenizer.convert_tokens_to_ids(["<|end_header_id|>"])
            asst_end_ids = [i for i in asst_end_ids if i != tokenizer.unk_token_id]

            last_asst_end = -1
            for t in range(last_real, first_real - 1, -1):
                if ids[t].item() in asst_end_ids:
                    last_asst_end = t
                    break
            pos_map["last_post_instruction"] = last_asst_end if last_asst_end >= 0 else last_real

            # last: the final real token in the padded sequence
            pos_map["last"] = last_real

            for pos_name in token_positions:
                idx = pos_map.get(pos_name, last_real)
                pos_indices[pos_name].append(min(idx, seq_len - 1))

        # NNsight trace
        saved = {}
        with torch.no_grad():
            with lm.trace(input_ids.to(lm.device)):
                
        
                for layer_idx in layer_indices:
                    try:
                        layer = lm.model.layers[layer_idx]
                    except AttributeError:
                        layer = lm.model.model.layers[layer_idx]
        
                    # Access in forward order
                    attn_out = None
                    if "attn_out" in components:
                        attn_out = layer.self_attn.o_proj.output
        
                    mlp_out = None
                    if "mlp_out" in components:
                        mlp_out = layer.mlp.output
        
                    resid = None
                    if "residual" in components:
                        resid = layer.output
        
                    for pos_name in token_positions:
                        if attn_out is not None:
                            saved[(pos_name, "attn_out", layer_idx)] = attn_out.save()
                        if mlp_out is not None:
                            saved[(pos_name, "mlp_out", layer_idx)] = mlp_out.save()
                        if resid is not None:
                            saved[(pos_name, "residual", layer_idx)] = resid.save()

        # Extract per-position activations from saved tensors
        for layer_idx in layer_indices:
            for pos_name in token_positions:
                for comp in components:
                    key = (pos_name, comp, layer_idx)
                    
                    
                    
                    
                    tensor = saved.get(key)
                    if tensor is None:
                        continue
                    
                    batch_extracted = []
                    
                    if tensor.ndim == 3:
                        # expected: (batch, seq, hidden)
                        logger.info(f"{key} tensor shape: {tuple(tensor.shape)}")
                        for sample_idx in range(tensor.shape[0]):
                            tok_idx = pos_indices[pos_name][sample_idx]
                            tok_idx = max(0, min(tok_idx, tensor.shape[1] - 1))
                            vec = tensor[sample_idx, tok_idx, :].detach().to(torch_dtype).cpu()
                            batch_extracted.append(vec)
                    
                    elif tensor.ndim == 2:
                        # likely single-example trace: (seq, hidden)
                        if len(batch_prompts) != 1:
                            raise RuntimeError(
                                f"Got 2D tensor for key={key} with batch size {len(batch_prompts)}; "
                                f"shape={tuple(tensor.shape)}"
                            )
                        tok_idx = pos_indices[pos_name][0]
                        tok_idx = max(0, min(tok_idx, tensor.shape[0] - 1))
                        vec = tensor[tok_idx, :].detach().to(torch_dtype).cpu()
                        batch_extracted.append(vec)
                    
                    else:
                        raise RuntimeError(f"Unexpected tensor rank for key={key}: shape={tuple(tensor.shape)}")
                    
                    accumulators[key].extend(batch_extracted)

        logger.info(
            f"Extracted batch {batch_start // batch_size + 1}/"
            f"{(len(prompts) + batch_size - 1) // batch_size}"
        )
        log_gpu_memory(f"batch {batch_start}")

    # Save to disk, one file per (position, component)
    # Shape: (n_prompts, n_layers, hidden_dim)
    for pos_name in token_positions:
        for comp in components:
            layer_tensors = []
            for layer_idx in layer_indices:
                key = (pos_name, comp, layer_idx)
                if not accumulators[key]:
                    continue
                stacked = torch.stack(accumulators[key], dim=0)  # (n_prompts, hidden)
                layer_tensors.append(stacked)

            if not layer_tensors:
                continue

            # Stack along new layer dimension: (n_prompts, n_layers, hidden)
            all_layers = torch.stack(layer_tensors, dim=1)

            # Determine filename
            if prompt_metadata:
                # Use first item's metadata for naming (assume homogeneous batch)
                meta = prompt_metadata[0]
                lang = meta.get("language", "unknown")
                pert = meta.get("perturbation", "unknown")
                model_short = _model_short_name(model_name)
                fname = f"{model_short}_{lang}_{pert}_{pos_name}_{comp}.safetensors"
            else:
                model_short = _model_short_name(model_name)
                fname = f"{model_short}_{pos_name}_{comp}.safetensors"

            save_path = output_dir / fname
            save_file({"activations": all_layers}, str(save_path))
            logger.info(f"Saved activations: {save_path} shape={all_layers.shape}")

    del lm
    clear_gpu_memory()
    logger.info("Activation extraction complete.")


def _model_short_name(model_name: str) -> str:
    """Extract a short model identifier from the full model name."""
    name = model_name.split("/")[-1].lower()
    # Simplify common names
    if "llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    if "aya" in name or "cohere" in name:
        return "aya"
    return name[:20].replace("-", "_")
