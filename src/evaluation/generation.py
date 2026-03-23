"""
Model Generation Pipeline

Run all prompts through target models with greedy decoding and collect responses.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_responses(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    output_path: Optional[str] = None,
    prompt_ids: Optional[List[str]] = None,
    hf_token: Optional[str] = None,   # add this
) -> List[dict]:
    """
    Run prompts through a causal language model and collect responses.

    Args:
        model_name: HuggingFace model identifier.
        prompts: List of already-formatted prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature (0.0 = greedy).
        batch_size: Number of prompts per GPU batch.
        output_path: If given, append results to this JSONL file incrementally.
        prompt_ids: Optional list of IDs corresponding to each prompt.

    Returns:
        List of result dicts, one per prompt.
    """
    logger.info(f"HF token present? {hf_token is not None}")
    if hf_token is not None:
        logger.info(f"HF token prefix: {hf_token[:8]}")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.gpu import log_gpu_memory, clear_gpu_memory

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log_gpu_memory("after model load")

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Load already-completed prompt IDs to support resume
        completed_ids = set()
        if Path(output_path).exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_ids.add(rec.get("prompt_id", ""))
                    except json.JSONDecodeError:
                        pass
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} prompts already completed.")

    all_results = []
    n = len(prompts)

    for batch_start in range(0, n, batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        batch_ids = (
            prompt_ids[batch_start : batch_start + batch_size]
            if prompt_ids is not None
            else [f"prompt_{batch_start + i}" for i in range(len(batch_prompts))]
        )

        # Skip already-completed prompts
        to_run_indices = []
        for i, pid in enumerate(batch_ids):
            if output_path is not None and pid in completed_ids:
                continue
            to_run_indices.append(i)

        if not to_run_indices:
            logger.debug(f"Batch {batch_start//batch_size}: all already completed, skipping.")
            continue

        run_prompts = [batch_prompts[i] for i in to_run_indices]
        run_ids = [batch_ids[i] for i in to_run_indices]

        t0 = time.time()
        inputs = tokenizer(
            run_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        elapsed = time.time() - t0

        for j, (pid, out) in enumerate(zip(run_ids, output_ids)):
            input_len = inputs["input_ids"].shape[1]
            new_tokens = out[input_len:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            result = {
                "prompt_id": pid,
                "model": model_name,
                "prompt": run_prompts[j],
                "response": response_text,
                "num_tokens_generated": len(new_tokens),
                "generation_time_s": round(elapsed / len(run_prompts), 3),
            }
            all_results.append(result)

            if output_path is not None:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        log_gpu_memory(f"batch {batch_start//batch_size}")
        logger.info(
            f"Batch {batch_start//batch_size + 1}/{(n + batch_size - 1)//batch_size}: "
            f"generated {len(run_prompts)} responses in {elapsed:.1f}s."
        )

    clear_gpu_memory()
    logger.info(f"Generation complete. Total responses: {len(all_results)}.")
    return all_results


def load_responses(path: str) -> List[dict]:
    """Load previously saved JSONL responses file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
