"""
Token Position Selection

Utilities for identifying meaningful token positions in formatted prompts:
- last_instruction: last token of the user message (t_inst from Zhao et al.)
- last_post_instruction: last token of the full formatted prompt (t_post_inst)
- last: same as last_post_instruction for single-turn prompts
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Template boundary markers for each model family
LLAMA_END_INST_TOKENS = ["[/INST]", "<|eot_id|>"]
GEMMA_END_INST_TOKENS = ["<end_of_turn>"]
QWEN_END_INST_TOKENS = ["<|im_end|>"]

# Beginning-of-assistant-turn markers (after these, generation starts)
LLAMA_ASST_START = ["<|start_header_id|>assistant<|end_header_id|>"]
GEMMA_ASST_START = ["<start_of_turn>model"]
QWEN_ASST_START = ["<|im_start|>assistant"]


def find_token_positions(
    tokenizer,
    formatted_prompt: str,
    model_name: str = "",
) -> Dict[str, int]:
    """
    Find key token positions in a formatted prompt.

    Args:
        tokenizer: HuggingFace tokenizer.
        formatted_prompt: The fully formatted prompt string (output of apply_chat_template).
        model_name: Model name to select the correct template markers.

    Returns:
        Dict mapping position_name -> token_index (0-based, from start of sequence).
        Keys: "last_instruction", "last_post_instruction", "last"
    """
    token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    n = len(token_ids)

    if n == 0:
        return {"last_instruction": 0, "last_post_instruction": 0, "last": 0}

    # Determine template style
    model_lower = model_name.lower()
    if "gemma" in model_lower:
        end_inst_markers = GEMMA_END_INST_TOKENS
        asst_start_markers = GEMMA_ASST_START
    elif "qwen" in model_lower:
        end_inst_markers = QWEN_END_INST_TOKENS
        asst_start_markers = QWEN_ASST_START
    else:
        # Default: Llama-style
        end_inst_markers = LLAMA_END_INST_TOKENS
        asst_start_markers = LLAMA_ASST_START

    # Find last_instruction: the last token of the user message
    # We do this by finding the end-of-user-turn marker in the token sequence
    last_instruction_idx = _find_last_marker_position(
        tokenizer, token_ids, formatted_prompt, end_inst_markers
    )

    # last_post_instruction: the last token before generation begins
    # This is the token just before the model's response starts
    last_post_inst_idx = _find_last_marker_position(
        tokenizer, token_ids, formatted_prompt, asst_start_markers
    )
    if last_post_inst_idx == 0:
        last_post_inst_idx = n - 1  # fallback to last token

    return {
        "last_instruction": max(0, last_instruction_idx - 1),
        "last_post_instruction": last_post_inst_idx,
        "last": n - 1,
    }


def _find_last_marker_position(tokenizer, token_ids: List[int], text: str, markers: List[str]) -> int:
    """
    Find the token index just after the last occurrence of any marker string.

    Args:
        tokenizer: HuggingFace tokenizer.
        token_ids: Full token ID sequence.
        text: The original text (for character-level search).
        markers: List of marker strings to search for.

    Returns:
        Token index after the last marker, or 0 if not found.
    """
    best_char_pos = -1
    for marker in markers:
        pos = text.rfind(marker)
        if pos > best_char_pos:
            best_char_pos = pos

    if best_char_pos < 0:
        return 0

    # Count tokens up to character position
    char_count = 0
    for i, tok_id in enumerate(token_ids):
        tok_str = tokenizer.decode([tok_id])
        char_count += len(tok_str)
        if char_count > best_char_pos:
            return i + 1

    return len(token_ids)


def get_position_index(
    tokenizer,
    formatted_prompt: str,
    position_name: str,
    model_name: str = "",
) -> int:
    """
    Get the token index for a named position.

    Args:
        tokenizer: HuggingFace tokenizer.
        formatted_prompt: Formatted prompt string.
        position_name: One of "last_instruction", "last_post_instruction", "last".
        model_name: Model name for template detection.

    Returns:
        Token index (0-based).
    """
    positions = find_token_positions(tokenizer, formatted_prompt, model_name)
    if position_name not in positions:
        raise ValueError(
            f"Unknown position name '{position_name}'. "
            f"Valid: {list(positions.keys())}"
        )
    return positions[position_name]
