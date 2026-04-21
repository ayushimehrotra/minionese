"""
GPU memory management utilities.
"""

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    """
    Return current GPU memory usage statistics.

    Returns:
        Dict with keys: allocated_gb, reserved_gb, free_gb, total_gb
        (or empty dict if CUDA is not available).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "free_gb": round(free, 3),
            "total_gb": round(total, 3),
        }
    except Exception as e:
        logger.warning(f"Could not query GPU memory: {e}")
        return {}


def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    if info:
        logger.info(
            f"GPU memory [{tag}]: "
            f"allocated={info['allocated_gb']:.2f}GB / "
            f"reserved={info['reserved_gb']:.2f}GB / "
            f"free={info['free_gb']:.2f}GB / "
            f"total={info['total_gb']:.2f}GB"
        )


def clear_gpu_memory() -> None:
    """
    Free unused GPU memory by running garbage collection and emptying the
    CUDA cache.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Could not clear GPU memory: {e}")


def delete_model_from_gpu(model) -> None:
    """
    Delete a model and free its GPU memory.

    Args:
        model: A torch.nn.Module instance.
    """
    try:
        import torch
        model.cpu()
        del model
        clear_gpu_memory()
        logger.info("Model deleted from GPU and memory cleared.")
    except Exception as e:
        logger.warning(f"Error deleting model from GPU: {e}")


def estimate_model_memory_gb(num_params: int, dtype_bytes: int = 2) -> float:
    """
    Estimate GPU memory required to load a model.

    Args:
        num_params: Number of model parameters.
        dtype_bytes: Bytes per parameter (2 for float16/bfloat16, 4 for float32).

    Returns:
        Estimated memory in GB.
    """
    return (num_params * dtype_bytes) / 1e9
