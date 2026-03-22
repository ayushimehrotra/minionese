"""
Reproducibility utilities: seed setting and deterministic ops.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"All seeds set to {seed}.")
    except ImportError:
        logger.warning("PyTorch not available; only Python/NumPy seeds set.")


def enable_deterministic(warn_on_failure: bool = True) -> None:
    """
    Enable deterministic algorithms in PyTorch where possible.

    Some CUDA operations do not have deterministic implementations.
    When `warn_on_failure=True`, a warning is issued instead of raising.

    Args:
        warn_on_failure: If True, catch errors from deterministic ops and log a warning.
    """
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=warn_on_failure)
        # Ensure deterministic cuDNN benchmark is disabled
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("Deterministic algorithms enabled.")
    except ImportError:
        logger.warning("PyTorch not available; cannot enable deterministic algorithms.")
    except Exception as e:
        if warn_on_failure:
            logger.warning(f"Could not fully enable deterministic algorithms: {e}")
        else:
            raise


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """
    Convenience wrapper: set seed and optionally enable deterministic ops.

    Args:
        seed: Random seed.
        deterministic: Whether to call enable_deterministic().
    """
    set_seed(seed)
    if deterministic:
        enable_deterministic(warn_on_failure=True)
