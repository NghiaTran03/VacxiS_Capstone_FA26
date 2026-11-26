"""CUDA OOM auto-backoff helper for batched model inference."""
from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TypeVar

import torch

logger = logging.getLogger("vacxis.retrieval.batching")

_T = TypeVar("_T")


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True when an exception indicates a CUDA out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def run_with_oom_backoff(
    fn: Callable[[int], _T],
    batch_size: int,
    *,
    min_batch_size: int = 1,
    operation: str = "inference",
) -> _T:
    """Execute a batched inference callable, halving batch_size on CUDA OOM.

    Args:
        fn: Callable receiving the active ``batch_size`` and returning inference output.
        batch_size: Initial positive batch size.
        min_batch_size: Minimum batch size before re-raising OOM (default: 1).
        operation: Descriptive label used in warning logs.

    Returns:
        Result produced by ``fn(current_batch_size)``.

    Raises:
        ValueError: If ``batch_size`` or ``min_batch_size`` is not a positive integer.
        RuntimeError: Re-raises the OOM error if execution still fails at ``min_batch_size``.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if type(min_batch_size) is not int or min_batch_size < 1:
        raise ValueError("min_batch_size must be a positive integer.")

    current_bs = batch_size
    while True:
        try:
            return fn(current_bs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom(exc) or current_bs <= min_batch_size:
                raise
            next_bs = max(min_batch_size, current_bs // 2)
            logger.warning(
                "CUDA OOM during %s with batch_size=%d; clearing CUDA cache and retrying with batch_size=%d.",
                operation,
                current_bs,
                next_bs,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_bs = next_bs
