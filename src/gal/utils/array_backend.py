"""Helpers for selecting NumPy or PyTorch array backends at runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArrayBackend:
    """Minimal facade over NumPy/PyTorch operations used in the search code."""

    name: str
    xp: Any
    _asarray: Callable[[Any], Any]
    _to_cpu: Callable[[Any], Any]

    def asarray(self, array: Any) -> Any:
        return self._asarray(array)

    def to_cpu(self, array: Any) -> Any:
        return self._to_cpu(array)

    def scalar(self, value: Any) -> float:
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        value_cpu = self.to_cpu(value)
        if isinstance(value_cpu, (float, int, np.floating, np.integer)):
            return float(value_cpu)
        return float(np.asarray(value_cpu).item())

    def bool_scalar(self, value: Any) -> bool:
        return bool(self.scalar(value))


def _numpy_backend() -> ArrayBackend:
    return ArrayBackend(
        name="cpu",
        xp=np,
        _asarray=lambda array: np.asarray(array, dtype=float),
        _to_cpu=lambda array: array,
    )


def _torch_backend() -> ArrayBackend | None:
    try:
        import torch  # type: ignore
    except ImportError:
        logger.warning("PyTorch not available; falling back to NumPy backend.")
        return None

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to NumPy backend.")
        return None

    device = torch.device("cuda")

    def _asarray(array: Any) -> Any:
        return torch.as_tensor(array, device=device, dtype=torch.float64)

    def _to_cpu(array: Any) -> Any:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return array

    return ArrayBackend(name="torch", xp=torch, _asarray=_asarray, _to_cpu=_to_cpu)


_CPU_BACKEND = _numpy_backend()


def get_array_backend(use_gpu: bool) -> ArrayBackend:
    if not use_gpu:
        return _CPU_BACKEND
    torch_backend = _torch_backend()
    if torch_backend is None:
        return _CPU_BACKEND
    return torch_backend


__all__ = ["ArrayBackend", "get_array_backend"]
