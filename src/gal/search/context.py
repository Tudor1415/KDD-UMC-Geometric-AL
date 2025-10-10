"""Context objects shared across search engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils import ArrayBackend


@dataclass(frozen=True)
class SearchContext:
    """Per-search snapshot shared by leaf evaluators and traversal routines."""

    data: np.ndarray
    wc: np.ndarray
    tau: float
    eps: float
    seen_pairs: frozenset[tuple[int, int]]
    backend: ArrayBackend
    orientation: Optional[np.ndarray] = None
    orientation_mode: bool = False


__all__ = ["SearchContext"]
