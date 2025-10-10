"""Visiting order strategies for the search engine."""

from __future__ import annotations

from typing import Generic, Optional, Tuple, TypeVar

import random
import numpy as np

from ..trees.common import Node
from .bounds import BoundsResult

TNode = TypeVar('TNode')


class VisitStrategy(Generic[TNode]):
    """Base class for visit-ordering strategies."""

    def setup(
        self,
        *,
        wc: np.ndarray | None = None,
        tau: float = float('inf'),
        eps: float = 1e-12,
        orientation: np.ndarray | None = None,
        orientation_mode: bool = False,
    ) -> None:
        """Prepare the strategy for a new search tree."""

    def priority(
        self,
        a: TNode,
        b: TNode,
        bounds: BoundsResult,
        *,
        mass: int,
    ) -> Optional[Tuple[float, ...]]:
        """Return a tuple used to order candidate node pairs or ``None`` to skip."""
        raise NotImplementedError

class LowerBoundVisitStrategy(VisitStrategy[Node]):
    """Simple ordering based on lower bounds and optional orientation cues."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng: random.Random = rng or random.Random()
        self._query: np.ndarray | None = None
        self._tau: float = float('inf')
        self._eps: float = 1e-12
        self._orientation: np.ndarray | None = None
        self._orientation_mode: bool = False

    def setup(
        self,
        *,
        wc: np.ndarray | None = None,
        tau: float = float('inf'),
        eps: float = 1e-12,
        orientation: np.ndarray | None = None,
        orientation_mode: bool = False,
    ) -> None:
        """Configure per-search state (no precomputation required)."""
        if wc is not None:
            self._query = np.asarray(wc, dtype=float)
        else:
            self._query = None
        self._tau = float(tau)
        self._eps = float(eps)

        self._orientation = None
        self._orientation_mode = False
        if orientation_mode and orientation is not None:
            vec = np.asarray(orientation, dtype=float).reshape(-1)
            norm1 = np.linalg.norm(vec, ord=1)
            if norm1 > self._eps:
                self._orientation = vec / norm1
                self._orientation_mode = True

    def _centers_match(self, a: Node, b: Node) -> bool:
        diff = np.asarray(a.center, dtype=float) - np.asarray(b.center, dtype=float)
        return bool(np.linalg.norm(diff) <= self._eps)

    def _center_distance(self, a: Node, b: Node) -> float:
        if self._query is None:
            return 0.0
        diff = np.asarray(a.center, dtype=float) - np.asarray(b.center, dtype=float)
        denom = float(np.linalg.norm(diff))
        if denom <= self._eps:
            return 0.0
        return abs(float(np.dot(diff, self._query))) / max(denom, self._eps)

    def _center_orientation(self, a: Node, b: Node) -> float:
        if self._orientation is None:
            return 0.0
        diff = np.asarray(a.center, dtype=float) - np.asarray(b.center, dtype=float)
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm <= self._eps:
            return 0.0
        return np.dot(diff, self._orientation)
    
    def priority(
        self,
        a: Node,
        b: Node,
        bounds: BoundsResult,
        *,
        mass: int,
    ) -> Optional[Tuple[float, ...]]:

        if self._centers_match(a, b):
            key0 = -1.0 if bounds.upper <= self._tau and not self._orientation_mode else float(bounds.lower)
                
            return (
                float(key0),
                float(bounds.upper),
                float(self._rng.random()),
            )

        distance = self._center_distance(a, b)
        if self._orientation_mode and self._orientation is not None:   
            return (
                float(distance),
                -float(self._center_orientation(a, b)),
                float(bounds.lower),
                float(bounds.upper),
                float(self._rng.random()),
            )

        else:
            key0 = -1.0 if bounds.upper <= self._tau else float(distance)

            return (
                float(key0),
                float(bounds.lower),
                float(bounds.upper),
                float(self._rng.random()),
            )
            
            
def get_strategy(name: str, **kwargs) -> VisitStrategy[Node]:
    """Factory for visit-ordering strategies.

    Parameters
    ----------
    name:
        Strategy name. Supported: "lower_bound", "diversity".
    kwargs:
        Additional keyword arguments forwarded to the strategy constructor.
    """
    key = str(name).strip().lower()
    if key in {"lb", "lower", "lower_bound"}:
        return LowerBoundVisitStrategy(**kwargs)
    raise ValueError(f"Unknown search strategy: {name}")
