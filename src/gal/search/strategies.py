"""Visiting order strategies for the search engine."""

from __future__ import annotations

from typing import Dict, Generic, List, Tuple, TypeVar

import logging
import random
import numpy as np

from ..trees.common import Node
from .bounds import BoundsResult

TNode = TypeVar('TNode')


logger = logging.getLogger(__name__)


class VisitStrategy(Generic[TNode]):
    """Base class for visit-ordering strategies."""

    def setup(
        self,
        root: TNode,
        *,
        data: np.ndarray | None = None,
        wc: np.ndarray | None = None,
        tau: float = float('inf'),
        eps: float = 1e-12,
    ) -> None:
        """Prepare the strategy for a new search tree."""

    def priority(self, a: TNode, b: TNode, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        """Return a tuple used to order candidate node pairs."""
        raise NotImplementedError


class LowerBoundVisitStrategy(VisitStrategy[Node]):
    """Order candidate pairs using query-aligned centre distances and bounds."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._query: np.ndarray | None = None
        self._tau: float = float('inf')
        self._eps: float = 1e-12
        self._rng: random.Random = rng or random.Random()

    def setup(
        self,
        root: Node,
        *,
        data: np.ndarray | None = None,
        wc: np.ndarray | None = None,
        tau: float = float('inf'),
        eps: float = 1e-12,
    ) -> None:
        """Capture query vector and thresholds for upcoming priority calls."""
        if wc is not None:
            self._query = np.asarray(wc, dtype=float)
        else:
            self._query = None
        self._tau = float(tau)
        self._eps = float(eps)

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

    def priority(self, a: Node, b: Node, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        if self._centers_match(a, b):
            key0 = -1.0 if bounds.upper <= self._tau else float(bounds.lower)
            logger.debug(
                "Matching centres: key=%s lower=%s upper=%s mass=%s",
                key0,
                float(bounds.lower),
                float(bounds.upper),
                mass,
            )
            return (float(key0), float(bounds.lower), float(self._rng.random()))

        distance = self._center_distance(a, b)
        key0 = -1.0 if bounds.upper <= self._tau else float(distance)
        logger.debug(
            "Computed centre distance=%s key=%s lower=%s upper=%s mass=%s",
            distance,
            key0,
            float(bounds.lower),
            float(bounds.upper),
            mass,
        )
        return (float(key0), float(bounds.lower), float(self._rng.random()))


class DiversityVisitStrategy(VisitStrategy[Node]):
    """
    Order candidate pairs by decreasing diversity, then by bound tightness.
    """

    def __init__(self, queries: np.ndarray | None = None, rng: random.Random | None = None) -> None:
        self.queries = None if queries is None else np.asarray(queries, dtype=float)
        self._diversity_cache: Dict[int, float] = {}
        self._query: np.ndarray | None = None
        self._tau: float = float('inf')
        self._eps: float = 1e-12
        self._rng: random.Random = rng or random.Random()

    def _get_diversity_score(self, node: Node) -> float:
        """
        Retrieves the diversity score for a node, computing it if not cached.
        """
        node_id = id(node)
        if node_id not in self._diversity_cache:
            if self.queries is None or len(self.queries) == 0:
                self._diversity_cache[node_id] = 0.0
            else:
                self._diversity_cache[node_id] = float(
                    np.min(np.linalg.norm(self.queries - node.center, axis=1))
                )
        return self._diversity_cache[node_id]

    def setup(
        self,
        root: Node,
        *,
        data: np.ndarray | None = None,
        wc: np.ndarray | None = None,
        tau: float = float('inf'),
        eps: float = 1e-12,
    ) -> None:
        """
        Pre-computes diversity scores for all nodes in the tree.
        """
        if self.queries is None and data is not None:
            self.queries = np.asarray(data, dtype=float)

        if wc is not None:
            self._query = np.asarray(wc, dtype=float)
        else:
            self._query = None
        self._tau = float(tau)
        self._eps = float(eps)

        stack = [root]
        while stack:
            node = stack.pop()
            self._get_diversity_score(node)  # This will compute and cache the score
            stack.extend(node.children)

    def _center_distance(self, a: Node, b: Node) -> float:
        if self._query is None:
            return 0.0
        diff = np.asarray(a.center, dtype=float) - np.asarray(b.center, dtype=float)
        denom = float(np.linalg.norm(diff))
        if denom <= self._eps:
            return 0.0
        return abs(float(np.dot(diff, self._query))) / max(denom, self._eps)

    def priority(self, a: Node, b: Node, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        div_a = self._get_diversity_score(a)
        div_b = self._get_diversity_score(b)
        diversity_score = max(div_a, div_b)
        distance = self._center_distance(a, b)
        key0 = -1 if bounds.upper <= self._tau else distance
        return (float(key0), -float(diversity_score), float(bounds.lower), float(bounds.upper))



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
        return LowerBoundVisitStrategy()
    if key in {"diversity", "div"}:
        return DiversityVisitStrategy(**kwargs)
    raise ValueError(f"Unknown search strategy: {name}")
