"""Visiting order strategies for the search engine."""

from __future__ import annotations

from typing import Dict, Generic, List, Tuple, TypeVar

import logging
import random
import numpy as np
from scipy.spatial import KDTree

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
        self._history_dirty = False

        # KD-Tree built once for distance queries when diversity is computed.
        self._query_tree: KDTree | None = None
        if self.queries is not None and self.queries.shape[0] > 0:
            self._query_tree = KDTree(self.queries)

    def register_queries(self, queries: np.ndarray) -> None:
        """Record the latest point queries so diversity penalises revisits."""
        arr = np.asarray(queries, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return
        if self.queries is None or self.queries.size == 0:
            self.queries = arr.copy()
        else:
            self.queries = np.vstack([self.queries, arr])
        self._history_dirty = True
        self._diversity_cache.clear()

    def _get_diversity_score(self, node: Node) -> float:
        """
        Retrieves the diversity score for a node, computing it if not cached.
        """
        node_id = id(node)
        if node_id not in self._diversity_cache:
            if self._query_tree is None:
                self._diversity_cache[node_id] = 0.0
            else:
                q_count = int(self.queries.shape[0]) if self.queries is not None else 0
                k = min(2, max(1, q_count))
                distances, _ = self._query_tree.query(node.center, k=k)
                if np.isscalar(distances):
                    dist_arr = np.array([float(distances)])
                else:
                    dist_arr = np.asarray(distances, dtype=float).reshape(-1)
                positive = dist_arr[dist_arr > self._eps]
                if positive.size > 0:
                    choice = float(np.min(positive))
                else:
                    choice = float(np.max(dist_arr))
                self._diversity_cache[node_id] = choice
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
        self._diversity_cache.clear()
        if self.queries is not None and self.queries.shape[0] > 0:
            if self._history_dirty or self._query_tree is None:
                self._query_tree = KDTree(self.queries)
                self._history_dirty = False
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Rebuilt query KD-tree with %d points",
                        int(self.queries.shape[0]),
                    )
        else:
            self._query_tree = None
            
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
        div_a = self._get_diversity_score(a)
        div_b = self._get_diversity_score(b)
        diversity_score = min(div_a, div_b)
        
        if diversity_score == 0.0:
            return (float('inf'))

        if self._centers_match(a, b):
            key0 = -1.0 if bounds.upper <= self._tau else float(bounds.lower)
            return (float(key0), float(bounds.upper), -float(diversity_score), float(self._rng.random()))

        distance = self._center_distance(a, b)
        key0 = -1.0 if bounds.upper <= self._tau else float(distance)

        return (float(key0), float(bounds.lower), float(bounds.upper), -float(diversity_score), float(self._rng.random()))

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
