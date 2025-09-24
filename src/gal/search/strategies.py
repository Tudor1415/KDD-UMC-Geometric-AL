"""Visiting order strategies for the search engine."""

from __future__ import annotations

from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np

from ..trees.common import Node
from .bounds import BoundsResult

TNode = TypeVar('TNode')


class VisitStrategy(Generic[TNode]):
    """Base class for visit-ordering strategies."""

    def setup(self, root: TNode, *, data: np.ndarray | None = None) -> None:
        """Prepare the strategy for a new search tree."""

    def priority(self, a: TNode, b: TNode, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        """Return a tuple used to order candidate node pairs."""
        raise NotImplementedError


class LowerBoundVisitStrategy(VisitStrategy[Node]):
    """Order candidate pairs by their bound tightness."""

    def priority(self, a: Node, b: Node, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        return (bounds.lower, bounds.upper, float(mass))


class DiversityVisitStrategy(VisitStrategy[Node]):
    """
    Order candidate pairs by decreasing diversity, then by bound tightness.
    """

    def __init__(self, queries: np.ndarray | None = None) -> None:
        self.queries = None if queries is None else np.asarray(queries, dtype=float)
        self._diversity_cache: Dict[int, float] = {}

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

    def setup(self, root: Node, *, data: np.ndarray | None = None) -> None:
        """
        Pre-computes diversity scores for all nodes in the tree.
        """
        if self.queries is None and data is not None:
            self.queries = np.asarray(data, dtype=float)

        stack = [root]
        while stack:
            node = stack.pop()
            self._get_diversity_score(node)  # This will compute and cache the score
            stack.extend(node.children)

    def priority(self, a: Node, b: Node, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        div_a = self._get_diversity_score(a)
        div_b = self._get_diversity_score(b)
        diversity_score = max(div_a, div_b)
        return (-diversity_score, bounds.lower, bounds.upper, float(mass))