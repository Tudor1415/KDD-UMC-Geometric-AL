"""Visiting order strategies for the search engine."""

from __future__ import annotations

from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np

from ..trees.common import Node
from .bounds import BoundsResult

TNode = TypeVar('TNode')


def _collect_nodes(root: Node) -> List[Node]:
    """Return a flat list of all nodes in the subtree rooted at `root`."""
    stack = [root]
    nodes: List[Node] = []
    while stack:
        nd = stack.pop()
        nodes.append(nd)
        stack.extend(nd.children)
    return nodes


def _precompute_diversity(root: Node, Q: np.ndarray | None) -> Dict[int, float]:
    """Novelty score per node: min distance to any reference query point."""
    nodes = _collect_nodes(root)
    diversity: Dict[int, float] = {}
    if Q is None or len(Q) == 0:
        for nd in nodes:
            diversity[id(nd)] = 0.0
        return diversity

    Q = np.asarray(Q, dtype=float)
    for nd in nodes:
        diversity[id(nd)] = float(np.min(np.linalg.norm(Q - nd.center, axis=1)))
    return diversity


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
    """Order candidate pairs by decreasing diversity, then by bound tightness."""

    def __init__(self, queries: np.ndarray | None = None) -> None:
        self.queries = None if queries is None else np.asarray(queries, dtype=float)
        self._diversity: Dict[int, float] = {}

    def setup(self, root: Node, *, data: np.ndarray | None = None) -> None:
        queries = self.queries
        if queries is None and data is not None:
            queries = np.asarray(data, dtype=float)
        self._diversity = _precompute_diversity(root, queries)

    def priority(self, a: Node, b: Node, bounds: BoundsResult, mass: int) -> Tuple[float, ...]:
        div_a = self._diversity.get(id(a), 0.0)
        div_b = self._diversity.get(id(b), 0.0)
        diversity_score = max(div_a, div_b)
        return (-diversity_score, bounds.lower, bounds.upper, float(mass))
