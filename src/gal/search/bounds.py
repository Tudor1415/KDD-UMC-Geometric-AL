"""Bound evaluators used by the search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

import numpy as np

from ..trees.common import Node

TNode = TypeVar('TNode')


@dataclass(frozen=True)
class BoundContext:
    """Context provided to bound evaluators.

    Parameters
    ----------
    wc:
        Weighting coefficients defining the optimisation direction.
    eps:
        Numerical stability floor used across bound computations.
    """

    wc: np.ndarray
    eps: float = 1e-12


@dataclass(frozen=True)
class BoundsResult:
    lower: float
    upper: float


class BoundsStrategy(Generic[TNode]):
    """Base class for pairwise bound evaluators on tree nodes."""

    def __call__(self, a: TNode, b: TNode, context: BoundContext) -> BoundsResult:
        raise NotImplementedError


class BallTreeBounds(BoundsStrategy[Node]):
    """Bounds for weighted cosine distance between two ball-tree nodes."""

    def __call__(self, a: Node, b: Node, context: BoundContext) -> BoundsResult:
        lower, upper = self._bounds_ball_pair(a, b, context.wc, context.eps)
        return BoundsResult(lower=lower, upper=upper)

    @staticmethod
    def _bounds_ball_pair(a: Node, b: Node, wc: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
        """Tight lower/upper bounds on d_wc(f1, f2) for all f1 in Ba and f2 in Bb."""
        c1, r1 = a.center, float(a.radius)
        c2, r2 = b.center, float(b.radius)
        rho = r1 + r2
        d = c2 - c1

        gamma = float(np.linalg.norm(wc))
        if gamma <= eps:
            return 0.0, 0.0  # degenerate direction => all distances 0

        delta = float(np.dot(d, wc))

        if abs(delta) <= rho * gamma:
            lower = 0.0
        else:
            num = gamma * abs(delta - rho * gamma)
            denom = float(np.linalg.norm(d * gamma - rho * wc))
            lower = num / max(denom, eps)

        if delta <= 0:
            num = gamma * abs(delta - rho * gamma)
            denom = float(np.linalg.norm(d * gamma - rho * wc))
        else:
            num = gamma * abs(delta + rho * gamma)
            denom = float(np.linalg.norm(d * gamma + rho * wc))
        upper = num / max(denom, eps)

        return lower, upper
