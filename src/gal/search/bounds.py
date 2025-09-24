"""Bound evaluators used by the search engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np

from ..trees.common import Node


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


class Bounds(Protocol):
    """Protocol implemented by all bounders."""

    def __call__(self, a: Node, b: Node, context: BoundContext) -> BoundsResult:
        ...


class AngularBounds:
    """Bounds derived from the angular disparity objective."""

    def __call__(self, a: Node, b: Node, context: BoundContext) -> BoundsResult:
        wc = context.wc
        eps = context.eps

        c1, r1 = a.center, float(a.radius)
        c2, r2 = b.center, float(b.radius)
        rho = r1 + r2
        diff = c2 - c1

        gamma = float(np.linalg.norm(wc))
        if gamma <= eps:
            return BoundsResult(lower=0.0, upper=0.0)

        delta = float(np.dot(diff, wc))

        if abs(delta) <= rho * gamma:
            lb = 0.0
        else:
            num = gamma * abs(delta - rho * gamma)
            denom = float(np.linalg.norm(diff * gamma - rho * wc))
            lb = num / max(denom, eps)

        if delta <= 0:
            num = gamma * abs(delta - rho * gamma)
            denom = float(np.linalg.norm(diff * gamma - rho * wc))
        else:
            num = gamma * abs(delta + rho * gamma)
            denom = float(np.linalg.norm(diff * gamma + rho * wc))
        ub = num / max(denom, eps)
        return BoundsResult(lower=lb, upper=ub)
