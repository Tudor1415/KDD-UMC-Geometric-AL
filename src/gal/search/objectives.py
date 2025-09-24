"""Visiting order objectives for the search engine."""

from __future__ import annotations

from typing import Protocol, Tuple


class VisitingObjective(Protocol):
    """Protocol determining the queue priority for candidate node pairs."""

    def __call__(self, lower: float, upper: float, mass: int) -> Tuple[float, ...]:
        ...


class LowerBoundObjective:
    """Visit candidates by increasing lower bound, breaking ties on the upper bound."""

    def __call__(self, lower: float, upper: float, mass: int) -> Tuple[float, ...]:
        return (lower, upper, float(mass))
