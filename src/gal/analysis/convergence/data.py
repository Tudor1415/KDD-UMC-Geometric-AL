"""Dataclasses used across convergence analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class OrientationCDFEntry:
    """Single entry of the estimated versus null orientation CDF."""

    tau: float
    cdf_hat: float
    cdf_null: float


@dataclass
class ConvergenceStats:
    """Aggregate statistics describing a polytope convergence snapshot."""

    sphericity: float
    median_cosine_distance: float
    expected_theta: float
    orientation_score: float
    orientation_cdf: List[OrientationCDFEntry]


__all__ = [
    "OrientationCDFEntry",
    "ConvergenceStats",
]
