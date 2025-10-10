"""Dataclasses used across convergence analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class OrientationCDFEntry:
    """Single entry of the estimated versus null orientation CDF."""

    tau: float
    cdf_hat: float
    cdf_null: float


@dataclass
class ConvergenceStats:
    """Aggregate statistics describing a polytope convergence snapshot."""

    rho_hat: float
    var_hat: float
    lambda_hat: float
    john_center: Sequence[float]
    john_vol: float
    cheby_center: Sequence[float]
    cheby_radius: float
    cheby_ball_vol: float
    r_max_from_a: float
    r_min_from_a: float
    sphericity: float
    ks_stat: float
    ks_p_value: float
    orientation_cdf: List[OrientationCDFEntry]
    median_cosine_distance: float
    expected_theta: float


__all__ = [
    "OrientationCDFEntry",
    "ConvergenceStats",
]
