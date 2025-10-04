from __future__ import annotations

import numpy as np

from gal.centers.poly_centers import (
    analytical_center,
    chebyshev_center,
    minkowski_center,
    volumetric_center,
)


def _center_fn(name: str):
    """Return a callable (A, b) -> center for any supported polyhedral center.

    Supported names (case-insensitive, with synonyms):
      - analytic, analytical, analytic_center, analytical_center
      - chebyshev, chebyshev_center, inscribed, largest_ball
      - minkowski, minkowski_center
      - volumetric, volumetric_center, john, john_ellipsoid
    """
    key = (name or "analytic").strip().lower().replace("-", "_")
    if key in {"analytic", "analytical", "analytic_center", "analytical_center", "analyticcenter", "analyticalcenter", "barrier"}:
        return lambda A, b: analytical_center(A, b)
    if key in {"chebyshev", "chebyshev_center", "chebyshevcenter", "inscribed", "largest_ball"}:
        return lambda A, b: chebyshev_center(A, b)[0]
    if key in {"minkowski", "minkowski_center", "minkowskicenter"}:
        return lambda A, b: minkowski_center(A, b)[0]
    if key in {"volumetric", "volumetric_center", "volumetriccenter", "john", "john_ellipsoid"}:
        return lambda A, b: volumetric_center(A, b)[0]
    raise ValueError(f"Unknown center method: {name}")


def _chebyshev_radius(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    slack = b - A @ c
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    return max(0.0, float(np.min(vals, initial=np.inf)))

