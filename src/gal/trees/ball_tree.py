"""Legacy compatibility entry-point for ball-tree construction.

This module bridges the historical ``gal.trees.ball_tree`` API to the new
geometry-aware builders living under :mod:`trees`.  It exposes a single
:func:`build_tree` function whose signature matches the other constructors and
forwards to the requested strategy.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from trees import axis_median, bottom_up, middle_out, pca_ballstar, two_pivot
from trees.common import BallTree

__all__ = ["AVAILABLE_METHODS", "build_tree"]

BuilderFn = Callable[[np.ndarray, dict | None], BallTree]

AVAILABLE_METHODS: Dict[str, BuilderFn] = {
    "axis_median": axis_median.build_tree,
    "two_pivot": two_pivot.build_tree,
    "pca_ballstar": pca_ballstar.build_tree,
    "bottom_up": bottom_up.build_tree,
    "middle_out": middle_out.build_tree,
}


def build_tree(
    X: np.ndarray,
    config: dict | None = None,
    *,
    method: str = "axis_median",
) -> BallTree:
    """Build a ball-tree using one of the registered strategies.

    Parameters
    ----------
    X : ndarray
        Input dataset with shape ``(n_samples, n_features)`` and dtype
        ``float64``.  Validation is delegated to the selected builder.
    config : dict, optional
        Configuration overrides accepted by the target builder.  The dictionary
        is copied before being forwarded so callers can safely reuse it.
    method : {"axis_median", "two_pivot", "pca_ballstar", "bottom_up", "middle_out"}
        Name of the construction algorithm.  Defaults to ``"axis_median"`` for
        backwards compatibility.

    Returns
    -------
    trees.common.BallTree
        The constructed ball-tree instance.

    Raises
    ------
    ValueError
        If ``method`` is not one of :data:`AVAILABLE_METHODS`.
    """

    try:
        builder = AVAILABLE_METHODS[method]
    except KeyError as exc:  # pragma: no cover - defensive guard
        allowed = ", ".join(sorted(AVAILABLE_METHODS))
        raise ValueError(f"Unknown ball-tree construction method: {method}. "
                         f"Choose one of: {allowed}.") from exc

    cfg = None if config is None else dict(config)
    return builder(X, cfg)
