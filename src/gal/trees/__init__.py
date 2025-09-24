"""Ball-tree builder implementations."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from . import axis_median, bottom_up, disjoint_greedy, middle_out, pca_ballstar, two_pivot
from .common import GeometricTree, Node

BuilderFn = Callable[[np.ndarray, dict | None], GeometricTree]

AVAILABLE_BUILDERS: Dict[str, BuilderFn] = {
    "axis_median": axis_median.build_tree,
    "two_pivot": two_pivot.build_tree,
    "pca_ballstar": pca_ballstar.build_tree,
    "bottom_up": bottom_up.build_tree,
    "middle_out": middle_out.build_tree,
    "disjoint_greedy": disjoint_greedy.build_tree,
}

AVAILABLE_METHODS = tuple(sorted(AVAILABLE_BUILDERS))


def get_builder(method: str) -> BuilderFn:
    try:
        return AVAILABLE_BUILDERS[method]
    except KeyError as exc:  # pragma: no cover
        allowed = ", ".join(sorted(AVAILABLE_BUILDERS))
        raise ValueError(
            f"Unknown ball-tree construction method: {method}. Choose one of: {allowed}."
        ) from exc


def build_tree(
    X: np.ndarray,
    config: dict | None = None,
    *,
    method: str = "axis_median",
) -> GeometricTree:
    builder = get_builder(method)
    cfg = None if config is None else dict(config)
    return builder(X, cfg)


def build_ball_tree(
    X: np.ndarray,
    *,
    k: int | None = None,
    P: int | None = None,
    radius_divisor: float | None = None,
    config: dict | None = None,
) -> GeometricTree:
    cfg = {} if config is None else dict(config)
    if k is not None and "max_children" not in cfg:
        cfg["max_children"] = int(k)
    if P is not None:
        cfg.setdefault("leaf_size", int(P))
        cfg.setdefault("min_child_size", int(P))
    if radius_divisor is not None and "radius_divisor" not in cfg:
        cfg["radius_divisor"] = float(radius_divisor)
    return build_tree(X, cfg or None, method="disjoint_greedy")


__all__ = [
    "AVAILABLE_BUILDERS",
    "AVAILABLE_METHODS",
    "GeometricTree",
    "BuilderFn",
    "Node",
    "axis_median",
    "bottom_up",
    "build_ball_tree",
    "build_tree",
    "disjoint_greedy",
    "middle_out",
    "pca_ballstar",
    "two_pivot",
]

