"""Compatibility helpers tying legacy GAL API to the refactored trees package."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from trees import axis_median, bottom_up, disjoint_greedy, middle_out, pca_ballstar, two_pivot
from trees.common import BallTree, Node
from trees.search import search_pair as _search_pair

__all__ = ["AVAILABLE_METHODS", "build_tree", "build_ball_tree", "search_pair"]

BuilderFn = Callable[[np.ndarray, dict | None], BallTree]

AVAILABLE_METHODS: Dict[str, BuilderFn] = {
    "axis_median": axis_median.build_tree,
    "two_pivot": two_pivot.build_tree,
    "pca_ballstar": pca_ballstar.build_tree,
    "bottom_up": bottom_up.build_tree,
    "middle_out": middle_out.build_tree,
    "disjoint_greedy": disjoint_greedy.build_tree,
}


def build_tree(
    X: np.ndarray,
    config: dict | None = None,
    *,
    method: str = "axis_median",
) -> BallTree:
    """Build a ball-tree using one of the registered strategies."""

    try:
        builder = AVAILABLE_METHODS[method]
    except KeyError as exc:  # pragma: no cover
        allowed = ", ".join(sorted(AVAILABLE_METHODS))
        raise ValueError(
            f"Unknown ball-tree construction method: {method}. Choose one of: {allowed}."
        ) from exc

    cfg = None if config is None else dict(config)
    return builder(X, cfg)


def build_ball_tree(
    X: np.ndarray,
    *,
    k: int | None = None,
    P: int | None = None,
    radius_divisor: float | None = None,
    config: dict | None = None,
) -> BallTree:
    """Backward-compatible constructor for the proposed disjoint-greedy tree."""

    cfg = {} if config is None else dict(config)
    if k is not None and "max_children" not in cfg:
        cfg["max_children"] = int(k)
    if P is not None:
        cfg.setdefault("leaf_size", int(P))
        cfg.setdefault("min_child_size", int(P))
    if radius_divisor is not None and "radius_divisor" not in cfg:
        cfg["radius_divisor"] = float(radius_divisor)
    return build_tree(X, cfg or None, method="disjoint_greedy")


def search_pair(tree_or_node: BallTree | Node, X: np.ndarray, wc: np.ndarray, tau: float, **kwargs):
    """Wrapper around :func:`trees.search.search_pair`."""

    return _search_pair(tree_or_node, X, wc, tau, **kwargs)
