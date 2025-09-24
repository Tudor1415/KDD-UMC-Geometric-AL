"""KD-tree builder with AABB summaries.

This module constructs a kd-tree where each node stores an axis-aligned
bounding box (AABB) summary alongside a ball enclosure (center+radius)
for compatibility with the generic Search engine.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .common import GeometricTree, Node


class KdNode(Node):
    """KD-tree node with AABB summaries.

    Additional attributes:
    - bbox_min: np.ndarray (shape: [d])
    - bbox_max: np.ndarray (shape: [d])
    - count: int (number of descendant points)
    - ir2_min: float (min squared L2 norm inside AABB)
    - ir2_max: float (max squared L2 norm inside AABB)
    """

    __slots__ = Node.__slots__ + ("bbox_min", "bbox_max", "count", "ir2_min", "ir2_max")

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        *,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        count: int,
        children: List[Node] | None = None,
        indices: np.ndarray | None = None,
        is_leaf: bool = False,
    ) -> None:
        super().__init__(center=center, radius=radius, children=children, indices=indices, is_leaf=is_leaf)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.count = int(count)
        # Precompute Ir2 interval from bbox to avoid per-query cost
        self.ir2_min, self.ir2_max = _bbox_ir2_interval(bbox_min, bbox_max)


def _bbox_enclosing_ball(bmin: np.ndarray, bmax: np.ndarray) -> Tuple[np.ndarray, float]:
    ctr = 0.5 * (bmin + bmax)
    rad = 0.5 * float(np.linalg.norm(bmax - bmin))
    return ctr, rad


def _bbox_ir2_interval(bmin: np.ndarray, bmax: np.ndarray) -> Tuple[float, float]:
    # For each dim: if interval crosses 0, min contribution is 0; otherwise
    # min is the smaller absolute squared endpoint. Max is larger endpoint squared.
    l = bmin
    u = bmax
    # min squared contribution per dim
    crosses = (l <= 0) & (u >= 0)
    min_sq = np.where(crosses, 0.0, np.minimum(l * l, u * u))
    max_sq = np.maximum(l * l, u * u)
    return float(np.sum(min_sq)), float(np.sum(max_sq))


def build_tree(X: np.ndarray, config: Dict | None = None) -> GeometricTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    # Defaults for kd-tree construction
    cfg = {
        "leaf_size": 32,
        "split_rule": "widest_axis_median",
    }
    if config is not None:
        cfg.update(config)

    leaf_size = int(cfg.get("leaf_size", 32))
    split_rule = str(cfg.get("split_rule", "widest_axis_median"))
    if split_rule not in {"widest_axis_median"}:
        raise ValueError("Unsupported split_rule for kd_tree: {split_rule}")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def bbox_from_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = data[indices]
        return pts.min(axis=0), pts.max(axis=0)

    def widest_axis(bmin: np.ndarray, bmax: np.ndarray) -> int:
        spreads = bmax - bmin
        return int(np.argmax(spreads))

    def split_indices(indices: np.ndarray, axis: int) -> List[np.ndarray]:
        # Stable median split on the chosen axis
        order = indices[np.argsort(data[indices, axis], kind="mergesort")]
        mid = order.size // 2
        left = order[:mid]
        right = order[mid:]
        if left.size == 0 or right.size == 0:
            # Fallback split
            if order.size <= 1:
                return [order]
            left = order[::2]
            right = order[1::2]
        return [left, right]

    def build(indices: np.ndarray) -> KdNode:
        bmin, bmax = bbox_from_indices(indices)
        center, radius = _bbox_enclosing_ball(bmin, bmax)
        if indices.size <= leaf_size:
            return KdNode(
                center=center,
                radius=radius,
                bbox_min=bmin,
                bbox_max=bmax,
                count=int(indices.size),
                children=None,
                indices=indices.copy(),
                is_leaf=True,
            )
        axis = widest_axis(bmin, bmax)
        left_idx, right_idx = split_indices(indices, axis)
        children: List[KdNode] = [build(left_idx), build(right_idx)]
        # Derive bbox from children to avoid recomputation
        cbmin = np.minimum(children[0].bbox_min, children[1].bbox_min)
        cbmax = np.maximum(children[0].bbox_max, children[1].bbox_max)
        ccenter, cradius = _bbox_enclosing_ball(cbmin, cbmax)
        return KdNode(
            center=ccenter,
            radius=cradius,
            bbox_min=cbmin,
            bbox_max=cbmax,
            count=int(children[0].count + children[1].count),
            children=children,
            indices=None,
            is_leaf=False,
        )

    root = build(indices_all)
    return GeometricTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="kd_tree",
        config=cfg,
    )


__all__ = ["KdNode", "build_tree"]

