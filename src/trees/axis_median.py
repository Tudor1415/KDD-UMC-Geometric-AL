"""Axis-aligned (kd-style) median ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List

import numpy as np

from .common import BallTree, Node
from ..utils.meb import meb
from ..utils.geometry import enclose_many_balls
from ..utils.partitions import axis_median_split


def build_tree(X: np.ndarray, config: Dict | None = None) -> BallTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.axis_median").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    leaf_size = int(cfg["leaf_size"])
    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    if max_children < 2:
        raise ValueError("max_children must be >= 2")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def split_axis(indices: np.ndarray) -> int:
        spreads = np.ptp(data[indices], axis=0)
        return int(np.argmax(spreads))

    def split_indices(indices: np.ndarray, axis: int) -> List[np.ndarray]:
        if max_children == 2:
            left, right = axis_median_split(data, indices, axis)
            if left.size == 0 or right.size == 0:
                mid = indices.size // 2
                left = indices[:mid]
                right = indices[mid:]
            return [left, right]
        order = indices[np.argsort(data[indices, axis], kind="mergesort")]
        parts: List[np.ndarray] = []
        for child in range(max_children):
            start = (child * order.size) // max_children
            end = ((child + 1) * order.size) // max_children if child < max_children - 1 else order.size
            if end > start:
                parts.append(order[start:end])
        if len(parts) < 2:
            return [indices.copy()]
        return parts

    def build(indices: np.ndarray) -> Node:
        if indices.size <= leaf_size:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        axis = split_axis(indices)
        children = [build(child_idx) for child_idx in split_indices(indices, axis)]
        centers = [(child.center, child.radius) for child in children]
        center, radius = enclose_many_balls(centers)
        return Node(center=center, radius=radius, children=children, indices=None, is_leaf=False)

    root = build(indices_all)
    return BallTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="axis_median",
        config=cfg,
    )