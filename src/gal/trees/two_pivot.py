"""Two-pivot (farthest-point) ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List

import numpy as np

from .common import GeometricTree, Node
from utils.geometry import centroid, enclose_many_balls
from utils.meb import meb
from utils.partitions import axis_median_split

EPSILON = 1e-12


def build_tree(X: np.ndarray, config: Dict | None = None) -> GeometricTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.two_pivot").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    leaf_size = int(cfg["leaf_size"])
    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    fallback = cfg["degeneracy_fallback"]
    if max_children < 2:
        raise ValueError("max_children must be >= 2")
    if fallback not in {"direction_median", "axis_median"}:
        raise ValueError("Unsupported degeneracy_fallback option")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def choose_pivots(indices: np.ndarray) -> tuple[int, int]:
        pts = data[indices]
        center = centroid(pts)
        diff = pts - center
        dist2 = np.einsum("ij,ij->i", diff, diff)
        left_local = int(np.argmax(dist2))
        left = int(indices[left_local])
        diff = data[indices] - data[left]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        right_local = int(np.argmax(dist2))
        right = int(indices[right_local])
        return left, right

    def axis_split(indices: np.ndarray) -> List[np.ndarray]:
        spreads = np.ptp(data[indices], axis=0)
        axis = int(np.argmax(spreads))
        left_part, right_part = axis_median_split(data, indices, axis)
        if left_part.size == 0 or right_part.size == 0:
            mid = max(1, indices.size // 2)
            left_part = indices[:mid]
            right_part = indices[mid:]
        return [left_part, right_part]

    def fallback_split(indices: np.ndarray, left: int, right: int) -> List[np.ndarray]:
        if fallback == "direction_median":
            direction = data[right] - data[left]
            norm = np.linalg.norm(direction)
            if norm <= 0.0:
                return axis_split(indices)
            direction /= norm
            projections = data[indices] @ direction
            order = indices[np.argsort(projections, kind="mergesort")]
        else:
            return axis_split(indices)
        mid = max(1, order.size // 2)
        left_part = order[:mid]
        right_part = order[mid:]
        if right_part.size == 0:
            right_part = order[-1:]
            left_part = order[:-1]
        return [left_part, right_part]

    def voronoi_split(indices: np.ndarray, seeds: List[int]) -> List[np.ndarray]:
        pts = data[indices]
        centers = data[np.array(seeds, dtype=np.int64)]
        dist2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = np.argmin(dist2, axis=1)
        groups = [indices[assignment == k] for k in range(len(seeds))]
        empty = [i for i, g in enumerate(groups) if g.size == 0]
        if not empty:
            return groups
        pool = sorted(range(len(groups)), key=lambda idx: groups[idx].size, reverse=True)
        for cluster in empty:
            donor = next((idx for idx in pool if groups[idx].size > 1), None)
            if donor is None:
                return axis_split(indices)
            donor_points = groups[donor]
            moved = donor_points[-1]
            groups[donor] = donor_points[:-1].copy()
            groups[cluster] = np.array([moved], dtype=indices.dtype)
        return groups

    def choose_seed_set(indices: np.ndarray, max_k: int) -> List[int]:
        left, right = choose_pivots(indices)
        seeds: List[int] = [left]
        if right != left:
            seeds.append(right)
        target = min(max_k, indices.size)
        pts = data[indices]
        while len(seeds) < target:
            dist2 = np.full(indices.size, np.inf, dtype=np.float64)
            for s in seeds:
                diff = pts - data[s]
                dist2 = np.minimum(dist2, np.einsum("ij,ij->i", diff, diff))
            candidate_local = int(np.argmax(dist2))
            if not np.isfinite(dist2[candidate_local]) or dist2[candidate_local] <= EPSILON:
                break
            candidate = int(indices[candidate_local])
            if candidate not in seeds:
                seeds.append(candidate)
            else:
                break
        return seeds

    def build(indices: np.ndarray) -> Node:
        if indices.size <= leaf_size:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        left, right = choose_pivots(indices)
        if max_children == 2:
            seeds = [left, right] if right != left else [left]
        else:
            seeds = choose_seed_set(indices, max_children)
        if len(seeds) == 1:
            parts = axis_split(indices)
        else:
            parts = voronoi_split(indices, seeds)
            if len(parts) < 2 or any(part.size == 0 for part in parts):
                parts = fallback_split(indices, left, right)
        children = [build(part) for part in parts]
        center, radius = enclose_many_balls([(child.center, child.radius) for child in children])
        return Node(center=center, radius=radius, children=children, is_leaf=False)

    root = build(indices_all)
    return GeometricTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="two_pivot",
        config=cfg,
    )
