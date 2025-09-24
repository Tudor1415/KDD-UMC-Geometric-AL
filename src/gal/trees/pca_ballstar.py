"""PCA / Ball* ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List

import numpy as np

from .common import GeometricTree, Node
from utils.geometry import enclose_many_balls
from utils.meb import meb
from utils.partitions import direction_quantile_splits

SMALL_NORM = 1e-15


def build_tree(X: np.ndarray, config: Dict | None = None) -> GeometricTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.pca_ballstar").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    leaf_size = int(cfg["leaf_size"])
    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    balance = cfg["balance"]
    balance_steps = int(cfg["balance_max_refine_steps"])
    pca_method = cfg["pca_method"]
    pca_iters = int(cfg["pca_iters"])
    rng = np.random.default_rng(cfg["random_state"])
    if max_children < 2:
        raise ValueError("max_children must be >= 2")
    if balance not in {"median", "argmin_sum_radii"}:
        raise ValueError("Unsupported balance option")
    if pca_method not in {"power", "svd"}:
        raise ValueError("Unsupported pca_method option")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def fallback_direction(indices: np.ndarray) -> np.ndarray:
        spreads = np.ptp(data[indices], axis=0)
        axis = int(np.argmax(spreads)) if spreads.size > 0 else 0
        direction = np.zeros(n_features, dtype=np.float64)
        direction[axis] = 1.0
        return direction

    def first_component(indices: np.ndarray) -> np.ndarray:
        pts = data[indices]
        centered = pts - pts.mean(axis=0)
        if pca_method == "svd":
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return fallback_direction(indices)
            if vt.size == 0:
                return fallback_direction(indices)
            direction = vt[0]
        else:
            direction = rng.normal(size=n_features)
            norm = np.linalg.norm(direction)
            if norm <= SMALL_NORM:
                direction = fallback_direction(indices)
                norm = np.linalg.norm(direction)
            direction /= norm
            cov_mul = centered.T @ centered
            for _ in range(pca_iters):
                direction = cov_mul @ direction
                norm = np.linalg.norm(direction)
                if norm <= SMALL_NORM:
                    break
                direction /= norm
            if np.linalg.norm(direction) <= SMALL_NORM:
                direction = fallback_direction(indices)
        norm = np.linalg.norm(direction)
        if norm <= SMALL_NORM:
            direction = fallback_direction(indices)
            norm = np.linalg.norm(direction)
        return direction / norm

    def split_two(indices: np.ndarray, direction: np.ndarray) -> List[np.ndarray]:
        projections = data[indices] @ direction
        order = indices[np.argsort(projections, kind="mergesort")]
        if order.size < 2:
            return [order]
        mid_base = max(1, order.size // 2)
        candidates = {mid_base}
        if balance == "argmin_sum_radii":
            for offset in range(1, balance_steps + 1):
                candidates.add(min(order.size - 1, max(1, mid_base + offset)))
                candidates.add(min(order.size - 1, max(1, mid_base - offset)))
        best_split = None
        best_cost = np.inf
        for mid in sorted(candidates):
            left = order[:mid]
            right = order[mid:]
            if left.size == 0 or right.size == 0:
                continue
            if balance == "argmin_sum_radii":
                r_left = meb(data[left], method="ritter")[1]
                r_right = meb(data[right], method="ritter")[1]
                cost = r_left + r_right
            else:
                cost = abs(mid - order.size / 2.0)
            if cost < best_cost:
                best_cost = cost
                best_split = (left, right)
        if best_split is None:
            mid = mid_base
            best_split = (order[:mid], order[mid:])
        return [best_split[0], best_split[1]]

    def build(indices: np.ndarray) -> Node:
        if indices.size <= leaf_size:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        direction = first_component(indices)
        if max_children == 2:
            parts = split_two(indices, direction)
        else:
            parts = direction_quantile_splits(data, indices, direction, max_children)
            if len(parts) < 2:
                parts = split_two(indices, direction)
        children = [build(part) for part in parts if part.size > 0]
        if len(children) == 0:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        center, radius = enclose_many_balls([(child.center, child.radius) for child in children])
        return Node(center=center, radius=radius, children=children, is_leaf=False)

    root = build(indices_all)
    return GeometricTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="pca_ballstar",
        config=cfg,
    )
