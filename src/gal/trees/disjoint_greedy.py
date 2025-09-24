"""Disjoint greedy ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.neighbors import KDTree
except ImportError:  # pragma: no cover
    KDTree = None

from .common import BallTree, Node
from utils.geometry import enclose_many_balls
from utils.meb import meb

EPS = 1e-12


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    gram = points @ points.T
    sq = np.diag(gram)
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, out=d2)


def _greedy_children_bruteforce(
    data: np.ndarray,
    indices: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    max_children: int,
    min_child_size: int,
    radius_divisor: float,
    eps: float = EPS,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    if indices.size == 0 or parent_radius <= 0.0 or max_children <= 0:
        return []

    local_points = data[indices]
    if indices.size == 1:
        return [(indices.copy(), local_points[0].copy(), 0.0)]

    distances = _pairwise_distances(local_points)
    dist_to_center = np.linalg.norm(local_points - parent_center[None, :], axis=1)
    radius_cap = np.minimum(parent_radius / radius_divisor, parent_radius - dist_to_center)
    np.maximum(radius_cap, 0.0, out=radius_cap)

    chosen: List[Tuple[int, np.ndarray, float]] = []
    chosen_centers: List[int] = []
    chosen_radii: List[float] = []

    while len(chosen) < max_children:
        best_choice = None
        best_gain = -1
        best_radius = 0.0

        for li in range(indices.size):
            rmax = radius_cap[li]
            if rmax <= 0.0:
                continue
            for cj, rj in zip(chosen_centers, chosen_radii):
                rmax = min(rmax, distances[li, cj] - rj)
                if rmax <= 0.0:
                    break
            if rmax <= 0.0:
                continue

            cover = np.where(distances[li] <= rmax + eps)[0]
            gain = cover.size
            if gain < min_child_size:
                continue

            if gain > best_gain or (gain == best_gain and rmax > best_radius):
                best_choice = (li, cover, float(rmax))
                best_gain = gain
                best_radius = float(rmax)

        if best_choice is None:
            break

        li, cover, rsel = best_choice
        chosen.append((li, cover, rsel))
        chosen_centers.append(li)
        chosen_radii.append(rsel)

    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for li, cover, rsel in chosen:
        child_indices = indices[cover]
        child_center = local_points[li]
        results.append((child_indices, child_center.copy(), float(rsel)))
    return results




def _greedy_children_kdtree(
    data: np.ndarray,
    indices: np.ndarray,
    parent_center: np.ndarray,
    parent_radius: float,
    max_children: int,
    min_child_size: int,
    radius_divisor: float,
    eps: float = EPS,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Greedy child selection accelerated with :class:~sklearn.neighbors.KDTree."""

    if indices.size == 0 or parent_radius <= 0.0 or max_children <= 0:
        return []

    if KDTree is None:
        return _greedy_children_bruteforce(
            data,
            indices,
            parent_center,
            parent_radius,
            max_children,
            min_child_size,
            radius_divisor,
            eps=eps,
        )

    local_points = data[indices]
    if indices.size == 1:
        return [(indices.copy(), local_points[0].copy(), 0.0)]

    kdtree = KDTree(local_points)

    dist_to_center = np.linalg.norm(local_points - parent_center[None, :], axis=1)
    radius_cap = np.minimum(parent_radius / radius_divisor, parent_radius - dist_to_center)
    np.maximum(radius_cap, 0.0, out=radius_cap)

    chosen: List[Tuple[int, np.ndarray, float]] = []
    chosen_centers: List[int] = []
    chosen_radii: List[float] = []

    while len(chosen) < max_children:
        best_choice = None
        best_gain = -1
        best_radius = 0.0

        for li in range(local_points.shape[0]):
            rmax = radius_cap[li]
            if rmax <= 0.0:
                continue

            candidate_point = local_points[li]
            for cj, rj in zip(chosen_centers, chosen_radii):
                centre_dist = float(np.linalg.norm(candidate_point - local_points[cj]))
                rmax = min(rmax, centre_dist - rj)
                if rmax <= 0.0:
                    break
            if rmax <= 0.0:
                continue

            cover_local = kdtree.query_radius(candidate_point[None, :], r=rmax + eps)[0]
            gain = int(cover_local.size)
            if gain < min_child_size:
                continue

            if gain > best_gain or (gain == best_gain and rmax > best_radius):
                rsel = float(max(rmax, 0.0))
                best_choice = (li, cover_local, rsel)
                best_gain = gain
                best_radius = rsel

        if best_choice is None:
            break

        li_sel, cover_sel, rsel = best_choice
        chosen.append((li_sel, cover_sel, rsel))
        chosen_centers.append(li_sel)
        chosen_radii.append(rsel)

    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for li_sel, cover_sel, rsel in chosen:
        cover_sel = np.asarray(cover_sel, dtype=np.int64)
        child_indices = indices[cover_sel]
        child_center = local_points[li_sel]
        results.append((child_indices, child_center.copy(), float(rsel)))
    return results



def build_tree(X: np.ndarray, config: Dict | None = None) -> BallTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.disjoint_greedy").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    leaf_size = int(cfg["leaf_size"])
    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    min_child_size = int(cfg["min_child_size"])
    radius_divisor = float(cfg["radius_divisor"])
    if leaf_size <= 0:
        raise ValueError("leaf_size must be positive")
    if max_children < 2:
        raise ValueError("max_children must be >= 2")
    if min_child_size <= 0:
        raise ValueError("min_child_size must be positive")
    if radius_divisor <= 1.0:
        raise ValueError("radius_divisor must be > 1")

    n_samples, n_features = data.shape
    all_indices = np.arange(n_samples, dtype=np.int64)

    center_root, radius_root = meb(data, method=meb_method)
    root = Node(center=center_root, radius=radius_root, indices=None, children=[], is_leaf=False)

    queue: List[Tuple[Node, np.ndarray]] = [(root, all_indices)]

    while queue:
        node, node_indices = queue.pop()
        if node_indices.size <= leaf_size:
            center_leaf, radius_leaf = meb(data[node_indices], method=meb_method)
            node.center = center_leaf
            node.radius = radius_leaf
            node.indices = node_indices.copy()
            node.children = []
            node.is_leaf = True
            continue

        children_specs = _greedy_children(
            data,
            node_indices,
            node.center,
            node.radius,
            max_children=max_children,
            min_child_size=min_child_size,
            radius_divisor=radius_divisor,
            eps=EPS,
        )

        if len(children_specs) < 2:
            center_leaf, radius_leaf = meb(data[node_indices], method=meb_method)
            node.center = center_leaf
            node.radius = radius_leaf
            node.indices = node_indices.copy()
            node.children = []
            node.is_leaf = True
            continue

        child_nodes: List[Node] = []
        child_balls: List[Tuple[np.ndarray, float]] = []
        for child_idx, child_center, child_radius in children_specs:
            if child_idx.size == 0:
                continue
            is_leaf = child_idx.size <= leaf_size
            indices_attr = child_idx.copy() if is_leaf else None
            if is_leaf:
                c_center, c_radius = meb(data[child_idx], method=meb_method)
            else:
                c_center = child_center
                c_radius = child_radius
            child = Node(center=c_center, radius=float(c_radius), indices=indices_attr, children=[], is_leaf=is_leaf)
            child_nodes.append(child)
            child_balls.append((child.center, child.radius))
            if not is_leaf:
                queue.append((child, child_idx))
        if len(child_nodes) < 2:
            center_leaf, radius_leaf = meb(data[node_indices], method=meb_method)
            node.center = center_leaf
            node.radius = radius_leaf
            node.indices = node_indices.copy()
            node.children = []
            node.is_leaf = True
            continue

        node.children = child_nodes
        node.is_leaf = False
        node.indices = None
        node.center, node.radius = enclose_many_balls([(ch.center, ch.radius) for ch in child_nodes])


    fallback_method = cfg.get("degeneracy_fallback", "axis_median")

    if root.is_leaf and root.indices is not None and root.indices.size > leaf_size:
        if fallback_method == "axis_median":
            from . import axis_median

            fallback_tree = axis_median.build_tree(data, {"leaf_size": leaf_size})
            fallback_tree.method = "disjoint_greedy"
            fallback_tree.config = cfg
            return fallback_tree
        raise RuntimeError(
            "disjoint_greedy failed to split data; consider adjusting configuration"
        )
    return BallTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="disjoint_greedy",
        config=cfg,
    )




