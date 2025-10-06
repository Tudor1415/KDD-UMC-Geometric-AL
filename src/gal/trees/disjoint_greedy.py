"""Disjoint greedy ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.neighbors import KDTree

from .common import GeometricTree, Node
from utils.geometry import enclose_many_balls
from utils.meb import meb

EPS = 1e-12

try:
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    hnswlib = None  # type: ignore


class _RadiusSearcher:
    def query_radius(self, point: np.ndarray, radius: float, min_results: int) -> np.ndarray:
        raise NotImplementedError


class _KDTreeRadiusSearcher(_RadiusSearcher):
    def __init__(self, points: np.ndarray) -> None:
        self._tree = KDTree(points)

    def query_radius(self, point: np.ndarray, radius: float, min_results: int) -> np.ndarray:  # noqa: ARG002
        return self._tree.query_radius(point.reshape(1, -1), r=radius)[0]


class _HNSWRadiusSearcher(_RadiusSearcher):
    def __init__(
        self,
        points: np.ndarray,
        params: Dict[str, float | int],
    ) -> None:
        if hnswlib is None:
            raise RuntimeError("hnswlib is required for ann_backend='hnsw'")

        num_points, dim = points.shape
        index = hnswlib.Index(space="l2", dim=dim)
        ef_construction = int(params.get("ef_construction", 200))
        m = int(params.get("M", 16))
        index.init_index(max_elements=num_points, ef_construction=ef_construction, M=m)
        index.add_items(points)

        self._index = index
        self._max_neighbors = int(params.get("max_neighbors", min(max(64, num_points), 8192)))
        self._initial_neighbors = int(params.get("initial_neighbors", min(256, self._max_neighbors)))
        ef = int(params.get("ef", min(max(ef_construction, 64), max(ef_construction, self._max_neighbors))))
        self._index.set_ef(ef)

    def query_radius(self, point: np.ndarray, radius: float, min_results: int) -> np.ndarray:
        radius_sq = radius * radius
        k = max(self._initial_neighbors, min_results)
        k = min(k, self._max_neighbors)

        while True:
            labels, distances = self._index.knn_query(point.reshape(1, -1), k=k)
            labels = labels[0]
            distances = distances[0]
            valid_mask = (labels != -1) & (distances <= radius_sq)
            if not np.any(valid_mask):
                matches = np.empty(0, dtype=np.int64)
            else:
                matches = labels[valid_mask].astype(np.int64, copy=False)

            if matches.size >= min_results or k == self._max_neighbors or labels.size < k:
                return matches

            prev_k = k
            k = min(self._max_neighbors, k * 2)
            if k == prev_k:
                return matches


def _make_radius_searcher(
    points: np.ndarray,
    backend: str,
    ann_params: Optional[Dict[str, float | int]] = None,
) -> _RadiusSearcher:
    backend_normalized = backend.lower()
    if backend_normalized == "auto":
        if hnswlib is not None and points.shape[1] >= 25:
            backend_normalized = "hnsw"
        else:
            backend_normalized = "kdtree"
    if backend_normalized == "hnsw":
        if ann_params is None:
            ann_params = {}
        return _HNSWRadiusSearcher(points, ann_params)
    if backend_normalized != "kdtree":
        raise ValueError(f"Unsupported ann_backend '{backend}'")
    return _KDTreeRadiusSearcher(points)


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
    *,
    rng: Optional[np.random.Generator] = None,
    candidate_sample_size: Optional[int] = None,
    ann_backend: str = "auto",
    ann_params: Optional[Dict[str, float | int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Find disjoint children using a greedy strategy with ANN-backed radius queries."""

    if indices.size < 2 or parent_radius <= 0.0 or max_children <= 0:
        return []

    local_points = data[indices]

    radius_searcher = _make_radius_searcher(local_points, backend=ann_backend, ann_params=ann_params)

    dist_to_center = np.linalg.norm(local_points - parent_center[None, :], axis=1)
    radius_cap = np.minimum(parent_radius / radius_divisor, parent_radius - dist_to_center)
    np.maximum(radius_cap, 0.0, out=radius_cap)

    chosen: List[Tuple[int, np.ndarray, float]] = []
    chosen_centers_indices: List[int] = []
    chosen_radii: List[float] = []

    n_local = local_points.shape[0]
    draw_rng = rng if rng is not None else np.random.default_rng()
    all_indices = np.arange(n_local, dtype=np.int64)
    use_subsample = (
        candidate_sample_size is not None and candidate_sample_size > 0 and candidate_sample_size < n_local
    )

    while len(chosen) < max_children:
        best_choice = None
        best_gain = -1
        best_radius = 0.0

        if use_subsample:
            candidate_indices = np.asarray(
                draw_rng.choice(all_indices, size=candidate_sample_size, replace=False), dtype=np.int64
            )
        else:
            candidate_indices = all_indices

        for li in candidate_indices:
            rmax = radius_cap[li]
            if rmax <= 0.0:
                continue

            candidate_point = local_points[li]
            for c_idx, rj in zip(chosen_centers_indices, chosen_radii):
                dist_to_cj = float(np.linalg.norm(candidate_point - local_points[c_idx]))
                rmax = min(rmax, dist_to_cj - rj)
                if rmax <= 0.0:
                    break

            if rmax <= 0.0:
                continue

            cover_indices_local = radius_searcher.query_radius(candidate_point, rmax + eps, min_child_size)

            gain = cover_indices_local.size
            if gain < min_child_size:
                continue

            if gain > best_gain or (gain == best_gain and rmax > best_radius):
                best_choice = (li, cover_indices_local, float(rmax))
                best_gain = gain
                best_radius = float(rmax)

        if best_choice is None:
            break

        li_local, cover_local, rsel = best_choice
        chosen.append((li_local, cover_local, rsel))
        chosen_centers_indices.append(li_local)
        chosen_radii.append(rsel)

    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for li_local, cover_local, rsel in chosen:
        cover_local = np.asarray(cover_local, dtype=np.int64)
        child_indices_global = indices[cover_local]
        child_center = local_points[li_local]
        results.append((child_indices_global, child_center.copy(), float(rsel)))
    return results


def build_tree(X: np.ndarray, config: Dict | None = None) -> GeometricTree:
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
    ann_backend = cfg.get("ann_backend", "auto")
    ann_params = dict(
        M=int(cfg.get("ann_M", 16)),
        ef_construction=int(cfg.get("ann_ef_construction", 200)),
        ef=int(cfg.get("ann_ef", 200)),
        max_neighbors=int(cfg.get("ann_max_neighbors", 2048)),
        initial_neighbors=int(cfg.get("ann_initial_neighbors", 256)),
    )
    candidate_sample_size_raw = cfg.get("candidate_sample_size", None)
    candidate_sample_size = (
        int(candidate_sample_size_raw)
        if candidate_sample_size_raw is not None and int(candidate_sample_size_raw) > 0
        else None
    )
    random_seed = cfg.get("random_seed", None)
    rng: Optional[np.random.Generator]
    if random_seed is None:
        rng = None
    else:
        rng = np.random.default_rng(random_seed)
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

        children_specs = _greedy_children_kdtree(
            data,
            node_indices,
            node.center,
            node.radius,
            max_children=max_children,
            min_child_size=min_child_size,
            radius_divisor=radius_divisor,
            eps=EPS,
            rng=rng,
            candidate_sample_size=candidate_sample_size,
            ann_backend=ann_backend,
            ann_params=ann_params,
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
    return GeometricTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="disjoint_greedy",
        config=cfg,
    )
