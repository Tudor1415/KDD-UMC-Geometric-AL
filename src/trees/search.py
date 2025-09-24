"""Branch-and-bound search routines for generic ball trees."""

from __future__ import annotations

import heapq
from itertools import count
from typing import Dict, Iterable, Tuple

import numpy as np

from .common import BallTree, Node

EPS = 1e-12


def _node_is_leaf(node: Node) -> bool:
    return bool(node.is_leaf or not node.children)


def _descendant_size(node: Node, cache: Dict[int, int]) -> int:
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]
    if node.indices is not None and _node_is_leaf(node):
        size = int(node.indices.size)
    else:
        size = sum(_descendant_size(child, cache) for child in node.children)
    cache[node_id] = size
    return size


def _gather_leaf_indices(node: Node) -> np.ndarray:
    stack = [node]
    leaves: list[np.ndarray] = []
    while stack:
        nd = stack.pop()
        if _node_is_leaf(nd) and nd.indices is not None:
            leaves.append(nd.indices)
        else:
            stack.extend(nd.children)
    if not leaves:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(leaves).astype(np.int64, copy=False))


def _dominates(a: Node, b: Node, eps: float = EPS) -> bool:
    amin = a.center - a.radius
    amax = a.center + a.radius
    bmin = b.center - b.radius
    bmax = b.center + b.radius
    return bool(np.all(amin >= bmax - eps) or np.all(bmin >= amax - eps))


def _bounds_ball_pair(a: Node, b: Node, wc: np.ndarray, eps: float = EPS) -> Tuple[float, float]:
    c1, r1 = a.center, float(a.radius)
    c2, r2 = b.center, float(b.radius)
    rho = r1 + r2
    diff = c2 - c1

    gamma = float(np.linalg.norm(wc))
    if gamma <= eps:
        return 0.0, 0.0

    delta = float(np.dot(diff, wc))

    if abs(delta) <= rho * gamma:
        lb = 0.0
    else:
        num = gamma * abs(delta - rho * gamma)
        denom = float(np.linalg.norm(diff * gamma - rho * wc))
        lb = num / max(denom, eps)

    if delta <= 0:
        num = gamma * abs(delta - rho * gamma)
        denom = float(np.linalg.norm(diff * gamma - rho * wc))
    else:
        num = gamma * abs(delta + rho * gamma)
        denom = float(np.linalg.norm(diff * gamma + rho * wc))
    ub = num / max(denom, eps)
    return lb, ub


def _objective_value(p: np.ndarray, q: np.ndarray, wc: np.ndarray, eps: float = EPS) -> float:
    diff = p - q
    denom = float(np.linalg.norm(diff))
    if denom <= eps:
        return 0.0
    return abs(float(np.dot(diff, wc))) / denom


def _exact_leaf_eval(a: Node, b: Node, X: np.ndarray, wc: np.ndarray, eps: float = EPS) -> Tuple[Tuple[int, int] | None, float, int]:
    Ai = a.indices
    Bi = b.indices
    if Ai is None or Bi is None or Ai.size == 0 or Bi.size == 0:
        return None, float("inf"), 0
    XA = X[Ai]
    XB = X[Bi]
    diff = XA[:, None, :] - XB[None, :, :]
    num = np.abs(np.tensordot(diff, wc, axes=(2, 0)))
    denom = np.linalg.norm(diff, axis=2)
    denom = np.maximum(denom, eps)
    dist = num / denom
    m_idx, n_idx = np.unravel_index(np.argmin(dist), dist.shape)
    evals = int(Ai.size) * int(Bi.size)
    return (int(Ai[m_idx]), int(Bi[n_idx])), float(dist[m_idx, n_idx]), evals


def _exact_leaf_self(node: Node, X: np.ndarray, wc: np.ndarray, eps: float = EPS) -> Tuple[Tuple[int, int] | None, float, int]:
    idx = node.indices
    if idx is None or idx.size < 2:
        return None, float("inf"), 0
    best_pair = None
    best_dist = float("inf")
    evals = 0
    XA = X[idx]
    for i in range(idx.size - 1):
        pi = XA[i]
        for j in range(i + 1, idx.size):
            pj = XA[j]
            evals += 1
            dist = _objective_value(pi, pj, wc, eps=eps)
            if dist < best_dist:
                best_dist = dist
                best_pair = (int(idx[i]), int(idx[j]))
    return best_pair, best_dist, evals


def search_pair(
    tree: BallTree | Node,
    X: np.ndarray,
    wc: np.ndarray,
    tau: float,
    *,
    dominance_prune: bool = True,
    return_stats: bool = False,
    eps: float = EPS,
):
    """Dual-tree branch-and-bound search on any :class:BallTree."""

    data = np.ascontiguousarray(X, dtype=np.float64)
    wc = np.asarray(wc, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if wc.ndim != 1:
        raise ValueError("wc must be a 1D vector")
    if wc.size != data.shape[1]:
        raise ValueError("wc must have length equal to X.shape[1]")

    root = tree.root if isinstance(tree, BallTree) else tree

    leaf_indices = _gather_leaf_indices(root)
    total_pairs = int(len(leaf_indices) * (len(leaf_indices) - 1) // 2)

    stats = dict(
        total_point_pairs=total_pairs,
        pruned_lb_point_pairs=0,
        pruned_dom_point_pairs=0,
        pruned_point_pairs=0,
        explored_point_pairs=0,
        objective_evals=0,
        best_origin=None,
        best_distance=None,
        best_pair=None,
    )

    if len(leaf_indices) < 2:
        result = (None, None, float("inf"))
        stats["unexplored_point_pairs"] = 0
        return (*result, stats) if return_stats else result

    size_cache: Dict[int, int] = {}

    def mass(a: Node, b: Node) -> int:
        return _descendant_size(a, size_cache) * _descendant_size(b, size_cache)

    best_pair: Tuple[int, int] | None = None
    best_distance = float("inf")
    heap: list[tuple[float, float, Node, Node, int]] = []
    visited = set()
    tie = count()

    def enqueue(a: Node, b: Node):
        nonlocal best_distance
        if id(a) > id(b):
            a, b = b, a
        key = (id(a), id(b))
        if key in visited:
            return
        visited.add(key)

        if dominance_prune and _dominates(a, b, eps=eps):
            stats["pruned_dom_point_pairs"] += mass(a, b)
            return

        lb, ub = _bounds_ball_pair(a, b, wc, eps=eps)
        if lb >= min(best_distance, tau) - eps:
            stats["pruned_lb_point_pairs"] += mass(a, b)
            return

        if ub < best_distance:
            best_distance = ub

        heapq.heappush(heap, (lb, ub, a, b, next(tie)))

    if len(root.children) < 2:
        result = (None, None, float("inf"))
        stats["unexplored_point_pairs"] = total_pairs
        return (*result, stats) if return_stats else result

    for i in range(len(root.children)):
        for j in range(i + 1, len(root.children)):
            enqueue(root.children[i], root.children[j])

    while heap and best_distance > tau + eps:
        lb, ub, a, b, _ = heapq.heappop(heap)
        if lb >= min(best_distance, tau):
            continue

        a_leaf = _node_is_leaf(a)
        b_leaf = _node_is_leaf(b)

        if a_leaf and b_leaf:
            stats["explored_point_pairs"] += mass(a, b)
            pair, dist, evals = _exact_leaf_eval(a, b, data, wc, eps=eps)
            stats["objective_evals"] += evals
            if pair is not None and dist < best_distance:
                best_pair = pair
                best_distance = dist
                stats["best_origin"] = "leaf"
            continue

        if not a_leaf and (b_leaf or a.radius >= b.radius):
            for child in a.children:
                enqueue(child, b)
        else:
            for child in b.children:
                enqueue(a, child)

    if best_pair is None or best_distance > tau:
        stack = [root]
        while stack:
            node = stack.pop()
            if _node_is_leaf(node):
                pair, dist, evals = _exact_leaf_self(node, data, wc, eps=eps)
                stats["objective_evals"] += evals
                stats["explored_point_pairs"] += evals
                if pair is not None and dist < best_distance:
                    best_pair = pair
                    best_distance = dist
                    stats["best_origin"] = "leaf"
            else:
                stack.extend(node.children)

    stats["best_pair"] = best_pair
    stats["best_distance"] = None if best_pair is None else best_distance
    stats["pruned_point_pairs"] = stats["pruned_lb_point_pairs"] + stats["pruned_dom_point_pairs"]
    stats["unexplored_point_pairs"] = stats["total_point_pairs"] - stats["pruned_point_pairs"] - stats["explored_point_pairs"]

    if best_pair is None:
        result = (None, None, float("inf"))
    else:
        result = (*best_pair, best_distance)
    return (*result, stats) if return_stats else result
