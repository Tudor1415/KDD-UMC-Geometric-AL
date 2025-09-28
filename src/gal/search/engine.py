"""Generic branch-and-bound search engine."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
import math
import time
from itertools import count
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..trees.common import GeometricTree, Node
from .bounds import BallTreeBounds, BoundContext, BoundsStrategy
from .strategies import DiversityVisitStrategy, VisitStrategy


@dataclass(frozen=True)
class SearchContext:
    data: np.ndarray
    wc: np.ndarray
    tau: float
    eps: float


class Search:
    """Dual-tree branch-and-bound search operating on :class:`GeometricTree` nodes."""

    def __init__(
        self,
        *,
        bounder: BoundsStrategy[Node] | None = None,
        strategy: VisitStrategy[Node] | None = None,
    ) -> None:
        self.bounder = bounder or BallTreeBounds()
        self.strategy = strategy or DiversityVisitStrategy()

    @staticmethod
    def _node_is_leaf(node: Node) -> bool:
        return bool(node.is_leaf or not node.children)

    @staticmethod
    def _descendant_size(node: Node, cache: Dict[int, int]) -> int:
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]
        if node.indices is not None and Search._node_is_leaf(node):
            size = int(node.indices.size)
        else:
            size = sum(Search._descendant_size(child, cache) for child in node.children)
        cache[node_id] = size
        return size

    @staticmethod
    def _gather_leaf_indices(node: Node) -> np.ndarray:
        stack = [node]
        leaves: list[np.ndarray] = []
        while stack:
            nd = stack.pop()
            if Search._node_is_leaf(nd) and nd.indices is not None:
                leaves.append(nd.indices)
            else:
                stack.extend(nd.children)
        if not leaves:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(leaves).astype(np.int64, copy=False))

    @staticmethod
    def _dominates(a: Node, b: Node, eps: float) -> bool:
        amin = a.center - a.radius
        amax = a.center + a.radius
        bmin = b.center - b.radius
        bmax = b.center + b.radius
        return bool(np.all(amin >= bmax - eps) or np.all(bmin >= amax - eps))

    @staticmethod
    def _objective_value(p: np.ndarray, q: np.ndarray, wc: np.ndarray, eps: float) -> float:
        diff = p - q
        denom = float(np.linalg.norm(diff))
        if denom <= eps:
            return 0.0
        return abs(float(np.dot(diff, wc))) / denom

    @staticmethod
    def _exact_leaf_eval(a: Node, b: Node, context: SearchContext) -> Tuple[Tuple[int, int] | None, float, int]:
        Ai = a.indices
        Bi = b.indices
        if Ai is None or Bi is None or Ai.size == 0 or Bi.size == 0:
            return None, float("inf"), 0
        XA = context.data[Ai]
        XB = context.data[Bi]
        diff = XA[:, None, :] - XB[None, :, :]
        num = np.abs(np.tensordot(diff, context.wc, axes=(2, 0)))
        denom = np.linalg.norm(diff, axis=2)
        denom = np.maximum(denom, context.eps)
        dist = num / denom
        m_idx, n_idx = np.unravel_index(np.argmin(dist), dist.shape)
        evals = int(Ai.size) * int(Bi.size)
        return (int(Ai[m_idx]), int(Bi[n_idx])), float(dist[m_idx, n_idx]), evals

    @staticmethod
    def _exact_leaf_self(node: Node, context: SearchContext) -> Tuple[Tuple[int, int] | None, float, int]:
        idx = node.indices
        if idx is None or idx.size < 2:
            return None, float("inf"), 0
        best_pair: Tuple[int, int] | None = None
        best_dist = float("inf")
        evals = 0
        XA = context.data[idx]
        for i in range(idx.size - 1):
            pi = XA[i]
            for j in range(i + 1, idx.size):
                pj = XA[j]
                evals += 1
                dist = Search._objective_value(pi, pj, context.wc, context.eps)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (int(idx[i]), int(idx[j]))
        return best_pair, best_dist, evals

    @staticmethod
    def _normalize_score(score: Sequence[float] | float | int) -> Tuple[float, ...]:
        if isinstance(score, (float, int)):
            return (float(score),)
        if isinstance(score, tuple):
            return tuple(float(x) for x in score)
        return tuple(float(x) for x in score)

    def search_pair(
        self,
        tree: GeometricTree | Node,
        X: np.ndarray,
        wc: np.ndarray,
        *,
        tau: float = float("inf"),
        return_stats: bool = False,
        dominance_prune: bool = True,
        eps: float = 1e-12,
        ensure_optimal: bool = True,
        time_checkpoints: Optional[Sequence[float]] = None,
        calls_checkpoints: Optional[Sequence[int]] = None,
        collect_bound_gaps: bool = False,
        collect_events: bool = False,
    ) -> Tuple[Optional[int], Optional[int], float] | Tuple[Optional[int], Optional[int], float, Dict[str, object]]:
        data = np.ascontiguousarray(X, dtype=np.float64)
        wc = np.asarray(wc, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X must be a 2D array")
        if wc.ndim != 1:
            raise ValueError("wc must be a 1D vector")
        if wc.size != data.shape[1]:
            raise ValueError("wc must have length equal to X.shape[1]")

        root = tree.root if isinstance(tree, GeometricTree) else tree
        context = SearchContext(data=data, wc=wc, tau=float(tau), eps=float(eps))

        leaf_indices = self._gather_leaf_indices(root)
        total_pairs = int(len(leaf_indices) * (len(leaf_indices) - 1) // 2)

        stats: Dict[str, object] = dict(
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
            return self._descendant_size(a, size_cache) * self._descendant_size(b, size_cache)

        best_pair: Tuple[int, int] | None = None
        best_distance = float("inf")
        # Heap items are tuples ordered for heapq comparisons.
        # IMPORTANT: ensure no Node objects appear before a numeric tiebreaker,
        # otherwise Python may attempt to compare Node instances when earlier
        # fields tie, causing TypeError. Use (score, tie, lb, ub, a, b).
        heap: list[Tuple[Tuple[float, ...], int, float, float, Node, Node]] = []
        visited: set[Tuple[int, int]] = set()
        tie = count()
        # Provide a per-search cache so bounders can reuse per-center computations
        bound_context = BoundContext(wc=wc, eps=float(eps), proj_cache={})
        self.strategy.setup(root, data=data)

        # Tracing support (optional)
        t0 = time.perf_counter()
        time_grid = None if time_checkpoints is None else list(time_checkpoints)
        calls_grid = None if calls_checkpoints is None else list(calls_checkpoints)
        time_idx = 0
        calls_idx = 0
        time_best: list[float] = []
        calls_best: list[float] = []
        bound_gaps: list[float] = []
        # Track heap size statistics (max size up to each calls checkpoint)
        heap_size_max = 0
        calls_heap_max: list[int] = []

        # Optional event logging (creation, expansion, pruning)
        events: list[Dict[str, float | int | str]] | None = [] if collect_events else None
        pair_ids: Dict[Tuple[int, int], int] = {}
        next_pair_id = 0

        def _pair_key(a: Node, b: Node) -> Tuple[int, int]:
            ia, ib = id(a), id(b)
            return (ia, ib) if ia <= ib else (ib, ia)

        def _assign_pair_id(a: Node, b: Node) -> int:
            nonlocal next_pair_id
            key = _pair_key(a, b)
            pid = pair_ids.get(key)
            if pid is None:
                pid = next_pair_id
                pair_ids[key] = pid
                next_pair_id += 1
            return pid

        def _log_event(ev_type: str, a: Node, b: Node, lb: float, ub: float, parent_id: int | None) -> int:
            if events is None:
                return -1
            pid = _assign_pair_id(a, b)
            events.append({
                "event_type": str(ev_type),
                "node_id": int(pid),
                "parent_id": int(-1 if parent_id is None else parent_id),
                "timestamp": float(time.perf_counter() - t0),
                "lower_bound": float(lb),
                "upper_bound": float(ub),
            })
            return pid

        def record_time_if_needed() -> None:
            nonlocal time_idx
            if time_grid is None:
                return
            now = time.perf_counter() - t0
            while time_idx < len(time_grid) and now >= float(time_grid[time_idx]):
                time_best.append(float(best_distance))
                time_idx += 1

        def record_calls_if_needed() -> None:
            nonlocal calls_idx
            if calls_grid is None:
                return
            cur = int(stats["objective_evals"]) if "objective_evals" in stats else 0
            while calls_idx < len(calls_grid) and cur >= int(calls_grid[calls_idx]):
                calls_best.append(float(best_distance))
                calls_heap_max.append(int(heap_size_max))
                calls_idx += 1

        def enqueue(a: Node, b: Node, *, parent_id: int | None = None) -> None:
            nonlocal best_distance, heap_size_max
            if id(a) > id(b):
                a, b = b, a
            key = (id(a), id(b))
            if key in visited:
                return
            visited.add(key)

            pair_mass = mass(a, b)

            if dominance_prune and self._dominates(a, b, eps=eps):
                stats["pruned_dom_point_pairs"] = int(stats["pruned_dom_point_pairs"]) + pair_mass
                # Log prune with NaN bounds (no bounder invoked)
                if collect_events:
                    _log_event("PRUNED", a, b, float("nan"), float("nan"), parent_id)
                return

            bounds = self.bounder(a, b, bound_context)
            if bounds.lower >= min(best_distance, tau) - eps:
                stats["pruned_lb_point_pairs"] = int(stats["pruned_lb_point_pairs"]) + pair_mass
                # Pruned by lower bound at enqueue time
                if collect_events:
                    _log_event("PRUNED", a, b, float(bounds.lower), float(bounds.upper), parent_id)
                return

            # Do not update the incumbent best distance with an upper bound.
            # Only exact evaluations are allowed to improve best_distance to avoid
            # prematurely terminating under a finite tau without a feasible pair.

            score = self._normalize_score(self.strategy.priority(a, b, bounds, pair_mass))
            # Push with a numeric tie-breaker before Node objects to avoid
            # comparisons between Node instances when tuple prefixes tie.
            heapq.heappush(heap, (score, next(tie), bounds.lower, bounds.upper, a, b))
            if collect_events:
                _log_event("CREATED", a, b, float(bounds.lower), float(bounds.upper), parent_id)
            # Update heap size maximum after every push
            if len(heap) > heap_size_max:
                heap_size_max = len(heap)
            record_time_if_needed()

        # Even if the root has fewer than two children, we can still evaluate
        # within-leaf pairs and/or fall back to exhaustive evaluation.

        for i in range(len(root.children)):
            for j in range(i + 1, len(root.children)):
                enqueue(root.children[i], root.children[j], parent_id=None)
        # Also explore within-subtree pairs by enqueuing (child, child)
        # so that pairs across different leaves under the same branch are considered.
        for ch in root.children:
            if not self._node_is_leaf(ch):
                enqueue(ch, ch, parent_id=None)

        # If tau is infinite, explore all queued pairs; otherwise stop early when possible.
        while heap and (math.isinf(tau) or best_distance > tau + eps):
            _, _, lb, ub, a, b = heapq.heappop(heap)
            if collect_bound_gaps:
                gap = float(max(0.0, ub - lb))
                bound_gaps.append(gap)
            if lb >= min(best_distance, tau):
                # Popped but immediately pruned by updated incumbent/tau
                if collect_events:
                    # parent unknown here (already created earlier); set to existing id
                    pid = _assign_pair_id(a, b)
                    events.append({
                        "event_type": "PRUNED",
                        "node_id": int(pid),
                        "parent_id": int(-1),
                        "timestamp": float(time.perf_counter() - t0),
                        "lower_bound": float(lb),
                        "upper_bound": float(ub),
                    })
                continue

            a_leaf = self._node_is_leaf(a)
            b_leaf = self._node_is_leaf(b)

            parent_pid: int | None = None
            if collect_events:
                parent_pid = _log_event("EXPANDED", a, b, float(lb), float(ub), None)

            if a_leaf and b_leaf:
                pair_mass = mass(a, b)
                stats["explored_point_pairs"] = int(stats["explored_point_pairs"]) + pair_mass
                # When both nodes are the same leaf, evaluate within-leaf pairs only once
                # to avoid counting i==j pairs.
                if a is b:
                    pair, dist, evals = self._exact_leaf_self(a, context)
                else:
                    pair, dist, evals = self._exact_leaf_eval(a, b, context)
                stats["objective_evals"] = int(stats["objective_evals"]) + evals
                if pair is not None and dist < min(best_distance, tau):
                    best_pair = pair
                    best_distance = dist
                    stats["best_origin"] = "leaf"
                record_calls_if_needed()
                record_time_if_needed()
                continue

            # Special handling when exploring within the same subtree (a is b):
            # generate child-pair combinations to cover cross-leaf candidates under this branch.
            if a is b and not a_leaf:
                for i in range(len(a.children)):
                    for j in range(i + 1, len(a.children)):
                        enqueue(a.children[i], a.children[j], parent_id=parent_pid)
                # Continue descending within each child as needed
                for child in a.children:
                    if not self._node_is_leaf(child):
                        enqueue(child, child, parent_id=parent_pid)
            elif not a_leaf and (b_leaf or a.radius >= b.radius):
                for child in a.children:
                    enqueue(child, b, parent_id=parent_pid)
            else:
                for child in b.children:
                    enqueue(a, child, parent_id=parent_pid)
            record_time_if_needed()

        # Evaluate pairs within the same leaf across the whole tree only as a last recourse.
        # Policy (applies to all modes):
        # - Run the sweep only if no pair was found during BnB (best_pair is None).
        # - Otherwise, skip sweeping to avoid upfront within-leaf costs.
        # Only sweep when BnB found nothing and there is nothing left to explore.
        # This prevents invoking sweeps when an early exit occurred due to bounds.
        if best_pair is None and not heap:
            stack = [root]
            # Early-stop if running under a finite tau: stop once satisfied
            stop_after_found = (not math.isinf(tau))
            done = False
            while stack and not done:
                node = stack.pop()
                if self._node_is_leaf(node):
                    pair, dist, evals = self._exact_leaf_self(node, context)
                    stats["objective_evals"] = int(stats["objective_evals"]) + evals
                    stats["explored_point_pairs"] = int(stats["explored_point_pairs"]) + evals
                    if pair is not None and dist < min(best_distance, tau):
                        best_pair = pair
                        best_distance = dist
                        stats["best_origin"] = "leaf"
                        # Under a finite tau, stop once satisfied
                        if stop_after_found and best_distance <= tau + eps:
                            done = True
                            break
                else:
                    # Continue traversing to reach all leaves
                    stack.extend(node.children)
        # Optionally run exhaustive verification to ensure the optimal solution.
        if ensure_optimal:
            data_points = context.data
            unique_indices = leaf_indices
            for idx_a in range(len(unique_indices) - 1):
                ia = int(unique_indices[idx_a])
                pa = data_points[ia]
                for idx_b in range(idx_a + 1, len(unique_indices)):
                    ib = int(unique_indices[idx_b])
                    pb = data_points[ib]
                    dist = self._objective_value(pa, pb, wc, eps)
                    stats["objective_evals"] = int(stats["objective_evals"]) + 1
                    stats["explored_point_pairs"] = int(stats["explored_point_pairs"]) + 1
                    if dist < min(best_distance, tau):
                        best_pair = (ia, ib)
                        best_distance = dist
                        stats["best_origin"] = "exhaustive"

        # Finalize traces
        if time_grid is not None:
            while time_idx < len(time_grid):
                time_best.append(float(best_distance))
                time_idx += 1
        if calls_grid is not None:
            while calls_idx < len(calls_grid):
                calls_best.append(float(best_distance))
                calls_heap_max.append(int(heap_size_max))
                calls_idx += 1

        stats["best_pair"] = best_pair
        stats["best_distance"] = None if best_pair is None else best_distance
        stats["pruned_point_pairs"] = int(stats["pruned_lb_point_pairs"]) + int(stats["pruned_dom_point_pairs"])
        stats["unexplored_point_pairs"] = stats["total_point_pairs"] - stats["pruned_point_pairs"] - stats["explored_point_pairs"]
        if time_grid is not None or calls_grid is not None or collect_bound_gaps or collect_events:
            trace: Dict[str, object] = {}
            if time_grid is not None:
                trace["time_grid"] = list(map(float, time_grid))
                trace["time_best"] = list(map(float, time_best))
            if calls_grid is not None:
                trace["calls_grid"] = list(map(int, calls_grid))
                trace["calls_best"] = list(map(float, calls_best))
                trace["calls_heap_max"] = list(map(int, calls_heap_max))
            if collect_bound_gaps:
                trace["bound_gaps"] = list(map(float, bound_gaps))
            if collect_events and events is not None:
                trace["events"] = events
            stats["trace"] = trace

        if best_pair is None:
            result = (None, None, float("inf"))
        else:
            result = (*best_pair, best_distance)
        return (*result, stats) if return_stats else result


def search_pair(
    tree: GeometricTree | Node,
    X: np.ndarray,
    wc: np.ndarray,
    *,
    tau: float = float("inf"),
    return_stats: bool = False,
    dominance_prune: bool = True,
    eps: float = 1e-12,
    ensure_optimal: bool = True,
    bounder: BoundsStrategy[Node] | None = None,
    strategy: VisitStrategy[Node] | None = None,
    time_checkpoints: Optional[Sequence[float]] = None,
    calls_checkpoints: Optional[Sequence[int]] = None,
    collect_bound_gaps: bool = False,
    collect_events: bool = False,
) -> Tuple[Optional[int], Optional[int], float] | Tuple[Optional[int], Optional[int], float, Dict[str, object]]:
    """Convenience wrapper using the :class:`Search` engine."""

    engine = Search(bounder=bounder, strategy=strategy)
    return engine.search_pair(
        tree,
        X,
        wc,
        tau=tau,
        return_stats=return_stats,
        dominance_prune=dominance_prune,
        eps=eps,
        ensure_optimal=ensure_optimal,
        time_checkpoints=time_checkpoints,
        calls_checkpoints=calls_checkpoints,
        collect_bound_gaps=collect_bound_gaps,
        collect_events=collect_events,
    )
