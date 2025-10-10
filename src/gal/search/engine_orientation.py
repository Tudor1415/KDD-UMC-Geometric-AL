"""Orientation-aware branch-and-bound search implementation."""

from __future__ import annotations

import heapq
import math
import time
from itertools import count
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..trees.common import Node
from .bounds import BoundContext
from .context import SearchContext
from .leaf_eval import exact_leaf_eval, exact_leaf_self
from .priority import normalize_priority
from .tree_utils import descendant_size, gather_leaf_indices, node_is_leaf, dominates


def run_orientation_search(
    search: "Search",
    root: Node,
    data: np.ndarray,
    wc: np.ndarray,
    *,
    tau: float,
    orientation: np.ndarray,
    return_stats: bool,
    dominance_prune: bool,
    eps: float,
    time_checkpoints: Optional[Sequence[float]],
    calls_checkpoints: Optional[Sequence[int]],
    collect_events: bool,
) -> Tuple[Optional[int], Optional[int], float] | Tuple[Optional[int], Optional[int], float, Dict[str, object]]:
    context = SearchContext(
        data=data,
        wc=wc,
        tau=float(tau),
        eps=float(eps),
        seen_pairs=frozenset(search._seen_pairs),
        orientation=orientation,
        orientation_mode=True,
    )

    leaf_indices = gather_leaf_indices(root)
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
        best_orientation=None,
        pruned_orientation_point_pairs=0,
        orientation_mode=True,
    )

    if len(leaf_indices) < 2:
        result = (None, None, float("inf"))
        stats["unexplored_point_pairs"] = 0
        return (*result, stats) if return_stats else result

    size_cache: Dict[int, int] = {}

    def mass(a: Node, b: Node) -> int:
        return descendant_size(a, size_cache) * descendant_size(b, size_cache)

    best_pair: Tuple[int, int] | None = None
    best_distance = float("inf")
    best_orientation_val = float("-inf")
    heap: list[Tuple[Tuple[float, ...], int, float, float, Node, Node]] = []
    visited: set[Tuple[int, int]] = set()
    tie = count()
    bound_context = BoundContext(
        wc=wc,
        eps=float(eps),
        proj_cache={},
        orientation=orientation,
    )
    search.strategy.setup(
        root,
        data=data,
        wc=wc,
        tau=float(tau),
        eps=float(eps),
        orientation=orientation,
        orientation_mode=True,
    )

    t0 = time.perf_counter()
    time_grid = None if time_checkpoints is None else list(time_checkpoints)
    calls_grid = None if calls_checkpoints is None else list(calls_checkpoints)
    time_idx = 0
    calls_idx = 0
    time_best: list[float] = []
    calls_best: list[float] = []
    heap_size_max = 0
    calls_heap_max: list[int] = []

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
        nonlocal best_distance, best_orientation_val, heap_size_max
        if id(a) > id(b):
            a, b = b, a
        key = (id(a), id(b))
        if key in visited:
            return
        visited.add(key)

        pair_mass = mass(a, b)

        if dominance_prune and dominates(a, b, eps=eps):
            stats["pruned_dom_point_pairs"] = int(stats["pruned_dom_point_pairs"]) + pair_mass
            if collect_events:
                _log_event("PRUNED", a, b, float("nan"), float("nan"), parent_id)
            return

        bounds = search.bounder(a, b, bound_context)
        distance_cutoff = tau
        if bounds.lower >= distance_cutoff - eps:
            stats["pruned_lb_point_pairs"] = int(stats["pruned_lb_point_pairs"]) + pair_mass
            if collect_events:
                _log_event("PRUNED", a, b, float(bounds.lower), float(bounds.upper), parent_id)
            return

        orientation_upper = None
        if bounds.orientation is not None:
            lb_orient, ub_orient = bounds.orientation
            orientation_upper = max(abs(float(lb_orient)), abs(float(ub_orient)))
        if (
            best_orientation_val > -math.inf
            and orientation_upper is not None
            and orientation_upper <= best_orientation_val + eps
        ):
            stats["pruned_orientation_point_pairs"] = int(stats["pruned_orientation_point_pairs"]) + pair_mass
            if collect_events:
                _log_event("PRUNED", a, b, float(bounds.lower), float(bounds.upper), parent_id)
            return

        raw_score = search.strategy.priority(a, b, bounds, pair_mass)
        if raw_score is None:
            return
        score = normalize_priority(raw_score)
        heapq.heappush(heap, (score, next(tie), bounds.lower, bounds.upper, a, b))
        if collect_events:
            _log_event("CREATED", a, b, float(bounds.lower), float(bounds.upper), parent_id)
        if len(heap) > heap_size_max:
            heap_size_max = len(heap)
        record_time_if_needed()

    for i in range(len(root.children)):
        for j in range(i + 1, len(root.children)):
            enqueue(root.children[i], root.children[j], parent_id=None)
    for ch in root.children:
        if not node_is_leaf(ch):
            enqueue(ch, ch, parent_id=None)

    while heap:
        _, _, lb, ub, a, b = heapq.heappop(heap)
        bounds = search.bounder(a, b, bound_context)
        lb_val = float(bounds.lower)
        ub_val = float(bounds.upper)
        distance_cutoff = tau
        if lb_val >= distance_cutoff - eps:
            if collect_events:
                pid = _assign_pair_id(a, b)
                events.append({
                    "event_type": "PRUNED",
                    "node_id": int(pid),
                    "parent_id": int(-1),
                    "timestamp": float(time.perf_counter() - t0),
                    "lower_bound": lb_val,
                    "upper_bound": ub_val,
                })
            continue

        orientation_upper = None
        if bounds.orientation is not None:
            lb_orient, ub_orient = bounds.orientation
            orientation_upper = max(abs(float(lb_orient)), abs(float(ub_orient)))
        if (
            best_orientation_val > -math.inf
            and orientation_upper is not None
            and orientation_upper <= best_orientation_val + eps
        ):
            if collect_events:
                pid = _assign_pair_id(a, b)
                events.append({
                    "event_type": "PRUNED",
                    "node_id": int(pid),
                    "parent_id": int(-1),
                    "timestamp": float(time.perf_counter() - t0),
                    "lower_bound": lb_val,
                    "upper_bound": ub_val,
                })
            stats["pruned_orientation_point_pairs"] = int(stats["pruned_orientation_point_pairs"]) + mass(a, b)
            continue

        a_leaf = node_is_leaf(a)
        b_leaf = node_is_leaf(b)

        parent_pid: int | None = None
        if collect_events:
            parent_pid = _log_event("EXPANDED", a, b, lb_val, ub_val, None)

        if a_leaf and b_leaf:
            pair_mass = mass(a, b)
            stats["explored_point_pairs"] = int(stats["explored_point_pairs"]) + pair_mass
            if a is b:
                pair, dist, evals, orient_val = exact_leaf_self(a, context)
            else:
                pair, dist, evals, orient_val = exact_leaf_eval(a, b, context)
            stats["objective_evals"] = int(stats["objective_evals"]) + evals

            if pair is not None and dist <= tau + eps and orient_val is not None:
                if (
                    orient_val > best_orientation_val + eps
                    or (abs(orient_val - best_orientation_val) <= eps and dist < best_distance)
                ):
                    best_pair = pair
                    best_distance = dist
                    best_orientation_val = orient_val
                    stats["best_origin"] = "leaf"
                    stats["best_orientation"] = best_orientation_val
            record_calls_if_needed()
            record_time_if_needed()
            continue

        if a is b and not a_leaf:
            for i in range(len(a.children)):
                for j in range(i + 1, len(a.children)):
                    enqueue(a.children[i], a.children[j], parent_id=parent_pid)
            for child in a.children:
                if not node_is_leaf(child):
                    enqueue(child, child, parent_id=parent_pid)
        elif not a_leaf and (b_leaf or a.radius >= b.radius):
            for child in a.children:
                enqueue(child, b, parent_id=parent_pid)
        else:
            for child in b.children:
                enqueue(a, child, parent_id=parent_pid)
        record_time_if_needed()

    if best_pair is None and not heap:
        stack = [root]
        stop_after_found = (not math.isinf(tau))
        done = False
        while stack and not done:
            node = stack.pop()
            if node_is_leaf(node):
                pair, dist, evals, orient_val = exact_leaf_self(node, context)
                stats["objective_evals"] = int(stats["objective_evals"]) + evals
                stats["explored_point_pairs"] = int(stats["explored_point_pairs"]) + evals
                if (
                    pair is not None
                    and dist <= tau + eps
                    and orient_val is not None
                    and (
                        orient_val > best_orientation_val + eps
                        or (abs(orient_val - best_orientation_val) <= eps and dist < best_distance)
                    )
                ):
                    best_pair = pair
                    best_distance = dist
                    best_orientation_val = orient_val
                    stats["best_origin"] = "leaf"
                    stats["best_orientation"] = best_orientation_val
                    if stop_after_found and best_distance <= tau + eps:
                        done = True
                        break
            else:
                stack.extend(node.children)

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
    if best_pair is not None and best_orientation_val > -math.inf:
        stats["best_orientation"] = best_orientation_val
    else:
        stats["best_orientation"] = None
    stats["pruned_point_pairs"] = (
        int(stats["pruned_lb_point_pairs"]) +
        int(stats["pruned_dom_point_pairs"]) +
        int(stats["pruned_orientation_point_pairs"])
    )
    stats["unexplored_point_pairs"] = stats["total_point_pairs"] - stats["pruned_point_pairs"] - stats["explored_point_pairs"]
    if time_grid is not None or calls_grid is not None or collect_events:
        trace: Dict[str, object] = {}
        if time_grid is not None:
            trace["time_grid"] = list(map(float, time_grid))
            trace["time_best"] = list(map(float, time_best))
        if calls_grid is not None:
            trace["calls_grid"] = list(map(int, calls_grid))
            trace["calls_best"] = list(map(float, calls_best))
            trace["calls_heap_max"] = list(map(int, calls_heap_max))
        if collect_events and events is not None:
            trace["events"] = events
        stats["trace"] = trace

    if best_pair is None:
        result = (None, None, float("inf"))
    else:
        i_idx, j_idx = best_pair
        key = (i_idx, j_idx) if i_idx <= j_idx else (j_idx, i_idx)
        search._seen_pairs.add(key)
        result = (*best_pair, best_distance)
    return (*result, stats) if return_stats else result


__all__ = ["run_orientation_search"]
