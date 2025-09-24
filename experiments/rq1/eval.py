from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from gal.search import Search
from gal.search.kd_bounds import KdTreeBounds
from gal.search.strategies import get_strategy
from gal.trees import kd_tree as kd
from gal import trees as bt


def _bootstrap_ci(mats: np.ndarray, n_bootstrap: int, ci_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # mats shape: [n_runs, n_points]
    rng = np.random.default_rng(12345)
    n_runs, n_pts = mats.shape
    if n_runs == 0:
        zeros = np.zeros(n_pts)
        return zeros, zeros, zeros
    meds = np.median(mats, axis=0)
    if n_runs == 1:
        return meds, meds, meds
    boot = np.empty((n_bootstrap, n_pts), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[b] = np.median(mats[idx], axis=0)
    alpha = (1.0 - ci_level) / 2.0
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1.0 - alpha, axis=0)
    return meds, lo, hi


def _sample_center(d: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(d)
    s = u.sum()
    return (u / s) if s > 0 else np.ones(d) / float(d)


# no brute-force oracle; d_star is min of kd- and ball-tree BnB completions


@dataclass
class MethodResult:
    A_time: np.ndarray
    A_calls: np.ndarray
    bound_gaps: np.ndarray


def evaluate_dataset(
    X: np.ndarray,
    *,
    kd_strategy_name: str,
    bt_strategy_name: str,
    bt_build_method: str = "disjoint_greedy",
    kd_config: Dict[str, Any] | None = None,
    bt_config: Dict[str, Any] | None = None,
    time_fracs: List[float],
    call_fracs: List[float],
    timing_repeats: int,
    eps: float,
    rng: np.random.Generator,
    center: np.ndarray | None = None,
    random_pair_mode: str = "with_replacement",
    kd_tree_obj: Any | None = None,
    bt_tree_obj: Any | None = None,
) -> Dict[str, MethodResult]:
    n = X.shape[0]
    Pmax = n * (n - 1) // 2
    time_fracs = list(map(float, time_fracs))
    call_fracs = list(map(float, call_fracs))

    kd_tree = kd_tree_obj if kd_tree_obj is not None else kd.build_tree(X, None if kd_config is None else dict(kd_config))
    bt_tree = bt_tree_obj if bt_tree_obj is not None else bt.build_tree(X, None if bt_config is None else dict(bt_config), method=str(bt_build_method))

    kd_strategy = get_strategy(kd_strategy_name)
    bt_strategy = get_strategy(bt_strategy_name, queries=X)
    kd_engine = Search(bounder=KdTreeBounds(), strategy=kd_strategy)
    bt_engine = Search(strategy=bt_strategy)

    # Sample a shared center
    wc = _sample_center(X.shape[1], rng) if center is None else np.asarray(center, dtype=float)
    # Oracle d*: min of exact BnB using kd- and ball-tree
    kd_i, kd_j, kd_star = kd_engine.search_pair(kd_tree, X, wc, tau=float("inf"))
    bt_i, bt_j, bt_star = bt_engine.search_pair(bt_tree, X, wc, tau=float("inf"))
    d_star = min(kd_star, bt_star)

    # Time normalization: run both methods to completion multiple times
    def time_to_completion(engine: Search, tree) -> float:
        import time as _t
        best = float("inf")
        elapsed = 0.0
        for _ in range(timing_repeats):
            t0 = _t.perf_counter()
            i, j, d = engine.search_pair(tree, X, wc, tau=float("inf"))
            elapsed = max(elapsed, _t.perf_counter() - t0)
            best = min(best, d)
        return elapsed

    T_kd = time_to_completion(kd_engine, kd_tree)
    T_bt = time_to_completion(bt_engine, bt_tree)
    T_max = max(T_kd, T_bt)

    time_grid = [f * T_max for f in time_fracs]
    calls_grid = [int(round(f * Pmax)) for f in call_fracs]

    # Run anytime with tracing enabled
    _, _, kd_best, kd_stats = kd_engine.search_pair(
        kd_tree,
        X,
        wc,
        tau=float("inf"),
        return_stats=True,
        dominance_prune=True,
        eps=eps,
        time_checkpoints=time_grid,
        calls_checkpoints=calls_grid,
        collect_bound_gaps=True,
    )
    _, _, bt_best, bt_stats = bt_engine.search_pair(
        bt_tree,
        X,
        wc,
        tau=float("inf"),
        return_stats=True,
        dominance_prune=True,
        eps=eps,
        time_checkpoints=time_grid,
        calls_checkpoints=calls_grid,
        collect_bound_gaps=True,
    )

    def to_A(trace: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        tb = np.array(trace["time_best"], dtype=float)
        cb = np.array(trace["calls_best"], dtype=float)
        # Use eps in numerator and denominator to avoid 0/0 when d* = 0 and best = 0
        A_t = np.minimum(1.0, (d_star + eps) / (tb + eps))
        A_m = np.minimum(1.0, (d_star + eps) / (cb + eps))
        return A_t, A_m

    kd_A_t, kd_A_m = to_A(kd_stats["trace"])  # type: ignore
    bt_A_t, bt_A_m = to_A(bt_stats["trace"])  # type: ignore

    # Random sampling baseline
    def random_anytime(
        X: np.ndarray,
        wc: np.ndarray,
        time_grid: List[float],
        calls_grid: List[int],
        eps: float,
        rng: np.random.Generator,
        *,
        mode: str = "with_replacement",
    ) -> Tuple[np.ndarray, np.ndarray]:
        import time as _t
        n = X.shape[0]
        best = float("inf")
        A_t = np.zeros(len(time_grid), dtype=float)
        A_m = np.zeros(len(calls_grid), dtype=float)
        ti = ci = 0
        t0 = _t.perf_counter()
        calls = 0
        mode = str(mode).strip().lower()
        if mode in {"w", "with", "with_replacement"}:
            sampler = None  # on-the-fly indices
        elif mode in {"wo", "without", "without_replacement"}:
            # Use index mapping from [0, Pmax) -> (i,j) to avoid storing pairs
            P = n * (n - 1) // 2
            order = rng.permutation(P)
            k_ptr = 0
            def idx_to_pair(k: int) -> Tuple[int, int]:
                # Map 0 <= k < nC2 to unique (i,j), 0 <= i < j < n
                # Compute i such that T(i) <= k < T(i+1), where T(i)=i*(2n - i -1)/2
                lo, hi = 0, n - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    Tmid = mid * (2 * n - mid - 1) // 2
                    if Tmid <= k:
                        lo = mid + 1
                    else:
                        hi = mid
                i = lo - 1
                Ti = i * (2 * n - i - 1) // 2
                j = i + 1 + (k - Ti)
                return int(i), int(j)
        else:
            sampler = None

        while (ti < len(time_grid)) or (ci < len(calls_grid)):
            if mode in {"wo", "without", "without_replacement"}:
                if k_ptr >= order.size:
                    # Exhausted all pairs
                    break
                i, j = idx_to_pair(int(order[k_ptr]))
                k_ptr += 1
            else:
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                if i == j:
                    continue
                if j < i:
                    i, j = j, i
            diff = X[i] - X[j]
            denom = float(np.linalg.norm(diff))
            val = 0.0 if denom <= eps else abs(float(np.dot(diff, wc))) / denom
            best = min(best, val)
            calls += 1
            now = _t.perf_counter() - t0
            while ti < len(time_grid) and now >= float(time_grid[ti]):
                A_t[ti] = min(1.0, d_star / max(best, np.finfo(float).tiny))
                ti += 1
            while ci < len(calls_grid) and calls >= int(calls_grid[ci]):
                A_m[ci] = min(1.0, d_star / max(best, np.finfo(float).tiny))
                ci += 1
            if calls >= (n * (n - 1) // 2) and ti >= len(time_grid):
                break
        # Pad remaining
        while ti < len(time_grid):
            A_t[ti] = min(1.0, d_star / max(best, np.finfo(float).tiny))
            ti += 1
        while ci < len(calls_grid):
            A_m[ci] = min(1.0, d_star / max(best, np.finfo(float).tiny))
            ci += 1
        return A_t, A_m

    rnd_A_t, rnd_A_m = random_anytime(X, wc, time_grid, calls_grid, eps, rng, mode=random_pair_mode)

    return {
        "kd": MethodResult(A_time=kd_A_t, A_calls=kd_A_m, bound_gaps=np.array(kd_stats["trace"]["bound_gaps"], dtype=float)),  # type: ignore
        "bt": MethodResult(A_time=bt_A_t, A_calls=bt_A_m, bound_gaps=np.array(bt_stats["trace"]["bound_gaps"], dtype=float)),  # type: ignore
        "rnd": MethodResult(A_time=rnd_A_t, A_calls=rnd_A_m, bound_gaps=np.zeros(0)),
    }


def aggregate_runs(
    run_results: List[Dict[str, MethodResult]],
    *,
    n_bootstrap: int,
    ci_level: float,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    # Stack per-method A@t and A@m
    methods = run_results[0].keys()
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for m in methods:
        A_t = np.stack([r[m].A_time for r in run_results], axis=0)
        A_m = np.stack([r[m].A_calls for r in run_results], axis=0)
        out[m] = {
            "A_time": _bootstrap_ci(A_t, n_bootstrap, ci_level),
            "A_calls": _bootstrap_ci(A_m, n_bootstrap, ci_level),
        }
    return out
