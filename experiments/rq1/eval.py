from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import logging
import time as _t

import numpy as np
try:  # optional parallel bootstrap
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore

from gal.search import Search
from gal.search.kd_bounds import KdTreeBounds
from gal.search.strategies import get_strategy
from gal.trees import kd_tree as kd
from gal import trees as bt


def _bootstrap_ci(mats: np.ndarray, n_bootstrap: int, ci_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median + bootstrap CI across the first axis of mats.

    Uses joblib for parallel bootstrap when available; otherwise falls back to serial.
    Deterministic given the fixed RNG seed below.
    """
    # mats shape: [n_runs, n_points]
    rng = np.random.default_rng(12345)
    n_runs, n_pts = mats.shape
    if n_runs == 0:
        zeros = np.zeros(n_pts)
        return zeros, zeros, zeros
    meds = np.median(mats, axis=0)
    if n_runs == 1 or n_bootstrap <= 1:
        return meds, meds, meds

    def _one_boot(seed: int) -> np.ndarray:
        r = np.random.default_rng(int(seed))
        idx = r.integers(0, n_runs, size=n_runs)
        return np.median(mats[idx], axis=0)

    seeds = rng.integers(0, 2**32 - 1, size=n_bootstrap, dtype=np.uint64)
    if Parallel is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot_list = Parallel(n_jobs=-1)(delayed(_one_boot)(int(s)) for s in seeds)
        boot = np.vstack(boot_list)
    else:
        boot = np.empty((n_bootstrap, n_pts), dtype=float)
        for b, s in enumerate(seeds):
            boot[b] = _one_boot(int(s))

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
    heap_calls: np.ndarray


# --- Fixed-width scientific-notation logging helpers ---
_COLW_PHASE = 10
_COLW_METH = 6
_COLW_NUM = 12
_COLW_PCT = 10


def _sci(x: float) -> str:
    try:
        return f"{float(x):>{_COLW_NUM}.4e}"
    except Exception:
        return f"{x!s:>{_COLW_NUM}}"


def _row_prefix(phase: str, method: str) -> str:
    return f"{phase:<{_COLW_PHASE}} {method:<{_COLW_METH}}"


def _pct_str(x: int | float, total: int | float) -> str:
    try:
        t = float(total) if float(total) > 0 else 1.0
        p = 100.0 * float(x) / t
        return f"{p:>{_COLW_PCT}.2f}%"
    except Exception:
        return f"{x!s:>{_COLW_PCT}}"


def _fmt_time_norm_row(method: str, time_s: float, evals: int, best: float,
                       pr_lb: int, pr_dom: int, pr_tot: int, explored: int, total: int) -> str:
    return (
        f"{_row_prefix('time-norm', method)} "
        f"{_sci(time_s)} {_sci(best)} {_pct_str(evals, total)} {_pct_str(pr_lb, total)} {_pct_str(pr_dom, total)} "
        f"{_pct_str(pr_tot, total)} {_pct_str(explored, total)} {_sci(total)}"
    )


def _fmt_trace_row(method: str, dur_s: float, evals: int, best: float,
                   pr_lb: int, pr_dom: int, pr_tot: int, explored: int, total: int) -> str:
    return (
        f"{_row_prefix('trace', method)} "
        f"{_sci(dur_s)} {_pct_str(evals, total)} {_sci(best)} {_pct_str(pr_lb, total)} {_pct_str(pr_dom, total)} "
        f"{_pct_str(pr_tot, total)} {_pct_str(explored, total)} {_sci(total)}"
    )


def _time_norm_header(tau: float) -> str:
    return (
        f"Threshold TAU = {_sci(tau).strip()}\n"
        f"{'PHASE':<{_COLW_PHASE}} {'METH':<{_COLW_METH}} "
        f"{'TIME_S':>{_COLW_NUM}} {'BEST':>{_COLW_NUM}} {'EVALS%':>{_COLW_PCT}} "
        f"{'PR_LB%':>{_COLW_PCT}} {'PR_DOM%':>{_COLW_PCT}} {'PR_TOT%':>{_COLW_PCT}} "
        f"{'EXPLORED%':>{_COLW_PCT}} {'TOTAL':>{_COLW_NUM}}"
    )


def _trace_header() -> str:
    return (
        f"{'PHASE':<{_COLW_PHASE}} {'METH':<{_COLW_METH}} "
        f"{'DUR_S':>{_COLW_NUM}} {'EVALS%':>{_COLW_PCT}} {'BEST':>{_COLW_NUM}} "
        f"{'PR_LB%':>{_COLW_PCT}} {'PR_DOM%':>{_COLW_PCT}} {'PR_TOT%':>{_COLW_PCT}} "
        f"{'EXPLORED%':>{_COLW_PCT}} {'TOTAL':>{_COLW_NUM}}"
    )


"""Header printing is controlled by the caller to avoid duplication."""


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
    trace_tau: float | None = None,
    trace_certify: bool = False,
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
    # Backward-compatible default: if no threshold provided, use 1.0
    tau_tr = 1.0 if (trace_tau is None) else float(trace_tau)
    # Time normalization: run both methods until threshold

    def time_to_completion(engine: Search, tree, label: str) -> float:
        best = float("inf")
        elapsed = 0.0
        for _ in range(timing_repeats):
            t0 = _t.perf_counter()
            i, j, d, stats = engine.search_pair(
                tree,
                X,
                wc,
                tau=float(tau_tr),
                ensure_optimal=False,
                return_stats=True,
                collect_bound_gaps=False,
            )
            elapsed = max(elapsed, _t.perf_counter() - t0)
            best = min(best, d)
        lbp = int(stats.get("pruned_lb_point_pairs", 0)) if isinstance(stats, dict) else 0
        dmp = int(stats.get("pruned_dom_point_pairs", 0)) if isinstance(stats, dict) else 0
        prn = int(stats.get("pruned_point_pairs", 0)) if isinstance(stats, dict) else lbp + dmp
        expl = int(stats.get("explored_point_pairs", 0)) if isinstance(stats, dict) else 0
        tot = int(stats.get("total_point_pairs", 0)) if isinstance(stats, dict) else 0
        evl = int(stats.get("objective_evals", 0)) if isinstance(stats, dict) else 0
        logging.info(_fmt_time_norm_row(label, elapsed, evl, best, lbp, dmp, prn, expl, tot))
        return elapsed

    T_kd = time_to_completion(kd_engine, kd_tree, "kd")
    T_bt = time_to_completion(bt_engine, bt_tree, "ball")
    T_max = max(T_kd, T_bt)

    time_grid = [f * T_max for f in time_fracs]
    calls_grid = [int(round(f * Pmax)) for f in call_fracs]

    # Run anytime with tracing enabled
    t0_kd = _t.perf_counter()
    _, _, kd_best, kd_stats = kd_engine.search_pair(
        kd_tree,
        X,
        wc,
        tau=tau_tr,
        return_stats=True,
        dominance_prune=True,
        eps=eps,
        ensure_optimal=trace_certify,
        time_checkpoints=time_grid,
        calls_checkpoints=calls_grid,
        collect_bound_gaps=True,
    )
    dt_kd = _t.perf_counter() - t0_kd
    kd_lbp = int(kd_stats.get("pruned_lb_point_pairs", 0))
    kd_dmp = int(kd_stats.get("pruned_dom_point_pairs", 0))
    kd_prn = int(kd_stats.get("pruned_point_pairs", kd_lbp + kd_dmp))
    kd_expl = int(kd_stats.get("explored_point_pairs", 0))
    kd_tot = int(kd_stats.get("total_point_pairs", 0))
    logging.info(
        _fmt_trace_row(
            "kd",
            dur_s=dt_kd,
            evals=int(kd_stats.get("objective_evals", 0)),
            best=kd_best,
            pr_lb=kd_lbp,
            pr_dom=kd_dmp,
            pr_tot=kd_prn,
            explored=kd_expl,
            total=kd_tot,
        )
    )

    t0_bt = _t.perf_counter()
    _, _, bt_best, bt_stats = bt_engine.search_pair(
        bt_tree,
        X,
        wc,
        tau=tau_tr,
        return_stats=True,
        dominance_prune=True,
        eps=eps,
        ensure_optimal=trace_certify,
        time_checkpoints=time_grid,
        calls_checkpoints=calls_grid,
        collect_bound_gaps=True,
    )
    dt_bt = _t.perf_counter() - t0_bt
    bt_lbp = int(bt_stats.get("pruned_lb_point_pairs", 0))
    bt_dmp = int(bt_stats.get("pruned_dom_point_pairs", 0))
    bt_prn = int(bt_stats.get("pruned_point_pairs", bt_lbp + bt_dmp))
    bt_expl = int(bt_stats.get("explored_point_pairs", 0))
    bt_tot = int(bt_stats.get("total_point_pairs", 0))
    logging.info(
        _fmt_trace_row(
            "ball",
            dur_s=dt_bt,
            evals=int(bt_stats.get("objective_evals", 0)),
            best=bt_best,
            pr_lb=bt_lbp,
            pr_dom=bt_dmp,
            pr_tot=bt_prn,
            explored=bt_expl,
            total=bt_tot,
        )
    )

    def to_A(trace: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        tb = np.array(trace["time_best"], dtype=float)
        cb = np.array(trace["calls_best"], dtype=float)
        # Threshold-based normalization: A=1 when best <= tau
        A_t = np.minimum(1.0, (tau_tr + eps) / (tb + eps))
        A_m = np.minimum(1.0, (tau_tr + eps) / (cb + eps))
        return A_t, A_m

    kd_A_t, kd_A_m = to_A(kd_stats["trace"])  # type: ignore
    bt_A_t, bt_A_m = to_A(bt_stats["trace"])  # type: ignore

    # Heap size traces at calls checkpoints (max heap size up to each m/Pmax)
    def to_H(trace: Dict[str, Any]) -> np.ndarray:
        return np.array(trace.get("calls_heap_max", []), dtype=float)

    kd_H_m = to_H(kd_stats["trace"])  # type: ignore
    bt_H_m = to_H(bt_stats["trace"])  # type: ignore

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
                A_t[ti] = min(1.0, (tau_tr + eps) / (best + eps))
                ti += 1
            while ci < len(calls_grid) and calls >= int(calls_grid[ci]):
                A_m[ci] = min(1.0, (tau_tr + eps) / (best + eps))
                ci += 1
            if calls >= (n * (n - 1) // 2) and ti >= len(time_grid):
                break
        # Pad remaining
        while ti < len(time_grid):
            A_t[ti] = min(1.0, (tau_tr + eps) / (best + eps))
            ti += 1
        while ci < len(calls_grid):
            A_m[ci] = min(1.0, (tau_tr + eps) / (best + eps))
            ci += 1
        return A_t, A_m

    rnd_A_t, rnd_A_m = random_anytime(X, wc, time_grid, calls_grid, eps, rng, mode=random_pair_mode)

    return {
        "kd": MethodResult(A_time=kd_A_t, A_calls=kd_A_m, bound_gaps=np.array(kd_stats["trace"]["bound_gaps"], dtype=float), heap_calls=kd_H_m),  # type: ignore
        "bt": MethodResult(A_time=bt_A_t, A_calls=bt_A_m, bound_gaps=np.array(bt_stats["trace"]["bound_gaps"], dtype=float), heap_calls=bt_H_m),  # type: ignore
        "rnd": MethodResult(A_time=rnd_A_t, A_calls=rnd_A_m, bound_gaps=np.zeros(0), heap_calls=np.zeros_like(bt_H_m)),
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
        # Optional heap calls curve (only meaningful for BnB methods)
        try:
            H_m = np.stack([r[m].heap_calls for r in run_results], axis=0)
            out[m]["heap_calls"] = _bootstrap_ci(H_m, n_bootstrap, ci_level)
        except Exception:
            pass
    return out
