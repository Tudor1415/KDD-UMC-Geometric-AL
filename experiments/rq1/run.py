"""RQ1 experiment runner: traces, aggregation, and plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
import logging
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import warnings
import matplotlib.pyplot as plt

from .eval import aggregate_runs, evaluate_dataset, _time_norm_header
from .plots import CurveCI, plot_anytime_curves, plot_bound_tightness_kde, plot_scaling_bars, plot_scaling_lines, plot_heap_curves
from gal.utils.helpers import augment_with_minimums
from gal.trees import kd_tree as kd
from gal import trees as bt
from types import SimpleNamespace


def _eval_center_runs(task: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Worker: run n_runs repeats for a single center and return serialized results.

    Returns (runs_serialized_list, evals_serialized_list) where each element corresponds
    to one run. The evals are kept as simple dicts mirroring arrays to avoid
    cross-process class pickling surprises.
    """
    # Local imports here were originally used to avoid cross-process pickling
    # surprises. Since this module already imports numpy as np and
    # evaluate_dataset at the top level, we reuse those to avoid redundant
    # imports inside each worker call.

    # Load data array: prefer memory-mapped path to avoid large pickles
    if "X_path" in task and task["X_path"] is not None:
        X = np.load(task["X_path"], mmap_mode="r")
    else:
        X = task["X"]
    center = task["center"]
    n_runs = int(task["n_runs"])  # repeats per center
    kd_strategy = task["kd_strategy"]
    bt_strategy = task["bt_strategy"]
    bt_build_method = task["bt_build_method"]
    kd_cfg = task["kd_cfg"]
    bt_cfg = task["bt_cfg"]
    t_fracs = task["t_fracs"]
    c_fracs = task["c_fracs"]
    timing_repeats = int(task["timing_repeats"])
    eps = float(task["eps"])
    seed_base = int(task["seed_base"])
    rnd_mode = task["rnd_mode"]
    rnd_enabled = bool(task.get("rnd_enabled", True))
    kd_enabled = bool(task.get("kd_enabled", True))
    bt_enabled = bool(task.get("bt_enabled", True))
    # Build trees once per worker to avoid serializing heavy objects across processes
    kd_tree_obj = task.get("kd_tree_obj")
    bt_tree_obj = task.get("bt_tree_obj")
    if kd_tree_obj is None:
        kd_tree_obj = kd.build_tree(X, None if kd_cfg is None else dict(kd_cfg))
    if bt_tree_obj is None:
        bt_tree_obj = bt.build_tree(X, None if bt_cfg is None else dict(bt_cfg), method=str(bt_build_method))
    trace_tau = float(task["trace_tau"]) if "trace_tau" in task else None
    trace_certify = bool(task.get("trace_certify", False))

    out_runs: List[Dict[str, Any]] = []
    out_evals: List[Dict[str, Any]] = []
    # Per-run cap on bound gap values to ship back to the parent process
    gap_cap_per_run = int(task.get("gap_cap_per_run", 0) or 0)

    def _sample_array(arr: np.ndarray, cap: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=float).ravel()
        if cap <= 0 or arr.size <= cap:
            return arr
        idx = np.random.default_rng(int(seed_base)).choice(arr.size, size=cap, replace=False)
        return arr[idx]
    for run_i in range(n_runs):
        rng_rnd = np.random.default_rng(seed_base + 1_000 * run_i)
        res = evaluate_dataset(
            X,
            kd_strategy_name=kd_strategy,
            bt_strategy_name=bt_strategy,
            bt_build_method=bt_build_method,
            kd_config=kd_cfg,
            bt_config=bt_cfg,
            time_fracs=t_fracs,
            call_fracs=c_fracs,
            timing_repeats=timing_repeats,
            eps=eps,
            rng=rng_rnd,
            center=center,
            random_pair_mode=rnd_mode,
            random_enabled=rnd_enabled,
            kd_enabled=kd_enabled,
            bt_enabled=bt_enabled,
            kd_tree_obj=kd_tree_obj,
            bt_tree_obj=bt_tree_obj,
            trace_tau=trace_tau,
            trace_certify=trace_certify,
        )
        # Downsample bound gaps before sending results across process boundaries to
        # avoid heavy pickling and long waits after worker completion.
        res_serialized: Dict[str, Dict[str, Any]] = {}
        res_evals: Dict[str, Dict[str, Any]] = {}
        for k, v in res.items():
            if k in ("kd", "bt"):
                gaps_sampled = _sample_array(v.bound_gaps, gap_cap_per_run)
            else:
                gaps_sampled = np.asarray([], dtype=float)
            res_serialized[k] = {
                "A_time": v.A_time.tolist(),
                "A_calls": v.A_calls.tolist(),
                "bound_gaps": gaps_sampled.tolist(),
                "heap_calls": v.heap_calls.tolist(),
            }
            # Only send what's needed for aggregation (A_time, A_calls, heap_calls)
            res_evals[k] = {"A_time": v.A_time, "A_calls": v.A_calls, "heap_calls": v.heap_calls}
        out_runs.append(res_serialized)
        out_evals.append(res_evals)
    return out_runs, out_evals


@dataclass
class Rq1Config:
    content: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Rq1Config":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return Rq1Config(content=dict(cfg))

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.content
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def run_dataset(
    cfg: Rq1Config,
    *,
    dataset_name: str,
    X: np.ndarray,
    out_dir: Path,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    eps = float(cfg.get("global", "epsilon", default=1e-12))
    n_runs = int(cfg.get("global", "num_runs", default=5))
    # Always use a single timing repeat for T_max normalization
    timing_repeats = 1
    t_fracs: List[float] = list(cfg.get("budgets", "time_checkpoints", default=[0.05, 0.1, 0.2, 0.5, 1.0]))
    c_fracs: List[float] = list(cfg.get("budgets", "calls_checkpoints", default=[0.05, 0.1, 0.2, 0.5, 1.0]))
    kd_strategy = str(cfg.get("methods", "dual_kdtree_bnb", "strategy", default="lower_bound"))
    bt_strategy = str(cfg.get("methods", "balltree_bnb", "strategy", default="lower_bound"))
    kd_enabled = bool(cfg.get("methods", "dual_kdtree_bnb", "enabled", default=True))
    bt_enabled = bool(cfg.get("methods", "balltree_bnb", "enabled", default=True))
    bt_build_method_raw = str(cfg.get("methods", "balltree_bnb", "construction", default="disjoint_greedy"))
    # Normalize builder names (accept synonyms)
    _builder_alias = {
        "disjoint": "disjoint_greedy",
        "disjoint_greedy": "disjoint_greedy",
    }
    bt_build_method = _builder_alias.get(bt_build_method_raw, bt_build_method_raw)
    kd_cfg = cfg.get("methods", "dual_kdtree_bnb", default={}) or {}
    bt_cfg = cfg.get("methods", "balltree_bnb", default={}) or {}
    rnd_pair_mode = str(cfg.get("methods", "random_sampling", "pair_sampling", default="with_replacement"))
    rnd_enabled = bool(cfg.get("methods", "random_sampling", "enabled", default=True))

    # Plot/export style
    dpi = int(cfg.get("evaluation", "plot_style", "dpi", default=150))
    line_w = float(cfg.get("evaluation", "plot_style", "line_width", default=2.0))
    time_xscale = str(cfg.get("evaluation", "plot_style", "time_xscale", default="linear")).lower()
    calls_xscale = str(cfg.get("evaluation", "plot_style", "calls_xscale", default="linear")).lower()
    export_csv = bool(cfg.get("evaluation", "exports", "csv", default=False))
    export_json = bool(cfg.get("evaluation", "exports", "json", default=True))
    export_figs = list(cfg.get("evaluation", "exports", "figures", default=["png"]))
    # Hard-code scaling x-axis to linear (no config option)
    scaling_xscale = "linear"

    # Threshold mode support (optional)
    tau_conf = cfg.get("global", "tau", default=None)
    if tau_conf is None:
        raise ValueError("Missing required config: global.tau. Set a finite threshold (e.g., 1e-10).")
    try:
        tau_conf = float(tau_conf)
    except Exception as exc:
        raise ValueError(f"Invalid global.tau value: {tau_conf}") from exc
    if not np.isfinite(tau_conf):
        raise ValueError("global.tau must be finite; do not use infinity.")
    # Do not certify optimality in traces when a finite tau is provided
    trace_certify = False

    # Centers per dataset
    per_ds = int(cfg.get("centers", "per_dataset", default=1))
    # Additivity values
    add_vals = cfg.get("additivity", "values", default=[None])
    if not isinstance(add_vals, list):
        add_vals = [add_vals]

    def apply_additivity(Xin: np.ndarray, add: Any, all_vals: List[Any]) -> Tuple[np.ndarray, str, int, int]:
        """Apply Choquet additivity by augmenting with minimums up to order k.

        Rules:
        - If add is an int k>=1: X_aug = augment_with_minimums(Xin, k)
        - If add is a dict: supports keys {k, n}; slice rows to n (if given), then augment with k.
        - If add is None: no augmentation.
        Returns X_aug and a label including k and resulting dimensionality.
        """
        n0, d0 = Xin.shape
        # Defaults
        k = None
        n_rows = n0
        if isinstance(add, dict):
            if "n" in add:
                n_rows = max(1, min(int(add.get("n", n0)), n0))
            if "k" in add:
                k = int(add["k"])
        elif add is None:
            k = None
        else:
            # Treat scalar add as k-additivity
            try:
                k = int(add)
            except Exception:
                k = None

        Xsub = Xin[:n_rows, :]
        if k is None or k <= 1:
            Xaug = Xsub.copy()
            label = f"k=1 (n={Xsub.shape[0]}, d={Xsub.shape[1]})"
        else:
            Xaug = augment_with_minimums(Xsub, k)
            label = f"k={k} (n={Xaug.shape[0]}, d={Xaug.shape[1]})"
        return Xaug, label, Xaug.shape[0], Xaug.shape[1]

    figs_dir = out_dir / dataset_name
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate scaling across additivities
    scaling_categories: List[str] = []
    methods: List[str] = []
    if kd_enabled:
        methods.append("kd-tree BnB")
    if bt_enabled:
        methods.append("ball-tree BnB")
    if rnd_enabled:
        methods.append("Random Sampling")
    scaling_vals: List[List[float]] = []
    scaling_los: List[List[float]] = []
    scaling_his: List[List[float]] = []
    scaling_x: List[float] = []

    all_outputs: Dict[str, Any] = {"groups": []}

    logging.info(f"Begin dataset={dataset_name} | X.shape={X.shape} | centers={per_ds} | additivity={add_vals}")
    for add in add_vals:
        Xadd, add_label, n_sub, d_sub = apply_additivity(X, add, add_vals)
        group_dir = figs_dir / ("add_" + (str(add).replace(" ", "_") if add is not None else "base"))
        group_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"  Additivity group: {add_label} -> X_add.shape={Xadd.shape}")

        # Build trees once per (dataset, additivity) group; optionally in parallel
        build_parallel = bool(cfg.get("global", "parallel_tree_build", default=True))
        logging.info("  Building trees (kd-tree and ball-tree)…")
        t0 = time.perf_counter()
        kd_tree_obj = None
        bt_tree_obj = None
        kd_cfg_local = dict(kd_cfg)
        bt_cfg_local = dict(bt_cfg)

        def _build_kd():
            t = time.perf_counter()
            tree = kd.build_tree(Xadd, kd_cfg_local or None)
            return tree, time.perf_counter() - t

        def _build_bt():
            t = time.perf_counter()
            tree = bt.build_tree(Xadd, bt_cfg_local or None, method=bt_build_method)
            return tree, time.perf_counter() - t

        if build_parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {ex.submit(_build_kd): "kd", ex.submit(_build_bt): "bt"}
                kd_dur = bt_dur = None
                for fut in as_completed(futs):
                    tag = futs[fut]
                    tree, dur = fut.result()
                    if tag == "kd":
                        kd_tree_obj, kd_dur = tree, dur
                    else:
                        bt_tree_obj, bt_dur = tree, dur
            logging.info(f"    Trees built in {time.perf_counter()-t0:.2f}s (kd: {kd_dur:.2f}s, bt: {bt_dur:.2f}s)")
        else:
            kd_tree_obj, kd_dur = _build_kd()
            bt_tree_obj, bt_dur = _build_bt()
            logging.info(f"    Trees built sequentially (kd: {kd_dur:.2f}s, bt: {bt_dur:.2f}s)")

        logging.info("  Starting search phase…")

        # Sample centers once per dataset/add group
        centers = []
        for _ in range(per_ds):
            u = rng.random(d_sub)
            s = u.sum()
            centers.append((u / s) if s > 0 else np.ones(d_sub) / float(d_sub))
        logging.info(f"  Sampling centers: count={len(centers)}")

        runs_serialized: List[Dict[str, Any]] = []
        evals = []
        # Repeat runs per center (optionally parallel over centers)
        parallel_centers = bool(cfg.get("global", "parallel_centers", default=True))
        max_workers = int(cfg.get("global", "center_workers", default=os.cpu_count() or 1))
        # Bound-gap sampling budget: spread max_samples roughly evenly across all (center, run)
        max_gap_samples_total = int(cfg.get("evaluation", "bound_tightness", "max_samples", default=100000))
        per_run_gap_cap = int(math.ceil(max_gap_samples_total / max(1, per_ds * n_runs)))
        if parallel_centers and len(centers) > 1:
            n_workers = min(max_workers, len(centers))
            logging.info(f"  Launching {n_workers} worker(s) over {len(centers)} center(s)…")
            # Print header once before any worker logs, include tau from config
            _hdr = _time_norm_header(float(tau_conf))
            try:
                _h1, _h2 = _hdr.splitlines()
                logging.info(_h1)
                logging.info(_h2)
            except Exception:
                logging.info(_hdr)
            from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed
            tasks = []
            # Persist data once to disk for memory-mapped loading by workers
            mm_path = group_dir / "X_group.npy"
            if not mm_path.exists():
                np.save(mm_path, Xadd)
            logging.info(f"  Materialized group data to {mm_path} for worker memmap loading.")
            for ci, cvec in enumerate(centers):
                seed_base = int(cfg.get("global", "rng_seed_base", default=1729)) + 100_000 * ci
                tasks.append(dict(
                    X_path=str(mm_path),
                    center_idx=ci,
                    center=cvec,
                    n_runs=n_runs,
                    kd_strategy=kd_strategy,
                    bt_strategy=bt_strategy,
                    bt_build_method=bt_build_method,
                    kd_cfg=kd_cfg,
                    bt_cfg=bt_cfg,
                    kd_enabled=kd_enabled,
                    bt_enabled=bt_enabled,
                    t_fracs=t_fracs,
                    c_fracs=c_fracs,
                    timing_repeats=timing_repeats,
                    eps=eps,
                    seed_base=seed_base,
                    rnd_mode=rnd_pair_mode,
                    rnd_enabled=rnd_enabled,
                    # Trees are built in each worker to avoid heavy pickling
                    kd_tree_obj=None,
                    bt_tree_obj=None,
                    trace_tau=tau_conf,
                    trace_certify=trace_certify,
                    gap_cap_per_run=per_run_gap_cap,
                ))
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                starts: Dict[int, float] = {}
                fut_map = {}
                for t in tasks:
                    ci = int(t.get("center_idx", -1))
                    starts[ci] = time.perf_counter()
                    f = ex.submit(_eval_center_runs, t)
                    fut_map[f] = ci
                for fut in _as_completed(list(fut_map.keys())):
                    ci = fut_map[fut]
                    part_runs, part_evals = fut.result()
                    dt = time.perf_counter() - starts.get(ci, time.perf_counter())
                    logging.info(f"  Center {ci+1}/{len(centers)} finished in {dt:.2f}s; merging results…")
                    runs_serialized.extend(part_runs)
                    # Convert simple dict evals back to MethodResult-like objects for aggregation
                    for e in part_evals:
                        evals.append({k: SimpleNamespace(**v) for k, v in e.items()})
        else:
            # Serial path: print header once before first run
            _hdr = _time_norm_header(float(tau_conf))
            try:
                _h1, _h2 = _hdr.splitlines()
                logging.info(_h1)
                logging.info(_h2)
            except Exception:
                logging.info(_hdr)
            for ci, cvec in enumerate(centers):
                for run_i in range(n_runs):
                    # Use a deterministic RNG for random baseline fairness
                    rnd_seed = int(cfg.get("global", "rng_seed_base", default=1729))
                    rnd_seed = rnd_seed + 100_000 * ci + 1_000 * run_i
                    rng_rnd = np.random.default_rng(rnd_seed)
                    res = evaluate_dataset(
                        Xadd,
                        kd_strategy_name=kd_strategy,
                        bt_strategy_name=bt_strategy,
                        bt_build_method=bt_build_method,
                        kd_config=kd_cfg,
                        bt_config=bt_cfg,
                        time_fracs=t_fracs,
                        call_fracs=c_fracs,
                        timing_repeats=timing_repeats,
                        eps=eps,
                        rng=rng_rnd,  # only used by baseline or sampling
                        center=cvec,
                        random_pair_mode=rnd_pair_mode,
                        random_enabled=rnd_enabled,
                        kd_enabled=kd_enabled,
                        bt_enabled=bt_enabled,
                        kd_tree_obj=kd_tree_obj,
                        bt_tree_obj=bt_tree_obj,
                        trace_tau=tau_conf,
                        trace_certify=trace_certify,
                    )
                    # Apply the same per-run bound-gap cap for consistency with parallel path
                    def _sample_local(arr: np.ndarray, cap: int) -> np.ndarray:
                        arr = np.asarray(arr, dtype=float).ravel()
                        if cap <= 0 or arr.size <= cap:
                            return arr
                        idx = rng.integers(0, arr.size, size=cap, endpoint=False)
                        return arr[idx]
                    sampled_serialized: Dict[str, Dict[str, Any]] = {}
                    for k, v in res.items():
                        if k in ("kd", "bt"):
                            gaps = _sample_local(v.bound_gaps, per_run_gap_cap)
                        else:
                            gaps = np.asarray([], dtype=float)
                        sampled_serialized[k] = {
                            "A_time": v.A_time.tolist(),
                            "A_calls": v.A_calls.tolist(),
                            "bound_gaps": gaps.tolist(),
                            "heap_calls": v.heap_calls.tolist(),
                        }
                    runs_serialized.append(sampled_serialized)
                    evals.append(res)
        logging.info("  Aggregating runs and generating figures…")
        t_agg0 = time.perf_counter()

        # Aggregate using empirical CIs across runs (no resampling)
        agg = aggregate_runs(
            evals,
            ci_level=float(cfg.get("global", "ci_level", default=0.95)),
        )
        logging.info(f"    Aggregation completed in {time.perf_counter()-t_agg0:.2f}s")

        # Build curves for plotting
        t = np.array(t_fracs, dtype=float)
        c = np.array(c_fracs, dtype=float)
        has_kd = "kd" in agg
        has_bt = "bt" in agg
        if has_kd:
            kd_t_m, kd_t_lo, kd_t_hi = agg["kd"]["A_time"]
            kd_c_m, kd_c_lo, kd_c_hi = agg["kd"]["A_calls"]
        if has_bt:
            bt_t_m, bt_t_lo, bt_t_hi = agg["bt"]["A_time"]
            bt_c_m, bt_c_lo, bt_c_hi = agg["bt"]["A_calls"]
        # Heap size vs calls (BnB methods only)
        kd_h_m = kd_h_lo = kd_h_hi = None
        bt_h_m = bt_h_lo = bt_h_hi = None
        if has_kd and ("heap_calls" in agg["kd"]):
            kd_h_m, kd_h_lo, kd_h_hi = agg["kd"]["heap_calls"]
        if has_bt and ("heap_calls" in agg["bt"]):
            bt_h_m, bt_h_lo, bt_h_hi = agg["bt"]["heap_calls"]
        has_rnd = "rnd" in agg
        if has_rnd:
            rnd_t_m, rnd_t_lo, rnd_t_hi = agg["rnd"]["A_time"]
            rnd_c_m, rnd_c_lo, rnd_c_hi = agg["rnd"]["A_calls"]

        curves_t: Dict[str, CurveCI] = {}
        curves_c: Dict[str, CurveCI] = {}
        if has_kd:
            curves_t["kd-tree BnB"] = CurveCI(x=t, median=kd_t_m, low=kd_t_lo, high=kd_t_hi)  # type: ignore
            curves_c["kd-tree BnB"] = CurveCI(x=c, median=kd_c_m, low=kd_c_lo, high=kd_c_hi)  # type: ignore
        if has_bt:
            curves_t["ball-tree BnB"] = CurveCI(x=t, median=bt_t_m, low=bt_t_lo, high=bt_t_hi)  # type: ignore
            curves_c["ball-tree BnB"] = CurveCI(x=c, median=bt_c_m, low=bt_c_lo, high=bt_c_hi)  # type: ignore
        if has_rnd:
            curves_t["Random Sampling"] = CurveCI(x=t, median=rnd_t_m, low=rnd_t_lo, high=rnd_t_hi)
            curves_c["Random Sampling"] = CurveCI(x=c, median=rnd_c_m, low=rnd_c_lo, high=rnd_c_hi)

        # Figures per group
        t_fig0 = time.perf_counter()
        fig1 = plot_anytime_curves(curves_t, xlabel="Normalized Wall-Clock Time (t/T_max)", ylabel="Anytime Performance (A@t)", title=f"A@time on {dataset_name} ({add_label})", line_width=line_w, xscale=time_xscale)
        if "png" in export_figs:
            fig1.savefig(group_dir / "A_at_time.png", dpi=dpi)
        if "pdf" in export_figs:
            fig1.savefig(group_dir / "A_at_time.pdf", dpi=dpi)
        plt.close(fig1)
        fig2 = plot_anytime_curves(curves_c, xlabel="Normalized Objective Calls (m/P_max)", ylabel="Anytime Performance (A@m)", title=f"A@calls on {dataset_name} ({add_label})", line_width=line_w, xscale=calls_xscale)
        if "png" in export_figs:
            fig2.savefig(group_dir / "A_at_calls.png", dpi=dpi)
        if "pdf" in export_figs:
            fig2.savefig(group_dir / "A_at_calls.pdf", dpi=dpi)
        plt.close(fig2)

        # Heap size @ calls (if available)
        if kd_h_m is not None and bt_h_m is not None:
            heap_curves = {
                "kd-tree BnB": CurveCI(x=c, median=kd_h_m, low=kd_h_lo, high=kd_h_hi),  # type: ignore
                "ball-tree BnB": CurveCI(x=c, median=bt_h_m, low=bt_h_lo, high=bt_h_hi),  # type: ignore
            }
            fig_h = plot_heap_curves(heap_curves, xlabel="Normalized Objective Calls (m/P_max)", ylabel="Max Heap Size", title=f"Max Heap Size @ Calls on {dataset_name} ({add_label})", line_width=line_w, xscale=calls_xscale)
            if "png" in export_figs:
                fig_h.savefig(group_dir / "heap_at_calls.png", dpi=dpi)
            if "pdf" in export_figs:
                fig_h.savefig(group_dir / "heap_at_calls.pdf", dpi=dpi)
            plt.close(fig_h)

        # Bound tightness KDE (sample to limit size)
        def _sample_array(arr: np.ndarray, max_samples: int) -> np.ndarray:
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size <= max_samples:
                return arr
            idx = rng.choice(arr.size, size=max_samples, replace=False)
            return arr[idx]

        max_gap_samples = int(cfg.get("evaluation", "bound_tightness", "max_samples", default=100000))
        kd_all = np.concatenate([np.asarray(r["kd"]["bound_gaps"], dtype=float) for r in runs_serialized if "kd" in r]) if runs_serialized and has_kd else np.zeros(0)
        bt_all = np.concatenate([np.asarray(r["bt"]["bound_gaps"], dtype=float) for r in runs_serialized if "bt" in r]) if runs_serialized and has_bt else np.zeros(0)
        gaps: Dict[str, np.ndarray] = {}
        if has_kd:
            gaps_kd = _sample_array(kd_all, max_gap_samples)
            gaps["kd-tree Bounds"] = gaps_kd
        if has_bt:
            gaps_bt = _sample_array(bt_all, max_gap_samples)
            gaps["ball-tree Bounds"] = gaps_bt
        if gaps:
            logging.info("    Bound gap samples: " + ", ".join([f"{k.split()[0].lower()}={v.size}" for k, v in gaps.items()]) + f" (cap={max_gap_samples})")
            fig3 = plot_bound_tightness_kde(gaps, title=f"Bound Tightness on {dataset_name} ({add_label})")
            if "png" in export_figs:
                fig3.savefig(group_dir / "bound_tightness.png", dpi=dpi)
            if "pdf" in export_figs:
                fig3.savefig(group_dir / "bound_tightness.pdf", dpi=dpi)
            plt.close(fig3)

        # Scaling pick: A@t at 0.2 T_max
        def pick_at(fracs: List[float], med: np.ndarray, lo: np.ndarray, hi: np.ndarray, f: float = 0.2) -> tuple[float, float, float]:
            idx = int(np.argmin(np.abs(np.array(fracs) - f)))
            return float(med[idx]), float(lo[idx]), float(hi[idx])

        vals_row: List[float] = []
        los_row: List[float] = []
        his_row: List[float] = []
        if has_kd:
            kd_v, kd_l, kd_h = pick_at(t_fracs, kd_t_m, kd_t_lo, kd_t_hi)  # type: ignore
            vals_row.append(kd_v); los_row.append(kd_l); his_row.append(kd_h)
        if has_bt:
            bt_v, bt_l, bt_h = pick_at(t_fracs, bt_t_m, bt_t_lo, bt_t_hi)  # type: ignore
            vals_row.append(bt_v); los_row.append(bt_l); his_row.append(bt_h)
        scaling_categories.append(f"{dataset_name} (n={n_sub}, d={d_sub})")
        if has_rnd:
            rnd_v, rnd_l, rnd_h = pick_at(t_fracs, rnd_t_m, rnd_t_lo, rnd_t_hi)
            vals_row.append(rnd_v); los_row.append(rnd_l); his_row.append(rnd_h)
        else:
            pass
        scaling_vals.append(vals_row)
        scaling_los.append(los_row)
        scaling_his.append(his_row)
        # Use dimension (after augmentation) on the x-axis
        scaling_x.append(float(d_sub))

        logging.info(f"    Figure generation took {time.perf_counter()-t_fig0:.2f}s")
        # Optional CSV/JSON exports of curves per group
        if export_csv:
            import csv
            with (group_dir / "curves_time.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "x", "median", "low", "high"])
                for mname, ci in curves_t.items():
                    for xi, med, lo, hi in zip(ci.x, ci.median, ci.low, ci.high):
                        w.writerow([mname, float(xi), float(med), float(lo), float(hi)])
            with (group_dir / "curves_calls.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "x", "median", "low", "high"])
                for mname, ci in curves_c.items():
                    for xi, med, lo, hi in zip(ci.x, ci.median, ci.low, ci.high):
                        w.writerow([mname, float(xi), float(med), float(lo), float(hi)])
        # Save raw traces: prefer compact NPZ if configured
        export_npz = bool(cfg.get("evaluation", "exports", "npz", default=False))
        if export_npz:
            # Pack curves and sampled gaps for compact storage
            try:
                payload: Dict[str, np.ndarray] = {}
                if has_kd:
                    payload["kd_A_time"] = np.stack([np.asarray(run["kd"]["A_time"], dtype=float) for run in runs_serialized if "kd" in run], axis=0) if runs_serialized else np.zeros((0, len(t_fracs)))
                    payload["kd_A_calls"] = np.stack([np.asarray(run["kd"]["A_calls"], dtype=float) for run in runs_serialized if "kd" in run], axis=0) if runs_serialized else np.zeros((0, len(c_fracs)))
                    payload["kd_heap_calls"] = np.stack([np.asarray(run["kd"].get("heap_calls", np.zeros(len(c_fracs))), dtype=float) for run in runs_serialized if "kd" in run], axis=0) if runs_serialized else np.zeros((0, len(c_fracs)))
                    if 'gaps_kd' in locals():
                        payload["kd_gaps"] = gaps_kd  # type: ignore
                if has_bt:
                    payload["bt_A_time"] = np.stack([np.asarray(run["bt"]["A_time"], dtype=float) for run in runs_serialized if "bt" in run], axis=0) if runs_serialized else np.zeros((0, len(t_fracs)))
                    payload["bt_A_calls"] = np.stack([np.asarray(run["bt"]["A_calls"], dtype=float) for run in runs_serialized if "bt" in run], axis=0) if runs_serialized else np.zeros((0, len(c_fracs)))
                    payload["bt_heap_calls"] = np.stack([np.asarray(run["bt"].get("heap_calls", np.zeros(len(c_fracs))), dtype=float) for run in runs_serialized if "bt" in run], axis=0) if runs_serialized else np.zeros((0, len(c_fracs)))
                    if 'gaps_bt' in locals():
                        payload["bt_gaps"] = gaps_bt  # type: ignore
                np.savez_compressed(group_dir / "runs.npz", **payload)
            except Exception as _e:
                logging.warning(f"Failed to save NPZ traces in {group_dir}: {_e}")
        elif export_json:
            (group_dir / "runs.json").write_text(json.dumps(runs_serialized, indent=2))

        all_outputs["groups"].append({
            "label": add_label,
            "dir": str(group_dir),
        })

    # Global scaling plot over categories for this dataset
    vals = np.array(scaling_vals, dtype=float)
    los = np.array(scaling_los, dtype=float)
    his = np.array(scaling_his, dtype=float)
    x_arr = np.asarray(scaling_x, dtype=float)
    fig_scale = plot_scaling_lines(x_arr, methods, vals, los, his, title=f"Scaling on {dataset_name}", xlabel="d", xscale=scaling_xscale)
    if "png" in export_figs:
        fig_scale.savefig(figs_dir / "scaling.png", dpi=dpi)
    if "pdf" in export_figs:
        fig_scale.savefig(figs_dir / "scaling.pdf", dpi=dpi)
    plt.close(fig_scale)

    # Optional CSV export of scaling panel
    if export_csv and scaling_categories:
        import csv
        with (figs_dir / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "method", "value", "ci_low", "ci_high"])
            for cat, row_v, row_l, row_h in zip(scaling_categories, vals, los, his):
                for mname, v, l, h in zip(methods, row_v, row_l, row_h):
                    w.writerow([cat, mname, float(v), float(l), float(h)])

    all_outputs["scaling"] = {
        "categories": scaling_categories,
        "methods": methods,
        "values": vals.tolist(),
        "lower": los.tolist(),
        "upper": his.tolist(),
        "x": x_arr.tolist(),
        "dir": str(figs_dir),
    }
    logging.info(f"Done dataset={dataset_name}. Outputs in {figs_dir}")
    return all_outputs


def main(config_path: str) -> None:  # pragma: no cover - convenience entry
    cfg = Rq1Config.load(config_path)
    # Configure logging
    log_level_str = str(cfg.get("global", "log_level", default="INFO")).upper()
    level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    # Quiet numexpr warnings by setting max threads early
    try:
        ne_threads = int(cfg.get("global", "numexpr_max_threads", default=32))
        if os.environ.get("NUMEXPR_MAX_THREADS") is None:
            os.environ["NUMEXPR_MAX_THREADS"] = str(ne_threads)
    except Exception:
        pass
    # Filter pandas' numexpr version warning if present
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"Pandas requires version '2\.7\.3' or newer of 'numexpr'",
            category=UserWarning,
        )
    except Exception:
        pass
    rng = np.random.default_rng(int(cfg.get("global", "rng_seed_base", default=1729)))
    out = Path(cfg.get("global", "output_dir", default="./results/rq1"))
    out.mkdir(parents=True, exist_ok=True)
    datasets = cfg.get("datasets", default=[{"name": "TOY", "paths": {"matrix_npy": None}}])
    global_scaling_cats: List[str] = []
    global_scaling_vals: List[List[float]] = []
    global_scaling_los: List[List[float]] = []
    global_scaling_his: List[List[float]] = []
    global_scaling_x: List[float] = []
    global_methods: List[str] | None = None

    logging.info("Starting RQ1 experiment…")
    # Helper: load dataset matrix X from configured paths (prefer mined_rules CSV)
    def _load_points_for_dataset(name: str, ds_entry: Dict[str, Any]) -> np.ndarray:
        paths = ds_entry.get("paths", {}) if isinstance(ds_entry, dict) else {}
        cols = [
            "supportY","supportZ","support","confidence","lift","cosine",
            "phi","kruskal","yuleQ","added_value","certainty","revsupport",
        ]

        # 1) Try explicit mined_rules CSV path
        mnr_path = paths.get("mnr_rules")
        if mnr_path is None:
            # 2) Derive default: mined_rules/<lower>_mnr.csv
            derived = Path("mined_rules") / f"{name.lower()}_mnr.csv"
            if derived.exists():
                mnr_path = str(derived)
        if mnr_path is not None and Path(mnr_path).exists():
            logging.info(f"Loading mined rule features from {mnr_path} (dataset={name})…")
            # Prefer pandas if available for speed
            try:
                import pandas as pd  # type: ignore
                df = pd.read_csv(mnr_path, usecols=cols)
                X = df.to_numpy(dtype=float, copy=False)
                return np.ascontiguousarray(X, dtype=float)
            except Exception:  # fallback to csv reader
                import csv
                with open(mnr_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        raise RuntimeError(f"Empty CSV: {mnr_path}")
                    # Map column names to indices
                    idx_map: Dict[str, int] = {h.strip(): i for i, h in enumerate(header)}
                    use_idx = []
                    for c in cols:
                        if c not in idx_map:
                            raise RuntimeError(f"Column '{c}' not found in {mnr_path}")
                        use_idx.append(idx_map[c])
                    rows: List[List[float]] = []
                    for row in reader:
                        try:
                            rows.append([float(row[i]) for i in use_idx])
                        except Exception:
                            continue  # skip malformed rows
                X = np.asarray(rows, dtype=float)
                return np.ascontiguousarray(X, dtype=float)

        # 3) Fallback to matrix_npy if provided
        npy_path = paths.get("matrix_npy")
        if npy_path is not None and Path(npy_path).exists():
            logging.info(f"Loading matrix from {npy_path} (dataset={name})…")
            return np.ascontiguousarray(np.load(npy_path), dtype=float)

        # 4) Last resort: synthetic
        logging.warning(f"Dataset {name}: no mined_rules CSV or matrix_npy found. Using synthetic data.")
        rng_local = np.random.default_rng(int(cfg.get("global", "rng_seed_base", default=1729)))
        return rng_local.normal(size=(256, 12))

    for ds in datasets:
        name = ds.get("name", "DATA")
        X = _load_points_for_dataset(name, ds)
        # Uniformly subsample to maximum points if requested
        try:
            max_pts = int(cfg.get("global", "max_points", default=0))
        except Exception:
            max_pts = 0
        if max_pts and X.shape[0] > max_pts:
            logging.info(f"Dataset {name}: downsampling from {X.shape[0]} to {max_pts} points uniformly…")
            idx = rng.choice(X.shape[0], size=max_pts, replace=False)
            X = np.ascontiguousarray(X[idx], dtype=float)
        # Touch artifacts (optional)
        mnr_path = ds.get("paths", {}).get("mnr_rules")
        tx_path = ds.get("paths", {}).get("transactions_csv")
        meta: Dict[str, Any] = {"dataset": name, "artifacts": {}}
        if mnr_path and Path(mnr_path).exists():
            meta["artifacts"]["mnr_rules"] = str(Path(mnr_path).resolve())
        if tx_path and Path(tx_path).exists():
            meta["artifacts"]["transactions_csv"] = str(Path(tx_path).resolve())

        # Apply per-dataset overrides if present
        ds_cfg = copy.deepcopy(cfg.content)
        if ds.get("centers_override") is not None:
            ds_cfg.setdefault("centers", {})
            ds_cfg["centers"]["per_dataset"] = ds["centers_override"]
            logging.info(f"Dataset {name}: centers_override -> {ds.get('centers_override')}")
        if ds.get("additivity_override") is not None:
            ds_cfg.setdefault("additivity", {})
            ds_cfg["additivity"]["values"] = ds["additivity_override"]
            logging.info(f"Dataset {name}: additivity_override -> {ds.get('additivity_override')}")

        res = run_dataset(Rq1Config(content=ds_cfg), dataset_name=name, X=X, out_dir=out, rng=rng)
        # Accumulate for global scaling
        sc = res.get("scaling", {})
        cats = sc.get("categories", []) or []
        methods = sc.get("methods") or ["kd-tree BnB", "ball-tree BnB", "Random Sampling"]
        values = sc.get("values", []) or []
        lowers = sc.get("lower", []) or []
        uppers = sc.get("upper", []) or []
        xs = sc.get("x", []) or []
        if global_methods is None:
            global_methods = methods
        for cat, row_v, row_l, row_h, x in zip(cats, values, lowers, uppers, xs):
            global_scaling_cats.append(f"{name} — {cat}")
            global_scaling_vals.append([float(x) for x in row_v])
            global_scaling_los.append([float(x) for x in row_l])
            global_scaling_his.append([float(x) for x in row_h])
            global_scaling_x.append(float(x))
        summary = {"meta": meta, "outputs": res}
        (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))

    # Global scaling across datasets if available
    if global_scaling_cats:
        methods = global_methods or ["kd-tree BnB", "ball-tree BnB", "Random Sampling"]
        vals = np.array(global_scaling_vals, dtype=float)
        los = np.array(global_scaling_los, dtype=float)
        his = np.array(global_scaling_his, dtype=float)
        x_arr = np.asarray(global_scaling_x, dtype=float)
        fig = plot_scaling_lines(x_arr, methods, vals, los, his, title="Scaling Across Datasets", xlabel="d", xscale="linear")
        if cfg.get("evaluation", "exports", "figures", default=["png"]) and ("png" in cfg.get("evaluation", "exports", "figures", default=["png"])):
            fig.savefig(out / "scaling_all.png", dpi=int(cfg.get("evaluation", "plot_style", "dpi", default=150)))
        if cfg.get("evaluation", "exports", "figures", default=["png"]) and ("pdf" in cfg.get("evaluation", "exports", "figures", default=["png"])):
            fig.savefig(out / "scaling_all.pdf", dpi=int(cfg.get("evaluation", "plot_style", "dpi", default=150)))
        plt.close(fig)
        # Optional CSV
        if bool(cfg.get("evaluation", "exports", "csv", default=False)):
            import csv
            with (out / "scaling_all.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["x", "category", "method", "value", "ci_low", "ci_high"])
                for cat, row_v, row_l, row_h, x in zip(global_scaling_cats, vals, los, his, x_arr):
                    for mname, v, l, h in zip(methods, row_v, row_l, row_h):
                        w.writerow([float(x), cat, mname, float(v), float(l), float(h)])
    logging.info("RQ1 experiment completed.")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run RQ1 experiment")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
