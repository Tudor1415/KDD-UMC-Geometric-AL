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
import warnings
import matplotlib.pyplot as plt

from .eval import aggregate_runs, evaluate_dataset
from .plots import CurveCI, plot_anytime_curves, plot_bound_tightness_kde, plot_scaling_bars, plot_scaling_lines
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
    import numpy as _np
    from .eval import evaluate_dataset as _eval

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
    kd_tree_obj = task.get("kd_tree_obj")
    bt_tree_obj = task.get("bt_tree_obj")

    out_runs: List[Dict[str, Any]] = []
    out_evals: List[Dict[str, Any]] = []
    for run_i in range(n_runs):
        rng_rnd = _np.random.default_rng(seed_base + 1_000 * run_i)
        res = _eval(
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
            kd_tree_obj=kd_tree_obj,
            bt_tree_obj=bt_tree_obj,
        )
        out_runs.append({k: {"A_time": v.A_time.tolist(), "A_calls": v.A_calls.tolist(), "bound_gaps": v.bound_gaps.tolist()} for k, v in res.items()})
        out_evals.append({k: {"A_time": v.A_time, "A_calls": v.A_calls, "bound_gaps": v.bound_gaps} for k, v in res.items()})
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

    # Plot/export style
    dpi = int(cfg.get("evaluation", "plot_style", "dpi", default=150))
    line_w = float(cfg.get("evaluation", "plot_style", "line_width", default=2.0))
    time_xscale = str(cfg.get("evaluation", "plot_style", "time_xscale", default="linear")).lower()
    calls_xscale = str(cfg.get("evaluation", "plot_style", "calls_xscale", default="linear")).lower()
    export_csv = bool(cfg.get("evaluation", "exports", "csv", default=False))
    export_json = bool(cfg.get("evaluation", "exports", "json", default=True))
    export_figs = list(cfg.get("evaluation", "exports", "figures", default=["png"]))
    scaling_xscale = str(cfg.get("evaluation", "plot_style", "scaling_xscale", default="log")).lower()

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
    methods = ["kd-tree BnB", "ball-tree BnB", "Random Sampling"]
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
        if parallel_centers and len(centers) > 1:
            n_workers = min(max_workers, len(centers))
            logging.info(f"  Launching {n_workers} worker(s) over {len(centers)} center(s)…")
            from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed
            tasks = []
            for ci, cvec in enumerate(centers):
                seed_base = int(cfg.get("global", "rng_seed_base", default=1729)) + 100_000 * ci
                tasks.append(dict(
                    X=Xadd,
                    center=cvec,
                    n_runs=n_runs,
                    kd_strategy=kd_strategy,
                    bt_strategy=bt_strategy,
                    bt_build_method=bt_build_method,
                    kd_cfg=kd_cfg,
                    bt_cfg=bt_cfg,
                    t_fracs=t_fracs,
                    c_fracs=c_fracs,
                    timing_repeats=timing_repeats,
                    eps=eps,
                    seed_base=seed_base,
                    rnd_mode=rnd_pair_mode,
                    kd_tree_obj=kd_tree_obj,
                    bt_tree_obj=bt_tree_obj,
                ))
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_eval_center_runs, t) for t in tasks]
                for fut in _as_completed(futs):
                    part_runs, part_evals = fut.result()
                    runs_serialized.extend(part_runs)
                    # Convert simple dict evals back to MethodResult-like objects for aggregation
                    for e in part_evals:
                        evals.append({k: SimpleNamespace(**v) for k, v in e.items()})
        else:
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
                        kd_tree_obj=kd_tree_obj,
                        bt_tree_obj=bt_tree_obj,
                    )
                    runs_serialized.append({k: {"A_time": v.A_time.tolist(), "A_calls": v.A_calls.tolist(), "bound_gaps": v.bound_gaps.tolist()} for k, v in res.items()})
                    evals.append(res)
        logging.info("  Aggregating runs and generating figures…")

        agg = aggregate_runs(
            evals,
            n_bootstrap=int(cfg.get("global", "n_bootstrap", default=300)),
            ci_level=float(cfg.get("global", "ci_level", default=0.95)),
        )

        # Build curves for plotting
        t = np.array(t_fracs, dtype=float)
        c = np.array(c_fracs, dtype=float)
        kd_t_m, kd_t_lo, kd_t_hi = agg["kd"]["A_time"]
        kd_c_m, kd_c_lo, kd_c_hi = agg["kd"]["A_calls"]
        bt_t_m, bt_t_lo, bt_t_hi = agg["bt"]["A_time"]
        bt_c_m, bt_c_lo, bt_c_hi = agg["bt"]["A_calls"]
        rnd_t_m, rnd_t_lo, rnd_t_hi = agg["rnd"]["A_time"]
        rnd_c_m, rnd_c_lo, rnd_c_hi = agg["rnd"]["A_calls"]

        curves_t = {
            "kd-tree BnB": CurveCI(x=t, median=kd_t_m, low=kd_t_lo, high=kd_t_hi),
            "ball-tree BnB": CurveCI(x=t, median=bt_t_m, low=bt_t_lo, high=bt_t_hi),
            "Random Sampling": CurveCI(x=t, median=rnd_t_m, low=rnd_t_lo, high=rnd_t_hi),
        }
        curves_c = {
            "kd-tree BnB": CurveCI(x=c, median=kd_c_m, low=kd_c_lo, high=kd_c_hi),
            "ball-tree BnB": CurveCI(x=c, median=bt_c_m, low=bt_c_lo, high=bt_c_hi),
            "Random Sampling": CurveCI(x=c, median=rnd_c_m, low=rnd_c_lo, high=rnd_c_hi),
        }

        # Figures per group
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

        # Bound tightness KDE
        gaps_kd = np.concatenate([np.array(r["kd"]["bound_gaps"]) for r in runs_serialized])
        gaps_bt = np.concatenate([np.array(r["bt"]["bound_gaps"]) for r in runs_serialized])
        fig3 = plot_bound_tightness_kde({"kd-tree Bounds": gaps_kd, "ball-tree Bounds": gaps_bt}, title=f"Bound Tightness on {dataset_name} ({add_label})")
        if "png" in export_figs:
            fig3.savefig(group_dir / "bound_tightness.png", dpi=dpi)
        if "pdf" in export_figs:
            fig3.savefig(group_dir / "bound_tightness.pdf", dpi=dpi)
        plt.close(fig3)

        # Scaling pick: A@t at 0.2 T_max
        def pick_at(fracs: List[float], med: np.ndarray, lo: np.ndarray, hi: np.ndarray, f: float = 0.2) -> tuple[float, float, float]:
            idx = int(np.argmin(np.abs(np.array(fracs) - f)))
            return float(med[idx]), float(lo[idx]), float(hi[idx])

        kd_v, kd_l, kd_h = pick_at(t_fracs, kd_t_m, kd_t_lo, kd_t_hi)
        bt_v, bt_l, bt_h = pick_at(t_fracs, bt_t_m, bt_t_lo, bt_t_hi)
        rnd_v, rnd_l, rnd_h = pick_at(t_fracs, rnd_t_m, rnd_t_lo, rnd_t_hi)

        scaling_categories.append(f"{dataset_name} (n={n_sub}, d={d_sub})")
        scaling_vals.append([kd_v, bt_v, rnd_v])
        scaling_los.append([kd_l, bt_l, rnd_l])
        scaling_his.append([kd_h, bt_h, rnd_h])
        # Use dimension (after augmentation) on the x-axis
        scaling_x.append(float(d_sub))

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
        if export_json:
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
    fig_scale = plot_scaling_lines(x_arr, methods, vals, los, his, title=f"Scaling on {dataset_name}", xlabel="d (log)", xscale=scaling_xscale)
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
        fig = plot_scaling_lines(x_arr, methods, vals, los, his, title="Scaling Across Datasets", xlabel="d (log)", xscale=str(cfg.get("evaluation", "plot_style", "scaling_xscale", default="log")).lower())
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
