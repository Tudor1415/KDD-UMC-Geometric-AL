from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gal.trees import kd_tree as kd
from gal import trees as bt
from gal.search.engine import Search
from gal.search.kd_bounds import KdTreeBounds
from gal.search.strategies import get_strategy

from gal.oracles.linear import (
    get_oracle as get_linear_oracle,
    get_oracle_weights as get_linear_oracle_weights,
    PickledLinearOracle,
)
from gal.core.data import Dataset
from gal.oracles.oracles import (
    Oracle,
    ObjectiveMeasureOracle,
    SumOracle,
    SurpriseOracle,
    MDLOracle,
)

from .config import (
    ALConfig,
    _configure_runtime_from_config,
    _dataset_entry_from_cfg,
    _rand_uid,
    _sanitize_tag,
    _timestamp,
)
from .data import _load_points_for_dataset
from .space import CapacitySpace, _prepare_capacity_space
from .centers import _center_fn, _chebyshev_radius
from .export import _export_tree_h5


# ------------------------------ setup helpers ------------------------------ #

def _setup_logging(cfg: ALConfig) -> Tuple[logging.Logger, int]:
    level_name = str(cfg.get("logging", "level", default="INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__), log_level


def _setup_rng(cfg: ALConfig) -> np.random.Generator:
    return np.random.default_rng(int(cfg.get("global", "seed", default=1729)))


def _setup_dataset(cfg: ALConfig) -> Tuple[Dataset, np.ndarray, Dict[str, Any], str, str]:
    ds_entry = _dataset_entry_from_cfg(cfg)
    ds_name = str(ds_entry["name"])
    ds_paths = ds_entry.get("paths", {}) or {}
    rules_csv = ds_paths.get("dataset_path") or ds_paths.get("mnr_rules")
    if not rules_csv:
        raise ValueError("Missing rules CSV path: set paths.dataset_path or paths.mnr_rules in config.")
    tx_path = ds_paths.get("transactions_path") or ds_paths.get("transactions")
    irm_path = ds_paths.get("item_rule_map_path") or ds_paths.get("item_rule_map")
    ds = Dataset(
        dataset_path=rules_csv,
        transactions_path=tx_path,
        item_rule_map_path=irm_path,
        measures=ds_entry.get("measures"),
        name=ds_name,
    ).load()
    X = np.ascontiguousarray(ds.points, dtype=float)
    return ds, X, ds_entry, ds_name, rules_csv


def _setup_space(cfg: ALConfig, X: np.ndarray, log: logging.Logger) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray, Callable, str]:
    add_k_cfg = int(cfg.get("experiment", "additivity_k", default=1) or 1)
    X_aug, space, A0, b0 = _prepare_capacity_space(X, add_k=add_k_cfg, log=log)
    center_name = str(cfg.get("experiment", "center_name", default=cfg.get("global", "center", default="analytic")))
    center_fn = _center_fn(center_name)
    return X_aug, space, A0, b0, center_fn, center_name


def _setup_trees(cfg: ALConfig, X: np.ndarray, log: logging.Logger) -> Tuple[Any, str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    kd_cfg = cfg.get("trees", "kd", "config", default={}) or {}
    bt_cfg = cfg.get("trees", "ball", "config", default={}) or {}
    bt_method = str(cfg.get("trees", "ball", "method", default="")).strip()

    algo = cfg.get("algorithm_parameters", default=None) or {}
    if algo:
        leaf = algo.get("leaf_size")
        kd_leaf = leaf if leaf is not None else (algo.get("kd_tree", {}) or {}).get("leaf_size")
        bt_leaf = leaf if leaf is not None else (algo.get("ball_tree", {}) or {}).get("leaf_size")
        if kd_leaf is not None:
            kd_cfg.setdefault("leaf_size", int(kd_leaf))
        if bt_leaf is not None:
            bt_cfg.setdefault("leaf_size", int(bt_leaf))
        tbm = algo.get("tree_build_methods", {}) or {}
        b_raw = tbm.get("balltree") or tbm.get("ball_tree") or []
        if not bt_method and b_raw:
            bt_method = str(b_raw[0]) if isinstance(b_raw, (list, tuple)) else str(b_raw)

    kd_flag = cfg.get("trees", "kd", "enabled", default=None)
    bt_flag = cfg.get("trees", "ball", "enabled", default=None)
    if kd_flag is True and bt_flag is True:
        raise ValueError("Enable exactly one tree family: set only one of trees.kd.enabled or trees.ball.enabled to true.")
    if kd_flag is True:
        tree_family = "kdtree"
    elif bt_flag is True:
        tree_family = "balltree"
    else:
        tree_family = "balltree"  # default

    if tree_family == "kdtree":
        log.info("Building kd-tree with config: %s", kd_cfg)
        tree = kd.build_tree(X, kd_cfg)
        tree_method = "kd_tree"
    else:
        if not bt_method:
            bt_method = "two_pivot"
        log.info("Building ball-tree (method=%s) with config: %s", bt_method, bt_cfg)
        tree = bt.build_tree(X, bt_cfg, method=bt_method)
        tree_method = bt_method

    return tree, tree_family, tree_method, kd_cfg, bt_cfg, (algo or {})


def _setup_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    otype = str(cfg.get("oracle", "type", default="objective")).strip().lower()
    if otype in {"objective", "objective_measure", "measure"}:
        measure = cfg.get("oracle", "measure", default=None)
        if not measure:
            measure = ds.measures[0] if ds.measures else None
        if not measure:
            raise ValueError("Objective oracle requires a measure name.")
        oracle: Oracle = ObjectiveMeasureOracle(str(measure))
    elif otype == "sum":
        measures = cfg.get("oracle", "measures", default=None)
        if not measures:
            measures = list(ds.measures)
        oracle = SumOracle([str(m) for m in measures])
    elif otype == "surprise":
        ptype = str(cfg.get("oracle", "prior_type", default="independent"))
        prior_kwargs = cfg.get("oracle", "prior_kwargs", default={}) or {}
        oracle = SurpriseOracle(prior_type=ptype, **prior_kwargs)
    elif otype in {"mdl", "mdl_oracle"}:
        c0 = float(cfg.get("oracle", "c0", default=8.0))
        c_item = float(cfg.get("oracle", "c_item", default=4.0))
        oracle = MDLOracle(c0=c0, c_item=c_item)
    else:
        raise ValueError(f"Unsupported oracle type '{otype}'.")
    oracle.set_dataset(ds)
    return oracle


def _setup_run_params(cfg: ALConfig, algo: Dict[str, Any]) -> Tuple[int, float, float, bool, int, str]:
    n_iter = int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))
    tau_cap = float(cfg.get("experiment", "tau_max", default=1e-5))
    tau_multiplier_cfg = cfg.get("experiment", "tau_radius_multiplier", default=None)
    if tau_multiplier_cfg is None:
        tau_multiplier_cfg = cfg.get("algorithm_parameters", "tau_radius_multiplier", default=None)
    try:
        tau_multiplier = float(tau_multiplier_cfg)
    except (TypeError, ValueError):
        tau_multiplier = 0.5
    if tau_multiplier <= 0:
        tau_multiplier = 0.5
    collect_events = bool(cfg.get("logging", "search_events", default=False))
    log_every = int(cfg.get("logging", "log_every", default=10) or 10)
    search_strategy = str((algo or {}).get("search_strategy", "lower_bound"))
    return n_iter, tau_cap, tau_multiplier, collect_events, log_every, search_strategy


def _prepare_output_dir(
    *,
    cfg: ALConfig,
    ds_name: str,
    ds_entry: Dict[str, Any],
    rules_csv: str,
    oracle: Oracle,
    center_name: str,
    tree_family: str,
    tree_method: str,
    kd_cfg: Dict[str, Any],
    bt_cfg: Dict[str, Any],
    algo: Dict[str, Any],
    rng: np.random.Generator,
    d: int,
) -> Path:
    out_root = Path(cfg.get("global", "output_root", default="./results/al"))
    out_root.mkdir(parents=True, exist_ok=True)
    exp_uid = _rand_uid(rng)
    exp_dir = out_root / f"{ds_name}_{oracle.name}_{center_name}_{tree_family}-{tree_method}_{_timestamp()}_{exp_uid}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg_json = {
        "experiment_uid": exp_uid,
        "dataset_name": ds_name,
        "dataset_path": rules_csv,
        "oracle_name": str(oracle.name),
        "center_name": center_name,
        "random_seed": int(cfg.get("global", "seed", default=1729)),
        "active_learning_budget": int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25))),
        "algorithm_parameters": {
            "leaf_size": int((algo or {}).get("leaf_size", kd_cfg.get("leaf_size", bt_cfg.get("leaf_size", 25)))),
            "search_strategies": [str((algo or {}).get("search_strategy", "lower_bound"))],
            "tree_build_methods": ({"kdtree": ["kd_tree"]} if tree_family == "kdtree" else {"balltree": [str(tree_method)]}),
        },
    }
    if ds_entry.get("measures"):
        cfg_json["measures"] = list(ds_entry["measures"])
    (exp_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    return exp_dir


@dataclass
class ExperimentSetup:
    log: logging.Logger
    log_level: int
    rng: np.random.Generator
    ds: Dataset
    X: np.ndarray
    ds_entry: Dict[str, Any]
    ds_name: str
    rules_csv: str
    space: CapacitySpace
    A0: np.ndarray
    b0: np.ndarray
    center_fn: Callable
    center_name: str
    tree: Any
    tree_family: str
    tree_method: str
    kd_cfg: Dict[str, Any]
    bt_cfg: Dict[str, Any]
    algo: Dict[str, Any]
    oracle: Oracle
    n_iter: int
    tau_cap: float
    tau_multiplier: float
    collect_events: bool
    log_every: int
    search_strategy: str


def setup_experiment(cfg: ALConfig) -> ExperimentSetup:
    log, log_level = _setup_logging(cfg)
    _configure_runtime_from_config(cfg)
    rng = _setup_rng(cfg)
    ds, X, ds_entry, ds_name, rules_csv = _setup_dataset(cfg)
    max_pts = int(cfg.get("global", "max_points", default=0) or 0)
    if max_pts and X.shape[0] > max_pts:
        idx = rng.choice(X.shape[0], size=max_pts, replace=False)
        X = np.ascontiguousarray(X[idx], dtype=float)
    X, space, A0, b0, center_fn, center_name = _setup_space(cfg, X, log)
    tree, tree_family, tree_method, kd_cfg, bt_cfg, algo = _setup_trees(cfg, X, log)
    oracle = _setup_oracle(cfg, ds)
    n_iter, tau_cap, tau_multiplier, collect_events, log_every, search_strategy = _setup_run_params(cfg, algo)
    return ExperimentSetup(
        log=log,
        log_level=log_level,
        rng=rng,
        ds=ds,
        X=X,
        ds_entry=ds_entry,
        ds_name=ds_name,
        rules_csv=rules_csv,
        space=space,
        A0=A0,
        b0=b0,
        center_fn=center_fn,
        center_name=center_name,
        tree=tree,
        tree_family=tree_family,
        tree_method=tree_method,
        kd_cfg=kd_cfg,
        bt_cfg=bt_cfg,
        algo=algo,
        oracle=oracle,
        n_iter=n_iter,
        tau_cap=tau_cap,
        tau_multiplier=tau_multiplier,
        collect_events=collect_events,
        log_every=log_every,
        search_strategy=search_strategy,
    )


# --------------------------- iterative learning --------------------------- #

# ----------------------- learning loop helpers (NPZ I/O) ---------------------- #

def _init_streaming_outputs(exp_dir: Path) -> tuple[csv.writer, Any, Path]:
    """Initialize on-disk streaming outputs using NPZ files.

    - iterations.csv with a fixed schema
    - queries/ directory to store per-iteration query vectors as NPZ
    Returns (csv_writer, csv_file_handle, queries_dir)
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    it_csv = open(exp_dir / "iterations.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(it_csv)
    csv_writer.writerow(["iteration_id", "query_path", "oracle_response", "i", "j", "timestamp_start", "timestamp_end"])  # schema
    q_dir = exp_dir / "queries"
    q_dir.mkdir(parents=True, exist_ok=True)
    return csv_writer, it_csv, q_dir


def _ensure_search_engine(engine: Optional[Search], search_strategy: str, X: np.ndarray) -> Search:
    if engine is not None:
        return engine
    strat = get_strategy(search_strategy, queries=X)
    return Search(strategy=strat)


def _log_iteration(it: int, i: Optional[int], j: Optional[int], dist: Optional[float], radius: float, *, log_level: int, log_every: int, exp_dir: Path) -> None:
    if it == 0 or (log_level <= logging.DEBUG and (it % log_every == 0)):
        logging.getLogger(__name__).debug(
            "Iter %d: i=%s j=%s dist=%s radius=%.4f",
            it,
            str(i),
            str(j),
            "{:.4f}".format(float(dist)) if dist is not None else "nan",
            float(radius),
        )
    iter_dir = exp_dir / f"iteration_{it:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    return iter_dir


def _export_search_events_npz(path: Path, events: List[Dict[str, Any]]) -> None:
    """Export search events to NPZ for portability (no h5py).

    Stores arrays: event_type, node_id, parent_id, timestamp, lower_bound, upper_bound.
    """
    if not events:
        np.savez(
            path,
            event_type=np.array([], dtype=object),
            node_id=np.array([], dtype=np.int64),
            parent_id=np.array([], dtype=np.int64),
            timestamp=np.array([], dtype=float),
            lower_bound=np.array([], dtype=float),
            upper_bound=np.array([], dtype=float),
        )
        return
    ev_type = np.array([str(e.get("event_type", "")) for e in events], dtype=object)
    node_id = np.array([int(e.get("node_id", -1)) for e in events], dtype=np.int64)
    parent_id = np.array([int(e.get("parent_id", -1)) for e in events], dtype=np.int64)
    timestamp = np.array([float(e.get("timestamp", 0.0)) for e in events], dtype=float)
    lower = np.array([float(e.get("lower_bound", np.nan)) for e in events], dtype=float)
    upper = np.array([float(e.get("upper_bound", np.nan)) for e in events], dtype=float)
    np.savez(
        path,
        event_type=ev_type,
        node_id=node_id,
        parent_id=parent_id,
        timestamp=timestamp,
        lower_bound=lower,
        upper_bound=upper,
    )


def _record_query_npz(*, it: int, diff: np.ndarray, q_dir: Path, csv_writer: csv.writer, y: int, i: int, j: int, t_start: float) -> None:
    """Persist the iteration query vector and append a CSV row.

    Writes queries/query_###.npz with key 'vector' and logs row into iterations.csv.
    """
    q_path = q_dir / f"query_{it:03d}.npz"
    np.savez(q_path, vector=np.asarray(diff, dtype=float))
    t_end = time.time()
    csv_writer.writerow([
        it,
        f"queries/query_{it:03d}.npz:vector",
        int(y),
        int(i),
        int(j),
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t_start)) + f".{int((t_start%1)*1000):03d}Z",
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t_end)) + f".{int((t_end%1)*1000):03d}Z",
    ])


def _save_center_snapshot(iter_dir: Path, center_full: np.ndarray, radius: float, tau: float) -> None:
    np.save(iter_dir / "center_model.npy", center_full)
    try:
        np.savez(
            iter_dir / "center_model.npz",
            center=np.asarray(center_full, dtype=float),
            radius=float(radius),
            tau=float(tau),
        )
    except Exception:
        pass


def _finalize_version_space_npz(exp_dir: Path, A: np.ndarray, b: np.ndarray) -> None:
    np.savez(
        exp_dir / "final_version_space.npz",
        A=np.asarray(A, dtype=float),
        b=np.asarray(b, dtype=float).reshape(-1, 1),
    )


def learning_loop(
    *,
    tree: Any,
    X: np.ndarray,
    space: CapacitySpace,
    A0: np.ndarray,
    b0: np.ndarray,
    center_fn: Callable,
    n_iter: int,
    tau_cap: float,
    tau_multiplier: float,
    exp_dir: Path,
    oracle_compare: Callable[[np.ndarray, np.ndarray], int],
    collect_events: bool,
    log_every: int,
    log_level: int,
    search_strategy: str,
    engine: Optional[Search] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Outputs: iterations.csv and queries/ (NPZ-based, no h5py)
    csv_writer, it_csv, q_dir = _init_streaming_outputs(exp_dir)

    # Init version space
    A = np.asarray(A0, dtype=float).copy()
    b = np.asarray(b0, dtype=float).copy()
    center_proj = np.asarray(center_fn(A, b), dtype=float)
    center_full = space.expand_center(center_proj)
    radius = _chebyshev_radius(A, b, center_proj)

    engine = _ensure_search_engine(engine, search_strategy, X)

    for it in range(n_iter):
        t_start = time.time()
        if not (np.isfinite(radius) and radius > 0):
            break
        tau = min(radius * float(tau_multiplier), float(tau_cap))
        i, j, dist, stats = engine.search_pair(
            tree,
            X,
            center_full,
            tau=float(tau),
            return_stats=True,
            ensure_optimal=True,
            collect_events=collect_events,
        )

        iter_dir = _log_iteration(it, i, j, dist, float(radius), log_level=log_level, log_every=log_every, exp_dir=exp_dir)

        # Save search events (NPZ)
        if collect_events:
            events = list(stats.get("trace", {}).get("events", []))
            _export_search_events_npz(iter_dir / "search_trace.npz", events)

        if i is None or j is None:
            break
        
        q_a, q_b = X[int(i)], X[int(j)]
        diff = q_a - q_b
        
        y = oracle_compare(q_a, q_b)
        constraint = -float(y) * diff
        
        proj_row, proj_rhs = space.project(constraint)
        A = np.vstack([A, proj_row.reshape(1, -1)])
        b = np.concatenate([b, np.array([proj_rhs], dtype=float)])
        
        center_proj = np.asarray(center_fn(A, b), dtype=float)
        center_full = space.expand_center(center_proj)
        radius = _chebyshev_radius(A, b, center_proj)
        
        _record_query_npz(it=it, diff=diff, q_dir=q_dir, csv_writer=csv_writer, y=int(y), i=int(i), j=int(j), t_start=t_start)
        _save_center_snapshot(iter_dir, center_full, float(radius), float(tau))

    # Close CSV stream
    it_csv.close()

    # Final constraints snapshot (NPZ only)
    _finalize_version_space_npz(exp_dir, A, b)

    return A, b


## Deprecated single-run entry was removed. Use run_all(), or setup_experiment()+learning_loop().


def run_all(cfg: ALConfig) -> Path:
    # Configure logging
    level_name = str(cfg.get("logging", "level", default="INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = logging.getLogger(__name__)

    # Runtime knobs (threads, etc.) before dataset loading
    _configure_runtime_from_config(cfg)

    rng = np.random.default_rng(int(cfg.get("global", "seed", default=1729)))

    # Resolve dataset entries
    datasets_cfg = cfg.get("datasets", default=None)
    ds_entries: List[Dict[str, Any]] = []
    if datasets_cfg is not None:
        if not isinstance(datasets_cfg, (list, tuple)):
            raise ValueError("datasets must be a list when using the new schema.")
        for item in datasets_cfg:
            ds_entries.append(_dataset_entry_from_cfg(cfg, item))
    else:
        ds_entries.append(_dataset_entry_from_cfg(cfg))
    if not ds_entries:
        raise ValueError("No datasets resolved from configuration.")

    # Global algorithmic config reused across datasets
    algo = cfg.get("algorithm_parameters", default={}) or {}
    kd_cfg = cfg.get("trees", "kd", "config", default={}) or {}
    bt_cfg = cfg.get("trees", "ball", "config", default={}) or {}
    leaf = algo.get("leaf_size")
    if leaf is not None:
        kd_cfg.setdefault("leaf_size", int(leaf))
        bt_cfg.setdefault("leaf_size", int(leaf))
    tbm = (algo.get("tree_build_methods", {}) or {})
    kd_raw = tbm.get("kdtree")
    if kd_raw is None:
        kd_raw = tbm.get("kd_tree")
    if kd_raw is None:
        kd_methods = [] if tbm else ["kd_tree"]
    elif isinstance(kd_raw, (list, tuple)):
        kd_methods = [str(m) for m in kd_raw]
    else:
        kd_methods = [str(kd_raw)]
    bt_raw = tbm.get("balltree") or tbm.get("ball_tree") or []
    if isinstance(bt_raw, (list, tuple)):
        bt_methods = [str(m) for m in bt_raw]
    elif bt_raw:
        bt_methods = [str(bt_raw)]
    else:
        bt_methods = []
    strategies_raw = algo.get("search_strategies", None)
    if strategies_raw is None:
        search_strategies = [str(algo.get("search_strategy", "lower_bound"))]
    elif isinstance(strategies_raw, (list, tuple)):
        search_strategies = [str(s) for s in strategies_raw]
    else:
        search_strategies = [str(strategies_raw)]

    # Oracle names
    oracle_names_cfg = cfg.get("oracles", "names", default=None)
    if oracle_names_cfg:
        oracle_names = [str(x) for x in oracle_names_cfg]
    else:
        oracle_names = [
            str(
                cfg.get(
                    "experiment",
                    "oracle_name",
                    default=cfg.get("oracle", "type", default="linear_simplex"),
                )
            )
        ]

    center_name = str(
        cfg.get("experiment", "center_name", default=cfg.get("global", "center", default="analytic"))
    )
    center_fn = _center_fn(center_name)

    out_root = Path(cfg.get("global", "output_root", default="./results/al"))
    out_root.mkdir(parents=True, exist_ok=True)
    n_iter = int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))
    collect_events = bool(cfg.get("logging", "search_events", default=False))
    tau_multiplier_cfg = cfg.get("experiment", "tau_radius_multiplier", default=None)
    if tau_multiplier_cfg is None:
        tau_multiplier_cfg = cfg.get("algorithm_parameters", "tau_radius_multiplier", default=None)
    try:
        tau_multiplier = float(tau_multiplier_cfg)
    except (TypeError, ValueError):
        tau_multiplier = 0.5
    if tau_multiplier <= 0:
        tau_multiplier = 0.5
    timestamp = _timestamp()

    last_dir: Optional[Path] = None

    # Loop over datasets and spawn runs
    for ds_entry in ds_entries:
        ds_name = str(ds_entry.get("name", cfg.get("experiment", "dataset_name", default="DATA")))
        X = _load_points_for_dataset(ds_name, ds_entry)
        log.info("Loaded dataset '%s' with shape %s", ds_name, getattr(X, 'shape', None))
        # Optional uniform downsampling per dataset
        max_pts = int(cfg.get("global", "max_points", default=0) or 0)
        if max_pts and X.shape[0] > max_pts:
            idx = rng.choice(X.shape[0], size=max_pts, replace=False)
            X = np.ascontiguousarray(X[idx], dtype=float)
        # Feature augmentation via additivity_k and constraint initialization
        add_k_cfg = int(cfg.get("experiment", "additivity_k", default=1) or 1)
        X, space, A0, b0 = _prepare_capacity_space(X, add_k=add_k_cfg, log=log)

        # Build trees per dataset
        if kd_methods:
            log.info("Building kd-tree with config: %s", kd_cfg)
            kd_tree_obj = kd.build_tree(X, kd_cfg)
        else:
            kd_tree_obj = None
        log.info("Building ball-trees for methods: %s", ", ".join(map(str, bt_methods)) or "<none>")
        bt_trees = {str(m): bt.build_tree(X, bt_cfg, method=str(m)) for m in bt_methods}

        for oracle_name in oracle_names:
            oracle_fn = get_linear_oracle(oracle_name, X.shape[1], rng)
            oracle_w = get_linear_oracle_weights(oracle_name, X.shape[1])
            # KD-tree combinations
            if kd_tree_obj is not None and kd_methods:
                for strat_name in search_strategies:
                    engine = Search(bounder=KdTreeBounds(), strategy=get_strategy(strat_name, queries=X))
                    uid = _rand_uid(rng)
                    run_name = f"{ds_name}_kdtree-kd_tree_{_sanitize_tag(strat_name)}_{_sanitize_tag(oracle_name)}_{_sanitize_tag(center_name)}_{timestamp}_{uid}"
                    exp_dir = out_root / run_name
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    log.info(
                        "Start run: %s × strategy=%s × oracle=%s × center=%s",
                        "kdtree-kd_tree",
                        strat_name,
                        oracle_name,
                        center_name,
                    )
                    _run_single_experiment(
                        X=X,
                        tree_kd=kd_tree_obj,
                        tree_bt=None,
                        engine=engine,
                        exp_dir=exp_dir,
                        n_iter=n_iter,
                        center_name=center_name,
                        center_fn=center_fn,
                        oracle_fn=oracle_fn,
                        oracle_weights=oracle_w,
                        space=space,
                        A0=A0,
                        b0=b0,
                        kd_cfg=kd_cfg,
                        bt_cfg=bt_cfg,
                        tree_family="kdtree",
                        tree_method="kd_tree",
                        search_strategy=strat_name,
                        ds_entry=ds_entry,
                        cfg=cfg,
                        tau_multiplier=tau_multiplier,
                        collect_events=collect_events,
                    )
                    last_dir = exp_dir
            # Ball-tree combinations
            for bt_method_name, tree_obj in bt_trees.items():
                for strat_name in search_strategies:
                    engine = Search(strategy=get_strategy(strat_name, queries=X))
                    uid = _rand_uid(rng)
                    run_name = f"{ds_name}_balltree-{_sanitize_tag(bt_method_name)}_{_sanitize_tag(strat_name)}_{_sanitize_tag(oracle_name)}_{_sanitize_tag(center_name)}_{timestamp}_{uid}"
                    exp_dir = out_root / run_name
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    log.info(
                        "Start run: balltree-%s × strategy=%s × oracle=%s × center=%s",
                        bt_method_name,
                        strat_name,
                        oracle_name,
                        center_name,
                    )
                    _run_single_experiment(
                        X=X,
                        tree_kd=kd_tree_obj,
                        tree_bt=tree_obj,
                        engine=engine,
                        exp_dir=exp_dir,
                        n_iter=n_iter,
                        center_name=center_name,
                        center_fn=center_fn,
                        oracle_fn=oracle_fn,
                        oracle_weights=oracle_w,
                        space=space,
                        A0=A0,
                        b0=b0,
                        kd_cfg=kd_cfg,
                        bt_cfg=bt_cfg,
                        tree_family="balltree",
                        tree_method=str(bt_method_name),
                        search_strategy=strat_name,
                        ds_entry=ds_entry,
                        cfg=cfg,
                        tau_multiplier=tau_multiplier,
                        collect_events=collect_events,
                    )
                    last_dir = exp_dir

    return last_dir or out_root


def _run_single_experiment(
    *,
    X: np.ndarray,
    tree_kd: kd.GeometricTree | None,
    tree_bt: bt.GeometricTree | None,
    engine: Search,
    exp_dir: Path,
    n_iter: int,
    center_name: str,
    center_fn,
    oracle_fn,
    oracle_weights: np.ndarray,
    space: CapacitySpace,
    A0: np.ndarray,
    b0: np.ndarray,
    kd_cfg: Dict[str, Any],
    bt_cfg: Dict[str, Any],
    tree_family: str,
    tree_method: str,
    search_strategy: str,
    ds_entry: Dict[str, Any],
    cfg: ALConfig,
    tau_multiplier: float,
    collect_events: bool,
) -> None:
    # Tau cap (maximum tau per iteration)
    tau_cap = float(cfg.get("experiment", "tau_max", default=1e-5))

    # Serialize per-run config.json (no dataset hash)
    ds_paths = ds_entry.get("paths", {}) or {}
    dataset_path = ds_paths.get("dataset_path") or ds_paths.get("matrix_npy")
    cfg_json = {
        "dataset_name": str(ds_entry.get("name", cfg.get("experiment", "dataset_name", default="DATA"))),
        "dataset_path": dataset_path,
        "oracle_name": str(cfg.get("experiment", "oracle_name", default=cfg.get("oracle", "type", default="Linear"))),
        "center_name": str(center_name),
        "random_seed": int(cfg.get("global", "seed", default=1729)),
        "active_learning_budget": int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25))),
        "additivity_k": int(cfg.get("experiment", "additivity_k", default=1)),
        "tree_family": str(tree_family),
        "tree_method": str(tree_method),
        "search_strategy": str(search_strategy),
        "algorithm_parameters": {
            "leaf_size": int((cfg.get("algorithm_parameters", "leaf_size", default=0) or kd_cfg.get("leaf_size", bt_cfg.get("leaf_size", 25)))),
            "search_strategies": [str(search_strategy)],
            "tree_build_methods": {"kdtree": ["kd_tree"], "balltree": [str(tree_method)]},
        },
        "oracle_weights": [float(x) for x in np.asarray(oracle_weights, dtype=float).ravel().tolist()],
    }
    if ds_entry.get("measures"):
        cfg_json["measures"] = list(ds_entry["measures"])
    (exp_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    # Pickle a scoring-oracle object for reproducibility and analysis
    with open(exp_dir / "oracle.pkl", "wb") as f:
        obj = PickledLinearOracle(cfg_json.get("oracle_name"), np.asarray(oracle_weights, dtype=float))
        pickle.dump(obj, f)

    # Export trees used in this run
    _export_tree_h5(exp_dir / "tree.h5", tree_kd if tree_family == "kdtree" else None, tree_bt if tree_family == "balltree" else None, X.shape[1])

    # Choose the concrete tree and delegate to the shared learning loop
    tree = tree_bt if tree_family.startswith("ball") else tree_kd
    if tree is None:
        raise RuntimeError("No tree available for search.")

    learning_loop(
        tree=tree,
        X=X,
        space=space,
        A0=A0,
        b0=b0,
        center_fn=center_fn,
        n_iter=n_iter,
        tau_cap=tau_cap,
        tau_multiplier=tau_multiplier,
        exp_dir=exp_dir,
        oracle_compare=oracle_fn,
        collect_events=collect_events,
        log_every=int(cfg.get("logging", "log_every", default=10) or 10),
        log_level=getattr(logging, str(cfg.get("logging", "level", default="INFO")).upper(), logging.INFO),
        search_strategy=str(search_strategy),
        engine=engine,
    )


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Run active learning experiment (raw logs)")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level (e.g., DEBUG, INFO)")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N iterations at DEBUG level")
    args = parser.parse_args()
    cfg = ALConfig.load(args.config)
    if args.log_level:
        cfg.raw.setdefault("logging", {})["level"] = str(args.log_level)
    if args.log_every is not None:
        cfg.raw.setdefault("logging", {})["log_every"] = int(args.log_every)
    out_dir = run_all(cfg)
    print(f"Experiment outputs in: {out_dir}")
