from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gal.search.engine import Search
from gal.search.strategies import get_strategy
from gal import trees as bt

from .config import ALConfig, _configure_runtime_from_config, _dataset_entry_from_cfg, _rand_uid, _timestamp
from .space import CapacitySpace, _prepare_capacity_space
from .centers import _center_fn, _chebyshev_radius
from gal.core.data import Dataset
from gal.oracles.oracles import Oracle, ObjectiveMeasureOracle, SumOracle, MDLOracle


# ------------------------------ tiny setup utils ------------------------------ #

def setup_logging(cfg: ALConfig) -> None:
    lvl = str(cfg.get("logging", "level", default="INFO")).upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO), format="%(levelname)s %(message)s", force=True)


def setup_rng(cfg: ALConfig) -> np.random.Generator:
    return np.random.default_rng(int(cfg.get("global", "seed", default=1729)))


def get_dataset_entries(cfg: ALConfig) -> List[Dict[str, Any]]:
    items = cfg.get("datasets", default=None)
    if items is None:
        return [_dataset_entry_from_cfg(cfg)]
    return [_dataset_entry_from_cfg(cfg, item) for item in list(items)]


def load_dataset(entry: Dict[str, Any]) -> Tuple[Dataset, np.ndarray]:
    p = entry.get("paths", {}) or {}
    ds = Dataset(dataset_path=p.get("dataset_path") or p.get("mnr_rules"), transactions_path=p.get("transactions_path") or p.get("transactions"), item_rule_map_path=p.get("item_rule_map_path") or p.get("item_rule_map"), measures=entry.get("measures"), name=str(entry.get("name")))
    ds = ds.load()
    return ds, np.ascontiguousarray(ds.points, dtype=float)


def downsample(X: np.ndarray, cfg: ALConfig, rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.get("global", "max_points", default=0) or 0)
    if n and X.shape[0] > n:
        return np.ascontiguousarray(X[rng.choice(X.shape[0], size=n, replace=False)], dtype=float)
    return np.ascontiguousarray(X, dtype=float)


def choose_center_fn(cfg: ALConfig) -> Tuple[Callable, str]:
    name = str(cfg.get("experiment", "center_name", default=cfg.get("global", "center", default="analytic")))
    return _center_fn(name), name


def prepare_space(cfg: ALConfig, X: np.ndarray, log: logging.Logger) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray]:
    k = int(cfg.get("experiment", "additivity_k", default=1) or 1)
    return _prepare_capacity_space(X, add_k=k, log=log)


def build_tree(cfg: ALConfig, X: np.ndarray) -> Tuple[Any, str, str]:
    method = str(cfg.get("trees", "ball", "method", default="two_pivot"))
    return bt.build_tree(X, cfg.get("trees", "ball", "config", default={}) or {}, method=method), "balltree", method


def choose_strategy(cfg: ALConfig, X: np.ndarray) -> Tuple[Search, str]:
    algo = cfg.get("algorithm_parameters", default={}) or {}
    name = str(algo.get("search_strategy", (algo.get("search_strategies") or ["lower_bound"])[0]))
    return Search(strategy=get_strategy(name, queries=X)), name


def choose_oracle(cfg: ALConfig, ds: Dataset) -> Tuple[Oracle, Callable[[np.ndarray, np.ndarray], int], str]:
    otype = str(cfg.get("oracle", "type", default="objective"))
    if otype in {"objective", "measure"}:  # single measure
        m = cfg.get("oracle", "measure", default=(ds.measures[0] if ds.measures else None))
        oracle: Oracle = ObjectiveMeasureOracle(str(m))
    elif otype == "sum":
        oracle = SumOracle(list(ds.measures))
    else:
        oracle = MDLOracle()
    oracle.set_dataset(ds)
    return oracle, oracle.compare_vectors, str(oracle.name)


def exp_dir_path(cfg: ALConfig, ds_name: str, oracle: str, center: str, tree_family: str, tree_method: str, strategy: str, rng: np.random.Generator) -> Path:
    root = Path(cfg.get("global", "output_root", default="./results/al"))
    root.mkdir(parents=True, exist_ok=True)
    uid = _rand_uid(rng)
    name = f"{ds_name}_{oracle}_{center}_{tree_family}-{tree_method}_{strategy}_{_timestamp()}_{uid}"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_config_json(exp_dir: Path, meta: Dict[str, Any]) -> None:
    (exp_dir / "config.json").write_text(json.dumps(meta, indent=2))


def init_streaming(exp_dir: Path) -> Tuple[csv.writer, Any, Path]:
    f = open(exp_dir / "iterations.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["iteration_id", "query_path", "oracle_response", "i", "j", "timestamp_start", "timestamp_end"])  # schema
    qdir = exp_dir / "queries"
    qdir.mkdir(parents=True, exist_ok=True)
    return w, f, qdir


def init_state(A0: np.ndarray, b0: np.ndarray, space: CapacitySpace, center_fn: Callable, engine: Search) -> Dict[str, Any]:
    A = np.asarray(A0, dtype=float).copy(); b = np.asarray(b0, dtype=float).copy()
    cproj = np.asarray(center_fn(A, b), dtype=float); cfull = space.expand_center(cproj)
    rad = _chebyshev_radius(A, b, cproj)
    return {"A": A, "b": b, "center_full": cfull, "radius": float(rad), "engine": engine}


def should_stop(state: Dict[str, Any]) -> bool:
    r = state["radius"]
    return not (np.isfinite(r) and r > 0)


def compute_tau(radius: float, cap: float, mult: float) -> float:
    return float(min(radius * float(mult), float(cap)))


def record_query(qdir: Path, w: csv.writer, it: int, diff: np.ndarray, i: int, j: int, y: int, t0: float) -> None:
    np.savez(qdir / f"query_{it:03d}.npz", vector=np.asarray(diff, dtype=float))
    t1 = time.time()
    w.writerow([it, f"queries/query_{it:03d}.npz:vector", int(y), int(i), int(j), time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)) + f".{int((t0%1)*1000):03d}Z", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t1)) + f".{int((t1%1)*1000):03d}Z"])  # noqa: E501


def apply_constraint(state: Dict[str, Any], diff: np.ndarray, y: int, space: CapacitySpace, center_fn: Callable) -> Dict[str, Any]:
    A, b = state["A"], state["b"]
    row, rhs = space.project(-float(y) * diff)
    A = np.vstack([A, row.reshape(1, -1)]); b = np.concatenate([b, np.array([rhs], dtype=float)])
    cproj = np.asarray(center_fn(A, b), dtype=float); cfull = space.expand_center(cproj)
    r = float(_chebyshev_radius(A, b, cproj))
    return {**state, "A": A, "b": b, "center_full": cfull, "radius": r}


def save_center_snapshot(exp_dir: Path, it: int, center: np.ndarray, radius: float, tau: float) -> None:
    d = exp_dir / f"iteration_{it:03d}"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "center_model.npy", center)
    np.savez(d / "center_model.npz", center=np.asarray(center, dtype=float), radius=float(radius), tau=float(tau))


def finalize(exp_dir: Path, A: np.ndarray, b: np.ndarray, it_file) -> None:
    it_file.close()
    np.savez(exp_dir / "final_version_space.npz", A=np.asarray(A, dtype=float), b=np.asarray(b, dtype=float).reshape(-1, 1))


def do_iteration(it: int, state: Dict[str, Any], *, tree: Any, X: np.ndarray, oracle: Callable[[np.ndarray, np.ndarray], int], cap: float, mult: float, exp_dir: Path, qdir: Path, w: csv.writer, space: CapacitySpace, center_fn: Callable) -> Tuple[bool, Dict[str, Any]]:
    t0 = time.time(); tau = compute_tau(state["radius"], cap, mult)
    i, j, dist = state["engine"].search_pair(tree, X, state["center_full"], tau=float(tau))[:3]
    if i is None or j is None:
        return True, state
    a, b = X[int(i)], X[int(j)]; y = oracle(a, b); diff = a - b
    record_query(qdir, w, it, diff, int(i), int(j), int(y), t0)
    new_state = apply_constraint(state, diff, int(y), space=space, center_fn=center_fn)
    save_center_snapshot(exp_dir, it, new_state["center_full"], new_state["radius"], float(tau))
    return False, new_state


def learning_loop(*, tree: Any, X: np.ndarray, space: CapacitySpace, A0: np.ndarray, b0: np.ndarray, center_fn: Callable, n_iter: int, tau_cap: float, tau_multiplier: float, exp_dir: Path, oracle_compare: Callable[[np.ndarray, np.ndarray], int]) -> Tuple[np.ndarray, np.ndarray]:
    globals()["_LEARN_SPACE"] = space; globals()["_LEARN_CENTER_FN"] = center_fn
    w, f, qdir = init_streaming(exp_dir)
    state = init_state(A0, b0, space, center_fn, Search(strategy=get_strategy("lower_bound", queries=X)))
    for it in range(n_iter):
        if should_stop(state): break
        stop, state = do_iteration(it, state, tree=tree, X=X, oracle=oracle_compare, cap=tau_cap, mult=tau_multiplier, exp_dir=exp_dir, qdir=qdir, w=w)
        if stop: break
    finalize(exp_dir, state["A"], state["b"], f); return state["A"], state["b"]


# ----------------------------------- run API -------------------------------- #

def run_single(cfg: ALConfig, entry: Dict[str, Any]) -> Path:
    log = logging.getLogger(__name__); _configure_runtime_from_config(cfg)
    rng = setup_rng(cfg); ds, X0 = load_dataset(entry); X0 = downsample(X0, cfg, rng)
    X, space, A0, b0 = prepare_space(cfg, X0, log); center_fn, c_name = choose_center_fn(cfg)
    tree, t_family, t_method = build_tree(cfg, X); engine, s_name = choose_strategy(cfg, X)
    oracle_obj, oracle_cmp, o_name = choose_oracle(cfg, ds)
    exp = exp_dir_path(cfg, str(entry.get("name")), o_name, c_name, t_family, t_method, s_name, rng)
    write_config_json(exp, {"dataset_name": entry.get("name"), "oracle_name": o_name, "center_name": c_name, "tree_family": t_family, "tree_method": t_method, "search_strategy": s_name, "active_learning_budget": int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))})
    n_iter = int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))
    tau_cap = float(cfg.get("experiment", "tau_max", default=1e-5)); tau_mult = float(cfg.get("experiment", "tau_radius_multiplier", default=0.5))
    learning_loop(tree=tree, X=X, space=space, A0=A0, b0=b0, center_fn=center_fn, n_iter=n_iter, tau_cap=tau_cap, tau_multiplier=tau_mult, exp_dir=exp, oracle_compare=oracle_cmp)
    return exp


def run_all(cfg: ALConfig) -> Path:
    setup_logging(cfg); last = None
    for entry in get_dataset_entries(cfg):
        last = run_single(cfg, entry)
    return last or Path(cfg.get("global", "output_root", default="./results/al"))


def run(cfg: ALConfig) -> Path:
    setup_logging(cfg)
    entries = get_dataset_entries(cfg)
    return run_single(cfg, entries[0])


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run active learning")
    ap.add_argument("config", type=str); args = ap.parse_args()
    cfg = ALConfig.load(args.config)
    out = run_all(cfg)
    print(str(out))
