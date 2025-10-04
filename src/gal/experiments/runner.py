"""Top-level experiment runner using the rich helpers from src.gal.learning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from gal.core.data import Dataset
from gal.search.engine import Search
from gal.search.strategies import get_strategy
from gal import trees as bt

from .config import (
    ALConfig,
    _configure_runtime_from_config,
    _rand_uid,
    _timestamp,
    dataset_entry_from_cfg,
)
from gal.oracles.oracles import ObjectiveMeasureOracle, SumOracle, MDLOracle, Oracle
from gal.core.constraints import CapacitySpace  # type: ignore[attr-defined]
from experiments.active.space import _prepare_capacity_space
from experiments.active.centers import _center_fn
from src.gal.learning.learn import learning_loop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_all(cfg: ALConfig) -> Path:
    """Run all experiments declared in the configuration and return last path."""
    log = _setup_logging(cfg)
    _configure_runtime_from_config(cfg)
    rng = _rng(cfg)

    dataset_entries = _collect_dataset_entries(cfg)
    out_root = _output_root(cfg)

    last_dir: Path | None = None
    for entry in dataset_entries:
        ds, X = _load_dataset(entry)
        X = _maybe_downsample(cfg, X, rng)
        X_aug, space, A0, b0 = _prepare_space(cfg, X, log)

        center_fn, center_name = _choose_center(cfg)
        tree, tree_family, tree_method = _build_tree(cfg, X_aug)
        search_strategy = _strategy_name(cfg)
        oracle = _build_oracle(cfg, ds)

        exp_dir = _prepare_experiment_dir(
            cfg,
            entry,
            oracle,
            center_name,
            tree_family,
            tree_method,
            search_strategy,
            rng,
            out_root,
        )

        engine = _build_engine(search_strategy, X_aug)
        params = _run_params(cfg)

        learning_loop(
            tree=tree,
            X=X_aug,
            space=space,
            A0=A0,
            b0=b0,
            center_fn=center_fn,
            n_iter=params.n_iter,
            tau_cap=params.tau_cap,
            tau_multiplier=params.tau_multiplier,
            exp_dir=exp_dir,
            oracle_compare=oracle.compare_vectors,
            collect_events=params.collect_events,
            log_every=params.log_every,
            log_level=params.log_level,
            search_strategy=search_strategy,
            engine=engine,
        )

        last_dir = exp_dir

    return last_dir or out_root


# ---------------------------------------------------------------------------
# Logging / RNG helpers
# ---------------------------------------------------------------------------


def _setup_logging(cfg: ALConfig) -> logging.Logger:
    level_name = str(cfg.get("logging", "level", default="INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger(__name__)


def _rng(cfg: ALConfig) -> np.random.Generator:
    seed = int(cfg.get("global", "seed", default=1729))
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def _collect_dataset_entries(cfg: ALConfig) -> List[Dict[str, Any]]:
    ds_cfg = cfg.get("datasets", default=None)
    if ds_cfg is None:
        return [dataset_entry_from_cfg(cfg)]
    if not isinstance(ds_cfg, Iterable):
        raise ValueError("datasets must be a list in the configuration")
    return [dataset_entry_from_cfg(cfg, item) for item in ds_cfg]


def _load_dataset(entry: Dict[str, Any]) -> Tuple[Dataset, np.ndarray]:
    paths = entry.get("paths", {}) or {}
    ds = Dataset(
        dataset_path=paths.get("dataset_path") or paths.get("mnr_rules"),
        transactions_path=paths.get("transactions_path") or paths.get("transactions"),
        item_rule_map_path=paths.get("item_rule_map_path") or paths.get("item_rule_map"),
        measures=entry.get("measures"),
        name=str(entry.get("name")),
    ).load()
    return ds, np.ascontiguousarray(ds.points, dtype=float)


def _maybe_downsample(cfg: ALConfig, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    max_pts = int(cfg.get("global", "max_points", default=0) or 0)
    if max_pts and X.shape[0] > max_pts:
        idx = rng.choice(X.shape[0], size=max_pts, replace=False)
        return np.ascontiguousarray(X[idx], dtype=float)
    return X


def _prepare_space(cfg: ALConfig, X: np.ndarray, log: logging.Logger) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray]:
    add_k = int(cfg.get("experiment", "additivity_k", default=1) or 1)
    return _prepare_capacity_space(X, add_k=add_k, log=log)


# ---------------------------------------------------------------------------
# Search components
# ---------------------------------------------------------------------------


def _choose_center(cfg: ALConfig) -> Tuple[Any, str]:
    name = str(cfg.get("experiment", "center_name", default=cfg.get("global", "center", default="analytic")))
    return _center_fn(name), name


def _build_tree(cfg: ALConfig, X: np.ndarray) -> Tuple[Any, str, str]:
    method = str(cfg.get("trees", "ball", "method", default="two_pivot"))
    tree = bt.build_tree(X, cfg.get("trees", "ball", "config", default={}) or {}, method=method)
    return tree, "balltree", method


def _strategy_name(cfg: ALConfig) -> str:
    algo = cfg.get("algorithm_parameters", default={}) or {}
    strategies = algo.get("search_strategies")
    if isinstance(strategies, (list, tuple)) and strategies:
        return str(strategies[0])
    return str(algo.get("search_strategy", "lower_bound"))


def _build_engine(strategy_name: str, X: np.ndarray) -> Search:
    return Search(strategy=get_strategy(strategy_name, queries=X))


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------


def _build_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    otype = str(cfg.get("oracle", "type", default="objective"))
    if otype in {"objective", "objective_measure", "measure"}:
        measure = cfg.get("oracle", "measure", default=None) or (ds.measures[0] if ds.measures else None)
        oracle: Oracle = ObjectiveMeasureOracle(str(measure))
    elif otype == "sum":
        measures = cfg.get("oracle", "measures", default=None) or list(ds.measures)
        oracle = SumOracle([str(m) for m in measures])
    elif otype in {"mdl", "mdl_oracle"}:
        oracle = MDLOracle(
            c0=float(cfg.get("oracle", "c0", default=8.0)),
            c_item=float(cfg.get("oracle", "c_item", default=4.0)),
        )
    else:
        raise ValueError(f"Unsupported oracle type '{otype}'.")
    oracle.set_dataset(ds)
    return oracle


# ---------------------------------------------------------------------------
# Experiment directory & metadata
# ---------------------------------------------------------------------------


def _output_root(cfg: ALConfig) -> Path:
    root = Path(cfg.get("global", "output_root", default="./results/al"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prepare_experiment_dir(
    cfg: ALConfig,
    entry: Dict[str, Any],
    oracle: Oracle,
    center_name: str,
    tree_family: str,
    tree_method: str,
    strategy: str,
    rng: np.random.Generator,
    root: Path,
) -> Path:
    uid = _rand_uid(rng)
    name = f"{entry['name']}_{oracle.name}_{center_name}_{tree_family}-{tree_method}_{strategy}_{_timestamp()}_{uid}"
    exp_dir = root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metadata = _experiment_metadata(cfg, entry, oracle, center_name, tree_family, tree_method, strategy)
    (exp_dir / "config.json").write_text(metadata)
    return exp_dir


def _experiment_metadata(
    cfg: ALConfig,
    entry: Dict[str, Any],
    oracle: Oracle,
    center_name: str,
    tree_family: str,
    tree_method: str,
    strategy: str,
) -> str:
    data = {
        "dataset_name": entry.get("name"),
        "oracle_name": str(oracle.name),
        "center_name": center_name,
        "tree_family": tree_family,
        "tree_method": tree_method,
        "search_strategy": strategy,
        "active_learning_budget": int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25))),
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------


class _RunParams:
    def __init__(self, n_iter: int, tau_cap: float, tau_multiplier: float, collect_events: bool, log_every: int, log_level: int) -> None:
        self.n_iter = n_iter
        self.tau_cap = tau_cap
        self.tau_multiplier = tau_multiplier
        self.collect_events = collect_events
        self.log_every = log_every
        self.log_level = log_level


def _run_params(cfg: ALConfig) -> _RunParams:
    n_iter = int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))
    tau_cap = float(cfg.get("experiment", "tau_max", default=1e-5))
    tau_mult = float(cfg.get("experiment", "tau_radius_multiplier", default=0.5))
    collect_events = bool(cfg.get("logging", "search_events", default=False))
    log_every = int(cfg.get("logging", "log_every", default=10) or 10)
    level_name = str(cfg.get("logging", "level", default="INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    return _RunParams(n_iter, tau_cap, tau_mult, collect_events, log_every, log_level)
