"""Top-level experiment runner using the rich helpers from src.gal.learning."""

from __future__ import annotations

import json
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from gal.core.data import Dataset, augment_with_minimums
from gal.search.engine import Search
from gal.search.strategies import get_strategy
from gal import trees as bt

from .config import ALConfig, dataset_entry_from_cfg, _configure_runtime_from_config
from gal.oracles.oracles import ObjectiveMeasureOracle, SumOracle, MDLOracle, SurpriseOracle, Oracle
from gal.centers import _center_fn
from gal.core.space import CapacitySpace
from gal.core.constraints import k_additive_constraints, enumerate_subsets
from gal.learning.learn import learning_loop


# ---------------------------------------------------------------------------
# Small utility helpers reused across the experiment runner
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _rand_uid(_rng) -> str:
    return secrets.token_hex(4)


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
        entry_name = str(entry.get("name", "<unnamed>"))
        log.debug("Preparing dataset entry '%s'", entry_name)

        ds, X = _load_dataset(entry)
        rows_read = ds.rows_read if getattr(ds, "rows_read", None) is not None else X.shape[0]
        duplicates_dropped = (
            ds.duplicates_dropped if getattr(ds, "duplicates_dropped", None) is not None else 0
        )
        log.debug(
            "Dataset '%s' loaded from %s → %d×%d (read=%d dropped_dupes=%d max_rows=%s drop_dupes=%s) measures=%s",
            ds.name,
            ds.dataset_path,
            X.shape[0],
            X.shape[1],
            rows_read,
            duplicates_dropped,
            ds.max_rows,
            ds.drop_duplicate_measure_vectors,
            ds.measures,
        )

        original_points = X.shape[0]
        max_points = int(cfg.get("global", "max_points", default=0) or 0)
        X = _maybe_downsample(cfg, X, rng)
        if X.shape[0] != original_points:
            log.debug(
                "Downsampled '%s' from %d to %d points (max_points=%d)",
                ds.name,
                original_points,
                X.shape[0],
                max_points,
            )
        else:
            log.debug("Using all %d points for '%s' (max_points=%d)", X.shape[0], ds.name, max_points)

        base_dim = X.shape[1]
        X_aug, space, A0, b0 = _prepare_space(cfg, X, log)
        log.debug(
            "Capacity space ready for '%s': base_dim=%d augmented_dim=%d constraints=%d",
            ds.name,
            base_dim,
            X_aug.shape[1],
            A0.shape[0],
        )

        center_fn, center_name = _choose_center(cfg)
        center_repr = getattr(center_fn, "__name__", center_fn.__class__.__name__)
        log.debug("Center function '%s' selected (%s)", center_name, center_repr)

        tree, tree_family, tree_method = _build_tree(cfg, X_aug)
        log.debug(
            "Tree built (%s:%s): samples=%d features=%d leaf_size=%d",
            tree_family,
            tree_method,
            tree.n_samples,
            tree.n_features,
            tree.leaf_size,
        )

        search_strategy = _strategy_name(cfg)
        log.debug("Search strategy set to '%s'", search_strategy)

        oracle = _build_oracle(cfg, ds)
        log.debug("Oracle '%s' initialised for dataset '%s'", oracle.name, ds.name)

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
        log.debug("Experiment directory created: %s", exp_dir)

        engine = _build_engine(search_strategy, X_aug)
        log.debug(
            "Search engine ready: bounder=%s strategy_impl=%s",
            engine.bounder.__class__.__name__,
            engine.strategy.__class__.__name__,
        )

        params = _run_params(cfg)
        log.debug(
            "Run parameters: n_iter=%d tau_cap=%g tau_multiplier=%g collect_events=%s log_every=%d",
            params.n_iter,
            params.tau_cap,
            params.tau_multiplier,
            params.collect_events,
            params.log_every,
        )

        log.debug("Starting learning loop for '%s'", ds.name)
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
            align_orientation=params.align_orientation,
            use_gpu=params.use_gpu,
        )
        log.debug("Learning loop completed for '%s'", ds.name)

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
    max_rows_raw = entry.get("max_rows", None)
    max_rows = None
    if max_rows_raw not in (None, ""):
        max_rows = int(max_rows_raw)

    drop_dupes_raw = entry.get("drop_duplicate_measures")
    if drop_dupes_raw is None:
        drop_dupes_raw = entry.get("drop_duplicate_measure_vectors")
    drop_dupes = False
    if isinstance(drop_dupes_raw, str):
        drop_dupes = drop_dupes_raw.strip().lower() in {"1", "true", "yes", "on"}
    elif drop_dupes_raw is not None:
        drop_dupes = bool(drop_dupes_raw)

    ds = Dataset(
        dataset_path=paths.get("dataset_path") or paths.get("mnr_rules"),
        transactions_path=paths.get("transactions_path") or paths.get("transactions"),
        item_rule_map_path=paths.get("item_rule_map_path") or paths.get("item_rule_map"),
        measures=entry.get("measures"),
        name=str(entry.get("name")),
        max_rows=max_rows,
        drop_duplicate_measure_vectors=drop_dupes,
    ).load()
    return ds, np.ascontiguousarray(ds.points, dtype=float)


def _maybe_downsample(cfg: ALConfig, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    max_pts = int(cfg.get("global", "max_points", default=0) or 0)
    if max_pts and X.shape[0] > max_pts:
        idx = rng.choice(X.shape[0], size=max_pts, replace=False)
        return np.ascontiguousarray(X[idx], dtype=float)
    return X


def _prepare_space(
    cfg: ALConfig, X: np.ndarray, log: logging.Logger
) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray]:
    add_k = int(cfg.get("experiment", "additivity_k", default=1) or 1)
    return _prepare_capacity_space(X, add_k=add_k, log=log)


def _prepare_capacity_space(
    X: np.ndarray,
    *,
    add_k: int,
    log: logging.Logger,
) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray]:
    if add_k < 1:
        raise ValueError("additivity_k must be at least 1")

    X = np.ascontiguousarray(X, dtype=float)
    X_aug = augment_with_minimums(X, add_k) if add_k > 1 else X.copy()

    n_features = X.shape[1]
    subsets = list(enumerate_subsets(n_features, add_k))
    if not subsets:
        raise ValueError("Failed to enumerate subsets for capacity space.")

    A0, b0, proj_index = k_additive_constraints(n_features, add_k)
    space = CapacitySpace(
        subsets=subsets,
        proj_index=proj_index,
        n_single=sum(1 for s in subsets if len(s) == 1),
        add_k=add_k,
    )

    log.debug(
        "Capacity space ready: n=%d add_k=%d subsets=%d constraints=%d",
        n_features,
        add_k,
        len(subsets),
        A0.shape[0],
    )

    return X_aug, space, np.asarray(A0, dtype=float), np.asarray(b0, dtype=float)


# ---------------------------------------------------------------------------
# Search components
# ---------------------------------------------------------------------------


def _choose_center(cfg: ALConfig) -> Tuple[Any, str]:
    name = str(
        cfg.get(
            "experiment",
            "center_name",
            default=cfg.get("global", "center", default="analytic"),
        )
    )
    raw_params = cfg.get("experiment", "center_params", default=None)
    if raw_params is None:
        params: Mapping[str, Any] = {}
    elif isinstance(raw_params, Mapping):
        params = raw_params
    else:
        raise ValueError("experiment.center_params must be a mapping when provided")
    return _center_fn(name, **dict(params)), name


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
    return Search(strategy=get_strategy(strategy_name))


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------


def _build_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    builders = {
        "objective": lambda: _objective_oracle(cfg, ds),
        "sum": lambda: _sum_oracle(cfg, ds),
        "surprise": lambda: _surprise_oracle(cfg),
        "mdl": lambda: _mdl_oracle(cfg),
    }
    otype = str(cfg.get("oracle", "type", default="objective")).lower()
    if otype not in builders:
        raise ValueError(f"Unsupported oracle type '{otype}'.")
    oracle = builders[otype]()
    oracle.set_dataset(ds)
    return oracle


def _objective_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    measure = cfg.get("oracle", "measure", default=None)
    if not measure:
        if not ds.measures:
            raise ValueError("Objective oracle requires at least one measure in dataset.")
        measure = ds.measures[0]
    return ObjectiveMeasureOracle(str(measure))


def _sum_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    measures = cfg.get("oracle", "measures", default=None)
    if not measures:
        measures = list(ds.measures)
    if not measures:
        raise ValueError("Sum oracle requires at least one measure.")
    return SumOracle([str(m) for m in measures])


def _mdl_oracle(cfg: ALConfig) -> Oracle:
    return MDLOracle(
        c0=float(cfg.get("oracle", "c0", default=8.0)),
        c_item=float(cfg.get("oracle", "c_item", default=4.0)),
    )


def _surprise_oracle(cfg: ALConfig) -> Oracle:
    prior_type = str(cfg.get("oracle", "prior_type", default="independent"))
    prior_kwargs = cfg.get("oracle", "prior_kwargs", default={}) or {}
    if not isinstance(prior_kwargs, dict):
        raise ValueError("oracle.prior_kwargs must be a mapping if provided.")
    return SurpriseOracle(prior_type=prior_type, **prior_kwargs)


# ---------------------------------------------------------------------------
# Experiment directory & metadata
# ---------------------------------------------------------------------------


def _output_root(cfg: ALConfig) -> Path:
    root = Path(cfg.get("global", "output_root", default="./results"))
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
        "align_orientation": bool(cfg.get("experiment", "align_orientation", default=False)),
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Run parameters
# ---------------------------------------------------------------------------


class _RunParams:
    def __init__(
        self,
        n_iter: int,
        tau_cap: float,
        tau_multiplier: float,
        collect_events: bool,
        log_every: int,
        log_level: int,
        align_orientation: bool,
        use_gpu: bool,
    ) -> None:
        self.n_iter = n_iter
        self.tau_cap = tau_cap
        self.tau_multiplier = tau_multiplier
        self.collect_events = collect_events
        self.log_every = log_every
        self.log_level = log_level
        self.align_orientation = align_orientation
        self.use_gpu = use_gpu


def _run_params(cfg: ALConfig) -> _RunParams:
    n_iter = int(cfg.get("experiment", "active_learning_budget", default=cfg.get("global", "n_iter", default=25)))
    tau_cap = float(cfg.get("experiment", "tau_max", default=1e-5))
    tau_mult = float(cfg.get("experiment", "tau_radius_multiplier", default=0.5))
    collect_events = bool(cfg.get("logging", "search_events", default=False))
    log_every = int(cfg.get("logging", "log_every", default=10) or 10)
    level_name = str(cfg.get("logging", "level", default="INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    raw_align_orientation = cfg.get("experiment", "align_orientation", default=None)
    if raw_align_orientation is None:
        raw_align_orientation = cfg.get("align_orientation", default=None)

    if isinstance(raw_align_orientation, str):
        align_orientation = raw_align_orientation.strip().lower() in {"1", "true", "yes", "on"}
    elif raw_align_orientation is None:
        align_orientation = False
    else:
        align_orientation = bool(raw_align_orientation)
    use_gpu = bool(cfg.get("experiment", "use_gpu", default=False))
    return _RunParams(
        n_iter,
        tau_cap,
        tau_mult,
        collect_events,
        log_every,
        log_level,
        align_orientation,
        use_gpu,
    )
