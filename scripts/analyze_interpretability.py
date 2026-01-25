#!/usr/bin/env python3
"""Compute iteration-wise distances between normalized centers and oracle weights."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from gal.core.constraints import enumerate_subsets
from gal.core.data import Dataset
from gal.experiments.config import ALConfig, dataset_entry_from_cfg
from gal.oracles.oracles import ChoquetOracle


logger = logging.getLogger(__name__)


@dataclass
class Inputs:
    run_dir: Path
    config_path: Path
    rules_override: Path | None
    transactions_override: Path | None
    out_csv: Path


def parse_args() -> Inputs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing iteration_* folders")
    parser.add_argument("--config", type=Path, required=True, help="YAML config used to launch the run")
    parser.add_argument("--rules", type=Path, default=None, help="Override rules CSV path")
    parser.add_argument(
        "--transactions", type=Path, default=None, help="Override transactions CSV path"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination CSV (default: <run_dir>/interpretability_stats.csv)",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    config_path = args.config
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    out_csv = args.out if args.out is not None else (run_dir / "interpretability_stats.csv")

    return Inputs(
        run_dir=run_dir,
        config_path=config_path,
        rules_override=args.rules,
        transactions_override=args.transactions,
        out_csv=out_csv,
    )


def _load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "config.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, Mapping) else {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s", meta_path, exc)
        return {}


def _dataset_entries(cfg: ALConfig) -> List[Dict[str, Any]]:
    raw = cfg.get("datasets", default=None)
    if raw is None:
        return [dataset_entry_from_cfg(cfg)]
    if not isinstance(raw, Iterable):
        raise SystemExit("'datasets' must be a list in the config file")
    return [dataset_entry_from_cfg(cfg, item) for item in raw]


def _select_entry(entries: List[Dict[str, Any]], dataset_name: str | None) -> Dict[str, Any]:
    if not entries:
        raise SystemExit("No dataset entries defined in config")
    if not dataset_name:
        return entries[0]
    for entry in entries:
        if str(entry.get("name", "")).strip() == str(dataset_name):
            return entry
    available = [entry.get("name") for entry in entries]
    raise SystemExit(
        f"Dataset '{dataset_name}' not found in config. Available: {available}"
    )


def _resolve_path(base_dir: Path, override: Path | None, value: Any) -> Path | None:
    if override is not None:
        return override
    if not value:
        return None

    raw = Path(str(value))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((base_dir / raw).resolve())
        candidates.append((Path.cwd() / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else raw


def _coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        norm = value.strip().lower()
        if not norm:
            return False
        return norm in {"1", "true", "yes", "on"}
    return bool(value)


def _load_dataset(inp: Inputs, cfg: ALConfig) -> tuple[Dataset, List[tuple[int, ...]], int]:
    entries = _dataset_entries(cfg)
    metadata = _load_run_metadata(inp.run_dir)
    entry = _select_entry(entries, metadata.get("dataset_name"))

    base_dir = inp.config_path.parent
    paths = entry.get("paths", {}) or {}

    dataset_path = _resolve_path(
        base_dir,
        inp.rules_override,
        paths.get("dataset_path") or paths.get("mnr_rules"),
    )
    if dataset_path is None:
        raise SystemExit("Dataset path not specified")

    transactions_path = _resolve_path(
        base_dir,
        inp.transactions_override,
        paths.get("transactions_path") or paths.get("transactions"),
    )
    item_rule_map_path = _resolve_path(
        base_dir,
        None,
        paths.get("item_rule_map_path") or paths.get("item_rule_map"),
    )

    drop_dupes = _coerce_bool(
        entry.get("drop_duplicate_measures") or entry.get("drop_duplicate_measure_vectors")
    )
    measures = entry.get("measures")
    max_rows = entry.get("max_rows")
    if max_rows is not None and str(max_rows).strip():
        max_rows = int(max_rows)
        if max_rows <= 0:
            max_rows = None

    ds = Dataset(
        dataset_path=dataset_path,
        transactions_path=transactions_path,
        item_rule_map_path=item_rule_map_path,
        measures=measures,
        name=str(entry.get("name") or dataset_path.stem),
        max_rows=max_rows,
        drop_duplicate_measure_vectors=drop_dupes,
    ).load()

    if not ds.measures:
        raise SystemExit("Dataset does not expose any measures; cannot compute weights")

    add_k_raw = cfg.get("experiment", "additivity_k", default=1) or 1
    add_k = max(1, int(add_k_raw))

    subsets = list(enumerate_subsets(len(ds.measures), add_k))
    if not subsets:
        raise SystemExit("Failed to enumerate subsets for capacity space")

    return ds, subsets, add_k


def _linear_weights(cfg: ALConfig, ds: Dataset) -> np.ndarray:
    oracle_section = cfg.get("oracle", default={}) or {}
    weights_cfg = oracle_section.get("weights")
    measures_cfg = oracle_section.get("measures")

    measure_names = [str(m) for m in ds.measures]

    if isinstance(weights_cfg, Mapping):
        mapping = {str(k): float(v) for k, v in weights_cfg.items()}
        weights = np.array([mapping.get(name, 0.0) for name in measure_names], dtype=float)
        return weights

    if isinstance(weights_cfg, Sequence) and not isinstance(weights_cfg, (str, bytes)):
        weights_arr = np.asarray(list(weights_cfg), dtype=float)
        if measures_cfg is not None:
            names = [str(m) for m in measures_cfg]
            if weights_arr.size != len(names):
                raise SystemExit(
                    f"Length of oracle.weights ({weights_arr.size}) does not match oracle.measures ({len(names)})"
                )
            weights = np.zeros(len(measure_names), dtype=float)
            name_to_idx = {name: idx for idx, name in enumerate(measure_names)}
            for weight, name in zip(weights_arr, names):
                if name not in name_to_idx:
                    raise SystemExit(
                        f"Measure '{name}' from oracle.measures missing in dataset measures {measure_names}"
                    )
                weights[name_to_idx[name]] = float(weight)
            return weights
        if weights_arr.size != len(measure_names):
            raise SystemExit(
                f"oracle.weights must have length {len(measure_names)} when oracle.measures is omitted"
            )
        return weights_arr.astype(float)

    measure_choice = oracle_section.get("measure")
    if measure_choice:
        name = str(measure_choice)
        if name not in measure_names:
            raise SystemExit(f"Measure '{name}' not found in dataset measures {measure_names}")
        weights = np.zeros(len(measure_names), dtype=float)
        weights[measure_names.index(name)] = 1.0
        return weights

    raise SystemExit(
        "Linear oracle requires 'oracle.weights' (list or mapping) or 'oracle.measure' in the config"
    )


def _sum_weights(cfg: ALConfig, ds: Dataset) -> np.ndarray:
    oracle_section = cfg.get("oracle", default={}) or {}
    measures = oracle_section.get("measures")
    measure_names = [str(m) for m in ds.measures]
    if measures is None:
        active = set(measure_names)
    else:
        active = {str(m) for m in measures}
    weights = np.zeros(len(measure_names), dtype=float)
    for idx, name in enumerate(measure_names):
        if name in active:
            weights[idx] = 1.0
    if not np.any(weights):
        raise SystemExit("Sum oracle produced an all-zero weight vector; check oracle.measures in config")
    return weights


def _choquet_weights(
    cfg: ALConfig,
    ds: Dataset,
    subsets: Sequence[tuple[int, ...]],
) -> np.ndarray:
    oracle_section = cfg.get("oracle", default={}) or {}
    subsets_cfg = oracle_section.get("subsets")
    if subsets_cfg is not None and not isinstance(subsets_cfg, Iterable):
        raise SystemExit("oracle.subsets must be iterable when provided")
    oracle = ChoquetOracle(subsets=subsets_cfg)
    oracle.set_dataset(ds)
    if getattr(oracle, "_weights", None) is None or getattr(oracle, "_subset_indices", None) is None:
        raise SystemExit("Choquet oracle did not expose weights after dataset attachment")
    weight_map: Dict[tuple[int, ...], float] = {
        tuple(subset): float(weight)
        for subset, weight in zip(oracle._subset_indices, oracle._weights)
    }
    weights = np.zeros(len(subsets), dtype=float)
    for idx, subset in enumerate(subsets):
        weights[idx] = weight_map.get(tuple(subset), 0.0)
    return weights


def _oracle_weights(cfg: ALConfig, ds: Dataset, subsets: Sequence[tuple[int, ...]]) -> np.ndarray:
    oracle_section = cfg.get("oracle", default={}) or {}
    otype = str(oracle_section.get("type", "linear")).lower()

    if otype in {"linear", "objective"}:
        base = _linear_weights(cfg, ds)
    elif otype == "sum":
        base = _sum_weights(cfg, ds)
    elif otype == "choquet":
        return _choquet_weights(cfg, ds, subsets)
    else:
        raise SystemExit(f"Unsupported oracle type '{otype}'. Supported: linear, sum, choquet.")

    weights = np.zeros(len(subsets), dtype=float)
    for idx, subset in enumerate(subsets):
        if len(subset) == 1:
            weights[idx] = base[subset[0]]
        else:
            weights[idx] = 0.0
    return weights


def _list_iterations(run_dir: Path) -> List[int]:
    csv_path = run_dir / "iterations.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing iterations.csv in {run_dir}")
    iterations: set[int] = set()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "iteration_id" not in reader.fieldnames:
            raise SystemExit("iterations.csv must contain an 'iteration_id' column")
        for row in reader:
            try:
                iterations.add(int(row["iteration_id"]))
            except Exception:
                continue
    ordered = sorted(iterations)
    if not ordered:
        raise SystemExit("No iterations found in iterations.csv")
    return ordered


def _load_center_strict(
    run_dir: Path,
    iteration: int,
    expected_dim: int,
) -> tuple[np.ndarray, float | None, float | None]:
    it_dir = run_dir / f"iteration_{iteration:03d}"
    npz_path = it_dir / "center_model.npz"
    center: np.ndarray
    radius: float | None = None
    tau: float | None = None

    if npz_path.exists():
        with np.load(npz_path) as data:
            if "center" not in data:
                raise SystemExit(f"'center' array missing in {npz_path}")
            center = np.asarray(data["center"], dtype=float)
            if "radius" in data:
                radius_val = data["radius"]
                radius = float(radius_val) if np.size(radius_val) else None
            if "tau" in data:
                tau_val = data["tau"]
                tau = float(tau_val) if np.size(tau_val) else None
    else:
        npy_path = it_dir / "center_model.npy"
        if not npy_path.exists():
            raise SystemExit(
                f"Missing center model for iteration {iteration}: {npz_path} or {npy_path}"
            )
        center = np.asarray(np.load(npy_path), dtype=float)

    center = center.reshape(-1)
    if center.size != expected_dim:
        raise SystemExit(
            f"Center dimensionality mismatch at iteration {iteration}: expected {expected_dim}, got {center.size}"
        )
    return center, radius, tau


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalized(vec: np.ndarray) -> tuple[np.ndarray | None, float]:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return None, norm
    return vec / norm, norm


def compute_interpretability(inp: Inputs) -> Path:
    cfg = ALConfig.load(inp.config_path)
    ds, subsets, _ = _load_dataset(inp, cfg)
    oracle_weights = _oracle_weights(cfg, ds, subsets)
    oracle_unit, oracle_norm = _normalized(oracle_weights)

    iterations = _list_iterations(inp.run_dir)
    rows: List[Dict[str, Any]] = []

    for iteration in iterations:
        center, radius, tau = _load_center_strict(inp.run_dir, iteration, len(subsets))
        cosine = _safe_cosine(center, oracle_weights)
        center_unit, center_norm = _normalized(center)
        if center_unit is not None and oracle_unit is not None:
            l2 = float(np.linalg.norm(center_unit - oracle_unit))
        else:
            l2 = float("nan")

        rows.append(
            {
                "iteration": iteration,
                "cosine_to_oracle": cosine,
                "l2_to_oracle": l2,
                "center_norm": center_norm,
                "oracle_norm": oracle_norm,
                "radius": radius if radius is not None else "",
                "tau": tau if tau is not None else "",
            }
        )

    inp.out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "iteration",
        "cosine_to_oracle",
        "l2_to_oracle",
        "center_norm",
        "oracle_norm",
        "radius",
        "tau",
    ]
    with inp.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    logger.info("Saved interpretability statistics to %s", inp.out_csv)
    return inp.out_csv


def main() -> None:  # pragma: no cover
    inp = parse_args()
    out_path = compute_interpretability(inp)
    print(f"Saved interpretability statistics -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
