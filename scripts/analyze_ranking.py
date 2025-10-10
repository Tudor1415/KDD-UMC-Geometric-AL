"""Analyze AL run rankings for select top-K summaries.

This script scores stored centers against the dataset used in the run and
computes ranking metrics for three summaries: top10, top1pct, top5pct.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, recall_score

from gal.core.data import Dataset, augment_with_minimums
from gal.experiments.config import ALConfig, dataset_entry_from_cfg
from gal.oracles.oracles import (
    ChoquetOracle,
    MDLOracle,
    ObjectiveMeasureOracle,
    Oracle,
    SumOracle,
    SurpriseOracle,
)


logger = logging.getLogger(__name__)


@dataclass
class Inputs:
    run_dir: Path
    config_path: Path
    rules_override: Path | None
    transactions_override: Path | None
    out_csv: Path
    jobs: int


def parse_args() -> Inputs:
    parser = argparse.ArgumentParser(description="Analyze ranking quality for an AL run")
    parser.add_argument("run_dir", type=Path, help="Run directory containing iteration_* folders")
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment config used for the run")
    parser.add_argument("--rules", type=Path, default=None, help="Override rules CSV (defaults to config entry)")
    parser.add_argument(
        "--transactions",
        type=Path,
        default=None,
        help="Override transactions CSV (defaults to config entry)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (default: <run_dir>/ranking_stats.csv)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker threads for per-iteration ranking stats (1 disables parallelism)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    out_csv = args.out if args.out is not None else (run_dir / "ranking_stats.csv")

    return Inputs(
        run_dir=run_dir,
        config_path=args.config,
        rules_override=args.rules,
        transactions_override=args.transactions,
        out_csv=out_csv,
        jobs=max(1, int(args.jobs)),
    )


def _load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "config.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _dataset_entries(cfg: ALConfig) -> List[Dict[str, Any]]:
    raw = cfg.get("datasets", default=None)
    if raw is None:
        return [dataset_entry_from_cfg(cfg)]
    if not isinstance(raw, Iterable):
        raise ValueError("'datasets' in config must be an iterable")
    return [dataset_entry_from_cfg(cfg, item) for item in raw]


def _select_entry(entries: List[Dict[str, Any]], dataset_name: str | None) -> Dict[str, Any]:
    if not entries:
        raise ValueError("No datasets defined in config")
    if not dataset_name:
        return entries[0]
    for entry in entries:
        if str(entry.get("name", "")).strip() == str(dataset_name):
            return entry
    raise ValueError(
        f"Dataset '{dataset_name}' not found in config. Available: {[e.get('name') for e in entries]}"
    )


def _resolve_path(base: Path, override: Path | None, value: Any) -> Path | None:
    if override is not None:
        return override
    if not value:
        return None

    raw = Path(str(value))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((base / raw).resolve())
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


def _load_dataset(inp: Inputs, cfg: ALConfig) -> tuple[Dataset, np.ndarray, int]:
    entries = _dataset_entries(cfg)
    metadata = _load_run_metadata(inp.run_dir)
    entry = _select_entry(entries, metadata.get("dataset_name"))

    base_dir = inp.config_path.parent
    paths = entry.get("paths", {}) or {}

    dataset_path = _resolve_path(base_dir, inp.rules_override, paths.get("dataset_path") or paths.get("mnr_rules"))
    if dataset_path is None:
        raise SystemExit("Dataset path not specified in config or CLI")

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

    add_k = cfg.get("experiment", "additivity_k", default=1) or 1
    add_k = max(1, int(add_k))

    X = np.asarray(ds.points, dtype=float)
    if add_k > 1:
        X = np.ascontiguousarray(augment_with_minimums(X, add_k), dtype=float)
    else:
        X = np.ascontiguousarray(X, dtype=float)

    return ds, X, add_k


def _build_oracle(cfg: ALConfig, ds: Dataset) -> Oracle:
    otype = str(cfg.get("oracle", "type", default="objective")).lower()

    if otype == "objective":
        measure = cfg.get("oracle", "measure", default=None)
        if not measure:
            if not ds.measures:
                raise SystemExit("Objective oracle requires at least one measure in dataset")
            measure = ds.measures[0]
        oracle = ObjectiveMeasureOracle(str(measure))
    elif otype == "sum":
        measures = cfg.get("oracle", "measures", default=None) or ds.measures
        if not measures:
            raise SystemExit("Sum oracle requires at least one measure")
        oracle = SumOracle([str(m) for m in measures])
    elif otype == "mdl":
        oracle = MDLOracle(
            c0=float(cfg.get("oracle", "c0", default=8.0)),
            c_item=float(cfg.get("oracle", "c_item", default=4.0)),
        )
    elif otype == "surprise":
        prior_type = str(cfg.get("oracle", "prior_type", default="independent"))
        prior_kwargs = cfg.get("oracle", "prior_kwargs", default={}) or {}
        if not isinstance(prior_kwargs, dict):
            raise SystemExit("oracle.prior_kwargs must be a mapping")
        oracle = SurpriseOracle(prior_type=prior_type, **prior_kwargs)
    elif otype == "choquet":
        subsets = cfg.get("oracle", "subsets", default=None)
        if subsets is not None and not isinstance(subsets, Iterable):
            raise SystemExit("oracle.subsets must be iterable when provided")
        oracle = ChoquetOracle(subsets=subsets)
    else:
        raise SystemExit(f"Unsupported oracle type '{otype}'")

    oracle.set_dataset(ds)
    return oracle



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
    return sorted(iterations)


def _load_center(
    run_dir: Path,
    iteration: int,
    expected_dim: int | None,
) -> tuple[np.ndarray, float | None, float | None]:
    it_dir = run_dir / f"iteration_{iteration:03d}"
    npz_path = it_dir / "center_model.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            center = np.asarray(data.get("center"), dtype=float)
            if center is None:
                raise SystemExit(f"'center' missing in {npz_path}")
            radius = data.get("radius")
            tau = data.get("tau")
    else:
        npy_path = it_dir / "center_model.npy"
        if not npy_path.exists():
            raise SystemExit(f"Missing center model for iteration {iteration}: {npz_path} or {npy_path}")
        center = np.asarray(np.load(npy_path), dtype=float)
        radius = tau = None

    if center.ndim != 1:
        center = center.reshape(-1)

    if expected_dim is not None:
        if center.size < expected_dim:
            padded = np.zeros(expected_dim, dtype=float)
            padded[: center.size] = center
            center = padded
        elif center.size > expected_dim:
            center = center[:expected_dim]

    radius_value = float(radius) if radius is not None else None
    tau_value = float(tau) if tau is not None else None
    return center, radius_value, tau_value


def _compute_metrics(
    pred_scores: np.ndarray,
    oracle_scores: np.ndarray,
    k: int,
) -> tuple[float, float, float]:
    n = pred_scores.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    K = max(1, min(k, n))
    oracle_order = np.argsort(-oracle_scores)
    pred_order = np.argsort(-pred_scores)

    y_true = np.zeros(n, dtype=int)
    y_true[oracle_order[:K]] = 1

    ap = average_precision_score(y_true, pred_scores)

    y_pred = np.zeros(n, dtype=int)
    y_pred[pred_order[:K]] = 1
    recall = recall_score(y_true, y_pred, zero_division=0)

    ndcg = float(ndcg_score([y_true], [pred_scores], k=K))
    return float(ap), float(recall), ndcg


def compute_ranking(inp: Inputs) -> Path:
    cfg = ALConfig.load(inp.config_path)
    ds, X, _ = _load_dataset(inp, cfg)
    oracle = _build_oracle(cfg, ds)
    oracle_scores = np.asarray(oracle.score_dataset(ds), dtype=float)
    iterations = _list_iterations(inp.run_dir)

    rows: List[Dict[str, Any]] = []
    rows_map: Dict[int, Dict[str, Any]] = {}

    def process_iteration(it: int) -> Dict[str, Any] | None:
        try:
            center, radius, tau = _load_center(inp.run_dir, it, X.shape[1])
        except SystemExit as exc:
            exc_code = exc.code
            message = exc_code if isinstance(exc_code, str) else str(exc)
            if message and "Missing center model" in message:
                logger.warning("Skipping iteration %d due to missing center model: %s", it, message)
                return None
            raise

        pred_scores = X @ center
        n = X.shape[0]

        row: Dict[str, Any] = {"iteration": it}
        top10_ap, top10_rec, top10_ndcg = _compute_metrics(pred_scores, oracle_scores, 10)
        row["top10_ap"] = top10_ap
        row["top10_recall"] = top10_rec
        row["top10_ndcg"] = top10_ndcg

        k1 = max(1, int(np.ceil(0.01 * n)))
        top1pct_ap, top1pct_rec, top1pct_ndcg = _compute_metrics(pred_scores, oracle_scores, k1)
        row["top1pct_ap"] = top1pct_ap
        row["top1pct_recall"] = top1pct_rec
        row["top1pct_ndcg"] = top1pct_ndcg

        k5 = max(1, int(np.ceil(0.05 * n)))
        top5pct_ap, top5pct_rec, top5pct_ndcg = _compute_metrics(pred_scores, oracle_scores, k5)
        row["top5pct_ap"] = top5pct_ap
        row["top5pct_recall"] = top5pct_rec
        row["top5pct_ndcg"] = top5pct_ndcg

        row["radius"] = radius if radius is not None else ""
        row["tau"] = tau if tau is not None else ""
        return row

    if inp.jobs == 1:
        for it in iterations:
            row = process_iteration(it)
            if row is not None:
                rows_map[it] = row
    else:
        logger.info("Computing ranking metrics with %d worker threads", inp.jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=inp.jobs) as executor:
            future_to_iter = {executor.submit(process_iteration, it): it for it in iterations}
            for future in concurrent.futures.as_completed(future_to_iter):
                iteration = future_to_iter[future]
                try:
                    row = future.result()
                    if row is not None:
                        rows_map[iteration] = row
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Ranking computation failed for iteration %d", iteration)

    for it in iterations:
        if it in rows_map:
            rows.append(rows_map[it])

    headers = [
        "iteration",
        "top10_ap",
        "top10_recall",
        "top10_ndcg",
        "top1pct_ap",
        "top1pct_recall",
        "top1pct_ndcg",
        "top5pct_ap",
        "top5pct_recall",
        "top5pct_ndcg",
        "radius",
        "tau",
    ]

    inp.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with inp.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    return inp.out_csv


def main() -> None:  # pragma: no cover
    inp = parse_args()
    out = compute_ranking(inp)
    print(f"Saved ranking statistics → {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
