"""Simple ranking analyzer for AL runs.

This script loads stored centers for each iteration, scores the dataset using
those centers, and compares the rankings against the oracle used during the
experiment. Metrics (average precision, recall, NDCG) rely on scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, recall_score

from gal.core.data import Dataset, augment_with_minimums
from gal.experiments.config import ALConfig, dataset_entry_from_cfg
from gal.oracles.oracles import MDLOracle, ObjectiveMeasureOracle, Oracle, SumOracle, SurpriseOracle


@dataclass
class Inputs:
    run_dir: Path
    config_path: Path
    rules_override: Path | None
    transactions_override: Path | None
    topk: List[int]
    out_csv: Path


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
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[5, 10, 20, 50],
        help="List of K values for metrics",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path (default: <run_dir>/ranking_stats.csv)")
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
        topk=list(args.topk),
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


def _load_center(run_dir: Path, iteration: int, expected_dim: int) -> tuple[np.ndarray, float | None, float | None]:
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

    if center.size < expected_dim:
        padded = np.zeros(expected_dim, dtype=float)
        padded[: center.size] = center
        center = padded
    elif center.size > expected_dim:
        center = center[:expected_dim]

    radius_value = float(radius) if radius is not None else None
    tau_value = float(tau) if tau is not None else None
    return center, radius_value, tau_value


def _metric_labels(prefix: str, k_list: Sequence[int]) -> List[str]:
    labels: List[str] = []
    for k in k_list:
        labels.append(f"{prefix}{k}_ap")
        labels.append(f"{prefix}{k}_recall")
        labels.append(f"{prefix}{k}_ndcg")
    return labels


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
    topk_unique = sorted({k for k in inp.topk if k > 0})
    topk_labels = _metric_labels("top", topk_unique)
    top1pct_labels = ["top1pct_ap", "top1pct_recall", "top1pct_ndcg"]

    for it in iterations:
        center, radius, tau = _load_center(inp.run_dir, it, X.shape[1])
        pred_scores = X @ center

        row: Dict[str, Any] = {"iteration": it}
        for k in topk_unique:
            ap, recall, ndcg = _compute_metrics(pred_scores, oracle_scores, k)
            row[f"top{k}_ap"] = ap
            row[f"top{k}_recall"] = recall
            row[f"top{k}_ndcg"] = ndcg

        n = X.shape[0]
        k1 = max(1, int(np.ceil(0.01 * n)))
        ap1, rec1, ndcg1 = _compute_metrics(pred_scores, oracle_scores, k1)
        row["top1pct_ap"] = ap1
        row["top1pct_recall"] = rec1
        row["top1pct_ndcg"] = ndcg1

        row["radius"] = radius if radius is not None else ""
        row["tau"] = tau if tau is not None else ""
        rows.append(row)

    headers = ["iteration", *topk_labels, *top1pct_labels, "radius", "tau"]

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
