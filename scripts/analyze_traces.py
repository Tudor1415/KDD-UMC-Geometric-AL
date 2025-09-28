"""Analyze search traces across iterations and export CSV statistics.

Usage:
  python -m scripts.analyze_traces /path/to/run_dir --out stats.csv

Where /path/to/run_dir is a single experiment output folder that contains
subdirectories named iteration_XXX with search_trace.h5 files inside.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import h5py

def _read_events(path: Path) -> Dict[str, np.ndarray]:
    """Read events from a search_trace.h5 file.

    Returns empty arrays if h5py is unavailable or the file cannot be opened
    as HDF5 (e.g., placeholder/touched files), so the analyzer can proceed.
    """
    if h5py is None:
        return {k: np.array([]) for k in ("event_type", "node_id", "parent_id", "timestamp", "lower_bound", "upper_bound")}
    try:
        with h5py.File(path, "r") as h5:
            if "events" not in h5:
                return {k: np.array([]) for k in ("event_type", "node_id", "parent_id", "timestamp", "lower_bound", "upper_bound")}
            g = h5["events"]
            # Ensure arrays are numpy arrays and convert strings
            evt = g["event_type"][...]
            if evt.dtype.kind in {"S", "U", "O"}:  # decode ASCII bytes if needed
                evt = np.array([str(x, "ascii") if isinstance(x, (bytes, bytearray)) else str(x) for x in evt])
            else:
                evt = np.array([str(x) for x in evt])
            return {
                "event_type": evt,
                "node_id": np.asarray(g["node_id"][...], dtype=np.int64),
                "parent_id": np.asarray(g["parent_id"][...], dtype=np.int64),
                "timestamp": np.asarray(g["timestamp"][...], dtype=float),
                "lower_bound": np.asarray(g["lower_bound"][...], dtype=float),
                "upper_bound": np.asarray(g["upper_bound"][...], dtype=float),
            }
    except Exception:
        return {k: np.array([]) for k in ("event_type", "node_id", "parent_id", "timestamp", "lower_bound", "upper_bound")}


def _safe_percentile(x: np.ndarray, q: float, *, nan: float = np.nan) -> float:
    try:
        if x.size == 0:
            return nan
        return float(np.nanpercentile(x, q))
    except Exception:
        return nan


def _compute_stats(ev: Dict[str, np.ndarray]) -> Dict[str, float | int]:
    et = ev["event_type"]
    n = int(et.size)
    if n == 0:
        return {
            "created": 0,
            "expanded": 0,
            "pruned_total": 0,
            "pruned_lb": 0,
            "pruned_dom": 0,
            "pruned_ratio": 0.0,
            "gap_median": np.nan,
            "gap_p95": np.nan,
            "time_to_50pct_prunes": np.nan,
            "time_last_event": np.nan,
            "avg_branching_factor": np.nan,
            "median_branching_factor": np.nan,
            "node_lifetime_median": np.nan,
            "node_lifetime_p95": np.nan,
        }

    lb = ev["lower_bound"]
    ub = ev["upper_bound"]
    t = ev["timestamp"]
    nid = ev["node_id"]
    pid = ev["parent_id"]

    created = (et == "CREATED")
    expanded = (et == "EXPANDED")
    pruned = (et == "PRUNED")

    # Dominance prunes are logged with NaN bounds; LB prunes have finite bounds
    pruned_dom = pruned & (np.isnan(lb) | np.isnan(ub))
    pruned_lb = pruned & (~pruned_dom)

    # Gap only for finite bounds
    finite = np.isfinite(lb) & np.isfinite(ub)
    gap = np.maximum(0.0, ub - lb)
    gap[~finite] = np.nan

    # Branching: children count per parent among CREATED with valid parent_id
    children_per_parent: Dict[int, int] = {}
    for p in pid[created]:
        if int(p) >= 0:
            children_per_parent[int(p)] = children_per_parent.get(int(p), 0) + 1
    # Only consider parents that were EXPANDED at least once (better semantics)
    expanded_ids = set(map(int, nid[expanded]))
    bf_values = np.array([children_per_parent.get(eid, 0) for eid in expanded_ids], dtype=float)

    # Node lifetimes: time an enqueued pair spends in the queue
    # Interpret lifetime as: CREATED -> first of {PRUNED, EXPANDED}
    # This captures either immediate prune before expansion or time-to-pop.
    first_created: Dict[int, float] = {}
    first_pruned: Dict[int, float] = {}
    first_expanded: Dict[int, float] = {}
    for e, i, ts in zip(et, nid, t):
        ii = int(i)
        if e == "CREATED" and ii not in first_created:
            first_created[ii] = float(ts)
        elif e == "PRUNED" and ii not in first_pruned:
            first_pruned[ii] = float(ts)
        elif e == "EXPANDED" and ii not in first_expanded:
            first_expanded[ii] = float(ts)
    lifetimes: List[float] = []
    for ii, ts_c in first_created.items():
        ts_p = first_pruned.get(ii, float("inf"))
        ts_e = first_expanded.get(ii, float("inf"))
        ts_end = ts_p if ts_p <= ts_e else ts_e
        if np.isfinite(ts_end) and ts_end >= ts_c:
            lifetimes.append(ts_end - ts_c)
    lifetimes_arr = np.array(lifetimes, dtype=float) if lifetimes else np.array([], dtype=float)

    n_created = int(created.sum())
    n_pruned = int(pruned.sum())
    return {
        "created": n_created,
        "expanded": int(expanded.sum()),
        "pruned_total": n_pruned,
        "pruned_lb": int(pruned_lb.sum()),
        "pruned_dom": int(pruned_dom.sum()),
        "pruned_ratio": float(n_pruned / n_created) if n_created > 0 else 0.0,
        "gap_median": float(np.nanmedian(gap)) if np.isfinite(gap).any() else np.nan,
        "gap_p95": _safe_percentile(gap, 95.0),
        "time_to_50pct_prunes": _safe_percentile(t[pruned], 50.0) if n_pruned > 0 else np.nan,
        "time_last_event": float(np.nanmax(t)) if t.size else np.nan,
        "avg_branching_factor": float(np.nanmean(bf_values)) if bf_values.size else np.nan,
        "median_branching_factor": float(np.nanmedian(bf_values)) if bf_values.size else np.nan,
        "node_lifetime_median": float(np.nanmedian(lifetimes_arr)) if lifetimes_arr.size else np.nan,
        "node_lifetime_p95": _safe_percentile(lifetimes_arr, 95.0),
    }


def _find_iteration_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    iters: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("iteration_"):
            try:
                k = int(p.name.split("_")[-1])
            except Exception:
                continue
            iters.append((k, p))
    iters.sort(key=lambda x: x[0])
    return iters


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze search_trace.h5 across iterations and export CSV stats")
    ap.add_argument("run_dir", type=str, help="Path to a single run directory (contains iteration_XXX)")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: <run_dir>/trace_stats.csv)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    out_csv = Path(args.out) if args.out else (run_dir / "trace_stats.csv")

    rows: List[Dict[str, object]] = []
    for it_id, it_dir in _find_iteration_dirs(run_dir):
        st_path = it_dir / "search_trace.h5"
        if not st_path.exists():
            rows.append({"iteration": it_id, "error": "missing search_trace.h5"})
            continue
        ev = _read_events(st_path)
        stats = _compute_stats(ev)
        stats_row: Dict[str, object] = {"iteration": it_id}
        stats_row.update(stats)
        rows.append(stats_row)

    # Minimal CSV writer (no pandas dependency)
    import csv

    # Collect all keys to stabilize header
    keys: List[str] = ["iteration"]
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved statistics → {out_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
