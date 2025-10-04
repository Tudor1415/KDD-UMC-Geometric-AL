from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gal.search.engine import Search
from gal.search.strategies import get_strategy


def _init_streaming_outputs(exp_dir: Path) -> tuple[csv.writer, Any, Path]:
    """Initialize on-disk streaming outputs using NPZ files.

    - iterations.csv with a fixed schema
    - queries/ directory to store per-iteration query vectors as NPZ
    Returns (csv_writer, csv_file_handle, queries_dir)
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    it_csv = open(exp_dir / "iterations.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(it_csv)
    csv_writer.writerow([
        "iteration_id",
        "query_path",
        "oracle_response",
        "i",
        "j",
        "timestamp_start",
        "timestamp_end",
    ])  # schema
    q_dir = exp_dir / "queries"
    q_dir.mkdir(parents=True, exist_ok=True)
    return csv_writer, it_csv, q_dir


def _ensure_search_engine(engine: Optional[Search], search_strategy: str, X: np.ndarray) -> Search:
    if engine is not None:
        return engine
    strat = get_strategy(search_strategy, queries=X)
    return Search(strategy=strat)


def _log_iteration(
    it: int,
    i: Optional[int],
    j: Optional[int],
    dist: Optional[float],
    radius: float,
    *,
    log_level: int,
    log_every: int,
    exp_dir: Path,
) -> Path:
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


def _record_query_npz(
    *,
    it: int,
    diff: np.ndarray,
    q_dir: Path,
    csv_writer: csv.writer,
    y: int,
    i: int,
    j: int,
    t_start: float,
) -> None:
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
        # Best-effort; keep going even if NPZ fails
        pass


def _finalize_version_space_npz(exp_dir: Path, A: np.ndarray, b: np.ndarray) -> None:
    np.savez(
        exp_dir / "final_version_space.npz",
        A=np.asarray(A, dtype=float),
        b=np.asarray(b, dtype=float).reshape(-1, 1),
    )

