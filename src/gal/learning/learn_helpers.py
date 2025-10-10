from __future__ import annotations

import csv
import math
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        "orientation_score",
        "timestamp_start",
        "timestamp_end",
    ])  # schema
    q_dir = exp_dir / "queries"
    q_dir.mkdir(parents=True, exist_ok=True)
    return csv_writer, it_csv, q_dir


def _ensure_search_engine(engine: Optional[Search], search_strategy: str, X: np.ndarray) -> Search:
    if engine is not None:
        return engine
    strat = get_strategy(search_strategy)
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
    orientation_score: float | None,
) -> None:
    """Persist the iteration query vector and append a CSV row.

    Writes queries/query_###.npz with key 'vector' and logs row into iterations.csv.
    """
    q_path = q_dir / f"query_{it:03d}.npz"
    np.savez(q_path, vector=np.asarray(diff, dtype=float))
    t_end = time.time()
    orientation_cell = "" if orientation_score is None or not math.isfinite(orientation_score) else float(orientation_score)
    csv_writer.writerow([
        it,
        f"queries/query_{it:03d}.npz:vector",
        int(y),
        int(i),
        int(j),
        orientation_cell,
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


def _farthest_point_socp(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    *,
    max_directions: int = 64,
    atol: float = 1e-9,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Return the farthest feasible point from ``center`` using LP or fail gracefully."""

    A_mat = np.asarray(A, dtype=float)
    if A_mat.size == 0:
        logging.getLogger(__name__).debug("Skipping farthest-point search: empty constraint set")
        return None, None

    center_vec = np.asarray(center, dtype=float).reshape(-1)
    if center_vec.size == 0:
        raise ValueError("Center must be a non-empty vector")
    if A_mat.shape[1] != center_vec.size:
        raise ValueError(
            "Constraint matrix column count does not match center dimension: "
            f"{A_mat.shape[1]} vs {center_vec.size}"
        )

    try:
        from scipy.optimize import linprog
    except ImportError:
        logging.getLogger(__name__).warning(
            "scipy not installed; skipping farthest-point computation"
        )
        return None, None

    b_vec = np.asarray(b, dtype=float).reshape(-1)
    dim = center_vec.size
    if dim == 0:
        return None, None

    def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vec))
        if norm <= atol:
            return None
        return vec / norm

    unbounded_detected = False

    def _lp(direction: np.ndarray) -> Optional[np.ndarray]:
        nonlocal unbounded_detected
        res = linprog(
            -direction,
            A_ub=A_mat,
            b_ub=b_vec,
            bounds=[(None, None)] * dim,
            method="highs",
        )
        if res.status == 3:
            logging.getLogger(__name__).warning(
                "Farthest-point search encountered an unbounded LP direction"
            )
            unbounded_detected = True
            return None
        if not res.success or res.x is None:
            logging.getLogger(__name__).debug(
                "Farthest-point LP failed (status=%s message=%s)", res.status, res.message
            )
            return None
        return np.asarray(res.x, dtype=float)

    initial_dirs: List[np.ndarray] = []
    for idx in range(dim):
        e = np.zeros(dim, dtype=float)
        e[idx] = 1.0
        initial_dirs.append(e)
        initial_dirs.append(-e)
    normalized_center = _normalize(center_vec)
    if normalized_center is not None:
        initial_dirs.append(normalized_center)

    queue: List[np.ndarray] = initial_dirs
    seen: List[np.ndarray] = []
    best_point: Optional[np.ndarray] = None
    best_dist: float = 0.0
    processed = 0

    while queue and processed < max_directions:
        direction = queue.pop(0)
        unit_dir = _normalize(direction)
        if unit_dir is None:
            continue
        if any(np.allclose(unit_dir, s, atol=1e-8) or np.allclose(unit_dir, -s, atol=1e-8) for s in seen):
            continue
        seen.append(unit_dir)
        candidate = _lp(unit_dir)
        processed += 1
        if candidate is None:
            continue
        dist = float(np.linalg.norm(candidate - center_vec))
        if dist <= best_dist + atol:
            continue
        best_point = candidate
        best_dist = dist
        improvement_dir = _normalize(candidate - center_vec)
        if improvement_dir is not None:
            queue.append(improvement_dir)

    if best_point is None or best_dist <= atol:
        return None, None

    if unbounded_detected:
        return None, None

    return best_point, best_dist
