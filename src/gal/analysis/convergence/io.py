"""Input/output helpers for convergence analysis workflows."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data import ConvergenceStats


def load_array(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "arr_0" in arr.files:
                data = arr["arr_0"]
            else:
                raise ValueError(f"NPZ at {path} must contain key 'arr_0'.")
        else:
            data = arr
        return np.asarray(data, dtype=float)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported file extension for {path}")


def parse_anchor(anchor_arg: str, expected_dim: int | None) -> np.ndarray:
    maybe_path = Path(anchor_arg)
    if maybe_path.exists():
        vec = np.asarray(load_array(maybe_path), dtype=float).reshape(-1)
    else:
        parts = [p.strip() for p in anchor_arg.split(",") if p.strip()]
        if not parts:
            raise ValueError("Anchor must be provided as path or comma-separated list of numbers.")
        vec = np.asarray([float(p) for p in parts], dtype=float)
    if expected_dim is not None and vec.size != expected_dim:
        raise ValueError(f"Anchor dimension mismatch: expected {expected_dim}, got {vec.size}.")
    return vec


def load_final_constraints(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    npz = run_dir / "final_version_space.npz"
    if npz.exists():
        with np.load(npz) as data:
            if "A" not in data or "b" not in data:
                raise ValueError(f"NPZ archive {npz} missing 'A' or 'b' datasets")
            A = np.asarray(data["A"], dtype=float)
            b = np.asarray(data["b"], dtype=float).reshape(-1)
        return A, b

    h5 = run_dir / "final_version_space.h5"
    if h5.exists():
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "final_version_space.h5 present but h5py is not installed"
            ) from exc
        with h5py.File(h5, "r") as hfile:
            if "A" not in hfile or "b" not in hfile:
                raise ValueError(f"HDF5 file {h5} missing 'A' or 'b' datasets")
            A = np.asarray(hfile["A"][...], dtype=float)
            b = np.asarray(hfile["b"][...], dtype=float).reshape(-1)
        return A, b

    raise FileNotFoundError(
        f"Could not find final_version_space.(npz|h5) under {run_dir}"
    )


def constraints_prefix_for_iteration(
    A: np.ndarray,
    b: np.ndarray,
    iteration: int,
    total_iterations: int,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if A.size == 0 or b.size == 0:
        return None, None
    base = max(0, int(A.shape[0]) - int(total_iterations))
    end = int(min(A.shape[0], base + iteration + 1))
    if end <= 0:
        return None, None
    return A[:end], b[:end]


def read_iterations_csv(path: Path) -> List[int]:
    rows: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "iteration_id" not in reader.fieldnames:
            raise ValueError(f"iterations.csv missing 'iteration_id' column: {reader.fieldnames}")
        for row in reader:
            token = row.get("iteration_id")
            if token is None or token == "":
                continue
            try:
                rows.append(int(token))
            except ValueError:
                continue
    if not rows:
        raise ValueError("No iteration records found in iterations.csv")
    return sorted(set(rows))


def clean_for_json(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_for_json(val) for key, val in value.items()}
    return value


def stats_to_row(iteration: int, stats: ConvergenceStats) -> Dict[str, object]:
    cleaned = clean_for_json(
        {
            "sphericity": stats.sphericity,
            "median_cosine_distance": stats.median_cosine_distance,
            "expected_theta": stats.expected_theta,
            "orientation_score": stats.orientation_score,
        }
    )

    theta = cleaned.get("expected_theta")
    if theta is not None:
        alpha = (2.0 / math.pi) * theta
        rho_from_theta = 1.0 - (theta / math.pi)
        varR_over_V2 = 0.25 * (1.0 - alpha)
        varV_over_V2 = 0.25 * alpha * (1.0 - alpha)
    else:
        rho_from_theta = None
        varR_over_V2 = None
        varV_over_V2 = None

    row: Dict[str, object] = {
        "iteration": iteration,
        "sphericity": cleaned.get("sphericity"),
        "median_cosine_distance": cleaned.get("median_cosine_distance"),
        "expected_theta": theta,
        "orientation_score": cleaned.get("orientation_score"),
        "rho_from_theta": rho_from_theta,
        "varR_over_V2_from_theta": varR_over_V2,
        "varV_over_V2_from_theta": varV_over_V2,
        "orientation_cdf_path": None,
        "error": "",
    }
    return row


def empty_row(iteration: int, error: str) -> Dict[str, object]:
    return {
        "iteration": iteration,
        "sphericity": None,
        "median_cosine_distance": None,
        "expected_theta": None,
        "orientation_score": None,
        "rho_from_theta": None,
        "varR_over_V2_from_theta": None,
        "varV_over_V2_from_theta": None,
        "orientation_cdf_path": None,
        "error": error,
    }


def write_orientation_cdf(
    iteration: int,
    output_path: Path | None,
    stats: ConvergenceStats,
) -> Path | None:
    if output_path is None or not stats.orientation_cdf:
        return None

    orientation_records = [
        clean_for_json(asdict(entry)) for entry in stats.orientation_cdf
    ]
    payload = {
        "iteration": iteration,
        "orientation_cdf": orientation_records,
    }
    payload = clean_for_json(payload)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
    except OSError as exc:  # pragma: no cover - filesystem failures
        import logging

        logging.getLogger(__name__).warning(
            "Failed to write orientation CDF for iteration %d: %s",
            iteration,
            exc,
        )
        return None

    return output_path


def relativize_path(path_str: str | None, base_dir: Path) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


__all__ = [
    "clean_for_json",
    "constraints_prefix_for_iteration",
    "empty_row",
    "load_array",
    "load_final_constraints",
    "parse_anchor",
    "read_iterations_csv",
    "relativize_path",
    "stats_to_row",
    "write_orientation_cdf",
]
