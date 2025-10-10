"""High-level orchestration for convergence analysis workflows."""

from __future__ import annotations

import concurrent.futures
import csv
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .io import (
    constraints_prefix_for_iteration,
    empty_row,
    load_final_constraints,
    read_iterations_csv,
    relativize_path,
    stats_to_row,
    write_orientation_cdf,
)
from .logging_utils import configure_worker_logging
from .progress import ProgressBar
from .statistics import compute_all_stats

logger = logging.getLogger(__name__)


def _extract_first_finite(value: object) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    for item in arr:
        val = float(item)
        if math.isfinite(val):
            return val
    return None


def _load_orientation_scores_from_queries(
    run_dir: Path,
    iterations: Sequence[int],
) -> Dict[int, float]:
    scores: Dict[int, float] = {int(it): float("nan") for it in iterations}
    q_dir = run_dir / "queries"
    if not q_dir.is_dir():
        logger.debug("Queries directory %s missing; orientation scores default to NaN", q_dir)
        return scores

    for iteration in iterations:
        query_path = q_dir / f"query_{int(iteration):03d}.npz"
        if not query_path.is_file():
            logger.debug("Query file %s missing; orientation score set to NaN", query_path)
            continue
        try:
            with np.load(query_path) as data:
                extracted: float | None = None
                for key in ("orientation_score", "orientation", "score"):
                    if key in data.files:
                        extracted = _extract_first_finite(data[key])
                    if extracted is not None:
                        break
                if extracted is not None:
                    scores[int(iteration)] = extracted
                else:
                    logger.debug(
                        "Orientation score not found in %s; leaving NaN",
                        query_path,
                    )
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.debug(
                "Failed to read orientation score from %s: %s",
                query_path,
                exc,
            )
    return scores


def _load_orientation_scores_from_iterations_csv(
    iterations_csv: Path,
    iterations: Sequence[int],
) -> Dict[int, float]:
    scores: Dict[int, float] = {int(it): float("nan") for it in iterations}
    if not iterations_csv.is_file():
        return scores

    with iterations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token = row.get("iteration_id")
            if token is None or token == "":
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx not in scores:
                continue
            raw_orient = row.get("orientation_score")
            if raw_orient in (None, ""):
                continue
            try:
                value = float(raw_orient)
            except ValueError:
                continue
            if math.isfinite(value):
                scores[idx] = value
    return scores


def _compute_iteration_job(
    payload: Tuple[object, ...]
) -> Tuple[Dict[str, object], str | None]:
    (
        iteration,
        A,
        b,
        pool_size,
        hr_burn_in,
        hr_thinning,
        hr_seed,
        num_directions,
        epsilon_net_size,
        epsilon_seed,
        num_pairs,
        orientation_grid_size,
        orientation_output,
        collect_orientation,
        log_level,
    ) = payload
    configure_worker_logging(int(log_level))
    logger = logging.getLogger(__name__)
    logger.debug(
        "Worker starting iteration %d (constraints=%d)",
        iteration,
        A.shape[0],
    )
    try:
        stats = compute_all_stats(
            A,
            b,
            anchor=None,
            pool_size=pool_size,
            hr_burn_in=hr_burn_in,
            hr_thinning=hr_thinning,
            hr_seed=hr_seed,
            num_directions=num_directions,
            epsilon_net_size=epsilon_net_size,
            epsilon_seed=epsilon_seed,
            num_pairs=num_pairs,
            orientation_grid_size=orientation_grid_size,
            collect_orientation=bool(collect_orientation),
        )
        orientation_path = write_orientation_cdf(
            int(iteration),
            Path(orientation_output) if orientation_output else None,
            stats,
        )
        row = stats_to_row(int(iteration), stats)
        logger.debug("Worker finished iteration %d", iteration)
        return row, str(orientation_path) if orientation_path else None
    except Exception as exc:  # pragma: no cover - worker guard
        logger.exception("Worker failed for iteration %d", iteration)
        return empty_row(int(iteration), str(exc)), None


def compute_run_convergence(
    run_dir: Path,
    *,
    output_csv: Path | None,
    pool_size: int,
    hr_burn_in: int,
    hr_thinning: int,
    hr_seed: int,
    num_directions: int,
    epsilon_net_size: int,
    epsilon_seed: int,
    num_pairs: int,
    orientation_grid_size: int,
    jobs: int = 1,
) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    if output_csv is None:
        output_csv = run_dir / "convergence_stats.csv"

    output_csv = output_csv.expanduser().resolve()
    base_dir = output_csv.parent
    orientation_file = base_dir / f"{output_csv.stem}_orientation_cdf.json"

    iterations_csv = run_dir / "iterations.csv"
    if not iterations_csv.is_file():
        raise FileNotFoundError(f"Missing iterations.csv in {run_dir}")

    iterations = read_iterations_csv(iterations_csv)
    last_iteration = max(iterations)
    total_iterations = last_iteration + 1

    A_full, b_full = load_final_constraints(run_dir)
    if b_full.shape[0] != A_full.shape[0]:
        raise ValueError(
            f"Constraint mismatch: A has {A_full.shape[0]} rows, b has {b_full.shape[0]} entries"
        )

    effective_jobs = max(1, int(jobs))
    log_level = logging.getLogger().getEffectiveLevel()
    logger.debug(
        "compute_run_convergence: iterations=%d jobs=%d log_level=%s",
        len(iterations),
        effective_jobs,
        logging.getLevelName(log_level),
    )

    pending_payloads: List[Tuple[object, ...]] = []
    rows_map: Dict[int, Dict[str, object]] = {}

    for iteration in iterations:
        Ai, bi = constraints_prefix_for_iteration(A_full, b_full, iteration, total_iterations)
        if Ai is None or bi is None:
            logger.debug("Iteration %d skipped (insufficient constraints)", iteration)
            rows_map[iteration] = empty_row(iteration, "insufficient constraints")
            continue
        is_last = iteration == last_iteration
        orientation_path = orientation_file if is_last else None
        payload = (
            iteration,
            np.asarray(Ai, dtype=float),
            np.asarray(bi, dtype=float),
            pool_size,
            hr_burn_in,
            hr_thinning,
            hr_seed + iteration,
            num_directions,
            epsilon_net_size,
            epsilon_seed + iteration,
            num_pairs,
            orientation_grid_size,
            str(orientation_path) if orientation_path else None,
            is_last,
            log_level,
        )
        pending_payloads.append(payload)

    show_progress = log_level > logging.DEBUG
    progress = (
        ProgressBar(len(pending_payloads), "Computing iterations")
        if show_progress and pending_payloads
        else None
    )

    if effective_jobs == 1:
        for payload in pending_payloads:
            iteration = int(payload[0])
            logger.debug("Processing iteration %d sequentially", iteration)
            row, orient_path = _compute_iteration_job(payload)
            row["orientation_cdf_path"] = relativize_path(orient_path, base_dir)
            rows_map[iteration] = row
            if progress is not None:
                progress.update()
    else:
        logger.debug("Launching ProcessPoolExecutor with %d workers", effective_jobs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            future_to_iter = {
                executor.submit(_compute_iteration_job, payload): int(payload[0])
                for payload in pending_payloads
            }
            for future in concurrent.futures.as_completed(future_to_iter):
                iteration = future_to_iter[future]
                try:
                    row, orient_path = future.result()
                    row["orientation_cdf_path"] = relativize_path(orient_path, base_dir)
                    rows_map[iteration] = row
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Parallel worker crashed for iteration %d", iteration)
                    rows_map[iteration] = empty_row(iteration, str(exc))
                finally:
                    if progress is not None:
                        progress.update()

    if progress is not None:
        progress.close()

    config_path = run_dir / "config.json"
    orientation_enabled = False
    if config_path.exists():
        try:
            cfg_payload = json.loads(config_path.read_text())
            orientation_enabled = bool(cfg_payload.get("align_orientation"))
        except Exception:
            orientation_enabled = False

    orientation_map: Dict[int, float] = {}
    if orientation_enabled:
        orientation_map = _load_orientation_scores_from_queries(run_dir, iterations)
        if not any(math.isfinite(val) for val in orientation_map.values()):
            logger.debug(
                "Orientation alignment enabled but queries yielded no finite orientation scores",
            )

    rows: List[Dict[str, object]] = []
    for iteration in iterations:
        row = rows_map.get(iteration, empty_row(iteration, "missing result"))
        print(orientation_enabled)
        print(orientation_map.get(iteration, float("nan")))
        if orientation_enabled:
            row["orientation_score"] = orientation_map.get(iteration, float("nan"))
        else:
            row.pop("orientation_score", None)
        rows.append(row)

    fieldnames = [
        "iteration",
        "sphericity",
        "median_cosine_distance",
        "expected_theta",
    ]
    if orientation_enabled:
        fieldnames.append("orientation_score")
    fieldnames.extend(
        [
            "rho_from_theta",
            "varR_over_V2_from_theta",
            "varV_over_V2_from_theta",
            "orientation_cdf_path",
            "error",
        ]
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if orientation_enabled:
                value = row.get("orientation_score")
                if value is None:
                    row["orientation_score"] = ""
                else:
                    try:
                        val_float = float(value)
                        row["orientation_score"] = "" if not math.isfinite(val_float) else val_float
                    except (TypeError, ValueError):
                        row["orientation_score"] = ""
            writer.writerow(row)

    return output_csv


__all__ = ["compute_run_convergence"]
