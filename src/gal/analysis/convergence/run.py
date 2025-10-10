"""High-level orchestration for convergence analysis workflows."""

from __future__ import annotations

import concurrent.futures
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .io import (
    constraints_prefix_for_iteration,
    empty_row,
    load_final_constraints,
    read_iterations_csv,
    relativize_path,
    stats_to_row,
    write_center_vector,
    write_orientation_cdf,
)
from .logging_utils import configure_worker_logging
from .progress import ProgressBar
from .statistics import compute_all_stats

logger = logging.getLogger(__name__)


def _compute_iteration_job(
    payload: Tuple[object, ...]
) -> Tuple[Dict[str, object], str | None, str | None, str | None]:
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
        john_center_output,
        cheby_center_output,
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
        )
        orientation_path = write_orientation_cdf(
            int(iteration),
            Path(orientation_output) if orientation_output else None,
            stats,
        )
        john_path = write_center_vector(
            int(iteration),
            Path(john_center_output) if john_center_output else None,
            stats.john_center,
            "john_center",
        )
        cheby_path = write_center_vector(
            int(iteration),
            Path(cheby_center_output) if cheby_center_output else None,
            stats.cheby_center,
            "chebyshev_center",
        )
        row = stats_to_row(int(iteration), stats)
        logger.debug("Worker finished iteration %d", iteration)
        return (
            row,
            str(orientation_path) if orientation_path else None,
            str(john_path) if john_path else None,
            str(cheby_path) if cheby_path else None,
        )
    except Exception as exc:  # pragma: no cover - worker guard
        logger.exception("Worker failed for iteration %d", iteration)
        return empty_row(int(iteration), str(exc)), None, None, None


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
    orientation_dir = base_dir / f"{output_csv.stem}_orientation_cdf"
    john_center_dir = base_dir / f"{output_csv.stem}_john_center"
    cheby_center_dir = base_dir / f"{output_csv.stem}_chebyshev_center"

    iterations_csv = run_dir / "iterations.csv"
    if not iterations_csv.is_file():
        raise FileNotFoundError(f"Missing iterations.csv in {run_dir}")

    iterations = read_iterations_csv(iterations_csv)
    total_iterations = max(iterations) + 1
    index_width = max(4, len(str(max(iterations))))

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
        orientation_path = orientation_dir / f"iteration_{iteration:0{index_width}d}.json"
        john_center_path = john_center_dir / f"iteration_{iteration:0{index_width}d}.json"
        cheby_center_path = cheby_center_dir / f"iteration_{iteration:0{index_width}d}.json"
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
            str(orientation_path),
            str(john_center_path),
            str(cheby_center_path),
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
            row, orient_path, john_path, cheby_path = _compute_iteration_job(payload)
            row["orientation_cdf_path"] = relativize_path(orient_path, base_dir)
            row["john_center_path"] = relativize_path(john_path, base_dir)
            row["cheby_center_path"] = relativize_path(cheby_path, base_dir)
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
                    row, orient_path, john_path, cheby_path = future.result()
                    row["orientation_cdf_path"] = relativize_path(orient_path, base_dir)
                    row["john_center_path"] = relativize_path(john_path, base_dir)
                    row["cheby_center_path"] = relativize_path(cheby_path, base_dir)
                    rows_map[iteration] = row
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Parallel worker crashed for iteration %d", iteration)
                    rows_map[iteration] = empty_row(iteration, str(exc))
                finally:
                    if progress is not None:
                        progress.update()

    if progress is not None:
        progress.close()

    rows: List[Dict[str, object]] = []
    for iteration in iterations:
        rows.append(rows_map.get(iteration, empty_row(iteration, "missing result")))

    fieldnames = [
        "iteration",
        "rho_hat",
        "var_hat",
        "lambda_hat",
        "john_vol",
        "cheby_ball_vol",
        "cheby_radius",
        "r_max_from_a",
        "r_min_from_a",
        "sphericity",
        "ks_stat",
        "ks_p_value",
        "median_cosine_distance",
        "expected_theta",
        "rho_from_theta",
        "varR_over_V2_from_theta",
        "varV_over_V2_from_theta",
        "orientation_cdf_path",
        "john_center_path",
        "cheby_center_path",
        "error",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_csv


__all__ = ["compute_run_convergence"]
