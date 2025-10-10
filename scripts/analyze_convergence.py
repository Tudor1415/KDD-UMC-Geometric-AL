#!/usr/bin/env python3
"""Analyze polytope convergence statistics."""

from __future__ import annotations

import argparse
import json
import math
import logging
import warnings
from pathlib import Path
from typing import Sequence


from src.gal.analysis.convergence import (
    compute_all_stats,
    compute_run_convergence,
    load_array,
    parse_anchor,
)
from src.gal.analysis.convergence.io import clean_for_json, write_orientation_cdf

warnings.filterwarnings("ignore")
logging.getLogger("gurobipy").setLevel(logging.ERROR)
logging.getLogger("cvxpy").setLevel(logging.ERROR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Active learning run directory (per-iteration mode).")
    parser.add_argument("--output-csv", type=Path, help="Destination CSV path (defaults to <run_dir>/convergence_stats.csv).")
    parser.add_argument("--A", type=Path, help="Path to matrix A (.npy/.npz/.csv).")
    parser.add_argument("--b", type=Path, help="Path to vector b (.npy/.npz/.csv).")
    parser.add_argument(
        "--anchor",
        type=str,
        help="Anchor point either as file path or comma-separated list of coordinates.",
    )
    parser.add_argument("--pool-size", type=int, default=5000, help="Number of hit-and-run samples to retain.")
    parser.add_argument("--hr-burn-in", type=int, default=1000, help="Burn-in steps for hit-and-run.")
    parser.add_argument("--hr-thinning", type=int, default=10, help="Thinning interval for hit-and-run.")
    parser.add_argument("--hr-seed", type=int, default=42, help="RNG seed for hit-and-run sampling.")
    parser.add_argument("--num-directions", type=int, default=1000, help="Monte Carlo directions for rho.")
    parser.add_argument("--epsilon-net-size", type=int, default=4096, help="Size of spherical epsilon-net for Minkowski symmetry.")
    parser.add_argument("--epsilon-seed", type=int, default=123, help="RNG seed for epsilon-net and cosine sampling.")
    parser.add_argument("--num-pairs", type=int, default=100000, help="Number of constraint pairs for cosine distance sampling.")
    parser.add_argument("--orientation-grid", type=int, default=181, help="Number of angle grid points for orientation CDF.")
    parser.add_argument(
        "--orientation-output",
        type=Path,
        help="Orientation CDF JSON path (defaults to <output>_* when using direct mode).",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for per-iteration stats (1 disables parallelism).")
    parser.add_argument("--output", type=Path, help="Optional path to JSON file for results.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    return parser


def main(args: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    options = parser.parse_args(args)
    logging.basicConfig(level=getattr(logging, options.log_level.upper(), logging.INFO))

    if options.run_dir is not None:
        stats_path = compute_run_convergence(
            options.run_dir,
            output_csv=options.output_csv,
            pool_size=options.pool_size,
            hr_burn_in=options.hr_burn_in,
            hr_thinning=options.hr_thinning,
            hr_seed=options.hr_seed,
            num_directions=options.num_directions,
            epsilon_net_size=options.epsilon_net_size,
            epsilon_seed=options.epsilon_seed,
            num_pairs=options.num_pairs,
            orientation_grid_size=options.orientation_grid,
            jobs=options.jobs,
        )
        logging.info("Wrote per-iteration convergence stats to %s", stats_path)
        return

    if not (options.A and options.b and options.anchor):
        parser.error("Provide --run-dir or the trio --A, --b, --anchor")

    A = load_array(options.A)
    b = load_array(options.b).reshape(-1)
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2D.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Vector b must be 1D with length equal to rows of A.")
    anchor = parse_anchor(options.anchor, expected_dim=A.shape[1])

    stats_result = compute_all_stats(
        A,
        b,
        anchor,
        pool_size=options.pool_size,
        hr_burn_in=options.hr_burn_in,
        hr_thinning=options.hr_thinning,
        hr_seed=options.hr_seed,
        num_directions=options.num_directions,
        epsilon_net_size=options.epsilon_net_size,
        epsilon_seed=options.epsilon_seed,
        num_pairs=options.num_pairs,
        orientation_grid_size=options.orientation_grid,
    )

    orientation_output = options.orientation_output
    if orientation_output is None:
        if options.output:
            orientation_output = options.output.with_name(f"{options.output.stem}_orientation_cdf.json")
        else:
            parser.error("Provide --orientation-output or --output when not using --run-dir")

    orientation_output = orientation_output.expanduser().resolve()
    orientation_written = write_orientation_cdf(0, orientation_output, stats_result)
    if orientation_written:
        logging.info("Wrote orientation CDF to %s", orientation_written)

    cleaned_stats = clean_for_json(
        {
            "sphericity": stats_result.sphericity,
            "median_cosine_distance": stats_result.median_cosine_distance,
            "expected_theta": stats_result.expected_theta,
            "orientation_score": stats_result.orientation_score,
        }
    )

    theta = cleaned_stats.get("expected_theta")
    if theta is not None:
        alpha = (2.0 / math.pi) * theta
        cleaned_stats["rho_from_theta"] = 1.0 - (theta / math.pi)
        cleaned_stats["varR_over_V2_from_theta"] = 0.25 * (1.0 - alpha)
        cleaned_stats["varV_over_V2_from_theta"] = 0.25 * alpha * (1.0 - alpha)
    else:
        cleaned_stats["rho_from_theta"] = None
        cleaned_stats["varR_over_V2_from_theta"] = None
        cleaned_stats["varV_over_V2_from_theta"] = None

    cleaned_stats["orientation_cdf_path"] = str(orientation_written) if orientation_written else None

    text = json.dumps(cleaned_stats, indent=2)
    if options.output:
        options.output = options.output.expanduser().resolve()
        options.output.write_text(text)
        logging.info("Wrote statistics to %s", options.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
