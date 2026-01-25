#!/usr/bin/env python3
"""Generate a synthetic rule-metric dataset for geometric active learning experiments.

The script mimics the structure of the rule mining CSVs under ``DATA/mined_rules/`` by
producing ``antecedent``/``consequent`` columns plus a configurable set of numeric
measure columns. Values are sampled uniformly at random, which makes the dataset a
handy placeholder for smoke-testing the active learning loop on bespoke shapes.

Example
-------
    PYTHONPATH=src python scripts/create_random_dataset.py \\
        --n-samples 500 --n-dim 6 --seed 1234 \\
        --output DATA/mined_rules/random_demo.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_MEASURE_NAMES: Sequence[str] = (
    "supportX",
    "supportY",
    "supportZ",
    "support",
    "confidence",
    "lift",
    "cosine",
    "phi",
    "kruskal",
    "yuleQ",
    "added_value",
    "certainty",
    "revsupport",
)


def _normalize_measure_names(n_dim: int, requested: Iterable[str] | None) -> List[str]:
    if n_dim <= 0:
        raise ValueError("Number of dimensions must be positive")

    if requested is not None:
        names = [str(name) for name in requested]
        if len(names) < n_dim:
            raise ValueError(
                f"Not enough measure names provided ({len(names)} < {n_dim})"
            )
        return names[:n_dim]

    names: List[str] = list(DEFAULT_MEASURE_NAMES[:n_dim])
    idx = 0
    while len(names) < n_dim:
        names.append(f"measure_{idx}")
        idx += 1
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a random rule metric dataset CSV."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of rows (synthetic rules) to generate.",
    )
    parser.add_argument(
        "--n-dim",
        type=int,
        required=True,
        help="Number of numeric measure columns to sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DATA/mined_rules/random_dataset.csv"),
        help="Destination CSV path (default: DATA/mined_rules/random_dataset.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Lower bound for sampled measure values (default: 0).",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=1.0,
        help="Upper bound for sampled measure values (default: 1).",
    )
    parser.add_argument(
        "--measure-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of column names for the measures.",
    )
    parser.add_argument(
        "--antecedent-prefix",
        type=str,
        default="A",
        help="Prefix for antecedent labels (default: 'A').",
    )
    parser.add_argument(
        "--consequent-prefix",
        type=str,
        default="C",
        help="Prefix for consequent labels (default: 'C').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise SystemExit("Number of samples must be positive")
    if args.n_dim <= 0:
        raise SystemExit("Number of dimensions must be positive")
    if args.min_value >= args.max_value:
        raise SystemExit("min-value must be strictly less than max-value")

    measure_names = _normalize_measure_names(args.n_dim, args.measure_names)

    rng = np.random.default_rng(args.seed)
    measures = rng.uniform(args.min_value, args.max_value, size=(args.n_samples, args.n_dim))

    data = {
        "antecedent": [f"{args.antecedent_prefix}{i}" for i in range(args.n_samples)],
        "consequent": [f"{args.consequent_prefix}{i}" for i in range(args.n_samples)],
    }
    for idx, name in enumerate(measure_names):
        data[name] = measures[:, idx]

    df = pd.DataFrame(data)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset -> {output_path}")


if __name__ == "__main__":
    main()
