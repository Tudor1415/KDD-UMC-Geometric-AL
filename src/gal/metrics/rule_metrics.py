#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import measures

# -----------------------------------------------------------------------------
# Association-rule measures (vectorised)
# -----------------------------------------------------------------------------
MEASURE_FUNCS = [
    ("support", measures.support),
    ("confidence", measures.confidence),
    ("lift", measures.lift),
    ("cosine", measures.cosine),
    ("phi", measures.phi),
    ("kruskal", measures.kruskal),
    ("yuleQ", measures.yuleQ),
    ("added_value", measures.added_value),
    ("certainty", measures.certainty),
    ("revsupport", measures.revsupport),
]


def compute_metrics(df: pd.DataFrame, n: int, smooth_counts: float) -> pd.DataFrame:
    x = df["supportX"].to_numpy(float)
    y = df["supportY"].to_numpy(float)
    z = df["supportZ"].to_numpy(float)
    total_transactions = float(n)

    for column, func in MEASURE_FUNCS:
        df[column] = func(
            x,
            y,
            z,
            total_transactions,
            smooth_counts=smooth_counts,
        )

    return df


def main():
    ap = argparse.ArgumentParser(description="Update rule CSV with quality measures")
    ap.add_argument("csv_path", help="Path to the rule CSV to overwrite")
    ap.add_argument(
        "-N",
        "--transactions",
        type=int,
        required=True,
        help="Total number of transactions (n)",
    )
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Smoothing constant (default 0 - no smoothing)",
    )
    args = ap.parse_args()

    csv_file = Path(args.csv_path).expanduser()
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)
    existing_columns = set(df.columns)
    updated = compute_metrics(df, args.transactions, args.smooth)

    # Overwrite in place
    updated.to_csv(csv_file, index=False)
    added_columns = [c for c in updated.columns if c not in existing_columns]
    print(
        f"{csv_file} updated with {len(added_columns)} new measure columns."
    )


if __name__ == "__main__":
    main()