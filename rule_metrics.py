from __future__ import annotations

#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Association-rule measures (vectorised)
# ─────────────────────────────────────────────────────────────
def apply_smoothing(n, x, y, z, k):
    n += 4 * k
    x += 2 * k
    y += 2 * k
    z += k
    return n, x, y, z


def confidence(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    return z / x


def lift(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    return n * z / (x * y)


def cosine(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    return z / np.sqrt(x * y)


def phi(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    xn, yn = n - x, n - y
    zn = xn - (y - z)
    return (n * z - x * y) / np.sqrt(x * y * xn * yn)


def kruskal(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    xn, yn = n - x, n - y
    max_y = np.maximum(y, yn)
    return (np.maximum(z, x - z) + np.maximum(y - z, xn - (y - z)) - max_y) / (
        n - max_y
    )


def yuleQ(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    xn, yn = n - x, n - y
    zn = xn - (y - z)
    OR = z * zn / ((x - z) * (y - z))
    return (OR - 1) / (OR + 1)


def added_value(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    return z / x - y / n


def certainty(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    val1 = (z / x - y / n) / (1 - y / n)
    val2 = (z / y - x / n) / (1 - x / n)
    return np.maximum(val1, val2)


def support(x, y, z, n, k=0.0):
    if k:
        n, x, y, z = apply_smoothing(n, x, y, z, k)
    return z / n


def revsupport(x, y, z, n, k=0.0):
    return 1 - support(x, y, z, n, k)


# ─────────────────────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, n: int, k: float) -> pd.DataFrame:
    x = df["supportX"].to_numpy(float)
    y = df["supportY"].to_numpy(float)
    z = df["supportZ"].to_numpy(float)
    N = float(n)

    df["support"] = support(x, y, z, N, k)
    df["confidence"] = confidence(x, y, z, N, k)
    df["lift"] = lift(x, y, z, N, k)
    df["cosine"] = cosine(x, y, z, N, k)
    df["phi"] = phi(x, y, z, N, k)
    df["kruskal"] = kruskal(x, y, z, N, k)
    df["yuleQ"] = yuleQ(x, y, z, N, k)
    df["added_value"] = added_value(x, y, z, N, k)
    df["certainty"] = certainty(x, y, z, N, k)
    df["revsupport"] = revsupport(x, y, z, N, k)

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
        help="Smoothing constant (default 0 – no smoothing)",
    )
    args = ap.parse_args()

    csv_file = Path(args.csv_path).expanduser()
    if not csv_file.is_file():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)
    updated = compute_metrics(df, args.transactions, args.smooth)

    # Overwrite in place
    updated.to_csv(csv_file, index=False)
    print(
        f"✔  {csv_file} updated with "
        f"{len(updated.columns) - len(df.columns)} new measure columns."
    )


if __name__ == "__main__":
    main()
