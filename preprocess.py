#!/usr/bin/env python3
from __future__ import annotations

"""
preprocess.py
────────────────────────────
Batch-process every *_mnr.csv* inside a directory.

Modes
-----
1. Compute rule-quality *metrics* with optional Laplace smoothing.
2. 0-1 *min–max normalise* all numeric columns.
3. Export a *binary rule-item matrix* (`.npy` or compressed `.npz`).

Choose the work you want with these **mutually-exclusive** flags:

    --normalize              metrics + normalise  (default)
    --normalize-only         only normalise
    --binary-only            only export matrices (all rules)
    --binary-pareto-only     only export matrices **for Pareto rules**

Combine any mode with:

    --write-binary DIR       output matrices into DIR/
    --npz                    store compressed  .npz  instead of .npy+.items.npy
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np

if not hasattr(np, "long"):  # NumPy ≥ 2.0
    np.long = np.int_
import pandas as pd
from fast_pareto import is_pareto_front

# ────────────────────────────────────────────────────────────────────────────
# metric wrapper
try:
    from rule_metrics import compute_metrics
except ImportError as exc:
    sys.exit(f"❌  Cannot import compute_metrics from rule_metrics.py → {exc}")

# ────────────────────────────────────────────────────────────────────────────
POSSIBLE_EXTS: tuple[str, ...] = ("", ".csv", ".txt", ".data")
PARETO_MEASURES = ["yuleQ", "cosine", "kruskal", "added_value", "certainty"]


def find_dataset_file(datasets_dir: Path, name: str) -> Path | None:
    for ext in POSSIBLE_EXTS:
        p = datasets_dir / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 16), b""))


def min_max_normalise(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").columns.difference(
        ["antecedent", "consequent"]
    )
    for col in numeric:
        lo, hi = df[col].min(), df[col].max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            df[col] = 0.0
        else:
            df[col] = (df[col] - lo) / (hi - lo)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Binary matrix helpers
def _parse_items(cell: str) -> List[int]:
    cell = str(cell).strip().strip('"').strip("'")
    return [] if not cell else [int(tok) for tok in cell.split(",")]


def write_rule_item_matrix(df: pd.DataFrame, dest: Path, compress: bool = False):
    items = sorted(
        {
            it
            for col in ("antecedent", "consequent")
            for cell in df[col]
            for it in _parse_items(cell)
        }
    )
    if not items:
        raise ValueError("no items found")

    col_of = {it: j for j, it in enumerate(items)}
    M = np.zeros((len(df), len(items)), dtype=np.uint8)

    for r, (_, row) in enumerate(df.iterrows()):
        for col in ("antecedent", "consequent"):
            for it in _parse_items(row[col]):
                M[r, col_of[it]] = 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(dest.with_suffix(".npz"), M=M, items=np.array(items))
    else:
        np.save(dest.with_suffix(".npy"), M)
        np.save(dest.with_suffix(".items.npy"), np.array(items, dtype=int))


# ────────────────────────────────────────────────────────────────────────────
# CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch update rule CSVs.")
    ap.add_argument("rules_dir", type=Path, help="Folder with *_mnr.csv files")

    ap.add_argument(
        "--datasets",
        dest="datasets_dir",
        type=Path,
        default=None,
        help="Folder with raw datasets (default: <rules_dir>/../datasets)",
    )
    ap.add_argument(
        "--smooth", type=float, default=0.0, help="Laplace smoothing for metrics"
    )

    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--normalize", action="store_true", help="Compute metrics and normalise"
    )
    g.add_argument("--normalize-only", action="store_true", help="Only normalise")
    g.add_argument("--binary-only", action="store_true", help="Only export matrices")
    g.add_argument(
        "--binary-pareto-only",
        action="store_true",
        help="Only export matrices for the Pareto front of "
        "[yuleQ, cosine, kruskal, added_value, certainty]",
    )

    ap.add_argument(
        "--write-binary",
        metavar="DIR",
        type=Path,
        help="Directory for rule-item matrices",
    )
    ap.add_argument(
        "--npz", action="store_true", help="With --write-binary, save .npz not .npy"
    )
    return ap.parse_args()


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    rules_dir = args.rules_dir.expanduser().resolve()
    if not rules_dir.is_dir():
        sys.exit(f"❌  {rules_dir} is not a directory")

    datasets_dir = (
        args.datasets_dir.expanduser().resolve()
        if args.datasets_dir
        else (rules_dir.parent / "datasets").resolve()
    )

    if (
        not args.binary_only
        and not args.normalize_only
        and not args.binary_pareto_only
        and not datasets_dir.is_dir()
    ):
        sys.exit(f"❌  Datasets directory {datasets_dir} not found")

    out_bin_dir: Path | None = (
        args.write_binary.expanduser().resolve() if args.write_binary else None
    )
    if (args.binary_only or args.binary_pareto_only) and not out_bin_dir:
        sys.exit("❌  --write-binary DIR is required for the chosen mode")

    rule_files: Iterable[Path] = sorted(rules_dir.glob("*_mnr.csv"))
    if not rule_files:
        print("No *_mnr.csv files found – nothing to do.")
        return

    updated = 0
    for rule_csv in rule_files:
        name = rule_csv.stem[:-4]  # strip "_mnr"
        try:
            df = pd.read_csv(rule_csv)
        except Exception as e:
            print(f"⚠️  {rule_csv.name}: unreadable CSV ({e}) – skipped")
            continue

        # ---------------------------------------------------------------
        # 1. Pareto filtering (if requested)
        # ---------------------------------------------------------------
        if args.binary_pareto_only:
            missing = [m for m in PARETO_MEASURES if m not in df.columns]
            if missing:
                print(f"⚠️  {rule_csv.name}: missing measures {missing} – skipped")
                continue
            mask = is_pareto_front(
                df[PARETO_MEASURES].to_numpy(copy=False),  # ← convert once
                larger_is_better_objectives=list(range(len(PARETO_MEASURES))),
            )
            df = df[mask].drop_duplicates(subset=PARETO_MEASURES).reset_index(drop=True)
            if df.empty:
                print(f"⚠️  {rule_csv.name}: Pareto front empty – skipped")
                continue

        # ---------------------------------------------------------------
        # 2. Metric computation
        # ---------------------------------------------------------------
        if not (args.normalize_only or args.binary_only or args.binary_pareto_only):
            dataset_path = find_dataset_file(datasets_dir, name)
            if dataset_path is None:
                print(f"⚠️  {rule_csv.name}: dataset '{name}' not found – skipped")
                continue
            try:
                n_transactions = line_count(dataset_path)
                df = compute_metrics(df, n_transactions, args.smooth)
            except Exception as e:
                print(f"⚠️  {rule_csv.name}: metric computation failed ({e}) – skipped")
                continue

        # ---------------------------------------------------------------
        # 3. Normalisation
        # ---------------------------------------------------------------
        if args.normalize or args.normalize_only:
            df = min_max_normalise(df)
            uniq_cols = [c for c in PARETO_MEASURES if c in df.columns]
            df = df.drop_duplicates(subset=uniq_cols).reset_index(drop=True)

        # ---------------------------------------------------------------
        # 4. Write binary matrix (if requested)
        # ---------------------------------------------------------------
        if out_bin_dir:
            try:
                dest = out_bin_dir / f"{name}_rules"
                write_rule_item_matrix(df, dest, compress=args.npz)
            except Exception as e:
                print(f"⚠️  {rule_csv.name}: cannot write matrix ({e})")

        # ---------------------------------------------------------------
        # 5. Save CSV unless in “binary-only” mode
        # ---------------------------------------------------------------
        if not args.binary_only:
            try:
                df.to_csv(rule_csv, index=False)
            except Exception as e:
                print(f"⚠️  {rule_csv.name}: cannot write ({e}) – skipped")
                continue

        # ---------------------------------------------------------------
        # 6. Console tag / bookkeeping
        # ---------------------------------------------------------------
        tag: List[str] = []
        if args.binary_pareto_only:
            tag.append("pareto-bin")
        else:
            if not (args.normalize_only or args.binary_only):
                tag.append("metrics")
            if args.normalize or args.normalize_only:
                tag.append("norm")
            if out_bin_dir:
                tag.append("bin")
        print(f"✔  {rule_csv.name} updated ({'+'.join(tag) or 'no-op'})")
        updated += 1

    print(f"Done. {updated} file(s) processed in {rules_dir}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Elapsed: {time.time()-t0:.2f}s")
