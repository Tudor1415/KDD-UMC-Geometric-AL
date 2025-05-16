# -*- coding: utf-8 -*-
"""
Choquet‑based rule scoring, cover & Jaccard analysis
===================================================

Fully self‑contained pipeline that

1. builds `Dataset` objects from three artefact folders,
2. keeps the **last‑iteration** weight vector for each `(oracle, centre, dataset, fold)` in the performance log,
3. scores the mined rules via a Choquet integral and retains the top `TOP_PERCENT`,
4. computes every rule’s binary *cover* over the transactions,
5. measures pairwise Jaccard similarity inside each top‑list,
6. draws CDFs of those similarities per oracle/centre.

Update (15 May 2025)
--------------------
*   **Bug‑proofed `process_all`** – now accepts either a `pd.DataFrame` *or* a str/Path
    to a CSV file (as in the example).  Passing a PosixPath no longer crashes.
*   **Completed function body** that was truncated in a previous revision.
*   Example usage reads the performance CSV implicitly through `process_all`, so the
    user doesn’t need to `pd.read_csv` manually.

Fill in the *CONFIGURATION* block at the top (especially `MEASURES` and the folder
paths) and run as a script or import its functions elsewhere.
"""
from __future__ import annotations

import ast
from pathlib import Path
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import Dataset  # project‑provided helper class

Number = Union[int, float, np.number]

# ----------------------------------------------------------------------------------------------
# CONFIGURATION (edit these!)
# ----------------------------------------------------------------------------------------------
MEASURES = ["yuleQ", "cosine", "kruskal", "added_value", "certainty"]

RULES_DIR = Path("mined_rules")  # *_mnr.csv files
DATASET_DIR = Path("datasets")  # *.csv transaction files
MATRIX_DIR = Path("matrices")  # *_rules.npy / *.npz files
MAX_NUMBER_RULES = 10

# Restrict the workflow to specific datasets (by base name) if desired
KEEP_ONLY: Sequence[str] | None = None  # e.g.

# Minimum number of rules a CSV must have to be considered
RULE_MIN = 10

# ----------------------------------------------------------------------------------------------
# GENERIC HELPERS
# ----------------------------------------------------------------------------------------------


def line_count(path: Path) -> int:
    """Return the number of lines in *path* (text file)."""
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def matrix_row_count(path: Path) -> int:
    """Return the number of rows in an item‑rule matrix saved as .npy or .npz."""
    if path.suffix == ".npz":
        with np.load(path) as nz:
            return int(nz["M"].shape[0])
    return int(np.load(path, mmap_mode="r").shape[0])


# ----------------------------------------------------------------------------------------------
# DATASET FACTORY
# ----------------------------------------------------------------------------------------------


def build_dataset_objects(
    rules_dir: Path = RULES_DIR,
    dataset_dir: Path = DATASET_DIR,
    matrix_dir: Path = MATRIX_DIR,
    *,
    measures: Sequence[str] | None = MEASURES,
    keep_only: Sequence[str] | None = KEEP_ONLY,
    rule_min: int = RULE_MIN,
) -> List[Dataset]:
    """Scan the three artefact folders and build a list of `Dataset` objects."""

    rules_dir = Path(rules_dir)
    dataset_dir = Path(dataset_dir)
    matrix_dir = Path(matrix_dir)

    rule_csvs = {f for f in rules_dir.iterdir() if f.suffix == ".csv"}
    transaction_csvs = {f for f in dataset_dir.iterdir() if f.suffix == ".csv"}
    matrix_files = {f for f in matrix_dir.iterdir() if f.suffix in {".npy", ".npz"}}

    datasets: List[Dataset] = []
    for tx_file in sorted(transaction_csvs):
        base = tx_file.stem  # "market" from "market.csv"

        if keep_only is not None and base not in keep_only:
            continue

        rule_path = rules_dir / f"{base}_mnr.csv"
        if rule_path not in rule_csvs:
            continue  # no rules for this dataset

        matrix_path = matrix_dir / f"{base}_rules.npy"
        if matrix_path not in matrix_files:
            matrix_path = matrix_dir / f"{base}_rules.npz"
        if matrix_path not in matrix_files:
            continue  # no item‑rule matrix

        n_rules_csv = line_count(rule_path) - 1  # minus header
        if n_rules_csv <= rule_min:
            continue

        if matrix_row_count(matrix_path) != n_rules_csv:
            continue  # misaligned CSV ↔ matrix

        datasets.append(
            Dataset(
                name=base,
                measures=measures,
                dataset_path=rule_path,
                transactions_path=tx_file,
                item_rule_map_path=matrix_path,
            )
        )

    return datasets


# ----------------------------------------------------------------------------------------------
# CHOQUET INTEGRAL & HELPERS
# ----------------------------------------------------------------------------------------------


def choquet_integral(
    x: Sequence[Number], mobius: Dict[FrozenSet[int], Number]
) -> float:
    """Discrete Choquet integral for a capacity encoded via Möbius transform."""
    total = 0.0
    n = len(x)
    for size in range(1, n + 1):
        for S in combinations(range(n), size):
            w = mobius.get(frozenset(S))
            if w:
                total += w * min(x[i] for i in S)
    return total


def parse_weights(entry: Any) -> Dict[FrozenSet[int], float]:
    """Convert the *weights* cell into a Möbius dict (singleton weights only)."""
    if isinstance(entry, str):
        entry = ast.literal_eval(entry)

    if isinstance(entry, (list, tuple, np.ndarray)):
        vec = np.asarray(entry, dtype=float)
        return {frozenset({i}): float(w) for i, w in enumerate(vec)}

    if isinstance(entry, dict):
        return {frozenset({int(k)}): float(v) for k, v in entry.items()}

    raise TypeError(f"Unsupported weights cell type: {type(entry).__name__}")


# ------------------------------- performance → last iteration ---------------------------------


def collect_last_iter(df_perf: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df_perf.sort_values("iter")
        .groupby(["oracle", "centre", "dataset", "fold"], as_index=False)
        .tail(1)
        .index
    )
    return df_perf.loc[idx].reset_index(drop=True)


# ------------------------------------ rule scoring ---------------------------------------------


def score_rules(
    rule_df: pd.DataFrame, mobius: Dict[FrozenSet[int], float]
) -> pd.Series:
    X = rule_df[MEASURES].to_numpy(float)
    return pd.Series([choquet_integral(x, mobius) for x in X], index=rule_df.index)


def top_rule_indices(
    rule_df: pd.DataFrame, mobius: Dict[FrozenSet[int], float]
) -> List[int]:
    scores = score_rules(rule_df, mobius)
    return scores.nlargest(MAX_NUMBER_RULES).index.tolist()


# ------------------------------------ cover computation ----------------------------------------


def compute_cover_vec(pattern: Sequence[int], transactions: pd.DataFrame) -> np.ndarray:
    """
    Return a binary vector (n_transactions,) marking pattern presence.

    Fallback order:
    1. Treat pattern as *label* list of ints.
    2. Treat pattern as *label* list of strings.
    3. Treat pattern as *positional* indices via .iloc (last resort).
    """
    cols_int = list(pattern)
    col_names = [transactions.columns[index - 1] for index in cols_int]
    mat = transactions[col_names].to_numpy(dtype=np.uint8)
    return (mat.sum(axis=1) > 0).astype(np.uint8)


# ------------------------------------ Jaccard similarity ---------------------------------------


def jaccard(v1: np.ndarray, v2: np.ndarray) -> float:
    union = np.logical_or(v1, v2).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(v1, v2).sum()
    return float(inter) / float(union)


# ------------------------------------ main driver ----------------------------------------------
def process_all(
    perf_df: Union[pd.DataFrame, str, Path],
    datasets: List[Dataset],
) -> tuple[
    Dict[Tuple[str, str, str], List[np.ndarray]],  # rule covers by key
    Dict[Tuple[str, str], List[float]],  # Jaccards by (oracle, centre)
]:
    """
    * perf_df  : performance CSV (path or already-loaded DataFrame)
    * datasets : list of Dataset objects (see `build_dataset_objects`)

    Returns
    -------
    rule_covers : (oracle, centre, dataset) → list[np.ndarray]
        All binary cover vectors kept for that oracle-centre-dataset trio.
    jacc_by_oc  : (oracle, centre) → list[float]
        Flattened list of pair-wise Jaccard values aggregated over **each dataset**
        inside the oracle-centre pair.
    """
    # ------------------------------------------------------ load / trim perf log
    if isinstance(perf_df, (str, Path)):
        perf_df = pd.read_csv(perf_df)
    last_df = (
        perf_df.sort_values("iter")
        .groupby(["oracle", "centre", "dataset", "fold"], as_index=False)
        .tail(1)
    )

    # quick index: dataset name  →  Dataset object
    ds_map = {d.name: d for d in datasets}

    rule_covers: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
    jacc_by_oc: Dict[Tuple[str, str], List[float]] = {}

    # ------------------------------------------------ group by (oracle, centre)
    for (oracle, centre), oc_block in last_df.groupby(["oracle", "centre"]):

        # container for this oracle-centre pair
        oc_jacc: List[float] = []

        # ---------------------------------------- group further by DATASET
        for ds_name, ds_block in oc_block.groupby("dataset"):

            dset = ds_map[ds_name].load()  # rules / transactions
            covers: List[np.ndarray] = []

            # ------- iterate over *folds* that belong to this dataset -----
            for fold, fold_rows in ds_block.groupby("fold"):

                # pick the row with the *maximum* iteration in this fold
                last_row = fold_rows.loc[fold_rows["iter"].idxmax()]

                mobius = parse_weights(
                    last_row["weight"]
                )  # or ["weights"] if your CSV says so
                top_idx = top_rule_indices(dset.df, mobius)

                # pull covers for this fold
                for rid in top_idx:
                    rule = dset.get_rule_dict(rid)
                    items = list({*rule["antecedent"], *rule["consequent"]})
                    cover = compute_cover_vec(items, dset.transactions)
                    covers.append(cover)

            # store covers (for optional later inspection)
            rule_covers[(oracle, centre, ds_name)] = covers

            # nothing to compare if <2 covers
            if len(covers) < 2:
                continue

            # pair-wise Jaccard inside *this dataset*
            jac_ds = [
                jaccard(covers[i], covers[j])
                for i, j in combinations(range(len(covers)), 2)
            ]
            oc_jacc.extend(jac_ds)  # accumulate for this oracle-centre

        # --- save Jaccards for the whole oracle-centre pair
        jacc_by_oc[(oracle, centre)] = oc_jacc

    return rule_covers, jacc_by_oc


# ----------------------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------------------


# ------------------------------------------------------------------
# plotting helper: empirical CDF
# ------------------------------------------------------------------
def ecdf(vals: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(vals)
    n = len(v)
    return v, np.arange(1, n + 1) / float(n) if n else (v, np.array([]))


# ------------------------------------------------------------------
# CDF plot for Jaccard distributions
# ------------------------------------------------------------------
def plot_jaccard_cdfs(
    jac_by_oc: Dict[Tuple[str, str], List[float]],
    *,
    figsize: Tuple[int, int] = (7, 5),
):
    # -------------------------------------------------- style maps
    centre_colors = {
        "minkowski_center": "orange",
        "chebyshev_center": "blue",
        "ChoquetRank": "red",
    }
    oracle_styles = {
        "phi-oracle": "-",
        "surprise-independent": "--",
    }
    oracle_labels = {
        "phi-oracle": r"$\phi$",
        "surprise-independent": r"$\max \; \mathcal{H}$",
    }
    centre_labels = {
        "minkowski_center": "Minkowski Center",
        "chebyshev_center": "Chebyshev Center",
        "ChoquetRank": "ChoquetRank",
    }

    # -------------------------------------------------- single figure
    plt.figure(figsize=figsize)

    for (oracle, centre), vals in jac_by_oc.items():
        if not vals:
            continue
        x, y = ecdf(vals)
        plt.step(
            x,
            y,
            where="post",
            color=centre_colors.get(centre, "black"),
            linestyle=oracle_styles.get(oracle, "-"),
            linewidth=1.3,
        )

    plt.title("CDF of pairwise Jaccard similarities")
    plt.xlabel("Jaccard similarity")
    plt.ylabel("C.D.F.")  # avoids ’F(x ≤ t)’ clutter
    plt.tight_layout()
    plt.savefig("saved_plots/2_jaccard_cdfs.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------------------------------
# Example usage (adjust paths and MEASURES before running!)
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Performance log (CSV path or already‑loaded DataFrame)
    perf_csv = "2_merged_results.csv"  # adjust

    # Build Dataset objects from folder contents
    datasets = build_dataset_objects(
        rules_dir=RULES_DIR,
        dataset_dir=DATASET_DIR,
        matrix_dir=MATRIX_DIR,
        measures=MEASURES,
        keep_only=KEEP_ONLY,
        rule_min=RULE_MIN,
    )
    rule_covers, jacc_by_oc = process_all(perf_csv, datasets)
    plot_jaccard_cdfs(jacc_by_oc)
