"""
Analyze query diversity per iteration for an AL run and export CSV stats.

This script measures how diverse the queries selected by your active
learning system are at each iteration of a single experimental run.

It supports:
- Query feature-space diversity via average pairwise cosine distance
  computed cumulatively up to each iteration i.
- Rule-cover diversity via the pairwise Jaccard distance distribution, using either:
  • a precomputed transaction×item boolean matrix (preferred), or
  • a transactions CSV (fallback), or
  • skipped if neither is provided.
- Top-k metrics per iteration for a list of k values. At iteration i,
  score all rules using the model center (weights) from that iteration
  and compute diversity among the top‑k rules.

Inputs
------
- run_dir: Folder with a single AL run (contains iterations.csv and query_vectors.h5).
- rules:   Path to the dataset rule CSV (e.g., mined_rules/mushroom_mnr.csv).
- item_rule: Path to the item–rule matrix (rules×items) as .npy or .npz.
- txn_matrix (optional): Path to a transaction×item boolean matrix (.npy/.npz).
- transactions (optional): Path to transactions CSV (used only if txn_matrix is missing).

Output
------
Saves a CSV (default <run_dir>/diversity_stats.csv) with one row per iteration.
Columns include overall metrics and, for each requested k, top-k metrics.

Notes
-----
- Query vectors are loaded from run_dir/query_vectors.h5 per the iteration rows
  in run_dir/iterations.csv and accumulated. If an iteration has multiple
  rows, they are all included for that iteration and all future cumulative
  computations.
- Mapping a query vector to a rule row uses the Dataset helper (vector hash → row).
- Covers are computed as transactions where ALL rule items are present by default.
  Use --cover any to switch to presence of ANY item.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import h5py

from gal.core.data import Dataset
from gal.centers.poly_centers import (
    chebyshev_center,
    analytical_center,
    minkowski_center,
    volumetric_center,
    mse_center,
)


# ------------------------------- helpers -------------------------------------


def _read_iterations_csv(path: Path) -> List[Tuple[int, str]]:
    """Return list of (iteration_id, query_path) from iterations.csv.

    Supports files where the same iteration_id appears multiple times.
    """
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            it_idx = header.index("iteration_id")
            qp_idx = header.index("query_path")
        except ValueError as exc:
            raise RuntimeError(
                f"iterations.csv must contain 'iteration_id' and 'query_path' columns. Got: {header}"
            ) from exc

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(it_idx, qp_idx):
                continue
            try:
                it = int(parts[it_idx])
            except Exception:
                continue
            qp = parts[qp_idx]
            rows.append((it, qp))

    return rows


    


def _group_by_iteration(pairs: Iterable[Tuple[int, str]]) -> Dict[int, List[str]]:
    groups: Dict[int, List[str]] = {}
    for it, qp in pairs:
        groups.setdefault(it, []).append(qp)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def _load_query_vectors(h5_path: Path, names: Sequence[str]) -> List[np.ndarray]:
    vecs: List[np.ndarray] = []
    with h5py.File(h5_path, "r") as h5:
        for name in names:
            if name not in h5:
                # tolerate missing; skip silently
                continue
            v = np.asarray(h5[name][...], dtype=float)
            vecs.append(v)
    return vecs


def _pairwise_cosine_mean(X: np.ndarray) -> float:
    """Average pairwise cosine distance among rows of X (n×d).

    Returns NaN if n < 2.
    """
    n = X.shape[0]
    if n < 2:
        return float("nan")
    # normalise rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # guard against zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    Y = X / norms
    # cosine similarity matrix
    S = Y @ Y.T
    # distances = 1 - similarity
    D = 1.0 - S
    tri = np.triu_indices(n, k=1)
    return float(np.mean(D[tri]))


def _cosine_stats(X: np.ndarray) -> Dict[str, float]:
    """Summary stats over pairwise cosine distances among rows of X (n×d).

    Returns dict with mean/median/p95/min/max; NaNs if n < 2.
    """
    n = X.shape[0]
    if n < 2:
        return {k: float("nan") for k in ("mean", "median", "p95", "min", "max")}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Y = X / norms
    S = Y @ Y.T
    D = 1.0 - S
    tri = np.triu_indices(n, k=1)
    v = D[tri].astype(float)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95.0)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }




def _jaccard_stats(masks: Sequence[np.ndarray]) -> Dict[str, float]:
    """Return summary stats over all pairwise Jaccard distances.

    Computes median and p95 (plus mean/min/max for reference). Returns NaN
    if fewer than 2 masks.
    """
    m = len(masks)
    if m < 2:
        return {k: float("nan") for k in ("mean", "median", "p95", "min", "max")}
    vals: List[float] = []
    for i in range(m):
        ai = masks[i]
        for j in range(i + 1, m):
            aj = masks[j]
            union = np.count_nonzero(ai | aj)
            if union == 0:
                continue
            inter = np.count_nonzero(ai & aj)
            vals.append(1.0 - (inter / union))
    if not vals:
        return {k: float("nan") for k in ("mean", "median", "p95", "min", "max")}
    v = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95.0)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def _compute_covers_from_txn_matrix(
    item_lists: Sequence[Sequence[int]],
    txn_matrix: np.ndarray,  # shape (n_tx, n_items)
    *,
    cover_mode: str = "all",
) -> List[np.ndarray]:
    """Return boolean cover masks given a transaction×item matrix.

    cover_mode: 'all' → rows where all items present, 'any' → any present.
    """
    if txn_matrix.ndim != 2:
        raise ValueError("txn_matrix must be 2-D (n_tx × n_items)")
    covers: List[np.ndarray] = []
    any_mode = cover_mode.lower().startswith("any")
    for items in item_lists:
        if not items:
            covers.append(np.zeros(txn_matrix.shape[0], dtype=bool))
            continue
        cols = np.asarray(items, dtype=int)
        # If items are 1-based, subtract 1. We assume 1-based unless 0 is present.
        if cols.min(initial=1) >= 1:
            cols = cols - 1
        cols = np.clip(cols, 0, txn_matrix.shape[1] - 1)
        slab = txn_matrix[:, cols]
        mask = np.any(slab, axis=1) if any_mode else np.all(slab, axis=1)
        covers.append(mask)
    return covers


def _compute_covers_from_transactions_csv(
    item_lists: Sequence[Sequence[int]],
    transactions_df,  # pandas DataFrame
    *,
    cover_mode: str = "all",
) -> List[np.ndarray]:
    import pandas as pd  # local import, not required if not used

    assert isinstance(transactions_df, pd.DataFrame)
    cols_all = list(transactions_df.columns)
    covers: List[np.ndarray] = []
    any_mode = cover_mode.lower().startswith("any")
    for items in item_lists:
        if not items:
            covers.append(np.zeros(len(transactions_df), dtype=bool))
            continue
        # Map item labels (assumed 1-based) to column names by position.
        idx = np.asarray(items, dtype=int)
        if idx.min(initial=1) >= 1:
            idx = idx - 1
        idx = np.clip(idx, 0, len(cols_all) - 1)
        sel_cols = [cols_all[i] for i in idx]
        slab = transactions_df[sel_cols].to_numpy(dtype=bool, copy=False)
        mask = np.any(slab, axis=1) if any_mode else np.all(slab, axis=1)
        covers.append(mask)
    return covers


# ------------------------------- driver --------------------------------------


@dataclass
class Inputs:
    run_dir: Path
    rules_csv: Path
    item_rule: Path
    txn_matrix: Path | None
    transactions_csv: Path | None
    cover_mode: str
    topk: List[int]
    out_csv: Path
    center_method: str | None = None


def _normalize_center_method(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key in {"analytic", "analytical", "analytic_center", "analytical_center", "analyticcenter", "analyticalcenter", "barrier"}:
        return "analytical"
    if key in {"chebyshev", "chebyshev_center", "chebyshevcenter", "inscribed", "largest_ball"}:
        return "chebyshev"
    if key in {"minkowski", "minkowski_center", "minkowskicenter"}:
        return "minkowski"
    if key in {"volumetric", "volumetric_center", "volumetriccenter", "john", "john_ellipsoid"}:
        return "volumetric"
    if key in {"mse"}:
        return "mse"
    return key


def _load_final_constraints(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    fvs = run_dir / "final_version_space.h5"
    if not fvs.exists():
        return None, None
    try:
        with h5py.File(fvs, "r") as h5file:
            A = np.asarray(h5file["A"][...], dtype=float)
            b = np.asarray(h5file["b"][...], dtype=float).reshape(-1)
        return A, b
    except Exception:
        return None, None


def _constraints_prefix_for_iteration(A: np.ndarray, b: np.ndarray, it: int, n_iters: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if A is None or b is None or A.size == 0 or b.size == 0:
        return None, None
    base = max(0, A.shape[0] - int(n_iters))
    end = int(min(A.shape[0], base + int(it) + 1))
    if end <= 0:
        return None, None
    return A[:end], b[:end]


def _load_center_for_iteration(run_dir: Path, it: int, *, n_iters: int | None = None, method: str = "chebyshev") -> np.ndarray | None:
    """Return center vector for iteration it if stored on disk, else None.

    Only loads on-disk artifacts; does not reconstruct from constraints.
    Expected files per iteration:
      - iteration_XXX/center_model.npz (with key 'center'), or
      - iteration_XXX/center_model.npy
    """
    it_dir = run_dir / f"iteration_{it:03d}"
    npz = it_dir / "center_model.npz"
    if npz.is_file():
        try:
            with np.load(npz) as z:
                c = z.get("center")
                if c is not None:
                    return np.asarray(c, dtype=float)
        except Exception:
            pass
    npy = it_dir / "center_model.npy"
    if npy.is_file():
        try:
            c = np.load(npy)
            return np.asarray(c, dtype=float)
        except Exception:
            pass

    # No reconstruction fallback
    return None


def compute_diversity(inp: Inputs) -> Path:
    run_dir = inp.run_dir
    it_csv = run_dir / "iterations.csv"
    qv_h5 = run_dir / "query_vectors.h5"
    if not it_csv.is_file():
        raise SystemExit(f"Missing iterations.csv in {run_dir}")
    if not qv_h5.is_file():
        raise SystemExit(f"Missing query_vectors.h5 in {run_dir}")

    # --------------------------- load dataset & matrices
    ds = Dataset(
        dataset_path=inp.rules_csv,
        item_rule_map_path=inp.item_rule,
        max_rows=None,  # load all rule features so top‑k ranking covers full set
    ).load()
    n_measures = len(ds.measures)

    txn_matrix = None
    txn_df = None
    if inp.txn_matrix is not None:
        if inp.txn_matrix.suffix == ".npz":
            with np.load(inp.txn_matrix) as nz:
                txn_matrix = nz[list(nz.keys())[0]]  # pick first array
        else:
            txn_matrix = np.load(inp.txn_matrix, mmap_mode="r")
        if txn_matrix.ndim != 2:
            raise ValueError("--txn-matrix must contain a 2-D array (n_tx × n_items)")
    elif inp.transactions_csv is not None:
        import pandas as pd

        # Most datasets in this project use ';' as delimiter
        txn_df = pd.read_csv(inp.transactions_csv, sep=';')

    # --------------------------- read iteration→query paths
    groups = _group_by_iteration(_read_iterations_csv(it_csv))
    n_iters = len(groups)

    # Optionally load final constraints for on-the-fly reconstruction when --center is given
    A_full, B_full = _load_final_constraints(run_dir)

    # No query-cover computation here; Jaccard stats are only computed over
    # top‑k rule covers per iteration.

    # --------------------------- iterate + compute stats
    rows: List[Dict[str, object]] = []

    # Cumulative containers for queries up to iteration i
    cum_vecs: List[np.ndarray] = []

    # Precompute covers helper for a rule index → items
    def rule_items(ridx: int) -> List[int]:
        if ds.item_rule_map is not None and ds.items is not None:
            row_m = ds.item_rule_map[int(ridx)]
            cols = np.flatnonzero(row_m)
            return [int(ds.items[j]) for j in cols]
        # Fall back to antecedent/cons columns
        r = ds.get_rule_dict(int(ridx))
        return list({*r["antecedent"], *r["consequent"]})  # type: ignore

    for it, qpaths in groups.items():
        # ------------------- accumulate queries up to this iteration
        it_vecs = _load_query_vectors(qv_h5, qpaths)
        # Map to rule items now and extend cumulatives
        for v in it_vecs:
            cum_vecs.append(v)

        row: Dict[str, object] = {
            "iteration": it,
            "n_queries": len(it_vecs),
        }

        # ------------------- cumulative feature-space diversity
        if len(cum_vecs) >= 2:
            X = np.vstack([v for v in cum_vecs if v.size >= n_measures])
            row["cosine_mean"] = _pairwise_cosine_mean(X)
        else:
            row["cosine_mean"] = float("nan")
        # Jaccard statistics are computed only for top‑k rule sets per iteration.
        row["jaccard_mean"] = float("nan")

        # ------------------- per-iteration top‑k metrics using model center
        # Select center: reconstruct if --center was provided; else load stored
        if inp.center_method:
            method = _normalize_center_method(inp.center_method)
            if A_full is None or B_full is None:
                raise SystemExit(
                    "Cannot reconstruct centers: final_version_space.h5 is missing or invalid. "
                    "Ensure the run contains 'A' and 'b' datasets."
                )
            Ai, Bi = _constraints_prefix_for_iteration(A_full, B_full, it, n_iters)
            if Ai is None or Bi is None:
                raise SystemExit("Cannot reconstruct center: invalid constraints sizing for iteration prefix")
            try:
                if method == "chebyshev":
                    c, _ = chebyshev_center(Ai, Bi)
                    center = np.asarray(c, dtype=float)
                elif method == "analytical":
                    center = np.asarray(analytical_center(Ai, Bi), dtype=float)
                elif method == "minkowski":
                    c, _ = minkowski_center(Ai, Bi)
                    center = np.asarray(c, dtype=float)
                elif method == "volumetric":
                    c, _ = volumetric_center(Ai, Bi)
                    center = np.asarray(c, dtype=float)
                elif method == "mse":
                    c = mse_center(Ai, Bi, X=np.empty((0, Ai.shape[1])), y=np.array([]))
                    center = np.asarray(c, dtype=float) if c is not None else None
                else:
                    raise SystemExit(f"Unknown center method: {inp.center_method}")
            except Exception as e:
                raise SystemExit(f"Failed to reconstruct center for iteration {it} with method '{inp.center_method}': {e}")
            if center is None:
                raise SystemExit(f"Center reconstruction returned None for method '{inp.center_method}' at iteration {it}")
        else:
            center = _load_center_for_iteration(run_dir, it, n_iters=n_iters, method="disk")
            if center is None:
                it_dir = run_dir / f"iteration_{it:03d}"
                npz = it_dir / "center_model.npz"
                npy = it_dir / "center_model.npy"
                raise SystemExit(
                    f"Missing center weights for iteration {it} in {it_dir}. "
                    f"Expected {npz} (key 'center') or {npy}. "
                    f"Either pass --center to reconstruct for what-if analysis, or re-run the experiment to persist centers."
                )

        if center is not None:
            w = center[:n_measures].astype(float, copy=False)
            # Rule feature matrix (n_rules × d)
            Xrules = ds.points.astype(float, copy=False)
            scores = Xrules @ w
            order = np.argsort(-scores)  # descending
            max_k = max(inp.topk) if inp.topk else 0
            top_idx = order[:max_k]

            # features for top-K rules
            Xtop = Xrules[top_idx]
            # covers for top-K rules (if possible)
            top_item_lists = [rule_items(int(i)) for i in top_idx]
            if txn_matrix is not None:
                top_covers = _compute_covers_from_txn_matrix(top_item_lists, txn_matrix, cover_mode=inp.cover_mode)
            elif txn_df is not None:
                top_covers = _compute_covers_from_transactions_csv(top_item_lists, txn_df, cover_mode=inp.cover_mode)
            else:
                top_covers = None

            for k in inp.topk:
                if k >= 2:
                    cstats = _cosine_stats(Xtop[:k])
                    row[f"top{k}_cosine_mean"] = cstats["mean"]
                    row[f"top{k}_cosine_median"] = cstats["median"]
                    row[f"top{k}_cosine_p95"] = cstats["p95"]
                else:
                    row[f"top{k}_cosine_mean"] = float("nan")
                    row[f"top{k}_cosine_median"] = float("nan")
                    row[f"top{k}_cosine_p95"] = float("nan")
                if top_covers is not None:
                    stats = _jaccard_stats(top_covers[:k])
                    # Distributional stats for Jaccard distances among all rule pairs
                    row[f"top{k}_jaccard_median"] = stats["median"]
                    row[f"top{k}_jaccard_p95"] = stats["p95"]
                    # Keep mean for reference, but primary focus is on distribution
                    row[f"top{k}_jaccard_mean"] = stats["mean"]
                else:
                    row[f"top{k}_jaccard_median"] = float("nan")
                    row[f"top{k}_jaccard_p95"] = float("nan")
                    row[f"top{k}_jaccard_mean"] = float("nan")
        else:
            # unreachable: we error above, but keep branch for safety
            raise SystemExit(f"Missing center for iteration {it}")

        rows.append(row)

    # --------------------------- write CSV
    keys: List[str] = ["iteration", "n_queries", "cosine_mean", "jaccard_mean"]
    for k in inp.topk:
        keys.append(f"top{k}_cosine_mean")
        keys.append(f"top{k}_cosine_median")
        keys.append(f"top{k}_cosine_p95")
        keys.append(f"top{k}_jaccard_mean")
        keys.append(f"top{k}_jaccard_median")
        keys.append(f"top{k}_jaccard_p95")

    inp.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with inp.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in keys})

    return inp.out_csv


def parse_args() -> Inputs:
    ap = argparse.ArgumentParser(description="Analyze query diversity per iteration and export CSV stats")
    ap.add_argument("run_dir", type=str, help="Path to a single run directory (contains iterations.csv and query_vectors.h5)")
    ap.add_argument("--rules", type=str, required=True, help="Path to the dataset rules CSV (e.g., mined_rules/<ds>_mnr.csv)")
    ap.add_argument("--item-rule", type=str, required=True, help="Path to the item–rule matrix (.npy/.npz), rows align with rules CSV")
    ap.add_argument("--txn-matrix", type=str, default=None, help="Optional transaction×item boolean matrix (.npy/.npz) for cover computation")
    ap.add_argument("--transactions", type=str, default=None, help="Optional transactions CSV (fallback if --txn-matrix is missing)")
    ap.add_argument("--cover", type=str, choices=["all", "any"], default="all", help="Cover definition: all=all items present, any=any item present")
    ap.add_argument("--topk", type=int, nargs="*", default=[5, 10, 20, 50], help="List of k values for top-k metrics")
    ap.add_argument(
        "--center",
        type=str,
        default=None,
        help="Optional: reconstruct centers with this method (analytical|chebyshev|minkowski|volumetric|mse). If omitted, use stored centers.",
    )
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: <run_dir>/diversity_stats.csv)")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    out_csv = Path(args.out) if args.out else (run_dir / "diversity_stats.csv")

    rules = Path(args.rules)
    item_rule = Path(args.item_rule)

    txn_matrix = Path(args.txn_matrix) if args.txn_matrix else None
    transactions_csv = Path(args.transactions) if args.transactions else None

    return Inputs(
        run_dir=run_dir,
        rules_csv=rules,
        item_rule=item_rule,
        txn_matrix=txn_matrix,
        transactions_csv=transactions_csv,
        cover_mode=args.cover,
        topk=list(args.topk),
        out_csv=out_csv,
        center_method=args.center,
    )


def main() -> None:  # pragma: no cover
    inp = parse_args()
    out = compute_diversity(inp)
    print(f"Saved diversity statistics → {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
