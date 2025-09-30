"""
Analyze ranking performance per iteration for an AL run and export CSV stats.

This script measures, at each iteration, how well the model center's
ranking matches the oracle's ranking. The procedure per iteration is:
  1) Load the center (weights) for that iteration.
  2) Rank all rules by dot-product between rule vectors and center weights.
  3) Rank all rules by the oracle (from run_dir/oracle.pkl or config.json).
  4) For a list of K values, compare the model’s top‑K versus the oracle’s
     top‑K using AP@K and Recall@K. Additionally computes AP and Recall at top 1%.

Inputs
------
 - run_dir: Folder with a single AL run (contains iterations.csv and centers).
 - rules:   Path to the dataset rule CSV (e.g., mined_rules/mushroom_mnr.csv).
 - topk:    Space-separated list of K values.
 - center:  Centre reconstruction method if the center vector isn’t stored.

Output
------
Saves a CSV (default <run_dir>/ranking_stats.csv) with one row per iteration:
  iteration, top{K}_ap, top{K}_recall (for each requested K), top1pct_ap, top1pct_recall,
  centered_inscribed_radius (radius of the largest Euclidean ball centered at the model
  that fits in the iteration’s feasible polyhedron). The radius is written in scientific
  notation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import h5py  # noqa: F401
except Exception:  # pragma: no cover
    h5py = None

from gal.core.data import Dataset
from gal.oracles.oracles import Oracle  # non-linear oracle support
from gal.centers.poly_centers import (
    chebyshev_center,
    analytical_center,
    minkowski_center,
    volumetric_center,
    mse_center,
)


# ------------------------------- helpers -------------------------------------


def _read_iterations_csv(path: Path) -> List[Tuple[int, str]]:
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


def _unique_iterations(path: Path) -> List[int]:
    pairs = _read_iterations_csv(path)
    its = sorted({it for it, _ in pairs})
    return its


def _load_center_for_iteration(
    run_dir: Path, it: int, *, n_iters: int | None = None, method: str = "chebyshev"
) -> np.ndarray | None:
    """Return center vector for iteration it, or None if unavailable.

    Order:
      1) iteration_XXX/center_model.npz['center'] (or .npy)
      2) Compute from version-space constraints in final_version_space.h5 by
         taking base constraints + the first (it+1) query constraints and
         solving for the requested centre method.
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
            return np.asarray(np.load(npy), dtype=float)
        except Exception:
            pass

    # 2) Reconstruct centre from final_version_space.h5
    fvs = run_dir / "final_version_space.h5"
    if not fvs.exists():
        return None
    try:
        import h5py as _h5

        with _h5.File(fvs, "r") as h5:
            A = np.asarray(h5["A"][...], dtype=float)
            b = np.asarray(h5["b"][...], dtype=float).reshape(-1)
    except Exception:
        return None

    # Determine total iterations
    if n_iters is None:
        try:
            n_iters = sum(
                1 for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("iteration_")
            )
        except Exception:
            n_iters = None
        if not n_iters:
            it_csv = run_dir / "iterations.csv"
            try:
                pairs = _read_iterations_csv(it_csv)
                n_iters = max((x for x, _ in pairs), default=-1) + 1
            except Exception:
                n_iters = 0

    if n_iters <= 0 or it < 0:
        return None

    base = A.shape[0] - n_iters
    base = max(0, base)
    end = min(A.shape[0], base + it + 1)
    if end <= 0:
        return None

    Ai = A[:end]
    bi = b[:end]

    try:
        if method == "chebyshev":
            c, _ = chebyshev_center(Ai, bi)
            return np.asarray(c, dtype=float)
        elif method == "analytical":
            c = analytical_center(Ai, bi)
            return np.asarray(c, dtype=float)
        elif method == "minkowski":
            c, _ = minkowski_center(Ai, bi)
            return np.asarray(c, dtype=float)
        elif method == "volumetric":
            c, _ = volumetric_center(Ai, bi)
            return np.asarray(c, dtype=float)
        elif method == "mse":
            c = mse_center(Ai, bi, X=np.empty((0, Ai.shape[1])), y=np.array([]))
            return np.asarray(c, dtype=float) if c is not None else None
    except Exception:
        return None

    return None


def _load_final_constraints(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load the final version space constraints (A, b) if available.

    Returns (A, b) or (None, None) if the file is missing/unreadable.
    """
    fvs = run_dir / "final_version_space.h5"
    if not fvs.exists():
        return None, None
    try:
        import h5py as _h5

        with _h5.File(fvs, "r") as h5:
            A = np.asarray(h5["A"][...], dtype=float)
            b = np.asarray(h5["b"][...], dtype=float).reshape(-1)
        return A, b
    except Exception:
        return None, None


def _constraints_prefix_for_iteration(
    A: np.ndarray, b: np.ndarray, it: int, n_iters: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return constraints up to and including iteration `it`.

    Interpretation: final A stacks base constraints first, then one row per iteration.
    For iteration id `it` (0-based), we take rows [: base + it + 1].
    """
    if A is None or b is None or A.size == 0 or b.size == 0:
        return None, None
    base = max(0, A.shape[0] - int(n_iters))
    end = int(min(A.shape[0], base + int(it) + 1))
    if end <= 0:
        return None, None
    return A[:end], b[:end]


def _centered_inscribed_radius(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Radius of the largest Euclidean ball centered at c contained in {x | A x ≤ b}.

    r(c) = min_i (b_i − a_i^T c) / ||a_i||_2, clamped at 0 if c is infeasible.
    """
    if A is None or b is None:
        return float("nan")
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[1] == 0:
        return float("nan")
    d = A.shape[1]
    c = np.asarray(c, dtype=float).reshape(-1)
    if c.size < d:
        # pad with zeros if shorter
        c = np.pad(c, (0, d - c.size), mode="constant")
    elif c.size > d:
        # truncate extra dims if longer
        c = c[:d]
    slack = b - A @ c
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    # clamp to [0, +inf] since negative means c is outside the polyhedron
    r = float(np.min(vals)) if vals.size else float("nan")
    return max(0.0, r) if np.isfinite(r) else float("nan")


def _load_oracle_scores(run_dir: Path, ds: Dataset, Xrules: np.ndarray, n_measures: int) -> np.ndarray:
    """Load oracle from the run directory and score rules.

    Order of attempts:
      1) oracle.pkl contains a pickled Oracle object → use set_dataset + score_dataset
      2) oracle.pkl contains a dict with 'weights' → linear dot-product
      3) config.json contains 'oracle_weights' → linear dot-product
    """
    # 1) Try a pickled Oracle object
    pkl = run_dir / "oracle.pkl"
    if pkl.exists():
        try:
            import pickle

            with pkl.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, Oracle) or (
                hasattr(obj, "set_dataset") and hasattr(obj, "score_dataset")
            ):
                try:
                    obj.set_dataset(ds)  # type: ignore[attr-defined]
                except Exception:
                    pass
                scores = obj.score_dataset(ds)  # type: ignore[attr-defined]
                return np.asarray(scores, dtype=float).reshape(-1)
            if isinstance(obj, dict) and "weights" in obj:
                w = np.asarray(obj["weights"], dtype=float)[:n_measures]
                return Xrules @ w
        except Exception:
            # fall through
            pass

    # 2) Fallback to config.json with linear weights
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            import json

            with cfg.open("r", encoding="utf-8") as f:
                conf = json.load(f)
            if isinstance(conf, dict) and "oracle_weights" in conf:
                w = np.asarray(conf["oracle_weights"], dtype=float)[:n_measures]
                return Xrules @ w
        except Exception:
            pass

    raise SystemExit(
        "Cannot load oracle. Provide 'oracle.pkl' (Oracle object or weights) or "
        "'config.json' with 'oracle_weights'. For non-linear oracles that need "
        "transactions, pass --transactions so Dataset can initialize them."
    )


def _ap_at_k(pred_topk: Sequence[int], rel_set: set[int]) -> float:
    """Average Precision at K for a predicted list against a set of relevant ids.

    AP@K = (sum_{i=1..K} P@i * rel_i) / min(K, |rel_set|).
    For our usage, |rel_set| == K, but we implement the general form.
    """
    if not pred_topk:
        return float("nan")
    hits = 0
    acc = 0.0
    denom = max(1, min(len(pred_topk), len(rel_set)))
    for i, idx in enumerate(pred_topk, start=1):
        if idx in rel_set:
            hits += 1
            acc += hits / i
    return float(acc / denom)


def _recall_at_k(pred_topk: Sequence[int], rel_set: set[int]) -> float:
    if not pred_topk:
        return float("nan")
    K = max(1, min(len(pred_topk), len(rel_set)))
    hits = sum(1 for idx in pred_topk[:K] if idx in rel_set)
    return hits / float(K)


# ------------------------------- driver --------------------------------------


@dataclass
class Inputs:
    run_dir: Path
    rules_csv: Path
    transactions_csv: Path | None
    topk: List[int]
    out_csv: Path
    center_method: str = "auto"


def _normalize_center_method(name: str | None) -> str | None:
    if not name:
        return None
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


def _infer_center_method_from_config(run_dir: Path) -> str | None:
    cfg = run_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        import json
        with cfg.open("r", encoding="utf-8") as f:
            conf = json.load(f)
        name = None
        if isinstance(conf, dict):
            name = conf.get("center_name") or conf.get("center")
        return _normalize_center_method(name)
    except Exception:
        return None


def compute_ranking(inp: Inputs) -> Path:
    run_dir = inp.run_dir
    it_csv = run_dir / "iterations.csv"
    if not it_csv.is_file():
        raise SystemExit(f"Missing iterations.csv in {run_dir}")

    # --------------------------- load dataset
    ds = Dataset(
        dataset_path=inp.rules_csv,
        transactions_path=inp.transactions_csv,
        max_rows=None,  # rank all rules
    ).load()
    Xrules = ds.points.astype(float, copy=False)
    n_measures = Xrules.shape[1]

    # --------------------------- oracle ranking (once)
    oracle_scores = _load_oracle_scores(run_dir, ds, Xrules, n_measures)
    oracle_order = np.argsort(-oracle_scores)

    rows: List[Dict[str, float | int | str]] = []
    its = _unique_iterations(it_csv)
    n_iters = len(its)
    Rmax = len(Xrules)

    # Determine centre reconstruction method if needed
    center_method = inp.center_method
    if center_method == "auto":
        inferred = _infer_center_method_from_config(run_dir)
        center_method = inferred or "chebyshev"

    # Load final constraints once (if available) to compute centered radius per iteration
    A_full, B_full = _load_final_constraints(run_dir)

    for it in its:
        row: Dict[str, float | int] = {"iteration": int(it)}

        center = _load_center_for_iteration(run_dir, it, n_iters=n_iters, method=center_method)
        if center is None:
            # fill NaNs for all requested K APs and the 1% metrics
            for k in inp.topk:
                row[f"top{k}_ap"] = float("nan")
                row[f"top{k}_recall"] = float("nan")
            row["top1pct_ap"] = float("nan")
            row["top1pct_recall"] = float("nan")
            # centered radius not computable without a center
            row["centered_inscribed_radius"] = "nan"
            rows.append(row)
            continue

        w = center[:n_measures].astype(float, copy=False)
        pred_scores = Xrules @ w
        pred_order = np.argsort(-pred_scores)

        # Regular top-K APs
        for k in inp.topk:
            K = int(min(k, Rmax))
            if K <= 0:
                row[f"top{k}_ap"] = float("nan")
                row[f"top{k}_recall"] = float("nan")
                continue

            pred_topk = pred_order[:K].tolist()
            true_topk_set = set(map(int, oracle_order[:K]))
            ap = _ap_at_k(pred_topk, true_topk_set)
            row[f"top{k}_ap"] = float(ap)
            # Recall@K: fraction of oracle top-K retrieved by model top-K
            row[f"top{k}_recall"] = float(_recall_at_k(pred_topk, true_topk_set))

        # Top 1% AP and Recall
        K1 = max(1, int(np.ceil(Rmax * 0.01)))
        pred_top1 = pred_order[:K1].tolist()
        true_top1 = set(map(int, oracle_order[:K1]))
        row["top1pct_ap"] = float(_ap_at_k(pred_top1, true_top1))
        row["top1pct_recall"] = float(_recall_at_k(pred_top1, true_top1))

        # Compute centered inscribed radius for this iteration (scientific notation)
        if A_full is not None and B_full is not None:
            Ai, Bi = _constraints_prefix_for_iteration(A_full, B_full, it, n_iters)
        else:
            Ai, Bi = None, None
        rc = _centered_inscribed_radius(Ai, Bi, center) if (Ai is not None and Bi is not None) else float("nan")
        # write as scientific notation string
        row["centered_inscribed_radius"] = (f"{rc:.6e}" if np.isfinite(rc) else "nan")

        rows.append(row)

    # --------------------------- write CSV
    keys: List[str] = ["iteration"]
    for k in inp.topk:
        keys.append(f"top{k}_ap")
        keys.append(f"top{k}_recall")
    keys.extend(["top1pct_ap", "top1pct_recall", "centered_inscribed_radius"])

    inp.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with inp.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in keys})

    return inp.out_csv


def parse_args() -> Inputs:
    ap = argparse.ArgumentParser(
        description="Analyze ranking performance per iteration and export CSV stats"
    )
    ap.add_argument(
        "run_dir",
        type=str,
        help="Path to a single run directory (contains iterations.csv and centers)",
    )
    ap.add_argument(
        "--rules",
        type=str,
        required=True,
        help="Path to the dataset rules CSV (e.g., mined_rules/<ds>_mnr.csv)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=[5, 10, 20, 50],
        help="List of K values for top-K metrics",
    )
    ap.add_argument(
        "--transactions",
        type=str,
        default=None,
        help="Optional transactions CSV to initialize non-linear oracles (sep=';')",
    )
    ap.add_argument(
        "--center",
        type=str,
        choices=["auto", "chebyshev", "analytical", "minkowski", "volumetric", "mse"],
        default="auto",
        help="Centre used to reconstruct model per iteration if not stored (default: auto from config.json)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: <run_dir>/ranking_stats.csv)",
    )

    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    out_csv = Path(args.out) if args.out else (run_dir / "ranking_stats.csv")

    return Inputs(
        run_dir=run_dir,
        rules_csv=Path(args.rules),
        transactions_csv=(Path(args.transactions) if args.transactions else None),
        topk=list(args.topk),
        out_csv=out_csv,
        center_method=args.center,
    )


def main() -> None:  # pragma: no cover
    inp = parse_args()
    out = compute_ranking(inp)
    print(f"Saved ranking statistics → {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
