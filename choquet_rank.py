import os
import math
from pathlib import Path
import json, subprocess, tempfile

import time

# --------------------------------------------------------------------------- #
# R-side capacity identification (GLS, cumulative preferences)               #
# --------------------------------------------------------------------------- #
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple, Union

import numpy as np
from data import Dataset
from learn import project_constraint
from oracles import ObjectiveMeasureOracle, SurpriseOracle
from helpers import augment_with_minimums, k_additive_constraints

R_SCRIPT = Path("scripts/call_kappalab.R")  # ← path to your R file
EMPTY = frozenset()


def _vector_to_mobius(n: int, vec: Sequence[float]) -> Dict[FrozenSet[int], float]:
    """Flat kappalab vector → {frozenset(indices) : weight}."""
    out: Dict[FrozenSet[int], float] = {}
    idx = 0
    for size in range(1, n + 1):
        for S in combinations(range(n), size):
            if idx >= len(vec):  # safety
                return out
            out[frozenset(S)] = float(vec[idx])
            idx += 1
    return out


def identify_capacity_gls(
    alternatives: np.ndarray,
    preferences: List[List[float]],
    *,
    k: int,
    sigf: float = 0.05,
) -> Dict[FrozenSet[int], float]:
    """
    Call kappalab’s GLS routine (via the call_kappalab wrapper) and return a
    Möbius dictionary {frozenset(indices): weight}.
    """

    # --------------- build the payload ------------------------------------
    payload = {
        "alternatives": alternatives.tolist(),
        "preferences": preferences,
        "k": k,
        "approachType": "gls",
        "sigf": sigf,
    }

    # --------------- temp input JSON (TEXT mode!) -------------------------
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f_in:
        json.dump(payload, f_in)
        in_path = f_in.name

    # --------------- temp output path -------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out:
        out_path = f_out.name

    # --------------- fire the R script ------------------------------------
    subprocess.run(
        ["Rscript", str(R_SCRIPT), in_path, out_path],
        check=True,
    )

    # --------------- read Möbius coefficients -----------------------------
    with open(out_path, "r", encoding="utf-8") as fh:
        res = json.load(fh)

    n_crit = alternatives.shape[1]
    return _vector_to_mobius(n_crit, res["capacities"])


# --------------------------------------------------------------------------- #
# Choquet integral (same as before)                                           #
# --------------------------------------------------------------------------- #
Number = Union[int, float]


def choquet_integral(
    x: Sequence[Number],
    mobius: Dict[FrozenSet[int], Number],
) -> float:
    total = 0.0
    n = len(x)
    for size in range(1, n + 1):
        for S in combinations(range(n), size):
            w = mobius.get(frozenset(S))
            if w:
                total += w * min(x[i] for i in S)
    return total


# --------------------------------------------------------------------------- #
# Query heuristic (same as before)                                            #
# --------------------------------------------------------------------------- #
def select_query(
    points: np.ndarray,
    mobius: Dict[FrozenSet[int], float],
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> Tuple[int, int, np.ndarray]:
    """Return (row_i, row_j, diff = points[i]-points[j])."""
    sample_idx = rng.choice(len(points), size=sample_size, replace=False)
    P = points[sample_idx]
    vals = np.array([choquet_integral(p, mobius) for p in P])

    best, best_val = (-1, -1), float("inf")
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            d = np.linalg.norm(P[a] - P[b])
            if d == 0:
                continue
            h = abs(vals[a] - vals[b]) / d
            if h < best_val:
                best, best_val = (a, b), h

    ia, ib = best
    return (int(sample_idx[ia]), int(sample_idx[ib]))


# --------------------------------------------------------------------------- #
# Main experiment loop                                                        #
# --------------------------------------------------------------------------- #
import pandas as pd
from sklearn.model_selection import KFold
from metrics import compute_ranking_metrics


# --------------------------------------------------------------------------- #
# 4.  Chebyshev radius of the constraint polyhedron                           #
# --------------------------------------------------------------------------- #
def _chebyshev_radius(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Chebyshev radius (inscribed-ball) wrt Euclidean norm."""
    slack = b - A @ c[:-1]
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    return max(0.0, vals.min(initial=np.inf))


# --------------------------------------------------------------------------- #
# 5.  Flat-uniform initial Möbius vector                                          #
# --------------------------------------------------------------------------- #
def init_mobius_uniform(n_crit: int, k: int) -> Dict[FrozenSet[int], float]:
    """
    Return a Möbius dictionary whose coefficients are **uniformly** initialised
    so that their sum equals 1.  All non-empty subsets S ⊆ {0,…,n_crit−1}
    with |S| ≤ k receive the same weight 1/ N, where

        N = Σ_{j=1}^{min(k, n_crit)} C(n_crit, j).

    Parameters
    ----------
    n_crit : int
        Number of elementary criteria / measures (size of original vector x).
    k : int
        Additivity order.  Only subsets of size ≤ k get non-zero weight.

    Returns
    -------
    Dict[frozenset[int], float]
        Möbius transform m(S) with uniform weights summing to 1.
    """
    max_order = min(k, n_crit)
    # total number of Möbius coefficients that will be non-zero
    n_coeffs = sum(math.comb(n_crit, j) for j in range(1, max_order + 1))
    w = 1.0 / n_coeffs

    mob: Dict[FrozenSet[int], float] = {EMPTY: 0.0}
    for size in range(1, max_order + 1):
        for S in combinations(range(n_crit), size):
            mob[frozenset(S)] = w

    return mob


# --------------------------------------------------------------------------- #
# 6.  Main experiment loop  —  writes CSVs *after each fold*                  #
# --------------------------------------------------------------------------- #
def run_experiments(
    datasets,
    oracles,
    *,
    nb_folds: int,
    metric_names: List[str],
    n_iters: int,
    k_add: int,
    sample_size: int,
    rng_seed: int,
    out_csv: Path,
    diff_csv: Path,
):
    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ helpers
    def _flush(metrics_buf: List[dict], diff_buf: List[dict]) -> None:
        """Append the two buffers to their CSV files and empty them."""
        if metrics_buf:
            pd.DataFrame(metrics_buf).to_csv(
                out_csv,
                mode="a",
                header=not out_csv.exists(),
                index=False,
            )
            metrics_buf.clear()

        if diff_buf:
            pd.DataFrame(diff_buf).to_csv(
                diff_csv,
                mode="a",
                header=not diff_csv.exists(),
                index=False,
            )
            diff_buf.clear()

    # ------------------------------------------------------------------ main loops
    for ds in datasets:
        ds.load()
        pts_aug = augment_with_minimums(ds.points, k_add)
        kf = KFold(n_splits=nb_folds, shuffle=True, random_state=rng_seed)

        A_base, b_base = k_additive_constraints(len(ds.measures), k_add)

        for fold_id, (tr, te) in enumerate(kf.split(ds.points)):
            metrics_rows, diff_rows = [], []  # fresh per-fold buffers

            train_raw, test_raw = ds.points[tr], ds.points[te]
            train_aug, _ = pts_aug[tr], pts_aug[te]

            # -------- global→local bookkeeping
            alt_points: List[np.ndarray] = []
            g2l: Dict[int, int] = {}

            def _local_index(gidx: int, p: np.ndarray) -> int:
                if gidx not in g2l:
                    g2l[gidx] = len(alt_points)
                    alt_points.append(p)
                return g2l[gidx]

            # -------- fold-level state
            A, b = A_base.copy(), b_base.copy()
            prefs: List[List[float]] = []
            mobius = init_mobius_uniform(len(ds.measures), k_add)

            for oracle in oracles:
                oracle.set_dataset(ds)
                oracle_scores = oracle.score_dataset(ds)[te]

                for it in range(n_iters):
                    # ---------------- TIMING START -------------------
                    t_start = time.perf_counter()

                    # ---------------- metrics
                    preds = np.array([choquet_integral(x, mobius) for x in test_raw])
                    m = compute_ranking_metrics(metric_names, preds, oracle_scores)

                    weight_vec = np.array(
                        [
                            mobius[S]
                            for S in sorted(mobius, key=lambda s: (len(s), tuple(s)))
                        ]
                    )[
                        1:
                    ]  # drop μ(∅)

                    radius = _chebyshev_radius(A, b, weight_vec)

                    # ---------------- live debug
                    dbg = ", ".join(f"{k}={m[k]:.4f}" for k in metric_names[:2])
                    print(
                        f"[{ds.name} f{fold_id} {oracle.name} i{it:02d}] "
                        f"radius={radius:.2e}  {dbg}",
                        flush=True,
                    )

                    # ---------------- query pair
                    gi, gj = select_query(
                        train_raw, mobius, sample_size=sample_size, rng=rng
                    )
                    diff_aug = train_aug[gi] - train_aug[gj]
                    y = oracle.compare_vectors(train_raw[gi], train_raw[gj])

                    # ---------------- preference row (local indices)
                    li = _local_index(gi, train_raw[gi])
                    lj = _local_index(gj, train_raw[gj])
                    prefs.append(
                        [li + 1, lj + 1, 0.1] if y > 0 else [lj + 1, li + 1, 0.1]
                    )

                    # ---------------- constraint update
                    constr = -y * diff_aug
                    A_new, b_new = project_constraint(constr)
                    A = np.vstack([A, A_new])
                    b = np.concatenate([b, [b_new]])

                    # ---------------- re-identify capacity
                    alts_np = np.asarray(alt_points, dtype=float)
                    mobius = identify_capacity_gls(alts_np, prefs, k=k_add)

                    # ---------------- TIMING END ---------------------
                    iter_time = time.perf_counter() - t_start

                    # ---------------- metrics row --------------------
                    metrics_rows.append(
                        dict(
                            **{k: float(v) for k, v in m.items()},
                            dataset=ds.name,
                            fold=fold_id,
                            oracle=oracle.name,
                            centre="ChoquetRank",
                            iter=it,
                            radius=radius,
                            weight=json.dumps(weight_vec.tolist()),
                            iter_time=iter_time,  # NEW COLUMN
                        )
                    )

                    # ---------------- diff row ----------------------
                    diff_rows.append(
                        dict(
                            dataset=ds.name,
                            centre="ChoquetRank",
                            oracle=oracle.name,
                            diffvector=json.dumps(diff_aug.tolist()),
                            answer=int(y),
                        )
                    )

                    if y == 0:
                        break  # oracle indifferent → stop loop

                # -------- end-oracle (flush inside oracle loop to keep timings granular)
                _flush(metrics_rows, diff_rows)

            # -------- end-of-fold (ensure nothing left) ---------------
            _flush(metrics_rows, diff_rows)

    print(f"\u2713 results continuously appended to {out_csv} and {diff_csv}")


# ---------------------------------------------------------------------------
# ---------------------------  GLOBAL CONSTANTS  -----------------------------
# ---------------------------------------------------------------------------
ADD_K: int = 1  # Choquet additivity order
N_ITERS: int = 50  # maximum iterations per fold/oracle
N_FOLDS: int = 3

METRICS: List[str] = [
    "recall@10",
    "ap@10",
    "recall@p(1)",
    "ap@p(1)",
    "recall@p(5)",
    "ap@p(5)",
    "recall@p(10)",
    "ap@p(10)",
    "recall@p(50)",
    "ap@p(50)",
]

RULES_DIR = Path("mined_rules")
DATASET_DIR = Path("datasets")
MATRIX_DIR = Path("matrices")

KEEP_ONLY = {"credit", "magic", "mushroom", "tictactoe", "twitter"}
RULE_MIN = 10_000  # minimum #rules per dataset
MEASURES = ["yuleQ", "cosine", "kruskal", "added_value", "certainty"]


# ---------------------------------------------------------------------------
# ----------------------------  HELPER FUNCS  --------------------------------
# ---------------------------------------------------------------------------
def line_count(path: Path) -> int:
    """Number of *data* rows (excludes the CSV header)."""
    with path.open("rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 16), b"")) - 1


def matrix_row_count(path: Path) -> int:
    """Rows in a rule-item matrix (.npy or .npz)."""
    if path.suffix == ".npz":
        with np.load(path) as nz:
            return nz["M"].shape[0]
    return np.load(path, mmap_mode="r").shape[0]


def collect_datasets() -> List[Dataset]:
    """
    Scan the three directories and build Dataset objects that
    satisfy all checks (KEEP_ONLY, #rules ≥ RULE_MIN, row alignment).
    """
    rule_csvs = {f for f in os.listdir(RULES_DIR) if f.endswith(".csv")}
    transaction_cs = {f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")}
    matrix_files = {f for f in os.listdir(MATRIX_DIR) if f.endswith((".npy", ".npz"))}

    sets: List[Dataset] = []
    for tx_file in transaction_cs:
        base = Path(tx_file).stem  # e.g. "credit"

        if base not in KEEP_ONLY:
            continue

        rule_f = f"{base}_mnr.csv"
        matrix_f = f"{base}_rules.npy"
        if matrix_f not in matrix_files:
            matrix_f = f"{base}_rules.npz"

        if rule_f not in rule_csvs or matrix_f not in matrix_files:
            continue  # incomplete triple

        rule_path = RULES_DIR / rule_f
        matrix_path = MATRIX_DIR / matrix_f

        n_rules = line_count(rule_path)
        if n_rules < RULE_MIN:
            continue

        if matrix_row_count(matrix_path) != n_rules:
            continue  # mis-aligned; skip

        sets.append(
            Dataset(
                name=base,
                measures=MEASURES,
                dataset_path=rule_path,
                transactions_path=DATASET_DIR / tx_file,
                item_rule_map_path=matrix_path,
            )
        )

    return sets


def make_oracles():
    """Instantiate the two oracles used in the paper."""
    return [
        SurpriseOracle(prior_type="independent"),
        ObjectiveMeasureOracle("phi"),
    ]


from typing import List


# ---------------------------------------------------------------------------
# Dataset statistics ---------------------------------------------------------
# ---------------------------------------------------------------------------
def print_stats(datasets: List[Dataset]) -> None:  # use typing.List
    """Print “items / transactions / rules” for every dataset."""
    print(f"{'dataset':<12}  {'|X|':>6}  {'|D|':>9}  {'#rules':>9}")
    print("-" * 40)
    for ds in datasets:
        ds.load()
        n_items = len(ds.items) if ds.items is not None else ds.transactions.shape[1]
        n_tx = ds.n_transactions
        n_rules = len(ds.df)
        print(f"{ds.name:<12}  {n_items:6d}  {n_tx:9d}  {n_rules:9d}")
    print("-" * 40)


# ---------------------------------------------------------------------------
# -------------------------------  MAIN  -------------------------------------
# ---------------------------------------------------------------------------
def main() -> None:
    datasets = collect_datasets()
    if not datasets:
        raise RuntimeError("No datasets passed all filters – check directory paths!")

    print_stats(datasets)
    oracles = make_oracles()

    run_experiments(
        datasets=datasets,
        oracles=oracles,
        nb_folds=N_FOLDS,
        metric_names=METRICS,
        n_iters=N_ITERS,
        k_add=ADD_K,
        rng_seed=42,
        sample_size=100,
        out_csv=Path(f"{ADD_K}_ChoquetRank_results.csv"),
        diff_csv=Path(f"{ADD_K}_ChoquetRank_diffvectors.csv"),
    )


if __name__ == "__main__":
    main()
