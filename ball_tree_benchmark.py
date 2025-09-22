"""Benchmark ball-tree pruning and query diversity on real datasets."""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from data import Dataset
from helpers import augment_with_minimums, k_additive_constraints
from poly_centers import chebyshev_center, minkowski_center
from ball_tree import build_ball_tree, collect_kept_indices, search_pair

logger = logging.getLogger(__name__)

RULES_DIR = Path("mined_rules")
DATASETS_DIR = Path("datasets")
MATRICES_DIR = Path("matrices")
MEASURES = ["yuleQ", "cosine", "kruskal", "added_value", "certainty"]
KEEP_ONLY = {"credit", "magic", "mushroom", "tictactoe", "twitter"}

CENTER_FUNCTIONS = {
    "chebyshev_center": chebyshev_center,
    "minkowski_center": minkowski_center,
}

ADD_K = 3
BALL_TREE_K = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ball-tree pruning and query diversity."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Subset of dataset names to include (default: all recognised).",
    )
    parser.add_argument(
        "--p-values",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Leaf thresholds P to try.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 1.0],
        help="Radius scaling fractions (0 < fraction <= 1).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of AL iterations per run.",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=None,
        help="Optional cap on the number of rules sampled per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for the simulated oracle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs"),
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def discover_datasets(
    include: Optional[Sequence[str]], max_rows: Optional[int]
) -> List[Dataset]:
    if not RULES_DIR.exists():
        logger.error("Rules directory %s not found.", RULES_DIR)
        return []
    if not DATASETS_DIR.exists():
        logger.error("Dataset directory %s not found.", DATASETS_DIR)
        return []

    include_set = set(include) if include else None

    rule_csvs = {f for f in RULES_DIR.iterdir() if f.suffix == ".csv"}
    tx_csvs = {f for f in DATASETS_DIR.iterdir() if f.suffix == ".csv"}
    matrix_files = {f for f in MATRICES_DIR.iterdir() if f.suffix in (".npy", ".npz")}
    datasets: List[Dataset] = []

    for tx_path in sorted(tx_csvs):
        base = tx_path.stem
        if base not in KEEP_ONLY:
            continue
        if include_set and base not in include_set:
            continue

        rule_path = RULES_DIR / f"{base}_mnr.csv"
        if rule_path not in rule_csvs:
            logger.warning("Skipping %s: rule file %s missing.", base, rule_path.name)
            continue

        matrix_path = MATRICES_DIR / f"{base}_rules.npy"
        if matrix_path not in matrix_files:
            alt = MATRICES_DIR / f"{base}_rules.npz"
            if alt in matrix_files:
                matrix_path = alt
            else:
                logger.warning("Skipping %s: rule-item matrix missing.", base)
                continue

        ds = Dataset(
            dataset_path=rule_path,
            transactions_path=tx_path,
            item_rule_map_path=matrix_path,
            measures=MEASURES,
            name=base,
            max_rows=max_rows,
        )
        datasets.append(ds)

    if include_set:
        missing = include_set - {ds.name for ds in datasets}
        for name in sorted(missing):
            logger.warning("Requested dataset '%s' not available.", name)

    return datasets


def _chebyshev_radius(A: np.ndarray, b: np.ndarray, center: np.ndarray) -> float:
    slack = b - A @ center
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    return float(max(0.0, vals.min(initial=np.inf)))


def _project_constraint(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    v = np.asarray(vec, dtype=float)
    if v.ndim != 1 or v.size < 2:
        raise ValueError("Constraint vector must be 1-D with length >= 2.")
    return v[:-1] - v[-1], -v[-1]


def _compute_diversity(vectors: List[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    arr = np.vstack(vectors)
    if arr.shape[0] == 2:
        return float(np.linalg.norm(arr[0] - arr[1]))
    return float(pdist(arr).mean())


def run_single_benchmark(
    ds: Dataset,
    *,
    center_name: str,
    center_fn,
    P: int,
    fraction: float,
    iterations: int,
    rng: np.random.Generator,
    max_rules: Optional[int],
) -> pd.DataFrame:
    ds.load()
    raw_points = ds.points.astype(np.float64, copy=False)

    if max_rules is not None and raw_points.shape[0] > max_rules:
        sample_idx = rng.choice(raw_points.shape[0], size=max_rules, replace=False)
        raw_points = raw_points[sample_idx]
    else:
        sample_idx = None

    augmented = augment_with_minimums(raw_points, ADD_K).astype(np.float64, copy=False)
    dim = augmented.shape[1] if augmented.ndim == 2 else 1
    radius_divisor = float(fraction ** (1.0 / max(dim, 1)))
    root, _ = build_ball_tree(
        augmented,
        k=BALL_TREE_K,
        P=P,
        radius_divisor=radius_divisor,
    )
    coverage = len(collect_kept_indices(root, P=P)) / float(augmented.shape[0])

    n_single = ds.points.shape[1]
    A0, b0, _ = k_additive_constraints(n_single, ADD_K)
    A = A0.astype(np.float64, copy=True)
    b = b0.astype(np.float64, copy=True)

    center_vec = center_fn(A, b)
    if isinstance(center_vec, tuple):
        center_vec = center_vec[0]
    center_vec = np.asarray(center_vec, dtype=float)
    full_center = np.concatenate([center_vec, [1.0 - center_vec.sum()]])
    radius = _chebyshev_radius(A, b, center_vec)

    query_aug: List[np.ndarray] = []
    query_raw: List[np.ndarray] = []

    records: List[Dict[str, object]] = []

    for iteration in range(1, iterations + 1):
        Q = np.vstack(query_aug) if query_aug else None
        tau = max(radius / 8.0, 1e-13)

        i_idx, j_idx, best_distance, stats = search_pair(
            root,
            augmented,
            full_center,
            tau,
            P=P,
            Q=Q,
            return_stats=True,
        )
        if i_idx is None or j_idx is None:
            logger.info(
                "Terminating early on dataset=%s center=%s P=%d f=%.2f (no pair).",
                ds.name,
                center_name,
                P,
                fraction,
            )
            break

        a_aug = augmented[int(i_idx)]
        b_aug = augmented[int(j_idx)]
        a_raw = raw_points[int(i_idx)]
        b_raw = raw_points[int(j_idx)]

        oracle_label = int(rng.choice([-1, 1]))
        diff = a_aug - b_aug
        proj_row, proj_rhs = _project_constraint(-oracle_label * diff)
        A = np.vstack([A, proj_row.reshape(1, -1)])
        b = np.append(b, proj_rhs)

        try:
            center_vec = center_fn(A, b)
            if isinstance(center_vec, tuple):
                center_vec = center_vec[0]
            center_vec = np.asarray(center_vec, dtype=float)
            full_center = np.concatenate([center_vec, [1.0 - center_vec.sum()]])
            radius = _chebyshev_radius(A, b, center_vec)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Center computation failed at iter %d (dataset=%s, center=%s): %s",
                iteration,
                ds.name,
                center_name,
                exc,
            )
            break

        query_aug.extend([a_aug.copy(), b_aug.copy()])
        query_raw.extend([a_raw.copy(), b_raw.copy()])
        diversity = _compute_diversity(query_raw)

        records.append(
            {
                "dataset": ds.name,
                "center": center_name,
                "P": P,
                "fraction": fraction,
                "iteration": iteration,
                "i_idx": int(i_idx) if sample_idx is None else int(sample_idx[int(i_idx)]),
                "j_idx": int(j_idx) if sample_idx is None else int(sample_idx[int(j_idx)]),
                "best_distance": float(best_distance) if np.isfinite(best_distance) else None,
                "pruned_pairs": int(stats["pruned_point_pairs"]),
                "pruned_lb_pairs": int(stats["pruned_lb_point_pairs"]),
                "pruned_dom_pairs": int(stats["pruned_dom_point_pairs"]),
                "total_pairs": int(stats["total_point_pairs"]),
                "explored_pairs": int(stats["explored_point_pairs"]),
                "objective_evals": int(stats["objective_evals"]),
                "diversity": diversity,
                "coverage": coverage,
                "radius": radius,
                "tau": tau,
                "oracle_label": oracle_label,
                "best_origin": stats.get("best_origin"),
            }
        )

    return pd.DataFrame.from_records(records)


def plot_diversity_vs_pruned(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values("iteration")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(
        df_sorted["pruned_pairs"],
        df_sorted["diversity"],
        c=df_sorted["iteration"],
        cmap="viridis",
        s=70,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.plot(
        df_sorted["pruned_pairs"],
        df_sorted["diversity"],
        color="#999999",
        linewidth=0.8,
        alpha=0.6,
    )
    for _, row in df_sorted.iterrows():
        ax.annotate(
            str(int(row["iteration"])),
            (row["pruned_pairs"], row["diversity"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    coverage = df_sorted["coverage"].iloc[0]
    ax.set_xlabel("Pruned point pairs")
    ax.set_ylabel("Query set diversity (mean distance)")
    title = (
        f"{df_sorted['dataset'].iloc[0]} · {df_sorted['center'].iloc[0]} · "
        f"P={df_sorted['P'].iloc[0]} · f={df_sorted['fraction'].iloc[0]:.2f} · "
        f"coverage={coverage:.2%}"
    )
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    rng = np.random.default_rng(args.seed)
    output_dir: Path = args.output_dir
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(args.datasets, args.max_rules)
    if not datasets:
        logger.error("No datasets available – aborting.")
        return

    all_records: List[pd.DataFrame] = []

    for ds in datasets:
        for center_name, center_fn in CENTER_FUNCTIONS.items():
            for P, fraction in itertools.product(args.p_values, args.fractions):
                logger.info(
                    "Benchmarking dataset=%s center=%s P=%d fraction=%.2f",
                    ds.name,
                    center_name,
                    P,
                    fraction,
                )
                try:
                    df_run = run_single_benchmark(
                        ds,
                        center_name=center_name,
                        center_fn=center_fn,
                        P=P,
                        fraction=fraction,
                        iterations=args.iterations,
                        rng=rng,
                        max_rules=args.max_rules,
                    )
                except Exception as exc:
                    logger.exception(
                        "Run failed for dataset=%s center=%s P=%d f=%.2f: %s",
                        ds.name,
                        center_name,
                        P,
                        fraction,
                        exc,
                    )
                    continue

                if df_run.empty:
                    logger.warning(
                        "No iterations recorded for dataset=%s center=%s P=%d f=%.2f",
                        ds.name,
                        center_name,
                        P,
                        fraction,
                    )
                    continue

                all_records.append(df_run)
                plot_path = plot_dir / (
                    f"{ds.name}__{center_name}__P{P}__frac{fraction:.2f}.png"
                )
                plot_diversity_vs_pruned(df_run, plot_path)

    if not all_records:
        logger.warning("No benchmark data collected.")
        return

    all_df = pd.concat(all_records, ignore_index=True)
    csv_path = output_dir / "ball_tree_benchmark.csv"
    all_df.to_csv(csv_path, index=False)
    logger.info("Saved aggregate metrics to %s", csv_path)


if __name__ == "__main__":
    main()
