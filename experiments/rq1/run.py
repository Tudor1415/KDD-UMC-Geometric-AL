"""RQ1 experiment skeleton runner.

This script provides a minimal, dependency-light evaluator to:
- load a small config YAML (optional),
- build kd-tree and ball-tree,
- sample centers,
- run the generic Search with a strategy specified in config.

Note: Full plotting/aggregation is intentionally omitted to avoid external
dependencies. Extend as needed in your environment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - YAML optional (tests don't rely on it)
    yaml = None

from gal.search import Search
from gal.search.kd_bounds import KdTreeBounds
from gal.search.strategies import get_strategy
from gal.trees import axis_median
from gal.trees import kd_tree as kd


@dataclass
class Rq1Config:
    content: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Rq1Config":
        p = Path(path)
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot load YAML config.")
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return Rq1Config(content=dict(cfg))

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.content
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def positive_l1_normalized(d: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(d)
    s = u.sum()
    return (u / s) if s > 0 else np.ones(d) / float(d)


def brute_force_min(X: np.ndarray, wc: np.ndarray, eps: float) -> Tuple[Tuple[int, int] | None, float]:
    n = X.shape[0]
    best_pair = None
    best_val = float("inf")
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            denom = float(np.linalg.norm(diff))
            val = 0.0 if denom <= eps else abs(float(np.dot(diff, wc))) / denom
            if val < best_val:
                best_val = val
                best_pair = (i, j)
    return best_pair, best_val


def run_once(X: np.ndarray, cfg: Rq1Config, *, rng: np.random.Generator) -> Dict[str, Any]:
    eps = float(cfg.get("global", "epsilon", default=1e-12))

    # Build trees
    kd_leaf = int(cfg.get("methods", "dual_kdtree_bnb", "leaf_size", default=16))
    kd_tree = kd.build_tree(X, {"leaf_size": kd_leaf})

    bt_leaf = int(cfg.get("methods", "balltree_bnb", "leaf_size", default=32))
    bt_tree = axis_median.build_tree(X, {"leaf_size": bt_leaf})

    # Strategy selection
    kd_strategy_name = cfg.get("methods", "dual_kdtree_bnb", "strategy", default="lower_bound")
    bt_strategy_name = cfg.get("methods", "balltree_bnb", "strategy", default="diversity")
    kd_strategy = get_strategy(kd_strategy_name)
    bt_strategy = get_strategy(bt_strategy_name, queries=X)

    # Sample center
    c = positive_l1_normalized(X.shape[1], rng)

    # Compute oracle via brute force (small datasets only)
    _, d_star = brute_force_min(X, c, eps)

    # Run kd-tree BnB
    kd_engine = Search(bounder=KdTreeBounds(), strategy=kd_strategy)
    ki, kj, kd_best, kd_stats = kd_engine.search_pair(
        kd_tree, X, c, tau=float("inf"), return_stats=True, dominance_prune=True, eps=eps
    )

    # Run ball-tree BnB (reusing ball bounds)
    bt_engine = Search(strategy=bt_strategy)
    bi, bj, bt_best, bt_stats = bt_engine.search_pair(
        bt_tree, X, c, tau=float("inf"), return_stats=True, dominance_prune=True, eps=eps
    )

    return {
        "d_star": float(d_star),
        "kd": {"best": kd_best, "pair": (ki, kj), "stats": kd_stats},
        "bt": {"best": bt_best, "pair": (bi, bj), "stats": bt_stats},
    }


def main(config_path: str) -> None:  # pragma: no cover - convenience entry
    cfg = Rq1Config.load(config_path)
    rng = np.random.default_rng(int(cfg.get("global", "rng_seed_base", default=1729)))
    # Expect a single dataset path for simplicity
    ds = cfg.get("datasets", default=[{"paths": {"matrix_npy": None}}])[0]
    npy_path = ds["paths"]["matrix_npy"]
    X = np.load(npy_path)
    result = run_once(X, cfg, rng=rng)
    out = Path(cfg.get("global", "output_dir", default="./results/rq1"))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "last_run.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run RQ1 experiment (skeleton)")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

