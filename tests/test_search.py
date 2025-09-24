import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.trees import axis_median
from src.trees.search import search_pair


def brute_force_metric(X: np.ndarray, wc: np.ndarray):
    n = X.shape[0]
    best_pair = None
    best_val = float("inf")
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            denom = np.linalg.norm(diff)
            if denom == 0:
                val = 0.0
            else:
                val = abs(np.dot(diff, wc)) / denom
            if val < best_val:
                best_val = val
                best_pair = (i, j)
    return best_pair, best_val


def test_search_pair_matches_bruteforce():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(128, 4))
    wc = rng.normal(size=4)

    tree = axis_median.build_tree(X)

    i, j, dist = search_pair(tree, X, wc, tau=float("inf"))
    pair_bf, dist_bf = brute_force_metric(X, wc)

    assert {i, j} == set(pair_bf)
    assert pytest.approx(dist_bf, rel=1e-9, abs=1e-12) == dist


def test_search_pair_respects_tau():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(64, 3))
    wc = rng.normal(size=3)
    tree = axis_median.build_tree(X)

    pair_bf, dist_bf = brute_force_metric(X, wc)
    tau = dist_bf * 0.5

    i, j, dist = search_pair(tree, X, wc, tau=tau)
    assert (i, j) == (None, None)
    assert dist == float("inf")

