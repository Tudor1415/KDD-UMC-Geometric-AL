import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gal.search.kd_bounds import KdTreeBounds
from gal.search.engine import search_pair
from gal.trees import kd_tree


def brute_force_metric(X: np.ndarray, wc: np.ndarray, eps: float = 1e-12):
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


def test_kdtree_bounds_matches_bruteforce_small():
    rng = np.random.default_rng(202)
    X = rng.normal(size=(96, 4))
    wc = rng.normal(size=4)
    tree = kd_tree.build_tree(X, {"leaf_size": 12})

    i, j, dist = search_pair(tree, X, wc, tau=float("inf"), bounder=KdTreeBounds())
    pair_bf, dist_bf = brute_force_metric(X, wc)
    assert {i, j} == set(pair_bf)
    assert pytest.approx(dist_bf, rel=1e-9, abs=1e-12) == dist


def test_kdtree_builder_carries_aabb():
    rng = np.random.default_rng(7)
    X = rng.random((64, 3))
    tree = kd_tree.build_tree(X, {"leaf_size": 8})
    # Traverse a few nodes and check bbox fields exist
    stack = [tree.root]
    touched = 0
    while stack and touched < 10:
        node = stack.pop()
        assert hasattr(node, "bbox_min") and hasattr(node, "bbox_max")
        assert hasattr(node, "ir2_min") and hasattr(node, "ir2_max")
        assert node.bbox_min.shape == (X.shape[1],)
        assert node.bbox_max.shape == (X.shape[1],)
        assert node.ir2_min <= node.ir2_max + 1e-12
        stack.extend(node.children)
        touched += 1

