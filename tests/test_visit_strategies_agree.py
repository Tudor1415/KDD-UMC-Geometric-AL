import sys
from pathlib import Path

import numpy as np

# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gal.search import Search
from gal.search.kd_bounds import KdTreeBounds
from gal.search.strategies import LowerBoundVisitStrategy
from gal.trees import build_tree
from gal.trees import kd_tree as kd


def _make_dataset(n: int = 128, d: int = 5, seed: int = 2024):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # Guarantee a zero-distance pair under the objective by duplicating a point
    if n >= 2:
        X[n // 2] = X[0]
    wc = rng.normal(size=d)
    return X, wc


def test_balltree_lower_bound_respects_tau_threshold():
    X, wc = _make_dataset(n=160, d=6, seed=1)
    tau_pos = 1e-6
    tau_neg = -1e-6

    ball_tree = build_tree(X, method="two_pivot")
    eng = Search(strategy=LowerBoundVisitStrategy())

    i_pos, j_pos, d_pos = eng.search_pair(ball_tree, X, wc, tau=tau_pos, return_stats=False)
    assert np.isfinite(d_pos) and d_pos <= tau_pos + 1e-12
    assert i_pos is not None and j_pos is not None

    i_neg, j_neg, d_neg = eng.search_pair(ball_tree, X, wc, tau=tau_neg, return_stats=False)
    assert (i_neg, j_neg) == (None, None)
    assert not np.isfinite(d_neg)


def test_kdtree_lower_bound_matches_bounds_implementation():
    X, wc = _make_dataset(n=160, d=6, seed=2)
    tau = 1e-6

    kd_tree = kd.build_tree(X, {"leaf_size": 16})
    eng = Search(bounder=KdTreeBounds(), strategy=LowerBoundVisitStrategy())

    i_idx, j_idx, dist = eng.search_pair(kd_tree, X, wc, tau=tau, return_stats=False)
    assert i_idx is not None and j_idx is not None
    assert np.isfinite(dist)
    assert dist <= tau + 1e-12


def test_orientation_mode_reports_flag():
    X, wc = _make_dataset(n=96, d=4, seed=3)
    orientation = np.random.default_rng(4).normal(size=wc.shape)
    ball_tree = build_tree(X, method="two_pivot")

    eng = Search(strategy=LowerBoundVisitStrategy())
    i_idx, j_idx, dist, stats = eng.search_pair(
        ball_tree,
        X,
        wc,
        tau=1e-4,
        orientation=orientation,
        maximize_orientation=True,
        return_stats=True,
        collect_events=True,
    )
    assert stats.get("orientation_mode") is True
    if i_idx is not None and j_idx is not None:
        assert np.isfinite(dist)
