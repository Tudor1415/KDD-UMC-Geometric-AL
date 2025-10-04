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
from gal.search.strategies import DiversityVisitStrategy, LowerBoundVisitStrategy
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


def test_balltree_visit_strategies_agree_on_tau():
    X, wc = _make_dataset(n=160, d=6, seed=1)
    tau_pos = 1e-6  # positive threshold; with duplicate pair, both must satisfy
    tau_neg = -1e-6  # impossible threshold; both must fail to satisfy

    ball_tree = build_tree(X, method="two_pivot")

    # Lower-bound first
    eng_lb = Search(strategy=LowerBoundVisitStrategy())
    i_lb, j_lb, d_lb = eng_lb.search_pair(ball_tree, X, wc, tau=tau_pos)

    # Diversity-driven
    eng_div = Search(strategy=DiversityVisitStrategy())
    i_div, j_div, d_div = eng_div.search_pair(ball_tree, X, wc, tau=tau_pos)

    assert np.isfinite(d_lb) and d_lb <= tau_pos + 1e-12
    assert np.isfinite(d_div) and d_div <= tau_pos + 1e-12
    assert i_lb is not None and j_lb is not None
    assert i_div is not None and j_div is not None

    # Now with an unsatisfiable threshold, both must conclude no pair was found
    _, _, d_lb_neg = eng_lb.search_pair(ball_tree, X, wc, tau=tau_neg)
    _, _, d_div_neg = eng_div.search_pair(ball_tree, X, wc, tau=tau_neg)
    assert not np.isfinite(d_lb_neg)
    assert not np.isfinite(d_div_neg)


def test_kdtree_visit_strategies_agree_on_tau():
    X, wc = _make_dataset(n=160, d=6, seed=2)
    tau_pos = 1e-6
    tau_neg = -1e-6

    kd_tree = kd.build_tree(X, {"leaf_size": 16})

    # Kd-tree requires KdTreeBounds
    eng_lb = Search(bounder=KdTreeBounds(), strategy=LowerBoundVisitStrategy())
    eng_div = Search(bounder=KdTreeBounds(), strategy=DiversityVisitStrategy())

    i_lb, j_lb, d_lb = eng_lb.search_pair(kd_tree, X, wc, tau=tau_pos)
    i_div, j_div, d_div = eng_div.search_pair(kd_tree, X, wc, tau=tau_pos)

    assert np.isfinite(d_lb) and d_lb <= tau_pos + 1e-12
    assert np.isfinite(d_div) and d_div <= tau_pos + 1e-12
    assert i_lb is not None and j_lb is not None
    assert i_div is not None and j_div is not None

    _, _, d_lb_neg = eng_lb.search_pair(kd_tree, X, wc, tau=tau_neg)
    _, _, d_div_neg = eng_div.search_pair(kd_tree, X, wc, tau=tau_neg)
    assert not np.isfinite(d_lb_neg)
    assert not np.isfinite(d_div_neg)
