import sys
from pathlib import Path

import numpy as np
import pytest

# Add the source directory to the path to import the necessary modules.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gal.trees import axis_median
from gal.search import search_pair
from gal.search.bounds import BoundsStrategy, BoundsResult
from gal.search.strategies import VisitStrategy
from gal.trees.common import Node


def brute_force_metric(X: np.ndarray, wc: np.ndarray):
    """
    A brute-force search for the pair of points with the smallest metric value.
    This provides a ground truth for testing the more efficient search algorithm.
    """
    n = X.shape[0]
    best_pair = None
    best_val = float("inf")
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = X[i] - X[j]
            denom = np.linalg.norm(diff)
            if denom == 0:
                # If the points are identical, the distance is 0.
                val = 0.0
            else:
                val = abs(np.dot(diff, wc)) / denom
            if val < best_val:
                best_val = val
                best_pair = (i, j)
    return best_pair, best_val


def test_search_pair_matches_bruteforce():
    """
    Tests if the result from search_pair matches the brute-force approach.
    """
    rng = np.random.default_rng(123)
    X = rng.normal(size=(128, 4))
    wc = rng.normal(size=4)

    tree = axis_median.build_tree(X)

    i, j, dist = search_pair(tree, X, wc, tau=float("inf"))
    pair_bf, dist_bf = brute_force_metric(X, wc)

    assert {i, j} == set(pair_bf)
    assert pytest.approx(dist_bf, rel=1e-9, abs=1e-12) == dist


def test_search_pair_respects_tau():
    """
    Tests if the search correctly terminates early when a pair is found with a
    distance less than the given tau.
    """
    rng = np.random.default_rng(321)
    X = rng.normal(size=(64, 3))
    wc = rng.normal(size=3)
    tree = axis_median.build_tree(X)

    pair_bf, dist_bf = brute_force_metric(X, wc)
    # Set tau to half the actual minimum distance.
    tau = dist_bf * 0.5

    i, j, dist = search_pair(tree, X, wc, tau=tau)
    # The search should not find any pair and return None.
    assert (i, j) == (None, None)
    assert dist == float("inf")


def test_search_with_single_point():
    """
    Tests the search function with a dataset containing only a single point.
    """
    X = np.array([[1.0, 2.0, 3.0]])
    wc = np.array([1.0, 1.0, 1.0])
    tree = axis_median.build_tree(X)
    i, j, dist = search_pair(tree, X, wc, tau=float("inf"))
    assert (i, j) == (None, None)
    assert dist == float("inf")


def test_search_with_duplicate_points():
    """
    Tests the search with a dataset that includes duplicate points.
    The search should correctly identify the duplicate pair as having the
    minimum distance of 0.
    """
    X = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]])
    wc = np.array([1.0, 1.0])
    tree = axis_median.build_tree(X)
    i, j, dist = search_pair(tree, X, wc, tau=float("inf"))
    pair_bf, dist_bf = brute_force_metric(X, wc)
    assert {i, j} == set(pair_bf)
    assert pytest.approx(dist, abs=1e-12) == dist_bf
    assert dist_bf == 0.0


def test_search_with_zero_query_vector():
    """
    Tests the search with a zero vector as the query.
    In this case, the distance for all pairs should be 0.
    The search should return the first pair it evaluates.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 3))
    wc = np.zeros(3)
    tree = axis_median.build_tree(X)

    i, j, dist = search_pair(tree, X, wc, tau=float("inf"))
    assert i is not None and j is not None
    assert dist == 0.0


def test_heap_tie_breaker_no_node_comparison():
    """
    Regression test for a heapq TypeError where Node objects were compared
    when multiple heap entries had identical (score, lb, ub).
    We force identical priorities and bounds across enqueued pairs to
    validate that a numeric tie-breaker prevents Node comparisons.
    """

    class ConstantBounder(BoundsStrategy[Node]):
        def __call__(self, a: Node, b: Node, context) -> BoundsResult:  # type: ignore[override]
            # Return identical bounds for any pair
            return BoundsResult(lower=0.5, upper=0.5)

    class ConstantStrategy(VisitStrategy[Node]):
        def setup(self, root: Node, *, data=None) -> None:  # type: ignore[override]
            return None

        def priority(self, a: Node, b: Node, bounds, mass):  # type: ignore[override]
            # Return a constant priority tuple for all pairs
            return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(7)
    # Use more than default leaf_size (=32) to ensure multiple enqueues
    X = rng.normal(size=(65, 3))
    wc = rng.normal(size=3)
    tree = axis_median.build_tree(X)

    # Should not raise; should return a valid pair matching brute force
    i, j, dist = search_pair(
        tree,
        X,
        wc,
        tau=float("inf"),
        bounder=ConstantBounder(),
        strategy=ConstantStrategy(),
    )

    # Validate against brute force to ensure correctness remains intact
    (bi, bj), bf_dist = brute_force_metric(X, wc)
    assert {i, j} == {bi, bj}
    assert pytest.approx(dist, rel=1e-9, abs=1e-12) == bf_dist
