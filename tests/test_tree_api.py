import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gal.trees import AVAILABLE_METHODS, build_tree


def test_default_method_matches_axis_median():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(128, 4))

    tree = build_tree(X)
    assert tree.method == "axis_median"
    assert tree.n_samples == X.shape[0]
    assert tree.n_features == X.shape[1]


def test_dispatch_to_other_methods():
    rng = np.random.default_rng(321)
    X = rng.random((64, 3))

    tree = build_tree(X, {"degeneracy_fallback": "axis_median"}, method="two_pivot")
    assert tree.method == "two_pivot"
    assert tree.config["degeneracy_fallback"] == "axis_median"


def test_unknown_method_raises():
    X = np.zeros((10, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        build_tree(X, method="does-not-exist")


def test_available_methods_matches_keys():
    assert set(AVAILABLE_METHODS) == {
        "axis_median",
        "kd_tree",
        "two_pivot",
        "pca_ballstar",
        "bottom_up",
        "middle_out",
        "disjoint_greedy",
    }
