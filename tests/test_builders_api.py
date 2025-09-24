import sys
from pathlib import Path
import importlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gal.trees import axis_median, two_pivot, pca_ballstar, bottom_up, middle_out, disjoint_greedy
from gal.trees.common import GeometricTree, Node

API_CASES = [
    ("axis_median", axis_median.build_tree, None),
    ("axis_median", axis_median.build_tree, {"leaf_size": 12}),
    ("two_pivot", two_pivot.build_tree, None),
    ("two_pivot", two_pivot.build_tree, {"degeneracy_fallback": "axis_median"}),
    ("pca_ballstar", pca_ballstar.build_tree, None),
    ("pca_ballstar", pca_ballstar.build_tree, {"balance": "median", "pca_method": "svd"}),
    ("bottom_up", bottom_up.build_tree, None),
    ("bottom_up", bottom_up.build_tree, {"precluster_leaf_size": 3}),
    ("middle_out", middle_out.build_tree, None),
    ("disjoint_greedy", disjoint_greedy.build_tree, None),
    ("middle_out", middle_out.build_tree, {"leaf_size": 16, "k_anchor": 8, "random_state": 1}),
]


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(7)
    return rng.normal(size=(256, 5))


@pytest.mark.parametrize("builder_name,builder,override", API_CASES)
def test_builder_api(builder_name, builder, override, sample_data):
    defaults = dict(importlib.import_module(f"configs.{builder_name}").DEFAULT)
    config = dict(defaults)
    if override is not None:
        config.update(override)
    cfg_arg = None if override is None else override
    tree = builder(sample_data, cfg_arg)

    assert isinstance(tree, GeometricTree)
    assert tree.root is not None and isinstance(tree.root, Node)
    assert tree.n_samples == sample_data.shape[0]
    assert tree.n_features == sample_data.shape[1]
    expected_leaf = config.get("leaf_size", config.get("precluster_leaf_size", tree.leaf_size))
    assert tree.leaf_size == expected_leaf
    assert tree.method == builder_name
    assert tree.config["meb"] == config.get("meb", defaults.get("meb"))

