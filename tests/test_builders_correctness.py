import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trees import axis_median, two_pivot, pca_ballstar, bottom_up, middle_out
from src.trees.common import BallTree

BUILDERS = [
    ("axis_median", axis_median.build_tree),
    ("two_pivot", two_pivot.build_tree),
    ("pca_ballstar", pca_ballstar.build_tree),
    ("bottom_up", bottom_up.build_tree),
    ("middle_out", middle_out.build_tree),
]


@pytest.fixture(scope="module")
def sample_data():
    rng = np.random.default_rng(21)
    return rng.normal(size=(512, 6))


def iter_nodes(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


@pytest.mark.parametrize("name,builder", BUILDERS)
def test_leaves_respect_leaf_size(name, builder, sample_data):
    tree = builder(sample_data)
    assert isinstance(tree, BallTree)
    leaves = 0
    for node in iter_nodes(tree.root):
        if node.is_leaf:
            leaves += 1
            assert node.indices is not None
            assert node.indices.size > 0
            assert node.indices.size <= tree.leaf_size
        else:
            assert node.children, "Internal node must have children"
    assert leaves > 0


@pytest.mark.parametrize("name,builder", BUILDERS)
def test_no_empty_children(name, builder, sample_data):
    tree = builder(sample_data)
    for node in iter_nodes(tree.root):
        if not node.is_leaf:
            assert node.children
            for child in node.children:
                if child.is_leaf:
                    assert child.indices is not None and child.indices.size > 0
                else:
                    assert child.children


def test_pca_balance_prefers_smaller_sum_of_radii():
    rng = np.random.default_rng(99)
    left = rng.normal(loc=-1.0, scale=0.05, size=(60, 6))
    middle = rng.normal(loc=0.5, scale=0.02, size=(8, 6))
    right = rng.normal(loc=1.2, scale=0.05, size=(60, 6))
    X = np.vstack([left, middle, right])

    tree_median = pca_ballstar.build_tree(X, {"balance": "median", "random_state": 0})
    tree_opt = pca_ballstar.build_tree(
        X,
        {
            "balance": "argmin_sum_radii",
            "balance_max_refine_steps": 6,
            "random_state": 0,
        },
    )

    assert not tree_median.root.is_leaf
    assert not tree_opt.root.is_leaf

    sum_median = sum(child.radius for child in tree_median.root.children)
    sum_opt = sum(child.radius for child in tree_opt.root.children)
    assert sum_opt < sum_median - 1e-6