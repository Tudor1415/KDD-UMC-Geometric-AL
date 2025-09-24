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

from src.trees import axis_median, two_pivot, pca_ballstar, bottom_up, middle_out, disjoint_greedy

BUILDERS = [
    ("axis_median", axis_median.build_tree),
    ("two_pivot", two_pivot.build_tree),
    ("pca_ballstar", pca_ballstar.build_tree),
    ("bottom_up", bottom_up.build_tree),
    ("middle_out", middle_out.build_tree),
    ("disjoint_greedy", disjoint_greedy.build_tree),
]


@pytest.fixture(scope="module")
def datasets():
    rng = np.random.default_rng(42)
    random = rng.normal(size=(256, 6))
    duplicates = np.zeros((32, 6), dtype=np.float64)
    line = np.zeros((48, 6), dtype=np.float64)
    line[:, 0] = np.linspace(-5.0, 5.0, line.shape[0])
    return [
        ("random", random),
        ("duplicates", duplicates),
        ("collinear", line),
    ]


def collect_indices(node) -> np.ndarray:
    if node.is_leaf:
        if node.indices is None or node.indices.size == 0:
            return np.array([], dtype=np.int64)
        return node.indices.astype(np.int64, copy=False)
    child_indices = [collect_indices(child) for child in node.children]
    if not child_indices:
        return np.array([], dtype=np.int64)
    return np.concatenate(child_indices)


@pytest.mark.parametrize("builder_name,builder", BUILDERS)
@pytest.mark.parametrize("meb_method", ["ritter", "welzl"])
def test_nodes_enclose_points(builder_name, builder, meb_method, datasets):
    for dataset_name, X in datasets:
        cfg = {"meb": meb_method}
        if builder_name == "middle_out":
            cfg["random_state"] = 0
        tree = builder(X, cfg)
        assert tree.root is not None

        def dfs(node):
            indices = collect_indices(node)
            if indices.size > 0:
                distances = np.linalg.norm(X[indices] - node.center, axis=1)
                assert np.all(distances <= node.radius + 1e-9)
            for child in node.children:
                dfs(child)

        dfs(tree.root)
