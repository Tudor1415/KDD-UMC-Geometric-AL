import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gal.trees import kd_tree
from gal.search.kd_bounds import KdTreeBounds
from gal.search.engine import search_pair


def test_anytime_traces_have_expected_lengths():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    wc = rng.normal(size=4)
    tree = kd_tree.build_tree(X, {"leaf_size": 10})
    tgrid = [0.0, 0.001, 0.002]
    cgrid = [1, 10, 100]
    i, j, d, stats = search_pair(
        tree,
        X,
        wc,
        tau=float("inf"),
        return_stats=True,
        bounder=KdTreeBounds(),
        time_checkpoints=tgrid,
        calls_checkpoints=cgrid,
    )
    tr = stats["trace"]
    assert len(tr["time_grid"]) == len(tgrid)
    assert len(tr["calls_grid"]) == len(cgrid)
    assert len(tr["time_best"]) == len(tgrid)
    assert len(tr["calls_best"]) == len(cgrid)
