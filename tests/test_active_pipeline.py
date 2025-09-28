import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_exp_oracles_mapping():
    pytest.importorskip("numpy")
    from experiments.active.exp_oracles import get_oracle

    d = 3
    rng = np.random.default_rng(0)
    # axis_0
    o0 = get_oracle("linear_axis_0", d, rng)
    assert o0(np.array([1.0, 0.0, 0.0]), np.zeros(3)) == 1
    assert o0(np.zeros(3), np.array([1.0, 0.0, 0.0])) == -1
    # equal weights
    oe = get_oracle("linear_equal", d, rng)
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0])
    assert oe(a, b) == 1


def test_search_engine_emits_events():
    from gal.trees import kd_tree
    from gal.search.engine import Search

    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 3))
    wc = rng.normal(size=3)
    tree = kd_tree.build_tree(X, {"leaf_size": 8})
    eng = Search()
    i, j, d, stats = eng.search_pair(
        tree,
        X,
        wc,
        tau=float("inf"),
        return_stats=True,
        collect_events=True,
    )
    trace = stats.get("trace", {})
    events = trace.get("events")
    assert events is not None
    assert isinstance(events, list)
    # At least creation event entries should exist
    assert len(events) >= 1
    # Check schema keys on one event
    e0 = events[0]
    for k in ("event_type", "node_id", "parent_id", "timestamp", "lower_bound", "upper_bound"):
        assert k in e0


def test_run_all_generates_outputs(tmp_path: Path):
    # Be robust to environments without compatible h5py
    try:
        import importlib
        h5py = importlib.import_module("h5py")
    except Exception:
        pytest.skip("h5py unavailable or incompatible in test environment")
    from experiments.active.run import ALConfig, run_all

    cfg = {
        "global": {"output_root": str(tmp_path), "seed": 1, "max_points": 128},
        "experiment": {
            "dataset_name": "SYNTH",
            "oracle_name": "linear_equal",
            "center_name": "AnalyticCenter",
            "active_learning_budget": 1,
        },
        "paths": {},
        "algorithm_parameters": {
            "leaf_size": 16,
            "search_strategies": ["lower_bound"],
            "tree_build_methods": {"kdtree": ["kd_tree"], "balltree": ["disjoint_greedy"]},
        },
        "oracles": {"names": ["linear_equal"]},
        "logging": {"search_events": True},
    }
    out_dir = run_all(ALConfig(raw=cfg))
    assert Path(out_dir).exists()
    # Expect at least one run directory under tmp_path
    runs = [p for p in Path(tmp_path).iterdir() if p.is_dir()]
    assert len(runs) >= 1
    run_dir = runs[0]
    # Required files per NOTES/experiments/general.md
    for fn in ("config.json", "tree.h5", "query_vectors.h5", "iterations.csv", "final_version_space.h5"):
        assert (run_dir / fn).exists()
    # Check per-iteration folder exists
    it0 = run_dir / "iteration_000"
    assert (it0 / "search_trace.h5").exists()
    assert (it0 / "center_model.npy").exists()
    # Validate final_version_space.h5 structure
    with h5py.File(run_dir / "final_version_space.h5", "r") as h5:
        assert "A" in h5 and "b" in h5
        A = h5["A"][...]
        b = h5["b"][...]
        assert A.shape[0] == b.shape[0]
        assert b.shape[1] == 1
    # Basic config.json sanity
    cfg_json = json.loads((run_dir / "config.json").read_text())
    assert cfg_json.get("tree_family") in {"kdtree", "balltree"}
    assert cfg_json.get("search_strategy") == "lower_bound"
