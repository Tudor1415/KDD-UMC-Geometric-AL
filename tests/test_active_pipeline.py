import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_toy_dataset(path: Path, *, n_rows: int = 32, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    support = np.linspace(1.0, 0.1, n_rows)
    confidence = rng.random(n_rows)
    df = pd.DataFrame(
        {
            "antecedent": [str(i % 7) for i in range(n_rows)],
            "consequent": [str((i + 1) % 9) for i in range(n_rows)],
            "support": support,
            "confidence": confidence,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_objective_oracle_scores_rules(tmp_path: Path):
    from gal.core.data import Dataset
    from gal.oracles.oracles import ObjectiveMeasureOracle

    dataset_path = _write_toy_dataset(tmp_path / "rules.csv", n_rows=8, seed=7)
    ds = Dataset(dataset_path=dataset_path, measures=["support", "confidence"]).load()
    oracle = ObjectiveMeasureOracle("support")
    oracle.set_dataset(ds)

    best = ds.get_rule_dict(0)
    worst = ds.get_rule_dict(len(ds) - 1)
    assert oracle.compare(best, worst) == 1
    assert oracle.compare(worst, best) == -1


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
    assert len(events) >= 1
    e0 = events[0]
    for key in ("event_type", "node_id", "parent_id", "timestamp", "lower_bound", "upper_bound"):
        assert key in e0


def test_run_all_generates_outputs(tmp_path: Path):
    from gal.experiments.config import ALConfig
    from gal.experiments.runner import run_all

    data_path = _write_toy_dataset(tmp_path / "toy_rules.csv", n_rows=24, seed=3)
    output_root = tmp_path / "runs"
    cfg = {
        "global": {"output_root": str(output_root), "seed": 1, "max_points": 0},
        "experiment": {
            "dataset_name": "toy",
            "center_name": "chebyshev",
            "active_learning_budget": 1,
            "additivity_k": 1,
        },
        "oracle": {"type": "objective", "measure": "support"},
        "trees": {"ball": {"method": "two_pivot", "config": {"leaf_size": 8}}},
        "algorithm_parameters": {"search_strategy": "lower_bound"},
        "logging": {"search_events": True, "level": "ERROR", "log_every": 1},
        "datasets": [
            {
                "name": "toy",
                "paths": {"dataset_path": str(data_path)},
                "measures": ["support", "confidence"],
            }
        ],
    }

    out_dir = run_all(ALConfig(raw=cfg))
    assert Path(out_dir).exists()

    runs = [p for p in output_root.iterdir() if p.is_dir()]
    assert runs
    run_dir = runs[0]

    for fn in ("config.json", "iterations.csv", "final_version_space.npz"):
        assert (run_dir / fn).exists()
    assert (run_dir / "queries").is_dir()

    iteration_dir = run_dir / "iteration_000"
    assert iteration_dir.is_dir()
    assert any((iteration_dir / fname).exists() for fname in ("search_trace.npz", "search_trace.h5"))

    cfg_json = json.loads((run_dir / "config.json").read_text())
    assert cfg_json.get("tree_family") == "balltree"
    assert cfg_json.get("search_strategy") == "lower_bound"
