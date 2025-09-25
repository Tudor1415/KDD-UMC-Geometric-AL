import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rq1.eval import evaluate_dataset, aggregate_runs


def test_evaluate_and_aggregate_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    res = evaluate_dataset(
        X,
        kd_strategy_name="lower_bound",
        bt_strategy_name="diversity",
        time_fracs=[0.0, 0.1, 0.2],
        call_fracs=[0.05, 0.1, 0.2],
        timing_repeats=1,
        eps=1e-12,
        rng=rng,
    )
    assert set(res.keys()) == {"kd", "bt", "rnd"}
    runs = [res, res]
    agg = aggregate_runs(runs, ci_level=0.9)
    assert "kd" in agg and "A_time" in agg["kd"]
