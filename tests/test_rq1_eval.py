import sys
from pathlib import Path

import numpy as np

# Make package importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rq1.eval import evaluate_dataset, MethodResult  # noqa: E402


def test_evaluate_dataset_threshold_mode_small_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 4))

    res = evaluate_dataset(
        X,
        kd_strategy_name="lower_bound",
        bt_strategy_name="lower_bound",
        time_fracs=[0.25, 0.5, 1.0],
        call_fracs=[0.5, 1.0],
        timing_repeats=1,
        eps=1e-12,
        rng=rng,
        random_pair_mode="without_replacement",
        trace_tau=10.0,          # generous threshold so A should reach 1
        trace_certify=False,     # threshold mode; no exhaustive check
    )

    # Structure checks
    assert set(res.keys()) == {"kd", "bt", "rnd"}
    for k, v in res.items():
        assert isinstance(v, MethodResult)
        # Shapes
        assert v.A_time.shape == (3,)
        assert v.A_calls.shape == (2,)
        # Ranges
        assert np.all(v.A_time >= 0) and np.all(v.A_time <= 1)
        assert np.all(v.A_calls >= 0) and np.all(v.A_calls <= 1)
    # Final point should hit 1 for the generous tau
    assert np.isclose(res["kd"].A_calls[-1], 1.0, atol=1e-9)
    assert np.isclose(res["bt"].A_calls[-1], 1.0, atol=1e-9)
    assert np.isclose(res["rnd"].A_calls[-1], 1.0, atol=1e-9)

