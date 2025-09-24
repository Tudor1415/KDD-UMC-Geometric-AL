from pathlib import Path


def test_experiments_rq1_scaffold_present():
    base = Path(__file__).resolve().parents[1]
    cfg = base / "experiments" / "rq1" / "config.sample.yaml"
    runner = base / "experiments" / "rq1" / "run.py"
    assert cfg.exists(), "Sample config YAML missing"
    assert runner.exists(), "RQ1 runner script missing"
    text = cfg.read_text(encoding="utf-8")
    assert "methods:" in text
    assert "dual_kdtree_bnb:" in text
    assert "balltree_bnb:" in text
    assert "strategy:" in text, "Strategy should be configurable in YAML"

