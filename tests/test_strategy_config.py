import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gal.search.strategies import get_strategy, LowerBoundVisitStrategy, DiversityVisitStrategy


def test_get_strategy_lower_bound():
    s = get_strategy("lower_bound")
    assert isinstance(s, LowerBoundVisitStrategy)


def test_get_strategy_diversity():
    X = np.random.default_rng(0).normal(size=(10, 3))
    s = get_strategy("diversity", queries=X)
    assert isinstance(s, DiversityVisitStrategy)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("does-not-exist")

