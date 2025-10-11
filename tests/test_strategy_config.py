import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gal.search.strategies import get_strategy, LowerBoundVisitStrategy


def test_get_strategy_lower_bound():
    s = get_strategy("lower_bound")
    assert isinstance(s, LowerBoundVisitStrategy)


def test_get_strategy_alias_lb():
    s = get_strategy("lb")
    assert isinstance(s, LowerBoundVisitStrategy)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("diversity")
