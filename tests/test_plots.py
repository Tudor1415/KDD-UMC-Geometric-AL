import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rq1.plots import CurveCI, plot_anytime_curves, plot_scaling_bars, plot_bound_tightness_kde


def test_plot_anytime_curves_returns_figure():
    x = np.linspace(0, 1, 5)
    ci = CurveCI(x=x, median=np.linspace(0, 1, 5), low=np.zeros(5), high=np.ones(5))
    fig = plot_anytime_curves({"m1": ci, "m2": ci}, xlabel="x", ylabel="y")
    assert hasattr(fig, "savefig")


def test_plot_scaling_bars_returns_figure():
    cats = ["A", "B"]
    methods = ["m1", "m2"]
    values = np.array([[0.8, 0.6], [0.9, 0.7]])
    lows = np.array([[0.7, 0.5], [0.85, 0.6]])
    highs = np.array([[0.9, 0.7], [0.95, 0.8]])
    fig = plot_scaling_bars(cats, methods, values, lows, highs)
    assert hasattr(fig, "savefig")


def test_plot_bound_tightness_kde_returns_figure():
    rng = np.random.default_rng(0)
    gaps = rng.random(100)
    fig = plot_bound_tightness_kde({"A": gaps, "B": gaps * 0.5})
    assert hasattr(fig, "savefig")

