from __future__ import annotations

"""Predefined oracle factory for active-learning experiments.

Each oracle is identified by a stable string name and returns a callable
oracle(a: np.ndarray, b: np.ndarray) -> int in {+1, -1}.

Names (dimension-agnostic):
- linear_equal     : w = (1/d) * 1
- linear_simplex   : w ~ Dirichlet(1) (rng provided by caller)
- linear_random    : w ~ N(0,1) then |w| normalized by L1 (nonnegative)
- linear_axis_0    : w = e_0
- linear_axis_last : w = e_{d-1}
- linear_axis_{k}  : w = e_k (0-based index)
"""

from typing import Callable
import numpy as np


def _sign_from_w(w: np.ndarray) -> Callable[[np.ndarray, np.ndarray], int]:
    w = np.asarray(w, dtype=float).reshape(-1)
    def oracle(a: np.ndarray, b: np.ndarray) -> int:
        val = float(np.dot(a - b, w))
        return 1 if val >= 0 else -1
    return oracle


def get_oracle_weights(name: str, d: int) -> np.ndarray:
    """Deterministic weights for a named oracle.

    This function is fully reproducible given (name, d). For names that were
    previously stochastic (e.g., "linear_simplex", "linear_random"), we choose
    a deterministic mapping:
      - linear_simplex: proportional to [1,2,...,d], L1-normalized.
      - linear_random : pseudo-random but seeded from (name, d) via numpy PCG64.
    """
    key = (name or "linear_equal").strip().lower()
    if d <= 0:
        raise ValueError("Dimension d must be positive")
    if key == "linear_equal":
        w = np.ones(d, dtype=float)
        return w / float(w.sum())
    if key == "linear_simplex":
        u = np.arange(1, d + 1, dtype=float)
        return u / float(u.sum())
    if key == "linear_random":
        # Deterministic PRNG seeded by (name, d)
        seed = abs(hash((key, int(d)))) % (2**32)
        rng = np.random.default_rng(int(seed))
        u = np.abs(rng.standard_normal(d))
        s = float(u.sum()) or 1.0
        return u / s
    if key == "linear_axis_0":
        e = np.zeros(d, dtype=float); e[0] = 1.0
        return e
    if key == "linear_axis_last":
        e = np.zeros(d, dtype=float); e[d - 1] = 1.0
        return e
    if key.startswith("linear_axis_"):
        try:
            k = int(key.split("_")[-1])
        except Exception as _:
            raise ValueError(f"Invalid oracle name '{name}': expected linear_axis_<k> with integer k")
        if not (0 <= k < d):
            raise ValueError(f"Invalid axis index {k} for dimension {d}")
        e = np.zeros(d, dtype=float); e[k] = 1.0
        return e
    raise ValueError(f"Unknown oracle name: {name}")


def get_oracle(name: str, d: int, rng: np.random.Generator) -> Callable[[np.ndarray, np.ndarray], int]:
    """Return a sign oracle function using deterministic weights for reproducibility."""
    w = get_oracle_weights(name, d)
    return _sign_from_w(w)


# ----------------------------------------------------------------------------
# Picklable scoring-oracle object for downstream analysis tools
# ----------------------------------------------------------------------------
class PickledLinearOracle:
    """
    Lightweight, picklable oracle object with a scoring API.

    Implements the minimal interface expected by analysis tools:
      - set_dataset(dataset)
      - score_dataset(dataset) -> np.ndarray of scores (one per rule)

    Scoring is linear with a fixed weight vector.
    """

    def __init__(self, name: str, weights: np.ndarray):
        self.name = str(name or "linear")
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self._d = int(self.weights.size)
        self.ds = None  # attached later via set_dataset

    # make sure we don't pickle heavy dataset accidentally
    def __getstate__(self):
        state = dict(self.__dict__)
        state["ds"] = None
        return state

    def set_dataset(self, dataset) -> None:
        self.ds = dataset

    def score_dataset(self, dataset) -> np.ndarray:
        # Use dataset.points; only the first d measures if points wider
        X = dataset.points.astype(float, copy=False)
        d = min(X.shape[1], self._d)
        w = self.weights[:d]
        return X[:, :d] @ w
