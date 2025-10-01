"""Linear sign oracles used by active learning experiments."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["get_oracle_weights", "get_oracle", "PickledLinearOracle"]


def _sign_from_weights(w: np.ndarray) -> Callable[[np.ndarray, np.ndarray], int]:
    w = np.asarray(w, dtype=float).reshape(-1)

    def oracle(a: np.ndarray, b: np.ndarray) -> int:
        val = float(np.dot(a - b, w))
        return 1 if val >= 0 else -1

    return oracle


def get_oracle_weights(name: str, d: int) -> np.ndarray:
    """Return a deterministic weight vector for a named linear oracle."""
    if d <= 0:
        raise ValueError("Dimension d must be positive")
    key = (name or "linear_equal").strip().lower()
    if key == "linear_equal":
        w = np.ones(d, dtype=float)
        return w / float(w.sum())
    if key == "linear_simplex":
        u = np.arange(1, d + 1, dtype=float)
        return u / float(u.sum())
    if key == "linear_random":
        seed = abs(hash((key, int(d)))) % (2**32)
        rng = np.random.default_rng(int(seed))
        u = np.abs(rng.standard_normal(d))
        s = float(u.sum()) or 1.0
        return u / s
    if key == "linear_axis_0":
        e = np.zeros(d, dtype=float)
        e[0] = 1.0
        return e
    if key == "linear_axis_last":
        e = np.zeros(d, dtype=float)
        e[d - 1] = 1.0
        return e
    if key.startswith("linear_axis_"):
        try:
            idx = int(key.split("_")[-1])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Invalid oracle name '{name}': expected linear_axis_<k> with integer k"
            ) from exc
        if not (0 <= idx < d):
            raise ValueError(f"Invalid axis index {idx} for dimension {d}")
        e = np.zeros(d, dtype=float)
        e[idx] = 1.0
        return e
    raise ValueError(f"Unknown oracle name: {name}")


def get_oracle(
    name: str,
    d: int,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[np.ndarray, np.ndarray], int]:
    """Return a sign oracle ``(a, b) -> {+1, -1}`` based on linear weights."""
    # ``rng`` is accepted for backward compatibility but ignored because
    # ``get_oracle_weights`` is deterministic by design.
    _ = rng
    weights = get_oracle_weights(name, d)
    return _sign_from_weights(weights)


class PickledLinearOracle:
    """Picklable scoring oracle wrapper using a fixed weight vector."""

    def __init__(self, name: str, weights: np.ndarray):
        self.name = str(name or "linear")
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self._d = int(self.weights.size)
        self.ds = None

    def __getstate__(self):  # pragma: no cover - simple pickling helper
        state = dict(self.__dict__)
        state["ds"] = None
        return state

    def set_dataset(self, dataset) -> None:
        self.ds = dataset

    def score_dataset(self, dataset) -> np.ndarray:
        X = dataset.points.astype(float, copy=False)
        d = min(X.shape[1], self._d)
        w = self.weights[:d]
        return X[:, :d] @ w
