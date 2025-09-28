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


def get_oracle(name: str, d: int, rng: np.random.Generator) -> Callable[[np.ndarray, np.ndarray], int]:
    key = (name or "linear_equal").strip().lower()
    if key == "linear_equal":
        w = np.ones(d, dtype=float) / float(d)
        return _sign_from_w(w)
    if key == "linear_simplex":
        u = rng.random(d)
        s = float(u.sum()) or 1.0
        return _sign_from_w(u / s)
    if key == "linear_random":
        u = np.abs(rng.standard_normal(d))
        s = float(u.sum()) or 1.0
        return _sign_from_w(u / s)
    if key == "linear_axis_0":
        e = np.zeros(d, dtype=float); e[0] = 1.0
        return _sign_from_w(e)
    if key == "linear_axis_last":
        e = np.zeros(d, dtype=float); e[d-1] = 1.0
        return _sign_from_w(e)
    if key.startswith("linear_axis_"):
        try:
            k = int(key.split("_")[-1])
        except Exception as _:
            raise ValueError(f"Invalid oracle name '{name}': expected linear_axis_<k> with integer k")
        if not (0 <= k < d):
            raise ValueError(f"Invalid axis index {k} for dimension {d}")
        e = np.zeros(d, dtype=float); e[k] = 1.0
        return _sign_from_w(e)
    raise ValueError(f"Unknown oracle name: {name}")

