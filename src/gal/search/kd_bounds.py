"""AABB-based bounds for kd-tree dual-tree BnB.

Implements the interval-based lower/upper bound formulas described in the RQ1
specification using per-node AABBs and precomputed Ir2 intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..trees.kd_tree import KdNode
from .bounds import BoundContext, BoundsResult, BoundsStrategy


def _proj_interval(bmin: np.ndarray, bmax: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    # Support function of AABB under linear functional <c, x>
    # min is attained by choosing L_k if c_k >= 0 else U_k; max is opposite.
    # But careful: for min, if c_k>=0 choose L, else U; for max, if c_k>=0 choose U, else L.
    ck = c
    use_min = np.where(ck >= 0.0, bmin, bmax)
    use_max = np.where(ck >= 0.0, bmax, bmin)
    s_min = float(np.dot(ck, use_min))
    s_max = float(np.dot(ck, use_max))
    return s_min, s_max


def _min_max_dist_bbox(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> Tuple[float, float]:
    # Min distance between boxes: sum over dims of gaps where intervals do not overlap
    left_gap = np.maximum(0.0, b_min - a_max)
    right_gap = np.maximum(0.0, a_min - b_max)
    gap = left_gap + right_gap
    dmin = float(np.linalg.norm(gap))

    # Max distance between boxes: farthest opposite corners per dim
    # per dim choose max(|U_a - L_b|, |U_b - L_a|)
    d1 = np.abs(a_max - b_min)
    d2 = np.abs(b_max - a_min)
    far = np.maximum(d1, d2)
    dmax = float(np.linalg.norm(far))
    return dmin, dmax


@dataclass(frozen=True)
class KdTreeBounds(BoundsStrategy[KdNode]):
    """Interval arithmetic bounds for kd-tree node pairs."""

    def __call__(self, a: KdNode, b: KdNode, context: BoundContext) -> BoundsResult:
        eps = float(context.eps)
        c = context.wc

        # Per-center projection-interval cache (I_s(N))
        cache = context.proj_cache
        if cache is None:
            a_s_min, a_s_max = _proj_interval(a.bbox_min, a.bbox_max, c)
            b_s_min, b_s_max = _proj_interval(b.bbox_min, b.bbox_max, c)
        else:
            aid = id(a)
            bid = id(b)
            if aid in cache:
                a_s_min, a_s_max = cache[aid]
            else:
                a_s_min, a_s_max = _proj_interval(a.bbox_min, a.bbox_max, c)
                cache[aid] = (a_s_min, a_s_max)
            if bid in cache:
                b_s_min, b_s_max = cache[bid]
            else:
                b_s_min, b_s_max = _proj_interval(b.bbox_min, b.bbox_max, c)
                cache[bid] = (b_s_min, b_s_max)
        # Is(A) - Is(B)
        n_min = a_s_min - b_s_max
        n_max = a_s_max - b_s_min
        # minus 1/2 Ir2(A)
        n_min -= 0.5 * a.ir2_max
        n_max -= 0.5 * a.ir2_min
        # plus 1/2 Ir2(B)
        n_min += 0.5 * b.ir2_min
        n_max += 0.5 * b.ir2_max

        dmin, dmax = _min_max_dist_bbox(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)

        # absolute interval |I_N|
        # min abs is 0 if 0 in [n_min, n_max], else min(|n_min|, |n_max|)
        if n_min <= 0.0 <= n_max:
            abs_min = 0.0
        else:
            abs_min = min(abs(n_min), abs(n_max))
        abs_max = max(abs(n_min), abs(n_max))

        lb = abs_min / (dmax + eps)
        ub = abs_max / max(dmin, eps)
        return BoundsResult(lower=float(lb), upper=float(ub))


__all__ = ["KdTreeBounds"]
