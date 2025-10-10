"""Exact evaluations for leaf pairs within the search."""

from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np

from ..trees.common import Node
from .context import SearchContext


def _tensordot(xp, a, b, axes):
    if xp is np:
        return np.tensordot(a, b, axes=axes)
    if isinstance(axes, tuple):
        axes = ([axes[0]] if isinstance(axes[0], int) else list(axes[0]),
                 [axes[1]] if isinstance(axes[1], int) else list(axes[1]))
    return xp.tensordot(a, b, dims=axes)


def objective_value(p, q, wc, eps: float, backend) -> float:
    xp = backend.xp
    diff = p - q
    denom = backend.scalar(xp.linalg.norm(diff))
    if denom <= eps:
        return 0.0
    num = backend.scalar(xp.abs(xp.dot(diff, wc)))
    return num / denom


def exact_leaf_eval(
    a: Node,
    b: Node,
    context: SearchContext,
) -> Tuple[Tuple[int, int] | None, float, int, Optional[float]]:
    backend = context.backend
    xp = backend.xp
    Ai = a.indices
    Bi = b.indices
    if Ai is None or Bi is None or Ai.size == 0 or Bi.size == 0:
        return None, float("inf"), 0, None
    XA = backend.asarray(context.data[Ai])
    XB = backend.asarray(context.data[Bi])
    diff = XA[:, None, :] - XB[None, :, :]
    num = xp.abs(_tensordot(xp, diff, context.wc, axes=(2, 0)))
    denom = xp.linalg.norm(diff, axis=2)
    close_mask = denom <= context.eps
    denom = xp.where(close_mask, xp.ones_like(denom), denom)
    dist = num / denom
    dist = xp.where(close_mask, xp.zeros_like(dist), dist)
    orientation_score: Optional[np.ndarray] = None
    use_orientation = bool(context.orientation_mode and context.orientation is not None)
    if use_orientation:
        orient_vec = context.orientation
        orient_norm = backend.scalar(xp.linalg.norm(orient_vec))
        if orient_norm <= context.eps:
            use_orientation = False
        else:
            orient_unit = orient_vec / orient_norm
            diff_norm = xp.linalg.norm(diff, axis=2, keepdims=True)
            diff_safe = xp.where(diff_norm <= context.eps, diff, diff / diff_norm)
            raw_scores = _tensordot(xp, diff_safe, orient_unit, axes=(2, 0))
            orientation_score = xp.where(
                close_mask,
                xp.full_like(raw_scores, float("-inf")),
                xp.abs(raw_scores),
            )
    seen = context.seen_pairs
    if seen:
        for m in range(Ai.size):
            ia = int(Ai[m])
            for n in range(Bi.size):
                ib = int(Bi[n])
                key = (ia, ib) if ia <= ib else (ib, ia)
                if key in seen:
                    dist[m, n] = float("inf")
                    if orientation_score is not None:
                        orientation_score[m, n] = float("-inf")

    evals = int(Ai.size) * int(Bi.size)

    if use_orientation and orientation_score is not None:
        feasible = dist <= context.tau + context.eps
        orientation_score = xp.where(
            feasible,
            orientation_score,
            xp.full_like(orientation_score, float("-inf")),
        )
        if not backend.bool_scalar(xp.any(xp.isfinite(orientation_score))):
            return None, float("inf"), evals, None
        flat_idx = int(backend.scalar(xp.argmax(orientation_score)))
        m_idx, n_idx = np.unravel_index(flat_idx, orientation_score.shape)
        dist_val = backend.scalar(dist[m_idx, n_idx])
        orient_val = backend.scalar(orientation_score[m_idx, n_idx])
        if not math.isfinite(dist_val) or dist_val > context.tau + context.eps:
            return None, float("inf"), evals, None
        return (
            (int(Ai[m_idx]), int(Bi[n_idx])),
            dist_val,
            evals,
            orient_val,
        )

    if not backend.bool_scalar(xp.any(xp.isfinite(dist))):
        return None, float("inf"), evals, None
    flat_idx = int(backend.scalar(xp.argmin(dist)))
    m_idx, n_idx = np.unravel_index(flat_idx, dist.shape)
    return (int(Ai[m_idx]), int(Bi[n_idx])), backend.scalar(dist[m_idx, n_idx]), evals, None


def exact_leaf_self(
    node: Node,
    context: SearchContext,
) -> Tuple[Tuple[int, int] | None, float, int, Optional[float]]:
    backend = context.backend
    xp = backend.xp
    idx = node.indices
    if idx is None or idx.size < 2:
        return None, float("inf"), 0, None
    best_pair: Tuple[int, int] | None = None
    best_dist = float("inf")
    best_orient = float("-inf")
    evals = 0
    XA = backend.asarray(context.data[idx])
    seen = context.seen_pairs
    use_orientation = bool(context.orientation_mode and context.orientation is not None)
    orient_vec = context.orientation if use_orientation else None
    for i in range(idx.size - 1):
        pi = XA[i]
        for j in range(i + 1, idx.size):
            pj = XA[j]
            evals += 1
            ia = int(idx[i])
            ib = int(idx[j])
            key = (ia, ib) if ia <= ib else (ib, ia)
            if key in seen:
                continue
            diff_vec = pi - pj
            if backend.scalar(xp.linalg.norm(diff_vec)) <= context.eps:
                dist = 0.0
            else:
                dist = objective_value(pi, pj, context.wc, context.eps, backend)
            if not math.isfinite(dist):
                continue
            if use_orientation and orient_vec is not None:
                if dist > context.tau + context.eps:
                    continue
                orient_norm = backend.scalar(xp.linalg.norm(orient_vec))
                if orient_norm <= context.eps:
                    continue
                orient_unit = orient_vec / orient_norm
                diff_norm = backend.scalar(xp.linalg.norm(diff_vec))
                if diff_norm <= context.eps:
                    orient_val = 0.0
                else:
                    orient_val = backend.scalar(xp.abs(xp.dot(diff_vec / diff_norm, orient_unit)))
                if orient_val > best_orient + context.eps or (
                    abs(orient_val - best_orient) <= context.eps and dist < best_dist
                ):
                    best_orient = orient_val
                    best_dist = dist
                    best_pair = (int(idx[i]), int(idx[j]))
            else:
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (int(idx[i]), int(idx[j]))
    if best_pair is None:
        best_dist = float("inf")
        return None, best_dist, evals, None
    if use_orientation and orient_vec is not None:
        return best_pair, best_dist, evals, best_orient
    return best_pair, best_dist, evals, None


__all__ = [
    "objective_value",
    "exact_leaf_eval",
    "exact_leaf_self",
]
