"""Exact evaluations for leaf pairs within the search."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..trees.common import Node
from .context import SearchContext


def objective_value(p: np.ndarray, q: np.ndarray, wc: np.ndarray, eps: float) -> float:
    diff = p - q
    denom = float(np.linalg.norm(diff))
    if denom <= eps:
        return 0.0
    return abs(float(np.dot(diff, wc))) / denom


def exact_leaf_eval(
    a: Node,
    b: Node,
    context: SearchContext,
) -> Tuple[Tuple[int, int] | None, float, int, Optional[float]]:
    Ai = a.indices
    Bi = b.indices
    if Ai is None or Bi is None or Ai.size == 0 or Bi.size == 0:
        return None, float("inf"), 0, None
    XA = context.data[Ai]
    XB = context.data[Bi]
    diff = XA[:, None, :] - XB[None, :, :]
    num = np.abs(np.tensordot(diff, context.wc, axes=(2, 0)))
    denom = np.linalg.norm(diff, axis=2)
    close_mask = denom <= context.eps
    denom = np.where(close_mask, 1.0, denom)
    dist = num / denom
    dist = np.where(close_mask, 0.0, dist)
    orientation_score: Optional[np.ndarray] = None
    use_orientation = bool(context.orientation_mode and context.orientation is not None)
    if use_orientation:
        orient_vec = np.asarray(context.orientation, dtype=float).reshape(-1)
        raw_scores = np.tensordot(diff, orient_vec, axes=(2, 0))
        orientation_score = np.where(close_mask, -np.inf, np.abs(raw_scores))
    seen = context.seen_pairs
    if seen:
        for m in range(Ai.size):
            ia = int(Ai[m])
            for n in range(Bi.size):
                ib = int(Bi[n])
                key = (ia, ib) if ia <= ib else (ib, ia)
                if key in seen:
                    dist[m, n] = np.inf
                    if orientation_score is not None:
                        orientation_score[m, n] = -np.inf

    evals = int(Ai.size) * int(Bi.size)

    if use_orientation and orientation_score is not None:
        feasible = dist <= context.tau + context.eps
        orientation_score = np.where(feasible, orientation_score, -np.inf)
        if not np.isfinite(orientation_score).any():
            return None, float("inf"), evals, None
        m_idx, n_idx = np.unravel_index(np.argmax(orientation_score), orientation_score.shape)
        dist_val = float(dist[m_idx, n_idx])
        orient_val = float(orientation_score[m_idx, n_idx])
        if not np.isfinite(dist_val) or dist_val > context.tau + context.eps:
            return None, float("inf"), evals, None
        return (
            (int(Ai[m_idx]), int(Bi[n_idx])),
            dist_val,
            evals,
            orient_val,
        )

    if not np.isfinite(dist).any():
        return None, float("inf"), evals, None
    m_idx, n_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return (int(Ai[m_idx]), int(Bi[n_idx])), float(dist[m_idx, n_idx]), evals, None


def exact_leaf_self(
    node: Node,
    context: SearchContext,
) -> Tuple[Tuple[int, int] | None, float, int, Optional[float]]:
    idx = node.indices
    if idx is None or idx.size < 2:
        return None, float("inf"), 0, None
    best_pair: Tuple[int, int] | None = None
    best_dist = float("inf")
    best_orient = float("-inf")
    evals = 0
    XA = context.data[idx]
    seen = context.seen_pairs
    use_orientation = bool(context.orientation_mode and context.orientation is not None)
    orient_vec = None
    if use_orientation:
        orient_vec = np.asarray(context.orientation, dtype=float).reshape(-1)
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
            if np.linalg.norm(diff_vec) <= context.eps:
                dist = 0.0
            else:
                dist = objective_value(pi, pj, context.wc, context.eps)
            if not np.isfinite(dist):
                continue
            if use_orientation and orient_vec is not None:
                if dist > context.tau + context.eps:
                    continue
                orient_val = float(abs(np.dot(diff_vec, orient_vec)))
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
