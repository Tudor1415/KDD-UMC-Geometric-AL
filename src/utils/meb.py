"""Minimum enclosing ball (MEB) implementations."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

EPS = 1e-12


def meb(points: np.ndarray, method: str = "ritter", rng=None) -> Tuple[np.ndarray, float]:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("points must be a non-empty 2D array")

    if method == "ritter":
        return ritter(pts)
    if method == "welzl":
        if rng is None:
            rng = np.random.default_rng(0)
        return welzl(pts, rng)
    raise ValueError(f"Unknown MEB method: {method}")


def ritter(points: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n, d = pts.shape
    if n == 1:
        return pts[0].copy(), 0.0

    diffs = pts - pts[0]
    dist2 = np.einsum("ij,ij->i", diffs, diffs)
    idx1 = int(np.argmax(dist2))
    p1 = pts[idx1]

    diffs = pts - p1
    dist2 = np.einsum("ij,ij->i", diffs, diffs)
    idx2 = int(np.argmax(dist2))
    p2 = pts[idx2]

    center = 0.5 * (p1 + p2)
    radius = 0.5 * float(np.linalg.norm(p1 - p2))

    def expand(C: np.ndarray, R: float) -> Tuple[np.ndarray, float]:
        center_local = C.copy()
        radius_local = float(R)
        for p in pts:
            diff = p - center_local
            dist = float(np.linalg.norm(diff))
            if dist > radius_local + EPS:
                new_radius = 0.5 * (radius_local + dist)
                if dist > 0.0:
                    center_local += ((new_radius - radius_local) / dist) * diff
                radius_local = new_radius
        return center_local, radius_local

    center, radius = expand(center, radius)
    center, radius = expand(center, radius)
    radius = max(radius, 0.0)
    return center.astype(np.float64, copy=False), float(radius)


def welzl(points: np.ndarray, rng) -> Tuple[np.ndarray, float]:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    n, d = pts.shape
    order = rng.permutation(n)

    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2 * n + 10))

    def recurse(end: int, support: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        if end == 0 or len(support) == d + 1:
            return _ball_from_support(support, d)
        point = pts[order[end - 1]]
        center, radius = recurse(end - 1, support)
        if np.linalg.norm(point - center) <= radius + EPS:
            return center, radius
        support.append(point)
        center, radius = recurse(end - 1, support)
        support.pop()
        return center, radius

    center, radius = recurse(n, [])
    return center.astype(np.float64, copy=False), float(radius)


def _ball_from_support(support: Sequence[np.ndarray], d: int) -> Tuple[np.ndarray, float]:
    if not support:
        return np.zeros(d, dtype=np.float64), 0.0
    if len(support) == 1:
        return support[0].astype(np.float64, copy=True), 0.0

    pts = np.array(support, dtype=np.float64)
    p0 = pts[0]
    A = pts[1:] - p0
    if A.size == 0:
        return p0.copy(), 0.0
    G = A @ A.T
    rhs = 0.5 * np.einsum("ij,ij->i", A, A)
    try:
        u = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(G, rhs, rcond=None)[0]
    center = p0 + A.T @ u
    diff = pts - center
    radius = float(np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff))))
    return center.astype(np.float64, copy=False), radius