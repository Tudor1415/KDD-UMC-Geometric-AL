"""Basic geometric helpers for ball-tree construction."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a - b, a - b))


def enclose_two_balls(
    b1: Tuple[np.ndarray, float], b2: Tuple[np.ndarray, float]
) -> Tuple[np.ndarray, float]:
    c1, r1 = b1
    c2, r2 = b2
    diff = c2 - c1
    d = float(np.linalg.norm(diff))
    if r1 >= r2 + d:
        return c1, float(r1)
    if r2 >= r1 + d:
        return c2, float(r2)
    if d == 0.0:
        return c1, float(max(r1, r2))
    radius = 0.5 * (d + r1 + r2)
    center = c1 + ((radius - r1) / d) * diff
    return center, float(radius)


def enclose_many_balls(balls: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
    center, radius = balls[0]
    radius = float(radius)
    for child in balls[1:]:
        center, radius = enclose_two_balls((center, radius), child)
    return center, radius


def centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def project(points: np.ndarray, v: np.ndarray) -> np.ndarray:
    return points @ v