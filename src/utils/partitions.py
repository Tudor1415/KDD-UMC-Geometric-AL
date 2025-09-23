"""Partition helpers for ball-tree builders."""

from __future__ import annotations

from typing import List

import numpy as np


def nth_element_inplace(values: np.ndarray, k: int) -> None:
    np.partition(values, k, axis=0)


def axis_median_split(
    X: np.ndarray, indices: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    if indices.size <= 1:
        return indices.copy(), indices.copy()[:0]
    mid = indices.size // 2
    partition = np.argpartition(X[indices, axis], mid)
    ordered = indices[partition]
    left = ordered[:mid]
    right = ordered[mid:]
    return left, right


def direction_quantile_splits(
    X: np.ndarray, indices: np.ndarray, direction: np.ndarray, k_children: int
) -> List[np.ndarray]:
    projections = X[indices] @ direction
    n = indices.size
    if k_children <= 1 or n == 0:
        return [indices.copy()]
    quantiles = [((i + 1) * n) // k_children for i in range(k_children - 1)]
    order = np.argpartition(projections, quantiles)
    sorted_idx = indices[order]
    bins: List[np.ndarray] = []
    for child in range(k_children):
        start = (child * n) // k_children
        end = ((child + 1) * n) // k_children if child < k_children - 1 else n
        if end > start:
            bins.append(sorted_idx[start:end])
    return bins