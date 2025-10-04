from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gal.learning.learn import project_constraint
from gal.utils.helpers import augment_with_minimums, k_additive_constraints, enumerate_subsets


@dataclass
class CapacitySpace:
    subsets: List[Tuple[int, ...]]
    proj_index: Dict[Tuple[int, ...], int]
    n_single: int
    add_k: int

    def __post_init__(self) -> None:
        if not self.subsets:
            raise ValueError("Expected at least one subset for capacity space.")
        self.full_index: Dict[Tuple[int, ...], int] = {
            subset: idx for idx, subset in enumerate(self.subsets)
        }
        self.last_subset: Tuple[int, ...] = self.subsets[-1]
        self.full_dim: int = len(self.subsets)
        if len(self.proj_index) != self.full_dim - 1:
            raise ValueError("Projected index map must omit exactly one subset.")
        if self.last_subset not in self.full_index:
            raise ValueError("Last subset missing from full index.")
        self.last_pos: int = self.full_index[self.last_subset]
        self.permutation: np.ndarray = np.asarray(
            [self.full_index[s] for s in self.subsets],
            dtype=np.int64,
        )

    def expand_center(self, center_proj: np.ndarray) -> np.ndarray:
        vec = np.asarray(center_proj, dtype=float).reshape(-1)
        expected = self.full_dim - 1
        if vec.size != expected:
            raise ValueError(
                f"Center length {vec.size} does not match projected dim {expected}."
            )
        full = np.zeros(self.full_dim, dtype=float)
        for subset, idx in self.proj_index.items():
            full[self.full_index[subset]] = vec[idx]
        full[self.last_pos] = 1.0 - float(np.sum(vec))
        return full

    def project(self, constraint: np.ndarray) -> Tuple[np.ndarray, float]:
        vec = np.asarray(constraint, dtype=float).reshape(-1)
        if vec.size != self.full_dim:
            raise ValueError(
                f"Constraint length {vec.size} does not match full dim {self.full_dim}."
            )
        ordered = vec[self.permutation]
        proj_row, proj_rhs = project_constraint(ordered)
        return np.asarray(proj_row, dtype=float), float(proj_rhs)


def _prepare_capacity_space(
    X: np.ndarray,
    *,
    add_k: int,
    log: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, CapacitySpace, np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n_single = X.shape[1]
    k_val = int(add_k) if add_k else 1
    if k_val < 1:
        k_val = 1
    if k_val > n_single:
        k_val = n_single
    extra_subsets: List[Tuple[int, ...]] = []
    if k_val > 1:
        X_aug, extra_subsets = augment_with_minimums(X, k_val, return_index_map=True)
        X_work = np.ascontiguousarray(X_aug, dtype=float)
        if log is not None:
            log.info(
                "Applied additivity augmentation (k=%d) - shape %s",
                k_val,
                X_work.shape,
            )
    else:
        X_work = X.copy()
    subsets = [(i,) for i in range(n_single)] + list(extra_subsets)
    if not subsets:
        raise RuntimeError("Failed to enumerate subsets for capacity space.")
    if subsets != enumerate_subsets(n_single, k_val):
        raise RuntimeError("Subset ordering mismatch between augmentation and canonical order.")
    A0, b0, proj_index = k_additive_constraints(n_single, k_val)
    space = CapacitySpace(
        subsets=subsets,
        proj_index=proj_index,
        n_single=n_single,
        add_k=k_val,
    )
    return X_work, space, np.asarray(A0, dtype=float), np.asarray(b0, dtype=float)

