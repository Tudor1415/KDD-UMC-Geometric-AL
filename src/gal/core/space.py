from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from gal.learning.learn import project_constraint


@dataclass
class CapacitySpace:
    """k-additive capacity polytope helper."""

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


__all__ = ["CapacitySpace"]

