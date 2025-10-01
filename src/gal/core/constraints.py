from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

__all__ = ["enumerate_subsets", "k_additive_constraints"]


def enumerate_subsets(n: int, k: int) -> List[Tuple[int, ...]]:
    """Enumerate all non-empty subsets of ``{0, ..., n-1}`` with size up to ``k``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        raise ValueError("k must be non-negative")
    subs: List[Tuple[int, ...]] = []
    for r in range(1, min(k, n) + 1):
        subs.extend(itertools.combinations(range(n), r))
    return subs


def k_additive_constraints(
    n: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, ...], int]]:
    """Construct monotonicity inequalities for a projected k-additive capacity."""
    if k < 1 or k > n:
        raise ValueError("Require 1 <= k <= n")

    subsets = enumerate_subsets(n, k)
    L = len(subsets)
    if L == 0:
        raise RuntimeError("No mass variables -- check n and k")

    last_subset = subsets[-1]
    col_full: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(subsets)}

    col_proj: Dict[Tuple[int, ...], int] = {}
    col_counter = 0
    for s in subsets[:-1]:
        col_proj[s] = col_counter
        col_counter += 1

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    full_index_set = set(subsets[:-1])
    subset_cache: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    for i in range(n):
        others = [o for o in range(n) if o != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                if S in subset_cache:
                    Ts = subset_cache[S]
                else:
                    Ts = []
                    max_t = min(k - 1, len(S))
                    for t in range(max_t + 1):
                        Ts.extend(itertools.combinations(S, t))
                    subset_cache[S] = Ts

                J: List[Tuple[int, ...]] = []
                for T in Ts:
                    U = tuple(sorted(T + (i,)))
                    if len(U) <= k:
                        J.append(U)

                last_included = last_subset in J

                if last_included:
                    J_no_last = [s for s in J if s != last_subset]
                    row = np.zeros(L - 1)
                    complement = full_index_set.difference(J_no_last)
                    for s in complement:
                        row[col_proj[s]] = 1.0
                    rows.append(row)
                    rhs.append(1.0)
                else:
                    row = np.zeros(L - 1)
                    for s in J:
                        row[col_proj[s]] = -1.0
                    rows.append(row)
                    rhs.append(0.0)

    A = np.vstack(rows) if rows else np.empty((0, L - 1))
    b = np.asarray(rhs)
    index_map = dict(col_proj)
    return A, b, index_map
