import itertools
from typing import List, Tuple, Dict

import numpy as np

__all__ = [
    "augment_with_minimums",
    "k_additive_constraints",
    "enumerate_subsets",
]

################################################################################
# 1. Feature augmentation – min‑pooling over every subset up to size k
################################################################################


def augment_with_minimums(
    X: np.ndarray,
    k: int,
    *,
    return_index_map: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Return `X_aug` where each sample *x* has, in addition to the original
    *n* singleton co‑ordinates, one extra feature per subset of indices of size
    2 … *k* holding the **minimum** of those co‑ordinates::

        x' = [ x_0, …, x_{n−1},
                min(x_i,x_j)              for |{i,j}| = 2,
                min(x_i,x_j,x_ℓ)          for |{i,j,ℓ}| = 3,
                …,
                min(x_{i₁},…,x_{i_k})     for |⋅| = k ]

    Parameters
    ----------
    X : (m, n) array_like
        Original design matrix – one row per sample, `n = |N|` columns.
    k : int
        Additivity level (``k ≥ 1``).  The number of new columns is
        ``sum_{r=2}^k C(n, r)``.
    return_index_map : bool, default=False
        If *True*, also return the list of index tuples so that callers can map
        augmented co‑ordinates back to the corresponding subset.

    Notes
    -----
    *Runs in Python/NumPy; aimed at datasets where ``n`` is ≲ 15.  For larger
    `n`, consider lazily computing features.*
    """
    X = np.asanyarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2‑D array (samples × features)")

    m, n = X.shape
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 ≤ k ≤ n_features")

    # ------------------------------------------------------------------
    # Build list of subsets (combinations) and compute mins column‑wise
    # ------------------------------------------------------------------
    subset_list: List[Tuple[int, ...]] = []
    new_columns: List[np.ndarray] = []
    for r in range(2, k + 1):
        for comb in itertools.combinations(range(n), r):
            subset_list.append(comb)
            new_columns.append(np.min(X[:, comb], axis=1))

    if new_columns:
        X_aug = np.hstack([X] + [col[:, None] for col in new_columns])
    else:  # k == 1 – no augmentation
        X_aug = X.copy()

    if return_index_map:
        return X_aug, subset_list
    return X_aug


################################################################################
# 2. Linear constraints for k‑additive capacities (mass variables m(T))
################################################################################


def enumerate_subsets(n: int, k: int) -> List[Tuple[int, ...]]:
    """Enumerate all non‑empty subsets of ``{0,…,n−1}`` of size ≤ ``k`` in the
    canonical order: first singletons, then pairs, triples, …  The returned
    list is used to index the mass vector *m*.
    """
    subs: List[Tuple[int, ...]] = []
    for r in range(1, k + 1):
        subs.extend(itertools.combinations(range(n), r))
    return subs


def k_additive_constraints(
    n: int, k: int
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, ...], int]]:
    """Return a *half‑space* representation ``A x ≤ b`` (with *no* equality
    constraint) describing the monotonicity inequalities of a *k‑additive
    capacity* after **eliminating the last variable** via the projection

    ``m_last = 1 − Σ_{j ≠ last} m_j``.

    Parameters
    ----------
    n : int
        Size of the ground set *N* (aka number of *singleton* variables).
    k : int
        Additivity order (``1 ≤ k ≤ n``).

    Returns
    -------
    A : (p, d) ndarray
        Constraint matrix where ``d = L−1`` (all masses except the eliminated
        last one) and rows implement inequalities of form ``A @ x ≤ b``.
    b : (p,) ndarray
        Right‑hand side vector.
    index_map : dict
        Mapping from subset tuple -> column index (in the *projected* vector).
    """
    if k < 1 or k > n:
        raise ValueError("Require 1 ≤ k ≤ n")

    # ------------------------------------------------------------------
    # 1. Enumerate mass variables and pick *last* for projection
    # ------------------------------------------------------------------
    subsets = enumerate_subsets(n, k)
    L = len(subsets)
    if L == 0:
        raise RuntimeError("No mass variables – check n and k")

    last_subset = subsets[-1]  # arbitrary but deterministic

    # Map subset -> column index *before* projection
    col_full: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(subsets)}

    # Map subset -> column index *after* projection (skip last)
    col_proj: Dict[Tuple[int, ...], int] = {}
    col_counter = 0
    for s in subsets[:-1]:  # exclude last
        col_proj[s] = col_counter
        col_counter += 1

    # ------------------------------------------------------------------
    # 2. Build every inequality (1) in the black‑board snapshot
    # ------------------------------------------------------------------
    rows: List[np.ndarray] = []
    rhs: List[float] = []

    full_index_set = set(subsets[:-1])  # columns that remain explicit

    # pre-compute subsets of size ≤ k‑1 for efficiency
    subset_cache: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    for i in range(n):
        others = [o for o in range(n) if o != i]
        # iterate over *all* 2^{n-1} subsets S ⊆ N\{i}
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                # compute J = { T∪{i} : T ⊆ S, |T| ≤ k−1 }
                J: List[Tuple[int, ...]] = []
                # quick look‑up in cache
                if S in subset_cache:
                    Ts = subset_cache[S]
                else:
                    Ts = []
                    max_t = min(k - 1, len(S))
                    for t in range(max_t + 1):
                        Ts.extend(itertools.combinations(S, t))
                    subset_cache[S] = Ts
                for T in Ts:
                    U = tuple(sorted(T + (i,)))  # T∪{i}
                    if len(U) <= k:
                        J.append(U)
                # ----------------------------------------------------------------
                # Build the inequality  sum_{U∈J} m(U) ≥ 0
                # After projection m(last) = 1 − Σ other m
                # ----------------------------------------------------------------
                last_included = last_subset in J

                if last_included:
                    J_no_last = [s for s in J if s != last_subset]
                    # Inequality: Σ_{P not in J_no_last} m(P) ≤ 1
                    # => coefficients +1 on columns in complement, RHS = 1
                    row = np.zeros(L - 1)
                    complement = full_index_set.difference(J_no_last)
                    for s in complement:
                        row[col_proj[s]] = 1.0
                    rows.append(row)
                    rhs.append(1.0)
                else:
                    # −Σ_{U∈J} m(U) ≤ 0
                    row = np.zeros(L - 1)
                    for s in J:
                        row[col_proj[s]] = -1.0
                    rows.append(row)
                    rhs.append(0.0)

    A = np.vstack(rows) if rows else np.empty((0, L - 1))
    b = np.asarray(rhs)
    return A, b
