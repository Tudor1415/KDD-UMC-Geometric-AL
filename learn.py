"""learn.py
==========
An *active‑learning* loop that iteratively tightens a half‑space description
of an unknown direction **q⋆**.  At each round we:

1.   **center selection** – given the current feasible region
     `P = {q : A q ≤ b}` choose a center `c` via an arbitrary `center_fn` –
     e.g. `poly_centers.chebyshev_center`, `analytical_center`, …
2.   **Uncertainty sampling** – over all data points stored in a Ball‑Tree we
     find a pair `(a, b)` whose difference vector is *most ambiguous* wrt `c`
     using the best‑first search from *simple_balltree.py* (minimises
     `|⟨a − b, c⟩| / ‖a − b‖`).
3.   **Oracle query** – ask the user‑supplied `oracle(a, b)` for the sign
     `y ∈ {−1,+1}` of the true, hidden direction: `y = sign(⟨a − b, q⋆⟩)`.
4.   **Constraint update** – append the linear constraint
     `y · (a − b)ᵀ q ≥ 0`, i.e. `A ← [A ; y·(a−b)ᵀ]`, `b ← [b ; 0]`.

The function stops after `n_iter` rounds and returns the final center and the
expanded polyhedron.  A `report_hook(iter_idx, center, radius)` callback can be
used for live monitoring / plotting.
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional

import numpy as np
from helpers import augment_with_minimums, k_additive_constraints

import numpy as np
from ball_tree import *
from viz import visualize_iteration
from poly_centers import chebyshev_center

# ---------------------------------------------------------------------------
# typing helpers
# ---------------------------------------------------------------------------
Array = np.ndarray
CenterFn = Callable[[Array, Array], Tuple[Array, float]]  # returns (center, radius)
OracleFn = Callable[[Array, Array], int]  # ±1
ReportHook = Optional[Callable[[int, Array, float], None]]


# ---------------------------------------------------------------------------
# utility: radius of largest ball around c inside {x: A x ≤ b}
# ---------------------------------------------------------------------------


def _chebyshev_radius(A: Array, b: Array, c: Array) -> float:
    """Chebyshev (inscribed) radius of polyhedron *along Euclidean norm*.

    r(c) = min_i  (b_i − a_iᵀ c) / ‖a_i‖₂,   A c ≤ b assumed.
    Returns **0** if `c` lies outside the polyhedron (negative slack).
    """
    slack = b - A @ c  # (m,)
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    return max(0.0, vals.min(initial=np.inf))


# ---------------------------------------------------------------------------
# main learning loop
# ---------------------------------------------------------------------------


def project_constraint(h):
    return h[:-1] - h[-1], -h[-1]


def learn(
    root,  # ball‑tree root
    A0: Array,
    b0: Array,
    center_fn: CenterFn,
    oracle: OracleFn,
    n_iter: int = 10,
    report_hook: ReportHook = None,
    viz_2D: bool = False,
) -> Tuple[Array, Array, Array]:
    """Active learning loop.

    Parameters
    ----------
    root        : Ball‑Tree root (see *simple_balltree.py*).
    A0, b0      : Initial linear constraints so that feasible set is
                  `{q | A q ≤ b}`.  Shapes `(m0, d)` and `(m0,)`.
    center_fn   : Callable that, given `(A, b)`, returns a feasible center
                  (and optionally a radius – any extra information is ignored).
    oracle      : Function `(a, b) -> {−1, +1}` implementing the true sign.
    n_iter      : Number of active‑learning rounds.
    report_hook : Optional callback `(k, center, radius)` executed after each
                  iteration (including the initial center, k = 0).

    Returns
    -------
    center : ndarray  – the final center estimate.
    A      : ndarray  – all accumulated constraint normals.
    b      : ndarray  – rhs (always zeros appended after the initial `b0`).
    """
    # Copy so we do not mutate caller’s arrays
    A = np.asarray(A0, dtype=float).copy()
    b = np.asarray(b0, dtype=float).copy()

    # --- iteration 0: compute initial center & radius --------------------
    center = center_fn(A, b)  # ignore extra outputs
    complete_center = np.hstack([center, 1 - np.sum(center)])
    radius = _chebyshev_radius(A, b, center)

    if report_hook is not None:
        report_hook(0, complete_center, radius, np.empty(0), 0)

    # --- main loop --------------------------------------------------------
    for it in range(1, n_iter + 1):
        if viz_2D:
            visualize_iteration(A, b, center, radius, iter_idx=it, show=True)
        # 1) pick most ambiguous pair wrt current center
        pair, score, *_ = search_pair(root, complete_center, radius / 2.0)
        if score > radius:
            # no more ambiguous pairs – terminate early
            break

        if pair is None:
            print("No more pairs found, terminating early.")
            break

        # 2) ask the oracle
        a_pt, b_pt = pair
        if np.linalg.norm(a_pt - b_pt) == 0:
            print("Degenerate pair found, skipping.")
            return center, A, b

        # try:
        y = int(np.sign(oracle(a_pt, b_pt)))
        # except Exception as e:
        # print(f"Oracle error: {e}")
        # return center, A, b

        # 3) add linear constraint y·(a‑b)ᵀ q ≥ 0
        diff = a_pt - b_pt
        constraint = -y * (a_pt - b_pt)  # shape (d,)

        if y == 0:
            report_hook(it, complete_center, radius, diff, y)
            return center, A, b

        A_new, b_new = project_constraint(constraint)
        A = np.vstack([A, A_new])
        b = np.concatenate([b, [b_new]])

        # 4) recompute center & radius given new polyhedron
        center = center_fn(A, b)
        complete_center = np.hstack([center, 1 - np.sum(center)])
        radius = _chebyshev_radius(A, b, center)
        if report_hook is not None:
            report_hook(it, complete_center, radius, diff, y)

    return center, A, b


# ---------------------------------------------------------------------------
# demo ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- synthetic data & true direction -------------------------------
    rng = np.random.default_rng(0)
    n_features = 2
    add_k = 2
    N = 10_000_000
    points = rng.standard_normal((N, n_features))
    points = augment_with_minimums(points, add_k)

    tree_root = build_balltree(points, min_leaf_size=5)
    q_true = np.array([1.0, -0.5, 1.0])  # hidden separator

    def oracle(a: Array, b: Array) -> int:
        return 1 if (a - b) @ q_true >= 0 else -1

    # --- initial constraints -------------------------------------------
    A0, b0 = k_additive_constraints(n_features, add_k)

    def reporter(k, c, r, diff):
        print(f"Iter {k:2d}: center={c},  radius={r:.3g}")

    learn(
        root=tree_root,
        A0=A0,
        b0=b0,
        center_fn=lambda A, b: chebyshev_center(A, b),
        oracle=oracle,
        n_iter=10,
        report_hook=reporter,
        viz_2D=True,
    )
