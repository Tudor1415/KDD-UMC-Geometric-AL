import numpy as np
import cvxpy as cp
from typing import Tuple

"""poly_centers.py
Utility functions to compute various geometric “centres’’ of a convex polyhedron
specified by a system of linear inequalities

    P := { x ∈ R^n | A x ≤ b }

Functions
---------
chebyshev_center(A, b)
    Largest Euclidean ball contained in P.
analytical_center(A, b, eps)
    Analytic‑barrier centre (log‑barrier maximiser).
minkowski_center(A, b)
    Minkowski (Helly) centre via Belloni‑Freund robust LP reformulation.
volumetric_center(A, b)
    Centre of the maximum‑volume inscribed ellipsoid (John/volumetric centre).
max_inscribed_ball_radius(A, b, c)
    Radius of the largest ball with fixed centre c contained in P.

The implementations follow formulations from:
* Dick den Hertog, Jean Pauphilet, Mohamed Yahya Soali – “Minkowski Centers via Robust Optimization” (2023)  citeturn0file0
* György Sonnevend – “An ‘Analytical Centre’ for Polyhedrons …” (1985)  citeturn0file1
* Boyd & Vandenberghe – "Convex Optimization", §8.4 & §8.5.
"""

__all__ = [
    "chebyshev_center",
    "analytical_center",
    "minkowski_center",
    "volumetric_center",
    "mse_center",
]


# -----------------------------------------------------------------------------
# 1. Chebyshev centre (largest inscribed ball)
# -----------------------------------------------------------------------------


def chebyshev_center(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute Chebyshev centre x and radius r of the largest Euclidean ball in P={x|Ax≤b}."""
    m, n = A.shape
    x = cp.Variable(n)
    r = cp.Variable()
    norms = np.linalg.norm(A, axis=1)
    constraints = [A @ x + cp.multiply(norms, r) <= b, r >= 0]
    prob = cp.Problem(cp.Maximize(r), constraints)
    prob.solve(
        solver=cp.GUROBI if cp.GUROBI in cp.installed_solvers() else cp.ECOS,
        verbose=False,
    )
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError("Chebyshev centre computation failed – polyhedron may be empty.")
    return x.value, float(r.value)


# -----------------------------------------------------------------------------
# 2. Analytical centre (log‑barrier maximiser)
# -----------------------------------------------------------------------------


def analytical_center(A: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a strictly‑feasible analytic centre of *P*.

    Parameters
    ----------
    eps : float
        Margin pushed inside the facets so that log arguments remain positive.
    """
    m, n = A.shape
    x = cp.Variable(n)
    constraints = [A @ x <= b - eps]
    objective = cp.Maximize(cp.sum(cp.log(b - A @ x)))  # concave -> maximise OK
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(
            "Analytical centre computation failed – check eps or feasibility."
        )
    return x.value


# -----------------------------------------------------------------------------
# 3. Minkowski centre (robust‑LP formulation)
# -----------------------------------------------------------------------------


def minkowski_center(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute a Minkowski centre x_M and its symmetry λ (bounded polyhedron)."""
    m, n = A.shape
    delta = np.empty(m)
    for i in range(m):
        x = cp.Variable(n)
        obj = cp.Minimize(A[i] @ x)
        prob = cp.Problem(obj, [A @ x <= b])
        prob.solve(solver=cp.GUROBI if cp.GUROBI in cp.installed_solvers() else cp.ECOS)
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError("Polyhedron appears empty while computing δ.")
        delta[i] = obj.value

    w = cp.Variable(n)
    lam = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Maximize(lam), [A @ w - cp.multiply(delta, lam) <= b])
    prob.solve(solver=cp.GUROBI if cp.GUROBI in cp.installed_solvers() else cp.ECOS)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError("Minkowski centre computation failed – P may be empty/unbounded.")

    x_m = w.value / (1.0 + lam.value)
    return x_m, float(lam.value)


# -----------------------------------------------------------------------------
# 4. Volumetric / John centre (maximum‑volume inscribed ellipsoid)
# -----------------------------------------------------------------------------


def volumetric_center(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centre c (and shape matrix P) of the maximum-volume inscribed ellipsoid."""
    m, n = A.shape
    c = cp.Variable(n)
    P = cp.Variable((n, n), PSD=True)
    constraints = [A[i] @ c + cp.norm(P.T @ A[i], 2) <= b[i] for i in range(m)]
    prob = cp.Problem(cp.Maximize(cp.log_det(P)), constraints)
    solver = cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.SCS
    prob.solve(solver=solver, verbose=False)
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError("Volumetric centre computation failed – solver accuracy.")
    return c.value, P.value


# -----------------------------------------------------------------------------
# 5. Constrained least‑squares (MSE centre) -----------------------------------
# -----------------------------------------------------------------------------


def mse_center(
    A: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = "GUROBI",
    verbose: bool = False,
):
    """
    Minimise   Σ(⟨C,xᵢ⟩ − yᵢ)²   subject to   A·C ≤ b.

    If `X` or `y` is empty, the problem reduces to finding *any* vector C
    that satisfies the constraints (objective = 0).
    """
    # ---- basic sanity ---------------------------------------------------
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    m, n = A.shape
    if X.size and X.shape[1] != n:
        raise ValueError("X and A must have the same column dimension.")

    # ---- build the optimisation problem --------------------------------
    C = cp.Variable(n)
    constraints = [A @ C <= b]

    if X.size == 0 or y.size == 0:
        # pure feasibility – minimise the constant 0
        objective = cp.Minimize(0)
    else:
        objective = cp.Minimize(cp.sum_squares(X @ C - y))

    prob = cp.Problem(objective, constraints)

    # ---- solver with graceful fallback ---------------------------------
    def _try_solve(name):
        try:
            prob.solve(solver=name, verbose=verbose)
            return prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except cp.error.SolverError:
            return False

    ok = _try_solve(solver) or _try_solve("ECOS") or _try_solve("SCS")

    if not ok:
        print("[mse_center] problem status:", prob.status)
        return None

    return C.value.squeeze()


if __name__ == "__main__":
    # Small self‑test on a 2‑D square [0,1]^2.
    A = np.array(
        [
            [1, 0],  #  x ≤ 1
            [-1, 0],  # -x ≤ 0
            [0, 1],  #  y ≤ 1
            [0, -1],  # -y ≤ 0
        ],
        dtype=float,
    )
    b = np.array([1, 0, 1, 0], dtype=float)

    print("Chebyshev:", chebyshev_center(A, b))
    print("Analytical:", analytical_center(A, b))
    print("Minkowski :", minkowski_center(A, b))
    print("Volumetric :", volumetric_center(A, b)[0])
