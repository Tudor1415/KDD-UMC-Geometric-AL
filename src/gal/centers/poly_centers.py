from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Tuple, Any, Dict

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
    Centre of the maximum-volume inscribed ellipsoid (John/volumetric centre).
hit_and_run_centroid(A, b, ...)
    Approximate centroid via hit-and-run Monte Carlo sampling.
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
    "hit_and_run_centroid",
    "_center_fn",
    "_chebyshev_radius",
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


# -----------------------------------------------------------------------------
# 6. Helper utilities used by the experiment runner
# -----------------------------------------------------------------------------


def _hit_and_run_samples(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray,
    *,
    pool_size: int,
    burn_in: int,
    thinning: int,
    seed: int,
    tol: float = 1e-12,
) -> np.ndarray:
    """Draw uniform samples from the bounded polytope via hit-and-run MCMC."""

    rng = np.random.default_rng(seed)
    x = anchor.astype(float, copy=True)
    d = x.size
    samples = []
    Ax = A @ x
    steps = 0

    while len(samples) < pool_size:
        direction = rng.normal(size=d)
        norm = np.linalg.norm(direction)
        if norm <= tol:
            continue
        direction /= norm
        Ad = A @ direction

        upper = np.inf
        lower = -np.inf
        positive = Ad > tol
        if np.any(positive):
            upper = np.min((b[positive] - Ax[positive]) / Ad[positive])
        negative = Ad < -tol
        if np.any(negative):
            lower = np.max((b[negative] - Ax[negative]) / Ad[negative])
        if not np.isfinite(upper) or not np.isfinite(lower):
            raise ValueError(
                "Hit-and-run detected unbounded direction; check that the polytope is bounded."
            )
        if upper < lower:
            # Numerical instability – reject this direction.
            continue
        step = rng.uniform(lower, upper)
        x = x + step * direction
        Ax = Ax + step * Ad
        steps += 1
        if steps <= burn_in:
            continue
        if (steps - burn_in) % max(thinning, 1) == 0:
            samples.append(x.copy())
    return np.vstack(samples)


def hit_and_run_centroid(
    A: np.ndarray,
    b: np.ndarray,
    *,
    anchor: np.ndarray | None = None,
    pool_size: int = 4096,
    burn_in: int | None = None,
    thinning: int = 1,
    seed: int | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Approximate the centroid of a bounded polyhedron using hit-and-run samples."""

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2-D array of shape (m, n).")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("b must have length equal to the number of rows of A.")

    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    if burn_in is None:
        burn_in = max(pool_size, 10 * n)
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thinning <= 0:
        raise ValueError("thinning must be positive.")

    if anchor is None:
        center, radius = chebyshev_center(A, b)
        if radius <= tol:
            raise ValueError("Failed to find a strict interior point for hit-and-run sampling.")
        anchor_vec = np.asarray(center, dtype=float).reshape(-1)
    else:
        anchor_vec = np.asarray(anchor, dtype=float).reshape(-1)
    if anchor_vec.size != n:
        raise ValueError("anchor dimension must match the number of columns of A.")

    margin = b - A @ anchor_vec
    if np.any(margin <= tol):
        raise ValueError("anchor must lie strictly inside the polytope (A @ anchor < b).")

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63 - 1))

    samples = _hit_and_run_samples(
        A,
        b,
        anchor_vec,
        pool_size=pool_size,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
        tol=tol,
    )
    return np.asarray(np.mean(samples, axis=0), dtype=float)


def _ensure_no_params(center: str, options: Dict[str, Any]) -> None:
    if options:
        keys = ", ".join(sorted(str(k) for k in options))
        raise ValueError(
            f"Center '{center}' does not accept parameters (got: {keys})."
        )


def _validate_hit_and_run_options(options: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"anchor", "pool_size", "burn_in", "thinning", "seed", "tol"}
    invalid = sorted(set(options) - allowed)
    if invalid:
        raise ValueError(
            "Unsupported parameters for hit-and-run centroid: "
            + ", ".join(str(k) for k in invalid)
        )
    return dict(options)


def _center_fn(name: str, **options: Any):
    key = str(name).strip().lower()
    if key in {"chebyshev", "chebyshev_center"}:
        _ensure_no_params("chebyshev", options)
        return _chebyshev_center_wrapper
    if key in {"analytic", "analytical", "analytic_center"}:
        _ensure_no_params("analytic", options)
        return _analytic_center_wrapper
    if key in {"minkowski", "minkowski_center"}:
        _ensure_no_params("minkowski", options)
        return _minkowski_center_wrapper
    if key in {"volumetric", "john", "john_center", "volumetric_center"}:
        _ensure_no_params("volumetric", options)
        return _volumetric_center_wrapper
    if key in {"zero", "origin"}:
        _ensure_no_params("zero", options)
        return _zero_center_wrapper
    if key in {"centroid", "hit_and_run", "hit_and_run_centroid"}:
        params = _validate_hit_and_run_options(options)
        return lambda A, b: _hit_and_run_centroid_wrapper(A, b, **params)
    raise ValueError(
        (
            "Unknown center '{name}'. Available: chebyshev, analytic, "
            "minkowski, volumetric, centroid, zero"
        ).format(name=name)
    )


def _chebyshev_center_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)
    center, _ = chebyshev_center(A, b)
    return np.asarray(center, dtype=float).reshape(-1)


def _analytic_center_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)
    center = analytical_center(A, b)
    return np.asarray(center, dtype=float).reshape(-1)


def _zero_center_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    dim = A.shape[1] if A.ndim == 2 else 0
    return np.zeros(dim, dtype=float)


def _minkowski_center_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)
    center, _ = minkowski_center(A, b)
    return np.asarray(center, dtype=float).reshape(-1)


def _volumetric_center_wrapper(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)
    center, _ = volumetric_center(A, b)
    return np.asarray(center, dtype=float).reshape(-1)


def _hit_and_run_centroid_wrapper(A: np.ndarray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)
    return hit_and_run_centroid(A, b, **kwargs)


def _chebyshev_radius(A: np.ndarray, b: np.ndarray, center: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return float("inf")
    b = np.asarray(b, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    slacks = (b - A @ center) / norms
    return float(np.min(slacks))


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
