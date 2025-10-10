"""Core geometric helpers for convergence analysis."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FullDimensionalPolytope:
    """Representation of a polytope expressed in a full-dimensional basis."""

    A: np.ndarray
    b: np.ndarray
    origin: np.ndarray
    basis: np.ndarray

    @property
    def dimension(self) -> int:
        if self.basis.ndim != 2:
            return 0
        return int(self.basis.shape[1])

    def project_point(self, point: np.ndarray) -> np.ndarray:
        point_vec = np.asarray(point, dtype=float).reshape(-1)
        if self.dimension == 0:
            return np.zeros(0, dtype=float)
        return self.basis.T @ (point_vec - self.origin)

    def lift_point(self, reduced: np.ndarray) -> np.ndarray:
        reduced_vec = np.asarray(reduced, dtype=float).reshape(-1)
        if self.dimension == 0:
            return self.origin.astype(float, copy=True)
        return self.origin + self.basis @ reduced_vec


def ensure_interior(A: np.ndarray, b: np.ndarray, anchor: np.ndarray, tol: float = 1e-9) -> None:
    margin = b - A @ anchor
    if np.any(margin <= tol):
        raise ValueError("Anchor must lie strictly inside the polytope (A @ a < b).")


def strict_interior_point(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    margin = b - A @ anchor
    if np.all(margin > tol):
        return anchor

    delta = cp.Variable(anchor.size)
    tightened = b - tol * 10.0
    constraints = [A @ (anchor + delta) <= tightened]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), constraints)
    for solver in ("GUROBI", "CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            logger.debug("Strict interior projection: attempting solver %s", solver)
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            logger.debug("Solver %s failed while seeking interior point", solver, exc_info=True)
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and delta.value is not None:
            candidate = np.asarray(anchor + delta.value, dtype=float)
            if np.all(b - A @ candidate > tol):
                logger.debug("Strict interior projection succeeded with solver %s", solver)
                return candidate
    logger.warning("Strict interior projection failed; status=%s", problem.status)
    raise ValueError("Failed to find strict interior point for the polytope.")


def find_feasible_point(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float)

    m, n = A.shape
    x = cp.Variable(n)
    constraints = [A @ x <= b + tol]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
    for solver in ("GUROBI", "CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            logger.debug("Feasible point search: attempting solver %s", solver)
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            logger.debug("Solver %s failed while seeking feasible point", solver, exc_info=True)
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
            candidate = np.asarray(x.value, dtype=float).reshape(-1)
            if candidate.size != n:
                continue
            if np.all(A @ candidate <= b + tol * 10.0):
                return candidate
    logger.warning("Feasible point search failed; status=%s", problem.status)
    raise ValueError("Failed to locate a feasible point for the polytope.")


def support_value(
    A: np.ndarray,
    b: np.ndarray,
    *,
    u: np.ndarray,
    solver_sequence: Sequence[str] = ("GUROBI", "CLARABEL", "ECOS", "SCS"),
) -> Tuple[float, np.ndarray]:
    """Return support value h_V(u) = max_x u^T x with maximiser."""
    if u.ndim != 1:
        raise ValueError("Direction u must be a 1D array for support computation.")
    d = A.shape[1]
    if u.size != d:
        raise ValueError("Direction dimension mismatch with polytope dimension.")
    x = cp.Variable(d)
    constraints = [A @ x <= b]
    problem = cp.Problem(cp.Maximize(u @ x), constraints)
    for solver in solver_sequence:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and x.value is not None:
            point = np.asarray(x.value, dtype=float)
            value = float(u @ point)
            return value, point
        if problem.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
            return float("inf"), np.full(d, np.nan)
    logger.error("Support LP failed to solve; last status=%s", problem.status)
    raise ValueError("Support LP did not converge with available solvers.")


def full_dimensional_form(
    A: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-9,
    seed: int = 123,
    max_attempts: int | None = None,
) -> FullDimensionalPolytope:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2-D array of shape (m, n).")
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("Incompatible shapes between A and b.")

    origin = find_feasible_point(A, b, tol=tol)
    if origin.size != n:
        raise ValueError("Feasible point dimensionality mismatch.")

    if max_attempts is None:
        max_attempts = max(64, 8 * max(n, 1))

    rng = np.random.default_rng(seed)
    basis_vectors: List[np.ndarray] = []
    attempts_left = max_attempts

    if n == 0:
        basis = np.zeros((0, 0), dtype=float)
    else:
        while attempts_left > 0 and len(basis_vectors) < n:
            direction = rng.normal(size=n)
            norm = np.linalg.norm(direction)
            if norm <= tol:
                continue
            direction /= norm
            try:
                value_plus, x_plus = support_value(A, b, u=direction)
                value_minus, x_minus = support_value(A, b, u=-direction)
            except ValueError as exc:  # pragma: no cover - defensive path
                raise ValueError("Support computation failed while determining affine hull.") from exc

            if not np.isfinite(value_plus) or not np.isfinite(value_minus):
                raise ValueError("Polytope appears unbounded; cannot determine affine hull.")
            if x_plus is None or x_minus is None:
                attempts_left -= 1
                continue
            diff = np.asarray(x_plus, dtype=float).reshape(-1) - np.asarray(x_minus, dtype=float).reshape(-1)
            if diff.size != n:
                attempts_left -= 1
                continue
            for basis_vec in basis_vectors:
                diff = diff - np.dot(basis_vec, diff) * basis_vec
            diff_norm = np.linalg.norm(diff)
            if diff_norm <= tol:
                attempts_left -= 1
                continue
            basis_vectors.append(diff / diff_norm)
            attempts_left = max_attempts

        if basis_vectors:
            basis_stack = np.column_stack(basis_vectors)
            q, _ = np.linalg.qr(basis_stack, mode="reduced")
            basis = np.asarray(q, dtype=float)
        else:
            basis = np.zeros((n, 0), dtype=float)

    A_basis = A @ basis if basis.size else np.zeros((m, 0), dtype=float)
    offset = A @ origin
    b_shift = b - offset

    if basis.size:
        mask = np.abs(A_basis) < tol * 10.0
        if np.any(mask):
            A_basis = A_basis.copy()
            A_basis[mask] = 0.0
    b_shift = b_shift.astype(float, copy=True)

    return FullDimensionalPolytope(
        A=A_basis,
        b=b_shift,
        origin=np.asarray(origin, dtype=float).reshape(-1),
        basis=basis,
    )


def ray_extent(A: np.ndarray, b: np.ndarray, anchor: np.ndarray, direction: np.ndarray, tol: float = 1e-12) -> float:
    Ad = A @ direction
    Aa = A @ anchor
    mask_pos = Ad > tol
    if not np.any(mask_pos):
        return math.inf
    numerator = b[mask_pos] - Aa[mask_pos]
    denom = Ad[mask_pos]
    radius = np.min(numerator / denom)
    if radius < 0:
        raise ValueError("Anchor is outside the polytope along the provided direction.")
    return float(radius)


def outradius_via_support(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray,
    *,
    n_dirs: int = 4096,
    seed: int = 123,
    refine_rounds: int = 2,
) -> float:
    """Approximate outradius by maximising support function directions."""
    if n_dirs <= 0:
        raise ValueError("n_dirs must be positive for outradius estimation.")
    rng = np.random.default_rng(seed)
    d = anchor.size
    U = rng.normal(size=(n_dirs, d))
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    mask = norms[:, 0] > 1e-12
    if not np.any(mask):
        raise ValueError("Failed to sample valid directions for outradius computation.")
    U = U[mask] / norms[mask]

    best_r = -math.inf
    best_u = None

    logger.debug(
        "Outradius support search: n_dirs=%d refine_rounds=%d seed=%d",
        U.shape[0],
        refine_rounds,
        seed,
    )

    for _ in range(max(refine_rounds, 1)):
        for u in U:
            supp, _ = support_value(A, b, u=u)
            if not np.isfinite(supp):
                return float("inf")
            r = supp - float(u @ anchor)
            if r > best_r:
                best_r = r
                best_u = u.copy()
                logger.debug("Outradius support update: r=%.6f", best_r)
        if best_u is None:
            break
        logger.debug("Outradius refinement round complete (best_r=%.6f)", best_r)
        cap_size = max(64, U.shape[0] // 8)
        noise = rng.normal(size=(cap_size, d))
        proj = noise @ best_u
        noise = noise - proj[:, None] * best_u[None, :]
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-12
        if not np.any(valid):
            U = np.tile(best_u, (1, 1))
            continue
        noise = noise[valid] / norms[valid]
        alpha = 0.1
        U = np.vstack([best_u[None, :], best_u[None, :] + alpha * noise])
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        mask = norms[:, 0] > 1e-12
        if not np.any(mask):
            break
        U = U[mask] / norms[mask]

    result = float(best_r if best_r > 0 else 0.0)
    logger.debug("Outradius support result=%.6f", result)
    return result


def min_distance_from_anchor(A: np.ndarray, b: np.ndarray, anchor: np.ndarray) -> float:
    margins = b - A @ anchor
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-valued constraint normal encountered.")
    return float(np.min(margins / norms))


def max_distance_socp(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    *,
    solver_sequence: Sequence[str] = ("GUROBI", "MOSEK", "CLARABEL", "ECOS", "SCS"),
) -> float:
    if A.size == 0:
        return 0.0

    center_vec = np.asarray(center, dtype=float).reshape(-1)
    n = center_vec.size
    x = cp.Variable(n)
    radius = cp.Variable(nonneg=True)
    constraints = [A @ x <= b, cp.norm(x - center_vec, 2) <= radius]
    problem = cp.Problem(cp.Maximize(radius), constraints)

    for solver in solver_sequence:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and radius.value is not None:
            return float(radius.value)
    logger.warning("Outradius SOCP failed; status=%s", problem.status)
    return float("nan")


__all__ = [
    "FullDimensionalPolytope",
    "ensure_interior",
    "strict_interior_point",
    "find_feasible_point",
    "support_value",
    "full_dimensional_form",
    "ray_extent",
    "outradius_via_support",
    "min_distance_from_anchor",
    "max_distance_socp",
]
