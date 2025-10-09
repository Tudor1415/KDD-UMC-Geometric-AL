#!/usr/bin/env python3
"""Analyze polytope convergence statistics.

This script implements the nine general-case statistics described in the
specification for a bounded polytope V = {x in R^d : A x <= b} and an
anchor point a in int(V).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import math
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy import special, stats


logger = logging.getLogger(__name__)

_WORKER_LOG_CONFIGURED = False


def _configure_worker_logging(level: int) -> None:
    """Initialize logging in worker processes exactly once."""
    global _WORKER_LOG_CONFIGURED
    if _WORKER_LOG_CONFIGURED:
        return
    logging.basicConfig(level=level)
    _WORKER_LOG_CONFIGURED = True


class _ProgressBar:
    """Minimal text progress bar for CLI output."""

    def __init__(self, total: int, message: str = "Progress") -> None:
        self.total = max(total, 0)
        self.message = message
        self.count = 0
        self._lock = threading.Lock()
        self._stream = sys.stderr
        self._last_len = 0
        self._done = False

    def update(self, step: int = 1) -> None:
        if self.total <= 0 or step <= 0:
            return
        with self._lock:
            self.count = min(self.total, self.count + step)
            pct = (100.0 * self.count / self.total) if self.total else 100.0
            text = f"{self.message}: {self.count}/{self.total} ({pct:5.1f}%)"
            padding = max(0, self._last_len - len(text))
            self._stream.write("\r" + text + " " * padding)
            self._stream.flush()
            self._last_len = len(text)
            if self.count >= self.total and not self._done:
                self._stream.write("\n")
                self._stream.flush()
                self._done = True

    def close(self) -> None:
        with self._lock:
            if not self._done and self.total > 0:
                self._stream.write("\n")
                self._stream.flush()
                self._done = True

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class OrientationCDFEntry:
    tau: float
    cdf_hat: float
    cdf_null: float


@dataclass
class ConvergenceStats:
    rho_hat: float
    var_hat: float
    lambda_hat: float
    reflection_asymmetry: float
    john_center: Sequence[float]
    john_matrix: Sequence[Sequence[float]]
    john_vol: float
    cheby_center: Sequence[float]
    cheby_radius: float
    cheby_ball_vol: float
    r_max_from_a: float
    r_min_from_a: float
    sphericity: float
    ks_stat: float
    ks_p_value: float
    orientation_cdf: List[OrientationCDFEntry]
    median_cosine_distance: float
    expected_theta: float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_array(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "arr_0" in arr.files:
                data = arr["arr_0"]
            else:
                raise ValueError(f"NPZ at {path} must contain key 'arr_0'.")
        else:
            data = arr
        return np.asarray(data, dtype=float)
    if path.suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported file extension for {path}")


def _parse_anchor(anchor_arg: str, expected_dim: int | None) -> np.ndarray:
    maybe_path = Path(anchor_arg)
    if maybe_path.exists():
        vec = np.asarray(_load_array(maybe_path), dtype=float).reshape(-1)
    else:
        parts = [p.strip() for p in anchor_arg.split(",") if p.strip()]
        if not parts:
            raise ValueError("Anchor must be provided as path or comma-separated list of numbers.")
        vec = np.asarray([float(p) for p in parts], dtype=float)
    if expected_dim is not None and vec.size != expected_dim:
        raise ValueError(f"Anchor dimension mismatch: expected {expected_dim}, got {vec.size}.")
    return vec


def _unit_ball_volume(dim: int) -> float:
    return math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ensure_interior(A: np.ndarray, b: np.ndarray, anchor: np.ndarray, tol: float = 1e-9) -> None:
    margin = b - A @ anchor
    if np.any(margin <= tol):
        raise ValueError("Anchor must lie strictly inside the polytope (A @ a < b).")


def _strict_interior_point(
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


def _ray_extent(A: np.ndarray, b: np.ndarray, anchor: np.ndarray, direction: np.ndarray, tol: float = 1e-12) -> float:
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
            raise ValueError("Hit-and-run detected unbounded direction; check that the polytope is bounded.")
        if upper < lower:
            # Numerical issue – skip this direction.
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


# ---------------------------------------------------------------------------
# Statistic computations
# ---------------------------------------------------------------------------


def _expected_one_step_convergence(
    samples: np.ndarray,
    anchor: np.ndarray,
    *,
    num_directions: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray]:
    diffs = samples - anchor
    d = anchor.size
    r_vals = np.empty(num_directions, dtype=float)
    for j in range(num_directions):
        while True:
            n = rng.normal(size=d)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n /= norm
                break
        projections = diffs @ n
        p_plus = np.mean(projections >= 0)
        r_vals[j] = min(p_plus, 1.0 - p_plus)
    rho_hat = float(np.mean(r_vals)) if num_directions else float("nan")
    var_hat = float(np.var(r_vals, ddof=1)) if num_directions > 1 else float("nan")
    return rho_hat, var_hat, r_vals


def _minkowski_symmetry(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray,
    directions: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[float, float, float]:
    ratios = []
    max_radius = 0.0  # reuse the same epsilon-net to approximate the outradius
    max_asym = 0.0
    for u in directions:
        norm = np.linalg.norm(u)
        if norm <= tol:
            continue
        u = u / norm
        rho_plus = _ray_extent(A, b, anchor, u, tol=tol)
        rho_minus = _ray_extent(A, b, anchor, -u, tol=tol)
        if rho_plus <= 0 or not np.isfinite(rho_plus):
            logger.debug("E-net direction skipped due to invalid forward radius", extra={"rho_plus": rho_plus})
            continue
        if rho_minus <= 0 or not np.isfinite(rho_minus):
            logger.debug("E-net direction skipped due to invalid backward radius", extra={"rho_minus": rho_minus})
            continue
        rmin = float(min(rho_plus, rho_minus))
        rmax = float(max(rho_plus, rho_minus))
        if rmax <= 0.0:
            continue
        lam_dir = rmin / rmax
        ratios.append(lam_dir)
        max_radius = max(max_radius, rmax)
        max_asym = max(max_asym, (rmax / rmin) - 1.0)
    lambda_hat = float(np.min(ratios)) if ratios else float("nan")
    if max_radius == 0.0:
        max_radius = float("nan")
    if not ratios:
        max_asym = float("nan")
    return lambda_hat, max_radius, float(max_asym)


def _john_ellipsoid(
    A: np.ndarray,
    b: np.ndarray,
    *,
    solver_sequence: Sequence[str] = ("GUROBI", "MOSEK", "CLARABEL", "SCS", "ECOS"),
) -> Tuple[np.ndarray, np.ndarray, float]:
    m, d = A.shape
    c = cp.Variable(d)
    P = cp.Variable((d, d), PSD=True)
    constraints = [A[i] @ c + cp.norm(P.T @ A[i], 2) <= b[i] for i in range(m)]
    problem = cp.Problem(cp.Maximize(cp.log_det(P)), constraints)
    solved = False
    for solver in solver_sequence:
        if solver not in cp.installed_solvers():
            continue
        try:
            logger.debug("John ellipsoid: attempting solver %s", solver)
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            logger.debug("Solver %s raised SolverError in John ellipsoid", solver, exc_info=True)
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            solved = True
            logger.debug("John ellipsoid solved with %s (status=%s)", solver, problem.status)
            break
    if not solved:
        logger.error("John ellipsoid SDP did not converge; last status=%s", problem.status)
        raise ValueError("John ellipsoid SDP did not converge with available solvers.")
    center = np.asarray(c.value, dtype=float)
    P_val = np.asarray(P.value, dtype=float)
    P_sym = 0.5 * (P_val + P_val.T)
    sign, logdet = np.linalg.slogdet(P_sym)
    if sign <= 0:
        raise ValueError("John ellipsoid matrix is not positive definite; cannot compute volume.")
    vol = _unit_ball_volume(d) * float(np.exp(logdet))
    return center, P_sym, vol


def _chebyshev_ball(
    A: np.ndarray,
    b: np.ndarray,
    *,
    solver_sequence: Sequence[str] = ("CLARABEL", "ECOS", "GUROBI", "SCS"),
) -> Tuple[np.ndarray, float, float]:
    m, d = A.shape
    c = cp.Variable(d)
    r = cp.Variable()
    norms = np.linalg.norm(A, axis=1)
    constraints = [A @ c + cp.multiply(norms, r) <= b, r >= 0]
    problem = cp.Problem(cp.Maximize(r), constraints)
    solved = False
    for solver in solver_sequence:
        if solver not in cp.installed_solvers():
            continue
        try:
            logger.debug("Chebyshev ball: attempting solver %s", solver)
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            logger.debug("Solver %s raised SolverError in Chebyshev ball", solver, exc_info=True)
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            solved = True
            logger.debug("Chebyshev ball solved with %s (status=%s)", solver, problem.status)
            break
    if not solved:
        logger.error("Chebyshev ball LP did not converge; last status=%s", problem.status)
        raise ValueError("Chebyshev ball LP did not converge with available solvers.")
    center = np.asarray(c.value, dtype=float)
    radius = float(r.value)
    vol = _unit_ball_volume(d) * radius**d
    return center, radius, vol


def _support_value(
    A: np.ndarray,
    b: np.ndarray,
    *,
    u: np.ndarray,
    solver_sequence: Sequence[str] = ("GUROBI", "CLARABEL", "ECOS", "SCS"),
) -> Tuple[float, np.ndarray]:
    """Return support value h_V(u) = max_x u^T x s.t. A x <= b and a maximizer."""
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


def _outradius_via_support(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray,
    *,
    n_dirs: int = 4096,
    seed: int = 123,
    refine_rounds: int = 2,
) -> float:
    """Approximate outradius by maximizing support function directions."""
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
            supp, _ = _support_value(A, b, u=u)
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


def _min_distance_from_anchor(A: np.ndarray, b: np.ndarray, anchor: np.ndarray) -> float:
    margins = b - A @ anchor
    norms = np.linalg.norm(A, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-valued constraint normal encountered.")
    return float(np.min(margins / norms))


def _orientation_stats(
    A: np.ndarray,
    anchor: np.ndarray,
    *,
    grid_size: int,
) -> Tuple[List[OrientationCDFEntry], float, float]:
    normals = A.astype(float, copy=False)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 0
    normals = normals[valid]
    norms = norms[valid]
    normals = normals / norms[:, None]

    anchor_norm = np.linalg.norm(anchor)
    if anchor_norm <= 1e-12:
        axis = np.zeros(anchor.size, dtype=float)
        axis[0] = 1.0
    else:
        axis = anchor / anchor_norm

    projections = normals - np.outer(normals @ axis, axis)
    proj_norms = np.linalg.norm(projections, axis=1)
    mask = proj_norms > 1e-12
    projections = projections[mask]
    proj_norms = proj_norms[mask]
    if projections.size == 0:
        entries = [OrientationCDFEntry(float(t), float("nan"), float("nan")) for t in np.linspace(0, math.pi, grid_size)]
        return entries, float("nan"), float("nan")

    vectors = projections / proj_norms[:, None]
    cov = vectors.T @ vectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = int(np.argmax(eigvals))
    u0 = eigvecs[:, idx]
    if vectors.shape[0] and (vectors[0] @ u0) < 0:
        u0 = -u0

    angles = np.arccos(np.clip(vectors @ u0, -1.0, 1.0))
    s = anchor.size - 1
    if s < 2:
        grid = np.linspace(0.0, math.pi, grid_size)
        entries = [
            OrientationCDFEntry(float(tau), float(np.mean(angles <= tau)), float("nan"))
            for tau in grid
        ]
        return entries, float("nan"), float("nan")

    alpha = (s - 1) / 2.0
    beta = 0.5

    def _null_cdf(theta: np.ndarray) -> np.ndarray:
        x = np.sin(theta) ** 2
        return special.betainc(alpha, beta, x)

    # KS statistic via uniform transformation
    u = _null_cdf(angles)
    ks_res = stats.kstest(u, "uniform")
    ks_stat = float(ks_res.statistic)
    ks_p_value = float(ks_res.pvalue)

    grid = np.linspace(0.0, math.pi, grid_size)
    entries: List[OrientationCDFEntry] = []
    for tau in grid:
        empirical = float(np.mean(angles <= tau))
        null = float(_null_cdf(np.array([tau]))[0])
        entries.append(OrientationCDFEntry(float(tau), empirical, null))
    return entries, ks_stat, ks_p_value


def _median_pairwise_cosine_distance(
    A: np.ndarray,
    *,
    num_pairs: int,
    seed: int,
) -> float:
    normals = A.astype(float, copy=False)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 0
    normals = normals[valid]
    norms = norms[valid]
    if normals.shape[0] < 2:
        return float("nan")
    normals = normals / norms[:, None]
    m = normals.shape[0]
    total_pairs = m * (m - 1) // 2
    if num_pairs >= total_pairs:
        sim = normals @ normals.T
        triu = np.triu_indices(m, k=1)
        distances = 1.0 - sim[triu]
        return float(np.median(distances))
    weights = (m - 1) - np.arange(m)
    cdf = np.cumsum(weights) / weights.sum()
    rng = np.random.default_rng(seed)
    u = rng.random(num_pairs)
    i_idx = np.searchsorted(cdf, u, side="right")
    j_idx = np.empty(num_pairs, dtype=int)
    for k, i in enumerate(i_idx):
        j_idx[k] = rng.integers(i + 1, m)
    sims = np.einsum("ij,ij->i", normals[i_idx], normals[j_idx])
    distances = 1.0 - sims
    return float(np.median(distances))


def _expected_pairwise_angle(
    samples: np.ndarray,
    anchor: np.ndarray,
    *,
    num_pairs: int,
    seed: int,
    batch: int = 8192,
) -> float:
    diffs = samples - anchor
    norms = np.linalg.norm(diffs, axis=1)
    mask = norms > 1e-12
    if not np.any(mask):
        return float("nan")
    unit = diffs[mask] / norms[mask][:, None]

    anchor_norm = np.linalg.norm(anchor)
    if anchor_norm <= 1e-12:
        axis = np.zeros(anchor.size, dtype=float)
        axis[0] = 1.0
    else:
        axis = anchor / anchor_norm

    projections = unit - np.outer(unit @ axis, axis)
    proj_norms = np.linalg.norm(projections, axis=1)
    mask2 = proj_norms > 1e-12
    if not np.any(mask2):
        return float("nan")
    vectors = projections[mask2] / proj_norms[mask2][:, None]

    m, d = vectors.shape
    if m < 2:
        return float("nan")

    total_pairs = m * (m - 1) // 2
    if total_pairs == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    effective_pairs = min(num_pairs, total_pairs)
    dim_cap = 2000 * d
    if dim_cap > 0:
        effective_pairs = min(effective_pairs, dim_cap)
    if effective_pairs <= 0:
        return float("nan")

    if effective_pairs >= total_pairs:
        sim = vectors @ vectors.T
        tri = np.triu_indices(m, k=1)
        vals = np.clip(sim[tri], -1.0, 1.0)
        angles = np.arccos(vals)
        return float(np.mean(angles))

    weights = (m - 1) - np.arange(m)
    cum_weights = np.cumsum(weights)
    draws = rng.integers(0, cum_weights[-1], size=effective_pairs)
    i_idx = np.searchsorted(cum_weights, draws, side="right")
    prev_cum = np.concatenate(([0], cum_weights[:-1]))
    offsets = draws - prev_cum[i_idx]
    j_idx = i_idx + 1 + offsets

    sims = np.empty(effective_pairs, dtype=float)
    for start in range(0, effective_pairs, max(batch, 1)):
        end = min(start + batch, effective_pairs)
        sims[start:end] = np.sum(
            vectors[i_idx[start:end]] * vectors[j_idx[start:end]],
            axis=1,
        )

    sims = np.clip(sims, -1.0, 1.0)
    angles = np.arccos(sims)
    return float(np.mean(angles))


def compute_all_stats(
    A: np.ndarray,
    b: np.ndarray,
    anchor: np.ndarray | None,
    *,
    pool_size: int,
    hr_burn_in: int,
    hr_thinning: int,
    hr_seed: int,
    num_directions: int,
    epsilon_net_size: int,
    epsilon_seed: int,
    num_pairs: int,
    orientation_grid_size: int,
) -> ConvergenceStats:
    logger.debug(
        "compute_all_stats: m=%d, d=%d, pool_size=%d, epsilon_net_size=%d",
        A.shape[0],
        A.shape[1] if A.size else 0,
        pool_size,
        epsilon_net_size,
    )
    cheby_center, cheby_radius, cheby_vol = _chebyshev_ball(A, b)

    if anchor is None:
        anchor_vec = np.asarray(cheby_center, dtype=float)
        anchor_vec = _strict_interior_point(A, b, anchor_vec)
    else:
        anchor_vec = np.asarray(anchor, dtype=float)
        if anchor_vec.size != A.shape[1]:
            raise ValueError(
                f"Anchor dimension {anchor_vec.size} does not match polytope dimension {A.shape[1]}"
            )

    _ensure_interior(A, b, anchor_vec)

    samples = _hit_and_run_samples(
        A,
        b,
        anchor_vec,
        pool_size=pool_size,
        burn_in=hr_burn_in,
        thinning=hr_thinning,
        seed=hr_seed,
    )
    rng = np.random.default_rng(hr_seed + 1)
    rho_hat, var_hat, _ = _expected_one_step_convergence(
        samples,
        anchor_vec,
        num_directions=num_directions,
        rng=rng,
    )
    rng_dirs = np.random.default_rng(epsilon_seed)
    directions = rng_dirs.normal(size=(epsilon_net_size, anchor_vec.size))
    lambda_hat, r_max_sample, reflection_asym = _minkowski_symmetry(A, b, anchor_vec, directions)
    john_center, john_matrix, john_vol = _john_ellipsoid(A, b)
    r_min = _min_distance_from_anchor(A, b, anchor_vec)
    orientation_cdf, ks_stat, ks_p_value = _orientation_stats(
        A,
        anchor_vec,
        grid_size=orientation_grid_size,
    )
    median_cos = _median_pairwise_cosine_distance(A, num_pairs=num_pairs, seed=epsilon_seed)
    theta_mean = _expected_pairwise_angle(
        samples,
        anchor_vec,
        num_pairs=num_pairs,
        seed=epsilon_seed + 7,
    )
    try:
        r_max_exact = _outradius_via_support(
            A,
            b,
            anchor_vec,
            n_dirs=max(128, epsilon_net_size),
            seed=epsilon_seed,
            refine_rounds=2,
        )
    except Exception as exc:
        logger.warning(
            "Outradius support search failed; falling back to epsilon-net estimate (%s)",
            exc,
        )
        r_max_exact = float(r_max_sample) if np.isfinite(r_max_sample) else float("inf")
    if not np.isfinite(r_max_exact):
        r_max_exact = float("inf")
    if not np.isfinite(r_max_exact) or r_max_exact <= 0 or not np.isfinite(r_min) or r_min < 0:
        sph = 0.0 if not np.isfinite(r_max_exact) else float("nan")
    else:
        sph = float(r_min / r_max_exact)
    logger.debug(
        "compute_all_stats complete: r_max=%.6f, r_min=%.6f, sphericity=%s",
        r_max_exact,
        r_min,
        sph,
    )
    return ConvergenceStats(
        rho_hat=rho_hat,
        var_hat=var_hat,
        lambda_hat=lambda_hat,
        reflection_asymmetry=reflection_asym,
        john_center=john_center.tolist(),
        john_matrix=john_matrix.tolist(),
        john_vol=john_vol,
        cheby_center=cheby_center.tolist(),
        cheby_radius=cheby_radius,
        cheby_ball_vol=cheby_vol,
        r_max_from_a=r_max_exact,
        r_min_from_a=r_min,
        sphericity=sph,
        ks_stat=ks_stat,
        ks_p_value=ks_p_value,
        orientation_cdf=orientation_cdf,
        median_cosine_distance=median_cos,
        expected_theta=theta_mean,
    )


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------


def _load_final_constraints(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    npz = run_dir / "final_version_space.npz"
    if npz.exists():
        with np.load(npz) as data:
            if "A" not in data or "b" not in data:
                raise ValueError(f"NPZ archive {npz} missing 'A' or 'b' datasets")
            A = np.asarray(data["A"], dtype=float)
            b = np.asarray(data["b"], dtype=float).reshape(-1)
        return A, b

    h5 = run_dir / "final_version_space.h5"
    if h5.exists():
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "final_version_space.h5 present but h5py is not installed"
            ) from exc
        with h5py.File(h5, "r") as hfile:
            if "A" not in hfile or "b" not in hfile:
                raise ValueError(f"HDF5 file {h5} missing 'A' or 'b' datasets")
            A = np.asarray(hfile["A"][...], dtype=float)
            b = np.asarray(hfile["b"][...], dtype=float).reshape(-1)
        return A, b

    raise FileNotFoundError(
        f"Could not find final_version_space.(npz|h5) under {run_dir}"
    )


def _constraints_prefix_for_iteration(
    A: np.ndarray,
    b: np.ndarray,
    iteration: int,
    total_iterations: int,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if A.size == 0 or b.size == 0:
        return None, None
    base = max(0, int(A.shape[0]) - int(total_iterations))
    end = int(min(A.shape[0], base + iteration + 1))
    if end <= 0:
        return None, None
    return A[:end], b[:end]


def _read_iterations_csv(path: Path) -> List[int]:
    rows: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "iteration_id" not in reader.fieldnames:
            raise ValueError(f"iterations.csv missing 'iteration_id' column: {reader.fieldnames}")
        for row in reader:
            token = row.get("iteration_id")
            if token is None or token == "":
                continue
            try:
                rows.append(int(token))
            except ValueError:
                continue
    if not rows:
        raise ValueError("No iteration records found in iterations.csv")
    return sorted(set(rows))


def _clean_for_json(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_for_json(val) for key, val in value.items()}
    return value


def _stats_to_row(iteration: int, stats: ConvergenceStats) -> Dict[str, object]:
    payload = asdict(stats)
    payload = _clean_for_json(payload)
    row: Dict[str, object] = {
        "iteration": iteration,
        "rho_hat": payload.get("rho_hat"),
        "var_hat": payload.get("var_hat"),
        "lambda_hat": payload.get("lambda_hat"),
        "reflection_asymmetry": payload.get("reflection_asymmetry"),
        "john_vol": payload.get("john_vol"),
        "cheby_ball_vol": payload.get("cheby_ball_vol"),
        "cheby_radius": payload.get("cheby_radius"),
        "r_max_from_a": payload.get("r_max_from_a"),
        "r_min_from_a": payload.get("r_min_from_a"),
        "sphericity": payload.get("sphericity"),
        "ks_stat": payload.get("ks_stat"),
        "ks_p_value": payload.get("ks_p_value"),
        "median_cosine_distance": payload.get("median_cosine_distance"),
        "expected_theta": payload.get("expected_theta"),
    }

    row["john_center"] = json.dumps(payload.get("john_center"), separators=(",", ":"))
    row["john_matrix"] = json.dumps(payload.get("john_matrix"), separators=(",", ":"))
    row["cheby_center"] = json.dumps(payload.get("cheby_center"), separators=(",", ":"))
    row["orientation_cdf"] = json.dumps(payload.get("orientation_cdf"), separators=(",", ":"))
    row["error"] = ""
    return row


def _empty_row(iteration: int, error: str) -> Dict[str, object]:
    fields = {
        "rho_hat": None,
        "var_hat": None,
        "lambda_hat": None,
        "reflection_asymmetry": None,
        "john_vol": None,
        "cheby_ball_vol": None,
        "cheby_radius": None,
        "r_max_from_a": None,
        "r_min_from_a": None,
        "sphericity": None,
        "ks_stat": None,
        "ks_p_value": None,
        "median_cosine_distance": None,
        "expected_theta": None,
        "john_center": json.dumps(None),
        "john_matrix": json.dumps(None),
        "cheby_center": json.dumps(None),
        "orientation_cdf": json.dumps(None),
    }
    fields.update({"iteration": iteration, "error": error})
    return fields


def _compute_iteration_job(payload: Tuple[object, ...]) -> Dict[str, object]:
    (
        iteration,
        A,
        b,
        pool_size,
        hr_burn_in,
        hr_thinning,
        hr_seed,
        num_directions,
        epsilon_net_size,
        epsilon_seed,
        num_pairs,
        orientation_grid_size,
        log_level,
    ) = payload
    _configure_worker_logging(int(log_level))
    logger = logging.getLogger(__name__)
    logger.debug(
        "Worker starting iteration %d (constraints=%d)",
        iteration,
        A.shape[0],
    )
    try:
        stats = compute_all_stats(
            A,
            b,
            anchor=None,
            pool_size=pool_size,
            hr_burn_in=hr_burn_in,
            hr_thinning=hr_thinning,
            hr_seed=hr_seed,
            num_directions=num_directions,
            epsilon_net_size=epsilon_net_size,
            epsilon_seed=epsilon_seed,
            num_pairs=num_pairs,
            orientation_grid_size=orientation_grid_size,
        )
        row = _stats_to_row(int(iteration), stats)
        logger.debug("Worker finished iteration %d", iteration)
        return row
    except Exception as exc:  # pragma: no cover - worker guard
        logger.exception("Worker failed for iteration %d", iteration)
        return _empty_row(int(iteration), str(exc))


def compute_run_convergence(
    run_dir: Path,
    *,
    output_csv: Path | None,
    pool_size: int,
    hr_burn_in: int,
    hr_thinning: int,
    hr_seed: int,
    num_directions: int,
    epsilon_net_size: int,
    epsilon_seed: int,
    num_pairs: int,
    orientation_grid_size: int,
    jobs: int = 1,
) -> Path:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    iterations_csv = run_dir / "iterations.csv"
    if not iterations_csv.is_file():
        raise FileNotFoundError(f"Missing iterations.csv in {run_dir}")

    iterations = _read_iterations_csv(iterations_csv)
    total_iterations = max(iterations) + 1

    A_full, b_full = _load_final_constraints(run_dir)
    if b_full.shape[0] != A_full.shape[0]:
        raise ValueError(
            f"Constraint mismatch: A has {A_full.shape[0]} rows, b has {b_full.shape[0]} entries"
        )

    effective_jobs = max(1, int(jobs))
    log_level = logging.getLogger().getEffectiveLevel()
    logger.debug(
        "compute_run_convergence: iterations=%d jobs=%d log_level=%s",
        len(iterations),
        effective_jobs,
        logging.getLevelName(log_level),
    )

    pending_payloads: List[Tuple[object, ...]] = []
    rows_map: Dict[int, Dict[str, object]] = {}

    for iteration in iterations:
        Ai, bi = _constraints_prefix_for_iteration(A_full, b_full, iteration, total_iterations)
        if Ai is None or bi is None:
            logger.debug("Iteration %d skipped (insufficient constraints)", iteration)
            rows_map[iteration] = _empty_row(iteration, "insufficient constraints")
            continue
        payload = (
            iteration,
            np.asarray(Ai, dtype=float),
            np.asarray(bi, dtype=float),
            pool_size,
            hr_burn_in,
            hr_thinning,
            hr_seed + iteration,
            num_directions,
            epsilon_net_size,
            epsilon_seed + iteration,
            num_pairs,
            orientation_grid_size,
            log_level,
        )
        pending_payloads.append(payload)

    show_progress = log_level > logging.DEBUG
    progress = (
        _ProgressBar(len(pending_payloads), "Computing iterations")
        if show_progress and pending_payloads
        else None
    )

    if effective_jobs == 1:
        for payload in pending_payloads:
            iteration = int(payload[0])
            logger.debug("Processing iteration %d sequentially", iteration)
            rows_map[iteration] = _compute_iteration_job(payload)
            if progress is not None:
                progress.update()
    else:
        logger.debug("Launching ProcessPoolExecutor with %d workers", effective_jobs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            future_to_iter = {
                executor.submit(_compute_iteration_job, payload): int(payload[0])
                for payload in pending_payloads
            }
            for future in concurrent.futures.as_completed(future_to_iter):
                iteration = future_to_iter[future]
                try:
                    rows_map[iteration] = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Parallel worker crashed for iteration %d", iteration)
                    rows_map[iteration] = _empty_row(iteration, str(exc))
                finally:
                    if progress is not None:
                        progress.update()

    if progress is not None:
        progress.close()

    rows: List[Dict[str, object]] = []
    for iteration in iterations:
        rows.append(rows_map.get(iteration, _empty_row(iteration, "missing result")))

    if output_csv is None:
        output_csv = run_dir / "convergence_stats.csv"

    fieldnames = [
        "iteration",
        "rho_hat",
        "var_hat",
        "lambda_hat",
        "reflection_asymmetry",
        "john_vol",
        "john_center",
        "john_matrix",
        "cheby_ball_vol",
        "cheby_radius",
        "cheby_center",
        "r_max_from_a",
        "r_min_from_a",
        "sphericity",
        "ks_stat",
        "ks_p_value",
        "orientation_cdf",
        "median_cosine_distance",
        "expected_theta",
        "error",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Active learning run directory (per-iteration mode).")
    parser.add_argument("--output-csv", type=Path, help="Destination CSV path (defaults to <run_dir>/convergence_stats.csv).")
    parser.add_argument("--A", type=Path, help="Path to matrix A (.npy/.npz/.csv).")
    parser.add_argument("--b", type=Path, help="Path to vector b (.npy/.npz/.csv).")
    parser.add_argument(
        "--anchor",
        type=str,
        help="Anchor point either as file path or comma-separated list of coordinates.",
    )
    parser.add_argument("--pool-size", type=int, default=5000, help="Number of hit-and-run samples to retain.")
    parser.add_argument("--hr-burn-in", type=int, default=1000, help="Burn-in steps for hit-and-run.")
    parser.add_argument("--hr-thinning", type=int, default=10, help="Thinning interval for hit-and-run.")
    parser.add_argument("--hr-seed", type=int, default=42, help="RNG seed for hit-and-run sampling.")
    parser.add_argument("--num-directions", type=int, default=1000, help="Monte Carlo directions for rho.")
    parser.add_argument("--epsilon-net-size", type=int, default=4096, help="Size of spherical epsilon-net for Minkowski symmetry.")
    parser.add_argument("--epsilon-seed", type=int, default=123, help="RNG seed for epsilon-net and cosine sampling.")
    parser.add_argument("--num-pairs", type=int, default=100000, help="Number of constraint pairs for cosine distance sampling.")
    parser.add_argument("--orientation-grid", type=int, default=181, help="Number of angle grid points for orientation CDF.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for per-iteration stats (1 disables parallelism).")
    parser.add_argument("--output", type=Path, help="Optional path to JSON file for results.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    return parser


def main(args: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    options = parser.parse_args(args)
    logging.basicConfig(level=getattr(logging, options.log_level.upper(), logging.INFO))

    if options.run_dir is not None:
        stats_path = compute_run_convergence(
            options.run_dir,
            output_csv=options.output_csv,
            pool_size=options.pool_size,
            hr_burn_in=options.hr_burn_in,
            hr_thinning=options.hr_thinning,
            hr_seed=options.hr_seed,
            num_directions=options.num_directions,
            epsilon_net_size=options.epsilon_net_size,
            epsilon_seed=options.epsilon_seed,
            num_pairs=options.num_pairs,
            orientation_grid_size=options.orientation_grid,
            jobs=options.jobs,
        )
        logging.info("Wrote per-iteration convergence stats to %s", stats_path)
        return

    if not (options.A and options.b and options.anchor):
        parser.error("Provide --run-dir or the trio --A, --b, --anchor")

    A = _load_array(options.A)
    b = _load_array(options.b).reshape(-1)
    if A.ndim != 2:
        raise ValueError("Matrix A must be 2D.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("Vector b must be 1D with length equal to rows of A.")
    anchor = _parse_anchor(options.anchor, expected_dim=A.shape[1])

    stats_result = compute_all_stats(
        A,
        b,
        anchor,
        pool_size=options.pool_size,
        hr_burn_in=options.hr_burn_in,
        hr_thinning=options.hr_thinning,
        hr_seed=options.hr_seed,
        num_directions=options.num_directions,
        epsilon_net_size=options.epsilon_net_size,
        epsilon_seed=options.epsilon_seed,
        num_pairs=options.num_pairs,
        orientation_grid_size=options.orientation_grid,
    )

    payload = asdict(stats_result)
    payload["orientation_cdf"] = [asdict(entry) for entry in stats_result.orientation_cdf]

    text = json.dumps(payload, indent=2)
    if options.output:
        options.output.write_text(text)
        logging.info("Wrote results to %s", options.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
