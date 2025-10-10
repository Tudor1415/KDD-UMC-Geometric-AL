"""Statistical routines for convergence diagnostics."""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
from scipy import special

from src.gal.centers.poly_centers import (
    chebyshev_center as _poly_chebyshev_center,
    _hit_and_run_samples,
)

from .core import (
    ensure_interior,
    full_dimensional_form,
    max_distance_socp,
    min_distance_from_anchor,
    strict_interior_point,
    outradius_via_support,
)
from .data import ConvergenceStats, OrientationCDFEntry

logger = logging.getLogger(__name__)


def unit_ball_volume(dim: int) -> float:
    return math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)


def chebyshev_ball(
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    if A.size == 0:
        dim = A.shape[1] if A.ndim == 2 else 0
        return np.zeros(dim, dtype=float), float("nan"), float("nan")

    center_raw, radius = _poly_chebyshev_center(A, b)
    center = np.asarray(center_raw, dtype=float).reshape(-1)
    radius_val = float(radius)
    dim = center.size
    if not np.isfinite(radius_val) or radius_val < 0:
        vol = float("nan")
    else:
        vol = unit_ball_volume(dim) * radius_val**dim
    return center, radius_val, vol


def orientation_stats(
    A: np.ndarray,
    anchor: np.ndarray,
    *,
    grid_size: int,
    collect_entries: bool = True,
) -> List[OrientationCDFEntry]:
    normals = A.astype(float, copy=False)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 0
    normals = normals[valid]
    norms = norms[valid]
    normals = normals / norms[:, None]

    anchor_norm = np.linalg.norm(anchor)
    if anchor_norm <= 1e-12:
        axis = np.zeros(anchor.size, dtype=float)
        if axis.size:
            axis[0] = 1.0
    else:
        axis = anchor / anchor_norm

    projections = normals - np.outer(normals @ axis, axis)
    proj_norms = np.linalg.norm(projections, axis=1)
    mask = proj_norms > 1e-12
    projections = projections[mask]
    proj_norms = proj_norms[mask]
    if projections.size == 0:
        entries = [
            OrientationCDFEntry(float(t), float("nan"), float("nan"))
            for t in np.linspace(0, math.pi, grid_size)
        ] if collect_entries else []
        return entries

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
        if collect_entries:
            grid = np.linspace(0.0, math.pi, grid_size)
            entries = [
                OrientationCDFEntry(float(tau), float(np.mean(angles <= tau)), float("nan"))
                for tau in grid
            ]
        else:
            entries = []
        return entries

    def _null_cdf(theta: np.ndarray, s: int) -> np.ndarray:
        a = (s - 1) / 2.0
        b = 0.5
        th = np.asarray(theta)
        x = np.sin(th)**2                     # in [0,1]
        Ix = special.betainc(a, b, x)         # regularized I_x(a,b)
        F = 0.5 * Ix
        F = np.where(th <= np.pi/2, F, 1.0 - F)
        return F

    entries: List[OrientationCDFEntry] = []
    if collect_entries:
        grid = np.linspace(0.0, math.pi, grid_size)
        for tau in grid:
            empirical = float(np.mean(angles <= tau))
            null = float(_null_cdf(np.array([tau]), anchor.size - 1)[0])
            entries.append(OrientationCDFEntry(float(tau), empirical, null))
    return entries


def median_pairwise_cosine_distance(
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


def expected_pairwise_angle(
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
        if axis.size:
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
    collect_orientation: bool = True,
) -> ConvergenceStats:
    """Compute the reduced set of convergence statistics required downstream."""

    # The inputs often arrive as lists; coerce to numeric arrays up front.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D array")

    logger.debug(
        "compute_all_stats (slim): m=%d, d=%d, pool_size=%d",
        A.shape[0],
        A.shape[1] if A.size else 0,
        pool_size,
    )

    fd_poly = full_dimensional_form(A, b, tol=1e-9, seed=epsilon_seed)
    A_fd = fd_poly.A
    b_fd = fd_poly.b
    dim = fd_poly.dimension

    logger.debug(
        "full-dimensional reduction: dim=%d (original=%d)",
        dim,
        A.shape[1] if A.size else 0,
    )

    if dim == 0:
        orientation_cdf = [
            OrientationCDFEntry(float(t), float("nan"), float("nan"))
            for t in np.linspace(0.0, math.pi, orientation_grid_size)
        ] if collect_orientation else []
        return ConvergenceStats(
            sphericity=float("nan"),
            median_cosine_distance=float("nan"),
            expected_theta=float("nan"),
            orientation_score=float("nan"),
            orientation_cdf=orientation_cdf,
        )

    cheby_center_y, _, _ = chebyshev_ball(A_fd, b_fd)
    cheby_center_y = np.asarray(cheby_center_y, dtype=float).reshape(-1)

    if anchor is None:
        anchor_y = strict_interior_point(A_fd, b_fd, cheby_center_y)
    else:
        anchor_vec_full = np.asarray(anchor, dtype=float).reshape(-1)
        if anchor_vec_full.size != A.shape[1]:
            raise ValueError(
                f"Anchor dimension {anchor_vec_full.size} does not match polytope dimension {A.shape[1]}"
            )
        anchor_y = fd_poly.project_point(anchor_vec_full)
        anchor_y = strict_interior_point(A_fd, b_fd, anchor_y)

    ensure_interior(A_fd, b_fd, anchor_y)

    samples = _hit_and_run_samples(
        A_fd,
        b_fd,
        anchor_y,
        pool_size=pool_size,
        burn_in=hr_burn_in,
        thinning=hr_thinning,
        seed=hr_seed,
    )

    orientation_score = float("nan")
    if collect_orientation:
        try:
            orientation_cdf = orientation_stats(
                A_fd,
                anchor_y,
                grid_size=orientation_grid_size,
                collect_entries=collect_orientation,
            )
        except Exception as exc:
            logger.warning("Orientation statistics failed: %s", exc)
            orientation_cdf = []
    else:
        orientation_cdf = []
    median_cos = median_pairwise_cosine_distance(A_fd, num_pairs=num_pairs, seed=epsilon_seed)
    theta_mean = expected_pairwise_angle(
        samples,
        anchor_y,
        num_pairs=num_pairs,
        seed=epsilon_seed + 7,
    )

    errors: List[str] = []
    r_max_exact = float("nan")
    attempts = [
        ("support", lambda: outradius_via_support(
            A_fd,
            b_fd,
            anchor_y,
            n_dirs=max(128, epsilon_net_size),
            seed=epsilon_seed,
            refine_rounds=2,
        )),
        ("socp", lambda: max_distance_socp(A_fd, b_fd, anchor_y)),
    ]

    for label, fn in attempts:
        try:
            candidate = float(fn())
        except Exception as exc:  # pragma: no cover - defensive fallback
            errors.append(f"{label}: {exc}")
            logger.debug("Outradius %s attempt failed", label, exc_info=True)
            continue
        if not np.isfinite(candidate) or candidate <= 0:
            errors.append(f"{label}: invalid {candidate}")
            continue
        r_max_exact = candidate
        break

    if not np.isfinite(r_max_exact) or r_max_exact <= 0:
        logger.warning(
            "Outradius estimation failed (%s); defaulting to NaN",
            "; ".join(errors) if errors else "no attempts succeeded",
        )
        sphericity = float("nan")
    else:
        r_min = min_distance_from_anchor(A_fd, b_fd, anchor_y)
        if not np.isfinite(r_min) or r_min <= 0:
            sphericity = float("nan")
        else:
            sphericity = float(r_min / r_max_exact)

    logger.debug(
        "compute_all_stats (slim) complete: dim=%d r_max=%.6f sphericity=%s",
        dim,
        r_max_exact,
        sphericity,
    )

    return ConvergenceStats(
        sphericity=sphericity,
        median_cosine_distance=median_cos,
        expected_theta=theta_mean,
        orientation_score=orientation_score,
        orientation_cdf=orientation_cdf,
    )


__all__ = [
    "compute_all_stats",
    "chebyshev_ball",
    "median_pairwise_cosine_distance",
    "expected_pairwise_angle",
    "orientation_stats",
    "unit_ball_volume",
]
