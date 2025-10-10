"""learn.py
==========
An *active‑learning* loop that iteratively tightens a half‑space description
of an unknown direction **q⋆**.  At each round we:

1.   **center selection** – given the current feasible region
     `P = {q : A q ≤ b}` choose a center `c` via an arbitrary `center_fn` –
     e.g. `poly_centers.chebyshev_center`, `analytical_center`, …
2.   **Uncertainty sampling** – over all data points stored in a Ball‑Tree we
     find a pair `(a, b)` whose difference vector is *most ambiguous* wrt `c`
     using the best‑first search from *simple_GeometricTree.py* (minimises
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

from typing import Any, Callable, Optional, Tuple
from pathlib import Path
import time
import logging

from gal.centers import _chebyshev_radius
import numpy as np
from gal.search.engine import Search
from .learn_helpers import (
    _init_streaming_outputs,
    _ensure_search_engine,
    _log_iteration,
    _export_search_events_npz,
    _record_query_npz,
    _save_center_snapshot,
    _finalize_version_space_npz,
    _farthest_point_socp,
)


def project_constraint(h: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (row, rhs) for constraint vector expressed in Möbius basis."""

    vec = np.asarray(h, dtype=float).reshape(-1)
    if vec.size == 0:
        raise ValueError("Constraint vector must be non-empty")
    return vec[:-1] - vec[-1], -float(vec[-1])



def learning_loop(
    *,
    tree: Any,
    X: np.ndarray,
    space: Any,
    A0: np.ndarray,
    b0: np.ndarray,
    center_fn: Callable,
    n_iter: int,
    tau_cap: float,
    tau_multiplier: float,
    exp_dir: Path,
    oracle_compare: Callable[[np.ndarray, np.ndarray], int],
    collect_events: bool,
    log_every: int,
    log_level: int,
    search_strategy: str,
    engine: Optional[Search] = None,
    align_orientation: bool = False,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Active learning loop with on-disk logging and search traces.

    Persists per-iteration queries and snapshots as NPZ files under `exp_dir`.
    """
    # Outputs: iterations.csv and queries/ (NPZ-based, no h5py)
    csv_writer, it_csv, q_dir = _init_streaming_outputs(exp_dir)

    # Init version space
    A = np.asarray(A0, dtype=float).copy()
    b = np.asarray(b0, dtype=float).copy()

    logger = logging.getLogger(__name__)

    def _compute_center_state(iter_idx: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return the projected/expanded center and radius or log and abort."""

        try:
            proj = np.asarray(center_fn(A, b), dtype=float)
        except ValueError as exc:
            phase = "initialisation" if iter_idx is None else f"iteration {iter_idx}"
            logger.error(
                "Failed to compute center during %s: %s. Stopping learning loop.",
                phase,
                exc,
            )
            return None
        full = space.expand_center(proj)
        radius_val = _chebyshev_radius(A, b, proj)
        return proj, full, radius_val

    initial_state = _compute_center_state()
    if initial_state is None:
        it_csv.close()
        _finalize_version_space_npz(exp_dir, A, b)
        return A, b

    center_proj, center_full, radius = initial_state

    engine = _ensure_search_engine(engine, search_strategy, X)
    register_query = getattr(engine.strategy, "register_queries", None)
    register_pair = getattr(engine, "register_seen_pair", None)

    for it in range(n_iter):
        t_start = time.time()
        if not (np.isfinite(radius) and radius > 0):
            break
        tau = min(radius * float(tau_multiplier), float(tau_cap))
        if log_level <= logging.DEBUG and (it % log_every == 0):
            logging.getLogger(__name__).debug(
                "Iter %d: starting search (tau=%g radius=%g)",
                it,
                tau,
                float(radius),
            )
        orientation_vec: Optional[np.ndarray] = None
        if align_orientation and np.isfinite(radius) and radius > 0:
            farthest_proj, _ = _farthest_point_socp(A, b, center_proj)
            if farthest_proj is not None:
                try:
                    farthest_full = space.expand_center(farthest_proj)
                except ValueError as exc:
                    logger.warning("Failed to expand farthest point during iteration %d: %s", it, exc)
                else:
                    direction = np.asarray(farthest_full, dtype=float) - np.asarray(center_full, dtype=float)
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-12:
                        orientation_vec = direction / norm
                    else:
                        logger.debug(
                            "Iter %d: farthest-point direction nearly zero; skipping orientation alignment",
                            it,
                        )
            else:
                logger.debug(
                    "Iter %d: SOCP farthest-point solver did not return a point; orientation alignment disabled",
                    it,
                )
        i, j, dist, stats = engine.search_pair(
            tree,
            X,
            center_full,
            tau=float(tau),
            orientation=orientation_vec,
            maximize_orientation=bool(orientation_vec is not None),
            return_stats=True,
            collect_events=collect_events,
            use_gpu=use_gpu,
        )

        iter_dir = _log_iteration(
            it,
            i,
            j,
            dist,
            float(radius),
            log_level=log_level,
            log_every=log_every,
            exp_dir=exp_dir,
        )

        # Save search events (NPZ)
        if collect_events:
            events = list(stats.get("trace", {}).get("events", []))  # type: ignore[arg-type]
            _export_search_events_npz(iter_dir / "search_trace.npz", events)  # type: ignore[arg-type]

        if i is None or j is None:
            break

        q_a, q_b = X[int(i)], X[int(j)]
        diff = q_a - q_b

        if callable(register_query):
            register_query(np.vstack([q_a, q_b]))
        if callable(register_pair):
            register_pair(int(i), int(j))

        y = oracle_compare(q_a, q_b)

        if y != 0:
            constraint = -float(y) * diff

            proj_row, proj_rhs = space.project(constraint)
            A = np.vstack([A, proj_row.reshape(1, -1)])
            b = np.concatenate([b, np.array([proj_rhs], dtype=float)])

            updated_state = _compute_center_state(iter_idx=it)
            if updated_state is None:
                _record_query_npz(
                    it=it,
                    diff=diff,
                    q_dir=q_dir,
                    csv_writer=csv_writer,
                    y=int(y),
                    i=int(i),
                    j=int(j),
                    t_start=t_start,
                )
                break

            center_proj, center_full, radius = updated_state
        else:
            radius = _chebyshev_radius(A, b, center_proj)

        _record_query_npz(
            it=it,
            diff=diff,
            q_dir=q_dir,
            csv_writer=csv_writer,
            y=int(y),
            i=int(i),
            j=int(j),
            t_start=t_start,
        )
        _save_center_snapshot(iter_dir, center_full, float(radius), float(tau))

    # Close CSV stream
    it_csv.close()

    # Final constraints snapshot (NPZ only)
    _finalize_version_space_npz(exp_dir, A, b)

    return A, b
