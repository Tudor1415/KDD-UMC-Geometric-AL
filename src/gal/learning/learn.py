"""learn.py
==========
Detailed implementation of the active-learning loop used across the Geometry-
Aware Learning (GAL) toolkit.  The module exposes `learning_loop`, which takes
care of streaming artefacts to disk, interacting with the branch-and-bound
search engine, and maintaining the version-space polytope, and
`project_constraint`, which converts full Möbius constraints into the projected
space understood by the chosen centre function.

Execution outline for :func:`learning_loop` (refer to inline comments for the
line-by-line trace):

1. Initialise on-disk logging helpers so every iteration is captured.
2. Copy the initial constraint system `(A0, b0)` and compute the starting
   polyhedral centre and radius.
3. For each iteration:
   a. Stop early when the radius collapses or the search cannot find a pair.
   b. Derive the admissible ambiguity threshold `tau` from the current radius.
   c. Optionally align the search with the farthest feasible point.
   d. Query the dual-tree search for the most ambiguous pair `(i, j)`.
   e. Ask the oracle for the sign and project the resulting constraint.
   f. Append the constraint, recompute the centre, and persist artefacts.
4. Flush all open streams and write the final version-space snapshot.

Every helper invoked here is documented in :mod:`gal.learning.learn_helpers`;
the Sphinx page ``learning_procedure`` cross-references this module with a
conceptual walkthrough.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
from pathlib import Path
import time
import logging
import math

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
    """Project a Möbius-space constraint into the reduced coordinate system.

    Parameters
    ----------
    h:
        Full constraint vector expressed in the Möbius basis. The last entry
        enforces the sum-to-one condition and becomes the right-hand side after
        projection.

    Returns
    -------
    row, rhs:
        ``row`` is the projected constraint coefficients (last column removed);
        ``rhs`` is the scalar right-hand side created from the dropped entry.

    Notes
    -----
    The projection mirrors :func:`gal.core.space.CapacitySpace.project`.  The
    helper is intentionally separate so tests can exercise the algebra
    independently.
    """

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
    """Execute the active-learning loop and stream diagnostics to ``exp_dir``.

    Parameters
    ----------
    tree:
        Geometry-aware tree (ball tree or kd-tree) built on top of ``X`` for
        dual-tree search.
    X:
        Augmented rule matrix supplied to the search engine.
    space:
        Capacity-space helper providing ``expand_center`` and ``project``.
    A0, b0:
        Initial version-space half-space description.
    center_fn:
        Callable that returns the projected centre for a given ``(A, b)`` pair.
    n_iter:
        Maximum number of iterations before the loop stops.
    tau_cap, tau_multiplier:
        Parameters controlling the ambiguity threshold forwarded to the search
        engine (``tau = min(radius * tau_multiplier, tau_cap)``).
    exp_dir:
        Output directory for CSV and NPZ artefacts.
    oracle_compare:
        Oracle callback returning ``{-1, 0, +1}`` when comparing two vectors.
    collect_events:
        Toggle for saving per-iteration search traces.
    log_every, log_level:
        Logging cadence and verbosity.
    search_strategy:
        String identifier resolved through
        :func:`gal.search.strategies.get_strategy`.
    engine:
        Optional pre-built :class:`gal.search.engine.Search` instance.
    align_orientation:
        When true, attempts to align the search with the farthest feasible point.
    use_gpu:
        Requests the CUDA backend; silently falls back to NumPy when unavailable.

    Returns
    -------
    A, b:
        Final constraint matrices containing every oracle-imposed inequality.
    """
    # Prepare on-disk logging folders and files so partial runs still leave
    # inspectable artefacts.
    csv_writer, it_csv, q_dir = _init_streaming_outputs(exp_dir)

    # Work with local copies of the constraint matrices to avoid mutating the
    # caller's arrays.
    A = np.asarray(A0, dtype=float).copy()
    b = np.asarray(b0, dtype=float).copy()

    logger = logging.getLogger(__name__)

    def _compute_center_state(iter_idx: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Return the projected centre, expanded centre, and Chebyshev radius.

        Encapsulates centre computation so both the initialisation phase and
        the per-iteration updates share identical error handling and logging.
        """

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

    # Establish the starting centre and radius; abort immediately if the
    # polyhedron is infeasible.
    initial_state = _compute_center_state()
    if initial_state is None:
        it_csv.close()
        _finalize_version_space_npz(exp_dir, A, b)
        return A, b

    center_proj, center_full, radius = initial_state

    # Prepare the search engine and cache optional callbacks used for bookkeeping.
    engine = _ensure_search_engine(engine, search_strategy, X)
    register_query = getattr(engine.strategy, "register_queries", None)
    register_pair = getattr(engine, "register_seen_pair", None)

    for it in range(n_iter):
        # Track iteration run-time to add precise timestamps to the CSV output.
        t_start = time.time()
        if not (np.isfinite(radius) and radius > 0):
            break
        # The admissible ambiguity threshold is bounded above by ``tau_cap`` so
        # early iterations remain selective.
        tau = min(radius * float(tau_multiplier), float(tau_cap))
        if log_level <= logging.DEBUG and (it % log_every == 0):
            logging.getLogger(__name__).debug(
                "Iter %d: starting search (tau=%g radius=%g)",
                it,
                tau,
                float(radius),
            )
        # When orientation alignment is enabled, approximate the direction of
        # greatest uncertainty via a farthest-point LP.
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
        # Execute the dual-tree branch-and-bound search for the most ambiguous
        # pair.  ``stats`` bundles optional diagnostics (orientation score,
        # trace events, bound counters).
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
        best_orientation = stats.get("best_orientation") if isinstance(stats, dict) else None

        # Create the iteration sub-directory and append a row to iterations.csv.
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

        # Optionally persist the detailed event trace emitted by the search
        # engine; down-stream analysis scripts consume this structure.
        if collect_events:
            events = list(stats.get("trace", {}).get("events", []))  # type: ignore[arg-type]
            _export_search_events_npz(iter_dir / "search_trace.npz", events)  # type: ignore[arg-type]

        if i is None or j is None:
            break

        # Retrieve the candidate vectors and build the difference used for the
        # oracle query and constraint projection.
        q_a, q_b = X[int(i)], X[int(j)]
        diff = q_a - q_b

        best_orientation = stats.get("best_orientation") if isinstance(stats, dict) else None
        orientation_score: Optional[float] = None
        if best_orientation is not None and math.isfinite(float(best_orientation)):
            orientation_score = float(best_orientation)

        # Feed the encountered queries back to the search strategy when it
        # exposes feedback hooks (used by some diversity-aware strategies).
        if callable(register_query):
            register_query(np.vstack([q_a, q_b]))
        if callable(register_pair):
            register_pair(int(i), int(j))

        # Query the oracle exactly once per iteration.
        y = oracle_compare(q_a, q_b)

        if y != 0:
            # Non-zero oracle response tightens the polytope with a new
            # half-space derived from the difference vector.
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
                    csv_file=it_csv,
                    y=int(y),
                    i=int(i),
                    j=int(j),
                    t_start=t_start,
                    orientation_score=orientation_score,
                )
                break

            center_proj, center_full, radius = updated_state
        else:
            # A neutral vote leaves ``A`` and ``b`` untouched, but we still
            # recompute the radius so the next iteration has an up-to-date tau.
            radius = _chebyshev_radius(A, b, center_proj)

        # Regardless of oracle outcome, persist the query metadata and the
        # updated centre snapshot.
        _record_query_npz(
            it=it,
            diff=diff,
            q_dir=q_dir,
            csv_writer=csv_writer,
            csv_file=it_csv,
            y=int(y),
            i=int(i),
            j=int(j),
            t_start=t_start,
            orientation_score=orientation_score,
        )
        # Store the full-dimensional centre (plus radius/tau) for quick replays.
        _save_center_snapshot(iter_dir, center_full, float(radius), float(tau))

    # Close the CSV stream to ensure all rows reach disk before returning.
    it_csv.close()

    # Write the terminal version-space snapshot so downstream tooling can
    # inspect or resume from the final `(A, b)`.
    _finalize_version_space_npz(exp_dir, A, b)

    return A, b
