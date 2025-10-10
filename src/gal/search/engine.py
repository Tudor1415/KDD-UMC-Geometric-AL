"""Generic branch-and-bound search engine."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..trees.common import GeometricTree, Node
from ..utils import get_array_backend
from .bounds import BallTreeBounds, BoundsStrategy
from .engine_default import run_default_search
from .engine_orientation import run_orientation_search
from .strategies import VisitStrategy, LowerBoundVisitStrategy

logger = logging.getLogger(__name__)


class Search:
    """Dual-tree branch-and-bound search operating on :class:`GeometricTree` nodes."""

    def __init__(
        self,
        *,
        bounder: BoundsStrategy[Node] | None = None,
        strategy: VisitStrategy[Node] | None = None,
    ) -> None:
        self.bounder = bounder or BallTreeBounds()
        self.strategy = strategy or LowerBoundVisitStrategy()
        self._seen_pairs: set[tuple[int, int]] = set()

    def register_seen_pair(self, i: int, j: int) -> None:
        key = (i, j) if i <= j else (j, i)
        self._seen_pairs.add(key)

    def search_pair(
        self,
        tree: GeometricTree | Node,
        X: np.ndarray,
        wc: np.ndarray,
        *,
        tau: float = float("inf"),
        orientation: np.ndarray | None = None,
        maximize_orientation: bool = False,
        return_stats: bool = False,
        dominance_prune: bool = True,
        eps: float = 1e-12,
        time_checkpoints: Optional[Sequence[float]] = None,
        calls_checkpoints: Optional[Sequence[int]] = None,
        collect_events: bool = False,
        use_gpu: bool = False,
    ) -> Tuple[Optional[int], Optional[int], float] | Tuple[Optional[int], Optional[int], float, Dict[str, object]]:
        data = np.ascontiguousarray(X, dtype=np.float64)
        wc = np.asarray(wc, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X must be a 2D array")
        if wc.ndim != 1:
            raise ValueError("wc must be a 1D vector")
        if wc.size != data.shape[1]:
            raise ValueError("wc must have length equal to X.shape[1]")

        backend = get_array_backend(use_gpu)

        orientation_vec: Optional[np.ndarray] = None
        use_orientation = False
        if orientation is not None:
            orientation_arr = np.asarray(orientation, dtype=np.float64).reshape(-1)
            if orientation_arr.size != data.shape[1]:
                raise ValueError("orientation must match feature dimension")
            norm2 = float(np.linalg.norm(orientation_arr))
            if norm2 > float(eps):
                orientation_vec = orientation_arr / norm2
                use_orientation = bool(maximize_orientation)

        root = tree.root if isinstance(tree, GeometricTree) else tree

        if not use_orientation:
            return run_default_search(
                self,
                root,
                data,
                wc,
                backend=backend,
                tau=float(tau),
                return_stats=return_stats,
                dominance_prune=dominance_prune,
                eps=float(eps),
                time_checkpoints=time_checkpoints,
                calls_checkpoints=calls_checkpoints,
                collect_events=collect_events,
            )

        return run_orientation_search(
            self,
            root,
            data,
            wc,
            tau=float(tau),
            orientation=orientation_vec,
            backend=backend,
            return_stats=return_stats,
            dominance_prune=dominance_prune,
            eps=float(eps),
            time_checkpoints=time_checkpoints,
            calls_checkpoints=calls_checkpoints,
            collect_events=collect_events,
        )


def search_pair(
    tree: GeometricTree | Node,
    X: np.ndarray,
    wc: np.ndarray,
    *,
    tau: float = float("inf"),
    orientation: np.ndarray | None = None,
    maximize_orientation: bool = False,
    return_stats: bool = False,
    dominance_prune: bool = True,
    eps: float = 1e-12,
    bounder: BoundsStrategy[Node] | None = None,
    strategy: VisitStrategy[Node] | None = None,
    time_checkpoints: Optional[Sequence[float]] = None,
    calls_checkpoints: Optional[Sequence[int]] = None,
    collect_events: bool = False,
    use_gpu: bool = False,
) -> Tuple[Optional[int], Optional[int], float] | Tuple[Optional[int], Optional[int], float, Dict[str, object]]:
    """Convenience wrapper using the :class:`Search` engine."""

    engine = Search(bounder=bounder, strategy=strategy)
    return engine.search_pair(
        tree,
        X,
        wc,
        tau=tau,
        orientation=orientation,
        maximize_orientation=maximize_orientation,
        return_stats=return_stats,
        dominance_prune=dominance_prune,
        eps=eps,
        time_checkpoints=time_checkpoints,
        calls_checkpoints=calls_checkpoints,
        collect_events=collect_events,
        use_gpu=use_gpu,
    )


__all__ = [
    "Search",
    "search_pair",
]
