"""Shared data structures for ball-tree builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class Node:
    """Generic n-ary ball-tree node."""

    __slots__ = ("center", "radius", "children", "indices", "is_leaf")

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        children: Optional[List["Node"]] = None,
        indices: Optional[np.ndarray] = None,
        is_leaf: bool = False,
    ) -> None:
        self.center = center
        self.radius = float(radius)
        self.children = [] if children is None else children
        self.indices = indices
        self.is_leaf = is_leaf


@dataclass(slots=True)
class BallTree:
    root: Node
    n_samples: int
    n_features: int
    leaf_size: int
    method: str
    config: dict