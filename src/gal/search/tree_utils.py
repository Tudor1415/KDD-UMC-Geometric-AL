"""Tree-related helpers used by the search engines."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from ..trees.common import Node


def node_is_leaf(node: Node) -> bool:
    return bool(node.is_leaf or not node.children)


def descendant_size(node: Node, cache: Dict[int, int]) -> int:
    node_id = id(node)
    if node_id in cache:
        return cache[node_id]
    if node.indices is not None and node_is_leaf(node):
        size = int(node.indices.size)
    else:
        size = sum(descendant_size(child, cache) for child in node.children)
    cache[node_id] = size
    return size


def gather_leaf_indices(node: Node) -> np.ndarray:
    stack = [node]
    leaves: list[np.ndarray] = []
    while stack:
        nd = stack.pop()
        if node_is_leaf(nd) and nd.indices is not None:
            leaves.append(nd.indices)
        else:
            stack.extend(nd.children)
    if not leaves:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(leaves).astype(np.int64, copy=False))


def dominates(a: Node, b: Node, eps: float) -> bool:
    amin = a.center - a.radius
    amax = a.center + a.radius
    bmin = b.center - b.radius
    bmax = b.center + b.radius
    return bool(np.all(amin >= bmax - eps) or np.all(bmin >= amax - eps))


__all__ = ["node_is_leaf", "descendant_size", "gather_leaf_indices", "dominates"]
