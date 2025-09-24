"""Bottom-up (agglomerative) ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import BallTree, Node
from utils.geometry import enclose_many_balls
from utils.meb import meb
from utils.partitions import axis_median_split


def build_tree(X: np.ndarray, config: Dict | None = None) -> BallTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.bottom_up").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    merge_cost = cfg["merge_cost"]
    pre_leaf = int(cfg["precluster_leaf_size"])
    leaf_size_attr = max(1, int(cfg.get("leaf_size", pre_leaf)))
    if max_children < 2:
        raise ValueError("max_children must be >= 2")
    if merge_cost not in {"radius", "delta_radius", "volume_proxy"}:
        raise ValueError("Unsupported merge_cost option")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def cost_value(children: List[Node], center_radius: Tuple[np.ndarray, float]) -> float:
        _, radius = center_radius
        if merge_cost == "radius":
            return radius
        if merge_cost == "delta_radius":
            max_child = max((child.radius for child in children), default=0.0)
            return radius - max_child
        return radius ** n_features

    def precluster(indices: np.ndarray) -> List[Node]:
        if indices.size == 0:
            return []
        if pre_leaf <= 1:
            nodes = []
            for idx in indices:
                point = data[int(idx)]
                nodes.append(
                    Node(
                        center=point.copy(),
                        radius=0.0,
                        indices=np.array([int(idx)], dtype=np.int64),
                        is_leaf=True,
                    )
                )
            return nodes
        if indices.size <= pre_leaf:
            center, radius = meb(data[indices], method=meb_method)
            return [Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)]
        spreads = np.ptp(data[indices], axis=0)
        axis = int(np.argmax(spreads))
        left, right = axis_median_split(data, indices, axis)
        if left.size == 0 or right.size == 0:
            mid = max(1, indices.size // 2)
            left = indices[:mid]
            right = indices[mid:]
        clusters = []
        clusters.extend(precluster(left))
        clusters.extend(precluster(right))
        return clusters

    active = precluster(indices_all)
    if not active:
        center = np.zeros(n_features, dtype=np.float64)
        root = Node(center=center, radius=0.0, indices=np.array([], dtype=np.int64), is_leaf=True)
        return BallTree(root, n_samples, n_features, leaf_size_attr, "bottom_up", cfg)

    while len(active) > 1:
        best_pair: Optional[Tuple[Node, Node]] = None
        best_enclosure: Optional[Tuple[np.ndarray, float]] = None
        best_cost = np.inf
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                candidate_nodes = [active[i], active[j]]
                enclosure = enclose_many_balls([(node.center, node.radius) for node in candidate_nodes])
                cost = cost_value(candidate_nodes, enclosure)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (active[i], active[j])
                    best_enclosure = enclosure
        if best_pair is None or best_enclosure is None:
            break
        group_nodes: List[Node] = [best_pair[0], best_pair[1]]
        remaining = [node for node in active if node not in group_nodes]
        enclosure = best_enclosure
        if max_children > 2 and remaining:
            target = min(max_children, len(active))
            while len(group_nodes) < target and remaining:
                best_idx = None
                best_extra_enclosure = None
                best_extra_cost = np.inf
                for idx, node in enumerate(remaining):
                    candidate_group = group_nodes + [node]
                    enclosure_candidate = enclose_many_balls(
                        [(child.center, child.radius) for child in candidate_group]
                    )
                    cost = cost_value(candidate_group, enclosure_candidate)
                    if cost < best_extra_cost:
                        best_idx = idx
                        best_extra_enclosure = enclosure_candidate
                        best_extra_cost = cost
                if best_idx is None:
                    break
                group_nodes.append(remaining.pop(best_idx))
                enclosure = best_extra_enclosure
            best_enclosure = enclosure
        for node in group_nodes:
            if node in active:
                active.remove(node)
        parent = Node(center=best_enclosure[0], radius=best_enclosure[1], children=group_nodes, is_leaf=False)
        active.append(parent)

    root = active[0]
    return BallTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size_attr,
        method="bottom_up",
        config=cfg,
    )