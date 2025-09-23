"""Middle-out (anchors hierarchy) ball-tree builder."""

from __future__ import annotations

import importlib
from typing import Dict, List

import numpy as np

from .common import BallTree, Node
from utils.geometry import enclose_many_balls
from utils.meb import meb

EPSILON = 1e-12


def build_tree(X: np.ndarray, config: Dict | None = None) -> BallTree:
    data = np.ascontiguousarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")

    defaults = dict(importlib.import_module("configs.middle_out").DEFAULT)
    cfg = defaults if config is None else {**defaults, **config}

    leaf_size = int(cfg["leaf_size"])
    meb_method = cfg["meb"]
    max_children = int(cfg["max_children"])
    k_anchor = int(cfg["k_anchor"])
    assign_iters = int(cfg["anchor_assign_max_iters"])
    refine_leaves = bool(cfg["refine_leaves"])
    rng_master = np.random.default_rng(cfg["random_state"])
    if max_children < 2:
        raise ValueError("max_children must be >= 2")

    n_samples, n_features = data.shape
    indices_all = np.arange(n_samples, dtype=np.int64)

    def spawn_rng(seed_rng: np.random.Generator) -> np.random.Generator:
        return np.random.default_rng(seed_rng.integers(2**63))

    def choose_seeds(indices: np.ndarray, rng: np.random.Generator, count: int) -> List[int]:
        if indices.size == 0:
            return []
        seeds: List[int] = []
        first_local = int(rng.integers(indices.size))
        seeds.append(int(indices[first_local]))
        pts = data[indices]
        while len(seeds) < count:
            dist2 = np.full(indices.size, np.inf, dtype=np.float64)
            for s in seeds:
                diff = pts - data[s]
                dist2 = np.minimum(dist2, np.einsum("ij,ij->i", diff, diff))
            candidate_local = int(np.argmax(dist2))
            if not np.isfinite(dist2[candidate_local]) or dist2[candidate_local] <= EPSILON:
                break
            candidate = int(indices[candidate_local])
            if candidate in seeds:
                break
            seeds.append(candidate)
        return seeds

    def assign_points(indices: np.ndarray, centers: np.ndarray) -> np.ndarray:
        pts = data[indices]
        dist2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return dist2.argmin(axis=1)

    def build_anchors(indices: np.ndarray, rng: np.random.Generator) -> List[Node]:
        if indices.size == 0:
            return []
        num_anchors = min(k_anchor, indices.size)
        seeds = choose_seeds(indices, rng, num_anchors)
        if not seeds:
            seeds = [int(indices[0])]
            num_anchors = 1
        centers = data[np.array(seeds, dtype=np.int64)].copy()
        assignment = np.full(indices.size, -1, dtype=np.int64)
        pts = data[indices]
        for _ in range(assign_iters):
            new_assignment = assign_points(indices, centers)
            if np.array_equal(new_assignment, assignment):
                break
            assignment = new_assignment
            for a_idx in range(len(centers)):
                members = pts[assignment == a_idx]
                if members.size > 0:
                    centers[a_idx] = members.mean(axis=0)
                else:
                    farthest_local = int(np.argmax(((pts - centers[a_idx]) ** 2).sum(axis=1)))
                    centers[a_idx] = pts[farthest_local]
        if assignment.min() < 0:
            assignment = assign_points(indices, centers)
        clusters = [indices[assignment == a_idx] for a_idx in range(len(centers))]
        for a_idx, cluster in enumerate(clusters):
            if cluster.size == 0:
                donor_idx = max(range(len(clusters)), key=lambda j: clusters[j].size)
                donor = clusters[donor_idx]
                if donor.size <= 1:
                    clusters[a_idx] = donor.copy()
                    clusters[donor_idx] = donor[:0]
                else:
                    moved = donor[-1]
                    clusters[donor_idx] = donor[:-1]
                    clusters[a_idx] = np.array([moved], dtype=indices.dtype)
        clusters = [c for c in clusters if c.size > 0]
        nodes = []
        for cluster in clusters:
            center, radius = meb(data[cluster], method=meb_method)
            nodes.append(Node(center=center, radius=radius, indices=cluster.copy(), is_leaf=True))
        return nodes

    def fallback_anchor_partition(indices: np.ndarray) -> List[Node]:
        pts = data[indices]
        spreads = np.ptp(pts, axis=0)
        axis = int(np.argmax(spreads)) if spreads.size > 0 else 0
        order = indices[np.argsort(pts[:, axis], kind="mergesort")]
        parts = max(2, min(max_children, order.size))
        nodes: List[Node] = []
        for i in range(parts):
            start = (i * order.size) // parts
            end = ((i + 1) * order.size) // parts if i < parts - 1 else order.size
            if end > start:
                subset = order[start:end]
                center, radius = meb(data[subset], method=meb_method)
                nodes.append(Node(center=center, radius=radius, indices=subset.copy(), is_leaf=True))
        return nodes

    def agglomerate(nodes: List[Node]) -> Node:
        active = nodes[:]
        if not active:
            center = np.zeros(n_features, dtype=np.float64)
            return Node(center=center, radius=0.0, indices=np.array([], dtype=np.int64), is_leaf=True)
        while len(active) > 1:
            best_pair = None
            best_enclosure = None
            best_cost = np.inf
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    cand = [active[i], active[j]]
                    enclosure = enclose_many_balls([(node.center, node.radius) for node in cand])
                    radius = enclosure[1]
                    if radius < best_cost:
                        best_cost = radius
                        best_pair = (active[i], active[j])
                        best_enclosure = enclosure
            if best_pair is None or best_enclosure is None:
                break
            group = [best_pair[0], best_pair[1]]
            remaining = [node for node in active if node not in group]
            enclosure = best_enclosure
            if max_children > 2 and remaining:
                target = min(max_children, len(active))
                while len(group) < target and remaining:
                    best_idx = None
                    best_extra_enclosure = None
                    best_extra_radius = np.inf
                    for idx, node in enumerate(remaining):
                        candidate_group = group + [node]
                        enclosure_candidate = enclose_many_balls(
                            [(child.center, child.radius) for child in candidate_group]
                        )
                        radius = enclosure_candidate[1]
                        if radius < best_extra_radius:
                            best_extra_radius = radius
                            best_idx = idx
                            best_extra_enclosure = enclosure_candidate
                    if best_idx is None:
                        break
                    group.append(remaining.pop(best_idx))
                    enclosure = best_extra_enclosure
            for node in group:
                if node in active:
                    active.remove(node)
            parent = Node(center=enclosure[0], radius=enclosure[1], children=group, is_leaf=False)
            active.append(parent)
        return active[0]

    def build_middle(indices: np.ndarray, rng: np.random.Generator) -> Node:
        if indices.size == 0:
            center = np.zeros(n_features, dtype=np.float64)
            return Node(center=center, radius=0.0, indices=indices.copy(), is_leaf=True)
        if indices.size <= leaf_size:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        anchors = build_anchors(indices, rng)
        if not anchors:
            center, radius = meb(data[indices], method=meb_method)
            return Node(center=center, radius=radius, indices=indices.copy(), is_leaf=True)
        if len(anchors) == 1 and anchors[0].indices is not None and anchors[0].indices.size == indices.size:
            anchors = fallback_anchor_partition(indices)
        core = agglomerate(anchors)

        if not refine_leaves:
            return core

        def refine(node: Node, local_rng: np.random.Generator) -> Node:
            if node.is_leaf:
                if node.indices is not None and node.indices.size > leaf_size:
                    sub_rng = spawn_rng(local_rng)
                    return build_middle(node.indices, sub_rng)
                return node
            new_children = []
            for child in node.children:
                sub_rng = spawn_rng(local_rng)
                new_children.append(refine(child, sub_rng))
            node.children = new_children
            node.center, node.radius = enclose_many_balls(
                [(child.center, child.radius) for child in node.children]
            )
            return node

        return refine(core, rng)

    root = build_middle(indices_all, rng_master)
    return BallTree(
        root=root,
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        method="middle_out",
        config=cfg,
    )