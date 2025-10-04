from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _export_tree_h5(path: Path, kd_tree: Any | None, bt_tree: Any | None, d: int) -> None:
    try:
        import h5py  # type: ignore
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return
    with h5py.File(path, "w") as h5:
        if kd_tree is not None:
            g = h5.create_group("kdtree")
            _dump_tree_group(g, kd_tree, d, is_kd=True)
        if bt_tree is not None:
            g = h5.create_group("balltree")
            _dump_tree_group(g, bt_tree, d, is_kd=False)
        try:
            if "A" not in h5:
                h5.create_dataset("A", data=np.zeros((0, d), dtype=float))
            if "b" not in h5:
                h5.create_dataset("b", data=np.zeros((0, 1), dtype=float))
        except Exception:
            pass


def _dump_tree_group(g: Any, tree: Any, d: int, *, is_kd: bool) -> None:
    import h5py  # type: ignore

    nodes: List[Any] = []
    parents: List[int] = []
    queue: List[Tuple[Any, int]] = [(tree.root, -1)]
    id_to_nid: Dict[int, int] = {}
    while queue:
        node, parent = queue.pop(0)
        nid = len(nodes)
        nodes.append(node)
        parents.append(parent)
        id_to_nid[id(node)] = nid
        for ch in getattr(node, "children", []) or []:
            queue.append((ch, nid))

    n = len(nodes)
    node_id = np.arange(n, dtype=np.int64)
    parent_id = np.array(parents, dtype=np.int64)
    is_leaf = np.array([bool(getattr(nd, "is_leaf", False)) for nd in nodes], dtype=np.bool_)
    child_left = np.full(n, -1, dtype=np.int64)
    child_right = np.full(n, -1, dtype=np.int64)
    for nid, nd in enumerate(nodes):
        ch = getattr(nd, "children", []) or []
        if len(ch) >= 1:
            child_left[nid] = id_to_nid.get(id(ch[0]), -1)
        if len(ch) >= 2:
            child_right[nid] = id_to_nid.get(id(ch[1]), -1)

    if is_kd:
        split_dim = np.array([getattr(nd, "split_axis", -1) for nd in nodes], dtype=np.int64)
        split_val = np.array([getattr(nd, "split_val", np.nan) for nd in nodes], dtype=float)
    else:
        split_dim = np.full(n, -1, dtype=np.int64)
        split_val = np.full(n, np.nan, dtype=float)

    centers = np.vstack([np.asarray(getattr(nd, "center"), dtype=float).reshape(1, -1) for nd in nodes])
    radii = np.array([float(getattr(nd, "radius", np.nan)) for nd in nodes], dtype=float)
    vlen_i64 = h5py.vlen_dtype(np.dtype("int64"))
    point_indices = np.empty(n, dtype=object)
    for nid, nd in enumerate(nodes):
        idx = getattr(nd, "indices", None)
        point_indices[nid] = np.asarray(idx, dtype=np.int64) if idx is not None else np.asarray([], dtype=np.int64)

    g.create_dataset("nodes/node_id", data=node_id)
    g.create_dataset("nodes/parent_id", data=parent_id)
    g.create_dataset("nodes/is_leaf", data=is_leaf)
    g.create_dataset("nodes/child_left", data=child_left)
    g.create_dataset("nodes/child_right", data=child_right)
    g.create_dataset("nodes/split_dim", data=split_dim)
    g.create_dataset("nodes/split_val", data=split_val)
    g.create_dataset("nodes/ball_center", data=centers)
    g.create_dataset("nodes/ball_radius", data=radii)
    g.create_dataset("nodes/point_indices", data=point_indices, dtype=vlen_i64)


def _export_search_events_h5(path: Path, events: List[Dict[str, Any]]) -> None:
    try:
        import h5py  # type: ignore
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return
    if not events:
        with h5py.File(path, "w") as h5:
            dt = h5py.string_dtype("ascii", 16)
            h5.create_dataset("events/event_type", data=np.array([], dtype=dt))
            for name, dtype in [
                ("node_id", np.int64),
                ("parent_id", np.int64),
                ("timestamp", float),
                ("lower_bound", float),
                ("upper_bound", float),
            ]:
                h5.create_dataset(f"events/{name}", data=np.array([], dtype=dtype))
        return
    dt = h5py.string_dtype("ascii", 16)
    ev_type = np.array([str(e.get("event_type", "")).upper() for e in events], dtype=dt)
    node_id = np.array([int(e.get("node_id", -1)) for e in events], dtype=np.int64)
    parent_id = np.array([int(e.get("parent_id", -1)) for e in events], dtype=np.int64)
    timestamp = np.array([float(e.get("timestamp", 0.0)) for e in events], dtype=float)
    lower = np.array([float(e.get("lower_bound", np.nan)) for e in events], dtype=float)
    upper = np.array([float(e.get("upper_bound", np.nan)) for e in events], dtype=float)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("events/event_type", data=ev_type)
        h5.create_dataset("events/node_id", data=node_id)
        h5.create_dataset("events/parent_id", data=parent_id)
        h5.create_dataset("events/timestamp", data=timestamp)
        h5.create_dataset("events/lower_bound", data=lower)
        h5.create_dataset("events/upper_bound", data=upper)

