"""Active learning experiment runner that emits raw artifacts.

Outputs per NOTES/experiments/general.md:
- config.json with run metadata
- final_version_space.h5 with /A and /b (chronological constraints)
- query_vectors.h5 with one dataset per iteration: /query_<k>
- iterations.csv with iteration timeline
- tree.h5 with kd-tree and ball-tree node tables
- per-iteration subdirs iteration_XXX/{search_trace.h5, center_model.npy}

Usage:
  python -m experiments.active.run path/to/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gal.trees import kd_tree as kd
from gal import trees as bt
from gal.search.engine import Search
from gal.centers.poly_centers import analytical_center, chebyshev_center


try:
    import yaml  # type: ignore
except Exception as _e:  # pragma: no cover - CLI convenience
    yaml = None
try:
    import h5py  # type: ignore
except Exception as _e:  # pragma: no cover - runtime error if missing
    h5py = None


def _load_yaml(p: str | Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the config file.")
    with open(p, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _sha256_of_array(X: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    return "sha256:" + h.hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _rand_uid(rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("abcdef0123456789"), size=8))


def _load_points_for_dataset(name: str, ds_entry: Dict[str, Any]) -> np.ndarray:
    paths = ds_entry.get("paths", {}) if isinstance(ds_entry, dict) else {}
    cols = [
        "supportY",
        "supportZ",
        "support",
        "confidence",
        "lift",
        "cosine",
        "phi",
        "kruskal",
        "yuleQ",
        "added_value",
        "certainty",
        "revsupport",
    ]

    mnr_path = paths.get("mnr_rules")
    if mnr_path is None:
        derived = Path("mined_rules") / f"{name.lower()}_mnr.csv"
        if derived.exists():
            mnr_path = str(derived)
    if mnr_path is not None and Path(mnr_path).exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_csv(mnr_path, usecols=cols)
            X = df.to_numpy(dtype=float, copy=False)
            return np.ascontiguousarray(X, dtype=float)
        except Exception:
            import csv as _csv

            with open(mnr_path, "r", encoding="utf-8") as f:
                reader = _csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise RuntimeError(f"Empty CSV: {mnr_path}")
                idx_map: Dict[str, int] = {h.strip(): i for i, h in enumerate(header)}
                use_idx = []
                for c in cols:
                    if c not in idx_map:
                        raise RuntimeError(f"Column '{c}' not found in {mnr_path}")
                    use_idx.append(idx_map[c])
                rows: List[List[float]] = []
                for row in reader:
                    try:
                        rows.append([float(row[i]) for i in use_idx])
                    except Exception:
                        continue
            X = np.asarray(rows, dtype=float)
            return np.ascontiguousarray(X, dtype=float)

    npy_path = paths.get("matrix_npy")
    if npy_path is not None and Path(npy_path).exists():
        return np.ascontiguousarray(np.load(npy_path), dtype=float)

    # synthetic fallback
    rng = np.random.default_rng(int(ds_entry.get("seed", 1729)))
    return rng.normal(size=(256, 12))


def _simplex_constraints(d: int) -> Tuple[np.ndarray, np.ndarray]:
    # Sum <= 1 and w >= 0 (componentwise)
    A = []
    b = []
    A.append(np.ones(d))
    b.append(1.0)
    A.extend([-np.eye(d)[i] for i in range(d)])
    b.extend([0.0] * d)
    return np.asarray(A, dtype=float), np.asarray(b, dtype=float)


def _center_fn(name: str):
    name = (name or "analytic").strip().lower()
    if name in {"analytic", "analytical", "analytic_center", "analytical_center"}:
        return lambda A, b: analytical_center(A, b)
    if name in {"chebyshev", "chebyshev_center"}:
        return lambda A, b: chebyshev_center(A, b)[0]
    raise ValueError(f"Unknown center method: {name}")


def _chebyshev_radius(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    slack = b - A @ c
    norms = np.linalg.norm(A, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(norms > 0, slack / norms, np.inf)
    return max(0.0, float(np.min(vals, initial=np.inf)))


def _export_tree_h5(path: Path, kd_tree: kd.GeometricTree | None, bt_tree: bt.GeometricTree | None, d: int) -> None:
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to write tree.h5")
    with h5py.File(path, "w") as h5:
        if kd_tree is not None:
            g = h5.create_group("kdtree")
            _dump_tree_group(g, kd_tree, d, is_kd=True)
        if bt_tree is not None:
            g = h5.create_group("balltree")
            _dump_tree_group(g, bt_tree, d, is_kd=False)


def _dump_tree_group(g: Any, tree: bt.GeometricTree, d: int, *, is_kd: bool) -> None:
    import h5py  # type: ignore

    # BFS traversal to assign node ids and parents
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

    # Store columns as datasets for readability
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
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to write search_trace.h5")
    if not events:
        # Create empty file with empty datasets
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


@dataclass
class ALConfig:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "ALConfig":
        return ALConfig(raw=_load_yaml(path))

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def run(cfg: ALConfig) -> Path:
    rng = np.random.default_rng(int(cfg.get("global", "seed", default=1729)))
    ds_entry = cfg.get("dataset", default={})
    ds_name = str(ds_entry.get("name", "DATA"))
    X = _load_points_for_dataset(ds_name, ds_entry)
    # Optional uniform downsampling
    max_pts = int(cfg.get("global", "max_points", default=0) or 0)
    if max_pts and X.shape[0] > max_pts:
        idx = rng.choice(X.shape[0], size=max_pts, replace=False)
        X = np.ascontiguousarray(X[idx], dtype=float)

    # Trees (build once)
    kd_enabled = bool(cfg.get("trees", "kd", "enabled", default=True))
    bt_enabled = bool(cfg.get("trees", "ball", "enabled", default=True))
    kd_cfg = cfg.get("trees", "kd", "config", default={}) or {}
    bt_cfg = cfg.get("trees", "ball", "config", default={}) or {}
    bt_method = str(cfg.get("trees", "ball", "method", default="disjoint_greedy"))
    kd_tree_obj = kd.build_tree(X, kd_cfg) if kd_enabled else None
    bt_tree_obj = bt.build_tree(X, bt_cfg, method=bt_method) if bt_enabled else None

    # Oracle – linear sign with w_star
    w_star_raw = cfg.get("oracle", "w_star", default=None)
    if w_star_raw is None:
        u = rng.random(X.shape[1])
        s = float(u.sum())
        w_star = (u / s) if s > 0 else np.ones(X.shape[1]) / float(X.shape[1])
    else:
        w_star = np.array(w_star_raw, dtype=float).reshape(-1)
    if w_star.shape[0] != X.shape[1]:
        raise ValueError("oracle.w_star dimension must match X.shape[1]")

    def oracle(a: np.ndarray, b: np.ndarray) -> int:
        val = float(np.dot(a - b, w_star))
        return 1 if val >= 0 else -1

    # Initial constraints: simplex
    A, b = _simplex_constraints(X.shape[1])
    center_name = str(cfg.get("global", "center", default="analytic"))
    center_fn = _center_fn(center_name)
    center = np.asarray(center_fn(A, b), dtype=float)
    radius = _chebyshev_radius(A, b, center)

    # Output directory
    out_root = Path(cfg.get("global", "output_root", default="./results/al"))
    out_root.mkdir(parents=True, exist_ok=True)
    exp_uid = _rand_uid(rng)
    exp_name = f"{ds_name}_{cfg.get('oracle','type',default='Linear')}_{center_name}_{_timestamp()}_{exp_uid}"
    exp_dir = out_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Serialize config.json
    cfg_json = {
        "experiment_uid": exp_uid,
        "dataset_name": ds_name,
        "dataset_hash": _sha256_of_array(X),
        "oracle_name": str(cfg.get("oracle", "type", default="Linear")),
        "center_name": center_name,
        "random_seed": int(cfg.get("global", "seed", default=1729)),
        "n_iterations": int(cfg.get("global", "n_iter", default=20)),
        "algorithm_parameters": {
            "kd": {"enabled": kd_enabled, **(kd_cfg or {})},
            "ball": {"enabled": bt_enabled, "method": bt_method, **(bt_cfg or {})},
        },
    }
    (exp_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    # Save tree structures once
    _export_tree_h5(exp_dir / "tree.h5", kd_tree_obj, bt_tree_obj, X.shape[1])

    # Prepare global files
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py is required to write HDF5 outputs")
    qv_path = exp_dir / "query_vectors.h5"
    qv_h5 = h5py.File(qv_path, "w")
    it_csv = open(exp_dir / "iterations.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(it_csv)
    csv_writer.writerow(["iteration_id", "query_path", "oracle_response", "i", "j", "timestamp_start", "timestamp_end"])  # schema per notes + indices

    # Iterations
    n_iter = int(cfg.get("global", "n_iter", default=20))
    collect_events = bool(cfg.get("logging", "search_events", default=True))
    engine = Search()
    for it in range(n_iter):
        t_start = time.time()
        # Choose tree: prefer ball-tree if enabled, else kd
        tree = bt_tree_obj or kd_tree_obj
        if tree is None:
            raise RuntimeError("No tree enabled to perform search.")

        # Use current radius to set a finite tau for early stopping
        tau = radius if np.isfinite(radius) and radius > 0 else float("inf")
        i, j, dist, stats = engine.search_pair(
            tree,
            X,
            center,
            tau=float(tau),
            return_stats=True,
            ensure_optimal=False,
            collect_bound_gaps=False,
            collect_events=collect_events,
        )
        events = list(stats.get("trace", {}).get("events", [])) if collect_events else []
        iter_dir = exp_dir / f"iteration_{it:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Save search events
        _export_search_events_h5(iter_dir / "search_trace.h5", events)

        if i is None or j is None:
            # No more informative pairs – stop early
            break
        a = X[int(i)]
        bpt = X[int(j)]
        y = oracle(a, bpt)
        diff = a - bpt
        # Save query vector to H5 (/query_k)
        qds = qv_h5.create_dataset(f"query_{it}", data=np.asarray(diff, dtype=float))
        t_end = time.time()
        csv_writer.writerow([
            it,
            f"/query_{it}",
            y,
            int(i),
            int(j),
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t_start)) + f".{int((t_start%1)*1000):03d}Z",
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t_end)) + f".{int((t_end%1)*1000):03d}Z",
        ])

        # Append constraint: y·(a-b)^T w >= 0  =>  A <- [A; -y*(a-b)], b <- [b; 0]
        h = -float(y) * diff
        A = np.vstack([A, h.reshape(1, -1)])
        b = np.concatenate([b, np.array([0.0])])

        # Recompute center and radius, save center snapshot for this iteration
        center = np.asarray(center_fn(A, b), dtype=float)
        radius = _chebyshev_radius(A, b, center)
        np.save(iter_dir / "center_model.npy", center)

    # Close streaming files
    qv_h5.close()
    it_csv.close()

    # Final version space
    with h5py.File(exp_dir / "final_version_space.h5", "w") as h5:
        h5.create_dataset("A", data=np.asarray(A, dtype=float))
        h5.create_dataset("b", data=np.asarray(b, dtype=float).reshape(-1, 1))

    return exp_dir


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Run active learning experiment (raw logs)")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = ALConfig.load(args.config)
    out_dir = run(cfg)
    print(f"Experiment outputs in: {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
