from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import DEFAULT_MEASURE_COLUMNS, _normalize_measure_list


def _load_points_for_dataset(name: str, ds_entry: Dict[str, Any]) -> np.ndarray:
    paths = ds_entry.get("paths", {}) if isinstance(ds_entry, dict) else {}
    if isinstance(ds_entry, dict):
        measures = _normalize_measure_list(ds_entry.get("measures"))
        if measures:
            ds_entry["measures"] = measures
        else:
            ds_entry["measures"] = list(DEFAULT_MEASURE_COLUMNS)
        cols = list(ds_entry["measures"])
    else:
        cols = list(DEFAULT_MEASURE_COLUMNS)

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
                use_idx: List[int] = []
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
    seed_raw = ds_entry.get("seed", 1729) if isinstance(ds_entry, dict) else 1729
    try:
        seed_val = int(seed_raw)
    except Exception:
        seed_val = 1729
    rng = np.random.default_rng(seed_val)
    X = rng.normal(size=(256, len(cols)))
    return np.ascontiguousarray(X, dtype=float)

