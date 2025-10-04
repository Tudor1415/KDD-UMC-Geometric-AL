from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - CLI convenience
    yaml = None


def _load_yaml(p: str | Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the config file.")
    with open(p, "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _rand_uid(rng) -> str:
    return "".join(rng.choice(list("abcdef0123456789"), size=8))


def _sanitize_tag(s: str) -> str:
    return str(s).strip().replace(" ", "_")


def _configure_runtime_from_config(cfg: "ALConfig") -> None:
    """Apply lightweight runtime settings driven by the YAML config.

    Currently supports:
      - global.numexpr_max_threads -> sets NUMEXPR_MAX_THREADS env var
      - numexpr.max_threads        -> same as above (alternative section)
    """
    try:
        val = cfg.get("global", "numexpr_max_threads", default=None)
        if val is None:
            val = cfg.get("numexpr", "max_threads", default=None)
        if val is not None:
            os.environ["NUMEXPR_MAX_THREADS"] = str(int(val))
    except Exception:
        # Never fail run due to a tuning knob
        pass


DEFAULT_MEASURE_COLUMNS = [
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


def _normalize_measure_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        cleaned = str(values).strip()
        return [cleaned] if cleaned else []
    try:
        cleaned = [str(v).strip() for v in list(values)]
    except TypeError as exc:
        raise ValueError(f"Expected an iterable of measure names, got {values!r}") from exc
    return [c for c in cleaned if c]


_ALLOWED_PATH_KEYS = {
    "mnr_rules",
    "matrix_npy",
    "dataset_path",
    # Optional artefacts for Dataset-based oracles
    "transactions_path",
    "item_rule_map_path",
    # Allow shorter aliases too
    "transactions",
    "item_rule_map",
}


def _dataset_entry_from_cfg(cfg: "ALConfig", item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if item is not None and not isinstance(item, dict):
        raise ValueError("The new schema requires each datasets[] entry to be a mapping.")
    base_name = str(cfg.get("experiment", "dataset_name", default="DATA"))
    name = str((item or {}).get("name", base_name))
    if not name:
        raise ValueError("experiment.dataset_name must be provided in the configuration.")
    base_paths_raw = cfg.get("paths", default={}) or {}
    base_paths = {k: v for k, v in base_paths_raw.items() if k in _ALLOWED_PATH_KEYS and v is not None}
    item_paths_raw = ((item or {}).get("paths", {}) or {})
    item_paths = {k: v for k, v in item_paths_raw.items() if k in _ALLOWED_PATH_KEYS and v is not None}
    paths: Dict[str, Any] = {}
    paths.update(base_paths)
    paths.update(item_paths)
    entry: Dict[str, Any] = {}
    if item:
        entry.update({k: v for k, v in item.items() if k not in {"paths", "measures", "name"}})
    entry["name"] = name
    entry["paths"] = paths
    for source in (
        (item or {}).get("measures"),
        cfg.get("experiment", "measures", default=None),
        cfg.get("global", "measures", default=None),
    ):
        measures = _normalize_measure_list(source)
        if measures:
            entry["measures"] = measures
            break
    entry.setdefault("measures", list(DEFAULT_MEASURE_COLUMNS))
    return entry


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
