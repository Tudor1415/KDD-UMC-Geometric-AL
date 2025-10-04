"""Lightweight configuration helpers for active-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml  # type: ignore


# ---------------------------------------------------------------------------
# YAML loading and high-level config wrapper
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    """Return the parsed YAML file as a plain dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


@dataclass
class ALConfig:
    """Tiny facade around the raw YAML mapping with dot-style access."""

    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "ALConfig":
        return ALConfig(raw=_load_yaml(path))

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self.raw
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return default
            value = value[key]
        return value

# ---------------------------------------------------------------------------
# Dataset entry normalisation
# ---------------------------------------------------------------------------

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

_ALLOWED_PATH_KEYS = {
    "dataset_path",
    "mnr_rules",
    "matrix_npy",
    "transactions_path",
    "transactions",
    "item_rule_map_path",
    "item_rule_map",
}


def _resolve_dataset_name(cfg: ALConfig, entry: Dict[str, Any]) -> str:
    name = str(entry.get("name", cfg.get("experiment", "dataset_name", default="DATA")))
    if not name:
        raise ValueError("experiment.dataset_name must be provided in the configuration")
    return name


def _merge_allowed_paths(cfg: ALConfig, entry: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(_filter_paths(cfg.get("paths", default={}) or {}))
    merged.update(_filter_paths(entry.get("paths", {}) or {}))
    return merged


def _filter_paths(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in raw.items() if k in _ALLOWED_PATH_KEYS and v is not None}


def _resolve_measures(cfg: ALConfig, entry: Dict[str, Any]) -> List[str]:
    for source in (
        entry.get("measures"),
        cfg.get("experiment", "measures", default=None),
        cfg.get("global", "measures", default=None),
    ):
        names = _normalize_measures(source)
        if names:
            return names
    return list(DEFAULT_MEASURE_COLUMNS)


def _normalize_measures(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if isinstance(values, Iterable):
        return [str(v).strip() for v in values if str(v).strip()]
    raise ValueError(f"Expected a string or iterable of measure names, got {values!r}")


def dataset_entry_from_cfg(cfg: ALConfig, item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a single dataset entry, merging global defaults with overrides."""

    if item is not None and not isinstance(item, Mapping):
        raise ValueError("datasets[] entries must be mappings")
    entry = dict(item or {})

    name = _resolve_dataset_name(cfg, entry)
    paths = _merge_allowed_paths(cfg, entry)
    measures = _resolve_measures(cfg, entry)
    extras = {k: v for k, v in entry.items() if k not in {"name", "paths", "measures"}}

    return {
        "name": name,
        "paths": paths,
        "measures": measures,
        **extras,
    }


__all__ = [
    "ALConfig",
    "DEFAULT_MEASURE_COLUMNS",
    "dataset_entry_from_cfg",
]
