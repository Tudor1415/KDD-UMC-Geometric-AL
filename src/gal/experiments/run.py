"""
Thin CLI wrapper for running active-learning experiments based on a YAML config.

This module intentionally reuses the existing implementation in
`experiments.active.runner` to avoid duplicating logic. It simply loads the
config and delegates to `run_all`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ALConfig
from .runner import run_all


def run_all_from_config(config_path: str | Path) -> Path:
    cfg = ALConfig.load(config_path)
    return run_all(cfg)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run AL experiments from config")
    ap.add_argument("config", type=str, help="Path to YAML config file")
    args = ap.parse_args()
    out_dir = run_all_from_config(args.config)
    print(str(out_dir))


if __name__ == "__main__":  # pragma: no cover
    main()
