from __future__ import annotations

# Thin experiments namespace; see src/gal/experiments/run.py

from .run import run_all_from_config, main  # re-export

__all__ = [
    "run_all_from_config",
    "main",
]

