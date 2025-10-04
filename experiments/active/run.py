"""Active learning experiment runner (thin wrapper).

This module re-exports the CLI and core run functions from `experiments.active.runner`
to keep imports stable while making the codebase modular.

Usage:
  python -m experiments.active.run path/to/config.yaml
"""

from __future__ import annotations

from .runner import main, run_all  # re-export entrypoints


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
