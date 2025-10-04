from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type checkers; avoid importing at runtime
    from .run import run_all_from_config as _run_all_from_config
    from .run import main as _main

__all__ = ["run_all_from_config", "main"]


def run_all_from_config(*args, **kwargs):
    from .run import run_all_from_config as _delegate

    return _delegate(*args, **kwargs)


def main(*args, **kwargs):
    from .run import main as _delegate

    return _delegate(*args, **kwargs)
