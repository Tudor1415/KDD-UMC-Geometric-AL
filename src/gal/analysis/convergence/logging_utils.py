"""Logging helpers shared by convergence analysis workers."""

from __future__ import annotations

import logging
from threading import Lock


_worker_log_configured = False
_worker_log_lock = Lock()


def configure_worker_logging(level: int) -> None:
    """Initialise logging in worker processes once."""
    global _worker_log_configured
    if _worker_log_configured:
        return
    with _worker_log_lock:
        if _worker_log_configured:
            return
        logging.basicConfig(level=level)
        _worker_log_configured = True


__all__ = ["configure_worker_logging"]
