"""Light-weight progress reporting for CLI scripts."""

from __future__ import annotations

import sys
import threading
from typing import TextIO


class ProgressBar:
    """Minimal text progress bar for CLI output."""

    def __init__(self, total: int, message: str = "Progress", stream: TextIO | None = None) -> None:
        self.total = max(total, 0)
        self.message = message
        self.count = 0
        self._lock = threading.Lock()
        self._stream: TextIO = stream if stream is not None else sys.stderr
        self._last_len = 0
        self._done = False

    def update(self, step: int = 1) -> None:
        if self.total <= 0 or step <= 0:
            return
        with self._lock:
            self.count = min(self.total, self.count + step)
            pct = (100.0 * self.count / self.total) if self.total else 100.0
            text = f"{self.message}: {self.count}/{self.total} ({pct:5.1f}%)"
            padding = max(0, self._last_len - len(text))
            self._stream.write("\r" + text + " " * padding)
            self._stream.flush()
            self._last_len = len(text)
            if self.count >= self.total and not self._done:
                self._stream.write("\n")
                self._stream.flush()
                self._done = True

    def close(self) -> None:
        with self._lock:
            if not self._done and self.total > 0:
                self._stream.write("\n")
                self._stream.flush()
                self._done = True


__all__ = ["ProgressBar"]
