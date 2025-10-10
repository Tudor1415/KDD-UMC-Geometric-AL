"""Helpers for transforming strategy priorities into sortable tuples."""

from __future__ import annotations

from typing import Sequence, Tuple


def normalize_priority(score: Sequence[float] | float | int) -> Tuple[float, ...]:
    if isinstance(score, (float, int)):
        return (float(score),)
    if isinstance(score, tuple):
        return tuple(float(x) for x in score)
    return tuple(float(x) for x in score)


__all__ = ["normalize_priority"]
