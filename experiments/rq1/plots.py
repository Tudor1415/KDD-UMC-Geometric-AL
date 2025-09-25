from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import warnings


@dataclass
class CurveCI:
    x: np.ndarray
    median: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _maybe_sns():  # pragma: no cover - visual
    """Lazily import seaborn with 3rd-party warnings silenced; return module or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import seaborn as sns  # type: ignore
        except Exception:
            return None
    try:
        sns.set_context("talk")
        sns.set_style("whitegrid")
    except Exception:
        pass
    return sns


def plot_anytime_curves(
    curves: Dict[str, CurveCI],
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    use_seaborn: bool = False,
    line_width: float | None = None,
    xscale: str | None = None,
) -> plt.Figure:
    if use_seaborn:
        _maybe_sns()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, c in curves.items():
        if line_width is None:
            ax.plot(c.x, c.median, label=label)
        else:
            ax.plot(c.x, c.median, label=label, linewidth=float(line_width))
        ax.fill_between(c.x, c.low, c.high, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # Axis scaling
    if xscale is not None and str(xscale).lower() in {"log", "symlog", "logit"}:
        # In log scale, ensure lower bound is > 0
        try:
            min_pos = min(float(np.min(c.x[c.x > 0])) for c in curves.values())
            if np.isfinite(min_pos) and min_pos > 0:
                ax.set_xscale(str(xscale).lower())
                ax.set_xlim(left=min_pos)
            else:
                ax.set_xscale(str(xscale).lower())
        except Exception:
            ax.set_xscale(str(xscale).lower())
    else:
        ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scaling_bars(
    categories: List[str],
    methods: List[str],
    values: np.ndarray,  # shape [n_cat, n_methods]
    lower: np.ndarray,   # shape [n_cat, n_methods]
    upper: np.ndarray,   # shape [n_cat, n_methods]
    *,
    ylabel: str = "A@t at 0.2 T_max",
    title: str | None = None,
    use_seaborn: bool = False,
) -> plt.Figure:
    if use_seaborn:
        _maybe_sns()
    n_cat = len(categories)
    n_m = len(methods)
    x = np.arange(n_cat)
    width = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * n_cat), 4))
    for m_idx, name in enumerate(methods):
        offsets = x - 0.4 + width / 2 + m_idx * width
        ax.bar(offsets, values[:, m_idx], width, label=name)
        yerr_low = np.clip(values[:, m_idx] - lower[:, m_idx], 0.0, None)
        yerr_high = np.clip(upper[:, m_idx] - values[:, m_idx], 0.0, None)
        ax.errorbar(offsets, values[:, m_idx], yerr=[yerr_low, yerr_high], fmt='none', ecolor='k', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scaling_lines(
    x: np.ndarray,
    methods: List[str],
    values: np.ndarray,  # shape [n_cat, n_methods]
    lower: np.ndarray,   # shape [n_cat, n_methods]
    upper: np.ndarray,   # shape [n_cat, n_methods]
    *,
    xlabel: str = "n (log scale)",
    title: str | None = None,
    xscale: str | None = "log",
) -> plt.Figure:
    """Line plot with shaded CI bands over a (possibly) logarithmic x-axis.

    x: numeric array encoding problem size (e.g., n). Arrays are sorted by x.
    """
    _maybe_sns()
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m_idx, name in enumerate(methods):
        y = values[:, m_idx][order]
        lo = np.clip(lower[:, m_idx][order], 0.0, None)
        hi = np.clip(upper[:, m_idx][order], 0.0, None)
        ax.plot(xs, y, marker="o", label=name)
        ax.fill_between(xs, lo, hi, alpha=0.2)
    if xscale is not None and str(xscale).lower() in {"log", "symlog", "logit"}:
        ax.set_xscale(str(xscale).lower())
        # avoid zero lower bound
        xmin = float(np.min(xs[xs > 0])) if np.any(xs > 0) else float(np.min(xs))
        ax.set_xlim(left=xmin)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.05)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_bound_tightness_kde(all_gaps: Dict[str, np.ndarray], *, title: str | None = None, use_seaborn: bool = False) -> plt.Figure:
    sns = _maybe_sns() if use_seaborn else None
    fig, ax = plt.subplots(figsize=(7, 4))
    eps = 1e-9
    plotted = False
    for label, gaps in all_gaps.items():
        # Spec: plot KDE of log(UB - LB + eps)
        x = np.log(np.maximum(np.asarray(gaps, dtype=float), 0.0) + eps)
        # Drop non-finite/empty inputs to avoid density normalization warnings
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        if sns is not None:  # pragma: no cover - visual
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.kdeplot(x=x, label=label, ax=ax)
                plotted = True
                continue
            except Exception:
                pass
        # Fallback: histogram approximation
        ax.hist(x, bins=50, alpha=0.3, density=True, label=label)
        plotted = True
    ax.set_xlabel("Bound Gap: log(UB - LB + ε)")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No bound gaps to plot", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    return fig


def plot_heap_curves(
    curves: Dict[str, CurveCI],
    *,
    xlabel: str,
    ylabel: str = "Max Heap Size",
    title: str | None = None,
    use_seaborn: bool = False,
    line_width: float | None = None,
    xscale: str | None = None,
) -> plt.Figure:
    if use_seaborn:
        _maybe_sns()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, c in curves.items():
        if line_width is None:
            ax.plot(c.x, c.median, label=label)
        else:
            ax.plot(c.x, c.median, label=label, linewidth=float(line_width))
        ax.fill_between(c.x, c.low, c.high, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # Axis scaling for x
    if xscale is not None and str(xscale).lower() in {"log", "symlog", "logit"}:
        try:
            min_pos = min(float(np.min(c.x[c.x > 0])) for c in curves.values())
            if np.isfinite(min_pos) and min_pos > 0:
                ax.set_xscale(str(xscale).lower())
                ax.set_xlim(left=min_pos)
            else:
                ax.set_xscale(str(xscale).lower())
        except Exception:
            ax.set_xscale(str(xscale).lower())
    else:
        ax.set_xlim(0, 1)
    # Let y autoscale to the heap sizes
    ax.legend()
    fig.tight_layout()
    return fig
