from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:  # Optional seaborn
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None


@dataclass
class CurveCI:
    x: np.ndarray
    median: np.ndarray
    low: np.ndarray
    high: np.ndarray


def _maybe_sns():  # pragma: no cover - visual
    if sns is not None:
        sns.set_context("talk")
        sns.set_style("whitegrid")


def plot_anytime_curves(
    curves: Dict[str, CurveCI],
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> plt.Figure:
    _maybe_sns()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, c in curves.items():
        ax.plot(c.x, c.median, label=label)
        ax.fill_between(c.x, c.low, c.high, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
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
) -> plt.Figure:
    _maybe_sns()
    n_cat = len(categories)
    n_m = len(methods)
    x = np.arange(n_cat)
    width = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * n_cat), 4))
    for m_idx, name in enumerate(methods):
        offsets = x - 0.4 + width / 2 + m_idx * width
        ax.bar(offsets, values[:, m_idx], width, label=name)
        yerr_low = values[:, m_idx] - lower[:, m_idx]
        yerr_high = upper[:, m_idx] - values[:, m_idx]
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


def plot_bound_tightness_kde(all_gaps: Dict[str, np.ndarray], *, title: str | None = None) -> plt.Figure:
    _maybe_sns()
    fig, ax = plt.subplots(figsize=(7, 4))
    eps = 1e-9
    for label, gaps in all_gaps.items():
        x = np.log(np.maximum(gaps - np.min(gaps), 0.0) + eps + (np.min(gaps) if np.min(gaps) > 0 else 0))
        if sns is not None:  # pragma: no cover - visual
            sns.kdeplot(x=x, label=label, ax=ax)
        else:
            # Fallback: histogram approximation
            ax.hist(x, bins=50, alpha=0.3, density=True, label=label)
    ax.set_xlabel("Bound Gap: log(UB - LB + ε)")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

