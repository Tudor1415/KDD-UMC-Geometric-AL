from __future__ import annotations

# metrics.py
import re
import numpy as np
from scipy.stats import rankdata
from typing import List, Dict, Tuple
from sklearn.metrics import (
    recall_score,  # confusion‑matrix recall
    average_precision_score,  # AP / MAP for binary relevance
)


# ─────────────────────────────────── helpers ──────────────────────────────────
def _parse_metric_name(metric: str) -> Tuple[str, int | float | None, str]:
    metric = metric.lower()
    if "@" not in metric:
        return metric, None, "all"

    name, kpart = metric.split("@", 1)
    if m := re.fullmatch(r"p\((\d+(?:\.\d+)?)\)", kpart):
        return name, float(m[1]) / 100.0, "%"
    if kpart.endswith("%"):
        return name, float(kpart[:-1]) / 100.0, "%"
    return name, int(kpart), "k"


# ─────────────────────── tie-aware compute_ranking_metrics ───────────────────
def compute_ranking_metrics(
    metric_names: List[str],
    pred_scores: np.ndarray,
    oracle_scores: np.ndarray,
) -> Dict[str, float]:
    n = len(pred_scores)

    # 1. tie-aware 1-based ranks (1 = best)
    true_rank = rankdata(-oracle_scores, method="min")
    pred_rank = rankdata(-pred_scores, method="min")
    out: Dict[str, float] = {}
    for m in metric_names:
        name, kparam, mode = _parse_metric_name(m)
        k = (
            int(kparam)
            if mode == "k"
            else max(1, int(np.ceil(n * kparam))) if mode == "%" else n
        )

        y_true_bin = (true_rank <= k).astype(int)  # shape (n,)
        y_pred_bin = (pred_rank <= k).astype(int)  # shape (n,)

        if name == "recall":
            recall_at_k = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            out[m] = recall_at_k

        elif name == "ap":
            ap = average_precision_score(y_true_bin, y_pred_bin)
            out[m] = ap

        else:
            raise ValueError(f"Unknown metric '{m}'")

    return out


__all__ = [
    "compute_ranking_metrics",
]
