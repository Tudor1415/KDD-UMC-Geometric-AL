"""RQ1 experiment runner: traces, aggregation, and plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from .eval import aggregate_runs, evaluate_dataset
from .plots import CurveCI, plot_anytime_curves, plot_bound_tightness_kde, plot_scaling_bars


@dataclass
class Rq1Config:
    content: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Rq1Config":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return Rq1Config(content=dict(cfg))

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.content
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def run_dataset(cfg: Rq1Config, *, dataset_name: str, X: np.ndarray, out_dir: Path, rng: np.random.Generator) -> Dict[str, Any]:
    eps = float(cfg.get("global", "epsilon", default=1e-12))
    n_runs = int(cfg.get("global", "num_runs", default=5))
    timing_repeats = int(cfg.get("global", "timing_repeats", default=3))
    t_fracs: List[float] = list(cfg.get("budgets", "time_checkpoints", default=[0.05, 0.1, 0.2, 0.5, 1.0]))
    c_fracs: List[float] = list(cfg.get("budgets", "calls_checkpoints", default=[0.05, 0.1, 0.2, 0.5, 1.0]))
    kd_strategy = str(cfg.get("methods", "dual_kdtree_bnb", "strategy", default="lower_bound"))
    bt_strategy = str(cfg.get("methods", "balltree_bnb", "strategy", default="diversity"))

    runs: List[Dict[str, Any]] = []
    evals = []
    for _ in range(n_runs):
        res = evaluate_dataset(
            X,
            kd_strategy_name=kd_strategy,
            bt_strategy_name=bt_strategy,
            time_fracs=t_fracs,
            call_fracs=c_fracs,
            timing_repeats=timing_repeats,
            eps=eps,
            rng=rng,
        )
        runs.append({k: {"A_time": v.A_time.tolist(), "A_calls": v.A_calls.tolist(), "bound_gaps": v.bound_gaps.tolist()} for k, v in res.items()})
        evals.append(res)

    agg = aggregate_runs(
        evals,
        n_bootstrap=int(cfg.get("global", "n_bootstrap", default=300)),
        ci_level=float(cfg.get("global", "ci_level", default=0.95)),
    )

    # Build curves for plotting
    t = np.array(t_fracs, dtype=float)
    c = np.array(c_fracs, dtype=float)
    kd_t_m, kd_t_lo, kd_t_hi = agg["kd"]["A_time"]
    kd_c_m, kd_c_lo, kd_c_hi = agg["kd"]["A_calls"]
    bt_t_m, bt_t_lo, bt_t_hi = agg["bt"]["A_time"]
    bt_c_m, bt_c_lo, bt_c_hi = agg["bt"]["A_calls"]
    rnd_t_m, rnd_t_lo, rnd_t_hi = agg["rnd"]["A_time"]
    rnd_c_m, rnd_c_lo, rnd_c_hi = agg["rnd"]["A_calls"]

    curves_t = {
        "kd-tree BnB": CurveCI(x=t, median=kd_t_m, low=kd_t_lo, high=kd_t_hi),
        "ball-tree BnB": CurveCI(x=t, median=bt_t_m, low=bt_t_lo, high=bt_t_hi),
        "Random Sampling": CurveCI(x=t, median=rnd_t_m, low=rnd_t_lo, high=rnd_t_hi),
    }
    curves_c = {
        "kd-tree BnB": CurveCI(x=c, median=kd_c_m, low=kd_c_lo, high=kd_c_hi),
        "ball-tree BnB": CurveCI(x=c, median=bt_c_m, low=bt_c_lo, high=bt_c_hi),
        "Random Sampling": CurveCI(x=c, median=rnd_c_m, low=rnd_c_lo, high=rnd_c_hi),
    }

    figs_dir = out_dir / dataset_name
    figs_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_anytime_curves(curves_t, xlabel="Normalized Wall-Clock Time (t/T_max)", ylabel="Anytime Performance (A@t)", title=f"A@time on {dataset_name}")
    fig1.savefig(figs_dir / "A_at_time.png", dpi=150)
    fig2 = plot_anytime_curves(curves_c, xlabel="Normalized Objective Calls (m/P_max)", ylabel="Anytime Performance (A@m)", title=f"A@calls on {dataset_name}")
    fig2.savefig(figs_dir / "A_at_calls.png", dpi=150)

    # Bound tightness KDE
    gaps_kd = np.concatenate([np.array(r["kd"]["bound_gaps"]) for r in runs])
    gaps_bt = np.concatenate([np.array(r["bt"]["bound_gaps"]) for r in runs])
    fig3 = plot_bound_tightness_kde({"kd-tree Bounds": gaps_kd, "ball-tree Bounds": gaps_bt}, title=f"Bound Tightness on {dataset_name}")
    fig3.savefig(figs_dir / "bound_tightness.png", dpi=150)

    # Scaling bar (A@t at 0.2 T_max)
    def pick_at(fracs: List[float], med: np.ndarray, lo: np.ndarray, hi: np.ndarray, f: float = 0.2) -> tuple[float, float, float]:
        idx = int(np.argmin(np.abs(np.array(fracs) - f)))
        return float(med[idx]), float(lo[idx]), float(hi[idx])

    kd_v, kd_l, kd_h = pick_at(t_fracs, kd_t_m, kd_t_lo, kd_t_hi)
    bt_v, bt_l, bt_h = pick_at(t_fracs, bt_t_m, bt_t_lo, bt_t_hi)
    cats = [dataset_name]
    rnd_v, rnd_l, rnd_h = pick_at(t_fracs, rnd_t_m, rnd_t_lo, rnd_t_hi)
    methods = ["kd-tree BnB", "ball-tree BnB", "Random Sampling"]
    vals = np.array([[kd_v, bt_v, rnd_v]], dtype=float)
    los = np.array([[kd_l, bt_l, rnd_l]], dtype=float)
    his = np.array([[kd_h, bt_h, rnd_h]], dtype=float)
    fig4 = plot_scaling_bars(cats, methods, vals, los, his, title="Scaling: A@t at 0.2 T_max")
    fig4.savefig(figs_dir / "scaling.png", dpi=150)

    # Save raw JSON
    (figs_dir / "runs.json").write_text(json.dumps(runs, indent=2))

    return {
        "curves": {
            "time": {k: {"median": v.median.tolist(), "low": v.low.tolist(), "high": v.high.tolist()} for k, v in curves_t.items()},
            "calls": {k: {"median": v.median.tolist(), "low": v.low.tolist(), "high": v.high.tolist()} for k, v in curves_c.items()},
        },
        "figures": {
            "A_at_time": str(figs_dir / "A_at_time.png"),
            "A_at_calls": str(figs_dir / "A_at_calls.png"),
            "bound_tightness": str(figs_dir / "bound_tightness.png"),
            "scaling": str(figs_dir / "scaling.png"),
        },
    }


def main(config_path: str) -> None:  # pragma: no cover - convenience entry
    cfg = Rq1Config.load(config_path)
    rng = np.random.default_rng(int(cfg.get("global", "rng_seed_base", default=1729)))
    out = Path(cfg.get("global", "output_dir", default="./results/rq1"))
    out.mkdir(parents=True, exist_ok=True)
    datasets = cfg.get("datasets", default=[{"name": "TOY", "paths": {"matrix_npy": None}}])
    for ds in datasets:
        name = ds.get("name", "DATA")
        npy_path = ds.get("paths", {}).get("matrix_npy")
        if npy_path is None or not Path(npy_path).exists():
            X = rng.normal(size=(256, 5))
        else:
            X = np.load(npy_path)
        res = run_dataset(cfg, dataset_name=name, X=X, out_dir=out, rng=rng)
        (out / f"{name}_summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run RQ1 experiment")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
