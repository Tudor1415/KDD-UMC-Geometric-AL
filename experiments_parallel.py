"""Parallel experiment runner (fold/centre parallel, self-writing, *verbose*)
========================================================================

Adds **rich debug logging** inside each fold so you can see progress even
when 32 processes are crunching away:

* Child processes configure their own ``logging`` if none is present.
* High-level ``INFO`` milestones:
  * fold start / finished Ball-Tree / each oracle + centre / CSV write.
* Fine-grained ``DEBUG``:
  * per-iteration hook – first iter and then every ``log_every`` steps
    (default = 10) – includes elapsed time and current metric snapshot.
* Log records include the **process name** and **fold id** for clarity.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple
from concurrent.futures import (
    as_completed,
    ThreadPoolExecutor,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from learn import learn
from ball_tree import build_balltree
from metrics import compute_ranking_metrics
from experiments import Config, Dataset  # Dataset for type hints
from helpers import (
    augment_with_minimums,
    k_additive_constraints,
)

__all__ = ["run_experiments"]

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    logging.getLogger(__name__).debug(
        "%s finished in %.2fs", label, time.perf_counter() - t0
    )


@contextmanager
def _csv_lock(path: Path):
    """Portable advisory lock (``*.lock`` file next to CSV)."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as fh:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Worker – one *oracle × centre* --------------------------------------------
# ---------------------------------------------------------------------------


def _run_pair(
    ds,
    root,
    pts_all: np.ndarray,
    train_idx: Tuple[int],
    test_idx: Tuple[int],
    add_k: int,
    n_iters: int,
    fold_id: int,
    oracle,
    centre_fn,
    metrics: List[str],
    out_csv: Path,
    diff_csv: Path,
    log_every: int = 10,
) -> None:
    """
    Learn one (oracle, centre) pair, append rows to CSV immediately.
    Runs in its own thread (or process) → **never blocks anyone else**.
    """

    # ------------------------------------------------------------------
    # logging -----------------------------------------------------------
    if not logging.getLogger().handlers:  # child process safety
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s │ %(processName)-12s │ %(levelname)-7s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    log = logging.getLogger(__name__)
    oracle_name = oracle.name
    centre_name = centre_fn.__name__
    ds_name = ds.name

    # ------------------------------------------------------------------
    log.info("Fold %d ‹%s› – start %s × %s", fold_id, ds_name, oracle_name, centre_name)

    # ------------------------------------------------------------------
    # constraints & bookkeeping
    n_single = len(ds.measures)
    A0, b0 = k_additive_constraints(n_single, add_k)
    oracle_scores = oracle.score_dataset(ds)[list(test_idx)]

    rows: List[Dict[str, Any]] = []
    last_t = time.perf_counter()

    # ------------------------------------------------------------------
    def _hook(iter_id: int, c: np.ndarray, r: float, diff: np.ndarray, answer: int):
        nonlocal last_t
        pred = pts_all[list(test_idx)] @ c
        m = compute_ranking_metrics(metrics, pred, oracle_scores)
        now = time.perf_counter()
        m["iter_time"] = now - last_t
        last_t = now
        m.update(
            dataset=ds_name,
            fold=fold_id,
            oracle=oracle_name,
            centre=centre_name,
            iter=iter_id,
            diffvector=json.dumps(diff.tolist()),
            answer=answer,
            weight=json.dumps(c.tolist()),
            radius=r,
        )
        rows.append(m)

        # Check if diff is all zeros
        if np.all(diff == 0):
            log.error(
                "Fold %d ‹%s› – %s/%s iter %d → diff is all zeros!",
                fold_id,
                ds_name,
                oracle_name,
                centre_name,
                iter_id,
            )

        if iter_id == 0 or iter_id % log_every == 0:
            log.debug(
                "Fold %d ‹%s› – %s/%s iter %d → %s, radius: %.4f",
                fold_id,
                ds_name,
                oracle_name,
                centre_name,
                iter_id,
                {k: m[k] for k in list(metrics)[:2]},
                r,
            )

    # ------------------------------------------------------------------
    with _timed(f"[{ds_name} f{fold_id}] {oracle_name} × {centre_name}"):
        learn(
            root,
            A0,
            b0,
            centre_fn,
            oracle.compare_vectors,
            n_iter=n_iters,
            report_hook=_hook,
        )

    # ------------------------------------------------------------------
    # flush rows directly to disk – no waiting for others ---------------
    diff_cols = ["dataset", "centre", "oracle", "diffvector", "answer"]
    df = pd.DataFrame(rows)

    with _csv_lock(out_csv):
        hdr = not out_csv.exists()
        df.drop(columns=["diffvector", "answer"]).to_csv(
            out_csv, mode="a", header=hdr, index=False
        )

    with _csv_lock(diff_csv):
        hdr = not diff_csv.exists()
        df[diff_cols].to_csv(diff_csv, mode="a", header=hdr, index=False)

    log.info(
        "Fold %d ‹%s› – done %s × %s (%d rows)",
        fold_id,
        ds_name,
        oracle_name,
        centre_name,
        len(rows),
    )


# ---------------------------------------------------------------------------
# Worker – one *fold*  (build Ball-Tree here, once)  -------------------------
# ---------------------------------------------------------------------------
def _run_fold(job, *, pair_workers: int = 10) -> None:
    (
        ds,
        pts_aug,
        train_idx,
        test_idx,
        add_k,
        n_iters,
        fold_id,
        oracles,
        centres,
        metrics,
        out_csv,
        diff_csv,
    ) = job

    log = logging.getLogger(__name__)

    # ---------- build shared Ball-Tree once per fold --------------------
    root = build_balltree(pts_aug[list(train_idx)])
    log.debug("Fold %d ‹%s› – Ball-Tree built once.", fold_id, ds.name)

    # ---------- spawn oracle × centre workers ---------------------------
    with ThreadPoolExecutor(max_workers=pair_workers) as pool:
        futures = []
        for oracle in oracles:
            clone_oracle = oracle.clone()
            clone_oracle.set_dataset(ds)
            for centre_fn in centres:
                futures.append(
                    pool.submit(
                        _run_pair,
                        ds=ds,
                        root=root,  # ← shared tree
                        pts_all=pts_aug,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        add_k=add_k,
                        n_iters=n_iters,
                        fold_id=fold_id,
                        oracle=clone_oracle,
                        centre_fn=centre_fn,
                        metrics=metrics,
                        out_csv=out_csv,
                        diff_csv=diff_csv,
                    )
                )

        for fut in as_completed(futures):
            fut.result()


# ---------------------------------------------------------------------------
# Driver --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def run_experiments(config: Config) -> None:
    """
    Run folds **serially** (one after the other).  Inside each fold the centre
    functions still run in a ThreadPool, but we never start >1 fold at a time,
    so the global thread count stays comfortably below 100.
    """

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    out_csv = Path(config.out_csv)
    diff_csv = out_csv.parent / f"{out_csv.stem}_diffvectors{out_csv.suffix}"

    for ds_idx, ds in enumerate(config.datasets):
        ds.load()
        log.info(
            "(%d/%d) Dataset '%s' – %d rules",
            ds_idx + 1,
            len(config.datasets),
            ds.name,
            len(ds),
        )

        with _timed(f"augment {ds.name}"):
            pts_aug = augment_with_minimums(ds.points, config.additivity_k)  # type: ignore[arg-type]

        kf = KFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.random_state
        )

        # ---- run folds **sequentially** ---------------------------------
        for fid, (tr, te) in enumerate(kf.split(pts_aug)):
            _run_fold(
                (
                    ds,
                    pts_aug,
                    tuple(tr),
                    tuple(te),
                    config.additivity_k,
                    config.n_iters,
                    fid,
                    config.oracles,
                    config.centers,
                    config.metrics,
                    out_csv,
                    diff_csv,
                ),
            )

    log.info("✓ All experiments finished → %s", out_csv)
