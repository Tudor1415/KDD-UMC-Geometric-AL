from __future__ import annotations

# experiments.py

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from learn import learn
from data import Dataset
from ball_tree import build_balltree
from oracles import Oracle  # <- NEW
from metrics import compute_ranking_metrics  # <- NEW
from helpers import k_additive_constraints, augment_with_minimums

################################################################################
logging.basicConfig(
    level=logging.DEBUG, format="[%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger(__name__)


################################################################################
@dataclass
class Config:
    datasets: List[Dataset]
    oracles: List[Oracle]  # <- Oracle objects
    centers: List  # unchanged
    metrics: List[str]
    n_folds: int = 1
    additivity_k: int = 1
    n_iters: int = 20
    random_state: int | None = None
    out_csv: str | Path = "results.csv"
    debug: bool = False


################################################################################
def run_experiments(config: Config) -> pd.DataFrame:
    if config.debug and logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []

    for ds in config.datasets:
        ds.load()
        logger.info("Dataset '%s' – %d rules", ds.name, len(ds.points))
        points_aug_all = augment_with_minimums(ds.points, config.additivity_k)
        logger.debug("Augmented points shape: %s", points_aug_all.shape)

        splitter = KFold(
            n_splits=max(1, config.n_folds),
            shuffle=True,
            random_state=config.random_state,
        )

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(points_aug_all)):
            logger.debug("Processing fold %d", fold_id)
            points_train = points_aug_all[train_idx]
            points_test = points_aug_all[test_idx]
            logger.debug(
                "Train points shape: %s, Test points shape: %s",
                points_train.shape,
                points_test.shape,
            )

            root = build_balltree(points_train)
            logger.debug("Built ball tree for training points")
            n_single = ds.points.shape[1]
            A0, b0 = k_additive_constraints(n_single, config.additivity_k)
            logger.debug(
                "Initial constraints A0 shape: %s, b0 shape: %s", A0.shape, b0.shape
            )

            for oracle in config.oracles:
                logger.info("Using oracle: %s", oracle.__class__.__name__)
                oracle_scores = oracle.score_dataset(ds)[test_idx]

                logger.debug("Oracle scores computed for test set")

                for center_fn in config.centers:
                    logger.info("Using center function: %s", center_fn.__name__)
                    metric_hist: List[Dict[str, Any]] = []
                    diff_hist: List[np.ndarray] = []
                    A, b = A0.copy(), b0.copy()

                    def _hook(iter_id: int, c: np.ndarray, r: float, diff: np.ndarray):
                        logger.debug("Iteration %d: c: %s, r: %f", iter_id, c, r)
                        pred_scores = points_test @ c
                        metrics = compute_ranking_metrics(
                            config.metrics, pred_scores, oracle_scores
                        )
                        logger.debug("Metrics computed: %s", metrics)
                        metrics.update(
                            dataset=ds.name,
                            fold=fold_id,
                            oracle=oracle.__class__.__name__,
                            center=center_fn.__name__,
                            iter=iter_id,
                        )
                        metric_hist.append(metrics)

                        diff_rows.append(
                            dict(
                                dataset=ds.name,
                                oracle=oracle.__class__.__name__,
                                center=center_fn.__name__,
                                iter=iter_id,
                                diff=json.dumps(diff.tolist()),
                            )
                        )

                    learn(
                        root=root,
                        A0=A,
                        b0=b,
                        center_fn=center_fn,
                        oracle=lambda a, b: oracle.compare(a, b),
                        n_iter=config.n_iters,
                        report_hook=_hook,
                    )
                    logger.debug(
                        "Learning completed for center function: %s", center_fn.__name__
                    )
                    rows.extend(metric_hist)

    df_raw = pd.DataFrame(rows)
    group_cols = ["dataset", "oracle", "center", "iter"]
    df_agg = df_raw.groupby(group_cols).agg(
        {m: ["mean", "var"] for m in config.metrics}
    )
    df_agg.columns = [
        "_".join(col) if isinstance(col, tuple) else col for col in df_agg.columns
    ]
    df_agg = df_agg.reset_index()

    # ------------- build the difference-vector frame ---------------------------
    df_diff = pd.DataFrame(
        diff_rows, columns=["dataset", "oracle", "center", "iter", "diffvector"]
    )

    # ------------- save both CSVs ---------------------------------------------
    df_agg.to_csv(config.out_csv, index=False)
    diff_path = (
        Path(config.out_csv)
        .with_stem(Path(config.out_csv).stem + "_diffvectors")
        .with_suffix(".csv")
    )
    df_diff.to_csv(diff_path, index=False)

    logger.info("Saved results  → %s", config.out_csv)
    logger.info("Saved diffvecs → %s", diff_path)
    return df_agg


################################################################################
if __name__ == "__main__":
    #  quick smoke-test ----------------------------------------------------
    from oracles import SumOracle

    rng = np.random.default_rng(0)
    fake = Dataset.__new__(Dataset)
    fake.points = rng.random((300, 4))
    fake.name = "fake"
    fake.n_measures = 4

    cfg = Config(
        datasets=[fake],
        oracles=[SumOracle()],
        centers=[lambda X, *_: X.mean(0)],
        metrics=["recall@10", "map@10", "ndcg@10"],
        n_folds=2,
        n_iters=5,
        debug=True,
        random_state=1,
        out_csv="results_fake.csv",
    )
    run_experiments(cfg)
