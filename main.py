import os
import numpy as np
from data import Dataset
from pathlib import Path
from poly_centers import *
from experiments import Config
from experiments_parallel import run_experiments
from oracles import ObjectiveMeasureOracle, SurpriseOracle

rules_dir = Path("mined_rules")
dataset_dir = Path("datasets")  # raw transaction tables
matrix_dir = Path("matrices")  # binary rule-item matrices
measures = ["yuleQ", "cosine", "kruskal", "added_value", "certainty"]
RULE_MIN = 10_000  # threshold for enough rules


def line_count(path: Path) -> int:  # data rows (excluding header)
    with path.open("rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 16), b"")) - 1


def matrix_row_count(path: Path) -> int:  # rows in rule-item matrix
    if path.suffix == ".npz":
        with np.load(path) as nz:
            return nz["M"].shape[0]
    else:  # .npy (memory-mapped OK)
        return np.load(path, mmap_mode="r").shape[0]


KEEP_ONLY = {"credit", "magic", "mushroom", "tictactoe", "twitter"}

# ---------------------------------------------------------------- collect files
rule_csvs = {f for f in os.listdir(rules_dir) if f.endswith(".csv")}
transaction_csvs = {f for f in os.listdir(dataset_dir) if f.endswith(".csv")}
matrix_files = {f for f in os.listdir(matrix_dir) if f.endswith((".npy", ".npz"))}

# ---------------------------------------------------------------- build Dataset objects
datasets = []
for tx_file in transaction_csvs:
    base = os.path.splitext(tx_file)[0]  # e.g. "market"

    if base not in KEEP_ONLY:  # skip anything not on the list
        continue

    rule_f = f"{base}_mnr.csv"
    matrix_f = f"{base}_rules.npy"
    if matrix_f not in matrix_files:  # maybe .npz instead
        matrix_f = f"{base}_rules.npz"

    # ------------------------------------------------ existence checks
    if rule_f not in rule_csvs or matrix_f not in matrix_files:
        continue

    rule_path = rules_dir / rule_f
    matrix_path = matrix_dir / matrix_f

    # ------------------------------------------------ minimum-rule check
    n_rules_csv = line_count(rule_path)
    if n_rules_csv <= RULE_MIN:
        continue

    # ------------------------------------------------ row-alignment check
    if matrix_row_count(matrix_path) != n_rules_csv:
        # skip datasets whose matrix is out of sync with mined rules
        continue

    # ------------------------------------------------ build Dataset
    datasets.append(
        Dataset(
            name=base,
            measures=measures,
            dataset_path=rule_path,
            transactions_path=dataset_dir / tx_file,
            item_rule_map_path=matrix_path,
        )
    )
# --------------------------------------------------------------------------- #
# 2.  Experiment configuration
# --------------------------------------------------------------------------- #
cfg = Config(
    datasets=datasets,  # list of Dataset objects
    oracles=[
        SurpriseOracle(prior_type="independent"),
        ObjectiveMeasureOracle("phi"),
    ],
    centers=[
        chebyshev_center,
        minkowski_center,
    ],  # list of centre-finding functions
    metrics=[
        "recall@10",
        "ap@10",
        "recall@p(1)",
        "ap@p(1)",
        "recall@p(5)",
        "ap@p(5)",
        "recall@p(10)",
        "ap@p(10)",
        "recall@p(50)",
        "ap@p(50)",
    ],
    n_folds=3,  # 3-fold cross-validation
    n_iters=50,  # 50 AL iterations
    additivity_k=3,  # 2-additive moment constraints
    out_csv="3_add_results.csv",  # where to dump the metric table
    random_state=42,  # for reproducibility
    debug=True,
)

# --------------------------------------------------------------------------- #
# 3.  Run
# --------------------------------------------------------------------------- #
run_experiments(cfg)
