from __future__ import annotations

import itertools

# data.py
# ===============================
from pathlib import Path
from functools import wraps
from collections import defaultdict
from typing import Dict, List, Sequence, Union, Tuple

import numpy as np
import pandas as pd

Rule = Dict[str, Union[List[int], Dict[str, float]]]


def _hash_measures(vec: np.ndarray) -> int:
    return hash(vec.tobytes())


def _needs_load(method):
    @wraps(method)
    def wrapper(self, *args, **kw):
        if not self._loaded:
            raise RuntimeError("call Dataset.load() first")
        return method(self, *args, **kw)

    return wrapper


def augment_with_minimums(
    X: np.ndarray,
    k: int,
    *,
    return_index_map: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Augment input samples with minimums over all subsets up to size k."""
    X = np.asanyarray(X)
    if X.ndim != 2:
        raise ValueError('X must be a 2-D array (samples x features)')

    m, n = X.shape
    if not (1 <= k <= n):
        raise ValueError('k must satisfy 1 <= k <= n_features')

    subset_list: List[Tuple[int, ...]] = []
    new_columns: List[np.ndarray] = []
    for r in range(2, k + 1):
        for comb in itertools.combinations(range(n), r):
            subset_list.append(comb)
            new_columns.append(np.min(X[:, comb], axis=1))

    if new_columns:
        X_aug = np.hstack([X] + [col[:, None] for col in new_columns])
    else:
        X_aug = X.copy()

    if return_index_map:
        return X_aug, subset_list
    return X_aug


class Dataset:
    """
    CSV-based association-rule set plus optional artefacts.

    Mandatory
    ---------
    • rules CSV  → self.df

    Optional
    --------
    • transactions CSV  → self.transactions
    • rule-item matrix  → self.item_rule_map  (rows align with rules)
    """

    _item_matrix_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    # -------------------------------------------------------------- init
    def __init__(
        self,
        dataset_path: Path | str,
        *,
        transactions_path: Path | str | None = None,
        item_rule_map_path: Path | str | None = None,
        measures: Sequence[str] | None = None,
        name: str | None = None,
        float_dtype=np.float32,
        max_rows: int | None = None,
        drop_duplicate_measure_vectors: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.transactions_path = Path(transactions_path) if transactions_path else None
        self.item_rule_map_path = (
            Path(item_rule_map_path) if item_rule_map_path else None
        )

        self.name = name or self.dataset_path.stem
        self.float_dtype = float_dtype
        self.measures = list(measures) if measures else []
        self.all_measures: list[str] | None = None
        if max_rows is not None and int(max_rows) <= 0:
            raise ValueError("max_rows must be positive when provided")
        self.max_rows = int(max_rows) if max_rows is not None else None
        self.drop_duplicate_measure_vectors = bool(drop_duplicate_measure_vectors)

        # filled by load()
        self.df: pd.DataFrame | None = None
        self.points: np.ndarray | None = None
        self.transactions: pd.DataFrame | None = None
        self.item_rule_map: np.ndarray | None = None
        self.items: np.ndarray | None = None

        self._hash2rows: Dict[int, List[int]] | None = None
        self._n_rules: int | None = None
        self._loaded = False

        # basic load diagnostics
        self.rows_read: int | None = None
        self.duplicates_dropped: int | None = None

    # -------------------------------------------------------------- load
    def load(self) -> "Dataset":
        """
        Read the rule CSV, optionally drop duplicate measure-vectors,
        and propagate that selection everywhere (points, helper maps,
        rule-item matrix).
        """
        if self._loaded:
            return self

        # ---------- discover numeric measure columns -----------------
        sample = pd.read_csv(self.dataset_path, nrows=5)
        self.all_measures = [
            c for c in sample.columns if c not in {"antecedent", "consequent"}
        ]
        if not self.measures:
            self.measures = self.all_measures.copy()

        # ---------- read full rules CSV ------------------------------
        dtype_map = {m: self.float_dtype for m in self.measures}
        dtype_map.update(
            {"antecedent": "string[pyarrow]", "consequent": "string[pyarrow]"}
        )

        use_pyarrow = pd.__version__ >= "2.0" and self.max_rows is None
        read_csv_kw = dict(
            filepath_or_buffer=self.dataset_path,
            dtype=dtype_map,
            engine="pyarrow" if use_pyarrow else "c",
        )
        if self.max_rows is not None:
            read_csv_kw["nrows"] = self.max_rows

        self.df = pd.read_csv(**read_csv_kw)

        # ---------- optionally drop duplicate measure-vectors --------
        n_loaded_rules = len(self.df)
        keep_mask_np: np.ndarray | None = None
        duplicates_dropped = 0
        if self.drop_duplicate_measure_vectors and self.measures:
            keep_mask = ~self.df.duplicated(subset=self.measures, keep="first")
            keep_mask_np = keep_mask.to_numpy()
            duplicates_dropped = int(n_loaded_rules - int(keep_mask_np.sum()))
            self.df = self.df.loc[keep_mask].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        # ---------- rebuild helpers on filtered rules ----------------
        self.points = self.df[self.measures].to_numpy(
            copy=False, dtype=self.float_dtype
        )

        self._hash2rows = defaultdict(list)
        for i, v in enumerate(self.points):
            self._hash2rows[_hash_measures(v)].append(i)

        self._n_rules = len(self.df)
        self.rows_read = n_loaded_rules
        self.duplicates_dropped = duplicates_dropped

        # ---------- optional transactions CSV ------------------------
        if self.transactions_path:
            self.transactions = pd.read_csv(
                self.transactions_path, dtype=np.uint8, sep=";"
            )

        # ---------- optional rule-item matrix ------------------------
        if self.item_rule_map_path:
            p = self.item_rule_map_path
            if p.suffix == ".npz":
                with np.load(p) as nz:
                    M_full = nz["M"]  # 2-D array  (n_rules × n_items)
                    items = nz["items"]
            else:  # .npy memory-mapped
                M_full = np.load(p, mmap_mode="r")
                items_path = p.with_suffix(".items.npy")
                if not items_path.is_file():
                    raise FileNotFoundError(f"Companion item list {items_path} missing")
                items = np.load(items_path)

            # ---------- sanity checks --------------------------------
            if M_full.shape[1] != len(items):
                raise ValueError(
                    f"Matrix width ({M_full.shape[1]}) ≠ items length ({len(items)})"
                )
            if M_full.shape[0] < n_loaded_rules:
                raise ValueError(
                    f"item_rule_map rows ({M_full.shape[0]}) < rules read from CSV "
                    f"({n_loaded_rules})"
                )

            # ---------- apply *same* mask to rows --------------------
            # first align lengths: take only the rules we actually read
            M_subset = M_full[:n_loaded_rules]
            if keep_mask_np is not None:
                # then drop the duplicate-measure rows
                self.item_rule_map = M_subset[keep_mask_np]
            else:
                self.item_rule_map = M_subset
            self.items = items.astype(int, copy=False)

        self._loaded = True
        return self

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse_item_list(entry: Union[str, int, float]) -> List[int]:
        if isinstance(entry, (int, np.integer)):
            return [int(entry)]
        if isinstance(entry, float) and entry.is_integer():
            return [int(entry)]
        if isinstance(entry, str):
            entry = entry.strip().strip('"').strip("'")
            return [] if not entry else [int(tok.strip()) for tok in entry.split(",")]
        raise ValueError(f"Cannot parse item list from {entry!r}")

    @_needs_load
    def get_rule_dict(self, idx: int) -> Rule:
        row = self.df.iloc[int(idx)]
        measures_dict = {col: float(row[col]) for col in self.all_measures}
        return {
            "row_idx": int(idx),
            "antecedent": self._parse_item_list(row["antecedent"]),
            "consequent": self._parse_item_list(row["consequent"]),
            "measures": measures_dict,
        }

    @_needs_load
    def vector_to_rule(self, vec: np.ndarray) -> Rule:
        vec_np = np.asarray(vec, dtype=self.float_dtype).reshape(-1)
        rows = self._hash2rows.get(_hash_measures(vec_np))
        if not rows:
            raise KeyError("Vector not found in dataset")
        return self.get_rule_dict(rows[0])

    @_needs_load
    def __len__(self) -> int:
        return self._n_rules

        # ------------------------------------------------------ new helper -----

    @property
    @_needs_load
    def n_transactions(self) -> int:
        """
        Total number of transactions in the source data.

        • If `transactions` were loaded (transactions_path given),
          we just return its number of rows.

        • Otherwise we try to find a column called 'n_transactions'
          in the rule CSV (some generators put it there).

        Raise AttributeError if nothing works.
        """
        if self.transactions is not None:
            return len(self.transactions)

        if self.df is not None and "n_transactions" in self.df.columns:
            # assume the value is constant across rows
            return int(self.df["n_transactions"].iloc[0])

        raise AttributeError(
            "Dataset does not know the total transaction count. "
            "Load the transactions file or store the value in a "
            "'n_transactions' column."
        )
