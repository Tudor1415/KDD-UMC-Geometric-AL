# oracles.py
from __future__ import annotations

import random
from typing import Dict, Callable
from abc import ABC, abstractmethod

import copy
import numpy as np
import pandas as pd
from typing import TypeVar
from data import Dataset, Rule
from priors import PRIOR_FACTORY, Prior

T = TypeVar("T", bound="Oracle")

# --------------------------------------------------------------------------- #
#  Base class with mandatory `name`
# --------------------------------------------------------------------------- #


class Oracle(ABC):
    """Scalar-scoring oracle that can work with vectors *after* a dataset is set."""

    def __init__(self, name: str):
        if not name:
            raise ValueError("Oracle needs a non-empty `name`.")
        self.name = name
        self.ds: Dataset | None = None
        self._n_measures: int | None = None

    # --------------- lifecycle -------------------------------------------
    def set_dataset(self, dataset: Dataset) -> None:
        """Attach a dataset so that vector-level calls become possible."""
        self.ds = dataset
        self._n_measures = len(dataset.measures)

    def _ensure_dataset(self):
        if self.ds is None:
            raise RuntimeError(
                f"Dataset not set for oracle '{self.name}'. "
                "Call oracle.set_dataset(ds) first."
            )

    # --------------- core scoring API ------------------------------------
    @abstractmethod
    def score(self, rule: Rule) -> float: ...

    # --------------- convenience wrappers --------------------------------
    def compare(self, a_rule: Rule, b_rule: Rule) -> int:
        sa, sb = self.score(a_rule), self.score(b_rule)
        return 1 if sa > sb else -1 if sb > sa else 0

    def compare_vectors(self, a_vec: np.ndarray, b_vec: np.ndarray) -> int:
        """Used by the learner – promotes vectors to rule dicts on the fly."""
        self._ensure_dataset()
        ra = self.ds.vector_to_rule(a_vec[: self._n_measures])
        rb = self.ds.vector_to_rule(b_vec[: self._n_measures])
        return self.compare(ra, rb)

    def score_dataset(self, dataset: Dataset) -> np.ndarray:
        """Slow fallback; subclasses can override for speed."""
        return np.vectorize(lambda idx: self.score(dataset.get_rule_dict(idx)))(
            np.arange(len(dataset.df))
        )

    def clone(self: T) -> T:
        """
        Cheap, concurrency-safe clone.

        • The new Oracle carries over *all* immutable configuration
          (name, hyper-params, etc.) but starts with ``ds = None``.

        • You **must** call ``clone.set_dataset(ds)`` in each worker
          before using ``compare_vectors`` or ``score_dataset``.
        """
        clone: T = copy.copy(self)  # shallow is enough
        clone.ds = None  # detach shared state
        return clone


# --------------------------------------------------------------------------- #
# 1. Objective-measure oracle
# --------------------------------------------------------------------------- #
class ObjectiveMeasureOracle(Oracle):
    def __init__(self, measure_name: int, *, name: str | None = None):
        super().__init__(name or f"{measure_name}-oracle")
        self.measure_name = measure_name

    def score(self, rule):
        return rule["measures"][self.measure_name]

    def score_dataset(self, dataset) -> np.ndarray:
        col = dataset.df[self.measure_name]
        return col.astype(float, copy=False).to_numpy()

    @staticmethod
    def random():
        measure = "phi"
        return ObjectiveMeasureOracle(measure)


# --------------------------------------------------------------------------- #
# 2. Item-based oracle
# --------------------------------------------------------------------------- #
class SumOracle(Oracle):
    def __init__(self, measures: list, *, name=None):
        super().__init__(name or f"Sum-Oracle")
        self.measures = measures

    def score(self, rule):
        sum = 0.0
        for measure in self.measures:
            sum += rule["measures"][measure]
        return sum

    def score_dataset(self, dataset):
        return np.sum(dataset.df[self.measures], axis=1)


# --------------------------------------------------------------------------- #
# 3. Surprise oracle
# --------------------------------------------------------------------------- #
class SurpriseOracle(Oracle):
    """
    Parameters
    ----------
    prior_type : str
        Key registered in `priors.PRIOR_FACTORY`.
    phi : callable(fo, fe) -> float, optional
        Surprise function.  Default: `lambda fo, fe: log(fo) - log(fe)`.
    name : str, optional
        Identifier used in logs / result tables.
    **prior_kwargs
        Extra arguments forwarded to the prior constructor.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        prior_type: str = "independent",
        *,
        phi: Callable[[float, float], float] | None = None,
        name: str | None = None,
        **prior_kwargs,
    ):
        prior_type = prior_type.lower()
        if prior_type not in PRIOR_FACTORY:
            raise ValueError(
                f"Unknown prior_type '{prior_type}'. "
                f"Valid keys: {list(PRIOR_FACTORY)}"
            )

        self.prior_type = prior_type
        self.prior_kwargs = prior_kwargs
        self.phi = phi or (lambda fo, fe: np.log(fo + 1e-12) - np.log(fe + 1e-12))

        self._prior: Prior | None = None
        super().__init__(name or f"surprise-{prior_type}")

    # ------------------------------------------------------------ set_dataset
    def set_dataset(self, dataset: Dataset) -> None:
        super().set_dataset(dataset)

        # ------------------------------------------------ instantiate prior
        if self.prior_type in {"independent", "higher-order"}:
            # 1) grab the item universe from the transaction table columns
            if dataset.transactions is None:
                raise AttributeError(
                    "Dataset.transactions missing – required for log-linear priors."
                )
            all_items = dataset.items

            # 2) create the prior with that list
            self._prior = PRIOR_FACTORY[self.prior_type](all_items, **self.prior_kwargs)

            # 3) fit on the full transaction table
            self._prior.fit(dataset.transactions)

        else:  # priors that don’t need the item list in __init__
            if dataset.transactions is None:
                raise AttributeError("Dataset.transactions missing")
            self._prior = PRIOR_FACTORY[self.prior_type](**self.prior_kwargs)
            self._prior.fit(dataset.transactions)

    # ---------------------------------------------------------------- single-rule score
    def score(self, rule: Rule) -> float:
        if self._prior is None:
            raise RuntimeError("Prior not fitted – call set_dataset() first")

        f_obs = float(rule["measures"]["support"])
        f_exp = self._prior.expected(rule)
        return self.phi(f_obs + 1e-12, f_exp + 1e-12)

    # ---------------------------------------------------------- compare helpers
    def compare(self, a_rule: Rule, b_rule: Rule) -> int:
        sa, sb = self.score(a_rule), self.score(b_rule)
        return 1 if sa > sb else -1 if sb > sa else 0

    def compare_vectors(self, a_vec: np.ndarray, b_vec: np.ndarray) -> int:
        self._ensure_dataset()
        ra = self.ds.vector_to_rule(a_vec[: len(self.ds.measures)])
        rb = self.ds.vector_to_rule(b_vec[: len(self.ds.measures)])
        return self.compare(ra, rb)

    # ---------------------------------------------------------- batch scoring
    def score_dataset(self, dataset: Dataset) -> np.ndarray:
        """
        Vectorised scoring for all rules in `dataset`.
        Falls back to a Python loop only if the prior lacks `expected_batch`.
        """
        self._ensure_dataset()

        # fast extraction via pandas — clear and concise
        f_obs_vec = dataset.df["support"].to_numpy(dtype=float, copy=False)

        f_exp_vec = (
            self._prior.expected_batch(dataset)
            if hasattr(self._prior, "expected_batch")
            else None
        )

        if f_exp_vec is None:
            # generic slow path
            f_exp_vec = np.empty_like(f_obs_vec)
            for i in range(len(dataset)):
                f_exp_vec[i] = self._prior.expected(dataset.get_rule_dict(i))

        return self.phi(f_obs_vec + 1e-12, f_exp_vec + 1e-12)

    # ---------------------------------------------------------------- clone
    def clone(self) -> "SurpriseOracle":
        clone = copy.copy(self)
        clone.ds = None
        clone._prior = copy.deepcopy(self._prior)
        return clone


# --------------------------------------------------------------------------- #
# 4. MDL oracle  (using supportX / supportY column names)
# --------------------------------------------------------------------------- #
class MDLOracle(Oracle):
    """
    MDL-based interestingness: positive values mean the rule compresses
    the data better than chance.

    Expected columns in dataset.df
    --------------------------------
      support        – |X∪Y|
      supportX      – |X|
      supportY      – |Y|          (single-item rhs assumed)
      len_X          – |X| (number of items, *not* support)
      len_Y          – |Y|
    """

    def __init__(
        self,
        c0: float = 8.0,  # bits: “there is a rule”
        c_item: float = 4.0,  # bits per item mentioned in the rule
        name: str | None = None,
    ):
        super().__init__(name or "mdl-oracle")
        self.c0 = c0
        self.c_item = c_item
        self._n_total: int | None = None
        self._supportY: np.ndarray | None = None

    # ---------- helpers --------------------------------------------------
    @staticmethod
    def _binary_entropy(p: np.ndarray) -> np.ndarray:
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    # ---------- dataset attachment --------------------------------------
    def set_dataset(self, dataset: Dataset) -> None:
        super().set_dataset(dataset)
        self._n_total = int(dataset.n_transactions)
        # cache |Y| counts for every rule
        self._supportY = dataset.df["supportY"].to_numpy(dtype=np.int32, copy=False)
        if dataset.item_rule_map is not None:
            self._k_items = dataset.item_rule_map.sum(axis=1).astype(np.int16)
        else:
            # slow fallback, executed only once
            self._k_items = (
                dataset.df["antecedent"].str.count(",").add(1, fill_value=0)
                + dataset.df["consequent"].str.count(",").add(1, fill_value=0)
            ).astype(np.int16)

        # cache inside the dataframe for the fast vectorised path
        dataset.df["_mdl_items"] = self._k_items

    # ---------- single-rule score ---------------------------------------
    def score(self, rule: Rule) -> float:
        n = self._n_total
        nXY = rule["measures"]["support"]  # |X∪Y|
        nX = rule["measures"]["supportX"]  # |X|
        nY = rule["measures"]["supportY"]  # |Y|

        if nX == 0 or nY == 0:
            return -np.inf

        H_post = self._binary_entropy(nXY / nX) * nX
        H_prior = self._binary_entropy(nY / n) * nX
        data_bits = H_post - H_prior

        k_items = len(rule["antecedent"]) + len(rule["consequent"])
        model_bits = self.c0 + self.c_item * k_items
        return data_bits - model_bits

    # ---------- fast vectorised variant ---------------------------------
    def score_dataset(self, dataset: Dataset) -> np.ndarray:
        self._ensure_dataset()

        n = self._n_total
        nXY = dataset.df["support"].to_numpy(dtype=np.int32, copy=False)
        nX = dataset.df["supportX"].to_numpy(dtype=np.int32, copy=False)
        nY = self._supportY  # cached

        H_post = self._binary_entropy(nXY / nX) * nX
        H_prior = self._binary_entropy(nY / n) * nX
        data_bits = H_post - H_prior

        if "_mdl_items" in dataset.df.columns:
            k_items = dataset.df["_mdl_items"].to_numpy(dtype=np.int16, copy=False)
        elif dataset.item_rule_map is not None:
            # one vectorised row-sum → int16 is enough for item counts
            k_items = dataset.item_rule_map.sum(axis=1).astype(np.int16)
            # cache for next call
            dataset.df["_mdl_items"] = k_items
        else:
            # last-resort: parse antecedent/consequent strings (slow)
            k_items = (
                dataset.df["antecedent"].str.count(",").add(1, fill_value=0)
                + dataset.df["consequent"].str.count(",").add(1, fill_value=0)
            ).astype(np.int16)
            dataset.df["_mdl_items"] = k_items

        k_items = dataset.df["_mdl_items"].to_numpy(dtype=np.int16, copy=False)
        model_bits = self.c0 + self.c_item * k_items

        return data_bits - model_bits


# --------------------------------------------------------------------------- #
# 5. Noisy oracle wrapper
# --------------------------------------------------------------------------- #
class NoisyOracle(Oracle):
    def __init__(self, base_oracle: Oracle, noise_rate: float = 0.1):
        super().__init__(f"Noisy({base_oracle.name})")
        self.base_oracle = base_oracle
        self.noise_rate = noise_rate

    def score(self, rule):
        if random.random() < self.noise_rate:
            return 1 - self.base_oracle.score(rule)
        return self.base_oracle.score(rule)

    def score_dataset(self, dataset) -> np.ndarray:
        scores = self.base_oracle.score_dataset(dataset)
        mask = np.random.rand(len(scores)) < self.noise_rate
        scores[mask] = 1 - scores[mask]
        return scores


# --------------------------------------------------------------------------- #
# 6. Biased oracle wrapper
# --------------------------------------------------------------------------- #
class BiasedOracle(Oracle):
    def __init__(self, base_oracle: Oracle, bias_fn: Callable):
        super().__init__(f"Biased({base_oracle.name})")
        self.base_oracle = base_oracle
        self.bias_fn = bias_fn

    def score(self, rule):
        return self.bias_fn(rule, self.base_oracle.score(rule))

    def score_dataset(self, dataset) -> np.ndarray:
        base_scores = self.base_oracle.score_dataset(dataset)
        try:  # vectorised bias
            biased = self.bias_fn(dataset, base_scores)
            if isinstance(biased, np.ndarray):
                return biased.astype(float)
        except Exception:
            pass
        return np.vectorize(lambda idx: self.bias_fn(dataset[idx], base_scores[idx]))(
            np.arange(len(dataset))
        )
