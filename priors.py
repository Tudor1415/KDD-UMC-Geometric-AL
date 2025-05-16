from __future__ import annotations

"""
priors.py – background models for SurpriseOracle
================================================

Each class implements two methods:

    • expected(rule)            – mandatory, scalar expectation
    • expected_batch(dataset)   – optional, fast vectorised path
"""

import math
import time
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Sequence, Callable, List

# external (only for Bayesian-network prior)
try:
    from pgmpy.estimators import TreeSearch
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch, BicScore
except ImportError:  # pgmpy is optional
    HillClimbSearch = BicScore = VariableElimination = None


# --------------------------------------------------------------------------- #
#                               Type aliases                                   #
# --------------------------------------------------------------------------- #
Rule = Dict[str, object]  # antecedent, consequent, measures
ItemSet = Iterable[int]


# --------------------------------------------------------------------------- #
#                       Abstract base for all priors                          #
# --------------------------------------------------------------------------- #
class Prior(ABC):
    """Background model API."""

    # ---- mandatory ---------------------------------------------------------
    @abstractmethod
    def fit(self, rules: Sequence[Rule]) -> None: ...

    @abstractmethod
    def expected(self, rule: Rule) -> float: ...

    # ---- optional vectorised path -----------------------------------------
    def expected_batch(self, dataset) -> np.ndarray | None:  # noqa: ANN001
        return None


# --------------------------------------------------------------------------- #
#              Low-level helper: design matrices for log-linear               #
# --------------------------------------------------------------------------- #
def _rule_to_indicator(
    rule: Rule,
    item_to_col: Dict[int, int],
) -> np.ndarray:
    """
    Build a 0/1 vector x ∈ {0,1}^d  where
        x[j] = 1  ⟺  item `it` with item_to_col[it] == j
                    appears in antecedent ∪ consequent.
    Items outside *item_to_col* are silently ignored.
    """
    d = len(item_to_col)
    v = np.zeros(d, dtype=np.uint8)
    for it in rule["antecedent"] + rule["consequent"]:
        j = item_to_col.get(int(it))
        if j is not None:
            v[j] = 1
    return v


# --------------------------------------------------------------------------- #
#                       helper: robust device picker                           #
# --------------------------------------------------------------------------- #
def _pick_device(
    want: str = "auto",
    max_retry: int = 3,
    wait_seconds: int = 60,
    probe_gib: float = 0.5,
) -> torch.device:
    """
    Choose a CUDA device with the most free memory and verify it can
    actually allocate `probe_gib` GiB in one chunk.

    Parameters
    ----------
    want          : "auto" | "cpu" | "cuda:N"
    max_retry     : times to re-probe a GPU after OOM
    wait_seconds  : pause between retries (sec)
    probe_gib     : size of the test allocation (GiB)
    """

    # ---------------- internal probe ------------------------------------
    def _probe(dev: torch.device) -> bool:
        """Try to allocate `probe_gib` GiB; True if success, else False."""
        bytes_needed = int(probe_gib * 1024**3)
        try:
            # float32 → 4 bytes ⇒ elements = bytes/4
            tmp = torch.empty(bytes_needed // 4, dtype=torch.float32, device=dev)
            del tmp
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            # other CUDA errors → treat as unusable
            return False

    # ---------------- explicit request ----------------------------------
    if want != "auto":
        dev = torch.device(want)
        for _ in range(max_retry):
            if _probe(dev) or dev.type == "cpu":
                return dev
            time.sleep(wait_seconds)
        return torch.device("cpu")

    # ---------------- automatic selection -------------------------------
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # pick GPU with most free mem first
    best_gpu = sorted(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.mem_get_info(i)[0],  # free bytes
        reverse=True,
    )

    for idx in best_gpu:
        dev = torch.device(f"cuda:{idx}")
        for _ in range(max_retry):
            if _probe(dev):
                return dev
            time.sleep(wait_seconds)

    # nothing passed the probe → fall back
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
#                   A. Independent log-linear  (Xin et al.)                   #
# --------------------------------------------------------------------------- #
class IndependentLogLinear(Prior):
    def __init__(self, all_items: ItemSet):
        # store the *exact* universe of items, keep natural order
        self.all_items = np.array(sorted(set(all_items)), dtype=int)
        self.item_to_col = {it: j for j, it in enumerate(self.all_items)}

        self.d = len(self.all_items)
        self.log_p: np.ndarray | None = None
        self.log_n: float | None = None
        self.device = _pick_device("auto")

    # ---------------------------------------------------------------- fit
    def fit(self, transactions: pd.DataFrame) -> None:
        n_tx = len(transactions)
        self.log_n = math.log(n_tx + 1e-9)

        # frequency vector aligned to all_items
        p = np.full(self.d, 1e-9, dtype=float)
        counts = transactions.sum(axis=0).to_dict()  # item → count
        for it, cnt in counts.items():
            if it in self.item_to_col:
                p[self.item_to_col[it]] = max(cnt / n_tx, 1e-9)

        self.log_p = np.log(p)

    # ------------------------------------------------------------- scalar rule
    def expected(self, rule: Rule) -> float:
        if self.log_p is None or self.log_n is None:
            raise RuntimeError("Prior not fitted")
        v = _rule_to_indicator(rule, self.item_to_col)
        return float(math.exp(self.log_n + v @ self.log_p))

    # ------------------------------------------------------------- batch
    def expected_batch(self, dataset) -> np.ndarray:
        if self.log_p is None or self.log_n is None:
            raise RuntimeError("Prior not fitted")
        if dataset.item_rule_map is None or dataset.items is None:
            raise AttributeError("Dataset lacks .item_rule_map or .items")

        X = dataset.item_rule_map  # (N×d_subset)
        d_subset = X.shape[1]

        # map dataset.items (subset) → self.log_p
        log_p_subset = np.empty(d_subset, dtype=np.float32)
        for j, it in enumerate(dataset.items):
            log_p_subset[j] = (
                self.log_p[self.item_to_col[it]] if it in self.item_to_col else -20.7
            )  # log(1e-9)

        # ------- back-end selection (CuPy → Torch → NumPy) -------------
        try:
            import cupy as cp

            log_fe = self.log_n + cp.asarray(X, dtype=cp.float32) @ cp.asarray(
                log_p_subset
            )
            return cp.asnumpy(cp.exp(log_fe))
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                Xg = torch.tensor(X, dtype=torch.float32, device=self.device)
                pg = torch.tensor(log_p_subset, device=self.device)
                log_fe = self.log_n + Xg @ pg
                return torch.exp(log_fe).cpu().numpy()
        except Exception:
            pass

        # CPU fallback
        log_fe = self.log_n + X.astype(np.float32, copy=False) @ log_p_subset
        return np.exp(log_fe)


# --------------------------------------------------------------------------- #
#                     D. Bayesian network prior (IIM flavour)                 #
# --------------------------------------------------------------------------- #
class BayesianNetworkPrior:
    """
    *Sparse* Bayesian network learned with pgmpy, but evaluated on the GPU
    with PyTorch.  Assumes binary (0/1) item variables.

    Parameters
    ----------
    max_parents : int            (≤ 2 keeps CPTs tiny, default 2)
    device      : "cuda"|"cpu"
    batch_size  : int            rules per GPU chunk in expected_batch
    """

    def __init__(
        self, max_parents: int = 2, device: str = "cuda", batch_size: int = 1_000_000
    ):
        self.max_parents = max_parents
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # learned later -------------------------------------------------
        self.item_order: List[int] | None = None  # variable order
        self.parents: List[List[int]] | None = None  # indices per var
        self.cpd: List[torch.Tensor] | None = None  # CPTs (GPU tensors)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        transactions: pd.DataFrame,
        *,
        max_iter: int = 300,  # ← early exit
        epsilon: float = 1e-3,  # ← stop when improvement is tiny
        structure_sample: int | None = 10_000,
        use_chow_liu_start: bool = True,
        show_progress: bool = True,
    ):
        """
        Learn a sparse BN from binary transactions, but do it *fast*.

        Fast-path tricks
        ----------------
        1. **Row subsampling** (`structure_sample`)
        ─ use at most this many rows for the *structure* search
            (CPDs are still fitted on the full data).

        2. **Good starting point** (`use_chow_liu_start`)
        ─ start HC from a Chow-Liu tree instead of an empty graph.

        3. **Fewer iterations / earlier convergence**
        (`max_iter`, `epsilon`).

        Parameters are exposed so callers can trade speed vs. accuracy.
        """
        # 1 ────────────────────────────────────────────────────────────────
        df_full = transactions.rename(
            columns={c: f"I{k}" for k, c in enumerate(transactions.columns)}, copy=True
        )
        if structure_sample is not None and len(df_full) > structure_sample:
            df_struct = df_full.sample(structure_sample, random_state=0)
        else:
            df_struct = df_full

        # 2 ───────────────────── optional Chow-Liu starting DAG ───────────
        start_dag = None
        if use_chow_liu_start:
            chow = TreeSearch(df_struct)
            start_dag = chow.estimate()

        # 3 ────────────────────── Hill-Climb structure search ─────────────
        est = HillClimbSearch(df_struct, use_cache=True)
        dag = est.estimate(
            scoring_method=BicScore(df_struct),
            max_indegree=self.max_parents,
            start_dag=start_dag,
            max_iter=max_iter,
            epsilon=epsilon,
            show_progress=show_progress,  # keeps the console quiet
        )

        # 4 ────────────────────── parameter learning  ─────────────────────
        model = BayesianNetwork(dag.edges())
        model.fit(df_full, estimator=BayesianEstimator, prior_type="BDeu")

        # 5 ────────────────────── export to tensors  ──────────────────────
        topo = list(model.nodes())
        self.item_order = [int(v[1:]) for v in topo]

        self.parents, self.cpd = [], []
        for var in topo:
            cpd = model.get_cpds(var)
            self.parents.append([topo.index(p) for p in cpd.get_evidence()])
            self.cpd.append(
                torch.tensor(
                    cpd.values,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                )
            )

    # ------------------------------------------------------------------ scalar
    def expected(self, rule: Rule) -> float:
        if self.cpd is None:
            raise RuntimeError("Prior not fitted")

        present = set(rule["antecedent"]) | set(rule["consequent"])
        states = torch.zeros(len(self.item_order), dtype=torch.int8, device=self.device)
        for it in present:
            try:
                states[self.item_order.index(int(it))] = 1
            except ValueError:
                pass  # unseen item → ignored

        return float(self._log_prob(states.unsqueeze(0)).exp().cpu())

    # ------------------------------------------------------------------ batch
    def expected_batch(self, dataset) -> np.ndarray:  # noqa: ANN001
        if self.cpd is None:
            raise RuntimeError("Prior not fitted")
        if dataset.item_rule_map is None or dataset.items is None:
            raise AttributeError("Dataset lacks .item_rule_map / .items")

        X_src = dataset.item_rule_map  # (N × d_src) uint8
        src_items = list(map(int, dataset.items))
        # --- re-order columns to match self.item_order -----------------
        col_map = {it: j for j, it in enumerate(src_items)}
        idx = [col_map.get(it, None) for it in self.item_order]

        N, D = X_src.shape
        out = np.empty(N, dtype=np.float32)

        for offset in range(0, N, self.batch_size):
            slab = X_src[offset : offset + self.batch_size]  # view
            # gather & cast directly on GPU
            rows = torch.zeros(
                (len(slab), len(self.item_order)), dtype=torch.int8, device=self.device
            )
            # fill present items
            for col_dest, col_src in enumerate(idx):
                if col_src is not None:
                    rows[:, col_dest] = torch.tensor(
                        slab[:, col_src], dtype=torch.int8, device=self.device
                    )
            out[offset : offset + len(slab)] = self._log_prob(rows).exp().cpu().numpy()

        return out

    # ------------------------------------------------------------------ internals
    def _log_prob(self, R: torch.Tensor) -> torch.Tensor:
        """
        R : (B × d) int8 tensor of 0/1 item states (GPU)
        Returns
        -------
        logp : (B,) float32 tensor   log joint probability
        """
        B, d = R.shape
        logp = torch.zeros(B, device=self.device)

        for j, t in enumerate(self.cpd):  # variable j in topo order
            x = R[:, j].long()  # (B,) 0/1
            par_idx = self.parents[j]

            if not par_idx:  # root
                probs = t[x]  # gather row 0/1
            elif len(par_idx) == 1:
                p1 = R[:, par_idx[0]].long()
                probs = t[x, p1]
            else:  # ≤ 2 parents by design
                p1 = R[:, par_idx[0]].long()
                p2 = R[:, par_idx[1]].long()
                probs = t[x, p1, p2]

            logp += torch.log(probs + 1e-12)

        return logp


# --------------------------------------------------------------------------- #
#                   C. Chow-Liu (tree-structured) prior                       #
# --------------------------------------------------------------------------- #
class ChowLiuPrior(Prior):
    """
    Fast approximation of a Bayesian network: optimal tree structure
    learned with the Chow-Liu algorithm, CPDs fitted with Bayesian
    estimation, and stored as GPU tensors for batched scoring.

    Parameters
    ----------
    device : "cuda" | "cpu"
    """

    def __init__(self, device: str = "auto"):
        self.device = _pick_device(device)

        # learned later -------------------------------------------------
        self.item_order: list[int] | None = None  # variable order
        self.parents: list[list[int]] | None = None
        self.cpd: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, transactions: pd.DataFrame) -> None:
        """
        Learn a Chow-Liu tree from a binary transaction table *fast*.
        """
        # --- 1. rename columns to "I0", "I1", … for pgmpy -----------------
        df = transactions.rename(
            columns={c: f"I{k}" for k, c in enumerate(transactions.columns)}, copy=True
        )

        # --- 2. learn the tree structure (≃ O(d² n) but very small const) -
        from pgmpy.estimators import TreeSearch

        dag = TreeSearch(df).estimate(estimator_type="chow-liu")  # already a DAG

        # --- 3. parameter learning on the *full* data --------------------
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import BayesianEstimator

        model = BayesianNetwork(dag.edges())
        model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")

        # --- 4. export to our tensors ------------------------------------
        topo = list(model.nodes())  # topo order
        self.item_order = [int(v[1:]) for v in topo]

        self.parents, self.cpd = [], []
        for var in topo:
            cpd = model.get_cpds(var)
            self.parents.append([topo.index(p) for p in cpd.get_evidence()])
            self.cpd.append(
                torch.tensor(
                    cpd.values,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False,
                )
            )

    # ------------------------------------------------------------------ scalar
    def expected(self, rule: Rule) -> float:
        if self.cpd is None:
            raise RuntimeError("Prior not fitted")

        present = set(rule["antecedent"]) | set(rule["consequent"])
        states = torch.zeros(len(self.item_order), dtype=torch.int8, device=self.device)
        for it in present:
            try:
                states[self.item_order.index(int(it))] = 1
            except ValueError:
                pass  # unseen item → ignored

        return float(self._log_prob(states.unsqueeze(0)).exp().cpu())

    # ------------------------------------------------------------------ batch
    def expected_batch(self, dataset) -> np.ndarray:  # noqa: ANN001
        if self.cpd is None:
            raise RuntimeError("Prior not fitted")
        if dataset.item_rule_map is None or dataset.items is None:
            raise AttributeError("Dataset lacks .item_rule_map / .items")

        X_src = dataset.item_rule_map  # (N × d_src)
        src_items = list(map(int, dataset.items))

        # map source columns → our topo order
        col_map = {it: j for j, it in enumerate(src_items)}
        idx = [col_map.get(it, None) for it in self.item_order]

        N = X_src.shape[0]
        out = np.empty(N, dtype=np.float32)

        # big batches, but memory-safe
        try:
            batch = torch.zeros(
                (N, len(self.item_order)), dtype=torch.int8, device=self.device
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == "cuda":
                # current GPU is full → pick the next available device
                self.device = _pick_device("auto")
                batch = torch.zeros(
                    (N, len(self.item_order)), dtype=torch.int8, device=self.device
                )
            else:
                raise

        for col_dest, col_src in enumerate(idx):
            if col_src is not None:
                batch[:, col_dest] = torch.tensor(
                    X_src[:, col_src], dtype=torch.int8, device=self.device
                )
        out[:] = self._log_prob(batch).exp().cpu().numpy()
        return out

    # ---------------------------------------------------------------- internals
    def _log_prob(self, R: torch.Tensor) -> torch.Tensor:
        B, _ = R.shape
        logp = torch.zeros(B, device=self.device)

        for j, t in enumerate(self.cpd):
            x = R[:, j].long()  # (B,)
            par_idx = self.parents[j]

            if not par_idx:
                probs = t[x]
            else:
                p = R[:, par_idx[0]].long()  # at most one parent in a tree
                probs = t[x, p]

            logp += torch.log(probs + 1e-12)
        return logp


# --------------------------------------------------------------------------- #
#                Registry so SurpriseOracle can look them up                  #
# --------------------------------------------------------------------------- #
PRIOR_FACTORY = {
    "independent": IndependentLogLinear,
    "bayesian": BayesianNetworkPrior,
    "chowliu": ChowLiuPrior,
}
