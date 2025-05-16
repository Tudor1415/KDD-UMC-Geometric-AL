import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence


@dataclass
class BallNode:
    """A node in a 2-D Ball-tree."""

    points: np.ndarray  # 2-D points inside this node
    center: np.ndarray  # centroid of those points  (shape: (2,))
    radius: float  # max ‖x − center‖₂ over x in points
    left: Optional["BallNode"] = None  # left  child
    right: Optional["BallNode"] = None  # right child


# ----------------------------------------------------------------------
# Helper #1 – bounding ball of an arbitrary point set
# ----------------------------------------------------------------------
def _bounding_ball(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return (centroid, radius) for a given 2-D array `P` of shape (n, 2).
    """
    c = P.mean(axis=0)
    r = np.linalg.norm(P - c, axis=1).max() if len(P) else 0.0
    return c, r


# ----------------------------------------------------------------------
# Helper #2 – Algorithm 2: Split(S)
# ----------------------------------------------------------------------
def _split_seed_grow(P: np.ndarray, rng=np.random) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose two pivots (x_l, x_r) using the 'seed-and-grow farthest pair' rule:

        1.  pick a random seed v
        2.  x_l = arg max ‖x − v‖
        3.  x_r = arg max ‖x − x_l‖
    """
    n = len(P)
    if n == 0:
        raise ValueError("Cannot split an empty point set")

    v_idx = rng.randint(n)
    v = P[v_idx]
    # farthest from v   ------------------------------
    dl = np.linalg.norm(P - v, axis=1)
    x_l = P[dl.argmax()]
    # farthest from x_l ------------------------------
    dr = np.linalg.norm(P - x_l, axis=1)
    x_r = P[dr.argmax()]
    return x_l, x_r


# ----------------------------------------------------------------------
# The main recursive builder — Algorithm 1
# ----------------------------------------------------------------------
def build_balltree(
    points: np.ndarray,
    *,
    max_leaf_size: int = 10,
    rng=np.random,
) -> BallNode:
    """
    Build a Ball-tree exactly following Algorithm 1 in the paper.

    Parameters
    ----------
    points : ndarray of shape (n, 2)
        Input 2-D data set (support, confidence) or any 2-vector.
    max_leaf_size : int, default 40
        Threshold N₀ at which a subset is turned into a leaf.
    rng : numpy.random.Generator or module, default numpy.random
        Source of randomness for the seed in Split().
    """
    n = len(points)
    if n == 0:  # guard against misuse
        raise ValueError("Ball-tree cannot be built from an empty array")

    # ---- Lines 1–3: attach statistics to the *current* node ------------
    c, r = _bounding_ball(points)
    node = BallNode(points=points, center=c, radius=r)

    # ---- Line 4: stopping rule (leaf) ----------------------------------
    if n <= max_leaf_size:
        return node  # leaf node, done

    # ---- Lines 7–8: internal node – split & recurse --------------------
    x_l, x_r = _split_seed_grow(points, rng=rng)  # Algorithm 2

    # Voronoi partition: send every point to its nearer pivot
    dist_to_l = np.linalg.norm(points - x_l, axis=1)
    dist_to_r = np.linalg.norm(points - x_r, axis=1)
    mask_left = dist_to_l <= dist_to_r
    S_l, S_r = points[mask_left], points[~mask_left]

    # Edge case: extremely skewed split  →  make this node a leaf instead
    if len(S_l) == 0 or len(S_r) == 0:
        return node

    # Recursively build children (Lines 9–10)
    node.left = build_balltree(S_l, max_leaf_size=max_leaf_size, rng=rng)
    node.right = build_balltree(S_r, max_leaf_size=max_leaf_size, rng=rng)
    return node


def _prep(node1, node2, w):
    """Return d, ρ, γ with γ = ‖w‖."""
    d = node2.center - node1.center
    rho = node1.radius + node2.radius
    gamma = np.linalg.norm(w)
    return d, rho, gamma


def compute_lower_bound(node1, node2, w, eps: float = 1e-15) -> float:
    """
    Lower bound from the theorem (unchanged).
    """
    d, rho, gamma = _prep(node1, node2, w)
    if gamma < eps:
        return 0.0
    dot = np.dot(d, w)

    # Hyper-plane intersects the Minkowski ball ⇒ minimum 0
    if abs(dot) <= rho * gamma:
        return 0.0

    num = abs(dot - rho * gamma)
    denom = np.linalg.norm(d * gamma - rho * w)
    return gamma * num / max(denom, eps)


def compute_upper_bound(node1, node2, w, eps: float = 1e-15) -> float:
    """
    Correct upper bound (matches your piece-wise definition).
    """
    d, rho, gamma = _prep(node1, node2, w)
    if gamma < eps:
        return 0.0
    dot = np.dot(d, w)

    if dot <= 0:  # ⟨d,w⟩  ≤ 0
        num = abs(dot - rho * gamma)  # |⟨d,w⟩ - ργ|
        denom = np.linalg.norm(d * gamma - rho * w)
    else:  # ⟨d,w⟩  ≥ 0
        num = abs(dot + rho * gamma)  # |⟨d,w⟩ + ργ|
        denom = np.linalg.norm(d * gamma + rho * w)

    return gamma * num / max(denom, eps)


def compute_score(a, b, q):
    """
    Computes the score between two points a and b with respect to query q.
    The score is defined as the absolute value of the dot product between (a - b) and q.
    """
    if np.isclose(np.linalg.norm(a - b), 0):
        return float("inf")
    return abs(np.dot(a - b, q)) / np.linalg.norm(a - b)


def search_leaf_nodes(L: np.ndarray, R: np.ndarray, q: np.ndarray, eps=1e-10):
    """
    Return ((a*, b*), score) with
        score = |⟨a* - b*, q⟩| / ‖a* - b*‖₂
    but do *no* work for pairs whose difference-vector is (near-)zero.
    """
    # all pairwise differences  (m, n, d)
    diffs = L[:, None, :] - R[None, :, :]

    # squared Euclidean norm ‖a-b‖² _without_ the expensive sqrt
    sq_norm = np.einsum("ijk,ijk->ij", diffs, diffs)  # (m, n)

    # mask-out coincident (or almost-coincident) pairs -----------------
    mask = sq_norm > eps  # True everywhere we must evaluate

    # |⟨a-b, q⟩|
    dots = np.abs(diffs @ q)  # (m, n)

    # score²   … only where mask==True; elsewhere set to +∞
    score2 = np.full_like(sq_norm, np.inf)
    score2[mask] = (dots[mask] ** 2) / sq_norm[mask]

    # arg-min in the compressed matrix
    i, j = divmod(score2.argmin(), score2.shape[1])

    # final score (take the sqrt only once, for the winner)
    best_score = np.sqrt(score2[i, j])

    return (L[i], R[j]), best_score


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _push(heap, a, b, query, tau):
    """
    Push (a,b) into the heap with keys
        centre_score  →  lower_bound  →  upper_bound  →  FIFO
    """
    centre_score = compute_score(a.center, b.center, query)
    lower_bound = compute_lower_bound(a, b, query)
    upper_bound = compute_upper_bound(a, b, query)
    tie_breaker = np.random.random()

    # Early prune: impossible to contain a good pair
    if lower_bound > tau:
        return

    # Early accept: whole subtree is already good enough
    if upper_bound <= tau:
        # push a *sentinel* with lowest score
        heapq.heappush(heap, (0, lower_bound, upper_bound, tie_breaker, (a, b)))
        return

    # Ordinary push – still undecided
    heapq.heappush(heap, (centre_score, lower_bound, upper_bound, tie_breaker, (a, b)))


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def search_pair(
    root, query, stopping_radius: float, max_iter=100_000
) -> Tuple[Tuple, float, int, int]:
    """
    Best-first traversal over pairs of BVH nodes (balls):
    returns the pair of leaves closest to `query` under the chosen metric.

    Parameters
    ----------
    root : Node
        Root of a ball-tree / BVH whose children have `center`, `radius`,
        `left`, `right`, and (at leaves) `points` attributes.
    query : ndarray
        Point against which distances are measured by `compute_bound`
        and (in leaves) by `search_leaf_nodes`.
    stopping_radius : float
        Early exit once we find a pair with score < stopping_radius.

    Returns
    -------
    best_pair        : Tuple[Node, Node] | None
    best_score       : float
    iterations       : int   – times the main loop executed
    discarded_pairs  : int   – popped pairs that were pruned because
                               their bound ≥ current best_score
    """
    # ---------------------------------------------------------------------
    # Priority queue seeded with the root’s two children
    # ---------------------------------------------------------------------
    heap: list = []
    _push(heap, root.left, root.right, query, stopping_radius)

    best_score = float("inf")
    best_pair: Optional[Tuple] = None
    iterations = 0
    discarded = 0

    # ---------------------------------------------------------------------
    # Best-first search
    # ---------------------------------------------------------------------
    while heap:
        iterations += 1
        center_score, lower, upper, tie_breaker, (left, right) = heapq.heappop(heap)

        # Early stopping if we are already good enough
        if best_score <= stopping_radius or best_score == 0 or iterations >= max_iter:
            # print(
            #     f"Early stopping: best_score = {best_score:.2e} <= {stopping_radius:.2e}"
            # )
            break

        # -- 1. Sub-tree is *guaranteed* good enough -------------------------
        if upper <= stopping_radius:
            for a in left.points:
                for b in right.points:
                    if not np.isclose(np.linalg.norm(a - b), 0):
                        score = compute_score(a, b, query)
                        return (a, b), score, iterations, discarded
            continue

        if lower >= min(best_score, stopping_radius):
            discarded += 1
            continue

        # -----------------------------------------------------------------
        # Case 1 ─ both nodes are leaves: evaluate exactly
        # -----------------------------------------------------------------
        if not (left.left or left.right or right.left or right.right):
            candidate_pair, candidate_score = search_leaf_nodes(
                left.points, right.points, query
            )
            if candidate_score < best_score:
                best_score, best_pair = candidate_score, candidate_pair
            continue

        # -----------------------------------------------------------------
        # Case 2 ─ at least one node is internal: expand promising children
        # -----------------------------------------------------------------
        # Cross-pairs L.x – R.y
        for a in (left.left, left.right):
            if a is None:
                continue
            for b in (right.left, right.right):
                if b is None:
                    continue
                _push(heap, a, b, query, stopping_radius)

        # Intra-pairs inside each side (helps when true pair lies within one subtree)
        if left.left and left.right:
            _push(heap, left.left, left.right, query, stopping_radius)
        if right.left and right.right:
            _push(heap, right.left, right.right, query, stopping_radius)

    return best_pair, best_score, iterations, discarded
