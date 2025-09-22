import numpy as np
import sys
from itertools import count
import heapq
from typing import Any, Dict, Union
from scipy.spatial.distance import pdist

def _farthest_point_idx(P, ref):
    # return index of point in P farthest from ref (both 1D arrays)
    d2 = np.sum((P - ref) ** 2, axis=1)
    return int(np.argmax(d2))

def _dedup_points(P, eps=1e-12):
    if P.shape[0] <= 1:
        return P
    # round to grid to kill near-duplicates; preserves dtype=float
    scale = 1.0 / max(eps, 1e-12)
    Q = np.unique((P * scale).round().astype(np.int64), axis=0)
    return (Q.astype(np.float64) / scale).astype(np.float64, copy=False)

def minimum_enclosing_ball(points, eps=1e-12, max_passes=3):
    """
    Iterative minimum enclosing ball (Ritter 1990) with tightening passes.
    Returns (center, radius).
    No recursion; robust on large n and d.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    n, d = P.shape

    if n == 0:
        return np.zeros(d, dtype=np.float64), 0.0
    if n == 1:
        return P[0].copy(), 0.0

    # Deduplicate near-identical points to avoid numeric churn
    P = _dedup_points(P, eps)
    n = P.shape[0]
    if n == 1:
        return P[0].copy(), 0.0

    # --- Ritter initialisation ---
    # 1) pick an arbitrary point a (use first)
    a = P[0]
    # 2) b = farthest from a
    ib = _farthest_point_idx(P, a)
    b = P[ib]
    # 3) c = farthest from b
    ic = _farthest_point_idx(P, b)
    c = P[ic]
    # 4) initial ball: center at midpoint(b,c), radius half their distance
    center = 0.5 * (b + c)
    radius = 0.5 * np.linalg.norm(b - c)

    # --- Expansion pass: move center toward outside points ---
    # Single linear pass; then a few tightening passes
    def _expand_once(center, radius):
        R = float(radius)
        C = center.copy()
        for p in P:
            diff = p - C
            dist = np.linalg.norm(diff)
            if dist > R + eps:
                # move center toward p just enough to include it
                new_R = 0.5 * (R + dist)
                # avoid division by zero
                if dist > 0:
                    C += (1.0 - R / dist) * 0.5 * diff
                R = new_R
        return C, R

    center, radius = _expand_once(center, radius)
    for _ in range(max_passes):
        center, radius = _expand_once(center, radius)

    # Final tiny safety inflation to cover eps rounding
    radius = float(max(0.0, radius))
    return center.astype(np.float64, copy=False), radius



rng = np.random.default_rng(20250921)

# ---------- Helpers ----------

def pairwise_distances(X):
    G = X @ X.T
    sq = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    np.maximum(sq, 0, out=sq)
    return np.sqrt(sq, dtype=float)

# ---------- Ball tree ----------

class BallNode:
    __slots__ = ("center","radius","indices","children","level","center_idx")
    def __init__(self, center, radius, indices, level, center_idx=None):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.indices = np.asarray(indices, dtype=int)
        self.children = []
        self.level = level
        self.center_idx = None if center_idx is None else int(center_idx)

def build_children_variable_radii(X, parent: BallNode, k=5, P=50, radius_divisor=2.0, eps=1e-12):
    """Greedy: up to k disjoint children inside parent.
       Each child radius <= R/radius_divisor; each child must cover at least P points.
       Centers restricted to data points within parent.
    """
    C, R = parent.center, parent.radius
    pts_idx = parent.indices
    if pts_idx.size == 0 or R <= 0:
        return []
    Xp = X[pts_idx]
    Dp = pairwise_distances(Xp)

    # inclusion cap + R/2 cap
    dist_to_C = np.linalg.norm(Xp - C[None, :], axis=1)
    rmax0 = np.minimum(R/2.0, R - dist_to_C)
    rmax0 = np.maximum(rmax0, 0.0)

    chosen = []            # (local_center_idx, radius, covered_local_indices)
    chosen_centers = []
    chosen_radii = []

    # Greedy loop: pick balls with >= P points, pairwise disjoint, max gain-first
    while len(chosen) < k:
        best_li, best_r, best_cover, best_gain = None, 0.0, None, -1

        for li in range(len(pts_idx)):
            # feasible radius bound under current chosen children
            rmax = rmax0[li]
            if rmax <= 0:
                continue
            for cj, rj in zip(chosen_centers, chosen_radii):
                rmax = min(rmax, Dp[li, cj] - rj)  # disjointness
                if rmax <= 0:
                    break
            if rmax <= 0:
                continue

            cover = np.where(Dp[li] <= rmax + eps)[0]
            gain  = cover.size

            # require at least P points to become a child
            if gain < P:
                continue

            if gain > best_gain or (gain == best_gain and rmax > best_r):
                best_li, best_r, best_cover, best_gain = li, rmax, cover, gain

        if best_li is None:
            break  # no more feasible children with >= P points

        # commit this child
        chosen.append((best_li, best_r, best_cover))
        chosen_centers.append(best_li)
        chosen_radii.append(best_r)

    # materialize children (could be fewer than k, possibly zero)
    children = []
    for (li, rr, cover_loc) in chosen:
        child_center = Xp[li]
        child_inds   = pts_idx[cover_loc]
        child_idx = int(pts_idx[li])
        children.append(BallNode(center=child_center, radius=rr, indices=child_inds, level=parent.level+1, center_idx=child_idx))
    return children

def build_ball_tree(X, k=5, P=10, radius_divisor=2.0, *, return_levels: bool = False):
    """Build ball tree with optional radius divisor (child radius <= R/radius_divisor).

    Parameters
    ----------
    X : ndarray
        Input feature matrix.
    k : int
        Maximum number of children per internal node.
    P : int
        Maximum leaf size.
    radius_divisor : float
        Upper bound factor for child radii relative to parent radius.
    return_levels : bool, keyword-only
        When True, also return a dictionary mapping tree level to the nodes at
        that depth (useful for visualisation or diagnostics).
    """
    C0, R0 = minimum_enclosing_ball(X)
    root = BallNode(center=C0, radius=R0, indices=np.arange(X.shape[0]), level=0, center_idx=None)
    queue = [root]
    by_level = {0: [root]}

    while queue:
        node = queue.pop(0)
        if node.indices.size <= P:
            continue  # leaf

        children = build_children_variable_radii(X, node, k=k, P=P, radius_divisor=radius_divisor)

        # expand only if we have a real split
        if len(children) >= 2:
            node.children = children
            for ch in children:
                queue.append(ch)
                by_level.setdefault(ch.level, []).append(ch)
        else:
            node.children = []  # dead-end; not expanded

    return (root, by_level) if return_levels else root

# ---------- Gather kept points and compute kept ratio ----------

def collect_kept_indices(root: BallNode, P=10):
    kept = set()
    stack = [root]
    while stack:
        nd = stack.pop()
        cidx = getattr(nd, "center_idx", None)
        if cidx is not None:
            kept.add(int(cidx))
        if len(nd.children) < 2:
            kept.update(nd.indices.tolist())
        stack.extend(nd.children)
    return np.array(sorted(kept), dtype=int)

# ------------------------- Geometry helpers -------------------------

def _dist_center_to_pair_hyperplane(wc: np.ndarray, f1: np.ndarray, f2: np.ndarray, eps=1e-12) -> float:
    """
    Distance from model-center wc to the hyperplane with normal q = f1 - f2:
        d = |<wc, q>| / ||q||
    """
    q = f1 - f2
    nq = np.linalg.norm(q)
    if nq <= eps:
        return 0.0  # same features -> zero distance
    return abs(float(np.dot(wc, q))) / nq

def _bounds_ball_pair(a, b, wc: np.ndarray, eps=1e-12):
    """
    Tight lower/upper bounds on d_wc(f1, f2) for all f1 ∈ Ba, f2 ∈ Bb.

    Let rho = ra + rb, d = c2 - c1, gamma = ||wc||, delta = <d, wc>.
    Then:
        LB = 0                       if |delta| ≤ rho * gamma
           = gamma * |delta - rho*gamma| / || d*gamma - rho*wc ||   otherwise
        UB = gamma * |delta - rho*gamma| / || d*gamma - rho*wc ||   if delta ≤ 0
           = gamma * |delta + rho*gamma| / || d*gamma + rho*wc ||   if delta ≥ 0
    """
    c1, r1 = a.center, float(a.radius)
    c2, r2 = b.center, float(b.radius)
    rho = r1 + r2
    d = c2 - c1

    gamma = float(np.linalg.norm(wc))
    if gamma <= eps:
        return 0.0, 0.0  # degenerate direction => all distances 0

    delta = float(np.dot(d, wc))

    # Lower bound
    if abs(delta) <= rho * gamma:
        LB = 0.0
    else:
        num = gamma * abs(delta - rho * gamma)
        denom = float(np.linalg.norm(d * gamma - rho * wc))
        LB = num / max(denom, eps)

    # Upper bound
    if delta <= 0:
        num = gamma * abs(delta - rho * gamma)
        denom = float(np.linalg.norm(d * gamma - rho * wc))
    else:
        num = gamma * abs(delta + rho * gamma)
        denom = float(np.linalg.norm(d * gamma + rho * wc))
    UB = num / max(denom, eps)

    return LB, UB

def _objective_value(p: np.ndarray, q: np.ndarray, wc: np.ndarray, eps=1e-12) -> float:
    diff = p - q
    denom = float(np.linalg.norm(diff))
    if denom <= eps:
        return 0.0
    num = abs(float(np.dot(diff, wc)))
    return num / max(denom, eps)

def _dominates(a, b, eps=1e-12) -> bool:
    """
    Sufficient, axis-aligned dominance test:
    If every point of ball A component-wise ≥ every point of ball B, or vice versa,
    then comparing A vs B is uninformative for a monotone aggregator — skip.
    """
    amin = a.center - a.radius
    amax = a.center + a.radius
    bmin = b.center - b.radius
    bmax = b.center + b.radius
    a_dom_b = np.all(amin >= bmax - eps)
    b_dom_a = np.all(bmin >= amax - eps)
    return bool(a_dom_b or b_dom_a)

def _exact_leaf_eval(a, b, wc: np.ndarray, X: np.ndarray, eps=1e-12):
    """
    O(|A|*|B|) exact distance on leaves; returns ((i_idx, j_idx), best_distance, eval_count).
    """
    Ai = a.indices
    Bi = b.indices
    if Ai.size == 0 or Bi.size == 0:
        return None, np.inf, 0

    XA = X[Ai]              # (m, d)
    XB = X[Bi]              # (n, d)
    diff = XA[:, None, :] - XB[None, :, :]         # (m, n, d)
    num = np.abs(np.tensordot(diff, wc, axes=(2, 0)))  # (m, n)
    denom = np.linalg.norm(diff, axis=2)
    denom = np.maximum(denom, eps)
    dist = num / denom

    # argmin
    m_idx, n_idx = np.unravel_index(np.argmin(dist), dist.shape)
    evals = int(Ai.size) * int(Bi.size)
    return (int(Ai[m_idx]), int(Bi[n_idx])), float(dist[m_idx, n_idx]), evals

def _exact_leaf_self_eval(node, wc: np.ndarray, X: np.ndarray, kept_mask: np.ndarray, eps=1e-12):
    """
    Exact objective over all point pairs within a single leaf.
    """
    idx = [int(i) for i in node.indices if kept_mask[int(i)]]
    m = len(idx)
    if m < 2:
        return None, np.inf, 0

    best_pair = None
    best_dist = float("inf")
    evals = 0
    for ii in range(m - 1):
        pi = X[idx[ii]]
        for jj in range(ii + 1, m):
            pj = X[idx[jj]]
            evals += 1
            dist = _objective_value(pi, pj, wc, eps=eps)
            if dist < best_dist:
                best_dist = dist
                best_pair = (idx[ii], idx[jj])
    return best_pair, best_dist, evals

def _collect_nodes(root):
    """Flatten tree nodes."""
    out, stack = [], [root]
    while stack:
        nd = stack.pop()
        out.append(nd)
        stack.extend(nd.children)
    return out

def _precompute_diversity(root, Q: np.ndarray):
    """
    Novelty score per node: div[node_id] = min_q ||center(node) - q||.
    Used only for visit order; independent of pruning logic.
    """
    nodes = _collect_nodes(root)
    div = {}
    if Q is None or len(Q) == 0:
        for nd in nodes:
            div[id(nd)] = 0.0
        return div

    Q = np.asarray(Q, dtype=float)
    for nd in nodes:
        div[id(nd)] = float(np.min(np.linalg.norm(Q - nd.center, axis=1)))
    return div

def _is_leaf(node, P: int) -> bool:
    """Leaf iff the node does not split (len(children) < 2)."""
    return len(node.children) < 2

# ------------------------------ Search --------------------------------

def search_pair(
    root,                      # BallNode
    X: np.ndarray,             # (n, d) feature matrix used to build the tree
    wc: np.ndarray,            # model-space center vector (same dim as X columns here)
    tau: float,                # acceptance threshold (e.g., r_max or r_max/2)
    P: int = 50,               # leaf size threshold (must match build_ball_tree)
    *,
    Q: np.ndarray = None,      # optional: visit diverse pairs first (set of query vectors)
    split_policy: str = "one", # {"one","both"}: expand one side only vs both sides
    dominance_prune: bool = True,
    return_stats: bool = False,
    eps: float = 1e-12,
    audit: Union[bool, Dict[str, Any]] = False,
):
    """
    Branch-and-Bound query search on a k-ary, disjoint ball-tree.

    Returns:
      - If return_stats=False (default):           (i_idx, j_idx, best_distance)
      - If return_stats=True:                      (i_idx, j_idx, best_distance, stats)

    'stats' fields (counts are over kept points only = node centers plus points covered by leaves):
      - total_point_pairs
      - pruned_lb_point_pairs
      - pruned_dom_point_pairs
      - pruned_point_pairs          = pruned_lb + pruned_dom
      - explored_point_pairs        = pairs evaluated exactly (leaf–leaf + leaf self + credited center-only)
      - unexplored_point_pairs      = 0 (by construction)
      - objective_evals             = #objective evaluations (centers + leaves + center-only pass)
      - best_origin                 = 'center', 'leaf', 'bound', or None
      - best_distance               = best objective value found (None if no candidate)
      - best_pair                   = pair yielding best_distance (or None)
    """
    wc = np.asarray(wc, dtype=float)
    n = X.shape[0]

    # ---- universe of points for counting: KEPT points only ----
    kept_idx = collect_kept_indices(root, P=P)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept_idx] = True
    K = kept_idx.size
    total_point_pairs = int(K * (K - 1) // 2)

    nodes = _collect_nodes(root)

    # Precompute per-node kept counts so we can count pairs fast at node level
    def _precompute_kept_counts(nodes, kept_mask):
        kc = {}
        for nd in nodes:
            count = int(np.count_nonzero(kept_mask[nd.indices]))
            cidx = getattr(nd, "center_idx", None)
            if cidx is not None and 0 <= cidx < kept_mask.size and not kept_mask[cidx]:
                count += 1
            kc[id(nd)] = count
        return kc

    kept_counts = _precompute_kept_counts(nodes, kept_mask)
    leaf_nodes = [nd for nd in nodes if _is_leaf(nd, P)]
    leaf_point_mask = np.zeros(n, dtype=bool)
    for leaf in leaf_nodes:
        leaf_point_mask[leaf.indices] = True
    leaf_point_mask &= kept_mask
    center_only_idx = np.array(sorted(np.where(kept_mask & ~leaf_point_mask)[0]), dtype=int)

    # Diversity scores (optional, visit order only)
    div = _precompute_diversity(root, Q)
    use_diversity = Q is not None and len(Q) > 0

    # Stats
    stats = dict(
        total_point_pairs=total_point_pairs,
        pruned_lb_point_pairs=0,
        pruned_dom_point_pairs=0,
        pruned_point_pairs=0,
        explored_point_pairs=0,
        unexplored_point_pairs=None,  # filled at the end
        objective_evals=0,
        best_origin=None,
        best_distance=None,
        best_pair=None,
        global_lb=None,
        lb_gap=None,
        lb_tightness=None,
    )

    H = []                          # min-heap of ((key...), lb, a, b, ub, _tie)
    uid_pairs = set()               # visited node-pairs (canonicalized)
    self_done = set()               # nodes whose intra-sibling pairs were scheduled
    best_pair = None
    d_best = float("inf")
    tau_cur = float(tau)
    tie = count()
    best_origin = None
    best_lb_seen = float("inf")

    def kept_pair_mass(a, b) -> int:
        """#(kept points in A) * #(kept points in B)."""
        return kept_counts.get(id(a), 0) * kept_counts.get(id(b), 0)

    def maybe_schedule_siblings(nd):
        """Schedule all sibling pairs under this node *on first sight*."""
        nid = id(nd)
        if len(nd.children) >= 2 and nid not in self_done:
            self_done.add(nid)
            ch = nd.children
            for i in range(len(ch)):
                for j in range(i + 1, len(ch)):
                    _ = push(ch[i], ch[j])  # may LB-prune immediately

    def push(a, b):
        """Push (a,b) with all guards. Return True if the search can stop early."""
        nonlocal tau_cur, stats, best_pair, d_best, best_origin, best_lb_seen
        a, b = (a, b) if id(a) < id(b) else (b, a)
        key = (id(a), id(b))
        if key in uid_pairs:
            return False
        uid_pairs.add(key)

        # NEW: schedule siblings for both endpoints right away
        maybe_schedule_siblings(a)
        maybe_schedule_siblings(b)

        mass = kept_pair_mass(a, b)  # how many (point,point) pairs this node-pair represents (over kept points)

        if dominance_prune and _dominates(a, b, eps=eps):
            stats["pruned_dom_point_pairs"] += mass
            return False

        lb, ub = _bounds_ball_pair(a, b, wc, eps=eps)
        if lb < best_lb_seen:
            best_lb_seen = lb

        # evaluate center-center distance once (cheap)
        center_dist = _objective_value(a.center, b.center, wc, eps=eps)
        stats["objective_evals"] += 1
        ci_a = getattr(a, "center_idx", None)
        ci_b = getattr(b, "center_idx", None)
        if ci_a is not None and ci_b is not None:
            idx_a = int(ci_a)
            idx_b = int(ci_b)
            if center_dist < d_best:
                d_best = center_dist
                tau_cur = min(tau, d_best)
                best_pair = (idx_a, idx_b)
                best_origin = "center"
            if center_dist <= tau:
                return True

        if lb > tau_cur:
            stats["pruned_lb_point_pairs"] += mass
            return False

        if ub <= tau_cur:
            # bound witness; we can accept and stop
            if a.indices.size and b.indices.size:
                pair_idx = (int(a.indices[0]), int(b.indices[0]))
                if d_best > ub:
                    d_best = ub
                    tau_cur = min(tau, d_best)
                    best_pair = pair_idx
                    best_origin = "bound"
                elif best_pair is None:
                    best_pair = pair_idx
                    if best_origin is None:
                        best_origin = "bound"
                return True
            return False

        # not pruned -> queue for best-first
        cs = _dist_center_to_pair_hyperplane(wc, a.center, b.center)
        cs_key = (-min(div.get(id(a), 0.0), div.get(id(b), 0.0)), cs) if use_diversity else (cs,)
        heapq.heappush(H, (cs_key, lb, a, b, ub, next(tie)))
        return False

    def _finalize_result(extra_explored_from_centers=0):
        stats["pruned_point_pairs"] = stats["pruned_lb_point_pairs"] + stats["pruned_dom_point_pairs"]
        # incorporate credited explored pairs from the center-only pass
        if extra_explored_from_centers:
            stats["explored_point_pairs"] += int(extra_explored_from_centers)
        stats["unexplored_point_pairs"] = stats["total_point_pairs"] - stats["pruned_point_pairs"] - stats["explored_point_pairs"]
        stats["best_origin"] = best_origin
        stats["best_distance"] = d_best if best_pair is not None else None
        stats["best_pair"] = best_pair

        candidates = []
        if best_lb_seen < float("inf"):
            candidates.append(best_lb_seen)
        if H:
            heap_lb = min(item[1] for item in H)
            if heap_lb < float("inf"):
                candidates.append(heap_lb)
        if candidates:
            global_lb = min(candidates)
            stats["global_lb"] = float(global_lb)
            if best_pair is not None and np.isfinite(d_best) and d_best > 0:
                gap = max(0.0, float(d_best - global_lb))
                stats["lb_gap"] = gap
                stats["lb_tightness"] = float(global_lb / d_best)

        if isinstance(audit, dict):
            audit.update({k: stats.get(k) for k in ("global_lb", "lb_gap", "lb_tightness")})

        if best_pair is not None and d_best <= tau:
            result = (best_pair[0], best_pair[1], d_best)
        else:
            result = (None, None, float("inf"))

        return (*result, stats) if return_stats else result

    # Seed the heap with all unordered pairs among root's children
    if len(root.children) < 2:
        return _finalize_result()

    for i in range(len(root.children)):
        for j in range(i + 1, len(root.children)):
            if push(root.children[i], root.children[j]):
                return _finalize_result()

    # Best-first search
    while H:
        (_key, lb, A, B, ub, _t) = heapq.heappop(H)

        if d_best <= tau_cur:
            break

        # bound witness found upon pop
        if ub <= tau_cur and A.indices.size and B.indices.size:
            pair_idx = (int(A.indices[0]), int(B.indices[0]))
            if d_best > ub:
                d_best = ub
                tau_cur = min(tau, d_best)
                best_pair = pair_idx
                best_origin = "bound"
            elif best_pair is None:
                best_pair = pair_idx
                if best_origin is None:
                    best_origin = "bound"
            return _finalize_result()

        # Exact evaluation at leaves: count explored point pairs
        if _is_leaf(A, P) and _is_leaf(B, P):
            mass = kept_pair_mass(A, B)
            stats["explored_point_pairs"] += mass

            pair, dist, evals = _exact_leaf_eval(A, B, wc, X, eps=eps)
            stats["objective_evals"] += evals
            if pair is not None and dist < d_best:
                best_pair, d_best = pair, dist
                best_origin = "leaf"
                tau_cur = min(tau, d_best)
            continue

        A_has = len(A.children) >= 2
        B_has = len(B.children) >= 2

        if A_has and B_has:
            if split_policy == "one":
                if A.radius >= B.radius:
                    for a in A.children:
                        if push(a, B):
                            return _finalize_result()
                else:
                    for b in B.children:
                        if push(A, b):
                            return _finalize_result()
            else:  # "both"
                for a in A.children:
                    for b in B.children:
                        if push(a, b):
                            return _finalize_result()
        elif A_has:
            for a in A.children:
                if push(a, B):
                    return _finalize_result()
        elif B_has:
            for b in B.children:
                if push(A, b):
                    return _finalize_result()

        # also schedule sibling pairs for A and B when popped
        maybe_schedule_siblings(A)
        maybe_schedule_siblings(B)

    # End-game: within-leaf exact pairs (self-pairs)
    if best_pair is None or d_best > tau:
        for leaf in leaf_nodes:
            pair, dist, evals = _exact_leaf_self_eval(leaf, wc, X, kept_mask, eps=eps)
            if evals == 0:
                continue
            stats["explored_point_pairs"] += evals
            stats["objective_evals"] += evals
            if pair is not None and dist < d_best:
                best_pair, d_best = pair, dist
                best_origin = "leaf"

    # End-game: center-only coverage
    extra_explored = 0
    if (best_pair is None or d_best > tau) and center_only_idx.size > 0:
        # Only credit as many evaluations as remain "unexplored" to avoid double counting
        remaining = stats["total_point_pairs"] - (stats["pruned_lb_point_pairs"] + stats["pruned_dom_point_pairs"] + stats["explored_point_pairs"])
        need = int(max(0, remaining))
        seen_pairs = set()
        for idx in center_only_idx:
            pi = X[int(idx)]
            for j_idx in kept_idx:
                j_idx = int(j_idx)
                if j_idx == int(idx):
                    continue
                a, b = sorted((int(idx), j_idx))
                key = (a, b)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                stats["objective_evals"] += 1
                dist = _objective_value(pi, X[j_idx], wc, eps=eps)
                if dist < d_best:
                    best_pair, d_best = key
                    best_origin = "center"
                if extra_explored < need:
                    extra_explored += 1

    return _finalize_result(extra_explored_from_centers=extra_explored)


# ---------- Run on a random 2D dataset ----------

if __name__ == "__main__":
    dim = 3
    n = 10_000
    fraction = 0.75
    radius_divisor = fraction**(1/dim)
    X = rng.random((n, dim))
    k = 10000
    P = 25

    root, _ = build_ball_tree(
        X, k=k, P=P, radius_divisor=radius_divisor, return_levels=True
    )
    kept_idx = collect_kept_indices(root, P=P)

    total_kept = kept_idx.size
    kept_ratio = total_kept / n if n > 0 else 0.0
    print(f"Kept points (including centers): {total_kept} / {n}  -> ratio = {kept_ratio:.3f}")

    wc = rng.random(dim)          # example; must match X.shape[1]
    wc /= np.sum(wc)
    tau = 5e-7                           # e.g., r_max or r_max/2
    Q = None                             # or an array of previously asked feature vectors to get diversity-first visitation

    for it in range(2):
        print(f"\nSearch iteration {it+1}:")
        i_idx, j_idx, d, st = search_pair(
            root, X, wc, tau, P=P,
            Q=Q, split_policy="one", dominance_prune=True,
            return_stats=True
        )
        if i_idx is not None and j_idx is not None:
            print("Suggested pair:", (np.round(X[i_idx], 2), np.round(X[j_idx], 2)),
                  "exact distance (if found):", f"{d:.2e}")
        else:
            print("No pair found.")

        print("\n[Stats over kept points only]")
        total_pairs = st["total_point_pairs"]
        label_w = 36
        count_w = 12
        pct_w = 8

        def pct_line(label, value):
            pct = f"{value/total_pairs:.2%}" if total_pairs > 0 else "n/a"
            print(f"{label:<{label_w}}: {value:>{count_w},}    {pct:>{pct_w}}")

        def count_line(label, value):
            print(f"{label:<{label_w}}: {int(value):>{count_w},}")

        count_line("Total point pairs", total_pairs)
        pct_line("Pruned by LB", st["pruned_lb_point_pairs"])
        pct_line("Pruned by dominance", st["pruned_dom_point_pairs"])
        pct_line("Pruned total", st["pruned_point_pairs"])
        # Label reflects that we account leaf–leaf + leaf self + credited center-only
        pct_line("Explored exactly (accounted)", st["explored_point_pairs"])
        pct_line("Objective evaluations", st["objective_evals"])
        pct_line("Unexplored (not pruned/explored)", st["unexplored_point_pairs"])
        best_dist = st.get("best_distance")
        best_dist_str = f"{best_dist:.3e}" if best_dist is not None else "n/a"
        print(f"{'Best distance':<{label_w}}: {best_dist_str}")
        origin_map = {"center": "node center", "leaf": "leaf exploration", "bound": "bound witness"}
        origin = origin_map.get(st.get("best_origin"), "unknown")
        print(f"{'Best pair source':<{label_w}}: {origin}")

        if i_idx is not None and j_idx is not None:
            if Q is None:
                Q = np.vstack([X[i_idx], X[j_idx]])
            else:
                Q = np.vstack([Q, X[i_idx], X[j_idx]])
        if Q is not None and len(Q) > 1:
            diversity = pdist(Q).mean()
            print(f"Current Q diversity (mean pairwise distance): {diversity:.4f}")


