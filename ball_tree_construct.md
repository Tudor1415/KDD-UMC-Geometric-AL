Here are the **main families of ball-tree construction algorithms**:

1. **Axis-aligned (kd-style) median splits**
   Split by the coordinate of largest spread at the median; wrap each side in its minimum enclosing ball. Fast, balanced by count, but balls can be loose.

2. **Farthest-point / max-diameter (“two-pivot”) splits**
   Pick two far-apart pivots; assign points to nearest pivot; wrap in balls. Produces tight siblings, can be unbalanced without extra controls.

3. **Data-aware balanced splits (e.g., Ball* / PCA-guided / min-volume)**
   Choose a split hyperplane using data shape (PCA axis, min sum of child radii, or an explicit cost proxy); then threshold for balance. Tighter balls *and* shorter trees; slight extra build cost.

4. **Bottom-up (agglomerative) merging**
   Start with singletons; repeatedly merge the pair whose union increases enclosing-ball cost the least. Very tight clusters; expensive to build (offline use).

5. **Incremental / insertion (incl. “cheap” insertion) & middle-out**
   Insert points into the leaf that minimally expands its ball (or via a cheaper heuristic), updating up the tree; “middle-out” variants grow a good core first. Supports dynamic data; quality depends on insertion policy.

6. **Sphere-trees (application-specific ball trees for geometry/physics)**
   Same hierarchical balls but built over mesh parts/objects; top-down or bottom-up recipes tuned for collision/N-body culling (tightness vs depth trade-offs).

   Awesome—here’s a faithful, implementation-ready version of the **k-d construction algorithm for ball trees** as introduced by **Omohundro (ICSI TR-89-063, 1989)**, followed by a plain-English, step-by-step intuition. I’ve adapted the pseudocode into clean, Python-friendly pseudocode (same logic/structure as the original Eiffel-like code with median partition and recursive build). Citations point to the exact section where each piece appears.

---

1. **Axis-aligned (kd-style) median splits**

**Key helpers you’ll need:**

* `argmax_spread(points, dims)` → returns index of the coordinate with largest spread. (Largest max−min across that dimension.)
* `nth_element_by_coord(points, coord, k)` → partitions `points` **in place** so that position `k` holds the k-th smallest value on `coord`, and everything left/right is ≤/≥ that value (like C++ `nth_element`). This is the “select\_on\_coord” operation from the paper.
* `min_enclosing_ball_union(B_left, B_right)` → the minimal ball enclosing two balls:

  * Let centers be `c1, c2` and radii `r1, r2`, distance `d = ||c2 - c1||`.
  * If one ball fully contains the other (e.g., `r1 >= r2 + d`), return the larger.
  * Else radius `R = (d + r1 + r2)/2` and center `C = c1 + ((R - r1)/d) * (c2 - c1)`.
    (This is the standard 2-ball enclosure; Omohundro’s code calls `to_bound_balls` to compute it.)

**Node structure:**

* `Node.left`, `Node.right`, `Node.ball = (center, radius)`, and for leaves, store point(s).

**Algorithm:**

```text
function BUILD_BALLTREE(points, leaf_size = 1):
    # returns root Node

    def build_range(A):      # A is a list (or view) of points in this node
        if len(A) <= leaf_size:
            node = Node()
            node.left = node.right = None
            node.ball = minimal_enclosing_ball_of_points(A)   # e.g., Welzl
            node.points = A                                    # optional: store raw points
            return node

        # 1) choose split coordinate with largest spread
        c = argmax_spread(A, dims = A[0].dim)                  # 

        # 2) split by median along coord c
        k = len(A) // 2                                        # median rank
        nth_element_by_coord(A, coord = c, k = k)              # in-place partition at median

        A_left  = A[0 : k]
        A_right = A[k : len(A)]

        # 3) recurse
        left  = build_range(A_left)                            # 
        right = build_range(A_right)

        # 4) set parent ball as min enclosing ball of children’s balls
        node = Node()
        node.left  = left
        node.right = right
        node.ball  = min_enclosing_ball_union(left.ball, right.ball)  # 
        return node

    return build_range(list(points))
```

**Notes on complexity & shape.** Median selection is linear within a node; with \~log N levels the whole build is **O(N log N)** and “balanced by count,” but it may not adapt to hierarchical structure (exactly as stated by Omohundro).

---

## Step-by-step intuition (natural language)

1. **What a ball tree stores.**
   Each node stores a **ball** (center + radius) covering everything in its subtree. Leaves hold the actual items (points). Internal balls are just for pruning during search.

2. **Top-down, offline construction.**
   You start with all points and build from the root down (you need the whole dataset available). At each step you **split the current point set into two** and recurse.

3. **Pick the split direction that matters most.**
   Measure the spread along each coordinate (max−min). Choose the coordinate with **largest spread**; that’s where the cluster is “widest,” so splitting there tends to separate points best.

4. **Median split for balance.**
   Partition **at the median** value on that coordinate. This yields two subsets with (almost) equal size → a **balanced tree depth**, which helps keep the search stack shallow. Median can be found/partitioned in linear time via a selection routine (no full sort).

5. **Recurse on each side.**
   Build a left subtree on the “≤ median” half and a right subtree on the “> median” half. This repeats until the subset is small (leaf).

6. **Tight parent bounds from children.**
   After both children return, compute the **smallest ball that encloses both child balls** (closed-form for two balls). Store that as the parent’s ball. This keeps internal bounds **as tight as possible**, which is essential for pruning later.

7. **Why this works well in practice.**

   * **Speed to build:** median partition per level → total **O(N log N)**.
   * **Balanced depth:** halves the data each level (by count).
   * **Potential downside:** because splits are **axis-aligned**, the sibling balls can be **loose** if the cloud isn’t aligned with axes. That’s why later variants (e.g., PCA/volume-minimizing) were proposed, but the kd-style builder is the canonical baseline.

---

## Where this comes from (original source)

* **Omohundro, S. M. (1989).** *Five Balltree Construction Algorithms*, ICSI Tech Report TR-89-063 — section “K-d Construction Algorithm,” including the median partition routine and the recursive builder (originally shown in Eiffel-like code). See the lines describing the max-spread dimension, median split, and the recursive `build` that sets the parent ball from its two children.&#x20;


**Farthest-point / max-diameter (“two-pivot”) splits**

**Key helpers you’ll need:**

* `centroid(points)` → arithmetic mean of the points in the current node.
* `farthest_point(points, ref)` → index of the point in `points` with maximal squared Euclidean distance to vector `ref`.
* `min_enclosing_ball(points)` → minimum-enclosing ball (MEB) for a *set* of points (Welzl exact or Ritter approx.).
* `enclose_two_balls(BL, BR)` → minimal ball enclosing **two** balls
  Let `BL=(c1,r1)`, `BR=(c2,r2)`, `d = ||c2−c1||`.
  If `r1 >= r2 + d` return `BL`; elif `r2 >= r1 + d` return `BR`; else
  `R = (d + r1 + r2)/2` and `C = c1 + ((R − r1)/d) * (c2 − c1)`.

*(This is the same 2-ball enclosure you used for kd-style; the split rule is the only change.)*

**Node structure:**

* `Node.left`, `Node.right`, `Node.ball = (center, radius)`, and, for leaves, `Node.points` (optional).

---

## Algorithm (two-pivot / farthest-point split)

```text
function BUILD_BALLTREE_FARTHEST(points, leaf_size = 1):
    # returns root Node

    def build(P):                       # P is the list/view of points in this node
        if len(P) <= leaf_size:
            node = Node()
            node.left = node.right = None
            node.ball = min_enclosing_ball(P)
            node.points = P
            return node

        # 1) pick two far-apart pivots by farthest-point heuristics
        c   = centroid(P)               # first, a coarse "center" of the cluster
        iL  = farthest_point(P, c)      # p_L: farthest from centroid
        iR  = farthest_point(P, P[iL])  # p_R: farthest from p_L (opposite extreme)

        # 2) Voronoi partition: assign to nearest pivot
        Left, Right = [], []
        for x in P:
            if dist2(x, P[iL]) <= dist2(x, P[iR]):
                Left.append(x)
            else:
                Right.append(x)

        # 3) guard for degeneracy (all points to one side)
        if len(Left) == 0 or len(Right) == 0:
            # fallback: split along the p_R - p_L direction at its median
            direction = P[iR] - P[iL]
            proj = [(dot(x, direction), x) for x in P]
            k = len(P)//2
            nth_element(proj, k, key=lambda t: t[0])   # in-place selection by projection
            Left  = [t[1] for t in proj[:k]]
            Right = [t[1] for t in proj[k:]]

        # 4) recurse
        L = build(Left)
        R = build(Right)

        # 5) parent ball is the minimal ball enclosing the two child balls
        node = Node()
        node.left  = L
        node.right = R
        node.ball  = enclose_two_balls(L.ball, R.ball)
        return node

    return build(list(points))
```

**Notes on complexity & shape.**
Each split does a few linear passes (centroid, two farthest-point scans, one assignment pass), so per-level work is O(|P|); with \~log N levels overall build is **O(N log N)**. The heuristic directly targets **max diameter**, so sibling balls are usually **tight** (great for pruning). Without the fallback, splits can be **unbalanced** on skewed data; the median-on-direction fallback keeps depth reasonable while preserving geometric separation. The “recursive farthest-point partitioning” corresponds to: pick two far-apart points, assign remaining points to the nearer pivot, then wrap each subset in its own minimal-radius ball—repeat.

---

## Step-by-step intuition (natural language)

1. **Make siblings as far apart as possible.**
   You want two children that “pull apart” the cluster. Grab a crude center (centroid), choose the point farthest from it (one extreme), then choose the point farthest from that extreme (the opposite extreme). This approximates the **max-diameter** pair of the set.

2. **Voronoi split with two seeds.**
   Assign every point to its *nearest* of the two pivots. Geometrically, you’re cutting along the **perpendicular bisector** of the segment between the pivots. This tends to produce **small-radius** children because points cluster around their nearest extreme rather than being sliced by an axis.

3. **Fix pathologies, keep balance acceptable.**
   If a side ends up empty (or tiny), split along the **pivot direction** at its median (or move a few far points) so you don’t grow a degenerate chain. This keeps the depth \~log N while retaining the geometry-aware split.

4. **Wrap children tightly; parent encloses children.**
   For each child, compute a **minimum-enclosing ball** (MEB). The parent’s ball is the **minimal enclosure of the two child balls** (closed-form), keeping bounds tight for pruning.

5. **Why it works.**
   By targeting the **diameter**, you reduce each child’s **radius** quickly across levels, which strengthens **branch-and-bound pruning** for NN and related searches. This is precisely the recipe used to get **nearly disjoint sibling balls** depicted in the “Ball-Tree Construction Algorithm” (Fig. 2, steps 1–5) where farthest-point partitioning “drives sibling balls away from one another”.

**Data-aware balanced splits (e.g., Ball* / PCA-guided / min-volume)**

**Key helpers you’ll need:**

* `principal_component(points, k=1)` → returns the first principal direction `v1` (unit vector).

  * Implementation options: full SVD on centered data; or power iteration on the covariance for speed.
* `project_on(points, direction)` → returns scalar projections `t_i = ⟨x_i, direction⟩`.
* `nth_element(vals, k)` → in-place selection so that `vals[k]` is the k-th order statistic (median if `k = n//2`).
* `min_enclosing_ball(points)` → minimum-enclosing ball (Welzl exact or Ritter approx.).
* `enclose_two_balls(BL, BR)` → minimal ball enclosing **two** balls (same 2-ball formula you already use).

**Node structure:**

* `Node.left`, `Node.right`, `Node.ball = (center, radius)`, and, for leaves, `Node.points` (optional).

---

## Algorithm (data-aware split by “direction that matters most” — PCA/Ball\* style)

```text
function BUILD_BALLTREE_PCA(points, leaf_size = 1):
    # returns root Node

    def build(P):
        if len(P) <= leaf_size:
            node = Node()
            node.left = node.right = None
            node.ball = min_enclosing_ball(P)
            node.points = P
            return node

        # 1) find the direction of maximal variance (principal component)
        v1 = principal_component(P, k=1)        # unit vector

        # 2) split by the median along that direction (balanced cut)
        t = project_on(P, v1)                   # t[i] = <P[i], v1>
        k = len(P) // 2
        nth_element(t, k)                       # in-place; t[k] is median
        tau = t[k]

        Left, Right = [], []
        for xi, ti in zip(P, t):
            if ti <= tau:
                Left.append(xi)
            else:
                Right.append(xi)

        # (optional) if severely imbalanced or radii large, refine threshold:
        #   tau* = argmin_tau [ radius(MEB({ti<=tau})) + radius(MEB({ti>tau})) ]
        # Or enforce min leaf fraction by nudging tau to get |Left|≈|Right|.

        # 3) recurse
        L = build(Left)
        R = build(Right)

        # 4) parent ball is minimal ball enclosing the two child balls
        node = Node()
        node.left  = L
        node.right = R
        node.ball  = enclose_two_balls(L.ball, R.ball)
        return node

    return build(list(points))
```

**Notes on complexity & shape.**

* If you use **power iteration** for the first PC (a few iterations), step (1) is \~`O(#points × dim)` per node; projections + partition are `O(#points)`. With \~log N levels overall: **\~O(N log N)** work in practice.
* Full SVD per node is heavier (can approach `O(N dim²)` overall), so prefer power iteration or randomized PCA.
* Compared to axis-median: this keeps the tree **balanced by count** *and* aligns the cut with the **widest spread**, usually yielding **tighter child balls** than axis-aligned splits.

---

## Step-by-step intuition (natural language)

1. **Measure where the cluster really spreads.**
   Instead of guessing via axis spreads, compute the **first principal component**: the unit direction along which your points vary most. That is the direction that “matters most” for separating the cloud.

2. **Cut orthogonal to that direction.**
   Project points onto `v1` and **split at the median projection**. This keeps subtrees **balanced by count** while respecting the cluster’s geometry (you’re cutting across its longest axis).

3. **Why this gives tighter balls.**
   Since the cut respects the intrinsic shape, each child inherits a **shorter extent** along its longest dimension, so its **minimum-enclosing ball radius shrinks faster** than with arbitrary axis cuts—leading to stronger pruning.

4. **Refine when needed.**
   If the two child balls are still large (or the split skews), you can slide the threshold slightly (still near the median) to **minimize the sum of child radii** or enforce a min fraction per side. This is the Ball\* flavor: data-aware direction + balance-aware threshold.

5. **Wrap & recurse.**
   Build children the same way; the parent stores the **minimal ball enclosing the two child balls** (closed form), keeping bounds tight for branch-and-bound queries.

---
