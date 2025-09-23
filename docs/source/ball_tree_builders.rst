Ball-tree builders
==================

This page documents the geometry-aware ball-tree builders introduced in
``src/trees`` together with the helper utilities they share.  It collects the
intuition behind each construction algorithm, summarises the most relevant
complexity bounds, and explains how the test-suite exercises the
implementation.

All builders operate on dense ``numpy.ndarray`` inputs, reuse the common
:class:`~trees.common.Node` data model, and rely on the same minimum enclosing
ball (MEB) primitives.  Configuration dictionaries are loaded from ``configs``
and can be overridden per call.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Helper utilities
----------------

Minimum enclosing ball (``utils.meb``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~utils.meb.meb` is a thin dispatcher that selects either the fast
  Ritter approximation or Welzl's exact algorithm based on ``config["meb"]``.
  Dispatching itself is :math:`O(1)` and every builder calls it exactly once
  per node.
* :func:`~utils.meb.ritter` implements the classic two-pass heuristic
  [Omohundro1989]_.  It scans the point set a constant number of times and
  therefore runs in :math:`O(m d)` time for :math:`m` points of dimension
  :math:`d`.  Builders use it for leaf nodes (when ``meb="ritter"``), for
  heuristic balance evaluation in :mod:`trees.pca_ballstar`, and during
  pre-clustering in :mod:`trees.bottom_up`.
* :func:`~utils.meb.welzl` follows Welzl's recursive incremental algorithm
  [Welzl1991]_ with an expected :math:`O(m d)` running time.  It is chosen
  whenever ``config["meb"] == "welzl"`` and guarantees exact leaf balls.
  Internally it depends on :func:`~utils.meb._ball_from_support`, which solves
  the support-system via dense linear algebra; those solves are bounded by
  :math:`O(d^3)` but the support set never exceeds :math:`d+1` points.

Geometry helpers (``utils.geometry``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~utils.geometry.enclose_two_balls` and
  :func:`~utils.geometry.enclose_many_balls` are used whenever a parent node is
  created.  The two-ball enclosure is :math:`O(d)`, while the accumulation over
  :math:`k` children is :math:`O(k d)`.  All builders therefore inherit the
  :math:`O(d)` per-node cost for updating parent radii.
* :func:`~utils.geometry.centroid` provides the mean vector in
  :mod:`trees.two_pivot`; the computation is :math:`O(m d)` for the active
  index view.
* :func:`~utils.geometry.project` realises projection onto a unit vector for
  :mod:`trees.pca_ballstar`, costing :math:`O(m d)`.
* :func:`~utils.geometry.dist2` is used indirectly inside the MEB routines and
  has constant :math:`O(d)` cost.

Partition helpers (``utils.partitions``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :func:`~utils.partitions.nth_element_inplace` and
  :func:`~utils.partitions.axis_median_split` wrap :func:`numpy.argpartition`
  to supply :math:`O(m)` median splits without allocating intermediate arrays.
  :mod:`trees.axis_median`, :mod:`trees.two_pivot` (for degeneracy handling),
  and :mod:`trees.bottom_up` all reuse them.
* :func:`~utils.partitions.direction_quantile_splits` partitions projections
  into quantile bins in :math:`O(m)` time.  It is central to the k-ary general
  cases in :mod:`trees.pca_ballstar` and :mod:`trees.axis_median` (when
  ``max_children > 2``).

Ball-tree construction methods
------------------------------

Axis-aligned median splits
~~~~~~~~~~~~~~~~~~~~~~~~~~
Module: :mod:`trees.axis_median`.

Intuition (after [Omohundro1989]_): split along the coordinate with the largest
spread so that sibling subtrees are balanced by count.  The builder evaluates
:math:`\mathrm{ptp}` to find the widest axis, partitions indices using
:func:`~utils.partitions.axis_median_split`, and wraps each child in an MEB.

Helper interplay:

* Axis selection uses numpy's ``ptp`` on the view created by the recursion.
* Leaves call :func:`~utils.meb.meb` with the configured backend.
* Internal nodes aggregate child balls with
  :func:`~utils.geometry.enclose_many_balls`.

Complexity: the recursion performs one :math:`O(m)` partition per level, so the
expected running time is :math:`O(n \log n \cdot d)` for balanced inputs with
:math:`n` points of dimension :math:`d`.

Two-pivot / max-diameter splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module: :mod:`trees.two_pivot`.

Intuition (after [Moore2000]_): choose a pair of far-apart pivots and assign
points to the nearer ball so that siblings are tight.  Degenerate cases fall
back to axis or directional medians to keep both sides non-empty.

Helper interplay:

* :func:`~utils.geometry.centroid` establishes the reference point for picking
  the farthest pivot.
* Farthest-point seeding for multi-way splits reuses vectorised distance
  computations and, when needed, :func:`~utils.partitions.axis_median_split` as
  a safety net.
* As in all other builders, child balls are combined via
  :func:`~utils.geometry.enclose_many_balls`.

Complexity: farthest-point seeding scans the current subset a constant number
of times (:math:`O(m d)`), the Voronoi assignment is :math:`O(m k d)` for
:math:`k` children, and the tree as a whole remains :math:`O(n \log n \cdot d)`
for the binary default.

PCA / Ball* splits
~~~~~~~~~~~~~~~~~~
Module: :mod:`trees.pca_ballstar`.

Intuition (after [Dolatshah2015]_): align the split with the dominant principal
component and - optionally - refine the threshold to minimise the sum of child
radii (Ball* criterion).

Helper interplay:

* ``power`` PCA uses iterative multiplication by the covariance matrix with
  RNG seeding; the fallback ``svd`` option calls :func:`numpy.linalg.svd`.
* Projections reuse :func:`~utils.geometry.project`, and quantile slicing uses
  :func:`~utils.partitions.direction_quantile_splits`.
* Balance refinement invokes :func:`~utils.meb.ritter` on candidate halves to
  evaluate the radius surrogate quickly.

Complexity: computing the first component via power iteration is
:math:`O(p \cdot m d)` for ``pca_iters = p``.  The median split and MEB calls
contribute :math:`O(m d)`.  Overall the builder is near
:math:`O(n \log n \cdot d)` while offering tighter sibling balls.

Bottom-up agglomeration
~~~~~~~~~~~~~~~~~~~~~~~
Module: :mod:`trees.bottom_up`.

Intuition (after [Omohundro1989]_): start from very small clusters and merge
the most compatible pair at each step so that node radii stay compact.  An
optional pre-clustering stage reduces the number of leaf nodes for large
inputs.

Helper interplay:

* ``precluster_leaf_size`` controls a preliminary top-down pass that reuses
  :func:`~utils.partitions.axis_median_split` and the MEB helpers.
* During agglomeration every merge computes
  :func:`~utils.geometry.enclose_many_balls` to evaluate cost proxies such as
  radius or delta-radius.

Complexity: with singleton pre-clusters the naive pairwise search is
:math:`O(n^2 d)` merges; picking the optimal partner also incurs
:math:`O(n^2)` comparisons.  The implementation is intentionally simple, with a
TODO note for NN-chain acceleration.

Middle-out anchors hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Module: :mod:`trees.middle_out`.

Intuition (after [Moore2000]_): seed the space with far-apart anchors, grow a
balanced core by merging the most compatible anchor groups, and recursively
refine oversized leaves.  This "middle-out" strategy combines tight balls with
well-balanced depth.

Helper interplay:

* Farthest-point seeding mirrors the logic in
  :mod:`trees.two_pivot`, repeatedly applying vectorised distance scans.
* Anchor assignments loop through ``numpy`` distance computations and reuse the
  MEB dispatcher for every anchor cluster.
* Agglomeration mirrors :mod:`trees.bottom_up`, relying on
  :func:`~utils.geometry.enclose_many_balls` to score candidate groups.

Complexity: building :math:`k` anchors from :math:`m` points costs
:math:`O(k m d)` per assignment round.  The hierarchical merging behaves like
:math:`O(k^2 d)` for the number of anchor nodes.  Recursive leaf refinement
only triggers when a leaf exceeds ``leaf_size``.

Legacy compatibility
~~~~~~~~~~~~~~~~~~~~
Module: :mod:`gal.trees.ball_tree`.

``gal.trees.ball_tree`` forwards to the modular builders so existing code can
keep importing the historical module.  The ``build_tree`` function mirrors the
new constructors: it accepts ``X`` and an optional configuration dictionary, and
algorithm selection happens through the keyword-only ``method`` parameter,
which defaults to ``"axis_median"``.

Internally the wrapper simply dispatches to the concrete modules documented
above (:mod:`trees.axis_median`, :mod:`trees.two_pivot`, :mod:`trees.pca_ballstar`,
:mod:`trees.bottom_up`, and :mod:`trees.middle_out`), so all invariants and
helper usage carry over unchanged.

Testing strategy
----------------

The test-suite in ``tests/`` documents the behavioural contract and is designed
to be read alongside the builders:

* ``tests/test_enclosure.py`` performs depth-first traversals for random,
  duplicated, and collinear datasets under both MEB backends.  It asserts that
  every node radius covers all of its descendant points within a :math:`10^{-9}`
  tolerance.
* ``tests/test_builders_api.py`` checks the user-facing API: each builder must
  return a populated :class:`~trees.common.BallTree`, propagate default or
  overridden leaf sizes, and record the selected MEB backend.
* ``tests/test_builders_correctness.py`` verifies structural invariants and
  the PCA balance heuristic.  It ensures that leaves respect ``leaf_size``,
  that no recursion emits empty children, and that the Ball* configuration
  chooses a split with a strictly smaller sum of child radii on an elongated
  synthetic dataset.

References
----------

.. [Omohundro1989] Omohundro, S. M. *Five Balltree Construction Algorithms*.
   ICSI Technical Report TR-89-063, 1989.
.. [Moore2000] Moore, A. W. *The Anchors Hierarchy: Using the Triangle
   Inequality to Survive High Dimensional Data*. CMU-RI-TR-00-05, 2000.
.. [Dolatshah2015] Dolatshah, M., Hadian, A., and Minaei-Bidgoli, B. *Ball*-tree:
   Efficient spatial indexing for constrained nearest-neighbor search in metric
   spaces. arXiv:1511.00628, 2015.
.. [Welzl1991] Welzl, E. *Smallest Enclosing Disks (Balls and Ellipsoids)* in
   H. Maurer (ed.), *New Results and New Trends in Computer Science*, 1991.
