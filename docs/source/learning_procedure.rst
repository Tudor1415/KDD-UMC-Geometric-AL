Understanding the learning procedure
====================================

This guide expands on :mod:`gal.learning` and the active-learning runner so you can
trace every transformation that happens between a mined-rules CSV and the
final ``final_version_space.npz`` artefact.  It follows the same execution path
as :func:`gal.experiments.runner.run_all`, surfacing the key data
structures, configuration knobs, and termination criteria along the way.

.. contents::
   :local:
   :depth: 2

Workflow at a glance
--------------------

The experiment driver orchestrates the loop below.  Each component is documented in
separate sections so you can drill into the details.

1. **Dataset ingestion** (:class:`gal.core.data.Dataset`) loads the rule metrics
   into a dense matrix ``X ∈ ℝ^{n×d}`` and remembers how to convert rows back
   into rule dictionaries.  Optional artefacts (transactions, rule-item matrices)
   tag along for downstream analysis.
2. **Feature augmentation** (:func:`gal.core.data.augment_with_minimums`) adds
   all minimum combinations up to ``additivity_k`` to support Choquet-style
   capacities.
3. **Capacity space set-up** (:func:`gal.core.constraints.k_additive_constraints`,
   :class:`gal.core.space.CapacitySpace`) constructs the initial half-space system
   ``A₀ x ≤ b₀`` that defines the feasible Möbius coefficients.
4. **Tree construction** (:func:`gal.trees.build_tree`) builds a ball tree over
   the augmented rows.  The tree exposes fast bounding boxes for the search step.
5. **Search engine initialisation** (:class:`gal.search.engine.Search`) binds a
   visit strategy and bounds object that implement branch-and-bound queries.
6. **Active-learning loop** (:func:`gal.learning.learn.learning_loop`) repeats:

   a. select the current centre with ``center_fn(A, b)``, expand it through the
      capacity space, and compute the Chebyshev radius;
   b. run a dual-tree search to find the most ambiguous pair ``(a, b)`` using the
      selected strategy and ``τ`` threshold;
   c. ask the oracle for ``sign(⟨a − b, q⋆⟩)`` and project the resulting inequality
      back into the reduced (projected) space;
   d. append the projected row to the version space, stream all artefacts to disk,
      and continue unless a stopping condition fires.

Dataset ingestion and diagnostics
---------------------------------

Datasets are declared in the YAML config through ``datasets[]`` entries.  Each
entry is normalised by :func:`gal.experiments.config.dataset_entry_from_cfg` so
that consistent defaults apply across runs.

When :func:`gal.experiments.runner._load_dataset` executes it:

* reads the rules CSV (PyArrow backend when available) and keeps only numeric
  measuring columns determined by ``entry.measures`` or the global defaults;
* optionally drops duplicate rows in measure space (controlled by each dataset
  entry via ``drop_duplicate_measure_vectors``);
* converts the filtered frame into the dense matrix ``X`` that feeds the rest of
  the pipeline; and
* tracks ``rows_read`` / ``duplicates_dropped`` so logs can report how much
  filtering took place.

If you provide a transactions CSV or rule-item matrix, they are loaded but not
used directly by the learner—the artefacts are stored on the :class:`Dataset`
instance for reporting and custom oracles.

Feature augmentation
--------------------

Many experiments operate over *k*-additive capacities, so the feature matrix is
augmented with all minimum combinations of size ``2…k``.  The helper
:func:`gal.core.data.augment_with_minimums` appends one column per subset and
returns the augmented matrix ``X_aug``.  For ``additivity_k = 1`` the routine is a
no-op and ``X_aug`` equals the original metric matrix.

The augmentation order mirrors the subset enumeration used to build the version
space.  If you need the explicit subset list (e.g., to interpret Möbius weights)
call the helper with ``return_index_map=True``.

Capacity space and initial constraints
--------------------------------------

The configuration knob ``experiment.additivity_k`` feeds directly into
:func:`gal.core.constraints.k_additive_constraints`, which emits:

* ``A₀`` and ``b₀`` describing the monotone capacity polytope in projected
  coordinates, and
* ``proj_index``—a mapping from subset tuples to their position inside the
  projected vector.

Those outputs populate :class:`gal.core.space.CapacitySpace`.  The class exposes
two operations the learner uses every iteration:

``space.expand_center(center_proj)``
    Lifts a projected centre back into the full Möbius coordinate system.  The
    last subset is reconstructed so that the weights sum to one.

``space.project(constraint_full)``
    Reorders a constraint expressed in the full space and calls
    :func:`gal.learning.learn.project_constraint` to drop the last coordinate
    (again, the sum-to-one row).  This projection guarantees that every stored
    inequality matches the dimensionality of ``center_fn`` and ``A``.

Initial constraints are saved to disk through
``final_version_space.npz`` even when the loop stops immediately.  You can load
the file and feed ``A`` / ``b`` back into :func:`gal.learning.learn.learning_loop`
to resume progress.

Tree construction
-----------------

Active learning uses a dual-tree search to find ambiguous rule pairs.  The tree is
constructed once per dataset by :func:`gal.trees.build_tree`.  The default method is
``two_pivot`` but any :mod:`gal.trees` builder listed in
``AVAILABLE_METHODS`` is supported.  The runner records:

* tree family (currently hard-coded to ``balltree`` for the geometry-aware
  builders),
* constructor name (e.g., ``two_pivot``), and
* effective ``leaf_size`` after the builder processes the config.

The search engine only needs the resulting :class:`gal.trees.common.GeometricTree`
interface.  The tree stores bounding spheres for every node so bounds stay tight
throughout the search.

Search engine and visit strategies
----------------------------------

``gal.search.engine.Search`` wraps a branch-and-bound queue that accepts
pluggable **visit strategies** (:mod:`gal.search.strategies`) and
**bounding schemes** (:mod:`gal.search.bounds`).  The default configuration mirrors
the behaviour described in the paper:

* :class:`gal.search.bounds.BallTreeBounds` maintains angular lower/upper bounds
  for pairwise differences by combining child balls.
* :class:`gal.search.strategies.LowerBoundVisitStrategy` implements a best-first
  traversal prioritising the smallest absolute inner product with the current
  centre.

Key runtime parameters that travel from the YAML config into
:func:`gal.learning.learn.learning_loop` are:

``tau``
    ``min(radius × tau_multiplier, tau_cap)``.  Acts as an admissible bound on
    the ambiguity score.  When the lower bound exceeds ``τ`` the engine can stop
    early.

``align_orientation``
    Enables a second search phase that encourages queries aligned with the
    farthest feasible point.  The helper
    :func:`gal.learning.learn._farthest_point_socp` solves a small LP (via SciPy)
    to find that vector; when the solve fails or SciPy is missing, the alignment
    gracefully degrades to the standard search.

``use_gpu``
    Requests a CUDA-backed array backend.  :func:`gal.utils.get_array_backend`
    falls back to NumPy if PyTorch/CUDA are unavailable so GPU runs are optional.

The engine returns indices ``(i, j)``, the best score achieved so far, and—when
``return_stats`` is true—a dictionary with diagnostic traces (orientation score,
trace events, bound counters, etc.).

Oracle contract
---------------

Oracles implement :class:`gal.oracles.oracles.Oracle`.  The runner builds one
instance per dataset and binds it through ``oracle.compare_vectors``.  Two
contracts matter for the learning loop:

* ``oracle.set_dataset(ds)`` must run before comparisons so the implementation
  can recover rule metadata (e.g., measure names).
* ``compare_vectors(a_vec, b_vec)`` returns ``+1``, ``-1``, or ``0`` and is
  expected to be deterministic for repeatability.

The ``oracles`` package ships ready-made variants:

* :class:`~gal.oracles.oracles.ObjectiveMeasureOracle` – single metric.
* :class:`~gal.oracles.oracles.SumOracle` – sum of multiple metrics.
* :class:`~gal.oracles.oracles.MDLOracle` – minimum-description-length prior.
* :class:`~gal.oracles.oracles.SurpriseOracle` – Bayesian surprise based on
  configurable priors.

You can plug your own implementation by deriving from :class:`Oracle`; the runner
will pick it up automatically if you expose it in the config builder map.

Constraint projection and version-space updates
-----------------------------------------------

Each iteration produces a difference vector ``δ = a − b``.  After querying the
oracle, the loop forms the half-space:

.. math::

   y · δ^⊤ q ≥ 0,   where  y = \mathrm{sign}(⟨δ, q⋆⟩).

Two code paths take care of dimensionality:

1. ``space.project(constraint_full)`` reorders and drops the last Möbius
   coordinate so the inequality matches the centre’s working dimension.
2. ``project_constraint`` subtracts the redundant column and adjusts the right-hand
   side.  (The last Möbius coordinate is implicit once the others are known.)

The projected row is concatenated to the existing ``A`` / ``b`` matrices.  The
next centre computation therefore takes every past vote into account.

Stopping criteria
-----------------

The loop stops as soon as one of these conditions is met:

* ``radius`` returned by :func:`gal.centers._chebyshev_radius` is ``≤ 0`` or not
  finite; the version space collapsed to an empty set.
* The search engine fails to produce a pair (``i`` or ``j`` becomes ``None``),
  typically because every candidate violates the ``τ`` bound.
* The iteration budget ``active_learning_budget`` is exhausted.

When the loop breaks, ``iterations.csv`` is closed, ``final_version_space.npz`` is
written, and the function returns ``A`` / ``b`` for programmatic inspection.

Runtime instrumentation
-----------------------

Per iteration the learner streams artefacts to the experiment directory:

``iterations.csv``
    Structured log with columns ``iteration_id``, ``query_path``, ``oracle_response``,
    ``i``, ``j``, ``orientation_score``, ``timestamp_start``, ``timestamp_end``.
    Timestamps use UTC ISO-8601 with millisecond precision.

``queries/query_###.npz``
    Stores the difference vector ``δ`` under the ``vector`` key so you can replay
    the exact queries without recomputing trees.

``iteration_###/center_model.npy``
    Numpy dump of the expanded centre after the update.  The accompanying
    ``center_model.npz`` contains ``center``, ``radius``, and ``tau`` for quick
    tabular analysis.

``iteration_###/search_trace.npz``
    Present when ``logging.search_events`` is true.  Holds event arrays
    (``event_type``, ``node_id``, ``lower_bound``, …) suitable for plotting queue
    dynamics or bound tightness distributions.

``config.json``
    Metadata snapshot written once before the loop starts.  Records the oracle
    name, centre, search strategy, tree builder, and budget so downstream tools
    can reconstruct the run configuration.

Configuration knobs cheat sheet
-------------------------------

The most relevant YAML keys (see ``experiments/config.sample.yaml``):

``experiment.active_learning_budget``
    Maximum number of iterations :math:`T`.  Higher values explore more queries at
    the cost of potentially larger runs.

``experiment.tau_max`` and ``experiment.tau_radius_multiplier``
    Control the admissible ambiguity threshold.  ``tau_radius_multiplier`` scales
    the current Chebyshev radius; ``tau_max`` caps the value so the search does not
    become too permissive early on.

``experiment.center_name``
    Chooses the polyhedral centre rerun each iteration (analytic, Chebyshev,
    volumetric, Minkowski).  All centres share the same interface
    ``center_fn(A, b)``.

``experiment.additivity_k``
    Sets the augmentation order and therefore the dimensionality of the capacity
    space.  ``1`` keeps the native rule metrics; higher values introduce more
    constraints and typically require higher budgets.

``experiment.align_orientation``
    Enables farthest-point orientation to diversify queries.  Requires SciPy for
    the fallback LP.

``logging.search_events``
    Toggle detailed trace capture.  Keep it off for faster runs unless you need
    per-node diagnostics.

``trees.ball.method`` / ``trees.ball.config``
    Select a tree builder and pass builder-specific options.  For example,
    ``{"method": "two_pivot", "config": {"leaf_size": 64}}``.

``oracle.type`` and nested options
    Pick which oracle factory to invoke (``objective``, ``sum``, ``surprise``,
    ``mdl``).  Each sub-type exposes additional parameters documented in
    :mod:`gal.oracles.oracles`.

Extending or customising the loop
---------------------------------

The documented pieces compose cleanly:

* **Centres:** implement a callable ``center_fn(A, b)`` returning the projected
  centre and pass it through ``experiment.center_name`` or ``center_params``.
* **Search strategies:** subclass :class:`gal.search.strategies.VisitStrategy`,
  expose it through :func:`gal.search.strategies.get_strategy`, or inject it by
  creating a custom :class:`gal.search.engine.Search` in the runner before calling
  :func:`gal.learning.learn.learning_loop`.
* **Oracles:** subclass :class:`Oracle`, expose a factory in
  :func:`gal.experiments.runner._build_oracle`, and reference it in the config.
* **Outputs:** extend :mod:`gal.learning.learn_helpers` to stream additional
  artefacts (e.g., JSON summaries) – the helper centralises file naming so
  new formats follow the same conventions.

Because the loop never mutates the input dataset or tree, you can safely replay
experiments with tweaked parameters using the same base artefacts.  When in
doubt, inspect ``iterations.csv`` alongside the search traces; together they form
a lossless view of the learning trajectory.
