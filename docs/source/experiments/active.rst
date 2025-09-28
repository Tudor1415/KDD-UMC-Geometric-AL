Active Learning Runner
======================

Overview
--------

The active learning runner under ``experiments.active.run`` drives an
iterative branch-and-bound search to query an oracle and shrink a version
space of weight vectors. Each iteration selects a pair of points using the
configured tree and visit strategy, asks the oracle for a sign, adds a linear
constraint, and recomputes the chosen polyhedral centre.

How to Run
----------

.. code-block:: bash

   # Single run
   python -m experiments.active.run experiments/config.sample.yaml

   # Multiple datasets and/or combinations
   python -m experiments.active.run experiments/config.sample.yaml \
       --log-level INFO --log-every 10

Configuration
-------------

The runner consumes a YAML file (see ``experiments/config.sample.yaml``) with
these sections. Keys are optional unless stated otherwise.

global
^^^^^^

- ``output_root``: Folder to write per-run directories.
- ``seed``: RNG seed used for reproducibility.
- ``max_points``: Uniform downsampling cap (``0`` disables).
- ``numexpr_max_threads``: Caps threads used by ``numexpr`` (useful to silence
  multi-thread warnings from pandas/numexpr). This sets the environment
  variable ``NUMEXPR_MAX_THREADS`` before loading datasets.

experiment
^^^^^^^^^^

- ``dataset_name``: Name used in run folders when a single dataset is provided.
- ``oracle_name``: String identifier for the oracle (see ``experiments/active/exp_oracles.py``).
  Examples: ``linear_equal``, ``linear_simplex``, ``linear_random``, ``linear_axis_0``.
- ``center_name``: Polyhedral centre to recompute each iteration. Accepted
  values (case-insensitive, synonyms allowed): ``AnalyticCenter``,
  ``ChebyshevCenter``, ``MinkowskiCenter``, ``VolumetricCenter``.
- ``active_learning_budget``: Maximum iterations (default 25).
- ``additivity_k``: Optional feature augmentation order (``1`` = no augmentation).
- ``tau_max``: Cap on the per-iteration accuracy threshold ``tau`` (default ``1e-5``).

paths
^^^^^

- ``mnr_rules``: Preferred input; CSV with the 12 rule metrics columns.
- ``matrix_npy``: Optional ``.npy`` matrix fallback.
- ``dataset_path``: Alternate single path used only for metadata export.

algorithm_parameters
^^^^^^^^^^^^^^^^^^^^

- ``leaf_size``: Target leaf size applied to both kd- and ball-trees unless
  overridden below.
- ``search_strategies``: List of strategy names (e.g., ``["lower_bound"]``).
- ``tree_build_methods``:

  - ``kdtree``: List of kd-tree methods (currently ``["kd_tree"]``).
  - ``balltree``: List of ball-tree builders to evaluate. New: supports the
    farthest two-point builder via ``["two_pivot"]``.

- Optional fine-grained toggles (if present):

  - ``preferred_tree``: ``"balltree"`` or ``"kdtree"`` to steer selection in
    the single-run path.
  - ``kd_tree.enabled`` / ``ball_tree.enabled``: Enable/disable families.
  - ``kd_tree.leaf_size`` / ``ball_tree.leaf_size``: Per-family leaf size.

logging
^^^^^^^

- ``search_events``: When true, record detailed event traces per iteration.
- ``level``: ``INFO`` or ``DEBUG``.
- ``log_every``: Emit DEBUG summary every N iterations.

oracles
^^^^^^^

- ``names``: Optional list of oracle names to sweep in the multi-run path.

datasets (alternative to ``experiment``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- A list where each item is either a string name or an object
  ``{name: ..., paths: {mnr_rules: ..., matrix_npy: ...}}``.

Outputs
-------

Each run writes a timestamped directory under ``global.output_root`` with:

- ``config.json``: Run metadata and algorithm parameters.
- ``oracle.pkl``: Pickled oracle description (name, weights, dimension).
- ``tree.h5``: kd-/ball-tree structure as HDF5 tables.
- ``query_vectors.h5``: One dataset per iteration named ``/query_<k>``.
- ``iterations.csv``: Iteration timeline (indices and timestamps).
- ``final_version_space.h5``: Final constraints datasets ``/A`` and ``/b``.
- Per-iteration subfolders ``iteration_XXX`` with:

  - ``search_trace.h5``: If ``search_events`` is on, a compact event log with
    datasets ``events/event_type``, ``node_id``, ``parent_id``, ``timestamp``,
    ``lower_bound``, ``upper_bound``.
  - ``center_model.(npy|npz)``: The current centre and auxiliary scalars
    (radius, tau) saved for reproducibility.

Stopping criteria
-----------------

The loop stops when any of the following occur:

- The version-space Chebyshev radius becomes non-positive (degenerate/empty set).
- The search returns no feasible pair (``i`` or ``j`` missing).
- The iteration budget is exhausted.

Trace analysis
--------------

Use ``scripts/analyze_traces.py`` to aggregate per-iteration statistics into a
CSV (``trace_stats.csv``) for a run directory.

.. code-block:: bash

   python -m scripts.analyze_traces path/to/run_dir --out path/to/trace_stats.csv

Key metrics:

- ``avg_branching_factor`` / ``median_branching_factor``: For each EXPANDED
  parent pair, count CREATED child pairs; report the mean/median across parents.

- ``node_lifetime_median`` / ``node_lifetime_p95``: Time a pair spends in the
  priority queue. Defined as ``time(min{PRUNED, EXPANDED}) − time(CREATED)``
  per node_id, then summarised by median and 95th percentile. This captures
  either immediate prune before expansion or time-to-pop and replaces the
  earlier PRUNED-only definition that could yield NaNs.

Notes
-----

- The builder list now includes the farthest two-point ball tree
  (``gal.trees.two_pivot``); select it via
  ``algorithm_parameters.tree_build_methods.balltree: ["two_pivot"]``.
- The runner honours ``global.numexpr_max_threads`` to cap numexpr threads,
  which helps silence warning messages on multi-core machines.

