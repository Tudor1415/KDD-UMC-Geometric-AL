Experiment RQ1: Anytime BnB
===========================

Overview
--------

This experiment compares three methods for finding the closest pair under a
Choquet-style objective over rule feature points:

- kd-tree branch-and-bound (kd-tree BnB)
- ball-tree branch-and-bound (ball-tree BnB)
- random pair sampling (baseline)

It reports anytime performance curves normalized by:

- wall-clock time (A@time), using a shared T_max across BnB methods, and
- objective-call budget (A@calls), normalized by P_max = n(n-1)/2.

Additional outputs include bound tightness histograms/KDE and simple scaling
summaries across different additivity levels and datasets.


Prerequisites
-------------

- Install the package in editable mode and optional docs extras::

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -e .[docs]

- Prepare dataset artefacts. The experiment expects rule feature points from a
  pre-mined rules CSV. If present, it preferentially loads the following
  columns from ``mined_rules/<dataset>_mnr.csv`` (or a path you specify in the
  config):

  supportY, supportZ, support, confidence, lift, cosine,
  phi, kruskal, yuleQ, added_value, certainty, revsupport

  Alternatively, you may provide a ``matrix_npy`` path to a precomputed
  ``.npy`` matrix with shape ``[n_rules, n_features]``.


How to Run
----------

1) Start from the sample config and edit dataset paths:

   - Copy ``experiments/rq1/config.sample.yaml`` to a working file, e.g.::

       cp experiments/rq1/config.sample.yaml my_rq1.yaml

   - Update ``datasets:`` entries (e.g., ``paths.mnr_rules``) to point to your
     local artefacts.

2) Launch the experiment as a Python module (recommended)::

       python -m experiments.rq1.run my_rq1.yaml

   You can also execute the script directly::

       python experiments/rq1/run.py my_rq1.yaml

3) Inspect figures and summary files written under
   ``<global.output_dir>/<DATASET_NAME>/...``.


Outputs
-------

Per dataset and additivity group, the runner exports:

- ``A_at_time.(png|pdf)`` — Anytime A@time curves with bootstrap CIs
- ``A_at_calls.(png|pdf)`` — Anytime A@calls curves with bootstrap CIs
- ``heap_at_calls.(png|pdf)`` — Max heap size vs. normalized calls (BnB only)
- ``bound_tightness.(png|pdf)`` — KDE of bound gaps collected along the trace
- ``curves_time.csv`` and ``curves_calls.csv`` when CSV export is enabled

A JSON summary per dataset (artifacts + outputs metadata) is saved next to the
figures. When multiple datasets are configured, a global scaling figure
``scaling_all.(png|pdf)`` is created at ``global.output_dir``.


Configuration Reference
-----------------------

The runner consumes a YAML file with these top-level sections. The example
``experiments/rq1/config.sample.yaml`` is a good starting point.

- ``global``:

  - ``output_dir``: Directory for results (figures, CSV/NPZ/JSON).
  - ``rng_seed_base``: Base RNG seed used for reproducible sampling.
  - ``num_runs``: Repeats per center for variance estimation.
  - ``epsilon``: Small positive constant to avoid divide-by-zero.
  - ``n_bootstrap``: Number of bootstrap resamples for confidence bands.
  - ``ci_level``: Confidence level for bootstrap intervals (e.g., 0.95).
  - ``numexpr_max_threads``: Caps threads for optional numexpr warnings.
  - ``max_points``: Uniformly subsample at most this many rules per dataset (0 disables subsampling).
  - ``parallel_tree_build``: Build kd/ball trees once per group in parallel.
  - ``parallel_centers``: Run centers in parallel using processes.
  - ``center_workers``: Max worker processes when ``parallel_centers`` is true.
  - ``tau``: Finite accuracy threshold used to normalize A@time and A@calls
    (A=1 when the best-so-far <= tau). Required and must be finite.

- ``budgets``:

  - ``time_checkpoints``: Monotone list of fractions in (0, 1], each used as
    a fraction of T_max to evaluate A@time.
  - ``calls_checkpoints``: Fractions of P_max = n(n-1)/2 to evaluate A@calls.

- ``centers``:

  - ``per_dataset``: Number of random centers to evaluate per dataset.
  - ``sampler``: Currently ``"positive_l1_normalized"`` (uniform on the simplex).

- ``additivity``:

  - ``values``: A list controlling Choquet additivity augmentation:

    - Integer ``k >= 1``: augments features via
      ``gal.utils.helpers.augment_with_minimums(X, k)``. ``k=1`` means no
      augmentation.
    - Dict ``{n: int, k: int}``: first slice the dataset to ``n`` rows, then
      apply ``k``-additivity as above. Useful to keep run-times bounded while
      sweeping dimensionality.

- ``methods``:

  - ``random_sampling``:

    - ``enabled``: Enable the baseline.
    - ``pair_sampling``: ``with_replacement`` or ``without_replacement``.
  - ``dual_kdtree_bnb`` (kd-tree):

    - ``enabled``: Enable kd-tree BnB.
    - ``leaf_size``: Target leaf size for the kd-tree builder.
    - ``split_rule``: Currently ``widest_axis_median``.
    - ``strategy``: Search visit strategy; ``lower_bound`` or ``diversity``.
  - ``balltree_bnb`` (ball-tree):

    - ``enabled``: Enable ball-tree BnB.
    - ``leaf_size``: Target leaf size for the ball-tree builder.
    - ``construction``: One of ``axis_median``, ``two_pivot``, ``pca_ballstar``, ``bottom_up``, ``middle_out``, ``disjoint_greedy``. Aliases like ``disjoint`` map to ``disjoint_greedy``.
    - ``strategy``: Search visit strategy; ``lower_bound`` or ``diversity``.

- ``datasets``: A list of dataset entries, each with:

  - ``name``: Dataset name (used for output folders).
  - ``paths``: Object with any of:

    - ``mnr_rules``: Path to mined-rules CSV (preferred input).
    - ``matrix_npy``: Path to a precomputed numpy matrix.
    - ``transactions_csv``: Optional transactions file; recorded for metadata.
  - Optional per-dataset overrides:

    - ``centers_override``: Overrides ``centers.per_dataset`` for this dataset.
    - ``additivity_override``: Overrides ``additivity.values`` for this dataset.

- ``evaluation``:

  - ``metrics``: Included for completeness; the runner currently computes anytime curves and bound tightness irrespective of names here.
  - ``plot_style``:

    - ``dpi``: Figure DPI; ``line_width``; ``time_xscale``/``calls_xscale`` (``linear`` or ``log``).
  - ``bound_tightness``:

    - ``max_samples``: Cap on the number of bound-gap samples in the KDE.
  - ``exports``:

    - ``csv``/``json``/``npz``: Toggle auxiliary exports (per-group curves CSV/JSON or compact NPZ traces).
    - ``figures``: List of formats to save (e.g., ["png", "pdf"]).


Minimal Example
---------------

This configuration runs a single center on the Mushroom dataset with the
default strategies and a modest threshold:

.. code-block:: yaml

   global:
     output_dir: ./results/rq1
     rng_seed_base: 1729
     num_runs: 1
     epsilon: 1e-12
     tau: 1e-1

   centers:
     per_dataset: 1

   additivity:
     values: [1]   # no augmentation

   methods:
     random_sampling:
       enabled: true
       pair_sampling: with_replacement
     dual_kdtree_bnb:
       enabled: true
       leaf_size: 32
       split_rule: widest_axis_median
       strategy: lower_bound
     balltree_bnb:
       enabled: true
       leaf_size: 32
       construction: disjoint_greedy
       strategy: lower_bound

   datasets:
     - name: MUSHROOM
       paths:
         mnr_rules: ./mined_rules/mushroom_mnr.csv

   evaluation:
     plot_style:
       time_xscale: log
       calls_xscale: log
     exports:
       figures: [png]

See also
--------

- The runner and helpers live under ``experiments/rq1/``.
- For background and additional notes, see ``NOTES/experiments/Q1.md``.
- Ball-tree builders and options are described in :doc:`../ball_tree_builders`.
