Getting started
===============

This section walks you through installing the Geometry-Aware Learning package,
preparing datasets, and running the main experiment pipelines.  It concludes
with a short Python example that wires together the modular components.

Installation
------------

The repository follows a ``src/`` layout.  Installing it in editable mode keeps
your virtual environment in sync with local changes:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   pip install -U pip
   pip install -e .[docs]

The ``docs`` extra installs Sphinx and the Furo theme so you can rebuild this
documentation locally with ``make html`` (see the ``docs/`` folder).

Quick preprocessing pass
------------------------

Most experiments expect three artefact folders: ``datasets/`` (transactions),
``mined_rules/`` (rule CSV files) and ``matrices/`` (rule-item matrices).  Use
the preprocessing CLI to normalise metrics and rebuild matrices in bulk.  It
operates on an input *directory* and processes every ``*_mnr.csv`` it finds:

.. code-block:: bash

   python -m scripts.preprocess mined_rules \
       --datasets datasets \
       --normalize \
       --write-binary matrices

Running built-in experiments
----------------------------

The config-driven runner now lives under ``gal.experiments.run``.  Provide a
YAML file describing datasets, oracle/centre combinations, and logging flags:

.. code-block:: bash

   python -m gal.experiments.run configs/exp_demo.yaml

It returns the path to the timestamped run directory where all artefacts are
stored.  Copy ``configs/exp_demo.yaml`` and tweak the ``datasets`` section to
point at your own rule CSVs or matrices before launching a longer job.

To benchmark the ball-tree branch-and-bound search, the explicit benchmark
script still lives under ``scripts.benchmarks.ball_tree_benchmark``.  Results are
streamed to ``benchmark_outputs/ball_tree_benchmark.csv`` so you can analyse
them incrementally:

.. code-block:: bash

   python -m scripts.benchmarks.ball_tree_benchmark \
       --datasets credit magic \
       --p-values 25 50 \
       --fractions 0.3 0.5 \
       --iterations 20

Experiment RQ1 (Anytime BnB)
----------------------------

To reproduce the RQ1 anytime curves comparing kd-tree vs ball-tree BnB and a
random sampling baseline, use the dedicated runner under ``experiments/rq1``.

1) Copy and edit the sample config to point to your artefacts::

     cp experiments/rq1/config.sample.yaml my_rq1.yaml

2) Run the experiment as a module::

     python -m experiments.rq1.run my_rq1.yaml

Figures are written under ``global.output_dir`` specified in the YAML. See
:doc:`experiments/rq1` for a full configuration reference and output details.

Hands-on walkthrough (Mushroom dataset)
---------------------------------------

The repository ships with a lightweight configuration and sample artefacts so
you can run an end-to-end experiment immediately.

1. **Regenerate the sample metrics (optional).**  This refreshes the normalised
   metrics and binary matrices under ``DATA/``::

     python -m scripts.preprocess DATA/mined_rules \
         --datasets DATA/datasets \
         --normalize \
         --write-binary DATA/matrices

2. **Create a working config.**  Copy the demo YAML and adjust the ``datasets``
   block if you stored artefacts elsewhere::

     cp configs/exp_demo.yaml my_first_run.yaml

   The default file already targets the packaged Mushroom artefacts under
   ``DATA/`` and writes outputs to ``results/``.

3. **Launch the run and capture the output directory.**

   .. code-block:: bash

      RUN_DIR=$(python -m gal.experiments.run my_first_run.yaml)
      echo "Results live in: $RUN_DIR"

   Each run produces ``config.json``, ``iterations.csv``, the final version
   space, and per-iteration folders with search traces and centre snapshots.

4. **Inspect the results.**  The quick commands below confirm the run finished
   and summarise the first few iterations::

     head -n5 "$RUN_DIR/iterations.csv"
     python -m scripts.analyze_ranking "$RUN_DIR" \
         --rules DATA/mined_rules/mushroom_mnr.csv \
         --topk 10 20

Python API example
------------------

The package exposes high-level helpers if you prefer building experiments in
Python rather than using the CLIs.  The example below creates a small synthetic
problem, builds a ball tree, and runs the active learning loop for a handful of
iterations.

.. code-block:: python

   import numpy as np
   from gal.trees import build_ball_tree
   from gal.core.data import augment_with_minimums
   from gal.core.constraints import k_additive_constraints
   from gal.centers.poly_centers import chebyshev_center
   from gal.learning.learn import learn

   rng = np.random.default_rng(0)
   base_points = rng.random((1_000, 3))
   augmented = augment_with_minimums(base_points, k=2)

   tree = build_ball_tree(augmented, P=20)
   A0, b0, _ = k_additive_constraints(base_points.shape[1], k=2)

   # simple linear oracle
   q_star = rng.random(augmented.shape[1])
   def oracle(a_vec, b_vec):
       return 1 if (a_vec - b_vec) @ q_star >= 0 else -1

   center, A, b = learn(
       tree=tree,
       data=augmented,
       A0=A0,
       b0=b0,
       center_fn=chebyshev_center,
       oracle=oracle,
       n_iter=8,
   )
   print("Final centre", center)

The search routine powering ``learn`` now lives in ``gal.search``; see
:doc:`api/search` for details on swapping bounds or visit strategies.

Next steps
----------

* Explore the :doc:`api/index` section for detailed module documentation.
* Review :doc:`project_layout` for a high-level map of the repository.
* Read :doc:`learning_procedure` for a step-by-step breakdown of the active learning loop.
* Consult :doc:`logging` to understand how to configure verbosity and capture search traces.
* Check the ``benchmark_outputs/`` directory after running experiments to
  inspect generated CSV files and plots.
