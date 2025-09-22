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
the preprocessing CLI to normalise metrics and create matrices:

.. code-block:: bash

   python -m scripts.preprocess \
       --input mined_rules/credit_mnr.csv \
       --output matrices/ \
       --normalize --write-binary matrices/

Running built-in experiments
----------------------------

The refactored entry point lives under ``scripts.main``.  It automatically
loads every dataset that has matching artefacts and executes the configured set
of centres, oracles, and metrics:

.. code-block:: bash

   python -m scripts.main --log-level INFO

To benchmark the ball-tree branch-and-bound search, use the explicit benchmark
script.  Results are streamed to ``benchmark_outputs/ball_tree_benchmark.csv``
so you can analyse them incrementally.

.. code-block:: bash

   python -m scripts.benchmarks.ball_tree_benchmark \
       --datasets credit magic \
       --p-values 25 50 \
       --fractions 0.3 0.5 \
       --iterations 20

Python API example
------------------

The package exposes high-level helpers if you prefer building experiments in
Python rather than using the CLIs.  The example below creates a small synthetic
problem, builds a ball tree, and runs the active learning loop for a handful of
iterations.

.. code-block:: python

   import numpy as np
   from gal.trees.ball_tree import build_ball_tree
   from gal.utils.helpers import augment_with_minimums, k_additive_constraints
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
       root=tree,
       A0=A0,
       b0=b0,
       center_fn=chebyshev_center,
       oracle=oracle,
       n_iter=8,
   )
   print("Final centre", center)

Next steps
----------

* Explore the :doc:`api/index` section for detailed module documentation.
* Review :doc:`project_layout` for a high-level map of the repository.
* Check the ``benchmark_outputs/`` directory after running experiments to
  inspect generated CSV files and plots.
