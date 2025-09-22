Project layout
==============

The repository follows a package-oriented structure so the source code can be
installed and imported as ``gal``.  The most important directories are listed
below.

.. code-block:: text

   repo/
   ├── src/gal/              # Installable package root
   │   ├── core/             # Dataset definitions and shared datatypes
   │   ├── trees/            # Ball-tree construction and search logic
   │   ├── centers/          # Polyhedral centre solvers
   │   ├── learning/         # Active learning loop and rule mining helpers
   │   ├── metrics/          # Ranking metrics and rule-quality measures
   │   ├── oracles/          # Oracle implementations plus priors
   │   └── utils/            # Feature augmentation and general helpers
   ├── scripts/              # Command-line utilities and experiment entry points
   │   ├── benchmarks/       # Ball-tree benchmark runner
   │   ├── plots/            # Plotting scripts for metrics and diversity
   │   └── preprocess.py     # Data normalisation and matrix export CLI
   ├── datasets/             # Transaction CSV files (input data)
   ├── mined_rules/          # Minimal non-redundant rule CSVs
   ├── matrices/             # Rule-item matrices produced by the preprocess step
   ├── benchmark_outputs/    # Streaming CSV logs and generated plots
   └── sota/                 # Third-party state-of-the-art baselines

Configuration files
-------------------

``pyproject.toml``
    Declares the package metadata, build system, and the ``docs`` optional
    dependency that pulls in Sphinx when you run ``pip install -e .[docs]``.

``README.md``
    High-level orientation, installation notes, and commands for the refactored
    CLIs.

``docs/``
    The Sphinx project you are reading right now.  Builds with
    ``sphinx-build -M html docs/source docs/_build``.

Generated artefacts
-------------------

* ``benchmark_outputs/`` receives CSV snapshots from long-running experiments.
* ``scripts/plots`` produce publication-style figures using those CSV files.
* ``docs/_build`` is ignored from version control but will appear after you
  build the documentation locally.
