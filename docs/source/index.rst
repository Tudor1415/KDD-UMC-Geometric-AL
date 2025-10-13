Geometry-Aware Learning documentation
=====================================

Welcome to the documentation for the Geometry-Aware Learning (GAL) toolkit.
This project provides modular components for building and benchmarking
geometry-driven active learning workflows over rule-based datasets.

.. note::
   The codebase has been refactored around the ``gal`` Python package.  Most
   examples in this documentation assume you installed it in editable mode::

       pip install -e .[docs]

   Alternatively, export ``PYTHONPATH=src`` before running any scripts.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   learning_procedure
   logging
   ball_tree_builders
   experiments/active
   experiments/rq1

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   project_layout
