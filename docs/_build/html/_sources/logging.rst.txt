Logging and instrumentation
===========================

The Geometry-Aware Learning toolkit relies on Python's built-in :mod:`logging`
package to keep long experiments observable without overwhelming the terminal.
This page explains the loggers exposed by the codebase, how to configure their
verbosity through the YAML files, and which artefacts are written to disk on
each run.

Overview
--------

Every module uses `logging.getLogger(__name__)` so that log records inherit the
fully-qualified module name, making it easy to target specific components with
filters or overrides.  The most common logger namespaces are:

``gal.experiments.runner``
    High-level orchestration of dataset loading, tree construction, and the
    active-learning loop.

``gal.learning.learn`` / ``gal.learning.learn_helpers``
    Iteration-level telemetry, including radius updates, search diagnostics,
    and failure paths.

``gal.search.*``
    Branch-and-bound engine internals, such as queue activity and bound checks.

``gal.utils.array_backend``
    Reports whether the CUDA backend is available when ``use_gpu`` is enabled.

The experiments default to the standard ``INFO`` level but expose
YAML toggles so you can selectively elevate verbosity or capture structured
event traces.

Configuration controls
----------------------

Logging is configured in :func:`gal.experiments.runner._setup_logging` through
the ``logging`` section of ``experiments/config.sample.yaml`` (or any derived
file).  The most relevant keys are:

``logging.level``
    Desired logging level (`"INFO"`, `"DEBUG"`, `"WARNING"`, ...).  Defaults to
    ``"INFO"``.  The runner calls :func:`logging.basicConfig` with
    ``force=True``, replacing any pre-existing configuration so the YAML value
    always wins.

``logging.log_every``
    When the level is ``DEBUG`` this controls how frequently the learning loop
    emits iteration summaries (iteration index, indices of the queried points,
    ambiguity score, and current radius).  A typical value is ``10`` to avoid
    noisy logs on large budgets.

``logging.search_events``
    Boolean toggle that enables per-iteration search trace capture.  When true,
    :func:`gal.learning.learn.learning_loop` asks the search engine to record
    event tuples and writes them to ``iteration_XXX/search_trace.npz``.  These
    NPZ files pair well with ``scripts/analyze_traces`` for queue-depth plots
    and bound-tightness histograms.

``logging.handlers`` (advanced)
    If you need finer control, call ``logging.config.dictConfig`` from a custom
    entry point before invoking :func:`gal.experiments.run_all`.  Because the
    built-in configuration uses ``force=True`` you can skip `_setup_logging`
    entirely and rely on your own handler tree.

Run artefacts
-------------

Each experiment directory created under ``global.output_root`` contains the
following logging-related files:

``iterations.csv``
    One row per iteration with timestamped start/end columns (UTC,
    millisecond-precision), oracle response, queried indices, and orientation
    scores.  The CSV is flushed after every iteration so you can `tail -f`
    long-running jobs.

``queries/query_###.npz``
    Difference vectors used to query the oracle.  These files complement the
    CSV by storing the high-dimensional data associated with each iteration.

``iteration_###/center_model.(npy|npz)``
    Snapshots of the centre, radius, and ``tau`` threshold after each update.
    This makes it easy to plot radius shrinkage or inspect convergence without
    re-running the experiment.

``iteration_###/search_trace.npz`` (optional)
    Present only when ``logging.search_events`` is true.  The archive contains
    arrays for ``event_type``, ``node_id``, ``parent_id``, ``lower_bound``, and
    ``upper_bound``; every entry corresponds to a search-engine transition.

Customising log output
----------------------

You can further tailor logging behaviour with standard Python logging tricks:

* **Module-specific levels:** call ``logging.getLogger("gal.learning").setLevel("DEBUG")`` to
  debug the learning loop while leaving the rest of the stack on ``INFO``.
* **File handlers:** attach a :class:`logging.FileHandler` in a custom driver to
  archive logs separately from the standard streams that the runner uses.
* **Structured logging:** drop in a JSON formatter by adding a handler
  configured via ``logging.config.dictConfig`` before any GAL module is imported.

Remember that :func:`gal.experiments.runner._setup_logging` calls
``logging.basicConfig(..., force=True)``.  If you want to take full control,
bypass the helper and initialise logging yourself before invoking
:func:`gal.experiments.runner.run_all`.

Troubleshooting
---------------

* **Missing trace files:** ensure ``logging.search_events`` is truthy in your
  YAML configuration.  Also verify that the budget is greater than zero, as
  early termination (e.g., infeasible constraints) ends the loop before any
  iteration directories are created.
* **Unexpected log level:** when running inside notebooks or other frameworks
  that configure logging, supply ``force=True`` to :func:`logging.basicConfig`
  (as the runner already does) or reset handlers manually to avoid duplicate
  formatting.
* **No timestamps in custom scripts:** reuse
  :func:`gal.experiments.runner._setup_logging` or apply the same ``format`` /
  ``datefmt`` strings so log output remains consistent across entry points.

Next steps
----------

* Review :doc:`learning_procedure` to see how iteration logs map to the inner
  workings of the active-learning loop.
* Inspect ``scripts/analyze_traces`` for an example of how search-trace NPZ
  files can be aggregated into CSV summaries and plots.
