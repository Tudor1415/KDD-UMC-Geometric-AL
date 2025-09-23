from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure the package root is on sys.path so autodoc works when building locally.
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


project = "Geometry-Aware Learning"
author = "Project Contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "announcement": "This documentation covers the refactored geometry-aware active learning codebase.",
}

todo_include_todos = True

# Mock optional heavy dependencies so autodoc succeeds without installing them.
autodoc_mock_imports = [
    "cvxpy",
    "pandas",
    "torch",
    "pgmpy",
    "sklearn",
    "adjustText",
    "fast_pareto",
    "gal.learning.merge",
]

suppress_warnings = ["autodoc.mocked_object"]

