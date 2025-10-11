"""Public surface for the :mod:`gal.learning` package.

The package bundles the high-level active-learning loop in :mod:`gal.learning.learn`
and the associated persistence helpers in :mod:`gal.learning.learn_helpers`.
Importing from here keeps consumer code concise::

    from gal.learning import learn

The detailed algorithmic walkthrough lives in the ``learning_procedure`` Sphinx
page and the inline documentation inside the modules mentioned above.
"""

from .learn import learning_loop, project_constraint  # re-export for convenience

__all__ = ["learning_loop", "project_constraint"]
