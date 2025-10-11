import warnings

# Third-party packages (e.g., pgmpy, Sphinx extras) still rely on pkg_resources'
# namespace helpers, which emit noisy DeprecationWarnings under pytest. Suppress
# those globally so test output stays signal-focused.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"^pkg_resources is deprecated as an API\.",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"^Deprecated call to `pkg_resources\.declare_namespace",
)
