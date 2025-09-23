"""Default configuration for two-pivot ball-tree builder."""

DEFAULT = dict(
    leaf_size=32,
    meb="ritter",
    max_children=2,
    degeneracy_fallback="direction_median",
)