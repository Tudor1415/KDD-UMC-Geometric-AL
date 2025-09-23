"""Default configuration for bottom-up ball-tree builder."""

DEFAULT = dict(
    meb="ritter",
    max_children=2,
    merge_cost="radius",
    precluster_leaf_size=1,
)