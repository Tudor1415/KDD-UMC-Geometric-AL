"""Default configuration for middle-out ball-tree builder."""

DEFAULT = dict(
    leaf_size=32,
    meb="ritter",
    max_children=2,
    k_anchor=64,
    anchor_assign_max_iters=2,
    refine_leaves=True,
    random_state=0,
)