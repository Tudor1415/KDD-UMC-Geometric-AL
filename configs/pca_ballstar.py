"""Default configuration for PCA / Ball* ball-tree builder."""

DEFAULT = dict(
    leaf_size=32,
    meb="ritter",
    max_children=2,
    pca_method="power",
    pca_iters=10,
    balance="median",
    balance_max_refine_steps=3,
    random_state=0,
)