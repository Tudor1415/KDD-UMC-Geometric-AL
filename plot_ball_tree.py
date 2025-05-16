import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update(
    {
        "axes.titlesize": 16,  # subplot titles  (was 9)
        "axes.labelsize": 14,  # x- / y-axis labels
        "xtick.labelsize": 12,  # tick labels
        "ytick.labelsize": 12,
        "legend.fontsize": 12,  # if you ever add legends
    }
)

# --------------------------------------------------------
# 1.  load a larger random sample
# --------------------------------------------------------
df = pd.read_csv("mined_rules/ilpd_mnr.csv", usecols=["support", "confidence"])
pts = df.sample(n=3000, random_state=11).values


# --------------------------------------------------------
# 2.  minimal Ball‑tree helpers
# --------------------------------------------------------
def bounding_ball(P):
    c = P.mean(axis=0)
    r = np.linalg.norm(P - c, axis=1).max()
    return c, r


def seed_grow_split(P, rng):
    v = P[rng.integers(len(P))]
    xl = P[np.linalg.norm(P - v, axis=1).argmax()]
    xr = P[np.linalg.norm(P - xl, axis=1).argmax()]
    return xl, xr


rng = np.random.default_rng(3)

# root + split
root_c, root_r = bounding_ball(pts)
r_xl, r_xr = seed_grow_split(pts, rng)
d_l = np.linalg.norm(pts - r_xl, axis=1)
d_r = np.linalg.norm(pts - r_xr, axis=1)
left_pts, right_pts = pts[d_l <= d_r], pts[d_l > d_r]
left_c, left_r = bounding_ball(left_pts)
right_c, right_r = bounding_ball(right_pts)

# split each child
l_xl, l_xr = seed_grow_split(left_pts, rng)
ld_l = np.linalg.norm(left_pts - l_xl, axis=1)
ld_r = np.linalg.norm(left_pts - l_xr, axis=1)
g1_pts, g2_pts = left_pts[ld_l <= ld_r], left_pts[ld_l > ld_r]
g1_c, g1_r = bounding_ball(g1_pts)
g2_c, g2_r = bounding_ball(g2_pts)

r_xl2, r_xr2 = seed_grow_split(right_pts, rng)
rd_l = np.linalg.norm(right_pts - r_xl2, axis=1)
rd_r = np.linalg.norm(right_pts - r_xr2, axis=1)
g3_pts, g4_pts = right_pts[rd_l <= rd_r], right_pts[rd_l > rd_r]
g3_c, g3_r = bounding_ball(g3_pts)
g4_c, g4_r = bounding_ball(g4_pts)


# --------------------------------------------------------
# 3.  plotting helpers  – BIGGER crosses (ms=14, mew=2)
# --------------------------------------------------------
def ball(ax, c, r):
    ax.add_patch(Circle(c, r, fill=False, lw=1, color="k"))


def xmark(ax, c):
    ax.plot(*c, "kx", ms=11, mew=2)


def fmt(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    ax.set_title(title)


# --------------------------------------------------------
# 4.  compose 1×5 storyboard
# --------------------------------------------------------
fig, A = plt.subplots(1, 5, figsize=(22, 4), dpi=120)

# Step 1
A[0].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6)
ball(A[0], root_c, root_r)
xmark(A[0], root_c)
fmt(A[0], "Step 1 – root")

# Step 2
A[1].scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.6)
ball(A[1], root_c, root_r)
xmark(A[1], root_c)
mid = (r_xl + r_xr) / 2
vec = r_xr - r_xl
line = mid + np.outer([-2, 2], [-vec[1], vec[0]]) / np.linalg.norm(vec)
A[1].plot(line[:, 0], line[:, 1], "k--", lw=0.8)
fmt(A[1], "Step 2 – split root")

# Step 3
for P, c, r in ((left_pts, left_c, left_r), (right_pts, right_c, right_r)):
    A[2].scatter(P[:, 0], P[:, 1], s=4, alpha=0.7)
    ball(A[2], c, r)
    xmark(A[2], c)
fmt(A[2], "Step 3 – children")

# Step 4
A[3].scatter(left_pts[:, 0], left_pts[:, 1], s=4, alpha=0.7)
A[3].scatter(right_pts[:, 0], right_pts[:, 1], s=4, alpha=0.7)
ball(A[3], left_c, left_r)
ball(A[3], right_c, right_r)
xmark(A[3], left_c)
xmark(A[3], right_c)
mid = (l_xl + l_xr) / 2
vec = l_xr - l_xl
line = mid + np.outer([-2, 2], [-vec[1], vec[0]]) / np.linalg.norm(vec)
A[3].plot(line[:, 0], line[:, 1], "k--", lw=0.8)
mid = (r_xl2 + r_xr2) / 2
vec = r_xr2 - r_xl2
line = mid + np.outer([-2, 2], [-vec[1], vec[0]]) / np.linalg.norm(vec)
A[3].plot(line[:, 0], line[:, 1], "k--", lw=0.8)
fmt(A[3], "Step 4 – split children")

# Step 5
for P, c, r in (
    (g1_pts, g1_c, g1_r),
    (g2_pts, g2_c, g2_r),
    (g3_pts, g3_c, g3_r),
    (g4_pts, g4_c, g4_r),
):
    A[4].scatter(P[:, 0], P[:, 1], s=4, alpha=0.8)
    ball(A[4], c, r)
    xmark(A[4], c)
fmt(A[4], "Step 5 – grandchildren")

plt.tight_layout()
plt.savefig("saved_plots/ball_tree.pdf", format="pdf", bbox_inches="tight")
plt.show()
