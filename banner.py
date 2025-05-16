"""
banner.py  •  k-additive edition
================================
Creates three publication-quality PDF figures plus a consolidated legend:

  • constraints_centers.pdf
  • inscribed_ball.pdf
  • ball_hyperplane.pdf
  • legend.pdf

Changes vs. the previous version
--------------------------------
* The initial constraints come from
      A0, b0 = k_additive_constraints(n_features=2, k_add=2)
  and every hyper-plane in (A0, b0) is drawn as a solid black line.
* For the ambiguous-pair search we use the same augmentation as `learn.py`:
      points = augment_with_minimums(raw_points, add_k)
* All visuals, markers, and the horizontal legend remain exactly as specified
  earlier (orange ♦ Minkowski, red ● Chebyshev, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency: adjustText (labels) ----------------------------------
# ---------------------------------------------------------------------------
try:
    from adjustText import adjust_text  # type: ignore
except ImportError:  # pragma: no cover

    def adjust_text(*_args, **_kwargs):  # type: ignore
        """Fallback noop when *adjustText* is unavailable."""
        pass


# ---------------------------------------------------------------------------
# Project-internal helpers ---------------------------------------------------
# ---------------------------------------------------------------------------
try:
    from helpers import augment_with_minimums, k_additive_constraints  # type: ignore
except ImportError as err:
    sys.exit(
        "helpers.{augment_with_minimums,k_additive_constraints} not found: " + str(err)
    )

try:
    from poly_centers import chebyshev_center, minkowski_center  # type: ignore
except ImportError as err:
    sys.exit("poly_centers with the required functions not found: " + str(err))

try:
    from ball_tree import build_balltree, search_pair
except ImportError as err:
    sys.exit("ball_tree module not found: " + str(err))

from learn import _chebyshev_radius  # type: ignore

# ---------------------------------------------------------------------------
Array = np.ndarray
OUTPUT_DIR = Path(__file__).resolve().parent


def _set_ticks(ax):
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np  # ← already present later


# ------------------------------------------------------------------
# Helper: grow a single rcParam by `scale` only if it is numeric
# ------------------------------------------------------------------
def grow(key: str, scale: float, fallback: float):
    """Multiply rcParams[key] by scale; if it is a string, convert it."""
    val = mpl.rcParams[key]
    if isinstance(val, (int, float)):
        mpl.rcParams[key] = val * scale
    else:  # string like 'small', 'large', …
        mpl.rcParams[key] = fallback * scale


scale = 1.5
base = float(mpl.rcParams["font.size"])  # usually 10.0

# Increase the global default first
mpl.rcParams["font.size"] = base * scale

# Now grow the size-related children safely
for key in [
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
]:
    grow(key, scale, base)

PHI_SIZE = 20


# --- oracle used everywhere -------------------------------------------------
def sum_oracle(a: Array, b: Array) -> int:
    """Return +1 iff sum(a) > sum(b), else −1."""
    return 1 if a.sum() > b.sum() else -1


# ---------------------------------------------------------------------------
# Geometry helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------
def plot_constraints(ax: plt.Axes, A: Array, b: Array, xlim, ylim, lw: float = 1.5):
    """Draw every 2-D hyper-plane aᵀq = b as a black line."""
    xs = np.array(xlim)
    for a, bi in zip(A, b):
        a1, a2 = a[:2]  # first two coordinates only
        if np.hypot(a1, a2) < 1e-12:
            continue  # plane orthogonal to plotting plane – skip
        if abs(a2) > 1e-12:
            ys = (bi - a1 * xs) / a2
            ax.plot(xs, ys, color="black", lw=lw, zorder=1)
        else:  # vertical line x = bi / a1
            x0 = bi / a1
            ax.plot([x0, x0], ylim, color="black", lw=lw, zorder=1)


def project_constraint(h):
    return h[:-1] - h[-1], -h[-1]


# ---------------------------------------------------------------------------
# 1) Initial constraints + centres ------------------------------------------
# ---------------------------------------------------------------------------
def figure_constraints_centers(
    A: Array, b: Array, c_cheby: Array, c_minko: Array, xlim, ylim
):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_constraints(ax, A, b, xlim, ylim)

    ax.scatter(*c_minko[:2], marker="D", color="orange", s=70, zorder=3)
    ax.scatter(*c_cheby[:2], marker="o", color="blue", s=70, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _set_ticks(ax)
    ax.set_xlabel(r"$support$")
    ax.set_ylabel(r"$confidence$")

    fig.tight_layout()
    outfile = OUTPUT_DIR / "constraints_centers.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outfile.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# 2) Inscribed ball(s) -------------------------------------------------------
# ---------------------------------------------------------------------------
def figure_inscribed_ball(
    A: Array, b: Array, centers: Tuple[Array, Array], r_max: float, xlim, ylim
):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_constraints(ax, A, b, xlim, ylim)

    uniq: List[Array] = []
    for c in centers:
        if not any(np.allclose(c, u, atol=1e-9) for u in uniq):
            uniq.append(c)

    for c in uniq:
        r = _chebyshev_radius(A, b, c)
        ax.add_patch(plt.Circle(c[:2], r, fill=False, linestyle="--", lw=1.2))

    ax.scatter(*centers[0][:2], marker="D", color="orange", s=70, zorder=3)
    ax.scatter(*centers[1][:2], marker="o", color="blue", s=70, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _set_ticks(ax)
    ax.set_xlabel(r"$support$")
    ax.set_ylabel(r"$confidence$")

    fig.tight_layout()
    outfile = OUTPUT_DIR / "inscribed_ball.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outfile.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# 3) Ball + hyper-plane from ambiguous pair
# ---------------------------------------------------------------------------
def figure_ball_and_hplane(
    A: Array,
    b: Array,
    c_cheby: Array,
    c_minko: Array,
    tree_root,
    xlim,
    ylim,
    oracle,  # ← NEW
):
    r = _chebyshev_radius(A, b, c_cheby)

    # -------------- most-ambiguous Φ-pair ---------------------------------
    complete_c = np.hstack([c_cheby, 1 - np.sum(c_cheby)])
    (a_pt, b_pt), *_ = search_pair(tree_root, complete_c, r / 2.0)
    if a_pt is None:
        raise RuntimeError("search_pair returned no ambiguous pair.")

    # orient the hyper-plane with the oracle
    y = oracle(a_pt, b_pt)
    diff_xy = -y * (a_pt - b_pt)  # matches learning update rule

    a_xy = a_pt[:2]  # first two coordinates only
    b_xy = b_pt[:2]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_constraints(ax, A, b, xlim, ylim)

    # inscribed ball (Chebyshev)
    ax.add_patch(
        plt.Circle(c_cheby[:2], r, fill=False, linestyle="--", lw=1.5, color="black")
    )

    # both centres
    ax.scatter(*c_minko[:2], marker="D", color="orange", s=70, zorder=3)
    ax.scatter(*c_cheby[:2], marker="o", color="blue", s=70, zorder=3)

    # ------------------------------------------------ hyper-plane & shading
    a, beta = project_constraint(diff_xy)  #  a·x ≤ β is the constraint

    xs = np.array(xlim)
    if abs(a[1]) > 1e-12:  # non-vertical line
        ys = (beta - a[0] * xs) / a[1]  #  a₁x + a₂y = β
        ax.plot(xs, ys, color="#1f77b4", lw=1.2)

        # which side to shade?
        sign_ref = np.sign(a @ c_cheby[:2] - beta)
        ax.fill_between(
            xs,
            ys,
            ylim[1] if sign_ref >= 0 else ylim[0],
            color="green",
            alpha=0.15,
            zorder=0,
        )

    else:  # vertical line  x = β/a₁
        x0 = beta / a[0]
        ax.axvline(x0, color="#1f77b4", lw=1.2)

        if a[0] * (c_cheby[0] - x0) >= 0:  # Chebyshev centre tells side
            ax.axvspan(x0, xlim[1], color="green", alpha=0.15, zorder=0)
        else:
            ax.axvspan(xlim[0], x0, color="green", alpha=0.15, zorder=0)

    # Φ-points
    ax.scatter(*a_xy, color="#ff7f0e", marker="o", s=55, zorder=3)
    ax.scatter(*b_xy, color="#2ca02c", marker="s", s=55, zorder=3)
    texts = [
        ax.text(*a_xy, r"  $\Phi(r_i)$", fontsize=PHI_SIZE, va="center"),
        ax.text(*b_xy, r"  $\Phi(r_j)$", fontsize=PHI_SIZE, va="center"),
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _set_ticks(ax)
    ax.set_xlabel(r"$support$")
    ax.set_ylabel(r"$confidence$")
    fig.tight_layout()
    (OUTPUT_DIR / "ball_hyperplane.pdf").unlink(missing_ok=True)
    fig.savefig(OUTPUT_DIR / "ball_hyperplane.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {Path('ball_hyperplane.pdf').resolve().relative_to(Path.cwd())}")
    return a, beta


# ---------------------------------------------------------------------------
#  New figure – full feasible set after k iterations  (initial planes shown
#  individually in black, learned planes in blue)
# ---------------------------------------------------------------------------
def figure_feasible_iter_k(
    A0: Array,
    b0: Array,  # initial constraints
    query_A: Array,
    query_b: Array,
    c_cheb: Array,
    c_mink: Array,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    filename: str = "ball_hyperplane_iter3.pdf",
):
    # ------------------ polygon of feasible region ------------------------
    A = np.vstack([A0, query_A])
    b = np.hstack([b0, query_b])

    from scipy.spatial import HalfspaceIntersection, ConvexHull

    halfspaces = np.hstack([A, -(b[:, None])])  # a·x – β ≤ 0
    verts = HalfspaceIntersection(halfspaces, c_cheb[:2]).intersections
    verts = verts[ConvexHull(verts).vertices]

    # plotting window from polygon
    margin = 0.15
    xmin, ymin = -margin, -margin
    xmax, ymax = 1 + margin, 1 + margin
    xs_line = np.linspace(xmin, xmax, 2)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # ----------------------------------------------------------------------
    #   1) shade feasible polygon
    # ----------------------------------------------------------------------
    ax.add_patch(plt.Polygon(verts, closed=True, color="green", alpha=0.15, zorder=0))

    # ----------------------------------------------------------------------
    #   2) *initial* constraints – each drawn separately in black
    # ----------------------------------------------------------------------
    for a, β in zip(A0, b0):
        a1, a2 = a[:2]
        if abs(a2) > 1e-12:  # non-vertical
            ys = (β - a1 * xs_line) / a2
            ax.plot(xs_line, ys, color="black", lw=1.3)
        else:  # vertical
            ax.axvline(β / a1, color="black", lw=1.3)

    # ----------------------------------------------------------------------
    #   3) learned hyper-planes (blue) with arrows
    # ----------------------------------------------------------------------
    for a, β in zip(query_A, query_b):
        a1, a2 = a[:2]
        if abs(a2) > 1e-12:
            ys = (β - a1 * xs_line) / a2
            ax.plot(xs_line, ys, color="#1f77b4", lw=1.1)
            t = (a1 * c_cheb[0] + a2 * c_cheb[1] - β) / (a1**2 + a2**2)
            p_proj = np.array([c_cheb[0] - a1 * t, c_cheb[1] - a2 * t])
        else:
            x0 = β / a1
            ax.axvline(x0, color="#1f77b4", lw=1.1)
            p_proj = np.array([x0, c_cheb[1]])

        # arrow into feasible side (5 % of width)
        normal = np.array([a1, a2])
        vec = -normal / np.linalg.norm(normal) * 0.05 * (xmax - xmin)
        ax.annotate(
            "",
            xytext=p_proj,
            xy=p_proj + vec,
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
        )

    # ----------------------------------------------------------------------
    #   4) centres and their Chebyshev balls
    # ----------------------------------------------------------------------
    for centre, col, mark in [(c_cheb, "blue", "o"), (c_mink, "orange", "D")]:
        r = _chebyshev_radius(A, b, centre)
        ax.add_patch(
            plt.Circle(centre[:2], r, fill=False, linestyle="--", lw=1.2, color="black")
        )
        ax.scatter(*centre[:2], marker=mark, color=col, s=70, zorder=3)

    # ----------------------------------------------------------------------
    #   5) cosmetics
    # ----------------------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _set_ticks(ax)
    ax.set_xlabel(r"$support$")
    ax.set_ylabel(r"$confidence$")
    fig.tight_layout()

    out = OUTPUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# 4) Shared horizontal legend ------------------------------------------------
# ---------------------------------------------------------------------------
def figure_legend():
    handles = [
        plt.Line2D([], [], color="black", linewidth=1.5, label="initial constraints"),
        plt.Line2D(
            [],
            [],
            marker="D",
            color="orange",
            linestyle="None",
            markersize=8,
            label="Minkowski centre",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="blue",
            linestyle="None",
            markersize=8,
            label="Chebyshev centre",
        ),
        plt.Line2D(
            [], [], color="black", linestyle="--", linewidth=1.2, label="inscribed ball"
        ),
        plt.Line2D([], [], color="#1f77b4", linewidth=1.2, label="hyper-plane"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="#ff7f0e",
            linestyle="None",
            markersize=8,
            label=r"$\Phi(r_i)$",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            color="#2ca02c",
            linestyle="None",
            markersize=8,
            label=r"$\Phi(r_j)$",
        ),
    ]

    fig, ax = plt.subplots(figsize=(6, 0.6))
    ax.axis("off")
    ax.legend(
        handles=handles,
        ncol=len(handles),
        frameon=False,
        loc="center",
        bbox_to_anchor=(0, 0),
        handlelength=1,
    )
    fig.tight_layout()
    outfile = OUTPUT_DIR / "legend.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outfile.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# Main driver ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    n_features, add_k, N_points = 2, 2, 100_000
    rng = np.random.default_rng(3)

    # ----- data & tree ----------------------------------------------------
    raw_pts = rng.random((N_points, n_features))
    pts = augment_with_minimums(raw_pts, add_k)
    tree = build_balltree(pts, min_leaf_size=10)

    # ----- initial constraints & centres ---------------------------------
    A0, b0 = k_additive_constraints(n_features, add_k)
    c_cheb0 = chebyshev_center(A0, b0)
    c_mink0 = minkowski_center(A0, b0)
    r0 = _chebyshev_radius(A0, b0, c_cheb0)
    pad = 1.3
    xlim0 = (c_cheb0[0] - pad * r0, c_cheb0[0] + pad * r0)
    ylim0 = (c_cheb0[1] - pad * r0, c_cheb0[1] + pad * r0)

    # ----- “before learning” figures -------------------------------------
    figure_constraints_centers(A0, b0, c_cheb0, c_mink0, xlim0, ylim0)
    figure_inscribed_ball(A0, b0, (c_mink0, c_cheb0), r0, xlim0, ylim0)
    a, beta = figure_ball_and_hplane(
        A0, b0, c_cheb0, c_mink0, tree, xlim0, ylim0, oracle=sum_oracle
    )

    # ----- run active learning for **3** iterations ----------------------
    from learn import learn

    centre_final, A_fin, b_fin = learn(
        root=tree,
        A0=A0,
        b0=b0,
        center_fn=chebyshev_center,
        oracle=sum_oracle,
        n_iter=3,
    )
    query_A = A_fin[len(A0) :]
    query_b = b_fin[len(b0) :]

    # ----- centres & window AFTER 3 iterations ---------------------------
    c_cheb3 = centre_final
    c_mink3 = minkowski_center(A_fin, b_fin)
    r3 = _chebyshev_radius(A_fin, b_fin, c_cheb3)
    query_A = A_fin[len(A0) :]  # rows added during learning
    figure_feasible_iter_k(
        A0,
        b0,
        query_A,
        query_b,
        c_cheb3,
        c_mink3,
        xlim0,
        ylim0,
        filename="ball_hyperplane_iter3.pdf",
    )

    figure_legend()


if __name__ == "__main__":
    main()
