import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------  PARAMETERS  -------------------------------- #
CSV_PATH_ITER = "1_add_results.csv"
CSV_PATH_DIFF = "1_add_results_diffvectors.csv"
OUT_DIR = "saved_plots"
N_ITER = 20
os.makedirs(OUT_DIR, exist_ok=True)

# colour by centre (filled for φ, dashed for Surprise)
centre_colors = {"minkowski_center": "orange", "chebyshev_center": "blue"}
oracle_styles = {"phi-oracle": "-", "surprise-independent": "--"}

# ==================  ITER-TIME & RADIUS DATA  ============================== #
df_iter = pd.read_csv(CSV_PATH_ITER)

iter_rad = (
    df_iter.groupby(["oracle", "centre", "iter"])[["iter_time", "radius"]]
    .mean()
    .reset_index()
)

centres = sorted(iter_rad["centre"].unique())  # mink, cheby
oracles = sorted(iter_rad["oracle"].unique())  # φ, Surprise
max_iter = int(iter_rad["iter"].max())

# ==================  DIFF-VECTOR DATA → ANGLES  ============================ #
df_diff = pd.read_csv(CSV_PATH_DIFF)
df_diff = df_diff[
    df_diff["diffvector"].notna() & (df_diff["diffvector"].str.strip() != "[]")
]


def project_constraint(h):
    return np.asarray(h[:-1]) - h[-1], -h[-1]


angles_dict = {}
for (oracle, centre), g in df_diff.groupby(["oracle", "centre"]):
    normals = []
    for v in g["diffvector"]:
        h = ast.literal_eval(v)
        a, _ = project_constraint(h)
        n = np.linalg.norm(a)
        if n:
            normals.append(a / n)
    if len(normals) < 2:
        continue
    ang = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cosθ = np.clip(normals[i] @ normals[j], -1.0, 1.0)
            ang.append(np.arccos(cosθ))
    angles_dict[(oracle, centre)] = np.sort(ang)

# ===========================  PLOT  ======================================== #
fig, axes = plt.subplots(1, 3, figsize=(14, 3), sharex=False)

# --- (0) Mean iteration time ---------------------------------------------- #
ax_time = axes[0]
for oracle in oracles:
    for centre in centres:
        d = iter_rad[(iter_rad["oracle"] == oracle) & (iter_rad["centre"] == centre)]
        if not d.empty:
            ax_time.plot(
                d["iter"],
                d["iter_time"],
                color=centre_colors[centre],
                linestyle=oracle_styles[oracle],
                linewidth=1.6,
            )
ax_time.set_title("Mean Iteration Time")
ax_time.set_xlabel("Iteration")
ax_time.set_ylabel("Time")
ax_time.set_xticks(np.arange(0, max_iter + 1, 25))
ax_time.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# --- (1) Mean radius ------------------------------------------------------- #
ax_rad = axes[1]
for oracle in oracles:
    for centre in centres:
        d = iter_rad[(iter_rad["oracle"] == oracle) & (iter_rad["centre"] == centre)]
        if not d.empty:
            ax_rad.plot(
                d["iter"],
                d["radius"],
                color=centre_colors[centre],
                linestyle=oracle_styles[oracle],
                linewidth=1.6,
            )
ax_rad.set_title("Mean Radius")
ax_rad.set_xlabel("Iteration")
ax_rad.set_ylabel("Radius")
ax_rad.set_xticks(np.arange(0, max_iter + 1, 25))
ax_rad.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# --- (2) Constraint orientation C.D.F ------------------------------------- #
ax_cdf = axes[2]
for oracle in oracles:
    for centre in centres:
        ang = angles_dict.get((oracle, centre))
        if ang is None or len(ang) == 0:
            continue
        probs = np.arange(1, len(ang) + 1) / len(ang)
        ax_cdf.plot(
            probs,
            ang,
            color=centre_colors[centre],
            linestyle=oracle_styles[oracle],
            linewidth=1.6,
        )

ax_cdf.set_title("Constraint Orientation C.D.F")
ax_cdf.set_xlim(0, 1)
ax_cdf.set_xticks(np.arange(0, 1.01, 0.1))
ax_cdf.set_ylim(0, np.pi)
ax_cdf.set_yticks([0, np.pi / 2, np.pi])
ax_cdf.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
ax_cdf.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

fig.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "summary_one_row.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig)

# ---------------  STAND-ALONE LEGEND FIGURE  ------------------------------ #
from matplotlib.lines import Line2D

handles = [
    Line2D([], [], color="orange", linestyle="-", lw=1.6, label="Minkowsky"),
    Line2D([], [], color="blue", linestyle="-", lw=1.6, label="Cheby"),
    Line2D([], [], color="black", linestyle="-", lw=1.6, label=r"$\phi$"),
    Line2D([], [], color="black", linestyle="--", lw=1.6, label=r"$\max \mathcal{H}$"),
]

fig_leg = plt.figure(figsize=(5, 1.2))
fig_leg.legend(handles=handles, ncol=4, loc="center", frameon=False)
fig_leg.tight_layout()
fig_leg.savefig(
    os.path.join(OUT_DIR, "summary_legend.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig_leg)
