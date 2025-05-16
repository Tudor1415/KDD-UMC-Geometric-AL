import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# USER-CONFIGURABLE PARAMETERS
CSV_PATH = "2_merged_results.csv"
N_ITER = 2
OUT_DIR = "saved_plots"
os.makedirs(OUT_DIR, exist_ok=True)
# ------------------------------------------------------------------

metrics = {
    "ap@p(1)": "AP @ 1 %",
    "ap@p(5)": "AP @ 5 %",
    "recall@p(1)": "Recall @ 1 %",
    "recall@p(5)": "Recall @ 5 %",
}

# ⬇ style dictionaries -------------------------------------------------------
centre_colors = {
    "minkowski_center": "orange",
    "chebyshev_center": "blue",
    "ChoquetRank": "red",
}
oracle_styles = {
    "phi-oracle": "-",
    "surprise-independent": "--",
}
oracle_labels = {
    "phi-oracle": r"$\phi$",
    "surprise-independent": r"$\max \; \mathcal{H}$",
}
centre_labels = {
    "minkowski_center": "Minkowski Center",
    "chebyshev_center": "Chebyshev Center",
    "ChoquetRank": "ChoquetRank",
}
# --------------------------------------------------------------------------- #

# ───────── LOAD + FILTER ─────────────────────────────────────────────────── #
df = pd.read_csv(CSV_PATH)

run_lengths = (
    df.groupby(["dataset", "fold", "oracle", "centre"])["iter"]
    .max()
    .reset_index(name="max_iter")
)
valid_runs = run_lengths[run_lengths["max_iter"] > N_ITER]

df = df.merge(
    valid_runs[["dataset", "fold", "oracle", "centre"]],
    on=["dataset", "fold", "oracle", "centre"],
    how="inner",
)

agg = (
    df.groupby(["oracle", "centre", "iter"])[list(metrics.keys())].mean().reset_index()
)

centres = sorted(agg["centre"].unique())  # e.g. ["cheby", "mink"]
oracles = sorted(agg["oracle"].unique())  # ["phi-oracle", "surprise-independent"]
max_iter = int(agg["iter"].max())

x_ticks = np.arange(0, max_iter + 1, 25)
y_ticks = np.arange(0, 1.01, 0.25)

# ───────── PLOT: one row, four metrics ───────────────────────────────────── #
fig, axes = plt.subplots(
    nrows=1, ncols=len(metrics), figsize=(12, 3), sharex=True, sharey=True
)

for col, (csv_col, pretty_name) in enumerate(metrics.items()):
    ax = axes[col]
    for oracle in oracles:
        for centre in centres:
            d = agg[(agg["oracle"] == oracle) & (agg["centre"] == centre)]
            if d.empty:
                continue
            ax.plot(
                d["iter"].to_numpy(),
                d[csv_col].to_numpy(),
                color=centre_colors.get(centre, "black"),
                linestyle=oracle_styles.get(oracle, "-"),
                linewidth=1.6,
            )
    ax.set_title(pretty_name)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

fig.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "2_metrics_single_row.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig)

# ───────── SEPARATE LEGEND FIGURE ────────────────────────────────────────── #
legend_handles = [
    Line2D([], [], color="orange", linestyle="-", label="Minkowsky"),
    Line2D([], [], color="blue", linestyle="-", label="Chebychev"),
    Line2D([], [], color="red", linestyle="-", label="ChoquetRank"),
    Line2D([], [], color="black", linestyle="-", label=r"$\phi$"),
    Line2D([], [], color="black", linestyle="--", label=r"$\max \; \mathcal{H}$"),
]

fig_leg = plt.figure(figsize=(6, 1.2))
fig_leg.legend(handles=legend_handles, ncol=5, loc="center", frameon=False, fontsize=10)
fig_leg.tight_layout()
fig_leg.savefig(
    os.path.join(OUT_DIR, "metrics_legend.pdf"), format="pdf", bbox_inches="tight"
)
plt.close(fig_leg)
