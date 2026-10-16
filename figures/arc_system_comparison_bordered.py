from pathlib import Path
from plot_style import FIGURE_WIDTH, FONT_SIZES as SHARED_FONT_SIZES, legend_bottom
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

datasets = ["CRIMES", "NASDAQ"]
dataset_tex_labels = [r"\textsc{crimes}", r"\textsc{nasdaq}"]
# Matplotlib's mathtext does not support \textsc; it requires external LaTeX.
use_tex_labels = bool(shutil.which("latex") and shutil.which("dvipng"))
# Model names abbreviate the corresponding LLM-rewriting methods.
methods = [
    "DS-4.1 (w. Thinking)",
    "GPT-5.6-Sol (w. Reasoning)",
    "DS-4.1 (w.o. Thinking)",
    "GPT-5.6-Sol (w.o. Reasoning)",
    "GPT-6-Astra (w. Reasoning)",
    "SOTA-R2",
    "R3RP (Ours)",
]
# Related model variants share a hue with different lightness; hatches provide
# an additional cue independent of color. Use sand gold for SOTA and vermilion for Ours.
colors = ["#0072B2", "#CC79A7", "#56B4E9", "#E8BDD3", "#9B8DB5", "#D9C391", "#D55E00"]
hatches = ["/", "\\", "-", "|", "+", "x", ""]
plt.rcParams["hatch.linewidth"] = 0.45
# Rows follow methods; columns follow datasets. NaN means unreported.
systems = {
    "CDB": np.array(
        [
            [90.8, 74.0],
            [36.2, 30.3],
            [1.2, 1.0],
            [2.9, 2.3],
            [25.3, 24.7],
            [0.02, 0.1],
            [0.1, 0.3],
        ]
    ),
    "Trino + PostgreSQL": np.array(
        [
            [95.0, 90.6],
            [65.2, 30.5],
            [1.2, 1.0],
            [2.2, 2.4],
            [26.0, 33.4],
            [5.5, 1.9],
            [5.0, 2.6],
        ]
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.2), sharey=True)
x = np.arange(len(datasets))
width = 0.135
# Centralized font sizes for readability after scaling in a paper.
font_sizes = SHARED_FONT_SIZES

for ax, (system, data) in zip(axes, systems.items()):
    for i, (method, color, hatch, values) in enumerate(
        zip(methods, colors, hatches, data)
    ):
        valid = np.isfinite(values)
        positions = x[valid] + (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            positions,
            values[valid],
            width,
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:g}" for v in values[valid]],
            padding=1,
            rotation=90,
            fontsize=font_sizes["values"],
        )

    ax.set_title(system, fontsize=font_sizes["system"], fontweight="bold", pad=4)
    if use_tex_labels:
        ax.set_xticks(
            x, dataset_tex_labels, fontsize=font_sizes["dataset"], usetex=True
        )
    else:
        # Use larger, bold plain labels when LaTeX is unavailable.
        ax.set_xticks(x, datasets, fontsize=font_sizes["dataset"], fontweight="bold")
    ax.set_xlim(-0.50, 1.50)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.tick_params(axis="y", labelsize=font_sizes["ticks"])
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.3, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

axes[0].set_ylabel("ARC (seconds)", fontsize=font_sizes["axis"], fontweight="bold")
# fig.text(0.5, 0.02, "ARC Comparison Across Database Systems",
#          ha="center", va="bottom", fontsize=font_sizes["title"], fontweight="bold")
# Keep each method name on one line; arrange the legend in two columns.
legend_handles = [
    Patch(facecolor=c, edgecolor="#333333", linewidth=0.6, hatch=h)
    for c, h in zip(colors, hatches)
]
legend_labels = methods
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.995),
    ncol=2,
    prop={"size": font_sizes["legend"], "weight": "bold"},
    frameon=False,
    columnspacing=0.6,
    borderaxespad=0,
    handlelength=1.6,
    handleheight=1.4,
    handletextpad=0.4,
)
# Only reserve space above the axes for the shared legend.
# The bottom title is disabled, so no extra bottom margin is needed.
fig.tight_layout(rect=(0, 0, 1, legend_bottom(fig, legend) - 0.055), pad=0.4, w_pad=0.5)

output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "arc_system_comparison_bordered.png", dpi=300)
fig.savefig(output_dir / "6_ARC_comparison.pdf")
