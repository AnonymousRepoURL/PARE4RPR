"""Paper-sized Oracle ablation plot; run: python3 figures/plot_oracle_ablation.py.

Input values are supplied mean latencies. Normalize by each full-method mean.
Save only PNG and PDF beside this script. No uncertainty is inferred.
"""

from pathlib import Path
from plot_style import FIGURE_WIDTH, FONT_SIZES as SHARED_FONT_SIZES, legend_bottom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

LABELS = [
    "Full method",
    "w/o Column Pruning",
    "w/o Row Pruning",
    "w/o Predicate Reconstruction",
    "w/o Pathological Pattern Repair",
]
DATA = {
    "CRIMES": [42.1, 62.6, 56.8, 55.4, 51.2],
    "NASDAQ": [65.3, 68.6, 137.9, 100.1, 72.7],
}
COLORS = ["#D55E00", "#9ECAE1", "#4292C6", "#BDBDBD", "#E2D3AB"]
HATCHES = ["", "///", "\\\\", "...", "xx"]
# Match the centralized font sizes in arc_system_comparison_bordered.py.
FONT_SIZES = SHARED_FONT_SIZES


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FONT_SIZES["ticks"],
            "axes.labelsize": FONT_SIZES["axis"],
            "axes.labelweight": "bold",
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["ticks"],
            "ytick.labelsize": FONT_SIZES["ticks"],
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main():
    paper_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.72)
    x = np.arange(len(DATA))
    width = 0.14
    for i, (label, color, hatch) in enumerate(zip(LABELS, COLORS, HATCHES)):
        ratios = [values[i] / values[0] for values in DATA.values()]
        bars = ax.bar(
            x + (i - 2) * width,
            ratios,
            width=width,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.2f}" for v in ratios],
            padding=2,
            rotation=90,
            fontsize=FONT_SIZES["values"],
        )
    ax.axhline(1.0, color="black", linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.set_ylabel("Normalized mean latency")
    ax.set_xticks(x, list(DATA), fontsize=FONT_SIZES["dataset"], fontweight="bold")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 2.7)
    ax.set_yticks(np.arange(0, 2.5, 0.5))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3, width=0.7)
    handles = [
        Patch(facecolor=c, edgecolor="black", linewidth=0.6, hatch=h)
        for c, h in zip(COLORS, HATCHES)
    ]
    legend = fig.legend(
        handles,
        LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        prop={"size": FONT_SIZES["legend"], "weight": "bold"},
        frameon=False,
        columnspacing=1.4,
        handlelength=1.9,
        handleheight=1.0,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.16, bottom=0.16, top=legend_bottom(fig, legend) - 0.05)
    output = Path(__file__).resolve().parent
    for extension in ("png", "pdf"):
        path = output / f"oracle_ablation_comparison.{extension}"
        fig.savefig(path, dpi=600, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
