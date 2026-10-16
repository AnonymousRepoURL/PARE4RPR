"""Horizontal version of the grouped Oracle ablation comparison.

Run: python3 figures/plot_oracle_ablation_horizontal.py
Save independent PNG and PDF files to fig_output/; do not generate SVG.
Reuse the vertical plot's data, colors, hatches, and ARC-style typography.
"""

from pathlib import Path
from plot_style import FIGURE_WIDTH, legend_bottom

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot_oracle_ablation import paper_style, LABELS, DATA, COLORS, HATCHES, FONT_SIZES

FIGURE_HEIGHT = 1.65  # Inches; keep value labels readable at single-column width.
VALUE_FONT_SIZE = 5  # Smaller numeric labels leave clearance around adjacent bars.
VALUE_Y_OFFSET = -0.4  # Points; negative offsets move text down on the page.
DATASET_FONT_SIZE = 5


def main():
    paper_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    y = np.arange(len(DATA))
    height = 0.14
    row_spacing = 0.185
    order = [1, 2, 3, 4, 0]  # Ablated modules first; full method last.
    for row, i in enumerate(order):
        color, hatch = COLORS[i], HATCHES[i]
        ratios = [values[i] / values[0] for values in DATA.values()]
        bars = ax.barh(
            y + (row - 2) * row_spacing,
            ratios,
            height=height,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        labels = ax.bar_label(
            bars,
            labels=[f"{v:.2f}" for v in ratios],
            padding=4,
            fontsize=VALUE_FONT_SIZE,
        )
        for label in labels:
            dx, dy = label.get_position()
            label.set_position((dx, dy + VALUE_Y_OFFSET))
    ax.axvline(1.0, color="black", linestyle=(0, (4, 3)), linewidth=0.8, zorder=2)
    ax.set_xlabel("Normalized Query Execution Time")
    ax.set_yticks(y, list(DATA), fontsize=DATASET_FONT_SIZE, fontweight="bold")
    ax.set_ylim(-0.55, len(DATA) - 0.45)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.6)
    ax.set_xticks(np.arange(0, 2.5, 0.5))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.8", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3, width=0.7)
    # Match the reference: CP/RP/PR on the left; PPR/Full method on the right.
    legend_order = [1, 2, 3, 4, 0]
    handles = [
        Patch(facecolor=COLORS[i], edgecolor="black", linewidth=0.6, hatch=HATCHES[i])
        for i in legend_order
    ]
    legend = fig.legend(
        handles,
        [LABELS[i] for i in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        prop={"size": FONT_SIZES["legend"], "weight": "bold"},
        frameon=False,
        # The long third entry can extend into the empty third row on the right.
        columnspacing=-1.2,
        handlelength=2.0,
        handleheight=1.0,
        labelspacing=0.45,
    )
    fig.tight_layout(rect=(0, 0, 1, legend_bottom(fig, legend) - 0.02), pad=0.4)
    output = Path(__file__).resolve().parents[1] / "fig_output"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        if extension == "pdf":
            path = output / f"6_ablation_comparison_horizontal.{extension}"
        else:
            path = output / f"oracle_ablation_comparison_horizontal.{extension}"
        fig.savefig(
            path,
            dpi=600,
            facecolor="white",
        )
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
