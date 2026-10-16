"""Compare rewrite effectiveness rate (RER) across systems and datasets.

Run: python3 figures/rer_system_comparison.py
Writes PNG and PDF files to fig_output/. Values are percentages (0–100).
Typography, colors, and hatches follow arc_system_comparison_bordered.py.
"""

from pathlib import Path
from plot_style import FIGURE_WIDTH, FONT_SIZES as SHARED_FONT_SIZES

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

DATASETS = ["CRIMES", "NASDAQ"]
METHODS = [
    "DS-4.1 (w/o thinking)",
    "GPT-5.6-Sol (w/o reasoning)",
    "DS-4.1 (w/ thinking)",
    "GPT-5.6-Sol (w/o reasoning)",
    "GPT-6-Astra (w/o reasoning)",
    "SOTA-R2",
    "PARE (Ours)",
]
# Rows follow METHODS; columns follow DATASETS.
SYSTEMS = {
    "CDB": np.array(
        [
            [0, 2],
            [2, 0],
            [21, 28],
            [22, 33],
            [11, 26],
            [45, 24],
            [83, 71],
        ]
    ),
    "Trino": np.array(
        [
            [0, 4],
            [4, 1],
            [26, 35],
            [29, 40],
            [16, 31],
            [19, 21],
            [54, 58],
        ]
    ),
}
COLORS = ["#56B4E9", "#E8BDD3", "#0072B2", "#CC79A7", "#9B8DB5", "#D9C391", "#D55E00"]
HATCHES = ["--", "||", "//", "\\\\", "++", "xx", ""]
FONT_SIZES = SHARED_FONT_SIZES
# Horizontal geometry is independent of the figure's vertical layout.
BAR_WIDTH = 0.13
BAR_STEP = 0.13  # Distance between adjacent method centers; >= BAR_WIDTH.
DATASET_STEP = 1.0  # Distance between the two dataset centers.
FIGURE_HEIGHT = 1.6
LEGEND_Y = 0.97  # Leave a small top margin above the legend.


def main():
    plt.rcParams["hatch.linewidth"] = 0.45
    # Embed TrueType fonts for editable text in the paper's PDF.
    plt.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharey=True)
    x = np.arange(len(DATASETS)) * DATASET_STEP

    for ax, (system, data) in zip(axes, SYSTEMS.items()):
        for i, (color, hatch, values) in enumerate(zip(COLORS, HATCHES, data)):
            positions = x + (i - (len(METHODS) - 1) / 2) * BAR_STEP
            bars = ax.bar(
                positions,
                values,
                BAR_WIDTH,
                color=color,
                hatch=hatch,
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
            # Explicit numeric labels include zero-height bars; units are on the axis.
            ax.bar_label(
                bars,
                labels=[f"{v:g}" for v in values],
                padding=2,
                rotation=90,
                fontsize=FONT_SIZES["values"],
            )

        ax.set_title(system, fontsize=FONT_SIZES["system"], fontweight="bold", pad=4)
        ax.set_xticks(x, DATASETS, fontsize=FONT_SIZES["dataset"], fontweight="bold")
        ax.set_xlim(x[0] - DATASET_STEP / 2, x[-1] + DATASET_STEP / 2)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis="y", labelsize=FONT_SIZES["ticks"])
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    axes[0].set_ylabel(
        "RER (%)", fontsize=FONT_SIZES["axis"], fontweight="bold", labelpad=1
    )
    # One base legend aligns all three rows in columns one and two. Add SOTA
    # and PARE to its final row without reserving a third column above them.
    base_order = [0, 1, 4, 2, 3]
    handles = [
        Patch(facecolor=COLORS[i], edgecolor="#333333", linewidth=0.6, hatch=HATCHES[i])
        for i in range(len(METHODS))
    ]
    # Fit complete names while keeping the requested 2/2/3 arrangement.
    legend_size = FONT_SIZES["legend"]
    while legend_size >= 5:
        legend_style = dict(
            prop={"size": legend_size, "weight": "bold"},
            frameon=False,
            columnspacing=0.6,
            borderaxespad=0,
            borderpad=0,
            labelspacing=0.5,
            handlelength=1.4,
            handleheight=1.4,
            handletextpad=0.4,
        )
        base_legend = fig.legend(
            [handles[i] for i in base_order],
            [METHODS[i] for i in base_order],
            loc="upper center",
            bbox_to_anchor=(0.5, LEGEND_Y),
            ncol=2,
            **legend_style,
        )
        legends = [base_legend]
        anchors = [(0.5, LEGEND_Y)]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        last_row = base_legend.get_texts()[2]
        baseline_y = last_row.get_transform().transform(last_row.get_position())[1]
        anchor_y = last_row.get_window_extent(renderer).y1 / fig.bbox.height
        anchor_x = (
            base_legend.legend_handles[3].get_window_extent(renderer).x0
            / fig.bbox.width
        )
        column_gap = 0.6 * legend_size / (fig.get_figwidth() * 72)
        for index in (5, 6):
            legend = fig.legend(
                [handles[index]],
                [METHODS[index]],
                loc="upper left",
                bbox_to_anchor=(anchor_x, anchor_y),
                ncol=1,
                **legend_style,
            )
            fig.canvas.draw()
            text = legend.get_texts()[0]
            text_y = text.get_transform().transform(text.get_position())[1]
            aligned_y = anchor_y + (baseline_y - text_y) / fig.bbox.height
            legend.set_bbox_to_anchor((anchor_x, aligned_y))
            legends.append(legend)
            anchors.append((anchor_x, aligned_y))
            fig.canvas.draw()
            box = legend.get_window_extent().transformed(fig.transFigure.inverted())
            anchor_x = box.x1 + column_gap
        bounds = [
            legend.get_window_extent().transformed(fig.transFigure.inverted())
            for legend in legends
        ]
        # Center the whole group while preserving column and baseline alignment.
        shift_x = (
            0.5 - (min(box.x0 for box in bounds) + max(box.x1 for box in bounds)) / 2
        )
        for legend, (anchor_x, anchor_y) in zip(legends, anchors):
            legend.set_bbox_to_anchor((anchor_x + shift_x, anchor_y))
        fig.canvas.draw()
        bounds = [
            legend.get_window_extent().transformed(fig.transFigure.inverted())
            for legend in legends
        ]
        if all(box.x0 >= 0.01 and box.x1 <= 0.99 for box in bounds):
            break
        for legend in legends:
            legend.remove()
        legend_size -= 0.25
    else:
        raise ValueError(
            "Legend names are too wide for the 2/2/3 layout at a readable size."
        )
    # Reserve the figure-level legend and size margins from the actual text.
    # A fixed percentage bottom margin can clip dataset labels on short figures.
    fig.tight_layout(
        rect=(0, 0, 1, min(box.y0 for box in bounds) - 0.02), pad=0.4, w_pad=0.5
    )

    output_dir = Path(__file__).resolve().parents[1] / "fig_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"6_RER_comparison.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
