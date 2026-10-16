"""Compare end-to-end query times for SYN10M (left) and SYN50M (right).

Run: python3 figures/plot_pattern_time_comparison.py
Exports a 3.5-inch-wide vector PDF and a 600-dpi PNG to output/pdf/.
Both datasets share one figure and legend, with separate vertical ranges.
Our and SOTA values include execution plus total_overhead_seconds (sampling
and local rewriting). Use the supplied rounded totals directly: summing the
rounded components can differ by 0.001 s. The overall row is also supplied,
not recomputed from the rounded per-pattern means.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import FIGURE_WIDTH, FONT_SIZES as SHARED_FONT_SIZES

# Size the type for a single column without changing the other figures.
FONT_SIZES = {
    **SHARED_FONT_SIZES,
    "values": 6,
    "ticks": 6,
    "axis": 6,
    "system": 6,
    "legend": 6,
}


@dataclass(frozen=True)
class PatternTimes:
    pattern: str
    valid_count: int
    our_total: float
    sota_total: float
    original: float


DATA_10M = (
    PatternTimes("(A Z* B Z* C)", 18, 25.806, 72.850, 106.215),
    PatternTimes("(A B C D E F G)", 20, 37.833, 44.523, 53.708),
    PatternTimes("(A B+ C D+ E F+)", 20, 40.160, 73.781, 69.637),
    PatternTimes("(A (B+ C)+ D)", 20, 30.048, 71.697, 79.555),
    PatternTimes("((A|B) (C|D))", 20, 34.481, 40.098, 45.288),
    PatternTimes("(A Z* (B+|C+) Z* D)", 18, 56.342, 99.313, 112.819),
    PatternTimes("All patterns", 116, 37.320, 66.387, 76.779),
)

DATA_50M = (
    PatternTimes("(A Z* B Z* C)", 18, 138.714, 188.325, 271.433),
    PatternTimes("(A B C D E F G)", 20, 185.679, 228.534, 246.025),
    PatternTimes("(A B+ C D+ E F+)", 20, 183.713, 253.163, 248.171),
    PatternTimes("(A (B+ C)+ D)", 20, 158.991, 222.101, 253.035),
    PatternTimes("((A|B) (C|D))", 20, 185.410, 219.821, 232.916),
    PatternTimes("(A Z* (B+|C+) Z* D)", 18, 163.409, 233.259, 285.008),
    PatternTimes("All patterns", 116, 169.949, 224.663, 255.335),
)

# Keep the requested SYN10M range; allow room for SYN50M's 285.008 s maximum.
DATASETS = {
    "SYN10M": (DATA_10M, 125, (0, 30, 60, 90, 120)),
    "SYN50M": (DATA_50M, 300, (0, 50, 100, 150, 200, 250, 300)),
}

# Match the hues in plot_query_time_cdf.py, used by the reference 2x2 plot.
METHODS = (
    ("PARE (Ours)", "our_total", "#D55E00", ""),
    ("SOTA-R2", "sota_total", "#B69A50", "///"),
    ("Original", "original", "#555555", "..."),
)


def draw_dataset(ax, dataset: str) -> None:
    data, y_max, y_ticks = DATASETS[dataset]
    centers = np.arange(len(data), dtype=float)
    centers[-1] += 0.4
    ax.axvspan(centers[-1] - 0.46, centers[-1] + 0.46, color="#F1F3F5", zorder=0)
    ax.axvline(5.72, color="#ABB1B7", linestyle="--", linewidth=0.55, zorder=1)

    for index, (label, attribute, color, hatch) in enumerate(METHODS):
        values = [getattr(row, attribute) for row in data]
        ax.bar(
            centers + (index - 1) * 0.23,
            values,
            width=0.20,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="#333333",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_xticks(
        centers,
        [f"P{index + 1}" for index in range(len(data) - 1)] + ["All"],
    )
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.6, centers[-1] + 0.54)
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.set_xlabel("Pattern", labelpad=4)
    ax.grid(axis="y", color="#DDE1E5", linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#60666C")


def main() -> None:
    # Keep the shared font family and PDF embedding, with larger column-sized type.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZES["values"],
            "axes.labelsize": FONT_SIZES["axis"],
            "axes.labelweight": "bold",
            "axes.titlesize": FONT_SIZES["system"],
            "axes.titleweight": "bold",
            "xtick.labelsize": FONT_SIZES["ticks"],
            "ytick.labelsize": FONT_SIZES["ticks"],
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 1.5))
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        draw_dataset(ax, dataset)
        ax.set_title(f"({'ab'[index]}) {dataset}", pad=4)
    y_label = fig.supylabel(
        "End-to-end latency (s)",
        x=0.018,
        y=0.5,
        fontsize=FONT_SIZES["axis"],
        fontweight="bold",
    )
    legend = fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        prop={"size": FONT_SIZES["legend"], "weight": "bold"},
        frameon=False,
        borderaxespad=0,
        columnspacing=3.5,
        handlelength=2.6,
        handleheight=1.1,
        handletextpad=0.4,
    )
    # Figure-level legends are not handled by tight_layout; reserve their height.
    fig.canvas.draw()
    legend_bottom = (
        legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    )
    fig.tight_layout(pad=0.35, w_pad=0.35, rect=(0, 0, 1, legend_bottom - 0.015))
    y_label.set_y((fig.subplotpars.bottom + fig.subplotpars.top) / 2)

    output_dir = Path(__file__).resolve().parents[1] / "fig_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"6_pattern_time_comparison.{extension}"
        fig.savefig(path, dpi=600, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
