"""ECDF comparison: CDB/Trino rows, CRIMES/NASDAQ columns.

Run: python3 figures/plot_query_time_cdf_2x2.py
     python3 figures/plot_query_time_cdf_2x2.py --xscale log
Exports a 3.5-inch-wide vector PDF and a 300-dpi PNG to fig_output/.
Log-scale exports use a _log suffix and display the 1-300 s range.
ECDF probabilities include all queries, including those below 1 s.
Uses the same effective execution latencies as plot_query_time_cdf.py.
"""

import argparse
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter

from plot_style import FIGURE_WIDTH
from plot_query_time_cdf import FONT_SIZES, METHODS, ROOT, ecdf_points, prepare_results

FIGURE_HEIGHT = 3.3
MARKER_SCALE = FIGURE_WIDTH / 3.5
ZOOM_Y = (0.40, 0.60)  # Change these two values to adjust the inset's ECDF band.
# Each method has a distinct marker, so color is not required for identification.
PRINT_STYLES = {
    "original": ("--", "s"),
    "deepseek_rewriting": ("-", "o"),
    "gpt5.6_rewriting": ("-", "^"),
    "deepseek_fast_rewriting": ("--", "v"),
    "gpt5.6_fast_rewriting": ("--", "D"),
    "gpt6_rewriting": ("-.", "P"),
    "sota_rewriting": (":", "x"),
    "our_rewriting": ("-", "*"),
}


def add_trino_zoom(ax, results, dataset: str, log_main: bool = False) -> None:
    """Show ZOOM_Y's ECDF band without obscuring the full curves."""
    band_label = f"{ZOOM_Y[0]:.0%}–{ZOOM_Y[1]:.0%}"
    lower, upper = [], []
    for result in results:
        ordered = sorted(result.values)
        lower.append(ordered[math.ceil(ZOOM_Y[0] * len(ordered)) - 1])
        upper.append(ordered[math.ceil(ZOOM_Y[1] * len(ordered)) - 1])
    left = max(0, math.floor((min(lower) - 3) / 10) * 10)
    right = math.ceil((max(upper) + 3) / 10) * 10
    zoom_bounds = [0.16, 0.58, 0.42, 0.32] if log_main else [0.47, 0.22, 0.49, 0.43]
    zoom = ax.inset_axes(zoom_bounds, zorder=7)
    zoom.set_facecolor("white")
    for line in ax.lines:
        zoom.step(
            line.get_xdata(),
            line.get_ydata(),
            where="post",
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markevery=line.get_markevery(),
            markersize=line.get_markersize(),
            markerfacecolor="white",
            markeredgewidth=0.75,
            zorder=line.get_zorder(),
        )
    zoom.set(xlim=(left, right), ylim=ZOOM_Y)
    zoom.set_xticks([left, (left + right) / 2, right])
    zoom.set_yticks([ZOOM_Y[0], sum(ZOOM_Y) / 2, ZOOM_Y[1]])
    zoom.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    zoom.tick_params(labelsize=4.5, length=2, width=0.8, pad=1)
    if log_main:
        zoom.set_title("Time (s, linear)", fontsize=5.5, pad=2, fontweight="bold")
    else:
        zoom.set_xlabel("Time (s)", fontsize=5.5, labelpad=2, fontweight="bold")
    zoom.grid(color="#E5E7EA", linestyle=":", linewidth=0.35)
    zoom.set_axisbelow(True)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#60666C")
    ax.indicate_inset_zoom(
        zoom, edgecolor="#60666C", alpha=0.65, linewidth=0.5, linestyle="--", zorder=6
    )
    print(f"Trino / {dataset.upper()} zoom: {left:g}–{right:g} s, {band_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xscale",
        choices=("linear", "log"),
        default="linear",
        help="Horizontal time scale (default: linear).",
    )
    args = parser.parse_args()
    log_scale = args.xscale == "log"
    all_results = {}
    for system in ("oracle", "trino"):
        for dataset in ("crimes", "nasdaq"):
            directories = [
                ROOT / "experiments" / f"{system}_{dataset}_{suffix}"
                for suffix, *_ in METHODS
            ]
            all_results[system, dataset] = prepare_results(directories)
    x_left = 1.0 if log_scale else 0.0
    x_right = 300.0 if log_scale else 306.0
    ecdf_left = 0.0
    if log_scale:
        minimum = min(
            value
            for results in all_results.values()
            for result in results
            for value in result.values
        )
        if minimum <= 0:
            raise ValueError("A logarithmic time axis requires positive query times.")
        ecdf_left = minimum * 0.8
        print(
            f"Log time axis: {x_left:g}-{x_right:g} s; minimum observed time: {minimum:g} s"
        )

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True, sharey=True
    )
    for row, system in enumerate(("oracle", "trino")):
        for col, dataset in enumerate(("crimes", "nasdaq")):
            ax = axes[row, col]
            prefix = f"{system}_{dataset}"
            results = all_results[system, dataset]
            for result, (suffix, label, color, _) in zip(results, METHODS):
                x, y = ecdf_points(result.values, x_right)
                # Move only the synthetic zero-probability boundary on a log axis.
                # Observed latencies, ECDF probabilities, and timeouts stay intact.
                x[0] = ecdf_left
                ours, original = suffix == "our_rewriting", suffix == "original"
                linestyle, marker = PRINT_STYLES[suffix]
                ax.step(
                    x,
                    y,
                    where="post",
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.4 if ours else 1.0,
                    marker=marker,
                    markevery=(10, 20),
                    markersize=(4.2 if ours else 3.4) * MARKER_SCALE,
                    markerfacecolor="white",
                    markeredgewidth=0.75,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    zorder=5 if ours else (4 if original else 3),
                )
            letter = "abcd"[row * 2 + col]
            system_label = "CDB" if system == "oracle" else "Trino"
            ax.set_title(f"({letter}) {system_label} / {dataset.upper()}", pad=4)
            ax.set_xscale(args.xscale)
            ax.set(xlim=(x_left, x_right), ylim=(0, 1.04))
            if log_scale:
                ax.set_xticks([1, 10, 100, 300], ["1", "10", "100", "300"])
                ax.xaxis.set_minor_locator(NullLocator())
            else:
                ax.set_xticks([0, 100, 200, 300])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.grid(axis="y", color="#DDE1E5", linestyle="--", linewidth=0.4)
            ax.grid(axis="x", color="#ECEEF0", linestyle=":", linewidth=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(direction="out", length=3, width=0.8, pad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#60666C")
            print(f"{prefix}: {len(results)} methods, {len(results[0].values)} queries")
            if system == "trino" and dataset == "crimes":
                add_trino_zoom(ax, results, dataset, log_main=log_scale)

    # Keep full method names on one line in a compact shared legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Pair reasoning variants by row.
    order = [1, 2, 5, 7, 3, 4, 6, 0]
    legend = fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        prop={"size": FONT_SIZES["legend"], "weight": "bold"},
        frameon=False,
        columnspacing=2.4,
        borderaxespad=0,
        handlelength=1.6,
        handleheight=1.4,
        handletextpad=0.4,
        labelspacing=0.4,
    )
    fig.canvas.draw()
    legend_bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if legend_bounds.x0 < 0 or legend_bounds.x1 > 1:
        raise ValueError("Legend exceeds the figure width")
    fig.subplots_adjust(
        left=0.135,
        right=0.975,
        bottom=0.10,
        top=legend_bounds.y0 - 0.045,
        wspace=0.16,
        hspace=0.30,
    )
    fig.supxlabel(
        (
            "Query Execution Time (s, log scale; excluding rewriting time)"
            if log_scale
            else "Query Execution Time (excluding rewriting time)"
        ),
        x=0.5,
        y=0.020,
        fontsize=FONT_SIZES["axis"],
        fontweight="bold",
    )
    fig.supylabel(
        "Cumulative Distribution Function (CDF)",
        x=0.015,
        y=(fig.subplotpars.top + fig.subplotpars.bottom) / 2,
        fontsize=FONT_SIZES["axis"],
        fontweight="bold",
    )
    output_dir = Path(__file__).resolve().parents[1] / "fig_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if log_scale else ""
    for extension in ("png", "pdf"):
        path = output_dir / f"query_time_cdf_2x2{suffix}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    if not log_scale:
        fig.savefig(output_dir / "6_execution_time_cdf_2x2.pdf", facecolor="white")
    # A monochrome export provides a direct print-ready alternative.
    for ax in fig.axes:
        for plot_ax in (ax, *ax.child_axes):
            for line in plot_ax.lines:
                line.set_color("#222222")
    for handle in legend.legend_handles:
        handle.set_color("#222222")
    for extension in ("png", "pdf"):
        path = output_dir / f"6_execution_time_cdf_2x2{suffix}_monochrome.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
