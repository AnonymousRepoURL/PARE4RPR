"""Plot query execution-time ECDFs from experiment results.

Run: python3 figures/plot_query_time_cdf.py
Other results: add --system trino and/or --dataset nasdaq.

Each query contributes one three-run mean. Use the same failure substitution
as experiments/get_main_results.py: failed rewrites fall back to the original
query, and failed originals use 300 seconds. These are effective latencies,
so the curve includes fallback/timeout values, not just successful executions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from plot_style import FIGURE_WIDTH, FONT_SIZES as SHARED_FONT_SIZES, legend_bottom
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.get_main_results import prepare_results

# Keep labels and hues consistent with the existing method-comparison figures.
METHODS = (
    ("original", "Original", "#555555", "--"),
    ("deepseek_rewriting", "DS-4.1 (w/ thinking)", "#0072B2", "-"),
    ("gpt5.6_rewriting", "GPT-5.6-Sol (w/ reasoning)", "#CC79A7", "-"),
    ("deepseek_fast_rewriting", "DS-4.1 (w/o thinking)", "#56B4E9", "--"),
    ("gpt5.6_fast_rewriting", "GPT-5.6-Sol (w/o reasoning)", "#E8BDD3", "--"),
    ("gpt6_rewriting", "GPT-6-Astra (w/ reasoning)", "#9B8DB5", "-"),
    ("sota_rewriting", "SOTA-R2", "#B69A50", "-."),
    ("our_rewriting", "PARE (Ours)", "#D55E00", "-"),
)

# Match arc_system_comparison_bordered.py typography and legend settings.
FONT_SIZES = SHARED_FONT_SIZES


def ecdf_points(values: list[float], right: float) -> tuple[list[float], list[float]]:
    """Include both plot boundaries; steps jump at each observed latency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot plot an empty query set")
    n = len(ordered)
    return [0.0, *ordered, right], [0.0, *(i / n for i in range(1, n + 1)), 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("oracle", "trino"), default="oracle")
    parser.add_argument("--dataset", choices=("crimes", "nasdaq"), default="crimes")
    args = parser.parse_args()
    prefix = f"{args.system}_{args.dataset}"
    directories = [
        ROOT / "experiments" / f"{prefix}_{suffix}" for suffix, *_ in METHODS
    ]
    results = prepare_results(directories)
    right = max(max(result.values) for result in results) * 1.02

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
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    for result, (_, label, color, linestyle) in zip(results, METHODS):
        x, y = ecdf_points(result.values, right)
        is_original = label == "Original"
        is_ours = label == "PARE (Ours)"
        ax.step(
            x,
            y,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.4 if is_ours else 1.0,
            marker="s" if is_original else None,
            markevery=20,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=5 if is_ours else (4 if is_original else 3),
        )
        print(
            f"{label}: {len(result.values)} queries; "
            f"{len(result.substituted_ids)} fallback/timeout values"
        )

    ax.set(
        xlabel="Query Execution Time (s)",
        ylabel="Cumulative Proportion",
        xlim=(0, right),
        ylim=(0, 1.035),
    )
    system_label = "CDB" if args.system == "oracle" else "Trino"
    ax.set_title(f"{system_label} / {args.dataset.upper()}", pad=4)
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4
    ax.set_yticks([i / 5 for i in range(6)])
    ax.set_xticks(range(0, int(right) + 1, 50))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", color="#DDE1E5", linestyle="--", linewidth=0.7)
    ax.grid(axis="x", color="#ECEEF0", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=4, width=0.8, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#60666C")
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=2,
        prop={"size": FONT_SIZES["legend"], "weight": "bold"},
        frameon=False,
        columnspacing=0.6,
        borderaxespad=0,
        handlelength=1.6,
        handleheight=1.4,
        handletextpad=0.4,
    )
    # Reserve only the actual legend height, keeping it close to the title.
    fig.canvas.draw()
    legend_y0 = legend_bottom(fig, legend)
    fig.tight_layout(rect=(0, 0, 1, legend_y0 - 0.025), pad=0.7)
    for extension in ("png", "pdf"):
        output = (
            Path(__file__).resolve().parent / f"{prefix}_query_time_cdf.{extension}"
        )
        fig.savefig(output, dpi=300, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
