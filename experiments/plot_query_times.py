"""Plot per-query average execution times for multiple experiment methods.

Each optional positional argument is either a directory name under
``experiments`` or an explicit directory path.  With no arguments, the three
currently available experiment directories are compared.  A method whose
query did not complete uses the average time of the same query from the
original-query baseline.

Example::

    python experiments/plot_query_times.py

Or pass a different directory list::

    python experiments/plot_query_times.py \
        oracle_crimes_deepseek_rewriting \
        oracle_crimes_gpt5.6_rewriting \
        oracle_crimes_original
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = "oracle_crimes_original"
DEFAULT_DIRECTORIES = (
    "oracle_crimes_deepseek_rewriting",
    "oracle_crimes_gpt5.6_rewriting",
    "oracle_crimes_original",
)
# Enable this together with --output and fig.savefig() below when an image
# file is needed again.
# DEFAULT_OUTPUT = EXPERIMENTS_DIR / "query_times_comparison.png"


@dataclass(frozen=True)
class MethodTimes:
    """Plot-ready data for one experiment method."""

    name: str
    values: list[float]
    substituted_ids: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        default=DEFAULT_DIRECTORIES,
        help=(
            "directory names under experiments, or explicit directory paths; "
            "defaults to the current DeepSeek, GPT-5.6, and original folders"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help=(
            "original-query directory used when a method fails "
            f"(default: {DEFAULT_BASELINE})"
        ),
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=100,
        help="number of query IDs to plot, starting at 1 (default: 100)",
    )
    # parser.add_argument(
    #     "--output",
    #     type=Path,
    #     default=DEFAULT_OUTPUT,
    #     help=f"output image path (default: {DEFAULT_OUTPUT})",
    # )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="interactive figure resolution (default: 100)",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=16.0,
        help="figure width in inches (default: 16)",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=5.5,
        help="figure height in inches (default: 5.5)",
    )
    return parser.parse_args()


def resolve_directory(value: str) -> Path:
    """Resolve a folder name relative to experiments, while accepting paths."""

    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()

    under_experiments = EXPERIMENTS_DIR / supplied
    if under_experiments.is_dir():
        return under_experiments.resolve()
    return supplied.resolve()


def load_queries(directory: Path) -> dict[int, dict[str, Any]]:
    metrics_path = directory / "query_times.json"
    if not directory.is_dir():
        raise ValueError(f"Experiment directory does not exist: {directory}")
    if not metrics_path.is_file():
        raise ValueError(f"query_times.json does not exist: {metrics_path}")

    try:
        with metrics_path.open(encoding="utf-8") as source:
            document = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {metrics_path}: {exc}") from exc

    raw_queries = document.get("queries") if isinstance(document, dict) else None
    if not isinstance(raw_queries, list):
        raise ValueError(f"The queries array is missing from {metrics_path}")

    queries: dict[int, dict[str, Any]] = {}
    for item in raw_queries:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid query entry in {metrics_path}")
        query_id = item.get("id")
        if not isinstance(query_id, int) or isinstance(query_id, bool) or query_id < 1:
            raise ValueError(f"Invalid query ID {query_id!r} in {metrics_path}")
        if query_id in queries:
            raise ValueError(f"Duplicate query ID {query_id} in {metrics_path}")
        queries[query_id] = item
    return queries


def average_seconds(query: dict[str, Any], context: str) -> float:
    value = query.get("average_query_latency_seconds")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid average_query_latency_seconds for {context}")
    return float(value)


def query_failed(query: dict[str, Any]) -> bool:
    """Return whether a non-baseline query should use the baseline time.

    Current result files expose ``output_complete``.  The run-status fallback
    keeps the plotter compatible with older/future files that omit that field.
    If neither signal exists, a valid stored average is treated as successful.
    """

    output_complete = query.get("output_complete")
    if isinstance(output_complete, bool):
        return not output_complete

    runs = query.get("runs")
    if isinstance(runs, list) and runs:
        return not any(
            isinstance(run, dict) and run.get("status") == "success" for run in runs
        )
    return False


def prepare_method_times(
    directories: Sequence[Path], baseline_directory: Path, query_count: int
) -> list[MethodTimes]:
    if query_count < 1:
        raise ValueError("query-count must be positive")

    baseline_queries = load_queries(baseline_directory)
    baseline_values: dict[int, float] = {}
    for query_id in range(1, query_count + 1):
        if query_id not in baseline_queries:
            raise ValueError(
                f"Baseline {baseline_directory.name} is missing query {query_id}"
            )
        baseline_values[query_id] = average_seconds(
            baseline_queries[query_id],
            f"query {query_id} in {baseline_directory.name}",
        )

    methods: list[MethodTimes] = []
    for directory in directories:
        queries = (
            baseline_queries
            if directory == baseline_directory
            else load_queries(directory)
        )
        values: list[float] = []
        substituted_ids: set[int] = set()

        for query_id in range(1, query_count + 1):
            query = queries.get(query_id)
            use_baseline = directory != baseline_directory and (
                query is None or query_failed(query)
            )
            if use_baseline:
                values.append(baseline_values[query_id])
                substituted_ids.add(query_id)
                continue

            if query is None:
                # Only reachable for an incomplete baseline, checked above.
                raise ValueError(f"{directory.name} is missing query {query_id}")
            try:
                values.append(
                    average_seconds(query, f"query {query_id} in {directory.name}")
                )
            except ValueError:
                if directory == baseline_directory:
                    raise
                values.append(baseline_values[query_id])
                substituted_ids.add(query_id)

        methods.append(
            MethodTimes(
                name=directory.name,
                values=values,
                substituted_ids=substituted_ids,
            )
        )
    return methods


def plot_methods(
    methods: Sequence[MethodTimes],
    dpi: int,
    figure_width: float,
    figure_height: float,
) -> None:
    if not methods:
        raise ValueError("At least one experiment directory is required")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    if figure_width <= 0 or figure_height <= 0:
        raise ValueError("figure width and height must be positive")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install project dependencies with "
            "`python -m pip install -r requirements.txt`"
        ) from exc

    query_count = len(methods[0].values)
    query_ids = list(range(1, query_count + 1))
    method_count = len(methods)
    group_width = 0.84
    bar_width = group_width / method_count

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=dpi,
        constrained_layout=True,
    )
    color_map = plt.get_cmap("tab20")
    plotted_bars: list[Any] = []

    for method_index, method in enumerate(methods):
        offset = (method_index - (method_count - 1) / 2) * bar_width
        positions = [query_id + offset for query_id in query_ids]
        bars = ax.bar(
            positions,
            method.values,
            width=bar_width * 0.94,
            label=method.name,
            color=color_map(method_index % color_map.N),
            edgecolor="black",
            linewidth=0.25,
        )
        for query_id, bar in zip(query_ids, bars):
            if query_id in method.substituted_ids:
                bar.set_hatch("////")
        plotted_bars.append(bars)

    tick_step = 1 if query_count <= 30 else 5 if query_count <= 120 else 10
    shown_ticks = list(range(1, query_count + 1, tick_step))
    if shown_ticks[-1] != query_count:
        shown_ticks.append(query_count)
    ax.set_xticks(shown_ticks)
    ax.set_xlim(0.35, query_count + 0.65)
    ax.set_xlabel("Query ID")
    ax.set_ylabel("Average execution time (seconds)")
    ax.set_title("Average Execution Time by Query and Method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    handles = [bars[0] for bars in plotted_bars]
    labels = [method.name for method in methods]
    if any(method.substituted_ids for method in methods):
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="////",
                label="Failed query: original time used",
            )
        )
        labels.append("Failed query: original time used")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )

    # Re-enable these lines together with --output above to write a PNG file.
    # output = DEFAULT_OUTPUT  # Or pass args.output into this function.
    # output.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    directories = [resolve_directory(value) for value in args.directories]
    if len(set(directories)) != len(directories):
        raise ValueError("Experiment directories must not be repeated")
    baseline_directory = resolve_directory(args.baseline)

    methods = prepare_method_times(directories, baseline_directory, args.query_count)
    plot_methods(
        methods,
        args.dpi,
        args.figure_width,
        args.figure_height,
    )

    # print(f"Saved chart to {args.output.resolve()}")
    print(f"\nAverage query latency over {args.query_count} queries:")
    for method in methods:
        mean_latency = statistics.fmean(method.values)
        print(f"{method.name}: {mean_latency:.6f} seconds")

    for method in methods:
        if method.substituted_ids:
            ids = ", ".join(map(str, sorted(method.substituted_ids)))
            print(
                f"{method.name}: used original times for "
                f"{len(method.substituted_ids)} failed queries ({ids})"
            )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error
