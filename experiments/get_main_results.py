"""Compute aggregate execution-time metrics for experiment results.

Each positional argument is either a directory name under ``experiments`` or
an explicit directory path.  Exactly one supplied directory must end in
``_original``; it is used as the original-query baseline for every rewriting
method.

For each completed query, execution time is the mean of its three recorded
runs.  A failed rewritten query uses the corresponding original-query time,
while a failed original query is assigned the 300-second timeout value.

Running example:

python experiments/get_main_results.py \
    oracle_crimes_deepseek_fast_rewriting \
    oracle_crimes_deepseek_rewriting \
    oracle_crimes_gpt5.6_fast_rewriting \
    oracle_crimes_gpt5.6_rewriting \
    oracle_crimes_gpt6_rewriting \
    oracle_crimes_our_rewriting \
    oracle_crimes_sota_rewriting \
    oracle_crimes_original

python experiments/get_main_results.py \
    trino_crimes_deepseek_fast_rewriting \
    trino_crimes_deepseek_rewriting \
    trino_crimes_gpt5.6_fast_rewriting \
    trino_crimes_gpt5.6_rewriting \
    trino_crimes_gpt6_rewriting \
    trino_crimes_our_rewriting \
    trino_crimes_sota_rewriting \
    trino_crimes_original    


python experiments/get_main_results.py \
    oracle_nasdaq_deepseek_fast_rewriting \
    oracle_nasdaq_deepseek_rewriting \
    oracle_nasdaq_gpt5.6_fast_rewriting \
    oracle_nasdaq_gpt5.6_rewriting \
    oracle_nasdaq_gpt6_rewriting \
    oracle_nasdaq_our_rewriting \
    oracle_nasdaq_sota_rewriting \
    oracle_nasdaq_original    

python experiments/get_main_results.py \
    trino_nasdaq_deepseek_fast_rewriting \
    trino_nasdaq_deepseek_rewriting \
    trino_nasdaq_gpt5.6_fast_rewriting \
    trino_nasdaq_gpt5.6_rewriting \
    trino_nasdaq_gpt6_rewriting \
    trino_nasdaq_our_rewriting \
    trino_nasdaq_sota_rewriting \
    trino_nasdaq_original      


Use a different RSR threshold (in percentage points) with, for example,
``--rsr-threshold 10``.
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
DEFAULT_DIRECTORIES = (
    "oracle_crimes_deepseek_rewriting",
    "oracle_crimes_gpt5.6_rewriting",
    "oracle_crimes_original",
)
RUNS_PER_QUERY = 3
TIMEOUT_SECONDS = 300.0
DEFAULT_RSR_THRESHOLD_PERCENT = 20.0


@dataclass(frozen=True)
class MethodResults:
    """Effective execution times and rewrite results for one method."""

    name: str
    values: list[float]
    substituted_ids: set[int]
    successful_rewrites: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="*",
        default=DEFAULT_DIRECTORIES,
        help=(
            "directory names under experiments, or explicit directory paths; "
            "exactly one directory must end in _original"
        ),
    )
    parser.add_argument(
        "--rsr-threshold",
        type=float,
        default=DEFAULT_RSR_THRESHOLD_PERCENT,
        metavar="PERCENT",
        help=(
            "minimum percentage reduction required for a successful rewrite; "
            f"the reduction must be strictly greater (default: "
            f"{DEFAULT_RSR_THRESHOLD_PERCENT:g})"
        ),
    )
    return parser.parse_args()


def resolve_directory(value: str) -> Path:
    """Resolve a directory name relative to experiments, while accepting paths."""

    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()

    under_experiments = EXPERIMENTS_DIR / supplied
    if under_experiments.is_dir():
        return under_experiments.resolve()
    return supplied.resolve()


def load_queries(directory: Path) -> dict[int, dict[str, Any]]:
    """Load and validate the query entries from an experiment directory."""

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
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError(f"The queries array is missing or empty in {metrics_path}")

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


def query_completed(query: dict[str, Any]) -> bool:
    """Return whether the result has usable output.

    ``output_complete`` is authoritative in current result files.  The runs
    fallback retains compatibility with files created before that field was
    added.
    """

    output_complete = query.get("output_complete")
    if isinstance(output_complete, bool):
        return output_complete

    runs = query.get("runs")
    return isinstance(runs, list) and any(
        isinstance(run, dict) and run.get("status") == "success" for run in runs
    )


def three_run_average(query: dict[str, Any]) -> float | None:
    """Return the three-run mean, or ``None`` for a failed/invalid query."""

    if not query_completed(query):
        return None

    runs = query.get("runs")
    if not isinstance(runs, list) or len(runs) != RUNS_PER_QUERY:
        return None

    elapsed_values: list[float] = []
    for run in runs:
        if not isinstance(run, dict):
            return None
        elapsed = run.get("query_elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            return None
        elapsed_values.append(float(elapsed))

    return statistics.fmean(elapsed_values)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Calculate a linearly interpolated percentile (the common R-7 method)."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty data set")
    if not 0 <= fraction <= 1:
        raise ValueError("Percentile fraction must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]

    upper_weight = position - lower_index
    return (
        ordered[lower_index] * (1 - upper_weight) + ordered[upper_index] * upper_weight
    )


def prepare_results(
    directories: Sequence[Path],
    rsr_threshold_percent: float = DEFAULT_RSR_THRESHOLD_PERCENT,
) -> list[MethodResults]:
    """Apply failure substitution and calculate rewrite successes."""

    if not directories:
        raise ValueError("At least one experiment directory is required")
    if len(set(directories)) != len(directories):
        raise ValueError("Experiment directories must not be repeated")
    if (
        not math.isfinite(rsr_threshold_percent)
        or not 0 <= rsr_threshold_percent <= 100
    ):
        raise ValueError("rsr-threshold must be between 0 and 100")

    success_time_ratio = 1 - rsr_threshold_percent / 100

    original_directories = [
        directory for directory in directories if directory.name.endswith("_original")
    ]
    if len(original_directories) != 1:
        raise ValueError(
            "Exactly one input directory must have a name ending in _original"
        )
    original_directory = original_directories[0]
    original_queries = load_queries(original_directory)
    query_ids = sorted(original_queries)

    original_values: dict[int, float] = {}
    original_failed_ids: set[int] = set()
    for query_id in query_ids:
        average = three_run_average(original_queries[query_id])
        if average is None:
            average = TIMEOUT_SECONDS
            original_failed_ids.add(query_id)
        original_values[query_id] = average

    results: list[MethodResults] = []
    for directory in directories:
        if directory == original_directory:
            results.append(
                MethodResults(
                    name=directory.name,
                    values=[original_values[query_id] for query_id in query_ids],
                    substituted_ids=original_failed_ids,
                    successful_rewrites=None,
                )
            )
            continue

        queries = load_queries(directory)
        values: list[float] = []
        substituted_ids: set[int] = set()
        successful_rewrites = 0

        for query_id in query_ids:
            query = queries.get(query_id)
            average = three_run_average(query) if query is not None else None
            if average is None:
                average = original_values[query_id]
                substituted_ids.add(query_id)

            values.append(average)
            if average < original_values[query_id] * success_time_ratio:
                successful_rewrites += 1

        results.append(
            MethodResults(
                name=directory.name,
                values=values,
                substituted_ids=substituted_ids,
                successful_rewrites=successful_rewrites,
            )
        )

    return results


def print_results(
    results: Sequence[MethodResults], rsr_threshold_percent: float
) -> None:
    """Print aggregate latency statistics and rewrite success rates."""

    query_count = len(results[0].values)
    print(f"Results across {query_count} original queries (seconds):")
    print(
        "RSR criterion: execution-time reduction strictly greater than "
        f"{rsr_threshold_percent:g}%"
    )
    print(
        f"{'Method':<42} {'Mean':>10} {'Median':>10} "
        f"{'75th':>10} {'95th':>10} {'RSR':>14}"
    )
    print("-" * 102)

    for result in results:
        if result.successful_rewrites is None:
            rsr = "-"
        else:
            rsr = (
                f"{result.successful_rewrites / query_count:.2%} "
                f"({result.successful_rewrites}/{query_count})"
            )

        print(
            f"{result.name:<42} "
            f"{statistics.fmean(result.values):>10.3f} "
            f"{statistics.median(result.values):>10.3f} "
            f"{percentile(result.values, 0.75):>10.3f} "
            f"{percentile(result.values, 0.95):>10.3f} "
            f"{rsr:>14}"
        )

    for result in results:
        if not result.substituted_ids:
            continue
        ids = ", ".join(map(str, sorted(result.substituted_ids)))
        if result.successful_rewrites is None:
            action = f"assigned {TIMEOUT_SECONDS:g} seconds"
        else:
            action = "used original-query times"
        print(
            f"{result.name}: {action} for {len(result.substituted_ids)} "
            f"failed queries ({ids})"
        )


def main() -> None:
    args = parse_args()
    directories = [resolve_directory(value) for value in args.directories]
    results = prepare_results(directories, args.rsr_threshold)
    print_results(results, args.rsr_threshold)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
