"""Summarize LLM query-rewrite times from experiment log directories.

Each positional argument may be either a directory name under ``experiments``
or an explicit directory path.  Every directory must contain exactly one
``.txt`` file with 100 log entries in either of these forms::

    [1/100] (oracle): 5.962s, 1506 tokens, estimated $0.00906400
    [1/100] 0.017s (sampling 0.000s): column_pruning

Example::

python experiments/summarize_rewrite_times.py \
    oracle_crimes_deepseek_fast_rewriting \
    oracle_crimes_deepseek_rewriting \
    oracle_crimes_gpt5.6_fast_rewriting \
    oracle_crimes_gpt5.6_rewriting \
    oracle_crimes_gpt6_rewriting \
    oracle_crimes_our_rewriting \
    oracle_crimes_sota_rewriting

python experiments/summarize_rewrite_times.py \
    oracle_nasdaq_deepseek_fast_rewriting \
    oracle_nasdaq_deepseek_rewriting \
    oracle_nasdaq_gpt5.6_fast_rewriting \
    oracle_nasdaq_gpt5.6_rewriting \
    oracle_nasdaq_gpt6_rewriting \
    oracle_nasdaq_our_rewriting \
    oracle_nasdaq_sota_rewriting

python experiments/summarize_rewrite_times.py \
    trino_crimes_deepseek_fast_rewriting \
    trino_crimes_deepseek_rewriting \
    trino_crimes_gpt5.6_fast_rewriting \
    trino_crimes_gpt5.6_rewriting \
    trino_crimes_gpt6_rewriting \
    trino_crimes_our_rewriting \
    trino_crimes_sota_rewriting    

python experiments/summarize_rewrite_times.py \
    trino_nasdaq_deepseek_fast_rewriting \
    trino_nasdaq_deepseek_rewriting \
    trino_nasdaq_gpt5.6_fast_rewriting \
    trino_nasdaq_gpt5.6_rewriting \
    trino_nasdaq_gpt6_rewriting \
    trino_nasdaq_our_rewriting \
    trino_nasdaq_sota_rewriting
"""

from __future__ import annotations

import argparse
import re
from decimal import Decimal
from pathlib import Path
from typing import Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parent
QUERY_COUNT = 100
QUERY_TIME_PATTERN = re.compile(
    r"^\[(?P<query_id>\d+)/(?P<query_count>\d+)\]"
    r"\s+(?:\([^)]*\):\s+)?"
    r"(?P<seconds>\d+(?:\.\d+)?)s(?:\s|,|$)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        help="directory names under experiments, or explicit directory paths",
    )
    return parser.parse_args()


def resolve_directory(value: str) -> Path:
    """Resolve a directory name relative to ``experiments`` when possible."""

    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()

    under_experiments = EXPERIMENTS_DIR / supplied
    if under_experiments.is_dir():
        return under_experiments.resolve()
    return supplied.resolve()


def find_log_file(directory: Path) -> Path:
    """Return the directory's only directly contained TXT file."""

    if not directory.is_dir():
        raise ValueError(f"experiment directory does not exist: {directory}")

    log_files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix == ".txt"
    )
    if len(log_files) != 1:
        raise ValueError(
            f"expected exactly one .txt file in {directory}, found {len(log_files)}"
        )
    return log_files[0]


def load_rewrite_times(log_file: Path) -> list[Decimal]:
    """Read and validate the 100 per-query rewrite times in a log file."""

    times_by_id: dict[int, Decimal] = {}
    try:
        lines = log_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {log_file}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        match = QUERY_TIME_PATTERN.match(line.strip())
        if match is None:
            continue

        query_id = int(match.group("query_id"))
        query_count = int(match.group("query_count"))
        if query_count != QUERY_COUNT:
            raise ValueError(
                f"unexpected query count {query_count} at {log_file}:{line_number}; "
                f"expected {QUERY_COUNT}"
            )
        if query_id in times_by_id:
            raise ValueError(
                f"duplicate query ID {query_id} at {log_file}:{line_number}"
            )
        times_by_id[query_id] = Decimal(match.group("seconds"))

    expected_ids = set(range(1, QUERY_COUNT + 1))
    actual_ids = set(times_by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing IDs: {', '.join(map(str, missing))}")
        if unexpected:
            details.append(f"unexpected IDs: {', '.join(map(str, unexpected))}")
        raise ValueError(
            f"expected query IDs 1-{QUERY_COUNT} in {log_file}; " + "; ".join(details)
        )

    return [times_by_id[query_id] for query_id in range(1, QUERY_COUNT + 1)]


def summarize_directories(directories: Sequence[Path]) -> None:
    """Print total and mean rewrite time for each experiment directory."""

    for directory in directories:
        log_file = find_log_file(directory)
        rewrite_times = load_rewrite_times(log_file)
        total = sum(rewrite_times, start=Decimal("0"))
        average = total / Decimal(QUERY_COUNT)
        print(
            f"{directory.name}: total={total:.3f}s, "
            f"average={average:.6f}s/query ({log_file.name})"
        )


def main() -> None:
    args = parse_args()
    directories = [resolve_directory(value) for value in args.directories]
    summarize_directories(directories)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
