#!/usr/bin/env python3
"""Rerun Trino queries that previously hit a memory limit or timed out.

The script scans ``experiments/trino_crimes_*`` and
``experiments/trino_nasdaq_*`` directories that currently contain a
``query_times.json`` file.  A query is selected when any recorded run contains
``EXCEEDED_LOCAL_MEMORY_LIMIT`` or has ``status=timeout``.  Every selected
query is actually rerun three times and only that query's metrics entry is
replaced.  No third-round timeout is inferred from the first two rounds.

Special SQL handling during reruns (source query files are not modified):

* Nasdaq query IDs 39, 42, 43, 50, 59, 77, 78, and 86 replace
  ``AFTER MATCH SKIP TO NEXT ROW`` with
  ``AFTER MATCH SKIP PAST LAST ROW``.
* Crimes query IDs 15, 23, 63, 67, and 71 apply the same replacement.

The corresponding numeric result file (for example ``42.txt``) is replaced as
well.  The first complete result from the three rounds is preferred; if all
rounds fail or time out, the partial result with the most fetched rows is used.
Before the first metrics update in each directory, a timestamped copy of
``query_times.json`` is created.  Each completed query result and metrics entry
is committed with atomic file replacement, so an interruption does not discard
metrics for queries that were already completed.

Use ``--dry-run`` to print the work list without contacting Trino or changing
any files.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_crimes_queries_trino as runner  # noqa: E402

MEMORY_ERROR = "EXCEEDED_LOCAL_MEMORY_LIMIT"
RERUNS_PER_QUERY = 3
SKIP_TO_NEXT_ROW_RE = re.compile(
    r"\bAFTER\s+MATCH\s+SKIP\s+TO\s+NEXT\s+ROW\b", re.IGNORECASE
)
SPECIAL_SKIP_QUERY_IDS = {
    "nasdaq": frozenset({39, 42, 43, 50, 59, 77, 78, 86}),
    "crimes": frozenset({15, 23, 63, 67, 71}),
}
METHOD_EXECUTION_ORDER = (
    "original",
    "our_rewriting",
    "sota_rewriting",
    "gpt5.6_rewriting",
    "gpt6_rewriting",
    "gpt5.6_fast_rewriting",
    "deepseek_fast_rewriting",
    "deepseek_rewriting",
)


@dataclass(frozen=True)
class RerunJob:
    dataset: str
    output_dir: Path
    metrics_path: Path
    query_file: Path
    query_ids: tuple[int, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_metrics(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("queries"), list):
        raise ValueError(f"Metrics file has no queries array: {path}")
    return document


def entry_needs_rerun(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    runs = entry.get("runs")
    if not isinstance(runs, list):
        return False
    return any(
        isinstance(run, dict)
        and (
            MEMORY_ERROR in str(run.get("error") or "")
            or str(run.get("status") or "").lower() == "timeout"
        )
        for run in runs
    )


def rerun_query_ids(document: dict[str, Any]) -> tuple[int, ...]:
    ids = {
        entry["id"]
        for entry in document["queries"]
        if entry_needs_rerun(entry)
        and isinstance(entry.get("id"), int)
        and entry["id"] > 0
    }
    return tuple(sorted(ids))


def query_for_rerun(
    dataset: str, query_id: int, query: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Return a query copy with the configured MATCH_RECOGNIZE adjustment."""
    adjusted = deepcopy(query)
    if query_id not in SPECIAL_SKIP_QUERY_IDS.get(dataset, frozenset()):
        return adjusted, False

    sql = adjusted.get("sql")
    if not isinstance(sql, str):
        raise ValueError(f"Query {query_id} in {dataset} has no SQL string")
    rewritten, replacements = SKIP_TO_NEXT_ROW_RE.subn(
        "AFTER MATCH SKIP PAST LAST ROW", sql
    )
    if replacements == 0:
        # Some rewritten variants omit the optional clause entirely, whose
        # MATCH_RECOGNIZE default is PAST LAST ROW.  Already-rewritten and
        # omitted forms therefore need no textual change.
        return adjusted, False
    adjusted["sql"] = rewritten
    return adjusted, True


def resolve_query_file(recorded_path: object) -> Path:
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise ValueError("query_file is missing from query_times.json")
    candidate = Path(recorded_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    # Metrics may have been copied from another checkout.  The basename is
    # stable for this project, so try the current data directory as a fallback.
    fallback = PROJECT_ROOT / "data" / candidate.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Recorded query file does not exist: {candidate}; "
        f"fallback does not exist: {fallback}"
    )


def discover_jobs(
    experiments_dir: Path, datasets: Sequence[str]
) -> tuple[list[RerunJob], list[Path]]:
    jobs: list[RerunJob] = []
    missing_metrics: list[Path] = []
    method_priority = {
        method: position for position, method in enumerate(METHOD_EXECUTION_ORDER)
    }
    for dataset in datasets:
        prefix = f"trino_{dataset}_*"
        directory_prefix = f"trino_{dataset}_"
        output_dirs = [path for path in experiments_dir.glob(prefix) if path.is_dir()]
        output_dirs.sort(
            key=lambda path: (
                method_priority.get(
                    path.name.removeprefix(directory_prefix),
                    len(method_priority),
                ),
                path.name,
            )
        )
        for output_dir in output_dirs:
            metrics_path = output_dir / "query_times.json"
            if not metrics_path.is_file():
                missing_metrics.append(output_dir)
                continue
            document = load_metrics(metrics_path)
            query_ids = rerun_query_ids(document)
            if not query_ids:
                continue
            query_file = resolve_query_file(document.get("query_file"))
            queries = runner.load_queries(query_file)
            if query_ids[-1] > len(queries):
                raise ValueError(
                    f"{query_file} has {len(queries)} queries, but metrics require "
                    f"query {query_ids[-1]}"
                )
            for query_id in query_ids:
                if query_id in SPECIAL_SKIP_QUERY_IDS[dataset]:
                    query_for_rerun(dataset, query_id, queries[query_id - 1])
            jobs.append(
                RerunJob(
                    dataset=dataset,
                    output_dir=output_dir.resolve(),
                    metrics_path=metrics_path.resolve(),
                    query_file=query_file,
                    query_ids=query_ids,
                )
            )
    return jobs, missing_metrics


def positive_int(document: dict[str, Any], key: str, default: int) -> int:
    value = document.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def positive_float(document: dict[str, Any], key: str, default: float) -> float:
    value = document.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be positive, got {value!r}")
    return float(value)


def find_query_entry(document: dict[str, Any], query_id: int) -> dict[str, Any]:
    matches = [
        entry
        for entry in document["queries"]
        if isinstance(entry, dict) and entry.get("id") == query_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one metrics entry for query {query_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def rerun_query_entry(
    *,
    trino: runner.TrinoClient,
    query_id: int,
    query: dict[str, Any],
    old_entry: dict[str, Any],
    staged_result_path: Path,
    fetch_size: int,
    timeout_seconds: float,
    max_output_rows: int,
    trino_demo_dir: Path | None,
    recovery_poll_seconds: float,
    recovery_timeout_seconds: float,
    query_count: int,
) -> dict[str, Any]:
    """Rerun three rounds and stage the best result text file for replacement."""
    new_entry = deepcopy(old_entry)
    new_entry["runs"] = []
    new_entry["average_query_latency_seconds"] = None
    new_entry["output_file"] = f"{query_id}.txt"
    new_entry["output_complete"] = False
    new_entry["output_source_run"] = None
    new_entry["best_partial_rows"] = -1
    staged_result_path.unlink(missing_ok=True)

    for run_number in range(1, RERUNS_PER_QUERY + 1):
        capture_results = not new_entry["output_complete"]
        attempt_path = (
            staged_result_path.with_name(
                f".{query_id}.memory_rerun.{os.getpid()}.run{run_number}.tmp"
            )
            if capture_results
            else None
        )
        if attempt_path is not None:
            attempt_path.unlink(missing_ok=True)
        print(
            f"  [run {run_number}/{RERUNS_PER_QUERY}] running "
            f"(capture={'yes' if capture_results else 'no'})",
            flush=True,
        )
        try:
            run_metric = runner.run_query_with_coordinator_recovery(
                trino,
                query_id,
                run_number,
                query,
                attempt_path,
                fetch_size,
                timeout_seconds,
                max_output_rows,
                trino_demo_dir,
                recovery_poll_seconds,
                recovery_timeout_seconds,
            )
        except Exception:
            if attempt_path is not None:
                attempt_path.unlink(missing_ok=True)
            raise

        new_entry["runs"].append(run_metric)
        if attempt_path is not None and run_metric["status"] == "success":
            os.replace(attempt_path, staged_result_path)
            new_entry["output_complete"] = True
            new_entry["output_source_run"] = run_number
            new_entry["best_partial_rows"] = run_metric["rows_fetched"]
        elif attempt_path is not None:
            if run_metric["rows_fetched"] > new_entry["best_partial_rows"]:
                os.replace(attempt_path, staged_result_path)
                new_entry["best_partial_rows"] = run_metric["rows_fetched"]
                new_entry["output_source_run"] = run_number
            else:
                attempt_path.unlink(missing_ok=True)

        latencies = [item["query_elapsed_seconds"] for item in new_entry["runs"]]
        new_entry["average_query_latency_seconds"] = round(
            sum(latencies) / len(latencies), 6
        )
        summary = (
            f"status={run_metric['status']}, "
            f"rows={run_metric['rows_fetched']:,}, "
            f"query_time={run_metric['query_elapsed_seconds']:.3f}s"
        )
        if run_metric.get("error"):
            summary += f", error={run_metric['error']}"
        print(f"  [query {query_id:03d}/{query_count:03d}] {summary}", flush=True)

    if not staged_result_path.exists():
        staged_result_path.touch()
    return new_entry


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_backup(metrics_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = metrics_path.with_name(
        f"{metrics_path.stem}.before_memory_rerun.{timestamp}{metrics_path.suffix}"
    )
    counter = 1
    while backup.exists():
        backup = metrics_path.with_name(
            f"{metrics_path.stem}.before_memory_rerun.{timestamp}.{counter}"
            f"{metrics_path.suffix}"
        )
        counter += 1
    shutil.copy2(metrics_path, backup)
    return backup


def execute_job(job: RerunJob, args: argparse.Namespace) -> None:
    document = load_metrics(job.metrics_path)
    current_ids = rerun_query_ids(document)
    target_ids = tuple(
        query_id for query_id in job.query_ids if query_id in current_ids
    )
    if not target_ids:
        print(f"No remaining memory-limit failures in {job.output_dir.name}.")
        return

    queries = runner.load_queries(job.query_file)
    if target_ids[-1] > len(queries):
        raise ValueError(
            f"{job.query_file} has {len(queries)} queries, but metrics require "
            f"query {target_ids[-1]}"
        )

    catalog = runner.validated_identifier(
        str(document.get("catalog", "postgresql")), "catalog"
    )
    schema = runner.validated_identifier(
        str(document.get("schema", "public")), "schema"
    )
    user = str(document.get("user", "demo"))
    fetch_size = positive_int(document, "fetch_size", runner.DEFAULT_FETCH_SIZE)
    max_output_rows = positive_int(
        document, "max_output_rows", runner.DEFAULT_MAX_OUTPUT_ROWS
    )
    timeout_seconds = positive_float(
        document, "timeout_seconds", runner.DEFAULT_TIMEOUT_SECONDS
    )
    preferred_url = args.trino_url or document.get("trino_url")
    trino_url = runner.discover_or_recover_trino_url(
        str(preferred_url) if preferred_url else None,
        args.trino_demo_dir,
        args.recovery_poll_seconds,
        args.recovery_timeout_seconds,
    )
    trino = runner.TrinoClient(
        trino_url,
        user,
        f"rerun-{job.dataset}-memory-failures",
        catalog,
        schema,
    )
    backup = create_backup(job.metrics_path)
    print(f"Backup: {backup}", flush=True)
    print(
        f"Using Trino: {trino_url} (catalog={catalog}, schema={schema}, user={user})",
        flush=True,
    )

    for position, query_id in enumerate(target_ids, start=1):
        print(
            f"=== {job.output_dir.name}: query {query_id} "
            f"[{position}/{len(target_ids)}] ===",
            flush=True,
        )
        old_entry = find_query_entry(document, query_id)
        query, was_adjusted = query_for_rerun(
            job.dataset, query_id, queries[query_id - 1]
        )
        if was_adjusted:
            print(
                "  SQL adjustment: AFTER MATCH SKIP TO NEXT ROW -> "
                "AFTER MATCH SKIP PAST LAST ROW",
                flush=True,
            )
        staged_result_path = job.output_dir / (
            f".{query_id}.memory_rerun.{os.getpid()}.result.tmp"
        )
        try:
            new_entry = rerun_query_entry(
                trino=trino,
                query_id=query_id,
                query=query,
                old_entry=old_entry,
                staged_result_path=staged_result_path,
                fetch_size=fetch_size,
                timeout_seconds=timeout_seconds,
                max_output_rows=max_output_rows,
                trino_demo_dir=args.trino_demo_dir,
                recovery_poll_seconds=args.recovery_poll_seconds,
                recovery_timeout_seconds=args.recovery_timeout_seconds,
                query_count=len(queries),
            )
        except Exception:
            staged_result_path.unlink(missing_ok=True)
            raise
        entry_index = document["queries"].index(old_entry)
        document["queries"][entry_index] = new_entry
        document["generated_at"] = utc_now()
        document["trino_url"] = trino.url
        document["runs_per_query"] = RERUNS_PER_QUERY
        result_path = job.output_dir / f"{query_id}.txt"
        os.replace(staged_result_path, result_path)
        write_json_atomic(job.metrics_path, document)
        print(
            f"Updated: {result_path} and {job.metrics_path} (query {query_id})",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("crimes", "nasdaq"),
        default=("crimes", "nasdaq"),
    )
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument(
        "--trino-demo-dir",
        type=Path,
        default=(
            Path(os.environ["TRINO_DEMO_DIR"])
            if os.environ.get("TRINO_DEMO_DIR")
            else None
        ),
    )
    parser.add_argument(
        "--recovery-poll-seconds",
        type=float,
        default=runner.DEFAULT_RECOVERY_POLL_SECONDS,
    )
    parser.add_argument(
        "--recovery-timeout-seconds",
        type=float,
        default=runner.DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list matching directories and query IDs without contacting Trino",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.experiments_dir = args.experiments_dir.expanduser().resolve()
    if args.trino_demo_dir is not None:
        args.trino_demo_dir = args.trino_demo_dir.expanduser().resolve()
    if args.recovery_poll_seconds <= 0:
        raise ValueError("recovery-poll-seconds must be positive")
    if args.recovery_timeout_seconds <= 0:
        raise ValueError("recovery-timeout-seconds must be positive")

    jobs, missing_metrics = discover_jobs(args.experiments_dir, args.datasets)
    for directory in missing_metrics:
        print(f"Skipped (query_times.json missing): {directory}")
    if not jobs:
        print("No memory-limit or timeout query runs were found.")
        return

    total_queries = sum(len(job.query_ids) for job in jobs)
    print(f"Found {total_queries} query IDs in {len(jobs)} result directories:")
    for job in jobs:
        ids = ", ".join(str(query_id) for query_id in job.query_ids)
        print(f"  {job.output_dir.name}: {ids}")
        adjusted_ids = sorted(set(job.query_ids) & SPECIAL_SKIP_QUERY_IDS[job.dataset])
        if adjusted_ids:
            print(
                "    special SKIP-clause handling: "
                + ", ".join(str(query_id) for query_id in adjusted_ids)
            )
    if args.dry_run:
        print("Dry run complete; Trino was not contacted and no files were changed.")
        return

    for job_number, job in enumerate(jobs, start=1):
        print(
            f"\n##### Directory {job_number}/{len(jobs)}: {job.output_dir.name} #####"
        )
        execute_job(job, args)
    print("\nAll matching Trino memory-limit and timeout queries were rerun.")


if __name__ == "__main__":
    main()
