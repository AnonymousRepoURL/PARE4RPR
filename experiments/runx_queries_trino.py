#!/usr/bin/env python3
"""Run rewritten SQL in Trino while reusing unchanged original results.

``--query-file`` is a semicolon-separated text file.  Its statements are
matched by position against the ``sqls`` array in ``--original-file``.  When
the two statements have the same canonical project AST, the original Trino
metric and result file are reused.  Parse failures, changed statements, and
statements without an original metric are executed through Trino.

By default, the original result directory is inferred from the original file:
both ``data/crimes_sql.json`` and ``data/crimes_sql_trino.json`` map to
``experiments/trino_crimes_original``.  Use ``--original-results-dir`` when
the results are stored elsewhere.
All relative file and directory arguments are resolved from the project root
using the current operating system's native path rules.

Example:
    python experiments/runx_queries_trino.py \
      --query-file data/crimes_sql_oracle_gpt-5.6-sol_v1_thinking.txt \
      --original-file data/crimes_sql.json \
      --output-dir experiments/trino_crimes_gpt5.6_rewriting
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from data.compare_sql_ast import canonical_ast_sql  # noqa: E402
from experiments import run_crimes_queries_trino as runner  # noqa: E402


def resolve_project_path(path: Path) -> Path:
    """Resolve a CLI path with the current operating system's path rules."""
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_argument_paths(args: argparse.Namespace) -> None:
    """Normalize every file-system argument before it is used."""
    args.query_file = resolve_project_path(args.query_file)
    args.original_file = resolve_project_path(args.original_file)
    args.output_dir = resolve_project_path(args.output_dir)
    if args.original_results_dir is not None:
        args.original_results_dir = resolve_project_path(args.original_results_dir)
    if args.trino_demo_dir is not None:
        args.trino_demo_dir = resolve_project_path(args.trino_demo_dir)


def load_original_queries(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"Original query file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Original query file must contain a JSON object: {path}")
    schemas = payload.get("table_schemas")
    queries = payload.get("sqls")
    if not isinstance(schemas, dict) or not schemas:
        raise ValueError(f"Missing table_schemas in {path}")
    if not isinstance(queries, list):
        raise ValueError(f"Missing sqls array in {path}")
    for query_id, item in enumerate(queries, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
            raise ValueError(f"Original query {query_id} has no SQL string")
    return schemas, queries


def load_text_queries(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".txt":
        raise ValueError(f"--query-file must be a .txt file: {path}")
    if not path.is_file():
        raise ValueError(f"Query file does not exist: {path}")
    statements = runner.split_sql_statements(path.read_text(encoding="utf-8-sig"))
    if not statements:
        raise ValueError(f"No SQL statements found in {path}")
    return [{"sql": sql} for sql in statements]


def inferred_original_results_dir(original_file: Path) -> Path:
    dataset = original_file.stem
    for suffix in ("_sql_trino", "_sql"):
        if dataset.endswith(suffix):
            dataset = dataset[: -len(suffix)]
            break
    return EXPERIMENTS_DIR / f"trino_{dataset}_original"


def load_original_metrics(
    path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"Original metrics file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Original metrics file has no queries array: {path}")
    metrics = {
        entry["id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    }
    return payload, metrics


def find_reusable_queries(
    queries: list[dict[str, Any]],
    originals: list[dict[str, Any]],
    table_schemas: dict[str, Any],
    original_metrics: dict[int, dict[str, Any]],
) -> tuple[set[int], dict[int, str]]:
    """Match the Oracle runx skip/reuse behavior exactly."""
    reusable: set[int] = set()
    reasons: dict[int, str] = {}
    for query_id, query in enumerate(queries, start=1):
        if query_id > len(originals):
            reasons[query_id] = "no corresponding original SQL"
            continue
        try:
            original_ast = canonical_ast_sql(
                originals[query_id - 1]["sql"], table_schemas
            )
            candidate_ast = canonical_ast_sql(query["sql"], table_schemas)
        except Exception as exc:
            reasons[query_id] = f"parse failed: {type(exc).__name__}: {exc}"
            continue
        if original_ast != candidate_ast:
            reasons[query_id] = "parsed SQL differs"
            continue
        if query_id not in original_metrics:
            reasons[query_id] = "original metric is missing"
            continue
        reusable.add(query_id)
    return reusable, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument(
        "--original-file",
        type=Path,
        required=True,
        help="original JSON file containing table_schemas and sqls",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="result directory (relative paths are based on the project root)",
    )
    parser.add_argument(
        "--original-results-dir",
        type=Path,
        help=(
            "directory containing original query_times.json and result files; "
            "inferred as experiments/trino_<dataset>_original by default"
        ),
    )
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument("--trino-user", default=os.environ.get("TRINO_USER", "demo"))
    parser.add_argument("--source", default="runx-queries")
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--schema", default=os.environ.get("TRINO_SCHEMA", "public"))
    parser.add_argument("--fetch-size", type=int, default=runner.DEFAULT_FETCH_SIZE)
    parser.add_argument("--timeout", type=float, default=runner.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--runs", type=int, default=runner.DEFAULT_RUNS)
    parser.add_argument(
        "--max-output-rows", type=int, default=runner.DEFAULT_MAX_OUTPUT_ROWS
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
        "--trino-demo-dir",
        type=Path,
        default=(
            Path(os.environ["TRINO_DEMO_DIR"])
            if os.environ.get("TRINO_DEMO_DIR")
            else None
        ),
    )
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.fetch_size < 1:
        raise ValueError("fetch-size must be positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.max_output_rows < 1:
        raise ValueError("max-output-rows must be positive")
    if args.recovery_poll_seconds <= 0:
        raise ValueError("recovery-poll-seconds must be positive")
    if args.recovery_timeout_seconds <= 0:
        raise ValueError("recovery-timeout-seconds must be positive")
    return (
        runner.validated_identifier(args.catalog, "catalog"),
        runner.validated_identifier(args.schema, "schema"),
    )


def save(
    args: argparse.Namespace,
    trino_url: str,
    catalog: str,
    schema: str,
    query_count: int,
    metrics_by_id: dict[int, dict[str, Any]],
) -> None:
    runner.save_metrics(
        args.output_dir / "query_times.json",
        args.query_file,
        trino_url,
        catalog,
        schema,
        args.trino_user,
        args.fetch_size,
        args.timeout,
        args.runs,
        args.max_output_rows,
        query_count,
        metrics_by_id,
    )


def execute_queries(
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    ids: Sequence[int],
    execute_ids: set[int],
    metrics_by_id: dict[int, dict[str, Any]],
    trino_url: str,
    catalog: str,
    schema: str,
) -> str:
    if not execute_ids:
        return trino_url

    trino_url = runner.discover_or_recover_trino_url(
        args.trino_url,
        args.trino_demo_dir,
        args.recovery_poll_seconds,
        args.recovery_timeout_seconds,
    )
    trino = runner.TrinoClient(trino_url, args.trino_user, args.source, catalog, schema)
    print(
        f"Using Trino: {trino_url} (catalog={catalog}, schema={schema}, "
        f"user={args.trino_user})",
        flush=True,
    )

    for run_number in range(1, args.runs + 1):
        print(f"=== round {run_number}/{args.runs} ===", flush=True)
        for query_id in ids:
            if query_id not in execute_ids:
                continue
            query_metric = metrics_by_id.setdefault(
                query_id,
                {
                    "id": query_id,
                    "url": None,
                    "output_file": f"{query_id}.txt",
                    "output_complete": False,
                    "output_source_run": None,
                    "best_partial_rows": -1,
                    "runs": [],
                    "average_query_latency_seconds": None,
                },
            )
            infer_timeout = runner.should_infer_third_timeout(
                run_number, query_metric["runs"]
            )
            capture_results = not query_metric["output_complete"]
            attempt_path = (
                args.output_dir / f".{query_id}.run{run_number}.tmp"
                if capture_results and not infer_timeout
                else None
            )
            if attempt_path is not None:
                attempt_path.unlink(missing_ok=True)

            if infer_timeout:
                print(
                    f"[round {run_number}/{args.runs}] "
                    f"[{query_id:03d}/{len(queries):03d}] inferred timeout "
                    "(runs 1 and 2 timed out)",
                    flush=True,
                )
                run_metric = runner.inferred_timeout_metric(run_number, args.timeout)
            else:
                print(
                    f"[round {run_number}/{args.runs}] "
                    f"[{query_id:03d}/{len(queries):03d}] running "
                    f"(capture={'yes' if capture_results else 'no'})",
                    flush=True,
                )
                run_metric = runner.run_query_with_coordinator_recovery(
                    trino,
                    query_id,
                    run_number,
                    queries[query_id - 1],
                    attempt_path,
                    args.fetch_size,
                    args.timeout,
                    args.max_output_rows,
                    args.trino_demo_dir,
                    args.recovery_poll_seconds,
                    args.recovery_timeout_seconds,
                )
                trino_url = trino.url

            query_metric["runs"] = [
                item for item in query_metric["runs"] if item["run"] != run_number
            ]
            query_metric["runs"].append(run_metric)
            query_metric["runs"].sort(key=lambda item: item["run"])

            output_path = args.output_dir / f"{query_id}.txt"
            partial_path = args.output_dir / f".{query_id}.best_partial.tmp"
            if attempt_path is not None and run_metric["status"] == "success":
                attempt_path.replace(output_path)
                partial_path.unlink(missing_ok=True)
                query_metric["output_complete"] = True
                query_metric["output_source_run"] = run_number
                query_metric["best_partial_rows"] = run_metric["rows_fetched"]
            elif attempt_path is not None:
                if run_metric["rows_fetched"] > query_metric["best_partial_rows"]:
                    attempt_path.replace(partial_path)
                    query_metric["best_partial_rows"] = run_metric["rows_fetched"]
                    query_metric["output_source_run"] = run_number
                else:
                    attempt_path.unlink(missing_ok=True)
            if run_number == args.runs and not query_metric["output_complete"]:
                if partial_path.exists():
                    partial_path.replace(output_path)
                else:
                    output_path.touch()

            latencies = [item["query_elapsed_seconds"] for item in query_metric["runs"]]
            query_metric["average_query_latency_seconds"] = round(
                sum(latencies) / len(latencies), 6
            )
            save(
                args,
                trino_url,
                catalog,
                schema,
                len(queries),
                metrics_by_id,
            )
            summary = (
                f"status={run_metric['status']}, "
                f"rows={run_metric['rows_fetched']:,}, "
                f"query_time={run_metric['query_elapsed_seconds']:.3f}s, "
                f"average={query_metric['average_query_latency_seconds']:.3f}s"
            )
            if run_metric["error"]:
                summary += f", error={run_metric['error']}"
            print(
                f"[round {run_number}/{args.runs}] "
                f"[{query_id:03d}/{len(queries):03d}] {summary}",
                flush=True,
            )
    return trino_url


def main() -> None:
    args = parse_args()
    resolve_argument_paths(args)
    catalog, schema = validate_args(args)
    queries = load_text_queries(args.query_file)
    table_schemas, originals = load_original_queries(args.original_file)
    original_results_dir = (
        args.original_results_dir
        if args.original_results_dir is not None
        else inferred_original_results_dir(args.original_file)
    )
    original_document, original_metrics = load_original_metrics(
        original_results_dir / "query_times.json"
    )
    ids = tuple(runner.selected_ids(args, len(queries)))
    selected = set(ids)
    reusable, reasons = find_reusable_queries(
        queries, originals, table_schemas, original_metrics
    )
    reusable &= selected
    execute_ids = selected - reusable

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "query_times.json"
    metrics_by_id: dict[int, dict[str, Any]] = {}
    for query_id in ids:
        (args.output_dir / f"{query_id}.txt").unlink(missing_ok=True)
        (args.output_dir / f".{query_id}.best_partial.tmp").unlink(missing_ok=True)
        for run_number in range(1, args.runs + 1):
            (args.output_dir / f".{query_id}.run{run_number}.tmp").unlink(
                missing_ok=True
            )

    for query_id in sorted(reusable):
        metrics_by_id[query_id] = deepcopy(original_metrics[query_id])
        original_output = original_results_dir / f"{query_id}.txt"
        if original_output.is_file():
            shutil.copy2(original_output, args.output_dir / f"{query_id}.txt")
        print(
            f"[{query_id:03d}/{len(queries):03d}] reused original result",
            flush=True,
        )
    for query_id in sorted(execute_ids):
        print(
            f"[{query_id:03d}/{len(queries):03d}] will execute: {reasons[query_id]}",
            flush=True,
        )

    trino_url = str(
        args.trino_url or original_document.get("trino_url") or "unresolved"
    )
    save(args, trino_url, catalog, schema, len(queries), metrics_by_id)
    trino_url = execute_queries(
        args,
        queries,
        ids,
        execute_ids,
        metrics_by_id,
        trino_url,
        catalog,
        schema,
    )
    save(args, trino_url, catalog, schema, len(queries), metrics_by_id)
    print(f"Reused: {len(reusable)}; executed: {len(execute_ids)}")
    print(f"Results: {args.output_dir.resolve()}")
    print(f"Timings: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
