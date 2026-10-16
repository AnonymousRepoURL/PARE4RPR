"""Rewrite and benchmark the synthetic RPR workload on Oracle.

The workflow uses one shared sample table for both SYN10M and SYN50M.  The
sample contains IDs 1 through 10,000 from SYN10M; before any rewrite starts,
the script verifies that the same prefix exists byte-for-byte in SYN50M.

The default workflow is:

1. create or validate EXPR.SYN_SAMPLE_10K;
2. rewrite both workloads with sota_method.py and our_method.py;
3. execute the original, SOTA, and our-method SQL three times each.

For each method, a SYN10M query that timed out in all configured benchmark
runs is not executed on SYN50M.  Its SYN50M runs are recorded as inferred
300-second timeouts instead.

No work is performed merely by importing this module.  Use ``--dry-run`` to
inspect the child commands without connecting to Oracle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")
SAMPLE_COLUMNS = ("ID", "TS", "A1", "A2", "A3", "REGION")


@dataclass(frozen=True)
class Dataset:
    size: str
    table: str
    query_file: Path

    @property
    def label(self) -> str:
        return self.size.lower()


DATASETS = (
    Dataset("10M", "SYN10M", DATA_DIR / "synthetic_sql_10M.json"),
    Dataset("50M", "SYN50M", DATA_DIR / "synthetic_sql_50M.json"),
)


def oracle_identifier(value: str, label: str) -> str:
    normalized = value.upper()
    if not IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"Invalid Oracle {label}: {value!r}")
    return normalized


def _qualified(schema: str, table: str) -> str:
    return f"{oracle_identifier(schema, 'schema')}.{oracle_identifier(table, 'table')}"


def validate_workload(dataset: Dataset) -> None:
    if not dataset.query_file.is_file():
        raise FileNotFoundError(f"Query file not found: {dataset.query_file}")
    payload = json.loads(dataset.query_file.read_text(encoding="utf-8"))
    schemas = payload.get("table_schemas")
    if not isinstance(schemas, dict) or set(schemas) != {dataset.table}:
        raise ValueError(
            f"{dataset.query_file} must describe only table {dataset.table}"
        )
    queries = payload.get("sqls")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"No sqls array found in {dataset.query_file}")


def _count(connection: object, sql: str) -> int:
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        cursor.execute(sql)
        return int(cursor.fetchone()[0])


def _difference_count(
    connection: object,
    left_relation: str,
    right_relation: str,
    rows: int,
    *,
    right_is_sample: bool = False,
) -> int:
    columns = ", ".join(SAMPLE_COLUMNS)
    left_filter = f"WHERE ID BETWEEN 1 AND {rows}"
    right_filter = "" if right_is_sample else f"WHERE ID BETWEEN 1 AND {rows}"
    return _count(
        connection,
        "SELECT COUNT(*) FROM ("
        f"SELECT {columns} FROM {left_relation} {left_filter} "
        "MINUS "
        f"SELECT {columns} FROM {right_relation} {right_filter}"
        ")",
    )


def _assert_prefix_equal(
    connection: object, left_relation: str, right_relation: str, rows: int
) -> None:
    left_count = _count(
        connection,
        f"SELECT COUNT(*) FROM {left_relation} WHERE ID BETWEEN 1 AND {rows}",
    )
    right_count = _count(
        connection,
        f"SELECT COUNT(*) FROM {right_relation} WHERE ID BETWEEN 1 AND {rows}",
    )
    if left_count != rows or right_count != rows:
        raise RuntimeError(
            f"Expected IDs 1..{rows} in both source tables; found "
            f"{left_count} in {left_relation} and {right_count} in {right_relation}"
        )
    left_only = _difference_count(connection, left_relation, right_relation, rows)
    right_only = _difference_count(connection, right_relation, left_relation, rows)
    if left_only or right_only:
        raise RuntimeError(
            f"The first {rows} rows differ between {left_relation} and "
            f"{right_relation}: left_only={left_only}, right_only={right_only}"
        )


def _sample_exists(connection: object, schema: str, table: str) -> bool:
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        cursor.execute(
            "SELECT COUNT(*) FROM ALL_TABLES " "WHERE OWNER = :1 AND TABLE_NAME = :2",
            (schema, table),
        )
        return bool(cursor.fetchone()[0])


def prepare_shared_sample(
    connection: object,
    schema: str,
    sample_table: str,
    sample_rows: int,
    replace: bool,
) -> None:
    schema = oracle_identifier(schema, "schema")
    sample_table = oracle_identifier(sample_table, "sample table")
    source_10m = _qualified(schema, "SYN10M")
    source_50m = _qualified(schema, "SYN50M")
    sample_relation = _qualified(schema, sample_table)

    _assert_prefix_equal(connection, source_10m, source_50m, sample_rows)
    exists = _sample_exists(connection, schema, sample_table)
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        if exists and replace:
            cursor.execute(f"DROP TABLE {sample_relation} PURGE")
            exists = False
            print(f"Dropped existing sample table {sample_relation}", flush=True)
        if not exists:
            columns = ", ".join(SAMPLE_COLUMNS)
            cursor.execute(
                f"CREATE TABLE {sample_relation} AS "
                f"SELECT {columns} FROM {source_10m} "
                f"WHERE ID BETWEEN 1 AND {sample_rows} ORDER BY ID"
            )
            cursor.execute(f"ALTER TABLE {sample_relation} ADD PRIMARY KEY (ID)")
            connection.commit()  # type: ignore[attr-defined]
            print(f"Created sample table {sample_relation}", flush=True)

    sample_count = _count(connection, f"SELECT COUNT(*) FROM {sample_relation}")
    source_only = _difference_count(
        connection, source_10m, sample_relation, sample_rows, right_is_sample=True
    )
    sample_only = _difference_count(
        connection, sample_relation, source_10m, sample_rows
    )
    if sample_count != sample_rows or source_only or sample_only:
        raise RuntimeError(
            f"Invalid sample {sample_relation}: rows={sample_count}, "
            f"source_only={source_only}, sample_only={sample_only}"
        )
    print(
        f"Verified shared sample {sample_relation}: IDs 1..{sample_rows} "
        "match both SYN10M and SYN50M",
        flush=True,
    )


def rewritten_path(dataset: Dataset, method: str) -> Path:
    return DATA_DIR / f"synthetic_sql_{dataset.size}_{method}.txt"


def metrics_path(dataset: Dataset, method: str) -> Path:
    return DATA_DIR / f"synthetic_sql_{dataset.size}_{method}.metrics.json"


def result_dir(dataset: Dataset, method: str) -> Path:
    return EXPERIMENTS_DIR / f"oracle_synthetic_{dataset.label}_{method}"


def connection_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--schema",
        args.schema,
        "--admin-user",
        args.admin_user,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--service-name",
        args.service_name,
    ]


def rewrite_commands(args: argparse.Namespace, dataset: Dataset) -> list[list[str]]:
    common = [
        "--sample-table",
        args.sample_table,
        "--runs",
        str(args.rewrite_runs),
        "--timeout",
        str(args.rewrite_timeout),
        "--fetch-size",
        str(args.fetch_size),
        *connection_arguments(args),
    ]
    sota = [
        sys.executable,
        str(PROJECT_ROOT / "sota_method.py"),
        str(dataset.query_file),
        *common,
        "--output",
        str(rewritten_path(dataset, "sota_method")),
        "--metrics-output",
        str(metrics_path(dataset, "sota_method")),
    ]
    ours = [
        sys.executable,
        str(PROJECT_ROOT / "our_method.py"),
        str(dataset.query_file),
        "--sample-must-appear-partition-filter",
        "--sample-window-condition-prefilter",
        *common,
        "--output",
        str(rewritten_path(dataset, "our_method")),
        "--metrics-output",
        str(metrics_path(dataset, "our_method")),
    ]
    return [sota, ours]


def benchmark_command(
    args: argparse.Namespace, dataset: Dataset, method: str
) -> list[str]:
    query_file = (
        dataset.query_file if method == "original" else rewritten_path(dataset, method)
    )
    command = [
        sys.executable,
        str(EXPERIMENTS_DIR / "run_crimes_queries_oracle.py"),
        "--query-file",
        str(query_file),
        "--output-dir",
        str(result_dir(dataset, method)),
        "--runs",
        str(args.query_runs),
        "--timeout",
        str(args.query_timeout),
        "--fetch-size",
        str(args.fetch_size),
        "--max-output-rows",
        str(args.max_output_rows),
        "--start-id",
        str(args.start_id),
        *connection_arguments(args),
    ]
    if args.end_id is not None:
        command.extend(("--end-id", str(args.end_id)))
    if dataset != DATASETS[0]:
        command.extend(
            (
                "--infer-timeouts-from",
                str(result_dir(DATASETS[0], method) / "query_times.json"),
            )
        )
    if args.resume:
        command.append("--resume")
    return command


def command_plan(args: argparse.Namespace) -> list[list[str]]:
    plan: list[list[str]] = []
    for dataset in DATASETS:
        if not args.skip_rewrite:
            plan.extend(rewrite_commands(args, dataset))
        if not args.skip_benchmark:
            for method in ("original", "sota_method", "our_method"):
                plan.append(benchmark_command(args, dataset, method))
    return plan


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="EXPR")
    parser.add_argument(
        "--admin-user", default=os.environ.get("ORACLE_ADMIN_USER", "system")
    )
    parser.add_argument("--host", default=os.environ.get("ORACLE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ORACLE_PORT", "1521"))
    )
    parser.add_argument(
        "--service-name", default=os.environ.get("ORACLE_SERVICE_NAME", "ORCL")
    )
    parser.add_argument("--sample-table", default="SYN_SAMPLE_10K")
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument(
        "--replace-sample",
        action="store_true",
        help="drop and recreate the shared sample table",
    )
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help="do not create or validate the shared sample table",
    )
    parser.add_argument("--skip-rewrite", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--rewrite-runs", type=int, default=3)
    parser.add_argument("--rewrite-timeout", type=float, default=5.0)
    parser.add_argument("--query-runs", type=int, default=3)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--fetch-size", type=int, default=2_000)
    parser.add_argument("--max-output-rows", type=int, default=5_000)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed benchmark rounds in each result directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local inputs and print commands without Oracle access",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        args.schema = oracle_identifier(args.schema, "schema")
        args.sample_table = oracle_identifier(args.sample_table, "sample table")
    except ValueError as exc:
        parser.error(str(exc))
    positive_values = {
        "sample-rows": args.sample_rows,
        "rewrite-runs": args.rewrite_runs,
        "rewrite-timeout": args.rewrite_timeout,
        "query-runs": args.query_runs,
        "query-timeout": args.query_timeout,
        "fetch-size": args.fetch_size,
        "max-output-rows": args.max_output_rows,
        "start-id": args.start_id,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.end_id is not None and args.end_id < args.start_id:
        parser.error("--end-id must be at least --start-id")
    if args.replace_sample and args.skip_sample:
        parser.error("--replace-sample cannot be combined with --skip-sample")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    validate_args(parser, args)
    for dataset in DATASETS:
        validate_workload(dataset)

    plan = command_plan(args)
    if args.dry_run:
        if not args.skip_sample:
            print(
                f"[dry-run] prepare/validate {args.schema}.{args.sample_table} "
                f"from IDs 1..{args.sample_rows}"
            )
        for command in plan:
            print(f"[dry-run] {_display_command(command)}")
        return 0

    from experiments.connect_oracle import (
        OracleConnection,
        password_from_environment_or_prompt,
    )

    password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.admin_user
    )
    child_environment = os.environ.copy()
    child_environment["ORACLE_ADMIN_PASSWORD"] = password

    if not args.skip_sample:
        oracle = OracleConnection(
            args.admin_user, password, args.host, args.port, args.service_name
        )
        with oracle.connect() as connection:
            prepare_shared_sample(
                connection,
                args.schema,
                args.sample_table,
                args.sample_rows,
                args.replace_sample,
            )

    for index, command in enumerate(plan, start=1):
        print(
            f"\n=== synthetic workflow step {index}/{len(plan)} ===\n"
            f"{_display_command(command)}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment,
            check=True,
        )
    print("Synthetic Oracle workflow completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
