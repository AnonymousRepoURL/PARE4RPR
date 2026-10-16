"""Build an Oracle or Trino sample table.

For Oracle, the CSV can be a plain file or a ZIP containing one CSV and the
first ``--rows`` data rows are inserted, preserving the original behaviour.
For Trino, the sample is created from the existing source relation with a
``CREATE TABLE AS ... ORDER BY TS LIMIT`` statement. This command only prepares
the sample data; run ``sota_method.py`` or ``our_method.py`` separately.

Examples::

python experiments/build_data_sample.py data/crimes_sql.json \
    --rows 10000 --replace

python experiments/build_data_sample.py data/nasdaq_sql.json \
    --sample-table NASDAQ_SAMPLE --rows 10000 --replace

python experiments/build_data_sample.py data/crimes_sql_trino.json \
    --system trino --rows 10000 --replace
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import sys
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_SYSTEMS = ("oracle", "trino")
TRINO_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_TRINO_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class TrinoQueryMeasurement:
    """Backend-neutral measurement consumed by both rewrite pipelines."""

    completed: bool
    elapsed_seconds: float
    fetched_rows: int


@dataclass(frozen=True)
class _TrinoQueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    measurement: TrinoQueryMeasurement


class TrinoSampleExecutor:
    """Trino Statement API executor with bounded fetch and benchmark calls."""

    def __init__(
        self,
        client,
        *,
        fetch_size: int = 2_000,
        timeout_seconds: float = 5.0,
    ) -> None:
        if fetch_size < 1:
            raise ValueError("fetch_size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.client = client
        self.fetch_size = fetch_size
        self.timeout_seconds = timeout_seconds

    @classmethod
    def connect(
        cls,
        *,
        trino_url: str | None = None,
        user: str = "demo",
        source: str = "rpr-sample",
        catalog: str = "postgresql",
        schema: str = "public",
        fetch_size: int = 2_000,
        timeout_seconds: float = 5.0,
    ) -> "TrinoSampleExecutor":
        from experiments import run_crimes_queries_trino as trino_runner

        url = trino_runner.discover_trino_url(trino_url)
        normalized_catalog = trino_runner.validated_identifier(catalog, "catalog")
        normalized_schema = trino_runner.validated_identifier(schema, "schema")
        client = trino_runner.TrinoClient(
            url, user, source, normalized_catalog, normalized_schema
        )
        return cls(
            client,
            fetch_size=fetch_size,
            timeout_seconds=timeout_seconds,
        )

    @property
    def url(self) -> str:
        return str(self.client.url)

    def _run(
        self,
        sql: str,
        timeout_seconds: float,
        *,
        retain_rows: bool,
    ) -> _TrinoQueryResult:
        from experiments import run_crimes_queries_trino as trino_runner

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        effective_timeout = min(self.timeout_seconds, timeout_seconds)
        deadline = time.perf_counter() + effective_timeout
        started = time.perf_counter()
        columns: tuple[str, ...] = ()
        retained: list[tuple[object, ...]] = []
        fetched_rows = 0
        next_uri: str | None = None
        completed = False
        try:
            request = self.client.initial_request(
                trino_runner.normalize_sql(sql), effective_timeout
            )
            while request is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("sample query exceeded its time limit")
                payload = self.client.read_page(request, remaining)
                returned_error = trino_runner.trino_error(payload)
                if returned_error is not None:
                    if returned_error.name == "EXCEEDED_TIME_LIMIT":
                        raise TimeoutError("sample query exceeded its time limit")
                    raise returned_error
                if payload.get("columns"):
                    columns = tuple(
                        str(column["name"]) for column in payload["columns"]
                    )
                rows = payload.get("data") or []
                fetched_rows += len(rows)
                if retain_rows:
                    retained.extend(tuple(row) for row in rows)
                next_uri = payload.get("nextUri")
                request = self.client.next_request(next_uri) if next_uri else None
            completed = True
        except (TimeoutError, socket.timeout):
            completed = False
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise
            completed = False
        finally:
            if not completed and next_uri:
                self.client.cancel(next_uri)

        return _TrinoQueryResult(
            columns=columns,
            rows=tuple(retained),
            measurement=TrinoQueryMeasurement(
                completed=completed,
                elapsed_seconds=time.perf_counter() - started,
                fetched_rows=fetched_rows,
            ),
        )

    def execute(
        self, sql: str, timeout_seconds: float | None = None
    ) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
        budget = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        result = self._run(sql, budget, retain_rows=True)
        if not result.measurement.completed:
            raise TimeoutError("sample query exceeded its time limit")
        return result.columns, result.rows

    def fetch_one(self, sql: str) -> Mapping[str, object]:
        return self.fetch_one_with_timeout(sql, self.timeout_seconds)

    def fetch_one_with_timeout(
        self, sql: str, timeout_seconds: float
    ) -> Mapping[str, object]:
        result = self._run(sql, timeout_seconds, retain_rows=True)
        if not result.measurement.completed:
            raise TimeoutError("sample query exceeded its time limit")
        if not result.rows:
            raise ValueError("sample query returned no row")
        return dict(zip(result.columns, result.rows[0]))

    def measure_query(self, sql: str, runs: int) -> TrinoQueryMeasurement:
        return self.measure_query_with_timeout(sql, runs, self.timeout_seconds)

    def measure_query_with_timeout(
        self, sql: str, runs: int, timeout_seconds: float
    ) -> TrinoQueryMeasurement:
        if runs < 1:
            raise ValueError("runs must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        measurements: list[TrinoQueryMeasurement] = []
        for _ in range(runs):
            measurement = self._run(sql, timeout_seconds, retain_rows=False).measurement
            measurements.append(measurement)
            if not measurement.completed:
                return measurement
        return TrinoQueryMeasurement(
            completed=True,
            elapsed_seconds=statistics.median(
                item.elapsed_seconds for item in measurements
            ),
            fetched_rows=int(
                statistics.median(item.fetched_rows for item in measurements)
            ),
        )


def _source_table(json_path: Path, requested: str | None) -> str:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schemas = payload.get("table_schemas")
    if not isinstance(schemas, dict) or not schemas:
        raise ValueError(f"table_schemas is missing or empty in {json_path}")
    if requested is not None:
        if not TRINO_IDENTIFIER_RE.fullmatch(requested):
            raise ValueError(f"Invalid source table: {requested!r}")
        table = requested.upper()
        matches = {str(name).upper(): name for name in schemas}
        if table not in matches:
            raise ValueError(
                f"Table {requested!r} is absent from {json_path}; "
                f"available: {', '.join(sorted(schemas))}"
            )
        return str(matches[table])
    if len(schemas) != 1:
        raise ValueError(
            "JSON contains multiple table schemas; select one with --source-table"
        )
    return str(next(iter(schemas)))


def _trino_identifier(value: str, label: str) -> str:
    if not TRINO_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value.lower()


def _trino_name(catalog: str, schema: str, table: str) -> str:
    return ".".join(
        (
            _trino_identifier(catalog, "catalog"),
            _trino_identifier(schema, "schema"),
            _trino_identifier(table, "table"),
        )
    )


def _verify_row_count(connection, owner: str, table: str, expected: int) -> None:
    from experiments.import_crimes_oracle import oracle_identifier

    owner = oracle_identifier(owner)
    table = oracle_identifier(table)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {owner}.{table}")
        actual = int(cursor.fetchone()[0])
    if actual != expected:
        raise RuntimeError(
            f"Expected {expected:,} rows in {owner}.{table}, found {actual:,}"
        )
    print(f"Verified {owner}.{table}: rows={actual:,}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--system",
        choices=SUPPORTED_SYSTEMS,
        default="oracle",
        help="sample-table backend (default: oracle)",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--source-table")
    parser.add_argument("--sample-table")
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help=(
            "sample row count; Oracle loads the first N CSV rows and Trino "
            "uses ORDER BY TS LIMIT N on the source relation"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="drop and recreate the sample table if it already exists",
    )
    parser.add_argument(
        "--schema",
        help="Oracle schema (default: EXPR) or Trino schema (default: public)",
    )
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
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument("--trino-user", default=os.environ.get("TRINO_USER", "demo"))
    parser.add_argument("--trino-source", default="build-data-sample")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TRINO_TIMEOUT_SECONDS)
    return parser


def _build_oracle_sample(args, json_path: Path, source_name: str) -> None:
    from experiments.connect_oracle import (
        OracleConnection,
        password_from_environment_or_prompt,
    )
    from experiments.import_crimes_oracle import (
        add_primary_key,
        create_schema_if_missing,
        load_csv,
        load_table_schema,
        oracle_identifier,
        prepare_table,
    )

    source_table = oracle_identifier(source_name)
    sample_table = oracle_identifier(args.sample_table or f"{source_table}_SAMPLE")
    schema = oracle_identifier(args.schema or "EXPR")
    csv_path = args.csv or PROJECT_ROOT / "data" / f"{source_table.lower()}.csv.zip"
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV/ZIP file not found: {csv_path}")

    table_schema = load_table_schema(json_path, source_table)
    password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.admin_user
    )
    schema_password = os.environ.get("ORACLE_SCHEMA_PASSWORD")
    oracle = OracleConnection(
        args.admin_user, password, args.host, args.port, args.service_name
    )

    with oracle.connect() as connection:
        create_schema_if_missing(connection, schema, schema_password)
        prepare_table(connection, schema, sample_table, table_schema, args.replace)
        loaded = load_csv(
            connection,
            csv_path,
            schema,
            sample_table,
            table_schema,
            args.batch_size,
            args.progress_every,
            args.rows,
        )
        if loaded != args.rows:
            print(f"Input ended after {loaded:,} rows; requested {args.rows:,} rows")
        add_primary_key(
            connection,
            schema,
            sample_table,
            table_schema.get("primary_key", []),
        )
        _verify_row_count(connection, schema, sample_table, loaded)

    print(f"Sample table ready: {schema}.{sample_table} ({loaded:,} rows)")


def _build_trino_sample(args, source_name: str) -> None:
    if args.csv is not None:
        raise ValueError("--csv is only supported with --system oracle")
    catalog = _trino_identifier(args.catalog, "catalog")
    schema = _trino_identifier(args.schema or "public", "schema")
    source_table = _trino_identifier(source_name, "source table")
    sample_table = _trino_identifier(
        args.sample_table or f"{source_table}_sample", "sample table"
    )
    source_relation = _trino_name(catalog, schema, source_table)
    sample_relation = _trino_name(catalog, schema, sample_table)
    executor = TrinoSampleExecutor.connect(
        trino_url=args.trino_url,
        user=args.trino_user,
        source=args.trino_source,
        catalog=catalog,
        schema=schema,
        fetch_size=args.batch_size,
        timeout_seconds=args.timeout,
    )

    existence = executor.fetch_one(
        "SELECT COUNT(*) AS TABLE_COUNT "
        f"FROM {catalog}.information_schema.tables "
        f"WHERE table_schema = '{schema}' AND table_name = '{sample_table}'"
    )
    exists = int(next(iter(existence.values()))) > 0
    if exists and not args.replace:
        raise RuntimeError(
            f"Sample table {sample_relation} already exists; use --replace"
        )
    if exists:
        executor.execute(f"DROP TABLE {sample_relation}")
    executor.execute(
        f"CREATE TABLE {sample_relation} AS "
        f"SELECT * FROM {source_relation} ORDER BY TS LIMIT {args.rows}"
    )
    row = executor.fetch_one(
        f"SELECT COUNT(*) AS SAMPLE_ROW_COUNT FROM {sample_relation}"
    )
    loaded = int(next(iter(row.values())))
    if loaded > args.rows:
        raise RuntimeError(
            f"Expected at most {args.rows:,} rows in {sample_relation}, "
            f"found {loaded:,}"
        )
    if loaded != args.rows:
        print(f"Source ended after {loaded:,} rows; requested {args.rows:,} rows")
    print(
        f"Sample table ready through {executor.url}: "
        f"{sample_relation} ({loaded:,} rows)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.batch_size < 1 or args.progress_every < 1:
        parser.error("--batch-size and --progress-every must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    json_path = args.json_path.expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    source_name = _source_table(json_path, args.source_table)
    if args.system == "oracle":
        _build_oracle_sample(args, json_path, source_name)
    else:
        _build_trino_sample(args, source_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
