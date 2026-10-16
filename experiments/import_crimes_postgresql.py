#!/usr/bin/env python3
"""Load crimes.csv.zip directly into PostgreSQL, then verify it through Trino.

The PostgreSQL ``COPY FROM STDIN`` path is used for fast, streaming ingestion.
If the target table already exists, its row count is shown and the user is
asked whether it should be dropped and rebuilt.

Run command: python experiments\\import_crimes_postgresql.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_JSON = PROJECT_ROOT / "data" / "crimes_sql.json"
DEFAULT_CSV = PROJECT_ROOT / "data" / "crimes.csv.zip"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VARCHAR_RE = re.compile(r"^VARCHAR\((\d+)\)$", re.IGNORECASE)


def identifier(value: str) -> str:
    """Validate and normalize an unquoted PostgreSQL/Trino identifier."""
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return value.lower()


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def postgresql_type(source_type: str) -> str:
    normalized = source_type.strip().upper()
    varchar_match = VARCHAR_RE.fullmatch(normalized)
    if varchar_match:
        length = int(varchar_match.group(1))
        if length < 1:
            raise ValueError(f"Invalid VARCHAR length: {source_type!r}")
        return f"varchar({length})"
    mappings = {
        "INT": "integer",
        "LONG": "bigint",
        "FLOAT": "real",
        "DOUBLE": "double precision",
    }
    try:
        return mappings[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported schema type: {source_type!r}") from exc


def load_table_schema(path: Path, table_name: str) -> dict:
    with path.open("r", encoding="utf-8") as source:
        document = json.load(source)
    schemas = document.get("table_schemas")
    if not isinstance(schemas, dict):
        raise ValueError(f"table_schemas is missing from {path}")

    matching_name = next(
        (name for name in schemas if name.lower() == table_name.lower()), None
    )
    if matching_name is None:
        raise ValueError(
            f"Table {table_name!r} is absent from {path}; "
            f"available: {', '.join(sorted(schemas))}"
        )
    schema = schemas[matching_name]
    columns = schema.get("columns")
    column_types = schema.get("column_types")
    nullables = schema.get("column_nullables")
    if not isinstance(columns, list) or not columns:
        raise ValueError("Schema columns must be a non-empty list")
    if not isinstance(column_types, dict) or not isinstance(nullables, dict):
        raise ValueError("Schema must define column_types and column_nullables")

    normalized_columns = [identifier(column) for column in columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Schema contains duplicate column names")
    missing_types = [column for column in columns if column not in column_types]
    missing_nullables = [column for column in columns if column not in nullables]
    if missing_types or missing_nullables:
        raise ValueError(
            f"Incomplete schema: missing types={missing_types}, "
            f"missing nullability={missing_nullables}"
        )
    for column in columns:
        postgresql_type(column_types[column])
    for column in schema.get("primary_key", []):
        if column not in columns:
            raise ValueError(f"Primary-key column is absent from columns: {column}")
    return schema


def postgresql_name(schema: str, table: str) -> str:
    return f"{identifier(schema)}.{identifier(table)}"


def trino_name(catalog: str, schema: str, table: str) -> str:
    return ".".join(map(identifier, (catalog, schema, table)))


def create_table_ddl(schema: str, table: str, table_schema: dict) -> str:
    definitions = []
    for raw_column in table_schema["columns"]:
        definition = (
            f"    {identifier(raw_column)} "
            f"{postgresql_type(table_schema['column_types'][raw_column])}"
        )
        if not table_schema["column_nullables"][raw_column]:
            definition += " NOT NULL"
        definitions.append(definition)
    return (
        f"CREATE TABLE {postgresql_name(schema, table)} (\n"
        + ",\n".join(definitions)
        + "\n)"
    )


def discover_psql(explicit_path: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    configured = os.environ.get("POSTGRES_PSQL")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "trino-postgres-demo" / "postgresql" / "bin" / "psql.exe",
            PROJECT_ROOT.parent
            / "trino-postgres-demo"
            / "postgresql"
            / "bin"
            / "psql.exe",
        ]
    )
    on_path = shutil.which("psql")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "system PATH"
    raise RuntimeError(f"psql was not found; searched: {searched}")


class PostgreSQLClient:
    """Run SQL and COPY directly against PostgreSQL with the native psql client."""

    def __init__(
        self,
        psql: Path,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> None:
        self.psql = psql
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.environment = os.environ.copy()
        self.environment["PGPASSWORD"] = password

    def command(self, sql: str) -> list[str]:
        return [
            str(self.psql),
            "--no-psqlrc",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--username",
            self.user,
            "--dbname",
            self.database,
            "--set",
            "ON_ERROR_STOP=1",
            "--tuples-only",
            "--no-align",
            "--quiet",
            "--command",
            sql,
        ]

    def execute(self, sql: str) -> str:
        result = subprocess.run(
            self.command(sql),
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"PostgreSQL command failed: {message}")
        return result.stdout.strip()

    def scalar(self, sql: str) -> str:
        output = self.execute(sql)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"Expected one PostgreSQL value, got: {output!r}")
        return lines[0]

    def copy_csv(
        self,
        csv_path: Path,
        schema: str,
        table: str,
        table_schema: dict,
        progress_every: int,
    ) -> int:
        columns = [identifier(column) for column in table_schema["columns"]]
        copy_sql = (
            f"COPY {postgresql_name(schema, table)} ({', '.join(columns)}) "
            "FROM STDIN WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
        )
        process = subprocess.Popen(
            self.command(copy_sql),
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        output_stream = io.TextIOWrapper(
            process.stdin, encoding="utf-8", newline="", write_through=False
        )
        writer = csv.writer(output_stream, lineterminator="\n")
        total = 0
        next_progress = progress_every
        started = time.perf_counter()
        try:
            with open_csv(csv_path) as source:
                reader = csv.reader(source)
                try:
                    raw_header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"CSV is empty: {csv_path}") from exc
                header = [identifier(value) for value in raw_header]
                if header != columns:
                    raise ValueError(
                        f"CSV header {header} does not match schema columns {columns}"
                    )
                writer.writerow(raw_header)

                for line_number, row in enumerate(reader, start=2):
                    validate_csv_row(row, line_number, table_schema)
                    writer.writerow(row)
                    total += 1
                    if total >= next_progress:
                        output_stream.flush()
                        elapsed = time.perf_counter() - started
                        print(
                            f"Sent {total:,} rows to PostgreSQL "
                            f"({total / elapsed:,.0f} rows/s)",
                            flush=True,
                        )
                        next_progress = total + progress_every
            output_stream.flush()
            output_stream.detach().close()
        except BaseException:
            try:
                output_stream.close()
            except (BrokenPipeError, OSError):
                pass
            process.terminate()
            process.wait(timeout=10)
            raise

        stdout = process.stdout.read().decode("utf-8", errors="replace")
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code:
            message = stderr.strip() or stdout.strip()
            raise RuntimeError(f"PostgreSQL COPY failed: {message}")

        elapsed = time.perf_counter() - started
        rate = total / elapsed if elapsed else 0
        print(
            f"PostgreSQL COPY completed: {total:,} rows in {elapsed:,.1f}s "
            f"({rate:,.0f} rows/s)",
            flush=True,
        )
        return total


class TrinoClient:
    """Small dependency-free client for Trino's statement HTTP API."""

    def __init__(self, url: str, user: str, source: str, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "X-Trino-User": user,
            "X-Trino-Source": source,
            "Content-Type": "text/plain; charset=utf-8",
        }
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def query(self, sql: str) -> tuple[list[str], list[list[object]]]:
        request: urllib.request.Request | None = urllib.request.Request(
            f"{self.url}/v1/statement",
            data=sql.encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        columns: list[str] = []
        rows: list[list[object]] = []
        while request is not None:
            with self.http.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
            if "error" in payload:
                error = payload["error"]
                name = error.get("errorName", "TRINO_ERROR")
                raise RuntimeError(f"{name}: {error.get('message', error)}")
            if payload.get("columns"):
                columns = [column["name"] for column in payload["columns"]]
            rows.extend(payload.get("data", []))
            next_uri = payload.get("nextUri")
            request = (
                urllib.request.Request(next_uri, headers=self.headers)
                if next_uri
                else None
            )
        return columns, rows


def discover_trino_url(explicit_url: str | None, timeout: float) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    candidates = ["http://localhost:8080"]
    try:
        output = subprocess.check_output(
            ["wsl.exe", "-d", "Ubuntu", "--", "hostname", "-I"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        addresses = output.strip().split()
        if addresses:
            candidates.append(f"http://{addresses[0]}:8080")
    except (OSError, subprocess.SubprocessError):
        pass

    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for candidate in candidates:
        try:
            with http.open(f"{candidate}/v1/info", timeout=min(timeout, 3.0)):
                return candidate
        except (urllib.error.URLError, TimeoutError):
            continue
    raise RuntimeError(f"Trino is unreachable at: {', '.join(candidates)}")


def trino_row_count(client: TrinoClient, catalog: str, schema: str, table: str) -> int:
    _, rows = client.query(f"SELECT count(*) FROM {trino_name(catalog, schema, table)}")
    if len(rows) != 1 or len(rows[0]) != 1:
        raise RuntimeError(f"Unexpected Trino row-count result: {rows!r}")
    return int(rows[0][0])


@contextmanager
def open_csv(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() != ".zip":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            yield source
        return

    with zipfile.ZipFile(path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if len(csv_members) != 1:
            raise ValueError(
                f"Expected exactly one CSV in {path}, found: {csv_members}"
            )
        with archive.open(csv_members[0], "r") as raw_source:
            with io.TextIOWrapper(
                raw_source, encoding="utf-8-sig", newline=""
            ) as source:
                yield source


def validate_csv_row(row: Sequence[str], line_number: int, table_schema: dict) -> None:
    columns = table_schema["columns"]
    if len(row) != len(columns):
        raise ValueError(
            f"CSV line {line_number} has {len(row)} fields; expected {len(columns)}"
        )
    for column, value in zip(columns, row):
        if not value and not table_schema["column_nullables"][column]:
            raise ValueError(
                f"CSV line {line_number}: non-nullable column {column} is empty"
            )


def table_exists(postgres: PostgreSQLClient, schema: str, table: str) -> bool:
    relation = postgresql_name(schema, table)
    result = postgres.scalar(
        f"SELECT CASE WHEN to_regclass({sql_string(relation)}) IS NULL "
        "THEN 0 ELSE 1 END"
    )
    return result == "1"


def postgresql_row_count(postgres: PostgreSQLClient, schema: str, table: str) -> int:
    return int(
        postgres.scalar(f"SELECT count(*) FROM {postgresql_name(schema, table)}")
    )


def confirm_reimport() -> bool:
    try:
        answer = input(
            "Delete the existing table and re-import it? " "Type Yes to continue: "
        )
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


def add_primary_key(
    postgres: PostgreSQLClient,
    schema: str,
    table: str,
    primary_key: Sequence[str],
) -> None:
    if not primary_key:
        return
    columns = ", ".join(identifier(column) for column in primary_key)
    constraint = identifier(f"{table}_pkey")
    postgres.execute(
        f"ALTER TABLE {postgresql_name(schema, table)} "
        f"ADD CONSTRAINT {constraint} PRIMARY KEY ({columns})"
    )
    print(f"Created primary key {constraint} ({columns}).", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--schema", default=os.environ.get("PGSCHEMA", "public"))
    parser.add_argument("--table", default="crimes")
    parser.add_argument("--psql", type=Path)
    parser.add_argument("--pg-host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument(
        "--pg-port", type=int, default=int(os.environ.get("PGPORT", "54329"))
    )
    parser.add_argument(
        "--pg-database", default=os.environ.get("PGDATABASE", "trino_demo")
    )
    parser.add_argument("--pg-user", default=os.environ.get("PGUSER", "trino"))
    parser.add_argument(
        "--pg-password",
        default=os.environ.get("PGPASSWORD", "trino_demo_password"),
        help="PostgreSQL password (prefer the PGPASSWORD environment variable)",
    )
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument("--trino-user", default=os.environ.get("TRINO_USER", "demo"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pg_port < 1 or args.progress_every < 1 or args.timeout <= 0:
        raise ValueError("pg-port, progress-every, and timeout must be positive")
    catalog = identifier(args.catalog)
    schema = identifier(args.schema)
    table = identifier(args.table)
    table_schema = load_table_schema(args.schema_json, table)
    psql = discover_psql(args.psql)
    postgres = PostgreSQLClient(
        psql,
        args.pg_host,
        args.pg_port,
        args.pg_database,
        args.pg_user,
        args.pg_password,
    )
    pg_target = postgresql_name(schema, table)
    print(
        f"Using PostgreSQL: {args.pg_host}:{args.pg_port}/{args.pg_database} "
        f"(psql: {psql})",
        flush=True,
    )

    if table_exists(postgres, schema, table):
        existing_count = postgresql_row_count(postgres, schema, table)
        print(
            f"Table {pg_target} already exists; rows={existing_count:,}.",
            flush=True,
        )
        if not confirm_reimport():
            print("Re-import cancelled; the existing table was not changed.")
            return 0
        postgres.execute(f"DROP TABLE {pg_target}")
        print(f"Dropped existing table {pg_target}.", flush=True)

    postgres.execute(create_table_ddl(schema, table, table_schema))
    print(f"Created PostgreSQL table {pg_target}.", flush=True)

    loaded = postgres.copy_csv(
        args.csv,
        schema,
        table,
        table_schema,
        args.progress_every,
    )
    add_primary_key(
        postgres,
        schema,
        table,
        table_schema.get("primary_key", []),
    )

    postgres_count = postgresql_row_count(postgres, schema, table)
    if postgres_count != loaded:
        raise RuntimeError(
            f"COPY sent {loaded:,} rows, but PostgreSQL contains {postgres_count:,}"
        )
    print(f"PostgreSQL verified: rows={postgres_count:,}.", flush=True)

    trino_url = discover_trino_url(args.trino_url, args.timeout)
    print(f"Using Trino endpoint for final verification: {trino_url}", flush=True)
    trino = TrinoClient(
        trino_url,
        args.trino_user,
        "verify-crimes-postgresql",
        args.timeout,
    )
    trino_count = trino_row_count(trino, catalog, schema, table)
    if trino_count != loaded:
        raise RuntimeError(
            f"Expected {loaded:,} rows, but Trino reads {trino_count:,} rows"
        )
    print(
        f"Trino verified {trino_name(catalog, schema, table)}: "
        f"rows={trino_count:,}; expected={loaded:,}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
