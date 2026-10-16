"""Create EXPR.CRIMES from crimes_sql.json and bulk-load the crimes CSV.

The input may be a CSV or a ZIP containing one CSV. ZIP input is streamed, so
the nearly 2 GiB uncompressed CSV does not have to be extracted first.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence, TextIO

import cx_Oracle

from connect_oracle import OracleConnection, password_from_environment_or_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_JSON = PROJECT_ROOT / "data" / "crimes_sql.json"
DEFAULT_CSV = PROJECT_ROOT / "data" / "crimes.csv.zip"
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")
VARCHAR_RE = re.compile(r"^VARCHAR\((\d+)\)$", re.IGNORECASE)


def oracle_identifier(value: str) -> str:
    """Validate and normalize an unquoted Oracle 19c identifier."""
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Oracle identifier: {value!r}")
    if len(value.encode("ascii")) > 30:
        raise ValueError(f"Oracle 19c identifier is longer than 30 bytes: {value!r}")
    return value.upper()


def oracle_type(source_type: str) -> str:
    """Map a generic JSON type to an Oracle 19c type."""
    normalized = source_type.strip().upper()
    varchar_match = VARCHAR_RE.fullmatch(normalized)
    if varchar_match:
        length = int(varchar_match.group(1))
        if not 1 <= length <= 4000:
            raise ValueError(f"Unsupported VARCHAR length: {source_type!r}")
        return f"VARCHAR2({length} CHAR)"
    mappings = {
        "INT": "NUMBER(10, 0)",
        # JSON LONG means int64, not Oracle's legacy LONG type.
        "LONG": "NUMBER(19, 0)",
        "FLOAT": "BINARY_FLOAT",
        "DOUBLE": "BINARY_DOUBLE",
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
    try:
        schema = schemas[table_name]
    except KeyError as exc:
        available = ", ".join(sorted(schemas))
        raise ValueError(
            f"Table {table_name!r} is absent from {path}; available: {available}"
        ) from exc

    columns = schema.get("columns")
    column_types = schema.get("column_types")
    nullables = schema.get("column_nullables")
    if not isinstance(columns, list) or not columns:
        raise ValueError("Schema columns must be a non-empty list")
    if not isinstance(column_types, dict) or not isinstance(nullables, dict):
        raise ValueError("Schema must define column_types and column_nullables")
    normalized_columns = [oracle_identifier(column) for column in columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Schema contains duplicate column names")
    missing_types = [column for column in columns if column not in column_types]
    missing_nullables = [column for column in columns if column not in nullables]
    if missing_types or missing_nullables:
        raise ValueError(
            f"Incomplete schema: missing types={missing_types}, "
            f"missing nullability={missing_nullables}"
        )
    return schema


def create_table_ddl(owner: str, table_name: str, table_schema: dict) -> str:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    definitions = []
    for raw_column in table_schema["columns"]:
        column = oracle_identifier(raw_column)
        definition = (
            f"    {column} {oracle_type(table_schema['column_types'][raw_column])}"
        )
        if not table_schema["column_nullables"][raw_column]:
            definition += " NOT NULL"
        definitions.append(definition)
    return f"CREATE TABLE {owner}.{table_name} (\n" + ",\n".join(definitions) + "\n)"


def quote_password(password: str) -> str:
    if not password or '"' in password or "\n" in password or "\r" in password:
        raise ValueError(
            "Schema password must be non-empty and cannot contain quotes or newlines"
        )
    return f'"{password}"'


def create_schema_if_missing(
    connection: cx_Oracle.Connection,
    schema_name: str,
    schema_password: str | None,
) -> bool:
    schema_name = oracle_identifier(schema_name)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM DBA_USERS WHERE USERNAME = :username",
            username=schema_name,
        )
        if cursor.fetchone()[0]:
            print(f"Schema {schema_name} already exists")
            return False

        authentication = (
            f"IDENTIFIED BY {quote_password(schema_password)}"
            if schema_password
            else "NO AUTHENTICATION"
        )
        cursor.execute(
            f"CREATE USER {schema_name} {authentication} "
            "DEFAULT TABLESPACE USERS TEMPORARY TABLESPACE TEMP "
            "QUOTA UNLIMITED ON USERS"
        )
        if schema_password:
            cursor.execute(f"GRANT CREATE SESSION TO {schema_name}")
        mode = (
            "with login enabled" if schema_password else "without login authentication"
        )
        print(f"Created schema {schema_name} {mode}")
        return True


def table_exists(connection: cx_Oracle.Connection, owner: str, table_name: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM ALL_TABLES
            WHERE OWNER = :owner AND TABLE_NAME = :table_name
            """,
            owner=oracle_identifier(owner),
            table_name=oracle_identifier(table_name),
        )
        return bool(cursor.fetchone()[0])


def prepare_table(
    connection: cx_Oracle.Connection,
    owner: str,
    table_name: str,
    table_schema: dict,
    replace: bool,
) -> None:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    exists = table_exists(connection, owner, table_name)
    with connection.cursor() as cursor:
        if exists:
            if not replace:
                raise RuntimeError(
                    f"{owner}.{table_name} already exists; use --replace to rebuild it"
                )
            print(f"Dropping existing table {owner}.{table_name}")
            cursor.execute(f"DROP TABLE {owner}.{table_name} PURGE")
        cursor.execute(create_table_ddl(owner, table_name, table_schema))
    print(f"Created table {owner}.{table_name}")


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
            ) as text_source:
                yield text_source


def converters_for(table_schema: dict) -> list[Callable[[str], object | None]]:
    converters: list[Callable[[str], object | None]] = []
    for column in table_schema["columns"]:
        normalized = table_schema["column_types"][column].strip().upper()
        if normalized in {"INT", "LONG"}:
            converters.append(lambda value: None if not value.strip() else int(value))
        elif normalized in {"FLOAT", "DOUBLE"}:
            converters.append(lambda value: None if not value.strip() else float(value))
        else:
            converters.append(lambda value: None if not value.strip() else value)
    return converters


def convert_row(
    row: Sequence[str],
    line_number: int,
    table_schema: dict,
    converters: Sequence[Callable[[str], object | None]],
) -> tuple[object | None, ...]:
    columns = table_schema["columns"]
    if len(row) != len(columns):
        raise ValueError(
            f"CSV line {line_number} has {len(row)} fields; expected {len(columns)}"
        )
    try:
        converted = tuple(convert(value) for convert, value in zip(converters, row))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Cannot convert CSV line {line_number}: {exc}") from exc
    for index, value in enumerate(converted):
        column = columns[index]
        if value is None and not table_schema["column_nullables"][column]:
            raise ValueError(
                f"CSV line {line_number}: non-nullable column {column} is empty"
            )
    return converted


def insert_batch(
    connection: cx_Oracle.Connection,
    cursor: cx_Oracle.Cursor,
    insert_sql: str,
    rows: list[tuple[object | None, ...]],
    first_line_number: int,
) -> None:
    cursor.executemany(insert_sql, rows, batcherrors=True)
    errors = cursor.getbatcherrors()
    if errors:
        connection.rollback()
        details = "; ".join(
            f"CSV line {first_line_number + error.offset}: {error.message.strip()}"
            for error in errors[:10]
        )
        raise RuntimeError(f"Oracle rejected a batch: {details}")
    connection.commit()


def load_csv(
    connection: cx_Oracle.Connection,
    csv_path: Path,
    owner: str,
    table_name: str,
    table_schema: dict,
    batch_size: int,
    progress_every: int,
    row_limit: int | None = None,
) -> int:
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be positive")
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    columns = [oracle_identifier(column) for column in table_schema["columns"]]
    placeholders = ", ".join(f":{index}" for index in range(1, len(columns) + 1))
    insert_sql = (
        f"INSERT INTO {owner}.{table_name} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    converters = converters_for(table_schema)
    total = 0
    next_progress = progress_every
    started = time.perf_counter()

    with open_csv(csv_path) as source, connection.cursor() as cursor:
        reader = csv.reader(source)
        try:
            header = [oracle_identifier(value) for value in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc
        if header != columns:
            raise ValueError(
                f"CSV header {header} does not match schema columns {columns}"
            )

        batch: list[tuple[object | None, ...]] = []
        batch_first_line = 2
        for line_number, row in enumerate(reader, start=2):
            if row_limit is not None and total + len(batch) >= row_limit:
                break
            batch.append(convert_row(row, line_number, table_schema, converters))
            if len(batch) >= batch_size:
                insert_batch(connection, cursor, insert_sql, batch, batch_first_line)
                total += len(batch)
                batch.clear()
                batch_first_line = line_number + 1
                if total >= next_progress:
                    elapsed = time.perf_counter() - started
                    print(f"Loaded {total:,} rows ({total / elapsed:,.0f} rows/s)")
                    next_progress = total + progress_every
        if batch:
            insert_batch(connection, cursor, insert_sql, batch, batch_first_line)
            total += len(batch)

    elapsed = time.perf_counter() - started
    print(f"Loaded {total:,} rows in {elapsed:,.1f}s ({total / elapsed:,.0f} rows/s)")
    return total


def add_primary_key(
    connection: cx_Oracle.Connection,
    owner: str,
    table_name: str,
    primary_key: Sequence[str],
) -> None:
    if not primary_key:
        return
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    columns = ", ".join(oracle_identifier(column) for column in primary_key)
    constraint = oracle_identifier(f"{table_name}_PK")
    with connection.cursor() as cursor:
        cursor.execute(
            f"ALTER TABLE {owner}.{table_name} "
            f"ADD CONSTRAINT {constraint} PRIMARY KEY ({columns})"
        )
    print(f"Created primary key {constraint} ({columns})")


def verify_import(
    connection: cx_Oracle.Connection,
    owner: str,
    table_name: str,
    expected_count: int,
) -> None:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*), MIN(ID), MAX(ID), MIN(TS), MAX(TS) "
            f"FROM {owner}.{table_name}"
        )
        count, min_id, max_id, min_ts, max_ts = cursor.fetchone()
    if count != expected_count:
        raise RuntimeError(f"Expected {expected_count:,} rows but Oracle has {count:,}")
    print(
        f"Verified {owner}.{table_name}: rows={count:,}, "
        f"ID=[{min_id}, {max_id}], TS=[{min_ts}, {max_ts}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--schema", default="EXPR")
    parser.add_argument("--table", default="CRIMES")
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
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--row-limit",
        type=int,
        help="load only the first N data rows (the header is not counted)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="drop and recreate the target table if it already exists",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="create the schema/table without loading the CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.progress_every < 1:
        raise ValueError("batch-size and progress-every must be positive")
    if args.row_limit is not None and args.row_limit < 1:
        raise ValueError("row-limit must be positive")
    schema_name = oracle_identifier(args.schema)
    table_name = oracle_identifier(args.table)
    table_schema = load_table_schema(args.schema_json, table_name)
    admin_password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.admin_user
    )
    schema_password = os.environ.get("ORACLE_SCHEMA_PASSWORD")
    oracle = OracleConnection(
        args.admin_user, admin_password, args.host, args.port, args.service_name
    )

    with oracle.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT SYS_CONTEXT('USERENV', 'CON_NAME'), BANNER_FULL "
                "FROM V$VERSION WHERE BANNER_FULL LIKE 'Oracle Database%'"
            )
            container_name, version = cursor.fetchone()
        print(f"Connected to {container_name}: {version.splitlines()[0]}")
        create_schema_if_missing(connection, schema_name, schema_password)
        prepare_table(connection, schema_name, table_name, table_schema, args.replace)
        if args.create_only:
            print("Table creation complete; CSV load skipped (--create-only)")
            return
        loaded_count = load_csv(
            connection,
            args.csv,
            schema_name,
            table_name,
            table_schema,
            args.batch_size,
            args.progress_every,
            args.row_limit,
        )
        add_primary_key(
            connection,
            schema_name,
            table_name,
            table_schema.get("primary_key", []),
        )
        verify_import(connection, schema_name, table_name, loaded_count)


if __name__ == "__main__":
    main()
