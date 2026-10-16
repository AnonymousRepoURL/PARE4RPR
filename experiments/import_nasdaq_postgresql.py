#!/usr/bin/env python3
"""Load nasdaq.csv.zip directly into PostgreSQL, then verify it through Trino.

The PostgreSQL ``COPY FROM STDIN`` path is used for fast, streaming ingestion.
If the target table already exists, its row count is shown and the user is
asked whether it should be dropped and rebuilt.

Run command: python experiments\\import_nasdaq_postgresql.py
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

import import_crimes_postgresql as importer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_JSON = PROJECT_ROOT / "data" / "nasdaq_sql.json"
DEFAULT_CSV = PROJECT_ROOT / "data" / "nasdaq.csv.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--schema", default=os.environ.get("PGSCHEMA", "public"))
    parser.add_argument("--table", default="nasdaq")
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

    catalog = importer.identifier(args.catalog)
    schema = importer.identifier(args.schema)
    table = importer.identifier(args.table)
    table_schema = importer.load_table_schema(args.schema_json, table)
    psql = importer.discover_psql(args.psql)
    postgres = importer.PostgreSQLClient(
        psql,
        args.pg_host,
        args.pg_port,
        args.pg_database,
        args.pg_user,
        args.pg_password,
    )
    pg_target = importer.postgresql_name(schema, table)
    print(
        f"Using PostgreSQL: {args.pg_host}:{args.pg_port}/{args.pg_database} "
        f"(psql: {psql})",
        flush=True,
    )

    if importer.table_exists(postgres, schema, table):
        existing_count = importer.postgresql_row_count(postgres, schema, table)
        print(
            f"Table {pg_target} already exists; rows={existing_count:,}.",
            flush=True,
        )
        if not importer.confirm_reimport():
            print("Re-import cancelled; the existing table was not changed.")
            return 0
        postgres.execute(f"DROP TABLE {pg_target}")
        print(f"Dropped existing table {pg_target}.", flush=True)

    postgres.execute(importer.create_table_ddl(schema, table, table_schema))
    print(f"Created PostgreSQL table {pg_target}.", flush=True)

    loaded = postgres.copy_csv(
        args.csv,
        schema,
        table,
        table_schema,
        args.progress_every,
    )
    importer.add_primary_key(
        postgres,
        schema,
        table,
        table_schema.get("primary_key", []),
    )

    postgres_count = importer.postgresql_row_count(postgres, schema, table)
    if postgres_count != loaded:
        raise RuntimeError(
            f"COPY sent {loaded:,} rows, but PostgreSQL contains {postgres_count:,}"
        )
    print(f"PostgreSQL verified: rows={postgres_count:,}.", flush=True)

    trino_url = importer.discover_trino_url(args.trino_url, args.timeout)
    print(f"Using Trino endpoint for final verification: {trino_url}", flush=True)
    trino = importer.TrinoClient(
        trino_url,
        args.trino_user,
        "verify-nasdaq-postgresql",
        args.timeout,
    )
    trino_count = importer.trino_row_count(trino, catalog, schema, table)
    if trino_count != loaded:
        raise RuntimeError(
            f"Expected {loaded:,} rows, but Trino reads {trino_count:,} rows"
        )
    print(
        f"Trino verified {importer.trino_name(catalog, schema, table)}: "
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
