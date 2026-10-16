"""Run SQL queries from JSON or plain text and stream Oracle results to files.

Run command:
python experiments\run_crimes_queries_oracle.py
python experiments\run_crimes_queries_oracle.py --resume

-----------------run llm generated queries-----------------
python experiments\run_crimes_queries_oracle.py `
  --query-file data\crimes_sql_oracle_gpt-5.6-sol_v1.txt `
  --output-dir experiments\oracle_crimes_gpt5.6_rewriting
-----------------------------------------------------------
JSON input uses the ``sqls`` array. Plain-text input contains semicolon-separated
SQL statements. Query IDs are their 1-based positions. The complete suite is run
three times. A third run is inferred as a timeout without querying Oracle when
both earlier runs timed out. Each result is written once as UTF-8 tab-separated
text to ``oracle_original/<id>.txt``, capped at 5,000 data rows. Only Oracle
execute/fetch calls count toward latency and the 300-second limit; result-file
writing is excluded.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import cx_Oracle

from connect_oracle import OracleConnection, password_from_environment_or_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_FILE = PROJECT_ROOT / "data" / "crimes_sql.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "oracle_original"
DEFAULT_FETCH_SIZE = 2_000
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_RUNS = 3
DEFAULT_MAX_OUTPUT_ROWS = 5_000


class QueryTimeLimitExceeded(Exception):
    """Raised when cumulative Oracle call time reaches the per-query limit."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_sql_statements(text: str) -> list[str]:
    """Split SQL terminators while preserving semicolons in quotes/comments."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    in_line_comment = False
    in_block_comment = False
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            current.append(char)
            if char in "\r\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                index += 2
                in_block_comment = False
            else:
                index += 1
            continue

        if quote is not None:
            current.append(char)
            if char == quote:
                if next_char == quote:
                    current.append(next_char)
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "-" and next_char == "-":
            in_line_comment = True
            current.extend((char, next_char))
            index += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            current.extend((char, next_char))
            index += 2
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    if quote is not None:
        raise ValueError(
            f"SQL text ends inside a quoted string or identifier ({quote})"
        )
    if in_block_comment:
        raise ValueError("SQL text ends inside a block comment")

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def load_queries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".json":
        statements = split_sql_statements(text)
        if not statements:
            raise ValueError(f"No SQL statements found in {path}")
        return [{"sql": statement} for statement in statements]

    document = json.loads(text)
    queries = document.get("sqls")
    if not isinstance(queries, list):
        raise ValueError(f"The sqls array is missing from {path}")
    for query_id, item in enumerate(queries, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
            raise ValueError(f"Query {query_id} does not contain a SQL string")
    return queries


def normalize_sql(sql: str) -> str:
    normalized = sql.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if not normalized:
        raise ValueError("SQL is empty")
    return normalized


def output_value(value: Any) -> Any:
    """Convert values that the csv module cannot serialize usefully."""
    if value is None:
        return r"\N"
    if isinstance(value, cx_Oracle.LOB):
        return value.read()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def load_existing_metrics(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = document.get("queries", [])
    return {
        int(entry["id"]): entry
        for entry in entries
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("id"), int)
            and isinstance(entry.get("runs"), list)
        )
    }


def save_metrics(
    path: Path,
    query_file: Path,
    schema: str,
    fetch_size: int,
    timeout_seconds: float,
    runs_per_query: int,
    max_output_rows: int,
    query_count: int,
    metrics_by_id: dict[int, dict[str, Any]],
) -> None:
    document = {
        "generated_at": utc_now(),
        "query_file": str(query_file.resolve()),
        "schema": schema.upper(),
        "fetch_size": fetch_size,
        "timeout_seconds": timeout_seconds,
        "runs_per_query": runs_per_query,
        "max_output_rows": max_output_rows,
        "query_count": query_count,
        "queries": [metrics_by_id[key] for key in sorted(metrics_by_id)],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_query_once(
    oracle: OracleConnection,
    schema: str,
    query_id: int,
    run_number: int,
    query: dict[str, Any],
    capture_path: Path | None,
    fetch_size: int,
    timeout_seconds: float,
    max_output_rows: int,
) -> dict[str, Any]:
    started_at = utc_now()
    query_elapsed = 0.0
    rows_fetched = 0
    rows_written = 0
    status = "success"
    error_message: str | None = None
    destination = None
    writer = None
    if capture_path is not None:
        destination = capture_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(
            destination,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

    try:
        with oracle.connect() as connection:
            with connection.cursor() as cursor:
                # Connection and session setup are deliberately outside latency.
                cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema}")
                cursor.arraysize = fetch_size
                cursor.prefetchrows = fetch_size

                def database_call(operation: Any) -> Any:
                    """Time one Oracle call, excluding all processing and I/O."""
                    nonlocal query_elapsed
                    remaining = timeout_seconds - query_elapsed
                    if remaining <= 0:
                        raise QueryTimeLimitExceeded
                    # OCI enforces this inside the blocking execute/fetch call.
                    # This is more reliable on Windows than cross-thread cancel().
                    connection.callTimeout = max(1, math.ceil(remaining * 1_000))
                    call_started = time.perf_counter()
                    try:
                        result = operation()
                    except cx_Oracle.DatabaseError as exc:
                        call_elapsed = time.perf_counter() - call_started
                        message = str(exc)
                        if (
                            "DPI-1067" in message
                            or "ORA-01013" in message
                            or call_elapsed >= remaining
                        ):
                            raise QueryTimeLimitExceeded from exc
                        raise
                    finally:
                        query_elapsed += time.perf_counter() - call_started
                    if query_elapsed >= timeout_seconds:
                        raise QueryTimeLimitExceeded
                    return result

                database_call(lambda: cursor.execute(normalize_sql(query["sql"])))
                if writer is not None and cursor.description:
                    writer.writerow(column[0] for column in cursor.description)
                    destination.flush()

                while cursor.description:
                    rows = database_call(lambda: cursor.fetchmany(fetch_size))
                    if not rows:
                        break
                    rows_fetched += len(rows)
                    if writer is not None and rows_written < max_output_rows:
                        # Conversion, serialization and disk I/O are not timed.
                        rows_to_write = rows[: max_output_rows - rows_written]
                        writer.writerows(
                            tuple(output_value(value) for value in row)
                            for row in rows_to_write
                        )
                        rows_written += len(rows_to_write)
                        destination.flush()
    except QueryTimeLimitExceeded:
        status = "timeout"
    except cx_Oracle.DatabaseError as exc:
        status = "error"
        error_message = str(exc).strip()
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        if destination is not None:
            destination.close()

    return {
        "run": run_number,
        "status": status,
        "query_elapsed_seconds": round(query_elapsed, 6),
        "rows_fetched": rows_fetched,
        "started_at": started_at,
        "finished_at": utc_now(),
        "error": error_message,
    }


def inferred_timeout_metric(run_number: int, timeout_seconds: float) -> dict[str, Any]:
    """Build the third-run metric when the first two runs both timed out."""
    timestamp = utc_now()
    return {
        "run": run_number,
        "status": "timeout",
        "query_elapsed_seconds": round(timeout_seconds, 6),
        "rows_fetched": 0,
        "started_at": timestamp,
        "finished_at": timestamp,
        "error": None,
        "inferred": True,
    }


def should_infer_third_timeout(run_number: int, runs: list[dict[str, Any]]) -> bool:
    """Return whether run three can be inferred from two actual timeouts."""
    if run_number != 3:
        return False
    status_by_run = {item.get("run"): item.get("status") for item in runs}
    return status_by_run.get(1) == "timeout" and status_by_run.get(2) == "timeout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=DEFAULT_QUERY_FILE,
        help="JSON file with a sqls array, or semicolon-separated SQL text",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--fetch-size", type=int, default=DEFAULT_FETCH_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=DEFAULT_MAX_OUTPUT_ROWS,
        help="maximum data rows written to each result file (default: 5000)",
    )
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip individual rounds already present in query_times.json",
    )
    return parser.parse_args()


def selected_ids(args: argparse.Namespace, query_count: int) -> Sequence[int]:
    end_id = args.end_id if args.end_id is not None else query_count
    if not 1 <= args.start_id <= query_count:
        raise ValueError(f"start-id must be between 1 and {query_count}")
    if not args.start_id <= end_id <= query_count:
        raise ValueError(
            f"end-id must be between start-id ({args.start_id}) and {query_count}"
        )
    return range(args.start_id, end_id + 1)


def main() -> None:
    args = parse_args()
    if args.fetch_size < 1:
        raise ValueError("fetch-size must be positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.max_output_rows < 1:
        raise ValueError("max-output-rows must be positive")
    schema = args.schema.upper()
    if not schema.isidentifier():
        raise ValueError(f"Invalid schema name: {args.schema!r}")

    queries = load_queries(args.query_file)
    ids = selected_ids(args, len(queries))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "query_times.json"
    metrics_by_id = load_existing_metrics(metrics_path) if args.resume else {}
    admin_password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.admin_user
    )
    oracle = OracleConnection(
        args.admin_user,
        admin_password,
        args.host,
        args.port,
        args.service_name,
    )

    if not args.resume:
        # Remove only files owned by the selected query IDs from the old run.
        for query_id in ids:
            (args.output_dir / f"{query_id}.txt").unlink(missing_ok=True)
            (args.output_dir / f".{query_id}.best_partial.tmp").unlink(missing_ok=True)
            for run_number in range(1, args.runs + 1):
                (args.output_dir / f".{query_id}.run{run_number}.tmp").unlink(
                    missing_ok=True
                )

    for run_number in range(1, args.runs + 1):
        print(f"=== round {run_number}/{args.runs} ===", flush=True)
        for query_id in ids:
            query_metric = metrics_by_id.setdefault(
                query_id,
                {
                    "id": query_id,
                    "url": queries[query_id - 1].get("url"),
                    "output_file": f"{query_id}.txt",
                    "output_complete": False,
                    "output_source_run": None,
                    "best_partial_rows": -1,
                    "runs": [],
                    "average_query_latency_seconds": None,
                },
            )
            # Successful and intentionally timed-out measurements are final.
            # Retrying errors on --resume avoids retaining transient network or
            # client failures as one of the requested three measurements.
            completed_rounds = {
                item["run"]
                for item in query_metric["runs"]
                if item["status"] in {"success", "timeout"}
            }
            if args.resume and run_number in completed_rounds:
                print(
                    f"[round {run_number}/{args.runs}] "
                    f"[{query_id:03d}/{len(queries):03d}] skipped",
                    flush=True,
                )
                continue

            infer_timeout = should_infer_third_timeout(run_number, query_metric["runs"])
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
                    f"[{query_id:03d}/{len(queries):03d}] "
                    "inferred timeout (runs 1 and 2 timed out)",
                    flush=True,
                )
                run_metric = inferred_timeout_metric(run_number, args.timeout)
            else:
                print(
                    f"[round {run_number}/{args.runs}] "
                    f"[{query_id:03d}/{len(queries):03d}] running "
                    f"(capture={'yes' if capture_results else 'no'})",
                    flush=True,
                )
                run_metric = run_query_once(
                    oracle,
                    schema,
                    query_id,
                    run_number,
                    queries[query_id - 1],
                    attempt_path,
                    args.fetch_size,
                    args.timeout,
                    args.max_output_rows,
                )
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
            save_metrics(
                metrics_path,
                args.query_file,
                schema,
                args.fetch_size,
                args.timeout,
                args.runs,
                args.max_output_rows,
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

    print(f"Results: {args.output_dir.resolve()}")
    print(f"Timings: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
