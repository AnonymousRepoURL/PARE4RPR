#!/usr/bin/env python3
r"""Run crimes SQL queries through Trino and stream results to files.

Run commands:
python experiments\run_crimes_queries_trino.py
python experiments\run_crimes_queries_trino.py --resume

-----------------run generated queries-----------------
python experiments\run_crimes_queries_trino.py `
  --query-file data\crimes_sql_oracle_gpt-5.6-sol_v1.txt `
  --output-dir experiments\trino_crimes_gpt5.6_rewriting
-------------------------------------------------------
JSON input uses the ``sqls`` array. Plain-text input contains semicolon-separated
SQL statements. Query IDs are their 1-based positions. The complete suite is run
three times. A third run is inferred as a timeout without querying Trino when
both earlier runs timed out. Each result is written once as UTF-8 tab-separated
text to ``trino_original/<id>.txt``, capped at 5,000 data rows. Only Trino HTTP
execute/fetch calls count toward latency and the 300-second limit; result-file
writing is excluded.

Trino config setting:
-Xmx4G
query.max-memory              2GB
query.max-memory-per-node     1GB
memory-revoking-threshold     0.12
memory-revoking-target        0.06
"""

from __future__ import annotations

import argparse
import base64
import csv
import http.client
import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_FILE = PROJECT_ROOT / "data" / "crimes_sql.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "trino_original"
DEFAULT_FETCH_SIZE = 2_000
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_RUNS = 3
DEFAULT_MAX_OUTPUT_ROWS = 5_000
DEFAULT_RECOVERY_POLL_SECONDS = 5.0
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 120.0
DEFAULT_SOURCE = "run-crimes-queries"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QueryTimeLimitExceeded(Exception):
    """Raised when cumulative Trino call time reaches the per-query limit."""


class TrinoQueryError(Exception):
    """A query error returned in a successful Trino HTTP response."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name}: {message}")
        self.name = name


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
    if value is None:
        return r"\N"
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
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
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), int)
        and isinstance(entry.get("runs"), list)
    }


def save_metrics(
    path: Path,
    query_file: Path,
    trino_url: str,
    catalog: str,
    schema: str,
    user: str,
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
        "trino_url": trino_url,
        "catalog": catalog,
        "schema": schema,
        "user": user,
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


class TrinoClient:
    """Minimal Trino Statement API client with no third-party dependency."""

    def __init__(
        self, url: str, user: str, source: str, catalog: str, schema: str
    ) -> None:
        self.set_url(url)
        self.headers = {
            "X-Trino-User": user,
            "X-Trino-Source": source,
            "X-Trino-Catalog": catalog,
            "X-Trino-Schema": schema,
            "Content-Type": "text/plain; charset=utf-8",
        }
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def set_url(self, url: str) -> None:
        self.url = url.rstrip("/")

    def coordinator_uri(self, uri: str) -> str:
        """Route Trino pagination URLs through the currently reachable address."""
        coordinator = urllib.parse.urlsplit(self.url)
        target = urllib.parse.urlsplit(uri)
        return urllib.parse.urlunsplit(
            (
                coordinator.scheme,
                coordinator.netloc,
                target.path,
                target.query,
                target.fragment,
            )
        )

    def initial_request(
        self, sql: str, timeout_seconds: float
    ) -> urllib.request.Request:
        headers = dict(self.headers)
        timeout_ms = max(1, math.ceil(timeout_seconds * 1_000))
        headers["X-Trino-Session"] = (
            f"query_max_execution_time={timeout_ms}ms,"
            "spill_enabled=true,task_concurrency=1"
        )
        return urllib.request.Request(
            f"{self.url}/v1/statement",
            data=sql.encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def next_request(self, uri: str) -> urllib.request.Request:
        return urllib.request.Request(self.coordinator_uri(uri), headers=self.headers)

    def read_page(
        self, request: urllib.request.Request, timeout_seconds: float
    ) -> dict[str, Any]:
        with self.http.open(request, timeout=timeout_seconds) as response:
            return json.load(response)

    def cancel(self, uri: str) -> None:
        request = urllib.request.Request(
            self.coordinator_uri(uri), headers=self.headers, method="DELETE"
        )
        try:
            with self.http.open(request, timeout=3):
                pass
        except (OSError, urllib.error.URLError):
            pass


def trino_error(payload: dict[str, Any]) -> TrinoQueryError | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    return TrinoQueryError(
        str(error.get("errorName", "TRINO_ERROR")),
        str(error.get("message", error)),
    )


def run_query_once(
    trino: TrinoClient,
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
    coordinator_unavailable = False
    next_uri: str | None = None
    destination = None
    writer = None
    if capture_path is not None:
        destination = capture_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(
            destination, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )

    try:
        request: urllib.request.Request | None = trino.initial_request(
            normalize_sql(query["sql"]), timeout_seconds
        )
        wrote_header = False
        while request is not None:
            remaining = timeout_seconds - query_elapsed
            if remaining <= 0:
                raise QueryTimeLimitExceeded
            call_started = time.perf_counter()
            try:
                payload = trino.read_page(request, remaining)
            except (TimeoutError, socket.timeout) as exc:
                raise QueryTimeLimitExceeded from exc
            finally:
                query_elapsed += time.perf_counter() - call_started
            if query_elapsed >= timeout_seconds:
                raise QueryTimeLimitExceeded
            returned_error = trino_error(payload)
            if returned_error is not None:
                if returned_error.name == "EXCEEDED_TIME_LIMIT":
                    raise QueryTimeLimitExceeded from returned_error
                raise returned_error

            columns = payload.get("columns")
            if writer is not None and columns and not wrote_header:
                writer.writerow(column["name"] for column in columns)
                destination.flush()
                wrote_header = True
            rows = payload.get("data") or []
            for offset in range(0, len(rows), fetch_size):
                batch = rows[offset : offset + fetch_size]
                rows_fetched += len(batch)
                if writer is not None and rows_written < max_output_rows:
                    rows_to_write = batch[: max_output_rows - rows_written]
                    writer.writerows(
                        tuple(output_value(value) for value in row)
                        for row in rows_to_write
                    )
                    rows_written += len(rows_to_write)
                    destination.flush()
            next_uri = payload.get("nextUri")
            request = trino.next_request(next_uri) if next_uri else None
    except QueryTimeLimitExceeded:
        status = "timeout"
        if next_uri:
            trino.cancel(next_uri)
    except TrinoQueryError as exc:
        status = "error"
        error_message = str(exc)
    except urllib.error.HTTPError as exc:
        status = "error"
        coordinator_unavailable = exc.code in {502, 503, 504}
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        error_message = f"HTTP {exc.code}: {detail or exc.reason}"
    except urllib.error.URLError as exc:
        coordinator_unavailable = True
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            status = "timeout"
            if next_uri:
                trino.cancel(next_uri)
        else:
            status = "error"
            error_message = f"{type(exc).__name__}: {exc}"
    except (ConnectionError, http.client.HTTPException) as exc:
        status = "error"
        coordinator_unavailable = True
        error_message = f"{type(exc).__name__}: {exc}"
    except (OSError, ValueError) as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
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
        "coordinator_unavailable": coordinator_unavailable,
    }


def inferred_timeout_metric(run_number: int, timeout_seconds: float) -> dict[str, Any]:
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
    if run_number != 3:
        return False
    status_by_run = {item.get("run"): item.get("status") for item in runs}
    return status_by_run.get(1) == "timeout" and status_by_run.get(2) == "timeout"


def completed_run_numbers(runs: Sequence[dict[str, Any]]) -> set[int]:
    """Exclude interrupted coordinator attempts so --resume retries that run."""
    return {
        item["run"]
        for item in runs
        if item.get("status") in {"success", "timeout", "error"}
        and not item.get("coordinator_unavailable", False)
    }


def trino_url_candidates(
    preferred_url: str | None, wsl_timeout: float = 3.0
) -> list[str]:
    candidates: list[str] = []
    if preferred_url:
        candidates.append(preferred_url.rstrip("/"))
    candidates.append("http://localhost:8080")
    try:
        output = subprocess.check_output(
            ["wsl.exe", "-d", "Ubuntu", "--", "hostname", "-I"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=wsl_timeout,
        )
        candidates.extend(
            f"http://{address}:8080" for address in output.strip().split()
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return list(dict.fromkeys(candidates))


def trino_is_ready(trino_url: str, timeout_seconds: float = 3.0) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"{trino_url.rstrip('/')}/v1/info", timeout=timeout_seconds
        ) as response:
            payload = json.load(response)
        return payload.get("state") == "ACTIVE" and not payload.get("starting", False)
    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return False


def first_ready_trino_url(
    candidates: Sequence[str], timeout_seconds: float
) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    for candidate in candidates:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if trino_is_ready(candidate, min(3.0, remaining)):
            return candidate
    return None


def discover_trino_url(explicit_url: str | None) -> str:
    candidates = trino_url_candidates(explicit_url, wsl_timeout=10.0)
    discovered = first_ready_trino_url(candidates, timeout_seconds=10.0)
    if discovered is not None:
        return discovered
    raise RuntimeError(f"Trino is unreachable at: {', '.join(candidates)}")


def resolve_trino_demo_dir(configured_dir: Path | None) -> Path:
    candidates = []
    if configured_dir is not None:
        candidates.append(configured_dir)
    candidates.extend(
        [
            PROJECT_ROOT.parent / "trino-postgres-demo",
            PROJECT_ROOT / "trino-postgres-demo",
        ]
    )
    for candidate in candidates:
        if all(
            (candidate / script).is_file()
            for script in ("stop_trino.ps1", "start_trino.ps1")
        ):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Trino restart scripts were not found in: {searched}")


def restart_trino(configured_dir: Path | None, deadline: float) -> None:
    demo_dir = resolve_trino_demo_dir(configured_dir)
    for script_name in ("stop_trino.ps1", "start_trino.ps1"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("recovery deadline reached while restarting Trino")
        script = demo_dir / script_name
        command_timeout = (
            min(35.0, remaining) if script_name.startswith("stop") else remaining
        )
        try:
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                cwd=demo_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if script_name.startswith("stop"):
                print("Trino stop timed out; attempting startup anyway.", flush=True)
                continue
            raise
        if completed.stdout.strip():
            print(completed.stdout.strip(), flush=True)
        if completed.returncode != 0:
            if script_name.startswith("stop"):
                print(
                    f"Trino stop exited with code {completed.returncode}; "
                    "attempting startup anyway.",
                    flush=True,
                )
                continue
            raise RuntimeError(f"{script_name} exited with code {completed.returncode}")


def recover_trino_url(
    previous_url: str,
    configured_demo_dir: Path | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> str:
    """Restart Trino, then rediscover a reachable URL before the deadline."""
    deadline = time.monotonic() + timeout_seconds
    restart_error: Exception | None = None
    try:
        restart_trino(configured_demo_dir, deadline)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        restart_error = exc
        print(f"Automatic Trino restart failed: {exc}", flush=True)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f" Restart error: {restart_error}" if restart_error else ""
            raise TimeoutError(
                f"Trino did not recover within {timeout_seconds:g} seconds.{detail}"
            )
        wsl_budget = min(2.0, remaining)
        candidates = trino_url_candidates(previous_url, wsl_timeout=wsl_budget)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        probe_budget = min(3.0 * len(candidates), remaining)
        recovered_url = first_ready_trino_url(candidates, probe_budget)
        if recovered_url is not None:
            print(f"Trino recovered at {recovered_url}.", flush=True)
            return recovered_url
        print(
            f"Trino is still unavailable; retrying in "
            f"{min(poll_seconds, remaining):g}s...",
            flush=True,
        )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def discover_or_recover_trino_url(
    explicit_url: str | None,
    configured_demo_dir: Path | None,
    poll_seconds: float,
    timeout_seconds: float,
) -> str:
    try:
        return discover_trino_url(explicit_url)
    except RuntimeError as exc:
        previous_url = explicit_url or "http://localhost:8080"
        print(
            f"Trino is unavailable during startup ({exc}); attempting recovery.",
            flush=True,
        )
        return recover_trino_url(
            previous_url,
            configured_demo_dir,
            poll_seconds,
            timeout_seconds,
        )


def run_query_with_coordinator_recovery(
    trino: TrinoClient,
    query_id: int,
    run_number: int,
    query: dict[str, Any],
    capture_path: Path | None,
    fetch_size: int,
    timeout_seconds: float,
    max_output_rows: int,
    configured_demo_dir: Path | None,
    recovery_poll_seconds: float,
    recovery_timeout_seconds: float,
) -> dict[str, Any]:
    recovery_count = 0
    while True:
        run_metric = run_query_once(
            trino,
            query_id,
            run_number,
            query,
            capture_path,
            fetch_size,
            timeout_seconds,
            max_output_rows,
        )
        if not run_metric["coordinator_unavailable"]:
            return run_metric

        recovery_count += 1
        if capture_path is not None:
            capture_path.unlink(missing_ok=True)
        print(
            f"Coordinator became unavailable during query {query_id}, "
            f"run {run_number}; recovering before retry {recovery_count}.",
            flush=True,
        )
        recovered_url = recover_trino_url(
            trino.url,
            configured_demo_dir,
            recovery_poll_seconds,
            recovery_timeout_seconds,
        )
        trino.set_url(recovered_url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=DEFAULT_QUERY_FILE,
        help="JSON file with a sqls array, or semicolon-separated SQL text",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument("--trino-user", default=os.environ.get("TRINO_USER", "demo"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--schema", default=os.environ.get("TRINO_SCHEMA", "public"))
    parser.add_argument("--fetch-size", type=int, default=DEFAULT_FETCH_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--recovery-poll-seconds",
        type=float,
        default=DEFAULT_RECOVERY_POLL_SECONDS,
        help="seconds between Trino health checks while it is restarting",
    )
    parser.add_argument(
        "--recovery-timeout-seconds",
        type=float,
        default=DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        help="maximum seconds to restart and rediscover Trino (default: 120)",
    )
    parser.add_argument(
        "--trino-demo-dir",
        type=Path,
        default=(
            Path(os.environ["TRINO_DEMO_DIR"])
            if os.environ.get("TRINO_DEMO_DIR")
            else None
        ),
        help="directory containing stop_trino.ps1 and start_trino.ps1",
    )
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
        help="skip completed rounds; retry coordinator-interrupted rounds",
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


def validated_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value.lower()


def main() -> None:
    args = parse_args()
    if args.fetch_size < 1:
        raise ValueError("fetch-size must be positive")
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.recovery_poll_seconds <= 0:
        raise ValueError("recovery-poll-seconds must be positive")
    if args.recovery_timeout_seconds <= 0:
        raise ValueError("recovery-timeout-seconds must be positive")
    if args.max_output_rows < 1:
        raise ValueError("max-output-rows must be positive")
    catalog = validated_identifier(args.catalog, "catalog")
    schema = validated_identifier(args.schema, "schema")
    queries = load_queries(args.query_file)
    ids = selected_ids(args, len(queries))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "query_times.json"
    metrics_by_id = load_existing_metrics(metrics_path) if args.resume else {}
    trino_url = discover_or_recover_trino_url(
        args.trino_url,
        args.trino_demo_dir,
        args.recovery_poll_seconds,
        args.recovery_timeout_seconds,
    )
    trino = TrinoClient(trino_url, args.trino_user, args.source, catalog, schema)
    print(
        f"Using Trino: {trino_url} (catalog={catalog}, schema={schema}, "
        f"user={args.trino_user})",
        flush=True,
    )

    if not args.resume:
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
            completed_rounds = completed_run_numbers(query_metric["runs"])
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
                run_metric = run_query_with_coordinator_recovery(
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
            save_metrics(
                metrics_path,
                args.query_file,
                trino_url,
                catalog,
                schema,
                args.trino_user,
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
