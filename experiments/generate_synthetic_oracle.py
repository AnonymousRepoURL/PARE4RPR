"""Generate reproducible synthetic event data and optionally load it into Oracle.

Rows are emitted in event-generation order.  Event ``id=1`` is generated at
2001-09-01 08:00:00 in Europe/Berlin (CEST, UTC+02:00), subsequent events are
generated every 1,000 ms, and an independent integer delay sampled uniformly
from [0, 2,000] ms is
added to form ``ts``.  Consequently, ``ts`` is an arrival timestamp and is
not necessarily monotonic in ``id`` order.

Examples::

# Generate SYN20M and write it to a CSV (no Oracle connection).
python experiments/generate_synthetic_oracle.py --size 20M --csv data/syn20m.csv

# Create EXPR.SYN20M and stream batches directly into Oracle.
python experiments/generate_synthetic_oracle.py --size 20M --insert

# Write CSV and Oracle in the same deterministic pass.
python experiments/generate_synthetic_oracle.py --size 100M \
    --csv data/syn100m.csv --insert --batch-size 10000

# Windows shell
python experiments\generate_synthetic_oracle.py `
  --size 20M `
  --seed 20010901 `
  --batch-size 10000 `
  --insert    

python experiments\generate_synthetic_oracle.py `
  --size 100M `
  --seed 20010901 `
  --batch-size 10000 `
  --insert  

Oracle insertion is enabled only by the explicit ``--insert`` flag.  The
connection settings and password handling match ``connect_oracle.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

BERLIN_TIMEZONE = timezone(timedelta(hours=2), "Europe/Berlin")
START_TIME = datetime(2001, 9, 1, 8, 0, 0, tzinfo=BERLIN_TIMEZONE)
START_TS_MS = int(START_TIME.timestamp() * 1000)
EVENT_INTERVAL_MS = 1_000
MAX_DELAY_MS = 2_000
A1_MIN = 0
A1_MAX = 1_000
A2_MIN = 0.0
A2_MAX = 1_000.0
A2_MEAN = 500.0
A2_STDDEV = 100.0
A3_HEX_LENGTH = 16
REGION_MIN = 1
REGION_MAX = 100
DEFAULT_SEED = 20010901
DEFAULT_BATCH_SIZE = 10_000
MAX_INT_ID = 2_147_483_647
SIZE_RE = re.compile(r"^([1-9][0-9]*)([KMB]?)$", re.IGNORECASE)
IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]*$")

Row = tuple[int, int, int, float, str, int]


def canonical_size_label(count: int) -> str:
    """Return a compact, valid table suffix for an exact event count."""
    for suffix, multiplier in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if count >= multiplier and count % multiplier == 0:
            return f"{count // multiplier}{suffix}"
    return str(count)


def parse_size(value: str) -> tuple[int, str]:
    """Parse values such as 20M and return (20_000_000, '20M')."""
    match = SIZE_RE.fullmatch(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            "size must be a positive integer, optionally followed by K, M, or B"
        )
    magnitude = int(match.group(1))
    unit = match.group(2).upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[unit]
    count = magnitude * multiplier
    if count > MAX_INT_ID:
        raise argparse.ArgumentTypeError(
            f"size exceeds the INT id limit ({MAX_INT_ID:,})"
        )
    return count, canonical_size_label(count)


def oracle_identifier(value: str) -> str:
    """Validate an unquoted Oracle identifier."""
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid Oracle identifier: {value!r}")
    if len(value.encode("ascii")) > 30:
        raise ValueError(f"Oracle identifier is longer than 30 bytes: {value!r}")
    return value.upper()


class SyntheticRows:
    """Stateful, deterministic row generator with independent RNG streams.

    Separate random streams keep output stable when ``batch_size`` changes.
    Python's Mersenne Twister is explicitly seeded, so the same Python program,
    seed, and row count reproduce the same rows.
    """

    def __init__(self, seed: int) -> None:
        self.delay_rng = random.Random(seed ^ 0x243F6A8885A308D3)
        self.a1_rng = random.Random(seed ^ 0x13198A2E03707344)
        self.a2_rng = random.Random(seed ^ 0xA4093822299F31D0)
        self.a3_rng = random.Random(seed ^ 0x082EFA98EC4E6C89)
        self.region_rng = random.Random(seed ^ 0x452821E638D01377)

    def _clipped_normal(self) -> float:
        """Sample the configured normal distribution and clip to its bounds."""
        value = self.a2_rng.gauss(A2_MEAN, A2_STDDEV)
        return min(A2_MAX, max(A2_MIN, value))

    def row(self, event_id: int) -> Row:
        logical_ts = START_TS_MS + (event_id - 1) * EVENT_INTERVAL_MS
        arrival_ts = logical_ts + self.delay_rng.randint(0, MAX_DELAY_MS)
        a1 = self.a1_rng.randint(A1_MIN, A1_MAX)
        a2 = self._clipped_normal()
        # Four bits produce one hexadecimal character, so the value always has
        # exactly A3_HEX_LENGTH characters (including any leading zeroes).
        a3_bits = self.a3_rng.getrandbits(A3_HEX_LENGTH * 4)
        a3 = f"{a3_bits:0{A3_HEX_LENGTH}x}"
        region = self.region_rng.randint(REGION_MIN, REGION_MAX)
        return event_id, arrival_ts, a1, a2, a3, region

    def batches(self, count: int, batch_size: int) -> Iterator[list[Row]]:
        next_id = 1
        while next_id <= count:
            end_id = min(count, next_id + batch_size - 1)
            yield [self.row(event_id) for event_id in range(next_id, end_id + 1)]
            next_id = end_id + 1


@dataclass
class DisorderStats:
    rows: int = 0
    out_of_order_rows: int = 0
    previous_max_ts: int | None = None

    def observe(self, batch: Sequence[Row]) -> None:
        for _, ts, *_ in batch:
            if self.previous_max_ts is not None and ts < self.previous_max_ts:
                self.out_of_order_rows += 1
            if self.previous_max_ts is None or ts > self.previous_max_ts:
                self.previous_max_ts = ts
            self.rows += 1

    @property
    def ratio(self) -> float:
        comparisons = max(0, self.rows - 1)
        return self.out_of_order_rows / comparisons if comparisons else 0.0


def theoretical_disorder_ratio() -> float:
    """Exact ratio for integer-ms uniform delays and a 1,000-ms interval."""
    # A row is late relative to the preceding maximum when
    # D_previous - D_current > EVENT_INTERVAL_MS.  Events two positions back
    # cannot be strictly later because MAX_DELAY_MS == 2 * interval.
    favorable = sum(
        MAX_DELAY_MS - current_delay - EVENT_INTERVAL_MS
        for current_delay in range(MAX_DELAY_MS - EVENT_INTERVAL_MS)
    )
    possibilities = (MAX_DELAY_MS + 1) ** 2
    return favorable / possibilities


def create_table_ddl(owner: str, table_name: str) -> str:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    return f"""CREATE TABLE {owner}.{table_name} (
    ID NUMBER(10, 0) NOT NULL,
    TS NUMBER(19, 0) NOT NULL,
    A1 NUMBER(19, 0) NOT NULL,
    A2 BINARY_DOUBLE NOT NULL,
    A3 VARCHAR2({A3_HEX_LENGTH} CHAR) NOT NULL,
    REGION NUMBER(3, 0) NOT NULL,
    PRIMARY KEY (ID)
)"""


def prepare_oracle_table(
    connection: object, owner: str, table_name: str, replace: bool
) -> None:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        cursor.execute(
            "SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = :1 AND TABLE_NAME = :2",
            (owner, table_name),
        )
        exists = bool(cursor.fetchone()[0])
        if exists and not replace:
            raise RuntimeError(
                f"{owner}.{table_name} already exists; use --replace to rebuild it"
            )
        if exists:
            cursor.execute(f"DROP TABLE {owner}.{table_name} PURGE")
            print(f"Dropped existing table {owner}.{table_name}")
        cursor.execute(create_table_ddl(owner, table_name))
    print(f"Created table {owner}.{table_name}")


def insert_oracle_batch(
    connection: object,
    cursor: object,
    insert_sql: str,
    rows: Sequence[Row],
    first_id: int,
) -> None:
    cursor.executemany(insert_sql, rows, batcherrors=True)  # type: ignore[attr-defined]
    errors = cursor.getbatcherrors()  # type: ignore[attr-defined]
    if errors:
        connection.rollback()  # type: ignore[attr-defined]
        details = "; ".join(
            f"id={first_id + error.offset}: {error.message.strip()}"
            for error in errors[:10]
        )
        raise RuntimeError(f"Oracle rejected a batch: {details}")
    connection.commit()  # type: ignore[attr-defined]


def connect_to_oracle(args: argparse.Namespace) -> tuple[object, object]:
    # Keep Oracle client initialization out of CSV-only and test runs.
    try:
        from connect_oracle import OracleConnection, password_from_environment_or_prompt
    except ModuleNotFoundError:
        from experiments.connect_oracle import (  # type: ignore[no-redef]
            OracleConnection,
            password_from_environment_or_prompt,
        )

    password = password_from_environment_or_prompt(
        "ORACLE_ADMIN_PASSWORD", args.admin_user
    )
    helper = OracleConnection(
        args.admin_user, password, args.host, args.port, args.service_name
    )
    return helper, helper.connect()


def verify_oracle_count(
    connection: object, owner: str, table_name: str, expected_count: int
) -> None:
    owner = oracle_identifier(owner)
    table_name = oracle_identifier(table_name)
    with connection.cursor() as cursor:  # type: ignore[attr-defined]
        cursor.execute(f"SELECT COUNT(*) FROM {owner}.{table_name}")
        actual_count = cursor.fetchone()[0]
    if actual_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count:,} rows, but {owner}.{table_name} has "
            f"{actual_count:,}"
        )
    print(f"Verified {owner}.{table_name}: {actual_count:,} rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        required=True,
        metavar="N[K|M|B]",
        help="event count, for example 20M or 100M",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional CSV output; parent directory must already exist",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="replace an existing --csv file",
    )
    parser.add_argument(
        "--insert",
        action="store_true",
        help="create the derived table and insert generated batches into Oracle",
    )
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
    parser.add_argument(
        "--replace",
        action="store_true",
        help="drop and recreate the Oracle table if it already exists",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the final Oracle COUNT(*) scan",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="print the first N generated rows (default: 5)",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[int, str]:
    try:
        count, size_label = parse_size(args.size)
        oracle_identifier(args.schema)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.progress_every < 1:
        parser.error("progress-every must be positive")
    if args.preview_rows < 0:
        parser.error("preview-rows cannot be negative")
    if args.replace and not args.insert:
        parser.error("--replace requires --insert")
    if args.skip_verify and not args.insert:
        parser.error("--skip-verify requires --insert")
    if args.overwrite_csv and args.csv is None:
        parser.error("--overwrite-csv requires --csv")
    if args.csv is not None:
        if not args.csv.parent.exists():
            parser.error(f"CSV parent directory does not exist: {args.csv.parent}")
        if args.csv.exists() and not args.overwrite_csv:
            parser.error(f"CSV already exists: {args.csv}; use --overwrite-csv")
    return count, size_label


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    count, size_label = validate_args(parser, args)
    table_name = oracle_identifier(f"SYN{size_label}")
    owner = oracle_identifier(args.schema)
    print(
        f"Generating {count:,} rows for {owner}.{table_name}; seed={args.seed}; "
        f"start={START_TIME.isoformat()} [{START_TIME.tzname()}]; "
        f"start_epoch_ms={START_TS_MS:,}"
    )
    if args.csv is None and not args.insert:
        print(
            "No output sink selected: rows will be generated for preview/statistics only"
        )

    stats = DisorderStats()
    generator = SyntheticRows(args.seed)
    started = time.perf_counter()
    next_progress = args.progress_every

    with ExitStack() as stack:
        csv_writer = None
        if args.csv is not None:
            mode = "w" if args.overwrite_csv else "x"
            csv_file = stack.enter_context(
                args.csv.open(mode, encoding="utf-8", newline="")
            )
            csv_writer = csv.writer(csv_file, lineterminator="\n")
            csv_writer.writerow(("id", "ts", "a1", "a2", "a3", "region"))

        connection = None
        oracle_cursor = None
        insert_sql = None
        if args.insert:
            _, connection_context = connect_to_oracle(args)
            connection = stack.enter_context(connection_context)
            prepare_oracle_table(connection, owner, table_name, args.replace)
            oracle_cursor = stack.enter_context(connection.cursor())
            insert_sql = (
                f"INSERT INTO {owner}.{table_name} (ID, TS, A1, A2, A3, REGION) "
                "VALUES (:1, :2, :3, :4, :5, :6)"
            )

        for batch in generator.batches(count, args.batch_size):
            if stats.rows < args.preview_rows:
                remaining = args.preview_rows - stats.rows
                for row in batch[:remaining]:
                    print(f"preview: {row!r}")
            if csv_writer is not None:
                csv_writer.writerows(batch)
            if connection is not None and oracle_cursor is not None and insert_sql:
                insert_oracle_batch(
                    connection, oracle_cursor, insert_sql, batch, batch[0][0]
                )
            stats.observe(batch)
            if stats.rows >= next_progress:
                elapsed = time.perf_counter() - started
                print(
                    f"Generated {stats.rows:,}/{count:,} rows "
                    f"({stats.rows / elapsed:,.0f} rows/s)"
                )
                next_progress = stats.rows + args.progress_every

        if connection is not None and not args.skip_verify:
            verify_oracle_count(connection, owner, table_name, count)

    elapsed = time.perf_counter() - started
    print(
        f"Completed {stats.rows:,} rows in {elapsed:,.1f}s; "
        f"observed out-of-order={stats.out_of_order_rows:,}/{max(0, stats.rows - 1):,} "
        f"({stats.ratio:.4%}); theoretical={theoretical_disorder_ratio():.4%}"
    )


if __name__ == "__main__":
    main()
