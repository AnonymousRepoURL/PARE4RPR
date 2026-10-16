#!/usr/bin/env python3
"""Clean the Nasdaq trades CSV and recommend a type for every column.

SYMBOL: STRING (non-nullable)
TS_ISO: STRING (non-nullable)
TS: LONG (non-nullable)
PRICE: FLOAT (non-nullable)
VOLUME: INT (non-nullable)
EXCHANGE: STRING (non-nullable)
CONDITIONS: STRING (nullable)
TRADE_ID: LONG (non-nullable)
TAPE: STRING (non-nullable)
FEED: STRING (non-nullable)

The file is processed as a stream so that the 1.6 GB source file does not need
to be loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUT = Path(__file__).with_name("nasdaq_trades_20240731_20260731.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("nasdaq.csv")

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1
LONG_MIN = -(2**63)
LONG_MAX = 2**63 - 1
INTEGER_RE = re.compile(r"^[+-]?\d+$")


@dataclass
class ColumnProfile:
    """Incremental statistics used to select one of the allowed data types."""

    null_count: int = 0
    non_null_count: int = 0
    all_integer: bool = True
    all_float: bool = True
    integer_min: int | None = None
    integer_max: int | None = None
    float_requires_double: bool = False

    def add(self, value: str) -> None:
        value = value.strip()
        if value == "":
            self.null_count += 1
            return

        self.non_null_count += 1
        if self.all_integer and INTEGER_RE.fullmatch(value):
            number = int(value)
            self.integer_min = (
                number if self.integer_min is None else min(self.integer_min, number)
            )
            self.integer_max = (
                number if self.integer_max is None else max(self.integer_max, number)
            )
            return

        self.all_integer = False
        if self.all_float:
            try:
                number = float(value)
                if not math.isfinite(number):
                    self.all_float = False
                    return
                # FLOAT is treated as IEEE-754 single precision. Values with
                # more than about seven significant digits are safer as DOUBLE.
                significant_digits = len(
                    value.lower().split("e", 1)[0].replace(".", "").lstrip("+-0")
                )
                if abs(number) > 3.4028235e38 or significant_digits > 7:
                    self.float_requires_double = True
            except ValueError:
                self.all_float = False

    def recommended_type(self) -> str:
        if self.non_null_count == 0:
            return "STRING"
        if self.all_integer:
            assert self.integer_min is not None and self.integer_max is not None
            if INT_MIN <= self.integer_min and self.integer_max <= INT_MAX:
                return "INT"
            if LONG_MIN <= self.integer_min and self.integer_max <= LONG_MAX:
                return "LONG"
            return "STRING"
        if self.all_float:
            return "DOUBLE" if self.float_requires_double else "FLOAT"
        return "STRING"


def normalized_column_name(name: str) -> str:
    """Apply the requested renames, replace spaces, and uppercase the name."""
    renamed = {
        "timestamp_ns": "ts",
        "timestamp": "ts_iso",
        "size": "volume",
    }.get(name.strip().lower(), name.strip())
    return renamed.replace(" ", "_").upper()


def clean_csv(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output files must be different to avoid overwriting the source data."
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)

    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    try:
        with input_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as source, temp_handle:
            reader = csv.reader(source)
            try:
                source_columns = next(reader)
            except StopIteration as exc:
                raise ValueError("The input CSV is empty.") from exc

            lookup = {
                name.strip().lower(): index for index, name in enumerate(source_columns)
            }
            missing_required = [
                name for name in ("symbol", "timestamp_ns") if name not in lookup
            ]
            if missing_required:
                raise ValueError(
                    "The input CSV is missing required columns: "
                    + ", ".join(missing_required)
                )

            output_columns = [normalized_column_name(name) for name in source_columns]
            if len(output_columns) != len(set(output_columns)):
                raise ValueError("Duplicate column names after normalization.")

            symbol_index = lookup["symbol"]
            timestamp_ns_index = lookup["timestamp_ns"]
            profiles = [ColumnProfile() for _ in output_columns]
            writer = csv.writer(temp_handle)
            writer.writerow(output_columns)

            for row_number, row in enumerate(reader, start=2):
                total_rows += 1
                if len(row) != len(source_columns):
                    raise ValueError(
                        f"Row {row_number} has {len(row)} columns, expected {len(source_columns)}."
                    )
                if (
                    row[symbol_index].strip() == ""
                    or row[timestamp_ns_index].strip() == ""
                ):
                    removed_rows += 1
                    continue

                writer.writerow(row)
                kept_rows += 1
                for profile, value in zip(profiles, row):
                    profile.add(value)

            temp_handle.flush()
            os.fsync(temp_handle.fileno())

        os.replace(temp_path, output_path)
        # NamedTemporaryFile starts as owner-only; make the finished CSV use
        # ordinary read permissions like a normal generated data file.
        output_path.chmod(0o644)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print(f"Cleaning complete: {output_path}")
    print(f"Source rows: {total_rows:,}")
    print(f"Rows kept: {kept_rows:,}")
    print(f"Rows removed (SYMBOL or TS was empty): {removed_rows:,}")

    nullable = [
        f"{name} ({profile.null_count:,})"
        for name, profile in zip(output_columns, profiles)
        if profile.null_count
    ]
    print(
        "Columns containing null values: "
        + (", ".join(nullable) if nullable else "None")
    )
    print("Recommended column types:")
    for name, profile in zip(output_columns, profiles):
        print(f"  {name}: {profile.recommended_type()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the Nasdaq trades CSV and recommend column types."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        clean_csv(args.input, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
