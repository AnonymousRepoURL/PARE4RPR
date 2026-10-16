#!/usr/bin/env python3
"""Clean the Chicago crimes CSV and recommend simple column types.

Output:
Columns containing null values (in the original input):
  LOCATION_DESCRIPTION: 16665
  DISTRICT: 47
  WARD: 614811
  COMMUNITY_AREA: 613722
  X_COORDINATE: 100034
  Y_COORDINATE: 100034
  LATITUDE: 100034
  LONGITUDE: 100034
  LOCATION: 100034

Recommended column types (based on cleaned rows):
  ID: INT
  CASE_NUMBER: STRING
  TS: LONG
  BLOCK: STRING
  IUCR: STRING
  PRIMARY_TYPE: STRING
  DESCRIPTION: STRING
  LOCATION_DESCRIPTION: STRING
  ARREST: STRING
  DOMESTIC: STRING
  BEAT: STRING
  DISTRICT: STRING
  WARD: INT
  COMMUNITY_AREA: INT
  FBI_CODE: STRING
  X_COORDINATE: INT
  Y_COORDINATE: INT
  YEAR: INT
  UPDATED_ON: STRING
  LATITUDE: DOUBLE
  LONGITUDE: DOUBLE
  LOCATION: STRING
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_INPUT = Path(__file__).with_name("Crimes_-_2001_to_Present_20260918.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("crimes.csv")
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
CHICAGO_TIMEZONE = ZoneInfo("America/Chicago")
INT_MIN, INT_MAX = -(2**31), 2**31 - 1
LONG_MIN, LONG_MAX = -(2**63), 2**63 - 1
INTEGER_RE = re.compile(r"^[+-]?\d+$")


def normalize_column_name(name: str) -> str:
    """Replace every run of whitespace with one underscore and uppercase it."""
    return re.sub(r"\s+", "_", name.strip()).upper()


def date_to_timestamp_ms(value: str) -> int:
    """Convert a Chicago local date string to a Unix timestamp in milliseconds."""
    parsed = datetime.strptime(value.strip(), DATE_FORMAT)
    aware = parsed.replace(tzinfo=CHICAGO_TIMEZONE)
    return int(aware.timestamp() * 1_000)


class TypeInference:
    """Infer one of INT, LONG, FLOAT, DOUBLE, or STRING."""

    def __init__(self) -> None:
        self.saw_value = False
        self.all_integer = True
        self.all_numeric = True
        self.integer_fits_int = True
        self.integer_fits_long = True
        self.float32_safe = True
        self.has_leading_zero = False

    def add(self, raw_value: str) -> None:
        value = raw_value.strip()
        if not value:
            return

        self.saw_value = True
        if INTEGER_RE.fullmatch(value):
            unsigned = value.lstrip("+-")
            if len(unsigned) > 1 and unsigned.startswith("0"):
                self.has_leading_zero = True
            number = int(value)
            self.integer_fits_int &= INT_MIN <= number <= INT_MAX
            self.integer_fits_long &= LONG_MIN <= number <= LONG_MAX
            return

        self.all_integer = False
        try:
            number = float(value)
        except ValueError:
            self.all_numeric = False
            return

        if not math.isfinite(number):
            self.all_numeric = False
            return

        # IEEE-754 float32 has roughly seven significant decimal digits.
        significant_digits = len(re.sub(r"[^0-9]", "", value.lstrip("+-").lstrip("0")))
        if significant_digits > 7 or abs(number) > 3.4028235e38:
            self.float32_safe = False

    def recommendation(self) -> str:
        if not self.saw_value or not self.all_numeric:
            return "STRING"
        if self.all_integer:
            # Values such as "008" are identifiers/codes, not quantities.
            if self.has_leading_zero:
                return "STRING"
            if self.integer_fits_int:
                return "INT"
            if self.integer_fits_long:
                return "LONG"
            return "STRING"
        return "FLOAT" if self.float32_safe else "DOUBLE"


def clean_csv(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            original_headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input file is empty: {input_path}") from exc

        headers = [normalize_column_name(header) for header in original_headers]
        if len(headers) != len(set(headers)):
            raise ValueError("Normalized column names are not unique")

        required = {"ID", "DATE"}
        missing_required = sorted(required.difference(headers))
        if missing_required:
            raise ValueError(
                "Missing required column(s): " + ", ".join(missing_required)
            )

        id_index = headers.index("ID")
        date_index = headers.index("DATE")
        # The source column is DATE; expose the converted timestamp as TS.
        headers[date_index] = "TS"
        null_counts = {header: 0 for header in headers}
        inferred_types = {header: TypeInference() for header in headers}
        total_rows = kept_rows = dropped_rows = malformed_rows = 0

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(headers)

            for line_number, row in enumerate(reader, start=2):
                total_rows += 1
                if len(row) != len(headers):
                    # The downloaded source can contain a trailing API error body.
                    # Such records cannot be mapped to the CSV schema and are skipped.
                    malformed_rows += 1
                    continue

                for index, header in enumerate(headers):
                    if not row[index].strip():
                        null_counts[header] += 1

                if not row[id_index].strip() or not row[date_index].strip():
                    dropped_rows += 1
                    continue

                try:
                    row[date_index] = str(date_to_timestamp_ms(row[date_index]))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid DATE value on line {line_number}: {row[date_index]!r}"
                    ) from exc

                for index, header in enumerate(headers):
                    inferred_types[header].add(row[index])

                writer.writerow(row)
                kept_rows += 1

    # TS is deliberately converted to epoch milliseconds, so its contract is LONG.
    recommendations = {
        header: ("LONG" if header == "TS" else inferred_types[header].recommendation())
        for header in headers
    }

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows read: {total_rows}")
    print(f"Rows kept: {kept_rows}")
    print(f"Rows dropped because ID or DATE was null: {dropped_rows}")
    print(f"Malformed records skipped: {malformed_rows}")
    print("\nColumns containing null values (in the original input):")
    columns_with_nulls = [
        (header, count) for header, count in null_counts.items() if count > 0
    ]
    if columns_with_nulls:
        for header, count in columns_with_nulls:
            print(f"  {header}: {count}")
    else:
        print("  None")

    print("\nRecommended column types (based on cleaned rows):")
    for header in headers:
        print(f"  {header}: {recommendations[header]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    clean_csv(arguments.input, arguments.output)
