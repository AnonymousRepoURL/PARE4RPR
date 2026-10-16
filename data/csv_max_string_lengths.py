#!/usr/bin/env python3
"""
Calculate the maximum string length of every column in a CSV file.

Run command:
python3 csv_max_string_lengths.py crimes.csv --output column_lengths.csv

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def analyze_csv(
    input_path: Path,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> tuple[list[str], list[int], list[int | None], list[str], int, int]:
    """Return headers, maximum lengths, row numbers, examples, rows, bad rows."""
    with input_path.open("r", encoding=encoding, newline="") as source:
        reader = csv.reader(source, delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {input_path}") from exc

        if not headers:
            raise ValueError(f"CSV file has no columns: {input_path}")

        max_lengths = [0] * len(headers)
        max_row_numbers: list[int | None] = [None] * len(headers)
        examples = [""] * len(headers)
        valid_rows = 0
        malformed_rows = 0

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                malformed_rows += 1
                continue

            valid_rows += 1
            for index, value in enumerate(row):
                value_length = len(value)
                if value_length > max_lengths[index]:
                    max_lengths[index] = value_length
                    max_row_numbers[index] = row_number
                    examples[index] = value

    return (
        headers,
        max_lengths,
        max_row_numbers,
        examples,
        valid_rows,
        malformed_rows,
    )


def shorten(value: str, limit: int = 80) -> str:
    """Make a value safe and compact enough for terminal display."""
    compact = value.replace("\r", "\\r").replace("\n", "\\n")
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def write_report(
    output_path: Path,
    headers: list[str],
    max_lengths: list[int],
    max_row_numbers: list[int | None],
    examples: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["COLUMN_NAME", "MAX_LENGTH", "ROW_NUMBER", "EXAMPLE_VALUE"])
        for header, length, row_number, example in zip(
            headers, max_lengths, max_row_numbers, examples
        ):
            writer.writerow([header, length, row_number or "", example])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV file to analyze")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for the statistics report",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Input encoding (default: utf-8-sig)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="One-character delimiter (default: comma)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if len(arguments.delimiter) != 1:
        raise ValueError("--delimiter must be exactly one character")

    (
        headers,
        max_lengths,
        max_row_numbers,
        examples,
        valid_rows,
        malformed_rows,
    ) = analyze_csv(arguments.input, arguments.encoding, arguments.delimiter)

    print(f"Input: {arguments.input}")
    print(f"Valid data rows: {valid_rows}")
    print(f"Malformed rows skipped: {malformed_rows}")
    print("\nMaximum string length by column:")
    for header, length, row_number, example in zip(
        headers, max_lengths, max_row_numbers, examples
    ):
        row_text = str(row_number) if row_number is not None else "N/A"
        print(
            f"  {header}: {length} " f"(row {row_text}, example: {shorten(example)!r})"
        )

    if arguments.output:
        write_report(
            arguments.output,
            headers,
            max_lengths,
            max_row_numbers,
            examples,
        )
        print(f"\nReport written to: {arguments.output}")


if __name__ == "__main__":
    main()
