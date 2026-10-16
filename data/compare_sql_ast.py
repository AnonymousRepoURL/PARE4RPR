#!/usr/bin/env python3
"""Compare source JSON queries with a text file by their parsed RPR ASTs.

Usage:
    python3 data/compare_sql_ast.py SOURCE.json REWRITTEN.txt

For example:

python3 data/compare_sql_ast.py \
  data/crimes_sql.json \
  data/crimes_sql_oracle_gpt-5.6-sol_v1_no_thinking.txt    

Changed query numbers (1-based): 5, 33, 68, 84  


python3 data/compare_sql_ast.py \
  data/crimes_sql.json \
  data/crimes_sql_oracle_deepseek_v1_no_thinking.txt

Changed query numbers (1-based): 12, 14, 93, 94, 100  

The source JSON must contain ``table_schemas`` and a ``sqls`` list, matching
the input format used by ``llm_method.py``.  The text file must contain the
same number of semicolon-terminated SQL statements in the same order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Running this file directly puts data/ rather than the repository root on
# sys.path.  Add the root so queryanalysis.py can be imported in both forms.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from queryanalysis import RPRAbstractSyntaxTree  # noqa: E402


def split_sql_statements(text: str) -> list[str]:
    """Split on SQL terminators while respecting quotes and comments."""
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
        raise ValueError("rewritten SQL file ends inside a quoted string or identifier")
    if in_block_comment:
        raise ValueError("rewritten SQL file ends inside a block comment")

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def canonical_ast_sql(sql: str, table_schemas: dict[str, Any]) -> str:
    """Parse SQL into the project AST and serialize it canonically."""
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, table_schemas)
    # This is the comparison style used by the rewrite-rule tests: compare the
    # AST serialization after removing formatting-only whitespace differences.
    return " ".join(ast.to_sql().strip().split())


def load_source(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    table_schemas = payload.get("table_schemas")
    sql_items = payload.get("sqls")
    if not isinstance(table_schemas, dict) or not table_schemas:
        raise ValueError("JSON 'table_schemas' must be a non-empty object")
    if not isinstance(sql_items, list):
        raise ValueError("JSON 'sqls' must be an array")

    for index, item in enumerate(sql_items, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("sql"), str):
            raise ValueError(f"JSON sqls[{index - 1}].sql must be a string")
    return table_schemas, sql_items


def compare_queries(
    table_schemas: dict[str, Any],
    source_items: list[dict[str, Any]],
    rewritten_sqls: list[str],
) -> tuple[list[int], list[int], list[tuple[int, str]]]:
    if len(source_items) != len(rewritten_sqls):
        raise ValueError(
            "query count mismatch: "
            f"JSON contains {len(source_items)}, text file contains {len(rewritten_sqls)}"
        )

    unchanged: list[int] = []
    changed: list[int] = []
    errors: list[tuple[int, str]] = []

    for index, (source_item, rewritten_sql) in enumerate(
        zip(source_items, rewritten_sqls), start=1
    ):
        try:
            source_ast = canonical_ast_sql(source_item["sql"], table_schemas)
        except Exception as exc:
            errors.append((index, f"original SQL parse failed: {exc}"))
            continue

        try:
            rewritten_ast = canonical_ast_sql(rewritten_sql, table_schemas)
        except Exception as exc:
            errors.append((index, f"rewritten SQL parse failed: {exc}"))
            continue

        if source_ast == rewritten_ast:
            unchanged.append(index)
        else:
            changed.append(index)

    return unchanged, changed, errors


def format_indices(indices: list[int]) -> str:
    return ", ".join(map(str, indices)) if indices else "(none)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original and rewritten MATCH_RECOGNIZE SQL via queryanalysis ASTs."
    )
    parser.add_argument("source_json", type=Path, help="original JSON query file")
    parser.add_argument("rewritten_txt", type=Path, help="rewritten SQL text file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        table_schemas, source_items = load_source(args.source_json)
        rewritten_sqls = split_sql_statements(
            args.rewritten_txt.read_text(encoding="utf-8")
        )
        unchanged, changed, errors = compare_queries(
            table_schemas, source_items, rewritten_sqls
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Total queries: {len(source_items)}")
    print(f"AST unchanged: {len(unchanged)}")
    print(f"Unchanged query numbers (1-based): {format_indices(unchanged)}")
    print(f"AST changed: {len(changed)}")
    print(f"Changed query numbers (1-based): {format_indices(changed)}")
    print(f"Parse errors: {len(errors)}")
    for index, message in errors:
        print(f"  Query {index}: {message}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
