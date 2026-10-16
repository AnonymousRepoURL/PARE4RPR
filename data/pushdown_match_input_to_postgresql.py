#!/usr/bin/env python3
"""Move the input of MATCH_RECOGNIZE into Trino's PostgreSQL query function.

By default this converts ``crimes_sql_trino_our_method.txt`` into
``crimes_sql_trino_our_method_postgresql_query.txt``.  Add ``--validate`` to
ask Trino to parse and validate every generated statement.

Example (run from the repository root):

    python data/pushdown_match_input_to_postgresql.py --validate
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "crimes_sql_trino_our_method.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "crimes_sql_trino_our_method_postgresql_query.txt"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    depth: int


def split_sql_statements(text: str) -> list[str]:
    """Split on semicolons outside quoted strings, identifiers and comments."""
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    line_comment = False
    block_comment = False
    index = 0

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            current.append(char)
            if char in "\r\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            current.append(char)
            if char == "*" and following == "/":
                current.append(following)
                index += 2
                block_comment = False
            else:
                index += 1
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                if following == quote:
                    current.append(following)
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
        if char == "-" and following == "-":
            current.extend((char, following))
            line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            current.extend((char, following))
            block_comment = True
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
        raise ValueError(f"SQL ends inside a {quote!r} quoted value")
    if block_comment:
        raise ValueError("SQL ends inside a block comment")
    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


def sql_tokens(sql: str) -> list[Token]:
    """Return unquoted SQL words and punctuation with parenthesis depths."""
    tokens: list[Token] = []
    depth = 0
    index = 0
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if char.isspace():
            index += 1
            continue
        if char == "-" and following == "-":
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline == -1 else newline + 1
            continue
        if char == "/" and following == "*":
            end = sql.find("*/", index + 2)
            if end == -1:
                raise ValueError("SQL ends inside a block comment")
            index = end + 2
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            while index < len(sql):
                if sql[index] == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                raise ValueError(f"SQL ends inside a {quote!r} quoted value")
            continue
        if char == "(":
            tokens.append(Token(char, index, index + 1, depth))
            depth += 1
            index += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("SQL has an unmatched closing parenthesis")
            tokens.append(Token(char, index, index + 1, depth))
            index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < len(sql) and (
                sql[index].isalnum() or sql[index] in {"_", "$"}
            ):
                index += 1
            tokens.append(Token(sql[start:index], start, index, depth))
            continue
        tokens.append(Token(char, index, index + 1, depth))
        index += 1

    if depth != 0:
        raise ValueError("SQL has unmatched parentheses")
    return tokens


def keyword_tokens(sql: str, keyword: str, depth: int = 0) -> list[Token]:
    wanted = keyword.upper()
    return [
        token
        for token in sql_tokens(sql)
        if token.depth == depth and token.text.upper() == wanted
    ]


def count_keyword(sql: str, keyword: str) -> int:
    wanted = keyword.upper()
    return sum(token.text.upper() == wanted for token in sql_tokens(sql))


def unwrap_query_relation(relation: str, allow_with: bool = True) -> str | None:
    """Return the query inside a wholly parenthesized relation when safe.

    ``FROM (SELECT ...)`` needs an outer SELECT when used as a relation, but
    the same SELECT can be sent to PostgreSQL directly through ``query()``.
    Do not unwrap relations with aliases, joins or other text outside the
    parentheses.  Repeated parentheses are accepted, as are parentheses and
    quotes inside the query itself.

    A nested WITH cannot follow an already extracted leading WITH directly,
    so callers can disable that form with ``allow_with=False``.
    """
    candidate = relation.strip()
    while candidate:
        tokens = sql_tokens(candidate)
        top_level = [token for token in tokens if token.depth == 0]
        wholly_parenthesized = (
            len(top_level) == 2
            and top_level[0].text == "("
            and top_level[0].start == 0
            and top_level[1].text == ")"
            and top_level[1].end == len(candidate)
        )
        if not wholly_parenthesized:
            return None

        candidate = candidate[top_level[0].end : top_level[1].start].strip()
        inner_tokens = sql_tokens(candidate)
        first_word = next(
            (
                token
                for token in inner_tokens
                if token.depth == 0 and token.text[0].isalpha()
            ),
            None,
        )
        if first_word is None:
            # The relation may contain another redundant pair of parentheses.
            continue

        query_starters = {"SELECT", "VALUES", "TABLE"}
        if allow_with:
            query_starters.add("WITH")
        if first_word.text.upper() in query_starters:
            return candidate
        return None

    return None


def build_remote_sql(remote_prefix: str, relation: str) -> str:
    """Build the least-nested PostgreSQL query equivalent to ``relation``."""
    direct_query = unwrap_query_relation(relation, allow_with=not remote_prefix)
    remote_body = direct_query or f"SELECT *\nFROM {relation}"
    if remote_prefix:
        return f"{remote_prefix}\n{remote_body}"
    return remote_body


def convert_statement(sql: str, catalog: str) -> tuple[str, int]:
    """Convert one query and return it plus its pushed-down CASE count."""
    tokens = sql_tokens(sql)
    matches = [
        token
        for token in tokens
        if token.depth == 0 and token.text.upper() == "MATCH_RECOGNIZE"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one top-level MATCH_RECOGNIZE, found {len(matches)}"
        )
    match_token = matches[0]
    from_tokens = [
        token
        for token in tokens
        if token.depth == 0
        and token.text.upper() == "FROM"
        and token.start < match_token.start
    ]
    if not from_tokens:
        raise ValueError("could not find the top-level FROM before MATCH_RECOGNIZE")
    from_token = from_tokens[-1]
    select_tokens = [
        token
        for token in tokens
        if token.depth == 0
        and token.text.upper() == "SELECT"
        and token.start < from_token.start
    ]
    if not select_tokens:
        raise ValueError("could not find the top-level SELECT before MATCH_RECOGNIZE")
    select_token = select_tokens[-1]

    relation = sql[from_token.end : match_token.start].strip()
    if not relation:
        raise ValueError("MATCH_RECOGNIZE input relation is empty")

    # A leading WITH clause defines the input relation.  It must be sent to
    # PostgreSQL too; a CTE evaluated by Trino is invisible inside query().
    leading_tokens = [token for token in tokens if token.start < select_token.start]
    first_word = next(
        (token for token in leading_tokens if token.text[0].isalpha()), None
    )
    has_cte = first_word is not None and first_word.text.upper() == "WITH"
    remote_prefix = sql[: select_token.start].strip() if has_cte else ""
    remote_sql = build_remote_sql(remote_prefix, relation)

    pushed_case_count = count_keyword(remote_sql, "CASE")
    input_case_count = count_keyword(relation, "CASE")
    if pushed_case_count < input_case_count:
        raise AssertionError("a CASE expression was lost while building remote SQL")

    escaped_remote_sql = remote_sql.replace("'", "''")
    # MATCH_RECOGNIZE cannot be applied directly to a polymorphic table
    # function invocation in Trino.  Materializing its shape through a
    # derived SELECT makes it a regular relation while preserving pushdown.
    table_function = (
        "(\n"
        "    SELECT *\n"
        "    FROM TABLE(\n"
        f"        {catalog}.system.query(\n"
        f"            query => '{escaped_remote_sql}'\n"
        "        )\n"
        "    )\n"
        ")"
    )

    outer_leading = "" if has_cte else sql[: select_token.start]
    outer_select_through_from = sql[select_token.start : from_token.end].rstrip()
    match_and_rest = sql[match_token.start :].lstrip()
    converted = (
        outer_leading
        + outer_select_through_from
        + "\n"
        + table_function
        + "\n"
        + match_and_rest
    )

    # No CASE from the original input is left for Trino to evaluate.  CASEs
    # in MATCH_RECOGNIZE clauses or post-MR joins, if any, are out of scope.
    outer_input = converted[
        converted.upper().find("FROM") : converted.upper().find("MATCH_RECOGNIZE")
    ]
    if count_keyword(outer_input, "CASE") != 0:
        # CASE text inside the query string is ignored by sql_tokens, so a hit
        # here really means an expression escaped the string literal.
        raise AssertionError("CASE remains in the Trino-side MATCH input")
    return converted, pushed_case_count


class TrinoValidationError(RuntimeError):
    pass


def trino_request(
    request: urllib.request.Request, timeout_seconds: float
) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_seconds) as response:
        return json.load(response)


def coordinator_uri(base_url: str, uri: str) -> str:
    base = urllib.parse.urlsplit(base_url)
    target = urllib.parse.urlsplit(uri)
    return urllib.parse.urlunsplit(
        (base.scheme, base.netloc, target.path, target.query, target.fragment)
    )


def validate_with_trino(
    statements: list[str],
    url: str,
    user: str,
    catalog: str,
    schema: str,
    timeout_seconds: float,
) -> None:
    """Run EXPLAIN(TYPE VALIDATE), including connector-side query analysis."""
    base_url = url.rstrip("/")
    headers = {
        "X-Trino-User": user,
        "X-Trino-Source": "postgresql-query-pushdown-converter",
        "X-Trino-Catalog": catalog,
        "X-Trino-Schema": schema,
        "Content-Type": "text/plain; charset=utf-8",
    }
    for number, statement in enumerate(statements, 1):
        explain = f"EXPLAIN (TYPE VALIDATE) {statement}"
        request = urllib.request.Request(
            f"{base_url}/v1/statement",
            data=explain.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            page = trino_request(request, timeout_seconds)
            while True:
                error = page.get("error")
                if isinstance(error, dict):
                    name = error.get("errorName", "TRINO_ERROR")
                    message = error.get("message", error)
                    raise TrinoValidationError(f"query {number}: {name}: {message}")
                next_uri = page.get("nextUri")
                if not next_uri:
                    break
                request = urllib.request.Request(
                    coordinator_uri(base_url, str(next_uri)), headers=headers
                )
                page = trino_request(request, timeout_seconds)
        except (OSError, urllib.error.URLError) as exc:
            raise TrinoValidationError(
                f"query {number}: could not contact Trino: {exc}"
            ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", default="postgresql")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--trino-url", default="http://localhost:8080")
    parser.add_argument("--trino-user", default="trino")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate every generated query with the running Trino instance",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not IDENTIFIER_RE.fullmatch(args.catalog):
        raise SystemExit(f"invalid catalog identifier: {args.catalog!r}")
    if not IDENTIFIER_RE.fullmatch(args.schema):
        raise SystemExit(f"invalid schema identifier: {args.schema!r}")

    source = args.input.read_text(encoding="utf-8-sig")
    statements = split_sql_statements(source)
    if not statements:
        raise SystemExit(f"no SQL statements found in {args.input}")

    converted: list[str] = []
    case_queries = 0
    case_expressions = 0
    for number, statement in enumerate(statements, 1):
        try:
            result, cases = convert_statement(statement, args.catalog)
        except (ValueError, AssertionError) as exc:
            raise SystemExit(f"query {number}: {exc}") from exc
        converted.append(result)
        case_queries += int(cases > 0)
        case_expressions += cases

    output_text = ";\n\n".join(converted) + ";\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")
    print(
        f"Converted {len(converted)} queries; pushed {case_expressions} CASE "
        f"expressions in {case_queries} queries to PostgreSQL."
    )
    print(f"Output: {args.output.resolve()}")

    if args.validate:
        validate_with_trino(
            converted,
            args.trino_url,
            args.trino_user,
            args.catalog,
            args.schema,
            args.timeout,
        )
        print(f"Trino validation passed: {len(converted)}/{len(converted)}")


if __name__ == "__main__":
    main()
