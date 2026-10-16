import re

from queryanalysis import RPRAbstractSyntaxTree

from .base import RewriteEngine


def normalize_test_sql(sql: str) -> str:
    try:
        ast = RPRAbstractSyntaxTree()
        ast.parse_sql(sql, _schema_fixture_for_sql(sql))
        sql = ast.to_sql()
    except Exception:
        pass
    normalized = sql.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    normalized = " ".join(normalized.split())
    normalized = normalized.replace("__RPR_TAIL_RN__", "RPR_TAIL_RN")
    return (
        normalized.replace("( SELECT", "(SELECT")
        .replace(" )", ")")
        .replace("AS (", "AS(")
    )


def _schema_fixture_for_sql(sql: str):
    """Provide explicit metadata for rewrite tests that are not schema-specific."""
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql))
    columns = {identifier.upper(): "DOUBLE" for identifier in identifiers}
    source_match = re.search(
        r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+MATCH_RECOGNIZE\b",
        sql,
        re.IGNORECASE,
    )
    if source_match is not None:
        table_name = source_match.group(1).upper()
    else:
        derived_sources = {
            value.upper()
            for value in re.findall(
                r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                sql,
                re.IGNORECASE,
            )
        }
        table_name = (
            next(iter(derived_sources))
            if len(derived_sources) == 1
            else "__REWRITE_TEST_INPUT__"
        )
    return {
        table_name: {
            "columns": columns,
            "column_nullables": {name: False for name in columns},
        }
    }


def _complete_schema_fixture(sql: str, table_schemas):
    if not table_schemas:
        return _schema_fixture_for_sql(sql)

    source_match = re.search(
        r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+MATCH_RECOGNIZE\b",
        sql,
        re.IGNORECASE,
    )
    if source_match is not None:
        source_table = source_match.group(1).upper()
        if source_table not in {str(name).upper() for name in table_schemas}:
            return _schema_fixture_for_sql(sql)

    completed = dict(table_schemas)
    schema_table_names = [
        name
        for name, metadata in completed.items()
        if isinstance(metadata, dict)
        and any(key in metadata for key in ("columns", "column_types", "primary_key"))
    ]
    inferred_columns = {
        value.upper() for value in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql)
    }
    for table_name in schema_table_names:
        metadata = dict(completed[table_name])
        raw_columns = metadata.get("columns") or []
        column_names = {
            str(name).upper()
            for name in (
                raw_columns.keys() if isinstance(raw_columns, dict) else raw_columns
            )
            if not isinstance(name, dict)
        }
        if len(schema_table_names) == 1:
            column_names |= inferred_columns
        column_types = {
            str(name).upper(): data_type
            for name, data_type in (metadata.get("column_types") or {}).items()
        }
        for column_name in column_names:
            column_types.setdefault(column_name, "DOUBLE")
        metadata["column_types"] = column_types
        column_nullables = {
            str(name).upper(): nullable
            for name, nullable in (metadata.get("column_nullables") or {}).items()
        }
        for column_name in column_names:
            column_nullables.setdefault(column_name, False)
        metadata["column_nullables"] = column_nullables
        completed[table_name] = metadata
    return completed


def _schemas_referenced_by_sql(sql: str, table_schemas):
    """Keep table metadata for relations referenced by this test SQL.

    A module may describe several fixture tables.  Passing unrelated table
    metadata into an AST can unnecessarily reserve their column names, so each
    test receives only the tables it actually references.  Non-table settings,
    such as selectivity metadata, are retained.
    """
    upper_sql = sql.upper()
    from_pos = upper_sql.find("FROM")
    match_recognize_pos = upper_sql.find("MATCH_RECOGNIZE")
    input_fragment = (
        sql[from_pos + len("FROM") : match_recognize_pos]
        if from_pos >= 0 and match_recognize_pos > from_pos
        else sql
    )
    selected = {}
    for name, metadata in (table_schemas or {}).items():
        is_table_schema = isinstance(metadata, dict) and any(
            key in metadata for key in ("columns", "column_types", "primary_key")
        )
        is_referenced = (
            re.search(
                rf"(?<![A-Za-z0-9_$]){re.escape(str(name))}(?![A-Za-z0-9_$])",
                input_fragment,
                re.IGNORECASE,
            )
            is not None
        )
        if not is_table_schema or is_referenced:
            selected[name] = metadata
    return selected


def run_rule_tests(
    rule_cls,
    test_cases,
    label=None,
    verbose=True,
    default_table_schemas=None,
):
    """Run three-field SQL cases with schema metadata supplied separately.

    Each case must be ``(test_num, original_sql, expected_sql)``.  Use
    ``default_table_schemas`` to define metadata by table name.
    """
    passed = 0
    failed = 0
    failures = []
    label = label or rule_cls.__name__

    if verbose:
        print(f"Running {label} test cases...")

    for test_num, original_sql, expected_sql in test_cases:
        expect_error = expected_sql.strip().upper() == "ERROR"
        effective_schemas = _schemas_referenced_by_sql(
            original_sql,
            default_table_schemas,
        )
        effective_schemas = _complete_schema_fixture(
            original_sql,
            effective_schemas,
        )
        try:
            ast = RPRAbstractSyntaxTree()
            ast.parse_sql(original_sql, effective_schemas)
            engine = RewriteEngine([rule_cls()])
            rewritten_ast, _ = engine.rewrite(ast)
        except ValueError as exc:
            if expect_error:
                passed += 1
                continue
            failed += 1
            failures.append(
                {
                    "test": test_num,
                    "expected": expected_sql,
                    "actual": f"Unexpected error: {exc}",
                }
            )
            if verbose:
                print(f"FAILED Test case {test_num}")
                print(f"   Unexpected error: {exc}")
            continue

        if expect_error:
            failed += 1
            failures.append(
                {
                    "test": test_num,
                    "expected": expected_sql,
                    "actual": rewritten_ast.to_sql(),
                }
            )
            if verbose:
                print(f"FAILED Test case {test_num}")
                print("   Expected an error but rewrite succeeded")
            continue

        # please note that we normalize both the actual and expected SQL
        # to avoid false negatives due to formatting differences
        actual_sql = normalize_test_sql(rewritten_ast.to_sql())
        expected_sql = normalize_test_sql(expected_sql)

        if actual_sql != expected_sql:
            failed += 1
            failures.append(
                {
                    "test": test_num,
                    "expected": expected_sql,
                    "actual": actual_sql,
                }
            )
            if verbose:
                print(f"FAILED Test case {test_num}")
                print(f"   Expected: {expected_sql}")
                print(f"     Actual: {actual_sql}")
        else:
            passed += 1

    return {
        "label": label,
        "total": passed + failed,
        "passed": passed,
        "failed": failed,
        "failures": failures,
    }
