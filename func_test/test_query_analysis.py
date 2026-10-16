import re

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from queryanalysis import RPRAbstractSyntaxTree


def test_optimize_from_clause():
    ast = RPRAbstractSyntaxTree()
    ast.input_table = "(SELECT * FROM (SELECT * FROM T WHERE A = 1) WHERE B = 2)"
    assert ast.optimize_from_clause()
    assert ast.input_table == "(SELECT * FROM T WHERE (A = 1) AND (B = 2))"
    assert not ast.optimize_from_clause()

    ast.input_table = "(SELECT * FROM (SELECT * FROM T ORDER BY TS) WHERE B = 2)"
    assert not ast.optimize_from_clause()

    ast.input_table = (
        "(SELECT * FROM (SELECT * FROM T WHERE NOTE = 'ORDER BY X') WHERE B = 2)"
    )
    assert ast.optimize_from_clause()
    assert (
        ast.input_table == "(SELECT * FROM T WHERE (NOTE = 'ORDER BY X') AND (B = 2))"
    )


def test_flatten_safe_select_star_input():
    ast = RPRAbstractSyntaxTree()
    assert ast.flatten_safe_select_star_input(
        "(SELECT * FROM (SELECT * FROM APP.T WHERE VAL > 10) WHERE TYPE IN ('A', 'C'))"
    ) == ("APP.T", ["VAL > 10", "TYPE IN ('A', 'C')"])

    # Query-block-sensitive pseudo-columns and potentially effectful/erroring
    # expressions must retain their original evaluation boundary.
    assert (
        ast.flatten_safe_select_star_input("(SELECT * FROM T WHERE ROWNUM <= 10)")
        is None
    )
    assert (
        ast.flatten_safe_select_star_input("(SELECT * FROM T WHERE RANDOM() > 0.5)")
        is None
    )
    assert (
        ast.flatten_safe_select_star_input("(SELECT * FROM T WHERE 10 / DENOM > 1)")
        is None
    )

    # Row shape, ordering/cardinality, binding, and unknown syntax all make a
    # wrapper ineligible rather than triggering a best-effort rewrite.
    unsafe = [
        "(SELECT PT, TYPE FROM T WHERE VAL > 10)",
        "(SELECT * FROM T X WHERE VAL > 10)",
        "(SELECT * FROM T ORDER BY TS)",
        "(SELECT * FROM T FETCH FIRST 10 ROWS ONLY)",
        "(SELECT * FROM T WHERE PT IN (SELECT PT FROM U))",
        "(SELECT * FROM T WHERE X.VAL > 10)",
    ]
    for sql in unsafe:
        assert ast.flatten_safe_select_star_input(sql) is None, sql


# Each tuple contains: (id, original_input, expected_output)
analysis_test_cases = [
    (
        1,
        """
SELECT * FROM (SELECT * FROM t WHERE v > 10) MATCH_RECOGNIZE(
    PARTITION BY pt
    ORDER BY ts
    MEASURES
        A.ID AS AID,
        B.ID AS BID,
        C.ID AS CID,
        AB.value / 2 AS FIRST_TWO_AVG_VALUE,
        SUM(ABC.value) AS SUM_VALUE
    PATTERN(A B C)
    SUBSET AB = (A, B), ABC = (A, B, C)
    DEFINE A AS A.type ='A', B AS B.type ='B', C AS C.type ='C'
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE V > 10) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID, (AB.VALUE / 2) AS FIRST_TWO_AVG_VALUE, SUM(ABC.VALUE) AS SUM_VALUE
    PATTERN(A B C)
    SUBSET AB = (A,  B), ABC = (A,  B,  C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)
""",
    ),
    (
        2,
        """
SELECT * FROM table MATCH_RECOGNIZE(
    PARTITION BY pt
    ORDER BY ts ASC, col1 DESC
    MEASURES
        A.ID AS AID,
        B.ID AS BID,
        C.ID AS CID,
        RUNNING SUM(A.value) AS RUNNING_SUM,
        FINAL SUM(A.value) AS FINAL_SUM
    PATTERN(A{2} B C)
    DEFINE
        A AS A.type ='A',
        B AS B.type ='B',
        C AS C.type ='C'
)""",
        """
SELECT *
FROM TABLE
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS, COL1 DESC
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID, SUM(A.VALUE) AS RUNNING_SUM, FINAL SUM(A.VALUE) AS FINAL_SUM
    PATTERN(A{2} B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    (
        3,
        """
SELECT * FROM table MATCH_RECOGNIZE(
    PARTITION BY pt
    ORDER BY ts ASC, col1 DESC
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID
    PATTERN(A{2} B C)
    DEFINE A AS A.type ='A', B AS B.type ='B', C AS C.type ='C'
) AS MR
WHERE MR.pt = 'test' AND MR.aid > 1000
GROUP BY MR.pt
HAVING COUNT(*) > 5
ORDER BY MR.pt DESC
LIMIT 10
""",
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS, COL1 DESC
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID
    PATTERN(A{2} B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
) AS MR 
WHERE (MR.PT = 'test' AND MR.AID > 1000)
GROUP BY MR.PT
HAVING COUNT(*) > 5
ORDER BY MR.PT DESC
LIMIT 10
""",
    ),
    (
        4,
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO PAST LAST ROW
    PATTERN(A B C)
    DEFINE 
        A AS TS > 300 OR TS > 100, 
        B AS (TYPE = 'B' AND TYPE != 'B') OR TYPE = 'C', 
        C AS (TS < 200 AND TYPE = 'C') OR (TS < 200 AND NOT TYPE = 'C')
)""",
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS TS > 100, B AS TYPE = 'C', C AS TS < 200 AND (TYPE IS NOT NULL)
)""",
    ),
    (
        5,
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO PAST LAST ROW
    PATTERN(A B C)
    DEFINE 
        A AS TS > 300 AND TS < 100, 
        B AS (OPEN > 10 AND OPEN < 10) OR TYPE = 'C', 
        C AS (TS < 200 OR TYPE = 'C') AND (TS < 200 OR NOT TYPE = 'C')
)""",
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS 1 != 1, B AS TYPE = 'C', C AS TS < 200
)""",
    ),
    (
        6,
        """
SELECT * FROM table MATCH_RECOGNIZE(
    PARTITION BY pt
    ORDER BY ts
    MEASURES
        PREV(A.price, 0) AS PREV_PRICE,
        NEXT(price, 0) AS NEXT_PRICE
    PATTERN(A)
    DEFINE A AS PREV(A.price, 0) > 100 AND NEXT(price, 0) < 200
)""",
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.PRICE AS PREV_PRICE, PRICE AS NEXT_PRICE
    PATTERN(A)
    DEFINE A AS PRICE > 100 AND PRICE < 200
)""",
    ),
    (
        7,
        """
SELECT * FROM table MATCH_RECOGNIZE(
    ORDER BY ts
    MEASURES A.name AS match_name
    PATTERN(A)
    DEFINE A AS A.type = 'c' AND A.name LIKE 'ab%'
) AS mr
WHERE mr.match_name = 'Alice'
""",
        """
SELECT * FROM TABLE MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.NAME AS MATCH_NAME
    PATTERN(A)
    DEFINE A AS TYPE = 'c' AND (NAME LIKE 'ab%')
) AS MR 
WHERE MR.MATCH_NAME = 'Alice'
""",
    ),
]


def _normalize_sql_for_compare(sql: str) -> str:
    return " ".join(sql.strip().split())


def _schema_for_sql(sql: str):
    source_match = re.search(
        r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+MATCH_RECOGNIZE\b",
        sql,
        re.IGNORECASE,
    )
    table_name = source_match.group(1).upper() if source_match else "T"
    identifiers = {
        value.upper() for value in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sql)
    }
    return {
        table_name: {
            "columns": {name: "DOUBLE" for name in identifiers},
            "column_nullables": {name: True for name in identifiers},
        }
    }


def _parse_sql(ast, sql, schema=None):
    ast.parse_sql(sql, schema or _schema_for_sql(sql))


def test_rpr_analysis_cases():
    for case_id, original_input, expected_output in analysis_test_cases:
        ast = RPRAbstractSyntaxTree()
        _parse_sql(ast, original_input)
        actual_output = ast.to_sql()
        assert _normalize_sql_for_compare(actual_output) == _normalize_sql_for_compare(
            expected_output
        ), f"case {case_id} failed, expected:\n{expected_output}\nbut got:\n{actual_output}"


def test_parse_requires_complete_schema_metadata():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS VALUE > 1
)
"""
    try:
        RPRAbstractSyntaxTree().parse_sql(sql)
    except ValueError as error:
        assert "input table schema metadata is required" in str(error)
    else:
        raise AssertionError("expected missing table schema to be rejected")

    incomplete_schema = {
        "T": {
            "columns": {"TS": "LONG"},
            "column_nullables": {"TS": False},
        }
    }
    try:
        RPRAbstractSyntaxTree().parse_sql(sql, incomplete_schema)
    except ValueError as error:
        assert "missing schema metadata for column 'VALUE'" in str(error)
    else:
        raise AssertionError("expected missing column metadata to be rejected")


def _assert_parse_error(sql, expected_message):
    try:
        _parse_sql(RPRAbstractSyntaxTree(), sql)
    except ValueError as error:
        assert expected_message in str(error)
    else:
        raise AssertionError("expected parse_sql() to raise ValueError")


def test_match_recognize_input_accepts_supported_relation_shapes():
    schemas = {
        "T": {
            "columns": {"TS": "LONG", "ID": "LONG", "TYPE": "VARCHAR(20)"},
            "column_nullables": {"TS": False, "ID": False, "TYPE": False},
        },
        "U": {
            "columns": {"ID": "LONG"},
            "column_nullables": {"ID": False},
        },
        "APP.VIEW_T": {
            "columns": {"TS": "LONG", "ID": "LONG", "TYPE": "VARCHAR(20)"},
            "column_nullables": {"TS": False, "ID": False, "TYPE": False},
        },
        "NAMED_QUERY": {
            "columns": {"TS": "LONG", "ID": "LONG", "TYPE": "VARCHAR(20)"},
            "column_nullables": {"TS": False, "ID": False, "TYPE": False},
        },
    }
    sources = [
        "T",
        "APP.VIEW_T V",
        "NAMED_QUERY AS NQ",
        "(SELECT * FROM T)",
        "((SELECT * FROM T) UNION ALL (SELECT * FROM T))",
        "(SELECT T.* FROM T LEFT JOIN U ON T.ID = U.ID) J",
        "(SELECT T.* FROM T LEFT JOIN U USING (ID)) J",
        "(SELECT T.* FROM T CROSS JOIN U) J",
    ]

    for source in sources:
        sql = f"""
SELECT * FROM {source} MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
        ast = RPRAbstractSyntaxTree()
        ast.parse_sql(sql, schemas)
        assert ast.input_table == source.upper()


def test_match_recognize_input_rejects_non_relation_shapes():
    invalid_sources = [
        "T JOIN U ON T.ID = U.ID",
        "T LEFT JOIN U ON T.ID = U.ID",
        "T, U",
        "(T JOIN U ON T.ID = U.ID)",
        "T WHERE TYPE = 'A'",
        "(SELECT * FROM T) D EXTRA",
    ]

    for source in invalid_sources:
        sql = f"""
SELECT * FROM {source} MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
        _assert_parse_error(
            sql,
            "MATCH_RECOGNIZE input must be a table, view, named query, or "
            "parenthesized derived table",
        )

    _assert_parse_error(
        """
SELECT * FROM (SELECT * FROM T LEFT JOIN U WHERE T.ID = U.ID) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        "derived-table JOIN must have an ON or USING condition",
    )


def _parse_one_row_without_measures(rows_per_match):
    sql = f"""
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    {rows_per_match}
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
    ast = RPRAbstractSyntaxTree()
    _parse_sql(ast, sql)
    return ast


def test_explicit_one_row_per_match_allows_omitted_measures_with_partition():
    ast = _parse_one_row_without_measures("ONE ROW PER MATCH")
    assert ast.get_match_recognize_output_columns() == ["PT"]
    assert ast.expand_select_star()
    assert ast.output_names.to_string() == "SELECT PT"


def test_default_one_row_per_match_allows_omitted_measures_with_partition():
    ast = _parse_one_row_without_measures("")
    assert ast.get_match_recognize_output_columns() == ["PT"]


def test_all_rows_output_columns_drop_source_qualifiers():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT * FROM T X MATCH_RECOGNIZE(
    PARTITION BY X.PT
    ORDER BY X.TS, X.ID
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        {
            "T": {
                "columns": {
                    "PT": "LONG",
                    "TS": "LONG",
                    "ID": "LONG",
                    "TYPE": "VARCHAR(20)",
                    "VALUE": "DOUBLE",
                },
                "column_nullables": {
                    "PT": False,
                    "TS": False,
                    "ID": False,
                    "TYPE": False,
                    "VALUE": False,
                },
            },
        },
    )
    assert ast.get_match_recognize_output_columns() == [
        "PT",
        "TS",
        "ID",
        "TYPE",
        "VALUE",
    ]


def test_one_row_per_match_rejects_zero_column_output():
    _assert_parse_error(
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ONE ROW PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        "requires at least one PARTITION BY column or MEASURES definition",
    )


def test_explicit_empty_measures_clause_is_invalid():
    _assert_parse_error(
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    MEASURES
    ONE ROW PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        "'MEASURES' subclause requires at least one measure definition",
    )


def test_measures_rejects_variable_not_declared_by_pattern_or_subset():
    _assert_parse_error(
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, LAST(N.ID) AS NID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*?-} B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', N1 AS TYPE != A
)
""",
        "MEASURES variable(s) not present in PATTERN or SUBSET: N",
    )


def test_measures_and_define_accept_pattern_and_subset_variables():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT NID FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES LAST(N.ID) AS NID
    PATTERN(A B)
    SUBSET N = (A, B)
    DEFINE A AS A.TYPE = 'A', B AS SUM(N.VALUE) > 0
)
""",
    )

    assert ast.measures.to_string() == "MEASURES LAST(N.ID) AS NID"


def test_define_rejects_referenced_variable_not_declared_by_pattern_or_subset():
    _assert_parse_error(
        """
SELECT AID FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A)
    DEFINE A AS N.TYPE = 'A'
)
""",
        "DEFINE variable(s) not present in PATTERN or SUBSET: N",
    )


def test_variable_validation_happens_before_expression_simplification():
    _assert_parse_error(
        """
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES CASE WHEN 1 = 1 THEN A.ID ELSE N.ID END AS X
    PATTERN(A)
    DEFINE A AS TRUE
)
""",
        "MEASURES variable(s) not present in PATTERN or SUBSET: N",
    )


def test_reused_ast_does_not_inherit_subset_variables():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES N.ID AS X
    PATTERN(A)
    SUBSET N = (A)
    DEFINE A AS TRUE
)
""",
    )

    try:
        _parse_sql(
            ast,
            """
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES N.ID AS X
    PATTERN(A)
    DEFINE A AS TRUE
)
""",
        )
    except ValueError as error:
        assert "MEASURES variable(s) not present in PATTERN or SUBSET: N" in str(error)
    else:
        raise AssertionError(
            "expected the omitted SUBSET declaration not to be inherited"
        )


def test_one_row_required_input_columns_include_match_order_by():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.VALUE) AS FIRST_VALUE
    ONE ROW PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
    )
    assert ast.get_required_input_table_columns() == {"PT", "TS", "VALUE", "TYPE"}


def test_projection_columns_preserve_all_rows_select_star():
    ast = RPRAbstractSyntaxTree()
    schema = {
        "T": {
            "columns": {
                "PT": "INT",
                "TS": "LONG",
                "VALUE": "DOUBLE",
                "TYPE": "VARCHAR(20)",
                "UNUSED": "VARCHAR(20)",
            },
            "column_nullables": {
                "PT": False,
                "TS": False,
                "VALUE": True,
                "TYPE": True,
                "UNUSED": True,
            },
        }
    }
    ast.parse_sql(
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.VALUE) AS FIRST_VALUE
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        schema,
    )

    assert ast.get_required_input_table_columns() == "*"
    assert ast.get_input_table_columns_for_projection() == "*"


def test_projection_columns_preserve_qualified_all_rows_wildcard():
    ast = RPRAbstractSyntaxTree()
    schema = {
        "T": {
            "columns": {"TS": "LONG", "TYPE": "VARCHAR(20)"},
            "column_nullables": {"TS": False, "TYPE": True},
        }
    }
    ast.parse_sql(
        """
SELECT MR.* FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
) AS MR
""",
        schema,
    )

    assert ast.select_contains_wildcard()
    assert ast.get_input_table_columns_for_projection() == "*"


def test_all_rows_projection_includes_explicit_outer_source_columns():
    ast = RPRAbstractSyntaxTree()
    schema = {
        "T": {
            "columns": {
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
                "EXTRA": "DOUBLE",
            },
            "column_nullables": {"TS": False, "TYPE": True, "EXTRA": True},
        }
    }
    ast.parse_sql(
        """
SELECT EXTRA, MATCH_NO FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES MATCH_NUMBER() AS MATCH_NO
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
        schema,
    )

    assert ast.get_input_table_columns_for_projection() == {
        "TS",
        "TYPE",
        "EXTRA",
    }


def test_one_row_rejects_outer_references_not_in_match_output():
    invalid_queries = {
        "SELECT": "SELECT TS FROM T MATCH_RECOGNIZE(",
        "WHERE": "SELECT * FROM T MATCH_RECOGNIZE(",
        "GROUP BY": "SELECT * FROM T MATCH_RECOGNIZE(",
        "HAVING": "SELECT * FROM T MATCH_RECOGNIZE(",
        "ORDER BY": "SELECT * FROM T MATCH_RECOGNIZE(",
    }
    tail_by_clause = {
        "SELECT": "",
        "WHERE": " WHERE TS > 0",
        "GROUP BY": " GROUP BY TS",
        "HAVING": " HAVING TS > 0",
        "ORDER BY": " ORDER BY TS",
    }
    body = """
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.VALUE) AS FIRST_VALUE
    ONE ROW PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)"""
    for clause_name, prefix in invalid_queries.items():
        _assert_parse_error(
            prefix + body + tail_by_clause[clause_name],
            f"{clause_name} references column(s) not produced by MATCH_RECOGNIZE: TS",
        )


def test_outer_order_by_may_reference_select_alias():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT FIRST_VALUE AS RESULT_VALUE FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.VALUE) AS FIRST_VALUE
    ONE ROW PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
ORDER BY RESULT_VALUE
""",
    )
    assert ast.validate() == []


def test_all_rows_outer_references_are_validated_when_schema_is_available():
    sql = """
SELECT UNKNOWN_COL FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
    schema = {
        "T": {
            "columns": {"TS": "LONG", "TYPE": "VARCHAR(20)", "VALUE": "DOUBLE"},
            "column_nullables": {"TS": False, "TYPE": True, "VALUE": True},
        }
    }
    try:
        RPRAbstractSyntaxTree().parse_sql(sql, schema)
    except ValueError as error:
        assert str(error) == (
            "SELECT references column(s) not produced by MATCH_RECOGNIZE: "
            "UNKNOWN_COL"
        )
    else:
        raise AssertionError("expected unavailable output column to be rejected")


def test_all_rows_per_match_allows_omitted_measures():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
    ast = RPRAbstractSyntaxTree()
    _parse_sql(ast, sql)

    assert ast.rows_per_match == "ALL ROWS PER MATCH"
    assert ast.measures is None


def test_all_rows_per_match_suboptions_are_preserved():
    for rows_per_match in (
        "ALL ROWS PER MATCH SHOW EMPTY MATCHES",
        "ALL ROWS PER MATCH OMIT EMPTY MATCHES",
        "ALL ROWS PER MATCH WITH UNMATCHED ROWS",
    ):
        sql = f"""
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    {rows_per_match}
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
        ast = RPRAbstractSyntaxTree()
        _parse_sql(ast, sql)

        assert ast.rows_per_match == rows_per_match
        assert rows_per_match in ast.to_sql()
        assert ast.get_required_input_table_columns() == "*"


def test_with_unmatched_rows_rejects_excluded_patterns():
    for pattern in ("A {- B -} C", "A ({- B -} | C)+"):
        _assert_parse_error(
            f"""
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH WITH UNMATCHED ROWS
    PATTERN({pattern})
    DEFINE A AS TRUE, B AS TRUE, C AS TRUE
)
""",
            "ALL ROWS PER MATCH WITH UNMATCHED ROWS does not allow excluded patterns",
        )


def test_simplified_pattern_retains_rewrite_boundary_metadata():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY ID
    ONE ROW PER MATCH
    PATTERN(A {- B C -} PERMUTE(D, E))
    DEFINE A AS TRUE, B AS TRUE, C AS TRUE, D AS TRUE, E AS TRUE
)
""",
    )

    boundaries = ast.pattern_rewrite_boundaries
    assert [(item.kind, item.variables) for item in boundaries] == [
        ("exclude", frozenset({"B", "C"})),
        ("permute", frozenset({"D", "E"})),
    ]
    assert boundaries[1].elements == (
        frozenset({"D"}),
        frozenset({"E"}),
    )


def test_omitted_per_match_does_not_inherit_previous_all_rows_mode():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
""",
    )

    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""
    _parse_sql(ast, sql)
    assert ast.rows_per_match == "ONE ROW PER MATCH"
    assert ast.get_match_recognize_output_columns() == ["PT"]


def test_physical_navigation_cannot_nest_physical_navigation():
    for expression in (
        "PREV(PREV(A.ID, 2), 3)",
        "NEXT(NEXT(A.ID, 2), 3)",
        "PREV(NEXT(A.ID, 2), 3)",
        "NEXT(PREV(A.ID, 2), 3)",
    ):
        _assert_parse_error(
            f"""
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES {expression} AS X
    PATTERN(A)
    DEFINE A AS TRUE
)
""",
            "cannot contain PREV or NEXT",
        )


def test_physical_navigation_allows_logical_navigation_base():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES PREV(LAST(A.ID), 3) AS X
    PATTERN(A+)
    DEFINE A AS TRUE
)
""",
    )
    assert ast.measures.has_func("PREV")
    assert ast.measures.has_func("LAST")


def test_navigation_cannot_contain_aggregate():
    for expression in (
        "PREV(SUM(A.ID))",
        "NEXT(A.ID + AVG(A.VALUE))",
        "FIRST(COUNT(A.*))",
        "LAST(MAX(A.ID))",
    ):
        _assert_parse_error(
            f"""
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES {expression} AS X
    PATTERN(A)
    DEFINE A AS TRUE
)
""",
            "does not allow nested aggregates or navigation operators",
        )


def test_aggregate_cannot_contain_navigation():
    for expression in (
        "SUM(PREV(A.ID))",
        "AVG(NEXT(A.VALUE))",
        "MIN(FIRST(A.ID))",
        "MAX(LAST(A.ID))",
    ):
        _assert_parse_error(
            f"""
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES {expression} AS X
    PATTERN(A+)
    DEFINE A AS TRUE
)
""",
            "does not allow nested aggregates or navigation operators",
        )


def test_oracle_navigation_nesting_exception_is_allowed():
    ast = RPRAbstractSyntaxTree()
    _parse_sql(
        ast,
        """
SELECT X, Y FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES PREV(LAST(A.ID), 3) AS X, NEXT(FIRST(A.VALUE)) AS Y
    PATTERN(A+)
    DEFINE A AS TRUE
)
""",
    )
    assert ast.measures.has_func("PREV")
    assert ast.measures.has_func("NEXT")
    assert ast.measures.has_func("FIRST")
    assert ast.measures.has_func("LAST")


def test_match_metadata_functions_cannot_be_nested_in_rpr_operators():
    for expression in (
        "SUM(MATCH_NUMBER())",
        "PREV(CLASSIFIER())",
    ):
        _assert_parse_error(
            f"""
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES {expression} AS X
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS TRUE
)
""",
            "inside an aggregate or navigation operator",
        )


def test_rpr_operators_cannot_reference_multiple_pattern_variables():
    for expression in (
        "SUM(A.ID + B.VALUE)",
        "PREV(A.ID + B.VALUE)",
        "MAX(A.ID + VALUE)",
    ):
        _assert_parse_error(
            f"""
SELECT X FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES {expression} AS X
    PATTERN(A B)
    DEFINE A AS TRUE, B AS TRUE
)
""",
            "cannot reference multiple pattern variables",
        )


if __name__ == "__main__":
    test_optimize_from_clause()
    test_flatten_safe_select_star_input()
    test_rpr_analysis_cases()
    test_match_recognize_input_accepts_supported_relation_shapes()
    test_match_recognize_input_rejects_non_relation_shapes()
    test_explicit_one_row_per_match_allows_omitted_measures_with_partition()
    test_default_one_row_per_match_allows_omitted_measures_with_partition()
    test_one_row_per_match_rejects_zero_column_output()
    test_explicit_empty_measures_clause_is_invalid()
    test_measures_rejects_variable_not_declared_by_pattern_or_subset()
    test_measures_and_define_accept_pattern_and_subset_variables()
    test_define_rejects_referenced_variable_not_declared_by_pattern_or_subset()
    test_variable_validation_happens_before_expression_simplification()
    test_reused_ast_does_not_inherit_subset_variables()
    test_one_row_required_input_columns_include_match_order_by()
    test_one_row_rejects_outer_references_not_in_match_output()
    test_outer_order_by_may_reference_select_alias()
    test_all_rows_outer_references_are_validated_when_schema_is_available()
    test_all_rows_per_match_allows_omitted_measures()
    test_all_rows_per_match_suboptions_are_preserved()
    test_with_unmatched_rows_rejects_excluded_patterns()
    test_simplified_pattern_retains_rewrite_boundary_metadata()
    test_omitted_per_match_does_not_inherit_previous_all_rows_mode()
    test_physical_navigation_cannot_nest_physical_navigation()
    test_physical_navigation_allows_logical_navigation_base()
    test_navigation_cannot_contain_aggregate()
    test_aggregate_cannot_contain_navigation()
    test_oracle_navigation_nesting_exception_is_allowed()
    test_match_metadata_functions_cannot_be_nested_in_rpr_operators()
    test_rpr_operators_cannot_reference_multiple_pattern_variables()
    print("All RPR analysis test cases passed!")
