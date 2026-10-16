"""Tests for clause parsing, pattern-node analysis and column metadata."""

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from clause_parser import ExprParser
from expression_clause import DependentCondition, IndependentCondition
from patternnode import parse_pattern_string
from queryanalysis import RPRAbstractSyntaxTree
from rewriting_rules.anchor_partition_size_filter import AnchorPartitionSizeFilterRule
from rewriting_rules.column_pruning import ColumnPruningRule
from rewriting_rules.must_appear_partition_filter_pushdown import (
    MustAppearPartitionFilterPushdownRule,
)
from rewriting_rules.ordered_range_pushdown import OrderedRangePushdownRule
from rewriting_rules.physical_navigation import PatternNavigationReach
from rewriting_rules.sequence_filtering import SequenceFilteringRule

# ----------------------------------------------------------------------
# DEFINE / MEASURES clause parsing and special-function validation
# ----------------------------------------------------------------------
ORACLE_ONLY_FUNCTIONS = (
    "CLASSIFIER()",
    "classifier()",
)


def _assert_raises(exception_type, message, function, *args):
    try:
        function(*args)
    except exception_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"{exception_type.__name__} was not raised")


def test_define_rejects_oracle_only_functions():
    for function_sql in ORACLE_ONLY_FUNCTIONS:
        _assert_raises(
            ValueError,
            "only be used in the MEASURES clause",
            ExprParser().parse_define,
            f"DEFINE A AS {function_sql}",
        )


def test_define_rejects_classifier_inside_navigation():
    _assert_raises(
        ValueError,
        "inside an aggregate or navigation operator",
        ExprParser().parse_define,
        "DEFINE A AS PREV(CLASSIFIER()) = 'A'",
    )


def test_measures_accepts_oracle_functions_and_match_number():
    define = ExprParser().parse_define("DEFINE A AS MATCH_NUMBER() > 1")
    assert define.definitions["A"].expression.has_func("MATCH_NUMBER")

    measures = ExprParser().parse_measures(
        "MEASURES CLASSIFIER() AS CLS, MATCH_NUMBER() AS MATCH_NO"
    )
    assert measures.has_func("CLASSIFIER")
    assert measures.has_func("MATCH_NUMBER")


# test cases for condition classfication
def test_scalar_udf_over_current_row_is_independent():
    define = ExprParser().parse_define(
        "DEFINE A AS my_func(A.val) > 5, "
        "B AS built_in_scalar(my_func(B.val), B.type) = 1"
    )

    assert isinstance(define.definitions["A"], IndependentCondition)
    assert isinstance(define.definitions["B"], IndependentCondition)


def test_scalar_udf_with_cross_variable_argument_is_dependent():
    condition = (
        ExprParser()
        .parse_define("DEFINE B AS my_comp(A.val, B.val) = 1")
        .definitions["B"]
    )

    assert isinstance(condition, DependentCondition)
    assert condition.reasons == ["Cross-variable ref: A.val"]


def test_udf_arguments_drive_independent_filter_extraction():
    condition = (
        ExprParser()
        .parse_define("DEFINE B AS my_func(B.val) > 5 " "AND my_comp(A.val, B.val) = 1")
        .definitions["B"]
    )

    assert isinstance(condition, DependentCondition)
    assert condition.get_independent_filter().to_string() == "my_func(B.val) > 5"


def test_udf_wrapping_navigation_function_remains_dependent():
    condition = (
        ExprParser()
        .parse_define("DEFINE A AS my_func(PREV(A.val)) > 5")
        .definitions["A"]
    )

    assert isinstance(condition, DependentCondition)
    assert condition.reasons == ["Function 'PREV'"]


# ----------------------------------------------------------------------
# Pattern analysis: can_be_skipped() and leftmost/rightmost variables
# ----------------------------------------------------------------------
# Each case is (pattern, expected_leftmost, expected_rightmost, can_skip).
CAN_BE_SKIPPED_CASES = (
    ("((A | B?) C A)", {"A", "B", "C"}, {"A"}, False),
    ("(A (B | C?))", {"A"}, {"B", "C", "A"}, False),
    ("((A? | B) C)", {"A", "B", "C"}, {"C"}, False),
    ("((A | B)? C)", {"A", "B", "C"}, {"C"}, False),
    ("A B C", {"A"}, {"C"}, False),
    ("(A | B)", {"A", "B"}, {"A", "B"}, False),
    ("()", set(), set(), True),
    ("A*", {"A"}, {"A"}, True),
    ("A+", {"A"}, {"A"}, False),
    ("A? B?", {"A", "B"}, {"A", "B"}, True),
    ("^(A? B?) C", {"A", "B", "C"}, {"C"}, False),
    ("C (A? B?)$", {"C"}, {"A", "B", "C"}, False),
)


def test_leftmost_rightmost_and_can_be_skipped():
    for pattern, leftmost, rightmost, can_skip in CAN_BE_SKIPPED_CASES:
        node = parse_pattern_string(pattern)
        assert node.get_leftmost_variables() == leftmost
        assert node.get_rightmost_variables() == rightmost
        assert node.can_be_skipped() is can_skip


# ----------------------------------------------------------------------
# Pattern analysis: get_left_max_row_num() / get_right_max_row_num()
# ----------------------------------------------------------------------
# Each case is (pattern, variable, expected_left, expected_right); a None
# count means the variable is not guaranteed to appear or the side is unbounded.
LEFT_RIGHT_MAX_CASES = (
    ("A B C", "B", 1, 1),
    ("A B C B", "B", 1, 0),
    ("(A B C*)", "B", 1, None),
    ("((A+ | B) C D)", "C", None, 1),
    ("(A B C{2,5} D)", "C", 2, 1),
    ("(A B C B D)", "B", 1, 1),
    ("(X PERMUTE(A, B, C) Y)", "B", 3, 3),
    ("PERMUTE(C B D, X, Y, Z)", "B", 4, 4),
    ("PERMUTE(B? C, D B E, X, Y)", "B", 4, 4),
    ("(X A? B C)", "B", 2, 1),
    ("(A B C? D)", "B", 1, 2),
    ("(A+ B C)", "B", None, 1),
    ("(A B C+)", "B", 1, None),
    ("(C A? B A)", "A", 2, 0),
    ("(A B A? C)", "A", 0, 2),
    ("(C (A? | D) B A)", "A", 3, 0),
    ("(A B (A? | D) C)", "A", 0, 3),
    ("(C (A | B*) A)", "A", None, 0),
    ("(A (A | B*) C)", "A", 0, None),
    ("((A? B){2})", "A", None, None),
    ("((B A?){2})", "A", None, None),
    ("(A? C ((B* A) | D))", "A", None, None),
    ("(((A B*) | D) C A?)", "A", None, None),
    ("(A? B)", "A", None, None),
    ("(A | B)", "A", None, None),
    ("(A{0,2} B)", "A", None, None),
    ("(A (B C | B) E)", "B", 1, 2),
    ("(A (B | B C) E)", "B", 1, 2),
    ("(A (B C | B D) E+)", "B", 1, None),
    ("(A B (C | D A) B A)", "B", 1, 1),
    ("(A (B | C) D)", "B", None, None),
    ("(A (B | C) D)", "C", None, None),
    ("(A (B | C D) E)", "C", None, None),
)


def test_left_and_right_max_row_num():
    for pattern, var, expected_left, expected_right in LEFT_RIGHT_MAX_CASES:
        node = parse_pattern_string(pattern)
        assert node.get_left_max_row_num(var) == expected_left
        assert node.get_right_max_row_num(var) == expected_right


# ----------------------------------------------------------------------
# Pattern analysis: get_left_min_row_num() / get_right_min_row_num()
# ----------------------------------------------------------------------
LEFT_RIGHT_MIN_CASES = (
    ("A B C", "B", 1, 1),
    ("((B A)? A)", "A", 0, 0),
    ("(A (A B)?)", "A", 0, 0),
    ("((B B A | C) A)", "A", 1, 0),
    ("(A (A B B | C))", "A", 0, 1),
)


def test_left_and_right_min_row_num_tracks_no_variable_paths():
    for pattern, var, expected_left, expected_right in LEFT_RIGHT_MIN_CASES:
        node = parse_pattern_string(pattern)
        assert node.get_left_min_row_num(var) == expected_left
        assert node.get_right_min_row_num(var) == expected_right


def test_optional_concatenation_does_not_overstate_navigation_safety():
    schema = {
        "T": {
            "columns": {"TS": "LONG", "VAL": "INT"},
            "column_nullables": {"TS": False, "VAL": False},
        }
    }
    cases = (
        ("(B A)? A", "PREV", "left"),
        ("A (A B)?", "NEXT", "right"),
    )
    for pattern, navigation, side in cases:
        sql = f"""
        SELECT * FROM T MATCH_RECOGNIZE(
            ORDER BY TS
            ALL ROWS PER MATCH
            PATTERN({pattern})
            DEFINE A AS {navigation}(A.VAL, 1) > 0, B AS TRUE
        )
        """
        ast = _parse(sql, schema)

        if side == "left":
            assert ast.pattern.get_left_min_row_num("A") == 0
        else:
            assert ast.pattern.get_right_min_row_num("A") == 0
        assert not PatternNavigationReach(ast).all_navigation_stays_within_match()


# ----------------------------------------------------------------------
# Pattern analysis: get_must_appear_var()
# ----------------------------------------------------------------------
# Each case is (pattern, expected_must_appear_variables).
MUST_APPEAR_CASES = (
    ("(A B C)", {"A", "B", "C"}),
    ("(A (B | C) D)", {"A", "D"}),
    ("(A B | A C)", {"A"}),
    ("PERMUTE(A B C)", {"A", "B", "C"}),
    ("(A B* C?)", {"A"}),
    ("({- A -} B C??)", {"A", "B"}),
    ("(^A B C$)", {"A", "B", "C"}),
    ("(A? B C)", {"B", "C"}),
    ("(A B? C)", {"A", "C"}),
    ("(A? B? C)", {"C"}),
    ("(A B C?)", {"A", "B"}),
    ("(A* B+ C?)", {"B"}),
    ("((A | B)* C)", {"C"}),
    ("((A | B)+ C)", {"C"}),
)


def test_must_appear_variables():
    for pattern, expected in MUST_APPEAR_CASES:
        node = parse_pattern_string(pattern)
        assert node.get_must_appear_var() == expected


# ----------------------------------------------------------------------
# Column metadata helpers
# ----------------------------------------------------------------------
def _parse(sql, schema):
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, schema)
    return ast


PARTITION_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS A.TYPE = 'A', C AS C.TYPE = 'C'
)
"""


# ----------------------------------------------------------------------
# Column metadata: schema normalization and lookup
# ----------------------------------------------------------------------
def test_column_metadata_supports_parallel_and_inline_formats():
    ast = RPRAbstractSyntaxTree()
    ast.table_schemas = {
        "ORDERS": {
            "columns": ["ID", "PT"],
            "primary_key": ["ID"],
            "column_types": {"id": "long", "pt": "varchar(20)"},
            "column_nullables": {"PT": "YES"},
        }
    }
    assert ast.get_column_type("orders.id") == "LONG"
    assert ast.is_column_nullable("ID") is False  # implied by PRIMARY KEY
    assert ast.get_column_type("A.PT") == "VARCHAR(20)"  # pattern qualifier
    assert ast.is_column_nullable("pt") is True

    # A database column without a declared type is invalid.
    ast.table_schemas = {"LEGACY": {"columns": ["ID", "VALUE"], "primary_key": ["ID"]}}
    _assert_raises(
        ValueError,
        "must declare a data type",
        ast.get_column_metadata,
        "VALUE",
    )

    ast.table_schemas = {
        "DECLARED": {
            "columns": ["ID", "VALUE"],
            "primary_key": ["ID"],
            "column_types": {"ID": "LONG", "VALUE": "DOUBLE"},
        }
    }
    assert ast.is_column_nullable("ID") is False
    assert ast.is_column_nullable("VALUE") is True

    ast.table_schemas = {
        "EVENTS": {
            "columns": [
                {"name": "TS", "data_type": "long", "nullable": False},
                {"column_name": "PAYLOAD", "type": "char ( 16 )", "is_nullable": "YES"},
            ]
        }
    }
    assert ast._normalize_schema_columns(ast.table_schemas) == {
        "EVENTS": ["TS", "PAYLOAD"]
    }
    assert ast.get_column_type("ts") == "LONG"
    assert ast.get_column_type("payload") == "CHAR(16)"
    assert ast.is_column_nullable("payload") is True


UNSUPPORTED_DATA_TYPES = ("BIGINT", "TEXT", "CHAR", "CHAR(0)", "VARCHAR(-1)")


def test_column_metadata_rejects_unsupported_or_invalid_types():
    for data_type in UNSUPPORTED_DATA_TYPES:
        ast = RPRAbstractSyntaxTree()
        ast.table_schemas = {"T": {"columns": ["C"], "column_types": {"C": data_type}}}
        _assert_raises(
            ValueError,
            "supported types",
            ast.get_column_metadata,
            "C",
        )


def test_all_supported_column_types_are_canonicalized():
    ast = RPRAbstractSyntaxTree()
    ast.table_schemas = {
        "T": {
            "columns": {
                "I": "int",
                "L": "long",
                "F": "float",
                "D": "double",
                "C": "char(8)",
                "V": "varchar ( 32 )",
            }
        }
    }
    assert {
        name: ast.get_column_type(name) for name in ("I", "L", "F", "D", "C", "V")
    } == {
        "I": "INT",
        "L": "LONG",
        "F": "FLOAT",
        "D": "DOUBLE",
        "C": "CHAR(8)",
        "V": "VARCHAR(32)",
    }


# ----------------------------------------------------------------------
# Column metadata: nullable-aware rewriting rules
# ----------------------------------------------------------------------
def test_nullable_partition_rules_generate_null_safe_exists():
    schema = {
        "T": {
            "columns": ["PT", "TS", "TYPE"],
            "column_types": {
                "PT": "VARCHAR(20)",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
            },
            "column_nullables": {"PT": True, "TS": False},
        }
    }

    sequence_ast = _parse(PARTITION_SQL, schema)
    assert SequenceFilteringRule().apply(sequence_ast)
    assert "WHERE EXISTS" in sequence_ast.input_table
    assert "RPR_INPUT.PT IS NULL" in sequence_ast.input_table
    assert "RPR_SEQUENCE_ROWS.PT IS NULL" in sequence_ast.input_table

    must_ast = _parse(PARTITION_SQL, schema)
    assert MustAppearPartitionFilterPushdownRule().apply(must_ast)
    assert "WHERE EXISTS" in must_ast.input_table
    assert "RPR_INPUT.PT IS NULL" in must_ast.input_table

    anchor_ast = _parse(PARTITION_SQL.replace("PATTERN(A C)", "PATTERN(^A C$)"), schema)
    assert AnchorPartitionSizeFilterRule().apply(anchor_ast)
    assert "WHERE EXISTS" in anchor_ast.input_table
    assert "RPR_INPUT.PT IS NULL" in anchor_ast.input_table


def test_sequence_filtering_treats_outer_join_keys_as_nullable():
    schema = {
        "T": {
            "columns": ["ID", "TS", "TYPE"],
            "column_types": {
                "ID": "INT",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
            },
            "column_nullables": {
                "ID": False,
                "TS": False,
                "TYPE": False,
            },
        },
        "U": {
            "columns": ["U_ID", "K"],
            "column_types": {"U_ID": "INT", "K": "INT"},
            "column_nullables": {"U_ID": False, "K": False},
        },
    }
    sql = """
SELECT * FROM T LEFT JOIN U ON T.ID = U.U_ID MATCH_RECOGNIZE(
    PARTITION BY K ORDER BY TS ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""

    ast = _parse(sql, schema)
    assert ast.is_column_nullable("K") is False
    assert SequenceFilteringRule().apply(ast)
    assert "WHERE EXISTS" in ast.input_table
    assert "RPR_INPUT.K IS NULL" in ast.input_table
    assert "RPR_SEQUENCE_ROWS.K IS NULL" in ast.input_table
    assert " K IN (" not in ast.input_table
    assert (
        "FROM (SELECT * FROM T LEFT JOIN U ON T.ID = U.U_ID) RPR_INPUT"
        in ast.input_table
    )


def test_sequence_filtering_preserves_simple_input_qualifier():
    for nullable in (False, True):
        schema = {
            "T": {
                "columns": ["PT", "TS", "TYPE"],
                "column_types": {
                    "PT": "INT",
                    "TS": "LONG",
                    "TYPE": "VARCHAR(20)",
                },
                "column_nullables": {
                    "PT": nullable,
                    "TS": False,
                    "TYPE": False,
                },
            }
        }
        sql = """
SELECT * FROM T X MATCH_RECOGNIZE(
    PARTITION BY X.PT ORDER BY X.TS ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""

        ast = _parse(sql, schema)
        assert SequenceFilteringRule().apply(ast)
        assert ast.input_table.endswith(") X")
        assert "PARTITION BY X.PT" in ast.to_string()
        assert "ORDER BY X.TS" in ast.to_string()
        if nullable:
            assert "FROM (SELECT * FROM T X) RPR_INPUT" in ast.input_table
            assert "FROM T X RPR_INPUT" not in ast.input_table
        else:
            assert "FROM T X WHERE PT IN" in ast.input_table


def test_sequence_filtering_rejects_qualified_join_members():
    schema = {
        "T": {
            "columns": ["ID", "PT", "TS", "TYPE"],
            "column_types": {
                "ID": "INT",
                "PT": "INT",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
            },
            "column_nullables": {
                "ID": False,
                "PT": False,
                "TS": False,
                "TYPE": False,
            },
        },
        "U": {
            "columns": ["U_ID"],
            "column_types": {"U_ID": "INT"},
            "column_nullables": {"U_ID": False},
        },
    }
    sql = """
SELECT * FROM T JOIN U ON T.ID = U.U_ID MATCH_RECOGNIZE(
    PARTITION BY T.PT ORDER BY T.TS ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TYPE = 'A'
)
"""

    ast = _parse(sql, schema)
    assert not SequenceFilteringRule().is_applicable(ast)
    assert not SequenceFilteringRule().apply(ast)


def test_must_appear_partition_filter_safely_aliases_complex_inputs():
    schema = {
        "T": {
            "columns": ["ID", "PT", "TS", "TYPE"],
            "column_types": {
                "ID": "INT",
                "PT": "INT",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
            },
            "column_nullables": {
                "ID": False,
                "PT": True,
                "TS": False,
                "TYPE": False,
            },
        },
        "U": {
            "columns": ["U_ID", "NOTE"],
            "column_types": {"U_ID": "INT", "NOTE": "VARCHAR(20)"},
            "column_nullables": {"U_ID": False, "NOTE": True},
        },
    }
    inputs = (
        ("T X", False),
        ("(SELECT ID, PT, TS, TYPE FROM T) X", False),
        ("T LEFT JOIN U ON T.ID = U.U_ID", True),
    )

    for input_sql, should_be_rejected in inputs:
        sql = f"""
SELECT * FROM {input_sql} MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)
"""
        if should_be_rejected:
            _assert_raises(
                ValueError,
                "wrap joined tables in a SELECT query",
                _parse,
                sql,
                schema,
            )
            continue

        ast = _parse(sql, schema)
        assert ast.is_column_nullable("PT") is True
        assert MustAppearPartitionFilterPushdownRule().apply(ast)
        assert f"FROM (SELECT * FROM {input_sql}) RPR_INPUT" in ast.input_table
        assert f"FROM {input_sql} RPR_INPUT" not in ast.input_table


def test_derived_case_partition_nullability_is_inferred_and_filtered_null_safely():
    schema = {
        "T": {
            "columns": ["PT", "TS"],
            "column_types": {"PT": "INT", "TS": "LONG"},
            "column_nullables": {"PT": False, "TS": False},
        }
    }
    sql = """
SELECT * FROM (
    SELECT CASE WHEN PT = 0 THEN NULL ELSE PT END AS K, TS FROM T
) MATCH_RECOGNIZE(
    PARTITION BY K ORDER BY TS ALL ROWS PER MATCH
    PATTERN(^A{2}$)
    DEFINE A AS TRUE
)
"""

    ast = _parse(sql, schema)
    assert ast.get_column_type("K") == "INT"
    assert ast.is_column_nullable("K") is True
    assert AnchorPartitionSizeFilterRule().apply(ast)
    assert "WHERE EXISTS" in ast.input_table
    assert "RPR_INPUT.K IS NULL" in ast.input_table
    assert " K IN (" not in ast.input_table


def test_unknown_partition_nullability_uses_null_safe_filter():
    schema = {
        "T": {
            "columns": ["PT", "TS"],
            "column_types": {"PT": "INT", "TS": "LONG"},
            "column_nullables": {"PT": False, "TS": False},
        }
    }
    sql = """
SELECT * FROM (
    SELECT UNKNOWN_NULLABILITY(PT) AS K, TS FROM T
) MATCH_RECOGNIZE(
    PARTITION BY K ORDER BY TS ALL ROWS PER MATCH
    PATTERN(^A{2}$)
    DEFINE A AS TRUE
)
"""

    ast = _parse(sql, schema)
    assert ast.is_column_nullable("K") is None
    assert AnchorPartitionSizeFilterRule().apply(ast)
    assert "WHERE EXISTS" in ast.input_table
    assert "RPR_INPUT.K IS NULL" in ast.input_table


def test_anchor_partition_size_safely_aliases_complex_inputs():
    schema = {
        "T": {
            "columns": ["ID", "PT", "TS", "TYPE"],
            "column_types": {
                "ID": "INT",
                "PT": "INT",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
            },
            "column_nullables": {
                "ID": False,
                "PT": True,
                "TS": False,
                "TYPE": False,
            },
        },
        "U": {
            "columns": ["U_ID", "NOTE"],
            "column_types": {"U_ID": "INT", "NOTE": "VARCHAR(20)"},
            "column_nullables": {"U_ID": False, "NOTE": True},
        },
    }
    inputs = (
        ("T X", False),
        ("(SELECT ID, PT, TS, TYPE FROM T) X", False),
        ("T LEFT JOIN U ON T.ID = U.U_ID", True),
    )

    for input_sql, should_be_rejected in inputs:
        sql = f"""
SELECT * FROM {input_sql} MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS ALL ROWS PER MATCH
    PATTERN(^A{{2}}$)
    DEFINE A AS TRUE
)
"""
        if should_be_rejected:
            _assert_raises(
                ValueError,
                "wrap joined tables in a SELECT query",
                _parse,
                sql,
                schema,
            )
            continue

        ast = _parse(sql, schema)
        assert ast.is_column_nullable("PT") is not False
        assert AnchorPartitionSizeFilterRule().apply(ast)
        assert f"FROM (SELECT * FROM {input_sql}) RPR_INPUT" in ast.input_table
        assert f"FROM {input_sql} RPR_INPUT" not in ast.input_table


def test_derived_expression_can_reuse_its_source_column_name():
    schema = {
        "T": {
            "columns": ["VAL", "TS"],
            "column_types": {"VAL": "DOUBLE", "TS": "LONG"},
            "column_nullables": {"VAL": False, "TS": False},
        }
    }
    sql = """
SELECT * FROM (
    SELECT VAL * 2 AS VAL, TS FROM T
) MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS VAL > 0
)
"""

    ast = _parse(sql, schema)
    assert ast.get_column_type("VAL") == "DOUBLE"
    assert ast.is_column_nullable("VAL") is False


ORDER_BY_TS_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TS > 100, B AS B.TYPE = 'B'
)
"""
ORDER_BY_TS_SCHEMA = {
    "T": {
        "columns": ["TS", "TYPE"],
        "column_types": {"TS": "LONG", "TYPE": "VARCHAR(20)"},
    }
}


def _order_by_ts_ast(ts_nullable):
    schema = {table: dict(metadata) for table, metadata in ORDER_BY_TS_SCHEMA.items()}
    schema["T"]["column_nullables"] = {"TS": ts_nullable}
    return _parse(ORDER_BY_TS_SQL, schema)


def test_nullable_order_column_blocks_range_pushdown():
    assert not OrderedRangePushdownRule().is_applicable(_order_by_ts_ast(True))
    assert OrderedRangePushdownRule().is_applicable(_order_by_ts_ast(False))


def test_unified_ordered_range_rule_selects_endpoint_and_middle_candidates():
    endpoint_ast = _order_by_ts_ast(False)
    endpoint_rule = OrderedRangePushdownRule()
    endpoint_plan = endpoint_rule._build_plan(endpoint_ast)
    assert endpoint_plan is not None
    assert endpoint_plan.origin == "endpoint"
    assert endpoint_rule.apply(endpoint_ast)
    assert "WHERE TS > 100" in endpoint_ast.input_table

    middle_sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS A.TYPE = 'A', B AS B.TS > 100, C AS C.TYPE = 'C'
)
"""
    middle_ast = _parse(
        middle_sql,
        _order_by_ts_ast(False).table_schemas,
    )
    middle_rule = OrderedRangePushdownRule()
    middle_plan = middle_rule._build_plan(middle_ast)
    assert middle_plan is not None
    assert middle_plan.origin == "middle"
    assert len(middle_plan.tails) == 1
    assert middle_rule.apply(middle_ast)
    assert "UNION ALL" in middle_ast.input_table


def test_unified_ordered_range_rule_uses_one_deterministic_candidate():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS A.TS > 50, B AS B.TS > 100, C AS C.TYPE = 'C'
)
"""
    plan = OrderedRangePushdownRule()._build_plan(
        _parse(sql, _order_by_ts_ast(False).table_schemas)
    )
    assert plan is not None
    assert plan.origin == "endpoint+middle"
    assert plan.keep_expr.to_string() == "TS >= 101"
    assert len(plan.tails) == 1
    assert plan.tails[0].expr.to_string() == "(TS BETWEEN 51 AND 100)"


def test_unknown_order_column_nullability_blocks_range_pushdown():
    ast = _order_by_ts_ast(False)
    ast.is_column_nullable = lambda _column: None

    assert not OrderedRangePushdownRule().is_applicable(ast)


def test_outer_join_input_must_be_wrapped():
    schema = {
        "L": {
            "columns": ["ID", "KIND"],
            "column_types": {"ID": "INT", "KIND": "VARCHAR(10)"},
            "column_nullables": {"ID": False, "KIND": False},
        },
        "R": {
            "columns": ["RID", "TS"],
            "column_types": {"RID": "INT", "TS": "LONG"},
            "column_nullables": {"RID": False, "TS": False},
        },
    }
    sql = """
SELECT * FROM L LEFT JOIN R ON L.ID = R.RID MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TS > 10, B AS B.KIND = 'B'
)
"""
    _assert_raises(
        ValueError,
        "wrap joined tables in a SELECT query",
        _parse,
        sql,
        schema,
    )


def test_label_prefilter_uses_declared_types_as_width_estimates():
    ast = RPRAbstractSyntaxTree()
    ast.input_table = "T"
    ast.table_schemas = {
        "T": {
            "columns": ["ID", "DESCRIPTION", "EXACT"],
            "column_types": {
                "ID": "INT",
                "DESCRIPTION": "VARCHAR(80)",
                "EXACT": "LONG",
            },
            "column_avg_sizes": {"EXACT": 3},
        }
    }
    sizes = ColumnPruningRule()._get_column_avg_sizes(ast)
    assert sizes == {"EXACT": 3.0, "ID": 4.0, "DESCRIPTION": 80.0}


DEFINE_TS_OR_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS ALL ROWS PER MATCH
    PATTERN(A)
    DEFINE A AS A.TS > 200 OR A.TS < 201
)
"""
DEFINE_TS_SCHEMA = {
    "T": {
        "columns": {"TS": "INT"},
        "column_nullables": {"TS": False},
    }
}


def _parse_define_ts(sql):
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, DEFINE_TS_SCHEMA)
    return ast.define.to_string()


def test_define_simplification_uses_schema_type_and_nullability():
    assert _parse_define_ts(DEFINE_TS_OR_SQL) == "DEFINE A AS 1 = 1"

    DEFINE_TS_SCHEMA["T"]["column_nullables"]["TS"] = True
    assert _parse_define_ts(DEFINE_TS_OR_SQL) == "DEFINE A AS TS IS NOT NULL"

    empty_integer_range_sql = DEFINE_TS_OR_SQL.replace(
        "A.TS > 200 OR A.TS < 201",
        "A.TS > 300 AND A.TS < 301",
    )
    assert _parse_define_ts(empty_integer_range_sql) == "DEFINE A AS 1 != 1"


WHERE_PT_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    PATTERN(A)
    DEFINE A AS TRUE
)
WHERE PT > 200 OR PT < 201
"""
WHERE_PT_SCHEMA = {
    "T": {
        "columns": {"PT": "INT", "TS": "LONG"},
        "column_nullables": {"PT": False, "TS": False},
    }
}


def _where_pt_to_string(pt_nullable):
    schema = {table: dict(metadata) for table, metadata in WHERE_PT_SCHEMA.items()}
    schema["T"]["column_nullables"] = {"PT": pt_nullable, "TS": False}
    return _parse(WHERE_PT_SQL, schema).where.to_string()


def test_trailing_where_simplification_uses_match_output_metadata():
    assert _where_pt_to_string(False) == "1 = 1"
    assert _where_pt_to_string(True) == "(PT IS NOT NULL)"


MATCH_ID_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS MATCH_ID
    PATTERN(A)
    DEFINE A AS TRUE
) AS MR
WHERE MR.MATCH_ID > 200 OR MR.MATCH_ID < 201
"""
MATCH_ID_SCHEMA = {
    "T": {
        "columns": {"ID": "INT", "TS": "LONG"},
        "column_nullables": {"ID": False, "TS": False},
    }
}


def test_trailing_where_infers_measure_metadata_conservatively():
    required_measure_ast = _parse(MATCH_ID_SQL, MATCH_ID_SCHEMA)
    assert required_measure_ast.where.to_string() == "1 = 1"

    optional_measure_sql = MATCH_ID_SQL.replace("PATTERN(A)", "PATTERN(A? B)").replace(
        "DEFINE A AS TRUE", "DEFINE A AS TRUE, B AS TRUE"
    )
    optional_measure_ast = _parse(optional_measure_sql, MATCH_ID_SCHEMA)
    assert optional_measure_ast.where.to_string() == "(MR.MATCH_ID IS NOT NULL)"


if __name__ == "__main__":
    tests = (
        test_define_rejects_oracle_only_functions,
        test_define_rejects_classifier_inside_navigation,
        test_measures_accepts_oracle_functions_and_match_number,
        test_scalar_udf_over_current_row_is_independent,
        test_scalar_udf_with_cross_variable_argument_is_dependent,
        test_udf_arguments_drive_independent_filter_extraction,
        test_udf_wrapping_navigation_function_remains_dependent,
        test_leftmost_rightmost_and_can_be_skipped,
        test_left_and_right_max_row_num,
        test_left_and_right_min_row_num_tracks_no_variable_paths,
        test_optional_concatenation_does_not_overstate_navigation_safety,
        test_must_appear_variables,
        test_column_metadata_supports_parallel_and_inline_formats,
        test_column_metadata_rejects_unsupported_or_invalid_types,
        test_all_supported_column_types_are_canonicalized,
        test_nullable_partition_rules_generate_null_safe_exists,
        test_must_appear_partition_filter_safely_aliases_complex_inputs,
        test_nullable_order_column_blocks_range_pushdown,
        test_unified_ordered_range_rule_selects_endpoint_and_middle_candidates,
        test_unified_ordered_range_rule_uses_one_deterministic_candidate,
        test_unknown_order_column_nullability_blocks_range_pushdown,
        test_outer_join_input_must_be_wrapped,
        test_label_prefilter_uses_declared_types_as_width_estimates,
        test_define_simplification_uses_schema_type_and_nullability,
        test_trailing_where_simplification_uses_match_output_metadata,
        test_trailing_where_infers_measure_metadata_conservatively,
    )
    for test in tests:
        test()
    print("All basic function tests passed.")
