"""Tests for expression / predicate simplification."""

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from clause_parser import ExprParser
from expression import ColumnRef, Literal
from expression_simplifier import (
    ExpressionColumnMetadata,
    RangeConstraint,
    RangeDomain,
    SimplifyContext,
)

TEST_COLUMN_METADATA = {
    "TS": ExpressionColumnMetadata("LONG", True),
    "OPEN": ExpressionColumnMetadata("DOUBLE", True),
    "CLOSE": ExpressionColumnMetadata("DOUBLE", True),
    "PRICE": ExpressionColumnMetadata("DOUBLE", True),
    "VOL": ExpressionColumnMetadata("DOUBLE", True),
    "X": ExpressionColumnMetadata("DOUBLE", True),
    "Y": ExpressionColumnMetadata("DOUBLE", True),
    "TYPE": ExpressionColumnMetadata("VARCHAR(32)", True),
}

INT_TS = {"TS": ExpressionColumnMetadata("INT", True)}
DOUBLE_TS = {"TS": ExpressionColumnMetadata("DOUBLE", True)}
TYPE_TS = {
    "TS": ExpressionColumnMetadata("INT", True),
    "TYPE": ExpressionColumnMetadata("VARCHAR(20)", True),
}
TYPE_ONLY = {"TYPE": ExpressionColumnMetadata("VARCHAR(20)", True)}


def _make_parser(column_metadata):
    metadata = TEST_COLUMN_METADATA if column_metadata is None else column_metadata
    context = SimplifyContext(lambda column: metadata.get(column.name.upper()))
    return ExprParser(context)


def _simplify_define(define_sql, column_metadata=None):
    parser = _make_parser(column_metadata)
    define = parser.parse_define(define_sql)
    return define.to_string()


def _simplify_where(where_sql):
    parser = _make_parser(None)
    where = parser.parse_where(where_sql)
    return "WHERE " + where.to_string()


def _assert_raises(exception_type, message, function, *args):
    try:
        function(*args)
    except exception_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"{exception_type.__name__} was not raised")


def test_range_constraint_bound_and_satisfiability():
    domain = RangeDomain.CONTINUOUS
    unbounded = RangeConstraint(domain, ColumnRef(None, "price"))
    assert not unbounded.has_bound()
    assert unbounded.is_satisfiable()

    assert RangeConstraint(domain, lower=1).has_bound()
    assert RangeConstraint(domain, lower=1, upper=2).is_satisfiable()
    assert RangeConstraint(domain, lower=1, upper=1).is_satisfiable()
    assert not RangeConstraint(
        domain, lower=1, lower_inclusive=False, upper=1
    ).is_satisfiable()
    assert not RangeConstraint(domain, lower=2, upper=1).is_satisfiable()
    assert not RangeConstraint(domain, lower=1, upper="1").is_satisfiable()


def test_range_constraint_domain_normalization():
    _assert_raises(
        TypeError, "missing 1 required positional argument: 'domain'", RangeConstraint
    )
    assert RangeDomain.from_data_type("int") is RangeDomain.INTEGER
    assert RangeDomain.from_data_type("LONG") is RangeDomain.INTEGER
    assert RangeDomain.from_data_type("double") is RangeDomain.CONTINUOUS
    assert RangeDomain.from_data_type("VARCHAR(20)") is RangeDomain.TEXT
    _assert_raises(
        ValueError, "a data type is required", RangeDomain.from_data_type, None
    )
    _assert_raises(
        ValueError, "unsupported range data type", RangeDomain.from_data_type, "BIGINT"
    )

    constraint = RangeConstraint(
        RangeDomain.INTEGER,
        lower=1.2,
        lower_inclusive=False,
        upper=3.8,
        upper_inclusive=False,
    )
    assert (constraint.lower, constraint.lower_inclusive) == (1.2, False)
    assert (constraint.upper, constraint.upper_inclusive) == (3.8, False)
    assert constraint.is_satisfiable()

    constraint.apply_lower(3.2, False)
    assert (constraint.lower, constraint.lower_inclusive) == (3.2, False)
    assert not constraint.is_satisfiable()

    column = ColumnRef(None, "value")
    canonical = RangeConstraint(
        RangeDomain.INTEGER, column, lower=1.2, lower_inclusive=False
    ).intersect(
        RangeConstraint(RangeDomain.INTEGER, column, upper=3.8, upper_inclusive=False)
    )
    assert canonical == RangeConstraint(RangeDomain.INTEGER, column, lower=2, upper=3)


def test_range_constraint_apply_boundaries():
    constraint = RangeConstraint(RangeDomain.CONTINUOUS)

    constraint.apply_lower(1, True)
    constraint.apply_lower(0, False)
    assert (constraint.lower, constraint.lower_inclusive) == (1, True)
    constraint.apply_lower(Literal(1), False)
    assert (constraint.lower, constraint.lower_inclusive) == (1, False)

    constraint.apply_upper(10, True)
    constraint.apply_upper(11, False)
    assert (constraint.upper, constraint.upper_inclusive) == (10, True)
    constraint.apply_upper(Literal(10), False)
    assert (constraint.upper, constraint.upper_inclusive) == (10, False)


def test_range_constraint_to_expr():
    price = ColumnRef(None, "price")
    domain = RangeDomain.CONTINUOUS

    assert RangeConstraint(domain, price).to_expr() is None
    assert RangeConstraint(domain, lower=1).to_expr() is None
    assert RangeConstraint(domain, lower=1).to_expr("price").to_string() == "price >= 1"
    assert (
        RangeConstraint(domain, price, 5, True, 5, True).to_expr().to_string()
        == "price = 5"
    )
    assert (
        RangeConstraint(domain, price, 1, True, 5, True).to_expr().to_string()
        == "(price BETWEEN 1 AND 5)"
    )
    assert (
        RangeConstraint(domain, price, 1, False, 5, True).to_expr().to_string()
        == "(price > 1 AND price <= 5)"
    )
    assert (
        RangeConstraint(domain, price, 2, True, 1, True).to_expr().to_string()
        == "1 != 1"
    )


def test_range_constraint_intersection():
    price = ColumnRef(None, "price")
    other_column = ColumnRef(None, "volume")
    domain = RangeDomain.CONTINUOUS

    result = RangeConstraint(domain, price, 1, True, 10, True).intersect(
        RangeConstraint(domain, price, 5, False, 15, True)
    )
    assert result == RangeConstraint(domain, price, 5, False, 10, True)
    assert (
        RangeConstraint(domain, price, 1, True, 2, True).intersect(
            RangeConstraint(domain, price, 3, True, 4, True)
        )
        is None
    )
    assert (
        RangeConstraint(domain, price, 1, True, 2, True).intersect(
            RangeConstraint(domain, other_column, 1, True, 2, True)
        )
        is None
    )


def test_range_constraint_union():
    price = ColumnRef(None, "price")
    other_column = ColumnRef(None, "volume")

    result = RangeConstraint(RangeDomain.CONTINUOUS, price, 1, True, 5, True).union(
        RangeConstraint(RangeDomain.CONTINUOUS, price, 5, False, 10, True)
    )
    assert result == RangeConstraint(RangeDomain.CONTINUOUS, price, 1, True, 10, True)

    left = RangeConstraint(RangeDomain.INTEGER, price, 1, True, 2, True)
    right = RangeConstraint(RangeDomain.INTEGER, price, 3, True, 4, True)
    expected = RangeConstraint(RangeDomain.INTEGER, price, 1, True, 4, True)
    continuous_right = RangeConstraint(RangeDomain.CONTINUOUS, price, 3, True, 4, True)

    assert left.union(right) == expected
    assert left.union(continuous_right) is None
    assert (
        RangeConstraint(RangeDomain.CONTINUOUS, price, 1, True, 2, True).union(
            RangeConstraint(RangeDomain.CONTINUOUS, price, 3, True, 4, True)
        )
        is None
    )
    assert (
        left.union(RangeConstraint(RangeDomain.INTEGER, other_column, 1, True, 2, True))
        is None
    )


def test_missing_metadata_rejected():
    parser = _make_parser({})
    _assert_raises(
        ValueError,
        "missing schema metadata for column 'ts'",
        parser.parse_define,
        "DEFINE A AS ts > 200 OR ts < 201",
    )


# Fractional and adjacent bounds depend on whether
# the column domain is discrete or continuous.
# test format: (define clause, has_overlap, data_type)
OVERLAP_CASES = (
    ("DEFINE A AS ts > 1.2, B AS ts < 2", False, INT_TS),
    ("DEFINE A AS ts > 1.2, B AS ts < 2", True, DOUBLE_TS),
    ("DEFINE A AS ts > 1, B AS ts < 2", False, INT_TS),
    ("DEFINE A AS ts > 1, B AS ts < 2", True, DOUBLE_TS),
    # Inclusive / exclusive boundaries and ordinary range intersections.
    ("DEFINE A AS ts <= 10, B AS ts >= 10", True, INT_TS),
    ("DEFINE A AS ts < 10, B AS ts >= 10", False, INT_TS),
    ("DEFINE A AS ts BETWEEN 10 AND 20, B AS ts > 15", True, INT_TS),
    ("DEFINE A AS ts BETWEEN 10 AND 20, B AS ts > 20", False, INT_TS),
    # OR branches overlap if any conjunction is satisfiable.
    ("DEFINE A AS ts < 0 OR ts > 10, B AS ts BETWEEN 5 AND 12", True, INT_TS),
    ("DEFINE A AS ts < 0 OR ts > 10, B AS ts BETWEEN 1 AND 9", False, INT_TS),
    # Set constraints, exclusions and string-valued columns.
    ("DEFINE A AS type IN ('A', 'B'), B AS type IN ('B', 'C')", True, TYPE_ONLY),
    ("DEFINE A AS type IN ('A', 'B'), B AS type IN ('C', 'D')", False, TYPE_ONLY),
    ("DEFINE A AS type NOT IN ('A', 'B'), B AS type = 'A'", False, TYPE_ONLY),
    ("DEFINE A AS ts IN (1, 2, 3, 4, 5), B AS ts > 4.5", True, DOUBLE_TS),
    ("DEFINE A AS ts IN (1, 2, 3, 4, 5), B AS ts > 5", False, INT_TS),
    ("DEFINE A AS ts IN (1, 2, 3, 4, 5), B AS ts = 6 OR ts = 1", True, INT_TS),
    ("DEFINE A AS ts IN (1, 2, 3, 4, 5), B AS ts = 6 OR ts = 7", False, INT_TS),
    # SQL NULL predicates conflict with non-NULL comparisons on the same
    # column, while unrelated column predicates may coexist.
    ("DEFINE A AS ts IS NULL, B AS ts IS NOT NULL", False, INT_TS),
    ("DEFINE A AS ts IS NULL, B AS ts > 0", False, INT_TS),
    ("DEFINE A AS ts IS NULL, B AS type = 'A'", True, TYPE_TS),
    # An internally contradictory variable never overlaps another variable.
    ("DEFINE A AS ts > 10 AND ts < 5, B AS ts > 0", False, INT_TS),
    # If no independent filter can be extracted, overlap is conservative;
    # it is false only when the other independent filter is itself impossible.
    ("DEFINE A AS PREV(A.ts) > 0, B AS ts > 10", True, INT_TS),
    ("DEFINE A AS PREV(A.ts) > 0, B AS ts > 10 AND ts < 5", False, INT_TS),
)


def test_expr_overlap():
    for define_sql, expected, metadata in OVERLAP_CASES:
        parser = _make_parser(metadata)
        define = parser.parse_define(define_sql)
        actual = define.has_expr_overlap(
            define.get_independent_expr("A"),
            define.get_independent_expr("B"),
            parser.simplify_context,
        )
        assert actual == expected, (
            f"overlap mismatch for A-B: expected {expected}, got {actual}\n"
            f"SQL: {define_sql}"
        )


# Type- and nullability-aware range simplification.
RANGE_SIMPLIFY_CASES = (
    (
        "DEFINE A AS ts > 300 AND ts < 301",
        "DEFINE A AS 1 != 1",
        {"TS": ExpressionColumnMetadata("INT", True)},
    ),
    (
        "DEFINE A AS ts > 300 AND ts < 301",
        "DEFINE A AS ts > 300 AND ts < 301",
        {"TS": ExpressionColumnMetadata("FLOAT", True)},
    ),
    (
        "DEFINE A AS ts > 300 AND ts < 301",
        "DEFINE A AS ts > 300 AND ts < 301",
        {"TS": ExpressionColumnMetadata("DOUBLE", False)},
    ),
    (
        "DEFINE A AS ts > 200 OR ts < 201",
        "DEFINE A AS 1 = 1",
        {"TS": ExpressionColumnMetadata("INT", False)},
    ),
    (
        "DEFINE A AS ts > 200 OR ts < 201",
        "DEFINE A AS ts IS NOT NULL",
        {"TS": ExpressionColumnMetadata("INT", True)},
    ),
    (
        "DEFINE A AS ts > 200 OR ts < 201",
        "DEFINE A AS ts IS NOT NULL",
        {"TS": ExpressionColumnMetadata("DOUBLE", True)},
    ),
    (
        "DEFINE A AS CASE WHEN ts > 5 THEN 10 ELSE 20 END = 20",
        "DEFINE A AS (ts IS NULL) OR ts <= 5",
        {"TS": ExpressionColumnMetadata("INT", True)},
    ),
    (
        "DEFINE A AS CASE WHEN ts > 5 THEN 10 ELSE 20 END = 20",
        "DEFINE A AS ts <= 5",
        {"TS": ExpressionColumnMetadata("INT", False)},
    ),
)


def test_range_aware_define_simplification():
    for define_sql, expected, metadata in RANGE_SIMPLIFY_CASES:
        assert _simplify_define(define_sql, metadata) == expected


# Literal and linear arithmetic simplification (default nullable schema).
ARITHMETIC_SIMPLIFY_CASES = (
    ("DEFINE A AS 2 > 1", "DEFINE A AS 1 = 1"),
    ("DEFINE A AS open * 2 > 10", "DEFINE A AS open > 5"),
    ("DEFINE A AS open * 2 + 2 > 6", "DEFINE A AS open > 2"),
    ("DEFINE A AS 2 * open + 2 > 6", "DEFINE A AS open > 2"),
    ("DEFINE A AS open / 2 >= 3", "DEFINE A AS open >= 6"),
    ("DEFINE A AS open * -2 > 10", "DEFINE A AS open < -5"),
    ("DEFINE A AS open + open > 8", "DEFINE A AS open > 4"),
    ("DEFINE A AS open - 2 = 8", "DEFINE A AS open = 10"),
    ("DEFINE A AS 10 < open * 2", "DEFINE A AS open > 5"),
    ("DEFINE A AS PREV(A.PRICE, 0) > 10", "DEFINE A AS PRICE > 10"),
)


def test_arithmetic_define_simplification():
    for define_sql, expected in ARITHMETIC_SIMPLIFY_CASES:
        assert _simplify_define(define_sql) == expected


# Logical identities, range merging, contradiction, and absorption.
LOGIC_SIMPLIFY_CASES = (
    ("DEFINE A AS ts > 300 OR ts > 100", "DEFINE A AS ts >= 101"),
    ("DEFINE A AS ts > 300 AND ts < 100", "DEFINE A AS 1 != 1"),
    ("DEFINE A AS ts > 300 AND ts > 100", "DEFINE A AS ts >= 301"),
    ("DEFINE A AS 1 = 1 OR ts > 100", "DEFINE A AS 1 = 1"),
    ("DEFINE A AS 1 != 1 AND ts > 100", "DEFINE A AS 1 != 1"),
    ("DEFINE A AS ts > 100 OR ts > 100", "DEFINE A AS ts > 100"),
    ("DEFINE A AS type = 'A' AND type = 'A'", "DEFINE A AS type = 'A'"),
    ("DEFINE A AS NOT(ts > 100) AND ts > 100", "DEFINE A AS 1 != 1"),
    ("DEFINE A AS NOT(ts > 100) OR ts > 100", "DEFINE A AS ts IS NOT NULL"),
    (
        "DEFINE A AS (type = 'B' AND type != 'B') OR type = 'C'",
        "DEFINE A AS type = 'C'",
    ),
    ("DEFINE A AS ts > 100 OR (ts > 100 AND type = 'A')", "DEFINE A AS ts > 100"),
    ("DEFINE A AS ts > 100 AND (ts > 100 OR type = 'A')", "DEFINE A AS ts > 100"),
    (
        "DEFINE A AS (ts > 100 AND type = 'A') OR (ts > 100 AND NOT(type = 'A'))",
        "DEFINE A AS ts > 100 AND (type IS NOT NULL)",
    ),
    (
        "DEFINE A AS (ts > 100 OR type = 'A') AND (ts > 100 OR NOT(type = 'A'))",
        "DEFINE A AS ts > 100",
    ),
    ("DEFINE A AS ts >= 100 AND ts <= 200", "DEFINE A AS ts BETWEEN 100 AND 200"),
    ("DEFINE A AS ts < 100 OR ts >= 100", "DEFINE A AS ts IS NOT NULL"),
    (
        "DEFINE A AS ts BETWEEN 100 AND 200 AND ts > 150",
        "DEFINE A AS ts BETWEEN 151 AND 200",
    ),
)


def test_logic_define_simplification():
    for define_sql, expected in LOGIC_SIMPLIFY_CASES:
        assert _simplify_define(define_sql) == expected


# CASE expressions with literal results, including strings and implicit
# NULL ELSE branches.
CASE_LITERAL_SIMPLIFY_CASES = (
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 1 ELSE 0 END = 1", "DEFINE A AS VOL > 5"),
    (
        "DEFINE C AS CASE WHEN COUNT(A.*) = 1 AND C.PRICE > 100 THEN 1 WHEN COUNT(B.*) = 1 AND C.PRICE < 100 THEN 1 ELSE 0 END = 1",
        "DEFINE C AS (COUNT(A.*) = 1 AND PRICE > 100) OR (COUNT(B.*) = 1 AND PRICE < 100)",
    ),
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 1 ELSE 0 END > 1", "DEFINE A AS 1 != 1"),
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 1 ELSE 0 END >= 0", "DEFINE A AS 1 = 1"),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN 10 ELSE 20 END = 20",
        "DEFINE A AS (VOL IS NULL) OR VOL <= 5",
    ),
    (
        "DEFINE A AS CASE WHEN X > 0 THEN 10 WHEN Y > 0 THEN 20 ELSE 30 END = 20",
        "DEFINE A AS ((X IS NULL) OR X <= 0) AND Y > 0",
    ),
    (
        "DEFINE A AS 15 < CASE WHEN VOL > 5 THEN 10 ELSE 20 END",
        "DEFINE A AS (VOL IS NULL) OR VOL <= 5",
    ),
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 1 END = 1", "DEFINE A AS VOL > 5"),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN 'HIGH' ELSE 'LOW' END = 'HIGH'",
        "DEFINE A AS VOL > 5",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN 'HIGH' ELSE 'LOW' END != 'HIGH'",
        "DEFINE A AS (VOL IS NULL) OR VOL <= 5",
    ),
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 10 ELSE 20 END <= 20", "DEFINE A AS 1 = 1"),
    ("DEFINE A AS CASE WHEN VOL > 5 THEN 10 ELSE 20 END < 10", "DEFINE A AS 1 != 1"),
)


def test_case_literal_define_simplification():
    for define_sql, expected in CASE_LITERAL_SIMPLIFY_CASES:
        assert _simplify_define(define_sql) == expected


# CASE expressions with column, function, aggregate and arithmetic results.
CASE_EXPR_SIMPLIFY_CASES = (
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN PRICE ELSE OPEN END > 100",
        "DEFINE A AS (VOL > 5 AND PRICE > 100) OR (((VOL IS NULL) OR VOL <= 5) AND OPEN > 100)",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN AVG(A.PRICE) ELSE SUM(A.PRICE) END >= 100",
        "DEFINE A AS (VOL > 5 AND AVG(A.PRICE) >= 100) OR (((VOL IS NULL) OR VOL <= 5) AND SUM(A.PRICE) >= 100)",
    ),
    (
        "DEFINE A AS 100 < CASE WHEN VOL > 5 THEN PRICE ELSE OPEN END",
        "DEFINE A AS (VOL > 5 AND PRICE > 100) OR (((VOL IS NULL) OR VOL <= 5) AND OPEN > 100)",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN PRICE ELSE OPEN END = CLOSE",
        "DEFINE A AS (VOL > 5 AND PRICE = CLOSE) OR (((VOL IS NULL) OR VOL <= 5) AND OPEN = CLOSE)",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN PRICE + 10 ELSE OPEN * 2 END >= 100",
        "DEFINE A AS (VOL > 5 AND PRICE >= 90) OR (((VOL IS NULL) OR VOL <= 5) AND OPEN >= 50)",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN PRICE END > 100",
        "DEFINE A AS VOL > 5 AND PRICE > 100",
    ),
    (
        "DEFINE A AS CASE WHEN X > 0 THEN PRICE WHEN Y > 0 THEN OPEN ELSE CLOSE END > 100",
        "DEFINE A AS ((X > 0 AND PRICE > 100) OR ((((X IS NULL) OR X <= 0) AND Y > 0) AND OPEN > 100)) OR ((((X IS NULL) OR X <= 0) AND ((Y IS NULL) OR Y <= 0)) AND CLOSE > 100)",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN 1 WHEN VOL > 0 THEN 2 ELSE 0 END >= 1",
        "DEFINE A AS VOL > 0",
    ),
    (
        "DEFINE A AS CASE WHEN VOL > 5 THEN 1 WHEN VOL > 0 THEN 2 ELSE 0 END = 2",
        "DEFINE A AS VOL > 0 AND VOL <= 5",
    ),
)


def test_case_expression_define_simplification():
    for define_sql, expected in CASE_EXPR_SIMPLIFY_CASES:
        assert _simplify_define(define_sql) == expected


WHERE_SIMPLIFY_CASES = (
    ("WHERE open * 2 > 10 AND close > 5", "WHERE (open > 5 AND close > 5)"),
    ("WHERE ts > 300 OR ts > 100", "WHERE ts >= 101"),
    ("WHERE ts > 300 AND ts < 100", "WHERE 1 != 1"),
    ("WHERE 1 = 1 OR ts > 100", "WHERE 1 = 1"),
    ("WHERE 1 != 1 AND ts > 100", "WHERE 1 != 1"),
    ("WHERE ts > 100 OR ts > 100", "WHERE ts > 100"),
    ("WHERE (type = 'B' AND type != 'B') OR type = 'C'", "WHERE type = 'C'"),
    ("WHERE ts > 100 OR (ts > 100 AND type = 'A')", "WHERE ts > 100"),
    ("WHERE ts > 100 AND (ts > 100 OR type = 'A')", "WHERE ts > 100"),
    (
        "WHERE (ts > 100 AND type = 'A') OR (ts > 100 AND NOT(type = 'A'))",
        "WHERE (ts > 100 AND (type IS NOT NULL))",
    ),
    (
        "WHERE (ts > 100 OR type = 'A') AND (ts > 100 OR NOT(type = 'A'))",
        "WHERE ts > 100",
    ),
)


def test_where_simplification():
    for where_sql, expected in WHERE_SIMPLIFY_CASES:
        assert _simplify_where(where_sql) == expected


if __name__ == "__main__":
    tests = (
        test_range_constraint_bound_and_satisfiability,
        test_range_constraint_domain_normalization,
        test_range_constraint_apply_boundaries,
        test_range_constraint_to_expr,
        test_range_constraint_intersection,
        test_range_constraint_union,
        test_missing_metadata_rejected,
        test_expr_overlap,
        test_range_aware_define_simplification,
        test_arithmetic_define_simplification,
        test_logic_define_simplification,
        test_case_literal_define_simplification,
        test_case_expression_define_simplification,
        test_where_simplification,
    )
    for test in tests:
        test()
    print("All expression simplification tests passed.")
