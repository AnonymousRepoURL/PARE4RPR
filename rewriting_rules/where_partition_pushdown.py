if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *


class WherePartitionPushdownRule(RewriteRule):
    """
    Rule description:
        Push WHERE-clause conditions that reference only PARTITION BY columns down
        into the input table as a subquery pre-filter.  Remaining (non-partition)
        conditions are kept in the outer WHERE clause.

    Handles: equality, inequality, LIKE, NOT LIKE, BETWEEN, and IN/NOT IN with
    literal lists. Predicates containing subqueries or qualified column
    references stay outside MATCH_RECOGNIZE because moving them changes their
    correlation/name-resolution scope.

    Note that we allow the following sql to be rewrittern:
        SELECT * FROM T MATCH_RECOGNIZE(PARTITION BY PT ORDER BY TS ALL ROWS PER MATCH
        PATTERN(A) DEFINE A AS TYPE = 'A') WHERE 10 / PT > 1;

    ---- Example ----
    SELECT *
    FROM NASDAQ
    MATCH_RECOGNIZE(
        PARTITION BY TICKER
        ORDER BY TIMESTAMP
        MEASURES MATCH_NUMBER() AS MN
        ALL ROWS PER MATCH
        PATTERN(A B C)
        DEFINE A AS OPEN > 110 AND VOL > 150
    )
    WHERE TICKER IN ('AAPL', 'MSFT')

    is rewritten to:

    SELECT *
    FROM (SELECT * FROM NASDAQ WHERE (TICKER IN ('AAPL', 'MSFT')))
    MATCH_RECOGNIZE(
        PARTITION BY TICKER
        ORDER BY TIMESTAMP
        MEASURES MATCH_NUMBER() AS MN
        ALL ROWS PER MATCH
        PATTERN(A B C)
        DEFINE A AS OPEN > 110 AND VOL > 150
    )
    """

    @property
    def name(self) -> str:
        return "where_partition_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        if ast.where is None:
            return False
        if ast.partition_by is None or len(ast.partition_by) == 0:
            return False
        partition_cols = {col.strip().upper() for col in ast.partition_by}
        conjuncts = _split_conjuncts(ast.where)
        return any(_is_safe_partition_only(c, partition_cols) for c in conjuncts)

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if ast.where is None:
            return False
        if ast.partition_by is None or len(ast.partition_by) == 0:
            return False

        partition_cols = {col.strip().upper() for col in ast.partition_by}
        conjuncts = _split_conjuncts(ast.where)
        pushable = [c for c in conjuncts if _is_safe_partition_only(c, partition_cols)]
        remaining = [
            c for c in conjuncts if not _is_safe_partition_only(c, partition_cols)
        ]

        if not pushable:
            return False

        # Build the WHERE string for the inner subquery
        push_expr = _join_conjuncts(pushable)
        push_sql = push_expr.to_string()

        input_table = ast.input_table.strip()
        ast.input_table = f"(SELECT * FROM {input_table} WHERE {push_sql})"

        # Update (or remove) the outer WHERE clause
        ast.where = _join_conjuncts(remaining)
        return True


def _is_safe_partition_only(expr: Expr, partition_cols: Set[str]) -> bool:
    """Return whether *expr* can move across MATCH_RECOGNIZE safely.

    ``_is_partition_only`` intentionally treats a ``SubqueryExpr`` as opaque.
    That is useful for column collection, but is not sufficient for a scope-
    changing rewrite: an opaque subquery may be correlated to a measure or
    another MATCH_RECOGNIZE output column. Likewise, a qualified reference
    such as ``MR.PT`` loses the ``MR`` alias below MATCH_RECOGNIZE.

    Keep both cases outside unless a future implementation can resolve and
    rewrite their names against concrete schemas.
    """
    return (
        _is_partition_only(expr, partition_cols)
        and not _contains_subquery(expr)
        and not _contains_qualified_column(expr)
    )


def _contains_subquery(expr: Optional[Expr]) -> bool:
    if expr is None:
        return False
    if isinstance(expr, SubqueryExpr):
        return True
    return any(_contains_subquery(child) for child in _expression_children(expr))


def _contains_qualified_column(expr: Optional[Expr]) -> bool:
    if expr is None:
        return False
    if isinstance(expr, ColumnRef):
        return expr.qualifier is not None
    return any(
        _contains_qualified_column(child) for child in _expression_children(expr)
    )


def _expression_children(expr: Expr) -> List[Expr]:
    """Return direct children for expression nodes supported by the parser."""
    if isinstance(expr, SemanticsExpr):
        return [expr.expr]
    if isinstance(expr, BinaryExpr):
        return [expr.left, expr.right]
    if isinstance(expr, BetweenExpr):
        return [expr.expr, expr.lower, expr.upper]
    if isinstance(expr, CaseExpr):
        children = []
        for condition, result in expr.cases:
            children.extend([condition, result])
        if expr.else_expr is not None:
            children.append(expr.else_expr)
        return children
    if isinstance(expr, InExpr):
        return [expr.expr, *expr.values]
    if isinstance(expr, (IsNullExpr, NotExpr)):
        return [expr.expr]
    if isinstance(expr, FunctionCall):
        return list(expr.args)
    return []


where_partition_pushdown_test_cases = [
    # Test 1: An opaque subquery may be correlated, so keep it outside.
    (
        1,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    PATTERN(A B C)
    DEFINE A AS OPEN > 110 AND VOL > 150
)
WHERE TICKER IN (SELECT * FROM TB)""",
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    PATTERN(A B C)
    DEFINE A AS OPEN > 110 AND VOL > 150
)
WHERE TICKER IN (SELECT * FROM TB)""",
    ),
    # Test 2: Simple equality on partition column
    (
        2,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE TICKER = 'AAPL'
""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE TICKER = 'AAPL') MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)""",
    ),
    # Test 3: LIKE on partition column
    (
        3,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE TICKER LIKE 'A%'
""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE (TICKER LIKE 'A%')) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)""",
    ),
    # Test 4: IN with literal list on partition column
    (
        4,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE TICKER IN ('AAPL', 'MSFT')""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE (TICKER IN ('AAPL', 'MSFT'))) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)""",
    ),
    # Test 5: NOT IN on partition column
    (
        5,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE TICKER NOT IN ('BAD', 'SKIP')""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE (TICKER NOT IN ('BAD', 'SKIP'))) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)""",
    ),
    # Test 6: Mixed condition — partition part pushed, non-partition part kept in WHERE
    (
        6,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE TICKER = 'AAPL' AND VOL > 100
""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE TICKER = 'AAPL') MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE VOL > 100
""",
    ),
    # Test 7: No partition column referenced in WHERE — rule does not apply
    (
        7,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE VOL > 100
""",
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS OPEN > 100, B AS VOL > 50
)
WHERE VOL > 100
""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        WherePartitionPushdownRule,
        where_partition_pushdown_test_cases,
        "WHERE partition pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
