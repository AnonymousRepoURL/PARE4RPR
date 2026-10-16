if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import re

from .utils import *


class MustAppearPartitionFilterPushdownRule(RewriteRule):
    """
    Rule description:
        For must-appear pattern variables with independent DEFINE predicates,
        precompute the PARTITION BY values whose partitions contain rows
        satisfying those predicates, then filter the input by those partition
        values before MATCH_RECOGNIZE runs. If the query has no PARTITION BY
        clause, treat the whole input as a single partition and use a global
        EXISTS/HAVING check to either keep all rows or return no rows.

    Nullable partition columns are handled with a NULL-safe correlated EXISTS
    filter. Inputs containing an outer join use the same path because the join
    can null-extend a source column declared NOT NULL. Known non-null and legacy
    unknown schemas otherwise retain the compact IN form.

    Example:
        PATTERN(A {-B+?-} C)
        DEFINE A AS OPEN BETWEEN 70 AND 75, C AS OPEN BETWEEN 80 AND 90

    Since A and C must appear, every matching partition (or the single global
    partition when PARTITION BY is absent) must contain at least one row
    satisfying A's independent predicate and at least one row satisfying C's
    independent predicate.
    """

    satisfying_partitions_sql = None

    def __init__(self):
        self.satisfying_partitions_sql = None

    @property
    def name(self) -> str:
        return "must_appear_partition_filter_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        if (
            plan is None
            or ast.rows_per_match == "ALL ROWS PER MATCH WITH UNMATCHED ROWS"
        ):
            return False

        self.satisfying_partitions_sql = plan["satisfying_partitions_sql"]
        filter_sql = plan["filter_sql"]
        input_table = ast.input_table.strip()
        target_fragment = f"WHERE {filter_sql}"
        if target_fragment in input_table:
            return False

        if plan.get("has_nullable_partition"):
            input_alias = choose_generated_relation_name(ast, "RPR_INPUT")
            partitions_alias = choose_generated_relation_name(
                ast, "RPR_SATISFYING_PARTITIONS", {input_alias}
            )
            key_equalities = " AND ".join(
                f"({input_alias}.{col.split('.')[-1]} = {partitions_alias}.{col.split('.')[-1]} OR "
                f"({input_alias}.{col.split('.')[-1]} IS NULL AND "
                f"{partitions_alias}.{col.split('.')[-1]} IS NULL))"
                for col in plan["partition_columns"]
            )
            aliased_input = bind_input_relation_alias(input_table, input_alias)
            rewritten_input = (
                f"(SELECT {input_alias}.* FROM {aliased_input} "
                f"WHERE EXISTS (SELECT 1 FROM ({self.satisfying_partitions_sql}) "
                f"{partitions_alias} WHERE {key_equalities}))"
            )
        else:
            rewritten_input = f"(SELECT * FROM {input_table} WHERE {filter_sql})"
        ast.input_table = self._rebind_rewritten_input(
            rewritten_input, plan.get("input_rebind_alias")
        )
        return True

    def get_satisfying_partitions_sql(self) -> Optional[str]:
        return self.satisfying_partitions_sql

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        if ast.define is None or ast.pattern is None:
            self.satisfying_partitions_sql = None
            return None

        must_appear_vars = sorted(ast.pattern.get_must_appear_var())
        if not must_appear_vars:
            self.satisfying_partitions_sql = None
            return None

        qualifier_safe, input_rebind_alias = self._resolve_input_rebind_alias(ast)
        if not qualifier_safe:
            self.satisfying_partitions_sql = None
            return None

        having_predicates = []
        seen_predicates = set()
        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None:
                continue

            independent_expr = condition.get_independent_filter()
            if independent_expr is None:
                continue

            stripped_expr = _strip_var_qualifier(independent_expr, var_name)
            if stripped_expr is None or self._is_trivial_true(stripped_expr):
                continue

            predicate_sql = stripped_expr.to_string()
            if predicate_sql in seen_predicates:
                continue
            seen_predicates.add(predicate_sql)
            having_predicates.append(
                f"MAX(CASE WHEN {predicate_sql} THEN 1 ELSE 0 END) = 1"
            )

        if not having_predicates:
            self.satisfying_partitions_sql = None
            return None

        having_sql = " AND ".join(having_predicates)
        input_table_sql = ast.input_table.strip()

        partition_cols = [
            col.strip() for col in (ast.partition_by or []) if col.strip()
        ]
        if not partition_cols:
            satisfying_partitions_sql = (
                f"SELECT 1 FROM {input_table_sql} HAVING {having_sql}"
            )
            return {
                "filter_sql": f"EXISTS ({satisfying_partitions_sql})",
                "satisfying_partitions_sql": satisfying_partitions_sql,
                "input_rebind_alias": input_rebind_alias,
            }

        projection_sql = ", ".join(partition_cols)
        satisfying_partitions_sql = (
            f"SELECT {projection_sql} FROM {input_table_sql} "
            f"GROUP BY {projection_sql} HAVING {having_sql}"
        )
        if len(partition_cols) == 1:
            filter_sql = f"{partition_cols[0]} IN ({satisfying_partitions_sql})"
        else:
            filter_sql = f"({projection_sql}) IN ({satisfying_partitions_sql})"

        return {
            "filter_sql": filter_sql,
            "satisfying_partitions_sql": satisfying_partitions_sql,
            "partition_columns": partition_cols,
            "input_rebind_alias": input_rebind_alias,
            "has_nullable_partition": (
                self._input_may_null_extend_columns(input_table_sql)
                or any(
                    ast.is_column_nullable(column) is True for column in partition_cols
                )
            ),
        }

    @staticmethod
    def _rebind_rewritten_input(input_sql: str, alias: Optional[str]) -> str:
        """Expose an original input correlation name on the rewritten source."""
        return f"{input_sql} {alias}" if alias is not None else input_sql

    @staticmethod
    def _resolve_input_rebind_alias(ast: RPRAbstractSyntaxTree):
        """Find a safely reusable qualifier referenced by PARTITION/ORDER BY.

        Wrapping the input hides its correlation name. A simple input can bind
        that name to the complete rewritten source. For a JOIN or another
        complex source, a qualifier can identify one constituent relation, so
        rebinding it to the complete source would change name resolution; such
        inputs are rejected conservatively.
        """
        qualifiers = {}
        columns = list(ast.partition_by or []) + [
            column for column, _sort_type in (ast.match_order_by or [])
        ]
        for column in columns:
            parts = str(column).strip().split(".")
            if len(parts) < 2:
                continue
            qualifier = parts[-2]
            qualifiers.setdefault(qualifier.upper(), qualifier)

        if not qualifiers:
            return True, None
        if len(qualifiers) != 1:
            return False, None

        identifier = r"[A-Za-z_][A-Za-z0-9_$]*"
        match = re.fullmatch(
            rf"\s*({identifier}(?:\.{identifier})*)"
            rf"(?:\s+(?:AS\s+)?({identifier}))?\s*",
            ast.input_table,
            re.IGNORECASE,
        )
        if match is None:
            return False, None

        relation_name, explicit_alias = match.groups()
        visible_name = explicit_alias or relation_name.split(".")[-1]
        qualifier = next(iter(qualifiers.values()))
        if visible_name.upper() != qualifier.upper():
            return False, None
        return True, qualifier

    @staticmethod
    def _input_may_null_extend_columns(input_table_sql: str) -> bool:
        """Whether the row source can null-extend a declared NOT NULL key."""
        if not input_table_sql:
            return False
        return bool(
            re.search(
                r"\b(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?\s+JOIN\b",
                input_table_sql,
                re.IGNORECASE,
            )
            # Oracle's legacy outer-join operator can also introduce NULLs.
            or re.search(r"\(\s*\+\s*\)", input_table_sql)
        )

    def _is_trivial_true(self, expr: Expr) -> bool:
        if isinstance(expr, SemanticsExpr):
            return self._is_trivial_true(expr.expr)
        if isinstance(expr, BinaryExpr):
            return expr.to_string() in {"1 = 1", "TRUE = TRUE"}
        if isinstance(expr, Literal):
            return expr.value is True
        return False


must_appear_partition_filter_pushdown_test_cases = [
    (
        1,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A {-B+?-} C)
    DEFINE A AS OPEN BETWEEN 70 AND 75, C AS OPEN BETWEEN 80 AND 90
)""",
        """
SELECT * FROM (
    SELECT * FROM NASDAQ 
    WHERE TICKER IN (
        SELECT TICKER 
        FROM NASDAQ 
        GROUP BY TICKER 
        HAVING MAX(CASE WHEN (OPEN BETWEEN 70 AND 75) THEN 1 ELSE 0 END) = 1 
            AND MAX(CASE WHEN (OPEN BETWEEN 80 AND 90) THEN 1 ELSE 0 END) = 1
    )
) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A {-B+?-} C)
    DEFINE A AS OPEN BETWEEN 70 AND 75, C AS OPEN BETWEEN 80 AND 90
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY C1, C2
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.OPEN > 10, B AS B.VOL < 100
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (C1, C2) IN (SELECT C1, C2 FROM TA GROUP BY C1, C2 HAVING MAX(CASE WHEN OPEN > 10 THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN VOL < 100 THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY C1, C2
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.OPEN > 10, B AS B.VOL < 100
)""",
    ),
    (
        3,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS OPEN BETWEEN 70 AND 75, C AS OPEN BETWEEN 80 AND 90
)""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE EXISTS (SELECT 1 FROM NASDAQ HAVING MAX(CASE WHEN (OPEN BETWEEN 70 AND 75) THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN (OPEN BETWEEN 80 AND 90) THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS OPEN BETWEEN 70 AND 75, C AS OPEN BETWEEN 80 AND 90
)""",
    ),
    # Optional variables are not must-appear, so their predicates must not be
    # included in the partition HAVING filter.
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B? C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN TYPE = 'A' THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN TYPE = 'C' THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B? C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    # Dependent DEFINE conditions should contribute only their independent
    # top-level AND conjuncts.
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.PRICE > 10 AND B.PRICE > A.PRICE
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN TYPE = 'A' THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN PRICE > 10 THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.PRICE > 10 AND B.PRICE > A.PRICE
)""",
    ),
    # Identical independent predicates across must-appear variables are
    # deduplicated in the HAVING clause.
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.OPEN > 10, B AS B.OPEN > 10
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN OPEN > 10 THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.OPEN > 10, B AS B.OPEN > 10
)""",
    ),
    # If all must-appear predicates are trivial and every nontrivial predicate
    # belongs to optional variables, the rule should not fire.
    (
        7,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B?)
    DEFINE A AS 1 = 1, B AS VOL > 100
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B?)
    DEFINE A AS 1 = 1, B AS VOL > 100
)""",
    ),
    # In alternation, only variables present in every branch are must-appear.
    (
        8,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A (B | C) D)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C', D AS TYPE = 'D'
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN TYPE = 'A' THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN TYPE = 'D' THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A (B | C) D)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C', D AS TYPE = 'D'
)""",
    ),
    # PERMUTE keeps all listed variables mandatory, even though their order is flexible.
    (
        9,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(PERMUTE(A, B, C))
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN TYPE = 'A' THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN TYPE = 'B' THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN TYPE = 'C' THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(PERMUTE(A, B, C))
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    # No PARTITION BY with a derived input table: the whole derived input is a
    # single partition checked by EXISTS/HAVING.
    (
        10,
        """
SELECT * FROM (SELECT * FROM T WHERE MARKET = 'NASDAQ') MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS OPEN > 10, B AS CLOSE < 20
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE (MARKET = 'NASDAQ') AND (EXISTS (SELECT 1 FROM (SELECT * FROM T WHERE MARKET = 'NASDAQ') HAVING MAX(CASE WHEN OPEN > 10 THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN CLOSE < 20 THEN 1 ELSE 0 END) = 1))) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS OPEN > 10, B AS CLOSE < 20
)""",
    ),
    # More complex independent predicates should be kept as a single CASE WHEN
    # predicate after stripping the current variable qualifier.
    (
        11,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.TYPE IN ('A', 'B') OR A.TYPE IS NULL, B AS B.PRICE BETWEEN 10 AND 20
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE PT IN (SELECT PT FROM T GROUP BY PT HAVING MAX(CASE WHEN ((TYPE IN ('A', 'B')) OR (TYPE IS NULL)) THEN 1 ELSE 0 END) = 1 AND MAX(CASE WHEN (PRICE BETWEEN 10 AND 20) THEN 1 ELSE 0 END) = 1)) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN(A B)
    DEFINE A AS A.TYPE IN ('A', 'B') OR A.TYPE IS NULL, B AS B.PRICE BETWEEN 10 AND 20
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        MustAppearPartitionFilterPushdownRule,
        must_appear_partition_filter_pushdown_test_cases,
        "must-appear partition filter pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
