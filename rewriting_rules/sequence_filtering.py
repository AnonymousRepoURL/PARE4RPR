if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import re

from .utils import *


class SequenceFilteringRule(RewriteRule):
    """
    Baseline Sequence Filtering rewrite from
    "Optimization of Row Pattern Matching over Sequence Data in Spark SQL."

    A partition survives when it contains a row satisfying at least one
    independent predicate belonging to a must-appear pattern variable.

    For known non-null (or legacy unknown) partition columns the compact IN
    form is used.  When schema metadata marks any key nullable, or when the
    input contains an outer join that can null-extend a declared NOT NULL key,
    the rule emits a NULL-safe correlated EXISTS predicate instead.  A simple
    input qualifier used by PARTITION BY / ORDER BY is rebound to the rewritten
    derived table; unsafe qualifiers into a join tree make the rule inapplicable.

    If the partition columns are nullable, we can use EXISTS to filter, e.g.,
    SELECT * FROM T MATCH_RECOGNIZE(
        PARTITION BY PT
        ORDER BY TS
        PATTERN(A B? C)
        DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
    )

    If PT is nullable, the above query can be rewritten as:

    SELECT RPR_INPUT.*
    FROM (SELECT * FROM T) RPR_INPUT
    WHERE EXISTS (
        SELECT 1
        FROM (SELECT * FROM T) RPR_SEQUENCE_ROWS
        WHERE ((TYPE = 'A') OR (TYPE = 'C'))
        AND ( RPR_INPUT.PT = RPR_SEQUENCE_ROWS.PT
            OR ( RPR_INPUT.PT IS NULL AND RPR_SEQUENCE_ROWS.PT IS NULL)
        )
    )
    """

    @property
    def name(self) -> str:
        return "sequence_filtering"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_sequence_filtering_applied", False):
            return False
        plan = self._build_plan(ast)
        if plan is None:
            return False

        source = ast.input_table.strip()
        keys = plan["partition_columns"]
        condition = plan["condition"]
        selected_keys = ", ".join(keys)
        if plan["has_nullable_partition"]:
            input_alias = choose_generated_relation_name(ast, "RPR_INPUT")
            rows_alias = choose_generated_relation_name(
                ast, "RPR_SEQUENCE_ROWS", {input_alias}
            )
            key_equalities = " AND ".join(
                f"({input_alias}.{key} = {rows_alias}.{key} OR "
                f"({input_alias}.{key} IS NULL AND {rows_alias}.{key} IS NULL))"
                for key in keys
            )
            aliased_input = bind_input_relation_alias(source, input_alias)
            aliased_rows = bind_input_relation_alias(source, rows_alias)
            rewritten_input = (
                f"(SELECT {input_alias}.* FROM {aliased_input} "
                f"WHERE EXISTS (SELECT 1 FROM {aliased_rows} "
                f"WHERE ({condition}) AND {key_equalities}))"
            )
        else:
            # Deliberately use IN as the compact Oracle-compatible baseline.
            # This preserves the historical behavior when nullability is
            # unknown.
            partition_value = keys[0] if len(keys) == 1 else f"({selected_keys})"
            flattened = ast.flatten_safe_select_star_input(source)
            if flattened is None:
                rewritten_input = (
                    f"(SELECT * FROM {source} WHERE {partition_value} IN "
                    f"(SELECT {selected_keys} FROM {source} WHERE {condition}))"
                )
                # The generic post-rule wrapper optimizer is intentionally more
                # permissive for backwards compatibility.  Do not let it
                # flatten a source that failed the strict proof above (ROWNUM
                # is a typical counterexample).
                ast._skip_from_clause_optimization_once = True
            else:
                base_source, input_predicates = flattened
                input_filter = " AND ".join(f"({item})" for item in input_predicates)
                sequence_filter = condition
                if input_filter:
                    sequence_filter = f"{input_filter} AND ({condition})"
                partition_filter = (
                    f"{partition_value} IN (SELECT {selected_keys} FROM {base_source} "
                    f"WHERE {sequence_filter})"
                )
                outer_filter = partition_filter
                if input_filter:
                    outer_filter = f"{input_filter} AND ({partition_filter})"
                rewritten_input = f"(SELECT * FROM {base_source} WHERE {outer_filter})"

        # Restore a source qualifier used by PARTITION BY / ORDER BY after the
        # original source has been replaced by a derived table.
        input_rebind_alias = plan["input_rebind_alias"]
        ast.input_table = (
            f"{rewritten_input} {input_rebind_alias}"
            if input_rebind_alias is not None
            else rewritten_input
        )
        ast._sequence_filtering_applied = True
        return True

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        if (
            ast.pattern.get_min_row_num == 0
            or ast.rows_per_match == "ALL ROWS PER MATCH WITH UNMATCHED ROWS"
        ):
            return None

        qualifier_safe, input_rebind_alias = resolve_input_rebind_alias(ast)
        if not qualifier_safe:
            return None

        partition_columns = []
        partition_column_refs = []
        for column in ast.partition_by or []:
            column = column.strip()
            # Building a scalar/row-value IN predicate is only safe for ordinary
            # (possibly qualified) column references, not arbitrary expressions.
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*)?", column):
                return None
            partition_column_refs.append(column)
            partition_columns.append(column.split(".")[-1])
        if not partition_columns or len(set(partition_columns)) != len(
            partition_columns
        ):
            return None

        predicates = []
        seen = set()
        for var_name in sorted(ast.pattern.get_must_appear_var()):
            condition = ast.define.definitions.get(var_name)
            independent = (
                condition.get_independent_filter() if condition is not None else None
            )
            independent = _strip_var_qualifier(independent, var_name)
            if independent is None:
                continue
            if _is_true(independent):
                # TRUE OR anything is TRUE, so no partition can be removed.
                return None
            sql = independent.to_string()
            if sql not in seen:
                seen.add(sql)
                predicates.append(independent)
        if not predicates:
            return None
        combined = predicates[0]
        simplify_context = ast.expression_simplify_context()
        for predicate in predicates[1:]:
            combined = _simplify_logical_expr(
                combined,
                "OR",
                predicate,
                simplify_context,
            )
        if _is_true(combined):
            return None
        return {
            "partition_columns": partition_columns,
            "input_rebind_alias": input_rebind_alias,
            "has_nullable_partition": (
                input_may_null_extend_columns(ast.input_table)
                or any(
                    ast.is_column_nullable(column) is True
                    for column in partition_column_refs
                )
            ),
            "condition": " OR ".join(
                f"({predicate.to_string()})" for predicate in predicates
            ),
        }


def _is_true(expr: Expr) -> bool:
    if isinstance(expr, SemanticsExpr):
        return _is_true(expr.expr)
    if isinstance(expr, Literal):
        return expr.value is True
    # The lightweight expression lexer represents bare SQL TRUE as an
    # unqualified ColumnRef.
    if isinstance(expr, ColumnRef):
        return expr.qualifier is None and expr.name == "TRUE"
    return isinstance(expr, BinaryExpr) and expr.to_string() in {"1 = 1", "TRUE = TRUE"}


sequence_filtering_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B? C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
        """
SELECT * 
FROM (
    SELECT * 
    FROM T 
    WHERE PT IN (SELECT PT FROM T WHERE (TYPE = 'A') OR (TYPE = 'C'))
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B? C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS TRUE, B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS TRUE, B AS B.TYPE = 'B'
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.VALUE > 5, B AS B.VALUE <= 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.VALUE > 5, B AS B.VALUE <= 5
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY C1, C2
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM (SELECT * FROM T WHERE (C1, C2) IN (SELECT C1, C2 FROM T WHERE (TYPE = 'A') OR (TYPE = 'B'))) MATCH_RECOGNIZE(
    PARTITION BY C1, C2
    ALL ROWS PER MATCH
    ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
    ),
    (
        6,
        """
SELECT * FROM (SELECT * FROM T WHERE VAL > 10) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B? C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
        """
SELECT * 
FROM (
    SELECT * 
    FROM T
    WHERE (VAL > 10) AND (PT IN (SELECT PT FROM T WHERE (VAL > 10) AND ((TYPE = 'A') OR (TYPE = 'C'))))
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B? C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
    ),
    (
        7,
        """
SELECT * FROM (SELECT * FROM T WHERE ROWNUM <= 10) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS A.TYPE = 'A', C AS C.TYPE = 'C'
)""",
        """
SELECT * FROM (
    SELECT * FROM (SELECT * FROM T WHERE ROWNUM <= 10)
    WHERE PT IN (
        SELECT PT FROM (SELECT * FROM T WHERE ROWNUM <= 10)
        WHERE (TYPE = 'A') OR (TYPE = 'C')
    )
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS A.TYPE = 'A', C AS C.TYPE = 'C'
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        SequenceFilteringRule,
        sequence_filtering_test_cases,
        "sequence filtering baseline",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
