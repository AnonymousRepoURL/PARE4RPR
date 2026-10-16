if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from expression_clause import collect_define_dependency_reasons


class RemoveAggregationFuncRule(RewriteRule):
    """
    Remove DEFINE aggregates whose qualified row set is proven to contain
    exactly one row when the predicate is evaluated.

    Example:
        PATTERN(A (B C | C D) E)
        DEFINE D AS SUM(C.OPEN) > 10
    becomes:
        DEFINE D AS C.OPEN > 10

    Note that different database dialects have different semantics for AGG_FUNCS.
    e.g., SUM(C.OPEN) / 2 > 10, if OPEN is integer, in trino, SUM(C.OPEN) is double
    if we rewrite SUM(C.OPEN) / 2 > 10 to C.OPEN / 2 > 10, the results may differ.

    In Oracle, SUM(C.OPEN) / 2 > 10 can be safely rewritten to C.OPEN / 2 > 10
    """

    VALUE_AGG_FUNCS = {"SUM", "AVG", "MAX", "MIN"}
    NUMERIC_TYPES = {"INT", "LONG", "FLOAT", "DOUBLE"}

    @property
    def name(self) -> str:
        return "remove_aggregation_func"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        for var_name, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            simplified = self._simplify_expr(condition.expression, ast, var_name)
            if simplified.to_string() != condition.expression.to_string():
                return True
        return False

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        changed = False
        for var_name, condition in list(ast.define.definitions.items()):
            if condition is None or condition.expression is None:
                continue
            simplified = self._simplify_expr(condition.expression, ast, var_name)
            if simplified.to_string() == condition.expression.to_string():
                continue

            reasons = sorted(collect_define_dependency_reasons(var_name, simplified))
            if reasons:
                ast.define.definitions[var_name] = DependentCondition(
                    var_name, simplified, reasons
                )
            else:
                ast.define.definitions[var_name] = IndependentCondition(
                    var_name, simplified
                )
            changed = True

        return changed

    def _is_singleton_source(self, pattern, source_var: str, current_var: str) -> bool:
        """Whether RUNNING aggregate *source_var* has at most one input row.

        The current variable's row exists while its DEFINE predicate is being
        evaluated.  A different source must be a preceding variable: it can
        occur at most once in the entire pattern and cannot occur after the
        current variable on any path.  Callers separately require that such a
        source is present, because a zero-row aggregate is not generally
        equivalent to evaluating its argument against an absent variable.
        """
        max_count = self._max_occurrence_count(pattern, source_var)
        if max_count != 1:
            return False
        if source_var == current_var:
            return True
        return not self._has_var_after_target(pattern, current_var, source_var)[0]

    def _has_var_after_target(self, node, target_var: str, source_var: str):
        """Return (unsafe_order, has_target, has_source) for pattern *node*."""
        if isinstance(node, VariablePattern):
            return False, node.name == target_var, node.name == source_var

        if isinstance(node, QuantifiedPattern):
            unsafe, has_target, has_source = self._has_var_after_target(
                node.pattern, target_var, source_var
            )
            if has_target and has_source and (node.atMost is None or node.atMost > 1):
                unsafe = True
            return unsafe, has_target, has_source

        if isinstance(node, ConcatenationPattern):
            unsafe = False
            seen_target = False
            has_target = False
            has_source = False
            for child in node.patterns:
                child_unsafe, child_target, child_source = self._has_var_after_target(
                    child, target_var, source_var
                )
                unsafe = unsafe or child_unsafe or (seen_target and child_source)
                seen_target = seen_target or child_target
                has_target = has_target or child_target
                has_source = has_source or child_source
            return unsafe, has_target, has_source

        if isinstance(node, AlternationPattern):
            results = [
                self._has_var_after_target(child, target_var, source_var)
                for child in node.patterns
            ]
            return (
                any(result[0] for result in results),
                any(result[1] for result in results),
                any(result[2] for result in results),
            )

        if isinstance(node, PermutationPattern):
            has_target = target_var in node.get_variables()
            has_source = source_var in node.get_variables()
            return has_target and has_source, has_target, has_source

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._has_var_after_target(node.pattern, target_var, source_var)

        return False, False, False

    def _max_occurrence_count(self, node, var_name: str) -> Optional[int]:
        if isinstance(node, VariablePattern):
            return 1 if node.name == var_name else 0

        if isinstance(node, EmptyPattern):
            return 0

        if isinstance(node, QuantifiedPattern):
            inner = self._max_occurrence_count(node.pattern, var_name)
            if inner is None:
                return None
            if node.atMost is None:
                return None if inner > 0 else 0
            return inner * node.atMost

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._max_occurrence_count(node.pattern, var_name)

        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            total = 0
            for child in node.patterns:
                child_count = self._max_occurrence_count(child, var_name)
                if child_count is None:
                    return None
                total += child_count
            return total

        if isinstance(node, AlternationPattern):
            counts = [
                self._max_occurrence_count(child, var_name) for child in node.patterns
            ]
            if any(count is None for count in counts):
                return None
            return max(counts) if counts else 0

        return None

    def _simplify_expr(
        self, expr: Expr, ast: RPRAbstractSyntaxTree, current_var: str
    ) -> Expr:
        if expr is None:
            return expr

        if isinstance(expr, SemanticsExpr):
            # FINAL observes the whole match rather than the running prefix.
            # Keep it conservative; removing the aggregate would also leave a
            # FINAL modifier on a scalar expression.
            if expr.semantics and expr.semantics.upper() == "FINAL":
                return expr
            inner = self._simplify_expr(expr.expr, ast, current_var)
            return (
                SemanticsExpr(expr.semantics, inner)
                if inner.to_string() != expr.expr.to_string()
                else expr
            )

        if isinstance(expr, FunctionCall):
            func_name = expr.name.upper()
            if func_name in self.VALUE_AGG_FUNCS and len(expr.args) == 1:
                arg = expr.args[0]
                source_var = self._single_qualified_source(arg)
                if (
                    source_var is not None
                    and self._is_singleton_source(ast.pattern, source_var, current_var)
                    and self._source_exists_at_target(
                        ast.pattern, source_var, current_var
                    )
                    and self._aggregate_type_is_safe(ast, func_name, arg)
                ):
                    replacement = self._drop_current_var_qualifier(arg, current_var)
                    return self._simplify_expr(replacement, ast, current_var)

            if func_name == "COUNT" and len(expr.args) == 1:
                arg = expr.args[0]
                source_var = self._qualified_star_source(arg)
                if (
                    source_var is not None
                    and self._is_singleton_source(ast.pattern, source_var, current_var)
                    and self._source_exists_at_target(
                        ast.pattern, source_var, current_var
                    )
                ):
                    return Literal(1)

                source_var = self._single_qualified_source(arg)
                if (
                    source_var is not None
                    and self._is_singleton_source(ast.pattern, source_var, current_var)
                    and self._source_exists_at_target(
                        ast.pattern, source_var, current_var
                    )
                ):
                    replacement = self._drop_current_var_qualifier(arg, current_var)
                    if self._expr_is_non_null_at_target(
                        ast, replacement, source_var, current_var
                    ):
                        return Literal(1)
                    return CaseExpr([(IsNullExpr(replacement), Literal(0))], Literal(1))

            # Aggregates and row-pattern navigation establish their own row
            # evaluation context.  If the outer call itself was not removed,
            # do not recursively rewrite inside it.  Oracle rejects nesting
            # among these operators (except PREV/NEXT(FIRST/LAST(...))); this
            # guard also protects ASTs built programmatically.
            if func_name in {
                "SUM",
                "AVG",
                "COUNT",
                "MAX",
                "MIN",
                "PREV",
                "NEXT",
                "FIRST",
                "LAST",
            }:
                return expr

            new_args = [self._simplify_expr(arg, ast, current_var) for arg in expr.args]
            if any(
                new.to_string() != old.to_string()
                for new, old in zip(new_args, expr.args)
            ):
                return FunctionCall(expr.name, new_args)
            return expr

        if isinstance(expr, BinaryExpr):
            left = self._simplify_expr(expr.left, ast, current_var)
            right = self._simplify_expr(expr.right, ast, current_var)
            if (
                left.to_string() != expr.left.to_string()
                or right.to_string() != expr.right.to_string()
            ):
                return BinaryExpr(left, expr.op, right)
            return expr

        if isinstance(expr, BetweenExpr):
            inner = self._simplify_expr(expr.expr, ast, current_var)
            lower = self._simplify_expr(expr.lower, ast, current_var)
            upper = self._simplify_expr(expr.upper, ast, current_var)
            if (
                inner.to_string() != expr.expr.to_string()
                or lower.to_string() != expr.lower.to_string()
                or upper.to_string() != expr.upper.to_string()
            ):
                return BetweenExpr(inner, lower, upper)
            return expr

        if isinstance(expr, CaseExpr):
            new_cases = [
                (
                    self._simplify_expr(cond, ast, current_var),
                    self._simplify_expr(result, ast, current_var),
                )
                for cond, result in expr.cases
            ]
            new_else = (
                self._simplify_expr(expr.else_expr, ast, current_var)
                if expr.else_expr
                else None
            )
            if any(
                c.to_string() != old_c.to_string() or r.to_string() != old_r.to_string()
                for (c, r), (old_c, old_r) in zip(new_cases, expr.cases)
            ) or (
                new_else is not None
                and new_else.to_string() != expr.else_expr.to_string()
            ):
                return CaseExpr(new_cases, new_else)
            return expr

        if isinstance(expr, InExpr):
            inner = self._simplify_expr(expr.expr, ast, current_var)
            values = [
                self._simplify_expr(value, ast, current_var) for value in expr.values
            ]
            if inner.to_string() != expr.expr.to_string() or any(
                new.to_string() != old.to_string()
                for new, old in zip(values, expr.values)
            ):
                return InExpr(inner, values, expr.is_not)
            return expr

        if isinstance(expr, IsNullExpr):
            inner = self._simplify_expr(expr.expr, ast, current_var)
            return (
                IsNullExpr(inner, expr.is_not)
                if inner.to_string() != expr.expr.to_string()
                else expr
            )

        if isinstance(expr, NotExpr):
            inner = self._simplify_expr(expr.expr, ast, current_var)
            return (
                NotExpr(inner) if inner.to_string() != expr.expr.to_string() else expr
            )

        return expr

    def _qualified_star_source(self, expr: Expr) -> Optional[str]:
        if isinstance(expr, SemanticsExpr):
            return self._qualified_star_source(expr.expr)
        if isinstance(expr, ColumnRef) and expr.name == "*" and expr.qualifier:
            return expr.qualifier
        return None

    def _single_qualified_source(self, expr: Expr) -> Optional[str]:
        """Return the sole explicit pattern-variable qualifier in *expr*.

        Unqualified columns use the universal row set and therefore make the
        aggregate ineligible. Nested aggregates/navigation are also rejected.
        """
        qualifiers = set()

        def visit(node):
            if isinstance(node, SemanticsExpr):
                return visit(node.expr)
            if isinstance(node, ColumnRef):
                if node.name == "*" or not node.qualifier:
                    return False
                qualifiers.add(node.qualifier)
                return len(qualifiers) <= 1
            if isinstance(node, FunctionCall):
                if node.name.upper() in {
                    "SUM",
                    "AVG",
                    "COUNT",
                    "MAX",
                    "MIN",
                    "PREV",
                    "NEXT",
                    "FIRST",
                    "LAST",
                }:
                    return False
                return all(visit(arg) for arg in node.args)
            if isinstance(node, BinaryExpr):
                return visit(node.left) and visit(node.right)
            if isinstance(node, BetweenExpr):
                return visit(node.expr) and visit(node.lower) and visit(node.upper)
            if isinstance(node, CaseExpr):
                return all(
                    visit(cond) and visit(result) for cond, result in node.cases
                ) and (node.else_expr is None or visit(node.else_expr))
            if isinstance(node, InExpr):
                return visit(node.expr) and all(visit(value) for value in node.values)
            if isinstance(node, (IsNullExpr, NotExpr)):
                return visit(node.expr)
            if isinstance(node, SubqueryExpr):
                return False
            return True

        return next(iter(qualifiers)) if visit(expr) and len(qualifiers) == 1 else None

    def _aggregate_type_is_safe(self, ast, func_name: str, arg: Expr) -> bool:
        if func_name in {"MIN", "MAX"}:
            return ast._infer_expression_metadata(arg).data_type is not None
        return ast._infer_expression_metadata(arg).data_type in self.NUMERIC_TYPES

    def _source_exists_at_target(
        self, pattern, source_var: str, current_var: str
    ) -> bool:
        if source_var == current_var:
            return True
        # This is intentionally conservative. A source required by every full
        # match and forbidden after current must already exist at current.
        return source_var in pattern.get_must_appear_var()

    def _expr_is_non_null_at_target(
        self, ast, expr: Expr, source_var: str, current_var: str
    ) -> bool:
        if not self._source_exists_at_target(ast.pattern, source_var, current_var):
            return False
        return ast._infer_expression_metadata(expr).nullable is False

    def _drop_current_var_qualifier(self, expr: Expr, current_var: str) -> Expr:
        if isinstance(expr, SemanticsExpr):
            return SemanticsExpr(
                expr.semantics, self._drop_current_var_qualifier(expr.expr, current_var)
            )
        if isinstance(expr, ColumnRef):
            if expr.qualifier is not None and expr.qualifier == current_var:
                return ColumnRef(None, expr.name)
            return copy.deepcopy(expr)
        if isinstance(expr, FunctionCall):
            return FunctionCall(
                expr.name,
                [
                    self._drop_current_var_qualifier(arg, current_var)
                    for arg in expr.args
                ],
            )
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                self._drop_current_var_qualifier(expr.left, current_var),
                expr.op,
                self._drop_current_var_qualifier(expr.right, current_var),
            )
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(
                self._drop_current_var_qualifier(expr.expr, current_var),
                self._drop_current_var_qualifier(expr.lower, current_var),
                self._drop_current_var_qualifier(expr.upper, current_var),
            )
        if isinstance(expr, CaseExpr):
            return CaseExpr(
                [
                    (
                        self._drop_current_var_qualifier(cond, current_var),
                        self._drop_current_var_qualifier(result, current_var),
                    )
                    for cond, result in expr.cases
                ],
                (
                    self._drop_current_var_qualifier(expr.else_expr, current_var)
                    if expr.else_expr
                    else None
                ),
            )
        if isinstance(expr, InExpr):
            return InExpr(
                self._drop_current_var_qualifier(expr.expr, current_var),
                [
                    self._drop_current_var_qualifier(value, current_var)
                    for value in expr.values
                ],
                expr.is_not,
            )
        if isinstance(expr, IsNullExpr):
            return IsNullExpr(
                self._drop_current_var_qualifier(expr.expr, current_var), expr.is_not
            )
        if isinstance(expr, NotExpr):
            return NotExpr(self._drop_current_var_qualifier(expr.expr, current_var))
        return copy.deepcopy(expr)


remove_aggregation_func_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLA
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS SUM(C.OPEN) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLA
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS OPEN > 10
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS AVG(C.OPEN) > 10 AND MAX(C.HIGH) < 20 AND MIN(C.LOW) >= 1
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS OPEN > 10 AND HIGH < 20 AND LOW >= 1
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS COUNT(C.*) = 1 AND SUM(C.OPEN + C.CLOSE) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS 1 = 1 AND OPEN + CLOSE > 10
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C+ E)
    DEFINE C AS SUM(C.OPEN) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C+ E)
    DEFINE C AS SUM(C.OPEN) > 10
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | D) E)
    DEFINE C AS SUM(C.OPEN) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | D) E)
    DEFINE C AS OPEN > 10
)""",
    ),
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS COUNT(C.OPEN) = 1 AND SUM(A.OPEN) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B C | C D) E)
    DEFINE C AS A.OPEN > 10
)""",
    ),
]


default_remove_aggregation_func_table_schemas = {
    "T": {
        "columns": [
            "PT",
            "TS",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VAL",
            "VOL",
            "TYPE",
        ],
        "column_types": {
            "PT": "LONG",
            "TS": "LONG",
            "OPEN": "DOUBLE",
            "HIGH": "DOUBLE",
            "LOW": "DOUBLE",
            "CLOSE": "DOUBLE",
            "VAL": "DOUBLE",
            "VOL": "DOUBLE",
            "TYPE": "VARCHAR(20)",
        },
        "column_nullables": {
            "PT": False,
            "TS": False,
            "OPEN": False,
            "HIGH": False,
            "LOW": False,
            "CLOSE": False,
            "VAL": False,
            "VOL": True,
            "TYPE": False,
        },
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        RemoveAggregationFuncRule,
        remove_aggregation_func_test_cases,
        "remove aggregation function",
        verbose,
        default_remove_aggregation_func_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
