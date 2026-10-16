if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from expression_clause import DEPENDENT_FUNCS


class SingleVariablePatternPushdownRule(RewriteRule):
    """
    For PATTERN(A) with a single variable, push down independent predicates
    and simplified aggregate conditions from DEFINE to the input table's WHERE clause.

    Rules:
    1. Only applies to PATTERN(A) (single variable patterns)
    2. If DEFINE or MEASURES has non-zero-offset PREV/NEXT, cannot push down
       (arbitrary row deletion would change physical navigation)
    3. Push all non-navigation predicates
    4. Simplify aggregate functions: SUM(A.col) > val becomes A.col > val
    """

    @property
    def name(self) -> str:
        return "single_variable_pattern_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        # Check if pattern is single variable
        if not isinstance(ast.pattern, VariablePattern):
            return False

        var_name = ast.pattern.name

        # This rewrite removes arbitrary input rows.  A physical navigation in
        # MEASURES can observe one of those rows even though it is not part of
        # the singleton match, so its value cannot be preserved by pushdown.
        if any(
            _has_nonzero_physical_navigation(expr)
            for expr in _iter_measure_expressions(ast)
        ):
            return False

        # Get the condition for this variable
        condition = ast.define.definitions.get(var_name)
        if condition is None or condition.expression is None:
            return False

        if (
            self._build_match_number_plan(ast, condition.expression, var_name)
            is not None
        ):
            return True

        # Check if there are non-zero offset PREV/NEXT functions
        has_physical_nav = self._has_physical_navigation(condition.expression, var_name)
        if has_physical_nav:
            return False

        # DEFINE expressions live in the MATCH_RECOGNIZE evaluation context.
        # Only move expressions which can also be evaluated as an ordinary
        # input-table WHERE predicate. Pattern functions are accepted only
        # when singleton semantics gives them an exact scalar equivalent.
        if not self._is_safe_pushdown_expression(condition.expression, var_name):
            return False

        # Check if there are any pushable conditions
        pushable_expr = self._extract_pushable_conditions(
            condition.expression, var_name
        )
        return pushable_expr is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if not isinstance(ast.pattern, VariablePattern):
            return False

        var_name = ast.pattern.name
        if any(
            _has_nonzero_physical_navigation(expr)
            for expr in _iter_measure_expressions(ast)
        ):
            return False
        condition = ast.define.definitions.get(var_name)
        if condition is None or condition.expression is None:
            return False

        match_number_plan = self._build_match_number_plan(
            ast, condition.expression, var_name
        )
        if match_number_plan is not None:
            return self._apply_match_number_plan(ast, var_name, match_number_plan)

        # Keep apply safe when it is called directly instead of through
        # RewriteEngine.is_applicable().
        if self._has_physical_navigation(condition.expression, var_name):
            return False
        if not self._is_safe_pushdown_expression(condition.expression, var_name):
            return False

        # Extract pushable conditions
        pushable_expr = self._extract_pushable_conditions(
            condition.expression, var_name
        )
        if pushable_expr is None:
            return False

        # Simplify the pushable expression (remove nav 0 offsets, simplify aggregates)
        simplified_expr = self._simplify_expression(pushable_expr, var_name)
        if simplified_expr is None:
            return False

        # Flatten unnecessary parentheses from AND/OR chains
        flattened_expr = self._flatten_and_or_expression(simplified_expr)

        # Push to WHERE clause
        pushed_where = flattened_expr.to_string()

        # Remove unnecessary parentheses from pure AND/OR chains
        pushed_where = self._remove_unnecessary_parentheses(pushed_where)
        input_table = ast.input_table.strip()
        target_fragment = f"WHERE {pushed_where}"
        if target_fragment in input_table:
            return False

        ast.input_table = f"(SELECT * FROM {ast.input_table} WHERE {pushed_where})"

        # Update DEFINE to be trivial (1 = 1)
        ast.define.definitions[var_name] = IndependentCondition(
            var_name, BinaryExpr(Literal(1), "=", Literal(1))
        )

        return True

    def _build_match_number_plan(self, ast, expr: Expr, var_name: str):
        """Build a per-partition ordered prefix plan for MATCH_NUMBER()."""
        if not ast.match_order_by or not expr.has_func("MATCH_NUMBER"):
            return None

        conjuncts: List[Expr] = []
        if not self._flatten_and(expr, conjuncts):
            return None

        max_rows = None
        row_predicates = []
        for conjunct in conjuncts:
            if conjunct.has_func("MATCH_NUMBER"):
                bound = self._match_number_prefix_bound(conjunct)
                if bound is None:
                    return None
                max_rows = bound if max_rows is None else min(max_rows, bound)
            else:
                if not self._is_safe_pushdown_expression(conjunct, var_name):
                    return None
                row_predicates.append(conjunct)

        if max_rows is None or max_rows < 1:
            return None

        projection_columns = self._projection_columns(ast)
        if not projection_columns:
            return None
        row_number_alias = choose_generated_column_name(ast, "MNO", projection_columns)
        numbered_rows = choose_generated_relation_name(ast, "RPR_MATCH_NUMBER_ROWS")
        return {
            "max_rows": max_rows,
            "row_predicates": row_predicates,
            "projection_columns": projection_columns,
            "row_number_alias": row_number_alias,
            "numbered_rows": numbered_rows,
        }

    def _match_number_prefix_bound(self, expr: Expr) -> Optional[int]:
        """Return K when expr accepts exactly the reachable prefix 1..K."""
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if not isinstance(expr, BinaryExpr):
            return None

        left = expr.left.expr if isinstance(expr.left, SemanticsExpr) else expr.left
        right = expr.right.expr if isinstance(expr.right, SemanticsExpr) else expr.right
        left_is_mn = self._is_match_number_call(left)
        right_is_mn = self._is_match_number_call(right)
        if left_is_mn == right_is_mn:
            return None

        literal = right if left_is_mn else left
        if (
            not isinstance(literal, Literal)
            or isinstance(literal.value, bool)
            or not isinstance(literal.value, (int, float))
            or int(literal.value) != literal.value
        ):
            return None
        value = int(literal.value)
        op = expr.op
        if right_is_mn:
            op = {">": "<", ">=": "<=", "<": ">", "<=": ">="}.get(op, op)

        if op == "<=":
            return value
        if op == "<":
            return value - 1
        if op == "=" and value == 1:
            return 1
        return None

    @staticmethod
    def _is_match_number_call(expr: Expr) -> bool:
        return (
            isinstance(expr, FunctionCall)
            and expr.name.upper() == "MATCH_NUMBER"
            and not expr.args
        )

    @staticmethod
    def _is_singleton_star(expr: Expr, var_name: str) -> bool:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        return (
            isinstance(expr, ColumnRef)
            and expr.name == "*"
            and expr.qualifier in (None, var_name)
        )

    def _apply_match_number_plan(self, ast, var_name: str, plan) -> bool:
        projection_sql = ", ".join(plan["projection_columns"])
        partition_sql = ""
        if ast.partition_by:
            partition_sql = f"PARTITION BY {', '.join(ast.partition_by)} "
        order_sql = ", ".join(
            column if direction == SortType.ASC else f"{column} DESC"
            for column, direction in ast.match_order_by
        )

        where_sql = ""
        if plan["row_predicates"]:
            simplified = [
                self._simplify_expression(predicate, var_name)
                for predicate in plan["row_predicates"]
            ]
            if any(predicate is None for predicate in simplified):
                return False
            predicate_sql = " AND ".join(
                f"({predicate.to_string()})" for predicate in simplified
            )
            where_sql = f" WHERE {predicate_sql}"

        source = ast.input_table.strip()
        alias = plan["row_number_alias"]
        numbered_rows = plan["numbered_rows"]
        ast.input_table = (
            f"(SELECT {projection_sql} FROM "
            f"(SELECT {projection_sql}, ROW_NUMBER() OVER "
            f"({partition_sql}ORDER BY {order_sql}) AS {alias} "
            f"FROM {source}{where_sql}) {numbered_rows} "
            f"WHERE {alias} <= {plan['max_rows']})"
        )
        ast.define.definitions[var_name] = IndependentCondition(
            var_name, BinaryExpr(Literal(1), "=", Literal(1))
        )
        return True

    def _projection_columns(self, ast):
        rows_per_match = (
            ast.rows_per_match.strip().upper()
            if ast.rows_per_match is not None
            else "ONE ROW PER MATCH"
        )
        if rows_per_match == "ALL ROWS PER MATCH":
            if ast.table_schemas is None:
                return None
            columns = ast._extract_join_source_columns()
            if not columns:
                return None
            return self._normalize_columns(columns)

        ordered = list(ast.partition_by or [])
        ordered.extend(column for column, _ in (ast.match_order_by or []))
        required = ast.get_required_input_table_columns()
        if not isinstance(required, set):
            return None
        ordered.extend(sorted(required))
        return self._normalize_columns(ordered)

    @staticmethod
    def _normalize_columns(columns):
        result = []
        seen = set()
        for column in columns:
            clean = str(column).strip().split(".")[-1]
            if clean and clean.upper() not in seen:
                seen.add(clean.upper())
                result.append(clean)
        return result

    def _is_safe_pushdown_expression(self, expr: Expr, var_name: str) -> bool:
        """Whether expr is evaluable per input row after singleton simplification."""
        if expr is None:
            return False
        if isinstance(expr, Literal):
            return True
        if isinstance(expr, ColumnRef):
            return expr.name != "*" and expr.qualifier in (None, var_name)
        if isinstance(expr, SemanticsExpr):
            return self._is_safe_pushdown_expression(expr.expr, var_name)
        if isinstance(expr, BinaryExpr):
            return self._is_safe_pushdown_expression(
                expr.left, var_name
            ) and self._is_safe_pushdown_expression(expr.right, var_name)
        if isinstance(expr, BetweenExpr):
            return (
                self._is_safe_pushdown_expression(expr.expr, var_name)
                and self._is_safe_pushdown_expression(expr.lower, var_name)
                and self._is_safe_pushdown_expression(expr.upper, var_name)
            )
        if isinstance(expr, IsNullExpr):
            return self._is_safe_pushdown_expression(expr.expr, var_name)
        if isinstance(expr, NotExpr):
            return self._is_safe_pushdown_expression(expr.expr, var_name)
        if isinstance(expr, InExpr):
            return self._is_safe_pushdown_expression(expr.expr, var_name) and all(
                self._is_safe_pushdown_expression(v, var_name) for v in expr.values
            )
        if isinstance(expr, CaseExpr):
            return all(
                self._is_safe_pushdown_expression(c, var_name)
                and self._is_safe_pushdown_expression(r, var_name)
                for c, r in expr.cases
            ) and (
                expr.else_expr is None
                or self._is_safe_pushdown_expression(expr.else_expr, var_name)
            )
        if isinstance(expr, FunctionCall):
            func_name = expr.name.upper()
            # With PATTERN(A), these functions over A's only row equal their
            # argument. Offsets are intentionally not accepted.
            if (
                func_name in ("SUM", "AVG", "MAX", "MIN", "FIRST", "LAST")
                and len(expr.args) == 1
            ):
                return self._is_safe_pushdown_expression(expr.args[0], var_name)
            # COUNT(*) and COUNT(A.*) count the singleton row and equal 1.
            # COUNT(A.col) is different because a NULL column produces 0.
            if func_name == "COUNT" and len(expr.args) == 1:
                return self._is_singleton_star(expr.args[0], var_name)
            # Ordinary scalar functions (including UDFs) do not introduce a
            # row dependency by themselves. They are safe when every argument
            # can be evaluated from the singleton pattern variable's current
            # input row. Known pattern/aggregate functions must not fall
            # through to this generic handling.
            if func_name in DEPENDENT_FUNCS:
                return False
            return all(
                self._is_safe_pushdown_expression(arg, var_name) for arg in expr.args
            )

        # Subqueries and unknown/future AST nodes are conservative no-push.
        return False

    def _has_physical_navigation(self, expr: Expr, var_name: str) -> bool:
        """Check if expression contains PREV or NEXT with non-zero offset applied to var_name."""
        unsafe = False

        def visit(node: Expr):
            nonlocal unsafe
            if unsafe or node is None:
                return
            if isinstance(node, SemanticsExpr):
                visit(node.expr)
                return
            if isinstance(node, FunctionCall):
                if node.name in ("PREV", "NEXT") and self._function_applies_to_var(
                    node, var_name
                ):
                    offset = self._extract_navigation_offset(node)
                    if offset is None or offset > 0:
                        unsafe = True
                        return
                for arg in node.args:
                    visit(arg)
                return
            if isinstance(node, BinaryExpr):
                visit(node.left)
                visit(node.right)
                return
            if isinstance(node, BetweenExpr):
                visit(node.expr)
                visit(node.lower)
                visit(node.upper)
                return

        visit(expr)
        return unsafe

    def _extract_pushable_conditions(self, expr: Expr, var_name: str) -> Optional[Expr]:
        """Extract conditions that can be pushed down (all except pure aggregates without comparison)."""
        if expr is None:
            return None

        conjuncts: List[Expr] = []
        if not self._flatten_and(expr, conjuncts):
            return None

        # Include all conditions - they will be simplified later
        # (aggregates in comparisons will be simplified during simplification)
        if len(conjuncts) == 0:
            return None

        merged = conjuncts[0]
        for i in range(1, len(conjuncts)):
            merged = BinaryExpr(merged, "AND", conjuncts[i])
        return merged

    def _is_pure_aggregate(self, expr: Expr, var_name: str) -> bool:
        """Check if expression contains only aggregates (SUM, AVG, COUNT, MAX, MIN)."""
        if isinstance(expr, SemanticsExpr):
            return self._is_pure_aggregate(expr.expr, var_name)

        if isinstance(expr, FunctionCall):
            func_upper = expr.name
            # Aggregates that should not be pushed down as-is (need special handling)
            if func_upper in ("SUM", "AVG", "COUNT", "MAX", "MIN"):
                return True
            return False

        if isinstance(expr, BinaryExpr):
            # Check if it's an aggregate comparison like SUM(A.col) > 10
            return self._is_aggregate_comparison(expr, var_name)

        return False

    def _is_aggregate_comparison(self, expr: Expr, var_name: str) -> bool:
        """Check if this is an aggregate function in a comparison."""
        if not isinstance(expr, BinaryExpr):
            return False
        if expr.op not in (">", ">=", "<", "<=", "=", "!="):
            return False

        # Check if either side is an aggregate
        left_is_agg = self._contains_aggregate(expr.left)
        right_is_agg = self._contains_aggregate(expr.right)

        return left_is_agg or right_is_agg

    def _contains_aggregate(self, expr: Expr) -> bool:
        """Check if expression contains aggregate functions."""
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._contains_aggregate(expr.expr)
        if isinstance(expr, FunctionCall):
            func_upper = expr.name
            if func_upper in ("SUM", "AVG", "COUNT", "MAX", "MIN"):
                return True
            return any(self._contains_aggregate(arg) for arg in expr.args)
        if isinstance(expr, BinaryExpr):
            return self._contains_aggregate(expr.left) or self._contains_aggregate(
                expr.right
            )
        if isinstance(expr, BetweenExpr):
            return (
                self._contains_aggregate(expr.expr)
                or self._contains_aggregate(expr.lower)
                or self._contains_aggregate(expr.upper)
            )
        return False

    def _simplify_expression(self, expr: Expr, var_name: str) -> Optional[Expr]:
        """
        Simplify expression:
        Simplify aggregates: SUM(A.col) becomes A.col
        """
        if expr is None:
            return None

        if isinstance(expr, SemanticsExpr):
            inner = self._simplify_expression(expr.expr, var_name)
            if inner is None:
                return None
            # RUNNING/FINAL only changes the row set seen by a pattern
            # aggregate.  PATTERN(A) has one row, so the modifier is redundant
            # and is not valid syntax in the generated WHERE clause.
            return inner

        if isinstance(expr, FunctionCall):
            # Handle aggregates: SUM(A.col) -> col
            func_upper = expr.name.upper()
            if (
                func_upper in ("SUM", "AVG", "MAX", "MIN", "FIRST", "LAST")
                and len(expr.args) == 1
            ):
                arg = expr.args[0]
                if isinstance(arg, SemanticsExpr):
                    arg = arg.expr
                # Extract column reference from aggregate
                col_ref = self._extract_column_reference(arg)
                if col_ref is not None:
                    # Simplify the extracted column reference to remove variable qualifier
                    return self._simplify_expression(col_ref, var_name)
                # If cannot extract, simplify argument
                return self._simplify_expression(arg, var_name)
            if (
                func_upper == "COUNT"
                and len(expr.args) == 1
                and self._is_singleton_star(expr.args[0], var_name)
            ):
                return Literal(1)

            # Recursively simplify arguments
            new_args = [self._simplify_expression(a, var_name) for a in expr.args]
            if any(arg is None for arg in new_args):
                return None
            if new_args != expr.args:
                return FunctionCall(expr.name, new_args)
            return expr

        if isinstance(expr, BinaryExpr):
            left = self._simplify_expression(expr.left, var_name)
            right = self._simplify_expression(expr.right, var_name)
            if left is None or right is None:
                return None
            if left is not expr.left or right is not expr.right:
                return BinaryExpr(left, expr.op, right)
            return expr

        if isinstance(expr, BetweenExpr):
            e = self._simplify_expression(expr.expr, var_name)
            lo = self._simplify_expression(expr.lower, var_name)
            hi = self._simplify_expression(expr.upper, var_name)
            if e is None or lo is None or hi is None:
                return None
            if e is not expr.expr or lo is not expr.lower or hi is not expr.upper:
                return BetweenExpr(e, lo, hi)
            return expr

        if isinstance(expr, IsNullExpr):
            inner = self._simplify_expression(expr.expr, var_name)
            return None if inner is None else IsNullExpr(inner, expr.is_not)

        if isinstance(expr, NotExpr):
            inner = self._simplify_expression(expr.expr, var_name)
            return None if inner is None else NotExpr(inner)

        if isinstance(expr, InExpr):
            inner = self._simplify_expression(expr.expr, var_name)
            values = [self._simplify_expression(v, var_name) for v in expr.values]
            if inner is None or any(v is None for v in values):
                return None
            return InExpr(inner, values, expr.is_not)

        if isinstance(expr, CaseExpr):
            cases = []
            for condition, result in expr.cases:
                new_condition = self._simplify_expression(condition, var_name)
                new_result = self._simplify_expression(result, var_name)
                if new_condition is None or new_result is None:
                    return None
                cases.append((new_condition, new_result))
            else_expr = (
                self._simplify_expression(expr.else_expr, var_name)
                if expr.else_expr is not None
                else None
            )
            if expr.else_expr is not None and else_expr is None:
                return None
            return CaseExpr(cases, else_expr)

        if isinstance(expr, ColumnRef):
            # Remove variable qualifier: A.TS -> TS
            if expr.qualifier and expr.qualifier == var_name:
                return ColumnRef(None, expr.name)
            return expr

        return expr

    def _extract_column_reference(self, expr: Expr) -> Optional[Expr]:
        """Extract simple column reference from expression."""
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            return self._extract_column_reference(expr.expr)
        if isinstance(expr, ColumnRef):
            return expr
        # For other expressions, return as-is (e.g., arithmetic expressions)
        return expr

    def _flatten_and_or_expression(self, expr: Expr) -> Expr:
        """
        Flatten AND/OR expressions to remove unnecessary parentheses.
        Converts ((A AND B) AND C) to (A AND B AND C) for cleaner SQL output.
        Converts ((A OR B) OR C) to (A OR B OR C) for cleaner SQL output.
        """
        if expr is None:
            return expr

        if isinstance(expr, SemanticsExpr):
            inner = self._flatten_and_or_expression(expr.expr)
            return (
                SemanticsExpr(expr.semantics, inner) if inner is not expr.expr else expr
            )

        if not isinstance(expr, BinaryExpr):
            return expr

        # Recursively flatten left and right sides first
        left = self._flatten_and_or_expression(expr.left)
        right = self._flatten_and_or_expression(expr.right)

        # Collect all operands of the same operator (AND or OR)
        if expr.op in ("AND", "OR"):
            operands = []
            self._collect_same_op_operands(
                BinaryExpr(left, expr.op, right), expr.op, operands
            )

            # Rebuild the expression in a flat structure
            if len(operands) > 1:
                result = operands[0]
                for i in range(1, len(operands)):
                    result = BinaryExpr(result, expr.op, operands[i])
                return result

        # For other operators, just return with flattened children
        if left is not expr.left or right is not expr.right:
            return BinaryExpr(left, expr.op, right)

        return expr

    def _collect_same_op_operands(self, expr: Expr, op: str, operands: List[Expr]):
        """
        Recursively collect all operands of the same operator.
        E.g., for AND: collects all operands from ((A AND B) AND C) into [A, B, C]
        """
        if expr is None:
            return

        if isinstance(expr, SemanticsExpr):
            self._collect_same_op_operands(expr.expr, op, operands)
            return

        if isinstance(expr, BinaryExpr) and expr.op == op:
            # Recursively collect from left and right
            self._collect_same_op_operands(expr.left, op, operands)
            self._collect_same_op_operands(expr.right, op, operands)
        else:
            # This is an operand (not the same operator)
            operands.append(expr)

    def _remove_unnecessary_parentheses(self, where_str: str) -> str:
        """
        Remove unnecessary parentheses from pure AND or OR chains.
        E.g., "(A > 100 AND B < 200) AND C = 300" becomes "A > 100 AND B < 200 AND C = 300"
        """
        import re

        # Pattern: (expr AND ...) AND expr  or  (expr OR ...) OR expr
        # Keep removing until no more changes
        prev = None
        while prev != where_str:
            prev = where_str

            # Remove parentheses from (AND chain) AND expr pattern
            # Match: (...AND...) AND (...)
            where_str = re.sub(
                r"\(([^()]*(?:AND|OR)[^()]*)\)\s+AND\s+", r"\1 AND ", where_str
            )

            # Remove parentheses from (OR chain) OR expr pattern
            where_str = re.sub(
                r"\(([^()]*(?:AND|OR)[^()]*)\)\s+OR\s+", r"\1 OR ", where_str
            )

        return where_str

    def _function_applies_to_var(self, func: FunctionCall, var_name: str) -> bool:
        """Check if function is applied to the variable."""
        if not func.args:
            return False
        return self._expr_mentions_var(func.args[0], var_name)

    def _expr_mentions_var(self, expr: Expr, var_name: str) -> bool:
        """Check if expression mentions the variable."""
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._expr_mentions_var(expr.expr, var_name)
        if isinstance(expr, ColumnRef):
            return expr.qualifier == var_name or expr.qualifier is None
        if isinstance(expr, FunctionCall):
            return any(self._expr_mentions_var(arg, var_name) for arg in expr.args)
        if isinstance(expr, BinaryExpr):
            return self._expr_mentions_var(
                expr.left, var_name
            ) or self._expr_mentions_var(expr.right, var_name)
        if isinstance(expr, BetweenExpr):
            return (
                self._expr_mentions_var(expr.expr, var_name)
                or self._expr_mentions_var(expr.lower, var_name)
                or self._expr_mentions_var(expr.upper, var_name)
            )
        return False

    def _extract_navigation_offset(self, func: FunctionCall) -> Optional[int]:
        """Extract navigation function offset."""
        return _extract_navigation_offset(func)

    def _flatten_and(self, expr: Expr, conjuncts: List[Expr]) -> bool:
        """Flatten AND conjunctions."""
        return _flatten_and_strict(expr, conjuncts)


single_var_pushdown_test_cases = [
    # Test case 1: Simple pattern with PREV(), this query cannot be rewritten
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES MATCH_NUMBER() AS NO, A.ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS A.TS > 100 AND PREV(A.VOL, 5) < 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES MATCH_NUMBER() AS NO, A.ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS A.TS > 100 AND PREV(A.VOL, 5) < 100
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES MATCH_NUMBER() AS NO, A.ID AS AID
    PATTERN(A)
    DEFINE A AS A.TS > 100 AND SUM(A.VOL) > 10 AND OPEN < 100
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (TS > 100 AND VOL > 10 AND OPEN < 100)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES MATCH_NUMBER() AS NO, A.ID AS AID
    PATTERN(A)
    DEFINE A AS 1 = 1
)""",
    ),
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS (A.OPEN * A.VOL) > 1000 AND NEXT(A.TS, 0) > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE ((OPEN * VOL) > 1000 AND TS > 10)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 = 1
)""",
    ),
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS (SUM(A.HIGH) * 2) > 100 AND A.OPEN > 50
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE ((HIGH * 2) > 100 AND OPEN > 50)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 = 1
)""",
    ),
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 = 1 AND PREV(A.OPEN, 3) > 3
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS PREV(A.OPEN, 3) > 3
)""",
    ),
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 != 1 AND PREV(A.OPEN, 3) > 3
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 = 1
)""",
    ),
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS 1 != 1 AND PREV(A.OPEN, 3) > 3
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, CLASSIFIER() AS CLASSY
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS 1 != 1
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        SingleVariablePatternPushdownRule,
        single_var_pushdown_test_cases,
        "single variable pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
