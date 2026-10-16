if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from expression_clause import DEPENDENT_FUNCS
from queryanalysis import ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS


class DefinePartitionPushdownRule(RewriteRule):
    """
    Rule description:
        If the query has a PARTITION BY clause, and must-appear variables have equality
        constraints on partition columns, push these constraints to the input table's WHERE clause.

    Example:
        SELECT * FROM TABLE MATCH_RECOGNIZE(
            PARTITION BY C1, C2, C3
            ORDER BY TS
            MEASURES A.ID AS AID, B.VOL AS BVOL
            AFTER MATCH SKIP TO NEXT ROW
            PATTERN(A B C)
            DEFINE A AS NEXT(A.C1) = 4, B AS C2 = 'B', C AS PREV(C.C3) > 10
        )

    Since A, B, C are must-appear variables:
    - A constrains C1 = 4 (via NEXT(A.C1) = 4)
    - B constrains C2 = 'B' (via C2 = 'B')
    - C constrains C3 > 10 (via PREV(C.C3) > 10)

    These can be pushed down to: WHERE C1 = 4 AND C2 = 'B' AND C3 > 10

    Rule handles:
    - Simple equality constraints: A.COL = value
    - Independent scalar functions of partition columns: MY_FUNC(A.COL) = value
    - Constraints through navigation functions: NEXT(A.COL) = value or PREV(A.COL) = value
    - Multiple must-appear variables with constraints on different partition columns
    - Merging constraints into a single WHERE clause
    """

    @property
    def name(self) -> str:
        return "define_partition_pushdown"

    @staticmethod
    def _input_pruning_is_safe(ast: RPRAbstractSyntaxTree) -> bool:
        if ast.pattern is None or ast.pattern.get_min_row_num() == 0:
            return False
        return (
            ast.rows_per_match or ""
        ).strip().upper() != ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        if not self._input_pruning_is_safe(ast):
            return False
        # Need PARTITION BY clause
        if (
            ast.partition_by is None
            or len(ast.partition_by) == 0
            or ast.rows_per_match == "ALL ROWS PER MATCH WITH UNMATCHED ROWS"
        ):
            return False
        partition_cols_upper = {c.strip().upper() for c in ast.partition_by}

        # Get must-appear variables
        must_appear_vars = ast.pattern.get_must_appear_var()
        if not must_appear_vars or len(must_appear_vars) == 0:
            return False

        # Check if any must-appear variable has constraints on partition columns
        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue

            # Extract all constraints on partition columns for this variable
            constraints = self._extract_all_partition_constraints(
                condition.expression, var_name, partition_cols_upper
            )
            if len(constraints) > 0:
                # Found at least one pushable constraint
                return True

        return False

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if not self._input_pruning_is_safe(ast):
            return False
        if ast.partition_by is None or len(ast.partition_by) == 0:
            return False

        must_appear_vars = ast.pattern.get_must_appear_var()
        if not must_appear_vars or len(must_appear_vars) == 0:
            return False

        partition_cols_upper = {c.strip().upper() for c in ast.partition_by}

        # Step 1: Collect per-variable constraints
        per_var_constraints = {}  # {var_name: {col_upper: [normalized_Expr]}}
        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue
            constraints = self._extract_all_partition_constraints(
                condition.expression, var_name, partition_cols_upper
            )
            if constraints:
                per_var_constraints[var_name] = constraints

        if not per_var_constraints:
            return False

        # Step 2: Build WHERE clauses (deduplicated by string representation).
        # Keep every predicate instead of intersecting IN lists in Python: textual
        # equality is not SQL value equality (for example, 1 and 1.0), and replacing
        # the IN predicates must not discard other constraints on the same column.
        seen_where = set()
        where_clauses = []

        for col in ast.partition_by:
            col_upper = col.strip().upper()

            # Collect individual constraints for this column (deduplicated), then
            # sort for deterministic output regardless of set iteration order.
            col_where_strs = []
            for var_name, var_constr in per_var_constraints.items():
                if col_upper in var_constr:
                    for expr in var_constr[col_upper]:
                        expr_str = expr.to_string()
                        if expr_str.startswith("(") and expr_str.endswith(")"):
                            expr_str = expr_str[1:-1]
                        if expr_str not in seen_where:
                            seen_where.add(expr_str)
                            col_where_strs.append(expr_str)
            col_where_strs.sort()
            where_clauses.extend(col_where_strs)

        if not where_clauses:
            return False

        where_str = " AND ".join(where_clauses)

        input_table = ast.input_table.strip()
        target_fragment = f"WHERE {where_str}"
        if target_fragment in input_table:
            return False

        # Push to WHERE clause
        ast.input_table = f"(SELECT * FROM {ast.input_table} WHERE {where_str})"

        # Step 3: Simplify DEFINE conditions — remove pushed partition constraints
        # that have no physical navigation (they are guaranteed true for all input rows)
        pushed_cols = set()
        for var_constr in per_var_constraints.values():
            for col in var_constr:
                if col in partition_cols_upper:
                    pushed_cols.add(col)

        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue
            new_expr = self._simplify_define_condition(
                condition.expression, var_name, pushed_cols
            )
            if new_expr is None:
                condition.expression = BinaryExpr(Literal(1), "=", Literal(1))
            else:
                condition.expression = new_expr

        return True

    def _extract_all_partition_constraints(
        self, expr: Expr, var_name: str, partition_cols: set
    ) -> Dict[str, List[Expr]]:
        """
        Extract ALL constraints (equality, inequalities, IN, NOT IN, etc.) on partition columns.

        Returns: {partition_col -> [Expr1, Expr2, ...]}
        where each Expr is an expression that constrains that partition column.

        Examples:
        - From "A.COL1 = 'ABC'" returns {'COL1': [ColumnRef(.COL1) = 'ABC']}
        - From "NEXT(A.C1) = 4" returns {'C1': [ColumnRef(.C1) = 4]}
        - From "C.C3 > 10" returns {'C3': [ColumnRef(.C3) > 10]}
        - From "A.REGION IN ('US', 'CA')" returns {'REGION': [InExpr(...)]}
        - From "A.REGION NOT IN (...)" returns {'REGION': [InExpr(..., is_not=True)]}
        """
        constraints = {}

        if expr is None:
            return constraints

        if isinstance(expr, SemanticsExpr):
            return self._extract_all_partition_constraints(
                expr.expr, var_name, partition_cols
            )

        if isinstance(expr, BinaryExpr):
            if expr.op == "AND":
                # Recursively extract from both sides
                left_constraints = self._extract_all_partition_constraints(
                    expr.left, var_name, partition_cols
                )
                right_constraints = self._extract_all_partition_constraints(
                    expr.right, var_name, partition_cols
                )

                # Merge constraints
                for col, exprs in left_constraints.items():
                    if col in constraints:
                        constraints[col].extend(exprs)
                    else:
                        constraints[col] = list(exprs)

                for col, exprs in right_constraints.items():
                    if col in constraints:
                        constraints[col].extend(exprs)
                    else:
                        constraints[col] = list(exprs)

                return constraints

            else:
                # Physical NEXT/PREV may yield NULL at a partition boundary. It is
                # not equivalent to reading the (constant) partition column itself.
                if self._has_physical_navigation(expr):
                    return constraints

                # This is a comparison or other operator
                # Check if it constrains a partition column
                col_name = self._get_constrained_partition_column(expr, var_name)
                if col_name and self._is_row_independent_binary_constraint(
                    expr, col_name, var_name
                ):
                    # Extract just the column part without qualifier
                    normalized_expr = self._normalize_to_column_reference(
                        expr, col_name, var_name
                    )
                    if normalized_expr:
                        constraints[col_name] = [normalized_expr]
                else:
                    referenced_cols = self._partition_only_independent_columns(
                        expr, var_name, partition_cols
                    )
                    if referenced_cols:
                        normalized_expr = self._remove_column_qualifier(expr)
                        for referenced_col in referenced_cols:
                            constraints.setdefault(referenced_col, []).append(
                                normalized_expr
                            )

        elif isinstance(expr, InExpr):
            # Handle IN and NOT IN expressions: expr IN (v1, v2, ...) or expr NOT IN (v1, v2, ...)
            if self._has_physical_navigation(expr):
                return constraints
            col_name = self._extract_column_from_expr(expr.expr, var_name)
            if (
                col_name
                and self._is_target_column_expr(expr.expr, col_name, var_name)
                and all(
                    isinstance(self._unwrap_semantics(v), Literal) for v in expr.values
                )
            ):
                # Normalize the InExpr (remove qualifiers and navigation functions)
                normalized_expr = self._normalize_in_expr(expr, col_name, var_name)
                if normalized_expr:
                    constraints[col_name] = [normalized_expr]
            else:
                referenced_cols = self._partition_only_independent_columns(
                    expr, var_name, partition_cols
                )
                if referenced_cols:
                    normalized_expr = self._remove_column_qualifier(expr)
                    for referenced_col in referenced_cols:
                        constraints.setdefault(referenced_col, []).append(
                            normalized_expr
                        )

        return constraints

    def _partition_only_independent_columns(
        self, expr: Expr, var_name: str, partition_cols: set
    ) -> Optional[set]:
        """Return columns when *expr* is constant throughout a partition.

        Ordinary scalar functions, including UDFs, are independent according
        to the DEFINE expression model.  They remain partition-constant when
        all their arguments recursively contain only literals and references
        to the current variable's partition columns.  Known navigation,
        aggregate, and match-dependent functions are rejected.
        """
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            return self._partition_only_independent_columns(
                expr.expr, var_name, partition_cols
            )
        if isinstance(expr, Literal):
            return set()
        if isinstance(expr, ColumnRef):
            if expr.name == "*":
                return None
            column = expr.name.upper()
            if column not in partition_cols:
                return None
            if (
                expr.qualifier is not None
                and expr.qualifier.upper() != var_name.upper()
            ):
                return None
            return {column}
        if isinstance(expr, FunctionCall):
            if not expr.name or expr.name.upper() in DEPENDENT_FUNCS:
                return None
            children = tuple(expr.args)
        elif isinstance(expr, BinaryExpr):
            children = (expr.left, expr.right)
        elif isinstance(expr, BetweenExpr):
            children = (expr.expr, expr.lower, expr.upper)
        elif isinstance(expr, InExpr):
            children = (expr.expr, *expr.values)
        elif isinstance(expr, CaseExpr):
            children = tuple(
                child
                for condition, result in expr.cases
                for child in (condition, result)
            ) + ((expr.else_expr,) if expr.else_expr is not None else ())
        elif isinstance(expr, (IsNullExpr, NotExpr)):
            children = (expr.expr,)
        else:
            return None

        referenced_cols = set()
        for child in children:
            child_cols = self._partition_only_independent_columns(
                child, var_name, partition_cols
            )
            if child_cols is None:
                return None
            referenced_cols.update(child_cols)
        return referenced_cols

    def _unwrap_semantics(self, expr: Expr) -> Expr:
        """Remove transparent RUNNING/FINAL wrappers for structural checks."""
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        return expr

    def _is_target_column_expr(self, expr: Expr, col_name: str, var_name: str) -> bool:
        """Whether *expr* denotes only the current row's target partition column."""
        expr = self._unwrap_semantics(expr)
        if isinstance(expr, ColumnRef):
            return bool(
                expr.name
                and expr.name.upper() == col_name
                and (
                    expr.qualifier is None or expr.qualifier.upper() == var_name.upper()
                )
            )
        if (
            isinstance(expr, FunctionCall)
            and expr.name
            and expr.name.upper() in ("NEXT", "PREV")
        ):
            # _has_physical_navigation rejects every non-zero/dynamic offset before
            # this method is reached; offset zero is equivalent to the current row.
            return bool(
                expr.args
                and self._is_target_column_expr(expr.args[0], col_name, var_name)
            )
        return False

    def _is_row_independent_binary_constraint(
        self, expr: BinaryExpr, col_name: str, var_name: str
    ) -> bool:
        """Accept only target-column versus literal predicates.

        A predicate such as A.PT = A.X is true only for a candidate A row.  It
        cannot filter every row of the partition without potentially deleting
        rows needed by other pattern variables.
        """
        if expr.op.upper() not in (
            "=",
            "!=",
            "<>",
            ">",
            ">=",
            "<",
            "<=",
            "LIKE",
            "NOT LIKE",
        ):
            return False
        left = self._unwrap_semantics(expr.left)
        right = self._unwrap_semantics(expr.right)
        return (
            self._is_target_column_expr(left, col_name, var_name)
            and isinstance(right, Literal)
        ) or (
            self._is_target_column_expr(right, col_name, var_name)
            and isinstance(left, Literal)
        )

    def _get_constrained_partition_column(
        self, expr: Expr, var_name: str
    ) -> Optional[str]:
        """
        Check if expression constrains a partition column and return the column name.
        Returns: 'COL_NAME' if expression constrains partition column, None otherwise.

        Examples:
        - "A.C1 = 4" -> 'C1'
        - "NEXT(A.C1) = 4" -> 'C1'
        - "C2 = 'B'" -> 'C2'
        - "PREV(C.C3) > 10" -> 'C3'
        """
        if isinstance(expr, SemanticsExpr):
            return self._get_constrained_partition_column(expr.expr, var_name)

        if not isinstance(expr, BinaryExpr):
            return None

        # Extract column names from both sides
        left_col = self._extract_column_from_expr(expr.left, var_name)
        right_col = self._extract_column_from_expr(expr.right, var_name)

        if left_col:
            return left_col
        if right_col:
            return right_col

        return None

    def _extract_column_from_expr(self, expr: Expr, var_name: str) -> Optional[str]:
        """
        Extract column name from an expression.
        Handles: ColumnRef, NEXT(var.col), PREV(var.col), etc.
        Returns: 'COL_NAME' in uppercase, or None if no column found.
        """
        if expr is None:
            return None

        if isinstance(expr, SemanticsExpr):
            return self._extract_column_from_expr(expr.expr, var_name)

        if isinstance(expr, ColumnRef):
            # Direct column reference
            if expr.name:
                return expr.name.upper()
            return None

        if isinstance(expr, FunctionCall):
            # Handle NEXT(A.C1), PREV(A.C3), etc.
            if (
                expr.name
                and expr.name.upper() in ("NEXT", "PREV")
                and len(expr.args) > 0
            ):
                arg = expr.args[0]
                if isinstance(arg, SemanticsExpr):
                    arg = arg.expr

                if isinstance(arg, ColumnRef):
                    # Check if this is the same variable
                    if arg.name:
                        if (
                            arg.qualifier is None
                            or arg.qualifier.upper() == var_name.upper()
                        ):
                            return arg.name.upper()

        return None

    def _normalize_to_column_reference(
        self, expr: Expr, col_name: str, var_name: str
    ) -> Optional[Expr]:
        """
        Normalize expression by removing navigation function wrappers but keeping constraint.

        Examples:
        - "NEXT(A.C1) = 4" -> "C1 = 4"
        - "PREV(C.C3) > 10" -> "C3 > 10"
        - "A.C2 = 'B'" -> "C2 = 'B'"
        """
        if isinstance(expr, SemanticsExpr):
            normalized = self._normalize_to_column_reference(
                expr.expr, col_name, var_name
            )
            if normalized is None:
                return None
            return (
                SemanticsExpr(expr.semantics, normalized)
                if normalized != expr.expr
                else expr
            )

        if not isinstance(expr, BinaryExpr):
            return expr

        # Process left and right sides
        left = expr.left
        right = expr.right

        if isinstance(left, SemanticsExpr):
            left = left.expr
        if isinstance(right, SemanticsExpr):
            right = right.expr

        # Check if left side is navigation function that should be unwrapped
        if (
            isinstance(left, FunctionCall)
            and left.name
            and left.name.upper() in ("NEXT", "PREV")
        ):
            if len(left.args) > 0:
                arg = left.args[0]
                if isinstance(arg, SemanticsExpr):
                    arg = arg.expr

                if isinstance(arg, ColumnRef):
                    col_from_arg = arg.name.upper() if arg.name else None
                    if col_from_arg == col_name and (
                        arg.qualifier is None
                        or arg.qualifier.upper() == var_name.upper()
                    ):
                        # Replace navigation function with column reference (keep original case)
                        left = ColumnRef(
                            None, arg.name if arg.name else col_name.lower()
                        )

        # Check if right side is navigation function that should be unwrapped
        if (
            isinstance(right, FunctionCall)
            and right.name
            and right.name.upper() in ("NEXT", "PREV")
        ):
            if len(right.args) > 0:
                arg = right.args[0]
                if isinstance(arg, SemanticsExpr):
                    arg = arg.expr

                if isinstance(arg, ColumnRef):
                    col_from_arg = arg.name.upper() if arg.name else None
                    if col_from_arg == col_name and (
                        arg.qualifier is None
                        or arg.qualifier.upper() == var_name.upper()
                    ):
                        # Replace navigation function with column reference (keep original case)
                        right = ColumnRef(
                            None, arg.name if arg.name else col_name.lower()
                        )

        # Remove qualifier from column references to match partition column names
        left = self._remove_column_qualifier(left)
        right = self._remove_column_qualifier(right)

        return BinaryExpr(left, expr.op, right)

    def _remove_column_qualifier(self, expr: Expr) -> Expr:
        """Remove variable qualifier from column references."""
        if expr is None:
            return expr

        if isinstance(expr, SemanticsExpr):
            inner = self._remove_column_qualifier(expr.expr)
            return SemanticsExpr(expr.semantics, inner) if inner != expr.expr else expr

        if isinstance(expr, ColumnRef):
            # Create a new ColumnRef without qualifier
            return ColumnRef(None, expr.name)

        if isinstance(expr, BinaryExpr):
            left = self._remove_column_qualifier(expr.left)
            right = self._remove_column_qualifier(expr.right)
            return BinaryExpr(left, expr.op, right)

        if isinstance(expr, FunctionCall):
            return FunctionCall(
                expr.name, [self._remove_column_qualifier(arg) for arg in expr.args]
            )

        if isinstance(expr, BetweenExpr):
            e = self._remove_column_qualifier(expr.expr)
            lower = self._remove_column_qualifier(expr.lower)
            upper = self._remove_column_qualifier(expr.upper)
            return BetweenExpr(e, lower, upper)

        if isinstance(expr, InExpr):
            e = self._remove_column_qualifier(expr.expr)
            values = [self._remove_column_qualifier(v) for v in expr.values]
            return InExpr(e, values, expr.is_not)

        if isinstance(expr, CaseExpr):
            cases = [
                (
                    self._remove_column_qualifier(condition),
                    self._remove_column_qualifier(result),
                )
                for condition, result in expr.cases
            ]
            else_expr = (
                self._remove_column_qualifier(expr.else_expr)
                if expr.else_expr is not None
                else None
            )
            return CaseExpr(cases, else_expr)

        if isinstance(expr, IsNullExpr):
            return IsNullExpr(self._remove_column_qualifier(expr.expr), expr.is_not)

        if isinstance(expr, NotExpr):
            return NotExpr(self._remove_column_qualifier(expr.expr))

        return expr

    def _has_physical_navigation(self, expr: "Expr") -> bool:
        """Check if expression contains physical navigation (NEXT/PREV with non-zero offset)."""
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._has_physical_navigation(expr.expr)
        if isinstance(expr, FunctionCall):
            if expr.name and expr.name.upper() in ("NEXT", "PREV"):
                # Default offset (no second arg) means offset=1, which is physical navigation
                if len(expr.args) <= 1:
                    return True
                offset_arg = expr.args[1]
                if isinstance(offset_arg, SemanticsExpr):
                    offset_arg = offset_arg.expr
                if isinstance(offset_arg, Literal):
                    try:
                        return int(offset_arg.value) != 0
                    except (ValueError, TypeError):
                        return True
                return True
            return any(self._has_physical_navigation(a) for a in expr.args)
        if isinstance(expr, BinaryExpr):
            return self._has_physical_navigation(
                expr.left
            ) or self._has_physical_navigation(expr.right)
        if isinstance(expr, InExpr):
            return self._has_physical_navigation(expr.expr)
        if isinstance(expr, BetweenExpr):
            return self._has_physical_navigation(expr.expr)
        return False

    def _can_simplify_atom(self, expr: "Expr", var_name: str, pushed_cols: set) -> bool:
        """Check if this expression atom can be simplified to 1=1 after pushdown."""
        # A column may have been pushed because of another variable. Apply the same
        # safety checks used during extraction so that an unpushed row-dependent
        # predicate on that column is never erased here.
        if self._has_physical_navigation(expr):
            return False
        if isinstance(expr, SemanticsExpr):
            return self._can_simplify_atom(expr.expr, var_name, pushed_cols)
        if isinstance(expr, BinaryExpr) and expr.op != "AND":
            col = self._get_constrained_partition_column(expr, var_name)
            if bool(
                col is not None
                and col in pushed_cols
                and self._is_row_independent_binary_constraint(expr, col, var_name)
            ):
                return True
            referenced_cols = self._partition_only_independent_columns(
                expr, var_name, pushed_cols
            )
            return bool(referenced_cols)
        if isinstance(expr, InExpr):
            col = self._extract_column_from_expr(expr.expr, var_name)
            if bool(
                col is not None
                and col in pushed_cols
                and self._is_target_column_expr(expr.expr, col, var_name)
                and all(
                    isinstance(self._unwrap_semantics(v), Literal) for v in expr.values
                )
            ):
                return True
            referenced_cols = self._partition_only_independent_columns(
                expr, var_name, pushed_cols
            )
            return bool(referenced_cols)
        return False

    def _simplify_define_condition(
        self, expr: "Expr", var_name: str, pushed_cols: set
    ) -> "Optional[Expr]":
        """
        Simplify DEFINE condition by removing pushed partition constraints.
        Returns: simplified expression, or None if the entire condition was simplified away.
        """
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            inner = self._simplify_define_condition(expr.expr, var_name, pushed_cols)
            if inner is None:
                return None
            return SemanticsExpr(expr.semantics, inner)
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            left = self._simplify_define_condition(expr.left, var_name, pushed_cols)
            right = self._simplify_define_condition(expr.right, var_name, pushed_cols)
            if left is None and right is None:
                return None
            if left is None:
                return right
            if right is None:
                return left
            return BinaryExpr(left, expr.op, right)
        # Leaf atom: check if it can be simplified away
        if self._can_simplify_atom(expr, var_name, pushed_cols):
            return None
        return expr

    def _normalize_in_expr(
        self, expr: InExpr, col_name: str, var_name: str
    ) -> Optional[InExpr]:
        """
        Normalize InExpr by removing navigation function wrappers and column qualifiers.

        Examples:
        - "A.REGION IN ('US', 'CA')" -> "REGION IN ('US', 'CA')"
        - "NEXT(A.REGION) IN (...)" -> "REGION IN (...)"
        - "A.REGION NOT IN (...)" -> "REGION NOT IN (...)"
        """
        if expr is None or not isinstance(expr, InExpr):
            return None

        # Process the expr part (column reference)
        col_expr = expr.expr

        if isinstance(col_expr, SemanticsExpr):
            col_expr = col_expr.expr

        # Check if it's a navigation function that should be unwrapped
        if (
            isinstance(col_expr, FunctionCall)
            and col_expr.name
            and col_expr.name.upper() in ("NEXT", "PREV")
        ):
            if len(col_expr.args) > 0:
                arg = col_expr.args[0]
                if isinstance(arg, SemanticsExpr):
                    arg = arg.expr

                if isinstance(arg, ColumnRef):
                    col_from_arg = arg.name.upper() if arg.name else None
                    if col_from_arg == col_name and (
                        arg.qualifier is None
                        or arg.qualifier.upper() == var_name.upper()
                    ):
                        # Replace navigation function with column reference (keep original case)
                        col_expr = ColumnRef(
                            None, arg.name if arg.name else col_name.lower()
                        )

        # Remove qualifier from column reference
        col_expr = self._remove_column_qualifier(col_expr)

        # Remove qualifiers from value expressions (though they're usually literals)
        normalized_values = [self._remove_column_qualifier(v) for v in expr.values]

        return InExpr(col_expr, normalized_values, expr.is_not)


partition_constraint_pushdown_test_cases = [
    # Test case 1: Basic partition constraint pushdown with multiple must-appear variables
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY C1, C2, C3
    ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS NEXT(A.C1) = 4 AND VAL > 10, B AS C2 = 'B', C AS PREV(C.C3) > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE C2 = 'B') MATCH_RECOGNIZE(
    PARTITION BY C1, C2, C3
    ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS NEXT(A.C1) = 4 AND VAL > 10, B AS 1 = 1, C AS PREV(C.C3) > 10
)""",
    ),
    # Test case 2: Single variable constraint on partition column
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS A.PT = 'NYC'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE PT = 'NYC') MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS 1 = 1
)""",
    ),
    # Test case 3: Multiple variables constrain different partition columns
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY REGION, CITY
    ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.REGION = 'US', B AS B.CITY = 'NYC'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE REGION = 'US' AND CITY = 'NYC') MATCH_RECOGNIZE(
    PARTITION BY REGION, CITY
    ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS 1 = 1, B AS 1 = 1
)""",
    ),
    # Test case 4: Constraint through navigation function (NEXT), simplified after pushdown
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS NEXT(A.PT) = 'LA'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS NEXT(A.PT) = 'LA'
)""",
    ),
    # Test case 5: Constraint through navigation function (PREV), simplified after pushdown
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY DEPT
    ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS PREV(A.DEPT) = 'SALES'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY DEPT
    ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A)
    DEFINE A AS PREV(A.DEPT) = 'SALES'
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        DefinePartitionPushdownRule,
        partition_constraint_pushdown_test_cases,
        "define partition pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
