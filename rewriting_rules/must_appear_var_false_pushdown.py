if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from decimal import Decimal, InvalidOperation

from .utils import *


class MustAppearVarFalsePushdownRule(RewriteRule):
    """
    Rule description:
        If a variable MUST appear in PATTERN (checked via get_must_appear_var),
        and its DEFINE condition can never be satisfied, then the entire query
        can never match any rows. Push 1 != 1 to the input table's WHERE clause
        so the input scan returns no rows early.

    Always-false condition families:
    - A AS 1 != 1
    - A AS A.OPEN > A.OPEN
    - A AS OPEN > OPEN
    - A AS PREV(A.OPEN) < PREV(A.OPEN)
    - A AS A.OPEN + 1 < A.OPEN
    - A AS PREV(A.PT) != A.PT when PT is a PARTITION BY column
    - A AS A.PT IN ('US') AND A.PT IN ('EU')
    - A AS A.CODE LIKE 'ERR%' AND A.CODE LIKE 'WARN%'
    - A AS B.VOL > 10 when B cannot appear before A under RUNNING semantics
    - A AS D.TS < 100 when D is a SUBSET whose members all appear after A
    - B AS MATCH_NUMBER() > 1 because the first match number is 1 and must be
      satisfiable before later match numbers can exist

    Boolean handling is conservative:
    - AND is false if any conjunct is known false, or if an IN/LIKE chain has a conflict.
    - OR is only treated as false for supported path-sensitive occurrence-count
      constraints. Unknown OR branches are left unchanged to avoid over-pruning.

    Please note that non-deterministic functions are not allowed in the DEFINE clause.
    Besides, we think 'A' = 'A ' is false.

    Rule Application
    ↓
    is_applicable()
    ├─→ Check must-appear variables exist
    ├─→ For each must-appear variable:
    │   └─→ Check if DEFINE condition is always false
    │       ├─→ Literal false? (1 != 1)
    │       ├─→ Self-comparison? (X > X)
    │       ├─→ Arithmetic contradiction?
    │       ├─→ Partition/navigation contradiction?
    │       ├─→ IN/LIKE conflicts in an AND chain?
    │       ├─→ Future-only variable or SUBSET reference?
    │       ├─→ MATCH_NUMBER() condition false for match number 1?
    │       └─→ Supported occurrence-count OR paths all impossible?
    │
    └─→ Check partition key conflicts
        ├─→ Multiple must-appear variables?
        ├─→ PARTITION BY clause exists?
        └─→ Conflicting equality constraints?

    apply()
    ├─→ Confirm always-false condition found
    ├─→ Replace WHERE clause with: WHERE 1 != 1
    └─→ Return True (optimization applied)
    """

    @property
    def name(self) -> str:
        return "must_appear_var_false_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        result = self._has_no_output_condition(ast)
        self._pending_decision = (self._decision_key(ast), result)
        return result

    def _decision_key(self, ast: RPRAbstractSyntaxTree):
        definitions = getattr(getattr(ast, "define", None), "definitions", {})
        return (
            id(ast),
            id(ast.pattern),
            id(ast.subsets),
            tuple(ast.partition_by or []),
            tuple(
                (name, id(condition), id(getattr(condition, "expression", None)))
                for name, condition in definitions.items()
            ),
        )

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        pending = getattr(self, "_pending_decision", None)
        if pending is not None and pending[0] == self._decision_key(ast):
            has_false_condition = pending[1]
        else:
            has_false_condition = self._has_no_output_condition(ast)
        self._pending_decision = None

        if not has_false_condition:
            return False

        # Push 1 != 1 to WHERE clause
        input_table = ast.input_table.strip()

        # Handle case: input_table is already a SELECT statement with WHERE clause
        # e.g., "(SELECT * FROM TABLE WHERE OPEN < 10)"
        if "WHERE" in input_table.upper():
            # Replace the WHERE clause with WHERE 1 != 1
            import re

            # Match WHERE clause (case insensitive) and replace with WHERE 1 != 1
            pattern = r"WHERE\s+[^)]*(?=\)?\s*$)"
            replacement = "WHERE 1 != 1"
            new_input_table = re.sub(
                pattern, replacement, input_table, flags=re.IGNORECASE, count=1
            )
            if new_input_table == input_table:
                return False
            ast.input_table = new_input_table
        else:
            # No WHERE clause yet, wrap it
            ast.input_table = f"(SELECT * FROM {input_table} WHERE 1 != 1)"

        return True

    def _has_no_output_condition(self, ast: RPRAbstractSyntaxTree) -> bool:
        # Get must-appear variables
        must_appear_vars = ast.pattern.get_must_appear_var()
        if not must_appear_vars or len(must_appear_vars) == 0:
            return False

        # Get partition columns as list
        partition_cols = ast.partition_by if ast.partition_by else []

        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue

            # Pass pattern/subsets so RUNNING-semantics dependencies can be checked.
            if self._is_always_false(
                condition.expression, var_name, partition_cols, ast.pattern, ast.subsets
            ):
                return True

        # Check for partition column conflicts
        if len(must_appear_vars) > 1 and partition_cols:
            return self._has_partition_column_conflict(ast, must_appear_vars)

        return False

    def _is_always_false(
        self,
        expr: Expr,
        var_name: str = None,
        partition_cols: List[str] = None,
        pattern=None,
        subsets: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        """
        Check if an expression is always false.

        Cases:
        1. Literal false comparisons: 1 != 1, 0 = 1, etc.
        2. Self-comparisons: X > X, X < X, X != X
        3. Same function calls with contradictory operators: PREV(x, k) < PREV(x, k)
        4. Arithmetic contradictions: X + 1 < X, X * 2 > X * 2 + 1
        5. Navigation partition column conflicts: PREV(A.PT) != A.PT when PT is partition column
        6. IN clause conflicts: X IN (set1) AND X IN (set2) with disjoint sets
        7. LIKE clause conflicts: X LIKE 'pattern1' AND X LIKE 'pattern2' with incompatible patterns
        8. Future-only cross-variable references: A depends on B.COL, but B cannot
           appear before A anywhere in PATTERN under RUNNING semantics
        9. Future-only SUBSET references: A depends on D.COL, but every member
           of D can only appear after A
        10. MATCH_NUMBER() constraints that reject match number 1
        """
        if expr is None:
            return False

        # Unwrap SemanticsExpr
        if isinstance(expr, SemanticsExpr):
            return self._is_always_false(
                expr.expr, var_name, partition_cols, pattern, subsets
            )

        if var_name and pattern is not None:
            if self._is_match_number_one_false(expr):
                return True

        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            # Check IN conflicts
            if self._is_in_clause_false_in_chain(expr, var_name):
                return True

            # Check LIKE conflicts
            if self._is_like_clause_false_in_chain(expr, var_name):
                return True

            # Check if any conjunct is always false
            return self._is_always_false(
                expr.left, var_name, partition_cols, pattern, subsets
            ) or self._is_always_false(
                expr.right, var_name, partition_cols, pattern, subsets
            )

        if isinstance(expr, BinaryExpr) and expr.op == "OR":
            occurrence_false_paths = self._false_occurrence_paths(
                expr, var_name, pattern
            )
            if occurrence_false_paths is None:
                return False
            return all(any(path) for path in occurrence_false_paths)

        # Check literal false: 1 != 1, 0 = 1, etc.
        if isinstance(expr, BinaryExpr):
            if self._is_literal_false(expr):
                return True

            # Check self-comparison: X > X, X < X
            if self._is_self_comparison_false(expr):
                return True

            # Check arithmetic contradictions: X + 1 > X, etc.
            if self._is_arithmetic_always_false(expr):
                return True

            # Check navigation partition column conflicts
            if var_name and partition_cols:
                if self._is_navigation_partition_false(expr, var_name, partition_cols):
                    return True

            if var_name and pattern is not None:
                if self._is_future_only_dependency_false(expr, var_name, pattern):
                    return True
                if self._is_future_only_subset_dependency_false(
                    expr, var_name, pattern, subsets
                ):
                    return True

            # Check LIKE clause conflicts
            if self._is_like_clause_false(expr, var_name):
                return True

        return False

    def _is_count_constraint_false(self, expr: Expr, var_name: str, pattern) -> bool:
        parsed = self._parse_count_comparison(expr)
        if parsed is None:
            return False

        count_range = self._count_range_at_var(pattern, parsed["var_name"], var_name)
        if count_range is None:
            return False

        min_count, max_count = count_range
        if self._count_interval_conflicts(
            min_count, max_count, parsed["op"], parsed["value"]
        ):
            return True

        occurrence_paths = self._count_occurrence_paths_at_var(
            pattern, parsed["var_name"], var_name
        )
        if occurrence_paths is None:
            return False

        return all(
            any(
                self._count_interval_conflicts(
                    occ_min, occ_max, parsed["op"], parsed["value"]
                )
                for occ_min, occ_max in path
            )
            for path in occurrence_paths
        )

    def _false_occurrence_paths(
        self, expr: Expr, var_name: str, pattern
    ) -> Optional[List[List[bool]]]:
        if expr is None or var_name is None or pattern is None:
            return None

        if isinstance(expr, SemanticsExpr):
            return self._false_occurrence_paths(expr.expr, var_name, pattern)

        if isinstance(expr, BinaryExpr) and expr.op == "OR":
            left_paths = self._false_occurrence_paths(expr.left, var_name, pattern)
            right_paths = self._false_occurrence_paths(expr.right, var_name, pattern)
            if not self._same_bool_path_shape(left_paths, right_paths):
                return None
            return [
                [
                    left_value and right_value
                    for left_value, right_value in zip(left_path, right_path)
                ]
                for left_path, right_path in zip(left_paths, right_paths)
            ]

        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            left_paths = self._false_occurrence_paths(expr.left, var_name, pattern)
            right_paths = self._false_occurrence_paths(expr.right, var_name, pattern)
            if not self._same_bool_path_shape(left_paths, right_paths):
                return None
            return [
                [
                    left_value or right_value
                    for left_value, right_value in zip(left_path, right_path)
                ]
                for left_path, right_path in zip(left_paths, right_paths)
            ]

        parsed = self._parse_count_comparison(expr)
        if parsed is None:
            return None

        occurrence_paths = self._count_occurrence_paths_at_var(
            pattern, parsed["var_name"], var_name
        )
        if occurrence_paths is None:
            return None

        return [
            [
                self._count_interval_conflicts(
                    occ_min, occ_max, parsed["op"], parsed["value"]
                )
                for occ_min, occ_max in path
            ]
            for path in occurrence_paths
        ]

    def _same_bool_path_shape(
        self,
        left_paths: Optional[List[List[bool]]],
        right_paths: Optional[List[List[bool]]],
    ) -> bool:
        if left_paths is None or right_paths is None:
            return False
        if len(left_paths) != len(right_paths):
            return False
        return all(
            len(left_path) == len(right_path)
            for left_path, right_path in zip(left_paths, right_paths)
        )

    def _count_occurrence_paths_at_var(
        self, pattern, counted_var: Optional[str], current_var: str
    ) -> Optional[List[List[Tuple[int, Optional[int]]]]]:
        if (
            counted_var is not None
            and pattern.get_left_min_row_num(counted_var) is None
        ):
            return None

        summaries = self._count_occurrence_summaries(pattern, counted_var, current_var)
        if summaries is None:
            return None

        return [occurrences for _, _, occurrences in summaries]

    def _count_occurrence_summaries(
        self, pattern, counted_var: Optional[str], current_var: str
    ) -> Optional[List[Tuple[int, Optional[int], List[Tuple[int, Optional[int]]]]]]:
        if isinstance(pattern, VariablePattern):
            count = 1 if counted_var is None or pattern.name == counted_var else 0
            occurrences = [(count, count)] if pattern.name == current_var else []
            return [(count, count, occurrences)]

        if isinstance(pattern, QuantifiedPattern):
            child_summaries = self._count_occurrence_summaries(
                pattern.pattern, counted_var, current_var
            )
            if child_summaries is None:
                return None

            total_min, total_max = self._count_range_in_pattern(pattern, counted_var)
            _, inner_max = self._count_range_in_pattern(pattern.pattern, counted_var)
            repeat_prefix_max = self._repeat_prefix_max(pattern.atMost, inner_max)

            summaries: List[
                Tuple[int, Optional[int], List[Tuple[int, Optional[int]]]]
            ] = []
            if pattern.atLeast == 0:
                summaries.append((0, 0, []))

            if pattern.atMost == 0:
                return summaries

            for _, _, child_occurrences in child_summaries:
                adjusted = [
                    (occ_min, self._add_optional_counts(repeat_prefix_max, occ_max))
                    for occ_min, occ_max in child_occurrences
                ]
                summaries.append((total_min, total_max, adjusted))
            return summaries

        if isinstance(pattern, ConcatenationPattern):
            summaries: List[
                Tuple[int, Optional[int], List[Tuple[int, Optional[int]]]]
            ] = [(0, 0, [])]
            for child in pattern.patterns:
                child_summaries = self._count_occurrence_summaries(
                    child, counted_var, current_var
                )
                if child_summaries is None:
                    return None

                combined: List[
                    Tuple[int, Optional[int], List[Tuple[int, Optional[int]]]]
                ] = []
                for prefix_min, prefix_max, prefix_occurrences in summaries:
                    for child_min, child_max, child_occurrences in child_summaries:
                        adjusted_child_occurrences = [
                            (
                                prefix_min + occ_min,
                                self._add_optional_counts(prefix_max, occ_max),
                            )
                            for occ_min, occ_max in child_occurrences
                        ]
                        combined.append(
                            (
                                prefix_min + child_min,
                                self._add_optional_counts(prefix_max, child_max),
                                prefix_occurrences + adjusted_child_occurrences,
                            )
                        )
                summaries = combined
            return summaries

        if isinstance(pattern, AlternationPattern):
            summaries: List[
                Tuple[int, Optional[int], List[Tuple[int, Optional[int]]]]
            ] = []
            for child in pattern.patterns:
                child_summaries = self._count_occurrence_summaries(
                    child, counted_var, current_var
                )
                if child_summaries is None:
                    return None
                summaries.extend(child_summaries)
            return summaries

        if isinstance(pattern, PermutationPattern):
            return None

        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._count_occurrence_summaries(
                pattern.pattern, counted_var, current_var
            )

        if isinstance(pattern, EmptyPattern):
            return [(0, 0, [])]

        return None

    def _parse_count_comparison(self, expr: Expr):
        if not isinstance(expr, BinaryExpr) or expr.op not in (
            ">",
            ">=",
            "<",
            "<=",
            "=",
            "!=",
            "<>",
        ):
            return None

        left = self._unwrap_semantics(expr.left)
        right = self._unwrap_semantics(expr.right)
        op = expr.op

        right_count_var = self._extract_count_star_var(right)
        if isinstance(left, Literal) and right_count_var is not False:
            op = self._reverse_count_comparison_op(op)
            left, right = right, left

        count_var = self._extract_count_star_var(left)
        if count_var is False:
            return None
        if not isinstance(right, Literal) or not isinstance(right.value, (int, float)):
            return None

        return {"var_name": count_var, "op": op, "value": float(right.value)}

    def _extract_count_star_var(self, expr: Expr):
        expr = self._unwrap_semantics(expr)
        if (
            not isinstance(expr, FunctionCall)
            or expr.name != "COUNT"
            or len(expr.args) != 1
        ):
            return False
        arg = self._unwrap_semantics(expr.args[0])
        if not isinstance(arg, ColumnRef) or arg.name != "*":
            return False
        return arg.qualifier

    def _reverse_count_comparison_op(self, op: str) -> Optional[str]:
        return {
            ">": "<",
            ">=": "<=",
            "<": ">",
            "<=": ">=",
            "=": "=",
            "!=": "!=",
            "<>": "<>",
        }.get(op)

    def _count_interval_conflicts(
        self, min_count: int, max_count: Optional[int], op: str, value: float
    ) -> bool:
        if op == ">":
            return max_count is not None and max_count <= value
        if op == ">=":
            return max_count is not None and max_count < value
        if op == "<":
            return min_count >= value
        if op == "<=":
            return min_count > value
        if op == "=":
            if not value.is_integer():
                return True
            target = int(value)
            return target < min_count or (max_count is not None and target > max_count)
        if op in ("!=", "<>"):
            return (
                value.is_integer()
                and max_count is not None
                and min_count == max_count == int(value)
            )
        return False

    def _count_range_at_var(
        self, pattern, counted_var: Optional[str], current_var: str
    ) -> Optional[Tuple[int, Optional[int]]]:
        if (
            counted_var is not None
            and pattern.get_left_min_row_num(counted_var) is None
        ):
            return None
        return self._count_range_at_var_node(pattern, counted_var, current_var)

    def _count_range_at_var_node(
        self, pattern, counted_var: Optional[str], current_var: str
    ) -> Optional[Tuple[int, Optional[int]]]:
        if isinstance(pattern, VariablePattern):
            if pattern.name != current_var:
                return None
            count = 1 if counted_var is None or pattern.name == counted_var else 0
            return count, count

        if isinstance(pattern, QuantifiedPattern):
            inner_occurrence = self._count_range_at_var_node(
                pattern.pattern, counted_var, current_var
            )
            if inner_occurrence is None:
                return None

            _, inner_count_max = self._count_range_in_pattern(
                pattern.pattern, counted_var
            )
            repeat_prefix_max = self._repeat_prefix_max(pattern.atMost, inner_count_max)
            return inner_occurrence[0], self._add_optional_counts(
                repeat_prefix_max, inner_occurrence[1]
            )

        if isinstance(pattern, ConcatenationPattern):
            prefix_min = 0
            prefix_max: Optional[int] = 0
            candidates: List[Tuple[int, Optional[int]]] = []
            for child in pattern.patterns:
                inner = self._count_range_at_var_node(child, counted_var, current_var)
                if inner is not None:
                    inner_min, inner_max = inner
                    candidates.append(
                        (
                            prefix_min + inner_min,
                            self._add_optional_counts(prefix_max, inner_max),
                        )
                    )

                child_min, child_max = self._count_range_in_pattern(child, counted_var)
                prefix_min += child_min
                prefix_max = self._add_optional_counts(prefix_max, child_max)
            return self._merge_count_ranges(candidates)

        if isinstance(pattern, AlternationPattern):
            ranges = [
                result
                for result in (
                    self._count_range_at_var_node(child, counted_var, current_var)
                    for child in pattern.patterns
                )
                if result is not None
            ]
            return self._merge_count_ranges(ranges)

        if isinstance(pattern, PermutationPattern):
            candidates = []
            for index, child in enumerate(pattern.patterns):
                inner = self._count_range_at_var_node(child, counted_var, current_var)
                if inner is None:
                    continue
                other_max: Optional[int] = 0
                for other_index, other in enumerate(pattern.patterns):
                    if other_index == index:
                        continue
                    _, other_child_max = self._count_range_in_pattern(
                        other, counted_var
                    )
                    other_max = self._add_optional_counts(other_max, other_child_max)
                candidates.append(
                    (inner[0], self._add_optional_counts(inner[1], other_max))
                )
            return self._merge_count_ranges(candidates)

        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._count_range_at_var_node(
                pattern.pattern, counted_var, current_var
            )

        return None

    def _count_range_in_pattern(
        self, pattern, counted_var: Optional[str]
    ) -> Tuple[int, Optional[int]]:
        if counted_var is None:
            return pattern.get_min_row_num(), pattern.get_max_row_num()

        if isinstance(pattern, VariablePattern):
            count = 1 if pattern.name == counted_var else 0
            return count, count

        if isinstance(pattern, QuantifiedPattern):
            inner_min, inner_max = self._count_range_in_pattern(
                pattern.pattern, counted_var
            )
            min_count = pattern.atLeast * inner_min
            if pattern.atMost is None:
                max_count = 0 if inner_max == 0 else None
            elif inner_max is None:
                max_count = None
            else:
                max_count = pattern.atMost * inner_max
            return min_count, max_count

        if isinstance(pattern, ConcatenationPattern):
            total_min = 0
            total_max: Optional[int] = 0
            for child in pattern.patterns:
                child_min, child_max = self._count_range_in_pattern(child, counted_var)
                total_min += child_min
                total_max = self._add_optional_counts(total_max, child_max)
            return total_min, total_max

        if isinstance(pattern, AlternationPattern):
            if not pattern.patterns:
                return 0, 0
            ranges = [
                self._count_range_in_pattern(child, counted_var)
                for child in pattern.patterns
            ]
            min_count = min(item[0] for item in ranges)
            max_count = (
                None
                if any(item[1] is None for item in ranges)
                else max(item[1] for item in ranges)
            )
            return min_count, max_count

        if isinstance(pattern, PermutationPattern):
            total_min = 0
            total_max: Optional[int] = 0
            for child in pattern.patterns:
                child_min, child_max = self._count_range_in_pattern(child, counted_var)
                total_min += child_min
                total_max = self._add_optional_counts(total_max, child_max)
            return total_min, total_max

        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._count_range_in_pattern(pattern.pattern, counted_var)

        return 0, 0

    def _merge_count_ranges(
        self, ranges: List[Tuple[int, Optional[int]]]
    ) -> Optional[Tuple[int, Optional[int]]]:
        if not ranges:
            return None
        min_count = min(item[0] for item in ranges)
        max_count = (
            None
            if any(item[1] is None for item in ranges)
            else max(item[1] for item in ranges)
        )
        return min_count, max_count

    def _repeat_prefix_max(
        self, at_most: Optional[int], inner_max: Optional[int]
    ) -> Optional[int]:
        if at_most is None:
            return 0 if inner_max == 0 else None
        if inner_max is None:
            return None
        return max(0, at_most - 1) * inner_max

    def _add_optional_counts(
        self, left: Optional[int], right: Optional[int]
    ) -> Optional[int]:
        if left is None or right is None:
            return None
        return left + right

    def _extract_navigation_offset(self, func: FunctionCall) -> Optional[int]:
        return _extract_navigation_offset(func)

    def _unwrap_semantics(self, expr: Expr) -> Expr:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        return expr

    def _is_match_number_one_false(self, expr: Expr) -> bool:
        """
        MATCH_NUMBER() starts from 1. If a must-appear variable's DEFINE
        condition cannot be true for MATCH_NUMBER() = 1, the first match can
        never be emitted, so no later match number can be reached either.
        """
        return self._truth_at_match_number_one(expr) is False

    def _truth_at_match_number_one(self, expr: Expr) -> Optional[bool]:
        if expr is None:
            return None

        expr = self._unwrap_semantics(expr)

        if isinstance(expr, BinaryExpr):
            op = expr.op
            if op == "AND":
                left = self._truth_at_match_number_one(expr.left)
                right = self._truth_at_match_number_one(expr.right)
                if left is False or right is False:
                    return False
                if left is True and right is True:
                    return True
                return None
            if op == "OR":
                left = self._truth_at_match_number_one(expr.left)
                right = self._truth_at_match_number_one(expr.right)
                if left is True or right is True:
                    return True
                if left is False and right is False:
                    return False
                return None

            return self._eval_match_number_literal_comparison(expr.left, op, expr.right)

        if isinstance(expr, BetweenExpr):
            target = self._unwrap_semantics(expr.expr)
            lower = self._unwrap_semantics(expr.lower)
            upper = self._unwrap_semantics(expr.upper)
            if (
                self._is_match_number_call(target)
                and self._is_numeric_literal(lower)
                and self._is_numeric_literal(upper)
            ):
                return lower.value <= 1 <= upper.value
            return None

        if isinstance(expr, InExpr):
            target = self._unwrap_semantics(expr.expr)
            if not self._is_match_number_call(target):
                return None
            literal_values = []
            for value in expr.values:
                value = self._unwrap_semantics(value)
                if not self._is_numeric_literal(value):
                    return None
                literal_values.append(value.value)
            contains_one = 1 in literal_values
            return not contains_one if expr.is_not else contains_one

        if isinstance(expr, NotExpr):
            inner = self._truth_at_match_number_one(expr.expr)
            return None if inner is None else not inner

        return None

    def _eval_match_number_literal_comparison(
        self,
        left: Expr,
        op: str,
        right: Expr,
    ) -> Optional[bool]:
        left = self._unwrap_semantics(left)
        right = self._unwrap_semantics(right)

        if self._is_match_number_call(left) and self._is_numeric_literal(right):
            return self._compare_numeric_literals(1, op, right.value)
        if self._is_numeric_literal(left) and self._is_match_number_call(right):
            return self._compare_numeric_literals(left.value, op, 1)
        return None

    def _is_match_number_call(self, expr: Expr) -> bool:
        expr = self._unwrap_semantics(expr)
        return (
            isinstance(expr, FunctionCall)
            and expr.name is not None
            and expr.name.upper() == "MATCH_NUMBER"
            and len(expr.args) == 0
        )

    def _is_numeric_literal(self, expr: Expr) -> bool:
        expr = self._unwrap_semantics(expr)
        return isinstance(expr, Literal) and isinstance(expr.value, (int, float))

    def _compare_numeric_literals(
        self, left: float, op: str, right: float
    ) -> Optional[bool]:
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == "=":
            return left == right
        if op in ("!=", "<>"):
            return left != right
        return None

    def _is_future_only_dependency_false(
        self, expr: Expr, current_var: str, pattern
    ) -> bool:
        """
        If current_var references another variable that cannot possibly appear before
        current_var in PATTERN, the predicate is always false under RUNNING semantics.
        """
        if pattern.get_left_min_row_num(current_var) is None:
            return False

        for ref_var in self._collect_cross_variable_refs(expr, current_var):
            if pattern.get_left_min_row_num(ref_var) is None:
                continue
            if not self._has_dependency_satisfying_path(
                pattern, {ref_var}, current_var
            ):
                return True
        return False

    def _has_dependency_satisfying_path(
        self, pattern, earlier_vars: Set[str], current_var: str
    ) -> bool:
        """Whether a path can define every current_var after an earlier variable.

        State is ``(seen_earlier, seen_current)``.  A path is discarded as soon
        as it encounters current_var before any earlier variable.  Tracking whole
        paths avoids combining minimum positions taken from different alternatives.
        """
        end_states = self._dependency_path_states(
            pattern, {(False, False)}, earlier_vars, current_var
        )
        return any(seen_current for _, seen_current in end_states)

    def _dependency_path_states(
        self,
        pattern,
        states: Set[Tuple[bool, bool]],
        earlier_vars: Set[str],
        current_var: str,
    ) -> Set[Tuple[bool, bool]]:
        if not states:
            return set()

        if isinstance(pattern, VariablePattern):
            result = set()
            for seen_earlier, seen_current in states:
                next_seen_earlier = seen_earlier or pattern.name in earlier_vars
                if pattern.name == current_var and not next_seen_earlier:
                    continue
                result.add(
                    (next_seen_earlier, seen_current or pattern.name == current_var)
                )
            return result

        if isinstance(pattern, ConcatenationPattern):
            result = states
            for child in pattern.patterns:
                result = self._dependency_path_states(
                    child, result, earlier_vars, current_var
                )
            return result

        if isinstance(pattern, AlternationPattern):
            result = set()
            for child in pattern.patterns:
                result.update(
                    self._dependency_path_states(
                        child, states, earlier_vars, current_var
                    )
                )
            return result

        if isinstance(pattern, QuantifiedPattern):
            repeated = states
            for _ in range(pattern.atLeast):
                repeated = self._dependency_path_states(
                    pattern.pattern, repeated, earlier_vars, current_var
                )
                if not repeated:
                    return set()

            result = set(repeated)
            remaining = (
                None if pattern.atMost is None else pattern.atMost - pattern.atLeast
            )
            iterations = 0
            while remaining is None or iterations < remaining:
                next_states = self._dependency_path_states(
                    pattern.pattern, repeated, earlier_vars, current_var
                )
                if not next_states:
                    break
                before = set(result)
                result.update(next_states)
                iterations += 1
                if next_states == repeated or (remaining is None and result == before):
                    break
                repeated = next_states
            return result

        if isinstance(pattern, PermutationPattern):
            # A permutation can choose an earlier-variable child before a
            # current-variable child.  Keep this deliberately conservative to
            # avoid claiming impossibility for complex nested permutations.
            has_earlier = any(
                self._pattern_contains_variable(pattern, name) for name in earlier_vars
            )
            has_current = self._pattern_contains_variable(pattern, current_var)
            result = set()
            for seen_earlier, seen_current in states:
                if has_current and not (seen_earlier or has_earlier):
                    continue
                result.add((seen_earlier or has_earlier, seen_current or has_current))
            return result

        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._dependency_path_states(
                pattern.pattern, states, earlier_vars, current_var
            )

        if isinstance(pattern, EmptyPattern):
            return states

        return set(states)

    def _pattern_contains_variable(self, pattern, var_name: str) -> bool:
        if isinstance(pattern, VariablePattern):
            return pattern.name == var_name
        if isinstance(pattern, QuantifiedPattern):
            return pattern.atMost != 0 and self._pattern_contains_variable(
                pattern.pattern, var_name
            )
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(
                self._pattern_contains_variable(child, var_name)
                for child in pattern.patterns
            )
        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._pattern_contains_variable(pattern.pattern, var_name)
        return False

    def _is_future_only_subset_dependency_false(
        self,
        expr: Expr,
        current_var: str,
        pattern,
        subsets: Optional[Dict[str, List[str]]],
    ) -> bool:
        """
        A SUBSET variable is visible under RUNNING semantics only through rows
        already mapped to any of its member variables. If all members of the
        referenced SUBSET can only appear after the current variable, a direct
        SUBSET column reference is null/unknown and the predicate cannot pass.
        """
        if not subsets:
            return False

        if pattern.get_left_min_row_num(current_var) is None:
            return False

        subset_refs = self._collect_subset_refs(expr, subsets)
        for subset_name in subset_refs:
            members = self._lookup_subset_members(subset_name, subsets)
            if not members:
                continue

            if current_var in members:
                continue
            present_members = {
                member
                for member in members
                if pattern.get_left_min_row_num(member) is not None
            }
            if present_members and not self._has_dependency_satisfying_path(
                pattern, present_members, current_var
            ):
                return True

        return False

    def _collect_subset_refs(
        self, expr: Expr, subsets: Dict[str, List[str]]
    ) -> Set[str]:
        subset_names = set(subsets)
        result: Set[str] = set()

        def visit(node: Expr):
            if node is None:
                return
            node = self._unwrap_semantics(node)
            if isinstance(node, BinaryExpr):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, BetweenExpr):
                visit(node.expr)
                visit(node.lower)
                visit(node.upper)
            elif isinstance(node, InExpr):
                visit(node.expr)
                for value in node.values:
                    visit(value)
            elif isinstance(node, NotExpr):
                visit(node.expr)
            elif isinstance(node, FunctionCall):
                # Function calls are opaque here.  A future-only reference does
                # not by itself make the surrounding predicate false: functions
                # such as COALESCE can turn the unavailable (NULL) value into a
                # value that satisfies the predicate.  Only direct references in
                # expressions whose SQL NULL propagation is evident are safe for
                # this optimization.
                return
            elif isinstance(node, ColumnRef):
                if node.qualifier and node.qualifier in subset_names:
                    result.add(node.qualifier)

        visit(expr)
        return result

    def _lookup_subset_members(
        self, subset_name: str, subsets: Dict[str, List[str]]
    ) -> Optional[List[str]]:
        for name, members in subsets.items():
            if name == subset_name:
                return members
        return None

    def _collect_cross_variable_refs(self, expr: Expr, current_var: str) -> Set[str]:
        result: Set[str] = set()

        def visit(node: Expr):
            if node is None:
                return
            if isinstance(node, SemanticsExpr):
                visit(node.expr)
            elif isinstance(node, BinaryExpr):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, BetweenExpr):
                visit(node.expr)
                visit(node.lower)
                visit(node.upper)
            elif isinstance(node, InExpr):
                visit(node.expr)
                for value in node.values:
                    visit(value)
            elif isinstance(node, NotExpr):
                visit(node.expr)
            elif isinstance(node, FunctionCall):
                # Do not assume arbitrary SQL functions propagate NULL.  In
                # particular, COALESCE(future_var.col, ...) can be true before
                # that pattern variable has been mapped.
                return
            elif isinstance(node, ColumnRef):
                if node.qualifier and node.qualifier != current_var:
                    result.add(node.qualifier)

        visit(expr)
        return result

    def _is_literal_false(self, expr: BinaryExpr) -> bool:
        """Check if expression is a literal false comparison like 1 != 1 or 0 = 1."""
        if not isinstance(expr, BinaryExpr):
            return False

        left = expr.left
        right = expr.right

        # Unwrap SemanticsExpr
        if isinstance(left, SemanticsExpr):
            left = left.expr
        if isinstance(right, SemanticsExpr):
            right = right.expr

        # Check if both sides are literals
        if not isinstance(left, Literal) or not isinstance(right, Literal):
            return False

        # Compare using a conservative subset of SQL implicit-conversion
        # semantics.  If conversion is ambiguous or fails, do not claim that
        # the predicate is always false.
        coerced = self._coerce_comparable_literal_values(left.value, right.value)
        if coerced is None:
            return False
        left_val, right_val = coerced
        op = expr.op

        # Compare values
        if left_val == right_val:
            # Same value: X = X is true, X != X is false, X < X is false, etc.
            return op in ("!=", "<>", "<", ">")
        else:
            # Different values: check if operator makes it false
            # e.g., 1 = 2 is false, 1 > 2 is false
            if op == "=":
                return True  # Different values cannot be equal

        return False

    def _coerce_comparable_literal_values(self, left_value, right_value):
        """Coerce a literal pair when SQL comparison semantics are predictable."""
        if left_value is None or right_value is None:
            return None

        left_is_number = self._is_non_boolean_number(left_value)
        right_is_number = self._is_non_boolean_number(right_value)

        if left_is_number and right_is_number:
            left_number = self._to_finite_decimal(left_value)
            right_number = self._to_finite_decimal(right_value)
            if left_number is None or right_number is None:
                return None
            return left_number, right_number

        if left_is_number and isinstance(right_value, str):
            left_number = self._to_finite_decimal(left_value)
            right_number = self._to_finite_decimal(right_value)
            if left_number is None or right_number is None:
                return None
            return left_number, right_number

        if isinstance(left_value, str) and right_is_number:
            left_number = self._to_finite_decimal(left_value)
            right_number = self._to_finite_decimal(right_value)
            if left_number is None or right_number is None:
                return None
            return left_number, right_number

        if type(left_value) is type(right_value) and isinstance(
            left_value, (str, bool)
        ):
            return left_value, right_value

        return None

    def _is_non_boolean_number(self, value) -> bool:
        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)

    def _to_finite_decimal(self, value) -> Optional[Decimal]:
        try:
            converted = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            return None
        return converted if converted.is_finite() else None

    def _implicit_literal_key(self, literal: Literal):
        """Return a conservative equality key for IN/partition intersections.

        Numeric-looking strings share a key with numbers.  This may miss an
        optimization for a character-typed column, but it cannot create an
        incorrect empty-query rewrite.
        """
        value = literal.value
        if value is None:
            return None
        if self._is_non_boolean_number(value):
            number = self._to_finite_decimal(value)
            return None if number is None else ("number", number)
        if isinstance(value, str):
            number = self._to_finite_decimal(value)
            if number is not None:
                return ("number", number)
            return ("string", value)
        if isinstance(value, bool):
            return ("boolean", value)
        return None

    def _implicit_literal_set(self, literals: List[Literal]) -> Optional[Set[tuple]]:
        result = set()
        for literal in literals:
            key = self._implicit_literal_key(literal)
            if key is None:
                return None
            result.add(key)
        return result

    def _is_self_comparison_false(self, expr: BinaryExpr) -> bool:
        """
        Check if expression is a self-comparison that's always false.
        E.g., A.OPEN > A.OPEN, OPEN > OPEN, PREV(X, k) < PREV(X, k)
        """
        if not isinstance(expr, BinaryExpr):
            return False

        op = expr.op
        left = expr.left
        right = expr.right

        # Unwrap SemanticsExpr
        if isinstance(left, SemanticsExpr):
            left = left.expr
        if isinstance(right, SemanticsExpr):
            right = right.expr

        # Check if left and right are the same
        if self._expr_equals(left, right):
            # Same expression: X > X, X < X, X >= X are false
            # Only X = X and X >= X (when X is same) and X <= X are true
            return op in (">", "<", "!=", "<>")

        return False

    def _expr_equals(self, expr1: Expr, expr2: Expr) -> bool:
        """Check if two expressions are syntactically equal."""
        if expr1 is None and expr2 is None:
            return True
        if expr1 is None or expr2 is None:
            return False

        # Unwrap SemanticsExpr
        if isinstance(expr1, SemanticsExpr):
            expr1 = expr1.expr
        if isinstance(expr2, SemanticsExpr):
            expr2 = expr2.expr

        # Compare literals
        if isinstance(expr1, Literal) and isinstance(expr2, Literal):
            return expr1.value == expr2.value

        # Compare column references
        if isinstance(expr1, ColumnRef) and isinstance(expr2, ColumnRef):
            return expr1.name == expr2.name and expr1.qualifier == expr2.qualifier

        # Compare function calls
        if isinstance(expr1, FunctionCall) and isinstance(expr2, FunctionCall):
            if expr1.name != expr2.name or len(expr1.args) != len(expr2.args):
                return False
            return all(
                self._expr_equals(a1, a2) for a1, a2 in zip(expr1.args, expr2.args)
            )

        # Compare binary expressions
        if isinstance(expr1, BinaryExpr) and isinstance(expr2, BinaryExpr):
            return (
                expr1.op == expr2.op
                and self._expr_equals(expr1.left, expr2.left)
                and self._expr_equals(expr1.right, expr2.right)
            )

        # Compare BETWEEN expressions
        if isinstance(expr1, BetweenExpr) and isinstance(expr2, BetweenExpr):
            return (
                self._expr_equals(expr1.expr, expr2.expr)
                and self._expr_equals(expr1.lower, expr2.lower)
                and self._expr_equals(expr1.upper, expr2.upper)
            )

        return False

    def _is_arithmetic_always_false(self, expr: BinaryExpr) -> bool:
        """
        Check if arithmetic expressions are always false.
        Examples:
        - X + 1 < X (left side always > right side)
        - X * 2 > X * 2 + 1 (left side always <= right side)
        - X + 1 > X + 2 (never true for any X)
        """
        if not isinstance(expr, BinaryExpr):
            return False

        op = expr.op
        left = expr.left
        right = expr.right

        # Unwrap SemanticsExpr
        if isinstance(left, SemanticsExpr):
            left = left.expr
        if isinstance(right, SemanticsExpr):
            right = right.expr

        # Try to extract base expression and offset from both sides
        left_base, left_offset = self._extract_expr_offset(left)
        right_base, right_offset = self._extract_expr_offset(right)

        # If both sides have the same base expression, we can compare offsets
        if (
            left_base is not None
            and right_base is not None
            and self._expr_equals(left_base, right_base)
        ):
            # Both sides are the same base with different constants
            # Compare: (base + left_offset) op (base + right_offset)
            # Simplifies to: left_offset op right_offset

            if op == ">":
                # X + 1 > X + 2 means 1 > 2, which is false
                return left_offset < right_offset
            elif op == "<":
                # X + 2 < X + 1 means 2 < 1, which is false
                return left_offset > right_offset
            elif op == "!=":
                # X + 1 != X + 1 is false (always equal)
                return left_offset == right_offset
            elif op == "<>":
                # Same as !=
                return left_offset == right_offset

        # Special case: Try to detect X * C > X * C + D pattern
        # This works for expressions like: (A.OPEN * 2) > (A.OPEN * 2 + 1)
        if op in (">", "<", ">=", "<="):
            left_mult_expr, left_factor = self._extract_multiplication_base(left)

            if left_mult_expr is not None and left_factor is not None:
                # Now try to extract the same multiplication from right
                right_mult_expr, right_factor, right_offset = (
                    self._extract_mult_with_offset(right)
                )

                if (
                    right_mult_expr is not None
                    and right_factor is not None
                    and isinstance(left_factor, (int, float))
                    and isinstance(right_factor, (int, float))
                ):

                    # Check if both have the same base and factor
                    if left_factor == right_factor and self._expr_equals(
                        left_mult_expr, right_mult_expr
                    ):

                        # Now we have: (base * factor) op (base * factor + offset)
                        # This simplifies to: 0 op offset
                        if right_offset is not None:
                            if op == ">":
                                # X * C > X * C + offset means 0 > offset, false if offset >= 0
                                return right_offset >= 0
                            elif op == "<":
                                # X * C < X * C + offset means 0 < offset, false if offset <= 0
                                return right_offset <= 0
                            elif op == ">=":
                                # X * C >= X * C + offset means 0 >= offset, false if offset > 0
                                return right_offset > 0
                            elif op == "<=":
                                # X * C <= X * C + offset means 0 <= offset, false if offset < 0
                                return right_offset < 0

        return False

    def _extract_multiplication_base(self, expr: Expr) -> tuple:
        """
        Extract base and factor from multiplication expressions.
        Examples:
        - A.OPEN * 2 -> (A.OPEN, 2)
        - 2 * A.OPEN -> (A.OPEN, 2)
        Returns: (base_expr, factor_value)
        """
        if expr is None:
            return None, None

        if isinstance(expr, SemanticsExpr):
            return self._extract_multiplication_base(expr.expr)

        if isinstance(expr, BinaryExpr) and expr.op == "*":
            left = expr.left
            right = expr.right

            if isinstance(left, SemanticsExpr):
                left = left.expr
            if isinstance(right, SemanticsExpr):
                right = right.expr

            # Case: base * factor
            if isinstance(right, Literal) and isinstance(right.value, (int, float)):
                return left, right.value

            # Case: factor * base
            if isinstance(left, Literal) and isinstance(left.value, (int, float)):
                return right, left.value

        return None, None

    def _extract_mult_with_offset(self, expr: Expr) -> tuple:
        """
        Extract multiplication base and factor, with optional additive offset.
        Examples:
        - A.OPEN * 2 -> (A.OPEN, 2, 0)
        - A.OPEN * 2 + 1 -> (A.OPEN, 2, 1)
        - A.OPEN * 2 - 3 -> (A.OPEN, 2, -3)
        Returns: (mult_base, factor, offset)
        """
        if expr is None:
            return None, None, None

        if isinstance(expr, SemanticsExpr):
            return self._extract_mult_with_offset(expr.expr)

        # Direct multiplication
        mult_base, factor = self._extract_multiplication_base(expr)
        if mult_base is not None and factor is not None:
            return mult_base, factor, 0

        # Addition/subtraction on top of multiplication
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
            left = expr.left
            right = expr.right

            if isinstance(left, SemanticsExpr):
                left = left.expr
            if isinstance(right, SemanticsExpr):
                right = right.expr

            # Try extracting multiplication from left side
            left_mult_base, left_factor = self._extract_multiplication_base(left)
            if left_mult_base is not None and left_factor is not None:
                # Right should be the offset
                if isinstance(right, Literal) and isinstance(right.value, (int, float)):
                    offset = right.value if expr.op == "+" else -right.value
                    return left_mult_base, left_factor, offset

            # Try extracting multiplication from right side
            right_mult_base, right_factor = self._extract_multiplication_base(right)
            if (
                right_mult_base is not None
                and right_factor is not None
                and expr.op == "+"
            ):
                # Left should be the offset
                if isinstance(left, Literal) and isinstance(left.value, (int, float)):
                    return right_mult_base, right_factor, left.value

        return None, None, None

    def _extract_expr_offset(self, expr: Expr) -> tuple:
        """
        Extract base expression and offset from arithmetic expression.

        Returns: (base_expr, offset_value)
        Examples:
        - X + 5 -> (X, 5)
        - X - 3 -> (X, -3)
        - 2 * X -> (X, factor=2)  [approximate as (X, 0) for now]
        - X -> (X, 0)
        - 10 -> (None, 10)

        Returns (None, None) if cannot extract.
        """
        if expr is None:
            return None, None

        if isinstance(expr, SemanticsExpr):
            return self._extract_expr_offset(expr.expr)

        # Literal: no base, just offset
        if isinstance(expr, Literal):
            if isinstance(expr.value, (int, float)):
                return None, expr.value
            return None, None

        # Column or function: base with no offset
        if isinstance(expr, (ColumnRef, FunctionCall)):
            return expr, 0

        # Binary expression: try to decompose
        if isinstance(expr, BinaryExpr):
            # Handle: base + constant or constant + base
            if expr.op in ("+", "-"):
                left = expr.left
                right = expr.right

                if isinstance(left, SemanticsExpr):
                    left = left.expr
                if isinstance(right, SemanticsExpr):
                    right = right.expr

                # Case: X + 5 or X - 5
                if isinstance(right, Literal) and isinstance(right.value, (int, float)):
                    offset = right.value if expr.op == "+" else -right.value
                    left_base, left_offset = self._extract_expr_offset(left)
                    if left_base is not None:
                        return left_base, left_offset + offset
                    elif left_offset is not None:
                        return None, left_offset + offset

                # Case: 5 + X
                if (
                    isinstance(left, Literal)
                    and isinstance(left.value, (int, float))
                    and expr.op == "+"
                ):
                    offset = left.value
                    right_base, right_offset = self._extract_expr_offset(right)
                    if right_base is not None:
                        return right_base, right_offset + offset
                    elif right_offset is not None:
                        return None, right_offset + offset

        return None, None

    def _has_partition_column_conflict(
        self, ast: RPRAbstractSyntaxTree, must_appear_vars: Set[str]
    ) -> bool:
        """
        Check if partition columns have conflicting constraints across must-appear variables.

        Cases:
        1. Equality conflicts: A.COL1 = 'ABC', B.COL1 = 'DEF'
        2. IN clause conflicts: A.COL1 IN ('A', 'B'), B.COL1 IN ('C', 'D')
        3. LIKE pattern conflicts: A.COL1 LIKE 'A%', B.COL1 LIKE 'B%'
        """
        if ast.partition_by is None or len(ast.partition_by) == 0:
            return False

        # Build a map of partition column constraints per variable
        # var_name -> {partition_col -> constraint}
        var_constraints = {}
        var_in_constraints = {}
        var_like_constraints = {}

        for var_name in must_appear_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue

            constraints = self._extract_partition_constraint_groups(
                condition.expression, var_name
            )
            var_constraints[var_name] = constraints["eq"]
            var_in_constraints[var_name] = constraints["in"]
            var_like_constraints[var_name] = constraints["like"]

        # Check for equality conflicts
        for partition_col in ast.partition_by:
            col_clean = partition_col.strip()

            # Collect all constraints on this partition column
            constraints_by_var = {}
            for var_name, constraints in var_constraints.items():
                if col_clean in constraints:
                    constraints_by_var[var_name] = constraints[col_clean]

            # If multiple variables constrain the same column to different values, conflict!
            if len(constraints_by_var) >= 2:
                values = list(constraints_by_var.values())
                # Check if all values are literals and different
                if all(isinstance(v, Literal) for v in values):
                    literal_keys = [self._implicit_literal_key(v) for v in values]
                    if None not in literal_keys and len(set(literal_keys)) > 1:
                        return True  # Conflict found!

            # Check for IN conflicts on same partition column
            in_by_var = {}
            for var_name, in_constraints in var_in_constraints.items():
                if col_clean in in_constraints:
                    in_by_var[var_name] = in_constraints[col_clean]

            if len(in_by_var) >= 2:
                # Check if IN value sets are disjoint
                in_sets = list(in_by_var.values())
                if all(all(isinstance(v, Literal) for v in s) for s in in_sets):
                    value_sets = [self._implicit_literal_set(s) for s in in_sets]
                    if any(value_set is None for value_set in value_sets):
                        continue
                    intersection = value_sets[0]
                    for vset in value_sets[1:]:
                        intersection &= vset
                    if len(intersection) == 0:
                        return True  # Disjoint IN sets - conflict!

            # Check for LIKE conflicts on same partition column
            like_by_var = {}
            for var_name, like_constraints in var_like_constraints.items():
                if col_clean in like_constraints:
                    like_by_var[var_name] = like_constraints[col_clean]

            if len(like_by_var) >= 2:
                # Check if LIKE patterns are incompatible
                patterns = list(like_by_var.values())
                if self._are_like_patterns_incompatible(patterns):
                    return True  # Incompatible LIKE patterns - conflict!

        return False

    def _extract_partition_constraint_groups(self, expr: Expr, var_name: str) -> dict:
        """
        Extract equality, IN, and LIKE constraints from a DEFINE expression in one pass.

        Returns:
            {
                'eq': {partition_col -> Literal value},
                'in': {partition_col -> [Literal values]},
                'like': {partition_col -> pattern_string},
            }
        """
        constraints = {"eq": {}, "in": {}, "like": {}}

        def add_column_constraint(target: Expr, value: Expr, kind: str):
            target = self._unwrap_semantics(target)
            value = self._unwrap_semantics(value)
            if not isinstance(target, ColumnRef):
                return
            if target.qualifier is not None and target.qualifier != var_name:
                return
            if not target.name:
                return
            if kind == "eq" and isinstance(value, Literal):
                constraints["eq"][target.name] = value
            elif (
                kind == "like"
                and isinstance(value, Literal)
                and isinstance(value.value, str)
            ):
                constraints["like"][target.name] = value.value

        def visit(node: Expr):
            if node is None:
                return
            node = self._unwrap_semantics(node)
            if isinstance(node, BinaryExpr):
                if node.op == "AND":
                    visit(node.left)
                    visit(node.right)
                elif node.op == "=":
                    add_column_constraint(node.left, node.right, "eq")
                    add_column_constraint(node.right, node.left, "eq")
                elif node.op == "LIKE":
                    add_column_constraint(node.left, node.right, "like")
                return
            if isinstance(node, InExpr):
                col_expr = self._unwrap_semantics(node.expr)
                if (
                    not node.is_not
                    and isinstance(col_expr, ColumnRef)
                    and col_expr.name
                    and (col_expr.qualifier is None or col_expr.qualifier == var_name)
                ):
                    constraints["in"][col_expr.name] = node.values

        visit(expr)
        return constraints

    def _is_navigation_partition_false(
        self, expr: Expr, var_name: str, partition_cols: List[str]
    ) -> bool:
        """
        Check if a navigation function creates an always-false condition with partition columns.

        Cases:
        1. PREV(var.PARTITION_COL) != var.PARTITION_COL is always FALSE
           (Different partitions have different partition column values)

        2. NEXT(var.PARTITION_COL) != var.PARTITION_COL is always FALSE
           (Different partitions have different partition column values)

        Example: PARTITION BY PT
                 DEFINE A AS PREV(A.PT) != A.PT
                 This is always false because PREV navigates to a different partition (different PT).
        """
        if expr is None or partition_cols is None or len(partition_cols) == 0:
            return False

        partition_cols_set = {col.strip() for col in partition_cols}

        if isinstance(expr, SemanticsExpr):
            return self._is_navigation_partition_false(
                expr.expr, var_name, partition_cols
            )

        if not isinstance(expr, BinaryExpr):
            return False

        # Check for patterns like: PREV(A.COL) != A.COL or NEXT(A.COL) != A.COL
        # where COL is a partition column
        if expr.op == "!=":
            left = expr.left
            right = expr.right

            if isinstance(left, SemanticsExpr):
                left = left.expr
            if isinstance(right, SemanticsExpr):
                right = right.expr

            # Check left is navigation function, right is column ref
            if isinstance(left, FunctionCall) and isinstance(right, ColumnRef):
                nav_func, nav_col = left, right
            # Check right is navigation function, left is column ref
            elif isinstance(right, FunctionCall) and isinstance(left, ColumnRef):
                nav_func, nav_col = right, left
            else:
                return False

            # Check if navigation function is PREV or NEXT
            nav_name = nav_func.name if nav_func.name else ""
            if nav_name not in ["PREV", "NEXT"]:
                return False

            # Check if the accessed column is a partition column
            col_name = nav_col.name if nav_col.name else None
            if col_name not in partition_cols_set:
                return False

            # Check if navigation is on the same variable
            if nav_col.qualifier and nav_col.qualifier != var_name:
                return False

            # Check if the function argument is the same column
            if nav_func.args and len(nav_func.args) > 0:
                func_arg = nav_func.args[0]
                if isinstance(func_arg, SemanticsExpr):
                    func_arg = func_arg.expr

                if isinstance(func_arg, ColumnRef):
                    arg_col_name = func_arg.name if func_arg.name else None
                    arg_qualifier = func_arg.qualifier if func_arg.qualifier else None

                    # Check if argument matches the right side of comparison
                    if arg_col_name == col_name and (
                        arg_qualifier is None or arg_qualifier == var_name
                    ):
                        # PREV/NEXT(A.PT) != A.PT where PT is partition column is always false
                        return True

        # Check AND chain
        if expr.op == "AND":
            return self._is_navigation_partition_false(
                expr.left, var_name, partition_cols
            ) or self._is_navigation_partition_false(
                expr.right, var_name, partition_cols
            )

        return False

    def _is_in_clause_false_in_chain(
        self, expr: BinaryExpr, var_name: str = None
    ) -> bool:
        """
        Check if IN clauses in AND chain create an always-false condition.
        Detects: X IN (set1) AND X IN (set2) where set1 ∩ set2 = ∅
        """
        if expr is None or not isinstance(expr, BinaryExpr):
            return False

        # Collect all IN constraints from AND chain
        in_constraints = {}
        self._collect_in_expressions(expr, in_constraints, var_name)

        # Check for conflicts: same column with disjoint value sets
        for col_name, in_expr_list in in_constraints.items():
            if len(in_expr_list) >= 2:
                # Check if all constraints are literals
                all_literals = all(
                    all(isinstance(v, Literal) for v in in_expr.values)
                    for in_expr in in_expr_list
                )
                if all_literals:
                    # Convert to sets of values
                    value_sets = [
                        self._implicit_literal_set(in_expr.values)
                        for in_expr in in_expr_list
                    ]
                    if any(value_set is None for value_set in value_sets):
                        continue
                    # Check if sets are disjoint
                    intersection = value_sets[0]
                    for vset in value_sets[1:]:
                        intersection &= vset
                    if len(intersection) == 0:
                        return True

        return False

    def _collect_in_expressions(
        self, expr: Expr, constraints: dict, var_name: str = None
    ) -> None:
        """
        Collect InExpr from AND chain.
        Builds: {column_name -> [InExpr1, InExpr2, ...]}
        """
        if expr is None:
            return

        if isinstance(expr, SemanticsExpr):
            return self._collect_in_expressions(expr.expr, constraints, var_name)

        # Need to check for InExpr type (from expression module)
        if hasattr(expr, "__class__") and expr.__class__.__name__ == "InExpr":
            # NOT IN is an exclusion constraint, so disjoint value lists do
            # not make multiple predicates contradictory.  It must not be
            # mixed into the positive-IN intersection calculation below.
            if getattr(expr, "is_not", False):
                return

            # This is an InExpr
            col_expr = expr.expr if hasattr(expr, "expr") else None

            if isinstance(col_expr, SemanticsExpr):
                col_expr = col_expr.expr

            if isinstance(col_expr, ColumnRef):
                col_name = col_expr.name if col_expr.name else None
                if col_name and (
                    var_name is None
                    or col_expr.qualifier is None
                    or col_expr.qualifier == var_name
                ):
                    if col_name not in constraints:
                        constraints[col_name] = []
                    constraints[col_name].append(expr)
            return

        # Recursively collect from AND chain
        if isinstance(expr, BinaryExpr):
            if expr.op == "AND":
                self._collect_in_expressions(expr.left, constraints, var_name)
                self._collect_in_expressions(expr.right, constraints, var_name)

    def _is_like_clause_false_in_chain(
        self, expr: BinaryExpr, var_name: str = None
    ) -> bool:
        """
        Check if LIKE clauses in AND chain create an always-false condition.
        Detects: X LIKE 'pattern1' AND X LIKE 'pattern2' where patterns don't overlap
        """
        if expr is None or not isinstance(expr, BinaryExpr):
            return False

        # Collect all LIKE constraints from AND chain
        like_constraints = {}
        self._collect_like_expressions(expr, like_constraints, var_name)

        # Check for conflicts: same column with incompatible patterns
        for col_name, patterns in like_constraints.items():
            if len(patterns) >= 2:
                # Check if patterns are incompatible (start with different prefixes)
                if self._are_like_patterns_incompatible(patterns):
                    return True

        return False

    def _collect_like_expressions(
        self, expr: Expr, constraints: dict, var_name: str = None
    ) -> None:
        """
        Collect LIKE patterns from AND chain.
        Builds: {column_name -> ['pattern1', 'pattern2', ...]}
        """
        if expr is None:
            return

        if isinstance(expr, SemanticsExpr):
            return self._collect_like_expressions(expr.expr, constraints, var_name)

        if isinstance(expr, BinaryExpr):
            if expr.op == "LIKE":
                left = expr.left
                right = expr.right

                if isinstance(left, SemanticsExpr):
                    left = left.expr
                if isinstance(right, SemanticsExpr):
                    right = right.expr

                if isinstance(left, ColumnRef):
                    col_name = left.name if left.name else None
                    if col_name and (
                        var_name is None
                        or left.qualifier is None
                        or left.qualifier == var_name
                    ):
                        if isinstance(right, Literal) and isinstance(right.value, str):
                            pattern = right.value
                            if col_name not in constraints:
                                constraints[col_name] = []
                            constraints[col_name].append(pattern)

            elif expr.op == "AND":
                self._collect_like_expressions(expr.left, constraints, var_name)
                self._collect_like_expressions(expr.right, constraints, var_name)

    def _is_in_clause_false(self, expr: BinaryExpr, var_name: str = None) -> bool:
        """
        Check if an IN clause creates an always-false condition.
        Cases:
        1. Empty IN set: X IN () is always false
        """
        if expr is None or not isinstance(expr, BinaryExpr):
            return False

        return False

    def _is_like_clause_false(self, expr: BinaryExpr, var_name: str = None) -> bool:
        """
        Check if a single LIKE clause creates an always-false condition (not typically possible).
        This is a placeholder for pattern-based detection if needed.
        """
        if expr is None or not isinstance(expr, BinaryExpr):
            return False

        return False

    def _collect_in_constraints(
        self, expr: Expr, constraints: dict, var_name: str = None
    ) -> None:
        """
        Placeholder method - no longer used.
        """
        pass

    def _collect_like_constraints(
        self, expr: Expr, constraints: dict, var_name: str = None
    ) -> None:
        """
        Placeholder method - no longer used.
        """
        pass

    def _are_like_patterns_incompatible(self, patterns: list) -> bool:
        """
        Check if LIKE patterns are incompatible (cannot all match the same string).

        Examples:
        - 'A%' and 'B%' are incompatible (different starting letters)
        - 'A%' and 'A%' are compatible (same pattern)
        - 'A%' and 'AB%' are compatible (nested patterns)
        - 'ERR_%' and 'WARN_%' are incompatible (different starting words)

        Simple heuristic: patterns with different starting literals are incompatible
        """
        if len(patterns) < 2:
            return False

        # Extract starting literal from each pattern
        starts = []
        for pattern in patterns:
            if isinstance(pattern, str) and len(pattern) > 0:
                # Extract characters before first wildcard
                literal_start = ""
                for char in pattern:
                    if char in "%_":
                        break
                    literal_start += char
                starts.append(literal_start.upper())
            else:
                starts.append(None)

        # Two literal prefixes are incompatible only when they diverge before
        # either prefix reaches a wildcard.  Different prefix lengths alone do
        # not imply a conflict: for example, both A% and AB% match "ABC".
        #
        # Keep this check conservative.  More complicated suffix constraints
        # may also be incompatible, but failing to recognize those only misses
        # an optimization; treating compatible patterns as incompatible would
        # incorrectly turn a non-empty query into an empty one.
        valid_starts = [s for s in starts if s and len(s) > 0]
        for index, left in enumerate(valid_starts):
            for right in valid_starts[index + 1 :]:
                if not (left.startswith(right) or right.startswith(left)):
                    return True

        return False


var_always_false_test_cases = [
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS 1 != 1, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS 1 != 1, B AS VOL > 10
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS OPEN > OPEN, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS OPEN > OPEN, B AS VOL > 10
)""",
    ),
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.OPEN > A.OPEN AND PREV(B.VOL) > 10, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.OPEN > A.OPEN AND PREV(B.VOL) > 10, B AS VOL > 10
)""",
    ),
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B*)
    DEFINE A AS A.OPEN > 100, B AS B.OPEN > B.OPEN
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B*)
    DEFINE A AS A.OPEN > 100, B AS B.OPEN > B.OPEN
)""",
    ),
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+)
    DEFINE A AS A.OPEN > 100, B AS B.OPEN > B.OPEN
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+)
    DEFINE A AS A.OPEN > 100, B AS B.OPEN > B.OPEN
)""",
    ),
    (
        6,
        """
SELECT * FROM (SELECT * FROM TA WHERE OPEN < 10) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.OPEN + 1 < A.OPEN, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS (A.OPEN + 1) < A.OPEN, B AS VOL > 10
)""",
    ),
    (
        7,
        """
SELECT * FROM (SELECT * FROM TA WHERE OPEN < 10) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.PT = 'A', B AS PT = 'B'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.PT = 'A', B AS PT = 'B'
)""",
    ),
    (
        8,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.OPEN + 1 < A.OPEN, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS (A.OPEN + 1) < A.OPEN, B AS VOL > 10
)""",
    ),
    (
        9,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS A.OPEN * 2 > A.OPEN * 2 + 1, B AS VOL > 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE 1 != 1) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.VOL AS BVOL
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS OPEN * 2) > ((OPEN * 2) + 1, B AS VOL > 10
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        MustAppearVarFalsePushdownRule,
        var_always_false_test_cases,
        "var always false pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
