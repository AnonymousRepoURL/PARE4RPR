if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from decimal import Decimal

from .utils import *


class IncrementalConditionPushdownRule(RewriteRule):
    """
    Push first-ORDER-BY-column DEFINE constraints to earlier variables.

    For example, TS - A.TS < 10 is monotone under ascending TS order.  For
    descending order we normalize it to the monotone form A.TS - TS < 10.

    The rule only pushes constraints that remain sound under running semantics.
    """

    @property
    def name(self) -> str:
        return "incremental_condition_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        return plan is not None and len(plan) > 0

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        plans = self._build_plan(ast)
        if not plans:
            return False

        changed = False
        parser = None

        for plan in plans:
            for var_name, pushed_expr in plan["updates"].items():
                current_condition = ast.define.definitions.get(var_name)
                current_expr = (
                    current_condition.expression
                    if current_condition is not None
                    else None
                )
                merged_expr = self._merge_condition_expr(
                    current_expr, pushed_expr, plan, var_name
                )
                if (
                    current_expr is not None
                    and merged_expr.to_string() == current_expr.to_string()
                ):
                    continue
                if parser is None:
                    from clause_parser import ExprParser

                    parser = ExprParser()
                ast.define.definitions[var_name] = parser._analyze_dependency(
                    var_name, merged_expr
                )
                changed = True

        return changed

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        if ast.match_order_by is None or len(ast.match_order_by) == 0:
            return None

        order_col = ast.match_order_by[0][0].strip()
        if "." in order_col:
            order_col = order_col.split(".")[-1]
        sort_type = ast.match_order_by[0][1]
        must_appear_vars = ast.pattern.get_must_appear_var()
        primary_vars = ast.pattern.get_variables()

        plans = []
        for var_name, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            if var_name not in must_appear_vars:
                continue
            conjuncts = self._flatten_and(condition.expression)
            for atom in conjuncts:
                order_constraint = self._parse_order_incremental_constraint(
                    atom, var_name, order_col, sort_type
                )
                if order_constraint is not None:
                    source_var = order_constraint["source_var"]
                    # A SUBSET name is a union pattern variable, not a primary
                    # variable with one stable row.  More importantly, an
                    # ordinary source reference has RUNNING-last semantics. If
                    # the source can match more than once, an earlier recipient
                    # and the target may observe different source rows.
                    if source_var not in primary_vars:
                        continue
                    if source_var not in must_appear_vars:
                        continue
                    if self._max_var_occurrences(ast.pattern, source_var) != 1:
                        continue
                    updates = self._build_order_updates(
                        ast,
                        source_var,
                        var_name,
                        order_constraint["normalized_expr"],
                        order_col,
                    )
                    if updates:
                        plans.append(
                            {
                                "updates": updates,
                                "target_var": var_name,
                                "order_col": order_col,
                                "sort_type": sort_type,
                            }
                        )
        return plans

    def _build_order_updates(
        self,
        ast: RPRAbstractSyntaxTree,
        source_var: str,
        target_var: str,
        pushed_expr: Expr,
        order_col: Optional[str] = None,
    ):
        if source_var == target_var:
            return {}
        if order_col is None:
            if ast.match_order_by is None or len(ast.match_order_by) == 0:
                return {}
            order_col = ast.match_order_by[0][0].strip().split(".")[-1]
        edge_updates = self._build_edge_spanning_updates(
            ast,
            source_var,
            target_var,
            pushed_expr,
            order_col,
        )
        if edge_updates is not None:
            return edge_updates
        recipient_vars = self._collect_between(ast.pattern, source_var, target_var)
        if not recipient_vars:
            return {}
        sequences = self._enumerate_order_sequences(ast.pattern)
        if sequences is None:
            return {}
        return {
            var_name: pushed_expr
            for var_name in sorted(recipient_vars)
            if var_name == target_var
            or self._all_occurrences_between(
                sequences, source_var, var_name, target_var
            )
        }

    def _build_edge_spanning_updates(
        self,
        ast: RPRAbstractSyntaxTree,
        source_var: str,
        target_var: str,
        pushed_expr: Expr,
        order_col: str,
    ) -> Optional[Dict[str, Expr]]:
        """Use a monotone endpoint bound without expanding the pattern.

        When a singleton source is necessarily the first matched row and a
        singleton target is necessarily the last, every other pattern row is
        ordered between them.  A supported first-ORDER-BY-column upper bound
        that holds at the target therefore holds at every intermediate row.

        Requiring both endpoint variables to occur exactly once avoids
        RUNNING-last remapping.  Requiring a known non-null order column avoids
        treating a NULL-sorted intermediate row as numerically monotone.
        """
        pattern = ast.pattern
        must_appear_vars = pattern.get_must_appear_var()
        if source_var not in must_appear_vars or target_var not in must_appear_vars:
            return None
        if self._max_var_occurrences(pattern, source_var) != 1:
            return None
        if self._max_var_occurrences(pattern, target_var) != 1:
            return None
        if pattern.get_leftmost_variables() != {source_var}:
            return None
        if pattern.get_rightmost_variables() != {target_var}:
            return None
        if pattern.get_left_max_row_num(source_var) != 0:
            return None
        if pattern.get_right_max_row_num(target_var) != 0:
            return None
        if ast.is_column_nullable(order_col) is not False:
            return None

        return {
            var_name: pushed_expr
            for var_name in sorted(pattern.get_variables())
            if var_name != source_var
        }

    def _flatten_and(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]

    def _parse_order_incremental_constraint(
        self, expr: Expr, current_var: str, order_col: str, sort_type: SortType
    ):
        if not isinstance(expr, BinaryExpr) or expr.op not in ("<", "<=", ">", ">="):
            return None

        lhs = expr.left
        rhs = expr.right
        op = expr.op
        if isinstance(lhs, Literal) and isinstance(rhs, BinaryExpr):
            lhs, rhs = rhs, lhs
            op = self._reverse_comparison_op(op)

        if (
            not isinstance(lhs, BinaryExpr)
            or lhs.op != "-"
            or not isinstance(rhs, Literal)
            or not isinstance(rhs.value, (int, float, Decimal))
        ):
            return None

        left_col = self._extract_order_ref(lhs.left, current_var, order_col)
        right_col = self._extract_order_ref(lhs.right, current_var, order_col)
        if left_col is None or right_col is None:
            return None

        if left_col["role"] == right_col["role"]:
            return None

        # Keep integer literals exact.  Converting them to float makes
        # adjacent SQL BIGINT values above 2**53 indistinguishable.
        literal_value = rhs.value
        if sort_type == SortType.ASC:
            if left_col["role"] == "current" and right_col["role"] == "source":
                normalized_op = op
                normalized_value = literal_value
                source_var = right_col["source_var"]
            elif left_col["role"] == "source" and right_col["role"] == "current":
                normalized_op = self._reverse_comparison_op(op)
                normalized_value = -literal_value
                source_var = left_col["source_var"]
            else:
                return None
            if normalized_op not in ("<", "<="):
                return None
            normalized_expr = BinaryExpr(
                BinaryExpr(
                    ColumnRef(None, order_col), "-", ColumnRef(source_var, order_col)
                ),
                normalized_op,
                self._numeric_literal(normalized_value),
            )
            if source_var is None:
                return None
            return {"source_var": source_var, "normalized_expr": normalized_expr}

        if left_col["role"] == "source" and right_col["role"] == "current":
            normalized_op = op
            normalized_value = literal_value
        elif left_col["role"] == "current" and right_col["role"] == "source":
            normalized_op = self._reverse_comparison_op(op)
            normalized_value = -literal_value
        else:
            return None
        if normalized_op not in ("<", "<="):
            return None
        source_var = (
            left_col["source_var"]
            if left_col["role"] == "source"
            else right_col["source_var"]
        )
        if source_var is None:
            return None
        normalized_expr = BinaryExpr(
            BinaryExpr(
                ColumnRef(source_var, order_col), "-", ColumnRef(None, order_col)
            ),
            normalized_op,
            self._numeric_literal(normalized_value),
        )
        return {"source_var": source_var, "normalized_expr": normalized_expr}

    def _extract_order_ref(self, expr: Expr, current_var: str, order_col: str):
        if not isinstance(expr, ColumnRef) or expr.name != order_col:
            return None
        if expr.qualifier is None or expr.qualifier == current_var:
            return {"role": "current", "source_var": None}
        return {"role": "source", "source_var": expr.qualifier}

    def _reverse_comparison_op(self, op: str) -> Optional[str]:
        return {
            ">": "<",
            ">=": "<=",
            "<": ">",
            "<=": ">=",
            "=": "=",
            "!=": "!=",
            "<>": "<>",
        }.get(op)

    def _numeric_literal(self, value: float) -> Literal:
        if isinstance(value, int):
            return Literal(value)
        if isinstance(value, Decimal):
            return (
                Literal(int(value))
                if value == value.to_integral_value()
                else Literal(value)
            )
        if float(value).is_integer():
            return Literal(int(value))
        return Literal(value)

    def _collect_all_vars(self, node) -> Set[str]:
        if isinstance(node, VariablePattern):
            return {node.name}
        if isinstance(node, QuantifiedPattern):
            return self._collect_all_vars(node.pattern)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._collect_all_vars(node.pattern)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            result = set()
            for child in node.patterns:
                result |= self._collect_all_vars(child)
            return result
        return set()

    def _contains_var(self, node, var_name: str) -> bool:
        return var_name in self._collect_all_vars(node)

    def _max_var_occurrences(self, node, var_name: str) -> int:
        """Return the maximum occurrence count, capped at two.

        The rule only needs to distinguish a singleton source from a source
        that can be remapped.  Capping the result avoids special handling for
        unbounded quantifiers while preserving that distinction.
        """
        if isinstance(node, VariablePattern):
            return 1 if node.name == var_name else 0
        if isinstance(node, QuantifiedPattern):
            inner_count = self._max_var_occurrences(node.pattern, var_name)
            if inner_count == 0 or node.atMost == 0:
                return 0
            if inner_count >= 2 or node.atMost is None or node.atMost >= 2:
                return 2
            return inner_count
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._max_var_occurrences(node.pattern, var_name)
        if isinstance(node, AlternationPattern):
            return min(
                2,
                max(
                    (
                        self._max_var_occurrences(child, var_name)
                        for child in node.patterns
                    ),
                    default=0,
                ),
            )
        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            count = 0
            for child in node.patterns:
                count += self._max_var_occurrences(child, var_name)
                if count >= 2:
                    return 2
            return count
        return 0

    def _enumerate_order_sequences(
        self, node, limit: int = 128
    ) -> Optional[List[List[str]]]:
        """Enumerate representative variable orders for recipient validation.

        Two repetitions are enough to expose a variable that escapes the
        source-to-target interval across quantifier iterations.  Returning
        None makes the caller conservatively skip the rewrite when branching
        would make the analysis too large.
        """
        if isinstance(node, VariablePattern):
            return [[node.name]]
        if isinstance(node, EmptyPattern):
            return [[]]
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._enumerate_order_sequences(node.pattern, limit)
        if isinstance(node, AlternationPattern):
            result = []
            for child in node.patterns:
                child_sequences = self._enumerate_order_sequences(child, limit)
                if child_sequences is None:
                    return None
                result.extend(child_sequences)
                if len(result) > limit:
                    return None
            return result
        if isinstance(node, ConcatenationPattern):
            return self._combine_order_sequences(node.patterns, limit)
        if isinstance(node, QuantifiedPattern):
            base_sequences = self._enumerate_order_sequences(node.pattern, limit)
            if base_sequences is None:
                return None

            if node.atMost is None:
                repeat_counts = {
                    node.atLeast,
                    max(node.atLeast, 1),
                    max(node.atLeast, 2),
                }
            elif node.atMost - node.atLeast <= 2:
                repeat_counts = set(range(node.atLeast, node.atMost + 1))
            else:
                repeat_counts = {node.atLeast, node.atLeast + 1, node.atMost}

            result = []
            for repeat_count in sorted(repeat_counts):
                repeated = [[]]
                for _ in range(repeat_count):
                    next_repeated = []
                    for prefix in repeated:
                        for suffix in base_sequences:
                            next_repeated.append(prefix + suffix)
                            if len(next_repeated) > limit:
                                return None
                    repeated = next_repeated
                result.extend(repeated)
                if len(result) > limit:
                    return None
            return result
        # _collect_between does not currently produce updates for a
        # PermutationPattern, so decline rather than duplicating permutation
        # expansion here.
        return None

    def _combine_order_sequences(
        self, children, limit: int
    ) -> Optional[List[List[str]]]:
        result = [[]]
        for child in children:
            child_sequences = self._enumerate_order_sequences(child, limit)
            if child_sequences is None:
                return None
            next_result = []
            for prefix in result:
                for suffix in child_sequences:
                    next_result.append(prefix + suffix)
                    if len(next_result) > limit:
                        return None
            result = next_result
        return result

    def _all_occurrences_between(
        self,
        sequences: List[List[str]],
        source_var: str,
        recipient_var: str,
        target_var: str,
    ) -> bool:
        """Whether every occurrence is after source and before a later target."""
        saw_recipient = False
        for sequence in sequences:
            recipient_indexes = [
                index for index, name in enumerate(sequence) if name == recipient_var
            ]
            if not recipient_indexes:
                continue
            saw_recipient = True

            source_indexes = [
                index for index, name in enumerate(sequence) if name == source_var
            ]
            target_indexes = [
                index for index, name in enumerate(sequence) if name == target_var
            ]
            if len(source_indexes) != 1 or not target_indexes:
                return False

            source_index = source_indexes[0]
            last_target_index = target_indexes[-1]
            if any(
                index <= source_index or index >= last_target_index
                for index in recipient_indexes
            ):
                return False
        return saw_recipient

    def _collect_between(self, node, source_var: str, target_var: str) -> Set[str]:
        if isinstance(node, VariablePattern):
            if node.name == target_var:
                return {node.name}
            return set()
        if isinstance(node, QuantifiedPattern):
            inner = self._collect_between(node.pattern, source_var, target_var)
            if self._contains_var(node.pattern, source_var) and node.atMost is None:
                inner |= self._collect_all_vars(node.pattern) - {source_var}
            return inner
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._collect_between(node.pattern, source_var, target_var)
        if isinstance(node, AlternationPattern):
            result = set()
            for child in node.patterns:
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    result |= self._collect_between(child, source_var, target_var)
            return result
        if isinstance(node, ConcatenationPattern):
            for i, left in enumerate(node.patterns):
                for j in range(i, len(node.patterns)):
                    right = node.patterns[j]
                    if not self._contains_var(
                        left, source_var
                    ) or not self._contains_var(right, target_var):
                        continue
                    if i == j:
                        nested = self._collect_between(left, source_var, target_var)
                        if nested:
                            return nested
                        continue
                    result = self._collect_after_source(left, source_var)
                    for mid in node.patterns[i + 1 : j]:
                        result |= self._collect_all_vars(mid)
                    result |= self._collect_before_target_inclusive(right, target_var)
                    result.discard(source_var)
                    return result
            return set()
        return set()

    def _collect_after_source(self, node, source_var: str) -> Set[str]:
        if isinstance(node, VariablePattern):
            return set()
        if isinstance(node, QuantifiedPattern):
            result = self._collect_after_source(node.pattern, source_var)
            if self._contains_var(node.pattern, source_var) and node.atMost is None:
                result |= self._collect_all_vars(node.pattern) - {source_var}
            return result
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._collect_after_source(node.pattern, source_var)
        if isinstance(node, AlternationPattern):
            result = set()
            for child in node.patterns:
                if self._contains_var(child, source_var):
                    result |= self._collect_after_source(child, source_var)
            return result
        if isinstance(node, ConcatenationPattern):
            result = set()
            seen_source = False
            for child in node.patterns:
                if seen_source:
                    result |= self._collect_all_vars(child)
                    continue
                if self._contains_var(child, source_var):
                    result |= self._collect_after_source(child, source_var)
                    seen_source = True
            return result
        return set()

    def _collect_before_target_inclusive(self, node, target_var: str) -> Set[str]:
        if isinstance(node, VariablePattern):
            return {node.name} if node.name == target_var else set()
        if isinstance(node, QuantifiedPattern):
            return self._collect_before_target_inclusive(node.pattern, target_var)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._collect_before_target_inclusive(node.pattern, target_var)
        if isinstance(node, AlternationPattern):
            result = set()
            for child in node.patterns:
                if self._contains_var(child, target_var):
                    result |= self._collect_before_target_inclusive(child, target_var)
            return result
        if isinstance(node, ConcatenationPattern):
            result = set()
            for child in node.patterns:
                if self._contains_var(child, target_var):
                    result |= self._collect_before_target_inclusive(child, target_var)
                    return result
                result |= self._collect_all_vars(child)
            return result
        return set()

    def _merge_condition_expr(
        self, current_expr: Optional[Expr], pushed_expr: Expr, plan, var_name: str
    ) -> Expr:
        if current_expr is None:
            return pushed_expr

        conjuncts = self._flatten_and(current_expr)
        pushed_key = self._constraint_key(pushed_expr, plan, var_name)
        merged_terms = []

        for term in conjuncts:
            term_key = self._constraint_key(term, plan, var_name)
            if term_key is not None and term_key == pushed_key:
                stronger = self._pick_stronger_constraint(
                    term, pushed_expr, plan, var_name
                )
                if stronger.to_string() == term.to_string():
                    pushed_expr = term
                continue
            merged_terms.append(term)

        merged_terms.append(pushed_expr)
        result = merged_terms[0]
        for term in merged_terms[1:]:
            result = _simplify_logical_expr(result, "AND", term)
        return result

    def _constraint_key(self, expr: Expr, plan, var_name: str):
        parsed = self._order_constraint_info(
            expr, var_name, plan["order_col"], plan["sort_type"]
        )
        if parsed is None:
            return None
        return ("ORDER_INC", parsed["source_var"], parsed["direction"])

    def _pick_stronger_constraint(
        self, left: Expr, right: Expr, plan, var_name: str
    ) -> Expr:
        left_info = self._order_constraint_info(
            left, var_name, plan["order_col"], plan["sort_type"]
        )
        right_info = self._order_constraint_info(
            right, var_name, plan["order_col"], plan["sort_type"]
        )
        if left_info is None or right_info is None:
            return right
        if left_info["upper"] < right_info["upper"]:
            return left
        if left_info["upper"] > right_info["upper"]:
            return right
        if left_info["inclusive"] and not right_info["inclusive"]:
            return right
        if not left_info["inclusive"] and right_info["inclusive"]:
            return left
        return right

    def _order_constraint_info(
        self, expr: Expr, current_var: str, order_col: str, sort_type: SortType
    ):
        # Even expressions that already look normalized must be checked
        # against the actual first ORDER BY column and sort direction.  Shape
        # alone is insufficient: e.g. SEQ - A.SEQ <= 10 must not be treated
        # as an ORDER BY TS constraint.
        parsed = self._parse_order_incremental_constraint(
            expr, current_var, order_col, sort_type
        )
        if parsed is None:
            return None
        return self._parse_normalized_order_constraint(parsed["normalized_expr"])

    def _parse_normalized_order_constraint(self, expr: Expr):
        if not isinstance(expr, BinaryExpr) or expr.op not in ("<", "<="):
            return None
        if (
            not isinstance(expr.left, BinaryExpr)
            or expr.left.op != "-"
            or not isinstance(expr.right, Literal)
        ):
            return None
        left_col = expr.left.left
        right_col = expr.left.right
        if not isinstance(left_col, ColumnRef) or not isinstance(right_col, ColumnRef):
            return None
        if not isinstance(expr.right.value, (int, float, Decimal)):
            return None
        if left_col.qualifier is None and right_col.qualifier is not None:
            return {
                "direction": "asc",
                "source_var": right_col.qualifier,
                "upper": expr.right.value,
                "inclusive": expr.op == "<=",
            }
        if left_col.qualifier is not None and right_col.qualifier is None:
            return {
                "direction": "desc",
                "source_var": left_col.qualifier,
                "upper": expr.right.value,
                "inclusive": expr.op == "<=",
            }
        return None


incremental_condition_pushdown_test_cases = [
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B C)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND TS - A.TS < 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B C)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND (TS - A.TS) < 10, B AS (TS - A.TS) < 10, N AS (TS - A.TS) < 10
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS DESC, COL1
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B C)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND TS - A.TS > -10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS DESC, COL1
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B C)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND (A.TS - TS) < 10, B AS (A.TS - TS) < 10, N AS (A.TS - TS) < 10
)""",
    ),
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B* C D)
    DEFINE A AS A.TYPE = 'A', B AS TS - A.TS < 100, C AS TYPE = 'C', D AS TYPE = 'D' AND TS - A.TS < 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B* C D)
    DEFINE A AS A.TYPE = 'A', B AS (TS - A.TS) < 10, C AS TYPE = 'C' AND (TS - A.TS) < 10, D AS TYPE = 'D' AND (TS - A.TS) < 10
)""",
    ),
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B (N* | C) D)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C', D AS TYPE = 'D' AND TS - B.TS < 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B (N* | C) D)
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND (TS - B.TS) < 10, D AS TYPE = 'D' AND (TS - B.TS) < 10, N AS (TS - B.TS) < 10
)""",
    ),
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND B.TS - A.TS > 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND (B.TS - A.TS) > 10
)""",
    ),
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* (B | C))
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND TS - A.TS < 10, B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N* (B | C))
    DEFINE A AS A.TYPE = 'A', C AS TYPE = 'C' AND (TS - A.TS) < 10, B AS B.TYPE = 'B'
)""",
    ),
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B | N) C D)
    DEFINE A AS A.TYPE = 'A', C AS C.TYPE = 'C', D AS D.TYPE = 'D' AND TS - B.TS < 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A (B | N) C D)
    DEFINE A AS A.TYPE = 'A', C AS C.TYPE = 'C', D AS D.TYPE = 'D' AND (TS - B.TS) < 10
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        IncrementalConditionPushdownRule,
        incremental_condition_pushdown_test_cases,
        "incremental condition pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
