if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import re
from typing import Dict, List, Optional, Set, Tuple

from clause_parser import ExprParser
from expression import (
    BinaryExpr,
    BetweenExpr,
    CaseExpr,
    ColumnRef,
    Expr,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    NullLiteral,
    SemanticsExpr,
)

from .physical_navigation import PatternNavigationReach
from .row_filtering import RowFilteringRule
from .sequence_filtering import _is_true
from .utils import *


class RowFilteringViaWindowFuncRule(RewriteRule):
    """Prefilter rows while preserving true-variable gaps with windows.

    ``ALL ROWS PER MATCH`` uses the ordinary :class:`RowFilteringRule`: every
    pattern variable must have an independent predicate and original adjacency
    is restored with a generated row number.

    ``ONE ROW PER MATCH`` also permits unconditionally true variables between
    selective singleton variables, for example ``A Z{2,4} B``. Rows
    are numbered before filtering, the lower bound of each true-variable run
    is relaxed to zero, and endpoint row-number constraints restore the run's
    original length. References to removed variables are accepted only below
    ``SUM`` and ``COUNT``. Prefix aggregates recover those values before rows
    are removed; ``SUM`` requires a schema-proven non-null input column and a
    row source that cannot NULL-extend it through an outer join.

    The accepted pattern shape is intentionally conservative. In particular,
    selective endpoint variables must be unique, while a true variable may be
    repeated in multiple separated runs (``A Z+ B Z+ C``).
    """

    _ROW_PATTERN_AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
    _SUPPORTED_REMOVED_ROW_AGGREGATES = {"COUNT", "SUM"}

    @property
    def name(self) -> str:
        return "row_filtering_via_window_func"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_row_filtering_via_window_func_applied", False):
            return False
        if self._is_all_rows_per_match(ast):
            return RowFilteringRule().is_applicable(ast)
        return self._build_one_row_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_row_filtering_via_window_func_applied", False):
            return False

        if self._is_all_rows_per_match(ast):
            changed = RowFilteringRule().apply(ast)
            if changed:
                ast._row_filtering_via_window_func_applied = True
            return changed

        plan = self._build_one_row_plan(ast)
        if plan is None:
            return False

        source = ast.input_table.strip()
        partition_sql = ""
        if ast.partition_by:
            partition_sql = f"PARTITION BY {', '.join(ast.partition_by)} "
        order_sql = ", ".join(
            column if direction == SortType.ASC else f"{column} DESC"
            for column, direction in ast.match_order_by
        )
        over_prefix = f"{partition_sql}ORDER BY {order_sql}"

        generated_items = [f"ROW_NUMBER() OVER ({over_prefix}) AS {plan['row_id']}"]
        for helper in plan["prefix_helpers"]:
            generated_items.append(
                f"{helper['function']}({helper['argument']}) OVER "
                f"({over_prefix} ROWS BETWEEN UNBOUNDED PRECEDING "
                f"AND CURRENT ROW) AS {helper['name']}"
            )

        projection = ", ".join(plan["projection_columns"])
        ast.input_table = (
            f"(SELECT * FROM (SELECT {projection}, "
            f"{', '.join(generated_items)} FROM {source}) "
            f"{plan['numbered_rows']} WHERE {plan['filter_condition']})"
        )

        parser = ExprParser()
        constraint_by_right = {
            segment["right_var"]: self._gap_constraint(segment, plan["row_id"])
            for segment in plan["gaps"]
        }
        for event_var in plan["event_vars"]:
            expression = plan["rewritten_definitions"][event_var]
            constraint = constraint_by_right.get(event_var)
            if constraint is not None:
                expression = BinaryExpr(expression, "AND", constraint)
            define_sql = f"DEFINE {event_var} AS {expression.to_string()}"
            ast.define.definitions[event_var] = parser.parse_define(
                define_sql
            ).definitions[event_var]

        # WNO checks the physical gap in the original input, so retained rows
        # may safely consume a true-variable run after its minimum becomes 0.
        for segment in plan["noise_segments"]:
            if segment["was_bare"]:
                ast.pattern.patterns[segment["pattern_index"]] = QuantifiedPattern(
                    segment["node"],
                    0,
                    1,
                    isGreedy=True,
                )
            else:
                segment["node"].atLeast = 0

        if ast.measures is not None:
            ast.measures.measures = plan["rewritten_measures"]

        ast._row_filtering_via_window_func_applied = True
        return True

    def _build_one_row_plan(self, ast: RPRAbstractSyntaxTree):
        if (
            ast.pattern is None
            or ast.define is None
            or not ast.input_table
            or not ast.match_order_by
            or not self._is_one_row_per_match(ast)
            or not isinstance(ast.pattern, ConcatenationPattern)
        ):
            return None

        # First prove that every PREV/NEXT is inside the original match. Even
        # an in-bounds non-zero call is rejected afterward because it could
        # land on a true-variable row removed by this rewrite.
        if not PatternNavigationReach(ast).all_navigation_stays_within_match():
            return None
        if self._has_nonzero_navigation(ast):
            return None

        # These expressions must remain valid on both sides of the generated
        # derived table. Keep window keys to unqualified simple columns.
        column_ref = r"[A-Z_][A-Z0-9_$]*"
        if any(
            re.fullmatch(column_ref, column.strip()) is None
            for column in (ast.partition_by or [])
        ) or any(
            re.fullmatch(column_ref, column.strip()) is None
            for column, _ in ast.match_order_by
        ):
            return None

        pattern_plan = self._analyze_pattern(ast)
        if pattern_plan is None:
            return None
        event_vars, gaps, noise_segments, noise_vars = pattern_plan

        noise_dependent_names = self._noise_dependent_names(ast, noise_vars)
        if self._skip_references_noise(ast, noise_dependent_names):
            return None

        predicates = []
        seen_predicates = set()
        for event_var in event_vars:
            condition = ast.define.definitions.get(event_var)
            if condition is None:
                return None
            independent = _strip_var_qualifier(
                condition.get_independent_filter(),
                event_var,
            )
            if independent is None or _is_true(independent):
                return None
            predicate_sql = independent.to_string()
            if predicate_sql not in seen_predicates:
                seen_predicates.add(predicate_sql)
                predicates.append(predicate_sql)
        if not predicates:
            return None

        projection_columns = self._projection_columns(ast)
        if not projection_columns:
            return None
        row_id = choose_generated_column_name(ast, "WNO", projection_columns)
        used_columns = set(projection_columns)
        used_columns.add(row_id)

        prefix_helpers: List[Dict[str, str]] = []
        helper_by_key: Dict[Tuple[str, str], str] = {}

        def get_prefix_helper(function: str, argument: str) -> str:
            key = (function, argument)
            existing = helper_by_key.get(key)
            if existing is not None:
                return existing
            token = re.sub(
                r"[^A-Z0-9_$]+",
                "_",
                argument.upper(),
            ).strip("_")
            base = f'{function}_{token or "VALUE"}'
            name = choose_generated_column_name(ast, base, used_columns)
            used_columns.add(name)
            helper_by_key[key] = name
            prefix_helpers.append(
                {
                    "name": name,
                    "function": function,
                    "argument": argument,
                }
            )
            return name

        segments_by_noise = {
            noise_var: [
                segment
                for segment in noise_segments
                if segment["noise_var"] == noise_var
            ]
            for noise_var in noise_vars
        }
        first_event = event_vars[0]
        last_event = event_vars[-1]
        event_indexes = {event_var: index for index, event_var in enumerate(event_vars)}

        rewritten_definitions = {}
        for event_var in event_vars:
            condition = ast.define.definitions[event_var]
            visible_segments = {
                noise_var: [
                    segment
                    for segment in segments
                    if segment["right_event_index"] <= event_indexes[event_var]
                ]
                for noise_var, segments in segments_by_noise.items()
            }
            rewritten = self._rewrite_expression(
                ast,
                condition.expression,
                noise_dependent_names,
                noise_vars,
                visible_segments,
                first_event,
                event_var,
                row_id,
                get_prefix_helper,
            )
            if rewritten is None:
                return None
            rewritten_definitions[event_var] = rewritten

        rewritten_measures = {}
        for alias, expression in (
            ast.measures.measures.items() if ast.measures is not None else []
        ):
            rewritten = self._rewrite_expression(
                ast,
                expression,
                noise_dependent_names,
                noise_vars,
                segments_by_noise,
                first_event,
                last_event,
                row_id,
                get_prefix_helper,
            )
            if rewritten is None:
                return None
            rewritten_measures[alias] = rewritten

        numbered_rows = choose_generated_relation_name(ast, "ADD_WNO_ROWS")
        return {
            "event_vars": event_vars,
            "gaps": gaps,
            "noise_segments": noise_segments,
            "row_id": row_id,
            "numbered_rows": numbered_rows,
            "projection_columns": projection_columns,
            "prefix_helpers": prefix_helpers,
            "rewritten_definitions": rewritten_definitions,
            "rewritten_measures": rewritten_measures,
            "filter_condition": " OR ".join(
                f"({predicate})" for predicate in predicates
            ),
        }

    def _analyze_pattern(self, ast: RPRAbstractSyntaxTree):
        items = ast.pattern.patterns
        if len(items) < 3 or not isinstance(items[0], VariablePattern):
            return None
        if self._is_true_variable(ast, items[0].name):
            return None

        event_vars: List[str] = [items[0].name]
        gaps = []
        noise_segments = []
        noise_vars: Set[str] = set()
        previous_event = items[0].name
        pending_noise = None

        for pattern_index, item in enumerate(items[1:], start=1):
            if isinstance(item, QuantifiedPattern):
                if (
                    pending_noise is not None
                    or not isinstance(item.pattern, VariablePattern)
                    or (item.atMost is not None and item.atMost < item.atLeast)
                ):
                    return None
                if not self._is_true_variable(ast, item.pattern.name):
                    return None
                pending_noise = {
                    "node": item,
                    "noise_var": item.pattern.name,
                    "minimum": item.atLeast,
                    "maximum": item.atMost,
                    "pattern_index": pattern_index,
                    "was_bare": False,
                }
                continue

            if not isinstance(item, VariablePattern):
                return None
            if self._is_true_variable(ast, item.name):
                if pending_noise is not None:
                    return None
                pending_noise = {
                    "node": item,
                    "noise_var": item.name,
                    "minimum": 1,
                    "maximum": 1,
                    "pattern_index": pattern_index,
                    "was_bare": True,
                }
                continue
            event_vars.append(item.name)
            gap = {
                "node": (pending_noise["node"] if pending_noise is not None else None),
                "noise_var": (
                    pending_noise["noise_var"] if pending_noise is not None else None
                ),
                "left_var": previous_event,
                "right_var": item.name,
                "minimum": (
                    pending_noise["minimum"] if pending_noise is not None else 0
                ),
                "maximum": (
                    pending_noise["maximum"] if pending_noise is not None else 0
                ),
                "index": len(gaps),
                "right_event_index": len(event_vars) - 1,
                "pattern_index": (
                    pending_noise["pattern_index"]
                    if pending_noise is not None
                    else None
                ),
                "was_bare": (
                    pending_noise["was_bare"] if pending_noise is not None else False
                ),
            }
            gaps.append(gap)
            if pending_noise is not None:
                noise_var = pending_noise["noise_var"]
                noise_vars.add(noise_var)
                noise_segments.append(gap)
            previous_event = item.name
            pending_noise = None

        if pending_noise is not None:
            return None

        # DEFINE and WNO endpoint expressions are keyed by variable name.
        if len(set(event_vars)) != len(event_vars):
            return None
        if noise_vars & set(event_vars):
            return None
        if not noise_vars or not any(
            segment["maximum"] is None or segment["maximum"] > 0
            for segment in noise_segments
        ):
            return None

        return event_vars, gaps, noise_segments, noise_vars

    @staticmethod
    def _is_true_variable(ast: RPRAbstractSyntaxTree, variable: str) -> bool:
        condition = ast.define.definitions.get(variable)
        return condition is None or _is_true(condition.expression)

    def _rewrite_expression(
        self,
        ast: RPRAbstractSyntaxTree,
        expression: Expr,
        noise_names: Set[str],
        primary_noise_vars: Set[str],
        segments_by_noise: Dict[str, List[dict]],
        first_event: str,
        current_event: str,
        row_id: str,
        get_prefix_helper,
    ) -> Optional[Expr]:
        """Rewrite SUM/COUNT observations of rows removed by the filter."""

        def rewrite(node: Expr) -> Optional[Tuple[Expr, bool]]:
            if isinstance(node, SemanticsExpr):
                rewritten = rewrite(node.expr)
                if rewritten is None:
                    return None
                inner, changed = rewritten
                # A rewritten aggregate is now an endpoint scalar.
                return (
                    (inner, True)
                    if changed
                    else (
                        SemanticsExpr(node.semantics, inner),
                        False,
                    )
                )

            if isinstance(node, ColumnRef):
                if node.qualifier in noise_names:
                    return None
                return node, False

            if isinstance(node, (Literal, NullLiteral)):
                return node, False

            if isinstance(node, FunctionCall):
                name = node.name.upper()
                if name in ("PREV", "NEXT"):
                    return None

                if name in self._ROW_PATTERN_AGGREGATES:
                    aggregate = self._rewrite_aggregate(
                        ast,
                        node,
                        noise_names,
                        primary_noise_vars,
                        segments_by_noise,
                        first_event,
                        current_event,
                        row_id,
                        get_prefix_helper,
                    )
                    if aggregate is not None:
                        return aggregate, True
                    # None also represents an unchanged aggregate scoped only
                    # to retained variables. Distinguish that safe case here.
                    if self._aggregate_observes_removed_rows(node, noise_names):
                        return None

                if name in ("FIRST", "LAST") and self._function_has_unqualified_column(
                    node
                ):
                    return None

                args = []
                changed = False
                for argument in node.args:
                    rewritten_arg = rewrite(argument)
                    if rewritten_arg is None:
                        return None
                    value, argument_changed = rewritten_arg
                    args.append(value)
                    changed = changed or argument_changed
                return FunctionCall(node.name, args), changed

            if isinstance(node, BinaryExpr):
                left = rewrite(node.left)
                right = rewrite(node.right)
                if left is None or right is None:
                    return None
                return BinaryExpr(left[0], node.op, right[0]), left[1] or right[1]

            if isinstance(node, BetweenExpr):
                value = rewrite(node.expr)
                lower = rewrite(node.lower)
                upper = rewrite(node.upper)
                if value is None or lower is None or upper is None:
                    return None
                return (
                    BetweenExpr(value[0], lower[0], upper[0]),
                    value[1] or lower[1] or upper[1],
                )

            if isinstance(node, CaseExpr):
                cases = []
                changed = False
                for condition, result in node.cases:
                    new_condition = rewrite(condition)
                    new_result = rewrite(result)
                    if new_condition is None or new_result is None:
                        return None
                    cases.append((new_condition[0], new_result[0]))
                    changed = changed or new_condition[1] or new_result[1]
                else_expr = None
                if node.else_expr is not None:
                    rewritten_else = rewrite(node.else_expr)
                    if rewritten_else is None:
                        return None
                    else_expr = rewritten_else[0]
                    changed = changed or rewritten_else[1]
                return CaseExpr(cases, else_expr), changed

            if isinstance(node, InExpr):
                value = rewrite(node.expr)
                if value is None:
                    return None
                values = []
                changed = value[1]
                for item in node.values:
                    rewritten_item = rewrite(item)
                    if rewritten_item is None:
                        return None
                    values.append(rewritten_item[0])
                    changed = changed or rewritten_item[1]
                return InExpr(value[0], values, node.is_not), changed

            if isinstance(node, IsNullExpr):
                value = rewrite(node.expr)
                if value is None:
                    return None
                return IsNullExpr(value[0], node.is_not), value[1]

            if isinstance(node, NotExpr):
                value = rewrite(node.expr)
                if value is None:
                    return None
                return NotExpr(value[0]), value[1]

            return None

        result = rewrite(expression)
        return result[0] if result is not None else None

    def _rewrite_aggregate(
        self,
        ast: RPRAbstractSyntaxTree,
        function: FunctionCall,
        noise_names: Set[str],
        primary_noise_vars: Set[str],
        segments_by_noise: Dict[str, List[dict]],
        first_event: str,
        current_event: str,
        row_id: str,
        get_prefix_helper,
    ) -> Optional[Expr]:
        name = function.name.upper()
        if name not in self._SUPPORTED_REMOVED_ROW_AGGREGATES:
            return None
        if len(function.args) != 1:
            return None
        argument = function.args[0]

        if isinstance(argument, ColumnRef):
            if argument.qualifier in primary_noise_vars:
                segments = segments_by_noise.get(argument.qualifier, [])
                if name == "COUNT" and argument.name == "*":
                    return self._noise_row_count_expression(segments, row_id)
                if argument.name == "*":
                    return None
                if name == "SUM" and not self._sum_input_is_non_null(
                    ast,
                    argument.name,
                ):
                    return None
                return self._noise_column_aggregate_expression(
                    name,
                    argument.name,
                    segments,
                    row_id,
                    get_prefix_helper,
                )

            if argument.qualifier in noise_names:
                # A SUBSET containing a true variable has no unique gap map.
                return None

            if argument.qualifier is None:
                if argument.name == "*":
                    if name == "COUNT":
                        return self._match_row_count_expression(
                            first_event,
                            current_event,
                            row_id,
                        )
                    return None
                if name == "SUM" and not self._sum_input_is_non_null(
                    ast,
                    argument.name,
                ):
                    return None
                return self._match_column_aggregate_expression(
                    name,
                    argument.name,
                    first_event,
                    current_event,
                    get_prefix_helper,
                )

            # Retained-variable aggregates remain unchanged.
            return None

        if isinstance(argument, NullLiteral):
            return Literal(0) if name == "COUNT" else NullLiteral()

        if isinstance(argument, Literal):
            count = self._match_row_count_expression(
                first_event,
                current_event,
                row_id,
            )
            if name == "COUNT":
                return count
            if isinstance(argument.value, (int, float)):
                return (
                    count
                    if argument.value == 1
                    else BinaryExpr(count, "*", Literal(argument.value))
                )
            return None

        return None

    @staticmethod
    def _sum_input_is_non_null(
        ast: RPRAbstractSyntaxTree,
        column: str,
    ) -> bool:
        """Whether prefix-SUM subtraction cannot encounter a NULL prefix.

        Base-table metadata alone is insufficient: a declared NOT NULL column
        can become NULL on the null-supplying side of an outer join. The AST
        retains the input relation as SQL text, so reject this optimization
        conservatively whenever that row source may perform NULL extension.
        """
        return (
            not input_may_null_extend_columns(ast.input_table)
            and ast.is_column_nullable(column) is False
        )

    def _aggregate_observes_removed_rows(
        self,
        function: FunctionCall,
        noise_names: Set[str],
    ) -> bool:
        if _expr_references_any_var(function, noise_names):
            return True
        if self._function_has_unqualified_column(function):
            return True
        if function.name.upper() in self._ROW_PATTERN_AGGREGATES:
            # SUM(1), COUNT(1), and COUNT(*) have no collected column but still
            # aggregate across all matched rows.
            if not function.args:
                return True
            argument = function.args[0]
            if isinstance(argument, ColumnRef) and argument.qualifier is not None:
                return False
            return not argument.collect_columns()
        return False

    def _noise_column_aggregate_expression(
        self,
        function: str,
        column: str,
        segments: List[dict],
        row_id: str,
        get_prefix_helper,
    ) -> Expr:
        if not segments:
            return Literal(0) if function == "COUNT" else NullLiteral()

        helper = get_prefix_helper(function, column)
        values = [
            self._prefix_between_endpoints(
                function,
                column,
                helper,
                segment["left_var"],
                segment["right_var"],
            )
            for segment in segments
        ]
        result = self._sum_expressions(values)
        if function == "SUM" and sum(segment["minimum"] for segment in segments) == 0:
            row_count = self._noise_row_count_expression(segments, row_id)
            return CaseExpr(
                [(BinaryExpr(row_count, "=", Literal(0)), NullLiteral())],
                result,
            )
        return result

    @staticmethod
    def _match_column_aggregate_expression(
        function: str,
        column: str,
        first_event: str,
        current_event: str,
        get_prefix_helper,
    ) -> Expr:
        helper = get_prefix_helper(function, column)
        result = BinaryExpr(
            ColumnRef(current_event, helper),
            "-",
            ColumnRef(first_event, helper),
        )
        contribution = (
            ColumnRef(first_event, column)
            if function == "SUM"
            else CaseExpr(
                [(IsNullExpr(ColumnRef(first_event, column)), Literal(0))],
                Literal(1),
            )
        )
        return BinaryExpr(result, "+", contribution)

    @staticmethod
    def _prefix_between_endpoints(
        function: str,
        column: str,
        helper: str,
        left_var: str,
        right_var: str,
    ) -> Expr:
        result = BinaryExpr(
            ColumnRef(right_var, helper),
            "-",
            ColumnRef(left_var, helper),
        )
        right_contribution = (
            ColumnRef(right_var, column)
            if function == "SUM"
            else CaseExpr(
                [(IsNullExpr(ColumnRef(right_var, column)), Literal(0))],
                Literal(1),
            )
        )
        return BinaryExpr(result, "-", right_contribution)

    @staticmethod
    def _match_row_count_expression(
        first_event: str,
        current_event: str,
        row_id: str,
    ) -> Expr:
        return BinaryExpr(
            BinaryExpr(
                ColumnRef(current_event, row_id),
                "-",
                ColumnRef(first_event, row_id),
            ),
            "+",
            Literal(1),
        )

    @staticmethod
    def _noise_row_count_expression(
        segments: List[dict],
        row_id: str,
    ) -> Expr:
        if not segments:
            return Literal(0)

        # A Z B Z C collapses to C.WNO - A.WNO - 2.
        runs: List[List[dict]] = []
        for segment in segments:
            if runs and segment["index"] == runs[-1][-1]["index"] + 1:
                runs[-1].append(segment)
            else:
                runs.append([segment])

        values = []
        for run in runs:
            value = BinaryExpr(
                ColumnRef(run[-1]["right_var"], row_id),
                "-",
                ColumnRef(run[0]["left_var"], row_id),
            )
            values.append(BinaryExpr(value, "-", Literal(len(run))))
        return RowFilteringViaWindowFuncRule._sum_expressions(values)

    @staticmethod
    def _sum_expressions(expressions: List[Expr]) -> Expr:
        result = expressions[0]
        for expression in expressions[1:]:
            result = BinaryExpr(result, "+", expression)
        return result

    @staticmethod
    def _gap_constraint(segment: dict, row_id: str) -> Expr:
        distance = BinaryExpr(
            ColumnRef(segment["right_var"], row_id),
            "-",
            ColumnRef(segment["left_var"], row_id),
        )
        lower = Literal(segment["minimum"] + 1)
        if segment["maximum"] is None:
            return BinaryExpr(distance, ">=", lower)
        return BetweenExpr(
            distance,
            lower,
            Literal(segment["maximum"] + 1),
        )

    @staticmethod
    def _has_nonzero_navigation(ast: RPRAbstractSyntaxTree) -> bool:
        for condition in ast.define.definitions.values():
            if condition is not None and _has_nonzero_physical_navigation(
                condition.expression,
            ):
                return True
        if ast.measures is not None:
            return any(
                _has_nonzero_physical_navigation(expression)
                for expression in ast.measures.measures.values()
            )
        return False

    @staticmethod
    def _function_has_unqualified_column(function: FunctionCall) -> bool:
        return any(
            isinstance(node, ColumnRef) and node.qualifier is None
            for argument in function.args
            for node in _walk_expression(argument)
        )

    @staticmethod
    def _noise_dependent_names(
        ast: RPRAbstractSyntaxTree,
        noise_vars: Set[str],
    ) -> Set[str]:
        result = set(noise_vars)
        changed = True
        while changed:
            changed = False
            for subset_name, members in (ast.subsets or {}).items():
                if subset_name not in result and any(
                    member in result for member in members
                ):
                    result.add(subset_name)
                    changed = True
        return result

    @staticmethod
    def _skip_references_noise(
        ast: RPRAbstractSyntaxTree,
        noise_names: Set[str],
    ) -> bool:
        skip = (ast.after_match_skip_to or "").upper()
        identifiers = set(re.findall(r"[A-Z_][A-Z0-9_$#]*", skip))
        return bool(identifiers & noise_names)

    @staticmethod
    def _projection_columns(ast: RPRAbstractSyntaxTree) -> Optional[List[str]]:
        required = ast.get_required_input_table_columns()
        if not isinstance(required, set):
            return None
        ordered = list(ast.partition_by or [])
        ordered.extend(column for column, _ in (ast.match_order_by or []))
        ordered.extend(sorted(required))

        result = []
        seen = set()
        for column in ordered:
            clean = str(column).strip().split(".")[-1]
            if not clean or clean.upper() in {"TRUE", "FALSE"} or clean.upper() in seen:
                continue
            seen.add(clean.upper())
            result.append(clean)
        return result

    @staticmethod
    def _is_one_row_per_match(ast: RPRAbstractSyntaxTree) -> bool:
        mode = (
            ast.rows_per_match.strip().upper()
            if ast.rows_per_match is not None
            else "ONE ROW PER MATCH"
        )
        return mode == "ONE ROW PER MATCH"

    @staticmethod
    def _is_all_rows_per_match(ast: RPRAbstractSyntaxTree) -> bool:
        return (
            ast.rows_per_match is not None
            and ast.rows_per_match.strip().upper() == "ALL ROWS PER MATCH"
        )


# Backward compatibility for the earlier, unexported draft class name.
WindowRowFilteringRule = RowFilteringViaWindowFuncRule


window_row_filtering_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A Z{2,4} B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM (
    SELECT * FROM (
        SELECT PT, TS, ID, TYPE,
               ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS, ID DESC) AS WNO
        FROM T
    ) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A Z{0,4} B)
    DEFINE A AS A.TYPE = 'A',
           B AS B.TYPE = 'B' AND B.WNO - A.WNO BETWEEN 3 AND 5
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID, COUNT(Z.*) AS ZCNT
    PATTERN(A Z{2,4} B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM (
    SELECT * FROM (
        SELECT PT, TS, ID, TYPE,
               ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS, ID DESC) AS WNO
        FROM T
    ) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID,
             B.WNO - A.WNO - 1 AS ZCNT
    PATTERN(A Z{0,4} B)
    DEFINE A AS A.TYPE = 'A',
           B AS B.TYPE = 'B' AND B.WNO - A.WNO BETWEEN 3 AND 5
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID,
             COUNT(Z.*) AS ZCNT
    PATTERN(A Z{2,4} B Z{1,4} C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM (
    SELECT * FROM (
        SELECT PT, TS, ID, TYPE,
               ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS, ID DESC) AS WNO
        FROM T
    ) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B') OR (TYPE = 'C')
) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID, C.ID AS CID,
             C.WNO - A.WNO - 2 AS ZCNT
    PATTERN(A Z{0,4} B Z{0,4} C)
    DEFINE A AS TYPE = 'A',
           B AS TYPE = 'B' AND B.WNO - A.WNO BETWEEN 3 AND 5,
           C AS TYPE = 'C' AND C.WNO - B.WNO BETWEEN 2 AND 5
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES SUM(Z.VAL) AS ZSUM, A.ID AS AID, B.ID AS BID
    PATTERN(A Z{1,100} B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
        """
SELECT * FROM (
    SELECT * FROM (
        SELECT PT, TS, ID, TYPE, VAL,
               ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO,
               SUM(VAL) OVER (PARTITION BY PT ORDER BY TS ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS SUM_VAL
        FROM T
    ) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES B.SUM_VAL - A.SUM_VAL - B.VAL AS ZSUM,
             A.ID AS AID, B.ID AS BID
    PATTERN(A Z{0,100} B)
    DEFINE A AS TYPE = 'A',
           B AS TYPE = 'B' AND B.WNO - A.WNO BETWEEN 2 AND 101
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B+)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT PT, TS, ID, TYPE, VAL
FROM (
    SELECT * FROM (
        SELECT PT, TS, ID, TYPE, VAL,
               ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO
        FROM T
    ) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B+)
    DEFINE A AS A.TYPE = 'A',
           B AS B.TYPE = 'B' AND WNO = PREV(WNO) + 1
)""",
    ),
    # SUM over a nullable column cannot use prefix subtraction safely.
    (
        6,
        """
SELECT * FROM T_NULLABLE MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES SUM(Z.VAL) AS ZSUM
    PATTERN(A Z{1,4} B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
        """
SELECT * FROM T_NULLABLE MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES SUM(Z.VAL) AS ZSUM
    PATTERN(A Z{1,4} B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
    ),
]


default_window_row_filtering_table_schemas = {
    "T": {
        "columns": {
            "PT": "LONG",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(20)",
            "VAL": "DOUBLE",
        },
        "column_nullables": {
            "PT": False,
            "TS": False,
            "ID": False,
            "TYPE": False,
            "VAL": False,
        },
    },
    "T_NULLABLE": {
        "columns": {
            "PT": "LONG",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(20)",
            "VAL": "DOUBLE",
        },
        "column_nullables": {
            "PT": False,
            "TS": False,
            "ID": False,
            "TYPE": False,
            "VAL": True,
        },
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        RowFilteringViaWindowFuncRule,
        window_row_filtering_test_cases,
        "row filtering via window function",
        verbose,
        default_window_row_filtering_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
