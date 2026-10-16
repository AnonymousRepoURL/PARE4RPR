if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
import re
from dataclasses import dataclass

from .utils import *
from .physical_navigation import (
    PatternNavigationReach,
    analyze_navigation_operand,
    collect_expression_navigation,
    expand_navigation_base_vars,
    max_navigation_offset,
)
from .pattern_analysis import enumerate_pattern_sequences
from patternnode import PatternNode


@dataclass(frozen=True)
class OrderedRangeTailPlan:
    """One finite boundary fragment appended to a pushed input range."""

    expr: Expr
    order_sql: str
    limit: int


@dataclass(frozen=True)
class OrderedRangePushdownPlan:
    """Canonical final plan shared by endpoint and middle-variable analysis."""

    keep_expr: Expr
    tails: Tuple[OrderedRangeTailPlan, ...] = ()
    remove_specs: Tuple[tuple, ...] = ()
    input_rebind_alias: Optional[str] = None
    origin: str = ""
    update_define: bool = True


@dataclass(frozen=True)
class OrderedRangeCompositePlan:
    """One ordered pipeline of independently proven input reducers.

    Keeping stages explicit is important when both endpoint and middle
    analysis need a finite Top-K tail.  In general, filtering a Top-K result
    is not equivalent to applying the filter before Top-K, so those plans must
    not be flattened by merely AND-ing their predicates.  A pure endpoint
    filter is flattened separately by the composer below.
    """

    stages: Tuple[OrderedRangePushdownPlan, ...]
    origin: str = "composite"


class OrderedRangePushdownSupport:
    """
    Shared planning and SQL-generation support for ordered range pushdown.

        Currently, only the first ORDER BY column is considered for range pushdown.
        The first ORDER BY column must be provably non-nullable in the effective
        input row source because the boundary calculations do not preserve NULL
        rows.  Inputs containing an outer join are rejected conservatively: an
        outer join can null-extend a source column declared NOT NULL.

    Caution:
        - Physical navigation functions PREV(, k), NEXT(, k) in DEFINE or
        MEASURES influence pushdown range boundaries and require special handling.
        Even if PREV(, k), NEXT(, k) occur, we can still push down the range predicate;
        detailed test cases can see test_rewrite_rule.py
        - NULLIF or other function that may generate null value will lead to wrong state
        - if we UNOIN NULL value also incur same problem
    """

    def _apply_ordered_range_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: OrderedRangePushdownPlan,
    ) -> bool:
        """Apply one canonical WHERE/UNION plan."""

        if len(plan.tails) == 0:
            pushed_where = plan.keep_expr.to_string()
            input_table = ast.input_table.strip()
            target_fragment = f"WHERE {pushed_where}"
            if target_fragment in input_table:
                return False
            rewritten_input = f"(SELECT * FROM {ast.input_table} WHERE {pushed_where})"
            ast.input_table = self._rebind_rewritten_input(
                rewritten_input,
                plan.input_rebind_alias,
            )
            if plan.update_define:
                self._prune_define_conditions(ast, plan.remove_specs)
                self._ensure_non_empty_define(ast)
            return True

        input_table_sql = ast.input_table
        if ast.partition_by is not None and len(ast.partition_by) > 0:
            partition_union_sql = self._build_partition_aware_union_sql(ast, plan)
            if partition_union_sql is None:
                return False
            ast.input_table = self._rebind_rewritten_input(
                partition_union_sql,
                plan.input_rebind_alias,
            )
        else:
            keep_sql = plan.keep_expr.to_string()
            parts = [f"SELECT * FROM {input_table_sql} WHERE {keep_sql}"]
            for tail in plan.tails:
                parts.append(
                    _build_ordered_tail_select(
                        input_table_sql,
                        tail.expr.to_string(),
                        tail.order_sql,
                        tail.limit,
                    )
                )
            rewritten_input = "(" + " UNION ALL ".join(parts) + ")"
            ast.input_table = self._rebind_rewritten_input(
                rewritten_input,
                plan.input_rebind_alias,
            )

        if plan.update_define:
            self._prune_define_conditions(ast, plan.remove_specs)
            self._ensure_non_empty_define(ast)
        return True

    def _apply_ordered_range_composite_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: OrderedRangeCompositePlan,
    ) -> bool:
        """Apply every reducer stage atomically in its proven order.

        Each stage performs exactly the DEFINE pruning justified by its own
        proof.  A working copy prevents a late SQL-generation failure from
        leaving a partially rewritten AST.
        """

        if len(plan.stages) == 0:
            return False

        working = copy.deepcopy(ast)
        changed = False
        for stage in plan.stages:
            if not self._apply_ordered_range_plan(working, stage):
                return False
            changed = True

        if not changed:
            return False

        working._ordered_range_pushdown_applied = True
        ast.__dict__.clear()
        ast.__dict__.update(working.__dict__)
        return True

    def _to_ordered_range_plan(
        self,
        legacy_plan,
        input_rebind_alias: Optional[str],
        origin: str,
    ) -> OrderedRangePushdownPlan:
        """Normalize the existing internal planner dictionaries at one boundary."""
        keep_expr = (
            legacy_plan["expr"]
            if legacy_plan["mode"] == "where"
            else legacy_plan["keep_expr"]
        )
        tails = tuple(
            OrderedRangeTailPlan(expr, order_sql, limit)
            for expr, order_sql, limit in legacy_plan.get("tail_exprs", [])
        )
        return OrderedRangePushdownPlan(
            keep_expr=keep_expr,
            tails=tails,
            remove_specs=tuple(legacy_plan.get("remove_specs", [])),
            input_rebind_alias=input_rebind_alias,
            origin=origin,
        )

    def _plan_range_constraint(
        self,
        expr: Expr,
        order_col: str,
        domain: RangeDomain,
    ) -> Optional[RangeConstraint]:
        """Return a column-bound constraint suitable for plan composition."""

        extracted = self._extract_range_from_expr(
            expr,
            "",
            order_col,
            domain,
            {None},
        )
        if extracted is None:
            return None
        return RangeConstraint(
            domain=domain,
            column=ColumnRef(None, order_col),
            lower=extracted.lower,
            lower_inclusive=extracted.lower_inclusive,
            upper=extracted.upper,
            upper_inclusive=extracted.upper_inclusive,
        )

    def _compose_endpoint_middle_plan(
        self,
        endpoint: OrderedRangePushdownPlan,
        middle: OrderedRangePushdownPlan,
        order_col: str,
        domain: RangeDomain,
    ):
        """Compose endpoint and middle reducers without reordering Top-K.

        A tail-free endpoint plan is a global input guard.  It can therefore
        be intersected into the middle keep region and into every middle tail
        *before* the tail's Top-K is evaluated.  When endpoint itself has a
        tail, keep both reducers as ordered stages; flattening two Top-K plans
        is not valid in general.
        """

        if endpoint.input_rebind_alias != middle.input_rebind_alias:
            return None

        if len(endpoint.tails) > 0:
            return OrderedRangeCompositePlan(
                stages=(endpoint, middle),
                origin="endpoint+middle",
            )

        endpoint_constraint = self._plan_range_constraint(
            endpoint.keep_expr,
            order_col,
            domain,
        )
        middle_constraint = self._plan_range_constraint(
            middle.keep_expr,
            order_col,
            domain,
        )
        if endpoint_constraint is None or middle_constraint is None:
            return None

        keep_constraint = endpoint_constraint.intersect(middle_constraint)
        if keep_constraint is None:
            # The original query may be unsatisfiable, but proving and
            # replacing it with an empty input belongs to the false-pushdown
            # rule.  Fall back to the independently proven endpoint plan.
            return None
        keep_expr = keep_constraint.to_expr(order_col)
        if keep_expr is None:
            return None

        combined_tails = []
        for tail in middle.tails:
            tail_constraint = self._plan_range_constraint(
                tail.expr,
                order_col,
                domain,
            )
            if tail_constraint is None:
                return None
            guarded_tail = endpoint_constraint.intersect(tail_constraint)
            if guarded_tail is None:
                # Endpoint filtering proves that this tail region contains no
                # row that can participate in a match.
                continue
            guarded_tail_expr = guarded_tail.to_expr(order_col)
            if guarded_tail_expr is None:
                return None
            combined_tails.append(
                OrderedRangeTailPlan(
                    guarded_tail_expr,
                    tail.order_sql,
                    tail.limit,
                )
            )

        return OrderedRangePushdownPlan(
            keep_expr=keep_expr,
            tails=tuple(combined_tails),
            remove_specs=endpoint.remove_specs + middle.remove_specs,
            input_rebind_alias=endpoint.input_rebind_alias,
            origin="endpoint+middle",
            update_define=endpoint.update_define or middle.update_define,
        )

    def _build_endpoint_plan(
        self,
        ast: RPRAbstractSyntaxTree,
    ) -> Optional[OrderedRangePushdownPlan]:
        context = self._prepare_order_range_context(ast)
        if context is None:
            return None
        order_col = context["order_col"]
        if not self._order_range_input_is_safe(ast, order_col):
            return None

        plan = self._build_plan_without_anchor_guard(ast, context)
        if plan is None:
            return None

        filter_expr = (
            plan.get("expr") if plan.get("mode") == "where" else plan.get("keep_expr")
        )
        if not self._anchor_filter_is_safe(ast, filter_expr, order_col):
            return None
        return self._to_ordered_range_plan(
            plan,
            context["input_rebind_alias"],
            "endpoint",
        )

    def _prepare_order_range_context(self, ast: RPRAbstractSyntaxTree):
        """Build syntax/type context shared by endpoint and middle rules."""
        if ast.match_order_by is None or len(ast.match_order_by) == 0:
            return None
        if ast.pattern.get_min_row_num() == 0:
            return None

        order_col = ast.match_order_by[0][0].strip()
        if not order_col:
            return None
        order_col = order_col.split(".")[-1]
        domain = resolve_range_domain(ast, order_col)
        if domain is None:
            return None

        qualifier_safe, input_rebind_alias = self._resolve_input_rebind_alias(ast)
        if not qualifier_safe:
            return None
        return {
            "order_col": order_col,
            "sort_type": ast.match_order_by[0][1],
            "domain": domain,
            "input_rebind_alias": input_rebind_alias,
        }

    def _order_range_input_is_safe(
        self,
        ast: RPRAbstractSyntaxTree,
        order_col: str,
    ) -> bool:
        """Whether range complements preserve every effective input row."""
        return ast.is_column_nullable(
            order_col
        ) is False and not self._input_may_null_extend_columns(ast.input_table)

    @staticmethod
    def _rebind_rewritten_input(input_sql: str, alias: Optional[str]) -> str:
        """Expose an original input correlation name on a generated source."""
        return f"{input_sql} {alias}" if alias is not None else input_sql

    def _resolve_input_rebind_alias(
        self,
        ast: RPRAbstractSyntaxTree,
    ) -> Tuple[bool, Optional[str]]:
        """Find the input qualifier that must remain visible after wrapping.

        A correlation name used by PARTITION BY or ORDER BY is scoped to the
        original row-pattern input. Wrapping that input in a derived table
        hides the name, so bind the same name to the complete generated input.
        Complex sources are rejected because rebinding one constituent of a
        join could change which duplicate output column a qualifier denotes.
        """
        qualifiers: Dict[str, str] = {}
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

    def _build_plan_without_anchor_guard(self, ast: RPRAbstractSyntaxTree, context):
        order_col = context["order_col"]
        domain = context["domain"]
        sort_type = context["sort_type"]
        left_vars = ast.pattern.get_leftmost_variables()
        right_vars = ast.pattern.get_rightmost_variables()
        subset_sequences = (
            self._enumerate_pattern_sequences(ast.pattern) if ast.subsets else None
        )

        left_plan = self._build_side_plan(
            ast,
            order_col,
            sort_type,
            left_vars,
            "left",
            domain,
            subset_sequences,
        )
        right_plan = self._build_side_plan(
            ast,
            order_col,
            sort_type,
            right_vars,
            "right",
            domain,
            subset_sequences,
        )

        # Prefer union plan because it captures finite-tail optimization.
        if left_plan is not None and left_plan["mode"] == "union":
            return left_plan
        if right_plan is not None and right_plan["mode"] == "union":
            return right_plan

        cross_union_plan = self._build_cross_side_union_plan(
            ast,
            order_col,
            sort_type,
            left_plan,
            right_plan,
            domain,
        )
        if cross_union_plan is not None:
            return cross_union_plan

        nav_union_plan = self._build_nav_single_side_union_plan(
            ast,
            order_col,
            left_plan,
            right_plan,
            domain,
        )
        if nav_union_plan is not None:
            return nav_union_plan

        global_constraint = RangeConstraint(domain)
        remove_specs = []
        has_any = False

        if left_plan is not None and left_plan["mode"] == "where":
            has_any = True
            if left_plan["bound_kind"] == "lower":
                global_constraint.apply_lower(
                    left_plan["bound_lit"], left_plan["bound_inclusive"]
                )
            else:
                global_constraint.apply_upper(
                    left_plan["bound_lit"], left_plan["bound_inclusive"]
                )
            remove_specs.extend(left_plan.get("remove_specs", []))

        if right_plan is not None and right_plan["mode"] == "where":
            has_any = True
            if right_plan["bound_kind"] == "lower":
                global_constraint.apply_lower(
                    right_plan["bound_lit"], right_plan["bound_inclusive"]
                )
            else:
                global_constraint.apply_upper(
                    right_plan["bound_lit"], right_plan["bound_inclusive"]
                )
            remove_specs.extend(right_plan.get("remove_specs", []))

        if not has_any:
            return None

        merged = global_constraint.to_expr(order_col)
        if merged is None:
            return None

        return {"mode": "where", "expr": merged, "remove_specs": remove_specs}

    @staticmethod
    def _input_may_null_extend_columns(input_table_sql: str) -> bool:
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

    def _contains_anchor(self, pattern: PatternNode) -> bool:
        if isinstance(pattern, AnchorPattern):
            return True
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(self._contains_anchor(child) for child in pattern.patterns)
        if isinstance(pattern, (QuantifiedPattern, ExcludedPattern)):
            return self._contains_anchor(pattern.pattern)
        return False

    def _collect_anchor_types(self, pattern: PatternNode) -> Set[AnchorType]:
        if isinstance(pattern, AnchorPattern):
            return {pattern.anchor_type} | self._collect_anchor_types(pattern.pattern)
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            result: Set[AnchorType] = set()
            for child in pattern.patterns:
                result.update(self._collect_anchor_types(child))
            return result
        if isinstance(pattern, (QuantifiedPattern, ExcludedPattern)):
            return self._collect_anchor_types(pattern.pattern)
        return set()

    def _anchor_filter_is_safe(
        self, ast: RPRAbstractSyntaxTree, filter_expr: Expr, col_name: str
    ) -> bool:
        """Whether a range filter preserves every partition edge used by anchors."""
        anchor_types = self._collect_anchor_types(ast.pattern)
        if not anchor_types:
            return True
        if filter_expr is None:
            return False

        domain = resolve_range_domain(ast, col_name)
        if domain is None:
            return False
        constraint = self._extract_range_from_expr(
            filter_expr,
            "",
            col_name,
            domain,
            {None},
        )
        if constraint is None:
            return False

        sort_type = ast.match_order_by[0][1]
        removes_start = (
            sort_type == SortType.ASC and constraint.lower is not None
        ) or (sort_type == SortType.DESC and constraint.upper is not None)
        removes_end = (sort_type == SortType.ASC and constraint.upper is not None) or (
            sort_type == SortType.DESC and constraint.lower is not None
        )
        if AnchorType.START in anchor_types and removes_start:
            return False
        if AnchorType.END in anchor_types and removes_end:
            return False
        return True

    def _build_cross_side_union_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        col_name: str,
        sort_type: SortType,
        left_plan,
        right_plan,
        domain: RangeDomain,
    ):
        if left_plan is None or right_plan is None:
            return None
        if left_plan.get("mode") != "where" or right_plan.get("mode") != "where":
            return None

        left_kind = left_plan.get("bound_kind")
        right_kind = right_plan.get("bound_kind")
        if {left_kind, right_kind} != {"lower", "upper"}:
            return None

        # Cross-side UNION is only needed when a navigation function (PREV/NEXT) requires
        # history rows outside the main range.
        left_nav = left_plan.get("nav_tail_limit", 0) or 0
        right_nav = right_plan.get("nav_tail_limit", 0) or 0
        if left_nav == 0 and right_nav == 0:
            return None

        # Build the main window constraint (both bounds combined).
        keep_constraint = RangeConstraint(domain)
        if left_kind == "lower":
            keep_constraint.apply_lower(
                left_plan["bound_lit"], left_plan["bound_inclusive"]
            )
            keep_constraint.apply_upper(
                right_plan["bound_lit"], right_plan["bound_inclusive"]
            )
        else:
            keep_constraint.apply_upper(
                left_plan["bound_lit"], left_plan["bound_inclusive"]
            )
            keep_constraint.apply_lower(
                right_plan["bound_lit"], right_plan["bound_inclusive"]
            )

        keep_expr = keep_constraint.to_expr(col_name)
        if keep_expr is None:
            return None

        # Generate one tail SELECT per side that requires navigation rows.
        # Left tail first (PREV), then right tail (NEXT).
        tail_exprs = []

        if left_nav > 0:
            left_tail = self._tail_expr(
                col_name,
                left_plan["bound_lit"],
                left_plan["bound_kind"],
                left_plan["bound_inclusive"],
            )
            if left_tail is None:
                return None
            # Navigation tail ORDER: get rows closest to the boundary first.
            # Lower-bound tail (TS <= lower) → DESC; upper-bound tail (TS >= upper) → ASC.
            left_tail_order_sql = self._build_tail_order_sql(
                ast, left_plan["bound_kind"]
            )
            tail_exprs.append((left_tail, left_tail_order_sql, left_nav))

        if right_nav > 0:
            right_tail = self._tail_expr(
                col_name,
                right_plan["bound_lit"],
                right_plan["bound_kind"],
                right_plan["bound_inclusive"],
            )
            if right_tail is None:
                return None
            right_tail_order_sql = self._build_tail_order_sql(
                ast, right_plan["bound_kind"]
            )
            tail_exprs.append((right_tail, right_tail_order_sql, right_nav))

        if not tail_exprs:
            return None

        # Only remove DEFINE conditions from the side with NO navigation tail.
        # When a side HAS a tail, its endpoint variable rows may come from that tail
        # (beyond the main window), so its condition must stay in DEFINE to filter them.
        if left_nav == 0:
            remove_specs = list(left_plan.get("remove_specs", []))
        elif right_nav == 0:
            remove_specs = list(right_plan.get("remove_specs", []))
        else:
            remove_specs = []  # Both sides have tails; keep all conditions.

        return {
            "mode": "union",
            "col_name": col_name,
            "keep_expr": keep_expr,
            "tail_exprs": tail_exprs,
            "remove_specs": remove_specs,
        }

    def _normalize_column_sequence(self, columns: List[str]) -> List[str]:
        normalized: List[str] = []
        seen: Set[str] = set()

        for col in columns:
            clean = col.strip()
            if len(clean) == 0:
                continue
            output_name = clean.split(".")[-1] if "." in clean else clean
            key = output_name.upper()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(output_name)

        return normalized

    def _get_partition_union_projection_columns(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[List[str]]:
        rows_per_match = (
            ast.rows_per_match.strip().upper()
            if ast.rows_per_match is not None
            else "ONE ROW PER MATCH"
        )

        if rows_per_match == "ALL ROWS PER MATCH":
            if ast.table_schemas is None:
                return None
            input_columns = ast._extract_join_source_columns()
            if input_columns is None or len(input_columns) == 0:
                return None
            return self._normalize_column_sequence(input_columns)

        ordered_columns: List[str] = []
        for col in ast.partition_by or []:
            ordered_columns.append(col)
        for col, _ in ast.match_order_by or []:
            ordered_columns.append(col)

        required_columns = ast.get_required_input_table_columns()
        if isinstance(required_columns, set):
            ordered_columns.extend(sorted(required_columns))
        elif required_columns != "*":
            return None

        projection_columns = self._normalize_column_sequence(ordered_columns)
        return projection_columns if len(projection_columns) > 0 else None

    def _build_partition_keep_select(
        self, input_table_sql: str, projection_columns: List[str], where_sql: str
    ) -> str:
        select_list = ", ".join(projection_columns)
        return f"(SELECT {select_list} FROM {input_table_sql} WHERE {where_sql})"

    def _build_partition_window_tail_select(
        self,
        input_table_sql: str,
        projection_columns: List[str],
        partition_columns: List[str],
        where_sql: str,
        order_sql: str,
        limit: int,
        rownum_alias: str,
    ) -> str:
        select_list = ", ".join(projection_columns)
        partition_sql = ", ".join(self._normalize_column_sequence(partition_columns))
        return (
            f"(SELECT {select_list} FROM ("
            f"SELECT {select_list}, ROW_NUMBER() OVER (PARTITION BY {partition_sql} ORDER BY {order_sql}) AS {rownum_alias} "
            f"FROM {input_table_sql} WHERE {where_sql}"
            f") WHERE {rownum_alias} <= {limit})"
        )

    def _build_partition_aware_union_sql(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: OrderedRangePushdownPlan,
    ) -> Optional[str]:
        projection_columns = self._get_partition_union_projection_columns(ast)
        if projection_columns is None or len(projection_columns) == 0:
            return None

        keep_sql = self._build_partition_keep_select(
            ast.input_table,
            projection_columns,
            plan.keep_expr.to_string(),
        )

        parts = [keep_sql]
        rownum_alias = choose_generated_column_name(
            ast, "RPR_TAIL_RN", projection_columns
        )
        for tail in plan.tails:
            parts.append(
                self._build_partition_window_tail_select(
                    ast.input_table,
                    projection_columns,
                    ast.partition_by or [],
                    tail.expr.to_string(),
                    tail.order_sql,
                    tail.limit,
                    rownum_alias,
                )
            )

        return "(" + " UNION ALL ".join(parts) + ")"

    def _build_nav_single_side_union_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        col_name: str,
        left_plan,
        right_plan,
        domain: RangeDomain,
    ):
        candidate_plans = []
        if left_plan is not None and left_plan.get("mode") == "where":
            candidate_plans.append(left_plan)
        if right_plan is not None and right_plan.get("mode") == "where":
            candidate_plans.append(right_plan)

        if len(candidate_plans) == 0:
            return None

        keep_constraint = RangeConstraint(domain)
        tail_exprs = []
        remove_specs = []

        for side_plan in candidate_plans:
            if side_plan["bound_kind"] == "lower":
                keep_constraint.apply_lower(
                    side_plan["bound_lit"], side_plan["bound_inclusive"]
                )
            else:
                keep_constraint.apply_upper(
                    side_plan["bound_lit"], side_plan["bound_inclusive"]
                )

            nav_limit = side_plan.get("nav_tail_limit", 0) or 0
            if nav_limit > 0:
                tail_expr = self._tail_expr(
                    col_name,
                    side_plan["bound_lit"],
                    side_plan["bound_kind"],
                    side_plan["bound_inclusive"],
                )
                if tail_expr is None:
                    return None
                tail_order_sql = self._build_tail_order_sql(
                    ast, side_plan["bound_kind"]
                )
                tail_exprs.append((tail_expr, tail_order_sql, nav_limit))
            else:
                remove_specs.extend(side_plan.get("remove_specs", []))

        if len(tail_exprs) == 0:
            return None

        keep_expr = keep_constraint.to_expr(col_name)
        if keep_expr is None:
            return None

        return {
            "mode": "union",
            "col_name": col_name,
            "keep_expr": keep_expr,
            "tail_exprs": tail_exprs,
            "remove_specs": remove_specs,
        }

    def _build_side_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        col_name: str,
        sort_type: SortType,
        endpoint_vars: Set[str],
        endpoint_kind: str,
        domain: RangeDomain,
        subset_sequences: Optional[List[List[str]]] = None,
    ):
        source_constraints, coverage = self._collect_source_constraints(
            ast,
            col_name,
            endpoint_vars,
            domain,
            subset_sequences,
        )
        if len(source_constraints) == 0:
            return None

        direct_bound = self._direct_bound_kind(sort_type, endpoint_kind)
        opposite_bound = "upper" if direct_bound == "lower" else "lower"

        direct_sources = [
            src
            for src, con in source_constraints.items()
            if self._has_bound(con, direct_bound)
        ]
        direct_union_bound = None
        removable_sources = []
        if len(direct_sources) > 0:
            direct_union_bound = self._union_boundary(
                [source_constraints[src] for src in direct_sources], direct_bound
            )
            if direct_union_bound is None:
                return None
            removable_sources = self._pick_removable_sources(
                direct_sources, source_constraints, direct_bound, direct_union_bound
            )

        subset_tail_limit = None
        if direct_union_bound is not None:
            subset_tail_limit = self._estimate_subset_tail_limit(
                subset_sequences, endpoint_kind, endpoint_vars, direct_sources, coverage
            )

            # A direct bound may be pushed only when it covers every variable
            # that can occur at this endpoint.  Otherwise an optional/alternate
            # branch can start or end outside the bound.  SUBSET analysis is the
            # one exception: it computes a finite tail that preserves the
            # uncovered endpoint rows.
            direct_all_covered = all(
                self._covered_by_bound(
                    endpoint_var, coverage, source_constraints, direct_bound
                )
                for endpoint_var in endpoint_vars
            )
            if not direct_all_covered and subset_tail_limit is None:
                direct_union_bound = None
                direct_sources = []
                removable_sources = []

        subset_nav_tail_limit = self._estimate_subset_navigation_tail_limit(
            ast, endpoint_kind, direct_sources
        )
        if subset_nav_tail_limit is None:
            return None

        remove_specs = []
        for removable_source in removable_sources:
            source_info = self._parse_constraint_source_name(removable_source)
            remove_specs.append(
                (
                    source_info["define_var"],
                    col_name,
                    direct_bound,
                    direct_union_bound[0],
                    direct_union_bound[1],
                    source_info["qualifiers"],
                )
            )

        # opposite-side union optimization requires every endpoint variable covered by opposite bound
        opposite_all_covered = True
        for endpoint_var in endpoint_vars:
            if not self._covered_by_bound(
                endpoint_var, coverage, source_constraints, opposite_bound
            ):
                opposite_all_covered = False
                break

        tail_limit = self._estimate_union_tail_limit(ast, endpoint_kind, endpoint_vars)
        direct_nav_tail_limit = self._estimate_navigation_tail_limit(
            ast,
            endpoint_kind,
            endpoint_vars,
        )

        opposite_union_bound = None
        if opposite_all_covered:
            opposite_sources = [
                src
                for src, con in source_constraints.items()
                if self._has_bound(con, opposite_bound)
            ]
            if len(opposite_sources) > 0:
                opposite_union_bound = self._union_boundary(
                    [source_constraints[src] for src in opposite_sources],
                    opposite_bound,
                )

        pattern_max = ast.pattern.get_max_row_num()
        if (
            opposite_union_bound is not None
            and tail_limit is not None
            and tail_limit > 0
            and pattern_max is not None
            and subset_nav_tail_limit == 0
            # A simultaneous direct bound may need a second tail on the other
            # side.  This single-tail plan cannot represent that; let the
            # direct-bound navigation plan handle it instead.
            and not (direct_union_bound is not None and direct_nav_tail_limit)
        ):
            if direct_union_bound is None:
                keep_expr = self._direct_expr(
                    col_name,
                    opposite_union_bound[0],
                    opposite_bound,
                    opposite_union_bound[1],
                )
            else:
                keep_constraint = RangeConstraint(domain)
                if direct_bound == "lower":
                    keep_constraint.apply_lower(
                        direct_union_bound[0], direct_union_bound[1]
                    )
                    keep_constraint.apply_upper(
                        opposite_union_bound[0], opposite_union_bound[1]
                    )
                else:
                    keep_constraint.apply_upper(
                        direct_union_bound[0], direct_union_bound[1]
                    )
                    keep_constraint.apply_lower(
                        opposite_union_bound[0], opposite_union_bound[1]
                    )
                keep_expr = keep_constraint.to_expr(col_name)

            if keep_expr is None:
                return None

            tail_expr = self._tail_expr(
                col_name,
                opposite_union_bound[0],
                opposite_bound,
                opposite_union_bound[1],
            )
            if tail_expr is None:
                return None

            tail_order_sql = self._build_tail_order_sql(ast, opposite_bound)
            return {
                "mode": "union",
                "col_name": col_name,
                "keep_expr": keep_expr,
                "tail_exprs": [(tail_expr, tail_order_sql, tail_limit)],
                "remove_specs": remove_specs,
            }

        if direct_union_bound is None:
            return None

        boundary_lit, boundary_inclusive = direct_union_bound
        where_expr = self._direct_expr(
            col_name, boundary_lit, direct_bound, boundary_inclusive
        )
        if where_expr is None:
            return None
        nav_tail_limit = self._estimate_navigation_tail_limit(
            ast, endpoint_kind, endpoint_vars
        )
        combined_nav_tail_limit = max(
            limit
            for limit in [nav_tail_limit, subset_tail_limit, 0]
            if limit is not None
        )
        return {
            "mode": "where",
            "bound_kind": direct_bound,
            "bound_lit": boundary_lit,
            "bound_inclusive": boundary_inclusive,
            "expr": where_expr,
            "tail_limit": tail_limit,
            "nav_tail_limit": max(combined_nav_tail_limit, subset_nav_tail_limit),
            "remove_specs": remove_specs,
        }

    def _estimate_union_tail_limit(
        self, ast: RPRAbstractSyntaxTree, endpoint_kind: str, endpoint_vars: Set[str]
    ) -> Optional[int]:
        side_counts = []
        for var in endpoint_vars:
            if endpoint_kind == "left":
                cnt = ast.pattern.get_right_max_row_num(var)
            else:
                cnt = ast.pattern.get_left_max_row_num(var)

            if cnt is None:
                return None
            side_counts.append(cnt)

        pattern_tail = max(side_counts) if len(side_counts) > 0 else None

        if pattern_tail is None:
            return None

        # This UNION branch supplements the side opposite the endpoint's
        # direct bound: a left endpoint needs a right tail, and a right
        # endpoint needs a left tail.  Navigation on that tail is therefore
        # NEXT and PREV respectively (the reverse of direct-bound navigation).
        nav_offset = self._estimate_opposite_tail_navigation_offset(
            ast,
            endpoint_kind,
        )
        if nav_offset is None:
            return None
        return pattern_tail + nav_offset

    def _estimate_opposite_tail_navigation_offset(
        self,
        ast: RPRAbstractSyntaxTree,
        endpoint_kind: str,
    ) -> Optional[int]:
        target_func = "NEXT" if endpoint_kind == "left" else "PREV"
        best = 0

        for def_var, condition in ast.define.definitions.items():
            if condition is None:
                continue
            for _base_var, offset in self._collect_nav_calls(
                condition.expression,
                target_func,
                def_var,
            ):
                if offset is None:
                    return None
                best = max(best, offset)

        # A qualified reference in MEASURES applies only to its variable;
        # unqualified references are conservatively checked against all of
        # them, matching _estimate_navigation_tail_limit.
        for expr in _iter_measure_expressions(ast):
            for default_var in ast.pattern.get_variables():
                for _base_var, offset in self._collect_nav_calls(
                    expr,
                    target_func,
                    default_var,
                ):
                    if offset is None:
                        return None
                    best = max(best, offset)

        return best

    def _estimate_navigation_tail_limit(
        self, ast: RPRAbstractSyntaxTree, endpoint_kind: str, endpoint_vars: Set[str]
    ) -> Optional[int]:
        target_func = "PREV" if endpoint_kind == "left" else "NEXT"
        best = 0
        found = False

        def account_navigation(base_var: Optional[str], offset: Optional[int]) -> bool:
            nonlocal best, found
            if base_var is None or offset is None:
                return False
            if offset == 0:
                return True

            for resolved_base_var in self._expand_navigation_base_vars(
                ast,
                base_var,
            ):
                # Rows of the base variable that already sit inside the match
                # window between the filtered endpoint and the navigation base.
                if endpoint_kind == "left":
                    min_rows_inside = ast.pattern.get_left_min_row_num(
                        resolved_base_var,
                    )
                else:
                    min_rows_inside = ast.pattern.get_right_min_row_num(
                        resolved_base_var,
                    )
                if min_rows_inside is None:
                    continue

                extra = max(0, offset - min_rows_inside)
                if extra > 0:
                    found = True
                    best = max(best, extra)
            return True

        for def_var, condition in ast.define.definitions.items():
            if condition is None:
                continue
            for base_var, offset in self._collect_nav_calls(
                condition.expression,
                target_func,
                def_var,
            ):
                if not account_navigation(base_var, offset):
                    # Unknown navigation offset — unsafe for UNION optimization.
                    return None

        # MEASURES has the same physical PREV/NEXT navigation capability as
        # DEFINE.  There is no enclosing DEFINE variable for an unqualified
        # reference, so consider every pattern variable as a possible base;
        # qualified references naturally match only their own variable.
        for expr in _iter_measure_expressions(ast):
            for default_var in ast.pattern.get_variables():
                for base_var, offset in self._collect_nav_calls(
                    expr,
                    target_func,
                    default_var,
                ):
                    if not account_navigation(base_var, offset):
                        return None

        return best if found else None

    def _collect_nav_calls(
        self,
        expr: Expr,
        target_func: str,
        default_var: str,
    ) -> List[Tuple[Optional[str], Optional[int]]]:
        """Collect navigation calls and resolve each call's actual base variable.

        A qualified argument such as ``PREV(A.ID, 2)`` is based on ``A`` even
        when it occurs in another variable's DEFINE clause.  An unqualified
        argument is based on the DEFINE variable supplied as ``default_var``.
        FIRST/LAST wrappers and compound scalar operands preserve the single
        qualifier used by their inner row-pattern column references.
        """
        return [
            (base_var, offset)
            for base_var, offset, _selector, _logical_offset, _is_universal in self._collect_nav_call_details(
                expr, target_func, default_var
            )
        ]

    def _collect_nav_call_details(
        self,
        expr: Expr,
        target_func: str,
        default_var: str,
    ) -> List[Tuple[Optional[str], Optional[int], str, Optional[int], bool]]:
        """Collect physical navigation plus its logical base selection.

        ``FIRST`` selects from the start of the rows mapped to the base
        variable; ``LAST`` selects from their end. A plain operand has running
        LAST semantics in DEFINE. The logical offset is zero-based, unlike the
        default physical PREV/NEXT offset of one.
        """
        return [
            (
                call.base_variable,
                call.physical_offset,
                call.selector,
                call.logical_offset,
                call.is_universal,
            )
            for call in collect_expression_navigation(
                expr,
                default_var,
                target_func,
            )
        ]

    def _analyze_navigation_operand(
        self,
        expr: Expr,
        default_var: str,
    ) -> Tuple[Optional[str], str, Optional[int], bool]:
        """Analyze one physical-navigation operand in one shared routine.

        SQL/RPR permits a navigation operand to be a scalar expression such as
        ``PREV(LAST(A.PRICE + A.TAX), 2)``. Every row-pattern reference in a
        valid operand denotes the same variable. The returned tuple contains
        its variable, FIRST/LAST/RUNNING_LAST selector, logical offset, and
        whether the operand uses the implicit universal variable.
        """
        return analyze_navigation_operand(expr, default_var)

    def _expand_navigation_base_vars(
        self,
        ast: RPRAbstractSyntaxTree,
        base_var: str,
    ) -> Set[str]:
        """Expand a SUBSET navigation base to its primary pattern variables."""
        return expand_navigation_base_vars(ast, base_var)

    def _extract_navigation_offset(self, func: FunctionCall) -> Optional[int]:
        # PREV(x) / NEXT(x) default offset is 1.
        return _extract_navigation_offset(func)

    def _direct_bound_kind(self, sort_type: SortType, endpoint_kind: str) -> str:
        # ASC + left => lower, ASC + right => upper
        # DESC + left => upper, DESC + right => lower
        if sort_type == SortType.ASC:
            return "lower" if endpoint_kind == "left" else "upper"
        return "upper" if endpoint_kind == "left" else "lower"

    def _tail_order(self, sort_type: SortType, endpoint_kind: str):
        if endpoint_kind == "left":
            return "ASC" if sort_type == SortType.ASC else "DESC"
        return "DESC" if sort_type == SortType.ASC else "ASC"

    def _build_tail_order_sql(self, ast: RPRAbstractSyntaxTree, bound_kind: str) -> str:
        if ast.match_order_by is None or len(ast.match_order_by) == 0:
            return ""

        first_col, first_sort = ast.match_order_by[0]
        first_col_name = first_col.split(".")[-1] if "." in first_col else first_col
        first_dir = "DESC" if bound_kind == "lower" else "ASC"

        first_sort_dir = "ASC" if first_sort == SortType.ASC else "DESC"
        reverse_rest = first_dir != first_sort_dir

        parts = [f"{first_col_name} {first_dir}"]
        for i in range(1, len(ast.match_order_by)):
            col, stype = ast.match_order_by[i]
            col_name = col.split(".")[-1] if "." in col else col
            dir_str = "ASC" if stype == SortType.ASC else "DESC"
            if reverse_rest:
                dir_str = "DESC" if dir_str == "ASC" else "ASC"
            parts.append(f"{col_name} {dir_str}")
        return ", ".join(parts)

    def _direct_expr(
        self, col_name: str, lit: Literal, bound_kind: str, inclusive: bool
    ) -> Optional[Expr]:
        column = ColumnRef(None, col_name)
        if bound_kind == "lower":
            op = ">=" if inclusive else ">"
            return BinaryExpr(column, op, lit)
        if bound_kind == "upper":
            op = "<=" if inclusive else "<"
            return BinaryExpr(column, op, lit)
        return None

    def _tail_expr(
        self, col_name: str, lit: Literal, bound_kind: str, inclusive: bool
    ) -> Optional[Expr]:
        column = ColumnRef(None, col_name)
        if bound_kind == "upper":
            op = ">" if inclusive else ">="
            return BinaryExpr(column, op, lit)
        if bound_kind == "lower":
            op = "<" if inclusive else "<="
            return BinaryExpr(column, op, lit)
        return None

    def _union_boundary(self, constraints, bound_kind: str):
        target = None
        target_inclusive = False

        for constraint in constraints:
            if bound_kind == "lower":
                lit = constraint.lower
                inclusive = constraint.lower_inclusive
            else:
                lit = constraint.upper
                inclusive = constraint.upper_inclusive

            if lit is None:
                return None

            if target is None:
                target = lit
                target_inclusive = inclusive
                continue

            cmp_result = _compare_literal_values(
                _range_bound_value(lit),
                _range_bound_value(target),
            )
            if cmp_result is None:
                return None

            if bound_kind == "lower":
                if cmp_result < 0:
                    target = lit
                    target_inclusive = inclusive
                elif cmp_result == 0:
                    target_inclusive = target_inclusive or inclusive
            else:
                if cmp_result > 0:
                    target = lit
                    target_inclusive = inclusive
                elif cmp_result == 0:
                    target_inclusive = target_inclusive or inclusive

        if target is None:
            return None

        return target, target_inclusive

    def _has_bound(self, constraint: RangeConstraint, bound_kind: str) -> bool:
        if bound_kind == "lower":
            return constraint.lower is not None
        return constraint.upper is not None

    def _covered_by_bound(
        self,
        var_name: str,
        coverage: Dict[str, Set[str]],
        source_constraints: Dict[str, RangeConstraint],
        bound_kind: str,
    ) -> bool:
        for source_name, covered_vars in coverage.items():
            if var_name not in covered_vars:
                continue
            constraint = source_constraints.get(source_name)
            if constraint is None:
                continue
            if self._has_bound(constraint, bound_kind):
                return True
        return False

    def _pick_removable_sources(
        self,
        source_names: List[str],
        source_constraints: Dict[str, RangeConstraint],
        bound_kind: str,
        global_bound,
    ) -> List[str]:
        if len(source_names) == 0 or global_bound is None:
            return []

        global_lit, global_inclusive = global_bound
        removable = []
        for name in source_names:
            constraint = source_constraints[name]
            lit = constraint.lower if bound_kind == "lower" else constraint.upper
            inclusive = (
                constraint.lower_inclusive
                if bound_kind == "lower"
                else constraint.upper_inclusive
            )
            if lit is None:
                continue
            cmp_result = _compare_literal_values(
                _range_bound_value(lit),
                _range_bound_value(global_lit),
            )
            if cmp_result is None or cmp_result != 0:
                continue
            if inclusive == global_inclusive:
                removable.append(name)

        return removable

    def _collect_source_constraints(
        self,
        ast: RPRAbstractSyntaxTree,
        col_name: str,
        endpoint_vars: Set[str],
        domain: RangeDomain,
        subset_sequences: Optional[List[List[str]]] = None,
    ):
        source_constraints: Dict[str, RangeConstraint] = {}
        coverage: Dict[str, Set[str]] = {}
        for var_name in endpoint_vars:
            condition = ast.define.definitions.get(var_name)
            # A bound on another pattern variable is not necessarily a bound on
            # this endpoint.  For example, with ORDER BY TS ASC and PATTERN(A B C),
            # B.TS < 100 does not imply C.TS < 100.  Only the endpoint's own
            # qualified column (or its unqualified equivalent) is safe here.
            expr = condition.expression if condition is not None else None
            constraint = self._extract_range_from_expr(
                expr,
                var_name,
                col_name,
                domain,
                {var_name, None},
            )
            if constraint is None:
                constraint = RangeConstraint(domain)
            source_constraints[var_name] = constraint
            coverage[var_name] = {var_name}

        subset_sources = self._collect_subset_source_constraints(
            ast,
            col_name,
            domain,
            subset_sequences,
        )
        for source_name, source_constraint, covered_vars in subset_sources:
            source_constraints[source_name] = source_constraint
            coverage[source_name] = covered_vars

        return source_constraints, coverage

    def _collect_subset_source_constraints(
        self,
        ast: RPRAbstractSyntaxTree,
        col_name: str,
        domain: RangeDomain,
        sequences: Optional[List[List[str]]],
    ):
        if (
            ast.subsets is None
            or len(ast.subsets) == 0
            or sequences is None
            or len(sequences) == 0
        ):
            return []

        result = []
        for define_var, condition in ast.define.definitions.items():
            if condition is None:
                continue
            for subset_name, subset_vars in ast.subsets.items():
                covered_vars = self._running_subset_last_vars(
                    sequences, define_var, set(subset_vars)
                )
                if covered_vars is None or len(covered_vars) == 0:
                    continue
                constraint = self._extract_range_from_expr(
                    condition.expression,
                    define_var,
                    col_name,
                    domain,
                    {subset_name},
                )
                if constraint is None:
                    continue
                result.append(
                    (
                        self._constraint_source_name(define_var, subset_name),
                        constraint,
                        covered_vars,
                    )
                )
        return result

    def _constraint_source_name(self, define_var: str, subset_name: str) -> str:
        return f"{define_var}::{subset_name}"

    def _parse_constraint_source_name(self, source_name: str) -> dict:
        if "::" not in source_name:
            return {"define_var": source_name, "qualifiers": None}
        define_var, subset_name = source_name.split("::", 1)
        return {"define_var": define_var, "qualifiers": {subset_name}}

    def _running_subset_last_vars(
        self, sequences: List[List[str]], define_var: str, subset_vars: Set[str]
    ) -> Optional[Set[str]]:
        selected_by_sequence: List[Tuple[List[str], Set[str]]] = []

        for seq in sequences:
            selected_vars: Set[str] = set()
            for idx, var_name in enumerate(seq):
                if var_name != define_var:
                    continue
                last_subset_var = None
                for prior_var in seq[: idx + 1]:
                    if prior_var in subset_vars:
                        last_subset_var = prior_var
                if last_subset_var is None:
                    return None

                # Coverage is represented by primary-variable name, not by a
                # particular occurrence in the expanded pattern sequence.  If
                # that name occurs more than once, a running SUBSET reference
                # can constrain (for example) the second A while the endpoint
                # analysis incorrectly treats the first A as constrained.  Do
                # not derive endpoint coverage until the representation can
                # prove occurrence identity.
                if seq.count(last_subset_var) != 1:
                    return None
                selected_vars.add(last_subset_var)

            selected_by_sequence.append((seq, selected_vars))

        candidates = {
            var_name
            for _seq, selected_vars in selected_by_sequence
            for var_name in selected_vars
        }
        if not candidates:
            return None

        # A DEFINE variable may exist in only some alternation branches, and
        # the running last member of a SUBSET may differ by branch.  A primary
        # variable is globally covered by this source only when every expanded
        # sequence containing that variable selects it at least once.  Merely
        # selecting A in one branch cannot constrain an A in another branch
        # where the DEFINE variable is absent or the SUBSET selects B instead.
        covered_vars = {
            candidate
            for candidate in candidates
            if all(
                candidate not in seq or candidate in selected_vars
                for seq, selected_vars in selected_by_sequence
            )
        }

        return covered_vars if covered_vars else None

    def _enumerate_pattern_sequences(
        self, pattern: PatternNode, limit: int = 128
    ) -> Optional[List[List[str]]]:
        return enumerate_pattern_sequences(pattern, limit)

    def _estimate_subset_tail_limit(
        self,
        sequences: Optional[List[List[str]]],
        endpoint_kind: str,
        endpoint_vars: Set[str],
        direct_sources: List[str],
        coverage: Dict[str, Set[str]],
    ) -> Optional[int]:
        if len(direct_sources) == 0 or sequences is None or len(sequences) == 0:
            return None

        covered_vars: Set[str] = set()
        for source in direct_sources:
            covered_vars.update(coverage.get(source, set()))

        if all(var in covered_vars for var in endpoint_vars):
            return None

        best = 0
        for seq in sequences:
            if endpoint_kind == "left":
                endpoint_indexes = [
                    idx for idx, var in enumerate(seq) if var in endpoint_vars
                ]
                if len(endpoint_indexes) == 0:
                    return None
                start_idx = min(endpoint_indexes)
                candidate_indexes = [
                    idx
                    for idx in range(start_idx, len(seq))
                    if seq[idx] in covered_vars
                ]
                if len(candidate_indexes) == 0:
                    return None
                best = max(best, min(candidate_indexes) - start_idx)
            else:
                endpoint_indexes = [
                    idx for idx, var in enumerate(seq) if var in endpoint_vars
                ]
                if len(endpoint_indexes) == 0:
                    return None
                end_idx = max(endpoint_indexes)
                candidate_indexes = [
                    idx for idx in range(0, end_idx + 1) if seq[idx] in covered_vars
                ]
                if len(candidate_indexes) == 0:
                    return None
                best = max(best, end_idx - max(candidate_indexes))

        return best if best > 0 else None

    def _estimate_subset_navigation_tail_limit(
        self, ast: RPRAbstractSyntaxTree, endpoint_kind: str, direct_sources: List[str]
    ) -> Optional[int]:
        target_func = "PREV" if endpoint_kind == "left" else "NEXT"
        best = 0

        for source_name in direct_sources:
            source_info = self._parse_constraint_source_name(source_name)
            qualifiers = source_info.get("qualifiers")
            if not qualifiers:
                continue

            condition = ast.define.definitions.get(source_info["define_var"])
            if condition is None:
                continue

            for qualifier in qualifiers:
                nav_limit = self._collect_navigation_limit_for_qualifier(
                    condition.expression, qualifier, target_func
                )
                if nav_limit is None:
                    return None
                best = max(best, nav_limit)

        return best

    def _collect_navigation_limit_for_qualifier(
        self, expr: Expr, qualifier: str, target_func: str
    ) -> Optional[int]:
        best = 0
        for call in collect_expression_navigation(expr, None, target_func):
            if qualifier not in call.operand_qualifiers:
                continue
            offset = call.physical_offset
            if offset is None:
                return None
            best = max(best, offset)
        return best

    def _extract_range_from_expr(
        self,
        expr: Expr,
        var_name: str,
        col_name: str,
        domain: RangeDomain,
        qualifier_filter: Optional[Set[str]] = None,
    ) -> Optional[RangeConstraint]:
        if expr is None:
            return None

        conjuncts: List[Expr] = []
        if not self._flatten_and(expr, conjuncts):
            return None

        constraint = RangeConstraint(domain)
        has_range = False
        for atom in conjuncts:
            applied = self._apply_range_atom(
                constraint, atom, var_name, col_name, qualifier_filter
            )
            has_range = has_range or applied

        if not has_range:
            return None
        return constraint

    def _flatten_and(self, expr: Expr, conjuncts: List[Expr]) -> bool:
        return _flatten_and_strict(expr, conjuncts)

    def _is_target_column(
        self,
        expr: Expr,
        var_name: str,
        col_name: str,
        qualifier_filter: Optional[Set[str]] = None,
    ) -> bool:
        if isinstance(expr, SemanticsExpr):
            return self._is_target_column(
                expr.expr, var_name, col_name, qualifier_filter
            )
        if not isinstance(expr, ColumnRef):
            return False
        if expr.name != col_name:
            return False
        if qualifier_filter is not None:
            return expr.qualifier in qualifier_filter
        # Callers that do not supply a filter intentionally accept any qualifier.
        return True

    def _to_literal(self, expr: Expr) -> Optional[Literal]:
        if isinstance(expr, SemanticsExpr):
            return self._to_literal(expr.expr)
        return expr if isinstance(expr, Literal) else None

    def _reverse_op(self, op: str) -> Optional[str]:
        return {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}.get(op)

    def _apply_range_atom(
        self,
        constraint: RangeConstraint,
        atom: Expr,
        var_name: str,
        col_name: str,
        qualifier_filter: Optional[Set[str]] = None,
    ) -> bool:
        if isinstance(atom, SemanticsExpr):
            return self._apply_range_atom(
                constraint, atom.expr, var_name, col_name, qualifier_filter
            )

        if isinstance(atom, BetweenExpr):
            if not self._is_target_column(
                atom.expr, var_name, col_name, qualifier_filter
            ):
                return False
            lower_lit = self._to_literal(atom.lower)
            upper_lit = self._to_literal(atom.upper)
            if lower_lit is None or upper_lit is None:
                return False
            constraint.apply_lower(lower_lit, True)
            constraint.apply_upper(upper_lit, True)
            return True

        if not isinstance(atom, BinaryExpr):
            return False
        if atom.op not in (">", ">=", "<", "<=", "="):
            return False

        op = atom.op
        left = atom.left
        right = atom.right
        if not self._is_target_column(left, var_name, col_name, qualifier_filter):
            if not self._is_target_column(right, var_name, col_name, qualifier_filter):
                return False
            reversed_op = self._reverse_op(op)
            if reversed_op is None:
                return False
            op = reversed_op
            left = atom.right
            right = atom.left

        lit = self._to_literal(right)
        if lit is None:
            return False

        if op == ">":
            constraint.apply_lower(lit, False)
        elif op == ">=":
            constraint.apply_lower(lit, True)
        elif op == "<":
            constraint.apply_upper(lit, False)
        elif op == "<=":
            constraint.apply_upper(lit, True)
        elif op == "=":
            constraint.apply_lower(lit, True)
            constraint.apply_upper(lit, True)
        else:
            return False

        return True

    def _prune_define_conditions(self, ast: RPRAbstractSyntaxTree, remove_specs):
        if remove_specs is None:
            return

        for remove_spec in remove_specs:
            if len(remove_spec) == 5:
                source_name, col_name, bound_kind, lit, inclusive = remove_spec
                qualifier_filter = None
            else:
                source_name, col_name, bound_kind, lit, inclusive, qualifier_filter = (
                    remove_spec
                )
            condition = ast.define.definitions.get(source_name)
            if condition is None:
                continue
            new_expr = self._remove_bound_from_expr(
                condition.expression,
                source_name,
                col_name,
                bound_kind,
                lit,
                inclusive,
                qualifier_filter,
            )
            condition.expression = new_expr

    def _ensure_non_empty_define(self, ast: RPRAbstractSyntaxTree):
        if len(ast.define.definitions) > 0:
            return

        leftmost_vars = ast.pattern.get_leftmost_variables()
        if leftmost_vars is None or len(leftmost_vars) == 0:
            return

        leftmost_var = sorted(leftmost_vars)[0]
        ast.define.definitions[leftmost_var] = IndependentCondition(
            leftmost_var, BinaryExpr(Literal(1), "=", Literal(1))
        )

    def _is_true_expr(self, expr: Expr) -> bool:
        if not isinstance(expr, BinaryExpr):
            return False
        if expr.op != "=":
            return False
        if not isinstance(expr.left, Literal) or not isinstance(expr.right, Literal):
            return False
        return expr.left.value == 1 and expr.right.value == 1

    def _remove_bound_from_expr(
        self,
        expr: Expr,
        var_name: str,
        col_name: str,
        bound_kind: str,
        lit: Literal,
        inclusive: bool,
        qualifier_filter: Optional[Set[str]] = None,
    ) -> Expr:
        conjuncts: List[Expr] = []
        if not self._flatten_and(expr, conjuncts):
            return expr

        kept: List[Expr] = []
        removed = False
        for atom in conjuncts:
            if removed:
                kept.append(atom)
                continue

            new_atom, atom_removed = self._remove_bound_from_atom(
                atom, var_name, col_name, bound_kind, lit, inclusive, qualifier_filter
            )
            if atom_removed:
                removed = True
                if new_atom is not None:
                    kept.append(new_atom)
            else:
                kept.append(atom)

        if len(kept) == 0:
            return BinaryExpr(Literal(1), "=", Literal(1))

        merged = kept[0]
        for i in range(1, len(kept)):
            merged = BinaryExpr(merged, "AND", kept[i])
        return merged

    def _remove_bound_from_atom(
        self,
        atom: Expr,
        var_name: str,
        col_name: str,
        bound_kind: str,
        lit: Literal,
        inclusive: bool,
        qualifier_filter: Optional[Set[str]] = None,
    ):
        if isinstance(atom, SemanticsExpr):
            new_inner, removed = self._remove_bound_from_atom(
                atom.expr,
                var_name,
                col_name,
                bound_kind,
                lit,
                inclusive,
                qualifier_filter,
            )
            if not removed:
                return atom, False
            if new_inner is None:
                return None, True
            return SemanticsExpr(atom.semantics, new_inner), True

        if isinstance(atom, BetweenExpr):
            if not self._is_target_column(
                atom.expr, var_name, col_name, qualifier_filter
            ):
                return atom, False

            lower_lit = self._to_literal(atom.lower)
            upper_lit = self._to_literal(atom.upper)
            if lower_lit is None or upper_lit is None:
                return atom, False

            if bound_kind == "lower":
                cmp_result = _compare_literal_values(lower_lit.value, lit.value)
                if cmp_result is None or cmp_result != 0 or not inclusive:
                    return atom, False
                return BinaryExpr(atom.expr, "<=", upper_lit), True

            cmp_result = _compare_literal_values(upper_lit.value, lit.value)
            if cmp_result is None or cmp_result != 0 or not inclusive:
                return atom, False
            return BinaryExpr(atom.expr, ">=", lower_lit), True

        if self._atom_matches_bound(
            atom, var_name, col_name, bound_kind, lit, inclusive, qualifier_filter
        ):
            return None, True

        return atom, False

    def _atom_matches_bound(
        self,
        atom: Expr,
        var_name: str,
        col_name: str,
        bound_kind: str,
        lit: Literal,
        inclusive: bool,
        qualifier_filter: Optional[Set[str]] = None,
    ) -> bool:
        if isinstance(atom, SemanticsExpr):
            return self._atom_matches_bound(
                atom.expr,
                var_name,
                col_name,
                bound_kind,
                lit,
                inclusive,
                qualifier_filter,
            )

        if not isinstance(atom, BinaryExpr):
            return False
        # Keep equality predicates intact for readability and safety.  Although
        # ``x = c`` implies both ``x >= c`` and ``x <= c``, bound pruning is
        # performed one side at a time.  Replacing equality with a residual
        # inequality obscures the original intent, while deleting it can let a
        # UNION-tail row become the endpoint variable.  The input filter still
        # receives the equality-derived bounds; only DEFINE pruning skips it.
        if atom.op not in (">", ">=", "<", "<="):
            return False

        op = atom.op
        left = atom.left
        right = atom.right

        if not self._is_target_column(left, var_name, col_name, qualifier_filter):
            if not self._is_target_column(right, var_name, col_name, qualifier_filter):
                return False
            reversed_op = self._reverse_op(op)
            if reversed_op is None:
                return False
            op = reversed_op
            right = atom.left

        right_lit = self._to_literal(right)
        if right_lit is None:
            return False

        cmp_result = _compare_literal_values(right_lit.value, lit.value)
        if cmp_result is None or cmp_result != 0:
            return False

        target_ops = []
        if bound_kind == "lower":
            target_ops = [">="] if inclusive else [">"]
        else:
            target_ops = ["<="] if inclusive else ["<"]

        return op in target_ops


class OrderedRangePushdownRule(OrderedRangePushdownSupport, RewriteRule):
    """Push DEFINE ranges on endpoint or guaranteed middle variables.

    Endpoint and middle-variable rewrites share one validation/execution
    pipeline.  Their pattern-specific candidate analysis remains separated so
    the more involved endpoint/SUBSET behavior can evolve independently.
    """

    @property
    def name(self) -> str:
        return "ordered_range_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_ordered_range_pushdown_applied", False):
            return False
        return self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_ordered_range_pushdown_applied", False):
            return False
        plan = self._build_plan(ast)
        if plan is None:
            return False
        if isinstance(plan, OrderedRangeCompositePlan):
            return self._apply_ordered_range_composite_plan(ast, plan)
        changed = self._apply_ordered_range_plan(ast, plan)
        if changed:
            ast._ordered_range_pushdown_applied = True
        return changed

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        """Build endpoint, middle, and compatible composite candidates."""

        if getattr(ast, "_ordered_range_pushdown_applied", False):
            return None

        endpoint_plan = self._build_endpoint_plan(ast)
        middle_plans = self._build_middle_plans(ast)

        if endpoint_plan is not None and middle_plans:
            context = self._prepare_order_range_context(ast)
            if context is not None:
                # Middle candidates are deterministic by variable name.  One
                # middle reducer is composed for now; composing multiple
                # middle witnesses requires a separate cost model because it
                # can multiply Top-K branches.
                for middle_plan in middle_plans:
                    composite = self._compose_endpoint_middle_plan(
                        endpoint_plan,
                        middle_plan,
                        context["order_col"],
                        context["domain"],
                    )
                    if composite is not None:
                        return composite

        if endpoint_plan is not None:
            return endpoint_plan
        return middle_plans[0] if middle_plans else None

    def _build_middle_plan(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[OrderedRangePushdownPlan]:
        plans = self._build_middle_plans(ast)
        return plans[0] if plans else None

    def _build_middle_plans(
        self, ast: RPRAbstractSyntaxTree
    ) -> List[OrderedRangePushdownPlan]:
        """Return every independently valid middle-variable candidate."""

        context = self._prepare_order_range_context(ast)
        if context is None:
            return []
        order_col = context["order_col"]
        if not self._order_range_input_is_safe(ast, order_col):
            return []

        middle_vars = ast.pattern.get_must_middle_var()
        if not middle_vars:
            return []

        plans = []
        for var_name in sorted(middle_vars):
            legacy_plan = self._build_plan_for_middle_var(
                ast,
                order_col,
                context["sort_type"],
                var_name,
                context["domain"],
            )
            if legacy_plan is None:
                continue
            if not self._anchor_filter_is_safe(
                ast,
                legacy_plan["keep_expr"],
                order_col,
            ):
                continue
            plans.append(
                self._to_ordered_range_plan(
                    legacy_plan,
                    context["input_rebind_alias"],
                    "middle",
                )
            )
        return plans

    def _build_plan_for_middle_var(
        self,
        ast: RPRAbstractSyntaxTree,
        order_col: str,
        sort_type,
        var_name: str,
        domain: RangeDomain,
    ):
        """Build a UNION ALL plan for a must-appear middle variable."""
        source_constraints, _ = self._collect_source_constraints(
            ast,
            order_col,
            {var_name},
            domain,
        )
        constraint = source_constraints.get(var_name)
        if constraint is None or not constraint.has_bound():
            return None
        if not constraint.is_satisfiable():
            return None

        keep_expr = constraint.to_expr(order_col)
        if keep_expr is None:
            return None

        tail_exprs = []

        if constraint.lower is not None:
            tail_dir = "left" if sort_type == SortType.ASC else "right"
            tail_limit = self._estimate_middle_tail_limit(
                ast,
                var_name,
                tail_dir,
            )
            if tail_limit is None:
                return None
            if tail_limit > 0:
                tail_expr = self._tail_expr(
                    order_col,
                    constraint.lower,
                    "lower",
                    constraint.lower_inclusive,
                )
                if tail_expr is None:
                    return None
                tail_exprs.append(
                    (
                        tail_expr,
                        self._build_tail_order_sql(ast, "lower"),
                        tail_limit,
                    )
                )

        if constraint.upper is not None:
            tail_dir = "right" if sort_type == SortType.ASC else "left"
            tail_limit = self._estimate_middle_tail_limit(
                ast,
                var_name,
                tail_dir,
            )
            if tail_limit is None:
                return None
            if tail_limit > 0:
                tail_expr = self._tail_expr(
                    order_col,
                    constraint.upper,
                    "upper",
                    constraint.upper_inclusive,
                )
                if tail_expr is None:
                    return None
                tail_exprs.append(
                    (
                        tail_expr,
                        self._build_tail_order_sql(ast, "upper"),
                        tail_limit,
                    )
                )

        # A middle-variable filter must retain at least one surrounding tail.
        # Otherwise the endpoint planner is the simpler representation.
        if not tail_exprs:
            return None

        return {
            "mode": "union",
            "col_name": order_col,
            "keep_expr": keep_expr,
            "tail_exprs": tail_exprs,
        }

    def _estimate_middle_tail_limit(
        self,
        ast: RPRAbstractSyntaxTree,
        var_name: str,
        tail_dir: str,
    ) -> Optional[int]:
        """Return required rows outside a middle-variable range boundary."""
        if tail_dir == "left":
            pattern_rows = ast.pattern.get_left_max_row_num(var_name)
        else:
            pattern_rows = ast.pattern.get_right_max_row_num(var_name)
        if pattern_rows is None:
            return None

        nav_extra = self._estimate_all_vars_nav_extra(
            ast,
            var_name,
            tail_dir,
        )
        if nav_extra is None:
            return None

        target_func = "PREV" if tail_dir == "left" else "NEXT"
        measure_offset = 0
        for expr in _iter_measure_expressions(ast):
            offset = max_navigation_offset(
                collect_expression_navigation(expr, None, target_func),
                target_func,
            )
            if offset is None:
                return None
            measure_offset = max(measure_offset, offset)

        return max(pattern_rows, nav_extra, pattern_rows + measure_offset)

    def _estimate_all_vars_nav_extra(
        self,
        ast: RPRAbstractSyntaxTree,
        var_name: str,
        tail_dir: str,
    ) -> Optional[int]:
        """Account for DEFINE navigation relative to a middle witness."""
        return PatternNavigationReach(ast).max_define_reach_from_variable(
            var_name,
            tail_dir,
        )


_endpoint_ordered_range_pushdown_test_cases = [
    # Test case 1: Range pushdown: reverse order + pattern with PREV()
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(C B A)
    DEFINE C AS VOL < 200, B AS PREV(B.VOL, 5) < 100, A AS A.TS > 100
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE TS > 100)
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(C B A)
    DEFINE C AS VOL < 200, B AS PREV(B.VOL, 5) < 100, A AS 1 = 1
)""",
    ),
    # Test case 2: Complex pattern matching
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, B AS PREV(FIRST(B.VOL), 5) < 100, C AS NEXT(C.OPEN, 10) < 100, D AS B.TS < 200
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS > 100) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 100) WHERE RPR_TAIL_RN <= 4))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, B AS PREV(FIRST(B.VOL), 5) < 100, C AS NEXT(C.OPEN, 10) < 100, D AS B.TS < 200
)""",
    ),
    # Test case 3
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, B AS PREV(B.VOL, 5) < 100, C AS NEXT(C.OPEN, 10) < 100, D AS B.TS < 200
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS > 100) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 100) WHERE RPR_TAIL_RN <= 4))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, B AS PREV(B.VOL, 5) < 100, C AS NEXT(C.OPEN, 10) < 100, D AS B.TS < 200
)""",
    ),
    # Test case 4
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, C AS PREV(C.VOL, 5) < 100, D AS B.TS < 200
)""",
        """
SELECT *
FROM ((SELECT PT, TS, ID, VOL FROM TA WHERE TS > 100) UNION ALL (SELECT PT, TS, ID, VOL FROM (SELECT PT, TS, ID, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 100) WHERE RPR_TAIL_RN <= 4))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B* C D)
    DEFINE A AS A.TS > 100, C AS PREV(C.VOL, 5) < 100, D AS B.TS < 200
)""",
    ),
    # Test case 5
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS A.TS > 100, B AS PREV(B.VOL, 5) < 100, C AS B.TS < 200
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, VOL FROM TA WHERE TS > 100) UNION ALL (SELECT PT, TS, ID, VOL FROM (SELECT PT, TS, ID, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 100) WHERE RPR_TAIL_RN <= 4))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS A.TS > 100, B AS PREV(B.VOL, 5) < 100, C AS B.TS < 200
)""",
    ),
    # Test case 6
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TS > 100 AND PREV(A.VOL, 5) < 100, B AS B.TS < 200 AND NEXT(B.VALUE, 0) < 10
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, VALUE, VOL FROM TA WHERE (TS > 100 AND TS < 200)) UNION ALL (SELECT PT, TS, ID, VALUE, VOL FROM (SELECT PT, TS, ID, VALUE, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 100) WHERE RPR_TAIL_RN <= 5))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TS > 100 AND PREV(A.VOL, 5) < 100, B AS B.VALUE < 10
)""",
    ),
    # Test case 7
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A{2,5} B)
    DEFINE A AS A.TS <= 200, B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, TYPE FROM TA WHERE TS <= 200) UNION ALL (SELECT PT, TS, ID, TYPE FROM (SELECT PT, TS, ID, TYPE, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS ASC) AS RPR_TAIL_RN FROM TA WHERE TS > 200) WHERE RPR_TAIL_RN <= 1))
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A{2,5} B)
    DEFINE A AS A.TS <= 200, B AS B.TYPE = 'B'
)""",
    ),
]


_middle_ordered_range_pushdown_test_cases = [
    # Test 1: Basic lower bound on middle var (user example)
    # B is middle var with TS > 10; A needs left tail (1 row), ASC order
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS OPEN > 10, B AS TS > 10, C AS VOL < 100
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS > 10) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 10) WHERE RPR_TAIL_RN <= 1)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS OPEN > 10, B AS TS > 10, C AS VOL < 100
)""",
    ),
    # Test 2: A former middle-rule negative case is handled by the endpoint planner.
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS OPEN > 10, B AS TS > 5
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN FROM TA WHERE TS > 5) UNION ALL (SELECT PT, TS, ID, OPEN FROM (SELECT PT, TS, ID, OPEN, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 5) WHERE RPR_TAIL_RN <= 1)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS OPEN > 10, B AS TS > 5
)""",
    ),
    # Test 3: Middle var with upper bound (right tail needed, ASC order)
    # B is middle with TS < 100; C may exceed 100, needs right tail
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS VOL > 5, B AS TS < 100, C AS OPEN > 3
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS < 100) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS ASC) AS RPR_TAIL_RN FROM TA WHERE TS >= 100) WHERE RPR_TAIL_RN <= 1)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS VOL > 5, B AS TS < 100, C AS OPEN > 3
)""",
    ),
    # Test 4: Middle var with both lower and upper bounds on TS (two tails)
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS VOL > 5, B AS TS > 10 AND TS < 100, C AS OPEN > 3
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE (TS BETWEEN 11 AND 99)) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS < 11) WHERE RPR_TAIL_RN <= 1) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS ASC) AS RPR_TAIL_RN FROM TA WHERE TS > 99) WHERE RPR_TAIL_RN <= 1)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS VOL > 5, B AS TS BETWEEN 11 AND 99, C AS OPEN > 3
)""",
    ),
    # Test 5: Middle var has constraint on non-ORDER-BY column - rule does NOT apply
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 10, B AS VOL > 50, C AS CLOSE < 200
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 10, B AS VOL > 50, C AS CLOSE < 200
)""",
    ),
    # Test 6: Middle var with PREV navigation (PREV(B.VOL, 3) needs extra left tail rows)
    # pattern: A B C, B is middle with TS > 10 and PREV(B.VOL, 3)
    # PREV(B, k=3): base_var=B, dist_from_B=0, tail_rows = max(0, 3-0) = 3
    # tail_limit = max(pattern_rows=1, nav_extra=3) = 3
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 5, B AS TS > 10 AND PREV(B.VOL, 3) > 50, C AS VOL < 200
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS > 10) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS DESC) AS RPR_TAIL_RN FROM TA WHERE TS <= 10) WHERE RPR_TAIL_RN <= 3)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 5, B AS TS > 10 AND PREV(B.VOL, 3) > 50, C AS VOL < 200
)""",
    ),
    # Test 7: Middle var with NEXT navigation (NEXT(B.VOL, 2) needs extra right tail rows)
    # B is middle with TS < 100 and NEXT(B.VOL, 2)
    # NEXT(B, k=2): base_var=B, dist_from_B=0, tail_rows = max(0, 0+2) = 2
    # tail_limit = max(pattern_rows=1, nav_extra=2) = 2
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 5, B AS TS < 100 AND NEXT(B.VOL, 2) < 200, C AS VOL > 10
)""",
        """
SELECT * FROM ((SELECT PT, TS, ID, OPEN, VOL FROM TA WHERE TS < 100) UNION ALL (SELECT PT, TS, ID, OPEN, VOL FROM (SELECT PT, TS, ID, OPEN, VOL, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS ASC) AS RPR_TAIL_RN FROM TA WHERE TS >= 100) WHERE RPR_TAIL_RN <= 2)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS OPEN > 5, B AS TS < 100 AND NEXT(B.VOL, 2) < 200, C AS VOL > 10
)""",
    ),
    # Test 8: Unbounded left (A+ before B) - rule does NOT apply because get_left_max_row_num = None
    (
        8,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ B C)
    DEFINE A AS OPEN > 5, B AS TS > 10, C AS VOL < 200
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ B C)
    DEFINE A AS OPEN > 5, B AS TS > 10, C AS VOL < 200
)""",
    ), 
]


ordered_range_pushdown_test_cases = [
    *(
        (f"endpoint-{test_num}", original_sql, expected_sql)
        for test_num, original_sql, expected_sql in _endpoint_ordered_range_pushdown_test_cases
    ),
    *(
        (f"middle-{test_num}", original_sql, expected_sql)
        for test_num, original_sql, expected_sql in _middle_ordered_range_pushdown_test_cases
    ),
]


default_ordered_range_pushdown_table_schemas = {
    "TA": {
        "columns": [
            "PT",
            "TS",
            "ID",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOL",
            "VALUE",
            "TYPE",
            "C1",
            "C2",
        ],
        "column_types": {"TS": "LONG"},
        "column_nullables": {"TS": False},
    },
    "NASDAQ": {
        "columns": ["TICKER", "TIMESTAMP", "OPEN", "HIGH"],
        "column_types": {"TIMESTAMP": "LONG"},
        "column_nullables": {"TIMESTAMP": False},
    },
    "TN": {
        "columns": ["TS", "ID", "TYPE"],
        "column_types": {"TS": "LONG"},
        "column_nullables": {"TS": True},
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        OrderedRangePushdownRule,
        ordered_range_pushdown_test_cases,
        "ordered range pushdown",
        verbose,
        default_ordered_range_pushdown_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
