if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import re

from .utils import *
from .utils import _walk_expression


class ColumnPruningRule(RewriteRule):
    """Replace DEFINE predicates with compact row-classification markers.

    Only ONE ROW PER MATCH can trigger this rule.

    The first planning phase keeps the original whole-predicate/AND-subset
    optimization.  A second expression-level phase lifts unary predicates
    from OR expressions, navigation functions, and variable-qualified column
    references.  Pairwise-disjoint predicates share a marker; overlapping
    predicates are assigned to separate markers.  A ONE ROW PER MATCH rewrite
    is emitted only when at least one MATCH_RECOGNIZE input column disappears.
    """

    @property
    def name(self) -> str:
        return "column_pruning"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        # The rewrite engine calls is_applicable() and apply() back-to-back on
        # the same rule instance.  Planning is the expensive part of this
        # rule, so retain the result for that single apply() call.
        self._pending_plan = (id(ast), plan)
        return plan is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        pending = getattr(self, "_pending_plan", None)
        plan = (
            pending[1]
            if pending is not None and pending[0] == id(ast)
            else self._build_plan(ast)
        )
        self._pending_plan = None
        if plan is None:
            return False

        ast.input_table = plan["input_table"]
        ast.partition_by = plan["partition_by"]
        ast.match_order_by = plan["match_order_by"]
        ast.define.definitions = plan["definitions"]
        return True

    def _build_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        extra_projection_cols: Optional[List[str]] = None,
        extra_projection_items: Optional[List[str]] = None,
    ):
        if (
            ast.rows_per_match is None
            or ast.rows_per_match.strip().upper() != "ONE ROW PER MATCH"
        ):
            return None
        pattern_vars = self._collect_pattern_vars_in_order(ast.pattern)
        var_options: Dict[str, List[dict]] = {}

        for var_name in pattern_vars:
            condition = ast.define.definitions.get(var_name)
            if condition is None:
                continue

            # A navigation function can produce NULL because its target row
            # does not exist, even when the underlying input column is NOT
            # NULL.  Preclassifying ``PREV(col) IS NULL`` as ``col IS NULL``
            # and then navigating the marker changes that boundary TRUE into
            # UNKNOWN.  Decline the whole rewrite rather than exchange the
            # navigation with this boundary-sensitive null test.
            if self._has_navigation_is_null(condition.expression):
                return None

            dependent_residual_expr = self._extract_residual_expr(condition)
            independent_expr = condition.get_independent_filter()

            if independent_expr is None:
                if dependent_residual_expr is not None:
                    continue
                independent_expr = BinaryExpr(Literal(1), "=", Literal(1))

            if self._is_effectively_true_for_input(ast, independent_expr):
                continue

            options = self._build_preclass_options(
                var_name, independent_expr, dependent_residual_expr
            )
            if len(options) == 0:
                return None
            var_options[var_name] = options

        input_column_refs = self._collect_input_column_refs(ast)
        if input_column_refs is None:
            return None

        marker_col = self._generate_marker_col_name(ast)
        overlap_cache: Dict[Tuple[str, str], bool] = {}
        plan = self._choose_best_preclass_plan(
            ast,
            pattern_vars,
            var_options,
            marker_col,
            overlap_cache,
        )
        if plan is None:
            plan = {
                "selected_options": {},
                "case_items": [],
                "tag_map": {},
                "score": 0,
            }

        selected_options = plan["selected_options"]
        tag_map = plan["tag_map"]

        new_definitions = {}

        for var_name, condition in ast.define.definitions.items():
            if var_name not in selected_options:
                new_definitions[var_name] = condition
                continue

            marker_expr = BinaryExpr(
                ColumnRef(None, marker_col), "=", Literal(tag_map[var_name])
            )
            residual_expr = selected_options[var_name]["residual_expr"]
            rewritten_expr = (
                marker_expr
                if residual_expr is None
                else BinaryExpr(marker_expr, "AND", residual_expr)
            )

            if isinstance(condition, DependentCondition):
                new_definitions[var_name] = DependentCondition(
                    var_name, rewritten_expr, condition.reasons
                )
            else:
                new_definitions[var_name] = IndependentCondition(
                    var_name, rewritten_expr
                )

        new_definitions, marker_groups = self._extend_with_lifted_atoms(
            ast,
            new_definitions,
            marker_col,
            plan["case_items"],
            set(var.upper() for var in pattern_vars),
            overlap_cache,
        )
        if hasattr(self, "_last_search_stats"):
            self._last_search_stats["overlap_cache_entries"] = len(overlap_cache)
        marker_cols = {group["marker_col"].upper() for group in marker_groups}
        define_projection_cols = self._collect_definition_columns(
            new_definitions, marker_cols, set(var.upper() for var in pattern_vars)
        )
        projection_cols = self._build_projection_columns(
            ast, define_projection_cols, marker_cols
        )
        if extra_projection_cols is not None:
            projection_cols = self._prepend_projection_columns(
                projection_cols, extra_projection_cols, marker_cols
            )

        original_cols = self._collect_original_required_columns(
            ast,
            set(var.upper() for var in pattern_vars),
        )
        projected_set = {col.upper() for col in projection_cols}
        if len(original_cols - projected_set) == 0 and extra_projection_cols is None:
            return None

        select_items = self._build_select_items(
            projection_cols,
            extra_projection_cols,
            extra_projection_items,
            input_column_refs,
        )
        for group in marker_groups:
            case_expr = CaseExpr(
                [(item["expr"], Literal(item["tag"])) for item in group["case_items"]],
                Literal(self._index_to_tag(len(group["case_items"]))),
            )
            select_items.append(f"{case_expr.to_string()} AS {group['marker_col']}")
        input_table_sql = f"(SELECT {', '.join(select_items)} FROM {ast.input_table})"

        return {
            "input_table": input_table_sql,
            "partition_by": self._rewrite_partition_by(ast.partition_by),
            "match_order_by": self._rewrite_match_order_by(ast.match_order_by),
            "definitions": new_definitions,
            "projected_columns": projection_cols,
        }

    def _build_select_items(
        self,
        projection_cols: List[str],
        extra_projection_cols: Optional[List[str]],
        extra_projection_items: Optional[List[str]],
        input_column_refs: Dict[str, str],
    ) -> List[str]:
        if extra_projection_items is None:
            return [
                self._render_projection_column(col, input_column_refs)
                for col in projection_cols
            ]

        extra_col_set = set()
        if extra_projection_cols is not None:
            extra_col_set = {
                norm
                for norm in (
                    self._normalize_column_name(col) for col in extra_projection_cols
                )
                if norm is not None
            }
        result = list(extra_projection_items)
        result.extend(
            self._render_projection_column(col, input_column_refs)
            for col in projection_cols
            if col.upper() not in extra_col_set
        )
        return result

    def _render_projection_column(
        self, col_name: str, input_column_refs: Dict[str, str]
    ) -> str:
        source_ref = input_column_refs.get(col_name.upper())
        if source_ref is None or source_ref.upper() == col_name.upper():
            return col_name
        # Oracle permits AS for a select-list alias (but not for a table alias).
        return f"{source_ref} AS {col_name}"

    def _collect_input_column_refs(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[Dict[str, str]]:
        """
        Preserve source qualifiers while crossing the derived-table boundary.

        T1.P must be selected as ``T1.P AS P`` and subsequently referenced as
        ``P``.  If two distinct qualified references collapse to the same
        output name, declining the rewrite is safer than silently changing
        which source column MATCH_RECOGNIZE consumes.
        """
        refs: Dict[str, str] = {}

        def add_ref(col_name: str) -> bool:
            raw = col_name.strip()
            norm = self._normalize_column_name(raw)
            if norm is None:
                return True
            existing = refs.get(norm)
            if existing is not None and existing.upper() != raw.upper():
                return False
            refs[norm] = raw
            return True

        for col_name in ast.partition_by or []:
            if not add_ref(col_name):
                return None
        for col_name, _ in ast.match_order_by or []:
            if not add_ref(col_name):
                return None
        return refs

    def _rewrite_partition_by(
        self, partition_by: Optional[List[str]]
    ) -> Optional[List[str]]:
        if partition_by is None:
            return None
        return [self._normalize_column_name(col) for col in partition_by]

    def _rewrite_match_order_by(self, match_order_by):
        if match_order_by is None:
            return None
        return [
            (self._normalize_column_name(col), sort_type)
            for col, sort_type in match_order_by
        ]

    def _prepend_projection_columns(
        self,
        projection_cols: List[str],
        extra_projection_cols: List[str],
        marker_cols,
    ) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for col in list(extra_projection_cols) + list(projection_cols):
            norm = self._normalize_column_name(col)
            if norm is None or norm in marker_cols or norm in seen:
                continue
            result.append(norm)
            seen.add(norm)
        return result

    def _build_preclass_options(
        self,
        var_name: str,
        independent_expr: Expr,
        dependent_residual_expr: Optional[Expr],
    ) -> List[dict]:
        conjuncts = self._split_and_conjuncts(independent_expr)
        if len(conjuncts) == 0:
            return []

        options = []
        seen_keys: Set[str] = set()
        max_mask = 1 << len(conjuncts)
        for mask in range(1, max_mask):
            selected_parts = [
                conjuncts[idx] for idx in range(len(conjuncts)) if mask & (1 << idx)
            ]
            remaining_parts = [
                conjuncts[idx] for idx in range(len(conjuncts)) if not mask & (1 << idx)
            ]
            selected_expr = self._combine_and(selected_parts)
            stripped_expr = self._strip_var_qualifier(selected_expr, var_name)
            if stripped_expr is None:
                continue

            key = stripped_expr.to_string()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            residual_expr = self._combine_and(
                remaining_parts
                + (
                    [dependent_residual_expr]
                    if dependent_residual_expr is not None
                    else []
                )
            )
            options.append(
                {
                    "var_name": var_name,
                    "expr": stripped_expr,
                    "key": key,
                    "residual_expr": residual_expr,
                    "conjunct_count": len(selected_parts),
                }
            )
        return options

    def _choose_best_preclass_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        pattern_vars: List[str],
        var_options: Dict[str, List[dict]],
        marker_col: str,
        overlap_cache: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> Optional[dict]:
        """Choose an exact best plan with branch-and-bound state pruning.

        A variable with ``n`` independent conjuncts has ``2**n - 1`` possible
        preclassification subsets plus the no-rewrite choice.  Multiplying
        those choices across all pattern variables made the old exhaustive
        search exponential even when most branches could not improve the
        current plan.  This search keeps the same metric and tie-breaking, but
        computes it from precomputed column sets, visits promising branches
        first, prunes branches whose optimistic metric cannot win, and merges
        dominated states with the same future compatibility constraints.
        """
        pattern_var_set = set(var.upper() for var in pattern_vars)
        original_cols = self._collect_original_required_columns(ast, pattern_var_set)
        col_sizes = self._get_column_avg_sizes(ast)
        best_plan = None
        best_metric = None
        best_tie_key = None

        if overlap_cache is None:
            overlap_cache = {}
        simplify_context = ast.expression_simplify_context()

        def expr_columns(expr: Optional[Expr]) -> Set[str]:
            if expr is None:
                return set()
            return {
                normalized
                for normalized in (
                    self._normalize_column_name(name) for name in expr.collect_columns()
                )
                if normalized is not None
            }

        hard_keep: Set[str] = set()
        hard_keep.update(self._normalize_column_sequence(ast.partition_by or []))
        hard_keep.update(
            self._normalize_column_sequence(
                [col for col, _ in (ast.match_order_by or [])]
            )
        )
        if ast.measures is not None:
            hard_keep.update(
                normalized
                for normalized in (
                    self._normalize_column_name(name)
                    for name in ast.measures.get_columns()
                )
                if normalized is not None
            )

        original_cols_by_var: Dict[str, Set[str]] = {}
        for var_name in pattern_vars:
            condition = ast.define.definitions.get(var_name)
            original_cols_by_var[var_name] = expr_columns(
                condition.expression if condition is not None else None
            )

        max_conjuncts_by_var: List[int] = []
        for var_name in pattern_vars:
            options = var_options.get(var_name) or []
            max_conjuncts_by_var.append(
                max((option["conjunct_count"] for option in options), default=0)
            )
            for choice_index, option in enumerate(options, start=1):
                option["_choice_index"] = choice_index
                option["_residual_cols"] = expr_columns(option["residual_expr"])
                option["_preclass_cols"] = expr_columns(option["expr"])

        remaining_selectable = [0] * (len(pattern_vars) + 1)
        remaining_conjuncts = [0] * (len(pattern_vars) + 1)
        for idx in range(len(pattern_vars) - 1, -1, -1):
            remaining_selectable[idx] = remaining_selectable[idx + 1] + int(
                bool(var_options.get(pattern_vars[idx]))
            )
            remaining_conjuncts[idx] = (
                remaining_conjuncts[idx + 1] + max_conjuncts_by_var[idx]
            )

        selected: Dict[str, dict] = {}
        selected_case_options: Dict[str, dict] = {}
        case_refcounts: Dict[str, int] = {}
        required_col_counts: Dict[str, int] = {}
        preclass_col_counts: Dict[str, int] = {}
        choice_path: List[int] = []
        dominance: Dict[Tuple[int, frozenset, frozenset], tuple] = {}
        stats = {
            "states_visited": 0,
            "leaves_evaluated": 0,
            "bound_prunes": 0,
            "dominated_prunes": 0,
            "overlap_cache_entries": 0,
        }

        def add_columns(counts: Dict[str, int], columns: Set[str]) -> None:
            for column in columns:
                counts[column] = counts.get(column, 0) + 1

        def remove_columns(counts: Dict[str, int], columns: Set[str]) -> None:
            for column in columns:
                remaining = counts[column] - 1
                if remaining == 0:
                    del counts[column]
                else:
                    counts[column] = remaining

        add_columns(required_col_counts, hard_keep)

        def plan_from_selection(score: float) -> dict:
            case_items = []
            case_key_to_tag = {}
            tag_map = {}
            for var_name in pattern_vars:
                option = selected.get(var_name)
                if option is None:
                    continue
                key = option["key"]
                if key not in case_key_to_tag:
                    tag = self._index_to_tag(len(case_items))
                    case_key_to_tag[key] = tag
                    case_items.append({"key": key, "expr": option["expr"], "tag": tag})
                tag_map[var_name] = case_key_to_tag[key]
            return {
                "selected_options": dict(selected),
                "case_items": case_items,
                "tag_map": tag_map,
                "score": score,
            }

        def current_score() -> float:
            return sum(
                self._column_size(col_sizes, col)
                for col in original_cols
                if col not in required_col_counts
            )

        def can_add_option(option: dict) -> bool:
            existing = selected_case_options.get(option["key"])
            if existing is not None:
                return True
            for case_option in selected_case_options.values():
                if self._has_cached_expr_overlap(
                    ast,
                    option["expr"],
                    case_option["expr"],
                    simplify_context,
                    overlap_cache,
                ):
                    return False
            return True

        def optimistic_metric(
            var_idx: int,
            selected_count: int,
            selected_conjunct_count: int,
        ) -> tuple:
            score = current_score()
            max_selected = selected_count + remaining_selectable[var_idx]
            max_conjuncts = selected_conjunct_count + remaining_conjuncts[var_idx]
            if len(col_sizes) > 0:
                current_preclass_weight = sum(
                    self._column_size(col_sizes, col) for col in preclass_col_counts
                )
                return (
                    score,
                    max_selected,
                    -current_preclass_weight,
                    max_conjuncts,
                )
            return (score, max_selected, max_conjuncts)

        def search(
            var_idx: int,
            selected_count: int,
            selected_conjunct_count: int,
        ) -> None:
            nonlocal best_plan, best_metric, best_tie_key
            stats["states_visited"] += 1

            upper_metric = optimistic_metric(
                var_idx, selected_count, selected_conjunct_count
            )
            if best_metric is not None and upper_metric < best_metric:
                stats["bound_prunes"] += 1
                return

            state_key = (
                var_idx,
                frozenset(selected_case_options),
                frozenset(required_col_counts),
            )
            prefix_tie_key = tuple(-choice for choice in choice_path)
            prefix_metric = (
                selected_count,
                selected_conjunct_count,
                prefix_tie_key,
            )
            previous = dominance.get(state_key)
            if previous is not None and previous >= prefix_metric:
                stats["dominated_prunes"] += 1
                return
            dominance[state_key] = prefix_metric

            if var_idx >= len(pattern_vars):
                if selected_count == 0:
                    return
                stats["leaves_evaluated"] += 1
                score = current_score()
                if len(col_sizes) > 0:
                    preclass_weight = sum(
                        self._column_size(col_sizes, col) for col in preclass_col_counts
                    )
                    metric = (
                        score,
                        selected_count,
                        -preclass_weight,
                        selected_conjunct_count,
                    )
                else:
                    metric = (score, selected_count, selected_conjunct_count)
                tie_key = tuple(-choice for choice in choice_path)
                if (
                    best_metric is None
                    or metric > best_metric
                    or (metric == best_metric and tie_key > best_tie_key)
                ):
                    best_metric = metric
                    best_tie_key = tie_key
                    best_plan = plan_from_selection(score)
                return

            var_name = pattern_vars[var_idx]
            options = var_options.get(var_name) or []

            # Full predicates normally remove the most columns.  Trying those
            # first establishes a strong score bound before exploring partial
            # subsets and the no-rewrite branch.  _choice_index retains the
            # old deterministic tie order independently of visit order.
            ordered_options = sorted(
                options,
                key=lambda option: (
                    sum(
                        self._column_size(col_sizes, col)
                        for col in option["_residual_cols"]
                    ),
                    -option["conjunct_count"],
                    option["_choice_index"],
                ),
            )
            for option in ordered_options:
                if not can_add_option(option):
                    continue
                selected[var_name] = option
                key = option["key"]
                is_new_case = key not in case_refcounts
                case_refcounts[key] = case_refcounts.get(key, 0) + 1
                if is_new_case:
                    selected_case_options[key] = option
                    add_columns(preclass_col_counts, option["_preclass_cols"])
                add_columns(required_col_counts, option["_residual_cols"])
                choice_path.append(option["_choice_index"])

                search(
                    var_idx + 1,
                    selected_count + 1,
                    selected_conjunct_count + option["conjunct_count"],
                )

                choice_path.pop()
                remove_columns(required_col_counts, option["_residual_cols"])
                del selected[var_name]
                case_refcounts[key] -= 1
                if case_refcounts[key] == 0:
                    del case_refcounts[key]
                    del selected_case_options[key]
                    remove_columns(preclass_col_counts, option["_preclass_cols"])

            skip_cols = original_cols_by_var[var_name]
            add_columns(required_col_counts, skip_cols)
            choice_path.append(0)
            search(var_idx + 1, selected_count, selected_conjunct_count)
            choice_path.pop()
            remove_columns(required_col_counts, skip_cols)

        search(0, 0, 0)
        stats["overlap_cache_entries"] = len(overlap_cache)
        self._last_search_stats = stats
        return best_plan

    def _has_cached_expr_overlap(
        self,
        ast: RPRAbstractSyntaxTree,
        left: Expr,
        right: Expr,
        simplify_context,
        overlap_cache: Dict[Tuple[str, str], bool],
    ) -> bool:
        """Return predicate overlap while evaluating each expression pair once."""
        left_key = left.to_string()
        right_key = right.to_string()
        cache_key = tuple(sorted((left_key, right_key)))
        if cache_key not in overlap_cache:
            overlap_cache[cache_key] = ast.define.has_expr_overlap(
                left,
                right,
                simplify_context,
            )
        return overlap_cache[cache_key]

    def _extend_with_lifted_atoms(
        self,
        ast: RPRAbstractSyntaxTree,
        definitions: Dict[str, object],
        marker_col: str,
        initial_case_items: List[dict],
        pattern_var_set: Set[str],
        overlap_cache: Optional[Dict[Tuple[str, str], bool]] = None,
    ):
        """Lift remaining unary row predicates into one or more marker columns.

        Unlike ``get_independent_filter()``, this pass works at expression-node
        granularity.  A comparison over one input column can be lifted even
        when it occurs below OR, PREV/NEXT/FIRST/LAST, or a reference to a
        different pattern variable.  The navigation/variable accessor is kept
        at the use site and redirected to the generated marker column.

        A normal CASE label is single-valued, so predicates that may overlap
        are placed in different groups.  Every group is therefore a set of
        pairwise-disjoint predicates; a singleton group is always valid.
        """
        hard_keep: Set[str] = set()
        hard_keep.update(self._normalize_column_sequence(ast.partition_by or []))
        hard_keep.update(
            self._normalize_column_sequence(
                [col for col, _ in (ast.match_order_by or [])]
            )
        )
        if ast.measures is not None:
            hard_keep.update(
                norm
                for norm in (
                    self._normalize_column_name(col)
                    for col in ast.measures.get_columns()
                )
                if norm is not None
            )

        generated_markers = {marker_col.upper()}
        eligible_cols: Set[str] = set()
        ineligible_cols: Set[str] = set()

        for var_name, condition in definitions.items():
            if var_name.upper() not in pattern_var_set or condition is None:
                continue
            self._classify_liftable_column_uses(
                condition.expression,
                var_name,
                generated_markers,
                eligible_cols,
                ineligible_cols,
            )

        target_cols = eligible_cols - ineligible_cols - hard_keep
        if len(target_cols) == 0:
            groups = []
            if initial_case_items:
                groups.append(
                    {
                        "marker_col": marker_col,
                        "case_items": [dict(item) for item in initial_case_items],
                    }
                )
            return definitions, groups

        groups: List[dict] = []
        if initial_case_items:
            groups.append(
                {
                    "marker_col": marker_col,
                    "case_items": [dict(item) for item in initial_case_items],
                }
            )

        predicate_locations: Dict[str, Tuple[str, str]] = {}
        for group in groups:
            for item in group["case_items"]:
                predicate_locations[item["key"].upper()] = (
                    group["marker_col"],
                    item["tag"],
                )

        used_marker_cols = {group["marker_col"].upper() for group in groups}
        simplify_context = ast.expression_simplify_context()
        if overlap_cache is None:
            overlap_cache = {}

        def allocate_marker(template: Expr, key: str) -> Tuple[str, str]:
            existing = predicate_locations.get(key)
            if existing is not None:
                return existing

            target_group = None
            for group in groups:
                if all(
                    not self._has_cached_expr_overlap(
                        ast,
                        template,
                        item["expr"],
                        simplify_context,
                        overlap_cache,
                    )
                    for item in group["case_items"]
                ):
                    target_group = group
                    break

            if target_group is None:
                new_marker = choose_generated_column_name(ast, "TAG", used_marker_cols)
                used_marker_cols.add(new_marker.upper())
                target_group = {"marker_col": new_marker, "case_items": []}
                groups.append(target_group)

            tag = self._index_to_tag(len(target_group["case_items"]))
            target_group["case_items"].append(
                {"key": key, "expr": template, "tag": tag}
            )
            predicate_locations[key] = (target_group["marker_col"], tag)
            return target_group["marker_col"], tag

        rewritten_definitions = {}
        for var_name, condition in definitions.items():
            if (
                var_name.upper() not in pattern_var_set
                or condition is None
                or condition.expression is None
            ):
                rewritten_definitions[var_name] = condition
                continue

            rewritten_expr = self._rewrite_liftable_atoms(
                condition.expression,
                var_name,
                target_cols,
                generated_markers,
                allocate_marker,
            )
            if isinstance(condition, DependentCondition):
                rewritten_definitions[var_name] = DependentCondition(
                    var_name,
                    rewritten_expr,
                    condition.reasons,
                )
            else:
                rewritten_definitions[var_name] = IndependentCondition(
                    var_name, rewritten_expr
                )

        return rewritten_definitions, groups

    def _classify_liftable_column_uses(
        self,
        expr: Optional[Expr],
        current_var: str,
        generated_markers: Set[str],
        eligible_cols: Set[str],
        ineligible_cols: Set[str],
        below_not: bool = False,
    ) -> None:
        if expr is None:
            return

        atom = None if below_not else self._liftable_unary_atom(expr, current_var)
        if atom is not None:
            col = atom["column"]
            if col not in generated_markers:
                eligible_cols.add(col)
            return

        if isinstance(expr, ColumnRef):
            col = self._normalize_column_name(expr.name)
            if col is not None and col not in generated_markers:
                ineligible_cols.add(col)
            return
        if isinstance(expr, BinaryExpr):
            self._classify_liftable_column_uses(
                expr.left,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
            self._classify_liftable_column_uses(
                expr.right,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
        elif isinstance(expr, BetweenExpr):
            self._classify_liftable_column_uses(
                expr.expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
            self._classify_liftable_column_uses(
                expr.lower,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
            self._classify_liftable_column_uses(
                expr.upper,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
        elif isinstance(expr, InExpr):
            self._classify_liftable_column_uses(
                expr.expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
            for value in expr.values:
                self._classify_liftable_column_uses(
                    value,
                    current_var,
                    generated_markers,
                    eligible_cols,
                    ineligible_cols,
                    below_not,
                )
        elif isinstance(expr, IsNullExpr):
            self._classify_liftable_column_uses(
                expr.expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
        elif isinstance(expr, NotExpr):
            # CASE WHEN collapses UNKNOWN to the default label.  Replacing a
            # predicate below NOT would turn NOT UNKNOWN into TRUE, so leave
            # such references untouched unless a future three-valued marker is
            # used.
            self._classify_liftable_column_uses(
                expr.expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                True,
            )
        elif isinstance(expr, SemanticsExpr):
            self._classify_liftable_column_uses(
                expr.expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
        elif isinstance(expr, CaseExpr):
            for cond, result in expr.cases:
                self._classify_liftable_column_uses(
                    cond,
                    current_var,
                    generated_markers,
                    eligible_cols,
                    ineligible_cols,
                    below_not,
                )
                self._classify_liftable_column_uses(
                    result,
                    current_var,
                    generated_markers,
                    eligible_cols,
                    ineligible_cols,
                    below_not,
                )
            self._classify_liftable_column_uses(
                expr.else_expr,
                current_var,
                generated_markers,
                eligible_cols,
                ineligible_cols,
                below_not,
            )
        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._classify_liftable_column_uses(
                    arg,
                    current_var,
                    generated_markers,
                    eligible_cols,
                    ineligible_cols,
                    below_not,
                )

    def _liftable_unary_atom(self, expr: Expr, current_var: str):
        """Return a base-row predicate and accessor for a unary SQL atom."""
        if isinstance(expr, BinaryExpr) and expr.op not in (
            "AND",
            "OR",
            "+",
            "-",
            "*",
            "/",
        ):
            left = self._extract_column_accessor(expr.left, current_var)
            right = self._extract_column_accessor(expr.right, current_var)
            if left is not None and len(expr.right.collect_columns()) == 0:
                template = BinaryExpr(
                    ColumnRef(None, left["column"]), expr.op, expr.right
                )
                return {
                    "column": left["column"],
                    "template": template,
                    "accessor": left["accessor"],
                }
            if right is not None and len(expr.left.collect_columns()) == 0:
                template = BinaryExpr(
                    expr.left, expr.op, ColumnRef(None, right["column"])
                )
                return {
                    "column": right["column"],
                    "template": template,
                    "accessor": right["accessor"],
                }
            return None

        if (
            isinstance(expr, BetweenExpr)
            and not expr.lower.collect_columns()
            and not expr.upper.collect_columns()
        ):
            accessed = self._extract_column_accessor(expr.expr, current_var)
            if accessed is not None:
                return {
                    "column": accessed["column"],
                    "template": BetweenExpr(
                        ColumnRef(None, accessed["column"]), expr.lower, expr.upper
                    ),
                    "accessor": accessed["accessor"],
                }

        if isinstance(expr, InExpr) and all(
            not value.collect_columns() for value in expr.values
        ):
            accessed = self._extract_column_accessor(expr.expr, current_var)
            if accessed is not None:
                return {
                    "column": accessed["column"],
                    "template": InExpr(
                        ColumnRef(None, accessed["column"]), expr.values, expr.is_not
                    ),
                    "accessor": accessed["accessor"],
                }

        if isinstance(expr, IsNullExpr):
            accessed = self._extract_column_accessor(expr.expr, current_var)
            if accessed is not None:
                return {
                    "column": accessed["column"],
                    "template": IsNullExpr(
                        ColumnRef(None, accessed["column"]), expr.is_not
                    ),
                    "accessor": accessed["accessor"],
                }
        return None

    def _has_navigation_is_null(self, expr: Optional[Expr]) -> bool:
        """Whether a positive IS NULL tests a navigation expression.

        PREV/NEXT may run past a partition boundary, while FIRST/LAST may
        have no qualifying row.  In either case the navigation itself creates
        NULL, which cannot be represented by a marker computed on input rows.
        IS NOT NULL does not have the same TRUE-at-the-boundary behavior.
        """
        navigation_names = {"PREV", "NEXT", "FIRST", "LAST"}
        for node in _walk_expression(expr):
            if not isinstance(node, IsNullExpr) or node.is_not:
                continue
            if any(
                isinstance(inner, FunctionCall)
                and inner.name.upper() in navigation_names
                for inner in _walk_expression(node.expr)
            ):
                return True
        return False

    def _extract_column_accessor(self, expr: Expr, current_var: str):
        if isinstance(expr, ColumnRef):
            qualifier = expr.qualifier

            def accessor(marker_col: str):
                marker_qualifier = qualifier
                if (
                    marker_qualifier is not None
                    and marker_qualifier.upper() == current_var.upper()
                ):
                    marker_qualifier = None
                return ColumnRef(marker_qualifier, marker_col)

            return {
                "column": self._normalize_column_name(expr.name),
                "accessor": accessor,
            }

        if (
            isinstance(expr, FunctionCall)
            and expr.name.upper() in ("PREV", "NEXT", "FIRST", "LAST")
            and len(expr.args) >= 1
            and all(len(arg.collect_columns()) == 0 for arg in expr.args[1:])
        ):
            inner = self._extract_column_accessor(expr.args[0], current_var)
            if inner is None:
                return None

            def accessor(marker_col: str, inner=inner, expr=expr):
                return FunctionCall(
                    expr.name,
                    [inner["accessor"](marker_col)] + list(expr.args[1:]),
                )

            return {"column": inner["column"], "accessor": accessor}
        return None

    def _rewrite_liftable_atoms(
        self,
        expr: Expr,
        current_var: str,
        target_cols: Set[str],
        generated_markers: Set[str],
        allocate_marker,
        below_not: bool = False,
    ) -> Expr:
        atom = None if below_not else self._liftable_unary_atom(expr, current_var)
        if (
            atom is not None
            and atom["column"] in target_cols
            and atom["column"] not in generated_markers
        ):
            key = atom["template"].to_string().upper()
            marker, tag = allocate_marker(atom["template"], key)
            return BinaryExpr(atom["accessor"](marker), "=", Literal(tag))

        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                self._rewrite_liftable_atoms(
                    expr.left,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
                expr.op,
                self._rewrite_liftable_atoms(
                    expr.right,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
            )
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(
                self._rewrite_liftable_atoms(
                    expr.expr,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
                self._rewrite_liftable_atoms(
                    expr.lower,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
                self._rewrite_liftable_atoms(
                    expr.upper,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
            )
        if isinstance(expr, InExpr):
            return InExpr(
                self._rewrite_liftable_atoms(
                    expr.expr,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
                [
                    self._rewrite_liftable_atoms(
                        value,
                        current_var,
                        target_cols,
                        generated_markers,
                        allocate_marker,
                        below_not,
                    )
                    for value in expr.values
                ],
                expr.is_not,
            )
        if isinstance(expr, IsNullExpr):
            return IsNullExpr(
                self._rewrite_liftable_atoms(
                    expr.expr,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
                expr.is_not,
            )
        if isinstance(expr, NotExpr):
            return NotExpr(
                self._rewrite_liftable_atoms(
                    expr.expr,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    True,
                )
            )
        if isinstance(expr, SemanticsExpr):
            return SemanticsExpr(
                expr.semantics,
                self._rewrite_liftable_atoms(
                    expr.expr,
                    current_var,
                    target_cols,
                    generated_markers,
                    allocate_marker,
                    below_not,
                ),
            )
        if isinstance(expr, CaseExpr):
            return CaseExpr(
                [
                    (
                        self._rewrite_liftable_atoms(
                            cond,
                            current_var,
                            target_cols,
                            generated_markers,
                            allocate_marker,
                            below_not,
                        ),
                        self._rewrite_liftable_atoms(
                            result,
                            current_var,
                            target_cols,
                            generated_markers,
                            allocate_marker,
                            below_not,
                        ),
                    )
                    for cond, result in expr.cases
                ],
                (
                    self._rewrite_liftable_atoms(
                        expr.else_expr,
                        current_var,
                        target_cols,
                        generated_markers,
                        allocate_marker,
                        below_not,
                    )
                    if expr.else_expr is not None
                    else None
                ),
            )
        if isinstance(expr, FunctionCall):
            return FunctionCall(
                expr.name,
                [
                    self._rewrite_liftable_atoms(
                        arg,
                        current_var,
                        target_cols,
                        generated_markers,
                        allocate_marker,
                        below_not,
                    )
                    for arg in expr.args
                ],
            )
        return expr

    def _split_and_conjuncts(self, expr: Optional[Expr]) -> List[Expr]:
        if expr is None:
            return []
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            return self._split_and_conjuncts(expr.left) + self._split_and_conjuncts(
                expr.right
            )
        return [expr]

    def _combine_and(self, parts: List[Optional[Expr]]) -> Optional[Expr]:
        filtered = [part for part in parts if part is not None]
        if len(filtered) == 0:
            return None
        result = filtered[0]
        for part in filtered[1:]:
            result = _simplify_logical_expr(result, "AND", part)
        return result

    def _collect_original_required_columns(
        self, ast: RPRAbstractSyntaxTree, pattern_var_set: Set[str]
    ) -> Set[str]:
        columns = set()
        if ast.partition_by is not None:
            columns.update(
                col for col in self._normalize_column_sequence(ast.partition_by)
            )
        if ast.match_order_by is not None:
            columns.update(
                col
                for col in self._normalize_column_sequence(
                    [col for col, _ in ast.match_order_by]
                )
            )
        if ast.measures is not None:
            columns.update(
                self._normalize_column_name(col) for col in ast.measures.get_columns()
            )
        for var_name, condition in ast.define.definitions.items():
            if (
                var_name.upper() not in pattern_var_set
                or condition is None
                or condition.expression is None
            ):
                continue
            columns.update(
                self._normalize_column_name(col)
                for col in condition.expression.collect_columns()
            )
        columns.discard(None)
        return columns

    def _collect_definition_columns(
        self,
        definitions: Dict[str, object],
        marker_cols: Set[str],
        pattern_var_set: Set[str],
    ) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for var_name, condition in definitions.items():
            if var_name.upper() not in pattern_var_set:
                continue
            if condition is None or condition.expression is None:
                continue
            for col_name in sorted(
                condition.expression.collect_columns(),
                key=lambda name: self._normalize_column_name(name) or "",
            ):
                norm_col = self._normalize_column_name(col_name)
                if norm_col is None or norm_col in marker_cols or norm_col in seen:
                    continue
                result.append(norm_col)
                seen.add(norm_col)
        return result

    def _get_column_avg_sizes(self, ast: RPRAbstractSyntaxTree) -> Dict[str, float]:
        if ast.table_schemas is None:
            return {}

        table_name = self._extract_single_base_table_name(ast.input_table)
        schema = None
        if table_name is not None:
            schema = ast.table_schemas.get(table_name.upper()) or ast.table_schemas.get(
                table_name
            )
        if schema is None and len(ast.table_schemas) == 1:
            schema = next(iter(ast.table_schemas.values()))
        if not isinstance(schema, dict):
            return {}

        result = {}
        for key in ("column_avg_sizes", "avg_column_sizes", "column_sizes"):
            sizes = schema.get(key)
            if isinstance(sizes, dict):
                result.update({col.upper(): float(size) for col, size in sizes.items()})
                break

        # When measured widths are unavailable, declared SQL types still give
        # the planner a substantially better projection-cost estimate than
        # treating every column as one byte. Explicit size statistics win.
        declared_type_columns = set()
        for key in ("column_types", "data_types", "types"):
            type_map = schema.get(key)
            if isinstance(type_map, dict):
                declared_type_columns.update(str(col).upper() for col in type_map)
        columns_meta = schema.get("columns", [])
        if isinstance(columns_meta, dict):
            declared_type_columns.update(str(col).upper() for col in columns_meta)
        else:
            for item in columns_meta or []:
                if not isinstance(item, dict):
                    continue
                if item.get("data_type", item.get("type")) is None:
                    continue
                name = item.get("name", item.get("column_name"))
                if name is not None:
                    declared_type_columns.add(str(name).upper())

        for column in declared_type_columns:
            if column in result:
                continue
            width = self._estimate_type_width(ast.get_column_type(column))
            if width is not None:
                result[column] = width
        return result

    @staticmethod
    def _estimate_type_width(data_type: Optional[str]) -> Optional[float]:
        if data_type is None:
            return None
        normalized = data_type.strip().upper()
        length_match = re.fullmatch(r"(?:CHAR|VARCHAR)\((\d+)\)", normalized)
        if length_match:
            return float(length_match.group(1))
        base = normalized.split("(")[0].strip()
        widths = {
            "INT": 4.0,
            "LONG": 8.0,
            "FLOAT": 4.0,
            "DOUBLE": 8.0,
        }
        return widths.get(base)

    def _column_size(self, col_sizes: Dict[str, float], col_name: str) -> float:
        return col_sizes.get(col_name.upper(), 1.0)

    def _normalize_column_sequence(self, columns: List[str]) -> List[str]:
        result = []
        seen = set()
        for col in columns:
            norm = self._normalize_column_name(col)
            if norm is None or norm in seen:
                continue
            seen.add(norm)
            result.append(norm)
        return result

    def _extract_single_base_table_name(self, input_table_sql: str) -> Optional[str]:
        if input_table_sql is None:
            return None
        sql = input_table_sql.strip()
        if len(sql) == 0 or sql.startswith("("):
            return None
        if any(token in sql.upper() for token in ("SELECT ", " JOIN ", " UNION ", ",")):
            return None
        parts = sql.split()
        if len(parts) in (1, 2):
            return parts[0].upper()
        if len(parts) == 3 and parts[1].upper() == "AS":
            return parts[0].upper()
        return None

    def _collect_pattern_vars_in_order(self, pattern) -> List[str]:
        ordered_vars: List[str] = []
        seen: Set[str] = set()

        def visit(node):
            if isinstance(node, VariablePattern):
                key = node.name.upper()
                if key not in seen:
                    seen.add(key)
                    ordered_vars.append(node.name)
                return
            if isinstance(node, QuantifiedPattern):
                visit(node.pattern)
                return
            if isinstance(
                node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
            ):
                for child in node.patterns:
                    visit(child)
                return
            if isinstance(node, (ExcludedPattern, AnchorPattern)):
                visit(node.pattern)

        visit(pattern)
        return ordered_vars

    def _extract_residual_expr(self, condition) -> Optional[Expr]:
        if condition is None or condition.expression is None:
            return None
        if isinstance(condition, IndependentCondition):
            return None

        conjuncts: List[Expr] = []

        def flatten_and(node: Expr):
            if isinstance(node, BinaryExpr) and node.op == "AND":
                flatten_and(node.left)
                flatten_and(node.right)
            else:
                conjuncts.append(node)

        flatten_and(condition.expression)

        residual_parts = [
            part for part in conjuncts if condition._is_subtree_dependent(part)
        ]
        if len(residual_parts) == 0:
            return None

        result = residual_parts[0]
        for part in residual_parts[1:]:
            result = _simplify_logical_expr(result, "AND", part)
        return result

    def _build_projection_columns(
        self,
        ast: RPRAbstractSyntaxTree,
        dependent_projection_cols: List[str],
        marker_cols: Set[str],
    ) -> List[str]:
        projection_cols: List[str] = []
        seen: Set[str] = set()

        def add_col(col_name: Optional[str]):
            norm = self._normalize_column_name(col_name)
            if norm is None or norm in marker_cols or norm in seen:
                return
            projection_cols.append(norm)
            seen.add(norm)

        if ast.partition_by is not None:
            for col_name in ast.partition_by:
                add_col(col_name)

        if ast.match_order_by is not None:
            for col_name, _ in ast.match_order_by:
                add_col(col_name)

        if ast.measures is not None:
            for col_name in sorted(ast.measures.get_columns()):
                add_col(col_name)

        for col_name in dependent_projection_cols:
            add_col(col_name)

        return projection_cols

    def _generate_marker_col_name(self, ast: RPRAbstractSyntaxTree) -> str:
        used_cols = set()

        if ast.partition_by is not None:
            for col_name in ast.partition_by:
                norm = self._normalize_column_name(col_name)
                if norm is not None:
                    used_cols.add(norm)

        if ast.match_order_by is not None:
            for col_name, _ in ast.match_order_by:
                norm = self._normalize_column_name(col_name)
                if norm is not None:
                    used_cols.add(norm)

        if ast.measures is not None:
            used_cols.update(col.upper() for col in ast.measures.get_columns())

        used_cols.update(col.upper() for col in ast.define.get_columns())

        return choose_generated_column_name(ast, "TAG", used_cols)

    def _index_to_tag(self, idx: int) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        value = idx
        tag = ""
        while True:
            tag = alphabet[value % 26] + tag
            value = value // 26 - 1
            if value < 0:
                break
        return tag

    def _normalize_column_name(self, col_name: Optional[str]) -> Optional[str]:
        if col_name is None:
            return None
        col_name = col_name.strip()
        if len(col_name) == 0:
            return None
        if "." in col_name:
            col_name = col_name.split(".")[-1]
        return col_name.upper()

    def _strip_var_qualifier(self, expr: Expr, var_name: str) -> Optional[Expr]:
        return _strip_var_qualifier(expr, var_name)

    def _is_trivially_true(self, expr: Optional[Expr]) -> bool:
        if expr is None:
            return True
        if isinstance(expr, SemanticsExpr):
            return self._is_trivially_true(expr.expr)
        if isinstance(expr, BinaryExpr):
            if (
                expr.op == "="
                and isinstance(expr.left, Literal)
                and isinstance(expr.right, Literal)
            ):
                return expr.left.value == expr.right.value
        return False

    def _is_effectively_true_for_input(
        self, ast: RPRAbstractSyntaxTree, expr: Optional[Expr]
    ) -> bool:
        if expr is None:
            return True
        if self._is_trivially_true(expr):
            return True
        if self._is_tautology_expr(expr):
            return True
        return self._input_table_implies_expr(ast, expr)

    def _is_tautology_expr(self, expr: Optional[Expr]) -> bool:
        if expr is None:
            return True
        helper = DefineSubClause({})
        return not helper._is_expr_satisfiable(NotExpr(expr))

    def _input_table_implies_expr(
        self, ast: RPRAbstractSyntaxTree, expr: Optional[Expr]
    ) -> bool:
        if expr is None:
            return True

        input_where = self._extract_input_where_expr(ast.input_table)
        if input_where is None:
            return False

        helper = DefineSubClause({})
        return not helper._is_expr_satisfiable(
            BinaryExpr(input_where, "AND", NotExpr(expr))
        )

    def _extract_input_where_expr(self, input_table_sql: str) -> Optional[Expr]:
        if input_table_sql is None:
            return None
        sql = input_table_sql.strip()
        if len(sql) < 2 or sql[0] != "(" or sql[-1] != ")":
            return None

        inner = sql[1:-1].strip()
        upper_inner = inner.upper()
        if not upper_inner.startswith("SELECT "):
            return None

        where_pos = self._find_top_level_keyword(upper_inner, "WHERE")
        if where_pos == -1:
            return None

        clause_end = len(inner)
        for keyword in ("GROUP BY", "HAVING", "ORDER BY", "LIMIT"):
            next_pos = self._find_top_level_keyword(upper_inner, keyword)
            if next_pos != -1 and next_pos > where_pos and next_pos < clause_end:
                clause_end = next_pos

        where_sql = inner[where_pos:clause_end].strip()
        if len(where_sql) == 0:
            return None

        try:
            return ExprParser().parse_where(where_sql)
        except Exception:
            return None

    def _find_top_level_keyword(self, sql: str, keyword: str) -> int:
        depth = 0
        idx = 0
        keyword_len = len(keyword)
        while idx <= len(sql) - keyword_len:
            ch = sql[idx]
            if ch == "(":
                depth += 1
                idx += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                idx += 1
                continue
            if ch == "'":
                idx += 1
                while idx < len(sql):
                    if sql[idx] == "'":
                        idx += 1
                        break
                    idx += 1
                continue

            if depth == 0 and sql.startswith(keyword, idx):
                before_ok = idx == 0 or not (
                    sql[idx - 1].isalnum() or sql[idx - 1] == "_"
                )
                after_idx = idx + keyword_len
                after_ok = after_idx == len(sql) or not (
                    sql[after_idx].isalnum() or sql[after_idx] == "_"
                )
                if before_ok and after_ok:
                    return idx
            idx += 1
        return -1

    def _expr_references_any_var(
        self, expr: Optional[Expr], var_names: Set[str]
    ) -> bool:
        return _expr_references_any_var(expr, var_names)


column_pruning_test_cases = [
    (
        1,
        """
SELECT *
FROM (SELECT * FROM T WHERE TS > 10)
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    PATTERN(A+ (B | C)+)
    DEFINE
        A AS TYPE = 'BIKE' AND VAL > 10,
        B AS TYPE = 'BIKE' AND VAL < 5 AND DIS = A.DIS,
        C AS VAL BETWEEN 6 AND 8 AND OPEN > 5
)""",
        """
SELECT *
FROM (SELECT PT, TS, ID, DIS, 
        CASE WHEN (TYPE = 'BIKE' AND VAL > 10) THEN 'a' 
        WHEN (TYPE = 'BIKE' AND VAL < 5) THEN 'b' 
        WHEN ((VAL BETWEEN 6 AND 8) AND OPEN > 5) THEN 'c' 
        ELSE 'd' END AS TAG FROM (SELECT * FROM T WHERE TS > 10))
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    PATTERN(A+ (B | C)+)
    DEFINE A AS TAG = 'a', B AS TAG = 'b' AND DIS = A.DIS, C AS TAG = 'c'
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A+ B+)
    DEFINE A AS VAL > 10, B AS VAL > 5
)""",
        """
SELECT * FROM (SELECT PT, TS, ID,
    CASE WHEN VAL > 5 THEN 'a' ELSE 'b' END AS TAG,
    CASE WHEN VAL > 10 THEN 'a' ELSE 'b' END AS TAG_1
FROM T) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID
    PATTERN(A+ B+)
    DEFINE A AS TAG_1 = 'a', B AS TAG = 'a'
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND OPEN = A.OPEN
)""",
        """
SELECT * FROM (SELECT TS, ID, OPEN, CASE WHEN TYPE = 'A' THEN 'a' WHEN TYPE = 'B' THEN 'b' ELSE 'c' END AS TAG FROM T) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b' AND OPEN = A.OPEN
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES LAST(U.ID) AS UID
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM (SELECT TS, ID, CASE WHEN TYPE = 'A' THEN 'a' WHEN TYPE = 'B' THEN 'b' WHEN TYPE = 'C' THEN 'c' ELSE 'd' END AS TAG FROM T) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES LAST(U.ID) AS UID
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b', C AS TAG = 'c'
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES C.ID AS CID
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C' AND SUM(U.VAL) > 10
)""",
        """
SELECT * FROM (SELECT TS, ID, VAL, CASE WHEN TYPE = 'A' THEN 'a' WHEN TYPE = 'B' THEN 'b' WHEN TYPE = 'C' THEN 'c' ELSE 'd' END AS TAG FROM T) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES C.ID AS CID
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b', C AS TAG = 'c' AND SUM(U.VAL) > 10
)""",
    ),
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES LAST(U.ID) AS UID
    PATTERN(A B)
    SUBSET U = (A, B)
    DEFINE A AS VAL > 10, B AS VAL > 5
)""",
        """
SELECT * FROM (SELECT TS, ID,
    CASE WHEN VAL > 5 THEN 'a' ELSE 'b' END AS TAG,
    CASE WHEN VAL > 10 THEN 'a' ELSE 'b' END AS TAG_1
FROM T) MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES LAST(U.ID) AS UID
    PATTERN(A B)
    SUBSET U = (A, B)
    DEFINE A AS TAG_1 = 'a', B AS TAG = 'a'
)""",
    ),
]


default_column_pruning_table_schemas = {
    "T": {
        "columns": ["PT", "TS", "ID", "TYPE", "OPEN", "VAL", "DIS", "PRICE"],
        "column_types": {
            "PT": "VARCHAR(32)",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(32)",
            "OPEN": "DOUBLE",
            "VAL": "DOUBLE",
            "DIS": "DOUBLE",
            "PRICE": "DOUBLE",
        },
        "column_avg_sizes": {"OPEN": 4, "TYPE": 10},
    },
    "T1": {
        "columns": ["P", "TS", "ID", "TYPE", "K"],
        "column_types": {
            "P": "VARCHAR(32)",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(32)",
            "K": "LONG",
        },
    },
    "T2": {
        "columns": ["K"],
        "column_types": {
            "K": "LONG",
        },
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        ColumnPruningRule,
        column_pruning_test_cases,
        "column pruning",
        verbose,
        default_column_pruning_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
