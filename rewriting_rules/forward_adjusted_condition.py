if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
import re
from typing import Dict, List, Optional, Set, Tuple

from clause_parser import ExprParser
from expression import (
    BetweenExpr,
    BinaryExpr,
    CaseExpr,
    ColumnRef,
    Expr,
    InExpr,
    FunctionCall,
    IsNullExpr,
    Literal,
    NotExpr,
    SemanticsExpr,
)
from expression_clause import DependentCondition
from expression_simplifier import _simplify_logical_expr
from patternnode import (
    EmptyPattern,
    AnchorPattern,
    ConcatenationPattern,
    QuantifiedPattern,
    AlternationPattern,
    ExcludedPattern,
    PermutationPattern,
    VariablePattern,
)
from queryanalysis import RPRAbstractSyntaxTree

from .base import RewriteRule
from .utils import _expr_references_any_var


class ForwardAdjustedConditionRule(RewriteRule):
    """
    Push a Y-only conjunct that appears inside X's DEFINE expression to an
    earlier variable Y when this preserves MATCH_RECOGNIZE running semantics.

    Supported rewrite families:
      1. Y is guaranteed to match at most one row after optionality is fixed:
         move the conjunct directly from X to Y.
      2. Y may match multiple rows and Y has no dependent DEFINE condition:
         isolate the last occurrence of Y before X as a fresh variable and push
         the conjunct to that fresh variable.

    The rule is intentionally conservative.  A variable with dependent DEFINE
    semantics is not renamed because copying such a condition to a fresh label
    may change which rows and RUNNING state it observes.  When a safe multi-row
    isolation renames a pattern variable observed by CLASSIFIER(), MEASURES is
    patched with a CASE expression that maps the fresh label back to the
    original label.
    """

    _AGGREGATE_FUNCTIONS = frozenset({"AVG", "SUM", "COUNT", "MAX", "MIN"})
    _NAVIGATION_FUNCTIONS = frozenset({"PREV", "NEXT", "FIRST", "LAST"})
    _MATCH_CONTEXT_FUNCTIONS = frozenset({"MATCH_NUMBER", "CLASSIFIER"})
    _DEPENDENT_FUNCTIONS = (
        _AGGREGATE_FUNCTIONS | _NAVIGATION_FUNCTIONS | _MATCH_CONTEXT_FUNCTIONS
    )

    @property
    def name(self) -> str:
        return "forward_adjusted_condition"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_first_plan(ast)
        self._pending_plan = (self._plan_key(ast), plan)
        return plan is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        changed = False
        key = self._plan_key(ast)
        pending = getattr(self, "_pending_plan", None)
        plan = (
            pending[1]
            if pending is not None and pending[0] == key
            else self._build_first_plan(ast)
        )
        self._pending_plan = None
        while True:
            if plan is None:
                break
            self._apply_plan(ast, plan)
            ast.pattern = ast.pattern.simplify()
            changed = True
            plan = self._build_first_plan(ast)
        return changed

    def _plan_key(self, ast: RPRAbstractSyntaxTree):
        definitions = getattr(getattr(ast, "define", None), "definitions", {})
        return (
            id(ast),
            id(ast.pattern),
            id(getattr(ast, "measures", None)),
            id(getattr(ast, "subsets", None)),
            getattr(ast, "after_match_skip_to", None),
            tuple(
                (name, id(condition), id(getattr(condition, "expression", None)))
                for name, condition in definitions.items()
            ),
        )

    def _iter_defined_conditions(self, ast: RPRAbstractSyntaxTree, *, reverse=False):
        """Yield DEFINE entries that have an expression, preserving order."""
        items = ast.define.definitions.items()
        if reverse:
            items = reversed(list(items))
        for var_name, condition in items:
            if condition is not None and condition.expression is not None:
                yield var_name, condition

    def _build_first_plan(self, ast: RPRAbstractSyntaxTree):
        # SQL/Oracle CLASSIFIER has no arguments.  The expression parser is
        # deliberately permissive and can still represent CLASSIFIER(A), but
        # its dialect-specific meaning is unknown.  In particular, an
        # occurrence-isolation rewrite may introduce a fresh classifier label
        # that cannot be repaired soundly for such a call.  Leave the query
        # untouched instead of guessing at non-standard semantics.
        if self._has_classifier_with_arguments(ast):
            return None

        # More specialized rewrites take precedence over the generic
        # conjunct pushdown below.  Keeping that ordering in one table makes
        # it explicit and avoids another block for every new plan family.
        specialized_builders = (
            self._build_logical_nav_out_of_bounds_plan,
            self._build_next_alternation_expansion_plan,
            self._build_prev_optional_expansion_plan,
            self._build_fixed_physical_navigation_plan,
            self._build_deterministic_physical_navigation_plan,
        )
        for build_plan in specialized_builders:
            plan = build_plan(ast)
            if plan is not None:
                return plan

        must_appear_vars = set(ast.pattern.get_must_appear_var())
        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if x_var not in must_appear_vars:
                continue
            # One DEFINE entry applies to every occurrence of its pattern
            # variable.  Moving a conjunct away from X would otherwise remove
            # the check from all X rows while constraining only one selected Y
            # occurrence (for example PATTERN(A X A X)).
            if self._max_occurrence_count(ast.pattern, x_var) != 1:
                continue

            conjuncts = self._flatten_and(condition.expression)
            first_null_plan = self._build_first_null_optional_plan(
                ast, x_var, conjuncts
            )
            if first_null_plan is not None:
                return first_null_plan

            grouped_adjustment_plan = self._build_grouped_forward_adjustment_plan(
                ast, x_var, conjuncts
            )
            if grouped_adjustment_plan is not None:
                return grouped_adjustment_plan

            for idx, conjunct in enumerate(conjuncts):
                adjusted_plan = self._build_forward_adjustment_plan(
                    ast, x_var, conjuncts, idx, conjunct
                )
                if adjusted_plan is not None:
                    return adjusted_plan

                y_var = self._extract_single_pushable_var(conjunct, x_var)
                if y_var is None:
                    required_var = self._extract_presence_required_var(conjunct, x_var)
                    if required_var is None:
                        continue
                    if (
                        self._min_future_rows_to_target(
                            ast.pattern, required_var, x_var
                        )
                        is None
                    ):
                        continue
                    working_must_appear = set(ast.pattern.get_must_appear_var())
                    if required_var in working_must_appear:
                        continue
                    rewritten_pattern = self._require_var_to_appear(
                        ast.pattern, required_var
                    )
                    if rewritten_pattern is None:
                        continue
                    rewritten_pattern = rewritten_pattern.simplify()
                    if required_var not in rewritten_pattern.get_must_appear_var():
                        continue
                    if not self._must_have_var_before_target(
                        rewritten_pattern, required_var, x_var
                    ):
                        continue
                    removed_vars = {
                        var for var in self._collect_all_vars(ast.pattern)
                    } - {var for var in self._collect_all_vars(rewritten_pattern)}
                    if not self._removed_vars_allow_skip_rewrite(ast, removed_vars):
                        continue
                    return {
                        "kind": "require_presence",
                        "x_var": x_var,
                        "y_var": required_var,
                        "conjunct_index": idx,
                        "pattern": rewritten_pattern,
                        "removed_vars": removed_vars,
                    }
                if self._min_future_rows_to_target(ast.pattern, y_var, x_var) is None:
                    continue

                working_pattern = ast.pattern
                pattern_changed = False
                working_must_appear = set(working_pattern.get_must_appear_var())
                if y_var not in working_must_appear:
                    if self._is_nullable_null_check_expr(conjunct, y_var):
                        nullable_quantifier_plan = (
                            self._build_nullable_quantifier_split_plan(
                                ast, working_pattern, x_var, y_var, conjuncts, idx
                            )
                        )
                        if nullable_quantifier_plan is not None:
                            return nullable_quantifier_plan

                        alternative_vars = self._find_nullable_split_alternative_vars(
                            working_pattern, y_var, x_var
                        )
                        if not alternative_vars:
                            continue
                        # The pushed predicate becomes part of Y's global
                        # DEFINE and therefore applies to every Y row.  An
                        # ordinary Y.col reference in X denotes only the last
                        # running Y row, so this is equivalent only when no
                        # successful path can contain more than one Y.
                        if self._max_occurrence_count(working_pattern, y_var) != 1:
                            continue
                        return {
                            "kind": "null_optional_split",
                            "x_var": x_var,
                            "y_var": y_var,
                            "alternative_vars": alternative_vars,
                            "conjunct_index": idx,
                            "pattern": self._distribute_alternation_before_target(
                                working_pattern, y_var, x_var
                            )
                            or working_pattern,
                        }
                    # An unmatched pattern variable has the value NULL in
                    # Oracle.  Requiring an optional Y to appear is therefore
                    # valid only when the complete predicate is provably
                    # null-rejecting for Y.  In particular, scalar functions
                    # such as NVL/COALESCE and CASE may turn that NULL into a
                    # value for which the predicate is TRUE.
                    if not self._is_provably_null_rejecting(conjunct, y_var):
                        continue
                    rewritten_pattern = self._require_var_to_appear(
                        working_pattern, y_var
                    )
                    if rewritten_pattern is None:
                        continue
                    rewritten_pattern = rewritten_pattern.simplify()
                    working_must_appear = set(rewritten_pattern.get_must_appear_var())
                    if y_var not in working_must_appear:
                        continue
                    working_pattern = rewritten_pattern
                    pattern_changed = (
                        working_pattern.to_string() != ast.pattern.to_string()
                    )
                    removed_vars = {
                        var for var in self._collect_all_vars(ast.pattern)
                    } - {var for var in self._collect_all_vars(working_pattern)}
                    if not self._removed_vars_allow_skip_rewrite(ast, removed_vars):
                        continue
                else:
                    removed_vars = set()

                # A qualified Y reference in X observes rows already mapped
                # when X is evaluated.  An existential path from Y to X is
                # insufficient: every branch containing X must put Y first.
                # This rejects shapes such as (A X | X A).
                if not self._must_have_var_before_target(working_pattern, y_var, x_var):
                    continue

                max_occurrence = self._max_occurrence_count(working_pattern, y_var)
                if max_occurrence == 1:
                    return {
                        "kind": "direct",
                        "x_var": x_var,
                        "y_var": y_var,
                        "conjunct_index": idx,
                        "pattern": working_pattern,
                        "pattern_changed": pattern_changed,
                        "removed_vars": removed_vars,
                    }

                related_group_indices = self._collect_same_var_push_group_indices(
                    conjuncts, idx, x_var, y_var
                )
                if related_group_indices is not None and len(related_group_indices) > 1:
                    if not self._can_isolate_occurrence(ast, x_var, y_var):
                        continue
                    plan = self._build_last_occurrence_push_plan(
                        ast,
                        working_pattern,
                        x_var,
                        y_var,
                        idx,
                        kind="rename_group",
                        group_indices=sorted(related_group_indices),
                    )
                    if plan is not None:
                        return plan
                    continue

                if any(
                    other_idx != idx
                    and self._expr_references_any_var(other_conjunct, {y_var})
                    for other_idx, other_conjunct in enumerate(conjuncts)
                ):
                    continue

                if not self._can_isolate_occurrence(ast, x_var, y_var):
                    continue

                plan = self._build_last_occurrence_push_plan(
                    ast, working_pattern, x_var, y_var, idx, kind="rename"
                )
                if plan is not None:
                    return plan
        return None

    def _build_last_occurrence_push_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        pattern,
        x_var: str,
        y_var: str,
        conjunct_index: int,
        *,
        kind: str,
        group_indices=None,
    ):
        """Build the common plan for isolating Y's last occurrence before X."""
        # A DEFINE entry belongs to every occurrence of its variable.  This
        # plan moves a predicate from X to one isolated Y occurrence, so it
        # cannot also repair later Y occurrences whose self-DEFINE and running
        # population must continue to use the original label.
        if self._can_var_occur_after_target(pattern, y_var, x_var):
            return None

        pattern_counts = self._count_pattern_variables(pattern)
        new_var = self._generate_new_var_name(ast, pattern_counts)
        renamed_pattern = self._rename_last_occurrence_before_target(
            pattern, y_var, x_var, new_var
        )
        if renamed_pattern is None:
            return None
        renamed_pattern = renamed_pattern.simplify()
        if renamed_pattern.to_string() == pattern.to_string():
            return None
        if not self._skip_allows_renamed_pattern(
            ast,
            y_var,
            new_var,
            renamed_pattern,
            renamed_is_last=True,
        ):
            return None
        if not self._other_define_refs_allow_occurrence_rename(
            ast, renamed_pattern, x_var, y_var, new_var
        ):
            return None

        plan = {
            "kind": kind,
            "x_var": x_var,
            "y_var": y_var,
            "new_var": new_var,
            "conjunct_index": conjunct_index,
            "pattern": renamed_pattern,
            "rewrite_skip_to_new_var": self._skip_targets_last_var(ast, y_var),
            **self._renamed_var_plan_fields(ast, y_var, new_var, pattern_counts),
        }
        if group_indices is not None:
            plan["group_indices"] = group_indices
        return plan

    def _renamed_var_plan_fields(
        self,
        ast: RPRAbstractSyntaxTree,
        original_var: str,
        new_var: str,
        used_names: Dict[str, int],
    ) -> dict:
        """Return metadata shared by every occurrence-renaming plan."""
        measure_subset_var = None
        if self._measures_reference_var(ast, original_var):
            measure_subset_var = self._generate_new_var_name(
                ast, {**used_names, new_var: 1}
            )
        return {
            "rewrite_classifier": (
                ast.measures is not None and ast.measures.has_func("CLASSIFIER")
            ),
            "measure_subset_var": measure_subset_var,
        }

    def _build_logical_nav_out_of_bounds_plan(self, ast: RPRAbstractSyntaxTree):
        expanded_pattern = self._expand_fixed_linear_pattern(ast.pattern)
        linear_vars = self._get_fixed_linear_pattern_vars(expanded_pattern)
        if linear_vars is None:
            return None

        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if self._max_occurrence_count(ast.pattern, x_var) != 1:
                continue
            x_indices = [
                idx for idx, var_name in enumerate(linear_vars) if var_name == x_var
            ]
            if len(x_indices) != 1:
                continue
            x_idx = x_indices[0]

            conjuncts = self._flatten_and(condition.expression)
            for idx, conjunct in enumerate(conjuncts):
                if not self._is_logical_nav_null_rejecting_comparison(conjunct):
                    continue
                logical_target = self._extract_single_logical_offset_nav_target(
                    conjunct
                )
                if logical_target is None:
                    continue

                base_var, offset, _ = logical_target
                # A SUBSET name does not occur directly in the linearized
                # PATTERN, but it can still have preceding member rows.  An
                # empty primary-variable candidate list therefore does not
                # prove that FIRST/LAST on the union variable is out of
                # bounds.
                if ast.subsets is not None and base_var in ast.subsets:
                    continue
                # DEFINE is evaluated after the current candidate row has
                # been tentatively mapped to x_var.  Consequently, logical
                # navigation over x_var can see that current row.  Excluding
                # x_idx here made FIRST/LAST(x_var.col, 0) look out of bounds
                # when x_var occurred once in the fixed pattern.
                candidates = [
                    pos for pos in range(0, x_idx + 1) if linear_vars[pos] == base_var
                ]
                if offset < len(candidates):
                    continue

                return {
                    "kind": "replace_conjunct",
                    "x_var": x_var,
                    "y_var": base_var,
                    "conjunct_index": idx,
                    "replacement": BinaryExpr(Literal(1), "!=", Literal(1)),
                    "pattern": ast.pattern,
                }
        return None

    def _is_logical_nav_null_rejecting_comparison(self, expr: Expr) -> bool:
        """Return whether an out-of-bounds logical navigation makes expr false.

        FIRST/LAST return NULL when their logical offset is out of bounds, but
        that does not imply that an enclosing expression is NULL.  NVL,
        COALESCE, CASE, and user-defined functions can all turn that NULL into
        a value which makes the comparison true.  Only follow operators whose
        NULL propagation is known here; treating an unknown wrapper as safe
        would make this rewrite unsound.
        """
        if isinstance(expr, SemanticsExpr):
            return self._is_logical_nav_null_rejecting_comparison(expr.expr)
        if not (
            isinstance(expr, BinaryExpr)
            and expr.op in ("=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "NOT LIKE")
        ):
            return False
        return self._is_guaranteed_null_from_logical_nav(
            expr.left
        ) or self._is_guaranteed_null_from_logical_nav(expr.right)

    def _is_guaranteed_null_from_logical_nav(self, expr: Expr) -> bool:
        """Prove that expr is NULL when its FIRST/LAST result is NULL."""
        if isinstance(expr, SemanticsExpr):
            return self._is_guaranteed_null_from_logical_nav(expr.expr)
        if isinstance(expr, FunctionCall):
            # A logical navigation call is the NULL-producing leaf.  Do not
            # recurse through any other function: even a built-in function
            # may deliberately handle NULL, and unknown/UDF semantics cannot
            # be proven by this rule.
            return (
                expr.name in ("FIRST", "LAST")
                and self._extract_logical_nav_target(expr) is not None
            )
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*", "/", "%"):
            return self._is_guaranteed_null_from_logical_nav(
                expr.left
            ) or self._is_guaranteed_null_from_logical_nav(expr.right)
        return False

    def _build_nullable_quantifier_split_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        pattern,
        x_var: str,
        y_var: str,
        conjuncts: List[Expr],
        conjunct_index: int,
    ):
        # This plan isolates the last row contributed by one quantified Y
        # occurrence.  If Y also occurs elsewhere in PATTERN, that other
        # occurrence can become the actual running LAST(Y) seen by X.  Moving
        # the null predicate only to the selected quantifier would then test
        # the wrong row.
        if self._count_pattern_variables(pattern).get(y_var, 0) != 1:
            return None
        # This rewrite copies Y's DEFINE condition to a fresh final
        # occurrence.  Dependent/running conditions would observe two reset
        # populations, and a SKIP target would no longer denote the same Y
        # occurrence (or could become unmatched when only the fresh label is
        # present).
        if isinstance(ast.define.definitions.get(y_var), DependentCondition):
            return None
        if self._skip_references_var(ast, y_var):
            return None
        split = self._split_nullable_quantifier_before_target(
            pattern, y_var, x_var, ast
        )
        if split is None:
            return None
        new_var, rewritten_pattern, original_var = split
        pattern_counts = self._count_pattern_variables(pattern)
        return {
            "kind": "nullable_quantifier_split",
            "x_var": x_var,
            "y_var": original_var,
            "new_var": new_var,
            "conjunct_index": conjunct_index,
            "pattern": rewritten_pattern,
            **self._renamed_var_plan_fields(ast, original_var, new_var, pattern_counts),
        }

    def _build_first_null_optional_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        x_var: str,
        conjuncts: List[Expr],
    ):
        must_appear_vars = set(ast.pattern.get_must_appear_var())
        for idx, conjunct in enumerate(conjuncts):
            first_null = self._extract_first_null_check(conjunct)
            if first_null is None:
                continue
            y_var, col_name = first_null
            if y_var in must_appear_vars:
                continue
            # DEFINE has running semantics.  A Y after X is invisible while X
            # is evaluated, but merging this predicate into Y's global DEFINE
            # would constrain that future occurrence.
            if self._can_var_occur_after_target(ast.pattern, y_var, x_var):
                continue
            if self._min_future_rows_to_target(ast.pattern, y_var, x_var) is None:
                continue
            return {
                "kind": "first_null_optional",
                "x_var": x_var,
                "y_var": y_var,
                "col_name": col_name,
                "conjunct_index": idx,
                "pattern": ast.pattern,
            }
        return None

    def _build_grouped_forward_adjustment_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        x_var: str,
        conjuncts: List[Expr],
    ):
        groups: Dict[str, Dict[str, List[int]]] = {}
        for idx, conjunct in enumerate(conjuncts):
            bindable_var = self._extract_single_bindable_var(conjunct, x_var)
            if bindable_var is not None:
                groups.setdefault(bindable_var, {"bind": [], "push": []})[
                    "bind"
                ].append(idx)
                continue

            pushable_var = self._extract_single_pushable_var(conjunct, x_var)
            if pushable_var is not None:
                groups.setdefault(pushable_var, {"bind": [], "push": []})[
                    "push"
                ].append(idx)

        for y_var, group in groups.items():
            bind_indices = group["bind"]
            push_indices = group["push"]
            if not bind_indices or not push_indices:
                continue
            group_indices = set(bind_indices + push_indices)
            if any(
                idx not in group_indices
                and self._expr_references_any_var(conjunct, {y_var})
                for idx, conjunct in enumerate(conjuncts)
            ):
                continue
            # Renaming one occurrence of a multi-row variable changes the
            # population seen by COUNT(Y.*).  Re-qualifying that COUNT to the
            # fresh singleton variable is also wrong, so neither form can be
            # preserved by this occurrence-isolation rewrite.
            if any(
                self._has_count_star_for_var(conjuncts[idx], y_var)
                for idx in group_indices
            ):
                continue
            if self._min_future_rows_to_target(ast.pattern, y_var, x_var) is None:
                continue
            # The push-only terms are removed from X and evaluated only when
            # the isolated Y occurrence is present.  If an X path can omit Y,
            # a moved term that is false/unknown for absent Y would disappear
            # on that path, while a null-tolerant bind term could still pass.
            if not self._must_have_var_before_target(ast.pattern, y_var, x_var):
                continue
            if self._max_occurrence_count(ast.pattern, y_var) == 1:
                continue
            if not self._can_isolate_occurrence(ast, x_var, y_var):
                continue

            plan = self._build_last_occurrence_push_plan(
                ast,
                ast.pattern,
                x_var,
                y_var,
                min(group_indices),
                kind="adjust_bind_group",
            )
            if plan is None:
                continue
            plan.update(
                {
                    "bind_indices": bind_indices,
                    "push_indices": push_indices,
                }
            )
            return plan
        return None

    def _build_forward_adjustment_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        x_var: str,
        conjuncts: List[Expr],
        idx: int,
        conjunct: Expr,
    ):
        logical_nav = self._extract_single_logical_nav_var(conjunct)
        if logical_nav is not None:
            y_var, func_name, offset = logical_nav
            # With at most one Y row, FIRST(Y.col) and LAST(Y.col) at offset
            # zero are exactly the ordinary running Y.col reference.  Handle
            # this before the nullable-occurrence isolation path below: no
            # occurrence is renamed or made mandatory here, so the empty-Y
            # case continues to yield NULL.  A later planning pass can then
            # tighten a nullable Y when the rewritten predicate is itself
            # null-rejecting.
            if (
                offset == 0
                and y_var != x_var
                and self._max_occurrence_count(ast.pattern, y_var) == 1
            ):
                return {
                    "kind": "replace_singleton_logical_nav",
                    "x_var": x_var,
                    "y_var": y_var,
                    "conjunct_index": idx,
                    "pattern": ast.pattern,
                    "func_name": func_name,
                    "offset": offset,
                }

        if logical_nav is not None:
            y_var, _, offset = logical_nav
            # The generic splitter applies a logical offset inside one
            # syntactic Y occurrence.  With Y in several pattern locations,
            # the requested row may fall through from a short quantified
            # occurrence to an earlier/later Y occurrence instead.  Fixed
            # linear patterns are handled by the specialized builder above;
            # leave every other cross-occurrence offset unchanged.
            if (
                offset > 0
                and self._count_pattern_variables(ast.pattern).get(y_var, 0) != 1
            ):
                logical_nav = None

        if logical_nav is not None:
            y_var, func_name, offset = logical_nav
            # If Y is nullable, isolating a logical occurrence removes the
            # empty-Y branch.  That is valid only when the complete conjunct
            # is proven false/unknown for the NULL navigation result.  Scalar
            # wrappers such as NVL may instead make the empty branch succeed.
            if (
                y_var not in ast.pattern.get_must_appear_var()
                and not self._is_logical_nav_null_rejecting_comparison(conjunct)
            ):
                logical_nav = None

        if logical_nav is not None:
            y_var, func_name, offset = logical_nav
            isolates_last = func_name == "LAST" and offset == 0
            if self._can_adjust_reference_by_renaming(
                ast,
                x_var,
                y_var,
                conjuncts,
                idx,
                require_no_other_define_refs=not isolates_last,
            ):
                pattern_counts = self._count_pattern_variables(ast.pattern)
                new_var = self._generate_new_var_name(ast, pattern_counts)
                if offset > 0:
                    renamed_pattern = self._rename_logical_occurrence_before_target(
                        ast.pattern, y_var, x_var, new_var, func_name, offset
                    )
                elif func_name == "FIRST":
                    renamed_pattern = self._rename_first_occurrence_before_target(
                        ast.pattern, y_var, x_var, new_var
                    )
                else:
                    renamed_pattern = self._rename_last_occurrence_before_target(
                        ast.pattern, y_var, x_var, new_var
                    )
                if renamed_pattern is not None:
                    renamed_pattern = renamed_pattern.simplify()
                    if renamed_pattern.to_string() != ast.pattern.to_string():
                        isolates_last = func_name == "LAST" and offset == 0
                        if not self._skip_allows_renamed_pattern(
                            ast,
                            y_var,
                            new_var,
                            renamed_pattern,
                            renamed_is_last=isolates_last,
                        ):
                            return None
                        return {
                            "kind": "adjust_logical_nav",
                            "x_var": x_var,
                            "y_var": y_var,
                            "new_var": new_var,
                            "conjunct_index": idx,
                            "pattern": renamed_pattern,
                            "func_name": func_name,
                            "offset": offset,
                            **self._renamed_var_plan_fields(
                                ast, y_var, new_var, pattern_counts
                            ),
                            "rewrite_skip_to_new_var": (
                                self._skip_targets_last_var(ast, y_var)
                                and func_name == "LAST"
                                and offset == 0
                            ),
                        }

        y_var = self._extract_single_bindable_var(conjunct, x_var)
        if y_var is None:
            return None
        if not self._can_adjust_reference_by_renaming(
            ast, x_var, y_var, conjuncts, idx
        ):
            return None
        return self._build_last_occurrence_push_plan(
            ast,
            ast.pattern,
            x_var,
            y_var,
            idx,
            kind="adjust_bind",
        )

    def _can_adjust_reference_by_renaming(
        self,
        ast: RPRAbstractSyntaxTree,
        x_var: str,
        y_var: str,
        conjuncts: List[Expr],
        idx: int,
        require_no_other_define_refs: bool = False,
    ) -> bool:
        if y_var is None:
            return False
        if self._max_occurrence_count(ast.pattern, x_var) != 1:
            return False
        if self._can_var_occur_after_target(ast.pattern, y_var, x_var):
            return False
        if self._min_future_rows_to_target(ast.pattern, y_var, x_var) is None:
            return False
        if self._max_occurrence_count(ast.pattern, y_var) == 1:
            return False
        if not self._can_isolate_occurrence(
            ast,
            x_var,
            y_var,
            require_no_other_define_refs=require_no_other_define_refs,
        ):
            return False
        if self._has_count_star_for_var(conjuncts[idx], y_var):
            return False
        return not any(
            other_idx != idx and self._expr_references_any_var(other_conjunct, {y_var})
            for other_idx, other_conjunct in enumerate(conjuncts)
        )

    def _build_fixed_physical_navigation_plan(self, ast: RPRAbstractSyntaxTree):
        expanded_pattern = self._expand_fixed_linear_pattern(ast.pattern)
        linear_vars = self._get_fixed_linear_pattern_vars(expanded_pattern)
        if linear_vars is None:
            return None

        pattern_counts = self._count_pattern_variables(ast.pattern)
        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if self._max_occurrence_count(ast.pattern, x_var) != 1:
                continue
            x_indices = [
                idx for idx, var_name in enumerate(linear_vars) if var_name == x_var
            ]
            if len(x_indices) != 1:
                continue
            x_idx = x_indices[0]
            conjuncts = self._flatten_and(condition.expression)
            occurrence_exprs: Dict[int, List[Expr]] = {}
            pushed_indices: List[int] = []
            old_vars: Set[str] = set()
            has_navigation_push = False

            for idx, conjunct in enumerate(conjuncts):
                compound_push = None
                for nav_name in ("PREV", "NEXT"):
                    compound_target = self._extract_single_compound_physical_nav_target(
                        conjunct, nav_name
                    )
                    if compound_target is None:
                        continue
                    target_idx = self._resolve_fixed_compound_target_index(
                        linear_vars,
                        x_idx,
                        nav_name,
                        compound_target,
                    )
                    if target_idx is None:
                        break
                    rewritten = self._replace_single_compound_navigation_with_expr(
                        conjunct,
                        nav_name,
                        linear_vars[target_idx],
                    )
                    if rewritten is not None:
                        compound_push = target_idx, rewritten
                    break

                if compound_push is not None:
                    target_idx, rewritten = compound_push
                    occurrence_exprs.setdefault(target_idx, []).append(rewritten)
                    pushed_indices.append(idx)
                    old_vars.add(linear_vars[target_idx])
                    has_navigation_push = True
                    continue

                prev_target = self._extract_single_prev_target(conjunct)
                if prev_target is not None:
                    base_var, offset = prev_target
                    base_candidates = [
                        pos for pos in range(0, x_idx) if linear_vars[pos] == base_var
                    ]
                    if len(base_candidates) == 0:
                        continue
                    base_idx = base_candidates[-1]
                    target_idx = base_idx - offset
                    if target_idx < 0:
                        continue
                    rewritten = self._replace_single_navigation_with_column(
                        conjunct, {"PREV"}, linear_vars[target_idx]
                    )
                    if rewritten is None:
                        continue
                    occurrence_exprs.setdefault(target_idx, []).append(rewritten)
                    pushed_indices.append(idx)
                    old_vars.add(linear_vars[target_idx])
                    has_navigation_push = True
                    continue

                next_target = self._extract_single_next_target(conjunct)
                if next_target is not None:
                    base_var, offset = next_target
                    base_candidates = [
                        pos
                        for pos in range(0, x_idx + 1)
                        if linear_vars[pos] == base_var
                    ]
                    if len(base_candidates) == 0:
                        continue
                    base_idx = base_candidates[-1]
                    target_idx = base_idx + offset
                    if target_idx < 0 or target_idx >= len(linear_vars):
                        continue
                    rewritten = self._replace_single_navigation_with_column(
                        conjunct, {"NEXT"}, linear_vars[target_idx]
                    )
                    if rewritten is None:
                        continue
                    occurrence_exprs.setdefault(target_idx, []).append(rewritten)
                    pushed_indices.append(idx)
                    old_vars.add(linear_vars[target_idx])
                    has_navigation_push = True
                    continue

                logical_target = self._extract_single_logical_offset_nav_target(
                    conjunct
                )
                if logical_target is not None:
                    base_var, offset, func_name = logical_target
                    candidates = [
                        pos for pos in range(0, x_idx) if linear_vars[pos] == base_var
                    ]
                    if func_name == "FIRST":
                        target_idx = (
                            candidates[offset] if offset < len(candidates) else None
                        )
                    else:
                        target_idx = (
                            candidates[-1 - offset]
                            if offset < len(candidates)
                            else None
                        )
                    if target_idx is None:
                        continue
                    rewritten = self._replace_single_navigation_with_column(
                        conjunct, {func_name}, linear_vars[target_idx]
                    )
                    if rewritten is None:
                        continue
                    occurrence_exprs.setdefault(target_idx, []).append(rewritten)
                    pushed_indices.append(idx)
                    old_vars.add(linear_vars[target_idx])
                    has_navigation_push = True
                    continue

                y_var = self._extract_single_pushable_var(conjunct, x_var)
                if y_var is None:
                    continue
                candidates = [
                    pos for pos in range(0, x_idx) if linear_vars[pos] == y_var
                ]
                if len(candidates) == 0:
                    continue
                target_idx = candidates[-1]
                occurrence_exprs.setdefault(target_idx, []).append(conjunct)
                pushed_indices.append(idx)
                old_vars.add(linear_vars[target_idx])

            if len(occurrence_exprs) < 2 and not has_navigation_push:
                continue
            physical_actions = self._build_physical_push_actions(
                ast,
                expanded_pattern,
                linear_vars,
                occurrence_exprs,
                pattern_counts,
            )
            if physical_actions is None:
                continue
            renamed_pattern, actions = physical_actions

            return {
                "kind": "physical_multi_push",
                "x_var": x_var,
                "y_var": actions[0]["old_var"],
                "pattern": renamed_pattern.simplify(),
                "actions": actions,
                "pushed_indices": pushed_indices,
                "maybe_removed_vars": old_vars,
            }
        return None

    def _build_next_alternation_expansion_plan(self, ast: RPRAbstractSyntaxTree):
        pattern = ast.pattern
        if not isinstance(pattern, ConcatenationPattern):
            return None

        pattern_parts = pattern.patterns
        pattern_counts = self._count_pattern_variables(pattern)
        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if self._max_occurrence_count(pattern, x_var) != 1:
                continue

            x_indices = [
                idx
                for idx, child in enumerate(pattern_parts)
                if isinstance(child, VariablePattern) and child.name == x_var
            ]
            if len(x_indices) != 1:
                continue
            x_idx = x_indices[0]
            alt_idx = x_idx + 1
            if alt_idx >= len(pattern_parts):
                continue
            alternation_candidate = self._next_alternation_candidate(
                pattern_parts, alt_idx
            )
            if alternation_candidate is None:
                continue
            effective_pattern_parts, alternation = alternation_candidate

            suffix = effective_pattern_parts[alt_idx + 1 :]

            branch_infos = self._next_alternation_branch_infos(alternation)
            if not branch_infos:
                continue

            branch_vars = [info["first_var"] for info in branch_infos]
            if len(set(branch_vars)) != len(branch_vars):
                continue

            conjuncts = self._flatten_and(condition.expression)
            for idx, conjunct in enumerate(conjuncts):
                next_target = self._extract_single_next_target(conjunct)
                if next_target is None:
                    continue

                base_var, offset = next_target
                if base_var != x_var or offset != 1:
                    continue

                used_counts = dict(pattern_counts)
                actions = []
                rewritten_branches = []
                for branch_info in branch_infos:
                    branch_var = branch_info["first_var"]
                    branch = branch_info["branch"]
                    must_rename = self._next_branch_target_needs_rename(
                        pattern,
                        branch_info,
                        branch_infos,
                        suffix,
                    )
                    new_var = None
                    if must_rename:
                        if isinstance(
                            ast.define.definitions.get(branch_var),
                            DependentCondition,
                        ):
                            actions = []
                            break
                        if self._skip_references_var(ast, branch_var):
                            actions = []
                            break
                        new_var = self._generate_new_var_name(ast, used_counts)
                        used_counts[new_var] = 1

                    target_var = new_var if new_var is not None else branch_var
                    pushed_expr = self._replace_single_navigation_with_column(
                        conjunct,
                        {"NEXT"},
                        target_var,
                    )
                    if pushed_expr is None:
                        actions = []
                        break

                    rewritten_branch = self._rename_branch_first_var(branch, target_var)
                    if rewritten_branch is None:
                        actions = []
                        break
                    rewritten_branches.append(rewritten_branch)
                    actions.append(
                        self._make_push_action(
                            ast, branch_var, new_var, pushed_expr, used_counts
                        )
                    )

                if not actions:
                    continue

                rewritten_pattern = ConcatenationPattern(
                    [
                        *[
                            copy.deepcopy(part)
                            for part in effective_pattern_parts[:alt_idx]
                        ],
                        AlternationPattern(rewritten_branches),
                        *[copy.deepcopy(part) for part in suffix],
                    ]
                ).simplify()

                return {
                    "kind": "physical_multi_push",
                    "x_var": x_var,
                    "y_var": actions[0]["old_var"],
                    "pattern": rewritten_pattern,
                    "actions": actions,
                    "pushed_indices": [idx],
                    "maybe_removed_vars": set(branch_vars),
                }
        return None

    def _build_prev_optional_expansion_plan(self, ast: RPRAbstractSyntaxTree):
        pattern = ast.pattern
        if not isinstance(pattern, ConcatenationPattern):
            return None

        pattern_parts = pattern.patterns
        pattern_counts = self._count_pattern_variables(pattern)
        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if self._max_occurrence_count(pattern, x_var) != 1:
                continue

            x_indices = [
                idx
                for idx, child in enumerate(pattern_parts)
                if isinstance(child, VariablePattern) and child.name == x_var
            ]
            if len(x_indices) != 1:
                continue
            x_idx = x_indices[0]
            optional_idx = x_idx - 1
            fallback_idx = x_idx - 2
            if fallback_idx < 0:
                continue

            optional_part = pattern_parts[optional_idx]
            fallback_part = pattern_parts[fallback_idx]
            if not (
                isinstance(optional_part, QuantifiedPattern)
                and isinstance(optional_part.pattern, VariablePattern)
                and optional_part.atLeast == 0
                and optional_part.atMost == 1
                and isinstance(fallback_part, VariablePattern)
            ):
                continue

            conjuncts = self._flatten_and(condition.expression)
            for idx, conjunct in enumerate(conjuncts):
                prev_target = self._extract_single_prev_target(conjunct)
                if prev_target is None:
                    continue

                base_var, offset = prev_target
                if base_var != x_var or offset != 1:
                    continue

                used_counts = dict(pattern_counts)
                optional_var = optional_part.pattern.name
                fallback_var = fallback_part.name

                optional_new_var = None
                if self._max_occurrence_count(pattern, optional_var) != 1:
                    if isinstance(
                        ast.define.definitions.get(optional_var),
                        DependentCondition,
                    ):
                        continue
                    if self._skip_references_var(ast, optional_var):
                        continue
                    optional_new_var = self._generate_new_var_name(ast, used_counts)
                    used_counts[optional_new_var] = 1
                optional_target = (
                    optional_new_var if optional_new_var is not None else optional_var
                )

                # The fallback occurrence is always renamed in the branch
                # where the optional row is absent.
                if isinstance(
                    ast.define.definitions.get(fallback_var),
                    DependentCondition,
                ):
                    continue
                if self._skip_references_var(ast, fallback_var):
                    continue
                fallback_new_var = self._generate_new_var_name(ast, used_counts)
                used_counts[fallback_new_var] = 1

                optional_expr = self._replace_single_navigation_with_column(
                    conjunct,
                    {"PREV"},
                    optional_target,
                )
                fallback_expr = self._replace_single_navigation_with_column(
                    conjunct,
                    {"PREV"},
                    fallback_new_var,
                )
                if optional_expr is None or fallback_expr is None:
                    continue

                actions = []
                for old_var, new_var, pushed_expr in (
                    (optional_var, optional_new_var, optional_expr),
                    (fallback_var, fallback_new_var, fallback_expr),
                ):
                    actions.append(
                        self._make_push_action(
                            ast, old_var, new_var, pushed_expr, used_counts
                        )
                    )

                optional_present_branch = ConcatenationPattern(
                    [
                        copy.deepcopy(fallback_part),
                        VariablePattern(optional_target),
                    ]
                )
                optional_empty_branch = VariablePattern(fallback_new_var)
                optional_branches = (
                    [optional_present_branch, optional_empty_branch]
                    if optional_part.isGreedy
                    else [optional_empty_branch, optional_present_branch]
                )
                rewritten_pattern = ConcatenationPattern(
                    [
                        *[copy.deepcopy(part) for part in pattern_parts[:fallback_idx]],
                        AlternationPattern(optional_branches),
                        copy.deepcopy(pattern_parts[x_idx]),
                        *[copy.deepcopy(part) for part in pattern_parts[x_idx + 1 :]],
                    ]
                ).simplify()

                return {
                    "kind": "physical_multi_push",
                    "x_var": x_var,
                    "y_var": actions[0]["old_var"],
                    "pattern": rewritten_pattern,
                    "actions": actions,
                    "pushed_indices": [idx],
                    "maybe_removed_vars": {optional_var, fallback_var},
                }
        return None

    def _next_alternation_branch_infos(
        self, alternation: AlternationPattern
    ) -> Optional[List[dict]]:
        branch_infos = []
        for branch in alternation.patterns:
            first_var = self._branch_first_variable(branch)
            if first_var is None:
                return None
            branch_infos.append(
                {
                    "first_var": first_var,
                    "branch": branch,
                }
            )
        return branch_infos

    def _branch_first_variable(self, branch) -> Optional[str]:
        if isinstance(branch, VariablePattern):
            return branch.name
        if (
            isinstance(branch, ConcatenationPattern)
            and branch.patterns
            and isinstance(branch.patterns[0], VariablePattern)
        ):
            return branch.patterns[0].name
        return None

    def _build_physical_push_actions(
        self,
        ast: RPRAbstractSyntaxTree,
        pattern,
        target_vars: List[str],
        occurrence_exprs: Dict[int, List[Expr]],
        pattern_counts: Dict[str, int],
    ):
        renamed_pattern = copy.deepcopy(pattern)
        used_counts = dict(pattern_counts)
        actions = []
        for target_idx in sorted(occurrence_exprs.keys()):
            old_var = target_vars[target_idx]
            new_var = None
            if self._max_occurrence_count(pattern, old_var) != 1:
                # Copying a dependent DEFINE condition to a fresh label resets
                # the running population observed by expressions such as
                # COUNT(A.*).  A skip target would likewise stop denoting the
                # same occurrence after one A is renamed.  The general
                # occurrence-isolation path applies the same restrictions.
                if isinstance(ast.define.definitions.get(old_var), DependentCondition):
                    return None
                if self._skip_references_var(ast, old_var):
                    return None
                new_var = self._generate_new_var_name(ast, used_counts)
                used_counts[new_var] = 1
                renamed_pattern = self._rename_fixed_linear_position(
                    renamed_pattern, target_idx, new_var
                )

            target_var = new_var if new_var is not None else old_var
            pushed_expr = self._combine_and_terms(
                [
                    self._replace_var_qualifier(expr, old_var, target_var)
                    for expr in occurrence_exprs[target_idx]
                ]
            )
            actions.append(
                self._make_push_action(ast, old_var, new_var, pushed_expr, used_counts)
            )
        return renamed_pattern, actions

    def _make_push_action(
        self,
        ast: RPRAbstractSyntaxTree,
        old_var: str,
        new_var: Optional[str],
        expr: Expr,
        used_names: Dict[str, int],
    ) -> dict:
        """Build shared metadata for a physical-push action."""
        measure_subset_var = None
        if new_var is not None and self._measures_reference_var(ast, old_var):
            measure_subset_var = self._generate_new_var_name(ast, used_names)
            used_names[measure_subset_var] = 1
        return {
            "old_var": old_var,
            "new_var": new_var,
            "expr": expr,
            "measure_subset_var": measure_subset_var,
            "rewrite_classifier": (
                new_var is not None
                and ast.measures is not None
                and ast.measures.has_func("CLASSIFIER")
            ),
        }

    def _rename_branch_first_var(self, branch, target_var: str):
        if isinstance(branch, VariablePattern):
            return VariablePattern(target_var)
        if isinstance(branch, ConcatenationPattern):
            return ConcatenationPattern(
                [
                    VariablePattern(target_var),
                    *[copy.deepcopy(part) for part in branch.patterns[1:]],
                ]
            )
        return None

    def _branch_uses_target_name_later(self, branch, target_var: str) -> bool:
        if isinstance(branch, VariablePattern):
            return False
        if not isinstance(branch, ConcatenationPattern):
            return True
        return any(self._contains_var(part, target_var) for part in branch.patterns[1:])

    def _next_branch_target_needs_rename(
        self, pattern, branch_info, branch_infos: List[dict], suffix: List
    ) -> bool:
        branch_var = branch_info["first_var"]
        return (
            self._max_occurrence_count(pattern, branch_var) != 1
            or self._branch_uses_target_name_later(branch_info["branch"], branch_var)
            or any(
                other_info is not branch_info
                and self._contains_var(other_info["branch"], branch_var)
                for other_info in branch_infos
            )
            or any(
                self._contains_var(suffix_part, branch_var) for suffix_part in suffix
            )
        )

    def _next_alternation_candidate(self, pattern_parts: List, alt_idx: int):
        candidate = pattern_parts[alt_idx]
        if isinstance(candidate, AlternationPattern):
            return pattern_parts, candidate

        if (
            isinstance(candidate, QuantifiedPattern)
            and isinstance(candidate.pattern, VariablePattern)
            and candidate.atLeast == 0
            and candidate.atMost == 1
            and alt_idx + 1 < len(pattern_parts)
            and isinstance(pattern_parts[alt_idx + 1], VariablePattern)
        ):
            optional_var = candidate.pattern
            following_var = pattern_parts[alt_idx + 1]
            present_branch = ConcatenationPattern(
                [
                    VariablePattern(optional_var.name),
                    copy.deepcopy(following_var),
                ]
            )
            empty_branch = copy.deepcopy(following_var)
            alternation = AlternationPattern(
                [present_branch, empty_branch]
                if candidate.isGreedy
                else [empty_branch, present_branch]
            )
            rewritten_parts = [
                *[copy.deepcopy(part) for part in pattern_parts[:alt_idx]],
                alternation,
                *[copy.deepcopy(part) for part in pattern_parts[alt_idx + 2 :]],
            ]
            return rewritten_parts, alternation

        return None

    def _build_deterministic_physical_navigation_plan(self, ast: RPRAbstractSyntaxTree):
        expanded_pattern = self._expand_unbounded_tail_pattern(ast.pattern)
        sequence = self._get_deterministic_physical_sequence(expanded_pattern)
        if sequence is None:
            return None

        pattern_counts = self._count_pattern_variables(expanded_pattern)
        for x_var, condition in self._iter_defined_conditions(ast, reverse=True):
            if self._max_occurrence_count(ast.pattern, x_var) != 1:
                continue
            x_indices = [
                idx
                for idx, item in enumerate(sequence)
                if item.get("fixed") and item.get("var", "") == x_var
            ]
            if len(x_indices) != 1:
                continue
            x_idx = x_indices[0]
            conjuncts = self._flatten_and(condition.expression)
            occurrence_exprs: Dict[int, List[Expr]] = {}
            pushed_indices: List[int] = []
            old_vars: Set[str] = set()
            has_navigation_push = False

            for idx, conjunct in enumerate(conjuncts):
                compound_push = None
                for nav_name in ("PREV", "NEXT"):
                    compound_target = self._extract_single_compound_physical_nav_target(
                        conjunct, nav_name
                    )
                    if compound_target is None:
                        continue
                    target_idx = self._resolve_deterministic_compound_target_index(
                        sequence,
                        x_idx,
                        nav_name,
                        compound_target,
                    )
                    if target_idx is None:
                        break
                    rewritten = self._replace_single_compound_navigation_with_expr(
                        conjunct,
                        nav_name,
                        sequence[target_idx]["var"],
                    )
                    if rewritten is not None:
                        compound_push = target_idx, rewritten
                    break

                if compound_push is not None:
                    target_idx, rewritten = compound_push
                    occurrence_exprs.setdefault(target_idx, []).append(rewritten)
                    pushed_indices.append(idx)
                    old_vars.add(sequence[target_idx]["var"])
                    has_navigation_push = True
                    continue

                prev_target = self._extract_single_prev_target(conjunct)
                if prev_target is not None:
                    base_var, offset = prev_target
                    base_idx = self._last_fixed_var_index_before(
                        sequence, base_var, x_idx
                    )
                    if base_idx is None:
                        continue
                    target_idx = base_idx - offset
                    fixed_physical_walk = target_idx >= 0 and all(
                        sequence[pos].get("fixed")
                        for pos in range(target_idx, base_idx)
                    )
                    if fixed_physical_walk:
                        target_occurrence_idx = target_idx
                        rewritten = self._replace_single_navigation_with_column(
                            conjunct, {"PREV"}, sequence[target_idx]["var"]
                        )
                    else:
                        if self._has_column_ref_outside_navigation(conjunct, {"PREV"}):
                            continue
                        target_occurrence_idx = base_idx
                        rewritten = copy.deepcopy(conjunct)
                    if rewritten is None:
                        continue
                    occurrence_exprs.setdefault(target_occurrence_idx, []).append(
                        rewritten
                    )
                    pushed_indices.append(idx)
                    old_vars.add(sequence[target_occurrence_idx]["var"])
                    has_navigation_push = True
                    continue

                y_var = self._extract_single_pushable_var(conjunct, x_var)
                if y_var is None:
                    continue
                target_idx = self._last_fixed_var_index_before(sequence, y_var, x_idx)
                if target_idx is None:
                    continue
                occurrence_exprs.setdefault(target_idx, []).append(conjunct)
                pushed_indices.append(idx)
                old_vars.add(sequence[target_idx]["var"])

            if len(pushed_indices) < 2 and not has_navigation_push:
                continue

            physical_actions = self._build_physical_push_actions(
                ast,
                expanded_pattern,
                [item["var"] for item in sequence],
                occurrence_exprs,
                pattern_counts,
            )
            if physical_actions is None:
                continue
            renamed_pattern, actions = physical_actions

            return {
                "kind": "deterministic_physical_push",
                "x_var": x_var,
                "y_var": actions[0]["old_var"],
                "pattern": renamed_pattern.simplify(),
                "actions": actions,
                "pushed_indices": pushed_indices,
                "maybe_removed_vars": old_vars,
            }
        return None

    def _expand_unbounded_tail_pattern(self, pattern):
        if not isinstance(pattern, ConcatenationPattern):
            return pattern
        expanded = []
        changed = False
        for child in pattern.patterns:
            if (
                isinstance(child, QuantifiedPattern)
                and isinstance(child.pattern, VariablePattern)
                and child.atMost is None
                and child.atLeast >= 1
            ):
                fixed_tail_count = 2 if child.atLeast >= 2 else 1
                prefix_min = child.atLeast - fixed_tail_count
                if prefix_min > 0:
                    expanded.append(
                        QuantifiedPattern(
                            VariablePattern(child.pattern.name),
                            prefix_min,
                            None,
                            child.isGreedy,
                        )
                    )
                else:
                    expanded.append(
                        QuantifiedPattern(
                            VariablePattern(child.pattern.name), 0, None, child.isGreedy
                        )
                    )
                for _ in range(fixed_tail_count):
                    expanded.append(VariablePattern(child.pattern.name))
                changed = True
                continue
            expanded.append(child)
        return ConcatenationPattern(expanded) if changed else pattern

    def _get_deterministic_physical_sequence(self, pattern) -> Optional[List[dict]]:
        if not isinstance(pattern, ConcatenationPattern):
            return None
        sequence = []
        for idx, child in enumerate(pattern.patterns):
            if isinstance(child, VariablePattern):
                sequence.append(
                    {"fixed": True, "var": child.name, "pattern_index": idx}
                )
                continue
            if isinstance(child, QuantifiedPattern) and isinstance(
                child.pattern, VariablePattern
            ):
                sequence.append(
                    {"fixed": False, "var": child.pattern.name, "pattern_index": idx}
                )
                continue
            return None
        return sequence

    def _last_fixed_var_index_before(
        self, sequence: List[dict], var_name: str, before_idx: int
    ) -> Optional[int]:
        for idx in range(before_idx - 1, -1, -1):
            item = sequence[idx]
            if item.get("fixed") and item.get("var", "") == var_name:
                return idx
        return None

    def _get_fixed_linear_pattern_vars(self, pattern) -> Optional[List[str]]:
        if not isinstance(pattern, ConcatenationPattern):
            return None
        vars_in_order = []
        for child in pattern.patterns:
            if not isinstance(child, VariablePattern):
                return None
            vars_in_order.append(child.name)
        return vars_in_order

    def _expand_fixed_linear_pattern(self, pattern):
        if not isinstance(pattern, ConcatenationPattern):
            return pattern
        expanded = []
        changed = False
        for child in pattern.patterns:
            if isinstance(child, VariablePattern):
                expanded.append(child)
                continue
            if (
                isinstance(child, QuantifiedPattern)
                and isinstance(child.pattern, VariablePattern)
                and child.atMost == child.atLeast
            ):
                expanded.extend(
                    VariablePattern(child.pattern.name) for _ in range(child.atLeast)
                )
                changed = True
                continue
            return pattern
        return ConcatenationPattern(expanded) if changed else pattern

    def _rename_fixed_linear_position(self, pattern, target_idx: int, new_var: str):
        if not isinstance(pattern, ConcatenationPattern):
            return pattern
        new_patterns = []
        for idx, child in enumerate(pattern.patterns):
            if idx == target_idx and isinstance(child, VariablePattern):
                new_patterns.append(VariablePattern(new_var))
            else:
                new_patterns.append(child)
        return ConcatenationPattern(new_patterns)

    def _extract_single_prev_target(self, expr: Expr) -> Optional[Tuple[str, int]]:
        return self._extract_single_physical_nav_target(expr, "PREV")

    def _extract_single_next_target(self, expr: Expr) -> Optional[Tuple[str, int]]:
        return self._extract_single_physical_nav_target(expr, "NEXT")

    def _extract_single_compound_physical_nav_target(
        self,
        expr: Expr,
        nav_name: str,
    ):
        result = self._extract_single_navigation_call(
            expr,
            {nav_name},
            self._extract_compound_physical_nav_target,
        )
        return result[1] if result is not None else None

    def _extract_single_physical_nav_target(
        self, expr: Expr, nav_name: str
    ) -> Optional[Tuple[str, int]]:
        result = self._extract_single_navigation_call(
            expr, {nav_name}, self._extract_physical_nav_target
        )
        return result[1] if result is not None else None

    def _extract_single_logical_offset_nav_target(
        self, expr: Expr
    ) -> Optional[Tuple[str, int, str]]:
        result = self._extract_single_navigation_call(
            expr,
            {"FIRST", "LAST"},
            self._extract_explicit_logical_nav_target,
        )
        if result is None:
            return None
        func_name, (base_var, offset) = result
        return base_var, offset, func_name

    def _extract_single_navigation_call(
        self,
        expr: Expr,
        target_names: Set[str],
        target_extractor,
        allowed_opaque=None,
    ):
        """Return one valid row-pattern navigation call from ``expr``.

        Ordinary scalar functions are transparent wrappers. Other RPR
        functions make the expression ambiguous unless the caller explicitly
        accepts one as an opaque leaf (currently COUNT(var.*)).
        """
        found = []
        for node in self._walk_expr(
            expr,
            prune=lambda current: (
                isinstance(current, FunctionCall)
                and current.name in self._DEPENDENT_FUNCTIONS
            ),
        ):
            if not isinstance(node, FunctionCall):
                continue
            if node.name in target_names:
                target = target_extractor(node)
                if target is None:
                    return None
                found.append((node.name, target))
                continue
            if node.name in self._DEPENDENT_FUNCTIONS and not (
                allowed_opaque is not None and allowed_opaque(node)
            ):
                return None
        return found[0] if len(found) == 1 else None

    def _extract_explicit_logical_nav_target(
        self, func: FunctionCall
    ) -> Optional[Tuple[str, int]]:
        if len(func.args) != 2:
            return None
        return self._extract_logical_nav_target(func)

    def _extract_logical_nav_target(
        self, func: FunctionCall
    ) -> Optional[Tuple[str, int]]:
        if len(func.args) == 0 or len(func.args) > 2:
            return None
        arg = self._unwrap_semantics(func.args[0])
        if not isinstance(arg, ColumnRef) or arg.qualifier is None:
            return None
        offset = 0
        if len(func.args) == 2:
            offset_expr = self._unwrap_semantics(func.args[1])
            if not isinstance(offset_expr, Literal) or not isinstance(
                offset_expr.value, (int, float)
            ):
                return None
            offset = int(offset_expr.value)
        if offset < 0:
            return None
        return arg.qualifier, offset

    def _extract_physical_nav_target(
        self, func: FunctionCall
    ) -> Optional[Tuple[str, int]]:
        if len(func.args) == 0 or len(func.args) > 2:
            return None
        arg = self._unwrap_semantics(func.args[0])
        if not isinstance(arg, ColumnRef) or arg.qualifier is None:
            return None
        offset = self._extract_navigation_offset(func)
        if offset is None or offset <= 0:
            return None
        return arg.qualifier, offset

    def _extract_compound_physical_nav_target(self, func: FunctionCall):
        """Describe PREV/NEXT(FIRST/LAST(expr), physical_offset)."""
        if len(func.args) == 0 or len(func.args) > 2:
            return None
        logical_nav = self._unwrap_semantics(func.args[0])
        if not (
            isinstance(logical_nav, FunctionCall)
            and logical_nav.name in ("FIRST", "LAST")
        ):
            return None
        logical_target = self._extract_logical_expression_target(logical_nav)
        if logical_target is None:
            return None
        physical_offset = self._extract_navigation_offset(func)
        if physical_offset is None or physical_offset < 0:
            return None
        base_var, logical_offset = logical_target
        return base_var, logical_nav.name, logical_offset, physical_offset

    def _extract_logical_expression_target(
        self,
        func: FunctionCall,
    ) -> Optional[Tuple[str, int]]:
        """Extract the unique RPV and offset from FIRST/LAST(scalar_expr)."""
        if len(func.args) == 0 or len(func.args) > 2:
            return None
        value_expr = self._unwrap_semantics(func.args[0])
        column_refs = [
            node for node in self._walk_expr(value_expr) if isinstance(node, ColumnRef)
        ]
        if not column_refs or any(ref.qualifier is None for ref in column_refs):
            return None
        qualifiers = {ref.qualifier for ref in column_refs}
        if len(qualifiers) != 1:
            return None
        if any(
            isinstance(node, FunctionCall) and node.name in self._DEPENDENT_FUNCTIONS
            for node in self._walk_expr(value_expr)
        ):
            return None

        logical_offset = 0
        if len(func.args) == 2:
            offset_expr = self._unwrap_semantics(func.args[1])
            if not (
                isinstance(offset_expr, Literal)
                and isinstance(offset_expr.value, (int, float))
            ):
                return None
            logical_offset = int(offset_expr.value)
        if logical_offset < 0:
            return None
        return next(iter(qualifiers)), logical_offset

    def _extract_navigation_offset(self, func: FunctionCall) -> Optional[int]:
        if len(func.args) < 2:
            return 1

        offset_expr = self._unwrap_semantics(func.args[1])
        if not isinstance(offset_expr, Literal) or not isinstance(
            offset_expr.value, (int, float)
        ):
            return None

        offset = int(offset_expr.value)
        if offset <= 0:
            return 0
        return offset

    def _unwrap_semantics(self, expr: Expr) -> Expr:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        return expr

    def _replace_single_navigation_with_column(
        self, expr: Expr, nav_names: Set[str], target_var: str
    ) -> Optional[Expr]:
        """Replace one navigation call only when the whole conjunct can move.

        A column reference outside the navigation call is evaluated at the
        current row of the DEFINE that owns the conjunct.  Moving the
        conjunct to another pattern variable changes that current row (and
        can also change the RUNNING population observed by a qualified
        reference), so such a conjunct is not safe to relocate.
        """
        replaced = [False]
        invalid = [False]
        nav_name_set = set(nav_names)

        def rewrite(node: Expr):
            if node is None or invalid[0]:
                return None
            if isinstance(node, SemanticsExpr):
                return SemanticsExpr(node.semantics, rewrite(node.expr))
            if isinstance(node, FunctionCall):
                if node.name in nav_name_set:
                    if replaced[0]:
                        invalid[0] = True
                        return None
                    arg = self._unwrap_semantics(node.args[0]) if node.args else None
                    if not isinstance(arg, ColumnRef):
                        invalid[0] = True
                        return None
                    replaced[0] = True
                    return ColumnRef(target_var, arg.name)
                return FunctionCall(node.name, [rewrite(arg) for arg in node.args])
            if isinstance(node, ColumnRef):
                # Navigation arguments are handled by the FunctionCall branch
                # above and never reach here.  Every ColumnRef seen here is
                # outside the navigation operation and would retain the wrong
                # evaluation context after the conjunct is moved.
                invalid[0] = True
                return None
            if isinstance(node, Literal):
                return Literal(node.value)
            mapped = self._map_expr_children(node, rewrite)
            return mapped if mapped is not None else copy.deepcopy(node)

        rewritten = rewrite(expr)
        if invalid[0] or not replaced[0]:
            return None
        return rewritten

    def _replace_single_compound_navigation_with_expr(
        self,
        expr: Expr,
        nav_name: str,
        target_var: str,
    ) -> Optional[Expr]:
        """Replace one compound navigation with its value at a fixed row."""
        replaced = False
        invalid = False

        def rewrite(node: Expr):
            nonlocal replaced, invalid
            if node is None or invalid:
                return None
            if isinstance(node, FunctionCall) and node.name == nav_name:
                if replaced:
                    invalid = True
                    return None
                target = self._extract_compound_physical_nav_target(node)
                if target is None:
                    invalid = True
                    return None
                base_var = target[0]
                logical_nav = self._unwrap_semantics(node.args[0])
                value_expr = self._unwrap_semantics(logical_nav.args[0])
                replaced = True
                return self._replace_var_qualifier(value_expr, base_var, target_var)
            if isinstance(node, ColumnRef):
                invalid = True
                return None
            if isinstance(node, Literal):
                return Literal(node.value)
            mapped = self._map_expr_children(node, rewrite)
            return mapped if mapped is not None else copy.deepcopy(node)

        rewritten = rewrite(expr)
        return rewritten if replaced and not invalid else None

    def _resolve_fixed_compound_target_index(
        self,
        linear_vars: List[str],
        current_idx: int,
        physical_nav: str,
        target,
    ) -> Optional[int]:
        """Resolve a compound logical-then-physical navigation in a fixed pattern."""
        base_var, logical_nav, logical_offset, physical_offset = target
        candidates = [
            pos for pos in range(0, current_idx + 1) if linear_vars[pos] == base_var
        ]
        if logical_offset >= len(candidates):
            return None
        if logical_nav == "FIRST":
            base_idx = candidates[logical_offset]
        else:
            base_idx = candidates[-1 - logical_offset]
        delta = -physical_offset if physical_nav == "PREV" else physical_offset
        target_idx = base_idx + delta
        if target_idx < 0 or target_idx >= len(linear_vars):
            return None
        return target_idx

    def _resolve_deterministic_compound_target_index(
        self,
        sequence: List[dict],
        current_idx: int,
        physical_nav: str,
        target,
    ) -> Optional[int]:
        """Resolve a compound target without crossing variable-length nodes."""
        base_var, logical_nav, logical_offset, physical_offset = target
        base_candidates = [
            pos for pos in range(0, current_idx + 1) if sequence[pos]["var"] == base_var
        ]
        if logical_nav == "LAST":
            base_candidates.reverse()
        if logical_offset >= len(base_candidates):
            return None

        # A variable-length occurrence before the selected logical offset can
        # contribute an unknown number of rows mapped to base_var.
        relevant_candidates = base_candidates[: logical_offset + 1]
        if any(not sequence[pos].get("fixed") for pos in relevant_candidates):
            return None
        base_idx = relevant_candidates[-1]

        direction = -1 if physical_nav == "PREV" else 1
        target_idx = base_idx
        for _ in range(physical_offset):
            target_idx += direction
            if target_idx < 0 or target_idx >= len(sequence):
                return None
            # One sequence entry represents exactly one physical row only for
            # fixed occurrences. Crossing a quantifier would be ambiguous.
            if not sequence[target_idx].get("fixed"):
                return None
        return target_idx

    def _has_column_ref_outside_navigation(
        self,
        expr: Expr,
        nav_names: Set[str],
    ) -> bool:
        """Return whether a moved conjunct depends on its DEFINE current row."""
        return any(
            isinstance(node, ColumnRef)
            for node in self._walk_expr(
                expr,
                # Navigation owns the row context of its entire argument.
                prune=lambda node: (
                    isinstance(node, FunctionCall) and node.name in nav_names
                ),
            )
        )

    def _apply_plan(self, ast: RPRAbstractSyntaxTree, plan) -> None:
        ast.pattern = plan["pattern"]

        x_var = plan["x_var"]
        y_var = plan["y_var"]
        if plan.get("rewrite_skip_to_new_var"):
            ast.after_match_skip_to = plan["new_var"]
        condition = ast.define.definitions.get(x_var)
        conjuncts = self._flatten_and(condition.expression)

        parser = ExprParser()
        if plan["kind"] == "replace_conjunct":
            adjusted_terms = list(conjuncts)
            adjusted_terms[plan["conjunct_index"]] = copy.deepcopy(plan["replacement"])
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var,
                self._combine_and_terms(adjusted_terms),
            )
            return

        if plan["kind"] == "replace_singleton_logical_nav":
            adjusted_terms = list(conjuncts)
            adjusted_terms[plan["conjunct_index"]] = (
                self._replace_logical_nav_with_column(
                    conjuncts[plan["conjunct_index"]],
                    y_var,
                    y_var,
                    plan["func_name"],
                    plan["offset"],
                )
            )
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var,
                self._combine_and_terms(adjusted_terms),
            )
            return

        if plan["kind"] in ("physical_multi_push", "deterministic_physical_push"):
            self._apply_physical_push_plan(ast, plan, conjuncts, parser)
            return

        if plan["kind"] == "adjust_bind_group":
            new_var = plan["new_var"]
            push_indices = set(plan["push_indices"])
            bind_indices = set(plan["bind_indices"])
            adjusted_terms = []
            for idx, term in enumerate(conjuncts):
                if idx in push_indices:
                    continue
                if idx in bind_indices:
                    adjusted_terms.append(
                        self._replace_var_qualifier(term, y_var, new_var)
                    )
                else:
                    adjusted_terms.append(term)
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var, self._combine_and_terms(adjusted_terms)
            )

            pushed_expr = self._combine_and_terms(
                [
                    self._replace_var_qualifier(conjuncts[idx], y_var, new_var)
                    for idx in plan["push_indices"]
                ]
            )
            self._finish_occurrence_rename(
                ast,
                parser,
                plan,
                x_var,
                y_var,
                new_var,
                pushed_expr,
                retarget=True,
            )
            return

        pushed_conjunct = conjuncts[plan["conjunct_index"]]
        remaining_terms = [
            term for idx, term in enumerate(conjuncts) if idx != plan["conjunct_index"]
        ]
        new_x_expr = self._combine_and_terms(remaining_terms)

        if plan["kind"] != "require_presence":
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var, new_x_expr
            )

        if plan["kind"] == "require_presence":
            self._rewrite_clauses_after_removed_vars(
                ast, plan.get("removed_vars", set()), parser
            )
            return

        if plan["kind"] == "first_null_optional":
            count_expr = FunctionCall("COUNT", [ColumnRef(y_var, "*")])
            first_or_empty_expr = BinaryExpr(
                BinaryExpr(count_expr, ">", Literal(1)),
                "OR",
                IsNullExpr(ColumnRef(y_var, plan["col_name"]), False),
            )
            self._merge_conjunct_into_var_definition(
                ast,
                parser,
                y_var,
                first_or_empty_expr,
                insert_by_pattern_order=True,
            )
            return

        if plan["kind"] in ("null_optional_split", "direct"):
            self._merge_conjunct_into_var_definition(
                ast,
                parser,
                y_var,
                pushed_conjunct,
                insert_before_var=x_var,
            )
            if plan["kind"] == "direct":
                self._rewrite_clauses_after_removed_vars(
                    ast, plan.get("removed_vars", set()), parser
                )
            return

        if plan["kind"] == "nullable_quantifier_split":
            new_var = plan["new_var"]
            rewritten_pushed = self._replace_var_qualifier(
                pushed_conjunct, y_var, new_var
            )
            self._finish_occurrence_rename(
                ast,
                parser,
                plan,
                x_var,
                y_var,
                new_var,
                rewritten_pushed,
                retarget=False,
            )
            return

        if plan["kind"] == "adjust_bind":
            new_var = plan["new_var"]
            adjusted_terms = list(conjuncts)
            adjusted_terms[plan["conjunct_index"]] = self._replace_var_qualifier(
                pushed_conjunct, y_var, new_var
            )
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var, self._combine_and_terms(adjusted_terms)
            )
            self._finish_occurrence_rename(
                ast,
                parser,
                plan,
                x_var,
                y_var,
                new_var,
                retarget=False,
            )
            return

        if plan["kind"] == "adjust_logical_nav":
            new_var = plan["new_var"]
            adjusted_terms = list(conjuncts)
            adjusted_terms[plan["conjunct_index"]] = (
                self._replace_logical_nav_with_column(
                    pushed_conjunct,
                    y_var,
                    new_var,
                    plan["func_name"],
                    plan.get("offset", 0),
                )
            )
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var, self._combine_and_terms(adjusted_terms)
            )
            self._finish_occurrence_rename(
                ast,
                parser,
                plan,
                x_var,
                y_var,
                new_var,
                retarget=False,
            )
            return

        if plan["kind"] == "rename_group":
            new_var = plan["new_var"]
            group_indices = set(plan["group_indices"])
            remaining_terms = [
                term for idx, term in enumerate(conjuncts) if idx not in group_indices
            ]
            ast.define.definitions[x_var] = parser._analyze_dependency(
                x_var,
                self._combine_and_terms(remaining_terms),
            )

            pushed_expr = self._combine_and_terms(
                [
                    self._replace_var_qualifier(conjuncts[idx], y_var, new_var)
                    for idx in plan["group_indices"]
                ]
            )
            self._finish_occurrence_rename(
                ast,
                parser,
                plan,
                x_var,
                y_var,
                new_var,
                pushed_expr,
                retarget=True,
            )
            return

        new_var = plan["new_var"]
        rewritten_pushed = self._replace_var_qualifier(pushed_conjunct, y_var, new_var)
        self._finish_occurrence_rename(
            ast,
            parser,
            plan,
            x_var,
            y_var,
            new_var,
            rewritten_pushed,
            retarget=True,
        )

    def _repair_renamed_occurrence(
        self,
        ast: RPRAbstractSyntaxTree,
        plan,
        original_var: str,
        new_var: str,
    ) -> None:
        """Repair clauses whose view of a variable changed after isolation."""
        self._rewrite_subsets_for_renamed_var(ast, original_var, new_var)
        if plan.get("measure_subset_var") is not None:
            self._rewrite_measures_with_renamed_var_subset(
                ast,
                original_var,
                new_var,
                plan["measure_subset_var"],
            )
        if plan.get("rewrite_classifier"):
            self._rewrite_classifier_labels(ast, new_var, original_var)
        if not self._contains_var(ast.pattern, original_var):
            self._remove_definition(ast, original_var)

    def _install_renamed_definition(
        self,
        ast: RPRAbstractSyntaxTree,
        parser: ExprParser,
        before_var: str,
        original_var: str,
        new_var: str,
        pushed_expr: Optional[Expr] = None,
    ) -> None:
        """Copy a DEFINE condition to an isolated occurrence and extend it."""
        original_condition = ast.define.definitions.get(original_var)
        inherited_expr = None
        if original_condition is not None and original_condition.expression is not None:
            inherited_expr = self._replace_var_qualifier(
                original_condition.expression, original_var, new_var
            )
        if pushed_expr is None:
            if inherited_expr is None:
                return
            new_var_expr = inherited_expr
        else:
            new_var_expr = self._merge_into_var_expr(inherited_expr, pushed_expr)
        self._insert_definition_before(
            ast,
            before_var,
            new_var,
            parser._analyze_dependency(new_var, new_var_expr),
        )

    def _finish_occurrence_rename(
        self,
        ast: RPRAbstractSyntaxTree,
        parser: ExprParser,
        plan,
        x_var: str,
        y_var: str,
        new_var: str,
        pushed_expr: Optional[Expr] = None,
        *,
        retarget: bool,
    ) -> None:
        """Install Y's copied condition on the isolated occurrence and repair.

        ``retarget`` is False only for rewrites whose isolated row must not be
        substituted for every later reference to ``y_var`` (see the comments at
        each call site).
        """
        self._install_renamed_definition(
            ast, parser, x_var, y_var, new_var, pushed_expr
        )
        if retarget:
            self._retarget_define_refs_to_renamed_occurrence(
                ast, x_var, y_var, new_var, parser
            )
        else:
            # These rewrites intentionally make only one expression refer to
            # the isolated occurrence.  Any remaining reference to y_var in
            # another DEFINE still needs the complete pre-split population,
            # not just y_var with the renamed row missing.  A union variable
            # restores that population under running semantics.
            self._rewrite_define_refs_with_renamed_var_subset(
                ast,
                y_var,
                [
                    {
                        "new_var": new_var,
                        "measure_subset_var": plan.get("measure_subset_var"),
                    }
                ],
                parser,
            )
        self._repair_renamed_occurrence(ast, plan, y_var, new_var)

    def _merge_conjunct_into_var_definition(
        self,
        ast: RPRAbstractSyntaxTree,
        parser: ExprParser,
        y_var: str,
        pushed_expr: Expr,
        *,
        insert_before_var: Optional[str] = None,
        insert_by_pattern_order: bool = False,
    ) -> None:
        """Merge ``pushed_expr`` into Y's DEFINE, creating it when missing."""
        existing_condition = ast.define.definitions.get(y_var)
        merged_expr = self._merge_into_var_expr(
            existing_condition.expression if existing_condition is not None else None,
            pushed_expr,
        )
        analyzed = parser._analyze_dependency(y_var, merged_expr)
        if existing_condition is not None:
            ast.define.definitions[y_var] = analyzed
        elif insert_by_pattern_order:
            self._insert_definition_by_pattern_order(ast, y_var, analyzed)
        else:
            self._insert_definition_before(ast, insert_before_var, y_var, analyzed)

    def _apply_physical_push_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan,
        conjuncts: List[Expr],
        parser: ExprParser,
    ) -> None:
        x_var = plan["x_var"]
        pushed_indices = set(plan["pushed_indices"])
        remaining_terms = [
            term for idx, term in enumerate(conjuncts) if idx not in pushed_indices
        ]
        ast.define.definitions[x_var] = parser._analyze_dependency(
            x_var,
            self._combine_and_terms(remaining_terms),
        )

        classifier_rewrites = []
        insert_missing_by_pattern_order = plan.get(
            "insert_missing_by_pattern_order",
            plan["kind"] == "physical_multi_push",
        )
        for action in plan["actions"]:
            target_var = (
                action["new_var"]
                if action.get("new_var") is not None
                else action["old_var"]
            )
            base_condition = ast.define.definitions.get(action["old_var"])
            base_expr = None
            if base_condition is not None:
                base_expr = (
                    self._replace_var_qualifier(
                        base_condition.expression, action["old_var"], target_var
                    )
                    if action.get("new_var") is not None
                    else base_condition.expression
                )

            new_var_expr = self._merge_into_var_expr(base_expr, action["expr"])
            analyzed_condition = parser._analyze_dependency(target_var, new_var_expr)

            if action.get("new_var") is not None:
                self._insert_definition_before(
                    ast, x_var, target_var, analyzed_condition
                )
                if action.get("rewrite_classifier"):
                    classifier_rewrites.append((target_var, action["old_var"]))
                continue

            if base_condition is not None or not insert_missing_by_pattern_order:
                ast.define.definitions[target_var] = analyzed_condition
            else:
                self._insert_definition_by_pattern_order(
                    ast, target_var, analyzed_condition
                )

        if classifier_rewrites:
            self._rewrite_classifier_labels_many(ast, classifier_rewrites)

        # Repair every externally visible population once per original
        # variable, after all of its occurrence renames are known.  Repairing
        # one action at a time loses later fresh variables: after A is first
        # replaced by V1 in SUBSET U = (A), a second pass for V2 can no longer
        # find A and incorrectly leaves U = (V1).
        renamed_by_old: Dict[str, List[dict]] = {}
        for action in plan["actions"]:
            if action.get("new_var") is not None:
                renamed_by_old.setdefault(action["old_var"], []).append(action)

        # Install every pre-planned MEASURES subset before DEFINE repair is
        # allowed to generate additional names.  Otherwise a DEFINE subset
        # for one old variable could take a name reserved by a later group's
        # MEASURES action and then be overwritten.
        for old_var, renamed_actions in renamed_by_old.items():
            new_vars = [action["new_var"] for action in renamed_actions]
            self._rewrite_subsets_for_renamed_vars(ast, old_var, new_vars)

        for old_var, renamed_actions in renamed_by_old.items():
            new_vars = [action["new_var"] for action in renamed_actions]
            requested_measure_subset = next(
                (
                    action.get("measure_subset_var")
                    for action in renamed_actions
                    if action.get("measure_subset_var") is not None
                ),
                None,
            )
            shared_measure_subset = self._rewrite_measures_with_renamed_vars_subset(
                ast,
                old_var,
                new_vars,
                requested_measure_subset,
            )
            # DEFINE repair may reuse only a subset that was actually
            # installed.  The singleton case rewrites MEASURES directly.
            for action in renamed_actions:
                action["measure_subset_var"] = None
            if shared_measure_subset is not None:
                renamed_actions[0]["measure_subset_var"] = shared_measure_subset

        for old_var, renamed_actions in renamed_by_old.items():
            self._rewrite_define_refs_with_renamed_var_subset(
                ast,
                old_var,
                renamed_actions,
                parser,
                allow_direct_singleton=True,
            )

        for old_var in plan.get("maybe_removed_vars", set()):
            if not self._contains_var(ast.pattern, old_var):
                self._remove_definition(ast, old_var)

    def _rewrite_define_refs_with_renamed_var_subset(
        self,
        ast: RPRAbstractSyntaxTree,
        old_var: str,
        renamed_actions: List[dict],
        parser: ExprParser,
        *,
        allow_direct_singleton: bool = False,
    ) -> None:
        new_vars = {
            action["new_var"]
            for action in renamed_actions
            if action.get("new_var") is not None
        }
        referenced_by = []
        for def_var, condition in self._iter_defined_conditions(ast):
            # The original variable's condition and the copies installed for
            # its isolated occurrences describe individual candidate rows;
            # changing those self-references to a union variable would alter
            # their running/self semantics.
            if def_var == old_var or def_var in new_vars:
                continue
            if self._expr_references_any_var(condition.expression, {old_var}):
                referenced_by.append(def_var)

        if not referenced_by:
            return

        population_vars = []
        if self._contains_var(ast.pattern, old_var):
            population_vars.append(old_var)
        population_vars.extend(
            action["new_var"]
            for action in renamed_actions
            if action.get("new_var") is not None
        )
        population_vars = list(dict.fromkeys(population_vars))

        if allow_direct_singleton and len(population_vars) == 1:
            for def_var in referenced_by:
                condition = ast.define.definitions[def_var]
                rewritten = self._replace_var_qualifier(
                    condition.expression,
                    old_var,
                    population_vars[0],
                )
                ast.define.definitions[def_var] = parser._analyze_dependency(
                    def_var,
                    rewritten,
                )
            return

        # Reuse the exact union variable already created for MEASURES when
        # possible.  Otherwise create one containing every surviving piece of
        # the pre-split variable, and no unrelated pattern variables.
        subset_var = next(
            (
                action.get("measure_subset_var")
                for action in renamed_actions
                if action.get("measure_subset_var") is not None
            ),
            None,
        )
        if subset_var is None:
            subset_var = self._generate_new_var_name(
                ast,
                self._count_pattern_variables(ast.pattern),
            )
            if ast.subsets is None:
                ast.subsets = {}
            ast.subsets[subset_var] = population_vars

        for def_var in referenced_by:
            condition = ast.define.definitions[def_var]
            rewritten = self._replace_var_qualifier(
                condition.expression,
                old_var,
                subset_var,
            )
            ast.define.definitions[def_var] = parser._analyze_dependency(
                def_var,
                rewritten,
            )

    def _rewrite_subsets_for_renamed_var(
        self, ast: RPRAbstractSyntaxTree, old_var: str, new_var: str
    ) -> None:
        self._rewrite_subsets_for_renamed_vars(ast, old_var, [new_var])

    def _rewrite_subsets_for_renamed_vars(
        self, ast: RPRAbstractSyntaxTree, old_var: str, new_vars: List[str]
    ) -> None:
        if ast.subsets is None:
            return
        old_var_survives = self._contains_var(ast.pattern, old_var)
        new_subsets = {}
        for subset_name, members in ast.subsets.items():
            rewritten_members = []
            inserted = False
            for member in members:
                clean_member = member.strip()
                if clean_member == old_var:
                    if old_var_survives:
                        rewritten_members.append(clean_member)
                    rewritten_members.extend(new_vars)
                    inserted = True
                else:
                    rewritten_members.append(clean_member)
            if inserted:
                deduped_members = []
                seen = set()
                for member in rewritten_members:
                    member_name = member
                    if member_name in seen:
                        continue
                    seen.add(member_name)
                    deduped_members.append(member)
                new_subsets[subset_name] = deduped_members
            else:
                new_subsets[subset_name] = [member.strip() for member in members]
        ast.subsets = new_subsets

    def _rewrite_measures_with_renamed_var_subset(
        self,
        ast: RPRAbstractSyntaxTree,
        old_var: str,
        new_var: str,
        subset_var: str,
    ) -> None:
        self._rewrite_measures_with_renamed_vars_subset(
            ast, old_var, [new_var], subset_var
        )

    def _rewrite_measures_with_renamed_vars_subset(
        self,
        ast: RPRAbstractSyntaxTree,
        old_var: str,
        new_vars: List[str],
        subset_var: Optional[str],
    ) -> Optional[str]:
        if ast.measures is None or not self._measures_reference_var(ast, old_var):
            return None

        population_vars = []
        if self._contains_var(ast.pattern, old_var):
            population_vars.append(old_var)
        population_vars.extend(new_vars)
        population_vars = list(dict.fromkeys(population_vars))

        if len(population_vars) == 1:
            ast.measures.measures = {
                alias: self._replace_var_qualifier(expr, old_var, population_vars[0])
                for alias, expr in ast.measures.measures.items()
            }
            return None

        if subset_var is None:
            subset_var = self._generate_new_var_name(
                ast, self._count_pattern_variables(ast.pattern)
            )
        if ast.subsets is None:
            ast.subsets = {}
        ast.subsets[subset_var] = population_vars
        ast.measures.measures = {
            alias: self._replace_var_qualifier(expr, old_var, subset_var)
            for alias, expr in ast.measures.measures.items()
        }
        return subset_var

    def _rewrite_clauses_after_removed_vars(
        self,
        ast: RPRAbstractSyntaxTree,
        removed_vars: Set[str],
        parser: ExprParser,
    ) -> None:
        """Remove dead DEFINE entries and repair references to removed vars."""
        if not removed_vars:
            return

        singleton_subsets = {}
        empty_subsets = set()
        if ast.subsets is not None:
            new_subsets = {}
            for subset_name, members in ast.subsets.items():
                kept_members = [
                    member.strip()
                    for member in members
                    if member.strip() not in removed_vars
                ]
                if len(kept_members) == 1:
                    singleton_subsets[subset_name] = kept_members[0]
                elif len(kept_members) > 1:
                    new_subsets[subset_name] = kept_members
                else:
                    empty_subsets.add(subset_name)
            ast.subsets = new_subsets if new_subsets else None

        absent_vars = removed_vars | empty_subsets

        skip_target_var = self._skip_target_variable(ast)
        if skip_target_var in singleton_subsets:
            ast.after_match_skip_to = self._replace_skip_target_variable(
                ast.after_match_skip_to,
                skip_target_var,
                singleton_subsets[skip_target_var],
            )

        definitions = {}
        for var_name, condition in ast.define.definitions.items():
            if var_name in removed_vars:
                continue
            if condition is None or condition.expression is None:
                definitions[var_name] = condition
                continue
            rewritten = self._rewrite_measure_expr_after_removed_vars(
                condition.expression,
                absent_vars,
                singleton_subsets,
            )
            definitions[var_name] = parser._analyze_dependency(var_name, rewritten)
        ast.define.definitions = definitions

        if ast.measures is None:
            return
        ast.measures.measures = {
            alias: self._rewrite_measure_expr_after_removed_vars(
                expr, absent_vars, singleton_subsets
            )
            for alias, expr in ast.measures.measures.items()
        }

    def _skip_target_variable(self, ast: RPRAbstractSyntaxTree) -> Optional[str]:
        skip = (ast.after_match_skip_to or "").strip().upper()
        if skip in ("", "NEXT ROW", "PAST LAST ROW"):
            return None
        parts = skip.split()
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2 and parts[0] in ("FIRST", "LAST"):
            return parts[1]
        return None

    def _replace_skip_target_variable(
        self, skip: Optional[str], old_var: str, new_var: str
    ) -> Optional[str]:
        if skip is None:
            return None
        return re.sub(
            rf"(?<![A-Z0-9_$#]){re.escape(old_var)}(?![A-Z0-9_$#])",
            new_var,
            skip,
            flags=re.IGNORECASE,
        )

    def _removed_vars_allow_skip_rewrite(
        self, ast: RPRAbstractSyntaxTree, removed_vars: Set[str]
    ) -> bool:
        target = self._skip_target_variable(ast)
        if target is None or not removed_vars:
            return True
        if target in removed_vars:
            return False
        if ast.subsets is None or target not in ast.subsets:
            return True
        return any(member.strip() not in removed_vars for member in ast.subsets[target])

    def _sql_null_expr(self) -> Expr:
        return ColumnRef(None, "NULL")

    def _is_sql_null_expr(self, expr: Expr) -> bool:
        return (
            isinstance(expr, ColumnRef)
            and expr.qualifier is None
            and expr.name == "NULL"
        )

    def _rewrite_measure_expr_after_removed_vars(
        self,
        expr: Expr,
        removed_vars: Set[str],
        singleton_subsets: Dict[str, str],
    ) -> Expr:
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            rewritten = self._rewrite_measure_expr_after_removed_vars(
                expr.expr, removed_vars, singleton_subsets
            )
            return (
                rewritten
                if self._is_sql_null_expr(rewritten)
                else SemanticsExpr(expr.semantics, rewritten)
            )
        if isinstance(expr, ColumnRef):
            if expr.qualifier:
                qualifier_name = expr.qualifier
                if qualifier_name in removed_vars:
                    return self._sql_null_expr()
                if qualifier_name in singleton_subsets:
                    return ColumnRef(singleton_subsets[qualifier_name], expr.name)
            return ColumnRef(expr.qualifier, expr.name)
        if isinstance(expr, Literal):
            return Literal(expr.value)
        if isinstance(expr, FunctionCall):
            references_removed_var = self._expr_references_any_var(expr, removed_vars)
            rewritten_args = [
                self._rewrite_measure_expr_after_removed_vars(
                    arg, removed_vars, singleton_subsets
                )
                for arg in expr.args
            ]
            if references_removed_var:
                if expr.name == "COUNT":
                    return Literal(0)
                # These row-pattern operations cannot retain a reference to a
                # variable removed from PATTERN.  Their value over that empty
                # variable is NULL.  Ordinary scalar functions must instead
                # be kept: NVL(NULL, 999), for example, is 999.
                if expr.name in (
                    "SUM",
                    "AVG",
                    "MIN",
                    "MAX",
                    "PREV",
                    "NEXT",
                    "FIRST",
                    "LAST",
                ):
                    return self._sql_null_expr()
            return FunctionCall(expr.name, rewritten_args)
        if isinstance(expr, IsNullExpr):
            rewritten = self._rewrite_measure_expr_after_removed_vars(
                expr.expr, removed_vars, singleton_subsets
            )
            if self._is_sql_null_expr(rewritten):
                return Literal(1 if not expr.is_not else 0)
            return IsNullExpr(rewritten, expr.is_not)
        mapped = self._map_expr_children(
            expr,
            lambda child: self._rewrite_measure_expr_after_removed_vars(
                child, removed_vars, singleton_subsets
            ),
        )
        return mapped if mapped is not None else copy.deepcopy(expr)

    def _retarget_define_refs_to_renamed_occurrence(
        self,
        ast: RPRAbstractSyntaxTree,
        current_x_var: str,
        old_var: str,
        new_var: str,
        parser: ExprParser,
    ) -> None:
        for def_var, condition in self._iter_defined_conditions(ast):
            # The surviving old_var occurrences still need their original
            # self-DEFINE.  Replacing that qualifier with new_var makes a
            # later old_var row test the isolated earlier row instead of
            # itself.
            if def_var in {current_x_var, old_var, new_var}:
                continue
            if not self._expr_references_any_var(condition.expression, {old_var}):
                continue
            # The plan builder has proved that new_var either cannot precede
            # this DEFINE variable or precedes every one of its occurrences.
            # Retarget only in the latter case.
            if not self._can_var_occur_after_target(ast.pattern, def_var, new_var):
                continue
            rewritten = self._replace_var_qualifier(
                condition.expression, old_var, new_var
            )
            ast.define.definitions[def_var] = parser._analyze_dependency(
                def_var, rewritten
            )

    def _flatten_and(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]

    def _combine_and_terms(self, terms: List[Expr]) -> Expr:
        if not terms:
            return BinaryExpr(Literal(1), "=", Literal(1))
        expr = terms[0]
        for term in terms[1:]:
            expr = _simplify_logical_expr(expr, "AND", term)
        return expr

    def _iter_expr_children(self, node):
        """Yield direct children for every composite expression node."""
        if isinstance(node, SemanticsExpr):
            yield node.expr
        elif isinstance(node, FunctionCall):
            yield from node.args
        elif isinstance(node, BinaryExpr):
            yield node.left
            yield node.right
        elif isinstance(node, BetweenExpr):
            yield node.expr
            yield node.lower
            yield node.upper
        elif isinstance(node, InExpr):
            yield node.expr
            yield from node.values
        elif isinstance(node, (NotExpr, IsNullExpr)):
            yield node.expr
        elif isinstance(node, CaseExpr):
            for cond, result in node.cases:
                yield cond
                yield result
            if node.else_expr is not None:
                yield node.else_expr

    def _walk_expr(self, expr: Expr, prune=None):
        """Yield an expression tree in pre-order, optionally pruning nodes."""
        stack = [expr]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            yield node
            if prune is not None and prune(node):
                continue
            stack.extend(reversed(tuple(self._iter_expr_children(node))))

    def _column_reference_summary(
        self,
        expr: Expr,
        current_var: str,
    ) -> Tuple[Set[str], bool]:
        """Return other qualifiers and whether current/unqualified rows occur."""
        qualifiers: Set[str] = set()
        has_current_ref = False
        for node in self._walk_expr(expr):
            if not isinstance(node, ColumnRef):
                continue
            if node.qualifier is None or node.qualifier == current_var:
                has_current_ref = True
            else:
                qualifiers.add(node.qualifier)
        return qualifiers, has_current_ref

    def _map_expr_children(self, expr, fn) -> Expr:
        """Rebuild an expression node from children mapped through ``fn``.

        Handles every composite node kind used by this rule's recursive
        rewrites.  Leaf kinds (column references, literals, ...) return None so
        the caller keeps its own leaf handling.
        """
        if isinstance(expr, SemanticsExpr):
            return SemanticsExpr(expr.semantics, fn(expr.expr))
        if isinstance(expr, FunctionCall):
            return FunctionCall(expr.name, [fn(arg) for arg in expr.args])
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(fn(expr.left), expr.op, fn(expr.right))
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(fn(expr.expr), fn(expr.lower), fn(expr.upper))
        if isinstance(expr, InExpr):
            return InExpr(
                fn(expr.expr),
                [fn(value) for value in expr.values],
                expr.is_not,
            )
        if isinstance(expr, IsNullExpr):
            return IsNullExpr(fn(expr.expr), expr.is_not)
        if isinstance(expr, NotExpr):
            return NotExpr(fn(expr.expr))
        if isinstance(expr, CaseExpr):
            cases = [(fn(cond), fn(result)) for cond, result in expr.cases]
            else_expr = fn(expr.else_expr) if expr.else_expr is not None else None
            return CaseExpr(cases, else_expr)
        return None

    def _extract_single_pushable_var(
        self, expr: Expr, current_var: str
    ) -> Optional[str]:
        if self._has_dependent_func(expr):
            return None
        qualifiers, has_current_ref = self._column_reference_summary(expr, current_var)
        if has_current_ref or len(qualifiers) != 1:
            return None
        return next(iter(qualifiers))

    def _extract_single_bindable_var(
        self, expr: Expr, current_var: str
    ) -> Optional[str]:
        if self._has_blocking_dependent_func(expr):
            return None
        qualifiers, has_current_ref = self._column_reference_summary(expr, current_var)
        if not has_current_ref or len(qualifiers) != 1:
            return None
        return next(iter(qualifiers))

    def _extract_single_logical_nav_var(
        self, expr: Expr
    ) -> Optional[Tuple[str, str, int]]:
        result = self._extract_single_navigation_call(
            expr,
            {"FIRST", "LAST"},
            self._extract_logical_nav_target,
            allowed_opaque=self._is_count_variable_star,
        )
        if result is None:
            return None
        func_name, (ref_var, offset) = result
        return ref_var, func_name, offset

    def _is_count_variable_star(self, func: FunctionCall) -> bool:
        if func.name != "COUNT" or len(func.args) != 1:
            return False
        arg = self._unwrap_semantics(func.args[0])
        return (
            isinstance(arg, ColumnRef) and arg.qualifier is not None and arg.name == "*"
        )

    def _has_count_star_for_var(self, expr: Expr, var_name: str) -> bool:
        """Return whether ``expr`` contains COUNT(var_name.*).

        Isolating one occurrence of a multi-row pattern variable changes the
        set counted by the original variable, while changing the qualifier to
        the fresh variable reduces the count to that singleton occurrence.
        Both outcomes differ from COUNT over the pre-rewrite variable.
        """
        for node in self._walk_expr(expr):
            if not isinstance(node, FunctionCall):
                continue
            if not self._is_count_variable_star(node):
                continue
            arg = self._unwrap_semantics(node.args[0])
            if arg.qualifier == var_name:
                return True
        return False

    def _has_blocking_dependent_func(self, expr: Expr) -> bool:
        if expr is None:
            return False
        for func_name in self._DEPENDENT_FUNCTIONS - {"COUNT"}:
            if expr.has_func(func_name):
                return True
        if expr.has_func("COUNT"):
            return not self._all_count_calls_are_variable_star(expr)
        return False

    def _all_count_calls_are_variable_star(self, expr: Expr) -> bool:
        return all(
            not isinstance(node, FunctionCall)
            or node.name != "COUNT"
            or self._is_count_variable_star(node)
            for node in self._walk_expr(expr)
        )

    def _extract_presence_required_var(
        self, expr: Expr, current_var: str
    ) -> Optional[str]:
        if self._has_dependent_func(expr) or self._is_nullable_null_check_expr(expr):
            return None
        qualifiers, _ = self._column_reference_summary(expr, current_var)
        if len(qualifiers) != 1:
            return None
        required_var = next(iter(qualifiers))
        if not self._is_provably_null_rejecting(expr, required_var):
            return None
        return required_var

    def _is_provably_null_rejecting(self, expr: Expr, var_name: str) -> bool:
        """Return True only when an absent ``var_name`` cannot make expr TRUE.

        This is intentionally a proof, not a guess.  Oracle gives an ordinary
        reference to an unmatched pattern variable the value NULL.  Operators
        below are handled only where SQL three-valued logic proves rejection;
        function calls and CASE are left unproven because they can replace or
        otherwise inspect NULL (NVL/COALESCE are the important examples).
        """
        if isinstance(expr, SemanticsExpr):
            return self._is_provably_null_rejecting(expr.expr, var_name)

        if isinstance(expr, BinaryExpr):
            if expr.op == "AND":
                return self._is_provably_null_rejecting(
                    expr.left, var_name
                ) or self._is_provably_null_rejecting(expr.right, var_name)
            if expr.op == "OR":
                return self._is_provably_null_rejecting(
                    expr.left, var_name
                ) and self._is_provably_null_rejecting(expr.right, var_name)
            if expr.op in ("=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "NOT LIKE"):
                return self._is_guaranteed_null_when_var_absent(
                    expr.left, var_name
                ) or self._is_guaranteed_null_when_var_absent(expr.right, var_name)
            return False

        if isinstance(expr, BetweenExpr):
            return self._is_guaranteed_null_when_var_absent(expr.expr, var_name)

        if isinstance(expr, InExpr):
            return self._is_guaranteed_null_when_var_absent(expr.expr, var_name)

        if isinstance(expr, IsNullExpr):
            return expr.is_not and self._is_guaranteed_null_when_var_absent(
                expr.expr, var_name
            )

        return False

    def _is_guaranteed_null_when_var_absent(self, expr: Expr, var_name: str) -> bool:
        """Recognize expressions whose value must be NULL when var is absent."""
        if isinstance(expr, SemanticsExpr):
            return self._is_guaranteed_null_when_var_absent(expr.expr, var_name)
        if isinstance(expr, ColumnRef):
            return expr.qualifier == var_name
        # Oracle treats NULL as an empty string for concatenation, so ``||``
        # is deliberately not included among the NULL-propagating operators.
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*", "/", "%"):
            return self._is_guaranteed_null_when_var_absent(
                expr.left, var_name
            ) or self._is_guaranteed_null_when_var_absent(expr.right, var_name)
        return False

    def _has_dependent_func(self, expr: Expr) -> bool:
        if expr is None:
            return False
        return any(expr.has_func(name) for name in self._DEPENDENT_FUNCTIONS)

    def _is_nullable_null_check_expr(
        self,
        expr: Expr,
        var_name: Optional[str] = None,
    ) -> bool:
        """Recognize an IS NULL predicate safe for an unmatched-variable arm.

        With no variable name this remains the syntactic check used to keep
        presence inference away from IS NULL.  For the nullable-pattern split,
        however, the empty arm is valid only when the inner expression is
        provably NULL if that variable has no mapped rows.  COALESCE/NVL and
        unknown scalar functions deliberately fail that proof.
        """
        if isinstance(expr, SemanticsExpr):
            return self._is_nullable_null_check_expr(expr.expr, var_name)
        if isinstance(expr, IsNullExpr):
            if expr.is_not:
                return False
            return var_name is None or self._is_guaranteed_null_when_var_absent(
                expr.expr, var_name
            )
        return False

    def _extract_first_null_check(self, expr: Expr) -> Optional[Tuple[str, str]]:
        if isinstance(expr, SemanticsExpr):
            return self._extract_first_null_check(expr.expr)
        if not isinstance(expr, IsNullExpr) or expr.is_not:
            return None
        inner = self._unwrap_semantics(expr.expr)
        if not isinstance(inner, FunctionCall) or inner.name != "FIRST":
            return None
        if len(inner.args) == 0 or len(inner.args) > 2:
            return None
        if len(inner.args) == 2:
            offset = self._unwrap_semantics(inner.args[1])
            if not isinstance(offset, Literal) or offset.value != 0:
                return None
        arg = self._unwrap_semantics(inner.args[0])
        if not isinstance(arg, ColumnRef) or arg.qualifier is None:
            return None
        return arg.qualifier, arg.name

    def _find_nullable_split_alternative_vars(
        self, node, target_var: str, before_var: str
    ) -> List[str]:
        if not isinstance(node, ConcatenationPattern):
            return []
        for idx, child in enumerate(node.patterns):
            if self._contains_var(child, before_var):
                return []
            if not isinstance(child, AlternationPattern) or not self._contains_var(
                child, target_var
            ):
                continue
            if not any(
                self._contains_var(part, before_var)
                for part in node.patterns[idx + 1 :]
            ):
                continue
            alternatives = []
            has_target_branch = False
            for branch in child.patterns:
                if isinstance(branch, VariablePattern):
                    if branch.name == target_var:
                        has_target_branch = True
                    else:
                        alternatives.append(branch.name)
                    continue
                return []
            return alternatives if has_target_branch else []
        return []

    def _distribute_alternation_before_target(
        self, node, target_var: str, before_var: str
    ):
        if not isinstance(node, ConcatenationPattern):
            return None
        for idx, child in enumerate(node.patterns):
            if self._contains_var(child, before_var):
                return None
            if not isinstance(child, AlternationPattern) or not self._contains_var(
                child, target_var
            ):
                continue
            if not any(
                self._contains_var(part, before_var)
                for part in node.patterns[idx + 1 :]
            ):
                continue

            distributed = []
            prefix = node.patterns[:idx]
            suffix = node.patterns[idx + 1 :]
            for branch in child.patterns:
                if not isinstance(branch, VariablePattern):
                    return None
                parts = prefix + [branch] + suffix
                distributed.append(ConcatenationPattern(parts).simplify())
            if not any(
                isinstance(branch, VariablePattern) and branch.name == target_var
                for branch in child.patterns
            ):
                return None
            return AlternationPattern(distributed)
        return None

    def _split_nullable_quantifier_before_target(
        self,
        node,
        target_var: str,
        before_var: str,
        ast: RPRAbstractSyntaxTree,
    ):
        if not isinstance(node, ConcatenationPattern):
            return None

        pattern_counts = self._count_pattern_variables(node)
        new_var = self._generate_new_var_name(ast, pattern_counts)

        for idx, child in enumerate(node.patterns):
            if self._contains_var(child, before_var):
                return None
            if not any(
                self._contains_var(part, before_var)
                for part in node.patterns[idx + 1 :]
            ):
                continue
            if not (
                isinstance(child, QuantifiedPattern)
                and isinstance(child.pattern, VariablePattern)
                and child.pattern.name == target_var
                and child.atLeast == 0
                and (child.atMost is None or child.atMost > 0)
            ):
                continue

            prefix_parts = []
            branch_max = None if child.atMost is None else child.atMost - 1
            if branch_max is None or branch_max > 0:
                prefix_parts.append(
                    QuantifiedPattern(
                        VariablePattern(child.pattern.name),
                        0,
                        branch_max,
                        child.isGreedy,
                    )
                )
            prefix_parts.append(VariablePattern(new_var))
            present_branch = ConcatenationPattern(prefix_parts).simplify()
            optional_branch = AlternationPattern(
                [present_branch, EmptyPattern()]
                if child.isGreedy
                else [EmptyPattern(), present_branch]
            )
            rewritten = ConcatenationPattern(
                node.patterns[:idx] + [optional_branch] + node.patterns[idx + 1 :]
            )
            return new_var, rewritten, child.pattern.name

        return None

    def _collect_same_var_push_group_indices(
        self,
        conjuncts: List[Expr],
        seed_idx: int,
        current_var: str,
        y_var: str,
    ) -> Optional[Set[int]]:
        group = {seed_idx}
        for idx, conjunct in enumerate(conjuncts):
            if idx == seed_idx:
                continue
            if not self._expr_references_any_var(conjunct, {y_var}):
                continue
            if not self._is_same_var_physical_push_expr(conjunct, current_var, y_var):
                return None
            group.add(idx)
        return group

    def _is_same_var_physical_push_expr(
        self, expr: Expr, current_var: str, y_var: str
    ) -> bool:
        blockers = self._DEPENDENT_FUNCTIONS - {"PREV"}
        if any(
            isinstance(node, FunctionCall) and node.name in blockers
            for node in self._walk_expr(expr)
        ):
            return False
        qualifiers, has_current_ref = self._column_reference_summary(expr, current_var)
        return not has_current_ref and qualifiers == {y_var}

    def _remove_definition(self, ast: RPRAbstractSyntaxTree, target_var: str) -> None:
        ast.define.definitions = {
            var_name: condition
            for var_name, condition in ast.define.definitions.items()
            if var_name != target_var
        }

    def _insert_definition_before(
        self, ast: RPRAbstractSyntaxTree, before_var: str, new_var: str, new_condition
    ) -> None:
        self._insert_definition_when(
            ast,
            new_var,
            new_condition,
            lambda var_name: var_name == before_var,
        )

    def _insert_definition_by_pattern_order(
        self, ast: RPRAbstractSyntaxTree, new_var: str, new_condition
    ) -> None:
        order = self._pattern_var_first_order(ast.pattern)
        new_pos = order.get(new_var)
        if new_pos is None:
            ast.define.definitions[new_var] = new_condition
            return

        self._insert_definition_when(
            ast,
            new_var,
            new_condition,
            lambda var_name: (
                order.get(var_name) is not None and order[var_name] > new_pos
            ),
        )

    def _insert_definition_when(
        self,
        ast: RPRAbstractSyntaxTree,
        new_var: str,
        new_condition,
        insert_before,
    ) -> None:
        """Insert a DEFINE entry before the first key matching a predicate."""
        new_defs = {}
        inserted = False
        for var_name, condition in ast.define.definitions.items():
            if not inserted and insert_before(var_name):
                new_defs[new_var] = new_condition
                inserted = True
            new_defs[var_name] = condition
        if not inserted:
            new_defs[new_var] = new_condition
        ast.define.definitions = new_defs

    def _iter_pattern_leaf_variables(self, node):
        """Yield the VariablePattern leaves of a pattern node in pre-order.

        Shared by this rule's set/count/order scans so the structural
        recursion over pattern node kinds lives in exactly one place.
        """
        if isinstance(node, VariablePattern):
            yield node
        elif isinstance(node, QuantifiedPattern):
            yield from self._iter_pattern_leaf_variables(node.pattern)
        elif isinstance(node, (ExcludedPattern, AnchorPattern)):
            yield from self._iter_pattern_leaf_variables(node.pattern)
        elif isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            for child in node.patterns:
                yield from self._iter_pattern_leaf_variables(child)

    def _rewrap_unary_pattern(self, wrapper, inner):
        """Rebuild an excluded/anchored pattern around a rewritten child."""
        if inner is None:
            return None
        if isinstance(wrapper, ExcludedPattern):
            return ExcludedPattern(inner)
        return AnchorPattern(inner, wrapper.anchor_type)

    def _pattern_var_first_order(self, pattern) -> Dict[str, int]:
        order: Dict[str, int] = {}
        for leaf in self._iter_pattern_leaf_variables(pattern):
            name = leaf.name
            if name not in order:
                order[name] = len(order)
        return order

    def _merge_into_var_expr(
        self, base_expr: Optional[Expr], pushed_expr: Expr
    ) -> Expr:
        if base_expr is None:
            return copy.deepcopy(pushed_expr)
        return _simplify_logical_expr(
            copy.deepcopy(base_expr), "AND", copy.deepcopy(pushed_expr)
        )

    def _replace_var_qualifier(self, expr: Expr, old_var: str, new_var: str) -> Expr:
        if expr is None:
            return None
        if isinstance(expr, ColumnRef):
            if expr.qualifier and expr.qualifier == old_var:
                return ColumnRef(new_var, expr.name)
            return ColumnRef(expr.qualifier, expr.name)
        if isinstance(expr, Literal):
            return Literal(expr.value)
        mapped = self._map_expr_children(
            expr,
            lambda child: self._replace_var_qualifier(child, old_var, new_var),
        )
        return mapped if mapped is not None else copy.deepcopy(expr)

    def _replace_logical_nav_with_column(
        self,
        expr: Expr,
        old_var: str,
        new_var: str,
        func_name: str,
        offset: int,
    ) -> Expr:
        if expr is None:
            return None
        if isinstance(expr, FunctionCall):
            if expr.name == func_name:
                target = self._extract_logical_nav_target(expr)
                arg = self._unwrap_semantics(expr.args[0]) if expr.args else None
                if (
                    target is not None
                    and target[0] == old_var
                    and target[1] == offset
                    and isinstance(arg, ColumnRef)
                ):
                    return ColumnRef(new_var, arg.name)
            return FunctionCall(
                expr.name,
                [
                    self._replace_logical_nav_with_column(
                        arg, old_var, new_var, func_name, offset
                    )
                    for arg in expr.args
                ],
            )
        if isinstance(expr, ColumnRef):
            return ColumnRef(expr.qualifier, expr.name)
        if isinstance(expr, Literal):
            return Literal(expr.value)
        mapped = self._map_expr_children(
            expr,
            lambda child: self._replace_logical_nav_with_column(
                child, old_var, new_var, func_name, offset
            ),
        )
        return mapped if mapped is not None else copy.deepcopy(expr)

    def _require_var_to_appear(self, node, target_var: str):
        if isinstance(node, VariablePattern):
            return node if node.name == target_var else None
        if isinstance(node, QuantifiedPattern):
            if not self._contains_var(node.pattern, target_var):
                return None

            # Requiring ``target_var`` somewhere in a repetition is not the
            # same as requiring it in every repetition.  For example, the
            # A-containing part of ``(A | B){0,2}`` is ``A | A A | A B | B A``;
            # recursively filtering the body and then changing the lower
            # bound to one would produce only ``A | A A``.
            #
            # Preserve the body unchanged when it already guarantees the
            # target.  A nullable single repetition can also be filtered
            # safely because there is only one body occurrence.  General
            # repeated optional bodies need a priority-preserving automaton
            # construction, so leave those patterns untouched rather than
            # narrowing their match language.
            if node.atMost is not None and node.atMost <= 0:
                return None
            if target_var in node.pattern.get_must_appear_var():
                return QuantifiedPattern(
                    node.pattern, max(1, node.atLeast), node.atMost, node.isGreedy
                )
            if node.atMost != 1:
                return None

            inner = self._require_var_to_appear(node.pattern, target_var)
            if inner is None:
                return None
            return QuantifiedPattern(inner, 1, 1, node.isGreedy)
        if isinstance(node, AlternationPattern):
            kept = []
            for child in node.patterns:
                if not self._contains_var(child, target_var):
                    continue
                rewritten = self._require_var_to_appear(child, target_var)
                if rewritten is not None:
                    kept.append(rewritten)
            if not kept:
                return None
            if len(kept) == 1:
                return kept[0]
            return AlternationPattern(kept)
        if isinstance(node, ConcatenationPattern):
            target_indices = [
                idx
                for idx, child in enumerate(node.patterns)
                if self._contains_var(child, target_var)
            ]
            if not target_indices:
                return None

            first_idx = target_indices[0]
            if len(target_indices) == 1:
                rewritten = self._require_var_to_appear(
                    node.patterns[first_idx], target_var
                )
                if rewritten is None:
                    return None
                new_patterns = list(node.patterns)
                new_patterns[first_idx] = rewritten
                return ConcatenationPattern(new_patterns)

            first_child = node.patterns[first_idx]
            if not (
                isinstance(first_child, QuantifiedPattern)
                and isinstance(first_child.pattern, VariablePattern)
                and first_child.pattern.name == target_var
                and first_child.atLeast == 0
                and (first_child.atMost is None or first_child.atMost > 0)
            ):
                # Requiring a variable across several compound occurrences
                # needs both the with-variable and without-variable language
                # of the first child.  Leave unsupported shapes unchanged
                # instead of arbitrarily forcing one syntactic occurrence.
                return None

            last_idx = target_indices[-1]
            decision_suffix = node.patterns[first_idx + 1 : last_idx + 1]
            common_prefix = node.patterns[:first_idx]
            common_suffix = node.patterns[last_idx + 1 :]

            present_child = QuantifiedPattern(
                VariablePattern(target_var),
                1,
                first_child.atMost,
                first_child.isGreedy,
            ).simplify()
            present_branch = ConcatenationPattern(
                [present_child] + decision_suffix
            ).simplify()

            fallback_input = (
                decision_suffix[0]
                if len(decision_suffix) == 1
                else ConcatenationPattern(decision_suffix)
            )
            absent_branch = self._require_var_to_appear(fallback_input, target_var)
            if absent_branch is None:
                return None

            branches = (
                [present_branch, absent_branch]
                if first_child.isGreedy
                else [absent_branch, present_branch]
            )
            required_region = AlternationPattern(branches).simplify()
            return ConcatenationPattern(
                common_prefix + [required_region] + common_suffix
            )
        if isinstance(node, ExcludedPattern):
            inner = self._require_var_to_appear(node.pattern, target_var)
            return ExcludedPattern(inner) if inner is not None else None
        if isinstance(node, AnchorPattern):
            inner = self._require_var_to_appear(node.pattern, target_var)
            return AnchorPattern(inner, node.anchor_type) if inner is not None else None
        return None

    def _contains_var(self, node, var_name: str) -> bool:
        return var_name in self._collect_all_vars(node)

    def _collect_all_vars(self, node) -> Set[str]:
        return {leaf.name for leaf in self._iter_pattern_leaf_variables(node)}

    def _max_occurrence_count(self, node, var_name: str) -> Optional[int]:
        if isinstance(node, VariablePattern):
            return 1 if node.name == var_name else 0
        if isinstance(node, QuantifiedPattern):
            inner = self._max_occurrence_count(node.pattern, var_name)
            if inner is None:
                return None
            if inner == 0:
                return 0
            if node.atMost is None:
                return None
            return inner * node.atMost
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._max_occurrence_count(node.pattern, var_name)
        if isinstance(node, ConcatenationPattern):
            total = 0
            for child in node.patterns:
                child_count = self._max_occurrence_count(child, var_name)
                if child_count is None:
                    return None
                total += child_count
            return total
        if isinstance(node, AlternationPattern):
            maximum = 0
            for child in node.patterns:
                child_count = self._max_occurrence_count(child, var_name)
                if child_count is None:
                    return None
                maximum = max(maximum, child_count)
            return maximum
        if isinstance(node, PermutationPattern):
            total = 0
            for child in node.patterns:
                child_count = self._max_occurrence_count(child, var_name)
                if child_count is None:
                    return None
                total += child_count
            return total
        return 0

    def _rename_logical_occurrence_before_target(
        self,
        node,
        source_var: str,
        target_var: str,
        new_var: str,
        func_name: str,
        offset: int,
    ):
        if offset < 0:
            return None
        if offset == 0:
            if func_name == "FIRST":
                return self._rename_first_occurrence_before_target(
                    node, source_var, target_var, new_var
                )
            return self._rename_last_occurrence_before_target(
                node, source_var, target_var, new_var
            )

        if isinstance(node, ConcatenationPattern):
            new_patterns = list(node.patterns)
            indices = (
                range(len(node.patterns))
                if func_name == "FIRST"
                else range(len(node.patterns) - 1, -1, -1)
            )
            for idx in indices:
                child = node.patterns[idx]
                if self._contains_var(child, target_var):
                    if func_name == "FIRST":
                        return None
                    if self._contains_var(child, source_var):
                        renamed_child = self._rename_logical_occurrence_before_target(
                            child, source_var, target_var, new_var, func_name, offset
                        )
                        if renamed_child is not None:
                            new_patterns[idx] = renamed_child
                            return ConcatenationPattern(new_patterns)
                    continue

                if not self._contains_var(child, source_var):
                    continue
                if not any(
                    self._contains_var(part, target_var)
                    for part in node.patterns[idx + 1 :]
                ):
                    continue
                renamed_child = self._rename_logical_occurrence_in_node(
                    child, source_var, new_var, func_name, offset
                )
                if renamed_child is None:
                    return None
                new_patterns[idx] = renamed_child
                return ConcatenationPattern(new_patterns)
            return None

        if isinstance(node, AlternationPattern):
            new_patterns = []
            changed = False
            for child in node.patterns:
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    renamed_child = self._rename_logical_occurrence_before_target(
                        child, source_var, target_var, new_var, func_name, offset
                    )
                    if renamed_child is None:
                        return None
                    new_patterns.append(renamed_child)
                    changed = True
                else:
                    new_patterns.append(child)
            return AlternationPattern(new_patterns) if changed else None

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            renamed_inner = self._rename_logical_occurrence_before_target(
                node.pattern, source_var, target_var, new_var, func_name, offset
            )
            return self._rewrap_unary_pattern(node, renamed_inner)

        return None

    def _rename_logical_occurrence_in_node(
        self,
        node,
        source_var: str,
        new_var: str,
        func_name: str,
        offset: int,
    ):
        if isinstance(node, QuantifiedPattern):
            return self._split_quantified_variable_for_logical_nav(
                node, source_var, new_var, func_name, offset
            )
        if isinstance(node, ConcatenationPattern):
            patterns = list(node.patterns)
            indices = (
                range(len(patterns))
                if func_name == "FIRST"
                else range(len(patterns) - 1, -1, -1)
            )
            for idx in indices:
                if not self._contains_var(patterns[idx], source_var):
                    continue
                renamed_child = self._rename_logical_occurrence_in_node(
                    patterns[idx], source_var, new_var, func_name, offset
                )
                if renamed_child is None:
                    return None
                patterns[idx] = renamed_child
                return ConcatenationPattern(patterns)
            return None
        if isinstance(node, AlternationPattern):
            containing = [
                child
                for child in node.patterns
                if self._contains_var(child, source_var)
            ]
            if len(containing) != 1:
                return None
            new_patterns = []
            for child in node.patterns:
                if self._contains_var(child, source_var):
                    renamed_child = self._rename_logical_occurrence_in_node(
                        child, source_var, new_var, func_name, offset
                    )
                    if renamed_child is None:
                        return None
                    new_patterns.append(renamed_child)
                else:
                    new_patterns.append(child)
            return AlternationPattern(new_patterns)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            renamed_inner = self._rename_logical_occurrence_in_node(
                node.pattern, source_var, new_var, func_name, offset
            )
            return self._rewrap_unary_pattern(node, renamed_inner)
        return None

    def _split_quantified_variable_for_logical_nav(
        self,
        node: QuantifiedPattern,
        source_var: str,
        new_var: str,
        func_name: str,
        offset: int,
    ):
        if (
            not isinstance(node.pattern, VariablePattern)
            or node.pattern.name != source_var
        ):
            return None
        required = offset + 1
        if node.atMost is not None and node.atMost < required:
            return None

        renamed = VariablePattern(new_var)

        def bounded_repeat(count: int):
            if count <= 0:
                return None
            if count == 1:
                return VariablePattern(source_var)
            return QuantifiedPattern(
                VariablePattern(source_var), count, count, node.isGreedy
            )

        def ranged_repeat(min_count: int, max_count: Optional[int]):
            if max_count is not None and max_count <= 0:
                return None
            if max_count is not None and min_count > max_count:
                return None
            min_count = max(0, min_count)
            if max_count == 1 and min_count == 1:
                return VariablePattern(source_var)
            return QuantifiedPattern(
                VariablePattern(source_var), min_count, max_count, node.isGreedy
            )

        parts = []
        if func_name == "FIRST":
            fixed_prefix = bounded_repeat(offset)
            if fixed_prefix is not None:
                parts.append(fixed_prefix)
            parts.append(renamed)
            suffix_min = max(0, node.atLeast - required)
            suffix_max = None if node.atMost is None else node.atMost - required
            suffix = ranged_repeat(suffix_min, suffix_max)
            if suffix is not None:
                parts.append(suffix)
        else:
            prefix_min = max(0, node.atLeast - required)
            prefix_max = None if node.atMost is None else node.atMost - required
            prefix = ranged_repeat(prefix_min, prefix_max)
            if prefix is not None:
                parts.append(prefix)
            parts.append(renamed)
            fixed_suffix = bounded_repeat(offset)
            if fixed_suffix is not None:
                parts.append(fixed_suffix)

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return ConcatenationPattern(parts).simplify()

    def _rename_last_occurrence_before_target(
        self, node, source_var: str, target_var: str, new_var: str
    ):
        if isinstance(node, ConcatenationPattern):
            new_patterns = list(node.patterns)
            for idx in range(len(node.patterns) - 1, -1, -1):
                child = node.patterns[idx]
                later_has_target = any(
                    self._contains_var(p, target_var) for p in node.patterns[idx + 1 :]
                )
                if later_has_target and self._contains_var(child, source_var):
                    if isinstance(child, AlternationPattern) and all(
                        self._contains_var(branch, source_var)
                        for branch in child.patterns
                    ):
                        renamed_child = (
                            self._rename_last_occurrence_in_each_alternative(
                                child, source_var, new_var
                            )
                        )
                        if renamed_child is not None:
                            new_patterns[idx] = renamed_child
                            return ConcatenationPattern(new_patterns)
                    renamed_prefix = self._rename_last_occurrence_in_prefix(
                        node.patterns[: idx + 1], source_var, new_var
                    )
                    if renamed_prefix is not None:
                        return ConcatenationPattern(
                            [renamed_prefix] + node.patterns[idx + 1 :]
                        )
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    renamed_child = self._rename_last_occurrence_before_target(
                        child, source_var, target_var, new_var
                    )
                    if renamed_child is not None:
                        new_patterns[idx] = renamed_child
                        return ConcatenationPattern(new_patterns)
            return None
        if isinstance(node, QuantifiedPattern):
            if not self._contains_var(
                node.pattern, source_var
            ) or not self._contains_var(node.pattern, target_var):
                return None
            renamed_inner = self._rename_last_occurrence_before_target(
                node.pattern, source_var, target_var, new_var
            )
            if renamed_inner is None:
                return None
            return QuantifiedPattern(
                renamed_inner, node.atLeast, node.atMost, node.isGreedy
            )
        if isinstance(node, AlternationPattern):
            new_patterns = []
            changed = False
            for child in node.patterns:
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    renamed_child = self._rename_last_occurrence_before_target(
                        child, source_var, target_var, new_var
                    )
                    if renamed_child is None:
                        return None
                    new_patterns.append(renamed_child)
                    changed = True
                else:
                    new_patterns.append(child)
            return AlternationPattern(new_patterns) if changed else None
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            renamed_inner = self._rename_last_occurrence_before_target(
                node.pattern, source_var, target_var, new_var
            )
            return self._rewrap_unary_pattern(node, renamed_inner)
        return None

    def _rename_last_occurrence_in_each_alternative(
        self, node: AlternationPattern, source_var: str, new_var: str
    ):
        renamed_patterns = []
        for child in node.patterns:
            if not self._contains_var(child, source_var):
                return None
            renamed_child = self._rename_last_occurrence(child, source_var, new_var)
            if renamed_child is None:
                return None
            renamed_patterns.append(renamed_child)
        return AlternationPattern(renamed_patterns)

    def _rename_first_occurrence_before_target(
        self, node, source_var: str, target_var: str, new_var: str
    ):
        if isinstance(node, ConcatenationPattern):
            for target_idx, child in enumerate(node.patterns):
                if not self._contains_var(child, target_var):
                    continue

                prefix = node.patterns[:target_idx]
                if any(self._contains_var(part, source_var) for part in prefix):
                    renamed_prefix = self._rename_first_occurrence_in_suffix(
                        prefix, source_var, new_var
                    )
                    if renamed_prefix is None:
                        return None
                    return ConcatenationPattern(
                        [renamed_prefix] + node.patterns[target_idx:]
                    )

                if self._contains_var(child, source_var):
                    renamed_child = self._rename_first_occurrence_before_target(
                        child, source_var, target_var, new_var
                    )
                    if renamed_child is not None:
                        new_patterns = list(node.patterns)
                        new_patterns[target_idx] = renamed_child
                        return ConcatenationPattern(new_patterns)
                return None
            return None
        if isinstance(node, QuantifiedPattern):
            if not self._contains_var(
                node.pattern, source_var
            ) or not self._contains_var(node.pattern, target_var):
                return None
            renamed_inner = self._rename_first_occurrence_before_target(
                node.pattern, source_var, target_var, new_var
            )
            if renamed_inner is None:
                return None
            return QuantifiedPattern(
                renamed_inner, node.atLeast, node.atMost, node.isGreedy
            )
        if isinstance(node, AlternationPattern):
            new_patterns = []
            changed = False
            for child in node.patterns:
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    renamed_child = self._rename_first_occurrence_before_target(
                        child, source_var, target_var, new_var
                    )
                    if renamed_child is None:
                        return None
                    new_patterns.append(renamed_child)
                    changed = True
                else:
                    new_patterns.append(child)
            return AlternationPattern(new_patterns) if changed else None
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            renamed_inner = self._rename_first_occurrence_before_target(
                node.pattern, source_var, target_var, new_var
            )
            return self._rewrap_unary_pattern(node, renamed_inner)
        return None

    def _rename_first_occurrence_in_suffix(
        self, patterns: List, source_var: str, new_var: str
    ):
        """Rename the first actual source occurrence using a local split.

        For a nullable occurrence, its non-empty branch owns the first source
        row, while its empty branch falls through to a later occurrence.  Only
        the interval through that later occurrence is placed in an ordered
        alternation; unrelated prefix and suffix structure stays factored.
        """
        for idx, child in enumerate(patterns):
            if not self._contains_var(child, source_var):
                continue

            prefix = patterns[:idx]
            suffix = patterns[idx + 1 :]
            if isinstance(child, AlternationPattern):
                branches_with_source = [
                    self._contains_var(branch, source_var) for branch in child.patterns
                ]
                if not all(branches_with_source) and any(
                    self._contains_var(part, source_var) for part in suffix
                ):
                    renamed_suffix = self._rename_first_occurrence_in_suffix(
                        suffix, source_var, new_var
                    )
                    if renamed_suffix is None:
                        return None
                    options = []
                    for branch, has_source in zip(child.patterns, branches_with_source):
                        if has_source:
                            renamed_branch = self._rename_first_occurrence(
                                branch, source_var, new_var
                            )
                            if renamed_branch is None:
                                return None
                            option = ConcatenationPattern(
                                [renamed_branch] + suffix
                            ).simplify()
                        else:
                            option = ConcatenationPattern(
                                [branch, renamed_suffix]
                            ).simplify()
                        options.append(option)
                    split = AlternationPattern(options).simplify()
                    return ConcatenationPattern(prefix + [split]).simplify()

            if (
                isinstance(child, QuantifiedPattern)
                and isinstance(child.pattern, VariablePattern)
                and child.pattern.name == source_var
                and child.atLeast == 0
                and (child.atMost is None or child.atMost > 0)
            ):
                later_source_indices = [
                    pos
                    for pos, part in enumerate(suffix)
                    if self._contains_var(part, source_var)
                ]
                if later_source_indices:
                    last_source_idx = later_source_indices[-1]
                    decision_suffix = suffix[: last_source_idx + 1]
                    common_suffix = suffix[last_source_idx + 1 :]
                else:
                    decision_suffix = []
                    common_suffix = suffix

                nonempty = QuantifiedPattern(
                    VariablePattern(source_var),
                    1,
                    child.atMost,
                    child.isGreedy,
                )
                renamed_nonempty = self._rename_first_occurrence(
                    nonempty, source_var, new_var
                )
                renamed_empty_fallback = self._rename_first_occurrence_in_suffix(
                    decision_suffix, source_var, new_var
                )
                if renamed_nonempty is None:
                    return None

                nonempty_branch = ConcatenationPattern(
                    [renamed_nonempty] + decision_suffix
                ).simplify()
                if renamed_empty_fallback is None:
                    split = nonempty_branch
                else:
                    branches = (
                        [nonempty_branch, renamed_empty_fallback]
                        if child.isGreedy
                        else [renamed_empty_fallback, nonempty_branch]
                    )
                    split = AlternationPattern(branches).simplify()

                return ConcatenationPattern(prefix + [split] + common_suffix).simplify()

            renamed_child = self._rename_first_occurrence(child, source_var, new_var)
            if renamed_child is None:
                return None
            return ConcatenationPattern(prefix + [renamed_child] + suffix).simplify()
        return None

    def _rename_first_occurrence(self, node, source_var: str, new_var: str):
        return self._rename_edge_occurrence(node, source_var, new_var, first=True)

    def _rename_edge_occurrence(
        self,
        node,
        source_var: str,
        new_var: str,
        *,
        first: bool,
    ):
        """Rename the first or last actual occurrence in one pattern node."""
        if isinstance(node, VariablePattern):
            if node.name != source_var:
                return None
            return VariablePattern(new_var)
        if isinstance(node, QuantifiedPattern):
            if (
                not isinstance(node.pattern, VariablePattern)
                or node.pattern.name != source_var
            ):
                return None
            # The caller must decide whether the empty branch falls through
            # to a later source occurrence or is impossible for the predicate.
            if node.atLeast == 0:
                return None
            if node.atMost == 1:
                return VariablePattern(new_var)
            remainder_min = max(0, node.atLeast - 1)
            remainder_max = None if node.atMost is None else node.atMost - 1
            remainder = None
            if remainder_max is None or remainder_max > 0:
                if remainder_min == 1 and remainder_max == 1:
                    remainder = VariablePattern(source_var)
                else:
                    remainder = QuantifiedPattern(
                        VariablePattern(source_var),
                        remainder_min,
                        remainder_max,
                        node.isGreedy,
                    )
            if remainder is None:
                return VariablePattern(new_var)
            parts = (
                [VariablePattern(new_var), remainder]
                if first
                else [remainder, VariablePattern(new_var)]
            )
            return ConcatenationPattern(parts).simplify()
        if isinstance(node, ConcatenationPattern):
            rename_sequence = (
                self._rename_first_occurrence_in_suffix
                if first
                else self._rename_last_occurrence_in_prefix
            )
            return rename_sequence(list(node.patterns), source_var, new_var)
        if isinstance(node, AlternationPattern):
            containing = [
                child
                for child in node.patterns
                if self._contains_var(child, source_var)
            ]
            if len(containing) != 1:
                return None
            new_patterns = []
            changed = False
            for child in node.patterns:
                if self._contains_var(child, source_var):
                    renamed_child = self._rename_edge_occurrence(
                        child, source_var, new_var, first=first
                    )
                    if renamed_child is None:
                        return None
                    new_patterns.append(renamed_child)
                    changed = True
                else:
                    new_patterns.append(child)
            return AlternationPattern(new_patterns) if changed else None
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            renamed_inner = self._rename_edge_occurrence(
                node.pattern, source_var, new_var, first=first
            )
            return self._rewrap_unary_pattern(node, renamed_inner)
        return None

    def _rename_last_occurrence_in_prefix(
        self, patterns: List, source_var: str, new_var: str
    ):
        """Rename the last source occurrence without distributing unrelated choices.

        Only an alternation whose branches disagree about containing ``source_var``
        affects where the last occurrence is.  The previous implementation first
        formed the Cartesian product of *all* top-level alternations, producing
        large temporary trees that simplify() then had to factor again.

        A nullable quantifier needs the same branch-sensitive treatment.  If its
        non-empty branch is chosen, the last occurrence is inside that quantifier;
        if its empty branch is chosen, the last occurrence is somewhere earlier
        in the prefix.  Keep the suffix outside the resulting alternation so this
        local split does not duplicate unrelated pattern structure.
        """
        for idx in range(len(patterns) - 1, -1, -1):
            child = patterns[idx]
            if not self._contains_var(child, source_var):
                continue

            prefix = patterns[:idx]
            suffix = patterns[idx + 1 :]
            if (
                isinstance(child, QuantifiedPattern)
                and isinstance(child.pattern, VariablePattern)
                and child.pattern.name == source_var
                and child.atLeast == 0
                and (child.atMost is None or child.atMost > 0)
            ):
                nonempty = QuantifiedPattern(
                    VariablePattern(source_var),
                    1,
                    child.atMost,
                    child.isGreedy,
                )
                renamed_nonempty = self._rename_last_occurrence(
                    nonempty, source_var, new_var
                )
                renamed_empty_fallback = self._rename_last_occurrence_in_prefix(
                    prefix, source_var, new_var
                )
                if renamed_nonempty is None:
                    return None

                nonempty_branch = ConcatenationPattern(
                    prefix + [renamed_nonempty]
                ).simplify()
                # No earlier source occurrence means the empty branch has no
                # value to retarget.  Callers reach this path only after
                # proving that such a NULL/absent source cannot satisfy the
                # conjunct, so retain only the non-empty branch.
                if renamed_empty_fallback is None:
                    if not suffix:
                        return nonempty_branch
                    return ConcatenationPattern([nonempty_branch] + suffix).simplify()
                branches = (
                    [nonempty_branch, renamed_empty_fallback]
                    if child.isGreedy
                    else [renamed_empty_fallback, nonempty_branch]
                )
                split_prefix = AlternationPattern(branches).simplify()
                if not suffix:
                    return split_prefix
                return ConcatenationPattern([split_prefix] + suffix).simplify()

            if isinstance(child, AlternationPattern):
                branches_with_source = [
                    self._contains_var(branch, source_var) for branch in child.patterns
                ]
                if not all(branches_with_source):
                    renamed_prefix = self._rename_last_occurrence_in_prefix(
                        prefix, source_var, new_var
                    )
                    if renamed_prefix is None:
                        return None
                    options = []
                    for branch, has_source in zip(child.patterns, branches_with_source):
                        if has_source:
                            renamed_branch = self._rename_last_occurrence(
                                branch, source_var, new_var
                            )
                            if renamed_branch is None:
                                return None
                            branch_prefix = ConcatenationPattern(
                                prefix + [renamed_branch]
                            ).simplify()
                        else:
                            branch_prefix = ConcatenationPattern(
                                [renamed_prefix, branch]
                            ).simplify()
                        options.append(
                            ConcatenationPattern([branch_prefix] + suffix).simplify()
                        )
                    return AlternationPattern(options)

                renamed_branches = []
                for branch in child.patterns:
                    renamed_branch = self._rename_last_occurrence(
                        branch, source_var, new_var
                    )
                    if renamed_branch is None:
                        return None
                    renamed_branches.append(renamed_branch)
                renamed_child = AlternationPattern(renamed_branches)
            else:
                renamed_child = self._rename_last_occurrence(child, source_var, new_var)
                if renamed_child is None:
                    return None
            return ConcatenationPattern(prefix + [renamed_child] + suffix).simplify()
        return None

    def _rename_last_occurrence(self, node, source_var: str, new_var: str):
        return self._rename_edge_occurrence(node, source_var, new_var, first=False)

    def _can_isolate_occurrence(
        self,
        ast: RPRAbstractSyntaxTree,
        x_var: str,
        y_var: str,
        require_no_other_define_refs: bool = False,
    ) -> bool:
        if isinstance(ast.pattern, PermutationPattern):
            return False
        if (
            self._skip_references_var(ast, y_var)
            and not self._skip_targets_last_var(ast, y_var)
            and not self._skip_targets_first_var(ast, y_var)
        ):
            return False
        y_condition = ast.define.definitions.get(y_var)
        if isinstance(y_condition, DependentCondition):
            return False
        for def_var, condition in self._iter_defined_conditions(ast):
            if def_var in {x_var, y_var}:
                continue
            expression = condition.expression
            if not self._expr_references_any_var(expression, {y_var}):
                continue
            if require_no_other_define_refs:
                # FIRST and non-zero logical offsets isolate an occurrence
                # other than the last one.  Retargeting any ordinary Y
                # reference to that row would change RUNNING LAST semantics.
                return False
            if self._has_reference_requiring_more_than_last(expression, y_var):
                # The existing retargeting replaces Y with the fresh variable
                # representing only the last Y row.  That is valid for an
                # ordinary running Y reference and LAST(Y, 0), but not for an
                # aggregate, FIRST, or a non-zero LAST offset, all of which
                # observe more than the isolated last occurrence.
                return False
        return True

    def _other_define_refs_allow_occurrence_rename(
        self,
        ast: RPRAbstractSyntaxTree,
        renamed_pattern,
        x_var: str,
        old_var: str,
        new_var: str,
    ) -> bool:
        """Prove that global DEFINE retargeting is path-independent.

        If a DEFINE variable can occur both before and after the isolated
        occurrence on different paths, its single global definition cannot be
        changed from old_var to new_var: one branch needs the old running
        population and the other needs the isolated last row.  Reject that
        isolation instead of making an existential, branch-insensitive
        retargeting decision.
        """
        for def_var, condition in self._iter_defined_conditions(ast):
            if def_var in {x_var, old_var}:
                continue
            if not self._expr_references_any_var(condition.expression, {old_var}):
                continue

            new_var_can_precede = self._can_var_occur_after_target(
                renamed_pattern, def_var, new_var
            )
            if not new_var_can_precede:
                # This DEFINE is always evaluated before the isolated row, so
                # retaining its old_var reference preserves running semantics.
                continue
            if not self._must_have_var_before_target(renamed_pattern, new_var, def_var):
                return False
        return True

    def _has_reference_requiring_more_than_last(
        self, expr: Expr, var_name: str
    ) -> bool:
        """Return whether expr cannot be retargeted to only the last var row."""
        for node in self._walk_expr(expr):
            if not isinstance(node, FunctionCall):
                continue
            if not self._expr_references_any_var(node, {var_name}):
                continue
            if node.name in self._AGGREGATE_FUNCTIONS or node.name == "FIRST":
                return True
            if node.name != "LAST":
                continue
            target = self._extract_logical_nav_target(node)
            if target is None or target[0] != var_name or target[1] != 0:
                return True
            # LAST(var, 0) still denotes the same row after isolating the last
            # occurrence and retargeting its qualifier to the fresh variable.
        return False

    def _expr_references_any_var(self, expr: Expr, var_names: Set[str]) -> bool:
        return _expr_references_any_var(expr, var_names)

    def _measures_reference_var(
        self, ast: RPRAbstractSyntaxTree, var_name: str
    ) -> bool:
        if ast.measures is None:
            return False
        for expr in ast.measures.measures.values():
            if expr is not None and self._expr_references_any_var(expr, {var_name}):
                return True
        return False

    def _skip_references_var(self, ast: RPRAbstractSyntaxTree, var_name: str) -> bool:
        if ast.after_match_skip_to is None:
            return False
        tokens = re.findall(r"[A-Z_][A-Z0-9_]*", ast.after_match_skip_to)
        return var_name in tokens

    def _skip_targets_last_var(self, ast: RPRAbstractSyntaxTree, var_name: str) -> bool:
        if ast.after_match_skip_to is None:
            return False
        return ast.after_match_skip_to.strip() == f"LAST {var_name}"

    def _skip_targets_first_var(
        self, ast: RPRAbstractSyntaxTree, var_name: str
    ) -> bool:
        if ast.after_match_skip_to is None:
            return False
        return ast.after_match_skip_to.strip() == f"FIRST {var_name}"

    def _skip_allows_renamed_pattern(
        self,
        ast: RPRAbstractSyntaxTree,
        old_var: str,
        new_var: str,
        pattern,
        *,
        renamed_is_last: bool = False,
    ) -> bool:
        if not self._skip_references_var(ast, old_var):
            return True
        if self._skip_targets_last_var(ast, old_var):
            # If the isolated row was the last old_var row, the caller will
            # retarget SKIP to the fresh singleton variable.  Otherwise the
            # old skip target remains valid only when every path has a later
            # surviving old_var row.  Without that proof, LAST old_var may
            # move to an earlier row or become unmatched altogether.
            return renamed_is_last or self._must_have_var_after_target(
                pattern, old_var, new_var
            )
        if self._skip_targets_first_var(ast, old_var):
            return self._must_have_var_before_target(pattern, old_var, new_var)
        return False

    def _must_have_var_after_target(
        self, node, after_var: str, target_var: str, seen_after: bool = False
    ) -> bool:
        """Prove that every target occurrence has a later ``after_var`` row."""
        if not self._contains_var(node, target_var):
            return True

        if isinstance(node, VariablePattern):
            return seen_after if node.name == target_var else True

        if isinstance(node, QuantifiedPattern):
            return self._must_have_var_after_target(
                node.pattern, after_var, target_var, seen_after
            )

        if isinstance(node, AlternationPattern):
            return all(
                self._must_have_var_after_target(
                    child, after_var, target_var, seen_after
                )
                for child in node.patterns
                if self._contains_var(child, target_var)
            )

        if isinstance(node, ConcatenationPattern):
            current_seen = seen_after
            for child in reversed(node.patterns):
                if self._contains_var(child, target_var):
                    if not self._must_have_var_after_target(
                        child, after_var, target_var, current_seen
                    ):
                        return False
                if after_var in child.get_must_appear_var():
                    current_seen = True
            return True

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._must_have_var_after_target(
                node.pattern, after_var, target_var, seen_after
            )

        return False

    def _must_have_var_before_target(
        self, node, before_var: str, target_var: str, seen_before: bool = False
    ) -> bool:
        if not self._contains_var(node, target_var):
            return True

        if isinstance(node, VariablePattern):
            return seen_before if node.name == target_var else True

        if isinstance(node, QuantifiedPattern):
            return self._must_have_var_before_target(
                node.pattern, before_var, target_var, seen_before
            )

        if isinstance(node, AlternationPattern):
            return all(
                self._must_have_var_before_target(
                    child, before_var, target_var, seen_before
                )
                for child in node.patterns
                if self._contains_var(child, target_var)
            )

        if isinstance(node, ConcatenationPattern):
            current_seen = seen_before
            for child in node.patterns:
                if self._contains_var(child, target_var):
                    if not self._must_have_var_before_target(
                        child, before_var, target_var, current_seen
                    ):
                        return False
                if before_var in child.get_must_appear_var():
                    current_seen = True
            return True

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._must_have_var_before_target(
                node.pattern, before_var, target_var, seen_before
            )

        return False

    def _can_var_occur_after_target(
        self,
        node,
        source_var: str,
        target_var: str,
        seen_target: bool = False,
    ) -> bool:
        """Return whether any pattern path puts source after target.

        Callers already require a single possible target occurrence.  Under
        that restriction one pass through a quantified child is sufficient to
        expose every relevant source/target ordering; a zero-iteration path
        cannot introduce a reversed pair.
        """
        states = {(seen_target, False)}

        def advance(current, part):
            next_states = set()
            for target_seen, bad in current:
                if bad:
                    next_states.add((target_seen, True))
                    continue
                if isinstance(part, VariablePattern):
                    reversed_order = target_seen and part.name == source_var
                    next_states.add(
                        (target_seen or part.name == target_var, reversed_order)
                    )
                    continue
                if isinstance(part, (EmptyPattern,)):
                    next_states.add((target_seen, False))
                    continue
                if isinstance(part, (ExcludedPattern, AnchorPattern)):
                    next_states.update(advance({(target_seen, False)}, part.pattern))
                    continue
                if isinstance(part, ConcatenationPattern):
                    branch_states = {(target_seen, False)}
                    for child in part.patterns:
                        branch_states = advance(branch_states, child)
                    next_states.update(branch_states)
                    continue
                if isinstance(part, AlternationPattern):
                    for child in part.patterns:
                        next_states.update(advance({(target_seen, False)}, child))
                    continue
                if isinstance(part, QuantifiedPattern):
                    if part.atLeast == 0:
                        next_states.add((target_seen, False))
                    next_states.update(advance({(target_seen, False)}, part.pattern))
                    continue
                # PERMUTE and unknown pattern nodes do not have one provable
                # order.  If both variables can occur, conservatively reject.
                contains_source = self._contains_var(part, source_var)
                contains_target = self._contains_var(part, target_var)
                next_states.add(
                    (
                        target_seen or contains_target,
                        contains_source and (target_seen or contains_target),
                    )
                )
            return next_states

        states = advance(states, node)
        return any(bad for _, bad in states)

    def _generate_new_var_name(
        self, ast: RPRAbstractSyntaxTree, pattern_counts: Dict[str, int]
    ) -> str:
        used_names = set(pattern_counts.keys())
        used_names.update(var for var in ast.define.definitions.keys())
        if ast.subsets is not None:
            used_names.update(var for var in ast.subsets.keys())

        idx = 1
        while True:
            candidate = f"V{idx}"
            if candidate not in used_names:
                return candidate
            idx += 1

    def _rewrite_classifier_labels(
        self, ast: RPRAbstractSyntaxTree, new_var: str, original_var: str
    ) -> None:
        if ast.measures is None:
            return
        ast.measures.measures = {
            alias: self._rewrite_classifier_expr(expr, new_var, original_var)
            for alias, expr in ast.measures.measures.items()
        }

    def _has_classifier_with_arguments(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Detect permissively parsed, non-standard CLASSIFIER(arg) calls."""
        if ast.measures is None:
            return False
        return any(
            isinstance(node, FunctionCall)
            and node.name == "CLASSIFIER"
            and bool(node.args)
            for expr in ast.measures.measures.values()
            for node in self._walk_expr(expr)
        )

    def _rewrite_classifier_labels_many(
        self, ast: RPRAbstractSyntaxTree, replacements: List[Tuple[str, str]]
    ) -> None:
        if ast.measures is None:
            return
        ast.measures.measures = {
            alias: self._rewrite_classifier_expr_many(expr, replacements)
            for alias, expr in ast.measures.measures.items()
        }

    def _rewrite_classifier_expr_many(
        self, expr: Expr, replacements: List[Tuple[str, str]]
    ) -> Expr:
        if expr is None:
            return None
        if isinstance(expr, FunctionCall) and expr.name == "CLASSIFIER":
            classifier = FunctionCall(
                expr.name, [copy.deepcopy(arg) for arg in expr.args]
            )
            cases = [
                (
                    BinaryExpr(copy.deepcopy(classifier), "=", Literal(new_var)),
                    Literal(original_var),
                )
                for new_var, original_var in replacements
            ]
            return CaseExpr(cases, classifier)
        mapped = self._map_expr_children(
            expr,
            lambda child: self._rewrite_classifier_expr_many(child, replacements),
        )
        return mapped if mapped is not None else copy.deepcopy(expr)

    def _rewrite_classifier_expr(
        self, expr: Expr, new_var: str, original_var: str
    ) -> Expr:
        return self._rewrite_classifier_expr_many(expr, [(new_var, original_var)])

    def _count_pattern_variables(self, pattern) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for leaf in self._iter_pattern_leaf_variables(pattern):
            name = leaf.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _min_future_rows_to_target(
        self, node, source_var: str, target_var: str
    ) -> Optional[int]:
        if source_var == target_var:
            return 0

        if isinstance(node, VariablePattern):
            return None

        if isinstance(node, QuantifiedPattern):
            return self._min_future_rows_to_target(node.pattern, source_var, target_var)

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._min_future_rows_to_target(node.pattern, source_var, target_var)

        if isinstance(node, AlternationPattern):
            candidates = []
            for child in node.patterns:
                if self._contains_var(child, source_var) and self._contains_var(
                    child, target_var
                ):
                    rows = self._min_future_rows_to_target(
                        child, source_var, target_var
                    )
                    if rows is not None:
                        candidates.append(rows)
            return min(candidates) if candidates else None

        if isinstance(node, ConcatenationPattern):
            for i, left in enumerate(node.patterns):
                if not self._contains_var(left, source_var):
                    continue
                for j in range(i, len(node.patterns)):
                    right = node.patterns[j]
                    if not self._contains_var(right, target_var):
                        continue
                    if i == j:
                        rows = self._min_future_rows_to_target(
                            left, source_var, target_var
                        )
                        if rows is not None:
                            return rows
                        continue

                    inside_left = self._min_rows_after_var_to_end(left, source_var)
                    if inside_left is None:
                        continue
                    middle = sum(
                        part.get_min_row_num() for part in node.patterns[i + 1 : j]
                    )
                    to_target = self._min_rows_to_last_mandatory_target_inclusive(
                        right, target_var
                    )
                    if to_target is None:
                        continue
                    return inside_left + middle + to_target
            return None

        if isinstance(node, PermutationPattern):
            return None
        return None

    def _min_rows_to_last_mandatory_target_inclusive(
        self, node, target_var: str
    ) -> Optional[int]:
        if isinstance(node, VariablePattern):
            return 1 if node.name == target_var else None

        if isinstance(node, QuantifiedPattern):
            inner_rows = self._min_rows_to_last_mandatory_target_inclusive(
                node.pattern, target_var
            )
            if inner_rows is None or node.atLeast <= 0:
                return None
            return ((node.atLeast - 1) * node.pattern.get_min_row_num()) + inner_rows

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._min_rows_to_last_mandatory_target_inclusive(
                node.pattern, target_var
            )

        if isinstance(node, AlternationPattern):
            candidates = []
            for child in node.patterns:
                if target_var not in child.get_must_appear_var():
                    continue
                rows = self._min_rows_to_last_mandatory_target_inclusive(
                    child, target_var
                )
                if rows is not None:
                    candidates.append(rows)
            return min(candidates) if candidates else None

        if isinstance(node, ConcatenationPattern):
            prefix = 0
            last_candidate = None
            for child in node.patterns:
                if target_var in child.get_must_appear_var():
                    inside = self._min_rows_to_last_mandatory_target_inclusive(
                        child, target_var
                    )
                    if inside is not None:
                        last_candidate = prefix + inside
                prefix += child.get_min_row_num()
            return last_candidate

        if isinstance(node, PermutationPattern):
            return None
        return None

    def _min_rows_after_var_to_end(self, node, var_name: str) -> Optional[int]:
        if isinstance(node, VariablePattern):
            return 0 if node.name == var_name else None

        if isinstance(node, QuantifiedPattern):
            return self._min_rows_after_var_to_end(node.pattern, var_name)

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._min_rows_after_var_to_end(node.pattern, var_name)

        if isinstance(node, AlternationPattern):
            candidates = []
            for child in node.patterns:
                rows = self._min_rows_after_var_to_end(child, var_name)
                if rows is not None:
                    candidates.append(rows)
            return min(candidates) if candidates else None

        if isinstance(node, ConcatenationPattern):
            for idx, child in enumerate(node.patterns):
                if not self._contains_var(child, var_name):
                    continue
                inside = self._min_rows_after_var_to_end(child, var_name)
                if inside is None:
                    continue
                suffix = sum(
                    part.get_min_row_num() for part in node.patterns[idx + 1 :]
                )
                return inside + suffix
            return None

        if isinstance(node, PermutationPattern):
            return None
        return None


forward_adjusted_condition_test_cases = [
    # (A B*? C), where VAL > 10 AND A.VAL < 5
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLA
    ALL ROWS PER MATCH
    PATTERN(A B*? C)
    DEFINE A AS TYPE = 'A', C AS VAL > 10 AND B.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CASE WHEN CLASSIFIER() = 'V1' THEN 'B' ELSE CLASSIFIER() END AS CLA
    ALL ROWS PER MATCH
    PATTERN(A B*? V1 C)
    DEFINE A AS TYPE = 'A', V1 AS VAL < 5, C AS VAL > 10
)""",
    ),
    # ((A | B) C), where C ==> VAL > 10 AND A.VAL < 5
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLA
    ALL ROWS PER MATCH
    PATTERN((A | B) C)
    DEFINE C AS VAL > 10 AND A.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLA
    ALL ROWS PER MATCH
    PATTERN(A C)
    DEFINE A AS A.VAL < 5, C AS VAL > 10
)""",
    ),
    # (A{0,5} C), where C ==> VAL > 10 AND A.VAL < 5
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A{0,5} V1)
    DEFINE V1 AS VAL > 10 AND A.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A{0,4} V2 V1)
    DEFINE V2 AS VAL < 5, V1 AS VAL > 10
)""",
    ),
    # (A{1,5} C), where C ==> VAL > 10 AND A.VAL < 5
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A{1,5} C)
    DEFINE C AS VAL > 10 AND A.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A{0,4} V1 C)
    DEFINE V1 AS VAL < 5, C AS VAL > 10
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A A B C)
    DEFINE B AS VAL = 5, C AS VAL > 10 AND A.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A V1 B C)
    DEFINE B AS VAL = 5, V1 AS VAL < 5, C AS VAL > 10
)""",
    ),
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A A B C)
    DEFINE B AS VAL = 5 AND A.VAL > 0, C AS VAL > 10 AND A.VAL < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A V1 B C)
    DEFINE B AS VAL = 5, V1 AS VAL > 0 AND VAL < 5, C AS VAL > 10
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        ForwardAdjustedConditionRule,
        forward_adjusted_condition_test_cases,
        "forward adjusted condition",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
