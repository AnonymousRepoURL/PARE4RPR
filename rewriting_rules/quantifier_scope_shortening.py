if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

from .utils import *
from expression_simplifier import _condition_is_not_true


@dataclass(frozen=True)
class _UnboundedQuantifier:
    """The information this rule needs from a simple ``A{x,}`` pattern."""

    variable: str
    minimum: int
    greedy: bool


@dataclass(frozen=True)
class _IndependentRange:
    """A DEFINE expression together with its normalized range DNF."""

    expression: Optional[Expr]
    dnf: list


@dataclass(frozen=True)
class _GuardRightAction:
    left_variable: str
    right_variable: str
    expression: Expr


@dataclass(frozen=True)
class _FixLeftReluctantAction:
    patterns: list
    index: int


@dataclass(frozen=True)
class _FixRightAction:
    patterns: list
    index: int


_RewriteAction = Union[
    _GuardRightAction,
    _FixLeftReluctantAction,
    _FixRightAction,
]


@dataclass(frozen=True)
class _PatternContext:
    """Pattern elements surrounding the concatenation currently being visited."""

    left: Tuple = ()
    right: Tuple = ()
    inside_alternation: bool = False

    def nested(self, prefix: list, suffix: list) -> "_PatternContext":
        return _PatternContext(
            left=self.left + tuple(prefix),
            right=tuple(suffix) + self.right,
            inside_alternation=self.inside_alternation,
        )

    def alternative(self) -> "_PatternContext":
        return _PatternContext(self.left, self.right, True)


@dataclass
class _CollectionState:
    actions: List[_RewriteAction]
    guarded_variables: Set[str]
    fixed_left_variables: Set[str]
    fixed_right_variables: Set[str]

    @classmethod
    def empty(cls) -> "_CollectionState":
        return cls([], set(), set(), set())

    def fork_alternative(self) -> "_CollectionState":
        """Share emitted actions/guards, but isolate branch-local quantifier fixes."""
        return _CollectionState(
            self.actions,
            self.guarded_variables,
            set(self.fixed_left_variables),
            set(self.fixed_right_variables),
        )


class QuantifierScopeShorteningRule(RewriteRule):
    """Shorten safe boundaries between adjacent ``B{x,}`` and ``C{y,}``.

    Candidate transformations are selected from normalized independent ranges:

    * equal ranges and greedy B: fix C at its minimum;
    * B range strictly inside C and greedy B: add a boundary guard to C;
    * C range strictly inside B and greedy B: fix C at its minimum;
    * B range contained in C, reluctant B and greedy C: fix B at its positive
      minimum.  A dependent B is allowed here only when B occurs once globally.

    Every candidate is rejected if another DEFINE can observe rows moving across
    the boundary, physical PREV/NEXT can reach the pair, a needed range proof is
    unavailable, or a global DEFINE guard could affect an unsafe occurrence.
    Guard actions are also disabled inside alternations because DEFINE is global.

    CLASSIFIER() in DEFINE is intentionally out of scope: Oracle does not allow
    CLASSIFIER() in that clause.
    """

    @property
    def name(self) -> str:
        return "adjacent_infinite_quantifier_overlap"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return len(self._collect_actions(ast)) > 0

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        actions = self._collect_actions(ast)
        if not actions:
            return False

        changed = False
        for action in actions:
            if isinstance(action, _GuardRightAction):
                self._set_define_condition(
                    ast,
                    action.right_variable,
                    action.left_variable,
                    DependentCondition(
                        action.right_variable,
                        action.expression,
                        ["overlap boundary guard"],
                    ),
                )
                changed = True
            elif isinstance(action, _FixLeftReluctantAction):
                changed = (
                    self._fix_quantifier_at_minimum(
                        action.patterns,
                        action.index,
                        allow_optional_zero=False,
                    )
                    or changed
                )
            elif isinstance(action, _FixRightAction):
                changed = (
                    self._fix_quantifier_at_minimum(
                        action.patterns,
                        action.index,
                        allow_optional_zero=True,
                    )
                    or changed
                )

        return changed

    def _fix_quantifier_at_minimum(
        self,
        patterns: list,
        index: int,
        *,
        allow_optional_zero: bool,
    ) -> bool:
        quantified = patterns[index]
        if not isinstance(quantified, QuantifiedPattern):
            # An overlapping action may already have fixed this node.
            return False
        if quantified.atLeast == 0:
            if not allow_optional_zero:
                # B*? prefers zero rows, whereas greedy B? prefers one.
                return False
            patterns[index] = QuantifiedPattern(quantified.pattern, 0, 1, True)
        elif quantified.atLeast == 1:
            patterns[index] = quantified.pattern
        else:
            patterns[index] = QuantifiedPattern(
                quantified.pattern,
                quantified.atLeast,
                quantified.atLeast,
                True,
            )
        return True

    def _collect_actions(self, ast: RPRAbstractSyntaxTree) -> List[_RewriteAction]:
        state = _CollectionState.empty()
        self._collect_actions_from_pattern(
            ast,
            ast.pattern,
            state,
            _PatternContext(),
        )
        return state.actions

    def _collect_actions_from_pattern(
        self,
        ast: RPRAbstractSyntaxTree,
        pattern,
        state: _CollectionState,
        context: _PatternContext,
    ) -> None:
        """Walk concatenations, including concatenations in alternatives."""
        if isinstance(pattern, ConcatenationPattern):
            self._collect_actions_from_concat(
                ast,
                pattern.patterns,
                state,
                context,
            )
            return

        if isinstance(pattern, AlternationPattern):
            for branch in pattern.patterns:
                self._collect_actions_from_pattern(
                    ast,
                    branch,
                    state.fork_alternative(),
                    context.alternative(),
                )
            return

        if isinstance(pattern, ExcludedPattern):
            self._collect_actions_from_pattern(
                ast,
                pattern.pattern,
                state,
                context,
            )

    def _collect_actions_from_concat(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        state: _CollectionState,
        context: _PatternContext,
    ) -> None:
        """Collect safe rewrites from one concatenation and nested scopes."""
        # ── recurse into nested scopes first ──────────────────────────────
        for i, pat in enumerate(patterns):
            if isinstance(
                pat, (ConcatenationPattern, AlternationPattern, ExcludedPattern)
            ):
                self._collect_actions_from_pattern(
                    ast,
                    pat,
                    state,
                    context.nested(patterns[:i], patterns[i + 1 :]),
                )

        # ── process adjacent pairs in the current patterns list ───────────────
        for idx in range(len(patterns) - 1):
            self._collect_action_for_pair(ast, patterns, idx, state, context)

    def _collect_action_for_pair(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        state: _CollectionState,
        context: _PatternContext,
    ) -> None:
        left = self._get_simple_unbounded_quantifier(patterns[idx])
        right = self._get_simple_unbounded_quantifier(patterns[idx + 1])
        if left is None or right is None or left.variable == right.variable:
            return
        if self._has_external_nav_to_pair(
            ast,
            patterns,
            idx,
            idx + 1,
            list(context.right),
            list(context.left),
        ):
            return

        if left.greedy:
            self._collect_greedy_left_action(
                ast, patterns, idx, left, right, state, context
            )
            return

        if not right.greedy or left.variable in state.fixed_left_variables:
            return
        action = self._build_fix_left_action(ast, patterns, idx, left, right)
        if action is not None:
            state.actions.append(action)
            state.fixed_left_variables.add(left.variable)

    def _collect_greedy_left_action(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        left: _UnboundedQuantifier,
        right: _UnboundedQuantifier,
        state: _CollectionState,
        context: _PatternContext,
    ) -> None:
        if (
            right.variable in state.guarded_variables
            or right.variable in state.fixed_right_variables
        ):
            return

        # A guard mutates a global DEFINE clause, so never install one from an
        # alternation branch where it could constrain a sibling occurrence.
        if not context.inside_alternation:
            action = self._build_guard_action(
                ast,
                patterns,
                idx,
                left,
                right,
                list(context.left),
                list(context.right),
            )
            if action is not None:
                state.actions.append(action)
                state.guarded_variables.add(right.variable)
                return

        action = self._build_equal_range_fix_right_action(
            ast, patterns, idx, left, right
        )
        if action is None:
            action = self._build_covered_fix_right_action(
                ast, patterns, idx, left, right
            )
        if action is not None:
            state.actions.append(action)
            state.fixed_right_variables.add(right.variable)

    def _get_simple_unbounded_quantifier(
        self, pattern
    ) -> Optional[_UnboundedQuantifier]:
        if not isinstance(pattern, QuantifiedPattern):
            return None
        if pattern.atMost is not None:
            return None
        if not isinstance(pattern.pattern, VariablePattern):
            return None
        return _UnboundedQuantifier(
            pattern.pattern.name,
            pattern.atLeast,
            pattern.isGreedy,
        )

    # ------------------------------------------------------------------
    # Physical navigation helpers
    # ------------------------------------------------------------------

    def _get_pattern_var_names(self, pattern) -> Set[str]:
        """Return the set of all variable names appearing in *pattern*."""
        if isinstance(pattern, VariablePattern):
            return {pattern.name}
        if isinstance(pattern, QuantifiedPattern):
            return self._get_pattern_var_names(pattern.pattern)
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            result: Set[str] = set()
            for child in pattern.patterns:
                result |= self._get_pattern_var_names(child)
            return result
        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._get_pattern_var_names(pattern.pattern)
        return set()

    def _expr_contains_nav_func(self, expr: Expr, func_name: str) -> bool:
        """Return True if *expr* contains a call to PREV or NEXT (physical navigation)."""
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._expr_contains_nav_func(expr.expr, func_name)
        if isinstance(expr, FunctionCall):
            if expr.name and expr.name.upper() == func_name.upper():
                return True
            return any(self._expr_contains_nav_func(a, func_name) for a in expr.args)
        if isinstance(expr, BinaryExpr):
            return self._expr_contains_nav_func(
                expr.left, func_name
            ) or self._expr_contains_nav_func(expr.right, func_name)
        if isinstance(expr, BetweenExpr):
            return (
                self._expr_contains_nav_func(expr.expr, func_name)
                or self._expr_contains_nav_func(expr.lower, func_name)
                or self._expr_contains_nav_func(expr.upper, func_name)
            )
        if isinstance(expr, InExpr):
            return self._expr_contains_nav_func(expr.expr, func_name) or any(
                self._expr_contains_nav_func(v, func_name) for v in expr.values
            )
        if isinstance(expr, IsNullExpr):
            return self._expr_contains_nav_func(expr.expr, func_name)
        if isinstance(expr, NotExpr):
            return self._expr_contains_nav_func(expr.expr, func_name)
        if isinstance(expr, CaseExpr):
            return any(
                self._expr_contains_nav_func(condition, func_name)
                or self._expr_contains_nav_func(result, func_name)
                for condition, result in expr.cases
            ) or (
                expr.else_expr is not None
                and self._expr_contains_nav_func(expr.else_expr, func_name)
            )
        return False

    def _extract_navigation_offset(self, func: FunctionCall) -> Optional[int]:
        """Extract PREV/NEXT offset; default is 1, unknown/non-literal is unsafe."""
        return _extract_navigation_offset(func)

    def _collect_nav_calls(
        self, expr: Expr, target_func: str, def_var: str
    ) -> List[Tuple[str, Optional[int], Expr]]:
        """Collect all target PREV/NEXT calls as (base_var, offset, first_arg_expr)."""
        results: List[Tuple[str, Optional[int], Expr]] = []

        def visit(node: Expr):
            if node is None:
                return
            if isinstance(node, SemanticsExpr):
                visit(node.expr)
                return
            if isinstance(node, FunctionCall):
                if node.name and node.name.upper() == target_func and node.args:
                    offset = self._extract_navigation_offset(node)
                    arg0 = node.args[0]
                    if isinstance(arg0, SemanticsExpr):
                        arg0 = arg0.expr
                    if isinstance(arg0, ColumnRef) and arg0.qualifier is not None:
                        base_var = arg0.qualifier
                    else:
                        base_var = def_var
                    results.append((base_var, offset, arg0))
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
            if isinstance(node, InExpr):
                visit(node.expr)
                for value in node.values:
                    visit(value)
                return
            if isinstance(node, IsNullExpr):
                visit(node.expr)
                return
            if isinstance(node, NotExpr):
                visit(node.expr)
                return
            if isinstance(node, CaseExpr):
                for condition, result in node.cases:
                    visit(condition)
                    visit(result)
                visit(node.else_expr)

        visit(expr)
        return results

    def _expr_depends_on_pair_assignment(
        self,
        expr: Expr,
        pair_vars: Set[str],
        default_var: str,
    ) -> bool:
        """Return True when expr's value may change if rows are repartitioned inside the pair."""
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._expr_depends_on_pair_assignment(
                expr.expr, pair_vars, default_var
            )
        if isinstance(expr, ColumnRef):
            qualifier = expr.qualifier if expr.qualifier is not None else default_var
            return qualifier in pair_vars
        if isinstance(expr, FunctionCall):
            return any(
                self._expr_depends_on_pair_assignment(arg, pair_vars, default_var)
                for arg in expr.args
            )
        if isinstance(expr, BinaryExpr):
            return self._expr_depends_on_pair_assignment(
                expr.left, pair_vars, default_var
            ) or self._expr_depends_on_pair_assignment(
                expr.right, pair_vars, default_var
            )
        if isinstance(expr, BetweenExpr):
            return (
                self._expr_depends_on_pair_assignment(expr.expr, pair_vars, default_var)
                or self._expr_depends_on_pair_assignment(
                    expr.lower, pair_vars, default_var
                )
                or self._expr_depends_on_pair_assignment(
                    expr.upper, pair_vars, default_var
                )
            )
        if isinstance(expr, InExpr):
            return self._expr_depends_on_pair_assignment(
                expr.expr, pair_vars, default_var
            ) or any(
                self._expr_depends_on_pair_assignment(v, pair_vars, default_var)
                for v in expr.values
            )
        if isinstance(expr, IsNullExpr):
            return self._expr_depends_on_pair_assignment(
                expr.expr, pair_vars, default_var
            )
        if isinstance(expr, NotExpr):
            return self._expr_depends_on_pair_assignment(
                expr.expr, pair_vars, default_var
            )
        if isinstance(expr, CaseExpr):
            return any(
                self._expr_depends_on_pair_assignment(condition, pair_vars, default_var)
                or self._expr_depends_on_pair_assignment(result, pair_vars, default_var)
                for condition, result in expr.cases
            ) or (
                expr.else_expr is not None
                and self._expr_depends_on_pair_assignment(
                    expr.else_expr, pair_vars, default_var
                )
            )
        return False

    def _nav_can_reach_pair(
        self,
        ast: RPRAbstractSyntaxTree,
        base_var: str,
        offset: Optional[int],
        pair_boundary_var: str,
        func_name: str,
    ) -> bool:
        """Return True when PREV/NEXT may physically navigate into the pair."""
        if offset is None:
            return True
        if offset == 0:
            return False
        base_left_min = ast.pattern.get_left_min_row_num(base_var)
        boundary_left_min = ast.pattern.get_left_min_row_num(pair_boundary_var)
        if base_left_min is None or boundary_left_min is None:
            return True

        if func_name.upper() == "PREV":
            min_distance = base_left_min - boundary_left_min
        else:
            min_distance = boundary_left_min - base_left_min

        if min_distance <= 0:
            return True
        return offset >= min_distance

    def _has_external_nav_to_pair(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx_left: int,
        idx_right: int,
        context_right: list,
        context_left: list,
    ) -> bool:
        """Return True if any variable outside the (idx_left, idx_right) pair may
        physically navigate into the pair's matched rows.

        Variables that appear AFTER the pair (in *patterns* or *context_right*)
        can navigate backward (PREV) into the pair.
        Variables that appear BEFORE the pair (in *context_left* or the leading
        part of *patterns*) can navigate forward (NEXT) into the pair.
        """
        left_pair_vars = self._get_pattern_var_names(patterns[idx_left])
        right_pair_vars = self._get_pattern_var_names(patterns[idx_right])
        pair_vars = left_pair_vars | right_pair_vars
        left_boundary_var = next(iter(left_pair_vars), None)
        right_boundary_var = next(iter(right_pair_vars), None)
        if left_boundary_var is None or right_boundary_var is None:
            return True

        # Patterns that appear to the RIGHT of the pair (PREV could reach pair rows)
        after_patterns = list(patterns[idx_right + 1 :]) + context_right
        for pat in after_patterns:
            for vname in self._get_pattern_var_names(pat):
                if vname in pair_vars:
                    continue
                if vname not in ast.define.definitions:
                    continue
                cond = ast.define.definitions[vname]
                if cond is None:
                    continue
                for base_var, offset, nav_arg in self._collect_nav_calls(
                    cond.expression, "PREV", vname
                ):
                    if not self._expr_depends_on_pair_assignment(
                        nav_arg, pair_vars, vname
                    ):
                        continue
                    if self._nav_can_reach_pair(
                        ast, base_var, offset, right_boundary_var, "PREV"
                    ):
                        return True

        # Patterns that appear to the LEFT of the pair (NEXT could reach pair rows)
        before_patterns = context_left + list(patterns[:idx_left])
        for pat in before_patterns:
            for vname in self._get_pattern_var_names(pat):
                if vname in pair_vars:
                    continue
                if vname not in ast.define.definitions:
                    continue
                cond = ast.define.definitions[vname]
                if cond is None:
                    continue
                for base_var, offset, nav_arg in self._collect_nav_calls(
                    cond.expression, "NEXT", vname
                ):
                    if not self._expr_depends_on_pair_assignment(
                        nav_arg, pair_vars, vname
                    ):
                        continue
                    if self._nav_can_reach_pair(
                        ast, base_var, offset, left_boundary_var, "NEXT"
                    ):
                        return True

        return False

    # ------------------------------------------------------------------
    # Pair-analysis helpers
    # ------------------------------------------------------------------

    def _has_dependent_condition(
        self,
        ast: RPRAbstractSyntaxTree,
        variable: str,
    ) -> bool:
        condition = ast.define.definitions.get(variable)
        return isinstance(condition, DependentCondition)

    def _get_independent_range(
        self,
        ast: RPRAbstractSyntaxTree,
        variable: str,
        *,
        require_constraint: bool,
    ) -> Optional[_IndependentRange]:
        """Normalize a variable's independent predicate, or reject it safely.

        An absent DEFINE means TRUE.  That is usable for the wider side of a
        subset proof, but not where the proof requires an actual constraint.
        """
        expression = (
            ast.define.get_independent_expr(variable)
            if variable in ast.define.definitions
            else None
        )
        dnf = self._extract_range_map_dnf(
            expression,
            variable,
            ast.expression_simplify_context(),
        )
        if dnf is None:
            return None
        if require_constraint and len(dnf) == 1 and len(dnf[0]) == 0:
            return None
        return _IndependentRange(expression, dnf)

    def _build_guard_action(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        left: _UnboundedQuantifier,
        right: _UnboundedQuantifier,
        context_left: list,
        context_right: list,
    ) -> Optional[_GuardRightAction]:
        left_var = left.variable
        right_var = right.variable

        # The generated guard is installed on right_var's global DEFINE clause,
        # so it would also constrain any occurrence after this adjacent pair.
        # Restrict the rewrite to patterns for which the guarded occurrence is
        # the final occurrence of right_var.
        suffix_patterns = list(patterns[idx + 2 :]) + list(context_right)
        if any(
            right_var in self._get_pattern_var_names(pattern)
            for pattern in suffix_patterns
        ):
            return None

        # The boundary guard is derived from the left variable's independent
        # range.  A dependent predicate on the left can nevertheless prevent it
        # from consuming overlapping rows, leaving multiple such rows that the
        # original right quantifier must accept.  Guarding those rows would then
        # remove a valid match.
        if self._has_dependent_condition(ast, left_var):
            return None

        if self._has_dependent_condition(ast, right_var):
            return None
        # The guard can change how rows are split between the adjacent
        # quantifiers.  A later DEFINE that observes either side (for example
        # LAST(B.ID)) can therefore make a previously valid backtracking split
        # disappear.
        if self._is_pair_assignment_referenced_elsewhere(ast, left_var, right_var):
            return None

        left_range = self._get_independent_range(
            ast,
            left_var,
            require_constraint=True,
        )
        if left_range is None or left_range.expression is None:
            return None

        right_range = self._get_independent_range(
            ast,
            right_var,
            require_constraint=False,
        )
        if right_range is None:
            return None

        if not self._is_strict_range_subset_dnf(left_range.dnf, right_range.dnf):
            return None

        # idx is local to patterns and may refer to a concatenation nested in an
        # ExcludedPattern.  Include the enclosing left context rather than
        # slicing the top-level pattern with a nested index.  The helper returns
        # None when an earlier occurrence count is not statically fixed.
        prior_patterns = list(context_left) + list(patterns[:idx])
        prior_count = self._fixed_prior_occurrence_count(prior_patterns, right_var)
        if prior_count is None:
            return None

        guard_expr = self._build_first_not_overlap_expr(
            left_range.expression,
            left_var,
            right_var,
            prior_count,
            ast.expression_simplify_context(),
            source_nullability_is_preserved=not input_may_null_extend_columns(
                ast.input_table,
            ),
        )
        if guard_expr is None:
            return None

        new_expr = self._compose_guarded_right_expression(
            right_var=right_var,
            right_minimum=right.minimum,
            prior_count=prior_count,
            original_expression=right_range.expression,
            boundary_guard=guard_expr,
        )

        return _GuardRightAction(left_var, right_var, new_expr)

    def _compose_guarded_right_expression(
        self,
        *,
        right_var: str,
        right_minimum: int,
        prior_count: int,
        original_expression: Optional[Expr],
        boundary_guard: Expr,
    ) -> Expr:
        """Activate a global DEFINE guard only at the rewritten occurrence.

        COUNT protection is required whenever the target has a positive lower
        bound or an earlier fixed occurrence exists.  In particular, the
        ``prior_count > 0, right_minimum == 0`` case must not guard those prior
        rows.
        """
        if right_minimum > 0 or prior_count > 0:
            return self._build_count_guard_expr(
                right_var,
                prior_count + right_minimum,
                original_expression,
                boundary_guard,
            )
        if original_expression is None:
            return boundary_guard
        return self._and_exprs([boundary_guard, original_expression])

    def _build_fix_left_action(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        left: _UnboundedQuantifier,
        right: _UnboundedQuantifier,
    ) -> Optional[_FixLeftReluctantAction]:
        """Fix left non-greedy quantifier to its minimum count.

        Applies in two situations:
          1. Normal case: left has no dependent condition, right has no dependent condition,
             left is not referenced elsewhere, left range ⊆ right range (subset or equal).
          2. Special case: left HAS a dependent condition, right has NO dependent condition,
             left is not referenced elsewhere, left independent range ⊆ right range.

        In both cases with x > 0: B{x,}? → B{x}.  The x == 0 case is
        deliberately skipped because greedy B? is not equivalent to B*?.
        """
        left_var = left.variable
        right_var = right.variable

        # Fixing a reluctant quantifier at its minimum is only representable by
        # the existing pattern variable when that minimum is positive.  In
        # particular, B*? cannot be approximated by greedy B?: the latter may
        # assign one row to B and thereby change observable match labels.
        if left.minimum == 0:
            return None

        # Right must never have a dependent condition in any case.
        if self._has_dependent_condition(ast, right_var):
            return None

        if self._has_dependent_condition(ast, left_var) and (
            self._pattern_variable_occurrence_count(ast.pattern, left_var) != 1
        ):
            # DEFINE is global to every occurrence of a pattern variable.  A
            # later occurrence can observe how many rows the reluctant one
            # consumed (for example through COUNT(A.*)), so fixing the earlier
            # quantifier to its minimum is safe only when this is A's sole
            # syntactic occurrence in the complete pattern.
            return None

        # Fixing the left quantifier can also change the rows assigned to the
        # right quantifier, so neither side may be observed by another DEFINE.
        if self._is_pair_assignment_referenced_elsewhere(ast, left_var, right_var):
            return None

        left_range = self._get_independent_range(
            ast,
            left_var,
            require_constraint=True,
        )
        if left_range is None:
            return None

        right_range = self._get_independent_range(
            ast,
            right_var,
            require_constraint=False,
        )
        if right_range is None:
            return None
        if not (
            self._is_strict_range_subset_dnf(left_range.dnf, right_range.dnf)
            or self._are_range_maps_equal_dnf(left_range.dnf, right_range.dnf)
        ):
            return None

        return _FixLeftReluctantAction(patterns, idx)

    def _pattern_variable_occurrence_count(self, pattern, target_var: str) -> int:
        """Count syntactic VariablePattern nodes, ignoring quantifier bounds."""
        if isinstance(pattern, VariablePattern):
            return 1 if pattern.name == target_var else 0
        if isinstance(pattern, QuantifiedPattern):
            return self._pattern_variable_occurrence_count(pattern.pattern, target_var)
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return sum(
                self._pattern_variable_occurrence_count(child, target_var)
                for child in pattern.patterns
            )
        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._pattern_variable_occurrence_count(pattern.pattern, target_var)
        return 0

    def _build_equal_range_fix_right_action(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        left: _UnboundedQuantifier,
        right: _UnboundedQuantifier,
    ) -> Optional[_FixRightAction]:
        """Feature 1a – equal independent range constraints, left greedy → fix right quantifier.

        Conditions:
          - Both variables have no dependent conditions.
          - Left and right independent range constraints are equal.
          - Neither variable is referenced by other DEFINE clauses.

        Action: shrink right's quantifier from VAR{x,} to VAR{x} (fixed),
                or to VAR? when right at_least == 0.
        """
        left_var = left.variable
        right_var = right.variable

        # Both must have no dependent conditions.
        if self._has_dependent_condition(ast, left_var):
            return None
        if self._has_dependent_condition(ast, right_var):
            return None

        # Fixing the right quantifier can force the greedy left quantifier to
        # retain rows that the original pattern could move to the right during
        # backtracking.  References to either variable can observe that split.
        if self._is_pair_assignment_referenced_elsewhere(ast, left_var, right_var):
            return None

        left_range = self._get_independent_range(
            ast,
            left_var,
            require_constraint=True,
        )
        if left_range is None:
            return None

        right_range = self._get_independent_range(
            ast,
            right_var,
            require_constraint=True,
        )
        if right_range is None:
            return None

        if not self._are_range_maps_equal_dnf(left_range.dnf, right_range.dnf):
            return None

        return _FixRightAction(patterns, idx + 1)

    def _build_covered_fix_right_action(
        self,
        ast: RPRAbstractSyntaxTree,
        patterns: list,
        idx: int,
        left: _UnboundedQuantifier,
        right: _UnboundedQuantifier,
    ) -> Optional[_FixRightAction]:
        """Case 3 – left greedy covers right (right range ⊂ left range) → fix right quantifier.

        Conditions:
          - Left must be greedy (guaranteed by caller).
          - Right's independent range is a strict subset of left's independent range.
          - Neither variable has a dependent condition or is referenced elsewhere.

        Action: shrink right's quantifier from VAR{x,}(?) to VAR{x} (or VAR? when x==0).
        """
        left_var = left.variable
        right_var = right.variable

        # A dependent predicate on the greedy left variable can prevent it from
        # consuming rows that satisfy the right variable's narrower independent
        # range.  In that case the original match may require more than the
        # right quantifier's minimum, so fixing the right side is not sound.
        if self._has_dependent_condition(ast, left_var):
            return None

        # Right must also have no dependent condition.
        if self._has_dependent_condition(ast, right_var):
            return None

        # References to either side can observe a different backtracking split.
        if self._is_pair_assignment_referenced_elsewhere(ast, left_var, right_var):
            return None

        left_range = self._get_independent_range(
            ast,
            left_var,
            require_constraint=True,
        )
        if left_range is None:
            return None

        right_range = self._get_independent_range(
            ast,
            right_var,
            require_constraint=False,
        )
        if right_range is None:
            return None

        # right must be a strict subset of left (left strictly covers right)
        if not self._is_strict_range_subset_dnf(right_range.dnf, left_range.dnf):
            return None

        return _FixRightAction(patterns, idx + 1)

    def _is_pair_assignment_referenced_elsewhere(
        self,
        ast: RPRAbstractSyntaxTree,
        left_var: str,
        right_var: str,
    ) -> bool:
        """Whether another DEFINE can observe rows moving across this boundary.

        A direct reference to either primary variable is observable.  A reference
        through a union variable is observable exactly when the SUBSET contains
        one side of the pair but not the other.  When it contains both sides, a
        boundary shift leaves the union's ordered row set unchanged.
        """
        observable_subset_names: Set[str] = set()
        for subset_name, members in (ast.subsets or {}).items():
            normalized_members = {member.strip() for member in members}
            if (left_var in normalized_members) != (right_var in normalized_members):
                observable_subset_names.add(subset_name)

        for var_name, condition in ast.define.definitions.items():
            if var_name in {left_var, right_var} or condition is None:
                continue
            if self._expr_qualifier_mentions(
                condition.expression, left_var
            ) or self._expr_qualifier_mentions(condition.expression, right_var):
                return True
            if any(
                self._expr_qualifier_mentions(condition.expression, subset_name)
                for subset_name in observable_subset_names
            ):
                return True

        return False

    def _expr_qualifier_mentions(self, expr: Expr, target_var: str) -> bool:
        if expr is None:
            return False
        if isinstance(expr, SemanticsExpr):
            return self._expr_qualifier_mentions(expr.expr, target_var)
        if isinstance(expr, ColumnRef):
            return expr.qualifier == target_var
        if isinstance(expr, FunctionCall):
            return any(
                self._expr_qualifier_mentions(arg, target_var) for arg in expr.args
            )
        if isinstance(expr, BinaryExpr):
            return self._expr_qualifier_mentions(
                expr.left, target_var
            ) or self._expr_qualifier_mentions(expr.right, target_var)
        if isinstance(expr, BetweenExpr):
            return (
                self._expr_qualifier_mentions(expr.expr, target_var)
                or self._expr_qualifier_mentions(expr.lower, target_var)
                or self._expr_qualifier_mentions(expr.upper, target_var)
            )
        if isinstance(expr, InExpr):
            if self._expr_qualifier_mentions(expr.expr, target_var):
                return True
            return any(
                self._expr_qualifier_mentions(value, target_var)
                for value in expr.values
            )
        if isinstance(expr, IsNullExpr):
            return self._expr_qualifier_mentions(expr.expr, target_var)
        if isinstance(expr, NotExpr):
            return self._expr_qualifier_mentions(expr.expr, target_var)
        if isinstance(expr, CaseExpr):
            for condition, result in expr.cases:
                if self._expr_qualifier_mentions(
                    condition, target_var
                ) or self._expr_qualifier_mentions(result, target_var):
                    return True
            return expr.else_expr is not None and self._expr_qualifier_mentions(
                expr.else_expr, target_var
            )
        return False

    def _extract_range_map(
        self, expr: Optional[Expr], var_name: str, context: SimplifyContext
    ) -> Optional[Dict[str, RangeConstraint]]:
        if expr is None:
            return {}

        conjuncts: List[Expr] = []
        if not self._flatten_and(expr, conjuncts):
            return None

        constraints: Dict[str, RangeConstraint] = {}
        for atom in conjuncts:
            if self._is_true_expr(atom):
                continue
            if not self._apply_generic_range_atom(
                constraints,
                atom,
                var_name,
                context,
            ):
                return None

        for constraint in constraints.values():
            if not constraint.is_satisfiable():
                return None

        return constraints

    def _flatten_and(self, expr: Expr, conjuncts: List[Expr]) -> bool:
        return _flatten_and_strict(expr, conjuncts)

    def _flatten_or(self, expr: Expr, disjuncts: List[Expr]) -> bool:
        """Flatten top-level OR into a list of conjunctive sub-expressions.
        Returns False if the structure cannot be represented (e.g. nested OR inside AND).
        """
        if isinstance(expr, SemanticsExpr):
            return self._flatten_or(expr.expr, disjuncts)
        if isinstance(expr, BinaryExpr) and expr.op == "OR":
            return self._flatten_or(expr.left, disjuncts) and self._flatten_or(
                expr.right, disjuncts
            )
        # Each non-OR sub-expression is treated as a conjunct (may itself be an AND chain)
        disjuncts.append(expr)
        return True

    def _extract_range_map_dnf(
        self, expr: Optional[Expr], var_name: str, context: SimplifyContext
    ) -> Optional[List[Dict[str, "RangeConstraint"]]]:
        """Extract range constraints in DNF form (list of conjunctive constraint maps).

        Returns a list where each element is a Dict[col -> RangeConstraint] representing
        one disjunct. Returns None if the expression cannot be represented.

        Examples:
          "OPEN < 10"                 → [{OPEN: (-∞,10)}]
          "OPEN < 10 OR VOL < 100"    → [{OPEN: (-∞,10)}, {VOL: (-∞,100)}]
          "OPEN < 10 AND VOL < 100"   → [{OPEN: (-∞,10), VOL: (-∞,100)}]
          None / no constraint        → [{}]  (universal set)
        """
        if expr is None:
            return [{}]

        disjuncts: List[Expr] = []
        if not self._flatten_or(expr, disjuncts):
            return None

        result: List[Dict[str, "RangeConstraint"]] = []
        for disjunct in disjuncts:
            constraint_map = self._extract_range_map(
                disjunct,
                var_name,
                context,
            )
            if constraint_map is None:
                return None
            result.append(constraint_map)
        return result

    def _to_literal(self, expr: Expr) -> Optional[Literal]:
        if isinstance(expr, SemanticsExpr):
            return self._to_literal(expr.expr)
        return expr if isinstance(expr, Literal) else None

    def _reverse_op(self, op: str) -> Optional[str]:
        return {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}.get(op)

    def _is_true_expr(self, expr: Expr) -> bool:
        if not isinstance(expr, BinaryExpr):
            return False
        if expr.op != "=":
            return False
        if not isinstance(expr.left, Literal) or not isinstance(expr.right, Literal):
            return False
        return expr.left.value == 1 and expr.right.value == 1

    def _apply_generic_range_atom(
        self,
        constraints: Dict[str, RangeConstraint],
        atom: Expr,
        var_name: str,
        context: SimplifyContext,
    ) -> bool:
        if isinstance(atom, SemanticsExpr):
            return self._apply_generic_range_atom(
                constraints,
                atom.expr,
                var_name,
                context,
            )

        if isinstance(atom, BetweenExpr):
            col_name = self._extract_simple_column_name(atom.expr, var_name)
            if col_name is None:
                return False
            lower_lit = self._to_literal(atom.lower)
            upper_lit = self._to_literal(atom.upper)
            if lower_lit is None or upper_lit is None:
                return False
            constraint = self._constraint_for_column(
                constraints,
                col_name,
                context,
            )
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
        col_name = self._extract_simple_column_name(left, var_name)
        if col_name is None:
            col_name = self._extract_simple_column_name(right, var_name)
            if col_name is None:
                return False
            reversed_op = self._reverse_op(op)
            if reversed_op is None:
                return False
            op = reversed_op
            right = atom.left

        lit = self._to_literal(right)
        if lit is None:
            return False
        # String ordering is collation-dependent in SQL.  Python's lexical
        # ordering cannot prove that one database string range contains
        # another unless the database collation is known, so do not use string
        # bounds to justify this rewrite.
        if isinstance(lit.value, str):
            return False

        constraint = self._constraint_for_column(
            constraints,
            col_name,
            context,
        )
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

    def _constraint_for_column(
        self,
        constraints: Dict[str, RangeConstraint],
        col_name: str,
        context: SimplifyContext,
    ) -> RangeConstraint:
        """Get or create a typed constraint, resolving schema only once."""
        constraint = constraints.get(col_name)
        if constraint is not None:
            return constraint
        metadata = context.get_column_metadata(ColumnRef(None, col_name))
        domain = RangeDomain.from_data_type(metadata.data_type)
        constraint = RangeConstraint(domain)
        constraints[col_name] = constraint
        return constraint

    def _extract_simple_column_name(self, expr: Expr, var_name: str) -> Optional[str]:
        if isinstance(expr, SemanticsExpr):
            return self._extract_simple_column_name(expr.expr, var_name)
        if not isinstance(expr, ColumnRef):
            return None
        if expr.qualifier is not None and expr.qualifier != var_name:
            return None
        return expr.name

    def _is_strict_range_subset(
        self,
        left_map: Dict[str, RangeConstraint],
        right_map: Dict[str, RangeConstraint],
    ) -> bool:
        if len(left_map) == 0:
            return False
        if len(right_map) == 0:
            return True

        strict = False
        for col_name, right_constraint in right_map.items():
            left_constraint = left_map.get(col_name)
            if left_constraint is None:
                return False
            implies, stronger = self._constraint_implies(
                left_constraint, right_constraint
            )
            if not implies:
                return False
            strict = strict or stronger

        for col_name in left_map.keys():
            if col_name not in right_map:
                strict = True

        return strict

    def _range_map_implies(
        self,
        left_map: Dict[str, RangeConstraint],
        right_map: Dict[str, RangeConstraint],
    ) -> Tuple[bool, bool]:
        """Check if left_map ⊆ right_map, returning (implies, strict).

        Empty map = universal (no constraints). A map with columns = restricted set.
        """
        if len(left_map) == 0:
            # left is universal; only implies right if right is also universal
            return len(right_map) == 0, False
        if len(right_map) == 0:
            # right is universal; left (non-empty) is a strict subset
            return True, True

        strict = False
        for col_name, right_constraint in right_map.items():
            left_constraint = left_map.get(col_name)
            if left_constraint is None:
                return False, False
            implies, stronger = self._constraint_implies(
                left_constraint, right_constraint
            )
            if not implies:
                return False, False
            strict = strict or stronger

        for col_name in left_map.keys():
            if col_name not in right_map:
                strict = True

        return True, strict

    def _is_strict_range_subset_dnf(
        self,
        left_dnf: List[Dict[str, RangeConstraint]],
        right_dnf: List[Dict[str, RangeConstraint]],
    ) -> bool:
        """Check if left_dnf (OR of conjuncts) is a *strict* subset of right_dnf.

        Uses a sufficient (sound but not complete) check: every left disjunct must
        be contained in at least one right disjunct, and at least one such pairing
        must be strict.

        This handles cases like:
          B = OPEN < 10 OR VOL < 100   →  [{OPEN:(-∞,10)}, {VOL:(-∞,100)}]
          C = OPEN < 20 OR VOL < 200   →  [{OPEN:(-∞,20)}, {VOL:(-∞,200)}]
          {OPEN<10} ⊆ {OPEN<20} (strict) and {VOL<100} ⊆ {VOL<200} (strict) → True
        """
        if len(left_dnf) == 0:
            return False

        any_strict = False
        for left_map in left_dnf:
            found = False
            for right_map in right_dnf:
                implies, strict = self._range_map_implies(left_map, right_map)
                if implies:
                    found = True
                    if strict:
                        any_strict = True
                    break
            if not found:
                return False

        # Also consider strict if right_dnf has disjuncts uncovered by left (heuristic)
        if not any_strict and len(right_dnf) > len(left_dnf):
            any_strict = True

        return any_strict

    def _is_range_subset_dnf(
        self,
        left_dnf: List[Dict[str, RangeConstraint]],
        right_dnf: List[Dict[str, RangeConstraint]],
    ) -> bool:
        """Non-strict containment check: returns True if left_dnf ⊆ right_dnf (equal is allowed).

        Every left disjunct must be covered by at least one right disjunct.
        """
        if len(left_dnf) == 0:
            return False
        for left_map in left_dnf:
            found = False
            for right_map in right_dnf:
                implies, _ = self._range_map_implies(left_map, right_map)
                if implies:
                    found = True
                    break
            if not found:
                return False
        return True

    def _are_range_maps_equal_dnf(
        self,
        left_dnf: List[Dict[str, RangeConstraint]],
        right_dnf: List[Dict[str, RangeConstraint]],
    ) -> bool:
        """Check if left_dnf and right_dnf represent equal range regions (mutual containment)."""
        return self._is_range_subset_dnf(
            left_dnf, right_dnf
        ) and self._is_range_subset_dnf(right_dnf, left_dnf)

    def _constraint_implies(
        self, left: RangeConstraint, right: RangeConstraint
    ) -> Tuple[bool, bool]:
        strict = False

        lower_ok, lower_strict = self._lower_bound_implies(
            left.lower, left.lower_inclusive, right.lower, right.lower_inclusive
        )
        if not lower_ok:
            return False, False
        strict = strict or lower_strict

        upper_ok, upper_strict = self._upper_bound_implies(
            left.upper, left.upper_inclusive, right.upper, right.upper_inclusive
        )
        if not upper_ok:
            return False, False
        strict = strict or upper_strict

        return True, strict

    def _lower_bound_implies(
        self,
        left_lit: Optional[Literal],
        left_inclusive: bool,
        right_lit: Optional[Literal],
        right_inclusive: bool,
    ) -> Tuple[bool, bool]:
        if right_lit is None:
            return True, left_lit is not None
        if left_lit is None:
            return False, False

        cmp_result = _compare_literal_values(left_lit.value, right_lit.value)
        if cmp_result is None:
            return False, False
        if cmp_result > 0:
            return True, True
        if cmp_result < 0:
            return False, False
        if left_inclusive and not right_inclusive:
            return False, False
        return True, left_inclusive != right_inclusive

    def _upper_bound_implies(
        self,
        left_lit: Optional[Literal],
        left_inclusive: bool,
        right_lit: Optional[Literal],
        right_inclusive: bool,
    ) -> Tuple[bool, bool]:
        if right_lit is None:
            return True, left_lit is not None
        if left_lit is None:
            return False, False

        cmp_result = _compare_literal_values(left_lit.value, right_lit.value)
        if cmp_result is None:
            return False, False
        if cmp_result < 0:
            return True, True
        if cmp_result > 0:
            return False, False
        if left_inclusive and not right_inclusive:
            return False, False
        return True, left_inclusive != right_inclusive

    def _fixed_prior_occurrence_count(
        self, prefix_patterns: List, target_var: str
    ) -> Optional[int]:
        total = 0
        for pattern in prefix_patterns:
            count = self._fixed_occurrence_count(pattern, target_var)
            if count is None:
                return None
            total += count
        return total

    def _fixed_occurrence_count(self, pattern, target_var: str) -> Optional[int]:
        if isinstance(pattern, VariablePattern):
            return 1 if pattern.name == target_var else 0
        if isinstance(pattern, QuantifiedPattern):
            if pattern.atMost is None or pattern.atLeast != pattern.atMost:
                inner_count = self._fixed_occurrence_count(pattern.pattern, target_var)
                if inner_count in (None, 0):
                    return 0 if inner_count == 0 else None
                return None
            inner_count = self._fixed_occurrence_count(pattern.pattern, target_var)
            if inner_count is None:
                return None
            return inner_count * pattern.atLeast
        if isinstance(pattern, ConcatenationPattern):
            total = 0
            for child in pattern.patterns:
                child_count = self._fixed_occurrence_count(child, target_var)
                if child_count is None:
                    return None
                total += child_count
            return total
        if isinstance(pattern, AlternationPattern):
            counts = [
                self._fixed_occurrence_count(child, target_var)
                for child in pattern.patterns
            ]
            if any(count is None for count in counts):
                return None
            if len(set(counts)) != 1:
                return None
            return counts[0]
        if isinstance(pattern, PermutationPattern):
            total = 0
            for child in pattern.patterns:
                child_count = self._fixed_occurrence_count(child, target_var)
                if child_count is None:
                    return None
                total += child_count
            return total
        if isinstance(pattern, (ExcludedPattern, AnchorPattern)):
            return self._fixed_occurrence_count(pattern.pattern, target_var)
        return 0

    def _build_first_not_overlap_expr(
        self,
        left_expr: Expr,
        left_var: str,
        right_var: str,
        prior_count: int,
        simplify_context,
        source_nullability_is_preserved: bool,
    ) -> Optional[Expr]:
        # A DEFINE predicate accepts a row only when it is TRUE.  Its boundary
        # complement must therefore accept both FALSE and UNKNOWN.  Plain
        # NOT(transformed) rejects UNKNOWN under SQL three-valued logic, which
        # would incorrectly discard broader-side rows containing NULLs.
        #
        # In this generated guard FIRST always addresses an existing occurrence:
        # prior_count is statically fixed, and the current right occurrence (or
        # the COUNT-gated late branch) proves that the requested row exists.  We
        # can therefore complement the source expression first, while its
        # ColumnRefs can still use schema nullability, and add FIRST afterwards.
        # This removes redundant ``FIRST(... ) IS NULL`` predicates for NOT NULL
        # source columns.  An outer join can NULL-extend such columns, so retain
        # the conservative transform-first order for those inputs.
        if source_nullability_is_preserved:
            not_true_expr = _condition_is_not_true(
                left_expr,
                simplify_context,
            )
            return self._rewrite_expr_for_first_of_var(
                not_true_expr,
                left_var,
                right_var,
                prior_count,
            )

        transformed = self._rewrite_expr_for_first_of_var(
            left_expr,
            left_var,
            right_var,
            prior_count,
        )
        if transformed is None:
            return None
        return _condition_is_not_true(transformed, simplify_context)

    def _rewrite_expr_for_first_of_var(
        self,
        expr: Expr,
        source_var: str,
        target_var: str,
        offset: int,
    ) -> Optional[Expr]:
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            inner = self._rewrite_expr_for_first_of_var(
                expr.expr, source_var, target_var, offset
            )
            if inner is None:
                return None
            return SemanticsExpr(expr.semantics, inner)
        if isinstance(expr, Literal):
            return expr
        if isinstance(expr, ColumnRef):
            if expr.qualifier is not None and expr.qualifier != source_var:
                return None
            first_args: List[Expr] = [ColumnRef(target_var, expr.name)]
            if offset > 0:
                first_args.append(Literal(offset))
            return FunctionCall("FIRST", first_args)
        if isinstance(expr, BinaryExpr):
            left = self._rewrite_expr_for_first_of_var(
                expr.left, source_var, target_var, offset
            )
            right = self._rewrite_expr_for_first_of_var(
                expr.right, source_var, target_var, offset
            )
            if left is None or right is None:
                return None
            return BinaryExpr(left, expr.op, right)
        if isinstance(expr, BetweenExpr):
            value_expr = self._rewrite_expr_for_first_of_var(
                expr.expr, source_var, target_var, offset
            )
            lower = self._rewrite_expr_for_first_of_var(
                expr.lower, source_var, target_var, offset
            )
            upper = self._rewrite_expr_for_first_of_var(
                expr.upper, source_var, target_var, offset
            )
            if value_expr is None or lower is None or upper is None:
                return None
            return BetweenExpr(value_expr, lower, upper)
        if isinstance(expr, InExpr):
            value_expr = self._rewrite_expr_for_first_of_var(
                expr.expr, source_var, target_var, offset
            )
            if value_expr is None:
                return None
            values = []
            for value in expr.values:
                rewritten_value = self._rewrite_expr_for_first_of_var(
                    value, source_var, target_var, offset
                )
                if rewritten_value is None:
                    return None
                values.append(rewritten_value)
            return InExpr(value_expr, values, expr.is_not)
        if isinstance(expr, IsNullExpr):
            inner = self._rewrite_expr_for_first_of_var(
                expr.expr,
                source_var,
                target_var,
                offset,
            )
            if inner is None:
                return None
            return IsNullExpr(inner, expr.is_not)
        if isinstance(expr, NotExpr):
            inner = self._rewrite_expr_for_first_of_var(
                expr.expr, source_var, target_var, offset
            )
            if inner is None:
                return None
            return NotExpr(inner)
        if isinstance(expr, FunctionCall):
            return None
        return None

    def _build_count_guard_expr(
        self,
        right_var: str,
        activation_count: int,
        base_expr: Optional[Expr],
        guard_expr: Expr,
    ) -> Expr:
        count_expr = FunctionCall("COUNT", [ColumnRef(right_var, "*")])
        early_branch_terms = [BinaryExpr(count_expr, "<=", Literal(activation_count))]
        late_branch_terms = [
            BinaryExpr(count_expr, ">", Literal(activation_count)),
            guard_expr,
        ]
        if base_expr is not None:
            early_branch_terms.append(base_expr)
            late_branch_terms.append(base_expr)
        return BinaryExpr(
            self._and_exprs(early_branch_terms),
            "OR",
            self._and_exprs(late_branch_terms),
        )

    def _and_exprs(self, exprs: List[Expr]) -> Expr:
        result = None
        for expr in exprs:
            if expr is None:
                continue
            if result is None:
                result = expr
            else:
                result = BinaryExpr(result, "AND", expr)
        return result if result is not None else BinaryExpr(Literal(1), "=", Literal(1))

    def _set_define_condition(
        self, ast: RPRAbstractSyntaxTree, target_var: str, after_var: str, condition
    ):
        if target_var in ast.define.definitions:
            ast.define.definitions[target_var] = condition
            return

        new_defs = {}
        inserted = False
        for var_name, existing in ast.define.definitions.items():
            new_defs[var_name] = existing
            if var_name == after_var:
                new_defs[target_var] = condition
                inserted = True
        if not inserted:
            new_defs[target_var] = condition
        ast.define.definitions = new_defs


adjacent_infinite_quantifier_overlap_test_cases = [
    # B+ C*: conditions of B is the subset of C
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS (FIRST(C.OPEN) >= 10 OR FIRST(C.VOL) >= 100) AND VOL < 200, D AS TYPE = 'D'
)""",
    ),
    # B+ C+: conditions of B is the subset of C; C+ means at least 1 C
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS (COUNT(C.*) <= 1 AND VOL < 200) OR ((COUNT(C.*) > 1 AND (FIRST(C.OPEN) >= 10 OR FIRST(C.VOL) >= 100)) AND VOL < 200), D AS TYPE = 'D'
)""",
    ),
    # B+ C*: conditions of B is the subset of C, and existing C before B
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C{3} A C B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C{3} A C B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS (COUNT(C.*) <= 4 AND VOL < 200) OR ((COUNT(C.*) > 4 AND (FIRST(C.OPEN, 4) >= 10 OR FIRST(C.VOL, 4) >= 100)) AND VOL < 200), D AS TYPE = 'D'
)""",
    ),
    # B+ C*: conditions of B is the subset of C
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS FIRST(C.OPEN) >= 10 OR FIRST(C.VOL) >= 100, D AS TYPE = 'D'
)""",
    ),
    # cannot optimze B+ C* because contains dependent conditions on C
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200 AND C.TYPE = A.TYPE, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200 AND C.TYPE = A.TYPE, D AS TYPE = 'D'
)""",
    ),
    # cannot optimze B+ C* because left has same variable, but number of this variable is non-fixed
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C{2,3} B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C{2,3} B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
    ),
    # same condition: B+ greedy, C* (y=0) with equal range → C* becomes C? to preserve MEASURES references
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS C.OPEN < 10 AND C.VOL < 100, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C? D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS C.OPEN < 10 AND C.VOL < 100, D AS TYPE = 'D'
)""",
    ),
    # not subset condition
    (
        8,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2,}? C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2,}? C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10, C AS VOL < 200, D AS TYPE = 'D'
)""",
    ),
    # B{2,}? C+: conditions of B is the subset of C
    (
        9,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2,}? C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, C.TS AS CTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2} C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
    ),
    # C B{2,}? C+: conditions of B is the subset of C
    (
        10,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C B+ C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A C B+ C+ D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS (COUNT(C.*) <= 2 AND VOL < 200) OR ((COUNT(C.*) > 2 AND (FIRST(C.OPEN, 1) >= 10 OR FIRST(C.VOL, 1) >= 100)) AND VOL < 200), D AS TYPE = 'D'
)""",
    ),
    # left: non-greedy, right: greedy, overlap condition holds
    (
        11,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2,}? C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B{2} C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 200, D AS TYPE = 'D'
)""",
    ),
    # left: non-greedy, right: greedy, overlap condition cannot hold
    (
        12,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+? C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 50, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+? C* D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 50, D AS TYPE = 'D'
)""",
    ),
    # B+? C*? non-greedy
    (
        13,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+? C*? D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 500, D AS TYPE = 'D'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.ID AS AID, B.TS AS BTS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+? C*? D)
    DEFINE A AS TS > 5, B AS OPEN < 10 AND VOL < 100, C AS VOL < 500, D AS TYPE = 'D'
)""",
    ),
]


adjacent_infinite_quantifier_overlap_nullable_test_cases = [
    # A nullable source column still needs the IS NULL branch: NULL makes the
    # left DEFINE predicate UNKNOWN, which means the row does not belong to A
    # but may validly belong to the unconstrained B variable.
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES COUNT(A.*) AS A_COUNT, COUNT(B.*) AS B_COUNT
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(S A+ B* D)
    DEFINE S AS KIND = 'S',
           A AS X < 10,
           D AS KIND = 'D'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES COUNT(A.*) AS A_COUNT, COUNT(B.*) AS B_COUNT
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(S A+ B* D)
    DEFINE S AS KIND = 'S',
           A AS X < 10,
           B AS (FIRST(B.X) IS NULL) OR FIRST(B.X) >= 10,
           D AS KIND = 'D'
)""",
    ),
]


_nullable_test_schemas = {
    "T": {
        "columns": {
            "TS": "INT",
            "KIND": "VARCHAR(20)",
            "X": "INT",
        },
        "column_nullables": {
            "TS": False,
            "KIND": False,
            "X": True,
        },
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    results = [
        run_rule_tests(
            QuantifierScopeShorteningRule,
            adjacent_infinite_quantifier_overlap_test_cases,
            "adjacent infinite quantifier overlap rewrite",
            verbose,
        ),
        run_rule_tests(
            QuantifierScopeShorteningRule,
            adjacent_infinite_quantifier_overlap_nullable_test_cases,
            "adjacent infinite quantifier overlap nullable guard",
            verbose,
            default_table_schemas=_nullable_test_schemas,
        ),
    ]
    return {
        "label": "adjacent infinite quantifier overlap rewrite",
        "total": sum(result["total"] for result in results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(result["failed"] for result in results),
        "failures": [failure for result in results for failure in result["failures"]],
    }


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
