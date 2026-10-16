if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import FrozenSet, List, Optional, Tuple, Union

from expression import NullLiteral

from .utils import *

CountedVar = Optional[Union[str, FrozenSet[str]]]


@dataclass(frozen=True)
class CountConstraint:
    counted_var: CountedVar
    op: str
    lower: int
    upper: Optional[int]


@dataclass(frozen=True)
class RewriteAttempt:
    pattern: object
    changed: bool


@dataclass(frozen=True)
class AdjustableQuantifier:
    node: object
    unit_count: int


@dataclass(frozen=True)
class TargetPath:
    min_count: int
    adjustable: List[AdjustableQuantifier]


@dataclass
class RewritePlan:
    pattern: object
    pattern_changed: bool
    simplified_exprs: dict


class CountRangePushdownRule(RewriteRule):
    """
    Simplify RUNNING COUNT predicates and tighten PATTERN occurrence ranges.

    PATTERN and DEFINE are evaluated to a fixed point, so a simplified
    expression can expose another range constraint.  COUNT-vs-COUNT predicates
    are folded when their ranges prove a result; otherwise exact operands are
    substituted and overlapping ranges are kept unchanged.

    In DEFINE, FINAL is not permitted; RUNNING may be used for added clarity if desired.
    """

    @property
    def name(self) -> str:
        return "count_range_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        self._pending_plan = (self._plan_key(ast), plan)
        return plan is not None and (
            plan.pattern_changed or bool(plan.simplified_exprs)
        )

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        key = self._plan_key(ast)
        pending = getattr(self, "_pending_plan", None)
        plan = (
            pending[1]
            if pending is not None and pending[0] == key
            else self._build_plan(ast)
        )
        self._pending_plan = None
        if plan is None or (not plan.pattern_changed and not plan.simplified_exprs):
            return False

        old_vars = {name.upper() for name in ast.pattern.get_variables()}
        new_vars = {name.upper() for name in plan.pattern.get_variables()}
        removed_vars = old_vars - new_vars
        cleaned_subsets, empty_subsets = self._clean_subsets_after_removed_vars(
            ast.subsets, removed_vars
        )
        if self._skip_references_removed_var(
            ast.after_match_skip_to, removed_vars | empty_subsets
        ):
            # A skip target is structural, not a value expression.  It cannot
            # be replaced with NULL/0, so retain the original query.
            return False

        ast.pattern = plan.pattern
        changed = plan.pattern_changed
        parser = ExprParser()

        for var_name, expression in plan.simplified_exprs.items():
            if var_name.upper() in removed_vars:
                continue
            ast.define.definitions[var_name] = parser._analyze_dependency(
                var_name, expression
            )
            changed = True

        if removed_vars:
            empty_vars = removed_vars | empty_subsets
            definitions = {}
            for var_name, condition in ast.define.definitions.items():
                if var_name.upper() in removed_vars:
                    changed = True
                    continue
                rewritten = self._rewrite_expr_after_removed_vars(
                    condition.expression, empty_vars
                )
                definitions[var_name] = parser._analyze_dependency(var_name, rewritten)
                changed = (
                    changed or rewritten.to_string() != condition.expression.to_string()
                )
            ast.define.definitions = definitions

            if ast.measures is not None:
                ast.measures.measures = {
                    alias: self._rewrite_expr_after_removed_vars(expr, empty_vars)
                    for alias, expr in ast.measures.measures.items()
                }

            ast.subsets = cleaned_subsets or None

        return changed

    def _clean_subsets_after_removed_vars(self, subsets, removed_vars):
        cleaned = {}
        empty = set()
        for subset_name, members in (subsets or {}).items():
            kept = [member for member in members if member.upper() not in removed_vars]
            if kept:
                cleaned[subset_name] = kept
            else:
                empty.add(subset_name.upper())
        return cleaned, empty

    def _skip_references_removed_var(self, skip_target, removed_vars) -> bool:
        if not skip_target or not removed_vars:
            return False
        tokens = set(re.findall(r"[A-Z_][A-Z0-9_$#]*", skip_target.upper()))
        return bool(tokens & removed_vars)

    def _expr_references_empty_var(self, expr: Expr, empty_vars) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ColumnRef):
            return bool(expr.qualifier and expr.qualifier.upper() in empty_vars)
        if isinstance(expr, SemanticsExpr):
            return self._expr_references_empty_var(expr.expr, empty_vars)
        if isinstance(expr, FunctionCall):
            if expr.name.upper() == "CLASSIFIER" and expr.args:
                arg = expr.args[0]
                if (
                    isinstance(arg, ColumnRef)
                    and arg.qualifier is None
                    and arg.name.upper() in empty_vars
                ):
                    return True
            return any(
                self._expr_references_empty_var(arg, empty_vars) for arg in expr.args
            )
        if isinstance(expr, BinaryExpr):
            return self._expr_references_empty_var(
                expr.left, empty_vars
            ) or self._expr_references_empty_var(expr.right, empty_vars)
        if isinstance(expr, BetweenExpr):
            return any(
                self._expr_references_empty_var(item, empty_vars)
                for item in (expr.expr, expr.lower, expr.upper)
            )
        if isinstance(expr, InExpr):
            return self._expr_references_empty_var(expr.expr, empty_vars) or any(
                self._expr_references_empty_var(value, empty_vars)
                for value in expr.values
            )
        if isinstance(expr, (IsNullExpr, NotExpr)):
            return self._expr_references_empty_var(expr.expr, empty_vars)
        if isinstance(expr, CaseExpr):
            return any(
                self._expr_references_empty_var(condition, empty_vars)
                or self._expr_references_empty_var(result, empty_vars)
                for condition, result in expr.cases
            ) or self._expr_references_empty_var(expr.else_expr, empty_vars)
        return False

    def _rewrite_expr_after_removed_vars(self, expr: Expr, empty_vars) -> Expr:
        if expr is None:
            return None
        if isinstance(expr, SemanticsExpr):
            rewritten = self._rewrite_expr_after_removed_vars(expr.expr, empty_vars)
            if isinstance(rewritten, (Literal, NullLiteral)):
                return rewritten
            return SemanticsExpr(expr.semantics, rewritten)
        if isinstance(expr, ColumnRef):
            if expr.qualifier and expr.qualifier.upper() in empty_vars:
                return NullLiteral()
            return expr
        if isinstance(expr, FunctionCall):
            references_empty = self._expr_references_empty_var(expr, empty_vars)
            if references_empty:
                name = expr.name.upper()
                if name == "COUNT":
                    return Literal(0)
                if name in {
                    "SUM",
                    "AVG",
                    "MIN",
                    "MAX",
                    "FIRST",
                    "LAST",
                    "PREV",
                    "NEXT",
                    "CLASSIFIER",
                }:
                    return NullLiteral()
            return FunctionCall(
                expr.name,
                [
                    self._rewrite_expr_after_removed_vars(arg, empty_vars)
                    for arg in expr.args
                ],
            )
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(
                self._rewrite_expr_after_removed_vars(expr.left, empty_vars),
                expr.op,
                self._rewrite_expr_after_removed_vars(expr.right, empty_vars),
            )
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(
                self._rewrite_expr_after_removed_vars(expr.expr, empty_vars),
                self._rewrite_expr_after_removed_vars(expr.lower, empty_vars),
                self._rewrite_expr_after_removed_vars(expr.upper, empty_vars),
            )
        if isinstance(expr, InExpr):
            return InExpr(
                self._rewrite_expr_after_removed_vars(expr.expr, empty_vars),
                [
                    self._rewrite_expr_after_removed_vars(value, empty_vars)
                    for value in expr.values
                ],
                expr.is_not,
            )
        if isinstance(expr, IsNullExpr):
            return IsNullExpr(
                self._rewrite_expr_after_removed_vars(expr.expr, empty_vars),
                expr.is_not,
            )
        if isinstance(expr, NotExpr):
            return NotExpr(self._rewrite_expr_after_removed_vars(expr.expr, empty_vars))
        if isinstance(expr, CaseExpr):
            return CaseExpr(
                [
                    (
                        self._rewrite_expr_after_removed_vars(condition, empty_vars),
                        self._rewrite_expr_after_removed_vars(result, empty_vars),
                    )
                    for condition, result in expr.cases
                ],
                (
                    self._rewrite_expr_after_removed_vars(expr.else_expr, empty_vars)
                    if expr.else_expr is not None
                    else None
                ),
            )
        return expr

    def _plan_key(self, ast: RPRAbstractSyntaxTree):
        definitions = getattr(getattr(ast, "define", None), "definitions", {})
        subsets = getattr(ast, "subsets", None) or {}
        return (
            id(ast),
            id(ast.pattern),
            tuple(
                (str(name).upper(), tuple(str(member).upper() for member in members))
                for name, members in sorted(subsets.items())
            ),
            tuple(
                (name, id(condition), id(condition.expression))
                for name, condition in definitions.items()
            ),
        )

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        """Compute the PATTERN/DEFINE fixed point for all COUNT predicates."""
        self._subset_members = {
            str(name).upper(): frozenset(str(member).upper() for member in members)
            for name, members in (getattr(ast, "subsets", None) or {}).items()
        }
        pattern = ast.pattern
        expressions = {
            var_name: condition.expression
            for var_name, condition in ast.define.definitions.items()
            if condition is not None and condition.expression is not None
        }
        original_expr_strings = {
            var_name: expression.to_string()
            for var_name, expression in expressions.items()
        }
        pattern_changed = False

        while True:
            self._reset_analysis_caches()
            pass_changed = False
            must_appear_vars = pattern.get_must_appear_var()

            constraints = list(
                self._iter_expression_constraints(
                    expressions, must_appear_vars, pattern
                )
            )
            for var_name, constraint in constraints:
                if self._running_self_constraint_impossible(
                    pattern, var_name, constraint
                ):
                    false_expr = self._false_expr()
                    if expressions[var_name].to_string() != false_expr.to_string():
                        expressions[var_name] = false_expr
                        pass_changed = True
                    continue

                self_rewritten, _ = self._tighten_running_self_count(
                    pattern, var_name, constraint
                )
                attempt = self._apply_count_constraint(
                    self_rewritten, var_name, constraint
                )
                candidate = attempt.pattern.simplify()
                if candidate.to_string() != pattern.to_string():
                    pattern = candidate
                    pattern_changed = True
                    pass_changed = True
                    self._reset_analysis_caches()

            for var_name, expression in list(expressions.items()):
                if not self._contains_var(pattern, var_name):
                    continue
                simplified, changed = self._simplify_count_expression(
                    expression, pattern, var_name
                )
                if changed and simplified.to_string() != expression.to_string():
                    expressions[var_name] = simplified
                    pass_changed = True

            if not pass_changed:
                break

        simplified_exprs = {
            var_name: expression
            for var_name, expression in expressions.items()
            if expression.to_string() != original_expr_strings[var_name]
        }
        return RewritePlan(pattern, pattern_changed, simplified_exprs)

    def _reset_analysis_caches(self):
        self._count_range_cache = {}
        self._contains_var_cache = {}

    def _iter_expression_constraints(self, expressions, must_appear_vars, pattern):
        for var_name, expression in expressions.items():
            if var_name not in must_appear_vars:
                continue
            for atom in self._flatten_and(expression):
                constraint = self._parse_count_constraint(atom)
                if constraint is None:
                    continue
                if constraint.counted_var is not None and not self._counted_var_occurs(
                    pattern, constraint.counted_var
                ):
                    continue
                yield var_name, constraint

    def _simplify_count_expression(self, expr: Expr, pattern, target_var: str):
        """Fold provable COUNT comparisons and inline exact COUNT values."""
        if isinstance(expr, SemanticsExpr):
            inner, changed = self._simplify_count_expression(
                expr.expr, pattern, target_var
            )
            return (
                (SemanticsExpr(expr.semantics, inner), True)
                if changed
                else (expr, False)
            )

        if isinstance(expr, NotExpr):
            inner, changed = self._simplify_count_expression(
                expr.expr, pattern, target_var
            )
            value = self._constant_boolean(inner)
            if value is not None:
                return self._false_expr() if value else self._true_expr(), True
            return (NotExpr(inner), True) if changed else (expr, False)

        if not isinstance(expr, BinaryExpr):
            return expr, False

        left, left_changed = self._simplify_count_expression(
            expr.left, pattern, target_var
        )
        right, right_changed = self._simplify_count_expression(
            expr.right, pattern, target_var
        )
        candidate = BinaryExpr(left, expr.op, right)

        if expr.op in ("AND", "OR"):
            left_bool = self._constant_boolean(left)
            right_bool = self._constant_boolean(right)
            if expr.op == "AND":
                if left_bool is False or right_bool is False:
                    return self._false_expr(), True
                if left_bool is True:
                    return right, True
                if right_bool is True:
                    return left, True
            else:
                if left_bool is True or right_bool is True:
                    return self._true_expr(), True
                if left_bool is False:
                    return right, True
                if right_bool is False:
                    return left, True
            return candidate, left_changed or right_changed

        if expr.op not in ("<", "<=", "=", "!=", "<>", ">", ">="):
            return candidate, left_changed or right_changed

        left_range = self._count_operand_range(left, pattern, target_var)
        right_range = self._count_operand_range(right, pattern, target_var)
        if left_range is None or right_range is None:
            return candidate, left_changed or right_changed

        truth = self._compare_count_ranges(left_range, expr.op, right_range)
        if truth is not None:
            return self._true_expr() if truth else self._false_expr(), True

        new_left = Literal(left_range[0]) if left_range[1] == left_range[0] else left
        new_right = (
            Literal(right_range[0]) if right_range[1] == right_range[0] else right
        )
        changed = (
            left_changed
            or right_changed
            or new_left is not left
            or new_right is not right
        )
        return BinaryExpr(new_left, expr.op, new_right), changed

    def _tighten_running_self_count(
        self, pattern, target_var: str, constraint: CountConstraint
    ):
        """Cap a unique target quantifier using RUNNING COUNT(target.*).

        At the nth occurrence of the target, COUNT(target.*) is n.  An upper
        bound therefore caps the consecutive target quantifier.  We only do
        this when the target has one syntactic occurrence in PATTERN; with
        multiple occurrences, the count has a non-zero path-dependent offset.
        """
        if constraint.counted_var != target_var or constraint.upper is None:
            return pattern, False
        if constraint.lower > 1 or constraint.upper < 1:
            return pattern, False
        if self._variable_node_count(pattern, target_var) != 1:
            return pattern, False

        return self._cap_direct_target_quantifier(pattern, target_var, constraint.upper)

    def _running_self_constraint_impossible(
        self, pattern, target_var: str, constraint: CountConstraint
    ) -> bool:
        if constraint.counted_var != target_var:
            return False
        if self._variable_node_count(pattern, target_var) != 1:
            return False

        # The first target row is tested with COUNT(target.*) = 1.  If it
        # fails, no later occurrence can ever be reached under RUNNING DEFINE.
        if constraint.lower > 1 or (
            constraint.upper is not None and constraint.upper < 1
        ):
            return True

        minimum = self._direct_target_quantifier_min(pattern, target_var)
        return (
            minimum is not None
            and constraint.upper is not None
            and minimum > constraint.upper
        )

    def _direct_target_quantifier_min(self, node, target_var: str):
        if (
            isinstance(node, QuantifiedPattern)
            and isinstance(node.pattern, VariablePattern)
            and node.pattern.name == target_var
        ):
            return node.atLeast
        if isinstance(node, QuantifiedPattern):
            return self._direct_target_quantifier_min(node.pattern, target_var)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._direct_target_quantifier_min(node.pattern, target_var)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            values = [
                value
                for value in (
                    self._direct_target_quantifier_min(child, target_var)
                    for child in node.patterns
                )
                if value is not None
            ]
            return values[0] if len(values) == 1 else None
        if isinstance(node, VariablePattern) and node.name == target_var:
            return 1
        return None

    def _variable_node_count(self, node, var_name: str) -> int:
        if isinstance(node, VariablePattern):
            return int(node.name == var_name)
        if isinstance(node, QuantifiedPattern):
            return self._variable_node_count(node.pattern, var_name)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._variable_node_count(node.pattern, var_name)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return sum(
                self._variable_node_count(child, var_name) for child in node.patterns
            )
        return 0

    def _cap_direct_target_quantifier(self, node, target_var: str, upper: int):
        if (
            isinstance(node, QuantifiedPattern)
            and isinstance(node.pattern, VariablePattern)
            and node.pattern.name == target_var
        ):
            new_upper = upper if node.atMost is None else min(node.atMost, upper)
            if new_upper < node.atLeast or new_upper == node.atMost:
                return node, False
            return (
                self._make_quantified(
                    node.pattern, node.atLeast, new_upper, node.isGreedy
                ),
                True,
            )

        if isinstance(node, QuantifiedPattern):
            inner, changed = self._cap_direct_target_quantifier(
                node.pattern, target_var, upper
            )
            if changed:
                return (
                    QuantifiedPattern(inner, node.atLeast, node.atMost, node.isGreedy),
                    True,
                )
            return node, False

        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            changed = False
            children = []
            for child in node.patterns:
                rewritten, child_changed = self._cap_direct_target_quantifier(
                    child, target_var, upper
                )
                children.append(rewritten)
                changed = changed or child_changed
            return (type(node)(children), True) if changed else (node, False)

        if isinstance(node, ExcludedPattern):
            inner, changed = self._cap_direct_target_quantifier(
                node.pattern, target_var, upper
            )
            return (ExcludedPattern(inner), True) if changed else (node, False)
        if isinstance(node, AnchorPattern):
            inner, changed = self._cap_direct_target_quantifier(
                node.pattern, target_var, upper
            )
            return (
                (AnchorPattern(inner, node.anchor_type), True)
                if changed
                else (node, False)
            )
        return node, False

    def _count_operand_range(self, expr: Expr, pattern, target_var: str):
        expr = self._unwrap_semantics(expr)
        if (
            isinstance(expr, Literal)
            and isinstance(expr.value, (Real, Decimal))
            and not isinstance(expr.value, bool)
        ):
            return expr.value, expr.value
        counted_var = self._extract_count_star_var(expr)
        if counted_var is False:
            return None
        counted_var = self._resolve_counted_var(counted_var)
        if counted_var is not None and not self._counted_var_occurs(
            pattern, counted_var
        ):
            # A previous fixed-point pass may have restricted this variable to
            # zero occurrences and removed its EmptyPattern from concatenation.
            return 0, 0
        return self._count_range_at_target(pattern, counted_var, target_var)

    def _compare_count_ranges(self, left, op: str, right):
        left_min, left_max = left
        right_min, right_max = right
        if op == "<":
            if left_max is not None and left_max < right_min:
                return True
            if right_max is not None and left_min >= right_max:
                return False
        elif op == "<=":
            if left_max is not None and left_max <= right_min:
                return True
            if right_max is not None and left_min > right_max:
                return False
        elif op == ">":
            return self._compare_count_ranges(right, "<", left)
        elif op == ">=":
            return self._compare_count_ranges(right, "<=", left)
        elif op == "=":
            if left_max == left_min == right_min == right_max:
                return True
            if (left_max is not None and left_max < right_min) or (
                right_max is not None and right_max < left_min
            ):
                return False
        elif op in ("!=", "<>"):
            equality = self._compare_count_ranges(left, "=", right)
            return None if equality is None else not equality
        return None

    def _constant_boolean(self, expr: Expr):
        if not isinstance(expr, BinaryExpr) or expr.op not in ("=", "!=", "<>"):
            return None
        if not isinstance(expr.left, Literal) or not isinstance(expr.right, Literal):
            return None
        equal = expr.left.value == expr.right.value
        return equal if expr.op == "=" else not equal

    def _true_expr(self):
        return BinaryExpr(Literal(1), "=", Literal(1))

    def _false_expr(self):
        return BinaryExpr(Literal(1), "!=", Literal(1))

    def _parse_count_constraint(self, expr: Expr):
        if not isinstance(expr, BinaryExpr) or expr.op not in (
            "<",
            "<=",
            "=",
            ">",
            ">=",
        ):
            return None

        left = self._unwrap_semantics(expr.left)
        right = self._unwrap_semantics(expr.right)
        op = expr.op

        if (
            isinstance(left, Literal)
            and self._extract_count_star_var(right) is not False
        ):
            left, right = right, left
            op = self._reverse_comparison_op(op)

        count_var = self._extract_count_star_var(left)
        if count_var is False:
            return None
        count_var = self._resolve_counted_var(count_var)
        if op not in ("<", "<=", "=", ">", ">="):
            return None
        if (
            not isinstance(right, Literal)
            or isinstance(right.value, bool)
            or not isinstance(right.value, (Real, Decimal))
        ):
            return None

        bounds = self._integer_count_bounds(op, right.value)
        if bounds is None:
            return None
        lower, upper = bounds
        return CountConstraint(count_var, op, lower, upper)

    def _integer_count_bounds(self, op: str, value):
        # Do not coerce integers to float: values above 2**53 would silently
        # lose precision. Non-finite float literals cannot define useful bounds.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if op == "<":
            return 0, math.ceil(value) - 1
        if op == "<=":
            return 0, math.floor(value)
        if op == ">":
            return max(0, math.floor(value) + 1), None
        if op == ">=":
            return max(0, math.ceil(value)), None
        if op == "=":
            if (isinstance(value, float) and not value.is_integer()) or (
                isinstance(value, Decimal) and value != value.to_integral_value()
            ):
                return None
            target = int(value)
            if target < 0:
                return None
            return target, target
        return None

    def _apply_count_constraint(
        self, pattern, target_var: str, constraint: CountConstraint
    ) -> RewriteAttempt:
        rewritten, changed = self._tighten_count_range_to_target(
            pattern, target_var, constraint
        )

        exact_changed = False
        if changed and constraint.op == "=":
            rewritten, exact_changed = self._restrict_exact_count_before_target(
                rewritten, target_var, constraint.counted_var, constraint.lower
            )

        lower_changed = False
        if not exact_changed and constraint.lower > 0:
            rewritten, lower_changed = (
                self._tighten_single_adjustable_lower_before_target(
                    rewritten, target_var, constraint.counted_var, constraint.lower
                )
            )

        return RewriteAttempt(rewritten, changed or exact_changed or lower_changed)

    def _tighten_count_range_to_target(
        self, pattern, target_var: str, constraint: CountConstraint
    ):
        changed = False
        rewritten = pattern

        if constraint.lower > 0:
            lowered, lower_changed = self._tighten_lower_bound_to_target(
                rewritten, target_var, constraint
            )
            if lower_changed:
                rewritten = lowered
            changed = changed or lower_changed

        if constraint.upper is not None:
            uppered, upper_changed = self._tighten_upper_bound_to_target(
                rewritten, target_var, constraint
            )
            if upper_changed:
                rewritten = uppered
            changed = changed or upper_changed

        return rewritten, changed

    def _tighten_lower_bound_to_target(
        self, pattern, target_var: str, constraint: CountConstraint
    ):
        rewritten, changed, _ = self._tighten_until_target(
            pattern, target_var, constraint, 0, "left"
        )
        return rewritten, changed

    def _tighten_upper_bound_to_target(
        self, pattern, target_var: str, constraint: CountConstraint
    ):
        distributed_pattern, distributed = (
            self._distribute_counted_window_before_target(
                pattern, target_var, constraint.counted_var, "right"
            )
        )
        rewritten, tightened, _ = self._tighten_until_target(
            distributed_pattern, target_var, constraint, 0, "right"
        )
        return rewritten, distributed or tightened

    def _distribute_counted_window_before_target(
        self,
        node,
        target_var: str,
        counted_var: Optional[str],
        target_side: str = "left",
    ):
        if isinstance(node, ConcatenationPattern):
            target_idx = self._target_child_index(
                node.patterns, target_var, target_side
            )

            if target_idx is not None:
                patterns = list(node.patterns)
                prefix = patterns[:target_idx]

                # The target itself may be inside an alternation.  In that
                # case the relevant quantity is not the size of each complete
                # branch, but the count accumulated up to the target in that
                # branch.  For example, B C B and C B C both contain three
                # rows, while their rightmost B is reached after 3 and 2 rows.
                target_child = patterns[target_idx]
                if isinstance(
                    target_child, AlternationPattern
                ) and self._alternation_has_branch_dependent_count_to_target(
                    target_child, counted_var, target_var, target_side
                ):
                    adjustable_indexes = [
                        idx
                        for idx, item in enumerate(prefix)
                        if self._has_adjustable_counted_quantifier(item, counted_var)
                    ]
                    if adjustable_indexes:
                        start = min(adjustable_indexes)
                        left = prefix[start:]
                        branches = [
                            ConcatenationPattern(left + [branch]).simplify()
                            for branch in target_child.patterns
                        ]
                        distributed = AlternationPattern(branches)
                        new_patterns = (
                            patterns[:start]
                            + [distributed]
                            + patterns[target_idx + 1 :]
                        )
                        return ConcatenationPattern(new_patterns), True

                for alt_idx, child in enumerate(prefix):
                    if not isinstance(child, AlternationPattern):
                        continue
                    if not self._alternation_has_branch_dependent_count(
                        child, counted_var
                    ):
                        continue
                    if not any(
                        self._count_max_possible(item, counted_var)
                        for item in prefix[:alt_idx]
                    ):
                        if not any(
                            self._count_max_possible(item, counted_var)
                            for item in prefix[alt_idx + 1 :]
                        ):
                            continue

                    counted_indexes = [
                        i
                        for i, item in enumerate(prefix)
                        if self._count_max_possible(item, counted_var)
                    ]
                    start = min(i for i in counted_indexes if i <= alt_idx)
                    end = max(i for i in counted_indexes if i >= alt_idx)

                    if start == alt_idx and end == alt_idx:
                        continue

                    left = prefix[start:alt_idx]
                    right = prefix[alt_idx + 1 : end + 1]
                    branches = [
                        ConcatenationPattern(left + [branch] + right).simplify()
                        for branch in child.patterns
                    ]
                    distributed = AlternationPattern(branches)
                    new_patterns = (
                        patterns[:start] + [distributed] + patterns[end + 1 :]
                    )
                    return ConcatenationPattern(new_patterns), True

                child = patterns[target_idx]
                new_child, changed = self._distribute_counted_window_before_target(
                    child, target_var, counted_var, target_side
                )
                if changed:
                    patterns[target_idx] = new_child
                    return ConcatenationPattern(patterns), True
                return node, False

            changed = False
            new_patterns = []
            for child in node.patterns:
                new_child, child_changed = (
                    self._distribute_counted_window_before_target(
                        child, target_var, counted_var, target_side
                    )
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
            if changed:
                return ConcatenationPattern(new_patterns), True
            return node, False

        if isinstance(node, QuantifiedPattern):
            inner, changed = self._distribute_counted_window_before_target(
                node.pattern, target_var, counted_var, target_side
            )
            if changed:
                return (
                    QuantifiedPattern(inner, node.atLeast, node.atMost, node.isGreedy),
                    True,
                )
            return node, False

        if isinstance(node, AlternationPattern):
            changed = False
            new_patterns = []
            for child in node.patterns:
                new_child, child_changed = (
                    self._distribute_counted_window_before_target(
                        child, target_var, counted_var, target_side
                    )
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
            if changed:
                return AlternationPattern(new_patterns), True
            return node, False

        if isinstance(node, PermutationPattern):
            return node, False

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            inner, changed = self._distribute_counted_window_before_target(
                node.pattern, target_var, counted_var, target_side
            )
            if not changed:
                return node, False
            if isinstance(node, ExcludedPattern):
                return ExcludedPattern(inner), True
            return AnchorPattern(inner, node.anchor_type), True

        return node, False

    def _alternation_has_branch_dependent_count(
        self, node: AlternationPattern, counted_var: Optional[str]
    ) -> bool:
        """Return whether choosing a branch changes the counted window size.

        Moving surrounding nodes into every branch only helps when the branches
        contribute different count ranges.  If every branch contributes the
        same exact count, the surrounding quantifier can be tightened in place;
        distributing it would only make the pattern larger.
        """
        ranges = [
            self._count_range_in_pattern(branch, counted_var)
            for branch in node.patterns
        ]
        return bool(ranges) and any(
            count_range != ranges[0] for count_range in ranges[1:]
        )

    def _alternation_has_branch_dependent_count_to_target(
        self,
        node: AlternationPattern,
        counted_var: Optional[str],
        target_var: str,
        target_side: str,
    ) -> bool:
        counts = [
            self._count_min_until_target(branch, counted_var, target_var, target_side)
            for branch in node.patterns
            if self._contains_var(branch, target_var)
        ]
        return bool(counts) and any(count != counts[0] for count in counts[1:])

    def _has_adjustable_counted_quantifier(
        self, node, counted_var: Optional[str]
    ) -> bool:
        if isinstance(node, QuantifiedPattern):
            if self._count_min(node.pattern, counted_var) > 0 and (
                node.atMost is None or node.atMost > node.atLeast
            ):
                return True
            return self._has_adjustable_counted_quantifier(node.pattern, counted_var)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(
                self._has_adjustable_counted_quantifier(child, counted_var)
                for child in node.patterns
            )
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._has_adjustable_counted_quantifier(node.pattern, counted_var)
        return False

    def _tighten_until_target(
        self,
        node,
        target_var: str,
        constraint: CountConstraint,
        prior_min: int,
        target_side: str = "left",
    ) -> Tuple[object, bool, bool]:
        if isinstance(node, VariablePattern):
            return node, False, node.name == target_var

        if isinstance(node, QuantifiedPattern):
            if not self._contains_var(node.pattern, target_var):
                rewritten, changed = self._tighten_whole_before_target(
                    node, constraint, prior_min
                )
                return rewritten, changed, False
            inner, changed, contains_target = self._tighten_until_target(
                node.pattern, target_var, constraint, prior_min, target_side
            )
            if changed:
                return (
                    QuantifiedPattern(inner, node.atLeast, node.atMost, node.isGreedy),
                    True,
                    contains_target,
                )
            return node, False, contains_target

        if isinstance(node, ConcatenationPattern):
            changed = False
            found_target = False
            current_prior = prior_min
            new_patterns = []
            target_idx = self._target_child_index(
                node.patterns, target_var, target_side
            )
            suffix_to_target = [0] * (len(node.patterns) + 1)
            if target_idx is not None:
                for idx in range(target_idx, -1, -1):
                    if idx == target_idx:
                        target_min = self._count_min_until_target(
                            node.patterns[idx],
                            constraint.counted_var,
                            target_var,
                            target_side,
                        )
                        suffix_to_target[idx] = suffix_to_target[idx + 1] + target_min
                    else:
                        suffix_to_target[idx] = suffix_to_target[
                            idx + 1
                        ] + self._count_min(node.patterns[idx], constraint.counted_var)

            for idx, child in enumerate(node.patterns):
                if found_target:
                    new_patterns.append(child)
                    continue

                if self._contains_var(child, target_var) and (
                    target_idx is None or idx == target_idx
                ):
                    new_child, child_changed, child_contains = (
                        self._tighten_until_target(
                            child, target_var, constraint, current_prior, target_side
                        )
                    )
                    new_patterns.append(new_child)
                    changed = changed or child_changed
                    found_target = found_target or child_contains
                    continue

                new_child, child_changed = self._tighten_whole_before_target(
                    child, constraint, current_prior + suffix_to_target[idx + 1]
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
                current_prior += self._count_min(new_child, constraint.counted_var)

            if changed:
                return ConcatenationPattern(new_patterns), True, found_target
            return node, False, found_target

        if isinstance(node, AlternationPattern):
            changed = False
            contains_target = False
            new_patterns = []
            for child in node.patterns:
                if not self._contains_var(child, target_var):
                    new_patterns.append(child)
                    continue
                new_child, child_changed, child_contains = self._tighten_until_target(
                    child, target_var, constraint, prior_min, target_side
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
                contains_target = contains_target or child_contains
            if changed:
                return AlternationPattern(new_patterns), True, contains_target
            return node, False, contains_target

        if isinstance(node, PermutationPattern):
            return node, False, self._contains_var(node, target_var)

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            inner, changed, contains_target = self._tighten_until_target(
                node.pattern, target_var, constraint, prior_min, target_side
            )
            if not changed:
                return node, False, contains_target
            if isinstance(node, ExcludedPattern):
                return ExcludedPattern(inner), True, contains_target
            return AnchorPattern(inner, node.anchor_type), True, contains_target

        return node, False, False

    def _target_child_index(
        self, patterns: List, target_var: str, target_side: str
    ) -> Optional[int]:
        indexes = [
            idx
            for idx, child in enumerate(patterns)
            if self._contains_var(child, target_var)
        ]
        if not indexes:
            return None
        return indexes[-1] if target_side == "right" else indexes[0]

    def _count_min_until_target(
        self, node, counted_var: Optional[str], target_var: str, target_side: str
    ) -> int:
        if isinstance(node, VariablePattern):
            return 1 if self._var_counts(node.name, counted_var) else 0

        if isinstance(node, QuantifiedPattern):
            if target_side == "right":
                return node.atLeast * self._count_min(node.pattern, counted_var)
            return self._count_min_until_target(
                node.pattern, counted_var, target_var, target_side
            )

        if isinstance(node, ConcatenationPattern):
            target_idx = self._target_child_index(
                node.patterns, target_var, target_side
            )
            if target_idx is None:
                return self._count_min(node, counted_var)
            total = 0
            for idx, child in enumerate(node.patterns[:target_idx]):
                total += self._count_min(child, counted_var)
            total += self._count_min_until_target(
                node.patterns[target_idx], counted_var, target_var, target_side
            )
            return total

        if isinstance(node, AlternationPattern):
            values = [
                self._count_min_until_target(
                    child, counted_var, target_var, target_side
                )
                for child in node.patterns
                if self._contains_var(child, target_var)
            ]
            return min(values) if values else self._count_min(node, counted_var)

        if isinstance(node, PermutationPattern):
            values = [
                self._count_min_until_target(
                    child, counted_var, target_var, target_side
                )
                for child in node.patterns
                if self._contains_var(child, target_var)
            ]
            if target_side == "right":
                return sum(values) if values else self._count_min(node, counted_var)
            return min(values) if values else self._count_min(node, counted_var)

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._count_min_until_target(
                node.pattern, counted_var, target_var, target_side
            )

        return self._count_min(node, counted_var)

    def _tighten_whole_before_target(
        self, node, constraint: CountConstraint, prior_min: int
    ):
        counted_var = constraint.counted_var

        if isinstance(node, VariablePattern):
            return node, False

        if isinstance(node, QuantifiedPattern):
            inner_min = self._count_min(node.pattern, counted_var)
            changed = False
            new_at_least = node.atLeast
            new_at_most = node.atMost
            upper = constraint.upper
            if inner_min > 0:
                if upper is not None:
                    remaining = (
                        upper
                        if counted_var is not None and node.atLeast > 1
                        else upper - prior_min
                    )
                    max_repeats = int(remaining // inner_min)
                    if max_repeats >= node.atLeast and (
                        node.atMost is None or max_repeats < node.atMost
                    ):
                        new_at_most = max_repeats
                        changed = True

            new_pattern = node.pattern
            if not changed:
                inner, inner_changed = self._tighten_whole_before_target(
                    node.pattern, constraint, prior_min
                )
                if inner_changed:
                    new_pattern = inner
                    changed = True

            if changed:
                return (
                    self._make_quantified(
                        new_pattern, new_at_least, new_at_most, node.isGreedy
                    ),
                    True,
                )
            return node, False

        if isinstance(node, ConcatenationPattern):
            changed = False
            current_prior = prior_min
            new_patterns = []
            suffix_min = [0] * (len(node.patterns) + 1)
            for idx in range(len(node.patterns) - 1, -1, -1):
                suffix_min[idx] = suffix_min[idx + 1] + self._count_min(
                    node.patterns[idx], counted_var
                )

            for idx, child in enumerate(node.patterns):
                new_child, child_changed = self._tighten_whole_before_target(
                    child, constraint, current_prior + suffix_min[idx + 1]
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
                current_prior += self._count_min(new_child, constraint.counted_var)
            if changed:
                return ConcatenationPattern(new_patterns), True
            return node, False

        if isinstance(node, AlternationPattern):
            changed = False
            new_patterns = []
            for child in node.patterns:
                new_child, child_changed = self._tighten_whole_before_target(
                    child, constraint, prior_min
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
            if changed:
                return AlternationPattern(new_patterns), True
            return node, False

        if isinstance(node, PermutationPattern):
            changed = False
            new_patterns = []
            for child in node.patterns:
                new_child, child_changed = self._tighten_whole_before_target(
                    child, constraint, prior_min
                )
                new_patterns.append(new_child)
                changed = changed or child_changed
            if changed:
                return PermutationPattern(new_patterns), True
            return node, False

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            inner, changed = self._tighten_whole_before_target(
                node.pattern, constraint, prior_min
            )
            if not changed:
                return node, False
            if isinstance(node, ExcludedPattern):
                return ExcludedPattern(inner), True
            return AnchorPattern(inner, node.anchor_type), True

        return node, False

    def _has_counted_rows_after_target(
        self, node, target_var: str, counted_var: Optional[str]
    ) -> bool:
        return self._after_target_result(node, target_var, counted_var)[0]

    def _pattern_implies_count_constraint(
        self, node, target_var: str, constraint: CountConstraint
    ) -> bool:
        count_range = self._count_range_at_target(
            node, constraint.counted_var, target_var
        )
        if count_range is None:
            return False

        min_count, max_count = count_range
        if min_count < constraint.lower:
            return False
        if constraint.upper is not None and (
            max_count is None or max_count > constraint.upper
        ):
            return False
        return True

    def _pattern_contradicts_count_constraint(
        self, node, target_var: str, constraint: CountConstraint
    ) -> bool:
        count_range = self._count_range_at_target(
            node, constraint.counted_var, target_var
        )
        if count_range is None:
            return False

        min_count, max_count = count_range
        if constraint.upper is not None and min_count > constraint.upper:
            return True
        if max_count is not None and max_count < constraint.lower:
            return True
        return False

    def _count_range_at_target(
        self, pattern, counted_var: Optional[str], target_var: str
    ):
        return self._count_range_at_target_node(pattern, counted_var, target_var)

    def _count_range_at_target_node(
        self, node, counted_var: Optional[str], target_var: str
    ):
        if isinstance(node, VariablePattern):
            if node.name != target_var:
                return None
            count = 1 if self._var_counts(node.name, counted_var) else 0
            return count, count

        if isinstance(node, QuantifiedPattern):
            inner = self._count_range_at_target_node(
                node.pattern, counted_var, target_var
            )
            if inner is None:
                return None
            _, inner_max = self._count_range_in_pattern(node.pattern, counted_var)
            repeat_prefix_max = self._repeat_prefix_max(node.atMost, inner_max)
            return inner[0], self._add_optional_counts(repeat_prefix_max, inner[1])

        if isinstance(node, ConcatenationPattern):
            prefix_min = 0
            prefix_max: Optional[int] = 0
            ranges = []
            for child in node.patterns:
                inner = self._count_range_at_target_node(child, counted_var, target_var)
                if inner is not None:
                    inner_min, inner_max = inner
                    ranges.append(
                        (
                            prefix_min + inner_min,
                            self._add_optional_counts(prefix_max, inner_max),
                        )
                    )

                child_min, child_max = self._count_range_in_pattern(child, counted_var)
                prefix_min += child_min
                prefix_max = self._add_optional_counts(prefix_max, child_max)
            return self._merge_count_ranges(ranges)

        if isinstance(node, AlternationPattern):
            ranges = [
                result
                for result in (
                    self._count_range_at_target_node(child, counted_var, target_var)
                    for child in node.patterns
                )
                if result is not None
            ]
            return self._merge_count_ranges(ranges)

        if isinstance(node, PermutationPattern):
            return None

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._count_range_at_target_node(
                node.pattern, counted_var, target_var
            )

        return None

    def _count_range_in_pattern(
        self, node, counted_var: Optional[str]
    ) -> Tuple[int, Optional[int]]:
        cache = getattr(self, "_count_range_cache", None)
        key = (id(node), counted_var)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None and cached[0] is node:
                return cached[1]

        result = self._compute_count_range_in_pattern(node, counted_var)
        if cache is not None:
            # Retaining the node also prevents an id from being reused during
            # this plan and accidentally hitting an unrelated cached result.
            cache[key] = (node, result)
        return result

    def _compute_count_range_in_pattern(
        self, node, counted_var: Optional[str]
    ) -> Tuple[int, Optional[int]]:
        if counted_var is None:
            return node.get_min_row_num(), node.get_max_row_num()

        if isinstance(node, VariablePattern):
            count = int(self._var_counts(node.name, counted_var))
            return count, count

        if isinstance(node, QuantifiedPattern):
            inner_min, inner_max = self._count_range_in_pattern(
                node.pattern, counted_var
            )
            minimum = node.atLeast * inner_min
            if node.atMost is None:
                maximum = 0 if inner_max == 0 else None
            elif inner_max is None:
                maximum = None
            else:
                maximum = node.atMost * inner_max
            return minimum, maximum

        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            minimum = 0
            maximum: Optional[int] = 0
            for child in node.patterns:
                child_min, child_max = self._count_range_in_pattern(child, counted_var)
                minimum += child_min
                maximum = self._add_optional_counts(maximum, child_max)
            return minimum, maximum

        if isinstance(node, AlternationPattern):
            if not node.patterns:
                return 0, 0
            ranges = [
                self._count_range_in_pattern(child, counted_var)
                for child in node.patterns
            ]
            return min(item[0] for item in ranges), (
                None
                if any(item[1] is None for item in ranges)
                else max(item[1] for item in ranges)
            )

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._count_range_in_pattern(node.pattern, counted_var)
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

    def _tighten_single_adjustable_lower_before_target(
        self, node, target_var: str, counted_var: Optional[str], lower: int
    ):
        branch_rewritten, branch_changed = (
            self._tighten_branch_adjustable_lower_before_target(
                node, target_var, counted_var, lower
            )
        )
        if branch_changed:
            return branch_rewritten, True

        analysis = self._analyze_lower_before_target_count(
            node, target_var, counted_var
        )
        if analysis is None:
            return node, False

        _, adjustable = analysis
        if not adjustable:
            return node, False

        current_range = self._count_range_at_target(node, counted_var, target_var)
        if current_range is None:
            return node, False

        quantifier = adjustable[0]
        remaining = lower - current_range[0]
        if remaining <= 0 or quantifier.unit_count <= 0:
            return node, False

        repetitions = math.ceil(remaining / quantifier.unit_count)
        new_at_least = quantifier.node.atLeast + repetitions
        if quantifier.node.atMost is not None and new_at_least > quantifier.node.atMost:
            return node, False
        if new_at_least == quantifier.node.atLeast:
            return node, False

        replacement = self._make_quantified(
            quantifier.node.pattern,
            new_at_least,
            quantifier.node.atMost,
            quantifier.node.isGreedy,
        )
        return self._replace_node_by_identity(node, quantifier.node, replacement), True

    def _tighten_branch_adjustable_lower_before_target(
        self, node, target_var: str, counted_var: Optional[str], lower: int
    ):
        path_analysis = self._analyze_lower_target_paths(node, target_var, counted_var)
        if path_analysis is None:
            return node, False

        paths = path_analysis
        if not paths:
            return node, False

        first_adjustable = paths[0].adjustable
        if len(first_adjustable) != 1:
            return node, False

        quantifier = first_adjustable[0]
        if quantifier.unit_count <= 0:
            return node, False

        for path in paths[1:]:
            if len(path.adjustable) != 1:
                return node, False
            other = path.adjustable[0]
            if (
                other.node is not quantifier.node
                or other.unit_count != quantifier.unit_count
            ):
                return node, False

        required_repetitions = [
            max(0, math.ceil((lower - path.min_count) / quantifier.unit_count))
            for path in paths
        ]
        repetitions = min(required_repetitions)
        if repetitions <= 0:
            return node, False

        new_at_least = quantifier.node.atLeast + repetitions
        if quantifier.node.atMost is not None and new_at_least > quantifier.node.atMost:
            return node, False

        replacement = self._make_quantified(
            quantifier.node.pattern,
            new_at_least,
            quantifier.node.atMost,
            quantifier.node.isGreedy,
        )
        return self._replace_node_by_identity(node, quantifier.node, replacement), True

    def _analyze_lower_target_paths(
        self, node, target_var: str, counted_var: Optional[str]
    ):
        contains_target, paths, uncertain = self._collect_lower_target_paths(
            node, target_var, counted_var
        )
        if not contains_target or uncertain:
            return None
        return paths

    def _collect_lower_target_paths(
        self, node, target_var: str, counted_var: Optional[str]
    ):
        if isinstance(node, VariablePattern):
            if node.name == target_var:
                count = 1 if self._var_counts(node.name, counted_var) else 0
                return True, [TargetPath(count, [])], False
            return False, [], False

        if isinstance(node, QuantifiedPattern):
            if self._contains_var(node.pattern, target_var):
                return True, [], True
            return False, [], False

        if isinstance(node, ConcatenationPattern):
            prefix_count = 0
            prefix_adjustable = []
            for child in node.patterns:
                contains_target, child_paths, uncertain = (
                    self._collect_lower_target_paths(child, target_var, counted_var)
                )
                if uncertain:
                    return contains_target, [], True
                if contains_target:
                    paths = [
                        TargetPath(
                            prefix_count + child_path.min_count,
                            prefix_adjustable + child_path.adjustable,
                        )
                        for child_path in child_paths
                    ]
                    return True, paths, False

                fixed_count, adjustable, uncertain = (
                    self._analyze_whole_count_for_exact(child, counted_var)
                )
                if uncertain:
                    return False, [], True
                prefix_count += self._count_min(child, counted_var)
                prefix_adjustable.extend(adjustable)
            return False, [], False

        if isinstance(node, AlternationPattern):
            paths = []
            saw_target_branch = False
            saw_non_target_branch = False
            for child in node.patterns:
                contains_target, child_paths, uncertain = (
                    self._collect_lower_target_paths(child, target_var, counted_var)
                )
                if uncertain:
                    return contains_target, [], True
                if contains_target:
                    saw_target_branch = True
                    paths.extend(child_paths)
                else:
                    saw_non_target_branch = True

            if not saw_target_branch:
                return False, [], False
            if saw_non_target_branch:
                return True, [], True
            return True, paths, False

        if isinstance(node, PermutationPattern):
            if self._contains_var(node, target_var):
                return True, [], True
            return False, [], False

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._collect_lower_target_paths(
                node.pattern, target_var, counted_var
            )

        return False, [], False

    def _analyze_lower_before_target_count(
        self, node, target_var: str, counted_var: Optional[str]
    ):
        contains_target, fixed_count, adjustable, uncertain = (
            self._analyze_lower_until_target_count(node, target_var, counted_var)
        )
        if not contains_target or uncertain:
            return None
        return fixed_count, adjustable

    def _analyze_lower_until_target_count(
        self, node, target_var: str, counted_var: Optional[str]
    ):
        return self._analyze_until_target_count(
            node, target_var, counted_var, strict_target_paths=True
        )

    def _make_quantified(
        self, pattern, at_least: int, at_most: Optional[int], is_greedy: bool
    ):
        if at_least == 0 and at_most == 0:
            return EmptyPattern()
        if at_least == 1 and at_most == 1 and is_greedy:
            return pattern
        return QuantifiedPattern(pattern, at_least, at_most, is_greedy)

    def _restrict_exact_count_before_target(
        self, node, target_var: str, counted_var: Optional[str], exact_count: int
    ):
        restricted = self._restrict_exact_count_until_target(
            node, target_var, counted_var, exact_count
        )
        if restricted is not None:
            return restricted, True
        return self._tighten_exact_single_adjustable_before_target(
            node, target_var, counted_var, exact_count
        )

    def _restrict_exact_count_until_target(
        self, node, target_var: str, counted_var: Optional[str], exact_count: int
    ):
        if isinstance(node, ConcatenationPattern):
            for idx, child in enumerate(node.patterns):
                if not self._contains_var(child, target_var):
                    continue
                if not isinstance(child, VariablePattern) or child.name != target_var:
                    return None

                target_count = 1 if self._var_counts(child.name, counted_var) else 0
                prefix_count = exact_count - target_count
                if prefix_count < 0:
                    return None

                prefix = self._concat_or_empty(node.patterns[:idx])
                restricted_prefix = self._restrict_exact_count(
                    prefix, counted_var, prefix_count
                )
                if restricted_prefix is None:
                    return None

                new_patterns = []
                if not isinstance(restricted_prefix, EmptyPattern):
                    if isinstance(restricted_prefix, ConcatenationPattern):
                        new_patterns.extend(restricted_prefix.patterns)
                    else:
                        new_patterns.append(restricted_prefix)
                new_patterns.append(child)
                new_patterns.extend(node.patterns[idx + 1 :])
                return ConcatenationPattern(new_patterns)
            return None

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            inner = self._restrict_exact_count_until_target(
                node.pattern, target_var, counted_var, exact_count
            )
            if inner is None:
                return None
            if isinstance(node, ExcludedPattern):
                return ExcludedPattern(inner)
            return AnchorPattern(inner, node.anchor_type)

        if isinstance(node, VariablePattern) and node.name == target_var:
            target_count = 1 if self._var_counts(node.name, counted_var) else 0
            return node if target_count == exact_count else None

        return None

    def _restrict_exact_count(self, node, counted_var: Optional[str], exact_count: int):
        if exact_count < 0:
            return None

        min_count = self._count_min(node, counted_var)
        max_count = self._count_max(node, counted_var)
        if exact_count < min_count:
            return None
        if max_count is not None and exact_count > max_count:
            return None
        if min_count == max_count == exact_count:
            return node

        if isinstance(node, EmptyPattern):
            return node if exact_count == 0 else None

        if isinstance(node, VariablePattern):
            count = 1 if self._var_counts(node.name, counted_var) else 0
            return node if count == exact_count else None

        if isinstance(node, QuantifiedPattern):
            inner_min = self._count_min(node.pattern, counted_var)
            inner_max = self._count_max(node.pattern, counted_var)
            if inner_min == inner_max:
                unit = inner_min
                if unit == 0:
                    return node if exact_count == 0 else None
                if exact_count % unit != 0:
                    return None
                repetitions = exact_count // unit
                if repetitions < node.atLeast:
                    return None
                if node.atMost is not None and repetitions > node.atMost:
                    return None
                return self._make_quantified(
                    node.pattern, repetitions, repetitions, node.isGreedy
                )
            return None

        if isinstance(node, AlternationPattern):
            branches = []
            seen = set()
            for child in node.patterns:
                restricted = self._restrict_exact_count(child, counted_var, exact_count)
                if restricted is None:
                    continue
                key = restricted.to_string()
                if key in seen:
                    continue
                seen.add(key)
                branches.append(restricted)
            if not branches:
                return None
            if len(branches) == 1:
                return branches[0]
            return AlternationPattern(branches)

        if isinstance(node, ConcatenationPattern):
            return self._restrict_exact_count_concatenation(
                node.patterns, counted_var, exact_count
            )

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            inner = self._restrict_exact_count(node.pattern, counted_var, exact_count)
            if inner is None:
                return None
            if isinstance(node, ExcludedPattern):
                return ExcludedPattern(inner)
            return AnchorPattern(inner, node.anchor_type)

        return None

    def _restrict_exact_count_concatenation(
        self, patterns: List, counted_var: Optional[str], exact_count: int
    ):
        states = [(0, [])]
        for child in patterns:
            next_states = []
            child_min = self._count_min(child, counted_var)
            child_max = self._count_max(child, counted_var)
            max_for_child = (
                exact_count if child_max is None else min(exact_count, child_max)
            )
            for used, built in states:
                for child_count in range(child_min, max_for_child + 1):
                    if used + child_count > exact_count:
                        break
                    restricted = self._restrict_exact_count(
                        child, counted_var, child_count
                    )
                    if restricted is None:
                        continue
                    next_states.append((used + child_count, built + [restricted]))
            states = next_states
            if not states:
                return None

        branches = [
            ConcatenationPattern(built).simplify()
            for used, built in states
            if used == exact_count
        ]
        if not branches:
            return None

        unique_branches = []
        seen = set()
        for branch in branches:
            key = branch.to_string()
            if key in seen:
                continue
            seen.add(key)
            unique_branches.append(branch)

        if len(unique_branches) == 1:
            return unique_branches[0]
        return AlternationPattern(unique_branches)

    def _concat_or_empty(self, patterns: List):
        if not patterns:
            return EmptyPattern()
        if len(patterns) == 1:
            return patterns[0]
        return ConcatenationPattern(patterns)

    def _tighten_exact_single_adjustable_before_target(
        self, node, target_var: str, counted_var: Optional[str], exact_count: int
    ):
        analysis = self._analyze_before_target_count(node, target_var, counted_var)
        if analysis is None:
            return node, False

        fixed_count, adjustable = analysis
        if len(adjustable) != 1:
            return node, False

        quantifier = adjustable[0]
        remaining = exact_count - fixed_count
        if (
            remaining < 0
            or quantifier.unit_count <= 0
            or remaining % quantifier.unit_count != 0
        ):
            return node, False

        repetitions = remaining // quantifier.unit_count
        if repetitions < quantifier.node.atLeast:
            return node, False
        if quantifier.node.atMost is not None and repetitions > quantifier.node.atMost:
            return node, False

        replacement = self._make_quantified(
            quantifier.node.pattern, repetitions, repetitions, quantifier.node.isGreedy
        )
        return self._replace_node_by_identity(node, quantifier.node, replacement), True

    def _analyze_before_target_count(
        self, node, target_var: str, counted_var: Optional[str]
    ):
        contains_target, fixed_count, adjustable, uncertain = (
            self._analyze_until_target_count(node, target_var, counted_var)
        )
        if not contains_target or uncertain:
            return None
        return fixed_count, adjustable

    def _analyze_until_target_count(
        self,
        node,
        target_var: str,
        counted_var: Optional[str],
        strict_target_paths: bool = False,
    ):
        if isinstance(node, VariablePattern):
            if node.name == target_var:
                count = 1 if self._var_counts(node.name, counted_var) else 0
                return True, count, [], False
            return False, self._count_min(node, counted_var), [], False

        if isinstance(node, QuantifiedPattern):
            if self._contains_var(node.pattern, target_var):
                if strict_target_paths:
                    return True, 0, [], True
                return self._analyze_until_target_count(
                    node.pattern, target_var, counted_var, strict_target_paths
                )
            fixed_count, adjustable, uncertain = self._analyze_whole_count_for_exact(
                node, counted_var
            )
            return False, fixed_count, adjustable, uncertain

        if isinstance(node, ConcatenationPattern):
            fixed_count = 0
            adjustable = []
            for child in node.patterns:
                contains_target, child_fixed, child_adjustable, uncertain = (
                    self._analyze_until_target_count(
                        child, target_var, counted_var, strict_target_paths
                    )
                )
                if uncertain:
                    return contains_target, 0, [], True
                fixed_count += child_fixed
                adjustable.extend(child_adjustable)
                if contains_target:
                    return True, fixed_count, adjustable, False
            return False, fixed_count, adjustable, False

        if isinstance(node, AlternationPattern):
            branch_results = [
                self._analyze_until_target_count(
                    child, target_var, counted_var, strict_target_paths
                )
                for child in node.patterns
                if self._contains_var(child, target_var)
            ]
            if not branch_results:
                fixed_count, adjustable, uncertain = (
                    self._analyze_whole_count_for_exact(node, counted_var)
                )
                return False, fixed_count, adjustable, uncertain
            if strict_target_paths:
                return True, 0, [], True
            if len(branch_results) != 1:
                return True, 0, [], True
            return branch_results[0]

        if isinstance(node, PermutationPattern):
            return self._contains_var(node, target_var), 0, [], True

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._analyze_until_target_count(
                node.pattern, target_var, counted_var, strict_target_paths
            )

        return False, 0, [], False

    def _analyze_whole_count_for_exact(self, node, counted_var: Optional[str]):
        if isinstance(node, VariablePattern):
            return self._count_min(node, counted_var), [], False

        if isinstance(node, QuantifiedPattern):
            inner_min = self._count_min(node.pattern, counted_var)
            inner_max = self._count_max(node.pattern, counted_var)
            if inner_min == 0 and inner_max == 0:
                return 0, [], False
            if inner_max is None or inner_min != inner_max:
                return 0, [], True
            if node.atMost == node.atLeast:
                return node.atLeast * inner_min, [], False
            return 0, [AdjustableQuantifier(node, inner_min)], False

        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            fixed_count = 0
            adjustable = []
            for child in node.patterns:
                child_fixed, child_adjustable, uncertain = (
                    self._analyze_whole_count_for_exact(child, counted_var)
                )
                if uncertain:
                    return 0, [], True
                fixed_count += child_fixed
                adjustable.extend(child_adjustable)
            return fixed_count, adjustable, False

        if isinstance(node, AlternationPattern):
            if not node.patterns:
                return 0, [], False
            branch_results = [
                self._analyze_whole_count_for_exact(child, counted_var)
                for child in node.patterns
            ]
            if any(result[2] or result[1] for result in branch_results):
                return 0, [], True
            fixed_values = {result[0] for result in branch_results}
            if len(fixed_values) != 1:
                return 0, [], True
            return branch_results[0][0], [], False

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._analyze_whole_count_for_exact(node.pattern, counted_var)

        return 0, [], False

    def _replace_node_by_identity(self, node, target_node, replacement):
        if node is target_node:
            return replacement
        if isinstance(node, QuantifiedPattern):
            inner = self._replace_node_by_identity(
                node.pattern, target_node, replacement
            )
            if inner is not node.pattern:
                return QuantifiedPattern(
                    inner, node.atLeast, node.atMost, node.isGreedy
                )
            return node
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            patterns = [
                self._replace_node_by_identity(child, target_node, replacement)
                for child in node.patterns
            ]
            if any(
                new_child is not old_child
                for new_child, old_child in zip(patterns, node.patterns)
            ):
                return type(node)(patterns)
            return node
        if isinstance(node, ExcludedPattern):
            inner = self._replace_node_by_identity(
                node.pattern, target_node, replacement
            )
            if inner is not node.pattern:
                return ExcludedPattern(inner)
            return node
        if isinstance(node, AnchorPattern):
            inner = self._replace_node_by_identity(
                node.pattern, target_node, replacement
            )
            if inner is not node.pattern:
                return AnchorPattern(inner, node.anchor_type)
            return node
        return node

    def _after_target_result(
        self, node, target_var: str, counted_var: Optional[str]
    ) -> Tuple[bool, bool, bool]:
        if isinstance(node, VariablePattern):
            is_target = node.name == target_var
            is_counted = self._var_counts(node.name, counted_var)
            return False, is_target, is_counted

        if isinstance(node, QuantifiedPattern):
            after, has_target, has_counted = self._after_target_result(
                node.pattern, target_var, counted_var
            )
            if has_target and has_counted and (node.atMost is None or node.atMost > 1):
                after = True
            return after, has_target, has_counted

        if isinstance(node, ConcatenationPattern):
            after = False
            seen_target = False
            has_target = False
            has_counted = False
            for child in node.patterns:
                child_after, child_has_target, child_has_counted = (
                    self._after_target_result(child, target_var, counted_var)
                )
                if seen_target and child_has_counted:
                    after = True
                if child_after:
                    after = True
                seen_target = seen_target or child_has_target
                has_target = has_target or child_has_target
                has_counted = has_counted or child_has_counted
            return after, has_target, has_counted

        if isinstance(node, AlternationPattern):
            after = False
            has_target = False
            has_counted = False
            for child in node.patterns:
                child_after, child_has_target, child_has_counted = (
                    self._after_target_result(child, target_var, counted_var)
                )
                after = after or child_after
                has_target = has_target or child_has_target
                has_counted = has_counted or child_has_counted
            return after, has_target, has_counted

        if isinstance(node, PermutationPattern):
            has_target = any(
                self._contains_var(child, target_var) for child in node.patterns
            )
            has_counted = any(
                self._count_min(child, counted_var) > 0
                or self._count_max_possible(child, counted_var)
                for child in node.patterns
            )
            return has_target and has_counted, has_target, has_counted

        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._after_target_result(node.pattern, target_var, counted_var)

        return False, False, False

    def _flatten_and(self, expr: Expr) -> List[Expr]:
        if isinstance(expr, SemanticsExpr):
            return self._flatten_and(expr.expr)
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]

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

    def _contains_var(self, node, var_name: str) -> bool:
        cache = getattr(self, "_contains_var_cache", None)
        key = (id(node), var_name)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None and cached[0] is node:
                return cached[1]

        result = self._compute_contains_var(node, var_name)
        if cache is not None:
            cache[key] = (node, result)
        return result

    def _compute_contains_var(self, node, var_name: str) -> bool:
        if isinstance(node, VariablePattern):
            return node.name == var_name
        if isinstance(node, QuantifiedPattern):
            return self._contains_var(node.pattern, var_name)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._contains_var(node.pattern, var_name)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(self._contains_var(child, var_name) for child in node.patterns)
        return False

    def _count_min(self, node, counted_var: Optional[str]) -> int:
        return self._count_range_in_pattern(node, counted_var)[0]

    def _count_max(self, node, counted_var: Optional[str]) -> Optional[int]:
        return self._count_range_in_pattern(node, counted_var)[1]

    def _count_max_possible(self, node, counted_var: CountedVar) -> bool:
        if counted_var is None:
            return node.get_max_row_num() != 0
        return self._counted_var_occurs(node, counted_var)

    def _var_counts(self, var_name: str, counted_var: CountedVar) -> bool:
        if counted_var is None:
            return True
        if isinstance(counted_var, frozenset):
            return var_name.upper() in counted_var
        return var_name.upper() == counted_var.upper()

    def _counted_var_occurs(self, node, counted_var: CountedVar) -> bool:
        if counted_var is None:
            return node.get_max_row_num() != 0
        if isinstance(counted_var, frozenset):
            return any(self._contains_var(node, member) for member in counted_var)
        return self._contains_var(node, counted_var)

    def _resolve_counted_var(self, counted_var):
        if counted_var in (None, False):
            return counted_var
        return getattr(self, "_subset_members", {}).get(
            str(counted_var).upper(), counted_var
        )

    def _unwrap_semantics(self, expr: Expr) -> Expr:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        return expr


count_range_pushdown_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ B+ C)
    DEFINE B AS B.VAL > 10, C AS VAL > 10 AND COUNT(A.*) < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A{1,4} B+ C)
    DEFINE B AS B.VAL > 10, C AS VAL > 10
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B (A+ | B+) C)
    DEFINE A AS OPEN > 100, B AS VAL > 10, C AS VAL > 10 AND COUNT(A.*) <= 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN((A B A{1,4} | A B{2,}) C)
    DEFINE A AS OPEN > 100, B AS VAL > 10, C AS VAL > 10
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ D (A+ | B+) A+ C A+)
    DEFINE A AS OPEN > 100, D AS COL = 10, B AS VAL > 10, C AS VAL > 10 AND COUNT(A.*) < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN((A{1,2} D A{2,4} | A{1,3} D B+ A{1,3}) C A+)
    DEFINE A AS OPEN > 100, D AS COL = 10, B AS VAL > 10, C AS VAL > 10 AND COUNT(A.*) < 5
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ B{1,2} C)
    DEFINE A AS OPEN > 5, C AS COUNT(*) <= 7 AND COUNT(*) >= 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A{3,5} B{1,2} C)
    DEFINE A AS OPEN > 5, C AS COUNT(*) <= 7
)""",
    ),
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A+ B{6,} C)
    DEFINE C AS COUNT(A.*) = 2
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A{2} B{6,} C)
    DEFINE C AS 1 = 1
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        CountRangePushdownRule,
        count_range_pushdown_test_cases,
        "count range pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
