"""Shared analysis of MATCH_RECOGNIZE physical navigation.

This module describes PREV/NEXT calls and their possible pattern positions.
It deliberately does not decide whether a rewrite should reject a call, add a
boundary tail, or enlarge a top-K window; those policies belong to the rules
that change the input row set.
"""

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple

from expression import ColumnRef, Expr, FunctionCall, Literal, SemanticsExpr
from queryanalysis import RPRAbstractSyntaxTree

from .pattern_analysis import enumerate_pattern_sequences
from .utils import (
    _extract_navigation_offset,
    _iter_measure_expressions,
    _walk_expression,
)


@dataclass(frozen=True)
class PhysicalNavigationCall:
    """One resolved PREV/NEXT call in DEFINE, MEASURES, or an expression."""

    direction: str
    base_variable: Optional[str]
    physical_offset: Optional[int]
    selector: str = "RUNNING_LAST"
    logical_offset: Optional[int] = 0
    is_universal: bool = False
    context: str = "EXPRESSION"
    define_variable: Optional[str] = None
    operand_qualifiers: FrozenSet[str] = frozenset()


def analyze_navigation_operand(
    expr: Expr,
    default_var: Optional[str],
) -> Tuple[Optional[str], str, Optional[int], bool]:
    """Resolve the pattern base and logical selector of a PREV/NEXT operand."""
    root = expr
    while isinstance(root, SemanticsExpr):
        root = root.expr

    selector = "RUNNING_LAST"
    logical_offset: Optional[int] = 0
    if isinstance(root, FunctionCall) and root.name.upper() in ("FIRST", "LAST"):
        selector = root.name.upper()
        if len(root.args) >= 2:
            offset_expr = root.args[1]
            while isinstance(offset_expr, SemanticsExpr):
                offset_expr = offset_expr.expr
            if isinstance(offset_expr, Literal) and isinstance(
                offset_expr.value, (int, float)
            ):
                logical_offset = max(0, int(offset_expr.value))
            else:
                logical_offset = None

    qualifiers = {}
    saw_unqualified = False
    classifier_args = set()
    for node in _walk_expression(expr):
        if isinstance(node, FunctionCall) and node.name.upper() == "CLASSIFIER":
            if node.args and isinstance(node.args[0], ColumnRef):
                classifier_arg = node.args[0]
                classifier_args.add(id(classifier_arg))
                name = classifier_arg.qualifier or classifier_arg.name
                qualifiers.setdefault(name.upper(), name)
            else:
                saw_unqualified = True

    for node in _walk_expression(expr):
        if isinstance(node, ColumnRef) and id(node) not in classifier_args:
            if node.qualifier is None:
                saw_unqualified = True
            else:
                qualifiers.setdefault(node.qualifier.upper(), node.qualifier)

    if len(qualifiers) == 1 and not saw_unqualified:
        return (
            next(iter(qualifiers.values())),
            selector,
            logical_offset,
            False,
        )
    if not qualifiers and saw_unqualified:
        return default_var, selector, logical_offset, True
    if not qualifiers:
        # Keep the established fallback for invalid/no-column operands.  The
        # consuming safety policy can still reject an unresolved default.
        return default_var, selector, logical_offset, False
    return None, selector, logical_offset, False


def collect_expression_navigation(
    expr: Optional[Expr],
    default_var: Optional[str],
    direction: Optional[str] = None,
    *,
    context: str = "EXPRESSION",
    define_variable: Optional[str] = None,
) -> List[PhysicalNavigationCall]:
    """Collect physical-navigation calls from one expression."""
    target = direction.upper() if direction is not None else None
    result = []
    for node in _walk_expression(expr):
        if not isinstance(node, FunctionCall):
            continue
        node_direction = node.name.upper()
        if node_direction not in ("PREV", "NEXT"):
            continue
        if target is not None and node_direction != target:
            continue

        if node.args:
            base_var, selector, logical_offset, is_universal = (
                analyze_navigation_operand(node.args[0], default_var)
            )
            operand_qualifiers = frozenset(
                child.qualifier
                for child in _walk_expression(node.args[0])
                if isinstance(child, ColumnRef) and child.qualifier is not None
            )
        else:
            base_var = None
            selector = "RUNNING_LAST"
            logical_offset = 0
            is_universal = False
            operand_qualifiers = frozenset()

        result.append(
            PhysicalNavigationCall(
                direction=node_direction,
                base_variable=base_var,
                physical_offset=_extract_navigation_offset(node),
                selector=selector,
                logical_offset=logical_offset,
                is_universal=is_universal,
                context=context,
                define_variable=define_variable,
                operand_qualifiers=operand_qualifiers,
            )
        )
    return result


def collect_physical_navigation(
    ast: RPRAbstractSyntaxTree,
) -> List[PhysicalNavigationCall]:
    """Collect every DEFINE and MEASURES physical-navigation call."""
    result = []
    for define_var, condition in ast.define.definitions.items():
        if condition is None:
            continue
        result.extend(
            collect_expression_navigation(
                condition.expression,
                define_var,
                context="DEFINE",
                define_variable=define_var,
            )
        )
    for expression in _iter_measure_expressions(ast):
        result.extend(
            collect_expression_navigation(
                expression,
                None,
                context="MEASURES",
            )
        )
    return result


def max_navigation_offset(
    calls: Iterable[PhysicalNavigationCall],
    direction: str,
) -> Optional[int]:
    """Return the largest fixed offset in one direction, or None if unknown."""
    target = direction.upper()
    best = 0
    for call in calls:
        if call.direction != target:
            continue
        if call.physical_offset is None:
            return None
        best = max(best, call.physical_offset)
    return best


def has_physical_navigation(expr: Optional[Expr]) -> bool:
    """Whether an expression contains PREV or NEXT, including offset zero."""
    return any(
        isinstance(node, FunctionCall) and node.name.upper() in ("PREV", "NEXT")
        for node in _walk_expression(expr)
    )


def expand_navigation_base_vars(
    ast: RPRAbstractSyntaxTree,
    base_var: str,
) -> Set[str]:
    """Expand a SUBSET navigation base to its primary pattern variables."""
    normalized = str(base_var).upper()
    for subset_name, members in (ast.subsets or {}).items():
        if str(subset_name).upper() == normalized:
            return {str(member).upper() for member in members}
    return {base_var}


class PatternNavigationReach:
    """Position proofs shared by row-removing rewrite policies."""

    def __init__(self, ast: RPRAbstractSyntaxTree):
        self.ast = ast

    def expression_stays_within_match(
        self,
        expression: Expr,
        default_var: str,
    ) -> bool:
        """Whether every call is bounded by guaranteed matched rows."""
        primary_vars = {
            str(var_name).upper(): var_name
            for var_name in self.ast.pattern.get_variables()
        }
        subsets = {
            str(subset_name).upper(): members
            for subset_name, members in (self.ast.subsets or {}).items()
        }

        for call in collect_expression_navigation(expression, default_var):
            offset = call.physical_offset
            if offset is None:
                return False
            if offset == 0:
                continue
            if call.base_variable is None:
                return False

            normalized_base = str(call.base_variable).upper()
            if normalized_base in subsets:
                resolved_vars = [
                    primary_vars.get(str(member).upper())
                    for member in subsets[normalized_base]
                ]
            else:
                resolved_vars = [primary_vars.get(normalized_base)]
            if not resolved_vars or any(var is None for var in resolved_vars):
                return False

            for resolved_var in resolved_vars:
                if call.direction == "PREV":
                    inside_rows = self.ast.pattern.get_left_min_row_num(
                        resolved_var,
                    )
                else:
                    inside_rows = self.ast.pattern.get_right_min_row_num(
                        resolved_var,
                    )
                if inside_rows is None or offset > inside_rows:
                    return False
        return True

    def all_navigation_stays_within_match(self) -> bool:
        """Apply the match-edge proof to all DEFINE and MEASURES calls."""
        for define_var, condition in self.ast.define.definitions.items():
            if condition is None:
                continue
            if not self.expression_stays_within_match(
                condition.expression,
                define_var,
            ):
                return False

        pattern_vars = self.ast.pattern.get_variables()
        for expression in _iter_measure_expressions(self.ast):
            for default_var in pattern_vars:
                if not self.expression_stays_within_match(
                    expression,
                    default_var,
                ):
                    return False
        return True

    def max_define_reach_from_variable(
        self,
        variable: str,
        side: str,
    ) -> Optional[int]:
        """Return DEFINE navigation reach past a pattern-variable position."""
        target_direction = "PREV" if side == "left" else "NEXT"
        sequences = enumerate_pattern_sequences(self.ast.pattern)
        if sequences is None:
            return None

        normalized_variable = variable.upper()
        best = 0
        for define_var, condition in self.ast.define.definitions.items():
            if condition is None:
                continue
            calls = collect_expression_navigation(
                condition.expression,
                define_var,
                target_direction,
                context="DEFINE",
                define_variable=define_var,
            )
            for call in calls:
                if (
                    call.base_variable is None
                    or call.physical_offset is None
                    or call.logical_offset is None
                ):
                    return None
                if call.physical_offset == 0:
                    continue

                if call.is_universal:
                    base_vars = {
                        value.upper() for value in self.ast.pattern.get_variables()
                    }
                else:
                    base_vars = {
                        value.upper()
                        for value in expand_navigation_base_vars(
                            self.ast,
                            call.base_variable,
                        )
                    }
                normalized_define_var = define_var.upper()

                for sequence in sequences:
                    normalized_sequence = [value.upper() for value in sequence]
                    variable_positions = [
                        index
                        for index, value in enumerate(normalized_sequence)
                        if value == normalized_variable
                    ]
                    define_positions = [
                        index
                        for index, value in enumerate(normalized_sequence)
                        if value == normalized_define_var
                    ]
                    if not variable_positions or not define_positions:
                        continue

                    for define_position in define_positions:
                        available_base_positions = [
                            index
                            for index, value in enumerate(normalized_sequence)
                            if index <= define_position and value in base_vars
                        ]
                        if call.selector == "FIRST":
                            selected_index = call.logical_offset
                        else:
                            selected_index = (
                                len(available_base_positions) - 1 - call.logical_offset
                            )
                        if not 0 <= selected_index < len(available_base_positions):
                            continue
                        base_position = available_base_positions[selected_index]

                        for variable_position in variable_positions:
                            target_position = (
                                base_position + call.physical_offset
                                if side == "right"
                                else base_position - call.physical_offset
                            )
                            rows = (
                                max(0, target_position - variable_position)
                                if side == "right"
                                else max(0, variable_position - target_position)
                            )
                            best = max(best, rows)
        return best
