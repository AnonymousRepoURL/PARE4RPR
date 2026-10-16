"""Expression-bearing clause models and DEFINE condition analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import re

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
    SemanticsExpr,
)
from expression_simplifier import (
    SimplifyContext,
    _compare_literal_values,
    _define_expr_to_string,
    _reverse_comparison_op,
    _simplify_comparison_expr,
    _simplify_logical_expr,
)

# Note that:
# (1) the MATCH_RECOGNIZE clause supports only five aggregation functions:
# COUNT, SUM, AVG, MIN, and MAX. Distinct aggregates are not supported at this time.
# (2) scalar functions (including UDF) is allowed to appear MEASURES and DEFINE subclauses

DEPENDENT_FUNCS = {
    "PREV",
    "NEXT",
    "FIRST",
    "LAST",  # navigational functions
    "MATCH_NUMBER",
    "CLASSIFIER",  # pattern matching functions
    "AVG",
    "SUM",
    "COUNT",
    "MAX",
    "MIN",  # aggregation functions
}


def collect_define_dependency_reasons(var_name: str, expression: Expr) -> Set[str]:
    """Return the reasons why a DEFINE expression needs more than its current row.

    An ordinary scalar function, including an unknown UDF, does not introduce a
    dependency by itself.  Its arguments are still visited: a UDF over columns
    of ``var_name`` is row-local, while a UDF receiving another pattern
    variable is dependent.
    """
    reasons = set()

    def visit(node):
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
        elif isinstance(node, CaseExpr):
            for condition, result in node.cases:
                visit(condition)
                visit(result)
            visit(node.else_expr)
        elif isinstance(node, FunctionCall):
            if node.name.upper() in DEPENDENT_FUNCS:
                reasons.add(f"Function '{node.name}'")
            # Scalar built-ins and UDFs are row-local unless their arguments
            # contain a cross-variable or otherwise dependent expression.
            for argument in node.args:
                visit(argument)
        elif isinstance(node, InExpr):
            visit(node.expr)
            for value in node.values:
                visit(value)
        elif isinstance(node, (IsNullExpr, NotExpr)):
            visit(node.expr)
        elif isinstance(node, ColumnRef):
            if node.qualifier and node.qualifier.upper() != var_name.upper():
                reasons.add(f"Cross-variable ref: {node.qualifier}.{node.name}")

    visit(expression)
    return reasons


# -------------------------------------------#
#               SelectClause                #
# -------------------------------------------#
@dataclass
class SelectClause:
    # expr with optional alias
    select_items: List[Tuple[Expr, Optional[str]]]

    def __init__(self, select_items: List[Tuple[Expr, Optional[str]]]):
        self.select_items = select_items

    def get_columns(self) -> Set[str]:
        cols = set()
        for expr, alias in self.select_items:
            cols.update(expr.collect_columns())
        return cols

    def to_string(self):
        parts = []
        for expr, alias in self.select_items:
            if alias:
                parts.append(f"{expr.to_string()} AS {alias}")
            else:
                parts.append(f"{expr.to_string()}")
        return "SELECT " + ", ".join(parts)


# -------------------------------------------#
#            MeasuresSubClause              #
# -------------------------------------------#
@dataclass
class MeasuresSubClause:
    measures: Dict[str, Expr]

    def __init__(self, measures: Dict[str, Expr]):
        self.measures = measures

    def get_measure(self, alias: str) -> Expr:
        return self.measures.get(alias)

    def get_all_measures(self) -> Dict[str, Expr]:
        return self.measures

    def has_func(self, func_name: str):
        # given a function name, check whether any measure contains this function
        for expr in self.measures.values():
            if expr.has_func(func_name):
                return True
        return False

    def get_columns(self) -> Set[str]:
        # get all columns used in measures
        cols = set()
        for expr in self.measures.values():
            cols.update(expr.collect_columns())
        return cols

    def to_string(self):
        parts = []
        for alias, expr in self.measures.items():
            parts.append(f"{expr.to_string()} AS {alias}")
        return "MEASURES " + ", ".join(parts)


# -------------------------------------------#
#               Condition                   #
# -------------------------------------------#
@dataclass
class Condition(ABC):
    """
    format: [var_name] AS [expression]
    """

    # var_name is the variable name in pattern, e.g., A, B, C
    var_name: str
    # expression is the boolean expression for this variable, e.g., A.price > 10 AND A.tax < 5
    expression: Expr
    # columns is the set of columns used in expression, e.g., {price, tax}
    columns: Set

    def __init__(self, var_name: str, expression: Expr):
        self.var_name = var_name
        self.expression = expression
        self.columns = self.expression.collect_columns()
        self._independent_filter_cache = None
        self._independent_filter_cache_valid = False
        self._independent_filter_expr_id = id(expression)

    @abstractmethod
    def describe(self):
        pass

    def get_columns(self) -> Set[str]:
        return self.columns

    def get_independent_filter(self) -> Optional[Expr]:
        """
        obtain independent condition for prefiltering
        Logic: Break down the top-level AND, retaining the parts that do not depend on cross-row/aggregation.
        """
        expr_id = id(self.expression)
        if (
            self._independent_filter_cache_valid
            and self._independent_filter_expr_id == expr_id
        ):
            return self._independent_filter_cache

        if isinstance(self, IndependentCondition):
            self._independent_filter_cache = self.expression
            self._independent_filter_cache_valid = True
            self._independent_filter_expr_id = expr_id
            return self.expression

        # 1. Flat AND expression
        conjuncts = []

        def flatten_and(node):
            if isinstance(node, BinaryExpr) and node.op == "AND":
                flatten_and(node.left)
                flatten_and(node.right)
            else:
                conjuncts.append(node)

        flatten_and(self.expression)

        # 2. capture independent condition
        independent_parts = []
        for part in conjuncts:
            if not self._is_subtree_dependent(part):
                independent_parts.append(part)

        # 3. restructure AST
        if not independent_parts:
            self._independent_filter_cache = None
            self._independent_filter_cache_valid = True
            self._independent_filter_expr_id = expr_id
            return None

        result = independent_parts[0]
        for i in range(1, len(independent_parts)):
            result = _simplify_logical_expr(result, "AND", independent_parts[i])
        self._independent_filter_cache = result
        self._independent_filter_cache_valid = True
        self._independent_filter_expr_id = expr_id
        return result

    def _is_subtree_dependent(self, node: Expr) -> bool:
        """check subtree whether contain dependent condition"""
        return bool(collect_define_dependency_reasons(self.var_name, node))


@dataclass
class IndependentCondition(Condition):
    def __init__(self, var_name, expression):
        super().__init__(var_name, expression)

    def describe(self):
        return f"independent condition"


@dataclass
class DependentCondition(Condition):
    def __init__(self, var_name, expression, reasons: List[str]):
        super().__init__(var_name, expression)
        self.reasons = reasons

    def describe(self):
        return f"Dependent: {', '.join(self.reasons)}"


@dataclass
class ColumnConstraint:
    lower: Optional[Union[int, float, str]] = None
    lower_inclusive: bool = True
    upper: Optional[Union[int, float, str]] = None
    upper_inclusive: bool = True
    allowed_values: Optional[Set[Union[int, float, str]]] = None
    excluded_values: Set[Union[int, float, str]] = field(default_factory=set)
    null_state: str = "any"  # any / only_null / not_null

    def apply_comparison(self, op: str, value: Union[int, float, str]) -> bool:
        if op == "=":
            return self.apply_allowed_values({value})
        if op in ("!=", "<>"):
            return self.apply_excluded_values({value})
        if op == ">":
            return self._mark_not_null() and self._update_lower(value, False)
        if op == ">=":
            return self._mark_not_null() and self._update_lower(value, True)
        if op == "<":
            return self._mark_not_null() and self._update_upper(value, False)
        if op == "<=":
            return self._mark_not_null() and self._update_upper(value, True)
        return True

    def apply_between(
        self, lower: Union[int, float, str], upper: Union[int, float, str]
    ) -> bool:
        return (
            self._mark_not_null()
            and self._update_lower(lower, True)
            and self._update_upper(upper, True)
        )

    def apply_allowed_values(self, values: Set[Union[int, float, str]]) -> bool:
        if not self._mark_not_null():
            return False
        if self.allowed_values is None:
            self.allowed_values = set(values)
        else:
            self.allowed_values &= values
        return self.is_satisfiable()

    def apply_excluded_values(self, values: Set[Union[int, float, str]]) -> bool:
        if not self._mark_not_null():
            return False
        self.excluded_values |= values
        return self.is_satisfiable()

    def apply_null_check(self, is_not: bool) -> bool:
        if is_not:
            return self._mark_not_null()
        if self.null_state == "not_null":
            return False
        self.null_state = "only_null"
        return self.is_satisfiable()

    def is_satisfiable(self) -> bool:
        if self.null_state == "only_null":
            return (
                self.allowed_values is None
                and self.lower is None
                and self.upper is None
            )

        if self.allowed_values is not None:
            candidates = {
                value
                for value in self.allowed_values
                if value not in self.excluded_values and self._within_bounds(value)
            }
            return len(candidates) > 0

        if self.lower is not None and self.upper is not None:
            compare_result = _compare_literal_values(self.lower, self.upper)
            if compare_result is None:
                return False
            if compare_result > 0:
                return False
            if compare_result == 0:
                if not self.lower_inclusive or not self.upper_inclusive:
                    return False
                return self.lower not in self.excluded_values

        return True

    def _mark_not_null(self) -> bool:
        if self.null_state == "only_null":
            return False
        self.null_state = "not_null"
        return True

    def _update_lower(self, value: Union[int, float, str], inclusive: bool) -> bool:
        if self.lower is None:
            self.lower = value
            self.lower_inclusive = inclusive
            return self.is_satisfiable()

        compare_result = _compare_literal_values(value, self.lower)
        if compare_result is None:
            return False
        if compare_result > 0:
            self.lower = value
            self.lower_inclusive = inclusive
        elif compare_result == 0:
            self.lower_inclusive = self.lower_inclusive and inclusive
        return self.is_satisfiable()

    def _update_upper(self, value: Union[int, float, str], inclusive: bool) -> bool:
        if self.upper is None:
            self.upper = value
            self.upper_inclusive = inclusive
            return self.is_satisfiable()

        compare_result = _compare_literal_values(value, self.upper)
        if compare_result is None:
            return False
        if compare_result < 0:
            self.upper = value
            self.upper_inclusive = inclusive
        elif compare_result == 0:
            self.upper_inclusive = self.upper_inclusive and inclusive
        return self.is_satisfiable()

    def _within_bounds(self, value: Union[int, float, str]) -> bool:
        if self.lower is not None:
            compare_result = _compare_literal_values(value, self.lower)
            if compare_result is None or compare_result < 0:
                return False
            if compare_result == 0 and not self.lower_inclusive:
                return False

        if self.upper is not None:
            compare_result = _compare_literal_values(value, self.upper)
            if compare_result is None or compare_result > 0:
                return False
            if compare_result == 0 and not self.upper_inclusive:
                return False
        return True


# -------------------------------------------#
#             DefineSubClause               #
# -------------------------------------------#
@dataclass
class DefineSubClause:
    definitions: Dict[str, Condition]

    def __init__(self, definitions: Dict):
        self.definitions = definitions

    def get_condition(self, var_name: str) -> Condition:
        return self.definitions[var_name]

    def get_columns(self) -> Set[str]:
        col_list = set()
        for cond in self.definitions.values():
            if cond is None:
                continue
            col_list |= cond.get_columns()
        return col_list

    def to_string(self) -> str:
        parts = []
        for var_name, condition in self.definitions.items():
            # here we remove the outer parentheses
            cond_str = _define_expr_to_string(condition.expression, var_name)
            cond_str = (
                cond_str[1:-1]
                if cond_str.startswith("(") and cond_str.endswith(")")
                else cond_str
            )
            parts.append(f"{var_name} AS {cond_str}")
        return "DEFINE " + ", ".join(parts)

    def get_ic_map(self, var_names: List[str]) -> List[Tuple[str, Expr]]:
        """
        ic is an abbreviation for independent condition
        only when all variables have independent condition, then return a noy null list
        please note that len(var_names) may greater than len(definitions)
        """
        ic_list = []
        for var_name in var_names:
            condition = self.definitions.get(var_name)
            if condition is None:
                return None
            expr = condition.get_independent_filter()
            if expr is None:
                return None
            else:
                ic_list.append((var_name, expr))
        return ic_list

    def all_var_with_ic(self, var_names: List[str]) -> bool:
        for var_name in var_names:
            condition = self.definitions.get(var_name)
            if condition is None or condition.get_independent_filter() is None:
                return False
        return True

    def without_dc(self) -> bool:
        for condition in self.definitions.values():
            if condition is not None and isinstance(condition, DependentCondition):
                return False
        return True

    def get_independent_expr(self, var_name: str) -> Optional[Expr]:
        if var_name not in self.definitions:
            raise KeyError(f"Unknown variable: {var_name}")
        condition = self.definitions[var_name]
        if condition is None:
            return None
        return condition.get_independent_filter()

    def has_expr_overlap(
        self,
        expr1: Optional[Expr],
        expr2: Optional[Expr],
        context: SimplifyContext,
    ) -> bool:
        if expr1 is None and expr2 is None:
            return True
        if expr1 is None:
            return self._is_expr_satisfiable(expr2)
        if expr2 is None:
            return self._is_expr_satisfiable(expr1)
        # Overlap asks only whether both predicates can be TRUE.
        combined = _simplify_logical_expr(
            expr1, "AND", expr2, context, preserve_unknown=False
        )
        return self._is_expr_satisfiable(combined)

    def _is_expr_satisfiable(self, expr: Optional[Expr]) -> bool:
        if expr is None:
            return True
        for atoms in self._iter_conjunctive_terms(expr):
            if self._is_conjunction_satisfiable(atoms):
                return True
        return False

    def _to_conjunctive_terms(self, expr: Expr) -> List[List[Expr]]:
        return list(self._iter_conjunctive_terms(expr))

    def _iter_conjunctive_terms(self, expr: Expr):
        if isinstance(expr, SemanticsExpr):
            yield from self._iter_conjunctive_terms(expr.expr)
            return

        if isinstance(expr, BinaryExpr) and expr.op == "OR":
            yield from self._iter_conjunctive_terms(expr.left)
            yield from self._iter_conjunctive_terms(expr.right)
            return

        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            right_terms = list(self._iter_conjunctive_terms(expr.right))
            for left_term in self._iter_conjunctive_terms(expr.left):
                for right_term in right_terms:
                    yield left_term + right_term
            return

        yield [expr]

    def _is_conjunction_satisfiable(self, atoms: List[Expr]) -> bool:
        constraints: Dict[str, ColumnConstraint] = {}
        for atom in atoms:
            if not self._apply_atom_constraint(atom, constraints):
                return False
        return all(constraint.is_satisfiable() for constraint in constraints.values())

    def _apply_atom_constraint(
        self, atom: Expr, constraints: Dict[str, ColumnConstraint]
    ) -> bool:
        if isinstance(atom, SemanticsExpr):
            return self._apply_atom_constraint(atom.expr, constraints)

        if isinstance(atom, NotExpr):
            negated_atom = self._negate_simple_atom(atom.expr)
            if negated_atom is None:
                return True
            return self._apply_atom_constraint(negated_atom, constraints)

        if isinstance(atom, BinaryExpr):
            if atom.op in ("AND", "OR"):
                return self._is_expr_satisfiable(atom)
            if atom.op in ("LIKE", "NOT LIKE"):
                return self._evaluate_like_expr(atom)
            return self._apply_binary_comparison(atom, constraints)

        if isinstance(atom, BetweenExpr):
            column_name = self._extract_column_name(atom.expr)
            lower_value = self._extract_literal_value(atom.lower)
            upper_value = self._extract_literal_value(atom.upper)
            if (
                column_name is not None
                and lower_value is not None
                and upper_value is not None
            ):
                return constraints.setdefault(
                    column_name, ColumnConstraint()
                ).apply_between(lower_value, upper_value)
            if lower_value is not None and upper_value is not None:
                expr_value = self._extract_literal_value(atom.expr)
                if expr_value is not None:
                    return lower_value <= expr_value <= upper_value
            return True

        if isinstance(atom, InExpr):
            column_name = self._extract_column_name(atom.expr)
            literal_values = self._extract_literal_values(atom.values)
            if column_name is not None and literal_values is not None:
                column_constraint = constraints.setdefault(
                    column_name, ColumnConstraint()
                )
                if atom.is_not:
                    return column_constraint.apply_excluded_values(set(literal_values))
                return column_constraint.apply_allowed_values(set(literal_values))
            expr_value = self._extract_literal_value(atom.expr)
            if expr_value is not None and literal_values is not None:
                contained = expr_value in literal_values
                return not contained if atom.is_not else contained
            return True

        if isinstance(atom, IsNullExpr):
            column_name = self._extract_column_name(atom.expr)
            if column_name is None:
                return True
            return constraints.setdefault(
                column_name, ColumnConstraint()
            ).apply_null_check(atom.is_not)

        return True

    def _apply_binary_comparison(
        self, atom: BinaryExpr, constraints: Dict[str, ColumnConstraint]
    ) -> bool:
        normalized_atom = _simplify_comparison_expr(atom.left, atom.op, atom.right)

        left_column = self._extract_column_name(normalized_atom.left)
        right_column = self._extract_column_name(normalized_atom.right)
        left_value = self._extract_literal_value(normalized_atom.left)
        right_value = self._extract_literal_value(normalized_atom.right)

        if left_column is not None and right_value is not None:
            return constraints.setdefault(
                left_column, ColumnConstraint()
            ).apply_comparison(normalized_atom.op, right_value)

        if right_column is not None and left_value is not None:
            reversed_op = self._reverse_comparison(normalized_atom.op)
            if reversed_op is None:
                return True
            return constraints.setdefault(
                right_column, ColumnConstraint()
            ).apply_comparison(reversed_op, left_value)

        if left_value is not None and right_value is not None:
            return self._evaluate_literal_comparison(
                left_value, normalized_atom.op, right_value
            )

        return True

    def _negate_simple_atom(self, atom: Expr) -> Optional[Expr]:
        if isinstance(atom, NotExpr):
            return atom.expr

        if isinstance(atom, BinaryExpr):
            reverse_op = {
                "=": "!=",
                "!=": "=",
                "<>": "=",
                ">": "<=",
                ">=": "<",
                "<": ">=",
                "<=": ">",
                "LIKE": "NOT LIKE",
                "NOT LIKE": "LIKE",
            }.get(atom.op)
            if reverse_op is None:
                return None
            return BinaryExpr(atom.left, reverse_op, atom.right)

        if isinstance(atom, InExpr):
            return InExpr(atom.expr, atom.values, not atom.is_not)

        if isinstance(atom, IsNullExpr):
            return IsNullExpr(atom.expr, not atom.is_not)

        return None

    def _extract_column_name(self, expr: Expr) -> Optional[str]:
        if isinstance(expr, SemanticsExpr):
            return self._extract_column_name(expr.expr)
        if isinstance(expr, ColumnRef) and expr.name != "*":
            return expr.name
        return None

    def _extract_literal_value(self, expr: Expr) -> Optional[Union[int, float, str]]:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, SemanticsExpr):
            return self._extract_literal_value(expr.expr)
        return None

    def _extract_literal_values(
        self, exprs: List[Expr]
    ) -> Optional[List[Union[int, float, str]]]:
        values = []
        for expr in exprs:
            literal_value = self._extract_literal_value(expr)
            if literal_value is None:
                return None
            values.append(literal_value)
        return values

    def _reverse_comparison(self, op: str) -> Optional[str]:
        return _reverse_comparison_op(op)

    def _evaluate_literal_comparison(
        self,
        left_value: Union[int, float, str],
        op: str,
        right_value: Union[int, float, str],
    ) -> bool:
        compare_result = _compare_literal_values(left_value, right_value)
        if compare_result is None:
            return False
        if op == "=":
            return compare_result == 0
        if op in ("!=", "<>"):
            return compare_result != 0
        if op == ">":
            return compare_result > 0
        if op == ">=":
            return compare_result >= 0
        if op == "<":
            return compare_result < 0
        if op == "<=":
            return compare_result <= 0
        return True

    def _evaluate_like_expr(self, atom: BinaryExpr) -> bool:
        left_value = self._extract_literal_value(atom.left)
        right_value = self._extract_literal_value(atom.right)
        if left_value is None or right_value is None:
            return True
        if not isinstance(left_value, str) or not isinstance(right_value, str):
            return False

        pattern = (
            "^" + re.escape(right_value).replace("%", ".*").replace("_", ".") + "$"
        )
        matched = re.match(pattern, left_value) is not None
        return not matched if atom.op == "NOT LIKE" else matched
