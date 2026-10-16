"""Expression simplification and normalization utilities."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

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
    SubqueryExpr,
)


@dataclass
class LinearExprForm:
    column: Optional[ColumnRef]
    coefficient: Decimal
    constant: Decimal


@dataclass(frozen=True)
class ExpressionColumnMetadata:
    """Column properties used by expression simplification."""

    data_type: Optional[str] = None
    nullable: Optional[bool] = None


@dataclass(frozen=True)
class SimplifyContext:
    """Resolve metadata without coupling the simplifier to the query AST."""

    resolve_column: Callable[[ColumnRef], Optional[object]]

    def get_column_metadata(self, column: ColumnRef) -> ExpressionColumnMetadata:
        metadata = self.resolve_column(column)
        if metadata is None:
            raise ValueError(
                f"missing schema metadata for column '{column.to_string()}'; "
                "provide the input table schema before simplifying expressions"
            )
        return ExpressionColumnMetadata(
            getattr(metadata, "data_type", None),
            getattr(metadata, "nullable", None),
        )


# we can add more data type in here
class RangeDomain(Enum):
    """Normalized value-domain semantics used by range operations."""

    INTEGER = "INTEGER"
    CONTINUOUS = "CONTINUOUS"
    TEXT = "TEXT"

    @classmethod
    def from_data_type(cls, data_type: str) -> "RangeDomain":
        """Map a schema data type to the range semantics that affect merging."""
        if not isinstance(data_type, str) or not data_type.strip():
            raise ValueError("a data type is required to determine the range domain")
        normalized = data_type.strip().upper()
        if normalized in ("INT", "LONG"):
            return cls.INTEGER
        if normalized in ("FLOAT", "DOUBLE"):
            return cls.CONTINUOUS
        if normalized.startswith(("CHAR(", "VARCHAR(")):
            return cls.TEXT
        raise ValueError(f"unsupported range data type '{data_type}'")


@dataclass
class RangeConstraint:
    """Represent a one-dimensional value range for a column.

    ``None`` means that the corresponding side is unbounded.  ``domain`` owns
    the type-specific range semantics.  Original boundaries are retained for
    expression rendering and normalized internally for semantic operations.
    """

    domain: RangeDomain
    column: Optional[ColumnRef] = None
    lower: Optional[Union[int, float, str, Literal]] = None
    lower_inclusive: bool = True
    upper: Optional[Union[int, float, str, Literal]] = None
    upper_inclusive: bool = True

    def __post_init__(self):
        """Validate that every constraint has explicit domain semantics."""
        if not isinstance(self.domain, RangeDomain):
            raise TypeError("domain must be a RangeDomain value")

    def _normalize_bound(self, value, lower: bool, inclusive: bool):
        """Return one boundary in this domain's canonical representation."""
        if value is None or self.domain is not RangeDomain.INTEGER:
            return value, inclusive

        raw_value = _range_bound_value(value)
        if not isinstance(raw_value, (int, float, Decimal)) or isinstance(
            raw_value, bool
        ):
            return value, inclusive

        decimal_value = (
            raw_value if isinstance(raw_value, Decimal) else Decimal(str(raw_value))
        )
        if lower:
            normalized = (
                decimal_value.to_integral_value(rounding=ROUND_CEILING)
                if inclusive
                else decimal_value.to_integral_value(rounding=ROUND_FLOOR) + 1
            )
        else:
            normalized = (
                decimal_value.to_integral_value(rounding=ROUND_FLOOR)
                if inclusive
                else decimal_value.to_integral_value(rounding=ROUND_CEILING) - 1
            )
        normalized_value = int(normalized)
        if isinstance(value, Literal):
            return Literal(normalized_value), True
        return normalized_value, True

    def _normalized_copy(self) -> "RangeConstraint":
        """Return a canonical copy for type-aware range calculations."""
        lower, lower_inclusive = self._normalize_bound(
            self.lower,
            lower=True,
            inclusive=self.lower_inclusive,
        )
        upper, upper_inclusive = self._normalize_bound(
            self.upper,
            lower=False,
            inclusive=self.upper_inclusive,
        )
        return RangeConstraint(
            self.domain,
            self.column,
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        )

    def has_bound(self) -> bool:
        """Return whether the range has at least one finite boundary."""
        return self.lower is not None or self.upper is not None

    def is_satisfiable(self) -> bool:
        """Return whether at least one value can satisfy both boundaries.

        A one-sided or fully unbounded range is always satisfiable.  When both
        bounds exist, the lower bound must be less than the upper bound; equal
        bounds are satisfiable only when both sides are inclusive.  Bounds
        that cannot be compared are treated as unsatisfiable.
        """
        normalized = self._normalized_copy()
        if normalized.lower is None or normalized.upper is None:
            return True

        compare_result = _compare_literal_values(
            _range_bound_value(normalized.lower),
            _range_bound_value(normalized.upper),
        )
        if compare_result is None:
            return False
        if compare_result < 0:
            return True
        if compare_result > 0:
            return False
        return normalized.lower_inclusive and normalized.upper_inclusive

    def apply_lower(self, value: Union[int, float, str, Literal], inclusive: bool):
        """Tighten this range in place with a candidate lower boundary.

        The greater lower bound wins.  If both bounds have the same value,
        the result is exclusive when either candidate is exclusive.  An
        incomparable candidate leaves the current boundary unchanged.
        """
        if self.lower is None:
            self.lower = value
            self.lower_inclusive = inclusive
            return

        compare_result = _compare_literal_values(
            _range_bound_value(value), _range_bound_value(self.lower)
        )
        if compare_result is None:
            return
        if compare_result > 0:
            self.lower = value
            self.lower_inclusive = inclusive
        elif compare_result == 0:
            self.lower_inclusive = self.lower_inclusive and inclusive

    def apply_upper(self, value: Union[int, float, str, Literal], inclusive: bool):
        """Tighten this range in place with a candidate upper boundary.

        The smaller upper bound wins.  If both bounds have the same value,
        the result is exclusive when either candidate is exclusive.  An
        incomparable candidate leaves the current boundary unchanged.
        """
        if self.upper is None:
            self.upper = value
            self.upper_inclusive = inclusive
            return

        compare_result = _compare_literal_values(
            _range_bound_value(value), _range_bound_value(self.upper)
        )
        if compare_result is None:
            return
        if compare_result < 0:
            self.upper = value
            self.upper_inclusive = inclusive
        elif compare_result == 0:
            self.upper_inclusive = self.upper_inclusive and inclusive

    def to_expr(self, col_name: Optional[str] = None) -> Optional[Expr]:
        """Convert the range back into an equivalent predicate expression.

        A point range becomes equality, a closed two-sided range becomes
        ``BETWEEN``, and all other bounded ranges use comparison predicates.
        An invalid range becomes false.  A fully unbounded range, or a range
        without either a stored column or fallback ``col_name``, returns
        ``None`` because no predicate can be emitted.
        """
        if not self.is_satisfiable():
            return _false_expr()

        column = self.column
        if column is None:
            if col_name is None:
                return None
            column = ColumnRef(None, col_name)

        if self.lower is None and self.upper is None:
            return None

        if (
            self.lower is not None
            and self.upper is not None
            and self.lower_inclusive
            and self.upper_inclusive
        ):
            if _range_bound_value(self.lower) == _range_bound_value(self.upper):
                return BinaryExpr(column, "=", _range_bound_literal(self.lower))
            return BetweenExpr(
                column,
                _range_bound_literal(self.lower),
                _range_bound_literal(self.upper),
            )

        clauses: List[Expr] = []
        if self.lower is not None:
            op = ">=" if self.lower_inclusive else ">"
            clauses.append(BinaryExpr(column, op, _range_bound_literal(self.lower)))
        if self.upper is not None:
            op = "<=" if self.upper_inclusive else "<"
            clauses.append(BinaryExpr(column, op, _range_bound_literal(self.upper)))

        if len(clauses) == 1:
            return clauses[0]
        return BinaryExpr(clauses[0], "AND", clauses[1])

    def intersect(self, other: "RangeConstraint") -> Optional["RangeConstraint"]:
        """Return the intersection of two constraints on the same column.

        The result keeps the tighter lower and upper boundaries.  ``None`` is
        returned when the columns or domains differ, a required comparison
        cannot be made, or the intersection is empty.  Neither input is
        mutated.
        """
        if (
            self.column is None
            or other.column is None
            or not _same_column_ref(self.column, other.column)
            or self.domain is not other.domain
        ):
            return None

        left = self._normalized_copy()
        other = other._normalized_copy()

        lower, lower_inclusive = left.lower, left.lower_inclusive
        upper, upper_inclusive = left.upper, left.upper_inclusive

        if other.lower is not None:
            if lower is None:
                lower, lower_inclusive = other.lower, other.lower_inclusive
            else:
                compare_result = _compare_literal_values(
                    _range_bound_value(other.lower), _range_bound_value(lower)
                )
                if compare_result is None:
                    return None
                if compare_result > 0:
                    lower, lower_inclusive = other.lower, other.lower_inclusive
                elif compare_result == 0:
                    lower_inclusive = lower_inclusive and other.lower_inclusive

        if other.upper is not None:
            if upper is None:
                upper, upper_inclusive = other.upper, other.upper_inclusive
            else:
                compare_result = _compare_literal_values(
                    _range_bound_value(other.upper), _range_bound_value(upper)
                )
                if compare_result is None:
                    return None
                if compare_result < 0:
                    upper, upper_inclusive = other.upper, other.upper_inclusive
                elif compare_result == 0:
                    upper_inclusive = upper_inclusive and other.upper_inclusive

        result = RangeConstraint(
            self.domain,
            self.column,
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        )
        return result if result.is_satisfiable() else None

    def union(self, other: "RangeConstraint") -> Optional["RangeConstraint"]:
        """Return one range covering the exact union when it is contiguous.

        Overlapping or boundary-touching constraints on the same column can
        be merged.  Inclusive adjacent ranges are also contiguous in the
        integer domain.  ``None`` is returned for different columns or
        domains, disjoint ranges, failed boundary comparisons, or an
        unsatisfiable result.  Inputs are not mutated.
        """
        if (
            self.column is None
            or other.column is None
            or not _same_column_ref(self.column, other.column)
            or self.domain is not other.domain
        ):
            return None

        left, right = self._normalized_copy()._ordered_with(
            other._normalized_copy(),
        )
        if not left._overlaps_or_connects(right):
            return None

        upper, upper_inclusive = left.upper, left.upper_inclusive
        if right.upper is None or upper is None:
            upper, upper_inclusive = None, True
        else:
            compare_result = _compare_literal_values(
                _range_bound_value(right.upper), _range_bound_value(upper)
            )
            if compare_result is None:
                return None
            if compare_result > 0:
                upper, upper_inclusive = right.upper, right.upper_inclusive
            elif compare_result == 0:
                upper_inclusive = upper_inclusive or right.upper_inclusive

        result = RangeConstraint(
            left.domain,
            left.column,
            left.lower,
            left.lower_inclusive,
            upper,
            upper_inclusive,
        )
        return result if result.is_satisfiable() else None

    def _ordered_with(
        self,
        other: "RangeConstraint",
    ) -> Tuple["RangeConstraint", "RangeConstraint"]:
        """Return these constraints ordered by their lower boundaries."""
        if self.lower is None:
            return self, other
        if other.lower is None:
            return other, self

        compare_result = _compare_literal_values(
            _range_bound_value(self.lower),
            _range_bound_value(other.lower),
        )
        if compare_result is None or compare_result < 0:
            return self, other
        if compare_result > 0:
            return other, self
        if self.lower_inclusive or not other.lower_inclusive:
            return self, other
        return other, self

    def _overlaps_or_connects(self, other: "RangeConstraint") -> bool:
        """Return whether two ordered ranges form one contiguous range."""
        if self.upper is None or other.lower is None:
            return True

        upper_value = _range_bound_value(self.upper)
        lower_value = _range_bound_value(other.lower)
        compare_result = _compare_literal_values(upper_value, lower_value)
        if compare_result is None:
            return False
        if compare_result > 0:
            return True
        if compare_result < 0:
            return (
                self.domain is RangeDomain.INTEGER
                and self.upper_inclusive
                and other.lower_inclusive
                and isinstance(upper_value, int)
                and isinstance(lower_value, int)
                and lower_value - upper_value == 1
            )
        return self.upper_inclusive or other.lower_inclusive


def _range_bound_value(
    value: Union[int, float, Decimal, str, Literal],
) -> Union[int, float, Decimal, str]:
    return value.value if isinstance(value, Literal) else value


def _range_bound_literal(value: Union[int, float, Decimal, str, Literal]) -> Literal:
    return value if isinstance(value, Literal) else _literal_to_expr(value)


def _compare_literal_values(
    left: Union[int, float, Decimal, str], right: Union[int, float, Decimal, str]
) -> Optional[int]:
    if isinstance(left, (int, float, Decimal)) and isinstance(
        right, (int, float, Decimal)
    ):
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    if isinstance(left, str) and isinstance(right, str):
        if left < right:
            return -1
        if left > right:
            return 1
        return 0

    return None


def _number_to_literal(value: Union[int, float, Decimal]) -> Literal:
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return Literal(int(value))
    if isinstance(value, float) and value.is_integer():
        return Literal(int(value))
    return Literal(value)


def _literal_to_expr(value: Union[int, float, Decimal, str]) -> Literal:
    if isinstance(value, (int, float, Decimal)):
        return _number_to_literal(value)
    return Literal(value)


def _true_expr() -> BinaryExpr:
    return BinaryExpr(Literal(1), "=", Literal(1))


def _false_expr() -> BinaryExpr:
    return BinaryExpr(Literal(1), "!=", Literal(1))


def _same_column_ref(left: ColumnRef, right: ColumnRef) -> bool:
    return left.qualifier == right.qualifier and left.name == right.name


def _scale_linear_form(form: LinearExprForm, factor: Decimal) -> LinearExprForm:
    return LinearExprForm(
        form.column, form.coefficient * factor, form.constant * factor
    )


def _combine_linear_forms(
    left: LinearExprForm,
    right: LinearExprForm,
    right_factor: Decimal = Decimal(1),
) -> Optional[LinearExprForm]:
    scaled_right = _scale_linear_form(right, right_factor)

    if (
        left.column is not None
        and scaled_right.column is not None
        and not _same_column_ref(left.column, scaled_right.column)
    ):
        return None

    column = left.column if left.column is not None else scaled_right.column
    return LinearExprForm(
        column,
        left.coefficient + scaled_right.coefficient,
        left.constant + scaled_right.constant,
    )


def _extract_linear_expr_form(expr: Expr) -> Optional[LinearExprForm]:
    if isinstance(expr, SemanticsExpr):
        return _extract_linear_expr_form(expr.expr)

    if isinstance(expr, Literal) and isinstance(expr.value, (int, float, Decimal)):
        value = (
            expr.value if isinstance(expr.value, Decimal) else Decimal(str(expr.value))
        )
        return LinearExprForm(None, Decimal(0), value)

    if isinstance(expr, ColumnRef) and expr.name != "*":
        return LinearExprForm(expr, Decimal(1), Decimal(0))

    if not isinstance(expr, BinaryExpr):
        return None

    left_form = _extract_linear_expr_form(expr.left)
    right_form = _extract_linear_expr_form(expr.right)
    if left_form is None or right_form is None:
        return None

    if expr.op == "+":
        return _combine_linear_forms(left_form, right_form)

    if expr.op == "-":
        return _combine_linear_forms(left_form, right_form, Decimal(-1))

    if expr.op == "*":
        if left_form.column is None:
            return _scale_linear_form(right_form, left_form.constant)
        if right_form.column is None:
            return _scale_linear_form(left_form, right_form.constant)
        return None

    if expr.op == "/":
        if right_form.column is None and right_form.constant != 0:
            return _scale_linear_form(left_form, Decimal(1) / right_form.constant)
        return None

    return None


def _reverse_comparison_op(op: str) -> Optional[str]:
    return {
        ">": "<",
        ">=": "<=",
        "<": ">",
        "<=": ">=",
        "=": "=",
        "!=": "!=",
        "<>": "<>",
    }.get(op)


def _evaluate_literal_comparison_value(
    left_value: Union[int, float, Decimal, str],
    op: str,
    right_value: Union[int, float, Decimal, str],
) -> Optional[bool]:
    compare_result = _compare_literal_values(left_value, right_value)
    if compare_result is None:
        return None
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
    return None


def _get_boolean_constant(expr: Expr) -> Optional[bool]:
    if isinstance(expr, SemanticsExpr):
        return _get_boolean_constant(expr.expr)

    if isinstance(expr, NotExpr):
        inner_value = _get_boolean_constant(expr.expr)
        return None if inner_value is None else not inner_value

    if isinstance(expr, BinaryExpr) and expr.op in (
        "=",
        "!=",
        "<>",
        ">",
        ">=",
        "<",
        "<=",
    ):
        if isinstance(expr.left, Literal) and isinstance(expr.right, Literal):
            return _evaluate_literal_comparison_value(
                expr.left.value, expr.op, expr.right.value
            )

    return None


def _flatten_logical_terms(expr: Expr, op: str) -> List[Expr]:
    if isinstance(expr, BinaryExpr) and expr.op == op:
        return _flatten_logical_terms(expr.left, op) + _flatten_logical_terms(
            expr.right, op
        )
    return [expr]


def _has_absorption(
    term: Expr,
    op: str,
    top_level_keys: Set[Tuple],
    current_key: Tuple,
    key_memo: Optional[Dict[int, Tuple]] = None,
) -> bool:
    opposite_op = "AND" if op == "OR" else "OR"
    if not isinstance(term, BinaryExpr) or term.op != opposite_op:
        return False

    for subterm in _flatten_logical_terms(term, opposite_op):
        subterm_key = _expr_key(subterm, key_memo)
        if subterm_key != current_key and subterm_key in top_level_keys:
            return True
    return False


def _get_complement_key(
    expr: Expr, key_memo: Optional[Dict[int, Tuple]] = None
) -> Tuple:
    if isinstance(expr, NotExpr):
        return _expr_key(expr.expr, key_memo)

    if isinstance(expr, BinaryExpr):
        opposite_op = {
            "=": "!=",
            "!=": "=",
            "<>": "=",
            ">": "<=",
            ">=": "<",
            "<": ">=",
            "<=": ">",
            "LIKE": "NOT LIKE",
            "NOT LIKE": "LIKE",
        }.get(expr.op)
        if opposite_op is not None:
            return (
                "BINARY",
                _expr_key(expr.left, key_memo),
                opposite_op,
                _expr_key(expr.right, key_memo),
            )

    if isinstance(expr, IsNullExpr):
        return ("ISNULL", _expr_key(expr.expr, key_memo), not expr.is_not)

    return ("NOT", _expr_key(expr, key_memo))


def _are_complementary(
    left: Expr, right: Expr, key_memo: Optional[Dict[int, Tuple]] = None
) -> bool:
    return _get_complement_key(left, key_memo) == _expr_key(
        right, key_memo
    ) or _get_complement_key(right, key_memo) == _expr_key(left, key_memo)


def _column_truth_guard(
    column: ColumnRef, context: Optional[SimplifyContext] = None
) -> Expr:
    """Represent TRUE for every non-NULL value of ``column``."""
    if context is None:
        raise ValueError(
            f"missing schema metadata for column '{column.to_string()}'; "
            "provide the input table schema before simplifying expressions"
        )
    metadata = context.get_column_metadata(column)
    if metadata.nullable is False:
        return _true_expr()
    return IsNullExpr(column, True)


def _is_two_valued(expr: Expr, context: Optional[SimplifyContext] = None) -> bool:
    """Conservatively prove that a predicate cannot evaluate to UNKNOWN."""
    if isinstance(expr, (NotExpr, SemanticsExpr)):
        return _is_two_valued(expr.expr, context)
    if isinstance(expr, IsNullExpr) or _get_boolean_constant(expr) is not None:
        return True
    if isinstance(expr, BinaryExpr):
        if expr.op in ("AND", "OR"):
            return _is_two_valued(expr.left, context) and _is_two_valued(
                expr.right, context
            )
        if expr.op in ("=", "!=", "<>", ">", ">=", "<", "<="):
            return all(
                (isinstance(operand, Literal) and operand.value != "")
                or (
                    isinstance(operand, ColumnRef)
                    and context is not None
                    and context.get_column_metadata(operand).nullable is False
                )
                for operand in (expr.left, expr.right)
            )
    return False


def _complement_or_guard(
    left: Expr,
    right: Expr,
    context: Optional[SimplifyContext] = None,
    *,
    preserve_unknown: bool = True,
) -> Optional[Expr]:
    """
    Reduce ``P OR NOT P`` without losing UNKNOWN in scalar contexts.

    Only a completed filter may replace UNKNOWN with FALSE and use
    ``column IS NOT NULL`` for the TRUE set of nullable comparisons.
    """
    if preserve_unknown:
        return _true_expr() if _is_two_valued(left, context) else None
    if isinstance(left, IsNullExpr) and isinstance(right, IsNullExpr):
        if (
            _expr_key(left.expr) == _expr_key(right.expr)
            and left.is_not != right.is_not
        ):
            return _true_expr()

    left_range = _extract_range_constraint(left, context)
    right_range = _extract_range_constraint(right, context)
    range_constraint = left_range if left_range is not None else right_range
    if range_constraint is not None:
        return _column_truth_guard(range_constraint.column, context)

    if isinstance(left, NotExpr):
        inner_range = _extract_range_constraint(left.expr, context)
        if inner_range is not None:
            return _column_truth_guard(inner_range.column, context)
    if isinstance(right, NotExpr):
        inner_range = _extract_range_constraint(right.expr, context)
        if inner_range is not None:
            return _column_truth_guard(inner_range.column, context)
    return None


def _get_distributive_absorption_result(
    left: Expr,
    right: Expr,
    op: str,
    key_memo: Optional[Dict[int, Tuple]] = None,
    context: Optional[SimplifyContext] = None,
    *,
    preserve_unknown: bool = True,
) -> Optional[Expr]:
    opposite_op = "AND" if op == "OR" else "OR"
    left_terms = _flatten_logical_terms(left, opposite_op)
    right_terms = _flatten_logical_terms(right, opposite_op)

    left_map = {}
    for term in left_terms:
        term_key = _expr_key(term, key_memo)
        if term_key not in left_map:
            left_map[term_key] = term

    right_map = {}
    for term in right_terms:
        term_key = _expr_key(term, key_memo)
        if term_key not in right_map:
            right_map[term_key] = term

    common_keys = set(left_map.keys()) & set(right_map.keys())
    left_only_keys = [key for key in left_map.keys() if key not in common_keys]
    right_only_keys = [key for key in right_map.keys() if key not in common_keys]

    if len(left_only_keys) != 1 or len(right_only_keys) != 1:
        return None

    left_only_expr = left_map[left_only_keys[0]]
    right_only_expr = right_map[right_only_keys[0]]
    if not _are_complementary(left_only_expr, right_only_expr, key_memo):
        return None
    if preserve_unknown and not _is_two_valued(left_only_expr, context):
        return None

    common_terms = [
        term for term in left_terms if _expr_key(term, key_memo) in common_keys
    ]
    if not common_terms:
        if op == "AND":
            return _false_expr()
        return _complement_or_guard(
            left_only_expr, right_only_expr, context, preserve_unknown=preserve_unknown
        )

    result = common_terms[0]
    for term in common_terms[1:]:
        result = _merge_logical_pair(
            result, opposite_op, term, context, preserve_unknown=preserve_unknown
        )
    if op == "OR":
        guard = _complement_or_guard(
            left_only_expr, right_only_expr, context, preserve_unknown=preserve_unknown
        )
        if guard is None:
            return None
        result = _simplify_logical_expr(
            result, "AND", guard, context, preserve_unknown=preserve_unknown
        )
    return result


def _expr_key(expr: Expr, memo: Optional[Dict[int, Tuple]] = None):
    if memo is None:
        memo = {}
    expr_id = id(expr)
    if expr_id in memo:
        return memo[expr_id]

    if isinstance(expr, Literal):
        key = ("LITERAL", expr.value)
    elif isinstance(expr, ColumnRef):
        key = ("COLUMN", expr.qualifier, expr.name)
    elif isinstance(expr, FunctionCall):
        key = ("FUNC", expr.name, tuple(_expr_key(arg, memo) for arg in expr.args))
    elif isinstance(expr, BinaryExpr):
        key = (
            "BINARY",
            _expr_key(expr.left, memo),
            expr.op,
            _expr_key(expr.right, memo),
        )
    elif isinstance(expr, BetweenExpr):
        key = (
            "BETWEEN",
            _expr_key(expr.expr, memo),
            _expr_key(expr.lower, memo),
            _expr_key(expr.upper, memo),
        )
    elif isinstance(expr, CaseExpr):
        key = (
            "CASE",
            tuple(
                (_expr_key(cond, memo), _expr_key(res, memo))
                for cond, res in expr.cases
            ),
            _expr_key(expr.else_expr, memo) if expr.else_expr is not None else None,
        )
    elif isinstance(expr, InExpr):
        key = (
            "IN",
            _expr_key(expr.expr, memo),
            tuple(_expr_key(value, memo) for value in expr.values),
            expr.is_not,
        )
    elif isinstance(expr, IsNullExpr):
        key = ("ISNULL", _expr_key(expr.expr, memo), expr.is_not)
    elif isinstance(expr, NotExpr):
        key = ("NOT", _expr_key(expr.expr, memo))
    elif isinstance(expr, SemanticsExpr):
        key = ("SEMANTICS", expr.semantics, _expr_key(expr.expr, memo))
    else:
        key = ("RAW", repr(expr))

    memo[expr_id] = key
    return key


def _range_domain_for_column(
    column: ColumnRef,
    context: Optional[SimplifyContext],
) -> Optional[RangeDomain]:
    """Resolve a column's schema type into range-operation semantics."""
    if context is None:
        return None
    metadata = context.get_column_metadata(column)
    if metadata.data_type is None:
        return None
    return RangeDomain.from_data_type(metadata.data_type)


def _merge_logical_pair(
    left: Expr,
    op: str,
    right: Expr,
    context: Optional[SimplifyContext] = None,
    *,
    preserve_unknown: bool = True,
) -> Expr:
    left_constant = _get_boolean_constant(left)
    right_constant = _get_boolean_constant(right)

    if op == "AND":
        if left_constant is False or right_constant is False:
            return _false_expr()
        if left_constant is True:
            return right
        if right_constant is True:
            return left
    else:
        if left_constant is True or right_constant is True:
            return _true_expr()
        if left_constant is False:
            return right
        if right_constant is False:
            return left

    left_range = _extract_range_constraint(left, context)
    right_range = _extract_range_constraint(right, context)
    if (
        left_range is not None
        and right_range is not None
        and _same_column_ref(left_range.column, right_range.column)
    ):
        if op == "AND":
            merged_range = left_range.intersect(right_range)
            if merged_range is None:
                if (
                    preserve_unknown
                    and context.get_column_metadata(left_range.column).nullable
                    is not False
                ):
                    return BinaryExpr(left, op, right)
                return _false_expr()
            return merged_range.to_expr()

        merged_range = left_range.union(right_range)
        if merged_range is not None:
            if (
                preserve_unknown
                and not merged_range.has_bound()
                and context.get_column_metadata(merged_range.column).nullable
                is not False
            ):
                return BinaryExpr(left, op, right)
            return merged_range.to_expr() or _column_truth_guard(
                merged_range.column,
                context,
            )

    return BinaryExpr(left, op, right)


def _extract_range_constraint(
    expr: Expr,
    context: Optional[SimplifyContext] = None,
) -> Optional[RangeConstraint]:
    if isinstance(expr, SemanticsExpr):
        return _extract_range_constraint(expr.expr, context)

    if isinstance(expr, BetweenExpr):
        if not isinstance(expr.expr, ColumnRef):
            return None
        if not isinstance(expr.lower, Literal) or not isinstance(expr.upper, Literal):
            return None
        domain = _range_domain_for_column(expr.expr, context)
        if domain is None:
            return None
        return RangeConstraint(
            domain,
            expr.expr,
            expr.lower.value,
            True,
            expr.upper.value,
            True,
        )

    if not isinstance(expr, BinaryExpr):
        return None

    normalized_expr = _simplify_comparison_expr(expr.left, expr.op, expr.right)
    if normalized_expr.op not in ("=", ">", ">=", "<", "<="):
        return None
    if not isinstance(normalized_expr.left, ColumnRef):
        return None
    if not isinstance(normalized_expr.right, Literal):
        return None

    value = normalized_expr.right.value
    domain = _range_domain_for_column(normalized_expr.left, context)
    if domain is None:
        return None
    if normalized_expr.op == "=":
        return RangeConstraint(
            domain,
            normalized_expr.left,
            value,
            True,
            value,
            True,
        )
    if normalized_expr.op == ">":
        return RangeConstraint(
            domain,
            normalized_expr.left,
            value,
            False,
            None,
            True,
        )
    if normalized_expr.op == ">=":
        return RangeConstraint(
            domain,
            normalized_expr.left,
            value,
            True,
            None,
            True,
        )
    if normalized_expr.op == "<":
        return RangeConstraint(
            domain,
            normalized_expr.left,
            None,
            True,
            value,
            False,
        )
    if normalized_expr.op == "<=":
        return RangeConstraint(
            domain,
            normalized_expr.left,
            None,
            True,
            value,
            True,
        )
    return None


def _simplify_logical_expr(
    left: Expr,
    op: str,
    right: Expr,
    context: Optional[SimplifyContext] = None,
    *,
    preserve_unknown: bool = True,
) -> Expr:
    """Preserve all three truth values unless called at a filter boundary."""
    if op not in ("AND", "OR"):
        return BinaryExpr(left, op, right)

    key_memo = {}
    raw_terms = _flatten_logical_terms(left, op) + _flatten_logical_terms(right, op)
    if op == "AND":
        raw_terms = _drop_redundant_not_true_null_guards(raw_terms, context)
    simplified_terms = []
    positive_keys = set()

    for term in raw_terms:
        boolean_constant = _get_boolean_constant(term)
        if op == "AND":
            if boolean_constant is False:
                return _false_expr()
            if boolean_constant is True:
                continue
        else:
            if boolean_constant is True:
                return _true_expr()
            if boolean_constant is False:
                continue

        term_key = _expr_key(term, key_memo)
        complement_key = _get_complement_key(term, key_memo)
        if complement_key in positive_keys:
            if op == "AND" and (not preserve_unknown or _is_two_valued(term, context)):
                return _false_expr()
            complement = next(
                existing
                for existing in simplified_terms
                if _expr_key(existing, key_memo) == complement_key
            )
            guard = (
                _complement_or_guard(
                    complement, term, context, preserve_unknown=preserve_unknown
                )
                if op == "OR"
                else None
            )
            if guard is not None:
                if _get_boolean_constant(guard) is True:
                    return guard
                # A nullable complement covers only non-NULL values. Other
                # OR terms can still accept NULL rows and must be retained.
                simplified_terms.remove(complement)
                positive_keys.remove(complement_key)
                term = guard
                term_key = _expr_key(term, key_memo)
        if term_key in positive_keys:
            continue
        positive_keys.add(term_key)
        simplified_terms.append(term)

    if not simplified_terms:
        return _true_expr() if op == "AND" else _false_expr()

    top_level_keys = {_expr_key(term, key_memo) for term in simplified_terms}
    filtered_terms = []
    for term in simplified_terms:
        term_key = _expr_key(term, key_memo)
        if _has_absorption(term, op, top_level_keys, term_key, key_memo):
            continue
        filtered_terms.append(term)

    if not filtered_terms:
        return _true_expr() if op == "AND" else _false_expr()

    for i in range(len(filtered_terms)):
        for j in range(i + 1, len(filtered_terms)):
            absorbed_expr = _get_distributive_absorption_result(
                filtered_terms[i],
                filtered_terms[j],
                op,
                key_memo,
                context,
                preserve_unknown=preserve_unknown,
            )
            if absorbed_expr is None:
                continue

            reduced_terms = [
                filtered_terms[k]
                for k in range(len(filtered_terms))
                if k != i and k != j
            ]
            reduced_terms.append(absorbed_expr)

            result = reduced_terms[0]
            for term in reduced_terms[1:]:
                result = _simplify_logical_expr(
                    result, op, term, context, preserve_unknown=preserve_unknown
                )
            return result

    result = filtered_terms[0]
    for term in filtered_terms[1:]:
        result = _merge_logical_pair(
            result, op, term, context, preserve_unknown=preserve_unknown
        )
    return result


def _negate_predicate(expr: Expr) -> Expr:
    if isinstance(expr, NotExpr):
        return expr.expr
    if isinstance(expr, BinaryExpr):
        opposite_op = {
            "=": "!=",
            "!=": "=",
            "<>": "=",
            ">": "<=",
            ">=": "<",
            "<": ">=",
            "<=": ">",
            "LIKE": "NOT LIKE",
            "NOT LIKE": "LIKE",
        }.get(expr.op)
        if opposite_op is not None:
            return BinaryExpr(expr.left, opposite_op, expr.right)
    if isinstance(expr, IsNullExpr):
        return IsNullExpr(expr.expr, not expr.is_not)
    return NotExpr(expr)


def simplify_filter_predicate(
    expr: Expr, context: Optional[SimplifyContext] = None
) -> Expr:
    """Preserve the TRUE set of a complete DEFINE/WHERE predicate.

    Push NOT through Boolean connectives before using filter-only reductions.
    Scalar operands, CASE results and function arguments keep the full SQL
    three-valued semantics established during parsing.
    """

    def term_count(node: Expr) -> int:
        if isinstance(node, (NotExpr, SemanticsExpr)):
            return term_count(node.expr)
        if isinstance(node, BinaryExpr) and node.op in ("AND", "OR"):
            return term_count(node.left) + term_count(node.right)
        return 1

    def visit(node: Expr, negated: bool = False) -> Expr:
        if isinstance(node, NotExpr):
            result = visit(node.expr, not negated)
            original = node.expr if negated else node
            # Keep the existing NOT spelling when De Morgan alone would
            # merely rearrange terms, with no predicate reduction.
            return result if term_count(result) < term_count(original) else original
        if isinstance(node, BinaryExpr) and node.op in ("AND", "OR"):
            op = ("OR" if node.op == "AND" else "AND") if negated else node.op
            return _simplify_logical_expr(
                visit(node.left, negated),
                op,
                visit(node.right, negated),
                context,
                preserve_unknown=False,
            )
        if isinstance(node, SemanticsExpr):
            return SemanticsExpr(node.semantics, visit(node.expr, negated))
        if negated:
            node = _negate_predicate(node)
        if isinstance(node, BinaryExpr) and node.op in (
            "=",
            "!=",
            "<>",
            ">",
            ">=",
            "<",
            "<=",
        ):
            return _simplify_comparison_expr(
                node.left, node.op, node.right, context, preserve_unknown=False
            )
        return node

    return visit(expr)


def _extract_not_true_guard(
    expr: Expr,
    context: Optional[SimplifyContext] = None,
) -> Optional[Tuple[Expr, Expr]]:
    if not isinstance(expr, BinaryExpr) or expr.op != "OR":
        return None
    terms = _flatten_logical_terms(expr, "OR")
    if len(terms) != 2:
        return None

    null_term = next(
        (term for term in terms if isinstance(term, IsNullExpr) and not term.is_not),
        None,
    )
    if null_term is None:
        return None
    complement = terms[1] if terms[0] is null_term else terms[0]
    if isinstance(null_term.expr, ColumnRef):
        complement_range = _extract_range_constraint(complement, context)
        if complement_range is not None and _same_column_ref(
            null_term.expr, complement_range.column
        ):
            return null_term.expr, complement
    if not _are_complementary(null_term.expr, complement):
        return None
    return null_term.expr, complement


def _drop_redundant_not_true_null_guards(
    terms: List[Expr],
    context: Optional[SimplifyContext] = None,
) -> List[Expr]:
    ranges = [_extract_range_constraint(term, context) for term in terms]
    result = []
    for index, term in enumerate(terms):
        guard = _extract_not_true_guard(term, context)
        if guard is None:
            result.append(term)
            continue

        guarded_column = guard[0] if isinstance(guard[0], ColumnRef) else None
        if guarded_column is None:
            guarded_range = _extract_range_constraint(guard[0], context)
            guarded_column = guarded_range.column if guarded_range is not None else None
        if guarded_column is None:
            result.append(term)
            continue
        has_non_null_guarantee = any(
            other_index != index
            and other_range is not None
            and _same_column_ref(guarded_column, other_range.column)
            for other_index, other_range in enumerate(ranges)
        )
        result.append(guard[1] if has_non_null_guarantee else term)
    return result


def _condition_is_not_true(
    expr: Expr,
    context: Optional[SimplifyContext] = None,
) -> Expr:
    """Return an Oracle-valid predicate that accepts FALSE and UNKNOWN."""

    def may_be_null(operand: Expr) -> bool:
        if isinstance(operand, Literal):
            return False
        if isinstance(operand, ColumnRef) and context is not None:
            return context.get_column_metadata(operand).nullable is not False
        return True

    if isinstance(expr, SemanticsExpr):
        return _condition_is_not_true(expr.expr, context)

    if isinstance(expr, BinaryExpr) and expr.op in ("AND", "OR"):
        result_op = "OR" if expr.op == "AND" else "AND"
        return _simplify_logical_expr(
            _condition_is_not_true(expr.left, context),
            result_op,
            _condition_is_not_true(expr.right, context),
            context,
        )

    if isinstance(expr, IsNullExpr):
        return IsNullExpr(expr.expr, not expr.is_not)

    comparison_ops = {
        "=",
        "!=",
        "<>",
        ">",
        ">=",
        "<",
        "<=",
        "LIKE",
        "NOT LIKE",
    }
    if isinstance(expr, BinaryExpr) and expr.op in comparison_ops:
        terms: List[Expr] = [
            IsNullExpr(operand)
            for operand in (expr.left, expr.right)
            if may_be_null(operand)
        ]
        terms.append(_negate_predicate(expr))
        result = terms[0]
        for term in terms[1:]:
            result = _simplify_logical_expr(result, "OR", term, context)
        return result

    if isinstance(expr, BetweenExpr):
        terms = [
            IsNullExpr(value)
            for value in (expr.expr, expr.lower, expr.upper)
            if may_be_null(value)
        ]
        terms.append(NotExpr(expr))
        result = terms[0]
        for term in terms[1:]:
            result = _simplify_logical_expr(result, "OR", term, context)
        return result

    if isinstance(expr, InExpr):
        # A NULL list member does not prevent another member from matching:
        # 1 IN (1, NULL) is TRUE. Keep CASE's exact TRUE test in that case.
        if any(
            isinstance(value, SubqueryExpr) or may_be_null(value)
            for value in expr.values
        ):
            return BinaryExpr(
                CaseExpr([(expr, Literal(0))], Literal(1)), "=", Literal(1)
            )
        terms = []
        if may_be_null(expr.expr):
            terms.append(IsNullExpr(expr.expr))
        terms.extend(
            IsNullExpr(value)
            for value in expr.values
            if not isinstance(value, SubqueryExpr) and may_be_null(value)
        )
        terms.append(InExpr(expr.expr, expr.values, not expr.is_not))
        result = terms[0]
        for term in terms[1:]:
            result = _simplify_logical_expr(result, "OR", term, context)
        return result

    # LNNVL is Oracle's native FALSE-or-UNKNOWN predicate. Compound AND/OR
    # conditions are handled above because Oracle restricts LNNVL's argument.
    return FunctionCall("LNNVL", [expr])


def _combine_case_terms(
    terms: List[Expr],
    context: Optional[SimplifyContext] = None,
) -> Expr:
    if not terms:
        return _false_expr()
    result = terms[0]
    for term in terms[1:]:
        result = _simplify_logical_expr(
            result, "OR", term, context, preserve_unknown=False
        )
    return result


def _simplify_case_comparison(
    left: Expr,
    op: str,
    right: Expr,
    context: Optional[SimplifyContext] = None,
) -> Optional[Expr]:
    """Preserve a searched CASE comparison's TRUE set at a filter boundary."""
    case_expr = None
    comparison_expr = None
    case_on_left = True
    if isinstance(left, CaseExpr) and not isinstance(right, CaseExpr):
        case_expr = left
        comparison_expr = right
    elif isinstance(right, CaseExpr) and not isinstance(left, CaseExpr):
        case_expr = right
        comparison_expr = left
        case_on_left = False

    if case_expr is None or not case_expr.cases:
        return None
    results = [result for _, result in case_expr.cases]

    def branch_comparison(result: Expr) -> Expr:
        if case_on_left:
            return _simplify_comparison_expr(
                result, op, comparison_expr, context, preserve_unknown=False
            )
        return _simplify_comparison_expr(
            comparison_expr, op, result, context, preserve_unknown=False
        )

    branch_predicates = [branch_comparison(result) for result in results]
    else_predicate = (
        branch_comparison(case_expr.else_expr)
        if case_expr.else_expr is not None
        else _false_expr()
    )
    branch_results = [
        _get_boolean_constant(predicate) for predicate in branch_predicates
    ]
    else_result = _get_boolean_constant(else_predicate)

    if all(result is True for result in branch_results) and else_result is True:
        return _true_expr()
    if all(result is False for result in branch_results) and else_result is False:
        return _false_expr()
    if all(result is True for result in branch_results) and else_result is False:
        return _combine_case_terms(
            [condition for condition, _ in case_expr.cases],
            context,
        )

    # CASE selects the first condition that is TRUE; FALSE and NULL both fall through.
    active: Expr = _true_expr()
    matching_terms = []
    for (condition, _), predicate, result in zip(
        case_expr.cases, branch_predicates, branch_results
    ):
        if result is not False:
            selected = _simplify_logical_expr(
                active, "AND", condition, context, preserve_unknown=False
            )
            if result is not True:
                selected = _simplify_logical_expr(
                    selected,
                    "AND",
                    predicate,
                    context,
                    preserve_unknown=False,
                )
            matching_terms.append(selected)
        active = _simplify_logical_expr(
            active,
            "AND",
            _condition_is_not_true(condition, context),
            context,
            preserve_unknown=False,
        )
    if else_result is not False:
        matching_terms.append(
            active
            if else_result is True
            else _simplify_logical_expr(
                active,
                "AND",
                else_predicate,
                context,
                preserve_unknown=False,
            )
        )
    return _combine_case_terms(matching_terms, context)


def _simplify_comparison_expr(
    left: Expr,
    op: str,
    right: Expr,
    context: Optional[SimplifyContext] = None,
    *,
    preserve_unknown: bool = True,
) -> Expr:
    # The branch disjunction preserves the TRUE set, but can replace FALSE
    # with UNKNOWN (and an omitted ELSE's UNKNOWN with FALSE). Defer it until
    # the complete filter is parsed, after any surrounding NOT is handled.
    if not preserve_unknown:
        simplified_case = _simplify_case_comparison(left, op, right, context)
        if simplified_case is not None:
            return simplified_case

    if isinstance(left, Literal) and isinstance(right, Literal):
        literal_result = _evaluate_literal_comparison_value(left.value, op, right.value)
        if literal_result is True:
            return _true_expr()
        if literal_result is False:
            return _false_expr()

    left_form = _extract_linear_expr_form(left)
    right_form = _extract_linear_expr_form(right)
    if left_form is None or right_form is None:
        return BinaryExpr(left, op, right)

    merged_form = _combine_linear_forms(left_form, right_form, Decimal(-1))
    if merged_form is None or merged_form.column is None:
        return BinaryExpr(left, op, right)

    if abs(merged_form.coefficient) < Decimal("1e-12"):
        return BinaryExpr(left, op, right)

    normalized_op = op
    if merged_form.coefficient < 0:
        normalized_op = _reverse_comparison_op(op)
        if normalized_op is None:
            return BinaryExpr(left, op, right)

    threshold = -merged_form.constant / merged_form.coefficient
    return BinaryExpr(merged_form.column, normalized_op, _number_to_literal(threshold))


def _define_expr_to_string(
    expr: Expr, current_var: str, inside_navigation: bool = False
) -> str:
    """
    Render DEFINE expressions consistently.

    In DEFINE A AS ..., A.COL and COL are equivalent for ordinary references,
    so self-qualified columns are printed as COL.  Navigation functions use
    the qualifier to identify the navigation base row, so PREV/NEXT/FIRST/LAST
    keep qualifiers inside their arguments.
    """
    if expr is None:
        return ""
    if isinstance(expr, Literal):
        return expr.to_string()
    if isinstance(expr, ColumnRef):
        if (
            not inside_navigation
            and expr.qualifier is not None
            and expr.qualifier.upper() == current_var.upper()
        ):
            return expr.name
        return expr.to_string()
    if isinstance(expr, FunctionCall):
        func_name = expr.name.upper()
        preserve_args = inside_navigation or func_name in (
            "PREV",
            "NEXT",
            "FIRST",
            "LAST",
            "AVG",
            "SUM",
            "COUNT",
            "MAX",
            "MIN",
        )
        arg_strs = [
            _define_expr_to_string(arg, current_var, preserve_args) for arg in expr.args
        ]
        if func_name == "LNNVL" and len(expr.args) == 1:
            return f"{expr.name}({arg_strs[0]})"
        if len(expr.args) == 1 and isinstance(expr.args[0], BinaryExpr):
            return f"{expr.name}{arg_strs[0]}"
        return f"{expr.name}({', '.join(arg_strs)})"
    if isinstance(expr, BinaryExpr):
        left = _define_expr_to_string(expr.left, current_var, inside_navigation)
        right = _define_expr_to_string(expr.right, current_var, inside_navigation)
        if expr.op in [">=", "<=", "!=", "<>", "=", ">", "<"]:
            return f"{left} {expr.op} {right}"
        return f"({left} {expr.op} {right})"
    if isinstance(expr, BetweenExpr):
        inner = _define_expr_to_string(expr.expr, current_var, inside_navigation)
        lower = _define_expr_to_string(expr.lower, current_var, inside_navigation)
        upper = _define_expr_to_string(expr.upper, current_var, inside_navigation)
        return f"({inner} BETWEEN {lower} AND {upper})"
    if isinstance(expr, CaseExpr):
        parts = ["CASE"]
        for cond, res in expr.cases:
            parts.append(
                f"WHEN {_define_expr_to_string(cond, current_var, inside_navigation)} "
                f"THEN {_define_expr_to_string(res, current_var, inside_navigation)}"
            )
        if expr.else_expr:
            parts.append(
                f"ELSE {_define_expr_to_string(expr.else_expr, current_var, inside_navigation)}"
            )
        parts.append("END")
        return " ".join(parts)
    if isinstance(expr, InExpr):
        inner = _define_expr_to_string(expr.expr, current_var, inside_navigation)
        vals = ", ".join(
            _define_expr_to_string(value, current_var, inside_navigation)
            for value in expr.values
        )
        op = "NOT IN" if expr.is_not else "IN"
        return f"({inner} {op} ({vals}))"
    if isinstance(expr, SubqueryExpr):
        return expr.to_string()
    if isinstance(expr, IsNullExpr):
        inner = _define_expr_to_string(expr.expr, current_var, inside_navigation)
        return f"({inner} IS {'NOT ' if expr.is_not else ''}NULL)"
    if isinstance(expr, NotExpr):
        return (
            f"NOT({_define_expr_to_string(expr.expr, current_var, inside_navigation)})"
        )
    if isinstance(expr, SemanticsExpr):
        inner = _define_expr_to_string(expr.expr, current_var, inside_navigation)
        if expr.semantics is None or expr.semantics == "RUNNING":
            return inner
        return f"{expr.semantics} {inner}"
    return expr.to_string()
