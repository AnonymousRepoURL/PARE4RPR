"""Expression AST nodes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Set, Tuple, Union


class Expr(ABC):
    """Expr is a basic class"""

    @abstractmethod
    def collect_columns(self) -> Set[str]:
        pass

    @abstractmethod
    def to_string(self):
        pass

    @abstractmethod
    def has_func(self, func_name: str) -> bool:
        pass


@dataclass
class Literal(Expr):
    """e.g., name_str, 10, 1.23"""

    value: Union[int, float, Decimal, str]

    def __init__(self, value: Union[int, float, Decimal, str]):
        self.value = value

    def collect_columns(self):
        return set()

    def to_string(self):
        if type(self.value) is str:
            return f"'{self.value}'"

        return f"{self.value}"

    def has_func(self, func_name: str):
        return False


@dataclass(frozen=True)
class NullLiteral(Expr):
    """The SQL NULL literal.

    Keep NULL separate from ``Literal`` so code that treats ordinary literals
    as concrete comparable values cannot accidentally treat SQL NULL as a
    Python value.
    """

    def collect_columns(self) -> Set[str]:
        return set()

    def to_string(self) -> str:
        return "NULL"

    def has_func(self, func_name: str) -> bool:
        return False


@dataclass
class ColumnRef(Expr):
    """e.g., A.price, price, *"""

    qualifier: Optional[str]
    name: str  # column name

    def __init__(self, qualifier: Optional[str], name: str):
        self.qualifier = qualifier
        self.name = name

    def collect_columns(self):
        if self.name == "*":
            return set()
        return {self.name}

    def to_string(self):
        return f"{self.qualifier}.{self.name}" if self.qualifier else self.name

    def has_func(self, func_name: str):
        return False


@dataclass
class FunctionCall(Expr):
    """e.g., MATCH_NUMBER(), SUM(A.pirce)"""

    name: str  # function name
    args: List[Expr]  # function argument

    def __init__(self, name: str, args: List[Expr]):
        self.name = name
        self.args = args

    def collect_columns(self) -> Set:
        cols = set()
        for arg in self.args:
            cols.update(arg.collect_columns())
        return cols

    def to_string(self) -> str:
        if self.name.upper() == "LNNVL" and len(self.args) == 1:
            return f"{self.name}({self.args[0].to_string()})"
        # add this special case to avoid extra parentheses
        if len(self.args) == 1 and isinstance(self.args[0], BinaryExpr):
            return f"{self.name}{self.args[0].to_string()}"
        return f"{self.name}({', '.join([arg.to_string() for arg in self.args])})"

    def has_func(self, func_name: str):
        if self.name.upper() == func_name.upper():
            return True
        return any(arg.has_func(func_name) for arg in self.args)


@dataclass
class BinaryExpr(Expr):
    """e.g., price > 50 OR tax < 20"""

    left: Expr
    op: str  # op may be 'AND', 'OR', >, <, *, +
    right: Expr

    def __init__(self, left: Expr, op: str, right: Expr):
        self.left = left
        self.op = op
        self.right = right

    def collect_columns(self):
        return self.left.collect_columns().union(self.right.collect_columns())

    def to_string(self):
        # special case to avoid extra parentheses
        if self.op in [">=", "<=", "!=", "<>", "=", ">", "<"]:
            return f"{self.left.to_string()} {self.op} {self.right.to_string()}"
        return f"({self.left.to_string()} {self.op} {self.right.to_string()})"

    def has_func(self, func_name: str):
        return self.left.has_func(func_name) or self.right.has_func(func_name)


@dataclass
class BetweenExpr(Expr):
    """e.g., Expr1 BETWEEN Expr2 AND Expr3"""

    expr: Expr
    lower: Expr
    upper: Expr

    def __init__(self, expr: Expr, lower: Expr, upper: Expr):
        self.expr = expr
        self.lower = lower
        self.upper = upper

    def collect_columns(self):
        # may be we can directly return self.expr.collect_columns()
        return (
            self.expr.collect_columns()
            | self.lower.collect_columns()
            | self.upper.collect_columns()
        )

    def to_string(self):
        return f"({self.expr.to_string()} BETWEEN {self.lower.to_string()} AND {self.upper.to_string()})"

    def has_func(self, func_name: str):
        return (
            self.expr.has_func(func_name)
            or self.lower.has_func(func_name)
            or self.upper.has_func(func_name)
        )


@dataclass
class CaseExpr(Expr):
    """e.g., CASE WHEN ... THEN ... ELSE ... END"""

    cases: List[Tuple[Expr, Expr]]
    else_expr: Optional[Expr]

    def __init__(self, cases: List[Tuple[Expr, Expr]], else_expr: Optional[Expr]):
        # cases is [(condition, result), (condition, result), ...]
        self.cases = cases
        self.else_expr = else_expr

    def collect_columns(self):
        cols = set()
        for c, r in self.cases:
            cols.update(c.collect_columns())
            cols.update(r.collect_columns())
        if self.else_expr:
            cols.update(self.else_expr.collect_columns())
        return cols

    def to_string(self):
        parts = ["CASE"]
        for cond, res in self.cases:
            parts.append(f"WHEN {cond.to_string()} THEN {res.to_string()}")
        if self.else_expr:
            parts.append(f"ELSE {self.else_expr.to_string()}")
        parts.append("END")
        return " ".join(parts)

    def has_func(self, func_name: str):
        if any(c.has_func(func_name) or r.has_func(func_name) for c, r in self.cases):
            return True
        return self.else_expr.has_func(func_name) if self.else_expr else False


@dataclass
class InExpr(Expr):
    """e.g.,  expr IN (v1, v2)"""

    expr: Expr
    values: List[Expr]
    is_not: bool

    def __init__(self, expr: Expr, values: List[Expr], is_not: bool = False):
        self.expr = expr
        self.values = values
        self.is_not = is_not

    def collect_columns(self):
        # maybe we can directly return self.expr.collect_columns()
        cols = self.expr.collect_columns()
        for v in self.values:
            cols.update(v.collect_columns())
        return cols

    def to_string(self):
        op = "NOT IN" if self.is_not else "IN"
        vals = ", ".join([v.to_string() for v in self.values])
        return f"({self.expr.to_string()} {op} ({vals}))"

    def has_func(self, func_name: str):
        return self.expr.has_func(func_name) or any(
            v.has_func(func_name) for v in self.values
        )


@dataclass
class SubqueryExpr(Expr):
    """Represents a subquery used inside IN / NOT IN, e.g., IN (SELECT * FROM TB)."""

    sql: str

    def __init__(self, sql: str):
        self.sql = sql

    def collect_columns(self) -> Set[str]:
        # Cannot easily analyze subquery internals; treat as opaque.
        return set()

    def to_string(self) -> str:
        return self.sql

    def has_func(self, func_name: str) -> bool:
        return False


class IsNullExpr(Expr):
    """e.g., IS NULL / IS NOT NULL"""

    def __init__(self, expr: Expr, is_not: bool = False):
        self.expr, self.is_not = expr, is_not

    def collect_columns(self):
        return self.expr.collect_columns()

    def to_string(self):
        return f"({self.expr.to_string()} IS {'NOT ' if self.is_not else ''}NULL)"

    def has_func(self, func_name: str):
        return self.expr.has_func(func_name)


@dataclass
class NotExpr(Expr):
    expr: Expr
    """e.g., NOT expr"""

    def __init__(self, expr: Expr):
        self.expr = expr

    def collect_columns(self):
        return self.expr.collect_columns()

    def to_string(self):
        return f"NOT({self.expr.to_string()})"

    def has_func(self, func_name: str):
        return self.expr.has_func(func_name)


@dataclass
class SemanticsExpr(Expr):
    """defualt is RUNNING"""

    semantics: str  # RUNNING or FINAL
    expr: Expr

    def __init__(self, semantics: str, expr: Expr):
        self.semantics = semantics
        self.expr = expr

    def to_string(self):
        if self.semantics is None or self.semantics == "RUNNING":
            return f"{self.expr.to_string()}"
        return f"{self.semantics} {self.expr.to_string()}"

    def collect_columns(self):
        return self.expr.collect_columns()

    def has_func(self, func_name: str):
        return self.expr.has_func(func_name)
