"""Recursive-descent parser for scalar/boolean expressions."""

from decimal import Decimal
from typing import List, Optional, Set, Tuple
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
    NullLiteral,
    NotExpr,
    SubqueryExpr,
)
from expression_simplifier import (
    SimplifyContext,
    _negate_predicate,
    _simplify_comparison_expr,
    _simplify_logical_expr,
)


class ExpressionParser:
    """
    Parse the expression-bearing SQL fragments used by MATCH_RECOGNIZE.
    """

    _TOKEN_SPEC = [
        ("STRING", r"'[^']*'"),
        ("NUMBER", r"\d+(\.\d*)?"),
        ("SEMANTICS", r"\b(RUNNING|FINAL)\b"),
        ("BETWEEN", r"\bBETWEEN\b"),
        ("IN", r"\bIN\b"),
        ("IS", r"\bIS\b"),
        ("NULL", r"\bNULL\b"),
        ("LIKE", r"\bLIKE\b"),
        ("NOT", r"\bNOT\b"),
        ("CASE_KW", r"\b(CASE|WHEN|THEN|ELSE|END)\b"),
        ("LOGIC_OP", r"\b(AND|OR)\b"),
        ("KEYWORD", r"\bAS\b"),
        ("COMP_OP", r"(>=|<=|!=|<>|=|>|<)"),
        ("MATH_OP", r"(\+|\-|\*|/)"),
        ("COMMA", r","),
        ("DOT", r"\."),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("ID", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("SKIP", r"[ \t\n]+"),
        ("MISMATCH", r"."),
    ]
    _TOKEN_REGEX = re.compile(
        "|".join("(?P<%s>%s)" % pair for pair in _TOKEN_SPEC),
        re.IGNORECASE,
    )
    _ID_COMPATIBLE_TOKENS = {
        "KEYWORD",
        "CASE_KW",
        "BETWEEN",
        "IN",
        "IS",
        "NULL",
        "LIKE",
        "NOT",
    }
    _RPR_AGGREGATES = ("AVG", "SUM", "COUNT", "MAX", "MIN")
    _RPR_NAVIGATION = ("PREV", "NEXT", "FIRST", "LAST")

    def __init__(self, simplify_context: Optional[SimplifyContext] = None):
        self.tokens: List[Tuple[str, str]] = []
        self.pos = 0
        self.simplify_context = simplify_context
        self._enforce_rpr_operator_rules = False
        self._referenced_qualifiers: Set[str] = set()

    def parse_expression(self, expression_sql: str) -> Expr:
        self._reset(expression_sql)
        return self._parse_or()

    def _reset(self, sql: str, clause_name: Optional[str] = None):
        self._referenced_qualifiers = set()
        self._enforce_rpr_operator_rules = (
            clause_name is not None and clause_name.upper() in ("MEASURES", "DEFINE")
        )
        if clause_name is None:
            sql_body = sql
        else:
            sql_body = re.sub(rf"^\s*{clause_name}\s+", "", sql, flags=re.IGNORECASE)
        self.tokens = self._tokenize(sql_body)
        self.pos = 0

    def _tokenize(self, text: str) -> List[Tuple[str, str]]:
        tokens = []
        for match in self._TOKEN_REGEX.finditer(text):
            kind = match.lastgroup
            value = match.group()
            if kind == "SKIP":
                continue
            if kind == "MISMATCH":
                raise ValueError(f"Unexpected char: {value}")
            tokens.append((kind, value))
        return tokens

    def _parse_comma_separated(self, parse_item):
        items = []
        while not self._at_end():
            items.append(parse_item())
            if not self._match("COMMA"):
                break
        return items

    def _parse_or(self):
        lhs = self._parse_and()
        while self._match("LOGIC_OP", "OR"):
            lhs = _simplify_logical_expr(
                lhs,
                "OR",
                self._parse_and(),
                self.simplify_context,
            )
        return lhs

    def _parse_and(self):
        lhs = self._parse_not()
        while self._match("LOGIC_OP", "AND"):
            lhs = _simplify_logical_expr(
                lhs,
                "AND",
                self._parse_not(),
                self.simplify_context,
            )
        return lhs

    def _parse_not(self):
        if self._match("NOT"):
            return _negate_predicate(self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self):
        lhs = self._parse_additive()
        if self._at_end():
            return lhs

        tk_type, tk_val = self._peek()
        if tk_type == "COMP_OP":
            self.pos += 1
            return _simplify_comparison_expr(
                lhs,
                tk_val.upper(),
                self._parse_additive(),
                self.simplify_context,
            )
        if tk_type == "LIKE":
            self.pos += 1
            return BinaryExpr(lhs, "LIKE", self._parse_additive())
        if tk_type == "BETWEEN":
            self.pos += 1
            lower = self._parse_additive()
            self._consume("LOGIC_OP", "AND")
            return BetweenExpr(lhs, lower, self._parse_additive())
        if tk_type == "IS":
            self.pos += 1
            is_not = self._match("NOT")
            self._consume("NULL")
            return IsNullExpr(lhs, is_not)
        if tk_type == "NOT":
            next_type = (
                self.tokens[self.pos + 1][0]
                if self.pos + 1 < len(self.tokens)
                else None
            )
            if next_type == "IN":
                self.pos += 2
                return InExpr(lhs, self._parse_in_list(), is_not=True)
            if next_type == "LIKE":
                self.pos += 2
                return BinaryExpr(lhs, "NOT LIKE", self._parse_additive())
            return lhs
        if tk_type == "IN":
            self.pos += 1
            return InExpr(lhs, self._parse_in_list(), is_not=False)

        return lhs

    def _parse_additive(self):
        lhs = self._parse_multiplicative()
        while self._peek_type("MATH_OP") and self._peek()[1] in ("+", "-"):
            op = self._consume("MATH_OP")
            lhs = BinaryExpr(lhs, op, self._parse_multiplicative())
        return lhs

    def _parse_multiplicative(self):
        lhs = self._parse_unary_math()
        while self._peek_type("MATH_OP") and self._peek()[1] in ("*", "/"):
            op = self._consume("MATH_OP")
            lhs = BinaryExpr(lhs, op, self._parse_unary_math())
        return lhs

    def _parse_unary_math(self):
        if self._peek_type("MATH_OP") and self._peek()[1] in ("+", "-"):
            op = self._consume("MATH_OP")
            operand = self._parse_unary_math()
            if isinstance(operand, Literal) and isinstance(
                operand.value, (int, float, Decimal)
            ):
                return Literal(-operand.value if op == "-" else operand.value)
            return BinaryExpr(Literal(0), op, operand)
        return self._parse_primary()

    def _parse_primary(self):
        if self._at_end():
            raise ValueError("Unexpected end")

        tk_type, tk_val = self._peek()
        if self._peek_type("CASE_KW", "CASE"):
            self.pos += 1
            return self._parse_case_expression()

        self.pos += 1

        if tk_type == "LPAREN":
            expr = self._parse_or()
            self._consume("RPAREN")
            return expr
        if tk_type == "NUMBER":
            return Literal(Decimal(tk_val) if "." in tk_val else int(tk_val))
        if tk_type == "STRING":
            return Literal(tk_val.strip("'"))
        if tk_type == "NULL":
            return NullLiteral()
        if tk_type == "ID":
            if self._match("LPAREN"):
                args = self._parse_function_args()
                return self._simplify_navigation_zero_offset(tk_val, args)
            if self._match("DOT"):
                self._referenced_qualifiers.add(tk_val)
                return ColumnRef(tk_val, self._parse_qualified_column_name())
            return ColumnRef(None, tk_val)

        raise ValueError(f"Unexpected token: {tk_val}")

    def get_referenced_qualifiers(self) -> Set[str]:
        """Return qualifiers seen in the most recently parsed expression text.

        This is captured while parsing so semantic validation still sees a
        reference that expression simplification subsequently removes.
        """
        return set(self._referenced_qualifiers)

    def _parse_function_args(self) -> List[Expr]:
        args = []
        if self._peek_type("RPAREN"):
            self._consume("RPAREN")
            return args

        while True:
            if self._match("MATH_OP", "*"):
                args.append(ColumnRef(None, "*"))
            else:
                args.append(self._parse_or())
            if not self._match("COMMA"):
                break

        self._consume("RPAREN")
        return args

    def _simplify_navigation_zero_offset(
        self, func_name: str, args: List[Expr]
    ) -> Expr:
        normalized_name = func_name.upper()
        if (
            normalized_name in ("PREV", "NEXT")
            and args
            and any(args[0].has_func(inner) for inner in ("PREV", "NEXT"))
        ):
            raise ValueError(
                f"{normalized_name} cannot contain PREV or NEXT in its first argument"
            )
        if self._enforce_rpr_operator_rules:
            self._validate_rpr_operator_nesting(normalized_name, args)
        if normalized_name in ("PREV", "NEXT", "FIRST", "LAST") and len(args) >= 2:
            offset_value = self._evaluate_navigation_offset(args[1])
            if offset_value is None:
                raise ValueError(
                    f"{normalized_name} offset must be a non-negative integer constant expression"
                )
            # Keep offsets in a canonical literal form.  Besides making the
            # generated SQL smaller, rewriting rules can then safely consume
            # the offset without each having to implement constant folding.
            args[1] = Literal(offset_value)
            if normalized_name in ("PREV", "NEXT") and offset_value == 0:
                return args[0]
        return FunctionCall(func_name, args)

    def _validate_rpr_operator_nesting(self, func_name: str, args: List[Expr]) -> None:
        """Enforce Oracle's MATCH_RECOGNIZE operator-nesting restrictions."""
        pattern_operators = self._RPR_AGGREGATES + self._RPR_NAVIGATION
        if func_name not in pattern_operators or not args:
            return

        first_arg = args[0]
        nested = {
            operator for operator in pattern_operators if first_arg.has_func(operator)
        }
        nested_special = {
            function
            for function in ("MATCH_NUMBER", "CLASSIFIER")
            if first_arg.has_func(function)
        }

        # Oracle permits only the compound navigation forms
        # PREV/NEXT(FIRST/LAST(...)); all other nesting among aggregates and
        # row-pattern navigation operators is rejected with ORA-62506.
        allowed_navigation = (
            func_name in ("PREV", "NEXT") and nested and nested <= {"FIRST", "LAST"}
        )
        if nested and not allowed_navigation:
            raise ValueError(
                "Oracle MATCH_RECOGNIZE does not allow nested aggregates or "
                "navigation operators except PREV/NEXT(FIRST/LAST(...))"
            )

        # MATCH_NUMBER and CLASSIFIER cannot occur inside an aggregate or a
        # row-pattern navigation operator (ORA-62507).
        if nested_special:
            names = ", ".join(sorted(nested_special))
            raise ValueError(
                f"Oracle MATCH_RECOGNIZE does not allow {names} inside an "
                "aggregate or navigation operator"
            )

        qualifiers = self._collect_rpr_column_qualifiers(first_arg)
        if len(qualifiers) > 1:
            raise ValueError(
                "Oracle MATCH_RECOGNIZE aggregates and navigation operators "
                "cannot reference multiple pattern variables"
            )

    def _collect_rpr_column_qualifiers(self, expr: Expr) -> set:
        """Collect explicit qualifiers plus the implicit universal variable."""
        qualifiers = set()

        def visit(node: Expr) -> None:
            if isinstance(node, ColumnRef):
                qualifiers.add(
                    node.qualifier.upper() if node.qualifier else "<UNIVERSAL>"
                )
                return
            if isinstance(node, FunctionCall):
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
            if isinstance(node, CaseExpr):
                for condition, result in node.cases:
                    visit(condition)
                    visit(result)
                if node.else_expr is not None:
                    visit(node.else_expr)
                return
            if isinstance(node, InExpr):
                visit(node.expr)
                for value in node.values:
                    visit(value)
                return
            if isinstance(node, (IsNullExpr, NotExpr)):
                visit(node.expr)

        visit(expr)
        return qualifiers

    def _evaluate_navigation_offset(self, expr: Expr) -> Optional[int]:
        """Evaluate a pure numeric offset expression, rejecting dynamic values."""
        value = self._evaluate_numeric_constant(expr)
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, Decimal):
            if value != value.to_integral_value():
                return None
            value = int(value)
        elif isinstance(value, float):
            if not value.is_integer():
                return None
            value = int(value)
        if not isinstance(value, int) or value < 0:
            return None
        return value

    def _evaluate_numeric_constant(self, expr: Expr):
        if isinstance(expr, Literal):
            if isinstance(expr.value, (int, float, Decimal)) and not isinstance(
                expr.value, bool
            ):
                return expr.value
            return None

        if not isinstance(expr, BinaryExpr) or expr.op not in ("+", "-", "*", "/"):
            return None

        left = self._evaluate_numeric_constant(expr.left)
        right = self._evaluate_numeric_constant(expr.right)
        if left is None or right is None:
            return None

        try:
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if right == 0:
                return None
            return left / right
        except (ArithmeticError, TypeError, OverflowError):
            return None

    def _parse_qualified_column_name(self) -> str:
        if self._at_end():
            raise ValueError("Unexpected end after '.'")
        if self._peek_type("ID"):
            return self._consume("ID")
        if self._match("MATH_OP", "*"):
            return "*"
        raise ValueError(f"Expected ID or *, got {self._peek()[0]}")

    def _parse_in_list(self):
        self._consume("LPAREN")
        values = []
        if self._peek_type("ID", "SELECT"):
            depth = 1
            subquery_parts = []
            while not self._at_end():
                tk_type, tk_val = self._peek()
                if tk_type == "LPAREN":
                    depth += 1
                    subquery_parts.append((tk_type, tk_val))
                elif tk_type == "RPAREN":
                    depth -= 1
                    if depth == 0:
                        break
                    subquery_parts.append((tk_type, tk_val))
                else:
                    subquery_parts.append((tk_type, tk_val))
                self.pos += 1
            self._consume("RPAREN")
            values.append(SubqueryExpr(self._reassemble_subquery(subquery_parts)))
        elif not self._peek_type("RPAREN"):
            while True:
                values.append(self._parse_additive())
                if not self._match("COMMA"):
                    break
            self._consume("RPAREN")
        else:
            self._consume("RPAREN")
        return values

    def _reassemble_subquery(self, parts) -> str:
        """Join the tokens captured inside IN (... SELECT ...) back into SQL text.

        A dot token belongs to a qualified reference such as ``S.PT``, so it
        must not get padded with spaces on either side.
        """
        out = ""
        for i, (tk_type, tk_val) in enumerate(parts):
            if i > 0 and tk_type != "DOT" and parts[i - 1][0] != "DOT":
                out += " "
            out += tk_val
        return out

    def _parse_case_expression(self):
        cases = []
        else_expr = None
        while True:
            if self._match("CASE_KW", "WHEN"):
                cond = self._parse_or()
                self._consume("CASE_KW", "THEN")
                cases.append((cond, self._parse_or()))
            elif self._match("CASE_KW", "ELSE"):
                else_expr = self._parse_or()
                self._consume("CASE_KW", "END")
                break
            elif self._match("CASE_KW", "END"):
                break
            else:
                raise ValueError("Expected WHEN/ELSE/END")
        return CaseExpr(cases, else_expr)

    def _at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    def _peek(self) -> Tuple[str, str]:
        if self._at_end():
            raise ValueError("End of input")
        return self.tokens[self.pos]

    def _peek_type(
        self, expected_type: str, expected_val: Optional[str] = None
    ) -> bool:
        if self._at_end():
            return False
        tk_type, tk_val = self.tokens[self.pos]
        if tk_type != expected_type:
            return False
        return expected_val is None or tk_val.upper() == expected_val.upper()

    def _match(self, expected_type: str, expected_val: Optional[str] = None) -> bool:
        if not self._peek_type(expected_type, expected_val):
            return False
        self.pos += 1
        return True

    def _consume(self, expected_type: str, expected_val: Optional[str] = None) -> str:
        if self._at_end():
            raise ValueError("End of input")

        tk_type, tk_val = self._peek()
        match = tk_type == expected_type or (
            expected_type == "ID" and tk_type in self._ID_COMPATIBLE_TOKENS
        )
        if not match:
            raise ValueError(f"Expected {expected_type}, got {tk_type} {tk_val}")

        if expected_val and tk_val.upper() != expected_val.upper():
            raise ValueError(f"Expected {expected_val}, got {tk_val}")

        self.pos += 1
        return tk_val
