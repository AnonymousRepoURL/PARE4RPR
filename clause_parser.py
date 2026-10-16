"""Parsers for SELECT, WHERE, MEASURES, and DEFINE fragments."""

from typing import Optional, Tuple

from expression import ColumnRef, Expr, SemanticsExpr

from expression_clause import (
    DefineSubClause,
    collect_define_dependency_reasons,
    DependentCondition,
    IndependentCondition,
    MeasuresSubClause,
    SelectClause,
)

from expression_parser import ExpressionParser
from expression_simplifier import simplify_filter_predicate


class ExprParser(ExpressionParser):
    """Clause-level parser built on top of the expression parser."""

    _ORACLE_MEASURES_ONLY_FUNCTIONS = ("CLASSIFIER",)

    def parse_define(self, define_sql: str) -> DefineSubClause:
        self._reset(define_sql, "DEFINE")
        definitions = {}
        for var_name, expr_ast in self._parse_comma_separated(self._parse_define_item):
            invalid_functions = [
                func_name
                for func_name in self._ORACLE_MEASURES_ONLY_FUNCTIONS
                if expr_ast.has_func(func_name)
            ]
            if invalid_functions:
                function_list = ", ".join(f"{name}()" for name in invalid_functions)
                raise ValueError(
                    f"Oracle: {function_list} can only be used in the MEASURES "
                    f"clause, not in DEFINE for {var_name}"
                )
            definitions[var_name] = self._analyze_dependency(var_name, expr_ast)
        return DefineSubClause(definitions)

    def parse_where(self, where_sql: str) -> Expr:
        self._reset(where_sql, "WHERE")
        return simplify_filter_predicate(self._parse_or(), self.simplify_context)

    def parse_measures(self, measures_sql: str) -> MeasuresSubClause:
        self._reset(measures_sql, "MEASURES")
        return MeasuresSubClause(
            dict(self._parse_comma_separated(self._parse_measure_item))
        )

    def parse_select(self, select_sql: str) -> SelectClause:
        self._reset(select_sql, "SELECT")
        if self._match("MATH_OP", "*"):
            return SelectClause([(ColumnRef(None, "*"), None)])
        return SelectClause(self._parse_comma_separated(self._parse_select_item))

    def _parse_define_item(self) -> Tuple[str, Expr]:
        var_name = self._consume("ID")
        self._consume("KEYWORD", "AS")
        expression = simplify_filter_predicate(self._parse_or(), self.simplify_context)
        return var_name, expression

    def _parse_measure_item(self) -> Tuple[str, Expr]:
        semantics = self._consume("SEMANTICS") if self._peek_type("SEMANTICS") else None
        expr_ast = self._parse_or()
        if semantics is not None and semantics.upper() == "FINAL":
            expr_ast = SemanticsExpr(semantics.upper(), expr_ast)

        self._consume("KEYWORD", "AS")
        return self._consume("ID"), expr_ast

    def _parse_select_item(self) -> Tuple[Expr, Optional[str]]:
        expr_ast = self._parse_or()
        alias = None
        if self._match("KEYWORD", "AS"):
            alias = self._consume("ID")
        elif self._peek_type("ID"):
            alias = self._consume("ID")
        return expr_ast, alias

    def _analyze_dependency(self, current_var, ast_root):
        """Split DEFINE conditions by whether they require multiple rows or variables."""
        reasons = collect_define_dependency_reasons(current_var, ast_root)
        if reasons:
            return DependentCondition(current_var, ast_root, sorted(reasons))
        return IndependentCondition(current_var, ast_root)
