"""Check SQL truth values independently of the expression simplifier."""

import itertools
import operator
import random
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from clause_parser import ExprParser
from expression import (
    BetweenExpr,
    BinaryExpr,
    CaseExpr,
    ColumnRef,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    NullLiteral,
    SemanticsExpr,
)
from expression_simplifier import (
    ExpressionColumnMetadata,
    SimplifyContext,
    _condition_is_not_true,
)
from queryanalysis import RPRAbstractSyntaxTree


def sql_not(value):
    return None if value is None else not value


def sql_and(left, right):
    if left is False or right is False:
        return False
    return None if left is None or right is None else True


def sql_or(left, right):
    if left is True or right is True:
        return True
    return None if left is None or right is None else False


def evaluate(expr, row):
    if isinstance(expr, NullLiteral):
        return None
    if isinstance(expr, Literal):
        return None if expr.value == "" else expr.value
    if isinstance(expr, ColumnRef):
        return row[expr.name.upper()]
    if isinstance(expr, SemanticsExpr):
        return evaluate(expr.expr, row)
    if isinstance(expr, NotExpr):
        return sql_not(evaluate(expr.expr, row))
    if isinstance(expr, IsNullExpr):
        result = evaluate(expr.expr, row) is None
        return not result if expr.is_not else result
    if isinstance(expr, BinaryExpr):
        left, right = evaluate(expr.left, row), evaluate(expr.right, row)
        if expr.op == "AND":
            return sql_and(left, right)
        if expr.op == "OR":
            return sql_or(left, right)
        if left is None or right is None:
            return None
        return {
            "=": operator.eq,
            "!=": operator.ne,
            "<>": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }[expr.op](left, right)
    if isinstance(expr, BetweenExpr):
        return sql_and(
            evaluate(BinaryExpr(expr.expr, ">=", expr.lower), row),
            evaluate(BinaryExpr(expr.expr, "<=", expr.upper), row),
        )
    if isinstance(expr, InExpr):
        result = False
        for value in expr.values:
            result = sql_or(result, evaluate(BinaryExpr(expr.expr, "=", value), row))
        return sql_not(result) if expr.is_not else result
    if isinstance(expr, CaseExpr):
        for condition, result in expr.cases:
            if evaluate(condition, row) is True:
                return evaluate(result, row)
        return evaluate(expr.else_expr, row) if expr.else_expr is not None else None
    if isinstance(expr, FunctionCall) and expr.name.upper() == "LNNVL":
        return evaluate(expr.args[0], row) is not True
    raise AssertionError(type(expr))


def context(nullable=True, data_type="DOUBLE"):
    return SimplifyContext(
        lambda column: ExpressionColumnMetadata(
            "VARCHAR(8)" if column.name.upper() == "TXT" else data_type, nullable
        )
    )


def unsimplified(sql):
    # Reuse the grammar, but construct the original AST without any reduction.
    def binary(left, op, right, *args):
        return BinaryExpr(left, op, right)

    with (
        patch("expression_parser._simplify_logical_expr", side_effect=binary),
        patch("expression_parser._simplify_comparison_expr", side_effect=binary),
        patch("expression_parser._negate_predicate", side_effect=NotExpr),
    ):
        return ExprParser().parse_expression(sql)


class ThreeValuedSimplificationTests(unittest.TestCase):
    def check_expression(self, sql, *, nullable=True, data_type="DOUBLE"):
        original = unsimplified(sql)
        parser = ExprParser(context(nullable, data_type))
        scalar = parser.parse_expression(sql)
        define = parser.parse_define("DEFINE A AS " + sql).definitions["A"].expression
        where = parser.parse_where("WHERE " + sql)
        domain = [-1, 0, 1, 2] + ([] if nullable is False else [None])
        for value, other in itertools.product(domain, repeat=2):
            for text in ("A", "B", None) if nullable is not False else ("A", "B"):
                row = {"VAL": value, "OTHER": other, "TXT": text}
                expected = evaluate(original, row)
                self.assertIs(
                    evaluate(scalar, row), expected, (sql, row, scalar.to_string())
                )
                for rewritten in (define, where):
                    self.assertEqual(
                        evaluate(rewritten, row) is True,
                        expected is True,
                        (sql, row, rewritten.to_string()),
                    )

    def test_negated_ranges_and_complements(self):
        for nullable, data_type in itertools.product(
            (True, False, None), ("INT", "DOUBLE")
        ):
            for sql in (
                "NOT(VAL > 0 AND VAL < 0)",
                "NOT(VAL > 0 AND VAL <= 0)",
                "NOT(VAL > 0 OR VAL <= 0)",
                "NOT(NOT(VAL > 0 OR VAL <= 0))",
                "NOT(VAL > 0 AND VAL < 1)",
                "NOT((VAL > 0 AND OTHER > 0) OR (VAL > 0 AND NOT(OTHER > 0)))",
                "NOT((VAL > 0 OR OTHER > 0) AND (VAL > 0 OR NOT(OTHER > 0)))",
                "NOT(TXT = 'A' AND TXT != 'A')",
                "NOT(TXT = 'A' OR TXT != 'A')",
                "OTHER > 0 OR VAL > 0 OR VAL <= 0",
                "NOT(OTHER > 0 OR VAL > 0 OR VAL <= 0)",
            ):
                with self.subTest(sql=sql, nullable=nullable, data_type=data_type):
                    self.check_expression(sql, nullable=nullable, data_type=data_type)

    def test_case_comparisons_preserve_unknown_until_filter_boundary(self):
        for sql in (
            "NOT(CASE WHEN VAL > 0 THEN 1 ELSE 0 END = 1)",
            "NOT(CASE WHEN VAL > 0 THEN 1 END = 1)",
            "NOT(CASE WHEN VAL > 0 THEN OTHER ELSE 0 END > 0)",
            "NOT(CASE WHEN VAL > 0 THEN 0 WHEN OTHER > 0 THEN 1 ELSE NULL END = 1)",
            "CASE WHEN NOT(VAL > 0 AND VAL < 0) THEN 1 ELSE 0 END = 1",
            "CASE WHEN NOT(VAL > 0 OR VAL <= 0) THEN 1 ELSE 0 END = 1",
            "CASE WHEN (CASE WHEN VAL > 0 THEN 1 END = 1) THEN 0 ELSE 1 END = 1",
            "CASE WHEN VAL IN (1, NULL) THEN 0 ELSE 1 END = 1",
            "CASE WHEN VAL NOT IN (1, NULL) THEN 0 ELSE 1 END = 1",
        ):
            with self.subTest(sql=sql):
                self.check_expression(sql)

    def test_filter_optimizations_are_retained(self):
        for nullable, expected in (
            (True, "VAL IS NOT NULL"),
            (None, "VAL IS NOT NULL"),
            (False, "1 = 1"),
        ):
            parser = ExprParser(context(nullable))
            result = parser.parse_define("DEFINE A AS NOT(VAL > 0 AND VAL < 0)")
            self.assertEqual(result.to_string(), "DEFINE A AS " + expected)
        parser = ExprParser(context())
        self.assertEqual(
            parser.parse_define("DEFINE A AS VAL > 0 AND VAL < 0").to_string(),
            "DEFINE A AS 1 != 1",
        )
        self.assertEqual(
            parser.parse_define("DEFINE A AS VAL > 0 OR VAL <= 0").to_string(),
            "DEFINE A AS VAL IS NOT NULL",
        )

    def test_select_and_measures_keep_nested_truth_values(self):
        sql = "CASE WHEN NOT(VAL > 0 AND VAL < 0) THEN 1 ELSE 0 END"
        parser = ExprParser(context())
        original = unsimplified(sql)
        selected = parser.parse_select("SELECT " + sql + " AS X")
        measured = parser.parse_measures("MEASURES " + sql + " AS X")
        # Both clause parsers must use scalar, rather than filter, semantics.
        for expression in (selected.select_items[0][0], measured.measures["X"]):
            for value in (None, -1, 0, 1):
                self.assertEqual(
                    evaluate(expression, {"VAL": value}),
                    evaluate(original, {"VAL": value}),
                )

    def test_boolean_combinations_against_independent_truth_tables(self):
        atoms = ("VAL > 0", "VAL <= 0", "VAL IS NULL", "OTHER > 0", "1 = 1", "1 != 1")
        for left, right, op in itertools.product(atoms, atoms, ("AND", "OR")):
            base = f"({left}) {op} ({right})"
            for sql in (
                base,
                f"NOT({base})",
                f"NOT(({base}) OR (VAL > 0 AND VAL < 0))",
            ):
                self.check_expression(sql)

    def test_not_true_of_nullable_in_list(self):
        for sql in ("VAL IN (1, NULL)", "VAL NOT IN (1, NULL)"):
            original = unsimplified(sql)
            result = _condition_is_not_true(original, context())
            for value in (None, 0, 1):
                row = {"VAL": value}
                self.assertIs(
                    evaluate(result, row), evaluate(original, row) is not True
                )

    def test_nested_boolean_and_case_combinations(self):
        rng = random.Random(710)
        atoms = (
            "VAL > 0",
            "VAL <= 0",
            "VAL < 0",
            "OTHER > 0",
            "VAL IS NULL",
            "VAL IN (1, NULL)",
        )

        def predicate(depth):
            if depth == 0 or rng.randrange(4) == 0:
                return rng.choice(atoms)
            kind = rng.randrange(4)
            if kind == 0:
                return f"NOT({predicate(depth - 1)})"
            if kind == 1:
                return (
                    f"CASE WHEN {predicate(depth - 1)} "
                    f"THEN {rng.choice(('VAL', '1', 'NULL'))} "
                    f"ELSE {rng.choice(('OTHER', '0', 'NULL'))} END > 0"
                )
            return (
                f"({predicate(depth - 1)}) {rng.choice(('AND', 'OR'))} "
                f"({predicate(depth - 1)})"
            )

        for _ in range(200):
            sql = predicate(3)
            self.check_expression(sql)

    def test_query_parser_keeps_null_rows_out(self):
        schema = {
            "T": {
                "columns": {"TS": "LONG", "VAL": "DOUBLE"},
                "column_nullables": {"TS": False, "VAL": True},
            }
        }
        ast = RPRAbstractSyntaxTree()
        ast.parse_sql(
            "SELECT * FROM T MATCH_RECOGNIZE(ORDER BY TS ALL ROWS PER MATCH "
            "PATTERN(A) DEFINE A AS NOT(VAL > 0 AND VAL < 0))",
            table_schemas=schema,
        )
        expression = ast.define.definitions["A"].expression
        self.assertIsNot(evaluate(expression, {"VAL": None}), True)
        self.assertIs(evaluate(expression, {"VAL": 0}), True)
        reparsed = RPRAbstractSyntaxTree()
        reparsed.parse_sql(ast.to_sql(), table_schemas=schema)
        self.assertEqual(ast.to_sql(), reparsed.to_sql())


if __name__ == "__main__":
    unittest.main()
