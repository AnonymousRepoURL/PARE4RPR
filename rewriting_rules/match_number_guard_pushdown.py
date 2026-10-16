"""Push match-number-only guards to the beginning of a row pattern.

``MATCH_NUMBER()`` is constant while Oracle searches for the next match in a
partition.  Consequently, a guard on a variable that occurs in every match
can be evaluated on every possible leftmost variable instead.  This is most
useful for an upper bound placed on a late variable: once the bound becomes
false, candidate matches fail on their first row instead of after the engine
has explored the rest of the pattern.
"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Iterable, List, Set

from clause_parser import ExprParser
from expression import (
    BetweenExpr,
    BinaryExpr,
    Expr,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    SemanticsExpr,
)
from patternnode import (
    AlternationPattern,
    AnchorPattern,
    ConcatenationPattern,
    ExcludedPattern,
    PermutationPattern,
    QuantifiedPattern,
    VariablePattern,
)
from queryanalysis import RPRAbstractSyntaxTree

from .base import RewriteRule
from .utils import _join_conjuncts, _split_conjuncts


class MatchNumberGuardPushdownRule(RewriteRule):
    """Move a pure ``MATCH_NUMBER()`` conjunct to all pattern entry variables.

    The source variable must be in ``pattern.get_must_appear_var()``.  This is
    the proof that its guard is a condition on every successful match rather
    than on one optional branch.  Only simple, side-effect-free comparisons
    against numeric literals are accepted; expressions involving columns,
    UDFs, arithmetic, or subqueries are deliberately left unchanged.

    Example::

        PATTERN((A | B){1,20} D)
        DEFINE A AS A.X > 0,
               B AS B.X > 0,
               D AS D.Y = 1 AND MATCH_NUMBER() < 200

    becomes::

        DEFINE A AS A.X > 0 AND MATCH_NUMBER() < 200,
               B AS B.X > 0 AND MATCH_NUMBER() < 200,
               D AS D.Y = 1
    """

    @property
    def name(self) -> str:
        return "match_number_guard_pushdown"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        if plan is None:
            return False

        parser = ExprParser()
        selected_keys: List[str] = plan["selected_keys"]
        removals: Dict[str, Set[int]] = plan["removals"]

        # First remove the late copies.  A guard-only DEFINE entry can be
        # deleted because an absent DEFINE condition is implicit TRUE.
        for var_name, indexes in removals.items():
            condition = ast.define.definitions.get(var_name)
            if condition is None:
                continue
            terms = _split_conjuncts(condition.expression)
            remaining = [
                term for index, term in enumerate(terms) if index not in indexes
            ]
            if remaining:
                ast.define.definitions[var_name] = parser._analyze_dependency(
                    var_name, _join_conjuncts(remaining)
                )
            else:
                del ast.define.definitions[var_name]

        # MATCH_NUMBER is the same for the entire candidate match.  Applying
        # every selected guard to every possible first variable therefore
        # preserves the accepted matches, including optional-prefix patterns.
        order = self._pattern_variable_order(ast.pattern)
        leftmost_vars = sorted(
            plan["leftmost_vars"], key=lambda name: order.get(name, len(order))
        )
        guards: Dict[str, Expr] = plan["guards"]
        for var_name in leftmost_vars:
            condition = ast.define.definitions.get(var_name)
            terms = (
                _split_conjuncts(condition.expression) if condition is not None else []
            )
            present = {
                self._guard_key(term)
                for term in terms
                if self._is_supported_guard(term)
            }
            for key in selected_keys:
                if key not in present:
                    terms.append(copy.deepcopy(guards[key]))
                    present.add(key)

            analyzed = parser._analyze_dependency(var_name, _join_conjuncts(terms))
            if condition is not None:
                ast.define.definitions[var_name] = analyzed
            else:
                self._insert_definition_by_pattern_order(ast, var_name, analyzed, order)

        return True

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        if ast.pattern is None or ast.define is None:
            return None

        # There is no input row on which to evaluate a pushed guard when the
        # complete pattern can be empty.
        if ast.pattern.get_min_row_num() == 0:
            return None

        leftmost_vars = ast.pattern.get_leftmost_variables()
        must_appear_vars = ast.pattern.get_must_appear_var()
        if not leftmost_vars or not must_appear_vars:
            return None

        guards: Dict[str, Expr] = {}
        occurrences = defaultdict(list)
        existing_by_var = defaultdict(set)

        for var_name, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            for index, term in enumerate(_split_conjuncts(condition.expression)):
                if not self._is_supported_guard(term):
                    continue
                key = self._guard_key(term)
                guards.setdefault(key, term)
                existing_by_var[var_name].add(key)
                if var_name in must_appear_vars:
                    occurrences[key].append((var_name, index))

        selected_keys: List[str] = []
        removals = defaultdict(set)
        for key, locations in occurrences.items():
            has_late_copy = any(
                var_name not in leftmost_vars for var_name, _ in locations
            )
            missing_at_entry = any(
                key not in existing_by_var[var_name] for var_name in leftmost_vars
            )
            if not has_late_copy and not missing_at_entry:
                continue

            selected_keys.append(key)
            for var_name, index in locations:
                removals[var_name].add(index)

        if not selected_keys:
            return None

        return {
            "selected_keys": selected_keys,
            "removals": dict(removals),
            "guards": guards,
            "leftmost_vars": leftmost_vars,
        }

    @staticmethod
    def _is_match_number_call(expr: Expr) -> bool:
        return (
            isinstance(expr, FunctionCall)
            and expr.name.upper() == "MATCH_NUMBER"
            and not expr.args
        )

    @staticmethod
    def _is_numeric_literal(expr: Expr) -> bool:
        return isinstance(expr, Literal) and isinstance(
            expr.value, (int, float, Decimal)
        )

    @classmethod
    def _is_supported_guard(cls, expr: Expr) -> bool:
        """Recognize guards that are safe to evaluate earlier.

        Keeping this grammar narrow also avoids changing when an exception
        from implicit conversion, arithmetic, or a user-defined function is
        raised.
        """
        if isinstance(expr, SemanticsExpr):
            return expr.semantics in (None, "RUNNING") and cls._is_supported_guard(
                expr.expr
            )
        if isinstance(expr, NotExpr):
            return cls._is_supported_guard(expr.expr)
        if isinstance(expr, IsNullExpr):
            return cls._is_match_number_call(expr.expr)
        if isinstance(expr, BetweenExpr):
            return (
                cls._is_match_number_call(expr.expr)
                and cls._is_numeric_literal(expr.lower)
                and cls._is_numeric_literal(expr.upper)
            )
        if isinstance(expr, InExpr):
            return cls._is_match_number_call(expr.expr) and all(
                cls._is_numeric_literal(value) for value in expr.values
            )
        if isinstance(expr, BinaryExpr) and expr.op in {
            "=",
            "!=",
            "<>",
            "<",
            "<=",
            ">",
            ">=",
        }:
            return (
                cls._is_match_number_call(expr.left)
                and cls._is_numeric_literal(expr.right)
            ) or (
                cls._is_numeric_literal(expr.left)
                and cls._is_match_number_call(expr.right)
            )
        return False

    @staticmethod
    def _guard_key(expr: Expr) -> str:
        return expr.to_string().upper()

    @classmethod
    def _iter_pattern_variables(cls, pattern) -> Iterable[str]:
        if isinstance(pattern, VariablePattern):
            yield pattern.name
        elif isinstance(pattern, (QuantifiedPattern, ExcludedPattern, AnchorPattern)):
            yield from cls._iter_pattern_variables(pattern.pattern)
        elif isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            for child in pattern.patterns:
                yield from cls._iter_pattern_variables(child)

    @classmethod
    def _pattern_variable_order(cls, pattern) -> Dict[str, int]:
        order: Dict[str, int] = {}
        for var_name in cls._iter_pattern_variables(pattern):
            if var_name not in order:
                order[var_name] = len(order)
        return order

    @staticmethod
    def _insert_definition_by_pattern_order(
        ast: RPRAbstractSyntaxTree,
        new_var: str,
        new_condition,
        order: Dict[str, int],
    ) -> None:
        new_position = order.get(new_var, len(order))
        definitions = {}
        inserted = False
        for var_name, condition in ast.define.definitions.items():
            if not inserted and order.get(var_name, len(order)) > new_position:
                definitions[new_var] = new_condition
                inserted = True
            definitions[var_name] = condition
        if not inserted:
            definitions[new_var] = new_condition
        ast.define.definitions = definitions


match_number_guard_pushdown_test_cases = [
    (
        1,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
  PARTITION BY SYMBOL
  ORDER BY TS, TRADE_ID
  MEASURES FIRST(TS) AS START_TS, D.TS AS END_TS, COUNT(*) AS ADJACENT_ROWS
  ONE ROW PER MATCH
  AFTER MATCH SKIP TO NEXT ROW
  PATTERN((A | B){1,20} (B | C){1,20} D)
  DEFINE
    A AS A.PRICE <= 200,
    B AS B.PRICE BETWEEN 150 AND 250,
    C AS C.PRICE >= 200,
    D AS D.VOLUME >= 10000 AND MATCH_NUMBER() < 200
)""",
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
  PARTITION BY SYMBOL
  ORDER BY TS, TRADE_ID
  MEASURES FIRST(TS) AS START_TS, D.TS AS END_TS, COUNT(*) AS ADJACENT_ROWS
  ONE ROW PER MATCH
  AFTER MATCH SKIP TO NEXT ROW
  PATTERN((A | B){1,20} (B | C){1,20} D)
  DEFINE
    A AS A.PRICE <= 200 AND MATCH_NUMBER() < 200,
    B AS B.PRICE BETWEEN 150 AND 250 AND MATCH_NUMBER() < 200,
    C AS C.PRICE >= 200,
    D AS D.VOLUME >= 10000
)""",
    ),
    # Every possible entry variable must receive the guard when the prefix is
    # optional.
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A? B D)
  DEFINE A AS A.X = 1, B AS B.X = 2,
         D AS D.X = 3 AND MATCH_NUMBER() <= 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A? B D)
  DEFINE A AS A.X = 1 AND MATCH_NUMBER() <= 10,
         B AS B.X = 2 AND MATCH_NUMBER() <= 10,
         D AS D.X = 3
)""",
    ),
    # A missing DEFINE entry is created in pattern order.
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE D AS D.X = 3 AND MATCH_NUMBER() < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE A AS MATCH_NUMBER() < 5, D AS D.X = 3
)""",
    ),
    # D is optional, so its predicate is not a global match guard.
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D?)
  DEFINE A AS A.X = 1, D AS D.X = 3 AND MATCH_NUMBER() < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D?)
  DEFINE A AS A.X = 1, D AS D.X = 3 AND MATCH_NUMBER() < 5
)""",
    ),
    # A mixed OR predicate is not a match-wide conjunct and must stay put.
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE A AS A.X = 1,
         D AS D.X = 3 AND (MATCH_NUMBER() < 5 OR D.Y = 9)
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE A AS A.X = 1,
         D AS D.X = 3 AND (MATCH_NUMBER() < 5 OR D.Y = 9)
)""",
    ),
    # A guard already on the sole entry variable cannot be moved earlier.
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE A AS A.X = 1 AND MATCH_NUMBER() < 5, D AS D.X = 3
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(A D)
  DEFINE A AS A.X = 1 AND MATCH_NUMBER() < 5, D AS D.X = 3
)""",
    ),
    # D starts one branch and ends another.  Its guard must also gate B, but
    # remains on D because D is itself a possible entry variable.
    (
        7,
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(D A | B D)
  DEFINE A AS A.X = 1, B AS B.X = 2,
         D AS D.X = 3 AND MATCH_NUMBER() < 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
  ORDER BY TS
  ALL ROWS PER MATCH
  PATTERN(D A | B D)
  DEFINE A AS A.X = 1,
         B AS B.X = 2 AND MATCH_NUMBER() < 5,
         D AS D.X = 3 AND MATCH_NUMBER() < 5
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        MatchNumberGuardPushdownRule,
        match_number_guard_pushdown_test_cases,
        "MATCH_NUMBER guard pushdown",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
