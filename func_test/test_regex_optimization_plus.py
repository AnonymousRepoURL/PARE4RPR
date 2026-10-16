"""SQL regression tests for candidates that preserve only preferred matches.

The differential oracle evaluates finite DEFINE predicates independently and
uses Python's ordered matcher.  It checks selected spans, emitted row positions,
and subsequent match starts, including NULL and rows accepted by no variable.
"""

import copy
import itertools
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401

from expression import (
    BetweenExpr,
    BinaryExpr,
    ColumnRef,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    NullLiteral,
    SemanticsExpr,
)
from patternnode import (
    AlternationPattern,
    AnchorPattern,
    ConcatenationPattern,
    EmptyPattern,
    ExcludedPattern,
    QuantifiedPattern,
    VariablePattern,
)
from priority_regex_optimizer import EquivalenceMode, ProofScope
from queryanalysis import RPRAbstractSyntaxTree
from rewriting_rules.priority_optimizer_service import (
    PRIORITY_OPTIMIZER_SERVICE,
    PriorityOptimizerOutcome,
)
from rewriting_rules.regex_optimization_plus import RegexOptimizationPlusRule
from rewriting_rules.regex_fragment_search import PatternLocation

SCHEMAS = {
    "T": {
        "columns": {"TS": "LONG", "VAL": "LONG"},
        "column_nullables": {"TS": False, "VAL": True},
    }
}
BRANCH_DEFINITIONS = "X AS VAL IN (3, 4, 5), C AS VAL = 3, D AS VAL = 4"


def _parse(
    pattern, definitions, *, measures="", subset="", skip="", rows="ALL ROWS PER MATCH"
):
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(
        f"SELECT * FROM T MATCH_RECOGNIZE(ORDER BY TS {measures} "
        f"{rows} {skip} PATTERN({pattern}) {subset} DEFINE {definitions})",
        table_schemas=SCHEMAS,
    )
    return ast


def _truth(expression, value):
    """Small independent SQL three-valued evaluator for finite test predicates."""
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, NullLiteral):
        return None
    if isinstance(expression, ColumnRef):
        if expression.name != "VAL":
            raise AssertionError(expression)
        return value
    if isinstance(expression, SemanticsExpr):
        return _truth(expression.expr, value)
    if isinstance(expression, NotExpr):
        result = _truth(expression.expr, value)
        return None if result is None else not result
    if isinstance(expression, IsNullExpr):
        result = _truth(expression.expr, value) is None
        return not result if expression.is_not else result
    if isinstance(expression, BinaryExpr):
        left = _truth(expression.left, value)
        right = _truth(expression.right, value)
        if expression.op == "AND":
            if left is False or right is False:
                return False
            return None if left is None or right is None else bool(left and right)
        if expression.op == "OR":
            if left is True or right is True:
                return True
            return None if left is None or right is None else bool(left or right)
        if left is None or right is None:
            return None
        return {
            "=": lambda: left == right,
            "!=": lambda: left != right,
            "<>": lambda: left != right,
            "<": lambda: left < right,
            "<=": lambda: left <= right,
            ">": lambda: left > right,
            ">=": lambda: left >= right,
        }[expression.op]()
    if isinstance(expression, InExpr):
        tested = _truth(expression.expr, value)
        values = [_truth(item, value) for item in expression.values]
        result = (
            None
            if tested is None
            else True if tested in values else None if None in values else False
        )
        return None if result is None else not result if expression.is_not else result
    if isinstance(expression, BetweenExpr):
        tested = _truth(expression.expr, value)
        lower = _truth(expression.lower, value)
        upper = _truth(expression.upper, value)
        return None if None in (tested, lower, upper) else lower <= tested <= upper
    raise AssertionError(f"unsupported test predicate: {expression}")


def _independent_regex(ast, domain):
    """Translate SQL patterns using actual finite rows, without optimizer encoding."""
    symbols = {value: chr(ord("a") + index) for index, value in enumerate(domain)}

    def convert(node):
        if isinstance(node, EmptyPattern):
            return ""
        if isinstance(node, VariablePattern):
            condition = ast.define.definitions.get(node.name)
            accepted = "".join(
                symbol
                for value, symbol in symbols.items()
                if condition is None or _truth(condition.expression, value) is True
            )
            return f"[{accepted}]" if accepted else "(?!)"
        if isinstance(node, ConcatenationPattern):
            return "".join(convert(child) for child in node.patterns)
        if isinstance(node, AlternationPattern):
            return "(?:" + "|".join(convert(child) for child in node.patterns) + ")"
        if isinstance(node, QuantifiedPattern):
            upper = "" if node.atMost is None else str(node.atMost)
            count = (
                f"{{{node.atLeast}}}"
                if node.atLeast == node.atMost
                else f"{{{node.atLeast},{upper}}}"
            )
            reluctant = "?" if not node.isGreedy and node.atLeast != node.atMost else ""
            return f"(?:{convert(node.pattern)}){count}{reluctant}"
        if isinstance(node, ExcludedPattern):
            return convert(node.pattern)
        if isinstance(node, AnchorPattern):
            anchor = node.anchor_type.value
            return (
                "^" + convert(node.pattern)
                if anchor == "^"
                else convert(node.pattern) + r"\Z"
            )
        raise AssertionError(node)

    return re.compile(convert(ast.pattern)), symbols


def _query_observations(regex, text, ast):
    """Observe match numbers/spans and emitted positions under supported skips."""
    matches = []
    emitted = []
    start = 0
    while start < len(text):
        match = regex.match(text, start)
        if match is None:
            start += 1
            continue
        end = match.end()
        matches.append((start, end))
        positions = (
            range(start, end) if ast.rows_per_match.startswith("ALL ROWS") else (start,)
        )
        emitted.extend((len(matches), position) for position in positions)
        start = (
            start + 1 if ast.after_match_skip_to == "NEXT ROW" or end == start else end
        )
    return matches, emitted


class MatchOnlySqlRewriteTests(unittest.TestCase):
    def assert_differential(
        self, original, rewritten, domain, max_length=4, extra_rows=()
    ):
        before, symbols = _independent_regex(original, domain)
        after, rewritten_symbols = _independent_regex(rewritten, domain)
        self.assertEqual(symbols, rewritten_symbols)
        rows = itertools.chain(
            (
                values
                for length in range(max_length + 1)
                for values in itertools.product(domain, repeat=length)
            ),
            extra_rows,
        )
        for values in rows:
            text = "".join(symbols[value] for value in values)
            for start in range(len(text) + 1):
                original_match = before.match(text, start)
                rewritten_match = after.match(text, start)
                self.assertEqual(
                    None if original_match is None else original_match.end(),
                    None if rewritten_match is None else rewritten_match.end(),
                    (values, start, original.to_sql(), rewritten.to_sql()),
                )
            self.assertEqual(
                _query_observations(before, text, original),
                _query_observations(after, text, rewritten),
                values,
            )

    def test_root_discards_unobservable_lower_priority_branch(self):
        ast = _parse("X | C D", BRANCH_DEFINITIONS)
        original = copy.deepcopy(ast)
        self.assertTrue(RegexOptimizationPlusRule().apply(ast))
        self.assertEqual("X", ast.pattern.to_string())
        self.assertEqual({"X"}, set(ast.define.definitions))
        self.assert_differential(original, ast, (None, 0, 3, 4, 5, 9))
        # The emitted SQL must still parse with consistent DEFINE metadata.
        reparsed = RPRAbstractSyntaxTree()
        reparsed.parse_sql(ast.to_sql(), table_schemas=SCHEMAS)
        self.assertEqual(ast.to_sql(), reparsed.to_sql())

    def test_terminal_fragment_uses_complete_joint_alphabet(self):
        ast = _parse("A B (X | C D)", "A AS VAL=1, B AS VAL=2, " + BRANCH_DEFINITIONS)
        original = copy.deepcopy(ast)
        with patch.object(
            PRIORITY_OPTIMIZER_SERVICE,
            "verify_equivalent",
            wraps=PRIORITY_OPTIMIZER_SERVICE.verify_equivalent,
        ) as verifier:
            self.assertTrue(RegexOptimizationPlusRule().apply(ast))
        self.assertEqual("A B X", ast.pattern.to_string())
        self.assertTrue(verifier.called)
        self.assertTrue(
            all(
                call.args[2].equivalence_mode is EquivalenceMode.MATCH_ONLY
                for call in verifier.call_args_list
            )
        )
        self.assert_differential(original, ast, (None, 1, 2, 3, 4, 5, 9))

    def test_next_row_skip_preserves_overlapping_matches(self):
        ast = _parse(
            "(X | C D) X",
            BRANCH_DEFINITIONS,
            skip="AFTER MATCH SKIP TO NEXT ROW",
        )
        original = copy.deepcopy(ast)
        self.assertTrue(RegexOptimizationPlusRule().apply(ast))
        self.assert_differential(original, ast, (None, 0, 3, 4, 5, 9))

    def test_suffix_that_needs_fallback_branch_is_preserved(self):
        ast = _parse(
            "A B (X | C D) E",
            "A AS VAL=1, B AS VAL=2, " + BRANCH_DEFINITIONS + ", E AS VAL=5",
        )
        original = copy.deepcopy(ast)
        RegexOptimizationPlusRule().apply(ast)
        self.assert_differential(
            original,
            ast,
            (None, 1, 2, 3, 4, 5, 9),
            max_length=3,
            extra_rows=((1, 2, 3, 4, 5),),
        )
        result, symbols = _independent_regex(ast, (1, 2, 3, 4, 5))
        self.assertEqual(
            5, result.match("".join(symbols[value] for value in (1, 2, 3, 4, 5))).end()
        )

    def test_enclosing_repetition_keeps_required_fallback(self):
        ast = _parse("(A | A B){2}", "A AS VAL=1, B AS VAL=2")
        original = copy.deepcopy(ast)
        RegexOptimizationPlusRule().apply(ast)
        self.assert_differential(original, ast, (None, 0, 1, 2, 9))
        result, symbols = _independent_regex(ast, (1, 2))
        self.assertEqual(3, result.match(symbols[1] + symbols[2] + symbols[1]).end())

    def test_root_requires_unobservable_labels_and_simple_skip(self):
        guarded = [
            {"subset": "SUBSET U=(X,C,D)"},
            {"measures": "MEASURES LAST(X.VAL) AS X_VAL"},
            {"measures": "MEASURES CLASSIFIER() AS LABEL"},
            {"skip": "AFTER MATCH SKIP TO LAST X"},
        ]
        for clauses in guarded:
            with self.subTest(clauses=clauses):
                ast = _parse("X | C D", BRANCH_DEFINITIONS, **clauses)
                before = ast.to_sql()
                self.assertFalse(RegexOptimizationPlusRule().apply(ast))
                self.assertEqual(before, ast.to_sql())

    def test_root_span_certificate_cannot_bypass_subset_observations(self):
        ast = _parse("X | C D", BRANCH_DEFINITIONS, subset="SUBSET U=(X,C,D)")
        # Even a supplied standalone span certificate is insufficient to
        # authorize changing labels exposed by SQL-level observations.
        plan = SimpleNamespace(
            proof_scope=ProofScope.STANDALONE, location=PatternLocation("root")
        )
        self.assertFalse(
            RegexOptimizationPlusRule()._certify_priority_plan(ast, plan, None)
        )
        self.assertEqual(ProofScope.STANDALONE, plan.proof_scope)

    def test_end_anchor_and_all_rows_exclusion_block_branch_deletion(self):
        for pattern in ("(X | C D)$", "{- (X | C D) -}"):
            with self.subTest(pattern=pattern):
                ast = _parse(pattern, BRANCH_DEFINITIONS)
                before = ast.to_sql()
                self.assertFalse(RegexOptimizationPlusRule().apply(ast))
                self.assertEqual(before, ast.to_sql())

    def test_unknown_and_timeout_leave_all_transaction_fields_unchanged(self):
        for effect in (None, TimeoutError("optimizer timed out")):
            with self.subTest(effect=effect):
                ast = _parse("X | C D", BRANCH_DEFINITIONS)
                before = copy.deepcopy(
                    (
                        ast.pattern,
                        ast.define.definitions,
                        ast.subsets,
                        ast.pattern_rewrite_boundaries,
                    )
                )
                parameters = (
                    {"return_value": [None]}
                    if effect is None
                    else {"side_effect": effect}
                )
                with patch.object(
                    PRIORITY_OPTIMIZER_SERVICE, "optimize_many", **parameters
                ):
                    self.assertFalse(RegexOptimizationPlusRule().apply(ast))
                self.assertEqual(
                    before,
                    (
                        ast.pattern,
                        ast.define.definitions,
                        ast.subsets,
                        ast.pattern_rewrite_boundaries,
                    ),
                )

    def test_failed_complete_context_proof_does_not_commit(self):
        ast = _parse("A B (X | C D)", "A AS VAL=1, B AS VAL=2, " + BRANCH_DEFINITIONS)
        before = ast.to_sql()
        with patch.object(
            PRIORITY_OPTIMIZER_SERVICE, "verify_equivalent", return_value=False
        ) as verifier:
            self.assertFalse(RegexOptimizationPlusRule().apply(ast))
        self.assertTrue(verifier.called)
        self.assertEqual(before, ast.to_sql())

    def test_commit_exception_rolls_back_pattern_define_subset_and_boundaries(self):
        ast = _parse("X | C D", BRANCH_DEFINITIONS)
        before = copy.deepcopy(
            (
                ast.pattern,
                ast.define.definitions,
                ast.subsets,
                ast.pattern_rewrite_boundaries,
            )
        )
        rule = RegexOptimizationPlusRule()
        with patch.object(
            rule,
            "_replace_definitions",
            side_effect=RuntimeError("injected commit failure"),
        ):
            self.assertFalse(rule.apply(ast))
        self.assertEqual(
            before,
            (
                ast.pattern,
                ast.define.definitions,
                ast.subsets,
                ast.pattern_rewrite_boundaries,
            ),
        )

    def test_unknown_prefix_function_is_not_treated_as_stable(self):
        ast = _parse(
            "P (A+ | B+)+",
            "P AS VAL=0, A AS VAL=1 AND VOLATILE_UDF(FIRST(P.VAL))>0, "
            "B AS VAL IN(1,2) AND VOLATILE_UDF(FIRST(P.VAL))>0",
        )
        before = ast.to_sql()
        self.assertFalse(RegexOptimizationPlusRule().apply(ast))
        self.assertEqual(before, ast.to_sql())

    def test_empty_worker_output_does_not_erase_all_define_variables(self):
        ast = _parse("X | C D", BRANCH_DEFINITIONS)
        before = ast.to_sql()
        with patch.object(
            PRIORITY_OPTIMIZER_SERVICE,
            "optimize_many",
            return_value=[PriorityOptimizerOutcome("", ProofScope.STANDALONE)],
        ):
            self.assertFalse(RegexOptimizationPlusRule().apply(ast))
        self.assertEqual(before, ast.to_sql())


if __name__ == "__main__":
    unittest.main()
