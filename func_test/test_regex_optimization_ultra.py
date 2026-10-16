"""Independent SQL/backtracking oracle and regressions for Ultra.

Run: .venv/bin/python -m unittest discover -s func_test -p '*_ultra.py' -v
No database service is required; these tests do not claim an Oracle integration
run. The oracle enumerates ordered AST paths, independent of the proof VM.
"""

import copy
import itertools
import random
from time import monotonic
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
from patternnode import (
    AlternationPattern,
    ConcatenationPattern,
    EmptyPattern,
    PatternLexer,
    PatternParser,
    QuantifiedPattern,
    VariablePattern,
    AnchorPattern,
    AnchorType,
    ExcludedPattern,
)
from queryanalysis import RPRAbstractSyntaxTree
from rewriting_rules.base import RewriteEngine
from rewriting_rules.predicates_ultra import (
    NullabilityUltra,
    PredicateAlgebraUltra,
    normalize,
)
from rewriting_rules.regex_automata_ultra import BudgetUltra, verify_labels
from rewriting_rules.regex_candidates_ultra import arrangements
from rewriting_rules.regex_optimization_ultra import (
    RegexOptimizationUltraRule,
    UltraConfig,
)

SCHEMA = {
    "GAS_PRICE": {
        "columns": {
            "STATION": "LONG",
            "TSTAMP": "LONG",
            "DIESEL": "DOUBLE",
            "E5": "DOUBLE",
        },
        "column_nullables": {
            "STATION": False,
            "TSTAMP": False,
            "DIESEL": False,
            "E5": False,
        },
    }
}
DEFINES = (
    "A AS A.DIESEL <= A.E5 AND A.E5 >= 8, "
    "B AS B.DIESEL > B.E5 AND B.DIESEL > A.DIESEL AND B.E5 < A.E5, "
    "C AS C.DIESEL > C.E5"
)
PATTERN = "A (B+ C*?)+ A"
MEASURES = "MATCH_NUMBER() AS MN, CLASSIFIER() AS TAG"
EXTRA_MEASURES = (
    ", LAST(D.TSTAMP) - FIRST(D.TSTAMP) AS DURA, ABS(AVG(C.DIESEL) - A.DIESEL) AS DIFF"
)


def parse(
    pattern=PATTERN,
    definitions=DEFINES,
    *,
    source="GAS_PRICE",
    schema=SCHEMA,
    measures=MEASURES,
    subset="SUBSET D=(B,C)",
    skip="TO LAST B",
    rows="ALL ROWS PER MATCH",
):
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(
        f"SELECT * FROM {source} MATCH_RECOGNIZE("
        f"PARTITION BY STATION ORDER BY TSTAMP MEASURES {measures} "
        f"{rows} AFTER MATCH SKIP {skip} PATTERN({pattern}) {subset} DEFINE {definitions})",
        table_schemas=schema,
    )
    return ast


def pattern(text):
    return PatternParser(PatternLexer(text)).parse()


def truth(expr, row, bindings, owner):
    """Small SQL evaluator, intentionally not imported from the implementation."""
    if isinstance(expr, NullLiteral):
        return None
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, ColumnRef):
        chosen = (
            row if expr.qualifier in (None, owner) else bindings.get(expr.qualifier, {})
        )
        return chosen.get(expr.name)
    if isinstance(expr, SemanticsExpr):
        return truth(expr.expr, row, bindings, owner)
    if isinstance(expr, NotExpr):
        value = truth(expr.expr, row, bindings, owner)
        return None if value is None else not value
    if isinstance(expr, IsNullExpr):
        value = truth(expr.expr, row, bindings, owner) is None
        return not value if expr.is_not else value
    if isinstance(expr, CaseExpr):
        for condition, result in expr.cases:
            if truth(condition, row, bindings, owner) is True:
                return truth(result, row, bindings, owner)
        return (
            None
            if expr.else_expr is None
            else truth(expr.else_expr, row, bindings, owner)
        )
    if isinstance(expr, FunctionCall):
        args = [truth(arg, row, bindings, owner) for arg in expr.args]
        if expr.name in {"NVL", "COALESCE"}:
            return next((value for value in args if value is not None), None)
        if expr.name == "NULLIF":
            return None if args[0] == args[1] else args[0]
        if expr.name == "ABS":
            return None if args[0] is None else abs(args[0])
        raise AssertionError(expr)
    if isinstance(expr, BetweenExpr):
        expr = BinaryExpr(
            BinaryExpr(expr.expr, ">=", expr.lower),
            "AND",
            BinaryExpr(expr.expr, "<=", expr.upper),
        )
    if isinstance(expr, InExpr):
        values = [
            truth(BinaryExpr(expr.expr, "=", value), row, bindings, owner)
            for value in expr.values
        ]
        result = True if True in values else None if None in values else False
        return None if result is None else not result if expr.is_not else result
    if isinstance(expr, BinaryExpr):
        left, right = truth(expr.left, row, bindings, owner), truth(
            expr.right, row, bindings, owner
        )
        if expr.op == "AND":
            return (
                False
                if left is False or right is False
                else None if left is None or right is None else True
            )
        if expr.op == "OR":
            return (
                True
                if left is True or right is True
                else None if left is None or right is None else False
            )
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
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "/": lambda: left / right,
        }[expr.op]()
    raise AssertionError(expr)


def ordered_paths(node, rows, position, labels, accepts):
    if isinstance(node, AnchorPattern):
        if node.anchor_type == AnchorType.START and position != 0:
            return
        for end, tags in ordered_paths(node.pattern, rows, position, labels, accepts):
            if node.anchor_type != AnchorType.END or end == len(rows):
                yield end, tags
        return
    if isinstance(node, ExcludedPattern):
        yield from ordered_paths(node.pattern, rows, position, labels, accepts)
        return
    if isinstance(node, EmptyPattern):
        yield position, labels
    elif isinstance(node, VariablePattern):
        if position < len(rows) and accepts(node.name, position, labels):
            yield position + 1, labels + (node.name,)
    elif isinstance(node, AlternationPattern):
        for child in node.patterns:
            yield from ordered_paths(child, rows, position, labels, accepts)
    elif isinstance(node, ConcatenationPattern):

        def rest(index, pos, tags):
            if index == len(node.patterns):
                yield pos, tags
            else:
                for end, output in ordered_paths(
                    node.patterns[index], rows, pos, tags, accepts
                ):
                    yield from rest(index + 1, end, output)

        yield from rest(0, position, labels)
    elif isinstance(node, QuantifiedPattern):

        def repeat(count, pos, tags):
            if not node.isGreedy and count >= node.atLeast:
                yield pos, tags
            if node.atMost is None or count < node.atMost:
                for end, output in ordered_paths(
                    node.pattern, rows, pos, tags, accepts
                ):
                    if end > pos:  # Test corpus excludes nullable repeated bodies.
                        yield from repeat(count + 1, end, output)
            if node.isGreedy and count >= node.atLeast:
                yield pos, tags

        yield from repeat(0, position, labels)
    else:
        raise AssertionError(node)


def preferred(ast, rows, start):
    def accepts(name, position, labels):
        condition = ast.define.definitions.get(name)
        bindings = {tag: rows[start + offset] for offset, tag in enumerate(labels)}
        return (
            condition is None
            or truth(condition.expression, rows[position], bindings, name) is True
        )

    return next(ordered_paths(ast.pattern, rows, start, (), accepts), None)


def observations(ast, rows):
    start, result = 0, []
    while start < len(rows):
        match = preferred(ast, rows, start)
        if match is None:
            start += 1
            continue
        end, labels = match
        running = []
        for prefix in range(1, len(labels) + 1):
            tags = labels[:prefix]
            d = [
                rows[start + index]
                for index, tag in enumerate(tags)
                if tag in {"B", "C"}
            ]
            c = [
                rows[start + index]["DIESEL"]
                for index, tag in enumerate(tags)
                if tag == "C"
            ]
            a = [
                rows[start + index]["DIESEL"]
                for index, tag in enumerate(tags)
                if tag == "A"
            ]
            c = [value for value in c if value is not None]
            running.append(
                (
                    None if not d else d[-1]["TSTAMP"] - d[0]["TSTAMP"],
                    (
                        None
                        if not c or not a or a[-1] is None
                        else abs(sum(c) / len(c) - a[-1])
                    ),
                )
            )
        result.append((start, end, labels, tuple(running)))
        skip = ast.after_match_skip_to
        if skip.startswith(("LAST ", "FIRST ")):
            mode, label = skip.split()
            members = (ast.subsets or {}).get(label, [label])
            positions = [
                start + index for index, tag in enumerate(labels) if tag in members
            ]
            if not positions:
                result.append(("skip error", label))
                break
            next_start = positions[-1] if mode == "LAST" else positions[0]
            if next_start <= start:
                result.append(("skip error", label))
                break
            start = next_start
        else:
            start = start + 1 if skip == "NEXT ROW" or end == start else end
    return result


class UltraSqlTests(unittest.TestCase):
    def test_requested_rewrite_and_roundtrip(self):
        for extra in ("", EXTRA_MEASURES):
            ast = parse(measures=MEASURES + extra)
            original = copy.deepcopy(ast)
            rule = RegexOptimizationUltraRule()
            self.assertTrue(rule.is_applicable(ast))
            self.assertTrue(rule.apply(ast), rule.last_diagnostics)
            self.assertEqual(ast.pattern.to_string(), "A B (B | C)* A")
            c = ast.define.definitions["C"].expression.to_string()
            self.assertIn("NOT", c)
            self.assertIn("DIESEL > A.DIESEL", c)
            self.assertNotIn("CASE", c)
            self.assertEqual(ast.measures, original.measures)
            self.assertEqual(ast.subsets, original.subsets)
            self.assertEqual(ast.after_match_skip_to, original.after_match_skip_to)
            reparsed = RPRAbstractSyntaxTree()
            reparsed.parse_sql(ast.to_sql(), table_schemas=SCHEMA)
            self.assertEqual(reparsed.to_sql(), ast.to_sql())
            once = ast.to_sql()
            self.assertFalse(rule.apply(ast))
            self.assertEqual(ast.to_sql(), once)

    def test_independent_differential_labels_measures_and_skips(self):
        domain = [(7, 9), (11, 7), (8, 7), (4, 2), (1, 2), (None, 9), (11, None)]
        # Both a overlap and b-only relative to the prefix are exercised, as
        # well as C-only followed by B again and adjacent/overlapping matches.
        schema = copy.deepcopy(SCHEMA)
        schema["GAS_PRICE"]["column_nullables"].update(DIESEL=True, E5=True)
        for skip in (
            "TO LAST B",
            "TO FIRST B",
            "TO LAST D",
            "TO NEXT ROW",
            "PAST LAST ROW",
        ):
            original = parse(
                schema=schema, measures=MEASURES + EXTRA_MEASURES, skip=skip
            )
            rewritten = copy.deepcopy(original)
            rule = RegexOptimizationUltraRule()
            self.assertTrue(rule.apply(rewritten), rule.last_diagnostics)
            self.assertIn(
                "CASE", rewritten.define.definitions["C"].expression.to_string()
            )
            samples = itertools.chain(
                itertools.product(domain, repeat=4),
                [[(7, 9), (11, 7), (8, 7), (11, 7), (7, 9), (11, 7), (7, 9)]],
            )
            for sample in samples:
                rows = [
                    dict(DIESEL=d, E5=e, TSTAMP=i, STATION=1)
                    for i, (d, e) in enumerate(sample)
                ]
                self.assertEqual(
                    observations(original, rows),
                    observations(rewritten, rows),
                    (skip, sample),
                )

    def test_nullable_residual_includes_unknown(self):
        schema = copy.deepcopy(SCHEMA)
        schema["GAS_PRICE"]["column_nullables"].update(DIESEL=True, E5=True)
        # Prefix A does not require DIESEL to be non-null. B is UNKNOWN, C is
        # TRUE on the middle row: replacing CASE with NOT would lose a match.
        ast = parse(
            "A (B | C)+ A",
            "A AS E5=0, B AS DIESEL>0 AND DIESEL>A.DIESEL, C AS DIESEL>0",
            schema=schema,
        )
        expressions = {
            name: normalize(cond.expression, name)
            for name, cond in ast.define.definitions.items()
        }
        algebra = PredicateAlgebraUltra(ast, expressions)
        guards = algebra.residuals(("A", "B", "C"))
        c = guards["C"]
        self.assertIn("CASE", c.to_string())
        self.assertIs(
            truth(c, {"DIESEL": 3, "E5": 1}, {"A": {"DIESEL": None, "E5": 0}}, "C"),
            True,
        )

    def test_renamed_symbols_and_different_shape(self):
        ast = parse(
            "START (HOT+ WARM*?)+ END",
            "START AS E5=0, HOT AS E5>0 AND DIESEL>10, WARM AS E5>0, END AS E5<0",
            subset="",
            skip="PAST LAST ROW",
        )
        rule = RegexOptimizationUltraRule()
        self.assertTrue(rule.apply(ast), rule.last_diagnostics)
        self.assertEqual(ast.pattern.to_string(), "START HOT (HOT | WARM)* END")
        different = parse("A (B C | B C B C) A", subset="")
        self.assertTrue(rule.apply(different), rule.last_diagnostics)
        self.assertIn(
            different.pattern.to_string(), ("A B C (B C)? A", "A (B C){1,2} A")
        )

    def test_order_search_can_recover_language_candidate(self):
        ast = parse(
            "A | A B", "A AS E5>0, B AS DIESEL>0", subset="", skip="PAST LAST ROW"
        )
        original = copy.deepcopy(ast)
        rule = RegexOptimizationUltraRule()
        # Keep this specifically an ordering-search test. The generic branch
        # pruner can now also prove the smaller candidate A for this query.
        with patch(
            "rewriting_rules.regex_optimization_ultra.observable_proposals",
            return_value=[],
        ):
            self.assertTrue(rule.apply(ast), rule.last_diagnostics)
        self.assertIn("??", ast.pattern.to_string())
        self.assertGreater(rule.last_diagnostics["candidate_checks"], 1)
        rows = [dict(DIESEL=1, E5=1, TSTAMP=i) for i in range(4)]
        self.assertEqual(observations(original, rows), observations(ast, rows))

    def test_moving_and_optional_prefixes_fail_closed(self):
        for source in ("A? (B+ C*?)+ A", "(A (B+ C*?)+)+", "A (B+ A C*?)+ A"):
            ast = parse(source)
            before = ast.to_sql()
            rule = RegexOptimizationUltraRule()
            self.assertFalse(rule.apply(ast))
            self.assertEqual(before, ast.to_sql())
            self.assertEqual(rule.last_diagnostics["status"], "inconclusive")

    def test_unsupported_boundaries_and_function(self):
        for text in ("A PERMUTE(B,C) A",):
            ast = parse(text)
            before = ast.to_sql()
            self.assertFalse(RegexOptimizationUltraRule().apply(ast))
            self.assertEqual(ast.to_sql(), before)
        ast = parse()
        ast.define.definitions["B"].expression = ExprParser().parse_expression(
            "SUM(B.DIESEL)>A.DIESEL"
        )
        self.assertFalse(RegexOptimizationUltraRule().apply(ast))

    def test_budgets_and_failed_validation_are_transactional(self):
        for config in (
            UltraConfig(timeout_seconds=0),
            UltraConfig(max_predicate_cells=1),
            UltraConfig(max_product_states=1),
            UltraConfig(max_nfa_states=2),
            UltraConfig(max_candidate_checks=0),
            UltraConfig(max_predicate_nodes=1),
        ):
            ast = parse()
            before = ast.to_sql()
            rule = RegexOptimizationUltraRule(config)
            self.assertFalse(rule.apply(ast))
            self.assertEqual(ast.to_sql(), before)
        ast = parse()
        before = ast.to_sql()
        with patch.object(RPRAbstractSyntaxTree, "validate", return_value=["invalid"]):
            self.assertFalse(
                RegexOptimizationUltraRule(UltraConfig(max_candidate_checks=4)).apply(
                    ast
                )
            )
        self.assertEqual(ast.to_sql(), before)
        with patch(
            "rewriting_rules.regex_optimization_ultra.proposals",
            side_effect=BudgetUltra("worker timeout"),
        ):
            self.assertFalse(RegexOptimizationUltraRule().apply(ast))
        self.assertEqual(ast.to_sql(), before)

    def test_engine_integration(self):
        ast, rules = RewriteEngine([RegexOptimizationUltraRule()]).rewrite(parse())
        self.assertEqual(rules, ["regex_optimization_ultra"])
        self.assertEqual(ast.pattern.to_string(), "A B (B | C)* A")

    def test_output_modes_preserved(self):
        for mode in (
            "ONE ROW PER MATCH",
            "ALL ROWS PER MATCH SHOW EMPTY MATCHES",
            "ALL ROWS PER MATCH OMIT EMPTY MATCHES",
            "ALL ROWS PER MATCH WITH UNMATCHED ROWS",
        ):
            ast = parse(rows=mode)
            self.assertTrue(RegexOptimizationUltraRule().apply(ast))
            self.assertEqual(ast.rows_per_match, mode)

    def test_unknown_udf_and_nullable_repetition_fail_closed(self):
        ast = parse(
            definitions=DEFINES.replace("B.DIESEL > B.E5", "MYSTERY(B.DIESEL) > B.E5")
        )
        self.assertFalse(RegexOptimizationUltraRule().apply(ast))
        ast = parse()
        ast.pattern = ConcatenationPattern(
            [
                VariablePattern("A"),
                QuantifiedPattern(
                    ConcatenationPattern(
                        [
                            QuantifiedPattern(VariablePattern("B"), 0, 1, True),
                            QuantifiedPattern(VariablePattern("C"), 0, 1, True),
                        ]
                    ),
                    0,
                    None,
                    True,
                ),
                VariablePattern("A"),
            ]
        )
        before = ast.to_sql()
        self.assertFalse(RegexOptimizationUltraRule().apply(ast))
        self.assertEqual(ast.to_sql(), before)


class UltraNullabilityTests(unittest.TestCase):
    def test_nested_alias_arithmetic_and_functions(self):
        sources = [
            "(SELECT STATION,TSTAMP,DIESEL+1 AS DIESEL,E5*2 AS E5 FROM GAS_PRICE)",
            "(SELECT STATION,TSTAMP,ABS(X) AS DIESEL,Y+1 AS E5 FROM "
            "(SELECT STATION,TSTAMP,DIESEL*2 AS X,E5 AS Y FROM GAS_PRICE))",
        ]
        for source in sources:
            ast = parse(source=source)
            facts = NullabilityUltra(ast).columns
            self.assertTrue(facts["DIESEL"])
            self.assertTrue(facts["E5"])
            rule = RegexOptimizationUltraRule()
            self.assertTrue(rule.apply(ast), rule.last_diagnostics)
            self.assertNotIn("CASE", ast.define.definitions["C"].expression.to_string())

    def test_nullable_alias_shadowing_does_not_borrow_base_fact(self):
        for expression in (
            "NULL",
            "NULLIF(DIESEL,DIESEL)",
            "CASE WHEN E5>0 THEN DIESEL END",
        ):
            # The legacy parser cannot infer a data TYPE for an all-NULL
            # projection. Supply the already parsed AST to test Ultra's
            # independent nullability analysis without modifying that parser.
            ast = parse()
            ast.input_table = (
                f"(SELECT STATION,TSTAMP,{expression} AS DIESEL,E5 FROM GAS_PRICE)"
            )
            self.assertFalse(NullabilityUltra(ast).columns["DIESEL"])
            rule = RegexOptimizationUltraRule()
            self.assertTrue(rule.apply(ast), rule.last_diagnostics)
            self.assertIn("CASE", ast.define.definitions["C"].expression.to_string())

    def test_filter_and_coalesce_propagation(self):
        schema = copy.deepcopy(SCHEMA)
        schema["GAS_PRICE"]["column_nullables"].update(DIESEL=True, E5=True)
        for source in (
            "(SELECT * FROM GAS_PRICE WHERE DIESEL IS NOT NULL AND E5 IS NOT NULL)",
            "(SELECT STATION,TSTAMP,COALESCE(DIESEL,0) AS DIESEL,NVL(E5,0) AS E5 FROM GAS_PRICE)",
        ):
            ast = parse(source=source, schema=schema)
            facts = NullabilityUltra(ast).columns
            self.assertTrue(facts["DIESEL"])
            self.assertTrue(facts["E5"])
            self.assertTrue(RegexOptimizationUltraRule().apply(ast))
            self.assertNotIn("CASE", ast.define.definitions["C"].expression.to_string())

    def test_join_and_unknown_expressions_are_conservative(self):
        ast = parse()
        ast.input_table = "(SELECT G.DIESEL,G.E5 FROM GAS_PRICE G LEFT JOIN GAS_PRICE H ON G.STATION=H.STATION)"
        self.assertEqual(NullabilityUltra(ast).columns, {})
        ast.input_table = "(SELECT MYSTERY(DIESEL) AS DIESEL,E5 FROM GAS_PRICE)"
        self.assertFalse(NullabilityUltra(ast).columns["DIESEL"])


class UltraAutomataTests(unittest.TestCase):
    def verify(self, left, right, cells):
        return verify_labels(
            pattern(left), pattern(right), cells, deadline=monotonic() + 5
        )

    def test_same_endpoint_different_classifier_rejected(self):
        overlap = ((frozenset("BC"), frozenset("BC")),)
        proof = self.verify("B | C", "C | B", overlap)
        self.assertFalse(proof.equivalent)
        self.assertEqual(len(proof.witness), 1)

    def test_priority_fallback_and_outside_alphabet(self):
        labels = [frozenset(), frozenset("A"), frozenset("B"), frozenset("AB")]
        cells = tuple((item, item) for item in labels)
        self.assertTrue(self.verify("A | A B", "A B??", cells).equivalent)
        self.assertFalse(self.verify("A | A B", "A B?", cells).equivalent)
        self.assertFalse(self.verify("A+?", "A+", cells).equivalent)
        self.assertTrue(self.verify("A+ A+", "A{2,}", cells).equivalent)

    def test_full_label_history_not_only_last_label(self):
        cells = ((frozenset("ABC"), frozenset("ABC")),)
        self.assertFalse(self.verify("A C", "B C", cells).equivalent)

    def test_greedy_c_variant_cannot_use_requested_residual(self):
        before = [frozenset(), frozenset("A"), frozenset("BC"), frozenset("C")]
        after = [frozenset(), frozenset("A"), frozenset("B"), frozenset("C")]
        proof = self.verify("A (B+ C*)+ A", "A B (B | C)* A", tuple(zip(before, after)))
        self.assertFalse(proof.equivalent)

    def test_exhaustive_oracle_for_certificates_and_counterexamples(self):
        texts = [
            "A",
            "A+",
            "A+?",
            "A B",
            "A | B",
            "B | A",
            "A | A B",
            "A B??",
            "A B?",
            "A* B",
            "A*? B",
            "(A B*)+",
            "A (A | B)*",
            "(A+ B*?)+",
            "(A+ B*)+",
            "A{1,3} B?",
            "A A*",
            "A* A",
            "(A | A B)+",
            "A B | B A",
        ]
        alphabet = [frozenset(), frozenset("A"), frozenset("B"), frozenset("AB")]
        for left, right in itertools.product(texts, repeat=2):
            lp, rp = pattern(left), pattern(right)
            proof = self.verify(left, right, tuple((cell, cell) for cell in alphabet))

            def run(node, word):
                return next(
                    ordered_paths(
                        node, word, 0, (), lambda label, index, _: label in word[index]
                    ),
                    None,
                )

            if proof.equivalent:
                for length in range(5):
                    for word in itertools.product(alphabet, repeat=length):
                        self.assertEqual(
                            run(lp, word), run(rp, word), (left, right, word)
                        )
            else:
                word = [alphabet[index] for index in proof.witness]
                self.assertNotEqual(run(lp, word), run(rp, word), (left, right, word))

    def test_joint_predicate_alphabets_against_oracle(self):
        rng = random.Random(17)
        texts = ["A+", "A B", "A | B", "A B?", "A | A B", "(A+ B*?)+", "A (A | B)*"]
        labels = [frozenset(), frozenset("A"), frozenset("B"), frozenset("AB")]
        for _ in range(80):
            left, right = rng.choice(texts), rng.choice(texts)
            cells = rng.sample(list(itertools.product(labels, repeat=2)), 4)
            proof = self.verify(left, right, cells)

            def run(text, indices, side):
                word = [cells[index][side] for index in indices]
                return next(
                    ordered_paths(
                        pattern(text),
                        word,
                        0,
                        (),
                        lambda label, index, _: label in word[index],
                    ),
                    None,
                )

            if proof.equivalent:
                for length in range(5):
                    for indices in itertools.product(range(4), repeat=length):
                        self.assertEqual(run(left, indices, 0), run(right, indices, 1))
            else:
                self.assertNotEqual(
                    run(left, proof.witness, 0), run(right, proof.witness, 1)
                )

    def test_arrangements_vary_nested_orders_and_greediness(self):
        options = list(arrangements(pattern("(A | B) (C | D)*"), limit=16))
        self.assertEqual(len(options), 8)
        self.assertIn("(B | A) (D | C)*?", [item.to_string() for item in options])


if __name__ == "__main__":
    unittest.main()
