import itertools
import re
import unittest
from dataclasses import dataclass

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from priority_regex_optimizer import (
    EquivalenceMode,
    ProofScope,
    RegexError,
    Verdict,
    generate_fast_priority_candidates,
    generate_language_candidates,
    optimize_priority_regex,
    prioritized_match_end,
    verify_match_equivalent,
    verify_priority_equivalent,
)


@dataclass(frozen=True)
class SafeRewrite:
    description: str
    original: str
    rewritten: str


@dataclass(frozen=True)
class UnsafeRewrite:
    description: str
    original: str
    rewritten: str
    counterexample: str
    original_end: int
    rewritten_end: int


SAFE_REWRITES = [
    SafeRewrite(
        "unbounded outer repetition absorbs the inner alternatives",
        "(a+|[ab]+)+b",
        "[ab]+b",
    ),
    SafeRewrite(
        "a finite upper bound >= 2 becomes irrelevant, but min=0 is retained",
        "(a+|[ab]+){0,2}b",
        "[ab]*b",
    ),
    SafeRewrite(
        "a finite upper bound >= 2 becomes irrelevant, but min=1 is retained",
        "(a+|[ab]+){1,2}b",
        "[ab]+b",
    ),
    SafeRewrite(
        "the lower bound of two repetitions must remain two",
        "(a+|[ab]+){2,5}b",
        "[ab]{2,}b",
    ),
    SafeRewrite(
        "the first a is already guaranteed by a+",
        "a+[ab]+",
        "a[ab]+",
    ),
    SafeRewrite(
        "the mandatory leading a makes the extra leading a repetitions redundant",
        "a+[ab]*",
        "a[ab]*",
    ),
    SafeRewrite(
        "before the required b, any earlier b can be chosen as that separator",
        "[ab]*b[ab]*",
        "a*b[ab]*",
    ),
    SafeRewrite(
        "the greedy first class absorbs a-only suffixes; the remainder must start with c",
        "[ab]*[ac]*",
        "[ab]*(c[ac]*)?",
    ),
    SafeRewrite(
        "the same overlap reduction remains valid with a nonempty first repetition",
        "[ab]+[ac]*",
        "[ab]+(c[ac]*)?",
    ),
    SafeRewrite(
        "the required middle b can be represented by the first b after the initial symbol",
        "[ab]+b[ab]+",
        "[ab]a*b[ab]+",
    ),
    SafeRewrite(
        "the trailing a repetition is needed only when the optional b is present",
        "a*b?a*",
        "a*(ba*)?",
    ),
    SafeRewrite(
        "after mandatory a characters, the suffix is either empty or b followed by a characters",
        "a+b?a*",
        "a+(ba*)?",
    ),
    SafeRewrite(
        "a suffix is mandatory, so the optional b splits into leading-a and leading-b cases",
        "a*b?a+",
        "a+(ba+)?|ba+",
    ),
    SafeRewrite(
        "the broader second repetition absorbs the leading a-only repetition",
        "a*[ab]*",
        "[ab]*",
    ),
    SafeRewrite(
        "with mandatory a on both sides, the optional b gives two equivalent branches",
        "a+b?a+",
        "a(a+(ba+)?|ba+)",
    ),
    SafeRewrite(
        "the final a repetition is needed only when at least one b is present",
        "a*b*a*",
        "a*(b+a*)?",
    ),
    SafeRewrite(
        "the DFA-derived middle preserves both mandatory a boundaries",
        "a+[ab]a+",
        "a(a+(ba*)?|ba*)a",
    ),
    SafeRewrite(
        "overlapping classes with a mandatory suffix use a DFA-derived form",
        "[ab]*[ac]+",
        "(a*b)*(a+(c[ac]*)?|c[ac]*)",
    ),
    SafeRewrite(
        "the mandatory broader suffix absorbs the leading a-only repetition",
        "a*[ab]+",
        "[ab]+",
    ),
    SafeRewrite(
        "a lazy prefix still permits simplification when the trailing class is greedy",
        "[ab]*?b[ab]*",
        "a*b[ab]*",
    ),
    SafeRewrite(
        "two adjacent greedy a repetitions collapse into one",
        "a*a*",
        "a*",
    ),
    SafeRewrite(
        "the greedy broader prefix absorbs the trailing a-only repetition",
        "[ab]+a*",
        "[ab]+",
    ),
    SafeRewrite(
        "the required b after a nonempty prefix has a DFA-derived form",
        "[ab]+b[ab]*",
        "[ab]a*b[ab]*",
    ),
    SafeRewrite(
        "the required bc core inside a broader class has a DFA-derived form",
        "[abc]*bc[abc]*",
        "([ac]|b+a)*b+c[a-c]*",
    ),
    SafeRewrite(
        "overlapping adjacent character-class repetitions can use the DFA-derived form",
        "[ab]+[ac]+",
        "[ab](a*b)*(a+(c[ac]*)?|c[ac]*)",
    ),
    SafeRewrite(
        "a long fixed-width alternation factors into two incomparable adjacent classes",
        "([ab][bc]|[ab][bd]|[ab][be]|[ab][bf])",
        "[ab][b-f]",
    ),
    SafeRewrite(
        "a Cartesian-product alternation collapses without containment between neighbors",
        "([ab][cd][ef]|[ab][cd][eg]|[ab][ce][ef]|[ab][ce][eg]|"
        "[ac][cd][ef]|[ac][cd][eg]|[ac][ce][ef]|[ac][ce][eg])",
        "[a-c][c-e][e-g]",
    ),
    SafeRewrite(
        "many alternatives sharing two prefixes collapse to one short final range",
        "([ab][bc][cd]|[ab][bc][ce]|[ab][bc][cf]|[ab][bc][cg]|"
        "[ab][bc][ch]|[ab][bc][ci]|[ab][bc][cj]|[ab][bc][ck]|"
        "[ab][bc][cl]|[ab][bc][cm])",
        "[ab][bc][c-m]",
    ),
]


UNSAFE_REWRITES = [
    UnsafeRewrite(
        "an optional outer repetition cannot absorb the preferred a+ branch",
        "(a+|[ab]+)?b",
        "[ab]*b",
        "abb",
        2,
        3,
    ),
    UnsafeRewrite(
        "one outer repetition has the same priority problem",
        "(a+|[ab]+)b",
        "[ab]+b",
        "abb",
        2,
        3,
    ),
    UnsafeRewrite(
        "left-to-right alternative order changes the chosen endpoint",
        "a|aa",
        "aa|a",
        "aa",
        1,
        2,
    ),
    UnsafeRewrite(
        "changing a lazy quantifier to greedy changes the chosen endpoint",
        "a+?",
        "a+",
        "aa",
        1,
        2,
    ),
    UnsafeRewrite(
        "making both sides of the required b greedy changes which suffix is consumed",
        "[ab]*?b[ab]*?",
        "a*b[ab]*",
        "ba",
        1,
        2,
    ),
    UnsafeRewrite(
        "a lazy broad suffix cannot absorb characters like its greedy version",
        "a+[ab]*?",
        "a[ab]*",
        "ab",
        1,
        2,
    ),
    UnsafeRewrite(
        "making the second overlapping class greedy consumes a leading c",
        "[ab]*[ac]*?",
        "[ab]*[ac]*",
        "c",
        0,
        1,
    ),
    UnsafeRewrite(
        "making the first overlapping class greedy consumes a leading b",
        "[ab]*?[ac]*",
        "[ab]*[ac]*",
        "b",
        0,
        1,
    ),
    UnsafeRewrite(
        "a lazy broad prefix may deliberately return empty before a trailing a repetition",
        "[ab]*?a*",
        "[ab]*",
        "b",
        0,
        1,
    ),
    UnsafeRewrite(
        "greedifying the prefix moves the mandatory trailing a to a later position",
        "[ab]*?a+",
        "[ab]*a+",
        "aba",
        1,
        3,
    ),
    UnsafeRewrite(
        "greedifying a mandatory broad prefix consumes an additional b",
        "[ab]+?[ac]*",
        "[ab]+[ac]*",
        "ab",
        1,
        2,
    ),
]


class OrderedMatcherTests(unittest.TestCase):
    def test_matches_python_re_on_supported_constructs(self):
        patterns = [
            "",
            "a|aa",
            "aa|a",
            "(a|ab)c?",
            "a*",
            "a*?",
            "a*a",
            "a*?a",
            "a+?a",
            "(a+|[ab]+)+b",
            "(a+|[ab]+)?b",
            "(ab|a)+?b",
            "a{1,3}",
            "a{1,3}?",
            "(a|ab){1,3}?b",
            "[ab]+[^a]?",
            ".+?b",
        ]
        texts = [
            "".join(chars)
            for length in range(5)
            for chars in itertools.product("ab\n", repeat=length)
        ]
        for pattern in patterns:
            for text in texts:
                with self.subTest(pattern=pattern, text=text):
                    match = re.match(pattern, text)
                    expected = None if match is None else match.end()
                    self.assertEqual(expected, prioritized_match_end(pattern, text))

    def test_nullable_body_loop_is_fail_closed(self):
        result = verify_priority_equivalent("(a?)*", "a*")
        self.assertEqual(Verdict.UNKNOWN, result.verdict)
        with self.assertRaises(RegexError):
            prioritized_match_end("(a?)*", "aaa")


class ExactVerifierTests(unittest.TestCase):
    def test_safe_rewrite_catalogue(self):
        for case in SAFE_REWRITES:
            with self.subTest(
                description=case.description,
                rewrite=f"{case.original}  ==>  {case.rewritten}",
            ):
                result = verify_priority_equivalent(case.original, case.rewritten)
                self.assertEqual(Verdict.PROVED, result.verdict, result)

    def test_unsafe_rewrite_catalogue(self):
        for case in UNSAFE_REWRITES:
            with self.subTest(
                description=case.description,
                rewrite=f"{case.original}  !=>  {case.rewritten}",
            ):
                result = verify_priority_equivalent(case.original, case.rewritten)
                self.assertEqual(Verdict.REJECTED, result.verdict, result)
                self.assertEqual(case.counterexample, result.counterexample)
                self.assertEqual(case.original_end, result.original_end)
                self.assertEqual(case.rewritten_end, result.candidate_end)

    def test_state_limit_never_becomes_a_proof(self):
        result = verify_priority_equivalent(
            "(a|b|c|d)+x", "[a-d]+x", max_product_states=1
        )
        self.assertEqual(Verdict.UNKNOWN, result.verdict)

    def test_hidden_lower_priority_language_is_not_treated_as_equivalent(self):
        # Both expressions prefer the empty match on every input, but the
        # right-hand expression also has a lower-priority two-row match.  A
        # surrounding suffix could expose that path during backtracking.
        result = verify_priority_equivalent("a??", "a??a??")
        self.assertEqual(Verdict.REJECTED, result.verdict)
        self.assertEqual("aa", result.counterexample)

    def test_suffix_exposes_reluctant_fallback_order(self):
        original = "(a*|b|c*)c*([ab]?|b+?)"
        greedified = "(a*|b|c*)c*([ab]?|b+)"

        # The fragments have the same language and preferred standalone end,
        # but that relation is not substitutive under concatenation.
        self.assertEqual(
            Verdict.PROVED,
            verify_priority_equivalent(original, greedified).verdict,
        )
        result = verify_priority_equivalent(
            original + "[ab]a",
            greedified + "[ab]a",
        )
        self.assertEqual(Verdict.REJECTED, result.verdict)
        self.assertEqual("bbbaa", result.counterexample)
        self.assertEqual(4, result.original_end)
        self.assertEqual(5, result.candidate_end)


class MatchOnlyVerifierTests(unittest.TestCase):
    def test_hidden_lower_priority_languages_can_differ(self):
        for original, candidate in (
            ("ab([cde]|cd)", "ab[cde]"),
            ("a|aa", "a"),
            ("a??", "a??a??"),
        ):
            with self.subTest(original=original, candidate=candidate):
                self.assertEqual(
                    Verdict.PROVED,
                    verify_match_equivalent(original, candidate).verdict,
                )
                self.assertEqual(
                    Verdict.REJECTED,
                    verify_priority_equivalent(original, candidate).verdict,
                )

    def test_enclosing_suffix_and_repetition_expose_deleted_fallbacks(self):
        for original, candidate, witness, original_end in (
            ("ab([cde]|cd)e", "ab[cde]e", "abcde", 5),
            ("(a|ab){2}", "a{2}", "aba", 3),
        ):
            with self.subTest(original=original, candidate=candidate):
                result = verify_match_equivalent(original, candidate)
                self.assertEqual(Verdict.REJECTED, result.verdict)
                self.assertEqual(witness, result.counterexample)
                self.assertEqual(original_end, result.original_end)
                self.assertIsNone(result.candidate_end)

    def test_branch_order_still_controls_selected_end(self):
        result = verify_match_equivalent("a|aa", "aa|a")
        self.assertEqual(Verdict.REJECTED, result.verdict)
        self.assertEqual("aa", result.counterexample)
        self.assertEqual(1, result.original_end)
        self.assertEqual(2, result.candidate_end)

    def test_limits_and_unsupported_constructs_remain_unknown(self):
        for original, candidate, limits in (
            ("(a?)*", "a*", {}),
            ("^a", "a", {}),
            ("a(?=b)", "a", {}),
            ("ab", "a", {"max_nfa_states": 1}),
            ("ab([cde]|cd)", "ab[cde]", {"max_product_states": 1}),
        ):
            with self.subTest(original=original, candidate=candidate, limits=limits):
                self.assertEqual(
                    Verdict.UNKNOWN,
                    verify_match_equivalent(original, candidate, **limits).verdict,
                )

    def test_symbolic_verdicts_and_witnesses_agree_with_python(self):
        patterns = (
            "a",
            "a|aa",
            "aa|a",
            "a??",
            "a??a??",
            "a+",
            "a+?",
            "(a|ab)+",
            "(a|ab){2}",
            "a{2}",
            "(a|ab)b?",
            "ab?",
        )
        texts = [
            "".join(chars)
            for size in range(5)
            for chars in itertools.product("ab", repeat=size)
        ]

        def match_end(pattern, text):
            match = re.match(pattern, text)
            return None if match is None else match.end()

        for index, original in enumerate(patterns):
            for candidate in patterns[index + 1 :]:
                with self.subTest(original=original, candidate=candidate):
                    result = verify_match_equivalent(original, candidate)
                    self.assertIsNot(Verdict.UNKNOWN, result.verdict)
                    if result.verdict is Verdict.PROVED:
                        for text in texts:
                            self.assertEqual(
                                match_end(original, text), match_end(candidate, text)
                            )
                    else:
                        text = result.counterexample
                        original_end = match_end(original, text)
                        candidate_end = match_end(candidate, text)
                        self.assertNotEqual(original_end, candidate_end)
                        self.assertEqual(original_end, result.original_end)
                        self.assertEqual(candidate_end, result.candidate_end)


class OptimizerPipelineTests(unittest.TestCase):
    def test_match_mode_prunes_fallback_branches_before_ordered_hazards(self):
        for source, expected in (
            ("ab([cde]|cd)", "ab[c-e]"),
            ("a|ab", "a"),
            ("a|ab|abc", "a"),
            ("a|b|bc", "a|b"),
            ("a|ab|b", "a|b"),
            ("|a", ""),
        ):
            with self.subTest(source=source):
                result = optimize_priority_regex(
                    source, equivalence_mode=EquivalenceMode.MATCH_ONLY
                )
                self.assertEqual(expected, result.optimized)
                self.assertEqual(Verdict.PROVED, result.verification.verdict)
                self.assertEqual(ProofScope.STANDALONE, result.proof_scope)

    def test_default_mode_preserves_the_language_contract(self):
        source = "ab([cde]|cd)"
        self.assertEqual(source, optimize_priority_regex(source).optimized)
        self.assertEqual("ab??", optimize_priority_regex("a|ab").optimized)
        self.assertEqual("a|ab|abc", optimize_priority_regex("a|ab|abc").optimized)

    def test_match_mode_checks_pruning_inside_the_supplied_context(self):
        suffix = "ab([cde]|cd)e"
        result = optimize_priority_regex(
            suffix, equivalence_mode=EquivalenceMode.MATCH_ONLY
        )
        self.assertNotEqual("ab[c-e]e", result.optimized)
        self.assertEqual(
            Verdict.PROVED, verify_match_equivalent(suffix, result.optimized).verdict
        )
        self.assertEqual(5, prioritized_match_end(result.optimized, "abcde"))
        self.assertGreater(result.rejected_candidates, 0)

        result = optimize_priority_regex(
            "(a|ab){2}", equivalence_mode=EquivalenceMode.MATCH_ONLY
        )
        self.assertEqual("(ab??){2}", result.optimized)
        self.assertEqual(ProofScope.COMPOSITIONAL, result.proof_scope)
        self.assertEqual(1, result.rejected_candidates)

    def test_match_mode_routes_external_candidates_to_match_verification(self):
        strict = optimize_priority_regex("a", extra_candidates=["a|ab"])
        match_only = optimize_priority_regex(
            "a", equivalence_mode="match_only", extra_candidates=["a|ab"]
        )
        self.assertEqual("a", strict.optimized)
        self.assertEqual("a|ab", match_only.optimized)
        self.assertEqual(ProofScope.STANDALONE, match_only.proof_scope)
        with self.assertRaises(ValueError):
            optimize_priority_regex("a", equivalence_mode="unsupported")

    def test_match_mode_has_one_total_candidate_budget(self):
        for budget, expected, checked in (
            (0, "a|b|bc", 0),
            (1, "a|b|bc", 1),
            (2, "a|b", 2),
        ):
            with self.subTest(budget=budget):
                result = optimize_priority_regex(
                    "a|b|bc",
                    equivalence_mode=EquivalenceMode.MATCH_ONLY,
                    max_candidates=budget,
                )
                self.assertEqual(expected, result.optimized)
                self.assertEqual(checked, result.candidates_checked)

        # One rejected pruning candidate uses the budget before the otherwise
        # valid compositional factoring candidate can be checked.
        result = optimize_priority_regex(
            "(a|ab){2}",
            equivalence_mode=EquivalenceMode.MATCH_ONLY,
            max_candidates=1,
        )
        self.assertEqual("(a|ab){2}", result.optimized)
        self.assertEqual(1, result.candidates_checked)

        result = optimize_priority_regex(
            "a|b",
            equivalence_mode=EquivalenceMode.MATCH_ONLY,
            max_candidates=2,
            extra_candidates=["b"],
        )
        self.assertEqual("a|b", result.optimized)
        self.assertEqual(2, result.candidates_checked)
        self.assertEqual(2, result.rejected_candidates)

    def test_match_mode_keeps_reluctance_and_unknown_fail_closed(self):
        reluctant = optimize_priority_regex(
            "a??|ab", equivalence_mode=EquivalenceMode.MATCH_ONLY
        )
        self.assertEqual("a??|ab", reluctant.optimized)
        self.assertEqual(0, reluctant.candidates_checked)
        for pattern, limits in (
            ("ab([cde]|cd)", {"max_product_states": 1}),
            ("(a?|ab)*", {}),
        ):
            with self.subTest(pattern=pattern, limits=limits):
                result = optimize_priority_regex(
                    pattern, equivalence_mode=EquivalenceMode.MATCH_ONLY, **limits
                )
                self.assertEqual(pattern, result.optimized)
                self.assertGreater(result.unknown_candidates, 0)

    def test_optimizer_reports_the_scope_of_its_proof(self):
        compositional = optimize_priority_regex("ab|abc")
        standalone = optimize_priority_regex("(a+|[ab]+)+b")

        self.assertTrue(compositional.changed)
        self.assertEqual(ProofScope.COMPOSITIONAL, compositional.proof_scope)
        self.assertTrue(standalone.changed)
        self.assertEqual(ProofScope.STANDALONE, standalone.proof_scope)

    def test_ordered_optional_factoring_preserves_branch_order(self):
        cases = {
            "ab|abc": "abc??",
            "abc|ab": "abc?",
            "ab|cab": "c??ab",
            "cab|ab": "c?ab",
        }
        for pattern, expected in cases.items():
            with self.subTest(pattern=pattern):
                self.assertIn(
                    expected,
                    generate_fast_priority_candidates(pattern),
                )
                result = optimize_priority_regex(pattern)
                self.assertEqual(expected, result.optimized)
                self.assertEqual(Verdict.PROVED, result.verification.verdict)
                for suffix in ("a", "[ab]", "a|bb", "[a-c]*d", "a?b"):
                    with self.subTest(pattern=pattern, suffix=suffix):
                        self.assertEqual(
                            Verdict.PROVED,
                            verify_priority_equivalent(
                                f"({pattern})({suffix})",
                                f"({expected})({suffix})",
                            ).verdict,
                        )

    def test_ordered_factoring_remains_equivalent_under_a_suffix(self):
        pattern = "a|(ab|abc)d?"
        priority_preserving = "a|abc??d?"
        priority_erasing = "a|abc?d?"

        result = optimize_priority_regex(pattern)
        self.assertEqual(priority_preserving, result.optimized)
        self.assertNotEqual(priority_erasing, result.optimized)

        self.assertEqual(
            Verdict.PROVED,
            verify_priority_equivalent(
                f"({pattern})c",
                f"({priority_preserving})c",
            ).verdict,
        )
        exposed = verify_priority_equivalent(
            f"({pattern})c",
            f"({priority_erasing})c",
        )
        self.assertEqual(Verdict.REJECTED, exposed.verdict)
        self.assertEqual("abcc", exposed.counterexample)
        self.assertEqual(3, exposed.original_end)
        self.assertEqual(4, exposed.candidate_end)

    def test_unhandled_ordered_alternation_fails_closed(self):
        pattern = "a|ab|abc"
        result = optimize_priority_regex(pattern)
        self.assertEqual(pattern, result.optimized)
        self.assertFalse(result.changed)
        self.assertEqual(0, result.candidates_checked)

    def test_variable_width_ordered_prefix_fails_closed(self):
        cases = (
            "a|(ab*|ab*c)d?",
            "a|(ab+|ab+c)d?",
            "a|(ab?|ab?c)d?",
            "a|(ab{1,3}|ab{1,3}c)d?",
            "a|(ab*c|ab*)d?",
            "a|(b*a|cb*a)d?",
            "a|((ab|ac)|(ab|ac)d)e?",
        )
        for pattern in cases:
            with self.subTest(pattern=pattern):
                result = optimize_priority_regex(pattern)
                self.assertEqual(pattern, result.optimized)
                self.assertFalse(result.changed)
                self.assertEqual(0, result.candidates_checked)

        # A reluctant optional is not a context-independent repair when the
        # shared prefix can itself backtrack.
        exposed = verify_priority_equivalent(
            "(ab{1,3}|ab{1,3}c)b",
            "(ab{1,3}c??)b",
        )
        self.assertEqual(Verdict.REJECTED, exposed.verdict)
        self.assertEqual("abbcb", exposed.counterexample)
        self.assertEqual(3, exposed.original_end)
        self.assertEqual(5, exposed.candidate_end)

    def test_exact_repetition_ordered_prefix_can_be_factored(self):
        pattern = "ab{2}|ab{2}c"
        expected = "ab{2}c??"
        result = optimize_priority_regex(pattern)
        self.assertEqual(expected, result.optimized)
        self.assertEqual(Verdict.PROVED, result.verification.verdict)
        for suffix in ("a", "b", "c", "[a-c]*d"):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    Verdict.PROVED,
                    verify_priority_equivalent(
                        f"({pattern})({suffix})",
                        f"({expected})({suffix})",
                    ).verdict,
                )

    def test_fast_candidates_cover_common_repetition_shapes(self):
        cases = {
            "a+[ab]*": "a[ab]*",
            "[ab]*b[ab]*": "a*b[ab]*",
            "a*b?a*": "a*(ba*)?",
            "a+b*a*": "a+(b+a*)?",
            "(a[ab]|a)+b": "(a[ab]?)+b",
        }
        for pattern, expected in cases.items():
            with self.subTest(pattern=pattern):
                self.assertIn(
                    expected,
                    generate_fast_priority_candidates(pattern),
                )

    def test_greenery_generates_multiple_untrusted_candidates(self):
        candidates = generate_language_candidates("(a+|[ab]+)+b")
        self.assertGreaterEqual(len(candidates), 2)
        self.assertIn("[ab]+b", candidates)

    def test_optimizer_selects_only_proved_candidate(self):
        safe = optimize_priority_regex("(a+|[ab]+)+b")
        unsafe = optimize_priority_regex("(a+|[ab]+)?b")
        self.assertEqual("[ab]+b", safe.optimized)
        self.assertEqual(Verdict.PROVED, safe.verification.verdict)
        self.assertEqual("(a+|[ab]+)?b", unsafe.optimized)

    def test_general_reducer_cannot_erase_unresolved_reluctance(self):
        pattern = "(a*|b|c*)c*([ab]?|b+?)"
        greedy_candidate = "(a*|b|c*)c*([ab]?|b+)"

        self.assertIn(greedy_candidate, generate_language_candidates(pattern))
        self.assertEqual(pattern, optimize_priority_regex(pattern).optimized)
        self.assertEqual(
            pattern,
            optimize_priority_regex(
                pattern, extra_candidates=[greedy_candidate]
            ).optimized,
        )

    def test_reluctance_blocks_structural_candidates_before_generation(self):
        pattern = "a??|(a*[ab]*)?"
        context_unsafe_candidate = "a??|[ab]*?"

        self.assertIn(
            context_unsafe_candidate,
            generate_fast_priority_candidates(pattern),
        )
        self.assertEqual(
            Verdict.PROVED,
            verify_priority_equivalent(
                pattern,
                context_unsafe_candidate,
            ).verdict,
        )

        exposed = verify_priority_equivalent(
            f"({pattern})a",
            f"({context_unsafe_candidate})a",
        )
        self.assertEqual(Verdict.REJECTED, exposed.verdict)
        self.assertEqual("baa", exposed.counterexample)
        self.assertEqual(3, exposed.original_end)
        self.assertEqual(2, exposed.candidate_end)

        result = optimize_priority_regex(pattern)
        self.assertEqual(pattern, result.optimized)
        self.assertFalse(result.changed)
        self.assertEqual(0, result.candidates_checked)

    def test_external_candidate_uses_the_same_exact_verifier(self):
        result = optimize_priority_regex("a+[ab]+", extra_candidates=["a[ab]+"])
        self.assertEqual("a[ab]+", result.optimized)
        self.assertEqual(Verdict.PROVED, result.verification.verdict)

    def test_default_limit_is_four_times_the_source_length(self):
        pattern = "[ab]+[ac]+"
        longer_candidate = "[ab](a*b)*(a+(c[ac]*)?|c[ac]*)"

        default_result = optimize_priority_regex(pattern)
        two_times_is_too_small = optimize_priority_regex(
            pattern,
            max_output_length_factor=2,
        )
        five_times_is_allowed = optimize_priority_regex(
            pattern,
            max_output_length_factor=5,
        )

        self.assertEqual(longer_candidate, default_result.optimized)
        self.assertEqual(pattern, two_times_is_too_small.optimized)
        self.assertEqual(longer_candidate, five_times_is_allowed.optimized)
        self.assertEqual(Verdict.PROVED, default_result.verification.verdict)

    def test_absolute_length_limit_overrides_the_factor(self):
        pattern = "[ab]+[ac]+"
        longer_candidate = "[ab](a*b)*(a+(c[ac]*)?|c[ac]*)"

        blocked = optimize_priority_regex(
            pattern,
            max_output_length_factor=100,
            max_output_length=len(longer_candidate) - 1,
        )
        allowed = optimize_priority_regex(
            pattern,
            max_output_length_factor=1,
            max_output_length=len(longer_candidate),
        )

        self.assertEqual(pattern, blocked.optimized)
        self.assertEqual(longer_candidate, allowed.optimized)


if __name__ == "__main__":
    unittest.main()
