"""Validation of quantified subpatterns that can match empty rows.

A quantifier must not be applied to a subpattern that can match zero rows.
``parse_pattern_string`` is expected to reject such patterns, while patterns
whose quantified subpattern always matches at least one row must be accepted.

Run this file directly to see the result of every test case::

    python func_test/test_valid_pattern.py
"""

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from patternnode import QuantifiedPattern, VariablePattern, parse_pattern_string

# Patterns that SHOULD be rejected: the quantified inner pattern can match
# zero rows, i.e. its minimum row count is 0.
INVALID_PATTERNS = (
    "(A*){5}",  # A* can match 0 rows
    "(A?){5}",  # A? can match 0 rows
    "(A* B*){3}",  # A* B* can match 0 rows
    "(A* | B*){2}",  # A* | B* can match 0 rows
    "((A | B)*){5}",  # (A | B)* can match 0 rows
    "(A*)*",  # A* can match 0 rows
    "(A* A*)*",  # A* A* can match 0 rows
    "(A* B*)*",  # A* B* can match 0 rows
    "(())*",  # () can match 0 rows
    "({-A-} | B*){3}",  # B* can also match 0 rows
)

# Patterns that SHOULD be accepted: the quantified inner pattern always
# matches at least one row.
VALID_PATTERNS = (
    "(A+){ 5 }",  # A+ must match at least 1 row
    "(A+ B+){3}",  # A+ B+ must match at least 2 rows
    "(A+ B){2}",  # A+ B must match at least 2 rows
    "(A* B){2}",  # A* B must match at least 1 row (B is required)
    "(A+ | B+){2}",  # A+ | B+ must match at least 1 row
    "(A+)*",  # optimized to A+ internally by a special rule
    "A? A?",  # A? is optional but is never quantified
    "({-A-}){3}",  # {-A-} must match at least 1 row by a special rule
    "A B+? C*? D??",
    "^A{3, 5}",
    "A () B C",
    # Additional accepted patterns: optional / quantified symbols, PERMUTE,
    # excluded subpatterns, ANYTHING and anchors are all still valid.
    "",
    "A B | C",
    "(A B)+ C",
    "(A | B)* C{ 4 }",
    "PERMUTE(A, B){1,2} C",
    "A? B C+",
    "PERMUTE(A, B)+ C",
    "A B+ C+",
    "A* A* B C",
    "PERMUTE(A, (B C), D+)",
    "(A B) | (C D)",
    "A {- B C+ -}{2,3} {- D -}*",
    "{- A {-B-} C -}",
    "{- A {-B-} C-}",
    "A ANYTHING*? B ANYTHING*? C",
    "A {- ANYTHING*? -} B {-ANYTHING*?-} C",
    "A+ B{2, 4} PERMUTE(C, D) E$",
    "A PERMUTE(B C) D{2, 4} E*",
    "A | {- B+ -} | ^C PERMUTE(D, E) F{1,}? $",
    "^A | B$",
    "^A | ()",
    "() | A$",
    "() ^A",
    "A$ ()",
    "(^A){1}",
    "(A$){1}",
)

ZERO_ONLY_QUANTIFIER_PATTERNS = (
    "A{0}",
    "A{0,0}",
    "(A B){0}?",
)

INVALID_ANCHOR_PATTERNS = (
    "A ^B",
    "A$ B",
    "(^A)+",
    "(A$)+",
    "PERMUTE(^A, B)",
    "PERMUTE(A$, B)",
)


def test_quantified_empty_match_subpattern_is_rejected():
    expected_message = (
        "Quantified subpatterns that can have empty matches are not yet supported"
    )
    for pattern in INVALID_PATTERNS:
        try:
            parse_pattern_string(pattern)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(f"pattern should have been rejected: {pattern!r}")


def test_quantified_non_empty_subpattern_is_accepted():
    for pattern in VALID_PATTERNS:
        parse_pattern_string(pattern)


def test_zero_only_quantifier_is_rejected():
    expected_message = "Pattern quantifier upper bound must be greater than 0"
    for pattern in ZERO_ONLY_QUANTIFIER_PATTERNS:
        try:
            parse_pattern_string(pattern)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(f"pattern should have been rejected: {pattern!r}")


def test_reversed_quantifier_bounds_are_rejected():
    for pattern in ("A{3,2}", "(A B){10,4}", "A{2,1}?"):
        try:
            parse_pattern_string(pattern)
        except ValueError as error:
            assert "lower bound must not exceed upper bound" in str(error)
        else:
            raise AssertionError(f"pattern should have been rejected: {pattern!r}")


def test_invalid_anchor_placement_is_rejected():
    for pattern in INVALID_ANCHOR_PATTERNS:
        try:
            parse_pattern_string(pattern)
        except ValueError as error:
            assert "anchor" in str(error).lower()
        else:
            raise AssertionError(f"anchor placement should be rejected: {pattern!r}")


def test_direct_quantifier_construction_validates_bounds():
    cases = (
        (-1, 2, "lower bound must not be negative"),
        (0, 0, "upper bound must be greater than 0"),
        (3, 2, "lower bound must not exceed upper bound"),
    )
    for at_least, at_most, expected_message in cases:
        try:
            QuantifiedPattern(VariablePattern("A"), at_least, at_most, isGreedy=True)
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(
                "direct construction should reject bounds " f"{{{at_least},{at_most}}}"
            )

    # Valid bounded and unbounded ranges remain constructible.
    QuantifiedPattern(VariablePattern("A"), 0, 1, isGreedy=True)
    QuantifiedPattern(VariablePattern("A"), 1, None, isGreedy=False)


if __name__ == "__main__":
    test_quantified_empty_match_subpattern_is_rejected()
    test_quantified_non_empty_subpattern_is_accepted()
    test_zero_only_quantifier_is_rejected()
    test_reversed_quantifier_bounds_are_rejected()
    test_invalid_anchor_placement_is_rejected()
    test_direct_quantifier_construction_validates_bounds()
    total = (
        len(INVALID_PATTERNS)
        + len(VALID_PATTERNS)
        + len(ZERO_ONLY_QUANTIFIER_PATTERNS)
        + len(INVALID_ANCHOR_PATTERNS)
        + 3
        + 5
    )
    print(f"All {total} pattern validation tests passed.")
