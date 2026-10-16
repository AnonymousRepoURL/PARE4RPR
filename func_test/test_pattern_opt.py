import _bootstrap  # noqa: F401  Ensures project modules are importable.

from patternnode import parse_pattern_string

PATTERN_OPTIMIZATION_CASES = (
    # Quantifier normalization and concatenation.
    ("A{1}", "A"),
    ("A{1}?", "A"),
    ("A{2}?", "A{2}"),
    ("A+? A+", "A{2,}"),
    ("A+ A+?", "A{2,}"),
    ("A A A A", "A{4}"),
    ("A A+", "A{2,}"),
    ("A+ A", "A{2,}"),
    ("A A*", "A+"),
    ("A A??", "A{1,2}?"),
    ("A?? A??", "A{0,2}?"),
    ("A? A?", "A{0,2}"),
    ("A{2,2} A??", "A{2,3}?"),
    ("A{2,3} A??", "A{2,3} A??"),
    ("A{2,3} A?", "A{2,4}"),
    ("A A{2,4}", "A{3,5}"),
    ("A* A*", "A*"),
    ("A+ A+ A+ B", "A{3,} B"),
    ("A+ A*? B", "A+ B"),
    ("A B+ (B+)?? C", "A B+ C"),
    ("A B+? (B+?)? C", "A B+? C"),
    # Nested quantifiers.
    # Flatten only when both the accepted counts and their recursive priority
    # order are preserved.  For a variable body, m > 1 is unsafe unless the
    # outer repetition count is fixed.
    ("(A+)?", "A*"),
    ("(A+)*", "A*"),
    ("(A+)+ B", "A+ B"),
    ("(A* A+)*", "A*"),
    ("(A{2,4}){2,5}", "(A{2,4}){2,5}"),
    ("(A{2,4}){2,}", "(A{2,4}){2,}"),
    ("(A{2,}){2,5}", "(A{2,}){2,5}"),
    ("(A{2,}){2,}", "(A{2,}){2,}"),
    ("(A{2,3}){1,5}", "(A{2,3}){1,5}"),
    ("(A{3}){2}", "A{6}"),
    ("(A{2,4}){2}", "A{4,8}"),
    ("(A{3}){1,2}", "(A{3}){1,2}"),
    ("(A{2,}){0,5}", "(A{2,}){0,5}"),
    ("(A{1,3}){0,5}?", "(A{1,3}){0,5}?"),
    ("(A{1,3}?){0,5}?", "A{0,15}?"),
    ("((A | B)? (A | B)?)", "(A | B){0,2}"),
    ("((A | B){1,2}){1,2}", "(A | B){1,4}"),
    ("((A | B | C){1,2}){1,3}", "(A | B | C){1,6}"),
    ("((A C | B D | C){1,2}){1,3}", "(A C | B D | C){1,6}"),
    # Note that (p{m,n}){k} cannot be rewritten
    ("((A | B){1,2}){2}", "((A | B){1,2}){2}"),
    ("((A | B)? (A | B){1,2})", "(A | B)? (A | B){1,2}"),
    ("((A | B)?? (A | B)??)", "(A | B)?? (A | B)??"),
    # Alternation ordering, overlap, and empty branches.
    ("A+? | A+", "A+? | A+"),
    ("A+ | A+?", "A+ | A+?"),
    ("A+ | (A+)?", "A*"),
    ("A{2,10} | A{2,9}", "A{2,10}"),
    ("A{2,10} | A{1,9}", "A{1,10}"),
    ("A{2,10}? | A{4,15}?", "A{2,15}?"),
    ("A{4,5} | A{1,2}", "A{4,5} | A{1,2}"),
    ("A{4,5} | A{2,3}", "A{2,5}"),
    ("A{1,2}? | A{4,5}?", "A{1,2}? | A{4,5}?"),
    ("A{1,2}? | A{3,5}?", "A{1,5}?"),
    ("A | A", "A"),
    ("A | A?", "A?"),
    ("A | A??", "A?"),
    ("A | A{2}", "A{1,2}?"),
    ("A{2} | A", "A{1,2}"),
    ("A | B | A", "A | B"),
    ("(A | B) | A", "A | B"),
    ("A | (B | A)", "A | B"),
    ("(A | B) | (C | D)", "A | B | C | D"),
    ("(A | B) | A C", "A | B | A C"),
    ("A | B | A A", "A | B | A{2}"),
    ("A | B | ()", "A | B?"),
    ("A | B+ | ()", "A | B*"),
    ("() | A", "A??"),
    ("A | ()", "A?"),
    ("() | A? | B", "() | A? | B"),
    ("A | A? | B?", "A? | B?"),
    ("() | A*", "() | A*"),
    ("B | () | A | A{2} | A{3}", "B | A{0,3}?"),
    ("B | A{4,}? | A{2,}?", "B | A{4,}? | A{2,3}?"),
    ("B | A{2,} | C | A{4,}", "B | A{2,} | C"),
    # Grouping, empty patterns, anchors, and exclusions.
    ("A (A | A) () A", "A{3}"),
    ("A () () B", "A B"),
    ("A (B | ()) C", "A B? C"),
    ("A (() | B) C", "A B?? C"),
    ("(A ()) ((() B) ())", "A B"),
    ("^(A | B)", "^(A | B)"),
    ("^(A | B) | C", "^(A | B) | C"),
    ("(A | B)$", "(A | B)$"),
    ("(^A){1}", "^A"),
    ("(A$){1}", "A$"),
    ("{- A {-B-} C -}", "{- A B C -}"),
    ("{- A | B -} | A", "{- A | B -} | A"),
    ("{- A () | (() | B) C-} (D | ())", "{- A | B?? C -} D?"),
    # Repetition of alternatives.
    ("(A | A A)+", "(A{1,2}?)+"),
    ("(A A | A)+", "A+"),
    ("(A | B | A B){2,}", "(A | B){2,}"),
    ("(A B | C | A B C)+", "(A B | C)+"),
    # A suffix exposes the non-monotone total-count order of nested m > 1
    # quantifiers, for both greedy and reluctant matching.
    (
        "^((A{2,3}){1,2} B{1,2})$",
        "^(A{2,3}){1,2} B{1,2}$",
    ),
    (
        "^((A{2,3}?){1,2}? B{1,2})$",
        "^(A{2,3}?){1,2}? B{1,2}$",
    ),
    # Reluctant unbounded repetition prefers fewer iterations.  Keep a branch
    # such as A B available as one iteration; replacing it with two iterations
    # can change the selected classifier trace from A B to A A.
    ("^(A | B | A B)+?$", "^(A | B | A B)+?$"),
    ("(A | B | A B)*?", "(A | B | A B)*?"),
    ("(A | B | A B){2,}?", "(A | B | A B){2,}?"),
    ("(A | B | C | B C)+?", "(A | B | C | B C)+?"),
    # PERMUTE expansion and simplification.
    ("A PERMUTE(B) C", "A B C"),
    ("A PERMUTE(B, ()) C", "A B C"),
    ("A PERMUTE(B, B, B) C", "A B{3} C"),
    ("PERMUTE(B?, B?, B?)", "B{0,3}"),
    ("PERMUTE(B+, B+, B+)", "B{3,}"),
    ("PERMUTE(B*, B*, B*)", "B*"),
    ("PERMUTE(A, B, A)", "A B A | A{2} B | B A{2}"),
    (
        "PERMUTE(A, B, C)",
        "A B C | A C B | B A C | B C A | C A B | C B A",
    ),
    (
        "PERMUTE(B, B B, D)",
        "B{3} D | B D B{2} | B{2} D B | D B{3}",
    ),
    (
        "PERMUTE(A, B, {- A -})",
        "A B {- A -} | A {- A -} B | B A {- A -} | B {- A -} A | "
        "{- A -} A B | {- A -} B A",
    ),
)


def test_pattern_optimization():
    failures = []

    for case_number, (original_pattern, optimized_pattern) in enumerate(
        PATTERN_OPTIMIZATION_CASES, start=1
    ):
        actual_pattern = parse_pattern_string(original_pattern).to_string()
        if actual_pattern != optimized_pattern:
            failures.append(
                (case_number, original_pattern, optimized_pattern, actual_pattern)
            )

    if failures:
        for case_number, original, expected, actual in failures:
            print(f"Case {case_number} is not equivalent:")
            print(f"  original: {original}")
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
        raise SystemExit(f"{len(failures)} pattern optimization case(s) failed.")

    print(f"All {len(PATTERN_OPTIMIZATION_CASES)} pattern optimization cases passed.")


def test_nested_alternation_round_trip_is_stable():
    for source in (
        "(A | B) | A",
        "A | (B | A)",
        "((A | B) | C) | (D | A)",
    ):
        first = parse_pattern_string(source)
        second = parse_pattern_string(first.to_string())
        assert first == second, source
        assert first.to_string() == second.to_string(), source
        assert first.get_var_num() == second.get_var_num(), source


if __name__ == "__main__":
    test_pattern_optimization()
    test_nested_alternation_round_trip_is_stable()
