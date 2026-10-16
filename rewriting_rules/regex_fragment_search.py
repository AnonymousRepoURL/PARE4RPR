"""Syntax-only discovery for regular MATCH_RECOGNIZE fragments.

The plus rule optimizes *islands*, not an entire row-pattern tree.  This module
contains the tree traversal and cheap structural predicates used by both search
strategies.  It deliberately knows nothing about SQL DEFINE expressions or the
regex optimizer, keeping phase 1 of the algorithm independent from later
encoding and proof phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from patternnode import (
    AlternationPattern,
    AnchorPattern,
    ConcatenationPattern,
    EmptyPattern,
    ExcludedPattern,
    PatternNode,
    PermutationPattern,
    QuantifiedPattern,
    VariablePattern,
)


@dataclass(frozen=True)
class PatternLocation:
    """A root, child attribute, or contiguous list slice to replace."""

    kind: str
    owner: Optional[object] = None
    attribute: Optional[str] = None
    patterns: Optional[List[PatternNode]] = None
    start: int = 0
    end: int = 0
    splice_concatenation: bool = False


def iter_regular_fragments(
    root: PatternNode, *, max_window_width: int = 8
) -> Iterator[Tuple[PatternLocation, PatternNode]]:
    """Yield large islands first, then smaller windows and children.

    Anchors, exclusions, and PERMUTE nodes are yielded so their contents remain
    searchable.  A provisional concatenation window may contain such a node,
    but the structural filter rejects it before predicate encoding.
    """

    def visit(
        node: PatternNode, location: PatternLocation
    ) -> Iterator[Tuple[PatternLocation, PatternNode]]:
        yield location, node

        if isinstance(node, ConcatenationPattern):
            size = len(node.patterns)
            for width in range(min(size, max_window_width), 1, -1):
                for start in range(0, size - width + 1):
                    if start == 0 and width == size:
                        continue
                    window = ConcatenationPattern(node.patterns[start : start + width])
                    yield (
                        PatternLocation(
                            "list",
                            patterns=node.patterns,
                            start=start,
                            end=start + width,
                            splice_concatenation=True,
                        ),
                        window,
                    )
            for index, child in enumerate(node.patterns):
                yield from visit(
                    child,
                    PatternLocation(
                        "list",
                        patterns=node.patterns,
                        start=index,
                        end=index + 1,
                    ),
                )
            return

        if isinstance(node, (AlternationPattern, PermutationPattern)):
            for index, child in enumerate(node.patterns):
                yield from visit(
                    child,
                    PatternLocation(
                        "list",
                        patterns=node.patterns,
                        start=index,
                        end=index + 1,
                    ),
                )
            return

        if isinstance(node, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
            yield from visit(
                node.pattern,
                PatternLocation(
                    "attribute",
                    owner=node,
                    attribute="pattern",
                ),
            )

    yield from visit(root, PatternLocation("root"))


def regular_node_count(
    node: PatternNode, *, max_quantifier_bound: int = 1000
) -> Optional[int]:
    """Count a regular subtree, or reject syntax outside the regex subset."""

    if isinstance(node, (VariablePattern, EmptyPattern)):
        return 1
    if isinstance(node, QuantifiedPattern):
        if (
            node.atLeast < 0
            or node.atLeast > max_quantifier_bound
            or (node.atMost is not None and node.atMost > max_quantifier_bound)
        ):
            return None
        child_count = regular_node_count(
            node.pattern, max_quantifier_bound=max_quantifier_bound
        )
        return None if child_count is None else child_count + 1
    if isinstance(node, (ConcatenationPattern, AlternationPattern)):
        total = 1
        for child in node.patterns:
            child_count = regular_node_count(
                child, max_quantifier_bound=max_quantifier_bound
            )
            if child_count is None:
                return None
            total += child_count
        return total
    # Anchors, exclusions, and PERMUTE are intentional island boundaries.
    return None


def has_optimization_opportunity(node: PatternNode) -> bool:
    """Cheaply reject fragments with no ambiguity-producing structure."""

    repetitions = 0
    atoms = 0
    has_alternation = False

    def visit(current: PatternNode) -> None:
        nonlocal repetitions, atoms, has_alternation
        if isinstance(current, VariablePattern):
            atoms += 1
        elif isinstance(current, QuantifiedPattern):
            if current.atLeast != current.atMost:
                repetitions += 1
            visit(current.pattern)
        elif isinstance(current, AlternationPattern):
            has_alternation = True
            for child in current.patterns:
                visit(child)
        elif isinstance(current, ConcatenationPattern):
            for child in current.patterns:
                visit(child)

    visit(node)
    return has_alternation or repetitions >= 2 or (repetitions and atoms >= 2)


def needs_multi_pass_optimization(node: PatternNode) -> bool:
    """Whether separate alternatives can require successive simplifications.

    Repeated names alone are too broad: linear patterns such as ``A* C? A*``
    finish in one pass.  Multiple passes are reserved for quantified variables
    repeated under alternation.
    """

    quantified_names_under_alternation: Dict[str, int] = {}

    def visit(current: PatternNode, under_alternation: bool = False) -> None:
        if isinstance(current, AlternationPattern):
            for child in current.patterns:
                visit(child, True)
            return
        if isinstance(current, QuantifiedPattern):
            if under_alternation and isinstance(current.pattern, VariablePattern):
                name = current.pattern.name
                quantified_names_under_alternation[name] = (
                    quantified_names_under_alternation.get(name, 0) + 1
                )
            visit(current.pattern, under_alternation)
            return
        if isinstance(current, ConcatenationPattern):
            for child in current.patterns:
                visit(child, under_alternation)

    visit(node)
    return any(count > 1 for count in quantified_names_under_alternation.values())


def is_effective_reluctant_quantifier(node: PatternNode) -> bool:
    """Whether a quantifier has multiple counts ordered shortest-first."""

    return (
        isinstance(node, QuantifiedPattern)
        and not node.isGreedy
        and node.atLeast != node.atMost
    )


def contains_effective_reluctant_quantifier(node: PatternNode) -> bool:
    """Return whether any descendant has observable reluctant behavior."""

    if is_effective_reluctant_quantifier(node):
        return True
    if isinstance(node, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
        return contains_effective_reluctant_quantifier(node.pattern)
    return any(
        contains_effective_reluctant_quantifier(child)
        for child in getattr(node, "patterns", ())
    )


def is_reluctant_prefix_fragment(node: PatternNode) -> bool:
    """Recognize the certified ``X*? Y X*`` elimination shape."""

    if not isinstance(node, ConcatenationPattern) or len(node.patterns) != 3:
        return False
    first, middle, third = node.patterns
    return (
        isinstance(first, QuantifiedPattern)
        and isinstance(first.pattern, VariablePattern)
        and first.atLeast == 0
        and first.atMost is None
        and not first.isGreedy
        and isinstance(middle, VariablePattern)
        and isinstance(third, QuantifiedPattern)
        and isinstance(third.pattern, VariablePattern)
        and third.pattern.name == first.pattern.name
        and third.atLeast == 0
        and third.atMost is None
        and third.isGreedy
    )


def is_greedy_variable_plus(node: PatternNode) -> bool:
    return (
        isinstance(node, QuantifiedPattern)
        and isinstance(node.pattern, VariablePattern)
        and node.atLeast == 1
        and node.atMost is None
        and node.isGreedy
    )


def is_adjacent_plus_fragment(node: PatternNode) -> bool:
    return (
        isinstance(node, ConcatenationPattern)
        and len(node.patterns) == 2
        and all(is_greedy_variable_plus(child) for child in node.patterns)
    )


def contains_controlled_adjacent_plus_rewrite(node: PatternNode) -> bool:
    """Recognize the fixed output of the bounded ``A+ B+`` construction."""

    def is_right_branch(option: PatternNode) -> bool:
        return (
            isinstance(option, ConcatenationPattern)
            and len(option.patterns) == 2
            and isinstance(option.patterns[0], VariablePattern)
            and isinstance(option.patterns[1], QuantifiedPattern)
            and isinstance(option.patterns[1].pattern, VariablePattern)
            and option.patterns[1].atLeast == 0
            and option.patterns[1].atMost is None
            and option.patterns[1].isGreedy
        )

    def is_controlled_boundary(left: PatternNode, right: PatternNode) -> bool:
        if not is_greedy_variable_plus(left):
            return False
        if not isinstance(right, AlternationPattern) or len(right.patterns) != 2:
            return False
        return (
            is_right_branch(right.patterns[0])
            and isinstance(right.patterns[1], VariablePattern)
        ) or (
            isinstance(right.patterns[0], VariablePattern)
            and is_right_branch(right.patterns[1])
        )

    if isinstance(node, ConcatenationPattern):
        if any(
            is_controlled_boundary(left, right)
            for left, right in zip(node.patterns, node.patterns[1:])
        ):
            return True
        return any(
            contains_controlled_adjacent_plus_rewrite(child) for child in node.patterns
        )
    if isinstance(node, AlternationPattern):
        return any(
            contains_controlled_adjacent_plus_rewrite(child) for child in node.patterns
        )
    if isinstance(node, QuantifiedPattern):
        return contains_controlled_adjacent_plus_rewrite(node.pattern)
    return False
