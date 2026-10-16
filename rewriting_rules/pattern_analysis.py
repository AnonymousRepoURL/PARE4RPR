"""Reusable structural analysis over row-pattern ASTs."""

from typing import List, Optional

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


def enumerate_pattern_sequences(
    pattern: PatternNode,
    limit: int = 128,
) -> Optional[List[List[str]]]:
    """Enumerate finite variable sequences represented by a pattern."""
    if isinstance(pattern, VariablePattern):
        return [[pattern.name]]
    if isinstance(pattern, EmptyPattern):
        return [[]]
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return enumerate_pattern_sequences(pattern.pattern, limit)
    if isinstance(pattern, AlternationPattern):
        result: List[List[str]] = []
        for child in pattern.patterns:
            child_sequences = enumerate_pattern_sequences(child, limit)
            if child_sequences is None:
                return None
            result.extend(child_sequences)
            if len(result) > limit:
                return None
        return result
    if isinstance(pattern, ConcatenationPattern):
        result: List[List[str]] = [[]]
        for child in pattern.patterns:
            child_sequences = enumerate_pattern_sequences(child, limit)
            if child_sequences is None:
                return None
            next_result: List[List[str]] = []
            for prefix in result:
                for suffix in child_sequences:
                    next_result.append(prefix + suffix)
                    if len(next_result) > limit:
                        return None
            result = next_result
        return result
    if isinstance(pattern, QuantifiedPattern):
        if pattern.atMost is None:
            return None
        base_sequences = enumerate_pattern_sequences(pattern.pattern, limit)
        if base_sequences is None:
            return None
        result: List[List[str]] = []
        for repeat_count in range(pattern.atLeast, pattern.atMost + 1):
            repeated: List[List[str]] = [[]]
            for _ in range(repeat_count):
                next_repeated: List[List[str]] = []
                for prefix in repeated:
                    for suffix in base_sequences:
                        next_repeated.append(prefix + suffix)
                        if len(next_repeated) > limit:
                            return None
                repeated = next_repeated
            result.extend(repeated)
            if len(result) > limit:
                return None
        return result
    if isinstance(pattern, PermutationPattern):
        child_sequences = []
        for child in pattern.patterns:
            sequences = enumerate_pattern_sequences(child, limit)
            if sequences is None:
                return None
            child_sequences.append(sequences)

        result: List[List[str]] = []

        def backtrack(remaining_indexes: List[int], current: List[str]):
            if len(result) > limit:
                return
            if not remaining_indexes:
                result.append(list(current))
                return
            for index in remaining_indexes:
                next_remaining = [item for item in remaining_indexes if item != index]
                for sequence in child_sequences[index]:
                    backtrack(next_remaining, current + sequence)

        backtrack(list(range(len(child_sequences))), [])
        if len(result) > limit:
            return None
        return result
    return None
