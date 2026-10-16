"""
This file aims to parse and analyze pattern subclauses,
and it supports all semantics in the SQL standard.

When parsing pattern, we will build a tree structure to represent the pattern.
Besdies, we also provide some optimization rules to simplify the pattern tree,
which can help improve the performance of pattern matching.

Currently, patternnode can only optimze some simple pattern, e.g.,
nested quantifiers, optional patterns, and alternation patterns.
Example:
    (A+ (A+)? B) ==> (A+ B)
    (A+ A+ A+ A) ==> (A{4,})
However, it cannot optimze (A+B+)+ to A(A|B)*B,
consider the following example:
------------------------------------
SELECT * FROM TA MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES CLASSIFIER() AS CLS
    PATTERN ((A+B+)+)
    DEFINE
        A AS VAL = 1,
        B AS TRUE
)
------------------------------------
row:        1 2 3 4
val:        1 3 1 3
classifier: A B B B     [for (A+B+)+]
classifier: A B A B     [for A(A|B)*B]

when we need reference classifier label, we cannot use A(A|B)*B,
because it will change the classifier label.
"""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple
from enum import Enum


class AnchorType(Enum):
    START = "^"  # (^A B C)
    END = "$"  # (A B C$)


_analysis_cache: ContextVar[Optional[dict]] = ContextVar(
    "pattern_analysis_cache", default=None
)
_simplification_active: ContextVar[bool] = ContextVar(
    "pattern_simplification_active", default=False
)


def _memoized_analysis(method):
    """Memoize recursive property queries for one top-level call only.

    The cache is discarded when the outermost query returns. Pattern trees are
    mutable during rewriting, so this deliberately never persists stale data
    across optimization operations.
    """

    @wraps(method)
    def wrapper(self, *args):
        cache = _analysis_cache.get()
        owns_cache = cache is None
        token = None
        if owns_cache:
            cache = {}
            token = _analysis_cache.set(cache)

        key = (method.__name__, id(self), args)
        try:
            if key not in cache:
                cache[key] = method(self, *args)
            value = cache[key]
            return value.copy() if isinstance(value, set) else value
        finally:
            if owns_cache:
                _analysis_cache.reset(token)

    return wrapper


def _fixed_point_simplification(method):
    """Make a default top-level ``simplify()`` call reach a normal form.

    Individual node methods remain single-pass rewrite functions while a
    simplification is already active.  Only the outermost default call repeats
    whole-tree passes.  This distinction preserves contextual calls such as
    ``QuantifiedPattern.simplify(fold_optional_plus=False)`` while ensuring
    that nodes created by a parent rewrite are simplified before returning.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Non-default arguments express a contextual single-pass operation.
        if _simplification_active.get() or args or kwargs:
            return method(self, *args, **kwargs)

        # Anchors are zero-width assertions whose position is semantic.  Check
        # their placement before any rewrite can discard grouping information.
        _validate_anchor_placement(self)

        token = _simplification_active.set(True)
        try:
            current = self
            seen = set()
            while True:
                current_key = _pattern_key(current)
                if current_key in seen:
                    raise RuntimeError(
                        "Pattern simplification entered a structural rewrite cycle"
                    )
                seen.add(current_key)

                # Dispatch on the current node's concrete type.  Its wrapper
                # observes the active context and performs exactly one pass.
                simplified = current.simplify()
                _validate_anchor_placement(simplified)
                if _pattern_key(simplified) == current_key:
                    return current
                current = simplified
        finally:
            _simplification_active.reset(token)

    wrapper._fixed_point_simplified = True
    return wrapper


class PatternNode(ABC):

    _ANALYSIS_METHODS = {
        "get_var_num",
        "get_min_row_num",
        "get_max_row_num",
        "get_leftmost_variables",
        "get_rightmost_variables",
        "get_must_appear_var",
        "get_left_max_row_num",
        "get_right_max_row_num",
        "get_left_min_row_num",
        "get_right_min_row_num",
        "can_be_skipped",
        "get_variables",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in PatternNode._ANALYSIS_METHODS:
            method = cls.__dict__.get(name)
            if method is not None and not getattr(method, "_analysis_memoized", False):
                wrapped = _memoized_analysis(method)
                wrapped._analysis_memoized = True
                setattr(cls, name, wrapped)

        simplify = cls.__dict__.get("simplify")
        if simplify is not None and not getattr(
            simplify, "_fixed_point_simplified", False
        ):
            setattr(cls, "simplify", _fixed_point_simplification(simplify))

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def get_var_num(self) -> int:
        pass

    @abstractmethod
    def get_min_row_num(self) -> int:
        pass

    @abstractmethod
    def get_max_row_num(self) -> int:
        pass

    @abstractmethod
    def get_leftmost_variables(self) -> Set[str]:
        """Retrieve all variables that may appear on the far left of the pattern"""
        pass

    @abstractmethod
    def get_rightmost_variables(self) -> Set[str]:
        """Retrieve all variables that may appear on the far right of the pattern"""
        pass

    @abstractmethod
    def get_must_appear_var(self) -> Set[str]:
        """Obtain variables that are certain to appear in the result."""
        pass

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Return every primary pattern variable in this subtree."""
        pass

    @abstractmethod
    def get_left_max_row_num(self, var: str) -> Optional[int]:
        """Get maximum rows left of var, or None if var is not guaranteed.

        ``None`` also represents an unbounded maximum.
        """
        pass

    @abstractmethod
    def get_right_max_row_num(self, var: str) -> Optional[int]:
        """Get maximum rows right of var, or None if var is not guaranteed.

        ``None`` also represents an unbounded maximum.
        """
        pass

    @abstractmethod
    def get_left_min_row_num(self, var: str) -> Optional[int]:
        """Get minimum rows to the left of the first occurrence of var.
        Returns None if var does not appear in this pattern."""
        pass

    @abstractmethod
    def get_right_min_row_num(self, var: str) -> Optional[int]:
        """Get minimum rows to the right of the last occurrence of var.
        Returns None if var does not appear in this pattern."""
        pass

    def can_be_skipped(self) -> bool:
        """Determine whether this pattern can be skipped (not appear), with the default requirement to appear"""
        return False

    def can_be_quantified(self) -> bool:
        """Whether this pattern can safely be wrapped by another quantifier."""
        return self.get_min_row_num() > 0 and not _contains_anchor_pattern(self)

    @_memoized_analysis
    def get_must_middle_var(self) -> Set[str]:
        """Obtain variables that are certain to appear in the middle (not leftmost or rightmost)"""
        leftmost = self.get_leftmost_variables()
        rightmost = self.get_rightmost_variables()
        must_appear = self.get_must_appear_var()
        return must_appear - leftmost - rightmost


def _union_sets(sets: Iterable[Set[str]]) -> Set[str]:
    result: Set[str] = set()
    for item in sets:
        result.update(item)
    return result


def _sum_max_rows(patterns: Iterable[PatternNode]) -> Optional[int]:
    total = 0
    for pattern in patterns:
        row_num = pattern.get_max_row_num()
        if row_num is None:
            return None
        total += row_num
    return total


def _sum_min_rows(patterns: Iterable[PatternNode]) -> int:
    return sum(pattern.get_min_row_num() for pattern in patterns)


def _contains_variable(pattern: PatternNode, var: str) -> bool:
    return var in pattern.get_variables()


def _can_match_variable(pattern: PatternNode, var: str) -> bool:
    """Return whether some match of ``pattern`` can produce ``var``."""
    if isinstance(pattern, VariablePattern):
        return pattern.name == var
    if isinstance(pattern, EmptyPattern):
        return False
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return _can_match_variable(pattern.pattern, var)
    if isinstance(pattern, QuantifiedPattern):
        return pattern.atMost != 0 and _can_match_variable(pattern.pattern, var)
    if isinstance(
        pattern,
        (
            ConcatenationPattern,
            PermutationPattern,
            AlternationPattern,
        ),
    ):
        return any(_can_match_variable(child, var) for child in pattern.patterns)
    return _contains_variable(pattern, var)


def _min_rows_without_variable(
    pattern: PatternNode,
    var: str,
) -> Tuple[bool, Optional[int]]:
    """Return whether *pattern* can omit *var* and its minimum such length.

    The existence flag distinguishes an impossible no-*var* path from a
    zero-row path.  This distinction is needed when a concatenation searches
    past an earlier child that can either produce *var* or omit it entirely.
    """
    if isinstance(pattern, VariablePattern):
        return (False, None) if pattern.name == var else (True, 1)
    if isinstance(pattern, EmptyPattern):
        return True, 0
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return _min_rows_without_variable(pattern.pattern, var)

    if isinstance(pattern, (ConcatenationPattern, PermutationPattern)):
        total = 0
        for child in pattern.patterns:
            exists, child_min = _min_rows_without_variable(child, var)
            if not exists:
                return False, None
            # An existing minimum is always finite because row counts are
            # non-negative and every pattern has a finite minimum length.
            assert child_min is not None
            total += child_min
        return True, total

    if isinstance(pattern, AlternationPattern):
        candidates = []
        for child in pattern.patterns:
            exists, child_min = _min_rows_without_variable(child, var)
            if exists:
                assert child_min is not None
                candidates.append(child_min)
        return (True, min(candidates)) if candidates else (False, None)

    if isinstance(pattern, QuantifiedPattern):
        if pattern.atLeast == 0:
            return True, 0
        exists, child_min = _min_rows_without_variable(pattern.pattern, var)
        if not exists:
            return False, None
        assert child_min is not None
        return True, pattern.atLeast * child_min

    if not _contains_variable(pattern, var):
        return True, pattern.get_min_row_num()
    return False, None


def _max_rows_without_variable(
    pattern: PatternNode,
    var: str,
) -> Tuple[bool, Optional[int]]:
    """Return whether *pattern* can omit *var* and its maximum such length.

    The row count is ``None`` when the no-*var* length is unbounded.  Keeping
    the existence flag separate is important: ``None`` is also used by the
    public side-distance APIs when a variable is absent, optional, or has an
    unbounded side distance.
    """
    if isinstance(pattern, VariablePattern):
        return (False, None) if pattern.name == var else (True, 1)
    if isinstance(pattern, EmptyPattern):
        return True, 0
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return _max_rows_without_variable(pattern.pattern, var)

    if isinstance(pattern, (ConcatenationPattern, PermutationPattern)):
        total = 0
        unbounded = False
        for child in pattern.patterns:
            exists, child_max = _max_rows_without_variable(child, var)
            if not exists:
                return False, None
            if child_max is None:
                unbounded = True
            elif not unbounded:
                total += child_max
        return True, None if unbounded else total

    if isinstance(pattern, AlternationPattern):
        found = False
        best = 0
        for child in pattern.patterns:
            exists, child_max = _max_rows_without_variable(child, var)
            if not exists:
                continue
            found = True
            if child_max is None:
                return True, None
            best = max(best, child_max)
        return found, best if found else None

    if isinstance(pattern, QuantifiedPattern):
        if pattern.atMost == 0:
            return True, 0
        exists, child_max = _max_rows_without_variable(pattern.pattern, var)
        if not exists:
            return (True, 0) if pattern.atLeast == 0 else (False, None)
        if child_max is None:
            return True, None
        if pattern.atMost is None:
            return (True, None) if child_max > 0 else (True, 0)
        return True, pattern.atMost * child_max

    if not _contains_variable(pattern, var):
        return True, pattern.get_max_row_num()
    return False, None


def _conditional_side_max_row_num(
    pattern: PatternNode,
    var: str,
    side: str,
) -> Tuple[bool, Optional[int]]:
    """Return whether ``var`` can occur and its side maximum when it does.

    Unlike the public APIs, this internal result does not discard the matches
    containing ``var`` merely because another match can omit it.  The separate
    existence flag keeps an absent variable distinct from an unbounded side.
    """
    if isinstance(pattern, VariablePattern):
        return (True, 0) if pattern.name == var else (False, None)
    if isinstance(pattern, EmptyPattern):
        return False, None
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return _conditional_side_max_row_num(pattern.pattern, var, side)

    if isinstance(pattern, QuantifiedPattern):
        if pattern.atMost == 0:
            return False, None
        found, internal = _conditional_side_max_row_num(
            pattern.pattern,
            var,
            side,
        )
        if not found or internal is None:
            return found, internal

        can_omit, no_var_max = _max_rows_without_variable(pattern.pattern, var)
        if not can_omit or pattern.atMost == 1:
            return True, internal
        if no_var_max is None:
            return True, None
        if pattern.atMost is None:
            return True, None if no_var_max > 0 else internal
        return True, internal + (pattern.atMost - 1) * no_var_max

    if isinstance(pattern, AlternationPattern):
        found = False
        best = 0
        for child in pattern.patterns:
            child_found, child_max = _conditional_side_max_row_num(
                child,
                var,
                side,
            )
            if not child_found:
                continue
            found = True
            if child_max is None:
                return True, None
            best = max(best, child_max)
        return found, best if found else None

    if isinstance(pattern, ConcatenationPattern):
        children = pattern.patterns if side == "left" else reversed(pattern.patterns)
        outside_max: Optional[int] = 0
        found = False
        best = 0

        for child in children:
            child_found, internal = _conditional_side_max_row_num(
                child,
                var,
                side,
            )
            if child_found:
                found = True
                if outside_max is None or internal is None:
                    return True, None
                best = max(best, outside_max + internal)

            can_omit, no_var_max = _max_rows_without_variable(child, var)
            if not can_omit:
                break
            if outside_max is not None:
                outside_max = None if no_var_max is None else outside_max + no_var_max

        return found, best if found else None

    if isinstance(pattern, PermutationPattern):
        found = False
        best = 0
        for index, child in enumerate(pattern.patterns):
            child_found, internal = _conditional_side_max_row_num(
                child,
                var,
                side,
            )
            if not child_found:
                continue
            found = True
            if internal is None:
                return True, None

            candidate = internal
            for other_index, other in enumerate(pattern.patterns):
                if other_index == index:
                    continue
                can_omit, no_var_max = _max_rows_without_variable(other, var)
                if not can_omit:
                    continue
                if no_var_max is None:
                    return True, None
                candidate += no_var_max
            best = max(best, candidate)
        return found, best if found else None

    if not _can_match_variable(pattern, var):
        return False, None
    value = (
        pattern.get_left_max_row_num(var)
        if side == "left"
        else pattern.get_right_max_row_num(var)
    )
    return True, value


def _guaranteed_side_max_row_num(
    pattern: PatternNode,
    var: str,
    side: str,
) -> Optional[int]:
    """Expose a side maximum only when every match contains ``var``."""
    can_omit, _ = _max_rows_without_variable(pattern, var)
    if can_omit:
        return None
    found, maximum = _conditional_side_max_row_num(pattern, var, side)
    return maximum if found else None


def _unique_permutations(
    patterns: List[PatternNode], keys: Optional[List[object]] = None
):
    keys = keys or [pattern.to_string() for pattern in patterns]
    used = [False] * len(patterns)
    current: List[PatternNode] = []

    def backtrack():
        if len(current) == len(patterns):
            yield list(current)
            return

        seen_at_depth = set()
        for index, pattern in enumerate(patterns):
            if used[index]:
                continue

            key = keys[index]
            if key in seen_at_depth:
                continue

            seen_at_depth.add(key)
            used[index] = True
            current.append(pattern)
            yield from backtrack()
            current.pop()
            used[index] = False

    yield from backtrack()


class _DelegatingPattern(PatternNode):
    """Base class for wrappers that do not change row-count semantics."""

    pattern: PatternNode

    def get_var_num(self) -> int:
        return self.pattern.get_var_num()

    def get_min_row_num(self) -> int:
        return self.pattern.get_min_row_num()

    def get_max_row_num(self) -> Optional[int]:
        return self.pattern.get_max_row_num()

    def get_leftmost_variables(self) -> Set[str]:
        return self.pattern.get_leftmost_variables()

    def get_rightmost_variables(self) -> Set[str]:
        return self.pattern.get_rightmost_variables()

    def get_must_appear_var(self) -> Set[str]:
        return self.pattern.get_must_appear_var()

    def get_variables(self) -> Set[str]:
        return self.pattern.get_variables()

    def can_be_skipped(self) -> bool:
        return self.pattern.can_be_skipped()

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        return self.pattern.get_left_max_row_num(var)

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        return self.pattern.get_right_max_row_num(var)

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        return self.pattern.get_left_min_row_num(var)

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        return self.pattern.get_right_min_row_num(var)


@dataclass
class ExcludedPattern(_DelegatingPattern):
    """
    format: {- pattern_node -}
    note that if output mode is one row per match, ExcludedPattern may be changed to a normal pattern, because it will not affect the output.
    i.e., {- pattern_node -} ==> pattern_node
    """

    pattern: "PatternNode"

    def to_string(self) -> str:
        return f"{{- {self.pattern.to_string()} -}}"

    def simplify(self) -> "PatternNode":
        simplified_pattern = self.pattern.simplify()

        # If the inner pattern is also an ExcludedPattern, unwrap it
        # {- A {-B-} C -} => {- A B C -}
        if isinstance(simplified_pattern, ExcludedPattern):
            inner_pattern = simplified_pattern.pattern
            return ExcludedPattern(inner_pattern)

        # If the inner pattern is a PatternConcatenation, recursively unwrap nested ExcludedPatterns
        if isinstance(simplified_pattern, ConcatenationPattern):
            unwrapped_patterns = []
            for p in simplified_pattern.patterns:
                # If an element is an ExcludedPattern, unwrap it
                if isinstance(p, ExcludedPattern):
                    unwrapped_patterns.append(p.pattern)
                else:
                    unwrapped_patterns.append(p)

            # If any patterns were unwrapped, create a new concatenation
            if unwrapped_patterns != simplified_pattern.patterns:
                new_concat = ConcatenationPattern(unwrapped_patterns)
                return ExcludedPattern(new_concat)

        # If the pattern changed, return a new ExcludedPattern with simplified content
        if simplified_pattern is not self.pattern:
            return ExcludedPattern(simplified_pattern)

        return self


@dataclass
class VariablePattern(PatternNode):
    """variable name, e.g., A"""

    name: str

    def to_string(self) -> str:
        return self.name

    def get_var_num(self) -> int:
        return 1

    def get_min_row_num(self) -> int:
        return 1

    def get_max_row_num(self) -> Optional[int]:
        return 1

    def get_leftmost_variables(self) -> Set[str]:
        return {self.name}

    def get_rightmost_variables(self) -> Set[str]:
        return {self.name}

    def get_must_appear_var(self) -> Set[str]:
        """A variable must appear exactly once"""
        return {self.name}

    def get_variables(self) -> Set[str]:
        return {self.name}

    def can_be_skipped(self) -> bool:
        """A variable always requires 1 row, cannot be skipped"""
        return False

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        """Variable itself has no left side"""
        if self.name == var:
            return 0
        return None

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        """Variable itself has no right side"""
        if self.name == var:
            return 0
        return None

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        return 0 if self.name == var else None

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        return 0 if self.name == var else None

    def simplify(self) -> "PatternNode":
        return self


def _variable_sequence(pattern: PatternNode) -> Optional[List[str]]:
    """Return a plain variable sequence, or None for a compound pattern."""
    if isinstance(pattern, VariablePattern):
        return [pattern.name]
    if isinstance(pattern, ConcatenationPattern):
        if all(isinstance(item, VariablePattern) for item in pattern.patterns):
            return [item.name for item in pattern.patterns]
    return None


def _simplify_unbounded_alternation_concatenations(pattern: PatternNode) -> PatternNode:
    """Simplify branches under a greedy, unbounded repetition.

    The caller must establish both preconditions.  In particular, this rewrite
    is not preference-preserving for reluctant repetition.
    """
    if not isinstance(pattern, AlternationPattern):
        return pattern

    previous_sequences = {}
    optimized_patterns: List[PatternNode] = []

    def can_concatenate_from_left(sequence: List[str]) -> bool:
        sequence_tuple = tuple(sequence)
        reachable = {0}
        for index in range(len(sequence)):
            if index not in reachable:
                continue
            for previous in previous_sequences.get(sequence[index], []):
                end = index + len(previous)
                if sequence_tuple[index:end] == previous:
                    reachable.add(end)
        return len(sequence) in reachable

    for branch in pattern.patterns:
        sequence = _variable_sequence(branch)
        if sequence and can_concatenate_from_left(sequence):
            continue

        optimized_patterns.append(branch)
        if sequence:
            previous_sequences.setdefault(sequence[0], []).append(tuple(sequence))

    if len(optimized_patterns) == len(pattern.patterns):
        return pattern
    return AlternationPattern(optimized_patterns).simplify()


def _nested_quantifier_range(
    inner: "QuantifiedPattern",
    outer: "QuantifiedPattern",
) -> Optional[tuple[int, Optional[int]]]:
    """Calculate a preference-preserving flattened repetition interval.

    Quantifier priority is recursive rather than globally length-based.  At
    every optional boundary a greedy quantifier tries ``repeat`` before
    ``exit`` (and a reluctant one does the reverse); choices inside that
    repetition are made before the next boundary is considered.

    A primary variable has one classifier trace per repetition count, so only
    the count order matters.  A compound body can have several traces.  For a
    greedy ``(P{1,n}){x,y}`` with ``x`` equal to zero or one, however, every
    boundary still presents the same ordered choice as ``P{x,ny}``: continue
    with the next P first, then exit.  The analogous reluctant equivalence
    holds only when the outer quantifier is optional once.  Other compound
    cases retain their grouping.
    """
    if inner.isGreedy != outer.isGreedy:
        return None

    m, n = inner.atLeast, inner.atMost
    x, y = outer.atLeast, outer.atMost
    only_one_outer_count = y is not None and x == y

    if isinstance(inner.pattern, VariablePattern):
        # For m > 1, a variable has only one trace for each total count, but
        # nested backtracking does not visit those counts monotonically.  For
        # example, (A{2,3}){1,2} tries total count 3 before 4, whereas A{2,6}
        # tries 4 before 3 when a suffix leaves both choices viable.  A fixed
        # outer count retains monotone sum order; m == 1 does so for a variable
        # number of outer repetitions as well.
        if m != 1 and not only_one_outer_count:
            return None
    else:
        if inner.atLeast != 1:
            return None
        if inner.isGreedy:
            if outer.atLeast > 1:
                return None
        elif outer.atMost != 1:
            return None

    # With multiple outer counts, adjacent intervals are contiguous iff
    # x(n-m) >= m-1.  A single outer count is already one interval.  An
    # unbounded n has infinite width when x > 0; when x == 0 the left side
    # remains zero rather than 0 * infinity.
    has_contiguous_counts = only_one_outer_count or (
        x * (n - m) >= m - 1 if n is not None else x > 0 or m <= 1
    )
    if not has_contiguous_counts:
        return None

    lower = m * x
    if y == 0:
        upper = 0
    elif n is None or y is None:
        upper = None
    else:
        upper = n * y
    return lower, upper


@dataclass
class QuantifiedPattern(PatternNode):
    """
    QuantifiedPattern:
    we unify the kleene and multiple operator
    A{1,5}?, B{2,}*, C{3}+
    """

    pattern: PatternNode
    atLeast: int  # minimum occurrences
    atMost: Optional[int]  # `None` means infinity, maximum occurrences
    isGreedy: bool  # true then try matching as many rows as possible

    def __post_init__(self) -> None:
        """Keep quantifier bounds valid for parser and rewrite-rule callers."""
        if self.atLeast < 0:
            raise ValueError("Pattern quantifier lower bound must not be negative")
        if self.atMost is not None:
            if self.atMost <= 0:
                raise ValueError(
                    "Pattern quantifier upper bound must be greater than 0"
                )
            if self.atLeast > self.atMost:
                raise ValueError(
                    "Pattern quantifier lower bound must not exceed upper bound"
                )

    def to_string(self) -> str:
        if self.atMost is None:
            if self.atLeast == 0:
                quantifier = "*"
            elif self.atLeast == 1:
                quantifier = "+"
            else:
                quantifier = f"{{{self.atLeast},}}"
        else:
            if self.atLeast == 0 and self.atMost == 1:
                quantifier = "?"
            elif self.atLeast == self.atMost:
                quantifier = f"{{{self.atLeast}}}"
            else:
                quantifier = f"{{{self.atLeast},{self.atMost}}}"

        # A fixed-count quantifier has no alternative match length, so its
        # greedy/reluctant modifier is semantically irrelevant.
        if not self.isGreedy and self.atLeast != self.atMost:
            quantifier += "?"

        # we need this branch to output: (A B)+, (A | B)+, (A+)*
        if isinstance(
            self.pattern,
            (
                PermutationPattern,
                AlternationPattern,
                ConcatenationPattern,
                QuantifiedPattern,
                AnchorPattern,
            ),
        ):
            return f"({self.pattern.to_string()}){quantifier}"
        else:
            return f"{self.pattern.to_string()}{quantifier}"

    def get_var_num(self) -> int:
        return self.pattern.get_var_num()

    def get_min_row_num(self) -> int:
        return self.atLeast * self.pattern.get_min_row_num()

    def get_max_row_num(self) -> Optional[int]:
        if self.atMost is None:
            return None
        pattern_max = self.pattern.get_max_row_num()
        if pattern_max is None:
            return None
        return self.atMost * pattern_max

    def can_be_skipped(self) -> bool:
        """If it occurs at least 0 times, it can be skipped"""
        return self.atLeast == 0

    def get_leftmost_variables(self) -> Set[str]:
        return self.pattern.get_leftmost_variables()

    def get_rightmost_variables(self) -> Set[str]:
        return self.pattern.get_rightmost_variables()

    def get_must_appear_var(self) -> Set[str]:
        """If minimum occurrences >= 1, variables must appear; otherwise cannot guarantee"""
        if self.atLeast >= 1:
            return self.pattern.get_must_appear_var()
        else:
            return set()

    def get_variables(self) -> Set[str]:
        return self.pattern.get_variables()

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "left")

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "right")

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        # First occurrence is in the first iteration; 0 prefix rows from prior iterations.
        return self.pattern.get_left_min_row_num(var)

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        # Last occurrence is in the last iteration; rows after var = inner right min.
        return self.pattern.get_right_min_row_num(var)

    def simplify(self, fold_optional_plus: bool = True) -> "PatternNode":
        simplified_pattern = self.pattern.simplify()

        # Check if the quantified pattern can have empty matches (BEFORE any special optimizations)
        if simplified_pattern.get_min_row_num() == 0:
            raise ValueError(
                "Quantified subpatterns that can have empty matches are not yet supported"
            )

        # Repeating a pattern exactly once adds no language or preference
        # semantics.  Drop the redundant wrapper so generated A{1} (and
        # equivalent compound forms such as (A B){1}) use the canonical
        # unquantified representation.
        if self.atLeast == 1 and self.atMost == 1:
            return simplified_pattern

        # {m}? and {m} describe exactly the same single repetition count.
        # Canonicalize the AST as well as its string representation so later
        # optimization passes do not treat them as distinct alternatives.
        if self.atMost == self.atLeast and not self.isGreedy:
            return QuantifiedPattern(
                simplified_pattern, self.atLeast, self.atMost, isGreedy=True
            )

        # Removing a branch that can be assembled from earlier alternatives is
        # safe only for greedy unbounded repetition.  Greedy repetition tries
        # the equivalent multi-iteration spelling first.  Reluctant repetition
        # does the opposite: a branch such as ``A B`` is tried as one iteration
        # before ``A`` followed by ``B`` as two iterations.  Dropping that
        # branch therefore changes the selected classifier trace even though
        # the accepted classifier language stays the same.
        if self.atMost is None and self.isGreedy:
            simplified_pattern = _simplify_unbounded_alternation_concatenations(
                simplified_pattern
            )

        if isinstance(simplified_pattern, QuantifiedPattern):
            inner = simplified_pattern

            # (A+)? => A* when both quantifiers are greedy. Alternation delays
            # this so A+ | (A+)? can first preserve branch priority as (A+)?.
            if (
                fold_optional_plus
                and self.isGreedy
                and inner.isGreedy
                and self.atLeast == 0
                and self.atMost == 1
                and inner.atLeast == 1
                and inner.atMost is None
            ):
                return QuantifiedPattern(inner.pattern, 0, None, True)
            # (A{m,n}){x,y} => A{mx,ny}; None denotes an unbounded n or y.
            merged_range = _nested_quantifier_range(inner, self)
            if merged_range is not None:
                new_at_least, new_at_most = merged_range
                return QuantifiedPattern(
                    inner.pattern, new_at_least, new_at_most, self.isGreedy
                )

        if simplified_pattern is not self.pattern:
            return QuantifiedPattern(
                simplified_pattern, self.atLeast, self.atMost, self.isGreedy
            )
        return self


@dataclass
class PermutationPattern(PatternNode):
    """format example: PERMUTE(A B C)"""

    patterns: List[PatternNode]

    def to_string(self) -> str:
        inner = ", ".join(p.to_string() for p in self.patterns)
        return f"PERMUTE({inner})"

    def get_var_num(self) -> int:
        return sum(p.get_var_num() for p in self.patterns)

    def get_min_row_num(self) -> int:
        return _sum_min_rows(self.patterns)

    def get_max_row_num(self) -> Optional[int]:
        return _sum_max_rows(self.patterns)

    def get_leftmost_variables(self) -> Set[str]:
        return _union_sets(p.get_leftmost_variables() for p in self.patterns)

    def get_rightmost_variables(self) -> Set[str]:
        return _union_sets(p.get_rightmost_variables() for p in self.patterns)

    def get_must_appear_var(self) -> Set[str]:
        """All variables in PERMUTE must appear"""
        return _union_sets(p.get_must_appear_var() for p in self.patterns)

    def get_variables(self) -> Set[str]:
        return _union_sets(p.get_variables() for p in self.patterns)

    def can_be_skipped(self) -> bool:
        """PERMUTE can be skipped if its minimum row count is 0"""
        return self.get_min_row_num() == 0

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "left")

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "right")

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        # The element containing var can be placed first, but rows preceding
        # var *inside that element* cannot be reordered away.
        valid = [
            distance
            for distance in (p.get_left_min_row_num(var) for p in self.patterns)
            if distance is not None
        ]
        return min(valid) if valid else None

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        # Symmetrically, the containing element can be placed last while its
        # own internal suffix remains mandatory.
        valid = [
            distance
            for distance in (p.get_right_min_row_num(var) for p in self.patterns)
            if distance is not None
        ]
        return min(valid) if valid else None

    def simplify(self) -> "PatternNode":
        simplified_patterns = [p.simplify() for p in self.patterns]
        simplified_patterns = [
            p for p in simplified_patterns if not isinstance(p, EmptyPattern)
        ]
        if not simplified_patterns:
            raise ValueError("PERMUTE cannot contain only empty patterns")
        if len(simplified_patterns) > 7:
            raise ValueError("PERMUTE cannot contain more than 7 elements")
        if len(simplified_patterns) == 1:
            return simplified_patterns[0]

        pattern_keys = [_pattern_key(p) for p in simplified_patterns]
        unique_keys = set(pattern_keys)
        # has_duplicates = len(unique_keys) < len(simplified_patterns)

        if len(unique_keys) == 1:
            return ConcatenationPattern(simplified_patterns).simplify()

        # if has_duplicates and len(unique_keys) <= 3:
        if len(unique_keys) <= 3:
            expanded_patterns = []
            seen_expansions = set()
            for ordering in _unique_permutations(simplified_patterns, pattern_keys):
                expanded = ConcatenationPattern(ordering).simplify()
                expanded_key = _pattern_key(expanded)
                if expanded_key in seen_expansions:
                    continue
                expanded_patterns.append(expanded)
                seen_expansions.add(expanded_key)
            if len(expanded_patterns) == 1:
                return expanded_patterns[0]
            return AlternationPattern(expanded_patterns).simplify()

        if simplified_patterns != self.patterns:
            return PermutationPattern(simplified_patterns)
        return self


@dataclass
class AlternationPattern(PatternNode):
    """
    Alternation pattern, e.g., A | B | C
    Optimization:
        1. Remove duplicates from left to right
        2. Handle optional patterns
        3. Handle explicit EmptyPattern
        4. Merge optional bases with explicit EmptyPattern
    Example:
    A | B | A => A | B
    A | A? => A | A | () ==> A?
    A?? B | A ==> () | A | B | A ==> () | A | B ==> A?? B
    A | () => A?, () | A => A??
    """

    patterns: List[PatternNode]

    def to_string(self) -> str:
        return " | ".join(p.to_string() for p in self.patterns)

    def get_var_num(self) -> int:
        return sum(p.get_var_num() for p in self.patterns)

    def get_min_row_num(self) -> int:
        if not self.patterns:
            return 0
        return min(p.get_min_row_num() for p in self.patterns)

    def get_max_row_num(self) -> Optional[int]:
        ans = 0
        for p in self.patterns:
            row_num = p.get_max_row_num()
            if row_num is None:
                return None
            ans = max(ans, row_num)
        return ans

    def get_leftmost_variables(self) -> Set[str]:
        """All variables that may appear on the left side of any branch"""
        return _union_sets(p.get_leftmost_variables() for p in self.patterns)

    def get_rightmost_variables(self) -> Set[str]:
        """All variables that may appear on the right side of any branch"""
        return _union_sets(p.get_rightmost_variables() for p in self.patterns)

    def get_must_appear_var(self) -> Set[str]:
        """Only variables that appear in ALL branches must appear"""
        if not self.patterns:
            return set()
        # Start with the first pattern's must-appear variables
        result = self.patterns[0].get_must_appear_var().copy()
        for p in self.patterns[1:]:
            result &= p.get_must_appear_var()
        return result

    def get_variables(self) -> Set[str]:
        return _union_sets(p.get_variables() for p in self.patterns)

    def can_be_skipped(self) -> bool:
        """Alternation can be skipped if its minimum row count is 0"""
        return self.get_min_row_num() == 0

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "left")

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        return _guaranteed_side_max_row_num(self, var, "right")

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        # Use the minimum across all branches (worst-case: branch with fewest inside rows).
        valid = [
            r
            for r in (p.get_left_min_row_num(var) for p in self.patterns)
            if r is not None
        ]
        return min(valid) if valid else None

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        valid = [
            r
            for r in (p.get_right_min_row_num(var) for p in self.patterns)
            if r is not None
        ]
        return min(valid) if valid else None

    def simplify(self) -> "PatternNode":
        simplified_patterns = [
            (
                p.simplify(fold_optional_plus=False)
                if isinstance(p, QuantifiedPattern)
                else p.simplify()
            )
            for p in self.patterns
        ]

        # Alternation is associative under the engine's left-to-right branch
        # priority: (A | B) | C and A | (B | C) both try A, then B, then C.
        # Flatten before duplicate/range handling so equivalent parenthesized
        # inputs reach the same normal form in the first parse.  Wrappers such
        # as anchors, exclusions and quantifiers remain explicit boundaries.
        flattened_patterns = []
        nested_alternations_flattened = False
        for pattern in simplified_patterns:
            if isinstance(pattern, AlternationPattern):
                flattened_patterns.extend(pattern.patterns)
                nested_alternations_flattened = True
            else:
                flattened_patterns.append(pattern)
        simplified_patterns = flattened_patterns

        def _finalize_quantified_result(p: PatternNode) -> PatternNode:
            if isinstance(p, QuantifiedPattern):
                return p.simplify()
            return p

        if len(simplified_patterns) == 1:
            return _finalize_quantified_result(simplified_patterns[0])

        # simplify optional pattern alternation: X | () => X? (greedy), () | X => X?? (reluctant)
        if len(simplified_patterns) == 2:
            first, second = simplified_patterns
            if isinstance(first, EmptyPattern) and not isinstance(second, EmptyPattern):
                if second.can_be_quantified():
                    return _finalize_quantified_result(
                        QuantifiedPattern(second, 0, 1, isGreedy=False)
                    )
                return AlternationPattern(simplified_patterns)
            if not isinstance(first, EmptyPattern) and isinstance(second, EmptyPattern):
                if first.can_be_quantified():
                    return _finalize_quantified_result(
                        QuantifiedPattern(first, 0, 1, isGreedy=True)
                    )
                return AlternationPattern(simplified_patterns)

        def _fixed_variable_count(p: PatternNode):
            if isinstance(p, EmptyPattern):
                return (None, 0)
            if isinstance(p, VariablePattern):
                return (p.name, 1)
            if (
                isinstance(p, QuantifiedPattern)
                and isinstance(p.pattern, VariablePattern)
                and p.atMost == p.atLeast
            ):
                return (p.pattern.name, p.atLeast)
            return None

        def _fold_fixed_count_runs(patterns: List[PatternNode]):
            fixed_counts = [_fixed_variable_count(p) for p in patterns]
            next_fixed_vars: List[Optional[str]] = [None] * (len(patterns) + 1)
            for idx in range(len(patterns) - 1, -1, -1):
                fixed = fixed_counts[idx]
                if fixed is not None and fixed[0] is not None:
                    next_fixed_vars[idx] = fixed[0]
                else:
                    next_fixed_vars[idx] = next_fixed_vars[idx + 1]

            folded = []
            changed = False
            i = 0
            while i < len(patterns):
                fixed = fixed_counts[i]
                if fixed is None:
                    folded.append(patterns[i])
                    i += 1
                    continue

                var_name = fixed[0]
                if var_name is None:
                    var_name = next_fixed_vars[i + 1]
                if var_name is None:
                    folded.append(patterns[i])
                    i += 1
                    continue

                run = []
                counts = []
                j = i
                while j < len(patterns):
                    current_fixed = fixed_counts[j]
                    if current_fixed is None:
                        break
                    current_var, current_count = current_fixed
                    if current_var is not None and current_var != var_name:
                        break
                    if current_var is None and j > i:
                        next_var = next_fixed_vars[j + 1]
                        if next_var is not None and next_var != var_name:
                            break
                    run.append(patterns[j])
                    counts.append(current_count)
                    j += 1

                is_ascending = len(counts) >= 2 and all(
                    counts[k] + 1 == counts[k + 1] for k in range(len(counts) - 1)
                )
                is_descending = len(counts) >= 2 and all(
                    counts[k] - 1 == counts[k + 1] for k in range(len(counts) - 1)
                )
                if is_ascending or is_descending:
                    folded.append(
                        QuantifiedPattern(
                            VariablePattern(var_name),
                            min(counts),
                            max(counts),
                            isGreedy=is_descending,
                        )
                    )
                    changed = True
                    i = j
                    continue

                folded.append(patterns[i])
                i += 1

            return folded, changed

        simplified_patterns, fixed_runs_changed = _fold_fixed_count_runs(
            simplified_patterns
        )

        # Main optimization: remove duplicates and handle optional patterns
        # Process left to right: keep first occurrence, remove later duplicates
        result = []
        seen_non_optional = {}  # maps pattern string to (index, pattern)
        seen_optional_bases = {}  # maps base pattern string to (index, is_greedy)
        key_cache = {}
        has_explicit_empty = False
        has_optional = False
        # Whether an earlier alternative already has an empty candidate.  An
        # optional branch contributes such a candidate even though it is not
        # represented by a standalone EmptyPattern.  Keeping this separate
        # from ``has_explicit_empty`` lets us retain an early explicit empty
        # branch when a later optional branch would otherwise move epsilon
        # behind its greedy non-empty arm.
        empty_already_represented = False
        patterns_changed = nested_alternations_flattened or fixed_runs_changed

        def cached_pattern_key(p: PatternNode):
            key = id(p)
            if key not in key_cache:
                key_cache[key] = _pattern_key(p)
            return key_cache[key]

        for p in simplified_patterns:
            if isinstance(p, EmptyPattern):
                has_explicit_empty = True
                patterns_changed = True
                if not empty_already_represented:
                    result.append(p)
                    empty_already_represented = True
                continue

            # Check if it's an optional pattern (A? or A??)
            if isinstance(p, QuantifiedPattern) and p.atLeast == 0 and p.atMost == 1:
                base_str = cached_pattern_key(p.pattern)
                # If the base pattern already exists as non-optional, skip this optional
                if base_str in seen_non_optional:
                    has_explicit_empty = True
                    patterns_changed = True
                    # The non-empty arm is redundant, but the optional branch
                    # may be the first alternative capable of matching empty.
                    # Retain epsilon at this exact point in the preference
                    # order instead of moving it behind a later branch.
                    if not empty_already_represented:
                        result.append(EmptyPattern())
                        empty_already_represented = True
                    continue
                # If we haven't seen this optional base before, add it
                if base_str not in seen_optional_bases:
                    result.append(p)
                    seen_optional_bases[base_str] = (len(result) - 1, p.isGreedy)
                    has_optional = True
                    empty_already_represented = True
            else:
                # Regular non-optional pattern
                p_str = cached_pattern_key(p)
                # If we've seen this pattern before (either non-optional or as optional base), skip it
                if p_str in seen_non_optional or p_str in seen_optional_bases:
                    patterns_changed = True
                    continue
                # Otherwise add it
                result.append(p)
                seen_non_optional[p_str] = (len(result) - 1, p)
                if p.get_min_row_num() == 0:
                    empty_already_represented = True

        def _same_variable_quantifier_key(p: PatternNode):
            if isinstance(p, QuantifiedPattern) and isinstance(
                p.pattern, VariablePattern
            ):
                return (p.pattern.name, p.atMost, p.isGreedy)
            return None

        # A{m,X} before A{n,X} subsumes it when m <= n. If they are adjacent
        # and greedy, A{m,X} | A{n,X} can collapse to the wider lower-bound
        # quantifier. For reluctant adjacent ranges, keep branch order but
        # shrink the later branch to the part not already covered first.
        quantifier_optimized = []
        seen_quantifier_min = {}
        seen_quantifier_ranges = {}

        def _max_covers(left_max: Optional[int], right_max: Optional[int]) -> bool:
            return left_max is None or (right_max is not None and left_max >= right_max)

        def _max_strictly_covers(
            left_max: Optional[int], right_max: Optional[int]
        ) -> bool:
            return (
                left_max is None
                and right_max is not None
                or (
                    left_max is not None
                    and right_max is not None
                    and left_max > right_max
                )
            )

        def _previous_range_covers(p: QuantifiedPattern) -> bool:
            if not isinstance(p.pattern, VariablePattern):
                return False
            for left_min, left_max, left_is_greedy in seen_quantifier_ranges.get(
                p.pattern.name, []
            ):
                strictly_covers = left_min < p.atLeast or _max_strictly_covers(
                    left_max, p.atMost
                )
                if (
                    left_min <= p.atLeast
                    and _max_covers(left_max, p.atMost)
                    and (strictly_covers or left_is_greedy == p.isGreedy)
                ):
                    return True
            return False

        def _remember_quantifier_range(p: PatternNode):
            if isinstance(p, QuantifiedPattern) and isinstance(
                p.pattern, VariablePattern
            ):
                seen_quantifier_ranges.setdefault(p.pattern.name, []).append(
                    (p.atLeast, p.atMost, p.isGreedy)
                )

        def _merge_overlapping_quantifier_ranges(
            left: PatternNode, right: PatternNode
        ) -> Optional[QuantifiedPattern]:
            if not (
                isinstance(left, QuantifiedPattern)
                and isinstance(right, QuantifiedPattern)
                and isinstance(left.pattern, VariablePattern)
                and isinstance(right.pattern, VariablePattern)
                and left.pattern.name == right.pattern.name
                and left.isGreedy == right.isGreedy
                and left.atMost is not None
                and right.atMost is not None
            ):
                return None

            # A{m,n} | A{k,j}, where n >= j and k < m, has the same
            # first-success length order as A{k,n}: n, ..., m, ..., k.
            if (
                left.isGreedy
                and left.atMost >= right.atMost
                and right.atMost + 1 >= left.atLeast
                and right.atLeast < left.atLeast
            ):
                return QuantifiedPattern(
                    left.pattern, right.atLeast, left.atMost, isGreedy=True
                )

            # A{m,n}? | A{k,j}?, where m <= k and n < j, has the same
            # first-success length order as A{m,j}?: m, ..., n, ..., j.
            if (
                not left.isGreedy
                and left.atLeast <= right.atLeast
                and left.atMost + 1 >= right.atLeast
                and right.atMost > left.atMost
            ):
                return QuantifiedPattern(
                    left.pattern, left.atLeast, right.atMost, isGreedy=False
                )
            return None

        for p in result:
            key = _same_variable_quantifier_key(p)
            if key is None:
                quantifier_optimized.append(p)
                _remember_quantifier_range(p)
                continue

            if isinstance(p, QuantifiedPattern) and _previous_range_covers(p):
                patterns_changed = True
                continue

            min_seen = seen_quantifier_min.get(key)
            if min_seen is not None and min_seen <= p.atLeast:
                patterns_changed = True
                continue

            if quantifier_optimized:
                previous = quantifier_optimized[-1]
                previous_key = _same_variable_quantifier_key(previous)
                merged_range = _merge_overlapping_quantifier_ranges(previous, p)
                if merged_range is not None:
                    quantifier_optimized[-1] = merged_range
                    seen_quantifier_min[_same_variable_quantifier_key(merged_range)] = (
                        merged_range.atLeast
                    )
                    _remember_quantifier_range(merged_range)
                    patterns_changed = True
                    continue
                if (
                    previous_key == key
                    and isinstance(previous, QuantifiedPattern)
                    and previous.isGreedy
                    and previous.atLeast > p.atLeast
                ):
                    quantifier_optimized[-1] = p
                    seen_quantifier_min[key] = p.atLeast
                    _remember_quantifier_range(p)
                    patterns_changed = True
                    continue
                if (
                    previous_key == key
                    and isinstance(previous, QuantifiedPattern)
                    and not previous.isGreedy
                    and previous.atLeast > p.atLeast
                ):
                    shrunk_at_most = previous.atLeast - 1
                    if p.atMost is None or shrunk_at_most < p.atMost:
                        p = QuantifiedPattern(
                            p.pattern, p.atLeast, shrunk_at_most, p.isGreedy
                        )
                        key = _same_variable_quantifier_key(p)
                        patterns_changed = True

            quantifier_optimized.append(p)
            seen_quantifier_min[key] = (
                p.atLeast if min_seen is None else min(min_seen, p.atLeast)
            )
            _remember_quantifier_range(p)
        result = quantifier_optimized

        # If epsilon is still explicit and there is no other optional branch,
        # absorb it into an adjacent alternative without changing preference:
        # ``X | ()`` -> ``X?`` and ``() | X`` -> ``X??``.  In particular, do
        # not move an empty branch across unrelated alternatives.
        if has_explicit_empty and not has_optional and result:
            empty_index = next(
                (
                    index
                    for index, pattern in enumerate(result)
                    if isinstance(pattern, EmptyPattern)
                ),
                None,
            )
            if empty_index is not None:
                if empty_index > 0 and result[empty_index - 1].can_be_quantified():
                    previous = result[empty_index - 1]
                    result[empty_index - 1] = QuantifiedPattern(
                        previous, 0, 1, isGreedy=True
                    )
                    del result[empty_index]
                elif (
                    empty_index + 1 < len(result)
                    and result[empty_index + 1].can_be_quantified()
                ):
                    following = result[empty_index + 1]
                    result[empty_index + 1] = QuantifiedPattern(
                        following, 0, 1, isGreedy=False
                    )
                    del result[empty_index]

        # Handle edge cases
        if not result:
            return EmptyPattern()

        if len(result) == 1:
            return _finalize_quantified_result(result[0])

        # Return new alternation if:
        # 1. Patterns were removed/modified/transformed
        # 2. Inner patterns were simplified (simplified_patterns != self.patterns)
        if (
            patterns_changed
            or len(result) != len(self.patterns)
            or result != self.patterns
        ):
            return AlternationPattern(result)

        # Also check if any inner patterns were simplified
        for i, (orig, simp) in enumerate(zip(self.patterns, simplified_patterns)):
            if orig is not simp:
                return AlternationPattern(simplified_patterns)

        return self


@dataclass
class AnchorPattern(_DelegatingPattern):
    """Anchor Pattern, e.g., ^(A B C), (A B C)$"""

    pattern: PatternNode
    anchor_type: AnchorType  # '^' for start, '$' for end

    def to_string(self) -> str:
        inner = self.pattern.to_string()
        # Alternation has lower precedence than concatenation with an anchor.
        # Without these parentheses, ^(A | B) would be emitted as ^A | B and
        # the second branch would silently lose the anchor.
        if isinstance(self.pattern, AlternationPattern):
            inner = f"({inner})"
        if self.anchor_type == AnchorType.START:
            return f"^{inner}"
        else:  # AnchorType.END
            return f"{inner}$"

    def simplify(self) -> "PatternNode":
        simplified_pattern = self.pattern.simplify()
        if simplified_pattern is not self.pattern:
            return AnchorPattern(simplified_pattern, self.anchor_type)
        return self


@dataclass
class ConcatenationPattern(PatternNode):
    """A B C"""

    patterns: List[PatternNode]

    def to_string(self) -> str:
        def format_part(part: PatternNode) -> str:
            if isinstance(part, AlternationPattern):
                return f"({part.to_string()})"
            return part.to_string()

        return " ".join(format_part(p) for p in self.patterns)

    def get_var_num(self) -> int:
        return sum(p.get_var_num() for p in self.patterns)

    def get_min_row_num(self) -> int:
        return _sum_min_rows(self.patterns)

    def get_max_row_num(self) -> Optional[int]:
        return _sum_max_rows(self.patterns)

    def can_be_skipped(self) -> bool:
        """A concatenation is nullable exactly when every child is nullable."""
        return self.get_min_row_num() == 0

    def get_leftmost_variables(self) -> Set[str]:
        """Retrieve the leftmost possible variable, considering skipable patterns"""
        result = set()
        for p in self.patterns:
            result.update(p.get_leftmost_variables())
            # If the current pattern cannot be skipped, stop
            if not p.can_be_skipped():
                break
        return result

    def get_rightmost_variables(self) -> Set[str]:
        """Retrieve the rightmost possible variable, considering skipable patterns"""
        result = set()
        for p in reversed(self.patterns):
            result.update(p.get_rightmost_variables())
            # If the current pattern cannot be skipped, stop
            if not p.can_be_skipped():
                break
        return result

    def get_must_appear_var(self) -> Set[str]:
        """Only patterns that cannot be skipped will guarantee their variables appear"""
        result = set()
        for p in self.patterns:
            # Skip optional patterns and only collect from patterns that cannot be skipped
            if not p.can_be_skipped():
                result.update(p.get_must_appear_var())
        return result

    def get_variables(self) -> Set[str]:
        return _union_sets(p.get_variables() for p in self.patterns)

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        """Get maximum rows to the left of the leftmost var occurrence."""
        return _guaranteed_side_max_row_num(self, var, "left")

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        """Get maximum rows to the right of the rightmost var occurrence."""
        return _guaranteed_side_max_row_num(self, var, "right")

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        """Min rows to the left of the first occurrence of var."""
        prefix_min = 0
        best = None
        for p in self.patterns:
            inner = p.get_left_min_row_num(var)
            if inner is not None:
                candidate = prefix_min + inner
                best = candidate if best is None else min(best, candidate)

            can_omit, no_var_min = _min_rows_without_variable(p, var)
            if not can_omit:
                break
            assert no_var_min is not None
            prefix_min += no_var_min
        return best

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        """Min rows to the right of the last occurrence of var."""
        suffix_min = 0
        best = None
        for p in reversed(self.patterns):
            inner = p.get_right_min_row_num(var)
            if inner is not None:
                candidate = inner + suffix_min
                best = candidate if best is None else min(best, candidate)

            can_omit, no_var_min = _min_rows_without_variable(p, var)
            if not can_omit:
                break
            assert no_var_min is not None
            suffix_min += no_var_min
        return best

    def simplify(self) -> "PatternNode":
        optimized: List[PatternNode] = []
        for p in self.patterns:
            simplified = p.simplify()
            if isinstance(simplified, ConcatenationPattern):
                optimized.extend(simplified.patterns)
            elif isinstance(simplified, EmptyPattern):
                continue
            else:
                optimized.append(simplified)

        if not optimized:
            return EmptyPattern()

        def _is_same_var(a: PatternNode, b: PatternNode) -> bool:
            return (
                isinstance(a, VariablePattern)
                and isinstance(b, VariablePattern)
                and a.name == b.name
            )

        def _add_one_variable(
            q: QuantifiedPattern, var: VariablePattern
        ) -> Optional[QuantifiedPattern]:
            if isinstance(q.pattern, VariablePattern) and q.pattern.name == var.name:
                if q.atMost is None:
                    return QuantifiedPattern(q.pattern, q.atLeast + 1, None, q.isGreedy)
                return QuantifiedPattern(
                    q.pattern, q.atLeast + 1, q.atMost + 1, q.isGreedy
                )
            return None

        def _can_merge_quantifiers(
            left: QuantifiedPattern, right: QuantifiedPattern
        ) -> bool:
            # When one side is greedy and the other side is reluctant, we only
            # merge if the greedy side is either unbounded or fixed-count.
            if left.isGreedy == right.isGreedy:
                return True

            greedy = left if left.isGreedy else right
            return greedy.atMost is None or greedy.atMost == greedy.atLeast

        def _preserves_compound_body_preference(
            left: QuantifiedPattern, right: QuantifiedPattern
        ) -> bool:
            """Whether merging keeps the recursive choice order for the body.

            With multiple possible body traces, concatenation makes every
            decision in the left quantifier before decisions in the right
            one.  A fixed left side introduces no repetition-count decision.
            Two greedy sides are also equivalent when the right side is
            nullable: after each body trace both trees prefer another body,
            and only then exiting.  Other shapes can interleave body choices
            differently even when their accepted classifier strings agree.
            """
            if isinstance(left.pattern, VariablePattern):
                return True
            if left.atMost == left.atLeast:
                return True
            return left.isGreedy and right.isGreedy and right.atLeast == 0

        def _merged_greediness(
            left: QuantifiedPattern, right: QuantifiedPattern
        ) -> bool:
            if left.isGreedy == right.isGreedy:
                return left.isGreedy

            greedy = left if left.isGreedy else right
            # Mixed mode with fixed-count greedy side: reluctant side decides
            # between the multiple valid total counts after merge.
            if greedy.atMost is not None and greedy.atMost == greedy.atLeast:
                return False
            # Mixed mode with unbounded greedy side: keep greedy behavior.
            return True

        def _optional_unbounded_quantifier_var(p: PatternNode) -> Optional[str]:
            if not (
                isinstance(p, QuantifiedPattern) and p.atLeast == 0 and p.atMost == 1
            ):
                return None
            inner = p.pattern
            if (
                isinstance(inner, QuantifiedPattern)
                and isinstance(inner.pattern, VariablePattern)
                and inner.atLeast >= 1
                and inner.atMost is None
            ):
                return inner.pattern.name
            return None

        merged: List[PatternNode] = []
        for p in optimized:
            if merged and _is_same_var(merged[-1], p):
                # A A => A{2}, then keep growing: A{2} A => A{3}
                merged[-1] = QuantifiedPattern(merged[-1], 2, 2, True)
                continue

            if (
                merged
                and isinstance(merged[-1], QuantifiedPattern)
                and isinstance(p, VariablePattern)
            ):
                expanded = _add_one_variable(merged[-1], p)
                if expanded is not None:
                    merged[-1] = expanded
                    continue

            if (
                merged
                and isinstance(merged[-1], VariablePattern)
                and isinstance(p, QuantifiedPattern)
            ):
                expanded = _add_one_variable(p, merged[-1])
                if expanded is not None:
                    merged[-1] = expanded
                    continue

            if merged and isinstance(merged[-1], QuantifiedPattern):
                optional_var = _optional_unbounded_quantifier_var(p)
                left = merged[-1]
                if (
                    optional_var is not None
                    and isinstance(left.pattern, VariablePattern)
                    and left.pattern.name == optional_var
                    and left.atLeast >= 1
                    and left.atMost is None
                ):
                    continue

            if (
                merged
                and isinstance(merged[-1], QuantifiedPattern)
                and isinstance(p, QuantifiedPattern)
            ):
                left = merged[-1]
                right = p
                if (
                    left.pattern == right.pattern
                    and _can_merge_quantifiers(left, right)
                    and _preserves_compound_body_preference(left, right)
                ):
                    min_count = left.atLeast + right.atLeast
                    if left.atMost is None or right.atMost is None:
                        max_count = None
                    else:
                        max_count = left.atMost + right.atMost
                    # A+ A+? ==> A{2, }
                    merged[-1] = QuantifiedPattern(
                        left.pattern,
                        min_count,
                        max_count,
                        _merged_greediness(left, right),
                    )
                    continue
            merged.append(p)

        if len(merged) == 1:
            return merged[0]
        if merged != self.patterns:
            return ConcatenationPattern(merged)
        return self


@dataclass
class EmptyPattern(PatternNode):
    def to_string(self) -> str:
        return "()"

    def get_var_num(self) -> int:
        return 0

    def get_min_row_num(self) -> int:
        return 0

    def get_max_row_num(self) -> int:
        return 0

    def get_leftmost_variables(self) -> Set[str]:
        return set()

    def get_rightmost_variables(self) -> Set[str]:
        return set()

    def get_must_appear_var(self) -> Set[str]:
        """Empty pattern has no variables"""
        return set()

    def get_variables(self) -> Set[str]:
        return set()

    def can_be_skipped(self) -> bool:
        return True

    def get_left_max_row_num(self, var: str) -> Optional[int]:
        """Empty pattern contains no variables"""
        return None

    def get_right_max_row_num(self, var: str) -> Optional[int]:
        """Empty pattern contains no variables"""
        return None

    def get_left_min_row_num(self, var: str) -> Optional[int]:
        return None

    def get_right_min_row_num(self, var: str) -> Optional[int]:
        return None

    def simplify(self) -> "PatternNode":
        return self


def _unwrap_excluded_without_simplifying(pattern: PatternNode) -> PatternNode:
    """Rebuild a tree without exclusion wrappers in one recursive pass."""
    if isinstance(pattern, ExcludedPattern):
        return _unwrap_excluded_without_simplifying(pattern.pattern)

    if isinstance(pattern, QuantifiedPattern):
        return QuantifiedPattern(
            _unwrap_excluded_without_simplifying(pattern.pattern),
            pattern.atLeast,
            pattern.atMost,
            pattern.isGreedy,
        )

    if isinstance(pattern, PermutationPattern):
        return PermutationPattern(
            [_unwrap_excluded_without_simplifying(p) for p in pattern.patterns]
        )

    if isinstance(pattern, AlternationPattern):
        return AlternationPattern(
            [_unwrap_excluded_without_simplifying(p) for p in pattern.patterns]
        )

    if isinstance(pattern, AnchorPattern):
        return AnchorPattern(
            _unwrap_excluded_without_simplifying(pattern.pattern), pattern.anchor_type
        )

    if isinstance(pattern, ConcatenationPattern):
        return ConcatenationPattern(
            [_unwrap_excluded_without_simplifying(p) for p in pattern.patterns]
        )

    return pattern


def unwrap_excluded_pattern(pattern: PatternNode) -> PatternNode:
    """Remove exclusions, then simplify the rebuilt tree exactly once."""
    return _unwrap_excluded_without_simplifying(pattern).simplify()


def get_pattern_variables(pattern: PatternNode) -> Set[str]:
    """Return every primary pattern variable contained in a pattern tree."""
    return pattern.get_variables()


@dataclass(frozen=True)
class PatternRewriteBoundary:
    """Semantic boundary retained when surface syntax is simplified away."""

    kind: str
    variables: FrozenSet[str]
    elements: Tuple[FrozenSet[str], ...] = ()


def collect_pattern_rewrite_boundaries(
    pattern: PatternNode,
) -> Tuple[PatternRewriteBoundary, ...]:
    """Capture PERMUTE/EXCLUDE ownership before pattern simplification."""

    boundaries: List[PatternRewriteBoundary] = []

    def visit(node: PatternNode) -> None:
        if isinstance(node, ExcludedPattern):
            boundaries.append(
                PatternRewriteBoundary(
                    "exclude",
                    frozenset(node.pattern.get_variables()),
                )
            )
            visit(node.pattern)
            return
        if isinstance(node, PermutationPattern):
            elements = tuple(
                frozenset(child.get_variables()) for child in node.patterns
            )
            boundaries.append(
                PatternRewriteBoundary(
                    "permute",
                    frozenset().union(*elements),
                    elements,
                )
            )
            for child in node.patterns:
                visit(child)
            return
        if isinstance(node, (QuantifiedPattern, AnchorPattern)):
            visit(node.pattern)
            return
        if isinstance(node, (ConcatenationPattern, AlternationPattern)):
            for child in node.patterns:
                visit(child)

    visit(pattern)
    return tuple(boundaries)


def contains_excluded_pattern(pattern: PatternNode) -> bool:
    """Return whether a pattern tree contains an exclusion wrapper."""
    if isinstance(pattern, ExcludedPattern):
        return True
    if isinstance(pattern, (QuantifiedPattern, AnchorPattern)):
        return contains_excluded_pattern(pattern.pattern)
    if isinstance(
        pattern,
        (ConcatenationPattern, PermutationPattern, AlternationPattern),
    ):
        return any(contains_excluded_pattern(child) for child in pattern.patterns)
    return False


def _contains_anchor_pattern(pattern: PatternNode) -> bool:
    """Return whether *pattern* contains a partition anchor."""
    if isinstance(pattern, AnchorPattern):
        return True
    if isinstance(pattern, (QuantifiedPattern, ExcludedPattern)):
        return _contains_anchor_pattern(pattern.pattern)
    if isinstance(
        pattern,
        (ConcatenationPattern, PermutationPattern, AlternationPattern),
    ):
        return any(_contains_anchor_pattern(child) for child in pattern.patterns)
    return False


def _validate_anchor_placement(pattern: PatternNode) -> None:
    """Reject anchor shapes whose zero-width semantics are not represented.

    Start and end anchors are supported only at the corresponding boundary of
    an alternation branch.  Repeating an anchored subtree would re-evaluate the
    assertion at a different input position, and PERMUTE does not give an
    anchored element one stable boundary, so both forms are rejected.  An
    exact-one quantifier and a one-element PERMUTE are identity wrappers and
    remain valid.
    """

    def visit(node: PatternNode, at_start: bool, at_end: bool) -> None:
        if isinstance(node, AnchorPattern):
            if node.anchor_type == AnchorType.START and not at_start:
                raise ValueError(
                    "Start anchor (^) must appear at the beginning of an "
                    "alternation branch"
                )
            if node.anchor_type == AnchorType.END and not at_end:
                raise ValueError(
                    "End anchor ($) must appear at the end of an alternation branch"
                )
            visit(node.pattern, at_start, at_end)
            return

        if isinstance(node, QuantifiedPattern):
            if _contains_anchor_pattern(node.pattern) and not (
                node.atLeast == 1 and node.atMost == 1
            ):
                raise ValueError("A pattern containing an anchor cannot be quantified")
            visit(node.pattern, at_start, at_end)
            return

        if isinstance(node, PermutationPattern):
            if len(node.patterns) > 1 and any(
                _contains_anchor_pattern(child) for child in node.patterns
            ):
                raise ValueError(
                    "PERMUTE with multiple elements cannot contain an anchor"
                )
            for child in node.patterns:
                visit(child, at_start, at_end)
            return

        if isinstance(node, AlternationPattern):
            for child in node.patterns:
                visit(child, at_start, at_end)
            return

        if isinstance(node, ConcatenationPattern):
            suffix_is_zero_width = [False] * (len(node.patterns) + 1)
            suffix_is_zero_width[-1] = at_end
            for index in range(len(node.patterns) - 1, -1, -1):
                suffix_is_zero_width[index] = (
                    suffix_is_zero_width[index + 1]
                    and node.patterns[index].get_max_row_num() == 0
                )

            prefix_is_zero_width = at_start
            for index, child in enumerate(node.patterns):
                visit(
                    child,
                    prefix_is_zero_width,
                    suffix_is_zero_width[index + 1],
                )
                prefix_is_zero_width = (
                    prefix_is_zero_width and child.get_max_row_num() == 0
                )
            return

        if isinstance(node, ExcludedPattern):
            visit(node.pattern, at_start, at_end)

    visit(pattern, True, True)


def _pattern_key(pattern: PatternNode):
    """Return a formatting-independent, immutable structural identity."""
    if isinstance(pattern, VariablePattern):
        return ("variable", pattern.name)
    if isinstance(pattern, EmptyPattern):
        return ("empty",)
    if isinstance(pattern, QuantifiedPattern):
        return (
            "quantified",
            _pattern_key(pattern.pattern),
            pattern.atLeast,
            pattern.atMost,
            pattern.isGreedy,
        )
    if isinstance(pattern, ExcludedPattern):
        return ("excluded", _pattern_key(pattern.pattern))
    if isinstance(pattern, AnchorPattern):
        return ("anchor", pattern.anchor_type, _pattern_key(pattern.pattern))
    if isinstance(pattern, PermutationPattern):
        return ("permutation", tuple(_pattern_key(p) for p in pattern.patterns))
    if isinstance(pattern, AlternationPattern):
        return ("alternation", tuple(_pattern_key(p) for p in pattern.patterns))
    if isinstance(pattern, ConcatenationPattern):
        return ("concatenation", tuple(_pattern_key(p) for p in pattern.patterns))
    raise TypeError(f"Unsupported pattern node: {type(pattern).__name__}")


class TokenType(Enum):
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    COMMA = ","
    PIPE = "|"
    CARET = "^"
    DOLLAR = "$"
    STAR = "*"
    PLUS = "+"
    QUESTION = "?"
    EXCLUDE_START = "{-"
    EXCLUDE_END = "-}"
    PERMUTE = "PERMUTE"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str


class PatternLexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self):
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def get_next_token(self) -> Token:
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._identifier()

            if self.current_char.isdigit():
                return self._number()

            if self.current_char == "{":
                # check for {-
                if self.pos + 1 < len(self.text) and self.text[self.pos + 1] == "-":
                    self.advance()  # {
                    self.advance()  # -
                    return Token(TokenType.EXCLUDE_START, "{-")
                return self._single_char_token(TokenType.LBRACE)

            if self.current_char == "-":
                # check for -}
                if self.pos + 1 < len(self.text) and self.text[self.pos + 1] == "}":
                    self.advance()  # -
                    self.advance()  # }
                    return Token(TokenType.EXCLUDE_END, "-}")
                # Treat single '-' as error or handle explicitly if needed
                raise ValueError(f"Unexpected character: -")

            char_map = {
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "}": TokenType.RBRACE,
                ",": TokenType.COMMA,
                "|": TokenType.PIPE,
                "^": TokenType.CARET,
                "$": TokenType.DOLLAR,
                "*": TokenType.STAR,
                "+": TokenType.PLUS,
                "?": TokenType.QUESTION,
            }

            if self.current_char in char_map:
                return self._single_char_token(char_map[self.current_char])

            raise ValueError(f"Unknown character: {self.current_char}")

        return Token(TokenType.EOF, "")

    def _single_char_token(self, token_type: TokenType) -> Token:
        value = self.current_char
        self.advance()
        return Token(token_type, value)

    def _identifier(self):
        start = self.pos
        while self.current_char is not None and (
            self.current_char.isalnum() or self.current_char == "_"
        ):
            self.advance()
        result = self.text[start : self.pos]

        if result.upper() == "PERMUTE":
            return Token(TokenType.PERMUTE, result)
        return Token(TokenType.IDENTIFIER, result)

    def _number(self):
        start = self.pos
        while self.current_char is not None and self.current_char.isdigit():
            self.advance()
        result = self.text[start : self.pos]
        return Token(TokenType.NUMBER, result)


class PatternParser:
    def __init__(self, lexer: PatternLexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        self.rewrite_boundaries: Tuple[PatternRewriteBoundary, ...] = ()

    def _consume(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise ValueError(
                f"Expected {token_type}, but got {self.current_token.type}"
            )

    def parse(self) -> PatternNode:
        """Entrance method"""
        node = self._parse_alternation()
        self.rewrite_boundaries = collect_pattern_rewrite_boundaries(node)
        node = node.simplify()
        if self.current_token.type != TokenType.EOF:
            raise ValueError(f"Unexpected token at end: {self.current_token}")
        return node

    def _parse_alternation(self) -> PatternNode:
        """analyze (Alternation): A | B | C"""
        node = self._parse_concatenation()

        patterns = [node]
        while self.current_token.type == TokenType.PIPE:
            self._consume(TokenType.PIPE)
            patterns.append(self._parse_concatenation())

        if len(patterns) > 1:
            return AlternationPattern(patterns)
        return patterns[0]

    def _parse_concatenation(self) -> PatternNode:
        """analyze (Concatenation): A B C"""
        patterns = []

        # As long as the current token is a character that may be
        # at the beginning of the pattern, continue parsing
        while self.current_token.type in (
            TokenType.IDENTIFIER,
            TokenType.LPAREN,
            TokenType.EXCLUDE_START,
            TokenType.PERMUTE,
            TokenType.CARET,
        ):
            patterns.append(self._parse_term())

        # Handle trailing $ for patterns like ^A{5}$ where the CARET branch in term()
        # returns immediately without consuming the $ (since $ check is in the else branch).
        # Wrap the last collected pattern in an END anchor.
        if self.current_token.type == TokenType.DOLLAR and patterns:
            self._consume(TokenType.DOLLAR)
            patterns[-1] = AnchorPattern(patterns[-1], AnchorType.END)

        if len(patterns) == 0:
            return EmptyPattern()
        if len(patterns) == 1:
            return patterns[0]
        return ConcatenationPattern(patterns)

    def _parse_term(self) -> PatternNode:
        """Analyze a single item, including anchor point handling: ^A or A$"""
        # Process 'Start Anchor Point (^)'
        if self.current_token.type == TokenType.CARET:
            self._consume(TokenType.CARET)
            node = self._parse_quantified_factor()  # Anchor -> core factors
            return AnchorPattern(node, AnchorType.START)

        # Analyze core factors (possibly with quantifiers)
        node = self._parse_quantified_factor()

        # Process 'End Anchor Point ($)'
        if self.current_token.type == TokenType.DOLLAR:
            self._consume(TokenType.DOLLAR)
            return AnchorPattern(node, AnchorType.END)

        return node

    def _parse_quantified_factor(self) -> PatternNode:
        """Analyze factors with quantifiers: A*, B{1,2}?"""
        node = self._parse_factor()

        # Check if there are quantifiers at the back
        if self.current_token.type in (
            TokenType.STAR,
            TokenType.PLUS,
            TokenType.QUESTION,
            TokenType.LBRACE,
        ):
            return self._parse_quantifier(node)

        return node

    def _parse_quantifier(self, node: PatternNode) -> PatternNode:
        min_occ = 1
        max_occ = 1
        is_greedy = True

        token = self.current_token

        if token.type == TokenType.STAR:
            self._consume(TokenType.STAR)
            min_occ, max_occ = 0, None
        elif token.type == TokenType.PLUS:
            self._consume(TokenType.PLUS)
            min_occ, max_occ = 1, None
        elif token.type == TokenType.QUESTION:
            self._consume(TokenType.QUESTION)
            min_occ, max_occ = 0, 1
        elif token.type == TokenType.LBRACE:
            self._consume(TokenType.LBRACE)
            min_str = self.current_token.value
            self._consume(TokenType.NUMBER)
            min_occ = int(min_str)

            if self.current_token.type == TokenType.COMMA:
                self._consume(TokenType.COMMA)
                if self.current_token.type == TokenType.NUMBER:
                    max_str = self.current_token.value
                    self._consume(TokenType.NUMBER)
                    max_occ = int(max_str)
                else:
                    max_occ = None  # {N,}
            else:
                max_occ = min_occ  # {N}

            self._consume(TokenType.RBRACE)

        # non-gready marker (Reluctant)
        if self.current_token.type == TokenType.QUESTION:
            self._consume(TokenType.QUESTION)
            is_greedy = False

        if max_occ == 0:
            raise ValueError("Pattern quantifier upper bound must be greater than 0")
        if max_occ is not None and min_occ > max_occ:
            raise ValueError(
                "Pattern quantifier lower bound must not exceed upper bound"
            )

        return QuantifiedPattern(node, min_occ, max_occ, is_greedy)

    def _parse_factor(self) -> PatternNode:
        """Analyze basic units: (A), {- A -}, PERMUTE(A B), VAR"""
        token = self.current_token

        if token.type == TokenType.LPAREN:
            self._consume(TokenType.LPAREN)
            # Recursive parsing of the content within parentheses
            node = self._parse_alternation()
            self._consume(TokenType.RPAREN)
            return node

        elif token.type == TokenType.EXCLUDE_START:
            self._consume(TokenType.EXCLUDE_START)
            node = self._parse_alternation()
            self._consume(TokenType.EXCLUDE_END)
            return ExcludedPattern(node)

        elif token.type == TokenType.PERMUTE:
            self._consume(TokenType.PERMUTE)
            self._consume(TokenType.LPAREN)
            patterns = []

            # The elements inside PERMUTE must be complete patterns (expr) and must be separated by commas
            if self.current_token.type != TokenType.RPAREN:  # Check if it's empty
                patterns.append(
                    self._parse_alternation()
                )  # Parse the first pattern (including alternation and concatenation)

                while self.current_token.type == TokenType.COMMA:
                    self._consume(TokenType.COMMA)  # Eat the comma
                    patterns.append(
                        self._parse_alternation()
                    )  # Parse the subsequent patterns

            self._consume(TokenType.RPAREN)
            return PermutationPattern(patterns)

        elif token.type == TokenType.IDENTIFIER:
            name = token.value
            self._consume(TokenType.IDENTIFIER)
            return VariablePattern(name)

        else:
            raise ValueError(f"Unexpected token in factor: {token}")

    # Backward-compatible aliases for older callers/tests that reached into parser internals.
    eat = _consume
    expr = _parse_alternation
    concat = _parse_concatenation
    term = _parse_term
    quantified_factor = _parse_quantified_factor
    parse_quantifier = _parse_quantifier
    factor = _parse_factor


def parse_pattern_string(pattern_str: str) -> PatternNode:
    lexer = PatternLexer(pattern_str)
    parser = PatternParser(lexer)
    return parser.parse()


if __name__ == "__main__":
    test_patterns = [
        "A {- B+ C? -} D",
        "A B+ C? D",
        "^A B C",
        "^(A B C)",
        # "(A B C)",
        # "(A B)*",
        # "(A B | A C)",
        # "(A | B?) C A",
        # "((A | () B | ()) C A)",
    ]

    for test_pattern in test_patterns:
        print(f"\nInput: {test_pattern}")
        root_node = parse_pattern_string(test_pattern)
        print(f"Reconstructed: {root_node.to_string()}")

        print("Pattern Structure==> ", root_node)

        print(f"Number of variables: {root_node.get_var_num()}")
        print(f"Number of minmimum rows: {root_node.get_min_row_num()}")
        print(f"Number of maximum rows: {root_node.get_max_row_num()}")
        print(f"Leftmost variables: {root_node.get_leftmost_variables()}")
        print(f"Rightmost variables: {root_node.get_rightmost_variables()}")
        print(f"Must appear variables: {root_node.get_must_appear_var()}")
        print(f"Must appear middle variables: {root_node.get_must_middle_var()}")
