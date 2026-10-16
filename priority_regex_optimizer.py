"""Priority-preserving simplification for a small, regular regex dialect.

The semantic contract is the one used by traditional backtracking engines:

* alternatives are tried from left to right;
* repetitions are greedy unless suffixed by ``?``;
* the result is the success/failure and end offset of an anchored match;
* captures, assertions, backreferences and flags are outside the contract.

Candidate generation and semantic verification are deliberately independent.
``greenery`` is allowed to generate ordinary language-equivalent candidates.
The default contract proves equality of both the complete languages and the
leftmost-first match results.  An explicit match-only mode also considers
lower-priority branch pruning and proves only the anchored match result;
those standalone proofs require verification in the complete caller context.

For sources containing effective reluctant quantifiers, optimization is
disabled before any candidate is generated.  Standalone selected-end
equivalence is not substitutive under an arbitrary suffix, including for
otherwise-safe-looking structural subtree rewrites.

The verifier is exact for accepted inputs.  It can return UNKNOWN when the
configured state limit is reached or for an unsupported construct.  In
particular, repeating a nullable body more than once is rejected because real
engines have special zero-length-iteration rules that are not represented by
the simple ordered Thompson machine used here.

----------------------------DEMO----------------------------
from priority_regex_optimizer import optimize_priority_regex

result = optimize_priority_regex("[ab]+[ac]+")

print("original regex: ", result.original)
print("optimized regex: ", result.optimized)
print("has changed: ", result.changed)
print("verification result: ", result.verification.verdict)
print("verification reason: ", result.verification.reason)
------------------------------------------------------------
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

UNICODE_MAX = 0x10FFFF


class RegexError(ValueError):
    """The expression is outside the supported regex dialect."""


class StateLimitExceeded(RuntimeError):
    pass


class EquivalenceMode(str, Enum):
    """The observations that an optimizer candidate must preserve."""

    LANGUAGE_AND_MATCH = "language_and_match"
    MATCH_ONLY = "match_only"


class ProofScope(str, Enum):
    """The contexts in which a successful verification may be reused.

    ``COMPOSITIONAL`` is reserved for an explicit rewrite theorem whose
    complete fallback order is preserved under surrounding concatenation.
    ``STANDALONE`` means the preferred result of this expression in isolation
    was proved equal (and its complete language under the default contract).
    That weaker result must be re-verified after substitution into its complete
    caller context.
    ``WHOLE_PATTERN`` records that this second proof has been completed by the
    SQL rewrite layer; the pure regex optimizer never emits it directly.
    """

    COMPOSITIONAL = "compositional"
    STANDALONE = "standalone"
    WHOLE_PATTERN = "whole_pattern"


# ---------------------------------------------------------------------------
# Character sets and AST


@dataclass(frozen=True)
class CharSet:
    """A normalized set of Unicode code-point intervals (inclusive)."""

    ranges: Tuple[Tuple[int, int], ...]

    @staticmethod
    def make(ranges: Iterable[Tuple[int, int]]) -> "CharSet":
        items = sorted(
            (max(0, lo), min(UNICODE_MAX, hi)) for lo, hi in ranges if lo <= hi
        )
        merged: List[List[int]] = []
        for lo, hi in items:
            if not merged or lo > merged[-1][1] + 1:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
        return CharSet(tuple((lo, hi) for lo, hi in merged))

    @staticmethod
    def one(codepoint: int) -> "CharSet":
        return CharSet(((codepoint, codepoint),))

    def contains(self, codepoint: int) -> bool:
        # Regex character classes normally contain few ranges.  A linear scan
        # is quicker than maintaining another indexed representation here.
        for lo, hi in self.ranges:
            if codepoint < lo:
                return False
            if codepoint <= hi:
                return True
        return False

    def complement(self) -> "CharSet":
        result: List[Tuple[int, int]] = []
        start = 0
        for lo, hi in self.ranges:
            if start < lo:
                result.append((start, lo - 1))
            start = hi + 1
        if start <= UNICODE_MAX:
            result.append((start, UNICODE_MAX))
        return CharSet(tuple(result))


ANY_EXCEPT_NEWLINE = CharSet.make(((0, 9), (11, UNICODE_MAX)))


class Node:
    pass


@dataclass(frozen=True)
class Empty(Node):
    pass


@dataclass(frozen=True)
class Atom(Node):
    chars: CharSet


@dataclass(frozen=True)
class Concatenation(Node):
    parts: Tuple[Node, ...]


@dataclass(frozen=True)
class Alternation(Node):
    options: Tuple[Node, ...]


@dataclass(frozen=True)
class Repetition(Node):
    body: Node
    minimum: int
    maximum: Optional[int]
    greedy: bool = True


def _concat(parts: Iterable[Node]) -> Node:
    flat: List[Node] = []
    for part in parts:
        if isinstance(part, Empty):
            continue
        if isinstance(part, Concatenation):
            flat.extend(part.parts)
        else:
            flat.append(part)
    if not flat:
        return Empty()
    if len(flat) == 1:
        return flat[0]
    return Concatenation(tuple(flat))


def _alternate(options: Iterable[Node]) -> Node:
    flat: List[Node] = []
    for option in options:
        if isinstance(option, Alternation):
            flat.extend(option.options)
        else:
            flat.append(option)
    if len(flat) == 1:
        return flat[0]
    return Alternation(tuple(flat))


def nullable(node: Node) -> bool:
    if isinstance(node, Empty):
        return True
    if isinstance(node, Atom):
        return False
    if isinstance(node, Concatenation):
        return all(nullable(part) for part in node.parts)
    if isinstance(node, Alternation):
        return any(nullable(option) for option in node.options)
    if isinstance(node, Repetition):
        return node.minimum == 0 or nullable(node.body)
    raise AssertionError(type(node))


# ---------------------------------------------------------------------------
# Parser


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0

    def parse(self) -> Node:
        node = self._alternation()
        if self.pos != len(self.source):
            raise RegexError(
                f"unexpected {self.source[self.pos]!r} at offset {self.pos}"
            )
        return node

    def _alternation(self) -> Node:
        options = [self._concatenation()]
        while self._take("|"):
            options.append(self._concatenation())
        return _alternate(options)

    def _concatenation(self) -> Node:
        parts: List[Node] = []
        while self.pos < len(self.source) and self.source[self.pos] not in ")|":
            parts.append(self._repetition())
        return _concat(parts)

    def _repetition(self) -> Node:
        atom = self._atom()
        quantifier = self._quantifier()
        if quantifier is None:
            return atom
        minimum, maximum = quantifier
        greedy = not self._take("?")
        if self.pos < len(self.source) and self.source[self.pos] in "*+?{":
            raise RegexError(f"multiple repeat at offset {self.pos}")
        return Repetition(atom, minimum, maximum, greedy)

    def _atom(self) -> Node:
        if self.pos >= len(self.source):
            raise RegexError("expected an atom")
        char = self.source[self.pos]
        if char == "(":
            self.pos += 1
            if self.source.startswith("?:", self.pos):
                self.pos += 2
            elif self.pos < len(self.source) and self.source[self.pos] == "?":
                raise RegexError(
                    "lookarounds, flags and special groups are unsupported"
                )
            node = self._alternation()
            if not self._take(")"):
                raise RegexError(f"unclosed group at offset {self.pos}")
            return node
        if char == "[":
            return Atom(self._character_class())
        if char == ".":
            self.pos += 1
            return Atom(ANY_EXCEPT_NEWLINE)
        if char == "\\":
            self.pos += 1
            return Atom(CharSet.one(self._escaped_codepoint(in_class=False)))
        if char in "^$":
            raise RegexError(
                "anchors are unsupported; verification is anchored implicitly"
            )
        if char in "*+?{}]":
            raise RegexError(f"unexpected metacharacter {char!r} at offset {self.pos}")
        self.pos += 1
        return Atom(CharSet.one(ord(char)))

    def _quantifier(self) -> Optional[Tuple[int, Optional[int]]]:
        if self._take("*"):
            return 0, None
        if self._take("+"):
            return 1, None
        if self._take("?"):
            return 0, 1
        if self.pos >= len(self.source) or self.source[self.pos] != "{":
            return None

        start = self.pos
        self.pos += 1
        minimum = self._decimal()
        if minimum is None:
            self.pos = start
            return None
        if self._take("}"):
            return minimum, minimum
        if not self._take(","):
            self.pos = start
            return None
        maximum = self._decimal()
        if not self._take("}"):
            self.pos = start
            return None
        if maximum is not None and maximum < minimum:
            raise RegexError(f"invalid repeat range at offset {start}")
        return minimum, maximum

    def _decimal(self) -> Optional[int]:
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
        if start == self.pos:
            return None
        value = int(self.source[start : self.pos])
        if value > 1000:
            raise RegexError("repeat bounds above 1000 are unsupported")
        return value

    def _character_class(self) -> CharSet:
        assert self._take("[")
        negate = self._take("^")
        ranges: List[Tuple[int, int]] = []
        first = True
        while True:
            if self.pos >= len(self.source):
                raise RegexError("unclosed character class")
            if self.source[self.pos] == "]" and not first:
                self.pos += 1
                break
            left = self._class_codepoint()
            first = False
            if (
                self.pos < len(self.source)
                and self.source[self.pos] == "-"
                and self.pos + 1 < len(self.source)
                and self.source[self.pos + 1] != "]"
            ):
                self.pos += 1
                right = self._class_codepoint()
                if right < left:
                    raise RegexError("character-class range is backwards")
                ranges.append((left, right))
            else:
                ranges.append((left, left))
        chars = CharSet.make(ranges)
        return chars.complement() if negate else chars

    def _class_codepoint(self) -> int:
        if self.pos >= len(self.source):
            raise RegexError("unclosed character class")
        char = self.source[self.pos]
        self.pos += 1
        if char == "\\":
            return self._escaped_codepoint(in_class=True)
        return ord(char)

    def _escaped_codepoint(self, *, in_class: bool) -> int:
        if self.pos >= len(self.source):
            raise RegexError("trailing backslash")
        char = self.source[self.pos]
        self.pos += 1
        simple = {"n": "\n", "r": "\r", "t": "\t", "f": "\f", "v": "\v", "a": "\a"}
        if char in simple:
            return ord(simple[char])
        widths = {"x": 2, "u": 4, "U": 8}
        if char in widths:
            width = widths[char]
            digits = self.source[self.pos : self.pos + width]
            if len(digits) != width or any(
                c not in "0123456789abcdefABCDEF" for c in digits
            ):
                raise RegexError(f"invalid \\{char} escape")
            self.pos += width
            value = int(digits, 16)
            if value > UNICODE_MAX:
                raise RegexError("Unicode escape is out of range")
            return value
        if char.isdigit():
            raise RegexError("backreferences and octal escapes are unsupported")
        if char in "dDsSwW":
            raise RegexError("shorthand Unicode character classes are unsupported")
        if not in_class and char in "AbBZG":
            raise RegexError("zero-width assertions are unsupported")
        if char.isalpha():
            raise RegexError(f"unsupported escape \\{char}")
        return ord(char)

    def _take(self, token: str) -> bool:
        if self.source.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False


def parse_regex(source: str) -> Node:
    return Parser(source).parse()


# ---------------------------------------------------------------------------
# Rendering (used for generated candidates)


def _escaped_char(codepoint: int, in_class: bool = False) -> str:
    char = chr(codepoint)
    named = {10: r"\n", 13: r"\r", 9: r"\t", 12: r"\f", 11: r"\v"}
    if codepoint in named:
        return named[codepoint]
    specials = "\\]-^" if in_class else "\\.^$|?*+()[]{}"
    if char in specials:
        return "\\" + char
    if 0x20 <= codepoint <= 0x7E or codepoint >= 0xA0:
        return char
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def _class_body(ranges: Sequence[Tuple[int, int]]) -> str:
    pieces: List[str] = []
    for lo, hi in ranges:
        if lo == hi:
            pieces.append(_escaped_char(lo, True))
        elif hi == lo + 1:
            # ``[ab]`` is one character shorter than ``[a-b]``.
            pieces.append(_escaped_char(lo, True) + _escaped_char(hi, True))
        else:
            pieces.append(f"{_escaped_char(lo, True)}-{_escaped_char(hi, True)}")
    return "".join(pieces)


def _charset_regex(chars: CharSet) -> str:
    if chars == ANY_EXCEPT_NEWLINE:
        return "."
    if len(chars.ranges) == 1 and chars.ranges[0][0] == chars.ranges[0][1]:
        return _escaped_char(chars.ranges[0][0])
    complement = chars.complement()
    if complement.ranges and len(complement.ranges) < len(chars.ranges):
        return "[^" + _class_body(complement.ranges) + "]"
    return "[" + _class_body(chars.ranges) + "]"


def to_regex(node: Node, parent_precedence: int = 0) -> str:
    if isinstance(node, Empty):
        text, precedence = "", 4
    elif isinstance(node, Atom):
        text, precedence = _charset_regex(node.chars), 4
    elif isinstance(node, Repetition):
        body = "()" if isinstance(node.body, Empty) else to_regex(node.body, 3)
        if node.minimum == 0 and node.maximum is None:
            suffix = "*"
        elif node.minimum == 1 and node.maximum is None:
            suffix = "+"
        elif node.minimum == 0 and node.maximum == 1:
            suffix = "?"
        elif node.maximum == node.minimum:
            suffix = "{" + str(node.minimum) + "}"
        elif node.maximum is None:
            suffix = "{" + str(node.minimum) + ",}"
        else:
            suffix = "{" + str(node.minimum) + "," + str(node.maximum) + "}"
        text = body + suffix + ("" if node.greedy else "?")
        precedence = 3
    elif isinstance(node, Concatenation):
        text = "".join(to_regex(part, 2) for part in node.parts)
        precedence = 2
    elif isinstance(node, Alternation):
        text = "|".join(to_regex(option, 1) for option in node.options)
        precedence = 1
    else:
        raise AssertionError(type(node))
    if precedence < parent_precedence:
        # Captured values are outside this optimizer's semantic contract, so a
        # plain group is sufficient to preserve operator precedence.
        return "(" + text + ")"
    return text


# ---------------------------------------------------------------------------
# Prioritized Thompson NFA


@dataclass
class _Instruction:
    opcode: str
    chars: Optional[CharSet] = None
    out1: Optional[int] = None
    out2: Optional[int] = None


@dataclass
class _Fragment:
    start: int
    outs: List[Tuple[int, str]]


@dataclass
class _Program:
    instructions: List[_Instruction]
    start: int


class _Compiler:
    def __init__(self, max_states: int):
        self.instructions: List[_Instruction] = []
        self.max_states = max_states

    def emit(self, instruction: _Instruction) -> int:
        if len(self.instructions) >= self.max_states:
            raise StateLimitExceeded(f"NFA exceeded {self.max_states} instructions")
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def patch(self, outs: Iterable[Tuple[int, str]], target: int) -> None:
        for index, field in outs:
            setattr(self.instructions[index], field, target)

    def empty(self) -> _Fragment:
        index = self.emit(_Instruction("jump"))
        return _Fragment(index, [(index, "out1")])

    def sequence(self, left: _Fragment, right: _Fragment) -> _Fragment:
        self.patch(left.outs, right.start)
        return _Fragment(left.start, right.outs)

    def compile_node(self, node: Node) -> _Fragment:
        if isinstance(node, Empty):
            return self.empty()
        if isinstance(node, Atom):
            index = self.emit(_Instruction("char", chars=node.chars))
            return _Fragment(index, [(index, "out1")])
        if isinstance(node, Concatenation):
            result = self.empty()
            for part in node.parts:
                result = self.sequence(result, self.compile_node(part))
            return result
        if isinstance(node, Alternation):
            fragments = [self.compile_node(option) for option in node.options]
            if not fragments:
                return self.empty()
            start = fragments[-1].start
            for fragment in reversed(fragments[:-1]):
                start = self.emit(
                    _Instruction("split", out1=fragment.start, out2=start)
                )
            outs = [out for fragment in fragments for out in fragment.outs]
            return _Fragment(start, outs)
        if isinstance(node, Repetition):
            if nullable(node.body) and (node.maximum is None or node.maximum > 1):
                raise RegexError(
                    "repetition of a nullable body more than once is unsupported"
                )
            result = self.empty()
            for _ in range(node.minimum):
                result = self.sequence(result, self.compile_node(node.body))

            if node.maximum is None:
                body = self.compile_node(node.body)
                split = self.emit(_Instruction("split"))
                if node.greedy:
                    self.instructions[split].out1 = body.start
                    exit_field = "out2"
                else:
                    self.instructions[split].out2 = body.start
                    exit_field = "out1"
                self.patch(body.outs, split)
                result = self.sequence(result, _Fragment(split, [(split, exit_field)]))
                return result

            for _ in range(node.maximum - node.minimum):
                body = self.compile_node(node.body)
                split = self.emit(_Instruction("split"))
                if node.greedy:
                    self.instructions[split].out1 = body.start
                    exit_field = "out2"
                else:
                    self.instructions[split].out2 = body.start
                    exit_field = "out1"
                optional = _Fragment(split, body.outs + [(split, exit_field)])
                result = self.sequence(result, optional)
            return result
        raise AssertionError(type(node))

    def finish(self, node: Node) -> _Program:
        fragment = self.compile_node(node)
        match = self.emit(_Instruction("match"))
        self.patch(fragment.outs, match)
        return _Program(self.instructions, fragment.start)


def _compile(node: Node, max_states: int) -> _Program:
    return _Compiler(max_states).finish(node)


@dataclass(frozen=True)
class _EngineState:
    # Only CHAR instructions before the first MATCH instruction remain live.
    frontier: Tuple[int, ...]
    tagged: bool
    halted: int = 0  # 0 running, 1 committed success, 2 permanent failure


class _OrderedMachine:
    """Deterministic simulation of an ordered Thompson/Pike VM."""

    def __init__(self, program: _Program):
        self.program = program
        threads = self._epsilon_closure((program.start,))
        self.initial, self.initial_reset = self._normalize(threads, False)
        self._transition_cache: Dict[
            Tuple[_EngineState, int], Tuple[_EngineState, bool]
        ] = {}

    def _epsilon_closure(self, seeds: Iterable[int]) -> Tuple[int, ...]:
        seen = set()
        result: List[int] = []
        stack = list(reversed(tuple(seeds)))
        while stack:
            pc = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            instruction = self.program.instructions[pc]
            if instruction.opcode == "jump":
                assert instruction.out1 is not None
                stack.append(instruction.out1)
            elif instruction.opcode == "split":
                assert instruction.out1 is not None and instruction.out2 is not None
                # Stack is LIFO: push the lower-priority branch first.
                stack.append(instruction.out2)
                stack.append(instruction.out1)
            else:
                result.append(pc)
        return tuple(result)

    def _normalize(
        self, threads: Tuple[int, ...], old_tag: bool
    ) -> Tuple[_EngineState, bool]:
        match_at: Optional[int] = None
        for index, pc in enumerate(threads):
            if self.program.instructions[pc].opcode == "match":
                match_at = index
                break
        if match_at is not None:
            frontier = tuple(
                pc
                for pc in threads[:match_at]
                if self.program.instructions[pc].opcode == "char"
            )
            if not frontier:
                return _EngineState((), True, 1), True
            return _EngineState(frontier, True, 0), True

        frontier = tuple(
            pc for pc in threads if self.program.instructions[pc].opcode == "char"
        )
        if frontier:
            return _EngineState(frontier, old_tag, 0), False
        if old_tag:
            return _EngineState((), True, 1), False
        return _EngineState((), False, 2), False

    def step(self, state: _EngineState, codepoint: int) -> Tuple[_EngineState, bool]:
        key = (state, codepoint)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        if state.halted:
            result = (state, False)
            self._transition_cache[key] = result
            return result

        seeds: List[int] = []
        for pc in state.frontier:
            instruction = self.program.instructions[pc]
            assert instruction.opcode == "char" and instruction.chars is not None
            if instruction.chars.contains(codepoint):
                assert instruction.out1 is not None
                seeds.append(instruction.out1)
        threads = self._epsilon_closure(seeds)
        result = self._normalize(threads, state.tagged)
        self._transition_cache[key] = result
        return result

    def match_end(self, text: str) -> Optional[int]:
        state = self.initial
        tag: Optional[int] = 0 if self.initial_reset else None
        for offset, char in enumerate(text, 1):
            state, reset = self.step(state, ord(char))
            if reset:
                tag = offset
        return tag if state.tagged else None


class _LanguageMachine:
    """Determinize a Thompson program without applying match priority.

    The ordered machine intentionally discards threads below the first MATCH
    instruction.  That is correct for the selected Perl-style result, but it
    hides lower-priority accepted strings.  A language certificate explicitly
    preserves those strings, which surrounding suffixes may expose during
    backtracking.  Match-only candidates can change them but require another
    match-result proof after substitution into their complete caller context.
    """

    def __init__(self, program: _Program):
        self.program = program
        self.initial = self._epsilon_closure((program.start,))
        self._transition_cache: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]] = {}

    def _epsilon_closure(self, seeds: Iterable[int]) -> Tuple[int, ...]:
        seen = set()
        stack = list(seeds)
        while stack:
            pc = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            instruction = self.program.instructions[pc]
            if instruction.opcode == "jump":
                assert instruction.out1 is not None
                stack.append(instruction.out1)
            elif instruction.opcode == "split":
                assert instruction.out1 is not None and instruction.out2 is not None
                stack.append(instruction.out1)
                stack.append(instruction.out2)
        return tuple(sorted(seen))

    def step(self, state: Tuple[int, ...], codepoint: int) -> Tuple[int, ...]:
        key = (state, codepoint)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        seeds = []
        for pc in state:
            instruction = self.program.instructions[pc]
            if (
                instruction.opcode == "char"
                and instruction.chars is not None
                and instruction.chars.contains(codepoint)
            ):
                assert instruction.out1 is not None
                seeds.append(instruction.out1)
        result = self._epsilon_closure(seeds)
        self._transition_cache[key] = result
        return result

    def accepting(self, state: Tuple[int, ...]) -> bool:
        return any(self.program.instructions[pc].opcode == "match" for pc in state)


# ---------------------------------------------------------------------------
# Exact product verifier


class Verdict(Enum):
    PROVED = "proved"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verification:
    verdict: Verdict
    reason: str
    counterexample: Optional[str] = None
    original_end: Optional[int] = None
    candidate_end: Optional[int] = None
    explored_states: int = 0


_NONE = 0
_LEFT_ONLY = 1
_RIGHT_ONLY = 2
_EQUAL = 3
_LEFT_NEWER = 4
_RIGHT_NEWER = 5


def _update_tag_relation(
    relation: int,
    left_reset: bool,
    right_reset: bool,
    left: _EngineState,
    right: _EngineState,
) -> int:
    if left_reset and right_reset:
        return _EQUAL
    if left_reset:
        return _LEFT_NEWER if right.tagged else _LEFT_ONLY
    if right_reset:
        return _RIGHT_NEWER if left.tagged else _RIGHT_ONLY
    return relation


def _is_mismatch(left: _EngineState, right: _EngineState, relation: int) -> bool:
    if left.tagged != right.tagged:
        return True
    return left.tagged and relation != _EQUAL


def _alphabet_representatives(left: _Program, right: _Program) -> Tuple[int, ...]:
    boundaries = {0, UNICODE_MAX + 1}
    for program in (left, right):
        for instruction in program.instructions:
            if instruction.chars is None:
                continue
            for lo, hi in instruction.chars.ranges:
                boundaries.add(lo)
                if hi < UNICODE_MAX:
                    boundaries.add(hi + 1)
    ordered = sorted(boundaries)
    return tuple(ordered[index] for index in range(len(ordered) - 1))


def _witness(
    key: Tuple[_EngineState, _EngineState, int],
    parents: Dict[
        Tuple[_EngineState, _EngineState, int],
        Optional[Tuple[Tuple[_EngineState, _EngineState, int], int]],
    ],
) -> str:
    codepoints: List[int] = []
    while parents[key] is not None:
        parent, codepoint = parents[key]  # type: ignore[misc]
        codepoints.append(codepoint)
        key = parent
    codepoints.reverse()
    return "".join(chr(codepoint) for codepoint in codepoints)


def _language_witness(
    key: Tuple[Tuple[int, ...], Tuple[int, ...]],
    parents: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...]],
        Optional[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], int]],
    ],
) -> str:
    codepoints: List[int] = []
    while parents[key] is not None:
        parent, codepoint = parents[key]  # type: ignore[misc]
        codepoints.append(codepoint)
        key = parent
    codepoints.reverse()
    return "".join(chr(codepoint) for codepoint in codepoints)


def _verify_language_programs(
    left_program: _Program,
    right_program: _Program,
    *,
    max_product_states: int,
) -> Verification:
    """Prove equality of the complete accepted languages."""

    left_machine = _LanguageMachine(left_program)
    right_machine = _LanguageMachine(right_program)
    start = (left_machine.initial, right_machine.initial)
    parents: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...]],
        Optional[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], int]],
    ] = {start: None}
    queue = deque([start])

    if left_machine.accepting(start[0]) != right_machine.accepting(start[1]):
        return Verification(
            Verdict.REJECTED,
            "the complete languages differ on the empty input",
            "",
            explored_states=1,
        )

    alphabet = _alphabet_representatives(left_program, right_program)
    while queue:
        left_state, right_state = current = queue.popleft()
        for codepoint in alphabet:
            following = (
                left_machine.step(left_state, codepoint),
                right_machine.step(right_state, codepoint),
            )
            if following in parents:
                continue
            parents[following] = (current, codepoint)
            if len(parents) > max_product_states:
                return Verification(
                    Verdict.UNKNOWN,
                    f"language product automaton exceeded {max_product_states} states",
                    explored_states=len(parents),
                )
            if left_machine.accepting(following[0]) != right_machine.accepting(
                following[1]
            ):
                return Verification(
                    Verdict.REJECTED,
                    "the complete accepted languages differ",
                    _language_witness(following, parents),
                    explored_states=len(parents),
                )
            queue.append(following)

    return Verification(
        Verdict.PROVED,
        "the complete language product was exhausted without a difference",
        explored_states=len(parents),
    )


def verify_language_equivalent(
    original: str,
    candidate: str,
    *,
    max_nfa_states: int = 2000,
    max_product_states: int = 200_000,
) -> Verification:
    """Prove bidirectional equality of the complete regular languages."""

    try:
        left_program = _compile(parse_regex(original), max_nfa_states)
        right_program = _compile(parse_regex(candidate), max_nfa_states)
    except (RegexError, StateLimitExceeded) as error:
        return Verification(Verdict.UNKNOWN, str(error))
    return _verify_language_programs(
        left_program,
        right_program,
        max_product_states=max_product_states,
    )


def _verify_match_programs(
    left_program: _Program,
    right_program: _Program,
    *,
    max_product_states: int,
) -> Verification:
    """Prove anchored success/failure and selected-end equality only."""

    left_machine = _OrderedMachine(left_program)
    right_machine = _OrderedMachine(right_program)
    relation = _update_tag_relation(
        _NONE,
        left_machine.initial_reset,
        right_machine.initial_reset,
        left_machine.initial,
        right_machine.initial,
    )
    start = (left_machine.initial, right_machine.initial, relation)
    parents: Dict[
        Tuple[_EngineState, _EngineState, int],
        Optional[Tuple[Tuple[_EngineState, _EngineState, int], int]],
    ] = {start: None}
    queue = deque([start])

    if _is_mismatch(*start):
        return Verification(
            Verdict.REJECTED,
            "the empty input has a different match result",
            "",
            left_machine.match_end(""),
            right_machine.match_end(""),
            1,
        )

    alphabet = _alphabet_representatives(left_program, right_program)
    while queue:
        current = queue.popleft()
        left_state, right_state, old_relation = current
        for codepoint in alphabet:
            next_left, left_reset = left_machine.step(left_state, codepoint)
            next_right, right_reset = right_machine.step(right_state, codepoint)
            next_relation = _update_tag_relation(
                old_relation, left_reset, right_reset, next_left, next_right
            )
            following = (next_left, next_right, next_relation)
            if following in parents:
                continue
            parents[following] = (current, codepoint)
            if len(parents) > max_product_states:
                return Verification(
                    Verdict.UNKNOWN,
                    f"product automaton exceeded {max_product_states} states",
                    explored_states=len(parents),
                )
            if _is_mismatch(*following):
                example = _witness(following, parents)
                return Verification(
                    Verdict.REJECTED,
                    "success/failure or the selected match end differs",
                    example,
                    left_machine.match_end(example),
                    right_machine.match_end(example),
                    len(parents),
                )
            queue.append(following)

    return Verification(
        Verdict.PROVED,
        "the ordered-NFA product was exhausted without a match-result difference",
        explored_states=len(parents),
    )


def verify_match_equivalent(
    original: str,
    candidate: str,
    *,
    max_nfa_states: int = 2000,
    max_product_states: int = 200_000,
) -> Verification:
    """Prove anchored leftmost-first success/failure and selected-end equality.

    Complete accepted languages may differ.  The proof applies only to the
    expression as a complete match, so substitution into a larger pattern
    requires another proof in that context.  REJECTED supplies a shortest
    match-result counterexample; UNKNOWN never authorizes a rewrite.
    """

    if original == candidate:
        return Verification(
            Verdict.PROVED, "the two expressions are identical", explored_states=1
        )
    try:
        left_program = _compile(parse_regex(original), max_nfa_states)
        right_program = _compile(parse_regex(candidate), max_nfa_states)
    except (RegexError, StateLimitExceeded) as error:
        return Verification(Verdict.UNKNOWN, str(error))
    return _verify_match_programs(
        left_program,
        right_program,
        max_product_states=max_product_states,
    )


def verify_priority_equivalent(
    original: str,
    candidate: str,
    *,
    max_nfa_states: int = 2000,
    max_product_states: int = 200_000,
) -> Verification:
    """Prove complete-language and anchored match-result equality.

    This stronger standalone contract still requires verification after
    substitution into an arbitrary caller context.  UNKNOWN is fail-closed.
    """

    if original == candidate:
        return Verification(
            Verdict.PROVED, "the two expressions are identical", explored_states=1
        )
    try:
        left_program = _compile(parse_regex(original), max_nfa_states)
        right_program = _compile(parse_regex(candidate), max_nfa_states)
    except (RegexError, StateLimitExceeded) as error:
        return Verification(Verdict.UNKNOWN, str(error))
    language_verification = _verify_language_programs(
        left_program,
        right_program,
        max_product_states=max_product_states,
    )
    if language_verification.verdict is not Verdict.PROVED:
        return language_verification
    match_verification = _verify_match_programs(
        left_program,
        right_program,
        max_product_states=max_product_states,
    )
    if match_verification.verdict is not Verdict.PROVED:
        return match_verification
    return Verification(
        Verdict.PROVED,
        "complete-language and ordered-NFA products were exhausted without a difference",
        explored_states=match_verification.explored_states,
    )


def prioritized_match_end(
    pattern: str, text: str, *, max_nfa_states: int = 2000
) -> Optional[int]:
    """Run the same anchored matcher used by the verifier (mainly for tests)."""

    return _OrderedMachine(_compile(parse_regex(pattern), max_nfa_states)).match_end(
        text
    )


# ---------------------------------------------------------------------------
# Candidate generation and optimizer


def _walk(
    node: Node, path: Tuple[int, ...] = ()
) -> Iterator[Tuple[Tuple[int, ...], Node]]:
    yield path, node
    if isinstance(node, (Concatenation, Alternation)):
        children = node.parts if isinstance(node, Concatenation) else node.options
        for index, child in enumerate(children):
            yield from _walk(child, path + (index,))
    elif isinstance(node, Repetition):
        yield from _walk(node.body, path + (0,))


def _contains_effective_reluctant_quantifier(node: Node) -> bool:
    if isinstance(node, Repetition):
        return (
            not node.greedy and node.minimum != node.maximum
        ) or _contains_effective_reluctant_quantifier(node.body)
    if isinstance(node, Concatenation):
        return any(
            _contains_effective_reluctant_quantifier(part) for part in node.parts
        )
    if isinstance(node, Alternation):
        return any(
            _contains_effective_reluctant_quantifier(option) for option in node.options
        )
    return False


def _replace(node: Node, path: Tuple[int, ...], replacement: Node) -> Node:
    if not path:
        return replacement
    index, rest = path[0], path[1:]
    if isinstance(node, Concatenation):
        parts = list(node.parts)
        parts[index] = _replace(parts[index], rest, replacement)
        return _concat(parts)
    if isinstance(node, Alternation):
        options = list(node.options)
        options[index] = _replace(options[index], rest, replacement)
        return _alternate(options)
    if isinstance(node, Repetition) and index == 0:
        return Repetition(
            _replace(node.body, rest, replacement),
            node.minimum,
            node.maximum,
            node.greedy,
        )
    raise AssertionError("invalid AST path")


def _charset_intersection(left: CharSet, right: CharSet) -> CharSet:
    ranges: List[Tuple[int, int]] = []
    left_index = right_index = 0
    while left_index < len(left.ranges) and right_index < len(right.ranges):
        left_lo, left_hi = left.ranges[left_index]
        right_lo, right_hi = right.ranges[right_index]
        lo, hi = max(left_lo, right_lo), min(left_hi, right_hi)
        if lo <= hi:
            ranges.append((lo, hi))
        if left_hi < right_hi:
            left_index += 1
        else:
            right_index += 1
    return CharSet.make(ranges)


def _charset_difference(left: CharSet, right: CharSet) -> CharSet:
    return _charset_intersection(left, right.complement())


def _charset_subset(left: CharSet, right: CharSet) -> bool:
    return not _charset_difference(left, right).ranges


def _atom_repetition(
    node: Node,
    *,
    minimum: Optional[int] = None,
    maximum: object = Ellipsis,
    greedy: Optional[bool] = None,
) -> Optional[Tuple[Repetition, Atom]]:
    if not isinstance(node, Repetition) or not isinstance(node.body, Atom):
        return None
    if minimum is not None and node.minimum != minimum:
        return None
    if maximum is not Ellipsis and node.maximum != maximum:
        return None
    if greedy is not None and node.greedy != greedy:
        return None
    return node, node.body


def _has_single_fixed_path(node: Node) -> bool:
    """Whether ``node`` consumes a fixed sequence without fallback choices."""

    if isinstance(node, (Empty, Atom)):
        return True
    if isinstance(node, Concatenation):
        return all(_has_single_fixed_path(part) for part in node.parts)
    if isinstance(node, Repetition):
        return node.minimum == node.maximum and _has_single_fixed_path(node.body)
    # Even a fixed-width alternation has multiple fallback paths, so it cannot
    # be moved across an optional without changing the ordered search.
    return False


def _fixed_path_sequence(node: Node) -> Optional[Tuple[Node, ...]]:
    """Return a fixed-width, structurally unambiguous top-level sequence.

    Exact repetitions such as ``B{2}`` have one fallback-free path and are
    safe to factor.  Variable-width repetitions and alternations are excluded.
    """

    if isinstance(node, Empty):
        return ()
    parts = node.parts if isinstance(node, Concatenation) else (node,)
    if not all(_has_single_fixed_path(part) for part in parts):
        return None
    return tuple(parts)


def _structural_paths(
    node: Node, *, limit: int = 256
) -> Optional[Tuple[Tuple[Node, ...], ...]]:
    """Expand nested alternatives into bounded structural concatenation paths.

    Repetitions remain opaque nodes.  This is enough to recognize both
    ``AB* | AB*C`` and flattened shapes such as
    ``AB | AC | (AB|AC)D`` without constructing their regular languages.
    ``None`` means the bounded expansion was too large to inspect safely.
    """

    if isinstance(node, Empty):
        return ((),)
    if isinstance(node, Alternation):
        paths: List[Tuple[Node, ...]] = []
        for option in node.options:
            option_paths = _structural_paths(option, limit=limit)
            if option_paths is None or len(paths) + len(option_paths) > limit:
                return None
            paths.extend(option_paths)
        return tuple(paths)
    if isinstance(node, Concatenation):
        paths: List[Tuple[Node, ...]] = [()]
        for part in node.parts:
            part_paths = _structural_paths(part, limit=limit)
            if part_paths is None or len(paths) * len(part_paths) > limit:
                return None
            paths = [prefix + suffix for prefix in paths for suffix in part_paths]
        return tuple(paths)
    return ((node,),)


def _is_strict_prefix(shorter: Sequence[Node], longer: Sequence[Node]) -> bool:
    return len(shorter) < len(longer) and tuple(longer[: len(shorter)]) == tuple(
        shorter
    )


def _is_strict_suffix(shorter: Sequence[Node], longer: Sequence[Node]) -> bool:
    return len(shorter) < len(longer) and tuple(longer[-len(shorter) :]) == tuple(
        shorter
    )


def _ordered_optional_replacements(node: Node) -> Iterator[Node]:
    """Factor a two-branch ordered choice without losing fallback order.

    ``X | X Y`` must become ``X Y??`` because the shorter branch is tried
    first, whereas ``X Y | X`` becomes the ordinary greedy ``X Y?``.  The
    symmetric common-suffix forms follow the same rule.  Only fixed,
    single-path sequences are admitted so the factoring is safe in every
    surrounding concatenation, not merely as a standalone match.
    """

    if not isinstance(node, Alternation) or len(node.options) != 2:
        return
    sequences = tuple(_fixed_path_sequence(option) for option in node.options)
    if any(sequence is None for sequence in sequences):
        return

    for long_index, short_index in ((0, 1), (1, 0)):
        longer = sequences[long_index]
        shorter = sequences[short_index]
        assert longer is not None and shorter is not None
        greedy = long_index < short_index

        if _is_strict_prefix(shorter, longer):
            tail = _concat(longer[len(shorter) :])
            yield _concat((*shorter, Repetition(tail, 0, 1, greedy)))

        if _is_strict_suffix(shorter, longer):
            prefix = _concat(longer[: -len(shorter)])
            yield _concat((Repetition(prefix, 0, 1, greedy), *shorter))


def _generate_order_preserving_candidates(
    pattern: str, *, limit: int = 32
) -> Tuple[List[str], bool]:
    """Return certified ordered-factor proposals and whether one is required.

    A strict prefix/suffix relationship between alternatives is priority
    sensitive: an ordinary language reducer may turn it into a greedy optional
    and silently reverse the shorter/longer fallback order.  Two alternatives
    can be factored by the theorem above.  Larger ordered choices are marked as
    hazards and left unchanged until an equally explicit construction exists.
    """

    root = parse_regex(pattern)
    candidates: Set[str] = set()
    has_hazard = False
    for path, subtree in _walk(root):
        if not isinstance(subtree, Alternation):
            continue
        # Detect ordered containment for every structural branch, not only for
        # branches that are safe to factor.  In particular, ``AB* | AB*C`` has
        # a strict common-prefix relationship, but factoring it as ``AB*C??``
        # interleaves the fallback choices of ``B*`` and ``C??``.  Marking the
        # relationship here makes the optimizer fail closed before a language
        # reducer can replace it with the even less faithful greedy ``C?``.
        path_groups: List[Tuple[Tuple[Node, ...], ...]] = []
        total_paths = 0
        for option in subtree.options:
            option_paths = _structural_paths(option)
            if option_paths is None or total_paths + len(option_paths) > 256:
                has_hazard = True
                path_groups = []
                break
            path_groups.append(option_paths)
            total_paths += len(option_paths)

        for left_index, left_group in enumerate(path_groups):
            for right_group in path_groups[left_index + 1 :]:
                if any(
                    _is_strict_prefix(left, right)
                    or _is_strict_prefix(right, left)
                    or _is_strict_suffix(left, right)
                    or _is_strict_suffix(right, left)
                    for left in left_group
                    for right in right_group
                ):
                    has_hazard = True
                    break
            if has_hazard:
                break

        for replacement in _ordered_optional_replacements(subtree):
            whole = to_regex(_replace(root, path, replacement))
            if whole != pattern:
                candidates.add(whole)
                if len(candidates) >= limit:
                    return (
                        sorted(candidates, key=lambda item: (len(item), item)),
                        has_hazard,
                    )
    return sorted(candidates, key=lambda item: (len(item), item)), has_hazard


def _generate_match_pruning_candidates(root: Node, *, limit: int = 32) -> List[str]:
    """Propose bounded deletions of lower-priority alternatives.

    Keeping the first branch and removing later branches can expose hidden
    language differences.  These proposals have no compositional certificate:
    the exact match verifier must check the entire supplied expression.
    """

    if limit <= 0:
        return []
    candidates: Set[str] = set()
    for path, subtree in _walk(root):
        if not isinstance(subtree, Alternation):
            continue
        options = subtree.options
        # Prefixes include the first-branch-only proposal and remove a whole
        # fallback tail.  Individual deletions also retain useful later paths.
        replacements = (_alternate(options[:kept]) for kept in range(1, len(options)))
        for replacement in replacements:
            candidates.add(to_regex(_replace(root, path, replacement)))
            if len(candidates) >= limit:
                return sorted(candidates, key=lambda item: (len(item), item))
        for removed in range(1, len(options) - 1):
            replacement = _alternate(options[:removed] + options[removed + 1 :])
            candidates.add(to_regex(_replace(root, path, replacement)))
            if len(candidates) >= limit:
                return sorted(candidates, key=lambda item: (len(item), item))
    return sorted(candidates, key=lambda item: (len(item), item))


def _fast_local_replacements(node: Node) -> Iterator[Node]:
    """Yield cheap algebraic proposals; the exact verifier remains mandatory.

    These transformations deliberately operate only on compact atom/repetition
    shapes.  They are proposals, not trusted rewrite theorems: callers must
    still prove equivalence under their selected observation contract.
    """

    if isinstance(node, Concatenation) and len(node.parts) == 2:
        left_info = _atom_repetition(node.parts[0], maximum=None, greedy=True)
        right_info = _atom_repetition(node.parts[1], maximum=None, greedy=True)
        if left_info is not None and right_info is not None:
            left, left_atom = left_info
            right, right_atom = right_info
            left_subset = _charset_subset(left_atom.chars, right_atom.chars)
            right_subset = _charset_subset(right_atom.chars, left_atom.chars)

            if left_subset and left.minimum in (0, 1):
                if left.minimum == 0:
                    yield right
                else:
                    yield _concat((left_atom, right))
            if right_subset and right.minimum == 0 and left.minimum >= 1:
                yield left

            intersection = _charset_intersection(left_atom.chars, right_atom.chars)
            right_only = _charset_difference(right_atom.chars, left_atom.chars)
            if (
                left.minimum in (0, 1)
                and right.minimum == 0
                and intersection.ranges
                and right_only.ranges
                and not left_subset
            ):
                suffix = _concat((Atom(right_only), right))
                yield _concat((left, Repetition(suffix, 0, 1, True)))

    if isinstance(node, Concatenation) and len(node.parts) == 3:
        first_info = _atom_repetition(node.parts[0], maximum=None)
        third_info = _atom_repetition(node.parts[2], maximum=None, greedy=True)
        middle = node.parts[1]
        if first_info is not None and third_info is not None:
            first, first_atom = first_info
            third, third_atom = third_info

            # X*? Y X* and X* Y X*: reserve the first Y from the prefix.
            if isinstance(middle, Atom) and first_atom.chars == third_atom.chars:
                difference = _charset_difference(first_atom.chars, middle.chars)
                if (
                    first.minimum == 0
                    and third.minimum == 0
                    and _charset_subset(middle.chars, first_atom.chars)
                    and difference.ranges
                ):
                    yield _concat(
                        (
                            Repetition(Atom(difference), 0, None, True),
                            middle,
                            third,
                        )
                    )
                if (
                    first.minimum == 1
                    and first.greedy
                    and third.minimum == 1
                    and _charset_subset(middle.chars, first_atom.chars)
                    and difference.ranges
                ):
                    yield _concat(
                        (
                            first_atom,
                            Repetition(Atom(difference), 0, None, True),
                            middle,
                            third,
                        )
                    )

            middle_info = _atom_repetition(middle)
            if middle_info is not None and first_atom.chars == third_atom.chars:
                middle_rep, middle_atom = middle_info
                if (
                    first.greedy
                    and third.greedy
                    and first.minimum in (0, 1)
                    and middle_rep.minimum == 0
                    and third.minimum in (0, 1)
                ):
                    # X{*,+} Y? X{*,+}
                    if middle_rep.maximum == 1:
                        suffix = _concat((middle_atom, third))
                        if third.minimum == 0:
                            yield _concat((first, Repetition(suffix, 0, 1, True)))
                        elif first.minimum == 0 and third.minimum == 1:
                            leading = Repetition(first_atom, 1, None, True)
                            yield _alternate(
                                (
                                    _concat(
                                        (
                                            leading,
                                            Repetition(suffix, 0, 1, True),
                                        )
                                    ),
                                    suffix,
                                )
                            )

                    # X{*,+} Y* X*: the trailing X rows are needed only
                    # after at least one disjoint Y row.
                    if (
                        middle_rep.maximum is None
                        and middle_rep.greedy
                        and third.minimum == 0
                        and not _charset_intersection(
                            first_atom.chars, middle_atom.chars
                        ).ranges
                    ):
                        required_middle = Repetition(middle_atom, 1, None, True)
                        suffix = _concat((required_middle, third))
                        yield _concat((first, Repetition(suffix, 0, 1, True)))

    if isinstance(node, Repetition) and isinstance(node.body, Alternation):
        # (X+ | Y+)+ can often collapse to (X union Y)+.  Priority safety is
        # decided only after substitution into the complete surrounding regex.
        branch_atoms: List[Atom] = []
        if node.minimum == 1 and node.maximum is None and node.greedy:
            for option in node.body.options:
                info = _atom_repetition(option, minimum=1, maximum=None, greedy=True)
                if info is None:
                    break
                branch_atoms.append(info[1])
            else:
                ranges = [item for atom in branch_atoms for item in atom.chars.ranges]
                yield Repetition(Atom(CharSet.make(ranges)), 1, None, True)


def generate_fast_priority_candidates(pattern: str, *, limit: int = 32) -> List[str]:
    """Generate bounded structural candidates without invoking ``greenery``."""

    root = parse_regex(pattern)
    ordered_candidates, _ = _generate_order_preserving_candidates(pattern, limit=limit)
    candidates = set(ordered_candidates)
    if len(candidates) >= limit:
        return sorted(candidates, key=lambda item: (len(item), item))
    for path, subtree in _walk(root):
        for replacement in _fast_local_replacements(subtree):
            whole = to_regex(_replace(root, path, replacement))
            if whole != pattern:
                candidates.add(whole)
                if len(candidates) >= limit:
                    return sorted(candidates, key=lambda item: (len(item), item))
    return sorted(candidates, key=lambda item: (len(item), item))


@lru_cache(maxsize=2048)
def _greenery_reduce(source: str) -> Optional[str]:
    try:
        from greenery import parse  # type: ignore[import-not-found]
        from greenery.rxelems import from_fsm  # type: ignore[import-not-found]

        return str(from_fsm(parse(source).to_fsm().reduce()))
    except Exception:
        return None


def generate_language_candidates(pattern: str, *, limit: int = 128) -> List[str]:
    """Generate unchecked, ordinary-language candidates.

    The whole tree and every subtree are independently passed through
    ``greenery``.  Replacing arbitrary subtrees makes this useful for nested
    alternations/repetitions without encoding a theorem for each shape.
    Returned candidates are *not* safe until the selected-contract verifier
    proves them in the complete surrounding expression.
    """

    root = parse_regex(pattern)
    candidates = {pattern}
    for path, subtree in _walk(root):
        reduced = _greenery_reduce(to_regex(subtree))
        if not reduced:
            continue
        try:
            replacement = parse_regex(reduced)
            # Ignore changes that only spell grouping differently. They do not
            # alter the regex AST and become especially noisy when the caller
            # permits candidates longer than the source.
            if replacement == subtree:
                continue
            whole = to_regex(_replace(root, path, replacement))
            parse_regex(whole)
        except RegexError:
            continue
        candidates.add(whole)
        if len(candidates) >= limit:
            break
    return sorted(candidates, key=lambda item: (len(item), item))


@dataclass(frozen=True)
class OptimizationResult:
    original: str
    optimized: str
    verification: Verification
    candidates_checked: int
    rejected_candidates: int
    unknown_candidates: int
    proof_scope: ProofScope = ProofScope.STANDALONE

    @property
    def changed(self) -> bool:
        return self.original != self.optimized


def optimize_priority_regex(
    pattern: str,
    *,
    equivalence_mode: EquivalenceMode = EquivalenceMode.LANGUAGE_AND_MATCH,
    extra_candidates: Iterable[str] = (),
    max_candidates: int = 128,
    max_output_length_factor: float = 4.0,
    max_output_length: Optional[int] = None,
    max_nfa_states: int = 2000,
    max_product_states: int = 200_000,
) -> OptimizationResult:
    """Select a proved priority-equivalent rewrite.

    By default, candidates may be up to four times the source length.  Change
    ``max_output_length_factor`` to 5 (or another positive value) to adjust the
    relative limit.  An explicit ``max_output_length`` overrides that computed
    limit.  Cheap structural proposals are tried first; if none is proved, the
    shortest proved Greenery candidate is selected.  The source is always
    retained as the fail-safe fallback.  A source containing an effective
    reluctant quantifier is returned unchanged before structural, external, or
    Greenery candidates are considered.

    ``MATCH_ONLY`` first tries bounded lower-priority branch deletions and
    accepts generic candidates using only anchored match-result equality.
    These candidates carry a standalone proof that must be checked again in
    their complete caller context.  In this mode ``max_candidates`` bounds
    the total verification attempts across all candidate classes; the default
    mode retains its existing Greenery-generation limit.
    """

    # Parse up front so an invalid source is reported to the caller instead of
    # being disguised as an optimizer no-op.
    root = parse_regex(pattern)
    equivalence_mode = EquivalenceMode(equivalence_mode)
    if max_output_length_factor <= 0:
        raise ValueError("max_output_length_factor must be positive")
    if max_output_length is not None and max_output_length < 0:
        raise ValueError("max_output_length must be non-negative or None")
    effective_max_output_length = (
        max_output_length
        if max_output_length is not None
        else int(len(pattern) * max_output_length_factor)
    )
    best_verification = Verification(
        Verdict.PROVED,
        "the original expression is retained",
        explored_states=1,
    )
    checked = rejected = unknown = 0

    def retained_result() -> OptimizationResult:
        return OptimizationResult(
            pattern,
            pattern,
            best_verification,
            checked,
            rejected,
            unknown,
            ProofScope.COMPOSITIONAL,
        )

    def budget_exhausted() -> bool:
        return (
            equivalence_mode is EquivalenceMode.MATCH_ONLY and checked >= max_candidates
        )

    generic_verifier = (
        verify_match_equivalent
        if equivalence_mode is EquivalenceMode.MATCH_ONLY
        else verify_priority_equivalent
    )

    # Equality of the complete language and the preferred standalone match end
    # is not a congruence for concatenation: a surrounding suffix can force a
    # reluctant expression onto a lower-priority fallback path.  This applies
    # even when a structural candidate leaves the reluctant quantifier itself
    # intact and rewrites only another subtree, so reject every candidate class
    # before candidate generation.  Context-certified reluctant rewrites live
    # in RegexOptimizationPlusRule and reach this optimizer only after all
    # effective reluctance has been removed.
    if _contains_effective_reluctant_quantifier(root):
        return retained_result()

    if budget_exhausted():
        return retained_result()

    # Match-only pruning must precede both the compositional factoring early
    # return and its unresolved-order guard, which intentionally preserve all
    # fallback branches under the default language-and-match contract.
    pruning_pool: Set[str] = set()
    if equivalence_mode is EquivalenceMode.MATCH_ONLY:
        pruning_pool = set(
            _generate_match_pruning_candidates(root, limit=min(32, max_candidates))
        )
        for candidate in sorted(pruning_pool, key=lambda item: (len(item), item)):
            if candidate == pattern or len(candidate) > effective_max_output_length:
                continue
            if budget_exhausted():
                break
            checked += 1
            verification = verify_match_equivalent(
                pattern,
                candidate,
                max_nfa_states=max_nfa_states,
                max_product_states=max_product_states,
            )
            if verification.verdict is Verdict.PROVED:
                return OptimizationResult(
                    pattern,
                    candidate,
                    verification,
                    checked,
                    rejected,
                    unknown,
                    ProofScope.STANDALONE,
                )
            if verification.verdict is Verdict.REJECTED:
                rejected += 1
            else:
                unknown += 1
        if budget_exhausted():
            return retained_result()

    # Ordered prefix/suffix factoring has a stronger contract than ordinary
    # standalone equivalence: it preserves the complete fallback order under
    # every surrounding concatenation.  Try those certified proposals first.
    ordered_pool, has_ordered_hazard = _generate_order_preserving_candidates(pattern)
    for candidate in ordered_pool:
        if len(candidate) > effective_max_output_length:
            continue
        if budget_exhausted():
            break
        checked += 1
        verification = verify_priority_equivalent(
            pattern,
            candidate,
            max_nfa_states=max_nfa_states,
            max_product_states=max_product_states,
        )
        if verification.verdict is Verdict.PROVED:
            return OptimizationResult(
                pattern,
                candidate,
                verification,
                checked,
                rejected,
                unknown,
                ProofScope.COMPOSITIONAL,
            )
        if verification.verdict is Verdict.REJECTED:
            rejected += 1
        else:
            unknown += 1

    # If an ordered containment relationship remains unresolved, an ordinary
    # language reduction could recreate it with the wrong greediness.  Fail
    # closed instead of falling through to untrusted fast/Greenery proposals.
    if has_ordered_hazard or budget_exhausted():
        return retained_result()

    # Try the remaining bounded structural proposals before importing/building
    # Greenery automata.  Every proposal goes through the exact standalone
    # verifier; priority-sensitive ordered factoring was handled above.
    fast_pool = (
        set(generate_fast_priority_candidates(pattern))
        - set(ordered_pool)
        - pruning_pool
    )
    fast_pool.update(extra_candidates)
    fast_pool.difference_update(pruning_pool)
    for candidate in sorted(fast_pool, key=lambda item: (len(item), item)):
        if candidate == pattern:
            continue
        if len(candidate) > effective_max_output_length:
            continue
        if budget_exhausted():
            break
        checked += 1
        verification = generic_verifier(
            pattern,
            candidate,
            max_nfa_states=max_nfa_states,
            max_product_states=max_product_states,
        )
        if verification.verdict is Verdict.PROVED:
            return OptimizationResult(
                pattern,
                candidate,
                verification,
                checked,
                rejected,
                unknown,
                ProofScope.STANDALONE,
            )
        elif verification.verdict is Verdict.REJECTED:
            rejected += 1
        else:
            unknown += 1

    if budget_exhausted():
        return retained_result()
    generated = generate_language_candidates(pattern, limit=max_candidates)
    pool = set(generated) - fast_pool - pruning_pool
    for candidate in sorted(pool, key=lambda item: (len(item), item)):
        if candidate == pattern or len(candidate) > effective_max_output_length:
            continue
        if budget_exhausted():
            break
        checked += 1
        verification = generic_verifier(
            pattern,
            candidate,
            max_nfa_states=max_nfa_states,
            max_product_states=max_product_states,
        )
        if verification.verdict is Verdict.PROVED:
            # Candidates are ordered by the optimizer cost function, so no
            # later candidate can be preferred.
            return OptimizationResult(
                pattern,
                candidate,
                verification,
                checked,
                rejected,
                unknown,
                ProofScope.STANDALONE,
            )
        if verification.verdict is Verdict.REJECTED:
            rejected += 1
        else:
            unknown += 1

    return retained_result()


__all__ = [
    "EquivalenceMode",
    "OptimizationResult",
    "ProofScope",
    "RegexError",
    "Verdict",
    "Verification",
    "generate_language_candidates",
    "generate_fast_priority_candidates",
    "optimize_priority_regex",
    "parse_regex",
    "prioritized_match_end",
    "to_regex",
    "verify_language_equivalent",
    "verify_match_equivalent",
    "verify_priority_equivalent",
]
