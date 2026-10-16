"""Ordered, output-sensitive automata for Ultra (independent of Plus).

Unlike ordinary DFA equivalence, the observation is the preferred word of
observable label-membership vectors, including its length. With CLASSIFIER it
distinguishes every label; otherwise SQL may observe only selected groups.
Thompson threads carry output registers.
All live registers have consumed the same input length. On a step their
outputs are equal iff their old outputs and appended labels are equal; a
saved earlier acceptance cannot equal a newly extended live output. Thus a
finite equality partition, rather than unbounded strings, is an exact
abstraction. BFS of the product proves equivalence for ALL finite inputs.

The alphabet is a sound predicate abstraction supplied by predicates_ultra.
This uses the symbolic guard/product principle of D'Antoni and Veanes,
POPL 2014, https://doi.org/10.1145/2535838.2535849, extended with priority and
output observations. It is not an implementation of the paper's MinN_SFA.
"""

from dataclasses import dataclass
from collections import deque
from time import monotonic

from patternnode import (
    AlternationPattern,
    ConcatenationPattern,
    EmptyPattern,
    QuantifiedPattern,
    VariablePattern,
    AnchorPattern,
    AnchorType,
    ExcludedPattern,
)


class UnsupportedUltra(ValueError):
    """A construct outside the implemented proof model."""


class BudgetUltra(RuntimeError):
    """An inconclusive proof/search; never permission to rewrite."""


def children(node):
    if isinstance(node, (ConcatenationPattern, AlternationPattern)):
        return node.patterns
    if isinstance(node, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
        return [node.pattern]
    if isinstance(node, (VariablePattern, EmptyPattern)):
        return []
    raise UnsupportedUltra(type(node).__name__)


def walk(node):
    yield node
    for child in children(node):
        yield from walk(child)


def names(node):
    return tuple(
        dict.fromkeys(n.name for n in walk(node) if isinstance(n, VariablePattern))
    )


def nullable(node):
    if isinstance(node, EmptyPattern):
        return True
    if isinstance(node, VariablePattern):
        return False
    if isinstance(node, ConcatenationPattern):
        return all(map(nullable, node.patterns))
    if isinstance(node, AlternationPattern):
        return any(map(nullable, node.patterns))
    if isinstance(node, QuantifiedPattern):
        return node.atLeast == 0 or nullable(node.pattern)
    if isinstance(node, (AnchorPattern, ExcludedPattern)):
        return nullable(node.pattern)
    raise UnsupportedUltra(type(node).__name__)


@dataclass
class InstructionUltra:
    op: str
    label: str = ""
    first: int = -1
    second: int = -1
    excluded: bool = False


class ProgramUltra:
    def __init__(self, pattern, max_states=256, observe=None):
        self.code = []
        self.observe = observe if observe is not None else lambda label: label
        self.max_states = max_states
        end = self.emit(InstructionUltra("accept"))
        self.start = self.compile(pattern, end)

    def emit(self, instruction):
        if len(self.code) >= self.max_states:
            raise BudgetUltra("NFA state limit")
        self.code.append(instruction)
        return len(self.code) - 1

    def compile(self, node, tail, excluded=False):
        if isinstance(node, EmptyPattern):
            return tail
        if isinstance(node, VariablePattern):
            return self.emit(
                InstructionUltra("row", node.name, tail, excluded=excluded)
            )
        if isinstance(node, ExcludedPattern):
            return self.compile(node.pattern, tail, True)
        if isinstance(node, AnchorPattern):
            if node.anchor_type == AnchorType.START:
                return self.emit(
                    InstructionUltra(
                        "start", first=self.compile(node.pattern, tail, excluded)
                    )
                )
            return self.compile(
                node.pattern, self.emit(InstructionUltra("end", first=tail)), excluded
            )
        if isinstance(node, ConcatenationPattern):
            for child in reversed(node.patterns):
                tail = self.compile(child, tail, excluded)
            return tail
        if isinstance(node, AlternationPattern):
            if not node.patterns:
                raise UnsupportedUltra("empty alternation")
            start = self.compile(node.patterns[-1], tail, excluded)
            for child in reversed(node.patterns[:-1]):
                start = self.emit(
                    InstructionUltra(
                        "split", first=self.compile(child, tail, excluded), second=start
                    )
                )
            return start
        if isinstance(node, QuantifiedPattern):
            if nullable(node.pattern) and (node.atMost is None or node.atMost > 1):
                raise UnsupportedUltra("repeated nullable body")
            if node.atLeast > self.max_states or (node.atMost or 0) > self.max_states:
                raise BudgetUltra("quantifier bound")
            if node.atMost is None:
                split = self.emit(InstructionUltra("split"))
                body = self.compile(node.pattern, split, excluded)
                self.code[split].first, self.code[split].second = (
                    (body, tail) if node.isGreedy else (tail, body)
                )
                tail = split
            else:
                for _ in range(node.atMost - node.atLeast):
                    body = self.compile(node.pattern, tail, excluded)
                    first, second = (body, tail) if node.isGreedy else (tail, body)
                    tail = self.emit(
                        InstructionUltra("split", first=first, second=second)
                    )
            for _ in range(node.atLeast):
                tail = self.compile(node.pattern, tail, excluded)
            return tail
        raise UnsupportedUltra(type(node).__name__)

    def close(self, seeds, saved, *, at_start=False, at_end=False):
        """DFS priority, first arrival wins; discard below first acceptance."""
        stack, seen, live = list(reversed(seeds)), set(), []
        while stack:
            pc, output = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            instruction = self.code[pc]
            if instruction.op == "split":
                stack.extend(
                    ((instruction.second, output), (instruction.first, output))
                )
            elif instruction.op == "accept":
                return tuple(live), output
            elif instruction.op == "start":
                if at_start:
                    stack.append((instruction.first, output))
            elif instruction.op == "end":
                if at_end:
                    stack.append((instruction.first, output))
                else:
                    live.append((pc, output))
            else:
                if not at_end:
                    live.append((pc, output))
        return tuple(live), saved

    def step(self, state, accepted_labels):
        live, saved = state
        seeds = []
        for pc, register in live:
            instruction = self.code[pc]
            if instruction.op == "row" and instruction.label in accepted_labels:
                emitted = (self.observe(instruction.label), instruction.excluded)
                seeds.append((instruction.first, ("append", register, emitted)))
        # Kept acceptance is strictly shorter than all newly consumed outputs.
        kept = None if saved is None else ("keep", saved)
        return self.close(seeds, kept)

    def finish(self, state, at_start=False):
        live, saved = state
        return self.close(live, saved, at_start=at_start, at_end=True)[1]


def canonical_pair(left, right):
    ids = {}

    def register(value):
        if value is None:
            return None
        return ids.setdefault(value, len(ids))

    def state(value):
        live, saved = value
        return tuple((pc, register(output)) for pc, output in live), register(saved)

    return state(left), state(right)


@dataclass(frozen=True)
class ProofUltra:
    equivalent: bool
    states: int
    witness: tuple = ()
    at_partition_start: bool = True


def verify_labels(
    source, target, alphabet, *, deadline, max_states=20000, max_nfa=256, observe=None
):
    """alphabet contains (source labels, target labels) for each feasible row.

    UNKNOWN (timeout/unsupported/budget) raises; only equivalent=True certifies.
    Witness entries index alphabet cells, including rows satisfying no DEFINE.
    """
    left, right = ProgramUltra(source, max_nfa, observe), ProgramUltra(
        target, max_nfa, observe
    )
    initials = [
        (
            canonical_pair(
                left.close([(left.start, 0)], None, at_start=start),
                right.close([(right.start, 0)], None, at_start=start),
            ),
            start,
        )
        for start in (False, True)
    ]
    queue = deque(initials)
    parents = {initial: None for initial in initials}
    while queue:
        if monotonic() > deadline:
            raise BudgetUltra("proof timeout")
        pair, at_start = queue.popleft()
        if left.finish(pair[0], at_start) != right.finish(pair[1], at_start):
            witness, cursor = [], (pair, at_start)
            while parents[cursor] is not None:
                cursor, cell = parents[cursor]
                witness.append(cell)
            return ProofUltra(False, len(parents), tuple(reversed(witness)), cursor[1])
        for cell, (before, after) in enumerate(alphabet):
            following = (
                canonical_pair(left.step(pair[0], before), right.step(pair[1], after)),
                False,
            )
            if following not in parents:
                if len(parents) >= max_states:
                    raise BudgetUltra("product state limit")
                parents[following] = ((pair, at_start), cell)
                queue.append(following)
    return ProofUltra(True, len(parents))


def ambiguity_profile(pattern, alphabet, *, deadline, max_states=2000, max_nfa=256):
    """A generic progress metric: simultaneously viable cyclic/all threads.

    It is NOT an equivalence certificate. A larger regex can be worthwhile if
    region factoring removes competing repetition loops. Strictly decreasing
    this metric plus structural cost prevents repeated predicate expansion.
    """
    program = ProgramUltra(pattern, max_nfa)
    edges = {
        pc: (
            [ins.first, ins.second]
            if ins.op == "split"
            else [ins.first] if ins.op in {"row", "start", "end"} else []
        )
        for pc, ins in enumerate(program.code)
    }
    cyclic = set()
    for origin, instruction in enumerate(program.code):
        if instruction.op != "row":
            continue
        queue, seen = list(edges[origin]), set()
        while queue:
            pc = queue.pop()
            if pc == origin:
                cyclic.add(origin)
                break
            if pc not in seen:
                seen.add(pc)
                queue.extend(edges[pc])

    def close(pcs, start=False):
        return tuple(
            pc
            for pc, _ in program.close([(pc, 0) for pc in pcs], None, at_start=start)[0]
        )

    initial = close([program.start], True)
    pending, seen = deque([initial]), {initial}
    worst_cycle, worst_all = 0, 0
    for_alphabet = tuple(set(map(frozenset, alphabet)))
    while pending:
        if monotonic() > deadline:
            raise BudgetUltra("complexity analysis timeout")
        live = pending.popleft()
        for accepted in for_alphabet:
            matched = [
                pc
                for pc in live
                if program.code[pc].op == "row" and program.code[pc].label in accepted
            ]
            worst_cycle = max(worst_cycle, sum(pc in cyclic for pc in matched))
            worst_all = max(worst_all, len(matched))
            following = close([program.code[pc].first for pc in matched])
            if following not in seen:
                if len(seen) >= max_states:
                    raise BudgetUltra("complexity state limit")
                seen.add(following)
                pending.append(following)
    return worst_cycle, worst_all
