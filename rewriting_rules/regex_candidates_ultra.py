"""Predicate-region encoding, SQL decoding and regex candidate generation.

Every generator uses the same reversible alphabet and CandidateUltra records.
Greenery, alternative DFA state orders and structural pruning only propose
rewrites; regex_automata_ultra verifies their observable behavior.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import monotonic

from expression import BinaryExpr, CaseExpr, Literal, NotExpr
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

from .predicates_ultra import conjunction, conjuncts, expr_walk
from .regex_automata_ultra import BudgetUltra, UnsupportedUltra, children, names, walk

# Only immutable worker output is shared across rewrites. SQL guards, regions,
# observers and proof results must always be rebuilt in the caller's context.
_WORKER_CACHE = OrderedDict()
_WORKER_CACHE_LOCK = Lock()
_WORKER_CACHE_MAX_ENTRIES = 32
_WORKER_CACHE_MAX_BYTES = 4 * 1024 * 1024


def _worker_output(payload, *, phase, deadline, worker_seconds):
    timeout = min(worker_seconds, deadline - monotonic())
    if timeout <= 0:
        raise BudgetUltra("candidate timeout")
    key = (sys.executable, phase, payload)
    with _WORKER_CACHE_LOCK:
        if key in _WORKER_CACHE:
            _WORKER_CACHE.move_to_end(key)
            return _WORKER_CACHE[key], 0.0, True
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    started = monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "rewriting_rules.regex_candidates_ultra", phase],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        # Keep completed lines usable for this attempt, but do not cache an
        # incomplete search: a later attempt may have more time available.
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode()
        return output, monotonic() - started, False
    except subprocess.CalledProcessError as error:
        raise BudgetUltra("greenery worker failed") from error
    elapsed = monotonic() - started
    output = result.stdout
    if len(payload) + len(output) <= _WORKER_CACHE_MAX_BYTES:
        with _WORKER_CACHE_LOCK:
            _WORKER_CACHE[key] = output
            _WORKER_CACHE.move_to_end(key)
            while (
                len(_WORKER_CACHE) > _WORKER_CACHE_MAX_ENTRIES
                or sum(
                    len(item[-1]) + len(value) for item, value in _WORKER_CACHE.items()
                )
                > _WORKER_CACHE_MAX_BYTES
            ):
                _WORKER_CACHE.popitem(last=False)
    return output, elapsed, False


# Shared regions, character encoding and SQL reconstruction
#
# Shared predicate regions, reversible character encoding and SQL decoding.
#
# A character denotes a TRUE-membership region, not a pattern variable. An
# optional label colour retains distinctions that language minimization would
# otherwise erase. Both encodings feed the same decoder and exact verifier.
# No decoding step assumes a particular pattern shape.


def not_true(expression, algebra):
    if algebra.nonnull(expression):
        return NotExpr(copy.deepcopy(expression))
    return BinaryExpr(
        CaseExpr([(copy.deepcopy(expression), Literal(0))], Literal(1)), "=", Literal(1)
    )


@dataclass
class CandidateUltra:
    pattern: object
    guards: dict
    aliases: dict
    generator: str
    encoding: str


class RegionsUltra:
    def __init__(self, algebra, observer, *, max_regions=128, max_predicate_nodes=256):
        self.algebra, self.observer = algebra, observer
        self.labels = tuple(algebra.expressions)
        grouped = {}
        for index in range(len(algebra.valuations)):
            signature = frozenset(
                name for name in self.labels if index in algebra.truth[name]
            )
            grouped.setdefault(signature, set()).add(index)
        if len(grouped) > max_regions:
            raise BudgetUltra("predicate region limit")
        self.signatures = tuple(sorted(grouped, key=lambda item: tuple(sorted(item))))
        self.cells = tuple(frozenset(grouped[key]) for key in self.signatures)
        self.universe = frozenset(range(len(self.cells)))
        self.max_predicate_nodes = max_predicate_nodes
        self.guard_cache = {}

    def codes(self, expression):
        truth = self.algebra.truths(expression)
        result = set()
        for index, cell in enumerate(self.cells):
            if cell <= truth:
                result.add(index)
            elif cell & truth:
                raise UnsupportedUltra("predicate splits an unrefined region")
        return frozenset(result)

    def plans(self, limit):
        # Original overlapping guards and global priority residuals are just
        # different SQL covers of exactly the same region alphabet.
        seen = set()
        originals = self.algebra.expressions
        orders = itertools.islice(itertools.permutations(self.labels), limit)
        candidates = [originals]
        candidates.extend(self.algebra.residuals(order) for order in orders)
        result = []
        for guards in candidates:
            guards = {
                name: self.algebra.simplify_guard(expr) for name, expr in guards.items()
            }
            key = tuple(self.codes(guards[name]) for name in self.labels)
            if key not in seen:
                seen.add(key)
                result.append(guards)
            for name, code in zip(self.labels, key):
                old = self.guard_cache.get(code)
                if old is None or len(guards[name].to_string()) < len(old.to_string()):
                    self.guard_cache[code] = guards[name]
        return result

    def guard(self, wanted):
        wanted = frozenset(wanted)
        if not wanted:
            raise UnsupportedUltra("empty character class")
        if wanted in self.guard_cache:
            return copy.deepcopy(self.guard_cache[wanted])
        positives = {
            name: self.codes(expr) for name, expr in self.algebra.expressions.items()
        }
        # Minimize Boolean cubes over TRUE membership. Negative literals mean
        # NOT TRUE (including UNKNOWN), never unchecked SQL NOT.
        cubes = []
        covered = set()
        for region in sorted(wanted):
            if region in covered:
                continue
            literals = [(name, name in self.signatures[region]) for name in self.labels]

            def matches(items):
                result = self.universe
                for name, positive in items:
                    result &= (
                        positives[name] if positive else self.universe - positives[name]
                    )
                return result

            # Prefer retaining positive guards; their FALSE/UNKNOWN rows are
            # safely rejected by DEFINE, yielding compact SQL complements.
            for literal in sorted(
                literals, key=lambda item: (item[1], self.labels.index(item[0]))
            ):
                trial = [item for item in literals if item != literal]
                if matches(trial) <= wanted:
                    literals = trial
            covered.update(matches(literals))
            positive_parts = [
                self.algebra.expressions[name] for name, flag in literals if flag
            ]
            base = conjunction(positive_parts)
            base_truth = self.algebra.truths(base)
            negative_parts = []
            for name, flag in literals:
                if flag:
                    continue
                residual = conjunction(
                    [
                        part
                        for part in conjuncts(self.algebra.expressions[name])
                        if not base_truth <= self.algebra.truths(part)
                    ]
                )
                negative_parts.append(not_true(residual, self.algebra))
            cubes.append(conjunction([*positive_parts, *negative_parts]))
        result = cubes[0]
        for cube in cubes[1:]:
            result = BinaryExpr(result, "OR", cube)
        if sum(1 for _ in expr_walk(result)) > self.max_predicate_nodes:
            raise BudgetUltra("decoded predicate size limit")
        if self.codes(result) != wanted:
            raise UnsupportedUltra("region predicate reconstruction failed")
        result = self.algebra.simplify_guard(result)
        self.guard_cache[wanted] = result
        return copy.deepcopy(result)

    def encoding(self, coloured=False):
        tokens, mapping = {}, {}
        for name in self.labels:
            chars = []
            # Stable, generic ordering: less-overlapping regions first.
            # Greenery's state elimination is sensitive to alphabet order.
            for index in sorted(
                range(len(self.signatures)), key=lambda i: (len(self.signatures[i]), i)
            ):
                signature = self.signatures[index]
                if name not in signature:
                    continue
                key = (index, name if coloured else None)
                if key not in tokens:
                    tokens[key] = chr(0xE000 + len(tokens))
                chars.append(tokens[key])
            # Greenery's [] is the empty language, including false DEFINEs.
            mapping[name] = "[" + "".join(chars) + "]"
        mapping.update({"@start": chr(0xD000), "@end": chr(0xD001)})
        reverse = {char: pair for pair, char in tokens.items()}
        reverse.update({chr(0xD000): ("start", None), chr(0xD001): ("end", None)})
        return mapping, reverse

    def decode(self, tree, tokens, plan, *, generator, encoding):
        guards = copy.deepcopy(plan)
        aliases, reused = {}, {}
        guard_codes = {name: self.codes(expr) for name, expr in guards.items()}
        original_codes = {
            name: self.codes(expr) for name, expr in self.algebra.expressions.items()
        }

        def charset(chars):
            relation = {tokens[char] for char in chars}
            if not relation:
                raise UnsupportedUltra("empty candidate language")
            assertions = {region for region, _ in relation if isinstance(region, str)}
            if assertions:
                if len(relation) != 1:
                    raise UnsupportedUltra("mixed row/assertion character class")
                kind = next(iter(assertions))
                return AnchorPattern(
                    EmptyPattern(),
                    AnchorType.START if kind == "start" else AnchorType.END,
                )
            wanted = frozenset(region for region, _ in relation)

            def allowed(label, codes):
                return codes and all(
                    (region, None) in relation or (region, label) in relation
                    for region in codes
                )

            eligible = [
                (name, code)
                for name, code in guard_codes.items()
                if code <= wanted and allowed(name, code)
            ]
            # Exact reusable guards avoid gratuitous variable renaming.
            exact = [name for name, code in eligible if code == wanted]
            if exact:
                return VariablePattern(exact[0])
            # A global residual plan can empty a label that is still needed
            # at another pattern position. Empty guards have no decoded uses,
            # so restoring an exact original predicate cannot rebind a use.
            # The complete candidate still goes through observable proof.
            for name in self.labels:
                if (
                    not guard_codes[name]
                    and original_codes[name] == wanted
                    and allowed(name, wanted)
                ):
                    guards[name] = self.algebra.simplify_guard(
                        self.algebra.expressions[name]
                    )
                    guard_codes[name] = wanted
                    return VariablePattern(name)
            remaining, selected = set(wanted), []
            for name, code in sorted(
                eligible, key=lambda item: (-len(item[1]), self.labels.index(item[0]))
            ):
                if remaining & code:
                    selected.append(VariablePattern(name))
                    remaining.difference_update(code)
            groups = {}
            for region in sorted(remaining):
                origins = [
                    label
                    for label in self.labels
                    if label in self.signatures[region] and allowed(label, {region})
                ]
                origins = [
                    label
                    for label in origins
                    if not self.observer.classifier
                    and label not in self.observer.required_variables
                ]
                if not origins:
                    raise UnsupportedUltra(
                        "character class requires renaming an observed primary label"
                    )
                origin = origins[0]
                groups.setdefault(self.observer.token(origin), (origin, set()))[1].add(
                    region
                )
            for token, (origin, codes) in groups.items():
                key = (frozenset(codes), token)
                if key not in reused:
                    number = 1
                    while (
                        f"ULTRA{number}" in guards
                        or f"ULTRA{number}" in self.algebra.subsets
                    ):
                        number += 1
                    label = f"ULTRA{number}"
                    guards[label] = self.guard(codes)
                    aliases[label] = origin
                    reused[key] = label
                selected.append(VariablePattern(reused[key]))
            return alternative(selected)

        def visit(node):
            kind = node[0]
            if kind == "set":
                return charset(node[1])
            if kind == "empty":
                return EmptyPattern()
            if kind == "repeat":
                return QuantifiedPattern(visit(node[1]), *node[2:])
            if kind == "sequence":
                return sequence([visit(child) for child in node[1]])
            if kind == "alternate":
                return alternative([visit(child) for child in node[1]])
            raise UnsupportedUltra("invalid region candidate")

        pattern = visit(tree)
        used = set(names(pattern))
        return CandidateUltra(
            pattern,
            {name: expr for name, expr in guards.items() if name in used},
            {name: origin for name, origin in aliases.items() if name in used},
            generator,
            encoding,
        )

    def reuse_original_names(self, candidate):
        """Reuse unused originals only after the complete pattern is assembled.

        Rebinding a label already used elsewhere could change its DEFINE or
        observed group. Both the TRUE set and observed membership must agree.
        """
        used = set(names(candidate.pattern))
        for alias, origin in list(candidate.aliases.items()):
            wanted = self.codes(candidate.guards[alias])
            for name in self.labels:
                if (
                    name in used
                    or self.observer.token(name) != self.observer.token(origin)
                    or self.codes(self.algebra.expressions[name]) != wanted
                ):
                    continue
                for node in walk(candidate.pattern):
                    if isinstance(node, VariablePattern) and node.name == alias:
                        node.name = name
                candidate.guards[name] = self.algebra.simplify_guard(
                    self.algebra.expressions[name]
                )
                del candidate.guards[alias], candidate.aliases[alias]
                used.remove(alias)
                used.add(name)
                break
        candidate.guards = {
            name: candidate.guards[name]
            for name in (*self.labels, *candidate.aliases)
            if name in candidate.guards
        }


# Alternative DFA state-elimination orders
#
# Bounded alternative state-elimination orders for a region DFA.
#
# This generator is independent of regex shapes: every choice is an automaton
# state order. Greenery's fixed elimination order can produce avoidable nested
# loops, so a few alternatives are useful even when the language is identical.
# The resulting regexes are untrusted candidates, never rewrite certificates.


def eliminate(fsm, order):
    from greenery.rxelems import Pattern, Conc, Mult, NULLPATTERN, EMPTYSTRING
    from greenery.multiplier import ONE, STAR

    end = object()
    active = list(order)
    edges = {}
    for state in active:
        if state in fsm.finals:
            edges[state, end] = Pattern(EMPTYSTRING)
        for charset, target in fsm.map.get(state, {}).items():
            key = state, target
            edges[key] = Pattern(
                *edges.get(key, NULLPATTERN).concs, Conc(Mult(charset, ONE))
            ).reduce()

    for state in reversed(active[1:]):
        loop = Mult(edges.get((state, state), NULLPATTERN), STAR)
        incoming = [
            (source, value)
            for (source, target), value in edges.items()
            if target == state and source != state
        ]
        outgoing = [
            (target, value)
            for (source, target), value in edges.items()
            if source == state and target != state
        ]
        for source, before in incoming:
            for target, after in outgoing:
                key = source, target
                value = Pattern(
                    *edges.get(key, NULLPATTERN).concs,
                    Conc(Mult(before, ONE), loop, Mult(after, ONE)),
                ).reduce()
                if len(str(value)) > 8192:
                    return None
                edges[key] = value
        edges = {key: value for key, value in edges.items() if state not in key}
    initial = fsm.initial
    return Pattern(
        Conc(
            Mult(edges.get((initial, initial), NULLPATTERN), STAR),
            Mult(edges.get((initial, end), NULLPATTERN), ONE),
        )
    ).reduce()


def alternative_eliminations(fsm, *, limit=8):
    # Remove states that cannot reach acceptance before elimination.
    useful = set(fsm.finals)
    while True:
        updated = useful | {
            source
            for source, edges in fsm.map.items()
            if any(target in useful for target in edges.values())
        }
        if updated == useful:
            break
        useful = updated
    others = sorted(useful - {fsm.initial})
    if len(others) > 8 or fsm.initial not in useful:
        return
    seen = set()
    orders = itertools.chain(
        [tuple(others), tuple(reversed(others))], itertools.permutations(others)
    )
    for order in orders:
        if order in seen:
            continue
        seen.add(order)
        result = eliminate(fsm, [fsm.initial, *order])
        if result is not None:
            yield result
        if len(seen) >= limit:
            break


# Candidate generation and ordering search
#
# Candidate generators over a shared, reversible predicate-region alphabet.
#
# Greenery is an untrusted language reducer. Generic structural pruning covers
# preferred-match reductions that need not preserve the ordinary language. Both
# return the same CandidateUltra records; no generator can authorize a rewrite.


def sequence(parts):
    flat = []
    for part in parts:
        if isinstance(part, ConcatenationPattern):
            flat.extend(part.patterns)
        elif not isinstance(part, EmptyPattern):
            flat.append(part)
    # Assertion characters decode to zero-width nodes. Attach these to a
    # neighbouring factor to retain the SQL AST's wrapper representation.
    index = 0
    while index < len(flat):
        part = flat[index]
        if isinstance(part, AnchorPattern) and isinstance(part.pattern, EmptyPattern):
            if part.anchor_type == AnchorType.START and index + 1 < len(flat):
                flat[index : index + 2] = [
                    AnchorPattern(flat[index + 1], part.anchor_type)
                ]
                continue
            if part.anchor_type == AnchorType.END and index > 0:
                flat[index - 1 : index + 1] = [
                    AnchorPattern(flat[index - 1], part.anchor_type)
                ]
                index -= 1
                continue
        index += 1
    return (
        EmptyPattern()
        if not flat
        else flat[0] if len(flat) == 1 else ConcatenationPattern(flat)
    )


def alternative(parts):
    return parts[0] if len(parts) == 1 else AlternationPattern(parts)


def encode(node, mapping):
    if isinstance(node, EmptyPattern):
        return ""
    if isinstance(node, VariablePattern):
        return mapping[node.name]
    if isinstance(node, AnchorPattern):
        body = encode(node.pattern, mapping)
        return (
            mapping["@start"] + body
            if node.anchor_type == AnchorType.START
            else body + mapping["@end"]
        )
    if isinstance(node, ConcatenationPattern):
        return "".join(encode(child, mapping) for child in node.patterns)
    if isinstance(node, AlternationPattern):
        return "(" + "|".join(encode(child, mapping) for child in node.patterns) + ")"
    if isinstance(node, QuantifiedPattern):
        hi = "" if node.atMost is None else str(node.atMost)
        return (
            "("
            + encode(node.pattern, mapping)
            + "){"
            + str(node.atLeast)
            + ","
            + hi
            + "}"
        )
    raise UnsupportedUltra(type(node).__name__)


def region_tree(node, mapping):
    if isinstance(node, EmptyPattern):
        return ["empty"]
    if isinstance(node, VariablePattern):
        return ["set", list(mapping[node.name][1:-1])]
    if isinstance(node, AnchorPattern):
        assertion = [
            "set",
            [mapping["@start" if node.anchor_type == AnchorType.START else "@end"]],
        ]
        body = region_tree(node.pattern, mapping)
        return [
            "sequence",
            (
                [assertion, body]
                if node.anchor_type == AnchorType.START
                else [body, assertion]
            ),
        ]
    if isinstance(node, QuantifiedPattern):
        return [
            "repeat",
            region_tree(node.pattern, mapping),
            node.atLeast,
            node.atMost,
            node.isGreedy,
        ]
    if not isinstance(node, (ConcatenationPattern, AlternationPattern)):
        raise UnsupportedUltra(type(node).__name__)
    return [
        "sequence" if isinstance(node, ConcatenationPattern) else "alternate",
        [region_tree(child, mapping) for child in children(node)],
    ]


def cost(node):
    """Generic structural cost, including nesting and reluctant choices."""

    def depth(part, level=0):
        extra = isinstance(part, QuantifiedPattern)
        return (level if extra else 0) + sum(
            depth(child, level + extra) for child in children(part)
        )

    size = sum(1 for _ in walk(node))
    reluctant = sum(
        isinstance(part, QuantifiedPattern) and not part.isGreedy for part in walk(node)
    )
    return size + 2 * depth(node), size, reluctant, len(node.to_string())


def convert_greenery(element):
    from greenery.rxelems import Pattern, Charclass

    def convert(element):
        if isinstance(element, Charclass):
            if element.negated:
                raise UnsupportedUltra("negated greenery alphabet")
            return [
                "set",
                [
                    chr(code)
                    for lo, hi in element.ord_ranges
                    for code in range(lo, hi + 1)
                ],
            ]
        if isinstance(element, Pattern):
            if not element.concs:
                raise UnsupportedUltra("empty greenery language")
            options = [
                compact_tree(["sequence", [convert(mult) for mult in conc.mults]])
                for conc in sorted(element.concs, key=str)
            ]
            return options[0] if len(options) == 1 else ["alternate", options]
        body = convert(element.multiplicand)
        lo, hi = element.multiplier.min.v, element.multiplier.max.v
        if hi == 0:
            return ["empty"]
        return body if lo == hi == 1 else ["repeat", body, lo, hi, True]

    return convert(element)


def compact_tree(tree):
    if tree[0] in {"sequence", "alternate"}:
        parts = []
        for child in tree[1]:
            child = compact_tree(child)
            if child[0] == tree[0]:
                parts.extend(child[1])
            elif tree[0] != "sequence" or child[0] != "empty":
                parts.append(child)
        return (
            ["empty"]
            if not parts
            else parts[0] if len(parts) == 1 else [tree[0], parts]
        )
    if tree[0] == "repeat":
        return ["repeat", compact_tree(tree[1]), *tree[2:]]
    return tree


def tree_source(tree):
    kind = tree[0]
    if kind == "set":
        return "[" + "".join(tree[1]) + "]"
    if kind == "empty":
        return ""
    if kind == "repeat":
        return (
            "("
            + tree_source(tree[1])
            + "){"
            + str(tree[2])
            + ","
            + ("" if tree[3] is None else str(tree[3]))
            + "}"
        )
    return (
        "(" + ("|" if kind == "alternate" else "").join(map(tree_source, tree[1])) + ")"
    )


def tree_cost(tree, level=0):
    if tree[0] in {"set", "empty"}:
        return 1
    if tree[0] == "repeat":
        return 1 + 2 * level + tree_cost(tree[1], level + 1)
    return 1 + sum(tree_cost(child, level) for child in tree[1])


def reduce_region_windows(tree, *, deadline=float("inf")):
    """Iterative DFA reduction of all subtrees and concatenation windows.

    State elimination can expose algebraic reductions across sibling nodes.
    Searching every bounded window avoids a catalogue of those identities.
    """
    from greenery import parse
    from greenery.rxelems import from_fsm

    cache = {}

    def reduce(part):
        if monotonic() > deadline:
            return part
        source = tree_source(part)
        if source not in cache:
            cache[source] = compact_tree(
                convert_greenery(from_fsm(parse(source).to_fsm().reduce()))
            )
        reduced = cache[source]
        return reduced if tree_cost(reduced) < tree_cost(part) else part

    def visit(part):
        kind = part[0]
        if kind == "repeat":
            part = [kind, visit(part[1]), *part[2:]]
        elif kind in {"sequence", "alternate"}:
            part = compact_tree([kind, [visit(child) for child in part[1]]])
        if part[0] == "sequence":
            parts = part[1]
            for width in range(min(8, len(parts)), 1, -1):
                for start in range(len(parts) - width + 1):
                    window = ["sequence", parts[start : start + width]]
                    reduced = reduce(window)
                    if tree_cost(reduced) < tree_cost(window):
                        return compact_tree(
                            [
                                "sequence",
                                [*parts[:start], reduced, *parts[start + width :]],
                            ]
                        )
        return reduce(part)

    for _ in range(4):
        if monotonic() > deadline:
            break
        following = visit(tree)
        # Local greenery algebra also folds equal neighbouring character sets.
        following = compact_tree(
            convert_greenery(parse(tree_source(following)).reduce())
        )
        if tree_cost(following) >= tree_cost(tree):
            break
        tree = following
    return tree


def worker_reduce(source):
    from greenery import parse
    from greenery.rxelems import from_fsm

    fsm = parse(source).to_fsm().reduce()
    first = compact_tree(convert_greenery(from_fsm(fsm)))
    yield "greenery", first
    yield "greenery_windows", reduce_region_windows(first)
    for expression in alternative_eliminations(fsm):
        tree = compact_tree(convert_greenery(expression))
        yield "state_elimination", tree
        yield "greenery_windows", reduce_region_windows(tree)


def paths(node, path=()):
    yield path, node
    for index, child in enumerate(children(node)):
        yield from paths(child, path + (index,))


def replace(node, path, replacement):
    if not path:
        return copy.deepcopy(replacement)
    node = copy.deepcopy(node)
    if isinstance(node, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
        node.pattern = replace(node.pattern, path[1:], replacement)
    else:
        node.patterns[path[0]] = replace(node.patterns[path[0]], path[1:], replacement)
        if isinstance(node, ConcatenationPattern):
            node = sequence(node.patterns)
    return node


def locations(pattern, limit):
    """All subtrees and bounded consecutive siblings, without shape filters."""
    visited = list(paths(pattern))
    result = [(((), None, None), pattern)]
    for path, node in visited:
        if isinstance(node, ConcatenationPattern):
            for width in range(min(len(node.patterns) - 1, 8), 1, -1):
                for start in range(len(node.patterns) - width + 1):
                    result.append(
                        (
                            (path, start, start + width),
                            sequence(node.patterns[start : start + width]),
                        )
                    )
    result.extend(((path, None, None), node) for path, node in visited if path)
    return result[:limit]


def replace_location(pattern, location, replacement):
    path, first, last = location
    if first is None:
        return replace(pattern, path, replacement)
    parent = pattern
    for index in path:
        parent = children(parent)[index]
    return replace(
        pattern,
        path,
        sequence([*parent.patterns[:first], replacement, *parent.patterns[last:]]),
    )


def observable_proposals(pattern, observer, *, limit=32):
    """Uniform AST pruning, without identities for particular regex shapes.

    Dropping a branch or fixing a repetition is intentionally untrusted. This
    also finds preferred-match reductions that a language minimizer cannot.
    """
    candidates, seen = [], set()
    for path, node in paths(pattern):
        replacements = []
        if isinstance(node, AlternationPattern):
            replacements.extend(node.patterns)
        if isinstance(node, QuantifiedPattern):
            replacements.append(
                EmptyPattern()
                if node.atLeast == 0
                else (
                    node.pattern
                    if node.atLeast == 1
                    else QuantifiedPattern(
                        node.pattern, node.atLeast, node.atLeast, True
                    )
                )
            )
        for replacement in replacements:
            candidate = replace(pattern, path, replacement)
            key = candidate.to_string()
            if key != pattern.to_string() and key not in seen:
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def proposals(
    pattern,
    *,
    regions,
    plans,
    deadline,
    worker_seconds=2.0,
    limit=24,
    extra=(),
    diagnostics=None,
):
    """Yield sorted basic candidates before paying for expanded generation.

    The caller may stop after any certified improvement. Each worker exits
    before yielding, so early success never leaves a subprocess behind. Both
    stages share the worker time allowance and the caller's overall deadline.
    """
    jobs, payload = [], []
    encodings = [
        ("regions", *regions.encoding(False)),
        ("coloured_regions", *regions.encoding(True)),
    ]
    for encoding, mapping, tokens in encodings:
        for location, node in locations(pattern, limit):
            try:
                source = encode(node, mapping)
            except UnsupportedUltra:
                continue
            jobs.append((location, tokens, encoding))
            payload.append(source)
    payload = json.dumps(payload)
    candidates, seen = [], set()

    def add(tree, location, tokens, encoding, generator):
        for plan in plans:
            if monotonic() > deadline:
                raise BudgetUltra("candidate decoding timeout")
            try:
                candidate = regions.decode(
                    tree, tokens, plan, generator=generator, encoding=encoding
                )
                candidate.pattern = replace_location(
                    pattern, location, candidate.pattern
                )
                for name in names(candidate.pattern):
                    if name not in candidate.guards:
                        candidate.guards[name] = plan[name]
                regions.reuse_original_names(candidate)
                key = (
                    candidate.pattern.to_string(),
                    tuple(
                        sorted(
                            (name, expr.to_string())
                            for name, expr in candidate.guards.items()
                        )
                    ),
                    tuple(sorted(candidate.aliases.items())),
                )
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
            except UnsupportedUltra:
                continue

    for phase in ("basic", "expanded"):
        output, elapsed, cache_hit = _worker_output(
            payload, phase=phase, deadline=deadline, worker_seconds=worker_seconds
        )
        worker_seconds -= elapsed
        if diagnostics is not None:
            diagnostics.setdefault("candidate_stages", []).append(phase)
            diagnostics["worker_cache_hits"] = diagnostics.get(
                "worker_cache_hits", 0
            ) + int(cache_hit)
            diagnostics["worker_elapsed_seconds"] = (
                diagnostics.get("worker_elapsed_seconds", 0.0) + elapsed
            )
        candidates.clear()
        for line in output.splitlines():
            try:
                index, generator, tree = json.loads(line)
            except ValueError:
                break
            location, tokens, encoding = jobs[index]
            add(tree, location, tokens, encoding, generator)
        if phase == "basic":
            # Unchanged syntax permits guard improvements; pruning uses the
            # same decoder and certificate, without expensive window search.
            for encoding, mapping, tokens in encodings:
                try:
                    tree = region_tree(pattern, mapping)
                except UnsupportedUltra:
                    tree = None
                if tree is not None:
                    add(tree, ((), None, None), tokens, encoding, "identity")
                for candidate in extra:
                    try:
                        tree = region_tree(candidate, mapping)
                    except UnsupportedUltra:
                        continue
                    add(tree, ((), None, None), tokens, encoding, "structural_pruning")
        if diagnostics is not None:
            diagnostics["candidates"] = diagnostics.get("candidates", 0) + len(
                candidates
            )
        yield from sorted(
            candidates,
            key=lambda item: (
                cost(item.pattern),
                bool(item.aliases),
                sum(len(expr.to_string()) for expr in item.guards.values()),
                item.generator != "greenery",
                item.pattern.to_string(),
            ),
        )


def arrangements(pattern, *, limit=96):
    """Joint branch permutations and quantifier priorities, with a hard bound."""
    sites, choices = [], []
    for path, part in paths(pattern):
        if isinstance(part, AlternationPattern) and len(part.patterns) > 1:
            sites.append((path, "alternate"))
            choices.append(
                list(
                    itertools.islice(
                        itertools.permutations(range(len(part.patterns))), limit
                    )
                )
            )
        elif isinstance(part, QuantifiedPattern) and part.atLeast != part.atMost:
            sites.append((path, "repeat"))
            choices.append([part.isGreedy, not part.isGreedy])
    for combination in itertools.islice(itertools.product(*choices), limit):
        candidate = copy.deepcopy(pattern)
        for (path, kind), choice in reversed(list(zip(sites, combination))):
            part = candidate
            for index in path:
                part = children(part)[index]
            if kind == "repeat":
                part.isGreedy = choice
            else:
                part.patterns = [part.patterns[index] for index in choice]
        yield candidate


def main(phase="all"):
    from greenery import parse
    from greenery.rxelems import from_fsm

    sources, cache, positions = json.load(sys.stdin), {}, {}
    # Finish one inexpensive pass across all jobs before expanding the search
    # for any job. A hard case cannot starve every following subtree/colour.
    for index, source in enumerate(sources):
        positions.setdefault(source, []).append(index)
        try:
            if source not in cache:
                fsm = parse(source).to_fsm().reduce()
                tree = compact_tree(convert_greenery(from_fsm(fsm)))
                cache[source] = fsm, tree
            if phase != "expanded":
                print(json.dumps([index, "greenery", cache[source][1]]), flush=True)
        except (ValueError, KeyError, OverflowError, RecursionError):
            continue
    if phase == "basic":
        return
    for source, (fsm, tree) in sorted(
        cache.items(), key=lambda item: (len(item[1][0].states), -tree_cost(item[1][1]))
    ):
        budget = monotonic() + 0.8
        seen = {json.dumps(tree)}

        def emit(generator, candidate):
            key = json.dumps(candidate)
            if key not in seen:
                seen.add(key)
                for index in positions[source]:
                    print(json.dumps([index, generator, candidate]), flush=True)

        # Alternative state orders address DFA-to-regex expression growth.
        variants = [tree]
        if tree_cost(tree) > tree_cost(compact_tree(convert_greenery(parse(source)))):
            for expression in alternative_eliminations(fsm, limit=3):
                if monotonic() > budget:
                    break
                candidate = compact_tree(convert_greenery(expression))
                emit("state_elimination", candidate)
                variants.append(candidate)
        for candidate in sorted(variants, key=tree_cost):
            if monotonic() > budget:
                break
            emit("greenery_windows", reduce_region_windows(candidate, deadline=budget))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
