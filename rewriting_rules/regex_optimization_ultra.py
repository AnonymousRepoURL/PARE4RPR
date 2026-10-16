"""Predicate regions -> regex candidates -> SQL decoding -> observable proof.

All generators share the same region model, decoder, ordering search and
whole-pattern certificate. Greenery preserves ordinary language only; neither
it nor structural pruning can authorize a rewrite. The former Plus snapshot
is not a runtime fallback. Existing SQL AST and the original rules are intact.

Inspired by the Boolean guard/partition principles of D'Antoni and Veanes,
POPL 2014; this is not their MinN_SFA minimization algorithm. Priority, output
membership and frozen-prefix SQL values require additional certificates here.

Register explicitly with RewriteEngine([RegexOptimizationUltraRule()]). A
failed or exhausted proof leaves semantic AST fields unchanged. last_diagnostics
reports the generator, encoding, proof scope and strictly decreasing score.

Note that this rule do not merge (A | B) to A, but it will try to ensure A ∩ B = ∅
"""

import copy
from dataclasses import dataclass
import itertools
from time import monotonic, perf_counter

from expression_clause import (
    DependentCondition,
    IndependentCondition,
    collect_define_dependency_reasons,
)

from patternnode import collect_pattern_rewrite_boundaries, ConcatenationPattern
from .base import RewriteRule
from .predicates_ultra import (
    PredicateAlgebraUltra,
    TRUE,
    expr_walk,
    normalize,
    observations,
    references,
)
from .regex_automata_ultra import (
    BudgetUltra,
    ProgramUltra,
    UnsupportedUltra,
    ambiguity_profile,
    names,
    verify_labels,
    walk,
)
from .regex_candidates_ultra import (
    RegionsUltra,
    arrangements,
    cost,
    observable_proposals,
    proposals,
    sequence,
)


@dataclass(frozen=True)
class UltraConfig:
    timeout_seconds: float = 8.0
    greenery_timeout_seconds: float = 3.0
    max_variables: int = 6
    max_pattern_nodes: int = 64
    max_predicate_nodes: int = 256
    max_predicate_cells: int = 10000
    max_regions: int = 128
    max_nfa_states: int = 256
    max_product_states: int = 20000
    max_complexity_states: int = 2000
    max_guard_orders: int = 24
    max_arrangements: int = 96
    max_proposals: int = 24
    max_candidate_checks: int = 2048
    # Accepted for callers of the first Ultra prototype. There is no longer
    # a compatibility path, regardless of this deprecated argument's value.
    enable_plus_compatibility: bool = False


class RegexOptimizationUltraRule(RewriteRule):
    def __init__(self, config=None):
        self.config = config or UltraConfig()
        self.last_rewrite_elapsed_seconds = 0.0
        self.last_diagnostics = {}

    @property
    def name(self):
        return "regex_optimization_ultra"

    def is_applicable(self, ast):
        return (
            ast.pattern is not None
            and ast.define is not None
            and getattr(ast, "raw_sql_override", None) is None
        )

    def apply(self, ast):
        started = perf_counter()
        try:
            return self._apply_symbolic(ast)
        finally:
            self.last_rewrite_elapsed_seconds = perf_counter() - started
            ast.regex_optimization_ultra_elapsed_seconds = (
                self.last_rewrite_elapsed_seconds
            )

    def _apply_symbolic(self, ast):
        self.last_diagnostics = {
            "status": "unchanged",
            "path": "regions",
            "candidate_checks": 0,
        }
        try:
            if not self.is_applicable(ast):
                return False
            if any(
                boundary.kind == "permute"
                for boundary in getattr(ast, "pattern_rewrite_boundaries", ())
            ):
                raise UnsupportedUltra("preserved rewrite boundaries")
            config = self.config
            deadline = monotonic() + config.timeout_seconds
            ordered_names = names(ast.pattern)
            if (
                len(ordered_names) > config.max_variables
                or sum(1 for _ in walk(ast.pattern)) > config.max_pattern_nodes
            ):
                raise BudgetUltra("input size limit")
            ProgramUltra(ast.pattern, config.max_nfa_states)
            expressions = {}
            correlations = (
                ast._input_correlation_names()
                - set(ordered_names)
                - set(ast.subsets or {})
            )
            for name in ordered_names:
                condition = ast.define.definitions.get(name)
                expression = condition.expression if condition is not None else TRUE()
                if (
                    sum(
                        1
                        for _ in itertools.islice(
                            expr_walk(expression), config.max_predicate_nodes + 1
                        )
                    )
                    > config.max_predicate_nodes
                ):
                    raise BudgetUltra("predicate syntax limit")
                expressions[name] = normalize(expression, name, correlations)
            algebra = PredicateAlgebraUltra(
                ast,
                expressions,
                deadline=deadline,
                max_cells=config.max_predicate_cells,
                allow_context=True,
            )
            observer = observations(ast, expressions)
            regions = RegionsUltra(
                algebra,
                observer,
                max_regions=config.max_regions,
                max_predicate_nodes=config.max_predicate_nodes,
            )
            plans = regions.plans(config.max_guard_orders)
            extra = observable_proposals(
                ast.pattern, observer, limit=config.max_proposals
            )
            self.last_diagnostics.update(
                predicate_regions=len(regions.cells), candidates=0
            )
            candidates = proposals(
                ast.pattern,
                regions=regions,
                plans=plans,
                deadline=deadline,
                worker_seconds=config.greenery_timeout_seconds,
                limit=config.max_proposals,
                extra=extra,
                diagnostics=self.last_diagnostics,
            )
            baseline_alphabet = algebra.alphabet(expressions)
            original_score = self._score(
                ast.pattern,
                expressions,
                [pair[0] for pair in baseline_alphabet],
                deadline,
            )
            checked = set()
            for candidate in candidates:
                candidate_names = set(names(candidate.pattern))
                # DEFINE needs at least one variable definition. A candidate
                # with no variables cannot carry one; retain the source query
                # unless another candidate preserves a definable variable.
                if not candidate_names:
                    continue
                # Enforce the candidate contract here as well as in the
                # decoder. A future/custom generator must not manufacture
                # CLASSIFIER identity or omit a guard (SQL defaults it TRUE).
                if set(candidate.guards) != candidate_names:
                    continue
                if any(
                    label in expressions
                    or label not in candidate_names
                    or origin not in expressions
                    or observer.classifier
                    or origin in observer.required_variables
                    for label, origin in candidate.aliases.items()
                ):
                    continue
                if not observer.required_variables <= candidate_names:
                    continue
                subsets = {
                    name: [
                        label
                        for label in names(candidate.pattern)
                        if candidate.aliases.get(label, label) in members
                    ]
                    for name, members in (ast.subsets or {}).items()
                }
                if any(not members for members in subsets.values()):
                    continue
                guards = candidate.guards
                try:
                    alphabet = algebra.alphabet(guards)
                except UnsupportedUltra:
                    continue
                observe = lambda label: observer.token(
                    candidate.aliases.get(label, label)
                )
                for arranged in arrangements(
                    candidate.pattern, limit=config.max_arrangements
                ):
                    if monotonic() > deadline:
                        raise BudgetUltra("search timeout")
                    key = (
                        arranged.to_string(),
                        alphabet,
                        tuple(sorted(candidate.aliases.items())),
                    )
                    if key in checked:
                        continue
                    checked.add(key)
                    try:
                        bindings = algebra.bindings(arranged, guards)
                        if any(
                            algebra.present.get(ref) != value
                            for ref, value in bindings.items()
                        ):
                            continue
                    except UnsupportedUltra:
                        continue
                    try:
                        left, right, scope = self._proof_patterns(
                            ast.pattern, arranged, guards, algebra
                        )
                        score = self._score(
                            arranged, guards, [pair[1] for pair in alphabet], deadline
                        )
                        if score >= original_score:
                            continue
                        self.last_diagnostics["candidate_checks"] += 1
                        if (
                            self.last_diagnostics["candidate_checks"]
                            > config.max_candidate_checks
                        ):
                            raise BudgetUltra("candidate check limit")
                        proof = verify_labels(
                            left,
                            right,
                            alphabet,
                            deadline=deadline,
                            max_states=config.max_product_states,
                            max_nfa=config.max_nfa_states,
                            observe=observe,
                        )
                        if not proof.equivalent:
                            self.last_diagnostics["last_counterexample"] = proof.witness
                            self.last_diagnostics[
                                "counterexample_at_partition_start"
                            ] = proof.at_partition_start
                            continue
                    except UnsupportedUltra:
                        continue
                    rewritten = copy.deepcopy(ast)
                    rewritten.pattern = arranged
                    rewritten.pattern_rewrite_boundaries = (
                        collect_pattern_rewrite_boundaries(arranged)
                    )
                    definitions = {}
                    for name, expression in guards.items():
                        if (
                            name in expressions
                            and name in ast.define.definitions
                            and expression.to_string() == expressions[name].to_string()
                        ):
                            definitions[name] = copy.deepcopy(
                                ast.define.definitions[name]
                            )
                            continue
                        # Materialize implicit TRUE guards as `name AS 1 = 1`,
                        # including when all explicitly defined labels vanished.
                        reasons = collect_define_dependency_reasons(name, expression)
                        definitions[name] = (
                            DependentCondition(name, copy.deepcopy(expression), reasons)
                            if reasons
                            else IndependentCondition(name, copy.deepcopy(expression))
                        )
                    rewritten.define.definitions = definitions
                    rewritten.subsets = subsets if ast.subsets is not None else None
                    if rewritten.validate():
                        continue
                    ast.pattern, ast.define, ast.subsets = (
                        rewritten.pattern,
                        rewritten.define,
                        rewritten.subsets,
                    )
                    ast.pattern_rewrite_boundaries = (
                        rewritten.pattern_rewrite_boundaries
                    )
                    self.last_diagnostics.update(
                        status="rewritten",
                        generator=candidate.generator,
                        encoding=candidate.encoding,
                        proof_states=proof.states,
                        proof_scope=scope,
                        alphabet_cells=len(alphabet),
                        generated_variables=sorted(candidate.aliases),
                        score_before=original_score,
                        score_after=score,
                        observed_groups=[sorted(group) for group in observer.groups],
                        nonnull_columns=sorted(
                            key
                            for key, value in algebra.nullability.columns.items()
                            if value
                        ),
                    )
                    return True
            return False
        except (UnsupportedUltra, BudgetUltra) as error:
            self.last_diagnostics.update(status="inconclusive", reason=str(error))
            return False

    def _proof_patterns(self, original, candidate, guards, algebra):
        if not algebra.needs_prefix_context:
            return original, candidate, "whole_pattern"
        # With an ambiguous prefix, its aggregate/navigation values differ
        # between backtracking paths. Preserve that entire prefix verbatim
        # and compare the COMPLETE remaining suffix for arbitrary frozen
        # values. For each prefix path the suffix then has the same preferred
        # result/failure, so the original prefix enumeration selects the same
        # path. This is a contextual proof, not a row-global frozen-value guess.
        if not isinstance(original, ConcatenationPattern) or not isinstance(
            candidate, ConcatenationPattern
        ):
            raise UnsupportedUltra(
                "ambiguous prefix requires a shared concatenation context"
            )
        cut = max(len(binding.signature) for binding in algebra.present.values())
        before, after = original.patterns, candidate.patterns
        if len(after) <= cut or [part.to_string() for part in before[:cut]] != [
            part.to_string() for part in after[:cut]
        ]:
            raise UnsupportedUltra("ambiguous prefix context was changed")
        for name in names(sequence(before[:cut])):
            if (
                references(algebra.expressions[name])
                or name not in guards
                or guards[name].to_string() != algebra.expressions[name].to_string()
            ):
                raise UnsupportedUltra(
                    "ambiguous prefix predicates were changed or depend on another prefix"
                )
        # No frozen reference may still acquire members before its dependent
        # consumers inside this suffix (frozen_prefix checks this separately).
        return (
            sequence(before[cut:]),
            sequence(after[cut:]),
            "shared_prefix_complete_suffix",
        )

    def _score(self, pattern, guards, alphabet, deadline):
        cyclic, all_threads = ambiguity_profile(
            pattern,
            alphabet,
            deadline=deadline,
            max_states=self.config.max_complexity_states,
            max_nfa=self.config.max_nfa_states,
        )
        structural = cost(pattern)
        return (
            cyclic,
            structural[0],
            all_threads,
            *structural[1:],
            len(guards),
            sum(len(expr.to_string()) for expr in guards.values()),
        )
