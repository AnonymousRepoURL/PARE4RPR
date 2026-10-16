"""Orchestrate priority-safe optimization of regular row-pattern islands.

The implementation follows five explicit phases:

1. ``regex_fragment_search`` discovers bounded regular islands and records
   where each island lives in the MATCH_RECOGNIZE pattern tree.
2. This module partitions the island's DEFINE predicates into disjoint truth
   regions and encodes those regions as a finite regex alphabet.
3. ``priority_optimizer_service`` runs ``priority_regex_optimizer`` behind a
   timeout and preserves whether a proof is compositional or standalone.
   General candidates may change the accepted language, but must preserve
   success/failure and the preferred endpoint on every abstract input.
4. Standalone fragment candidates are substituted back into the complete
   PATTERN and proved again over one joint predicate alphabet.  SQL clauses
   that can observe label assignments fail closed unless a stronger contextual
   theorem applies.
5. This module decodes the certified regex, updates PATTERN/DEFINE/SUBSET
   metadata, and commits only if all SQL-level invariants still hold.

Anchors, exclusions, and PERMUTE are semantic boundaries.  Prefix-stable
dependent predicates are modeled as constrained symbolic Boolean dimensions;
such rewrites may reuse or remove existing variables but may not synthesize a
new dependent predicate.  The common ``A+ B+`` case uses a fixed linear-size
construction before the bounded general search.  Reluctant repetitions are
greedified only at a mandatory, predicate-disjoint boundary.  The same
boundary also supports the certified ``(X{1,n}?)+ Y -> X+ Y`` normalization,
where repetition grouping is not observable because every consumed row keeps
the X label.  One explicit ``X*? Y X*`` theorem is supported, and every other
reluctant form blocks the general reducer.  UNKNOWN, timeout, and unsupported
output are safe no-ops.

Loris D'Antoni; Margus Veanes Minimization of Symbolic Automata. POPL, 2014.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from expression import (
    BetweenExpr,
    BinaryExpr,
    CaseExpr,
    ColumnRef,
    Expr,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    NullLiteral,
    SemanticsExpr,
    SubqueryExpr,
)
from expression_clause import (
    Condition,
    DEPENDENT_FUNCS,
    DependentCondition,
    IndependentCondition,
)
from expression_simplifier import (
    RangeConstraint,
    RangeDomain,
    _compare_literal_values,
    _condition_is_not_true,
    _extract_range_constraint,
    _simplify_logical_expr,
)
from patternnode import (
    AlternationPattern,
    AnchorPattern,
    ConcatenationPattern,
    EmptyPattern,
    ExcludedPattern,
    PatternNode,
    PatternRewriteBoundary,
    PermutationPattern,
    QuantifiedPattern,
    VariablePattern,
)
from priority_regex_optimizer import (
    Alternation as RegexAlternation,
    Atom as RegexAtom,
    CharSet,
    Concatenation as RegexConcatenation,
    Empty as RegexEmpty,
    EquivalenceMode,
    Node as RegexNode,
    ProofScope,
    Repetition as RegexRepetition,
    Verdict,
    parse_regex,
    to_regex,
    verify_priority_equivalent,
)
from queryanalysis import RPRAbstractSyntaxTree

from .base import RewriteRule
from .priority_optimizer_service import (
    PRIORITY_OPTIMIZER_SERVICE,
    PriorityOptimizerConfig,
    PriorityOptimizerOutcome,
)
from .regex_fragment_search import (
    PatternLocation,
    contains_effective_reluctant_quantifier,
    contains_controlled_adjacent_plus_rewrite,
    has_optimization_opportunity,
    is_adjacent_plus_fragment,
    is_effective_reluctant_quantifier,
    is_reluctant_prefix_fragment,
    iter_regular_fragments,
    needs_multi_pass_optimization,
    regular_node_count,
)
from .utils import (
    _expr_references_any_var,
    _strip_var_qualifier,
    choose_generated_pattern_variable_name,
)

_TRUE_EXPR = lambda: BinaryExpr(Literal(1), "=", Literal(1))
_PRIVATE_TOKEN_START = 0xE000


@dataclass(frozen=True)
class _SimplePattern:
    node: PatternNode
    name: str
    expression: Expr
    minimum: int
    maximum: Optional[int]


@dataclass
class _TruthRegion:
    signature: int
    expression: Expr
    token: str = ""


@dataclass(frozen=True)
class _FinitePredicate:
    column: str
    values: frozenset[object]


@dataclass
class _PriorityRewritePlan:
    location: PatternLocation
    old_names: Tuple[str, ...]
    expressions: Dict[str, Expr]
    regions: Tuple[_TruthRegion, ...]
    variable_tokens: Dict[str, frozenset[int]]
    candidate: RegexNode
    max_generated_variables: Optional[int] = None
    max_generated_predicate_length: Optional[int] = None
    original_conditions: Optional[Dict[str, Optional[Condition]]] = None
    existing_atoms_only: bool = False
    proof_scope: ProofScope = ProofScope.COMPOSITIONAL

    @property
    def region_codes(self) -> Set[int]:
        return {ord(region.token) for region in self.regions}


@dataclass
class _FragmentEncoding:
    """SQL predicates and a sound finite verification alphabet for a fragment."""

    old_names: Tuple[str, ...]
    expressions: Dict[str, Expr]
    regions: Tuple[_TruthRegion, ...]
    variable_tokens: Dict[str, frozenset[int]]
    source: str
    original_conditions: Optional[Dict[str, Optional[Condition]]] = None
    existing_atoms_only: bool = False

    @property
    def region_codes(self) -> Set[int]:
        return {ord(region.token) for region in self.regions}


@dataclass
class _PredicateModel:
    """DEFINE information before it is expanded into truth regions."""

    names: Tuple[str, ...]
    expressions: Dict[str, Expr]
    base_infos: List[_SimplePattern]
    independent_axes: Dict[int, int]
    dependent_axes: List[Tuple[int, int]]
    original_conditions: Dict[str, Optional[Condition]]

    @property
    def has_symbolic_dependencies(self) -> bool:
        return bool(self.dependent_axes)


@dataclass
class _AstRewriteState:
    """The mutable AST fields touched by a rewrite transaction."""

    pattern: Optional[PatternNode]
    definitions: Optional[Dict[str, Optional[Condition]]]
    subsets: Optional[Dict[str, List[str]]]
    boundaries: Tuple[PatternRewriteBoundary, ...]

    @classmethod
    def capture(cls, ast: RPRAbstractSyntaxTree) -> "_AstRewriteState":
        return cls(
            copy.deepcopy(ast.pattern),
            copy.deepcopy(ast.define.definitions if ast.define is not None else None),
            copy.deepcopy(ast.subsets),
            copy.deepcopy(getattr(ast, "pattern_rewrite_boundaries", ())),
        )

    def restore(self, ast: RPRAbstractSyntaxTree) -> None:
        ast.pattern = self.pattern
        if ast.define is not None and self.definitions is not None:
            ast.define.definitions = self.definitions
        ast.subsets = self.subsets
        ast.pattern_rewrite_boundaries = self.boundaries


@dataclass
class _DecodedRewrite:
    """A regex candidate translated back to SQL pattern objects."""

    replacement: PatternNode
    conditions: List[Condition]
    new_names: List[str]


class _PriorityPlanDecoder:
    """Decode one proved regex while consistently assigning SQL variables."""

    def __init__(
        self,
        rule: "RegexOptimizationPlusRule",
        ast: RPRAbstractSyntaxTree,
        plan: _PriorityRewritePlan,
    ) -> None:
        self.rule = rule
        self.ast = ast
        self.plan = plan
        self.allowed_codes = plan.region_codes
        self.used_names: Set[str] = set(ast._input_correlation_names())
        self.name_by_tokens: Dict[frozenset[int], str] = {}
        self.conditions: List[Condition] = []
        self.generated_variable_count = 0
        self.original_name_by_tokens: Dict[frozenset[int], str] = {}
        for name in plan.old_names:
            self.original_name_by_tokens.setdefault(plan.variable_tokens[name], name)

    def decode(self) -> Optional[_DecodedRewrite]:
        replacement = self._convert(self.plan.candidate)
        if replacement is None:
            return None
        return _DecodedRewrite(
            replacement,
            self.conditions,
            list(self.name_by_tokens.values()),
        )

    def _variable_for(self, chars: CharSet) -> Optional[VariablePattern]:
        codes = self.rule._charset_region_codes(chars, self.allowed_codes)
        if codes is None:
            return None
        existing = self.name_by_tokens.get(codes)
        if existing is not None:
            return VariablePattern(existing)

        original_name = self.original_name_by_tokens.get(codes)
        if original_name is not None:
            name = original_name
            if self.plan.original_conditions is not None:
                original_condition = self.plan.original_conditions.get(original_name)
                if original_condition is not None:
                    self.conditions.append(copy.deepcopy(original_condition))
                expression = None
            else:
                expression = self.plan.expressions[original_name]
        else:
            if self.plan.existing_atoms_only:
                return None
            if (
                self.plan.max_generated_variables is not None
                and self.generated_variable_count >= self.plan.max_generated_variables
            ):
                return None
            name = choose_generated_pattern_variable_name(self.ast, self.used_names)
            expression = None
            self.generated_variable_count += 1

        self.used_names.add(name)
        self.name_by_tokens[codes] = name
        if original_name is not None and self.plan.original_conditions is not None:
            return VariablePattern(name)

        if expression is None:
            region_expressions = [
                region.expression
                for region in self.plan.regions
                if ord(region.token) in codes
            ]
            expression = self.rule._or_expressions(region_expressions)
        expression = copy.deepcopy(expression)
        predicate_length_limit = (
            self.plan.max_generated_predicate_length
            if original_name is None
            and self.plan.max_generated_predicate_length is not None
            else self.rule.MAX_GENERATED_PREDICATE_LENGTH
        )
        if len(expression.to_string()) > predicate_length_limit:
            return None
        self.conditions.append(IndependentCondition(name, expression))
        return VariablePattern(name)

    def _convert(self, node: RegexNode) -> Optional[PatternNode]:
        if isinstance(node, RegexEmpty):
            return EmptyPattern()
        if isinstance(node, RegexAtom):
            return self._variable_for(node.chars)
        if isinstance(node, RegexConcatenation):
            children = [self._convert(part) for part in node.parts]
            if any(child is None for child in children):
                return None
            concrete = [child for child in children if child is not None]
            if not concrete:
                return EmptyPattern()
            if len(concrete) == 1:
                return concrete[0]
            return ConcatenationPattern(concrete)
        if isinstance(node, RegexAlternation):
            children = [self._convert(option) for option in node.options]
            if not children or any(child is None for child in children):
                return None
            concrete = [child for child in children if child is not None]
            return concrete[0] if len(concrete) == 1 else AlternationPattern(concrete)
        if isinstance(node, RegexRepetition):
            body = self._convert(node.body)
            if body is None:
                return None
            return QuantifiedPattern(
                body,
                node.minimum,
                node.maximum,
                node.greedy,
            )
        return None


class RegexOptimizationPlusRule(RewriteRule):
    """Optimize regular islands only with compositional or whole-pattern proof."""

    MAX_TRUTH_REGIONS = 15
    MAX_CONTEXT_TRUTH_REGIONS = 31
    MAX_CONTEXT_VARIABLES = 8
    MAX_GENERATED_PREDICATE_LENGTH = 2000
    MAX_FRAGMENT_VARIABLES = 4
    MAX_FRAGMENT_NODES = 24
    MAX_FRAGMENT_WINDOW_WIDTH = 8
    MAX_QUANTIFIER_BOUND = 1000
    MAX_FRAGMENT_ATTEMPTS = 16
    MAX_REGEX_LENGTH = 512
    MAX_CANDIDATES = 128
    MAX_OPTIMIZER_PASSES = 4
    MAX_NFA_STATES = 1200
    MAX_PRODUCT_STATES = 100_000
    MAX_OUTPUT_LENGTH_FACTOR = 4.0
    OPTIMIZER_TIMEOUT_SECONDS = 5.0
    OPTIMIZER_TOTAL_TIMEOUT_SECONDS = 5.0
    # A successful rewrite can introduce generated variables.  Feeding those
    # variables back into the same search was the main source of exponential
    # DEFINE growth, so one rule invocation commits at most one rewrite.
    MAX_REWRITES_PER_APPLY = 1

    # The common A+ B+ boundary has a fixed, linear-size construction and does
    # not need Greenery's general subtree enumeration.
    MAX_ADJACENT_PLUS_ATTEMPTS = 16
    MAX_ADJACENT_PLUS_GENERATED_VARIABLES = 2
    MAX_ADJACENT_PLUS_PREDICATE_LENGTH = 512

    # Reluctant quantifiers are normalized only when a mandatory, disjoint
    # successor makes the repetition boundary unique.  One additional
    # contextual theorem handles X*? Y X* by narrowing the prefix to X \ Y.
    MAX_RELUCTANT_ELIMINATION_ATTEMPTS = 16
    MAX_RELUCTANT_GENERATED_VARIABLES = 1
    MAX_RELUCTANT_PREDICATE_LENGTH = 512

    def __init__(self) -> None:
        # Duration of the most recent apply() call.  Keep it on both the rule
        # and the AST (see apply()) so callers can retrieve it regardless of
        # whether they retain the rule instance.
        self.last_rewrite_elapsed_seconds = 0.0

    @property
    def name(self) -> str:
        return "regex_optimization_plus"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Use a cheap gate; expensive proof search happens only in apply."""

        if (
            ast.pattern is None
            or ast.define is None
            or ast.raw_sql_override is not None
        ):
            return False
        if ast.measures is not None and ast.measures.has_func("CLASSIFIER"):
            return False
        return True

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        started_at = time.perf_counter()
        try:
            original_state = _AstRewriteState.capture(ast)
            optimizer_deadline = time.monotonic() + self.OPTIMIZER_TOTAL_TIMEOUT_SECONDS
            try:
                changed = False
                if self.is_applicable(ast):
                    changed = self._normalize_forced_reluctant_quantifiers(ast)
                for _ in range(self.MAX_REWRITES_PER_APPLY):
                    plan = self._find_priority_plan(
                        ast, optimizer_deadline=optimizer_deadline
                    )
                    if plan is None:
                        break
                    if not self._commit_priority_plan(ast, plan):
                        break
                    changed = True
                return changed
            except Exception:
                original_state.restore(ast)
                return False
            except BaseException:
                original_state.restore(ast)
                raise
        finally:
            elapsed = time.perf_counter() - started_at
            self.last_rewrite_elapsed_seconds = elapsed
            ast.regex_optimization_plus_elapsed_seconds = elapsed

    # --------------------- phase 1: candidate discovery ----------------------

    def _find_priority_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        *,
        optimizer_deadline: Optional[float] = None,
    ) -> Optional[_PriorityRewritePlan]:
        """Run certified special cases before bounded general optimization.

        An unresolved reluctant quantifier blocks general candidate generation:
        standalone priority equivalence is not substitutive under a surrounding
        suffix.  Only explicit contextual theorems may remove such a boundary.
        """

        if not self.is_applicable(ast):
            return None

        plan = self._find_reluctant_elimination_plan(ast)
        if plan is not None:
            return plan

        assert ast.pattern is not None
        if contains_effective_reluctant_quantifier(ast.pattern):
            return None

        plan = self._find_adjacent_plus_plan(ast)
        if plan is not None:
            return plan

        if contains_controlled_adjacent_plus_rewrite(ast.pattern):
            return None

        plan = self._find_absorbed_union_plus_plan(ast)
        if plan is not None:
            return plan

        return self._find_general_plan(ast, optimizer_deadline)

    def _normalize_forced_reluctant_quantifiers(
        self, ast: RPRAbstractSyntaxTree
    ) -> bool:
        """Normalize reluctant variables only at a proved, unique boundary.

        For ``X{m,n}? Y{p,q}`` with ``p >= 1`` and disjoint row predicates,
        every successful match has one possible X/Y split.  Greedy and
        reluctant enumeration therefore select the same rows in every context.

        The same reasoning makes ``(X{1,n}?)+ Y`` equivalent to ``X+ Y``.
        The inner interval contains one, so the nested repetition accepts every
        positive X count; changing its grouping cannot be observed because all
        rows retain the same X classifier.  This second normalization is also
        checked by the exact regex verifier before the AST is changed.

        Unknown, dependent, nullable, and complex-body cases remain unchanged.
        """

        assert ast.pattern is not None
        changed = False

        def unwrap_anchors(node: PatternNode) -> PatternNode:
            while isinstance(node, AnchorPattern):
                node = node.pattern
            return node

        def mandatory_variable_name(node: PatternNode) -> Optional[str]:
            node = unwrap_anchors(node)
            if isinstance(node, VariablePattern):
                return node.name
            if (
                isinstance(node, QuantifiedPattern)
                and isinstance(node.pattern, VariablePattern)
                and node.atLeast >= 1
            ):
                return node.pattern.name
            return None

        def visit(node: PatternNode) -> None:
            nonlocal changed
            if isinstance(node, ConcatenationPattern):
                for left, right in zip(node.patterns, node.patterns[1:]):
                    left_body = unwrap_anchors(left)
                    right_name = mandatory_variable_name(right)
                    if right_name is None:
                        continue
                    # A dependent successor may still expose a necessary,
                    # row-local conjunct.  It is sufficient for proving an
                    # X/Y boundary disjoint, while the complete dependent
                    # DEFINE (for example COUNT(*) <= n) remains untouched.
                    right_expression = self._independent_filter_expression(
                        ast, right_name
                    )
                    if right_expression is None:
                        continue

                    if (
                        is_effective_reluctant_quantifier(left_body)
                        and isinstance(left_body, QuantifiedPattern)
                        and isinstance(left_body.pattern, VariablePattern)
                    ):
                        left_expression = self._independent_expression(
                            ast, left_body.pattern.name
                        )
                        if (
                            left_expression is not None
                            and self._predicates_proven_disjoint(
                                ast, left_expression, right_expression
                            )
                        ):
                            left_body.isGreedy = True
                            changed = True
                        continue

                    if not (
                        isinstance(left_body, QuantifiedPattern)
                        and left_body.atLeast == 1
                        and left_body.atMost is None
                        and left_body.isGreedy
                        and isinstance(left_body.pattern, QuantifiedPattern)
                    ):
                        continue
                    inner = left_body.pattern
                    if not (
                        inner.atLeast == 1
                        and inner.atMost is not None
                        and inner.atMost > 1
                        and not inner.isGreedy
                        and isinstance(inner.pattern, VariablePattern)
                    ):
                        continue
                    left_expression = self._independent_expression(
                        ast, inner.pattern.name
                    )
                    if (
                        left_expression is None
                        or not self._predicates_proven_disjoint(
                            ast, left_expression, right_expression
                        )
                        or not self._verify_nested_reluctant_plus(inner.atMost)
                    ):
                        continue

                    # Keep the existing outer plus node (and any anchor that
                    # wraps it); only erase the unobservable inner grouping.
                    left_body.pattern = copy.deepcopy(inner.pattern)
                    changed = True

            if isinstance(node, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
                visit(node.pattern)
                return
            for child in getattr(node, "patterns", ()):
                visit(child)

        visit(ast.pattern)
        return changed

    def _verify_nested_reluctant_plus(self, inner_maximum: int) -> bool:
        """Prove ``(X{1,n}?)+ Y`` and ``X+ Y`` priority-equivalent.

        X and Y use distinct symbolic row classes.  The caller separately
        proves that their SQL predicates are disjoint, making this alphabet an
        exact model of every row that either variable can consume.  Including
        mandatory Y in the proof fixes the fragment boundary; UNKNOWN and all
        verifier failures are safe no-ops.
        """

        x_code = _PRIVATE_TOKEN_START
        y_code = _PRIVATE_TOKEN_START + 1

        def atom(code: int) -> RegexAtom:
            return RegexAtom(CharSet.make(((code, code),)))

        inner = RegexRepetition(atom(x_code), 1, inner_maximum, False)
        candidate = RegexConcatenation(
            (RegexRepetition(atom(x_code), 1, None, True), atom(y_code))
        )
        # Parenthesize explicitly: serializing a Repetition whose body is
        # another Repetition without this wrapper would produce the invalid
        # token sequence ``X{1,n}?+``.
        source_pattern = f"({to_regex(inner)})+{to_regex(atom(y_code))}"
        try:
            verification = verify_priority_equivalent(
                source_pattern,
                to_regex(candidate),
                max_nfa_states=self.MAX_NFA_STATES,
                max_product_states=self.MAX_PRODUCT_STATES,
            )
        except Exception:
            return False
        return verification.verdict is Verdict.PROVED

    def _find_reluctant_elimination_plan(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[_PriorityRewritePlan]:
        """Search the small catalogue of context-safe reluctant rewrites."""

        assert ast.pattern is not None
        attempts = 0
        for location, fragment in iter_regular_fragments(
            ast.pattern, max_window_width=self.MAX_FRAGMENT_WINDOW_WIDTH
        ):
            if self._crosses_preserved_boundary(
                ast, fragment
            ) or not is_reluctant_prefix_fragment(fragment):
                continue
            attempts += 1
            if attempts > self.MAX_RELUCTANT_ELIMINATION_ATTEMPTS:
                break
            plan = self._build_reluctant_prefix_plan(ast, location, fragment)
            if plan is not None:
                return plan
        return None

    def _find_adjacent_plus_plan(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[_PriorityRewritePlan]:
        """Search phase 1a: try the fixed construction for ``A+ B+``."""

        assert ast.pattern is not None
        attempts = 0
        for location, fragment in iter_regular_fragments(
            ast.pattern, max_window_width=self.MAX_FRAGMENT_WINDOW_WIDTH
        ):
            if self._crosses_preserved_boundary(
                ast, fragment
            ) or not is_adjacent_plus_fragment(fragment):
                continue
            attempts += 1
            if attempts > self.MAX_ADJACENT_PLUS_ATTEMPTS:
                break
            plan = self._build_adjacent_plus_plan(ast, location, fragment)
            if plan is not None:
                return plan
        return None

    def _find_absorbed_union_plus_plan(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[_PriorityRewritePlan]:
        """Search the contextual ``(X+ | Y+)+ -> (X union Y)+`` theorem."""

        assert ast.pattern is not None
        attempts = 0
        for location, fragment in iter_regular_fragments(
            ast.pattern, max_window_width=self.MAX_FRAGMENT_WINDOW_WIDTH
        ):
            if self._crosses_preserved_boundary(ast, fragment):
                continue
            if not (
                isinstance(fragment, QuantifiedPattern)
                and fragment.atLeast == 1
                and fragment.atMost is None
                and fragment.isGreedy
                and isinstance(fragment.pattern, AlternationPattern)
            ):
                continue
            attempts += 1
            if attempts > self.MAX_FRAGMENT_ATTEMPTS:
                break
            plan = self._build_absorbed_union_plus_plan(ast, location, fragment)
            if plan is not None:
                return plan
        return None

    def _find_general_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        optimizer_deadline: Optional[float],
    ) -> Optional[_PriorityRewritePlan]:
        """Search phases 1b-3: filter, encode, optimize, and validate islands."""

        assert ast.pattern is not None
        attempts = 0
        for location, fragment in iter_regular_fragments(
            ast.pattern, max_window_width=self.MAX_FRAGMENT_WINDOW_WIDTH
        ):
            if not self._is_general_candidate(ast, fragment):
                continue
            attempts += 1
            if attempts > self.MAX_FRAGMENT_ATTEMPTS:
                break
            encoding = self._prepare_fragment_encoding(
                ast, location, fragment, require_compact_regions=True
            )
            if encoding is None:
                continue
            if (
                optimizer_deadline is not None
                and time.monotonic() >= optimizer_deadline
            ):
                break

            # Optimize in discovery order and stop at the first usable plan.
            # The worker is persistent, so eagerly optimizing every remaining
            # fragment only adds exact-product work after the answer is known.
            pass_limit = (
                self.MAX_OPTIMIZER_PASSES
                if needs_multi_pass_optimization(fragment)
                else 1
            )
            # A match-only proposal can discard fallback exits.  Certify it
            # in its complete caller before committing; if that proof fails,
            # retain the strict search for existing compositional rewrites.
            modes = [EquivalenceMode.LANGUAGE_AND_MATCH]
            if self._standalone_context_observation_safe(ast):
                modes.insert(0, EquivalenceMode.MATCH_ONLY)
            for mode in modes:
                optimized = self._run_priority_optimizers(
                    [encoding.source],
                    pass_limits=[pass_limit],
                    deadline=optimizer_deadline,
                    equivalence_mode=mode,
                )[0]
                plan = self._priority_plan_from_optimized(location, encoding, optimized)
                if plan is not None and self._certify_priority_plan(
                    ast, plan, optimizer_deadline
                ):
                    return plan
        return None

    def _certify_priority_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: _PriorityRewritePlan,
        deadline: Optional[float],
    ) -> bool:
        """Authorize substitution only within the scope of its certificate.

        A standalone result proves selected spans, not variable traces.  The
        SQL observation guard applies even when the fragment is the root.
        Non-root fragments additionally require a complete-context proof;
        their syntactic right edge alone says nothing about outer suffixes
        or subsequent iterations of an enclosing repetition.
        """

        if plan.proof_scope is ProofScope.COMPOSITIONAL:
            return True
        if (
            plan.proof_scope is not ProofScope.STANDALONE
            or not self._standalone_context_observation_safe(ast)
        ):
            return False
        if plan.location.kind != "root" and not self._certify_in_complete_pattern(
            ast, plan, deadline
        ):
            return False
        plan.proof_scope = ProofScope.WHOLE_PATTERN
        return True

    def _is_general_candidate(
        self,
        ast: RPRAbstractSyntaxTree,
        fragment: PatternNode,
    ) -> bool:
        """Apply only cheap structural bounds before predicate encoding."""

        node_count = regular_node_count(
            fragment, max_quantifier_bound=self.MAX_QUANTIFIER_BOUND
        )
        return (
            not self._crosses_preserved_boundary(ast, fragment)
            and not contains_effective_reluctant_quantifier(fragment)
            and node_count is not None
            and node_count <= self.MAX_FRAGMENT_NODES
            and len(fragment.get_variables()) <= self.MAX_FRAGMENT_VARIABLES
            and has_optimization_opportunity(fragment)
        )

    def _crosses_preserved_boundary(
        self,
        ast: RPRAbstractSyntaxTree,
        fragment: PatternNode,
    ) -> bool:
        """Reject fragments crossing syntax boundaries erased by simplification."""

        fragment_variables = fragment.get_variables()
        if not fragment_variables:
            return False
        live_variables = (
            ast.pattern.get_variables() if ast.pattern is not None else set()
        )
        for boundary in getattr(ast, "pattern_rewrite_boundaries", ()):
            boundary_variables = set(boundary.variables) & live_variables
            touched = fragment_variables & boundary_variables
            if not touched:
                continue
            if not fragment_variables <= boundary_variables:
                return True
            if boundary.kind == "permute":
                touched_elements = sum(
                    bool(fragment_variables & (set(element) & live_variables))
                    for element in boundary.elements
                )
                if touched_elements > 1:
                    return True
        return False

    # ----------------- phase 2: predicate/alphabet encoding -----------------

    def _priority_plan_from_optimized(
        self,
        location: PatternLocation,
        encoding: _FragmentEncoding,
        optimized: Optional[PriorityOptimizerOutcome],
    ) -> Optional[_PriorityRewritePlan]:
        if (
            optimized is None
            or not optimized.pattern
            or optimized.pattern == encoding.source
        ):
            return None
        try:
            candidate = parse_regex(optimized.pattern)
        except Exception:
            return None
        if not self._regex_atoms_are_regions(candidate, encoding.region_codes):
            return None
        if encoding.existing_atoms_only:
            if not self._regex_atoms_are_existing_variables(
                candidate,
                encoding.region_codes,
                encoding.variable_tokens,
            ):
                return None

        return _PriorityRewritePlan(
            location,
            encoding.old_names,
            encoding.expressions,
            encoding.regions,
            encoding.variable_tokens,
            candidate,
            original_conditions=encoding.original_conditions,
            existing_atoms_only=encoding.existing_atoms_only,
            proof_scope=optimized.proof_scope,
        )

    def _prepare_fragment_encoding(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        fragment: PatternNode,
        *,
        require_compact_regions: bool = False,
    ) -> Optional[_FragmentEncoding]:
        """Phase 2: encode one SQL fragment over a finite truth alphabet."""

        names = tuple(self._ordered_variable_names(fragment))
        if not names or not self._fragment_labels_are_rewritable(ast, fragment, names):
            return None

        model = self._collect_predicate_model(ast, location, names)
        if model is None:
            return None
        if require_compact_regions and not self._has_compact_truth_regions(
            ast, model.base_infos
        ):
            return None

        regions = self._build_verification_regions(ast, model)
        if not regions or len(regions) > self.MAX_TRUTH_REGIONS:
            return None
        self._assign_region_tokens(regions)

        variable_tokens = self._build_variable_token_map(model.names, regions)
        if variable_tokens is None:
            return None

        source_node = self._to_regex_node(fragment, variable_tokens)
        if source_node is None:
            return None
        source = to_regex(source_node)
        if not source or len(source) > self.MAX_REGEX_LENGTH:
            return None

        return _FragmentEncoding(
            model.names,
            model.expressions,
            tuple(regions),
            variable_tokens,
            source,
            (model.original_conditions if model.has_symbolic_dependencies else None),
            model.has_symbolic_dependencies,
        )

    def _fragment_labels_are_rewritable(
        self,
        ast: RPRAbstractSyntaxTree,
        fragment: PatternNode,
        names: Sequence[str],
    ) -> bool:
        """Require every changed label to be local and externally unobservable."""

        global_counts = self._count_variable_occurrences(ast.pattern)
        fragment_counts = self._count_variable_occurrences(fragment)
        return (
            all(global_counts.get(name) == fragment_counts.get(name) for name in names)
            and not any(self._label_is_directly_observable(ast, name) for name in names)
            and self._subsets_are_preservable(ast, set(names))
        )

    def _collect_predicate_model(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        names: Tuple[str, ...],
    ) -> Optional[_PredicateModel]:
        """Classify DEFINE predicates as exact row-local or symbolic axes."""

        completed_prefix_names = self._completed_prefix_names(ast, location)
        expressions: Dict[str, Expr] = {}
        base_infos: List[_SimplePattern] = []
        independent_axes: Dict[int, int] = {}
        dependent_axes: List[Tuple[int, int]] = []
        original_conditions: Dict[str, Optional[Condition]] = {}

        assert ast.define is not None
        for variable_index, name in enumerate(names):
            condition = ast.define.definitions.get(name)
            if isinstance(condition, DependentCondition):
                if not self._is_prefix_stable_expression(
                    condition.expression, completed_prefix_names
                ):
                    return None
                expression = condition.get_independent_filter()
                expression = (
                    _strip_var_qualifier(expression, name)
                    if expression is not None
                    else BinaryExpr(Literal(1), "=", Literal(1))
                )
                if (
                    expression is None
                    or not self._is_safe_row_predicate(expression)
                    or self._is_constant_false(expression)
                ):
                    return None
                expressions[name] = copy.deepcopy(condition.expression)
                original_conditions[name] = condition
                dependent_axes.append((variable_index, len(base_infos)))
            else:
                expression = self._independent_expression(ast, name)
                if expression is None:
                    return None
                expressions[name] = expression
                original_conditions[name] = condition
                independent_axes[variable_index] = len(base_infos)

            base_infos.append(
                _SimplePattern(VariablePattern(name), name, expression, 1, 1)
            )

        return _PredicateModel(
            names,
            expressions,
            base_infos,
            independent_axes,
            dependent_axes,
            original_conditions,
        )

    def _build_verification_regions(
        self, ast: RPRAbstractSyntaxTree, model: _PredicateModel
    ) -> List[_TruthRegion]:
        """Partition predicates, then expand prefix-dependent Boolean axes.

        The expansion over-approximates every row membership possible for a
        fixed prefix state.  Equivalence over this larger alphabet therefore
        implies equivalence for the actual dependent predicates.
        """

        # First partition the exact row-local predicates and the independent
        # filters of dependent variables.  A dependent predicate then becomes
        # a free Boolean dimension wherever its necessary filter is true.  The
        # resulting alphabet is an over-approximation of all row memberships
        # possible for every fixed prefix state, so an equivalence proved over
        # it remains valid for the actual dependent expressions.
        base_regions = self._build_truth_regions(ast, model.base_infos)
        regions: List[_TruthRegion] = []
        if model.dependent_axes:
            for base_region in base_regions:
                enabled_dependent_axes = [
                    (variable_index, axis_index)
                    for variable_index, axis_index in model.dependent_axes
                    if base_region.signature & (1 << axis_index)
                ]
                for dependent_signature in range(1 << len(enabled_dependent_axes)):
                    variable_signature = 0
                    for variable_index, axis_index in model.independent_axes.items():
                        if base_region.signature & (1 << axis_index):
                            variable_signature |= 1 << variable_index
                    for bit_index, (variable_index, _) in enumerate(
                        enabled_dependent_axes
                    ):
                        if dependent_signature & (1 << bit_index):
                            variable_signature |= 1 << variable_index
                    if variable_signature:
                        regions.append(
                            _TruthRegion(
                                variable_signature,
                                copy.deepcopy(base_region.expression),
                            )
                        )
        else:
            regions = base_regions
        return regions

    @staticmethod
    def _assign_region_tokens(regions: Sequence[_TruthRegion]) -> None:
        for index, region in enumerate(regions):
            region.token = chr(_PRIVATE_TOKEN_START + index)

    @staticmethod
    def _build_variable_token_map(
        names: Sequence[str], regions: Sequence[_TruthRegion]
    ) -> Optional[Dict[str, frozenset[int]]]:
        variable_tokens: Dict[str, frozenset[int]] = {}
        for index, name in enumerate(names):
            codes = frozenset(
                ord(region.token)
                for region in regions
                if region.signature & (1 << index)
            )
            if not codes:
                return None
            variable_tokens[name] = codes
        return variable_tokens

    def _completed_prefix_names(
        self, ast: RPRAbstractSyntaxTree, location: PatternLocation
    ) -> Set[str]:
        """Return names whose every syntactic occurrence precedes a fragment."""

        if location.kind != "list" or location.patterns is None:
            return set()
        prefix_patterns = location.patterns[: location.start]
        if not prefix_patterns:
            return set()
        prefix = ConcatenationPattern(prefix_patterns)
        prefix_counts = self._count_variable_occurrences(prefix)
        global_counts = self._count_variable_occurrences(ast.pattern)
        return {
            name
            for name, count in prefix_counts.items()
            if count > 0 and global_counts.get(name) == count
        }

    def _has_compact_truth_regions(
        self,
        ast: RPRAbstractSyntaxTree,
        infos: Sequence[_SimplePattern],
    ) -> bool:
        """Avoid the exponential generic minterm fallback.

        Finite predicates on one column and ranges on one axis have compact,
        purpose-built partitions.  Heterogeneous predicates are admitted only
        by the bounded two-variable A+ B+ construction.
        """

        # Constant-TRUE predicates form exactly one truth region, so this case
        # does not need the generic minterm construction's exponential space.
        if all(self._is_constant_true(info.expression) for info in infos):
            return True

        finite = [self._extract_finite_predicate(info.expression) for info in infos]
        if all(item is not None for item in finite):
            columns = {item.column for item in finite if item is not None}
            if len(columns) == 1:
                return True

        context = ast.expression_simplify_context()
        ranges = [
            self._extract_single_range(info.expression, context) for info in infos
        ]
        if all(item is not None for item in ranges):
            concrete = [item for item in ranges if item is not None]
            return (
                len({item.column.to_string() for item in concrete}) == 1
                and len({item.domain for item in concrete}) == 1
            )
        return False

    def _build_reluctant_prefix_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        fragment: PatternNode,
    ) -> Optional[_PriorityRewritePlan]:
        """Rewrite X*? Y X* as (X minus Y)* Y X*.

        The reluctant prefix selects the earliest row that can serve as Y.
        Rows before that boundary must satisfy X and not Y, so replacing the
        prefix with the exact difference removes reluctance without changing
        the ordered fallback behavior of the surrounding pattern.
        """

        if not is_reluctant_prefix_fragment(fragment):
            return None
        assert isinstance(fragment, ConcatenationPattern)
        first, middle, third = fragment.patterns
        assert isinstance(first, QuantifiedPattern)
        assert isinstance(first.pattern, VariablePattern)
        assert isinstance(middle, VariablePattern)
        assert isinstance(third, QuantifiedPattern)

        encoding = self._prepare_fragment_encoding(ast, location, fragment)
        if encoding is None:
            return None

        repeated_codes = encoding.variable_tokens[first.pattern.name]
        middle_codes = encoding.variable_tokens[middle.name]
        prefix_codes = repeated_codes - middle_codes
        if not middle_codes <= repeated_codes or not prefix_codes:
            return None

        def atom(codes: Iterable[int]) -> RegexAtom:
            return RegexAtom(CharSet.make((code, code) for code in codes))

        candidate = RegexConcatenation(
            (
                RegexRepetition(atom(prefix_codes), 0, None, True),
                atom(middle_codes),
                RegexRepetition(atom(repeated_codes), 0, None, True),
            )
        )
        candidate_source = to_regex(candidate)
        try:
            verification = verify_priority_equivalent(
                encoding.source,
                candidate_source,
                max_nfa_states=self.MAX_NFA_STATES,
                max_product_states=self.MAX_PRODUCT_STATES,
            )
        except Exception:
            return None
        if verification.verdict is not Verdict.PROVED:
            return None
        if (
            encoding.existing_atoms_only
            and not self._regex_atoms_are_existing_variables(
                candidate,
                encoding.region_codes,
                encoding.variable_tokens,
            )
        ):
            return None

        return _PriorityRewritePlan(
            location,
            encoding.old_names,
            encoding.expressions,
            encoding.regions,
            encoding.variable_tokens,
            candidate,
            max_generated_variables=self.MAX_RELUCTANT_GENERATED_VARIABLES,
            max_generated_predicate_length=self.MAX_RELUCTANT_PREDICATE_LENGTH,
            original_conditions=encoding.original_conditions,
            existing_atoms_only=encoding.existing_atoms_only,
        )

    def _build_adjacent_plus_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        fragment: PatternNode,
    ) -> Optional[_PriorityRewritePlan]:
        """Rewrite A+ B+ with a constant-size, priority-safe construction."""

        if not is_adjacent_plus_fragment(fragment):
            return None
        assert isinstance(fragment, ConcatenationPattern)
        left_quantifier, right_quantifier = fragment.patterns
        assert isinstance(left_quantifier, QuantifiedPattern)
        assert isinstance(right_quantifier, QuantifiedPattern)
        assert isinstance(left_quantifier.pattern, VariablePattern)
        assert isinstance(right_quantifier.pattern, VariablePattern)
        if left_quantifier.pattern.name == right_quantifier.pattern.name:
            return None

        encoding = self._prepare_fragment_encoding(ast, location, fragment)
        if encoding is None or len(encoding.old_names) != 2:
            return None

        left_codes = encoding.variable_tokens[left_quantifier.pattern.name]
        right_codes = encoding.variable_tokens[right_quantifier.pattern.name]
        candidate = self._adjacent_plus_candidate(left_codes, right_codes)
        if candidate is None:
            return None

        candidate_source = to_regex(candidate)
        if candidate_source == encoding.source:
            return None
        try:
            verification = verify_priority_equivalent(
                encoding.source,
                candidate_source,
                max_nfa_states=self.MAX_NFA_STATES,
                max_product_states=self.MAX_PRODUCT_STATES,
            )
        except Exception:
            return None
        if verification.verdict is not Verdict.PROVED:
            return None
        if encoding.existing_atoms_only:
            if not self._regex_atoms_are_existing_variables(
                candidate,
                {ord(region.token) for region in encoding.regions},
                encoding.variable_tokens,
            ):
                return None

        return _PriorityRewritePlan(
            location,
            encoding.old_names,
            encoding.expressions,
            encoding.regions,
            encoding.variable_tokens,
            candidate,
            max_generated_variables=self.MAX_ADJACENT_PLUS_GENERATED_VARIABLES,
            max_generated_predicate_length=self.MAX_ADJACENT_PLUS_PREDICATE_LENGTH,
            original_conditions=encoding.original_conditions,
            existing_atoms_only=encoding.existing_atoms_only,
        )

    def _build_absorbed_union_plus_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        fragment: PatternNode,
    ) -> Optional[_PriorityRewritePlan]:
        """Collapse a greedy plus of greedy variable-plus alternatives.

        When one alternative predicate contains every other alternative, both
        patterns expose exactly the same non-empty endpoints in longest-first
        order in every suffix context.  Labels in the removed alternatives
        must already be unobservable, as enforced by fragment encoding.
        """

        if not (
            isinstance(fragment, QuantifiedPattern)
            and fragment.atLeast == 1
            and fragment.atMost is None
            and fragment.isGreedy
            and isinstance(fragment.pattern, AlternationPattern)
        ):
            return None

        branch_names: List[str] = []
        for option in fragment.pattern.patterns:
            if not (
                isinstance(option, QuantifiedPattern)
                and option.atLeast == 1
                and option.atMost is None
                and option.isGreedy
                and isinstance(option.pattern, VariablePattern)
            ):
                return None
            branch_names.append(option.pattern.name)
        if len(branch_names) < 2 or len(set(branch_names)) != len(branch_names):
            return None

        encoding = self._prepare_fragment_encoding(ast, location, fragment)
        if encoding is None:
            return None
        dominant_name = next(
            (
                name
                for name in branch_names
                if all(
                    encoding.variable_tokens[other] <= encoding.variable_tokens[name]
                    for other in branch_names
                )
            ),
            None,
        )
        if dominant_name is None:
            return None

        dominant_codes = encoding.variable_tokens[dominant_name]
        candidate = RegexRepetition(
            RegexAtom(CharSet.make((code, code) for code in dominant_codes)),
            1,
            None,
            True,
        )
        try:
            verification = verify_priority_equivalent(
                encoding.source,
                to_regex(candidate),
                max_nfa_states=self.MAX_NFA_STATES,
                max_product_states=self.MAX_PRODUCT_STATES,
            )
        except Exception:
            return None
        if verification.verdict is not Verdict.PROVED:
            return None
        if (
            encoding.existing_atoms_only
            and not self._regex_atoms_are_existing_variables(
                candidate, encoding.region_codes, encoding.variable_tokens
            )
        ):
            return None

        return _PriorityRewritePlan(
            location,
            encoding.old_names,
            encoding.expressions,
            encoding.regions,
            encoding.variable_tokens,
            candidate,
            original_conditions=encoding.original_conditions,
            existing_atoms_only=encoding.existing_atoms_only,
            proof_scope=ProofScope.COMPOSITIONAL,
        )

    def _is_prefix_stable_expression(
        self, expression: Expr, completed_prefix_names: Set[str]
    ) -> bool:
        """Whether all non-row-local inputs are fixed by a completed prefix."""

        if not completed_prefix_names:
            return False

        stable_functions = {"AVG", "SUM", "COUNT", "MAX", "MIN", "FIRST", "LAST"}
        found_prefix_dependency = False

        def has_prefix_reference(node: Expr) -> bool:
            if isinstance(node, ColumnRef):
                return node.qualifier in completed_prefix_names
            if isinstance(node, FunctionCall):
                return any(has_prefix_reference(argument) for argument in node.args)
            if isinstance(node, BinaryExpr):
                return has_prefix_reference(node.left) or has_prefix_reference(
                    node.right
                )
            if isinstance(node, BetweenExpr):
                return (
                    has_prefix_reference(node.expr)
                    or has_prefix_reference(node.lower)
                    or has_prefix_reference(node.upper)
                )
            if isinstance(node, InExpr):
                return has_prefix_reference(node.expr) or any(
                    has_prefix_reference(value) for value in node.values
                )
            if isinstance(node, (IsNullExpr, NotExpr, SemanticsExpr)):
                return has_prefix_reference(node.expr)
            if isinstance(node, CaseExpr):
                return any(
                    has_prefix_reference(condition) or has_prefix_reference(result)
                    for condition, result in node.cases
                ) or (
                    node.else_expr is not None and has_prefix_reference(node.else_expr)
                )
            return False

        def visit(node: Expr, inside_dependent_function: bool = False) -> bool:
            nonlocal found_prefix_dependency
            if isinstance(node, (Literal, NullLiteral)):
                return True
            if isinstance(node, SubqueryExpr):
                return False
            if isinstance(node, ColumnRef):
                if node.qualifier is not None:
                    if node.qualifier not in completed_prefix_names:
                        return False
                    found_prefix_dependency = True
                    return True
                # An unqualified column is the current variable's row.  That is
                # stable for the row being tested, but not as the input to a
                # running aggregate/navigation function.
                return not inside_dependent_function
            if isinstance(node, FunctionCall):
                function_name = node.name.upper()
                is_dependent = function_name in DEPENDENT_FUNCS
                # Fixed arguments do not make an unknown/UDF call stable:
                # reevaluation along another matching path may change it.
                if function_name not in stable_functions:
                    return False
                if is_dependent and not any(
                    has_prefix_reference(argument) for argument in node.args
                ):
                    # SUM(1), COUNT(*), and similar expressions are not tied to
                    # a completed prefix and can evolve while B grows.
                    return False
                if not all(
                    visit(argument, inside_dependent_function or is_dependent)
                    for argument in node.args
                ):
                    return False
                return True
            if isinstance(node, BinaryExpr):
                return visit(node.left, inside_dependent_function) and visit(
                    node.right, inside_dependent_function
                )
            if isinstance(node, BetweenExpr):
                return (
                    visit(node.expr, inside_dependent_function)
                    and visit(node.lower, inside_dependent_function)
                    and visit(node.upper, inside_dependent_function)
                )
            if isinstance(node, InExpr):
                return visit(node.expr, inside_dependent_function) and all(
                    visit(value, inside_dependent_function) for value in node.values
                )
            if isinstance(node, (IsNullExpr, NotExpr, SemanticsExpr)):
                return visit(node.expr, inside_dependent_function)
            if isinstance(node, CaseExpr):
                return all(
                    visit(condition, inside_dependent_function)
                    and visit(result, inside_dependent_function)
                    for condition, result in node.cases
                ) and (
                    node.else_expr is None
                    or visit(node.else_expr, inside_dependent_function)
                )
            return False

        return visit(expression) and found_prefix_dependency

    def _adjacent_plus_candidate(
        self,
        left_codes: frozenset[int],
        right_codes: frozenset[int],
    ) -> Optional[RegexNode]:
        """Return the fixed priority-safe construction for one A+ B+ pair."""

        intersection = left_codes & right_codes
        if not intersection:
            # Disjoint repetitions have no ambiguous boundary to remove.
            return None

        def atom(codes: Iterable[int]) -> RegexAtom:
            return RegexAtom(CharSet.make((code, code) for code in codes))

        left_atom = atom(left_codes)
        right_atom = atom(right_codes)
        if left_codes <= right_codes:
            # Every A row is also a B row: leave exactly one row with A.
            return RegexConcatenation(
                (
                    left_atom,
                    RegexRepetition(right_atom, 1, None, True),
                )
            )
        if right_codes <= left_codes:
            # Every B row is also an A row: leave exactly one row with B.
            return RegexConcatenation(
                (
                    RegexRepetition(left_atom, 1, None, True),
                    right_atom,
                )
            )

        # For a partial overlap, the first B row is either B-only (after which
        # arbitrary B rows follow) or the final overlap row.  This replaces an
        # ambiguous repeated/repeated boundary with two exact regions and never
        # enumerates a general Boolean DNF.
        right_only = right_codes - left_codes
        return RegexConcatenation(
            (
                RegexRepetition(left_atom, 1, None, True),
                RegexAlternation(
                    (
                        RegexConcatenation(
                            (
                                atom(right_only),
                                RegexRepetition(right_atom, 0, None, True),
                            )
                        ),
                        atom(intersection),
                    )
                ),
            )
        )

    def _ordered_variable_names(self, node: PatternNode) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()

        def visit(current: PatternNode) -> None:
            if isinstance(current, VariablePattern):
                if current.name not in seen:
                    seen.add(current.name)
                    result.append(current.name)
                return
            if isinstance(current, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
                visit(current.pattern)
                return
            for child in getattr(current, "patterns", []):
                visit(child)

        visit(node)
        return result

    def _to_regex_node(
        self,
        node: PatternNode,
        variable_tokens: Dict[str, frozenset[int]],
    ) -> Optional[RegexNode]:
        if isinstance(node, EmptyPattern):
            return RegexEmpty()
        if isinstance(node, VariablePattern):
            codes = variable_tokens.get(node.name)
            if not codes:
                return None
            return RegexAtom(CharSet.make((code, code) for code in codes))
        if isinstance(node, ConcatenationPattern):
            parts = [
                self._to_regex_node(child, variable_tokens) for child in node.patterns
            ]
            if any(part is None for part in parts):
                return None
            return RegexConcatenation(tuple(part for part in parts if part is not None))
        if isinstance(node, AlternationPattern):
            options = [
                self._to_regex_node(child, variable_tokens) for child in node.patterns
            ]
            if not options or any(option is None for option in options):
                return None
            return RegexAlternation(
                tuple(option for option in options if option is not None)
            )
        if isinstance(node, QuantifiedPattern):
            body = self._to_regex_node(node.pattern, variable_tokens)
            if body is None:
                return None
            return RegexRepetition(
                body,
                node.atLeast,
                node.atMost,
                node.isGreedy,
            )
        return None

    def _regex_atoms_are_regions(
        self, node: RegexNode, allowed_codes: Set[int]
    ) -> bool:
        if isinstance(node, RegexEmpty):
            return True
        if isinstance(node, RegexAtom):
            codes = self._charset_region_codes(node.chars, allowed_codes)
            return bool(codes)
        if isinstance(node, RegexConcatenation):
            return all(
                self._regex_atoms_are_regions(part, allowed_codes)
                for part in node.parts
            )
        if isinstance(node, RegexAlternation):
            return bool(node.options) and all(
                self._regex_atoms_are_regions(option, allowed_codes)
                for option in node.options
            )
        if isinstance(node, RegexRepetition):
            return self._regex_atoms_are_regions(node.body, allowed_codes)
        return False

    def _regex_atoms_are_existing_variables(
        self,
        node: RegexNode,
        allowed_codes: Set[int],
        variable_tokens: Dict[str, frozenset[int]],
    ) -> bool:
        """Reject symbolic outputs that would need a synthesized predicate."""

        existing_token_sets = set(variable_tokens.values())
        if isinstance(node, RegexEmpty):
            return True
        if isinstance(node, RegexAtom):
            codes = self._charset_region_codes(node.chars, allowed_codes)
            return codes is not None and codes in existing_token_sets
        if isinstance(node, RegexConcatenation):
            return all(
                self._regex_atoms_are_existing_variables(
                    part, allowed_codes, variable_tokens
                )
                for part in node.parts
            )
        if isinstance(node, RegexAlternation):
            return bool(node.options) and all(
                self._regex_atoms_are_existing_variables(
                    option, allowed_codes, variable_tokens
                )
                for option in node.options
            )
        if isinstance(node, RegexRepetition):
            return self._regex_atoms_are_existing_variables(
                node.body, allowed_codes, variable_tokens
            )
        return False

    def _charset_region_codes(
        self, chars: CharSet, allowed_codes: Set[int]
    ) -> Optional[frozenset[int]]:
        codes: Set[int] = set()
        for lower, upper in chars.ranges:
            if upper - lower + 1 > len(allowed_codes):
                return None
            codes.update(range(lower, upper + 1))
            if len(codes) > len(allowed_codes):
                return None
        if not codes or not codes.issubset(allowed_codes):
            return None
        return frozenset(codes)

    # ------------------ complete-context certification ---------------------

    def _certify_in_complete_pattern(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: _PriorityRewritePlan,
        deadline: Optional[float],
    ) -> bool:
        """Prove a standalone candidate after restoring its complete PATTERN.

        Preferred standalone-match equality is not substitutive: a suffix can
        reject the preferred local exit and expose a differently ordered
        fallback.  Every non-compositional fragment candidate is therefore
        rebuilt in the complete row pattern over one joint predicate alphabet.

        Prefix-dependent predicates deliberately fail closed here.  Their
        local symbolic regions cannot be lifted into a joint alphabet without
        retaining additional dependency identity.
        """

        if (
            ast.pattern is None
            or plan.original_conditions is not None
            or not self._standalone_context_observation_safe(ast)
        ):
            return False

        names = tuple(self._ordered_variable_names(ast.pattern))
        if not names or len(names) > self.MAX_CONTEXT_VARIABLES:
            return False

        infos: List[_SimplePattern] = []
        for name in names:
            expression = self._independent_expression(ast, name)
            if expression is None:
                return False
            infos.append(_SimplePattern(VariablePattern(name), name, expression, 1, 1))

        regions = self._build_truth_regions(ast, infos)
        if not regions or len(regions) > self.MAX_CONTEXT_TRUTH_REGIONS:
            return False
        self._assign_region_tokens(regions)
        variable_tokens = self._build_variable_token_map(names, regions)
        if variable_tokens is None:
            return False

        original = self._to_match_regex_node(ast.pattern, variable_tokens)
        lifted = self._lift_candidate_to_context(plan, names, regions)
        if original is None or lifted is None:
            return False
        candidate, replacements = self._to_context_candidate_regex(
            ast.pattern,
            variable_tokens,
            plan.location,
            lifted,
        )
        if candidate is None or replacements != 1:
            return False

        original_source = to_regex(original)
        candidate_source = to_regex(candidate)
        if (
            not original_source
            or not candidate_source
            or len(original_source) > self.MAX_REGEX_LENGTH * 2
            or len(candidate_source) > self.MAX_REGEX_LENGTH * 2
        ):
            return False
        remaining = (
            self.OPTIMIZER_TIMEOUT_SECONDS
            if deadline is None
            else min(self.OPTIMIZER_TIMEOUT_SECONDS, deadline - time.monotonic())
        )
        if remaining <= 0:
            return False
        config = PriorityOptimizerConfig(
            max_candidates=self.MAX_CANDIDATES,
            output_length_factor=self.MAX_OUTPUT_LENGTH_FACTOR,
            max_nfa_states=self.MAX_NFA_STATES,
            max_product_states=self.MAX_PRODUCT_STATES,
            equivalence_mode=EquivalenceMode.MATCH_ONLY,
        )
        return PRIORITY_OPTIMIZER_SERVICE.verify_equivalent(
            original_source,
            candidate_source,
            config,
            remaining,
        )

    def _standalone_context_observation_safe(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Whether whole-span equality determines every SQL-level result."""

        if ast.pattern is None or ast.subsets:
            return False
        skip = " ".join((ast.after_match_skip_to or "").upper().split())
        if skip not in {"PAST LAST ROW", "NEXT ROW"}:
            return False
        if any(
            self._label_is_directly_observable(ast, name)
            for name in ast.pattern.get_variables()
        ):
            return False
        if (ast.rows_per_match or "").upper().startswith(
            "ALL ROWS"
        ) and self._contains_excluded_pattern(ast.pattern):
            return False
        return True

    def _contains_excluded_pattern(self, node: PatternNode) -> bool:
        if isinstance(node, ExcludedPattern):
            return True
        if isinstance(node, (QuantifiedPattern, AnchorPattern)):
            return self._contains_excluded_pattern(node.pattern)
        return any(
            self._contains_excluded_pattern(child)
            for child in getattr(node, "patterns", ())
        )

    def _lift_candidate_to_context(
        self,
        plan: _PriorityRewritePlan,
        context_names: Sequence[str],
        context_regions: Sequence[_TruthRegion],
    ) -> Optional[RegexNode]:
        """Translate fragment-region atoms to the joint context alphabet."""

        local_signature_by_code = {
            ord(region.token): region.signature for region in plan.regions
        }
        context_index_by_name = {
            name: index for index, name in enumerate(context_names)
        }
        try:
            old_indices = [context_index_by_name[name] for name in plan.old_names]
        except KeyError:
            return None

        context_signature_by_code: Dict[int, int] = {}
        for region in context_regions:
            projected = 0
            for old_index, context_index in enumerate(old_indices):
                if region.signature & (1 << context_index):
                    projected |= 1 << old_index
            context_signature_by_code[ord(region.token)] = projected

        allowed_local_codes = set(local_signature_by_code)

        def visit(node: RegexNode) -> Optional[RegexNode]:
            if isinstance(node, RegexEmpty):
                return RegexEmpty()
            if isinstance(node, RegexAtom):
                codes = self._charset_region_codes(node.chars, allowed_local_codes)
                if codes is None:
                    return None
                allowed_signatures = {local_signature_by_code[code] for code in codes}
                context_codes = {
                    code
                    for code, signature in context_signature_by_code.items()
                    if signature in allowed_signatures
                }
                if not context_codes:
                    return None
                return RegexAtom(CharSet.make((code, code) for code in context_codes))
            if isinstance(node, RegexConcatenation):
                parts = [visit(part) for part in node.parts]
                if any(part is None for part in parts):
                    return None
                return RegexConcatenation(
                    tuple(part for part in parts if part is not None)
                )
            if isinstance(node, RegexAlternation):
                options = [visit(option) for option in node.options]
                if not options or any(option is None for option in options):
                    return None
                return RegexAlternation(
                    tuple(option for option in options if option is not None)
                )
            if isinstance(node, RegexRepetition):
                body = visit(node.body)
                if body is None:
                    return None
                return RegexRepetition(body, node.minimum, node.maximum, node.greedy)
            return None

        return visit(plan.candidate)

    def _to_match_regex_node(
        self,
        node: PatternNode,
        variable_tokens: Dict[str, frozenset[int]],
    ) -> Optional[RegexNode]:
        """Encode complete-pattern matching semantics, unwrapping EXCLUDE."""

        if isinstance(node, ExcludedPattern):
            return self._to_match_regex_node(node.pattern, variable_tokens)
        if isinstance(node, (AnchorPattern, PermutationPattern)):
            return None
        if isinstance(node, ConcatenationPattern):
            parts = [
                self._to_match_regex_node(child, variable_tokens)
                for child in node.patterns
            ]
            if any(part is None for part in parts):
                return None
            return RegexConcatenation(tuple(part for part in parts if part is not None))
        if isinstance(node, AlternationPattern):
            options = [
                self._to_match_regex_node(child, variable_tokens)
                for child in node.patterns
            ]
            if not options or any(option is None for option in options):
                return None
            return RegexAlternation(
                tuple(option for option in options if option is not None)
            )
        if isinstance(node, QuantifiedPattern):
            body = self._to_match_regex_node(node.pattern, variable_tokens)
            if body is None:
                return None
            return RegexRepetition(body, node.atLeast, node.atMost, node.isGreedy)
        return self._to_regex_node(node, variable_tokens)

    def _to_context_candidate_regex(
        self,
        root: PatternNode,
        variable_tokens: Dict[str, frozenset[int]],
        location: PatternLocation,
        replacement: RegexNode,
    ) -> Tuple[Optional[RegexNode], int]:
        """Encode ``root`` while substituting exactly one located fragment."""

        if location.kind == "root":
            return replacement, 1

        replacements = 0

        def visit(node: PatternNode) -> Optional[RegexNode]:
            nonlocal replacements
            if isinstance(node, EmptyPattern):
                return RegexEmpty()
            if isinstance(node, VariablePattern):
                codes = variable_tokens.get(node.name)
                if not codes:
                    return None
                return RegexAtom(CharSet.make((code, code) for code in codes))

            if isinstance(node, (ConcatenationPattern, AlternationPattern)):
                children: List[RegexNode] = []
                if location.kind == "list" and node.patterns is location.patterns:
                    before = node.patterns[: location.start]
                    after = node.patterns[location.end :]
                    for child in before:
                        converted = visit(child)
                        if converted is None:
                            return None
                        children.append(converted)
                    children.append(replacement)
                    replacements += 1
                    for child in after:
                        converted = visit(child)
                        if converted is None:
                            return None
                        children.append(converted)
                else:
                    for child in node.patterns:
                        converted = visit(child)
                        if converted is None:
                            return None
                        children.append(converted)
                if isinstance(node, ConcatenationPattern):
                    return RegexConcatenation(tuple(children))
                return RegexAlternation(tuple(children))

            if isinstance(node, (QuantifiedPattern, ExcludedPattern, AnchorPattern)):
                if isinstance(node, AnchorPattern):
                    return None
                if (
                    location.kind == "attribute"
                    and node is location.owner
                    and location.attribute == "pattern"
                ):
                    child = replacement
                    replacements += 1
                else:
                    child = visit(node.pattern)
                    if child is None:
                        return None
                if isinstance(node, ExcludedPattern):
                    return child
                assert isinstance(node, QuantifiedPattern)
                return RegexRepetition(child, node.atLeast, node.atMost, node.isGreedy)
            return None

        return visit(root), replacements

    # ---------------------- phase 3: bounded proof search --------------------

    def _run_priority_optimizers(
        self,
        sources: Sequence[str],
        *,
        pass_limits: Optional[Sequence[int]] = None,
        deadline: Optional[float] = None,
        equivalence_mode: EquivalenceMode = EquivalenceMode.LANGUAGE_AND_MATCH,
    ) -> List[Optional[PriorityOptimizerOutcome]]:
        """Run proved rewrites to a bounded per-source fixed point."""

        config = PriorityOptimizerConfig(
            max_candidates=self.MAX_CANDIDATES,
            output_length_factor=self.MAX_OUTPUT_LENGTH_FACTOR,
            max_nfa_states=self.MAX_NFA_STATES,
            max_product_states=self.MAX_PRODUCT_STATES,
            equivalence_mode=equivalence_mode,
        )
        originals = list(sources)
        optimized = list(sources)
        proof_scopes = [ProofScope.COMPOSITIONAL] * len(sources)
        limits = list(pass_limits) if pass_limits is not None else [1] * len(sources)
        if len(limits) != len(sources):
            raise ValueError("pass_limits must have the same length as sources")
        if deadline is None:
            deadline = time.monotonic() + self.OPTIMIZER_TOTAL_TIMEOUT_SECONDS
        active = list(range(len(sources)))
        for _ in range(self.MAX_OPTIMIZER_PASSES):
            if not active:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            pass_results = PRIORITY_OPTIMIZER_SERVICE.optimize_many(
                [optimized[index] for index in active],
                config,
                min(self.OPTIMIZER_TIMEOUT_SECONDS, remaining),
            )
            next_active: List[int] = []
            for index, outcome in zip(active, pass_results):
                if (
                    outcome is None
                    or outcome.proof_scope
                    not in {ProofScope.COMPOSITIONAL, ProofScope.STANDALONE}
                    or outcome.pattern == optimized[index]
                ):
                    continue
                candidate = outcome.pattern
                max_length = min(
                    self.MAX_REGEX_LENGTH,
                    int(len(originals[index]) * self.MAX_OUTPUT_LENGTH_FACTOR),
                )
                if len(candidate) > max_length:
                    continue
                optimized[index] = candidate
                if outcome.proof_scope is ProofScope.STANDALONE:
                    proof_scopes[index] = ProofScope.STANDALONE
                limits[index] -= 1
                if limits[index] > 0:
                    next_active.append(index)
            active = next_active

        # Every accepted edge was proved priority-equivalent by the worker.
        # Equivalence is transitive, so re-verifying original -> final here
        # repeats the most expensive operation without adding safety.
        return [
            PriorityOptimizerOutcome(pattern, proof_scope)
            for pattern, proof_scope in zip(optimized, proof_scopes)
        ]

    # --------------------- phase 4: transactional commit ---------------------

    def _commit_priority_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan: _PriorityRewritePlan,
    ) -> bool:
        """Phase 4: decode a proved candidate and atomically update the AST."""

        # This is the final safety barrier.  A standalone proof is never an
        # authorization to mutate a fragment, even if a future discovery path
        # accidentally bypasses _find_general_plan's context certification.
        if plan.proof_scope not in {ProofScope.COMPOSITIONAL, ProofScope.WHOLE_PATTERN}:
            return False
        if (
            plan.proof_scope is ProofScope.WHOLE_PATTERN
            and not self._standalone_context_observation_safe(ast)
        ):
            return False

        old_state = _AstRewriteState.capture(ast)
        decoded = _PriorityPlanDecoder(self, ast, plan).decode()
        if decoded is None:
            return False

        old_names = set(plan.old_names)
        self._replace_at_location(ast, plan.location, decoded.replacement)
        self._replace_definitions(ast, old_names, decoded.conditions)
        self._replace_subsets(ast, old_names, decoded.new_names)
        self._replace_boundary_variables(ast, old_names, set(decoded.new_names))

        if not self._post_rewrite_invariants(ast):
            old_state.restore(ast)
            return False
        return True

    def _replace_boundary_variables(
        self,
        ast: RPRAbstractSyntaxTree,
        old_names: Set[str],
        new_names: Set[str],
    ) -> None:
        updated = []
        for boundary in getattr(ast, "pattern_rewrite_boundaries", ()):
            variables = set(boundary.variables)
            if not old_names & variables:
                updated.append(boundary)
                continue
            variables = (variables - old_names) | new_names
            elements = []
            for element in boundary.elements:
                element_variables = set(element)
                if old_names & element_variables:
                    element_variables = (element_variables - old_names) | new_names
                elements.append(frozenset(element_variables))
            updated.append(
                PatternRewriteBoundary(
                    boundary.kind,
                    frozenset(variables),
                    tuple(elements),
                )
            )
        ast.pattern_rewrite_boundaries = tuple(updated)

    def _replace_at_location(
        self,
        ast: RPRAbstractSyntaxTree,
        location: PatternLocation,
        replacement: PatternNode,
    ) -> None:
        if location.kind == "root":
            ast.pattern = replacement
            return
        if location.kind == "attribute":
            assert location.owner is not None and location.attribute is not None
            setattr(location.owner, location.attribute, replacement)
            return
        assert location.kind == "list" and location.patterns is not None
        inserted: Sequence[PatternNode]
        if location.splice_concatenation and isinstance(
            replacement, ConcatenationPattern
        ):
            inserted = replacement.patterns
        else:
            inserted = [replacement]
        location.patterns[location.start : location.end] = inserted

    # ------------------- shared predicate/alphabet analysis ------------------

    def _independent_expression(
        self, ast: RPRAbstractSyntaxTree, name: str
    ) -> Optional[Expr]:
        condition = ast.define.definitions.get(name)
        if condition is None:
            expression = _TRUE_EXPR()
        elif not isinstance(condition, IndependentCondition):
            return None
        else:
            expression = _strip_var_qualifier(condition.expression, name)
        if (
            expression is None
            or not self._is_safe_row_predicate(expression)
            or self._is_constant_false(expression)
        ):
            return None
        return expression

    def _independent_filter_expression(
        self, ast: RPRAbstractSyntaxTree, name: str
    ) -> Optional[Expr]:
        """Return a safe row-local condition necessary for a DEFINE to hold.

        Unlike ``_independent_expression``, this may extract the independent
        top-level AND conjuncts of a ``DependentCondition``.  Callers may use
        the result only as a necessary filter (for example, to prove predicate
        disjointness); it must never replace the complete DEFINE expression.
        """

        condition = ast.define.definitions.get(name)
        if condition is None:
            expression = _TRUE_EXPR()
        else:
            expression = condition.get_independent_filter()
            expression = (
                _strip_var_qualifier(expression, name)
                if expression is not None
                else None
            )
        if (
            expression is None
            or not self._is_safe_row_predicate(expression)
            or self._is_constant_false(expression)
        ):
            return None
        return expression

    def _build_truth_regions(
        self,
        ast: RPRAbstractSyntaxTree,
        component: Sequence[_SimplePattern],
    ) -> List[_TruthRegion]:
        expressions = [item.expression for item in component]
        finite = [self._extract_finite_predicate(expr) for expr in expressions]
        if (
            all(item is not None for item in finite)
            and len({item.column for item in finite if item is not None}) == 1
        ):
            return self._build_finite_regions(finite)  # type: ignore[arg-type]

        context = ast.expression_simplify_context()
        ranges = [self._extract_single_range(expr, context) for expr in expressions]
        if all(item is not None for item in ranges):
            range_items = [item for item in ranges if item is not None]
            if (
                len({item.column.to_string() for item in range_items}) == 1
                and len({item.domain for item in range_items}) == 1
            ):
                return self._build_range_regions(range_items)

        implications = [
            [
                self._predicate_implies(expressions[i], expressions[j], context)
                for j in range(len(expressions))
            ]
            for i in range(len(expressions))
        ]
        regions = []
        for signature in range(1, 1 << len(expressions)):
            if self._signature_violates_implication(signature, implications):
                continue
            expression = self._build_minterm(expressions, signature, context)
            regions.append(_TruthRegion(signature, expression))
        return regions

    def _build_finite_regions(
        self, predicates: Sequence[_FinitePredicate]
    ) -> List[_TruthRegion]:
        column = predicates[0].column
        regions = []
        for signature in range(1, 1 << len(predicates)):
            true_sets = [
                predicate.values
                for index, predicate in enumerate(predicates)
                if signature & (1 << index)
            ]
            values = set(true_sets[0])
            for value_set in true_sets[1:]:
                values.intersection_update(value_set)
            for index, predicate in enumerate(predicates):
                if not signature & (1 << index):
                    values.difference_update(predicate.values)
            if values:
                regions.append(
                    _TruthRegion(
                        signature,
                        self._finite_values_expr(column, values),
                    )
                )
        return regions

    def _build_range_regions(
        self, predicates: Sequence[RangeConstraint]
    ) -> List[_TruthRegion]:
        regions = []
        for signature in range(1, 1 << len(predicates)):
            true_ranges = [
                predicate
                for index, predicate in enumerate(predicates)
                if signature & (1 << index)
            ]
            pieces = [copy.deepcopy(true_ranges[0])]
            for predicate in true_ranges[1:]:
                intersections = []
                for piece in pieces:
                    intersection = piece.intersect(predicate)
                    if intersection is not None:
                        intersections.append(intersection)
                pieces = intersections
                if not pieces:
                    break
            if not pieces:
                continue

            for index, predicate in enumerate(predicates):
                if signature & (1 << index):
                    continue
                next_pieces = []
                for piece in pieces:
                    next_pieces.extend(self._subtract_range(piece, predicate))
                pieces = next_pieces
                if not pieces:
                    break
            if not pieces:
                continue

            expressions = [piece.to_expr() for piece in pieces]
            if any(expression is None for expression in expressions):
                continue
            regions.append(
                _TruthRegion(
                    signature,
                    self._or_expressions(
                        [expression for expression in expressions if expression]
                    ),
                )
            )
        return regions

    def _extract_finite_predicate(self, expression: Expr) -> Optional[_FinitePredicate]:
        if isinstance(expression, SemanticsExpr):
            return self._extract_finite_predicate(expression.expr)

        if isinstance(expression, InExpr) and not expression.is_not:
            if not isinstance(expression.expr, ColumnRef) or expression.expr.qualifier:
                return None
            if not all(isinstance(value, Literal) for value in expression.values):
                return None
            return _FinitePredicate(
                expression.expr.name,
                frozenset(value.value for value in expression.values),
            )

        if isinstance(expression, BinaryExpr) and expression.op == "=":
            column = None
            literal = None
            if isinstance(expression.left, ColumnRef) and isinstance(
                expression.right, Literal
            ):
                column, literal = expression.left, expression.right
            elif isinstance(expression.right, ColumnRef) and isinstance(
                expression.left, Literal
            ):
                column, literal = expression.right, expression.left
            if column is not None and column.qualifier is None:
                return _FinitePredicate(column.name, frozenset((literal.value,)))

        if isinstance(expression, BinaryExpr) and expression.op in ("AND", "OR"):
            left = self._extract_finite_predicate(expression.left)
            right = self._extract_finite_predicate(expression.right)
            if left is None or right is None or left.column != right.column:
                return None
            values = (
                left.values & right.values
                if expression.op == "AND"
                else left.values | right.values
            )
            return _FinitePredicate(left.column, frozenset(values))
        return None

    def _extract_single_range(
        self, expression: Expr, context
    ) -> Optional[RangeConstraint]:
        if isinstance(expression, SemanticsExpr):
            return self._extract_single_range(expression.expr, context)
        if isinstance(expression, BinaryExpr) and expression.op in ("AND", "OR"):
            left = self._extract_single_range(expression.left, context)
            right = self._extract_single_range(expression.right, context)
            if left is None or right is None:
                return None
            result = (
                left.intersect(right) if expression.op == "AND" else left.union(right)
            )
            return result
        try:
            result = _extract_range_constraint(expression, context)
        except (TypeError, ValueError):
            return None
        if result is None or result.domain is RangeDomain.TEXT:
            # Ordering of SQL text is collation-dependent.
            return None
        return result

    def _predicate_implies(self, left: Expr, right: Expr, context) -> bool:
        if left.to_string() == right.to_string() or self._is_constant_true(right):
            return True
        left_finite = self._extract_finite_predicate(left)
        right_finite = self._extract_finite_predicate(right)
        if (
            left_finite is not None
            and right_finite is not None
            and left_finite.column == right_finite.column
        ):
            return left_finite.values.issubset(right_finite.values)

        left_range = self._extract_single_range(left, context)
        right_range = self._extract_single_range(right, context)
        if left_range is None or right_range is None:
            return False
        intersection = left_range.intersect(right_range)
        return intersection is not None and self._same_range(intersection, left_range)

    def _predicates_proven_disjoint(
        self, ast: RPRAbstractSyntaxTree, left: Expr, right: Expr
    ) -> bool:
        """Return True only for an exact, collation-independent proof."""

        context = ast.expression_simplify_context()
        try:
            conjunction = _simplify_logical_expr(
                copy.deepcopy(left),
                "AND",
                copy.deepcopy(right),
                context,
                preserve_unknown=False,  # Only simultaneous TRUE matters here.
            )
        except (TypeError, ValueError):
            conjunction = None
        if conjunction is not None and self._is_constant_false(conjunction):
            return True

        left_finite = self._extract_finite_predicate(left)
        right_finite = self._extract_finite_predicate(right)
        if (
            left_finite is not None
            and right_finite is not None
            and left_finite.column == right_finite.column
        ):
            return not (left_finite.values & right_finite.values)

        left_range = self._extract_single_range(left, context)
        right_range = self._extract_single_range(right, context)
        if (
            left_range is None
            or right_range is None
            or not self._ranges_share_axis(left_range, right_range)
        ):
            return False
        return self._ranges_proven_disjoint(left_range, right_range)

    def _ranges_share_axis(self, left: RangeConstraint, right: RangeConstraint) -> bool:
        return (
            left.domain is right.domain
            and left.column is not None
            and right.column is not None
            and left.column.to_string() == right.column.to_string()
        )

    def _ranges_proven_disjoint(
        self, left: RangeConstraint, right: RangeConstraint
    ) -> bool:
        """Prove two comparable ranges are separated by a boundary."""

        if not self._ranges_share_axis(left, right):
            return False
        left = left._normalized_copy()
        right = right._normalized_copy()

        def before(first: RangeConstraint, second: RangeConstraint) -> bool:
            if first.upper is None or second.lower is None:
                return False
            first_value = (
                first.upper.value if isinstance(first.upper, Literal) else first.upper
            )
            second_value = (
                second.lower.value
                if isinstance(second.lower, Literal)
                else second.lower
            )
            compared = _compare_literal_values(first_value, second_value)
            if compared is None:
                return False
            return compared < 0 or (
                compared == 0 and not (first.upper_inclusive and second.lower_inclusive)
            )

        return before(left, right) or before(right, left)

    def _build_minterm(
        self, expressions: Sequence[Expr], signature: int, context
    ) -> Expr:
        result = _TRUE_EXPR()
        for index, expression in enumerate(expressions):
            term = (
                copy.deepcopy(expression)
                if signature & (1 << index)
                else _condition_is_not_true(copy.deepcopy(expression), context)
            )
            result = _simplify_logical_expr(result, "AND", term, context)
        return result

    def _signature_violates_implication(
        self, signature: int, implications: Sequence[Sequence[bool]]
    ) -> bool:
        for left in range(len(implications)):
            if not signature & (1 << left):
                continue
            for right in range(len(implications)):
                if implications[left][right] and not signature & (1 << right):
                    return True
        return False

    def _subtract_range(
        self, base: RangeConstraint, removed: RangeConstraint
    ) -> List[RangeConstraint]:
        intersection = base.intersect(removed)
        if intersection is None:
            return [base]

        result = []
        if intersection.lower is not None:
            left = RangeConstraint(
                base.domain,
                copy.deepcopy(base.column),
                base.lower,
                base.lower_inclusive,
                intersection.lower,
                not intersection.lower_inclusive,
            )
            if left.is_satisfiable():
                result.append(left)
        if intersection.upper is not None:
            right = RangeConstraint(
                base.domain,
                copy.deepcopy(base.column),
                intersection.upper,
                not intersection.upper_inclusive,
                base.upper,
                base.upper_inclusive,
            )
            if right.is_satisfiable():
                result.append(right)
        return result

    def _same_range(self, left: RangeConstraint, right: RangeConstraint) -> bool:
        left_normalized = left._normalized_copy()
        right_normalized = right._normalized_copy()
        return (
            left_normalized.domain is right_normalized.domain
            and left_normalized.column == right_normalized.column
            and left_normalized.lower == right_normalized.lower
            and left_normalized.lower_inclusive == right_normalized.lower_inclusive
            and left_normalized.upper == right_normalized.upper
            and left_normalized.upper_inclusive == right_normalized.upper_inclusive
        )

    # ------------------------- metadata replacement --------------------------

    def _replace_definitions(
        self,
        ast: RPRAbstractSyntaxTree,
        old_names: Set[str],
        new_conditions: Sequence[IndependentCondition],
    ) -> None:
        definitions = ast.define.definitions
        rewritten = {}
        inserted = False
        for name, condition in definitions.items():
            if name in old_names:
                if not inserted:
                    for new_condition in new_conditions:
                        rewritten[new_condition.var_name] = new_condition
                    inserted = True
                continue
            rewritten[name] = condition
        if not inserted:
            for new_condition in new_conditions:
                rewritten[new_condition.var_name] = new_condition
        ast.define.definitions = rewritten

    def _replace_subsets(
        self,
        ast: RPRAbstractSyntaxTree,
        old_names: Set[str],
        new_names: Sequence[str],
    ) -> None:
        if not ast.subsets:
            return
        for subset_name, members in ast.subsets.items():
            if not old_names.intersection(members):
                continue
            rewritten = []
            inserted = False
            for member in members:
                if member in old_names:
                    if not inserted:
                        rewritten.extend(new_names)
                        inserted = True
                    continue
                rewritten.append(member)
            ast.subsets[subset_name] = rewritten

    def _post_rewrite_invariants(self, ast: RPRAbstractSyntaxTree) -> bool:
        pattern_names = ast.pattern.get_variables()
        if not pattern_names:
            return False
        if any(name not in pattern_names for name in ast.define.definitions):
            return False
        if ast.subsets and any(
            member not in pattern_names
            for members in ast.subsets.values()
            for member in members
        ):
            return False
        return not ast.validate()

    # ------------------------------- safety gates -----------------------------

    def _label_is_directly_observable(
        self, ast: RPRAbstractSyntaxTree, name: str
    ) -> bool:
        if ast.measures is not None:
            for expression in ast.measures.measures.values():
                if _expr_references_any_var(expression, {name}):
                    return True

        for definition_name, condition in ast.define.definitions.items():
            if definition_name == name or condition is None:
                continue
            if _expr_references_any_var(condition.expression, {name}):
                return True

        skip = (ast.after_match_skip_to or "").split()
        if len(skip) == 1 and skip[0] == name:
            return True
        if len(skip) == 2 and skip[0] in {"FIRST", "LAST"} and skip[1] == name:
            return True
        return False

    def _subsets_are_preservable(
        self, ast: RPRAbstractSyntaxTree, names: Set[str]
    ) -> bool:
        for members in (ast.subsets or {}).values():
            overlap = names.intersection(members)
            if overlap and overlap != names:
                return False
        return True

    def _is_safe_row_predicate(self, expression: Expr) -> bool:
        if isinstance(expression, SemanticsExpr):
            return self._is_safe_row_predicate(expression.expr)
        if isinstance(expression, (Literal, ColumnRef)):
            return not isinstance(expression, ColumnRef) or expression.name != "*"
        if isinstance(expression, BinaryExpr):
            if expression.op in ("AND", "OR"):
                return self._is_safe_row_predicate(
                    expression.left
                ) and self._is_safe_row_predicate(expression.right)
            if expression.op not in {
                "=",
                "!=",
                "<>",
                ">",
                ">=",
                "<",
                "<=",
                "LIKE",
                "NOT LIKE",
            }:
                return False
            return isinstance(expression.left, (ColumnRef, Literal)) and isinstance(
                expression.right, (ColumnRef, Literal)
            )
        if isinstance(expression, BetweenExpr):
            return (
                isinstance(expression.expr, ColumnRef)
                and isinstance(expression.lower, Literal)
                and isinstance(expression.upper, Literal)
            )
        if isinstance(expression, InExpr):
            return isinstance(expression.expr, ColumnRef) and all(
                isinstance(value, Literal) for value in expression.values
            )
        if isinstance(expression, IsNullExpr):
            return isinstance(expression.expr, ColumnRef)
        if isinstance(expression, NotExpr):
            return self._is_safe_row_predicate(expression.expr)
        # FunctionCall, CASE, subquery, and unknown nodes may be volatile or
        # cannot be partitioned without duplicating evaluation.
        return False

    def _count_variable_occurrences(self, pattern: PatternNode) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        def visit(node: PatternNode) -> None:
            if isinstance(node, VariablePattern):
                counts[node.name] = counts.get(node.name, 0) + 1
                return
            if isinstance(node, (QuantifiedPattern, ExcludedPattern, AnchorPattern)):
                visit(node.pattern)
                return
            for child in getattr(node, "patterns", []):
                visit(child)

        visit(pattern)
        return counts

    def _finite_values_expr(self, column: str, values: Iterable[object]) -> Expr:
        ordered = sorted(values, key=lambda value: (type(value).__name__, repr(value)))
        if len(ordered) == 1:
            return BinaryExpr(ColumnRef(None, column), "=", Literal(ordered[0]))
        return InExpr(
            ColumnRef(None, column),
            [Literal(value) for value in ordered],
        )

    def _or_expressions(self, expressions: Sequence[Expr]) -> Expr:
        if not expressions:
            return BinaryExpr(Literal(1), "!=", Literal(1))
        result = copy.deepcopy(expressions[0])
        for expression in expressions[1:]:
            result = _simplify_logical_expr(result, "OR", copy.deepcopy(expression))
        return result

    def _is_constant_true(self, expression: Expr) -> bool:
        return (
            isinstance(expression, BinaryExpr)
            and expression.op == "="
            and isinstance(expression.left, Literal)
            and isinstance(expression.right, Literal)
            and expression.left.value == expression.right.value
        )

    def _is_constant_false(self, expression: Expr) -> bool:
        if not (
            isinstance(expression, BinaryExpr)
            and isinstance(expression.left, Literal)
            and isinstance(expression.right, Literal)
        ):
            return False
        compared = _compare_literal_values(
            expression.left.value, expression.right.value
        )
        if compared is None:
            return False
        return (
            (expression.op == "=" and compared != 0)
            or (expression.op in ("!=", "<>") and compared == 0)
            or (expression.op == ">" and compared <= 0)
            or (expression.op == ">=" and compared < 0)
            or (expression.op == "<" and compared >= 0)
            or (expression.op == "<=" and compared > 0)
        )


# ---------------------------------------------------------------------------
# Rule-level regression tests (kept here to match the repository convention).
# ---------------------------------------------------------------------------


def _all_rows(pattern: str, definitions: str) -> str:
    """Keep the successful-rewrite catalogue compact and easy to scan."""

    return f"""
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN({pattern})
    DEFINE {definitions}
)"""


def _one_row(pattern: str, definitions: str) -> str:
    """Build a valid summary-output query, including the implicit mode form."""
    # You do not need the `MEASURES` subclause when using `PARTITION BY`.
    return f"""
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY ID
    ORDER BY TS
    PATTERN({pattern})
    DEFINE {definitions}
)"""


regex_optimization_plus_test_cases = [
    (
        1,
        _all_rows(
            "A B+ C+ D", "A AS VAL = 0, B AS VAL = 1, C AS VAL IN (1, 2), D AS VAL = 4"
        ),
        _all_rows(
            "A B C+ D", "A AS VAL = 0, B AS VAL = 1, C AS VAL IN (1, 2), D AS VAL = 4"
        ),
    ),
    (
        2,
        _all_rows("(A+ | B+)+ C", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 2"),
        _all_rows("B+ C", "B AS VAL IN (1, 2), C AS VAL = 2"),
    ),
    (
        3,
        _all_rows(
            "S A B+ C+ D E",
            "S AS VAL = 0, A AS VAL = 9, B AS VAL = 1, C AS VAL IN (1, 2), D AS VAL = 4, E AS VAL = 5",
        ),
        _all_rows(
            "S A B C+ D E",
            "S AS VAL = 0, A AS VAL = 9, B AS VAL = 1, C AS VAL IN (1, 2), D AS VAL = 4, E AS VAL = 5",
        ),
    ),
    (
        4,
        _all_rows("A+ B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _all_rows("A B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
    ),
    (
        5,
        _all_rows("A+ B*", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _all_rows("A B*", "A AS VAL = 1, B AS VAL IN (1, 2)"),
    ),
    (
        6,
        _all_rows("B* C B*", "B AS VAL IN (1, 2), C AS VAL = 2"),
        _all_rows("V1* C B*", "V1 AS VAL = 1, C AS VAL = 2, B AS VAL IN (1, 2)"),
    ),
    (
        7,
        _all_rows("B* D*", "B AS VAL IN (1, 2), D AS VAL IN (1, 3)"),
        _all_rows(
            "B* (V1 D*)?", "B AS VAL IN (1, 2), V1 AS VAL = 3, D AS VAL IN (1, 3)"
        ),
    ),
    (
        8,
        _all_rows("B+ D*", "B AS VAL IN (1, 2), D AS VAL IN (1, 3)"),
        _all_rows(
            "B+ (V1 D*)?", "B AS VAL IN (1, 2), V1 AS VAL = 3, D AS VAL IN (1, 3)"
        ),
    ),
    (
        9,
        _all_rows("B+ C B+", "B AS VAL IN (1, 2), C AS VAL = 2"),
        _all_rows("B V1* C B+", "B AS VAL IN (1, 2), V1 AS VAL = 1, C AS VAL = 2"),
    ),
    (
        10,
        _all_rows("A* C? A*", "A AS VAL = 1, C AS VAL = 2"),
        _all_rows("A* (C A*)?", "A AS VAL = 1, C AS VAL = 2"),
    ),
    (
        11,
        _all_rows("A+ C? A*", "A AS VAL = 1, C AS VAL = 2"),
        _all_rows("A+ (C A*)?", "A AS VAL = 1, C AS VAL = 2"),
    ),
    (
        12,
        _all_rows("A* C? A+", "A AS VAL = 1, C AS VAL = 2"),
        _all_rows("A+ (C A+)? | C A+", "A AS VAL = 1, C AS VAL = 2"),
    ),
    (
        13,
        _all_rows("A* B*", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _all_rows("B*", "B AS VAL IN (1, 2)"),
    ),
    (
        14,
        _all_rows("A+ C* A*", "A AS VAL = 1, C AS VAL = 2"),
        _all_rows("A+ (C+ A*)?", "A AS VAL = 1, C AS VAL = 2"),
    ),
    (
        15,
        _all_rows("A* C* A*", "A AS VAL = 1, C AS VAL = 2"),
        _all_rows("A* (C+ A*)?", "A AS VAL = 1, C AS VAL = 2"),
    ),
    (
        16,
        _all_rows("A* B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _all_rows("B+", "B AS VAL IN (1, 2)"),
    ),
    (
        17,
        _all_rows("B+ A*", "B AS VAL IN (1, 2), A AS VAL = 1"),
        _all_rows("B+", "B AS VAL IN (1, 2)"),
    ),
    (
        18,
        _all_rows("(A B | A)+ C", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 2"),
        _all_rows("(A B?)+ C", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 2"),
    ),
    (
        19,
        _all_rows("B*? C B*", "B AS VAL IN (1, 2), C AS VAL = 2"),
        _all_rows("V1* C B*", "V1 AS VAL = 1, C AS VAL = 2, B AS VAL IN (1, 2)"),
    ),
    (
        20,
        _all_rows(
            "B*? A*? C", "B AS TYPE IN ('a', 'A'), A AS TYPE = 'a', C AS TYPE = 'A'"
        ),
        _all_rows(
            "B*? A* C", "B AS TYPE IN ('a', 'A'), A AS TYPE = 'a', C AS TYPE = 'A'"
        ),
    ),
    (
        21,
        _all_rows("A+ B+ C", "A AS VAL > 10, B AS TYPE = 'A', C AS 1 = 1"),
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS ((VAL IS NULL) OR VAL <= 10) AND TYPE = 'A', "
            "B AS TYPE = 'A', V2 AS VAL > 10 AND TYPE = 'A', C AS 1 = 1",
        ),
    ),
    # The controlled partial-overlap form is a fixed point.  In particular,
    # its generated mixed-column predicates must not re-enter generic minterm
    # partitioning on a later rule invocation.
    (
        22,
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS ((VAL IS NULL) OR VAL <= 10) AND TYPE = 'A', "
            "B AS TYPE = 'A', V2 AS VAL > 10 AND TYPE = 'A', C AS 1 = 1",
        ),
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS ((VAL IS NULL) OR VAL <= 10) AND TYPE = 'A', "
            "B AS TYPE = 'A', V2 AS VAL > 10 AND TYPE = 'A', C AS 1 = 1",
        ),
    ),
    (
        23,
        _all_rows("A+ B+ C", "A AS VAL > 10, B AS VAL < 20, C AS 1 = 1"),
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS VAL <= 10, B AS VAL < 20, V2 AS VAL > 10 AND VAL < 20, C AS 1 = 1",
        ),
    ),
    (
        24,
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS VAL <= 10, B AS VAL < 20, "
            "V2 AS VAL > 10 AND VAL < 20, C AS 1 = 1",
        ),
        _all_rows(
            "A+ (V1 B* | V2) C",
            "A AS VAL > 10, V1 AS VAL <= 10, B AS VAL < 20, "
            "V2 AS VAL > 10 AND VAL < 20, C AS 1 = 1",
        ),
    ),
    (
        25,
        _one_row("A+ B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _one_row("A B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
    ),
    (
        26,
        _one_row("A+ B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
        _one_row("A B+", "A AS VAL = 1, B AS VAL IN (1, 2)"),
    ),
    (
        27,
        _one_row("PERMUTE(A+ B+, C)", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
        _one_row("PERMUTE(A+ B+, C)", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
    ),
    # No optimization may combine variables on opposite sides of PERMUTE.
    (
        28,
        _one_row(
            "A+ PERMUTE(C, D) B+",
            "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3, D AS VAL = 4",
        ),
        _one_row(
            "A+ PERMUTE(C, D) B+",
            "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3, D AS VAL = 4",
        ),
    ),
    # EXCLUDE is likewise a boundary.  Its contents can be optimized without
    # changing whether matched rows are emitted in ALL ROWS mode.
    (
        29,
        _all_rows("{- A+ B+ -} C", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
        _all_rows("{- A B+ -} C", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
    ),
    # Under ONE ROW, exclusion has no projection effect, but it still must be
    # preserved by this rule and must prevent cross-boundary optimization.
    (
        30,
        _one_row("A+ {- C -} B+", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
        _one_row("A+ {- C -} B+", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
    ),
    # Exercise a nested alternation/repetition island under EXCLUDE as well as
    # the simple adjacent-plus form above.
    (
        31,
        _one_row(
            "{-((A+ | B+)+ C)-}", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 2"
        ),
        _one_row("{-B+ C-}", "B AS VAL IN (1, 2), C AS VAL = 2"),
    ),
    (
        32,
        _one_row(
            "(A+ B+ C | C A+ B+)", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"
        ),
        _one_row("(A B+ C | C A B+)", "A AS VAL = 1, B AS VAL IN (1, 2), C AS VAL = 3"),
    ),
    (
        33,
        _all_rows(
            "(A+ D B+ C | C A+ D B+)",
            "A AS VAL IN (1, 2), B AS VAL IN (1, 2), C AS VAL = 3, D AS VAL = 1",
        ),
        _all_rows(
            "(A V1* D A+ C | C A V1* D A+)",
            "A AS VAL IN (1, 2), V1 AS VAL = 2, D AS VAL = 1, C AS VAL = 3",
        ),
    ),
    (
        34,
        _all_rows(
            "(A+ B+ C | C A+ B+ | D A+ B+)",
            "A AS VAL IN (1, 2), B AS VAL IN (1, 2), C AS VAL = 3, D AS VAL = 3",
        ),
        _all_rows("(A{2,} C | C A{2,})", "A AS VAL IN (1, 2), C AS VAL = 3"),
    ),
    (
        35,
        _all_rows("A* B+ C*", "A AS 1 = 1, B AS 1 = 1, C AS 1 = 1"),
        _all_rows("A+", "A AS 1 = 1"),
    ),
    (
        36,
        _all_rows("A+ B+", "A AS VAL > 10, B AS B.VAL > A.VAL"),
        _all_rows("A+ B+", "A AS VAL > 10, B AS B.VAL > A.VAL"),
    ),
    (
        37,
        _all_rows("A+ B+ C", "A AS VAL > 10, B AS B.VAL > LAST(A.VAL), C AS VAL < 100"),
        _all_rows("A+ B+ C", "A AS VAL > 10, B AS B.VAL > LAST(A.VAL), C AS VAL < 100"),
    ),
    (
        38,
        _all_rows("A+ B+", "A AS VAL BETWEEN 10 AND 20, B AS VAL >= FIRST(A.VAL)"),
        _all_rows("A+ B+", "A AS VAL BETWEEN 10 AND 20, B AS VAL >= FIRST(A.VAL)"),
    ),
    (
        39,
        _all_rows("A B+ C", "A AS VAL = 1, B AS B.VAL > PREV(B.VAL), C AS VAL = 3"),
        _all_rows("A B+ C", "A AS VAL = 1, B AS B.VAL > PREV(B.VAL), C AS VAL = 3"),
    ),
    (
        40,
        _all_rows(
            "A+ B+ C", "A AS A.VAL < NEXT(A.VAL), B AS VAL IN (1, 2), C AS VAL = 3"
        ),
        _all_rows(
            "A+ B+ C", "A AS A.VAL < NEXT(A.VAL), B AS VAL IN (1, 2), C AS VAL = 3"
        ),
    ),
    (
        41,
        _all_rows("A+ B+ C", "A AS VAL > 0, B AS B.VAL > AVG(A.VAL), C AS VAL = 100"),
        _all_rows("A+ B+ C", "A AS VAL > 0, B AS B.VAL > AVG(A.VAL), C AS VAL = 100"),
    ),
    (
        42,
        _all_rows(
            "A+ B+ C",
            "A AS VAL = 1, B AS COUNT(A.*) >= 2 AND VAL IN (1, 2), C AS VAL = 3",
        ),
        _all_rows(
            "A+ B+ C",
            "A AS VAL = 1, B AS COUNT(A.*) >= 2 AND VAL IN (1, 2), C AS VAL = 3",
        ),
    ),
    (
        43,
        _all_rows(
            "A+ (B+ C | D+ C)",
            "A AS VAL > 10, B AS B.VAL > LAST(A.VAL), C AS VAL = 3, D AS D.VAL < FIRST(A.VAL)",
        ),
        _all_rows(
            "A+ (B+ C | D+ C)",
            "A AS VAL > 10, B AS B.VAL > LAST(A.VAL), C AS VAL = 3, D AS D.VAL < FIRST(A.VAL)",
        ),
    ),
    (
        44,
        _all_rows(
            "A+ B+ C | C A+ B+",
            "A AS VAL IN (1, 2), B AS B.VAL >= LAST(A.VAL), C AS VAL = 3",
        ),
        _all_rows(
            "A+ B+ C | C A+ B+",
            "A AS VAL IN (1, 2), B AS B.VAL >= LAST(A.VAL), C AS VAL = 3",
        ),
    ),
    (
        45,
        _all_rows("A+ B+ C", "A AS VAL > 0, B AS SUM(A.VAL) > B.VAL, C AS VAL = 3"),
        _all_rows("A+ B+ C", "A AS VAL > 0, B AS SUM(A.VAL) > B.VAL, C AS VAL = 3"),
    ),
    (
        46,
        _all_rows(
            "A+ B+ C",
            "A AS VAL = 1 OR VAL = 2, B AS B.VAL > PREV(A.VAL, 2), C AS VAL = 3",
        ),
        _all_rows(
            "A+ B+ C",
            "A AS VAL = 1 OR VAL = 2, B AS B.VAL > PREV(A.VAL, 2), C AS VAL = 3",
        ),
    ),
    (
        47,
        _all_rows(
            "A+ B+ C",
            "A AS VAL > 0, B AS CASE WHEN LAST(A.VAL) > 10 THEN B.VAL > 20 ELSE B.VAL > 5 END, C AS VAL = 3",
        ),
        _all_rows(
            "A+ B+ C",
            "A AS VAL > 0, B AS CASE WHEN LAST(A.VAL) > 10 THEN B.VAL > 20 ELSE B.VAL > 5 END, C AS VAL = 3",
        ),
    ),
    (
        48,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 0, B AS B.VAL > LAST(A.VAL), C AS C.VAL > LAST(B.VAL), D AS D.VAL > LAST(C.VAL)",
        ),
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 0, B AS B.VAL > LAST(A.VAL), C AS C.VAL > LAST(B.VAL), D AS D.VAL > LAST(C.VAL)",
        ),
    ),
    # A dependent left predicate may still be fixed at one row when its
    # non-row-local inputs come only from a completed prefix and its row-local
    # filter proves that every B row is also a C row.
    (
        49,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(A.VAL) AND TYPE = 'A', "
            "C AS TYPE IN ('A', 'C')",
        ),
        _all_rows(
            "A+ B C+ D",
            "A AS VAL > 10, B AS VAL < SUM(A.VAL) AND TYPE = 'A', "
            "C AS TYPE IN ('A', 'C')",
        ),
    ),
    (
        50,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < LAST(A.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
        _all_rows(
            "A+ B C+ D",
            "A AS VAL > 10, B AS VAL < LAST(A.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
    ),
    # SUM(B.VAL) changes while B grows, so the same containment shortcut is
    # not sound even though TYPE = 'A' is an independently extractable filter.
    (
        51,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(B.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(B.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
    ),
    # A later DEFINE can observe the changed B/C boundary, so retain B+.
    (
        52,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(A.VAL) AND TYPE = 'A', "
            "C AS TYPE IN ('A', 'C'), D AS VAL > LAST(B.VAL)",
        ),
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(A.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C'), D AS VAL > LAST(B.VAL)",
        ),
    ),
    # Prefix-stable dependent variables participate in the general regex
    # optimizer too; this is not limited to an adjacent top-level B+ C+ pair.
    (
        53,
        _all_rows(
            "A (B+ | C+)+ D",
            "A AS VAL > 10, B AS VAL < LAST(A.VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
        _all_rows("A C+ D", "A AS VAL > 10, C AS TYPE IN ('A', 'C')"),
    ),
    (
        54,
        _all_rows(
            "(A* | B | C*) C* (D? | B+?) E F",
            "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3, D AS VAL IN (1, 2), E AS VAL IN (1, 2), F AS VAL = 1",
        ),
        _all_rows(
            "(A* | B | C*) C* (D? | B+?) E F",
            "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3, D AS VAL IN (1, 2), E AS VAL IN (1, 2), F AS VAL = 1",
        ),
    ),
    # A mandatory disjoint successor forces a unique A/B boundary, so the
    # reluctant repetition can be normalized before general optimization.
    (
        55,
        _all_rows("A+? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
        _all_rows("A+ B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
    ),
    # A nullable successor does not force a boundary.  On an A-only input the
    # reluctant and greedy repetitions select different match lengths.
    (
        56,
        _all_rows("A+? B*", "A AS VAL = 1, B AS VAL = 1"),
        _all_rows("A+? B*", "A AS VAL = 1, B AS VAL = 1"),
    ),
    # Even a mandatory successor is unsafe when both variables can classify
    # the same row.
    (
        57,
        _all_rows("A+? B", "A AS VAL IN (1, 2), B AS VAL = 2"),
        _all_rows("A+? B", "A AS VAL IN (1, 2), B AS VAL = 2"),
    ),
    # Cover the remaining effective reluctant-quantifier spellings.  In each
    # query B is mandatory and predicate-disjoint from A, so the A/B boundary
    # is unique and changing A's repetition to greedy is semantics-preserving.
    (
        58,
        _all_rows("A*? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
        _all_rows("A* B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
    ),
    (
        59,
        _all_rows("A?? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
        _all_rows("A? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
    ),
    (
        60,
        _all_rows("A{1,3}? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
        _all_rows("A{1,3} B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
    ),
    (
        61,
        _all_rows("A{2,}? B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
        _all_rows("A{2,} B C", "A AS VAL = 1, B AS VAL = 2, C AS VAL = 3"),
    ),
    (
        62,
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
        _all_rows(
            "A+ B+ C+ D",
            "A AS VAL > 10, B AS VAL < SUM(VAL) AND TYPE = 'A', C AS TYPE IN ('A', 'C')",
        ),
    ),
    (
        63,
        _all_rows(
            "A+ B C{0,2} D+",
            "A AS VAL > 10, B AS VAL > 10, C AS VAL > 10, D AS VAL > 10",
        ),
        _all_rows("A{3,}", "A AS VAL > 10"),
    ),
    (
        64,
        _all_rows(
            "A N* B N*? C",
            "A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'",
        ),
        _all_rows(
            "A N* B N*? C",
            "A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'",
        ),
    ),
]


def run_tests(verbose: bool = True):
    from .test_utils import run_rule_tests

    schemas = {
        "T": {
            "columns": {"TS": "LONG", "TYPE": "VARCHAR(8)", "VAL": "DOUBLE"},
            "column_nullables": {"TS": False, "TYPE": True, "VAL": True},
        }
    }
    return run_rule_tests(
        RegexOptimizationPlusRule,
        regex_optimization_plus_test_cases,
        "regex optimization plus",
        verbose,
        default_table_schemas=schemas,
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} second")

    raise SystemExit(0 if result["failed"] == 0 else 1)
