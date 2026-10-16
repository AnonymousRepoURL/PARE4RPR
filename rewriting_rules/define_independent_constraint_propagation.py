if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

"""Propagate and consolidate numeric DEFINE constraints.

The fixed-point interval analysis emits row-local predicates which can be
consumed by existing pre-filtering rules.  A dependent conjunct is removed
only when those emitted intervals also prove the original conjunct TRUE, so
the consolidation preserves both directions of implication.  If the same
proof establishes that a complete dependent DEFINE condition is FALSE, the
condition is replaced with an explicit contradiction.  A later rule may then
turn an impossible mandatory pattern variable into an empty input relation.

A companion symbolic-order analysis retains direct cross-variable ``<`` and
``<=`` relationships which ordinary interval projection loses.  Pattern-NFA
guards admit facts only from required, completed predecessors; a transitive
closure containing a strict self-cycle proves the current DEFINE impossible.

The implementation follows the same phases as the algorithm description:
analyze constraints without mutation, turn proofs into per-variable rewrite
actions, and commit the complete plan to the AST in one step.

Note that Integer division is modeled as real-number division:
e.g., 5 / 2 evaluates to 2.5 rather than 2.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum
from itertools import permutations
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from expression import (
    BinaryExpr,
    BetweenExpr,
    CaseExpr,
    ColumnRef,
    Expr,
    FunctionCall,
    Literal,
    SemanticsExpr,
)
from expression_simplifier import RangeConstraint, RangeDomain, _simplify_logical_expr
from clause_parser import ExprParser
from patternnode import (
    AlternationPattern,
    AnchorPattern,
    ConcatenationPattern,
    EmptyPattern,
    ExcludedPattern,
    PermutationPattern,
    QuantifiedPattern,
    VariablePattern,
)

from .base import RewriteRule


@dataclass
class _Interval:
    domain: RangeDomain
    lower: Optional[Decimal] = None
    lower_inclusive: bool = True
    upper: Optional[Decimal] = None
    upper_inclusive: bool = True
    from_aggregate: bool = False

    def copy(self):
        return _Interval(
            self.domain,
            self.lower,
            self.lower_inclusive,
            self.upper,
            self.upper_inclusive,
            self.from_aggregate,
        )

    def normalized(self):
        result = self.copy()
        if result.domain is not RangeDomain.INTEGER:
            return result
        if result.lower is not None:
            result.lower = (
                result.lower.to_integral_value(rounding=ROUND_CEILING)
                if result.lower_inclusive
                else result.lower.to_integral_value(rounding=ROUND_FLOOR) + 1
            )
            result.lower_inclusive = True
        if result.upper is not None:
            result.upper = (
                result.upper.to_integral_value(rounding=ROUND_FLOOR)
                if result.upper_inclusive
                else result.upper.to_integral_value(rounding=ROUND_CEILING) - 1
            )
            result.upper_inclusive = True
        return result

    def intersect(self, other: "_Interval") -> "_Interval":
        result = self.copy()
        if other.lower is not None:
            if result.lower is None or other.lower > result.lower:
                result.lower, result.lower_inclusive = (
                    other.lower,
                    other.lower_inclusive,
                )
            elif other.lower == result.lower:
                result.lower_inclusive &= other.lower_inclusive
        if other.upper is not None:
            if result.upper is None or other.upper < result.upper:
                result.upper, result.upper_inclusive = (
                    other.upper,
                    other.upper_inclusive,
                )
            elif other.upper == result.upper:
                result.upper_inclusive &= other.upper_inclusive
        result.from_aggregate |= other.from_aggregate
        return result.normalized()

    def same_bounds(self, other: "_Interval") -> bool:
        a, b = self.normalized(), other.normalized()
        return (a.lower, a.lower_inclusive, a.upper, a.upper_inclusive) == (
            b.lower,
            b.lower_inclusive,
            b.upper,
            b.upper_inclusive,
        )


class _OriginKind(Enum):
    CURRENT_ROW = "CURRENT_ROW"
    PATTERN_VARIABLE = "PATTERN_VARIABLE"
    LOGICAL_FIRST = "LOGICAL_FIRST"
    LOGICAL_LAST = "LOGICAL_LAST"
    AGGREGATE = "AGGREGATE"


class _TruthValue(Enum):
    ALWAYS_TRUE = "ALWAYS_TRUE"
    ALWAYS_FALSE = "ALWAYS_FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class _ValueOrigin:
    kind: _OriginKind
    variable: Optional[str] = None
    column: Optional[str] = None
    offset: int = 0


@dataclass(frozen=True)
class _AbstractValue:
    """A range together with the context needed to interpret it safely.

    ``required_labels`` records expression dependencies rather than asserting
    that those labels occur on every pattern path. ``may_be_null`` describes
    the expression before an enclosing comparison is required to be TRUE.
    """

    interval: _Interval
    required_labels: FrozenSet[str] = frozenset()
    may_be_null: bool = False
    origin: Optional[_ValueOrigin] = None


@dataclass(frozen=True)
class _OrderTerm:
    """One symbolic value participating in a cross-variable order proof.

    A term for the variable owning a DEFINE condition denotes an arbitrary
    row classified by that variable.  Because every such row must satisfy the
    DEFINE condition, a fact learned for that term also holds for FIRST/LAST
    selections of the same variable once the variable is complete.
    """

    variable: str
    column: str


@dataclass(frozen=True)
class _OrderFact:
    """A necessary ``lower <= upper`` relation; ``strict`` means ``<``."""

    owner: str
    lower: _OrderTerm
    upper: _OrderTerm
    strict: bool


@dataclass(frozen=True)
class _OrderPath:
    """One derived order relation together with its source facts."""

    strict: bool
    facts: Tuple[_OrderFact, ...]


@dataclass(frozen=True)
class _StrictOrderCycleProof:
    """Witness for an impossible symbolic inequality cycle."""

    facts: Tuple[_OrderFact, ...]


class _OrderGraph:
    """Transitive closure over conjunctive ``<=`` and ``<`` facts."""

    def __init__(self, facts: Iterable[_OrderFact]):
        self.facts = tuple(
            sorted(
                facts,
                key=lambda fact: (
                    fact.owner,
                    fact.lower.variable,
                    fact.lower.column,
                    fact.upper.variable,
                    fact.upper.column,
                    fact.strict,
                ),
            )
        )

    def strict_cycle(self) -> Optional[_StrictOrderCycleProof]:
        reach: Dict[Tuple[_OrderTerm, _OrderTerm], _OrderPath] = {}
        nodes: Set[_OrderTerm] = set()
        for fact in self.facts:
            nodes.update((fact.lower, fact.upper))
            key = (fact.lower, fact.upper)
            candidate = _OrderPath(fact.strict, (fact,))
            current = reach.get(key)
            if current is None or (candidate.strict and not current.strict):
                reach[key] = candidate

        ordered_nodes = sorted(nodes, key=lambda node: (node.variable, node.column))
        for via in ordered_nodes:
            for source in ordered_nodes:
                left = reach.get((source, via))
                if left is None:
                    continue
                for target in ordered_nodes:
                    right = reach.get((via, target))
                    if right is None:
                        continue
                    key = (source, target)
                    candidate = _OrderPath(
                        left.strict or right.strict,
                        left.facts + right.facts,
                    )
                    current = reach.get(key)
                    if current is None or (candidate.strict and not current.strict):
                        reach[key] = candidate

        for node in ordered_nodes:
            cycle = reach.get((node, node))
            if cycle is not None and cycle.strict:
                return _StrictOrderCycleProof(cycle.facts)
        return None


def _decimal(value) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _literal(value: Decimal) -> Literal:
    return Literal(int(value) if value == value.to_integral_value() else value)


class _PatternNFA:
    """Small epsilon-NFA used to resolve physical PREV/NEXT labels."""

    def __init__(self, pattern):
        self._next_state = 0
        self.edges: List[Tuple[int, int, Optional[str]]] = []
        self.start, self.accept = self._build(pattern)
        self.forward: Dict[int, List[Tuple[int, Optional[str]]]] = {}
        self.reverse: Dict[int, List[Tuple[int, Optional[str]]]] = {}
        for source, target, label in self.edges:
            self.forward.setdefault(source, []).append((target, label))
            self.reverse.setdefault(target, []).append((source, label))

    def _state(self):
        value = self._next_state
        self._next_state += 1
        return value

    def _edge(self, source, target, label=None):
        self.edges.append((source, target, label))

    def _build(self, node):
        if isinstance(node, VariablePattern):
            start, end = self._state(), self._state()
            self._edge(start, end, node.name)
            return start, end
        if isinstance(node, EmptyPattern):
            start, end = self._state(), self._state()
            self._edge(start, end)
            return start, end
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._build(node.pattern)
        if isinstance(node, ConcatenationPattern):
            return self._sequence(node.patterns)
        if isinstance(node, AlternationPattern):
            return self._alternation(node.patterns)
        if isinstance(node, PermutationPattern):
            # The parser caps PERMUTE at seven elements.  More than 720 paths
            # is not useful for this local analysis, so fail conservatively.
            if len(node.patterns) > 6:
                raise ValueError("PERMUTE is too large for navigation analysis")
            return self._alternation(
                [
                    ConcatenationPattern(list(order))
                    for order in permutations(node.patterns)
                ]
            )
        if isinstance(node, QuantifiedPattern):
            start, cursor = self._state(), None
            cursor = start
            for _ in range(node.atLeast):
                child_start, child_end = self._build(node.pattern)
                self._edge(cursor, child_start)
                cursor = child_end
            end = self._state()
            if node.atMost is None:
                self._edge(cursor, end)
                child_start, child_end = self._build(node.pattern)
                self._edge(cursor, child_start)
                self._edge(child_end, cursor)
            else:
                for _ in range(node.atMost - node.atLeast):
                    self._edge(cursor, end)
                    child_start, child_end = self._build(node.pattern)
                    self._edge(cursor, child_start)
                    cursor = child_end
                self._edge(cursor, end)
            return start, end
        raise ValueError(f"unsupported pattern node: {type(node).__name__}")

    def _sequence(self, nodes):
        start = self._state()
        cursor = start
        for node in nodes:
            child_start, child_end = self._build(node)
            self._edge(cursor, child_start)
            cursor = child_end
        end = self._state()
        self._edge(cursor, end)
        return start, end

    def _alternation(self, nodes):
        start, end = self._state(), self._state()
        for node in nodes:
            child_start, child_end = self._build(node)
            self._edge(start, child_start)
            self._edge(child_end, end)
        return start, end

    def _epsilon_closure(self, states: Set[int], reverse=False) -> Set[int]:
        graph = self.reverse if reverse else self.forward
        result, todo = set(states), list(states)
        while todo:
            state = todo.pop()
            for target, label in graph.get(state, []):
                if label is None and target not in result:
                    result.add(target)
                    todo.append(target)
        return result

    def navigation_targets(
        self, origin_var: str, offset: int, reverse=False
    ) -> Optional[Set[str]]:
        """Return labels reachable by a physical offset from *origin_var*."""
        if offset == 0:
            return {origin_var}
        graph = self.reverse if reverse else self.forward
        occurrences = [
            (source if reverse else target)
            for source, target, label in self.edges
            if label == origin_var
        ]
        if not occurrences:
            return None
        all_labels: Set[str] = set()
        terminal = self.start if reverse else self.accept
        for occurrence in occurrences:
            states = self._epsilon_closure({occurrence}, reverse)
            labels: Set[str] = set()
            for step in range(offset):
                next_states: Set[int] = set()
                step_labels: Set[str] = set()
                for state in states:
                    for target, label in graph.get(state, []):
                        if label is not None:
                            next_states.add(target)
                            step_labels.add(label)
                if terminal in states or not next_states:
                    return None
                states = self._epsilon_closure(next_states, reverse)
                labels = step_labels
            if not labels:
                return None
            all_labels.update(labels)
        return all_labels or None

    def label_can_follow(self, current_var: str, candidate: str) -> bool:
        """Whether a candidate-labelled transition can occur after current."""
        starts = [
            target for source, target, label in self.edges if label == current_var
        ]
        seen, todo = set(starts), list(starts)
        while todo:
            state = todo.pop()
            for target, label in self.forward.get(state, []):
                if label == candidate:
                    return True
                if target not in seen:
                    seen.add(target)
                    todo.append(target)
        return False

    def label_requires(self, target: str, required: str) -> bool:
        """Whether every complete path containing *target* also has *required*.

        A DEFINE predicate is evaluated only when its pattern variable is
        classified.  Consequently a fact learned from DEFINE(required) may be
        attached to another variable only if that variable cannot occur on a
        complete pattern path which omits required.
        """
        if target == required:
            return True

        # Look for a counterexample: an accepting path which contains target
        # while avoiding every required-labelled transition.  The second state
        # component records whether target has occurred on the path.
        todo = [(self.start, False)]
        seen = set(todo)
        while todo:
            state, saw_target = todo.pop()
            if state == self.accept and saw_target:
                return False
            for next_state, label in self.forward.get(state, []):
                if label == required:
                    continue
                item = (next_state, saw_target or label == target)
                if item not in seen:
                    seen.add(item)
                    todo.append(item)
        return True


class _FalseProofKind(Enum):
    INTERVAL = "INTERVAL"
    STRICT_ORDER_CYCLE = "STRICT_ORDER_CYCLE"


@dataclass(frozen=True)
class _FalseProof:
    """Why one complete DEFINE condition can never evaluate to TRUE."""

    kind: _FalseProofKind
    order_cycle: Optional[_StrictOrderCycleProof] = None


@dataclass
class _ConstraintAnalysis:
    """Read-only result of the interval, order, and truth analyses."""

    intervals: Dict[Tuple[str, str], _Interval]
    nfa: _PatternNFA
    dependent_interval_change: bool
    removable_atom_ids: FrozenSet[int]
    false_proofs: Dict[str, _FalseProof]

    @property
    def always_false_variables(self) -> FrozenSet[str]:
        return frozenset(self.false_proofs)

    @property
    def has_rewrite_opportunity(self) -> bool:
        return bool(
            self.dependent_interval_change
            or self.removable_atom_ids
            or self.false_proofs
        )


class _RewriteAction(Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"
    REMOVE = "REMOVE"


@dataclass(frozen=True)
class _VariableRewrite:
    action: _RewriteAction
    expression: Optional[Expr] = None
    proof: Optional[_FalseProof] = None


@dataclass
class _PropagationPlan:
    """Immutable-in-practice rewrite decisions followed by one AST commit."""

    analysis: _ConstraintAnalysis
    rewrites: Dict[str, _VariableRewrite]
    additions: Dict[str, Expr]

    @property
    def intervals(self) -> Dict[Tuple[str, str], _Interval]:
        return self.analysis.intervals

    @property
    def nfa(self) -> _PatternNFA:
        return self.analysis.nfa

    @property
    def always_false_variables(self) -> FrozenSet[str]:
        return self.analysis.always_false_variables

    def commit(self, ast) -> bool:
        """Apply all planned DEFINE changes in one dictionary replacement."""

        parser = ExprParser(ast.expression_simplify_context())
        definitions = {}
        for var_name, condition in ast.define.definitions.items():
            rewrite = self.rewrites.get(var_name)
            if rewrite is None or rewrite.action is _RewriteAction.KEEP:
                definitions[var_name] = condition
                continue
            if rewrite.action is _RewriteAction.REMOVE:
                continue
            assert rewrite.expression is not None
            definitions[var_name] = parser._analyze_dependency(
                var_name,
                rewrite.expression,
            )

        for var_name, expression in self.additions.items():
            definitions[var_name] = parser._analyze_dependency(var_name, expression)

        ast.define.definitions = definitions
        return True


class DefineIndependentConstraintPropagationRule(RewriteRule):
    """Propagate numeric bounds/order facts and simplify DEFINE conditions."""

    AGGREGATES = {"SUM", "AVG", "MIN", "MAX", "COUNT"}

    @property
    def name(self) -> str:
        return "define_independent_constraint_propagation"

    def is_applicable(self, ast) -> bool:
        plan = self._build_plan(ast)
        self._pending_plan = (id(ast), plan)
        return plan is not None

    def apply(self, ast) -> bool:
        pending = getattr(self, "_pending_plan", None)
        plan = (
            pending[1] if pending and pending[0] == id(ast) else self._build_plan(ast)
        )
        self._pending_plan = None
        if plan is None:
            return False
        return plan.commit(ast)

    def _build_plan(self, ast):
        """Algorithm driver: analyze first, then build a mutation-free plan."""

        analysis = self._analyze_constraints(ast)
        if analysis is None or not analysis.has_rewrite_opportunity:
            return None
        return self._plan_rewrites(ast, analysis)

    def _analyze_constraints(self, ast) -> Optional[_ConstraintAnalysis]:
        """Run pattern, interval, order, and truth analyses without mutation."""

        if ast.define is None or ast.pattern is None:
            return None
        try:
            nfa = _PatternNFA(ast.pattern)
        except ValueError:
            return None

        intervals, dependent_change = self._solve_interval_fixed_point(ast, nfa)
        removable_atom_ids = self._find_removable_atom_ids(ast, nfa, intervals)
        false_proofs = self._interval_false_proofs(ast, nfa, intervals)
        false_proofs.update(self._relational_false_proofs(ast, nfa))
        return _ConstraintAnalysis(
            intervals,
            nfa,
            dependent_change,
            frozenset(removable_atom_ids),
            false_proofs,
        )

    def _solve_interval_fixed_point(self, ast, nfa):
        """Compute necessary row-local bounds until no interval changes."""

        intervals: Dict[Tuple[str, str], _Interval] = {}
        dependent_change = False

        # Revisit comparisons until bounds stop moving.  This handles chains,
        # including a later navigation-derived bound flowing back over '='.
        for _ in range(max(8, len(ast.define.definitions) * 8)):
            changed = False
            for current_var, condition in ast.define.definitions.items():
                if condition is None or condition.expression is None:
                    continue
                branch_ranges = self._branch_local_ranges(
                    ast,
                    current_var,
                    condition.expression,
                )
                for key, candidate in branch_ranges.items():
                    old = intervals.get(key, _Interval(candidate.domain))
                    new = old.intersect(candidate)
                    if not old.same_bounds(new):
                        intervals[key] = new
                        changed = True
                        dependent_change |= self._expr_is_dependent(
                            current_var,
                            condition.expression,
                        )
                for atom in self._constraint_atoms(condition.expression):
                    atom_changed, atom_dependent = self._apply_atom(
                        ast,
                        nfa,
                        current_var,
                        atom,
                        intervals,
                    )
                    changed |= atom_changed
                    dependent_change |= atom_changed and atom_dependent
            if not changed:
                break
        return intervals, dependent_change

    def _find_removable_atom_ids(self, ast, nfa, intervals) -> Set[int]:
        """Return dependent conjuncts proved TRUE by the emitted intervals."""

        removable: Set[int] = set()
        for current_var, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            for atom in self._flatten_and(condition.expression):
                if (
                    self._expr_is_dependent(current_var, atom)
                    and self._direct_range_atom(atom, current_var, ast) is None
                    and self._predicate_truth(
                        ast,
                        nfa,
                        current_var,
                        atom,
                        intervals,
                    )
                    is _TruthValue.ALWAYS_TRUE
                ):
                    removable.add(id(atom))
        return removable

    def _interval_false_proofs(self, ast, nfa, intervals) -> Dict[str, _FalseProof]:
        """Prove complete dependent conditions FALSE using interval facts."""

        return {
            current_var: _FalseProof(_FalseProofKind.INTERVAL)
            for current_var, condition in ast.define.definitions.items()
            if condition is not None
            and condition.expression is not None
            and self._expr_is_dependent(current_var, condition.expression)
            and self._predicate_truth(
                ast,
                nfa,
                current_var,
                condition.expression,
                intervals,
            )
            is _TruthValue.ALWAYS_FALSE
        }

    def _plan_rewrites(self, ast, analysis) -> _PropagationPlan:
        """Translate proof results into explicit per-variable AST actions."""

        rewrites: Dict[str, _VariableRewrite] = {}
        additions: Dict[str, Expr] = {}
        removed_names: List[str] = []

        for var_name, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                rewrites[var_name] = _VariableRewrite(_RewriteAction.KEEP)
                continue

            false_proof = analysis.false_proofs.get(var_name)
            if false_proof is not None:
                # An omitted DEFINE item means TRUE, so impossibility must be
                # represented explicitly.  A later mandatory-variable rule can
                # then turn the whole MATCH_RECOGNIZE input into an empty set.
                rewrites[var_name] = _VariableRewrite(
                    _RewriteAction.REPLACE,
                    BinaryExpr(Literal(1), "!=", Literal(1)),
                    false_proof,
                )
                continue

            emitted_columns = self._emitted_columns(analysis.intervals, var_name)
            retained = []
            for atom in self._flatten_and(condition.expression):
                direct_range = self._direct_range_atom(atom, var_name, ast)
                if direct_range is not None and direct_range[0] in emitted_columns:
                    continue
                if id(atom) in analysis.removable_atom_ids:
                    continue
                retained.append(atom)
            retained.extend(
                self._range_expressions(
                    analysis.intervals,
                    var_name,
                    emitted_columns,
                )
            )

            if not retained:
                rewrites[var_name] = _VariableRewrite(_RewriteAction.REMOVE)
                removed_names.append(var_name)
            else:
                rewrites[var_name] = _VariableRewrite(
                    _RewriteAction.REPLACE,
                    self._join_and(retained, ast),
                )

        remaining_names = set(ast.define.definitions) - set(removed_names)
        for var_name in sorted(ast.pattern.get_variables()):
            if var_name in remaining_names:
                continue
            expressions = self._range_expressions(
                analysis.intervals,
                var_name,
                self._emitted_columns(analysis.intervals, var_name),
            )
            if expressions:
                additions[var_name] = self._join_and(expressions, ast)
                remaining_names.add(var_name)

        # The local parser requires a DEFINE clause.  Preserve one tautology
        # if every original definition was proved TRUE and removed.
        if not remaining_names and not additions and removed_names:
            additions[removed_names[0]] = BinaryExpr(Literal(1), "=", Literal(1))

        return _PropagationPlan(analysis, rewrites, additions)

    @staticmethod
    def _emitted_columns(intervals, var_name) -> Set[str]:
        return {
            column
            for (variable, column), interval in intervals.items()
            if variable == var_name
            and (interval.lower is not None or interval.upper is not None)
        }

    @staticmethod
    def _range_expressions(intervals, var_name, columns) -> List[Expr]:
        expressions = []
        for column in sorted(columns):
            interval = intervals[(var_name, column)].normalized()
            expression = RangeConstraint(
                interval.domain,
                ColumnRef(None, column),
                _literal(interval.lower) if interval.lower is not None else None,
                interval.lower_inclusive,
                _literal(interval.upper) if interval.upper is not None else None,
                interval.upper_inclusive,
            ).to_expr()
            if expression is not None:
                expressions.append(expression)
        return expressions

    def _relational_false_proofs(self, ast, nfa) -> Dict[str, _FalseProof]:
        """Prove DEFINE conditions impossible using strict order cycles.

        Interval projection deliberately forgets correlations: from ``B > A``
        it can retain only a lower bound for B.  This companion analysis keeps
        direct column-to-column comparisons symbolic.  Facts from a preceding
        variable are activated only when the pattern NFA proves that variable
        is required and complete before the current variable.

        Each DEFINE condition constrains every row carrying its variable.  A
        fact such as ``B.X > A.X`` therefore also holds when a later condition
        reads the completed ``LAST(B.X)``.  References which might change after
        their owner are rejected while facts are collected, preventing FIRST/
        LAST occurrences from different pattern positions from being merged.
        """

        facts: Set[_OrderFact] = set()
        for owner, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            for atom in self._constraint_atoms(condition.expression):
                facts.update(self._comparison_order_facts(ast, nfa, owner, atom))

        if not facts:
            return {}

        proofs: Dict[str, _FalseProof] = {}
        for current_var, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue
            active_facts = [
                fact
                for fact in facts
                if fact.owner == current_var
                or (
                    nfa.label_requires(current_var, fact.owner)
                    and not nfa.label_can_follow(current_var, fact.owner)
                )
            ]
            cycle = _OrderGraph(active_facts).strict_cycle()
            if cycle is not None:
                proofs[current_var] = _FalseProof(
                    _FalseProofKind.STRICT_ORDER_CYCLE,
                    cycle,
                )
        return proofs

    def _comparison_order_facts(self, ast, nfa, owner, atom) -> Set[_OrderFact]:
        """Translate one necessary comparison into symbolic order facts.

        Keep this deliberately narrower than interval evaluation.  Arithmetic,
        navigation functions, aggregates, subsets, and unstable references
        fail closed; the existing interval analysis can still handle them.
        """

        if not isinstance(atom, BinaryExpr) or atom.op not in (
            "=",
            ">",
            ">=",
            "<",
            "<=",
        ):
            return set()
        left = self._order_term(ast, nfa, owner, atom.left)
        right = self._order_term(ast, nfa, owner, atom.right)
        if left is None or right is None:
            return set()

        if atom.op == "=":
            return {
                _OrderFact(owner, left, right, False),
                _OrderFact(owner, right, left, False),
            }
        if atom.op in ("<", "<="):
            return {_OrderFact(owner, left, right, atom.op == "<")}
        return {_OrderFact(owner, right, left, atom.op == ">")}

    def _order_term(self, ast, nfa, owner, expr) -> Optional[_OrderTerm]:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if not isinstance(expr, ColumnRef) or expr.name == "*":
            return None

        variable = expr.qualifier or owner
        if variable in (ast.subsets or {}):
            return None
        if variable not in ast.pattern.get_variables():
            return None
        if self._domain(ast, expr.name) is None:
            return None

        if variable != owner and (
            not nfa.label_requires(owner, variable)
            or nfa.label_can_follow(owner, variable)
        ):
            return None
        return _OrderTerm(variable, expr.name)

    def _predicate_truth(self, ast, nfa, current_var, expr, intervals):
        """Prove a DEFINE predicate TRUE/FALSE under emitted intervals.

        ``UNKNOWN`` includes both genuinely three-valued SQL results and
        facts this interval analysis cannot establish.  ``ALWAYS_TRUE``
        dependent conjuncts may be removed; a complete dependent condition
        proved ``ALWAYS_FALSE`` may be replaced by an explicit contradiction.
        """
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr

        if isinstance(expr, BinaryExpr) and expr.op in ("AND", "OR"):
            left = self._predicate_truth(ast, nfa, current_var, expr.left, intervals)
            right = self._predicate_truth(ast, nfa, current_var, expr.right, intervals)
            if expr.op == "AND":
                if _TruthValue.ALWAYS_FALSE in (left, right):
                    return _TruthValue.ALWAYS_FALSE
                if left is _TruthValue.ALWAYS_TRUE and right is _TruthValue.ALWAYS_TRUE:
                    return _TruthValue.ALWAYS_TRUE
                return _TruthValue.UNKNOWN
            if _TruthValue.ALWAYS_TRUE in (left, right):
                return _TruthValue.ALWAYS_TRUE
            if left is _TruthValue.ALWAYS_FALSE and right is _TruthValue.ALWAYS_FALSE:
                return _TruthValue.ALWAYS_FALSE
            return _TruthValue.UNKNOWN

        if isinstance(expr, BinaryExpr) and expr.op in (
            "=",
            "!=",
            "<>",
            ">",
            ">=",
            "<",
            "<=",
        ):
            return self._comparison_truth(
                ast,
                nfa,
                current_var,
                expr.left,
                expr.op,
                expr.right,
                intervals,
            )

        if isinstance(expr, BetweenExpr):
            lower_truth = self._comparison_truth(
                ast,
                nfa,
                current_var,
                expr.expr,
                ">=",
                expr.lower,
                intervals,
            )
            upper_truth = self._comparison_truth(
                ast,
                nfa,
                current_var,
                expr.expr,
                "<=",
                expr.upper,
                intervals,
            )
            if _TruthValue.ALWAYS_FALSE in (lower_truth, upper_truth):
                return _TruthValue.ALWAYS_FALSE
            if (
                lower_truth is _TruthValue.ALWAYS_TRUE
                and upper_truth is _TruthValue.ALWAYS_TRUE
            ):
                return _TruthValue.ALWAYS_TRUE
            return _TruthValue.UNKNOWN

        if isinstance(expr, CaseExpr):
            # A searched CASE chooses the first condition which is TRUE;
            # FALSE and UNKNOWN both continue to the next branch.
            possible_results = []
            can_fall_through = True
            for condition, result in expr.cases:
                condition_truth = self._predicate_truth(
                    ast,
                    nfa,
                    current_var,
                    condition,
                    intervals,
                )
                if condition_truth is _TruthValue.ALWAYS_FALSE:
                    continue
                possible_results.append(
                    self._predicate_truth(
                        ast,
                        nfa,
                        current_var,
                        result,
                        intervals,
                    )
                )
                if condition_truth is _TruthValue.ALWAYS_TRUE:
                    can_fall_through = False
                    break
            if can_fall_through:
                possible_results.append(
                    self._predicate_truth(
                        ast,
                        nfa,
                        current_var,
                        expr.else_expr,
                        intervals,
                    )
                    if expr.else_expr is not None
                    else _TruthValue.UNKNOWN
                )
            if possible_results and all(
                result is _TruthValue.ALWAYS_TRUE for result in possible_results
            ):
                return _TruthValue.ALWAYS_TRUE
            if possible_results and all(
                result is _TruthValue.ALWAYS_FALSE for result in possible_results
            ):
                return _TruthValue.ALWAYS_FALSE
        return _TruthValue.UNKNOWN

    def _comparison_truth(
        self,
        ast,
        nfa,
        current_var,
        left_expr,
        op,
        right_expr,
        intervals,
    ):
        left = self._eval_for_truth(ast, nfa, current_var, left_expr, intervals)
        right = self._eval_for_truth(ast, nfa, current_var, right_expr, intervals)
        if left is None or right is None:
            return _TruthValue.UNKNOWN
        if not self._value_is_non_null(nfa, current_var, left):
            return _TruthValue.UNKNOWN
        if not self._value_is_non_null(nfa, current_var, right):
            return _TruthValue.UNKNOWN
        return self._interval_comparison_truth(left.interval, op, right.interval)

    def _eval_for_truth(self, ast, nfa, current_var, expr, intervals):
        """Evaluate proof-only expressions without widening propagation.

        General inference deliberately treats an unqualified SUM
        conservatively.  The truth pass may evaluate it when every running
        member is known, because the result is used only for the reverse
        implication check which decides whether an old predicate is removable.
        """
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*", "/"):
            left = self._eval_for_truth(ast, nfa, current_var, expr.left, intervals)
            right = self._eval_for_truth(ast, nfa, current_var, expr.right, intervals)
            return self._abstract_arithmetic(left, expr.op, right)
        if (
            isinstance(expr, FunctionCall)
            and expr.name.upper() == "SUM"
            and len(expr.args) == 1
            and isinstance(expr.args[0], ColumnRef)
            and expr.args[0].qualifier is None
        ):
            return self._universal_sum(
                ast,
                nfa,
                current_var,
                expr.args[0],
                intervals,
            )
        return self._eval(ast, nfa, current_var, expr, intervals)

    def _value_is_non_null(self, nfa, current_var, value):
        if value.may_be_null:
            return False
        for label in value.required_labels:
            if label == current_var:
                continue
            if not nfa.label_requires(current_var, label):
                return False
            # An ordinary qualified reference and a running aggregate cannot
            # consume a label which only occurs later in the match.
            if nfa.label_can_follow(current_var, label):
                return False
        return True

    def _interval_comparison_truth(self, left, op, right):
        left, right = left.normalized(), right.normalized()
        if self._interval_is_empty(left) or self._interval_is_empty(right):
            return _TruthValue.UNKNOWN

        if op in (">", ">="):
            strict = op == ">"
            if self._all_greater(left, right, strict):
                return _TruthValue.ALWAYS_TRUE
            if self._never_greater(left, right, strict):
                return _TruthValue.ALWAYS_FALSE
            return _TruthValue.UNKNOWN
        if op in ("<", "<="):
            return self._interval_comparison_truth(
                right,
                ">" if op == "<" else ">=",
                left,
            )

        left_singleton = self._singleton_value(left)
        right_singleton = self._singleton_value(right)
        disjoint = self._intervals_disjoint(left, right)
        if op == "=":
            if (
                left_singleton is not None
                and right_singleton is not None
                and left_singleton == right_singleton
            ):
                return _TruthValue.ALWAYS_TRUE
            if disjoint:
                return _TruthValue.ALWAYS_FALSE
            return _TruthValue.UNKNOWN
        if op in ("!=", "<>"):
            if disjoint:
                return _TruthValue.ALWAYS_TRUE
            if (
                left_singleton is not None
                and right_singleton is not None
                and left_singleton == right_singleton
            ):
                return _TruthValue.ALWAYS_FALSE
        return _TruthValue.UNKNOWN

    def _all_greater(self, left, right, strict):
        if left.lower is None or right.upper is None:
            return False
        if left.lower > right.upper:
            return True
        if left.lower < right.upper:
            return False
        return not strict or not (left.lower_inclusive and right.upper_inclusive)

    def _never_greater(self, left, right, strict):
        if left.upper is None or right.lower is None:
            return False
        if left.upper < right.lower:
            return True
        if left.upper > right.lower:
            return False
        if strict:
            return True
        return not (left.upper_inclusive and right.lower_inclusive)

    def _intervals_disjoint(self, left, right):
        def entirely_before(first, second):
            if first.upper is None or second.lower is None:
                return False
            if first.upper < second.lower:
                return True
            return first.upper == second.lower and not (
                first.upper_inclusive and second.lower_inclusive
            )

        return entirely_before(left, right) or entirely_before(right, left)

    def _singleton_value(self, interval):
        if (
            interval.lower is not None
            and interval.lower == interval.upper
            and interval.lower_inclusive
            and interval.upper_inclusive
        ):
            return interval.lower
        return None

    def _interval_is_empty(self, interval):
        if interval.lower is None or interval.upper is None:
            return False
        if interval.lower > interval.upper:
            return True
        return interval.lower == interval.upper and not (
            interval.lower_inclusive and interval.upper_inclusive
        )

    def _branch_local_ranges(self, ast, current_var, expr):
        """Find row-local bounds that hold on every boolean branch."""
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if isinstance(expr, BinaryExpr) and expr.op in ("AND", "OR"):
            left = self._branch_local_ranges(ast, current_var, expr.left)
            right = self._branch_local_ranges(ast, current_var, expr.right)
            if expr.op == "AND":
                result = {key: value.copy() for key, value in left.items()}
                for key, value in right.items():
                    result[key] = (
                        result[key].intersect(value) if key in result else value.copy()
                    )
                return result
            result = {}
            for key in left.keys() & right.keys():
                result[key] = self._hull([left[key], right[key]])
            return result

        direct = self._direct_range_atom(expr, current_var, ast)
        if direct is None:
            return {}
        column, atom = direct
        domain = self._domain(ast, column)
        if domain is None:
            return {}
        interval = _Interval(domain)
        if isinstance(atom, BetweenExpr):
            lower, upper = _decimal(atom.lower.value), _decimal(atom.upper.value)
            if lower is None or upper is None:
                return {}
            interval.lower, interval.upper = lower, upper
        else:
            column_on_left = isinstance(atom.left, ColumnRef)
            literal_expr = atom.right if column_on_left else atom.left
            if not isinstance(literal_expr, Literal):
                return {}
            value = _decimal(literal_expr.value)
            if value is None:
                return {}
            op = (
                atom.op
                if column_on_left
                else {
                    ">": "<",
                    ">=": "<=",
                    "<": ">",
                    "<=": ">=",
                    "=": "=",
                }[atom.op]
            )
            if op in (">", ">=", "="):
                interval.lower = value
                interval.lower_inclusive = op != ">"
            if op in ("<", "<=", "="):
                interval.upper = value
                interval.upper_inclusive = op != "<"
        return {(current_var, column): interval.normalized()}

    def _constraint_atoms(self, expr: Expr) -> Iterable[Expr]:
        if isinstance(expr, SemanticsExpr):
            yield from self._constraint_atoms(expr.expr)
        elif isinstance(expr, BinaryExpr) and expr.op == "AND":
            yield from self._constraint_atoms(expr.left)
            yield from self._constraint_atoms(expr.right)
        elif isinstance(expr, BinaryExpr) and expr.op in ("=", ">", ">=", "<", "<="):
            yield expr
        elif isinstance(expr, BetweenExpr):
            yield expr

    def _apply_atom(self, ast, nfa, current_var, atom, intervals):
        if isinstance(atom, BetweenExpr):
            key = self._target_key(ast, current_var, atom.expr)
            lower = self._eval(ast, nfa, current_var, atom.lower, intervals)
            upper = self._eval(ast, nfa, current_var, atom.upper, intervals)
            changed = False
            if (
                key is not None
                and lower is not None
                and self._can_target(nfa, current_var, key)
            ):
                changed |= self._infer_target(intervals, key, ">=", lower, ast)
            if (
                key is not None
                and upper is not None
                and self._can_target(nfa, current_var, key)
            ):
                changed |= self._infer_target(intervals, key, "<=", upper, ast)
            if lower is not None:
                changed |= self._infer_expression_targets(
                    ast,
                    nfa,
                    current_var,
                    atom.expr,
                    ">=",
                    lower,
                    intervals,
                )
            if upper is not None:
                changed |= self._infer_expression_targets(
                    ast,
                    nfa,
                    current_var,
                    atom.expr,
                    "<=",
                    upper,
                    intervals,
                )
            return changed, (
                self._expr_is_dependent(current_var, atom.lower)
                or self._expr_is_dependent(current_var, atom.upper)
                or self._expr_is_dependent(current_var, atom.expr)
            )
        return self._apply_comparison(ast, nfa, current_var, atom, intervals)

    def _apply_comparison(self, ast, nfa, current_var, atom, intervals):
        left = self._eval(ast, nfa, current_var, atom.left, intervals)
        right = self._eval(ast, nfa, current_var, atom.right, intervals)
        left_key = self._target_key(ast, current_var, atom.left)
        right_key = self._target_key(ast, current_var, atom.right)
        dependent = self._expr_is_dependent(
            current_var, atom.left
        ) or self._expr_is_dependent(current_var, atom.right)
        changed = False
        if (
            left_key is not None
            and right is not None
            and self._can_target(nfa, current_var, left_key)
        ):
            changed |= self._infer_target(intervals, left_key, atom.op, right, ast)
        reverse_op = {">": "<", ">=": "<=", "<": ">", "<=": ">=", "=": "="}[atom.op]
        if (
            right_key is not None
            and left is not None
            and self._can_target(nfa, current_var, right_key)
        ):
            changed |= self._infer_target(intervals, right_key, reverse_op, left, ast)
        if right is not None:
            changed |= self._infer_expression_targets(
                ast,
                nfa,
                current_var,
                atom.left,
                atom.op,
                right,
                intervals,
            )
        if left is not None:
            changed |= self._infer_expression_targets(
                ast,
                nfa,
                current_var,
                atom.right,
                reverse_op,
                left,
                intervals,
            )
        return changed, dependent

    def _infer_expression_targets(
        self,
        ast,
        nfa,
        current_var,
        expr,
        op,
        bound,
        intervals,
    ):
        """Back-propagate a comparison through additive expressions.

        Non-target subexpressions are represented by their abstract intervals.
        This supports useful deductions such as ``SUM(A.X) + SUM(B.X) >= 10``
        without pretending that a multi-row SUM constrains each contributing
        row independently.
        """
        universal_sum_change = self._infer_universal_sum_targets(
            ast,
            nfa,
            current_var,
            expr,
            op,
            bound,
            intervals,
        )
        if universal_sum_change is not None:
            return universal_sum_change

        target_key = self._singleton_expression_target(
            ast,
            nfa,
            current_var,
            expr,
        )
        if target_key is not None:
            if not self._can_target(nfa, current_var, target_key):
                return False
            return self._infer_target(intervals, target_key, op, bound, ast)

        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if not isinstance(expr, BinaryExpr) or expr.op not in ("+", "-"):
            return False

        left_value = self._eval(ast, nfa, current_var, expr.left, intervals)
        right_value = self._eval(ast, nfa, current_var, expr.right, intervals)
        changed = False

        if right_value is not None:
            left_bound = self._abstract_arithmetic(
                bound,
                "-" if expr.op == "+" else "+",
                right_value,
            )
            if left_bound is not None:
                changed |= self._infer_expression_targets(
                    ast,
                    nfa,
                    current_var,
                    expr.left,
                    op,
                    left_bound,
                    intervals,
                )

        if left_value is not None:
            right_bound = self._abstract_arithmetic(
                bound if expr.op == "+" else left_value,
                "-",
                left_value if expr.op == "+" else bound,
            )
            if right_bound is not None:
                right_op = (
                    op
                    if expr.op == "+"
                    else {
                        ">": "<",
                        ">=": "<=",
                        "<": ">",
                        "<=": ">=",
                        "=": "=",
                    }[op]
                )
                changed |= self._infer_expression_targets(
                    ast,
                    nfa,
                    current_var,
                    expr.right,
                    right_op,
                    right_bound,
                    intervals,
                )
        return changed

    def _infer_universal_sum_targets(
        self,
        ast,
        nfa,
        current_var,
        expr,
        op,
        bound,
        intervals,
    ):
        """Isolate singleton contributions in a universal running SUM.

        Returns ``None`` when *expr* is not an unqualified SUM.  A boolean
        result means that the expression was recognized, even if its residual
        interval was too loose to derive a new bound.
        """
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if (
            not isinstance(expr, FunctionCall)
            or expr.name.upper() != "SUM"
            or len(expr.args) != 1
            or not isinstance(expr.args[0], ColumnRef)
            or expr.args[0].qualifier is not None
            or expr.args[0].name == "*"
        ):
            return None

        column = expr.args[0].name
        members = self._universal_running_members(
            ast,
            nfa,
            current_var,
        )
        if members is None:
            return False

        contributions = {}
        for member in members:
            interval = intervals.get((member, column))
            if interval is None:
                domain = self._domain(ast, column)
                if domain is None:
                    return False
                interval = _Interval(domain)
            lo_count, hi_count = self._aggregate_occurrence_range(
                ast.pattern,
                member,
                current_var,
            )
            contribution = self._sum_contribution(
                interval,
                lo_count,
                hi_count,
            )
            if contribution is None:
                return False
            contributions[member] = contribution

        changed = False
        for target in members:
            if self._occurrence_range(ast.pattern, target) != (1, 1):
                continue
            target_interval = intervals.get((target, column))
            target_non_null = ast.is_column_nullable(column) is False or (
                target_interval is not None
                and (
                    target_interval.lower is not None
                    or target_interval.upper is not None
                )
            )
            if not target_non_null:
                continue
            target_key = (target, column)
            if not self._can_target(nfa, current_var, target_key):
                continue

            residual = _Interval(
                RangeDomain.CONTINUOUS,
                Decimal(0),
                True,
                Decimal(0),
                True,
            )
            for member, contribution in contributions.items():
                if member != target:
                    residual = self._arithmetic(residual, "+", contribution)
            target_bound = self._abstract_arithmetic(
                bound,
                "-",
                _AbstractValue(
                    residual,
                    frozenset(member for member in members if member != target),
                ),
            )
            if target_bound is not None:
                changed |= self._infer_target(
                    intervals,
                    target_key,
                    op,
                    target_bound,
                    ast,
                )
        return changed

    def _universal_running_members(self, ast, nfa, current_var):
        """Return labels whose rows are complete in the current running SUM."""
        members = []
        for member in ast.pattern.get_variables():
            if member == current_var:
                members.append(member)
                continue
            can_precede = nfa.label_can_follow(member, current_var)
            can_follow = nfa.label_can_follow(current_var, member)
            if can_precede and can_follow:
                # The global occurrence range cannot distinguish how many of
                # this label are before versus after the current row.
                return None
            if can_precede:
                members.append(member)
        return members

    def _singleton_expression_target(self, ast, nfa, current_var, expr):
        """Resolve a row-local column or a complete singleton SUM target."""
        direct = self._target_key(ast, current_var, expr)
        if direct is not None:
            return direct
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if (
            not isinstance(expr, FunctionCall)
            or expr.name.upper() != "SUM"
            or len(expr.args) != 1
            or not isinstance(expr.args[0], ColumnRef)
        ):
            return None
        ref = expr.args[0]
        if ref.qualifier is None or ref.qualifier in (ast.subsets or {}):
            return None
        if self._occurrence_range(ast.pattern, ref.qualifier) != (1, 1):
            return None
        if not self._members_complete_before(
            ast,
            nfa,
            current_var,
            [ref.qualifier],
        ):
            return None
        return ref.qualifier, ref.name

    def _can_target(self, nfa, current_var, key):
        """Return whether a DEFINE fact is valid whenever its target occurs."""
        target_var, _ = key
        return nfa.label_requires(target_var, current_var)

    def _target_key(self, ast, current_var, expr):
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if not isinstance(expr, ColumnRef) or expr.name == "*":
            return None
        variable = expr.qualifier or current_var
        if variable in (ast.subsets or {}):
            return None
        if (
            variable != current_var
            and self._occurrence_range(ast.pattern, variable)[1] != 1
        ):
            # A reference denotes RUNNING LAST for a multi-row variable; it
            # cannot justify constraining every row classified by that label.
            return None
        return variable, expr.name

    def _infer_target(self, intervals, key, op, source: _AbstractValue, ast):
        source_interval = source.interval
        domain = self._domain(ast, key[1])
        if domain is None:
            return False
        candidate = _Interval(domain)
        if op in (">", ">=", "=") and source_interval.lower is not None:
            candidate.lower = source_interval.lower
            candidate.lower_inclusive = (
                source_interval.lower_inclusive if op in (">=", "=") else False
            )
        if op in ("<", "<=", "=") and source_interval.upper is not None:
            candidate.upper = source_interval.upper
            candidate.upper_inclusive = (
                source_interval.upper_inclusive if op in ("<=", "=") else False
            )
        candidate = candidate.normalized()
        if candidate.lower is None and candidate.upper is None:
            return False
        old = intervals.get(key, _Interval(domain))
        new = old.intersect(candidate)
        if old.same_bounds(new):
            return False
        intervals[key] = new
        return True

    def _eval(self, ast, nfa, current_var, expr, intervals) -> Optional[_AbstractValue]:
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if isinstance(expr, Literal):
            value = _decimal(expr.value)
            if value is None:
                return None
            return _AbstractValue(
                _Interval(RangeDomain.CONTINUOUS, value, True, value, True),
            )
        if isinstance(expr, ColumnRef):
            if expr.name == "*":
                return None
            qualifier = expr.qualifier or current_var
            if qualifier in (ast.subsets or {}):
                members = ast.subsets[qualifier]
                interval = self._hull(
                    [intervals.get((member, expr.name)) for member in members]
                )
                if interval is None:
                    return None
                return _AbstractValue(
                    interval,
                    frozenset(members),
                    ast.is_column_nullable(expr.name) is not False
                    and interval.lower is None
                    and interval.upper is None,
                )
            stored_interval = intervals.get((qualifier, expr.name))
            if stored_interval is None:
                interval = _Interval(
                    self._domain(ast, expr.name) or RangeDomain.CONTINUOUS,
                )
            else:
                interval = stored_interval.copy()
            origin_kind = (
                _OriginKind.CURRENT_ROW
                if expr.qualifier is None or qualifier == current_var
                else _OriginKind.PATTERN_VARIABLE
            )
            return _AbstractValue(
                interval,
                frozenset({qualifier}),
                ast.is_column_nullable(expr.name) is not False
                and (
                    stored_interval is None
                    or (stored_interval.lower is None and stored_interval.upper is None)
                ),
                _ValueOrigin(origin_kind, qualifier, expr.name),
            )
        if isinstance(expr, BinaryExpr) and expr.op in ("+", "-", "*", "/"):
            left = self._eval(ast, nfa, current_var, expr.left, intervals)
            right = self._eval(ast, nfa, current_var, expr.right, intervals)
            return self._abstract_arithmetic(left, expr.op, right)
        if isinstance(expr, CaseExpr):
            # A CASE result is in the hull of all of its result branches.  A
            # missing ELSE can produce NULL, so it cannot supply a range.
            if expr.else_expr is None:
                return None
            values = [
                self._eval(ast, nfa, current_var, result, intervals)
                for _, result in expr.cases
            ]
            values.append(self._eval(ast, nfa, current_var, expr.else_expr, intervals))
            return self._abstract_hull(values)
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name in ("FIRST", "LAST") and expr.args:
                referenced = self._qualified_reference(expr.args[0], current_var)
                members = []
                if referenced is not None:
                    members = list((ast.subsets or {}).get(referenced, [referenced]))
                    if (
                        sum(
                            self._occurrence_range(ast.pattern, member)[0]
                            for member in members
                        )
                        == 0
                    ):
                        return None
                value = self._eval(ast, nfa, current_var, expr.args[0], intervals)
                if value is None:
                    return None
                logical_offset = 0
                if len(expr.args) > 1:
                    logical_offset = self._literal_int(expr.args[1])
                    if logical_offset is None or logical_offset < 0:
                        return None
                guaranteed_available = (
                    logical_offset == 0
                    and self._logical_reference_available(
                        ast,
                        nfa,
                        current_var,
                        members,
                    )
                )
                return _AbstractValue(
                    value.interval,
                    value.required_labels | frozenset(members),
                    value.may_be_null or not guaranteed_available,
                    _ValueOrigin(
                        (
                            _OriginKind.LOGICAL_FIRST
                            if name == "FIRST"
                            else _OriginKind.LOGICAL_LAST
                        ),
                        referenced,
                        value.origin.column if value.origin is not None else None,
                        logical_offset,
                    ),
                )
            if name in ("PREV", "NEXT") and expr.args:
                return self._eval_physical_navigation(
                    ast,
                    nfa,
                    current_var,
                    name,
                    expr.args,
                    intervals,
                )
            if name in self.AGGREGATES:
                interval = self._aggregate(
                    ast,
                    nfa,
                    current_var,
                    name,
                    expr.args,
                    intervals,
                )
                if interval is None:
                    return None
                required = frozenset()
                if expr.args and isinstance(expr.args[0], ColumnRef):
                    required = frozenset(
                        self._reference_members(ast, current_var, expr.args[0]),
                    )
                return _AbstractValue(
                    interval,
                    required,
                    False,
                    _ValueOrigin(
                        _OriginKind.AGGREGATE,
                        (
                            expr.args[0].qualifier
                            if expr.args and isinstance(expr.args[0], ColumnRef)
                            else None
                        ),
                    ),
                )
        return None

    def _universal_sum(self, ast, nfa, current_var, ref, intervals):
        """Evaluate a running unqualified SUM over the rows seen so far.

        This intentionally handles only paths for which the NFA can identify
        a stable set of completed/current labels.  Ambiguous loops and missing
        value ranges fail conservatively.
        """
        members = self._universal_running_members(ast, nfa, current_var)
        if not members:
            return None
        if any(
            member != current_var
            and (
                not nfa.label_requires(current_var, member)
                or nfa.label_can_follow(current_var, member)
            )
            for member in members
        ):
            return None

        result = _Interval(
            RangeDomain.CONTINUOUS,
            Decimal(0),
            True,
            Decimal(0),
            True,
            True,
        )
        for member in members:
            interval = intervals.get((member, ref.name))
            if interval is None:
                return None
            # Every interval stored by the fixed point came from a
            # null-rejecting comparison.  Without one, a nullable contribution
            # would make the SUM cardinality/range uncertain.
            if (
                ast.is_column_nullable(ref.name) is not False
                and interval.lower is None
                and interval.upper is None
            ):
                return None
            min_count, max_count = self._aggregate_occurrence_range(
                ast.pattern,
                member,
                current_var,
            )
            contribution = self._sum_contribution(
                interval,
                min_count,
                max_count,
            )
            if contribution is None:
                return None
            result = self._arithmetic(result, "+", contribution)
            if result is None:
                return None
            result.from_aggregate = True

        return _AbstractValue(
            result,
            frozenset(members),
            False,
            _ValueOrigin(_OriginKind.AGGREGATE, None, ref.name),
        )

    def _aggregate(self, ast, nfa, current_var, name, args, intervals):
        if name == "COUNT" and (not args or isinstance(args[0], ColumnRef)):
            # COUNT(*) and COUNT(unqualified_column) range over the universal
            # running row set, not the variable whose DEFINE is being tested.
            # Computing that prefix requires path-sensitive NFA cardinalities;
            # fail conservatively until those are available.  A qualified star
            # such as COUNT(A.*), however, has an explicit row set.
            if not args or args[0].qualifier is None:
                return None
            members = self._reference_members(ast, current_var, args[0])
            if not members or not self._members_complete_before(
                ast, nfa, current_var, members
            ):
                return None
            minimum = maximum = 0
            for member in members:
                lo, hi = self._aggregate_occurrence_range(
                    ast.pattern, member, current_var
                )
                minimum += lo
                maximum = None if maximum is None or hi is None else maximum + hi
            if args and args[0].name != "*":
                column = args[0].name
                all_members_non_null = all(
                    (value := intervals.get((member, column))) is not None
                    and (value.lower is not None or value.upper is not None)
                    for member in members
                )
                if (
                    ast.is_column_nullable(column) is not False
                    and not all_members_non_null
                ):
                    minimum = 0
            return _Interval(
                RangeDomain.INTEGER,
                Decimal(minimum),
                True,
                Decimal(maximum) if maximum is not None else None,
                True,
                True,
            )
        if len(args) != 1 or not isinstance(args[0], ColumnRef):
            return None
        ref = args[0]
        # An unqualified aggregate argument uses the universal pattern
        # variable.  Treating it as current_var changes SUM/AVG/MIN/MAX and can
        # create constraints that reject valid matches.
        if ref.qualifier is None:
            return None
        members = self._reference_members(ast, current_var, ref)
        if not members or not self._members_complete_before(
            ast, nfa, current_var, members
        ):
            return None
        member_intervals = [intervals.get((member, ref.name)) for member in members]
        if name in ("MIN", "MAX", "AVG"):
            result = self._hull(member_intervals)
            if result is not None:
                result.from_aggregate = True
            return result
        if name != "SUM":
            return None
        result = _Interval(
            RangeDomain.CONTINUOUS, Decimal(0), True, Decimal(0), True, True
        )
        for member, interval in zip(members, member_intervals):
            if interval is None:
                return None
            lo_count, hi_count = self._aggregate_occurrence_range(
                ast.pattern, member, current_var
            )
            contribution = self._sum_contribution(interval, lo_count, hi_count)
            if contribution is None:
                return None
            result = self._arithmetic(result, "+", contribution)
            result.from_aggregate = True
        return result

    def _sum_contribution(self, interval, min_count, max_count):
        result = _Interval(RangeDomain.CONTINUOUS, from_aggregate=True)
        if interval.lower is not None:
            if interval.lower >= 0:
                result.lower = interval.lower * min_count
                result.lower_inclusive = (
                    True if min_count == 0 else interval.lower_inclusive
                )
            elif max_count is not None:
                result.lower = interval.lower * max_count
                result.lower_inclusive = (
                    True if max_count == 0 else interval.lower_inclusive
                )
        if interval.upper is not None:
            if interval.upper <= 0:
                result.upper = interval.upper * min_count
                result.upper_inclusive = (
                    True if min_count == 0 else interval.upper_inclusive
                )
            elif max_count is not None:
                result.upper = interval.upper * max_count
                result.upper_inclusive = (
                    True if max_count == 0 else interval.upper_inclusive
                )
        return result

    def _members_complete_before(self, ast, nfa, current_var, members):
        return (
            all(
                member == current_var or not nfa.label_can_follow(current_var, member)
                for member in members
            )
            and sum(
                self._aggregate_occurrence_range(ast.pattern, member, current_var)[0]
                for member in members
            )
            > 0
        )

    def _logical_reference_available(self, ast, nfa, current_var, members):
        """Whether FIRST/LAST has at least one referenced row at current."""
        if not members:
            return False
        if current_var in members:
            return True
        return (
            all(not nfa.label_can_follow(current_var, member) for member in members)
            and sum(
                self._occurrence_range(ast.pattern, member)[0] for member in members
            )
            > 0
        )

    def _aggregate_occurrence_range(self, pattern, member, current_var):
        minimum, maximum = self._occurrence_range(pattern, member)
        # DEFINE aggregates use RUNNING semantics.  At a row classified as
        # the current variable, that variable has occurred at least once, but
        # its future mandatory repetitions have not necessarily occurred yet.
        if member == current_var:
            minimum = 1
        return minimum, maximum

    def _reference_members(self, ast, current_var, ref):
        qualifier = ref.qualifier or current_var
        return list((ast.subsets or {}).get(qualifier, [qualifier]))

    def _arithmetic(self, left, op, right):
        if left is None or right is None:
            return None
        aggregate = left.from_aggregate or right.from_aggregate
        result = _Interval(RangeDomain.CONTINUOUS, from_aggregate=aggregate)
        if op in ("+", "-"):
            right_lower, right_upper = right.lower, right.upper
            right_lower_inc, right_upper_inc = (
                right.lower_inclusive,
                right.upper_inclusive,
            )
            if op == "-":
                right_lower, right_upper = (
                    -right_upper if right_upper is not None else None,
                    -right_lower if right_lower is not None else None,
                )
                right_lower_inc, right_upper_inc = (
                    right.upper_inclusive,
                    right.lower_inclusive,
                )
            if left.lower is not None and right_lower is not None:
                result.lower = left.lower + right_lower
                result.lower_inclusive = left.lower_inclusive and right_lower_inc
            if left.upper is not None and right_upper is not None:
                result.upper = left.upper + right_upper
                result.upper_inclusive = left.upper_inclusive and right_upper_inc
            return result
        # General products need four finite endpoints.  A singleton constant
        # is common in DEFINE and remains useful with a one-sided input range.
        constant = None
        varying = None
        if left.lower == left.upper and left.lower is not None:
            constant, varying = left.lower, right
        elif right.lower == right.upper and right.lower is not None:
            constant, varying = right.lower, left
        if (
            op == "/"
            and right.lower == right.upper
            and right.lower not in (None, Decimal(0))
        ):
            constant, varying = Decimal(1) / right.lower, left
        elif op == "/" or constant is None:
            return None
        if constant == 0:
            # Multiplication by zero is the closed singleton {0}, even when the
            # varying interval has an open or missing endpoint.
            return _Interval(
                RangeDomain.CONTINUOUS,
                Decimal(0),
                True,
                Decimal(0),
                True,
                aggregate,
            )
        if constant >= 0:
            result.lower = (
                varying.lower * constant if varying.lower is not None else None
            )
            result.upper = (
                varying.upper * constant if varying.upper is not None else None
            )
            result.lower_inclusive, result.upper_inclusive = (
                varying.lower_inclusive,
                varying.upper_inclusive,
            )
        else:
            result.lower = (
                varying.upper * constant if varying.upper is not None else None
            )
            result.upper = (
                varying.lower * constant if varying.lower is not None else None
            )
            result.lower_inclusive, result.upper_inclusive = (
                varying.upper_inclusive,
                varying.lower_inclusive,
            )
        return result

    def _abstract_arithmetic(self, left, op, right):
        if left is None or right is None:
            return None
        interval = self._arithmetic(left.interval, op, right.interval)
        if interval is None:
            return None
        return _AbstractValue(
            interval,
            left.required_labels | right.required_labels,
            left.may_be_null or right.may_be_null,
        )

    def _hull(self, values):
        if not values or any(value is None for value in values):
            return None
        domains = {value.domain for value in values}
        if RangeDomain.TEXT in domains:
            return None
        # SQL resolves all CASE result expressions to a common numeric type.
        # Keeping the first branch's INTEGER domain when another branch is
        # continuous would later round the continuous endpoint inward during
        # truth proving (for example, [1, 1.5] would become [1, 1]).  Promote
        # mixed numeric hulls so that they remain an over-approximation.
        domain = (
            RangeDomain.CONTINUOUS
            if RangeDomain.CONTINUOUS in domains
            else RangeDomain.INTEGER
        )
        result = _Interval(domain)
        lowers = [value.lower for value in values]
        uppers = [value.upper for value in values]
        if all(value is not None for value in lowers):
            result.lower = min(lowers)
            result.lower_inclusive = any(
                value.lower == result.lower and value.lower_inclusive
                for value in values
            )
        if all(value is not None for value in uppers):
            result.upper = max(uppers)
            result.upper_inclusive = any(
                value.upper == result.upper and value.upper_inclusive
                for value in values
            )
        result.from_aggregate = any(value.from_aggregate for value in values)
        return result

    def _abstract_hull(self, values):
        if not values or any(value is None for value in values):
            return None
        interval = self._hull([value.interval for value in values])
        if interval is None:
            return None
        required_labels = frozenset().union(
            *(value.required_labels for value in values)
        )
        return _AbstractValue(
            interval,
            required_labels,
            any(value.may_be_null for value in values),
        )

    def _occurrence_range(self, node, variable):
        if isinstance(node, VariablePattern):
            return (1, 1) if node.name == variable else (0, 0)
        if isinstance(node, EmptyPattern):
            return 0, 0
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._occurrence_range(node.pattern, variable)
        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            ranges = [
                self._occurrence_range(child, variable) for child in node.patterns
            ]
            minimum = sum(value[0] for value in ranges)
            maximum = (
                None
                if any(value[1] is None for value in ranges)
                else sum(value[1] for value in ranges)
            )
            return minimum, maximum
        if isinstance(node, AlternationPattern):
            ranges = [
                self._occurrence_range(child, variable) for child in node.patterns
            ]
            minimum = min(value[0] for value in ranges)
            maximum = (
                None
                if any(value[1] is None for value in ranges)
                else max(value[1] for value in ranges)
            )
            return minimum, maximum
        if isinstance(node, QuantifiedPattern):
            minimum, maximum = self._occurrence_range(node.pattern, variable)
            return (
                minimum * node.atLeast,
                (
                    None
                    if maximum is None or node.atMost is None
                    else maximum * node.atMost
                ),
            )
        return 0, None

    def _domain(self, ast, column):
        data_type = ast.get_column_type(column)
        if data_type is None:
            return None
        domain = RangeDomain.from_data_type(data_type)
        return None if domain is RangeDomain.TEXT else domain

    def _expr_is_dependent(self, current_var, expr):
        if isinstance(expr, SemanticsExpr):
            return self._expr_is_dependent(current_var, expr.expr)
        if isinstance(expr, ColumnRef):
            return expr.qualifier is not None and expr.qualifier != current_var
        if isinstance(expr, FunctionCall):
            return expr.name.upper() in self.AGGREGATES | {
                "PREV",
                "NEXT",
                "FIRST",
                "LAST",
            } or any(self._expr_is_dependent(current_var, arg) for arg in expr.args)
        if isinstance(expr, BinaryExpr):
            return self._expr_is_dependent(
                current_var, expr.left
            ) or self._expr_is_dependent(current_var, expr.right)
        if isinstance(expr, BetweenExpr):
            return (
                self._expr_is_dependent(current_var, expr.expr)
                or self._expr_is_dependent(current_var, expr.lower)
                or self._expr_is_dependent(current_var, expr.upper)
            )
        if isinstance(expr, CaseExpr):
            return any(
                self._expr_is_dependent(current_var, condition)
                or self._expr_is_dependent(current_var, result)
                for condition, result in expr.cases
            ) or (
                expr.else_expr is not None
                and self._expr_is_dependent(current_var, expr.else_expr)
            )
        return False

    def _direct_range_atom(self, atom, current_var, ast):
        while isinstance(atom, SemanticsExpr):
            atom = atom.expr
        if isinstance(atom, BetweenExpr):
            key = self._target_key(ast, current_var, atom.expr)
            if (
                key == (current_var, getattr(atom.expr, "name", None))
                and isinstance(atom.lower, Literal)
                and isinstance(atom.upper, Literal)
            ):
                return key[1], atom
            return None
        if not isinstance(atom, BinaryExpr) or atom.op not in (
            "=",
            ">",
            ">=",
            "<",
            "<=",
        ):
            return None
        left_key = self._target_key(ast, current_var, atom.left)
        right_key = self._target_key(ast, current_var, atom.right)
        if left_key and left_key[0] == current_var and isinstance(atom.right, Literal):
            return left_key[1], atom
        if right_key and right_key[0] == current_var and isinstance(atom.left, Literal):
            return right_key[1], atom
        return None

    def _flatten_and(self, expr):
        if isinstance(expr, BinaryExpr) and expr.op == "AND":
            return self._flatten_and(expr.left) + self._flatten_and(expr.right)
        return [expr]

    def _join_and(self, exprs, ast):
        if not exprs:
            return None
        result = exprs[0]
        for expr in exprs[1:]:
            result = _simplify_logical_expr(
                result, "AND", expr, ast.expression_simplify_context()
            )
        return result

    def _literal_int(self, expr):
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if not isinstance(expr, Literal):
            return None
        value = _decimal(expr.value)
        return (
            int(value)
            if value is not None and value == value.to_integral_value()
            else None
        )

    def _eval_physical_navigation(self, ast, nfa, current_var, name, args, intervals):
        """Evaluate only physical navigation whose origin is current_var.

        An unqualified argument uses the universal row pattern variable and an
        explicitly self-qualified argument starts at the current row.  A
        different qualifier, or FIRST/LAST compound navigation, selects a
        different origin which this NFA analysis does not model yet.
        """
        operand = self._eval(ast, nfa, current_var, args[0], intervals)
        if (
            operand is None
            or operand.origin is None
            or operand.origin.kind is not _OriginKind.CURRENT_ROW
            or operand.origin.column is None
        ):
            return None
        column = operand.origin.column
        offset = 1
        if len(args) > 1:
            offset_value = self._literal_int(args[1])
            if offset_value is None or offset_value < 0:
                return None
            offset = offset_value
        targets = nfa.navigation_targets(
            current_var,
            offset,
            name == "PREV",
        )
        if not targets:
            return None
        interval = self._hull([intervals.get((target, column)) for target in targets])
        if interval is None:
            return None
        return _AbstractValue(
            interval,
            operand.required_labels | frozenset(targets),
            operand.may_be_null,
            _ValueOrigin(
                _OriginKind.CURRENT_ROW,
                current_var,
                column,
                -offset if name == "PREV" else offset,
            ),
        )

    def _qualified_reference(self, expr, current_var):
        while isinstance(expr, SemanticsExpr):
            expr = expr.expr
        if isinstance(expr, ColumnRef):
            return expr.qualifier or current_var
        if isinstance(expr, FunctionCall) and expr.args:
            return self._qualified_reference(expr.args[0], current_var)
        return None


define_independent_constraint_propagation_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A N{3,20} B)
    DEFINE A AS VAL > 10, B AS VAL > A.VAL * 1.05
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A N{3,20} B)
    DEFINE A AS VAL >= 11, B AS VAL > A.VAL * 1.05 AND VAL >= 12
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B N{3,20} C D)
    DEFINE A AS VAL > 10, B AS VAL > A.VAL * 1.05, C AS VAL = B.VAL AND VAL < NEXT(C.VAL), D AS VAL < 20
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B N{3,20} C D)
    DEFINE A AS VAL >= 11, B AS VAL > A.VAL * 1.05 AND VAL BETWEEN 12 AND 18,
        C AS VAL = B.VAL AND VAL < NEXT(C.VAL) AND VAL BETWEEN 12 AND 18, D AS VAL <= 19
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B N{3,20} C)
    SUBSET U = (A, B)
    DEFINE A AS VAL > 10, B AS VAL > A.VAL * 2, C AS VAL > SUM(U.VAL)
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS ALL ROWS PER MATCH PATTERN(A B N{3,20} C)
    SUBSET U = (A, B)
    DEFINE A AS VAL >= 11, B AS VAL > A.VAL * 2 AND VAL >= 23, C AS VAL > SUM(U.VAL) AND VAL >= 35
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(C (D | E))
    DEFINE C AS VAL < NEXT(C.VAL), D AS VAL < 20, E AS VAL < 30
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(C (D | E))
    DEFINE C AS VAL < NEXT(C.VAL) AND VAL <= 28, D AS VAL <= 19, E AS VAL <= 29
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(C (D | E))
    DEFINE C AS VAL < NEXT(C.VAL), D AS VAL < 20, E AS VAL > 30
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(C (D | E))
    DEFINE C AS VAL < NEXT(C.VAL), D AS VAL < 20, E AS VAL > 30
)""",
    ),
    # Logical navigation obtains the range of the referenced variable.
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS VAL >= 5, B AS VAL > LAST(A.VAL)
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS VAL >= 5, B AS VAL > LAST(A.VAL) AND VAL >= 6
)""",
    ),
]


integer_table_schemas = {
    "T": {
        "columns": ["TS", "VAL"],
        "column_types": {"TS": "LONG", "VAL": "INT"},
        "column_nullables": {"TS": False, "VAL": False},
    }
}


continuous_table_schemas = {
    "TF": {
        "columns": ["TS", "VAL"],
        "column_types": {"TS": "LONG", "VAL": "DOUBLE"},
        "column_nullables": {"TS": False, "VAL": False},
    },
    "TM": {
        "columns": ["TS", "I"],
        "column_types": {"TS": "LONG", "I": "INT"},
        "column_nullables": {"TS": False, "I": False},
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    results = [
        run_rule_tests(
            DefineIndependentConstraintPropagationRule,
            define_independent_constraint_propagation_test_cases,
            "DEFINE independent constraint propagation (integer)",
            verbose,
            integer_table_schemas,
        ),
    ]
    return {
        "label": "DEFINE independent constraint propagation",
        "total": sum(result["total"] for result in results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(result["failed"] for result in results),
        "failures": [failure for result in results for failure in result["failures"]],
    }


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
