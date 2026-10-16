if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
import re


from expression_clause import (
    DependentCondition,
    IndependentCondition,
    collect_define_dependency_reasons,
)
from expression import (
    BinaryExpr,
    BetweenExpr,
    CaseExpr,
    ColumnRef,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    SemanticsExpr,
)
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
from queryanalysis import ALL_ROWS_PER_MATCH, ONE_ROW_PER_MATCH, RPRAbstractSyntaxTree

from .base import RewriteRule


class RedundantSubsetVariableRemovalRule(RewriteRule):
    """Remove SUBSET variables whose union semantics are unnecessary.

    Singleton subsets are aliases and are replaced everywhere in DEFINE and
    MEASURES.  For a multi-member subset, a scalar reference is replaced only
    when every possible pattern path gives it the same source variable:

    * DEFINE uses the RUNNING prefix ending at the variable being defined.
    * ONE ROW PER MATCH MEASURES uses the complete match.
    * ALL ROWS PER MATCH MEASURES remains running and is therefore left alone
      (unless it is explicitly wrapped in FINAL semantics).

    After rewriting, an unreferenced SUBSET declaration is deleted.  A subset
    used as an AFTER MATCH SKIP target is retained because that use is not an
    expression qualifier and cannot safely be discarded.
    """

    _PROTECTED_FUNCTIONS = {
        "AVG",
        "SUM",
        "COUNT",
        "MAX",
        "MIN",
        "PREV",
        "NEXT",
    }

    @property
    def name(self) -> str:
        return "redundant_subset_variable_removal"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return bool(ast.subsets) and self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        if plan is None:
            return False

        measures, definitions, subsets, skip_target = plan
        if ast.measures is not None:
            ast.measures.measures = measures
        if ast.define is not None:
            ast.define.definitions = definitions
        ast.subsets = subsets
        ast.after_match_skip_to = skip_target
        return True

    def _build_plan(self, ast):
        measures = (
            copy.deepcopy(ast.measures.measures) if ast.measures is not None else {}
        )
        definitions = (
            copy.deepcopy(ast.define.definitions) if ast.define is not None else {}
        )
        subsets = copy.deepcopy(ast.subsets or {})
        skip_target = ast.after_match_skip_to
        changed = False

        # Process in declaration order. SQL SUBSET members are primary pattern
        # variables, so removing one declaration cannot alter another one's
        # membership.
        for subset_name, members in list(subsets.items()):
            if len(members) == 1:
                source = members[0]
                rewritten_measures = {
                    alias: self._replace_all(expr, subset_name, source)
                    for alias, expr in measures.items()
                }
                rewritten_definitions = {
                    var: self._rebuild_condition(
                        var,
                        condition,
                        self._replace_all(condition.expression, subset_name, source),
                    )
                    for var, condition in definitions.items()
                }
                if self._mapping_changed(
                    measures, rewritten_measures
                ) or self._conditions_changed(definitions, rewritten_definitions):
                    changed = True
                measures = rewritten_measures
                definitions = rewritten_definitions
                rewritten_skip = self._replace_skip_target(
                    skip_target, subset_name, source
                )
                if rewritten_skip != skip_target:
                    changed = True
                    skip_target = rewritten_skip

            elif len(members) > 1:
                member_set = set(members)
                rewritten_definitions = {}
                for var, condition in definitions.items():
                    expr = self._rewrite_define_expr(
                        condition.expression,
                        ast.pattern,
                        subset_name,
                        member_set,
                        var,
                    )
                    rewritten_definitions[var] = self._rebuild_condition(
                        var, condition, expr
                    )
                if self._conditions_changed(definitions, rewritten_definitions):
                    changed = True
                definitions = rewritten_definitions

                rewritten_measures = {
                    alias: self._rewrite_measure_expr(
                        expr,
                        ast,
                        subset_name,
                        member_set,
                    )
                    for alias, expr in measures.items()
                }
                if self._mapping_changed(measures, rewritten_measures):
                    changed = True
                measures = rewritten_measures

            if (
                not self._mapping_references(measures, subset_name)
                and not self._conditions_reference(definitions, subset_name)
                and not self._skip_references(skip_target, subset_name)
            ):
                del subsets[subset_name]
                changed = True

        if not changed:
            return None
        return measures, definitions, subsets, skip_target

    # --------------------------- expression rewrite ---------------------------
    def _rewrite_define_expr(
        self, expr, pattern, subset, members, current_var, protected=False, edge="last"
    ):
        if isinstance(expr, ColumnRef):
            if expr.qualifier == subset and not protected:
                source = self._unique_running_source(
                    pattern, members, current_var, edge
                )
                if source is not None:
                    return ColumnRef(source, expr.name)
            return copy.deepcopy(expr)

        if isinstance(expr, SemanticsExpr):
            if (expr.semantics or "").upper() == "FINAL":
                inner = self._rewrite_final_expr(expr.expr, pattern, subset, members)
                return SemanticsExpr(expr.semantics, inner)
            return SemanticsExpr(
                expr.semantics,
                self._rewrite_define_expr(
                    expr.expr,
                    pattern,
                    subset,
                    members,
                    current_var,
                    protected,
                    edge,
                ),
            )

        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name in {"FIRST", "LAST"} and expr.args:
                if not self._has_zero_navigation_offset(expr):
                    # A non-zero offset can cross SUBSET member boundaries.
                    # For example, FIRST(U.X, 1) over U=(A,B) may select B
                    # even when FIRST(U.X) always selects A.  Keep the whole
                    # navigation expression, and therefore U, intact.
                    return copy.deepcopy(expr)
                nested_edge = "first" if name == "FIRST" else "last"
                return FunctionCall(
                    expr.name,
                    [
                        self._rewrite_define_expr(
                            expr.args[0],
                            pattern,
                            subset,
                            members,
                            current_var,
                            False,
                            nested_edge,
                        )
                    ]
                    + copy.deepcopy(expr.args[1:]),
                )
            nested_protected = protected or name in self._PROTECTED_FUNCTIONS
            return FunctionCall(
                expr.name,
                [
                    self._rewrite_define_expr(
                        arg,
                        pattern,
                        subset,
                        members,
                        current_var,
                        nested_protected,
                        edge,
                    )
                    for arg in expr.args
                ],
            )

        return self._map_expr(
            expr,
            lambda child: self._rewrite_define_expr(
                child,
                pattern,
                subset,
                members,
                current_var,
                protected,
                edge,
            ),
        )

    def _rewrite_measure_expr(self, expr, ast, subset, members):
        if isinstance(expr, SemanticsExpr):
            if (expr.semantics or "").upper() == "FINAL":
                return SemanticsExpr(
                    expr.semantics,
                    self._rewrite_final_expr(expr.expr, ast.pattern, subset, members),
                )
            return copy.deepcopy(expr)

        mode = (ast.rows_per_match or ONE_ROW_PER_MATCH).strip().upper()
        # SHOW/OMIT EMPTY MATCHES and WITH UNMATCHED ROWS are suboptions of
        # ALL ROWS PER MATCH, not separate output modes.  Their unqualified
        # measures therefore have the same RUNNING semantics as the bare
        # form.  An exact equality check would incorrectly treat them as
        # complete-match (FINAL) measures and could collapse a union variable
        # to a member that has not been reached on an earlier output row.
        if mode.startswith(ALL_ROWS_PER_MATCH):
            return copy.deepcopy(expr)
        return self._rewrite_final_expr(expr, ast.pattern, subset, members)

    def _rewrite_final_expr(
        self, expr, pattern, subset, members, protected=False, edge="last"
    ):
        if isinstance(expr, ColumnRef):
            if expr.qualifier == subset and not protected:
                source = self._unique_complete_source(pattern, members, edge)
                if source is not None:
                    return ColumnRef(source, expr.name)
            return copy.deepcopy(expr)
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name in {"FIRST", "LAST"} and expr.args:
                if not self._has_zero_navigation_offset(expr):
                    # The edge source proves only the zero-offset FIRST/LAST
                    # row.  It says nothing about which member supplies an
                    # offset row in the union variable's ordered row set.
                    return copy.deepcopy(expr)
                nested_edge = "first" if name == "FIRST" else "last"
                return FunctionCall(
                    expr.name,
                    [
                        self._rewrite_final_expr(
                            expr.args[0],
                            pattern,
                            subset,
                            members,
                            False,
                            nested_edge,
                        )
                    ]
                    + copy.deepcopy(expr.args[1:]),
                )
            nested_protected = protected or name in self._PROTECTED_FUNCTIONS
            return FunctionCall(
                expr.name,
                [
                    self._rewrite_final_expr(
                        arg,
                        pattern,
                        subset,
                        members,
                        nested_protected,
                        edge,
                    )
                    for arg in expr.args
                ],
            )
        if isinstance(expr, SemanticsExpr):
            return SemanticsExpr(
                expr.semantics,
                self._rewrite_final_expr(
                    expr.expr,
                    pattern,
                    subset,
                    members,
                    protected,
                    edge,
                ),
            )
        return self._map_expr(
            expr,
            lambda child: self._rewrite_final_expr(
                child,
                pattern,
                subset,
                members,
                protected,
                edge,
            ),
        )

    def _replace_all(self, expr, old, new):
        if isinstance(expr, ColumnRef):
            return ColumnRef(
                new if expr.qualifier == old else expr.qualifier, expr.name
            )
        if isinstance(expr, FunctionCall):
            return FunctionCall(
                expr.name, [self._replace_all(arg, old, new) for arg in expr.args]
            )
        return self._map_expr(expr, lambda child: self._replace_all(child, old, new))

    @staticmethod
    def _has_zero_navigation_offset(expr):
        """Return whether FIRST/LAST uses its implicit or explicit zero offset."""
        if len(expr.args) == 1:
            return True
        return (
            len(expr.args) == 2
            and isinstance(expr.args[1], Literal)
            and expr.args[1].value == 0
        )

    def _map_expr(self, expr, transform):
        if isinstance(expr, SemanticsExpr):
            return SemanticsExpr(expr.semantics, transform(expr.expr))
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(transform(expr.left), expr.op, transform(expr.right))
        if isinstance(expr, BetweenExpr):
            return BetweenExpr(
                transform(expr.expr), transform(expr.lower), transform(expr.upper)
            )
        if isinstance(expr, CaseExpr):
            return CaseExpr(
                [(transform(cond), transform(value)) for cond, value in expr.cases],
                transform(expr.else_expr) if expr.else_expr is not None else None,
            )
        if isinstance(expr, InExpr):
            return InExpr(
                transform(expr.expr), [transform(v) for v in expr.values], expr.is_not
            )
        if isinstance(expr, IsNullExpr):
            return IsNullExpr(transform(expr.expr), expr.is_not)
        if isinstance(expr, NotExpr):
            return NotExpr(transform(expr.expr))
        return copy.deepcopy(expr)

    # ----------------------- deterministic source analysis --------------------
    def _unique_complete_source(self, pattern, members, edge):
        sources, _ = self._edge_sources(pattern, members, edge)
        return next(iter(sources)) if len(sources) == 1 else None

    def _edge_sources(self, node, members, edge):
        if isinstance(node, VariablePattern):
            return ({node.name}, False) if node.name in members else (set(), True)
        if isinstance(node, EmptyPattern):
            return set(), True
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._edge_sources(node.pattern, members, edge)
        if isinstance(node, QuantifiedPattern):
            sources, absent = self._edge_sources(node.pattern, members, edge)
            return sources, node.atLeast == 0 or absent
        if isinstance(node, AlternationPattern):
            values = [
                self._edge_sources(child, members, edge) for child in node.patterns
            ]
            return set().union(*(value for value, _ in values)), any(
                a for _, a in values
            )
        if isinstance(node, ConcatenationPattern):
            children = node.patterns if edge == "first" else reversed(node.patterns)
            result = set()
            absent_so_far = True
            for child in children:
                if not absent_so_far:
                    break
                child_sources, child_absent = self._edge_sources(child, members, edge)
                result.update(child_sources)
                absent_so_far = absent_so_far and child_absent
            return result, absent_so_far
        if isinstance(node, PermutationPattern):
            values = [
                self._edge_sources(child, members, edge) for child in node.patterns
            ]
            return set().union(*(value for value, _ in values)), all(
                a for _, a in values
            )
        return set(), True

    def _unique_running_source(self, pattern, members, target, edge):
        _, observed = self._walk(pattern, {None}, members, target, edge)
        observed.discard(None)
        return next(iter(observed)) if len(observed) == 1 else None

    def _walk(self, node, incoming, members, target, edge):
        """Return (states after node, states observed when target is read)."""
        if isinstance(node, VariablePattern):
            outgoing = self._consume_variable(incoming, node.name, members, edge)
            return outgoing, (set(outgoing) if node.name == target else set())
        if isinstance(node, EmptyPattern):
            return set(incoming), set()
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._walk(node.pattern, incoming, members, target, edge)
        if isinstance(node, ConcatenationPattern):
            states = set(incoming)
            observed = set()
            for child in node.patterns:
                states, child_observed = self._walk(
                    child,
                    states,
                    members,
                    target,
                    edge,
                )
                observed.update(child_observed)
            return states, observed
        if isinstance(node, AlternationPattern):
            outgoing, observed = set(), set()
            for child in node.patterns:
                child_out, child_observed = self._walk(
                    child,
                    incoming,
                    members,
                    target,
                    edge,
                )
                outgoing.update(child_out)
                observed.update(child_observed)
            return outgoing, observed
        if isinstance(node, QuantifiedPattern):
            observed = set()
            starts = set(incoming)
            seen_starts = set()
            iterations = 0
            max_iterations = node.atMost
            while max_iterations is None or iterations < max_iterations:
                key = frozenset(starts)
                if key in seen_starts:
                    break
                seen_starts.add(key)
                starts, child_observed = self._walk(
                    node.pattern,
                    starts,
                    members,
                    target,
                    edge,
                )
                observed.update(child_observed)
                iterations += 1
            return (
                self._repeat_outputs(
                    node.pattern,
                    incoming,
                    node.atLeast,
                    node.atMost,
                    members,
                    edge,
                ),
                observed,
            )
        if isinstance(node, PermutationPattern):
            outgoing = self._advance(node, incoming, members, edge)
            if target not in node.get_variables():
                return outgoing, set()
            # Any sibling may precede the target. This broad approximation is
            # conservative; it deliberately declines ambiguous rewrites.
            if edge == "last" and target in members:
                return outgoing, {target}
            possible = set(incoming) | (set(node.get_variables()) & members)
            return outgoing, possible
        return set(incoming), set()

    def _consume_variable(self, incoming, var, members, edge):
        if var not in members:
            return set(incoming)
        if edge == "last":
            return {var}
        return {state if state is not None else var for state in incoming}

    def _advance(self, node, incoming, members, edge):
        if isinstance(node, VariablePattern):
            return self._consume_variable(incoming, node.name, members, edge)
        if isinstance(node, EmptyPattern):
            return set(incoming)
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._advance(node.pattern, incoming, members, edge)
        if isinstance(node, ConcatenationPattern):
            states = set(incoming)
            for child in node.patterns:
                states = self._advance(child, states, members, edge)
            return states
        if isinstance(node, AlternationPattern):
            result = set()
            for child in node.patterns:
                result.update(self._advance(child, incoming, members, edge))
            return result
        if isinstance(node, QuantifiedPattern):
            return self._repeat_outputs(
                node.pattern,
                incoming,
                node.atLeast,
                node.atMost,
                members,
                edge,
            )
        if isinstance(node, PermutationPattern):
            sources, subset_can_be_absent = self._edge_sources(node, members, edge)
            if edge == "last":
                result = set(sources)
                if subset_can_be_absent:
                    result.update(incoming)
                return result

            # Once a first member has already been seen, a permutation cannot
            # change it. Only the paths entering with no member select one of
            # the permutation's possible first sources.
            result = {state for state in incoming if state is not None}
            if None in incoming:
                result.update(sources)
                if subset_can_be_absent:
                    result.add(None)
            return result
        return set(incoming)

    def _repeat_outputs(self, body, incoming, minimum, maximum, members, edge):
        current = set(incoming)
        for _ in range(minimum):
            current = self._advance(body, current, members, edge)
        result = set(current)
        if maximum is not None and maximum <= minimum:
            return result

        count = minimum
        seen = set()
        while maximum is None or count < maximum:
            key = frozenset(current)
            if key in seen:
                break
            seen.add(key)
            current = self._advance(body, current, members, edge)
            result.update(current)
            count += 1
        return result

    # ------------------------------- utilities -------------------------------
    def _rebuild_condition(self, var, old_condition, expr):
        reasons = sorted(collect_define_dependency_reasons(var, expr))
        if reasons:
            return DependentCondition(var, expr, reasons)
        return IndependentCondition(var, expr)

    def _expr_references(self, expr, qualifier):
        if isinstance(expr, ColumnRef):
            return expr.qualifier == qualifier
        if isinstance(expr, FunctionCall):
            return any(self._expr_references(arg, qualifier) for arg in expr.args)
        return any(
            self._expr_references(child, qualifier) for child in self._children(expr)
        )

    def _children(self, expr):
        if isinstance(expr, SemanticsExpr):
            return [expr.expr]
        if isinstance(expr, BinaryExpr):
            return [expr.left, expr.right]
        if isinstance(expr, BetweenExpr):
            return [expr.expr, expr.lower, expr.upper]
        if isinstance(expr, CaseExpr):
            values = [item for pair in expr.cases for item in pair]
            return values + ([expr.else_expr] if expr.else_expr is not None else [])
        if isinstance(expr, InExpr):
            return [expr.expr] + list(expr.values)
        if isinstance(expr, (IsNullExpr, NotExpr)):
            return [expr.expr]
        return []

    def _mapping_references(self, mapping, qualifier):
        return any(self._expr_references(expr, qualifier) for expr in mapping.values())

    def _conditions_reference(self, definitions, qualifier):
        return any(
            self._expr_references(condition.expression, qualifier)
            for condition in definitions.values()
        )

    def _skip_references(self, skip_target, subset):
        skip = skip_target or ""
        return subset.upper() in re.findall(r"[A-Z_][A-Z0-9_$#]*", skip.upper())

    def _replace_skip_target(self, skip_target, old, new):
        if skip_target is None:
            return None
        return re.sub(
            rf"(?<![A-Z0-9_$#]){re.escape(old)}(?![A-Z0-9_$#])",
            new,
            skip_target,
            flags=re.IGNORECASE,
        )

    def _mapping_changed(self, before, after):
        return any(before[key].to_string() != after[key].to_string() for key in before)

    def _conditions_changed(self, before, after):
        return any(
            before[key].expression.to_string() != after[key].expression.to_string()
            for key in before
        )


redundant_subset_variable_removal_test_cases = [
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(V1 C)
    SUBSET D = (V1, C)
    DEFINE V1 AS TYPE = 'A' OR TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(V1 C)
    DEFINE V1 AS TYPE = 'A' OR TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.TS AS DTS, C.TS AS CTS
    ALL ROWS PER MATCH
    PATTERN(V1 C)
    SUBSET D = (V1)
    DEFINE V1 AS TYPE = 'A' OR TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES V1.TS AS DTS, C.TS AS CTS
    ALL ROWS PER MATCH
    PATTERN(V1 C)
    DEFINE V1 AS TYPE = 'A' OR TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.TS AS DTS, E.TS AS ETS
    ALL ROWS PER MATCH
    PATTERN(A B C D)
    SUBSET E = (A, B, C)
    DEFINE A AS VAL = 10, B AS VAL > 10, C AS VAL < 20, D AS VAL = E.VAL
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.TS AS DTS, E.TS AS ETS
    ALL ROWS PER MATCH
    PATTERN(A B C D)
    SUBSET E = (A, B, C)
    DEFINE A AS VAL = 10, B AS VAL > 10, C AS VAL < 20, D AS VAL = C.VAL
)""",
    ),
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.TS AS DTS, E.TS AS ETS
    ONE ROW PER MATCH
    PATTERN(A B C D)
    SUBSET E = (A, B, C)
    DEFINE A AS VAL = 10, B AS VAL > 10, C AS VAL < 20, D AS VAL = E.VAL
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.TS AS DTS, C.TS AS ETS
    ONE ROW PER MATCH
    PATTERN(A B C D)
    DEFINE A AS VAL = 10, B AS VAL > 10, C AS VAL < 20, D AS VAL = C.VAL
)""",
    ),
    # Alternation makes the last contributing member ambiguous.
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES U.TS AS UTS
    PATTERN(A (B | C) D)
    SUBSET U = (B, C)
    DEFINE D AS VAL = U.VAL
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES U.TS AS UTS
    PATTERN(A (B | C) D)
    SUBSET U = (B, C)
    DEFINE D AS VAL = U.VAL
)""",
    ),
    # Aggregate union semantics must not be collapsed to the last member.
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES SUM(U.VAL) AS TOTAL
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE C AS SUM(U.VAL) > 10
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES SUM(U.VAL) AS TOTAL
    PATTERN(A B C)
    SUBSET U = (A, B)
    DEFINE C AS SUM(U.VAL) > 10
)""",
    ),
    # A singleton alias is replaceable even under aggregates and in SKIP.
    (
        7,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES SUM(U.VAL) AS TOTAL
    AFTER MATCH SKIP TO LAST U
    PATTERN(A B)
    SUBSET U = (A)
    DEFINE B AS VAL > SUM(U.VAL)
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES SUM(A.VAL) AS TOTAL
    AFTER MATCH SKIP TO LAST A
    PATTERN(A B)
    DEFINE B AS VAL > SUM(A.VAL)
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        RedundantSubsetVariableRemovalRule,
        redundant_subset_variable_removal_test_cases,
        "redundant SUBSET variable removal",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
