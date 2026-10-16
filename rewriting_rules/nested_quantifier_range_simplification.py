if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
from dataclasses import dataclass

from expression import BinaryExpr, ColumnRef, IsNullExpr, Literal, NotExpr, NullLiteral
from expression_clause import (
    Condition,
    DependentCondition,
    IndependentCondition,
    collect_define_dependency_reasons,
)
from expression_simplifier import (
    _are_complementary,
    _condition_is_not_true,
    _expr_key,
    _flatten_logical_terms,
)
from patternnode import (
    AlternationPattern,
    AnchorPattern,
    AnchorType,
    ConcatenationPattern,
    EmptyPattern,
    ExcludedPattern,
    PatternNode,
    PermutationPattern,
    QuantifiedPattern,
    VariablePattern,
    get_pattern_variables,
)
from queryanalysis import RPRAbstractSyntaxTree

from .base import RewriteRule
from .utils import _strip_var_qualifier


@dataclass
class _RewriteAction:
    container: QuantifiedPattern | list[PatternNode]
    index: int | None
    replacement: PatternNode
    condition: Condition | None = None

    def commit(self, ast: RPRAbstractSyntaxTree) -> None:
        if self.index is None:
            self.container.pattern = self.replacement
        else:
            self.container[self.index] = self.replacement
        if self.condition is not None:
            ast.define.definitions[self.condition.var_name] = self.condition


class NestedQuantifierRangeSimplificationRule(RewriteRule):
    """
    Simplify a nested quantifier without changing SQL/RPR preferment.

    The always-local forms have an inner lower bound of one:

        (p0 p1{1,n} p2)+  -> (p0 p1 p2)+
        (p0 p1+ p2)*      -> (p0 p1 p2)*

    For m > 1, merely replacing n by 2m-1 preserves possible repetition
    counts, but not the preferred match.  For example, (B{2,8})+ consumes four
    B rows whereas (B{2,3})+ prefers one three-row inner iteration.  The m > 1
    greedy rewrite is therefore allowed only when nullable boundaries are
    provably disjoint from p1 and a mandatory, provably-disjoint following
    pattern forces every consecutive p1 row to remain inside the repetition.
    A reluctant inner range tries the retained low counts first, so it may be
    narrowed without an exit barrier, still under the stronger disjoint-boundary
    requirement used for m > 1.

    A lower bound of zero is deliberately unsupported.  It makes the repeated
    body nullable and invokes the SQL/RPR empty-repetition stopping rules.

    A separate contextual theorem handles (B C*?)+ followed by a mandatory,
    row-local variable disjoint from both B and C.  If B/C predicates depend
    only on their current row and a fixed prefix, it becomes B (B | C)* with
    C narrowed to C AND not_true(B evaluated on that same row).  See
    _collect_flatten_actions for the label-preservation argument and guards.
    """

    @property
    def name(self) -> str:
        return "nested_quantifier_range_simplification"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return bool(self._collect_actions(ast))

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        changed = False
        # Recompute the traversal context after every mutation.  In particular,
        # an inner rewrite can change the safety of an enclosing candidate.
        while True:
            actions = self._collect_actions(ast)
            if not actions:
                break
            actions[0].commit(ast)
            changed = True
        return changed

    def _collect_actions(self, ast: RPRAbstractSyntaxTree):
        actions = self._collect_flatten_actions(ast)
        if actions:
            return actions

        def visit(pattern: PatternNode, following=None) -> None:
            if isinstance(pattern, ConcatenationPattern):
                for index, child in enumerate(pattern.patterns):
                    inherited = [] if following is None else following
                    child_following = pattern.patterns[index + 1 :] + inherited
                    visit(child, child_following)
            elif isinstance(pattern, (AlternationPattern, PermutationPattern)):
                # A branch/permutation does not have one unconditional lexical
                # continuation, so m > 1 candidates below it stay conservative.
                for child in pattern.patterns:
                    visit(child, None)
            elif isinstance(
                pattern, (QuantifiedPattern, ExcludedPattern, AnchorPattern)
            ):
                # Re-entering a wrapper can change preferment or output.  The
                # wrapper itself still receives its continuation below, but a
                # nested candidate cannot use it as a simple exit barrier.
                visit(pattern.pattern, None)

            if not isinstance(pattern, QuantifiedPattern):
                return

            for container, middle_index, left, middle, right in self._candidates(
                pattern
            ):
                replacement = self._build_replacement(
                    ast, pattern, left, middle, right, following
                )
                if replacement is not None:
                    actions.append(_RewriteAction(container, middle_index, replacement))
                    return

        visit(ast.pattern, [])
        return actions

    def _collect_flatten_actions(self, ast):
        """Use a deliberately narrow, context-sensitive, label-safe theorem.

        Scope: a direct child of the top-level concatenation, with only plain
        variables before/after it, and no B/C occurrence outside the fragment.
        Outside DEFINE predicates must be row-local; B/C may also read plain
        prefix variables, whose last bound rows stay fixed inside this loop.
        Functions, subqueries, evolving B/C/SUBSET references and wrappers are
        unsupported.  Surface boundaries erased by parsing also fail closed.

        Proof: the disjoint mandatory successor fixes the loop's endpoint.
        At each interior row, reluctant C*? first offers the next iteration's
        B.  When B succeeds, opening that iteration accepts exactly the same
        remaining row sequences as extending C*?: both allow arbitrary B/C
        rows until the fixed successor.  The successor cannot demand a
        different label history.  Thus backtracking can never force a B-true
        row to be C in the preferred successful match.  Every interior row
        has the preferred label B if B is TRUE, otherwise C if C is TRUE.
        The flattened pattern with C AND not_true(B) has exactly those labels
        and the same endpoint, including when B is UNKNOWN.  Prefix/suffix
        rows are unchanged.  Consequently CLASSIFIER, running/final measures,
        SUBSETs, and every AFTER MATCH SKIP observe the same assignment.

        This is not a compositional identity under an arbitrary continuation;
        in particular a missing/overlapping successor or a suffix depending
        on COUNT(C.*) invalidates the proof.
        """
        if (
            not isinstance(ast.pattern, ConcatenationPattern)
            or ast.define is None
            or ast.raw_sql_override is not None
            or getattr(ast, "pattern_rewrite_boundaries", ())
        ):
            return []
        parts = ast.pattern.patterns
        for index, outer in enumerate(parts[:-1]):
            if not (
                isinstance(outer, QuantifiedPattern)
                and outer.atLeast == 1
                and outer.atMost is None
                and outer.isGreedy
                and isinstance(outer.pattern, ConcatenationPattern)
                and len(outer.pattern.patterns) == 2
            ):
                continue
            left, right = outer.pattern.patterns
            # The existing local range normalization supplies a single B.
            if not (
                isinstance(left, VariablePattern)
                and isinstance(right, QuantifiedPattern)
                and isinstance(right.pattern, VariablePattern)
                and right.atLeast == 0
                and right.atMost is None
                and not right.isGreedy
            ):
                continue
            b, c = left.name, right.pattern.name
            outside = parts[:index] + parts[index + 1 :]
            if b == c or not all(isinstance(node, VariablePattern) for node in outside):
                continue
            outside_names = {node.name for node in outside}
            if {b, c} & outside_names:
                continue
            prefix_names = {node.name for node in parts[:index]}
            expressions = {}
            for name in outside_names | {b, c}:
                condition = ast.define.definitions.get(name)
                expression = (
                    condition.expression
                    if condition is not None
                    else BinaryExpr(Literal(1), "=", Literal(1))
                )
                allowed = {name} | (prefix_names if name in {b, c} else set())
                if not self._fixed_row_predicate(expression, allowed):
                    break
                expressions[name] = expression
            else:
                successor = parts[index + 1].name
                if not all(
                    self._row_predicates_are_disjoint(
                        expressions[name], name, expressions[successor], successor
                    )
                    for name in (b, c)
                ):
                    continue
                new_expression = self._exclusive_right_predicate(
                    ast, expressions, prefix_names, b, c
                )
                reasons = collect_define_dependency_reasons(c, new_expression)
                condition = (
                    DependentCondition(c, new_expression, sorted(reasons))
                    if reasons
                    else IndependentCondition(c, new_expression)
                )
                replacement = ConcatenationPattern(
                    [
                        VariablePattern(b),
                        QuantifiedPattern(
                            AlternationPattern(
                                [VariablePattern(b), VariablePattern(c)]
                            ),
                            0,
                            None,
                            True,
                        ),
                    ]
                )
                return [_RewriteAction(parts, index, replacement, condition)]
        return []

    @staticmethod
    def _fixed_row_operand(expression, allowed):
        if isinstance(expression, (Literal, NullLiteral)):
            return True
        if isinstance(expression, ColumnRef):
            return expression.name != "*" and (
                expression.qualifier is None or expression.qualifier in allowed
            )
        # Restrict arithmetic to division by a positive integer constant.
        # In particular, copying an expression must not introduce volatile
        # calls or division by a possibly-zero row value.
        return (
            isinstance(expression, BinaryExpr)
            and expression.op == "/"
            and isinstance(expression.right, Literal)
            and type(expression.right.value) is int
            and expression.right.value >= 1
            and NestedQuantifierRangeSimplificationRule._fixed_row_operand(
                expression.left, allowed
            )
        )

    @classmethod
    def _fixed_row_predicate(cls, expression, allowed):
        if isinstance(expression, BinaryExpr):
            if expression.op in ("AND", "OR"):
                return cls._fixed_row_predicate(
                    expression.left, allowed
                ) and cls._fixed_row_predicate(expression.right, allowed)
            if expression.op in ("=", "!=", "<>", "<", "<=", ">", ">="):
                return cls._fixed_row_operand(
                    expression.left, allowed
                ) and cls._fixed_row_operand(expression.right, allowed)
        if isinstance(expression, IsNullExpr):
            return cls._fixed_row_operand(expression.expr, allowed)
        return False

    @staticmethod
    def _row_predicates_are_disjoint(left, left_name, right, right_name):
        # Necessary AND conjuncts suffice.  Only exact structural complements
        # prove this barrier; do not infer that unrelated cross-row columns
        # are the same column, or rely on approximate satisfiability checks.
        left = _strip_var_qualifier(left, left_name)
        right = _strip_var_qualifier(right, right_name)
        return any(
            _are_complementary(p, q)
            for p in _flatten_logical_terms(left, "AND")
            for q in _flatten_logical_terms(right, "AND")
        )

    @classmethod
    def _operand_columns(cls, expression, current_name):
        if isinstance(expression, ColumnRef):
            return {(expression.qualifier or current_name, expression.name)}
        if isinstance(expression, BinaryExpr):
            return cls._operand_columns(
                expression.left, current_name
            ) | cls._operand_columns(expression.right, current_name)
        return set()

    @classmethod
    def _guarded_nonnull_columns(cls, expression, current_name):
        columns = set()
        for term in _flatten_logical_terms(expression, "AND"):
            if isinstance(term, BinaryExpr) and term.op in (
                "=",
                "!=",
                "<>",
                "<",
                "<=",
                ">",
                ">=",
            ):
                columns |= cls._operand_columns(term.left, current_name)
                columns |= cls._operand_columns(term.right, current_name)
            elif isinstance(term, IsNullExpr) and term.is_not:
                columns |= cls._operand_columns(term.expr, current_name)
        return columns

    @classmethod
    def _operand_is_nonnull(cls, expression, current_name, nonnull, context):
        if isinstance(expression, Literal):
            return True
        if isinstance(expression, ColumnRef):
            return (
                expression.qualifier or current_name,
                expression.name,
            ) in nonnull or context.get_column_metadata(expression).nullable is False
        if isinstance(expression, BinaryExpr) and expression.op == "/":
            return cls._operand_is_nonnull(
                expression.left, current_name, nonnull, context
            ) and cls._operand_is_nonnull(
                expression.right, current_name, nonnull, context
            )
        return False

    @classmethod
    def _predicate_is_two_valued(cls, expression, current_name, nonnull, context):
        if isinstance(expression, IsNullExpr):
            return True
        if isinstance(expression, BinaryExpr):
            if expression.op in ("AND", "OR"):
                return cls._predicate_is_two_valued(
                    expression.left, current_name, nonnull, context
                ) and cls._predicate_is_two_valued(
                    expression.right, current_name, nonnull, context
                )
            return cls._operand_is_nonnull(
                expression.left, current_name, nonnull, context
            ) and cls._operand_is_nonnull(
                expression.right, current_name, nonnull, context
            )
        return False

    def _exclusive_right_predicate(self, ast, expressions, prefix_names, b, c):
        b_expression = _strip_var_qualifier(copy.deepcopy(expressions[b]), b)
        c_expression = _strip_var_qualifier(copy.deepcopy(expressions[c]), c)
        c_keys = {
            _expr_key(term) for term in _flatten_logical_terms(c_expression, "AND")
        }
        # On a C-TRUE row these common conjuncts are TRUE, so they may be
        # removed from B before taking FALSE-or-UNKNOWN, even under SQL 3VL.
        remaining = [
            term
            for term in _flatten_logical_terms(b_expression, "AND")
            if _expr_key(term) not in c_keys
        ]
        residual = BinaryExpr(Literal(1), "=", Literal(1))
        if remaining:
            residual = remaining[0]
            for term in remaining[1:]:
                residual = BinaryExpr(residual, "AND", term)
        nonnull = self._guarded_nonnull_columns(c_expression, c)
        for name in prefix_names:
            nonnull |= self._guarded_nonnull_columns(expressions[name], name)
        context = ast.expression_simplify_context()
        exclusion = (
            NotExpr(residual)
            if self._predicate_is_two_valued(residual, c, nonnull, context)
            else _condition_is_not_true(residual, context)
        )
        return BinaryExpr(copy.deepcopy(expressions[c]), "AND", exclusion)

    def _build_replacement(self, ast, outer, left, middle, right, following):
        # With a reluctant outer quantifier, reducing the inner upper bound can
        # make a following pattern match earlier and therefore change the
        # selected match even when the accepted language is unchanged.
        if not outer.isGreedy:
            return None
        if left.get_min_row_num() != 0 or right.get_min_row_num() != 0:
            return None
        if not isinstance(middle, QuantifiedPattern) or middle.atLeast < 1:
            return None
        if not all(
            self._boundary_is_safe(ast, boundary, middle.pattern, middle.atLeast)
            for boundary in (left, right)
        ):
            return None

        # (p0 p1{m,n} p2)+, including an unbounded n.
        if outer.atLeast == 1 and outer.atMost is None:
            cap = 2 * middle.atLeast - 1
            if middle.atMost is None or middle.atMost > cap:
                # m == 1 gives the fixed range {1}; emit the inner pattern
                # directly instead of the redundant/non-portable p1{1}?.
                if cap == 1:
                    return middle.pattern
                if middle.isGreedy and not self._has_safe_exit_barrier(
                    ast, middle.pattern, following
                ):
                    return None
                return QuantifiedPattern(
                    middle.pattern, middle.atLeast, cap, middle.isGreedy
                )
            return None

        # (p0 p1+ p2)* -> (p0 p1 p2)*.
        if (
            outer.atLeast == 0
            and outer.atMost is None
            and middle.atLeast == 1
            and middle.atMost is None
        ):
            return middle.pattern
        return None

    def _candidates(self, outer):
        """Yield p0/p1/p2 candidates, filling omitted boundaries with epsilon."""
        empty = EmptyPattern()
        inner = outer.pattern

        # (p1{m,n})+ has no ConcatenationPattern after AST normalization.
        if isinstance(inner, QuantifiedPattern):
            yield outer, None, empty, inner, empty
            return

        if not isinstance(inner, ConcatenationPattern):
            return

        parts = inner.patterns
        if len(parts) == 3:
            yield parts, 1, parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            # p0 is omitted: (p1 p2); or p2 is omitted: (p0 p1).
            yield parts, 0, empty, parts[0], parts[1]
            yield parts, 1, parts[0], parts[1], empty

    def _boundary_is_safe(
        self,
        ast: RPRAbstractSyntaxTree,
        boundary: PatternNode,
        middle: PatternNode,
        minimum: int,
    ) -> bool:
        if self._contains_anchor(boundary):
            return False
        disjoint = self._languages_are_provably_disjoint(ast, boundary, middle)
        # With m == 1, another outer iteration can consume the very next middle
        # row, so a reluctant nullable boundary keeps preferring its empty arm.
        # With m > 1 there may be a short tail that cannot start another
        # iteration; such a boundary must not be able to absorb that tail.
        if isinstance(boundary, QuantifiedPattern):
            return (minimum == 1 and not boundary.isGreedy) or disjoint
        return disjoint

    def _has_safe_exit_barrier(self, ast, middle, following) -> bool:
        """Whether a mandatory disjoint suffix forces consumption of p1 rows.

        A missing/nullable suffix lets the capped inner quantifier stop with a
        tail shorter than m.  A suffix whose first consuming variables overlap
        p1 can consume that tail itself.  Only a mandatory suffix with a
        provably-disjoint nullable-aware FIRST set is a safe local barrier.
        """
        if not following:
            return False

        # A direct end-of-partition anchor is also a barrier: a successful
        # match cannot leave a short p1 tail unmatched.  Keep this deliberately
        # narrow; an anchored consuming suffix still needs the FIRST-set proof
        # below because it may compete with p1.
        if (
            len(following) == 1
            and isinstance(following[0], AnchorPattern)
            and following[0].anchor_type == AnchorType.END
            and following[0].get_max_row_num() == 0
        ):
            return True

        suffix = (
            following[0]
            if len(following) == 1
            else ConcatenationPattern(list(following))
        )
        if suffix.get_min_row_num() == 0:
            return False

        first_vars = suffix.get_leftmost_variables()
        if not first_vars:
            return False
        return self._variable_sets_are_provably_disjoint(
            ast,
            get_pattern_variables(middle),
            first_vars,
        )

    def _contains_anchor(self, pattern: PatternNode) -> bool:
        if isinstance(pattern, AnchorPattern):
            return True
        if isinstance(
            pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(self._contains_anchor(child) for child in pattern.patterns)
        if isinstance(pattern, (QuantifiedPattern, ExcludedPattern)):
            return self._contains_anchor(pattern.pattern)
        return False

    def _languages_are_provably_disjoint(
        self,
        ast: RPRAbstractSyntaxTree,
        left: PatternNode,
        right: PatternNode,
    ) -> bool:
        left_vars = get_pattern_variables(left)
        right_vars = get_pattern_variables(right)
        return self._variable_sets_are_provably_disjoint(ast, left_vars, right_vars)

    def _variable_sets_are_provably_disjoint(
        self,
        ast: RPRAbstractSyntaxTree,
        left_vars,
        right_vars,
    ) -> bool:
        if not left_vars or not right_vars:
            return True

        definitions = ast.define.definitions
        all_vars = left_vars | right_vars
        if not all_vars <= definitions.keys():
            return False

        independent_exprs = {
            var: ast.define.get_independent_expr(var) for var in all_vars
        }
        if any(expr is None for expr in independent_exprs.values()):
            return False

        simplify_context = ast.expression_simplify_context()
        return all(
            not ast.define.has_expr_overlap(
                independent_exprs[left_var],
                independent_exprs[right_var],
                simplify_context,
            )
            for left_var in left_vars
            for right_var in right_vars
        )


nested_quantifier_range_simplification_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A?? B{2,8} C??)+)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A?? B{2,8} C??)+)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A?? B+ C??)*)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A?? B C??)*)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    # A nullable alternation is not itself quantified, so its independent
    # predicates must all be provably disjoint from B.
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN (((A? | X?) B{3,20} C??)+)
    DEFINE A AS V < 0, X AS V > 10, B AS V BETWEEN 1 AND 9, C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN (((A? | X?) B{3,20} C??)+)
    DEFINE A AS V < 0, X AS V > 10, B AS V BETWEEN 1 AND 9, C AS TYPE = 'C'
)""",
    ),
    # Unknown/overlapping independent conditions are deliberately unchanged.
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN (((A? | X?) B{2,9} C??)+)
    DEFINE A AS SIN(V) > 0, X AS V > 10, B AS V > 1, C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN (((A? | X?) B{2,9} C??)+)
    DEFINE A AS SIN(V) > 0, X AS V > 10, B AS V > 1, C AS TYPE = 'C'
)""",
    ),
    # A greedy nullable boundary is safe when it is provably disjoint from p1.
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A? B{2,9} C??)+)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A? B{2,9} C??)+)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    # A greedy boundary that may overlap p1 remains unchanged.
    (
        6,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A? B{2,9} C??)+)
    DEFINE A AS TYPE IN ('A', 'B'), B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((A? B{2,9} C??)+)
    DEFINE A AS TYPE IN ('A', 'B'), B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
    (
        7,
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((B{2,8} C??)+)
    DEFINE B AS TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE (
    ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    PATTERN ((B{2,8} C??)+)
    DEFINE B AS TYPE = 'B', C AS TYPE = 'C'
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        NestedQuantifierRangeSimplificationRule,
        nested_quantifier_range_simplification_test_cases,
        "nested quantifier range simplification",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
