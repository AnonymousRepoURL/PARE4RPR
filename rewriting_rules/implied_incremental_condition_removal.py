if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from .incremental_condition_pushdown import IncrementalConditionPushdownRule


class ImpliedIncrementalConditionRemovalRule(IncrementalConditionPushdownRule):
    """
    Remove a first-ORDER-BY-column upper-bound constraint when a later,
    mandatory pattern variable has an equal or stronger constraint.

    COUNT predicates deliberately do not belong to this rule.  They are
    handled by CountRangePushdownRule.

    DEFINE predicates are shared by every syntactic occurrence of a pattern
    variable.  A recipient with multiple syntactic occurrences is eligible
    only when the inherited pushdown analysis proves that every occurrence is
    strictly after the source and protected by a later target.  This permits
    patterns such as ``A N* B N* C`` while rejecting ``A N* C N*``.

    In addition, the source variable must occur at most once at runtime.
    Ordinary references such as A.TS mean RUNNING LAST(A.TS), so a repeated
    source can reset the reference between an earlier recipient and the later
    target.  Recipient and target variables may still be quantified when every
    occurrence is protected by a later target constraint.
    """

    @property
    def name(self) -> str:
        return "implied_incremental_condition_removal"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return bool(self._build_removal_plan(ast))

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        removal_plan = self._build_removal_plan(ast)
        if not removal_plan:
            return False

        parser = None
        changed = False
        for var_name, removable_indexes in removal_plan.items():
            condition = ast.define.definitions.get(var_name)
            if condition is None or condition.expression is None:
                continue

            conjuncts = self._flatten_and(condition.expression)
            remaining = [
                term
                for index, term in enumerate(conjuncts)
                if index not in removable_indexes
            ]
            if len(remaining) == len(conjuncts):
                continue

            new_expr = self._join_with_and(remaining)
            if parser is None:
                from clause_parser import ExprParser

                parser = ExprParser()
            ast.define.definitions[var_name] = parser._analyze_dependency(
                var_name, new_expr
            )
            changed = True

        return changed

    def _build_removal_plan(self, ast: RPRAbstractSyntaxTree):
        if not ast.match_order_by:
            return None

        order_col = ast.match_order_by[0][0].strip()
        if "." in order_col:
            order_col = order_col.split(".")[-1]
        sort_type = ast.match_order_by[0][1]

        # A target bound proves a numeric ordering fact only for non-NULL
        # intermediate values.  If the ORDER BY column is nullable, a NULL
        # recipient can sort before the target on some supported systems: its
        # original DEFINE comparison is UNKNOWN, but removing that comparison
        # would admit it.  Be conservative unless metadata rules this out.
        if ast.is_column_nullable(order_col) is not False:
            return None

        order_plans = [
            plan
            for plan in self._build_order_implication_plans(ast, order_col, sort_type)
            if self._plan_has_unique_positions(ast, plan)
        ]
        if not order_plans:
            return None

        removal_plan = {}
        for var_name, condition in ast.define.definitions.items():
            if condition is None or condition.expression is None:
                continue

            removable_indexes = set()
            for index, atom in enumerate(self._flatten_and(condition.expression)):
                for plan in order_plans:
                    if not self._can_remove_from_var(plan, var_name):
                        continue
                    implied_expr = plan["updates"].get(var_name)
                    if implied_expr is None:
                        continue
                    if self._order_expr_implies(
                        implied_expr, atom, var_name, order_col, sort_type
                    ):
                        removable_indexes.add(index)
                        break

            if removable_indexes:
                removal_plan[var_name] = removable_indexes

        return removal_plan or None

    def _build_order_implication_plans(
        self,
        ast: RPRAbstractSyntaxTree,
        order_col: str,
        sort_type: SortType,
    ):
        """Build only first-order-column plans; never inspect COUNT predicates."""
        must_appear_vars = ast.pattern.get_must_appear_var()
        plans = []
        for target_var, condition in ast.define.definitions.items():
            if (
                target_var not in must_appear_vars
                or condition is None
                or condition.expression is None
            ):
                continue

            for atom in self._flatten_and(condition.expression):
                constraint = self._parse_order_incremental_constraint(
                    atom, target_var, order_col, sort_type
                )
                if (
                    constraint is None
                    or constraint["source_var"] not in must_appear_vars
                ):
                    continue
                updates = self._build_order_updates(
                    ast,
                    constraint["source_var"],
                    target_var,
                    constraint["normalized_expr"],
                )
                if updates:
                    plans.append(
                        {
                            "kind": "order",
                            "updates": updates,
                            "target_var": target_var,
                            "order_col": order_col,
                            "sort_type": sort_type,
                        }
                    )
        return plans

    def _order_expr_implies(
        self,
        stronger_expr: Expr,
        weaker_expr: Expr,
        current_var: str,
        order_col: str,
        sort_type: SortType,
    ) -> bool:
        stronger = self._order_constraint_info(
            stronger_expr, current_var, order_col, sort_type
        )
        weaker = self._order_constraint_info(
            weaker_expr, current_var, order_col, sort_type
        )
        if stronger is None or weaker is None:
            return False
        if stronger["source_var"].upper() != weaker["source_var"].upper():
            return False
        if stronger["direction"] != weaker["direction"]:
            return False
        return self._upper_bound_implies(
            stronger["upper"],
            stronger["inclusive"],
            weaker["upper"],
            weaker["inclusive"],
        )

    def _upper_bound_implies(
        self,
        stronger_upper: float,
        stronger_inclusive: bool,
        weaker_upper: float,
        weaker_inclusive: bool,
    ) -> bool:
        if stronger_upper != weaker_upper:
            return stronger_upper < weaker_upper
        return not stronger_inclusive or weaker_inclusive

    def _plan_has_unique_positions(self, ast: RPRAbstractSyntaxTree, plan) -> bool:
        target_var = plan["target_var"]
        source_vars = {
            info["source_var"]
            for expression in plan["updates"].values()
            for info in [self._parse_normalized_order_constraint(expression)]
            if info is not None
        }
        return (
            self._variable_node_count(ast.pattern, target_var) == 1
            and len(source_vars) == 1
            and all(
                self._variable_node_count(ast.pattern, source_var) == 1
                and not self._variable_can_repeat(ast.pattern, source_var)
                for source_var in source_vars
            )
        )

    def _can_remove_from_var(self, plan, var_name: str) -> bool:
        # Membership in updates is the occurrence-sensitive proof.  The
        # general path adds a recipient only after _all_occurrences_between()
        # succeeds; the endpoint fast path proves that every non-endpoint row
        # is bracketed by the unique source and target.  Requiring one syntax
        # node here would discard safe repeated-gap variables after that proof.
        return (
            plan["target_var"].upper() != var_name.upper()
            and var_name in plan["updates"]
        )

    def _variable_node_count(self, node, var_name: str) -> int:
        if isinstance(node, VariablePattern):
            return int(node.name.upper() == var_name.upper())
        if isinstance(node, (QuantifiedPattern, ExcludedPattern, AnchorPattern)):
            return self._variable_node_count(node.pattern, var_name)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return sum(
                self._variable_node_count(child, var_name) for child in node.patterns
            )
        return 0

    def _variable_can_repeat(self, node, var_name: str) -> bool:
        """Whether one syntactic variable node may produce multiple rows."""
        if isinstance(node, VariablePattern):
            return False
        if isinstance(node, QuantifiedPattern):
            contains_var = self._variable_node_count(node.pattern, var_name) > 0
            if contains_var and (node.atMost is None or node.atMost > 1):
                return True
            return self._variable_can_repeat(node.pattern, var_name)
        if isinstance(node, (ExcludedPattern, AnchorPattern)):
            return self._variable_can_repeat(node.pattern, var_name)
        if isinstance(
            node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
        ):
            return any(
                self._variable_can_repeat(child, var_name) for child in node.patterns
            )
        return False

    def _join_with_and(self, terms):
        if not terms:
            return BinaryExpr(Literal(1), "=", Literal(1))
        result = terms[0]
        for term in terms[1:]:
            result = _simplify_logical_expr(result, "AND", term)
        return result


implied_incremental_condition_removal_test_cases = [
    # A later, stronger order constraint removes only the earlier order constraint.
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 9 AND TS - A.TS <= 200, C AS TYPE = 'C' AND COUNT(*) <= 10 AND TS - A.TS <= 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 9, C AS (TYPE = 'C' AND COUNT(*) <= 10) AND (TS - A.TS) <= 100
)""",
    ),
    # Equal inclusive order bounds are redundant; unrelated COUNT bounds stay.
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 2 AND TS - A.TS <= 100, C AS TYPE = 'C' AND COUNT(*) <= 10 AND TS - A.TS <= 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 2, C AS (TYPE = 'C' AND COUNT(*) <= 10) AND (TS - A.TS) <= 100
)""",
    ),
    # COUNT-only predicates must never make this rule applicable.
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 8, C AS TYPE = 'C' AND COUNT(*) <= 10
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND COUNT(*) <= 8, C AS TYPE = 'C' AND COUNT(*) <= 10
)""",
    ),
    # Regression: a variable reused after the implying target shares the same
    # DEFINE predicate, so removing its condition would not be equivalent.
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND TS - A.TS <= 200, C AS TYPE = 'C' AND TS - A.TS <= 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B+ C B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND TS - A.TS <= 200, C AS TYPE = 'C' AND TS - A.TS <= 100
)""",
    ),
    # A weaker later bound does not imply a stronger earlier bound.
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND TS - A.TS < 50, C AS TYPE = 'C' AND TS - A.TS <= 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND TS - A.TS < 50, C AS TYPE = 'C' AND TS - A.TS <= 100
)""",
    ),
    # At the same bound, '<' implies '<=', but not the other way around.
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B' AND TS - A.TS <= 100, C AS TYPE = 'C' AND TS - A.TS < 100
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B', C AS TYPE = 'C' AND (TS - A.TS) < 100
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        ImpliedIncrementalConditionRemovalRule,
        implied_incremental_condition_removal_test_cases,
        "implied incremental condition removal",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
