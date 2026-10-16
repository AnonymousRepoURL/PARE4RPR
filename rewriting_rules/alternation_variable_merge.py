if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from expression_simplifier import _condition_is_not_true


class AlternationVariableMergeRule(RewriteRule):
    """
    Merge eligible VariablePattern branches inside an AlternationPattern.

    Example:
        PATTERN(A | B)
        DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
    =>
        PATTERN(V1)
        DEFINE V1 AS TYPE = 'A' OR TYPE = 'B'

    A branch variable is eligible for merging when:
      1. The branch is a direct VariablePattern child of the AlternationPattern.
         Other branch types are left unchanged.
      2. The variable either has an IndependentCondition in DEFINE or is not
         defined there (which is treated as the implicit TRUE condition).
      3. The variable is not referenced outside its own DEFINE entry:
         - not by other DEFINE entries
         - not by MEASURES
         - not by AFTER MATCH SKIP TO
      4. If the variable also appears elsewhere in PATTERN, its DEFINE entry is
         kept for those remaining occurrences, but its branch predicate may
         still be folded into the merged alternation variable.
      5. Merged branches are consecutive and have identical membership across
         every SUBSET.  Affected SUBSET entries are rewritten to contain the
         generated variable (and any original variable used elsewhere).

    If some alternation variables appear elsewhere in PATTERN, those variables
    are kept, while the remaining eligible variables in the same alternation
    may still be merged.

    Logical/interval reduction uses the query's required column metadata, so
    full-domain predicates preserve SQL NULL behavior.
    """

    @property
    def name(self) -> str:
        return "alternation_variable_merge"

    _DEPENDENT_FUNCTIONS = (
        "PREV",
        "NEXT",
        "FIRST",
        "LAST",
        "MATCH_NUMBER",
        "AVG",
        "SUM",
        "COUNT",
        "MAX",
        "MIN",
    )

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._find_first_candidate(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        changed = False
        while True:
            plan = self._find_first_candidate(ast)
            if plan is None:
                break
            if plan.get("type") == "classifier_preserving_narrow":
                self._apply_classifier_preserving_narrow_plan(ast, plan)
            else:
                self._apply_plan(ast, plan)
                ast.pattern = ast.pattern.simplify()
            changed = True
        return changed

    def _find_first_candidate(self, ast: RPRAbstractSyntaxTree):
        classifier_is_visible = ast.measures is not None and ast.measures.has_func(
            "CLASSIFIER"
        )

        pattern_counts = self._count_pattern_variables(ast.pattern)

        def visit(node):
            if isinstance(node, AlternationPattern):
                if classifier_is_visible:
                    plan = self._build_classifier_preserving_narrow_plan(
                        ast, node, pattern_counts
                    )
                else:
                    plan = self._build_candidate_plan(ast, node, pattern_counts)
                if plan is not None:
                    return plan
                for child in node.patterns:
                    nested = visit(child)
                    if nested is not None:
                        return nested
                return None

            if isinstance(node, ConcatenationPattern):
                for child in node.patterns:
                    nested = visit(child)
                    if nested is not None:
                        return nested
                return None

            if isinstance(node, PermutationPattern):
                for child in node.patterns:
                    nested = visit(child)
                    if nested is not None:
                        return nested
                return None

            if isinstance(node, QuantifiedPattern):
                return visit(node.pattern)

            if isinstance(node, ExcludedPattern):
                return visit(node.pattern)

            if isinstance(node, AnchorPattern):
                return visit(node.pattern)

            return None

        return visit(ast.pattern)

    def _build_classifier_preserving_narrow_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        alt: AlternationPattern,
        pattern_counts: Dict[str, int],
    ):
        if len(alt.patterns) < 2 or ast.define is None:
            return None

        definitions = ast.define.definitions
        simplify_context = ast.expression_simplify_context()
        all_direct_vars = [
            p.name for p in alt.patterns if isinstance(p, VariablePattern)
        ]
        if len(all_direct_vars) < 2:
            return None
        prior_expr = None
        prior_subset_signature = None
        for branch in alt.patterns:
            if isinstance(branch, VariablePattern):
                var_name = branch.name
                stripped = self._classifier_preserving_stripped_condition(
                    ast, var_name, definitions, pattern_counts
                )
                if stripped is None:
                    continue

                # Narrowing a later branch removes the possibility of
                # backtracking from an earlier branch to that later branch.
                # That is only safe when such a reassignment is invisible to
                # every SUBSET. A later DEFINE may observe the assignment via
                # a subset aggregate even without directly referring to either
                # primary pattern variable.
                subset_signature = self._subset_membership_signature(ast, var_name)
                if (
                    prior_expr is not None
                    and subset_signature != prior_subset_signature
                ):
                    prior_expr = None

                if prior_expr is not None:
                    if self._is_true_expr(prior_expr):
                        narrowed = BinaryExpr(Literal(1), "!=", Literal(1))
                    else:
                        narrowed = _simplify_logical_expr(
                            stripped,
                            "AND",
                            _condition_is_not_true(prior_expr, simplify_context),
                            simplify_context,
                        )
                    if narrowed.to_string() != stripped.to_string() and len(
                        narrowed.to_string()
                    ) <= len(stripped.to_string()):
                        return {
                            "type": "classifier_preserving_narrow",
                            "var_name": var_name,
                            "narrowed_expr": narrowed,
                        }

                prior_expr = (
                    stripped
                    if prior_expr is None
                    else _simplify_logical_expr(
                        prior_expr,
                        "OR",
                        stripped,
                        simplify_context,
                    )
                )
                prior_subset_signature = subset_signature
                continue

            prior_expr = None
            prior_subset_signature = None

        return None

    def _classifier_preserving_branch_has_independent_condition(
        self, condition
    ) -> bool:
        if condition is None:
            return True
        return (
            isinstance(condition, IndependentCondition)
            and condition.expression is not None
            and not self._has_dependent_function(condition.expression)
        )

    def _classifier_preserving_stripped_condition(
        self,
        ast: RPRAbstractSyntaxTree,
        var_name: str,
        definitions: Dict[str, object],
        pattern_counts: Dict[str, int],
    ) -> Optional[Expr]:
        condition = definitions.get(var_name)
        if (
            not self._classifier_preserving_branch_has_independent_condition(condition)
            or pattern_counts.get(var_name, 0) != 1
            or self._is_var_referenced_by_other_define(ast, var_name)
        ):
            return None

        if condition is None:
            return BinaryExpr(Literal(1), "=", Literal(1))
        return _strip_var_qualifier(condition.expression, var_name)

    def _subset_membership_signature(
        self,
        ast: RPRAbstractSyntaxTree,
        var_name: str,
    ) -> Tuple[str, ...]:
        """Return the SUBSET memberships through which a branch is observable."""
        return tuple(
            subset_name
            for subset_name, members in (ast.subsets or {}).items()
            if var_name in members
        )

    def _apply_classifier_preserving_narrow_plan(
        self, ast: RPRAbstractSyntaxTree, plan
    ) -> None:
        var_name = plan["var_name"]
        ast.define.definitions[var_name] = IndependentCondition(
            var_name, plan["narrowed_expr"]
        )

    def _is_true_expr(self, expr: Expr) -> bool:
        expr = _unwrap_semantics_expr(expr)
        return (
            isinstance(expr, BinaryExpr)
            and expr.op == "="
            and isinstance(expr.left, Literal)
            and isinstance(expr.right, Literal)
            and expr.left.value == expr.right.value
        )

    def _build_candidate_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        alt: AlternationPattern,
        pattern_counts: Dict[str, int],
    ):
        if len(alt.patterns) < 2:
            return None

        variable_branches = [p for p in alt.patterns if isinstance(p, VariablePattern)]
        if len(variable_branches) < 2:
            return None

        local_counts: Dict[str, int] = {}
        for pattern in variable_branches:
            local_counts[pattern.name] = local_counts.get(pattern.name, 0) + 1

        eligible_vars: Set[str] = set()
        vars_with_external_occurrence: Set[str] = set()
        definitions = ast.define.definitions if ast.define is not None else {}

        for pattern in variable_branches:
            var_name = pattern.name

            condition = definitions.get(var_name)
            if condition is not None:
                if not isinstance(condition, IndependentCondition):
                    continue
                if condition.expression is None:
                    continue
                # Do not rely solely on the condition classification here.
                # ASTs may be constructed or transformed without rerunning the
                # dependency analyzer, and changing the current variable also
                # changes the meaning of navigation and aggregate functions.
                if self._has_dependent_function(condition.expression):
                    continue
            if self._is_var_referenced_by_other_define(ast, var_name):
                continue
            if self._measures_reference_var(ast, var_name):
                continue
            if self._skip_references_var(ast, var_name):
                continue

            if pattern_counts.get(var_name, 0) > local_counts.get(var_name, 0):
                vars_with_external_occurrence.add(var_name)

            eligible_vars.add(var_name)

        # Only merge a consecutive run of direct variable branches.  Moving a
        # predicate across another branch can change alternation priority.
        # Within such a run, variables must have the same membership in every
        # SUBSET; otherwise a union variable would lose the ability to
        # distinguish one of the original variables after the merge.
        mergeable_vars = self._first_subset_compatible_run(ast, alt, eligible_vars)
        if len(mergeable_vars) < 2:
            return None
        mergeable_set = set(mergeable_vars)

        for def_var, condition in definitions.items():
            if condition is None or condition.expression is None:
                continue
            if def_var in mergeable_set:
                continue
            if _expr_references_any_var(condition.expression, mergeable_set):
                return None

        if ast.measures is not None:
            for expr in ast.measures.measures.values():
                if expr is not None and _expr_references_any_var(expr, mergeable_set):
                    return None

        if ast.after_match_skip_to is not None:
            for var_name in mergeable_vars:
                if self._skip_references_var(ast, var_name):
                    return None

        merged_expr = self._build_merged_expr(ast, mergeable_vars)
        if merged_expr is None:
            return None

        merged_name = choose_generated_pattern_variable_name(ast)
        return {
            "alternation": alt,
            "mergeable_vars": mergeable_vars,
            "mergeable_set": mergeable_set,
            "vars_with_external_occurrence": vars_with_external_occurrence,
            "merged_expr": merged_expr,
            "merged_name": merged_name,
        }

    def _first_subset_compatible_run(
        self,
        ast: RPRAbstractSyntaxTree,
        alt: AlternationPattern,
        eligible_vars: Set[str],
    ) -> List[str]:
        subsets = ast.subsets or {}

        def signature(var_name: str) -> Tuple[str, ...]:
            return tuple(
                subset_name
                for subset_name, members in subsets.items()
                if var_name in members
            )

        run: List[str] = []
        run_signature: Optional[Tuple[str, ...]] = None
        for branch in alt.patterns + [None]:
            if isinstance(branch, VariablePattern) and branch.name in eligible_vars:
                branch_signature = signature(branch.name)
                if run and branch_signature != run_signature:
                    if len(run) >= 2:
                        return run
                    run = []
                if not run:
                    run_signature = branch_signature
                run.append(branch.name)
                continue

            if len(run) >= 2:
                return run
            run = []
            run_signature = None

        return []

    def _apply_plan(self, ast: RPRAbstractSyntaxTree, plan) -> None:
        alt: AlternationPattern = plan["alternation"]
        mergeable_set: Set[str] = plan["mergeable_set"]
        merged_name: str = plan["merged_name"]
        merged_expr: Expr = plan["merged_expr"]

        new_patterns: List = []
        inserted = False
        for pattern in alt.patterns:
            if isinstance(pattern, VariablePattern) and pattern.name in mergeable_set:
                if not inserted:
                    new_patterns.append(VariablePattern(merged_name))
                    inserted = True
                continue
            new_patterns.append(pattern)

        if not new_patterns:
            new_patterns = [VariablePattern(merged_name)]

        alt.patterns = new_patterns
        self._rewrite_define(
            ast,
            plan["mergeable_vars"],
            plan["vars_with_external_occurrence"],
            merged_name,
            merged_expr,
        )
        self._rewrite_subsets(
            ast,
            plan["mergeable_vars"],
            plan["vars_with_external_occurrence"],
            merged_name,
        )

    def _rewrite_subsets(
        self,
        ast: RPRAbstractSyntaxTree,
        mergeable_vars: List[str],
        vars_with_external_occurrence: Set[str],
        merged_name: str,
    ) -> None:
        if not ast.subsets:
            return

        mergeable_set = set(mergeable_vars)
        for subset_name, members in ast.subsets.items():
            if not mergeable_set.issubset(set(members)):
                continue

            last_merged_index = max(
                index for index, member in enumerate(members) if member in mergeable_set
            )
            rewritten = []
            for index, member in enumerate(members):
                if (
                    member not in mergeable_set
                    or member in vars_with_external_occurrence
                ):
                    rewritten.append(member)
                if index == last_merged_index:
                    rewritten.append(merged_name)
            ast.subsets[subset_name] = rewritten

    def _rewrite_define(
        self,
        ast: RPRAbstractSyntaxTree,
        mergeable_vars: List[str],
        vars_with_external_occurrence: Set[str],
        merged_name: str,
        merged_expr: Expr,
    ) -> None:
        if ast.define is None:
            ast.define = DefineSubClause({})
        mergeable_set = set(mergeable_vars)
        new_defs = {}
        inserted = False
        last_merged_var = next(
            (
                var_name
                for var_name in reversed(list(ast.define.definitions.keys()))
                if var_name in mergeable_set
            ),
            None,
        )

        for var_name, condition in ast.define.definitions.items():
            if var_name in mergeable_set:
                if (
                    var_name == last_merged_var
                    and var_name in vars_with_external_occurrence
                ):
                    new_defs[merged_name] = IndependentCondition(
                        merged_name, merged_expr
                    )
                    inserted = True
                if var_name in vars_with_external_occurrence:
                    new_defs[var_name] = condition
                if var_name == last_merged_var and not inserted:
                    new_defs[merged_name] = IndependentCondition(
                        merged_name, merged_expr
                    )
                    inserted = True
                continue
            new_defs[var_name] = condition

        if not inserted:
            new_defs[merged_name] = IndependentCondition(merged_name, merged_expr)

        ast.define.definitions = new_defs

    def _build_merged_expr(
        self, ast: RPRAbstractSyntaxTree, mergeable_vars: List[str]
    ) -> Optional[Expr]:
        merged_expr = None
        simplify_context = ast.expression_simplify_context()
        definitions = ast.define.definitions if ast.define is not None else {}
        for var_name in mergeable_vars:
            condition = definitions.get(var_name)
            if condition is None:
                stripped = BinaryExpr(Literal(1), "=", Literal(1))
            else:
                if condition.expression is None:
                    return None
                stripped = _strip_var_qualifier(condition.expression, var_name)
                if stripped is None:
                    return None
            if merged_expr is None:
                merged_expr = stripped
            else:
                merged_expr = _simplify_logical_expr(
                    merged_expr,
                    "OR",
                    stripped,
                    simplify_context,
                    preserve_unknown=False,  # Complete DEFINE filter.
                )
        return merged_expr

    def _has_dependent_function(self, expr: Expr) -> bool:
        return any(expr.has_func(func_name) for func_name in self._DEPENDENT_FUNCTIONS)

    def _count_pattern_variables(self, pattern) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        def visit(node):
            if isinstance(node, VariablePattern):
                key = node.name
                counts[key] = counts.get(key, 0) + 1
                return
            if isinstance(node, QuantifiedPattern):
                visit(node.pattern)
                return
            if isinstance(
                node, (ConcatenationPattern, AlternationPattern, PermutationPattern)
            ):
                for child in node.patterns:
                    visit(child)
                return
            if isinstance(node, (ExcludedPattern, AnchorPattern)):
                visit(node.pattern)

        visit(pattern)
        return counts

    def _is_var_referenced_by_other_define(
        self, ast: RPRAbstractSyntaxTree, var_name: str
    ) -> bool:
        definitions = ast.define.definitions if ast.define is not None else {}
        for def_var, condition in definitions.items():
            if def_var == var_name:
                continue
            if condition is not None and condition.expression is not None:
                if _expr_references_any_var(condition.expression, {var_name}):
                    return True
        return False

    def _measures_reference_var(
        self, ast: RPRAbstractSyntaxTree, var_name: str
    ) -> bool:
        if ast.measures is None:
            return False
        for expr in ast.measures.measures.values():
            if expr is not None and _expr_references_any_var(expr, {var_name}):
                return True
        return False

    def _skip_references_var(self, ast: RPRAbstractSyntaxTree, var_name: str) -> bool:
        if ast.after_match_skip_to is None:
            return False
        skip_words = ast.after_match_skip_to.split()

        if len(skip_words) == 1:
            return skip_words[0] == var_name
        if len(skip_words) == 2 and skip_words[0] in {"FIRST", "LAST"}:
            return skip_words[1] == var_name
        return False


alternation_variable_merge_test_cases = [
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A | B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(V1)
    DEFINE V1 AS TYPE = 'A' OR TYPE = 'B'
)""",
    ),
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN((A | B | C) C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS TYPE = 'C'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(V1 C)
    DEFINE V1 AS (TYPE = 'A' OR TYPE = 'B') OR TYPE = 'C', C AS TYPE = 'C'
)""",
    ),
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN((A | B) C D)
    DEFINE A AS VAL < 10, B AS VAL > 5, C AS VAL < 1, D AS VAL < 5
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(V1 C D)
    DEFINE V1 AS 1 = 1, C AS VAL < 1, D AS VAL < 5
)""",
    ),
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER () AS CLS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A | B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A | B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    table_schemas = {
        "TA": {
            "columns": {
                "NULLABLE_VAL": "INT",
                "NOT_NULL_VAL": "INT",
            },
            "column_nullables": {
                "NULLABLE_VAL": True,
                "NOT_NULL_VAL": False,
            },
        },
    }
    return run_rule_tests(
        AlternationVariableMergeRule,
        alternation_variable_merge_test_cases,
        "alternation variable merge",
        verbose,
        default_table_schemas=table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
