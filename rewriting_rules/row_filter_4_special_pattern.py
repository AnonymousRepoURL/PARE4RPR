if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import re

from .utils import *
from .physical_navigation import has_physical_navigation
from expression_simplifier import (
    ExpressionColumnMetadata,
    SimplifyContext,
    _condition_is_not_true,
    _extract_range_constraint,
    _same_column_ref,
)


class RowFilter4SpecialPatternRule(RewriteRule):
    """
    Push a row filter through a sequence of singleton event variables and
    zero-or-more compressible gaps.

    The accepted pattern shape is ``E0 G0* E1 ... Gn* En``.  Each ``Ei`` is a
    singleton variable and each gap body is either one variable or an
    alternation of variables.  For ALL ROWS PER MATCH every complete gap must
    be excluded; ONE ROW PER MATCH may use a bare gap.

    Let ``E`` be the OR of all singleton-event DEFINE predicates, and ``Gi``
    the OR of the row-local predicates accepted by gap i.  The pushed filter
    is::

        E OR (G0 IS NOT TRUE) OR ... OR (Gn IS NOT TRUE)

    ``IS NOT TRUE`` is rendered with ordinary SQL predicates (or LNNVL as a
    conservative fallback).  Consequently, every row actually removed by
    the WHERE clause is guaranteed, under SQL three-valued logic, to satisfy
    every gap predicate.  Such rows can be inserted into or erased from any
    ``*`` gap without changing the retained event sequence.

    The remaining checks make that erasure unobservable: gap DEFINE predicates
    are deterministic row-local expressions; singleton events expose a safe
    row-local necessary condition, with any residual dependency restricted to
    retained singleton events; gap variables are not observed by MEASURES or
    skip targets; measures are not cardinality/physical-row sensitive; the
    match starts and ends with retained singleton variables; and WITH
    UNMATCHED ROWS is rejected.
    """

    @property
    def name(self) -> str:
        return "row_filter_4_special_pattern"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._check(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        info = self._check(ast)
        if info is None:
            return False

        or_expr = info["filter_expr"]
        if or_expr is None:
            return False

        where_sql = or_expr.to_string()
        source = ast.input_table.strip()
        rebind_alias = info.get("rebind_alias")
        flattened = ast.flatten_safe_select_star_input(source)
        if flattened is None:
            ast.input_table = f"(SELECT * FROM {source} WHERE {where_sql})"
            # The generic wrapper optimizer accepts a broader SQL subset than
            # flatten_safe_select_star_input().  Preserve this query-block
            # boundary when the strict proof fails; ROWNUM predicates are a
            # cardinality-sensitive example that must remain inside it.
            ast._skip_from_clause_optimization_once = True
        else:
            base_source, input_predicates = flattened
            predicates = [f"({predicate})" for predicate in input_predicates]
            predicates.append(f"({where_sql})")
            ast.input_table = (
                f"(SELECT * FROM {base_source} WHERE {' AND '.join(predicates)})"
            )
        if rebind_alias is not None:
            ast.input_table += f" {rebind_alias}"
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self, ast: RPRAbstractSyntaxTree):
        """Return a proved rewrite plan, or ``None`` when any proof fails."""
        if not isinstance(ast.pattern, ConcatenationPattern):
            return None

        can_rebind, rebind_alias = resolve_input_rebind_alias(ast)
        if not can_rebind:
            return None

        patterns = ast.pattern.patterns
        n = len(patterns)
        if n < 3 or n % 2 == 0:
            return None

        rows_pm = (ast.rows_per_match or "ONE ROW PER MATCH").strip().upper()
        if "WITH UNMATCHED ROWS" in rows_pm:
            return None
        is_all_rows = rows_pm.startswith("ALL ROWS PER MATCH")
        if not is_all_rows and rows_pm != "ONE ROW PER MATCH":
            return None

        event_vars: List[str] = []
        gaps: List[List[str]] = []

        for i, p in enumerate(patterns):
            if i % 2 == 0:
                if not isinstance(p, VariablePattern):
                    return None
                event_vars.append(p.name)
                continue

            if is_all_rows and not isinstance(p, ExcludedPattern):
                return None
            inner = p.pattern if isinstance(p, ExcludedPattern) else p
            if not isinstance(inner, QuantifiedPattern):
                return None
            if inner.atLeast != 0 or inner.atMost is not None:
                return None
            gap_vars = self._gap_atom_variables(inner.pattern)
            if gap_vars is None:
                return None
            gaps.append(gap_vars)

        # A singleton without DEFINE is a wildcard. Then no row-local
        # predicate can prove that an input row is not one of the retained
        # events, so there is no useful safe rewrite.
        event_var_set = set(event_vars)
        event_filters: Dict[str, Expr] = {}
        for var in event_vars:
            cond = ast.define.definitions.get(var)
            if (
                cond is None
                or cond.expression is None
                or not self._is_safe_event_predicate(
                    cond.expression,
                    event_var_set,
                )
            ):
                return None

            # A dependent singleton event may still have a row-local
            # necessary condition. For example, from
            #   B.TYPE = 'B' AND B.VAL = A.VAL
            # retain B.TYPE = 'B' for the input filter while leaving the full
            # DEFINE predicate inside MATCH_RECOGNIZE.
            independent = cond.get_independent_filter()
            if independent is None or not self._is_safe_row_predicate(independent):
                return None
            stripped = self._strip_var_qualifier(independent, var)
            if stripped is None:
                return None
            event_filters[var] = stripped

        # Gap membership needs a sufficient condition: every removed row must
        # really be accepted by every gap. A merely necessary independent
        # conjunct is not enough here, so defined gap variables must remain
        # completely row-local.
        for gap_vars in gaps:
            for var in gap_vars:
                cond = ast.define.definitions.get(var)
                if cond is not None and (
                    not isinstance(cond, IndependentCondition)
                    or cond.expression is None
                    or not self._is_safe_row_predicate(cond.expression)
                ):
                    return None

        # Only gap-only variables can classify a row that the filter removes.
        noise_var_set = set().union(*(set(item) for item in gaps)) - set(event_vars)
        noise_dependent_vars = self._noise_dependent_var_names(ast, noise_var_set)

        if ast.measures is not None:
            for expr in ast.measures.measures.values():
                if expr is not None and self._expr_references_pattern_vars(
                    expr, noise_dependent_vars
                ):
                    return None
                if expr is not None and self._measure_depends_on_removed_rows(
                    expr, set(event_vars)
                ):
                    return None

        if ast.after_match_skip_to is not None:
            skip_identifiers = set(
                re.findall(r"[A-Z_][A-Z0-9_$#]*", ast.after_match_skip_to.upper())
            )
            if skip_identifiers & noise_dependent_vars:
                return None

        filter_expr = self._build_filter_condition(
            ast,
            event_vars,
            gaps,
            event_filters,
        )
        if filter_expr is None or self._is_trivially_true(filter_expr):
            return None
        return {
            "event_vars": event_vars,
            "noise_vars": list(noise_var_set),
            "gaps": gaps,
            "filter_expr": filter_expr,
            "rebind_alias": rebind_alias,
        }

    @staticmethod
    def _gap_atom_variables(pattern) -> Optional[List[str]]:
        """Return the one-row alternatives accepted by a repeated gap atom."""
        if isinstance(pattern, VariablePattern):
            return [pattern.name]
        if isinstance(pattern, AlternationPattern):
            if not pattern.patterns or not all(
                isinstance(item, VariablePattern) for item in pattern.patterns
            ):
                return None
            return [item.name for item in pattern.patterns]
        return None

    def _build_filter_condition(
        self,
        ast: RPRAbstractSyntaxTree,
        event_vars: List[str],
        gaps: List[List[str]],
        event_filters: Dict[str, Expr],
    ) -> Optional[Expr]:
        """Build ``events OR any-gap-is-not-true`` under SQL 3VL."""
        exprs = [event_filters[var] for var in event_vars]

        if not exprs:
            return None

        result = exprs[0]
        for expr in exprs[1:]:
            result = BinaryExpr(result, "OR", expr)

        context = self._nullable_safe_simplify_context(ast)
        for gap_vars in gaps:
            gap_expr = self._build_gap_condition(ast, gap_vars, context)
            if gap_expr is None:
                return None
            if self._is_trivially_true(gap_expr):
                continue
            not_true = self._condition_is_not_true(gap_expr, context)
            result = self._merge_or_terms(result, not_true, context)
        return result

    def _build_gap_condition(
        self,
        ast: RPRAbstractSyntaxTree,
        gap_vars: List[str],
        context: SimplifyContext,
    ) -> Optional[Expr]:
        result = None
        for var in gap_vars:
            cond = ast.define.definitions.get(var)
            if cond is None or self._is_trivially_true(cond.expression):
                return BinaryExpr(Literal(1), "=", Literal(1))
            stripped = self._strip_var_qualifier(cond.expression, var)
            if stripped is None:
                return None
            result = (
                stripped
                if result is None
                else _simplify_logical_expr(result, "OR", stripped, context)
            )
        return result

    def _condition_is_not_true(self, expr: Expr, context: SimplifyContext) -> Expr:
        """Use an explicit range complement so common filters stay compact."""
        if isinstance(expr, SemanticsExpr):
            return self._condition_is_not_true(expr.expr, context)
        if (
            isinstance(expr, BetweenExpr)
            and isinstance(expr.expr, ColumnRef)
            and isinstance(expr.lower, Literal)
            and isinstance(expr.upper, Literal)
        ):
            terms = [
                BinaryExpr(expr.expr, "<", expr.lower),
                BinaryExpr(expr.expr, ">", expr.upper),
            ]
            if context.get_column_metadata(expr.expr).nullable is not False:
                terms.append(IsNullExpr(expr.expr))
            result = terms[0]
            for term in terms[1:]:
                result = _simplify_logical_expr(result, "OR", term, context)
            return result
        return _condition_is_not_true(expr, context)

    @staticmethod
    def _nullable_safe_simplify_context(ast: RPRAbstractSyntaxTree) -> SimplifyContext:
        null_extended = input_may_null_extend_columns(ast.input_table)

        def resolve(column: ColumnRef):
            metadata = ast.get_column_metadata(column.to_string())
            if metadata is None:
                return ExpressionColumnMetadata(None, True)
            return ExpressionColumnMetadata(
                metadata.data_type,
                True if null_extended else metadata.nullable,
            )

        return SimplifyContext(resolve)

    @staticmethod
    def _flatten_or(expr: Expr) -> List[Expr]:
        if isinstance(expr, SemanticsExpr):
            return RowFilter4SpecialPatternRule._flatten_or(expr.expr)
        if isinstance(expr, BinaryExpr) and expr.op == "OR":
            return RowFilter4SpecialPatternRule._flatten_or(
                expr.left
            ) + RowFilter4SpecialPatternRule._flatten_or(expr.right)
        return [expr]

    def _merge_or_terms(
        self, left: Expr, right: Expr, context: SimplifyContext
    ) -> Expr:
        """OR predicates while eliminating covered one-column range terms."""
        terms: List[Expr] = []
        ranges = []
        for term in self._flatten_or(left) + self._flatten_or(right):
            candidate = _extract_range_constraint(term, context)
            merged = False
            if candidate is not None:
                for index, current in enumerate(ranges):
                    if current is None:
                        continue
                    if not _same_column_ref(current.column, candidate.column):
                        continue
                    union = current.union(candidate)
                    if union is None:
                        continue
                    ranges[index] = union
                    merged_expr = union.to_expr()
                    if merged_expr is None:
                        # An unbounded range represents every non-NULL value,
                        # not SQL TRUE: comparisons still yield UNKNOWN for
                        # NULL.  Keep that three-valued-logic distinction
                        # explicit so a later ``column IS NULL`` term can
                        # correctly simplify the whole disjunction to TRUE.
                        merged_expr = IsNullExpr(union.column, True)
                    terms[index] = merged_expr
                    merged = True
                    break
            if not merged:
                terms.append(term)
                ranges.append(candidate)

        result = terms[0]
        for term in terms[1:]:
            result = _simplify_logical_expr(result, "OR", term, context)
        return result

    def _is_safe_row_predicate(self, expr: Expr) -> bool:
        """Accept only deterministic, current-row expression nodes."""
        if isinstance(expr, SemanticsExpr):
            return self._is_safe_row_predicate(expr.expr)
        if isinstance(expr, (Literal, ColumnRef)):
            return True
        if isinstance(expr, BinaryExpr):
            return self._is_safe_row_predicate(
                expr.left
            ) and self._is_safe_row_predicate(expr.right)
        if isinstance(expr, BetweenExpr):
            return all(
                self._is_safe_row_predicate(item)
                for item in (
                    expr.expr,
                    expr.lower,
                    expr.upper,
                )
            )
        if isinstance(expr, InExpr):
            return self._is_safe_row_predicate(expr.expr) and all(
                isinstance(value, Literal) for value in expr.values
            )
        if isinstance(expr, IsNullExpr):
            return self._is_safe_row_predicate(expr.expr)
        if isinstance(expr, NotExpr):
            return self._is_safe_row_predicate(expr.expr)
        # CASE and every function (including unknown UDFs) are deliberately
        # rejected: the rule cannot prove determinism/totality for them.
        return False

    def _is_safe_event_predicate(self, expr: Expr, event_vars: Set[str]) -> bool:
        """Accept deterministic predicates depending only on singleton events.

        Cross-variable column references among the retained singleton event
        variables are stable when gap-only rows are erased. Functions remain
        rejected because aggregates or physical navigation could observe the
        removed rows, while unknown scalar functions may be non-deterministic.
        """
        allowed_qualifiers = {name.upper() for name in event_vars}

        def visit(node: Expr) -> bool:
            if isinstance(node, SemanticsExpr):
                return visit(node.expr)
            if isinstance(node, Literal):
                return True
            if isinstance(node, ColumnRef):
                return (
                    node.qualifier is None
                    or node.qualifier.upper() in allowed_qualifiers
                )
            if isinstance(node, BinaryExpr):
                return visit(node.left) and visit(node.right)
            if isinstance(node, BetweenExpr):
                return visit(node.expr) and visit(node.lower) and visit(node.upper)
            if isinstance(node, InExpr):
                return visit(node.expr) and all(
                    isinstance(value, Literal) for value in node.values
                )
            if isinstance(node, IsNullExpr):
                return visit(node.expr)
            if isinstance(node, NotExpr):
                return visit(node.expr)
            return False

        return visit(expr)

    def _strip_var_qualifier(self, expr: Expr, var_name: str) -> Optional[Expr]:
        """Replace VAR.col references with unqualified col in expr."""
        return _strip_var_qualifier(expr, var_name)

    def _is_trivially_true(self, expr: Expr) -> bool:
        """Return True if expr is trivially 1 = 1."""
        if expr is None:
            return True
        if isinstance(expr, SemanticsExpr):
            return self._is_trivially_true(expr.expr)
        if isinstance(expr, BinaryExpr):
            if (
                expr.op == "="
                and isinstance(expr.left, Literal)
                and isinstance(expr.right, Literal)
            ):
                return expr.left.value == expr.right.value
        return False

    def _expr_references_any_var(self, expr: Expr, var_names: Set[str]) -> bool:
        """Return True if expr contains a ColumnRef whose qualifier is in var_names."""
        return _expr_references_any_var(expr, var_names)

    def _noise_dependent_var_names(
        self, ast: RPRAbstractSyntaxTree, noise_vars: Set[str]
    ) -> Set[str]:
        """Return noise variables and SUBSET names whose members include them."""
        result = set(noise_vars)
        subsets = ast.subsets or {}

        # ISO SUBSET members are primary variables, but using a fixed point is
        # inexpensive and keeps this safety check robust to nested parser input.
        changed = True
        while changed:
            changed = False
            for subset_name, members in subsets.items():
                if subset_name not in result and any(
                    member in result for member in members
                ):
                    result.add(subset_name)
                    changed = True
        return result

    def _expr_references_pattern_vars(self, expr: Expr, var_names: Set[str]) -> bool:
        """Detect qualified references and CLASSIFIER(pattern_variable)."""
        if self._expr_references_any_var(expr, var_names):
            return True

        def visit(node: Expr) -> bool:
            if node is None:
                return False
            if isinstance(node, SemanticsExpr):
                return visit(node.expr)
            if isinstance(node, FunctionCall):
                if node.name.upper() == "CLASSIFIER" and node.args:
                    arg = node.args[0]
                    if (
                        isinstance(arg, ColumnRef)
                        and arg.qualifier is None
                        and arg.name in var_names
                    ):
                        return True
                return any(visit(arg) for arg in node.args)
            if isinstance(node, BinaryExpr):
                return visit(node.left) or visit(node.right)
            if isinstance(node, BetweenExpr):
                return visit(node.expr) or visit(node.lower) or visit(node.upper)
            if isinstance(node, CaseExpr):
                return any(
                    visit(condition) or visit(result)
                    for condition, result in node.cases
                ) or visit(node.else_expr)
            if isinstance(node, InExpr):
                return visit(node.expr) or any(visit(value) for value in node.values)
            if isinstance(node, IsNullExpr):
                return visit(node.expr)
            if isinstance(node, NotExpr):
                return visit(node.expr)
            return False

        return visit(expr)

    def _measure_depends_on_removed_rows(
        self, expr: Expr, event_vars: Set[str]
    ) -> bool:
        """Return whether a measure can observe rows removed by the pre-filter.

        A row-set aggregate with an unqualified argument ranges over the whole
        match, which includes the otherwise "irrelevant" noise rows. Literal
        aggregates such as COUNT(1) have the same problem. An aggregate is
        safe only when its row set is explicitly restricted to retained event
        variables (for example COUNT(A.*)).

        PREV/NEXT are rejected conservatively because deleting an intervening
        noise row changes physical navigation. Unqualified FIRST/LAST with no
        offset remain safe here: this rule's pattern starts and ends with
        retained singleton event variables.
        """

        # currently, we only consider that SQL only have the below five functions
        aggregate_funcs = {"AVG", "SUM", "COUNT", "MAX", "MIN"}

        if has_physical_navigation(expr):
            return True

        def qualified_event_columns(node: Expr) -> Tuple[bool, bool]:
            """Return (has_event_column, has_other_or_unqualified_column)."""
            has_event = False
            has_other = False

            def visit_columns(current: Expr):
                nonlocal has_event, has_other
                if current is None:
                    return
                if isinstance(current, SemanticsExpr):
                    visit_columns(current.expr)
                elif isinstance(current, ColumnRef):
                    if current.qualifier in event_vars:
                        has_event = True
                    else:
                        has_other = True
                elif isinstance(current, FunctionCall):
                    for arg in current.args:
                        visit_columns(arg)
                elif isinstance(current, BinaryExpr):
                    visit_columns(current.left)
                    visit_columns(current.right)
                elif isinstance(current, BetweenExpr):
                    visit_columns(current.expr)
                    visit_columns(current.lower)
                    visit_columns(current.upper)
                elif isinstance(current, CaseExpr):
                    for condition, result in current.cases:
                        visit_columns(condition)
                        visit_columns(result)
                    visit_columns(current.else_expr)
                elif isinstance(current, InExpr):
                    visit_columns(current.expr)
                    for value in current.values:
                        visit_columns(value)
                elif isinstance(current, IsNullExpr):
                    visit_columns(current.expr)
                elif isinstance(current, NotExpr):
                    visit_columns(current.expr)

            visit_columns(node)
            return has_event, has_other

        def visit(node: Expr) -> bool:
            if node is None:
                return False
            if isinstance(node, SemanticsExpr):
                return visit(node.expr)
            if isinstance(node, FunctionCall):
                func_name = node.name.upper()
                if func_name in aggregate_funcs:
                    has_event, has_other = qualified_event_columns(node)
                    if not has_event or has_other:
                        return True
                if func_name in ("FIRST", "LAST") and len(node.args) > 1:
                    has_event, has_other = qualified_event_columns(node.args[0])
                    if not has_event or has_other:
                        return True
                return any(visit(arg) for arg in node.args)
            if isinstance(node, BinaryExpr):
                return visit(node.left) or visit(node.right)
            if isinstance(node, BetweenExpr):
                return visit(node.expr) or visit(node.lower) or visit(node.upper)
            if isinstance(node, CaseExpr):
                return any(
                    visit(condition) or visit(result)
                    for condition, result in node.cases
                ) or visit(node.else_expr)
            if isinstance(node, InExpr):
                return visit(node.expr) or any(visit(value) for value in node.values)
            if isinstance(node, IsNullExpr):
                return visit(node.expr)
            if isinstance(node, NotExpr):
                return visit(node.expr)
            return False

        return visit(expr)


row_filter_4_special_pattern_test_cases = [
    # Test 1 (from spec example 1): A {-N1*?-} B {-N2*-} C, ALL ROWS PER MATCH
    (
        1,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*?-} B {-N2*-} C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C' AND VOL < 10
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE ((TYPE = 'A' OR TYPE = 'B') OR (TYPE = 'C' AND VOL < 10))) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1*? -} B {- N2* -} C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C' AND VOL < 10
)""",
    ),
    # Test 2: A {-N1*?-} B, ALL ROWS PER MATCH
    (
        2,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*?-} B)
    DEFINE A AS A.TYPE = 'A' AND VOL < 10, B AS B.TYPE = 'B' AND B.VOL < 20
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE ((TYPE = 'A' AND VOL < 10) OR (TYPE = 'B' AND VOL < 20))) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1*? -} B)
    DEFINE A AS A.TYPE = 'A' AND VOL < 10, B AS B.TYPE = 'B' AND B.VOL < 20
)""",
    ),
    # Test 3: A {-N1*-} B {-N2*-} C {-N3*-} D, ALL ROWS PER MATCH
    (
        3,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*-} B {-N2*-} C {-N3*-} D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND B.VOL < 20, C AS TYPE = 'C', N3 AS 1 = 1, D AS TYPE = 'D'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (((TYPE = 'A' OR (TYPE = 'B' AND VOL < 20)) OR TYPE = 'C') OR TYPE = 'D')) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1* -} B {- N2* -} C {- N3* -} D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND B.VOL < 20, C AS TYPE = 'C', N3 AS 1 = 1, D AS TYPE = 'D'
)""",
    ),
    # Test 4: Simple two-var case, ALL ROWS PER MATCH with excluded noise
    (
        4,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*-} B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (TYPE = 'A' OR TYPE = 'B')) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1* -} B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
    ),
    # Test 5: Non-greedy {-N1*?-}, greedy {-N2*-}, ALL ROWS PER MATCH, BETWEEN condition
    (
        5,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*?-} B {-N2*-} C)
    DEFINE A AS A.VOL BETWEEN 10 AND 100, B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (((VOL BETWEEN 10 AND 100) OR TYPE = 'B') OR TYPE = 'C'))
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1*? -} B {- N2* -} C)
    DEFINE A AS A.VOL BETWEEN 10 AND 100, B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
    ),
    # Test 6: Four event variables (A N1* B N2* C N3* D), no N-var DEFINE entries
    (
        6,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N1* B N2* C N3* D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C', D AS D.TYPE = 'D'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (((TYPE = 'A' OR TYPE = 'B') OR TYPE = 'C') OR TYPE = 'D')) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N1* B N2* C N3* D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C', D AS D.TYPE = 'D'
)""",
    ),
    (
        7,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*-} B N2* C N3* D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C', D AS D.TYPE = 'D'
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE (((TYPE = 'A' OR TYPE = 'B') OR TYPE = 'C') OR TYPE = 'D')) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES FIRST(TS) AS FIRST_TS
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N1* B N2* C N3* D)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C', D AS D.TYPE = 'D'
)""",
    ),
    # Test 8: {-N1*-} and {-N2*-} not in DEFINE (implicit wildcard), ALL ROWS PER MATCH
    (
        8,
        """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*-} B {-N2*-} C)
    DEFINE A AS A.VOL > 50, B AS B.VOL > 100, C AS C.VOL > 200
)""",
        """
SELECT * FROM (SELECT * FROM TA WHERE ((VOL > 50 OR VOL > 100) OR VOL > 200)) MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {- N1* -} B {- N2* -} C)
    DEFINE A AS A.VOL > 50, B AS B.VOL > 100, C AS C.VOL > 200
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    table_schemas = {
        "T": {
            "columns": {
                "ID": "LONG",
                "PT": "LONG",
                "TS": "LONG",
                "TYPE": "VARCHAR(20)",
                "VAL": "DOUBLE",
            },
            "column_nullables": {
                "ID": False,
                "PT": False,
                "TS": False,
                "TYPE": False,
                "VAL": True,
            },
        },
    }
    return run_rule_tests(
        RowFilter4SpecialPatternRule,
        row_filter_4_special_pattern_test_cases,
        "row filter for special pattern",
        verbose,
        default_table_schemas=table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
