if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import copy
import re
from itertools import product
from typing import List, Set, Tuple

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
from expression_clause import DependentCondition, IndependentCondition
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
from queryanalysis import ONE_ROW_PER_MATCH, RPRAbstractSyntaxTree

from .base import RewriteRule


class MeasuresVariableRenamingRule(RewriteRule):
    """Avoid ambiguous quantified-variable references in one-row MEASURES.

    A scalar reference such as ``C.ID`` in ONE ROW PER MATCH denotes the last
    row classified as C.  When the last C can be isolated on every pattern
    path, this rule gives that occurrence a fresh name and updates MEASURES.
    It also resolves scalar/FIRST/LAST references to SUBSET variables whenever
    the first or last contributing primary variable is unique.

    The pattern rewrite is intentionally conservative.  A primary variable
    below a compound quantifier (for example ``(C | Z)+``), a dependent DEFINE
    condition, or an AFTER MATCH SKIP target is not renamed.
    """

    _AGGREGATE_FUNCS = {"SUM", "AVG", "COUNT", "MAX", "MIN"}
    _PROTECTED_FUNCS = _AGGREGATE_FUNCS | {"FIRST", "LAST", "PREV", "NEXT"}
    _MAX_EXPANDED_PATHS = 128

    @property
    def name(self) -> str:
        return "measures_variable_renaming"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        plan = self._build_plan(ast)
        if plan is None:
            return False

        subset_measures, renames = plan
        ast.measures.measures = subset_measures
        classifier_labels: List[Tuple[str, str]] = []

        for old_var, new_var, rewritten_pattern in renames:
            ast.pattern = rewritten_pattern
            ast.measures.measures = {
                alias: self._replace_unprotected_qualifier(expr, old_var, new_var)
                for alias, expr in ast.measures.measures.items()
            }

            old_condition = ast.define.definitions.get(old_var)
            if old_condition is not None:
                new_expr = self._replace_all_qualifiers(
                    old_condition.expression,
                    old_var,
                    new_var,
                )
                ast.define.definitions[new_var] = IndependentCondition(
                    new_var, new_expr
                )

            for members in (ast.subsets or {}).values():
                if old_var in members and new_var not in members:
                    members.append(new_var)
            classifier_labels.append((new_var, old_var))

        if classifier_labels:
            ast.measures.measures = {
                alias: self._preserve_classifier_labels(expr, classifier_labels)
                for alias, expr in ast.measures.measures.items()
            }
        return True

    def _build_plan(self, ast):
        if (
            ast.pattern is None
            or ast.measures is None
            or ast.define is None
            or (ast.rows_per_match or "").strip().upper() != ONE_ROW_PER_MATCH
        ):
            return None

        original_measures = ast.measures.get_all_measures()
        subset_measures = {
            alias: self._rewrite_subset_expr(expr, ast)
            for alias, expr in original_measures.items()
        }
        subset_changed = any(
            subset_measures[alias].to_string() != expr.to_string()
            for alias, expr in original_measures.items()
        )

        pattern = ast.pattern
        used_names = set(pattern.get_variables()) | set((ast.subsets or {}).keys())
        renames = []
        candidate_vars = sorted(self._unprotected_primary_refs(subset_measures, ast))
        for old_var in candidate_vars:
            if not self._has_direct_repeating_quantifier(pattern, old_var):
                continue
            if not self._safe_to_rename(ast, subset_measures, old_var):
                continue

            new_var = self._fresh_variable_name(used_names)
            rewritten = self._isolate_last_occurrence(pattern, old_var, new_var)
            if rewritten is None:
                continue
            pattern = rewritten
            used_names.add(new_var)
            renames.append((old_var, new_var, pattern))

        if not subset_changed and not renames:
            return None
        return subset_measures, renames

    def _unprotected_primary_refs(self, measures, ast) -> Set[str]:
        primary = set(ast.pattern.get_variables())
        result = set()
        for expr in measures.values():
            self._collect_unprotected_refs(expr, primary, result, False)
        return result

    def _collect_unprotected_refs(self, expr, names, result, protected):
        if expr is None:
            return
        if isinstance(expr, ColumnRef):
            if not protected and expr.qualifier in names:
                result.add(expr.qualifier)
            return
        if isinstance(expr, FunctionCall):
            nested = protected or expr.name.upper() in self._PROTECTED_FUNCS
            for arg in expr.args:
                self._collect_unprotected_refs(arg, names, result, nested)
            return
        for child in self._expr_children(expr):
            self._collect_unprotected_refs(child, names, result, protected)

    def _safe_to_rename(self, ast, measures, old_var):
        skip_tokens = set(
            re.findall(r"[A-Z_][A-Z0-9_$#]*", ast.after_match_skip_to or "")
        )
        if old_var in skip_tokens:
            return False

        condition = ast.define.definitions.get(old_var)
        if condition is not None and isinstance(condition, DependentCondition):
            return False

        # Other DEFINE predicates would observe a different C history after
        # the last C is renamed.  SUBSET histories are preserved separately by
        # adding the fresh variable to every affected subset.
        for var_name, other in ast.define.definitions.items():
            if var_name == old_var or other is None:
                continue
            if self._expr_has_qualifier(other.expression, old_var):
                return False

        # An aggregate/navigation over C must continue to see all C rows.  We
        # cannot express C union fresh-V without introducing a new SUBSET, so
        # do not perform this independent scalar optimization in that case.
        for expr in measures.values():
            if self._has_protected_reference(expr, old_var, False):
                return False
        return True

    def _has_protected_reference(self, expr, var_name, protected):
        if expr is None:
            return False
        if isinstance(expr, ColumnRef):
            return protected and expr.qualifier == var_name
        if isinstance(expr, FunctionCall):
            nested = protected or expr.name.upper() in self._PROTECTED_FUNCS
            return any(
                self._has_protected_reference(arg, var_name, nested)
                for arg in expr.args
            )
        return any(
            self._has_protected_reference(child, var_name, protected)
            for child in self._expr_children(expr)
        )

    def _has_direct_repeating_quantifier(self, node, var_name):
        if isinstance(node, QuantifiedPattern):
            if isinstance(node.pattern, VariablePattern):
                return node.pattern.name == var_name and (
                    node.atMost is None or node.atMost > 1
                )
            repeated_by_this_quantifier = var_name in node.pattern.get_variables() and (
                node.atMost is None or node.atMost > 1
            )
            return repeated_by_this_quantifier or self._has_direct_repeating_quantifier(
                node.pattern,
                var_name,
            )
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._has_direct_repeating_quantifier(node.pattern, var_name)
        if isinstance(
            node, (AlternationPattern, ConcatenationPattern, PermutationPattern)
        ):
            return any(
                self._has_direct_repeating_quantifier(p, var_name)
                for p in node.patterns
            )
        return False

    # Each variant is (pattern, contains-target).  A present variant already
    # renames its last target to new_var.  Concatenation restores earlier
    # variants to old_var, leaving only the rightmost occurrence renamed.
    def _isolate_last_occurrence(self, pattern, old_var, new_var):
        variants = self._expand_variants(pattern, old_var, new_var)
        if variants is None or len(variants) > self._MAX_EXPANDED_PATHS:
            return None
        patterns = [self._safe_simplify(p) for p, _ in variants]
        return self._factor_common_edges(patterns)

    def _expand_variants(self, node, old_var, new_var):
        if isinstance(node, VariablePattern):
            if node.name == old_var:
                return [(VariablePattern(new_var), True)]
            return [(copy.deepcopy(node), False)]

        if isinstance(node, QuantifiedPattern):
            if (
                isinstance(node.pattern, VariablePattern)
                and node.pattern.name == old_var
            ):
                prefix_max = None if node.atMost is None else node.atMost - 1
                prefix_min = max(0, node.atLeast - 1)
                pieces = []
                if prefix_max is None or prefix_max > 0:
                    pieces.append(
                        QuantifiedPattern(
                            VariablePattern(old_var),
                            prefix_min,
                            prefix_max,
                            node.isGreedy,
                        )
                    )
                pieces.append(VariablePattern(new_var))
                present = self._concat(pieces)
                if node.atLeast > 0:
                    return [(present, True)]
                absent = (EmptyPattern(), False)
                return (
                    [(present, True), absent]
                    if node.isGreedy
                    else [absent, (present, True)]
                )

            if old_var in node.pattern.get_variables():
                # When every iteration of the quantified body must contain
                # old_var, its final occurrence is necessarily in the final
                # iteration.  Split Q(X) into Q-1(X) followed by one rewritten
                # X, then isolate the last old_var recursively inside that X.
                #
                #     (C+ D+)+  =>  (C+ D+)* (C* V D+)
                #
                # This proof does not hold for (C | Z)+ because an iteration
                # may contain no C.  Such compound quantifiers remain
                # deliberately unsupported.
                if old_var not in node.pattern.get_must_appear_var():
                    return None

                rewritten_bodies = self._expand_variants(
                    node.pattern,
                    old_var,
                    new_var,
                )
                if rewritten_bodies is None or any(
                    not present for _, present in rewritten_bodies
                ):
                    return None

                prefix_max = None if node.atMost is None else node.atMost - 1
                prefix_min = max(0, node.atLeast - 1)
                prefix = None
                if prefix_max is None or prefix_max > 0:
                    prefix = QuantifiedPattern(
                        copy.deepcopy(node.pattern),
                        prefix_min,
                        prefix_max,
                        node.isGreedy,
                    )

                present_variants = [
                    (
                        self._concat(
                            ([copy.deepcopy(prefix)] if prefix is not None else [])
                            + [body]
                        ),
                        True,
                    )
                    for body, _ in rewritten_bodies
                ]
                if node.atLeast > 0:
                    return present_variants

                absent = (EmptyPattern(), False)
                return (
                    present_variants + [absent]
                    if node.isGreedy
                    else [absent] + present_variants
                )
            return [(copy.deepcopy(node), False)]

        if isinstance(node, AlternationPattern):
            result = []
            for branch in node.patterns:
                expanded = self._expand_variants(branch, old_var, new_var)
                if expanded is None:
                    return None
                result.extend(expanded)
                if len(result) > self._MAX_EXPANDED_PATHS:
                    return None
            return result

        if isinstance(node, ConcatenationPattern):
            expanded_children = []
            guaranteed_positions = [
                index
                for index, child in enumerate(node.patterns)
                if old_var in child.get_must_appear_var()
            ]
            # Once a suffix element is guaranteed to contain old_var, no
            # occurrence in an earlier element can be the last one.  Leaving
            # that prefix untouched is both cheaper and important after a
            # previous variable has already split a compound quantifier.
            first_relevant_index = (
                max(guaranteed_positions) if guaranteed_positions else 0
            )
            for index, child in enumerate(node.patterns):
                if index < first_relevant_index:
                    expanded = [(copy.deepcopy(child), False)]
                else:
                    expanded = self._expand_variants(child, old_var, new_var)
                if expanded is None:
                    return None
                expanded_children.append(expanded)
            count = 1
            for choices in expanded_children:
                count *= len(choices)
                if count > self._MAX_EXPANDED_PATHS:
                    return None

            result = []
            for choices in product(*expanded_children):
                last_present = max(
                    (index for index, (_, present) in enumerate(choices) if present),
                    default=-1,
                )
                rebuilt = []
                for index, (child, present) in enumerate(choices):
                    if present and index != last_present:
                        child = self._rename_pattern_var(child, new_var, old_var)
                    rebuilt.append(child)
                result.append((self._concat(rebuilt), last_present >= 0))
            return result

        if isinstance(node, (AnchorPattern, ExcludedPattern, PermutationPattern)):
            if old_var in node.get_variables():
                return None
            return [(copy.deepcopy(node), False)]
        if isinstance(node, EmptyPattern):
            return [(EmptyPattern(), False)]
        return None

    def _factor_common_edges(self, patterns):
        unique = []
        seen = set()
        for pattern in patterns:
            key = pattern.to_string()
            if key not in seen:
                unique.append(pattern)
                seen.add(key)
        if len(unique) == 1:
            return unique[0]

        rows = [
            (
                list(p.patterns)
                if isinstance(p, ConcatenationPattern)
                else ([] if isinstance(p, EmptyPattern) else [p])
            )
            for p in unique
        ]
        suffix = []
        while all(rows) and len({row[-1].to_string() for row in rows}) == 1:
            suffix.insert(0, rows[0][-1])
            for row in rows:
                row.pop()
        prefix = []
        # Keep an optional branch explicit as ``PRESENT | PREFIX`` rather
        # than factoring it into ``PREFIX (PRESENT | ())``.  The general
        # pattern simplifier deliberately folds the latter to ``...?``;
        # retaining non-empty branches both avoids () in emitted SQL and
        # preserves the branch shape needed by this rule.
        common_prefix_len = 0
        while (
            all(len(row) > common_prefix_len for row in rows)
            and len({row[common_prefix_len].to_string() for row in rows}) == 1
        ):
            common_prefix_len += 1
        if all(len(row) > common_prefix_len for row in rows):
            prefix = rows[0][:common_prefix_len]
            rows = [row[common_prefix_len:] for row in rows]

        alternatives = [self._concat(row) for row in rows]
        middle = AlternationPattern(alternatives)
        return self._concat(prefix + [middle] + suffix)

    def _concat(self, patterns):
        flat = []
        for pattern in patterns:
            if isinstance(pattern, EmptyPattern):
                continue
            if isinstance(pattern, ConcatenationPattern):
                flat.extend(pattern.patterns)
            else:
                flat.append(pattern)
        if not flat:
            return EmptyPattern()
        if len(flat) == 1:
            return flat[0]
        return ConcatenationPattern(flat)

    def _safe_simplify(self, pattern):
        if isinstance(pattern, AlternationPattern):
            return pattern
        try:
            return pattern.simplify()
        except ValueError:
            return pattern

    def _rename_pattern_var(self, node, old_var, new_var):
        if isinstance(node, VariablePattern):
            return VariablePattern(new_var if node.name == old_var else node.name)
        if isinstance(node, QuantifiedPattern):
            return QuantifiedPattern(
                self._rename_pattern_var(node.pattern, old_var, new_var),
                node.atLeast,
                node.atMost,
                node.isGreedy,
            )
        if isinstance(node, AlternationPattern):
            return AlternationPattern(
                [self._rename_pattern_var(p, old_var, new_var) for p in node.patterns]
            )
        if isinstance(node, ConcatenationPattern):
            return ConcatenationPattern(
                [self._rename_pattern_var(p, old_var, new_var) for p in node.patterns]
            )
        return copy.deepcopy(node)

    def _fresh_variable_name(self, used):
        if "V" not in used:
            return "V"
        index = 1
        while f"V{index}" in used:
            index += 1
        return f"V{index}"

    # ------------------------- SUBSET resolution -------------------------
    def _rewrite_subset_expr(self, expr, ast, protected=False):
        subsets = ast.subsets or {}
        if isinstance(expr, ColumnRef):
            if not protected and expr.qualifier in subsets:
                source = self._unique_subset_edge(
                    ast.pattern, set(subsets[expr.qualifier]), "last"
                )
                if source is not None:
                    return ColumnRef(source, expr.name)
            return copy.deepcopy(expr)
        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if (
                name in {"FIRST", "LAST"}
                and len(expr.args) == 1
                and isinstance(expr.args[0], ColumnRef)
            ):
                column = expr.args[0]
                if column.qualifier in subsets:
                    edge = "first" if name == "FIRST" else "last"
                    source = self._unique_subset_edge(
                        ast.pattern, set(subsets[column.qualifier]), edge
                    )
                    if source is not None:
                        replacement = ColumnRef(source, column.name)
                        occurrence_count = self._max_occurrences(ast.pattern, source)
                        if name == "LAST" or (
                            occurrence_count is not None and occurrence_count <= 1
                        ):
                            return replacement
                        return FunctionCall(expr.name, [replacement])
            nested = protected or name in self._PROTECTED_FUNCS
            return FunctionCall(
                expr.name,
                [self._rewrite_subset_expr(a, ast, nested) for a in expr.args],
            )
        return self._map_expr(
            expr, lambda child: self._rewrite_subset_expr(child, ast, protected)
        )

    def _unique_subset_edge(self, pattern, members, direction):
        possible, _ = self._subset_edge(pattern, members, direction)
        return next(iter(possible)) if len(possible) == 1 else None

    def _subset_edge(self, node, members, direction):
        if isinstance(node, VariablePattern):
            return ({node.name}, False) if node.name in members else (set(), True)
        if isinstance(node, EmptyPattern):
            return set(), True
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._subset_edge(node.pattern, members, direction)
        if isinstance(node, QuantifiedPattern):
            edges, absent = self._subset_edge(node.pattern, members, direction)
            return edges, node.atLeast == 0 or absent
        if isinstance(node, AlternationPattern):
            values = [self._subset_edge(p, members, direction) for p in node.patterns]
            return set().union(*(edge for edge, _ in values)), any(
                absent for _, absent in values
            )
        if isinstance(node, ConcatenationPattern):
            children = (
                node.patterns if direction == "first" else reversed(node.patterns)
            )
            result = set()
            absent_so_far = True
            for child in children:
                if not absent_so_far:
                    break
                edge, absent = self._subset_edge(child, members, direction)
                result.update(edge)
                absent_so_far = absent_so_far and absent
            return result, absent_so_far
        if isinstance(node, PermutationPattern):
            values = [self._subset_edge(p, members, direction) for p in node.patterns]
            return set().union(*(edge for edge, _ in values)), all(
                absent for _, absent in values
            )
        return set(), True

    def _max_occurrences(self, node, var_name):
        if isinstance(node, VariablePattern):
            return 1 if node.name == var_name else 0
        if isinstance(node, EmptyPattern):
            return 0
        if isinstance(node, (AnchorPattern, ExcludedPattern)):
            return self._max_occurrences(node.pattern, var_name)
        if isinstance(node, QuantifiedPattern):
            inner = self._max_occurrences(node.pattern, var_name)
            if inner is None or (node.atMost is None and inner):
                return None
            return inner * (node.atMost or 0)
        if isinstance(node, (ConcatenationPattern, PermutationPattern)):
            values = [self._max_occurrences(p, var_name) for p in node.patterns]
            return None if any(v is None for v in values) else sum(values)
        if isinstance(node, AlternationPattern):
            values = [self._max_occurrences(p, var_name) for p in node.patterns]
            return None if any(v is None for v in values) else max(values, default=0)
        return None

    # ------------------------- expression helpers -----------------------
    def _expr_children(self, expr):
        if isinstance(expr, SemanticsExpr):
            return [expr.expr]
        if isinstance(expr, BinaryExpr):
            return [expr.left, expr.right]
        if isinstance(expr, BetweenExpr):
            return [expr.expr, expr.lower, expr.upper]
        if isinstance(expr, CaseExpr):
            return [item for pair in expr.cases for item in pair] + (
                [expr.else_expr] if expr.else_expr else []
            )
        if isinstance(expr, InExpr):
            return [expr.expr] + list(expr.values)
        if isinstance(expr, (IsNullExpr, NotExpr)):
            return [expr.expr]
        return []

    def _expr_has_qualifier(self, expr, qualifier):
        if isinstance(expr, ColumnRef):
            return expr.qualifier == qualifier
        if isinstance(expr, FunctionCall):
            return any(self._expr_has_qualifier(a, qualifier) for a in expr.args)
        return any(
            self._expr_has_qualifier(child, qualifier)
            for child in self._expr_children(expr)
        )

    def _replace_unprotected_qualifier(self, expr, old_var, new_var, protected=False):
        if isinstance(expr, ColumnRef):
            if not protected and expr.qualifier == old_var:
                return ColumnRef(new_var, expr.name)
            return copy.deepcopy(expr)
        if isinstance(expr, FunctionCall):
            nested = protected or expr.name.upper() in self._PROTECTED_FUNCS
            return FunctionCall(
                expr.name,
                [
                    self._replace_unprotected_qualifier(a, old_var, new_var, nested)
                    for a in expr.args
                ],
            )
        return self._map_expr(
            expr,
            lambda child: self._replace_unprotected_qualifier(
                child, old_var, new_var, protected
            ),
        )

    def _replace_all_qualifiers(self, expr, old_var, new_var):
        if isinstance(expr, ColumnRef):
            return ColumnRef(
                new_var if expr.qualifier == old_var else expr.qualifier, expr.name
            )
        if isinstance(expr, FunctionCall):
            return FunctionCall(
                expr.name,
                [self._replace_all_qualifiers(a, old_var, new_var) for a in expr.args],
            )
        return self._map_expr(
            expr, lambda child: self._replace_all_qualifiers(child, old_var, new_var)
        )

    def _preserve_classifier_labels(self, expr, labels):
        if isinstance(expr, FunctionCall):
            if expr.name.upper() == "CLASSIFIER" and not expr.args:
                classifier = FunctionCall(expr.name, [])
                cases = [
                    (
                        BinaryExpr(copy.deepcopy(classifier), "=", Literal(new)),
                        Literal(old),
                    )
                    for new, old in labels
                ]
                return CaseExpr(cases, classifier)
            return FunctionCall(
                expr.name,
                [self._preserve_classifier_labels(a, labels) for a in expr.args],
            )
        return self._map_expr(
            expr, lambda child: self._preserve_classifier_labels(child, labels)
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
                [(transform(c), transform(r)) for c, r in expr.cases],
                transform(expr.else_expr) if expr.else_expr else None,
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


measures_variable_renaming_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, C.ID AS CID, D.ID AS DID, CLASSIFIER() AS TAG
    PATTERN(A B+ C+ D)
    DEFINE B AS VAL > 10, C AS VAL < 100, D AS VAL = 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, V.ID AS CID, D.ID AS DID, CASE WHEN CLASSIFIER() = 'V' THEN 'C' ELSE CLASSIFIER() END AS TAG
    PATTERN(A B+ C* V D)
    DEFINE B AS VAL > 10, C AS VAL < 100, D AS VAL = 5, V AS VAL < 100
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, C.ID AS CID, D.ID AS DID
    PATTERN(A B+ C* D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, V.ID AS CID, D.ID AS DID
    PATTERN((A B+ C* V | A B+) D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5, V AS V.VAL < 100
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, C.ID AS CID, D.ID AS DID
    PATTERN(A B+ C (C+ | Z+) D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, V.ID AS CID, D.ID AS DID
    PATTERN(A B+ (C+ V | V Z+) D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5, V AS V.VAL < 100
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES E1.ID AS XID, FIRST(E2.ID) AS YID
    PATTERN(A B+ C+ D)
    SUBSET E1 = (C, D), E2 = (A, D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5 AND SUM(E2.VAL) > 30
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES D.ID AS XID, A.ID AS YID
    PATTERN(A B+ C+ D)
    SUBSET E1 = (C, D), E2 = (A, D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5 AND SUM(E2.VAL) > 30
)""",
    ),
    (
        5,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, C.ID AS CID, D.ID AS DID
    PATTERN(A B+ (C | Z)+ D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    MEASURES A.ID AS AID, C.ID AS CID, D.ID AS DID
    PATTERN(A B+ (C | Z)+ D)
    DEFINE B AS B.VAL > 10, C AS C.VAL < 100, D AS D.VAL = 5
)""",
    ),
]


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        MeasuresVariableRenamingRule,
        measures_variable_renaming_test_cases,
        "MEASURES variable renaming",
        verbose,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
