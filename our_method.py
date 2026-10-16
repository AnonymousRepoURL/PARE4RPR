"""Run the complete rule-based RPR rewrite pipeline.

The module keeps the rule order explicit because an earlier rewrite may expose
or invalidate a later one. Rules which delete input rows are disabled for
patterns that can match zero rows and for ``ALL ROWS PER MATCH WITH UNMATCHED
ROWS`` queries.

Three rules have configurable sampling behaviour:

* ``MustAppearPartitionFilterPushdownRule`` is enabled by default. With
  ``--sample-must-appear-partition-filter`` it is adopted only when its
  lightweight sample filter removes at least one whole partition and more than
  a configurable fraction of sample rows (20% by default). It is disabled when
  schema metadata marks any ``PARTITION BY`` column nullable, or when
  ``AnchorPartitionSizeFilterRule`` has already been applied. The full
  ``MATCH_RECOGNIZE`` query is never executed for this decision.
* ``WindowConditionPreFilterRule`` is disabled unless
  ``--sample-window-condition-prefilter`` is supplied. When enabled, sample
  statistics generate row-reducing candidates for the same benchmark.
* ``ColumnPruningViaJoinRule`` is disabled for Trino. For Oracle it is applied
  directly for ``PAST LAST ROW``. Other skip strategies can produce
  overlapping matches, so the rewrite is adopted only when the original query
  emits at most 1.5 rows per sample input row. Supplying ``--sample-table``
  enables this cardinality check.

Window-prefilter candidate executions have a five-second default budget. If
any candidate completes, the fastest completed candidate wins. If all
candidates time out, the candidate that fetched the most rows within the
budget wins. The first completed single- or pair-variable candidate establishes
a dynamic baseline. Until then pair candidates keep using the configured query
timeout. A later faster candidate tightens the budget to at most twice the new
baseline and prunes cached candidates which no longer fit that budget. Every
candidate is measured only once.

The SQL output is a directly executable text file. A JSON sidecar records
per-query and per-rule local rewrite time, database sampling time, decisions,
and candidate measurements. SQL rendering defaults to Oracle syntax, replacing
numeric ``LIMIT n`` clauses with ``FETCH FIRST n ROWS ONLY``; pass
``--system trino`` to retain ``LIMIT n``.  Immediately before
``RegexOptimizationPlusRule``, numeric PATTERN quantifier bounds above 180 are
restored or split into equivalent portable chunks so the regex optimizer can
see the portable pattern.  A final defensive pass catches bounds introduced
after that point.  A DEFINE condition encoded into a restored finite range is
retained so the output stays equivalent; other successful quantifier rewrites
remain in the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from clause_parser import ExprParser
from patternnode import (
    ConcatenationPattern,
    PatternLexer,
    PatternParser,
    QuantifiedPattern,
    VariablePattern,
)
from queryanalysis import (
    ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS,
    PAST_LAST_ROW,
    RPRAbstractSyntaxTree,
)
from rewriting_rules import (
    AlternationVariableMergeRule,
    AnchorPartitionSizeFilterRule,
    AnchorTopKFilterRule,
    ColumnPruningRule,
    ColumnPruningViaJoinRule,
    CountRangePushdownRule,
    DefineIndependentConstraintPropagationRule,
    DefinePartitionPushdownRule,
    ForwardAdjustedConditionRule,
    ImpliedIncrementalConditionRemovalRule,
    IncrementalConditionPushdownRule,
    MeasuresVariableRenamingRule,
    MatchNumberGuardPushdownRule,
    MustAppearPartitionFilterPushdownRule,
    MustAppearVarFalsePushdownRule,
    NestedQuantifierRangeSimplificationRule,
    OrderedRangePushdownRule,
    QuantifierScopeShorteningRule,
    RedundantSubsetVariableRemovalRule,
    RegexOptimizationPlusRule,
    RemoveAggregationFuncRule,
    RewriteRule,
    RowFilter4SpecialPatternRule,
    RowFilteringViaWindowFuncRule,
    SingleVariablePatternPushdownRule,
    WherePartitionPushdownRule,
    WindowConditionPreFilterRule,
    WindowPrefilterCandidate,
    RegexOptimizationUltraRule,
)
from rewriting_rules.utils import bind_input_relation_alias, resolve_input_rebind_alias

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RUNS = 3
DEFAULT_FETCH_SIZE = 2_000
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION = 0.20
# The more columns there are, the higher this value can be set.
# 1.5 v.s. 1.2
DEFAULT_COLUMN_PRUNING_VIA_JOIN_MAX_OUTPUT_RATIO = 1.2
PAIR_CANDIDATE_TIMEOUT_MULTIPLIER = 2.0
MAX_PORTABLE_QUANTIFIER_BOUND = 180
DEFAULT_SYSTEM = "oracle"
SUPPORTED_SYSTEMS = ("oracle", "trino")


def _normalize_system(system: str) -> str:
    normalized = str(system).strip().lower()
    if normalized not in SUPPORTED_SYSTEMS:
        supported = ", ".join(SUPPORTED_SYSTEMS)
        raise ValueError(f"system must be one of: {supported}")
    return normalized


def _oracle_limit_syntax(sql: str) -> str:
    """Convert numeric LIMIT clauses without touching literals or comments."""

    limit_pattern = re.compile(r"\bLIMIT\s+([0-9]+)\b", re.IGNORECASE)
    result: list[str] = []
    index = 0
    length = len(sql)

    while index < length:
        if sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            if end == -1:
                result.append(sql[index:])
                break
            result.append(sql[index : end + 1])
            index = end + 1
            continue

        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end == -1:
                result.append(sql[index:])
                break
            result.append(sql[index : end + 2])
            index = end + 2
            continue

        quote = sql[index] if sql[index] in {"'", '"'} else None
        if quote is not None:
            end = index + 1
            while end < length:
                if sql[end] != quote:
                    end += 1
                    continue
                if end + 1 < length and sql[end + 1] == quote:
                    end += 2
                    continue
                end += 1
                break
            result.append(sql[index:end])
            index = end
            continue

        match = limit_pattern.match(sql, index)
        if match is not None:
            result.append(f"FETCH FIRST {match.group(1)} ROWS ONLY")
            index = match.end()
            continue

        result.append(sql[index])
        index += 1

    return "".join(result)


def _sql_for_system(sql: str, system: str) -> str:
    return _oracle_limit_syntax(sql) if system == "oracle" else sql


def _parse_unsimplified_input_pattern(sql: str, ast: RPRAbstractSyntaxTree):
    """Parse PATTERN without invoking PatternNode.simplify()."""

    match_pos = sql.find("MATCH_RECOGNIZE")
    start_pos = sql.find("(", match_pos + len("MATCH_RECOGNIZE"))
    end_pos = ast._find_matching_paren(sql, start_pos)
    if match_pos < 0 or start_pos < 0 or end_pos < 0:
        return None
    match_sql = sql[start_pos + 1 : end_pos]
    positions = ast._find_match_clause_positions(match_sql)
    clause_text = ast._get_clause_text(match_sql, positions, "PATTERN")
    if clause_text is None:
        return None
    pattern_start = clause_text.find("(")
    pattern_end = ast._find_matching_paren(clause_text, pattern_start)
    if pattern_start < 0 or pattern_end < 0:
        return None
    parser = PatternParser(PatternLexer(clause_text[pattern_start + 1 : pattern_end]))
    pattern = parser._parse_alternation()
    if getattr(parser.current_token.type, "name", None) != "EOF":
        return None
    return pattern


def _repair_numeric_quantifier_bounds(
    pattern,
    original_pattern,
    limit: int = MAX_PORTABLE_QUANTIFIER_BOUND,
) -> tuple[Any, list[dict[str, Any]]]:
    """Repair only numeric bounds above ``limit`` in the output pattern.

    Finite bounds produced by a rule recover the corresponding original
    quantifier when that occurrence is unambiguous.  A parser-merged range is
    split into equivalent chunks instead; for example, ``B{200,}`` becomes
    ``B{180} B{20,}``.
    """

    if pattern is None or original_pattern is None:
        return pattern, []
    repairs: list[dict[str, Any]] = []
    original_by_variable: dict[str, list[QuantifiedPattern]] = {}

    def collect_original(node) -> None:
        if isinstance(node, QuantifiedPattern) and isinstance(
            node.pattern, VariablePattern
        ):
            original_by_variable.setdefault(node.pattern.name, []).append(node)
        child = getattr(node, "pattern", None)
        if child is not None:
            collect_original(child)
        for item in getattr(node, "patterns", ()):
            collect_original(item)

    collect_original(original_pattern)

    def numeric_bound_is_oversized(node) -> bool:
        return isinstance(node, QuantifiedPattern) and (
            node.atLeast > limit or (node.atMost is not None and node.atMost > limit)
        )

    def original_bounds_are_portable(node) -> bool:
        return isinstance(node, QuantifiedPattern) and (
            node.atLeast <= limit and (node.atMost is None or node.atMost <= limit)
        )

    def restore_bounds(node, original, path: str, action: str) -> None:
        repairs.append(
            {
                "path": path,
                "variable": (
                    node.pattern.name
                    if isinstance(node.pattern, VariablePattern)
                    else None
                ),
                "action": action,
                "rewritten_minimum": node.atLeast,
                "rewritten_maximum": node.atMost,
                "output_minimum": original.atLeast,
                "output_maximum": original.atMost,
            }
        )
        node.atLeast = original.atLeast
        node.atMost = original.atMost
        node.isGreedy = original.isGreedy

    def visit(rewritten, original, path: str):
        if isinstance(rewritten, QuantifiedPattern):
            original_quantifier = None
            if isinstance(original, QuantifiedPattern):
                rewritten_variable = (
                    rewritten.pattern.name
                    if isinstance(rewritten.pattern, VariablePattern)
                    else None
                )
                original_variable = (
                    original.pattern.name
                    if isinstance(original.pattern, VariablePattern)
                    else None
                )
                compound_bodies_correspond = (
                    rewritten_variable is None
                    and original_variable is None
                    and rewritten.pattern.get_variables()
                    == original.pattern.get_variables()
                    # Nested quantifier flattening changes which quantifier
                    # owns the range.  In that case the outer raw quantifier
                    # is not the rewritten node's local counterpart.
                    and not (
                        isinstance(original.pattern, QuantifiedPattern)
                        and not isinstance(rewritten.pattern, QuantifiedPattern)
                    )
                )
                if (compound_bodies_correspond) or (
                    rewritten_variable is not None
                    and rewritten_variable == original_variable
                ):
                    original_quantifier = original
            original_child = (
                original_quantifier.pattern if original_quantifier is not None else None
            )
            if original_child is not None:
                rewritten.pattern = visit(
                    rewritten.pattern, original_child, f"{path}.pattern"
                )

            if (
                rewritten.atMost is not None
                and numeric_bound_is_oversized(rewritten)
                and original_bounds_are_portable(original_quantifier)
            ):
                restore_bounds(
                    rewritten,
                    original_quantifier,
                    path,
                    "restore_original_finite_range",
                )
                return rewritten

        rewritten_child = getattr(rewritten, "pattern", None)
        original_child = getattr(original, "pattern", None)
        if (
            not isinstance(rewritten, QuantifiedPattern)
            and rewritten_child is not None
            and original_child is not None
            and type(rewritten) is type(original)
            and getattr(rewritten, "anchor_type", None)
            == getattr(original, "anchor_type", None)
        ):
            rewritten.pattern = visit(
                rewritten_child, original_child, f"{path}.pattern"
            )

        rewritten_children = getattr(rewritten, "patterns", None)
        original_children = getattr(original, "patterns", None)
        if (
            rewritten_children is not None
            and original_children is not None
            and type(rewritten) is type(original)
            and len(rewritten_children) == len(original_children)
        ):
            rewritten.patterns = [
                visit(child, original_child, f"{path}.patterns[{index}]")
                for index, (child, original_child) in enumerate(
                    zip(rewritten_children, original_children)
                )
            ]
        return rewritten

    pattern = visit(pattern, original_pattern, "pattern")

    def finish(node, path: str):
        if isinstance(node, QuantifiedPattern):
            node.pattern = finish(node.pattern, f"{path}.pattern")
            if node.atMost is not None and numeric_bound_is_oversized(node):
                variable = (
                    node.pattern.name
                    if isinstance(node.pattern, VariablePattern)
                    else None
                )
                candidates = original_by_variable.get(variable, [])
                portable_bounds = {
                    (candidate.atLeast, candidate.atMost, candidate.isGreedy)
                    for candidate in candidates
                    if original_bounds_are_portable(candidate)
                }
                # A single original occurrence gives an unambiguous local
                # rollback.  Several adjacent equal-looking occurrences may
                # have been merged by PatternNode.simplify(); restoring just
                # one of them would lower the accepted count and is invalid.
                if (
                    variable is not None
                    and len(candidates) == 1
                    and len(portable_bounds) == 1
                ):
                    original = copy.deepcopy(candidates[0])
                    (
                        original.atLeast,
                        original.atMost,
                        original.isGreedy,
                    ) = next(iter(portable_bounds))
                    if original_bounds_are_portable(original):
                        restore_bounds(
                            node,
                            original,
                            path,
                            "restore_original_finite_range_by_variable",
                        )
            if numeric_bound_is_oversized(node):
                remaining_minimum = node.atLeast
                remaining_maximum = node.atMost
                chunks: list[QuantifiedPattern] = []
                while remaining_minimum > limit:
                    chunks.append(
                        QuantifiedPattern(
                            copy.deepcopy(node.pattern), limit, limit, True
                        )
                    )
                    remaining_minimum -= limit
                    if remaining_maximum is not None:
                        remaining_maximum -= limit

                while remaining_maximum is not None and remaining_maximum > limit:
                    chunks.append(
                        QuantifiedPattern(
                            copy.deepcopy(node.pattern),
                            remaining_minimum,
                            limit,
                            node.isGreedy,
                        )
                    )
                    remaining_minimum = 0
                    remaining_maximum -= limit

                if remaining_maximum is None or remaining_maximum > 0:
                    chunks.append(
                        QuantifiedPattern(
                            copy.deepcopy(node.pattern),
                            remaining_minimum,
                            remaining_maximum,
                            node.isGreedy,
                        )
                    )
                repairs.append(
                    {
                        "path": path,
                        "variable": (
                            node.pattern.name
                            if isinstance(node.pattern, VariablePattern)
                            else None
                        ),
                        "action": (
                            "split_unbounded_minimum"
                            if node.atMost is None
                            else "split_finite_range"
                        ),
                        "rewritten_minimum": node.atLeast,
                        "rewritten_maximum": node.atMost,
                        "chunks": [[item.atLeast, item.atMost] for item in chunks],
                    }
                )
                return ConcatenationPattern(chunks)
            return node

        child = getattr(node, "pattern", None)
        if child is not None:
            node.pattern = finish(child, f"{path}.pattern")
        children = getattr(node, "patterns", None)
        if children is not None:
            node.patterns = [
                finish(child, f"{path}.patterns[{index}]")
                for index, child in enumerate(children)
            ]
        return node

    return finish(pattern, "pattern"), repairs


def _finite_oversized_quantifier_variables(
    pattern, limit: int = MAX_PORTABLE_QUANTIFIER_BOUND
) -> set[str | None]:
    """Return direct variables whose finite range exceeds ``limit``."""

    result: set[str | None] = set()

    def visit(node) -> None:
        if (
            isinstance(node, QuantifiedPattern)
            and node.atMost is not None
            and (node.atLeast > limit or node.atMost > limit)
        ):
            result.add(
                node.pattern.name if isinstance(node.pattern, VariablePattern) else None
            )
        child = getattr(node, "pattern", None)
        if child is not None:
            visit(child)
        for item in getattr(node, "patterns", ()):
            visit(item)

    if pattern is not None:
        visit(pattern)
    return result


def _changed_define_conditions(before, after) -> dict[str, Any]:
    """Keep pre-rule DEFINE conditions changed by CountRangePushdownRule."""

    changed: dict[str, Any] = {}
    before_definitions = before.define.definitions
    after_definitions = after.define.definitions
    for name, condition in before_definitions.items():
        after_condition = after_definitions.get(name)
        before_sql = condition.expression.to_string()
        after_sql = (
            after_condition.expression.to_string()
            if after_condition is not None
            else None
        )
        if before_sql != after_sql:
            changed[name] = copy.deepcopy(condition)
    return changed


def _restore_define_guards(
    ast: RPRAbstractSyntaxTree, guards: Mapping[str, Any]
) -> list[str]:
    """Conjoin pre-count conditions needed after a quantifier rollback."""

    parser = ExprParser(ast.expression_simplify_context())
    restored: list[str] = []
    for name, guard in guards.items():
        current = ast.define.definitions.get(name)
        if current is None or current.expression is None:
            continue
        current_sql = current.expression.to_string()
        guard_sql = guard.expression.to_string()
        if current_sql == guard_sql:
            continue
        ast.define.definitions[name] = parser.parse_define(
            f"DEFINE {name} AS ({current_sql}) AND ({guard_sql})"
        ).definitions[name]
        restored.append(name)
    return restored


def _apply_quantifier_bound_repairs(
    ast: RPRAbstractSyntaxTree,
    original_pattern,
    count_range_oversized_variables: set[str | None],
    count_range_define_guards: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Repair oversized bounds on one AST and preserve their DEFINE guards.

    The same operation is used at the pre-regex boundary and by the final
    defensive check. Keeping it in one place is important when sampling is
    enabled because production and sample ASTs must remain structurally
    identical for every subsequent local rule.
    """

    ast.pattern, repairs = _repair_numeric_quantifier_bounds(
        ast.pattern,
        original_pattern,
    )
    if not repairs:
        return [], [], False

    restored_finite_variables = {
        repair.get("variable")
        for repair in repairs
        if repair.get("action", "").startswith("restore_original_finite_range")
    }
    restored_define_guards = []
    if (
        restored_finite_variables & count_range_oversized_variables
        and count_range_define_guards
    ):
        restored_define_guards = _restore_define_guards(ast, count_range_define_guards)

    refreshed_raw_sql_override = False
    if ast.raw_sql_override is not None:
        # ColumnPruningViaJoinRule renders and caches a complete SQL string.
        # Rebuild that cache after a defensive output repair or the cached
        # text would still contain the oversized range.
        previous_override = ast.raw_sql_override
        ast.raw_sql_override = None
        if ColumnPruningViaJoinRule().apply(ast):
            refreshed_raw_sql_override = True
        else:
            ast.raw_sql_override = previous_override

    return repairs, restored_define_guards, refreshed_raw_sql_override


class SampleExecutor(Protocol):
    """Minimal database interface used by sample-dependent decisions."""

    def fetch_one(self, sql: str) -> Mapping[str, object]: ...

    def measure_query(self, sql: str, runs: int) -> "QueryMeasurement | float": ...


@dataclass(frozen=True)
class RuleSpec:
    """One position in the ordered pipeline.

    ``requires_prunable_input`` marks a rule that may remove input rows. Such a
    rule is disabled when empty matches or unmatched-row output can expose the
    removed rows.
    """

    factory: Callable[[], RewriteRule]
    requires_prunable_input: bool = False


@dataclass
class RuleMetric:
    rule: str
    applicable: bool
    applied: bool
    decision: str
    overhead_seconds: float
    sampling_seconds: float = 0.0
    local_rewrite_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleBasedRewriteResult:
    sql: str
    applied_rules: list[str]
    total_overhead_seconds: float
    sampling_seconds: float
    local_rewrite_seconds: float
    rule_metrics: list[RuleMetric]
    stopped_after_rule: str | None = None

    def __iter__(self):
        """Keep ``sql, applied_rules = rewrite_sql(...)`` compatible."""

        yield self.sql
        yield self.applied_rules


@dataclass(frozen=True)
class QueryMeasurement:
    """Outcome of executing and fetching one sample candidate."""

    completed: bool
    elapsed_seconds: float
    fetched_rows: int


# Rule order is part of the optimizer. Duplicate entries are intentional: an
# earlier rule can expose another opportunity for the same rule later.
RULE_PIPELINE = (
    RuleSpec(SingleVariablePatternPushdownRule, True),  # row pruning
    RuleSpec(AnchorPartitionSizeFilterRule, True),  # row pruning
    RuleSpec(AnchorTopKFilterRule, True),  # row pruning
    RuleSpec(RemoveAggregationFuncRule),  # predicate reconstruction
    RuleSpec(DefineIndependentConstraintPropagationRule),  # predicate reconstruction
    RuleSpec(MeasuresVariableRenamingRule, True),  # predicate reconstruction
    RuleSpec(RedundantSubsetVariableRemovalRule),  # predicate reconstruction
    RuleSpec(AlternationVariableMergeRule),  # pathological pattern repair
    RuleSpec(RedundantSubsetVariableRemovalRule),  # predicate reconstruction
    RuleSpec(MatchNumberGuardPushdownRule),  # predicate reconstruction
    RuleSpec(MustAppearVarFalsePushdownRule, True),  # row pruning
    RuleSpec(DefinePartitionPushdownRule, True),  # row pruning
    RuleSpec(WherePartitionPushdownRule),  # row pruning
    # here we add RegexOptimizationPlusRule on NASDAQ dataset
    RuleSpec(RegexOptimizationPlusRule),  # pathological pattern repair
    
    # The optimizations currently supported by RegexOptimizationPlusRule are limited;
    # it is recommended to upgrade to RegexOptimizationUltraRule
    # their core idea is same: encoding --> search regex --> check
    # RuleSpec(RegexOptimizationUltraRule),  # pathological pattern repair
    
    RuleSpec(CountRangePushdownRule),  # predicate reconstruction
    RuleSpec(AnchorPartitionSizeFilterRule, True),  # row pruning
    RuleSpec(ForwardAdjustedConditionRule, True),  # predicate reconstruction
    RuleSpec(MustAppearVarFalsePushdownRule, True),  # row pruning
    RuleSpec(DefineIndependentConstraintPropagationRule),  # predicate reconstruction
    RuleSpec(NestedQuantifierRangeSimplificationRule),  # pathological pattern repair
    RuleSpec(OrderedRangePushdownRule, True),  # row pruning
    RuleSpec(MustAppearPartitionFilterPushdownRule, True),  # row pruning
    RuleSpec(ImpliedIncrementalConditionRemovalRule),  # predicate reconstruction
    RuleSpec(RowFilter4SpecialPatternRule, True),  # row pruning
    RuleSpec(RegexOptimizationPlusRule),  # pathological pattern repair
    RuleSpec(QuantifierScopeShorteningRule),  # predicate reconstruction
    RuleSpec(RowFilteringViaWindowFuncRule, True),  # row pruning
    RuleSpec(IncrementalConditionPushdownRule),  # predicate reconstruction
    RuleSpec(DefineIndependentConstraintPropagationRule),  # predicate reconstruction
    RuleSpec(WindowConditionPreFilterRule, True),  # row pruning
    RuleSpec(ColumnPruningRule),  # column pruning
    RuleSpec(ColumnPruningViaJoinRule),  # column pruning
)

# These rules may replace the input relation but retain its complete public
# column set. Remembering that set lets later rules reason about generated
# SELECT-star wrappers without attempting to parse arbitrary subqueries.
SCHEMA_PRESERVING_INPUT_RULES = (
    SingleVariablePatternPushdownRule,
    AnchorPartitionSizeFilterRule,
    AnchorTopKFilterRule,
    MustAppearVarFalsePushdownRule,
    DefinePartitionPushdownRule,
    WherePartitionPushdownRule,
    ForwardAdjustedConditionRule,
    OrderedRangePushdownRule,
    MustAppearPartitionFilterPushdownRule,
    RowFilter4SpecialPatternRule,
)


def _round_seconds(value: float) -> float:
    return round(max(0.0, value), 6)


def _coerce_measurement(value: QueryMeasurement | float) -> QueryMeasurement:
    """Accept legacy float results from custom and test executors."""

    if isinstance(value, QueryMeasurement):
        measurement = value
    elif all(
        hasattr(value, attribute)
        for attribute in ("completed", "elapsed_seconds", "fetched_rows")
    ):
        measurement = QueryMeasurement(
            bool(value.completed),
            float(value.elapsed_seconds),
            int(value.fetched_rows),
        )
    else:
        measurement = QueryMeasurement(True, float(value), 0)
    if (
        not math.isfinite(measurement.elapsed_seconds)
        or measurement.elapsed_seconds < 0
        or measurement.fetched_rows < 0
    ):
        raise ValueError("candidate benchmark returned an invalid measurement")
    return measurement


def _select_measurement(
    measurements: Mapping[str, QueryMeasurement],
) -> tuple[str, QueryMeasurement, str]:
    """Select the fastest completion, or most productive timeout."""

    completed = [
        (candidate_id, measurement)
        for candidate_id, measurement in measurements.items()
        if measurement.completed
    ]
    if completed:
        completed.sort(
            key=lambda item: (
                item[1].elapsed_seconds,
                0 if item[0] == "no_rewrite" else 1,
                item[0],
            )
        )
        candidate_id, measurement = completed[0]
        return candidate_id, measurement, "fastest_completed"

    timed_out = list(measurements.items())
    if not timed_out:
        raise ValueError("no valid candidate measurements were supplied")
    timed_out.sort(
        key=lambda item: (
            -item[1].fetched_rows,
            item[1].elapsed_seconds,
            0 if item[0] == "no_rewrite" else 1,
            item[0],
        )
    )
    candidate_id, measurement = timed_out[0]
    return candidate_id, measurement, "most_rows_fetched_before_timeout"


def _fetch_one_with_budget(
    executor: SampleExecutor,
    sql: str,
    timeout_seconds: float | None = None,
) -> Mapping[str, object]:
    """Use an executor's optional per-query timeout without breaking old fakes."""

    if timeout_seconds is not None:
        method = getattr(executor, "fetch_one_with_timeout", None)
        if callable(method):
            return method(sql, timeout_seconds)
    return executor.fetch_one(sql)


def _measure_query_with_budget(
    executor: SampleExecutor,
    sql: str,
    runs: int,
    timeout_seconds: float | None = None,
) -> QueryMeasurement:
    """Measure once in the scheduler, optionally with a smaller time budget."""

    if timeout_seconds is not None:
        method = getattr(executor, "measure_query_with_timeout", None)
        if callable(method):
            return _coerce_measurement(method(sql, runs, timeout_seconds))
    return _coerce_measurement(executor.measure_query(sql, runs))


def _normalize_sql(sql: str) -> str:
    normalized = sql.strip().rstrip(";").rstrip()
    if not normalized:
        raise ValueError("SQL is empty")
    return normalized


def requires_conservative_pipeline(ast: RPRAbstractSyntaxTree) -> bool:
    """Whether rules that remove input rows must be disabled."""

    rows_per_match = (ast.rows_per_match or "").strip().upper()
    preserves_unmatched_rows = rows_per_match == ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS
    pattern_can_be_empty = (
        ast.pattern is not None and ast.pattern.get_min_row_num() == 0
    )
    return preserves_unmatched_rows or pattern_can_be_empty


def _unsafe_pruning_reasons(ast: RPRAbstractSyntaxTree) -> list[str]:
    reasons = []
    if ast.pattern is not None and ast.pattern.get_min_row_num() == 0:
        reasons.append("pattern_can_match_zero_rows")
    if (
        ast.rows_per_match or ""
    ).strip().upper() == ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS:
        reasons.append("rows_per_match_preserves_unmatched_rows")
    return reasons


def build_rules(*, conservative: bool) -> list[RewriteRule]:
    """Compatibility helper returning fresh rule instances."""

    return [
        spec.factory()
        for spec in RULE_PIPELINE
        if not (conservative and spec.requires_prunable_input)
    ]


def _sample_ast(ast: RPRAbstractSyntaxTree, sample_input_table: str):
    """Copy an AST and bind its MATCH_RECOGNIZE input to a sample relation."""

    sampled = copy.deepcopy(ast)
    qualifier_safe, rebind_alias = resolve_input_rebind_alias(ast)
    if not qualifier_safe:
        raise ValueError("sample input cannot preserve source qualifiers")
    sample_relation = sample_input_table.strip()
    if not sample_relation:
        raise ValueError("sample_input_table must be non-empty")
    original_match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_.$]*)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_$]*)?",
        ast.input_table.strip(),
        re.IGNORECASE,
    )
    source = sample_relation
    if original_match is None:
        matching_tables = [
            str(name)
            for name in (ast.table_schemas or {})
            if re.search(
                rf"(?<![A-Za-z0-9_$#]){re.escape(str(name))}" r"(?![A-Za-z0-9_$#])",
                ast.input_table,
                re.IGNORECASE,
            )
        ]
        if len(matching_tables) == 1:
            table_pattern = (
                rf"(?<![A-Za-z0-9_$#]){re.escape(matching_tables[0])}"
                r"(?![A-Za-z0-9_$#])"
            )
            source = re.sub(
                table_pattern,
                lambda _match: sample_relation,
                ast.input_table,
                flags=re.IGNORECASE,
            )
    if rebind_alias is not None:
        source = bind_input_relation_alias(source, rebind_alias)
    sampled.input_table = source

    # A sample table/view must have the production relation's schema. Register
    # its name in the copied metadata so later type/nullability proofs work.
    sample_match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_.$]*)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_$]*)?",
        sample_relation,
        re.IGNORECASE,
    )
    if sample_match is not None:
        original_names = (
            {
                original_match.group(1).upper(),
                original_match.group(1).split(".")[-1].upper(),
            }
            if original_match is not None
            else set()
        )
        source_meta = next(
            (
                meta
                for name, meta in (ast.table_schemas or {}).items()
                if str(name).strip().upper() in original_names
                or str(name).strip().split(".")[-1].upper() in original_names
            ),
            None,
        )
        if source_meta is None and len(ast.table_schemas or {}) == 1:
            source_meta = next(iter(ast.table_schemas.values()))
        if source_meta is not None:
            schemas = copy.deepcopy(sampled.table_schemas)
            sample_name = sample_match.group(1).split(".")[-1].upper()
            schemas[sample_name] = copy.deepcopy(source_meta)
            sampled.table_schemas = schemas
            sampled._clear_metadata_cache()
    return sampled


def _with_prefix(ast: RPRAbstractSyntaxTree, select_sql: str) -> str:
    prefix = (ast.cte_prefix or "").strip()
    return f"{prefix}\n{select_sql}" if prefix else select_sql


def _input_row_count_sql(ast: RPRAbstractSyntaxTree, system: str) -> str:
    sql = _with_prefix(
        ast,
        f"SELECT COUNT(*) AS RPR_UNIT_COUNT FROM {ast.input_table.strip()}",
    )
    return _sql_for_system(sql, system)


def _match_output_row_count_sql(ast: RPRAbstractSyntaxTree, system: str) -> str:
    """Count rows emitted by MATCH_RECOGNIZE without materializing columns."""

    counted = copy.deepcopy(ast)
    counted.raw_sql_override = None
    counted.output_names = ExprParser().parse_select(
        "SELECT COUNT(*) AS RPR_OUTPUT_COUNT"
    )
    counted.where = None
    counted.group_by = None
    counted.having = None
    counted.order_by = None
    counted.limit = None
    counted.alias = None
    counted.cte_prefix = None
    return _sql_for_system(_with_prefix(ast, counted.to_sql()), system)


def _numeric_result(row: Mapping[str, object], alias: str) -> float:
    normalized = {str(key).upper(): value for key, value in row.items()}
    value = normalized.get(alias.upper())
    if value is None:
        raise ValueError(f"sample query did not return {alias}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"sample query returned an invalid {alias}")
    return numeric


def _partition_filter_row_counts_sql(
    before_ast: RPRAbstractSyntaxTree,
    after_ast: RPRAbstractSyntaxTree,
    system: str,
) -> str:
    """Count total and retained rows in one aggregate database request."""

    if (before_ast.cte_prefix or "").strip() != (after_ast.cte_prefix or "").strip():
        raise ValueError("sample candidates must preserve the original CTE prefix")
    aggregate = (
        "SELECT SUM(RPR_ROWS_BEFORE) AS RPR_ROWS_BEFORE, "
        "SUM(RPR_ROWS_AFTER) AS RPR_ROWS_AFTER FROM ("
        "SELECT COUNT(*) AS RPR_ROWS_BEFORE, 0 AS RPR_ROWS_AFTER "
        f"FROM {before_ast.input_table.strip()} UNION ALL "
        "SELECT 0 AS RPR_ROWS_BEFORE, COUNT(*) AS RPR_ROWS_AFTER "
        f"FROM {after_ast.input_table.strip()}) RPR_ROW_COUNTS"
    )
    return _sql_for_system(_with_prefix(before_ast, aggregate), system)


def _count_partition_filter_rows(
    executor: SampleExecutor,
    before_ast: RPRAbstractSyntaxTree,
    after_ast: RPRAbstractSyntaxTree,
    system: str,
) -> tuple[float, float, float]:
    started = time.perf_counter()
    row = executor.fetch_one(
        _partition_filter_row_counts_sql(before_ast, after_ast, system)
    )
    elapsed = time.perf_counter() - started
    return (
        _numeric_result(row, "RPR_ROWS_BEFORE"),
        _numeric_result(row, "RPR_ROWS_AFTER"),
        elapsed,
    )


def _count_sample_rows(
    executor: SampleExecutor,
    ast: RPRAbstractSyntaxTree,
    system: str,
    timeout_seconds: float | None = None,
) -> tuple[float, float]:
    started = time.perf_counter()
    row = _fetch_one_with_budget(
        executor, _input_row_count_sql(ast, system), timeout_seconds
    )
    elapsed = time.perf_counter() - started
    return _numeric_result(row, "RPR_UNIT_COUNT"), elapsed


def _count_sample_match_output_rows(
    executor: SampleExecutor,
    ast: RPRAbstractSyntaxTree,
    system: str,
    timeout_seconds: float | None = None,
) -> tuple[float, float]:
    started = time.perf_counter()
    row = _fetch_one_with_budget(
        executor, _match_output_row_count_sql(ast, system), timeout_seconds
    )
    elapsed = time.perf_counter() - started
    return _numeric_result(row, "RPR_OUTPUT_COUNT"), elapsed


def _partition_count_sql(ast: RPRAbstractSyntaxTree, system: str) -> str:
    """Count row-pattern partitions in an input relation without running RPR."""

    partition_columns = [
        column.strip() for column in (ast.partition_by or []) if column.strip()
    ]
    if partition_columns:
        partition_rows = (
            f"SELECT DISTINCT {', '.join(partition_columns)} "
            f"FROM {ast.input_table.strip()}"
        )
    else:
        # Without PARTITION BY the complete non-empty input is one partition.
        partition_rows = f"SELECT 1 FROM {ast.input_table.strip()} LIMIT 1"

    sql = _with_prefix(
        ast,
        "SELECT COUNT(*) AS RPR_PARTITION_COUNT "
        f"FROM ({partition_rows}) RPR_PARTITIONS",
    )
    return _sql_for_system(sql, system)


def _count_sample_partitions(
    executor: SampleExecutor, ast: RPRAbstractSyntaxTree, system: str
) -> tuple[float, float]:
    started = time.perf_counter()
    row = executor.fetch_one(_partition_count_sql(ast, system))
    elapsed = time.perf_counter() - started
    return _numeric_result(row, "RPR_PARTITION_COUNT"), elapsed


def _postprocess_input_change(
    ast: RPRAbstractSyntaxTree,
    input_table_before: str,
    applied: bool,
    preserved_columns: tuple[str, ...] | None = None,
    preserved_schema: dict[str, Any] | None = None,
) -> None:
    if not applied or ast.input_table == input_table_before:
        return
    if getattr(ast, "_skip_from_clause_optimization_once", False):
        ast._skip_from_clause_optimization_once = False
    else:
        ast.optimize_from_clause()
    if preserved_columns:
        ast._preserved_input_columns = preserved_columns
    # Only row-filtering rules known to preserve the public schema may carry
    # a primary key through an opaque generated relation.
    ast._preserved_input_schema = preserved_schema


def _known_input_columns(ast: RPRAbstractSyntaxTree) -> tuple[str, ...] | None:
    columns = ast._extract_join_source_columns()
    if not columns:
        columns = getattr(ast, "_preserved_input_columns", None)
    if not columns:
        columns = getattr(ast, "_window_prefilter_input_columns", None)
    return tuple(columns) if columns else None


def _known_input_schema(ast: RPRAbstractSyntaxTree) -> dict[str, Any] | None:
    """Resolve the actual input schema without guessing from matching columns."""

    columns = _known_input_columns(ast)
    if not columns:
        return None
    normalized_columns = [str(column).split(".")[-1].upper() for column in columns]
    preserved = getattr(ast, "_preserved_input_schema", None)
    if (
        preserved is not None
        and RPRAbstractSyntaxTree._schema_column_names(preserved.get("columns"))
        == normalized_columns
    ):
        return copy.deepcopy(preserved)

    source_match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_.$]*)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_$]*)?",
        ast.input_table.strip(),
        re.IGNORECASE,
    )
    if source_match is not None:
        source_name = source_match.group(1).upper()
        source_schemas = [
            metadata
            for name, metadata in (ast.table_schemas or {}).items()
            if str(name).upper() in {source_name, source_name.split(".")[-1]}
            and isinstance(metadata, dict)
        ]
        if (
            len(source_schemas) == 1
            and RPRAbstractSyntaxTree._schema_column_names(
                source_schemas[0].get("columns")
            )
            == normalized_columns
        ):
            return copy.deepcopy(source_schemas[0])

    # Derived projections and joins may change values or multiply rows. Their
    # resolved column types remain useful, but a base-table PK is not proof
    # that the generated relation is unique.
    metadata = {
        "columns": normalized_columns,
        "column_types": {},
        "column_nullables": {},
    }
    for column in normalized_columns:
        resolved = ast.get_column_metadata(column)
        if resolved is None:
            return None
        metadata["column_types"][column] = resolved.data_type
        metadata["column_nullables"][column] = resolved.nullable
    return metadata


def _evaluate_and_apply(ast: RPRAbstractSyntaxTree, rule: RewriteRule):
    applicable = rule.is_applicable(ast)
    input_table_before = ast.input_table
    preserved_columns = (
        _known_input_columns(ast)
        if applicable and isinstance(rule, SCHEMA_PRESERVING_INPUT_RULES)
        else None
    )
    preserved_schema = (
        _known_input_schema(ast)
        if applicable and isinstance(rule, SCHEMA_PRESERVING_INPUT_RULES)
        else None
    )
    applied = rule.apply(ast) if applicable else False
    _postprocess_input_change(
        ast,
        input_table_before,
        applied,
        preserved_columns=preserved_columns,
        preserved_schema=preserved_schema,
    )
    return applicable, applied


def _apply_local_rule_pair(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    rule_type,
) -> RuleMetric:
    started = time.perf_counter()
    production_rule = rule_type()
    applicable, applied = _evaluate_and_apply(production_ast, production_rule)

    if sample_ast is not None:
        sample_rule = rule_type()
        sample_applicable, sample_applied = _evaluate_and_apply(sample_ast, sample_rule)
        if (sample_applicable, sample_applied) != (applicable, applied):
            raise RuntimeError(
                f"{production_rule.name} diverged between production and sample ASTs"
            )

    elapsed = time.perf_counter() - started
    return RuleMetric(
        rule=production_rule.name,
        applicable=applicable,
        applied=applied,
        decision="applied" if applied else "not_applicable",
        overhead_seconds=_round_seconds(elapsed),
        local_rewrite_seconds=_round_seconds(elapsed),
    )


def _apply_column_pruning_via_join(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    executor: SampleExecutor | None,
    *,
    maximum_output_ratio: float,
    system: str,
) -> tuple[RPRAbstractSyntaxTree, RPRAbstractSyntaxTree | None, RuleMetric]:
    """Guard join-back pruning against overlapping-match row explosion."""

    started = time.perf_counter()
    rule = ColumnPruningViaJoinRule()
    applicable = rule.is_applicable(production_ast)
    applied = False
    sampling_seconds = 0.0
    decision = "not_applicable"
    skip_target = (production_ast.after_match_skip_to or PAST_LAST_ROW).strip().upper()
    details: dict[str, Any] = {"after_match_skip_to": skip_target}

    def apply_candidate_pair():
        production_candidate = copy.deepcopy(production_ast)
        production_applicable, production_applied = _evaluate_and_apply(
            production_candidate, ColumnPruningViaJoinRule()
        )
        sample_candidate = sample_ast
        if sample_ast is not None:
            sample_candidate = copy.deepcopy(sample_ast)
            sample_applicable, sample_applied = _evaluate_and_apply(
                sample_candidate, ColumnPruningViaJoinRule()
            )
            if (sample_applicable, sample_applied) != (
                production_applicable,
                production_applied,
            ):
                raise RuntimeError(
                    "column_pruning_via_join diverged between production "
                    "and sample ASTs"
                )
        return (
            production_candidate,
            sample_candidate,
            production_applicable,
            production_applied,
        )

    if applicable and skip_target == PAST_LAST_ROW:
        (
            production_candidate,
            sample_candidate,
            _candidate_applicable,
            candidate_applied,
        ) = apply_candidate_pair()
        if candidate_applied:
            production_ast = production_candidate
            sample_ast = sample_candidate
            applied = True
            decision = "applied_past_last_row_without_sampling"
        else:
            decision = "kept_original_apply_failed"
    elif applicable:
        details["maximum_output_ratio"] = maximum_output_ratio
        if sample_ast is None or executor is None:
            decision = "kept_original_no_sampler"
        else:
            sample_rule = ColumnPruningViaJoinRule()
            if not sample_rule.is_applicable(sample_ast):
                raise RuntimeError(
                    "column_pruning_via_join diverged between production "
                    "and sample ASTs"
                )
            try:
                sample_input_rows, elapsed = _count_sample_rows(
                    executor, sample_ast, system
                )
                sampling_seconds += elapsed
                sample_output_rows, elapsed = _count_sample_match_output_rows(
                    executor, sample_ast, system
                )
                sampling_seconds += elapsed
                output_ratio = (
                    sample_output_rows / sample_input_rows
                    if sample_input_rows > 0
                    else (0.0 if sample_output_rows == 0 else None)
                )
                details.update(
                    {
                        "sample_input_rows": sample_input_rows,
                        "sample_output_rows": sample_output_rows,
                        "sample_output_ratio": (
                            round(output_ratio, 6) if output_ratio is not None else None
                        ),
                    }
                )
                if sample_output_rows <= sample_input_rows * maximum_output_ratio:
                    (
                        production_candidate,
                        sample_candidate,
                        _candidate_applicable,
                        candidate_applied,
                    ) = apply_candidate_pair()
                    if candidate_applied:
                        production_ast = production_candidate
                        sample_ast = sample_candidate
                        applied = True
                        decision = "applied_sample_output_ratio"
                    else:
                        decision = "kept_original_apply_failed"
                else:
                    decision = "kept_original_excessive_sample_output_ratio"
            except Exception as exc:
                decision = "kept_original_sampling_error"
                details["error"] = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return (
        production_ast,
        sample_ast,
        RuleMetric(
            rule=rule.name,
            applicable=applicable,
            applied=applied,
            decision=decision,
            overhead_seconds=_round_seconds(elapsed),
            sampling_seconds=_round_seconds(sampling_seconds),
            local_rewrite_seconds=_round_seconds(elapsed - sampling_seconds),
            details=details,
        ),
    )


def _skipped_metric(rule_name: str, decision: str, **details) -> RuleMetric:
    return RuleMetric(
        rule=rule_name,
        applicable=False,
        applied=False,
        decision=decision,
        overhead_seconds=0.0,
        details=details,
    )


def _apply_must_appear_partition_filter(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    executor: SampleExecutor | None,
    *,
    use_sampling: bool,
    minimum_row_reduction: float,
    system: str,
) -> tuple[RPRAbstractSyntaxTree | None, RuleMetric]:
    """Apply the rule when its sample filter removes whole partitions."""

    metadata_started = time.perf_counter()
    production_rule = MustAppearPartitionFilterPushdownRule()
    applicable = production_rule.is_applicable(production_ast)
    nullable_partition_columns = [
        column.strip()
        for column in (production_ast.partition_by or [])
        if column.strip() and production_ast.is_column_nullable(column) is True
    ]
    if applicable and nullable_partition_columns:
        metadata_elapsed = time.perf_counter() - metadata_started
        return sample_ast, RuleMetric(
            rule=production_rule.name,
            applicable=True,
            applied=False,
            decision="kept_original_nullable_partition_key",
            overhead_seconds=_round_seconds(metadata_elapsed),
            local_rewrite_seconds=_round_seconds(metadata_elapsed),
            details={"nullable_partition_columns": nullable_partition_columns},
        )

    if not use_sampling:
        metric = _apply_local_rule_pair(
            production_ast, sample_ast, MustAppearPartitionFilterPushdownRule
        )
        if metric.applied:
            metric.decision = "applied_without_sampling"
        return sample_ast, metric

    started = time.perf_counter()
    sampling_seconds = 0.0
    applied = False
    decision = "not_applicable"
    details: dict[str, Any] = {}

    if applicable and (sample_ast is None or executor is None):
        decision = "kept_original_no_sampler"
    elif applicable:
        candidate_sample = copy.deepcopy(sample_ast)
        candidate_rule = MustAppearPartitionFilterPushdownRule()
        candidate_applicable, candidate_applied = _evaluate_and_apply(
            candidate_sample, candidate_rule
        )
        if not candidate_applicable or not candidate_applied:
            decision = "kept_original_sample_candidate_failed"
        else:
            try:
                before_partitions, elapsed = _count_sample_partitions(
                    executor, sample_ast, system
                )
                sampling_seconds += elapsed
                after_partitions, elapsed = _count_sample_partitions(
                    executor, candidate_sample, system
                )
                sampling_seconds += elapsed
                details.update(
                    {
                        "sample_partitions_before": before_partitions,
                        "sample_partitions_after": after_partitions,
                        "sample_partitions_removed": max(
                            0.0, before_partitions - after_partitions
                        ),
                    }
                )

                if after_partitions >= before_partitions:
                    decision = "kept_original_no_partitions_removed"
                else:
                    rows_before, rows_after, elapsed = _count_partition_filter_rows(
                        executor, sample_ast, candidate_sample, system
                    )
                    sampling_seconds += elapsed
                    rows_removed = max(0.0, rows_before - rows_after)
                    reduction_ratio = (
                        rows_removed / rows_before if rows_before > 0 else 0.0
                    )
                    details.update(
                        {
                            "sample_rows_before": rows_before,
                            "sample_rows_after": rows_after,
                            "sample_rows_removed": rows_removed,
                            "sample_row_reduction_ratio": reduction_ratio,
                            "minimum_row_reduction": minimum_row_reduction,
                        }
                    )
                    if reduction_ratio <= minimum_row_reduction:
                        decision = "kept_original_insufficient_row_reduction"
                    else:
                        _applicable, applied = _evaluate_and_apply(
                            production_ast, production_rule
                        )
                        if applied:
                            sample_ast = candidate_sample
                            decision = "applied_sample_row_reduction"
                        else:
                            decision = "kept_original_apply_failed"
            except Exception as exc:
                decision = "kept_original_sampling_error"
                details["error"] = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return sample_ast, RuleMetric(
        rule=production_rule.name,
        applicable=applicable,
        applied=applied,
        decision=decision,
        overhead_seconds=_round_seconds(elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(elapsed - sampling_seconds),
        details=details,
    )


def _postprocess_candidate(
    original_ast: RPRAbstractSyntaxTree, candidate_ast: RPRAbstractSyntaxTree
) -> RPRAbstractSyntaxTree:
    candidate_ast._window_prefilter_input_schema = _known_input_schema(original_ast)
    _postprocess_input_change(candidate_ast, original_ast.input_table, True)
    _register_generated_input_schema(candidate_ast)
    return candidate_ast


def _register_generated_input_schema(ast: RPRAbstractSyntaxTree) -> None:
    """Expose a generated window-filter CTE's schema to downstream rules."""

    exposed_columns = getattr(ast, "_window_prefilter_input_columns", None)
    if not exposed_columns:
        return
    input_match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_.$]*)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_$]*)?",
        ast.input_table.strip(),
        re.IGNORECASE,
    )
    if input_match is None:
        return
    relation_name = input_match.group(1).split(".")[-1].upper()
    if relation_name in {str(name).upper() for name in ast.table_schemas or {}}:
        return

    metadata = getattr(ast, "_window_prefilter_input_schema", None)
    if metadata is None:
        return
    required = [str(column).split(".")[-1].upper() for column in exposed_columns]
    if RPRAbstractSyntaxTree._schema_column_names(metadata.get("columns")) != required:
        return

    schemas = copy.deepcopy(ast.table_schemas)
    schemas[relation_name] = copy.deepcopy(metadata)
    ast.table_schemas = schemas
    ast._clear_metadata_cache()


def _apply_window_condition_prefilter(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    executor: SampleExecutor | None,
    *,
    use_sampling: bool,
    benchmark_runs: int,
    max_pair_candidates: int | None,
    system: str,
) -> tuple[RPRAbstractSyntaxTree, RPRAbstractSyntaxTree | None, RuleMetric]:
    """Select a window-prefilter candidate using sample measurements."""

    rule = WindowConditionPreFilterRule()
    if not use_sampling:
        started = time.perf_counter()
        applicable = rule.is_applicable(production_ast)
        elapsed = time.perf_counter() - started
        return (
            production_ast,
            sample_ast,
            RuleMetric(
                rule=rule.name,
                applicable=applicable,
                applied=False,
                decision="disabled_without_sampling",
                overhead_seconds=_round_seconds(elapsed),
                local_rewrite_seconds=_round_seconds(elapsed),
            ),
        )

    started = time.perf_counter()
    sampling_seconds = 0.0
    applicable = rule.is_applicable(production_ast)
    applied = False
    decision = "not_applicable"
    details: dict[str, Any] = {"benchmark_runs": benchmark_runs}

    if applicable and (sample_ast is None or executor is None):
        decision = "kept_original_no_sampler"
    elif applicable:
        try:
            request = rule.build_statistics_request(sample_ast)
            if request is None:
                raise ValueError("statistics request could not be built")
            sample_started = time.perf_counter()
            statistics_row = executor.fetch_one(_sql_for_system(request.sql, system))
            sampling_seconds += time.perf_counter() - sample_started
            window_statistics = request.parse_row(statistics_row)

            candidates = rule.generate_candidates(
                sample_ast,
                window_statistics,
                max_pair_candidates=max_pair_candidates,
            )
            no_rewrite = WindowPrefilterCandidate(
                candidate_id="no_rewrite",
                variables=(),
                strategy="no_rewrite",
                estimated_cost=1.0,
            )

            before_rows, elapsed = _count_sample_rows(executor, sample_ast, system)
            sampling_seconds += elapsed
            candidate_asts: dict[str, RPRAbstractSyntaxTree] = {
                no_rewrite.candidate_id: sample_ast
            }
            retained_rows: dict[str, float] = {no_rewrite.candidate_id: before_rows}
            eligible = [no_rewrite]
            measurements_by_candidate: dict[str, QueryMeasurement] = {}
            single_measurements: dict[str, QueryMeasurement] = {}
            pair_measurements: dict[str, QueryMeasurement] = {}
            candidate_pruned: dict[str, dict[str, Any]] = {}
            baseline_history: list[dict[str, Any]] = []
            baseline_id: str | None = None
            base_time: float | None = None
            configured_timeout = getattr(
                executor, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
            )
            if (
                not isinstance(configured_timeout, (int, float))
                or not math.isfinite(configured_timeout)
                or configured_timeout <= 0
            ):
                configured_timeout = DEFAULT_TIMEOUT_SECONDS
            configured_timeout = float(configured_timeout)
            single_candidates = [
                candidate for candidate in candidates if candidate.strategy == "single"
            ]
            pair_candidates = [
                candidate
                for candidate in candidates
                if candidate.strategy.startswith("pair_")
            ]

            # Probe each candidate at most once. A completed single starts the
            # dynamic baseline; otherwise pairs continue with the global budget.
            for candidate in single_candidates:
                candidate_ast = rule.rewrite_candidate(
                    sample_ast, candidate, window_statistics
                )
                if candidate_ast is None:
                    continue
                candidate_ast = _postprocess_candidate(sample_ast, candidate_ast)
                candidate_asts[candidate.candidate_id] = candidate_ast
                sample_sql = _sql_for_system(candidate_ast.to_sql(), system)
                measured = None
                try:
                    sample_started = time.perf_counter()
                    measured = _measure_query_with_budget(
                        executor, sample_sql, benchmark_runs
                    )
                except Exception as exc:
                    details.setdefault("candidate_errors", {})[
                        candidate.candidate_id
                    ] = f"{type(exc).__name__}: {exc}"
                finally:
                    sampling_seconds += time.perf_counter() - sample_started

                if measured is None:
                    retained_rows[candidate.candidate_id] = math.nan
                    continue
                single_measurements[candidate.candidate_id] = measured

                try:
                    sample_started = time.perf_counter()
                    after_rows, elapsed = _count_sample_rows(
                        executor, candidate_ast, system
                    )
                except Exception as exc:
                    retained_rows[candidate.candidate_id] = math.nan
                    details.setdefault("candidate_errors", {})[
                        candidate.candidate_id
                    ] = f"{type(exc).__name__}: {exc}"
                finally:
                    sampling_seconds += time.perf_counter() - sample_started
                if candidate.candidate_id not in retained_rows:
                    retained_rows[candidate.candidate_id] = after_rows
                    if after_rows < before_rows:
                        eligible.append(candidate)
                        measurements_by_candidate[candidate.candidate_id] = measured
                if (
                    measured.completed
                    and measured.elapsed_seconds > 0
                    and (base_time is None or measured.elapsed_seconds < base_time)
                ):
                    baseline_id = candidate.candidate_id
                    base_time = measured.elapsed_seconds
                    baseline_history.append(
                        {
                            "candidate_id": baseline_id,
                            "elapsed_seconds": base_time,
                            "candidate_timeout_seconds": min(
                                configured_timeout,
                                base_time * PAIR_CANDIDATE_TIMEOUT_MULTIPLIER,
                            ),
                        }
                    )

            completed_single = [
                (candidate_id, measurement)
                for candidate_id, measurement in single_measurements.items()
                if measurement.completed and measurement.elapsed_seconds > 0
            ]
            completed_single.sort(key=lambda item: (item[1].elapsed_seconds, item[0]))
            if completed_single:
                single_id, single_measurement = completed_single[0]
                details["single_baseline_candidate"] = single_id
                details["single_baseline_seconds"] = single_measurement.elapsed_seconds

            for candidate in pair_candidates:
                candidate_ast = rule.rewrite_candidate(
                    sample_ast, candidate, window_statistics
                )
                if candidate_ast is None:
                    continue
                candidate_ast = _postprocess_candidate(sample_ast, candidate_ast)
                candidate_asts[candidate.candidate_id] = candidate_ast
                candidate_timeout = (
                    configured_timeout
                    if base_time is None
                    else min(
                        configured_timeout,
                        base_time * PAIR_CANDIDATE_TIMEOUT_MULTIPLIER,
                    )
                )

                row_count_known = False
                row_reducing = False
                try:
                    sample_started = time.perf_counter()
                    after_rows, elapsed = _count_sample_rows(
                        executor,
                        candidate_ast,
                        system,
                        timeout_seconds=candidate_timeout,
                    )
                    retained_rows[candidate.candidate_id] = after_rows
                    row_count_known = elapsed <= candidate_timeout
                    row_reducing = row_count_known and after_rows < before_rows
                    if not row_count_known:
                        candidate_pruned[candidate.candidate_id] = {
                            "phase": "row_count",
                            "reason": "exceeded_candidate_time_budget",
                            "elapsed_seconds": elapsed,
                            "timeout_seconds": candidate_timeout,
                        }
                except Exception as exc:
                    retained_rows[candidate.candidate_id] = math.nan
                    candidate_pruned[candidate.candidate_id] = {
                        "phase": "row_count",
                        "reason": "exceeded_candidate_time_budget",
                        "timeout_seconds": candidate_timeout,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    sampling_seconds += time.perf_counter() - sample_started

                if row_count_known and not row_reducing:
                    continue
                # A row-count timeout is enough to reject the candidate only
                # after a baseline exists. Before that, run the full pair so a
                # completed pair can establish the first reliable baseline.
                if not row_count_known and base_time is not None:
                    continue

                sample_sql = _sql_for_system(candidate_ast.to_sql(), system)
                try:
                    sample_started = time.perf_counter()
                    measured = _measure_query_with_budget(
                        executor,
                        sample_sql,
                        benchmark_runs,
                        candidate_timeout,
                    )
                    if (
                        measured.completed
                        and measured.elapsed_seconds > candidate_timeout
                    ):
                        measured = QueryMeasurement(
                            False, measured.elapsed_seconds, measured.fetched_rows
                        )
                    pair_measurements[candidate.candidate_id] = measured
                except Exception as exc:
                    candidate_pruned[candidate.candidate_id] = {
                        "phase": "benchmark",
                        "reason": "benchmark_failed",
                        "timeout_seconds": candidate_timeout,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    continue
                finally:
                    sampling_seconds += time.perf_counter() - sample_started

                if measured.completed and measured.elapsed_seconds > 0:
                    if base_time is None or measured.elapsed_seconds < base_time:
                        baseline_id = candidate.candidate_id
                        base_time = measured.elapsed_seconds
                        new_timeout = min(
                            configured_timeout,
                            base_time * PAIR_CANDIDATE_TIMEOUT_MULTIPLIER,
                        )
                        baseline_history.append(
                            {
                                "candidate_id": baseline_id,
                                "elapsed_seconds": base_time,
                                "candidate_timeout_seconds": new_timeout,
                            }
                        )
                        # A faster baseline invalidates cached candidates that
                        # would not survive the newly tightened budget.
                        for cached_id, cached in list(
                            measurements_by_candidate.items()
                        ):
                            if cached_id == candidate.candidate_id:
                                continue
                            if (
                                not cached.completed
                                or cached.elapsed_seconds > new_timeout
                            ):
                                measurements_by_candidate.pop(cached_id, None)
                                eligible[:] = [
                                    item
                                    for item in eligible
                                    if item.candidate_id != cached_id
                                ]
                                candidate_pruned[cached_id] = {
                                    "phase": "rebaseline",
                                    "reason": "exceeded_updated_base_time_budget",
                                    "measurement": asdict(cached),
                                    "baseline_candidate": baseline_id,
                                    "base_time_seconds": base_time,
                                    "timeout_seconds": new_timeout,
                                }

                    if row_reducing:
                        eligible.append(candidate)
                        measurements_by_candidate[candidate.candidate_id] = measured
                elif base_time is None and row_reducing:
                    # Preserve timeout progress for the all-timeout fallback.
                    eligible.append(candidate)
                    measurements_by_candidate[candidate.candidate_id] = measured
                else:
                    candidate_pruned[candidate.candidate_id] = {
                        "phase": "benchmark",
                        "reason": "exceeded_candidate_time_budget",
                        "measurement": asdict(measured),
                        "timeout_seconds": candidate_timeout,
                    }

            if base_time is not None:
                details["baseline_candidate"] = baseline_id
                details["base_time_seconds"] = base_time
                details["pair_candidate_timeout_seconds"] = min(
                    configured_timeout,
                    base_time * PAIR_CANDIDATE_TIMEOUT_MULTIPLIER,
                )
            details["baseline_history"] = baseline_history

            # The original query participates in final selection but does not
            # define the pair budget.
            try:
                sample_started = time.perf_counter()
                measurements_by_candidate[no_rewrite.candidate_id] = (
                    _measure_query_with_budget(
                        executor,
                        _sql_for_system(sample_ast.to_sql(), system),
                        benchmark_runs,
                    )
                )
            except Exception as exc:
                details.setdefault("candidate_errors", {})[
                    no_rewrite.candidate_id
                ] = f"{type(exc).__name__}: {exc}"
            finally:
                sampling_seconds += time.perf_counter() - sample_started

            details["single_baseline_measurements"] = {
                candidate_id: asdict(measurement)
                for candidate_id, measurement in single_measurements.items()
            }
            details["pair_candidate_measurements"] = {
                candidate_id: asdict(measurement)
                for candidate_id, measurement in pair_measurements.items()
            }
            if candidate_pruned:
                details["adaptive_candidates_pruned"] = candidate_pruned
                pair_pruned = {
                    candidate_id: item
                    for candidate_id, item in candidate_pruned.items()
                    if candidate_id.startswith("pair_")
                }
                if pair_pruned:
                    details["pair_candidates_pruned"] = pair_pruned

            details["sample_rows_before"] = before_rows
            details["sample_rows_after"] = {
                key: (None if math.isnan(value) else value)
                for key, value in retained_rows.items()
            }
            details["candidate_measurements"] = {
                candidate_id: asdict(measurement)
                for candidate_id, measurement in measurements_by_candidate.items()
            }
            details["candidate_elapsed_seconds"] = {
                candidate_id: measurement.elapsed_seconds
                for candidate_id, measurement in measurements_by_candidate.items()
                if measurement.completed
            }

            all_candidate_counts_completed = all(
                candidate.candidate_id in retained_rows
                and not math.isnan(retained_rows[candidate.candidate_id])
                for candidate in candidates
            )
            if (
                len(eligible) == 1
                and "no_rewrite" in measurements_by_candidate
                and all_candidate_counts_completed
            ):
                decision = "kept_original_no_rows_removed"
            elif not measurements_by_candidate:
                decision = "kept_original_all_benchmarks_failed"
            else:
                selected_id, selected, selection_reason = _select_measurement(
                    measurements_by_candidate
                )
                candidates_by_id = {
                    candidate.candidate_id: candidate for candidate in eligible
                }
                selected_candidate = candidates_by_id[selected_id]
                details["selected_candidate"] = selected_id
                details["selected_sample_seconds"] = selected.elapsed_seconds
                details["selected_fetched_rows"] = selected.fetched_rows
                details["selection_reason"] = selection_reason
                original = measurements_by_candidate.get("no_rewrite")
                details["sample_speedup"] = (
                    original.elapsed_seconds / selected.elapsed_seconds
                    if original is not None
                    and original.completed
                    and selected.completed
                    and selected.elapsed_seconds > 0
                    else None
                )
                if selected_candidate.strategy == "no_rewrite":
                    decision = (
                        "kept_original_fastest_completed_candidate"
                        if selected.completed
                        else "kept_original_most_rows_fetched_before_timeout"
                    )
                else:
                    rewritten = rule.rewrite_candidate(
                        production_ast, selected_candidate, window_statistics
                    )
                    if rewritten is None:
                        decision = "kept_original_apply_failed"
                    else:
                        production_ast = _postprocess_candidate(
                            production_ast, rewritten
                        )
                        sample_ast = candidate_asts[selected_id]
                        applied = True
                        decision = (
                            "applied_fastest_completed_sample_candidate"
                            if selected.completed
                            else "applied_most_rows_fetched_sample_candidate"
                        )
        except Exception as exc:
            decision = "kept_original_sampling_error"
            details["error"] = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    metric = RuleMetric(
        rule=rule.name,
        applicable=applicable,
        applied=applied,
        decision=decision,
        overhead_seconds=_round_seconds(elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(elapsed - sampling_seconds),
        details=details,
    )
    return production_ast, sample_ast, metric


def rewrite_sql(
    sql: str,
    table_schemas: dict,
    *,
    system: str = DEFAULT_SYSTEM,
    sample_input_table: str | None = None,
    executor: SampleExecutor | None = None,
    sample_must_appear_partition_filter: bool = False,
    sample_window_condition_prefilter: bool = False,
    benchmark_runs: int = DEFAULT_RUNS,
    max_pair_candidates: int | None = None,
    partition_pruning_min_row_reduction: float = (
        DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION
    ),
    column_pruning_via_join_max_output_ratio: float = (
        DEFAULT_COLUMN_PRUNING_VIA_JOIN_MAX_OUTPUT_RATIO
    ),
) -> RuleBasedRewriteResult:
    """Parse and optimize one MATCH_RECOGNIZE query."""

    system = _normalize_system(system)
    if benchmark_runs < 1:
        raise ValueError("benchmark_runs must be positive")
    if max_pair_candidates is not None and max_pair_candidates < 0:
        raise ValueError("max_pair_candidates must be non-negative")
    if not 0.0 <= partition_pruning_min_row_reduction <= 1.0:
        raise ValueError("partition_pruning_min_row_reduction must be in [0, 1]")
    if (
        not math.isfinite(column_pruning_via_join_max_output_ratio)
        or column_pruning_via_join_max_output_ratio < 0
    ):
        raise ValueError(
            "column_pruning_via_join_max_output_ratio must be finite and "
            "non-negative"
        )

    total_started = time.perf_counter()
    normalized_input = _normalize_sql(sql)
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(normalized_input, table_schemas)
    original_pattern = _parse_unsimplified_input_pattern(normalized_input, ast)
    validation_errors = ast.validate()
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    sampling_requested = bool(sample_input_table) or (
        sample_must_appear_partition_filter or sample_window_condition_prefilter
    )
    sampled_ast = (
        _sample_ast(ast, sample_input_table)
        if sampling_requested and sample_input_table is not None
        else None
    )

    metrics: list[RuleMetric] = []
    anchor_partition_size_filter_applied = False
    row_filter_4_applied = False
    stopped_after_rule = None
    count_range_oversized_variables: set[str | None] = set()
    count_range_define_guards: dict[str, Any] = {}
    pre_regex_quantifier_repairs: list[dict[str, Any]] = []
    has_pre_regex_quantifier_repairs = False
    pre_regex_restored_define_guards: list[str] = []
    pre_regex_refreshed_raw_sql_override = False
    pre_regex_repair_seconds = 0.0

    for spec in RULE_PIPELINE:
        rule_name = spec.factory().name
        if spec.factory is RegexOptimizationPlusRule:
            repair_started = time.perf_counter()
            (
                pre_regex_quantifier_repairs,
                pre_regex_restored_define_guards,
                pre_regex_refreshed_raw_sql_override,
            ) = _apply_quantifier_bound_repairs(
                ast,
                original_pattern,
                count_range_oversized_variables,
                count_range_define_guards,
            )
            if pre_regex_quantifier_repairs:
                has_pre_regex_quantifier_repairs = True
            if sampled_ast is not None:
                sample_repair_result = _apply_quantifier_bound_repairs(
                    sampled_ast,
                    original_pattern,
                    count_range_oversized_variables,
                    count_range_define_guards,
                )
                production_repair_result = (
                    pre_regex_quantifier_repairs,
                    pre_regex_restored_define_guards,
                    pre_regex_refreshed_raw_sql_override,
                )
                if sample_repair_result != production_repair_result:
                    raise RuntimeError(
                        "quantifier bound repair diverged between production "
                        "and sample ASTs"
                    )
            pre_regex_repair_seconds = _round_seconds(
                time.perf_counter() - repair_started
            )
        count_range_before = (
            copy.deepcopy(ast) if spec.factory is CountRangePushdownRule else None
        )
        if spec.requires_prunable_input and requires_conservative_pipeline(ast):
            metrics.append(
                _skipped_metric(
                    rule_name,
                    "skipped_unsafe_input_pruning",
                    reasons=_unsafe_pruning_reasons(ast),
                )
            )
            continue

        if spec.factory is ColumnPruningViaJoinRule and system == "trino":
            metrics.append(
                _skipped_metric(
                    rule_name,
                    "skipped_unsupported_system",
                    system=system,
                )
            )
            continue

        if spec.factory is RowFilteringViaWindowFuncRule and row_filter_4_applied:
            metrics.append(
                _skipped_metric(
                    rule_name,
                    "skipped_conflict_with_row_filter_4_special_pattern",
                )
            )
            continue

        if (
            spec.factory is MustAppearPartitionFilterPushdownRule
            and anchor_partition_size_filter_applied
        ):
            metrics.append(
                _skipped_metric(
                    rule_name,
                    "skipped_conflict_with_anchor_partition_size_filter",
                )
            )
            continue

        if spec.factory is MustAppearPartitionFilterPushdownRule:
            sampled_ast, metric = _apply_must_appear_partition_filter(
                ast,
                sampled_ast,
                executor,
                use_sampling=sample_must_appear_partition_filter,
                minimum_row_reduction=partition_pruning_min_row_reduction,
                system=system,
            )
        elif spec.factory is WindowConditionPreFilterRule:
            ast, sampled_ast, metric = _apply_window_condition_prefilter(
                ast,
                sampled_ast,
                executor,
                use_sampling=sample_window_condition_prefilter,
                benchmark_runs=benchmark_runs,
                max_pair_candidates=max_pair_candidates,
                system=system,
            )
        elif spec.factory is ColumnPruningViaJoinRule:
            ast, sampled_ast, metric = _apply_column_pruning_via_join(
                ast,
                sampled_ast,
                executor,
                maximum_output_ratio=(column_pruning_via_join_max_output_ratio),
                system=system,
            )
        else:
            metric = _apply_local_rule_pair(ast, sampled_ast, spec.factory)

        if spec.factory is RegexOptimizationPlusRule:
            metric.overhead_seconds = _round_seconds(
                metric.overhead_seconds + pre_regex_repair_seconds
            )
            metric.local_rewrite_seconds = _round_seconds(
                metric.local_rewrite_seconds + pre_regex_repair_seconds
            )
            if pre_regex_quantifier_repairs:
                metric.details["quantifier_pre_regex_repairs"] = (
                    pre_regex_quantifier_repairs
                )
                metric.details["quantifier_bound_limit"] = MAX_PORTABLE_QUANTIFIER_BOUND
                if pre_regex_restored_define_guards:
                    metric.details["quantifier_restored_define_guards"] = (
                        pre_regex_restored_define_guards
                    )
                if pre_regex_refreshed_raw_sql_override:
                    metric.details["quantifier_refreshed_raw_sql_override"] = True

        metrics.append(metric)
        if spec.factory is AnchorPartitionSizeFilterRule and metric.applied:
            anchor_partition_size_filter_applied = True
        if (
            spec.factory is CountRangePushdownRule
            and metric.applied
            and count_range_before is not None
        ):
            count_range_oversized_variables = _finite_oversized_quantifier_variables(
                ast.pattern
            )
            count_range_define_guards = _changed_define_conditions(
                count_range_before, ast
            )
        if spec.factory is RowFilter4SpecialPatternRule and metric.applied:
            row_filter_4_applied = True

        # This rule proves that the query has no matches. Later rewrites cannot
        # improve that result and, per the pipeline contract, must not run.
        if spec.factory is MustAppearVarFalsePushdownRule and metric.applied:
            stopped_after_rule = metric.rule
            break

    applied_rules = [metric.rule for metric in metrics if metric.applied]
    # A later regex pass may need no repairs. Keep earlier bound changes even
    # when every regular rewrite rule is a no-op.
    if applied_rules or has_pre_regex_quantifier_repairs:
        output_ast = copy.deepcopy(ast)
        (
            quantifier_repairs,
            restored_define_guards,
            refreshed_raw_sql_override,
        ) = _apply_quantifier_bound_repairs(
            output_ast,
            original_pattern,
            count_range_oversized_variables,
            count_range_define_guards,
        )
        if quantifier_repairs:
            # This is a defensive fallback for an early pipeline stop or a
            # bound introduced after RegexOptimizationPlusRule.
            repair_metric = next(
                (metric for metric in reversed(metrics) if metric.applied),
                metrics[-1],
            )
            repair_metric.details["quantifier_output_repairs"] = quantifier_repairs
            repair_metric.details["quantifier_bound_limit"] = (
                MAX_PORTABLE_QUANTIFIER_BOUND
            )
            if restored_define_guards:
                repair_metric.details["quantifier_restored_define_guards"] = (
                    restored_define_guards
                )
            if refreshed_raw_sql_override:
                repair_metric.details["quantifier_refreshed_raw_sql_override"] = True
        output_sql = output_ast.to_sql()
    else:
        output_sql = normalized_input

    total_elapsed = time.perf_counter() - total_started
    sampling_seconds = sum(metric.sampling_seconds for metric in metrics)
    return RuleBasedRewriteResult(
        sql=_sql_for_system(output_sql, system),
        applied_rules=applied_rules,
        total_overhead_seconds=_round_seconds(total_elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(total_elapsed - sampling_seconds),
        rule_metrics=metrics,
        stopped_after_rule=stopped_after_rule,
    )


class OracleSampleExecutor:
    """Oracle implementation using one already-open connection."""

    def __init__(
        self,
        connection,
        *,
        fetch_size: int = DEFAULT_FETCH_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        if fetch_size < 1:
            raise ValueError("fetch_size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.connection = connection
        self.fetch_size = fetch_size
        self.timeout_seconds = timeout_seconds

    def _set_remaining_timeout(self, deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("sample query exceeded its time limit")
        self.connection.callTimeout = max(1, math.ceil(remaining * 1_000))

    def fetch_one(self, sql: str) -> Mapping[str, object]:
        return self.fetch_one_with_timeout(sql, self.timeout_seconds)

    def fetch_one_with_timeout(
        self, sql: str, timeout_seconds: float
    ) -> Mapping[str, object]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.perf_counter() + min(self.timeout_seconds, timeout_seconds)
        with self.connection.cursor() as cursor:
            self._set_remaining_timeout(deadline)
            cursor.execute(_normalize_sql(sql))
            self._set_remaining_timeout(deadline)
            row = cursor.fetchone()
            if row is None or cursor.description is None:
                raise ValueError("sample query returned no row")
            return {
                description[0]: value
                for description, value in zip(cursor.description, row)
            }

    def measure_query(self, sql: str, runs: int) -> QueryMeasurement:
        return self.measure_query_with_timeout(sql, runs, self.timeout_seconds)

    def measure_query_with_timeout(
        self, sql: str, runs: int, timeout_seconds: float
    ) -> QueryMeasurement:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        effective_timeout = min(self.timeout_seconds, timeout_seconds)
        measurements: list[float] = []
        fetched_counts: list[int] = []
        for _ in range(runs):
            deadline = time.perf_counter() + effective_timeout
            fetched_rows = 0
            with self.connection.cursor() as cursor:
                cursor.arraysize = self.fetch_size
                cursor.prefetchrows = self.fetch_size
                started = time.perf_counter()
                try:
                    self._set_remaining_timeout(deadline)
                    cursor.execute(_normalize_sql(sql))
                    while cursor.description:
                        self._set_remaining_timeout(deadline)
                        rows = cursor.fetchmany(self.fetch_size)
                        fetched_rows += len(rows)
                        if not rows:
                            break
                except Exception as exc:
                    if not self._is_timeout_exception(exc):
                        raise
                    return QueryMeasurement(
                        completed=False,
                        elapsed_seconds=time.perf_counter() - started,
                        fetched_rows=fetched_rows,
                    )
                measurements.append(time.perf_counter() - started)
                fetched_counts.append(fetched_rows)
        return QueryMeasurement(
            completed=True,
            elapsed_seconds=statistics.median(measurements),
            fetched_rows=int(statistics.median(fetched_counts)),
        )

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).upper()
        return "DPI-1067" in message or "ORA-01013" in message


def _load_payload(path: Path) -> tuple[dict[str, Any], dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    table_schemas = payload.get("table_schemas", {})
    sqls = payload.get("sqls", [])
    if not isinstance(table_schemas, dict):
        raise ValueError("'table_schemas' must be an object")
    if not isinstance(sqls, list):
        raise ValueError("'sqls' must be an array")
    return payload, table_schemas, sqls


def rewrite_json(
    json_path: Path,
    *,
    system: str = DEFAULT_SYSTEM,
    sample_input_table: str | None = None,
    executor: SampleExecutor | None = None,
    sample_must_appear_partition_filter: bool = False,
    sample_window_condition_prefilter: bool = False,
    output_path: Path | None = None,
    metrics_path: Path | None = None,
    benchmark_runs: int = DEFAULT_RUNS,
    max_pair_candidates: int | None = None,
    partition_pruning_min_row_reduction: float = (
        DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION
    ),
    column_pruning_via_join_max_output_ratio: float = (
        DEFAULT_COLUMN_PRUNING_VIA_JOIN_MAX_OUTPUT_RATIO
    ),
) -> tuple[Path, Path]:
    system = _normalize_system(system)
    payload, table_schemas, sqls = _load_payload(json_path)
    output_path = output_path or json_path.with_name(f"{json_path.stem}_our_method.txt")
    metrics_path = metrics_path or json_path.with_name(
        f"{json_path.stem}_our_method.metrics.json"
    )

    started_at = datetime.now(timezone.utc)
    output_blocks = []
    query_metrics = []
    for index, item in enumerate(sqls, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"sqls[{index - 1}] must be an object")
        label = item.get("url") or f"sqls[{index - 1}]"
        sql = item.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{label}: 'sql' must be a non-empty string")
        try:
            result = rewrite_sql(
                sql,
                table_schemas,
                system=system,
                sample_input_table=sample_input_table,
                executor=executor,
                sample_must_appear_partition_filter=(
                    sample_must_appear_partition_filter
                ),
                sample_window_condition_prefilter=(sample_window_condition_prefilter),
                benchmark_runs=benchmark_runs,
                max_pair_candidates=max_pair_candidates,
                partition_pruning_min_row_reduction=(
                    partition_pruning_min_row_reduction
                ),
                column_pruning_via_join_max_output_ratio=(
                    column_pruning_via_join_max_output_ratio
                ),
            )
        except Exception as exc:
            note = f"while rewriting {label} (item {index})"
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            else:
                exc.args = (*exc.args, note)
            raise

        normalized_output = _normalize_sql(result.sql)
        output_blocks.append(f"{normalized_output};")
        query_metrics.append(
            {
                "index": index,
                "label": label,
                "input_sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                "output_sql_sha256": hashlib.sha256(
                    normalized_output.encode("utf-8")
                ).hexdigest(),
                "changed": bool(result.applied_rules),
                "applied_rules": result.applied_rules,
                "stopped_after_rule": result.stopped_after_rule,
                "total_overhead_seconds": result.total_overhead_seconds,
                "sampling_seconds": result.sampling_seconds,
                "local_rewrite_seconds": result.local_rewrite_seconds,
                "rules": [asdict(metric) for metric in result.rule_metrics],
            }
        )
        print(
            f"[{index}/{len(sqls)}] {result.total_overhead_seconds:.3f}s "
            f"(sampling {result.sampling_seconds:.3f}s): "
            f"{', '.join(result.applied_rules) or 'unchanged'}",
            file=sys.stderr,
        )

    report = {
        "source_json": str(json_path.resolve()),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": payload.get("database"),
        "database_version": payload.get("database_version"),
        "target_system": system,
        "sample_input_table": sample_input_table,
        "pipeline": [spec.factory().name for spec in RULE_PIPELINE],
        "settings": {
            "system": system,
            "sample_must_appear_partition_filter": (
                sample_must_appear_partition_filter
            ),
            "sample_window_condition_prefilter": (sample_window_condition_prefilter),
            "benchmark_runs": benchmark_runs,
            "max_pair_candidates": max_pair_candidates,
            "partition_pruning_min_row_reduction": (
                partition_pruning_min_row_reduction
            ),
            "column_pruning_via_join_max_output_ratio": (
                column_pruning_via_join_max_output_ratio
            ),
            "pair_candidate_timeout_multiplier": (PAIR_CANDIDATE_TIMEOUT_MULTIPLIER),
            "sample_timeout_seconds": getattr(executor, "timeout_seconds", None),
            "fetch_size": getattr(executor, "fetch_size", None),
        },
        "totals": {
            "query_count": len(query_metrics),
            "changed_query_count": sum(item["changed"] for item in query_metrics),
            "short_circuited_query_count": sum(
                item["stopped_after_rule"] is not None for item in query_metrics
            ),
            "total_overhead_seconds": _round_seconds(
                sum(item["total_overhead_seconds"] for item in query_metrics)
            ),
            "sampling_seconds": _round_seconds(
                sum(item["sampling_seconds"] for item in query_metrics)
            ),
            "local_rewrite_seconds": _round_seconds(
                sum(item["local_rewrite_seconds"] for item in query_metrics)
            ),
        },
        "queries": query_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(output_blocks) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, metrics_path


def resolve_json_path(argument: str | None) -> Path:
    if argument is None:
        return DATA_DIR / "test_sql.json"
    path = Path(argument).expanduser()
    if not path.is_absolute() and not path.exists():
        path = DATA_DIR / path
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", help="input JSON file")
    parser.add_argument(
        "--system",
        choices=SUPPORTED_SYSTEMS,
        default=DEFAULT_SYSTEM,
        help=(
            "target SQL system; Oracle (default) emits FETCH FIRST n ROWS "
            "ONLY, while Trino keeps LIMIT n"
        ),
    )
    parser.add_argument(
        "--sample-must-appear-partition-filter",
        action="store_true",
        help=(
            "apply MustAppearPartitionFilterPushdownRule only when its sample "
            "filter reduces both partitions and enough input rows; without "
            "this flag the applicable rule is enabled directly"
        ),
    )
    parser.add_argument(
        "--sample-window-condition-prefilter",
        action="store_true",
        help=(
            "enable WindowConditionPreFilterRule and select its candidate on "
            "the sample; without this flag the rule is disabled"
        ),
    )
    parser.add_argument(
        "--sample-table",
        help=(
            "sample relation used by enabled sampling rules and by the "
            "column-pruning-via-join output-cardinality guard"
        ),
    )
    parser.add_argument("--output", type=Path, help="rewritten SQL text path")
    parser.add_argument("--metrics-output", type=Path, help="JSON metrics path")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--fetch-size", type=int, default=DEFAULT_FETCH_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-pair-candidates", type=int)
    parser.add_argument(
        "--partition-pruning-min-row-reduction",
        type=float,
        default=DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION,
        help=(
            "minimum fraction of sample rows that partition pruning must "
            "remove; the reduction must be strictly greater (default: 0.20)"
        ),
    )
    parser.add_argument(
        "--column-pruning-via-join-max-output-ratio",
        type=float,
        default=DEFAULT_COLUMN_PRUNING_VIA_JOIN_MAX_OUTPUT_RATIO,
        help=(
            "maximum sample MATCH_RECOGNIZE output rows per sample input row "
            "for non-PAST-LAST column-pruning-via-join rewrites "
            "(default: 1.5)"
        ),
    )
    parser.add_argument(
        "--schema",
        help="Oracle schema (default: EXPR) or Trino schema (default: public)",
    )
    parser.add_argument(
        "--admin-user", default=os.environ.get("ORACLE_ADMIN_USER", "system")
    )
    parser.add_argument("--host", default=os.environ.get("ORACLE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ORACLE_PORT", "1521"))
    )
    parser.add_argument(
        "--service-name", default=os.environ.get("ORACLE_SERVICE_NAME", "ORCL")
    )
    parser.add_argument(
        "--catalog", default=os.environ.get("TRINO_CATALOG", "postgresql")
    )
    parser.add_argument("--trino-url", default=os.environ.get("TRINO_URL"))
    parser.add_argument("--trino-user", default=os.environ.get("TRINO_USER", "demo"))
    parser.add_argument("--trino-source", default="our-method-sampling")
    return parser


def _rewrite_from_cli(args, json_path: Path, executor=None):
    return rewrite_json(
        json_path,
        system=args.system,
        sample_input_table=args.sample_table,
        executor=executor,
        sample_must_appear_partition_filter=(args.sample_must_appear_partition_filter),
        sample_window_condition_prefilter=(args.sample_window_condition_prefilter),
        output_path=args.output,
        metrics_path=args.metrics_output,
        benchmark_runs=args.runs,
        max_pair_candidates=args.max_pair_candidates,
        partition_pruning_min_row_reduction=(args.partition_pruning_min_row_reduction),
        column_pruning_via_join_max_output_ratio=(
            args.column_pruning_via_join_max_output_ratio
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.fetch_size < 1:
        parser.error("--fetch-size must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_pair_candidates is not None and args.max_pair_candidates < 0:
        parser.error("--max-pair-candidates must be non-negative")
    if not 0.0 <= args.partition_pruning_min_row_reduction <= 1.0:
        parser.error("--partition-pruning-min-row-reduction must be in [0, 1]")
    if (
        not math.isfinite(args.column_pruning_via_join_max_output_ratio)
        or args.column_pruning_via_join_max_output_ratio < 0
    ):
        parser.error(
            "--column-pruning-via-join-max-output-ratio must be finite and "
            "non-negative"
        )

    sampling_requested = bool(args.sample_table) or (
        args.sample_must_appear_partition_filter
        or args.sample_window_condition_prefilter
    )
    if (
        args.sample_must_appear_partition_filter
        or args.sample_window_condition_prefilter
    ) and not args.sample_table:
        parser.error("--sample-table is required when sampling is enabled")
    json_path = resolve_json_path(args.json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    if not sampling_requested:
        output_path, metrics_path = _rewrite_from_cli(args, json_path)
    elif args.system == "oracle":
        schema = (args.schema or "EXPR").upper()
        if not schema.isidentifier():
            parser.error("--schema must be an unquoted SQL identifier")

        # Keep Oracle optional for ordinary, non-sampling rewrites.
        from experiments.connect_oracle import (
            OracleConnection,
            password_from_environment_or_prompt,
        )

        password = password_from_environment_or_prompt(
            "ORACLE_ADMIN_PASSWORD", args.admin_user
        )
        oracle = OracleConnection(
            args.admin_user,
            password,
            args.host,
            args.port,
            args.service_name,
        )
        with oracle.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {schema}")
            executor = OracleSampleExecutor(
                connection,
                fetch_size=args.fetch_size,
                timeout_seconds=args.timeout,
            )
            output_path, metrics_path = _rewrite_from_cli(args, json_path, executor)
    else:
        from experiments.build_data_sample import TrinoSampleExecutor

        schema = args.schema or "public"
        executor = TrinoSampleExecutor.connect(
            trino_url=args.trino_url,
            user=args.trino_user,
            source=args.trino_source,
            catalog=args.catalog,
            schema=schema,
            fetch_size=args.fetch_size,
            timeout_seconds=args.timeout,
        )
        output_path, metrics_path = _rewrite_from_cli(args, json_path, executor)

    print(f"rewritten queries written to {output_path}")
    print(f"rewrite metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
