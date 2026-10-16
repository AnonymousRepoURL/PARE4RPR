"""Run the non-LLM baseline rewrite pipeline.

The baseline combines the three published rewrites used by this project with
``AlternationVariableMergeRule``, ``IncrementalConditionPushdownRule``, and
``DefinePartitionPushdownRule``::

    alternation merge -> incremental pushdown -> define-partition pushdown
        -> sequence filtering -> window-condition prefilter -> row filtering

Rules come from the following two papers:        
[1] High-Performance Row Pattern Recognition Using Joins.
[2] Optimization of Row Pattern Matching over Sequence Data in Spark SQL.

Sequence filtering and the window-condition prefilter are adaptive.  They are
first evaluated on a caller-supplied sample relation.  Sequence filtering is
kept only when it removes at least one sample partition and more than a
configurable fraction of sample rows (20% by default); it is rejected when a
PARTITION BY column is nullable according to schema metadata. Window candidates are
kept only when they remove sample input rows and are executed with a
five-second default budget. If any candidate completes, the fastest completed
candidate wins; otherwise the candidate that fetched the most rows before
timing out wins. The unmodified query is always benchmarked too, so sampling
can choose no rewrite. The first completed single- or pair-variable candidate
establishes a dynamic baseline. Until then pair candidates keep using the
configured query timeout. A later faster candidate tightens the budget to at
most twice the new baseline and prunes cached candidates which no longer fit
that budget. Every candidate is measured only once.

Example (Oracle)::

python3 sota_method.py data/crimes_sql.json \
  --sample-table CRIMES_SAMPLE \
  --schema EXPR \
  --runs 3 \
  --timeout 5

Example (Trino)::

python sota_method.py data/crimes_sql_trino.json \
  --system trino \
  --sample-table crimes_sample \
  --schema public \
  --runs 3 \
  --timeout 5

The SQL output remains a plain, semicolon-delimited text file.  A JSON sidecar
records per-query and per-rule rewrite overhead, including database sampling
time separately from local rewrite time.
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
from typing import Any, Mapping, Protocol, Sequence

from queryanalysis import RPRAbstractSyntaxTree
from rewriting_rules import (
    AlternationVariableMergeRule,
    DefinePartitionPushdownRule,
    IncrementalConditionPushdownRule,
    RowFilteringRule,
    SequenceFilteringRule,
    WindowConditionPreFilterRule,
    WindowPrefilterBenchmarkPlan,
    WindowPrefilterBenchmarkQuery,
    WindowPrefilterCandidate,
)
from rewriting_rules.utils import bind_input_relation_alias, resolve_input_rebind_alias

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RUNS = 3
DEFAULT_FETCH_SIZE = 2_000
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION = 0.20
PAIR_CANDIDATE_TIMEOUT_MULTIPLIER = 2.0
DEFAULT_SYSTEM = "oracle"
SUPPORTED_SYSTEMS = ("oracle", "trino")


def _normalize_system(system: str) -> str:
    normalized = str(system).strip().lower()
    if normalized not in SUPPORTED_SYSTEMS:
        supported = ", ".join(SUPPORTED_SYSTEMS)
        raise ValueError(f"system must be one of: {supported}")
    return normalized


class SampleExecutor(Protocol):
    """The small database interface needed by the adaptive rules."""

    def fetch_one(self, sql: str) -> Mapping[str, object]: ...

    def measure_query(self, sql: str, runs: int) -> "QueryMeasurement | float": ...


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
class BaselineRewriteResult:
    sql: str
    applied_rules: list[str]
    total_overhead_seconds: float
    sampling_seconds: float
    local_rewrite_seconds: float
    rule_metrics: list[RuleMetric]


@dataclass(frozen=True)
class QueryMeasurement:
    """Outcome of executing and fetching one sample candidate."""

    completed: bool
    elapsed_seconds: float
    fetched_rows: int


def _round_seconds(value: float) -> float:
    return round(max(0.0, value), 6)


def _coerce_measurement(value: QueryMeasurement | float) -> QueryMeasurement:
    """Accept legacy float results used by third-party and fake executors."""

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


def _sample_ast(ast: RPRAbstractSyntaxTree, sample_input_table: str):
    """Copy ``ast`` and replace only its MATCH_RECOGNIZE input relation."""

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
        # Preserve derived-table filters. For example, sampling
        # ``(SELECT * FROM NASDAQ WHERE TAPE = 'C')`` must only replace the
        # physical NASDAQ relation, not discard the entire subquery.
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

    # The sample table/view has the same columns as the production relation,
    # but its physical name is normally absent from the input JSON schema. Add
    # that name to the copied metadata so expression simplification remains
    # available while applying rules to the sample AST.
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
        # A derived MATCH_RECOGNIZE input such as ``(SELECT * FROM T ...)``
        # is not itself a schema key. Workloads in this project describe one
        # physical source table, so that sole schema is the safe fallback.
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


def _input_row_count_sql(ast: RPRAbstractSyntaxTree) -> str:
    return _with_prefix(
        ast,
        f"SELECT COUNT(*) AS RPR_UNIT_COUNT FROM {ast.input_table.strip()}",
    )


def _partition_count_sql(ast: RPRAbstractSyntaxTree) -> str:
    columns = []
    for expression in ast.partition_by or []:
        column = expression.strip().split(".")[-1]
        if not column:
            raise ValueError("empty PARTITION BY expression")
        columns.append(column)
    if not columns:
        return _input_row_count_sql(ast)
    projection = ", ".join(columns)
    sql = (
        "SELECT COUNT(*) AS RPR_UNIT_COUNT FROM "
        f"(SELECT DISTINCT {projection} FROM {ast.input_table.strip()}) "
        "RPR_PARTITIONS"
    )
    return _with_prefix(ast, sql)


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
    return _with_prefix(before_ast, aggregate)


def _count_partition_filter_rows(
    executor: SampleExecutor,
    before_ast: RPRAbstractSyntaxTree,
    after_ast: RPRAbstractSyntaxTree,
) -> tuple[float, float, float]:
    started = time.perf_counter()
    row = executor.fetch_one(_partition_filter_row_counts_sql(before_ast, after_ast))
    elapsed = time.perf_counter() - started
    return (
        _numeric_result(row, "RPR_ROWS_BEFORE"),
        _numeric_result(row, "RPR_ROWS_AFTER"),
        elapsed,
    )


def _count_sample_units(
    executor: SampleExecutor,
    ast: RPRAbstractSyntaxTree,
    *,
    partitions: bool,
    timeout_seconds: float | None = None,
) -> tuple[float, float]:
    sql = _partition_count_sql(ast) if partitions else _input_row_count_sql(ast)
    started = time.perf_counter()
    row = _fetch_one_with_budget(executor, sql, timeout_seconds)
    elapsed = time.perf_counter() - started
    return _numeric_result(row, "RPR_UNIT_COUNT"), elapsed


def _apply_local_rule_pair(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    rule_type,
) -> RuleMetric:
    started = time.perf_counter()
    production_rule = rule_type()
    applicable = production_rule.is_applicable(production_ast)
    applied = production_rule.apply(production_ast) if applicable else False

    if sample_ast is not None:
        sample_rule = rule_type()
        sample_applicable = sample_rule.is_applicable(sample_ast)
        sample_applied = sample_rule.apply(sample_ast) if sample_applicable else False
        if (sample_applicable, sample_applied) != (applicable, applied):
            raise RuntimeError(
                f"{production_rule.name} diverged between full and sample ASTs"
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


def _apply_adaptive_sequence_filter(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    executor: SampleExecutor | None,
    *,
    minimum_row_reduction: float,
) -> tuple[RPRAbstractSyntaxTree | None, RuleMetric]:
    started = time.perf_counter()
    sampling_seconds = 0.0
    rule = SequenceFilteringRule()
    applicable = rule.is_applicable(production_ast)
    details: dict[str, Any] = {}
    applied = False
    decision = "not_applicable"
    nullable_partition_columns = [
        column.strip()
        for column in (production_ast.partition_by or [])
        if column.strip() and production_ast.is_column_nullable(column) is True
    ]

    if applicable and nullable_partition_columns:
        decision = "kept_original_nullable_partition_key"
        details["nullable_partition_columns"] = nullable_partition_columns
    elif applicable and (sample_ast is None or executor is None):
        decision = "kept_original_no_sampler"
    elif applicable:
        try:
            candidate_sample = copy.deepcopy(sample_ast)
            if not SequenceFilteringRule().apply(candidate_sample):
                decision = "kept_original_sample_candidate_failed"
            else:
                before, elapsed = _count_sample_units(
                    executor, sample_ast, partitions=True
                )
                sampling_seconds += elapsed
                after, elapsed = _count_sample_units(
                    executor, candidate_sample, partitions=True
                )
                sampling_seconds += elapsed
                details.update(
                    {
                        "sample_partitions_before": before,
                        "sample_partitions_after": after,
                        "sample_partitions_removed": max(0.0, before - after),
                    }
                )
                if after < before:
                    rows_before, rows_after, elapsed = _count_partition_filter_rows(
                        executor, sample_ast, candidate_sample
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
                        applied = rule.apply(production_ast)
                        if applied:
                            sample_ast = candidate_sample
                            decision = "applied_sample_row_reduction"
                        else:
                            decision = "kept_original_apply_failed"
                else:
                    decision = "kept_original_no_partition_removed"
        except Exception as exc:
            decision = "kept_original_sampling_error"
            details["error"] = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return sample_ast, RuleMetric(
        rule=rule.name,
        applicable=applicable,
        applied=applied,
        decision=decision,
        overhead_seconds=_round_seconds(elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(elapsed - sampling_seconds),
        details=details,
    )


def _apply_adaptive_window_prefilter(
    production_ast: RPRAbstractSyntaxTree,
    sample_ast: RPRAbstractSyntaxTree | None,
    executor: SampleExecutor | None,
    *,
    benchmark_runs: int,
    max_pair_candidates: int | None,
) -> tuple[RPRAbstractSyntaxTree | None, RuleMetric]:
    started = time.perf_counter()
    sampling_seconds = 0.0
    rule = WindowConditionPreFilterRule()
    applicable = rule.is_applicable(production_ast)
    details: dict[str, Any] = {}
    applied = False
    decision = "not_applicable"

    if applicable and (sample_ast is None or executor is None):
        decision = "kept_original_no_sampler"
    elif applicable:
        try:
            request = rule.build_statistics_request(sample_ast)
            if request is None:
                raise ValueError("statistics request could not be built")
            sample_started = time.perf_counter()
            statistics_row = executor.fetch_one(request.sql)
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

            before_rows, elapsed = _count_sample_units(
                executor, sample_ast, partitions=False
            )
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
                candidate_asts[candidate.candidate_id] = candidate_ast
                measured = None
                try:
                    sample_started = time.perf_counter()
                    measured = _measure_query_with_budget(
                        executor, candidate_ast.to_sql(), benchmark_runs
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
                    after_rows, elapsed = _count_sample_units(
                        executor, candidate_ast, partitions=False
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
                    after_rows, elapsed = _count_sample_units(
                        executor,
                        candidate_ast,
                        partitions=False,
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
                # Once a baseline exists, a row-count timeout already proves
                # that this candidate exceeds the adaptive budget. Before a
                # baseline exists, still run the full pair so it can establish one.
                if not row_count_known and base_time is not None:
                    continue

                try:
                    sample_started = time.perf_counter()
                    measured = _measure_query_with_budget(
                        executor,
                        candidate_ast.to_sql(),
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
                        executor, sample_ast.to_sql(), benchmark_runs
                    )
                )
            except Exception as exc:
                details.setdefault("candidate_errors", {})[
                    no_rewrite.candidate_id
                ] = f"{type(exc).__name__}: {exc}"
            finally:
                sampling_seconds += time.perf_counter() - sample_started

            benchmark_queries = [
                WindowPrefilterBenchmarkQuery(
                    candidate, candidate_asts[candidate.candidate_id].to_sql()
                )
                for candidate in eligible
                if candidate.candidate_id in candidate_asts
            ]
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
            # Retain this legacy field for consumers that only understand
            # completed execution times.
            details["candidate_elapsed_seconds"] = {
                candidate_id: measurement.elapsed_seconds
                for candidate_id, measurement in measurements_by_candidate.items()
                if measurement.completed
            }
            details["benchmark_runs"] = benchmark_runs

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
                plan = WindowPrefilterBenchmarkPlan(
                    window_statistics, tuple(benchmark_queries)
                )
                candidates_by_id = {
                    query.candidate.candidate_id: query.candidate
                    for query in plan.queries
                }
                completed = [
                    (measurement, candidates_by_id[candidate_id])
                    for candidate_id, measurement in measurements_by_candidate.items()
                    if measurement.completed and candidate_id in candidates_by_id
                ]
                if completed:
                    # A completed query always beats a timed-out query. Prefer
                    # no rewrite on an exact tie to keep selection deterministic.
                    completed.sort(
                        key=lambda item: (
                            item[0].elapsed_seconds,
                            0 if item[1].strategy == "no_rewrite" else 1,
                            item[1].candidate_id,
                        )
                    )
                    selected_measurement, selected_candidate = completed[0]
                    selection_reason = "fastest_completed"
                else:
                    timed_out = [
                        (measurement, candidates_by_id[candidate_id])
                        for candidate_id, measurement in (
                            measurements_by_candidate.items()
                        )
                        if candidate_id in candidates_by_id
                    ]
                    timed_out.sort(
                        key=lambda item: (
                            -item[0].fetched_rows,
                            item[0].elapsed_seconds,
                            0 if item[1].strategy == "no_rewrite" else 1,
                            item[1].candidate_id,
                        )
                    )
                    selected_measurement, selected_candidate = timed_out[0]
                    selection_reason = "most_rows_fetched_before_timeout"

                details["selected_candidate"] = selected_candidate.candidate_id
                details["selected_sample_seconds"] = (
                    selected_measurement.elapsed_seconds
                )
                details["selected_fetched_rows"] = selected_measurement.fetched_rows
                details["selection_reason"] = selection_reason
                original = measurements_by_candidate.get("no_rewrite")
                details["sample_speedup"] = (
                    original.elapsed_seconds / selected_measurement.elapsed_seconds
                    if original is not None
                    and original.completed
                    and selected_measurement.completed
                    and selected_measurement.elapsed_seconds > 0
                    else None
                )
                if selected_candidate.strategy == "no_rewrite":
                    decision = (
                        "kept_original_fastest_completed_candidate"
                        if selected_measurement.completed
                        else "kept_original_most_rows_fetched_before_timeout"
                    )
                else:
                    rewritten = rule.rewrite_candidate(
                        production_ast, selected_candidate, window_statistics
                    )
                    if rewritten is None:
                        decision = "kept_original_apply_failed"
                    else:
                        production_ast.__dict__.clear()
                        production_ast.__dict__.update(rewritten.__dict__)
                        sample_ast = candidate_asts[selected_candidate.candidate_id]
                        applied = True
                        decision = (
                            "applied_fastest_completed_sample_candidate"
                            if selected_measurement.completed
                            else "applied_most_rows_fetched_sample_candidate"
                        )
        except Exception as exc:
            decision = "kept_original_sampling_error"
            details["error"] = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    return sample_ast, RuleMetric(
        rule=rule.name,
        applicable=applicable,
        applied=applied,
        decision=decision,
        overhead_seconds=_round_seconds(elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(elapsed - sampling_seconds),
        details=details,
    )


def rewrite_sql(
    sql: str,
    table_schemas: dict,
    *,
    sample_input_table: str | None = None,
    executor: SampleExecutor | None = None,
    benchmark_runs: int = DEFAULT_RUNS,
    max_pair_candidates: int | None = None,
    partition_pruning_min_row_reduction: float = (
        DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION
    ),
    system: str = DEFAULT_SYSTEM,
) -> BaselineRewriteResult:
    """Rewrite one query and return SQL plus detailed overhead measurements."""

    if benchmark_runs < 1:
        raise ValueError("benchmark_runs must be positive")
    if max_pair_candidates is not None and max_pair_candidates < 0:
        raise ValueError("max_pair_candidates must be non-negative")
    if not 0.0 <= partition_pruning_min_row_reduction <= 1.0:
        raise ValueError("partition_pruning_min_row_reduction must be in [0, 1]")
    _normalize_system(system)

    total_started = time.perf_counter()
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(_normalize_sql(sql), table_schemas)
    sampled_ast = _sample_ast(ast, sample_input_table) if sample_input_table else None

    metrics: list[RuleMetric] = []
    # These structural rules run first because they can simplify predicates and
    # expose better candidates to the adaptive baseline rewrites. Partition
    # pushdown precedes sampling so both adaptive rules observe the filtered
    # sample that corresponds to the production query.
    metrics.append(
        _apply_local_rule_pair(ast, sampled_ast, AlternationVariableMergeRule)
    )
    metrics.append(
        _apply_local_rule_pair(ast, sampled_ast, IncrementalConditionPushdownRule)
    )
    metrics.append(
        _apply_local_rule_pair(ast, sampled_ast, DefinePartitionPushdownRule)
    )

    sampled_ast, metric = _apply_adaptive_sequence_filter(
        ast,
        sampled_ast,
        executor,
        minimum_row_reduction=partition_pruning_min_row_reduction,
    )
    metrics.append(metric)
    sampled_ast, metric = _apply_adaptive_window_prefilter(
        ast,
        sampled_ast,
        executor,
        benchmark_runs=benchmark_runs,
        max_pair_candidates=max_pair_candidates,
    )
    metrics.append(metric)
    metrics.append(_apply_local_rule_pair(ast, sampled_ast, RowFilteringRule))

    total_elapsed = time.perf_counter() - total_started
    sampling_seconds = sum(item.sampling_seconds for item in metrics)
    applied_rules = [item.rule for item in metrics if item.applied]
    return BaselineRewriteResult(
        # Preserve the caller's SQL text when every rule declines the query.
        # This makes the adaptive "no rewrite" result a literal fallback, not
        # merely an equivalent AST pretty-print.
        sql=ast.to_sql() if applied_rules else _normalize_sql(sql),
        applied_rules=applied_rules,
        total_overhead_seconds=_round_seconds(total_elapsed),
        sampling_seconds=_round_seconds(sampling_seconds),
        local_rewrite_seconds=_round_seconds(total_elapsed - sampling_seconds),
        rule_metrics=metrics,
    )


class OracleSampleExecutor:
    """Oracle implementation that uses one already-open connection."""

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
        # cx_Oracle uses milliseconds and interrupts a blocking OCI call.
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
        # cx_Oracle callTimeout normally reports DPI-1067. ORA-01013 is kept
        # for compatible Oracle client versions that surface cancellation.
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
    sample_input_table: str,
    executor: SampleExecutor,
    output_path: Path | None = None,
    metrics_path: Path | None = None,
    benchmark_runs: int = DEFAULT_RUNS,
    max_pair_candidates: int | None = None,
    partition_pruning_min_row_reduction: float = (
        DEFAULT_PARTITION_PRUNING_MIN_ROW_REDUCTION
    ),
    system: str = DEFAULT_SYSTEM,
) -> tuple[Path, Path]:
    system = _normalize_system(system)
    payload, table_schemas, sqls = _load_payload(json_path)
    output_path = output_path or json_path.with_name(
        f"{json_path.stem}_sota_method.txt"
    )
    metrics_path = metrics_path or json_path.with_name(
        f"{json_path.stem}_sota_method.metrics.json"
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
                sample_input_table=sample_input_table,
                executor=executor,
                benchmark_runs=benchmark_runs,
                max_pair_candidates=max_pair_candidates,
                partition_pruning_min_row_reduction=(
                    partition_pruning_min_row_reduction
                ),
                system=system,
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
        "pipeline": [
            "alternation_variable_merge",
            "incremental_condition_pushdown",
            "define_partition_pushdown",
            "sequence_filtering",
            "window_condition_pre_filter",
            "row_filtering",
        ],
        "settings": {
            "benchmark_runs": benchmark_runs,
            "max_pair_candidates": max_pair_candidates,
            "partition_pruning_min_row_reduction": (
                partition_pruning_min_row_reduction
            ),
            "pair_candidate_timeout_multiplier": (PAIR_CANDIDATE_TIMEOUT_MULTIPLIER),
            "sample_timeout_seconds": getattr(executor, "timeout_seconds", None),
            "fetch_size": getattr(executor, "fetch_size", None),
        },
        "totals": {
            "query_count": len(query_metrics),
            "changed_query_count": sum(item["changed"] for item in query_metrics),
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
        help="sampling backend and target SQL system (default: oracle)",
    )
    parser.add_argument(
        "--sample-table",
        required=True,
        help="existing sampled relation used by the two adaptive rules",
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
    parser.add_argument("--trino-source", default="sota-method-sampling")
    return parser


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

    json_path = resolve_json_path(args.json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if args.system == "oracle":
        schema = (args.schema or "EXPR").upper()
        if not schema.isidentifier():
            parser.error("--schema must be an unquoted SQL identifier")

        # Keep the optional Oracle dependency out of import-time code so the
        # orchestration API and its fake executors remain usable in unit tests.
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
            output_path, metrics_path = rewrite_json(
                json_path,
                sample_input_table=args.sample_table,
                executor=executor,
                output_path=args.output,
                metrics_path=args.metrics_output,
                benchmark_runs=args.runs,
                max_pair_candidates=args.max_pair_candidates,
                partition_pruning_min_row_reduction=(
                    args.partition_pruning_min_row_reduction
                ),
                system=args.system,
            )
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
        output_path, metrics_path = rewrite_json(
            json_path,
            sample_input_table=args.sample_table,
            executor=executor,
            output_path=args.output,
            metrics_path=args.metrics_output,
            benchmark_runs=args.runs,
            max_pair_candidates=args.max_pair_candidates,
            partition_pruning_min_row_reduction=(
                args.partition_pruning_min_row_reduction
            ),
            system=args.system,
        )

    print(f"rewritten queries written to {output_path}")
    print(f"rewrite metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
