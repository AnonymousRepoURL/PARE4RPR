"""Adaptive orchestration for the window-condition prefilter rewrite.

The rewrite rule remains responsible for SQL construction.  This module
coordinates the two external observations it needs: one statistics row and
one elapsed-time value for each SQL executed on a caller-provided sample.
It deliberately contains no database driver code, so callers can use Oracle,
another engine, or simulated measurements in tests and demonstrations.
"""

import copy
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from queryanalysis import RPRAbstractSyntaxTree
from .utils import bind_input_relation_alias, resolve_input_rebind_alias
from .window_cond_prefilter import (
    WindowConditionPreFilterRule,
    WindowPrefilterCandidate,
    WindowPrefilterStatistics,
    WindowPrefilterStatisticsRequest,
)


@dataclass(frozen=True)
class WindowPrefilterBenchmarkQuery:
    candidate: WindowPrefilterCandidate
    sample_sql: str


@dataclass(frozen=True)
class WindowPrefilterBenchmarkPlan:
    statistics: WindowPrefilterStatistics
    queries: Tuple[WindowPrefilterBenchmarkQuery, ...]


@dataclass(frozen=True)
class WindowPrefilterOptimizationResult:
    candidate: WindowPrefilterCandidate
    elapsed_seconds: float
    sql: str


class WindowPrefilterOptimizer:
    """Prepare candidate SQL and select a winner from supplied measurements."""

    def __init__(
        self,
        rule: Optional[WindowConditionPreFilterRule] = None,
        *,
        include_original: bool = True,
        max_pair_candidates: Optional[int] = None,
        max_pair_estimated_cost: Optional[float] = None,
        minimum_improvement_ratio: float = 0.0,
    ):
        if not 0.0 <= minimum_improvement_ratio < 1.0:
            raise ValueError("minimum_improvement_ratio must be in [0, 1)")
        self.rule = rule or WindowConditionPreFilterRule()
        self.include_original = include_original
        self.max_pair_candidates = max_pair_candidates
        self.max_pair_estimated_cost = max_pair_estimated_cost
        self.minimum_improvement_ratio = minimum_improvement_ratio

    def build_statistics_request(
        self,
        ast: RPRAbstractSyntaxTree,
        sample_input_table: str,
    ) -> WindowPrefilterStatisticsRequest:
        request = self.rule.build_statistics_request(
            ast, input_table_override=sample_input_table
        )
        if request is None:
            raise ValueError("window-condition prefilter is not applicable")
        return request

    def build_benchmark_plan_from_row(
        self,
        ast: RPRAbstractSyntaxTree,
        sample_input_table: str,
        statistics_row: Mapping[str, object],
    ) -> WindowPrefilterBenchmarkPlan:
        request = self.build_statistics_request(ast, sample_input_table)
        return self.build_benchmark_plan(
            ast,
            sample_input_table,
            request.parse_row(statistics_row),
        )

    def build_benchmark_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        sample_input_table: str,
        statistics: WindowPrefilterStatistics,
    ) -> WindowPrefilterBenchmarkPlan:
        candidates = self.rule.generate_candidates(
            ast,
            statistics,
            max_pair_candidates=self.max_pair_candidates,
            max_pair_estimated_cost=self.max_pair_estimated_cost,
        )
        if not candidates:
            raise ValueError("statistics did not produce any rewrite candidates")

        if self.include_original:
            candidates = [
                WindowPrefilterCandidate(
                    candidate_id="no_rewrite",
                    variables=(),
                    strategy="no_rewrite",
                    estimated_cost=1.0,
                ),
                *candidates,
            ]

        queries = []
        for candidate in candidates:
            if candidate.strategy == "no_rewrite":
                sample_ast = copy.deepcopy(ast)
                qualifier_safe, rebind_alias = resolve_input_rebind_alias(ast)
                if not qualifier_safe:
                    raise ValueError("sample input cannot preserve source qualifiers")
                sample_source = sample_input_table.strip()
                if rebind_alias is not None:
                    sample_source = bind_input_relation_alias(
                        sample_source, rebind_alias
                    )
                sample_ast.input_table = sample_source
                sample_sql = sample_ast.to_sql()
            else:
                sample_ast = self.rule.rewrite_candidate(
                    ast,
                    candidate,
                    statistics,
                    input_table_override=sample_input_table,
                )
                if sample_ast is None:
                    continue
                sample_sql = sample_ast.to_sql()
            queries.append(WindowPrefilterBenchmarkQuery(candidate, sample_sql))

        if not queries:
            raise ValueError("no benchmark SQL could be generated")
        return WindowPrefilterBenchmarkPlan(statistics, tuple(queries))

    def select_fastest(
        self,
        ast: RPRAbstractSyntaxTree,
        benchmark_plan: WindowPrefilterBenchmarkPlan,
        elapsed_seconds: Mapping[str, float],
    ) -> WindowPrefilterOptimizationResult:
        measured = []
        for query in benchmark_plan.queries:
            value = elapsed_seconds.get(query.candidate.candidate_id)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric < 0 or not math.isfinite(numeric):
                continue
            measured.append((numeric, query.candidate))
        if not measured:
            raise ValueError("no valid candidate execution times were supplied")

        # Prefer no rewrite on an exact tie, then use candidate_id to keep the
        # choice deterministic across runs.
        measured.sort(
            key=lambda item: (
                item[0],
                0 if item[1].strategy == "no_rewrite" else 1,
                item[1].candidate_id,
            )
        )
        winning_time, winner = measured[0]

        original_measurement = next(
            (
                value
                for value, candidate in measured
                if candidate.strategy == "no_rewrite"
            ),
            None,
        )
        if (
            winner.strategy != "no_rewrite"
            and original_measurement is not None
            and winning_time
            > original_measurement * (1.0 - self.minimum_improvement_ratio)
        ):
            winning_time = original_measurement
            winner = next(
                candidate
                for _value, candidate in measured
                if candidate.strategy == "no_rewrite"
            )

        if winner.strategy == "no_rewrite":
            final_sql = ast.to_sql()
        else:
            rewritten = self.rule.rewrite_candidate(
                ast, winner, benchmark_plan.statistics
            )
            if rewritten is None:
                raise ValueError("winning candidate could not be rendered")
            final_sql = rewritten.to_sql()
        return WindowPrefilterOptimizationResult(winner, winning_time, final_sql)
