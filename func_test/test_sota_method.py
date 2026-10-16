"""Focused integration tests for the non-LLM baseline pipeline."""

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from queryanalysis import RPRAbstractSyntaxTree
from sota_method import (
    QueryMeasurement,
    _apply_adaptive_sequence_filter,
    _apply_adaptive_window_prefilter,
    _partition_filter_row_counts_sql,
    _sample_ast,
    rewrite_sql,
)
from rewriting_rules import SequenceFilteringRule

SCHEMA = {
    "T": {
        "columns": ["PT", "TS", "TYPE"],
        "column_types": {
            "PT": "VARCHAR(8)",
            "TS": "LONG",
            "TYPE": "VARCHAR(8)",
        },
        "column_nullables": {"PT": False, "TS": False, "TYPE": False},
    }
}


def test_define_partition_pushdown_precedes_adaptive_rules():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.PT = 'X', B AS B.TYPE = 'B'
)"""

    result = rewrite_sql(sql, SCHEMA, sample_input_table="T_SAMPLE")

    assert result.applied_rules == ["define_partition_pushdown"]
    assert "FROM (SELECT * FROM T WHERE PT = 'X')" in result.sql
    assert "T_SAMPLE" not in result.sql
    assert [metric.rule for metric in result.rule_metrics] == [
        "alternation_variable_merge",
        "incremental_condition_pushdown",
        "define_partition_pushdown",
        "sequence_filtering",
        "window_condition_pre_filter",
        "row_filtering",
    ]


SEQUENCE_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""


class SequenceReductionExecutor:
    def __init__(self, rows_after):
        self.rows_after = rows_after
        self.row_count_queries = []

    def fetch_one(self, sql):
        assert "MATCH_RECOGNIZE" not in sql
        if "RPR_ROWS_BEFORE" in sql:
            self.row_count_queries.append(sql)
            return {
                "RPR_ROWS_BEFORE": 1000,
                "RPR_ROWS_AFTER": self.rows_after,
            }
        assert "RPR_PARTITIONS" in sql
        return {"RPR_UNIT_COUNT": 5 if "WHERE PT IN" in sql else 10}

    def measure_query(self, sql, runs):
        raise AssertionError("sequence sampling must not execute MATCH_RECOGNIZE")


def _run_sequence_filter(rows_after, threshold=0.2):
    production = RPRAbstractSyntaxTree()
    production.parse_sql(SEQUENCE_SQL, SCHEMA)
    sampled = _sample_ast(production, "T_SAMPLE")
    executor = SequenceReductionExecutor(rows_after)
    _sampled, metric = _apply_adaptive_sequence_filter(
        production,
        sampled,
        executor,
        minimum_row_reduction=threshold,
    )
    return production, sampled, executor, metric


def test_sequence_filter_requires_partition_and_row_reduction():
    production, sampled, executor, metric = _run_sequence_filter(600)

    assert metric.applied
    assert metric.decision == "applied_sample_row_reduction"
    assert metric.details["sample_partitions_before"] == 10
    assert metric.details["sample_partitions_after"] == 5
    assert metric.details["sample_rows_before"] == 1000
    assert metric.details["sample_rows_after"] == 600
    assert metric.details["sample_row_reduction_ratio"] == 0.4
    assert len(executor.row_count_queries) == 1
    assert "T_SAMPLE" not in production.to_sql()

    candidate = _sample_ast(sampled, "T_SAMPLE")
    assert SequenceFilteringRule().apply(candidate)
    aggregate_sql = _partition_filter_row_counts_sql(sampled, candidate)
    assert "MATCH_RECOGNIZE" not in aggregate_sql
    assert "RPR_ROWS_BEFORE" in aggregate_sql
    assert "RPR_ROWS_AFTER" in aggregate_sql


def test_sequence_filter_threshold_is_strict_and_configurable():
    _production, _sampled, _executor, rejected = _run_sequence_filter(800)
    assert not rejected.applied
    assert rejected.decision == "kept_original_insufficient_row_reduction"
    assert rejected.details["sample_row_reduction_ratio"] == 0.2

    _production, _sampled, _executor, accepted = _run_sequence_filter(800, 0.19)
    assert accepted.applied
    assert accepted.decision == "applied_sample_row_reduction"


def test_sequence_filter_rejects_nullable_partition_metadata_without_sampling():
    nullable_schema = {
        "T": {
            **SCHEMA["T"],
            "column_nullables": {
                **SCHEMA["T"]["column_nullables"],
                "PT": True,
            },
        }
    }
    production = RPRAbstractSyntaxTree()
    production.parse_sql(SEQUENCE_SQL, nullable_schema)
    sampled = _sample_ast(production, "T_SAMPLE")

    class NoSamplingExecutor:
        def fetch_one(self, sql):
            raise AssertionError("nullable metadata must reject before sampling")

        def measure_query(self, sql, runs):
            raise AssertionError("nullable metadata must reject before sampling")

    _sampled, metric = _apply_adaptive_sequence_filter(
        production,
        sampled,
        NoSamplingExecutor(),
        minimum_row_reduction=0.2,
    )

    assert not metric.applied
    assert metric.decision == "kept_original_nullable_partition_key"
    assert metric.details["nullable_partition_columns"] == ["PT"]


PAIR_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE
        A AS A.TYPE = 'A',
        B AS B.TYPE = 'B',
        C AS C.TS - A.TS < 100 AND C.TYPE = 'C'
)"""


class PairBudgetExecutor:
    timeout_seconds = 5.0

    def __init__(self):
        self.normal_measure_calls = []
        self.budget_measure_calls = []
        self.budget_fetch_calls = []

    def fetch_one(self, sql):
        if "RPR_UNIT_COUNT" in sql:
            return {"RPR_UNIT_COUNT": 50 if "WITH BUCKETED_ROWS" in sql else 100}
        return {"A": 10, "B": 100, "C": 20, "N": 1000, "WLEN": 10000, "M": 5}

    def fetch_one_with_timeout(self, sql, timeout_seconds):
        self.budget_fetch_calls.append((sql, timeout_seconds))
        return self.fetch_one(sql)

    def measure_query(self, sql, runs):
        self.normal_measure_calls.append(sql)
        return 1.0 if "WITH BUCKETED_ROWS" in sql else 2.0

    def measure_query_with_timeout(self, sql, runs, timeout_seconds):
        measurement = 0.4 if not self.budget_measure_calls else 2.0
        self.budget_measure_calls.append((sql, timeout_seconds))
        return measurement


def test_pair_candidates_use_single_runtime_budget_and_are_not_repeated():
    production = RPRAbstractSyntaxTree()
    production.parse_sql(PAIR_SQL, SCHEMA)
    sampled = _sample_ast(production, "T_SAMPLE")
    executor = PairBudgetExecutor()

    _sampled, metric = _apply_adaptive_window_prefilter(
        production,
        sampled,
        executor,
        benchmark_runs=3,
        max_pair_candidates=None,
    )

    assert metric.applied
    assert metric.details["single_baseline_seconds"] == 1.0
    assert metric.details["base_time_seconds"] == 0.4
    assert metric.details["pair_candidate_timeout_seconds"] == 0.8
    assert metric.details["selected_candidate"].startswith("pair_")
    assert len(executor.normal_measure_calls) == 2  # single plus original
    assert executor.budget_measure_calls
    assert executor.budget_fetch_calls
    assert executor.budget_fetch_calls[0][1] == 2.0
    assert executor.budget_measure_calls[0][1] == 2.0
    assert all(timeout == 0.8 for _sql, timeout in executor.budget_fetch_calls[1:])
    assert all(timeout == 0.8 for _sql, timeout in executor.budget_measure_calls[1:])
    assert len(executor.budget_measure_calls) == len(
        {sql for sql, _timeout in executor.budget_measure_calls}
    )
    single_id = next(iter(metric.details["single_baseline_measurements"]))
    assert metric.details["adaptive_candidates_pruned"][single_id]["phase"] == (
        "rebaseline"
    )
    assert single_id not in metric.details["candidate_measurements"]
    pruned = metric.details["pair_candidates_pruned"]
    assert all(item["phase"] == "benchmark" for item in pruned.values())
    assert all(
        candidate_id not in metric.details["candidate_measurements"]
        for candidate_id in pruned
    )


class TimedOutSingleExecutor(PairBudgetExecutor):
    def measure_query(self, sql, runs):
        self.normal_measure_calls.append(sql)
        if "WITH BUCKETED_ROWS" in sql:
            return QueryMeasurement(False, 5.0, 0)
        return QueryMeasurement(True, 2.0, 10)

    def measure_query_with_timeout(self, sql, runs, timeout_seconds):
        call_index = len(self.budget_measure_calls)
        if call_index == 0:
            measurement = QueryMeasurement(False, timeout_seconds, 25)
        elif call_index == 1:
            measurement = QueryMeasurement(True, 0.4, 100)
        else:
            measurement = QueryMeasurement(False, timeout_seconds, 10)
        self.budget_measure_calls.append((sql, timeout_seconds))
        return measurement


def test_pair_candidate_establishes_baseline_when_single_times_out():
    production = RPRAbstractSyntaxTree()
    production.parse_sql(PAIR_SQL, SCHEMA)
    sampled = _sample_ast(production, "T_SAMPLE")
    executor = TimedOutSingleExecutor()

    _sampled, metric = _apply_adaptive_window_prefilter(
        production,
        sampled,
        executor,
        benchmark_runs=3,
        max_pair_candidates=None,
    )

    assert metric.applied
    assert "single_baseline_seconds" not in metric.details
    assert metric.details["base_time_seconds"] == 0.4
    assert metric.details["baseline_candidate"].startswith("pair_")
    assert executor.budget_fetch_calls
    assert executor.budget_measure_calls
    assert [timeout for _sql, timeout in executor.budget_fetch_calls[:2]] == [
        5.0,
        5.0,
    ]
    assert [timeout for _sql, timeout in executor.budget_measure_calls[:2]] == [
        5.0,
        5.0,
    ]
    assert all(timeout == 0.8 for _sql, timeout in executor.budget_fetch_calls[2:])
    assert all(timeout == 0.8 for _sql, timeout in executor.budget_measure_calls[2:])
    first_pair_id = next(iter(metric.details["pair_candidate_measurements"]))
    assert metric.details["adaptive_candidates_pruned"][first_pair_id]["phase"] == (
        "rebaseline"
    )
