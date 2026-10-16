"""Integration tests for the complete rule-based pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401  Ensures project modules are importable.

from queryanalysis import RPRAbstractSyntaxTree
import our_method
from our_method import (
    QueryMeasurement,
    RuleSpec,
    _apply_window_condition_prefilter,
    _partition_filter_row_counts_sql,
    _partition_count_sql,
    _postprocess_candidate,
    _register_generated_input_schema,
    _sample_ast,
    _sql_for_system,
    rewrite_sql,
)
from rewriting_rules import (
    MustAppearPartitionFilterPushdownRule,
    RegexOptimizationPlusRule,
)

SCHEMA = {
    "T": {
        "columns": ["PT", "TS", "ID", "TYPE", "VAL", "OPEN", "VOL"],
        "column_types": {
            "PT": "LONG",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(20)",
            "VAL": "DOUBLE",
            "OPEN": "DOUBLE",
            "VOL": "DOUBLE",
        },
        "column_nullables": {
            "PT": False,
            "TS": False,
            "ID": False,
            "TYPE": False,
            "VAL": False,
            "OPEN": False,
            "VOL": False,
        },
    }
}


def _metric(result, rule_name):
    return next(metric for metric in result.rule_metrics if metric.rule == rule_name)


def test_system_controls_limit_clause_syntax():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)
LIMIT 7"""

    oracle = rewrite_sql(sql, SCHEMA)
    trino = rewrite_sql(sql, SCHEMA, system="trino")

    assert "FETCH FIRST 7 ROWS ONLY" in oracle.sql
    assert "LIMIT 7" not in oracle.sql
    assert "LIMIT 7" in trino.sql
    assert "FETCH FIRST 7 ROWS ONLY" not in trino.sql

    lexical_edge_cases = "SELECT 'LIMIT 1' AS TXT /* LIMIT 2 */ LIMIT 3"
    converted = _sql_for_system(lexical_edge_cases, "oracle")
    assert "'LIMIT 1'" in converted
    assert "/* LIMIT 2 */" in converted
    assert converted.endswith("FETCH FIRST 3 ROWS ONLY")


def test_false_pushdown_short_circuits_the_pipeline():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(A B)
    DEFINE A AS 1 != 1, B AS B.TYPE = 'B'
)"""

    result = rewrite_sql(sql, SCHEMA)

    assert result.stopped_after_rule == "must_appear_var_false_pushdown"
    assert result.rule_metrics[-1].rule == "must_appear_var_false_pushdown"
    assert "must_appear_partition_filter_pushdown" not in {
        metric.rule for metric in result.rule_metrics
    }
    assert "WHERE 1 != 1" in result.sql


def test_propagated_false_define_short_circuits_the_pipeline():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(A+ B+ C+ D)
    DEFINE A AS A.VAL BETWEEN 100 AND 500,
        B AS B.VAL <= 1500,
        C AS C.VAL / 2 < LAST(B.VAL)
             AND C.VAL > FIRST(B.VAL) * 3
             AND C.VAL >= 4000,
        D AS D.VAL > LAST(C.VAL)
)"""

    result = rewrite_sql(sql, SCHEMA)

    propagation = _metric(result, "define_independent_constraint_propagation")
    assert propagation.applied
    assert result.stopped_after_rule == "must_appear_var_false_pushdown"
    assert result.rule_metrics[-1].rule == "must_appear_var_false_pushdown"
    assert "WHERE 1 != 1" in result.sql
    assert "C AS 1 != 1" in result.sql


def test_relational_false_define_short_circuits_the_pipeline():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(S1 G* B1 (B1 | G)* B2)
    DEFINE S1 AS S1.VOL >= 500 AND S1.VAL >= 500,
        G AS G.TS - S1.TS <= 600000000,
        B1 AS B1.VOL < 100 AND B1.VAL > S1.VAL,
        B2 AS B2.VOL >= 100
              AND B2.VAL < S1.VAL
              AND B2.VAL > B1.VAL
              AND B2.TS - S1.TS <= 600000000
)"""

    result = rewrite_sql(sql, SCHEMA)

    propagation = _metric(result, "define_independent_constraint_propagation")
    assert propagation.applied
    assert result.stopped_after_rule == "must_appear_var_false_pushdown"
    assert result.rule_metrics[-1].rule == "must_appear_var_false_pushdown"
    assert "WHERE 1 != 1" in result.sql
    assert "B2 AS 1 != 1" in result.sql


def test_special_row_filter_suppresses_window_function_row_filter():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A {-N1*-} B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""

    result = rewrite_sql(sql, SCHEMA)

    special = _metric(result, "row_filter_4_special_pattern")
    window_filter = _metric(result, "row_filtering_via_window_func")
    assert special.applied
    assert not window_filter.applied
    assert (
        window_filter.decision == "skipped_conflict_with_row_filter_4_special_pattern"
    )


def test_anchor_partition_size_filter_suppresses_must_appear_filter():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(^A B$)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""

    class NoSamplingExecutor:
        def fetch_one(self, sql):
            raise AssertionError("suppressed rule must not execute sampling SQL")

        def measure_query(self, sql, runs):
            raise AssertionError("suppressed rule must not benchmark candidates")

    result = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=NoSamplingExecutor(),
        sample_must_appear_partition_filter=True,
    )

    anchor_metrics = [
        metric
        for metric in result.rule_metrics
        if metric.rule == "anchor_partition_size_filter"
    ]
    must_appear = _metric(result, "must_appear_partition_filter_pushdown")
    assert any(metric.applied for metric in anchor_metrics)
    assert not must_appear.applied
    assert must_appear.decision == "skipped_conflict_with_anchor_partition_size_filter"
    assert "must_appear_partition_filter_pushdown" not in result.applied_rules


def test_input_pruning_rules_are_disabled_for_unmatched_rows():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH WITH UNMATCHED ROWS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""

    result = rewrite_sql(sql, SCHEMA)

    for rule_name in (
        "must_appear_partition_filter_pushdown",
        "row_filter_4_special_pattern",
        "row_filtering_via_window_func",
        "window_condition_pre_filter",
    ):
        metric = _metric(result, rule_name)
        assert not metric.applied
        assert metric.decision == "skipped_unsafe_input_pruning"
        assert "rows_per_match_preserves_unmatched_rows" in metric.details["reasons"]


def test_input_pruning_rules_are_disabled_for_empty_patterns():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A*)
    DEFINE A AS A.TYPE = 'A'
)"""

    result = rewrite_sql(sql, SCHEMA)

    for rule_name in (
        "must_appear_partition_filter_pushdown",
        "row_filter_4_special_pattern",
        "row_filtering_via_window_func",
        "window_condition_pre_filter",
    ):
        metric = _metric(result, rule_name)
        assert not metric.applied
        assert metric.decision == "skipped_unsafe_input_pruning"
        assert "pattern_can_match_zero_rows" in metric.details["reasons"]


class PartitionCountExecutor:
    def __init__(self, rows_after=600):
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
        return {
            "RPR_PARTITION_COUNT": 5 if "MAX(CASE" in sql else 10,
        }

    def measure_query(self, sql, runs):
        raise AssertionError("must-appear sampling must not execute full candidates")


def test_must_appear_partition_filter_requires_partition_and_row_reduction():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""

    direct = rewrite_sql(sql, SCHEMA)
    direct_metric = _metric(direct, "must_appear_partition_filter_pushdown")
    assert direct_metric.applied
    assert direct_metric.decision == "applied_without_sampling"

    executor = PartitionCountExecutor()
    sampled = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=executor,
        sample_must_appear_partition_filter=True,
    )
    sampled_metric = _metric(sampled, "must_appear_partition_filter_pushdown")
    assert sampled_metric.applied
    assert sampled_metric.decision == "applied_sample_row_reduction"
    assert sampled_metric.details["sample_partitions_before"] == 10
    assert sampled_metric.details["sample_partitions_after"] == 5
    assert sampled_metric.details["sample_partitions_removed"] == 5
    assert sampled_metric.details["sample_rows_before"] == 1000
    assert sampled_metric.details["sample_rows_after"] == 600
    assert sampled_metric.details["sample_rows_removed"] == 400
    assert sampled_metric.details["sample_row_reduction_ratio"] == 0.4
    assert sampled_metric.details["minimum_row_reduction"] == 0.2
    assert len(executor.row_count_queries) == 1
    assert "candidate_measurements" not in sampled_metric.details
    assert "T_SAMPLE" not in sampled.sql


class UnchangedPartitionCountExecutor(PartitionCountExecutor):
    def fetch_one(self, sql):
        assert "MATCH_RECOGNIZE" not in sql
        assert "RPR_ROWS_BEFORE" not in sql
        return {"RPR_PARTITION_COUNT": 10}


def test_must_appear_partition_filter_is_rejected_without_partition_reduction():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    result = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=UnchangedPartitionCountExecutor(),
        sample_must_appear_partition_filter=True,
    )
    metric = _metric(result, "must_appear_partition_filter_pushdown")

    assert not metric.applied
    assert metric.decision == "kept_original_no_partitions_removed"
    assert metric.details["sample_partitions_removed"] == 0


def test_must_appear_partition_filter_rejects_small_partitions_by_row_ratio():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    rejected = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=PartitionCountExecutor(rows_after=800),
        sample_must_appear_partition_filter=True,
    )
    metric = _metric(rejected, "must_appear_partition_filter_pushdown")
    assert not metric.applied
    assert metric.decision == "kept_original_insufficient_row_reduction"
    assert metric.details["sample_row_reduction_ratio"] == 0.2

    accepted = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=PartitionCountExecutor(rows_after=800),
        sample_must_appear_partition_filter=True,
        partition_pruning_min_row_reduction=0.19,
    )
    metric = _metric(accepted, "must_appear_partition_filter_pushdown")
    assert metric.applied
    assert metric.decision == "applied_sample_row_reduction"


def test_must_appear_partition_filter_rejects_nullable_partition_metadata():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    nullable_schema = {
        "T": {
            **SCHEMA["T"],
            "column_nullables": {
                **SCHEMA["T"]["column_nullables"],
                "PT": True,
            },
        }
    }

    class NoSamplingExecutor:
        def fetch_one(self, sql):
            raise AssertionError("nullable metadata must reject before sampling")

        def measure_query(self, sql, runs):
            raise AssertionError("nullable metadata must reject before sampling")

    result = rewrite_sql(
        sql,
        nullable_schema,
        sample_input_table="T_SAMPLE",
        executor=NoSamplingExecutor(),
        sample_must_appear_partition_filter=True,
    )
    metric = _metric(result, "must_appear_partition_filter_pushdown")

    assert not metric.applied
    assert metric.decision == "kept_original_nullable_partition_key"
    assert metric.details["nullable_partition_columns"] == ["PT"]

    direct = rewrite_sql(sql, nullable_schema)
    direct_metric = _metric(direct, "must_appear_partition_filter_pushdown")
    assert not direct_metric.applied
    assert direct_metric.decision == "kept_original_nullable_partition_key"


def test_must_appear_sampling_sql_never_contains_match_recognize():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, SCHEMA)
    sampled = _sample_ast(ast, "T_SAMPLE")

    before_sql = _partition_count_sql(sampled, "oracle")
    assert "MATCH_RECOGNIZE" not in before_sql
    assert "SELECT DISTINCT PT FROM T_SAMPLE" in before_sql

    rule = MustAppearPartitionFilterPushdownRule()
    assert rule.apply(sampled)
    after_sql = _partition_count_sql(sampled, "oracle")
    assert "MATCH_RECOGNIZE" not in after_sql
    assert "MAX(CASE" in after_sql
    row_count_sql = _partition_filter_row_counts_sql(ast, sampled, "oracle")
    assert "MATCH_RECOGNIZE" not in row_count_sql
    assert "RPR_ROWS_BEFORE" in row_count_sql
    assert "RPR_ROWS_AFTER" in row_count_sql


def test_partition_count_without_partition_by_counts_one_nonempty_partition():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, SCHEMA)
    sampled = _sample_ast(ast, "T_SAMPLE")

    oracle_sql = _partition_count_sql(sampled, "oracle")
    trino_sql = _partition_count_sql(sampled, "trino")
    assert "MATCH_RECOGNIZE" not in oracle_sql
    assert "FETCH FIRST 1 ROWS ONLY" in oracle_sql
    assert "LIMIT 1" not in oracle_sql
    assert "LIMIT 1" in trino_sql


def test_generated_window_input_registers_schema_for_downstream_rules():
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    PATTERN(A B)
    DEFINE A AS A.VOL < 100, B AS B.VOL >= 100
)""",
        SCHEMA,
    )
    ast.input_table = "RPR_FILTERED_ROWS"
    ast._window_prefilter_input_columns = tuple(SCHEMA["T"]["columns"])
    ast._window_prefilter_input_schema = SCHEMA["T"]

    assert ast.get_column_metadata("VOL") is None
    _register_generated_input_schema(ast)
    metadata = ast.get_column_metadata("VOL")
    assert metadata is not None
    assert metadata.data_type == "DOUBLE"


class FastWindowPrefilterExecutor:
    def fetch_one(self, sql):
        if "RPR_UNIT_COUNT" in sql:
            return {
                "RPR_UNIT_COUNT": 50 if "WITH BUCKETED_ROWS" in sql else 100,
            }
        return {"A": 10, "N": 100, "WLEN": 10000}

    def measure_query(self, sql, runs):
        assert runs == 3
        return 0.5 if "WITH BUCKETED_ROWS" in sql else 2.0


def test_window_prefilter_and_join_pruning_share_outer_ctes_and_actual_primary_key():
    schemas = {
        # Equal column names must not cause the unrelated table's PK to be used.
        "UNRELATED": {**SCHEMA["T"], "primary_key": ["VOL"]},
        "T": {**SCHEMA["T"], "primary_key": ["ID"]},
    }
    for projection in ("*", "TYPE, TS AS EVENT_TS, ID", "TS"):
        sql = f"""
SELECT {projection} FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TS - A.TS < 100
)"""
        result = rewrite_sql(
            sql,
            schemas,
            sample_input_table="T_SAMPLE",
            executor=FastWindowPrefilterExecutor(),
            sample_window_condition_prefilter=True,
        )

        assert _metric(result, "window_condition_pre_filter").applied
        assert _metric(result, "column_pruning_via_join").applied
        assert result.sql.startswith("WITH BUCKETED_ROWS AS(")
        assert result.sql.count("WITH BUCKETED_ROWS AS(") == 1
        assert "FROM (\nWITH" not in result.sql
        assert "T_SAMPLE" not in result.sql
        if projection == "TS":
            assert "LEFT JOIN" not in result.sql
            assert "\nSELECT TS\n" in result.sql
        else:
            assert "LEFT JOIN FILTERED_ROWS T_FULL" in result.sql
            assert "ON T_FULL.ID = MR.RPR_JOIN_PK_1" in result.sql
            assert "T_FULL.VOL =" not in result.sql
            if projection == "*":
                assert (
                    "SELECT MR.TS AS TS, T_FULL.PT AS PT, T_FULL.ID AS ID, "
                    "T_FULL.TYPE AS TYPE, T_FULL.VAL AS VAL, "
                    "T_FULL.OPEN AS OPEN, T_FULL.VOL AS VOL" in result.sql
                )
            else:
                assert "SELECT T_FULL.TYPE, MR.TS AS EVENT_TS, T_FULL.ID" in result.sql


def test_generated_window_input_does_not_infer_primary_key_from_derived_rows():
    schemas = {"T": {**SCHEMA["T"], "primary_key": ["ID"]}}
    source = RPRAbstractSyntaxTree()
    source.parse_sql(
        """SELECT * FROM T MATCH_RECOGNIZE(
            ORDER BY TS ALL ROWS PER MATCH PATTERN(A B)
            DEFINE A AS TYPE = 'A', B AS TYPE = 'B')""",
        schemas,
    )
    # A UNION ALL can duplicate keys even though its columns match T exactly.
    source.input_table = "(SELECT * FROM T UNION ALL SELECT * FROM T)"
    source._preserved_input_columns = tuple(SCHEMA["T"]["columns"])
    candidate = RPRAbstractSyntaxTree()
    candidate.table_schemas = schemas
    candidate.input_table = "FILTERED_ROWS"
    candidate._window_prefilter_input_columns = source._preserved_input_columns

    _postprocess_candidate(source, candidate)

    metadata = candidate.table_schemas["FILTERED_ROWS"]
    assert not metadata.get("primary_key")
    assert metadata["column_types"]["VOL"] == "DOUBLE"


def test_generated_window_input_without_provenance_does_not_borrow_schema():
    ast = RPRAbstractSyntaxTree()
    ast.table_schemas = SCHEMA
    ast.input_table = "FILTERED_ROWS"
    ast._window_prefilter_input_columns = tuple(SCHEMA["T"]["columns"])

    _register_generated_input_schema(ast)

    assert "FILTERED_ROWS" not in ast.table_schemas


def test_window_prefilter_is_disabled_by_default_and_sample_selected_when_enabled():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TS - A.TS < 100
)"""

    disabled = rewrite_sql(sql, SCHEMA)
    disabled_metric = _metric(disabled, "window_condition_pre_filter")
    assert not disabled_metric.applied
    assert disabled_metric.decision == "disabled_without_sampling"

    enabled = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=FastWindowPrefilterExecutor(),
        sample_window_condition_prefilter=True,
    )
    enabled_metric = _metric(enabled, "window_condition_pre_filter")
    assert enabled_metric.applied
    assert enabled_metric.decision == "applied_fastest_completed_sample_candidate"
    assert enabled_metric.details["selected_candidate"] == "single:A"
    assert "WITH BUCKETED_ROWS" in enabled.sql
    assert "T_SAMPLE" not in enabled.sql


class CompletedOriginalWindowExecutor(FastWindowPrefilterExecutor):
    def measure_query(self, sql, runs):
        assert runs == 3
        if "WITH BUCKETED_ROWS" in sql:
            return QueryMeasurement(False, 5.0, 1000)
        return QueryMeasurement(True, 4.0, 50)


def test_window_prefilter_prefers_any_completed_candidate():
    sql = """
SELECT * FROM T MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TS - A.TS < 100
)"""
    result = rewrite_sql(
        sql,
        SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=CompletedOriginalWindowExecutor(),
        sample_window_condition_prefilter=True,
    )
    metric = _metric(result, "window_condition_pre_filter")

    assert not metric.applied
    assert metric.details["selected_candidate"] == "no_rewrite"
    assert metric.details["selection_reason"] == "fastest_completed"


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


def test_window_pairs_use_single_runtime_budget_and_are_not_repeated():
    sql = """
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
    production = RPRAbstractSyntaxTree()
    production.parse_sql(sql, SCHEMA)
    sampled = _sample_ast(production, "T_SAMPLE")
    executor = PairBudgetExecutor()

    _production, _sampled, metric = _apply_window_condition_prefilter(
        production,
        sampled,
        executor,
        use_sampling=True,
        benchmark_runs=3,
        max_pair_candidates=None,
        system="oracle",
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


class TimedOutSinglePairBudgetExecutor(PairBudgetExecutor):
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


def test_window_pair_establishes_baseline_when_single_times_out():
    sql = """
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
    production = RPRAbstractSyntaxTree()
    production.parse_sql(sql, SCHEMA)
    sampled = _sample_ast(production, "T_SAMPLE")
    executor = TimedOutSinglePairBudgetExecutor()

    _production, _sampled, metric = _apply_window_condition_prefilter(
        production,
        sampled,
        executor,
        use_sampling=True,
        benchmark_runs=3,
        max_pair_candidates=None,
        system="oracle",
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


def test_sample_ast_preserves_derived_table_filter():
    sql = """
SELECT * FROM (SELECT * FROM T WHERE TYPE = 'A') MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)"""
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, SCHEMA)

    sampled = _sample_ast(ast, "T_SAMPLE")

    assert sampled.input_table == "(SELECT * FROM T_SAMPLE WHERE TYPE = 'A')"
    assert sampled.table_schemas["T_SAMPLE"] == SCHEMA["T"]


def test_oversized_quantifier_is_repaired_before_crimes_68_regex_optimization():
    workload_path = Path(__file__).resolve().parents[1] / "data" / "crimes_sql.json"
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    query = workload["sqls"][67]

    result = rewrite_sql(query["sql"], workload["table_schemas"])

    # The early regex pass simplifies the compound repetition; COUNT pushdown
    # then introduces {1,499}, which must be repaired before the later pass.
    regex_metrics = [
        metric
        for metric in result.rule_metrics
        if metric.rule == "regex_optimization_plus"
    ]
    assert len(regex_metrics) == 2
    assert regex_metrics[0].applied
    assert "quantifier_pre_regex_repairs" not in regex_metrics[0].details
    assert _metric(result, "count_range_pushdown").applied
    regex_metric = regex_metrics[1]
    assert regex_metric.details["quantifier_bound_limit"] == 180
    assert regex_metric.details["quantifier_pre_regex_repairs"] == [
        {
            "path": "pattern.patterns[0].pattern",
            "variable": "COMMON_EVENT",
            "action": "split_finite_range",
            "rewritten_minimum": 1,
            "rewritten_maximum": 499,
            "chunks": [[1, 180], [0, 180], [0, 139]],
        }
    ]
    assert (
        "PATTERN(^COMMON_EVENT{1,180} COMMON_EVENT{0,180} "
        "COMMON_EVENT{0,139} HOMICIDE_EVENT$)" in result.sql
    )
    assert "{1,499}" not in result.sql


def test_earlier_quantifier_repairs_are_rendered_when_later_regex_passes_are_noops():
    sql = """SELECT * FROM T MATCH_RECOGNIZE(
        ORDER BY TS ALL ROWS PER MATCH PATTERN(A{200} B)
        DEFINE A AS 1 = 1, B AS 1 = 1)"""
    for pass_count in (2, 3):
        for sample_input in (None, "T_SAMPLE"):
            with patch.object(
                our_method,
                "RULE_PIPELINE",
                tuple(RuleSpec(RegexOptimizationPlusRule) for _ in range(pass_count)),
            ):
                result = rewrite_sql(sql, SCHEMA, sample_input_table=sample_input)

            # No regular rule changes the query: the first bound repair alone
            # must cause the output AST to be rendered instead of the input SQL.
            assert result.applied_rules == []
            assert result.rule_metrics[0].details["quantifier_bound_limit"] == 180
            assert result.rule_metrics[0].details["quantifier_pre_regex_repairs"] == [
                {
                    "path": "pattern.patterns[0]",
                    "variable": "A",
                    "action": "split_finite_range",
                    "rewritten_minimum": 200,
                    "rewritten_maximum": 200,
                    "chunks": [[180, 180], [20, 20]],
                }
            ]
            assert all(
                "quantifier_pre_regex_repairs" not in metric.details
                for metric in result.rule_metrics[1:]
            )
            assert "PATTERN(A{180} A{20} B)" in result.sql
            assert "A{200}" not in result.sql
            assert "T_SAMPLE" not in result.sql


COLUMN_PRUNING_VIA_JOIN_SCHEMA = {
    "T": {
        **SCHEMA["T"],
        "primary_key": ["ID"],
    }
}

COLUMN_PRUNING_VIA_JOIN_SQL = """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS, ID
    ALL ROWS PER MATCH
    {skip_clause}
    PATTERN(COMMON_EVENT+ TARGET_EVENT)
    DEFINE
        COMMON_EVENT AS COMMON_EVENT.TYPE != 'TARGET',
        TARGET_EVENT AS TARGET_EVENT.TYPE = 'TARGET'
)"""


class ColumnPruningViaJoinCountExecutor:
    def __init__(self, output_rows):
        self.output_rows = output_rows
        self.queries = []

    def fetch_one(self, sql):
        self.queries.append(sql)
        assert "T_SAMPLE" in sql
        if "RPR_UNIT_COUNT" in sql:
            assert "MATCH_RECOGNIZE" not in sql
            return {"RPR_UNIT_COUNT": 100}
        assert "RPR_OUTPUT_COUNT" in sql
        assert "MATCH_RECOGNIZE" in sql
        assert "LEFT JOIN" not in sql
        return {"RPR_OUTPUT_COUNT": self.output_rows}

    def measure_query(self, sql, runs):
        raise AssertionError("the cardinality guard must only execute counts")


def test_column_pruning_via_join_applies_past_last_without_sampling():
    sql = COLUMN_PRUNING_VIA_JOIN_SQL.format(skip_clause="")
    result = rewrite_sql(sql, COLUMN_PRUNING_VIA_JOIN_SCHEMA)
    metric = _metric(result, "column_pruning_via_join")

    assert metric.applied
    assert metric.decision == "applied_past_last_row_without_sampling"
    assert metric.details["after_match_skip_to"] == "PAST LAST ROW"


def test_column_pruning_via_join_is_disabled_for_trino():
    sql = COLUMN_PRUNING_VIA_JOIN_SQL.format(skip_clause="")
    result = rewrite_sql(sql, COLUMN_PRUNING_VIA_JOIN_SCHEMA, system="trino")
    metric = _metric(result, "column_pruning_via_join")

    assert not metric.applicable
    assert not metric.applied
    assert metric.decision == "skipped_unsupported_system"
    assert metric.details == {"system": "trino"}
    assert "column_pruning_via_join" not in result.applied_rules


def test_column_pruning_via_join_keeps_overlapping_query_without_sampler():
    sql = COLUMN_PRUNING_VIA_JOIN_SQL.format(skip_clause="AFTER MATCH SKIP TO NEXT ROW")
    result = rewrite_sql(sql, COLUMN_PRUNING_VIA_JOIN_SCHEMA)
    metric = _metric(result, "column_pruning_via_join")

    assert metric.applicable
    assert not metric.applied
    assert metric.decision == "kept_original_no_sampler"


def test_column_pruning_via_join_accepts_output_ratio_at_threshold():
    sql = COLUMN_PRUNING_VIA_JOIN_SQL.format(skip_clause="AFTER MATCH SKIP TO NEXT ROW")
    executor = ColumnPruningViaJoinCountExecutor(output_rows=150)
    result = rewrite_sql(
        sql,
        COLUMN_PRUNING_VIA_JOIN_SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=executor,
    )
    metric = _metric(result, "column_pruning_via_join")

    assert metric.applied
    assert metric.decision == "applied_sample_output_ratio"
    assert metric.details["sample_input_rows"] == 100
    assert metric.details["sample_output_rows"] == 150
    assert metric.details["sample_output_ratio"] == 1.5
    assert len(executor.queries) == 2


def test_column_pruning_via_join_rejects_excessive_output_ratio():
    sql = COLUMN_PRUNING_VIA_JOIN_SQL.format(skip_clause="AFTER MATCH SKIP TO NEXT ROW")
    executor = ColumnPruningViaJoinCountExecutor(output_rows=151)
    result = rewrite_sql(
        sql,
        COLUMN_PRUNING_VIA_JOIN_SCHEMA,
        sample_input_table="T_SAMPLE",
        executor=executor,
    )
    metric = _metric(result, "column_pruning_via_join")

    assert not metric.applied
    assert metric.decision == "kept_original_excessive_sample_output_ratio"
    assert metric.details["sample_output_ratio"] == 1.51
