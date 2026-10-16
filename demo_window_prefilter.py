"""Simulate the adaptive window-condition prefilter workflow without a DB."""

from queryanalysis import RPRAbstractSyntaxTree
from rewriting_rules import WindowPrefilterOptimizer

SQL = """
SELECT * FROM TA MATCH_RECOGNIZE(
    PARTITION BY P1
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE
        A AS OPEN < 100,
        B AS OPEN > 50,
        C AS TS - A.TS < 100 AND OPEN >= 100 AND VOL = A.VOL
)"""

SCHEMAS = {
    "TA": {
        "columns": ["P1", "TS", "OPEN", "VOL"],
        "column_types": {
            "P1": "VARCHAR(32)",
            "TS": "LONG",
            "OPEN": "DOUBLE",
            "VOL": "DOUBLE",
        },
        "column_nullables": {"P1": True, "TS": False},
    }
}


def main():
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(SQL, SCHEMAS)

    optimizer = WindowPrefilterOptimizer(
        include_original=True,
        max_pair_candidates=8,
        minimum_improvement_ratio=0.05,
    )
    sample_table = "TA_SAMPLE"

    # Phase 1: the caller would execute this SQL and fetch exactly one row.
    statistics_request = optimizer.build_statistics_request(ast, sample_table)
    print("=== statistics SQL ===")
    print(statistics_request.sql)

    # Simulated database result for the statistics SQL.
    statistics_row = {
        "A": 120,
        "B": 900,
        "C": 180,
        statistics_request.aliases["N"]: 10000,
        statistics_request.aliases["WLEN"]: 50000,
        statistics_request.aliases["M"]: 20,
    }

    # Phase 2: generate every SQL that must be timed on the same sample table.
    benchmark_plan = optimizer.build_benchmark_plan_from_row(
        ast,
        sample_table,
        statistics_row,
    )
    print("\n=== sample candidates ===")
    for query in benchmark_plan.queries:
        print(
            query.candidate.candidate_id,
            f"estimated_cost={query.candidate.estimated_cost:.6f}",
        )

    # Simulated median execution times returned by the sampling executor.
    elapsed_seconds = {
        "no_rewrite": 2.40,
        "single:A": 1.35,
        "pair_intersection:A+B": 1.10,
        "pair_intersection:A+C": 0.82,
        "pair_intersection:B+C": 1.18,
        "pair_dependency_join:A+C": 0.56,
    }

    # Phase 3: select the fastest measurement and render it against TA, not
    # TA_SAMPLE. Missing/failed measurements are ignored.
    result = optimizer.select_fastest(ast, benchmark_plan, elapsed_seconds)
    print("\n=== selected candidate ===")
    print(result.candidate.candidate_id, result.elapsed_seconds)
    print("\n=== final SQL ===")
    print(result.sql)


if __name__ == "__main__":
    main()
