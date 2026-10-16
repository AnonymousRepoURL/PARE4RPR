if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *


class AnchorPartitionSizeFilterRule(RewriteRule):
    """
    Rule description:
        When a MATCH_RECOGNIZE pattern has BOTH a start anchor (^) AND an end anchor ($),
        the entire partition must be consumed by the pattern match, which means the partition
        row count must fall within the pattern's [min_rows, max_rows] range.  Pre-filter
        partitions (or the whole table when there is no PARTITION BY) using an inline aggregate
        subquery that applies a HAVING COUNT(*) constraint.

    Applicability conditions:
        1. pattern is not None.
        2. Every top-level alternation branch must carry BOTH ^ and $ anchors.
           If any branch lacks one of the anchors, the rule does not apply
           (e.g., PATTERN(^A B C$ | C D) — second branch has no anchors).
        3. Every fully-anchored branch must have a finite maximum row count.
           Patterns like ^A*$ or ^A+$ are excluded.

    Non-applicable patterns:
        ^A*$              — unbounded max rows
        ^A+$              — unbounded max rows
        ^A B C$ | C D     — second branch has no anchors

    Generated SQL — with a non-nullable PARTITION BY column:
        SELECT *
        FROM (SELECT * FROM source_table WHERE TICKER IN (
            SELECT TICKER
            FROM source_table
            GROUP BY TICKER
            HAVING <having_clause>
        ))
        MATCH_RECOGNIZE(PARTITION BY TICKER ...)

    Generated SQL — without PARTITION BY:
        SELECT *
        FROM (SELECT * FROM source_table WHERE EXISTS (
            SELECT 1 FROM (
                SELECT COUNT(*) AS ROW_COUNT
                FROM source_table
                HAVING <having_clause>
            ) FILTER_SIZE
        ))
        MATCH_RECOGNIZE(...)

    Nullable partition columns use a correlated EXISTS predicate, e.g.,
    WHERE EXISTS (
        SELECT 1
        FROM (<partition-size query>) fp
        WHERE (src.TICKER = fp.TICKER OR
               (src.TICKER IS NULL AND fp.TICKER IS NULL))
    )

    PREV/NEXT in MEASURES needs no boundary extension here: every accepted
    fully anchored match consumes its complete partition, and rejected
    partitions produce no match output at all.
    """

    @property
    def name(self) -> str:
        return "anchor_partition_size_filter"

    # ------------------------------------------------------------------
    # Anchor detection helpers
    # ------------------------------------------------------------------

    def _is_start_anchored(self, pattern) -> bool:
        """Return True if pattern carries a ^ anchor at the structural start."""
        return _is_start_anchored(pattern)

    def _is_end_anchored(self, pattern) -> bool:
        """Return True if pattern carries a $ anchor at the structural end."""
        return _is_end_anchored(pattern)

    def _is_branch_fully_anchored(self, pattern) -> bool:
        """Return True if pattern has both ^ and $ anchors."""
        return self._is_start_anchored(pattern) and self._is_end_anchored(pattern)

    # ------------------------------------------------------------------
    # Range computation
    # ------------------------------------------------------------------

    def _get_valid_ranges(self, pattern) -> Optional[List[Tuple[int, int]]]:
        """
        Return a list of (min_rows, max_rows) intervals for HAVING filtering,
        or None if the rule cannot be applied to this pattern.

        Returns None when:
          - Any top-level alternation branch lacks both anchors.
          - Any fully-anchored branch has an unbounded max row count (None).
        """
        if isinstance(pattern, AlternationPattern):
            ranges = []
            for branch in pattern.patterns:
                if not self._is_branch_fully_anchored(branch):
                    # At least one branch lacks anchors — cannot pre-filter.
                    return None
                min_rows = branch.get_min_row_num()
                max_rows = branch.get_max_row_num()
                if max_rows is None:
                    # Unbounded branch (^A*$, ^A+$) — cannot pre-filter.
                    return None
                ranges.append((min_rows, max_rows))
            return self._merge_ranges(ranges)
        else:
            if not self._is_branch_fully_anchored(pattern):
                return None
            min_rows = pattern.get_min_row_num()
            max_rows = pattern.get_max_row_num()
            if max_rows is None:
                return None
            return [(min_rows, max_rows)]

    # ------------------------------------------------------------------
    # Range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent (lo, hi) intervals.

        Sorted by lo; two intervals are merged when the next lo <= current hi.
        Example: [(4,7),(5,10)] -> [(4,10)]; [(4,7),(10,15)] -> [(4,7),(10,15)]
        """
        if not ranges:
            return ranges
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        merged = [sorted_ranges[0]]
        for lo, hi in sorted_ranges[1:]:
            if lo <= merged[-1][1]:  # overlapping or adjacent
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    # ------------------------------------------------------------------
    # SQL generation helpers
    # ------------------------------------------------------------------

    def _build_having_clause(self, ranges: List[Tuple[int, int]]) -> str:
        """Build the HAVING expression from a list of (min, max) row-count intervals."""
        conditions = []
        for min_rows, max_rows in ranges:
            if min_rows == max_rows:
                conditions.append(f"COUNT(*) = {min_rows}")
            elif min_rows <= 0:
                conditions.append(f"COUNT(*) <= {max_rows}")
            else:
                conditions.append(f"COUNT(*) >= {min_rows} AND COUNT(*) <= {max_rows}")
        if len(conditions) == 1:
            return conditions[0]
        return " OR ".join(f"({c})" for c in conditions)

    # ------------------------------------------------------------------
    # RewriteRule interface
    # ------------------------------------------------------------------

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        ranges = self._get_valid_ranges(ast.pattern)
        return ranges is not None and len(ranges) > 0

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        ranges = self._get_valid_ranges(ast.pattern)
        if ranges is None or len(ranges) == 0:
            return False

        having_clause = self._build_having_clause(ranges)
        original_input = ast.input_table.strip()

        if ast.partition_by and len(ast.partition_by) > 0:
            # ----------------------------------------------------------------
            # With PARTITION BY: filter partitions whose row count is in range.
            # ----------------------------------------------------------------
            partition_cols = [col.strip() for col in ast.partition_by]
            partition_cols_str = ", ".join(partition_cols)
            partition_size_query = (
                f"SELECT {partition_cols_str}\n"
                f"    FROM {original_input}\n"
                f"    GROUP BY {partition_cols_str}\n"
                f"    HAVING {having_clause}"
            )

            # IN cannot retain a NULL partition key.  Use it only when every
            # key is known to be NOT NULL; unknown metadata must take the
            # conservative null-safe path as well.
            if any(ast.is_column_nullable(col) is not False for col in partition_cols):
                input_alias = choose_generated_relation_name(ast, "RPR_INPUT")
                filter_alias = choose_generated_relation_name(
                    ast, "NEW_PARTITIONS", {input_alias}
                )
                key_equalities = " AND ".join(
                    f"({input_alias}.{col.split('.')[-1]} = {filter_alias}.{col.split('.')[-1]} OR "
                    f"({input_alias}.{col.split('.')[-1]} IS NULL AND "
                    f"{filter_alias}.{col.split('.')[-1]} IS NULL))"
                    for col in partition_cols
                )
                aliased_input = bind_input_relation_alias(original_input, input_alias)
                ast.input_table = (
                    f"(SELECT {input_alias}.* FROM {aliased_input} "
                    f"WHERE EXISTS (SELECT 1 FROM ({partition_size_query}) "
                    f"{filter_alias} WHERE {key_equalities}))"
                )
            else:
                # Build the compact IN condition when every key is non-nullable.
                if len(partition_cols) == 1:
                    col = partition_cols[0]
                    in_condition = f"{col} IN ({partition_size_query})"
                else:
                    cols_tuple = "(" + ", ".join(partition_cols) + ")"
                    in_condition = f"{cols_tuple} IN ({partition_size_query})"
                ast.input_table = (
                    f"(SELECT * FROM {original_input} WHERE {in_condition})"
                )

        else:
            # ----------------------------------------------------------------
            # Without PARTITION BY: filter when the whole-table row count is
            # within the valid range.
            # ----------------------------------------------------------------
            row_count_alias = choose_generated_column_name(ast, "ROW_COUNT")
            filter_size_name = choose_generated_relation_name(ast, "FILTER_SIZE")
            size_subquery = (
                f"(SELECT COUNT(*) AS {row_count_alias}\n"
                f"    FROM {original_input}\n"
                f"    HAVING {having_clause})"
            )

            ast.input_table = (
                f"(SELECT * FROM {original_input} "
                f"WHERE EXISTS (SELECT 1 FROM {size_subquery} {filter_size_name}))"
            )

        return True


anchor_partition_size_test_cases = [
    # Test 1: PATTERN(^A{5}$) — single-element both anchors, exact count
    (
        1,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT * FROM (
    SELECT * FROM NASDAQ WHERE TICKER IN (SELECT TICKER FROM NASDAQ GROUP BY TICKER HAVING COUNT(*) = 5)
)
MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 2: PATTERN(^A B{1,5} C$ | ^A{20,25}$) — two anchored branches,
    # subquery source; ranges [3,7] OR [20,25]
    (
        2,
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE TS > 100) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A B{1,5} C$ | ^A{20,25}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT * FROM (
    SELECT * FROM NASDAQ WHERE (TS > 100) AND (TICKER IN (
        SELECT TICKER FROM (SELECT * FROM NASDAQ WHERE TS > 100)
        GROUP BY TICKER
        HAVING (COUNT(*) >= 3 AND COUNT(*) <= 7) OR (COUNT(*) >= 20 AND COUNT(*) <= 25)))
)
MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A B{1,5} C$ | ^A{20,25}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 3: PATTERN(^A B{2,5} C$) — single branch range [4,7]
    (
        3,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
        """
SELECT * FROM (
    SELECT * FROM NASDAQ WHERE TICKER IN (
        SELECT TICKER FROM NASDAQ
        GROUP BY TICKER
        HAVING COUNT(*) >= 4 AND COUNT(*) <= 7
    )
) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
    ),
    # Test 4: PATTERN(^A B{2,5} C$ | ^A{5,10}$) — overlapping ranges
    # [4,7] OR [5,10] => merged to [4,10]
    (
        4,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP DESC
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$ | ^A{5,10}$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE TICKER IN (
    SELECT TICKER FROM NASDAQ
    GROUP BY TICKER
    HAVING COUNT(*) >= 4 AND COUNT(*) <= 10
)) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP DESC
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$ | ^A{5,10}$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
    ),
    # Test 5: PATTERN(^A B{2,5} C$ | ^A{10,15}$) — disjoint ranges
    # [4,7] OR [10,15]
    (
        5,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$ | ^A{10,15}$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
        """
SELECT * FROM (SELECT * FROM NASDAQ WHERE TICKER IN (
    SELECT TICKER FROM NASDAQ
    GROUP BY TICKER
    HAVING (COUNT(*) >= 4 AND COUNT(*) <= 7) OR (COUNT(*) >= 10 AND COUNT(*) <= 15)
))
MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    ALL ROWS PER MATCH
    PATTERN(^A B{2,5} C$ | ^A{10,15}$)
    DEFINE A AS OPEN > 110, B AS OPEN > 100, C AS OPEN > 90
)""",
    ),
]


anchor_partition_size_table_schemas = {
    "NASDAQ": {
        "columns": [
            "TICKER",
            "TIMESTAMP",
            "TS",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOL",
        ],
    },
    "T": {
        "columns": ["PT", "TS", "ID", "TYPE"],
        "column_types": {"TYPE": "VARCHAR(10)"},
        "column_nullables": {"PT": False},
    },
    "T_NULLABLE": {
        "columns": ["PT", "TS", "TYPE"],
        "column_types": {
            "PT": "INT",
            "TS": "INT",
            "TYPE": "VARCHAR(10)",
        },
        "column_nullables": {
            "PT": True,
            "TS": False,
            "TYPE": False,
        },
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        AnchorPartitionSizeFilterRule,
        anchor_partition_size_test_cases,
        "anchor partition size filter",
        verbose,
        anchor_partition_size_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
