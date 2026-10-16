if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from .physical_navigation import (
    collect_physical_navigation,
    max_navigation_offset,
)
from expression_clause import SelectClause
import re


class AnchorTopKFilterRule(RewriteRule):
    """
    Rule description:
        When a MATCH_RECOGNIZE pattern has a start anchor (^) or end anchor ($) and the pattern
        has a finite maximum length K, pre-filter input rows using ROW_NUMBER() OVER (...) before
        executing MATCH_RECOGNIZE. This avoids scanning the full partition. If
        DEFINE or MEASURES navigates past the retained edge with PREV/NEXT, K
        is enlarged by the maximum physical offset.

        If the sort key is not unique, the query results may not be exactly the same.

    Applicability conditions:
        1. The top-level pattern consists entirely of START (^) anchors or entirely of END ($) anchors
           (not a mix of both). An AlternationPattern whose every branch is anchored with the same
           type also qualifies; in that case K is the maximum length across all branches.
        2. K (maximum pattern length) must be finite.
        3. The outer SELECT must either be explicit, or have inferable output
           columns, so that the extra RN column does not silently appear. ONE
           ROW PER MATCH output is inferable without schema; ALL ROWS PER MATCH
           requires `table_schemas`.
        4. PARTITION BY and ORDER BY must be present.
        5. For SELECT * with ALL ROWS PER MATCH, table_schemas must be provided
           and sufficient to infer MATCH_RECOGNIZE output columns.
        6. Every ORDER BY column must have schema metadata declaring it non-nullable.
           This prevents NULL peers from making the top-K boundary nondeterministic.

    Generated SQL shape — ^ anchor example:
        SELECT ...
        FROM (
            SELECT *
            FROM (
                SELECT TABLENAME.*,
                       ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) AS WNO
                FROM TABLENAME
            ) RANKED
            WHERE WNO <= K
        )
        MATCH_RECOGNIZE( ... )

    Generated SQL shape — $ anchor example:
        SELECT ...
        FROM (
            SELECT *
            FROM (
                SELECT TABLENAME.*,
                       ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ... DESC) AS WNO
                FROM TABLENAME
            ) RANKED
            WHERE WNO <= K
        )
        MATCH_RECOGNIZE( ... )

    Subquery source:
        When FROM is a subquery (input_table starts with '('), the ranked
        subquery uses a SUBTABLE alias:
            SELECT SUBTABLE.*, ROW_NUMBER() ... FROM (subquery) SUBTABLE
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "anchor_top_k_filter"

    # ------------------------------------------------------------------
    # Anchor detection helpers
    # ------------------------------------------------------------------

    def _collect_anchors(self, pattern) -> set:
        """Recursively collect all AnchorType values present anywhere in the pattern tree."""
        return _collect_anchors(pattern)

    def _is_start_anchored(self, pattern) -> bool:
        """Return True if the pattern is anchored at the start (^).

        Accepted forms:
          - AnchorPattern(..., START)
          - ConcatenationPattern whose first element is start-anchored
          - AlternationPattern whose every branch is start-anchored
        """
        return _is_start_anchored(pattern)

    def _is_end_anchored(self, pattern) -> bool:
        """Return True if the pattern is anchored at the end ($).

        Accepted forms:
          - AnchorPattern(..., END)
          - ConcatenationPattern whose last element is end-anchored
          - AlternationPattern whose every branch is end-anchored
        """
        return _is_end_anchored(pattern)

    def _extract_anchor_info(self, pattern) -> Optional[Tuple[AnchorType, int]]:
        """Return (anchor_type, max_length) if the pattern qualifies, else None."""
        all_anchors = self._collect_anchors(pattern)

        # Must have exactly one anchor type (not both, not none).
        if AnchorType.START in all_anchors and AnchorType.END in all_anchors:
            return None
        if not all_anchors:
            return None

        anchor_type = (
            AnchorType.START if AnchorType.START in all_anchors else AnchorType.END
        )

        # The anchor must be at the correct structural position.
        if anchor_type == AnchorType.START and not self._is_start_anchored(pattern):
            return None
        if anchor_type == AnchorType.END and not self._is_end_anchored(pattern):
            return None

        max_len = pattern.get_max_row_num()
        if max_len is None or max_len <= 0:
            return None

        return (anchor_type, max_len)

    def _navigation_extension(
        self, ast: RPRAbstractSyntaxTree, anchor_type: AnchorType
    ) -> Optional[int]:
        """Rows outside the maximum match needed by physical navigation."""
        # A start anchor removes the suffix, while an end anchor removes the
        # prefix.  Only navigation toward that removed side needs compensation.
        target = "NEXT" if anchor_type == AnchorType.START else "PREV"
        return max_navigation_offset(
            collect_physical_navigation(ast),
            target,
        )

    # ------------------------------------------------------------------
    # RewriteRule interface
    # ------------------------------------------------------------------

    @staticmethod
    def _has_select_star(ast: RPRAbstractSyntaxTree) -> bool:
        """Whether any outer SELECT item is ``*`` or a qualified ``M.*``."""
        if ast.output_names is None:
            return True
        return any(
            isinstance(expr, ColumnRef) and expr.name == "*"
            for expr, _ in ast.output_names.select_items
        )

    def _can_expand_select_star(self, ast: RPRAbstractSyntaxTree) -> bool:
        if not self._has_select_star(ast):
            return True
        output_columns = ast.get_match_recognize_output_columns()
        return output_columns is not None and len(output_columns) > 0

    def _expand_select_stars(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Freeze every outer star to the original MATCH_RECOGNIZE schema.

        ``RPRAbstractSyntaxTree.expand_select_star`` handles only a SELECT list
        consisting solely of ``*``.  The ROW_NUMBER helper can also leak
        through a qualified star mixed with other items, for example
        ``SELECT M.*, 1 AS K``.  Preserve the star's qualifier while replacing
        it with the schema known before the helper column is introduced.
        """
        if not self._has_select_star(ast):
            return True

        output_columns = ast.get_match_recognize_output_columns()
        if output_columns is None or len(output_columns) == 0:
            return False

        expanded_items = []
        for expression, alias in ast.output_names.select_items:
            if isinstance(expression, ColumnRef) and expression.name == "*":
                expanded_items.extend(
                    (ColumnRef(expression.qualifier, column), None)
                    for column in output_columns
                )
            else:
                expanded_items.append((expression, alias))
        ast.output_names = SelectClause(expanded_items)
        return True

    @staticmethod
    def _is_one_row_per_match(ast: RPRAbstractSyntaxTree) -> bool:
        return (
            ast.rows_per_match is None
            or ast.rows_per_match.strip().upper() == "ONE ROW PER MATCH"
        )

    def _inferred_one_row_input_columns(self, ast: RPRAbstractSyntaxTree) -> List[str]:
        """Return the input columns needed when no source schema is available."""
        columns = set(ast.get_required_input_table_columns())
        columns.update(col.strip() for col, _ in (ast.match_order_by or []))
        return sorted(columns, key=lambda col: col.upper())

    def _source_columns(self, ast: RPRAbstractSyntaxTree) -> List[str]:
        if ast.table_schemas is None:
            return []
        columns = ast._extract_join_source_columns()
        if columns is None:
            flattened = ast.flatten_safe_select_star_input(ast.input_table)
            if flattened is not None:
                base_source, _ = flattened
                original_input = ast.input_table
                try:
                    ast.input_table = base_source
                    columns = ast._extract_join_source_columns()
                finally:
                    ast.input_table = original_input
        if columns is None:
            columns = self._derived_projection_columns(ast)
        return columns if columns is not None else []

    @staticmethod
    def _derived_projection_columns(
        ast: RPRAbstractSyntaxTree,
    ) -> Optional[List[str]]:
        """Return the explicit output columns of a bare derived table.

        ``_extract_join_source_columns`` resolves ordinary relations from
        schema metadata, but intentionally returns ``None`` for a derived
        table.  For an explicit SELECT list its row type is still knowable and
        must be considered when choosing the ROW_NUMBER helper alias.
        """
        input_sql = ast.input_table.strip()
        if not AnchorTopKFilterRule._is_bare_subquery(input_sql):
            return None

        query = input_sql[1:-1].strip()
        if not re.match(r"^SELECT\b", query, re.IGNORECASE):
            return None
        from_pos = ast._find_top_level_clause_positions(query, ["FROM"])["FROM"]
        if from_pos == -1:
            return None

        try:
            select_clause = ExprParser().parse_select(query[:from_pos].strip())
        except (TypeError, ValueError):
            return None

        columns = []
        for expression, alias in select_clause.select_items:
            if alias is not None:
                columns.append(alias)
            elif isinstance(expression, ColumnRef) and expression.name != "*":
                columns.append(expression.name.split(".")[-1])
            else:
                # SELECT * and unaliased expressions do not expose a complete,
                # dependable set of output names.
                return None
        return columns

    def _choose_rn_alias(self, ast: RPRAbstractSyntaxTree) -> str:
        used = {col.strip().upper() for col in self._source_columns(ast)}
        if self._is_one_row_per_match(ast):
            used.update(
                col.strip().split(".")[-1].upper()
                for col in self._inferred_one_row_input_columns(ast)
            )
        base_alias = "WNO"
        if base_alias not in used:
            return base_alias

        suffix = 1
        while True:
            candidate = f"{base_alias}_{suffix}"
            if candidate not in used:
                return candidate
            suffix += 1

    @staticmethod
    def _is_bare_subquery(input_sql: str) -> bool:
        """True only for ``(query)`` with no pre-existing trailing alias."""
        sql = input_sql.strip()
        if not sql.startswith("("):
            return False
        depth = 0
        quote = None
        for idx, char in enumerate(sql):
            if quote is not None:
                if char == quote:
                    quote = None
                continue
            if char in ("'", '"'):
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return idx == len(sql) - 1
                if depth < 0:
                    return False
        return False

    @staticmethod
    def _is_supported_input_relation(input_sql: str) -> bool:
        """Accept a single unaliased table name or a bare parenthesized query.

        Rebinding aliases and arbitrary JOIN trees requires a relational AST;
        emitting SQL for them from the current text-only ``input_table`` field
        is unsafe, so this rule deliberately declines those inputs.
        """
        sql = input_sql.strip()
        if sql.startswith("("):
            return AnchorTopKFilterRule._is_bare_subquery(sql)
        return (
            re.fullmatch(
                r'(?:[A-Za-z_][A-Za-z0-9_$]*|"[^"]+")'
                r'(?:\.(?:[A-Za-z_][A-Za-z0-9_$]*|"[^"]+"))*',
                sql,
            )
            is not None
        )

    def _has_known_source_columns(self, ast: RPRAbstractSyntaxTree) -> bool:
        return ast.table_schemas is not None and len(self._source_columns(ast)) > 0

    @staticmethod
    def _unqualify_input_column(column: str) -> str:
        """Remove a source-relation qualifier before wrapping the input.

        This rule accepts only one input relation (a table or a bare derived
        query), so an input qualifier is unnecessary.  More importantly, the
        original table name is no longer visible to MATCH_RECOGNIZE after the
        relation has been wrapped by the top-K derived table.
        """
        return column.strip().split(".")[-1]

    def _unbind_wrapped_input_qualifiers(self, ast: RPRAbstractSyntaxTree) -> None:
        ast.partition_by = [
            self._unqualify_input_column(column) for column in ast.partition_by or []
        ]
        ast.match_order_by = [
            (self._unqualify_input_column(column), sort_type)
            for column, sort_type in ast.match_order_by or []
        ]

    @staticmethod
    def _has_non_nullable_order_columns(ast: RPRAbstractSyntaxTree) -> bool:
        """Whether every MATCH_RECOGNIZE ordering column is known non-nullable."""
        for order_col, _ in ast.match_order_by or []:
            metadata = ast.get_column_metadata(order_col.strip())
            if metadata is None or metadata.nullable is not False:
                return False
        return True

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        if not self._is_supported_input_relation(ast.input_table):
            return False
        # ALL ROWS carries every source column through MATCH_RECOGNIZE.  Without
        # a schema there is no collision-free name we can prove to be fresh.
        if not self._is_one_row_per_match(ast) and not self._has_known_source_columns(
            ast
        ):
            return False
        if not self._can_expand_select_star(ast):
            return False
        anchor_info = self._extract_anchor_info(ast.pattern)
        if anchor_info is None:
            return False
        if self._navigation_extension(ast, anchor_info[0]) is None:
            return False
        if ast.partition_by is None or len(ast.partition_by) == 0:
            return False
        if ast.match_order_by is None or len(ast.match_order_by) == 0:
            return False
        if not self._has_non_nullable_order_columns(ast):
            return False
        return True

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        anchor_info = self._extract_anchor_info(ast.pattern)
        if anchor_info is None:
            return False
        anchor_type, max_len = anchor_info
        navigation_extension = self._navigation_extension(ast, anchor_type)
        if navigation_extension is None:
            return False
        max_len += navigation_extension

        # The generated derived table hides the original input relation name.
        # Keep the column references valid in both the ROW_NUMBER window and
        # the outer MATCH_RECOGNIZE clause.
        self._unbind_wrapped_input_qualifiers(ast)

        if not self._expand_select_stars(ast):
            return False

        rn_alias = self._choose_rn_alias(ast)
        partition_cols = ", ".join(col.strip() for col in ast.partition_by)

        def render_order(reverse: bool) -> str:
            rendered = []
            for order_col, sort_dir in ast.match_order_by:
                is_desc = sort_dir == SortType.DESC
                if reverse:
                    is_desc = not is_desc
                rendered.append(f"{order_col.strip()} {'DESC' if is_desc else 'ASC'}")
            return ", ".join(rendered)

        input_table = ast.input_table.strip()
        is_subquery = input_table.startswith("(")
        if is_subquery:
            table_ref = choose_generated_relation_name(ast, "SUBTABLE")
            from_clause = f"{input_table} {table_ref}"
        else:
            table_ref = input_table
            from_clause = input_table

        # ^ anchor: ROW_NUMBER uses same direction as ORDER BY (first K rows).
        # $ anchor: ROW_NUMBER uses opposite direction (last K rows = first K in reverse).
        rn_order = render_order(anchor_type == AnchorType.END)

        if self._is_one_row_per_match(ast):
            # ONE ROW PER MATCH only needs columns referenced by its clauses.
            # Projecting exactly those columns also prevents an explicit
            # derived-table output such as ``ID AS WNO`` from colliding with
            # the generated row-number helper.
            input_projection = ", ".join(self._inferred_one_row_input_columns(ast))
        else:
            input_projection = f"{table_ref}.*"

        ranked_name = choose_generated_relation_name(ast, "RANKED")
        ast.input_table = (
            f"(SELECT *\n"
            f"    FROM (SELECT {input_projection}, "
            f"ROW_NUMBER() OVER (PARTITION BY {partition_cols} ORDER BY {rn_order}) AS {rn_alias}\n"
            f"        FROM {from_clause}) {ranked_name}\n"
            f"    WHERE {rn_alias} <= {max_len})"
        )
        return True


anchor_topk_test_cases = [
    # Test 1: ^ anchor, ORDER BY ASC, fixed-length pattern A{5}
    (
        1,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (
        SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
        FROM NASDAQ
) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 2: ^ anchor, ORDER BY ASC, variable-range quantifier B{2,10}
    # max_len = 1 + 10 + 1 + 1 = 13
    (
        2,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A B{2, 10} C D)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 13) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A B{2,10} C D)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 3: ^ anchor, ORDER BY ASC, subquery source → SUBTABLE alias
    (
        3,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM NASDAQ WHERE TICKER IN ('AAPL', 'TSLA', 'META'))
MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE
      A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT SUBTABLE.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
    FROM (SELECT * FROM NASDAQ WHERE TICKER IN ('AAPL', 'TSLA', 'META')) SUBTABLE) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 4: $ anchor, ORDER BY ASC → ROW_NUMBER uses DESC (opposite)
    # Selects last 5 rows per partition.
    (
        4,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP DESC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 5: AlternationPattern with all ^ branches — k = max(6, 12) = 12
    (
        5,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN((^A{5} B) | (^B A{11}))
    DEFINE
      A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 12
) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5} B | ^B A{11})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 6: $ anchor, ORDER BY ASC, no table_columns → now rewritable
    # (previously not rewritable; table_columns no longer required)
    (
        6,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{3}$)
    DEFINE A AS OPEN > 110
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP DESC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 3
) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{3}$)
    DEFINE A AS OPEN > 110
)""",
    ),
    # Test 7: Not rewritable — unbounded ^ pattern (A+), no finite max_len
    (
        7,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A+ B)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A+ B)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 8: Not rewritable — mixed ^ and $ in alternation branches
    (
        8,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN((^A{5} B) | (A{5} B$))
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5} B | A{5} B$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 9: Not rewritable — ^ only on one branch of alternation
    (
        9,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN((^A{5} B) | (B A{5}))
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5} B | B A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 10: SELECT * is rewritable when table_schemas can expand output columns.
    (
        10,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 11: $ anchor, ORDER BY ASC, no table_columns
    (
        11,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP DESC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 12: $ anchor, ORDER BY DESC → ROW_NUMBER uses ASC (opposite)
    # When ORDER BY is DESC, "last K rows" are the K smallest timestamps.
    (
        12,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER
    ORDER BY TIMESTAMP DESC
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP ASC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP DESC
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(A{5}$)
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
    # Test 13: ^ anchor, ORDER BY DESC → ROW_NUMBER uses DESC (same direction)
    # "First K rows" in DESC order = K rows with the largest timestamps.
    (
        13,
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM NASDAQ MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP DESC
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
        """
SELECT TICKER, TIMESTAMP, MN, OPEN, HIGH, LOW, CLOSE, VOL
FROM (SELECT * FROM (SELECT NASDAQ.*, ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY TIMESTAMP DESC) AS WNO
    FROM NASDAQ) RANKED WHERE WNO <= 5) MATCH_RECOGNIZE(
    PARTITION BY TICKER ORDER BY TIMESTAMP DESC
    MEASURES MATCH_NUMBER() AS MN
    ALL ROWS PER MATCH
    PATTERN(^A{5})
    DEFINE A AS OPEN > 110 AND VOL > 150
)""",
    ),
]


default_anchor_topk_table_schemas = {
    "NASDAQ": {
        "columns": ["TICKER", "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"],
    },
    "NASDAQ_WITH_WNO": {
        "columns": [
            "TICKER",
            "TIMESTAMP",
            "WNO",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOL",
        ],
    },
    "NASDAQ_NULLABLE_ORDER": {
        "columns": ["TICKER", "TIMESTAMP", "OPEN"],
        "column_nullables": {"TIMESTAMP": True},
    },
    "TOPK_SOURCE": {
        "columns": ["ID", "P", "TS", "V"],
    },
    "TOPK_STAR_SOURCE": {
        "columns": ["ID", "PT", "TS", "VAL"],
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    schema_result = run_rule_tests(
        AnchorTopKFilterRule,
        anchor_topk_test_cases,
        "anchor top-k filter",
        verbose,
        default_anchor_topk_table_schemas,
    )
    return schema_result


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
