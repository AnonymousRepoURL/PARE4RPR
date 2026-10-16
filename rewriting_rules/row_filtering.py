if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from .physical_navigation import PatternNavigationReach
from .sequence_filtering import _is_true
from clause_parser import ExprParser
from expression_clause import SelectClause


class RowFilteringRule(RewriteRule):
    """
    Baseline Sequence Filtering rewrite from
    "Optimization of Row Pattern Matching over Sequence Data in Spark SQL."

    We implement the row filtering rule according to the description in the paper.
        - The condition to apply Row Filtering is that "there is at least one value condition
        for all pattern variables in DEFINE clause".
        - If Row Filtering ﬁlters out rows which do not meet valueconditions in DEFINE clause,
        it becomes diﬃcult to tell whether the remainingrows reside adjacent to each other
        in the original input table.
    """

    @property
    def name(self) -> str:
        return "row_filtering"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_plan(ast) is not None

    @staticmethod
    def _has_select_star(ast: RPRAbstractSyntaxTree) -> bool:
        """Whether any outer SELECT item is ``*`` or a qualified ``M.*``."""
        if ast.output_names is None:
            return False
        return any(
            isinstance(expression, ColumnRef) and expression.name == "*"
            for expression, _ in ast.output_names.select_items
        )

    def _expand_select_stars(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Freeze outer stars before adding the row-number helper column."""
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

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_row_filtering_applied", False):
            return False
        plan = self._build_plan(ast)
        if plan is None:
            return False

        # The helper row id is part of the MATCH_RECOGNIZE input. Under ALL
        # ROWS PER MATCH it would leak through any outer star item (including
        # M.* mixed with other items), so freeze stars to the original output
        # schema before adding it.
        if plan["expand_output_star"] and not self._expand_select_stars(ast):
            return False

        source = ast.input_table.strip()
        row_id = plan["row_id"]
        numbered_rows = plan["numbered_rows"]
        projection_sql = ", ".join(plan["projection_columns"])
        partition_sql = ""
        if ast.partition_by:
            partition_sql = f"PARTITION BY {', '.join(ast.partition_by)} "
        order_sql = ", ".join(
            col if direction == SortType.ASC else f"{col} DESC"
            for col, direction in ast.match_order_by
        )
        ast.input_table = (
            f"(SELECT * FROM (SELECT {projection_sql}, "
            f"ROW_NUMBER() OVER ({partition_sql}ORDER BY {order_sql}) "
            f"AS {row_id} FROM {source}) "
            f"{numbered_rows} "
            f"WHERE {plan['filter_condition']})"
        )
        # PARTITION BY / ORDER BY may use the correlation name of the original
        # row source (for example ``T X ... PARTITION BY X.PT``). Rebind the
        # rewritten derived table to that same name so those references remain
        # in scope at MATCH_RECOGNIZE.
        if plan["input_rebind_alias"] is not None:
            ast.input_table = f"{ast.input_table} {plan['input_rebind_alias']}"

        # A proven one-row leading variable may start at any retained row.
        # Every other variable must immediately follow the preceding original
        # input row; no FIRST() disjunction is needed on each DEFINE predicate.
        adjacency_sql = f"{row_id} = PREV({row_id}) + 1"
        parser = ExprParser()
        for var_name in plan["variables"]:
            if var_name in plan["leading_variables"]:
                continue
            old_condition = ast.define.definitions[var_name]
            combined = f"DEFINE {var_name} AS ({old_condition.expression.to_string()}) AND {adjacency_sql}"
            ast.define.definitions[var_name] = parser.parse_define(
                combined
            ).definitions[var_name]
        ast._row_filtering_applied = True
        return True

    def _build_plan(self, ast: RPRAbstractSyntaxTree):
        if (
            not ast.match_order_by
            or ast.rows_per_match == "ALL ROWS PER MATCH WITH UNMATCHED ROWS"
        ):
            return None
        qualifier_safe, input_rebind_alias = resolve_input_rebind_alias(ast)
        if not qualifier_safe:
            return None
        # The generated row number proves that every row of a surviving match
        # was adjacent in the original input.  Physical PREV/NEXT is therefore
        # preserved when its offset is guaranteed to remain between the first
        # and last rows of the match; navigation across either edge is unsafe.
        if not self._physical_navigation_stays_within_match(ast):
            return None
        variables = sorted(ast.pattern.get_variables())
        if not variables:
            return None
        leading_variables = self._single_row_leading_variables(ast.pattern)
        if not leading_variables:
            return None

        if (
            ast.rows_per_match != "ONE ROW PER MATCH"
            and ast.rows_per_match != "ALL ROWS PER MATCH"
        ):
            return None

        predicates = []
        seen = set()
        for var_name in variables:
            condition = ast.define.definitions.get(var_name)
            if condition is None:
                return None
            independent = condition.get_independent_filter()
            independent = _strip_var_qualifier(independent, var_name)
            if independent is None or _is_true(independent):
                return None
            sql = independent.to_string()
            if sql not in seen:
                seen.add(sql)
                predicates.append(sql)
        if not predicates:
            return None

        projection_columns = self._projection_columns(ast)
        if not projection_columns:
            return None
        all_rows = (
            ast.rows_per_match is not None
            and ast.rows_per_match.strip().upper() == "ALL ROWS PER MATCH"
        )
        expand_output_star = all_rows and self._has_select_star(ast)
        if expand_output_star and not ast.get_match_recognize_output_columns():
            return None

        # table_schemas catches collisions against the complete known schema;
        # projection_columns also protects ONE ROW PER MATCH when no schema was
        # supplied but the query itself references an WNO-named column.
        row_id = choose_generated_column_name(ast, "WNO", projection_columns)
        numbered_rows = choose_generated_relation_name(ast, "ADD_WNO_ROWS")
        return {
            "variables": variables,
            "leading_variables": leading_variables,
            "row_id": row_id,
            "numbered_rows": numbered_rows,
            "projection_columns": projection_columns,
            "expand_output_star": expand_output_star,
            "input_rebind_alias": input_rebind_alias,
            "filter_condition": " OR ".join(
                f"({predicate})" for predicate in predicates
            ),
        }

    def _physical_navigation_stays_within_match(
        self,
        ast: RPRAbstractSyntaxTree,
    ) -> bool:
        """Whether DEFINE/MEASURES PREV/NEXT can only read matched rows."""
        return PatternNavigationReach(ast).all_navigation_stays_within_match()

    def _single_row_leading_variables(self, pattern):
        """Return safe unconstrained first variables, otherwise ``None``.

        Accepted prefixes are either one bare variable or a flat alternation
        whose every branch is one bare variable. Thus every accepted prefix
        consumes exactly one row. Each leading variable must occur exactly once
        in the whole pattern, preventing an exemption intended for the first
        row from also applying to a later occurrence of the same label.
        """
        prefix = pattern
        if isinstance(pattern, ConcatenationPattern):
            if not pattern.patterns:
                return None
            prefix = pattern.patterns[0]

        if isinstance(prefix, VariablePattern):
            leading = {prefix.name}
        elif (
            isinstance(prefix, AlternationPattern)
            and prefix.patterns
            and all(isinstance(branch, VariablePattern) for branch in prefix.patterns)
        ):
            leading = {branch.name for branch in prefix.patterns}
            # Repeated labels in separate branches add no useful alternative
            # and make the syntactic occurrence proof below ambiguous.
            if len(leading) != len(prefix.patterns):
                return None
        else:
            return None
        for variable in leading:
            if self._count_variable_occurrences(pattern, variable) != 1:
                return None
        return leading

    def _count_variable_occurrences(self, pattern, variable):
        if isinstance(pattern, VariablePattern):
            return 1 if pattern.name == variable else 0
        if hasattr(pattern, "patterns"):
            return sum(
                self._count_variable_occurrences(child, variable)
                for child in pattern.patterns
            )
        child = getattr(pattern, "pattern", None)
        return (
            self._count_variable_occurrences(child, variable)
            if child is not None
            else 0
        )

    def _projection_columns(self, ast: RPRAbstractSyntaxTree):
        """Return the columns carried through numbering and row filtering.

        ONE ROW PER MATCH needs only columns consumed by MATCH_RECOGNIZE.
        ALL ROWS PER MATCH exposes every input column, so it requires a complete
        source schema and expands that schema instead of using ``*``.
        """
        rows_per_match = (
            ast.rows_per_match.strip().upper()
            if ast.rows_per_match is not None
            else "ONE ROW PER MATCH"
        )
        if rows_per_match == "ALL ROWS PER MATCH":
            if ast.table_schemas is None:
                return None
            columns = ast._extract_join_source_columns()
            if not columns:
                columns = getattr(ast, "_window_prefilter_input_columns", None)
            if not columns:
                columns = getattr(ast, "_preserved_input_columns", None)
            if not columns:
                return None
            return self._normalize_columns(columns)

        ordered = list(ast.partition_by or [])
        ordered.extend(column for column, _ in (ast.match_order_by or []))
        required = ast.get_required_input_table_columns()
        if not isinstance(required, set):
            return None
        ordered.extend(sorted(required))
        return self._normalize_columns(ordered)

    @staticmethod
    def _normalize_columns(columns):
        result = []
        seen = set()
        for column in columns:
            clean = str(column).strip()
            if not clean:
                continue
            # MATCH_RECOGNIZE clauses address columns of their immediate input;
            # source-table qualifiers do not survive the derived-table boundary.
            clean = clean.split(".")[-1]
            key = clean.upper()
            if key in seen:
                continue
            seen.add(key)
            result.append(clean)
        return result


row_filtering_test_cases = [
    (
        1,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B', C AS C.TYPE = 'C'
)""",
        """
SELECT * FROM (
    SELECT * 
    FROM (SELECT PT, TS, ID, TYPE, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS, ID DESC) AS WNO FROM T) ADD_WNO_ROWS 
    WHERE (TYPE = 'A') OR (TYPE = 'B') OR (TYPE = 'C')
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS, ID DESC
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B C)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND WNO = PREV(WNO) + 1, C AS C.TYPE = 'C' AND WNO = PREV(WNO) + 1
)""",
    ),
    (
        2,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A'
)""",
    ),
    (
        3,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS TRUE
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.ID AS AID, B.ID AS BID
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS TRUE
)""",
    ),
    (
        4,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT PT, TS, ID, TYPE, VALUE 
FROM (
    SELECT * 
    FROM (SELECT PT, TS, ID, TYPE, VALUE, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO FROM T) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND WNO = PREV(WNO) + 1
)""",
    ),
    (
        5,
        """
SELECT * FROM T_WITHOUT_VALUE MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT PT, TS, ID, TYPE
FROM (SELECT * FROM (SELECT PT, TS, ID, TYPE,
    ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO
    FROM T_WITHOUT_VALUE) ADD_WNO_ROWS
    WHERE (TYPE = 'A') OR (TYPE = 'B'))
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS TYPE = 'A',
           B AS TYPE = 'B' AND WNO = (PREV(WNO) + 1)
)""",
    ),
    (
        6,
        """
SELECT V FROM T MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES FIRST(A.VALUE) AS V
    PATTERN(A B)
    DEFINE A AS TYPE = 'A', B AS TYPE = 'B'
)""",
        """
SELECT V FROM (
    SELECT * 
    FROM (
        SELECT PT, TS, TYPE, VALUE, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO 
        FROM T
    ) ADD_WNO_ROWS 
    WHERE (TYPE = 'A') OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.VALUE) AS V
    PATTERN(A B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B' AND WNO = PREV(WNO) + 1
)""",
    ),
    (
        7,
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.ID) AS AID
    PATTERN(A+ B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM T MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES FIRST(A.ID) AS AID
    PATTERN(A+ B)
    DEFINE A AS A.TYPE = 'A', B AS B.TYPE = 'B'
)""",
    ),
    (
        8,
        """
SELECT * FROM T_WITH_WNO MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.WNO AS ORIGINAL_ID
    PATTERN(A B)
    DEFINE A AS A.WNO > 0, B AS B.TYPE = 'B'
)""",
        """
SELECT * FROM (
    SELECT * 
    FROM (SELECT PT, TS, TYPE, WNO, ROW_NUMBER() OVER (PARTITION BY PT ORDER BY TS) AS WNO_1 FROM T_WITH_WNO) ADD_WNO_ROWS
    WHERE (WNO > 0) OR (TYPE = 'B')
) MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.WNO AS ORIGINAL_ID
    PATTERN(A B)
    DEFINE A AS A.WNO > 0, B AS B.TYPE = 'B' AND WNO_1 = PREV(WNO_1) + 1
)""",
    ),
]


default_row_filtering_table_schemas = {
    "T": {
        "columns": ["PT", "TS", "ID", "TYPE", "VALUE"],
    },
    "T_WITHOUT_VALUE": {
        "columns": {
            "PT": "VARCHAR(20)",
            "TS": "LONG",
            "ID": "LONG",
            "TYPE": "VARCHAR(20)",
        },
        "column_nullables": {
            "PT": False,
            "TS": False,
            "ID": False,
            "TYPE": False,
        },
    },
    "T_WITH_WNO": {
        "columns": ["PT", "TS", "WNO", "TYPE"],
    },
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    return run_rule_tests(
        RowFilteringRule,
        row_filtering_test_cases,
        "row filtering baseline",
        verbose,
        default_row_filtering_table_schemas,
    )


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
