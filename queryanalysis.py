from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from enum import Enum
import re
from patternnode import (
    PatternNode,
    PatternLexer,
    PatternParser,
    contains_excluded_pattern,
    get_pattern_variables,
    unwrap_excluded_pattern,
)
from expression import (
    Expr,
    BinaryExpr,
    BetweenExpr,
    CaseExpr,
    ColumnRef,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    NullLiteral,
    SemanticsExpr,
)
from expression_clause import DefineSubClause, MeasuresSubClause, SelectClause
from clause_parser import ExprParser
from expression_simplifier import ExpressionColumnMetadata, SimplifyContext

# MATCH_RECOGNIZE "PER MATCH" output modes.
ONE_ROW_PER_MATCH = "ONE ROW PER MATCH"
ALL_ROWS_PER_MATCH = "ALL ROWS PER MATCH"
ALL_ROWS_PER_MATCH_SHOW_EMPTY_MATCHES = "ALL ROWS PER MATCH SHOW EMPTY MATCHES"
ALL_ROWS_PER_MATCH_OMIT_EMPTY_MATCHES = "ALL ROWS PER MATCH OMIT EMPTY MATCHES"
ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS = "ALL ROWS PER MATCH WITH UNMATCHED ROWS"
# Default "AFTER MATCH SKIP TO" target when the clause is omitted.
PAST_LAST_ROW = "PAST LAST ROW"


class SortType(Enum):
    ASC = 1  # default
    DESC = 2


class ColumnDataType(Enum):
    """Base column types supported by the project.

    CHAR and VARCHAR require a positive length and are stored in
    ``ColumnMetadata`` in canonical forms such as ``CHAR(8)``.
    """

    INT = "INT"
    LONG = "LONG"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    CHAR = "CHAR"
    VARCHAR = "VARCHAR"


@dataclass(frozen=True)
class ColumnMetadata:
    """Normalized metadata for a source-table column.

    Every resolved column has definite type/nullability metadata. The type is
    one of INT, LONG, FLOAT, DOUBLE, CHAR(n), or VARCHAR(n). Primary-key columns
    default to non-null while every other column defaults to nullable.
    """

    table_name: str
    name: str
    data_type: str
    nullable: bool


class RPRAbstractSyntaxTree:
    """
    Row Pattern Recognize (RPR) abstract syntax tree
    We implement the lightweight MATCH_RECOGNIZE parser rather than using antlr4 to analyze RPR sql.
    We only process simple wrong syntax. Thus, please keep the input sql syntax and semantic is correct.

    For aggregation functions, only COUNT, SUM, AVG, MIN, and MAX are currently supported
    in the MATCH_RECOGNIZE clause. Distinct aggregates (e.g., MEDIAN) are currently not supported in the
    MATCH_RECOGNIZE clause.

    Sql template we support is shown in below:
        SELECT column_name1, column_name2, ... column_nameN
        FROM input_table
        MATCH_RECOGNIZE(xxx) [AS alias]
        [WHERE xxx]
        [GROUP BY xxx]
        [HAVING xxx]
        [ORDER BY xxx]
        [LIMIT xxx]

    where input_table can be a real table or a complex derived table.

    **Please note that we do not implement the following features:**
    1. We do not support the column alias using quotes in SELECT clause and MEASURES subclause
    2. We do not support common table expression in original sql

    These feartures can be easily implemented in the future if needed.
    Our main goal is to support the MATCH_RECOGNIZE clause and its subclauses,
    including PARTITION BY, ORDER BY, MEASURES, PER MATCH, AFTER MATCH SKIP TO, PATTERN, SUBSET, and DEFINE.
    """

    output_names: SelectClause  # select clause output names, it can be a list of column names or a list of expressions with alias
    input_table: str  # input table can be a real table or a derived table, we store it as original text
    partition_by: List[
        str
    ]  # partition by columns for match_recognize, we need to support multiple columns
    match_order_by: List[
        Tuple[str, SortType]
    ]  # order by columns for match_recognize, we need to support multiple columns and sort type
    measures: MeasuresSubClause  # measures is a dict, key is alias, value is expression
    rows_per_match: str  # two output modes: ONE ROW PER MATCH | ALL ROWS PER MATCH, default is ONE ROW PER MATCH
    after_match_skip_to: (
        str  # skipping strategy after a match is found, default is PAST LAST ROW
    )
    pattern: PatternNode  # regex-style pattern, we use PatternLexer and PatternParser to parse it into a tree structure
    define: DefineSubClause  # Variables defined in the DEFINE clause must be in the PATTERN clause
    subsets: Dict[
        str, List[str]
    ]  # Note that variables defined in the SUBSET clause cannot be in the PATTERN clause.
    alias: str  # the output of MATCH_RECOGNIZE can be used as a derived table, so we need to support alias for it
    where: (
        Expr  # the WHERE clause is used to filter or sort the output of MATCH_RECOGNIZE
    )
    group_by: List[
        str
    ]  # the GROUP BY clause is used to group the output of MATCH_RECOGNIZE
    having: Expr  # the HAVING clause is used to filter the output of MATCH_RECOGNIZE after grouping
    order_by: List[
        Tuple[str, SortType]
    ]  # the ORDER BY clause is used to sort the output of MATCH_RECOGNIZE
    limit: int  # the LIMIT clause is used to limit the number of output rows of MATCH_RECOGNIZE

    # since most databses cannot support EXCLUDE, we need to know the schema of input table
    # Supported keys include columns, primary_key, column_types,
    # column_nullables, column_avg_sizes, avg_column_sizes and column_sizes.
    # ``columns`` may also contain dictionaries with name/data_type/nullable.
    table_schemas: Dict[str, Dict[str, Any]]

    # NOTE: this class intentionally defines its own ``__init__`` rather than
    # using ``@dataclass``: several instance attributes below (``cte_prefix``,
    # ``raw_sql_override``, and the metadata caches) are parser/optimizer
    # bookkeeping rather than AST fields, so an auto-generated
    # ``__init__``/``__eq__``/``__repr__`` covering only the annotated fields
    # above would silently ignore them.

    def __init__(self) -> None:
        self.output_names = []
        self.input_table = ""
        self.partition_by = None
        self.match_order_by = None
        self.measures = None
        self.rows_per_match = ONE_ROW_PER_MATCH
        self.after_match_skip_to = PAST_LAST_ROW
        self.pattern = None
        self.define = None
        self.subsets = None
        self.alias = None
        self.where = None
        self.group_by = None
        self.having = None
        self.order_by = None
        self.limit = None
        self.table_schemas = None
        self.cte_prefix = None  # set by WindowConditionPreFilterRule
        self.raw_sql_override = None
        self.pattern_rewrite_boundaries = ()
        self._normalized_schema_cache_key = None
        self._normalized_schema_cache = None
        self._join_source_columns_cache_key = None
        self._join_source_columns_cache = None
        self._derived_metadata_resolution_stack = set()

    _MATCH_CLAUSE_ORDER = [
        "PARTITION BY",
        "ORDER BY",
        "MEASURES",
        "PER MATCH",
        "AFTER MATCH SKIP",
        "PATTERN",
        "SUBSET",
        "DEFINE",
    ]

    _TRAILING_CLAUSE_ORDER = ["WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"]

    def parse_sql(self, sql: str, table_schemas: Optional[Dict] = None) -> None:
        """
        upper mode for sql
        1. parse output names, input table, and match_recognize content
        2. parse match_recognize content, including partition by, order by, measures, after match skip to, pattern, subset, and define

        ``table_schemas`` is required because DEFINE simplification depends on
        each resolved column's data type and nullability.
        """
        if not isinstance(table_schemas, dict) or not table_schemas:
            raise ValueError(
                "input table schema metadata is required; pass table_schemas "
                "to parse_sql()"
            )
        sql = self._normalize_sql(sql)
        # These optional clauses must be reset for every parse.  In
        # particular, an omitted PER MATCH clause means ONE ROW PER MATCH and
        # must not inherit an ALL ROWS mode from a previous parse.
        self.partition_by = None
        self.match_order_by = None
        self.measures = None
        self.rows_per_match = ONE_ROW_PER_MATCH
        self.after_match_skip_to = PAST_LAST_ROW
        self.subsets = None
        self.alias = None
        self.where = None
        self.group_by = None
        self.having = None
        self.order_by = None
        self.limit = None
        self.pattern_rewrite_boundaries = ()
        self.table_schemas = table_schemas
        self._clear_metadata_cache()
        # An expression parser is stateless between parse calls (each public
        # method resets it), so reuse one instance throughout this SQL parse.
        simplify_context = SimplifyContext(
            lambda column: self.get_column_metadata(column.to_string())
        )
        self._expr_parser = ExprParser(simplify_context)
        select_pos = sql.find("SELECT")
        from_pos = sql.find("FROM")
        match_recognize_pos = sql.find("MATCH_RECOGNIZE")

        # if we cannot capture these keywords, we raise exception
        if select_pos != 0 or from_pos == -1 or match_recognize_pos == -1:
            raise ValueError("illegal 'SELECT xxx FROM xxx MATCH_RECOGNIZE' statement")

        # set output_names
        output_str = sql[:from_pos]
        self.output_names = self._expr_parser.parse_select(output_str)

        # Keep the row-pattern input as original text so it can be emitted
        # unchanged, but validate its outer shape before treating it as an
        # opaque relation.  MATCH_RECOGNIZE accepts one table-like relation;
        # a JOIN tree must therefore be wrapped by a derived-table query.
        self.input_table = sql[from_pos + len("FROM") : match_recognize_pos].strip()
        self._validate_match_recognize_input_table()

        start_pos = sql.find("(", match_recognize_pos + len("MATCH_RECOGNIZE"))
        if start_pos == -1:
            raise ValueError(" 'MATCH_RECOGNIZE' miss left parenthesis")

        end_pos = self._find_matching_paren(sql, start_pos)
        if end_pos == -1:
            raise ValueError(" 'MATCH_RECOGNIZE' miss right parenthesis")

        # pre-compute
        sub_sql = sql[start_pos + 1 : end_pos]
        match_clauses = self._find_match_clause_positions(sub_sql)

        if match_clauses["PATTERN"] == -1:
            raise ValueError(" 'MATCH_RECOGNIZE' miss 'PATTERN' subclause ")
        if match_clauses["DEFINE"] == -1:
            raise ValueError(" 'MATCH_RECOGNIZE' miss 'DEFINE' subclause ")

        # analyze each subclause
        # Parse the clauses that do not depend on pattern-variable names first.
        # MEASURES appears before PATTERN in SQL text, but resolving a qualified
        # reference such as ``A.ID`` requires PATTERN and SUBSET to have already
        # established the available primary and union variables.
        self._parse_partition_by(sub_sql, match_clauses)
        self._parse_order_by(sub_sql, match_clauses)
        self._parse_rows_per_match(sub_sql, match_clauses)
        self._parse_after_match(sub_sql, match_clauses)
        self._parse_pattern(sub_sql, match_clauses)
        self._validate_excluded_pattern_for_output_mode()
        self._simplify_excluded_pattern_for_output_mode()
        self._parse_subset(sub_sql, match_clauses)
        self._parse_define(sub_sql, match_clauses)
        self._parse_measures(sub_sql, match_clauses)
        self._validate_measures_for_output_mode()
        self._validate_define_column_metadata()
        self._parse_trailing_subclauses(sql[end_pos + 1 :])

        validation_errors = self.validate()
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        self._expr_parser = None

    def parse(self, sql: str, table_schemas: Optional[Dict] = None) -> None:
        return self.parse_sql(sql, table_schemas)

    def expression_simplify_context(self) -> SimplifyContext:
        if not self.table_schemas:
            raise ValueError(
                "input table schema metadata is required before simplifying "
                "expressions"
            )
        return SimplifyContext(
            lambda column: self.get_column_metadata(column.to_string())
        )

    def match_output_simplify_context(self) -> SimplifyContext:
        """Resolve metadata for expressions evaluated after MATCH_RECOGNIZE."""
        return SimplifyContext(
            lambda column: (
                self.get_match_recognize_output_column_metadata(column.to_string())
                or ExpressionColumnMetadata()
            )
        )

    def optimize_from_clause(self) -> bool:
        """Simplify safe, redundant wrappers in the MATCH_RECOGNIZE input.

        Currently this folds nested ``SELECT *`` derived tables, combining
        their WHERE predicates when necessary.  Clauses that can change row
        shape, ordering, or cardinality deliberately stop the optimization.

        Return True when ``input_table`` was changed.
        """
        if self.input_table is None:
            return False

        original = self.input_table
        current = original.strip()
        while True:
            collapsed = self._collapse_one_select_star_wrapper(current)
            if collapsed is None or collapsed == current:
                break
            current = collapsed

        if current == original:
            return False
        self.input_table = current
        self._clear_metadata_cache()
        return True

    def flatten_safe_select_star_input(self, input_table_sql: str):
        """Return ``(base_source, predicates)`` for a provably safe wrapper chain.

        This is deliberately a small SQL subset, not a best-effort parser.  It
        accepts only unaliased ordinary relation names wrapped by one or more
        ``SELECT * ... WHERE`` query blocks whose predicates contain pure
        column/literal operations.  Anything whose evaluation or name binding
        could change when a query block is removed is rejected.

        ``None`` means that no safe wrapper was recognized.
        """
        current = input_table_sql.strip()
        predicates = []
        recognized_wrapper = False

        while current.startswith("(") and current.endswith(")"):
            closing = self._find_matching_paren(current, 0)
            if closing != len(current) - 1:
                return None
            query = current[1:-1].strip()
            prefix = "SELECT * FROM "
            if not query.upper().startswith(prefix):
                return None

            source_and_clauses = query[len(prefix) :].strip()
            clause_positions = self._find_top_level_clause_positions(
                source_and_clauses,
                [
                    "WHERE",
                    "GROUP",
                    "HAVING",
                    "ORDER",
                    "LIMIT",
                    "FETCH",
                    "OFFSET",
                    "CONNECT",
                    "START",
                    "MODEL",
                    "QUALIFY",
                    "UNION",
                    "INTERSECT",
                    "MINUS",
                ],
            )
            if any(
                clause_positions[name] != -1
                for name in clause_positions
                if name != "WHERE"
            ):
                return None

            where_pos = clause_positions["WHERE"]
            if where_pos == -1:
                source = source_and_clauses
            else:
                source = source_and_clauses[:where_pos].strip()
                predicate = source_and_clauses[where_pos + len("WHERE") :].strip()
                if not predicate or not self._is_safe_wrapper_predicate(predicate):
                    return None
                # Predicates are collected inside-out below.
                predicates.insert(0, predicate)

            if not source:
                return None
            recognized_wrapper = True
            current = source

        # Requiring a plain relation name excludes aliases, joins, table
        # functions, database links, quoted identifiers, and lateral sources.
        identifier = r"[A-Z_][A-Z0-9_$#]*"
        if not re.fullmatch(
            rf"{identifier}(?:\.{identifier}){{0,2}}", current, flags=re.IGNORECASE
        ):
            return None
        return (current, predicates) if recognized_wrapper else None

    def _is_safe_wrapper_predicate(self, predicate_sql: str) -> bool:
        """Whether selection composition is safe for this predicate subset."""
        try:
            parser = ExprParser()
            parser._reset(predicate_sql)
            expr = parser._parse_or()
            if not parser._at_end():
                return False
        except (ValueError, IndexError):
            return False

        pseudo_columns = {
            "ROWNUM",
            "ROWID",
            "ORA_ROWSCN",
            "LEVEL",
            "CONNECT_BY_ISLEAF",
            "CONNECT_BY_ISCYCLE",
        }
        comparison_ops = {"=", "!=", "<>", ">", "<", ">=", "<=", "LIKE", "NOT LIKE"}

        def scalar(node):
            return isinstance(node, Literal) or (
                isinstance(node, ColumnRef)
                and node.qualifier is None
                and node.name.upper() not in pseudo_columns
                and node.name != "*"
            )

        def safe(node):
            if isinstance(node, BinaryExpr):
                if node.op in {"AND", "OR"}:
                    return safe(node.left) and safe(node.right)
                return (
                    node.op in comparison_ops
                    and scalar(node.left)
                    and scalar(node.right)
                )
            if isinstance(node, BetweenExpr):
                return scalar(node.expr) and scalar(node.lower) and scalar(node.upper)
            if isinstance(node, IsNullExpr):
                return scalar(node.expr)
            if isinstance(node, InExpr):
                return scalar(node.expr) and all(
                    isinstance(value, Literal) for value in node.values
                )
            if isinstance(node, NotExpr):
                return safe(node.expr)
            return False

        return safe(expr)

    def _collapse_one_select_star_wrapper(self, input_table_sql: str) -> Optional[str]:
        sql = input_table_sql.strip()
        if not sql.startswith("(") or not sql.endswith(")"):
            return None

        inner = sql[1:-1].strip()
        prefix = "SELECT * FROM "
        if not inner.upper().startswith(prefix):
            return None

        source_and_clauses = inner[len(prefix) :].strip()
        where_pos = self._find_top_level_keyword(source_and_clauses, "WHERE")
        if where_pos == -1:
            outer_source, outer_where = source_and_clauses, None
        else:
            outer_source = source_and_clauses[:where_pos].strip()
            outer_where = source_and_clauses[where_pos + len("WHERE") :].strip()

        if (
            not outer_source.startswith("(")
            or not outer_source.endswith(")")
            or self._has_top_level_from_clause(
                source_and_clauses, {"GROUP", "HAVING", "ORDER", "LIMIT", "UNION"}
            )
        ):
            return None

        inner_select = outer_source[1:-1].strip()
        if not inner_select.upper().startswith(
            prefix
        ) or self._has_top_level_from_clause(
            inner_select[len(prefix) :], {"GROUP", "HAVING", "ORDER", "LIMIT", "UNION"}
        ):
            return None

        if outer_where is None:
            return outer_source

        inner_source_and_clauses = inner_select[len(prefix) :].strip()
        inner_where_pos = self._find_top_level_keyword(
            inner_source_and_clauses, "WHERE"
        )
        if inner_where_pos == -1:
            return f"({inner_select} WHERE {outer_where})"

        inner_source = inner_source_and_clauses[:inner_where_pos].strip()
        inner_where = inner_source_and_clauses[inner_where_pos + len("WHERE") :].strip()
        if not inner_source or not inner_where or not outer_where:
            return None
        return (
            f"(SELECT * FROM {inner_source} WHERE ({inner_where}) AND ({outer_where}))"
        )

    def _has_top_level_from_clause(self, sql: str, keywords: Set[str]) -> bool:
        return any(
            self._find_top_level_keyword(sql, keyword) != -1 for keyword in keywords
        )

    def _normalize_sql(self, sql: str) -> str:
        result = []
        quote_char = None
        pending_space = False
        idx = 0

        while idx < len(sql):
            char = sql[idx]

            if quote_char is not None:
                result.append(char)
                if char == quote_char:
                    next_idx = idx + 1
                    if next_idx < len(sql) and sql[next_idx] == quote_char:
                        result.append(sql[next_idx])
                        idx = next_idx
                    else:
                        quote_char = None
                idx += 1
                continue

            if char in {"'", '"'}:
                if pending_space and result:
                    result.append(" ")
                pending_space = False
                quote_char = char
                result.append(char)
            elif char.isspace():
                pending_space = True
            else:
                if pending_space and result:
                    result.append(" ")
                pending_space = False
                result.append(char.upper())
            idx += 1

        return "".join(result).strip()

    def _validate_match_recognize_input_table(self) -> None:
        """Validate the outer shape of the row-pattern input relation.

        A table, view, and named query are syntactically indistinguishable here:
        each is a single (possibly qualified) relation name with an optional
        correlation name.  A derived table must be a parenthesized SELECT/WITH
        query, also with an optional correlation name.  In particular, a bare
        JOIN tree or comma join is not a valid MATCH_RECOGNIZE input relation.

        This is deliberately a structural check rather than a full SQL parser;
        the database catalog determines whether a relation name denotes a table,
        view, or named query.  The inner query remains mostly opaque, apart from
        rejecting explicit joins that obviously lack ON/USING conditions.
        """
        source = (self.input_table or "").strip()
        identifier = r"[A-Z_][A-Z0-9_$#]*"
        relation_name = rf"{identifier}(?:\.{identifier})*"
        optional_alias = rf"(?:\s+(?:AS\s+)?{identifier})?"

        if re.fullmatch(relation_name + optional_alias, source, re.IGNORECASE):
            return

        if source.startswith("("):
            closing_pos = self._find_matching_paren(source, 0)
            if closing_pos != -1:
                query = source[1:closing_pos].strip()
                suffix = source[closing_pos + 1 :].strip()
                alias_is_valid = not suffix or bool(
                    re.fullmatch(rf"(?:AS\s+)?{identifier}", suffix, re.IGNORECASE)
                )
                if (
                    query
                    and self._is_derived_query_expression(query)
                    and alias_is_valid
                ):
                    if self._has_conditionless_join(query):
                        raise ValueError(
                            "derived-table JOIN must have an ON or USING "
                            "condition unless it is CROSS or NATURAL JOIN"
                        )
                    return

        raise ValueError(
            "MATCH_RECOGNIZE input must be a table, view, named query, or "
            "parenthesized derived table; wrap joined tables in a SELECT query"
        )

    def _is_derived_query_expression(self, sql: str) -> bool:
        """Recognize the leading shape of a derived-table query expression."""
        query = sql.strip()
        if any(
            self._matches_keyword_at(query, 0, keyword)
            for keyword in ("SELECT", "WITH")
        ):
            return True

        # Rewriters may parenthesize each operand of a set query, producing an
        # input such as ``((SELECT ...) UNION ALL (SELECT ...))``.  Recursively
        # recognize those query operands without accepting a parenthesized JOIN
        # tree such as ``(T JOIN U ON ...)``.
        if not query.startswith("("):
            return False
        closing_pos = self._find_matching_paren(query, 0)
        if closing_pos == -1 or not self._is_derived_query_expression(
            query[1:closing_pos]
        ):
            return False

        suffix = query[closing_pos + 1 :].strip()
        if not suffix:
            return True
        set_operator = re.match(
            r"^(?:UNION(?:\s+(?:ALL|DISTINCT))?|INTERSECT|EXCEPT|MINUS)\b",
            suffix,
            re.IGNORECASE,
        )
        if set_operator is None:
            return False
        return self._is_derived_query_expression(suffix[set_operator.end() :])

    def _has_conditionless_join(self, sql: str) -> bool:
        """Return whether an explicit JOIN is missing its ON/USING condition."""
        tokens = []
        keywords = (
            "INTERSECT",
            "QUALIFY",
            "CONNECT",
            "EXCEPT",
            "HAVING",
            "OFFSET",
            "UNION",
            "WHERE",
            "GROUP",
            "ORDER",
            "FETCH",
            "MINUS",
            "MODEL",
            "START",
            "USING",
            "JOIN",
            "ON",
        )
        depth = 0
        quote_char = None
        idx = 0
        while idx < len(sql):
            char = sql[idx]
            if quote_char is not None:
                if char == quote_char:
                    next_idx = idx + 1
                    if next_idx < len(sql) and sql[next_idx] == quote_char:
                        idx = next_idx
                    else:
                        quote_char = None
            elif char in {"'", '"'}:
                quote_char = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            else:
                for keyword in keywords:
                    if self._matches_keyword_at(sql, idx, keyword):
                        tokens.append((keyword, idx, depth))
                        idx += len(keyword) - 1
                        break
            idx += 1

        boundaries = {
            "JOIN",
            "WHERE",
            "GROUP",
            "HAVING",
            "ORDER",
            "FETCH",
            "OFFSET",
            "CONNECT",
            "START",
            "MODEL",
            "QUALIFY",
            "UNION",
            "INTERSECT",
            "EXCEPT",
            "MINUS",
        }
        for token_index, (keyword, position, token_depth) in enumerate(tokens):
            if keyword != "JOIN":
                continue
            prefix = sql[:position]
            if re.search(
                r"(?:\bCROSS|\bNATURAL(?:\s+(?:INNER|LEFT|RIGHT|FULL)"
                r"(?:\s+OUTER)?)?)\s*$",
                prefix,
                re.IGNORECASE,
            ):
                continue

            has_condition = False
            for next_keyword, _, next_depth in tokens[token_index + 1 :]:
                if next_depth != token_depth:
                    continue
                if next_keyword in {"ON", "USING"}:
                    has_condition = True
                    break
                if next_keyword in boundaries:
                    break
            if not has_condition:
                return True
        return False

    def _is_identifier_char(self, ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def _matches_keyword_at(self, content: str, pos: int, keyword: str) -> bool:
        if not content.startswith(keyword, pos):
            return False
        before = content[pos - 1] if pos > 0 else " "
        after_pos = pos + len(keyword)
        after = content[after_pos] if after_pos < len(content) else " "
        return not self._is_identifier_char(before) and not self._is_identifier_char(
            after
        )

    def _find_matching_paren(self, content: str, l_paren_pos: int) -> int:
        depth = 0
        quote_char = None
        idx = l_paren_pos
        while idx < len(content):
            char = content[idx]
            if quote_char is not None:
                if char == quote_char:
                    next_idx = idx + 1
                    if next_idx < len(content) and content[next_idx] == quote_char:
                        idx = next_idx
                    else:
                        quote_char = None
            elif char in {"'", '"'}:
                quote_char = char
            else:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        return idx
            idx += 1
        return -1

    def _find_top_level_keyword(
        self, content: str, keyword: str, start: int = 0
    ) -> int:
        depth = 0
        quote_char = None
        idx = start
        while idx < len(content):
            char = content[idx]
            if quote_char is not None:
                if char == quote_char:
                    next_idx = idx + 1
                    if next_idx < len(content) and content[next_idx] == quote_char:
                        idx = next_idx
                    else:
                        quote_char = None
            elif char in {"'", '"'}:
                quote_char = char
            else:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif depth == 0 and self._matches_keyword_at(content, idx, keyword):
                    return idx
            idx += 1
        return -1

    def _find_top_level_clause_positions(
        self,
        content: str,
        clause_names: List[str],
        aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        positions = {clause_name: -1 for clause_name in clause_names}
        candidates = [(clause_name, clause_name) for clause_name in clause_names]
        if aliases is not None:
            candidates.extend(
                (alias, clause_name) for alias, clause_name in aliases.items()
            )
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        candidates_by_initial = {}
        for candidate in candidates:
            candidates_by_initial.setdefault(candidate[0][0], []).append(candidate)

        depth = 0
        quote_char = None
        idx = 0
        while idx < len(content):
            char = content[idx]
            if quote_char is not None:
                if char == quote_char:
                    next_idx = idx + 1
                    if next_idx < len(content) and content[next_idx] == quote_char:
                        idx = next_idx
                    else:
                        quote_char = None
            elif char in {"'", '"'}:
                quote_char = char
            else:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif depth == 0:
                    for keyword, clause_name in candidates_by_initial.get(char, ()):
                        if positions[clause_name] == -1 and self._matches_keyword_at(
                            content, idx, keyword
                        ):
                            positions[clause_name] = idx
                            break
            idx += 1
        return positions

    def _find_match_clause_positions(self, content: str) -> Dict[str, int]:
        output_mode_aliases = {
            ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS: "PER MATCH",
            ALL_ROWS_PER_MATCH_SHOW_EMPTY_MATCHES: "PER MATCH",
            ALL_ROWS_PER_MATCH_OMIT_EMPTY_MATCHES: "PER MATCH",
            ALL_ROWS_PER_MATCH: "PER MATCH",
            ONE_ROW_PER_MATCH: "PER MATCH",
        }
        positions = self._find_top_level_clause_positions(
            content, self._MATCH_CLAUSE_ORDER, output_mode_aliases
        )
        per_match_pos = positions["PER MATCH"]
        if per_match_pos != -1:
            self.rows_per_match = self._rows_per_match_mode_at(content, per_match_pos)
        return positions

    @staticmethod
    def _rows_per_match_mode_at(content: str, start_pos: int) -> str:
        modes = (
            ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS,
            ALL_ROWS_PER_MATCH_SHOW_EMPTY_MATCHES,
            ALL_ROWS_PER_MATCH_OMIT_EMPTY_MATCHES,
            ALL_ROWS_PER_MATCH,
            ONE_ROW_PER_MATCH,
        )
        for mode in modes:
            if content.startswith(mode, start_pos):
                return mode
        return ONE_ROW_PER_MATCH

    def _get_clause_end(
        self, content: str, clause_positions: Dict[str, int], clause_name: str
    ) -> int:
        start_pos = clause_positions.get(clause_name, -1)
        if start_pos == -1:
            return -1

        end_pos = len(content)
        for other_pos in clause_positions.values():
            if start_pos < other_pos < end_pos:
                end_pos = other_pos
        return end_pos

    def _get_clause_text(
        self, content: str, clause_positions: Dict[str, int], clause_name: str
    ) -> Optional[str]:
        start_pos = clause_positions.get(clause_name, -1)
        if start_pos == -1:
            return None
        end_pos = self._get_clause_end(content, clause_positions, clause_name)
        return content[start_pos + len(clause_name) : end_pos].strip()

    def _parse_partition_by(
        self, content: str, clause_positions: Dict[str, int]
    ) -> None:
        clause_text = self._get_clause_text(content, clause_positions, "PARTITION BY")
        if clause_text is not None:
            self.partition_by = [
                col.strip() for col in clause_text.split(",") if col.strip()
            ]

    def _parse_order_by(self, content: str, clause_positions: Dict[str, int]) -> None:
        clause_text = self._get_clause_text(content, clause_positions, "ORDER BY")
        if clause_text is not None:
            self.match_order_by = self._parse_order_by_columns(clause_text)

    def _parse_measures(self, content: str, clause_positions: Dict[str, int]) -> None:
        clause_text = self._get_clause_text(content, clause_positions, "MEASURES")
        if clause_text is not None:
            self.measures = self._expr_parser.parse_measures("MEASURES " + clause_text)
            self._validate_expression_pattern_variables(
                "MEASURES", self._expr_parser.get_referenced_qualifiers()
            )

    def _parse_rows_per_match(
        self, content: str, clause_positions: Dict[str, int]
    ) -> None:
        start_pos = clause_positions.get("PER MATCH", -1)
        if start_pos == -1:
            return
        self.rows_per_match = self._rows_per_match_mode_at(content, start_pos)

    def _is_all_rows_per_match(self) -> bool:
        """Whether the (possibly unset) PER MATCH mode is ALL ROWS PER MATCH."""
        return (
            self.rows_per_match is not None
            and self.rows_per_match.strip().upper().startswith(ALL_ROWS_PER_MATCH)
        )

    def _is_one_row_per_match(self) -> bool:
        """Whether the (possibly unset) PER MATCH mode is ONE ROW PER MATCH."""
        return (
            self.rows_per_match is not None
            and self.rows_per_match.strip().upper() == ONE_ROW_PER_MATCH
        )

    def _validate_measures_for_output_mode(self) -> None:
        """Validate that the selected output mode produces a valid row type.

        MEASURES is optional.  Under ONE ROW PER MATCH the output consists of
        the PARTITION BY columns followed by the measure columns, so omitting
        MEASURES is legal when at least one partition column remains.  Writing
        the MEASURES keyword without a definition is still invalid.
        """
        if self.measures is not None and not self.measures.measures:
            raise ValueError(
                "'MEASURES' subclause requires at least one measure definition"
            )

        has_partition_columns = bool(self.partition_by)
        has_measures = self.measures is not None and bool(self.measures.measures)
        if (
            self._is_one_row_per_match()
            and not has_partition_columns
            and not has_measures
        ):
            raise ValueError(
                f"{ONE_ROW_PER_MATCH} requires at least one PARTITION BY column "
                "or MEASURES definition"
            )

    def _parse_after_match(
        self, content: str, clause_positions: Dict[str, int]
    ) -> None:
        clause_text = self._get_clause_text(
            content, clause_positions, "AFTER MATCH SKIP"
        )
        if clause_text is not None:
            self.after_match_skip_to = (
                clause_text[3:].strip()
                if clause_text.startswith("TO ")
                else clause_text
            )

    def _parse_pattern(self, content: str, clause_positions: Dict[str, int]) -> None:
        clause_text = self._get_clause_text(content, clause_positions, "PATTERN")
        if clause_text is None:
            return

        l_paren_pos = clause_text.find("(")
        if l_paren_pos == -1:
            raise ValueError(" illegal 'PATTERN' subclause ")
        r_paren_pos = self._find_matching_paren(clause_text, l_paren_pos)
        if r_paren_pos == -1:
            raise ValueError(" illegal 'PATTERN' subclause ")

        pattern_text = clause_text[l_paren_pos + 1 : r_paren_pos]
        parser = PatternParser(PatternLexer(pattern_text))
        self.pattern = parser.parse()
        self.pattern_rewrite_boundaries = parser.rewrite_boundaries

    def _simplify_excluded_pattern_for_output_mode(self) -> None:
        if self.pattern is None:
            return
        if self.rows_per_match is None or self._is_one_row_per_match():
            self.pattern = unwrap_excluded_pattern(self.pattern)

    def _validate_excluded_pattern_for_output_mode(self) -> None:
        if (
            self.rows_per_match == ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS
            and self.pattern is not None
            and contains_excluded_pattern(self.pattern)
        ):
            raise ValueError(
                f"{ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS} does not allow "
                "excluded patterns ({- ... -})"
            )

    def _parse_subset(self, content: str, clause_positions: Dict[str, int]) -> None:
        clause_text = self._get_clause_text(content, clause_positions, "SUBSET")
        if clause_text is None:
            return

        self.subsets = {}
        pos = 0
        while pos < len(clause_text):
            eq_pos = clause_text.find("=", pos)
            if eq_pos == -1:
                raise ValueError("illegal 'SUBSET' subclause ")
            subset_name = clause_text[pos:eq_pos].strip().strip(",")
            l_paren_pos = clause_text.find("(", eq_pos + 1)
            if l_paren_pos == -1:
                raise ValueError("illegal 'SUBSET' subclause ")
            r_paren_pos = self._find_matching_paren(clause_text, l_paren_pos)
            if r_paren_pos == -1:
                raise ValueError("illegal 'SUBSET' subclause ")
            members = [
                member.strip()
                for member in clause_text[l_paren_pos + 1 : r_paren_pos].split(",")
                if member.strip()
            ]
            self.subsets[subset_name] = members

            pos = r_paren_pos + 1
            while pos < len(clause_text) and clause_text[pos] in {" ", ","}:
                pos += 1

    def _parse_define(self, content: str, clause_positions: Dict[str, int]) -> None:
        define_pos = clause_positions["DEFINE"]
        self.define = self._expr_parser.parse_define(content[define_pos:])
        pattern_variables = {
            name.upper() for name in get_pattern_variables(self.pattern)
        }
        unknown_variables = {
            name
            for name in self.define.definitions
            if name.upper() not in pattern_variables
        }
        if unknown_variables:
            invalid_names = ", ".join(sorted(unknown_variables))
            raise ValueError(
                f"DEFINE variable(s) not present in PATTERN: {invalid_names}"
            )
        self._validate_expression_pattern_variables(
            "DEFINE", self._expr_parser.get_referenced_qualifiers()
        )

    def _input_correlation_names(self) -> Set[str]:
        """Return names that may legally qualify columns from a simple input."""
        identifier = r"[A-Z_][A-Z0-9_$#]*"
        match = re.fullmatch(
            rf"({identifier}(?:\.{identifier})*)" rf"(?:\s+(?:AS\s+)?({identifier}))?",
            (self.input_table or "").strip(),
            re.IGNORECASE,
        )
        if match is None:
            return set()
        relation_name, alias = match.groups()
        return {(alias or relation_name.split(".")[-1]).upper()}

    def _validate_expression_pattern_variables(
        self,
        clause_name: str,
        referenced_qualifiers: Iterable[str],
    ) -> None:
        """Reject qualified references to undeclared primary/union variables."""
        declared_variables = {
            name.upper() for name in get_pattern_variables(self.pattern)
        }
        declared_variables.update(name.upper() for name in (self.subsets or {}))
        allowed_qualifiers = declared_variables | self._input_correlation_names()

        normalized_references = {
            qualifier.upper() for qualifier in referenced_qualifiers
        }
        unknown_variables = sorted(normalized_references - allowed_qualifiers)
        if unknown_variables:
            raise ValueError(
                f"{clause_name} variable(s) not present in PATTERN or SUBSET: "
                f"{', '.join(unknown_variables)}"
            )

    def _validate_define_column_metadata(self) -> None:
        if self.define is None:
            return
        for column_name in sorted(self.define.get_columns()):
            if column_name.upper() in {"TRUE", "FALSE"}:
                continue
            if self.get_column_metadata(column_name) is None:
                raise ValueError(
                    f"missing schema metadata for column '{column_name}'; "
                    "provide its definition in table_schemas"
                )

    def _parse_order_by_columns(
        self, order_by_content: str
    ) -> List[Tuple[str, SortType]]:
        order_by_items = []
        for order_by_col in order_by_content.split(","):
            order_by_col = order_by_col.strip()
            if len(order_by_col) == 0:
                continue
            blank_pos = order_by_col.find(" ")
            if blank_pos != -1:
                sort_mode = order_by_col[blank_pos + 1 :].strip()
                column_name = order_by_col[:blank_pos].strip()
                if sort_mode == "ASC":
                    order_by_items.append((column_name, SortType.ASC))
                elif sort_mode == "DESC":
                    order_by_items.append((column_name, SortType.DESC))
                else:
                    raise ValueError(" 'ORDER BY' subclause has wrong sort type")
            else:
                order_by_items.append((order_by_col, SortType.ASC))
        return order_by_items

    def _parse_trailing_subclauses(self, trailing_sql: str) -> None:
        tail = trailing_sql.strip()
        if len(tail) == 0:
            return

        clause_positions = self._find_top_level_clause_positions(
            tail, self._TRAILING_CLAUSE_ORDER
        )
        found_positions = [pos for pos in clause_positions.values() if pos != -1]
        first_clause_pos = min(found_positions) if found_positions else -1
        alias_part = tail if first_clause_pos == -1 else tail[:first_clause_pos]
        alias_part = alias_part.strip()
        if len(alias_part) > 0:
            if alias_part.startswith("AS "):
                self.alias = alias_part[3:].strip()
            elif alias_part not in self._TRAILING_CLAUSE_ORDER:
                self.alias = alias_part

        if first_clause_pos == -1:
            return

        parser = ExprParser(self.match_output_simplify_context())

        if clause_positions["WHERE"] != -1:
            start_pos = clause_positions["WHERE"]
            end_pos = self._get_clause_end(tail, clause_positions, "WHERE")
            where_sql = tail[start_pos:end_pos]
            self.where = parser.parse_where(where_sql)

        if clause_positions["GROUP BY"] != -1:
            start_pos = clause_positions["GROUP BY"]
            end_pos = self._get_clause_end(tail, clause_positions, "GROUP BY")
            group_by_sql = tail[start_pos + len("GROUP BY") : end_pos].strip()
            if len(group_by_sql) > 0:
                self.group_by = [
                    col.strip()
                    for col in group_by_sql.split(",")
                    if len(col.strip()) > 0
                ]

        if clause_positions["HAVING"] != -1:
            start_pos = clause_positions["HAVING"]
            end_pos = self._get_clause_end(tail, clause_positions, "HAVING")
            having_sql = tail[start_pos:end_pos]
            self.having = parser.parse_where(having_sql.replace("HAVING", "WHERE", 1))

        if clause_positions["ORDER BY"] != -1:
            start_pos = clause_positions["ORDER BY"]
            end_pos = self._get_clause_end(tail, clause_positions, "ORDER BY")
            order_by_sql = tail[start_pos + len("ORDER BY") : end_pos].strip()
            self.order_by = self._parse_order_by_columns(order_by_sql)

        if clause_positions["LIMIT"] != -1:
            start_pos = clause_positions["LIMIT"]
            end_pos = self._get_clause_end(tail, clause_positions, "LIMIT")
            limit_sql = tail[start_pos + len("LIMIT") : end_pos].strip()
            if len(limit_sql) > 0:
                limit_token = limit_sql.split()[0]
                self.limit = int(limit_token)

    def validate(self) -> List[str]:
        """Validate semantic constraints of the parsed AST.

        In particular, expressions outside MATCH_RECOGNIZE can only reference
        columns produced by its row pattern output table.  ONE ROW PER MATCH
        has a fully inferable output schema; ALL ROWS PER MATCH additionally
        needs source schema metadata before all references can be checked.
        """
        output_columns = self.get_match_recognize_output_columns()
        if output_columns is None:
            return []

        available = {
            self._unqualified_column_name(column)
            for column in output_columns
            if self._unqualified_column_name(column)
        }
        errors: List[str] = []

        def check_columns(clause_name: str, columns) -> None:
            missing = sorted(
                {
                    self._unqualified_column_name(column)
                    for column in columns
                    if self._unqualified_column_name(column)
                    and self._unqualified_column_name(column) != "*"
                    and self._unqualified_column_name(column) not in available
                }
            )
            if missing:
                errors.append(
                    f"{clause_name} references column(s) not produced by "
                    f"MATCH_RECOGNIZE: {', '.join(missing)}"
                )

        if self.output_names is not None and not self.is_select_star():
            check_columns("SELECT", self.output_names.get_columns())

        if self.where is not None:
            check_columns("WHERE", self.where.collect_columns())

        if self.group_by is not None:
            check_columns("GROUP BY", self.group_by)

        if self.having is not None:
            check_columns("HAVING", self.having.collect_columns())

        if self.order_by is not None:
            select_aliases = {
                alias.strip().upper()
                for _, alias in (
                    self.output_names.select_items
                    if self.output_names is not None
                    else []
                )
                if alias is not None and alias.strip()
            }
            order_columns = [
                column
                for column, _ in self.order_by
                if self._unqualified_column_name(column) not in select_aliases
            ]
            check_columns("ORDER BY", order_columns)

        return errors

    @staticmethod
    def _unqualified_column_name(column: str) -> str:
        """Return a case-insensitive output-column key for a column reference."""
        return str(column).strip().split(".")[-1].upper()

    def is_select_star(self) -> bool:
        if self.output_names is None:
            return False
        items = self.output_names.select_items
        if len(items) != 1:
            return False
        expr, _ = items[0]
        return hasattr(expr, "name") and expr.name == "*"

    def select_contains_wildcard(self) -> bool:
        """Return whether SELECT directly projects an unqualified/qualified ``*``.
        ``COUNT(*)`` is not a wildcard projection and therefore does not make
        this method true.  In contrast, both ``SELECT *`` and ``SELECT M.*``
        expose every column of the MATCH_RECOGNIZE output relation.
        """

        if self.output_names is None:
            return False
        return any(
            isinstance(expr, ColumnRef) and expr.name == "*"
            for expr, _ in self.output_names.select_items
        )

    def _clear_metadata_cache(self) -> None:
        self._normalized_schema_cache_key = None
        self._normalized_schema_cache = None
        self._join_source_columns_cache_key = None
        self._join_source_columns_cache = None

    def _normalize_schema_columns(self, table_schemas: Dict) -> Dict[str, List[str]]:
        cache_key = id(table_schemas)
        if self._normalized_schema_cache_key == cache_key:
            return self._normalized_schema_cache

        normalized = {}
        for table_name, meta in table_schemas.items():
            if isinstance(meta, dict):
                columns = meta.get("columns", [])
            else:
                columns = meta
            normalized[table_name.upper()] = self._schema_column_names(columns)
        self._normalized_schema_cache_key = cache_key
        self._normalized_schema_cache = normalized
        return normalized

    @staticmethod
    def _schema_column_names(columns) -> List[str]:
        """Return normalized names from legacy and column-object schemas."""
        if columns is None:
            return []
        if isinstance(columns, dict):
            return [str(name).strip().upper() for name in columns if str(name).strip()]

        result = []
        for column in columns:
            if isinstance(column, dict):
                name = column.get("name", column.get("column_name"))
            else:
                name = column
            if name is not None and str(name).strip():
                result.append(str(name).strip().upper())
        return result

    @staticmethod
    def _normalize_nullable(value) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"YES", "Y", "TRUE", "T", "1", "NULLABLE"}:
                return True
            if normalized in {"NO", "N", "FALSE", "F", "0", "NOT NULL", "NOT_NULL"}:
                return False
        return None

    @staticmethod
    def _normalize_column_data_type(value, table_name: str, column_name: str) -> str:
        """Validate and canonicalize one of the six supported SQL types."""
        qualified_name = f"{table_name.upper()}.{column_name.upper()}"
        if value is None or not str(value).strip():
            raise ValueError(f"column '{qualified_name}' must declare a data type")

        normalized = re.sub(r"\s+", "", str(value).upper())
        scalar_types = {
            ColumnDataType.INT.value,
            ColumnDataType.LONG.value,
            ColumnDataType.FLOAT.value,
            ColumnDataType.DOUBLE.value,
        }
        if normalized in scalar_types:
            return normalized

        sized_match = re.fullmatch(r"(CHAR|VARCHAR)\((\d+)\)", normalized)
        if sized_match is not None and int(sized_match.group(2)) > 0:
            return f"{sized_match.group(1)}({int(sized_match.group(2))})"

        supported = "INT, LONG, FLOAT, DOUBLE, CHAR(n), VARCHAR(n)"
        raise ValueError(
            f"column '{qualified_name}' has unsupported data type '{value}'; "
            f"supported types: {supported}"
        )

    def _column_metadata_from_schema(
        self,
        table_name: str,
        meta,
        column_name: str,
    ) -> Optional[ColumnMetadata]:
        if not isinstance(meta, dict):
            names = self._schema_column_names(meta)
            if column_name not in names:
                return None
            self._normalize_column_data_type(None, table_name, column_name)

        columns = meta.get("columns", [])
        names = self._schema_column_names(columns)
        column_entry = None
        if isinstance(columns, dict):
            for name, value in columns.items():
                if str(name).strip().upper() == column_name:
                    column_entry = value
                    break
        else:
            for value in columns or []:
                if not isinstance(value, dict):
                    continue
                name = value.get("name", value.get("column_name"))
                if name is not None and str(name).strip().upper() == column_name:
                    column_entry = value
                    break

        def lookup_map(keys):
            for key in keys:
                values = meta.get(key)
                if not isinstance(values, dict):
                    continue
                for name, value in values.items():
                    if str(name).strip().upper() == column_name:
                        return value
            return None

        data_type = None
        nullable = None
        if isinstance(column_entry, dict):
            data_type = column_entry.get("data_type", column_entry.get("type"))
            nullable = column_entry.get("nullable", column_entry.get("is_nullable"))
        elif isinstance(column_entry, str) and isinstance(columns, dict):
            # Compact form: {'columns': {'ID': 'LONG'}}.
            data_type = column_entry

        if data_type is None:
            data_type = lookup_map(("column_types", "data_types", "types"))
        if nullable is None:
            nullable = lookup_map(("column_nullables", "nullables", "nullable"))

        # Side maps are allowed to be the source of the column list, but a
        # wholly unknown name must not acquire fabricated default metadata.
        if column_name not in names and data_type is None and nullable is None:
            return None

        primary_keys = {
            str(name).strip().upper() for name in (meta.get("primary_key") or [])
        }
        if nullable is None:
            nullable = column_name not in primary_keys
        normalized_type = self._normalize_column_data_type(
            data_type, table_name, column_name
        )
        normalized_nullable = self._normalize_nullable(nullable)
        return ColumnMetadata(
            table_name.upper(),
            column_name,
            normalized_type,
            True if normalized_nullable is None else normalized_nullable,
        )

    def get_column_metadata(self, column_name: str) -> Optional[ColumnMetadata]:
        """Resolve source metadata for a (possibly qualified) column.

        Resolution is case-insensitive.  An ambiguous unqualified name returns
        ``None`` rather than borrowing metadata from the wrong joined table.
        Pattern-variable qualifiers (for example ``A.PRICE``) naturally fall
        back to resolving ``PRICE`` against the input schema.
        """
        if self.table_schemas is None or column_name is None:
            return None
        raw_name = str(column_name).strip()
        if not raw_name:
            return None
        parts = raw_name.split(".")
        normalized_name = parts[-1].upper()
        qualifier = parts[-2].upper() if len(parts) > 1 else None

        derived_metadata = self._get_derived_projection_metadata(normalized_name)
        if (
            qualifier is None
            and derived_metadata is not None
            and derived_metadata.data_type is not None
            and derived_metadata.nullable is not None
        ):
            return ColumnMetadata(
                "<DERIVED>",
                normalized_name,
                derived_metadata.data_type,
                derived_metadata.nullable,
            )

        if qualifier is None and self.input_table.lstrip().startswith("("):
            derived_select = re.match(
                r"^\s*\(\s*SELECT\s+(.*?)\s+FROM\s",
                self.input_table,
                re.IGNORECASE | re.DOTALL,
            )
            if derived_select is not None:
                for select_item in derived_select.group(1).split(","):
                    alias_match = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s+"
                        r"(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*",
                        select_item,
                        re.IGNORECASE,
                    )
                    if (
                        alias_match is not None
                        and alias_match.group(2).upper() == normalized_name
                        and alias_match.group(1).upper() != normalized_name
                    ):
                        return self.get_column_metadata(alias_match.group(1))

        source_tables = None
        simple_source_match = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_.]*)(?:\s+(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*)?",
            self.input_table.strip(),
            re.IGNORECASE,
        )
        if simple_source_match is not None:
            source_tables = {simple_source_match.group(1).upper()}
        elif self.input_table.lstrip().startswith("("):
            derived_sources = {
                match.upper()
                for match in re.findall(
                    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                    self.input_table,
                    re.IGNORECASE,
                )
            }
            if derived_sources:
                source_tables = derived_sources

        candidates = []
        for table_name, meta in self.table_schemas.items():
            table_key = str(table_name).strip().upper()
            normalized_source_tables = {
                value
                for source_table in (source_tables or set())
                for value in (source_table, source_table.split(".")[-1])
            }
            if source_tables is not None and table_key not in normalized_source_tables:
                continue
            if qualifier is not None and qualifier not in {
                table_key,
                table_key.split(".")[-1],
            }:
                continue
            resolved = self._column_metadata_from_schema(
                table_key,
                meta,
                normalized_name,
            )
            if resolved is not None:
                candidates.append(resolved)

        if len(candidates) == 1:
            return candidates[0]
        if qualifier is not None and len(candidates) == 0:
            # The qualifier is often a MATCH_RECOGNIZE pattern variable rather
            # than a source-table name; retry by the underlying column name.
            return self.get_column_metadata(normalized_name)
        return None

    def _get_derived_projection_metadata(
        self,
        column_name: str,
    ) -> Optional[ExpressionColumnMetadata]:
        """Infer metadata for an explicitly aliased derived-table expression.

        The input relation is stored as SQL text, so this deliberately handles
        only the SELECT-list shape understood by ``ExprParser``.  Failure is
        reported as unknown metadata and callers must remain conservative.
        """
        input_sql = self.input_table.lstrip()
        if not input_sql.startswith("("):
            return None

        derived_select = re.match(
            r"^\s*\(\s*SELECT\s+(.*?)\s+FROM\s",
            self.input_table,
            re.IGNORECASE | re.DOTALL,
        )
        if derived_select is None:
            return None

        normalized_name = self._unqualified_column_name(column_name)
        if normalized_name in self._derived_metadata_resolution_stack:
            # An expression may deliberately reuse its source name, e.g.
            # ``SELECT VAL * 2 AS VAL``.  Resolve the inner VAL against the
            # source relation instead of following the output alias forever.
            return None

        self._derived_metadata_resolution_stack.add(normalized_name)
        try:
            try:
                # Do not attach a SimplifyContext here: metadata inference may
                # be called while the main parser is active, and projection
                # parsing must neither reset it nor recursively request the
                # same alias.
                select_clause = ExprParser().parse_select(
                    "SELECT " + derived_select.group(1)
                )
            except (TypeError, ValueError):
                return None

            for expression, alias in select_clause.select_items:
                if alias is None or alias.strip().upper() != normalized_name:
                    continue
                if (
                    isinstance(expression, ColumnRef)
                    and self._unqualified_column_name(expression.name)
                    == normalized_name
                ):
                    return None
                return self._infer_expression_metadata(expression)
            return None
        finally:
            self._derived_metadata_resolution_stack.discard(normalized_name)

    def get_column_type(self, column_name: str) -> Optional[str]:
        derived = self._get_derived_projection_metadata(column_name)
        if derived is not None and derived.data_type is not None:
            return derived.data_type
        metadata = self.get_column_metadata(column_name)
        return metadata.data_type if metadata is not None else None

    def is_column_nullable(self, column_name: str) -> Optional[bool]:
        derived = self._get_derived_projection_metadata(column_name)
        if derived is not None and derived.nullable is not None:
            return derived.nullable
        metadata = self.get_column_metadata(column_name)
        return metadata.nullable if metadata is not None else None

    def _extract_join_source_columns(self) -> Optional[List[str]]:
        input_sql = self.input_table.strip()
        if len(input_sql) == 0 or input_sql.startswith("("):
            return None

        cache_key = (input_sql, id(self.table_schemas))
        if self._join_source_columns_cache_key == cache_key:
            cached = self._join_source_columns_cache
            return list(cached) if cached is not None else None

        schema_map = self._normalize_schema_columns(self.table_schemas)
        tokens = input_sql.replace(",", " , ").split()
        join_modifiers = {"INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL"}
        join_keywords = {"JOIN", ","}
        clause_keywords = {"ON", "USING"}

        table_names: List[str] = []
        idx = 0
        expect_table = True
        skip_join_clause = False
        paren_depth = 0

        while idx < len(tokens):
            token = tokens[idx]
            upper_token = token.upper()
            paren_depth += token.count("(") - token.count(")")

            if skip_join_clause:
                if paren_depth <= 0 and (
                    upper_token in join_modifiers or upper_token in join_keywords
                ):
                    skip_join_clause = False
                    if upper_token in join_keywords:
                        expect_table = True
                idx += 1
                continue

            if upper_token in clause_keywords and paren_depth <= 0:
                skip_join_clause = True
                idx += 1
                continue

            if upper_token in join_modifiers:
                idx += 1
                continue

            if upper_token in join_keywords:
                expect_table = True
                idx += 1
                continue

            if expect_table:
                table_name = token
                if table_name.startswith("("):
                    return None
                table_names.append(table_name.upper())
                expect_table = False
                idx += 1
                if idx < len(tokens) and tokens[idx].upper() == "AS":
                    idx += 2
                elif idx < len(tokens):
                    next_token = tokens[idx].upper()
                    if (
                        next_token not in join_modifiers
                        and next_token not in join_keywords
                        and next_token not in clause_keywords
                    ):
                        idx += 1
                continue

            idx += 1

        if len(table_names) == 0:
            return None

        output_columns: List[str] = []
        for table_name in table_names:
            columns = schema_map.get(table_name)
            if columns is None:
                self._join_source_columns_cache_key = cache_key
                self._join_source_columns_cache = None
                return None
            output_columns.extend(columns)
        self._join_source_columns_cache_key = cache_key
        self._join_source_columns_cache = list(output_columns)
        return output_columns

    def get_match_recognize_output_columns(self) -> Optional[List[str]]:
        all_rows = self._is_all_rows_per_match()
        # ONE ROW PER MATCH output consists only of PARTITION BY columns and
        # MEASURES aliases, so it can be inferred without the input schema.
        if all_rows and self.table_schemas is None:
            return None
        input_columns = self._extract_join_source_columns() if all_rows else []
        if all_rows and input_columns is None:
            input_columns = getattr(self, "_window_prefilter_input_columns", None)
        if all_rows and input_columns is None:
            input_columns = getattr(self, "_preserved_input_columns", None)
        if all_rows and input_columns is None:
            return None

        result: List[str] = []
        seen: Set[str] = set()

        def append_col(col_name: str):
            # A correlation name used inside MATCH_RECOGNIZE is not part of
            # the name exposed by the row-pattern output table.
            normalized = self._unqualified_column_name(col_name)
            if len(normalized) == 0 or normalized in seen:
                return
            seen.add(normalized)
            result.append(normalized)

        for col in self.partition_by or []:
            append_col(col)

        if all_rows:
            for col_name, _ in self.match_order_by or []:
                append_col(col_name)

        if self.measures is not None:
            for alias in self.measures.measures.keys():
                append_col(alias)

        if all_rows:
            for col in input_columns:
                append_col(col)

        return result

    @staticmethod
    def _merge_expression_nullability(values: List[Optional[bool]]) -> Optional[bool]:
        if any(value is True for value in values):
            return True
        if any(value is None for value in values):
            return None
        return False

    @staticmethod
    def _merge_numeric_types(
        left: Optional[str], right: Optional[str], op: str
    ) -> Optional[str]:
        numeric_rank = {"INT": 0, "LONG": 1, "FLOAT": 2, "DOUBLE": 3}
        if left not in numeric_rank or right not in numeric_rank:
            return left if left == right else None
        if op == "/":
            return "DOUBLE"
        return max((left, right), key=numeric_rank.get)

    def _infer_expression_metadata(self, expr: Expr) -> ExpressionColumnMetadata:
        """Conservatively infer the type and nullability of a MEASURES expression."""
        if isinstance(expr, SemanticsExpr):
            return self._infer_expression_metadata(expr.expr)

        if isinstance(expr, ColumnRef):
            metadata = self.get_column_metadata(expr.to_string())
            if metadata is None:
                return ExpressionColumnMetadata()
            nullable = metadata.nullable
            if expr.qualifier and self.pattern is not None:
                must_appear = {
                    name.upper() for name in self.pattern.get_must_appear_var()
                }
                if expr.qualifier.upper() not in must_appear:
                    nullable = True
            return ExpressionColumnMetadata(metadata.data_type, nullable)

        if isinstance(expr, Literal):
            if isinstance(expr.value, str):
                return ExpressionColumnMetadata(
                    f"VARCHAR({max(1, len(expr.value))})",
                    not bool(expr.value),
                )
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return ExpressionColumnMetadata("INT", False)
            return ExpressionColumnMetadata("DOUBLE", False)

        if isinstance(expr, NullLiteral):
            return ExpressionColumnMetadata(None, True)

        if isinstance(expr, BinaryExpr):
            left = self._infer_expression_metadata(expr.left)
            right = self._infer_expression_metadata(expr.right)
            nullable = self._merge_expression_nullability(
                [left.nullable, right.nullable]
            )
            if expr.op in ("+", "-", "*", "/"):
                return ExpressionColumnMetadata(
                    self._merge_numeric_types(left.data_type, right.data_type, expr.op),
                    nullable,
                )
            return ExpressionColumnMetadata(None, nullable)

        if isinstance(expr, CaseExpr):
            branches = [
                self._infer_expression_metadata(result) for _, result in expr.cases
            ]
            if expr.else_expr is None:
                branches.append(ExpressionColumnMetadata(None, True))
            else:
                branches.append(self._infer_expression_metadata(expr.else_expr))
            known_types = {
                metadata.data_type
                for metadata in branches
                if metadata.data_type is not None
            }
            data_type = known_types.pop() if len(known_types) == 1 else None
            return ExpressionColumnMetadata(
                data_type,
                self._merge_expression_nullability(
                    [metadata.nullable for metadata in branches]
                ),
            )

        if isinstance(expr, FunctionCall):
            name = expr.name.upper()
            if name == "COUNT":
                return ExpressionColumnMetadata("LONG", False)
            if name == "MATCH_NUMBER":
                return ExpressionColumnMetadata("LONG", False)
            if name == "CLASSIFIER":
                return ExpressionColumnMetadata(None, False)
            if name in {"PREV", "NEXT", "FIRST", "LAST", "SUM", "AVG", "MIN", "MAX"}:
                argument = (
                    self._infer_expression_metadata(expr.args[0])
                    if expr.args
                    else ExpressionColumnMetadata()
                )
                data_type = "DOUBLE" if name == "AVG" else argument.data_type
                # Navigation and aggregate functions can yield NULL even when
                # their source column is declared NOT NULL.
                return ExpressionColumnMetadata(data_type, True)
            return ExpressionColumnMetadata()

        if isinstance(expr, IsNullExpr):
            return ExpressionColumnMetadata(None, False)

        if isinstance(expr, (BetweenExpr, InExpr, NotExpr)):
            return ExpressionColumnMetadata(None, None)

        return ExpressionColumnMetadata()

    def get_match_recognize_output_column_metadata(
        self,
        column_name: str,
    ) -> Optional[ExpressionColumnMetadata]:
        """Resolve one column in the row type produced by MATCH_RECOGNIZE."""
        if column_name is None:
            return None
        normalized_name = self._unqualified_column_name(column_name)
        if not normalized_name:
            return None

        for partition_column in self.partition_by or []:
            if self._unqualified_column_name(partition_column) == normalized_name:
                metadata = self.get_column_metadata(partition_column)
                if metadata is None:
                    return None
                return ExpressionColumnMetadata(
                    metadata.data_type,
                    metadata.nullable,
                )

        if self._is_all_rows_per_match():
            for order_column, _ in self.match_order_by or []:
                if self._unqualified_column_name(order_column) != normalized_name:
                    continue
                metadata = self.get_column_metadata(normalized_name)
                if metadata is not None:
                    return ExpressionColumnMetadata(
                        metadata.data_type,
                        metadata.nullable,
                    )

        if self.measures is not None:
            for alias, expr in self.measures.measures.items():
                if alias.strip().upper() == normalized_name:
                    return self._infer_expression_metadata(expr)

        if self._is_all_rows_per_match():
            input_columns = self._extract_join_source_columns() or []
            if normalized_name in {
                self._unqualified_column_name(name) for name in input_columns
            }:
                metadata = self.get_column_metadata(normalized_name)
                if metadata is not None:
                    return ExpressionColumnMetadata(
                        metadata.data_type,
                        metadata.nullable,
                    )

        return None

    def expand_select_star(self) -> bool:
        if not self.is_select_star():
            return False

        output_columns = self.get_match_recognize_output_columns()
        if output_columns is None or len(output_columns) == 0:
            return False

        self.output_names = SelectClause(
            [(ColumnRef(None, col), None) for col in output_columns]
        )
        return True

    def get_required_input_table_columns(self) -> Union[str, Set[str]]:
        """
        Return the input-table columns required by this MATCH_RECOGNIZE AST.

        - ALL ROWS PER MATCH needs every input column because the output contains
          the remaining columns of the row pattern input table.
        - ONE ROW PER MATCH needs:
          1. partition columns,
          2. ordering columns,
          3. columns referenced by MEASURES,
          4. columns referenced by DEFINE.
        """
        if self._is_all_rows_per_match():
            return "*"

        columns = set()

        if self.partition_by is not None:
            for col in self.partition_by:
                col = col.strip()
                if len(col) > 0:
                    columns.add(col)

        if self.match_order_by is not None:
            for col, _ in self.match_order_by:
                col = col.strip()
                if len(col) > 0:
                    columns.add(col)

        if self.measures is not None:
            columns |= self.measures.get_columns()

        if self.define is not None:
            columns |= self.define.get_columns()

        return columns

    def get_input_table_columns_for_projection(self) -> Union[str, Set[str]]:
        """Return source columns that must survive an explicit input projection.

        This is intentionally different from
        :meth:`get_required_input_table_columns`.  The latter returns ``"*"``
        for ALL ROWS PER MATCH because, by definition, all remaining input
        columns are part of that mode's output.  A caller that deliberately
        places a narrow derived table in front of MATCH_RECOGNIZE needs the
        columns used to *evaluate* the query instead:

        * PARTITION BY and row-pattern ORDER BY columns;
        * columns referenced by MEASURES and DEFINE; and
        * for ALL ROWS PER MATCH, source columns explicitly referenced by the
         outer SELECT/WHERE/GROUP BY/HAVING/ORDER BY clauses.

        Under ALL ROWS PER MATCH, a direct wildcard projection exposes all
        remaining input columns. In that case this method returns ``"*"``:
        pruning would change the result schema and cannot be equivalent.
        """

        if self._is_all_rows_per_match() and self.select_contains_wildcard():
            return "*"

        columns: Set[str] = set()

        def add_columns(values: Iterable[str]) -> None:
            for value in values:
                normalized = self._unqualified_column_name(value)
                if normalized and normalized != "*":
                    columns.add(normalized)

        add_columns(self.partition_by or [])
        add_columns(column for column, _ in (self.match_order_by or []))

        if self.measures is not None:
            add_columns(self.measures.get_columns())
        if self.define is not None:
            add_columns(self.define.get_columns())

        if not self._is_all_rows_per_match():
            return columns

        # Only genuine source columns need to be added from the clauses after
        # MATCH_RECOGNIZE.  In particular, MEASURES aliases such as MATCH_NO
        # must not accidentally become derived-table projection items.
        source_columns = self._extract_join_source_columns()
        if source_columns is None:
            return columns
        available = {self._unqualified_column_name(column) for column in source_columns}

        outer_columns: Set[str] = set()
        if self.output_names is not None and not self.is_select_star():
            outer_columns |= self.output_names.get_columns()
        if self.where is not None:
            outer_columns |= self.where.collect_columns()
        if self.group_by is not None:
            outer_columns.update(self.group_by)
        if self.having is not None:
            outer_columns |= self.having.collect_columns()
        if self.order_by is not None:
            outer_columns.update(column for column, _ in self.order_by)

        add_columns(
            column
            for column in outer_columns
            if self._unqualified_column_name(column) in available
        )
        return columns

    def to_sql(self) -> str:
        if self.raw_sql_override is not None:
            return self.raw_sql_override
        prefix = self.cte_prefix if self.cte_prefix is not None else ""
        parts = [
            prefix,
            self.output_names.to_string(),
            "\nFROM ",
            self.input_table,
            "\nMATCH_RECOGNIZE(\n",
        ]
        if self.partition_by is not None:
            parts.extend(["    PARTITION BY ", ", ".join(self.partition_by), "\n"])

        if self.match_order_by is not None:
            order_by_strs = []
            for order_by_col, sort_type in self.match_order_by:
                if sort_type == SortType.ASC:
                    order_by_strs.append(order_by_col)
                else:
                    order_by_strs.append(order_by_col + " DESC")
            parts.extend(["    ORDER BY ", ", ".join(order_by_strs), "\n"])

        if self.measures is not None:
            parts.extend(["    ", self.measures.to_string(), "\n"])
        rows_per_match = (
            self.rows_per_match.strip()
            if self.rows_per_match is not None
            else ONE_ROW_PER_MATCH
        )
        if rows_per_match.upper() != ONE_ROW_PER_MATCH:
            parts.extend(["    ", rows_per_match, "\n"])
        skip_target = (
            self.after_match_skip_to.strip()
            if self.after_match_skip_to is not None
            else PAST_LAST_ROW
        )
        if skip_target.upper() != PAST_LAST_ROW:
            parts.extend(["    AFTER MATCH SKIP TO ", skip_target, "\n"])
        parts.extend(["    PATTERN(", self.pattern.to_string(), ")\n"])
        if self.subsets is not None and len(self.subsets) > 0:
            subset_strs = []
            for subset_name, group_list in self.subsets.items():
                subset_strs.append(subset_name + " = (" + ", ".join(group_list) + ")")
            parts.extend(["    SUBSET ", ", ".join(subset_strs), "\n"])
        if self.define is not None:
            parts.extend(["    ", self.define.to_string(), "\n"])
        parts.append(")")
        if self.alias is not None:
            parts.extend([" AS ", self.alias, " "])
        if self.where is not None:
            parts.extend(["\nWHERE ", self.where.to_string()])
        if self.group_by is not None and len(self.group_by) > 0:
            parts.extend(["\nGROUP BY ", ", ".join(self.group_by)])
        if self.having is not None:
            parts.extend(["\nHAVING ", self.having.to_string()])
        if self.order_by is not None and len(self.order_by) > 0:
            order_by_strs = []
            for order_by_col, sort_type in self.order_by:
                if sort_type == SortType.ASC:
                    order_by_strs.append(order_by_col)
                else:
                    order_by_strs.append(order_by_col + " DESC")
            parts.extend(["\nORDER BY ", ", ".join(order_by_strs)])
        if self.limit is not None:
            parts.extend(["\nLIMIT ", str(self.limit)])
        return "".join(parts)

    def to_string(self) -> str:
        return self.to_sql()


if __name__ == "__main__":
    sql = """
    SELECT * FROM NASDAQ MATCH_RECOGNIZE(
        PARTITION BY TICKER
        ORDER BY timestamp
        MEASURES FIRST(timestamp) AS start_ts, LAST(timestamp) AS end_ts
        ONE ROW PER MATCH
        PATTERN (DOWN{5,} UP{1,3} DOWN{5,} UP{1,3})
        DEFINE 
            DOWN AS open < PREV(open), UP AS open > PREV(open)
    )
    WHERE TICKER = 'GOOG'
    """

    test_sql = """
    SELECT * FROM ORDERS MATCH_RECOGNIZE(
        PARTITION BY PT
        ORDER BY TS
        MEASURES A.ID AS AID, B.ID AS BID
        PATTERN(A B{2,2} C D)
        DEFINE 
            A AS TS > 300 OR TS > 100,
            C AS C.TYPE = 'c' AND B.TYPE IS NULL,
            D AS SUM(OPEN) < 100
    )
    """

    SCHEMA_META = {
        "ORDERS": {
            "columns": ["ORDER_ID", "CUSTOMER_ID", "PT", "TS", "AMOUNT", "STATUS"],
            "column_types": {
                "ORDER_ID": "LONG",
                "CUSTOMER_ID": "LONG",
                "PT": "VARCHAR(32)",
                "TS": "LONG",
                "AMOUNT": "DOUBLE",
                "STATUS": "VARCHAR(20)",
            },
            "column_nullables": {
                "ORDER_ID": False,
                "CUSTOMER_ID": False,
                "PT": False,
                "TS": False,
                "AMOUNT": True,
                "STATUS": True,
            },
        }
    }

    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(sql, SCHEMA_META)
    ast.expand_select_star()
    print(ast.to_sql())
