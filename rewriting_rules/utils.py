import copy
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from expression import (
    BinaryExpr,
    BetweenExpr,
    CaseExpr,
    ColumnRef,
    Expr,
    FunctionCall,
    InExpr,
    IsNullExpr,
    Literal,
    NotExpr,
    SemanticsExpr,
    SubqueryExpr,
)
from expression_clause import (
    DefineSubClause,
    DependentCondition,
    IndependentCondition,
    MeasuresSubClause,
)
from expression_simplifier import (
    RangeConstraint,
    RangeDomain,
    SimplifyContext,
    _compare_literal_values,
    _range_bound_value,
    _simplify_logical_expr,
)
from clause_parser import ExprParser
from queryanalysis import RPRAbstractSyntaxTree, SortType
from patternnode import (
    VariablePattern,
    AnchorPattern,
    QuantifiedPattern,
    AlternationPattern,
    ConcatenationPattern,
    AnchorType,
    PermutationPattern,
    ExcludedPattern,
    EmptyPattern,
)
from .base import RewriteRule, RewriteEngine


def resolve_range_domain(
    ast: RPRAbstractSyntaxTree,
    column_name: str,
) -> Optional[RangeDomain]:
    """Resolve a schema-backed column to the domain required by a range."""
    data_type = ast.get_column_type(column_name)
    if data_type is None:
        return None
    return RangeDomain.from_data_type(data_type)


def choose_generated_column_name(
    ast: RPRAbstractSyntaxTree,
    base_name: str,
    additional_used: Optional[Iterable[str]] = None,
) -> str:
    """Return a helper-column name that does not collide with the schema."""
    used: Set[str] = set()
    if ast.table_schemas is not None:
        for meta in ast.table_schemas.values():
            columns = meta.get("columns", []) if isinstance(meta, dict) else meta
            if columns is not None:
                used.update(ast._schema_column_names(columns))
    if additional_used is not None:
        used.update(
            str(column).strip().split(".")[-1].upper() for column in additional_used
        )

    # Even without a complete schema, protect every input column that the
    # current MATCH_RECOGNIZE query exposes through one of its clauses.
    try:
        used.update(
            str(column).strip().split(".")[-1].upper()
            for column in ast.get_required_input_table_columns()
        )
    except (AttributeError, TypeError, ValueError):
        # Some lightweight/test ASTs do not implement the inference helper.
        pass

    base = base_name.upper().lstrip("_")
    if not base:
        base = "RPR_COL"
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def collect_used_relation_names(
    ast: RPRAbstractSyntaxTree,
    additional_used: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Collect relation identifiers visible to a generated SQL fragment.

    ``input_table`` is text rather than a relational AST, so token collection
    is deliberately conservative: treating a non-relation identifier as used
    can only cause a harmless suffix, while missing a table/CTE/alias could
    make a generated name shadow it.
    """
    used: Set[str] = set()
    for fragment in (ast.input_table or "", ast.cte_prefix or ""):
        used.update(
            token.upper() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", fragment)
        )
    used.update(str(name).strip().upper() for name in (ast.table_schemas or {}))
    if additional_used is not None:
        used.update(str(name).strip().upper() for name in additional_used)
    return used


def choose_generated_relation_name(
    ast: RPRAbstractSyntaxTree,
    base_name: str,
    additional_used: Optional[Iterable[str]] = None,
) -> str:
    """Return a case-insensitively fresh CTE or derived-table name."""
    used = collect_used_relation_names(ast, additional_used)
    base = base_name.upper().lstrip("_") or "RPR_RELATION"
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def bind_input_relation_alias(input_sql: str, alias: str) -> str:
    """Return a FROM item that safely binds *alias* to the whole input.

    Appending an alias is valid for an unaliased table or a bare derived
    query, but not for text that already carries an alias or for a JOIN tree:
    ``T X RPR_INPUT`` is invalid, while ``... JOIN U ... RPR_INPUT`` binds (or
    appears to bind) only the final relation.  Wrap those complex forms in a
    SELECT so the generated alias always denotes the complete row source.

    ``input_table`` is intentionally stored as SQL text in the AST, so this
    helper recognizes only the two shapes for which direct aliasing is
    unambiguous.  All other shapes take the conservative wrapper path.
    """
    sql = input_sql.strip()
    simple_identifier = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")'
    if re.fullmatch(rf"{simple_identifier}(?:\.{simple_identifier})*", sql):
        return f"{sql} {alias}"

    if sql.startswith("("):
        depth = 0
        quote = None
        index = 0
        while index < len(sql):
            char = sql[index]
            if quote is not None:
                if char == quote:
                    # SQL escapes a quote by doubling it.
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    if index == len(sql) - 1:
                        return f"{sql} {alias}"
                    break
                if depth < 0:
                    break
            index += 1

    return f"(SELECT * FROM {sql}) {alias}"


def resolve_input_rebind_alias(
    ast: RPRAbstractSyntaxTree,
) -> Tuple[bool, Optional[str]]:
    """Return whether input qualifiers survive wrapping and an alias to restore.

    PARTITION BY and ORDER BY may refer to the original row source through a
    correlation name, for example ``T X ... PARTITION BY X.PT``.  A rewrite
    that replaces that source with a derived table must bind the derived table
    back to ``X``.  This is safe only when the qualifier denotes the complete
    original input, not one member of a join or another complex FROM item.
    """
    qualifiers = {}
    columns = list(ast.partition_by or []) + [
        column for column, _sort_type in (ast.match_order_by or [])
    ]
    for column in columns:
        parts = str(column).strip().split(".")
        if len(parts) < 2:
            continue
        qualifier = parts[-2]
        qualifiers.setdefault(qualifier.upper(), qualifier)

    if not qualifiers:
        return True, None
    if len(qualifiers) != 1:
        return False, None

    identifier = r"[A-Za-z_][A-Za-z0-9_$]*"
    match = re.fullmatch(
        rf"\s*({identifier}(?:\.{identifier})*)"
        rf"(?:\s+(?:AS\s+)?({identifier}))?\s*",
        ast.input_table,
        re.IGNORECASE,
    )
    if match is None:
        return False, None

    relation_name, explicit_alias = match.groups()
    visible_name = explicit_alias or relation_name.split(".")[-1]
    qualifier = next(iter(qualifiers.values()))
    if visible_name.upper() != qualifier.upper():
        return False, None
    return True, qualifier


def input_may_null_extend_columns(input_table_sql: str) -> bool:
    """Whether a row source may NULL-extend a declared NOT NULL column."""
    if not input_table_sql:
        return False
    return bool(
        re.search(
            r"\b(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?\s+JOIN\b",
            input_table_sql,
            re.IGNORECASE,
        )
        # Oracle's legacy outer-join operator can also introduce NULLs.
        or re.search(r"\(\s*\+\s*\)", input_table_sql)
    )


def choose_generated_pattern_variable_name(
    ast: RPRAbstractSyntaxTree,
    additional_used: Optional[Iterable[str]] = None,
) -> str:
    """Return the first free pattern-variable name in the V1, V2, ... series."""
    used: Set[str] = set()
    if ast.pattern is not None:
        used.update(ast.pattern.get_variables())
    if ast.define is not None:
        used.update(ast.define.definitions.keys())
    if ast.subsets is not None:
        used.update(ast.subsets.keys())
    if additional_used is not None:
        used.update(additional_used)

    index = 1
    while f"V{index}" in used:
        index += 1
    return f"V{index}"


# When using a window function to prefilter, this can be optimized further with
# QUALIFY in dialects that support it (future work).
# Oracle permits MATCH_NUMBER() in DEFINE, but CLASSIFIER() is restricted to
# MEASURES. The DEFINE parser enforces that dialect boundary.


def _build_ordered_tail_select(
    input_table_sql: str, where_sql: str, order_sql: str, limit: int
) -> str:
    return (
        f"(SELECT * FROM {input_table_sql}"
        f" WHERE {where_sql} ORDER BY {order_sql} LIMIT {limit})"
    )


def _split_conjuncts(expr: Expr) -> List[Expr]:
    """Recursively flatten a top-level AND tree into individual conjuncts."""
    if isinstance(expr, BinaryExpr) and expr.op == "AND":
        return _split_conjuncts(expr.left) + _split_conjuncts(expr.right)
    return [expr]


def _join_conjuncts(conjuncts: List[Expr]) -> Optional[Expr]:
    """Rebuild an AND expression from a list of conjuncts. Returns None for empty list."""
    if not conjuncts:
        return None
    result = conjuncts[0]
    for c in conjuncts[1:]:
        result = BinaryExpr(result, "AND", c)
    return result


def _unwrap_semantics_expr(expr: Expr) -> Expr:
    """Remove transparent semantics wrappers used by comparison helpers."""
    while isinstance(expr, SemanticsExpr):
        expr = expr.expr
    return expr


def _comparison_op_to_bound(op: str, literal: Literal):
    """Translate a range comparison into (lower/upper, literal, inclusive)."""
    mapping = {
        ">": ("lower", False),
        ">=": ("lower", True),
        "<": ("upper", False),
        "<=": ("upper", True),
    }
    bound = mapping.get(op)
    return (bound[0], literal, bound[1]) if bound is not None else None


def _extract_unqualified_column_literal_range(
    expr: Expr, column_name: Optional[str] = None
):
    """Extract a simple unqualified-column/literal range comparison.

    Return ``(column, bound_kind, literal, inclusive)``.  When *column_name*
    is supplied, comparisons on other columns are rejected.
    """
    expr = _unwrap_semantics_expr(expr)
    if not isinstance(expr, BinaryExpr) or expr.op not in (">", ">=", "<", "<="):
        return None

    left = _unwrap_semantics_expr(expr.left)
    right = _unwrap_semantics_expr(expr.right)
    op = expr.op
    if (
        isinstance(left, ColumnRef)
        and left.qualifier is None
        and isinstance(right, Literal)
    ):
        column, literal = left, right
    elif (
        isinstance(right, ColumnRef)
        and right.qualifier is None
        and isinstance(left, Literal)
    ):
        column, literal = right, left
        op = {">": "<", ">=": "<=", "<": ">", "<=": ">="}[op]
    else:
        return None

    if column_name is not None and column.name != column_name:
        return None
    bound = _comparison_op_to_bound(op, literal)
    return (column.name,) + bound if bound is not None else None


def _extract_qualified_column(expr: Expr):
    """Return ``(qualifier, column)`` for a plain column expression."""
    expr = _unwrap_semantics_expr(expr)
    if isinstance(expr, ColumnRef):
        return expr.qualifier, expr.name
    return None, None


def _build_column_bound_expr(
    column_name: str,
    bound_kind: str,
    literal: Literal,
    inclusive: bool,
) -> Optional[Expr]:
    """Build the direct lower- or upper-bound predicate for a column."""
    if bound_kind not in ("lower", "upper"):
        return None
    op = (
        (">=" if inclusive else ">")
        if bound_kind == "lower"
        else ("<=" if inclusive else "<")
    )
    return BinaryExpr(ColumnRef(None, column_name), op, literal)


def _build_column_bound_complement_expr(
    column_name: str,
    bound_kind: str,
    literal: Literal,
    inclusive: bool,
) -> Optional[Expr]:
    """Build the SQL two-valued complement region of a column bound."""
    if bound_kind == "lower":
        op = "<" if inclusive else "<="
    elif bound_kind == "upper":
        op = ">" if inclusive else ">="
    else:
        return None
    return BinaryExpr(ColumnRef(None, column_name), op, literal)


def _build_boundary_tail_order_sql(
    ast: RPRAbstractSyntaxTree,
    order_column: str,
    bound_kind: str,
) -> str:
    """Order complement rows from nearest to farthest from a range boundary."""
    first_dir = "DESC" if bound_kind == "lower" else "ASC"
    first_sort_dir = "ASC" if ast.match_order_by[0][1] == SortType.ASC else "DESC"
    reverse_rest = first_dir != first_sort_dir
    parts = [f"{order_column} {first_dir}"]
    for column, sort_type in ast.match_order_by[1:]:
        column = column.strip().split(".")[-1]
        direction = "ASC" if sort_type == SortType.ASC else "DESC"
        if reverse_rest:
            direction = "DESC" if direction == "ASC" else "ASC"
        parts.append(f"{column} {direction}")
    return ", ".join(parts)


def _flatten_and_strict(expr: Expr, conjuncts: List[Expr]) -> bool:
    """Flatten an AND tree, rejecting top-level OR branches."""
    if isinstance(expr, SemanticsExpr):
        return _flatten_and_strict(expr.expr, conjuncts)
    if isinstance(expr, BinaryExpr) and expr.op == "AND":
        return _flatten_and_strict(expr.left, conjuncts) and _flatten_and_strict(
            expr.right, conjuncts
        )
    if isinstance(expr, BinaryExpr) and expr.op == "OR":
        return False
    conjuncts.append(expr)
    return True


def _extract_navigation_offset(func: FunctionCall) -> Optional[int]:
    """Return the integer PREV/NEXT offset; default offset is 1."""
    if len(func.args) < 2:
        return 1

    offset_expr = func.args[1]
    if isinstance(offset_expr, SemanticsExpr):
        offset_expr = offset_expr.expr
    if not isinstance(offset_expr, Literal) or not isinstance(
        offset_expr.value, (int, float)
    ):
        return None

    offset = int(offset_expr.value)
    if offset <= 0:
        return 0
    return offset


def _iter_measure_expressions(ast: RPRAbstractSyntaxTree) -> Iterable[Expr]:
    """Yield every expression in MEASURES (the clause is optional)."""
    if ast.measures is not None:
        yield from ast.measures.get_all_measures().values()


def _walk_expression(expr: Optional[Expr]) -> Iterable[Expr]:
    """Walk the expression node kinds accepted by the clause parser."""
    if expr is None:
        return
    yield expr
    if isinstance(expr, SemanticsExpr):
        yield from _walk_expression(expr.expr)
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            yield from _walk_expression(arg)
    elif isinstance(expr, BinaryExpr):
        yield from _walk_expression(expr.left)
        yield from _walk_expression(expr.right)
    elif isinstance(expr, BetweenExpr):
        yield from _walk_expression(expr.expr)
        yield from _walk_expression(expr.lower)
        yield from _walk_expression(expr.upper)
    elif isinstance(expr, CaseExpr):
        for condition, result in expr.cases:
            yield from _walk_expression(condition)
            yield from _walk_expression(result)
        yield from _walk_expression(expr.else_expr)
    elif isinstance(expr, InExpr):
        yield from _walk_expression(expr.expr)
        for value in expr.values:
            yield from _walk_expression(value)
    elif isinstance(expr, (IsNullExpr, NotExpr)):
        yield from _walk_expression(expr.expr)


def _has_nonzero_physical_navigation(expr: Optional[Expr]) -> bool:
    """Whether an expression uses PREV/NEXT in a row-changing way.

    Unknown offsets are treated conservatively.  Offset zero is deliberately
    ignored because the parser/simplifier treats it as the current row.
    """
    for node in _walk_expression(expr):
        if isinstance(node, FunctionCall) and node.name.upper() in ("PREV", "NEXT"):
            offset = _extract_navigation_offset(node)
            if offset is None or offset > 0:
                return True
    return False


def _expr_references_any_var(expr: Optional[Expr], var_names: Set[str]) -> bool:
    """Return True if expr contains a qualified column reference to any variable."""
    if expr is None:
        return False

    target_vars = set(var_names)
    found = False

    def visit(node: Optional[Expr]) -> None:
        nonlocal found
        if found or node is None:
            return
        if isinstance(node, SemanticsExpr):
            visit(node.expr)
            return
        if isinstance(node, ColumnRef):
            if node.qualifier and node.qualifier in target_vars:
                found = True
            return
        if isinstance(node, FunctionCall):
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, BinaryExpr):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, BetweenExpr):
            visit(node.expr)
            visit(node.lower)
            visit(node.upper)
            return
        if isinstance(node, CaseExpr):
            for cond, result in node.cases:
                visit(cond)
                visit(result)
            visit(node.else_expr)
            return
        if isinstance(node, InExpr):
            visit(node.expr)
            for value in node.values:
                visit(value)
            return
        if isinstance(node, IsNullExpr):
            visit(node.expr)
            return
        if isinstance(node, NotExpr):
            visit(node.expr)

    visit(expr)
    return found


def _strip_var_qualifier(expr: Optional[Expr], var_name: str) -> Optional[Expr]:
    """Replace VAR.col references with unqualified col throughout an expression."""
    if expr is None:
        return None

    target_var = var_name

    if isinstance(expr, SemanticsExpr):
        inner = _strip_var_qualifier(expr.expr, var_name)
        if inner is None:
            return None
        return SemanticsExpr(expr.semantics, inner)
    if isinstance(expr, ColumnRef):
        if expr.qualifier and expr.qualifier == target_var:
            return ColumnRef(None, expr.name)
        return expr
    if isinstance(expr, Literal):
        return expr
    if isinstance(expr, BinaryExpr):
        left = _strip_var_qualifier(expr.left, var_name)
        right = _strip_var_qualifier(expr.right, var_name)
        if left is None or right is None:
            return None
        return BinaryExpr(left, expr.op, right)
    if isinstance(expr, BetweenExpr):
        inner_expr = _strip_var_qualifier(expr.expr, var_name)
        lower = _strip_var_qualifier(expr.lower, var_name)
        upper = _strip_var_qualifier(expr.upper, var_name)
        if inner_expr is None or lower is None or upper is None:
            return None
        return BetweenExpr(inner_expr, lower, upper)
    if isinstance(expr, CaseExpr):
        new_cases = []
        for cond, result in expr.cases:
            new_cond = _strip_var_qualifier(cond, var_name)
            new_result = _strip_var_qualifier(result, var_name)
            if new_cond is None or new_result is None:
                return None
            new_cases.append((new_cond, new_result))
        new_else = (
            _strip_var_qualifier(expr.else_expr, var_name) if expr.else_expr else None
        )
        return CaseExpr(new_cases, new_else)
    if isinstance(expr, InExpr):
        inner_expr = _strip_var_qualifier(expr.expr, var_name)
        if inner_expr is None:
            return None
        values = []
        for value in expr.values:
            rewritten_value = _strip_var_qualifier(value, var_name)
            if rewritten_value is None:
                return None
            values.append(rewritten_value)
        return InExpr(inner_expr, values, expr.is_not)
    if isinstance(expr, IsNullExpr):
        inner = _strip_var_qualifier(expr.expr, var_name)
        if inner is None:
            return None
        return IsNullExpr(inner, expr.is_not)
    if isinstance(expr, FunctionCall):
        new_args = []
        for arg in expr.args:
            rewritten_arg = _strip_var_qualifier(arg, var_name)
            if rewritten_arg is None:
                return None
            new_args.append(rewritten_arg)
        return FunctionCall(expr.name, new_args)
    if isinstance(expr, NotExpr):
        inner = _strip_var_qualifier(expr.expr, var_name)
        if inner is None:
            return None
        return NotExpr(inner)
    return expr


def _is_partition_only(expr: Expr, partition_cols: Set[str]) -> bool:
    """Return True if every column referenced by *expr* is a PARTITION BY column.

    Subquery values (SubqueryExpr) contribute no column names, so an
    ``InExpr(col, [SubqueryExpr(...)])`` passes when *col* is a partition column.
    """
    cols = expr.collect_columns()
    if not cols:
        # Pure constants or expressions without column refs — do not push
        return False
    return all(c in partition_cols for c in cols)


def _collect_anchors(pattern) -> Set[AnchorType]:
    """Recursively collect all anchors present in a pattern tree."""
    result: Set[AnchorType] = set()
    if isinstance(pattern, AnchorPattern):
        result.add(pattern.anchor_type)
        result.update(_collect_anchors(pattern.pattern))
    elif isinstance(
        pattern, (ConcatenationPattern, AlternationPattern, PermutationPattern)
    ):
        for child in pattern.patterns:
            result.update(_collect_anchors(child))
    elif isinstance(pattern, (QuantifiedPattern, ExcludedPattern)):
        result.update(_collect_anchors(pattern.pattern))
    return result


def _is_start_anchored(pattern) -> bool:
    """Return True if pattern carries a START anchor at its structural start."""
    if isinstance(pattern, AnchorPattern):
        if pattern.anchor_type == AnchorType.START:
            return True
        return _is_start_anchored(pattern.pattern)
    if isinstance(pattern, ConcatenationPattern) and pattern.patterns:
        return _is_start_anchored(pattern.patterns[0])
    if isinstance(pattern, AlternationPattern):
        return all(_is_start_anchored(child) for child in pattern.patterns)
    return False


def _is_end_anchored(pattern) -> bool:
    """Return True if pattern carries an END anchor at its structural end."""
    if isinstance(pattern, AnchorPattern):
        if pattern.anchor_type == AnchorType.END:
            return True
        return _is_end_anchored(pattern.pattern)
    if isinstance(pattern, ConcatenationPattern) and pattern.patterns:
        return _is_end_anchored(pattern.patterns[-1])
    if isinstance(pattern, AlternationPattern):
        return all(_is_end_anchored(child) for child in pattern.patterns)
    return False


__all__ = [name for name in globals() if not name.startswith("__")]
