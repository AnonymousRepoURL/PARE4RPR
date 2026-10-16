"""SQL analysis for Ultra: nullability, predicate algebra and observations.

These analyses share one expression model. Candidate construction lives in
regex_candidates_ultra; ordered observable equivalence is checked separately
by regex_automata_ultra.

In fact, the functionality below overlaps with and is redundant to the existing
functionality; we plan to refactor the entire codebase in the future to make it
more concise.
"""

import copy
import itertools
import re
from dataclasses import dataclass, field
from decimal import Decimal
from time import monotonic

from clause_parser import ExprParser
from expression import (
    BetweenExpr,
    BinaryExpr,
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
from patternnode import (
    AlternationPattern,
    ConcatenationPattern,
    EmptyPattern,
    QuantifiedPattern,
    VariablePattern,
    AnchorPattern,
    ExcludedPattern,
)

from .regex_automata_ultra import BudgetUltra, UnsupportedUltra, names, nullable

# Source and derived-column nullability
#
# Conservative source/derived-column non-null inference used only by Ultra.
#
# Each SELECT introduces a fresh scope: aliases cannot accidentally resolve to
# similarly named columns in an underlying table. Unhandled joins, set queries,
# window expressions and opaque relations yield no non-null facts. In particular
# we never borrow NOT NULL from the nullable side of an outer join.

STRICT_SCALARS = frozenset({"ABS", "CEIL", "CEILING", "FLOOR", "ROUND", "TRUNC"})
SCALARS = STRICT_SCALARS | {"COALESCE", "NVL", "NULLIF"}


def top_level_words(sql):
    """Positions of unquoted identifiers outside parentheses."""
    depth, quote, i = 0, None, 0
    while i < len(sql):
        char = sql[i]
        if quote:
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and (char.isalpha() or char == "_"):
            end = i + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            yield sql[i:end].upper(), i, end
            i = end
            continue
        i += 1


def close_paren(sql):
    depth, quote, i = 0, None, 0
    while i < len(sql):
        char = sql[i]
        if quote:
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def scalar_nonnull(expr, resolve):
    if isinstance(expr, NullLiteral):
        return False
    if isinstance(expr, Literal):
        return expr.value is not None and expr.value != ""
    if isinstance(expr, ColumnRef):
        return bool(resolve(expr))
    if isinstance(expr, SemanticsExpr):
        return scalar_nonnull(expr.expr, resolve)
    if isinstance(expr, IsNullExpr):
        return True
    if isinstance(expr, NotExpr):
        return scalar_nonnull(expr.expr, resolve)
    if isinstance(expr, BinaryExpr):
        return (
            expr.op.upper()
            in {"+", "-", "*", "/", "=", "!=", "<>", "<", "<=", ">", ">=", "AND", "OR"}
            and scalar_nonnull(expr.left, resolve)
            and scalar_nonnull(expr.right, resolve)
        )
    if isinstance(expr, FunctionCall):
        name = expr.name.upper()
        if name in STRICT_SCALARS:
            return bool(expr.args) and all(
                scalar_nonnull(arg, resolve) for arg in expr.args
            )
        if name in {"COALESCE", "NVL"}:
            return any(scalar_nonnull(arg, resolve) for arg in expr.args)
        return False
    if isinstance(expr, CaseExpr):
        return (
            expr.else_expr is not None
            and scalar_nonnull(expr.else_expr, resolve)
            and all(scalar_nonnull(result, resolve) for _, result in expr.cases)
        )
    return False


def required_nonnull(expr):
    """Columns necessarily non-null when a WHERE condition is TRUE."""
    if isinstance(expr, BinaryExpr):
        if expr.op == "AND":
            return required_nonnull(expr.left) | required_nonnull(expr.right)
        if expr.op == "OR":
            return required_nonnull(expr.left) & required_nonnull(expr.right)
        if expr.op in {"=", "!=", "<>", "<", "<=", ">", ">="}:
            return strict_columns(expr.left) | strict_columns(expr.right)
    if isinstance(expr, IsNullExpr) and expr.is_not:
        return strict_columns(expr.expr)
    return set()


def strict_columns(expr):
    if isinstance(expr, ColumnRef):
        return {expr.name.upper()}
    if isinstance(expr, BinaryExpr) and expr.op in {"+", "-", "*", "/"}:
        return strict_columns(expr.left) | strict_columns(expr.right)
    if isinstance(expr, FunctionCall) and expr.name.upper() in STRICT_SCALARS:
        return set().union(*(strict_columns(arg) for arg in expr.args))
    return set()


class NullabilityUltra:
    def __init__(self, ast):
        self.ast = ast
        self.columns = self.relation(ast.input_table, 0)

    def relation(self, sql, depth):
        if depth > 16:
            return {}
        sql = sql.strip()
        if sql.startswith("("):
            end = close_paren(sql)
            if end < 0 or not re.fullmatch(
                r"\s*(?:(?:AS\s+)?[A-Z_]\w*)?\s*", sql[end + 1 :], re.I
            ):
                return {}
            return self.select(sql[1:end], depth + 1)
        match = re.fullmatch(
            r"([A-Z_]\w*(?:\.[A-Z_]\w*)?)(?:\s+(?:AS\s+)?[A-Z_]\w*)?", sql, re.I
        )
        if not match:
            return {}
        table = match.group(1).upper()
        matching = [
            (name, meta)
            for name, meta in self.ast.table_schemas.items()
            if name.upper() == table or name.upper() == table.split(".")[-1]
        ]
        if len(matching) != 1:
            return {}
        name, meta = matching[0]
        columns = self.ast._schema_column_names(
            meta.get("columns", {}) if isinstance(meta, dict) else meta
        )
        result = {}
        for column in columns:
            metadata = self.ast._column_metadata_from_schema(name, meta, column)
            result[column.upper()] = metadata is not None and metadata.nullable is False
        return result

    def select(self, sql, depth):
        words = list(top_level_words(sql))
        if not words or words[0][0] != "SELECT":
            return {}
        # These relations can introduce NULL or require a different scope.
        if any(
            word
            in {
                "JOIN",
                "UNION",
                "INTERSECT",
                "EXCEPT",
                "MINUS",
                "GROUP",
                "HAVING",
                "WINDOW",
            }
            for word, _, _ in words
        ):
            return {}
        froms = [(start, end) for word, start, end in words if word == "FROM"]
        if len(froms) != 1:
            return {}
        start, end = froms[0]
        clauses = [
            (word, pos)
            for word, pos, _ in words
            if pos > end and word in {"WHERE", "ORDER", "FETCH", "LIMIT", "OFFSET"}
        ]
        source_end = min((pos for _, pos in clauses), default=len(sql))
        source = self.relation(sql[end:source_end], depth)
        if not source:
            return {}
        for word, pos in clauses:
            if word == "WHERE":
                where_end = min(
                    (other for _, other in clauses if other > pos), default=len(sql)
                )
                try:
                    condition = ExprParser().parse_expression(sql[pos + 5 : where_end])
                    for column in required_nonnull(condition):
                        if column in source:
                            source[column] = True
                except (ValueError, TypeError):
                    pass
        projection = sql[words[0][2] : start].strip()
        projection = re.sub(r"^DISTINCT\s+", "", projection, flags=re.I)
        try:
            items = ExprParser().parse_select("SELECT " + projection).select_items
        except (ValueError, TypeError):
            return {}
        result, ambiguous = {}, set()
        for expr, alias in items:
            if isinstance(expr, ColumnRef) and expr.name == "*":
                additions = source
            else:
                name = alias or (expr.name if isinstance(expr, ColumnRef) else None)
                if name is None:
                    continue
                additions = {
                    name.upper(): scalar_nonnull(
                        expr, lambda col: source.get(col.name.upper(), False)
                    )
                }
            for name, nonnull in additions.items():
                if name in result:
                    ambiguous.add(name)
                result[name] = nonnull
        for name in ambiguous:
            result[name] = False
        return result

    def nonnull(self, expression, present=()):
        present = set(present)
        return scalar_nonnull(
            expression,
            lambda col: (
                (col.qualifier is None or col.qualifier in present)
                and self.columns.get(col.name.upper(), False)
            ),
        )


# Three-valued predicate algebra and fixed-prefix dependencies
#
# SQL three-valued guards, residual predicates and frozen-prefix analysis.
#
# Numeric comparisons of one scalar with constants share ordered intervals;
# other comparisons share an axis for equal operand pairs. Different pairs are
# independent (a deliberate over-approximation, with no guessed SMT facts).
# NULL is a separate outcome unless non-nullness is proved. Exhausting
# this bounded abstract domain proves implication/disjointness; hitting the
# bound aborts. The outside-all-guards cell is retained for endpoint proofs.

COMPARISONS = {"=", "!=", "<>", "<", "<=", ">", ">="}
NUMERIC_TYPES = (int, float, Decimal)
REVERSE = {"=": "=", "!=": "!=", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}
COMPLEMENT = {
    "=": "<>",
    "!=": "=",
    "<>": "=",
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
}
TRUE = lambda: BinaryExpr(Literal(1), "=", Literal(1))
PREFIX_FUNCTIONS = frozenset({"SUM", "AVG", "MIN", "MAX", "COUNT", "FIRST", "LAST"})


def expr_children(expr):
    if isinstance(expr, BinaryExpr):
        return [expr.left, expr.right]
    if isinstance(expr, (NotExpr, IsNullExpr, SemanticsExpr)):
        return [expr.expr]
    if isinstance(expr, FunctionCall):
        return expr.args
    if isinstance(expr, BetweenExpr):
        return [expr.expr, expr.lower, expr.upper]
    if isinstance(expr, InExpr):
        return [expr.expr, *expr.values]
    if isinstance(expr, CaseExpr):
        return [item for pair in expr.cases for item in pair] + (
            [expr.else_expr] if expr.else_expr is not None else []
        )
    if isinstance(expr, (Literal, NullLiteral, ColumnRef)):
        return []
    raise UnsupportedUltra(type(expr).__name__)


def expr_walk(expr):
    yield expr
    for child in expr_children(expr):
        yield from expr_walk(child)


def normalize(expression, owner, correlations=()):
    result = copy.deepcopy(expression)

    def visit(part, history=False):
        if isinstance(part, SemanticsExpr) and part.semantics == "FINAL":
            raise UnsupportedUltra("FINAL is not a running DEFINE expression")
        if isinstance(part, ColumnRef):
            if history:
                if history == "universal" and part.qualifier in correlations:
                    part.qualifier = None
                elif history == "group" and (
                    part.qualifier is None or part.qualifier in correlations
                ):
                    raise UnsupportedUltra(
                        "history function without an explicit pattern group"
                    )
            elif part.qualifier == owner or part.qualifier in correlations:
                part.qualifier = None
        if isinstance(part, FunctionCall):
            name = part.name.upper()
            if name not in SCALARS | PREFIX_FUNCTIONS:
                raise UnsupportedUltra(
                    "physical navigation or unknown DEFINE function: " + part.name
                )
            if name in PREFIX_FUNCTIONS:
                if history or not part.args:
                    raise UnsupportedUltra("nested history function")
                qualifiers = {
                    node.qualifier if node.qualifier not in correlations else None
                    for node in expr_walk(part.args[0])
                    if isinstance(node, ColumnRef)
                }
                if len(qualifiers) > 1:
                    raise UnsupportedUltra("mixed history groups")
                if name in {"FIRST", "LAST"}:
                    if len(part.args) > 2 or (
                        len(part.args) == 2
                        and (
                            not isinstance(part.args[1], Literal)
                            or type(part.args[1].value) is not int
                            or part.args[1].value < 0
                        )
                    ):
                        raise UnsupportedUltra(
                            "navigation requires a nonnegative constant offset"
                        )
                elif len(part.args) != 1:
                    raise UnsupportedUltra("aggregate arity")
                # Unqualified arguments refer to the universal group. Its
                # running value depends on input position and match start,
                # not classification, so all threads at that row share it.
                history = (
                    "group" if qualifiers and None not in qualifiers else "universal"
                )
        for child in expr_children(part):
            visit(child, history)

    visit(result)
    return result


def references(expression):
    return {
        part.qualifier
        for part in expr_walk(expression)
        if isinstance(part, ColumnRef) and part.qualifier is not None
    }


def first_names(pattern):
    if isinstance(pattern, VariablePattern):
        return {pattern.name}
    if isinstance(pattern, EmptyPattern):
        return set()
    if isinstance(pattern, (QuantifiedPattern, AnchorPattern, ExcludedPattern)):
        return first_names(pattern.pattern)
    result = set()
    for child in pattern.patterns:
        result.update(first_names(child))
        if isinstance(pattern, ConcatenationPattern) and not nullable(child):
            break
    return result


def minimum_members(pattern, members):
    if isinstance(pattern, VariablePattern):
        return int(pattern.name in members)
    if isinstance(pattern, EmptyPattern):
        return 0
    if isinstance(pattern, QuantifiedPattern):
        return pattern.atLeast * minimum_members(pattern.pattern, members)
    if isinstance(pattern, (AnchorPattern, ExcludedPattern)):
        return minimum_members(pattern.pattern, members)
    values = [minimum_members(child, members) for child in pattern.patterns]
    return (
        min(values, default=0)
        if isinstance(pattern, AlternationPattern)
        else sum(values)
    )


@dataclass(frozen=True)
class PrefixBindingUltra:
    signature: tuple
    members: frozenset
    minimum: int
    fixed: bool = field(compare=False)
    prefix_names: frozenset = field(compare=False)
    following_names: frozenset = field(compare=False)


def frozen_prefix(pattern, expressions, subsets=None):
    """Find prefix-group bindings; symbolic boundary proofs follow separately.

    A reference may reappear AFTER its last dependent consumer. This admits
    A ... A delimiters without wrongly declaring the second A a moving input
    to B/C. A variable-length prefix additionally needs a unique cut and unique
    group membership, proved by PredicateAlgebraUltra.prove_bindings.
    """
    parts = pattern.patterns if isinstance(pattern, ConcatenationPattern) else [pattern]
    dependencies = set().union(*(references(expr) for expr in expressions.values()))
    result = {}
    for reference in dependencies:
        members = frozenset((subsets or {}).get(reference, [reference]))
        consumers = {
            name for name, expr in expressions.items() if reference in references(expr)
        }
        locations = [
            index
            for index, part in enumerate(parts)
            if consumers.intersection(names(part))
        ]
        if not locations:
            raise UnsupportedUltra("missing dependent consumer")
        first, last = min(locations), max(locations)
        if any(members.intersection(names(part)) for part in parts[first : last + 1]):
            raise UnsupportedUltra("dependency changes within consuming region")
        prefix = parts[:first]
        positions = [
            index
            for index, part in enumerate(prefix)
            if members.intersection(names(part))
        ]
        minimum = sum(minimum_members(part, members) for part in prefix)
        if not positions or minimum == 0:
            raise UnsupportedUltra("dependency must have a guaranteed prefix group")
        result[reference] = PrefixBindingUltra(
            tuple(part.to_string() for part in prefix[: positions[-1] + 1]),
            members,
            minimum,
            all(isinstance(part, VariablePattern) for part in prefix),
            frozenset().union(*(names(part) for part in prefix)),
            frozenset(first_names(ConcatenationPattern(parts[first:]))),
        )
    return result


def sql_and(left, right):
    if left is False or right is False:
        return False
    return None if left is None or right is None else True


def sql_or(left, right):
    if left is True or right is True:
        return True
    return None if left is None or right is None else False


def compare(outcome, op):
    if outcome is None:
        return None
    return {
        "=": outcome == 0,
        "!=": outcome != 0,
        "<>": outcome != 0,
        "<": outcome < 0,
        "<=": outcome <= 0,
        ">": outcome > 0,
        ">=": outcome >= 0,
    }[op]


def conjuncts(expr):
    if isinstance(expr, BinaryExpr) and expr.op == "AND":
        return conjuncts(expr.left) + conjuncts(expr.right)
    return [expr]


def conjunction(parts):
    if not parts:
        return TRUE()
    result = copy.deepcopy(parts[0])
    for part in parts[1:]:
        result = BinaryExpr(result, "AND", copy.deepcopy(part))
    return result


class PredicateAlgebraUltra:
    def __init__(
        self,
        ast,
        expressions,
        *,
        max_cells=10000,
        deadline=float("inf"),
        allow_context=False
    ):
        self.deadline = deadline
        self.nullability = NullabilityUltra(ast)
        self.subsets = ast.subsets or {}
        self.present = frozen_prefix(ast.pattern, expressions, self.subsets)
        self.axes, self.domains = {}, []
        self.expressions = expressions
        self.allow_context = allow_context
        self.needs_prefix_context = False
        # Comparisons of the same scalar with numeric constants share one
        # ordered domain. Independent (=1)/(=2) axes would invent impossible
        # intersections and hide precisely the regions needed by the reducer.
        self.constants = {}
        for expression in expressions.values():
            for part in expr_walk(expression):
                if isinstance(part, InExpr):
                    pairs = [(part.expr, item) for item in part.values]
                elif isinstance(part, BetweenExpr):
                    pairs = [(part.expr, part.lower), (part.expr, part.upper)]
                elif isinstance(part, BinaryExpr) and part.op in COMPARISONS:
                    pairs = [(part.left, part.right), (part.right, part.left)]
                else:
                    continue
                for term, constant in pairs:
                    if (
                        not isinstance(term, (Literal, NullLiteral, CaseExpr))
                        and isinstance(constant, Literal)
                        and type(constant.value) in NUMERIC_TYPES
                    ):
                        self.constants.setdefault(term.to_string(), set()).add(
                            constant.value
                        )
        self.constants = {
            key: tuple(sorted(values)) for key, values in self.constants.items()
        }
        self.compiled = {name: self.compile(expr) for name, expr in expressions.items()}
        size = 1
        for domain in self.domains:
            size *= len(domain)
        if size > max_cells:
            raise BudgetUltra("predicate cell limit")
        self.valuations = []
        for valuation in itertools.product(*self.domains):
            if monotonic() > deadline:
                raise BudgetUltra("predicate timeout")
            self.valuations.append(valuation)
        self.sealed = True
        self.truth = {
            name: frozenset(
                index
                for index, row in enumerate(self.valuations)
                if self.evaluate(code, row) is True
            )
            for name, code in self.compiled.items()
        }
        # A compiled predicate is meaningful only in this sealed valuation
        # domain. Cache complete TRUE sets, never substitute them for Boolean
        # subexpressions (FALSE and UNKNOWN differ under NOT/CASE).
        self._truth_cache = {
            self.compiled[name]: truth for name, truth in self.truth.items()
        }
        self.prove_bindings(self.present, self.truth)

    def prove_bindings(self, bindings, truths):
        for binding in bindings.values():
            if binding.fixed:
                continue
            # Disjoint prefix labels make capture sets unambiguous at a fixed
            # cut. Disjoint prefix/continuation labels make that cut unique.
            pairs = itertools.chain(
                itertools.combinations(binding.prefix_names, 2),
                itertools.product(binding.prefix_names, binding.following_names),
            )
            if any(truths[left].intersection(truths[right]) for left, right in pairs):
                if not self.allow_context:
                    raise UnsupportedUltra("prefix capture or boundary is ambiguous")
                self.needs_prefix_context = True

    def bindings(self, pattern, guards, *, source=False):
        bindings = frozen_prefix(pattern, guards, self.subsets)
        truths = (
            self.truth
            if source
            else {name: self.truths(expr) for name, expr in guards.items()}
        )
        self.prove_bindings(bindings, truths)
        return bindings

    def nonnull(self, expression):
        """Navigation offsets and possibly empty groups must not borrow NOT NULL."""
        if (
            isinstance(expression, FunctionCall)
            and expression.name.upper() in PREFIX_FUNCTIONS
        ):
            name = expression.name.upper()
            refs = references(expression.args[0])
            if not refs:
                if name == "COUNT":
                    return True
                offset = expression.args[1].value if len(expression.args) == 2 else 0
                return offset == 0 and self.nonnull(expression.args[0])
            if len(refs) != 1 or not refs.issubset(self.present):
                return False
            binding = self.present[next(iter(refs))]
            if name == "COUNT":
                return True
            offset = expression.args[1].value if len(expression.args) == 2 else 0
            return binding.minimum > offset and self.nonnull(expression.args[0])
        if isinstance(expression, BinaryExpr):
            return self.nonnull(expression.left) and self.nonnull(expression.right)
        if isinstance(expression, (NotExpr, SemanticsExpr)):
            return self.nonnull(expression.expr)
        if isinstance(expression, CaseExpr):
            return (
                expression.else_expr is not None
                and self.nonnull(expression.else_expr)
                and all(self.nonnull(result) for _, result in expression.cases)
            )
        if isinstance(expression, FunctionCall):
            # Evaluate scalar null propagation after replacing nested prefix
            # functions by literals of their proved nullability.
            folded = copy.deepcopy(expression)

            def fold(node):
                if (
                    isinstance(node, FunctionCall)
                    and node.name.upper() in PREFIX_FUNCTIONS
                ):
                    return Literal(1) if self.nonnull(node) else NullLiteral()
                if isinstance(node, FunctionCall):
                    node.args = [fold(arg) for arg in node.args]
                elif isinstance(node, BinaryExpr):
                    node.left, node.right = fold(node.left), fold(node.right)
                return node

            return self.nullability.nonnull(fold(folded), self.present)
        return self.nullability.nonnull(expression, self.present)

    def axis(self, key, domain):
        if key not in self.axes:
            if getattr(self, "sealed", False):
                raise UnsupportedUltra("candidate introduced an unmodeled predicate")
            self.axes[key] = len(self.domains)
            self.domains.append(domain)
        return self.axes[key]

    def compile(self, expr):
        if isinstance(expr, SemanticsExpr):
            return self.compile(expr.expr)
        if isinstance(expr, NotExpr):
            return "not", self.compile(expr.expr)
        if isinstance(expr, CaseExpr):
            return (
                "case",
                tuple(
                    (self.compile(condition), self.compile(result))
                    for condition, result in expr.cases
                ),
                (
                    self.compile(expr.else_expr)
                    if expr.else_expr is not None
                    else ("constant", None)
                ),
            )
        if isinstance(expr, IsNullExpr):
            key = ("null", expr.expr.to_string())
            domain = (False,) if self.nonnull(expr.expr) else (False, True)
            return "null", self.axis(key, domain), expr.is_not
        if isinstance(expr, BetweenExpr):
            return self.compile(
                BinaryExpr(
                    BinaryExpr(expr.expr, ">=", expr.lower),
                    "AND",
                    BinaryExpr(expr.expr, "<=", expr.upper),
                )
            )
        if isinstance(expr, InExpr):
            parts = [BinaryExpr(expr.expr, "=", value) for value in expr.values]
            if not parts:
                raise UnsupportedUltra("empty IN")
            result = parts[0]
            for part in parts[1:]:
                result = BinaryExpr(result, "OR", part)
            return self.compile(NotExpr(result) if expr.is_not else result)
        if isinstance(expr, BinaryExpr):
            op = expr.op.upper()
            if op in {"AND", "OR"}:
                return op, self.compile(expr.left), self.compile(expr.right)
            if op in COMPARISONS:
                left, right = expr.left, expr.right
                if isinstance(left, CaseExpr):
                    return (
                        "case",
                        tuple(
                            (
                                self.compile(cond),
                                self.compile(BinaryExpr(value, op, right)),
                            )
                            for cond, value in left.cases
                        ),
                        self.compile(
                            BinaryExpr(
                                (
                                    left.else_expr
                                    if left.else_expr is not None
                                    else NullLiteral()
                                ),
                                op,
                                right,
                            )
                        ),
                    )
                if isinstance(right, CaseExpr):
                    return self.compile(BinaryExpr(right, REVERSE[op], left))
                if isinstance(left, NullLiteral) or isinstance(right, NullLiteral):
                    return "constant", None
                if isinstance(left, Literal) and isinstance(right, Literal):
                    # Numeric literal comparisons do not depend on SQL collation.
                    if not isinstance(left.value, str) and not isinstance(
                        right.value, str
                    ):
                        sign = (left.value > right.value) - (left.value < right.value)
                        return "constant", compare(sign, op)
                if isinstance(left, Literal) and not isinstance(right, Literal):
                    left, right, op = right, left, REVERSE[op]
                if (
                    isinstance(right, Literal)
                    and type(right.value) in NUMERIC_TYPES
                    and left.to_string() in self.constants
                ):
                    constants = self.constants[left.to_string()]
                    # Even positions denote open intervals; odd positions
                    # denote equality to a constant. No floating midpoint is
                    # needed (adjacent floats and large integers are safe).
                    domain = tuple(
                        tuple(
                            (position > 2 * index + 1) - (position < 2 * index + 1)
                            for index in range(len(constants))
                        )
                        for position in range(2 * len(constants) + 1)
                    )
                    if not self.nonnull(left):
                        domain += (None,)
                    return (
                        "literal_cmp",
                        self.axis(("numeric", left.to_string()), domain),
                        constants.index(right.value),
                        op,
                    )
                a, b = left.to_string(), right.to_string()
                if a > b:
                    a, b, op = b, a, REVERSE[op]
                nonnull = self.nonnull(left) and self.nonnull(right)
                domain = (-1, 0, 1) if nonnull else (-1, 0, 1, None)
                return "cmp", self.axis(("compare", a, b), domain), op
        raise UnsupportedUltra("unsupported Boolean guard: " + expr.to_string())

    def evaluate(self, code, row):
        op = code[0]
        if op == "constant":
            return code[1]
        if op == "cmp":
            return compare(row[code[1]], code[2])
        if op == "literal_cmp":
            value = row[code[1]]
            return compare(None if value is None else value[code[2]], code[3])
        if op == "null":
            result = row[code[1]]
            return not result if code[2] else result
        if op == "not":
            result = self.evaluate(code[1], row)
            return None if result is None else not result
        if op == "AND":
            return sql_and(self.evaluate(code[1], row), self.evaluate(code[2], row))
        if op == "OR":
            return sql_or(self.evaluate(code[1], row), self.evaluate(code[2], row))
        if op == "case":
            for condition, value in code[1]:
                if self.evaluate(condition, row) is True:
                    return self.evaluate(value, row)
            return self.evaluate(code[2], row)
        raise AssertionError(op)

    def truths(self, expression):
        if monotonic() > self.deadline:
            raise BudgetUltra("predicate timeout")
        code = self.compile(expression)
        if code not in self._truth_cache:
            self._truth_cache[code] = frozenset(
                index
                for index, row in enumerate(self.valuations)
                if self.evaluate(code, row) is True
            )
        return self._truth_cache[code]

    def simplify_guard(self, expression):
        """Minimize a complete DEFINE guard using certified TRUE sets.

        FALSE and UNKNOWN both reject a row in DEFINE. This equivalence must
        not be used to replace a subexpression under NOT or CASE. Instead,
        every proposed replacement denotes exactly the complete guard's TRUE
        set, including all nullable cells in the abstract domain.

        The basis consists of existing Boolean expressions and comparison
        complements. Numeric interval axes prove redundant bounds; string
        comparisons retain their independent, collation-agnostic axes.
        """

        def cost(expr):
            return sum(1 for _ in expr_walk(expr)), len(expr.to_string())

        def remember(pool, expr):
            truth = self.truths(expr)
            old = pool.get(truth)
            if old is None or cost(expr) < cost(old):
                pool[truth] = expr

        if not hasattr(self, "_guard_basis"):
            basis = {}
            remember(basis, TRUE())
            remember(basis, BinaryExpr(Literal(1), "=", Literal(0)))
            for source in self.expressions.values():
                for part in expr_walk(source):
                    if isinstance(part, (NotExpr, IsNullExpr, InExpr, BetweenExpr)) or (
                        isinstance(part, BinaryExpr)
                        and part.op.upper() in COMPARISONS | {"AND", "OR"}
                    ):
                        remember(basis, part)
                    if isinstance(part, BinaryExpr) and part.op.upper() in COMPARISONS:
                        left, right, op = part.left, part.right, part.op.upper()
                        if isinstance(left, Literal) and not isinstance(right, Literal):
                            left, right, op = right, left, REVERSE[op]
                        remember(basis, BinaryExpr(left, op, right))
                        remember(basis, BinaryExpr(left, COMPLEMENT[op], right))
            self._guard_basis, self._guard_simplifications = basis, {}
        key = expression.to_string()
        if key in self._guard_simplifications:
            return copy.deepcopy(self._guard_simplifications[key])
        target = self.truths(expression)
        best = expression
        exact = self._guard_basis.get(target)
        if exact is not None and cost(exact) < cost(best):
            best = exact
        if cost(best)[0] > 3:
            # Find a short conjunction whose intersection is exactly target.
            # Original conjuncts keep compound residuals (e.g. NOT TRUE of
            # a nullable prefix dependency) available without expanding them.
            pool = dict(self._guard_basis)
            for part in conjuncts(expression):
                remember(pool, part)
            remaining = frozenset(range(len(self.valuations)))
            choices = [
                (truth, part, cost(part))
                for truth, part in pool.items()
                if target <= truth
            ]
            selected = []
            while remaining != target:
                if monotonic() > self.deadline:
                    raise BudgetUltra("predicate simplification timeout")
                useful = [item for item in choices if remaining - item[0]]
                if not useful:
                    break
                truth, part, _ = min(
                    useful,
                    key=lambda item: (-len(remaining - item[0]) / item[2][0], item[2]),
                )
                remaining &= truth
                selected.append(part)
            if remaining == target:
                proposed = conjunction(selected)
                if cost(proposed) < cost(best) and self.truths(proposed) == target:
                    best = proposed
        self._guard_simplifications[key] = copy.deepcopy(best)
        return copy.deepcopy(best)

    def residuals(self, order):
        """Disjointize a priority order, factoring guards by proved implication.

        If Q implies one conjunct of P, Q AND P-not-true only needs to subtract
        the remaining conjuncts. Plain NOT is used ONLY for two-valued guards;
        otherwise CASE implements SQL's 'is not TRUE', including UNKNOWN.
        """
        result = {}
        for index, name in enumerate(order):
            base = self.expressions[name]
            exclusions = []
            for earlier in order[:index]:
                if not self.truth[name].intersection(self.truth[earlier]):
                    continue
                remaining = [
                    part
                    for part in conjuncts(self.expressions[earlier])
                    if not self.truth[name].issubset(self.truths(part))
                ]
                excluded = conjunction(remaining)
                if self.nonnull(excluded):
                    exclusions.append(NotExpr(excluded))
                else:
                    exclusions.append(
                        BinaryExpr(
                            CaseExpr([(excluded, Literal(0))], Literal(1)),
                            "=",
                            Literal(1),
                        )
                    )
            result[name] = conjunction([base, *exclusions])
        return result

    def alphabet(self, candidate):
        codes = {name: self.compile(expr) for name, expr in candidate.items()}
        cells = set()
        for index, row in enumerate(self.valuations):
            before = frozenset(
                name for name, truth in self.truth.items() if index in truth
            )
            after = frozenset(
                name for name, code in codes.items() if self.evaluate(code, row) is True
            )
            cells.add((before, after))
        return tuple(
            sorted(
                cells, key=lambda pair: (tuple(sorted(pair[0])), tuple(sorted(pair[1])))
            )
        )


# Observable label membership
#
# The label information that the surrounding SQL can actually observe.
#
# Each consumed row emits a membership vector, including a neutral vector for
# unobserved labels. Keeping neutral rows (rather than deleting them) preserves
# endpoints and the exact row positions in every observed primary/subset group.
# CLASSIFIER observes all primary names. Qualified expressions and label SKIP
# observe their group's membership; a SUBSET declaration alone observes nothing.


@dataclass(frozen=True)
class ObservationsUltra:
    groups: tuple
    required_variables: frozenset
    classifier: bool

    def token(self, label):
        return tuple(label in group for group in self.groups)


def observations(ast, expressions):
    variables = set(names(ast.pattern))
    subsets = ast.subsets or {}
    requested, required = set(), set()
    classifier = False

    def require(name):
        if name in variables:
            requested.add(frozenset({name}))
            required.add(name)
        elif name in subsets:
            requested.add(frozenset(subsets[name]))

    # Includes current-group references retained inside aggregates/navigation,
    # and fixed-prefix references, including references to union variables.
    for expression in expressions.values():
        for name in references(expression):
            require(name)
    if ast.measures is not None:
        for expression in ast.measures.measures.values():
            for node in expr_walk(expression):
                if isinstance(node, FunctionCall) and node.name.upper() == "CLASSIFIER":
                    classifier = True
                elif isinstance(node, ColumnRef) and node.qualifier:
                    require(node.qualifier)
    skip = ast.after_match_skip_to or ""
    if skip.startswith(("FIRST ", "LAST ")):
        require(skip.split()[-1])
    elif skip not in {"NEXT ROW", "PAST LAST ROW", ""}:
        # Some AST producers spell an unqualified SKIP target directly.
        require(skip.removeprefix("TO "))
    if classifier:
        requested.update(frozenset({name}) for name in variables)
    return ObservationsUltra(
        tuple(sorted(requested, key=lambda group: tuple(sorted(group)))),
        frozenset(required),
        classifier,
    )
