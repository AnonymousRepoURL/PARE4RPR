if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

import math
import re
import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Tuple

from expression_clause import DEPENDENT_FUNCS
from queryanalysis import ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS

from .physical_navigation import PatternNavigationReach
from .utils import *


@dataclass(frozen=True)
class WindowPrefilterStatistics:
    """Runtime statistics consumed by the prefilter cost model."""

    variable_counts: Dict[str, float]
    row_count: float
    window_length: float
    partition_count: Optional[float]
    poisson_lambdas: Dict[str, float]

    @classmethod
    def from_lambdas(cls, values: Mapping[str, float]):
        lambdas = {str(name).upper(): float(value) for name, value in values.items()}
        return cls({}, 0.0, 0.0, None, lambdas)


@dataclass(frozen=True)
class WindowPrefilterStatisticsRequest:
    """SQL request plus enough metadata to parse its single result row."""

    sql: str
    tracked_variables: Tuple[str, ...]
    aliases: Dict[str, str]
    window: float

    def parse_row(self, row: Mapping[str, object]) -> WindowPrefilterStatistics:
        normalized = {str(name).upper(): value for name, value in row.items()}

        def numeric(alias: str):
            if alias.upper() not in normalized:
                raise ValueError(f"statistics column {alias} is missing")
            value = normalized[alias.upper()]
            if value is None:
                raise ValueError(f"statistics column {alias} is NULL")
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"statistics column {alias} is not numeric") from exc
            if not math.isfinite(result) or result < 0:
                raise ValueError(
                    f"statistics column {alias} must be finite and non-negative"
                )
            return result

        counts = {variable: numeric(variable) for variable in self.tracked_variables}
        row_count = numeric(self.aliases["N"])
        window_length = numeric(self.aliases["WLEN"])
        partition_count = numeric(self.aliases["M"]) if "M" in self.aliases else None

        # The rule's Poisson weight expects the mean event count in one
        # pattern-sized window.  With PARTITION BY, approximate total exposure
        # as global time span times the number of partitions.
        exposure = window_length
        if partition_count is not None:
            exposure *= max(partition_count, 1.0)
        lambdas = {
            variable.upper(): (count * self.window / exposure if exposure > 0 else 0.0)
            for variable, count in counts.items()
        }
        return WindowPrefilterStatistics(
            variable_counts=counts,
            row_count=row_count,
            window_length=window_length,
            partition_count=partition_count,
            poisson_lambdas=lambdas,
        )


@dataclass(frozen=True)
class WindowPrefilterCandidate:
    """A source-independent rewrite choice that can be rendered twice."""

    candidate_id: str
    variables: Tuple[str, ...]
    strategy: str
    estimated_cost: float


class WindowConditionPreFilterRule(RewriteRule):
    """
    Two-phase bucket pre-filter driven by collected input statistics.

    ``is_applicable`` performs structural and safety checks without reading
    statistics. The adaptive API builds a statistics request, generates one
    cost-selected single-variable candidate and all feasible two-variable
    candidates, then renders whichever candidate an external sampler selects.

    The older ``apply`` API remains supported for the regular RewriteEngine:
    constructor-provided lambdas or ``table_schemas['poisson_lambdas']`` select
    one fixed rewrite. Missing statistics make it return False rather than
    silently inventing an estimate.

    Complete source schemas are required because the helper bucket column must
    be projected away. Constructor-provided lambdas remain available as a
    convenient test/programmatic override.
    """

    def __init__(
        self,
        poisson_lambdas: Optional[Dict[str, float]] = None,
        filter_variable_count: int = 1,
    ):

        # "High-performance row pattern recognition using joins"
        # indicate that use one to two variables can improve query performance,
        # thus, more than two variables is not recommended.
        if filter_variable_count not in (1, 2):
            raise ValueError("filter_variable_count must be 1 or 2")
        self.poisson_lambdas = {
            str(name).upper(): float(value)
            for name, value in (poisson_lambdas or {}).items()
        }
        self.filter_variable_count = filter_variable_count
        self.statistics_aliases: Dict[str, str] = {}
        self.last_rewrite_variants: List[str] = []

    @property
    def name(self) -> str:
        return "window_condition_pre_filter"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_structural_plan(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        if getattr(ast, "_window_condition_prefilter_applied", False):
            return False
        plan = self._build_apply_plan(ast)
        if plan is None:
            return False
        return self._apply_independent_plan(ast, plan, plan["selected"])

    def _apply_independent_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan,
        selected,
        input_table_override: Optional[str] = None,
    ) -> bool:
        source = (
            input_table_override.strip()
            if input_table_override is not None
            else ast.input_table.strip()
        )
        cols = ", ".join(plan["columns"])
        bucket_col = plan["bucket_col"]
        bucketed_rows = plan["bucketed_rows"]
        bucket_list_name = plan["bucket_list_name"]
        filtered_rows = plan["filtered_rows"]
        partition_cols = plan["partition_cols"]
        window = self._format_number(plan["window"])
        bucketed = (
            f"WITH {bucketed_rows} AS(\n"
            f"    SELECT {cols}, FLOOR({plan['order_col']} / {window}) AS {bucket_col}\n"
            f"    FROM {source}\n"
            ")"
        )

        per_variable_sets = []
        for selected_candidate in selected:
            selects = []
            predicate = selected_candidate["predicate"].to_string()
            for offset in selected_candidate["offsets"]:
                expression = bucket_col
                if offset > 0:
                    expression += f" + {offset}"
                elif offset < 0:
                    expression += f" - {abs(offset)}"
                alias = "" if offset == 0 else f" AS {bucket_col}"
                projection = self._bucket_key_projection(
                    partition_cols, f"{expression}{alias}"
                )
                selects.append(
                    f"SELECT {projection}\n"
                    f"  FROM {bucketed_rows}\n"
                    f"  WHERE {predicate}"
                )
            per_variable_sets.append("\n   UNION\n   ".join(selects))

        bucket_list_body = "\n   INTERSECT\n   ".join(
            f"({sql})" if len(per_variable_sets) > 1 else sql
            for sql in per_variable_sets
        )
        bucket_list = f", {bucket_list_name} AS(\n  {bucket_list_body}\n)"
        filtered = self._build_filtered_rows_sql(plan, cols)
        ast.cte_prefix = self._append_cte_prefix(ast, bucketed + bucket_list + filtered)
        ast.input_table = self._rebind_filtered_input(plan, filtered_rows)
        # The AST does not parse generated WITH definitions. Keep the CTE's
        # public schema available to safe downstream rewrites.
        ast._window_prefilter_input_columns = tuple(plan["columns"])
        ast._window_condition_prefilter_applied = True
        return True

    def generate_candidates(
        self,
        ast: RPRAbstractSyntaxTree,
        statistics: WindowPrefilterStatistics,
        max_pair_candidates: Optional[int] = None,
        max_pair_estimated_cost: Optional[float] = None,
    ) -> List[WindowPrefilterCandidate]:
        """Generate one best single-variable candidate and all useful pairs.

        The current Poisson model chooses the sole single-variable candidate.
        Every pair produces an independent bucket-intersection candidate; when
        a safe scalar dependency exists, a dependency-join candidate is added
        as a separate choice for empirical benchmarking.
        """
        plan = self._build_costed_plan(ast, statistics)
        if plan is None:
            return []

        ranked = plan["ranked_candidates"]
        best = ranked[0]
        candidates = [
            WindowPrefilterCandidate(
                candidate_id=f"single:{best['var']}",
                variables=(best["var"],),
                strategy="single",
                estimated_cost=best["weight"],
            )
        ]

        pair_candidates = []
        for first, second in combinations(ranked, 2):
            variables = tuple(sorted((first["var"], second["var"])))
            pair_cost = first["weight"] * second["weight"]
            pair_candidates.append(
                WindowPrefilterCandidate(
                    candidate_id=f"pair_intersection:{'+'.join(variables)}",
                    variables=variables,
                    strategy="pair_intersection",
                    estimated_cost=pair_cost,
                )
            )
            if self._build_dependency_pair_plan(ast, plan, first, second) is not None:
                pair_candidates.append(
                    WindowPrefilterCandidate(
                        candidate_id=f"pair_dependency_join:{'+'.join(variables)}",
                        variables=variables,
                        strategy="pair_dependency_join",
                        estimated_cost=pair_cost,
                    )
                )

        if max_pair_estimated_cost is not None:
            pair_candidates = [
                candidate
                for candidate in pair_candidates
                if candidate.estimated_cost <= max_pair_estimated_cost
            ]
        pair_candidates.sort(
            key=lambda candidate: (candidate.estimated_cost, candidate.candidate_id)
        )
        if max_pair_candidates is not None:
            if max_pair_candidates < 0:
                raise ValueError("max_pair_candidates must be non-negative")
            pair_candidates = pair_candidates[:max_pair_candidates]
        candidates.extend(pair_candidates)
        return candidates

    def rewrite_candidate(
        self,
        ast: RPRAbstractSyntaxTree,
        candidate: WindowPrefilterCandidate,
        statistics: WindowPrefilterStatistics,
        input_table_override: Optional[str] = None,
    ) -> Optional[RPRAbstractSyntaxTree]:
        """Return a rewritten copy for one candidate and source relation."""
        rewritten = copy.deepcopy(ast)
        plan = self._build_costed_plan(ast, statistics)
        if plan is None:
            return None
        selected_by_name = {item["var"]: item for item in plan["ranked_candidates"]}
        try:
            selected = [selected_by_name[name] for name in candidate.variables]
        except KeyError:
            return None

        if candidate.strategy in {"single", "pair_intersection"}:
            expected_count = 1 if candidate.strategy == "single" else 2
            if len(selected) != expected_count:
                return None
            if not self._apply_independent_plan(
                rewritten, plan, selected, input_table_override
            ):
                return None
            return rewritten

        if candidate.strategy == "pair_dependency_join" and len(selected) == 2:
            pair_plan = self._build_dependency_pair_plan(
                ast, plan, selected[0], selected[1]
            )
            if pair_plan is None:
                return None
            if not self._apply_dependency_pair_plan(
                rewritten, pair_plan, input_table_override
            ):
                return None
            return rewritten
        return None

    def generate_rewrite_variants(self, ast: RPRAbstractSyntaxTree) -> List[str]:
        """Return every useful rewrite when two-variable filtering is enabled.

        The first result is the regular independent-predicate rewrite. Extra
        results use a safe scalar dependency between a pair of must-appear
        variables as a self-join over the bucketed input. The source AST is not
        mutated.
        """
        self.last_rewrite_variants = []
        plan = self._build_apply_plan(ast)
        if plan is None:
            return []

        independent_ast = copy.deepcopy(ast)
        if self.apply(independent_ast):
            self.last_rewrite_variants.append(independent_ast.to_sql())

        if self.filter_variable_count != 2:
            return list(self.last_rewrite_variants)

        for left_candidate, right_candidate in combinations(
            plan["ranked_candidates"], 2
        ):
            pair_plan = self._build_dependency_pair_plan(
                ast, plan, left_candidate, right_candidate
            )
            if pair_plan is None:
                continue
            variant_ast = copy.deepcopy(ast)
            self._apply_dependency_pair_plan(variant_ast, pair_plan)
            self.last_rewrite_variants.append(variant_ast.to_sql())
        return list(self.last_rewrite_variants)

    def get_rewrite_variants(self) -> List[str]:
        """Return variants produced by the latest generate_rewrite_variants call."""
        return list(self.last_rewrite_variants)

    def obtain_sta_sql(
        self, ast: RPRAbstractSyntaxTree
    ) -> Tuple[Optional[str], List[str]]:
        """
        if ast.is_applicable() is true then we excute this function
        to obtain the statistics query and the list of tracked variables.

        Build the query used to collect selectivity/cardinality statistics.

        One conditional count is emitted for every must-appear variable for
        which an independent DEFINE predicate can be extracted. The
        alias is exactly the pattern-variable name. ``N`` is the input row
        count, ``WLEN`` is the range of the first MATCH_RECOGNIZE ordering
        column, and ``M`` (when PARTITION BY is present) is the number of
        distinct partitions.

        Return ``(sql, tracked_variables)``. The variable list follows the
        conditional-count column order and contains only variables for which a
        count was emitted. When a meaningful query cannot be built, return
        ``(None, [])``.
        """
        request = self.build_statistics_request(ast)
        if request is None:
            return None, []
        return request.sql, list(request.tracked_variables)

    def build_statistics_request(
        self,
        ast: RPRAbstractSyntaxTree,
        input_table_override: Optional[str] = None,
    ) -> Optional[WindowPrefilterStatisticsRequest]:
        """Describe the one-row statistics query required by this rule.

        ``input_table_override`` lets an orchestrator collect statistics from
        the exact sample relation used for candidate benchmarking.  A source
        alias referenced by PARTITION BY / ORDER BY is rebound automatically.
        """
        self.statistics_aliases = {}
        structural_plan = self._build_structural_plan(ast)
        if structural_plan is None:
            return None

        order_col = structural_plan["order_col_sql"]
        # Variable-count aliases are required to remain identical to variable
        # names. Therefore generated statistics aliases must avoid all pattern
        # variable names, including variables without an independent filter.
        used_aliases = {name.upper() for name in ast.pattern.get_variables()}
        if ast.table_schemas is not None:
            for meta in ast.table_schemas.values():
                columns = meta.get("columns", []) if isinstance(meta, dict) else meta
                if columns is not None:
                    used_aliases.update(
                        str(column).strip().split(".")[-1].upper() for column in columns
                    )
        n_alias = self._choose_statistics_alias("N", used_aliases)
        wlen_alias = self._choose_statistics_alias("WLEN", used_aliases)
        m_alias = None
        if ast.partition_by:
            m_alias = self._choose_statistics_alias("M", used_aliases)
        self.statistics_aliases = {
            "N": n_alias,
            "WLEN": wlen_alias,
            **({"M": m_alias} if m_alias is not None else {}),
        }

        source = ast.input_table.strip()
        if input_table_override is not None:
            source = input_table_override.strip()
            rebind_alias = structural_plan["input_rebind_alias"]
            if rebind_alias is not None:
                source = bind_input_relation_alias(source, rebind_alias)

        select_items = []
        tracked_variables = []
        for candidate in structural_plan["candidates"]:
            var_name = candidate["var"]
            independent_expr = candidate["predicate"]
            select_items.append(
                f"COUNT(CASE WHEN {independent_expr.to_string()} THEN 1 END) AS {var_name}"
            )
            tracked_variables.append(var_name)

        select_items.append(f"COUNT(*) AS {n_alias}")
        select_items.append(
            f"COALESCE(MAX({order_col}) - MIN({order_col}), 0) AS {wlen_alias}"
        )

        projection = ",\n    ".join(select_items)
        sql = f"SELECT\n    {projection}\nFROM {source}"

        partition_cols = [
            col.strip() for col in (ast.partition_by or []) if col.strip()
        ]
        if partition_cols:
            # Oracle does not accept COUNT(DISTINCT (c1, c2)), and ordinary
            # COUNT(DISTINCT c1) omits the NULL partition.  A nested DISTINCT
            # is portable across the supported target databases and counts
            # every row-pattern partition, including nullable composite keys.
            partition_projection = ", ".join(partition_cols)
            partition_count = (
                "(SELECT COUNT(*) FROM "
                f"(SELECT DISTINCT {partition_projection} FROM {source}) "
                "RPR_PARTITIONS)"
            )
            # Oracle rejects a scalar subquery beside ungrouped aggregates
            # (ORA-00937). Attach partition cardinality to their one-row result.
            aggregate_sql = sql.replace("\n", "\n    ")
            sql = (
                f"SELECT RPR_STATS.*,\n    {partition_count} AS {m_alias}\n"
                f"FROM (\n    {aggregate_sql}\n) RPR_STATS"
            )
        cte_prefix = (ast.cte_prefix or "").strip()
        if cte_prefix:
            sql = f"{cte_prefix}\n{sql}"
        return WindowPrefilterStatisticsRequest(
            sql=sql,
            tracked_variables=tuple(tracked_variables),
            aliases=dict(self.statistics_aliases),
            window=structural_plan["window"],
        )

    def get_statistics_aliases(self) -> Dict[str, str]:
        """Return logical-to-actual aliases from the latest statistics query."""
        return dict(self.statistics_aliases)

    @staticmethod
    def _choose_statistics_alias(base_name: str, used_aliases: set) -> str:
        candidate = base_name.upper()
        suffix = 1
        while candidate in used_aliases:
            candidate = f"{base_name.upper()}_{suffix}"
            suffix += 1
        used_aliases.add(candidate)
        return candidate

    def _build_structural_plan(self, ast: RPRAbstractSyntaxTree):
        """Build the statistics-independent portion of the rewrite plan."""
        if (
            ast.pattern is None
            or ast.define is None
            or not ast.match_order_by
            or not ast.input_table
            or (ast.rows_per_match or "").strip().upper()
            == ALL_ROWS_PER_MATCH_WITH_UNMATCHED_ROWS
        ):
            return None
        qualifier_safe, input_rebind_alias = resolve_input_rebind_alias(ast)
        if not qualifier_safe:
            return None

        partition_cols = []
        for partition_col in ast.partition_by or []:
            partition_col = partition_col.strip()
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)?",
                partition_col,
            ):
                return None
            partition_cols.append(partition_col.split(".")[-1])
        if len({col.upper() for col in partition_cols}) != len(partition_cols):
            return None
        left = ast.pattern.get_leftmost_variables()
        right = ast.pattern.get_rightmost_variables()
        if len(left) != 1 or len(right) != 1:
            return None
        left_var, right_var = next(iter(left)), next(iter(right))
        if left_var == right_var:
            return None
        if not self._navigation_stays_within_pattern(ast):
            return None

        order_col_sql = ast.match_order_by[0][0].strip()
        if not order_col_sql:
            return None
        order_col = order_col_sql.split(".")[-1]
        sort_type = ast.match_order_by[0][1]
        window = self._find_window(ast, left_var, right_var, order_col, sort_type)
        if window is None or window <= 0:
            return None

        columns = self._complete_input_columns(ast)
        if not columns or len({col.upper() for col in columns}) != len(columns):
            return None
        bucket_col = choose_generated_column_name(ast, "BK_NO", columns)
        used_relation_names = self._collect_used_generated_names(ast, columns)
        used_relation_names.add(bucket_col.upper())
        bucketed_rows = self._choose_generated_relation_name(
            "BUCKETED_ROWS", used_relation_names
        )
        bucket_list_name = self._choose_generated_relation_name(
            "BK_NO_LIST", used_relation_names
        )
        filtered_rows = self._choose_generated_relation_name(
            "FILTERED_ROWS", used_relation_names
        )
        ranges_name = self._choose_generated_relation_name(
            "RANGES", used_relation_names
        )
        filtered_input_alias = self._choose_generated_relation_name(
            "PREFILTER_INPUT", used_relation_names
        )
        bucket_key_alias = self._choose_generated_relation_name(
            "PREFILTER_BUCKET", used_relation_names
        )

        candidates = []
        for var in sorted(ast.pattern.get_must_appear_var()):
            condition = ast.define.definitions.get(var)
            predicate = (
                condition.get_independent_filter() if condition is not None else None
            )
            if predicate is None:
                continue
            predicate = _strip_var_qualifier(predicate, var)
            if predicate is None:
                continue

            if var == left_var:
                offsets = [0, 1] if sort_type == SortType.ASC else [-1, 0]
            elif var == right_var:
                offsets = [-1, 0] if sort_type == SortType.ASC else [0, 1]
            else:
                offsets = [-1, 0, 1]
            candidates.append(
                {
                    "var": var,
                    "predicate": predicate,
                    "endpoint": var in {left_var, right_var},
                    "offsets": offsets,
                }
            )
        if len(candidates) < self.filter_variable_count:
            return None
        return {
            "columns": columns,
            "bucket_col": bucket_col,
            "bucketed_rows": bucketed_rows,
            "bucket_list_name": bucket_list_name,
            "filtered_rows": filtered_rows,
            "ranges_name": ranges_name,
            "filtered_input_alias": filtered_input_alias,
            "bucket_key_alias": bucket_key_alias,
            "partition_cols": partition_cols,
            "non_null_partition_cols": self._non_null_partition_columns(
                ast, partition_cols
            ),
            "input_rebind_alias": input_rebind_alias,
            "order_col": order_col,
            "order_col_sql": order_col_sql,
            "sort_type": sort_type,
            "left_var": left_var,
            "right_var": right_var,
            "window": window,
            "candidates": candidates,
        }

    def _collect_used_generated_names(
        self, ast: RPRAbstractSyntaxTree, columns: List[str]
    ) -> set:
        """Collect identifiers that generated CTE names must not shadow."""
        used = {str(column).strip().split(".")[-1].upper() for column in columns}
        used.update(str(name).upper() for name in (ast.table_schemas or {}).keys())

        sql_fragments = [ast.input_table or "", ast.cte_prefix or ""]
        for fragment in sql_fragments:
            used.update(
                token.upper()
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", fragment)
            )
        return used

    @staticmethod
    def _non_null_partition_columns(
        ast: RPRAbstractSyntaxTree, partition_cols: List[str]
    ) -> frozenset[str]:
        """Prove NOT NULL on the current input, not just an underlying table.

        Ordinary relations and SELECT-star selection wrappers preserve column
        values. Explicit derived projections and set/group operations are left
        unknown: resolving a reused output name against its base schema could
        incorrectly hide a NULL-producing expression.
        """
        source = ast.input_table.strip()
        if not partition_cols or input_may_null_extend_columns(source):
            return frozenset()
        while source.startswith("("):
            if ast._find_matching_paren(source, 0) != len(source) - 1:
                return frozenset()
            query = source[1:-1].strip()
            projection = re.match(r"SELECT\s+\*\s+FROM\s+", query, re.IGNORECASE)
            if projection is None:
                return frozenset()
            tail = query[projection.end() :].strip()
            if any(
                ast._find_top_level_keyword(tail, keyword) >= 0
                for keyword in (
                    "GROUP",
                    "HAVING",
                    "UNION",
                    "INTERSECT",
                    "EXCEPT",
                    "MINUS",
                    "CONNECT",
                    "START",
                    "MODEL",
                    "PIVOT",
                    "UNPIVOT",
                )
            ):
                return frozenset()
            clause_positions = [
                position
                for keyword in ("WHERE", "ORDER", "LIMIT", "FETCH", "OFFSET", "QUALIFY")
                if (position := ast._find_top_level_keyword(tail, keyword)) >= 0
            ]
            source = tail[: min(clause_positions)].strip() if clause_positions else tail

        identifier = r"[A-Za-z_][A-Za-z0-9_$]*"
        if (
            re.fullmatch(
                rf"{identifier}(?:\.{identifier}){{0,2}}"
                rf"(?:\s+(?:AS\s+)?{identifier})?",
                source,
                re.IGNORECASE,
            )
            is None
        ):
            return frozenset()
        probe = copy.copy(ast)
        probe.input_table = source
        probe._clear_metadata_cache()
        non_null_columns = set()
        for column in partition_cols:
            try:
                nullable = probe.is_column_nullable(column)
            except ValueError:
                # Missing type metadata must not turn this optional SQL
                # simplification into a failure of an otherwise valid plan.
                continue
            if nullable is False:
                non_null_columns.add(column.upper())
        return frozenset(non_null_columns)

    @staticmethod
    def _choose_generated_relation_name(base_name: str, used_names: set) -> str:
        base = base_name.upper()
        candidate = base
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    def _build_apply_plan(self, ast: RPRAbstractSyntaxTree):
        """Add statistics-dependent variable selection to a structural plan."""
        statistics = WindowPrefilterStatistics.from_lambdas(self._poisson_lambdas(ast))
        result = self._build_costed_plan(ast, statistics)
        if result is None:
            return None
        result["selected"] = result["ranked_candidates"][: self.filter_variable_count]
        return result

    def _build_costed_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        statistics: WindowPrefilterStatistics,
    ):
        """Attach current-model weights without choosing a candidate shape."""
        plan = self._build_structural_plan(ast)
        if plan is None:
            return None

        lambdas = {
            str(name).upper(): float(value)
            for name, value in statistics.poisson_lambdas.items()
            if float(value) >= 0 and math.isfinite(float(value))
        }
        ranked_candidates = []
        for candidate in plan["candidates"]:
            poisson_lambda = lambdas.get(candidate["var"].upper())
            if poisson_lambda is None:
                # Without every candidate's statistic we cannot know which
                # variable really has the lowest weight.
                return None
            # The expected number of windows spanned by the time interval is
            # \frac{T}{w} (1-e^{-multiplier * \lambda_{1} * w} )
            # since w is the same for all variables
            # thus we only need to consider the exponent
            multiplier = 2 if candidate["endpoint"] else 3
            weight = 1 - math.exp(-multiplier * poisson_lambda)
            ranked = dict(candidate)
            ranked["poisson_lambda"] = poisson_lambda
            ranked["weight"] = weight
            ranked_candidates.append(ranked)

        ranked_candidates.sort(key=lambda item: (item["weight"], item["var"]))
        result = dict(plan)
        result["ranked_candidates"] = ranked_candidates
        return result

    def _build_dependency_pair_plan(self, ast, plan, first, second):
        pair_vars = {first["var"], second["var"]}
        dependencies = self._collect_pair_dependencies(ast, plan, pair_vars)
        if not dependencies:
            return None

        first_before_second = self._must_appear_before(
            ast.pattern, first["var"], second["var"]
        )
        second_before_first = self._must_appear_before(
            ast.pattern, second["var"], first["var"]
        )
        if first_before_second == second_before_first:
            return None
        earlier, later = (first, second) if first_before_second else (second, first)

        used_names = self._collect_used_generated_names(ast, plan["columns"])
        used_names.update(
            {
                plan["bucketed_rows"],
                plan["bucket_list_name"],
                plan["filtered_rows"],
                plan["ranges_name"],
                plan["bucket_col"],
            }
        )
        aliases = {
            first["var"]: self._choose_generated_relation_name(
                f"{first['var']}_BUCKET_NO", used_names
            ),
            second["var"]: self._choose_generated_relation_name(
                f"{second['var']}_BUCKET_NO", used_names
            ),
        }
        flag_col = choose_generated_column_name(
            ast, "FLAG_NO", list(plan["columns"]) + [plan["bucket_col"]]
        )

        predicates = []
        for candidate in (first, second):
            rewritten = self._rewrite_pair_expr(
                candidate["predicate"], candidate["var"], aliases, True
            )
            if rewritten is None:
                return None
            predicates.append(rewritten.to_string())
        for def_var, dependency in dependencies:
            rewritten = self._rewrite_pair_expr(dependency, def_var, aliases, False)
            if rewritten is None:
                return None
            predicates.append(rewritten.to_string())

        flag = min((first, second), key=lambda item: (item["weight"], item["var"]))
        result = dict(plan)
        result.update(
            {
                "pair": (first, second),
                "earlier": earlier,
                "later": later,
                "aliases": aliases,
                "flag_col": flag_col,
                "where_predicates": predicates,
                "flag": flag,
            }
        )
        return result

    def _must_appear_before(
        self, node, before_var: str, target_var: str, seen_before: bool = False
    ) -> bool:
        """Whether every occurrence path to target has a mandatory before_var."""
        if target_var not in node.get_variables():
            return True
        if isinstance(node, VariablePattern):
            return seen_before if node.name == target_var else True
        if isinstance(node, (QuantifiedPattern, ExcludedPattern, AnchorPattern)):
            return self._must_appear_before(
                node.pattern, before_var, target_var, seen_before
            )
        if isinstance(node, AlternationPattern):
            return all(
                self._must_appear_before(child, before_var, target_var, seen_before)
                for child in node.patterns
                if target_var in child.get_variables()
            )
        if isinstance(node, ConcatenationPattern):
            current_seen = seen_before
            for child in node.patterns:
                if target_var in child.get_variables() and not self._must_appear_before(
                    child, before_var, target_var, current_seen
                ):
                    return False
                if before_var in child.get_must_appear_var():
                    current_seen = True
            return True
        return False

    def _collect_pair_dependencies(self, ast, plan, pair_vars: set):
        dependencies = []
        for def_var in sorted(pair_vars):
            condition = ast.define.definitions.get(def_var)
            if condition is None or condition.expression is None:
                continue
            for atom in _split_conjuncts(condition.expression):
                unwrapped = atom.expr if isinstance(atom, SemanticsExpr) else atom
                if (
                    def_var == plan["right_var"]
                    and plan["left_var"] in pair_vars
                    and self._parse_window_atom(
                        unwrapped,
                        plan["right_var"],
                        plan["left_var"],
                        plan["order_col"],
                        plan["sort_type"],
                    )
                    is not None
                ):
                    continue
                referenced, safe = self._pair_expr_variables(unwrapped, def_var)
                if safe and referenced == pair_vars:
                    dependencies.append((def_var, unwrapped))
        return dependencies

    def _pair_expr_variables(self, expr: Expr, current_var: str):
        referenced = set()
        safe = True

        def visit(node):
            nonlocal safe
            if node is None or not safe:
                return
            if isinstance(node, SemanticsExpr):
                visit(node.expr)
            elif isinstance(node, ColumnRef):
                referenced.add(node.qualifier or current_var)
            elif isinstance(node, FunctionCall):
                if node.name.upper() in DEPENDENT_FUNCS:
                    safe = False
                else:
                    for arg in node.args:
                        visit(arg)
            elif isinstance(node, SubqueryExpr):
                safe = False
            elif isinstance(node, BinaryExpr):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, BetweenExpr):
                visit(node.expr)
                visit(node.lower)
                visit(node.upper)
            elif isinstance(node, InExpr):
                visit(node.expr)
                for value in node.values:
                    visit(value)
            elif isinstance(node, IsNullExpr):
                visit(node.expr)
            elif isinstance(node, NotExpr):
                visit(node.expr)
            elif isinstance(node, CaseExpr):
                for condition, result in node.cases:
                    visit(condition)
                    visit(result)
                visit(node.else_expr)

        visit(expr)
        return referenced, safe

    def _rewrite_pair_expr(
        self,
        expr: Expr,
        current_var: str,
        aliases: Dict[str, str],
        allow_functions: bool,
    ) -> Optional[Expr]:
        if isinstance(expr, SemanticsExpr):
            inner = self._rewrite_pair_expr(
                expr.expr, current_var, aliases, allow_functions
            )
            return SemanticsExpr(expr.semantics, inner) if inner is not None else None
        if isinstance(expr, ColumnRef):
            var = expr.qualifier or current_var
            alias = aliases.get(var)
            return ColumnRef(alias, expr.name) if alias is not None else None
        if isinstance(expr, Literal):
            return Literal(expr.value)
        if isinstance(expr, FunctionCall):
            if not allow_functions and expr.name.upper() in DEPENDENT_FUNCS:
                return None
            args = [
                self._rewrite_pair_expr(arg, current_var, aliases, allow_functions)
                for arg in expr.args
            ]
            return (
                FunctionCall(expr.name, args)
                if all(arg is not None for arg in args)
                else None
            )
        if isinstance(expr, BinaryExpr):
            left = self._rewrite_pair_expr(
                expr.left, current_var, aliases, allow_functions
            )
            right = self._rewrite_pair_expr(
                expr.right, current_var, aliases, allow_functions
            )
            return (
                BinaryExpr(left, expr.op, right)
                if left is not None and right is not None
                else None
            )
        if isinstance(expr, BetweenExpr):
            values = [
                self._rewrite_pair_expr(value, current_var, aliases, allow_functions)
                for value in (expr.expr, expr.lower, expr.upper)
            ]
            return (
                BetweenExpr(*values)
                if all(value is not None for value in values)
                else None
            )
        if isinstance(expr, InExpr):
            inner = self._rewrite_pair_expr(
                expr.expr, current_var, aliases, allow_functions
            )
            values = [
                self._rewrite_pair_expr(v, current_var, aliases, allow_functions)
                for v in expr.values
            ]
            return (
                InExpr(inner, values, expr.is_not)
                if inner is not None and all(v is not None for v in values)
                else None
            )
        if isinstance(expr, IsNullExpr):
            inner = self._rewrite_pair_expr(
                expr.expr, current_var, aliases, allow_functions
            )
            return IsNullExpr(inner, expr.is_not) if inner is not None else None
        if isinstance(expr, NotExpr):
            inner = self._rewrite_pair_expr(
                expr.expr, current_var, aliases, allow_functions
            )
            return NotExpr(inner) if inner is not None else None
        if isinstance(expr, CaseExpr):
            cases = []
            for condition, value in expr.cases:
                new_condition = self._rewrite_pair_expr(
                    condition, current_var, aliases, allow_functions
                )
                new_value = self._rewrite_pair_expr(
                    value, current_var, aliases, allow_functions
                )
                if new_condition is None or new_value is None:
                    return None
                cases.append((new_condition, new_value))
            else_expr = (
                self._rewrite_pair_expr(
                    expr.else_expr, current_var, aliases, allow_functions
                )
                if expr.else_expr
                else None
            )
            return CaseExpr(cases, else_expr)
        if isinstance(expr, SubqueryExpr):
            return None
        return None

    def _apply_dependency_pair_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        plan,
        input_table_override: Optional[str] = None,
    ) -> bool:
        source = (
            input_table_override.strip()
            if input_table_override is not None
            else ast.input_table.strip()
        )
        cols = ", ".join(plan["columns"])
        bucket_col = plan["bucket_col"]
        bucketed_rows = plan["bucketed_rows"]
        ranges_name = plan["ranges_name"]
        bucket_list_name = plan["bucket_list_name"]
        filtered_rows = plan["filtered_rows"]
        window = self._format_number(plan["window"])
        bucketed = (
            f"WITH {bucketed_rows} AS(\n"
            f"    SELECT {cols}, FLOOR({plan['order_col']} / {window}) AS {bucket_col}\n"
            f"    FROM {source}\n"
            ")"
        )

        earlier_alias = plan["aliases"][plan["earlier"]["var"]]
        later_alias = plan["aliases"][plan["later"]["var"]]
        flag_alias = plan["aliases"][plan["flag"]["var"]]
        flag_col = plan["flag_col"]
        partition_conditions = []
        for col in plan["partition_cols"]:
            partition_conditions.append(
                self._partition_equality(
                    plan, col, f"{earlier_alias}.{col}", f"{later_alias}.{col}"
                )
            )

        range_selects = []
        direction = 1 if plan["sort_type"] == SortType.ASC else -1
        for delta in (0, direction):
            earlier_bucket = f"{earlier_alias}.{bucket_col}"
            later_bucket = f"{later_alias}.{bucket_col}"
            if delta > 0:
                bucket_condition = f"{earlier_bucket} + {delta} = {later_bucket}"
            elif delta < 0:
                bucket_condition = f"{earlier_bucket} - {abs(delta)} = {later_bucket}"
            else:
                bucket_condition = f"{earlier_bucket} = {later_bucket}"
            on_conditions = [bucket_condition] + partition_conditions
            range_projection = self._bucket_key_projection(
                [f"{flag_alias}.{col} AS {col}" for col in plan["partition_cols"]],
                f"{flag_alias}.{bucket_col} AS {flag_col}",
            )
            range_selects.append(
                f"SELECT {range_projection}\n"
                f"    FROM {bucketed_rows} {earlier_alias} "
                f"INNER JOIN {bucketed_rows} {later_alias}\n"
                f"      ON {' AND '.join(on_conditions)}\n"
                f"    WHERE {' AND '.join(plan['where_predicates'])}"
            )
        ranges = (
            f", {ranges_name} AS(\n    "
            + "\n    UNION\n    ".join(range_selects)
            + "\n)"
        )

        offsets = plan["flag"]["offsets"]
        bucket_selects = []
        for offset in offsets:
            bucket_value = self._offset_sql(flag_col, offset)
            projection = self._bucket_key_projection(
                plan["partition_cols"], f"{bucket_value} AS {bucket_col}"
            )
            bucket_selects.append(f"SELECT {projection}\n" f"    FROM {ranges_name}")
        bucket_list = (
            f", {bucket_list_name} AS(\n"
            f"    " + "\n    UNION\n    ".join(bucket_selects) + "\n"
            ")"
        )
        filtered = self._build_filtered_rows_sql(plan, cols)
        ast.cte_prefix = self._append_cte_prefix(
            ast, bucketed + ranges + bucket_list + filtered
        )
        ast.input_table = self._rebind_filtered_input(plan, filtered_rows)
        ast._window_prefilter_input_columns = tuple(plan["columns"])
        ast._window_condition_prefilter_applied = True
        return True

    @staticmethod
    def _append_cte_prefix(ast: RPRAbstractSyntaxTree, generated_prefix: str) -> str:
        existing_prefix = (ast.cte_prefix or "").strip()
        if not existing_prefix:
            return generated_prefix
        return existing_prefix + ",\n" + generated_prefix[len("WITH ") :]

    @staticmethod
    def _bucket_key_projection(partition_items: List[str], bucket_item: str) -> str:
        return ", ".join([*partition_items, bucket_item])

    @staticmethod
    def _null_safe_equality(left: str, right: str) -> str:
        return f"({left} = {right} OR ({left} IS NULL AND {right} IS NULL))"

    def _partition_equality(self, plan, column: str, left: str, right: str) -> str:
        if column.upper() in plan["non_null_partition_cols"]:
            return f"{left} = {right}"
        return self._null_safe_equality(left, right)

    def _build_filtered_rows_sql(self, plan, columns_sql: str) -> str:
        """Build the final bucket lookup, scoped to the same partition."""
        bucketed_rows = plan["bucketed_rows"]
        bucket_list_name = plan["bucket_list_name"]
        bucket_col = plan["bucket_col"]
        filtered_rows = plan["filtered_rows"]
        partition_cols = plan["partition_cols"]

        if not partition_cols:
            return (
                f", {filtered_rows} AS(\n"
                f"    SELECT {columns_sql}\n"
                f"    FROM {bucketed_rows}\n"
                f"    WHERE {bucket_col} IN "
                f"(SELECT {bucket_col} FROM {bucket_list_name})\n"
                ")\n"
            )

        input_alias = plan["filtered_input_alias"]
        bucket_alias = plan["bucket_key_alias"]
        conditions = [f"{input_alias}.{bucket_col} = {bucket_alias}.{bucket_col}"]
        conditions.extend(
            self._partition_equality(
                plan, col, f"{input_alias}.{col}", f"{bucket_alias}.{col}"
            )
            for col in partition_cols
        )
        return (
            f", {filtered_rows} AS(\n"
            f"    SELECT {columns_sql}\n"
            f"    FROM {bucketed_rows} {input_alias}\n"
            f"    WHERE EXISTS (\n"
            f"        SELECT 1 FROM {bucket_list_name} {bucket_alias}\n"
            f"        WHERE {' AND '.join(conditions)}\n"
            f"    )\n"
            ")\n"
        )

    @staticmethod
    def _rebind_filtered_input(plan, filtered_rows: str) -> str:
        alias = plan["input_rebind_alias"]
        return f"{filtered_rows} {alias}" if alias is not None else filtered_rows

    @staticmethod
    def _offset_sql(base: str, offset: int) -> str:
        if offset > 0:
            return f"{base} + {offset}"
        if offset < 0:
            return f"{base} - {abs(offset)}"
        return base

    def _complete_input_columns(
        self, ast: RPRAbstractSyntaxTree
    ) -> Optional[List[str]]:
        if ast.table_schemas is None:
            return None
        # This existing AST helper verifies that every FROM table has schema.
        columns = ast._extract_join_source_columns()
        if not columns:
            columns = getattr(ast, "_preserved_input_columns", None)
        return list(columns) if columns is not None else None

    def _poisson_lambdas(self, ast: RPRAbstractSyntaxTree) -> Dict[str, float]:
        """Read only explicitly supplied/calculated lambdas; never invent one."""
        result = {}
        for name, value in self.poisson_lambdas.items():
            if value >= 0 and math.isfinite(value):
                result[name] = value
        values = (ast.table_schemas or {}).get("poisson_lambdas", {})
        if isinstance(values, dict):
            for name, value in values.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric >= 0 and math.isfinite(numeric):
                    result.setdefault(str(name).upper(), numeric)
        return result

    def _navigation_stays_within_pattern(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Whether DEFINE/MEASURES physical navigation stays in the match.

        Reuse the shared operand analysis so compound forms such as
        ``PREV(LAST(A.VAL), 1)`` are checked relative to ``A``, rather than the
        variable whose DEFINE clause happens to contain the expression.
        """
        return PatternNavigationReach(ast).all_navigation_stays_within_match()

    def _find_window(
        self, ast, left_var: str, right_var: str, order_col: str, sort_type: SortType
    ) -> Optional[float]:
        condition = ast.define.definitions.get(right_var)
        if condition is None:
            return None
        for expr in _split_conjuncts(condition.expression):
            if isinstance(expr, SemanticsExpr):
                expr = expr.expr
            parsed = self._parse_window_atom(
                expr, right_var, left_var, order_col, sort_type
            )
            if parsed is not None:
                return parsed
        return None

    def _parse_window_atom(
        self, expr, right_var: str, left_var: str, order_col: str, sort_type: SortType
    ):
        """Normalize ASC/DESC window predicates and their negated equivalents."""
        if not isinstance(expr, BinaryExpr) or expr.op not in ("<", "<=", ">", ">="):
            return None
        lhs, rhs = expr.left, expr.right
        if not isinstance(lhs, BinaryExpr) or lhs.op != "-":
            return None
        if not isinstance(rhs, Literal) or not isinstance(rhs.value, (int, float)):
            return None
        current_minus_left = self._is_order_ref(
            lhs.left, right_var, order_col
        ) and self._is_order_ref(lhs.right, left_var, order_col, require_qualifier=True)
        left_minus_current = self._is_order_ref(
            lhs.left, left_var, order_col, require_qualifier=True
        ) and self._is_order_ref(lhs.right, right_var, order_col)
        if not current_minus_left and not left_minus_current:
            return None

        # ASC canonical form is CURRENT - LEFT < w; DESC canonical form is
        # LEFT - CURRENT < w. The reverse subtraction with > -w is equivalent.
        canonical_less = (
            current_minus_left if sort_type == SortType.ASC else left_minus_current
        )
        value = float(rhs.value)
        if canonical_less and expr.op in ("<", "<="):
            window = value
        elif not canonical_less and expr.op in (">", ">="):
            window = -value
        else:
            return None
        return window if window > 0 else None

    @staticmethod
    def _is_order_ref(
        expr, var: str, col: str, require_qualifier: bool = False
    ) -> bool:
        if not isinstance(expr, ColumnRef) or expr.name != col:
            return False
        if require_qualifier:
            return expr.qualifier == var
        return expr.qualifier in (None, var)

    @staticmethod
    def _format_number(value: float) -> str:
        return str(int(value)) if value.is_integer() else str(value)


window_condition_pre_filter_test_cases = [
    (
        1,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS TS - A.TS < 100 AND OPEN >= 100
)""",
        """
WITH BUCKETED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL, FLOOR(TS / 100) AS BK_NO
    FROM NASDAQ
), BK_NO_LIST AS(
    SELECT BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
    UNION
    SELECT BK_NO + 1 AS BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
), FILTERED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL
    FROM BUCKETED_ROWS
    WHERE BK_NO IN (SELECT BK_NO FROM BK_NO_LIST)
)
SELECT * FROM FILTERED_ROWS MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS (TS - A.TS) < 100 AND OPEN >= 100
)""",
    ),
    # DESC canonical form: the left endpoint has the larger sort-column value.
    (
        2,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS DESC
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS A.TS - TS < 100 AND OPEN >= 100
)""",
        """
WITH BUCKETED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL, FLOOR(TS / 100) AS BK_NO
    FROM NASDAQ
), BK_NO_LIST AS(
    SELECT BK_NO - 1 AS BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
    UNION
    SELECT BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
), FILTERED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL
    FROM BUCKETED_ROWS
    WHERE BK_NO IN (SELECT BK_NO FROM BK_NO_LIST)
)
SELECT * FROM FILTERED_ROWS MATCH_RECOGNIZE(
    ORDER BY TS DESC
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS (A.TS - TS) < 100 AND OPEN >= 100
)""",
    ),
    # Algebraically equivalent DESC form: TS - A.TS > -100.
    (
        3,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS DESC
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS TS - A.TS > -100 AND OPEN >= 100
)""",
        """
WITH BUCKETED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL, FLOOR(TS / 100) AS BK_NO
    FROM NASDAQ
), BK_NO_LIST AS(
    SELECT BK_NO - 1 AS BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
    UNION
    SELECT BK_NO
    FROM BUCKETED_ROWS
    WHERE OPEN < 100
), FILTERED_ROWS AS(
    SELECT TICKER, OPEN, TS, VOL
    FROM BUCKETED_ROWS
    WHERE BK_NO IN (SELECT BK_NO FROM BK_NO_LIST)
)
SELECT * FROM FILTERED_ROWS MATCH_RECOGNIZE(
    ORDER BY TS DESC
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50, C AS (TS - A.TS) > -100 AND OPEN >= 100
)""",
    ),
    # B has only one guaranteed row to its left, so PREV(..., 2) is unsafe.
    (
        4,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50 AND PREV(B.OPEN, 2) > 0,
           C AS TS - A.TS < 100 AND OPEN >= 100
)""",
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50 AND PREV(B.OPEN, 2) > 0,
           C AS TS - A.TS < 100 AND OPEN >= 100
)""",
    ),
    # C is the right endpoint, hence even NEXT(..., 1) crosses the boundary.
    (
        5,
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50,
           C AS TS - A.TS < 100 AND OPEN >= 100 AND NEXT(C.OPEN) > 0
)""",
        """
SELECT * FROM NASDAQ MATCH_RECOGNIZE(
    ORDER BY TS
    ALL ROWS PER MATCH
    PATTERN(A B C)
    DEFINE A AS OPEN < 100, B AS OPEN > 50,
           C AS TS - A.TS < 100 AND OPEN >= 100 AND NEXT(C.OPEN) > 0
)""",
    ),
]


default_window_condition_pre_filter_table_schemas = {
    "NASDAQ": {
        "columns": ["TICKER", "OPEN", "TS", "VOL"],
    },
    "poisson_lambdas": {"A": 0.01, "B": 0.1, "C": 0.2},
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    result = run_rule_tests(
        WindowConditionPreFilterRule,
        window_condition_pre_filter_test_cases,
        "window condition pre-filter",
        verbose,
        default_window_condition_pre_filter_table_schemas,
    )

    return result


def test_sql():
    test_sql1 = """SELECT * FROM TA MATCH_RECOGNIZE( ORDER BY TS ALL ROWS PER MATCH PATTERN(A N C) DEFINE A AS OPEN < 100, N AS OPEN > 50, C AS TS - A.TS < 100 AND OPEN >= 100)"""
    test_sql2 = """SELECT * FROM TA MATCH_RECOGNIZE( PARTITION BY P1 ORDER BY TS MEASURES FIRST(TS) AS FIRST_TS PATTERN(A B?) DEFINE A AS A.X > 0, B AS B.X < 10 AND TS - A.TS < 100)"""
    test_sql3 = """SELECT * FROM TA MATCH_RECOGNIZE(PARTITION BY PT ORDER BY TS MEASURES FIRST(TS) AS FIRST_TS PATTERN(A M N D) DEFINE A AS X = 1, M AS X = 2, N AS X = 3, D AS D.TS - A.TS <= 100 AND X = 4)"""
    # Two-variable rewrite with a scalar dependency between A and C. This
    # produces both the regular independent-filter variant and an additional
    # A/C self-join variant that uses C.VOL = A.VOL.
    test_sql4 = """SELECT * FROM TA MATCH_RECOGNIZE(
        ORDER BY TS
        ALL ROWS PER MATCH
        PATTERN(A B C)
        DEFINE 
            A AS OPEN < 100,
            B AS OPEN > 50,
            C AS TS - A.TS < 100 AND OPEN >= 100 AND VOL = A.VOL
    )"""

    # The rewrite adds BK_NO and must subsequently project it away, so the
    # complete schema of every source table is required.
    schemas = {
        "TA": {
            "columns": ["TICKER", "OPEN", "TS", "VOL"],
            "column_types": {
                "TICKER": "VARCHAR(32)",
                "OPEN": "DOUBLE",
                "TS": "LONG",
                "VOL": "DOUBLE",
            },
        }
    }
    ast = RPRAbstractSyntaxTree()
    ast.parse_sql(test_sql4, schemas)

    rule = WindowConditionPreFilterRule(filter_variable_count=2)
    if rule.is_applicable(ast):
        statistics_sql, tracked_variables = rule.obtain_sta_sql(ast)
        print(f"statistics_sql:\n{statistics_sql}")
        print(f"tracked_variables: {tracked_variables}")

        # In production these values are calculated from the row returned by
        # statistics_sql, then written back before apply().
        ast.table_schemas["poisson_lambdas"] = {
            "A": 0.01,
            "B": 1.0,
            "C": 0.02,
        }
        variants = rule.generate_rewrite_variants(ast)
        for index, rewritten_sql in enumerate(variants, start=1):
            print(f"rewrite_variant_{index}:\n{rewritten_sql}")
        if not variants:
            print("Statistics are missing; no rewrite variant was generated")
    else:
        print("Rule not applicable")


if __name__ == "__main__":
    # result = run_tests(verbose=True)
    # print(f"{result['passed']} passed, {result['failed']} failed")

    test_sql()
