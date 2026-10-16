if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "rewriting_rules"

from .utils import *
from .column_pruning import ColumnPruningRule


class ColumnPruningViaJoinRule(RewriteRule):
    """
    Rewrite a restricted ALL ROWS PER MATCH query into:
      1. MATCH_RECOGNIZE over a narrow projection of the input table expression
      2. LEFT JOIN back to the full source table(s) by primary key
    """

    @property
    def name(self) -> str:
        return "column_pruning_via_join"

    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        return self._build_sql(ast) is not None

    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        rewritten_sql = self._build_sql(ast)
        if rewritten_sql is None:
            return False
        ast.raw_sql_override = rewritten_sql
        return True

    def _build_sql(self, ast: RPRAbstractSyntaxTree) -> Optional[str]:
        working_ast = copy.deepcopy(ast)
        cte_prefix = (working_ast.cte_prefix or "").strip()
        # The CTEs must be visible to both MATCH_RECOGNIZE and its join-back
        # relations. Rendering them inside MR would hide them from outer joins.
        working_ast.cte_prefix = None
        working_ast.raw_sql_override = None
        rewritten_sql = self._build_sql_body(working_ast)
        if rewritten_sql is None:
            return None
        return f"{cte_prefix}\n{rewritten_sql}" if cte_prefix else rewritten_sql

    def _build_sql_body(self, ast: RPRAbstractSyntaxTree) -> Optional[str]:
        if (
            ast.rows_per_match is None
            or ast.rows_per_match.strip().upper() != "ALL ROWS PER MATCH"
        ):
            return None
        if ast.subsets is not None and len(ast.subsets) > 0:
            return None
        if ast.output_names is None:
            return None
        if (
            ast.where is not None
            or ast.group_by is not None
            or ast.having is not None
            or ast.limit is not None
        ):
            return None
        if ast.input_table is None:
            return None
        table_schemas = self._normalize_table_schemas(ast.table_schemas)
        if table_schemas is None:
            return None

        working_ast = ast
        working_ast.optimize_from_clause()

        projection_source_sql = self._extract_projection_source_sql(
            working_ast.input_table
        )
        if projection_source_sql is None:
            return None

        source_tables = self._extract_source_tables(projection_source_sql)
        if source_tables is None or len(source_tables) == 0:
            return None
        if len(source_tables) > 1:
            return self._build_simple_multi_join_sql(
                working_ast, projection_source_sql, source_tables, table_schemas
            )

        derived_projection = self._parse_derived_projection(working_ast.input_table)
        if derived_projection is not None and not derived_projection["select_star"]:
            # Rebuilding from the base table would otherwise discard computed
            # expressions, aliases, and the derived table's output shape.
            return None
        if self._has_source_qualified_match_reference(working_ast, source_tables[0]):
            # The narrow derived table does not expose the original source
            # qualifier.  Decline instead of emitting out-of-scope references.
            return None

        business_cols = self._collect_business_columns(working_ast)
        source_plan = self._build_source_plan(
            working_ast,
            source_tables,
            table_schemas,
            projection_source_sql,
            business_cols,
        )
        if source_plan is None:
            return None

        all_columns = source_plan["all_columns"]
        pk_cols = source_plan["pk_cols"]
        if len(pk_cols) == 0 or len(all_columns) == 0:
            return None

        join_aliases = [pk_info["join_alias"] for pk_info in source_plan["pk_infos"]]
        existing_match_no_alias = self._find_match_number_alias(working_ast)
        reserved_names = set(join_aliases)
        generated_match_no_alias = choose_generated_column_name(
            working_ast,
            "RPR_MATCH_NO",
            reserved_names
            | set(
                working_ast.measures.measures.keys()
                if working_ast.measures is not None
                else []
            ),
        )
        if existing_match_no_alias is None:
            reserved_names.add(generated_match_no_alias)
        if self._columns_conflict(working_ast, reserved_names):
            return None

        inner_ast = copy.deepcopy(working_ast)
        inner_ast.raw_sql_override = None
        inner_ast.output_names = ExprParser().parse_select("SELECT *")
        inner_ast.where = None
        inner_ast.group_by = None
        inner_ast.having = None
        inner_ast.order_by = None
        inner_ast.limit = None
        inner_ast.alias = None
        if inner_ast.measures is None:
            inner_ast.measures = MeasuresSubClause({})

        projection_input_sql = source_plan["projection_input_sql"]
        label_plan = self._build_label_plan(
            inner_ast,
            source_plan["pk_projected_cols"],
            projection_input_sql,
            source_plan["pk_projection_items"],
        )
        if label_plan is not None:
            inner_ast.input_table = label_plan["input_table"]
            inner_ast.define.definitions = label_plan["definitions"]
            narrow_cols = label_plan["projected_columns"]
        else:
            partition_cols = self._normalize_column_sequence(
                inner_ast.partition_by or []
            )
            order_cols = self._normalize_column_sequence(
                [col for col, _ in (inner_ast.match_order_by or [])]
            )
            measure_define_cols = sorted(
                col.upper()
                for col in (
                    inner_ast.measures.get_columns() | inner_ast.define.get_columns()
                )
            )
            narrow_cols = self._ordered_union(
                source_plan["pk_projected_cols"],
                partition_cols,
                order_cols,
                measure_define_cols,
            )
            select_items = self._build_narrow_projection_items(
                narrow_cols,
                source_plan["pk_projected_cols"],
                source_plan["pk_projection_items"],
            )
            inner_ast.input_table = (
                f"(SELECT {', '.join(select_items)} FROM {projection_input_sql})"
            )

        if len(
            [
                col
                for col in narrow_cols
                if col.upper() in {c.upper() for c in all_columns}
            ]
        ) == len(all_columns):
            return None

        partition_cols = self._normalize_column_sequence(inner_ast.partition_by or [])
        order_cols = self._normalize_column_sequence(
            [col for col, _ in (inner_ast.match_order_by or [])]
        )
        original_measure_aliases = (
            [alias.upper() for alias in working_ast.measures.measures.keys()]
            if working_ast.measures is not None
            else []
        )

        if self._can_use_direct_projection_prune(
            working_ast, partition_cols, order_cols, original_measure_aliases
        ):
            direct_ast = copy.deepcopy(working_ast)
            direct_ast.raw_sql_override = None
            if label_plan is not None:
                direct_ast.input_table = label_plan["input_table"]
                direct_ast.define.definitions = label_plan["definitions"]
            else:
                select_items = self._build_narrow_projection_items(
                    narrow_cols,
                    source_plan["pk_projected_cols"],
                    source_plan["pk_projection_items"],
                )
                direct_ast.input_table = (
                    f"(SELECT {', '.join(select_items)} FROM {projection_input_sql})"
                )
            return direct_ast.to_sql()

        reusable_pk_aliases = self._find_reusable_pk_measure_aliases(
            working_ast, source_plan["pk_infos"]
        )
        new_measures = dict(inner_ast.measures.measures)
        for pk_info in source_plan["pk_infos"]:
            if pk_info["join_alias"] not in reusable_pk_aliases:
                new_measures[pk_info["join_alias"]] = ColumnRef(
                    None, pk_info["projected_column"]
                )
        match_no_alias = (
            existing_match_no_alias.upper()
            if existing_match_no_alias is not None
            else generated_match_no_alias
        )
        if existing_match_no_alias is None:
            new_measures[match_no_alias] = FunctionCall("MATCH_NUMBER", [])
        inner_ast.measures.measures = new_measures

        relation_names = {
            source["join_table_alias"] for source in source_plan["sources"]
        }
        mr_alias = choose_generated_relation_name(working_ast, "MR", relation_names)
        mr_sql = self._format_match_recognize_relation(inner_ast, mr_alias)

        remaining_col_infos = [
            col_info
            for col_info in source_plan["column_infos"]
            if col_info["column"].upper()
            not in {c.upper() for c in partition_cols + order_cols}
        ]
        remaining_cols = [col_info["column"] for col_info in remaining_col_infos]
        joined_select_cols = []
        joined_select_cols.extend(
            f"{mr_alias}.{col} AS {col}" for col in partition_cols
        )
        joined_select_cols.extend(f"{mr_alias}.{col} AS {col}" for col in order_cols)
        joined_select_cols.extend(
            f"{mr_alias}.{alias} AS {alias}" for alias in original_measure_aliases
        )
        joined_select_cols.extend(
            f"{col_info['join_table_alias']}.{col_info['column']} AS {col_info['column']}"
            for col_info in remaining_col_infos
        )

        order_parts = self._build_default_outer_order_parts(
            working_ast, mr_alias, match_no_alias
        )

        joined_sql = "FROM " + mr_sql
        for source in source_plan["sources"]:
            join_conditions = [
                f"{source['join_table_alias']}.{pk_info['column']} = {mr_alias}.{reusable_pk_aliases.get(pk_info['join_alias'], pk_info['join_alias'])}"
                for pk_info in source["pk_infos"]
            ]
            joined_sql += (
                f"\nLEFT JOIN {source['table']} {source['join_table_alias']}\n"
                "ON " + " AND ".join(join_conditions)
            )
        if working_ast.output_names.to_string().strip().upper() == "SELECT *":
            outer_select = "SELECT " + ", ".join(joined_select_cols)
        else:
            outer_select = self._format_explicit_outer_select(
                working_ast,
                mr_alias,
                source_plan,
                partition_cols,
                order_cols,
                original_measure_aliases,
            )
            if outer_select is None:
                return None
        outer_sql = outer_select + "\n" + joined_sql
        if working_ast.order_by is not None and len(working_ast.order_by) > 0:
            formatted_order_by = self._format_outer_order_by(
                working_ast,
                mr_alias,
                source_plan,
                partition_cols,
                order_cols,
                original_measure_aliases,
            )
            if formatted_order_by is None:
                return None
            outer_sql += "\nORDER BY " + formatted_order_by
        else:
            outer_sql += "\nORDER BY " + ", ".join(order_parts)
        return outer_sql

    def _outer_column_qualifier(
        self,
        column_name: str,
        mr_alias: str,
        source_plan: dict,
        partition_cols: List[str],
        order_cols: List[str],
        measure_aliases: List[str],
    ) -> Optional[str]:
        """Return the relation that owns an original MATCH_RECOGNIZE output column."""
        norm = self._normalize_column_name(column_name)
        if norm is None:
            return None

        mr_output_cols = {
            col.upper() for col in partition_cols + order_cols + measure_aliases
        }
        if norm in mr_output_cols:
            return mr_alias

        owners = {
            col_info["join_table_alias"]
            for col_info in source_plan["column_infos"]
            if col_info["column"].upper() == norm
        }
        if len(owners) != 1:
            return None
        return next(iter(owners))

    def _qualify_outer_expr(
        self,
        expr: Expr,
        mr_alias: str,
        source_plan: dict,
        partition_cols: List[str],
        order_cols: List[str],
        measure_aliases: List[str],
    ) -> bool:
        if isinstance(expr, ColumnRef):
            # A qualified star cannot be rebound safely after the original
            # MATCH_RECOGNIZE relation is split into MR and the full table.
            if expr.name == "*":
                return expr.qualifier is None
            qualifier = self._outer_column_qualifier(
                expr.name,
                mr_alias,
                source_plan,
                partition_cols,
                order_cols,
                measure_aliases,
            )
            if qualifier is None:
                return False
            expr.qualifier = qualifier
            return True

        for value in vars(expr).values():
            if isinstance(value, Expr):
                if not self._qualify_outer_expr(
                    value,
                    mr_alias,
                    source_plan,
                    partition_cols,
                    order_cols,
                    measure_aliases,
                ):
                    return False
            elif isinstance(value, (list, tuple)):
                for item in value:
                    children = item if isinstance(item, tuple) else (item,)
                    for child in children:
                        if isinstance(child, Expr) and not self._qualify_outer_expr(
                            child,
                            mr_alias,
                            source_plan,
                            partition_cols,
                            order_cols,
                            measure_aliases,
                        ):
                            return False
        return True

    def _format_explicit_outer_select(
        self,
        ast: RPRAbstractSyntaxTree,
        mr_alias: str,
        source_plan: dict,
        partition_cols: List[str],
        order_cols: List[str],
        measure_aliases: List[str],
    ) -> Optional[str]:
        select_clause = copy.deepcopy(ast.output_names)
        for expr, _ in select_clause.select_items:
            if not self._qualify_outer_expr(
                expr,
                mr_alias,
                source_plan,
                partition_cols,
                order_cols,
                measure_aliases,
            ):
                return None
        return select_clause.to_string()

    def _format_outer_order_by(
        self,
        ast: RPRAbstractSyntaxTree,
        mr_alias: str,
        source_plan: dict,
        partition_cols: List[str],
        order_cols: List[str],
        measure_aliases: List[str],
    ) -> Optional[str]:
        select_aliases = {
            alias.upper()
            for _, alias in ast.output_names.select_items
            if alias is not None
        }
        parts = []
        for col_name, sort_type in ast.order_by:
            norm = self._normalize_column_name(col_name)
            if norm is None:
                return None
            if norm in select_aliases:
                rendered = norm
            else:
                qualifier = self._outer_column_qualifier(
                    norm,
                    mr_alias,
                    source_plan,
                    partition_cols,
                    order_cols,
                    measure_aliases,
                )
                if qualifier is None:
                    return None
                rendered = f"{qualifier}.{norm}"
            if sort_type != SortType.ASC:
                rendered += " DESC"
            parts.append(rendered)
        return ", ".join(parts)

    def _normalize_table_schemas(
        self, table_schemas: Optional[Dict]
    ) -> Optional[Dict[str, Dict[str, List[str]]]]:
        if table_schemas is None:
            return None

        normalized = {}
        for table_name, meta in table_schemas.items():
            if not isinstance(meta, dict):
                continue
            normalized[table_name.upper()] = {
                "primary_key": [col.upper() for col in meta.get("primary_key", [])],
                "columns": RPRAbstractSyntaxTree._schema_column_names(
                    meta.get("columns", [])
                ),
            }
        return normalized

    def _format_match_recognize_relation(
        self, ast: RPRAbstractSyntaxTree, alias: str
    ) -> str:
        full_sql = ast.to_sql().strip()
        prefix = ast.output_names.to_string().strip()
        if full_sql.upper().startswith(prefix.upper()):
            relation_sql = full_sql[len(prefix) :].strip()
            if relation_sql.upper().startswith("FROM "):
                relation_sql = relation_sql[len("FROM ") :].strip()
            return relation_sql + f" {alias}"
        return "(\n" + full_sql + f"\n) {alias}"

    def _find_reusable_pk_measure_aliases(
        self, ast: RPRAbstractSyntaxTree, pk_infos: List[dict]
    ) -> Dict[str, str]:
        if ast.measures is None:
            return {}

        result = {}
        for pk_info in pk_infos:
            pk_col = pk_info["projected_column"].upper()
            for alias, expr in ast.measures.measures.items():
                if isinstance(expr, SemanticsExpr):
                    if expr.semantics is not None and expr.semantics.upper() == "FINAL":
                        continue
                    target = expr.expr
                else:
                    target = expr
                if not isinstance(target, ColumnRef):
                    continue
                if target.qualifier is not None:
                    continue
                if target.name.upper() != pk_col:
                    continue
                result[pk_info["join_alias"]] = alias.upper()
                break
        return result

    def _collect_business_columns(self, ast: RPRAbstractSyntaxTree) -> Set[str]:
        columns: Set[str] = set()
        if ast.partition_by is not None:
            columns.update(self._normalize_column_sequence(ast.partition_by))
        if ast.match_order_by is not None:
            columns.update(
                self._normalize_column_sequence([col for col, _ in ast.match_order_by])
            )
        if ast.measures is not None:
            columns.update(
                norm
                for norm in (
                    self._normalize_column_name(col)
                    for col in ast.measures.get_columns()
                )
                if norm is not None
            )
        columns.update(
            norm
            for norm in (
                self._normalize_column_name(col) for col in ast.define.get_columns()
            )
            if norm is not None
        )
        return columns

    def _has_source_qualified_match_reference(
        self, ast: RPRAbstractSyntaxTree, source: dict
    ) -> bool:
        qualifiers = {source["table"].upper()}
        if source.get("alias") is not None:
            qualifiers.add(source["alias"].upper())

        column_refs: Set[str] = set(ast.partition_by or [])
        column_refs.update(col for col, _ in (ast.match_order_by or []))
        if ast.measures is not None:
            column_refs.update(ast.measures.get_columns())
        column_refs.update(ast.define.get_columns())

        for column_ref in column_refs:
            parts = column_ref.strip().upper().split(".", 1)
            if len(parts) == 2 and parts[0] in qualifiers:
                return True
        return False

    def _build_narrow_projection_items(
        self,
        narrow_cols: List[str],
        pk_projected_cols: List[str],
        pk_projection_items: List[str],
    ) -> List[str]:
        pk_col_set = {col.upper() for col in pk_projected_cols}
        result = list(pk_projection_items)
        result.extend(col for col in narrow_cols if col.upper() not in pk_col_set)
        return result

    def _can_use_direct_projection_prune(
        self,
        ast: RPRAbstractSyntaxTree,
        partition_cols: List[str],
        order_cols: List[str],
        measure_aliases: List[str],
    ) -> bool:
        if (
            ast.output_names is None
            or ast.output_names.to_string().strip().upper() == "SELECT *"
        ):
            return False
        if any(
            isinstance(expr, ColumnRef) and expr.name == "*"
            for expr, _ in ast.output_names.select_items
        ):
            return False

        allowed = {col.upper() for col in partition_cols + order_cols + measure_aliases}
        selected_cols = {
            self._normalize_column_name(col) for col in ast.output_names.get_columns()
        }
        selected_cols.discard(None)
        if any(col not in allowed for col in selected_cols):
            return False

        if ast.order_by is not None:
            order_cols_used = {
                self._normalize_column_name(col) for col, _ in ast.order_by
            }
            order_cols_used.discard(None)
            if any(col not in allowed for col in order_cols_used):
                return False

        return True

    def _find_match_number_alias(self, ast: RPRAbstractSyntaxTree) -> Optional[str]:
        if ast.measures is None:
            return None
        for alias, expr in ast.measures.measures.items():
            target = expr.expr if isinstance(expr, SemanticsExpr) else expr
            if (
                isinstance(target, FunctionCall)
                and target.name.upper() == "MATCH_NUMBER"
                and len(target.args) == 0
            ):
                return alias
        return None

    def _extract_projection_source_sql(self, input_table_sql: str) -> Optional[str]:
        sql = input_table_sql.strip()
        if len(sql) == 0:
            return None
        if not sql.startswith("("):
            return sql
        if not sql.endswith(")"):
            return None

        inner = sql[1:-1].strip()
        upper_inner = inner.upper()
        prefix = "SELECT * FROM "
        if not upper_inner.startswith(prefix):
            projection = self._parse_derived_projection(sql)
            return projection["source_sql"] if projection is not None else None
        return inner[len(prefix) :].strip()

    def _parse_derived_projection(self, input_table_sql: str) -> Optional[dict]:
        sql = input_table_sql.strip()
        if not sql.startswith("(") or not sql.endswith(")"):
            return None
        inner = sql[1:-1].strip()
        if not inner.upper().startswith("SELECT "):
            return None
        from_pos = self._find_top_level_keyword(inner, "FROM", start=len("SELECT "))
        if from_pos == -1:
            return None
        select_sql = inner[len("SELECT ") : from_pos].strip()
        source_sql = inner[from_pos + len("FROM") :].strip()
        items = self._split_top_level_commas(select_sql)
        return {
            "select_sql": select_sql,
            "source_sql": source_sql,
            "select_star": len(items) == 1 and items[0].strip() == "*",
            "items": [self._parse_projection_item(item) for item in items],
        }

    def _find_top_level_keyword(self, sql: str, keyword: str, start: int = 0) -> int:
        keyword_upper = keyword.upper()
        depth = 0
        idx = start
        while idx <= len(sql) - len(keyword):
            ch = sql[idx]
            if ch == "(":
                depth += 1
                idx += 1
                continue
            if ch == ")":
                depth -= 1
                idx += 1
                continue
            if depth == 0 and sql[idx : idx + len(keyword)].upper() == keyword_upper:
                before = sql[idx - 1] if idx > 0 else " "
                after_idx = idx + len(keyword)
                after = sql[after_idx] if after_idx < len(sql) else " "
                if not (before.isalnum() or before == "_") and not (
                    after.isalnum() or after == "_"
                ):
                    return idx
            idx += 1
        return -1

    def _split_top_level_commas(self, sql: str) -> List[str]:
        parts = []
        depth = 0
        start = 0
        for idx, ch in enumerate(sql):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(sql[start:idx].strip())
                start = idx + 1
        parts.append(sql[start:].strip())
        return [part for part in parts if len(part) > 0]

    def _parse_projection_item(self, item_sql: str) -> dict:
        item_sql = item_sql.strip()
        match = re.match(
            r"(.+?)\s+AS\s+([A-Z_][A-Z0-9_$#]*)$", item_sql, flags=re.IGNORECASE
        )
        if match:
            expr_sql = match.group(1).strip()
            alias = match.group(2).upper()
        else:
            pieces = item_sql.rsplit(" ", 1)
            if len(pieces) == 2 and re.match(
                r"^[A-Z_][A-Z0-9_$#]*$", pieces[1], flags=re.IGNORECASE
            ):
                expr_sql = pieces[0].strip()
                alias = pieces[1].upper()
            else:
                expr_sql = item_sql
                alias = None

        col_match = re.match(
            r"^(?:(?P<qual>[A-Z_][A-Z0-9_$#]*)\.)?(?P<col>[A-Z_][A-Z0-9_$#]*)$",
            expr_sql,
            flags=re.IGNORECASE,
        )
        qualifier = (
            col_match.group("qual").upper()
            if col_match and col_match.group("qual")
            else None
        )
        source_col = col_match.group("col").upper() if col_match else None
        output_col = (alias or source_col or expr_sql).upper()
        return {
            "sql": item_sql,
            "expr_sql": expr_sql,
            "alias": alias,
            "output_col": output_col,
            "qualifier": qualifier,
            "source_col": source_col,
        }

    def _extract_source_tables(self, input_table_sql: str) -> Optional[List[dict]]:
        sql = input_table_sql.strip()
        if len(sql) == 0 or sql.startswith("("):
            return None
        if (
            self._find_top_level_keyword(sql, "SELECT") != -1
            or self._find_top_level_keyword(sql, "UNION") != -1
        ):
            return None

        tokens = sql.replace(",", " , ").split()
        join_modifiers = {"INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL"}
        join_keywords = {"JOIN", ","}
        clause_keywords = {"ON", "USING"}
        trailing_clause_keywords = {"WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"}
        sources: List[dict] = []

        idx = 0
        expect_table = True
        skip_join_clause = False
        paren_depth = 0

        while idx < len(tokens):
            token = tokens[idx]
            upper_token = token.upper()
            paren_depth += token.count("(") - token.count(")")

            if paren_depth <= 0 and upper_token in trailing_clause_keywords:
                break

            if skip_join_clause:
                if paren_depth <= 0 and upper_token in trailing_clause_keywords:
                    break
                if paren_depth <= 0 and (
                    upper_token in join_modifiers or upper_token in join_keywords
                ):
                    skip_join_clause = False
                    if upper_token in join_keywords:
                        expect_table = True
                    continue
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
                alias = None
                idx += 1
                if idx < len(tokens) and tokens[idx].upper() == "AS":
                    if idx + 1 >= len(tokens):
                        return None
                    alias = tokens[idx + 1]
                    idx += 2
                elif idx < len(tokens):
                    next_token = tokens[idx].upper()
                    if (
                        next_token not in join_modifiers
                        and next_token not in join_keywords
                        and next_token not in clause_keywords
                        and next_token not in trailing_clause_keywords
                    ):
                        alias = tokens[idx]
                        idx += 1

                sources.append(
                    {
                        "table": table_name.upper(),
                        "alias": alias.upper() if alias else None,
                    }
                )
                expect_table = False
                continue

            idx += 1

        return sources if len(sources) > 0 else None

    def _build_source_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        source_tables: List[dict],
        table_schemas: Dict[str, Dict[str, List[str]]],
        input_table_sql: str,
        business_cols: Set[str],
    ) -> Optional[dict]:
        sources = []
        all_columns: List[str] = []
        column_infos: List[dict] = []
        pk_cols: List[str] = []
        pk_projected_cols: List[str] = []
        pk_projection_items: List[str] = []
        pk_infos: List[dict] = []
        column_counts: Dict[str, int] = {}
        pk_idx = 1
        single_source = len(source_tables) == 1
        generated_names = {
            col.upper()
            for schema in table_schemas.values()
            for col in schema.get("columns", [])
        }

        def allocate(base: str) -> str:
            candidate = base.upper()
            suffix = 1
            while candidate in generated_names:
                candidate = f"{base.upper()}_{suffix}"
                suffix += 1
            generated_names.add(candidate)
            return candidate

        used_relation_names = collect_used_relation_names(ast)

        def allocate_relation(base: str) -> str:
            candidate = choose_generated_relation_name(ast, base, used_relation_names)
            used_relation_names.add(candidate.upper())
            return candidate

        for source_idx, source in enumerate(source_tables):
            table_name = source["table"].upper()
            schema = table_schemas.get(table_name)
            if schema is None:
                return None

            table_pk_cols = schema.get("primary_key", [])
            table_columns = schema.get("columns", [])
            if len(table_pk_cols) == 0 or len(table_columns) == 0:
                return None

            source_pk_infos = []
            join_table_alias = allocate_relation(
                "T_FULL" if single_source else f"T_FULL_{source_idx + 1}"
            )
            table_ref = source.get("alias") or table_name
            for pk_col_idx, pk_col in enumerate(table_pk_cols):
                norm_pk = pk_col.upper()
                if norm_pk not in {col.upper() for col in table_columns}:
                    return None
                join_alias = allocate(f"RPR_JOIN_PK_{pk_idx}")
                projected_column = (
                    norm_pk
                    if single_source
                    else allocate(f"RPR_SRC_{source_idx + 1}_PK_{pk_col_idx + 1}")
                )
                projection_item = (
                    norm_pk
                    if single_source
                    else f"{table_ref}.{norm_pk} AS {projected_column}"
                )
                pk_idx += 1
                pk_cols.append(norm_pk)
                pk_projected_cols.append(projected_column)
                pk_projection_items.append(projection_item)
                pk_info = {
                    "table": table_name,
                    "column": norm_pk,
                    "projected_column": projected_column,
                    "projection_item": projection_item,
                    "join_alias": join_alias,
                    "join_table_alias": join_table_alias,
                }
                pk_infos.append(pk_info)
                source_pk_infos.append(pk_info)

            for col in table_columns:
                norm_col = col.upper()
                column_counts[norm_col] = column_counts.get(norm_col, 0) + 1
                if column_counts[norm_col] > 1 and norm_col in business_cols:
                    return None
                all_columns.append(norm_col)
                column_infos.append(
                    {
                        "table": table_name,
                        "column": norm_col,
                        "join_table_alias": join_table_alias,
                    }
                )

            sources.append(
                {
                    "table": table_name,
                    "alias": source.get("alias"),
                    "join_table_alias": join_table_alias,
                    "pk_infos": source_pk_infos,
                }
            )

        return {
            "sources": sources,
            "projection_input_sql": self._build_projection_input_sql(
                source_tables, input_table_sql
            ),
            "all_columns": all_columns,
            "column_infos": column_infos,
            "pk_cols": pk_cols,
            "pk_projected_cols": pk_projected_cols,
            "pk_projection_items": pk_projection_items,
            "pk_infos": pk_infos,
        }

    def _build_projection_input_sql(
        self, source_tables: List[dict], input_table_sql: str
    ) -> str:
        if len(source_tables) != 1:
            return input_table_sql

        # Dropping an input wrapper is safe only for a plain table reference
        # (optionally carrying an ordinary correlation name).  In particular,
        # the absence of WHERE does not mean that the remaining text is
        # row-preserving: ORDER BY ... FETCH, LIMIT, OFFSET, GROUP BY, HAVING,
        # QUALIFY, and other clauses can all change the rows seen by
        # MATCH_RECOGNIZE.  Keep every non-trivial FROM tail verbatim so the
        # narrow projection is evaluated over exactly the original input.
        identifier = r"[A-Z_][A-Z0-9_$#]*"
        plain_relation = re.fullmatch(
            rf"\s*({identifier}(?:\.{identifier}){{0,2}})"
            rf"(?:\s+(?:AS\s+)?{identifier})?\s*",
            input_table_sql,
            flags=re.IGNORECASE,
        )
        if plain_relation is not None:
            return plain_relation.group(1)
        return input_table_sql

    def _build_simple_multi_join_sql(
        self,
        ast: RPRAbstractSyntaxTree,
        projection_source_sql: str,
        source_tables: List[dict],
        table_schemas: Dict[str, Dict[str, List[str]]],
    ) -> Optional[str]:
        projection = self._parse_derived_projection(ast.input_table)
        if projection is None:
            projection = {
                "select_star": True,
                "items": [],
                "source_sql": projection_source_sql,
            }
        if len(source_tables) != 2:
            return None
        join_info = self._parse_simple_two_table_join(
            projection_source_sql, source_tables
        )
        if join_info is None:
            return None

        table_refs = self._build_table_refs(source_tables, table_schemas)
        if table_refs is None:
            return None
        if not self._all_join_conditions_cross_sources(join_info, table_refs):
            return None
        duplicate_cols = self._find_duplicate_source_columns(
            source_tables, table_schemas
        )
        if projection.get("select_star") and len(duplicate_cols) > 0:
            raise ValueError(
                "SELECT * over JOIN has duplicate column names: "
                + ", ".join(sorted(duplicate_cols))
            )

        projected_items = self._expand_projection_items(
            projection, source_tables, table_schemas, table_refs
        )
        if projected_items is None:
            return None
        if (
            ast.output_names.to_string().strip().upper() != "SELECT *"
            or (ast.measures is not None and len(ast.measures.measures) > 0)
            or (ast.order_by is not None and len(ast.order_by) > 0)
        ):
            # The multi-table builders reconstruct the input projection only;
            # they do not yet preserve explicit output lists, user measures,
            # or a top-level ORDER BY.
            return None

        business_cols = self._collect_business_columns(ast)
        column_owner = {item["output_col"]: item for item in projected_items}
        referenced_items = []
        for col in business_cols:
            item = column_owner.get(col.upper())
            if item is None:
                item = self._resolve_source_column_item(
                    col, source_tables, table_schemas, table_refs
                )
            if item is None:
                item = self._resolve_join_equivalent_column_item(
                    col, join_info, table_refs
                )
            if item is None:
                return None
            referenced_items.append(item)
        if any(item["output_col"] != item["source_col"] for item in referenced_items):
            # MATCH_RECOGNIZE sees derived-table output names.  The optimized
            # branches run over base tables, where renamed columns do not exist.
            return None
        referenced_sources = {item["source_idx"] for item in referenced_items}

        if len(referenced_sources) == 1:
            source_idx = next(iter(referenced_sources))
            preserved_idx = self._preserved_source_index(join_info)
            if preserved_idx == source_idx and self._can_postpone_outer_join(
                source_idx, join_info, table_refs
            ):
                return self._build_preserved_outer_join_sql(
                    ast,
                    source_idx,
                    join_info,
                    source_tables,
                    table_refs,
                    projected_items,
                )
            if self._can_post_join_by_semijoin(source_idx, join_info, table_refs):
                return self._build_semijoin_sql(
                    ast,
                    source_idx,
                    join_info,
                    source_tables,
                    table_refs,
                    projected_items,
                )

        if projection.get("select_star"):
            return None
        if join_info["join_type"] in {"INNER", "COMMA"}:
            return self._build_join_label_sql(
                ast,
                join_info,
                source_tables,
                table_refs,
                table_schemas,
                projected_items,
            )
        return None

    def _build_table_refs(
        self, source_tables: List[dict], table_schemas: Dict[str, Dict[str, List[str]]]
    ) -> Optional[List[dict]]:
        refs = []
        for idx, source in enumerate(source_tables):
            table_name = source["table"].upper()
            schema = table_schemas.get(table_name)
            if schema is None:
                return None
            columns = schema.get("columns", [])
            pk = schema.get("primary_key", [])
            if len(columns) == 0 or len(pk) == 0:
                return None
            ref = source.get("alias") or table_name
            refs.append(
                {
                    "idx": idx,
                    "table": table_name,
                    "ref": ref.upper(),
                    "columns": [col.upper() for col in columns],
                    "primary_key": [col.upper() for col in pk],
                }
            )
        return refs

    def _find_duplicate_source_columns(
        self,
        source_tables: List[dict],
        table_schemas: Dict[str, Dict[str, List[str]]],
    ) -> Set[str]:
        counts: Dict[str, int] = {}
        for source in source_tables:
            schema = table_schemas.get(source["table"].upper(), {})
            for col in schema.get("columns", []):
                norm = col.upper()
                counts[norm] = counts.get(norm, 0) + 1
        return {col for col, count in counts.items() if count > 1}

    def _all_join_conditions_cross_sources(
        self, join_info: dict, table_refs: List[dict]
    ) -> bool:
        if len(table_refs) != 2 or len(join_info["conditions"]) == 0:
            return False

        left_names = {table_refs[0]["ref"], table_refs[0]["table"]}
        right_names = {table_refs[1]["ref"], table_refs[1]["table"]}
        for condition in join_info["conditions"]:
            crosses_left_to_right = (
                condition["left_ref"] in left_names
                and condition["right_ref"] in right_names
            )
            crosses_right_to_left = (
                condition["left_ref"] in right_names
                and condition["right_ref"] in left_names
            )
            if not (crosses_left_to_right or crosses_right_to_left):
                return False
        return True

    def _parse_simple_two_table_join(
        self, source_sql: str, source_tables: List[dict]
    ) -> Optional[dict]:
        upper_sql = source_sql.upper()
        if re.search(r"\bFULL\s+(?:OUTER\s+)?JOIN\b", upper_sql):
            return None
        join_type = "INNER"
        if re.search(r"\bLEFT\s+(?:OUTER\s+)?JOIN\b", upper_sql):
            join_type = "LEFT"
        elif re.search(r"\bRIGHT\s+(?:OUTER\s+)?JOIN\b", upper_sql):
            join_type = "RIGHT"
        elif "," in upper_sql and " WHERE " in upper_sql:
            join_type = "COMMA"

        condition_sql = None
        if join_type == "COMMA":
            where_pos = self._find_top_level_keyword(source_sql, "WHERE")
            if where_pos == -1:
                return None
            condition_sql = source_sql[where_pos + len("WHERE") :].strip()
        else:
            on_pos = self._find_top_level_keyword(source_sql, "ON")
            if on_pos != -1:
                condition_sql = source_sql[on_pos + len("ON") :].strip()
            elif self._find_top_level_keyword(source_sql, "WHERE") != -1:
                where_pos = self._find_top_level_keyword(source_sql, "WHERE")
                condition_sql = source_sql[where_pos + len("WHERE") :].strip()
                join_type = "COMMA"
            else:
                return None
        eqs = self._parse_equi_join_conditions(condition_sql)
        if len(eqs) == 0:
            return None
        return {
            "join_type": join_type,
            "condition_sql": condition_sql,
            "conditions": eqs,
            "left_idx": 0,
            "right_idx": 1,
        }

    def _parse_equi_join_conditions(self, condition_sql: str) -> List[dict]:
        result = []
        for part in re.split(r"\s+AND\s+", condition_sql, flags=re.IGNORECASE):
            match = re.match(
                r"^\s*([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\s*=\s*([A-Z_][A-Z0-9_$#]*)\.([A-Z_][A-Z0-9_$#]*)\s*$",
                part.strip(),
                flags=re.IGNORECASE,
            )
            if match is None:
                return []
            result.append(
                {
                    "left_ref": match.group(1).upper(),
                    "left_col": match.group(2).upper(),
                    "right_ref": match.group(3).upper(),
                    "right_col": match.group(4).upper(),
                }
            )
        return result

    def _expand_projection_items(
        self,
        projection: dict,
        source_tables: List[dict],
        table_schemas: Dict[str, Dict[str, List[str]]],
        table_refs: List[dict],
    ) -> Optional[List[dict]]:
        if projection.get("select_star"):
            items = []
            for ref in table_refs:
                for col in ref["columns"]:
                    items.append(
                        {
                            "output_col": col,
                            "source_col": col,
                            "source_idx": ref["idx"],
                            "qualifier": ref["ref"],
                        }
                    )
            return items

        result = []
        seen = set()
        for item in projection["items"]:
            if item["source_col"] is None:
                return None
            owner = self._resolve_projection_owner(item, table_refs)
            if owner is None:
                return None
            output_col = item["output_col"].upper()
            if output_col in seen:
                raise ValueError(
                    "Derived table has duplicate column name: " + output_col
                )
            seen.add(output_col)
            result.append(
                {
                    "output_col": output_col,
                    "source_col": item["source_col"].upper(),
                    "source_idx": owner["idx"],
                    "qualifier": owner["ref"],
                }
            )
        return result

    def _resolve_projection_owner(
        self, item: dict, table_refs: List[dict]
    ) -> Optional[dict]:
        col = item["source_col"].upper()
        qualifier = item.get("qualifier")
        if qualifier is not None:
            for ref in table_refs:
                if qualifier in {ref["ref"], ref["table"]} and col in ref["columns"]:
                    return ref
            return None
        candidates = [ref for ref in table_refs if col in ref["columns"]]
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _resolve_source_column_item(
        self,
        col: str,
        source_tables: List[dict],
        table_schemas: Dict[str, Dict[str, List[str]]],
        table_refs: List[dict],
    ) -> Optional[dict]:
        norm = self._normalize_column_name(col)
        if norm is None:
            return None
        candidates = [ref for ref in table_refs if norm in ref["columns"]]
        if len(candidates) != 1:
            return None
        owner = candidates[0]
        return {
            "output_col": norm,
            "source_col": norm,
            "source_idx": owner["idx"],
            "qualifier": owner["ref"],
        }

    def _resolve_join_equivalent_column_item(
        self, col: str, join_info: dict, table_refs: List[dict]
    ) -> Optional[dict]:
        norm = self._normalize_column_name(col)
        if norm is None:
            return None
        ref_to_idx = {}
        for ref in table_refs:
            ref_to_idx[ref["ref"]] = ref["idx"]
            ref_to_idx[ref["table"]] = ref["idx"]
        for cond in join_info["conditions"]:
            if cond["left_col"] == norm and cond["right_col"] == norm:
                source_idx = ref_to_idx.get(cond["left_ref"])
                if source_idx is None:
                    return None
                return {
                    "output_col": norm,
                    "source_col": norm,
                    "source_idx": source_idx,
                    "qualifier": table_refs[source_idx]["ref"],
                }
        return None

    def _preserved_source_index(self, join_info: dict) -> Optional[int]:
        if join_info["join_type"] == "LEFT":
            return join_info["left_idx"]
        if join_info["join_type"] == "RIGHT":
            return join_info["right_idx"]
        return None

    def _can_postpone_outer_join(
        self,
        preserved_idx: int,
        join_info: dict,
        table_refs: List[dict],
    ) -> bool:
        """
        Moving an outer join after MATCH_RECOGNIZE is safe only when the join
        cannot multiply rows from the preserved side.  The schema metadata
        currently exposes primary keys (but not arbitrary unique constraints),
        so require the join columns on the non-preserved side to cover its
        complete primary key.
        """
        if join_info["join_type"] not in {"LEFT", "RIGHT"}:
            return False

        other_idx = 1 - preserved_idx
        preserved = table_refs[preserved_idx]
        other = table_refs[other_idx]
        preserved_names = {preserved["ref"], preserved["table"]}
        other_names = {other["ref"], other["table"]}
        other_join_cols: Set[str] = set()

        for condition in join_info["conditions"]:
            if (
                condition["left_ref"] in other_names
                and condition["right_ref"] in preserved_names
            ):
                other_join_cols.add(condition["left_col"])
            elif (
                condition["right_ref"] in other_names
                and condition["left_ref"] in preserved_names
            ):
                other_join_cols.add(condition["right_col"])

        other_pk = set(other["primary_key"])
        return len(other_pk) > 0 and other_pk.issubset(other_join_cols)

    def _can_post_join_by_semijoin(
        self,
        source_idx: int,
        join_info: dict,
        table_refs: List[dict],
    ) -> bool:
        if join_info["join_type"] not in {"INNER", "COMMA"}:
            return False

        other_idx = 1 - source_idx
        source = table_refs[source_idx]
        other = table_refs[other_idx]
        source_names = {source["ref"], source["table"]}
        other_names = {other["ref"], other["table"]}
        other_cross_join_cols: Set[str] = set()

        for condition in join_info["conditions"]:
            if (
                condition["left_ref"] in other_names
                and condition["right_ref"] in source_names
            ):
                other_cross_join_cols.add(condition["left_col"])
            elif (
                condition["right_ref"] in other_names
                and condition["left_ref"] in source_names
            ):
                other_cross_join_cols.add(condition["right_col"])

        other_pk = set(other["primary_key"])
        return len(other_pk) > 0 and other_pk.issubset(other_cross_join_cols)

    def _build_base_match_ast(
        self, ast: RPRAbstractSyntaxTree, input_table_sql: str
    ) -> RPRAbstractSyntaxTree:
        inner_ast = copy.deepcopy(ast)
        inner_ast.raw_sql_override = None
        inner_ast.output_names = ExprParser().parse_select("SELECT *")
        inner_ast.input_table = input_table_sql
        inner_ast.where = None
        inner_ast.group_by = None
        inner_ast.having = None
        inner_ast.order_by = None
        inner_ast.limit = None
        inner_ast.alias = None
        return inner_ast

    def _build_preserved_outer_join_sql(
        self, ast, source_idx, join_info, source_tables, table_refs, projected_items
    ) -> Optional[str]:
        preserved = table_refs[source_idx]
        other = table_refs[1 - source_idx]
        matched_alias = choose_generated_relation_name(
            ast,
            f"MATCHED_{preserved['ref']}",
            [ref["ref"] for ref in table_refs],
        )
        inner_ast = self._build_base_match_ast(ast, preserved["table"])
        mr_sql = inner_ast.to_sql()
        join_type = "LEFT JOIN"
        if join_info["join_type"] == "RIGHT":
            join_type = "LEFT JOIN"
        condition = self._rewrite_join_condition(
            join_info["conditions"],
            table_refs,
            {preserved["ref"]: matched_alias, preserved["table"]: matched_alias},
        )
        select_items = self._build_post_join_select_items(
            projected_items, table_refs, {source_idx: matched_alias}
        )
        sql = (
            "SELECT\n    " + ",\n    ".join(select_items) + "\n"
            "FROM (\n" + mr_sql + f"\n) {matched_alias}\n"
            f"{join_type} {self._format_table_ref(other['table'], other['ref'])}\n"
            "ON " + condition
        )
        order_cols = self._normalize_column_sequence(
            [col for col, _ in (ast.match_order_by or [])]
        )
        if len(order_cols) > 0:
            sql += "\nORDER BY " + ", ".join(order_cols)
        return sql

    def _build_semijoin_sql(
        self, ast, source_idx, join_info, source_tables, table_refs, projected_items
    ) -> Optional[str]:
        fact = table_refs[source_idx]
        other = table_refs[1 - source_idx]
        fact_alias = fact["ref"]
        other_alias = other["ref"]
        exists_condition = self._rewrite_join_condition(
            join_info["conditions"],
            table_refs,
            {
                fact["ref"]: fact_alias,
                fact["table"]: fact_alias,
                other["ref"]: other_alias,
                other["table"]: other_alias,
            },
        )
        filtered_source = (
            f"(SELECT * FROM {self._format_table_ref(fact['table'], fact_alias)} "
            f"WHERE EXISTS (SELECT 1 FROM {self._format_table_ref(other['table'], other_alias)} WHERE {exists_condition}))"
        )
        inner_ast = self._build_base_match_ast(ast, filtered_source)
        mr_sql = inner_ast.to_sql()
        matched_alias = choose_generated_relation_name(
            ast,
            f"MATCHED_{fact['ref']}",
            [ref["ref"] for ref in table_refs],
        )
        post_condition = self._rewrite_join_condition(
            join_info["conditions"],
            table_refs,
            {fact["ref"]: matched_alias, fact["table"]: matched_alias},
        )
        select_items = self._build_post_join_select_items(
            projected_items, table_refs, {source_idx: matched_alias}
        )
        sql = (
            "SELECT\n    " + ",\n    ".join(select_items) + "\n"
            "FROM (\n" + mr_sql + f"\n) {matched_alias}\n"
            f"JOIN {self._format_table_ref(other['table'], other['ref'])}\n"
            "ON " + post_condition
        )
        order_cols = self._normalize_column_sequence(
            [col for col, _ in (ast.match_order_by or [])]
        )
        if len(order_cols) > 0:
            sql += "\nORDER BY " + ", ".join(order_cols)
        return sql

    def _build_join_label_sql(
        self, ast, join_info, source_tables, table_refs, table_schemas, projected_items
    ) -> Optional[str]:
        label_input_items = []
        extra_projection_cols = []
        pk_aliases_by_source = {}
        next_pk_idx = 1
        used_names = {
            col.upper()
            for schema in table_schemas.values()
            for col in schema.get("columns", [])
        }

        def allocate(base: str) -> str:
            candidate = base.upper()
            suffix = 1
            while candidate in used_names:
                candidate = f"{base.upper()}_{suffix}"
                suffix += 1
            used_names.add(candidate)
            return candidate

        for ref in table_refs:
            pk_aliases = []
            for pk_col in ref["primary_key"]:
                alias = allocate(f"RPR_JOIN_PK_{next_pk_idx}")
                next_pk_idx += 1
                label_input_items.append(f"{ref['ref']}.{pk_col} AS {alias}")
                extra_projection_cols.append(alias)
                pk_aliases.append((pk_col, alias))
            pk_aliases_by_source[ref["idx"]] = pk_aliases

        extra_seen = {col.upper() for col in extra_projection_cols}
        for col in self._normalize_column_sequence(
            [col for col, _ in (ast.match_order_by or [])]
        ):
            owner = self._resolve_source_column_item(
                col, source_tables, table_schemas, table_refs
            )
            if owner is None:
                owner = self._resolve_join_equivalent_column_item(
                    col, join_info, table_refs
                )
            if owner is not None:
                out_col = owner["output_col"].upper()
                if out_col not in extra_seen:
                    label_input_items.append(
                        f"{table_refs[owner['source_idx']]['ref']}.{owner['source_col']} AS {out_col}"
                    )
                    extra_projection_cols.append(out_col)
                    extra_seen.add(out_col)

        projection_input_sql = join_info.get(
            "source_sql"
        ) or self._join_source_without_where(source_tables, join_info)
        label_ast = self._build_base_match_ast(ast, projection_input_sql)
        label_plan = self._build_label_plan(
            label_ast, extra_projection_cols, projection_input_sql, label_input_items
        )
        if label_plan is None:
            return None
        label_ast.input_table = label_plan["input_table"]
        label_ast.define.definitions = label_plan["definitions"]
        if label_ast.measures is None:
            label_ast.measures = MeasuresSubClause({})
        measures = dict(label_ast.measures.measures)
        for ref in table_refs:
            for _, pk_alias in pk_aliases_by_source[ref["idx"]]:
                measures[pk_alias] = ColumnRef(None, pk_alias)
        existing_match_no_alias = self._find_match_number_alias(ast)
        match_no_alias = (
            existing_match_no_alias.upper()
            if existing_match_no_alias is not None
            else choose_generated_column_name(ast, "RPR_MATCH_NO", measures.keys())
        )
        if existing_match_no_alias is None:
            measures[match_no_alias] = FunctionCall("MATCH_NUMBER", [])
        label_ast.measures.measures = measures
        mr_sql = label_ast.to_sql()
        matched_alias = choose_generated_relation_name(
            ast, "MR", [ref["ref"] for ref in table_refs]
        )
        select_items = self._build_post_join_select_items(
            projected_items, table_refs, {}
        )
        sql = (
            "SELECT\n    "
            + ",\n    ".join(select_items)
            + "\nFROM (\n"
            + mr_sql
            + f"\n) {matched_alias}"
        )
        for ref in table_refs:
            conds = []
            for pk_col, pk_alias in pk_aliases_by_source[ref["idx"]]:
                conds.append(f"{ref['ref']}.{pk_col} = {matched_alias}.{pk_alias}")
            sql += (
                f"\nLEFT JOIN {self._format_table_ref(ref['table'], ref['ref'])}\nON "
                + " AND ".join(conds)
            )
        order_parts = self._build_default_outer_order_parts(
            ast, matched_alias, match_no_alias
        )
        sql += "\nORDER BY " + ", ".join(order_parts)
        return sql

    def _join_source_without_where(
        self, source_tables: List[dict], join_info: dict
    ) -> str:
        left = source_tables[0]
        right = source_tables[1]
        left_ref = (
            f"{left['table']} {left['alias']}" if left.get("alias") else left["table"]
        )
        right_ref = (
            f"{right['table']} {right['alias']}"
            if right.get("alias")
            else right["table"]
        )
        return f"{left_ref} JOIN {right_ref} ON {join_info['condition_sql']}"

    def _format_table_ref(self, table_name: str, ref_name: Optional[str]) -> str:
        if ref_name is None or ref_name.upper() == table_name.upper():
            return table_name
        return f"{table_name} {ref_name}"

    def _rewrite_join_condition(
        self,
        conditions: List[dict],
        table_refs: List[dict],
        replacements: Dict[str, str],
    ) -> str:
        parts = []
        for cond in conditions:
            left_ref = replacements.get(cond["left_ref"], cond["left_ref"])
            right_ref = replacements.get(cond["right_ref"], cond["right_ref"])
            parts.append(
                f"{left_ref}.{cond['left_col']} = {right_ref}.{cond['right_col']}"
            )
        return " AND ".join(parts)

    def _build_post_join_select_items(
        self,
        projected_items: List[dict],
        table_refs: List[dict],
        replacements_by_idx: Dict[int, str],
    ) -> List[str]:
        result = []
        for item in projected_items:
            ref = replacements_by_idx.get(
                item["source_idx"], table_refs[item["source_idx"]]["ref"]
            )
            expr = f"{ref}.{item['source_col']}"
            if item["output_col"] != item["source_col"]:
                result.append(f"{expr} AS {item['output_col']}")
            else:
                result.append(f"{expr} AS {item['output_col']}")
        return result

    def _build_label_plan(
        self,
        ast: RPRAbstractSyntaxTree,
        pk_cols: List[str],
        base_table: str,
        pk_projection_items: Optional[List[str]] = None,
    ):
        helper = ColumnPruningRule()
        original_rows_per_match = ast.rows_per_match
        original_input_table = ast.input_table
        original_table_schemas = ast.table_schemas

        # ColumnPruningRule asks the AST for column nullability while deciding
        # whether DEFINE predicates can share a label.  Once a derived input is
        # flattened to e.g. ``D WHERE ...`` or ``E JOIN F ...``, the generic AST
        # resolver cannot always infer which entries in a repository-wide
        # schema mapping belong to that input.  Unqualified names such as INFO
        # or LOCATION then look ambiguous because unrelated tables are also
        # considered.  Scope metadata to the relations in this input while the
        # helper builds its plan, just as SQL name resolution does.
        source_tables = self._extract_source_tables(base_table)
        if source_tables is not None and ast.table_schemas is not None:
            source_names = {source["table"].upper() for source in source_tables}
            scoped_schemas = {
                table_name: metadata
                for table_name, metadata in ast.table_schemas.items()
                if str(table_name).upper() in source_names
            }
            if len(scoped_schemas) == len(source_names):
                ast.table_schemas = scoped_schemas

        ast.rows_per_match = "ONE ROW PER MATCH"
        ast.input_table = base_table
        try:
            return helper._build_plan(
                ast,
                extra_projection_cols=pk_cols,
                extra_projection_items=pk_projection_items,
            )
        finally:
            ast.rows_per_match = original_rows_per_match
            ast.input_table = original_input_table
            ast.table_schemas = original_table_schemas

    def _normalize_column_sequence(self, columns: List[str]) -> List[str]:
        result = []
        seen = set()
        for col in columns:
            norm = self._normalize_column_name(col)
            if norm is None or norm in seen:
                continue
            seen.add(norm)
            result.append(norm)
        return result

    def _ordered_union(self, *column_lists: List[str]) -> List[str]:
        result = []
        seen = set()
        for column_list in column_lists:
            for col in column_list:
                up = col.upper()
                if up in seen:
                    continue
                seen.add(up)
                result.append(col.upper())
        return result

    def _normalize_column_name(self, col_name: Optional[str]) -> Optional[str]:
        if col_name is None:
            return None
        col_name = col_name.strip()
        if len(col_name) == 0:
            return None
        if "." in col_name:
            col_name = col_name.split(".")[-1]
        return col_name.upper()

    def _columns_conflict(
        self, ast: RPRAbstractSyntaxTree, reserved_names: Set[str]
    ) -> bool:
        used = set()
        if ast.partition_by is not None:
            used.update(self._normalize_column_sequence(ast.partition_by))
        if ast.match_order_by is not None:
            used.update(
                self._normalize_column_sequence([col for col, _ in ast.match_order_by])
            )
        if ast.measures is not None:
            used.update(alias.upper() for alias in ast.measures.measures.keys())
            used.update(col.upper() for col in ast.measures.get_columns())
        used.update(col.upper() for col in ast.define.get_columns())
        return any(name.upper() in used for name in reserved_names)

    def _format_order_by(self, order_by_items: List[Tuple[str, SortType]]) -> str:
        parts = []
        for col_name, sort_type in order_by_items:
            norm = self._normalize_column_name(col_name)
            if norm is None:
                continue
            if sort_type == SortType.ASC:
                parts.append(norm)
            else:
                parts.append(norm + " DESC")
        return ", ".join(parts)

    def _build_default_outer_order_parts(
        self,
        ast: RPRAbstractSyntaxTree,
        relation_alias: str,
        match_no_alias: str,
    ) -> List[str]:
        """Restore MATCH_RECOGNIZE row order after joining full rows back.

        MATCH_NUMBER() restarts in every partition, so partition columns must
        precede it.  The MATCH_RECOGNIZE ordering columns then retain the row
        order inside each match, including their original sort directions.
        """
        parts = [
            f"{relation_alias}.{col}"
            for col in self._normalize_column_sequence(ast.partition_by or [])
        ]
        parts.append(f"{relation_alias}.{match_no_alias}")

        seen = set()
        for col_name, sort_type in ast.match_order_by or []:
            col = self._normalize_column_name(col_name)
            if col is None or col in seen:
                continue
            seen.add(col)
            part = f"{relation_alias}.{col}"
            if sort_type != SortType.ASC:
                part += " DESC"
            parts.append(part)
        return parts


column_pruning_via_join_test_cases = [
    # COURSE_REGISTRATION: ['STUDENT_ID', 'COURSE_ID', 'PT', 'INFO', 'VAL', 'TS']
    # primary key: ['STUDENT_ID', 'COURSE_ID']
    (
        1,
        """
SELECT * FROM COURSE_REGISTRATION MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS DESC
    MEASURES A.STUDENT_ID AS AID, B.STUDENT_ID AS BID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS COURSE_ID = 'A', B AS COURSE_ID = 'B' AND VAL > 10
)""",
        """
SELECT MR.PT AS PT, MR.TS AS TS, MR.AID AS AID, MR.BID AS BID, T_FULL.STUDENT_ID AS STUDENT_ID, T_FULL.COURSE_ID AS COURSE_ID, T_FULL.INFO AS INFO, T_FULL.VAL AS VAL
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, 
        CASE WHEN COURSE_ID = 'A' THEN 'a' 
        WHEN (COURSE_ID = 'B' AND VAL > 10) THEN 'b' 
        ELSE 'c' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS DESC
    MEASURES A.STUDENT_ID AS AID, B.STUDENT_ID AS BID, STUDENT_ID AS RPR_JOIN_PK_1, COURSE_ID AS RPR_JOIN_PK_2, MATCH_NUMBER() AS RPR_MATCH_NO
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
) MR
LEFT JOIN COURSE_REGISTRATION T_FULL
ON T_FULL.STUDENT_ID = MR.RPR_JOIN_PK_1 AND T_FULL.COURSE_ID = MR.RPR_JOIN_PK_2
ORDER BY MR.PT, MR.RPR_MATCH_NO, MR.TS DESC
""",
    ),
    (
        2,
        """
SELECT *
FROM COURSE_REGISTRATION
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES CLASSIFIER() AS CLS
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS COURSE_ID = 'A', B AS COURSE_ID = 'B' AND VAL > 10
)""",
        """
SELECT MR.PT AS PT, MR.TS AS TS, MR.CLS AS CLS, T_FULL.STUDENT_ID AS STUDENT_ID, T_FULL.COURSE_ID AS COURSE_ID, T_FULL.INFO AS INFO, T_FULL.VAL AS VAL
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, 
        CASE WHEN COURSE_ID = 'A' THEN 'a' 
        WHEN (COURSE_ID = 'B' AND VAL > 10) THEN 'b' 
        ELSE 'c' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES CLASSIFIER() AS CLS, STUDENT_ID AS RPR_JOIN_PK_1, COURSE_ID AS RPR_JOIN_PK_2, MATCH_NUMBER() AS RPR_MATCH_NO
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
 ) MR
LEFT JOIN COURSE_REGISTRATION T_FULL
ON T_FULL.STUDENT_ID = MR.RPR_JOIN_PK_1 AND T_FULL.COURSE_ID = MR.RPR_JOIN_PK_2
ORDER BY MR.PT, MR.RPR_MATCH_NO, MR.TS
""",
    ),
    # we need add left join to obtain 'INFO' column,
    (
        3,
        """
SELECT PT, AID, INFO
FROM COURSE_REGISTRATION AS CR
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.STUDENT_ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS COURSE_ID = 'A', B AS COURSE_ID = 'B' AND VAL > 10
)
ORDER BY AID DESC
""",
        """
SELECT MR.PT, MR.AID, T_FULL.INFO
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, CASE WHEN COURSE_ID = 'A' THEN 'a' WHEN (COURSE_ID = 'B' AND VAL > 10) THEN 'b' ELSE 'c' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.STUDENT_ID AS AID, STUDENT_ID AS RPR_JOIN_PK_1, COURSE_ID AS RPR_JOIN_PK_2, MATCH_NUMBER() AS RPR_MATCH_NO
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
) MR
LEFT JOIN COURSE_REGISTRATION T_FULL
ON T_FULL.STUDENT_ID = MR.RPR_JOIN_PK_1 AND T_FULL.COURSE_ID = MR.RPR_JOIN_PK_2
ORDER BY MR.AID DESC
""",
    ),
    # we do not need to add left join, since the final output columns are all from the MATCH_RECOGNIZE output
    (
        4,
        """
SELECT PT, AID
FROM COURSE_REGISTRATION AS CR
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.STUDENT_ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS COURSE_ID = 'A', B AS COURSE_ID = 'B' AND VAL > 10
)
ORDER BY AID DESC
""",
        """
SELECT PT, AID
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, 
        CASE WHEN COURSE_ID = 'A' THEN 'a' 
        WHEN (COURSE_ID = 'B' AND VAL > 10) THEN 'b' 
        ELSE 'c' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.STUDENT_ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
)
ORDER BY AID DESC
""",
    ),
    # we do not need to add left join, since the final output columns are all from the MATCH_RECOGNIZE output
    (
        5,
        """
SELECT PT, AID
FROM COURSE_REGISTRATION AS CR
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.STUDENT_ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N B)
    DEFINE A AS COURSE_ID = 'A', B AS TS - A.TS < 10
)
ORDER BY AID DESC
""",
        """
SELECT PT, AID
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, CASE WHEN COURSE_ID = 'A' THEN 'a' ELSE 'b' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.STUDENT_ID AS AID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A N B)
    DEFINE A AS TAG = 'a', B AS (TS - A.TS) < 10
)
ORDER BY AID DESC
""",
    ),
    (
        6,
        """
SELECT *
FROM COURSE_REGISTRATION
MATCH_RECOGNIZE(
    PARTITION BY PT ORDER BY TS
    MEASURES A.STUDENT_ID AS AID, B.STUDENT_ID AS BID
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS COURSE_ID = 'A', B AS COURSE_ID = 'B', C AS COURSE_ID = 'A'
)""",
        """
SELECT MR.PT AS PT, MR.TS AS TS, MR.AID AS AID, MR.BID AS BID, T_FULL.STUDENT_ID AS STUDENT_ID, T_FULL.COURSE_ID AS COURSE_ID, T_FULL.INFO AS INFO, T_FULL.VAL AS VAL
FROM (SELECT STUDENT_ID, COURSE_ID, PT, TS, CASE WHEN COURSE_ID = 'A' THEN 'a' WHEN COURSE_ID = 'B' THEN 'b' ELSE 'c' END AS TAG FROM COURSE_REGISTRATION)
MATCH_RECOGNIZE(
    PARTITION BY PT
    ORDER BY TS
    MEASURES A.STUDENT_ID AS AID, B.STUDENT_ID AS BID, STUDENT_ID AS RPR_JOIN_PK_1, COURSE_ID AS RPR_JOIN_PK_2, MATCH_NUMBER() AS RPR_MATCH_NO
    ALL ROWS PER MATCH
    AFTER MATCH SKIP TO NEXT ROW
    PATTERN(A B C)
    DEFINE A AS TAG = 'a', B AS TAG = 'b', C AS TAG = 'a'
) MR
LEFT JOIN COURSE_REGISTRATION T_FULL
ON T_FULL.STUDENT_ID = MR.RPR_JOIN_PK_1 AND T_FULL.COURSE_ID = MR.RPR_JOIN_PK_2
ORDER BY MR.PT, MR.RPR_MATCH_NO, MR.TS
""",
    ),
]


real_column_pruning_via_join_test_cases = [
    # 39s vs 26s
    (
        1,
        """
SELECT *
FROM CRIMES
MATCH_RECOGNIZE(
    ORDER BY TS, ID
    ALL ROWS PER MATCH
    PATTERN(A N{0,2}? B)
    DEFINE 
        A AS PRIMARY_TYPE = 'NARCOTICS',
        B AS PRIMARY_TYPE = 'RITUALISM'
)""",
        """
SELECT MR.TS AS TS, MR.ID AS ID, T_FULL.PRIMARY_TYPE AS PRIMARY_TYPE, T_FULL.BEAT AS BEAT, T_FULL.DIS AS DIS, T_FULL.LAT AS LAT, T_FULL.LON AS LON
FROM (SELECT ID, TS, CASE WHEN PRIMARY_TYPE = 'NARCOTICS' THEN 'a' WHEN PRIMARY_TYPE = 'RITUALISM' THEN 'b' ELSE 'c' END AS TAG FROM CRIMES)
MATCH_RECOGNIZE(
    ORDER BY TS, ID
    MEASURES ID AS RPR_JOIN_PK_1, MATCH_NUMBER() AS RPR_MATCH_NO
    ALL ROWS PER MATCH
    PATTERN(A N{0,2}? B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
) MR
LEFT JOIN CRIMES T_FULL
ON T_FULL.ID = MR.RPR_JOIN_PK_1
ORDER BY MR.RPR_MATCH_NO, MR.TS, MR.ID
""",
    ),
    (
        2,
        """
SELECT *
FROM CRIMES
MATCH_RECOGNIZE(
    ORDER BY TS, ID
    MEASURES MATCH_NUMBER() AS MATCH_NO
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE 
        A AS PRIMARY_TYPE = 'NARCOTICS',
        B AS PRIMARY_TYPE = 'RITUALISM'
)""",
        """
SELECT MR.TS AS TS, MR.ID AS ID, MR.MATCH_NO AS MATCH_NO, T_FULL.PRIMARY_TYPE AS PRIMARY_TYPE, T_FULL.BEAT AS BEAT, T_FULL.DIS AS DIS, T_FULL.LAT AS LAT, T_FULL.LON AS LON
FROM (SELECT ID, TS, CASE WHEN PRIMARY_TYPE = 'NARCOTICS' THEN 'a' WHEN PRIMARY_TYPE = 'RITUALISM' THEN 'b' ELSE 'c' END AS TAG FROM CRIMES)
MATCH_RECOGNIZE(
    ORDER BY TS, ID
    MEASURES MATCH_NUMBER() AS MATCH_NO, ID AS RPR_JOIN_PK_1
    ALL ROWS PER MATCH
    PATTERN(A B)
    DEFINE A AS TAG = 'a', B AS TAG = 'b'
) MR
LEFT JOIN CRIMES T_FULL
ON T_FULL.ID = MR.RPR_JOIN_PK_1
ORDER BY MR.MATCH_NO, MR.TS, MR.ID
""",
    ),
]


schema_meta = {
    "COURSE_REGISTRATION": {
        "primary_key": ["STUDENT_ID", "COURSE_ID"],
        "columns": ["STUDENT_ID", "COURSE_ID", "PT", "INFO", "VAL", "TS"],
    },
    "A": {
        "primary_key": ["AID"],
        "columns": ["AID", "INFO"],
    },
    "B": {
        "primary_key": ["BID"],
        "columns": ["BID", "AID", "LOCATION", "TS"],
    },
    "C": {
        "primary_key": ["ID"],
        "columns": ["ID", "INFO"],
    },
    "D": {
        "primary_key": ["ID"],
        "columns": ["ID", "VAL", "LOCATION", "TS"],
    },
    "E": {
        "primary_key": ["ID"],
        "columns": ["ID", "INFO"],
    },
    "F": {
        "primary_key": ["ID", "TS"],
        "columns": ["ID", "LOCATION", "TS"],
    },
}


real_schema_meta = {
    "CRIMES": {
        "primary_key": ["ID"],
        "columns": ["TS", "ID", "PRIMARY_TYPE", "BEAT", "DIS", "LAT", "LON"],
    }
}


def run_tests(verbose=True):
    from .test_utils import run_rule_tests

    first = run_rule_tests(
        ColumnPruningViaJoinRule,
        column_pruning_via_join_test_cases,
        "column pruning via join",
        verbose,
        schema_meta,
    )
    second = run_rule_tests(
        ColumnPruningViaJoinRule,
        real_column_pruning_via_join_test_cases,
        "more test cases for column pruning via join",
        verbose,
        real_schema_meta,
    )
    return {
        "label": "column pruning via join",
        "total": first["total"] + second["total"],
        "passed": first["passed"] + second["passed"],
        "failed": first["failed"] + second["failed"],
        "failures": first["failures"] + second["failures"],
    }


if __name__ == "__main__":
    result = run_tests(verbose=True)
    print(f"{result['passed']} passed, {result['failed']} failed")
