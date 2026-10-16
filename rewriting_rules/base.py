from abc import ABC, abstractmethod
from typing import List, Tuple

from queryanalysis import RPRAbstractSyntaxTree


class RewriteRule(ABC):
    """Base class for all rewrite rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def is_applicable(self, ast: RPRAbstractSyntaxTree) -> bool:
        pass

    @abstractmethod
    def apply(self, ast: RPRAbstractSyntaxTree) -> bool:
        """Apply rewrite in-place. Return True if ast changed."""
        pass


class RewriteEngine:
    def __init__(self, rules: List[RewriteRule]):
        self.rules = rules

    def rewrite(
        self, ast: RPRAbstractSyntaxTree
    ) -> Tuple[RPRAbstractSyntaxTree, List[str]]:
        validation_errors = ast.validate()
        if validation_errors:
            raise ValueError("; ".join(validation_errors))

        applied_rules: List[str] = []
        for rule in self.rules:
            if not rule.is_applicable(ast):
                continue
            input_table_before = ast.input_table
            if rule.apply(ast):
                applied_rules.append(rule.name)
                if ast.input_table != input_table_before:
                    # some subquery cannot be optimized such as
                    # (SELECT * FROM T WHERE ROWNUM) WHERE x > 10
                    if getattr(ast, "_skip_from_clause_optimization_once", False):
                        ast._skip_from_clause_optimization_once = False
                    else:
                        ast.optimize_from_clause()
                # if triggle MustAppearVarFalsePushdownRule, we can directly
                # return the rewritten query since this query without any results
                if rule.name == "must_appear_var_false_pushdown":
                    break
        return ast, applied_rules
