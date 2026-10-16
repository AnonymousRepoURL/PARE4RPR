from .base import RewriteRule, RewriteEngine
from .ordered_range_pushdown import OrderedRangePushdownRule
from .single_variable_pattern_pushdown import SingleVariablePatternPushdownRule
from .remove_aggregation_func import RemoveAggregationFuncRule
from .must_appear_var_false_pushdown import MustAppearVarFalsePushdownRule
from .define_partition_pushdown import DefinePartitionPushdownRule
from .quantifier_scope_shortening import (
    QuantifierScopeShorteningRule,
)
from .anchor_top_k_filter import AnchorTopKFilterRule
from .anchor_partition_size_filter import AnchorPartitionSizeFilterRule
from .where_partition_pushdown import WherePartitionPushdownRule
from .row_filter_4_special_pattern import RowFilter4SpecialPatternRule
from .alternation_variable_merge import AlternationVariableMergeRule
from .forward_adjusted_condition import ForwardAdjustedConditionRule
from .column_pruning import ColumnPruningRule
from .column_pruning_via_join import ColumnPruningViaJoinRule
from .incremental_condition_pushdown import IncrementalConditionPushdownRule
from .implied_incremental_condition_removal import (
    ImpliedIncrementalConditionRemovalRule,
)
from .count_range_pushdown import CountRangePushdownRule
from .nested_quantifier_range_simplification import (
    NestedQuantifierRangeSimplificationRule,
)
from .must_appear_partition_filter_pushdown import MustAppearPartitionFilterPushdownRule
from .window_cond_prefilter import (
    WindowConditionPreFilterRule,
    WindowPrefilterCandidate,
    WindowPrefilterStatistics,
    WindowPrefilterStatisticsRequest,
)
from .window_prefilter_optimizer import (
    WindowPrefilterBenchmarkPlan,
    WindowPrefilterBenchmarkQuery,
    WindowPrefilterOptimizationResult,
    WindowPrefilterOptimizer,
)
from .sequence_filtering import SequenceFilteringRule
from .row_filtering import RowFilteringRule
from .row_filtering_via_window_func import RowFilteringViaWindowFuncRule
from .measures_variable_renaming import MeasuresVariableRenamingRule
from .define_independent_constraint_propagation import (
    DefineIndependentConstraintPropagationRule,
)
from .redundant_subset_variable_removal import RedundantSubsetVariableRemovalRule
from .regex_optimization_plus import RegexOptimizationPlusRule
from .regex_optimization_ultra import RegexOptimizationUltraRule
from .match_number_guard_pushdown import MatchNumberGuardPushdownRule

# Public names exported when using `from rewriting_rules import *`.
__all__ = [
    "RewriteRule",
    "RewriteEngine",
    "OrderedRangePushdownRule",
    "SingleVariablePatternPushdownRule",
    "RemoveAggregationFuncRule",
    "MustAppearVarFalsePushdownRule",
    "DefinePartitionPushdownRule",
    "QuantifierScopeShorteningRule",
    "AnchorTopKFilterRule",
    "AnchorPartitionSizeFilterRule",
    "WherePartitionPushdownRule",
    "RowFilter4SpecialPatternRule",
    "AlternationVariableMergeRule",
    "ForwardAdjustedConditionRule",
    "ColumnPruningRule",
    "ColumnPruningViaJoinRule",
    "IncrementalConditionPushdownRule",
    "ImpliedIncrementalConditionRemovalRule",
    "CountRangePushdownRule",
    "NestedQuantifierRangeSimplificationRule",
    "MustAppearPartitionFilterPushdownRule",
    "WindowConditionPreFilterRule",
    "WindowPrefilterCandidate",
    "WindowPrefilterStatistics",
    "WindowPrefilterStatisticsRequest",
    "WindowPrefilterBenchmarkPlan",
    "WindowPrefilterBenchmarkQuery",
    "WindowPrefilterOptimizationResult",
    "WindowPrefilterOptimizer",
    "SequenceFilteringRule",
    "RowFilteringRule",
    "RowFilteringViaWindowFuncRule",
    "MeasuresVariableRenamingRule",
    "DefineIndependentConstraintPropagationRule",
    "RedundantSubsetVariableRemovalRule",
    "RegexOptimizationPlusRule",
    "RegexOptimizationUltraRule",
    "MatchNumberGuardPushdownRule",
]
