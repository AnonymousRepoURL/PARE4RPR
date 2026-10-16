from importlib import import_module
import time

import _bootstrap  # noqa: F401  Ensures project modules are importable.

TEST_MODULES = [
    "single_variable_pattern_pushdown",
    "remove_aggregation_func",
    "must_appear_var_false_pushdown",
    "define_partition_pushdown",
    "ordered_range_pushdown",
    "anchor_top_k_filter",
    "anchor_partition_size_filter",
    "quantifier_scope_shortening",
    "where_partition_pushdown",
    "row_filter_4_special_pattern",
    "alternation_variable_merge",
    "match_number_guard_pushdown",
    "incremental_condition_pushdown",
    "implied_incremental_condition_removal",
    "column_pruning",
    "column_pruning_via_join",
    "forward_adjusted_condition",
    "count_range_pushdown",
    "nested_quantifier_range_simplification",
    "must_appear_partition_filter_pushdown",
    "window_cond_prefilter",
    "sequence_filtering",
    "row_filtering",
    "row_filtering_via_window_func",
    "measures_variable_renaming",
    "define_independent_constraint_propagation",
    "redundant_subset_variable_removal",
    "regex_optimization_plus",
]


def main():
    start_time = time.time()
    results = [
        import_module(f"rewriting_rules.{module_name}").run_tests(verbose=True)
        for module_name in TEST_MODULES
    ]

    total = sum(result["total"] for result in results)
    passed = sum(result["passed"] for result in results)
    failed = sum(result["failed"] for result in results)

    print(f"{'=' * 60}")
    print(f"Final test results (all): {passed} passed, {failed} failed, {total} total")
    print(f"{'=' * 60}")
    print(f"==> Total time taken: {time.time() - start_time:.2f} seconds")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
