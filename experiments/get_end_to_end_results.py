"""Join rewriting overhead and effective execution time by query ID.

Run: python3 experiments/get_end_to_end_results.py
Uses get_main_results.py's three-run mean and failure substitution policy.
Writes per-query audit data, summaries, and a complete LaTeX table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from get_main_results import load_queries, percentile, prepare_results

ROOT = Path(__file__).resolve().parents[1]
# Experiment suffix, metrics suffix, display name.
METHODS = [
    ("original", None, "Original"),
    ("deepseek_rewriting", "deepseek_v1_thinking", "LLM-based (w. DS-4.1 Thinking)"),
    (
        "deepseek_fast_rewriting",
        "deepseek_v1_no_thinking",
        "LLM-based (w. DS-4.1 Non-thinking)",
    ),
    (
        "gpt5.6_rewriting",
        "gpt-5.6-sol_v1_thinking",
        "LLM-based (w. GPT-5.6-Sol Reasoning)",
    ),
    (
        "gpt5.6_fast_rewriting",
        "gpt-5.6-sol_v1_no_thinking",
        "LLM-based (w. GPT-5.6-Sol Non-reasoning)",
    ),
    ("gpt6_rewriting", "gpt-6-astra_v1", "LLM-based (w. GPT-6-Astra Reasoning)"),
    ("sota_rewriting", "sota_method", "SOTA-R2 (Rule-based)"),
    ("our_rewriting", "our_method", "R3PR (Ours)"),
]
GROUPS = [
    (system, dataset)
    for system in ("oracle", "trino")
    for dataset in ("crimes", "nasdaq")
]


def read_overheads(
    path: Path, ids: list[int], execution_queries: dict
) -> dict[int, float]:
    document = json.loads(path.read_text())
    queries = document["queries"]
    by_id = {}
    for query in queries:
        query_id = query["index"]
        if query_id in by_id:
            raise ValueError(f"Duplicate query ID {query_id} in {path}")
        by_id[query_id] = query
    if set(by_id) != set(ids) or document["totals"]["query_count"] != len(ids):
        raise ValueError(f"Query IDs/count do not match execution results: {path}")
    values = {}
    for query_id in ids:
        query = by_id[query_id]
        field = (
            "total_overhead_seconds"
            if "total_overhead_seconds" in query
            else "latency_seconds"
        )
        value = query[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid {field} for query {query_id} in {path}")
        # Labels/URLs, where both exist, provide an additional identity check.
        label = query.get("label")
        url = execution_queries[query_id].get("url")
        if label and url and label != url:
            raise ValueError(f"Query label mismatch for {query_id} in {path}")
        values[query_id] = float(value)
    return values


def collect() -> dict:
    output = {}
    for system, dataset in GROUPS:
        directories = [
            ROOT / "experiments" / f"{system}_{dataset}_{m[0]}" for m in METHODS
        ]
        baseline = load_queries(directories[0])
        ids = sorted(baseline)
        effective = prepare_results(directories)
        summaries = []
        for (suffix, metric_suffix, name), directory, result in zip(
            METHODS, directories, effective
        ):
            execution_queries = load_queries(directory)
            if set(execution_queries) != set(ids):
                raise ValueError(f"Query IDs differ from baseline: {directory}")
            overheads = dict.fromkeys(ids, 0.0)
            metrics_path = None
            if metric_suffix:
                prefix = (
                    ("trino_" if system == "trino" else "")
                    if metric_suffix in ("sota_method", "our_method")
                    else system + "_"
                )
                metrics_path = (
                    ROOT
                    / "data"
                    / f"{dataset}_sql_{prefix}{metric_suffix}.metrics.json"
                )
                overheads = read_overheads(metrics_path, ids, execution_queries)
            rows = [
                {
                    "id": query_id,
                    "rewriting_seconds": overheads[query_id],
                    "execution_seconds": execution,
                    "end_to_end_seconds": overheads[query_id] + execution,
                    "execution_substituted": query_id in result.substituted_ids,
                }
                for query_id, execution in zip(ids, result.values)
            ]
            e2e = [row["end_to_end_seconds"] for row in rows]
            rt = statistics.fmean(overheads.values())
            mean = statistics.fmean(e2e)
            assert math.isclose(
                mean, rt + statistics.fmean(result.values), abs_tol=1e-9
            )
            summaries.append(
                {
                    "method": name,
                    "query_count": len(ids),
                    "mean_rewriting_seconds": rt,
                    "mean_seconds": mean,
                    "median_seconds": statistics.median(e2e),
                    "p75_seconds": percentile(e2e, 0.75),
                    "execution_source": str(directory / "query_times.json"),
                    "rewriting_source": str(metrics_path) if metrics_path else None,
                    "queries": rows,
                }
            )
        output[f"{system}_{dataset}"] = summaries
    return output


def latex(output: dict) -> str:
    groups = [output[f"{system}_{dataset}"] for system, dataset in GROUPS]
    stats = ["mean_rewriting_seconds", "mean_seconds", "median_seconds", "p75_seconds"]
    rankings = {
        (g, stat): sorted(set(row[stat] for row in group))[:2]
        for g, group in enumerate(groups)
        for stat in stats[1:]
    }
    rows = []
    for i, (_, _, name) in enumerate(METHODS):
        cells = []
        for g, group in enumerate(groups):
            for stat in stats:
                value = group[i][stat]
                cell = f"{value:.2f}" if stat == stats[0] else f"{value:.1f}"
                if stat != stats[0]:
                    rank = rankings[g, stat]
                    if value == rank[0]:
                        cell = r"\textbf{" + cell + "}"
                    elif len(rank) > 1 and value == rank[1]:
                        cell = r"\underline{" + cell + "}"
                cells.append(cell)
        if i == len(METHODS) - 1:
            name = r"\textbf{" + name + "}"
        rows.append("        " + name + "\n        & " + " & ".join(cells) + r" \\")
    return (
        r"""% Preamble: \usepackage{booktabs,graphicx}
\subsection{Execution Efficiency Evaluation}
We evaluate end-to-end latency by adding each query's rewriting time to
its effective execution time. Execution time is averaged over three runs.
Following the execution-only evaluation protocol, failed rewritten queries
use the corresponding original-query execution time, and failed original
queries are assigned 300 seconds. Rewriting overhead is still included
for failed rewritten queries. Rule-based rewriting time includes sampling.
End-to-end times are not capped at 300 seconds. We compute the mean, median,
and P75 across per-query end-to-end times; P75 uses linear interpolation.

\begin{table*}[t]
    \centering
    \caption{End-to-end latency on CDB and Trino+PostgreSQL (seconds).
    ARC denotes mean rewriting time; Mean, Median, and P75 summarize
    rewriting plus effective query execution over 100 queries per dataset.
    DS-4.1 abbreviates DeepSeek-4.1-Flash. For each end-to-end statistic,
    the lowest value is bold and the second-lowest is underlined,
    with rankings determined before rounding.}
    \label{tab:main-results}
    \setlength{\tabcolsep}{3.5pt}
    \renewcommand{\arraystretch}{1.10}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{@{}l*{16}{r}@{}}
        \toprule
        & \multicolumn{8}{c}{\textbf{CDB}}
        & \multicolumn{8}{c}{\textbf{Trino+PostgreSQL}} \\
        \cmidrule(lr){2-9}\cmidrule(lr){10-17}
        & \multicolumn{4}{c}{\textsc{crimes}}
        & \multicolumn{4}{c}{\textsc{nasdaq}}
        & \multicolumn{4}{c}{\textsc{crimes}}
        & \multicolumn{4}{c}{\textsc{nasdaq}} \\
        \cmidrule(lr){2-5}\cmidrule(lr){6-9}
        \cmidrule(lr){10-13}\cmidrule(lr){14-17}
        \textbf{Method}
        & \textbf{ARC} & \textbf{Mean} & \textbf{Median} & \textbf{P75}
        & \textbf{ARC} & \textbf{Mean} & \textbf{Median} & \textbf{P75}
        & \textbf{ARC} & \textbf{Mean} & \textbf{Median} & \textbf{P75}
        & \textbf{ARC} & \textbf{Mean} & \textbf{Median} & \textbf{P75} \\
        \midrule
"""
        + "\n\n".join(rows)
        + r"""
        \bottomrule
    \end{tabular}%
    }
\end{table*}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments" / "end_to_end_results"
    )
    args = parser.parse_args()
    results = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "main_results.tex").write_text(latex(results))
    for group, rows in results.items():
        print(group)
        for row in rows:
            print(
                f"  {row['method']}: ARC={row['mean_rewriting_seconds']:.3f}, "
                f"Mean={row['mean_seconds']:.3f}, Median={row['median_seconds']:.3f}, P75={row['p75_seconds']:.3f}"
            )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
