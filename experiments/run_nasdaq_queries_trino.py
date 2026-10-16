#!/usr/bin/env python3
r"""Run Nasdaq SQL queries through Trino and stream results to files.

Run commands:
python experiments\run_nasdaq_queries_trino.py
python experiments\run_nasdaq_queries_trino.py --resume

-----------------run generated queries-----------------
python experiments\run_nasdaq_queries_trino.py `
  --query-file data\nasdaq_sql_oracle_gpt-5.6-sol_v1_thinking.txt `
  --output-dir experiments\trino_nasdaq_gpt5.6_rewriting
-------------------------------------------------------
JSON input uses the ``sqls`` array. Plain-text input contains semicolon-separated
SQL statements. This entry point uses the same execution, timeout, capped result
capture, third-run timeout inference, metrics, and resume behavior as
``run_crimes_queries_trino.py``. Its defaults keep the Nasdaq results separate
under ``trino_nasdaq_original``.
"""

from pathlib import Path

import run_crimes_queries_trino as trino_runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
trino_runner.__doc__ = __doc__
trino_runner.DEFAULT_QUERY_FILE = PROJECT_ROOT / "data" / "nasdaq_sql.json"
trino_runner.DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "trino_nasdaq_original"
)
trino_runner.DEFAULT_SOURCE = "run-nasdaq-queries"


if __name__ == "__main__":
    trino_runner.main()
