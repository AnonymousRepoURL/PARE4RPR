"""Run rewritten Nasdaq SQL while reusing unchanged original results.

The input must be a semicolon-separated ``.txt`` file.  Its statements are
matched by position against ``data/nasdaq_sql.json`` and compared through
``queryanalysis.py``.  Parsed-identical queries reuse their complete entries
from ``experiments/oracle_nasdaq_original/query_times.json`` and copy the
original result text when it exists.  Parse failures and changed queries use
the Oracle execution flow from ``run_nasdaq_queries_oracle.py``.

Example:
    python experiments/runx_nasdaq_queries_oracle.py \
      --query-file data/nasdaq_sql_oracle_gpt-5.6-sol_v1_thinking.txt \
      --output-dir experiments/oracle_nasdaq_gpt5.6_rewriting
"""

from pathlib import Path

import runx_crimes_queries_oracle as oracle_runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent

oracle_runner.__doc__ = __doc__
oracle_runner.DEFAULT_ORIGINAL_QUERY_FILE = PROJECT_ROOT / "data" / "nasdaq_sql.json"
oracle_runner.DEFAULT_ORIGINAL_RESULTS_DIR = EXPERIMENTS_DIR / "oracle_nasdaq_original"
oracle_runner.DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR / "oracle_nasdaq_rewriting"


if __name__ == "__main__":
    oracle_runner.main()
