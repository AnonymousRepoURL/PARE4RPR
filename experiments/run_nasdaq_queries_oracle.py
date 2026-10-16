"""Run Nasdaq SQL from JSON or plain text with the Oracle query runner.
Run command:
python experiments\run_nasdaq_queries_oracle.py

python experiments\run_nasdaq_queries_oracle.py --resume

This entry point uses the same execution, timeout, capped result capture,
third-run timeout inference, metrics, and resume behavior as
``run_crimes_queries_oracle.py``, including support for a JSON ``sqls`` array
or semicolon-separated SQL text. Its defaults keep the Nasdaq results separate
under ``oracle_nasdaq_original``.
"""

from pathlib import Path

import run_crimes_queries_oracle as oracle_runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
oracle_runner.__doc__ = __doc__
oracle_runner.DEFAULT_QUERY_FILE = PROJECT_ROOT / "data" / "nasdaq_sql.json"
oracle_runner.DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "oracle_nasdaq_original"
)


if __name__ == "__main__":
    oracle_runner.main()
