"""Make project modules importable when a test file is run directly."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root = str(PROJECT_ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
