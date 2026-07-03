"""Project-level defaults.

Keep these constants boring. The goal is for the early scripts to be easy to
change while you are still learning which abstractions are worth keeping.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
