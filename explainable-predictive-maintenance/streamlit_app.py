"""Portable Streamlit entry point, including deployment from the portfolio root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
for directory in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.main import main  # noqa: E402

main()
