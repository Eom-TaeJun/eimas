"""
Shared bootstrap utility for direct script execution.

Scripts under `scripts/` can import project modules even when executed as:
    python scripts/<name>.py
"""

import sys
from pathlib import Path


def ensure_project_root(current_file: str, levels_up: int = 1) -> Path:
    """Insert project root into sys.path once and return the resolved root."""
    project_root = Path(current_file).resolve().parents[levels_up]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root

