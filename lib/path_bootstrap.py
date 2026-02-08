"""Path bootstrap helpers for controlled dynamic imports."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_path(path: str | Path) -> None:
    """Insert a path once at front of sys.path for dynamic module loading."""
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

