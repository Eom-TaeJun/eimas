"""
Pipeline runner (compatibility wrapper).

Historically this module tried to import `pipeline.collection.runner`,
`pipeline.analysis.runner`, etc., but those modules were archived.
To keep one canonical execution path for full mode, this runner now
delegates to `main.run_integrated_pipeline`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Awaitable, Callable

from pipeline.schemas import EIMASResult

_RUN_MAIN_PIPELINE: Callable[..., Awaitable[EIMASResult]] | None = None


def _get_main_pipeline_runner() -> Callable[..., Awaitable[EIMASResult]]:
    """Load canonical `main.py` by file path (independent of current cwd)."""
    global _RUN_MAIN_PIPELINE
    if _RUN_MAIN_PIPELINE is not None:
        return _RUN_MAIN_PIPELINE

    root_path = Path(__file__).resolve().parents[1]
    root_str = str(root_path)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    main_path = root_path / "main.py"
    spec = importlib.util.spec_from_file_location("eimas_main_canonical", main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load canonical main module from {main_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RUN_MAIN_PIPELINE = module.run_integrated_pipeline
    return _RUN_MAIN_PIPELINE


async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    output_dir: str = "outputs",
    cron_mode: bool = False,
    full_mode: bool = False,
) -> EIMASResult:
    """Compatibility entrypoint that forwards to the canonical main pipeline."""
    run_main_pipeline = _get_main_pipeline_runner()
    return await run_main_pipeline(
        enable_realtime=enable_realtime,
        realtime_duration=realtime_duration,
        quick_mode=quick_mode,
        generate_report=not quick_mode,
        full_mode=full_mode,
        output_dir=output_dir,
        cron_mode=cron_mode,
    )


async def run_pipeline(*args, **kwargs) -> EIMASResult:
    """Legacy alias kept for historical callers."""
    return await run_integrated_pipeline(*args, **kwargs)
