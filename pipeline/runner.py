"""
Pipeline runner (compatibility wrapper).

Legacy split runners were archived. To keep one canonical execution path
for full mode, this wrapper delegates to `main.run_integrated_pipeline`.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from pipeline.schemas import EIMASResult

_RUN_MAIN_PIPELINE: Callable[..., Awaitable[EIMASResult]] | None = None


def _get_main_pipeline_runner() -> Callable[..., Awaitable[EIMASResult]]:
    """Load canonical runner from top-level `main.py`."""
    global _RUN_MAIN_PIPELINE
    if _RUN_MAIN_PIPELINE is not None:
        return _RUN_MAIN_PIPELINE

    from main import run_integrated_pipeline as run_main_pipeline

    _RUN_MAIN_PIPELINE = run_main_pipeline
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
