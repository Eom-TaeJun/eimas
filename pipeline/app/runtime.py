"""
Runtime support helpers for orchestrating the integrated pipeline.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable


def resolve_output_path(output_dir: str, anchor_file: str) -> Path:
    """
    Resolve output directory path.

    Relative paths are anchored at the project root (directory of `anchor_file`)
    to preserve legacy behavior.
    """
    raw_output_path = Path(output_dir).expanduser()
    if raw_output_path.is_absolute():
        return raw_output_path
    return Path(anchor_file).resolve().parent / raw_output_path


class PhaseRuntimeTracker:
    """Tracks phase execution time and provides timed call wrappers."""

    def __init__(self, phase_timings: dict[str, dict[str, Any]]):
        self.phase_timings = phase_timings

    @staticmethod
    def format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"[:300]

    def _record_phase_timing(
        self,
        phase_name: str,
        started_at: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        elapsed = perf_counter() - started_at
        entry: dict[str, Any] = {
            "duration_sec": round(elapsed, 3),
            "status": status,
        }
        if error:
            entry["error"] = error
        self.phase_timings[phase_name] = entry
        print(f"  [Timing] {phase_name}: {elapsed:.3f}s ({status})")

    def run_sync(
        self,
        phase_name: str,
        fn: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        started = perf_counter()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:
            self._record_phase_timing(
                phase_name,
                started,
                status="error",
                error=self.format_error(exc),
            )
            raise
        self._record_phase_timing(phase_name, started, status="ok")
        return value

    async def run_async(
        self,
        phase_name: str,
        fn: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> Any:
        started = perf_counter()
        try:
            value = await fn(*args, **kwargs)
        except Exception as exc:
            self._record_phase_timing(
                phase_name,
                started,
                status="error",
                error=self.format_error(exc),
            )
            raise
        self._record_phase_timing(phase_name, started, status="ok")
        return value

    @staticmethod
    def print_pipeline_banner(output_path: Path, cron_mode: bool = False) -> None:
        print("=" * 70)
        print("  EIMAS - Integrated Analysis Pipeline")
        print("=" * 70)
        print(f"  Output Dir: {output_path}")
        if cron_mode:
            print("  Cron Mode: Enabled (report generation skipped)")

    def record_total(self, elapsed: float) -> None:
        self.phase_timings["pipeline_total"] = {
            "duration_sec": round(elapsed, 3),
            "status": "ok",
        }

    def print_timing_summary(self, top_n: int = 8) -> None:
        ranked = sorted(
            (
                (phase_name, meta)
                for phase_name, meta in self.phase_timings.items()
                if phase_name != "pipeline_total"
            ),
            key=lambda item: item[1].get("duration_sec", 0.0),
            reverse=True,
        )
        print("\n[Pipeline Timing Summary] Top 8")
        for phase_name, meta in ranked[:top_n]:
            print(
                f"  - {phase_name}: {meta.get('duration_sec', 0.0):.3f}s"
                f" ({meta.get('status', 'n/a')})"
            )

