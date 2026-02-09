"""
Runtime support helpers for orchestrating the integrated pipeline.
"""

from __future__ import annotations

import json
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

    def apply_result_timing_metadata(
        self,
        result: Any,
        elapsed: float,
        recorded_at: str,
    ) -> None:
        elapsed_rounded = round(elapsed, 3)
        result.pipeline_elapsed_sec = elapsed_rounded
        result.audit_metadata["pipeline_elapsed_sec"] = elapsed_rounded
        result.audit_metadata["pipeline_phase_count"] = len(self.phase_timings) - 1
        result.audit_metadata["pipeline_timing_recorded_at"] = recorded_at

    def persist_final_snapshot(self, output_file: str | None, payload: Any) -> None:
        """Persist final payload to the same output JSON file if available."""
        if not output_file:
            return
        try:
            target_path = Path(output_file).expanduser()
            target_path.parent.mkdir(exist_ok=True, parents=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"  Final snapshot updated: {target_path}")
        except Exception as exc:
            print(f"⚠️ Final snapshot update failed: {self.format_error(exc)}")

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

    @staticmethod
    def print_pipeline_completion(elapsed: float, output_file: str | None) -> None:
        print("\n" + "=" * 70)
        print(f"EIMAS PIPELINE COMPLETE ({elapsed:.1f}s)")
        print(f"Output: {output_file}")
        print("=" * 70)
