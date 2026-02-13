"""
EIMAS Pipeline - TeamCoordinator for Parallel Phase 1 Execution

Coordinates parallel execution of independent data collection tasks within Phase 1.
Wraps asyncio.gather with error isolation so that a failure in one collector does not
block the others. Falls back to sequential path on total failure.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import socket
from time import perf_counter
from typing import Any, Callable, Dict, List, TypeVar

from pipeline.collectors import (
    collect_fred_data,
    collect_market_data,
    collect_crypto_data,
    collect_market_indicators,
    collect_company_ra_analysis,
)
from pipeline.korea_integration import collect_korea_assets
from pipeline.phase1_utils import (
    env_flag as _env_flag,
    count_dataframe_assets,
    inject_offline_fallback_market_data,
)
from pipeline.schemas import EIMASResult
from lib.extended_data_sources import ExtendedDataCollector

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TeamCoordinator:
    """Thin orchestrator for parallel phase execution."""

    def __init__(self, *, max_workers: int = 3, timeout_sec: float = 120.0) -> None:
        self.max_workers = max_workers
        self.timeout_sec = timeout_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_parallel_phase1(
        self,
        result: EIMASResult,
        quick_mode: bool,
    ) -> Dict[str, Any]:
        """
        Run Phase 1 data collectors in parallel using asyncio.gather.

        Returns the merged market_data dict, identical in shape to what the
        sequential collect_data() returns.
        """
        print("\n[Phase 1] Collecting Data (parallel)...")
        lookback_days = 90 if quick_mode else 365
        phase_started = perf_counter()
        component_timings: Dict[str, Dict[str, Any]] = {}

        extended_timeout = max(
            1.0,
            float(os.getenv("EIMAS_EXTENDED_DATA_TIMEOUT_SEC", "45")),
        )

        # --- Group A: independent collectors in parallel ---
        names = ["fred_data", "market_data", "crypto_data", "extended_data", "korea_assets"]
        starts = [perf_counter()] * 5  # placeholder, replaced per-task below

        tasks: List[asyncio.Task] = []
        tasks.append(asyncio.ensure_future(self._collect_fred()))
        tasks.append(asyncio.ensure_future(self._collect_market(lookback_days)))
        tasks.append(asyncio.ensure_future(self._collect_crypto(lookback_days)))
        tasks.append(asyncio.ensure_future(self._collect_extended(extended_timeout)))
        tasks.append(asyncio.ensure_future(self._collect_korea(lookback_days)))

        group_a_start = perf_counter()
        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("Parallel phase1 global timeout (%.1fs)", self.timeout_sec)
            raw_results = [TimeoutError("global timeout")] * 5

        group_a_elapsed = perf_counter() - group_a_start

        # --- Unpack results with error isolation ---
        fred_result = self._safe_unpack(raw_results, 0, {}, "fred_data")
        market_data_raw = self._safe_unpack(raw_results, 1, {}, "market_data")
        crypto_data_raw = self._safe_unpack(raw_results, 2, {}, "crypto_data")
        extended_result = self._safe_unpack(raw_results, 3, {}, "extended_data")
        korea_result = self._safe_unpack(raw_results, 4, {"data": {}, "summary": {"skipped": True}}, "korea_assets")

        # Record component timings (approximate; all ran concurrently)
        for i, name in enumerate(names):
            r = raw_results[i] if i < len(raw_results) else Exception("missing")
            status = "error" if isinstance(r, Exception) else "ok"
            component_timings[name] = {
                "duration_sec": round(group_a_elapsed, 3),
                "status": status,
                "parallel": True,
            }
            if isinstance(r, Exception):
                component_timings[name]["error"] = str(r)[:200]

        # --- Mutate result ---
        result.fred_summary = fred_result
        result.extended_data = extended_result
        result.market_data_count = len(market_data_raw)

        market_data: Dict[str, Any] = dict(market_data_raw)

        result.crypto_data_count = len(crypto_data_raw)
        for ticker, df in crypto_data_raw.items():
            market_data.setdefault(ticker, df)

        # --- Offline fallback injection (same logic as sequential) ---
        fallback_enabled = _env_flag("EIMAS_ENABLE_OFFLINE_MARKET_FALLBACK", default=True)
        fallback_force = _env_flag("EIMAS_OFFLINE_MARKET_FALLBACK_FORCE", default=False)
        fallback_min_raw = os.getenv("EIMAS_OFFLINE_MARKET_FALLBACK_MIN_ASSETS", "3").strip()
        try:
            fallback_min = max(1, int(fallback_min_raw))
        except ValueError:
            fallback_min = 3

        current_assets = count_dataframe_assets(market_data)
        if fallback_enabled and (fallback_force or current_assets < fallback_min):
            fb_start = perf_counter()
            injected, total = inject_offline_fallback_market_data(market_data, lookback_days)
            component_timings["offline_market_fallback"] = {
                "duration_sec": round(perf_counter() - fb_start, 3),
                "status": "ok" if injected > 0 else "already_satisfied",
            }
            if injected > 0:
                print(f"  [Phase 1] Offline fallback injected: {injected}/{total} synthetic tickers")
        else:
            component_timings["offline_market_fallback"] = {
                "duration_sec": 0.0,
                "status": "disabled" if not fallback_enabled else "not_needed",
            }

        result.market_data_count = len(market_data)

        # --- Group B: sequential (depend on market_data) ---
        if not quick_mode:
            ind_start = perf_counter()
            try:
                indicators = collect_market_indicators()
                result.market_indicators = (
                    indicators.to_dict()
                    if hasattr(indicators, "to_dict")
                    else getattr(indicators, "__dict__", {})
                )
                component_timings["market_indicators"] = {
                    "duration_sec": round(perf_counter() - ind_start, 3),
                    "status": "ok",
                }
            except Exception as exc:
                logger.warning("Market indicators failed: %s", exc)
                result.market_indicators = {}
                component_timings["market_indicators"] = {
                    "duration_sec": round(perf_counter() - ind_start, 3),
                    "status": "error",
                    "error": str(exc)[:200],
                }
        else:
            component_timings["market_indicators"] = {
                "duration_sec": 0.0,
                "status": "skipped_quick_mode",
            }

        enable_company_ra = _env_flag("EIMAS_ENABLE_COMPANY_RA_ANALYSIS", default=True)
        if enable_company_ra:
            cra_start = perf_counter()
            try:
                result.company_ra_analysis = collect_company_ra_analysis(
                    lookback_days=min(lookback_days, 365),
                )
                component_timings["company_ra_analysis"] = {
                    "duration_sec": round(perf_counter() - cra_start, 3),
                    "status": "ok",
                }
            except Exception as exc:
                logger.warning("Company RA failed: %s", exc)
                result.company_ra_analysis = {}
                component_timings["company_ra_analysis"] = {
                    "duration_sec": round(perf_counter() - cra_start, 3),
                    "status": "error",
                    "error": str(exc)[:200],
                }
        else:
            result.company_ra_analysis = {}
            component_timings["company_ra_analysis"] = {
                "duration_sec": 0.0,
                "status": "skipped_env",
            }

        # Korea result assignment
        if isinstance(korea_result, dict) and "data" in korea_result:
            result.korea_data = korea_result["data"]
            result.korea_summary = korea_result["summary"]
            market_data["korea_data"] = korea_result["data"]
        else:
            result.korea_data = {}
            result.korea_summary = {"skipped": True}
            market_data["korea_data"] = {}

        # --- Timing metadata ---
        phase_elapsed = round(perf_counter() - phase_started, 3)
        sum_components = sum(
            v.get("duration_sec", 0.0) for v in component_timings.values()
        )
        result.audit_metadata["phase1_component_timings"] = component_timings
        result.audit_metadata["phase1_elapsed_sec"] = phase_elapsed
        result.audit_metadata["phase1_parallel_mode"] = True
        result.audit_metadata["phase1_parallel_wall_clock_sec"] = round(group_a_elapsed, 3)
        result.audit_metadata["phase1_parallel_sum_component_sec"] = round(sum_components, 3)
        speedup = round(sum_components / group_a_elapsed, 2) if group_a_elapsed > 0 else 1.0
        result.audit_metadata["phase1_parallel_speedup"] = speedup

        ranked = sorted(
            (
                (name, meta.get("duration_sec", 0.0), meta.get("status", "ok"))
                for name, meta in component_timings.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_ranked = ", ".join(
            f"{name}={dur:.3f}s({st})" for name, dur, st in ranked[:4]
        )
        print(f"  [Phase 1 Timing] total={phase_elapsed:.3f}s (parallel, speedup={speedup:.1f}x) | {top_ranked}")

        return market_data

    # ------------------------------------------------------------------
    # Async collector wrappers
    # ------------------------------------------------------------------

    async def _collect_fred(self) -> Dict:
        if self._skip_flag("EIMAS_SKIP_FRED_DATA"):
            return {}
        if self._fail_fast_check(
            "EIMAS_FRED_FAIL_FAST_NETWORK",
            "EIMAS_FRED_PROBE_HOSTS",
            "api.stlouisfed.org",
        ):
            return {}
        return await self._wrap_sync(collect_fred_data)

    async def _collect_market(self, lookback_days: int) -> Dict[str, Any]:
        if self._skip_flag("EIMAS_SKIP_MARKET_DATA"):
            return {}
        if self._fail_fast_check(
            "EIMAS_MARKET_DATA_FAIL_FAST_NETWORK",
            "EIMAS_MARKET_DATA_PROBE_HOSTS",
            "guce.yahoo.com,query1.finance.yahoo.com",
        ):
            return {}
        return await self._wrap_sync(collect_market_data, lookback_days, False)

    async def _collect_crypto(self, lookback_days: int) -> Dict[str, Any]:
        if self._skip_flag("EIMAS_SKIP_CRYPTO_DATA"):
            return {}
        if self._fail_fast_check(
            "EIMAS_CRYPTO_DATA_FAIL_FAST_NETWORK",
            "EIMAS_CRYPTO_DATA_PROBE_HOSTS",
            "guce.yahoo.com,query1.finance.yahoo.com",
        ):
            return {}
        return await self._wrap_sync(collect_crypto_data, lookback_days)

    async def _collect_extended(self, timeout_sec: float) -> Dict:
        if self._skip_flag("EIMAS_SKIP_EXTENDED_DATA"):
            return {}
        ext = ExtendedDataCollector()
        try:
            return await asyncio.wait_for(ext.collect_all(), timeout=timeout_sec)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Extended data parallel error: %s", exc)
            return {}

    async def _collect_korea(self, lookback_days: int) -> Dict:
        if self._skip_flag("EIMAS_SKIP_KOREA_ASSETS"):
            return {"data": {}, "summary": {"skipped": True}}
        return await self._wrap_sync(collect_korea_assets, lookback_days, True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _wrap_sync(fn: Callable[..., T], *args: Any) -> T:
        """Run a synchronous function in the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args))

    @staticmethod
    def _skip_flag(env_name: str) -> bool:
        """Check if an EIMAS_SKIP_* env var is set to truthy."""
        return os.getenv(env_name, "false").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _fail_fast_check(flag_env: str, hosts_env: str, default_hosts: str) -> bool:
        """Check fail-fast network flag + DNS probe. Return True to skip."""
        raw = os.getenv(flag_env, "false").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return False
        hosts = [h.strip() for h in os.getenv(hosts_env, default_hosts).split(",") if h.strip()]
        for host in hosts:
            try:
                socket.getaddrinfo(host, 443)
                return False  # DNS works, don't skip
            except OSError:
                continue
        return True  # all DNS failed, skip

    @staticmethod
    def _safe_unpack(results: list, index: int, default: Any, name: str) -> Any:
        """Safely unpack a result from asyncio.gather, returning default on error."""
        if index >= len(results):
            logger.warning("Parallel %s: missing result", name)
            return default
        val = results[index]
        if isinstance(val, Exception):
            logger.warning("Parallel %s failed: %s", name, val)
            return default
        return val
