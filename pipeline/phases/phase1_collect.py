#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 1: Data Collection

Purpose:
    Data collection from FRED, yfinance, crypto/RWA, and Korea sources

Input:
    - None

Output:
    - market_data: Dict[str, Any]

Functions:
    - collect_data: Main data collection orchestrator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

import asyncio
import hashlib
import os
import socket
from time import perf_counter
from typing import Dict, Any

import numpy as np
import pandas as pd

from pipeline.collectors import (
    collect_fred_data,
    collect_market_data,
    collect_crypto_data,
    collect_market_indicators,
)
from pipeline.schemas import EIMASResult
from pipeline.korea_integration import collect_korea_assets
from lib.extended_data_sources import ExtendedDataCollector


_OFFLINE_FALLBACK_TICKERS = (
    "SPY,QQQ,TLT,GLD,^VIX,HYG,LQD,XLY,XLP,IWM,XLF,"
    "BTC-USD,SMH,NVDA,DXY,DX-Y.NYB,EEM"
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_probe_hosts(env_name: str, default_hosts: str) -> list[str]:
    raw = os.getenv(env_name, default_hosts)
    hosts = [item.strip() for item in raw.split(",") if item.strip()]
    return hosts


def _is_dns_available(hosts: list[str]) -> bool:
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            continue
    return False


def _resolve_fallback_tickers() -> list[str]:
    raw = os.getenv("EIMAS_OFFLINE_MARKET_FALLBACK_TICKERS", _OFFLINE_FALLBACK_TICKERS)
    tickers = [item.strip() for item in raw.split(",") if item.strip()]
    return tickers


def _count_dataframe_assets(market_data: Dict[str, Any]) -> int:
    count = 0
    for payload in market_data.values():
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            count += 1
    return count


def _seed_for_ticker(ticker: str) -> int:
    digest = hashlib.sha256(ticker.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _base_price_for_ticker(ticker: str) -> float:
    defaults = {
        "SPY": 520.0,
        "QQQ": 450.0,
        "TLT": 95.0,
        "GLD": 210.0,
        "^VIX": 18.0,
        "VIX": 18.0,
        "HYG": 78.0,
        "LQD": 108.0,
        "XLY": 180.0,
        "XLP": 76.0,
        "IWM": 205.0,
        "XLF": 43.0,
        "BTC-USD": 65000.0,
        "SMH": 250.0,
        "NVDA": 900.0,
        "DXY": 104.0,
        "DX-Y.NYB": 104.0,
        "EEM": 43.0,
    }
    return defaults.get(ticker, 100.0)


def _build_synthetic_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    points = max(180, min(lookback_days, 260))
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=points)
    rng = np.random.default_rng(_seed_for_ticker(ticker))

    base_price = _base_price_for_ticker(ticker)
    if ticker in {"^VIX", "VIX"}:
        drift = 0.00005
        vol = 0.012
    elif ticker == "BTC-USD":
        drift = 0.0007
        vol = 0.022
    elif ticker in {"SMH", "NVDA", "QQQ"}:
        drift = 0.0006
        vol = 0.018
    elif ticker in {"TLT", "LQD", "HYG"}:
        drift = 0.0001
        vol = 0.007
    else:
        drift = 0.0003
        vol = 0.011

    returns = drift + rng.normal(0, vol, size=points)
    close = base_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, vol * 0.2, size=points))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.18, size=points)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.18, size=points)))
    low = np.maximum(low, 0.01)
    volume = rng.integers(1_000_000, 9_000_000, size=points)

    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def _inject_offline_fallback_market_data(
    market_data: Dict[str, Any],
    lookback_days: int,
) -> tuple[int, int]:
    fallback_tickers = _resolve_fallback_tickers()
    injected = 0

    for ticker in fallback_tickers:
        payload = market_data.get(ticker)
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            continue
        market_data[ticker] = _build_synthetic_ohlcv(ticker, lookback_days=lookback_days)
        injected += 1

    return injected, len(fallback_tickers)


async def collect_data(result: EIMASResult, quick_mode: bool) -> Dict[str, Any]:
    """[Phase 1] Collect FRED/market/crypto/extended/Korea datasets."""
    print("\n[Phase 1] Collecting Data...")
    lookback_days = 90 if quick_mode else 365
    phase_started = perf_counter()
    phase1_component_timings: Dict[str, Dict[str, Any]] = {}

    def _record_component_timing(
        name: str,
        started_at: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        duration = perf_counter() - started_at
        entry: Dict[str, Any] = {
            "duration_sec": round(duration, 3),
            "status": status,
        }
        if error:
            entry["error"] = error[:200]
        phase1_component_timings[name] = entry

    fred_started = perf_counter()
    result.fred_summary = collect_fred_data()
    _record_component_timing("fred_data", fred_started, status="ok")

    ext_collector = ExtendedDataCollector()
    extended_data_timeout_sec = max(
        1.0,
        float(os.getenv("EIMAS_EXTENDED_DATA_TIMEOUT_SEC", "45")),
    )
    skip_extended_data = os.getenv(
        "EIMAS_SKIP_EXTENDED_DATA",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    extended_fail_fast_network = os.getenv(
        "EIMAS_EXTENDED_FAIL_FAST_NETWORK",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    extended_started = perf_counter()
    if skip_extended_data:
        result.extended_data = {}
        print("  [Phase 1] Extended data skipped by EIMAS_SKIP_EXTENDED_DATA")
        _record_component_timing("extended_data", extended_started, status="skipped_env")
    elif extended_fail_fast_network:
        probe_hosts = [
            host.strip()
            for host in os.getenv(
                "EIMAS_EXTENDED_NETWORK_PROBE_HOSTS",
                "api.llama.fi,api.alternative.me,search.cnbc.com",
            ).split(",")
            if host.strip()
        ]
        dns_available = False
        for host in probe_hosts:
            try:
                socket.getaddrinfo(host, 443)
                dns_available = True
                break
            except OSError:
                continue

        if not dns_available:
            result.extended_data = {}
            print(
                "  [Phase 1] Extended data fail-fast: DNS unavailable, skipping extended sources"
            )
            _record_component_timing(
                "extended_data",
                extended_started,
                status="skipped_network",
            )
        else:
            try:
                result.extended_data = await asyncio.wait_for(
                    ext_collector.collect_all(),
                    timeout=extended_data_timeout_sec,
                )
                _record_component_timing("extended_data", extended_started, status="ok")
            except asyncio.TimeoutError:
                print(
                    f"  [Phase 1] Extended data timeout ({extended_data_timeout_sec:.0f}s) - continuing with empty payload"
                )
                result.extended_data = {}
                _record_component_timing("extended_data", extended_started, status="timeout")
            except Exception as exc:
                print(f"  [Phase 1] Extended data collection error: {exc}")
                result.extended_data = {}
                _record_component_timing(
                    "extended_data",
                    extended_started,
                    status="error",
                    error=str(exc),
                )
    else:
        try:
            result.extended_data = await asyncio.wait_for(
                ext_collector.collect_all(),
                timeout=extended_data_timeout_sec,
            )
            _record_component_timing("extended_data", extended_started, status="ok")
        except asyncio.TimeoutError:
            print(
                f"  [Phase 1] Extended data timeout ({extended_data_timeout_sec:.0f}s) - continuing with empty payload"
            )
            result.extended_data = {}
            _record_component_timing("extended_data", extended_started, status="timeout")
        except Exception as exc:
            print(f"  [Phase 1] Extended data collection error: {exc}")
            result.extended_data = {}
            _record_component_timing(
                "extended_data",
                extended_started,
                status="error",
                error=str(exc),
            )

    # Avoid BTC/ETH duplicate download: crypto set is collected in phase 1.3.
    skip_market_data = os.getenv(
        "EIMAS_SKIP_MARKET_DATA",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    market_fail_fast = os.getenv(
        "EIMAS_MARKET_DATA_FAIL_FAST_NETWORK",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_market_data:
        market_data = {}
        phase1_component_timings["market_data"] = {
            "duration_sec": 0.0,
            "status": "skipped_env",
        }
        print("  [Phase 1] Market data skipped by EIMAS_SKIP_MARKET_DATA")
    elif market_fail_fast:
        probe_hosts = _resolve_probe_hosts(
            "EIMAS_MARKET_DATA_PROBE_HOSTS",
            "guce.yahoo.com,query1.finance.yahoo.com",
        )
        if not _is_dns_available(probe_hosts):
            market_data = {}
            phase1_component_timings["market_data"] = {
                "duration_sec": 0.0,
                "status": "skipped_network",
            }
            print("  [Phase 1] Market data fail-fast: DNS unavailable, skipping")
        else:
            market_started = perf_counter()
            market_data = collect_market_data(lookback_days=lookback_days, include_crypto=False)
            _record_component_timing("market_data", market_started, status="ok")
    else:
        market_started = perf_counter()
        market_data = collect_market_data(lookback_days=lookback_days, include_crypto=False)
        _record_component_timing("market_data", market_started, status="ok")
    result.market_data_count = len(market_data)

    skip_crypto_data = os.getenv(
        "EIMAS_SKIP_CRYPTO_DATA",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    crypto_fail_fast = os.getenv(
        "EIMAS_CRYPTO_DATA_FAIL_FAST_NETWORK",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_crypto_data:
        crypto_data = {}
        phase1_component_timings["crypto_data"] = {
            "duration_sec": 0.0,
            "status": "skipped_env",
        }
        print("  [Phase 1] Crypto data skipped by EIMAS_SKIP_CRYPTO_DATA")
    elif crypto_fail_fast:
        probe_hosts = _resolve_probe_hosts(
            "EIMAS_CRYPTO_DATA_PROBE_HOSTS",
            "guce.yahoo.com,query1.finance.yahoo.com",
        )
        if not _is_dns_available(probe_hosts):
            crypto_data = {}
            phase1_component_timings["crypto_data"] = {
                "duration_sec": 0.0,
                "status": "skipped_network",
            }
            print("  [Phase 1] Crypto data fail-fast: DNS unavailable, skipping")
        else:
            crypto_started = perf_counter()
            crypto_data = collect_crypto_data(lookback_days=lookback_days)
            _record_component_timing("crypto_data", crypto_started, status="ok")
    else:
        crypto_started = perf_counter()
        crypto_data = collect_crypto_data(lookback_days=lookback_days)
        _record_component_timing("crypto_data", crypto_started, status="ok")
    result.crypto_data_count = len(crypto_data)
    for ticker, df in crypto_data.items():
        market_data.setdefault(ticker, df)

    fallback_enabled = _env_flag("EIMAS_ENABLE_OFFLINE_MARKET_FALLBACK", default=True)
    fallback_force = _env_flag("EIMAS_OFFLINE_MARKET_FALLBACK_FORCE", default=False)
    fallback_min_assets_raw = os.getenv("EIMAS_OFFLINE_MARKET_FALLBACK_MIN_ASSETS", "3").strip()
    try:
        fallback_min_assets = max(1, int(fallback_min_assets_raw))
    except ValueError:
        fallback_min_assets = 3

    current_df_assets = _count_dataframe_assets(market_data)
    needs_fallback = current_df_assets < fallback_min_assets

    if fallback_enabled and (fallback_force or needs_fallback):
        fallback_started = perf_counter()
        injected_count, fallback_total = _inject_offline_fallback_market_data(
            market_data,
            lookback_days=lookback_days,
        )
        status = "ok" if injected_count > 0 else "already_satisfied"
        _record_component_timing("offline_market_fallback", fallback_started, status=status)
        if injected_count > 0:
            print(
                "  [Phase 1] Offline fallback injected: "
                f"{injected_count}/{fallback_total} synthetic tickers"
            )
    else:
        phase1_component_timings["offline_market_fallback"] = {
            "duration_sec": 0.0,
            "status": "disabled" if not fallback_enabled else "not_needed",
        }

    result.market_data_count = len(market_data)

    if not quick_mode:
        skip_market_indicators = os.getenv(
            "EIMAS_SKIP_MARKET_INDICATORS",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        market_indicators_fail_fast = os.getenv(
            "EIMAS_MARKET_INDICATORS_FAIL_FAST_NETWORK",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if skip_market_indicators:
            result.market_indicators = {}
            phase1_component_timings["market_indicators"] = {
                "duration_sec": 0.0,
                "status": "skipped_env",
            }
            print("  [Phase 1] Market indicators skipped by EIMAS_SKIP_MARKET_INDICATORS")
        elif market_indicators_fail_fast:
            probe_hosts = _resolve_probe_hosts(
                "EIMAS_MARKET_INDICATORS_PROBE_HOSTS",
                "guce.yahoo.com,query1.finance.yahoo.com",
            )
            if not _is_dns_available(probe_hosts):
                result.market_indicators = {}
                phase1_component_timings["market_indicators"] = {
                    "duration_sec": 0.0,
                    "status": "skipped_network",
                }
                print("  [Phase 1] Market indicators fail-fast: DNS unavailable, skipping")
            else:
                indicators_started = perf_counter()
                indicators = collect_market_indicators()
                _record_component_timing("market_indicators", indicators_started, status="ok")
                result.market_indicators = (
                    indicators.to_dict()
                    if hasattr(indicators, "to_dict")
                    else getattr(indicators, "__dict__", {})
                )
        else:
            indicators_started = perf_counter()
            indicators = collect_market_indicators()
            _record_component_timing("market_indicators", indicators_started, status="ok")
            result.market_indicators = (
                indicators.to_dict()
                if hasattr(indicators, "to_dict")
                else getattr(indicators, "__dict__", {})
            )
    else:
        phase1_component_timings["market_indicators"] = {
            "duration_sec": 0.0,
            "status": "skipped_quick_mode",
        }

    skip_korea_assets = os.getenv(
        "EIMAS_SKIP_KOREA_ASSETS",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if skip_korea_assets:
        print("\n[Phase 1.4] Korea assets skipped by EIMAS_SKIP_KOREA_ASSETS")
        result.korea_data = {}
        result.korea_summary = {"skipped": True, "reason": "EIMAS_SKIP_KOREA_ASSETS"}
        market_data["korea_data"] = {}
        phase1_component_timings["korea_assets"] = {
            "duration_sec": 0.0,
            "status": "skipped_env",
        }
    else:
        print("\n[Phase 1.4] Collecting Korea Assets...")
        korea_started = perf_counter()
        korea_result = collect_korea_assets(
            lookback_days=lookback_days,
            use_parallel=True,
        )
        _record_component_timing("korea_assets", korea_started, status="ok")
        result.korea_data = korea_result["data"]
        result.korea_summary = korea_result["summary"]
        print(f"  ✓ Korea assets: {korea_result['summary'].get('total_assets', 0)} collected")

        # Keep korea data inside market_data for downstream computations.
        market_data["korea_data"] = korea_result["data"]

    phase_elapsed = round(perf_counter() - phase_started, 3)
    result.audit_metadata["phase1_component_timings"] = phase1_component_timings
    result.audit_metadata["phase1_elapsed_sec"] = phase_elapsed
    ranked = sorted(
        (
            (name, meta.get("duration_sec", 0.0), meta.get("status", "ok"))
            for name, meta in phase1_component_timings.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_ranked = ", ".join(
        f"{name}={duration:.3f}s({status})"
        for name, duration, status in ranked[:4]
    )
    print(f"  [Phase 1 Timing] total={phase_elapsed:.3f}s | {top_ranked}")
    return market_data
