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

from typing import Dict, Any

from pipeline.collectors import (
    collect_fred_data,
    collect_market_data,
    collect_crypto_data,
    collect_market_indicators,
)
from pipeline.schemas import EIMASResult
from pipeline.korea_integration import collect_korea_assets
from lib.extended_data_sources import ExtendedDataCollector


async def collect_data(result: EIMASResult, quick_mode: bool) -> Dict[str, Any]:
    """[Phase 1] Collect FRED/market/crypto/extended/Korea datasets."""
    print("\n[Phase 1] Collecting Data...")

    result.fred_summary = collect_fred_data()

    ext_collector = ExtendedDataCollector()
    result.extended_data = await ext_collector.collect_all()

    market_data = collect_market_data(lookback_days=90 if quick_mode else 365)
    result.market_data_count = len(market_data)

    crypto_data = collect_crypto_data(lookback_days=90 if quick_mode else 365)
    result.crypto_data_count = len(crypto_data)
    for ticker, df in crypto_data.items():
        market_data.setdefault(ticker, df)

    if not quick_mode:
        indicators = collect_market_indicators()
        result.market_indicators = (
            indicators.to_dict()
            if hasattr(indicators, "to_dict")
            else getattr(indicators, "__dict__", {})
        )

    print("\n[Phase 1.4] Collecting Korea Assets...")
    korea_result = collect_korea_assets(
        lookback_days=90 if quick_mode else 365,
        use_parallel=True,
    )
    result.korea_data = korea_result["data"]
    result.korea_summary = korea_result["summary"]
    print(f"  ✓ Korea assets: {korea_result['summary'].get('total_assets', 0)} collected")

    # Keep korea data inside market_data for downstream computations.
    market_data["korea_data"] = korea_result["data"]
    return market_data
