#!/usr/bin/env python3
"""
Event Framework - Impact Analyzer
============================================================

Event impact assessment and analysis

Class:
    - EventImpactAnalyzer: Event impact analyzer
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .schemas import Event, EventImpact
from .enums import EventType

logger = logging.getLogger(__name__)


class EventImpactAnalyzer:
    """이벤트 임팩트 분석"""

    # 과거 이벤트별 평균 임팩트
    HISTORICAL_IMPACT = {
        EventType.FOMC: {
            "avg_move_pct": 1.5,
            "avg_duration_hours": 48,
            "affected_assets": ["SPY", "QQQ", "TLT", "DXY", "GLD"],
            "typical_pattern": "Initial spike, then mean reversion over 24h"
        },
        EventType.CPI: {
            "avg_move_pct": 1.2,
            "avg_duration_hours": 8,
            "affected_assets": ["SPY", "QQQ", "TLT", "TIPS"],
            "typical_pattern": "Morning spike, settled by close"
        },
        EventType.NFP: {
            "avg_move_pct": 0.8,
            "avg_duration_hours": 4,
            "affected_assets": ["SPY", "DXY", "TLT"],
            "typical_pattern": "Quick reaction, often reversed"
        },
        EventType.EARNINGS: {
            "avg_move_pct": 5.0,
            "avg_duration_hours": 24,
            "affected_assets": ["individual_stock"],
            "typical_pattern": "Gap up/down, then trend for days"
        }
    }

    def get_expected_impact(self, event_type: EventType) -> Dict[str, Any]:
        """예상 임팩트 조회"""
        return self.HISTORICAL_IMPACT.get(event_type, {
            "avg_move_pct": 0.5,
            "avg_duration_hours": 4,
            "affected_assets": ["general"],
            "typical_pattern": "Unknown"
        })

    def analyze_historical_impact(
        self,
        event_type: EventType,
        ticker: str,
        lookback_events: int = 10
    ) -> Dict[str, Any]:
        """과거 이벤트 임팩트 분석"""
        # TODO: 실제 과거 데이터로 분석
        expected = self.get_expected_impact(event_type)

        return {
            "event_type": event_type.value,
            "ticker": ticker,
            "sample_size": lookback_events,
            "avg_move_pct": expected["avg_move_pct"],
            "avg_duration_hours": expected["avg_duration_hours"],
            "win_rate_long": 0.52,  # placeholder
            "win_rate_short": 0.48,
            "typical_pattern": expected["typical_pattern"]
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # 테스트
    framework = EventFramework(verbose=True)

    # 1. 다가오는 이벤트 조회
    print(framework.get_event_summary(days_ahead=14))

    # 2. 시장 데이터에서 이벤트 감지
    print("\n" + "=" * 60)
    print("Quantitative Event Detection")
    print("=" * 60)

    print("\n[1] Price/Volume/Volatility Detection for SPY...")
    spy = yf.download("SPY", period="3mo", progress=False)
    if not spy.empty:
        events = framework.quantitative_detector.detect_all(spy, ticker="SPY")
        print(f"Found {len(events)} events")
        for event in events[:5]:
            print(f"  [{event.importance.name}] {event.timestamp.strftime('%Y-%m-%d')}: {event.name}")

    # 3. 상관관계 이탈 감지
    print("\n[2] Correlation Breakdown Detection (SPY vs QQQ)...")
    qqq = yf.download("QQQ", period="6mo", progress=False)
    if not spy.empty and not qqq.empty:
        corr_events = framework.quantitative_detector.detect_correlation_breakdown(
            spy, qqq, "SPY", "QQQ"
        )
        if corr_events:
            for event in corr_events:
                print(f"  {event.name}: {event.description}")
        else:
            print("  No correlation breakdown detected")

    # 4. 옵션 플로우 감지 (선택적)
    print("\n[3] Options Flow Detection for SPY...")
    try:
        opt_events = framework.quantitative_detector.detect_options_unusual("SPY")
        if opt_events:
            for event in opt_events[:3]:
                print(f"  [{event.importance.name}] {event.name}")
        else:
            print("  No unusual options activity detected")
    except Exception as e:
        print(f"  Options detection skipped: {e}")

    # 5. 유동성 이벤트 감지 (FRED 연동)
    print("\n" + "=" * 60)
    print("Liquidity Event Detection (FRED)")
    print("=" * 60)

    try:
        from lib.fred_collector import FREDCollector

        print("\n[4] Fetching FRED liquidity data...")
        collector = FREDCollector()
        summary = collector.collect_all()

        print(f"  RRP: ${summary.rrp:.0f}B (delta: {summary.rrp_delta:+.0f}B)")
        print(f"  TGA: ${summary.tga:.0f}B (delta: {summary.tga_delta:+.0f}B)")
        print(f"  Fed Assets: ${summary.fed_assets:.2f}T (delta: {summary.fed_assets_delta:+.0f}B/wk)")
        print(f"  Net Liquidity: ${summary.net_liquidity/1000:.2f}T")
        print(f"  Regime: {summary.liquidity_regime}")

        liquidity_data = {
            'rrp': summary.rrp,
            'rrp_delta': summary.rrp_delta,
            'rrp_delta_pct': summary.rrp_delta_pct,
            'tga': summary.tga,
            'tga_delta': summary.tga_delta,
            'fed_assets_delta': summary.fed_assets_delta,
            'net_liquidity': summary.net_liquidity,
            'liquidity_regime': summary.liquidity_regime
        }

        liq_events = framework.quantitative_detector.detect_liquidity_events(**liquidity_data)
        print(f"\n  Found {len(liq_events)} liquidity events:")
        for event in liq_events:
            signal = event.metadata.get('signal', 'unknown')
            print(f"    [{event.importance.name}] {event.name}")
            print(f"      {event.description}")
            print(f"      Signal: {signal.upper()}")

    except Exception as e:
        print(f"  Liquidity detection skipped: {e}")

    # 6. 임팩트 분석
    print("\n" + "=" * 60)
    print("Event Impact Analysis")
    print("=" * 60)
    analyzer = EventImpactAnalyzer()
    for event_type in [EventType.FOMC, EventType.CPI, EventType.NFP]:
        impact = analyzer.get_expected_impact(event_type)
        print(f"\n{event_type.value.upper()}:")
        print(f"  Avg Move: {impact['avg_move_pct']}%")
        print(f"  Duration: {impact['avg_duration_hours']}h")
        print(f"  Pattern: {impact['typical_pattern']}")

    print("\n" + "=" * 60)
