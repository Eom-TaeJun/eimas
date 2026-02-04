#!/usr/bin/env python3
"""
Bubble Detection - Utilities
============================================================

Utility functions for quick bubble checks and scanning

Functions:
    - quick_bubble_check: Quick check for a single ticker
    - scan_for_bubbles: Scan multiple tickers for bubbles
"""

from typing import Dict, List, Any
import logging

from .detector import BubbleDetector

logger = logging.getLogger(__name__)


def quick_bubble_check(ticker: str) -> Dict[str, Any]:
    """빠른 버블 체크"""
    detector = BubbleDetector()
    result = detector.analyze(ticker)

    return {
        'ticker': ticker,
        'warning_level': result.bubble_warning_level.value,
        'risk_score': result.risk_score,
        'runup_return': f"{result.runup.cumulative_return * 100:.1f}%",
        'is_runup': result.runup.is_runup,
        'volatility_zscore': result.volatility.zscore,
        'risk_signals': len(result.risk_signals)
    }


def scan_for_bubbles(tickers: List[str]) -> Dict[str, Dict]:
    """여러 티커 버블 스캔"""
    detector = BubbleDetector()
    results = detector.analyze_multiple(tickers)

    summary = {}
    for ticker, result in results.items():
        summary[ticker] = {
            'level': result.bubble_warning_level.value,
            'score': result.risk_score,
            'runup': f"{result.runup.cumulative_return * 100:.1f}%",
            'signals': [s.signal_type.value for s in result.risk_signals]
        }

    return summary


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Bubble Detector Test")
    print("=" * 60)

    detector = BubbleDetector()

    # 테스트 티커 (다양한 유형)
    test_tickers = ['TSLA', 'NVDA', 'AAPL', 'SPY', 'GME']

    for ticker in test_tickers:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {ticker}")
        print('=' * 60)

        try:
            result = detector.analyze(ticker)

            print(result.get_summary())
            print()

        except Exception as e:
            print(f"Error: {e}")

    # 요약 스캔
    print("\n" + "=" * 60)
    print("Bubble Scan Summary")
    print("=" * 60)

    try:
        scan_results = scan_for_bubbles(test_tickers)

        for ticker, data in scan_results.items():
            icon = {
                'NONE': '',
                'WATCH': '',
                'WARNING': '',
                'DANGER': ''
            }.get(data['level'], '')

            print(f"  {icon} {ticker}: {data['level']} (Score: {data['score']:.0f}, Run-up: {data['runup']})")

    except Exception as e:
        print(f"Scan error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
