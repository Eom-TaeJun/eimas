#!/usr/bin/env python3
"""
Microstructure - Utilities
============================================================

Convenience functions for quick analysis
"""

from typing import Dict, Any, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def quick_analysis(symbol: str = 'BTC/USDT', samples: int = 10) -> Dict[str, Any]:
    """
    빠른 분석 실행

    Parameters:
    -----------
    symbol : str
        분석할 심볼
    samples : int
        샘플 수

    Returns:
    --------
    Dict with analysis results
    """
    fetcher = ExchangeDataFetcher('binance')
    analyzer = MicrostructureAnalyzer()

    results = []

    for i in range(samples):
        try:
            # 호가창
            ob = fetcher.fetch_orderbook(symbol, limit=10)
            metrics = analyzer.process_orderbook(ob)
            results.append(metrics)

            # 체결
            if i == 0:
                trades = fetcher.fetch_trades(symbol, limit=100)
                for t in trades:
                    analyzer.process_trade(t)

            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"Sample {i} error: {e}")

    if not results:
        return {'error': 'No data collected'}

    # 요약
    final = results[-1]
    ofi_values = [r.ofi_normalized for r in results]
    depth_values = [r.depth_ratio for r in results]

    return {
        'symbol': symbol,
        'samples': len(results),
        'mid_price': final.mid_price,
        'spread_bps': final.spread_bps,
        'ofi_current': final.ofi_normalized,
        'ofi_mean': np.mean(ofi_values),
        'ofi_std': np.std(ofi_values),
        'vpin': final.vpin,
        'depth_ratio': final.depth_ratio,
        'depth_mean': np.mean(depth_values),
        'signal': final.signal,
        'signal_strength': final.signal_strength
    }


