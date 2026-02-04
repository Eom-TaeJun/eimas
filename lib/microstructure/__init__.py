#!/usr/bin/env python3
"""
Microstructure Package
============================================================

시장 미세구조 분석 통합 패키지

Public API:
    - MicrostructureAnalyzer: 실시간 미세구조 분석
    - DailyMicrostructureAnalyzer: 일별 AMFL Chapter 19 지표
    - RealtimeMicrostructureAnalyzer: 실시간 거래소 연동
    - MicrostructureMetrics: 결과 데이터 클래스
    - DailyMicrostructureResult: 일별 분석 결과

Usage:
    from lib.microstructure import MicrostructureAnalyzer, DailyMicrostructureAnalyzer

    # High-frequency analysis
    analyzer = MicrostructureAnalyzer()
    metrics = analyzer.analyze(order_book, trades)

    # Daily analysis (AMFL Chapter 19)
    daily_analyzer = DailyMicrostructureAnalyzer()
    result = daily_analyzer.analyze(market_data)
"""

from .analyzer import MicrostructureAnalyzer
from .daily import DailyMicrostructureAnalyzer, calculate_amihud, calculate_roll_spread_daily, calculate_vpin_daily
from .realtime import RealtimeMicrostructureAnalyzer
from .schemas import MicrostructureMetrics, DailyMicrostructureResult, OrderBook, OrderBookLevel, Trade
from .config import RollingWindowConfig
from .hft import tick_rule_classification, kyles_lambda, volume_clock_sampling, detect_quote_stuffing

__all__ = [
    "MicrostructureAnalyzer",
    "DailyMicrostructureAnalyzer",
    "RealtimeMicrostructureAnalyzer",
    "MicrostructureMetrics",
    "DailyMicrostructureResult",
    "RollingWindowConfig",
    # Data classes
    "OrderBook",
    "OrderBookLevel",
    "Trade",
    # HFT functions
    "tick_rule_classification",
    "kyles_lambda",
    "volume_clock_sampling",
    "detect_quote_stuffing",
    # Daily functions
    "calculate_amihud",
    "calculate_roll_spread_daily",
    "calculate_vpin_daily",
]
