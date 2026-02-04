#!/usr/bin/env python3
"""
ETF Flow Analyzer Package
============================================================

ETF fund flow analysis and sector rotation detection

Public API:
    - ETFFlowAnalyzer: Main analyzer
    - Data classes and enums

Usage:
    from lib.analyzers.etf import ETFFlowAnalyzer
    
    analyzer = ETFFlowAnalyzer()
    result = analyzer.analyze()
"""

from .flow_analyzer import ETFFlowAnalyzer
from .schemas import ETFData, FlowComparison, SectorRotationResult, MarketRegimeResult
from .enums import MarketSentiment, StyleRotation, CyclePhase

__all__ = [
    "ETFFlowAnalyzer",
    "ETFData",
    "FlowComparison",
    "SectorRotationResult",
    "MarketRegimeResult",
    "MarketSentiment",
    "StyleRotation",
    "CyclePhase",
]
