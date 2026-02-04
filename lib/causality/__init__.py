#!/usr/bin/env python3
"""
Causality Analysis Package
============================================================

Causal inference and network analysis for time series

Public API:
    - CausalityGraphEngine: Main graph engine
    - GrangerCausalityAnalyzer: Granger causality testing
    - CausalNetworkBuilder: Build causal networks
    - CausalNetworkAnalyzer: Analyze causal networks
    - Data classes and enums

Economic Foundation:
    - Granger (1969) causality
    - Causal inference in macroeconomic time series
    - Network propagation analysis

Usage:
    from lib.causality import CausalityGraphEngine
    
    engine = CausalityGraphEngine()
    insights = engine.analyze(data)
"""

from .graph import CausalityGraphEngine
from .granger import GrangerCausalityAnalyzer
from .builder import CausalNetworkBuilder
from .analyzer import CausalNetworkAnalyzer
from .schemas import (
    CausalNode,
    CausalEdge,
    CausalityPath,
    CausalityInsight,
    GrangerTestResult,
    NetworkAnalysisResult,
)
from .enums import (
    EdgeType,
    NodeType,
    CausalDirection,
)

__all__ = [
    "CausalityGraphEngine",
    "GrangerCausalityAnalyzer",
    "CausalNetworkBuilder",
    "CausalNetworkAnalyzer",
    "CausalNode",
    "CausalEdge",
    "CausalityPath",
    "CausalityInsight",
    "GrangerTestResult",
    "NetworkAnalysisResult",
    "EdgeType",
    "NodeType",
    "CausalDirection",
]
