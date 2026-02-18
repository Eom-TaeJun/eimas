#!/usr/bin/env python3
"""
lib/causal_network.py — Shim (패키지 호환 레이어)
실제 구현은 lib/causality/ 패키지로 이동됨.

    from lib.causal_network import CausalNetworkBuilder  # 기존 경로 (유지)
    from lib.causality import CausalNetworkBuilder       # 신규 경로 (권장)
"""
from lib.causality import (
    CausalNetworkBuilder,
    CausalNetworkAnalyzer,
    CausalityGraphEngine,
    GrangerCausalityAnalyzer,
    CausalNode,
    CausalEdge,
    CausalityPath,
    CausalityInsight,
    GrangerTestResult,
    NetworkAnalysisResult,
    EdgeType,
    NodeType,
    CausalDirection,
)

# 구 이름 호환 alias
CausalPath = CausalityPath

__all__ = [
    "CausalNetworkBuilder",
    "CausalNetworkAnalyzer",
    "CausalityGraphEngine",
    "GrangerCausalityAnalyzer",
    "CausalNode",
    "CausalEdge",
    "CausalityPath",
    "CausalPath",       # alias
    "CausalityInsight",
    "GrangerTestResult",
    "NetworkAnalysisResult",
    "EdgeType",
    "NodeType",
    "CausalDirection",
]
