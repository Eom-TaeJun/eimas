#!/usr/bin/env python3
"""
Shock Propagation Package
============================================================

Shock transmission network with Granger causality analysis

Public API:
    - ShockPropagationGraph: Main shock propagation analyzer
    - LeadLagAnalyzer: Lead-lag relationship detection
    - GrangerCausalityAnalyzer: Granger causality testing
    - PropagationAnalysis: Analysis result

Economic Foundation:
    - Granger Causality: Granger (1969)
    - Network topology: Asset correlation networks
    - Shock transmission: Path-based propagation analysis

Usage:
    from lib.shock_propagation import ShockPropagationGraph

    graph = ShockPropagationGraph()
    analysis = graph.analyze(returns_data)
"""

from .graph import ShockPropagationGraph
from .lead_lag import LeadLagAnalyzer
from .granger import GrangerCausalityAnalyzer
from .schemas import PropagationAnalysis, ShockPath, EconomicEdge, NodeAnalysis
from .enums import NodeLayer, CausalityStrength

__all__ = [
    "ShockPropagationGraph",
    "LeadLagAnalyzer",
    "GrangerCausalityAnalyzer",
    "PropagationAnalysis",
    "ShockPath",
    "EconomicEdge",
    "NodeAnalysis",
    "NodeLayer",
    "CausalityStrength",
]
