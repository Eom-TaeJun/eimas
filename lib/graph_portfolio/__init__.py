#!/usr/bin/env python3
"""
Graph Portfolio Package
============================================================

Graph-Clustered Hierarchical Risk Parity (GC-HRP)

무한 에셋(N → ∞) 환경에서의 포트폴리오 최적화

Public API:
    - GraphClusteredPortfolio: Main portfolio optimizer
    - PortfolioAllocation: Allocation result
    - ClusteringMethod, RepresentativeMethod: Enums

Economic Foundation:
    - Network: Mantegna (1999) correlation-based networks
    - Clustering: Louvain (Blondel 2008) community detection
    - HRP: Lopez de Prado (2016) hierarchical risk parity
    - MST: Systemic risk identification

Usage:
    from lib.graph_portfolio import GraphClusteredPortfolio

    optimizer = GraphClusteredPortfolio(
        n_clusters=10,
        clustering_method=ClusteringMethod.LOUVAIN
    )
    allocation = optimizer.optimize(returns, volumes)
"""

from .portfolio import GraphClusteredPortfolio
from .schemas import PortfolioAllocation, ClusterInfo, MSTAnalysisResult
from .enums import ClusteringMethod, RepresentativeMethod

__all__ = [
    "GraphClusteredPortfolio",
    "PortfolioAllocation",
    "ClusterInfo",
    "MSTAnalysisResult",
    "ClusteringMethod",
    "RepresentativeMethod",
]
