#!/usr/bin/env python3
"""
Graph Portfolio - Enumerations
============================================================

클러스터링 및 대표 자산 선정 방법론
"""

from enum import Enum


class ClusteringMethod(Enum):
    """클러스터링 방법론"""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gmm"
    LOUVAIN = "louvain"  # Graph-based
    HIERARCHICAL = "hierarchical"


class RepresentativeMethod(Enum):
    """대표 자산 선정 방법"""
    CENTRALITY = "centrality"      # 그래프 중심성
    VOLUME = "volume"              # 거래량 기반
    LIQUIDITY = "liquidity"        # 유동성 기반
    SHARPE = "sharpe"              # 샤프 비율 기반
