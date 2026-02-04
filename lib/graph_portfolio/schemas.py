#!/usr/bin/env python3
"""
Graph Portfolio - Data Schemas
============================================================

포트폴리오 최적화 데이터 클래스

Economic Foundation:
    - ClusterInfo: Community detection results
    - SystemicRiskNode: MST centrality analysis (Mantegna 1999)
    - MSTAnalysisResult: Systemic risk identification
    - PortfolioAllocation: Final weight allocation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class ClusterInfo:
    """클러스터 정보"""
    cluster_id: int
    assets: List[str]
    representative: str
    intra_correlation: float  # 클러스터 내 평균 상관관계
    size: int
    total_volume: float = 0.0
    avg_volatility: float = 0.0


@dataclass
class SystemicRiskNode:
    """시스템 리스크 유발 노드 정보 (v2 - Eigenvector 제거)"""
    ticker: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    composite_score: float  # 종합 중심성 점수
    mst_connections: int    # MST에서의 연결 수
    risk_interpretation: str
    # v2: eigenvector_centrality 제거됨 (트리 구조에서 비효율적)


@dataclass
class OutlierDetectionResult:
    """DBSCAN 이상치 탐지 결과"""
    timestamp: str
    n_total_assets: int
    n_outliers: int
    outlier_ratio: float
    outlier_tickers: List[str]
    normal_tickers: List[str]
    cluster_labels: Dict[str, int]  # ticker -> cluster_id (-1 = noise)
    n_clusters: int
    eps: float  # DBSCAN epsilon 파라미터
    min_samples: int  # DBSCAN min_samples 파라미터
    interpretation: str


@dataclass
class MSTAnalysisResult:
    """MST 분석 결과"""
    timestamp: str
    n_nodes: int
    n_edges: int
    total_mst_weight: float
    avg_distance: float
    systemic_risk_nodes: List[SystemicRiskNode]
    mst_edges: List[Tuple[str, str, float]]  # (node1, node2, distance)
    risk_summary: str


@dataclass
class PortfolioAllocation:
    """포트폴리오 배분 결과"""
    timestamp: str
    weights: Dict[str, float]
    cluster_weights: Dict[int, float]
    risk_contributions: Dict[str, float]
    expected_volatility: float
    diversification_ratio: float
    effective_n: float  # 실효 자산 수
    methodology: str
    clusters: List[ClusterInfo] = field(default_factory=list)
    mst_analysis: Optional[MSTAnalysisResult] = None  # MST 시스템 리스크 분석

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['clusters'] = [asdict(c) for c in self.clusters]
        if self.mst_analysis:
            result['mst_analysis'] = {
                'timestamp': self.mst_analysis.timestamp,
                'n_nodes': self.mst_analysis.n_nodes,
                'n_edges': self.mst_analysis.n_edges,
                'total_mst_weight': self.mst_analysis.total_mst_weight,
                'avg_distance': self.mst_analysis.avg_distance,
                'systemic_risk_nodes': [
                    {
                        'ticker': n.ticker,
                        'degree_centrality': n.degree_centrality,
                        'betweenness_centrality': n.betweenness_centrality,
                        'closeness_centrality': n.closeness_centrality,
                        'composite_score': n.composite_score,
                        'mst_connections': n.mst_connections,
                        'risk_interpretation': n.risk_interpretation
                    }
                    for n in self.mst_analysis.systemic_risk_nodes
                ],
                'risk_summary': self.mst_analysis.risk_summary
            }
        return result

