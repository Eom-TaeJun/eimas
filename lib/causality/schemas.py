#!/usr/bin/env python3
"""
Causality Analysis - Data Schemas
============================================================

Data classes for causality analysis results

Economic Foundation:
    - Granger (1969) causality framework
    - Causal graph representation
    - Network analysis metrics
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .enums import EdgeType, NodeType, CausalDirection


@dataclass
class CausalNode:
    """인과관계 그래프 노드"""
    id: str
    name: str
    node_type: NodeType
    layer: str = ""                              # 공급망 레이어

    # 메트릭
    centrality_score: float = 0.0                # 중심성 점수
    pagerank: float = 0.0                        # PageRank
    is_bottleneck: bool = False                  # 병목점 여부
    criticality: float = 0.0                     # 중요도 (0-1)

    # 시장 데이터
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    volume_ratio: float = 1.0

    # 메타데이터
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.node_type.value,
            'layer': self.layer,
            'centrality': self.centrality_score,
            'pagerank': self.pagerank,
            'is_bottleneck': self.is_bottleneck,
            'criticality': self.criticality,
            'price_change_1d': self.price_change_1d,
            'volume_ratio': self.volume_ratio
        }


@dataclass
class CausalEdge:
    """인과관계 그래프 엣지"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0                          # 연결 강도
    lag: int = 0                                 # 시차 (일)
    direction: str = "forward"                   # forward, backward, bidirectional
    confidence: float = 0.8                      # 신뢰도

    # Granger Causality 결과
    granger_pvalue: float = 1.0                  # p-value (낮을수록 유의)
    granger_fstat: float = 0.0                   # F-statistic

    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.edge_type.value,
            'weight': self.weight,
            'lag': self.lag,
            'direction': self.direction,
            'confidence': self.confidence,
            'granger_pvalue': self.granger_pvalue
        }


@dataclass
class CausalityPath:
    """인과관계 경로"""
    nodes: List[str]
    edges: List[CausalEdge]
    total_weight: float = 0.0
    total_lag: int = 0
    path_type: str = ""                          # supply_chain, correlation, mixed
    narrative: str = ""                          # LLM 생성 내러티브

    def to_string(self) -> str:
        return " → ".join(self.nodes)


@dataclass
class CausalityInsight:
    """인과관계 인사이트"""
    path: CausalityPath
    insight_type: str                            # bottleneck_risk, shock_propagation, hub_influence
    severity: str                                # low, medium, high, critical
    confidence: float
    narrative: str
    affected_assets: List[str]
    recommended_action: str = ""

    def to_dict(self) -> Dict:
        return {
            'path': self.path.to_string(),
            'type': self.insight_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'narrative': self.narrative,
            'affected_assets': self.affected_assets,
            'action': self.recommended_action
        }


# =============================================================================
# Causality Graph Engine
# =============================================================================

@dataclass
class GrangerTestResult:
    """Granger Causality 검정 결과"""
    cause: str                  # 원인 변수
    effect: str                 # 결과 변수
    optimal_lag: int            # 최적 래그
    p_value: float              # p-value (F-test)
    f_statistic: float          # F-statistic
    is_significant: bool        # 유의미 여부
    direction: CausalDirection
    lead_time_days: int         # 선행 일수


@dataclass
class NetworkAnalysisResult:
    """네트워크 분석 결과"""
    target_variable: str
    all_paths: List[CausalPath]
    critical_path: Optional[CausalPath]  # 가장 중요한 경로
    key_drivers: List[str]          # 핵심 선행 지표
    network_stats: Dict[str, Any]   # 네트워크 통계
    granger_results: List[GrangerTestResult]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Granger Causality Analyzer
# ============================================================================
