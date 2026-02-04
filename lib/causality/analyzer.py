#!/usr/bin/env python3
"""
Causality Analysis - Network Analyzer
============================================================

Analyzes causal networks for insights

Class:
    - CausalNetworkAnalyzer: Network analysis and insights
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging

from .schemas import NetworkAnalysisResult, CausalEdge

logger = logging.getLogger(__name__)


class CausalNetworkAnalyzer:
    """
    인과관계 네트워크 분석 통합 클래스

    Granger Causality 분석 → 네트워크 구축 → 핵심 경로 추출
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05
    ):
        self.granger_analyzer = GrangerCausalityAnalyzer(
            max_lag=max_lag,
            significance_level=significance_level
        )
        self.network_builder = CausalNetworkBuilder()

    def analyze(
        self,
        data: pd.DataFrame,
        target_variable: str,
        variables: Optional[List[str]] = None,
        make_stationary: bool = True,
        max_paths: int = 10
    ) -> NetworkAnalysisResult:
        """
        전체 분석 실행

        Parameters:
        -----------
        data : DataFrame
            시계열 데이터
        target_variable : str
            분석 타겟 변수
        variables : List[str]
            분석할 변수 목록
        make_stationary : bool
            정상성 변환 여부
        max_paths : int
            반환할 최대 경로 수

        Returns:
        --------
        NetworkAnalysisResult
            분석 결과
        """
        # 1. Granger Causality 분석
        granger_results = self.granger_analyzer.test_all_pairs(
            data,
            variables=variables,
            make_stationary=make_stationary
        )

        # 2. 네트워크 구축
        self.network_builder.build_network(granger_results)

        # 3. 네트워크 통계
        network_stats = self.network_builder.get_network_stats()

        # 4. 핵심 드라이버 식별
        key_drivers_raw = self.network_builder.get_key_drivers(target_variable)
        key_drivers = [d[0] for d in key_drivers_raw]

        # 5. Critical Paths 추출
        all_paths = self.network_builder.get_critical_paths_to_target(
            target_variable,
            max_paths=max_paths
        )

        # 가장 중요한 경로
        critical_path = all_paths[0] if all_paths else None

        return NetworkAnalysisResult(
            target_variable=target_variable,
            all_paths=all_paths,
            critical_path=critical_path,
            key_drivers=key_drivers,
            network_stats=network_stats,
            granger_results=granger_results
        )

    def get_network(self) -> 'nx.DiGraph':
        """현재 네트워크 그래프 반환"""
        return self.network_builder.graph

    def get_visualization_data(self) -> Dict[str, Any]:
        """시각화용 데이터 반환"""
        return self.network_builder.get_visualization_data()


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Causal Network Analysis Module Demo")
    print("=" * 60)

    # 샘플 데이터 생성
    np.random.seed(42)
    n = 500

    # X1 → X2 → Y 관계 시뮬레이션
    X1 = np.random.randn(n).cumsum()
    X2 = np.zeros(n)
    Y = np.zeros(n)

    for i in range(3, n):
        X2[i] = 0.5 * X1[i-2] + 0.3 * X2[i-1] + np.random.randn() * 0.5
        Y[i] = 0.4 * X2[i-1] + 0.2 * Y[i-1] + np.random.randn() * 0.5

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y,
        'Noise': np.random.randn(n).cumsum()
    })

    print(f"\n[Sample Data]")
    print(f"  Shape: {data.shape}")
    print(f"  Simulated relationship: X1 → X2 → Y")

    # 분석 실행
    analyzer = CausalNetworkAnalyzer(max_lag=5, significance_level=0.05)
    result = analyzer.analyze(
        data=data,
        target_variable='Y',
        make_stationary=True
    )

    print(f"\n[Granger Causality Results]")
    for gr in result.granger_results[:5]:
        print(f"  {gr.cause} → {gr.effect}: lag={gr.optimal_lag}, p={gr.p_value:.4f}")

    print(f"\n[Key Drivers for Y]")
    for driver in result.key_drivers:
        print(f"  - {driver}")

    print(f"\n[Critical Paths to Y]")
    for path in result.all_paths[:3]:
        print(f"  {path.description}")
        print(f"    Total lag: {path.total_lag}, Strength: {path.path_strength:.4f}")

    if result.critical_path:
        print(f"\n[Most Critical Path]")
        print(f"  {result.critical_path.description}")

    print(f"\n[Network Stats]")
    for key, value in result.network_stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
