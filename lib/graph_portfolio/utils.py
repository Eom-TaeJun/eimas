#!/usr/bin/env python3
"""
Graph Portfolio - Utilities
============================================================

Test 및 샘플 데이터 생성
"""

from typing import Tuple
import pandas as pd
import numpy as np


def create_sample_data(n_assets: int = 100, n_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)

    # 팩터 기반 수익률 생성
    n_factors = 5
    factor_returns = np.random.randn(n_days, n_factors) * 0.01

    # 자산별 팩터 로딩
    loadings = np.random.randn(n_assets, n_factors)

    # 자산 수익률 = 팩터 수익률 × 로딩 + 고유 수익률
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.02
    returns = np.dot(factor_returns, loadings.T) + idiosyncratic

    # 거래량 (로그정규분포)
    volumes = np.exp(np.random.randn(n_days, n_assets) + 10)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    assets = [f'ASSET_{i:03d}' for i in range(n_assets)]

    returns_df = pd.DataFrame(returns, index=dates, columns=assets)
    volumes_df = pd.DataFrame(volumes, index=dates, columns=assets)

    return returns_df, volumes_df


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Graph-Clustered HRP Test")
    print("=" * 60)

    # 샘플 데이터 생성
    print("\n1. Generating sample data (100 assets, 252 days)...")
    returns, volumes = create_sample_data(n_assets=100, n_days=252)
    print(f"   Returns shape: {returns.shape}")
    print(f"   Volumes shape: {volumes.shape}")

    # GC-HRP 실행
    print("\n2. Running Graph-Clustered HRP...")
    gc_hrp = GraphClusteredPortfolio(
        correlation_threshold=0.3,
        clustering_method=ClusteringMethod.KMEANS if SKLEARN_AVAILABLE else ClusteringMethod.HIERARCHICAL,
        representative_method=RepresentativeMethod.CENTRALITY,
        max_representatives_per_cluster=2
    )

