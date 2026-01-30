#!/usr/bin/env python3
"""
DBSCAN & DTW 통합 테스트
========================
Phase 2.19, 2.20 기능 검증
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import new analyzer functions
from pipeline.analyzers import (
    detect_outliers_with_dbscan,
    analyze_dtw_similarity
)

def create_test_market_data(n_assets=20, n_days=252):
    """테스트용 시장 데이터 생성"""
    print("\n[Setup] Creating test market data...")

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    market_data = {}

    # 정상 자산 (상관관계 높음)
    factor = np.cumsum(np.random.randn(n_days)) * 0.01
    for i in range(15):
        prices = 100 * np.exp(factor + np.random.randn(n_days) * 0.005)
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1e6, 1e7, n_days)
        }, index=dates)
        market_data[f'NORMAL_{i:02d}'] = df

    # 이상치 자산 (독립적인 랜덤 워크)
    for i in range(5):
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_days)) * 0.02)
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(1e6, 1e7, n_days)
        }, index=dates)
        market_data[f'OUTLIER_{i:02d}'] = df

    # SPY, QQQ 추가 (리드-래그 테스트용)
    spy_prices = 100 * np.exp(factor + np.random.randn(n_days) * 0.003)
    qqq_prices = 100 * np.exp(np.concatenate([np.zeros(3), factor[:-3]]) + np.random.randn(n_days) * 0.004)

    market_data['SPY'] = pd.DataFrame({
        'Close': spy_prices,
        'Open': spy_prices * 0.99,
        'High': spy_prices * 1.01,
        'Low': spy_prices * 0.98,
        'Volume': np.random.randint(1e8, 1e9, n_days)
    }, index=dates)

    market_data['QQQ'] = pd.DataFrame({
        'Close': qqq_prices,
        'Open': qqq_prices * 0.99,
        'High': qqq_prices * 1.01,
        'Low': qqq_prices * 0.98,
        'Volume': np.random.randint(1e8, 1e9, n_days)
    }, index=dates)

    print(f"   ✓ Created {len(market_data)} assets ({n_days} days)")
    print(f"   ✓ Normal assets: 15, Outliers: 5, SPY/QQQ: 2")

    return market_data


def main():
    print("=" * 70)
    print("EIMAS Pipeline - New Analyzers Integration Test")
    print("=" * 70)

    # 테스트 데이터 생성
    market_data = create_test_market_data(n_assets=22, n_days=252)

    # Test 1: DBSCAN Outlier Detection
    print("\n" + "=" * 70)
    print("Test 1: DBSCAN Outlier Detection (Phase 2.19)")
    print("=" * 70)

    dbscan_result = detect_outliers_with_dbscan(market_data)

    if dbscan_result:
        print("\n[Results Summary]")
        print(f"  Total Assets: {dbscan_result.get('n_total_assets', 'N/A')}")
        print(f"  Outliers Detected: {dbscan_result.get('n_outliers', 'N/A')}")
        print(f"  Outlier Ratio: {dbscan_result.get('outlier_ratio', 0):.1%}")
        print(f"  Number of Clusters: {dbscan_result.get('n_clusters', 'N/A')}")
        print(f"  Interpretation: {dbscan_result.get('interpretation', 'N/A')}")

        # 정확도 검증
        outliers = dbscan_result.get('outlier_tickers', [])
        true_outliers = [f'OUTLIER_{i:02d}' for i in range(5)]
        detected_true_outliers = [o for o in outliers if o in true_outliers]

        print(f"\n[Validation]")
        print(f"  True outliers in data: 5")
        print(f"  Detected true outliers: {len(detected_true_outliers)}")
        print(f"  Detection rate: {len(detected_true_outliers)/5:.1%}")

        if len(detected_true_outliers) >= 3:
            print(f"  ✅ DBSCAN successfully detected outliers!")
        else:
            print(f"  ⚠️ DBSCAN needs parameter tuning")

    # Test 2: DTW Similarity Analysis
    print("\n" + "=" * 70)
    print("Test 2: DTW Similarity Analysis (Phase 2.20)")
    print("=" * 70)

    dtw_result = analyze_dtw_similarity(market_data)

    if dtw_result:
        print("\n[Results Summary]")
        print(f"  Assets Analyzed: {dtw_result.get('n_series', 'N/A')}")
        print(f"  Average DTW Distance: {dtw_result.get('avg_distance', 0):.4f}")

        most_sim = dtw_result.get('most_similar_pair', {})
        print(f"\n  Most Similar Pair:")
        print(f"    {most_sim.get('asset1', 'N/A')} ↔ {most_sim.get('asset2', 'N/A')}")
        print(f"    DTW Distance: {most_sim.get('distance', 0):.4f}")

        most_dissim = dtw_result.get('most_dissimilar_pair', {})
        print(f"\n  Most Dissimilar Pair:")
        print(f"    {most_dissim.get('asset1', 'N/A')} ↔ {most_dissim.get('asset2', 'N/A')}")
        print(f"    DTW Distance: {most_dissim.get('distance', 0):.4f}")

        # 리드-래그 결과
        lead_lag = dtw_result.get('lead_lag_spy_qqq', {})
        if lead_lag:
            print(f"\n  Lead-Lag Analysis (SPY vs QQQ):")
            print(f"    Lead Asset: {lead_lag.get('lead_asset', 'N/A')}")
            print(f"    Lag Asset: {lead_lag.get('lag_asset', 'N/A')}")
            print(f"    Optimal Lag: {lead_lag.get('optimal_lag', 0)} days")
            print(f"    Cross-Correlation: {lead_lag.get('cross_correlation', 0):.3f}")
            print(f"    {lead_lag.get('interpretation', 'N/A')}")

            if lead_lag.get('optimal_lag', 0) == 3:
                print(f"\n  ✅ Lead-Lag detection successful! (Expected 3-day lag)")
            else:
                print(f"\n  ⚠️ Lead-Lag detection: Expected 3 days, got {lead_lag.get('optimal_lag', 0)}")

    print("\n" + "=" * 70)
    print("Integration Test Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
