#!/usr/bin/env python3
"""
EIMAS 전체 기능 종합 테스트
==========================
2026-01-24~25 구현된 모든 기능 실행 및 검증

포함 기능:
1. HFT Microstructure (Tick Rule, Kyle's Lambda, Volume Clock, Quote Stuffing)
2. GARCH Volatility Modeling
3. Information Flow Analysis
4. Proof-of-Index
5. Systemic Similarity
6. DBSCAN Outlier Detection
7. DTW Time Series Similarity
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EIMAS 전체 기능 종합 테스트")
print("=" * 80)
print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# 테스트 데이터 생성
# ============================================================================

def create_comprehensive_test_data(n_assets=30, n_days=500):
    """종합 테스트용 시장 데이터 생성"""
    print("\n[0] Creating comprehensive test market data...")

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    market_data = {}

    # 1. 공통 팩터 (시장 베타)
    market_factor = np.cumsum(np.random.randn(n_days)) * 0.01

    # 2. 정상 자산 (20개) - 상관관계 높음
    for i in range(20):
        beta = 0.8 + np.random.rand() * 0.4  # 0.8-1.2
        returns = market_factor * beta + np.random.randn(n_days) * 0.005
        prices = 100 * np.exp(np.cumsum(returns))

        volumes = np.exp(np.random.randn(n_days) * 0.3 + 15)

        df = pd.DataFrame({
            'Open': prices * 0.995,
            'High': prices * 1.005,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': volumes,
            'Adj Close': prices
        }, index=dates)

        market_data[f'NORMAL_{i:02d}'] = df

    # 3. 이상치 자산 (5개) - 독립적인 움직임
    for i in range(5):
        returns = np.random.randn(n_days) * 0.03
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.exp(np.random.randn(n_days) * 0.5 + 14)

        df = pd.DataFrame({
            'Open': prices * 0.995,
            'High': prices * 1.005,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': volumes,
            'Adj Close': prices
        }, index=dates)

        market_data[f'OUTLIER_{i:02d}'] = df

    # 4. SPY (시장 대표)
    spy_prices = 400 * np.exp(market_factor + np.random.randn(n_days) * 0.003)
    spy_volumes = np.exp(np.random.randn(n_days) * 0.2 + 18)

    market_data['SPY'] = pd.DataFrame({
        'Open': spy_prices * 0.995,
        'High': spy_prices * 1.005,
        'Low': spy_prices * 0.99,
        'Close': spy_prices,
        'Volume': spy_volumes,
        'Adj Close': spy_prices
    }, index=dates)

    # 5. QQQ (SPY보다 3일 늦게 움직임 - Lead-Lag 테스트용)
    lag_days = 3
    lagged_factor = np.concatenate([np.zeros(lag_days), market_factor[:-lag_days]])
    qqq_prices = 350 * np.exp(lagged_factor * 1.2 + np.random.randn(n_days) * 0.004)
    qqq_volumes = np.exp(np.random.randn(n_days) * 0.2 + 17.5)

    market_data['QQQ'] = pd.DataFrame({
        'Open': qqq_prices * 0.995,
        'High': qqq_prices * 1.005,
        'Low': qqq_prices * 0.99,
        'Close': qqq_prices,
        'Volume': qqq_volumes,
        'Adj Close': qqq_prices
    }, index=dates)

    # 6. TLT (채권, 역상관)
    tlt_prices = 100 * np.exp(-market_factor * 0.3 + np.random.randn(n_days) * 0.002)
    tlt_volumes = np.exp(np.random.randn(n_days) * 0.3 + 15.5)

    market_data['TLT'] = pd.DataFrame({
        'Open': tlt_prices * 0.995,
        'High': tlt_prices * 1.005,
        'Low': tlt_prices * 0.99,
        'Close': tlt_prices,
        'Volume': tlt_volumes,
        'Adj Close': tlt_prices
    }, index=dates)

    # 7. GLD (금, 방어 자산)
    gld_prices = 180 * np.exp(np.cumsum(np.random.randn(n_days) * 0.008))
    gld_volumes = np.exp(np.random.randn(n_days) * 0.3 + 16)

    market_data['GLD'] = pd.DataFrame({
        'Open': gld_prices * 0.995,
        'High': gld_prices * 1.005,
        'Low': gld_prices * 0.99,
        'Close': gld_prices,
        'Volume': gld_volumes,
        'Adj Close': gld_prices
    }, index=dates)

    # 8. BTC-USD (크립토)
    btc_prices = 50000 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))
    btc_volumes = np.exp(np.random.randn(n_days) * 0.4 + 20)

    market_data['BTC-USD'] = pd.DataFrame({
        'Open': btc_prices * 0.99,
        'High': btc_prices * 1.02,
        'Low': btc_prices * 0.98,
        'Close': btc_prices,
        'Volume': btc_volumes,
        'Adj Close': btc_prices
    }, index=dates)

    print(f"   ✓ Created {len(market_data)} assets")
    print(f"   ✓ Normal assets: 20, Outliers: 5, Major indices: 4, Crypto: 1")
    print(f"   ✓ Data period: {n_days} days")

    return market_data


# ============================================================================
# Test 1: HFT Microstructure
# ============================================================================

def test_hft_microstructure(market_data):
    """HFT 미세구조 분석 테스트"""
    print("\n" + "=" * 80)
    print("[Test 1] HFT Microstructure Analysis")
    print("=" * 80)

    try:
        from lib.microstructure import (
            tick_rule_classification,
            kyles_lambda,
            volume_clock_sampling,
            detect_quote_stuffing
        )

        # SPY 데이터 사용
        spy_df = market_data['SPY'].copy()
        spy_prices = spy_df['Close']
        spy_volumes = spy_df['Volume']

        # 1.1 Tick Rule Classification
        print("\n[1.1] Tick Rule Classification")
        directions = tick_rule_classification(spy_prices)
        buy_ratio = (directions == 1).sum() / len(directions)
        sell_ratio = (directions == -1).sum() / len(directions)
        print(f"   Buy signals: {buy_ratio:.1%}")
        print(f"   Sell signals: {sell_ratio:.1%}")
        print(f"   ✅ Tick Rule completed")

        # 1.2 Kyle's Lambda (Market Impact)
        print("\n[1.2] Kyle's Lambda (Market Impact)")
        price_changes = spy_prices.diff()
        signed_volume = directions * spy_volumes

        lambda_result = kyles_lambda(price_changes, signed_volume, return_details=True)

        if isinstance(lambda_result, dict):
            print(f"   Lambda: {lambda_result.get('lambda', 'N/A')}")
            print(f"   Interpretation: {lambda_result.get('interpretation', 'N/A')}")
            print(f"   ✅ Kyle's Lambda completed")
        else:
            print(f"   ⚠️ Insufficient data for Kyle's Lambda")

        # 1.3 Volume Clock Sampling
        print("\n[1.3] Volume Clock Sampling")
        spy_df_reset = spy_df.reset_index()
        spy_df_reset.rename(columns={'index': 'timestamp'}, inplace=True)

        volume_col = 'Volume' if 'Volume' in spy_df_reset.columns else 'volume'
        volume_bucket = int(spy_volumes.mean())

        sampled = volume_clock_sampling(
            spy_df_reset,
            volume_bucket=volume_bucket,
            volume_col=volume_col
        )

        print(f"   Original samples: {len(spy_df)}")
        print(f"   Volume-based samples: {len(sampled)}")
        print(f"   Volume bucket size: {volume_bucket:,.0f}")
        print(f"   ✅ Volume Clock Sampling completed")

        # 1.4 Quote Stuffing Detection (시뮬레이션 데이터)
        print("\n[1.4] Quote Stuffing Detection")
        order_data = pd.DataFrame({
            'order_id': range(1000),
            'action': np.random.choice(['NEW', 'CANCEL'], 1000, p=[0.7, 0.3]),
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='ms')
        })

        stuffing_result = detect_quote_stuffing(order_data)
        print(f"   Cancel rate: {stuffing_result['cancel_rate']:.1%}")
        print(f"   Severity: {stuffing_result['severity']}")
        print(f"   ✅ Quote Stuffing Detection completed")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 2: GARCH Volatility Modeling
# ============================================================================

def test_garch_volatility(market_data):
    """GARCH 변동성 모델링 테스트"""
    print("\n" + "=" * 80)
    print("[Test 2] GARCH Volatility Modeling")
    print("=" * 80)

    try:
        from lib.regime_analyzer import GARCHModel

        # SPY 수익률
        spy_returns = market_data['SPY']['Close'].pct_change().dropna()

        print(f"\n   Data: {len(spy_returns)} observations")
        print(f"   Return volatility: {spy_returns.std():.4f}")

        # GARCH(1,1) 모델 피팅
        print("\n   Fitting GARCH(1,1)...")
        garch = GARCHModel(p=1, q=1)
        params = garch.fit(spy_returns)

        print(f"\n   GARCH Parameters:")
        print(f"   ω (omega): {params['omega']:.6f}")
        print(f"   α (alpha): {params['alpha']:.6f}")
        print(f"   β (beta):  {params['beta']:.6f}")
        print(f"   Persistence (α+β): {params['persistence']:.6f}")
        print(f"   Half-life: {params['half_life']:.1f} days")

        # 변동성 예측
        forecast = garch.forecast(horizon=10)
        print(f"\n   Volatility Forecast (10 days):")
        for i, vol in enumerate(forecast.head(10).values, 1):
            print(f"   Day {i}: {vol:.4f} ({vol*100:.2f}%)")

        print(f"\n   ✅ GARCH modeling completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 3: Information Flow Analysis
# ============================================================================

def test_information_flow(market_data):
    """정보 플로우 분석 테스트"""
    print("\n" + "=" * 80)
    print("[Test 3] Information Flow Analysis")
    print("=" * 80)

    try:
        from lib.information_flow import InformationFlowAnalyzer

        analyzer = InformationFlowAnalyzer()

        # 3.1 Abnormal Volume Detection
        print("\n[3.1] Abnormal Volume Detection")
        spy_volume = market_data['SPY']['Volume']
        abnormal = analyzer.detect_abnormal_volume(spy_volume)

        print(f"   Total abnormal days: {abnormal.total_abnormal_days}")
        print(f"   Abnormal ratio: {abnormal.abnormal_ratio:.1%}")
        print(f"   Max ratio: {abnormal.max_ratio:.1f}x")
        print(f"   Interpretation: {abnormal.interpretation}")

        # 3.2 Private Information Score (시뮬레이션)
        print("\n[3.2] Private Information Score")
        buy_volume = market_data['SPY']['Volume'] * np.random.rand(len(market_data['SPY']))
        sell_volume = market_data['SPY']['Volume'] - buy_volume

        info_score = analyzer.calculate_private_info_score(buy_volume, sell_volume)
        print(f"   Mean score: {info_score.mean_score:+.3f}")
        print(f"   Buy pressure days: {info_score.buy_pressure_days}")
        print(f"   Sell pressure days: {info_score.sell_pressure_days}")
        print(f"   Interpretation: {info_score.interpretation}")

        # 3.3 CAPM Regression
        print("\n[3.3] CAPM Regression Analysis")
        spy_returns = market_data['SPY']['Close'].pct_change().dropna()
        qqq_returns = market_data['QQQ']['Close'].pct_change().dropna()

        capm = analyzer.estimate_capm(qqq_returns, spy_returns)
        print(f"   Alpha: {capm.alpha:.6f} (daily)")
        print(f"   Alpha Interpretation: {capm.alpha_interpretation}")
        print(f"   Beta: {capm.beta:.3f}")
        print(f"   Beta Interpretation: {capm.beta_interpretation}")
        print(f"   R²: {capm.r_squared:.3f}")

        print(f"\n   ✅ Information Flow Analysis completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 4: Proof-of-Index
# ============================================================================

def test_proof_of_index(market_data):
    """Proof-of-Index 테스트"""
    print("\n" + "=" * 80)
    print("[Test 4] Proof-of-Index (Blockchain-based Transparent Index)")
    print("=" * 80)

    try:
        from lib.proof_of_index import ProofOfIndex

        poi = ProofOfIndex()

        # 인덱스 구성 (SPY, QQQ, TLT, GLD)
        tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
        quantities = {
            'SPY': 100,
            'QQQ': 150,
            'TLT': 200,
            'GLD': 80
        }

        # 최신 가격
        prices = {ticker: market_data[ticker]['Close'].iloc[-1] for ticker in tickers}

        # 4.1 Index Calculation
        print("\n[4.1] Index Calculation")
        snapshot = poi.calculate_index(prices, quantities)

        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Index Value: ${snapshot.index_value:,.2f}")
        print(f"\n   Component Weights:")
        for ticker, weight in sorted(snapshot.weights.items(), key=lambda x: -x[1]):
            print(f"   {ticker}: {weight:.1%}")

        # 4.2 Hash Verification
        print("\n[4.2] SHA-256 Hash Verification")
        print(f"   Hash: {snapshot.hash_value[:32]}...")

        reference_hash = poi.hash_index_weights(snapshot.weights, snapshot.timestamp)
        verification = poi.verify_on_chain(snapshot.hash_value, reference_hash)

        print(f"   Verification: {'✅ PASS' if verification['is_valid'] else '❌ FAIL'}")

        # 4.3 Mean Reversion Signal
        print("\n[4.3] Mean Reversion Strategy Signal")
        spy_prices = market_data['SPY']['Close']
        signal = poi.mean_reversion_signal(spy_prices, window=20, threshold=2.0)

        print(f"   Current Price: ${signal.current_price:.2f}")
        print(f"   Mean: ${signal.mean:.2f}")
        print(f"   Std Dev: ${signal.std:.2f}")
        print(f"   Z-Score: {signal.z_score:.2f}")
        print(f"   Signal: {signal.signal}")
        print(f"   Strength: {signal.strength:.2f}")
        print(f"   Interpretation: {signal.interpretation}")

        print(f"\n   ✅ Proof-of-Index completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 5: Systemic Similarity
# ============================================================================

def test_systemic_similarity(market_data):
    """Systemic Similarity 테스트"""
    print("\n" + "=" * 80)
    print("[Test 5] Systemic Similarity Analysis")
    print("=" * 80)

    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork

        # Returns 생성
        returns_df = pd.DataFrame()
        for ticker in ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']:
            returns_df[ticker] = market_data[ticker]['Close'].pct_change()
        returns_df = returns_df.dropna()

        # 네트워크 구축
        print("\n   Building correlation network...")
        network = CorrelationNetwork()
        network.build_from_returns(returns_df)

        # Systemic Similarity 계산
        print("   Computing systemic similarity (D̄ matrix)...")
        d_bar = network.compute_systemic_similarity()

        print(f"\n   Systemic Similarity Matrix:")
        print(d_bar.to_string())

        # 가장 유사한 쌍
        d_bar_values = d_bar.values.copy()
        np.fill_diagonal(d_bar_values, np.inf)
        min_idx = np.unravel_index(d_bar_values.argmin(), d_bar_values.shape)
        most_similar = (d_bar.index[min_idx[0]], d_bar.columns[min_idx[1]])
        min_val = d_bar_values[min_idx]

        # 가장 상이한 쌍
        max_idx = np.unravel_index(d_bar_values.argmax(), d_bar_values.shape)
        most_different = (d_bar.index[max_idx[0]], d_bar.columns[max_idx[1]])
        max_val = d_bar_values[max_idx]

        print(f"\n   Most Similar Pair: {most_similar[0]} ↔ {most_similar[1]} (D̄={min_val:.3f})")
        print(f"   Most Different Pair: {most_different[0]} ↔ {most_different[1]} (D̄={max_val:.3f})")

        print(f"\n   ✅ Systemic Similarity completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 6: DBSCAN Outlier Detection
# ============================================================================

def test_dbscan_outliers(market_data):
    """DBSCAN 이상치 탐지 테스트"""
    print("\n" + "=" * 80)
    print("[Test 6] DBSCAN Outlier Detection")
    print("=" * 80)

    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork

        # Returns 생성 (전체 자산)
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        print(f"\n   Total assets: {len(returns_df.columns)}")

        # 네트워크 구축
        network = CorrelationNetwork(correlation_threshold=0.2)
        network.build_from_returns(returns_df)

        # DBSCAN 실행
        print("   Running DBSCAN (eps=0.6, min_samples=3)...")
        result = network.detect_outliers_dbscan(eps=0.6, min_samples=3)

        print(f"\n   Results:")
        print(f"   Total Assets: {result.n_total_assets}")
        print(f"   Outliers: {result.n_outliers} ({result.outlier_ratio:.1%})")
        print(f"   Normal Assets: {len(result.normal_tickers)}")
        print(f"   Clusters: {result.n_clusters}")
        print(f"   Interpretation: {result.interpretation}")

        if result.n_outliers > 0:
            print(f"\n   Detected Outliers (first 10):")
            for ticker in result.outlier_tickers[:10]:
                print(f"   - {ticker}")

        # 검증: OUTLIER_XX가 탐지되었는지 확인
        true_outliers = [t for t in result.outlier_tickers if t.startswith('OUTLIER_')]
        print(f"\n   Validation:")
        print(f"   True outliers detected: {len(true_outliers)}/5")

        print(f"\n   ✅ DBSCAN Outlier Detection completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 7: DTW Time Series Similarity
# ============================================================================

def test_dtw_similarity(market_data):
    """DTW 시계열 유사도 테스트"""
    print("\n" + "=" * 80)
    print("[Test 7] DTW Time Series Similarity Analysis")
    print("=" * 80)

    try:
        from lib.time_series_similarity import (
            compute_dtw_similarity_matrix,
            find_lead_lag_relationship,
            detect_regime_shift_dtw
        )

        # Returns 생성
        returns_df = pd.DataFrame()
        for ticker in ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']:
            returns_df[ticker] = market_data[ticker]['Close'].pct_change()
        returns_df = returns_df.dropna()

        # 7.1 DTW Similarity Matrix
        print("\n[7.1] DTW Similarity Matrix")
        dtw_result = compute_dtw_similarity_matrix(returns_df, window=20, normalize=True)

        print(f"   Assets: {dtw_result.n_series}")
        print(f"   Avg DTW Distance: {dtw_result.avg_distance:.4f}")
        print(f"\n   Most Similar: {dtw_result.most_similar_pair[0]} ↔ {dtw_result.most_similar_pair[1]}")
        print(f"   DTW Distance: {dtw_result.most_similar_pair[2]:.4f}")
        print(f"\n   Most Dissimilar: {dtw_result.most_dissimilar_pair[0]} ↔ {dtw_result.most_dissimilar_pair[1]}")
        print(f"   DTW Distance: {dtw_result.most_dissimilar_pair[2]:.4f}")

        # 7.2 Lead-Lag Relationship (SPY vs QQQ)
        print("\n[7.2] Lead-Lag Relationship (SPY vs QQQ)")
        lead_lag = find_lead_lag_relationship(
            returns_df['SPY'],
            returns_df['QQQ'],
            max_lag=10,
            series1_name='SPY',
            series2_name='QQQ'
        )

        print(f"   Lead Asset: {lead_lag.lead_asset}")
        print(f"   Lag Asset: {lead_lag.lag_asset}")
        print(f"   Optimal Lag: {lead_lag.optimal_lag} days")
        print(f"   Min DTW Distance: {lead_lag.min_distance:.4f}")
        print(f"   Cross-Correlation: {lead_lag.cross_correlation:.3f}")
        print(f"   Interpretation: {lead_lag.interpretation}")

        # 검증
        if lead_lag.optimal_lag == 3:
            print(f"   ✅ Expected lag detected! (3 days)")
        else:
            print(f"   ⚠️ Expected 3 days, detected {lead_lag.optimal_lag} days")

        # 7.3 Regime Shift Detection
        print("\n[7.3] Regime Shift Detection (DTW-based)")

        # Bull/Bear 템플릿 생성 (초반/후반)
        mid_point = len(returns_df) // 2
        bull_template = returns_df['SPY'].iloc[:100]  # 초반 100일
        bear_template = returns_df['SPY'].iloc[mid_point:mid_point+100]  # 중간 100일
        current_window = returns_df['SPY'].tail(20)  # 최근 20일

        signal = detect_regime_shift_dtw(current_window, bull_template, bear_template, threshold=0.3)

        print(f"   Current Regime: {signal.current_regime}")
        print(f"   Similarity to Bull: {signal.similarity_to_bull:.1%}")
        print(f"   Similarity to Bear: {signal.similarity_to_bear:.1%}")
        print(f"   Shift Probability: {signal.shift_probability:.1%}")
        print(f"   Signal: {signal.signal}")
        print(f"   {signal.interpretation}")

        print(f"\n   ✅ DTW Time Series Similarity completed")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """전체 테스트 실행"""

    # 테스트 데이터 생성
    market_data = create_comprehensive_test_data(n_assets=30, n_days=500)

    # 테스트 실행
    results = {}

    results['HFT Microstructure'] = test_hft_microstructure(market_data)
    results['GARCH Volatility'] = test_garch_volatility(market_data)
    results['Information Flow'] = test_information_flow(market_data)
    results['Proof-of-Index'] = test_proof_of_index(market_data)
    results['Systemic Similarity'] = test_systemic_similarity(market_data)
    results['DBSCAN Outliers'] = test_dbscan_outliers(market_data)
    results['DTW Similarity'] = test_dtw_similarity(market_data)

    # 최종 결과
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(results.values())

    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name:<30} {'PASS' if status else 'FAIL'}")

    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 모든 기능이 정상적으로 작동합니다!")
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
