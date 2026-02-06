#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Quant Module

Purpose: Quantitative analysis functions (Phase 2.14-2.21)
Functions: analyze_hft_microstructure, analyze_volatility_garch, analyze_information_flow, calculate_proof_of_index, enhance_portfolio_with_systemic_similarity, detect_outliers_with_dbscan, analyze_dtw_similarity, analyze_ark_trades
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# NEW: Enhanced Modules (2026-01-24 보완 작업)
from lib.microstructure import (
    tick_rule_classification,
    kyles_lambda,
    volume_clock_sampling,
    detect_quote_stuffing,
    DailyMicrostructureAnalyzer
)
from lib.regime_analyzer import GARCHModel
from lib.information_flow import InformationFlowAnalyzer
from lib.proof_of_index import ProofOfIndex
from lib.ark_holdings_analyzer import ARKHoldingsAnalyzer, ARKHoldingsCollector

# NEW: Advanced Clustering & Time Series (2026-01-25)
from lib.time_series_similarity import (
    compute_dtw_similarity_matrix,
    find_lead_lag_relationship,
    detect_regime_shift_dtw
)

# Schemas
from pipeline.schemas import (
    RegimeResult, Event, LiquiditySignal,
    CriticalPathResult, ETFFlowResult, FREDSummary,
    GeniusActResult, ThemeETFResult, ShockAnalysisResult, PortfolioResult
)
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers.quant")

def analyze_hft_microstructure(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    HFT 미세구조 분석 (Enhanced)

    기능:
    - Tick Rule: 거래 방향 분류
    - Kyle's Lambda: Market Impact 측정
    - Volume Clock: VPIN 정확도 향상
    - Quote Stuffing: 시장 교란 탐지

    References:
    - Lee & Ready (1991)
    - Kyle (1985)
    - Easley et al. (2012)
    """
    print("\n[2.14] HFT Microstructure Analysis (Enhanced)...")
    try:
        results = {}

        # SPY 데이터로 테스트
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            # Tick Rule Classification
            if 'Close' in spy_data.columns:
                prices = spy_data['Close']
                directions = tick_rule_classification(prices)
                buy_ratio = (directions == 1).sum() / len(directions)

                results['tick_rule'] = {
                    'buy_ratio': buy_ratio,
                    'sell_ratio': 1 - buy_ratio,
                    'interpretation': 'BUY_PRESSURE' if buy_ratio > 0.55 else 'SELL_PRESSURE' if buy_ratio < 0.45 else 'NEUTRAL'
                }
                print(f"      ✓ Tick Rule: Buy Ratio {buy_ratio:.1%}")

            # Kyle's Lambda (Market Impact)
            if 'Close' in spy_data.columns and 'Volume' in spy_data.columns:
                try:
                    # 1. Series 추출 및 기본 정제
                    close_s = spy_data['Close'].squeeze()
                    vol_s = spy_data['Volume'].squeeze()

                    if isinstance(close_s, pd.DataFrame): close_s = close_s.iloc[:, 0]
                    if isinstance(vol_s, pd.DataFrame): vol_s = vol_s.iloc[:, 0]

                    # 2. 계산
                    returns = close_s.pct_change().rename("returns")
                    directions = tick_rule_classification(close_s).rename("directions")

                    # 3. 데이터프레임으로 통합 (자동 인덱스 정렬)
                    df_micro = pd.concat([returns, directions, vol_s], axis=1)
                    df_micro.columns = ['returns', 'directions', 'volume'] # 컬럼명 강제 지정

                    # 4. 결측치 제거
                    df_micro = df_micro.dropna()

                    # 5. Signed Volume 계산 및 Lambda 추정
                    if not df_micro.empty:
                        signed_vol = df_micro['directions'] * df_micro['volume']
                        lambda_result = kyles_lambda(df_micro['returns'], signed_vol)
                        results['kyles_lambda'] = lambda_result
                        print(f"      ✓ Kyle's Lambda: {lambda_result['lambda']:.6f} ({lambda_result['interpretation']})")
                    else:
                        print("      ⚠️ Kyle's Lambda skipped: No valid data")

                except Exception as ex:
                    print(f"      ⚠️ Kyle's Lambda skipped: {ex}")

            # Volume Clock Sampling (VPIN 향상용)
            if 'Volume' in spy_data.columns:
                volume_bucket = spy_data['Volume'].sum() / 20  # 20 buckets
                # reset_index()로 인해 컬럼명 확인 필요
                spy_df = spy_data.reset_index()
                # 컬럼명 통일 (volume_col 매개변수 사용)
                volume_col = 'Volume' if 'Volume' in spy_df.columns else 'volume'
                sampled = volume_clock_sampling(spy_df, volume_bucket, volume_col=volume_col)
                results['volume_clock'] = {
                    'original_samples': len(spy_data),
                    'volume_samples': len(sampled),
                    'compression_ratio': len(sampled) / len(spy_data)
                }
                print(f"      ✓ Volume Clock: {len(spy_data)} → {len(sampled)} samples")

        return results
    except Exception as e:
        log_error(logger, "HFT Microstructure analysis failed", e)
        return {}


def analyze_volatility_garch(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    GARCH 변동성 모델링

    기능:
    - GARCH(1,1) 모델 피팅
    - 조건부 변동성 추정
    - 다중 기간 변동성 예측

    References:
    - Engle (1982)
    - Bollerslev (1986)
    """
    print("\n[2.15] GARCH Volatility Modeling...")
    try:
        results = {}

        # SPY 수익률로 GARCH 모델링
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            if 'Close' in spy_data.columns:
                returns = spy_data['Close'].pct_change().dropna()

                # GARCH(1,1) 모델
                garch = GARCHModel(p=1, q=1)
                params = garch.fit(returns)

                # 10일 변동성 예측
                vol_forecast = garch.forecast(horizon=10)

                results['garch_params'] = params
                results['volatility_forecast_10d'] = vol_forecast.to_dict()

                # 스칼라로 변환 (Series 문제 방지)
                curr_vol = returns.std() * np.sqrt(252)
                curr_vol_scalar = float(curr_vol.item() if hasattr(curr_vol, 'item') else curr_vol)

                forecast_vol = vol_forecast.mean() * np.sqrt(252)
                forecast_vol_scalar = float(forecast_vol.item() if hasattr(forecast_vol, 'item') else forecast_vol)

                results['current_volatility'] = curr_vol_scalar
                results['forecast_avg_volatility'] = forecast_vol_scalar

                print(f"      ✓ GARCH(1,1) Persistence: {params['persistence']:.3f}")
                print(f"      ✓ Half-life: {params['half_life']:.1f} days")
                print(f"      ✓ Current Vol: {curr_vol_scalar:.1%}")
                print(f"      ✓ Forecast Vol (10d avg): {forecast_vol_scalar:.1%}")

        return results
    except Exception as e:
        log_error(logger, "GARCH volatility modeling failed", e)
        return {}


def analyze_information_flow(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    정보 플로우 분석

    기능:
    - 거래량 이상 탐지
    - Private Information Score
    - CAPM Alpha/Beta 추정

    References:
    - 금융경제정리.docx
    """
    print("\n[2.16] Information Flow Analysis...")
    try:
        analyzer = InformationFlowAnalyzer()
        results = {}

        # SPY 데이터 분석
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            # 1. 거래량 이상 탐지
            if 'Volume' in spy_data.columns:
                abnormal_result = analyzer.detect_abnormal_volume(spy_data['Volume'])
                results['abnormal_volume'] = abnormal_result.to_dict()
                print(f"      ✓ Abnormal Volume: {abnormal_result.total_abnormal_days} days ({abnormal_result.abnormal_ratio:.1%})")

            # 2. CAPM Alpha/Beta (vs SPY)
            if 'Close' in spy_data.columns:
                spy_returns = spy_data['Close'].pct_change().dropna()

                # 다른 자산들과 비교
                for ticker in ['QQQ', 'TLT', 'GLD']:
                    if ticker in market_data and not market_data[ticker].empty:
                        asset_data = market_data[ticker]
                        if 'Close' in asset_data.columns:
                            asset_returns = asset_data['Close'].pct_change().dropna()

                            capm_result = analyzer.estimate_capm(asset_returns, spy_returns)
                            results[f'capm_{ticker}'] = capm_result.to_dict()

                            if ticker == 'QQQ':  # QQQ만 출력
                                print(f"      ✓ {ticker} CAPM: Alpha={capm_result.alpha*252:+.1%}/yr, Beta={capm_result.beta:.2f}")

        return results
    except Exception as e:
        log_error(logger, "Information flow analysis failed", e)
        return {}


def calculate_proof_of_index(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Proof-of-Index (블록체인 기반 투명 지수)

    기능:
    - 시가총액 가중 지수 계산
    - SHA-256 해시 검증
    - Mean Reversion 신호 생성

    References:
    - eco4.docx
    - Nakamoto (2008)
    """
    print("\n[2.17] Proof-of-Index Calculation...")
    try:
        poi = ProofOfIndex(divisor=100.0, name='EIMAS Portfolio Index')
        results = {}

        # 포트폴리오 구성 자산 선택
        portfolio_tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
        available_tickers = [t for t in portfolio_tickers if t in market_data and not market_data[t].empty]

        if available_tickers:
            # 최신 가격 수집 (스칼라로 변환)
            prices = {}
            for ticker in available_tickers:
                if 'Close' in market_data[ticker].columns:
                    price_val = market_data[ticker]['Close'].iloc[-1]
                    # Series/array를 스칼라로 변환
                    prices[ticker] = float(price_val.item() if hasattr(price_val, 'item') else price_val)

            # 동일 가중 (실제로는 시가총액 가중)
            quantities = {ticker: 1.0 for ticker in prices.keys()}

            # 인덱스 계산
            snapshot = poi.calculate_index(prices, quantities)

            results['index_value'] = snapshot.index_value
            results['weights'] = snapshot.weights
            results['hash'] = snapshot.hash_value
            results['timestamp'] = snapshot.timestamp.isoformat()

            # SHA-256 검증
            reference_hash = poi.hash_index_weights(snapshot.weights, snapshot.timestamp)
            verification = poi.verify_on_chain(snapshot.hash_value, reference_hash)
            results['verification'] = verification

            print(f"      ✓ Index Value: {snapshot.index_value:.2f}")
            print(f"      ✓ Components: {', '.join([f'{t}:{w:.0%}' for t, w in sorted(snapshot.weights.items(), key=lambda x: -x[1])[:3]])}")
            print(f"      ✓ Hash Verification: {'✅ PASS' if verification['is_valid'] else '❌ FAIL'}")

            # Mean Reversion Signal (SPY 기준)
            if 'SPY' in market_data and 'Close' in market_data['SPY'].columns:
                spy_prices = market_data['SPY']['Close']
                signal = poi.mean_reversion_signal(spy_prices, window=20, threshold=2.0)
                results['mean_reversion_signal'] = signal.to_dict()
                print(f"      ✓ Mean Reversion: {signal.signal} (Z={signal.z_score:.2f})")

        return results
    except Exception as e:
        log_error(logger, "Proof-of-Index calculation failed", e)
        return {}


def enhance_portfolio_with_systemic_similarity(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    포트폴리오에 Systemic Similarity 추가 분석

    기능:
    - compute_systemic_similarity() 통합
    - 자산 간 상호작용 강도 정량화

    References:
    - De Prado (2016)
    - eco1.docx
    """
    print("\n[2.18] Systemic Similarity Enhancement...")
    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork
        import numpy as np

        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 3:
            # 네트워크 구축
            network = CorrelationNetwork()
            network.build_from_returns(returns_df)

            # Systemic Similarity 계산
            d_bar = network.compute_systemic_similarity()

            # 가장 유사한 자산 쌍 찾기
            d_bar_values = d_bar.values.copy()
            np.fill_diagonal(d_bar_values, np.inf)  # 대각선 제외
            min_idx = np.unravel_index(d_bar_values.argmin(), d_bar_values.shape)
            most_similar_pair = (d_bar.index[min_idx[0]], d_bar.columns[min_idx[1]])
            min_similarity = d_bar_values[min_idx]

            # 가장 상이한 자산 쌍
            max_idx = np.unravel_index(d_bar_values.argmax(), d_bar_values.shape)
            most_different_pair = (d_bar.index[max_idx[0]], d_bar.columns[max_idx[1]])
            max_similarity = d_bar_values[max_idx]

            results['systemic_similarity_matrix'] = d_bar.to_dict()
            results['most_similar_pair'] = {
                'assets': most_similar_pair,
                'similarity': min_similarity
            }
            results['most_different_pair'] = {
                'assets': most_different_pair,
                'dissimilarity': max_similarity
            }

            print(f"      ✓ Most Similar: {most_similar_pair[0]} ↔ {most_similar_pair[1]} (D̄={min_similarity:.3f})")
            print(f"      ✓ Most Different: {most_different_pair[0]} ↔ {most_different_pair[1]} (D̄={max_similarity:.3f})")

        return results
    except Exception as e:
        log_error(logger, "Systemic similarity enhancement failed", e)
        return {}


def detect_outliers_with_dbscan(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    DBSCAN 기반 이상치 탐지 (Phase 2.19)

    경제학적 배경:
    - 밀도 기반 클러스터링으로 노이즈 자산 제거
    - HRP 포트폴리오 품질 향상
    - 비정상 자산 자동 식별

    기능:
    - CorrelationNetwork.detect_outliers_dbscan() 통합
    - 이상치 자산 리스트 반환
    - 클러스터링 품질 메트릭

    References:
    - Ester et al. (1996)
    - 금융경제정리.docx
    """
    print("\n[2.19] DBSCAN Outlier Detection...")
    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork

        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 5:  # 최소 5개 자산 필요
            # 네트워크 구축
            network = CorrelationNetwork(correlation_threshold=0.2)
            network.build_from_returns(returns_df)

            # DBSCAN 실행
            outlier_result = network.detect_outliers_dbscan(
                eps=0.6,
                min_samples=3
            )

            # 결과 저장
            results['n_total_assets'] = outlier_result.n_total_assets
            results['n_outliers'] = outlier_result.n_outliers
            results['outlier_ratio'] = outlier_result.outlier_ratio
            results['outlier_tickers'] = outlier_result.outlier_tickers
            results['normal_tickers'] = outlier_result.normal_tickers
            results['n_clusters'] = outlier_result.n_clusters
            results['interpretation'] = outlier_result.interpretation
            results['eps'] = outlier_result.eps
            results['min_samples'] = outlier_result.min_samples

            print(f"      ✓ Total Assets: {outlier_result.n_total_assets}")
            print(f"      ✓ Outliers: {outlier_result.n_outliers} ({outlier_result.outlier_ratio:.1%})")
            print(f"      ✓ Clusters: {outlier_result.n_clusters}")
            print(f"      ✓ {outlier_result.interpretation}")

            if outlier_result.n_outliers > 0:
                print(f"      ✓ Outlier Assets (first 5): {', '.join(outlier_result.outlier_tickers[:5])}")

        return results
    except Exception as e:
        log_error(logger, "DBSCAN outlier detection failed", e)
        return {}


def analyze_dtw_similarity(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    DTW (Dynamic Time Warping) 시계열 유사도 분석 (Phase 2.20)

    경제학적 배경:
    - 시차를 무시하고 패턴 유사도만 측정
    - 리드-래그 관계 파악 (선행/후행 자산)
    - 레짐 전환 조기 감지

    기능:
    - compute_dtw_similarity_matrix() - 유사도 행렬
    - find_lead_lag_relationship() - 리드-래그 탐지
    - 상관관계와 비교 분석

    References:
    - Berndt & Clifford (1994)
    - todolist.md
    """
    print("\n[2.20] DTW Time Series Similarity Analysis...")
    try:
        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 3:
            # DTW 유사도 행렬 계산
            dtw_result = compute_dtw_similarity_matrix(
                returns_df,
                window=20,
                normalize=True
            )

            results['n_series'] = dtw_result.n_series
            results['avg_distance'] = dtw_result.avg_distance
            results['most_similar_pair'] = {
                'asset1': dtw_result.most_similar_pair[0],
                'asset2': dtw_result.most_similar_pair[1],
                'distance': dtw_result.most_similar_pair[2]
            }
            results['most_dissimilar_pair'] = {
                'asset1': dtw_result.most_dissimilar_pair[0],
                'asset2': dtw_result.most_dissimilar_pair[1],
                'distance': dtw_result.most_dissimilar_pair[2]
            }

            print(f"      ✓ Assets Analyzed: {dtw_result.n_series}")
            print(f"      ✓ Avg DTW Distance: {dtw_result.avg_distance:.4f}")
            print(f"      ✓ Most Similar: {dtw_result.most_similar_pair[0]} ↔ "
                  f"{dtw_result.most_similar_pair[1]} (DTW={dtw_result.most_similar_pair[2]:.4f})")

            # 리드-래그 관계 탐지 (SPY vs QQQ 예시)
            if 'SPY' in returns_df.columns and 'QQQ' in returns_df.columns:
                lead_lag = find_lead_lag_relationship(
                    returns_df['SPY'],
                    returns_df['QQQ'],
                    max_lag=10,
                    series1_name='SPY',
                    series2_name='QQQ'
                )

                results['lead_lag_spy_qqq'] = {
                    'lead_asset': lead_lag.lead_asset,
                    'lag_asset': lead_lag.lag_asset,
                    'optimal_lag': lead_lag.optimal_lag,
                    'min_distance': lead_lag.min_distance,
                    'cross_correlation': lead_lag.cross_correlation,
                    'interpretation': lead_lag.interpretation
                }

                print(f"      ✓ Lead-Lag (SPY vs QQQ): {lead_lag.interpretation}")

        return results
    except Exception as e:
        log_error(logger, "DTW similarity analysis failed", e)
        return {}


def analyze_ark_trades() -> Dict[str, Any]:
    """
    ARK Invest (Cathie Wood) ETF 트레이딩 분석

    기능:
    - ARKK, ARKW 등 주요 ETF의 일일 보유량 변화 추적
    - 'Consensus Buy/Sell': 여러 ETF가 동시에 매수/매도한 종목 식별
    - 'New Position': 신규 편입 종목 식별

    경제학적 의미:
    - 스마트 머니(Smart Money)의 선행 지표 역할
    - 기술주/성장주 섹터의 센티먼트 파악
    """
    print("\n[2.21] ARK Invest Holdings Analysis...")
    try:
        # 데이터 수집 및 분석
        collector = ARKHoldingsCollector()
        analyzer = ARKHoldingsAnalyzer(collector)
        result = analyzer.run_analysis()

        # 신호 생성
        signals = analyzer.generate_signals(result)

        # 결과 요약
        summary = {
            'timestamp': result.timestamp,
            'consensus_buys': result.consensus_buys,
            'consensus_sells': result.consensus_sells,
            'new_positions': result.new_positions,
            'top_increases': [c.to_dict() for c in result.top_increases[:5]],
            'top_decreases': [c.to_dict() for c in result.top_decreases[:5]],
            'signals': [
                f"{s.direction.value} {s.ticker} ({s.description})"
                for s in signals
            ]
        }

        # 주요 발견 출력
        if result.consensus_buys:
            print(f"      ✓ ARK Consensus BUY: {', '.join(result.consensus_buys)}")
        if result.consensus_sells:
            print(f"      ✓ ARK Consensus SELL: {', '.join(result.consensus_sells)}")
        if result.new_positions:
            print(f"      ✓ New Positions: {', '.join(result.new_positions)}")
        if not (result.consensus_buys or result.consensus_sells or result.new_positions):
            print("      ✓ No major ARK trades detected today")

        return summary

    except Exception as e:
        log_error(logger, "ARK holdings analysis failed", e)
        return {}
