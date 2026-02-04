#!/usr/bin/env python3
"""
Critical Path - Crypto Sentiment Block
=======================================

암호화폐 심리 지표 블록 분석 모듈

Economic Foundation:
    IMF (2021): "Crypto-Asset Cross-Border Flows"
    State-Dependent Crypto-Equity Correlation Research

    핵심 개념:
    - 크립토는 State-Dependent (레짐별로 역할 변화)
    - Normal 시기: 리스크 자산으로 작동, 주식과 양의 상관관계
    - Crisis 시기: 주식과 높은 동조화 (safe haven 아님)
    - Granger Causality: BTC가 리스크 신호 선행할 수 있음

Classes:
    - CryptoSentimentBlock: 암호화폐 심리 분석 및 선행 신호 탐지

Returns:
    CryptoSentimentResult: 심리 점수, 상관관계 레짐, 선행 신호
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

# Import schemas from same package
from .schemas import CryptoSentimentResult, calculate_rolling_zscore, normalize_to_score

# Check if statsmodels is available
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class CryptoSentimentBlock:
    """
    암호화폐 심리 지표 블록
    
    IMF 연구 및 State-Dependent 연구에 기반하여 암호화폐를 별도 블록으로 분리하고,
    레짐에 따라 다르게 해석합니다.
    
    경제학적 배경:
    - BTC-주식 상관관계가 레짐에 따라 극적으로 다름
    - 평시: 낮은 상관관계 (0.1-0.3), 독자적 움직임
    - 위기시: 높은 상관관계 (0.6-0.8), 동반 하락
    - 암호화폐가 "유동성의 카나리아" 역할 가능성 (IMF)
    - BTC→주식 방향 spillover가 반대보다 강함 (특히 위기 시)
    
    핵심 특징:
    - 평시/위기시 해석이 다름 (state-dependent)
    - 선행성 테스트 내장
    - 레짐에 따라 전체 위험도 기여도 변동
    """
    
    # 상관관계 기반 레짐 정의
    CORRELATION_REGIMES = {
        'DECOUPLED': {'min': -0.2, 'max': 0.3},      # 독자적 움직임
        'COUPLED': {'min': 0.3, 'max': 0.6},         # 연동
        'CRISIS_COUPLED': {'min': 0.6, 'max': 1.0},  # 위기 동조화
    }
    
    # 레짐별 위험 기여도
    RISK_CONTRIBUTION = {
        'DECOUPLED': 0.05,       # 5% - 독자 신호로만 해석
        'COUPLED': 0.10,         # 10%
        'CRISIS_COUPLED': 0.20,  # 20% - 선행지표로 중요
    }
    
    def __init__(self, lookback: int = 20, correlation_window: int = 20):
        """
        Args:
            lookback: 롤링 윈도우 기간 (기본값 20일)
            correlation_window: 상관관계 계산 윈도우 (기본값 20일)
        """
        self.lookback = lookback
        self.correlation_window = correlation_window
    
    def calculate_btc_momentum(self, btc_data: pd.DataFrame) -> Dict:
        """
        BTC 모멘텀 계산
        
        지표:
        - 5일 수익률
        - 20일 수익률
        - 5일 MA vs 20일 MA 위치
        - 거래량 변화
        
        Returns:
            Dict with momentum indicators
        """
        if btc_data is None or (hasattr(btc_data, 'empty') and btc_data.empty) or (btc_data is not None and 'Close' not in btc_data.columns):
            return {
                'return_5d': 0.0,
                'return_20d': 0.0,
                'ma5_above_ma20': False,
                'volume_trend': 0.0,
            }
        
        close = btc_data['Close']
        
        # 수익률 계산
        if len(close) >= 5:
            return_5d = (float(close.iloc[-1]) / float(close.iloc[-5]) - 1) * 100
        else:
            return_5d = 0.0
        
        if len(close) >= 20:
            return_20d = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1) * 100
        else:
            return_20d = 0.0
        
        # 이동평균 계산
        ma_5 = close.rolling(window=5, min_periods=1).mean()
        ma_20 = close.rolling(window=20, min_periods=1).mean()
        
        ma5_above_ma20 = False
        if len(ma_5) > 0 and len(ma_20) > 0:
            current_ma5 = float(ma_5.iloc[-1])
            current_ma20 = float(ma_20.iloc[-1])
            ma5_above_ma20 = current_ma5 > current_ma20 if not pd.isna(current_ma5) and not pd.isna(current_ma20) else False
        
        # 거래량 추세
        volume_trend = 0.0
        if 'Volume' in btc_data.columns and len(btc_data) >= 20:
            volume = btc_data['Volume']
            if len(volume) >= 20:
                recent_volume = float(volume.tail(5).mean()) if len(volume) >= 5 else 0
                avg_volume = float(volume.tail(20).mean())
                if avg_volume > 0:
                    volume_trend = (recent_volume / avg_volume - 1) * 100
        
        return {
            'return_5d': return_5d,
            'return_20d': return_20d,
            'ma5_above_ma20': ma5_above_ma20,
            'volume_trend': volume_trend,
        }
    
    def calculate_btc_spy_correlation(
        self, 
        btc_data: pd.DataFrame,
        spy_data: pd.DataFrame
    ) -> float:
        """
        BTC-SPY 롤링 상관관계 계산
        
        Returns:
            float: 상관계수 (-1 to 1)
        """
        btc_empty = hasattr(btc_data, 'empty') and btc_data.empty if btc_data is not None else True
        spy_empty = hasattr(spy_data, 'empty') and spy_data.empty if spy_data is not None else True
        if btc_empty or spy_empty:
            return 0.0
        
        if 'Close' not in btc_data.columns or 'Close' not in spy_data.columns:
            return 0.0
        
        btc_close = btc_data['Close']
        spy_close = spy_data['Close']
        
        # 인덱스 정렬
        common_index = btc_close.index.intersection(spy_close.index)
        if len(common_index) < self.correlation_window:
            return 0.0
        
        # 최근 N일간 수익률 계산
        btc_returns = btc_close.loc[common_index].pct_change().dropna()
        spy_returns = spy_close.loc[common_index].pct_change().dropna()
        
        # 공통 인덱스로 정렬
        common_returns_index = btc_returns.index.intersection(spy_returns.index)
        if len(common_returns_index) < self.correlation_window:
            return 0.0
        
        btc_recent = btc_returns.loc[common_returns_index].tail(self.correlation_window)
        spy_recent = spy_returns.loc[common_returns_index].tail(self.correlation_window)
        
        if len(btc_recent) < self.correlation_window or len(spy_recent) < self.correlation_window:
            return 0.0
        
        # 상관관계 계산
        correlation = float(btc_recent.corr(spy_recent))
        
        return correlation if not pd.isna(correlation) else 0.0
    
    def calculate_granger_causality(
        self,
        series_x: pd.Series,  # 원인 후보 시계열 (예: BTC)
        series_y: pd.Series,  # 결과 후보 시계열 (예: SPY)
        max_lag: int = 5      # 최대 시차
    ) -> Dict:
        """
        Granger Causality 검정 수행
        
        경제학적 배경:
        - Granger(1969): "X의 과거값이 Y 예측에 도움이 되는가?"
        - 상관관계는 인과관계가 아님 (Correlation ≠ Causation)
        - Granger Causality는 통계적 인과관계를 검정
        
        Args:
            series_x: 원인 후보 시계열 (예: BTC 수익률)
            series_y: 결과 후보 시계열 (예: SPY 수익률)
            max_lag: 최대 시차 (기본값 5일)
        
        Returns:
            {
                'x_causes_y': bool,        # X가 Y를 Granger-cause하는지
                'y_causes_x': bool,        # Y가 X를 Granger-cause하는지
                'x_to_y_pvalue': float,    # X->Y 검정의 p-value
                'y_to_x_pvalue': float,    # Y->X 검정의 p-value
                'optimal_lag': int,        # 최적 시차
                'relationship': str        # "X_LEADS", "Y_LEADS", "BIDIRECTIONAL", "NO_CAUSALITY"
            }
        """
        if not STATSMODELS_AVAILABLE:
            return {
                'x_causes_y': False,
                'y_causes_x': False,
                'x_to_y_pvalue': 1.0,
                'y_to_x_pvalue': 1.0,
                'optimal_lag': 0,
                'relationship': 'NO_CAUSALITY'
            }
        
        # 데이터 정렬 및 정규화
        common_index = series_x.index.intersection(series_y.index)
        if len(common_index) < max_lag + 10:  # 최소 데이터 요구사항
            return {
                'x_causes_y': False,
                'y_causes_x': False,
                'x_to_y_pvalue': 1.0,
                'y_to_x_pvalue': 1.0,
                'optimal_lag': 0,
                'relationship': 'NO_CAUSALITY'
            }
        
        x_aligned = series_x.loc[common_index].dropna()
        y_aligned = series_y.loc[common_index].dropna()
        
        # 공통 인덱스로 정렬
        common_idx = x_aligned.index.intersection(y_aligned.index)
        if len(common_idx) < max_lag + 10:
            return {
                'x_causes_y': False,
                'y_causes_x': False,
                'x_to_y_pvalue': 1.0,
                'y_to_x_pvalue': 1.0,
                'optimal_lag': 0,
                'relationship': 'NO_CAUSALITY'
            }
        
        x_data = x_aligned.loc[common_idx].values
        y_data = y_aligned.loc[common_idx].values
        
        # 결측값 제거
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) < max_lag + 10 or len(y_data) < max_lag + 10:
            return {
                'x_causes_y': False,
                'y_causes_x': False,
                'x_to_y_pvalue': 1.0,
                'y_to_x_pvalue': 1.0,
                'optimal_lag': 0,
                'relationship': 'NO_CAUSALITY'
            }
        
        # DataFrame 생성 (grangercausalitytests는 2열 DataFrame 필요)
        data_xy = pd.DataFrame({'y': y_data, 'x': x_data})
        data_yx = pd.DataFrame({'x': x_data, 'y': y_data})
        
        # X->Y 검정
        x_to_y_pvalue = 1.0
        x_causes_y = False
        optimal_lag_xy = 1
        
        try:
            # 각 시차별로 검정하고 최소 p-value 찾기
            min_pvalue_xy = 1.0
            for lag in range(1, min(max_lag + 1, len(data_xy) // 10)):
                try:
                    gc_result = grangercausalitytests(data_xy, maxlag=lag, verbose=False)
                    # grangercausalitytests 반환값: {lag: (test_results_dict, test_statistics)}
                    # test_results_dict의 키: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
                    if lag in gc_result:
                        test_results = gc_result[lag][0]
                        # F-test p-value 추출
                        if 'ssr_ftest' in test_results:
                            pvalue = test_results['ssr_ftest'][1]  # (F-stat, p-value)
                            if pvalue < min_pvalue_xy:
                                min_pvalue_xy = pvalue
                                optimal_lag_xy = lag
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    # 특정 시차에서 실패해도 계속 진행
                    continue
            
            x_to_y_pvalue = min_pvalue_xy
            x_causes_y = x_to_y_pvalue < 0.05  # 5% 유의수준
        except Exception as e:
            # 검정 실패 시 기본값 유지
            pass
        
        # Y->X 검정
        y_to_x_pvalue = 1.0
        y_causes_x = False
        optimal_lag_yx = 1
        
        try:
            min_pvalue_yx = 1.0
            for lag in range(1, min(max_lag + 1, len(data_yx) // 10)):
                try:
                    gc_result = grangercausalitytests(data_yx, maxlag=lag, verbose=False)
                    if lag in gc_result:
                        test_results = gc_result[lag][0]
                        if 'ssr_ftest' in test_results:
                            pvalue = test_results['ssr_ftest'][1]
                            if pvalue < min_pvalue_yx:
                                min_pvalue_yx = pvalue
                                optimal_lag_yx = lag
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    continue
            
            y_to_x_pvalue = min_pvalue_yx
            y_causes_x = y_to_x_pvalue < 0.05
        except Exception as e:
            # 검정 실패 시 기본값 유지
            pass
        
        # 관계 유형 결정
        optimal_lag = max(optimal_lag_xy, optimal_lag_yx)
        
        if x_causes_y and y_causes_x:
            relationship = "BIDIRECTIONAL"
        elif x_causes_y:
            relationship = "X_LEADS"
        elif y_causes_x:
            relationship = "Y_LEADS"
        else:
            relationship = "NO_CAUSALITY"
        
        return {
            'x_causes_y': x_causes_y,
            'y_causes_x': y_causes_x,
            'x_to_y_pvalue': float(x_to_y_pvalue),
            'y_to_x_pvalue': float(y_to_x_pvalue),
            'optimal_lag': optimal_lag,
            'relationship': relationship
        }
    
    def determine_correlation_regime(self, correlation: float) -> str:
        """
        상관관계 기반 레짐 판단
        
        Returns:
            str: "DECOUPLED", "COUPLED", "CRISIS_COUPLED"
        """
        for regime, bounds in self.CORRELATION_REGIMES.items():
            if bounds['min'] <= correlation <= bounds['max']:
                return regime
        
        # 기본값
        if correlation < 0.3:
            return "DECOUPLED"
        elif correlation < 0.6:
            return "COUPLED"
        else:
            return "CRISIS_COUPLED"
    
    def calculate_sentiment_score(self, btc_data: pd.DataFrame) -> Tuple[float, str]:
        """
        심리 점수 계산 (Crypto Fear & Greed 유사 개념)
        
        구성요소:
        1. 모멘텀 (40%): 5일/20일 수익률
        2. 변동성 (20%): 최근 변동성 vs 평균
        3. 거래량 (20%): 거래량 추세
        4. MA 위치 (20%): 5MA vs 20MA
        
        Returns:
            Tuple[score (0-100), level]
        
        Level 정의:
        - 0-20: EXTREME_FEAR
        - 20-40: FEAR
        - 40-60: NEUTRAL
        - 60-80: GREED
        - 80-100: EXTREME_GREED
        """
        if btc_data is None or (hasattr(btc_data, 'empty') and btc_data.empty) or (btc_data is not None and 'Close' not in btc_data.columns):
            return 50.0, "NEUTRAL"
        
        close = btc_data['Close']
        
        # 1. 모멘텀 점수 (40%)
        momentum = self.calculate_btc_momentum(btc_data)
        return_5d = momentum['return_5d']
        return_20d = momentum['return_20d']
        
        # 5일 수익률 정규화 (-10% ~ +10% → 0-100)
        momentum_5d_score = normalize_to_score(return_5d, min_val=-10.0, max_val=10.0)
        # 20일 수익률 정규화 (-20% ~ +20% → 0-100)
        momentum_20d_score = normalize_to_score(return_20d, min_val=-20.0, max_val=20.0)
        momentum_score = (momentum_5d_score * 0.6 + momentum_20d_score * 0.4) * 0.4
        
        # 2. 변동성 점수 (20%) - 낮은 변동성 = 높은 점수 (안정적)
        if len(close) >= 20:
            returns = close.pct_change().dropna()
            recent_vol = returns.tail(5).std() * np.sqrt(252) * 100 if len(returns) >= 5 else 0
            avg_vol = returns.tail(20).std() * np.sqrt(252) * 100 if len(returns) >= 20 else recent_vol
            
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                # 변동성이 낮을수록 높은 점수 (역변환)
                volatility_score = (2.0 - vol_ratio) / 2.0 * 100 if vol_ratio <= 2.0 else 0
                volatility_score = max(0, min(100, volatility_score)) * 0.2
            else:
                volatility_score = 50.0 * 0.2
        else:
            volatility_score = 50.0 * 0.2
        
        # 3. 거래량 점수 (20%) - 거래량 증가 = 높은 점수 (관심 증가)
        volume_trend = momentum['volume_trend']
        # 거래량 추세 정규화 (-50% ~ +100% → 0-100)
        volume_score = normalize_to_score(volume_trend, min_val=-50.0, max_val=100.0) * 0.2
        
        # 4. MA 위치 점수 (20%) - 5MA > 20MA = 높은 점수
        ma_5 = close.rolling(window=5, min_periods=1).mean()
        ma_20 = close.rolling(window=20, min_periods=1).mean()
        
        ma_score = 50.0
        if len(ma_5) > 0 and len(ma_20) > 0:
            current_ma5 = float(ma_5.iloc[-1])
            current_ma20 = float(ma_20.iloc[-1])
            if not pd.isna(current_ma5) and not pd.isna(current_ma20) and current_ma20 > 0:
                ma_ratio = (current_ma5 / current_ma20 - 1) * 100
                # MA 비율 정규화 (-5% ~ +5% → 0-100)
                ma_score = normalize_to_score(ma_ratio, min_val=-5.0, max_val=5.0)
        
        ma_score = ma_score * 0.2
        
        # 종합 점수
        total_score = momentum_score + volatility_score + volume_score + ma_score
        total_score = max(0.0, min(100.0, total_score))
        
        # 레벨 결정
        if total_score < 20:
            level = "EXTREME_FEAR"
        elif total_score < 40:
            level = "FEAR"
        elif total_score < 60:
            level = "NEUTRAL"
        elif total_score < 80:
            level = "GREED"
        else:
            level = "EXTREME_GREED"
        
        return total_score, level
    
    def check_leading_indicator(
        self, 
        btc_data: pd.DataFrame,
        spy_data: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        """
        선행지표 역할 테스트
        
        조건:
        1. BTC가 5일간 -10% 이상 하락 AND
        2. SPY는 아직 -3% 미만 하락 AND
        3. 상관관계가 상승 추세
        → RISK_OFF_WARNING
        
        반대 조건:
        1. BTC가 5일간 +10% 이상 상승 AND
        2. SPY는 아직 +3% 미만 상승
        → RISK_ON_SIGNAL
        
        Returns:
            Tuple[is_leading, signal_type]
        """
        btc_empty = hasattr(btc_data, 'empty') and btc_data.empty if btc_data is not None else True
        spy_empty = hasattr(spy_data, 'empty') and spy_data.empty if spy_data is not None else True
        if btc_empty or spy_empty:
            return False, None
        
        if 'Close' not in btc_data.columns or 'Close' not in spy_data.columns:
            return False, None
        
        btc_close = btc_data['Close']
        spy_close = spy_data['Close']
        
        # 5일 수익률 계산
        if len(btc_close) < 5 or len(spy_close) < 5:
            return False, None
        
        btc_return_5d = (float(btc_close.iloc[-1]) / float(btc_close.iloc[-5]) - 1) * 100
        spy_return_5d = (float(spy_close.iloc[-1]) / float(spy_close.iloc[-5]) - 1) * 100
        
        # RISK_OFF_WARNING 체크
        if btc_return_5d <= -10.0 and spy_return_5d > -3.0:
            # 상관관계 상승 추세 확인 (선택적)
            correlation = self.calculate_btc_spy_correlation(btc_data, spy_data)
            if correlation > 0.3:  # 상관관계가 어느 정도 있으면
                return True, "RISK_OFF_WARNING"
        
        # RISK_ON_SIGNAL 체크
        if btc_return_5d >= 10.0 and spy_return_5d < 3.0:
            return True, "RISK_ON_SIGNAL"
        
        return False, None
    
    def calculate_btc_gld_ratio(
        self, 
        btc_data: pd.DataFrame,
        gld_data: pd.DataFrame
    ) -> Dict:
        """
        BTC/GLD 비율 분석
        
        해석:
        - 비율 상승: 투기적 선호 증가
        - 비율 하락: 안전자산 선호
        
        Returns:
            Dict with ratio and trend
        """
        btc_empty = hasattr(btc_data, 'empty') and btc_data.empty if btc_data is not None else True
        gld_empty = hasattr(gld_data, 'empty') and gld_data.empty if gld_data is not None else True
        if btc_empty or gld_empty:
            return {
                'ratio': 0.0,
                'ratio_change_5d': 0.0,
                'ratio_change_20d': 0.0,
                'trend': 'NEUTRAL'
            }
        
        if 'Close' not in btc_data.columns or 'Close' not in gld_data.columns:
            return {
                'ratio': 0.0,
                'ratio_change_5d': 0.0,
                'ratio_change_20d': 0.0,
                'trend': 'NEUTRAL'
            }
        
        btc_close = btc_data['Close']
        gld_close = gld_data['Close']
        
        # 공통 인덱스
        common_index = btc_close.index.intersection(gld_close.index)
        if len(common_index) == 0:
            return {
                'ratio': 0.0,
                'ratio_change_5d': 0.0,
                'ratio_change_20d': 0.0,
                'trend': 'NEUTRAL'
            }
        
        # 비율 계산
        ratio_series = btc_close.loc[common_index] / gld_close.loc[common_index]
        
        if len(ratio_series) == 0:
            return {
                'ratio': 0.0,
                'ratio_change_5d': 0.0,
                'ratio_change_20d': 0.0,
                'trend': 'NEUTRAL'
            }
        
        current_ratio = float(ratio_series.iloc[-1])
        
        # 5일/20일 변화율
        if len(ratio_series) >= 5:
            ratio_change_5d = (current_ratio / float(ratio_series.iloc[-5]) - 1) * 100
        else:
            ratio_change_5d = 0.0
        
        if len(ratio_series) >= 20:
            ratio_change_20d = (current_ratio / float(ratio_series.iloc[-20]) - 1) * 100
        else:
            ratio_change_20d = 0.0
        
        # 추세 판단
        if ratio_change_5d > 5.0:
            trend = 'RISING'  # 투기적 선호 증가
        elif ratio_change_5d < -5.0:
            trend = 'FALLING'  # 안전자산 선호
        else:
            trend = 'NEUTRAL'
        
        return {
            'ratio': current_ratio,
            'ratio_change_5d': ratio_change_5d,
            'ratio_change_20d': ratio_change_20d,
            'trend': trend
        }
    
    def calculate_risk_contribution(self, correlation_regime: str) -> float:
        """
        전체 위험도에 기여하는 비중 결정
        
        상관관계가 높을수록 (위기 시) 기여도 증가
        
        Returns:
            float: 위험 기여도 (0-0.2, 즉 0-20%)
        """
        return self.RISK_CONTRIBUTION.get(correlation_regime, 0.05)
    
    def generate_interpretation(
        self,
        sentiment_score: float,
        sentiment_level: str,
        correlation: float,
        correlation_regime: str,
        is_leading: bool,
        leading_signal: Optional[str],
        risk_contribution: float,
        causality_analysis: Dict = None
    ) -> str:
        """
        분석 결과 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        if causality_analysis is None:
            causality_analysis = {}
        
        base_text = f"암호화폐 시장 심리는 {sentiment_level} 상태입니다 (점수: {sentiment_score:.1f}). "
        
        # 상관관계 해석
        if correlation_regime == "DECOUPLED":
            base_text += f"BTC-주식 상관관계가 낮습니다 ({correlation:.2f}). 독자적 움직임이 관찰됩니다. "
        elif correlation_regime == "COUPLED":
            base_text += f"BTC-주식 상관관계가 보통입니다 ({correlation:.2f}). 일부 연동이 관찰됩니다. "
        else:  # CRISIS_COUPLED
            base_text += f"⚠️ BTC-주식 상관관계가 높습니다 ({correlation:.2f}). 위기 동조화가 진행 중입니다. "
        
        # Granger Causality 인과관계 해석
        if causality_analysis and causality_analysis.get('relationship') != 'NO_CAUSALITY':
            relationship = causality_analysis.get('relationship', 'NO_CAUSALITY')
            x_to_y_pvalue = causality_analysis.get('x_to_y_pvalue', 1.0)
            y_to_x_pvalue = causality_analysis.get('y_to_x_pvalue', 1.0)
            optimal_lag = causality_analysis.get('optimal_lag', 0)
            
            if relationship == "X_LEADS":
                base_text += f"📊 Granger Causality 검정 결과: BTC가 SPY를 선행합니다 (p={x_to_y_pvalue:.3f}, 시차 {optimal_lag}일). "
            elif relationship == "Y_LEADS":
                base_text += f"📊 Granger Causality 검정 결과: SPY가 BTC를 선행합니다 (p={y_to_x_pvalue:.3f}, 시차 {optimal_lag}일). "
            elif relationship == "BIDIRECTIONAL":
                base_text += f"📊 Granger Causality 검정 결과: BTC와 SPY가 양방향 인과관계를 보입니다 (BTC→SPY: p={x_to_y_pvalue:.3f}, SPY→BTC: p={y_to_x_pvalue:.3f}). "
        
        # 선행지표 해석
        if is_leading and leading_signal:
            if leading_signal == "RISK_OFF_WARNING":
                base_text += "🚨 BTC가 주식보다 먼저 하락하고 있어 위험 회피 신호로 작용할 수 있습니다. "
            elif leading_signal == "RISK_ON_SIGNAL":
                base_text += "BTC가 주식보다 먼저 상승하고 있어 위험 선호 신호로 작용할 수 있습니다. "
        
        # 위험 기여도 해석
        if risk_contribution >= 0.15:
            base_text += f"전체 위험도에 {risk_contribution*100:.0f}% 기여하고 있어 주의가 필요합니다."
        elif risk_contribution >= 0.10:
            base_text += f"전체 위험도에 {risk_contribution*100:.0f}% 기여하고 있습니다."
        else:
            base_text += f"독자적 신호로 해석됩니다 (위험 기여도 {risk_contribution*100:.0f}%)."
        
        return base_text
    
    def analyze(
        self,
        btc_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        gld_data: pd.DataFrame
    ) -> CryptoSentimentResult:
        """
        전체 분석 실행
        
        Args:
            btc_data: BTC-USD 가격 데이터
            spy_data: SPY 가격 데이터 (상관관계 계산용)
            gld_data: GLD 가격 데이터 (BTC/GLD 비율용)
        
        Returns:
            CryptoSentimentResult 객체
        """
        # 심리 점수 계산
        sentiment_score, sentiment_level = self.calculate_sentiment_score(btc_data)
        
        # BTC-SPY 상관관계 계산
        correlation = self.calculate_btc_spy_correlation(btc_data, spy_data)
        
        # 상관관계 레짐 판단
        correlation_regime = self.determine_correlation_regime(correlation)
        
        # Granger Causality 검정 (인과관계 방향성 파악)
        causality_analysis = {}
        btc_empty = hasattr(btc_data, 'empty') and btc_data.empty if btc_data is not None else True
        spy_empty = hasattr(spy_data, 'empty') and spy_data.empty if spy_data is not None else True
        if not btc_empty and not spy_empty:
            if 'Close' in btc_data.columns and 'Close' in spy_data.columns:
                # 수익률 계산
                btc_returns = btc_data['Close'].pct_change().dropna()
                spy_returns = spy_data['Close'].pct_change().dropna()
                
                # 공통 인덱스로 정렬
                common_index = btc_returns.index.intersection(spy_returns.index)
                if len(common_index) >= 30:  # 최소 30일 데이터 필요
                    btc_aligned = btc_returns.loc[common_index]
                    spy_aligned = spy_returns.loc[common_index]
                    
                    # Granger Causality 검정 수행
                    causality_analysis = self.calculate_granger_causality(
                        series_x=btc_aligned,  # BTC가 원인 후보
                        series_y=spy_aligned,  # SPY가 결과 후보
                        max_lag=5
                    )
        
        # 선행지표 체크
        is_leading, leading_signal = self.check_leading_indicator(btc_data, spy_data)
        
        # 위험 기여도 계산
        risk_contribution = self.calculate_risk_contribution(correlation_regime)
        
        # BTC/GLD 비율 분석
        btc_gld_ratio = self.calculate_btc_gld_ratio(btc_data, gld_data)
        
        # 모멘텀 계산
        momentum = self.calculate_btc_momentum(btc_data)
        
        # 구성요소 통합
        components = {
            **momentum,
            'btc_gld_ratio': btc_gld_ratio,
            'correlation': correlation,
            'causality_analysis': causality_analysis,
        }
        
        # 해석 텍스트 생성
        interpretation = self.generate_interpretation(
            sentiment_score,
            sentiment_level,
            correlation,
            correlation_regime,
            is_leading,
            leading_signal,
            risk_contribution,
            causality_analysis
        )
        
        return CryptoSentimentResult(
            timestamp=datetime.now().isoformat(),
            sentiment_score=sentiment_score,
            sentiment_level=sentiment_level,
            btc_spy_correlation=correlation,
            correlation_regime=correlation_regime,
            is_leading_indicator=is_leading,
            leading_signal=leading_signal,
            risk_contribution=risk_contribution,
            components=components,
            interpretation=interpretation,
            causality_analysis=causality_analysis
        )

# critical_path_analyzer.py 파일 끝에 추가

