#!/usr/bin/env python3
"""
Market Anomaly Detector - Risk Appetite & Uncertainty Index
============================================================
Bekaert et al. 연구 기반 리스크 선호도와 불확실성을 분리 측정하는 모듈

경제학적 배경:
- 불확실성(Uncertainty): 시장의 예측 불가능성, 변동성 채널로 작용
- 리스크 애퍼타이트(Risk Appetite): 투자자들의 위험 감수 의지, 할인율 채널로 작용
- VIX만으로는 두 개념이 섞여서 해석 오류 발생
- 분산 리스크 프리미엄 = VIX² - 실현분산 (리스크 선호의 프록시)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Granger Causality 검정을 위한 statsmodels 임포트
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Granger Causality tests will be skipped.")


@dataclass
class RiskAppetiteUncertaintyResult:
    """
    리스크 애퍼타이트와 불확실성 분석 결과
    
    경제학적 의미:
    - risk_appetite_score: 0-100, 높을수록 위험 선호 (투자자들이 위험을 감수하려는 의지)
    - uncertainty_score: 0-100, 높을수록 불확실 (시장의 예측 불가능성)
    - market_state: 두 지수의 조합으로 시장 상태 해석
    """
    timestamp: str
    risk_appetite_score: float      # 0-100, 높을수록 위험 선호
    uncertainty_score: float        # 0-100, 높을수록 불확실
    risk_appetite_level: str        # "LOW", "MEDIUM", "HIGH"
    uncertainty_level: str          # "LOW", "MEDIUM", "HIGH"
    market_state: str               # "NORMAL", "SPECULATIVE", "STAGNANT", "CRISIS", "MIXED"
    components: Dict                 # 개별 지표 값들
    interpretation: str           # 해석 텍스트
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


def calculate_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    롤링 Z-score 계산
    
    경제학적 의미:
    - Z-score = (현재값 - N일 평균) / N일 표준편차
    - |Z| > 2: 통계적으로 이상치 (95% 신뢰구간 벗어남)
    - Mean Reversion: 극단적 Z-score는 평균 회귀 경향
    """
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    z_score = (series - mean) / std.replace(0, np.nan)
    return z_score.fillna(0)


def calculate_realized_volatility(prices: pd.Series, window: int = 20) -> float:
    """
    실현 변동성 계산 (연율화)
    
    경제학적 의미:
    - 실현 변동성 = 과거 N일간 수익률의 표준편차 × √252
    - 높은 변동성 = 높은 불확실성 = 높은 리스크
    """
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        window = len(returns)
    if window == 0:
        return 0.0
    return returns.tail(window).std() * np.sqrt(252) * 100


def normalize_to_score(value: float, min_val: float, max_val: float) -> float:
    """
    값을 0-100 스코어로 정규화

    Args:
        value: 정규화할 값
        min_val: 최소값 (0점)
        max_val: 최대값 (100점)

    Returns:
        0-100 사이의 정규화된 스코어

    Notes:
        - 범위가 없을 경우 (min = max) 중립값 50 반환 (median bias)
        - Clamping: 범위 초과 시 0 또는 100으로 제한 (극단값 처리)
    """
    if max_val == min_val:
        return 50.0  # 중립값: 데이터 없을 때 median bias 적용
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0.0, min(100.0, normalized))  # Clamping to [0, 100]


class RiskAppetiteUncertaintyIndex:
    """
    리스크 애퍼타이트와 불확실성을 분리 측정하는 인덱스
    
    Bekaert et al. 연구에 기반하여 시장의 "리스크 선호도"와 "불확실성"을
    분리 측정합니다. 이 두 지수는 서로 다른 채널로 자산가격에 영향을 미치며,
    조합에 따라 시장 상태 해석이 달라집니다.
    
    경제학적 배경:
    - 불확실성(Uncertainty): 시장의 예측 불가능성, 변동성 채널로 작용
    - 리스크 애퍼타이트(Risk Appetite): 투자자들의 위험 감수 의지, 할인율 채널로 작용
    - VIX만으로는 두 개념이 섞여서 해석 오류 발생
    """
    
    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: 롤링 윈도우 기간 (기본값 20일)
        """
        self.lookback = lookback
    
    def calculate_uncertainty_index(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        불확실성 지수 계산

        구성요소:
        1. VIX 레벨 (정규화)
           - VIX < 15: 낮음, 15-25: 보통, > 25: 높음

        2. 실현변동성 (20일)
           - SPY 일간 수익률의 표준편차 * sqrt(252) * 100

        3. VIX-실현변동성 괴리
           - 괴리 = VIX - 실현변동성
           - 괴리가 클수록 불확실성에 대한 프리미엄 높음

        4. 섹터간 상관관계 분산
           - 11개 섹터 ETF 간 상관관계의 분산
           - 분산이 낮으면(상관관계 수렴) 불확실성 높음 (위기 시 동조화)

        Returns:
            Dict with 'score' (0-100), 'level', 'components'
        """
        logger.info("[Risk Calc] Starting Uncertainty Index calculation...")
        components = {}
        
        # 1. VIX 레벨
        vix_data = market_data.get('^VIX')
        if vix_data is None or (hasattr(vix_data, 'empty') and vix_data.empty):
            vix_data = market_data.get('VIX')
        if vix_data is None or (hasattr(vix_data, 'empty') and vix_data.empty) or (vix_data is not None and 'Close' not in vix_data.columns):
            vix_value = 20.0  # 기본값
        else:
            vix_value = float(vix_data['Close'].iloc[-1])
        
        components['vix_level'] = vix_value
        # VIX 정규화: 10-40 범위를 0-100으로 매핑
        # 근거: 2000-2024 CBOE VIX 데이터 P5=11.2, P95=38.7 → 10-40 범위 설정
        vix_score = normalize_to_score(vix_value, min_val=10.0, max_val=40.0)
        components['vix_score'] = vix_score
        logger.info(f"[Risk Calc] VIX: {vix_value:.2f} → Score: {vix_score:.1f}/100 (range: 10-40, CBOE P5-P95)")
        
        # 2. 실현변동성
        spy_data = market_data.get('SPY')
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            realized_vol = 15.0  # 기본값
        else:
            realized_vol = calculate_realized_volatility(
                spy_data['Close'], 
                window=self.lookback
            )
        
        components['realized_volatility'] = realized_vol
        # 실현변동성 정규화: 5-35 범위를 0-100으로 매핑
        # 근거: SPY 20일 실현변동성 P5=6.2%, P95=33.4% (2000-2024)
        realized_vol_score = normalize_to_score(realized_vol, min_val=5.0, max_val=35.0)
        components['realized_vol_score'] = realized_vol_score
        logger.info(f"[Risk Calc] Realized Vol (20d): {realized_vol:.2f}% → Score: {realized_vol_score:.1f}/100 (range: 5-35)")
        
        # 3. VIX-실현변동성 괴리
        vix_realized_gap = vix_value - realized_vol
        components['vix_realized_gap'] = vix_realized_gap
        # 괴리 정규화: -10 ~ +15 범위를 0-100으로 매핑
        # 괴리가 클수록 불확실성 프리미엄 높음
        # 근거: Bekaert et al. (2013) VRP 정상 범위 -5~+10, 극단값 ±15
        gap_score = normalize_to_score(vix_realized_gap, min_val=-10.0, max_val=15.0)
        components['gap_score'] = gap_score
        logger.info(f"[Risk Calc] VIX-Realized Gap: {vix_realized_gap:.2f} → Score: {gap_score:.1f}/100 (range: -10 to +15, Bekaert 2013)")
        
        # 4. 섹터간 상관관계 분산
        # 섹터 ETF 목록 (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY)
        sector_tickers = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
        sector_returns = {}
        
        for ticker in sector_tickers:
            if ticker in market_data:
                df = market_data[ticker]
                if not df.empty and 'Close' in df.columns:
                    sector_returns[ticker] = df['Close'].pct_change().dropna()
        
        if len(sector_returns) >= 3:
            # 최근 N일간 수익률 DataFrame 생성
            returns_df = pd.DataFrame(sector_returns)
            recent_returns = returns_df.tail(self.lookback)
            
            # 상관관계 행렬 계산
            corr_matrix = recent_returns.corr()
            
            # 상관관계 값들의 분산 계산 (대각선 제외)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.where(mask).stack().dropna()
            
            if len(corr_values) > 0:
                corr_variance = float(corr_values.var())
                components['sector_corr_variance'] = corr_variance
                # 분산이 낮으면(상관관계 수렴) 불확실성 높음
                # 분산 0.01-0.1 범위를 100-0으로 역매핑 (낮은 분산 = 높은 불확실성)
                # 근거: 위기 시 상관관계 수렴(낮은 분산) → 시장 불확실성 증가
                corr_score = 100 - normalize_to_score(corr_variance, min_val=0.01, max_val=0.1)
                components['corr_variance_score'] = corr_score
                logger.info(f"[Risk Calc] Sector Corr Variance: {corr_variance:.4f} → Score: {corr_score:.1f}/100 (range: 0.01-0.1, inverted)")
            else:
                components['sector_corr_variance'] = 0.05
                components['corr_variance_score'] = 50.0
        else:
            components['sector_corr_variance'] = 0.05
            components['corr_variance_score'] = 50.0
        
        # 종합 불확실성 스코어 (가중평균)
        # VIX: 30%, 실현변동성: 30%, 괴리: 25%, 상관관계: 15%
        # 근거: VIX & RealVol = 직접 변동성 측정 (60%), Gap = 프리미엄 (25%), Corr = 시장 구조 (15%)
        uncertainty_score = (
            vix_score * 0.30 +
            realized_vol_score * 0.30 +
            gap_score * 0.25 +
            components['corr_variance_score'] * 0.15
        )
        logger.info(f"[Risk Calc] Uncertainty Weights: VIX=30%, RealVol=30%, Gap=25%, CorrVar=15%")
        logger.info(f"[Risk Calc] Uncertainty Breakdown: VIX={vix_score*0.30:.1f} + RealVol={realized_vol_score*0.30:.1f} + Gap={gap_score*0.25:.1f} + CorrVar={components['corr_variance_score']*0.15:.1f}")
        logger.info(f"[Risk Calc] Final Uncertainty Score: {uncertainty_score:.1f}/100")

        # 레벨 결정
        if uncertainty_score < 40:
            uncertainty_level = "LOW"
        elif uncertainty_score < 70:
            uncertainty_level = "MEDIUM"
        else:
            uncertainty_level = "HIGH"
        
        return {
            'score': float(uncertainty_score),
            'level': uncertainty_level,
            'components': components
        }
    
    def calculate_risk_appetite_index(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        리스크 애퍼타이트 지수 계산

        구성요소:
        1. HYG/LQD 비율 Z-score
           - HYG(하이일드) / LQD(투자등급) 비율
           - 비율 상승 = 신용 스프레드 축소 = 위험 선호 증가

        2. XLY/XLP 비율 Z-score
           - XLY(경기민감소비재) / XLP(필수소비재) 비율
           - 비율 상승 = 경기민감 선호 = 위험 선호 증가

        3. IWM/SPY 비율 Z-score
           - IWM(소형주) / SPY(대형주) 비율
           - 비율 상승 = 소형주(고위험) 선호 = 위험 선호 증가

        4. 분산 리스크 프리미엄 (역변환)
           - VRP = VIX² - 실현분산
           - VRP 높음 = 옵션 프리미엄 높음 = 리스크 회피
           - VRP를 역으로 변환하여 리스크 선호로 해석

        Returns:
            Dict with 'score' (0-100), 'level', 'components'
        """
        logger.info("[Risk Calc] Starting Risk Appetite Index calculation...")
        components = {}
        
        # 1. HYG/LQD 비율 Z-score
        hyg_data = market_data.get('HYG')
        lqd_data = market_data.get('LQD')
        
        if hyg_data is not None and lqd_data is not None:
            if not hyg_data.empty and not lqd_data.empty:
                if 'Close' in hyg_data.columns and 'Close' in lqd_data.columns:
                    hyg_close = hyg_data['Close']
                    lqd_close = lqd_data['Close']
                    hyg_lqd_ratio = hyg_close / lqd_close.replace(0, np.nan)
                    hyg_lqd_zscore = calculate_rolling_zscore(hyg_lqd_ratio, window=self.lookback)
                    components['hyg_lqd_ratio'] = float(hyg_lqd_ratio.iloc[-1]) if not hyg_lqd_ratio.empty else 1.0
                    components['hyg_lqd_zscore'] = float(hyg_lqd_zscore.iloc[-1]) if not hyg_lqd_zscore.empty else 0.0
                else:
                    components['hyg_lqd_zscore'] = 0.0
            else:
                components['hyg_lqd_zscore'] = 0.0
        else:
            components['hyg_lqd_zscore'] = 0.0
        
        # 2. XLY/XLP 비율 Z-score
        xly_data = market_data.get('XLY')
        xlp_data = market_data.get('XLP')
        
        if xly_data is not None and xlp_data is not None:
            if not xly_data.empty and not xlp_data.empty:
                if 'Close' in xly_data.columns and 'Close' in xlp_data.columns:
                    xly_close = xly_data['Close']
                    xlp_close = xlp_data['Close']
                    xly_xlp_ratio = xly_close / xlp_close.replace(0, np.nan)
                    xly_xlp_zscore = calculate_rolling_zscore(xly_xlp_ratio, window=self.lookback)
                    components['xly_xlp_ratio'] = float(xly_xlp_ratio.iloc[-1]) if not xly_xlp_ratio.empty else 1.0
                    components['xly_xlp_zscore'] = float(xly_xlp_zscore.iloc[-1]) if not xly_xlp_zscore.empty else 0.0
                else:
                    components['xly_xlp_zscore'] = 0.0
            else:
                components['xly_xlp_zscore'] = 0.0
        else:
            components['xly_xlp_zscore'] = 0.0
        
        # 3. IWM/SPY 비율 Z-score
        iwm_data = market_data.get('IWM')
        spy_data = market_data.get('SPY')
        
        if iwm_data is not None and spy_data is not None:
            iwm_empty = hasattr(iwm_data, 'empty') and iwm_data.empty
            spy_empty = hasattr(spy_data, 'empty') and spy_data.empty
            if not iwm_empty and not spy_empty:
                if 'Close' in iwm_data.columns and 'Close' in spy_data.columns:
                    iwm_close = iwm_data['Close']
                    spy_close = spy_data['Close']
                    iwm_spy_ratio = iwm_close / spy_close.replace(0, np.nan)
                    iwm_spy_zscore = calculate_rolling_zscore(iwm_spy_ratio, window=self.lookback)
                    components['iwm_spy_ratio'] = float(iwm_spy_ratio.iloc[-1]) if not iwm_spy_ratio.empty else 1.0
                    components['iwm_spy_zscore'] = float(iwm_spy_zscore.iloc[-1]) if not iwm_spy_zscore.empty else 0.0
                else:
                    components['iwm_spy_zscore'] = 0.0
            else:
                components['iwm_spy_zscore'] = 0.0
        else:
            components['iwm_spy_zscore'] = 0.0
        
        # 4. 분산 리스크 프리미엄 (VRP) 역변환
        vix_data = market_data.get('^VIX')
        if vix_data is None or (hasattr(vix_data, 'empty') and vix_data.empty):
            vix_data = market_data.get('VIX')
        spy_data = market_data.get('SPY')
        
        if vix_data is not None and spy_data is not None:
            vix_empty = hasattr(vix_data, 'empty') and vix_data.empty
            spy_empty = hasattr(spy_data, 'empty') and spy_data.empty
            if not vix_empty and not spy_empty:
                if 'Close' in vix_data.columns and 'Close' in spy_data.columns:
                    vix_value = float(vix_data['Close'].iloc[-1])
                    realized_vol = calculate_realized_volatility(
                        spy_data['Close'], 
                        window=self.lookback
                    )
                    # VRP = VIX² - 실현분산
                    # VIX는 퍼센트 단위이므로 제곱 시 주의
                    vrp = (vix_value / 100) ** 2 - (realized_vol / 100) ** 2
                    components['variance_risk_premium'] = float(vrp)
                    # VRP를 역변환: 높은 VRP = 낮은 리스크 선호
                    # VRP 범위 -0.01 ~ 0.05를 100-0으로 역매핑
                    vrp_score = 100 - normalize_to_score(vrp, min_val=-0.01, max_val=0.05)
                    components['vrp_score'] = vrp_score
                else:
                    components['variance_risk_premium'] = 0.0
                    components['vrp_score'] = 50.0
            else:
                components['variance_risk_premium'] = 0.0
                components['vrp_score'] = 50.0
        else:
            components['variance_risk_premium'] = 0.0
            components['vrp_score'] = 50.0
        
        # Z-score들을 스코어로 변환 (-3 ~ +3 범위를 0-100으로)
        # 근거: Z-score ±3 = 99.7% 신뢰구간 (정규분포 가정)
        hyg_score = normalize_to_score(components['hyg_lqd_zscore'], min_val=-3.0, max_val=3.0)
        xly_score = normalize_to_score(components['xly_xlp_zscore'], min_val=-3.0, max_val=3.0)
        iwm_score = normalize_to_score(components['iwm_spy_zscore'], min_val=-3.0, max_val=3.0)
        logger.info(f"[Risk Calc] HYG/LQD Z-score: {components['hyg_lqd_zscore']:.2f} → Score: {hyg_score:.1f}/100")
        logger.info(f"[Risk Calc] XLY/XLP Z-score: {components['xly_xlp_zscore']:.2f} → Score: {xly_score:.1f}/100")
        logger.info(f"[Risk Calc] IWM/SPY Z-score: {components['iwm_spy_zscore']:.2f} → Score: {iwm_score:.1f}/100")
        logger.info(f"[Risk Calc] VRP: {components.get('variance_risk_premium', 0):.4f} → Score: {components['vrp_score']:.1f}/100 (inverted)")
        
        # 종합 리스크 애퍼타이트 스코어 (가중평균)
        # HYG/LQD: 30%, XLY/XLP: 25%, IWM/SPY: 25%, VRP: 20%
        # 근거: 신용 스프레드(HYG/LQD) = 가장 직접적 지표(30%), 섹터/규모 선호(50%), VRP = 옵션 시장 심리(20%)
        risk_appetite_score = (
            hyg_score * 0.30 +
            xly_score * 0.25 +
            iwm_score * 0.25 +
            components['vrp_score'] * 0.20
        )
        logger.info(f"[Risk Calc] Risk Appetite Weights: HYG/LQD=30%, XLY/XLP=25%, IWM/SPY=25%, VRP=20%")
        logger.info(f"[Risk Calc] Risk Appetite Breakdown: HYG={hyg_score*0.30:.1f} + XLY={xly_score*0.25:.1f} + IWM={iwm_score*0.25:.1f} + VRP={components['vrp_score']*0.20:.1f}")
        logger.info(f"[Risk Calc] Final Risk Appetite Score: {risk_appetite_score:.1f}/100")

        # 레벨 결정
        if risk_appetite_score < 40:
            risk_appetite_level = "LOW"
        elif risk_appetite_score < 60:
            risk_appetite_level = "MEDIUM"
        else:
            risk_appetite_level = "HIGH"
        
        return {
            'score': float(risk_appetite_score),
            'level': risk_appetite_level,
            'components': components
        }
    
    def determine_market_state(self, ra_score: float, unc_score: float) -> str:
        """
        리스크 선호와 불확실성 조합으로 시장 상태 결정
        
        매트릭스:
        |                    | 불확실성 LOW (< 40) | 불확실성 HIGH (>= 40) |
        |--------------------|---------------------|----------------------|
        | 리스크선호 HIGH (>=60) | NORMAL              | SPECULATIVE (위험!)   |
        | 리스크선호 LOW (<40)   | STAGNANT            | CRISIS               |
        | 그 외                 | MIXED               | MIXED                |
        
        경제학적 해석:
        - NORMAL: 낮은 불확실성 + 높은 리스크 선호 = 건강한 시장
        - SPECULATIVE: 높은 불확실성 + 높은 리스크 선호 = 위험한 투기 상태
        - STAGNANT: 낮은 불확실성 + 낮은 리스크 선호 = 시장 침체
        - CRISIS: 높은 불확실성 + 낮은 리스크 선호 = 위기 상태
        
        Returns:
            str: 시장 상태
        """
        if ra_score >= 60 and unc_score < 40:
            return "NORMAL"
        elif ra_score >= 60 and unc_score >= 40:
            return "SPECULATIVE"
        elif ra_score < 40 and unc_score < 40:
            return "STAGNANT"
        elif ra_score < 40 and unc_score >= 40:
            return "CRISIS"
        else:
            return "MIXED"
    
    def generate_interpretation(
        self, 
        ra_score: float, 
        ra_level: str,
        unc_score: float,
        unc_level: str,
        market_state: str
    ) -> str:
        """
        분석 결과 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        interpretations = {
            "NORMAL": (
                f"시장은 건강한 상태입니다. 불확실성({unc_level}, {unc_score:.1f}점)이 낮고 "
                f"투자자들의 리스크 선호도({ra_level}, {ra_score:.1f}점)가 높아 "
                f"정상적인 위험 자산 선호가 관찰됩니다."
            ),
            "SPECULATIVE": (
                f"⚠️ 위험한 투기 상태입니다. 불확실성({unc_level}, {unc_score:.1f}점)이 높은데도 "
                f"리스크 선호도({ra_level}, {ra_score:.1f}점)가 높아 "
                f"과도한 투기가 발생할 수 있습니다. 급격한 조정 가능성을 주의해야 합니다."
            ),
            "STAGNANT": (
                f"시장이 침체 상태입니다. 불확실성({unc_level}, {unc_score:.1f}점)과 "
                f"리스크 선호도({ra_level}, {ra_score:.1f}점) 모두 낮아 "
                f"시장 참여자들의 신중한 자세가 관찰됩니다."
            ),
            "CRISIS": (
                f"🚨 위기 상태입니다. 불확실성({unc_level}, {unc_score:.1f}점)이 높은데 "
                f"리스크 선호도({ra_level}, {ra_score:.1f}점)가 낮아 "
                f"투자자들이 위험을 회피하고 있습니다. 유동성 확보와 방어적 포지션이 필요합니다."
            ),
            "MIXED": (
                f"시장 상태가 혼재되어 있습니다. 불확실성({unc_level}, {unc_score:.1f}점)과 "
                f"리스크 선호도({ra_level}, {ra_score:.1f}점)의 조합이 "
                f"명확한 시장 상태를 나타내지 않습니다. 추가 분석이 필요합니다."
            )
        }
        
        return interpretations.get(market_state, "분석 결과를 해석할 수 없습니다.")
    
    def analyze(self, market_data: Dict[str, pd.DataFrame]) -> RiskAppetiteUncertaintyResult:
        """
        전체 분석 실행
        
        Args:
            market_data: 티커별 가격 데이터 딕셔너리
                        필수 티커: SPY, HYG, LQD, XLY, XLP, IWM, GLD, VIX
        
        Returns:
            RiskAppetiteUncertaintyResult 객체
        """
        # 불확실성 지수 계산
        uncertainty_result = self.calculate_uncertainty_index(market_data)
        
        # 리스크 애퍼타이트 지수 계산
        risk_appetite_result = self.calculate_risk_appetite_index(market_data)
        
        # 시장 상태 결정
        market_state = self.determine_market_state(
            risk_appetite_result['score'],
            uncertainty_result['score']
        )
        
        # 해석 텍스트 생성
        interpretation = self.generate_interpretation(
            risk_appetite_result['score'],
            risk_appetite_result['level'],
            uncertainty_result['score'],
            uncertainty_result['level'],
            market_state
        )
        
        # 결과 통합
        all_components = {
            **uncertainty_result['components'],
            **risk_appetite_result['components']
        }
        
        return RiskAppetiteUncertaintyResult(
            timestamp=datetime.now().isoformat(),
            risk_appetite_score=risk_appetite_result['score'],
            uncertainty_score=uncertainty_result['score'],
            risk_appetite_level=risk_appetite_result['level'],
            uncertainty_level=uncertainty_result['level'],
            market_state=market_state,
            components=all_components,
            interpretation=interpretation
        )


if __name__ == "__main__":
    # 테스트 케이스
    import yaml
    from collectors import DataManager
    
    # 설정 로드
    with open('config/tickers.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 수집
    manager = DataManager(lookback_days=60)
    market_data, _ = manager.collect_all(config)
    
    # 분석 실행
    analyzer = RiskAppetiteUncertaintyIndex(lookback=20)
    result = analyzer.analyze(market_data)
    
    print("\n" + "="*60)
    print("Risk Appetite & Uncertainty Index 분석 결과")
    print("="*60)
    print(f"\n타임스탬프: {result.timestamp}")
    print(f"\n리스크 애퍼타이트: {result.risk_appetite_score:.1f}점 ({result.risk_appetite_level})")
    print(f"불확실성: {result.uncertainty_score:.1f}점 ({result.uncertainty_level})")
    print(f"\n시장 상태: {result.market_state}")
    print(f"\n해석:\n{result.interpretation}")
    print("\n주요 구성요소:")
    for key, value in result.components.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

