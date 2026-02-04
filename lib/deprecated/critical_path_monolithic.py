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

warnings.filterwarnings('ignore')

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
    """
    if max_val == min_val:
        return 50.0  # 기본값
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0.0, min(100.0, normalized))


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
        vix_score = normalize_to_score(vix_value, min_val=10.0, max_val=40.0)
        components['vix_score'] = vix_score
        
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
        realized_vol_score = normalize_to_score(realized_vol, min_val=5.0, max_val=35.0)
        components['realized_vol_score'] = realized_vol_score
        
        # 3. VIX-실현변동성 괴리
        vix_realized_gap = vix_value - realized_vol
        components['vix_realized_gap'] = vix_realized_gap
        # 괴리 정규화: -10 ~ +15 범위를 0-100으로 매핑
        # 괴리가 클수록 불확실성 프리미엄 높음
        gap_score = normalize_to_score(vix_realized_gap, min_val=-10.0, max_val=15.0)
        components['gap_score'] = gap_score
        
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
                corr_score = 100 - normalize_to_score(corr_variance, min_val=0.01, max_val=0.1)
                components['corr_variance_score'] = corr_score
            else:
                components['sector_corr_variance'] = 0.05
                components['corr_variance_score'] = 50.0
        else:
            components['sector_corr_variance'] = 0.05
            components['corr_variance_score'] = 50.0
        
        # 종합 불확실성 스코어 (가중평균)
        # VIX: 30%, 실현변동성: 30%, 괴리: 25%, 상관관계: 15%
        uncertainty_score = (
            vix_score * 0.30 +
            realized_vol_score * 0.30 +
            gap_score * 0.25 +
            components['corr_variance_score'] * 0.15
        )
        
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
        hyg_score = normalize_to_score(components['hyg_lqd_zscore'], min_val=-3.0, max_val=3.0)
        xly_score = normalize_to_score(components['xly_xlp_zscore'], min_val=-3.0, max_val=3.0)
        iwm_score = normalize_to_score(components['iwm_spy_zscore'], min_val=-3.0, max_val=3.0)
        
        # 종합 리스크 애퍼타이트 스코어 (가중평균)
        # HYG/LQD: 30%, XLY/XLP: 25%, IWM/SPY: 25%, VRP: 20%
        risk_appetite_score = (
            hyg_score * 0.30 +
            xly_score * 0.25 +
            iwm_score * 0.25 +
            components['vrp_score'] * 0.20
        )
        
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

warnings.filterwarnings('ignore')

# Granger Causality 검정을 위한 statsmodels 임포트
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Granger Causality tests will be skipped.")


@dataclass
class RegimeResult:
    """
    레짐 분석 결과
    
    경제학적 의미:
    - current_regime: 현재 시장 국면 (BULL/BEAR/TRANSITION/CRISIS)
    - regime_confidence: 레짐 판단의 확신도 (0-100%)
    - transition_probability: 레짐 전환 확률 (0-100%)
    - thresholds: 현재 레짐에 맞는 임계값 세트 (레짐별로 다름)
    """
    timestamp: str
    current_regime: str           # "BULL", "BEAR", "TRANSITION", "CRISIS"
    regime_confidence: float      # 0-100%
    transition_probability: float  # 레짐 전환 확률 (0-100%)
    transition_direction: str      # "BULL_TO_BEAR", "BEAR_TO_BULL", "STABLE", "UNCERTAIN"
    thresholds: Dict               # 현재 레짐에 맞는 임계값 세트
    ma_status: Dict                # MA 상태 정보
    interpretation: str            # 해석 텍스트
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


class EnhancedRegimeDetector:
    """
    레짐 탐지 및 레짐별 임계값 제공
    
    Maheu & McCurdy 연구에 기반하여 Bull/Bear/Transition 레짐을 탐지하고,
    각 레짐별로 다른 임계값 세트를 제공합니다. 레짐 전환 감지가 핵심입니다.
    
    경제학적 배경:
    - Bull과 Bear 시장은 수익률 분포 자체가 다름 (Maheu & McCurdy)
    - Bull: 낮은 변동성, 양의 평균, 정규분포에 가까움
    - Bear: 높은 변동성, 음의 평균, fat tail
    - 같은 -3% 하락도 Bull에서는 2σ 이벤트, Bear에서는 1σ 이벤트
    - 레짐 전환 초기에 신호가 가장 가치 있음
    """
    
    # 레짐별 임계값 정의
    REGIME_THRESHOLDS = {
        'BULL': {
            'volume_spike': 2.5,      # 거래량 급증 기준 (평균 대비 배수)
            'ma_deviation': -2.5,     # MA 이탈 기준 (%)
            'zscore_alert': 2.5,      # Z-score 경고 기준
            'vix_warning': 22,        # VIX 경고 레벨
            'return_alert': -2.0,     # 일간 수익률 경고 (%)
        },
        'TRANSITION': {
            'volume_spike': 2.0,
            'ma_deviation': -2.0,
            'zscore_alert': 2.0,
            'vix_warning': 25,
            'return_alert': -1.5,
        },
        'BEAR': {
            'volume_spike': 1.8,
            'ma_deviation': -1.5,
            'zscore_alert': 1.5,
            'vix_warning': 30,
            'return_alert': -1.0,
        },
        'CRISIS': {
            'volume_spike': 1.5,
            'ma_deviation': -1.0,
            'zscore_alert': 1.0,
            'vix_warning': 35,
            'return_alert': -0.5,
        }
    }
    
    def __init__(self, short_ma: int = 20, long_ma: int = 120, crisis_vix: float = 30):
        """
        Args:
            short_ma: 단기 이동평균 기간 (기본값 20일)
            long_ma: 장기 이동평균 기간 (기본값 120일)
            crisis_vix: 위기 판단 VIX 임계값 (기본값 30)
        """
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.crisis_vix = crisis_vix
        
        # 레짐 히스토리 저장 (최근 20일)
        self.regime_history: List[str] = []
        self.max_history = 20
    
    def detect_regime(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
        """
        현재 레짐 판단
        
        로직:
        1. CRISIS 체크 (우선순위 최고)
        2. BULL 조건
        3. BEAR 조건
        4. TRANSITION
        
        Returns:
            str: 레짐 이름
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return "TRANSITION"
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_short = float(ma_short.iloc[-1])
        current_ma_long = float(ma_long.iloc[-1])
        
        # 1. CRISIS 체크 (우선순위 최고)
        vix_value = None
        vix_empty = hasattr(vix_data, 'empty') and vix_data.empty if vix_data is not None else True
        if vix_data is not None and not vix_empty and 'Close' in vix_data.columns:
            vix_value = float(vix_data['Close'].iloc[-1])
        
        if len(close) >= 5:
            return_5d = (current_price / close.iloc[-5] - 1) * 100
        else:
            return_5d = 0.0
        
        if (vix_value is not None and vix_value >= self.crisis_vix) or return_5d < -5.0:
            return "CRISIS"
        
        # 2. BULL 조건
        if not pd.isna(current_ma_long):
            price_above_long = current_price > current_ma_long
            ma_short_above_long = current_ma_short > current_ma_long if not pd.isna(current_ma_short) else False
            
            if price_above_long and ma_short_above_long:
                return "BULL"
        
        # 3. BEAR 조건
        if not pd.isna(current_ma_long):
            price_below_long = current_price < current_ma_long
            ma_short_below_long = current_ma_short < current_ma_long if not pd.isna(current_ma_short) else False
            
            if price_below_long and ma_short_below_long:
                return "BEAR"
        
        # 4. TRANSITION (그 외 모든 경우)
        return "TRANSITION"
    
    def calculate_regime_confidence(self, spy_data: pd.DataFrame) -> float:
        """
        레짐 확신도 계산
        
        Returns:
            float: 0-100 사이 확신도
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return 50.0
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_short = float(ma_short.iloc[-1])
        current_ma_long = float(ma_long.iloc[-1])
        
        if pd.isna(current_ma_long) or current_ma_long == 0:
            return 50.0
        
        # 1. 현재가와 120일 MA의 거리 (0-50점)
        price_distance = abs((current_price / current_ma_long - 1) * 100)
        price_score = min(50.0, normalize_to_score(price_distance, min_val=0.0, max_val=5.0) * 0.5)
        
        # 2. 20일 MA와 120일 MA의 거리 (0-30점)
        if not pd.isna(current_ma_short) and current_ma_long != 0:
            ma_distance = abs((current_ma_short / current_ma_long - 1) * 100)
            ma_score = min(30.0, normalize_to_score(ma_distance, min_val=0.0, max_val=3.0) * 0.3)
        else:
            ma_score = 15.0
        
        # 3. 최근 N일간 레짐 일관성 (0-20점)
        if len(self.regime_history) >= 5:
            from collections import Counter
            recent_regimes = self.regime_history[-5:]
            counter = Counter(recent_regimes)
            most_common_count = counter.most_common(1)[0][1] if counter else 0
            consistency_ratio = most_common_count / len(recent_regimes)
            consistency_score = consistency_ratio * 20.0
        else:
            consistency_score = 10.0
        
        total_confidence = price_score + ma_score + consistency_score
        return min(100.0, max(0.0, total_confidence))
    
    def calculate_transition_probability(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> Tuple[float, str]:
        """
        레짐 전환 확률 계산
        
        Returns:
            Tuple[확률, 방향]
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return 0.0, "STABLE"
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        signals = []
        
        # 1. MA 근접도 체크
        if len(ma_short) > 0 and len(ma_long) > 0:
            current_ma_short = float(ma_short.iloc[-1])
            current_ma_long = float(ma_long.iloc[-1])
            
            if not pd.isna(current_ma_short) and not pd.isna(current_ma_long) and current_ma_long != 0:
                ma_distance_pct = abs((current_ma_short / current_ma_long - 1) * 100)
                if ma_distance_pct < 3.0:
                    signals.append(('ma_proximity', 30.0))
        
        # 2. MA 기울기 변화 체크
        if len(ma_short) >= 10:
            recent_slope = (float(ma_short.iloc[-1]) / float(ma_short.iloc[-5]) - 1) * 100 if len(ma_short) >= 5 else 0
            if len(ma_short) >= 10:
                prev_slope = (float(ma_short.iloc[-5]) / float(ma_short.iloc[-10]) - 1) * 100
                if (recent_slope > 0 and prev_slope < 0) or (recent_slope < 0 and prev_slope > 0):
                    signals.append(('ma_slope_change', 25.0))
        
        # 3. 거래량 증가 + 가격 역방향
        if 'Volume' in spy_data.columns and len(spy_data) >= 20:
            volume = spy_data['Volume']
            volume_ma = volume.rolling(window=20, min_periods=1).mean()
            
            if len(volume) > 0 and len(volume_ma) > 0:
                current_volume = float(volume.iloc[-1])
                avg_volume = float(volume_ma.iloc[-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if len(close) >= 3:
                    return_3d = (float(close.iloc[-1]) / float(close.iloc[-3]) - 1) * 100
                    
                    if volume_ratio > 1.3 and return_3d < -1.0:
                        signals.append(('volume_price_divergence', 20.0))
        
        # 4. VIX 추세 체크
        vix_empty = hasattr(vix_data, 'empty') and vix_data.empty if vix_data is not None else True
        if vix_data is not None and not vix_empty and 'Close' in vix_data.columns:
            vix_close = vix_data['Close']
            if len(vix_close) >= 5:
                vix_trend = []
                for i in range(len(vix_close) - 4, len(vix_close)):
                    if i > 0:
                        change = (float(vix_close.iloc[i]) / float(vix_close.iloc[i-1]) - 1) * 100
                        vix_trend.append(change)
                
                if len(vix_trend) == 4:
                    all_positive = all(x > 0 for x in vix_trend)
                    all_negative = all(x < 0 for x in vix_trend)
                    if all_positive or all_negative:
                        signals.append(('vix_trend', 25.0))
        
        total_probability = min(100.0, sum(prob for _, prob in signals))
        
        current_regime = self.detect_regime(spy_data, vix_data)
        
        if total_probability < 30.0:
            direction = "STABLE"
        elif current_regime == "BULL":
            direction = "BULL_TO_BEAR"
        elif current_regime == "BEAR":
            direction = "BEAR_TO_BULL"
        else:
            direction = "UNCERTAIN"
        
        return total_probability, direction
    
    def get_thresholds_for_regime(self, regime: str) -> Dict:
        """레짐에 맞는 임계값 세트 반환"""
        return self.REGIME_THRESHOLDS.get(regime, self.REGIME_THRESHOLDS['TRANSITION'])
    
    def get_ma_status(self, spy_data: pd.DataFrame) -> Dict:
        """이동평균 상태 정보 반환"""
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return {}
        
        close = spy_data['Close']
        ma_5 = close.rolling(window=5, min_periods=1).mean()
        ma_20 = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_120 = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_5 = float(ma_5.iloc[-1]) if not ma_5.empty else None
        current_ma_20 = float(ma_20.iloc[-1]) if not ma_20.empty else None
        current_ma_120 = float(ma_120.iloc[-1]) if not ma_120.empty else None
        
        price_vs_ma20 = ((current_price / current_ma_20 - 1) * 100) if current_ma_20 and current_ma_20 != 0 else None
        price_vs_ma120 = ((current_price / current_ma_120 - 1) * 100) if current_ma_120 and current_ma_120 != 0 else None
        ma20_vs_ma120 = ((current_ma_20 / current_ma_120 - 1) * 100) if current_ma_20 and current_ma_120 and current_ma_120 != 0 else None
        
        ma20_slope = None
        if len(ma_20) >= 5:
            ma20_slope = ((float(ma_20.iloc[-1]) / float(ma_20.iloc[-5]) - 1) * 100) if len(ma_20) >= 5 else None
        
        ma120_slope = None
        if len(ma_120) >= 20:
            ma120_slope = ((float(ma_120.iloc[-1]) / float(ma_120.iloc[-20]) - 1) * 100) if len(ma_120) >= 20 else None
        
        return {
            'ma_5': current_ma_5,
            'ma_20': current_ma_20,
            'ma_120': current_ma_120,
            'price_vs_ma20': price_vs_ma20,
            'price_vs_ma120': price_vs_ma120,
            'ma20_vs_ma120': ma20_vs_ma120,
            'ma20_slope': ma20_slope,
            'ma120_slope': ma120_slope,
        }
    
    def _apply_regime_buffer(self, new_regime: str) -> str:
        """레짐 전환 버퍼 적용 (급격한 스위칭 방지)"""
        if len(self.regime_history) == 0:
            return new_regime
        
        last_regime = self.regime_history[-1]
        
        if new_regime == last_regime:
            return new_regime
        
        if new_regime == "CRISIS":
            return new_regime
        
        if len(self.regime_history) >= 2:
            recent_regimes = self.regime_history[-2:]
            if new_regime in recent_regimes:
                return new_regime
        
        return last_regime
    
    def generate_interpretation(self, regime: str, confidence: float, transition_prob: float, transition_dir: str) -> str:
        """해석 텍스트 생성"""
        regime_names = {
            "BULL": "강세장",
            "BEAR": "약세장",
            "TRANSITION": "전환기",
            "CRISIS": "위기"
        }
        
        regime_name = regime_names.get(regime, regime)
        base_text = f"현재 시장은 {regime_name} 국면입니다. "
        
        if confidence >= 70:
            base_text += f"레짐 판단 확신도가 높습니다 ({confidence:.1f}%). "
        elif confidence >= 50:
            base_text += f"레짐 판단 확신도가 보통입니다 ({confidence:.1f}%). "
        else:
            base_text += f"레짐 판단 확신도가 낮습니다 ({confidence:.1f}%). "
        
        if transition_prob >= 70:
            base_text += f"⚠️ 레짐 전환 가능성이 높습니다 ({transition_prob:.1f}%). "
            if transition_dir == "BULL_TO_BEAR":
                base_text += "강세장에서 약세장으로 전환될 가능성이 있습니다."
            elif transition_dir == "BEAR_TO_BULL":
                base_text += "약세장에서 강세장으로 전환될 가능성이 있습니다."
            else:
                base_text += f"전환 방향: {transition_dir}"
        elif transition_prob >= 50:
            base_text += f"레짐 전환 가능성이 있습니다 ({transition_prob:.1f}%). "
        else:
            base_text += f"현재 레짐이 안정적입니다 (전환 확률 {transition_prob:.1f}%). "
        
        return base_text
    
    def analyze(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> RegimeResult:
        """전체 분석 실행"""
        detected_regime = self.detect_regime(spy_data, vix_data)
        final_regime = self._apply_regime_buffer(detected_regime)
        
        self.regime_history.append(final_regime)
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
        
        confidence = self.calculate_regime_confidence(spy_data)
        transition_prob, transition_dir = self.calculate_transition_probability(spy_data, vix_data)
        thresholds = self.get_thresholds_for_regime(final_regime)
        ma_status = self.get_ma_status(spy_data)
        interpretation = self.generate_interpretation(final_regime, confidence, transition_prob, transition_dir)
        
        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            current_regime=final_regime,
            regime_confidence=confidence,
            transition_probability=transition_prob,
            transition_direction=transition_dir,
            thresholds=thresholds,
            ma_status=ma_status,
            interpretation=interpretation
        )


@dataclass
class SpilloverEdge:
    """
    자산간 충격 전이(spillover) 경로
    
    경제학적 의미:
    - source: 충격이 발생한 자산 (위험 진원지)
    - target: 충격이 전이될 자산
    - edge_type: 전이 방향 (POSITIVE: 같은 방향, NEGATIVE: 반대 방향)
    - adjusted_lag: 레짐에 따라 조정된 시차 (위기 시 단축)
    """
    source: str                    # 출발 노드 (예: "TLT")
    target: str                    # 도착 노드 (예: "QQQ")
    edge_type: str                 # "POSITIVE", "NEGATIVE"
    base_lag: int                  # 기본 시차 (일)
    adjusted_lag: int              # 레짐 조정된 시차
    signal_strength: float         # 신호 강도 (0-100)
    is_active: bool               # 현재 활성화 여부
    source_move: float            # 소스 자산 움직임 (%)
    expected_target_move: str     # 예상 타겟 방향 ("UP", "DOWN")
    theory_note: str              # 경제학적 설명
    category: str = ""            # 경로 카테고리: 'liquidity', 'volatility', 'credit', 'concentration', 'rotation'
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


@dataclass
class SpilloverResult:
    """
    Spillover 네트워크 분석 결과
    
    경제학적 의미:
    - active_paths: 현재 활성화된 충격 전이 경로들
    - risk_score: 전이 위험 점수 (활성 경로 수와 강도 기반)
    - primary_risk_source: 가장 많은 경로의 소스가 되는 자산 (위험 진원지)
    """
    timestamp: str
    active_paths: List[SpilloverEdge]    # 활성화된 경로들
    risk_score: float                     # 전이 위험 점수 (0-100)
    primary_risk_source: str              # 주요 위험 진원지
    expected_impacts: Dict[str, str]      # 자산별 예상 영향
    interpretation: str                  # 해석 텍스트
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        d = asdict(self)
        d['active_paths'] = [edge.to_dict() for edge in self.active_paths]
        return d


class SpilloverNetwork:
    """
    자산간 충격 전이(spillover) 네트워크
    
    Boeckelmann 연구에 기반하여 자산간 충격 전이를 그래프 구조로 모델링하고,
    경로별 전이 신호를 탐지합니다.
    
    경제학적 배경:
    - 자산간 spillover 강도와 시차가 시간에 따라 변함
    - 위기 시: spillover 강도 증가, 시차 단축 (빠른 전이)
    - 평시: spillover 약함, 시차 김 (느린 전이)
    - 금융 경로(유동성) vs 실물 경로(공급망)는 시차가 다름
    
    노드: 자산/자산군
    엣지: 경제학적 인과관계
    """
    
    # 경로 정의 (경제학 이론 기반)
    SPILLOVER_PATHS = [
        # === 유동성/금리 경로 ===
        {
            'source': 'TLT',
            'target': 'QQQ', 
            'edge_type': 'POSITIVE',      # TLT↓(금리↑) → QQQ↓
            'base_lag': 3,
            'category': 'liquidity',
            'theory': '금리 상승 시 성장주 할인율 증가로 밸류에이션 압박'
        },
        {
            'source': 'DXY',
            'target': 'GLD',
            'edge_type': 'NEGATIVE',      # DXY↑ → GLD↓
            'base_lag': 1,
            'category': 'liquidity',
            'theory': '달러 강세 시 달러 표시 금 가격 하락 압력'
        },
        {
            'source': 'DXY',
            'target': 'EEM',
            'edge_type': 'NEGATIVE',      # DXY↑ → EEM↓
            'base_lag': 3,
            'category': 'liquidity',
            'theory': '달러 강세 시 신흥국 자금 유출, 달러 부채 부담 증가'
        },
        
        # === 변동성/공포 경로 ===
        {
            'source': '^VIX',
            'target': 'SPY',
            'edge_type': 'NEGATIVE',      # VIX↑ → SPY↓
            'base_lag': 1,
            'category': 'volatility',
            'theory': 'VIX 급등 시 옵션 딜러 감마 헷지 매도, 하락 가속'
        },
        {
            'source': 'VIX',
            'target': 'SPY',
            'edge_type': 'NEGATIVE',      # VIX↑ → SPY↓ (대체 티커)
            'base_lag': 1,
            'category': 'volatility',
            'theory': 'VIX 급등 시 옵션 딜러 감마 헷지 매도, 하락 가속'
        },
        
        # === 신용 경로 ===
        {
            'source': 'HYG',
            'target': 'XLF',
            'edge_type': 'POSITIVE',      # HYG↓ → XLF↓
            'base_lag': 3,
            'category': 'credit',
            'theory': '하이일드 스프레드 확대 시 금융섹터 신용 우려'
        },
        {
            'source': 'HYG',
            'target': 'IWM',
            'edge_type': 'POSITIVE',      # HYG↓ → IWM↓
            'base_lag': 5,
            'category': 'credit',
            'theory': '신용 경색 시 소형주 자금조달 어려움'
        },
        
        # === 빅테크/집중도 경로 ===
        {
            'source': 'QQQ',
            'target': 'SPY',
            'edge_type': 'POSITIVE',      # QQQ↓ → SPY↓
            'base_lag': 1,
            'category': 'concentration',
            'theory': 'MAG7이 SPY의 30% 차지, 빅테크 하락이 지수 끌어내림'
        },
        {
            'source': 'NVDA',
            'target': 'SMH',
            'edge_type': 'POSITIVE',      # NVDA↓ → SMH↓
            'base_lag': 1,
            'category': 'concentration',
            'theory': 'AI 대장주 균열이 반도체 섹터 전체로 전이'
        },
        
        # === 섹터 로테이션 경로 ===
        {
            'source': 'XLY',
            'target': 'SPY',
            'edge_type': 'POSITIVE',      # XLY↓ → SPY↓ (with lag)
            'base_lag': 5,
            'category': 'rotation',
            'theory': '경기민감 섹터 약세가 전체 시장 약세로 확산'
        },
    ]
    
    # 레짐별 시차 조정 계수
    LAG_ADJUSTMENTS = {
        'BULL': 1.0,        # 기본 시차 유지
        'TRANSITION': 0.8,  # 20% 단축
        'BEAR': 0.7,        # 30% 단축
        'CRISIS': 0.5,      # 50% 단축 (빠른 전이)
    }
    
    # 활성화 임계값 (레짐별) - 완화된 임계값
    ACTIVATION_THRESHOLDS = {
        'BULL': {
            'min_move': 1.2,       # 최소 1.2% 움직임 (기존 2.0에서 완화)
            'volume_ratio': 1.3,   # 거래량 1.3배 (기존 1.5에서 완화)
        },
        'BEAR': {
            'min_move': 1.0,       # 최소 1.0% 움직임 (기존 1.5에서 완화)
            'volume_ratio': 1.2,   # 거래량 1.2배 (기존 1.3에서 완화)
        },
        'TRANSITION': {
            'min_move': 1.0,       # 최소 1.0% 움직임 (기존 1.8에서 완화)
            'volume_ratio': 1.2,   # 거래량 1.2배 (기존 1.4에서 완화)
        },
        'CRISIS': {
            'min_move': 0.8,       # 최소 0.8% 움직임 (기존 1.0에서 완화)
            'volume_ratio': 1.1,   # 거래량 1.1배 (기존 1.2에서 완화)
        }
    }
    
    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: 롤링 윈도우 기간 (기본값 20일)
        """
        self.lookback = lookback
        self.paths = self.SPILLOVER_PATHS
    
    def adjust_lag_for_regime(self, base_lag: int, regime: str) -> int:
        """
        레짐에 따라 시차 조정
        
        경제학적 의미:
        - 위기 시 시차가 단축됨 (빠른 전이)
        - 평시에는 시차가 길어짐 (느린 전이)
        
        Returns:
            int: 조정된 시차 (최소 1일)
        """
        adjustment = self.LAG_ADJUSTMENTS.get(regime, 1.0)
        adjusted = max(1, int(base_lag * adjustment))
        return adjusted
    
    def calculate_source_signal(
        self, 
        source_data: pd.DataFrame, 
        lag: int
    ) -> Tuple[float, float]:
        """
        소스 자산의 신호 계산
        
        로직:
        - lag일 전 대비 현재 수익률 계산
        - 거래량 대비 이상 여부 확인
        - 신호 강도 = |수익률| × 거래량비율 (정규화)
        
        Returns:
            Tuple[움직임(%), 신호강도(0-100)]
        """
        if source_data.empty or 'Close' not in source_data.columns:
            return 0.0, 0.0
        
        close = source_data['Close']
        
        if len(close) < lag + 1:
            return 0.0, 0.0
        
        # lag일 전 대비 현재 수익률
        move_pct = (float(close.iloc[-1]) / float(close.iloc[-lag-1]) - 1) * 100
        
        # 거래량 비율 계산
        volume_ratio = 1.0
        if 'Volume' in source_data.columns and len(source_data) >= self.lookback:
            volume = source_data['Volume']
            current_volume = float(volume.iloc[-1]) if len(volume) > 0 else 0
            avg_volume = float(volume.tail(self.lookback).mean()) if len(volume) >= self.lookback else current_volume
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
        
        # 신호 강도 계산: |움직임| × 거래량비율 (정규화)
        # 움직임 0-5% 범위를 0-50점으로, 거래량비율 1.0-2.0 범위를 0-50점으로
        move_score = min(50.0, abs(move_pct) / 5.0 * 50.0)
        volume_score = min(50.0, (volume_ratio - 1.0) / 1.0 * 50.0) if volume_ratio >= 1.0 else 0.0
        
        signal_strength = min(100.0, move_score + volume_score)
        
        return move_pct, signal_strength
    
    def check_path_activation(
        self, 
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        path: Dict,
        regime: str
    ) -> Optional[SpilloverEdge]:
        """
        개별 경로 활성화 여부 확인
        
        활성화 조건:
        1. 소스 자산이 임계값 이상 움직임 (예: 3일간 ±2%)
        2. 거래량이 평균 대비 1.5배 이상
        
        신호 강도 계산:
        - 움직임 크기 × 거래량 비율 × 레짐 가중치
        
        Returns:
            SpilloverEdge 객체 (활성화되지 않으면 None)
        """
        if source_data.empty or 'Close' not in source_data.columns:
            return None
        
        # 레짐별 임계값 가져오기
        thresholds = self.ACTIVATION_THRESHOLDS.get(regime, self.ACTIVATION_THRESHOLDS['BEAR'])
        min_move = thresholds['min_move']
        min_volume_ratio = thresholds['volume_ratio']
        
        # 시차 조정
        base_lag = path.get('base_lag', 3)
        adjusted_lag = self.adjust_lag_for_regime(base_lag, regime)
        
        # 소스 신호 계산
        source_move, signal_strength = self.calculate_source_signal(source_data, adjusted_lag)
        
        # 거래량 체크
        volume_ratio = 1.0
        if 'Volume' in source_data.columns and len(source_data) >= self.lookback:
            volume = source_data['Volume']
            current_volume = float(volume.iloc[-1]) if len(volume) > 0 else 0
            avg_volume = float(volume.tail(self.lookback).mean()) if len(volume) >= self.lookback else current_volume
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
        
        # 활성화 조건 체크
        abs_move = abs(source_move)
        is_active = (abs_move >= min_move) and (volume_ratio >= min_volume_ratio)
        
        if not is_active:
            return None
        
        # 레짐 가중치 적용 (위기 시 신호 강도 증가)
        regime_weights = {
            'BULL': 1.0,
            'TRANSITION': 1.1,
            'BEAR': 1.2,
            'CRISIS': 1.5
        }
        weight = regime_weights.get(regime, 1.0)
        final_signal_strength = min(100.0, signal_strength * weight)
        
        # 예상 타겟 방향 결정
        edge_type = path.get('edge_type', 'POSITIVE')
        if edge_type == 'POSITIVE':
            # 같은 방향: source가 하락하면 target도 하락
            expected_direction = "DOWN" if source_move < 0 else "UP"
        else:  # NEGATIVE
            # 반대 방향: source가 상승하면 target은 하락
            expected_direction = "DOWN" if source_move > 0 else "UP"
        
        return SpilloverEdge(
            source=path['source'],
            target=path['target'],
            edge_type=edge_type,
            base_lag=base_lag,
            adjusted_lag=adjusted_lag,
            signal_strength=final_signal_strength,
            is_active=True,
            source_move=source_move,
            expected_target_move=expected_direction,
            theory_note=path.get('theory', ''),
            category=path.get('category', '')
        )
    
    def get_expected_impacts(self, active_paths: List[SpilloverEdge]) -> Dict[str, str]:
        """
        활성화된 경로 기반으로 각 자산 예상 영향
        
        여러 경로가 같은 타겟을 가리킬 경우, 신호 강도가 높은 경로 우선
        
        Returns:
            Dict[ticker, expected_direction]
            예: {"QQQ": "DOWN", "GLD": "DOWN", "SPY": "DOWN"}
        """
        impacts = {}
        
        # 타겟별로 신호 강도가 가장 높은 경로 선택
        target_paths = {}
        for edge in active_paths:
            target = edge.target
            if target not in target_paths or edge.signal_strength > target_paths[target].signal_strength:
                target_paths[target] = edge
        
        # 예상 영향 결정
        for target, edge in target_paths.items():
            impacts[target] = edge.expected_target_move
        
        return impacts
    
    def identify_risk_source(self, active_paths: List[SpilloverEdge]) -> str:
        """
        가장 많은 경로의 소스가 되는 자산 = 위험 진원지
        
        Returns:
            str: 티커 이름 (활성 경로가 없으면 "NONE")
        """
        if not active_paths:
            return "NONE"
        
        # 소스별 경로 수 집계
        from collections import Counter
        source_counts = Counter(edge.source for edge in active_paths)
        
        if not source_counts:
            return "NONE"
        
        # 가장 많은 경로를 가진 소스 반환
        primary_source = source_counts.most_common(1)[0][0]
        return primary_source
    
    def calculate_risk_score(self, active_paths: List[SpilloverEdge], regime: str) -> float:
        """
        전이 위험 점수 계산 (0-100)
        
        로직:
        - 활성 경로 수 (최대 50점)
        - 평균 신호 강도 (최대 50점)
        - 레짐 가중치 적용
        
        Returns:
            float: 위험 점수 (0-100)
        """
        if not active_paths:
            return 0.0
        
        # 활성 경로 수 점수 (최대 50점)
        num_paths = len(active_paths)
        path_score = min(50.0, num_paths / 10.0 * 50.0)  # 10개 경로 = 50점
        
        # 평균 신호 강도 점수 (최대 50점)
        avg_strength = sum(edge.signal_strength for edge in active_paths) / len(active_paths)
        strength_score = avg_strength * 0.5  # 최대 50점
        
        base_score = path_score + strength_score
        
        # 레짐 가중치
        regime_weights = {
            'BULL': 0.8,
            'TRANSITION': 1.0,
            'BEAR': 1.2,
            'CRISIS': 1.5
        }
        weight = regime_weights.get(regime, 1.0)
        
        final_score = min(100.0, base_score * weight)
        return final_score
    
    def generate_interpretation(
        self,
        active_paths: List[SpilloverEdge],
        risk_score: float,
        primary_source: str,
        expected_impacts: Dict[str, str]
    ) -> str:
        """
        분석 결과 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        if not active_paths:
            return "현재 활성화된 충격 전이 경로가 없습니다. 시장이 상대적으로 안정적입니다."
        
        base_text = f"총 {len(active_paths)}개의 충격 전이 경로가 활성화되어 있습니다. "
        
        # 위험 점수 해석
        if risk_score >= 70:
            base_text += f"⚠️ 전이 위험 점수가 높습니다 ({risk_score:.1f}점). "
        elif risk_score >= 50:
            base_text += f"전이 위험 점수가 보통입니다 ({risk_score:.1f}점). "
        else:
            base_text += f"전이 위험 점수가 낮습니다 ({risk_score:.1f}점). "
        
        # 주요 위험 진원지
        if primary_source != "NONE":
            base_text += f"주요 위험 진원지는 {primary_source}입니다. "
        
        # 예상 영향
        if expected_impacts:
            down_assets = [ticker for ticker, direction in expected_impacts.items() if direction == "DOWN"]
            up_assets = [ticker for ticker, direction in expected_impacts.items() if direction == "UP"]
            
            if down_assets:
                base_text += f"하락 압력이 예상되는 자산: {', '.join(down_assets)}. "
            if up_assets:
                base_text += f"상승 압력이 예상되는 자산: {', '.join(up_assets)}. "
        
        return base_text
    
    def analyze(
        self, 
        market_data: Dict[str, pd.DataFrame],
        regime: str
    ) -> SpilloverResult:
        """
        전체 네트워크 분석
        
        1. 각 경로별 활성화 여부 확인
        2. 활성화된 경로들의 위험도 합산
        3. 주요 위험 진원지 식별
        4. 타겟 자산별 예상 영향 정리
        
        Args:
            market_data: 티커별 가격 데이터 딕셔너리
            regime: 현재 레짐 ("BULL", "BEAR", "TRANSITION", "CRISIS")
        
        Returns:
            SpilloverResult 객체
        """
        active_paths = []
        
        # 각 경로별 활성화 여부 확인
        for path in self.paths:
            source_ticker = path['source']
            target_ticker = path['target']
            
            # 소스 데이터 가져오기 (티커명 변형 시도)
            source_data = market_data.get(source_ticker)
            if source_data is None:
                # 대체 티커 시도 (예: ^VIX -> VIX)
                alt_source = source_ticker.replace('^', '')
                source_data = market_data.get(alt_source)
            
            # 타겟 데이터 가져오기
            target_data = market_data.get(target_ticker)
            
            if source_data is None or target_data is None:
                continue
            
            # 경로 활성화 체크
            edge = self.check_path_activation(source_data, target_data, path, regime)
            if edge is not None:
                active_paths.append(edge)
        
        # 위험 점수 계산
        risk_score = self.calculate_risk_score(active_paths, regime)
        
        # 주요 위험 진원지 식별
        primary_source = self.identify_risk_source(active_paths)
        
        # 예상 영향 계산
        expected_impacts = self.get_expected_impacts(active_paths)
        
        # 해석 텍스트 생성
        interpretation = self.generate_interpretation(
            active_paths,
            risk_score,
            primary_source,
            expected_impacts
        )
        
        return SpilloverResult(
            timestamp=datetime.now().isoformat(),
            active_paths=active_paths,
            risk_score=risk_score,
            primary_risk_source=primary_source,
            expected_impacts=expected_impacts,
            interpretation=interpretation
        )

# critical_path_analyzer.py 파일 끝에 추가

@dataclass
class CryptoSentimentResult:
    """
    암호화폐 심리 분석 결과
    
    경제학적 의미:
    - sentiment_score: 암호화폐 시장 심리 점수 (0-100)
    - btc_spy_correlation: BTC-주식 상관관계 (레짐에 따라 다름)
    - correlation_regime: 상관관계 기반 레짐 (DECOUPLED/COUPLED/CRISIS_COUPLED)
    - is_leading_indicator: 선행지표로 작동 중인지 여부
    - risk_contribution: 전체 위험도에 기여하는 비중 (위기 시 증가)
    - causality_analysis: Granger Causality 검정 결과 (인과관계 방향성)
    """
    timestamp: str
    sentiment_score: float             # 0-100
    sentiment_level: str               # "EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"
    btc_spy_correlation: float         # 20일 롤링 상관관계
    correlation_regime: str            # "DECOUPLED", "COUPLED", "CRISIS_COUPLED"
    is_leading_indicator: bool         # 선행지표로 작동 중인지
    leading_signal: Optional[str]      # "RISK_OFF_WARNING", "RISK_ON_SIGNAL", None
    risk_contribution: float           # 전체 위험도에 기여하는 비중 (0-20%)
    components: Dict                   # 개별 지표 값들
    interpretation: str                # 해석 텍스트
    causality_analysis: Dict = field(default_factory=lambda: {})  # Granger Causality 검정 결과
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


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

@dataclass
class CriticalPathResult:
    """
    Critical Path 분석 종합 결과
    
    경제학적 의미:
    - total_risk_score: 전체 시장 위험도 (0-100)
    - path_contributions: 경로별 위험 기여도 (합계 = total_risk_score)
    - primary_risk_path: 가장 큰 기여도를 가진 경로 (위험 진원지)
    """
    timestamp: str
    
    # 전체 위험도
    total_risk_score: float           # 0-100
    risk_level: str                   # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # 레짐 정보
    current_regime: str
    regime_confidence: float
    transition_probability: float
    
    # 경로별 기여도 (raw scores, 절대값)
    path_contributions: Dict[str, float]  
    # 예: {"liquidity": 25, "concentration": 22, "credit": 10, ...}
    
    # 경로별 분포 (100% 정규화, 시각화용)
    path_distribution: Dict[str, float]
    # 예: {"liquidity": 35.2%, "concentration": 30.1%, ...}
    
    # 하위 모듈 결과
    risk_appetite_result: RiskAppetiteUncertaintyResult
    regime_result: RegimeResult
    spillover_result: SpilloverResult
    crypto_result: CryptoSentimentResult
    
    # 해석 및 경고
    primary_risk_path: str            # 가장 큰 기여도 경로
    active_warnings: List[str]        # 활성화된 경고 목록
    interpretation: str               # 종합 해석
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        d = asdict(self)
        d['risk_appetite_result'] = self.risk_appetite_result.to_dict()
        d['regime_result'] = self.regime_result.to_dict()
        d['spillover_result'] = self.spillover_result.to_dict()
        d['crypto_result'] = self.crypto_result.to_dict()
        return d


# ============================================================================
# Stress Regime Multiplier (Elicit Report Enhancement)
# ============================================================================

@dataclass
class StressMultiplierResult:
    """스트레스 레짐 승수 결과"""
    timestamp: str
    base_multiplier: float           # 기본 레짐 승수
    correlation_adjustment: float    # 상관관계 조정 (Longin-Solnik)
    volatility_scaling: float        # 변동성 스케일링
    contagion_factor: float          # 전염 가속 계수
    final_multiplier: float          # 최종 승수
    regime: str                      # 현재 레짐
    methodology_notes: str           # 방법론 설명
    academic_references: List[str]   # 학술 참고문헌


class StressRegimeMultiplier:
    """
    스트레스 레짐 승수 계산기 (Elicit Report Enhancement)

    학술적 근거:
    - Longin & Solnik (2001): 극단적 시장에서 상관관계 비대칭 발견
    - Forbes & Rigobon (2002): 위기 시 "contagion" vs "interdependence" 구분
    - Elicit Report: 위기 시 상관관계 61.4% 증가 확인

    Perplexity 검증 결과:
    - 학술적 합의: 스트레스 기간에 상관관계 증가 (confirmatory bias 주의)
    - Forbes-Rigobon 조정: 변동성 증가로 인한 spurious correlation 보정 필요
    - 실무적 함의: 분산 효과 감소 → 리스크 과소평가 방지
    """

    # 레짐별 기본 승수 (기존 로직 기반)
    BASE_MULTIPLIERS = {
        'BULL': 0.8,
        'NEUTRAL': 1.0,
        'TRANSITION': 1.0,
        'BEAR': 1.2,
        'CRISIS': 1.5
    }

    # 상관관계 증가 계수 (Elicit: 61.4% 증가)
    CRISIS_CORRELATION_INCREASE = 0.614

    # VIX 임계값 (스트레스 레벨 결정)
    VIX_THRESHOLDS = {
        'normal': 20,
        'elevated': 25,
        'stress': 30,
        'crisis': 40
    }

    def __init__(
        self,
        correlation_window: int = 60,
        volatility_window: int = 20
    ):
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window

    def calculate_multiplier(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_regime: str,
        vix_level: Optional[float] = None
    ) -> StressMultiplierResult:
        """
        스트레스 레짐 승수 계산

        Parameters:
        -----------
        market_data : Dict[str, DataFrame]
            시장 데이터 (SPY, QQQ 등)
        current_regime : str
            현재 시장 레짐 (BULL/BEAR/NEUTRAL/CRISIS)
        vix_level : float (optional)
            현재 VIX 레벨

        Returns:
        --------
        StressMultiplierResult
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 1. 기본 레짐 승수
        base_multiplier = self.BASE_MULTIPLIERS.get(current_regime.upper(), 1.0)

        # 2. VIX 기반 변동성 스케일링
        if vix_level is None:
            vix_data = market_data.get('^VIX') or market_data.get('VIX')
            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                vix_level = float(vix_data['Close'].iloc[-1])
            else:
                vix_level = 20.0  # 기본값

        volatility_scaling = self._calculate_volatility_scaling(vix_level)

        # 3. 상관관계 조정 (Longin-Solnik / Forbes-Rigobon)
        correlation_adjustment = self._calculate_correlation_adjustment(
            market_data, current_regime, vix_level
        )

        # 4. 전염 가속 계수
        contagion_factor = self._calculate_contagion_factor(
            market_data, current_regime
        )

        # 5. 최종 승수 계산
        # 공식: Final = Base × (1 + VolScaling) × (1 + CorrAdj) × (1 + Contagion)
        final_multiplier = (
            base_multiplier
            * (1 + volatility_scaling)
            * (1 + correlation_adjustment)
            * (1 + contagion_factor)
        )

        # 상한선 (과도한 승수 방지)
        final_multiplier = min(final_multiplier, 3.0)

        # 방법론 설명
        methodology_notes = self._generate_methodology_notes(
            base_multiplier, volatility_scaling,
            correlation_adjustment, contagion_factor, vix_level
        )

        return StressMultiplierResult(
            timestamp=timestamp,
            base_multiplier=base_multiplier,
            correlation_adjustment=correlation_adjustment,
            volatility_scaling=volatility_scaling,
            contagion_factor=contagion_factor,
            final_multiplier=final_multiplier,
            regime=current_regime,
            methodology_notes=methodology_notes,
            academic_references=[
                "Longin & Solnik (2001): Extreme Correlation of International Equity Markets",
                "Forbes & Rigobon (2002): No Contagion, Only Interdependence",
                "Elicit Report (2026): 61.4% correlation increase during stress"
            ]
        )

    def _calculate_volatility_scaling(self, vix_level: float) -> float:
        """VIX 기반 변동성 스케일링 계산"""
        # 정상 VIX (20) 대비 초과분에 비례하여 스케일링
        if vix_level <= self.VIX_THRESHOLDS['normal']:
            return 0.0
        elif vix_level <= self.VIX_THRESHOLDS['elevated']:
            return (vix_level - 20) / 100  # 0~5% 추가
        elif vix_level <= self.VIX_THRESHOLDS['stress']:
            return (vix_level - 20) / 50   # 0~20% 추가
        elif vix_level <= self.VIX_THRESHOLDS['crisis']:
            return (vix_level - 20) / 40   # 0~50% 추가
        else:
            return 0.5 + (vix_level - 40) / 100  # 50%+ 추가

    def _calculate_correlation_adjustment(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime: str,
        vix_level: float
    ) -> float:
        """
        상관관계 조정 계산 (Longin-Solnik / Forbes-Rigobon)

        핵심 아이디어:
        - 위기 시 자산 간 상관관계가 비선형적으로 증가
        - Elicit Report: 평균 61.4% 상관관계 증가 관측
        - Forbes-Rigobon: 변동성 증가로 인한 spurious correlation 보정 필요
        """
        # 스트레스 레벨 결정
        if regime.upper() in ['CRISIS'] or vix_level > 35:
            stress_level = 'CRISIS'
        elif regime.upper() in ['BEAR'] or vix_level > 25:
            stress_level = 'STRESS'
        else:
            stress_level = 'NORMAL'

        # 상관관계 조정값
        if stress_level == 'CRISIS':
            # Elicit 61.4% × 0.7 (Forbes-Rigobon 보정)
            raw_adjustment = self.CRISIS_CORRELATION_INCREASE * 0.7
        elif stress_level == 'STRESS':
            raw_adjustment = self.CRISIS_CORRELATION_INCREASE * 0.4
        else:
            raw_adjustment = 0.0

        # 실제 상관관계 변화 측정 (데이터 있으면)
        try:
            empirical_adj = self._measure_empirical_correlation_change(market_data)
            if empirical_adj is not None:
                # 이론값과 실증값의 가중 평균
                return 0.6 * raw_adjustment + 0.4 * empirical_adj
        except Exception:
            pass

        return raw_adjustment

    def _measure_empirical_correlation_change(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """실제 상관관계 변화 측정"""
        # SPY-QQQ, SPY-TLT 등 주요 자산 쌍의 롤링 상관관계 변화
        spy_data = market_data.get('SPY')
        qqq_data = market_data.get('QQQ')
        tlt_data = market_data.get('TLT')

        if spy_data is None or qqq_data is None:
            return None

        try:
            # 수익률 계산
            spy_ret = spy_data['Close'].pct_change().dropna()
            qqq_ret = qqq_data['Close'].pct_change().dropna()

            # 최근 상관관계 vs 장기 상관관계
            if len(spy_ret) < self.correlation_window:
                return None

            short_window = min(20, len(spy_ret) // 2)
            long_corr = spy_ret.tail(self.correlation_window).corr(
                qqq_ret.tail(self.correlation_window)
            )
            short_corr = spy_ret.tail(short_window).corr(
                qqq_ret.tail(short_window)
            )

            # 상관관계 변화율
            if long_corr != 0:
                return (short_corr - long_corr) / abs(long_corr)
            return 0.0
        except Exception:
            return None

    def _calculate_contagion_factor(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime: str
    ) -> float:
        """
        전염 가속 계수 계산

        위기 시 자산 간 충격 전파 속도가 가속화됨을 반영
        """
        if regime.upper() not in ['BEAR', 'CRISIS']:
            return 0.0

        # 섹터 ETF 동조화 측정
        sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP']
        available_sectors = [s for s in sector_etfs if s in market_data]

        if len(available_sectors) < 3:
            # 데이터 부족 시 레짐 기반 기본값
            return 0.1 if regime.upper() == 'BEAR' else 0.2

        try:
            returns = {}
            for sector in available_sectors:
                if 'Close' in market_data[sector].columns:
                    returns[sector] = market_data[sector]['Close'].pct_change().dropna()

            if len(returns) < 3:
                return 0.1

            returns_df = pd.DataFrame(returns).dropna()
            if len(returns_df) < 20:
                return 0.1

            # 상관관계 행렬
            corr_matrix = returns_df.tail(20).corr()

            # 평균 상관관계 (대각선 제외)
            n = len(corr_matrix)
            if n < 2:
                return 0.1

            off_diag = corr_matrix.values[~np.eye(n, dtype=bool)]
            avg_corr = np.mean(off_diag)

            # 높은 동조화 = 높은 전염 가속
            # 평균 상관관계 > 0.7이면 전염 가속
            if avg_corr > 0.8:
                return 0.3
            elif avg_corr > 0.7:
                return 0.2
            elif avg_corr > 0.5:
                return 0.1
            return 0.0
        except Exception:
            return 0.1

    def _generate_methodology_notes(
        self,
        base: float, vol: float, corr: float, contagion: float, vix: float
    ) -> str:
        """방법론 설명 생성"""
        notes = []
        notes.append(f"기본 레짐 승수: {base:.2f}")

        if vol > 0:
            notes.append(f"변동성 스케일링: +{vol*100:.1f}% (VIX={vix:.1f})")

        if corr > 0:
            notes.append(
                f"상관관계 조정: +{corr*100:.1f}% "
                f"(Longin-Solnik/Forbes-Rigobon 기반)"
            )

        if contagion > 0:
            notes.append(f"전염 가속: +{contagion*100:.1f}% (섹터 동조화)")

        return " | ".join(notes)

    def apply_to_risk_score(
        self,
        base_risk_score: float,
        multiplier_result: StressMultiplierResult
    ) -> Tuple[float, str]:
        """
        리스크 점수에 스트레스 승수 적용

        Parameters:
        -----------
        base_risk_score : float
            기본 리스크 점수 (0-100)
        multiplier_result : StressMultiplierResult
            스트레스 승수 결과

        Returns:
        --------
        Tuple[adjusted_score, explanation]
        """
        adjusted_score = base_risk_score * multiplier_result.final_multiplier
        adjusted_score = min(100.0, adjusted_score)  # 상한 100

        explanation = (
            f"Base: {base_risk_score:.1f} × Multiplier: {multiplier_result.final_multiplier:.2f} "
            f"= Adjusted: {adjusted_score:.1f}"
        )

        return adjusted_score, explanation


class CriticalPathAggregator:
    """
    Critical Path 분석 통합 모듈
    
    4개 하위 모듈을 조율하고 최종 결과 산출
    
    하위 모듈:
    1. RiskAppetiteUncertaintyIndex: 리스크 선호도와 불확실성 분리 측정
    2. EnhancedRegimeDetector: 레짐 탐지 및 레짐별 임계값 제공
    3. SpilloverNetwork: 자산간 충격 전이 네트워크
    4. CryptoSentimentBlock: 암호화폐 심리 지표 블록
    """
    
    # 경로별 기본 가중치
    BASE_PATH_WEIGHTS = {
        'liquidity': 0.25,       # 유동성/금리 경로
        'concentration': 0.25,   # AI/빅테크 집중 경로
        'credit': 0.20,          # 신용 스트레스 경로
        'volatility': 0.15,      # 변동성/공포 경로
        'rotation': 0.10,        # 섹터 로테이션 경로
        'crypto': 0.05,          # 암호화폐 (기본값, 동적 조정)
    }
    
    # Perplexity 제안 기반 검증된 임계값 상수
    THRESHOLDS = {
        'zscore': {'warning': 1.5, 'alert': 2.0, 'critical': 2.5},
        'ml_prob': {'warning': 0.20, 'alert': 0.40, 'critical': 0.60},
        'rsi': {'overbought': 70, 'oversold': 30, 'extreme': {'ob': 80, 'os': 20}},
        'drawdown': {'days': 10, 'threshold': -0.05},
        'vix': {'normal': (15, 25), 'stress': 30, 'complacency': 12},
        'bb': {'window': 20, 'std': 2.0, 'compression_ratio': 0.5},
    }
    
    # Risk Appetite 가중합 (Bekaert 기반)
    RISK_APPETITE_WEIGHTS = {'HYG_LQD': 0.4, 'XLY_XLP': 0.3, 'IWM_SPY': 0.3}
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리 (선택적)
        """
        # 하위 모듈 초기화
        self.ra_uncertainty = RiskAppetiteUncertaintyIndex(lookback=20)
        self.regime_detector = EnhancedRegimeDetector(short_ma=20, long_ma=120)
        self.spillover_network = SpilloverNetwork(lookback=20)
        self.crypto_sentiment = CryptoSentimentBlock(lookback=20, correlation_window=20)
        
        # 설정 로드
        self.config = config or {}
        self.path_weights = self.BASE_PATH_WEIGHTS.copy()
    
    def collect_required_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        필요한 데이터가 모두 있는지 확인하고 정리
        
        필수 티커:
        - SPY, QQQ, TLT, GLD, VIX (기본)
        - HYG, LQD, XLY, XLP, IWM, XLF (RA/Spillover)
        - BTC-USD (Crypto)
        - SMH, NVDA, DXY (추가 경로)
        
        Returns:
            정리된 데이터 딕셔너리 및 누락된 티커 목록
        """
        required_tickers = [
            'SPY', 'QQQ', 'TLT', 'GLD', '^VIX', 'VIX',
            'HYG', 'LQD', 'XLY', 'XLP', 'IWM', 'XLF',
            'BTC-USD', 'SMH', 'NVDA', 'DXY', 'DX-Y.NYB', 'EEM'
        ]
        
        collected = {}
        missing = []
        
        for ticker in required_tickers:
            # 티커명 변형 시도
            data = market_data.get(ticker)
            if data is None:
                # 대체 티커 시도
                alt_tickers = {
                    '^VIX': 'VIX',
                    'DX-Y.NYB': 'DXY',
                }
                alt_ticker = alt_tickers.get(ticker)
                if alt_ticker:
                    data = market_data.get(alt_ticker)
            
            if data is not None and not data.empty:
                collected[ticker] = data
            else:
                missing.append(ticker)
        
        return {
            'data': collected,
            'missing': missing
        }
    
    def run_submodules(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        4개 하위 모듈 순차 실행
        
        실행 순서:
        1. Regime Detector (임계값 결정에 필요)
        2. Risk Appetite & Uncertainty
        3. Spillover Network (레짐 정보 활용)
        4. Crypto Sentiment
        
        Returns:
            Dict with all submodule results
        """
        results = {}
        
        # 1. Regime Detector (먼저 실행 - 다른 모듈에서 레짐 정보 필요)
        spy_data = market_data.get('SPY')
        vix_data = market_data.get('^VIX')
        if vix_data is None or (hasattr(vix_data, 'empty') and vix_data.empty):
            vix_data = market_data.get('VIX')
        
        if spy_data is not None and vix_data is not None:
            if hasattr(spy_data, 'empty') and spy_data.empty:
                spy_data = None
            if hasattr(vix_data, 'empty') and vix_data.empty:
                vix_data = None
        
        if spy_data is not None and vix_data is not None:
            regime_result = self.regime_detector.analyze(spy_data, vix_data)
            results['regime'] = regime_result
            current_regime = regime_result.current_regime
        else:
            current_regime = "TRANSITION"
            # 기본 RegimeResult 생성 (에러 방지)
            results['regime'] = None
        
        # 2. Risk Appetite & Uncertainty
        try:
            ra_result = self.ra_uncertainty.analyze(market_data)
            results['risk_appetite'] = ra_result
        except Exception as e:
            print(f"Warning: Risk Appetite analysis failed: {e}")
            results['risk_appetite'] = None
        
        # 3. Spillover Network (레짐 정보 활용)
        try:
            spillover_result = self.spillover_network.analyze(market_data, current_regime)
            results['spillover'] = spillover_result
        except Exception as e:
            print(f"Warning: Spillover analysis failed: {e}")
            results['spillover'] = None
        
        # 4. Crypto Sentiment
        btc_data = market_data.get('BTC-USD')
        spy_data = market_data.get('SPY')
        gld_data = market_data.get('GLD')
        
        if btc_data is not None and spy_data is not None and gld_data is not None:
            try:
                crypto_result = self.crypto_sentiment.analyze(btc_data, spy_data, gld_data)
                results['crypto'] = crypto_result
            except Exception as e:
                print(f"Warning: Crypto sentiment analysis failed: {e}")
                results['crypto'] = None
        else:
            results['crypto'] = None
        
        return results
    
    def calculate_path_contributions(
        self, 
        submodule_results: Dict,
        regime: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        경로별 위험 기여도 계산
        
        로직:
        1. 각 경로별 원시 점수(raw score) 계산
        2. 각 경로별 점수를 0-100 범위로 클리핑
        3. 시각화용 분포(path_distribution) 별도 계산 (100% 정규화)
        
        경로별 점수 산출:
        - liquidity: TLT 신호 + DXY 신호 + 수익률곡선
        - concentration: QQQ/SPY + RSP/SPY + NVDA/SMH
        - credit: HYG/LQD + XLF 신호
        - volatility: VIX 레벨 + 불확실성 지수
        - rotation: XLY/XLP + IWM/SPY
        - crypto: CryptoSentimentResult의 risk_contribution
        
        Returns:
            Tuple[path_contributions (raw scores), path_distribution (100% 정규화)]
        """
        contributions = {}
        
        # 1. Liquidity 경로 (유동성/금리)
        liquidity_score = 0.0
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            # TLT 관련 경로 찾기
            for edge in spillover.active_paths:
                if edge.source in ['TLT', 'DXY'] and edge.category == 'liquidity':
                    liquidity_score += edge.signal_strength * 0.5
        
        # Risk Appetite에서 불확실성 점수 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 불확실성이 높으면 유동성 경로 위험 증가
            liquidity_score += ra.uncertainty_score * 0.3
        
        # 0-100 범위로 클리핑
        contributions['liquidity'] = max(0.0, min(100.0, liquidity_score))
        
        # 2. Concentration 경로 (AI/빅테크 집중) - 개선
        concentration_score = 0.0
        
        # Spillover 경로에서 집중도 신호 확인
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'concentration':
                    concentration_score += edge.signal_strength * 0.6
        
        # 직접 계산 추가: Risk Appetite의 components 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            components = ra.components
            
            # HYG/LQD zscore가 높으면 신용 선호 (집중 위험 증가)
            hyg_lqd_z = abs(components.get('hyg_lqd_zscore', 0))
            if hyg_lqd_z > 1.0:  # 임계값 완화: 1.5 -> 1.0
                concentration_score += hyg_lqd_z * 6  # 가중치 조정
            
            # XLY/XLP zscore로 경기민감주 선호도 확인
            xly_xlp_z = components.get('xly_xlp_zscore', 0)
            if xly_xlp_z > 1.0:  # 양수일 때만 성장주 집중 (임계값 완화)
                concentration_score += xly_xlp_z * 5
            
            # 상관관계 분산이 낮으면 동조화 (집중 위험 증가)
            corr_var = components.get('corr_variance_score', 50)
            if corr_var < 30:  # 낮은 분산 = 높은 상관 = 동조화
                concentration_score += (30 - corr_var) * 0.5
        
        # 0-100 범위로 클리핑
        contributions['concentration'] = max(0.0, min(100.0, concentration_score))
        
        # 3. Credit 경로 (신용 스트레스)
        credit_score = 0.0
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'credit':
                    credit_score += edge.signal_strength * 0.5
        
        # Risk Appetite에서 리스크 선호도 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 리스크 선호도가 낮으면 신용 경로 위험 증가
            credit_score += (100 - ra.risk_appetite_score) * 0.3
        
        # 0-100 범위로 클리핑
        contributions['credit'] = max(0.0, min(100.0, credit_score))
        
        # 4. Volatility 경로 (변동성/공포)
        volatility_score = 0.0
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 불확실성 점수 활용
            volatility_score = ra.uncertainty_score * 0.5
        
        # Spillover에서 VIX 관련 경로
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'volatility':
                    volatility_score += edge.signal_strength * 0.5
        
        # 0-100 범위로 클리핑
        contributions['volatility'] = max(0.0, min(100.0, volatility_score))
        
        # 5. Rotation 경로 (섹터 로테이션) - 개선
        rotation_score = 0.0
        
        # Spillover 경로에서 로테이션 신호 확인
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'rotation':
                    rotation_score += edge.signal_strength * 0.5
        
        # 직접 계산 추가: Risk Appetite의 components 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            components = ra.components
            
            # IWM/SPY zscore로 소형주 로테이션 확인 (임계값 완화)
            iwm_spy_z = abs(components.get('iwm_spy_zscore', 0))
            if iwm_spy_z > 0.8:  # 임계값 완화: 1.0 -> 0.8
                rotation_score += iwm_spy_z * 10
            
            # XLY/XLP zscore로 섹터 로테이션 확인 (임계값 완화)
            xly_xlp_z = abs(components.get('xly_xlp_zscore', 0))
            if xly_xlp_z > 0.8:  # 임계값 완화: 1.0 -> 0.8
                rotation_score += xly_xlp_z * 8
            
            # VRP(Variance Risk Premium)가 높으면 로테이션 신호
            vrp_score = components.get('vrp_score', 50)
            if vrp_score > 60:
                rotation_score += (vrp_score - 60) * 0.3
        
        # 0-100 범위로 클리핑
        contributions['rotation'] = max(0.0, min(100.0, rotation_score))
        
        # 6. Crypto 경로
        crypto_score = 0.0
        if submodule_results.get('crypto'):
            crypto = submodule_results['crypto']
            # 위험 기여도 활용 (0-0.2 범위를 0-40으로 변환, 상한선 설정)
            crypto_score = min(crypto.risk_contribution * 200, 40)  # 최대 40점
            # 심리 점수도 반영 (극단적 상태일 때)
            if crypto.sentiment_level in ['EXTREME_FEAR', 'EXTREME_GREED']:
                crypto_score += 20
        
        # 0-100 범위로 클리핑
        contributions['crypto'] = max(0.0, min(100.0, crypto_score))
        
        # 시각화용 분포 계산 (100% 정규화)
        path_distribution = {}
        total_raw = sum(contributions.values())
        if total_raw > 0:
            for path_name, score in contributions.items():
                path_distribution[path_name] = (score / total_raw) * 100.0
        else:
            # 모든 경로가 0인 경우 균등 분배
            num_paths = len(contributions)
            if num_paths > 0:
                equal_share = 100.0 / num_paths
                path_distribution = {path_name: equal_share for path_name in contributions.keys()}
            else:
                path_distribution = {}
        
        return contributions, path_distribution
    
    def adjust_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        레짐에 따라 경로 가중치 조정
        
        BULL: 집중도 경로 가중치 증가 (균열 감지 중요)
        BEAR: 신용, 유동성 경로 가중치 증가
        CRISIS: 변동성 경로 가중치 증가
        
        Returns:
            조정된 가중치 딕셔너리
        """
        weights = self.BASE_PATH_WEIGHTS.copy()
        
        if regime == "BULL":
            # 집중도 경로 가중치 증가
            weights['concentration'] *= 1.3
            weights['liquidity'] *= 0.9
            weights['credit'] *= 0.8
        elif regime == "BEAR":
            # 신용, 유동성 경로 가중치 증가
            weights['credit'] *= 1.4
            weights['liquidity'] *= 1.2
            weights['volatility'] *= 1.1
        elif regime == "CRISIS":
            # 변동성 경로 가중치 증가
            weights['volatility'] *= 1.5
            weights['credit'] *= 1.3
            weights['liquidity'] *= 1.2
        # TRANSITION은 기본 가중치 유지
        
        # 합계를 1.0으로 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_total_risk(
        self, 
        path_contributions: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        전체 위험도 계산 (가중평균 방식)
        
        각 경로의 raw score에 BASE_PATH_WEIGHTS 가중치를 곱한 가중평균을 계산합니다.
        이는 포트폴리오 위험 측정의 표준 방법론입니다.
        
        Returns:
            Tuple[score (0-100), level]
        
        Level 정의:
        - 0-25: LOW
        - 25-50: MEDIUM
        - 50-75: HIGH
        - 75-100: CRITICAL
        """
        # 가중평균 계산: 각 경로의 raw score × 기본 가중치
        weighted_sum = 0.0
        total_weight = 0.0
        
        for path_name, raw_score in path_contributions.items():
            weight = self.BASE_PATH_WEIGHTS.get(path_name, 0.0)
            weighted_sum += raw_score * weight
            total_weight += weight
        
        # 가중평균 계산
        if total_weight > 0:
            total_score = weighted_sum / total_weight
        else:
            total_score = 0.0
        
        # 0-100 범위로 클리핑
        total_score = max(0.0, min(100.0, total_score))
        
        if total_score < 25:
            level = "LOW"
        elif total_score < 50:
            level = "MEDIUM"
        elif total_score < 75:
            level = "HIGH"
        else:
            level = "CRITICAL"
        
        return total_score, level
    
    def generate_warnings(self, submodule_results: Dict) -> List[str]:
        """
        활성화된 경고 목록 생성
        
        경고 예시:
        - "TLT 급락 중: 금리 상승 압력, QQQ 영향 예상 (3일 시차)"
        - "BTC 선행 하락 감지: RISK_OFF_WARNING"
        - "레짐 전환 징후: BULL → TRANSITION (확률 65%)"
        - "VIX-실현변동성 괴리 확대: 불확실성 프리미엄 증가"
        
        Returns:
            List of warning strings
        """
        warnings = []
        
        # 1. Spillover 경로 경고
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.signal_strength >= 70:
                    direction = "하락" if edge.expected_target_move == "DOWN" else "상승"
                    warnings.append(
                        f"{edge.source} 급변 중: {edge.theory_note}, "
                        f"{edge.target} {direction} 압력 예상 ({edge.adjusted_lag}일 시차)"
                    )
        
        # 2. 레짐 전환 경고
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            if regime.transition_probability >= 50:
                warnings.append(
                    f"레짐 전환 징후: {regime.current_regime} → "
                    f"{regime.transition_direction} (확률 {regime.transition_probability:.0f}%)"
                )
        
        # 3. Crypto 선행지표 경고
        if submodule_results.get('crypto'):
            crypto = submodule_results['crypto']
            if crypto.is_leading_indicator and crypto.leading_signal:
                if crypto.leading_signal == "RISK_OFF_WARNING":
                    warnings.append("BTC 선행 하락 감지: RISK_OFF_WARNING - 주식 시장 하락 선행 가능")
                elif crypto.leading_signal == "RISK_ON_SIGNAL":
                    warnings.append("BTC 선행 상승 감지: RISK_ON_SIGNAL - 주식 시장 상승 선행 가능")
        
        # 4. 불확실성 프리미엄 경고
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            if 'vix_realized_gap' in ra.components:
                gap = ra.components['vix_realized_gap']
                if gap > 10:
                    warnings.append(
                        f"VIX-실현변동성 괴리 확대 ({gap:.1f}%p): 불확실성 프리미엄 증가"
                    )
        
        # 5. 위기 상태 경고
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            if regime.current_regime == "CRISIS":
                warnings.append("🚨 위기 레짐 감지: 유동성 확보 및 방어적 포지션 권장")
        
        return warnings
    
    def generate_interpretation(
        self, 
        total_risk: float,
        risk_level: str,
        path_contributions: Dict,
        submodule_results: Dict
    ) -> str:
        """
        종합 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        # 기본 위험도 설명
        interpretation = f"현재 시장 위험도는 {total_risk:.1f}% ({risk_level}) 수준입니다. "
        
        # 주요 위험 경로
        sorted_paths = sorted(
            path_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_paths:
            top_path = sorted_paths[0]
            path_names = {
                'liquidity': '유동성/금리',
                'concentration': 'AI/빅테크 집중',
                'credit': '신용 스트레스',
                'volatility': '변동성/공포',
                'rotation': '섹터 로테이션',
                'crypto': '암호화폐'
            }
            top_path_name = path_names.get(top_path[0], top_path[0])
            interpretation += f"주요 위험 요인은 {top_path_name} 경로({top_path[1]:.1f}%)입니다. "
        
        # 레짐 정보
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            interpretation += f"현재 {regime.current_regime} 레짐이며, "
            if regime.transition_probability >= 50:
                interpretation += f"레짐 전환 확률이 {regime.transition_probability:.0f}%로 상승 중입니다. "
            else:
                interpretation += f"레짐 안정도는 {regime.regime_confidence:.0f}%입니다. "
        
        # Risk Appetite 정보
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            interpretation += (
                f"리스크 선호도는 {ra.risk_appetite_score:.0f}점({ra.risk_appetite_level}), "
                f"불확실성은 {ra.uncertainty_score:.0f}점({ra.uncertainty_level})입니다. "
            )
        
        # Spillover 정보
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            if spillover.active_paths:
                interpretation += (
                    f"활성화된 충격 전이 경로는 {len(spillover.active_paths)}개이며, "
                    f"주요 위험 진원지는 {spillover.primary_risk_source}입니다. "
                )
        
        return interpretation
    
    def analyze(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> CriticalPathResult:
        """
        전체 분석 파이프라인 실행
        
        1. 데이터 검증 및 정리
        2. 하위 모듈 실행
        3. 경로별 기여도 계산
        4. 전체 위험도 산출
        5. 경고 및 해석 생성
        6. 결과 객체 반환
        
        Args:
            market_data: 티커별 가격 데이터 딕셔너리
        
        Returns:
            CriticalPathResult 객체
        """
        # 1. 데이터 검증
        data_check = self.collect_required_data(market_data)
        if len(data_check['missing']) > 5:
            print(f"Warning: {len(data_check['missing'])} required tickers missing")
        
        # 2. 하위 모듈 실행
        submodule_results = self.run_submodules(market_data)
        
        # 레짐 정보 추출
        if submodule_results.get('regime'):
            current_regime = submodule_results['regime'].current_regime
            regime_confidence = submodule_results['regime'].regime_confidence
            transition_prob = submodule_results['regime'].transition_probability
        else:
            current_regime = "TRANSITION"
            regime_confidence = 50.0
            transition_prob = 0.0
        
        # 3. 경로별 기여도 계산
        path_contributions, path_distribution = self.calculate_path_contributions(
            submodule_results,
            current_regime
        )
        
        # 4. 전체 위험도 산출
        total_risk, risk_level = self.calculate_total_risk(path_contributions)
        
        # 5. 주요 위험 경로 식별
        if path_contributions:
            primary_risk_path = max(path_contributions.items(), key=lambda x: x[1])[0]
        else:
            primary_risk_path = "NONE"
        
        # 6. 경고 생성
        active_warnings = self.generate_warnings(submodule_results)
        
        # 7. 해석 생성
        interpretation = self.generate_interpretation(
            total_risk,
            risk_level,
            path_contributions,
            submodule_results
        )
        
        # 8. 결과 객체 생성 (None 값 처리)
        risk_appetite_result = submodule_results.get('risk_appetite')
        regime_result = submodule_results.get('regime')
        spillover_result = submodule_results.get('spillover')
        crypto_result = submodule_results.get('crypto')
        
        # 기본값 생성 (None인 경우)
        if risk_appetite_result is None:
            risk_appetite_result = RiskAppetiteUncertaintyResult(
                timestamp=datetime.now().isoformat(),
                risk_appetite_score=50.0,
                uncertainty_score=50.0,
                risk_appetite_level="MEDIUM",
                uncertainty_level="MEDIUM",
                market_state="MIXED",
                components={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if regime_result is None:
            regime_result = RegimeResult(
                timestamp=datetime.now().isoformat(),
                current_regime="TRANSITION",
                regime_confidence=50.0,
                transition_probability=0.0,
                transition_direction="STABLE",
                thresholds={},
                ma_status={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if spillover_result is None:
            spillover_result = SpilloverResult(
                timestamp=datetime.now().isoformat(),
                active_paths=[],
                risk_score=0.0,
                primary_risk_source="NONE",
                expected_impacts={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if crypto_result is None:
            crypto_result = CryptoSentimentResult(
                timestamp=datetime.now().isoformat(),
                sentiment_score=50.0,
                sentiment_level="NEUTRAL",
                btc_spy_correlation=0.0,
                correlation_regime="DECOUPLED",
                is_leading_indicator=False,
                leading_signal=None,
                risk_contribution=0.05,
                components={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        return CriticalPathResult(
            timestamp=datetime.now().isoformat(),
            total_risk_score=total_risk,
            risk_level=risk_level,
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            transition_probability=transition_prob,
            path_contributions=path_contributions,
            path_distribution=path_distribution,
            risk_appetite_result=risk_appetite_result,
            regime_result=regime_result,
            spillover_result=spillover_result,
            crypto_result=crypto_result,
            primary_risk_path=primary_risk_path,
            active_warnings=active_warnings,
            interpretation=interpretation
        )


# ============================================================
# 통합 함수 (main.py에서 호출)
# ============================================================

def run_critical_path_analysis(
    market_data: Dict[str, pd.DataFrame]
) -> CriticalPathResult:
    """
    Critical Path 분석 실행 (편의 함수)
    
    Usage:
        from critical_path_analyzer import run_critical_path_analysis
        result = run_critical_path_analysis(market_data)
        print(f"Total Risk: {result.total_risk_score}%")
    
    Args:
        market_data: 티커별 가격 데이터 딕셔너리
    
    Returns:
        CriticalPathResult 객체
    """
    aggregator = CriticalPathAggregator()
    return aggregator.analyze(market_data)