#!/usr/bin/env python3
"""
Market Regime Detector
======================
시장 레짐(국면) 탐지 시스템

레짐 분류:
1. BULL_LOW_VOL: 강세장 + 낮은 변동성 (최적 투자 환경)
2. BULL_HIGH_VOL: 강세장 + 높은 변동성 (조정 가능성)
3. BEAR_LOW_VOL: 약세장 + 낮은 변동성 (바닥 탐색)
4. BEAR_HIGH_VOL: 약세장 + 높은 변동성 (위기 국면)
5. TRANSITION: 전환기 (불확실)

방법론:
- 이동평균 기반 추세 판단 (50일, 200일)
- 변동성 레짐 (VIX, 실현 변동성)
- HMM 스타일 확률 기반 분류

참고: Hamilton(1989) Markov Switching Model의 간소화 버전
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from core.database import DatabaseManager


# ============================================================================
# Enums & Constants
# ============================================================================

class MarketRegime(str, Enum):
    """시장 레짐"""
    BULL_LOW_VOL = "Bull (Low Vol)"        # 강세 + 저변동
    BULL_HIGH_VOL = "Bull (High Vol)"      # 강세 + 고변동
    BEAR_LOW_VOL = "Bear (Low Vol)"        # 약세 + 저변동
    BEAR_HIGH_VOL = "Bear (High Vol)"      # 약세 + 고변동
    TRANSITION = "Transition"               # 전환기


class TrendState(str, Enum):
    """추세 상태"""
    STRONG_UP = "Strong Uptrend"
    WEAK_UP = "Weak Uptrend"
    NEUTRAL = "Neutral"
    WEAK_DOWN = "Weak Downtrend"
    STRONG_DOWN = "Strong Downtrend"


class VolatilityState(str, Enum):
    """변동성 상태"""
    VERY_LOW = "Very Low"       # VIX < 12
    LOW = "Low"                 # 12 <= VIX < 16
    NORMAL = "Normal"           # 16 <= VIX < 22
    HIGH = "High"               # 22 <= VIX < 30
    EXTREME = "Extreme"         # VIX >= 30


# 레짐별 특성
REGIME_CHARACTERISTICS = {
    MarketRegime.BULL_LOW_VOL: {
        "description": "최적의 투자 환경. 리스크 자산 선호",
        "strategy": "주식 비중 확대, 성장주/소형주 선호",
        "risk_appetite": "HIGH",
    },
    MarketRegime.BULL_HIGH_VOL: {
        "description": "상승세지만 조정 가능성. 차익실현 고려",
        "strategy": "분할 매도 준비, 헤지 비중 확대",
        "risk_appetite": "MEDIUM",
    },
    MarketRegime.BEAR_LOW_VOL: {
        "description": "하락세 바닥 탐색. 저점 매수 기회 탐색",
        "strategy": "분할 매수, 방어주/배당주 선호",
        "risk_appetite": "LOW_SELECTIVE",
    },
    MarketRegime.BEAR_HIGH_VOL: {
        "description": "위기 국면. 자산 보존 최우선",
        "strategy": "현금 비중 극대화, 안전자산 선호",
        "risk_appetite": "VERY_LOW",
    },
    MarketRegime.TRANSITION: {
        "description": "레짐 전환 중. 관망 필요",
        "strategy": "포지션 축소, 방향성 확인 후 진입",
        "risk_appetite": "NEUTRAL",
    },
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RegimeIndicators:
    """레짐 판단 지표"""
    # 가격 기반
    price_above_ma50: bool = False
    price_above_ma200: bool = False
    ma50_above_ma200: bool = False
    distance_from_high: float = 0.0      # 52주 고점 대비 %

    # 모멘텀
    rsi_14: float = 50.0
    momentum_20d: float = 0.0            # 20일 수익률 %

    # 변동성
    vix: float = 20.0
    vix_percentile: float = 50.0
    realized_vol_20d: float = 15.0

    # 추세 강도
    adx: float = 25.0                    # Average Directional Index

    def to_dict(self) -> Dict:
        return asdict(self)


from sklearn.mixture import GaussianMixture

@dataclass
class RegimeResult:
    """레짐 탐지 결과"""
    timestamp: str

    # 현재 레짐
    regime: MarketRegime
    confidence: float                    # 0-100%

    # 구성 요소
    trend_state: TrendState
    volatility_state: VolatilityState

    # 지표
    indicators: RegimeIndicators

    # 레짐 특성
    description: str
    strategy: str
    risk_appetite: str

    # 전환 확률
    transition_probs: Dict[str, float] = field(default_factory=dict)
    
    # GMM Probabilities
    gmm_probabilities: Dict[str, float] = field(default_factory=lambda: {"Bull": 0.33, "Neutral": 0.34, "Bear": 0.33})

    # 이력
    prev_regime: Optional[str] = None
    days_in_regime: int = 0

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'regime': self.regime.value,
            'confidence': round(self.confidence, 1),
            'trend_state': self.trend_state.value,
            'volatility_state': self.volatility_state.value,
            'indicators': self.indicators.to_dict(),
            'description': self.description,
            'strategy': self.strategy,
            'risk_appetite': self.risk_appetite,
            'transition_probs': {k: round(v, 2) for k, v in self.transition_probs.items()},
            'gmm_probabilities': {k: round(v, 3) for k, v in self.gmm_probabilities.items()},
            'prev_regime': self.prev_regime,
            'days_in_regime': self.days_in_regime,
        }


# ============================================================================
# Regime Detector
# ============================================================================

class RegimeDetector:
    """
    시장 레짐 탐지기

    사용법:
        detector = RegimeDetector()
        result = detector.detect()
        detector.print_report(result)
    """

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker
        self._cache = {}

    def _fetch_data(self, period: str = "1y") -> pd.DataFrame:
        """가격 데이터 수집"""
        if self.ticker in self._cache:
            return self._cache[self.ticker]

        try:
            df = yf.download(self.ticker, period=period, progress=False)
            self._cache[self.ticker] = df
            return df
        except Exception as e:
            print(f"Error fetching {self.ticker}: {e}")
            return pd.DataFrame()

    def _fetch_vix(self, period: str = "1y") -> pd.Series:
        """VIX 데이터 수집"""
        try:
            df = yf.download("^VIX", period=period, progress=False)
            return df['Close']
        except Exception as e:
            print(f"Error fetching VIX: {e}")
            return pd.Series()

    def _calculate_indicators(self, df: pd.DataFrame,
                              vix: pd.Series) -> RegimeIndicators:
        """지표 계산"""
        if df.empty:
            return RegimeIndicators()

        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Handle potential multi-level columns from yfinance
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]

        # 이동평균
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Safe scalar extraction
        def to_scalar(val):
            """Convert Series/array to scalar safely"""
            if hasattr(val, 'item'):
                return float(val.item())
            elif hasattr(val, 'iloc'):
                return float(val.iloc[0]) if len(val) > 0 else 0.0
            return float(val)

        current_price = to_scalar(close.iloc[-1])
        current_ma50 = to_scalar(ma50.iloc[-1]) if len(ma50) > 50 else current_price
        current_ma200 = to_scalar(ma200.iloc[-1]) if len(ma200) > 200 else current_price

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = to_scalar(rsi.iloc[-1]) if len(rsi) > 14 else 50

        # 52주 고점 대비
        high_52w = to_scalar(high.tail(252).max())
        distance_from_high = ((current_price / high_52w) - 1) * 100

        # 20일 모멘텀
        if len(close) >= 20:
            momentum_20d = ((current_price / to_scalar(close.iloc[-20])) - 1) * 100
        else:
            momentum_20d = 0.0

        # 실현 변동성 (20일)
        returns = close.pct_change()
        realized_vol = to_scalar(returns.tail(20).std()) * np.sqrt(252) * 100

        # VIX
        try:
            current_vix = float(vix.iloc[-1]) if len(vix) > 0 else 20.0
            if len(vix) > 20:
                vix_pct = float((vix < current_vix).mean())
                vix_percentile = vix_pct * 100
            else:
                vix_percentile = 50.0
        except Exception:
            current_vix = 20.0
            vix_percentile = 50.0

        # ADX (Average Directional Index) - 간소화
        # 14일 기준 추세 강도
        try:
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            dm_plus = (high - high.shift(1)).clip(lower=0)
            dm_minus = (low.shift(1) - low).clip(lower=0)

            di_plus = 100 * (dm_plus.rolling(14).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(14).mean() / atr)
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(14).mean()

            # 마지막 값 추출 (NaN 처리)
            adx_values = adx.dropna()
            current_adx = float(adx_values.iloc[-1]) if len(adx_values) > 0 else 25.0
        except Exception:
            current_adx = 25.0

        return RegimeIndicators(
            price_above_ma50=current_price > current_ma50,
            price_above_ma200=current_price > current_ma200,
            ma50_above_ma200=current_ma50 > current_ma200,
            distance_from_high=round(distance_from_high, 2),
            rsi_14=round(current_rsi, 1),
            momentum_20d=round(momentum_20d, 2),
            vix=round(current_vix, 2),
            vix_percentile=round(vix_percentile, 1),
            realized_vol_20d=round(realized_vol, 2),
            adx=round(current_adx, 1),
        )

    def _determine_trend(self, ind: RegimeIndicators) -> TrendState:
        """추세 상태 결정"""
        score = 0

        # 가격 위치
        if ind.price_above_ma50:
            score += 1
        if ind.price_above_ma200:
            score += 1
        if ind.ma50_above_ma200:
            score += 1

        # 고점 대비 거리
        if ind.distance_from_high > -5:
            score += 1
        elif ind.distance_from_high < -15:
            score -= 2

        # 모멘텀
        if ind.momentum_20d > 5:
            score += 1
        elif ind.momentum_20d < -5:
            score -= 1

        # 추세 강도 (ADX)
        is_strong_trend = ind.adx > 25

        # 판정
        if score >= 4:
            return TrendState.STRONG_UP if is_strong_trend else TrendState.WEAK_UP
        elif score >= 2:
            return TrendState.WEAK_UP
        elif score <= -2:
            return TrendState.STRONG_DOWN if is_strong_trend else TrendState.WEAK_DOWN
        elif score < 0:
            return TrendState.WEAK_DOWN
        else:
            return TrendState.NEUTRAL

    def _determine_volatility(self, ind: RegimeIndicators) -> VolatilityState:
        """변동성 상태 결정"""
        vix = ind.vix

        if vix < 12:
            return VolatilityState.VERY_LOW
        elif vix < 16:
            return VolatilityState.LOW
        elif vix < 22:
            return VolatilityState.NORMAL
        elif vix < 30:
            return VolatilityState.HIGH
        else:
            return VolatilityState.EXTREME

    def _determine_regime(self, trend: TrendState,
                          vol: VolatilityState) -> Tuple[MarketRegime, float]:
        """레짐 및 신뢰도 결정"""
        # Bull 조건
        is_bull = trend in [TrendState.STRONG_UP, TrendState.WEAK_UP]

        # Bear 조건
        is_bear = trend in [TrendState.STRONG_DOWN, TrendState.WEAK_DOWN]

        # Low Vol 조건
        is_low_vol = vol in [VolatilityState.VERY_LOW, VolatilityState.LOW, VolatilityState.NORMAL]

        # High Vol 조건
        is_high_vol = vol in [VolatilityState.HIGH, VolatilityState.EXTREME]

        # 레짐 결정
        if is_bull and is_low_vol:
            regime = MarketRegime.BULL_LOW_VOL
            confidence = 90 if trend == TrendState.STRONG_UP else 75
        elif is_bull and is_high_vol:
            regime = MarketRegime.BULL_HIGH_VOL
            confidence = 80 if trend == TrendState.STRONG_UP else 65
        elif is_bear and is_low_vol:
            regime = MarketRegime.BEAR_LOW_VOL
            confidence = 85 if trend == TrendState.STRONG_DOWN else 70
        elif is_bear and is_high_vol:
            regime = MarketRegime.BEAR_HIGH_VOL
            confidence = 90 if vol == VolatilityState.EXTREME else 75
        else:
            regime = MarketRegime.TRANSITION
            confidence = 50

        return regime, confidence

    def _calculate_transition_probs(self, regime: MarketRegime,
                                    ind: RegimeIndicators) -> Dict[str, float]:
        """전환 확률 계산 (간소화)"""
        probs = {}

        # 현재 레짐 유지 확률 기본값
        stay_prob = 70.0

        # RSI 기반 조정
        if ind.rsi_14 > 70:
            stay_prob -= 15 if regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL] else 0
        elif ind.rsi_14 < 30:
            stay_prob -= 15 if regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL] else 0

        # VIX 백분위 기반 조정
        if ind.vix_percentile > 80:
            stay_prob -= 10
        elif ind.vix_percentile < 20:
            stay_prob += 5

        probs['stay'] = max(30, min(95, stay_prob))
        probs['transition'] = 100 - probs['stay']

        return probs

    def _run_gmm(self, df: pd.DataFrame, vix: pd.Series) -> Dict[str, float]:
        """GMM 기반 레짐 확률 계산"""
        try:
            # 데이터 준비
            if df.empty or vix.empty:
                return {"Bull": 0.33, "Neutral": 0.34, "Bear": 0.33}
            
            data = pd.DataFrame()
            data['Returns'] = df['Close'].pct_change()
            
            # VIX 인덱스 맞추기
            data['VIX'] = vix.reindex(data.index).fillna(method='ffill')
            
            data = data.dropna()
            
            if len(data) < 50: # 데이터 부족
                return {"Bull": 0.33, "Neutral": 0.34, "Bear": 0.33}

            # 학습 데이터 (Returns, VIX)
            X = data[['Returns', 'VIX']].values
            
            # GMM 모델 학습 (3개 컴포넌트)
            gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
            gmm.fit(X)
            
            # 컴포넌트 해석 (VIX 기준으로 정렬: Low VIX -> Bull, High VIX -> Bear)
            # means_[:, 1] 은 VIX의 평균
            vix_means = gmm.means_[:, 1]
            sorted_indices = np.argsort(vix_means) # 오름차순 (Low VIX first)
            
            # 매핑: 0 -> Bull, 1 -> Neutral, 2 -> Bear (by VIX)
            component_map = {
                sorted_indices[0]: "Bull",
                sorted_indices[1]: "Neutral",
                sorted_indices[2]: "Bear"
            }
            
            # 현재 상태 확률 예측
            current_X = X[-1].reshape(1, -1)
            probs = gmm.predict_proba(current_X)[0]
            
            result = {}
            for i, prob in enumerate(probs):
                label = component_map[i]
                result[label] = float(prob)
                
            return result

        except Exception as e:
            print(f"GMM Error: {e}")
            return {"Bull": 0.33, "Neutral": 0.34, "Bear": 0.33}

    def detect(self) -> RegimeResult:
        """레짐 탐지 실행"""
        print(f"Detecting market regime for {self.ticker}...")

        # 데이터 수집
        df = self._fetch_data()
        vix = self._fetch_vix()

        # 지표 계산
        indicators = self._calculate_indicators(df, vix)
        
        # GMM 분석
        gmm_probs = self._run_gmm(df, vix)

        # 추세 판단
        trend = self._determine_trend(indicators)
        print(f"  Trend: {trend.value}")

        # 변동성 판단
        volatility = self._determine_volatility(indicators)
        print(f"  Volatility: {volatility.value}")

        # 레짐 결정
        regime, confidence = self._determine_regime(trend, volatility)
        print(f"  Regime: {regime.value} ({confidence:.0f}%)")

        # 전환 확률
        transition_probs = self._calculate_transition_probs(regime, indicators)

        # 레짐 특성
        chars = REGIME_CHARACTERISTICS[regime]

        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            confidence=confidence,
            trend_state=trend,
            volatility_state=volatility,
            indicators=indicators,
            description=chars['description'],
            strategy=chars['strategy'],
            risk_appetite=chars['risk_appetite'],
            transition_probs=transition_probs,
            gmm_probabilities=gmm_probs,
        )

    def save_to_db(self, result: RegimeResult,
                   db: DatabaseManager = None) -> bool:
        """DB에 저장"""
        if db is None:
            db = DatabaseManager()

        today = datetime.now().strftime("%Y-%m-%d")

        try:
            # market_regime 테이블에 저장
            regime_data = {
                'trend': result.trend_state.value,
                'volatility_level': result.volatility_state.value,
                'vix_estimate': result.indicators.vix,
                'risk_appetite_score': 100 if result.risk_appetite == 'HIGH' else
                                       75 if result.risk_appetite == 'MEDIUM' else
                                       50 if result.risk_appetite == 'LOW_SELECTIVE' else
                                       25 if result.risk_appetite == 'VERY_LOW' else 50,
                'confidence': result.confidence,
                'regime_name': result.regime.value,
                'description': result.description,
            }
            db.save_market_regime(regime_data, today)

            # ETF Analysis에도 저장
            db.save_etf_analysis('regime_detection', result.to_dict(), today)

            db.log_analysis('regime_detection', 'SUCCESS', 1, today)
            return True

        except Exception as e:
            print(f"Error saving to DB: {e}")
            return False

    def print_report(self, result: RegimeResult):
        """리포트 출력"""
        print("\n" + "=" * 60)
        print("MARKET REGIME REPORT")
        print(f"Generated: {result.timestamp[:19]}")
        print(f"Ticker: {self.ticker}")
        print("=" * 60)

        # 현재 레짐
        regime_emoji = {
            MarketRegime.BULL_LOW_VOL: "🟢",
            MarketRegime.BULL_HIGH_VOL: "🟡",
            MarketRegime.BEAR_LOW_VOL: "🟠",
            MarketRegime.BEAR_HIGH_VOL: "🔴",
            MarketRegime.TRANSITION: "⚪",
        }
        emoji = regime_emoji.get(result.regime, "⚪")

        print(f"\n{emoji} Current Regime: {result.regime.value}")
        print(f"   Confidence: {result.confidence:.0f}%")
        print(f"\n   {result.description}")

        # 추세 & 변동성
        print(f"\n[Market State]")
        print(f"  Trend:      {result.trend_state.value}")
        print(f"  Volatility: {result.volatility_state.value}")

        # 지표
        ind = result.indicators
        print(f"\n[Indicators]")
        print(f"  Price vs MA50:   {'Above' if ind.price_above_ma50 else 'Below'}")
        print(f"  Price vs MA200:  {'Above' if ind.price_above_ma200 else 'Below'}")
        print(f"  MA50 vs MA200:   {'Golden Cross' if ind.ma50_above_ma200 else 'Death Cross'}")
        print(f"  52W High Dist:   {ind.distance_from_high:+.1f}%")
        print(f"  RSI(14):         {ind.rsi_14:.1f}")
        print(f"  Momentum(20d):   {ind.momentum_20d:+.1f}%")
        print(f"  VIX:             {ind.vix:.1f} ({ind.vix_percentile:.0f}%ile)")
        print(f"  Realized Vol:    {ind.realized_vol_20d:.1f}%")
        print(f"  ADX:             {ind.adx:.1f}")

        # 전략
        print(f"\n[Strategy]")
        print(f"  Risk Appetite: {result.risk_appetite}")
        print(f"  {result.strategy}")

        # 전환 확률
        print(f"\n[Transition Probability]")
        print(f"  Stay in regime: {result.transition_probs.get('stay', 70):.0f}%")
        print(f"  Transition:     {result.transition_probs.get('transition', 30):.0f}%")

        print("\n" + "=" * 60)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Market Regime Detector Test")
    print("=" * 60)

    detector = RegimeDetector("SPY")
    result = detector.detect()
    detector.print_report(result)

    # DB 저장
    print("\n[Saving to Database]")
    db = DatabaseManager()
    if detector.save_to_db(result, db):
        print("  Saved successfully!")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
