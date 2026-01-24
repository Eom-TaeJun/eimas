"""
Information Flow Analyzer
==========================
정보 플로우 분석: 거래량 이상 탐지, Private Information Score, CAPM

경제학적 배경 (금융경제정리.docx):
- Causality vs. Correlation: 경제학은 인과관계
- 정보 플로우 → 거래량 변화 → 가격 변화
- 정보 우위자의 선제 행동 (가격 아님, 거래량으로 식별)

핵심 지표:
1. Abnormal Volume Detection
   - 거래량 이상 = 정보 주입 신호
   - MA(20) 대비 5배 이상 → 이상 거래

2. Private Information Score
   - (Buy Volume - Sell Volume) / Total Volume
   - > 0: 정보 우위 매수세
   - < 0: 정보 우위 매도세

3. CAPM Regression
   - E[R_i] = Alpha + Beta * E[R_m]
   - Alpha: 초과 수익 (정보 우위 프록시)
   - Beta: 시장 민감도

Author: EIMAS Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AbnormalVolumeResult:
    """거래량 이상 탐지 결과"""
    abnormal_dates: pd.DataFrame  # 이상 거래일 리스트
    total_abnormal_days: int  # 이상 거래일 수
    abnormal_ratio: float  # 이상 거래일 비율
    max_ratio: float  # 최대 거래량 비율
    interpretation: str  # 해석

    def to_dict(self) -> Dict[str, Any]:
        return {
            'abnormal_dates': self.abnormal_dates.to_dict('records') if not self.abnormal_dates.empty else [],
            'total_abnormal_days': self.total_abnormal_days,
            'abnormal_ratio': self.abnormal_ratio,
            'max_ratio': self.max_ratio,
            'interpretation': self.interpretation
        }


@dataclass
class PrivateInfoResult:
    """Private Information Score 결과"""
    info_score: pd.Series  # 시계열 정보 점수
    mean_score: float  # 평균 점수
    buy_pressure_days: int  # 매수 압력 일수
    sell_pressure_days: int  # 매도 압력 일수
    net_pressure: str  # 순 압력 (BUY/SELL/NEUTRAL)
    interpretation: str  # 해석

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_score': self.mean_score,
            'buy_pressure_days': self.buy_pressure_days,
            'sell_pressure_days': self.sell_pressure_days,
            'net_pressure': self.net_pressure,
            'interpretation': self.interpretation
        }


@dataclass
class CAPMResult:
    """CAPM 회귀 결과"""
    alpha: float  # 초과 수익 (정보 우위 프록시)
    beta: float  # 시장 민감도
    r_squared: float  # 결정계수
    n_observations: int  # 관측치 수
    alpha_interpretation: str  # Alpha 해석
    beta_interpretation: str  # Beta 해석

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'n_observations': self.n_observations,
            'alpha_interpretation': self.alpha_interpretation,
            'beta_interpretation': self.beta_interpretation
        }


# =============================================================================
# Information Flow Analyzer
# =============================================================================

class InformationFlowAnalyzer:
    """
    정보 플로우 분석기

    경제학적 의미:
    - 거래량 이상 = 정보 주입 (Informed Trading)
    - Private Information Score = 정보 비대칭 정량화
    - CAPM Alpha = 정보 우위로 인한 초과 수익
    """

    def __init__(self, ma_window: int = 20, threshold: float = 5.0):
        """
        Args:
            ma_window: 이동평균 윈도우 (기본: 20일)
            threshold: 거래량 이상 임계값 (기본: 5배)
        """
        self.ma_window = ma_window
        self.threshold = threshold

    def detect_abnormal_volume(self, volume: pd.Series) -> AbnormalVolumeResult:
        """
        거래량 이상 탐지

        Rule: volume[t] > MA(volume, 20) * 5

        경제학적 의미:
        - 평소보다 5배 이상 거래량 = 정보 주입 의심
        - Flash Crash (2010), Earnings Surprise 등

        Args:
            volume: 거래량 시계열

        Returns:
            AbnormalVolumeResult: 이상 거래 탐지 결과

        Example:
            >>> analyzer = InformationFlowAnalyzer()
            >>> result = analyzer.detect_abnormal_volume(spy_volume)
            >>> print(f"Abnormal days: {result.total_abnormal_days}")
            >>> print(f"Max ratio: {result.max_ratio:.1f}x")
        """
        # 이동평균 계산
        ma = volume.rolling(self.ma_window).mean()
        ratio = volume / ma

        # 이상 거래일 필터링
        abnormal = ratio > self.threshold

        # 결과 DataFrame (boolean indexing 사용, 1D로 변환)
        abnormal_dates = volume[abnormal].index.tolist()

        # .values가 2D일 수 있으므로 flatten
        def ensure_1d(arr):
            if isinstance(arr, np.ndarray) and arr.ndim > 1:
                return arr.flatten()
            return arr

        abnormal_volumes = ensure_1d(volume[abnormal].values)
        abnormal_mas = ensure_1d(ma[abnormal].values)
        abnormal_ratios = ensure_1d(ratio[abnormal].values)

        abnormal_df = pd.DataFrame({
            'date': abnormal_dates,
            'volume': abnormal_volumes,
            'ma': abnormal_mas,
            'ratio': abnormal_ratios
        })

        # 통계 (스칼라로 변환)
        def to_scalar(val):
            if hasattr(val, 'item'):
                return val.item()
            elif isinstance(val, (pd.Series, np.ndarray)):
                return float(val) if val.size == 1 else float(val.iloc[0] if hasattr(val, 'iloc') else val[0])
            return float(val)

        total_abnormal_days = to_scalar(abnormal.sum())
        total_days = len(volume)
        abnormal_ratio = total_abnormal_days / total_days if total_days > 0 else 0.0
        max_ratio = to_scalar(ratio.max()) if len(ratio) > 0 else 0.0

        # 해석
        abnormal_ratio_float = float(abnormal_ratio)
        if abnormal_ratio_float > 0.1:
            interpretation = f"HIGH: {abnormal_ratio_float:.1%}의 날이 이상 거래 (정보 불안정)"
        elif abnormal_ratio_float > 0.05:
            interpretation = f"MEDIUM: {abnormal_ratio_float:.1%}의 날이 이상 거래 (주기적 정보 주입)"
        else:
            interpretation = f"LOW: {abnormal_ratio_float:.1%}의 날이 이상 거래 (안정적)"

        return AbnormalVolumeResult(
            abnormal_dates=abnormal_df,
            total_abnormal_days=int(total_abnormal_days),
            abnormal_ratio=float(abnormal_ratio_float),
            max_ratio=float(max_ratio),
            interpretation=interpretation
        )

    def calculate_private_info_score(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series
    ) -> PrivateInfoResult:
        """
        Private Information Extraction Score

        Formula: (volume_buy - volume_sell) / total_volume

        경제학적 의미:
        - > 0: Buy pressure (정보 우위 매수세)
        - < 0: Sell pressure (정보 우위 매도세)
        - 높은 절댓값 = 강한 정보 비대칭

        Args:
            buy_volume: 매수 거래량 시계열
            sell_volume: 매도 거래량 시계열

        Returns:
            PrivateInfoResult: 정보 점수 결과

        Example:
            >>> result = analyzer.calculate_private_info_score(buy_vol, sell_vol)
            >>> print(f"Net pressure: {result.net_pressure}")
            >>> print(f"Mean score: {result.mean_score:+.3f}")
        """
        # Total volume
        total_volume = buy_volume + sell_volume

        # Private Info Score 계산
        score = (buy_volume - sell_volume) / total_volume
        score = score.replace([np.inf, -np.inf], np.nan).dropna()

        # 통계
        mean_score = score.mean()
        buy_pressure_days = (score > 0.1).sum()  # 10% 이상 매수 우세
        sell_pressure_days = (score < -0.1).sum()  # 10% 이상 매도 우세

        # 순 압력 판단
        if mean_score > 0.05:
            net_pressure = "BUY"
        elif mean_score < -0.05:
            net_pressure = "SELL"
        else:
            net_pressure = "NEUTRAL"

        # 해석
        if abs(mean_score) > 0.1:
            interpretation = f"STRONG {net_pressure} pressure (mean: {mean_score:+.3f})"
        elif abs(mean_score) > 0.05:
            interpretation = f"MODERATE {net_pressure} pressure (mean: {mean_score:+.3f})"
        else:
            interpretation = f"BALANCED market (mean: {mean_score:+.3f})"

        return PrivateInfoResult(
            info_score=score,
            mean_score=float(mean_score),
            buy_pressure_days=int(buy_pressure_days),
            sell_pressure_days=int(sell_pressure_days),
            net_pressure=net_pressure,
            interpretation=interpretation
        )

    def estimate_capm(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> CAPMResult:
        """
        CAPM Regression: E[R_i] = Alpha + Beta * E[R_m]

        경제학적 의미:
        - Alpha > 0: 초과 수익 (정보 우위, 매니저 스킬)
        - Alpha < 0: 손실 (비효율적)
        - Beta > 1: 시장보다 변동성 큼 (공격적)
        - Beta < 1: 시장보다 안정적 (방어적)

        Args:
            asset_returns: 자산 수익률 시계열
            market_returns: 시장 수익률 시계열 (예: SPY)
            risk_free_rate: 무위험 이자율 (연율, 예: 0.05 = 5%)

        Returns:
            CAPMResult: CAPM 회귀 결과

        Example:
            >>> result = analyzer.estimate_capm(aapl_returns, spy_returns)
            >>> print(f"Alpha: {result.alpha:.4f} ({result.alpha_interpretation})")
            >>> print(f"Beta: {result.beta:.2f} ({result.beta_interpretation})")
            >>> print(f"R²: {result.r_squared:.3f}")
        """
        # NaN 제거 및 정렬
        mask = ~(asset_returns.isna() | market_returns.isna())
        X = market_returns[mask].values.reshape(-1, 1)
        y = asset_returns[mask].values

        if len(X) < 10:
            # 관측치 부족
            return CAPMResult(
                alpha=np.nan,
                beta=np.nan,
                r_squared=np.nan,
                n_observations=len(X),
                alpha_interpretation="INSUFFICIENT_DATA",
                beta_interpretation="INSUFFICIENT_DATA"
            )

        # 초과 수익률 계산 (연율 → 일별)
        daily_rf = risk_free_rate / 252
        X = X - daily_rf
        y = y - daily_rf

        # OLS 회귀
        model = LinearRegression()
        model.fit(X, y)

        alpha = float(model.intercept_.item() if hasattr(model.intercept_, 'item') else model.intercept_)
        beta = float(model.coef_[0].item() if hasattr(model.coef_[0], 'item') else model.coef_[0])
        r_squared = float(model.score(X, y))

        # Alpha 해석
        # 연율 환산 (252 영업일)
        annual_alpha = alpha * 252

        if annual_alpha > 0.05:
            alpha_interpretation = f"OUTPERFORM: {annual_alpha:+.1%}/year (정보 우위 가능)"
        elif annual_alpha > 0.02:
            alpha_interpretation = f"SLIGHT_OUTPERFORM: {annual_alpha:+.1%}/year"
        elif annual_alpha > -0.02:
            alpha_interpretation = f"NEUTRAL: {annual_alpha:+.1%}/year"
        elif annual_alpha > -0.05:
            alpha_interpretation = f"SLIGHT_UNDERPERFORM: {annual_alpha:+.1%}/year"
        else:
            alpha_interpretation = f"UNDERPERFORM: {annual_alpha:+.1%}/year (비효율적)"

        # Beta 해석
        if beta > 1.2:
            beta_interpretation = f"AGGRESSIVE: β={beta:.2f} (높은 변동성)"
        elif beta > 0.8:
            beta_interpretation = f"MARKET: β={beta:.2f} (시장 유사)"
        elif beta > 0.3:
            beta_interpretation = f"DEFENSIVE: β={beta:.2f} (낮은 변동성)"
        else:
            beta_interpretation = f"INDEPENDENT: β={beta:.2f} (시장 무관)"

        return CAPMResult(
            alpha=float(alpha),
            beta=float(beta),
            r_squared=float(r_squared),
            n_observations=len(X),
            alpha_interpretation=alpha_interpretation,
            beta_interpretation=beta_interpretation
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_volume_analysis(volume: pd.Series, ma_window: int = 20, threshold: float = 5.0) -> Dict[str, Any]:
    """거래량 이상 탐지 간편 함수"""
    analyzer = InformationFlowAnalyzer(ma_window=ma_window, threshold=threshold)
    result = analyzer.detect_abnormal_volume(volume)
    return result.to_dict()


def quick_info_score(buy_volume: pd.Series, sell_volume: pd.Series) -> Dict[str, Any]:
    """Private Info Score 간편 함수"""
    analyzer = InformationFlowAnalyzer()
    result = analyzer.calculate_private_info_score(buy_volume, sell_volume)
    return result.to_dict()


def quick_capm(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.05) -> Dict[str, Any]:
    """CAPM 간편 함수"""
    analyzer = InformationFlowAnalyzer()
    result = analyzer.estimate_capm(asset_returns, market_returns, risk_free_rate)
    return result.to_dict()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Information Flow Analyzer Test")
    print("=" * 60)

    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_days = 252

    print("\n[1] Abnormal Volume Detection Test")
    print("-" * 40)

    # 정상 거래량 + 이상 거래 5번
    base_volume = np.random.randint(1000, 5000, n_days)
    base_volume[50] = 25000  # 이상 1
    base_volume[100] = 30000  # 이상 2
    base_volume[150] = 20000  # 이상 3
    base_volume[200] = 28000  # 이상 4
    base_volume[220] = 22000  # 이상 5

    volume_series = pd.Series(base_volume, index=pd.date_range('2024-01-01', periods=n_days))

    analyzer = InformationFlowAnalyzer()
    result = analyzer.detect_abnormal_volume(volume_series)

    print(f"  Total abnormal days: {result.total_abnormal_days}")
    print(f"  Abnormal ratio: {result.abnormal_ratio:.1%}")
    print(f"  Max ratio: {result.max_ratio:.1f}x")
    print(f"  Interpretation: {result.interpretation}")

    print("\n[2] Private Information Score Test")
    print("-" * 40)

    # 매수/매도 거래량 (매수 우세 시뮬레이션)
    buy_vol = pd.Series(np.random.randint(500, 3000, n_days), index=volume_series.index)
    sell_vol = pd.Series(np.random.randint(300, 2000, n_days), index=volume_series.index)

    result2 = analyzer.calculate_private_info_score(buy_vol, sell_vol)

    print(f"  Mean score: {result2.mean_score:+.3f}")
    print(f"  Buy pressure days: {result2.buy_pressure_days}")
    print(f"  Sell pressure days: {result2.sell_pressure_days}")
    print(f"  Net pressure: {result2.net_pressure}")
    print(f"  Interpretation: {result2.interpretation}")

    print("\n[3] CAPM Regression Test")
    print("-" * 40)

    # 시장 수익률 + 자산 수익률 (약한 outperform)
    market_returns = pd.Series(np.random.randn(n_days) * 0.01, index=volume_series.index)
    asset_returns = market_returns * 1.2 + np.random.randn(n_days) * 0.005 + 0.0001  # Alpha = 0.0001 (일별)

    result3 = analyzer.estimate_capm(asset_returns, market_returns, risk_free_rate=0.05)

    print(f"  Alpha: {result3.alpha:.6f} (daily)")
    print(f"    → Annual: {result3.alpha * 252:+.1%}")
    print(f"    → {result3.alpha_interpretation}")
    print(f"  Beta: {result3.beta:.3f}")
    print(f"    → {result3.beta_interpretation}")
    print(f"  R²: {result3.r_squared:.3f}")
    print(f"  Observations: {result3.n_observations}")

    print("\n" + "=" * 60)
    print("Information Flow Analyzer Test Complete!")
    print("=" * 60)
