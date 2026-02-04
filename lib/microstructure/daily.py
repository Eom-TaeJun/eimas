#!/usr/bin/env python3
"""
Microstructure - Daily Analysis (AMFL Chapter 19)
============================================================

일별 데이터 기반 시장 미세구조 분석

Economic Foundation:
    - Amihud (2002): Illiquidity = |return| / volume
    - Roll (1984): Spread from serial covariance
    - VPIN approximation for daily data

Class:
    - DailyMicrostructureAnalyzer: AMFL Chapter 19 지표

Functions:
    - calculate_amihud: Amihud lambda
    - calculate_roll_spread_daily: Roll spread
    - calculate_vpin_daily: VPIN approximation
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .schemas import (
    AmihudLambdaResult,
    RollSpreadResult,
    VPINApproxResult,
    DailyMicrostructureResult
)

logger = logging.getLogger(__name__)


class DailyMicrostructureAnalyzer:
    """
    일별 데이터 기반 미세구조 분석기 (AMFL Chapter 19)

    고빈도 데이터가 없을 때 일별 OHLCV로 시장 미세구조 근사
    Amihud Lambda, Roll Spread, VPIN Approximation 계산
    """

    def __init__(
        self,
        lookback_days: int = 252,
        vpin_window: int = 50,
        amihud_scale: float = 1e6
    ):
        """
        Args:
            lookback_days: 분석 기간 (일)
            vpin_window: VPIN 계산용 윈도우 크기
            amihud_scale: Amihud Lambda 스케일링 (기본 1e6)
        """
        self.lookback_days = lookback_days
        self.vpin_window = vpin_window
        self.amihud_scale = amihud_scale

    def calculate_amihud_lambda(
        self,
        returns: pd.Series,
        volume: pd.Series,
        price: pd.Series,
        min_periods: Optional[int] = None,
        winsorize: bool = True
    ) -> AmihudLambdaResult:
        """
        Amihud Lambda 계산 (AMFL Ch.19 비유동성 지표)

        Lambda = mean(|r_t| / DollarVolume_t) * scale

        높은 Lambda = 낮은 유동성 (가격 충격이 큼)
        낮은 Lambda = 높은 유동성 (가격 충격이 작음)

        NaN 처리 전략:
        - 거래량 0인 날: 유동성 측정 불가 → 제외
        - 극단적 Lambda: Winsorize로 영향 완화 (1-99 percentile)

        Args:
            returns: 일별 수익률
            volume: 일별 거래량
            price: 일별 가격 (종가)
            min_periods: 최소 데이터 포인트 (기본값: RollingWindowConfig 사용)
            winsorize: 극단치 Winsorize 여부 (기본값: True)

        Returns:
            AmihudLambdaResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('amihud_lambda', 'min_periods') or 20

        # Dollar Volume 계산
        dollar_volume = volume * price

        # 유효한 데이터만 필터링 (0 거래량 및 NaN 제외)
        valid_mask = (
            (dollar_volume > 0) &
            (returns.notna()) &
            (np.isfinite(returns))
        )
        abs_returns = returns.abs()[valid_mask]
        dv = dollar_volume[valid_mask]

        if len(abs_returns) < min_periods:
            return AmihudLambdaResult(
                lambda_value=np.nan,
                lambda_series=None,
                avg_daily_volume=0,
                interpretation=f"Insufficient data (need {min_periods}, got {len(abs_returns)})"
            )

        # 일별 Lambda 계산
        daily_lambda = (abs_returns / dv) * self.amihud_scale

        # Winsorize: 극단치 영향 완화 (1-99 percentile)
        if winsorize and len(daily_lambda.dropna()) > 10:
            lower, upper = np.percentile(daily_lambda.dropna(), [1, 99])
            daily_lambda = daily_lambda.clip(lower, upper)

        # 평균 Lambda (극단치 제거를 위해 중앙값 사용)
        lambda_value = float(daily_lambda.median())
        avg_volume = float(dv.mean())

        # 해석
        if lambda_value < 0.01:
            interpretation = "Very High Liquidity (대형주 수준)"
        elif lambda_value < 0.1:
            interpretation = "High Liquidity (유동성 양호)"
        elif lambda_value < 1.0:
            interpretation = "Moderate Liquidity (평균 수준)"
        elif lambda_value < 10.0:
            interpretation = "Low Liquidity (유동성 부족)"
        else:
            interpretation = "Very Low Liquidity (거래 주의)"

        return AmihudLambdaResult(
            lambda_value=lambda_value,
            lambda_series=daily_lambda,
            avg_daily_volume=avg_volume,
            interpretation=interpretation
        )

    def calculate_roll_spread(
        self,
        price: pd.Series,
        min_periods: Optional[int] = None
    ) -> RollSpreadResult:
        """
        Roll Spread 계산 (AMFL Ch.19 유효 스프레드 추정)

        Roll (1984) Model:
        - 가격 변화의 시리얼 공분산을 이용
        - Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
        - 공분산이 양수면 스프레드 = 0 (모델 가정 위배)

        NaN 처리 전략:
        - diff()로 생성된 첫 NaN은 dropna()로 제거
        - 연속적인 NaN 비율이 높으면 경고 로깅

        Args:
            price: 일별 가격 시계열
            min_periods: 최소 데이터 포인트 (기본값: RollingWindowConfig 사용)

        Returns:
            RollSpreadResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('roll_spread', 'min_periods') or 10

        # 가격 변화 계산
        delta_price = price.diff()

        # NaN 비율 체크 (데이터 품질 경고)
        nan_ratio = delta_price.isna().sum() / len(delta_price) if len(delta_price) > 0 else 1.0
        if nan_ratio > 0.1:
            import logging
            logging.getLogger('eimas.microstructure').warning(
                f"High NaN ratio ({nan_ratio:.1%}) in price series for Roll Spread"
            )

        delta_price = delta_price.dropna()

        if len(delta_price) < min_periods:
            return RollSpreadResult(
                spread=np.nan,
                covariance=np.nan,
                is_valid=False,
                interpretation=f"Insufficient data (need {min_periods}, got {len(delta_price)})"
            )

        # 시리얼 공분산 계산
        delta_price_lag = delta_price.shift(1).dropna()
        delta_price_curr = delta_price.iloc[1:]

        # 인덱스 맞추기
        common_idx = delta_price_curr.index.intersection(delta_price_lag.index)
        covariance = float(np.cov(
            delta_price_curr.loc[common_idx],
            delta_price_lag.loc[common_idx]
        )[0, 1])

        # Roll Spread 계산
        # 공분산이 양수면 0으로 처리 (모델 가정: 공분산은 음수여야 함)
        if covariance >= 0:
            spread = 0.0
            is_valid = False
            interpretation = "Positive covariance (모델 가정 위배, spread=0)"
        else:
            # Spread = 2 * sqrt(-Cov)
            spread_raw = 2 * np.sqrt(-covariance)
            is_valid = True

            # 스프레드를 퍼센트로 변환
            avg_price = price.mean()
            spread = float((spread_raw / avg_price) * 100)

            if spread < 0.05:
                interpretation = f"Very Tight Spread ({spread:.3f}%)"
            elif spread < 0.2:
                interpretation = f"Normal Spread ({spread:.3f}%)"
            elif spread < 0.5:
                interpretation = f"Wide Spread ({spread:.3f}%)"
            else:
                interpretation = f"Very Wide Spread ({spread:.3f}%) - 유동성 주의"

        return RollSpreadResult(
            spread=spread,
            covariance=covariance,
            is_valid=is_valid,
            interpretation=interpretation
        )

    def calculate_vpin_approximation(
        self,
        open_price: pd.Series,
        high_price: pd.Series,
        low_price: pd.Series,
        close_price: pd.Series,
        volume: pd.Series,
        min_periods: Optional[int] = None,
        fill_method: str = 'neutral'
    ) -> VPINApproxResult:
        """
        VPIN 근사치 계산 (일별 OHLC 기반, AMFL Ch.19)

        고빈도 데이터가 없는 경우 일별 데이터로 VPIN을 근사

        방법 (Bulk Volume Classification):
        1. 일중 가격 움직임으로 매수/매도 압력 추정
        2. Buy Volume = Volume * (Close - Low) / (High - Low)
        3. Sell Volume = Volume * (High - Close) / (High - Low)
        4. VPIN = |Buy - Sell| / Total Volume (rolling window)

        NaN 처리 전략 (fill_method):
        - 'neutral': 0.5 (매수/매도 균형 가정)
          경제학적 근거: 가격 변동 없음 = 정보 비대칭 없음
        - 'ffill': 이전 값 사용 (시계열 연속성 유지)
        - 'none': NaN 유지 (후속 계산에서 제외)

        Args:
            open_price, high_price, low_price, close_price: OHLC 가격
            volume: 거래량
            min_periods: 롤링 윈도우 최소 데이터 포인트 (기본값: RollingWindowConfig)
            fill_method: NaN 처리 방법 ('neutral', 'ffill', 'none')

        Returns:
            VPINApproxResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('vpin', 'min_periods') or 5

        # 가격 범위 계산
        price_range = high_price - low_price

        # 0 범위 처리 (가격 변동이 없는 날)
        price_range = price_range.replace(0, np.nan)

        # BVC (Bulk Volume Classification)
        # 종가가 고가에 가까우면 매수 우세, 저가에 가까우면 매도 우세
        buy_ratio = (close_price - low_price) / price_range
        sell_ratio = (high_price - close_price) / price_range

        # NaN 처리 (configurable)
        if fill_method == 'neutral':
            # 경제학적 근거: 가격 변동 없음 = 정보 비대칭 없음 → 50:50
            buy_ratio = buy_ratio.fillna(0.5)
            sell_ratio = sell_ratio.fillna(0.5)
        elif fill_method == 'ffill':
            # 시계열 연속성 유지 → 이전 값 사용, 첫 값 없으면 0.5
            buy_ratio = buy_ratio.ffill().fillna(0.5)
            sell_ratio = sell_ratio.ffill().fillna(0.5)
        # else: 'none' - NaN 유지

        # 매수/매도 거래량 추정
        buy_volume = volume * buy_ratio
        sell_volume = volume * sell_ratio

        # VPIN 계산 (rolling window with min_periods)
        window = min(self.vpin_window, len(volume))

        rolling_buy = buy_volume.rolling(window=window, min_periods=min_periods).sum()
        rolling_sell = sell_volume.rolling(window=window, min_periods=min_periods).sum()
        rolling_total = volume.rolling(window=window, min_periods=min_periods).sum()

        # VPIN = |V_buy - V_sell| / V_total
        vpin_series = (rolling_buy - rolling_sell).abs() / rolling_total

        # 최신 VPIN 값
        current_vpin = float(vpin_series.iloc[-1]) if not vpin_series.empty else np.nan

        if pd.isna(current_vpin):
            return VPINApproxResult(
                vpin=np.nan,
                buy_volume_ratio=0.5,
                sell_volume_ratio=0.5,
                toxicity_level="UNKNOWN",
                interpretation="Insufficient data"
            )

        # 최근 매수/매도 비율
        recent_buy_ratio = float(buy_ratio.iloc[-window:].mean())
        recent_sell_ratio = float(sell_ratio.iloc[-window:].mean())

        # Toxicity Level 결정
        # VPIN이 높을수록 정보 비대칭성(toxicity)이 높음
        if current_vpin < 0.2:
            toxicity = "LOW"
            interpretation = "Low order flow toxicity (정보 비대칭 낮음)"
        elif current_vpin < 0.4:
            toxicity = "MEDIUM"
            interpretation = "Moderate toxicity (주의 관찰 필요)"
        elif current_vpin < 0.6:
            toxicity = "HIGH"
            interpretation = "High toxicity (정보 비대칭 높음, 급변동 가능)"
        else:
            toxicity = "EXTREME"
            interpretation = "Extreme toxicity (Flash crash 위험)"

        return VPINApproxResult(
            vpin=current_vpin,
            buy_volume_ratio=recent_buy_ratio,
            sell_volume_ratio=recent_sell_ratio,
            toxicity_level=toxicity,
            interpretation=interpretation
        )

    def analyze(
        self,
        ticker: str,
        data: pd.DataFrame
    ) -> DailyMicrostructureResult:
        """
        일별 데이터 기반 통합 미세구조 분석

        Args:
            ticker: 티커 심볼
            data: OHLCV DataFrame (columns: Open, High, Low, Close, Volume)

        Returns:
            DailyMicrostructureResult
        """
        # 데이터 검증
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # 수익률 계산
        returns = data['Close'].pct_change()

        # 1. Amihud Lambda
        amihud = self.calculate_amihud_lambda(
            returns=returns,
            volume=data['Volume'],
            price=data['Close']
        )

        # 2. Roll Spread
        roll = self.calculate_roll_spread(price=data['Close'])

        # 3. VPIN Approximation
        vpin = self.calculate_vpin_approximation(
            open_price=data['Open'],
            high_price=data['High'],
            low_price=data['Low'],
            close_price=data['Close'],
            volume=data['Volume']
        )

        # 종합 유동성 점수 계산 (0-100)
        liquidity_score = self._calculate_liquidity_score(amihud, roll, vpin)

        # 리스크 플래그
        risk_flags = self._identify_risk_flags(amihud, roll, vpin)

        return DailyMicrostructureResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            amihud=amihud,
            roll_spread=roll,
            vpin_approx=vpin,
            overall_liquidity_score=liquidity_score,
            risk_flags=risk_flags
        )

    def _calculate_liquidity_score(
        self,
        amihud: AmihudLambdaResult,
        roll: RollSpreadResult,
        vpin: VPINApproxResult
    ) -> float:
        """유동성 점수 계산 (0-100, 높을수록 좋음)"""
        scores = []

        # Amihud 점수 (낮을수록 좋음)
        if not np.isnan(amihud.lambda_value):
            if amihud.lambda_value < 0.01:
                scores.append(100)
            elif amihud.lambda_value < 0.1:
                scores.append(80)
            elif amihud.lambda_value < 1.0:
                scores.append(60)
            elif amihud.lambda_value < 10.0:
                scores.append(40)
            else:
                scores.append(20)

        # Roll Spread 점수 (낮을수록 좋음)
        if not np.isnan(roll.spread):
            if roll.spread < 0.05:
                scores.append(100)
            elif roll.spread < 0.2:
                scores.append(80)
            elif roll.spread < 0.5:
                scores.append(60)
            else:
                scores.append(40)

        # VPIN 점수 (낮을수록 좋음)
        if not np.isnan(vpin.vpin):
            scores.append(max(0, 100 - vpin.vpin * 100))

        return float(np.mean(scores)) if scores else 50.0

    def _identify_risk_flags(
        self,
        amihud: AmihudLambdaResult,
        roll: RollSpreadResult,
        vpin: VPINApproxResult
    ) -> List[str]:
        """리스크 플래그 식별"""
        flags = []

        # Amihud 경고
        if not np.isnan(amihud.lambda_value) and amihud.lambda_value > 1.0:
            flags.append("LOW_LIQUIDITY")

        # Roll Spread 경고
        if not np.isnan(roll.spread) and roll.spread > 0.5:
            flags.append("WIDE_SPREAD")

        # VPIN 경고
        if not np.isnan(vpin.vpin):
            if vpin.vpin > 0.6:
                flags.append("EXTREME_TOXICITY")
            elif vpin.vpin > 0.4:
                flags.append("HIGH_TOXICITY")

        return flags

    def analyze_multiple(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, DailyMicrostructureResult]:
        """여러 티커 분석"""
        results = {}

        for ticker, data in market_data.items():
            try:
                results[ticker] = self.analyze(ticker, data)
            except Exception as e:
                print(f"Warning: Failed to analyze {ticker}: {e}")
                continue

        return results

    def get_summary(
        self,
        results: Dict[str, DailyMicrostructureResult]
    ) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not results:
            return {'error': 'No results'}

        # 유동성 점수 기준 정렬
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].overall_liquidity_score,
            reverse=True
        )

        # 최고/최저 유동성
        most_liquid = sorted_results[0] if sorted_results else None
        least_liquid = sorted_results[-1] if sorted_results else None

        # 위험 티커
        risky_tickers = [
            ticker for ticker, result in results.items()
            if result.risk_flags
        ]

        # 평균 VPIN
        vpins = [r.vpin_approx.vpin for r in results.values()
                 if not np.isnan(r.vpin_approx.vpin)]
        avg_vpin = float(np.mean(vpins)) if vpins else np.nan

        return {
            'total_analyzed': len(results),
            'avg_liquidity_score': float(np.mean([r.overall_liquidity_score for r in results.values()])),
            'avg_vpin': avg_vpin,
            'most_liquid': most_liquid[0] if most_liquid else None,
            'least_liquid': least_liquid[0] if least_liquid else None,
            'risky_tickers': risky_tickers,
            'risk_count': len(risky_tickers)
        }


# ============================================================================
# Convenience Functions for Daily Microstructure
# ============================================================================

def calculate_amihud(returns: pd.Series, volume: pd.Series, price: pd.Series) -> float:
    """Amihud Lambda 간편 계산"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_amihud_lambda(returns, volume, price)
    return result.lambda_value


def calculate_roll_spread_daily(price: pd.Series) -> float:
    """Roll Spread 간편 계산 (일별 데이터)"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_roll_spread(price)
    return result.spread


def calculate_vpin_daily(ohlcv: pd.DataFrame) -> float:
    """VPIN 간편 계산 (일별 OHLCV)"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_vpin_approximation(
        ohlcv['Open'], ohlcv['High'], ohlcv['Low'], ohlcv['Close'], ohlcv['Volume']
    )
    return result.vpin

