#!/usr/bin/env python3
"""
EIMAS Bubble Detector Module
=============================
"Bubbles for Fama" 논문 기반 버블 경고 시스템

핵심 로직:
1. Run-up Check: 최근 2년 누적 수익률 > 100%
2. Risk Signal Detection:
   - 변동성 급등 (과거 평균 대비 2표준편차 이상)
   - 주식 발행량 증가 (유상증자 신호)
3. Shares Outstanding 조회: yfinance Ticker 객체 활용

Reference:
- Greenwood, R., Shleifer, A., & You, Y. (2019). "Bubbles for Fama"
  Journal of Financial Economics

Usage:
    from lib.bubble_detector import BubbleDetector

    detector = BubbleDetector()
    result = detector.analyze(ticker='TSLA')
    print(result.bubble_warning_level)

Author: EIMAS Team
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.bubble_detector')


# =============================================================================
# Enums and Constants
# =============================================================================

class BubbleWarningLevel(str, Enum):
    """버블 경고 수준"""
    NONE = "NONE"           # 버블 징후 없음
    WATCH = "WATCH"         # 관찰 필요 (Run-up만 충족)
    WARNING = "WARNING"     # 경고 (Run-up + 1개 위험 신호)
    DANGER = "DANGER"       # 위험 (Run-up + 2개 이상 위험 신호)


class RiskSignalType(str, Enum):
    """위험 신호 유형"""
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"     # 변동성 급등
    SHARE_ISSUANCE = "SHARE_ISSUANCE"         # 주식 발행 증가
    PRICE_ACCELERATION = "PRICE_ACCELERATION"  # 가격 가속화
    VOLUME_SURGE = "VOLUME_SURGE"             # 거래량 급증


# Bubbles for Fama 논문 기준값
RUNUP_THRESHOLD = 1.0        # 2년 누적 수익률 100% (2배)
VOLATILITY_ZSCORE = 2.0      # 변동성 z-score 임계값
ISSUANCE_THRESHOLD = 0.05    # 주식 발행 증가율 5%
LOOKBACK_YEARS = 2           # Run-up 계산 기간 (년)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunUpResult:
    """Run-up 분석 결과"""
    cumulative_return: float      # 누적 수익률 (예: 1.5 = 150%)
    is_runup: bool                # Run-up 조건 충족 여부
    start_price: float            # 시작 가격
    end_price: float              # 종료 가격
    period_days: int              # 분석 기간 (일)
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'cumulative_return': self.cumulative_return,
            'cumulative_return_pct': f"{self.cumulative_return * 100:.1f}%",
            'is_runup': self.is_runup,
            'start_price': self.start_price,
            'end_price': self.end_price,
            'period_days': self.period_days,
            'interpretation': self.interpretation
        }


@dataclass
class VolatilityResult:
    """변동성 분석 결과"""
    current_volatility: float     # 현재 변동성 (연율화)
    historical_mean: float        # 과거 평균 변동성
    historical_std: float         # 과거 변동성 표준편차
    zscore: float                 # z-score
    is_spike: bool                # 변동성 급등 여부
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'current_volatility': self.current_volatility,
            'historical_mean': self.historical_mean,
            'zscore': self.zscore,
            'is_spike': self.is_spike,
            'interpretation': self.interpretation
        }


@dataclass
class IssuanceResult:
    """주식 발행 분석 결과"""
    current_shares: Optional[float]       # 현재 발행주식수
    previous_shares: Optional[float]      # 이전 발행주식수 (추정)
    change_rate: Optional[float]          # 변화율
    is_increasing: bool                   # 증가 여부
    data_available: bool                  # 데이터 가용성
    interpretation: str
    data_source: str = "unknown"          # 데이터 출처: sharesOutstanding, balance_sheet, marketCap_estimate
    is_estimated: bool = False            # 추정치 여부 (marketCap/price로 계산된 경우)

    def to_dict(self) -> Dict:
        return {
            'current_shares': self.current_shares,
            'change_rate': self.change_rate,
            'is_increasing': self.is_increasing,
            'data_available': self.data_available,
            'interpretation': self.interpretation,
            'data_source': self.data_source,
            'is_estimated': self.is_estimated
        }


@dataclass
class RiskSignal:
    """개별 위험 신호"""
    signal_type: RiskSignalType
    severity: float               # 심각도 (0-1)
    description: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            'type': self.signal_type.value,
            'severity': self.severity,
            'description': self.description,
            'evidence': self.evidence
        }


@dataclass
class BubbleDetectionResult:
    """버블 탐지 결과"""
    ticker: str
    timestamp: str

    # 핵심 분석 결과
    runup: RunUpResult
    volatility: VolatilityResult
    issuance: IssuanceResult

    # 종합 판단
    bubble_warning_level: BubbleWarningLevel
    risk_signals: List[RiskSignal]
    risk_score: float              # 0-100

    # 추가 정보
    company_name: str = ""
    sector: str = ""
    market_cap: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'company_name': self.company_name,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'runup': self.runup.to_dict(),
            'volatility': self.volatility.to_dict(),
            'issuance': self.issuance.to_dict(),
            'bubble_warning_level': self.bubble_warning_level.value,
            'risk_signals': [s.to_dict() for s in self.risk_signals],
            'risk_score': self.risk_score
        }

    def get_summary(self) -> str:
        """결과 요약 문자열"""
        lines = [
            f"=== Bubble Detection: {self.ticker} ===",
            f"Company: {self.company_name}",
            f"Warning Level: {self.bubble_warning_level.value}",
            f"Risk Score: {self.risk_score:.1f}/100",
            "",
            f"Run-up: {self.runup.cumulative_return * 100:.1f}% ({self.runup.interpretation})",
            f"Volatility: z-score {self.volatility.zscore:.2f} ({self.volatility.interpretation})",
            f"Issuance: {self.issuance.interpretation}",
        ]

        if self.risk_signals:
            lines.append("")
            lines.append("Risk Signals:")
            for signal in self.risk_signals:
                lines.append(f"  - [{signal.signal_type.value}] {signal.description}")

        return "\n".join(lines)


# =============================================================================
# Bubble Detector
# =============================================================================

class BubbleDetector:
    """
    버블 탐지기 (Bubbles for Fama 기반)

    Greenwood, Shleifer & You (2019) 논문의 핵심 발견:
    1. 과거 2년간 100% 이상 상승한 산업군은 버블 위험이 높음
    2. 변동성 급등과 주식 발행 증가는 버블 붕괴의 선행 지표
    3. Run-up 조건이 충족되면 향후 2년 내 40% 이상 하락 확률 증가
    """

    def __init__(
        self,
        runup_threshold: float = RUNUP_THRESHOLD,
        volatility_zscore_threshold: float = VOLATILITY_ZSCORE,
        issuance_threshold: float = ISSUANCE_THRESHOLD,
        lookback_years: int = LOOKBACK_YEARS
    ):
        """
        Args:
            runup_threshold: Run-up 임계값 (기본 1.0 = 100%)
            volatility_zscore_threshold: 변동성 z-score 임계값 (기본 2.0)
            issuance_threshold: 주식 발행 증가율 임계값 (기본 0.05 = 5%)
            lookback_years: Run-up 분석 기간 (기본 2년)
        """
        self.runup_threshold = runup_threshold
        self.vol_zscore_threshold = volatility_zscore_threshold
        self.issuance_threshold = issuance_threshold
        self.lookback_years = lookback_years

        logger.info(
            f"BubbleDetector initialized: "
            f"runup>{runup_threshold*100:.0f}%, "
            f"vol_z>{volatility_zscore_threshold}, "
            f"issuance>{issuance_threshold*100:.0f}%"
        )

    # -------------------------------------------------------------------------
    # 1. Run-up Analysis
    # -------------------------------------------------------------------------

    def check_runup(
        self,
        price_data: pd.Series,
        lookback_years: Optional[int] = None
    ) -> RunUpResult:
        """
        Run-up 체크: 최근 N년 누적 수익률 확인

        Bubbles for Fama 기준:
        - 2년 누적 수익률 > 100%이면 버블 위험군

        Args:
            price_data: 가격 시계열 (종가)
            lookback_years: 분석 기간 (기본값 사용 시 None)

        Returns:
            RunUpResult
        """
        years = lookback_years or self.lookback_years
        lookback_days = years * 252  # 거래일 기준

        if len(price_data) < lookback_days:
            # 데이터 부족 시 가용 데이터로 계산
            lookback_days = len(price_data)

        # 시작/종료 가격
        start_price = float(price_data.iloc[0])
        end_price = float(price_data.iloc[-1])

        # 누적 수익률
        cumulative_return = (end_price / start_price) - 1

        # Run-up 판단
        is_runup = cumulative_return >= self.runup_threshold

        # 해석
        if cumulative_return >= 2.0:
            interpretation = f"Extreme Run-up ({cumulative_return * 100:.0f}%) - 3배 이상 상승"
        elif cumulative_return >= 1.0:
            interpretation = f"Strong Run-up ({cumulative_return * 100:.0f}%) - 버블 위험군"
        elif cumulative_return >= 0.5:
            interpretation = f"Moderate Run-up ({cumulative_return * 100:.0f}%) - 관찰 필요"
        elif cumulative_return >= 0:
            interpretation = f"Normal ({cumulative_return * 100:.0f}%)"
        else:
            interpretation = f"Decline ({cumulative_return * 100:.0f}%)"

        return RunUpResult(
            cumulative_return=float(cumulative_return),
            is_runup=is_runup,
            start_price=start_price,
            end_price=end_price,
            period_days=lookback_days,
            interpretation=interpretation
        )

    # -------------------------------------------------------------------------
    # 2. Volatility Spike Detection
    # -------------------------------------------------------------------------

    def check_volatility_spike(
        self,
        returns: pd.Series,
        window: int = 21,
        historical_window: int = 252
    ) -> VolatilityResult:
        """
        변동성 급등 체크

        현재 변동성이 과거 평균 대비 2표준편차 이상이면 경고

        Args:
            returns: 일별 수익률
            window: 현재 변동성 계산 윈도우 (기본 21일)
            historical_window: 과거 변동성 계산 기간 (기본 252일)

        Returns:
            VolatilityResult
        """
        if len(returns) < historical_window:
            return VolatilityResult(
                current_volatility=np.nan,
                historical_mean=np.nan,
                historical_std=np.nan,
                zscore=0,
                is_spike=False,
                interpretation="Insufficient data"
            )

        # 롤링 변동성 계산 (연율화)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

        # 현재 변동성 (최근 window 기간)
        current_vol = float(rolling_vol.iloc[-1])

        # 과거 변동성 통계 (최근 window 제외)
        historical_vol = rolling_vol.iloc[:-window].dropna()
        hist_mean = float(historical_vol.mean())
        hist_std = float(historical_vol.std())

        # Z-score 계산
        if hist_std > 0:
            zscore = (current_vol - hist_mean) / hist_std
        else:
            zscore = 0

        # 급등 판단
        is_spike = zscore >= self.vol_zscore_threshold

        # 해석
        if zscore >= 3.0:
            interpretation = f"Extreme volatility spike (z={zscore:.2f})"
        elif zscore >= 2.0:
            interpretation = f"Significant volatility spike (z={zscore:.2f})"
        elif zscore >= 1.0:
            interpretation = f"Elevated volatility (z={zscore:.2f})"
        elif zscore >= -1.0:
            interpretation = f"Normal volatility (z={zscore:.2f})"
        else:
            interpretation = f"Low volatility (z={zscore:.2f})"

        return VolatilityResult(
            current_volatility=current_vol,
            historical_mean=hist_mean,
            historical_std=hist_std,
            zscore=float(zscore),
            is_spike=is_spike,
            interpretation=interpretation
        )

    # -------------------------------------------------------------------------
    # 3. Share Issuance Check
    # -------------------------------------------------------------------------

    def _estimate_shares_from_market_cap(self, info: Dict) -> Optional[float]:
        """
        시가총액/종가로 주식수 추정 (Fallback)

        경제학적 근거:
        - Market Cap = Shares Outstanding * Current Price
        - Shares = Market Cap / Price

        주의사항:
        - 장 마감 후 가격 변동으로 약간의 오차 발생 가능
        - 하지만 주식 발행 증가율 (5% 임계값) 판단에는 충분
        """
        market_cap = info.get('marketCap')
        price = info.get('regularMarketPrice') or info.get('previousClose')

        if market_cap and price and price > 0:
            return market_cap / price
        return None

    def _get_shares_from_financials(self, yf_ticker) -> Tuple[Optional[float], Optional[float]]:
        """
        분기별 재무제표에서 주식 발행량 조회

        우선순위:
        1. quarterly_balance_sheet의 'Ordinary Shares Number'
        2. balance_sheet의 'Ordinary Shares Number'
        3. 'Share Issued' 또는 'Common Stock' 필드

        Returns:
            (current_shares, previous_shares) 튜플
        """
        for sheet in [yf_ticker.quarterly_balance_sheet, yf_ticker.balance_sheet]:
            try:
                if sheet is None or sheet.empty:
                    continue

                for field in ['Ordinary Shares Number', 'Share Issued', 'Common Stock']:
                    if field in sheet.index:
                        series = sheet.loc[field].dropna()
                        if len(series) >= 2:
                            return float(series.iloc[0]), float(series.iloc[-1])
                        elif len(series) == 1:
                            return float(series.iloc[0]), None
            except Exception:
                continue

        return None, None

    def check_share_issuance(
        self,
        ticker: str
    ) -> IssuanceResult:
        """
        주식 발행량 확인 (3단계 Fallback 로직)

        데이터 소스 우선순위:
        1. yf.Ticker.info['sharesOutstanding'] (가장 정확, 현재값만)
        2. quarterly_balance_sheet['Ordinary Shares Number'] (시계열 비교 가능)
        3. marketCap / currentPrice (추정치, Fallback)

        Args:
            ticker: 티커 심볼

        Returns:
            IssuanceResult (data_source, is_estimated 필드로 출처 추적)
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info

            current_shares = None
            previous_shares = None
            data_source = "none"
            is_estimated = False

            # Priority 1: Direct sharesOutstanding (가장 정확)
            current_shares = info.get('sharesOutstanding')
            if current_shares is not None:
                data_source = "sharesOutstanding"

            # Priority 2: Balance Sheet (시계열 비교 가능)
            if current_shares is None:
                bs_current, bs_previous = self._get_shares_from_financials(yf_ticker)
                if bs_current is not None:
                    current_shares = bs_current
                    previous_shares = bs_previous
                    data_source = "balance_sheet"

            # Priority 3: Market Cap / Price (Fallback 추정치)
            if current_shares is None:
                estimated = self._estimate_shares_from_market_cap(info)
                if estimated is not None:
                    current_shares = estimated
                    data_source = "marketCap_estimate"
                    is_estimated = True
                    logger.info(f"{ticker}: Using marketCap/price fallback for shares estimation")

            # 데이터 없음 -> 반환
            if current_shares is None:
                return IssuanceResult(
                    current_shares=None,
                    previous_shares=None,
                    change_rate=None,
                    is_increasing=False,
                    data_available=False,
                    interpretation="No share data available from any source (sharesOutstanding, balance_sheet, marketCap)",
                    data_source="none",
                    is_estimated=False
                )

            # 이전 발행량이 없으면 재무제표에서 시도
            change_rate = None
            is_increasing = False

            if previous_shares is None and data_source != "balance_sheet":
                bs_current, bs_previous = self._get_shares_from_financials(yf_ticker)
                if bs_previous is not None:
                    previous_shares = bs_previous

            # 변화율 계산
            if previous_shares is not None and previous_shares > 0:
                change_rate = (current_shares - previous_shares) / previous_shares
                is_increasing = change_rate > self.issuance_threshold

            # 해석 생성
            source_note = f" [source: {data_source}]" if is_estimated else ""
            if change_rate is not None:
                if is_increasing:
                    interpretation = f"Share issuance increased by {change_rate * 100:.1f}% (Warning){source_note}"
                elif change_rate > 0:
                    interpretation = f"Minor share increase ({change_rate * 100:.1f}%){source_note}"
                elif change_rate < -0.05:
                    interpretation = f"Share buyback ({change_rate * 100:.1f}%){source_note}"
                else:
                    interpretation = f"Stable share count{source_note}"
            else:
                interpretation = f"Current shares: {current_shares:,.0f} (historical comparison unavailable){source_note}"

            return IssuanceResult(
                current_shares=float(current_shares),
                previous_shares=float(previous_shares) if previous_shares else None,
                change_rate=float(change_rate) if change_rate is not None else None,
                is_increasing=is_increasing,
                data_available=True,
                interpretation=interpretation,
                data_source=data_source,
                is_estimated=is_estimated
            )

        except Exception as e:
            logger.warning(f"Failed to get issuance data for {ticker}: {e}")
            return IssuanceResult(
                current_shares=None,
                previous_shares=None,
                change_rate=None,
                is_increasing=False,
                data_available=False,
                interpretation=f"Error retrieving data: {str(e)}",
                data_source="error",
                is_estimated=False
            )

    # -------------------------------------------------------------------------
    # 4. Additional Risk Signals
    # -------------------------------------------------------------------------

    def check_price_acceleration(
        self,
        price_data: pd.Series,
        short_window: int = 21,
        long_window: int = 63
    ) -> Optional[RiskSignal]:
        """
        가격 가속화 체크

        단기 모멘텀이 장기 모멘텀을 크게 초과하면 경고
        """
        if len(price_data) < long_window:
            return None

        short_return = float(price_data.iloc[-1] / price_data.iloc[-short_window] - 1)
        long_return = float(price_data.iloc[-1] / price_data.iloc[-long_window] - 1)

        # 단기 수익률이 장기 수익률의 2배 이상이면 가속화
        if long_return > 0 and short_return > long_return * 2:
            severity = min(1.0, (short_return / long_return - 2) / 2)
            return RiskSignal(
                signal_type=RiskSignalType.PRICE_ACCELERATION,
                severity=severity,
                description=f"Price acceleration detected: {short_window}d return ({short_return*100:.1f}%) > 2x {long_window}d return ({long_return*100:.1f}%)",
                evidence={
                    'short_return': short_return,
                    'long_return': long_return,
                    'ratio': short_return / long_return if long_return != 0 else 0
                }
            )
        return None

    def check_volume_surge(
        self,
        volume_data: pd.Series,
        window: int = 21,
        threshold: float = 3.0
    ) -> Optional[RiskSignal]:
        """
        거래량 급증 체크

        최근 거래량이 평균 대비 3배 이상이면 경고
        """
        if len(volume_data) < window * 2:
            return None

        recent_volume = float(volume_data.iloc[-window:].mean())
        historical_volume = float(volume_data.iloc[:-window].mean())

        if historical_volume > 0:
            volume_ratio = recent_volume / historical_volume

            if volume_ratio >= threshold:
                severity = min(1.0, (volume_ratio - threshold) / 3)
                return RiskSignal(
                    signal_type=RiskSignalType.VOLUME_SURGE,
                    severity=severity,
                    description=f"Volume surge: {volume_ratio:.1f}x average",
                    evidence={
                        'recent_volume': recent_volume,
                        'historical_volume': historical_volume,
                        'ratio': volume_ratio
                    }
                )
        return None

    # -------------------------------------------------------------------------
    # 5. Main Analysis
    # -------------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        price_data: Optional[pd.DataFrame] = None
    ) -> BubbleDetectionResult:
        """
        버블 탐지 분석 실행

        Args:
            ticker: 티커 심볼
            price_data: OHLCV 데이터 (없으면 yfinance에서 다운로드)

        Returns:
            BubbleDetectionResult
        """
        # 데이터 수집
        if price_data is None:
            # 2년 + 여유 데이터
            yf_ticker = yf.Ticker(ticker)
            price_data = yf_ticker.history(period='3y')

        if price_data.empty:
            raise ValueError(f"No price data available for {ticker}")

        # 기본 정보 수집
        company_name = ""
        sector = ""
        market_cap = None

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            company_name = info.get('shortName', ticker)
            sector = info.get('sector', '')
            market_cap = info.get('marketCap')
        except Exception:
            pass

        # 수익률 계산
        close_prices = price_data['Close']
        returns = close_prices.pct_change().dropna()

        # 1. Run-up 분석
        runup = self.check_runup(close_prices)

        # 2. 변동성 분석
        volatility = self.check_volatility_spike(returns)

        # 3. 주식 발행 분석
        issuance = self.check_share_issuance(ticker)

        # 4. 추가 위험 신호 수집
        risk_signals = []

        # 변동성 급등 신호
        if volatility.is_spike:
            risk_signals.append(RiskSignal(
                signal_type=RiskSignalType.VOLATILITY_SPIKE,
                severity=min(1.0, (volatility.zscore - 2) / 2),
                description=volatility.interpretation,
                evidence={
                    'current_vol': volatility.current_volatility,
                    'historical_mean': volatility.historical_mean,
                    'zscore': volatility.zscore
                }
            ))

        # 주식 발행 증가 신호
        if issuance.is_increasing:
            risk_signals.append(RiskSignal(
                signal_type=RiskSignalType.SHARE_ISSUANCE,
                severity=min(1.0, issuance.change_rate / 0.2) if issuance.change_rate else 0.5,
                description=issuance.interpretation,
                evidence={
                    'current_shares': issuance.current_shares,
                    'change_rate': issuance.change_rate
                }
            ))

        # 가격 가속화 신호
        accel_signal = self.check_price_acceleration(close_prices)
        if accel_signal:
            risk_signals.append(accel_signal)

        # 거래량 급증 신호
        if 'Volume' in price_data.columns:
            volume_signal = self.check_volume_surge(price_data['Volume'])
            if volume_signal:
                risk_signals.append(volume_signal)

        # 5. 버블 경고 수준 결정
        bubble_level = self._determine_warning_level(runup, risk_signals)

        # 6. 리스크 점수 계산
        risk_score = self._calculate_risk_score(runup, risk_signals)

        return BubbleDetectionResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            runup=runup,
            volatility=volatility,
            issuance=issuance,
            bubble_warning_level=bubble_level,
            risk_signals=risk_signals,
            risk_score=risk_score,
            company_name=company_name,
            sector=sector,
            market_cap=market_cap
        )

    def _determine_warning_level(
        self,
        runup: RunUpResult,
        risk_signals: List[RiskSignal]
    ) -> BubbleWarningLevel:
        """버블 경고 수준 결정"""
        if not runup.is_runup:
            return BubbleWarningLevel.NONE

        signal_count = len(risk_signals)

        if signal_count >= 2:
            return BubbleWarningLevel.DANGER
        elif signal_count == 1:
            return BubbleWarningLevel.WARNING
        else:
            return BubbleWarningLevel.WATCH

    def _calculate_risk_score(
        self,
        runup: RunUpResult,
        risk_signals: List[RiskSignal]
    ) -> float:
        """리스크 점수 계산 (0-100)"""
        score = 0

        # Run-up 점수 (최대 40점)
        if runup.cumulative_return >= 2.0:
            score += 40
        elif runup.cumulative_return >= 1.0:
            score += 30
        elif runup.cumulative_return >= 0.5:
            score += 15

        # 위험 신호 점수 (각 최대 20점)
        for signal in risk_signals:
            score += signal.severity * 20

        return min(100, score)

    def analyze_multiple(
        self,
        tickers: List[str]
    ) -> Dict[str, BubbleDetectionResult]:
        """여러 티커 분석"""
        results = {}

        for ticker in tickers:
            try:
                results[ticker] = self.analyze(ticker)
            except Exception as e:
                logger.warning(f"Failed to analyze {ticker}: {e}")
                continue

        return results

    def get_bubble_watchlist(
        self,
        results: Dict[str, BubbleDetectionResult],
        min_level: BubbleWarningLevel = BubbleWarningLevel.WATCH
    ) -> List[str]:
        """버블 경고 워치리스트 반환"""
        level_order = {
            BubbleWarningLevel.NONE: 0,
            BubbleWarningLevel.WATCH: 1,
            BubbleWarningLevel.WARNING: 2,
            BubbleWarningLevel.DANGER: 3
        }

        min_order = level_order[min_level]

        watchlist = [
            ticker for ticker, result in results.items()
            if level_order[result.bubble_warning_level] >= min_order
        ]

        # 위험도 순으로 정렬
        watchlist.sort(
            key=lambda t: (
                level_order[results[t].bubble_warning_level],
                results[t].risk_score
            ),
            reverse=True
        )

        return watchlist


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_bubble_check(ticker: str) -> Dict[str, Any]:
    """빠른 버블 체크"""
    detector = BubbleDetector()
    result = detector.analyze(ticker)

    return {
        'ticker': ticker,
        'warning_level': result.bubble_warning_level.value,
        'risk_score': result.risk_score,
        'runup_return': f"{result.runup.cumulative_return * 100:.1f}%",
        'is_runup': result.runup.is_runup,
        'volatility_zscore': result.volatility.zscore,
        'risk_signals': len(result.risk_signals)
    }


def scan_for_bubbles(tickers: List[str]) -> Dict[str, Dict]:
    """여러 티커 버블 스캔"""
    detector = BubbleDetector()
    results = detector.analyze_multiple(tickers)

    summary = {}
    for ticker, result in results.items():
        summary[ticker] = {
            'level': result.bubble_warning_level.value,
            'score': result.risk_score,
            'runup': f"{result.runup.cumulative_return * 100:.1f}%",
            'signals': [s.signal_type.value for s in result.risk_signals]
        }

    return summary


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Bubble Detector Test")
    print("=" * 60)

    detector = BubbleDetector()

    # 테스트 티커 (다양한 유형)
    test_tickers = ['TSLA', 'NVDA', 'AAPL', 'SPY', 'GME']

    for ticker in test_tickers:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {ticker}")
        print('=' * 60)

        try:
            result = detector.analyze(ticker)

            print(result.get_summary())
            print()

        except Exception as e:
            print(f"Error: {e}")

    # 요약 스캔
    print("\n" + "=" * 60)
    print("Bubble Scan Summary")
    print("=" * 60)

    try:
        scan_results = scan_for_bubbles(test_tickers)

        for ticker, data in scan_results.items():
            icon = {
                'NONE': '',
                'WATCH': '',
                'WARNING': '',
                'DANGER': ''
            }.get(data['level'], '')

            print(f"  {icon} {ticker}: {data['level']} (Score: {data['score']:.0f}, Run-up: {data['runup']})")

    except Exception as e:
        print(f"Scan error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
