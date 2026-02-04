#!/usr/bin/env python3
"""
Bubble Detection - Detector
============================================================

Main bubble detector class

Economic Foundation:
    - "Bubbles for Fama" (Greenwood et al. 2019)
    - Run-up check: 2-year cumulative return > 100%
    - Risk signals: Volatility spike, share issuance increase
    
Class:
    - BubbleDetector: Main bubble detection engine
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
import logging

from .enums import BubbleWarningLevel, RiskSignalType, JPMorganBubbleStage
from .schemas import (
    RunUpResult, VolatilityResult, IssuanceResult, RiskSignal,
    JPMorganFrameworkResult, BubbleDetectionResult
)

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.bubble_detector')

# Constants
RUNUP_THRESHOLD = 1.0        # 2년 누적 수익률 100% (2배)
VOLATILITY_ZSCORE = 2.0      # 변동성 z-score 임계값
ISSUANCE_THRESHOLD = 0.05    # 주식 발행 증가율 5%
LOOKBACK_YEARS = 2           # Run-up 계산 기간 (년)


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
    # 4. JP Morgan 5-Stage Bubble Framework (NEW)
    # -------------------------------------------------------------------------

    def analyze_jpmorgan_framework(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        runup: RunUpResult,
        volatility: VolatilityResult,
        issuance: IssuanceResult
    ) -> JPMorganFrameworkResult:
        """
        JP Morgan 5단계 버블 프레임워크 분석

        Reference: JP Morgan Wealth Management (제공된 J.docx)
        - Stage 1: 패러다임 전환 (새 기술/산업)
        - Stage 2: 신용 가용성 (자금 조달 용이성)
        - Stage 3: 레버리지/밸류에이션 괴리
        - Stage 4: 투기적 피드백 루프
        - Stage 5: 붕괴 트리거

        Args:
            ticker: 종목 코드
            price_data: OHLCV 데이터
            runup: Run-up 분석 결과
            volatility: 변동성 분석 결과
            issuance: 발행주식 분석 결과

        Returns:
            JPMorganFrameworkResult
        """
        stage_scores = {}

        # Stage 1: 패러다임 전환 (AI, 전기차 등 테마주 여부)
        paradigm_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
        ai_keywords = ['AI', 'artificial intelligence', 'machine learning', 'semiconductor', 'data center']

        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            sector = info.get('sector', '')
            industry = info.get('industry', '').lower()
            description = info.get('longBusinessSummary', '').lower()

            # 패러다임 전환 점수
            paradigm_score = 0
            if sector in paradigm_sectors:
                paradigm_score += 30
            if any(kw.lower() in industry for kw in ai_keywords):
                paradigm_score += 40
            if any(kw.lower() in description for kw in ai_keywords):
                paradigm_score += 30
            stage_scores['paradigm_shift'] = min(100, paradigm_score)

        except Exception:
            stage_scores['paradigm_shift'] = 0

        # Stage 2: 신용 가용성 (주식 발행 증가 = 자금 조달 용이)
        if issuance.data_available and issuance.change_rate:
            credit_score = min(100, max(0, issuance.change_rate * 500))  # 20% 발행 = 100점
        else:
            credit_score = 30  # 기본값
        stage_scores['credit_availability'] = credit_score

        # Stage 3: 레버리지/밸류에이션 괴리
        # PE ratio, PB ratio 기반
        leverage_score = 0
        try:
            pe_ratio = info.get('trailingPE', 0) or info.get('forwardPE', 0)
            pb_ratio = info.get('priceToBook', 0)

            # 닷컴 버블 기준: PE > 50, PB > 10
            if pe_ratio and pe_ratio > 100:
                leverage_score += 50
            elif pe_ratio and pe_ratio > 50:
                leverage_score += 30
            elif pe_ratio and pe_ratio > 30:
                leverage_score += 15

            if pb_ratio and pb_ratio > 10:
                leverage_score += 50
            elif pb_ratio and pb_ratio > 5:
                leverage_score += 30

        except Exception:
            pass
        stage_scores['leverage_gap'] = min(100, leverage_score)

        # Stage 4: 투기적 피드백 루프 (모멘텀 자기강화)
        # Run-up + 변동성 급등 + 거래량 급증
        speculation_score = 0
        if runup.is_runup:
            speculation_score += 40
        if runup.cumulative_return > 2.0:  # 3배 이상
            speculation_score += 30
        if volatility.is_spike:
            speculation_score += 30
        stage_scores['speculative_loop'] = min(100, speculation_score)

        # Stage 5: 붕괴 트리거 (아직 발생하지 않음 = 0점 유지)
        # 향후 VIX 스파이크, 금리 인상 등으로 감지 가능
        stage_scores['collapse_trigger'] = 0

        # 현재 단계 결정
        current_stage = self._determine_jpmorgan_stage(stage_scores)

        # 역사적 비교
        historical_comparison = self._get_historical_comparison(
            runup.cumulative_return,
            stage_scores['paradigm_shift'] > 50
        )

        # 핵심 지표
        key_indicators = []
        if stage_scores['paradigm_shift'] > 50:
            key_indicators.append(f"Paradigm Shift: AI/Tech Theme ({stage_scores['paradigm_shift']:.0f}%)")
        if stage_scores['credit_availability'] > 50:
            key_indicators.append(f"Credit Expansion: Share Issuance ({stage_scores['credit_availability']:.0f}%)")
        if stage_scores['leverage_gap'] > 50:
            key_indicators.append(f"Valuation Gap: PE/PB Elevated ({stage_scores['leverage_gap']:.0f}%)")
        if stage_scores['speculative_loop'] > 50:
            key_indicators.append(f"Speculative Loop: Momentum ({stage_scores['speculative_loop']:.0f}%)")

        # 해석
        interpretation = self._generate_jpmorgan_interpretation(current_stage, stage_scores)

        return JPMorganFrameworkResult(
            current_stage=current_stage,
            stage_scores=stage_scores,
            historical_comparison=historical_comparison,
            key_indicators=key_indicators,
            interpretation=interpretation
        )

    def _determine_jpmorgan_stage(self, stage_scores: Dict[str, float]) -> JPMorganBubbleStage:
        """JP Morgan 단계 결정"""
        scores = [
            (JPMorganBubbleStage.STAGE_4_SPECULATION, stage_scores.get('speculative_loop', 0)),
            (JPMorganBubbleStage.STAGE_3_LEVERAGE, stage_scores.get('leverage_gap', 0)),
            (JPMorganBubbleStage.STAGE_2_CREDIT, stage_scores.get('credit_availability', 0)),
            (JPMorganBubbleStage.STAGE_1_PARADIGM, stage_scores.get('paradigm_shift', 0)),
        ]

        # 가장 높은 점수 단계 (50점 이상만 인정)
        for stage, score in scores:
            if score >= 50:
                return stage

        return JPMorganBubbleStage.NONE

    def _get_historical_comparison(self, cumulative_return: float, is_tech: bool) -> str:
        """역사적 비교 (철도, 닷컴 등)"""
        if cumulative_return >= 5.0 and is_tech:
            return "Similar to 1990s Dot-com Bubble (Tech mania, extreme valuations)"
        elif cumulative_return >= 3.0 and is_tech:
            return "Early-stage Dot-com comparison (1998-1999)"
        elif cumulative_return >= 2.0:
            return "Similar to 1840s Railway Mania (Infrastructure overinvestment)"
        elif cumulative_return >= 1.0:
            return "Moderate run-up, within historical norms"
        else:
            return "No significant historical parallel"

    def _generate_jpmorgan_interpretation(
        self,
        stage: JPMorganBubbleStage,
        scores: Dict[str, float]
    ) -> str:
        """JP Morgan 프레임워크 해석 생성"""
        interpretations = {
            JPMorganBubbleStage.STAGE_1_PARADIGM:
                "패러다임 전환 단계: 새로운 기술/산업에 대한 관심 증가. 아직 버블 초기.",
            JPMorganBubbleStage.STAGE_2_CREDIT:
                "신용 확대 단계: 자금 조달이 용이해지고 있음. IPO, 유상증자 활발.",
            JPMorganBubbleStage.STAGE_3_LEVERAGE:
                "밸류에이션 괴리 단계: 가격이 펀더멘털을 크게 초과. 레버리지 주의.",
            JPMorganBubbleStage.STAGE_4_SPECULATION:
                "투기적 피드백 루프: 모멘텀이 모멘텀을 부르는 자기강화 단계. 위험 고조.",
            JPMorganBubbleStage.STAGE_5_COLLAPSE:
                "붕괴 임박: 트리거 이벤트 발생 가능성. 즉시 리스크 관리 필요.",
            JPMorganBubbleStage.NONE:
                "버블 징후 없음: 정상 범위 내 가격 움직임."
        }

        base = interpretations.get(stage, "분석 불가")

        # 추가 경고
        avg_score = sum(scores.values()) / max(len(scores), 1)
        if avg_score > 60:
            base += " [주의: 전반적 버블 지표 상승]"

        return base

    # -------------------------------------------------------------------------
    # 5. Additional Risk Signals
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

        # 5. JP Morgan 5단계 프레임워크 분석 (NEW)
        jpmorgan_result = self.analyze_jpmorgan_framework(
            ticker=ticker,
            price_data=price_data,
            runup=runup,
            volatility=volatility,
            issuance=issuance
        )

        # JP Morgan 단계별 위험 신호 추가
        if jpmorgan_result.current_stage != JPMorganBubbleStage.NONE:
            risk_signals.append(RiskSignal(
                signal_type=RiskSignalType.PARADIGM_SHIFT if 'PARADIGM' in jpmorgan_result.current_stage.value
                    else RiskSignalType.LEVERAGE_GAP if 'LEVERAGE' in jpmorgan_result.current_stage.value
                    else RiskSignalType.SPECULATIVE_LOOP,
                severity=max(jpmorgan_result.stage_scores.values()) / 100.0,
                description=f"JP Morgan Framework: {jpmorgan_result.current_stage.value}",
                evidence={
                    'stage_scores': jpmorgan_result.stage_scores,
                    'historical_comparison': jpmorgan_result.historical_comparison
                }
            ))

        # 6. 버블 경고 수준 결정
        bubble_level = self._determine_warning_level(runup, risk_signals)

        # 7. 리스크 점수 계산
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
            jpmorgan_framework=jpmorgan_result,
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

