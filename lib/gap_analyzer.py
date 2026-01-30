#!/usr/bin/env python3
"""
Market-Model Gap Analyzer
==========================
Goldman Sachs의 갭 분석 방법론 구현

시장 가격에 내재된 기대치와 모델 예측치 간의 괴리 분석.
괴리가 클수록 투자 기회 또는 리스크 존재.

핵심 분석:
1. Implied Growth: 채권/주식 시장 가격에서 역산한 성장률
2. Recession Probability: HY 스프레드에서 역산한 침체 확률
3. Fed Path: 시장이 반영한 금리 경로 vs FOMC 전망
4. Equity Risk Premium: 주식 vs 채권 기대수익률 차이

Usage:
    from lib.gap_analyzer import MarketModelGapAnalyzer

    analyzer = MarketModelGapAnalyzer()
    result = analyzer.analyze()
    print(result.signal, result.opportunity)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf

# FRED 데이터 (가능하면)
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

import os


@dataclass
class GapResult:
    """개별 갭 분석 결과"""
    metric: str
    market_implied: float
    model_forecast: float
    gap: float
    signal: str  # BULLISH, BEARISH, NEUTRAL
    interpretation: str


@dataclass
class GapAnalysisResult:
    """갭 분석 종합 결과"""
    timestamp: str
    overall_signal: str  # BULLISH, BEARISH, NEUTRAL
    opportunity: str
    gaps: List[GapResult] = field(default_factory=list)
    market_too_pessimistic: bool = False
    market_too_optimistic: bool = False
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'overall_signal': self.overall_signal,
            'opportunity': self.opportunity,
            'market_too_pessimistic': self.market_too_pessimistic,
            'market_too_optimistic': self.market_too_optimistic,
            'confidence': round(self.confidence, 2),
            'gaps': [
                {
                    'metric': g.metric,
                    'market_implied': round(g.market_implied, 2),
                    'model_forecast': round(g.model_forecast, 2),
                    'gap': round(g.gap, 2),
                    'signal': g.signal,
                    'interpretation': g.interpretation
                }
                for g in self.gaps
            ]
        }


class MarketModelGapAnalyzer:
    """
    Goldman Sachs의 시장-모델 갭 분석기

    시장 가격에 내재된 기대치 vs 모델 예측치 비교.
    괴리 방향에 따라 투자 기회 식별.
    """

    def __init__(self, fred_api_key: str = None):
        self.fred = None
        if FRED_AVAILABLE:
            api_key = fred_api_key or os.environ.get('FRED_API_KEY')
            if api_key:
                try:
                    self.fred = Fred(api_key=api_key)
                except Exception:
                    pass

    def analyze(self, model_forecasts: Dict[str, float] = None) -> GapAnalysisResult:
        """
        갭 분석 실행

        Args:
            model_forecasts: 모델 예측치 (없으면 기본값 사용)
                {
                    'gdp_growth': 2.4,      # 연간 GDP 성장률 %
                    'recession_prob': 0.15, # 침체 확률
                    'fed_rate_1y': 4.0,     # 1년 후 Fed Funds Rate
                    'inflation_1y': 2.5     # 1년 후 인플레이션
                }

        Returns:
            GapAnalysisResult: 종합 분석 결과
        """
        # 기본 모델 예측값 (EIMAS 또는 컨센서스 기반)
        if model_forecasts is None:
            model_forecasts = {
                'gdp_growth': 2.4,       # 견조한 성장 예상
                'recession_prob': 0.15,  # 15% 침체 확률
                'fed_rate_1y': 4.0,      # 1년 후 4%
                'inflation_1y': 2.5      # 2.5% 인플레
            }

        gaps = []

        # 1. Growth Gap
        gaps.append(self._analyze_growth_gap(model_forecasts.get('gdp_growth', 2.4)))

        # 2. Recession Probability Gap
        gaps.append(self._analyze_recession_gap(model_forecasts.get('recession_prob', 0.15)))

        # 3. Fed Rate Path Gap
        gaps.append(self._analyze_fed_gap(model_forecasts.get('fed_rate_1y', 4.0)))

        # 4. Equity Risk Premium
        gaps.append(self._analyze_erp_gap())

        # 종합 판단
        bullish_count = sum(1 for g in gaps if g.signal == 'BULLISH')
        bearish_count = sum(1 for g in gaps if g.signal == 'BEARISH')

        avg_gap = np.mean([abs(g.gap) for g in gaps])

        if bullish_count > bearish_count:
            overall_signal = 'BULLISH'
            market_too_pessimistic = True
            market_too_optimistic = False
            opportunity = "시장이 과도하게 비관적. 위험자산 재평가 가능성."
        elif bearish_count > bullish_count:
            overall_signal = 'BEARISH'
            market_too_pessimistic = False
            market_too_optimistic = True
            opportunity = "시장이 과도하게 낙관적. 방어적 포지션 고려."
        else:
            overall_signal = 'NEUTRAL'
            market_too_pessimistic = False
            market_too_optimistic = False
            opportunity = "시장 가격이 펀더멘털에 부합. 특별한 괴리 없음."

        confidence = min(0.5 + avg_gap * 0.1, 0.9)

        return GapAnalysisResult(
            timestamp=datetime.now().isoformat(),
            overall_signal=overall_signal,
            opportunity=opportunity,
            gaps=gaps,
            market_too_pessimistic=market_too_pessimistic,
            market_too_optimistic=market_too_optimistic,
            confidence=confidence
        )

    def _analyze_growth_gap(self, model_growth: float) -> GapResult:
        """
        성장률 갭 분석

        시장 내재 성장률 추정:
        1. 10Y Treasury - Fed Funds = 성장/인플레 기대
        2. SPY Forward PE → 암묵적 EPS 성장률
        """
        try:
            # Treasury 수익률
            tnx = yf.download('^TNX', period='5d', progress=False)['Close']  # 10Y
            fed_funds = 4.5  # 현재 Fed Funds Rate (하드코딩, FRED 없을 때)

            if self.fred:
                try:
                    ff = self.fred.get_series('FEDFUNDS', limit=1)
                    if len(ff) > 0:
                        fed_funds = float(ff.iloc[-1])
                except Exception:
                    pass

            if len(tnx) > 0:
                tnx_val = float(tnx.iloc[-1].iloc[0]) if hasattr(tnx.iloc[-1], 'iloc') else float(tnx.iloc[-1])

                # 암묵적 성장률 = 10Y - Fed Funds (단순화)
                # 양수면 성장 기대, 음수면 침체 기대 (역전)
                implied_growth = tnx_val - fed_funds + 2.0  # +2% 기본 성장

                gap = model_growth - implied_growth

                if gap > 0.5:
                    signal = 'BULLISH'
                    interpretation = f"시장({implied_growth:.1f}%)이 모델({model_growth:.1f}%)보다 비관적"
                elif gap < -0.5:
                    signal = 'BEARISH'
                    interpretation = f"시장({implied_growth:.1f}%)이 모델({model_growth:.1f}%)보다 낙관적"
                else:
                    signal = 'NEUTRAL'
                    interpretation = f"시장({implied_growth:.1f}%)과 모델({model_growth:.1f}%) 일치"

                return GapResult(
                    metric='gdp_growth',
                    market_implied=implied_growth,
                    model_forecast=model_growth,
                    gap=gap,
                    signal=signal,
                    interpretation=interpretation
                )
        except Exception as e:
            pass

        return GapResult(
            metric='gdp_growth',
            market_implied=2.0,
            model_forecast=model_growth,
            gap=model_growth - 2.0,
            signal='NEUTRAL',
            interpretation="성장률 데이터 불충분"
        )

    def _analyze_recession_gap(self, model_prob: float) -> GapResult:
        """
        침체 확률 갭 분석

        HY 스프레드에서 침체 확률 역산:
        - HY OAS > 500bp: 높은 침체 확률 (~40%+)
        - HY OAS 300-500bp: 중간 (~20-40%)
        - HY OAS < 300bp: 낮음 (~10-20%)
        """
        try:
            # HYG vs LQD 스프레드 프록시
            hyg = yf.download('HYG', period='1mo', progress=False)['Close']
            lqd = yf.download('LQD', period='1mo', progress=False)['Close']

            if len(hyg) > 5 and len(lqd) > 5:
                # HYG 수익률 (배당 제외 단순 price 기반)
                hyg_yield = 5.5  # 대략적 HYG 수익률
                lqd_yield = 4.5  # 대략적 LQD 수익률
                spread = hyg_yield - lqd_yield  # ~1% = 100bp

                # 스프레드 기반 침체 확률 추정 (단순 모델)
                # 실제로는 HY OAS 사용해야 함
                if spread > 2.0:
                    implied_prob = 0.40
                elif spread > 1.5:
                    implied_prob = 0.30
                elif spread > 1.0:
                    implied_prob = 0.20
                else:
                    implied_prob = 0.15

                gap = (model_prob - implied_prob) * 100  # 퍼센트포인트

                if gap < -10:  # 시장이 더 높은 침체 확률 반영
                    signal = 'BULLISH'
                    interpretation = f"시장 침체확률({implied_prob:.0%})이 과대평가. 실제({model_prob:.0%})"
                elif gap > 10:
                    signal = 'BEARISH'
                    interpretation = f"시장 침체확률({implied_prob:.0%})이 과소평가. 실제({model_prob:.0%})"
                else:
                    signal = 'NEUTRAL'
                    interpretation = f"침체확률 일치: 시장({implied_prob:.0%}) vs 모델({model_prob:.0%})"

                return GapResult(
                    metric='recession_probability',
                    market_implied=implied_prob * 100,
                    model_forecast=model_prob * 100,
                    gap=gap,
                    signal=signal,
                    interpretation=interpretation
                )
        except Exception:
            pass

        return GapResult(
            metric='recession_probability',
            market_implied=20.0,
            model_forecast=model_prob * 100,
            gap=(model_prob - 0.20) * 100,
            signal='NEUTRAL',
            interpretation="침체확률 데이터 불충분"
        )

    def _analyze_fed_gap(self, model_rate: float) -> GapResult:
        """
        Fed 금리 경로 갭 분석

        선물 시장에서 암묵적 금리 경로 추출.
        2Y Treasury가 프록시로 사용 가능.
        """
        try:
            # 2Y Treasury
            two_y = yf.download('^IRX', period='5d', progress=False)['Close']  # 13-week T-bill
            tnx = yf.download('^TNX', period='5d', progress=False)['Close']  # 10Y

            # 현재 Fed Funds
            current_ff = 4.5

            if len(two_y) > 0:
                # 단기물 수익률을 1년 후 금리 기대로 사용 (단순화)
                tbill = float(two_y.iloc[-1].iloc[0]) if hasattr(two_y.iloc[-1], 'iloc') else float(two_y.iloc[-1])

                # 시장 암묵적 1년 후 금리
                # T-bill이 낮으면 인하 기대, 높으면 유지/인상 기대
                implied_rate = tbill + (current_ff - tbill) * 0.5  # 점진적 조정

                gap = model_rate - implied_rate

                if gap > 0.25:
                    signal = 'BEARISH'  # 모델이 더 높은 금리 예상 = 채권 약세
                    interpretation = f"시장은 더 많은 인하 기대({implied_rate:.2f}%). 모델({model_rate:.2f}%)"
                elif gap < -0.25:
                    signal = 'BULLISH'  # 모델이 더 낮은 금리 예상 = 채권 강세
                    interpretation = f"시장은 금리 유지 기대({implied_rate:.2f}%). 모델은 인하({model_rate:.2f}%)"
                else:
                    signal = 'NEUTRAL'
                    interpretation = f"금리 경로 일치: 시장({implied_rate:.2f}%) vs 모델({model_rate:.2f}%)"

                return GapResult(
                    metric='fed_rate_1y',
                    market_implied=implied_rate,
                    model_forecast=model_rate,
                    gap=gap,
                    signal=signal,
                    interpretation=interpretation
                )
        except Exception:
            pass

        return GapResult(
            metric='fed_rate_1y',
            market_implied=4.25,
            model_forecast=model_rate,
            gap=model_rate - 4.25,
            signal='NEUTRAL',
            interpretation="금리 데이터 불충분"
        )

    def _analyze_erp_gap(self) -> GapResult:
        """
        Equity Risk Premium (ERP) 갭 분석

        ERP = 주식 기대수익률 - 무위험 수익률
        ERP가 높으면 주식이 저평가 (매수 기회)
        ERP가 낮으면 주식이 고평가 (매도 신호)
        """
        try:
            # SPY Forward Earnings Yield
            spy = yf.Ticker('SPY')
            info = spy.info
            forward_pe = info.get('forwardPE', 20)

            if forward_pe and forward_pe > 0:
                earnings_yield = (1 / forward_pe) * 100  # %

                # 10Y Treasury
                tnx = yf.download('^TNX', period='5d', progress=False)['Close']
                if len(tnx) > 0:
                    risk_free = float(tnx.iloc[-1].iloc[0]) if hasattr(tnx.iloc[-1], 'iloc') else float(tnx.iloc[-1])

                    # ERP = Earnings Yield - Risk Free Rate
                    erp = earnings_yield - risk_free

                    # 역사적 평균 ERP: ~3-4%
                    avg_erp = 3.5

                    gap = erp - avg_erp

                    if gap > 1.0:
                        signal = 'BULLISH'
                        interpretation = f"ERP {erp:.2f}%가 역사적 평균({avg_erp}%)보다 높음. 주식 저평가."
                    elif gap < -1.0:
                        signal = 'BEARISH'
                        interpretation = f"ERP {erp:.2f}%가 역사적 평균({avg_erp}%)보다 낮음. 주식 고평가."
                    else:
                        signal = 'NEUTRAL'
                        interpretation = f"ERP {erp:.2f}%가 정상 범위 (평균 {avg_erp}%)"

                    return GapResult(
                        metric='equity_risk_premium',
                        market_implied=erp,
                        model_forecast=avg_erp,
                        gap=gap,
                        signal=signal,
                        interpretation=interpretation
                    )
        except Exception:
            pass

        return GapResult(
            metric='equity_risk_premium',
            market_implied=3.0,
            model_forecast=3.5,
            gap=-0.5,
            signal='NEUTRAL',
            interpretation="ERP 데이터 불충분"
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Market-Model Gap Analyzer Test")
    print("=" * 60)

    analyzer = MarketModelGapAnalyzer()

    # 모델 예측값 설정 (EIMAS 예측값으로 대체 가능)
    forecasts = {
        'gdp_growth': 2.4,
        'recession_prob': 0.15,
        'fed_rate_1y': 4.0,
        'inflation_1y': 2.5
    }

    result = analyzer.analyze(forecasts)

    print(f"\nOverall Signal: {result.overall_signal}")
    print(f"Opportunity: {result.opportunity}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Market Too Pessimistic: {result.market_too_pessimistic}")
    print(f"Market Too Optimistic: {result.market_too_optimistic}")

    print("\n--- Gap Details ---")
    for g in result.gaps:
        print(f"\n{g.metric}:")
        print(f"  Market Implied: {g.market_implied:.2f}")
        print(f"  Model Forecast: {g.model_forecast:.2f}")
        print(f"  Gap: {g.gap:.2f}")
        print(f"  Signal: {g.signal}")
        print(f"  Interpretation: {g.interpretation}")

    print("\n--- JSON Output ---")
    import json
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
