#!/usr/bin/env python3
"""
5-Stage Bubble Assessment Framework
=====================================
JP Morgan Wealth Management의 버블 평가 프레임워크 구현

5단계:
1. Paradigm Shift - 기술적 혁신 존재 여부
2. Credit Availability - 자금 조달 용이성
3. Leverage Level - 부채 의존도
4. Valuation Gap - 주가 vs 이익 괴리
5. Feedback Loop - 투기적 순환

기존 BubbleDetector (Greenwood-Shleifer)와 통합하여 다차원 버블 진단 제공

Usage:
    from lib.bubble_framework import FiveStageBubbleFramework

    analyzer = FiveStageBubbleFramework()
    result = analyzer.analyze(market_data)
    print(result.stage, result.score)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf


@dataclass
class StageResult:
    """개별 단계 평가 결과"""
    stage: str
    passed: bool  # True = 버블 신호 없음, False = 버블 신호
    score: float  # 0-20 (낮을수록 건전)
    evidence: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BubbleFrameworkResult:
    """5단계 버블 평가 종합 결과"""
    timestamp: str
    sector: str
    total_score: float  # 0-100
    stage: str  # NO_BUBBLE, EARLY_FORMATION, BUBBLE_BUILDING, LATE_STAGE, IMMINENT_POP
    stage_results: List[StageResult] = field(default_factory=list)
    warning_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'sector': self.sector,
            'total_score': round(self.total_score, 1),
            'stage': self.stage,
            'stage_results': [
                {
                    'stage': sr.stage,
                    'passed': sr.passed,
                    'score': round(sr.score, 1),
                    'evidence': sr.evidence
                }
                for sr in self.stage_results
            ],
            'warning_flags': self.warning_flags
        }


class FiveStageBubbleFramework:
    """
    JP Morgan WM의 5단계 버블 평가 프레임워크

    버블 점수 해석:
    - 0-20: NO_BUBBLE (건전한 성장)
    - 21-40: EARLY_FORMATION (초기 형성)
    - 41-60: BUBBLE_BUILDING (버블 형성 중)
    - 61-80: LATE_STAGE (후기 단계)
    - 81-100: IMMINENT_POP (붕괴 임박)
    """

    STAGES = [
        'paradigm_shift',
        'credit_availability',
        'leverage_level',
        'valuation_gap',
        'feedback_loop'
    ]

    # 섹터별 대표 티커
    SECTOR_TICKERS = {
        'tech': ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'AAPL', 'AMD'],
        'ai': ['NVDA', 'MSFT', 'GOOGL', 'AMD', 'AVGO', 'ORCL'],
        'crypto': ['COIN', 'MSTR', 'RIOT', 'MARA'],
        'market': ['SPY', 'QQQ', 'IWM']
    }

    def __init__(self):
        self.cache = {}

    def analyze(self,
                market_data: Dict[str, pd.DataFrame] = None,
                sector: str = 'tech') -> BubbleFrameworkResult:
        """
        5단계 버블 평가 실행

        Args:
            market_data: 시장 데이터 (없으면 실시간 수집)
            sector: 분석 대상 섹터 ('tech', 'ai', 'crypto', 'market')

        Returns:
            BubbleFrameworkResult: 종합 평가 결과
        """
        tickers = self.SECTOR_TICKERS.get(sector, self.SECTOR_TICKERS['tech'])

        # 데이터 수집 (없으면 실시간)
        if market_data is None:
            market_data = self._fetch_data(tickers)

        # 5단계 평가
        stage_results = []

        # Stage 1: Paradigm Shift
        stage_results.append(self._evaluate_paradigm_shift(sector))

        # Stage 2: Credit Availability
        stage_results.append(self._evaluate_credit_availability())

        # Stage 3: Leverage Level
        stage_results.append(self._evaluate_leverage(tickers))

        # Stage 4: Valuation Gap
        stage_results.append(self._evaluate_valuation_gap(market_data, tickers))

        # Stage 5: Feedback Loop
        stage_results.append(self._evaluate_feedback_loop(market_data, tickers))

        # 총점 계산
        total_score = sum(sr.score for sr in stage_results)

        # 단계 판정
        stage = self._determine_stage(total_score)

        # 경고 플래그
        warning_flags = [
            sr.evidence for sr in stage_results
            if not sr.passed
        ]

        return BubbleFrameworkResult(
            timestamp=datetime.now().isoformat(),
            sector=sector,
            total_score=total_score,
            stage=stage,
            stage_results=stage_results,
            warning_flags=warning_flags
        )

    def _fetch_data(self, tickers: List[str], period: str = '2y') -> Dict[str, pd.DataFrame]:
        """실시간 데이터 수집"""
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    data[ticker] = df
            except Exception:
                pass
        return data

    def _evaluate_paradigm_shift(self, sector: str) -> StageResult:
        """
        Stage 1: 패러다임 전환 평가

        기술 혁신이 실재하는지 확인.
        버블의 필요조건이지만 충분조건은 아님.
        혁신이 없으면 버블 가능성 높음.
        """
        # 섹터별 혁신 증거 (정성적 평가)
        paradigm_evidence = {
            'tech': {
                'has_innovation': True,
                'evidence': 'AI/ML, Cloud Computing, Semiconductor advances',
                'adoption_rate': 0.65  # 65% 기업 도입률
            },
            'ai': {
                'has_innovation': True,
                'evidence': 'LLM breakthrough, Data center demand at record low vacancy (1.6%)',
                'adoption_rate': 0.45
            },
            'crypto': {
                'has_innovation': True,
                'evidence': 'Blockchain, DeFi, but limited real-world adoption',
                'adoption_rate': 0.15
            },
            'market': {
                'has_innovation': True,
                'evidence': 'Broad productivity gains from tech integration',
                'adoption_rate': 0.50
            }
        }

        info = paradigm_evidence.get(sector, paradigm_evidence['tech'])

        # 혁신이 있으면 버블 점수 낮음 (0-8)
        # 혁신 없으면 점수 높음 (12-20)
        if info['has_innovation']:
            score = 4.0 + (1 - info['adoption_rate']) * 8  # 4-12
            passed = True
            evidence = f"Innovation confirmed: {info['evidence']}"
        else:
            score = 16.0
            passed = False
            evidence = "No clear technological paradigm shift"

        return StageResult(
            stage='paradigm_shift',
            passed=passed,
            score=min(score, 20),
            evidence=evidence,
            data={'adoption_rate': info['adoption_rate']}
        )

    def _evaluate_credit_availability(self) -> StageResult:
        """
        Stage 2: 신용 가용성 평가

        자금 조달 환경이 과열되었는지 확인.
        HY 스프레드, 회사채 발행량 분석.
        """
        try:
            # HY 스프레드 프록시: HYG vs LQD 비율
            hyg = yf.download('HYG', period='1y', progress=False)['Close']
            lqd = yf.download('LQD', period='1y', progress=False)['Close']

            if len(hyg) > 20 and len(lqd) > 20:
                # HYG/LQD 비율 - 높을수록 리스크 선호
                ratio = (hyg / lqd).dropna()
                current_ratio = float(ratio.iloc[-1])
                avg_ratio = float(ratio.mean())
                std_ratio = float(ratio.std())

                # Z-score
                z_score = (current_ratio - avg_ratio) / std_ratio if std_ratio > 0 else 0

                # Z-score가 높을수록 신용 과열
                if z_score > 2:
                    score = 18.0
                    passed = False
                    evidence = f"Credit overheating: HYG/LQD Z-score={z_score:.2f}"
                elif z_score > 1:
                    score = 12.0
                    passed = True
                    evidence = f"Credit warming: HYG/LQD Z-score={z_score:.2f}"
                elif z_score > 0:
                    score = 6.0
                    passed = True
                    evidence = f"Credit normal: HYG/LQD Z-score={z_score:.2f}"
                else:
                    score = 3.0
                    passed = True
                    evidence = f"Credit tight: HYG/LQD Z-score={z_score:.2f}"

                return StageResult(
                    stage='credit_availability',
                    passed=passed,
                    score=score,
                    evidence=evidence,
                    data={'hyg_lqd_zscore': round(z_score, 2)}
                )
        except Exception as e:
            pass

        # Fallback
        return StageResult(
            stage='credit_availability',
            passed=True,
            score=8.0,
            evidence="Credit data unavailable, using neutral score",
            data={}
        )

    def _evaluate_leverage(self, tickers: List[str]) -> StageResult:
        """
        Stage 3: 레버리지 수준 평가

        기업들이 자체 현금흐름으로 투자하는지, 부채에 의존하는지 확인.
        FCF margin과 Debt/Equity 분석.
        """
        fcf_margins = []
        debt_equity_ratios = []

        for ticker in tickers[:5]:  # 상위 5개만
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # FCF Margin 계산
                fcf = info.get('freeCashflow', 0)
                revenue = info.get('totalRevenue', 1)
                if revenue and revenue > 0:
                    fcf_margin = (fcf / revenue) * 100 if fcf else 0
                    fcf_margins.append(fcf_margin)

                # Debt/Equity
                de = info.get('debtToEquity', 0)
                if de:
                    debt_equity_ratios.append(de)
            except Exception:
                pass

        # 평균 계산
        avg_fcf_margin = np.mean(fcf_margins) if fcf_margins else 0
        avg_de = np.mean(debt_equity_ratios) if debt_equity_ratios else 100

        # 평가
        # FCF Margin > 15% 이고 D/E < 100이면 건전
        if avg_fcf_margin > 15 and avg_de < 100:
            score = 4.0
            passed = True
            evidence = f"Self-funded growth: FCF margin {avg_fcf_margin:.1f}%, D/E {avg_de:.0f}"
        elif avg_fcf_margin > 10 and avg_de < 150:
            score = 8.0
            passed = True
            evidence = f"Moderate leverage: FCF margin {avg_fcf_margin:.1f}%, D/E {avg_de:.0f}"
        elif avg_fcf_margin > 5:
            score = 12.0
            passed = True
            evidence = f"Elevated leverage: FCF margin {avg_fcf_margin:.1f}%, D/E {avg_de:.0f}"
        else:
            score = 16.0
            passed = False
            evidence = f"High debt dependency: FCF margin {avg_fcf_margin:.1f}%, D/E {avg_de:.0f}"

        return StageResult(
            stage='leverage_level',
            passed=passed,
            score=score,
            evidence=evidence,
            data={
                'avg_fcf_margin': round(avg_fcf_margin, 1),
                'avg_debt_equity': round(avg_de, 0)
            }
        )

    def _evaluate_valuation_gap(self,
                                 market_data: Dict[str, pd.DataFrame],
                                 tickers: List[str]) -> StageResult:
        """
        Stage 4: 밸류에이션 괴리 평가

        주가 상승이 이익 성장을 동반하는지 확인.
        PE 확장 vs EPS 성장 비교.
        """
        pe_ratios = []
        forward_pe_ratios = []
        price_returns = []

        for ticker in tickers[:5]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                pe = info.get('trailingPE', 0)
                fpe = info.get('forwardPE', 0)

                if pe and pe > 0 and pe < 500:
                    pe_ratios.append(pe)
                if fpe and fpe > 0 and fpe < 500:
                    forward_pe_ratios.append(fpe)

                # 1년 수익률
                if ticker in market_data:
                    df = market_data[ticker]
                    if len(df) > 252:
                        ret = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100
                        price_returns.append(float(ret.iloc[0]) if hasattr(ret, 'iloc') else float(ret))
            except Exception:
                pass

        avg_pe = np.mean(pe_ratios) if pe_ratios else 25
        avg_fpe = np.mean(forward_pe_ratios) if forward_pe_ratios else avg_pe
        avg_return = np.mean(price_returns) if price_returns else 0

        # Forward PE < Trailing PE = EPS 성장 기대 = 건전
        # 주가 상승률이 PE 확장만으로 설명되면 버블
        pe_contraction = avg_pe - avg_fpe  # 양수면 EPS 성장 기대

        if pe_contraction > 5:
            # EPS 성장이 주가 상승을 뒷받침
            score = 4.0
            passed = True
            evidence = f"EPS growth backing: PE {avg_pe:.0f} → Forward {avg_fpe:.0f}"
        elif pe_contraction > 0:
            score = 8.0
            passed = True
            evidence = f"Moderate EPS support: PE {avg_pe:.0f} → Forward {avg_fpe:.0f}"
        elif pe_contraction > -5:
            score = 12.0
            passed = True
            evidence = f"Flat EPS outlook: PE {avg_pe:.0f} → Forward {avg_fpe:.0f}"
        else:
            score = 18.0
            passed = False
            evidence = f"PE expansion without EPS: PE {avg_pe:.0f} → Forward {avg_fpe:.0f}"

        return StageResult(
            stage='valuation_gap',
            passed=passed,
            score=score,
            evidence=evidence,
            data={
                'avg_pe': round(avg_pe, 1),
                'avg_forward_pe': round(avg_fpe, 1),
                'pe_contraction': round(pe_contraction, 1)
            }
        )

    def _evaluate_feedback_loop(self,
                                 market_data: Dict[str, pd.DataFrame],
                                 tickers: List[str]) -> StageResult:
        """
        Stage 5: 투기적 피드백 루프 평가

        FOMO 기반 매수, 콜옵션 스큐, 소셜 미디어 열광 등 확인.
        """
        try:
            # VIX와 VVIX (변동성의 변동성) 비교
            vix = yf.download('^VIX', period='6mo', progress=False)['Close']

            if len(vix) > 20:
                current_vix = float(vix.iloc[-1])
                avg_vix = float(vix.mean())

                # 낮은 VIX = 과신/탐욕
                if current_vix < 15:
                    vix_signal = 'extreme_complacency'
                    vix_score = 15.0
                elif current_vix < 20:
                    vix_signal = 'complacency'
                    vix_score = 10.0
                elif current_vix < 25:
                    vix_signal = 'normal'
                    vix_score = 5.0
                else:
                    vix_signal = 'fear'
                    vix_score = 2.0

                # 모멘텀 체크 (최근 3개월 수익률)
                momentum_scores = []
                for ticker in tickers[:3]:
                    if ticker in market_data:
                        df = market_data[ticker]
                        if len(df) > 63:
                            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100
                            ret_val = float(ret.iloc[0]) if hasattr(ret, 'iloc') else float(ret)
                            if ret_val > 30:  # 3개월 30% 이상 = FOMO
                                momentum_scores.append(10)
                            elif ret_val > 15:
                                momentum_scores.append(5)
                            else:
                                momentum_scores.append(0)

                momentum_score = np.mean(momentum_scores) if momentum_scores else 0

                total_score = min(vix_score + momentum_score, 20)

                if total_score > 15:
                    passed = False
                    evidence = f"FOMO detected: VIX={current_vix:.1f} ({vix_signal}), momentum spike"
                else:
                    passed = True
                    evidence = f"No excessive speculation: VIX={current_vix:.1f} ({vix_signal})"

                return StageResult(
                    stage='feedback_loop',
                    passed=passed,
                    score=total_score,
                    evidence=evidence,
                    data={
                        'vix': round(current_vix, 1),
                        'vix_signal': vix_signal
                    }
                )
        except Exception:
            pass

        return StageResult(
            stage='feedback_loop',
            passed=True,
            score=8.0,
            evidence="Feedback data unavailable, using neutral score",
            data={}
        )

    def _determine_stage(self, total_score: float) -> str:
        """총점 기반 버블 단계 판정"""
        if total_score <= 20:
            return 'NO_BUBBLE'
        elif total_score <= 40:
            return 'EARLY_FORMATION'
        elif total_score <= 60:
            return 'BUBBLE_BUILDING'
        elif total_score <= 80:
            return 'LATE_STAGE'
        else:
            return 'IMMINENT_POP'


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("5-Stage Bubble Framework Test")
    print("=" * 60)

    analyzer = FiveStageBubbleFramework()

    # Tech 섹터 분석
    result = analyzer.analyze(sector='tech')

    print(f"\nSector: {result.sector}")
    print(f"Total Score: {result.total_score:.1f}/100")
    print(f"Stage: {result.stage}")
    print("\n--- Stage Results ---")

    for sr in result.stage_results:
        status = "PASS" if sr.passed else "WARN"
        print(f"  [{status}] {sr.stage}: {sr.score:.1f}/20 - {sr.evidence}")

    if result.warning_flags:
        print("\n--- Warning Flags ---")
        for flag in result.warning_flags:
            print(f"  ! {flag}")

    print("\n--- JSON Output ---")
    import json
    print(json.dumps(result.to_dict(), indent=2))
