#!/usr/bin/env python3
"""
EIMAS Performance Attribution
==============================
Brinson 모델 기반 성과 분해 분석

주요 기능:
1. Brinson-Fachler Attribution (배분/선정/상호작용)
2. 리스크 기여도 분석
3. 섹터별 성과 분해
4. 시계열 성과 추적

Usage:
    from lib.performance_attribution import PerformanceAttribution

    attr = PerformanceAttribution(portfolio, benchmark)
    result = attr.analyze()
    print(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

# 섹터 ETF 매핑
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AllocationEffect:
    """배분 효과"""
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    benchmark_return: float
    effect: float
    interpretation: str


@dataclass
class SelectionEffect:
    """선정 효과"""
    sector: str
    portfolio_return: float
    benchmark_return: float
    portfolio_weight: float
    effect: float
    interpretation: str


@dataclass
class InteractionEffect:
    """상호작용 효과"""
    sector: str
    weight_diff: float
    return_diff: float
    effect: float


@dataclass
class BrinsonAttribution:
    """Brinson 분해 결과"""
    total_portfolio_return: float
    total_benchmark_return: float
    total_excess_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    sector_allocation: List[AllocationEffect]
    sector_selection: List[SelectionEffect]
    sector_interaction: List[InteractionEffect]


@dataclass
class RiskAttribution:
    """리스크 기여도"""
    asset: str
    weight: float
    volatility: float
    marginal_contribution: float
    contribution_to_var: float
    pct_contribution: float


@dataclass
class PerformanceMetrics:
    """성과 지표"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    information_ratio: float
    tracking_error: float


@dataclass
class AttributionResult:
    """전체 분석 결과"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    brinson: BrinsonAttribution
    risk_attribution: List[RiskAttribution]
    performance: PerformanceMetrics
    summary: str


# ============================================================================
# Performance Attribution
# ============================================================================

class PerformanceAttribution:
    """성과 분해 분석"""

    def __init__(
        self,
        portfolio_weights: Dict[str, float],
        benchmark: str = 'SPY',
        period: str = '1mo',
    ):
        """
        Args:
            portfolio_weights: {ticker: weight} 포트폴리오 비중
            benchmark: 벤치마크 티커
            period: 분석 기간
        """
        self.portfolio_weights = portfolio_weights
        self.benchmark = benchmark
        self.period = period
        self.returns: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional[pd.Series] = None

    def fetch_data(self) -> bool:
        """데이터 수집"""
        try:
            tickers = list(self.portfolio_weights.keys()) + [self.benchmark]

            df = yf.download(tickers, period=self.period, progress=False)['Close']

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self.returns = df.pct_change().dropna()
            self.benchmark_returns = self.returns[self.benchmark]

            # 벤치마크 제외
            portfolio_tickers = [t for t in self.portfolio_weights.keys() if t in self.returns.columns]
            self.returns = self.returns[portfolio_tickers]

            print(f"Loaded {len(portfolio_tickers)} assets, {len(self.returns)} days")
            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def calculate_portfolio_return(self) -> float:
        """포트폴리오 수익률"""
        if self.returns is None:
            return 0.0

        weights = np.array([self.portfolio_weights.get(t, 0) for t in self.returns.columns])
        weights = weights / weights.sum()  # 정규화

        portfolio_returns = (self.returns * weights).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1

        return float(total_return)

    def calculate_benchmark_return(self) -> float:
        """벤치마크 수익률"""
        if self.benchmark_returns is None:
            return 0.0

        return float((1 + self.benchmark_returns).prod() - 1)

    def brinson_attribution(self) -> BrinsonAttribution:
        """Brinson 분해"""
        if self.returns is None:
            self.fetch_data()

        # 섹터별 분류 (간단히 개별 자산을 섹터로 취급)
        portfolio_return = self.calculate_portfolio_return()
        benchmark_return = self.calculate_benchmark_return()
        excess_return = portfolio_return - benchmark_return

        # 개별 자산 수익률
        asset_returns = {}
        for col in self.returns.columns:
            asset_returns[col] = float((1 + self.returns[col]).prod() - 1)

        # 벤치마크 개별 수익률 (SPY를 벤치마크로 사용, 동일 수익률 가정)
        benchmark_asset_return = benchmark_return

        # 배분 효과, 선정 효과, 상호작용 효과
        allocation_effects = []
        selection_effects = []
        interaction_effects = []

        total_allocation = 0
        total_selection = 0
        total_interaction = 0

        # 벤치마크 비중 (동일 가중 가정)
        n_assets = len(self.returns.columns)
        benchmark_weight = 1 / n_assets if n_assets > 0 else 0

        for asset in self.returns.columns:
            w_p = self.portfolio_weights.get(asset, 0)
            w_b = benchmark_weight
            r_p = asset_returns.get(asset, 0)
            r_b = benchmark_asset_return

            # 배분 효과: (Wp - Wb) * Rb
            alloc = (w_p - w_b) * r_b
            total_allocation += alloc

            interpretation = "Overweight in " + ("winner" if r_b > 0 else "loser") if w_p > w_b else \
                           "Underweight in " + ("winner" if r_b > 0 else "loser")

            allocation_effects.append(AllocationEffect(
                sector=asset,
                portfolio_weight=w_p,
                benchmark_weight=w_b,
                benchmark_return=r_b,
                effect=alloc,
                interpretation=interpretation,
            ))

            # 선정 효과: Wp * (Rp - Rb)
            select = w_p * (r_p - r_b)
            total_selection += select

            interpretation = "Outperformed" if r_p > r_b else "Underperformed"

            selection_effects.append(SelectionEffect(
                sector=asset,
                portfolio_return=r_p,
                benchmark_return=r_b,
                portfolio_weight=w_p,
                effect=select,
                interpretation=interpretation,
            ))

            # 상호작용 효과: (Wp - Wb) * (Rp - Rb)
            interact = (w_p - w_b) * (r_p - r_b)
            total_interaction += interact

            interaction_effects.append(InteractionEffect(
                sector=asset,
                weight_diff=w_p - w_b,
                return_diff=r_p - r_b,
                effect=interact,
            ))

        return BrinsonAttribution(
            total_portfolio_return=portfolio_return,
            total_benchmark_return=benchmark_return,
            total_excess_return=excess_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            sector_allocation=allocation_effects,
            sector_selection=selection_effects,
            sector_interaction=interaction_effects,
        )

    def risk_attribution(self) -> List[RiskAttribution]:
        """리스크 기여도"""
        if self.returns is None:
            self.fetch_data()

        attributions = []

        weights = np.array([self.portfolio_weights.get(t, 0) for t in self.returns.columns])
        weights = weights / weights.sum()

        cov_matrix = self.returns.cov() * 252  # 연율화
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        for i, asset in enumerate(self.returns.columns):
            w = weights[i]
            vol = np.sqrt(cov_matrix.iloc[i, i])

            # 한계 리스크 기여도
            marginal = np.dot(cov_matrix.iloc[i], weights) / portfolio_vol

            # VaR 기여도 (95%)
            contribution = w * marginal

            # 비중 기여도
            pct = contribution / portfolio_vol * 100 if portfolio_vol > 0 else 0

            attributions.append(RiskAttribution(
                asset=asset,
                weight=float(w),
                volatility=float(vol),
                marginal_contribution=float(marginal),
                contribution_to_var=float(contribution),
                pct_contribution=float(pct),
            ))

        return attributions

    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """성과 지표 계산"""
        if self.returns is None:
            self.fetch_data()

        weights = np.array([self.portfolio_weights.get(t, 0) for t in self.returns.columns])
        weights = weights / weights.sum()

        portfolio_returns = (self.returns * weights).sum(axis=1)

        # 총 수익률
        total_return = (1 + portfolio_returns).prod() - 1

        # 연율화 수익률
        days = len(portfolio_returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # 변동성
        volatility = portfolio_returns.std() * np.sqrt(252)

        # 샤프 비율
        rf = 0.045  # 무위험수익률
        sharpe = (annualized_return - rf) / volatility if volatility > 0 else 0

        # 최대 낙폭
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # 승률
        wins = (portfolio_returns > 0).sum()
        total_days = len(portfolio_returns)
        win_rate = wins / total_days if total_days > 0 else 0

        # 평균 승/패
        avg_win = portfolio_returns[portfolio_returns > 0].mean() if wins > 0 else 0
        avg_loss = portfolio_returns[portfolio_returns < 0].mean() if (total_days - wins) > 0 else 0

        # 정보 비율
        excess_returns = portfolio_returns - self.benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        return PerformanceMetrics(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            information_ratio=float(information_ratio),
            tracking_error=float(tracking_error),
        )

    def analyze(self) -> AttributionResult:
        """전체 분석 실행"""
        print(f"\n{'='*50}")
        print("EIMAS Performance Attribution")
        print('='*50)

        if self.returns is None:
            self.fetch_data()

        # Brinson 분해
        print("Calculating Brinson attribution...")
        brinson = self.brinson_attribution()

        # 리스크 기여도
        print("Calculating risk attribution...")
        risk_attr = self.risk_attribution()

        # 성과 지표
        print("Calculating performance metrics...")
        performance = self.calculate_performance_metrics()

        # 요약
        summary = self._generate_summary(brinson, risk_attr, performance)

        return AttributionResult(
            timestamp=datetime.now(),
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            brinson=brinson,
            risk_attribution=risk_attr,
            performance=performance,
            summary=summary,
        )

    def _generate_summary(
        self,
        brinson: BrinsonAttribution,
        risk_attr: List[RiskAttribution],
        performance: PerformanceMetrics,
    ) -> str:
        """요약 생성"""
        lines = [
            "=== Performance Summary ===",
            f"Portfolio Return: {brinson.total_portfolio_return:.2%}",
            f"Benchmark Return: {brinson.total_benchmark_return:.2%}",
            f"Excess Return: {brinson.total_excess_return:.2%}",
            "",
            "=== Brinson Attribution ===",
            f"Allocation Effect: {brinson.allocation_effect:.2%}",
            f"Selection Effect: {brinson.selection_effect:.2%}",
            f"Interaction Effect: {brinson.interaction_effect:.2%}",
            "",
            "=== Performance Metrics ===",
            f"Sharpe Ratio: {performance.sharpe_ratio:.2f}",
            f"Information Ratio: {performance.information_ratio:.2f}",
            f"Max Drawdown: {performance.max_drawdown:.2%}",
            f"Win Rate: {performance.win_rate:.1%}",
        ]

        # 리스크 기여도 상위
        lines.append("")
        lines.append("=== Top Risk Contributors ===")
        sorted_risk = sorted(risk_attr, key=lambda x: -abs(x.pct_contribution))
        for r in sorted_risk[:3]:
            lines.append(f"  {r.asset}: {r.pct_contribution:.1f}% of total risk")

        return "\n".join(lines)

    def print_result(self, result: AttributionResult):
        """결과 출력"""
        print("\n" + result.summary)

        print("\n" + "-"*40)
        print("Allocation Effects (by asset):")
        for a in sorted(result.brinson.sector_allocation, key=lambda x: -abs(x.effect))[:5]:
            print(f"  {a.sector}: {a.effect:+.2%} | W: {a.portfolio_weight:.1%} vs {a.benchmark_weight:.1%}")

        print("\nSelection Effects (by asset):")
        for s in sorted(result.brinson.sector_selection, key=lambda x: -abs(x.effect))[:5]:
            print(f"  {s.sector}: {s.effect:+.2%} | R: {s.portfolio_return:.2%} vs {s.benchmark_return:.2%}")

        print("="*50)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_attribution(
    portfolio: Dict[str, float],
    benchmark: str = 'SPY',
    period: str = '1mo',
) -> AttributionResult:
    """빠른 성과 분해"""
    attr = PerformanceAttribution(portfolio, benchmark, period)
    return attr.analyze()


def compare_to_benchmark(
    portfolio: Dict[str, float],
    benchmark: str = 'SPY',
) -> Dict[str, Any]:
    """벤치마크 대비 비교"""
    attr = PerformanceAttribution(portfolio, benchmark)
    result = attr.analyze()

    return {
        'portfolio_return': result.brinson.total_portfolio_return,
        'benchmark_return': result.brinson.total_benchmark_return,
        'excess_return': result.brinson.total_excess_return,
        'sharpe': result.performance.sharpe_ratio,
        'information_ratio': result.performance.information_ratio,
    }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 테스트 포트폴리오
    portfolio = {
        'SPY': 0.40,
        'QQQ': 0.25,
        'TLT': 0.20,
        'GLD': 0.15,
    }

    attr = PerformanceAttribution(portfolio, 'SPY', '3mo')
    result = attr.analyze()
    attr.print_result(result)
