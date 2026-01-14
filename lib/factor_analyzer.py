#!/usr/bin/env python3
"""
EIMAS Factor Analyzer
======================
Fama-French 스타일 팩터 분석

주요 기능:
1. 팩터 익스포저 계산
2. 팩터 기반 성과 분해
3. 스타일 드리프트 감지
4. 팩터 리스크 분석

Usage:
    from lib.factor_analyzer import FactorAnalyzer

    analyzer = FactorAnalyzer(portfolio)
    result = analyzer.analyze()
    print(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

# 팩터 프록시 ETF
FACTOR_PROXIES = {
    'market': 'SPY',      # 시장 팩터
    'size': 'IWM',        # 소형주 (vs SPY 대형주)
    'value': 'IWD',       # 가치 (Russell 1000 Value)
    'growth': 'IWF',      # 성장 (Russell 1000 Growth)
    'momentum': 'MTUM',   # 모멘텀
    'quality': 'QUAL',    # 퀄리티
    'low_vol': 'USMV',    # 저변동성
    'dividend': 'DVY',    # 배당
}

# 스타일 박스 분류
STYLE_BOX = {
    'large_value': ['IVE', 'VTV'],
    'large_blend': ['SPY', 'IVV'],
    'large_growth': ['IVW', 'VUG'],
    'mid_value': ['IWS'],
    'mid_blend': ['IJH'],
    'mid_growth': ['IWP'],
    'small_value': ['IWN'],
    'small_blend': ['IWM'],
    'small_growth': ['IWO'],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FactorExposure:
    """팩터 익스포저"""
    factor: str
    beta: float
    t_stat: float
    p_value: float
    significant: bool
    interpretation: str


@dataclass
class FactorAttribution:
    """팩터 기반 성과 분해"""
    factor: str
    exposure: float
    factor_return: float
    contribution: float
    pct_explained: float


@dataclass
class StyleAnalysis:
    """스타일 분석"""
    style: str
    weight: float
    r_squared: float
    dominant_factors: List[str]


@dataclass
class FactorRisk:
    """팩터 리스크"""
    factor: str
    exposure: float
    factor_volatility: float
    contribution_to_risk: float
    pct_of_total_risk: float


@dataclass
class FactorAnalysisResult:
    """분석 결과"""
    timestamp: datetime
    portfolio_return: float
    factor_exposures: List[FactorExposure]
    factor_attribution: List[FactorAttribution]
    style_analysis: StyleAnalysis
    factor_risks: List[FactorRisk]
    alpha: float
    r_squared: float
    summary: str


# ============================================================================
# Factor Analyzer
# ============================================================================

class FactorAnalyzer:
    """팩터 분석기"""

    def __init__(
        self,
        portfolio_weights: Dict[str, float],
        period: str = '1y',
    ):
        self.portfolio_weights = portfolio_weights
        self.period = period
        self.portfolio_returns: Optional[pd.Series] = None
        self.factor_returns: Optional[pd.DataFrame] = None

    def fetch_data(self) -> bool:
        """데이터 수집"""
        try:
            # 포트폴리오 자산
            portfolio_tickers = list(self.portfolio_weights.keys())

            # 팩터 프록시
            factor_tickers = list(FACTOR_PROXIES.values())

            all_tickers = list(set(portfolio_tickers + factor_tickers))

            df = yf.download(all_tickers, period=self.period, progress=False)['Close']

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            returns = df.pct_change().dropna()

            # 포트폴리오 수익률
            weights = np.array([self.portfolio_weights.get(t, 0)
                              for t in portfolio_tickers if t in returns.columns])
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                return False

            portfolio_cols = [t for t in portfolio_tickers if t in returns.columns]
            self.portfolio_returns = (returns[portfolio_cols] * weights).sum(axis=1)

            # 팩터 수익률
            factor_cols = {name: ticker for name, ticker in FACTOR_PROXIES.items()
                          if ticker in returns.columns}
            self.factor_returns = returns[[t for t in factor_cols.values()]]
            self.factor_returns.columns = [name for name, ticker in FACTOR_PROXIES.items()
                                          if ticker in returns.columns]

            print(f"Loaded {len(portfolio_cols)} assets, {len(self.factor_returns.columns)} factors")
            print(f"Period: {len(returns)} days")

            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def calculate_factor_exposures(self) -> List[FactorExposure]:
        """팩터 익스포저 계산 (회귀 분석)"""
        if self.portfolio_returns is None:
            self.fetch_data()

        exposures = []

        for factor in self.factor_returns.columns:
            # 단일 팩터 회귀
            y = self.portfolio_returns.values
            x = self.factor_returns[factor].values

            # NaN 제거
            mask = ~(np.isnan(y) | np.isnan(x))
            y = y[mask]
            x = x[mask]

            if len(y) < 30:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            t_stat = slope / std_err if std_err > 0 else 0

            significant = p_value < 0.05

            # 해석
            if abs(slope) < 0.3:
                interpretation = "Low exposure"
            elif slope > 0.8:
                interpretation = "High positive exposure"
            elif slope < -0.8:
                interpretation = "High negative exposure"
            elif slope > 0:
                interpretation = "Moderate positive exposure"
            else:
                interpretation = "Moderate negative exposure"

            exposures.append(FactorExposure(
                factor=factor,
                beta=float(slope),
                t_stat=float(t_stat),
                p_value=float(p_value),
                significant=significant,
                interpretation=interpretation,
            ))

        return exposures

    def calculate_multi_factor_model(self) -> Tuple[Dict[str, float], float, float]:
        """다중 팩터 모델"""
        if self.portfolio_returns is None:
            self.fetch_data()

        y = self.portfolio_returns.values

        # 모든 팩터를 독립변수로
        X = self.factor_returns.values

        # NaN 제거
        mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        y = y[mask]
        X = X[mask]

        if len(y) < 30:
            return {}, 0, 0

        # 상수항 추가
        X_with_const = np.column_stack([np.ones(len(y)), X])

        try:
            # OLS 회귀
            betas, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

            alpha = betas[0] * 252  # 연율화 알파
            factor_betas = {col: float(betas[i+1])
                          for i, col in enumerate(self.factor_returns.columns)}

            # R-squared
            y_pred = X_with_const @ betas
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return factor_betas, float(alpha), float(r_squared)

        except Exception as e:
            print(f"Regression error: {e}")
            return {}, 0, 0

    def calculate_factor_attribution(self) -> List[FactorAttribution]:
        """팩터 기반 성과 분해"""
        factor_betas, alpha, r_squared = self.calculate_multi_factor_model()

        if not factor_betas:
            return []

        attributions = []
        total_return = float(self.portfolio_returns.sum())

        for factor in self.factor_returns.columns:
            beta = factor_betas.get(factor, 0)
            factor_ret = float(self.factor_returns[factor].sum())
            contribution = beta * factor_ret

            pct_explained = contribution / total_return * 100 if total_return != 0 else 0

            attributions.append(FactorAttribution(
                factor=factor,
                exposure=beta,
                factor_return=factor_ret,
                contribution=contribution,
                pct_explained=pct_explained,
            ))

        return attributions

    def analyze_style(self) -> StyleAnalysis:
        """스타일 분석"""
        exposures = self.calculate_factor_exposures()

        # 스타일 판단
        market_beta = next((e.beta for e in exposures if e.factor == 'market'), 1.0)
        size_beta = next((e.beta for e in exposures if e.factor == 'size'), 0)
        value_beta = next((e.beta for e in exposures if e.factor == 'value'), 0)
        growth_beta = next((e.beta for e in exposures if e.factor == 'growth'), 0)

        # 사이즈 판단
        if size_beta > 0.3:
            size_style = 'small'
        elif size_beta < -0.3:
            size_style = 'large'
        else:
            size_style = 'mid'

        # 스타일 판단
        if value_beta > growth_beta + 0.2:
            value_style = 'value'
        elif growth_beta > value_beta + 0.2:
            value_style = 'growth'
        else:
            value_style = 'blend'

        style = f"{size_style}_{value_style}"

        # 지배적 팩터
        significant = [e for e in exposures if e.significant]
        dominant = sorted(significant, key=lambda x: -abs(x.beta))[:3]
        dominant_factors = [e.factor for e in dominant]

        _, _, r_squared = self.calculate_multi_factor_model()

        return StyleAnalysis(
            style=style,
            weight=1.0,
            r_squared=r_squared,
            dominant_factors=dominant_factors,
        )

    def calculate_factor_risk(self) -> List[FactorRisk]:
        """팩터 리스크"""
        factor_betas, _, _ = self.calculate_multi_factor_model()

        if not factor_betas:
            return []

        risks = []
        total_risk = 0

        factor_vols = self.factor_returns.std() * np.sqrt(252)

        for factor in self.factor_returns.columns:
            beta = factor_betas.get(factor, 0)
            vol = float(factor_vols[factor])
            contribution = abs(beta) * vol
            total_risk += contribution

            risks.append(FactorRisk(
                factor=factor,
                exposure=beta,
                factor_volatility=vol,
                contribution_to_risk=contribution,
                pct_of_total_risk=0,  # 나중에 계산
            ))

        # 비율 계산
        for r in risks:
            r.pct_of_total_risk = r.contribution_to_risk / total_risk * 100 if total_risk > 0 else 0

        return risks

    def analyze(self) -> FactorAnalysisResult:
        """전체 분석 실행"""
        print(f"\n{'='*50}")
        print("EIMAS Factor Analysis")
        print('='*50)

        if self.portfolio_returns is None:
            self.fetch_data()

        portfolio_return = float(self.portfolio_returns.sum())

        # 팩터 익스포저
        print("Calculating factor exposures...")
        exposures = self.calculate_factor_exposures()

        # 팩터 분해
        print("Calculating factor attribution...")
        attribution = self.calculate_factor_attribution()

        # 스타일 분석
        print("Analyzing style...")
        style = self.analyze_style()

        # 팩터 리스크
        print("Calculating factor risks...")
        factor_risks = self.calculate_factor_risk()

        # 알파 및 R-squared
        _, alpha, r_squared = self.calculate_multi_factor_model()

        # 요약
        summary = self._generate_summary(exposures, attribution, style, alpha, r_squared)

        return FactorAnalysisResult(
            timestamp=datetime.now(),
            portfolio_return=portfolio_return,
            factor_exposures=exposures,
            factor_attribution=attribution,
            style_analysis=style,
            factor_risks=factor_risks,
            alpha=alpha,
            r_squared=r_squared,
            summary=summary,
        )

    def _generate_summary(
        self,
        exposures: List[FactorExposure],
        attribution: List[FactorAttribution],
        style: StyleAnalysis,
        alpha: float,
        r_squared: float,
    ) -> str:
        """요약 생성"""
        lines = [
            f"=== Factor Analysis Summary ===",
            f"Style: {style.style.replace('_', ' ').title()}",
            f"Alpha: {alpha:.2%} (annualized)",
            f"R-squared: {r_squared:.2%}",
            "",
            "=== Significant Factor Exposures ===",
        ]

        for e in sorted(exposures, key=lambda x: -abs(x.beta)):
            sig = "***" if e.significant else ""
            lines.append(f"  {e.factor}: {e.beta:.2f} {sig}")

        lines.append("")
        lines.append("=== Factor Attribution ===")
        for a in sorted(attribution, key=lambda x: -abs(x.contribution)):
            lines.append(f"  {a.factor}: {a.contribution:.2%} ({a.pct_explained:.1f}%)")

        lines.append("")
        lines.append(f"Dominant Factors: {', '.join(style.dominant_factors)}")

        return "\n".join(lines)

    def print_result(self, result: FactorAnalysisResult):
        """결과 출력"""
        print("\n" + result.summary)

        print("\n" + "-"*40)
        print("Factor Risks:")
        for r in sorted(result.factor_risks, key=lambda x: -x.pct_of_total_risk):
            print(f"  {r.factor}: {r.pct_of_total_risk:.1f}% of total risk")

        print("="*50)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_factor_analysis(
    portfolio: Dict[str, float],
) -> FactorAnalysisResult:
    """빠른 팩터 분석"""
    analyzer = FactorAnalyzer(portfolio)
    return analyzer.analyze()


def get_factor_exposures(portfolio: Dict[str, float]) -> Dict[str, float]:
    """팩터 익스포저 조회"""
    analyzer = FactorAnalyzer(portfolio, period='6mo')
    analyzer.fetch_data()
    exposures = analyzer.calculate_factor_exposures()
    return {e.factor: e.beta for e in exposures}


def detect_style_drift(
    portfolio: Dict[str, float],
    periods: List[str] = ['3mo', '6mo', '1y'],
) -> Dict[str, str]:
    """스타일 드리프트 감지"""
    styles = {}
    for period in periods:
        analyzer = FactorAnalyzer(portfolio, period=period)
        analyzer.fetch_data()
        style = analyzer.analyze_style()
        styles[period] = style.style

    return styles


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 테스트 포트폴리오
    portfolio = {
        'SPY': 0.30,
        'QQQ': 0.25,
        'IWM': 0.15,
        'TLT': 0.15,
        'GLD': 0.15,
    }

    analyzer = FactorAnalyzer(portfolio, period='6mo')
    result = analyzer.analyze()
    analyzer.print_result(result)

    # 팩터 익스포저
    print("\n\nFactor Exposures:")
    exposures = get_factor_exposures(portfolio)
    for factor, beta in sorted(exposures.items(), key=lambda x: -abs(x[1])):
        print(f"  {factor}: {beta:.2f}")
