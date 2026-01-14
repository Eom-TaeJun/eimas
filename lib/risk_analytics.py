#!/usr/bin/env python3
"""
EIMAS Portfolio Risk Analytics
==============================
VaR, drawdown, correlation, and risk metrics.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class RiskMetrics:
    portfolio_value: float
    daily_var_95: float
    daily_var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_to_spy: float


@dataclass
class PositionRisk:
    ticker: str
    weight: float
    volatility: float
    var_contribution: float
    beta: float
    correlation: float


class RiskAnalyzer:
    """Portfolio risk analytics"""

    def __init__(self):
        self.risk_free_rate = 0.05  # 5% annual

    def fetch_returns(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch historical returns"""
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    data[ticker] = df['Close'].pct_change().dropna()
            except Exception:
                continue
        return pd.DataFrame(data)

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self.calculate_var(returns, confidence)
        return float(returns[returns <= var].mean())

    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate max and current drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])
        return max_dd, current_dd

    def calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))

    def calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return float(excess_returns.mean() / downside_returns.std() * np.sqrt(252))

    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta"""
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0
        cov = aligned.cov().iloc[0, 1]
        var = benchmark_returns.var()
        return float(cov / var) if var > 0 else 1.0

    def analyze_portfolio(self, positions: Dict[str, float], portfolio_value: float = 100000) -> RiskMetrics:
        """Analyze portfolio risk"""
        print("\n" + "=" * 60)
        print("EIMAS Risk Analytics")
        print("=" * 60)

        tickers = list(positions.keys()) + ['SPY']
        returns_df = self.fetch_returns(tickers)

        if returns_df.empty:
            raise ValueError("Could not fetch data")

        spy_returns = returns_df.get('SPY', pd.Series())

        # Portfolio returns
        weights = np.array([positions.get(t, 0) for t in returns_df.columns if t != 'SPY'])
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        asset_returns = returns_df[[t for t in returns_df.columns if t != 'SPY']]
        portfolio_returns = (asset_returns * weights).sum(axis=1)

        # Fetch prices for drawdown
        spy_data = yf.download('SPY', period='1y', progress=False)
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)

        # Calculate metrics
        var_95 = self.calculate_var(portfolio_returns) * portfolio_value
        var_99 = self.calculate_var(portfolio_returns, 0.99) * portfolio_value
        es = self.calculate_expected_shortfall(portfolio_returns) * portfolio_value
        max_dd, current_dd = self.calculate_max_drawdown(spy_data['Close'])
        vol = float(portfolio_returns.std() * np.sqrt(252))
        sharpe = self.calculate_sharpe(portfolio_returns)
        sortino = self.calculate_sortino(portfolio_returns)
        beta = self.calculate_beta(portfolio_returns, spy_returns)
        corr = float(portfolio_returns.corr(spy_returns)) if not spy_returns.empty else 0

        return RiskMetrics(
            portfolio_value=portfolio_value,
            daily_var_95=abs(var_95),
            daily_var_99=abs(var_99),
            expected_shortfall=abs(es),
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            beta=beta,
            correlation_to_spy=corr
        )

    def analyze_position_risk(self, positions: Dict[str, float]) -> List[PositionRisk]:
        """Analyze risk by position"""
        tickers = list(positions.keys()) + ['SPY']
        returns_df = self.fetch_returns(tickers)

        if returns_df.empty:
            return []

        spy_returns = returns_df.get('SPY', pd.Series())
        total_weight = sum(positions.values())

        results = []
        for ticker, amount in positions.items():
            if ticker not in returns_df.columns:
                continue

            returns = returns_df[ticker]
            weight = amount / total_weight if total_weight > 0 else 0
            vol = float(returns.std() * np.sqrt(252))
            var_contrib = self.calculate_var(returns) * amount
            beta = self.calculate_beta(returns, spy_returns)
            corr = float(returns.corr(spy_returns)) if not spy_returns.empty else 0

            results.append(PositionRisk(
                ticker=ticker, weight=weight, volatility=vol,
                var_contribution=abs(var_contrib), beta=beta, correlation=corr
            ))

        results.sort(key=lambda x: x.var_contribution, reverse=True)
        return results

    def print_report(self, metrics: RiskMetrics, position_risks: List[PositionRisk]):
        """Print risk report"""
        print("\n" + "=" * 60)
        print("Portfolio Risk Report")
        print("=" * 60)

        print(f"\nPortfolio Value:      ${metrics.portfolio_value:,.2f}")
        print(f"\nVALUE AT RISK:")
        print(f"  Daily VaR (95%):    ${metrics.daily_var_95:,.2f}")
        print(f"  Daily VaR (99%):    ${metrics.daily_var_99:,.2f}")
        print(f"  Expected Shortfall: ${metrics.expected_shortfall:,.2f}")

        print(f"\nDRAWDOWN:")
        print(f"  Max Drawdown:       {metrics.max_drawdown:.1%}")
        print(f"  Current Drawdown:   {metrics.current_drawdown:.1%}")

        print(f"\nRISK-ADJUSTED RETURNS:")
        print(f"  Volatility:         {metrics.volatility:.1%}")
        print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:      {metrics.sortino_ratio:.2f}")

        print(f"\nMARKET EXPOSURE:")
        print(f"  Beta:               {metrics.beta:.2f}")
        print(f"  Correlation (SPY):  {metrics.correlation_to_spy:.2f}")

        if position_risks:
            print("\n" + "-" * 60)
            print("POSITION RISK CONTRIBUTION")
            print("-" * 60)
            print(f"{'Ticker':<8} {'Weight':>8} {'Vol':>8} {'Beta':>6} {'VaR Contrib':>12}")

            for pr in position_risks:
                print(f"{pr.ticker:<8} {pr.weight:>7.1%} {pr.volatility:>7.1%} {pr.beta:>6.2f} ${pr.var_contribution:>10,.0f}")

        print("=" * 60)


if __name__ == "__main__":
    analyzer = RiskAnalyzer()

    # Example portfolio
    positions = {
        'AAPL': 20000,
        'MSFT': 15000,
        'GOOGL': 15000,
        'NVDA': 10000,
        'SPY': 40000
    }

    metrics = analyzer.analyze_portfolio(positions, sum(positions.values()))
    position_risks = analyzer.analyze_position_risk(positions)
    analyzer.print_report(metrics, position_risks)
