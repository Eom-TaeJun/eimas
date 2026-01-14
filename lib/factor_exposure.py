#!/usr/bin/env python3
"""
EIMAS Factor Exposure Analyzer
==============================
Analyze portfolio exposure to systematic factors.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class FactorExposure:
    factor_name: str
    beta: float
    t_stat: float
    r_squared: float
    contribution: float  # % of return explained


@dataclass
class PortfolioFactorProfile:
    ticker: str
    market_beta: float
    size_exposure: float  # SMB
    value_exposure: float  # HML
    momentum_exposure: float  # UMD
    volatility_exposure: float  # Low Vol
    quality_exposure: float  # Profitability
    total_r_squared: float
    factor_attribution: Dict[str, float]


class FactorAnalyzer:
    """Analyze factor exposures using Fama-French style analysis"""

    def __init__(self):
        # Factor proxy ETFs
        self.factor_etfs = {
            'market': 'SPY',      # Market factor
            'size': 'IWM',        # Small cap (proxy for SMB)
            'value': 'IWD',       # Value (proxy for HML)
            'growth': 'IWF',      # Growth
            'momentum': 'MTUM',   # Momentum
            'low_vol': 'USMV',    # Low volatility
            'quality': 'QUAL',    # Quality
            'dividend': 'DVY',    # Dividend
        }

    def fetch_returns(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
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

    def calculate_factor_returns(self, period: str = "2y") -> pd.DataFrame:
        """Calculate factor returns from ETF proxies"""
        print("  Fetching factor ETF data...")
        all_tickers = list(self.factor_etfs.values())
        returns = self.fetch_returns(all_tickers, period)

        # Create factor-mimicking portfolios
        factor_returns = pd.DataFrame()

        # Market excess return (SPY - risk-free, simplified as SPY)
        if 'SPY' in returns.columns:
            factor_returns['MKT'] = returns['SPY']

        # Size (Small - Large): IWM - SPY
        if 'IWM' in returns.columns and 'SPY' in returns.columns:
            factor_returns['SMB'] = returns['IWM'] - returns['SPY']

        # Value (Value - Growth): IWD - IWF
        if 'IWD' in returns.columns and 'IWF' in returns.columns:
            factor_returns['HML'] = returns['IWD'] - returns['IWF']

        # Momentum
        if 'MTUM' in returns.columns and 'SPY' in returns.columns:
            factor_returns['MOM'] = returns['MTUM'] - returns['SPY']

        # Low Volatility
        if 'USMV' in returns.columns and 'SPY' in returns.columns:
            factor_returns['LVOL'] = returns['USMV'] - returns['SPY']

        # Quality
        if 'QUAL' in returns.columns and 'SPY' in returns.columns:
            factor_returns['QUAL'] = returns['QUAL'] - returns['SPY']

        return factor_returns

    def analyze_single_stock(self, ticker: str, factor_returns: pd.DataFrame) -> PortfolioFactorProfile:
        """Analyze factor exposures for a single stock"""
        # Fetch stock returns
        stock_returns = self.fetch_returns([ticker])

        if stock_returns.empty or ticker not in stock_returns.columns:
            return PortfolioFactorProfile(
                ticker=ticker,
                market_beta=1.0,
                size_exposure=0.0,
                value_exposure=0.0,
                momentum_exposure=0.0,
                volatility_exposure=0.0,
                quality_exposure=0.0,
                total_r_squared=0.0,
                factor_attribution={}
            )

        # Align data
        stock = stock_returns[ticker]
        aligned = pd.concat([stock, factor_returns], axis=1).dropna()

        if len(aligned) < 60:  # Need at least 60 observations
            return PortfolioFactorProfile(
                ticker=ticker, market_beta=1.0, size_exposure=0.0,
                value_exposure=0.0, momentum_exposure=0.0,
                volatility_exposure=0.0, quality_exposure=0.0,
                total_r_squared=0.0, factor_attribution={}
            )

        y = aligned[ticker].values
        X = aligned[list(factor_returns.columns)].values

        # Add constant for regression
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # Run OLS regression
        try:
            betas, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

            # Calculate R-squared
            y_pred = X_with_const @ betas
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Extract factor betas (skip constant)
            factor_betas = dict(zip(factor_returns.columns, betas[1:]))

            # Factor attribution (variance contribution)
            factor_var = {}
            for i, factor in enumerate(factor_returns.columns):
                factor_var[factor] = betas[i+1] ** 2 * np.var(X[:, i])

            total_var = sum(factor_var.values()) + np.var(y - y_pred)
            factor_attribution = {k: v / total_var * 100 for k, v in factor_var.items()}

            return PortfolioFactorProfile(
                ticker=ticker,
                market_beta=factor_betas.get('MKT', 1.0),
                size_exposure=factor_betas.get('SMB', 0.0),
                value_exposure=factor_betas.get('HML', 0.0),
                momentum_exposure=factor_betas.get('MOM', 0.0),
                volatility_exposure=factor_betas.get('LVOL', 0.0),
                quality_exposure=factor_betas.get('QUAL', 0.0),
                total_r_squared=r_squared,
                factor_attribution=factor_attribution
            )

        except Exception:
            return PortfolioFactorProfile(
                ticker=ticker, market_beta=1.0, size_exposure=0.0,
                value_exposure=0.0, momentum_exposure=0.0,
                volatility_exposure=0.0, quality_exposure=0.0,
                total_r_squared=0.0, factor_attribution={}
            )

    def analyze_portfolio(self, positions: Dict[str, float]) -> List[PortfolioFactorProfile]:
        """Analyze factor exposures for entire portfolio"""
        print("\n" + "=" * 60)
        print("EIMAS Factor Exposure Analyzer")
        print("=" * 60)

        print("  Calculating factor returns...")
        factor_returns = self.calculate_factor_returns()

        results = []
        for ticker in positions.keys():
            print(f"  Analyzing {ticker}...")
            profile = self.analyze_single_stock(ticker, factor_returns)
            results.append(profile)

        return results

    def calculate_portfolio_factor_exposure(self, positions: Dict[str, float],
                                           profiles: List[PortfolioFactorProfile]) -> Dict[str, float]:
        """Calculate weighted portfolio factor exposure"""
        total_value = sum(positions.values())
        weights = {k: v / total_value for k, v in positions.items()}

        portfolio_exposure = {
            'market_beta': 0.0,
            'size': 0.0,
            'value': 0.0,
            'momentum': 0.0,
            'low_vol': 0.0,
            'quality': 0.0
        }

        for profile in profiles:
            if profile.ticker in weights:
                w = weights[profile.ticker]
                portfolio_exposure['market_beta'] += w * profile.market_beta
                portfolio_exposure['size'] += w * profile.size_exposure
                portfolio_exposure['value'] += w * profile.value_exposure
                portfolio_exposure['momentum'] += w * profile.momentum_exposure
                portfolio_exposure['low_vol'] += w * profile.volatility_exposure
                portfolio_exposure['quality'] += w * profile.quality_exposure

        return portfolio_exposure

    def print_report(self, profiles: List[PortfolioFactorProfile], positions: Dict[str, float] = None):
        """Print factor exposure report"""
        print("\n" + "=" * 80)
        print("Factor Exposure Report")
        print("=" * 80)

        print(f"\n{'Ticker':<8} {'Beta':>8} {'Size':>8} {'Value':>8} {'Mom':>8} {'LowVol':>8} {'Qual':>8} {'R²':>8}")
        print("-" * 80)

        for p in profiles:
            print(f"{p.ticker:<8} {p.market_beta:>8.2f} {p.size_exposure:>+8.2f} "
                  f"{p.value_exposure:>+8.2f} {p.momentum_exposure:>+8.2f} "
                  f"{p.volatility_exposure:>+8.2f} {p.quality_exposure:>+8.2f} {p.total_r_squared:>7.1%}")

        # Portfolio summary
        if positions:
            portfolio_exp = self.calculate_portfolio_factor_exposure(positions, profiles)
            print("\n" + "-" * 80)
            print("PORTFOLIO FACTOR EXPOSURE (Weighted):")
            print("-" * 80)
            print(f"  Market Beta:    {portfolio_exp['market_beta']:>+.2f}")
            print(f"  Size (SMB):     {portfolio_exp['size']:>+.2f}")
            print(f"  Value (HML):    {portfolio_exp['value']:>+.2f}")
            print(f"  Momentum:       {portfolio_exp['momentum']:>+.2f}")
            print(f"  Low Volatility: {portfolio_exp['low_vol']:>+.2f}")
            print(f"  Quality:        {portfolio_exp['quality']:>+.2f}")

            # Interpretation
            print("\n" + "-" * 80)
            print("FACTOR TILT INTERPRETATION:")
            print("-" * 80)

            if portfolio_exp['market_beta'] > 1.1:
                print("  ⚡ High Beta - More volatile than market, aggressive positioning")
            elif portfolio_exp['market_beta'] < 0.9:
                print("  🛡️  Low Beta - Defensive positioning, less market sensitivity")

            if portfolio_exp['size'] > 0.1:
                print("  📈 Small Cap Tilt - Exposure to smaller companies")
            elif portfolio_exp['size'] < -0.1:
                print("  🏢 Large Cap Tilt - Mega-cap focused")

            if portfolio_exp['value'] > 0.1:
                print("  💰 Value Tilt - Cheap stocks relative to fundamentals")
            elif portfolio_exp['value'] < -0.1:
                print("  🚀 Growth Tilt - High growth expectation stocks")

            if portfolio_exp['momentum'] > 0.1:
                print("  📊 Momentum Tilt - Recent winners")

            if portfolio_exp['quality'] > 0.1:
                print("  ⭐ Quality Tilt - High profitability, low leverage")

        print("=" * 80)


if __name__ == "__main__":
    analyzer = FactorAnalyzer()

    # Example portfolio
    positions = {
        'AAPL': 25000,
        'MSFT': 20000,
        'NVDA': 15000,
        'JPM': 10000,
        'XOM': 10000,
        'JNJ': 10000,
        'SPY': 10000
    }

    profiles = analyzer.analyze_portfolio(positions)
    analyzer.print_report(profiles, positions)
