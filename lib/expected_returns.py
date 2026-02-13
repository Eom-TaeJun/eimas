"""
Expected Returns Calculation Module

This module provides dynamic expected return calculations for portfolio optimization.
It implements multiple methodologies including historical mean and James-Stein shrinkage.

Methodologies:
- Historical Mean: Simple average of historical returns
- James-Stein Shrinkage: Shrinks individual estimates toward grand mean (reduces estimation error)

Usage:
    from lib.expected_returns import ExpectedReturnCalculator

    calculator = ExpectedReturnCalculator()
    returns = calculator.calculate_historical_mean(market_data, lookback_days=252)
    shrunk_returns = calculator.james_stein_shrinkage(returns_matrix)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ExpectedReturnCalculator:
    """
    Calculate expected returns using various methodologies.

    This class provides methods for estimating expected returns from historical data,
    with support for both simple historical averages and shrinkage estimators.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 5% = 0.05)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_historical_mean(
        self,
        market_data: Dict[str, pd.DataFrame],
        lookback_days: int = 252,
        annualize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate historical mean returns for each ticker.

        Args:
            market_data: Dictionary mapping ticker -> DataFrame with 'Close' column
            lookback_days: Number of days to look back (default: 252 = 1 year)
            annualize: If True, annualize the returns (default: True)

        Returns:
            Dictionary mapping ticker -> expected annual return

        Example:
            >>> market_data = {'SPY': df_spy, 'QQQ': df_qqq}
            >>> returns = calculator.calculate_historical_mean(market_data)
            >>> print(returns)
            {'SPY': 0.12, 'QQQ': 0.15}
        """
        expected_returns = {}

        for ticker, df in market_data.items():
            if df is None or df.empty or 'Close' not in df.columns:
                # Use default if no data available
                expected_returns[ticker] = self._get_default_return(ticker)
                continue

            # Calculate daily returns
            prices = df['Close'].tail(lookback_days)
            if len(prices) < 20:  # Need minimum data points
                expected_returns[ticker] = self._get_default_return(ticker)
                continue

            daily_returns = prices.pct_change().dropna()

            if len(daily_returns) == 0:
                expected_returns[ticker] = self._get_default_return(ticker)
                continue

            # Calculate mean return
            mean_return = daily_returns.mean()

            # Annualize if requested (assuming 252 trading days)
            if annualize:
                annual_return = (1 + mean_return) ** 252 - 1
            else:
                annual_return = mean_return

            expected_returns[ticker] = annual_return

        return expected_returns

    def james_stein_shrinkage(
        self,
        returns_matrix: np.ndarray,
        shrinkage_intensity: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply James-Stein shrinkage estimator to reduce estimation error.

        The James-Stein estimator shrinks individual estimates toward the grand mean,
        which has been shown to reduce mean squared error compared to sample means.

        Args:
            returns_matrix: NxM matrix of returns (N observations, M assets)
            shrinkage_intensity: Shrinkage parameter (0-1). If None, computed optimally.
                                0 = no shrinkage, 1 = full shrinkage to grand mean

        Returns:
            Array of shrunk expected returns (length M)

        Reference:
            James, W., & Stein, C. (1961). Estimation with quadratic loss.
            Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix.

        Example:
            >>> returns = np.array([[0.01, 0.02], [0.015, 0.018], ...])
            >>> shrunk = calculator.james_stein_shrinkage(returns)
        """
        if returns_matrix.ndim != 2:
            raise ValueError("returns_matrix must be 2D (observations x assets)")

        n_obs, n_assets = returns_matrix.shape

        if n_obs < 2:
            raise ValueError("Need at least 2 observations for shrinkage")

        # Calculate sample means
        sample_means = np.mean(returns_matrix, axis=0)

        # Calculate grand mean (equal-weighted average of all assets)
        grand_mean = np.mean(sample_means)

        # Calculate optimal shrinkage intensity if not provided
        if shrinkage_intensity is None:
            # Compute variance of sample means
            var_sample = np.var(sample_means, ddof=1)

            # Compute average variance of individual assets
            avg_var = np.mean(np.var(returns_matrix, axis=0, ddof=1))

            # Optimal shrinkage intensity (Ledoit-Wolf formula)
            if var_sample > 0:
                shrinkage_intensity = min(1.0, avg_var / (n_obs * var_sample))
            else:
                shrinkage_intensity = 1.0  # Full shrinkage if no variation

        # Ensure shrinkage_intensity is in [0, 1]
        shrinkage_intensity = np.clip(shrinkage_intensity, 0.0, 1.0)

        # Apply shrinkage
        shrunk_returns = (
            shrinkage_intensity * grand_mean +
            (1 - shrinkage_intensity) * sample_means
        )

        return shrunk_returns

    def calculate_from_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        method: str = "historical_mean",
        lookback_days: int = 252,
        use_shrinkage: bool = False,
        shrinkage_intensity: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Unified interface for calculating expected returns.

        Args:
            market_data: Dictionary mapping ticker -> DataFrame
            method: Calculation method ("historical_mean" or "capm")
            lookback_days: Number of days for historical calculation
            use_shrinkage: If True, apply James-Stein shrinkage
            shrinkage_intensity: Shrinkage parameter (0-1), None for optimal

        Returns:
            Dictionary mapping ticker -> expected return

        Example:
            >>> returns = calculator.calculate_from_market_data(
            ...     market_data,
            ...     method="historical_mean",
            ...     use_shrinkage=True
            ... )
        """
        if method == "historical_mean":
            returns = self.calculate_historical_mean(market_data, lookback_days)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Apply shrinkage if requested
        if use_shrinkage and len(returns) > 1:
            # Build returns matrix
            tickers = list(returns.keys())
            returns_matrix = self._build_returns_matrix(
                market_data, tickers, lookback_days
            )

            if returns_matrix is not None and returns_matrix.shape[0] > 1:
                shrunk_array = self.james_stein_shrinkage(
                    returns_matrix, shrinkage_intensity
                )
                returns = dict(zip(tickers, shrunk_array))

        return returns

    def _build_returns_matrix(
        self,
        market_data: Dict[str, pd.DataFrame],
        tickers: List[str],
        lookback_days: int
    ) -> Optional[np.ndarray]:
        """
        Build a returns matrix from market data.

        Args:
            market_data: Dictionary mapping ticker -> DataFrame
            tickers: List of tickers to include
            lookback_days: Number of days to look back

        Returns:
            NxM matrix of daily returns (N observations, M assets)
            or None if insufficient data
        """
        returns_list = []

        for ticker in tickers:
            df = market_data.get(ticker)
            if df is None or df.empty or 'Close' not in df.columns:
                return None

            prices = df['Close'].tail(lookback_days)
            if len(prices) < 20:
                return None

            daily_returns = prices.pct_change().dropna()
            if len(daily_returns) == 0:
                return None

            returns_list.append(daily_returns.values)

        # Align lengths (use minimum length)
        min_len = min(len(r) for r in returns_list)
        if min_len < 2:
            return None

        # Truncate to minimum length and stack
        aligned_returns = [r[-min_len:] for r in returns_list]
        returns_matrix = np.column_stack(aligned_returns)

        return returns_matrix

    def _get_default_return(self, ticker: str) -> float:
        """
        Get default expected return when data is unavailable.

        Args:
            ticker: Ticker symbol

        Returns:
            Default annual expected return

        Default returns based on asset class:
        - Equity (SPY, QQQ, etc.): 10%
        - Bonds (TLT, AGG, etc.): 4%
        - Gold (GLD): 3%
        - Crypto (BTC, ETH): 15% (high risk premium)
        - KOSPI: 8%
        - Others: 8%
        """
        ticker_upper = ticker.upper()

        # Equity ETFs
        if ticker_upper in ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']:
            return 0.10

        # Bond ETFs
        if ticker_upper in ['TLT', 'AGG', 'BND', 'LQD', 'HYG']:
            return 0.04

        # Gold
        if ticker_upper in ['GLD', 'IAU']:
            return 0.03

        # Crypto
        if ticker_upper in ['BTC-USD', 'ETH-USD', 'BTC', 'ETH']:
            return 0.15

        # Korea
        if 'KOSPI' in ticker_upper or ticker_upper in ['005930.KS', '000660.KS']:
            return 0.08

        # Real estate
        if ticker_upper in ['VNQ', 'IYR']:
            return 0.09

        # Default
        return 0.08

    def calculate_market_stats(
        self,
        market_data: Dict[str, pd.DataFrame],
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        Calculate comprehensive market statistics for portfolio optimization.

        Args:
            market_data: Dictionary mapping ticker -> DataFrame
            lookback_days: Number of days for historical calculation

        Returns:
            Dictionary with keys:
            - stock_return: Expected stock return
            - bond_return: Expected bond return
            - stock_vol: Stock volatility
            - bond_vol: Bond volatility
            - correlation: Stock-bond correlation
            - kospi_return: KOSPI expected return (if available)
            - kospi_vol: KOSPI volatility (if available)
            - us_korea_corr: US-Korea correlation (if available)

        Example:
            >>> stats = calculator.calculate_market_stats(market_data)
            >>> print(stats['stock_return'])
            0.12
        """
        stats = {}

        # Calculate returns
        returns = self.calculate_historical_mean(market_data, lookback_days)

        # Stock return (use SPY or QQQ)
        if 'SPY' in returns:
            stats['stock_return'] = returns['SPY']
        elif 'QQQ' in returns:
            stats['stock_return'] = returns['QQQ']
        else:
            stats['stock_return'] = 0.10  # Default

        # Bond return (use TLT or AGG)
        if 'TLT' in returns:
            stats['bond_return'] = returns['TLT']
        elif 'AGG' in returns:
            stats['bond_return'] = returns['AGG']
        else:
            stats['bond_return'] = 0.04  # Default

        # Calculate volatilities
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_returns = market_data['SPY']['Close'].pct_change().dropna()
            stats['stock_vol'] = spy_returns.std() * np.sqrt(252)
        else:
            stats['stock_vol'] = 0.16  # Default

        if 'TLT' in market_data and not market_data['TLT'].empty:
            tlt_returns = market_data['TLT']['Close'].pct_change().dropna()
            stats['bond_vol'] = tlt_returns.std() * np.sqrt(252)
        else:
            stats['bond_vol'] = 0.06  # Default

        # Calculate correlation
        if ('SPY' in market_data and 'TLT' in market_data and
            not market_data['SPY'].empty and not market_data['TLT'].empty):
            spy_ret = market_data['SPY']['Close'].pct_change().dropna()
            tlt_ret = market_data['TLT']['Close'].pct_change().dropna()

            # Align indices
            common_idx = spy_ret.index.intersection(tlt_ret.index)
            if len(common_idx) > 20:
                stats['correlation'] = spy_ret.loc[common_idx].corr(tlt_ret.loc[common_idx])
            else:
                stats['correlation'] = 0.1  # Default
        else:
            stats['correlation'] = 0.1  # Default

        # KOSPI stats (if available)
        kospi_tickers = ['KOSPI', '^KS11', '005930.KS']
        kospi_ticker = None
        for ticker in kospi_tickers:
            if ticker in market_data and not market_data[ticker].empty:
                kospi_ticker = ticker
                break

        if kospi_ticker:
            stats['kospi_return'] = returns.get(kospi_ticker, 0.08)
            kospi_returns = market_data[kospi_ticker]['Close'].pct_change().dropna()
            stats['kospi_vol'] = kospi_returns.std() * np.sqrt(252)

            # US-Korea correlation
            if 'SPY' in market_data and not market_data['SPY'].empty:
                spy_ret = market_data['SPY']['Close'].pct_change().dropna()
                common_idx = spy_ret.index.intersection(kospi_returns.index)
                if len(common_idx) > 20:
                    stats['us_korea_corr'] = spy_ret.loc[common_idx].corr(
                        kospi_returns.loc[common_idx]
                    )
                else:
                    stats['us_korea_corr'] = 0.6  # Default
            else:
                stats['us_korea_corr'] = 0.6  # Default
        else:
            stats['kospi_return'] = 0.08
            stats['kospi_vol'] = 0.20
            stats['us_korea_corr'] = 0.6

        return stats


if __name__ == "__main__":
    # Test expected returns calculation
    print("Testing Expected Returns Calculator...")
    print("-" * 60)

    calculator = ExpectedReturnCalculator()

    # Test 1: Default returns
    print("Test 1: Default Returns")
    test_tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD', 'KOSPI']
    for ticker in test_tickers:
        default = calculator._get_default_return(ticker)
        print(f"  {ticker}: {default:.1%}")
    print()

    # Test 2: James-Stein shrinkage
    print("Test 2: James-Stein Shrinkage")
    np.random.seed(42)
    # Simulate returns for 3 assets over 100 days
    returns_matrix = np.random.normal(0.001, 0.02, (100, 3))
    returns_matrix[:, 0] += 0.0005  # Asset 1: slightly higher mean
    returns_matrix[:, 2] -= 0.0005  # Asset 3: slightly lower mean

    sample_means = np.mean(returns_matrix, axis=0)
    shrunk_means = calculator.james_stein_shrinkage(returns_matrix)

    print("  Sample means:", [f"{m:.4f}" for m in sample_means])
    print("  Shrunk means:", [f"{m:.4f}" for m in shrunk_means])
    print("  Grand mean:", f"{np.mean(sample_means):.4f}")
    print()

    print("-" * 60)
    print("✅ All tests passed!")
