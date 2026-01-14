#!/usr/bin/env python3
"""
EIMAS Pairs Trading Scanner
===========================
Find cointegrated pairs for statistical arbitrage.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
from scipy import stats


@dataclass
class PairResult:
    ticker1: str
    ticker2: str
    correlation: float
    cointegration_pvalue: float
    is_cointegrated: bool
    hedge_ratio: float
    current_spread: float
    spread_zscore: float
    spread_mean: float
    spread_std: float
    signal: str  # long_spread, short_spread, neutral
    half_life: float  # Mean reversion half-life in days


class PairsScanner:
    """Scan for cointegrated pairs"""

    def __init__(self):
        self.min_correlation = 0.7
        self.coint_threshold = 0.05  # p-value threshold

    def fetch_prices(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """Fetch historical prices"""
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    data[ticker] = df['Close']
            except Exception:
                continue
        return pd.DataFrame(data)

    def engle_granger_test(self, series1: pd.Series, series2: pd.Series) -> Tuple[float, float]:
        """
        Perform Engle-Granger cointegration test
        Returns: (p-value, hedge_ratio)
        """
        try:
            # OLS regression to find hedge ratio
            X = series2.values.reshape(-1, 1)
            y = series1.values
            X_with_const = np.column_stack([np.ones(len(X)), X])

            betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            hedge_ratio = betas[1]

            # Calculate spread
            spread = series1 - hedge_ratio * series2

            # ADF test on spread (simplified)
            # Using a simple unit root test approximation
            spread_diff = spread.diff().dropna()
            spread_lag = spread.shift(1).dropna()

            aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            if len(aligned) < 50:
                return 1.0, hedge_ratio

            y_adf = aligned.iloc[:, 0].values
            x_adf = aligned.iloc[:, 1].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_adf, y_adf)

            # Convert t-stat to p-value (simplified approximation)
            t_stat = slope / std_err if std_err > 0 else 0
            # More negative t-stat = more likely cointegrated
            # Approximate p-value from t-stat
            p_value_approx = 2 * stats.norm.cdf(-abs(t_stat))

            return p_value_approx, hedge_ratio

        except Exception:
            return 1.0, 1.0

    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life"""
        try:
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()

            aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            if len(aligned) < 20:
                return float('inf')

            y = aligned.iloc[:, 0].values
            x = aligned.iloc[:, 1].values

            slope, _, _, _, _ = stats.linregress(x, y)

            if slope >= 0:
                return float('inf')

            half_life = -np.log(2) / slope
            return float(half_life)

        except Exception:
            return float('inf')

    def analyze_pair(self, prices: pd.DataFrame, ticker1: str, ticker2: str) -> Optional[PairResult]:
        """Analyze a potential pair"""
        if ticker1 not in prices.columns or ticker2 not in prices.columns:
            return None

        series1 = prices[ticker1].dropna()
        series2 = prices[ticker2].dropna()

        # Align series
        aligned = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned) < 100:
            return None

        series1 = aligned[ticker1]
        series2 = aligned[ticker2]

        # Calculate correlation
        correlation = float(series1.corr(series2))
        if correlation < self.min_correlation:
            return None

        # Cointegration test
        p_value, hedge_ratio = self.engle_granger_test(series1, series2)
        is_cointegrated = p_value < self.coint_threshold

        # Calculate spread
        spread = series1 - hedge_ratio * series2
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())
        current_spread = float(spread.iloc[-1])
        spread_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Calculate half-life
        half_life = self.calculate_half_life(spread)

        # Generate signal
        if spread_zscore > 2:
            signal = "short_spread"  # Spread too high, expect mean reversion
        elif spread_zscore < -2:
            signal = "long_spread"  # Spread too low, expect mean reversion
        else:
            signal = "neutral"

        return PairResult(
            ticker1=ticker1,
            ticker2=ticker2,
            correlation=correlation,
            cointegration_pvalue=p_value,
            is_cointegrated=is_cointegrated,
            hedge_ratio=hedge_ratio,
            current_spread=current_spread,
            spread_zscore=spread_zscore,
            spread_mean=spread_mean,
            spread_std=spread_std,
            signal=signal,
            half_life=half_life
        )

    def scan_universe(self, tickers: List[str] = None) -> List[PairResult]:
        """Scan for cointegrated pairs"""
        if tickers is None:
            # Default universe - similar sector stocks
            tickers = [
                # Tech
                'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD',
                # Banks
                'JPM', 'BAC', 'C', 'WFC', 'GS',
                # Energy
                'XOM', 'CVX', 'COP', 'SLB',
                # Consumer
                'KO', 'PEP', 'MCD', 'SBUX',
                # Healthcare
                'JNJ', 'PFE', 'MRK', 'ABBV'
            ]

        print("\n" + "=" * 60)
        print("EIMAS Pairs Trading Scanner")
        print("=" * 60)
        print(f"  Scanning {len(tickers)} tickers ({len(list(combinations(tickers, 2)))} pairs)...")

        prices = self.fetch_prices(tickers)
        print(f"  Fetched data for {len(prices.columns)} tickers")

        results = []
        pairs = list(combinations(prices.columns, 2))

        for ticker1, ticker2 in pairs:
            result = self.analyze_pair(prices, ticker1, ticker2)
            if result and result.is_cointegrated:
                results.append(result)

        # Sort by z-score (most extreme first)
        results.sort(key=lambda x: abs(x.spread_zscore), reverse=True)

        print(f"  Found {len(results)} cointegrated pairs")
        return results

    def scan_sector_pairs(self) -> Dict[str, List[PairResult]]:
        """Scan pairs by sector"""
        sectors = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'CRM', 'ORCL'],
            'Banks': ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD'],
            'Consumer': ['KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'UNH']
        }

        results = {}
        for sector, tickers in sectors.items():
            print(f"\nScanning {sector}...")
            sector_results = self.scan_universe(tickers)
            if sector_results:
                results[sector] = sector_results

        return results

    def print_report(self, results: List[PairResult]):
        """Print pairs trading report"""
        print("\n" + "=" * 90)
        print("Pairs Trading Opportunities")
        print("=" * 90)

        if not results:
            print("\nNo cointegrated pairs found with current criteria.")
            return

        print(f"\n{'Pair':<15} {'Corr':>6} {'Coint p':>8} {'Hedge':>7} {'Z-Score':>8} {'Half-Life':>10} {'Signal':<12}")
        print("-" * 90)

        for r in results:
            pair_name = f"{r.ticker1}/{r.ticker2}"
            hl_str = f"{r.half_life:.1f}d" if r.half_life < 100 else ">100d"
            signal_icon = "🟢" if r.signal == "long_spread" else "🔴" if r.signal == "short_spread" else "⚪"
            print(f"{pair_name:<15} {r.correlation:>6.2f} {r.cointegration_pvalue:>8.4f} {r.hedge_ratio:>7.2f} "
                  f"{r.spread_zscore:>+8.2f} {hl_str:>10} {signal_icon} {r.signal:<10}")

        # Trading signals
        actionable = [r for r in results if r.signal != "neutral" and r.half_life < 30]

        if actionable:
            print("\n" + "-" * 90)
            print("ACTIONABLE SIGNALS:")
            print("-" * 90)

            for r in actionable:
                if r.signal == "long_spread":
                    print(f"  📈 LONG {r.ticker1}, SHORT {r.ticker2} (Z={r.spread_zscore:+.2f}, HL={r.half_life:.1f}d)")
                    print(f"      Hedge Ratio: {r.hedge_ratio:.2f} (for every $1 of {r.ticker1}, short ${r.hedge_ratio:.2f} of {r.ticker2})")
                else:
                    print(f"  📉 SHORT {r.ticker1}, LONG {r.ticker2} (Z={r.spread_zscore:+.2f}, HL={r.half_life:.1f}d)")
                    print(f"      Hedge Ratio: {r.hedge_ratio:.2f}")

        print("=" * 90)


if __name__ == "__main__":
    scanner = PairsScanner()

    # Scan specific pairs
    results = scanner.scan_universe()
    scanner.print_report(results)
