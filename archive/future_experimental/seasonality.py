#!/usr/bin/env python3
"""
EIMAS Seasonality Analysis
==========================
Analyze historical seasonal patterns in stocks.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MonthlySeasonality:
    month: int
    month_name: str
    avg_return: float
    win_rate: float
    std_dev: float
    best_year: int
    best_return: float
    worst_year: int
    worst_return: float


@dataclass
class SeasonalPattern:
    ticker: str
    years_analyzed: int
    monthly_patterns: List[MonthlySeasonality]
    best_months: List[int]
    worst_months: List[int]
    current_month_outlook: str
    sell_in_may_effect: bool  # May-Oct vs Nov-Apr
    january_effect: bool
    santa_rally_effect: bool


class SeasonalityAnalyzer:
    """Analyze seasonal patterns in stock returns"""

    def __init__(self):
        self.min_years = 5  # Minimum years of data required

    def fetch_data(self, ticker: str, years: int = 10) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        try:
            df = yf.download(ticker, period=f"{years}y", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if not df.empty else None
        except Exception:
            return None

    def calculate_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns"""
        df = df.copy()
        df['Month'] = df.index.month
        df['Year'] = df.index.year

        # Get month-end prices
        monthly = df.groupby(['Year', 'Month'])['Close'].last()
        monthly_returns = monthly.groupby(level='Year').pct_change() * 100

        return monthly_returns.reset_index()

    def analyze_ticker(self, ticker: str) -> Optional[SeasonalPattern]:
        """Analyze seasonality for a ticker"""
        df = self.fetch_data(ticker)
        if df is None or len(df) < 252 * self.min_years:
            return None

        monthly_returns = self.calculate_monthly_returns(df)
        if monthly_returns.empty:
            return None

        years_analyzed = len(monthly_returns['Year'].unique())

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly_patterns = []
        for month in range(1, 13):
            month_data = monthly_returns[monthly_returns['Month'] == month]['Close'].dropna()

            if len(month_data) < 3:
                continue

            avg_return = float(month_data.mean())
            std_dev = float(month_data.std())
            win_rate = float((month_data > 0).sum() / len(month_data) * 100)

            # Find best and worst years
            month_by_year = monthly_returns[monthly_returns['Month'] == month].set_index('Year')['Close']

            best_idx = month_by_year.idxmax() if not month_by_year.empty else 0
            worst_idx = month_by_year.idxmin() if not month_by_year.empty else 0

            monthly_patterns.append(MonthlySeasonality(
                month=month,
                month_name=month_names[month - 1],
                avg_return=avg_return,
                win_rate=win_rate,
                std_dev=std_dev,
                best_year=int(best_idx),
                best_return=float(month_by_year.max()) if not month_by_year.empty else 0,
                worst_year=int(worst_idx),
                worst_return=float(month_by_year.min()) if not month_by_year.empty else 0
            ))

        # Identify best and worst months
        sorted_by_return = sorted(monthly_patterns, key=lambda x: x.avg_return, reverse=True)
        best_months = [m.month for m in sorted_by_return[:3]]
        worst_months = [m.month for m in sorted_by_return[-3:]]

        # Sell in May effect
        may_oct_returns = [m.avg_return for m in monthly_patterns if m.month in range(5, 11)]
        nov_apr_returns = [m.avg_return for m in monthly_patterns if m.month in [11, 12, 1, 2, 3, 4]]
        sell_in_may = np.mean(nov_apr_returns) > np.mean(may_oct_returns) if may_oct_returns and nov_apr_returns else False

        # January effect
        jan_pattern = next((m for m in monthly_patterns if m.month == 1), None)
        january_effect = jan_pattern and jan_pattern.avg_return > 1.0 and jan_pattern.win_rate > 60

        # Santa rally (last 5 days of Dec + first 2 of Jan)
        dec_pattern = next((m for m in monthly_patterns if m.month == 12), None)
        santa_rally = dec_pattern and dec_pattern.avg_return > 0.5 and dec_pattern.win_rate > 60

        # Current month outlook
        current_month = datetime.now().month
        current_pattern = next((m for m in monthly_patterns if m.month == current_month), None)
        if current_pattern:
            if current_pattern.avg_return > 1.5 and current_pattern.win_rate > 65:
                outlook = "historically_bullish"
            elif current_pattern.avg_return < -0.5 and current_pattern.win_rate < 45:
                outlook = "historically_bearish"
            else:
                outlook = "neutral"
        else:
            outlook = "unknown"

        return SeasonalPattern(
            ticker=ticker,
            years_analyzed=years_analyzed,
            monthly_patterns=monthly_patterns,
            best_months=best_months,
            worst_months=worst_months,
            current_month_outlook=outlook,
            sell_in_may_effect=sell_in_may,
            january_effect=january_effect,
            santa_rally_effect=santa_rally
        )

    def scan_universe(self, tickers: List[str] = None) -> List[SeasonalPattern]:
        """Scan multiple tickers for seasonal patterns"""
        if tickers is None:
            tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

        print("\n" + "=" * 60)
        print("EIMAS Seasonality Analyzer")
        print("=" * 60)

        results = []
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            pattern = self.analyze_ticker(ticker)
            if pattern:
                results.append(pattern)

        return results

    def print_report(self, patterns: List[SeasonalPattern]):
        """Print seasonality report"""
        print("\n" + "=" * 100)
        print("Seasonality Analysis Report")
        print("=" * 100)

        current_month = datetime.now().month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for pattern in patterns:
            print(f"\n{'─' * 50}")
            print(f"📊 {pattern.ticker} ({pattern.years_analyzed} years analyzed)")
            print(f"{'─' * 50}")

            print(f"\n{'Month':<6} {'Avg Ret':>8} {'Win Rate':>10} {'Std Dev':>8} {'Best':>12} {'Worst':>12}")
            print("-" * 60)

            for m in pattern.monthly_patterns:
                highlight = "→ " if m.month == current_month else "  "
                print(f"{highlight}{m.month_name:<4} {m.avg_return:>+7.2f}% {m.win_rate:>9.0f}% {m.std_dev:>7.2f}% "
                      f"{m.best_year}:{m.best_return:>+.1f}% {m.worst_year}:{m.worst_return:>+.1f}%")

            # Effects
            print(f"\n📈 Best Months: {', '.join(month_names[m-1] for m in pattern.best_months)}")
            print(f"📉 Worst Months: {', '.join(month_names[m-1] for m in pattern.worst_months)}")

            print("\n🔍 Seasonal Effects:")
            print(f"   Sell in May Effect: {'✅ Yes' if pattern.sell_in_may_effect else '❌ No'}")
            print(f"   January Effect: {'✅ Yes' if pattern.january_effect else '❌ No'}")
            print(f"   Santa Rally: {'✅ Yes' if pattern.santa_rally_effect else '❌ No'}")

            outlook_icons = {
                "historically_bullish": "🟢 Bullish",
                "historically_bearish": "🔴 Bearish",
                "neutral": "⚪ Neutral",
                "unknown": "❓ Unknown"
            }
            print(f"\n📅 Current Month ({month_names[current_month-1]}) Outlook: {outlook_icons[pattern.current_month_outlook]}")

        # Market-wide summary
        print("\n" + "=" * 100)
        print("MARKET SEASONALITY SUMMARY")
        print("=" * 100)

        spy_pattern = next((p for p in patterns if p.ticker == 'SPY'), None)
        if spy_pattern:
            print("\n📊 S&P 500 (SPY) Historical Monthly Performance:")

            # Create bar chart representation
            for m in spy_pattern.monthly_patterns:
                bar_length = int(m.avg_return * 5)  # Scale for display
                if bar_length > 0:
                    bar = "█" * min(bar_length, 20)
                    print(f"   {m.month_name}: {bar} {m.avg_return:+.2f}%")
                else:
                    bar = "▒" * min(abs(bar_length), 20)
                    print(f"   {m.month_name}: {bar} {m.avg_return:+.2f}%")

        print("=" * 100)


if __name__ == "__main__":
    analyzer = SeasonalityAnalyzer()
    patterns = analyzer.scan_universe()
    analyzer.print_report(patterns)
