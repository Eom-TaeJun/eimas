#!/usr/bin/env python3
"""
EIMAS Earnings Calendar
=======================
Track earnings dates, estimates, and surprises.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EarningsEvent:
    ticker: str
    company_name: str
    earnings_date: datetime
    eps_estimate: Optional[float]
    eps_actual: Optional[float]
    surprise_pct: Optional[float]
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    days_until: int


class EarningsCalendar:
    """Track earnings events"""

    def __init__(self):
        self.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                         'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'UNH', 'JNJ', 'WMT']

    def get_earnings_dates(self, tickers: List[str] = None) -> List[EarningsEvent]:
        """Get upcoming earnings dates"""
        tickers = tickers or self.watchlist
        events = []

        print(f"Fetching earnings for {len(tickers)} tickers...")

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Get next earnings date
                earnings_ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
                if earnings_ts:
                    earnings_date = datetime.fromtimestamp(earnings_ts)
                    days_until = (earnings_date - datetime.now()).days

                    # Get estimates from calendar
                    calendar = stock.calendar
                    eps_est = None
                    rev_est = None

                    if calendar is not None and not calendar.empty:
                        if 'Earnings Average' in calendar.index:
                            eps_est = calendar.loc['Earnings Average'].iloc[0] if len(calendar.columns) > 0 else None
                        if 'Revenue Average' in calendar.index:
                            rev_est = calendar.loc['Revenue Average'].iloc[0] if len(calendar.columns) > 0 else None

                    events.append(EarningsEvent(
                        ticker=ticker,
                        company_name=info.get('shortName', ticker),
                        earnings_date=earnings_date,
                        eps_estimate=eps_est,
                        eps_actual=None,
                        surprise_pct=None,
                        revenue_estimate=rev_est,
                        revenue_actual=None,
                        days_until=days_until
                    ))
            except Exception as e:
                continue

        # Sort by date
        events.sort(key=lambda x: x.earnings_date)
        return events

    def get_recent_surprises(self, tickers: List[str] = None) -> List[EarningsEvent]:
        """Get recent earnings surprises"""
        tickers = tickers or self.watchlist
        events = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                earnings = stock.earnings_history

                if earnings is not None and not earnings.empty:
                    latest = earnings.iloc[-1]
                    eps_est = latest.get('epsEstimate', 0)
                    eps_act = latest.get('epsActual', 0)
                    surprise = ((eps_act - eps_est) / abs(eps_est) * 100) if eps_est else 0

                    events.append(EarningsEvent(
                        ticker=ticker,
                        company_name=stock.info.get('shortName', ticker),
                        earnings_date=datetime.now(),  # Most recent
                        eps_estimate=eps_est,
                        eps_actual=eps_act,
                        surprise_pct=surprise,
                        revenue_estimate=None,
                        revenue_actual=None,
                        days_until=0
                    ))
            except Exception:
                continue

        # Sort by surprise magnitude
        events.sort(key=lambda x: abs(x.surprise_pct or 0), reverse=True)
        return events

    def print_calendar(self, events: List[EarningsEvent]):
        """Print earnings calendar"""
        print("\n" + "=" * 70)
        print("EIMAS Earnings Calendar")
        print("=" * 70)

        upcoming = [e for e in events if e.days_until >= 0]
        past = [e for e in events if e.days_until < 0]

        if upcoming:
            print("\nUPCOMING EARNINGS:")
            print("-" * 70)
            print(f"{'Ticker':<8} {'Company':<25} {'Date':<12} {'Days':>6} {'EPS Est':>10}")
            print("-" * 70)

            for e in upcoming[:15]:
                eps_str = f"${e.eps_estimate:.2f}" if e.eps_estimate else "N/A"
                print(f"{e.ticker:<8} {e.company_name[:24]:<25} {e.earnings_date.strftime('%Y-%m-%d'):<12} {e.days_until:>6} {eps_str:>10}")

        print("=" * 70)


if __name__ == "__main__":
    calendar = EarningsCalendar()
    events = calendar.get_earnings_dates()
    calendar.print_calendar(events)
