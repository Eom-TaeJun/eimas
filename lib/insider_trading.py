#!/usr/bin/env python3
"""
EIMAS Insider Trading Tracker
=============================
Track insider buying/selling activity.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class InsiderTransaction:
    ticker: str
    insider_name: str
    title: str
    transaction_type: str  # buy/sell
    shares: int
    price: float
    value: float
    date: datetime
    shares_owned_after: int


@dataclass
class InsiderSummary:
    ticker: str
    buy_count: int
    sell_count: int
    net_shares: int
    net_value: float
    recent_transactions: List[InsiderTransaction]
    signal: str  # bullish/bearish/neutral


class InsiderTracker:
    """Track insider trading activity"""

    def __init__(self):
        pass

    def get_insider_transactions(self, ticker: str) -> List[InsiderTransaction]:
        """Get insider transactions for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            insiders = stock.insider_transactions

            if insiders is None or insiders.empty:
                return []

            transactions = []
            for _, row in insiders.iterrows():
                trans_type = "sell" if "Sale" in str(row.get('Text', '')) else "buy"
                shares = abs(int(row.get('Shares', 0) or 0))
                value = abs(float(row.get('Value', 0) or 0))

                transactions.append(InsiderTransaction(
                    ticker=ticker,
                    insider_name=str(row.get('Insider', 'Unknown')),
                    title=str(row.get('Position', '')),
                    transaction_type=trans_type,
                    shares=shares,
                    price=value / shares if shares > 0 else 0,
                    value=value,
                    date=row.get('Start Date', datetime.now()),
                    shares_owned_after=0
                ))

            return transactions[:20]  # Last 20

        except Exception as e:
            return []

    def analyze_ticker(self, ticker: str) -> InsiderSummary:
        """Analyze insider activity for a ticker"""
        transactions = self.get_insider_transactions(ticker)

        buy_count = sum(1 for t in transactions if t.transaction_type == "buy")
        sell_count = sum(1 for t in transactions if t.transaction_type == "sell")

        buy_shares = sum(t.shares for t in transactions if t.transaction_type == "buy")
        sell_shares = sum(t.shares for t in transactions if t.transaction_type == "sell")
        net_shares = buy_shares - sell_shares

        buy_value = sum(t.value for t in transactions if t.transaction_type == "buy")
        sell_value = sum(t.value for t in transactions if t.transaction_type == "sell")
        net_value = buy_value - sell_value

        # Determine signal
        if buy_count > sell_count * 2 or net_value > 1000000:
            signal = "bullish"
        elif sell_count > buy_count * 2 or net_value < -1000000:
            signal = "bearish"
        else:
            signal = "neutral"

        return InsiderSummary(
            ticker=ticker,
            buy_count=buy_count,
            sell_count=sell_count,
            net_shares=net_shares,
            net_value=net_value,
            recent_transactions=transactions[:5],
            signal=signal
        )

    def scan_universe(self, tickers: List[str] = None) -> List[InsiderSummary]:
        """Scan multiple tickers"""
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'BAC']

        print("\n" + "=" * 60)
        print("EIMAS Insider Trading Tracker")
        print("=" * 60)

        results = []
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            summary = self.analyze_ticker(ticker)
            if summary.buy_count + summary.sell_count > 0:
                results.append(summary)

        # Sort by net value
        results.sort(key=lambda x: x.net_value, reverse=True)
        return results

    def print_report(self, summaries: List[InsiderSummary]):
        """Print insider trading report"""
        print("\n" + "=" * 70)
        print("Insider Trading Report")
        print("=" * 70)

        print(f"\n{'Ticker':<8} {'Buys':>6} {'Sells':>6} {'Net Shares':>12} {'Net Value':>15} {'Signal':>10}")
        print("-" * 70)

        for s in summaries:
            net_val_str = f"${s.net_value/1e6:+.1f}M" if abs(s.net_value) > 1e6 else f"${s.net_value/1e3:+.0f}K"
            print(f"{s.ticker:<8} {s.buy_count:>6} {s.sell_count:>6} {s.net_shares:>+12,} {net_val_str:>15} {s.signal:>10}")

        # Highlight significant activity
        bullish = [s for s in summaries if s.signal == "bullish"]
        bearish = [s for s in summaries if s.signal == "bearish"]

        if bullish:
            print(f"\nInsider BUYING: {', '.join(s.ticker for s in bullish)}")
        if bearish:
            print(f"Insider SELLING: {', '.join(s.ticker for s in bearish)}")

        print("=" * 70)


if __name__ == "__main__":
    tracker = InsiderTracker()
    results = tracker.scan_universe()
    tracker.print_report(results)
