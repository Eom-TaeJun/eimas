#!/usr/bin/env python3
"""
EIMAS Options Flow Monitor
==========================
Monitor unusual options activity and large trades.

Usage:
    from lib.options_flow import OptionsFlowMonitor

    monitor = OptionsFlowMonitor()
    flows = monitor.scan_unusual_activity(['AAPL', 'TSLA', 'SPY'])
    monitor.print_report(flows)
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


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class FlowSignal(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class OptionFlow:
    """Single options flow entry"""
    ticker: str
    option_type: OptionType
    strike: float
    expiry: str
    volume: int
    open_interest: int
    implied_volatility: float
    volume_oi_ratio: float
    premium_estimate: float
    signal: FlowSignal
    reasoning: str


@dataclass
class OptionsFlowSummary:
    """Summary of options flow for a ticker"""
    ticker: str
    stock_price: float
    total_call_volume: int
    total_put_volume: int
    put_call_ratio: float
    unusual_flows: List[OptionFlow]
    overall_signal: FlowSignal
    bullish_premium: float
    bearish_premium: float
    max_pain: float
    iv_percentile: float


class OptionsFlowMonitor:
    """Monitor unusual options activity"""

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_options_chain(self, ticker: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, float]]:
        """Fetch options chain for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get('regularMarketPrice') or stock.info.get('previousClose', 0)

            expirations = stock.options
            if not expirations:
                return None

            all_calls = []
            all_puts = []

            for exp in expirations[:4]:
                try:
                    opt = stock.option_chain(exp)
                    calls = opt.calls.copy()
                    puts = opt.puts.copy()
                    calls['expiry'] = exp
                    puts['expiry'] = exp
                    all_calls.append(calls)
                    all_puts.append(puts)
                except Exception:
                    continue

            if not all_calls:
                return None

            calls_df = pd.concat(all_calls, ignore_index=True)
            puts_df = pd.concat(all_puts, ignore_index=True)

            return calls_df, puts_df, price

        except Exception as e:
            print(f"  Error fetching options for {ticker}: {e}")
            return None

    def find_unusual_activity(self, df: pd.DataFrame, option_type: OptionType,
                              stock_price: float, volume_threshold: int = 1000,
                              vol_oi_threshold: float = 0.5) -> List[OptionFlow]:
        """Find unusual options activity"""
        unusual = []
        ticker = df.attrs.get('ticker', 'UNK')

        for _, row in df.iterrows():
            volume = int(row.get('volume', 0) or 0)
            oi = int(row.get('openInterest', 0) or 0)
            strike = float(row.get('strike', 0))
            iv = float(row.get('impliedVolatility', 0) or 0)
            last_price = float(row.get('lastPrice', 0) or 0)
            expiry = str(row.get('expiry', ''))

            if volume < volume_threshold:
                continue

            vol_oi = volume / oi if oi > 0 else volume

            if vol_oi < vol_oi_threshold and volume < 5000:
                continue

            premium = volume * last_price * 100

            if option_type == OptionType.CALL:
                if strike > stock_price * 1.05:
                    signal = FlowSignal.BULLISH
                    reasoning = f"Large OTM call volume at ${strike:.0f}"
                else:
                    signal = FlowSignal.BULLISH
                    reasoning = f"Call activity at ${strike:.0f}"
            else:
                if strike < stock_price * 0.95:
                    signal = FlowSignal.BEARISH
                    reasoning = f"Large OTM put volume at ${strike:.0f}"
                else:
                    signal = FlowSignal.BEARISH
                    reasoning = f"Put activity at ${strike:.0f}"

            if vol_oi > 2:
                reasoning += f" [Vol/OI: {vol_oi:.1f}x - Very unusual]"
            elif vol_oi > 1:
                reasoning += f" [Vol/OI: {vol_oi:.1f}x]"

            unusual.append(OptionFlow(
                ticker=ticker, option_type=option_type, strike=strike,
                expiry=expiry, volume=volume, open_interest=oi,
                implied_volatility=iv, volume_oi_ratio=vol_oi,
                premium_estimate=premium, signal=signal, reasoning=reasoning
            ))

        unusual.sort(key=lambda x: x.premium_estimate, reverse=True)
        return unusual[:10]

    def calculate_max_pain(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
        """Calculate max pain strike"""
        try:
            strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
            if not strikes:
                return 0

            min_pain = float('inf')
            max_pain_strike = strikes[len(strikes)//2]

            for strike in strikes:
                call_pain = sum((strike - row['strike']) * row.get('openInterest', 0)
                               for _, row in calls_df.iterrows() if row['strike'] < strike)
                put_pain = sum((row['strike'] - strike) * row.get('openInterest', 0)
                              for _, row in puts_df.iterrows() if row['strike'] > strike)

                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike

            return max_pain_strike
        except Exception:
            return 0

    def analyze_ticker(self, ticker: str) -> Optional[OptionsFlowSummary]:
        """Analyze options flow for a single ticker"""
        result = self.get_options_chain(ticker)
        if result is None:
            return None

        calls_df, puts_df, stock_price = result
        calls_df.attrs['ticker'] = ticker
        puts_df.attrs['ticker'] = ticker

        total_call_vol = int(calls_df['volume'].sum() or 0)
        total_put_vol = int(puts_df['volume'].sum() or 0)
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1

        unusual_calls = self.find_unusual_activity(calls_df, OptionType.CALL, stock_price)
        unusual_puts = self.find_unusual_activity(puts_df, OptionType.PUT, stock_price)
        all_unusual = unusual_calls + unusual_puts

        bullish_premium = sum(f.premium_estimate for f in all_unusual if f.signal == FlowSignal.BULLISH)
        bearish_premium = sum(f.premium_estimate for f in all_unusual if f.signal == FlowSignal.BEARISH)

        if bullish_premium > bearish_premium * 1.5:
            overall_signal = FlowSignal.BULLISH
        elif bearish_premium > bullish_premium * 1.5:
            overall_signal = FlowSignal.BEARISH
        else:
            overall_signal = FlowSignal.NEUTRAL

        avg_iv = calls_df['impliedVolatility'].mean() if not calls_df.empty else 0
        iv_percentile = min(100, avg_iv * 200)

        max_pain = self.calculate_max_pain(calls_df, puts_df)

        return OptionsFlowSummary(
            ticker=ticker, stock_price=stock_price,
            total_call_volume=total_call_vol, total_put_volume=total_put_vol,
            put_call_ratio=pc_ratio, unusual_flows=all_unusual,
            overall_signal=overall_signal, bullish_premium=bullish_premium,
            bearish_premium=bearish_premium, max_pain=max_pain, iv_percentile=iv_percentile
        )

    def scan_unusual_activity(self, tickers: List[str] = None) -> List[OptionsFlowSummary]:
        """Scan multiple tickers for unusual options activity"""
        if tickers is None:
            tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'GOOGL']

        print("\n" + "=" * 60)
        print("EIMAS Options Flow Monitor")
        print("=" * 60)
        print(f"Scanning {len(tickers)} tickers...")

        results = []
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] {ticker}...")
            summary = self.analyze_ticker(ticker)
            if summary:
                results.append(summary)

        results.sort(key=lambda x: x.bullish_premium + x.bearish_premium, reverse=True)
        return results

    def print_report(self, summaries: List[OptionsFlowSummary]):
        """Print options flow report"""
        print("\n" + "=" * 70)
        print("Options Flow Summary")
        print("=" * 70)

        print(f"\n{'Ticker':<8} {'Price':>10} {'P/C Ratio':>10} {'Signal':>10} {'Max Pain':>10}")
        print("-" * 50)

        for s in summaries:
            print(f"{s.ticker:<8} ${s.stock_price:>8.2f} {s.put_call_ratio:>10.2f} {s.overall_signal.value:>10} ${s.max_pain:>8.0f}")

        print("\n" + "-" * 70)
        print("TOP UNUSUAL FLOWS")
        print("-" * 70)

        for s in summaries:
            if s.unusual_flows:
                for flow in s.unusual_flows[:3]:
                    type_str = "CALL" if flow.option_type == OptionType.CALL else "PUT"
                    premium_str = f"${flow.premium_estimate/1e6:.1f}M" if flow.premium_estimate > 1e6 else f"${flow.premium_estimate/1e3:.0f}K"
                    print(f"  {flow.ticker:<6} {type_str:<5} ${flow.strike:<8.0f} {flow.expiry:<12} {premium_str:<10} {flow.signal.value}")

        print("=" * 70)


if __name__ == "__main__":
    monitor = OptionsFlowMonitor()
    results = monitor.scan_unusual_activity(['SPY', 'QQQ', 'AAPL', 'TSLA'])
    monitor.print_report(results)
