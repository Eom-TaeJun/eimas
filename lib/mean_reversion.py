#!/usr/bin/env python3
"""
EIMAS Mean Reversion Signal Generator
=====================================
Detect mean reversion opportunities in stocks.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class RevertSignal(Enum):
    STRONG_BUY = "strong_buy"      # Extremely oversold
    BUY = "buy"                    # Oversold
    NEUTRAL = "neutral"
    SELL = "sell"                  # Overbought
    STRONG_SELL = "strong_sell"   # Extremely overbought


@dataclass
class MeanReversionSignal:
    ticker: str
    current_price: float
    sma_20: float
    sma_50: float
    sma_200: float
    pct_from_sma20: float
    pct_from_sma50: float
    pct_from_sma200: float
    rsi_14: float
    bollinger_position: float  # -1 to 1 (below lower to above upper)
    zscore_20d: float
    zscore_50d: float
    signal: RevertSignal
    expected_return: float  # Expected mean reversion return
    confidence: float


class MeanReversionScanner:
    """Scan for mean reversion opportunities"""

    def __init__(self):
        self.oversold_threshold = -2.0  # Z-score
        self.overbought_threshold = 2.0
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def fetch_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if not df.empty else None
        except Exception:
            return None

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()

        upper = sma + 2 * std
        lower = sma - 2 * std

        current = prices.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        mid_val = sma.iloc[-1]

        if pd.isna(upper_val) or pd.isna(lower_val):
            return 0.0

        # Normalize to -1 to 1
        band_width = upper_val - lower_val
        if band_width <= 0:
            return 0.0

        position = (current - mid_val) / (band_width / 2)
        return float(np.clip(position, -2, 2))

    def calculate_zscore(self, prices: pd.Series, window: int) -> float:
        """Calculate z-score of current price"""
        mean = prices.rolling(window).mean().iloc[-1]
        std = prices.rolling(window).std().iloc[-1]

        if pd.isna(mean) or pd.isna(std) or std <= 0:
            return 0.0

        return float((prices.iloc[-1] - mean) / std)

    def analyze_ticker(self, ticker: str) -> Optional[MeanReversionSignal]:
        """Analyze mean reversion for a ticker"""
        df = self.fetch_data(ticker)
        if df is None or len(df) < 200:
            return None

        prices = df['Close']
        current = float(prices.iloc[-1])

        # Moving averages
        sma_20 = float(prices.rolling(20).mean().iloc[-1])
        sma_50 = float(prices.rolling(50).mean().iloc[-1])
        sma_200 = float(prices.rolling(200).mean().iloc[-1])

        # Percent from MAs
        pct_from_sma20 = (current / sma_20 - 1) * 100 if sma_20 > 0 else 0
        pct_from_sma50 = (current / sma_50 - 1) * 100 if sma_50 > 0 else 0
        pct_from_sma200 = (current / sma_200 - 1) * 100 if sma_200 > 0 else 0

        # RSI
        rsi = self.calculate_rsi(prices)

        # Bollinger position
        bb_position = self.calculate_bollinger_position(prices)

        # Z-scores
        zscore_20d = self.calculate_zscore(prices, 20)
        zscore_50d = self.calculate_zscore(prices, 50)

        # Generate signal
        signal, expected_return, confidence = self._generate_signal(
            pct_from_sma20, pct_from_sma50, rsi, bb_position, zscore_20d, zscore_50d
        )

        return MeanReversionSignal(
            ticker=ticker,
            current_price=current,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            pct_from_sma20=pct_from_sma20,
            pct_from_sma50=pct_from_sma50,
            pct_from_sma200=pct_from_sma200,
            rsi_14=rsi,
            bollinger_position=bb_position,
            zscore_20d=zscore_20d,
            zscore_50d=zscore_50d,
            signal=signal,
            expected_return=expected_return,
            confidence=confidence
        )

    def _generate_signal(self, pct_sma20: float, pct_sma50: float, rsi: float,
                        bb_pos: float, z20: float, z50: float) -> tuple:
        """Generate trading signal based on indicators"""
        # Score from -5 (very oversold) to +5 (very overbought)
        score = 0

        # Z-score contributions
        score += z20 * 0.8
        score += z50 * 0.5

        # RSI contribution
        if rsi < 30:
            score -= (30 - rsi) / 10
        elif rsi > 70:
            score += (rsi - 70) / 10

        # Bollinger contribution
        score += bb_pos * 0.5

        # Determine signal
        if score < -3:
            signal = RevertSignal.STRONG_BUY
            expected_return = min(abs(score) * 2, 15)  # Up to 15%
            confidence = min(abs(score) / 5, 0.9)
        elif score < -1.5:
            signal = RevertSignal.BUY
            expected_return = min(abs(score) * 1.5, 10)
            confidence = min(abs(score) / 4, 0.7)
        elif score > 3:
            signal = RevertSignal.STRONG_SELL
            expected_return = -min(abs(score) * 2, 15)
            confidence = min(abs(score) / 5, 0.9)
        elif score > 1.5:
            signal = RevertSignal.SELL
            expected_return = -min(abs(score) * 1.5, 10)
            confidence = min(abs(score) / 4, 0.7)
        else:
            signal = RevertSignal.NEUTRAL
            expected_return = 0
            confidence = 0.3

        return signal, expected_return, confidence

    def scan_universe(self, tickers: List[str] = None) -> List[MeanReversionSignal]:
        """Scan for mean reversion opportunities"""
        if tickers is None:
            tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                'JPM', 'BAC', 'WFC', 'GS', 'C',
                'XOM', 'CVX', 'COP',
                'JNJ', 'PFE', 'UNH', 'MRK',
                'DIS', 'NFLX', 'CMCSA',
                'HD', 'LOW', 'TGT', 'WMT',
                'SPY', 'QQQ', 'IWM', 'DIA'
            ]

        print("\n" + "=" * 60)
        print("EIMAS Mean Reversion Scanner")
        print("=" * 60)

        results = []
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            signal = self.analyze_ticker(ticker)
            if signal:
                results.append(signal)

        # Sort by signal strength
        def signal_strength(s):
            strength_map = {
                RevertSignal.STRONG_BUY: -2,
                RevertSignal.BUY: -1,
                RevertSignal.NEUTRAL: 0,
                RevertSignal.SELL: 1,
                RevertSignal.STRONG_SELL: 2
            }
            return (strength_map[s.signal], -s.confidence)

        results.sort(key=signal_strength)
        return results

    def find_oversold(self, tickers: List[str] = None) -> List[MeanReversionSignal]:
        """Find oversold stocks"""
        all_signals = self.scan_universe(tickers)
        return [s for s in all_signals if s.signal in [RevertSignal.BUY, RevertSignal.STRONG_BUY]]

    def find_overbought(self, tickers: List[str] = None) -> List[MeanReversionSignal]:
        """Find overbought stocks"""
        all_signals = self.scan_universe(tickers)
        return [s for s in all_signals if s.signal in [RevertSignal.SELL, RevertSignal.STRONG_SELL]]

    def print_report(self, signals: List[MeanReversionSignal]):
        """Print mean reversion report"""
        print("\n" + "=" * 100)
        print("Mean Reversion Analysis Report")
        print("=" * 100)

        print(f"\n{'Ticker':<8} {'Price':>10} {'%SMA20':>8} {'%SMA50':>8} {'RSI':>6} {'BB Pos':>7} {'Z-20d':>7} {'Signal':<12} {'Exp.Ret':>8}")
        print("-" * 100)

        for s in signals:
            signal_icons = {
                RevertSignal.STRONG_BUY: "🟢🟢",
                RevertSignal.BUY: "🟢",
                RevertSignal.NEUTRAL: "⚪",
                RevertSignal.SELL: "🔴",
                RevertSignal.STRONG_SELL: "🔴🔴"
            }
            icon = signal_icons[s.signal]

            print(f"{s.ticker:<8} ${s.current_price:>8.2f} {s.pct_from_sma20:>+7.1f}% {s.pct_from_sma50:>+7.1f}% "
                  f"{s.rsi_14:>5.0f} {s.bollinger_position:>+6.2f} {s.zscore_20d:>+6.2f} {icon} {s.signal.value:<10} {s.expected_return:>+7.1f}%")

        # Summary
        oversold = [s for s in signals if s.signal in [RevertSignal.BUY, RevertSignal.STRONG_BUY]]
        overbought = [s for s in signals if s.signal in [RevertSignal.SELL, RevertSignal.STRONG_SELL]]

        print("\n" + "-" * 100)
        print("SUMMARY:")
        print("-" * 100)

        if oversold:
            print("\n📈 OVERSOLD (Mean Reversion BUY Candidates):")
            for s in oversold:
                print(f"   {s.ticker}: RSI={s.rsi_14:.0f}, Z-Score={s.zscore_20d:+.2f}, Expected Return={s.expected_return:+.1f}%")

        if overbought:
            print("\n📉 OVERBOUGHT (Mean Reversion SELL Candidates):")
            for s in overbought:
                print(f"   {s.ticker}: RSI={s.rsi_14:.0f}, Z-Score={s.zscore_20d:+.2f}, Expected Return={s.expected_return:+.1f}%")

        print("=" * 100)


if __name__ == "__main__":
    scanner = MeanReversionScanner()
    signals = scanner.scan_universe()
    scanner.print_report(signals)
