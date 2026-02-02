#!/usr/bin/env python3
"""
EIMAS Technical Pattern Scanner
===============================
Detect chart patterns and technical setups.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    BULLISH_FLAG = "bullish_flag"
    BEARISH_FLAG = "bearish_flag"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    OVERSOLD_BOUNCE = "oversold_bounce"
    OVERBOUGHT_PULLBACK = "overbought_pullback"


class PatternSignal(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class Pattern:
    ticker: str
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    price: float
    target_price: float
    stop_loss: float
    description: str
    detected_at: datetime


class PatternScanner:
    """Scan for technical patterns"""

    def __init__(self):
        pass

    def fetch_data(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if not df.empty else None
        except Exception:
            return None

    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels"""
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()

        resistance_levels = []
        support_levels = []

        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['Low'].iloc[i])

        return sorted(set(resistance_levels))[-3:], sorted(set(support_levels))[:3]

    def detect_double_top(self, df: pd.DataFrame, ticker: str) -> Optional[Pattern]:
        """Detect double top pattern"""
        highs = df['High'].values[-60:]
        if len(highs) < 60:
            return None

        # Find two peaks
        peak_indices = []
        for i in range(5, len(highs) - 5):
            if highs[i] == max(highs[i-5:i+5]):
                peak_indices.append(i)

        if len(peak_indices) < 2:
            return None

        # Check if two peaks are similar height
        last_two = peak_indices[-2:]
        peak1, peak2 = highs[last_two[0]], highs[last_two[1]]

        if abs(peak1 - peak2) / peak1 < 0.02 and last_two[1] - last_two[0] > 10:
            current_price = float(df['Close'].iloc[-1])
            if current_price < min(peak1, peak2) * 0.98:
                return Pattern(
                    ticker=ticker,
                    pattern_type=PatternType.DOUBLE_TOP,
                    signal=PatternSignal.BEARISH,
                    confidence=0.7,
                    price=current_price,
                    target_price=current_price * 0.95,
                    stop_loss=max(peak1, peak2) * 1.02,
                    description=f"Double top at ${peak1:.2f}, neckline break",
                    detected_at=datetime.now()
                )
        return None

    def detect_double_bottom(self, df: pd.DataFrame, ticker: str) -> Optional[Pattern]:
        """Detect double bottom pattern"""
        lows = df['Low'].values[-60:]
        if len(lows) < 60:
            return None

        trough_indices = []
        for i in range(5, len(lows) - 5):
            if lows[i] == min(lows[i-5:i+5]):
                trough_indices.append(i)

        if len(trough_indices) < 2:
            return None

        last_two = trough_indices[-2:]
        trough1, trough2 = lows[last_two[0]], lows[last_two[1]]

        if abs(trough1 - trough2) / trough1 < 0.02 and last_two[1] - last_two[0] > 10:
            current_price = float(df['Close'].iloc[-1])
            if current_price > max(trough1, trough2) * 1.02:
                return Pattern(
                    ticker=ticker,
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    signal=PatternSignal.BULLISH,
                    confidence=0.7,
                    price=current_price,
                    target_price=current_price * 1.05,
                    stop_loss=min(trough1, trough2) * 0.98,
                    description=f"Double bottom at ${trough1:.2f}, breakout",
                    detected_at=datetime.now()
                )
        return None

    def detect_ma_crossover(self, df: pd.DataFrame, ticker: str) -> Optional[Pattern]:
        """Detect MA crossovers (golden/death cross)"""
        if len(df) < 200:
            return None

        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()

        current_price = float(df['Close'].iloc[-1])

        # Golden cross (50 crosses above 200)
        if sma50.iloc[-2] < sma200.iloc[-2] and sma50.iloc[-1] > sma200.iloc[-1]:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.GOLDEN_CROSS,
                signal=PatternSignal.BULLISH,
                confidence=0.75,
                price=current_price,
                target_price=current_price * 1.10,
                stop_loss=current_price * 0.95,
                description="50-day SMA crossed above 200-day SMA",
                detected_at=datetime.now()
            )

        # Death cross (50 crosses below 200)
        if sma50.iloc[-2] > sma200.iloc[-2] and sma50.iloc[-1] < sma200.iloc[-1]:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.DEATH_CROSS,
                signal=PatternSignal.BEARISH,
                confidence=0.75,
                price=current_price,
                target_price=current_price * 0.90,
                stop_loss=current_price * 1.05,
                description="50-day SMA crossed below 200-day SMA",
                detected_at=datetime.now()
            )
        return None

    def detect_rsi_extremes(self, df: pd.DataFrame, ticker: str) -> Optional[Pattern]:
        """Detect RSI oversold/overbought conditions"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = float(rsi.iloc[-1])
        current_price = float(df['Close'].iloc[-1])

        if current_rsi < 30:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.OVERSOLD_BOUNCE,
                signal=PatternSignal.BULLISH,
                confidence=0.65,
                price=current_price,
                target_price=current_price * 1.05,
                stop_loss=current_price * 0.97,
                description=f"RSI oversold at {current_rsi:.0f}",
                detected_at=datetime.now()
            )

        if current_rsi > 70:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.OVERBOUGHT_PULLBACK,
                signal=PatternSignal.BEARISH,
                confidence=0.65,
                price=current_price,
                target_price=current_price * 0.95,
                stop_loss=current_price * 1.03,
                description=f"RSI overbought at {current_rsi:.0f}",
                detected_at=datetime.now()
            )
        return None

    def detect_breakout(self, df: pd.DataFrame, ticker: str) -> Optional[Pattern]:
        """Detect price breakouts"""
        if len(df) < 50:
            return None

        recent_high = df['High'].iloc[-50:-1].max()
        recent_low = df['Low'].iloc[-50:-1].min()
        current_price = float(df['Close'].iloc[-1])
        avg_volume = df['Volume'].iloc[-20:-1].mean()
        current_volume = float(df['Volume'].iloc[-1])

        # Breakout with volume confirmation
        if current_price > recent_high and current_volume > avg_volume * 1.5:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.BREAKOUT,
                signal=PatternSignal.BULLISH,
                confidence=0.70,
                price=current_price,
                target_price=current_price * 1.08,
                stop_loss=recent_high * 0.98,
                description=f"Breakout above ${recent_high:.2f} on high volume",
                detected_at=datetime.now()
            )

        if current_price < recent_low and current_volume > avg_volume * 1.5:
            return Pattern(
                ticker=ticker,
                pattern_type=PatternType.BREAKDOWN,
                signal=PatternSignal.BEARISH,
                confidence=0.70,
                price=current_price,
                target_price=current_price * 0.92,
                stop_loss=recent_low * 1.02,
                description=f"Breakdown below ${recent_low:.2f} on high volume",
                detected_at=datetime.now()
            )
        return None

    def scan_ticker(self, ticker: str) -> List[Pattern]:
        """Scan a single ticker for all patterns"""
        df = self.fetch_data(ticker)
        if df is None:
            return []

        patterns = []

        # Run all detectors
        detectors = [
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_ma_crossover,
            self.detect_rsi_extremes,
            self.detect_breakout,
        ]

        for detector in detectors:
            pattern = detector(df, ticker)
            if pattern:
                patterns.append(pattern)

        return patterns

    def scan_universe(self, tickers: List[str] = None) -> List[Pattern]:
        """Scan multiple tickers"""
        if tickers is None:
            tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                      'TSLA', 'META', 'AMD', 'JPM', 'BAC', 'XOM', 'GLD', 'TLT']

        print("\n" + "=" * 60)
        print("EIMAS Pattern Scanner")
        print("=" * 60)
        print(f"Scanning {len(tickers)} tickers...")

        all_patterns = []
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] {ticker}...")
            patterns = self.scan_ticker(ticker)
            all_patterns.extend(patterns)

        # Sort by confidence
        all_patterns.sort(key=lambda x: x.confidence, reverse=True)
        return all_patterns

    def print_report(self, patterns: List[Pattern]):
        """Print pattern report"""
        print("\n" + "=" * 80)
        print("Pattern Detection Results")
        print("=" * 80)

        if not patterns:
            print("No patterns detected")
            return

        bullish = [p for p in patterns if p.signal == PatternSignal.BULLISH]
        bearish = [p for p in patterns if p.signal == PatternSignal.BEARISH]

        if bullish:
            print("\nBULLISH PATTERNS:")
            print("-" * 80)
            print(f"{'Ticker':<8} {'Pattern':<25} {'Price':>10} {'Target':>10} {'Stop':>10} {'Conf':>6}")
            print("-" * 80)
            for p in bullish:
                print(f"{p.ticker:<8} {p.pattern_type.value:<25} ${p.price:>8.2f} ${p.target_price:>8.2f} ${p.stop_loss:>8.2f} {p.confidence:>5.0%}")
                print(f"         {p.description}")

        if bearish:
            print("\nBEARISH PATTERNS:")
            print("-" * 80)
            print(f"{'Ticker':<8} {'Pattern':<25} {'Price':>10} {'Target':>10} {'Stop':>10} {'Conf':>6}")
            print("-" * 80)
            for p in bearish:
                print(f"{p.ticker:<8} {p.pattern_type.value:<25} ${p.price:>8.2f} ${p.target_price:>8.2f} ${p.stop_loss:>8.2f} {p.confidence:>5.0%}")
                print(f"         {p.description}")

        print("=" * 80)


if __name__ == "__main__":
    scanner = PatternScanner()
    patterns = scanner.scan_universe()
    scanner.print_report(patterns)
