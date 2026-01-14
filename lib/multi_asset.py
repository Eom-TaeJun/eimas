#!/usr/bin/env python3
"""
EIMAS Multi-Asset Analyzer
==========================
Analyze individual stocks, sectors, and asset classes.

Usage:
    from lib.multi_asset import MultiAssetAnalyzer

    analyzer = MultiAssetAnalyzer()
    results = analyzer.analyze_universe()
    analyzer.print_report(results)
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
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# Asset Definitions
# ============================================================================

class AssetClass(Enum):
    EQUITY = "equity"
    SECTOR = "sector"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"


class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# Sector ETFs
SECTOR_ETFS = {
    'XLK': ('Technology', AssetClass.SECTOR),
    'XLF': ('Financials', AssetClass.SECTOR),
    'XLE': ('Energy', AssetClass.SECTOR),
    'XLV': ('Healthcare', AssetClass.SECTOR),
    'XLI': ('Industrials', AssetClass.SECTOR),
    'XLY': ('Consumer Discretionary', AssetClass.SECTOR),
    'XLP': ('Consumer Staples', AssetClass.SECTOR),
    'XLU': ('Utilities', AssetClass.SECTOR),
    'XLB': ('Materials', AssetClass.SECTOR),
    'XLRE': ('Real Estate', AssetClass.SECTOR),
    'XLC': ('Communication Services', AssetClass.SECTOR),
}

# Major stocks by sector
MAJOR_STOCKS = {
    # Technology
    'AAPL': ('Apple', 'XLK'),
    'MSFT': ('Microsoft', 'XLK'),
    'NVDA': ('NVIDIA', 'XLK'),
    'GOOGL': ('Alphabet', 'XLC'),
    'META': ('Meta', 'XLC'),
    'AMZN': ('Amazon', 'XLY'),
    'TSLA': ('Tesla', 'XLY'),
    # Financials
    'JPM': ('JP Morgan', 'XLF'),
    'BAC': ('Bank of America', 'XLF'),
    'GS': ('Goldman Sachs', 'XLF'),
    # Energy
    'XOM': ('Exxon Mobil', 'XLE'),
    'CVX': ('Chevron', 'XLE'),
    # Healthcare
    'UNH': ('UnitedHealth', 'XLV'),
    'JNJ': ('Johnson & Johnson', 'XLV'),
    'LLY': ('Eli Lilly', 'XLV'),
    # Industrials
    'CAT': ('Caterpillar', 'XLI'),
    'BA': ('Boeing', 'XLI'),
}

# Other asset classes
OTHER_ASSETS = {
    'GLD': ('Gold', AssetClass.COMMODITY),
    'SLV': ('Silver', AssetClass.COMMODITY),
    'USO': ('Oil', AssetClass.COMMODITY),
    'TLT': ('20+ Year Treasury', AssetClass.BOND),
    'IEF': ('7-10 Year Treasury', AssetClass.BOND),
    'HYG': ('High Yield Bonds', AssetClass.BOND),
    'LQD': ('Investment Grade Bonds', AssetClass.BOND),
    'UUP': ('US Dollar', AssetClass.CURRENCY),
    'FXE': ('Euro', AssetClass.CURRENCY),
}


@dataclass
class AssetSignal:
    """Signal for a single asset"""
    ticker: str
    name: str
    asset_class: AssetClass
    sector: Optional[str]
    signal: SignalStrength
    score: float  # -1 to 1
    conviction: float  # 0 to 1

    # Technical indicators
    price: float
    change_1d: float
    change_5d: float
    change_20d: float
    rsi: float
    sma_20: float
    sma_50: float
    volatility: float

    # Relative strength
    rs_vs_spy: float
    rs_percentile: float

    # Reasoning
    reasoning: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.reasoning:
            self.reasoning = []


@dataclass
class SectorAnalysis:
    """Analysis for a sector"""
    sector: str
    etf: str
    signal: SignalStrength
    score: float
    top_picks: List[str]
    laggards: List[str]
    rotation_signal: str  # "into", "out_of", "neutral"
    reasoning: str


@dataclass
class MultiAssetReport:
    """Complete multi-asset analysis report"""
    timestamp: datetime
    market_regime: str

    # Asset signals
    sector_signals: List[SectorAnalysis]
    stock_signals: List[AssetSignal]
    other_signals: List[AssetSignal]

    # Top picks
    top_buys: List[str]
    top_sells: List[str]
    sector_rotation: Dict[str, str]

    # Summary
    risk_on_score: float  # -1 (risk off) to 1 (risk on)
    summary: str


class MultiAssetAnalyzer:
    """Analyze multiple assets and generate signals"""

    def __init__(self, include_stocks: bool = True):
        self.include_stocks = include_stocks
        self.cache: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, tickers: List[str], period: str = "6mo") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers in parallel"""
        print(f"  Fetching data for {len(tickers)} assets...")

        results = {}

        def fetch_single(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return ticker, df if not df.empty else None
            except Exception as e:
                return ticker, None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_single, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    results[ticker] = df

        print(f"  Fetched {len(results)}/{len(tickers)} successfully")
        return results

    def calculate_technicals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        if df.empty or len(df) < 50:
            return {}

        try:
            close = df['Close']

            # Price changes
            change_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0.0
            change_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0.0
            change_20d = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0.0

            # Moving averages
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            rsi = rsi if not pd.isna(rsi) else 50.0

            # Volatility (annualized)
            volatility = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
            volatility = volatility if not pd.isna(volatility) else 0.2

            return {
                'price': float(close.iloc[-1]),
                'change_1d': change_1d if not pd.isna(change_1d) else 0.0,
                'change_5d': change_5d if not pd.isna(change_5d) else 0.0,
                'change_20d': change_20d if not pd.isna(change_20d) else 0.0,
                'sma_20': sma_20 if not pd.isna(sma_20) else float(close.iloc[-1]),
                'sma_50': sma_50 if not pd.isna(sma_50) else float(close.iloc[-1]),
                'rsi': rsi,
                'volatility': volatility,
            }
        except Exception:
            return {}

    def calculate_relative_strength(
        self,
        ticker_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        lookback: int = 20
    ) -> Tuple[float, float]:
        """Calculate relative strength vs SPY"""
        if ticker_data.empty or spy_data.empty:
            return 0.0, 50.0

        try:
            # Align data
            ticker_ret = ticker_data['Close'].pct_change(lookback).iloc[-1]
            spy_ret = spy_data['Close'].pct_change(lookback).iloc[-1]

            # Convert to float
            ticker_ret = float(ticker_ret) if not pd.isna(ticker_ret) else 0.0
            spy_ret = float(spy_ret) if not pd.isna(spy_ret) else 0.0

            rs = ticker_ret - spy_ret

            # Calculate percentile (simplified)
            percentile = 50.0 + (rs * 1000)  # Rough approximation
            percentile = max(0.0, min(100.0, percentile))

            return float(rs), float(percentile)
        except Exception:
            return 0.0, 50.0

    def generate_signal(self, technicals: Dict, rs: float) -> Tuple[SignalStrength, float, float, List[str]]:
        """Generate trading signal from technicals"""
        score = 0
        reasons = []

        if not technicals:
            return SignalStrength.NEUTRAL, 0, 0.3, ["Insufficient data"]

        price = technicals.get('price', 0)
        sma_20 = technicals.get('sma_20', price)
        sma_50 = technicals.get('sma_50', price)
        rsi = technicals.get('rsi', 50)
        change_20d = technicals.get('change_20d', 0)
        volatility = technicals.get('volatility', 0.2)

        # Trend (SMA crossover)
        if price > sma_20 > sma_50:
            score += 0.3
            reasons.append("Bullish trend (price > SMA20 > SMA50)")
        elif price < sma_20 < sma_50:
            score -= 0.3
            reasons.append("Bearish trend (price < SMA20 < SMA50)")
        elif price > sma_50:
            score += 0.1
            reasons.append("Above SMA50")
        else:
            score -= 0.1
            reasons.append("Below SMA50")

        # RSI
        if rsi < 30:
            score += 0.2
            reasons.append(f"Oversold (RSI: {rsi:.0f})")
        elif rsi > 70:
            score -= 0.2
            reasons.append(f"Overbought (RSI: {rsi:.0f})")
        elif rsi < 40:
            score += 0.1
            reasons.append(f"Approaching oversold (RSI: {rsi:.0f})")
        elif rsi > 60:
            score -= 0.1
            reasons.append(f"Approaching overbought (RSI: {rsi:.0f})")

        # Momentum
        if change_20d > 0.10:
            score += 0.2
            reasons.append(f"Strong momentum (+{change_20d:.1%} 20d)")
        elif change_20d < -0.10:
            score -= 0.2
            reasons.append(f"Weak momentum ({change_20d:.1%} 20d)")

        # Relative strength
        if rs > 0.03:
            score += 0.2
            reasons.append(f"Outperforming SPY (+{rs:.1%})")
        elif rs < -0.03:
            score -= 0.2
            reasons.append(f"Underperforming SPY ({rs:.1%})")

        # Volatility adjustment
        if volatility > 0.4:
            score *= 0.8
            reasons.append(f"High volatility ({volatility:.0%})")

        # Clamp score
        score = max(-1, min(1, score))

        # Determine signal
        if score >= 0.4:
            signal = SignalStrength.STRONG_BUY
        elif score >= 0.15:
            signal = SignalStrength.BUY
        elif score <= -0.4:
            signal = SignalStrength.STRONG_SELL
        elif score <= -0.15:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.NEUTRAL

        # Conviction based on score magnitude
        conviction = min(1.0, abs(score) + 0.3)

        return signal, score, conviction, reasons

    def analyze_sectors(self, data: Dict[str, pd.DataFrame], spy_data: pd.DataFrame) -> List[SectorAnalysis]:
        """Analyze all sectors"""
        print("  Analyzing sectors...")

        sector_results = []
        sector_scores = {}

        for etf, (name, asset_class) in SECTOR_ETFS.items():
            if etf not in data:
                continue

            technicals = self.calculate_technicals(data[etf])
            rs, rs_pct = self.calculate_relative_strength(data[etf], spy_data)
            signal, score, conviction, reasons = self.generate_signal(technicals, rs)

            sector_scores[etf] = score

            # Find top/bottom stocks in sector
            sector_stocks = [t for t, (n, s) in MAJOR_STOCKS.items() if s == etf]

            # Determine rotation signal
            if score > 0.2:
                rotation = "into"
            elif score < -0.2:
                rotation = "out_of"
            else:
                rotation = "neutral"

            sector_results.append(SectorAnalysis(
                sector=name,
                etf=etf,
                signal=signal,
                score=score,
                top_picks=sector_stocks[:3],
                laggards=[],
                rotation_signal=rotation,
                reasoning="; ".join(reasons)
            ))

        # Sort by score
        sector_results.sort(key=lambda x: x.score, reverse=True)

        return sector_results

    def analyze_stocks(self, data: Dict[str, pd.DataFrame], spy_data: pd.DataFrame) -> List[AssetSignal]:
        """Analyze individual stocks"""
        print("  Analyzing stocks...")

        stock_signals = []

        for ticker, (name, sector) in MAJOR_STOCKS.items():
            if ticker not in data:
                continue

            technicals = self.calculate_technicals(data[ticker])
            if not technicals:
                continue

            rs, rs_pct = self.calculate_relative_strength(data[ticker], spy_data)
            signal, score, conviction, reasons = self.generate_signal(technicals, rs)

            stock_signals.append(AssetSignal(
                ticker=ticker,
                name=name,
                asset_class=AssetClass.EQUITY,
                sector=sector,
                signal=signal,
                score=score,
                conviction=conviction,
                price=technicals['price'],
                change_1d=technicals['change_1d'],
                change_5d=technicals['change_5d'],
                change_20d=technicals['change_20d'],
                rsi=technicals['rsi'],
                sma_20=technicals['sma_20'],
                sma_50=technicals['sma_50'],
                volatility=technicals['volatility'],
                rs_vs_spy=rs,
                rs_percentile=rs_pct,
                reasoning=reasons
            ))

        # Sort by score
        stock_signals.sort(key=lambda x: x.score, reverse=True)

        return stock_signals

    def analyze_other_assets(self, data: Dict[str, pd.DataFrame], spy_data: pd.DataFrame) -> List[AssetSignal]:
        """Analyze bonds, commodities, currencies"""
        print("  Analyzing other assets...")

        signals = []

        for ticker, (name, asset_class) in OTHER_ASSETS.items():
            if ticker not in data:
                continue

            technicals = self.calculate_technicals(data[ticker])
            if not technicals:
                continue

            rs, rs_pct = self.calculate_relative_strength(data[ticker], spy_data)
            signal, score, conviction, reasons = self.generate_signal(technicals, rs)

            signals.append(AssetSignal(
                ticker=ticker,
                name=name,
                asset_class=asset_class,
                sector=None,
                signal=signal,
                score=score,
                conviction=conviction,
                price=technicals['price'],
                change_1d=technicals['change_1d'],
                change_5d=technicals['change_5d'],
                change_20d=technicals['change_20d'],
                rsi=technicals['rsi'],
                sma_20=technicals['sma_20'],
                sma_50=technicals['sma_50'],
                volatility=technicals['volatility'],
                rs_vs_spy=rs,
                rs_percentile=rs_pct,
                reasoning=reasons
            ))

        return signals

    def calculate_risk_on_score(
        self,
        sector_signals: List[SectorAnalysis],
        other_signals: List[AssetSignal]
    ) -> float:
        """Calculate overall risk-on/risk-off score"""
        score = 0

        # Sector rotation
        cyclical_sectors = ['XLK', 'XLY', 'XLF', 'XLI']
        defensive_sectors = ['XLU', 'XLP', 'XLV', 'XLRE']

        cyclical_score = np.mean([s.score for s in sector_signals if s.etf in cyclical_sectors]) if sector_signals else 0
        defensive_score = np.mean([s.score for s in sector_signals if s.etf in defensive_sectors]) if sector_signals else 0

        score += (cyclical_score - defensive_score) * 0.5

        # Safe haven assets
        for asset in other_signals:
            if asset.ticker in ['TLT', 'GLD']:
                score -= asset.score * 0.3  # Strong gold/bonds = risk off
            elif asset.ticker == 'HYG':
                score += asset.score * 0.2  # Strong HY = risk on

        return max(-1, min(1, score))

    def analyze_universe(self) -> MultiAssetReport:
        """Run complete multi-asset analysis"""
        print("\n" + "=" * 60)
        print("EIMAS Multi-Asset Analyzer")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print()

        # Collect all tickers
        all_tickers = ['SPY']
        all_tickers.extend(SECTOR_ETFS.keys())
        all_tickers.extend(OTHER_ASSETS.keys())
        if self.include_stocks:
            all_tickers.extend(MAJOR_STOCKS.keys())

        # Fetch data
        data = self.fetch_data(all_tickers)
        spy_data = data.get('SPY', pd.DataFrame())

        if spy_data.empty:
            raise ValueError("Could not fetch SPY data")

        # Analyze
        sector_signals = self.analyze_sectors(data, spy_data)
        stock_signals = self.analyze_stocks(data, spy_data) if self.include_stocks else []
        other_signals = self.analyze_other_assets(data, spy_data)

        # Risk score
        risk_score = self.calculate_risk_on_score(sector_signals, other_signals)

        # Top picks
        all_signals = stock_signals + [AssetSignal(
            ticker=s.etf, name=s.sector, asset_class=AssetClass.SECTOR,
            sector=None, signal=s.signal, score=s.score, conviction=0.6,
            price=0, change_1d=0, change_5d=0, change_20d=0, rsi=50,
            sma_20=0, sma_50=0, volatility=0.2, rs_vs_spy=0, rs_percentile=50,
            reasoning=[s.reasoning]
        ) for s in sector_signals]

        top_buys = [s.ticker for s in all_signals if s.signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]][:10]
        top_sells = [s.ticker for s in all_signals if s.signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]][:10]

        # Sector rotation
        sector_rotation = {s.sector: s.rotation_signal for s in sector_signals}

        # Market regime
        if risk_score > 0.3:
            regime = "Risk-On"
        elif risk_score < -0.3:
            regime = "Risk-Off"
        else:
            regime = "Neutral"

        # Summary
        summary = f"Market regime: {regime}. "
        if sector_signals:
            top_sector = sector_signals[0]
            bottom_sector = sector_signals[-1]
            summary += f"Favor {top_sector.sector} over {bottom_sector.sector}. "
        if top_buys:
            summary += f"Top picks: {', '.join(top_buys[:3])}. "

        return MultiAssetReport(
            timestamp=datetime.now(),
            market_regime=regime,
            sector_signals=sector_signals,
            stock_signals=stock_signals,
            other_signals=other_signals,
            top_buys=top_buys,
            top_sells=top_sells,
            sector_rotation=sector_rotation,
            risk_on_score=risk_score,
            summary=summary
        )

    def print_report(self, report: MultiAssetReport):
        """Print analysis report"""
        print("\n" + "=" * 70)
        print("Multi-Asset Analysis Report")
        print("=" * 70)

        print(f"\nMarket Regime: {report.market_regime}")
        print(f"Risk-On Score: {report.risk_on_score:+.2f} (-1 = Risk Off, +1 = Risk On)")

        # Sector rankings
        print("\n" + "-" * 70)
        print("SECTOR RANKINGS")
        print("-" * 70)
        print(f"{'Sector':<25} {'ETF':<6} {'Signal':<12} {'Score':>8} {'Rotation':<10}")
        print("-" * 70)

        for s in report.sector_signals:
            signal_str = s.signal.value.replace('_', ' ').title()
            print(f"{s.sector:<25} {s.etf:<6} {signal_str:<12} {s.score:>+8.2f} {s.rotation_signal:<10}")

        # Top stock picks
        if report.stock_signals:
            print("\n" + "-" * 70)
            print("TOP STOCK PICKS")
            print("-" * 70)
            print(f"{'Ticker':<8} {'Name':<20} {'Signal':<12} {'Score':>8} {'RSI':>6} {'20d%':>8}")
            print("-" * 70)

            for s in report.stock_signals[:10]:
                signal_str = s.signal.value.replace('_', ' ').title()
                print(f"{s.ticker:<8} {s.name:<20} {signal_str:<12} {s.score:>+8.2f} {s.rsi:>6.0f} {s.change_20d:>+7.1%}")

        # Other assets
        print("\n" + "-" * 70)
        print("OTHER ASSETS")
        print("-" * 70)
        print(f"{'Ticker':<8} {'Name':<25} {'Class':<12} {'Signal':<12} {'Score':>8}")
        print("-" * 70)

        for s in report.other_signals:
            signal_str = s.signal.value.replace('_', ' ').title()
            class_str = s.asset_class.value.title()
            print(f"{s.ticker:<8} {s.name:<25} {class_str:<12} {signal_str:<12} {s.score:>+8.2f}")

        # Summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Top Buys:  {', '.join(report.top_buys[:5]) if report.top_buys else 'None'}")
        print(f"Top Sells: {', '.join(report.top_sells[:5]) if report.top_sells else 'None'}")
        print(f"\n{report.summary}")
        print("=" * 70)

    def to_dataframe(self, report: MultiAssetReport) -> pd.DataFrame:
        """Convert report to DataFrame for dashboard"""
        rows = []

        for s in report.stock_signals:
            rows.append({
                'ticker': s.ticker,
                'name': s.name,
                'type': 'stock',
                'sector': s.sector,
                'signal': s.signal.value,
                'score': s.score,
                'conviction': s.conviction,
                'price': s.price,
                'change_1d': s.change_1d,
                'change_20d': s.change_20d,
                'rsi': s.rsi,
                'rs_vs_spy': s.rs_vs_spy
            })

        for s in report.sector_signals:
            rows.append({
                'ticker': s.etf,
                'name': s.sector,
                'type': 'sector',
                'sector': None,
                'signal': s.signal.value,
                'score': s.score,
                'conviction': 0.6,
                'price': 0,
                'change_1d': 0,
                'change_20d': 0,
                'rsi': 50,
                'rs_vs_spy': 0
            })

        for s in report.other_signals:
            rows.append({
                'ticker': s.ticker,
                'name': s.name,
                'type': s.asset_class.value,
                'sector': None,
                'signal': s.signal.value,
                'score': s.score,
                'conviction': s.conviction,
                'price': s.price,
                'change_1d': s.change_1d,
                'change_20d': s.change_20d,
                'rsi': s.rsi,
                'rs_vs_spy': s.rs_vs_spy
            })

        return pd.DataFrame(rows)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    analyzer = MultiAssetAnalyzer(include_stocks=True)
    report = analyzer.analyze_universe()
    analyzer.print_report(report)
