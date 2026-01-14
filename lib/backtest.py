#!/usr/bin/env python3
"""
EIMAS Backtest Engine
=====================
Backtest signal strategies against historical data.

Usage:
    from lib.backtest import BacktestEngine

    engine = BacktestEngine()
    results = engine.run(start_date="2024-01-01", end_date="2024-12-31")
    engine.print_report(results)
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


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    HEDGE = "hedge"


@dataclass
class BacktestTrade:
    """A single trade in the backtest"""
    date: datetime
    signal: SignalType
    conviction: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None


@dataclass
class BacktestMetrics:
    """Backtest performance metrics"""
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    avg_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_days: float
    profit_factor: float

    # By signal type
    metrics_by_signal: Dict[str, Dict] = field(default_factory=dict)


class BacktestEngine:
    """Engine for backtesting signal strategies"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []

    def fetch_historical_data(
        self,
        ticker: str = "SPY",
        start_date: str = "2024-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """Fetch historical price data"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"  Fetching {ticker} data: {start_date} to {end_date}...")

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def generate_historical_signals(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Generate signals based on historical indicators.
        Uses simplified versions of our signal logic.
        """
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['Close']
        signals['signal'] = SignalType.HOLD.value
        signals['conviction'] = 0.5

        # Calculate indicators
        signals['sma_20'] = df['Close'].rolling(20).mean()
        signals['sma_50'] = df['Close'].rolling(50).mean()
        signals['volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        signals['rsi'] = self._calculate_rsi(df['Close'], 14)
        signals['returns_5d'] = df['Close'].pct_change(5)

        # VIX proxy (realized volatility)
        signals['vix_proxy'] = signals['volatility'] * 100

        # Generate signals based on conditions
        for i in range(lookback, len(signals)):
            idx = signals.index[i]

            # Trend signals
            trend_bullish = signals.loc[idx, 'sma_20'] > signals.loc[idx, 'sma_50']
            low_vol = signals.loc[idx, 'volatility'] < 0.15

            # RSI signals
            rsi = signals.loc[idx, 'rsi']
            oversold = rsi < 30
            overbought = rsi > 70

            # VIX-like signals
            vix = signals.loc[idx, 'vix_proxy']
            vix_spike = vix > 25
            vix_low = vix < 12

            # Determine signal
            if trend_bullish and low_vol:
                signals.loc[idx, 'signal'] = SignalType.BUY.value
                signals.loc[idx, 'conviction'] = 0.7
            elif oversold and not vix_spike:
                signals.loc[idx, 'signal'] = SignalType.BUY.value
                signals.loc[idx, 'conviction'] = 0.8
            elif overbought and vix_low:
                signals.loc[idx, 'signal'] = SignalType.REDUCE.value
                signals.loc[idx, 'conviction'] = 0.6
            elif vix_spike:
                signals.loc[idx, 'signal'] = SignalType.HEDGE.value
                signals.loc[idx, 'conviction'] = 0.7
            elif not trend_bullish and not low_vol:
                signals.loc[idx, 'signal'] = SignalType.REDUCE.value
                signals.loc[idx, 'conviction'] = 0.6
            else:
                signals.loc[idx, 'signal'] = SignalType.HOLD.value
                signals.loc[idx, 'conviction'] = 0.5

        return signals

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run(
        self,
        ticker: str = "SPY",
        start_date: str = "2024-01-01",
        end_date: str = None,
        holding_period: int = 5,
        position_size: float = 0.2
    ) -> BacktestMetrics:
        """
        Run the backtest.

        Args:
            ticker: Stock ticker to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            holding_period: Days to hold each position
            position_size: Fraction of capital per trade
        """
        print("\n" + "=" * 60)
        print("EIMAS Backtest Engine")
        print("=" * 60)
        print(f"Ticker: {ticker}")
        print(f"Period: {start_date} to {end_date or 'now'}")
        print(f"Holding Period: {holding_period} days")
        print(f"Position Size: {position_size:.0%}")
        print()

        # Fetch data
        df = self.fetch_historical_data(ticker, start_date, end_date)

        if df.empty:
            raise ValueError("No data fetched")

        # Generate signals
        print("  Generating historical signals...")
        signals = self.generate_historical_signals(df)

        # Run simulation
        print("  Running simulation...")
        self.trades = []
        capital = self.initial_capital
        self.equity_curve = [capital]

        i = 0
        while i < len(signals) - holding_period:
            idx = signals.index[i]
            signal_val = signals.loc[idx, 'signal']

            # Only trade on non-hold signals
            if signal_val != SignalType.HOLD.value:
                entry_price = signals.loc[idx, 'price']
                exit_idx = signals.index[min(i + holding_period, len(signals) - 1)]
                exit_price = signals.loc[exit_idx, 'price']

                # Calculate return based on signal
                if signal_val == SignalType.BUY.value:
                    return_pct = (exit_price - entry_price) / entry_price
                elif signal_val in [SignalType.SELL.value, SignalType.REDUCE.value]:
                    return_pct = (entry_price - exit_price) / entry_price  # Short
                elif signal_val == SignalType.HEDGE.value:
                    return_pct = abs(exit_price - entry_price) / entry_price * 0.5  # Partial
                else:
                    return_pct = 0

                # Apply position sizing
                pnl = capital * position_size * return_pct
                capital += pnl

                trade = BacktestTrade(
                    date=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                    signal=SignalType(signal_val),
                    conviction=signals.loc[idx, 'conviction'],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_date=exit_idx.to_pydatetime() if hasattr(exit_idx, 'to_pydatetime') else exit_idx,
                    return_pct=return_pct,
                    holding_days=holding_period
                )
                self.trades.append(trade)

                # Skip holding period
                i += holding_period
            else:
                i += 1

            self.equity_curve.append(capital)

        # Calculate metrics
        print("  Calculating metrics...")
        metrics = self._calculate_metrics()

        return metrics

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return BacktestMetrics(
                total_signals=0, winning_signals=0, losing_signals=0,
                win_rate=0, avg_return=0, total_return=0,
                sharpe_ratio=0, max_drawdown=0, avg_holding_days=0,
                profit_factor=0, metrics_by_signal={}
            )

        returns = [t.return_pct for t in self.trades if t.return_pct is not None]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        # Basic metrics
        total_signals = len(self.trades)
        winning_signals = len(winning)
        losing_signals = len(losing)
        win_rate = winning_signals / total_signals if total_signals > 0 else 0

        avg_return = np.mean(returns) if returns else 0
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Avg holding days
        avg_holding_days = np.mean([t.holding_days for t in self.trades if t.holding_days])

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Metrics by signal type
        metrics_by_signal = {}
        for signal_type in SignalType:
            type_trades = [t for t in self.trades if t.signal == signal_type]
            if type_trades:
                type_returns = [t.return_pct for t in type_trades if t.return_pct is not None]
                type_winning = len([r for r in type_returns if r > 0])
                metrics_by_signal[signal_type.value] = {
                    'count': len(type_trades),
                    'win_rate': type_winning / len(type_trades) if type_trades else 0,
                    'avg_return': np.mean(type_returns) if type_returns else 0,
                }

        return BacktestMetrics(
            total_signals=total_signals,
            winning_signals=winning_signals,
            losing_signals=losing_signals,
            win_rate=win_rate,
            avg_return=avg_return,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_holding_days=avg_holding_days,
            profit_factor=profit_factor,
            metrics_by_signal=metrics_by_signal
        )

    def print_report(self, metrics: BacktestMetrics):
        """Print backtest results"""
        print("\n" + "=" * 60)
        print("Backtest Results")
        print("=" * 60)

        print(f"\n{'Metric':<25} {'Value':>15}")
        print("-" * 42)
        print(f"{'Total Signals':<25} {metrics.total_signals:>15}")
        print(f"{'Winning Signals':<25} {metrics.winning_signals:>15}")
        print(f"{'Losing Signals':<25} {metrics.losing_signals:>15}")
        print(f"{'Win Rate':<25} {metrics.win_rate:>14.1%}")
        print(f"{'Avg Return/Trade':<25} {metrics.avg_return:>14.2%}")
        print(f"{'Total Return':<25} {metrics.total_return:>14.1%}")
        print(f"{'Sharpe Ratio':<25} {metrics.sharpe_ratio:>15.2f}")
        print(f"{'Max Drawdown':<25} {metrics.max_drawdown:>14.1%}")
        print(f"{'Avg Holding Days':<25} {metrics.avg_holding_days:>15.1f}")
        print(f"{'Profit Factor':<25} {metrics.profit_factor:>15.2f}")

        if metrics.metrics_by_signal:
            print("\n" + "-" * 60)
            print("Performance by Signal Type")
            print("-" * 60)
            print(f"{'Signal':<12} {'Count':>8} {'Win Rate':>12} {'Avg Return':>12}")
            print("-" * 46)
            for signal, data in metrics.metrics_by_signal.items():
                print(f"{signal.upper():<12} {data['count']:>8} {data['win_rate']:>11.1%} {data['avg_return']:>11.2%}")

        print("\n" + "=" * 60)

    def get_equity_curve_data(self) -> Dict[str, List]:
        """Get equity curve data for plotting"""
        return {
            'values': self.equity_curve,
            'initial': self.initial_capital
        }

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        data = []
        for t in self.trades:
            data.append({
                'date': t.date,
                'signal': t.signal.value,
                'conviction': t.conviction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_date': t.exit_date,
                'return_pct': t.return_pct,
                'holding_days': t.holding_days
            })
        return pd.DataFrame(data)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    engine = BacktestEngine(initial_capital=100000)

    # Run backtest for 2024
    metrics = engine.run(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2024-12-31",
        holding_period=5,
        position_size=0.2
    )

    engine.print_report(metrics)

    # Show sample trades
    trades_df = engine.get_trades_df()
    if not trades_df.empty:
        print("\nSample Trades (first 10):")
        print(trades_df.head(10).to_string())
