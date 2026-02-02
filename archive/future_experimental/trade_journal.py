#!/usr/bin/env python3
"""
EIMAS Trade Journal
===================
Track and analyze trading performance.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Trade:
    id: str
    ticker: str
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    strategy: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    emotions: str = ""  # How you felt during the trade
    lessons: str = ""   # What you learned


@dataclass
class TradeStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    avg_hold_time: float  # days
    expectancy: float
    consecutive_wins: int
    consecutive_losses: int
    sharpe_ratio: float


class TradeJournal:
    """Track and analyze trades"""

    def __init__(self, journal_file: str = "data/trade_journal.json"):
        self.journal_file = journal_file
        self.trades: List[Trade] = []
        self._load_journal()

    def _load_journal(self):
        """Load journal from file"""
        if os.path.exists(self.journal_file):
            try:
                with open(self.journal_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['direction'] = TradeDirection(item['direction'])
                        item['status'] = TradeStatus(item['status'])
                        item['entry_date'] = datetime.fromisoformat(item['entry_date'])
                        if item.get('exit_date'):
                            item['exit_date'] = datetime.fromisoformat(item['exit_date'])
                        self.trades.append(Trade(**item))
            except Exception:
                self.trades = []

    def _save_journal(self):
        """Save journal to file"""
        os.makedirs(os.path.dirname(self.journal_file), exist_ok=True)
        data = []
        for trade in self.trades:
            d = asdict(trade)
            d['direction'] = trade.direction.value
            d['status'] = trade.status.value
            d['entry_date'] = trade.entry_date.isoformat()
            if trade.exit_date:
                d['exit_date'] = trade.exit_date.isoformat()
            data.append(d)
        with open(self.journal_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_trade(self, ticker: str, direction: TradeDirection, entry_price: float,
                 quantity: int, stop_loss: float = None, take_profit: float = None,
                 strategy: str = "", notes: str = "", tags: List[str] = None) -> Trade:
        """Add a new trade"""
        trade = Trade(
            id=f"T_{datetime.now().timestamp():.0f}",
            ticker=ticker.upper(),
            direction=direction,
            entry_date=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            notes=notes,
            tags=tags or []
        )
        self.trades.append(trade)
        self._save_journal()
        print(f"Added: {direction.value.upper()} {quantity} {ticker} @ ${entry_price:.2f}")
        return trade

    def close_trade(self, trade_id: str, exit_price: float, lessons: str = "", emotions: str = "") -> Optional[Trade]:
        """Close an existing trade"""
        trade = next((t for t in self.trades if t.id == trade_id), None)
        if not trade:
            print(f"Trade {trade_id} not found")
            return None

        trade.exit_date = datetime.now()
        trade.exit_price = exit_price
        trade.status = TradeStatus.CLOSED
        trade.lessons = lessons
        trade.emotions = emotions

        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            trade.pnl_pct = (exit_price / trade.entry_price - 1) * 100
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            trade.pnl_pct = (trade.entry_price / exit_price - 1) * 100

        self._save_journal()
        print(f"Closed: {trade.ticker} @ ${exit_price:.2f} | P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)")
        return trade

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades"""
        return [t for t in self.trades if t.status == TradeStatus.OPEN]

    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades"""
        return [t for t in self.trades if t.status == TradeStatus.CLOSED]

    def calculate_stats(self, period_days: int = None) -> TradeStats:
        """Calculate trading statistics"""
        closed = self.get_closed_trades()

        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            closed = [t for t in closed if t.exit_date and t.exit_date > cutoff]

        if not closed:
            return TradeStats(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, avg_win=0, avg_loss=0,
                profit_factor=0, largest_win=0, largest_loss=0,
                avg_hold_time=0, expectancy=0, consecutive_wins=0,
                consecutive_losses=0, sharpe_ratio=0
            )

        # Basic stats
        winners = [t for t in closed if t.pnl and t.pnl > 0]
        losers = [t for t in closed if t.pnl and t.pnl < 0]

        total_trades = len(closed)
        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in closed if t.pnl)

        avg_win = sum(t.pnl for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t.pnl for t in losers) / len(losers) if losers else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        largest_win = max((t.pnl for t in closed if t.pnl), default=0)
        largest_loss = min((t.pnl for t in closed if t.pnl), default=0)

        # Hold time
        hold_times = []
        for t in closed:
            if t.exit_date and t.entry_date:
                hold_times.append((t.exit_date - t.entry_date).days)
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for t in sorted(closed, key=lambda x: x.exit_date or datetime.now()):
            if t.pnl and t.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        # Sharpe-like ratio
        returns = [t.pnl_pct for t in closed if t.pnl_pct]
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_hold_time=avg_hold_time,
            expectancy=expectancy,
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            sharpe_ratio=sharpe_ratio
        )

    def get_stats_by_strategy(self) -> Dict[str, TradeStats]:
        """Get stats grouped by strategy"""
        strategies = set(t.strategy for t in self.trades if t.strategy)
        result = {}
        for strategy in strategies:
            strategy_trades = [t for t in self.trades if t.strategy == strategy]
            # Temporarily replace trades list for calculation
            original = self.trades
            self.trades = strategy_trades
            result[strategy] = self.calculate_stats()
            self.trades = original
        return result

    def print_open_positions(self):
        """Print open positions"""
        open_trades = self.get_open_trades()

        print("\n" + "=" * 80)
        print("Open Positions")
        print("=" * 80)

        if not open_trades:
            print("\nNo open positions")
            return

        print(f"\n{'Ticker':<8} {'Dir':<6} {'Qty':>8} {'Entry':>10} {'Stop':>10} {'Target':>10} {'Days':>6}")
        print("-" * 80)

        for t in open_trades:
            days_held = (datetime.now() - t.entry_date).days
            stop = f"${t.stop_loss:.2f}" if t.stop_loss else "-"
            target = f"${t.take_profit:.2f}" if t.take_profit else "-"
            print(f"{t.ticker:<8} {t.direction.value:<6} {t.quantity:>8} ${t.entry_price:>8.2f} {stop:>10} {target:>10} {days_held:>6}")

    def print_report(self, period_days: int = None):
        """Print comprehensive journal report"""
        stats = self.calculate_stats(period_days)

        period_str = f"Last {period_days} Days" if period_days else "All Time"

        print("\n" + "=" * 80)
        print(f"Trade Journal Report ({period_str})")
        print("=" * 80)

        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"  Total Trades:     {stats.total_trades}")
        print(f"  Win Rate:         {stats.win_rate:.1f}%")
        print(f"  Total P&L:        ${stats.total_pnl:+,.2f}")
        print(f"  Profit Factor:    {stats.profit_factor:.2f}")

        print(f"\n💰 P&L ANALYSIS:")
        print("-" * 40)
        print(f"  Average Win:      ${stats.avg_win:+,.2f}")
        print(f"  Average Loss:     ${stats.avg_loss:+,.2f}")
        print(f"  Largest Win:      ${stats.largest_win:+,.2f}")
        print(f"  Largest Loss:     ${stats.largest_loss:+,.2f}")
        print(f"  Expectancy:       ${stats.expectancy:+,.2f}")

        print(f"\n📈 TRADING PATTERNS:")
        print("-" * 40)
        print(f"  Avg Hold Time:    {stats.avg_hold_time:.1f} days")
        print(f"  Max Win Streak:   {stats.consecutive_wins}")
        print(f"  Max Loss Streak:  {stats.consecutive_losses}")
        print(f"  Sharpe Ratio:     {stats.sharpe_ratio:.2f}")

        # Recent trades
        recent = sorted(self.get_closed_trades(), key=lambda x: x.exit_date or datetime.min, reverse=True)[:10]

        if recent:
            print(f"\n📋 RECENT TRADES:")
            print("-" * 80)
            print(f"{'Date':<12} {'Ticker':<8} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'P&L':>12} {'%':>8}")
            print("-" * 80)

            for t in recent:
                date_str = t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "-"
                print(f"{date_str:<12} {t.ticker:<8} {t.direction.value:<6} "
                      f"${t.entry_price:>6.2f} ${t.exit_price if t.exit_price else 0:>6.2f} "
                      f"${t.pnl if t.pnl else 0:>+10,.2f} {t.pnl_pct if t.pnl_pct else 0:>+7.2f}%")

        # Strategy breakdown
        by_strategy = self.get_stats_by_strategy()
        if by_strategy:
            print(f"\n🎯 BY STRATEGY:")
            print("-" * 60)
            for strategy, s in by_strategy.items():
                print(f"  {strategy}: {s.total_trades} trades, Win Rate: {s.win_rate:.1f}%, P&L: ${s.total_pnl:+,.2f}")

        print("=" * 80)


if __name__ == "__main__":
    journal = TradeJournal()

    # Example trades
    # journal.add_trade("AAPL", TradeDirection.LONG, 175.50, 100, stop_loss=170, take_profit=190, strategy="momentum")
    # journal.add_trade("NVDA", TradeDirection.LONG, 450.00, 50, strategy="breakout")

    journal.print_open_positions()
    journal.print_report()
