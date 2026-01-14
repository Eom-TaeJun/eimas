#!/usr/bin/env python3
"""
EIMAS Tax Lot Optimizer
=======================
Optimize tax lots for selling to minimize tax burden.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class TaxStrategy(Enum):
    FIFO = "fifo"                    # First In, First Out
    LIFO = "lifo"                    # Last In, First Out
    HIFO = "hifo"                    # Highest Cost First
    LOFO = "lofo"                    # Lowest Cost First
    SPECIFIC_ID = "specific_id"      # Specific identification
    TAX_LOSS = "tax_loss"           # Harvest losses first
    LONG_TERM_FIRST = "lt_first"    # Long-term gains first
    SHORT_TERM_FIRST = "st_first"   # Short-term gains first


@dataclass
class TaxLot:
    id: str
    ticker: str
    purchase_date: datetime
    quantity: int
    cost_basis: float  # Per share
    current_price: float = 0
    unrealized_gain: float = 0
    unrealized_pct: float = 0
    is_long_term: bool = False
    holding_days: int = 0


@dataclass
class SaleRecommendation:
    lots_to_sell: List[TaxLot]
    total_shares: int
    total_proceeds: float
    total_cost_basis: float
    realized_gain: float
    short_term_gain: float
    long_term_gain: float
    estimated_tax: float
    strategy_used: TaxStrategy


class TaxLotManager:
    """Manage tax lots and optimize selling"""

    def __init__(self, lots_file: str = "data/tax_lots.json"):
        self.lots_file = lots_file
        self.lots: List[TaxLot] = []
        self.short_term_rate = 0.37  # Top marginal rate
        self.long_term_rate = 0.20   # Long-term cap gains
        self.long_term_threshold = 365  # Days
        self._load_lots()

    def _load_lots(self):
        """Load lots from file"""
        if os.path.exists(self.lots_file):
            try:
                with open(self.lots_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['purchase_date'] = datetime.fromisoformat(item['purchase_date'])
                        self.lots.append(TaxLot(**item))
            except Exception:
                self.lots = []

    def _save_lots(self):
        """Save lots to file"""
        os.makedirs(os.path.dirname(self.lots_file), exist_ok=True)
        data = []
        for lot in self.lots:
            d = asdict(lot)
            d['purchase_date'] = lot.purchase_date.isoformat()
            data.append(d)
        with open(self.lots_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_lot(self, ticker: str, purchase_date: datetime, quantity: int, cost_basis: float) -> TaxLot:
        """Add a new tax lot"""
        lot = TaxLot(
            id=f"LOT_{ticker}_{datetime.now().timestamp():.0f}",
            ticker=ticker.upper(),
            purchase_date=purchase_date,
            quantity=quantity,
            cost_basis=cost_basis
        )
        self.lots.append(lot)
        self._save_lots()
        return lot

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices and calculate gains"""
        for lot in self.lots:
            if lot.ticker in prices:
                lot.current_price = prices[lot.ticker]
                lot.unrealized_gain = (lot.current_price - lot.cost_basis) * lot.quantity
                lot.unrealized_pct = (lot.current_price / lot.cost_basis - 1) * 100

                lot.holding_days = (datetime.now() - lot.purchase_date).days
                lot.is_long_term = lot.holding_days >= self.long_term_threshold

    def get_lots_by_ticker(self, ticker: str) -> List[TaxLot]:
        """Get all lots for a ticker"""
        return [lot for lot in self.lots if lot.ticker == ticker.upper()]

    def optimize_sale(self, ticker: str, shares_to_sell: int, strategy: TaxStrategy = TaxStrategy.HIFO,
                     current_price: float = None) -> Optional[SaleRecommendation]:
        """Optimize which lots to sell"""
        lots = self.get_lots_by_ticker(ticker)
        if not lots:
            return None

        # Update prices
        if current_price:
            for lot in lots:
                lot.current_price = current_price
                lot.unrealized_gain = (lot.current_price - lot.cost_basis) * lot.quantity
                lot.unrealized_pct = (lot.current_price / lot.cost_basis - 1) * 100
                lot.holding_days = (datetime.now() - lot.purchase_date).days
                lot.is_long_term = lot.holding_days >= self.long_term_threshold

        total_shares = sum(lot.quantity for lot in lots)
        if shares_to_sell > total_shares:
            print(f"Not enough shares. Have {total_shares}, want to sell {shares_to_sell}")
            shares_to_sell = total_shares

        # Sort lots based on strategy
        if strategy == TaxStrategy.FIFO:
            sorted_lots = sorted(lots, key=lambda x: x.purchase_date)
        elif strategy == TaxStrategy.LIFO:
            sorted_lots = sorted(lots, key=lambda x: x.purchase_date, reverse=True)
        elif strategy == TaxStrategy.HIFO:
            sorted_lots = sorted(lots, key=lambda x: x.cost_basis, reverse=True)
        elif strategy == TaxStrategy.LOFO:
            sorted_lots = sorted(lots, key=lambda x: x.cost_basis)
        elif strategy == TaxStrategy.TAX_LOSS:
            # Losses first, then by cost basis
            sorted_lots = sorted(lots, key=lambda x: (x.unrealized_gain >= 0, -x.unrealized_gain))
        elif strategy == TaxStrategy.LONG_TERM_FIRST:
            sorted_lots = sorted(lots, key=lambda x: (not x.is_long_term, -x.cost_basis))
        elif strategy == TaxStrategy.SHORT_TERM_FIRST:
            sorted_lots = sorted(lots, key=lambda x: (x.is_long_term, -x.cost_basis))
        else:
            sorted_lots = lots

        # Select lots to sell
        lots_to_sell = []
        remaining = shares_to_sell

        for lot in sorted_lots:
            if remaining <= 0:
                break

            if lot.quantity <= remaining:
                lots_to_sell.append(lot)
                remaining -= lot.quantity
            else:
                # Partial lot sale (create a copy with reduced quantity)
                partial_lot = TaxLot(
                    id=lot.id + "_partial",
                    ticker=lot.ticker,
                    purchase_date=lot.purchase_date,
                    quantity=remaining,
                    cost_basis=lot.cost_basis,
                    current_price=lot.current_price,
                    unrealized_gain=(lot.current_price - lot.cost_basis) * remaining,
                    unrealized_pct=lot.unrealized_pct,
                    is_long_term=lot.is_long_term,
                    holding_days=lot.holding_days
                )
                lots_to_sell.append(partial_lot)
                remaining = 0

        # Calculate totals
        total_shares_sold = sum(lot.quantity for lot in lots_to_sell)
        total_proceeds = sum(lot.current_price * lot.quantity for lot in lots_to_sell)
        total_cost = sum(lot.cost_basis * lot.quantity for lot in lots_to_sell)
        realized_gain = total_proceeds - total_cost

        short_term_gain = sum(
            (lot.current_price - lot.cost_basis) * lot.quantity
            for lot in lots_to_sell if not lot.is_long_term
        )
        long_term_gain = sum(
            (lot.current_price - lot.cost_basis) * lot.quantity
            for lot in lots_to_sell if lot.is_long_term
        )

        # Estimate tax
        st_tax = max(0, short_term_gain) * self.short_term_rate
        lt_tax = max(0, long_term_gain) * self.long_term_rate
        estimated_tax = st_tax + lt_tax

        return SaleRecommendation(
            lots_to_sell=lots_to_sell,
            total_shares=total_shares_sold,
            total_proceeds=total_proceeds,
            total_cost_basis=total_cost,
            realized_gain=realized_gain,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            estimated_tax=estimated_tax,
            strategy_used=strategy
        )

    def compare_strategies(self, ticker: str, shares_to_sell: int, current_price: float) -> Dict[TaxStrategy, SaleRecommendation]:
        """Compare all strategies"""
        strategies = [
            TaxStrategy.FIFO, TaxStrategy.LIFO, TaxStrategy.HIFO,
            TaxStrategy.LOFO, TaxStrategy.TAX_LOSS, TaxStrategy.LONG_TERM_FIRST
        ]

        results = {}
        for strategy in strategies:
            rec = self.optimize_sale(ticker, shares_to_sell, strategy, current_price)
            if rec:
                results[strategy] = rec

        return results

    def find_tax_loss_harvesting(self, min_loss: float = 1000) -> List[TaxLot]:
        """Find lots eligible for tax loss harvesting"""
        losses = [lot for lot in self.lots if lot.unrealized_gain < -min_loss]
        return sorted(losses, key=lambda x: x.unrealized_gain)

    def print_lots(self, ticker: str = None):
        """Print all lots"""
        lots = self.get_lots_by_ticker(ticker) if ticker else self.lots

        print("\n" + "=" * 100)
        print("Tax Lot Report")
        print("=" * 100)

        if not lots:
            print("\nNo lots found")
            return

        print(f"\n{'ID':<25} {'Ticker':<8} {'Date':<12} {'Qty':>6} {'Cost':>10} {'Price':>10} {'Gain':>12} {'Type':>8}")
        print("-" * 100)

        for lot in sorted(lots, key=lambda x: (x.ticker, x.purchase_date)):
            lot_type = "LT" if lot.is_long_term else "ST"
            date_str = lot.purchase_date.strftime("%Y-%m-%d")
            print(f"{lot.id[:24]:<25} {lot.ticker:<8} {date_str:<12} {lot.quantity:>6} "
                  f"${lot.cost_basis:>8.2f} ${lot.current_price:>8.2f} "
                  f"${lot.unrealized_gain:>+10,.2f} {lot_type:>8}")

        # Summary
        total_value = sum(lot.current_price * lot.quantity for lot in lots)
        total_cost = sum(lot.cost_basis * lot.quantity for lot in lots)
        total_gain = total_value - total_cost

        print("\n" + "-" * 100)
        print(f"Total Value: ${total_value:,.2f}  |  Total Cost: ${total_cost:,.2f}  |  Unrealized Gain: ${total_gain:+,.2f}")

    def print_strategy_comparison(self, ticker: str, shares: int, price: float):
        """Print comparison of all strategies"""
        results = self.compare_strategies(ticker, shares, price)

        print("\n" + "=" * 100)
        print(f"Tax Strategy Comparison: Sell {shares} shares of {ticker} @ ${price:.2f}")
        print("=" * 100)

        print(f"\n{'Strategy':<18} {'Proceeds':>12} {'Cost Basis':>12} {'ST Gain':>12} {'LT Gain':>12} {'Est Tax':>12}")
        print("-" * 100)

        for strategy, rec in sorted(results.items(), key=lambda x: x[1].estimated_tax):
            print(f"{strategy.value:<18} ${rec.total_proceeds:>10,.2f} ${rec.total_cost_basis:>10,.2f} "
                  f"${rec.short_term_gain:>+10,.2f} ${rec.long_term_gain:>+10,.2f} ${rec.estimated_tax:>10,.2f}")

        # Best strategy
        best = min(results.items(), key=lambda x: x[1].estimated_tax)
        worst = max(results.items(), key=lambda x: x[1].estimated_tax)
        savings = worst[1].estimated_tax - best[1].estimated_tax

        print("\n" + "-" * 100)
        print(f"📊 RECOMMENDATION: Use {best[0].value.upper()} strategy")
        print(f"   Estimated Tax: ${best[1].estimated_tax:,.2f}")
        print(f"   Tax Savings vs Worst: ${savings:,.2f}")

        if best[1].lots_to_sell:
            print(f"\n   Lots to sell:")
            for lot in best[1].lots_to_sell:
                lt_str = "(LT)" if lot.is_long_term else "(ST)"
                print(f"      {lot.quantity} shares @ ${lot.cost_basis:.2f} cost, "
                      f"held {lot.holding_days} days {lt_str}")

        print("=" * 100)


if __name__ == "__main__":
    manager = TaxLotManager()

    # Example: Add some tax lots
    # manager.add_lot("AAPL", datetime(2023, 1, 15), 50, 145.00)
    # manager.add_lot("AAPL", datetime(2023, 6, 20), 30, 185.00)
    # manager.add_lot("AAPL", datetime(2024, 3, 10), 20, 170.00)
    # manager.add_lot("AAPL", datetime(2024, 11, 5), 25, 220.00)

    # Update with current prices
    current_prices = {"AAPL": 195.00, "MSFT": 420.00, "NVDA": 140.00}
    manager.update_prices(current_prices)

    # Print all lots
    manager.print_lots()

    # Compare strategies
    if manager.lots:
        manager.print_strategy_comparison("AAPL", 50, 195.00)

    # Find tax loss harvesting opportunities
    losses = manager.find_tax_loss_harvesting(500)
    if losses:
        print("\n📉 Tax Loss Harvesting Opportunities:")
        for lot in losses:
            print(f"   {lot.ticker}: {lot.quantity} shares, Loss: ${lot.unrealized_gain:,.2f}")
