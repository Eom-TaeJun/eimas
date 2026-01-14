#!/usr/bin/env python3
"""
EIMAS Paper Trading System
==========================
Simulated trading for strategy validation.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import json
import os
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Order:
    id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: float
    current_price: float = 0
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0


@dataclass
class PaperAccount:
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


class PaperTrader:
    """Paper trading simulator"""

    def __init__(self, initial_capital: float = 100000, data_file: str = "data/paper_account.json"):
        self.data_file = data_file
        self.account = self._load_account(initial_capital)

    def _load_account(self, initial_capital: float) -> PaperAccount:
        """Load account from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    account = PaperAccount(
                        initial_capital=data['initial_capital'],
                        cash=data['cash'],
                        positions={k: Position(**v) for k, v in data.get('positions', {}).items()},
                        orders=[],
                        trades=data.get('trades', [])
                    )
                    return account
            except Exception:
                pass
        return PaperAccount(initial_capital=initial_capital, cash=initial_capital)

    def _save_account(self):
        """Save account to file"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        data = {
            'initial_capital': self.account.initial_capital,
            'cash': self.account.cash,
            'positions': {k: asdict(v) for k, v in self.account.positions.items()},
            'trades': self.account.trades
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_price(self, ticker: str) -> float:
        """Get current price"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('regularMarketPrice') or stock.info.get('previousClose', 0)
        except Exception:
            return 0

    def buy(self, ticker: str, quantity: int = None, amount: float = None) -> Optional[Order]:
        """Execute buy order"""
        ticker = ticker.upper()
        price = self.get_price(ticker)
        if price <= 0:
            print(f"Could not get price for {ticker}")
            return None

        if amount:
            quantity = int(amount / price)

        if quantity <= 0:
            return None

        cost = quantity * price
        if cost > self.account.cash:
            print(f"Insufficient funds: need ${cost:.2f}, have ${self.account.cash:.2f}")
            return None

        # Execute order
        self.account.cash -= cost

        if ticker in self.account.positions:
            pos = self.account.positions[ticker]
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
        else:
            self.account.positions[ticker] = Position(
                ticker=ticker, quantity=quantity, avg_cost=price
            )

        order = Order(
            id=f"ORD_{datetime.now().timestamp():.0f}",
            ticker=ticker, side=OrderSide.BUY, quantity=quantity,
            price=price, status=OrderStatus.FILLED,
            created_at=datetime.now(), filled_at=datetime.now(), filled_price=price
        )

        self.account.trades.append({
            'date': datetime.now().isoformat(),
            'ticker': ticker, 'side': 'buy',
            'quantity': quantity, 'price': price, 'total': cost
        })

        self._save_account()
        print(f"BUY {quantity} {ticker} @ ${price:.2f} = ${cost:.2f}")
        return order

    def sell(self, ticker: str, quantity: int = None) -> Optional[Order]:
        """Execute sell order"""
        ticker = ticker.upper()

        if ticker not in self.account.positions:
            print(f"No position in {ticker}")
            return None

        pos = self.account.positions[ticker]
        quantity = quantity or pos.quantity

        if quantity > pos.quantity:
            print(f"Not enough shares: have {pos.quantity}, want {quantity}")
            return None

        price = self.get_price(ticker)
        if price <= 0:
            return None

        proceeds = quantity * price
        self.account.cash += proceeds

        # Update position
        if quantity == pos.quantity:
            del self.account.positions[ticker]
        else:
            pos.quantity -= quantity

        order = Order(
            id=f"ORD_{datetime.now().timestamp():.0f}",
            ticker=ticker, side=OrderSide.SELL, quantity=quantity,
            price=price, status=OrderStatus.FILLED,
            created_at=datetime.now(), filled_at=datetime.now(), filled_price=price
        )

        self.account.trades.append({
            'date': datetime.now().isoformat(),
            'ticker': ticker, 'side': 'sell',
            'quantity': quantity, 'price': price, 'total': proceeds
        })

        self._save_account()
        print(f"SELL {quantity} {ticker} @ ${price:.2f} = ${proceeds:.2f}")
        return order

    def update_prices(self):
        """Update all position prices"""
        for ticker, pos in self.account.positions.items():
            pos.current_price = self.get_price(ticker)
            pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.quantity
            pos.unrealized_pnl_pct = (pos.current_price / pos.avg_cost - 1) * 100

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        self.update_prices()
        positions_value = sum(p.current_price * p.quantity for p in self.account.positions.values())
        return self.account.cash + positions_value

    def print_summary(self):
        """Print account summary"""
        self.update_prices()
        total_value = self.get_portfolio_value()
        total_pnl = total_value - self.account.initial_capital
        total_pnl_pct = (total_value / self.account.initial_capital - 1) * 100

        print("\n" + "=" * 70)
        print("EIMAS Paper Trading Account")
        print("=" * 70)
        print(f"\nAccount Value:    ${total_value:,.2f}")
        print(f"Cash:             ${self.account.cash:,.2f}")
        print(f"Total P&L:        ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")

        if self.account.positions:
            print("\nPOSITIONS:")
            print("-" * 70)
            print(f"{'Ticker':<8} {'Qty':>8} {'Avg Cost':>10} {'Price':>10} {'Value':>12} {'P&L':>12}")
            print("-" * 70)

            for ticker, pos in self.account.positions.items():
                value = pos.current_price * pos.quantity
                print(f"{ticker:<8} {pos.quantity:>8} ${pos.avg_cost:>8.2f} ${pos.current_price:>8.2f} ${value:>10,.2f} ${pos.unrealized_pnl:>+10,.2f}")

        print("=" * 70)

    def reset(self, initial_capital: float = None):
        """Reset account"""
        capital = initial_capital or self.account.initial_capital
        self.account = PaperAccount(initial_capital=capital, cash=capital)
        self._save_account()
        print(f"Account reset with ${capital:,.2f}")


if __name__ == "__main__":
    trader = PaperTrader(initial_capital=100000)

    # Example trades
    trader.buy("AAPL", quantity=10)
    trader.buy("MSFT", amount=5000)
    trader.buy("SPY", quantity=20)

    trader.print_summary()
