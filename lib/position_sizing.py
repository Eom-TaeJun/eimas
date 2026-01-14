#!/usr/bin/env python3
"""
EIMAS Position Sizing Calculator
=================================
Calculate optimal position sizes based on risk management.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskModel(Enum):
    FIXED_DOLLAR = "fixed_dollar"       # Fixed dollar amount per trade
    FIXED_PERCENT = "fixed_percent"     # Fixed % of portfolio
    VOLATILITY = "volatility"           # Volatility-adjusted
    KELLY = "kelly"                     # Kelly criterion
    ATR = "atr"                         # ATR-based


@dataclass
class PositionSize:
    ticker: str
    current_price: float
    shares: int
    dollar_amount: float
    portfolio_pct: float
    risk_per_trade: float
    stop_loss_price: float
    stop_loss_pct: float
    take_profit_price: Optional[float]
    risk_reward_ratio: Optional[float]
    max_loss: float
    model_used: RiskModel


class PositionSizer:
    """Calculate position sizes using various methods"""

    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.max_position_pct = 0.10  # Max 10% per position
        self.max_risk_per_trade = 0.02  # Max 2% risk per trade

    def get_price(self, ticker: str) -> float:
        """Get current price"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('regularMarketPrice') or stock.info.get('previousClose', 0)
        except Exception:
            return 0

    def get_atr(self, ticker: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            df = yf.download(ticker, period="3mo", progress=False)
            if df.empty:
                return 0

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            high = df['High']
            low = df['Low']
            close = df['Close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]

            return float(atr) if not np.isnan(atr) else 0
        except Exception:
            return 0

    def get_volatility(self, ticker: str, period: int = 20) -> float:
        """Calculate annualized volatility"""
        try:
            df = yf.download(ticker, period="3mo", progress=False)
            if df.empty:
                return 0

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            returns = df['Close'].pct_change().dropna()
            vol = returns.rolling(period).std().iloc[-1] * np.sqrt(252)
            return float(vol) if not np.isnan(vol) else 0.3
        except Exception:
            return 0.3  # Default 30%

    def fixed_dollar(self, ticker: str, amount: float, stop_loss_pct: float = 5.0) -> PositionSize:
        """Fixed dollar amount position sizing"""
        price = self.get_price(ticker)
        if price <= 0:
            return None

        shares = int(amount / price)
        actual_amount = shares * price
        portfolio_pct = actual_amount / self.portfolio_value * 100

        stop_loss_price = price * (1 - stop_loss_pct / 100)
        max_loss = shares * (price - stop_loss_price)
        risk_per_trade = max_loss / self.portfolio_value * 100

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            dollar_amount=actual_amount,
            portfolio_pct=portfolio_pct,
            risk_per_trade=risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=None,
            risk_reward_ratio=None,
            max_loss=max_loss,
            model_used=RiskModel.FIXED_DOLLAR
        )

    def fixed_percent(self, ticker: str, risk_pct: float = 1.0, stop_loss_pct: float = 5.0) -> PositionSize:
        """Fixed percent risk position sizing"""
        price = self.get_price(ticker)
        if price <= 0:
            return None

        # Calculate position size based on risk tolerance
        risk_amount = self.portfolio_value * (risk_pct / 100)
        price_drop = price * (stop_loss_pct / 100)
        shares = int(risk_amount / price_drop)

        actual_amount = shares * price
        portfolio_pct = actual_amount / self.portfolio_value * 100

        # Apply max position limit
        if portfolio_pct > self.max_position_pct * 100:
            shares = int(self.portfolio_value * self.max_position_pct / price)
            actual_amount = shares * price
            portfolio_pct = actual_amount / self.portfolio_value * 100

        stop_loss_price = price * (1 - stop_loss_pct / 100)
        max_loss = shares * (price - stop_loss_price)
        risk_per_trade = max_loss / self.portfolio_value * 100

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            dollar_amount=actual_amount,
            portfolio_pct=portfolio_pct,
            risk_per_trade=risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=None,
            risk_reward_ratio=None,
            max_loss=max_loss,
            model_used=RiskModel.FIXED_PERCENT
        )

    def volatility_adjusted(self, ticker: str, target_vol: float = 0.15, risk_pct: float = 1.0) -> PositionSize:
        """Volatility-adjusted position sizing"""
        price = self.get_price(ticker)
        vol = self.get_volatility(ticker)

        if price <= 0 or vol <= 0:
            return None

        # Adjust position based on volatility
        vol_scalar = target_vol / vol
        base_amount = self.portfolio_value * (risk_pct / 100) * 10  # Base position
        adjusted_amount = base_amount * vol_scalar

        # Cap at max position
        max_amount = self.portfolio_value * self.max_position_pct
        adjusted_amount = min(adjusted_amount, max_amount)

        shares = int(adjusted_amount / price)
        actual_amount = shares * price
        portfolio_pct = actual_amount / self.portfolio_value * 100

        # Use 2 sigma for stop loss
        stop_loss_pct = vol / np.sqrt(252) * 2 * 100  # 2-day 2-sigma
        stop_loss_price = price * (1 - stop_loss_pct / 100)
        max_loss = shares * (price - stop_loss_price)
        risk_per_trade = max_loss / self.portfolio_value * 100

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            dollar_amount=actual_amount,
            portfolio_pct=portfolio_pct,
            risk_per_trade=risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=None,
            risk_reward_ratio=None,
            max_loss=max_loss,
            model_used=RiskModel.VOLATILITY
        )

    def atr_based(self, ticker: str, atr_multiplier: float = 2.0, risk_pct: float = 1.0) -> PositionSize:
        """ATR-based position sizing"""
        import pandas as pd

        price = self.get_price(ticker)
        atr = self.get_atr(ticker)

        if price <= 0 or atr <= 0:
            return None

        # Stop loss at N x ATR
        stop_distance = atr * atr_multiplier
        stop_loss_pct = stop_distance / price * 100

        # Position size based on risk
        risk_amount = self.portfolio_value * (risk_pct / 100)
        shares = int(risk_amount / stop_distance)

        actual_amount = shares * price
        portfolio_pct = actual_amount / self.portfolio_value * 100

        # Apply max position limit
        if portfolio_pct > self.max_position_pct * 100:
            shares = int(self.portfolio_value * self.max_position_pct / price)
            actual_amount = shares * price
            portfolio_pct = actual_amount / self.portfolio_value * 100

        stop_loss_price = price - stop_distance
        max_loss = shares * stop_distance
        risk_per_trade = max_loss / self.portfolio_value * 100

        # Target at 2:1 R:R
        take_profit_price = price + (stop_distance * 2)

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            dollar_amount=actual_amount,
            portfolio_pct=portfolio_pct,
            risk_per_trade=risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=take_profit_price,
            risk_reward_ratio=2.0,
            max_loss=max_loss,
            model_used=RiskModel.ATR
        )

    def kelly_criterion(self, ticker: str, win_rate: float = 0.55, avg_win_loss_ratio: float = 1.5,
                       fraction: float = 0.25) -> PositionSize:
        """Kelly criterion position sizing (fractional)"""
        price = self.get_price(ticker)
        if price <= 0:
            return None

        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win rate, q = 1-p
        b = avg_win_loss_ratio
        p = win_rate
        q = 1 - p

        kelly_pct = (b * p - q) / b
        kelly_pct = max(0, kelly_pct)  # Can't be negative

        # Use fractional Kelly for safety
        position_pct = kelly_pct * fraction
        position_pct = min(position_pct, self.max_position_pct)

        amount = self.portfolio_value * position_pct
        shares = int(amount / price)
        actual_amount = shares * price
        portfolio_pct = actual_amount / self.portfolio_value * 100

        # Estimate stop based on avg loss
        stop_loss_pct = 100 / avg_win_loss_ratio / (1 / (1 - win_rate)) * 2
        stop_loss_pct = min(stop_loss_pct, 10)  # Cap at 10%
        stop_loss_price = price * (1 - stop_loss_pct / 100)
        max_loss = shares * (price - stop_loss_price)
        risk_per_trade = max_loss / self.portfolio_value * 100

        return PositionSize(
            ticker=ticker,
            current_price=price,
            shares=shares,
            dollar_amount=actual_amount,
            portfolio_pct=portfolio_pct,
            risk_per_trade=risk_per_trade,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_price=price * (1 + stop_loss_pct / 100 * avg_win_loss_ratio),
            risk_reward_ratio=avg_win_loss_ratio,
            max_loss=max_loss,
            model_used=RiskModel.KELLY
        )

    def calculate_all_methods(self, ticker: str, risk_pct: float = 1.0) -> Dict[RiskModel, PositionSize]:
        """Calculate position size using all methods"""
        return {
            RiskModel.FIXED_DOLLAR: self.fixed_dollar(ticker, self.portfolio_value * 0.05),
            RiskModel.FIXED_PERCENT: self.fixed_percent(ticker, risk_pct),
            RiskModel.VOLATILITY: self.volatility_adjusted(ticker, risk_pct=risk_pct),
            RiskModel.ATR: self.atr_based(ticker, risk_pct=risk_pct),
            RiskModel.KELLY: self.kelly_criterion(ticker)
        }

    def print_report(self, ticker: str, risk_pct: float = 1.0):
        """Print position sizing report"""
        print("\n" + "=" * 80)
        print(f"Position Sizing Report: {ticker}")
        print("=" * 80)
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Risk Tolerance: {risk_pct}% (${self.portfolio_value * risk_pct / 100:,.2f})")

        results = self.calculate_all_methods(ticker, risk_pct)

        print(f"\n{'Method':<20} {'Shares':>8} {'Amount':>12} {'Port %':>8} {'Risk %':>8} {'Stop':>10} {'Max Loss':>12}")
        print("-" * 80)

        for model, pos in results.items():
            if pos:
                print(f"{model.value:<20} {pos.shares:>8} ${pos.dollar_amount:>10,.2f} "
                      f"{pos.portfolio_pct:>7.1f}% {pos.risk_per_trade:>7.2f}% "
                      f"${pos.stop_loss_price:>8.2f} ${pos.max_loss:>10,.2f}")

        # Recommendation
        print("\n" + "-" * 80)
        print("RECOMMENDATION:")
        print("-" * 80)

        atr_pos = results.get(RiskModel.ATR)
        if atr_pos:
            print(f"\n📊 ATR-Based (Recommended for Swing Trading):")
            print(f"   Buy {atr_pos.shares} shares of {ticker} at ${atr_pos.current_price:.2f}")
            print(f"   Stop Loss: ${atr_pos.stop_loss_price:.2f} ({atr_pos.stop_loss_pct:.1f}% below entry)")
            if atr_pos.take_profit_price:
                print(f"   Take Profit: ${atr_pos.take_profit_price:.2f} (2:1 R:R)")
            print(f"   Max Risk: ${atr_pos.max_loss:.2f} ({atr_pos.risk_per_trade:.2f}% of portfolio)")

        print("=" * 80)


# Need pandas for ATR calculation
import pandas as pd

if __name__ == "__main__":
    sizer = PositionSizer(portfolio_value=100000)

    # Calculate for a specific stock
    sizer.print_report("NVDA", risk_pct=1.0)
    sizer.print_report("AAPL", risk_pct=1.0)
