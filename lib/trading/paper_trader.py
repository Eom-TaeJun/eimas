#!/usr/bin/env python3
"""
EIMAS Paper Trading System
===========================
시뮬레이션 트레이딩 시스템

주요 기능:
1. 가상 계좌 관리
2. 주문 시뮬레이션 실행
3. 포지션 및 손익 추적
4. 시그널 자동 실행

Usage:
    from lib.paper_trader import PaperTrader

    trader = PaperTrader(initial_capital=100000)
    trader.execute_order('SPY', 'buy', 100)
    print(trader.get_portfolio_summary())
"""

import json
import sqlite3
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "paper_trading.db"
COMMISSION_RATE = 0.0  # 무수수료 (실제로는 0.001 등)
SLIPPAGE_RATE = 0.001  # 0.1% 슬리피지


def _configure_yfinance_cache_dir() -> None:
    cache_dir = os.getenv("EIMAS_YFINANCE_CACHE_DIR", "/tmp/eimas_yfinance_cache").strip()
    if not cache_dir or cache_dir.lower() in {"off", "none", "disable", "false", "0"}:
        return
    target = Path(cache_dir).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(target))
    except Exception:
        return


_configure_yfinance_cache_dir()


# ============================================================================
# Data Classes
# ============================================================================

class OrderType(str, Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """주문"""
    id: int = 0
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0
    filled_quantity: float = 0
    commission: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    """포지션"""
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        return cls(**data)


@dataclass
class Trade:
    """거래 내역"""
    id: int
    ticker: str
    side: str
    quantity: float
    price: float
    commission: float
    realized_pnl: float
    timestamp: datetime


@dataclass
class PortfolioSummary:
    """포트폴리오 요약"""
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    positions: Dict[str, Position]
    daily_pnl: float = 0


# ============================================================================
# Paper Trader
# ============================================================================

class PaperTrader:
    """페이퍼 트레이딩 시스템"""

    def __init__(
        self,
        initial_capital: float = 100000,
        account_name: str = "default",
    ):
        self.account_name = account_name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # {ticker: {quantity, avg_cost}}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []

        # 데이터베이스 초기화
        self._init_db()
        self._load_account()

    def _init_db(self):
        """데이터베이스 초기화"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        # 계좌 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            initial_capital REAL,
            cash REAL,
            positions TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )''')

        # 주문 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name TEXT,
            ticker TEXT,
            side TEXT,
            order_type TEXT,
            quantity REAL,
            limit_price REAL,
            stop_price REAL,
            status TEXT,
            filled_price REAL,
            filled_quantity REAL,
            commission REAL,
            created_at TIMESTAMP,
            filled_at TIMESTAMP
        )''')

        # 거래 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name TEXT,
            ticker TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            commission REAL,
            realized_pnl REAL,
            timestamp TIMESTAMP
        )''')

        # 일일 스냅샷 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS daily_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_name TEXT,
            date DATE,
            cash REAL,
            positions_value REAL,
            total_value REAL,
            total_pnl REAL,
            created_at TIMESTAMP,
            UNIQUE(account_name, date)
        )''')

        conn.commit()
        conn.close()

    def _load_account(self):
        """계좌 로드"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            'SELECT initial_capital, cash, positions FROM accounts WHERE name = ?',
            (self.account_name,)
        )
        row = c.fetchone()

        if row:
            self.initial_capital = row[0]
            self.cash = row[1]
            self.positions = json.loads(row[2]) if row[2] else {}
            print(f"Loaded account '{self.account_name}': ${self.cash:,.2f} cash")
        else:
            # 새 계좌 생성
            c.execute(
                '''INSERT INTO accounts (name, initial_capital, cash, positions, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (self.account_name, self.initial_capital, self.cash, '{}', datetime.now(), datetime.now())
            )
            conn.commit()
            print(f"Created new account '{self.account_name}': ${self.initial_capital:,.2f}")

        conn.close()

    def _save_account(self):
        """계좌 저장"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            '''UPDATE accounts SET cash = ?, positions = ?, updated_at = ?
               WHERE name = ?''',
            (self.cash, json.dumps(self.positions), datetime.now(), self.account_name)
        )

        conn.commit()
        conn.close()

    def _lookup_local_price(self, ticker: str) -> float:
        """로컬 시장 CSV 캐시에서 최근 가격 조회"""
        market_dir = PROJECT_ROOT / "data" / "market"
        if not market_dir.exists():
            return 0.0

        normalized = (
            ticker.replace("-", "_")
            .replace("^", "")
            .replace("/", "_")
        )
        candidates = [
            market_dir / f"yfinance_{normalized}_1d.csv",
            market_dir / f"yfinance_{normalized}_1h.csv",
            market_dir / f"cryptocompare_{normalized}_1d.csv",
            market_dir / f"cryptocompare_{normalized}_1h.csv",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            close_col = None
            for col in ("Close", "close", "Adj Close", "adj_close"):
                if col in df.columns:
                    close_col = col
                    break
            if close_col is None:
                continue
            try:
                value = float(df[close_col].dropna().iloc[-1])
            except Exception:
                continue
            if value > 0:
                return value

        return 0.0

    def get_current_price(self, ticker: str) -> float:
        """현재 가격 조회"""
        local_first = os.getenv("EIMAS_PAPER_LOCAL_FIRST", "true").strip().lower()
        if local_first not in {"0", "false", "off", "no"}:
            local_price = self._lookup_local_price(ticker)
            if local_price > 0:
                return local_price

        try:
            data = yf.download(ticker, period="1d", progress=False, threads=False)
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
        except Exception:
            pass
        local_price = self._lookup_local_price(ticker)
        if local_price > 0:
            return local_price
        return 0.0

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """여러 종목 현재 가격 조회"""
        if not tickers:
            return {}

        prices: Dict[str, float] = {}
        for ticker in tickers:
            px = self.get_current_price(ticker)
            if px > 0:
                prices[ticker] = px
        return prices

    # ========================================================================
    # Order Execution
    # ========================================================================

    def execute_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float = None,
    ) -> Order:
        """주문 실행"""
        side = OrderSide(side.lower())
        order_type = OrderType(order_type.lower())
        quantity = float(quantity)
        if quantity <= 0:
            return Order(
                ticker=ticker,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=OrderStatus.REJECTED,
            )

        # 현재 가격 조회
        current_price = self.get_current_price(ticker)
        if current_price <= 0:
            order = Order(
                ticker=ticker,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=OrderStatus.REJECTED,
            )
            return order

        # 리밋 주문 체크
        if order_type == OrderType.LIMIT and limit_price:
            if side == OrderSide.BUY and current_price > limit_price:
                order = Order(
                    ticker=ticker,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    limit_price=limit_price,
                    status=OrderStatus.PENDING,
                )
                self.orders.append(order)
                order.id = self._save_order(order)
                print(
                    f"Order pending: {side.value.upper()} {quantity} {ticker} "
                    f"@ LIMIT ${float(limit_price):.4f} (current ${current_price:.4f})"
                )
                return order
            elif side == OrderSide.SELL and current_price < limit_price:
                order = Order(
                    ticker=ticker,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    limit_price=limit_price,
                    status=OrderStatus.PENDING,
                )
                self.orders.append(order)
                order.id = self._save_order(order)
                print(
                    f"Order pending: {side.value.upper()} {quantity} {ticker} "
                    f"@ LIMIT ${float(limit_price):.4f} (current ${current_price:.4f})"
                )
                return order

        fill_price = self._compute_fill_price(
            side=side,
            current_price=current_price,
            order_type=order_type,
            limit_price=limit_price,
        )

        # 주문 금액 계산
        order_value = fill_price * quantity
        commission = order_value * COMMISSION_RATE

        # 매수 시 현금 체크
        if side == OrderSide.BUY:
            total_cost = order_value + commission
            if total_cost > self.cash:
                order = Order(
                    ticker=ticker,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.REJECTED,
                )
                print(f"Order rejected: Insufficient cash (${self.cash:,.2f} < ${total_cost:,.2f})")
                return order

        # 매도 시 포지션 체크
        if side == OrderSide.SELL:
            current_qty = self.positions.get(ticker, {}).get('quantity', 0)
            if quantity > current_qty:
                order = Order(
                    ticker=ticker,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    status=OrderStatus.REJECTED,
                )
                print(f"Order rejected: Insufficient position ({current_qty} < {quantity})")
                return order

        # 주문 실행
        realized_pnl = 0

        if side == OrderSide.BUY:
            self.cash -= (order_value + commission)

            if ticker in self.positions:
                # 기존 포지션에 추가
                old_qty = self.positions[ticker]['quantity']
                old_cost = self.positions[ticker]['avg_cost']
                new_qty = old_qty + quantity
                new_cost = (old_qty * old_cost + quantity * fill_price) / new_qty
                self.positions[ticker] = {'quantity': new_qty, 'avg_cost': new_cost}
            else:
                self.positions[ticker] = {'quantity': quantity, 'avg_cost': fill_price}

        else:  # SELL
            self.cash += (order_value - commission)

            avg_cost = self.positions[ticker]['avg_cost']
            realized_pnl = (fill_price - avg_cost) * quantity - commission

            new_qty = self.positions[ticker]['quantity'] - quantity
            if new_qty <= 0.0001:
                del self.positions[ticker]
            else:
                self.positions[ticker]['quantity'] = new_qty

        # 주문 생성
        order = Order(
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            status=OrderStatus.FILLED,
            filled_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            filled_at=datetime.now(),
        )

        # 거래 기록
        trade = Trade(
            id=len(self.trades) + 1,
            ticker=ticker,
            side=side.value,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            realized_pnl=realized_pnl,
            timestamp=datetime.now(),
        )
        self.trades.append(trade)

        # DB 저장
        order.id = self._save_order(order)
        self._save_trade(trade)
        self._save_account()

        print(f"Order filled: {side.value.upper()} {quantity} {ticker} @ ${fill_price:.2f}")
        return order

    def _compute_fill_price(
        self,
        side: OrderSide,
        current_price: float,
        order_type: OrderType,
        limit_price: Optional[float],
    ) -> float:
        """현재가/슬리피지/리밋을 반영한 체결가 계산"""
        if side == OrderSide.BUY:
            slipped = current_price * (1 + SLIPPAGE_RATE)
            if order_type == OrderType.LIMIT and limit_price is not None:
                return min(slipped, float(limit_price))
            return slipped

        slipped = current_price * (1 - SLIPPAGE_RATE)
        if order_type == OrderType.LIMIT and limit_price is not None:
            return max(slipped, float(limit_price))
        return slipped

    def _save_order(self, order: Order) -> int:
        """주문 DB 저장"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            '''INSERT INTO orders
               (account_name, ticker, side, order_type, quantity, limit_price, stop_price,
                status, filled_price, filled_quantity, commission, created_at, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (self.account_name, order.ticker, order.side.value, order.order_type.value,
             order.quantity, order.limit_price, order.stop_price, order.status.value,
             order.filled_price, order.filled_quantity, order.commission,
             order.created_at, order.filled_at)
        )

        order_id = c.lastrowid
        conn.commit()
        conn.close()
        return order_id

    def _save_trade(self, trade: Trade) -> int:
        """거래 DB 저장"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            '''INSERT INTO trades
               (account_name, ticker, side, quantity, price, commission, realized_pnl, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (self.account_name, trade.ticker, trade.side, trade.quantity,
             trade.price, trade.commission, trade.realized_pnl, trade.timestamp)
        )

        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def _update_order_fill(self, order_id: int, order: Order):
        """기존 대기 주문을 체결 상태로 업데이트"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """UPDATE orders
               SET status = ?, filled_price = ?, filled_quantity = ?, commission = ?, filled_at = ?
               WHERE id = ? AND account_name = ?""",
            (
                order.status.value,
                order.filled_price,
                order.filled_quantity,
                order.commission,
                order.filled_at,
                order_id,
                self.account_name,
            ),
        )
        conn.commit()
        conn.close()

    def _update_order_status(self, order_id: int, status: OrderStatus):
        """기존 주문 상태 업데이트"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "UPDATE orders SET status = ? WHERE id = ? AND account_name = ?",
            (status.value, order_id, self.account_name),
        )
        conn.commit()
        conn.close()

    def get_pending_orders(self, limit: int = 200) -> List[Order]:
        """대기 중인 주문 조회"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """SELECT id, ticker, side, order_type, quantity, limit_price, stop_price,
                      status, filled_price, filled_quantity, commission, created_at, filled_at
               FROM orders
               WHERE account_name = ? AND status = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (self.account_name, OrderStatus.PENDING.value, int(limit)),
        )
        rows = c.fetchall()
        conn.close()

        pending: List[Order] = []
        for row in rows:
            pending.append(
                Order(
                    id=row[0],
                    ticker=row[1],
                    side=OrderSide(row[2]),
                    order_type=OrderType(row[3]),
                    quantity=float(row[4]),
                    limit_price=float(row[5]) if row[5] is not None else None,
                    stop_price=float(row[6]) if row[6] is not None else None,
                    status=OrderStatus(row[7]),
                    filled_price=float(row[8] or 0.0),
                    filled_quantity=float(row[9] or 0.0),
                    commission=float(row[10] or 0.0),
                    created_at=datetime.fromisoformat(row[11]) if isinstance(row[11], str) else datetime.now(),
                    filled_at=datetime.fromisoformat(row[12]) if isinstance(row[12], str) and row[12] else None,
                )
            )
        return pending

    def process_pending_orders(self) -> Dict[str, Any]:
        """
        대기 주문 폴링 및 자동 체결.
        LIMIT 조건을 만족하면 즉시 모의 체결한다.
        """
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """SELECT id, ticker, side, order_type, quantity, limit_price
               FROM orders
               WHERE account_name = ? AND status = ?
               ORDER BY created_at ASC""",
            (self.account_name, OrderStatus.PENDING.value),
        )
        rows = c.fetchall()
        conn.close()

        processed = 0
        filled = 0
        rejected = 0
        filled_orders: List[Order] = []

        for row in rows:
            processed += 1
            order_id, ticker, side_raw, order_type_raw, quantity, limit_price = row
            side = OrderSide(side_raw)
            order_type = OrderType(order_type_raw)
            qty = float(quantity)
            limit = float(limit_price) if limit_price is not None else None

            current_price = self.get_current_price(ticker)
            if current_price <= 0:
                continue

            is_triggered = True
            if order_type == OrderType.LIMIT and limit is not None:
                if side == OrderSide.BUY:
                    is_triggered = current_price <= limit
                else:
                    is_triggered = current_price >= limit

            if not is_triggered:
                continue

            fill_price = self._compute_fill_price(
                side=side,
                current_price=current_price,
                order_type=order_type,
                limit_price=limit,
            )

            order_value = fill_price * qty
            commission = order_value * COMMISSION_RATE

            if side == OrderSide.BUY:
                total_cost = order_value + commission
                if total_cost > self.cash:
                    self._update_order_status(order_id, OrderStatus.REJECTED)
                    rejected += 1
                    continue
            else:
                current_qty = self.positions.get(ticker, {}).get("quantity", 0.0)
                if qty > current_qty + 1e-9:
                    self._update_order_status(order_id, OrderStatus.REJECTED)
                    rejected += 1
                    continue

            realized_pnl = 0.0
            if side == OrderSide.BUY:
                self.cash -= (order_value + commission)
                if ticker in self.positions:
                    old_qty = self.positions[ticker]["quantity"]
                    old_cost = self.positions[ticker]["avg_cost"]
                    new_qty = old_qty + qty
                    new_cost = (old_qty * old_cost + qty * fill_price) / new_qty
                    self.positions[ticker] = {"quantity": new_qty, "avg_cost": new_cost}
                else:
                    self.positions[ticker] = {"quantity": qty, "avg_cost": fill_price}
            else:
                self.cash += (order_value - commission)
                avg_cost = self.positions[ticker]["avg_cost"]
                realized_pnl = (fill_price - avg_cost) * qty - commission
                new_qty = self.positions[ticker]["quantity"] - qty
                if new_qty <= 0.0001:
                    del self.positions[ticker]
                else:
                    self.positions[ticker]["quantity"] = new_qty

            fill_ts = datetime.now()
            order = Order(
                id=int(order_id),
                ticker=ticker,
                side=side,
                order_type=order_type,
                quantity=qty,
                limit_price=limit,
                status=OrderStatus.FILLED,
                filled_price=fill_price,
                filled_quantity=qty,
                commission=commission,
                filled_at=fill_ts,
            )
            self._update_order_fill(int(order_id), order)

            trade = Trade(
                id=len(self.trades) + 1,
                ticker=ticker,
                side=side.value,
                quantity=qty,
                price=fill_price,
                commission=commission,
                realized_pnl=realized_pnl,
                timestamp=fill_ts,
            )
            self.trades.append(trade)
            self._save_trade(trade)

            filled += 1
            filled_orders.append(order)
            print(f"Pending order filled: {side.value.upper()} {qty} {ticker} @ ${fill_price:.2f}")

        if filled > 0:
            self._save_account()

        return {
            "account": self.account_name,
            "processed": processed,
            "filled": filled,
            "rejected": rejected,
            "filled_orders": [
                {
                    "id": o.id,
                    "ticker": o.ticker,
                    "side": o.side.value,
                    "order_type": o.order_type.value,
                    "quantity": o.quantity,
                    "limit_price": o.limit_price,
                    "status": o.status.value,
                    "filled_price": o.filled_price,
                    "filled_quantity": o.filled_quantity,
                    "commission": o.commission,
                    "created_at": o.created_at.isoformat() if o.created_at else None,
                    "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                }
                for o in filled_orders
            ],
        }

    # ========================================================================
    # Portfolio Management
    # ========================================================================

    def get_positions(self) -> List[Position]:
        """포지션 목록"""
        if not self.positions:
            return []

        tickers = list(self.positions.keys())
        prices = {}

        # 가격 조회
        prices = self.get_current_prices(tickers)

        positions = []
        for ticker, pos in self.positions.items():
            qty = pos['quantity']
            avg_cost = pos['avg_cost']
            current = prices.get(ticker, avg_cost)
            market_value = qty * current
            unrealized_pnl = (current - avg_cost) * qty
            unrealized_pnl_pct = (current / avg_cost - 1) * 100 if avg_cost > 0 else 0

            positions.append(Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=avg_cost,
                current_price=current,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
            ))

        return positions

    def get_portfolio_summary(self) -> PortfolioSummary:
        """포트폴리오 요약"""
        positions = self.get_positions()

        positions_value = sum(p.market_value for p in positions)
        total_value = self.cash + positions_value
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_value / self.initial_capital - 1) * 100

        return PortfolioSummary(
            timestamp=datetime.now(),
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            positions={p.ticker: p for p in positions},
        )

    def get_trade_history(self, days: int = 30) -> List[Trade]:
        """거래 내역 조회"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        cutoff = datetime.now() - timedelta(days=days)
        c.execute(
            '''SELECT id, ticker, side, quantity, price, commission, realized_pnl, timestamp
               FROM trades
               WHERE account_name = ? AND timestamp > ?
               ORDER BY timestamp DESC''',
            (self.account_name, cutoff)
        )

        trades = []
        for row in c.fetchall():
            trades.append(Trade(
                id=row[0],
                ticker=row[1],
                side=row[2],
                quantity=row[3],
                price=row[4],
                commission=row[5],
                realized_pnl=row[6],
                timestamp=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
            ))

        conn.close()
        return trades

    def get_realized_pnl(self) -> float:
        """실현 손익 합계"""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            'SELECT SUM(realized_pnl) FROM trades WHERE account_name = ?',
            (self.account_name,)
        )
        result = c.fetchone()[0]
        conn.close()

        return result or 0.0

    # ========================================================================
    # Signal Integration
    # ========================================================================

    def execute_portfolio_weights(
        self,
        target_weights: Dict[str, float],
        total_value: float = None,
    ) -> List[Order]:
        """목표 비중으로 포트폴리오 조정"""
        summary = self.get_portfolio_summary()
        total_value = total_value or summary.total_value

        current_weights = {}
        for ticker, pos in summary.positions.items():
            current_weights[ticker] = pos.market_value / total_value

        orders = []

        # 현재 가격 조회
        all_tickers = list(set(list(target_weights.keys()) + list(current_weights.keys())))
        prices = self.get_current_prices(all_tickers)

        # 매도 먼저 (현금 확보)
        for ticker, current in current_weights.items():
            target = target_weights.get(ticker, 0)
            if current > target + 0.01:  # 1% 이상 차이
                price = prices.get(ticker, 0)
                if price > 0:
                    diff_value = (current - target) * total_value
                    shares = int(diff_value / price)
                    if shares > 0:
                        order = self.execute_order(ticker, 'sell', shares)
                        orders.append(order)

        # 매수
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0)
            if target > current + 0.01:  # 1% 이상 차이
                price = prices.get(ticker, 0)
                if price > 0:
                    diff_value = (target - current) * total_value
                    shares = int(diff_value / price)
                    if shares > 0 and shares * price < self.cash:
                        order = self.execute_order(ticker, 'buy', shares)
                        orders.append(order)

        return orders

    # ========================================================================
    # Reporting
    # ========================================================================

    def print_summary(self):
        """요약 출력"""
        summary = self.get_portfolio_summary()

        print("\n" + "=" * 50)
        print(f"Paper Trading Account: {self.account_name}")
        print("=" * 50)

        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Cash: ${summary.cash:,.2f}")
        print(f"Positions Value: ${summary.positions_value:,.2f}")
        print(f"Total Value: ${summary.total_value:,.2f}")
        print(f"Total P&L: ${summary.total_pnl:,.2f} ({summary.total_pnl_pct:+.2f}%)")

        if summary.positions:
            print("\nPositions:")
            print(f"{'Ticker':<8} {'Qty':<10} {'Avg Cost':<12} {'Current':<12} {'Value':<12} {'P&L'}")
            print("-" * 70)
            for ticker, pos in summary.positions.items():
                print(f"{ticker:<8} {pos.quantity:<10.2f} ${pos.avg_cost:<10.2f} ${pos.current_price:<10.2f} ${pos.market_value:<10.2f} ${pos.unrealized_pnl:+.2f}")

        realized = self.get_realized_pnl()
        print(f"\nRealized P&L: ${realized:,.2f}")
        print("=" * 50)

    def save_daily_snapshot(self):
        """일일 스냅샷 저장"""
        summary = self.get_portfolio_summary()

        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            '''INSERT OR REPLACE INTO daily_snapshots
               (account_name, date, cash, positions_value, total_value, total_pnl, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (self.account_name, datetime.now().date(), summary.cash,
             summary.positions_value, summary.total_value, summary.total_pnl, datetime.now())
        )

        conn.commit()
        conn.close()

    def reset_account(self):
        """계좌 초기화"""
        self.cash = self.initial_capital
        self.positions = {}
        self._save_account()
        print(f"Account reset to ${self.initial_capital:,.2f}")


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_paper_trade(
    ticker: str,
    side: str,
    quantity: int,
    account: str = "default",
) -> Order:
    """빠른 페이퍼 트레이드"""
    trader = PaperTrader(account_name=account)
    return trader.execute_order(ticker, side, quantity)


def get_paper_portfolio(account: str = "default") -> PortfolioSummary:
    """페이퍼 포트폴리오 조회"""
    trader = PaperTrader(account_name=account)
    return trader.get_portfolio_summary()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Paper Trading Test")
    print("=" * 60)

    # 테스트 계좌 생성
    trader = PaperTrader(initial_capital=100000, account_name="test_account")

    # 현재 상태
    trader.print_summary()

    # 매수 테스트
    print("\n--- Buy Orders ---")
    trader.execute_order('SPY', 'buy', 10)
    trader.execute_order('TLT', 'buy', 20)
    trader.execute_order('GLD', 'buy', 15)

    # 상태 확인
    trader.print_summary()

    # 매도 테스트
    print("\n--- Sell Order ---")
    trader.execute_order('SPY', 'sell', 5)

    # 최종 상태
    trader.print_summary()

    # 거래 내역
    print("\nRecent Trades:")
    trades = trader.get_trade_history()
    for t in trades[:5]:
        print(f"  {t.ticker} {t.side.upper()} {t.quantity} @ ${t.price:.2f} | PnL: ${t.realized_pnl:.2f}")
