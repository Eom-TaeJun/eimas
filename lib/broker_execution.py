#!/usr/bin/env python3
"""
EIMAS broker execution adapter (IBKR-first simulation).

This module provides:
1) IBKR-oriented order state mapping
2) idempotency key generation
3) duplicate-safe execution registration to TradingDB

Current backend is a deterministic IBKR simulation over PaperTrader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from typing import Any, Dict, Optional

from lib.paper_trader import OrderStatus, PaperTrader
from lib.trading_db import Execution, SessionType, SignalAction, TradingDB


class BrokerName(str, Enum):
    IBKR = "ibkr"


class OrderState(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class BrokerOrderRequest:
    ticker: str
    side: str  # buy/sell
    quantity: float
    limit_price: float
    session: SessionType
    portfolio_id: int = 0
    reference_price: float = 0.0
    strategy_tag: str = "eimas.auto_paper_execution"
    explainability: Dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> SignalAction:
        return SignalAction.BUY if self.side.lower() == "buy" else SignalAction.SELL


def build_order_idempotency_key(
    *,
    broker: str,
    account_name: str,
    request: BrokerOrderRequest,
    scope_date: str,
) -> str:
    """
    Build deterministic idempotency key for duplicate-safe order registration.

    `scope_date` usually uses YYYY-MM-DD for daily dedupe behavior.
    """
    normalized = "|".join(
        [
            broker.strip().lower(),
            account_name.strip().lower(),
            scope_date.strip(),
            request.strategy_tag.strip().lower(),
            request.ticker.strip().upper(),
            request.side.strip().lower(),
            f"{float(request.quantity):.8f}",
            f"{float(request.limit_price):.8f}",
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _map_paper_status_to_execution(order_status: str) -> tuple[str, OrderState]:
    raw = (order_status or "").strip().lower()
    if raw == OrderStatus.FILLED.value:
        return "filled", OrderState.FILLED
    if raw == OrderStatus.PARTIAL.value:
        return "partial", OrderState.PARTIALLY_FILLED
    if raw == OrderStatus.CANCELLED.value:
        return "cancelled", OrderState.CANCELLED
    if raw == OrderStatus.REJECTED.value:
        return "rejected", OrderState.REJECTED
    if raw == OrderStatus.PENDING.value:
        return "pending", OrderState.SUBMITTED
    return "failed", OrderState.FAILED


class IBKRPaperExecutionRouter:
    """
    IBKR-first execution router using PaperTrader as current backend.

    This keeps the broker contract stable while we remain in paper mode.
    """

    def __init__(
        self,
        *,
        account_name: str,
        initial_capital: float,
        db: Optional[TradingDB] = None,
        trader: Optional[PaperTrader] = None,
        dry_run: bool = False,
    ):
        self.broker_name = BrokerName.IBKR.value
        self.account_name = account_name
        self.initial_capital = initial_capital
        self.db = db or TradingDB()
        self.trader = trader or PaperTrader(
            initial_capital=initial_capital,
            account_name=account_name,
        )
        self.dry_run = dry_run

    @staticmethod
    def _external_order_id_for_paper_id(order_id: int) -> str:
        return f"ibkr-sim-{int(order_id)}"

    def submit_limit_order(
        self,
        request: BrokerOrderRequest,
        *,
        scope_date: str,
    ) -> Dict[str, Any]:
        """Submit one order with idempotency guard."""
        idempotency_key = build_order_idempotency_key(
            broker=self.broker_name,
            account_name=self.account_name,
            request=request,
            scope_date=scope_date,
        )

        existing = self.db.find_execution_by_idempotency_key(idempotency_key)
        if existing and str(existing.get("status", "")).lower() in {"pending", "filled", "partial"}:
            return {
                "deduplicated": True,
                "execution_id": existing.get("id"),
                "external_order_id": existing.get("external_order_id"),
                "status": existing.get("status", "pending"),
                "order_state": existing.get("order_state", OrderState.SUBMITTED.value),
                "broker": existing.get("broker", self.broker_name),
                "idempotency_key": idempotency_key,
                "ticker": request.ticker,
                "side": request.side,
                "quantity": request.quantity,
                "limit_price": request.limit_price,
                "executed_price": existing.get("executed_price", 0.0),
                "commission": existing.get("commission", 0.0),
            }

        now = datetime.now()
        external_order_id = f"ibkr-sim-dry-{idempotency_key[:16]}"
        executed_price = 0.0
        commission = 0.0
        slippage = 0.0
        status = "pending"
        order_state = OrderState.SUBMITTED

        if not self.dry_run:
            order = self.trader.execute_order(
                ticker=request.ticker,
                side=request.side,
                quantity=request.quantity,
                order_type="limit",
                limit_price=request.limit_price,
            )
            status, order_state = _map_paper_status_to_execution(order.status.value)
            if order.id:
                external_order_id = self._external_order_id_for_paper_id(order.id)
            else:
                external_order_id = f"ibkr-sim-{now.strftime('%H%M%S%f')}"
            if order.status == OrderStatus.FILLED:
                executed_price = float(order.filled_price or 0.0)
                commission = float(order.commission or 0.0)
                if request.reference_price > 0:
                    slippage = abs(executed_price - request.reference_price) / request.reference_price

        explainability = {
            **(request.explainability or {}),
            "broker": self.broker_name,
            "strategy_tag": request.strategy_tag,
            "idempotency_scope_date": scope_date,
        }

        record = Execution(
            portfolio_id=request.portfolio_id,
            external_order_id=external_order_id,
            broker=self.broker_name,
            idempotency_key=idempotency_key,
            order_state=order_state.value,
            ticker=request.ticker,
            action=request.action,
            session=request.session,
            target_price=request.limit_price,
            executed_price=executed_price,
            shares=request.quantity,
            commission=commission,
            slippage=slippage,
            explainability=explainability,
            status=status,
            timestamp=now,
        )
        execution_id = self.db.save_execution(record)

        return {
            "deduplicated": False,
            "execution_id": execution_id,
            "external_order_id": external_order_id,
            "status": status,
            "order_state": order_state.value,
            "broker": self.broker_name,
            "idempotency_key": idempotency_key,
            "ticker": request.ticker,
            "side": request.side,
            "quantity": request.quantity,
            "limit_price": request.limit_price,
            "executed_price": executed_price,
            "commission": commission,
        }

    def poll_pending_orders(self) -> Dict[str, Any]:
        """Poll simulated broker pending orders and sync TradingDB state."""
        if self.dry_run:
            return {
                "account": self.account_name,
                "processed": 0,
                "filled": 0,
                "rejected": 0,
                "filled_orders": [],
                "dry_run": True,
            }

        poll_result = self.trader.process_pending_orders()
        for order_info in poll_result.get("filled_orders", []):
            order_id = order_info.get("id")
            if order_id is None:
                continue
            external_order_id = self._external_order_id_for_paper_id(int(order_id))
            self.db.update_execution_by_external_order_id(
                external_order_id=external_order_id,
                status="filled",
                order_state=OrderState.FILLED.value,
                executed_price=float(order_info.get("filled_price") or 0.0),
                commission=float(order_info.get("commission") or 0.0),
            )
        return poll_result

    def pending_order_count(self) -> int:
        return len(self.trader.get_pending_orders())


def build_ibkr_paper_router(
    *,
    account_name: str,
    initial_capital: float,
    db: Optional[TradingDB] = None,
    trader: Optional[PaperTrader] = None,
    dry_run: bool = False,
) -> IBKRPaperExecutionRouter:
    """Factory for IBKR-first execution router."""
    return IBKRPaperExecutionRouter(
        account_name=account_name,
        initial_capital=initial_capital,
        db=db,
        trader=trader,
        dry_run=dry_run,
    )

