#!/usr/bin/env python3
"""
Broker Execution Layer
======================
실제 브로커를 통한 주문 실행 레이어

지원 브로커:
1. Alpaca (무료 페이퍼 트레이딩)
2. Interactive Brokers (IBKR) - Future
3. TD Ameritrade - Future

경제학적 배경:
- Execution Quality (실행 품질): 최적 가격 달성
- Market Impact Cost: 대량 주문의 시장 충격 최소화
- Smart Order Routing: 최적 거래소 선택
- Limit Order Book Dynamics

핵심 기능:
1. 주문 실행 (시장가, 지정가, 스톱)
2. 포지션 관리
3. 계좌 정보 조회
4. 실행 품질 모니터링
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import time

# Optional imports
try:
    from alpaca_trade_api import REST, Stream
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARN] alpaca-trade-api not available. Install: pip install alpaca-trade-api")


class BrokerType(str, Enum):
    """브로커 타입"""
    ALPACA = "alpaca"
    IBKR = "ibkr"
    TD_AMERITRADE = "td_ameritrade"
    PAPER = "paper"  # 내부 페이퍼 트레이딩


class OrderType(str, Enum):
    """주문 유형"""
    MARKET = "market"          # 시장가
    LIMIT = "limit"            # 지정가
    STOP = "stop"              # 스톱
    STOP_LIMIT = "stop_limit"  # 스톱 지정가
    TRAILING_STOP = "trailing_stop"  # 트레일링 스톱


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    """주문 유효 시간"""
    DAY = "day"           # 당일 유효
    GTC = "gtc"           # 취소 시까지 유효 (Good Till Canceled)
    IOC = "ioc"           # 즉시 체결 또는 취소 (Immediate or Cancel)
    FOK = "fok"           # 전량 즉시 체결 또는 취소 (Fill or Kill)


@dataclass
class Order:
    """주문"""
    id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    status: OrderStatus
    submitted_at: str

    # 가격 정보
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0

    # 추가 정보
    time_in_force: TimeInForce = TimeInForce.DAY
    filled_at: Optional[str] = None
    broker_order_id: Optional[str] = None  # 브로커 측 주문 ID
    commission: float = 0.0

    # 실행 품질
    slippage: float = 0.0  # 예상가 대비 실제 체결가 차이
    market_impact: float = 0.0  # 시장 충격 비용

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'ticker': self.ticker,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'status': self.status.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'filled_price': self.filled_price,
            'filled_quantity': self.filled_quantity,
            'time_in_force': self.time_in_force.value,
            'submitted_at': self.submitted_at,
            'filled_at': self.filled_at,
            'broker_order_id': self.broker_order_id,
            'commission': self.commission,
            'slippage': self.slippage,
            'market_impact': self.market_impact
        }


@dataclass
class Position:
    """포지션"""
    ticker: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AccountInfo:
    """계좌 정보"""
    account_id: str
    broker: BrokerType
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    positions_value: float
    last_updated: str

    # 리스크 메트릭
    margin_used: float = 0.0
    margin_available: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'broker': self.broker.value,
            'cash': self.cash,
            'buying_power': self.buying_power,
            'portfolio_value': self.portfolio_value,
            'equity': self.equity,
            'positions_value': self.positions_value,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'last_updated': self.last_updated
        }


@dataclass
class ExecutionReport:
    """실행 품질 리포트"""
    order_id: str
    ticker: str
    intended_price: float      # 의도한 가격
    executed_price: float      # 실제 체결가
    slippage_bps: float        # 슬리피지 (basis points)
    slippage_dollars: float    # 슬리피지 (달러)
    commission: float
    total_cost: float          # 총 비용 (슬리피지 + 수수료)
    execution_time_ms: float   # 실행 시간 (밀리초)
    timestamp: str


class OrderExecutor:
    """
    주문 실행 레이어 (Broker-agnostic)

    역할:
    - 브로커 API를 추상화
    - 주문 실행 및 관리
    - 포지션 조회
    - 실행 품질 모니터링
    """

    def __init__(
        self,
        broker: BrokerType = BrokerType.ALPACA,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            broker: 브로커 타입
            api_key: API 키
            api_secret: API 시크릿
            paper_trading: 페이퍼 트레이딩 여부
            verbose: 로그 출력
        """
        self.broker = broker
        self.paper_trading = paper_trading
        self.verbose = verbose

        self.logger = self._setup_logger()

        # API 클라이언트 초기화
        if broker == BrokerType.ALPACA:
            if not ALPACA_AVAILABLE:
                raise ImportError("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")

            self.api_key = api_key or os.getenv('ALPACA_API_KEY')
            self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY')

            if not self.api_key or not self.api_secret:
                raise ValueError("Alpaca API credentials not provided")

            # Base URL (Paper vs Live)
            base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"

            self.client = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=base_url
            )

            self.logger.info(f"Initialized Alpaca {'PAPER' if paper_trading else 'LIVE'} trading")

        elif broker == BrokerType.PAPER:
            # 내부 페이퍼 트레이딩 (기존 PaperTrader 사용)
            self.logger.info("Using internal paper trading")
            self.client = None

        else:
            raise NotImplementedError(f"Broker {broker.value} not yet implemented")

        # 주문 추적
        self.orders: Dict[str, Order] = {}
        self.execution_reports: List[ExecutionReport] = []

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("OrderExecutor")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # =========================================================================
    # Order Execution
    # =========================================================================

    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """
        주문 제출

        Args:
            ticker: 티커
            side: 매수/매도
            quantity: 수량
            order_type: 주문 유형
            limit_price: 지정가 (LIMIT 주문 시)
            stop_price: 스톱 가격 (STOP 주문 시)
            time_in_force: 주문 유효 시간

        Returns:
            Order 객체
        """
        start_time = time.time()

        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Order 객체 생성
        order = Order(
            id=order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            status=OrderStatus.PENDING,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            submitted_at=datetime.now().isoformat()
        )

        # 브로커에 주문 제출
        if self.broker == BrokerType.ALPACA:
            order = self._submit_alpaca_order(order)
        elif self.broker == BrokerType.PAPER:
            order = self._submit_paper_order(order)
        else:
            raise NotImplementedError(f"Broker {self.broker.value} not implemented")

        # 주문 추적
        self.orders[order_id] = order

        execution_time = (time.time() - start_time) * 1000  # ms

        self.logger.info(
            f"Order submitted: {ticker} {side.value.upper()} {quantity} "
            f"({order_type.value}) - Status: {order.status.value} "
            f"(Execution time: {execution_time:.0f}ms)"
        )

        return order

    def _submit_alpaca_order(self, order: Order) -> Order:
        """Alpaca API로 주문 제출"""
        try:
            # Alpaca 주문 파라미터
            alpaca_side = 'buy' if order.side == OrderSide.BUY else 'sell'
            alpaca_type = order.order_type.value
            alpaca_tif = order.time_in_force.value

            # 주문 제출
            alpaca_order = self.client.submit_order(
                symbol=order.ticker,
                qty=order.quantity,
                side=alpaca_side,
                type=alpaca_type,
                time_in_force=alpaca_tif,
                limit_price=order.limit_price,
                stop_price=order.stop_price
            )

            # 주문 상태 업데이트
            order.broker_order_id = alpaca_order.id
            order.status = self._map_alpaca_status(alpaca_order.status)

            # 체결 정보 업데이트 (즉시 체결된 경우)
            if alpaca_order.filled_at:
                order.filled_at = alpaca_order.filled_at
                order.filled_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
                order.filled_quantity = float(alpaca_order.filled_qty)

            return order

        except APIError as e:
            self.logger.error(f"Alpaca API error: {e}")
            order.status = OrderStatus.REJECTED
            return order
        except Exception as e:
            self.logger.error(f"Error submitting Alpaca order: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def _submit_paper_order(self, order: Order) -> Order:
        """내부 페이퍼 트레이딩 주문 (시뮬레이션)"""
        # 간단한 시뮬레이션: 즉시 체결
        try:
            import yfinance as yf

            ticker_data = yf.Ticker(order.ticker)
            current_price = ticker_data.history(period='1d')['Close'].iloc[-1]

            # 시장가 주문은 현재가로 즉시 체결
            if order.order_type == OrderType.MARKET:
                order.filled_price = current_price
                order.filled_quantity = order.quantity
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now().isoformat()
            else:
                # 지정가/스톱 주문은 제출만
                order.status = OrderStatus.SUBMITTED

            return order

        except Exception as e:
            self.logger.error(f"Error in paper order: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Alpaca 주문 상태를 내부 OrderStatus로 매핑"""
        mapping = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.CANCELED,
            'replaced': OrderStatus.SUBMITTED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.CANCELED,
            'calculated': OrderStatus.SUBMITTED,
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)

    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if self.broker == BrokerType.ALPACA:
            try:
                self.client.cancel_order(order.broker_order_id)
                order.status = OrderStatus.CANCELED
                self.logger.info(f"Order {order_id} canceled")
                return True
            except Exception as e:
                self.logger.error(f"Error canceling order: {e}")
                return False

        elif self.broker == BrokerType.PAPER:
            order.status = OrderStatus.CANCELED
            return True

        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """주문 상태 조회"""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        # Alpaca에서 최신 상태 가져오기
        if self.broker == BrokerType.ALPACA and order.broker_order_id:
            try:
                alpaca_order = self.client.get_order(order.broker_order_id)
                order.status = self._map_alpaca_status(alpaca_order.status)

                if alpaca_order.filled_at:
                    order.filled_at = alpaca_order.filled_at
                    order.filled_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
                    order.filled_quantity = float(alpaca_order.filled_qty)

            except Exception as e:
                self.logger.error(f"Error fetching order status: {e}")

        return order

    # =========================================================================
    # Account & Positions
    # =========================================================================

    def get_account_info(self) -> AccountInfo:
        """계좌 정보 조회"""
        if self.broker == BrokerType.ALPACA:
            try:
                account = self.client.get_account()

                return AccountInfo(
                    account_id=account.id,
                    broker=BrokerType.ALPACA,
                    cash=float(account.cash),
                    buying_power=float(account.buying_power),
                    portfolio_value=float(account.portfolio_value),
                    equity=float(account.equity),
                    positions_value=float(account.portfolio_value) - float(account.cash),
                    margin_used=float(account.initial_margin) if hasattr(account, 'initial_margin') else 0.0,
                    margin_available=float(account.buying_power),
                    last_updated=datetime.now().isoformat()
                )

            except Exception as e:
                self.logger.error(f"Error fetching account info: {e}")
                raise

        elif self.broker == BrokerType.PAPER:
            # 내부 페이퍼 계좌 (예시)
            return AccountInfo(
                account_id="paper_account",
                broker=BrokerType.PAPER,
                cash=100000.0,
                buying_power=100000.0,
                portfolio_value=100000.0,
                equity=100000.0,
                positions_value=0.0,
                last_updated=datetime.now().isoformat()
            )

        raise NotImplementedError()

    def get_positions(self) -> Dict[str, Position]:
        """포지션 조회"""
        positions = {}

        if self.broker == BrokerType.ALPACA:
            try:
                alpaca_positions = self.client.list_positions()

                for pos in alpaca_positions:
                    position = Position(
                        ticker=pos.symbol,
                        quantity=float(pos.qty),
                        avg_entry_price=float(pos.avg_entry_price),
                        current_price=float(pos.current_price),
                        market_value=float(pos.market_value),
                        unrealized_pnl=float(pos.unrealized_pl),
                        unrealized_pnl_pct=float(pos.unrealized_plpc) * 100
                    )
                    positions[pos.symbol] = position

            except Exception as e:
                self.logger.error(f"Error fetching positions: {e}")

        return positions

    def close_position(self, ticker: str) -> bool:
        """포지션 청산"""
        positions = self.get_positions()

        if ticker not in positions:
            self.logger.warning(f"No position for {ticker}")
            return False

        position = positions[ticker]
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY

        order = self.submit_order(
            ticker=ticker,
            side=side,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET
        )

        return order.status in [OrderStatus.FILLED, OrderStatus.SUBMITTED]

    # =========================================================================
    # Execution Quality
    # =========================================================================

    def generate_execution_report(self, order: Order, intended_price: float) -> ExecutionReport:
        """실행 품질 리포트 생성"""
        if not order.filled_price:
            raise ValueError("Order not filled yet")

        # 슬리피지 계산
        slippage_dollars = abs(order.filled_price - intended_price) * order.filled_quantity
        slippage_bps = (abs(order.filled_price - intended_price) / intended_price) * 10000

        total_cost = slippage_dollars + order.commission

        report = ExecutionReport(
            order_id=order.id,
            ticker=order.ticker,
            intended_price=intended_price,
            executed_price=order.filled_price,
            slippage_bps=slippage_bps,
            slippage_dollars=slippage_dollars,
            commission=order.commission,
            total_cost=total_cost,
            execution_time_ms=0.0,  # TODO: Track execution time
            timestamp=datetime.now().isoformat()
        )

        self.execution_reports.append(report)

        return report

    def get_execution_quality_summary(self) -> Dict[str, float]:
        """실행 품질 요약"""
        if not self.execution_reports:
            return {
                'avg_slippage_bps': 0.0,
                'total_slippage_dollars': 0.0,
                'total_commission': 0.0,
                'total_cost': 0.0,
                'executions': 0
            }

        total_slippage_bps = sum(r.slippage_bps for r in self.execution_reports)
        total_slippage_dollars = sum(r.slippage_dollars for r in self.execution_reports)
        total_commission = sum(r.commission for r in self.execution_reports)
        total_cost = sum(r.total_cost for r in self.execution_reports)

        return {
            'avg_slippage_bps': total_slippage_bps / len(self.execution_reports),
            'total_slippage_dollars': total_slippage_dollars,
            'total_commission': total_commission,
            'total_cost': total_cost,
            'executions': len(self.execution_reports)
        }


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing OrderExecutor (Paper Trading Mode)")
    print("=" * 70)

    # Note: Alpaca 키가 필요합니다
    # 무료 계정: https://alpaca.markets/
    # export ALPACA_API_KEY="..."
    # export ALPACA_SECRET_KEY="..."

    try:
        # Paper mode executor
        executor = OrderExecutor(
            broker=BrokerType.PAPER,
            paper_trading=True,
            verbose=True
        )

        print("\n[1] Get Account Info")
        account = executor.get_account_info()
        print(f"Account ID: {account.account_id}")
        print(f"Cash: ${account.cash:,.2f}")
        print(f"Buying Power: ${account.buying_power:,.2f}")
        print(f"Portfolio Value: ${account.portfolio_value:,.2f}")

        print("\n[2] Submit Market Order (BUY SPY 10 shares)")
        order = executor.submit_order(
            ticker="SPY",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        print(f"Order ID: {order.id}")
        print(f"Status: {order.status.value}")
        if order.filled_price:
            print(f"Filled Price: ${order.filled_price:.2f}")

        print("\n[3] Get Order Status")
        order_status = executor.get_order_status(order.id)
        if order_status:
            print(f"Order {order_status.id}: {order_status.status.value}")

        print("\n[4] Get Positions")
        positions = executor.get_positions()
        print(f"Total positions: {len(positions)}")

        print("\n[5] Execution Quality Summary")
        summary = executor.get_execution_quality_summary()
        print(f"Avg Slippage: {summary['avg_slippage_bps']:.2f} bps")
        print(f"Total Cost: ${summary['total_cost']:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: For Alpaca testing, set environment variables:")
        print("  export ALPACA_API_KEY='...'")
        print("  export ALPACA_SECRET_KEY='...'")

    print("\n" + "=" * 70)
