"""
EIMAS Trading — 모의주문 및 자동 실행 패키지
=============================================
Paper trading and automated execution modules.

Modules:
    - paper_trader: PaperTrader, Order, Position, Trade
    - auto_paper_execution: AutoPaperExecutionConfig, execution logic
"""

from .paper_trader import PaperTrader, Order, Position, Trade, PortfolioSummary
from .paper_trader import OrderType, OrderSide, OrderStatus
# auto_paper_execution은 broker_execution → paper_trader 순환 의존성으로 인해
# __init__에서 직접 import 하지 않음. lib.trading.auto_paper_execution 으로 직접 접근.

__all__ = [
    'PaperTrader',
    'Order',
    'Position',
    'Trade',
    'PortfolioSummary',
    'OrderType',
    'OrderSide',
    'OrderStatus',
]
