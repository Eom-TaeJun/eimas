# lib/paper_trader.py — Shim (패키지 호환 레이어)
# 실제 구현은 lib/trading/paper_trader.py 로 이동됨.
from lib.trading.paper_trader import *  # noqa: F401, F403
from lib.trading.paper_trader import PaperTrader, Order, OrderType, OrderSide  # noqa: F401
