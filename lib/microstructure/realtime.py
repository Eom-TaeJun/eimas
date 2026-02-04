#!/usr/bin/env python3
"""
Microstructure - Real-time Analysis
============================================================

실시간 거래소 데이터 수집 및 분석

Classes:
    - ExchangeDataFetcher: ccxt 기반 데이터 수집
    - RealtimeMicrostructureAnalyzer: 실시간 분석기
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from .schemas import OrderBook, OrderBookLevel, Trade, MicrostructureMetrics
from .analyzer import MicrostructureAnalyzer
from .config import RollingWindowConfig

logger = logging.getLogger(__name__)

# Optional ccxt import
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None


class ExchangeDataFetcher:
    """
    거래소 데이터 수집기 (ccxt 기반)

    지원 거래소: Binance, Bybit, OKX 등
    """

    def __init__(self, exchange_id: str = 'binance'):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt is required. Install with: pip install ccxt")

        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        호가창 조회

        Parameters:
        -----------
        symbol : str
            심볼 (예: 'BTC/USDT')
        limit : int
            호가 레벨 수

        Returns:
        --------
        OrderBook
        """
        raw = self.exchange.fetch_order_book(symbol, limit)

        bids = [
            OrderBookLevel(price=b[0], quantity=b[1], side='bid')
            for b in raw['bids']
        ]
        asks = [
            OrderBookLevel(price=a[0], quantity=a[1], side='ask')
            for a in raw['asks']
        ]

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

    def fetch_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """
        최근 체결 조회

        Parameters:
        -----------
        symbol : str
            심볼
        limit : int
            조회 개수

        Returns:
        --------
        List[Trade]
        """
        raw = self.exchange.fetch_trades(symbol, limit=limit)

        return [
            Trade(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(t['timestamp'] / 1000),
                price=t['price'],
                quantity=t['amount'],
                side=t['side']
            )
            for t in raw
        ]

    def get_tickers(self, symbols: List[str] = None) -> Dict[str, Any]:
        """시세 조회"""
        if symbols:
            return {s: self.exchange.fetch_ticker(s) for s in symbols}
        return self.exchange.fetch_tickers()


# ============================================================================
# Real-time Analyzer
# ============================================================================

class RealtimeMicrostructureAnalyzer:
    """
    실시간 마이크로스트럭처 분석기

    주기적으로 호가창/체결 데이터를 가져와 분석
    """

    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        exchange_id: str = 'binance',
        interval_seconds: float = 1.0
    ):
        self.symbol = symbol
        self.interval = interval_seconds
        self.fetcher = ExchangeDataFetcher(exchange_id)
        self.analyzer = MicrostructureAnalyzer()

        self.running = False
        self.latest_metrics: Optional[MicrostructureMetrics] = None

    async def start(self, duration_seconds: int = 60):
        """
        분석 시작

        Parameters:
        -----------
        duration_seconds : int
            실행 시간 (초)
        """
        self.running = True
        start_time = datetime.now()
        iteration = 0

        print(f"Starting real-time analysis for {self.symbol}...")
        print(f"Duration: {duration_seconds}s, Interval: {self.interval}s")
        print("-" * 60)

        while self.running:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration_seconds:
                break

            try:
                # 호가창 조회
                orderbook = self.fetcher.fetch_orderbook(self.symbol, limit=10)
                metrics = self.analyzer.process_orderbook(orderbook)

                # 체결 조회 (처음 또는 10회마다)
                if iteration == 0 or iteration % 10 == 0:
                    trades = self.fetcher.fetch_trades(self.symbol, limit=50)
                    for trade in trades:
                        self.analyzer.process_trade(trade)

                self.latest_metrics = metrics

                # 출력
                if iteration % 5 == 0:  # 5초마다 출력
                    self._print_metrics(metrics)

                iteration += 1

            except Exception as e:
                print(f"Error: {e}")

            await asyncio.sleep(self.interval)

        self.running = False
        print("-" * 60)
        print("Analysis complete!")

    def _print_metrics(self, metrics: MicrostructureMetrics):
        """지표 출력"""
        signal_icon = {
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '⚪'
        }.get(metrics.signal, '⚪')

        print(
            f"[{metrics.timestamp.strftime('%H:%M:%S')}] "
            f"Price: ${metrics.mid_price:,.2f} | "
            f"OFI: {metrics.ofi_normalized:+.2f} | "
            f"VPIN: {metrics.vpin:.2f} | "
            f"Depth: {metrics.depth_ratio:.2f} | "
            f"{signal_icon} {metrics.signal.upper()}"
        )

    def stop(self):
        """분석 중지"""
        self.running = False


