#!/usr/bin/env python3
"""
Microstructure - Depth Analysis
============================================================

호가창 깊이 분석

Economic Foundation:
    - Kyle (1985): Market depth and lambda (price impact)
    - Depth imbalance = (Bid Depth - Ask Depth) / (Bid + Ask)
    - Positive imbalance → bullish pressure

Class:
    - DepthAnalyzer: Order book depth analysis
"""

from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import logging

from .schemas import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)


class DepthAnalyzer:
    """호가 깊이 분석기"""

    @staticmethod
    def calculate_depth(orderbook: OrderBook, levels: int = 5) -> Tuple[float, float, float, float]:
        """
        호가 깊이 계산

        Returns:
        --------
        (bid_depth_1, ask_depth_1, bid_depth_n, ask_depth_n)
        """
        bid_depth_1 = orderbook.bids[0].quantity if orderbook.bids else 0
        ask_depth_1 = orderbook.asks[0].quantity if orderbook.asks else 0

        bid_depth_n = sum(b.quantity for b in orderbook.bids[:levels])
        ask_depth_n = sum(a.quantity for a in orderbook.asks[:levels])

        return bid_depth_1, ask_depth_1, bid_depth_n, ask_depth_n

    @staticmethod
    def calculate_depth_ratio(bid_depth: float, ask_depth: float) -> float:
        """
        깊이 비율 계산

        Returns:
        --------
        float : bid_depth / ask_depth (>1 = 매수벽, <1 = 매도벽)
        """
        if ask_depth == 0:
            return float('inf') if bid_depth > 0 else 1.0
        return bid_depth / ask_depth

    @staticmethod
    def detect_wall(orderbook: OrderBook, threshold: float = 3.0) -> Optional[str]:
        """
        대형 호가벽 감지

        Parameters:
        -----------
        threshold : float
            평균 대비 배수 (기본 3배)

        Returns:
        --------
        Optional[str] : "bid_wall", "ask_wall", or None
        """
        if len(orderbook.bids) < 5 or len(orderbook.asks) < 5:
            return None

        bid_qtys = [b.quantity for b in orderbook.bids[:10]]
        ask_qtys = [a.quantity for a in orderbook.asks[:10]]

        avg_bid = np.mean(bid_qtys)
        avg_ask = np.mean(ask_qtys)

        # 상위 레벨에서 대형 주문 확인
        for i, b in enumerate(orderbook.bids[:5]):
            if b.quantity > avg_bid * threshold:
                return "bid_wall"

        for i, a in enumerate(orderbook.asks[:5]):
            if a.quantity > avg_ask * threshold:
                return "ask_wall"

        return None


