#!/usr/bin/env python3
"""
Microstructure - OFI (Order Flow Imbalance)
============================================================

Order Flow Imbalance 계산

Economic Foundation:
    - Cont et al. (2014): "The Price Impact of Order Book Events"
    - OFI = Δbid_volume - Δask_volume
    - Predicts short-term price movements

Classes:
    - OFICalculator: High-frequency OFI from order book
    - OFIEstimator: OHLC fallback (no order book data)
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np
from datetime import datetime
import logging

from .schemas import OrderBook, OrderBookLevel, Trade

logger = logging.getLogger(__name__)


class OFICalculator:
    """
    Order Flow Imbalance 계산기

    OFI = Σ (bid_qty_change - ask_qty_change) at each level

    참고: Cont et al. (2014) "The Price Impact of Order Book Events"
    """

    def __init__(self, levels: int = 5):
        """
        Parameters:
        -----------
        levels : int
            계산에 사용할 호가 레벨 수 (기본 5)
        """
        self.levels = levels
        self.prev_orderbook: Optional[OrderBook] = None
        self.ofi_history: deque = deque(maxlen=100)

    def calculate(self, orderbook: OrderBook) -> Tuple[float, float]:
        """
        OFI 계산

        Parameters:
        -----------
        orderbook : OrderBook
            현재 호가창

        Returns:
        --------
        (ofi_level1, ofi_deep) : Tuple[float, float]
        """
        if self.prev_orderbook is None:
            self.prev_orderbook = orderbook
            return 0.0, 0.0

        ofi_level1 = 0.0
        ofi_deep = 0.0
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # 레벨별 가중치

        # Level별 OFI 계산
        for i in range(min(self.levels, len(orderbook.bids), len(orderbook.asks))):
            # 현재 호가
            curr_bid_qty = orderbook.bids[i].quantity if i < len(orderbook.bids) else 0
            curr_ask_qty = orderbook.asks[i].quantity if i < len(orderbook.asks) else 0

            # 이전 호가
            prev_bid_qty = self.prev_orderbook.bids[i].quantity if i < len(self.prev_orderbook.bids) else 0
            prev_ask_qty = self.prev_orderbook.asks[i].quantity if i < len(self.prev_orderbook.asks) else 0

            # OFI = 매수잔량 변화 - 매도잔량 변화
            level_ofi = (curr_bid_qty - prev_bid_qty) - (curr_ask_qty - prev_ask_qty)

            if i == 0:
                ofi_level1 = level_ofi

            # 가중 평균
            weight = weights[i] if i < len(weights) else 0.1
            ofi_deep += level_ofi * weight

        # 정규화
        total_weight = sum(weights[:min(self.levels, len(orderbook.bids))])
        if total_weight > 0:
            ofi_deep /= total_weight

        self.prev_orderbook = orderbook
        self.ofi_history.append(ofi_deep)

        return ofi_level1, ofi_deep

    def get_normalized_ofi(self) -> float:
        """
        정규화된 OFI (-1 ~ 1)

        최근 100개 OFI의 z-score 기반
        """
        if len(self.ofi_history) < 10:
            return 0.0

        arr = np.array(self.ofi_history)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        z = (self.ofi_history[-1] - mean) / std
        # tanh로 -1 ~ 1 범위로 압축
        return float(np.tanh(z / 2))

    def reset(self):
        """상태 초기화"""
        self.prev_orderbook = None
        self.ofi_history.clear()




class OFIEstimator:
    """
    OFI 근사 추정기 (Tick 데이터 부재 시 OHLC 활용)
    
    Logic:
    - (Close - Open) / (High - Low) 를 통해 매수/매도 압력 강도 추정
    - 거래량을 곱하여 Flow Imbalance 근사
    """
    
    @staticmethod
    def estimate_from_ohlc(
        open_p: float, 
        high_p: float, 
        low_p: float, 
        close_p: float, 
        volume: float
    ) -> float:
        """
        OHLC 기반 OFI 근사치 계산
        """
        price_range = high_p - low_p
        
        if price_range == 0:
            return 0.0
            
        # CLV (Close Location Value) or Money Flow Multiplier
        # (Close - Low) - (High - Close) / (High - Low)
        # = (2 * Close - High - Low) / (High - Low)
        # 범위: -1 ~ 1
        pressure = (2 * close_p - high_p - low_p) / price_range
        
        # 거래량을 곱해 Flow 양 추정
        estimated_ofi = pressure * volume
        
        return estimated_ofi

