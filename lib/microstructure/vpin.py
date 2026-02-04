#!/usr/bin/env python3
"""
Microstructure - VPIN (Volume-Synchronized PIN)
============================================================

VPIN (Volume-Synchronized Probability of Informed Trading)

Economic Foundation:
    - Easley et al. (2012): "Flow Toxicity and Liquidity in a High Frequency World"
    - VPIN = |Buy Volume - Sell Volume| / Total Volume
    - Higher VPIN = higher informed trading risk

Class:
    - VPINCalculator: Volume-synchronized VPIN calculation
"""

from typing import List, Dict, Optional
from collections import deque
import numpy as np
from datetime import datetime
import logging

from .schemas import Trade, OrderBook

logger = logging.getLogger(__name__)


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading

    VPIN = Σ|V_buy - V_sell| / (n * V_bucket)

    참고: Easley et al. (2012) "Flow Toxicity and Liquidity in a High-frequency World"

    개선사항:
    - bucket_size 자동 조정 (adaptive)
    - 최소 버킷 수 5개로 낮춤 (빠른 초기화)
    - 시간 기반 버킷 완료 (30초 타임아웃)
    """

    def __init__(
        self,
        bucket_size: float = 50.0,    # 버킷당 거래량 (기존 1000 → 50)
        n_buckets: int = 20,           # VPIN 계산에 사용할 버킷 수 (기존 50 → 20)
        min_buckets_for_vpin: int = 5, # 최소 버킷 수 (기존 10 → 5)
        bucket_timeout: float = 30.0   # 버킷 타임아웃 (초)
    ):
        """
        Parameters:
        -----------
        bucket_size : float
            각 버킷의 목표 거래량
        n_buckets : int
            VPIN 계산에 사용할 버킷 수
        min_buckets_for_vpin : int
            VPIN 계산에 필요한 최소 버킷 수
        bucket_timeout : float
            버킷 강제 완료 타임아웃 (초)
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.min_buckets_for_vpin = min_buckets_for_vpin
        self.bucket_timeout = bucket_timeout

        # 현재 버킷
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        self.bucket_start_time = datetime.now()

        # 완료된 버킷들
        self.buckets: deque = deque(maxlen=n_buckets)

        # 통계 (bucket_size 자동 조정용)
        self.total_volume = 0.0
        self.trade_count = 0

    def add_trade(self, trade: Trade) -> Optional[float]:
        """
        거래 추가 및 VPIN 계산

        Parameters:
        -----------
        trade : Trade
            체결 데이터

        Returns:
        --------
        Optional[float] : 새 버킷 완료 시 VPIN, 아니면 None
        """
        qty = trade.quantity

        if trade.side == 'buy':
            self.current_bucket_buy += qty
        else:
            self.current_bucket_sell += qty

        self.current_bucket_volume += qty
        self.total_volume += qty
        self.trade_count += 1

        # 버킷 완료 체크 (볼륨 기반)
        if self.current_bucket_volume >= self.bucket_size:
            return self._complete_bucket()

        # 타임아웃 기반 버킷 완료 (최소 볼륨 있을 때만)
        elapsed = (datetime.now() - self.bucket_start_time).total_seconds()
        if elapsed >= self.bucket_timeout and self.current_bucket_volume > 0:
            return self._complete_bucket()

        return None

    def _complete_bucket(self) -> float:
        """버킷 완료 및 VPIN 반환"""
        # 버킷 크기가 0이면 기본값 사용
        effective_bucket_size = max(self.current_bucket_volume, self.bucket_size)

        imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
        # 정규화된 imbalance 저장 (버킷 크기로 나눔)
        normalized_imbalance = imbalance / effective_bucket_size if effective_bucket_size > 0 else 0
        self.buckets.append(normalized_imbalance)

        # 버킷 리셋
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        self.bucket_start_time = datetime.now()

        return self.calculate_vpin()

    def calculate_vpin(self) -> float:
        """
        현재 VPIN 계산

        Returns:
        --------
        float : VPIN (0 ~ 1)
        """
        if len(self.buckets) < self.min_buckets_for_vpin:
            # 버킷이 부족해도 현재 버킷으로 추정치 반환
            if self.current_bucket_volume > 0:
                current_imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
                return min(current_imbalance / self.current_bucket_volume, 1.0)
            return 0.0

        # VPIN = 평균 정규화된 imbalance
        vpin = sum(self.buckets) / len(self.buckets)
        return min(vpin, 1.0)  # 0~1 클리핑

    def reset(self):
        """상태 초기화"""
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        self.bucket_start_time = datetime.now()
        self.buckets.clear()
        self.total_volume = 0.0
        self.trade_count = 0


