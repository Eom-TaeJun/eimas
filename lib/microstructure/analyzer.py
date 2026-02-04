#!/usr/bin/env python3
"""
Microstructure - Main Analyzer
============================================================

통합 미세구조 분석기

Economic Foundation:
    - Combines OFI, VPIN, Depth, Volume anomaly
    - Unified microstructure metrics

Class:
    - MicrostructureAnalyzer: Main analysis engine
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging

from .schemas import OrderBook, Trade, MicrostructureMetrics
from .config import RollingWindowConfig
from .ofi import OFICalculator, OFIEstimator
from .vpin import VPINCalculator
from .depth import DepthAnalyzer
from .volume_anomaly import VolumeAnomalyDetector

logger = logging.getLogger(__name__)


class MicrostructureAnalyzer:
    """
    통합 마이크로스트럭처 분석기

    OFI + VPIN + Depth + Anomaly Volume 분석 통합
    """

    def __init__(
        self,
        ofi_levels: int = 5,
        vpin_bucket_size: float = 1000.0,
        vpin_n_buckets: int = 50,
        volume_anomaly_window: int = 20
    ):
        self.ofi_calculator = OFICalculator(levels=ofi_levels)
        self.vpin_calculator = VPINCalculator(
            bucket_size=vpin_bucket_size,
            n_buckets=vpin_n_buckets
        )
        self.depth_analyzer = DepthAnalyzer()
        self.volume_detector = VolumeAnomalyDetector(window=volume_anomaly_window)

        # 히스토리
        self.metrics_history: deque = deque(maxlen=1000)

    def process_orderbook(self, orderbook: OrderBook) -> MicrostructureMetrics:
        """
        호가창 처리 및 지표 계산

        Parameters:
        -----------
        orderbook : OrderBook
            호가창 스냅샷

        Returns:
        --------
        MicrostructureMetrics
        """
        # OFI 계산
        ofi_l1, ofi_deep = self.ofi_calculator.calculate(orderbook)
        ofi_norm = self.ofi_calculator.get_normalized_ofi()

        # 깊이 계산
        bid_d1, ask_d1, bid_d5, ask_d5 = self.depth_analyzer.calculate_depth(orderbook)
        depth_ratio = self.depth_analyzer.calculate_depth_ratio(bid_d5, ask_d5)

        # 현재 VPIN
        vpin = self.vpin_calculator.calculate_vpin()

        # 신호 결정
        signal, strength = self._determine_signal(ofi_norm, depth_ratio, vpin)

        metrics = MicrostructureMetrics(
            symbol=orderbook.symbol,
            timestamp=orderbook.timestamp,
            ofi=ofi_l1,
            ofi_deep=ofi_deep,
            ofi_normalized=ofi_norm,
            vpin=vpin,
            vpin_bucket_count=len(self.vpin_calculator.buckets),
            bid_depth_1=bid_d1,
            ask_depth_1=ask_d1,
            bid_depth_5=bid_d5,
            ask_depth_5=ask_d5,
            depth_ratio=depth_ratio,
            spread_bps=orderbook.spread_bps,
            mid_price=orderbook.mid_price,
            signal=signal,
            signal_strength=strength
        )

        self.metrics_history.append(metrics)
        return metrics

    def process_trade(self, trade: Trade) -> Dict[str, Any]:
        """
        체결 데이터 처리 및 이상 거래량 감지

        Parameters:
        -----------
        trade : Trade
            체결 데이터

        Returns:
        --------
        Dict: VPIN 업데이트 결과 및 이상 거래량 여부
        """
        # VPIN 업데이트
        new_vpin = self.vpin_calculator.add_trade(trade)
        
        # 이상 거래량 감지 (체결량 기준)
        is_anomaly, z_score, mean_vol = self.volume_detector.add_volume(trade.quantity)
        
        return {
            'vpin': new_vpin,
            'is_volume_anomaly': is_anomaly,
            'volume_z_score': z_score
        }

    def _determine_signal(
        self,
        ofi_norm: float,
        depth_ratio: float,
        vpin: float
    ) -> Tuple[str, float]:
        """
        매매 신호 결정

        Returns:
        --------
        (signal, strength) : Tuple[str, float]
        """
        score = 0.0

        # OFI 점수 (-1 ~ 1)
        score += ofi_norm * 0.4

        # Depth ratio 점수
        if depth_ratio > 1.5:
            score += 0.3  # 강한 매수벽
        elif depth_ratio > 1.2:
            score += 0.15
        elif depth_ratio < 0.67:
            score -= 0.3  # 강한 매도벽
        elif depth_ratio < 0.83:
            score -= 0.15

        # VPIN 점수 (높으면 변동성 증가 예상)
        if vpin > 0.7:
            # 고 VPIN = 방향 불확실, 강도만 증가
            pass

        # 신호 결정
        if score > 0.3:
            return "bullish", min(abs(score), 1.0)
        elif score < -0.3:
            return "bearish", min(abs(score), 1.0)
        else:
            return "neutral", abs(score)

    def get_summary(self) -> Dict[str, Any]:
        """현재 상태 요약"""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]
        return {
            'symbol': latest.symbol,
            'timestamp': latest.timestamp.isoformat(),
            'ofi_normalized': latest.ofi_normalized,
            'vpin': latest.vpin,
            'depth_ratio': latest.depth_ratio,
            'spread_bps': latest.spread_bps,
            'signal': latest.signal,
            'signal_strength': latest.signal_strength,
            'metrics_count': len(self.metrics_history)
        }

    def reset(self):
        """상태 초기화"""
        self.ofi_calculator.reset()
        self.vpin_calculator.reset()
        self.metrics_history.clear()


