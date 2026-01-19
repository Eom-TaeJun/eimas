"""
Market Microstructure Module
============================
OFI (Order Flow Imbalance) 및 VPIN (Volume-Synchronized PIT) 계산

핵심 지표:
1. OFI (Order Flow Imbalance)
   - 호가창 불균형 측정
   - 양수 = 매수 압력, 음수 = 매도 압력

2. OFI_deep (Multi-Level OFI)
   - Level 1-5 호가의 가중평균 OFI
   - 더 깊은 유동성 구조 파악

3. VPIN (Volume-Synchronized Probability of Informed Trading)
   - 거래량 동기화된 정보거래 확률
   - 0~1 범위, 높을수록 정보거래 활발

4. Depth Ratio
   - 호가 깊이 비율 (bid_depth / ask_depth)
   - >1 = 매수벽, <1 = 매도벽

데이터 소스:
- Binance WebSocket (암호화폐)
- ccxt (다중 거래소 지원)

Author: EIMAS Team
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import asyncio
import json

# ccxt for exchange data
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# binance for websocket
try:
    from binance.client import Client
    from binance.streams import BinanceSocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False


# ============================================================================
# Rolling Window Configuration
# ============================================================================

class RollingWindowConfig:
    """
    롤링 윈도우 표준 설정

    경제학적 근거:
    - min_periods: 통계적 유의성을 위한 최소 데이터 포인트
    - fill_method: NaN 처리 전략 (시계열 연속성 vs 명시적 결측)
    """

    DEFAULTS = {
        'amihud_lambda': {
            'window': 252,       # 1년 영업일
            'min_periods': 20,   # 최소 1개월
            'fill_method': None  # NaN 유지 (불확실성 명시)
        },
        'vpin': {
            'window': 50,        # VPIN 버킷 수
            'min_periods': 5,    # 최소 5 버킷
            'fill_method': 'neutral'  # 0.5 (균형 가정)
        },
        'roll_spread': {
            'window': 20,        # Roll (1984) 표준
            'min_periods': 10,   # 최소 절반
            'fill_method': None  # NaN 유지
        },
        'volatility': {
            'window': 21,        # 1개월 영업일
            'min_periods': 10,   # 최소 절반
            'fill_method': None  # NaN 유지
        }
    }

    @classmethod
    def get(cls, indicator: str, param: str) -> Any:
        """설정값 조회"""
        if indicator in cls.DEFAULTS and param in cls.DEFAULTS[indicator]:
            return cls.DEFAULTS[indicator][param]
        return None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OrderBookLevel:
    """호가 레벨"""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'


@dataclass
class OrderBook:
    """호가창 스냅샷"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # 매수호가 (높은 가격순)
    asks: List[OrderBookLevel]  # 매도호가 (낮은 가격순)

    @property
    def mid_price(self) -> float:
        """중간가"""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        """스프레드"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_bps(self) -> float:
        """스프레드 (bps)"""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0


@dataclass
class Trade:
    """체결 데이터"""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str  # 'buy' or 'sell' (taker side)


@dataclass
class MicrostructureMetrics:
    """마이크로스트럭처 지표"""
    symbol: str
    timestamp: datetime

    # OFI 관련
    ofi: float = 0.0              # Level 1 OFI
    ofi_deep: float = 0.0         # Level 1-5 가중평균 OFI
    ofi_normalized: float = 0.0   # 정규화된 OFI (-1 ~ 1)

    # VPIN 관련
    vpin: float = 0.0             # VPIN (0 ~ 1)
    vpin_bucket_count: int = 0    # 계산에 사용된 버킷 수

    # 호가 깊이
    bid_depth_1: float = 0.0      # Level 1 매수 수량
    ask_depth_1: float = 0.0      # Level 1 매도 수량
    bid_depth_5: float = 0.0      # Level 1-5 매수 총량
    ask_depth_5: float = 0.0      # Level 1-5 매도 총량
    depth_ratio: float = 1.0      # bid_depth / ask_depth

    # 스프레드
    spread_bps: float = 0.0
    mid_price: float = 0.0

    # 거래량
    buy_volume: float = 0.0       # 매수 체결량
    sell_volume: float = 0.0      # 매도 체결량
    volume_imbalance: float = 0.0 # (buy - sell) / total

    # 신호
    signal: str = "neutral"       # bullish, bearish, neutral
    signal_strength: float = 0.0  # 0 ~ 1

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'ofi': self.ofi,
            'ofi_deep': self.ofi_deep,
            'ofi_normalized': self.ofi_normalized,
            'vpin': self.vpin,
            'bid_depth_5': self.bid_depth_5,
            'ask_depth_5': self.ask_depth_5,
            'depth_ratio': self.depth_ratio,
            'spread_bps': self.spread_bps,
            'mid_price': self.mid_price,
            'volume_imbalance': self.volume_imbalance,
            'signal': self.signal,
            'signal_strength': self.signal_strength
        }


# ============================================================================
# OFI Calculator
# ============================================================================

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


# ============================================================================
# VPIN Calculator
# ============================================================================

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


# ============================================================================
# Depth Analyzer
# ============================================================================

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


# ============================================================================
# Volume Anomaly Detector
# ============================================================================

class VolumeAnomalyDetector:
    """
    이상 거래량(Anomaly Volume) 감지기
    
    Rule: 현재 거래량이 20일(또는 20주기) 이동평균 대비 
    3표준편차(3-sigma) 이상 급증 시 True 반환
    """
    
    def __init__(self, window: int = 20, threshold_sigma: float = 3.0):
        self.window = window
        self.threshold_sigma = threshold_sigma
        self.volume_history: deque = deque(maxlen=window + 1)
        
    def add_volume(self, volume: float) -> Tuple[bool, float, float]:
        """
        거래량 추가 및 이상 감지
        
        Returns:
            (is_anomaly, z_score, mean_volume)
        """
        self.volume_history.append(volume)
        
        if len(self.volume_history) < self.window:
            return False, 0.0, 0.0
            
        # 최근 volume을 제외한 이전 window개 데이터로 통계 계산
        recent_history = list(self.volume_history)[:-1]
        mean = np.mean(recent_history)
        std = np.std(recent_history)
        
        if std == 0:
            return False, 0.0, mean
            
        z_score = (volume - mean) / std
        is_anomaly = z_score > self.threshold_sigma
        
        return is_anomaly, z_score, mean


# ============================================================================
# OFI Estimator (OHLC Fallback)
# ============================================================================

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


# ============================================================================
# Unified Microstructure Analyzer
# ============================================================================

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


# ============================================================================
# Exchange Data Fetcher (ccxt)
# ============================================================================

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


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_analysis(symbol: str = 'BTC/USDT', samples: int = 10) -> Dict[str, Any]:
    """
    빠른 분석 실행

    Parameters:
    -----------
    symbol : str
        분석할 심볼
    samples : int
        샘플 수

    Returns:
    --------
    Dict with analysis results
    """
    fetcher = ExchangeDataFetcher('binance')
    analyzer = MicrostructureAnalyzer()

    results = []

    for i in range(samples):
        try:
            # 호가창
            ob = fetcher.fetch_orderbook(symbol, limit=10)
            metrics = analyzer.process_orderbook(ob)
            results.append(metrics)

            # 체결
            if i == 0:
                trades = fetcher.fetch_trades(symbol, limit=100)
                for t in trades:
                    analyzer.process_trade(t)

            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"Sample {i} error: {e}")

    if not results:
        return {'error': 'No data collected'}

    # 요약
    final = results[-1]
    ofi_values = [r.ofi_normalized for r in results]
    depth_values = [r.depth_ratio for r in results]

    return {
        'symbol': symbol,
        'samples': len(results),
        'mid_price': final.mid_price,
        'spread_bps': final.spread_bps,
        'ofi_current': final.ofi_normalized,
        'ofi_mean': np.mean(ofi_values),
        'ofi_std': np.std(ofi_values),
        'vpin': final.vpin,
        'depth_ratio': final.depth_ratio,
        'depth_mean': np.mean(depth_values),
        'signal': final.signal,
        'signal_strength': final.signal_strength
    }


# ============================================================================
# AMFL Chapter 19: Market Microstructure Metrics (Daily Data)
# ============================================================================
#
# 일별 데이터 기반 미세구조 지표:
# 1. Amihud Lambda (비유동성 측정)
# 2. Roll Spread (유효 스프레드 추정)
# 3. VPIN Approximation (일별 OHLC 기반 근사)
#
# Reference:
# - Amihud, Y. (2002). Illiquidity and stock returns
# - Roll, R. (1984). A simple implicit measure of the effective bid-ask spread
# - Easley, López de Prado, O'Hara (2012). VPIN
# ============================================================================

@dataclass
class AmihudLambdaResult:
    """Amihud Lambda 결과"""
    lambda_value: float          # Amihud Lambda (비유동성)
    lambda_series: Optional[pd.Series] = None  # 일별 Lambda 시계열
    avg_daily_volume: float = 0.0     # 평균 일일 거래대금
    interpretation: str = ""          # 해석

    def to_dict(self) -> Dict:
        return {
            'lambda': self.lambda_value,
            'avg_daily_volume': self.avg_daily_volume,
            'interpretation': self.interpretation
        }


@dataclass
class RollSpreadResult:
    """Roll Spread 결과"""
    spread: float                # 추정 유효 스프레드 (%)
    covariance: float            # 가격 변화 공분산
    is_valid: bool               # 공분산이 음수인지 (유효성)
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'spread_pct': self.spread,
            'covariance': self.covariance,
            'is_valid': self.is_valid,
            'interpretation': self.interpretation
        }


@dataclass
class VPINApproxResult:
    """VPIN 근사 결과 (일별 OHLC 기반)"""
    vpin: float                  # VPIN 값 (0-1)
    buy_volume_ratio: float      # 매수 거래량 비율
    sell_volume_ratio: float     # 매도 거래량 비율
    toxicity_level: str          # LOW/MEDIUM/HIGH
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'vpin': self.vpin,
            'buy_ratio': self.buy_volume_ratio,
            'sell_ratio': self.sell_volume_ratio,
            'toxicity': self.toxicity_level,
            'interpretation': self.interpretation
        }


@dataclass
class DailyMicrostructureResult:
    """일별 데이터 기반 미세구조 분석 결과"""
    ticker: str
    timestamp: str
    amihud: AmihudLambdaResult
    roll_spread: RollSpreadResult
    vpin_approx: VPINApproxResult
    overall_liquidity_score: float  # 0-100 (높을수록 유동성 좋음)
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'amihud': self.amihud.to_dict(),
            'roll_spread': self.roll_spread.to_dict(),
            'vpin_approx': self.vpin_approx.to_dict(),
            'liquidity_score': self.overall_liquidity_score,
            'risk_flags': self.risk_flags
        }


class DailyMicrostructureAnalyzer:
    """
    일별 데이터 기반 미세구조 분석기 (AMFL Chapter 19)

    고빈도 데이터가 없을 때 일별 OHLCV로 시장 미세구조 근사
    Amihud Lambda, Roll Spread, VPIN Approximation 계산
    """

    def __init__(
        self,
        lookback_days: int = 252,
        vpin_window: int = 50,
        amihud_scale: float = 1e6
    ):
        """
        Args:
            lookback_days: 분석 기간 (일)
            vpin_window: VPIN 계산용 윈도우 크기
            amihud_scale: Amihud Lambda 스케일링 (기본 1e6)
        """
        self.lookback_days = lookback_days
        self.vpin_window = vpin_window
        self.amihud_scale = amihud_scale

    def calculate_amihud_lambda(
        self,
        returns: pd.Series,
        volume: pd.Series,
        price: pd.Series,
        min_periods: Optional[int] = None,
        winsorize: bool = True
    ) -> AmihudLambdaResult:
        """
        Amihud Lambda 계산 (AMFL Ch.19 비유동성 지표)

        Lambda = mean(|r_t| / DollarVolume_t) * scale

        높은 Lambda = 낮은 유동성 (가격 충격이 큼)
        낮은 Lambda = 높은 유동성 (가격 충격이 작음)

        NaN 처리 전략:
        - 거래량 0인 날: 유동성 측정 불가 → 제외
        - 극단적 Lambda: Winsorize로 영향 완화 (1-99 percentile)

        Args:
            returns: 일별 수익률
            volume: 일별 거래량
            price: 일별 가격 (종가)
            min_periods: 최소 데이터 포인트 (기본값: RollingWindowConfig 사용)
            winsorize: 극단치 Winsorize 여부 (기본값: True)

        Returns:
            AmihudLambdaResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('amihud_lambda', 'min_periods') or 20

        # Dollar Volume 계산
        dollar_volume = volume * price

        # 유효한 데이터만 필터링 (0 거래량 및 NaN 제외)
        valid_mask = (
            (dollar_volume > 0) &
            (returns.notna()) &
            (np.isfinite(returns))
        )
        abs_returns = returns.abs()[valid_mask]
        dv = dollar_volume[valid_mask]

        if len(abs_returns) < min_periods:
            return AmihudLambdaResult(
                lambda_value=np.nan,
                lambda_series=None,
                avg_daily_volume=0,
                interpretation=f"Insufficient data (need {min_periods}, got {len(abs_returns)})"
            )

        # 일별 Lambda 계산
        daily_lambda = (abs_returns / dv) * self.amihud_scale

        # Winsorize: 극단치 영향 완화 (1-99 percentile)
        if winsorize and len(daily_lambda.dropna()) > 10:
            lower, upper = np.percentile(daily_lambda.dropna(), [1, 99])
            daily_lambda = daily_lambda.clip(lower, upper)

        # 평균 Lambda (극단치 제거를 위해 중앙값 사용)
        lambda_value = float(daily_lambda.median())
        avg_volume = float(dv.mean())

        # 해석
        if lambda_value < 0.01:
            interpretation = "Very High Liquidity (대형주 수준)"
        elif lambda_value < 0.1:
            interpretation = "High Liquidity (유동성 양호)"
        elif lambda_value < 1.0:
            interpretation = "Moderate Liquidity (평균 수준)"
        elif lambda_value < 10.0:
            interpretation = "Low Liquidity (유동성 부족)"
        else:
            interpretation = "Very Low Liquidity (거래 주의)"

        return AmihudLambdaResult(
            lambda_value=lambda_value,
            lambda_series=daily_lambda,
            avg_daily_volume=avg_volume,
            interpretation=interpretation
        )

    def calculate_roll_spread(
        self,
        price: pd.Series,
        min_periods: Optional[int] = None
    ) -> RollSpreadResult:
        """
        Roll Spread 계산 (AMFL Ch.19 유효 스프레드 추정)

        Roll (1984) Model:
        - 가격 변화의 시리얼 공분산을 이용
        - Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
        - 공분산이 양수면 스프레드 = 0 (모델 가정 위배)

        NaN 처리 전략:
        - diff()로 생성된 첫 NaN은 dropna()로 제거
        - 연속적인 NaN 비율이 높으면 경고 로깅

        Args:
            price: 일별 가격 시계열
            min_periods: 최소 데이터 포인트 (기본값: RollingWindowConfig 사용)

        Returns:
            RollSpreadResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('roll_spread', 'min_periods') or 10

        # 가격 변화 계산
        delta_price = price.diff()

        # NaN 비율 체크 (데이터 품질 경고)
        nan_ratio = delta_price.isna().sum() / len(delta_price) if len(delta_price) > 0 else 1.0
        if nan_ratio > 0.1:
            import logging
            logging.getLogger('eimas.microstructure').warning(
                f"High NaN ratio ({nan_ratio:.1%}) in price series for Roll Spread"
            )

        delta_price = delta_price.dropna()

        if len(delta_price) < min_periods:
            return RollSpreadResult(
                spread=np.nan,
                covariance=np.nan,
                is_valid=False,
                interpretation=f"Insufficient data (need {min_periods}, got {len(delta_price)})"
            )

        # 시리얼 공분산 계산
        delta_price_lag = delta_price.shift(1).dropna()
        delta_price_curr = delta_price.iloc[1:]

        # 인덱스 맞추기
        common_idx = delta_price_curr.index.intersection(delta_price_lag.index)
        covariance = float(np.cov(
            delta_price_curr.loc[common_idx],
            delta_price_lag.loc[common_idx]
        )[0, 1])

        # Roll Spread 계산
        # 공분산이 양수면 0으로 처리 (모델 가정: 공분산은 음수여야 함)
        if covariance >= 0:
            spread = 0.0
            is_valid = False
            interpretation = "Positive covariance (모델 가정 위배, spread=0)"
        else:
            # Spread = 2 * sqrt(-Cov)
            spread_raw = 2 * np.sqrt(-covariance)
            is_valid = True

            # 스프레드를 퍼센트로 변환
            avg_price = price.mean()
            spread = float((spread_raw / avg_price) * 100)

            if spread < 0.05:
                interpretation = f"Very Tight Spread ({spread:.3f}%)"
            elif spread < 0.2:
                interpretation = f"Normal Spread ({spread:.3f}%)"
            elif spread < 0.5:
                interpretation = f"Wide Spread ({spread:.3f}%)"
            else:
                interpretation = f"Very Wide Spread ({spread:.3f}%) - 유동성 주의"

        return RollSpreadResult(
            spread=spread,
            covariance=covariance,
            is_valid=is_valid,
            interpretation=interpretation
        )

    def calculate_vpin_approximation(
        self,
        open_price: pd.Series,
        high_price: pd.Series,
        low_price: pd.Series,
        close_price: pd.Series,
        volume: pd.Series,
        min_periods: Optional[int] = None,
        fill_method: str = 'neutral'
    ) -> VPINApproxResult:
        """
        VPIN 근사치 계산 (일별 OHLC 기반, AMFL Ch.19)

        고빈도 데이터가 없는 경우 일별 데이터로 VPIN을 근사

        방법 (Bulk Volume Classification):
        1. 일중 가격 움직임으로 매수/매도 압력 추정
        2. Buy Volume = Volume * (Close - Low) / (High - Low)
        3. Sell Volume = Volume * (High - Close) / (High - Low)
        4. VPIN = |Buy - Sell| / Total Volume (rolling window)

        NaN 처리 전략 (fill_method):
        - 'neutral': 0.5 (매수/매도 균형 가정)
          경제학적 근거: 가격 변동 없음 = 정보 비대칭 없음
        - 'ffill': 이전 값 사용 (시계열 연속성 유지)
        - 'none': NaN 유지 (후속 계산에서 제외)

        Args:
            open_price, high_price, low_price, close_price: OHLC 가격
            volume: 거래량
            min_periods: 롤링 윈도우 최소 데이터 포인트 (기본값: RollingWindowConfig)
            fill_method: NaN 처리 방법 ('neutral', 'ffill', 'none')

        Returns:
            VPINApproxResult
        """
        # 설정값 조회
        if min_periods is None:
            min_periods = RollingWindowConfig.get('vpin', 'min_periods') or 5

        # 가격 범위 계산
        price_range = high_price - low_price

        # 0 범위 처리 (가격 변동이 없는 날)
        price_range = price_range.replace(0, np.nan)

        # BVC (Bulk Volume Classification)
        # 종가가 고가에 가까우면 매수 우세, 저가에 가까우면 매도 우세
        buy_ratio = (close_price - low_price) / price_range
        sell_ratio = (high_price - close_price) / price_range

        # NaN 처리 (configurable)
        if fill_method == 'neutral':
            # 경제학적 근거: 가격 변동 없음 = 정보 비대칭 없음 → 50:50
            buy_ratio = buy_ratio.fillna(0.5)
            sell_ratio = sell_ratio.fillna(0.5)
        elif fill_method == 'ffill':
            # 시계열 연속성 유지 → 이전 값 사용, 첫 값 없으면 0.5
            buy_ratio = buy_ratio.ffill().fillna(0.5)
            sell_ratio = sell_ratio.ffill().fillna(0.5)
        # else: 'none' - NaN 유지

        # 매수/매도 거래량 추정
        buy_volume = volume * buy_ratio
        sell_volume = volume * sell_ratio

        # VPIN 계산 (rolling window with min_periods)
        window = min(self.vpin_window, len(volume))

        rolling_buy = buy_volume.rolling(window=window, min_periods=min_periods).sum()
        rolling_sell = sell_volume.rolling(window=window, min_periods=min_periods).sum()
        rolling_total = volume.rolling(window=window, min_periods=min_periods).sum()

        # VPIN = |V_buy - V_sell| / V_total
        vpin_series = (rolling_buy - rolling_sell).abs() / rolling_total

        # 최신 VPIN 값
        current_vpin = float(vpin_series.iloc[-1]) if not vpin_series.empty else np.nan

        if pd.isna(current_vpin):
            return VPINApproxResult(
                vpin=np.nan,
                buy_volume_ratio=0.5,
                sell_volume_ratio=0.5,
                toxicity_level="UNKNOWN",
                interpretation="Insufficient data"
            )

        # 최근 매수/매도 비율
        recent_buy_ratio = float(buy_ratio.iloc[-window:].mean())
        recent_sell_ratio = float(sell_ratio.iloc[-window:].mean())

        # Toxicity Level 결정
        # VPIN이 높을수록 정보 비대칭성(toxicity)이 높음
        if current_vpin < 0.2:
            toxicity = "LOW"
            interpretation = "Low order flow toxicity (정보 비대칭 낮음)"
        elif current_vpin < 0.4:
            toxicity = "MEDIUM"
            interpretation = "Moderate toxicity (주의 관찰 필요)"
        elif current_vpin < 0.6:
            toxicity = "HIGH"
            interpretation = "High toxicity (정보 비대칭 높음, 급변동 가능)"
        else:
            toxicity = "EXTREME"
            interpretation = "Extreme toxicity (Flash crash 위험)"

        return VPINApproxResult(
            vpin=current_vpin,
            buy_volume_ratio=recent_buy_ratio,
            sell_volume_ratio=recent_sell_ratio,
            toxicity_level=toxicity,
            interpretation=interpretation
        )

    def analyze(
        self,
        ticker: str,
        data: pd.DataFrame
    ) -> DailyMicrostructureResult:
        """
        일별 데이터 기반 통합 미세구조 분석

        Args:
            ticker: 티커 심볼
            data: OHLCV DataFrame (columns: Open, High, Low, Close, Volume)

        Returns:
            DailyMicrostructureResult
        """
        # 데이터 검증
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # 수익률 계산
        returns = data['Close'].pct_change()

        # 1. Amihud Lambda
        amihud = self.calculate_amihud_lambda(
            returns=returns,
            volume=data['Volume'],
            price=data['Close']
        )

        # 2. Roll Spread
        roll = self.calculate_roll_spread(price=data['Close'])

        # 3. VPIN Approximation
        vpin = self.calculate_vpin_approximation(
            open_price=data['Open'],
            high_price=data['High'],
            low_price=data['Low'],
            close_price=data['Close'],
            volume=data['Volume']
        )

        # 종합 유동성 점수 계산 (0-100)
        liquidity_score = self._calculate_liquidity_score(amihud, roll, vpin)

        # 리스크 플래그
        risk_flags = self._identify_risk_flags(amihud, roll, vpin)

        return DailyMicrostructureResult(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            amihud=amihud,
            roll_spread=roll,
            vpin_approx=vpin,
            overall_liquidity_score=liquidity_score,
            risk_flags=risk_flags
        )

    def _calculate_liquidity_score(
        self,
        amihud: AmihudLambdaResult,
        roll: RollSpreadResult,
        vpin: VPINApproxResult
    ) -> float:
        """유동성 점수 계산 (0-100, 높을수록 좋음)"""
        scores = []

        # Amihud 점수 (낮을수록 좋음)
        if not np.isnan(amihud.lambda_value):
            if amihud.lambda_value < 0.01:
                scores.append(100)
            elif amihud.lambda_value < 0.1:
                scores.append(80)
            elif amihud.lambda_value < 1.0:
                scores.append(60)
            elif amihud.lambda_value < 10.0:
                scores.append(40)
            else:
                scores.append(20)

        # Roll Spread 점수 (낮을수록 좋음)
        if not np.isnan(roll.spread):
            if roll.spread < 0.05:
                scores.append(100)
            elif roll.spread < 0.2:
                scores.append(80)
            elif roll.spread < 0.5:
                scores.append(60)
            else:
                scores.append(40)

        # VPIN 점수 (낮을수록 좋음)
        if not np.isnan(vpin.vpin):
            scores.append(max(0, 100 - vpin.vpin * 100))

        return float(np.mean(scores)) if scores else 50.0

    def _identify_risk_flags(
        self,
        amihud: AmihudLambdaResult,
        roll: RollSpreadResult,
        vpin: VPINApproxResult
    ) -> List[str]:
        """리스크 플래그 식별"""
        flags = []

        # Amihud 경고
        if not np.isnan(amihud.lambda_value) and amihud.lambda_value > 1.0:
            flags.append("LOW_LIQUIDITY")

        # Roll Spread 경고
        if not np.isnan(roll.spread) and roll.spread > 0.5:
            flags.append("WIDE_SPREAD")

        # VPIN 경고
        if not np.isnan(vpin.vpin):
            if vpin.vpin > 0.6:
                flags.append("EXTREME_TOXICITY")
            elif vpin.vpin > 0.4:
                flags.append("HIGH_TOXICITY")

        return flags

    def analyze_multiple(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, DailyMicrostructureResult]:
        """여러 티커 분석"""
        results = {}

        for ticker, data in market_data.items():
            try:
                results[ticker] = self.analyze(ticker, data)
            except Exception as e:
                print(f"Warning: Failed to analyze {ticker}: {e}")
                continue

        return results

    def get_summary(
        self,
        results: Dict[str, DailyMicrostructureResult]
    ) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not results:
            return {'error': 'No results'}

        # 유동성 점수 기준 정렬
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].overall_liquidity_score,
            reverse=True
        )

        # 최고/최저 유동성
        most_liquid = sorted_results[0] if sorted_results else None
        least_liquid = sorted_results[-1] if sorted_results else None

        # 위험 티커
        risky_tickers = [
            ticker for ticker, result in results.items()
            if result.risk_flags
        ]

        # 평균 VPIN
        vpins = [r.vpin_approx.vpin for r in results.values()
                 if not np.isnan(r.vpin_approx.vpin)]
        avg_vpin = float(np.mean(vpins)) if vpins else np.nan

        return {
            'total_analyzed': len(results),
            'avg_liquidity_score': float(np.mean([r.overall_liquidity_score for r in results.values()])),
            'avg_vpin': avg_vpin,
            'most_liquid': most_liquid[0] if most_liquid else None,
            'least_liquid': least_liquid[0] if least_liquid else None,
            'risky_tickers': risky_tickers,
            'risk_count': len(risky_tickers)
        }


# ============================================================================
# Convenience Functions for Daily Microstructure
# ============================================================================

def calculate_amihud(returns: pd.Series, volume: pd.Series, price: pd.Series) -> float:
    """Amihud Lambda 간편 계산"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_amihud_lambda(returns, volume, price)
    return result.lambda_value


def calculate_roll_spread_daily(price: pd.Series) -> float:
    """Roll Spread 간편 계산 (일별 데이터)"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_roll_spread(price)
    return result.spread


def calculate_vpin_daily(ohlcv: pd.DataFrame) -> float:
    """VPIN 간편 계산 (일별 OHLCV)"""
    analyzer = DailyMicrostructureAnalyzer()
    result = analyzer.calculate_vpin_approximation(
        ohlcv['Open'], ohlcv['High'], ohlcv['Low'], ohlcv['Close'], ohlcv['Volume']
    )
    return result.vpin


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Market Microstructure Module Test")
    print("=" * 60)

    # 1. 시뮬레이션 테스트
    print("\n[1] Simulation Test")
    print("-" * 40)

    ofi_calc = OFICalculator(levels=5)
    vpin_calc = VPINCalculator(bucket_size=100, n_buckets=20)

    # 시뮬레이션 호가창 생성
    np.random.seed(42)

    for i in range(20):
        # 랜덤 호가창
        mid = 50000 + np.random.randn() * 100
        bids = [
            OrderBookLevel(price=mid - j * 10, quantity=np.random.uniform(0.5, 5), side='bid')
            for j in range(5)
        ]
        asks = [
            OrderBookLevel(price=mid + j * 10, quantity=np.random.uniform(0.5, 5), side='ask')
            for j in range(5)
        ]
        ob = OrderBook(symbol='BTC/USDT', timestamp=datetime.now(), bids=bids, asks=asks)

        ofi_l1, ofi_deep = ofi_calc.calculate(ob)

        # 랜덤 체결
        for _ in range(10):
            trade = Trade(
                symbol='BTC/USDT',
                timestamp=datetime.now(),
                price=mid,
                quantity=np.random.uniform(0.1, 2),
                side='buy' if np.random.random() > 0.5 else 'sell'
            )
            vpin_calc.add_trade(trade)

    ofi_norm = ofi_calc.get_normalized_ofi()
    vpin = vpin_calc.calculate_vpin()

    print(f"  OFI (normalized): {ofi_norm:.3f}")
    print(f"  VPIN: {vpin:.3f}")
    print(f"  Buckets: {len(vpin_calc.buckets)}")

    # 2. 실제 데이터 테스트 (ccxt)
    print("\n[2] Real Data Test (Binance)")
    print("-" * 40)

    try:
        result = quick_analysis('BTC/USDT', samples=5)

        if 'error' not in result:
            print(f"  Symbol: {result['symbol']}")
            print(f"  Samples: {result['samples']}")
            print(f"  Mid Price: ${result['mid_price']:,.2f}")
            print(f"  Spread: {result['spread_bps']:.2f} bps")
            print(f"  OFI: {result['ofi_current']:+.3f} (mean: {result['ofi_mean']:+.3f})")
            print(f"  VPIN: {result['vpin']:.3f}")
            print(f"  Depth Ratio: {result['depth_ratio']:.3f}")
            print(f"  Signal: {result['signal'].upper()} (strength: {result['signal_strength']:.2f})")
        else:
            print(f"  Error: {result['error']}")

    except Exception as e:
        print(f"  Real data test skipped: {e}")

    # 3. Daily Microstructure Test (AMFL Ch.19)
    print("\n[3] Daily Microstructure Test (AMFL Ch.19)")
    print("-" * 40)

    try:
        import yfinance as yf

        # 테스트 티커
        test_tickers = ['SPY', 'AAPL', 'GME']
        daily_analyzer = DailyMicrostructureAnalyzer(lookback_days=252)

        for ticker in test_tickers:
            print(f"\n  --- {ticker} ---")

            # 데이터 다운로드
            data = yf.download(ticker, period='1y', progress=False)

            if data.empty:
                print(f"    No data for {ticker}")
                continue

            # 분석
            result = daily_analyzer.analyze(ticker, data)

            print(f"    Amihud Lambda: {result.amihud.lambda_value:.4f}")
            print(f"      -> {result.amihud.interpretation}")
            print(f"    Roll Spread: {result.roll_spread.spread:.4f}%")
            print(f"      -> {result.roll_spread.interpretation}")
            print(f"    VPIN Approx: {result.vpin_approx.vpin:.4f} ({result.vpin_approx.toxicity_level})")
            print(f"      -> {result.vpin_approx.interpretation}")
            print(f"    Liquidity Score: {result.overall_liquidity_score:.1f}/100")
            print(f"    Risk Flags: {result.risk_flags or 'None'}")

    except Exception as e:
        print(f"  Daily microstructure test error: {e}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
