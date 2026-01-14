#!/usr/bin/env python3
"""
EIMAS Event Framework
=====================
통합 이벤트 감지 및 분류 프레임워크

주요 기능:
1. 예정된 이벤트 (Calendar-based)
2. 시장 데이터 기반 감지 (Quantitative)
3. 외부 소스 기반 감지 (Qualitative)
4. 자산군별 특화 이벤트
5. 이벤트 임팩트 분석

Author: EIMAS Team
Created: 2026-01-01
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf


# ============================================================================
# Enums
# ============================================================================

class EventType(Enum):
    """이벤트 유형"""
    # 예정된 이벤트 (Calendar)
    FOMC = "fomc"
    CPI = "cpi"
    PPI = "ppi"
    NFP = "nfp"  # Non-Farm Payrolls
    GDP = "gdp"
    PCE = "pce"
    ISM = "ism"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"

    # 시장 감지 이벤트 (Quantitative)
    VOLUME_SPIKE = "volume_spike"
    PRICE_SHOCK = "price_shock"
    VOLATILITY_SURGE = "volatility_surge"
    SPREAD_WIDENING = "spread_widening"
    FLOW_REVERSAL = "flow_reversal"
    OPTIONS_UNUSUAL = "options_unusual"

    # 유동성 이벤트 (Liquidity) - Alpha 핵심
    RRP_SURGE = "rrp_surge"           # RRP 급등 (유동성 흡수)
    RRP_DRAIN = "rrp_drain"           # RRP 급감 (유동성 방출) - 불리시
    TGA_BUILDUP = "tga_buildup"       # TGA 증가 (유동성 흡수) - 베어리시
    TGA_DRAWDOWN = "tga_drawdown"     # TGA 감소 (유동성 방출) - 불리시
    QT_ACCELERATION = "qt_acceleration"  # Fed 자산 축소 가속
    LIQUIDITY_STRESS = "liquidity_stress"  # Net Liquidity 급감
    LIQUIDITY_INJECTION = "liquidity_injection"  # 유동성 주입

    # 외부 감지 이벤트 (Qualitative)
    NEWS_SURGE = "news_surge"
    RATING_CHANGE = "rating_change"
    ANALYST_REVISION = "analyst_revision"
    INSIDER_ACTIVITY = "insider_activity"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"


class AssetClass(Enum):
    """자산군"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    INDEX = "index"


class EventImportance(Enum):
    """이벤트 중요도"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventTiming(Enum):
    """이벤트 타이밍"""
    SCHEDULED = "scheduled"      # 예정됨
    DETECTED = "detected"        # 감지됨 (실시간)
    HISTORICAL = "historical"    # 과거


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Event:
    """이벤트 기본 클래스"""
    event_id: str
    event_type: EventType
    asset_class: AssetClass
    timestamp: datetime
    timing: EventTiming
    importance: EventImportance

    # 상세 정보
    name: str
    description: str = ""
    ticker: Optional[str] = None

    # 수치 정보
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    surprise: Optional[float] = None  # actual - expected

    # 메타데이터
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_surprise(self) -> bool:
        """서프라이즈 여부"""
        if self.surprise is None:
            return False
        return abs(self.surprise) > 0.1  # 10% 이상 차이

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "asset_class": self.asset_class.value,
            "timestamp": self.timestamp.isoformat(),
            "timing": self.timing.value,
            "importance": self.importance.value,
            "name": self.name,
            "ticker": self.ticker,
            "surprise": self.surprise,
            "confidence": self.confidence
        }


@dataclass
class EventImpact:
    """이벤트 임팩트"""
    event: Event

    # 가격 영향
    price_change_pct: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0

    # 변동성 영향
    volatility_before: float = 0.0
    volatility_after: float = 0.0
    volatility_change: float = 0.0

    # 거래량 영향
    volume_ratio: float = 1.0  # vs 평균

    # 지속 시간
    impact_duration_hours: float = 0.0

    # 방향성
    direction: str = "neutral"  # bullish, bearish, neutral


# ============================================================================
# Economic Calendar (실제 날짜)
# ============================================================================

# 2025-2026 주요 경제 이벤트 캘린더
ECONOMIC_CALENDAR_2025_2026 = {
    # FOMC 회의 (2025)
    "fomc": [
        {"date": "2025-01-29", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-03-19", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-05-07", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-06-18", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-07-30", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-09-17", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-11-05", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-12-17", "name": "FOMC Meeting + SEP", "importance": 4},
        # 2026
        {"date": "2026-01-28", "name": "FOMC Meeting", "importance": 4},
        {"date": "2026-03-18", "name": "FOMC Meeting + SEP", "importance": 4},
    ],

    # CPI 발표 (매월 둘째주 화~수)
    "cpi": [
        {"date": "2025-01-15", "name": "CPI Dec", "importance": 4},
        {"date": "2025-02-12", "name": "CPI Jan", "importance": 4},
        {"date": "2025-03-12", "name": "CPI Feb", "importance": 4},
        {"date": "2025-04-10", "name": "CPI Mar", "importance": 4},
        {"date": "2025-05-13", "name": "CPI Apr", "importance": 4},
        {"date": "2025-06-11", "name": "CPI May", "importance": 4},
        {"date": "2025-07-11", "name": "CPI Jun", "importance": 4},
        {"date": "2025-08-12", "name": "CPI Jul", "importance": 4},
        {"date": "2025-09-11", "name": "CPI Aug", "importance": 4},
        {"date": "2025-10-10", "name": "CPI Sep", "importance": 4},
        {"date": "2025-11-13", "name": "CPI Oct", "importance": 4},
        {"date": "2025-12-10", "name": "CPI Nov", "importance": 4},
        # 2026
        {"date": "2026-01-14", "name": "CPI Dec 2025", "importance": 4},
        {"date": "2026-02-11", "name": "CPI Jan", "importance": 4},
        {"date": "2026-03-11", "name": "CPI Feb", "importance": 4},
    ],

    # NFP (매월 첫째주 금요일)
    "nfp": [
        {"date": "2025-01-10", "name": "NFP Dec", "importance": 4},
        {"date": "2025-02-07", "name": "NFP Jan", "importance": 4},
        {"date": "2025-03-07", "name": "NFP Feb", "importance": 4},
        {"date": "2025-04-04", "name": "NFP Mar", "importance": 4},
        {"date": "2025-05-02", "name": "NFP Apr", "importance": 4},
        {"date": "2025-06-06", "name": "NFP May", "importance": 4},
        {"date": "2025-07-03", "name": "NFP Jun", "importance": 4},
        {"date": "2025-08-01", "name": "NFP Jul", "importance": 4},
        {"date": "2025-09-05", "name": "NFP Aug", "importance": 4},
        {"date": "2025-10-03", "name": "NFP Sep", "importance": 4},
        {"date": "2025-11-07", "name": "NFP Oct", "importance": 4},
        {"date": "2025-12-05", "name": "NFP Nov", "importance": 4},
        # 2026
        {"date": "2026-01-09", "name": "NFP Dec 2025", "importance": 4},
        {"date": "2026-02-06", "name": "NFP Jan", "importance": 4},
        {"date": "2026-03-06", "name": "NFP Feb", "importance": 4},
    ],

    # PCE (매월 마지막주)
    "pce": [
        {"date": "2025-01-31", "name": "PCE Dec", "importance": 4},
        {"date": "2025-02-28", "name": "PCE Jan", "importance": 4},
        {"date": "2025-03-28", "name": "PCE Feb", "importance": 4},
        {"date": "2025-04-30", "name": "PCE Mar", "importance": 4},
        {"date": "2025-05-30", "name": "PCE Apr", "importance": 4},
        {"date": "2025-06-27", "name": "PCE May", "importance": 4},
        {"date": "2025-07-31", "name": "PCE Jun", "importance": 4},
        {"date": "2025-08-29", "name": "PCE Jul", "importance": 4},
        {"date": "2025-09-26", "name": "PCE Aug", "importance": 4},
        {"date": "2025-10-31", "name": "PCE Sep", "importance": 4},
        {"date": "2025-11-26", "name": "PCE Oct", "importance": 4},
        {"date": "2025-12-23", "name": "PCE Nov", "importance": 4},
        # 2026
        {"date": "2026-01-30", "name": "PCE Dec 2025", "importance": 4},
        {"date": "2026-02-27", "name": "PCE Jan", "importance": 4},
        {"date": "2026-03-27", "name": "PCE Feb", "importance": 4},
    ],

    # GDP (분기별)
    "gdp": [
        {"date": "2025-01-30", "name": "GDP Q4 Advance", "importance": 4},
        {"date": "2025-02-27", "name": "GDP Q4 Second", "importance": 3},
        {"date": "2025-03-27", "name": "GDP Q4 Final", "importance": 3},
        {"date": "2025-04-30", "name": "GDP Q1 Advance", "importance": 4},
        {"date": "2025-05-29", "name": "GDP Q1 Second", "importance": 3},
        {"date": "2025-06-26", "name": "GDP Q1 Final", "importance": 3},
        {"date": "2025-07-30", "name": "GDP Q2 Advance", "importance": 4},
        {"date": "2025-08-28", "name": "GDP Q2 Second", "importance": 3},
        {"date": "2025-09-25", "name": "GDP Q2 Final", "importance": 3},
        {"date": "2025-10-30", "name": "GDP Q3 Advance", "importance": 4},
        {"date": "2025-11-27", "name": "GDP Q3 Second", "importance": 3},
        {"date": "2025-12-22", "name": "GDP Q3 Final", "importance": 3},
        # 2026
        {"date": "2026-01-29", "name": "GDP Q4 Advance", "importance": 4},
        {"date": "2026-02-26", "name": "GDP Q4 Second", "importance": 3},
        {"date": "2026-03-26", "name": "GDP Q4 Final", "importance": 3},
    ],
}


# ============================================================================
# Quantitative Event Detector
# ============================================================================

class QuantitativeEventDetector:
    """시장 데이터 기반 이벤트 감지"""

    def __init__(
        self,
        volume_threshold: float = 3.0,  # 표준편차
        price_threshold: float = 2.0,   # 퍼센트
        volatility_threshold: float = 2.0  # 표준편차
    ):
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        self.volatility_threshold = volatility_threshold

    def detect_volume_spike(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> List[Event]:
        """거래량 급변 감지"""
        events = []

        # 컬럼 이름 처리 (멀티인덱스 대응)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        if 'Volume' not in data.columns:
            return events

        volume = data['Volume'].copy()
        vol_mean = volume.rolling(lookback).mean()
        vol_std = volume.rolling(lookback).std()

        z_scores = (volume - vol_mean) / vol_std

        # 임계값 초과 시점 찾기
        spikes = z_scores[z_scores > self.volume_threshold].dropna()

        for idx in spikes.index:
            z_val = float(spikes[idx])
            vol_val = float(volume[idx])
            events.append(Event(
                event_id=f"vol_spike_{idx}",
                event_type=EventType.VOLUME_SPIKE,
                asset_class=AssetClass.EQUITY,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH if z_val > 4 else EventImportance.MEDIUM,
                name=f"Volume Spike ({z_val:.1f}σ)",
                description=f"거래량이 평균 대비 {z_val:.1f} 표준편차 상승",
                confidence=min(z_val / 5, 1.0),
                metadata={"z_score": z_val, "volume": vol_val}
            ))

        return events

    def detect_price_shock(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> List[Event]:
        """가격 급변 감지"""
        events = []

        # 컬럼 이름 처리 (멀티인덱스 대응)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        if 'Close' not in data.columns:
            return events

        close = data['Close'].copy()
        returns = close.pct_change() * 100  # 퍼센트

        # 큰 변동 찾기
        large_moves = returns[abs(returns) > self.price_threshold].dropna()

        for idx in large_moves.index:
            ret = float(large_moves[idx])
            direction = "상승" if ret > 0 else "하락"
            events.append(Event(
                event_id=f"price_shock_{idx}",
                event_type=EventType.PRICE_SHOCK,
                asset_class=AssetClass.EQUITY,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                timing=EventTiming.DETECTED,
                importance=EventImportance.CRITICAL if abs(ret) > 5 else EventImportance.HIGH,
                name=f"Price Shock ({ret:+.1f}%)",
                description=f"가격 {direction} {abs(ret):.1f}%",
                confidence=min(abs(ret) / 10, 1.0),
                metadata={"return_pct": ret, "close": float(close[idx])}
            ))

        return events

    def detect_volatility_surge(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> List[Event]:
        """변동성 급등 감지"""
        events = []

        # 컬럼 이름 처리 (멀티인덱스 대응)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        if 'Close' not in data.columns:
            return events

        # 실현 변동성 계산
        close = data['Close'].copy()
        returns = close.pct_change()
        realized_vol = returns.rolling(5).std() * np.sqrt(252) * 100

        vol_mean = realized_vol.rolling(lookback).mean()
        vol_std = realized_vol.rolling(lookback).std()

        z_scores = (realized_vol - vol_mean) / vol_std

        # 급등 시점
        surges = z_scores[z_scores > self.volatility_threshold].dropna()

        for idx in surges.index:
            z_val = float(surges[idx])
            rv = float(realized_vol[idx])
            events.append(Event(
                event_id=f"vol_surge_{idx}",
                event_type=EventType.VOLATILITY_SURGE,
                asset_class=AssetClass.EQUITY,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH,
                name=f"Volatility Surge ({z_val:.1f}σ)",
                description=f"변동성 급등 (연환산 {rv:.1f}%)",
                confidence=min(z_val / 4, 1.0),
                metadata={"z_score": z_val, "realized_vol": rv}
            ))

        return events

    def detect_spread_widening(
        self,
        spread_data: pd.DataFrame,
        lookback: int = 20
    ) -> List[Event]:
        """스프레드 확대 감지 (신용, 금리 등)"""
        events = []

        if spread_data.empty:
            return events

        spread = spread_data.iloc[:, 0]  # 첫 번째 컬럼
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()

        z_scores = (spread - spread_mean) / spread_std

        # 급등 시점
        widenings = z_scores[z_scores > 2.0].dropna()

        for idx in widenings.index:
            z_val = float(widenings[idx])
            sp = float(spread[idx])
            events.append(Event(
                event_id=f"spread_wide_{idx}",
                event_type=EventType.SPREAD_WIDENING,
                asset_class=AssetClass.BOND,
                timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH if z_val > 3 else EventImportance.MEDIUM,
                name=f"Spread Widening ({z_val:.1f}σ)",
                description=f"스프레드 확대 {sp:.2f}bp",
                confidence=min(z_val / 4, 1.0),
                metadata={"z_score": z_val, "spread": sp}
            ))

        return events

    def detect_options_unusual(
        self,
        ticker: str
    ) -> List[Event]:
        """옵션 이상 활동 감지"""
        events = []

        try:
            from lib.options_flow import OptionsFlowMonitor, FlowSignal

            monitor = OptionsFlowMonitor()
            summary = monitor.analyze_ticker(ticker)

            if summary is None:
                return events

            # 1. Put/Call Ratio 급변
            if summary.put_call_ratio > 1.5:
                events.append(Event(
                    event_id=f"pc_ratio_{ticker}_{datetime.now().date()}",
                    event_type=EventType.OPTIONS_UNUSUAL,
                    asset_class=AssetClass.EQUITY,
                    timestamp=datetime.now(),
                    timing=EventTiming.DETECTED,
                    importance=EventImportance.HIGH,
                    name=f"High Put/Call Ratio ({summary.put_call_ratio:.2f})",
                    ticker=ticker,
                    description=f"Put/Call 비율 상승 (베어리시 시그널)",
                    confidence=min(summary.put_call_ratio / 2, 1.0),
                    metadata={
                        "put_call_ratio": summary.put_call_ratio,
                        "total_call_volume": summary.total_call_volume,
                        "total_put_volume": summary.total_put_volume
                    }
                ))
            elif summary.put_call_ratio < 0.5:
                events.append(Event(
                    event_id=f"pc_ratio_{ticker}_{datetime.now().date()}",
                    event_type=EventType.OPTIONS_UNUSUAL,
                    asset_class=AssetClass.EQUITY,
                    timestamp=datetime.now(),
                    timing=EventTiming.DETECTED,
                    importance=EventImportance.MEDIUM,
                    name=f"Low Put/Call Ratio ({summary.put_call_ratio:.2f})",
                    ticker=ticker,
                    description=f"Put/Call 비율 하락 (불리시 시그널)",
                    confidence=min((1 - summary.put_call_ratio) * 2, 1.0),
                    metadata={
                        "put_call_ratio": summary.put_call_ratio,
                        "total_call_volume": summary.total_call_volume,
                        "total_put_volume": summary.total_put_volume
                    }
                ))

            # 2. IV Percentile 이상
            if summary.iv_percentile > 80:
                events.append(Event(
                    event_id=f"iv_high_{ticker}_{datetime.now().date()}",
                    event_type=EventType.OPTIONS_UNUSUAL,
                    asset_class=AssetClass.EQUITY,
                    timestamp=datetime.now(),
                    timing=EventTiming.DETECTED,
                    importance=EventImportance.HIGH,
                    name=f"High IV Percentile ({summary.iv_percentile:.0f}%)",
                    ticker=ticker,
                    description=f"내재변동성 상위 {100 - summary.iv_percentile:.0f}%",
                    confidence=summary.iv_percentile / 100,
                    metadata={
                        "iv_percentile": summary.iv_percentile,
                        "max_pain": summary.max_pain
                    }
                ))

            # 3. 대규모 프리미엄 편향
            total_premium = summary.bullish_premium + summary.bearish_premium
            if total_premium > 0:
                bullish_pct = summary.bullish_premium / total_premium
                if bullish_pct > 0.7:
                    events.append(Event(
                        event_id=f"flow_bullish_{ticker}_{datetime.now().date()}",
                        event_type=EventType.FLOW_REVERSAL,
                        asset_class=AssetClass.EQUITY,
                        timestamp=datetime.now(),
                        timing=EventTiming.DETECTED,
                        importance=EventImportance.HIGH,
                        name=f"Bullish Flow Dominance ({bullish_pct*100:.0f}%)",
                        ticker=ticker,
                        description=f"불리시 옵션 플로우 우세 (프리미엄 ${summary.bullish_premium/1e6:.1f}M)",
                        confidence=bullish_pct,
                        metadata={
                            "bullish_premium": summary.bullish_premium,
                            "bearish_premium": summary.bearish_premium,
                            "signal": "bullish"
                        }
                    ))
                elif bullish_pct < 0.3:
                    events.append(Event(
                        event_id=f"flow_bearish_{ticker}_{datetime.now().date()}",
                        event_type=EventType.FLOW_REVERSAL,
                        asset_class=AssetClass.EQUITY,
                        timestamp=datetime.now(),
                        timing=EventTiming.DETECTED,
                        importance=EventImportance.HIGH,
                        name=f"Bearish Flow Dominance ({(1-bullish_pct)*100:.0f}%)",
                        ticker=ticker,
                        description=f"베어리시 옵션 플로우 우세 (프리미엄 ${summary.bearish_premium/1e6:.1f}M)",
                        confidence=1 - bullish_pct,
                        metadata={
                            "bullish_premium": summary.bullish_premium,
                            "bearish_premium": summary.bearish_premium,
                            "signal": "bearish"
                        }
                    ))

            # 4. 개별 대형 거래
            for flow in summary.unusual_flows[:3]:
                if flow.premium_estimate > 1_000_000:  # $1M 이상
                    events.append(Event(
                        event_id=f"large_flow_{ticker}_{flow.strike}_{datetime.now().date()}",
                        event_type=EventType.OPTIONS_UNUSUAL,
                        asset_class=AssetClass.EQUITY,
                        timestamp=datetime.now(),
                        timing=EventTiming.DETECTED,
                        importance=EventImportance.CRITICAL if flow.premium_estimate > 5_000_000 else EventImportance.HIGH,
                        name=f"Large {flow.option_type.value.upper()} Flow ${flow.strike:.0f}",
                        ticker=ticker,
                        description=f"{flow.reasoning}. 프리미엄: ${flow.premium_estimate/1e6:.1f}M",
                        confidence=min(flow.premium_estimate / 10_000_000, 1.0),
                        metadata={
                            "option_type": flow.option_type.value,
                            "strike": flow.strike,
                            "expiry": flow.expiry,
                            "volume": flow.volume,
                            "premium": flow.premium_estimate
                        }
                    ))

        except Exception as e:
            pass  # 옵션 데이터 없으면 무시

        return events

    def detect_momentum_divergence(
        self,
        data: pd.DataFrame,
        lookback: int = 14
    ) -> List[Event]:
        """모멘텀 다이버전스 감지 (RSI/가격 괴리)"""
        events = []

        # 컬럼 이름 처리 (멀티인덱스 대응)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)

        if 'Close' not in data.columns:
            return events

        close = data['Close'].copy()

        # RSI 계산
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 최근 20일간 다이버전스 체크
        for i in range(-20, -5):
            try:
                # 가격은 상승하는데 RSI는 하락 (베어리시 다이버전스)
                price_higher = close.iloc[i] > close.iloc[i-5]
                rsi_lower = rsi.iloc[i] < rsi.iloc[i-5]

                if price_higher and rsi_lower and rsi.iloc[i] > 70:
                    events.append(Event(
                        event_id=f"bearish_div_{data.index[i]}",
                        event_type=EventType.PRICE_SHOCK,
                        asset_class=AssetClass.EQUITY,
                        timestamp=data.index[i] if isinstance(data.index[i], datetime) else datetime.now(),
                        timing=EventTiming.DETECTED,
                        importance=EventImportance.MEDIUM,
                        name=f"Bearish RSI Divergence (RSI: {rsi.iloc[i]:.0f})",
                        description="가격 상승 중 RSI 하락 - 모멘텀 약화 신호",
                        confidence=0.6,
                        metadata={"rsi": float(rsi.iloc[i]), "pattern": "bearish_divergence"}
                    ))

                # 가격은 하락하는데 RSI는 상승 (불리시 다이버전스)
                price_lower = close.iloc[i] < close.iloc[i-5]
                rsi_higher = rsi.iloc[i] > rsi.iloc[i-5]

                if price_lower and rsi_higher and rsi.iloc[i] < 30:
                    events.append(Event(
                        event_id=f"bullish_div_{data.index[i]}",
                        event_type=EventType.PRICE_SHOCK,
                        asset_class=AssetClass.EQUITY,
                        timestamp=data.index[i] if isinstance(data.index[i], datetime) else datetime.now(),
                        timing=EventTiming.DETECTED,
                        importance=EventImportance.MEDIUM,
                        name=f"Bullish RSI Divergence (RSI: {rsi.iloc[i]:.0f})",
                        description="가격 하락 중 RSI 상승 - 반등 가능성 신호",
                        confidence=0.6,
                        metadata={"rsi": float(rsi.iloc[i]), "pattern": "bullish_divergence"}
                    ))
            except (IndexError, KeyError):
                continue

        return events

    def detect_correlation_breakdown(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        lookback: int = 60,
        threshold: float = 0.3
    ) -> List[Event]:
        """상관관계 이탈 감지"""
        events = []

        try:
            # 컬럼 처리
            if isinstance(data1.columns, pd.MultiIndex):
                data1 = data1.droplevel(1, axis=1)
            if isinstance(data2.columns, pd.MultiIndex):
                data2 = data2.droplevel(1, axis=1)

            ret1 = data1['Close'].pct_change()
            ret2 = data2['Close'].pct_change()

            # 공통 인덱스
            common_idx = ret1.index.intersection(ret2.index)
            ret1 = ret1[common_idx]
            ret2 = ret2[common_idx]

            # 롤링 상관관계
            rolling_corr = ret1.rolling(lookback).corr(ret2)
            long_term_corr = ret1.corr(ret2)

            # 최근 상관관계가 장기 상관관계에서 크게 벗어남
            recent_corr = rolling_corr.iloc[-1]
            corr_change = abs(recent_corr - long_term_corr)

            if corr_change > threshold:
                events.append(Event(
                    event_id=f"corr_break_{ticker1}_{ticker2}_{datetime.now().date()}",
                    event_type=EventType.FLOW_REVERSAL,
                    asset_class=AssetClass.INDEX,
                    timestamp=datetime.now(),
                    timing=EventTiming.DETECTED,
                    importance=EventImportance.HIGH if corr_change > 0.5 else EventImportance.MEDIUM,
                    name=f"Correlation Breakdown {ticker1}/{ticker2}",
                    description=f"상관관계 변화: {long_term_corr:.2f} → {recent_corr:.2f}",
                    confidence=min(corr_change * 2, 1.0),
                    metadata={
                        "ticker1": ticker1,
                        "ticker2": ticker2,
                        "long_term_corr": float(long_term_corr),
                        "recent_corr": float(recent_corr),
                        "change": float(corr_change)
                    }
                ))

        except Exception:
            pass

        return events

    def detect_liquidity_events(
        self,
        rrp: float = None,
        rrp_delta: float = None,
        rrp_delta_pct: float = None,
        tga: float = None,
        tga_delta: float = None,
        fed_assets_delta: float = None,
        net_liquidity: float = None,
        liquidity_regime: str = None
    ) -> List[Event]:
        """
        유동성 이벤트 감지 - Alpha 핵심

        Parameters:
        -----------
        rrp : float
            현재 RRP 잔액 (Billions)
        rrp_delta : float
            RRP 일간 변화 (Billions)
        rrp_delta_pct : float
            RRP 변화율 (%)
        tga : float
            현재 TGA 잔액 (Billions)
        tga_delta : float
            TGA 주간 변화 (Billions)
        fed_assets_delta : float
            Fed 자산 주간 변화 (Billions)
        net_liquidity : float
            순유동성 (Billions)
        liquidity_regime : str
            유동성 레짐 (Abundant/Normal/Tight/Stressed)

        Returns:
        --------
        List[Event] : 감지된 유동성 이벤트들
        """
        events = []
        now = datetime.now()

        # 1. RRP 급감 (유동성 방출) - 불리시 신호
        if rrp_delta is not None and rrp_delta < -50:
            importance = EventImportance.CRITICAL if rrp_delta < -100 else EventImportance.HIGH
            events.append(Event(
                event_id=f"rrp_drain_{now.date()}",
                event_type=EventType.RRP_DRAIN,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=importance,
                name=f"RRP Drain ({rrp_delta:+.0f}B)",
                description=f"RRP 급감 ${abs(rrp_delta):.0f}B ({rrp_delta_pct:+.1f}%) - 유동성 시장 유입, Risk-On 우호적",
                confidence=min(abs(rrp_delta) / 100, 1.0),
                metadata={
                    "rrp": rrp,
                    "rrp_delta": rrp_delta,
                    "rrp_delta_pct": rrp_delta_pct,
                    "signal": "bullish",
                    "mechanism": "RRP 감소 → MMF/은행 자금 시장 유입 → 위험자산 상승 압력"
                }
            ))

        # 2. RRP 급증 (유동성 흡수) - 베어리시 신호
        elif rrp_delta is not None and rrp_delta > 50:
            events.append(Event(
                event_id=f"rrp_surge_{now.date()}",
                event_type=EventType.RRP_SURGE,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH,
                name=f"RRP Surge ({rrp_delta:+.0f}B)",
                description=f"RRP 급증 ${rrp_delta:.0f}B - 유동성 Fed로 회수, Risk-Off 압력",
                confidence=min(rrp_delta / 100, 1.0),
                metadata={
                    "rrp": rrp,
                    "rrp_delta": rrp_delta,
                    "signal": "bearish"
                }
            ))

        # 3. TGA 증가 (유동성 흡수) - 베어리시
        if tga_delta is not None and tga_delta > 50:
            events.append(Event(
                event_id=f"tga_buildup_{now.date()}",
                event_type=EventType.TGA_BUILDUP,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH,
                name=f"TGA Buildup ({tga_delta:+.0f}B)",
                description=f"재무부 현금 축적 ${tga_delta:.0f}B - 시장 유동성 흡수, 주의 필요",
                confidence=min(tga_delta / 100, 1.0),
                metadata={
                    "tga": tga,
                    "tga_delta": tga_delta,
                    "signal": "bearish"
                }
            ))

        # 4. TGA 감소 (유동성 방출) - 불리시
        elif tga_delta is not None and tga_delta < -50:
            events.append(Event(
                event_id=f"tga_drawdown_{now.date()}",
                event_type=EventType.TGA_DRAWDOWN,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH,
                name=f"TGA Drawdown ({tga_delta:+.0f}B)",
                description=f"재무부 지출 ${abs(tga_delta):.0f}B - 시장 유동성 주입, Risk-On 우호적",
                confidence=min(abs(tga_delta) / 100, 1.0),
                metadata={
                    "tga": tga,
                    "tga_delta": tga_delta,
                    "signal": "bullish"
                }
            ))

        # 5. QT 가속 (Fed 자산 축소)
        if fed_assets_delta is not None and fed_assets_delta < -20:
            events.append(Event(
                event_id=f"qt_accel_{now.date()}",
                event_type=EventType.QT_ACCELERATION,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.HIGH,
                name=f"QT Acceleration ({fed_assets_delta:+.0f}B/wk)",
                description=f"Fed 자산 주간 {abs(fed_assets_delta):.0f}B 축소 - 구조적 유동성 감소",
                confidence=min(abs(fed_assets_delta) / 50, 1.0),
                metadata={
                    "fed_assets_delta": fed_assets_delta,
                    "signal": "bearish"
                }
            ))

        # 6. 유동성 스트레스 (Net Liquidity 급감)
        if liquidity_regime == "Stressed" or (net_liquidity is not None and net_liquidity < 2500):
            events.append(Event(
                event_id=f"liq_stress_{now.date()}",
                event_type=EventType.LIQUIDITY_STRESS,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.CRITICAL,
                name=f"Liquidity Stress (${net_liquidity/1000:.2f}T)",
                description=f"순유동성 ${net_liquidity/1000:.2f}T - 스트레스 구간, 변동성 급등 위험",
                confidence=0.9,
                metadata={
                    "net_liquidity": net_liquidity,
                    "regime": liquidity_regime,
                    "signal": "high_risk"
                }
            ))

        # 7. 유동성 풍부 (Net Liquidity 높음)
        elif liquidity_regime == "Abundant" or (net_liquidity is not None and net_liquidity > 4000):
            events.append(Event(
                event_id=f"liq_injection_{now.date()}",
                event_type=EventType.LIQUIDITY_INJECTION,
                asset_class=AssetClass.INDEX,
                timestamp=now,
                timing=EventTiming.DETECTED,
                importance=EventImportance.MEDIUM,
                name=f"Liquidity Abundant (${net_liquidity/1000:.2f}T)",
                description=f"순유동성 풍부 ${net_liquidity/1000:.2f}T - Risk-On 환경",
                confidence=0.8,
                metadata={
                    "net_liquidity": net_liquidity,
                    "regime": liquidity_regime,
                    "signal": "bullish"
                }
            ))

        return events

    def detect_all(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        include_options: bool = False,
        liquidity_data: dict = None
    ) -> List[Event]:
        """모든 정량적 이벤트 감지"""
        all_events = []

        all_events.extend(self.detect_volume_spike(data))
        all_events.extend(self.detect_price_shock(data))
        all_events.extend(self.detect_volatility_surge(data))
        all_events.extend(self.detect_momentum_divergence(data))

        # 티커 정보 추가
        if ticker:
            for event in all_events:
                event.ticker = ticker

            # 옵션 이벤트 감지 (선택적)
            if include_options:
                all_events.extend(self.detect_options_unusual(ticker))

        # 유동성 이벤트 감지 (FRED 데이터 필요)
        if liquidity_data:
            all_events.extend(self.detect_liquidity_events(**liquidity_data))

        return sorted(all_events, key=lambda e: e.timestamp, reverse=True)


# ============================================================================
# Calendar Event Manager
# ============================================================================

class CalendarEventManager:
    """예정된 이벤트 관리"""

    def __init__(self):
        self.calendar = ECONOMIC_CALENDAR_2025_2026

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        event_types: List[EventType] = None,
        min_importance: int = 1
    ) -> List[Event]:
        """다가오는 이벤트 조회"""
        events = []
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        for event_type_str, event_list in self.calendar.items():
            # 필터링
            if event_types:
                try:
                    et = EventType(event_type_str)
                    if et not in event_types:
                        continue
                except ValueError:
                    continue

            for event_data in event_list:
                event_date = datetime.strptime(event_data["date"], "%Y-%m-%d")

                if today <= event_date <= end_date:
                    if event_data["importance"] >= min_importance:
                        events.append(Event(
                            event_id=f"{event_type_str}_{event_data['date']}",
                            event_type=EventType(event_type_str),
                            asset_class=AssetClass.INDEX,
                            timestamp=event_date,
                            timing=EventTiming.SCHEDULED,
                            importance=EventImportance(event_data["importance"]),
                            name=event_data["name"],
                            source="economic_calendar"
                        ))

        return sorted(events, key=lambda e: e.timestamp)

    def get_events_for_date(self, date: datetime) -> List[Event]:
        """특정 날짜의 이벤트"""
        date_str = date.strftime("%Y-%m-%d")
        events = []

        for event_type_str, event_list in self.calendar.items():
            for event_data in event_list:
                if event_data["date"] == date_str:
                    events.append(Event(
                        event_id=f"{event_type_str}_{date_str}",
                        event_type=EventType(event_type_str),
                        asset_class=AssetClass.INDEX,
                        timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                        timing=EventTiming.SCHEDULED,
                        importance=EventImportance(event_data["importance"]),
                        name=event_data["name"],
                        source="economic_calendar"
                    ))

        return events

    def days_to_next_event(self, event_type: EventType) -> int:
        """다음 특정 이벤트까지 일수"""
        today = datetime.now()

        if event_type.value not in self.calendar:
            return -1

        for event_data in self.calendar[event_type.value]:
            event_date = datetime.strptime(event_data["date"], "%Y-%m-%d")
            if event_date > today:
                return (event_date - today).days

        return -1


# ============================================================================
# Earnings Calendar
# ============================================================================

class EarningsCalendar:
    """실적 발표 캘린더"""

    def __init__(self):
        pass

    def get_upcoming_earnings(
        self,
        tickers: List[str],
        days_ahead: int = 14
    ) -> List[Event]:
        """다가오는 실적 발표 조회"""
        events = []
        today = datetime.now()

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar

                if calendar is not None and 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if isinstance(earnings_dates, list) and earnings_dates:
                        earnings_date = earnings_dates[0]
                        if isinstance(earnings_date, datetime):
                            if today <= earnings_date <= today + timedelta(days=days_ahead):
                                events.append(Event(
                                    event_id=f"earnings_{ticker}_{earnings_date.date()}",
                                    event_type=EventType.EARNINGS,
                                    asset_class=AssetClass.EQUITY,
                                    timestamp=earnings_date,
                                    timing=EventTiming.SCHEDULED,
                                    importance=EventImportance.HIGH,
                                    name=f"{ticker} Earnings",
                                    ticker=ticker,
                                    source="yfinance"
                                ))
            except Exception:
                pass

        return sorted(events, key=lambda e: e.timestamp)


# ============================================================================
# Unified Event Framework
# ============================================================================

class EventFramework:
    """통합 이벤트 프레임워크"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.calendar_manager = CalendarEventManager()
        self.quantitative_detector = QuantitativeEventDetector()
        self.earnings_calendar = EarningsCalendar()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventFramework] {msg}")

    def get_all_upcoming_events(
        self,
        days_ahead: int = 7,
        tickers: List[str] = None
    ) -> List[Event]:
        """모든 예정된 이벤트 조회"""
        events = []

        # 1. 경제 이벤트
        self._log("Fetching economic calendar events...")
        econ_events = self.calendar_manager.get_upcoming_events(
            days_ahead=days_ahead,
            min_importance=3  # HIGH 이상
        )
        events.extend(econ_events)
        self._log(f"  Found {len(econ_events)} economic events")

        # 2. 실적 발표
        if tickers:
            self._log("Fetching earnings calendar...")
            earnings_events = self.earnings_calendar.get_upcoming_earnings(
                tickers=tickers,
                days_ahead=days_ahead
            )
            events.extend(earnings_events)
            self._log(f"  Found {len(earnings_events)} earnings events")

        return sorted(events, key=lambda e: e.timestamp)

    def detect_market_events(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> List[Event]:
        """시장 데이터에서 이벤트 감지"""
        all_events = []

        for ticker, df in data.items():
            if df.empty:
                continue

            self._log(f"Detecting events for {ticker}...")
            events = self.quantitative_detector.detect_all(df, ticker=ticker)
            all_events.extend(events)
            self._log(f"  Found {len(events)} events")

        return all_events

    def analyze_event_context(
        self,
        date: datetime,
        ticker: str = None,
        lookback_days: int = 5,
        lookahead_days: int = 5
    ) -> Dict[str, Any]:
        """이벤트 컨텍스트 분석"""
        context = {
            "date": date,
            "ticker": ticker,
            "scheduled_events": [],
            "detected_events": [],
            "pre_event_setup": {},
            "post_event_reaction": {}
        }

        # 해당 날짜 예정 이벤트
        context["scheduled_events"] = self.calendar_manager.get_events_for_date(date)

        # 주변 이벤트
        nearby_events = self.calendar_manager.get_upcoming_events(
            days_ahead=lookahead_days
        )
        context["nearby_events"] = nearby_events

        # 다음 주요 이벤트까지 일수
        context["days_to_fomc"] = self.calendar_manager.days_to_next_event(EventType.FOMC)
        context["days_to_cpi"] = self.calendar_manager.days_to_next_event(EventType.CPI)
        context["days_to_nfp"] = self.calendar_manager.days_to_next_event(EventType.NFP)

        return context

    def get_event_summary(self, days_ahead: int = 7) -> str:
        """이벤트 요약 출력"""
        events = self.get_all_upcoming_events(days_ahead=days_ahead)

        lines = [
            "=" * 60,
            "EIMAS Event Framework - Upcoming Events",
            "=" * 60,
            ""
        ]

        # 날짜별 그룹화
        by_date = {}
        for event in events:
            date_str = event.timestamp.strftime("%Y-%m-%d (%a)")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(event)

        for date_str, date_events in sorted(by_date.items()):
            lines.append(f"\n{date_str}")
            lines.append("-" * 40)
            for event in date_events:
                importance_icon = "🔴" if event.importance == EventImportance.CRITICAL else "🟡"
                lines.append(f"  {importance_icon} {event.name}")

        # 다음 주요 이벤트
        lines.append("\n" + "=" * 60)
        lines.append("Days to Next Major Events:")
        lines.append(f"  FOMC: {self.calendar_manager.days_to_next_event(EventType.FOMC)} days")
        lines.append(f"  CPI:  {self.calendar_manager.days_to_next_event(EventType.CPI)} days")
        lines.append(f"  NFP:  {self.calendar_manager.days_to_next_event(EventType.NFP)} days")
        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# Event Impact Analyzer
# ============================================================================

class EventImpactAnalyzer:
    """이벤트 임팩트 분석"""

    # 과거 이벤트별 평균 임팩트
    HISTORICAL_IMPACT = {
        EventType.FOMC: {
            "avg_move_pct": 1.5,
            "avg_duration_hours": 48,
            "affected_assets": ["SPY", "QQQ", "TLT", "DXY", "GLD"],
            "typical_pattern": "Initial spike, then mean reversion over 24h"
        },
        EventType.CPI: {
            "avg_move_pct": 1.2,
            "avg_duration_hours": 8,
            "affected_assets": ["SPY", "QQQ", "TLT", "TIPS"],
            "typical_pattern": "Morning spike, settled by close"
        },
        EventType.NFP: {
            "avg_move_pct": 0.8,
            "avg_duration_hours": 4,
            "affected_assets": ["SPY", "DXY", "TLT"],
            "typical_pattern": "Quick reaction, often reversed"
        },
        EventType.EARNINGS: {
            "avg_move_pct": 5.0,
            "avg_duration_hours": 24,
            "affected_assets": ["individual_stock"],
            "typical_pattern": "Gap up/down, then trend for days"
        }
    }

    def get_expected_impact(self, event_type: EventType) -> Dict[str, Any]:
        """예상 임팩트 조회"""
        return self.HISTORICAL_IMPACT.get(event_type, {
            "avg_move_pct": 0.5,
            "avg_duration_hours": 4,
            "affected_assets": ["general"],
            "typical_pattern": "Unknown"
        })

    def analyze_historical_impact(
        self,
        event_type: EventType,
        ticker: str,
        lookback_events: int = 10
    ) -> Dict[str, Any]:
        """과거 이벤트 임팩트 분석"""
        # TODO: 실제 과거 데이터로 분석
        expected = self.get_expected_impact(event_type)

        return {
            "event_type": event_type.value,
            "ticker": ticker,
            "sample_size": lookback_events,
            "avg_move_pct": expected["avg_move_pct"],
            "avg_duration_hours": expected["avg_duration_hours"],
            "win_rate_long": 0.52,  # placeholder
            "win_rate_short": 0.48,
            "typical_pattern": expected["typical_pattern"]
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # 테스트
    framework = EventFramework(verbose=True)

    # 1. 다가오는 이벤트 조회
    print(framework.get_event_summary(days_ahead=14))

    # 2. 시장 데이터에서 이벤트 감지
    print("\n" + "=" * 60)
    print("Quantitative Event Detection")
    print("=" * 60)

    print("\n[1] Price/Volume/Volatility Detection for SPY...")
    spy = yf.download("SPY", period="3mo", progress=False)
    if not spy.empty:
        events = framework.quantitative_detector.detect_all(spy, ticker="SPY")
        print(f"Found {len(events)} events")
        for event in events[:5]:
            print(f"  [{event.importance.name}] {event.timestamp.strftime('%Y-%m-%d')}: {event.name}")

    # 3. 상관관계 이탈 감지
    print("\n[2] Correlation Breakdown Detection (SPY vs QQQ)...")
    qqq = yf.download("QQQ", period="6mo", progress=False)
    if not spy.empty and not qqq.empty:
        corr_events = framework.quantitative_detector.detect_correlation_breakdown(
            spy, qqq, "SPY", "QQQ"
        )
        if corr_events:
            for event in corr_events:
                print(f"  {event.name}: {event.description}")
        else:
            print("  No correlation breakdown detected")

    # 4. 옵션 플로우 감지 (선택적)
    print("\n[3] Options Flow Detection for SPY...")
    try:
        opt_events = framework.quantitative_detector.detect_options_unusual("SPY")
        if opt_events:
            for event in opt_events[:3]:
                print(f"  [{event.importance.name}] {event.name}")
        else:
            print("  No unusual options activity detected")
    except Exception as e:
        print(f"  Options detection skipped: {e}")

    # 5. 유동성 이벤트 감지 (FRED 연동)
    print("\n" + "=" * 60)
    print("Liquidity Event Detection (FRED)")
    print("=" * 60)

    try:
        from lib.fred_collector import FREDCollector

        print("\n[4] Fetching FRED liquidity data...")
        collector = FREDCollector()
        summary = collector.collect_all()

        print(f"  RRP: ${summary.rrp:.0f}B (delta: {summary.rrp_delta:+.0f}B)")
        print(f"  TGA: ${summary.tga:.0f}B (delta: {summary.tga_delta:+.0f}B)")
        print(f"  Fed Assets: ${summary.fed_assets:.2f}T (delta: {summary.fed_assets_delta:+.0f}B/wk)")
        print(f"  Net Liquidity: ${summary.net_liquidity/1000:.2f}T")
        print(f"  Regime: {summary.liquidity_regime}")

        liquidity_data = {
            'rrp': summary.rrp,
            'rrp_delta': summary.rrp_delta,
            'rrp_delta_pct': summary.rrp_delta_pct,
            'tga': summary.tga,
            'tga_delta': summary.tga_delta,
            'fed_assets_delta': summary.fed_assets_delta,
            'net_liquidity': summary.net_liquidity,
            'liquidity_regime': summary.liquidity_regime
        }

        liq_events = framework.quantitative_detector.detect_liquidity_events(**liquidity_data)
        print(f"\n  Found {len(liq_events)} liquidity events:")
        for event in liq_events:
            signal = event.metadata.get('signal', 'unknown')
            print(f"    [{event.importance.name}] {event.name}")
            print(f"      {event.description}")
            print(f"      Signal: {signal.upper()}")

    except Exception as e:
        print(f"  Liquidity detection skipped: {e}")

    # 6. 임팩트 분석
    print("\n" + "=" * 60)
    print("Event Impact Analysis")
    print("=" * 60)
    analyzer = EventImpactAnalyzer()
    for event_type in [EventType.FOMC, EventType.CPI, EventType.NFP]:
        impact = analyzer.get_expected_impact(event_type)
        print(f"\n{event_type.value.upper()}:")
        print(f"  Avg Move: {impact['avg_move_pct']}%")
        print(f"  Duration: {impact['avg_duration_hours']}h")
        print(f"  Pattern: {impact['typical_pattern']}")

    print("\n" + "=" * 60)
    print("Event Framework Test Complete!")
    print("=" * 60)
