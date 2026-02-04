#!/usr/bin/env python3
"""
Event Framework - Event Detector
============================================================

Quantitative event detection using statistical thresholds

Economic Foundation:
    - Volatility spikes: GARCH-based detection
    - Price jumps: z-score thresholds
    - Volume anomalies: Rolling mean comparison

Class:
    - QuantitativeEventDetector: Statistical event detector (~660 lines)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .schemas import Event, EventImpact
from .enums import EventType, AssetClass, EventImportance, EventTiming

logger = logging.getLogger(__name__)


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
