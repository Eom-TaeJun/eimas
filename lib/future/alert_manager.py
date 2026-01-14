#!/usr/bin/env python3
"""
EIMAS Alert Manager
===================
시그널 파이프라인 ↔ 알림 시스템 통합

주요 기능:
1. 시그널 기반 자동 알림
2. 리스크 임계값 알림
3. 리밸런싱 알림
4. 일일 요약 알림

Usage:
    from lib.alert_manager import AlertManager

    am = AlertManager()
    am.process_signals(signals)
    am.send_daily_summary()
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from lib.notifier import TelegramNotifier, AlertLevel, EIMASNotifier
from lib.trading_db import TradingDB, Signal, SignalAction


# ============================================================================
# Constants
# ============================================================================

# 알림 임계값
ALERT_THRESHOLDS = {
    'conviction_high': 0.7,      # 높은 확신 시그널
    'conviction_critical': 0.9,  # 매우 높은 확신
    'vix_spike': 25,             # VIX 급등
    'vix_extreme': 35,           # VIX 극단
    'drawdown_warning': 0.05,    # 5% 낙폭 경고
    'drawdown_critical': 0.10,   # 10% 낙폭 위험
    'drift_warning': 0.05,       # 5% 비중 이탈
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AlertEvent:
    """알림 이벤트"""
    timestamp: datetime
    level: AlertLevel
    category: str  # 'signal', 'risk', 'rebalance', 'market'
    title: str
    message: str
    data: Dict[str, Any] = None
    sent: bool = False


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """알림 관리자"""

    def __init__(self, db: TradingDB = None):
        self.db = db or TradingDB()
        self.notifier = EIMASNotifier()
        self.events: List[AlertEvent] = []

    def _create_event(
        self,
        level: AlertLevel,
        category: str,
        title: str,
        message: str,
        data: Dict = None
    ) -> AlertEvent:
        """알림 이벤트 생성"""
        event = AlertEvent(
            timestamp=datetime.now(),
            level=level,
            category=category,
            title=title,
            message=message,
            data=data or {},
        )
        self.events.append(event)
        return event

    def _send_alert(self, event: AlertEvent) -> bool:
        """알림 전송"""
        try:
            success = self.notifier.send_alert(
                title=event.title,
                message=event.message,
                level=event.level,
            )
            event.sent = success
            return success
        except Exception as e:
            print(f"Alert send failed: {e}")
            return False

    # ========================================================================
    # Signal Alerts
    # ========================================================================

    def process_signals(self, signals: List[Signal]) -> List[AlertEvent]:
        """시그널 처리 및 알림"""
        events = []

        for signal in signals:
            # 높은 확신 시그널만 알림
            if signal.conviction >= ALERT_THRESHOLDS['conviction_critical']:
                level = AlertLevel.CRITICAL
            elif signal.conviction >= ALERT_THRESHOLDS['conviction_high']:
                level = AlertLevel.WARNING
            else:
                continue  # 낮은 확신은 알림 안 함

            # 액션별 메시지
            action_emoji = {
                SignalAction.BUY: "🟢",
                SignalAction.SELL: "🔴",
                SignalAction.HOLD: "🟡",
                SignalAction.REDUCE: "🟠",
                SignalAction.HEDGE: "🛡️",
            }

            emoji = action_emoji.get(signal.action, "📊")
            title = f"{emoji} {signal.action.value.upper()} Signal"

            message = f"""
Source: {signal.source.value}
Ticker: {signal.ticker}
Conviction: {signal.conviction:.0%}
Reason: {signal.reasoning}
Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
""".strip()

            event = self._create_event(
                level=level,
                category='signal',
                title=title,
                message=message,
                data={'signal_id': signal.id, 'source': signal.source.value}
            )
            events.append(event)

        return events

    def alert_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float
    ) -> AlertEvent:
        """레짐 변화 알림"""
        # 위험 레짐으로 변화 시 높은 레벨
        danger_regimes = ['bear_high_vol', 'transition']
        if new_regime.lower().replace(' ', '_') in danger_regimes:
            level = AlertLevel.CRITICAL
        else:
            level = AlertLevel.INFO

        title = "🔄 Regime Change Detected"
        message = f"""
Previous: {old_regime}
Current: {new_regime}
Confidence: {confidence:.0%}

Action may be required.
""".strip()

        event = self._create_event(
            level=level,
            category='market',
            title=title,
            message=message,
            data={'old_regime': old_regime, 'new_regime': new_regime}
        )
        return event

    # ========================================================================
    # Risk Alerts
    # ========================================================================

    def alert_risk_warning(
        self,
        risk_type: str,
        current_value: float,
        threshold: float,
        message: str = ""
    ) -> AlertEvent:
        """리스크 경고 알림"""
        if current_value > threshold * 1.5:
            level = AlertLevel.CRITICAL
        else:
            level = AlertLevel.WARNING

        title = f"⚠️ Risk Alert: {risk_type}"
        alert_msg = f"""
Type: {risk_type}
Current: {current_value:.2f}
Threshold: {threshold:.2f}
{message}
""".strip()

        event = self._create_event(
            level=level,
            category='risk',
            title=title,
            message=alert_msg,
            data={'risk_type': risk_type, 'value': current_value}
        )
        return event

    def alert_drawdown(
        self,
        current_dd: float,
        max_dd_limit: float
    ) -> Optional[AlertEvent]:
        """낙폭 알림"""
        if current_dd < ALERT_THRESHOLDS['drawdown_warning']:
            return None

        if current_dd >= ALERT_THRESHOLDS['drawdown_critical']:
            level = AlertLevel.CRITICAL
            title = "🚨 CRITICAL Drawdown Alert"
        else:
            level = AlertLevel.WARNING
            title = "⚠️ Drawdown Warning"

        message = f"""
Current Drawdown: {current_dd:.1%}
Max Limit: {max_dd_limit:.1%}
Remaining Buffer: {max_dd_limit - current_dd:.1%}

Consider risk reduction.
""".strip()

        event = self._create_event(
            level=level,
            category='risk',
            title=title,
            message=message,
            data={'drawdown': current_dd}
        )
        return event

    def alert_vix_level(self, vix_value: float) -> Optional[AlertEvent]:
        """VIX 레벨 알림"""
        if vix_value < ALERT_THRESHOLDS['vix_spike']:
            return None

        if vix_value >= ALERT_THRESHOLDS['vix_extreme']:
            level = AlertLevel.CRITICAL
            title = "🔴 EXTREME VIX Alert"
            action = "Market panic detected. Consider contrarian buying."
        else:
            level = AlertLevel.WARNING
            title = "🟠 VIX Spike Alert"
            action = "Elevated volatility. Monitor closely."

        message = f"""
VIX Level: {vix_value:.1f}
Threshold: {ALERT_THRESHOLDS['vix_spike']}
Status: {'EXTREME' if vix_value >= 35 else 'ELEVATED'}

{action}
""".strip()

        event = self._create_event(
            level=level,
            category='market',
            title=title,
            message=message,
            data={'vix': vix_value}
        )
        return event

    # ========================================================================
    # Rebalance Alerts
    # ========================================================================

    def alert_rebalance_needed(
        self,
        trigger: str,
        max_drift: float,
        trades: Dict[str, float]
    ) -> AlertEvent:
        """리밸런싱 필요 알림"""
        level = AlertLevel.WARNING if max_drift < 0.10 else AlertLevel.CRITICAL

        title = "📊 Rebalance Required"

        trade_lines = []
        for ticker, change in sorted(trades.items(), key=lambda x: -abs(x[1])):
            if abs(change) > 0.01:
                action = "BUY" if change > 0 else "SELL"
                trade_lines.append(f"  {ticker}: {action} {abs(change):.1%}")

        message = f"""
Trigger: {trigger}
Max Drift: {max_drift:.1%}

Trades Needed:
{chr(10).join(trade_lines[:5])}
""".strip()

        event = self._create_event(
            level=level,
            category='rebalance',
            title=title,
            message=message,
            data={'trigger': trigger, 'drift': max_drift}
        )
        return event

    # ========================================================================
    # Summary Alerts
    # ========================================================================

    def send_daily_summary(
        self,
        signals_count: int,
        consensus_action: str,
        consensus_conviction: float,
        portfolios: List[Dict] = None,
        risk_level: str = "medium"
    ) -> bool:
        """일일 요약 알림"""
        title = "📈 EIMAS Daily Summary"

        # 포트폴리오 요약
        portfolio_lines = []
        if portfolios:
            for p in portfolios[:3]:
                profile = p.get('profile', 'unknown')
                sharpe = p.get('expected_sharpe', 0)
                portfolio_lines.append(f"  {profile}: Sharpe {sharpe:.2f}")

        message = f"""
Date: {date.today().isoformat()}

📊 Signals: {signals_count}
📌 Consensus: {consensus_action.upper()} ({consensus_conviction:.0%})
⚠️ Risk Level: {risk_level.upper()}

💼 Top Portfolios:
{chr(10).join(portfolio_lines) if portfolio_lines else '  No portfolios generated'}

Generated by EIMAS
""".strip()

        try:
            return self.notifier.send(message)
        except Exception as e:
            print(f"Daily summary send failed: {e}")
            return False

    def send_all_pending(self) -> int:
        """대기 중인 모든 알림 전송"""
        sent_count = 0
        for event in self.events:
            if not event.sent:
                if self._send_alert(event):
                    sent_count += 1
        return sent_count

    def print_events(self):
        """이벤트 목록 출력"""
        print("\n" + "=" * 60)
        print("Alert Events")
        print("=" * 60)

        for event in self.events:
            status = "✓" if event.sent else "○"
            level_emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
            }
            emoji = level_emoji.get(event.level, "📌")

            print(f"\n{status} [{event.category}] {emoji} {event.title}")
            print(f"   {event.timestamp.strftime('%H:%M:%S')} | {event.level.value}")
            for line in event.message.split('\n')[:3]:
                print(f"   {line}")

        print("=" * 60)


# ============================================================================
# Integration with Signal Pipeline
# ============================================================================

def run_alert_pipeline():
    """시그널 파이프라인 + 알림 통합 실행"""
    from lib.signal_pipeline import SignalPipeline, PortfolioGenerator
    from lib.risk_manager import RiskManager

    print("=" * 60)
    print("EIMAS Alert Pipeline")
    print("=" * 60)

    # 시그널 수집
    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()

    # 알림 관리자
    am = AlertManager(pipeline.db)

    # 시그널 알림
    signal_events = am.process_signals(signals)
    print(f"\nSignal alerts: {len(signal_events)}")

    # 포트폴리오 생성
    generator = PortfolioGenerator(pipeline.db)
    portfolios = generator.generate_all_profiles(signals)

    # 리스크 계산
    if portfolios:
        rm = RiskManager()
        # 첫 번째 포트폴리오로 리스크 계산
        holdings = portfolios[0].allocations
        risk = rm.calculate_portfolio_risk(holdings)

        # 리스크 알림
        if risk.max_drawdown > 5:
            am.alert_drawdown(risk.max_drawdown / 100, 0.10)

    # 일일 요약
    am.send_daily_summary(
        signals_count=len(signals),
        consensus_action=consensus['action'],
        consensus_conviction=consensus['conviction'],
        portfolios=[p.to_dict() for p in portfolios],
        risk_level=risk.risk_level.value if portfolios else "unknown"
    )

    # 이벤트 출력
    am.print_events()

    # 알림 전송 (설정 시)
    # sent = am.send_all_pending()
    # print(f"\nAlerts sent: {sent}")

    return am


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    run_alert_pipeline()
