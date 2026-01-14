#!/usr/bin/env python3
"""
Signal-Action Framework Integration Test
==========================================
기존 Market Anomaly Detector의 Signal과 연동 테스트
"""

import sys
import os

# 경로 설정 (절대 경로 사용)
EIMAS_DIR = '/home/tj/projects/autoai/eimas'
AUTOAI_DIR = '/home/tj/projects/autoai'
if EIMAS_DIR not in sys.path:
    sys.path.insert(0, EIMAS_DIR)
if AUTOAI_DIR not in sys.path:
    sys.path.insert(0, AUTOAI_DIR)

from core.signal_action import (
    SignalActionMapper,
    RiskProfile,
    EnhancedSignal,
    PositionDirection,
    MarketRegime,
    RiskProfileType,
)

# 기존 detectors.py의 Signal 클래스 임포트 시도
try:
    from detectors import Signal, SignalLevel
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False
    print("Note: detectors.py not found, using mock signals")


def create_mock_legacy_signals():
    """기존 Signal 형식의 모의 데이터 생성"""

    class MockSignal:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    return [
        MockSignal(
            type="statistical",
            ticker="SPY",
            name="S&P 500 ETF",
            indicator="price_z",
            value=452.30,
            threshold=2.0,
            z_score=2.45,
            level="ALERT",
            description="SPY 가격 급등 (Z=2.45)"
        ),
        MockSignal(
            type="statistical",
            ticker="GLD",
            name="Gold ETF",
            indicator="rsi",
            value=83.6,
            threshold=70,
            z_score=2.13,
            level="ALERT",
            description="GLD RSI 과매수 (RSI=83.6)"
        ),
        MockSignal(
            type="statistical",
            ticker="USD/CNY",
            name="달러/위안",
            indicator="return_z",
            value=-1.2,
            threshold=3.0,
            z_score=-3.16,
            level="CRITICAL",
            description="USD/CNY 일일 수익률 급락 (Z=-3.16)"
        ),
        MockSignal(
            type="statistical",
            ticker="Copper",
            name="구리",
            indicator="price_z",
            value=4.25,
            threshold=3.0,
            z_score=3.05,
            level="CRITICAL",
            description="구리 가격 급등 (Z=3.05)"
        ),
        MockSignal(
            type="theoretical",
            ticker="HYG",
            name="High Yield Bond ETF",
            indicator="hy_spread",
            value=4.5,
            threshold=4.0,
            z_score=1.8,
            level="WARNING",
            description="HY 스프레드 확대 경고"
        ),
    ]


def convert_legacy_to_enhanced(legacy_signals, vix_level=22.5):
    """
    기존 Signal을 EnhancedSignal로 변환

    변환 로직:
    1. Z-score 부호로 방향 결정
    2. level로 초기 신뢰도 추정
    3. 지표 유형으로 horizon 추정
    """
    enhanced = []

    level_to_confidence = {
        "NORMAL": 0.45,
        "WARNING": 0.55,
        "ALERT": 0.70,
        "CRITICAL": 0.85
    }

    indicator_to_horizon = {
        "price_z": "short",
        "return_z": "ultra_short",
        "rsi": "short",
        "volume": "ultra_short",
        "hy_spread": "long",
        "vix": "short"
    }

    for sig in legacy_signals:
        # Z-score로 방향 결정
        z = getattr(sig, 'z_score', 0)
        indicator = getattr(sig, 'indicator', '')

        # RSI는 특수 처리: 과매수면 SHORT, 과매도면 LONG
        if indicator == 'rsi':
            value = getattr(sig, 'value', 50)
            direction = PositionDirection.SHORT if value > 50 else PositionDirection.LONG
        else:
            if z > 0:
                direction = PositionDirection.LONG
            elif z < 0:
                direction = PositionDirection.SHORT
            else:
                direction = PositionDirection.NEUTRAL

        # 신뢰도
        level = getattr(sig, 'level', 'NORMAL')
        base_confidence = level_to_confidence.get(level, 0.5)

        # Z-score 절대값이 클수록 신뢰도 보정
        z_boost = min(abs(z) * 0.05, 0.15)  # 최대 15% 보정
        confidence = min(base_confidence + z_boost, 0.95)

        # Horizon
        horizon = indicator_to_horizon.get(indicator, 'short')

        enhanced.append(EnhancedSignal(
            signal_id="",
            type=getattr(sig, 'type', 'unknown'),
            ticker=getattr(sig, 'ticker', 'UNKNOWN'),
            name=getattr(sig, 'name', 'Unknown'),
            indicator=indicator,
            value=getattr(sig, 'value', 0.0),
            threshold=getattr(sig, 'threshold', 0.0),
            z_score=z,
            level=level,
            description=getattr(sig, 'description', ''),
            confidence=confidence,
            direction=direction,
            horizon=horizon,
            source="market_anomaly_detector"
        ))

    return enhanced


def test_profile_comparison():
    """프로파일별 결과 비교 테스트"""
    print("\n" + "=" * 70)
    print("Profile Comparison Test")
    print("=" * 70)

    # 동일한 신호 세트
    legacy_signals = create_mock_legacy_signals()
    enhanced_signals = convert_legacy_to_enhanced(legacy_signals)

    profiles = [
        ("Conservative", RiskProfile.conservative()),
        ("Moderate", RiskProfile.moderate()),
        ("Aggressive", RiskProfile.aggressive())
    ]

    print(f"\nInput: {len(enhanced_signals)} signals")
    print("-" * 70)

    for name, profile in profiles:
        mapper = SignalActionMapper(risk_profile=profile)
        mapper.set_regime(vix=22.5)  # NORMAL

        actions = mapper.process_signals(enhanced_signals)

        print(f"\n[{name}]")
        print(f"  Signal Threshold: {profile.signal_threshold:.0%}")
        print(f"  Actions Generated: {len(actions)}")

        for action in actions:
            print(f"    - {action.ticker}: {action.action_type.value}, "
                  f"Size: {action.position_size:.1%}, "
                  f"Direction: {action.direction.value}")


def test_regime_impact():
    """레짐별 결과 비교 테스트"""
    print("\n" + "=" * 70)
    print("Regime Impact Test")
    print("=" * 70)

    legacy_signals = create_mock_legacy_signals()
    enhanced_signals = convert_legacy_to_enhanced(legacy_signals)

    profile = RiskProfile.moderate()

    # SPY 신호만 사용 (일관된 비교를 위해)
    spy_signal = [s for s in enhanced_signals if s.ticker == "SPY"][0]

    regimes = [
        ("CALM", 12.0),
        ("NORMAL", 22.0),
        ("ELEVATED", 30.0),
        ("CRISIS", 40.0)
    ]

    print(f"\nTest Signal: SPY, Confidence: {spy_signal.confidence:.1%}")
    print("-" * 70)

    for regime_name, vix in regimes:
        mapper = SignalActionMapper(risk_profile=profile)
        mapper.set_regime(vix=vix)

        actions = mapper.process_signals([spy_signal])

        if actions:
            action = actions[0]
            print(f"  VIX {vix:5.1f} ({regime_name:8s}): "
                  f"Position Size = {action.position_size:.1%}")
        else:
            print(f"  VIX {vix:5.1f} ({regime_name:8s}): No Action")


def test_conflict_detection():
    """신호 충돌 탐지 테스트"""
    print("\n" + "=" * 70)
    print("Conflict Detection Test")
    print("=" * 70)

    # 충돌하는 신호 생성
    conflicting_signals = [
        EnhancedSignal(
            signal_id="",
            type="statistical",
            ticker="SPY",
            name="S&P 500 ETF",
            indicator="price_z",
            value=450,
            threshold=2.0,
            z_score=2.5,
            level="ALERT",
            description="SPY 가격 급등",
            confidence=0.75,
            direction=PositionDirection.LONG,
            horizon="short"
        ),
        EnhancedSignal(
            signal_id="",
            type="theoretical",
            ticker="SPY",
            name="S&P 500 ETF",
            indicator="rsi",
            value=78,
            threshold=70,
            z_score=2.1,
            level="ALERT",
            description="SPY RSI 과매수",
            confidence=0.72,
            direction=PositionDirection.SHORT,  # 반대 방향!
            horizon="short"  # 같은 horizon
        ),
    ]

    profile = RiskProfile.moderate()
    mapper = SignalActionMapper(risk_profile=profile)
    mapper.set_regime(vix=20)

    actions = mapper.process_signals(conflicting_signals)

    print(f"\nInput: 2 conflicting signals (SPY LONG vs SHORT, same horizon)")
    print("-" * 70)
    print(f"Actions Generated: {len(actions)}")
    print(f"Conflicts Detected: {len(mapper.conflicts)}")

    for conflict in mapper.conflicts:
        print(f"\n  Conflict: {conflict.conflict_type.value}")
        print(f"  Description: {conflict.description}")
        print(f"  Resolution: {conflict.resolution}")


def test_action_log():
    """액션 로그 테스트"""
    print("\n" + "=" * 70)
    print("Action Log Test")
    print("=" * 70)

    from core.signal_action import ActionLog

    profile = RiskProfile.moderate()
    mapper = SignalActionMapper(risk_profile=profile)
    mapper.set_regime(vix=18)

    signals = [
        EnhancedSignal(
            signal_id="",
            type="statistical",
            ticker="AAPL",
            name="Apple Inc.",
            indicator="price_z",
            value=195.0,
            threshold=2.0,
            z_score=2.3,
            level="ALERT",
            description="AAPL 가격 급등",
            confidence=0.72,
            direction=PositionDirection.LONG,
            horizon="short"
        )
    ]

    actions = mapper.process_signals(signals)

    if actions and mapper.action_logs:
        log = mapper.action_logs[0]

        # 실행 시뮬레이션
        log.executed_at = "2024-12-30T10:30:00"
        log.executed_price = 195.50
        log.executed_size = 10000
        log.execution_status = "executed"

        print("\n[Action Executed]")
        print(f"  Ticker: {log.action.ticker}")
        print(f"  Entry Price: ${log.executed_price:.2f}")
        print(f"  Size: ${log.executed_size:,.0f}")

        # 추적 업데이트
        log.update_tracking(current_price=198.00)
        print(f"\n[Tracking Update]")
        print(f"  Current Price: ${log.current_price:.2f}")
        print(f"  Unrealized P&L: {log.unrealized_pnl_pct:.2%}")

        # 청산 시뮬레이션
        log.close_position(close_price=200.00, lessons="Momentum play worked well")
        print(f"\n[Position Closed]")
        print(f"  Exit Price: ${log.closed_price:.2f}")
        print(f"  Realized P&L: {log.realized_pnl_pct:.2%}")
        print(f"  Signal Accurate: {log.signal_accuracy}")


def main():
    """메인 테스트 실행"""
    print("=" * 70)
    print("Signal-Action Framework Integration Test Suite")
    print("=" * 70)

    # 1. 기본 변환 테스트
    print("\n[1] Legacy Signal Conversion")
    print("-" * 70)
    legacy_signals = create_mock_legacy_signals()
    enhanced_signals = convert_legacy_to_enhanced(legacy_signals)

    for i, (legacy, enhanced) in enumerate(zip(legacy_signals, enhanced_signals), 1):
        print(f"  {i}. {legacy.ticker}: "
              f"Z={legacy.z_score:+.2f} → "
              f"Direction={enhanced.direction.value}, "
              f"Confidence={enhanced.confidence:.0%}")

    # 2. 프로파일 비교
    test_profile_comparison()

    # 3. 레짐 영향
    test_regime_impact()

    # 4. 충돌 탐지
    test_conflict_detection()

    # 5. 액션 로그
    test_action_log()

    print("\n" + "=" * 70)
    print("All Tests Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
