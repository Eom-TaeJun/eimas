#!/usr/bin/env python3
"""
Genius Act - Main Strategy
============================================================

Main Genius Act macro strategy engine

Economic Foundation:
    - Combines liquidity analysis + decision rules
    - Fed liquidity-driven asset allocation

Class:
    - GeniusActMacroStrategy: Main strategy class
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .schemas import MacroSignal, LiquidityIndicators, StrategyPosition
from .enums import LiquidityRegime, SignalType
from .liquidity import ExtendedLiquidityModel
from .rules import GeniusActRules

logger = logging.getLogger(__name__)


class GeniusActMacroStrategy:
    """Genius Act 기반 매크로 전략"""

    def __init__(self):
        self.liquidity_model = ExtendedLiquidityModel()
        self.rules = GeniusActRules()
        self.signal_history: List[MacroSignal] = []
        self.positions: List[StrategyPosition] = []

    def analyze(
        self,
        current: LiquidityIndicators,
        previous: LiquidityIndicators,
        treasury_supply_change: float = 0
    ) -> Dict:
        """전체 분석 실행"""

        # 1. 유동성 계산
        liquidity = self.liquidity_model.calculate_total_liquidity(current)

        # 2. 시그널 생성
        signals = []
        signals.extend(self.rules.check_stablecoin_signals(current, previous))
        signals.extend(self.rules.check_fed_liquidity_signals(current, previous))
        signals.extend(self.rules.check_treasury_signals(current, previous, treasury_supply_change))

        # 히스토리에 저장
        self.signal_history.extend(signals)

        # 3. 레짐 판단
        regime = self._determine_regime(liquidity, signals)

        # 4. 포지션 추천
        positions = self._generate_positions(signals, regime)

        return {
            "timestamp": current.timestamp.isoformat(),
            "liquidity": liquidity,
            "regime": regime.value,
            "signals": [self._signal_to_dict(s) for s in signals],
            "positions": [self._position_to_dict(p) for p in positions],
            "summary": self._generate_summary(liquidity, regime, signals, positions)
        }

    def _determine_regime(
        self,
        liquidity: Dict,
        signals: List[MacroSignal]
    ) -> LiquidityRegime:
        """유동성 레짐 판단"""

        # 시그널 기반 점수
        expansion_score = 0
        contraction_score = 0

        for signal in signals:
            if signal.signal_type in [
                SignalType.LIQUIDITY_INJECTION,
                SignalType.RRP_DRAIN,
                SignalType.TGA_DRAIN,
                SignalType.STABLECOIN_SURGE,
                SignalType.CRYPTO_RISK_ON
            ]:
                expansion_score += signal.strength * signal.confidence

            elif signal.signal_type in [
                SignalType.LIQUIDITY_DRAIN,
                SignalType.STABLECOIN_DRAIN,
                SignalType.CRYPTO_RISK_OFF,
                SignalType.TREASURY_SUPPLY
            ]:
                contraction_score += signal.strength * signal.confidence

        # 레짐 결정
        net_score = expansion_score - contraction_score

        if net_score > 0.5:
            return LiquidityRegime.EXPANSION
        elif net_score < -0.5:
            return LiquidityRegime.CONTRACTION
        elif abs(net_score) < 0.2:
            return LiquidityRegime.NEUTRAL
        else:
            return LiquidityRegime.TRANSITION

    def _generate_positions(
        self,
        signals: List[MacroSignal],
        regime: LiquidityRegime
    ) -> List[StrategyPosition]:
        """시그널 기반 포지션 생성"""
        positions = []

        # 레짐별 기본 포지션
        if regime == LiquidityRegime.EXPANSION:
            positions.append(StrategyPosition(
                asset="SPY",
                direction="LONG",
                size=0.3,
                entry_signal=SignalType.LIQUIDITY_INJECTION,
                rationale="유동성 확장 레짐 → 주식 강세"
            ))
            positions.append(StrategyPosition(
                asset="BTC-USD",
                direction="LONG",
                size=0.1,
                entry_signal=SignalType.LIQUIDITY_INJECTION,
                rationale="유동성 확장 → 위험자산 선호"
            ))

        elif regime == LiquidityRegime.CONTRACTION:
            positions.append(StrategyPosition(
                asset="TLT",
                direction="LONG",
                size=0.3,
                entry_signal=SignalType.LIQUIDITY_DRAIN,
                rationale="유동성 수축 → 안전자산 선호"
            ))
            positions.append(StrategyPosition(
                asset="SPY",
                direction="SHORT",
                size=0.1,
                entry_signal=SignalType.LIQUIDITY_DRAIN,
                rationale="유동성 수축 → 주식 약세"
            ))

        # 시그널별 추가 포지션
        for signal in signals:
            if signal.signal_type == SignalType.STABLECOIN_SURGE:
                positions.append(StrategyPosition(
                    asset="BTC-USD",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="스테이블코인 유입 → 크립토 매수 대기"
                ))
                positions.append(StrategyPosition(
                    asset="SHY",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="스테이블코인 담보 요건 → 단기 국채 수요"
                ))

            elif signal.signal_type == SignalType.RRP_DRAIN:
                positions.append(StrategyPosition(
                    asset="QQQ",
                    direction="LONG",
                    size=0.1,
                    entry_signal=signal.signal_type,
                    rationale="역레포 유동성 방출 → 성장주 강세"
                ))

            elif signal.signal_type == SignalType.TREASURY_SUPPLY:
                positions.append(StrategyPosition(
                    asset="TBT",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="국채 공급 증가 → 금리 상승 베팅"
                ))

        # 중복 제거 및 크기 조정
        return self._consolidate_positions(positions)

    def _consolidate_positions(
        self,
        positions: List[StrategyPosition]
    ) -> List[StrategyPosition]:
        """포지션 통합 및 정규화"""
        consolidated = {}

        for pos in positions:
            key = (pos.asset, pos.direction)
            if key in consolidated:
                existing = consolidated[key]
                existing.size += pos.size
                existing.rationale += f"; {pos.rationale}"
            else:
                consolidated[key] = pos

        result = list(consolidated.values())

        # 비중 정규화 (총합 1.0 이하)
        total = sum(p.size for p in result)
        if total > 1.0:
            for p in result:
                p.size /= total

        return result

    def _signal_to_dict(self, signal: MacroSignal) -> Dict:
        """시그널을 딕셔너리로 변환"""
        return {
            "type": signal.signal_type.value,
            "strength": f"{signal.strength:.2f}",
            "description": signal.description,
            "affected_assets": signal.affected_assets,
            "confidence": f"{signal.confidence*100:.0f}%",
            "metadata": signal.metadata
        }

    def _position_to_dict(self, position: StrategyPosition) -> Dict:
        """포지션을 딕셔너리로 변환"""
        return {
            "asset": position.asset,
            "direction": position.direction,
            "size": f"{position.size*100:.1f}%",
            "signal": position.entry_signal.value,
            "rationale": position.rationale
        }

    def _generate_summary(
        self,
        liquidity: Dict,
        regime: LiquidityRegime,
        signals: List[MacroSignal],
        positions: List[StrategyPosition]
    ) -> str:
        """분석 요약 생성"""
        summary_parts = []

        # 유동성 상태
        summary_parts.append(f"[유동성] {liquidity['formula']}")

        # 레짐
        regime_desc = {
            LiquidityRegime.EXPANSION: "확장 (Risk-On)",
            LiquidityRegime.CONTRACTION: "수축 (Risk-Off)",
            LiquidityRegime.NEUTRAL: "중립",
            LiquidityRegime.TRANSITION: "전환기 (주의)"
        }
        summary_parts.append(f"[레짐] {regime_desc[regime]}")

        # 핵심 시그널
        if signals:
            high_conf_signals = [s for s in signals if s.confidence > 0.7]
            if high_conf_signals:
                summary_parts.append(f"[시그널] {len(high_conf_signals)}개 고신뢰 시그널")

        # 포지션 추천
        if positions:
            longs = [p for p in positions if p.direction == "LONG"]
            shorts = [p for p in positions if p.direction == "SHORT"]
            summary_parts.append(f"[포지션] LONG {len(longs)}개, SHORT {len(shorts)}개")

        return " | ".join(summary_parts)


# =============================================================================
# 크립토 리스크 평가 (Genius Act 담보 유형별 차등화)
# =============================================================================

