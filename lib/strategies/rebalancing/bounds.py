"""
Dynamic Bounds Engine
=====================
시장 신호 + AI 토론 결과 → 자산배분 경계 동적 조정 엔진

lib/rebalancing_policy.py에서 분리됨 (2026-03-06).
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

class BoundsAdjustmentLog:
    """동적 경계 조정 내역 (감사 추적용)"""
    base_profile: str
    rules_applied: List[str]
    final_bounds: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "base_profile": self.base_profile,
            "rules_applied": self.rules_applied,
            "final_bounds": self.final_bounds,
        }


class DynamicBoundsEngine:
    """
    시장 신호 + AI 토론 결과 → 자산배분 경계 동적 조정 엔진

    조정 단위: 5% (STEP = 0.05)
    모든 규칙은 독립적으로 적용되며 누적됩니다.

    입력 신호:
        debate_signal    : 'BULLISH' | 'BEARISH' | 'NEUTRAL'  (AI 토론 최종 권고)
        confidence       : 0.0~1.0  (AI 토론 신뢰도)
        risk_score       : 0~100    (Phase 2 리스크 점수)
        regime           : 'BULL' | 'BEAR' | 'NEUTRAL'        (GMM 레짐)
        vix              : float    (VIX 지수)
        liquidity_regime : 'Tight' | 'Normal' | 'Loose'       (유동성 레짐)

    규칙 (5% 단위):
        ─ AI 토론 시그널 ─────────────────────────────────────
        BULLISH             → equity_max +5%,  bond_min  -5%
        BEARISH             → equity_max -5%,  bond_min  +5%,  cash_min +5%
        BULLISH & conf>0.70 → equity_max +5%  (추가 보너스)
        BEARISH & conf>0.70 → equity_max -5%  (추가 페널티)

        ─ 리스크 점수 ────────────────────────────────────────
        score > 75          → equity_max -10%, cash_min +10%
        score > 60          → equity_max  -5%, cash_min  +5%
        score < 30          → equity_max  +5%

        ─ 레짐 ──────────────────────────────────────────────
        BULL regime         → equity_max  +5%
        BEAR regime         → equity_max  -5%, bond_min  +5%

        ─ VIX ───────────────────────────────────────────────
        VIX > 40 (극단)     → equity_max -10%, cash_min +10%
        VIX > 30 (고변동)   → equity_max  -5%, cash_min  +5%

        ─ 유동성 ────────────────────────────────────────────
        Tight liquidity     → bond_min    +5%

    최종 보정:
        - 모든 값은 [0.0, 1.0] 클램핑
        - min <= max 보장 (위반 시 min = max - STEP)
    """

    STEP = 0.05

    def compute(
        self,
        base_profile: str = "moderate",
        debate_signal: str = "NEUTRAL",
        confidence: float = 0.5,
        risk_score: float = 50.0,
        regime: str = "NEUTRAL",
        vix: float = 20.0,
        liquidity_regime: str = "Normal",
    ) -> tuple:
        """
        Returns:
            (AssetClassBounds, BoundsAdjustmentLog)
        """
        s = self.STEP
        rules: List[str] = []

        # ── 베이스 프로파일 로드 ──────────────────────────────────────────
        profile_map = {
            "conservative": AssetClassBounds.conservative(),
            "moderate":     AssetClassBounds.moderate(),
            "aggressive":   AssetClassBounds.aggressive(),
        }
        base = profile_map.get(base_profile, AssetClassBounds.moderate())

        eq_min  = base.equity_min
        eq_max  = base.equity_max
        bd_min  = base.bond_min
        bd_max  = base.bond_max
        ca_min  = base.cash_min
        ca_max  = base.cash_max
        co_min  = base.commodity_min
        co_max  = base.commodity_max
        cr_min  = base.crypto_min
        cr_max  = base.crypto_max

        sig = debate_signal.upper() if debate_signal else "NEUTRAL"
        reg = regime.upper() if regime else "NEUTRAL"
        liq = liquidity_regime.lower() if liquidity_regime else "normal"

        # ── AI 토론 시그널 ────────────────────────────────────────────────
        if sig == "BULLISH":
            eq_max += s; bd_min -= s
            rules.append(f"BULLISH → equity_max+{s:.0%}, bond_min-{s:.0%}")
        elif sig == "BEARISH":
            eq_max -= s; bd_min += s; ca_min += s
            rules.append(f"BEARISH → equity_max-{s:.0%}, bond_min+{s:.0%}, cash_min+{s:.0%}")

        # 신뢰도 보너스/페널티
        if confidence > 0.70:
            if sig == "BULLISH":
                eq_max += s
                rules.append(f"BULLISH conf>{0.70:.0%} → equity_max+{s:.0%} (bonus)")
            elif sig == "BEARISH":
                eq_max -= s
                rules.append(f"BEARISH conf>{0.70:.0%} → equity_max-{s:.0%} (penalty)")

        # ── 리스크 점수 ───────────────────────────────────────────────────
        if risk_score > 75:
            eq_max -= 2 * s; ca_min += 2 * s
            rules.append(f"risk>{75} → equity_max-{2*s:.0%}, cash_min+{2*s:.0%}")
        elif risk_score > 60:
            eq_max -= s; ca_min += s
            rules.append(f"risk>{60} → equity_max-{s:.0%}, cash_min+{s:.0%}")
        elif risk_score < 30:
            eq_max += s
            rules.append(f"risk<{30} → equity_max+{s:.0%}")

        # ── GMM 레짐 ─────────────────────────────────────────────────────
        if "BULL" in reg:
            eq_max += s
            rules.append(f"BULL regime → equity_max+{s:.0%}")
        elif "BEAR" in reg:
            eq_max -= s; bd_min += s
            rules.append(f"BEAR regime → equity_max-{s:.0%}, bond_min+{s:.0%}")

        # ── VIX ──────────────────────────────────────────────────────────
        if vix > 40:
            eq_max -= 2 * s; ca_min += 2 * s
            rules.append(f"VIX>{40} → equity_max-{2*s:.0%}, cash_min+{2*s:.0%}")
        elif vix > 30:
            eq_max -= s; ca_min += s
            rules.append(f"VIX>{30} → equity_max-{s:.0%}, cash_min+{s:.0%}")

        # ── 유동성 레짐 ──────────────────────────────────────────────────
        if "tight" in liq:
            bd_min += s
            rules.append(f"Tight liquidity → bond_min+{s:.0%}")

        # ── 보정: 클램핑 + min<=max 보장 ─────────────────────────────────
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, round(v, 4)))

        def _fix_pair(mn: float, mx: float) -> tuple:
            mn, mx = _clamp(mn), _clamp(mx)
            if mn > mx:
                mn = max(0.0, mx - s)
            return mn, mx

        eq_min,  eq_max  = _fix_pair(eq_min,  eq_max)
        bd_min,  bd_max  = _fix_pair(bd_min,  bd_max)
        ca_min,  ca_max  = _fix_pair(ca_min,  ca_max)
        co_min,  co_max  = _fix_pair(co_min,  co_max)
        cr_min,  cr_max  = _fix_pair(cr_min,  cr_max)

        bounds = AssetClassBounds(
            equity_min=eq_min,    equity_max=eq_max,
            bond_min=bd_min,      bond_max=bd_max,
            cash_min=ca_min,      cash_max=ca_max,
            commodity_min=co_min, commodity_max=co_max,
            crypto_min=cr_min,    crypto_max=cr_max,
        )

        log = BoundsAdjustmentLog(
            base_profile=base_profile,
            rules_applied=rules if rules else ["no_adjustment (NEUTRAL + normal signals)"],
            final_bounds=bounds.to_dict(),
        )

        return bounds, log


