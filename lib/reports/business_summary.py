#!/usr/bin/env python3
"""
EIMAS Business Summary Generator
==================================
기술 지표 → 비즈니스 언어 변환 레이어.

기술적 분석 결과(VIX, 레짐, 리스크 점수 등)를
금융 담당자나 고객이 즉시 이해할 수 있는 언어로 변환합니다.

사용:
    from lib.reports.business_summary import generate_business_summary
    summary = generate_business_summary(result.to_dict())
    print(summary["customer_message"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# 비즈니스 요약 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class BusinessSummary:
    """고객/현업 보고용 비즈니스 요약."""

    market_status: str = ""          # 현재 시장 상태 (한 문장)
    risk_assessment: str = ""        # 위험도 평가 (한 문장)
    recommendation: str = ""         # 권고 조치 (한 문장)
    key_risks: List[str] = field(default_factory=list)       # 주요 위험 요인
    action_items: List[str] = field(default_factory=list)    # 구체적 액션 아이템
    customer_message: str = ""       # 고객 전달 메시지 (2~3문장)
    risk_level: str = "MEDIUM"       # HIGH / MEDIUM / LOW
    signal: str = "NEUTRAL"          # BULLISH / BEARISH / NEUTRAL
    data_quality: str = "OK"         # OK / DEGRADED / MISSING

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "## 시장 현황 요약",
            f"- **상태**: {self.market_status}",
            f"- **위험도**: {self.risk_level} — {self.risk_assessment}",
            f"- **권고**: {self.recommendation}",
        ]
        if self.key_risks:
            lines.append("\n**주요 위험 요인**")
            lines.extend(f"  - {r}" for r in self.key_risks)
        if self.action_items:
            lines.append("\n**액션 아이템**")
            lines.extend(f"  - {a}" for a in self.action_items)
        lines.append(f"\n> {self.customer_message}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────

def _get(d: dict, *keys, default=None):
    """중첩 dict에서 안전하게 값 추출."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


_REGIME_LABEL = {
    "BULL": "강세장",
    "BEAR": "약세장",
    "NEUTRAL": "중립장",
    "RISK_ON": "위험선호",
    "RISK_OFF": "위험회피",
    "CRISIS": "위기",
}

_RISK_LEVEL_LABEL = {
    "HIGH": "높음",
    "MEDIUM": "보통",
    "LOW": "낮음",
    "CRITICAL": "매우 높음",
}

_REC_LABEL = {
    "BULLISH":  "주식 비중 확대 검토",
    "BEARISH":  "안전자산 비중 확대 권고",
    "NEUTRAL":  "현재 포지션 유지",
    "HOLD":     "관망",
    "BUY":      "매수 검토",
    "SELL":     "매도 검토",
}


def _classify_vix(vix: float) -> tuple[str, str]:
    """VIX → (위험 수준, 설명)"""
    if vix >= 40:
        return "CRITICAL", f"VIX {vix:.1f} — 극도의 공포 상태"
    if vix >= 30:
        return "HIGH", f"VIX {vix:.1f} — 시장 불안감 고조"
    if vix >= 20:
        return "MEDIUM", f"VIX {vix:.1f} — 보통 수준의 변동성"
    return "LOW", f"VIX {vix:.1f} — 안정적인 시장 환경"


def _classify_risk_score(score: float) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def _build_key_risks(data: dict) -> List[str]:
    """결과 데이터에서 주요 위험 요인 추출."""
    risks: List[str] = []

    # VIX
    vix = _get(data, "market_indicators", "vix_current") or _get(data, "risk_indicators", "vix") or 0.0
    if float(vix) >= 30:
        risks.append(f"VIX {vix:.1f} — 변동성 급등 (시장 공포 구간)")

    # 레짐
    regime_dict = data.get("regime") or {}
    regime = regime_dict.get("regime", "") if isinstance(regime_dict, dict) else str(regime_dict)
    if regime in ("BEAR", "CRISIS", "RISK_OFF"):
        risks.append(f"시장 레짐: {_REGIME_LABEL.get(regime, regime)} — 위험 회피 선호")

    # 리스크 점수
    risk_score = float(data.get("risk_score") or 0)
    if risk_score >= 75:
        risks.append(f"종합 리스크 점수 {risk_score:.0f}/100 — 고위험 구간")

    # 유동성
    liquidity = _get(data, "fred_summary", "liquidity_regime") or _get(data, "liquidity_analysis", "signal") or ""
    if "Tight" in str(liquidity) or "TIGHT" in str(liquidity):
        risks.append("유동성 긴축 — 신용 스프레드 확대 주의")

    # 버블 신호
    bubble = data.get("bubble_alerts") or []
    if bubble:
        risks.append(f"버블 경보 {len(bubble)}건 감지")

    # 저축은행 건전성
    ksb = data.get("korea_savings_bank") or {}
    if isinstance(ksb, dict):
        npl = float(ksb.get("npl_ratio") or 0)
        if npl >= 8.0:
            risks.append(f"저축은행 NPL 비율 {npl:.1f}% — 위험 임계치 초과")

    return risks or ["특이 위험 요인 없음"]


def _build_action_items(recommendation: str, risk_level: str, risks: List[str]) -> List[str]:
    """권고 액션 아이템 생성."""
    actions: List[str] = []

    if recommendation in ("BEARISH", "SELL"):
        actions.append("주식 비중 단계적 축소, 채권·현금 비중 확대")
        actions.append("헤지 포지션 또는 방어적 섹터(필수소비재, 유틸리티) 검토")
    elif recommendation in ("BULLISH", "BUY"):
        actions.append("성장 자산 비중 확대 검토")
        actions.append("섹터 로테이션: 경기 민감주 비중 증가 고려")
    else:
        actions.append("현재 포지션 유지 — 추가 시그널 관찰")

    if risk_level in ("HIGH", "CRITICAL"):
        actions.append("리스크 한도 재점검 및 손절 기준 명확화")

    return actions


def _build_customer_message(
    market_status: str,
    risk_level: str,
    recommendation: str,
    regime: str,
) -> str:
    """고객 전달 메시지 (2~3문장)."""
    risk_kr = _RISK_LEVEL_LABEL.get(risk_level, risk_level)
    rec_kr = _REC_LABEL.get(recommendation, recommendation)
    regime_kr = _REGIME_LABEL.get(regime, regime) if regime else "불확실"

    return (
        f"현재 시장은 {regime_kr} 국면으로, 위험도는 {risk_kr} 수준입니다. "
        f"{market_status} "
        f"이에 따라 {rec_kr}을 권고드립니다."
    )


# ─────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────

def generate_business_summary(result_data: Dict[str, Any]) -> BusinessSummary:
    """
    EIMAS 기술 분석 결과 → 비즈니스 언어 변환.

    Args:
        result_data: EIMASResult.to_dict() 결과

    Returns:
        BusinessSummary: 고객/현업 보고용 요약

    Example:
        from lib.reports.business_summary import generate_business_summary
        summary = generate_business_summary(result.to_dict())
        print(summary.customer_message)
        # → "현재 시장은 약세장 국면으로, 위험도는 높음 수준입니다. ..."
    """
    if not isinstance(result_data, dict):
        return BusinessSummary(
            market_status="데이터 없음",
            data_quality="MISSING",
        )

    # ── 기본값 추출 ──
    recommendation = str(result_data.get("final_recommendation") or "NEUTRAL").upper()
    risk_score = float(result_data.get("risk_score") or 50.0)
    confidence = float(result_data.get("confidence") or 0.5)

    regime_raw = result_data.get("regime") or {}
    regime = regime_raw.get("regime", "NEUTRAL") if isinstance(regime_raw, dict) else str(regime_raw)

    vix_raw = (
        _get(result_data, "market_indicators", "vix_current")
        or _get(result_data, "risk_indicators", "vix")
        or 20.0
    )
    vix = float(vix_raw)

    # ── 위험 수준 결정 (VIX + risk_score 복합) ──
    risk_from_vix, vix_desc = _classify_vix(vix)
    risk_from_score = _classify_risk_score(risk_score)
    risk_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    risk_level = max(risk_from_vix, risk_from_score, key=lambda x: risk_priority.get(x, 0))

    # ── 시장 상태 문장 ──
    regime_kr = _REGIME_LABEL.get(regime, regime) or "불확실"
    rec_kr = _REC_LABEL.get(recommendation, recommendation)
    conf_pct = int(confidence * 100)

    market_status = (
        f"{vix_desc}, "
        f"레짐 '{regime_kr}', "
        f"리스크 점수 {risk_score:.0f}/100 — "
        f"신뢰도 {conf_pct}%로 분석 완료."
    )

    risk_assessment = f"종합 위험도 {_RISK_LEVEL_LABEL.get(risk_level, risk_level)}"

    # ── 위험 요인 및 액션 ──
    key_risks = _build_key_risks(result_data)
    action_items = _build_action_items(recommendation, risk_level, key_risks)

    # ── 고객 메시지 ──
    customer_message = _build_customer_message(market_status, risk_level, recommendation, regime)

    return BusinessSummary(
        market_status=market_status,
        risk_assessment=risk_assessment,
        recommendation=rec_kr,
        key_risks=key_risks,
        action_items=action_items,
        customer_message=customer_message,
        risk_level=risk_level,
        signal=recommendation,
        data_quality="OK",
    )
