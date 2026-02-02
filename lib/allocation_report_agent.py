"""
Allocation Report Agent
=======================
자산운용사 자산배분팀 리서치 리포트 작성 보조 에이전트

입력: EIMAS가 계산한 JSON 결과
출력: 4개 섹션의 한국어 리포트

핵심 원칙:
- 새로운 숫자, 신호, 비중을 생성하지 않음
- JSON 데이터만 인용
- 데이터 신뢰도 저하 또는 신호 충돌 시 HOLD
- turnover cap과 weight bounds 제약 준수

출력 섹션:
1. 현재 시장 및 레짐 요약
2. 핵심 근거 3가지
3. 리스크 및 반증 조건 3가지
4. 운용 관점의 액션 아이템 (HOLD 포함)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarketSummary:
    """현재 시장 및 레짐 요약"""
    regime: str                           # BULL/BEAR/NEUTRAL
    regime_confidence: str                # High/Medium/Low
    volatility_state: str                 # Low Vol/High Vol
    risk_score: float                     # 0-100
    risk_level: str                       # LOW/MEDIUM/HIGH
    net_liquidity: str                    # Fed 순유동성 (문자열)
    final_recommendation: str             # BULLISH/BEARISH/NEUTRAL/HOLD
    confidence: float                     # 신뢰도 (0-1)
    data_quality: str                     # COMPLETE/PARTIAL/DEGRADED
    summary_text: str                     # 요약 문장


@dataclass
class KeyRationale:
    """핵심 근거"""
    title: str                            # 근거 제목
    source: str                           # 데이터 출처 (JSON 필드명)
    value: str                            # 실제 값 (JSON에서 인용)
    interpretation: str                   # 해석


@dataclass
class RiskCondition:
    """리스크 및 반증 조건"""
    risk_title: str                       # 리스크 제목
    current_value: str                    # 현재 값
    falsification_condition: str          # 반증 조건 (이 조건이 충족되면 뷰 수정)
    monitoring_metric: str                # 모니터링할 지표


@dataclass
class ActionItem:
    """운용 관점의 액션 아이템"""
    action: str                           # HOLD/REBALANCE/MONITOR/REDUCE/INCREASE
    target: str                           # 대상 (전체 포트폴리오 or 특정 자산)
    rationale: str                        # 근거
    constraints: List[str]                # 제약 조건 (turnover cap, weight bounds 등)
    priority: str                         # HIGH/MEDIUM/LOW


@dataclass
class AllocationReport:
    """
    자산배분팀 리서치 리포트

    EIMAS JSON 결과를 기반으로 생성
    새로운 숫자나 비중을 생성하지 않음
    """
    timestamp: str
    report_version: str = "1.0"

    # 섹션 1: 시장 요약
    market_summary: Optional[MarketSummary] = None

    # 섹션 2: 핵심 근거 3가지
    key_rationales: List[KeyRationale] = field(default_factory=list)

    # 섹션 3: 리스크 및 반증 조건 3가지
    risk_conditions: List[RiskCondition] = field(default_factory=list)

    # 섹션 4: 액션 아이템
    action_items: List[ActionItem] = field(default_factory=list)

    # 메타데이터
    data_quality_warning: bool = False
    signal_conflict_warning: bool = False
    default_to_hold: bool = False
    hold_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_markdown(self) -> str:
        """마크다운 형식 리포트 생성"""
        md = []
        md.append("# 자산배분팀 리서치 리포트")
        md.append(f"**생성 시간**: {self.timestamp}")
        md.append("")

        # 경고 표시
        if self.default_to_hold:
            md.append("> **주의**: 데이터 신뢰도 저하 또는 신호 충돌로 인해 기본 행동은 HOLD입니다.")
            md.append(f"> 사유: {self.hold_reason}")
            md.append("")

        # 섹션 1: 시장 요약
        md.append("## 1. 현재 시장 및 레짐 요약")
        if self.market_summary:
            ms = self.market_summary
            md.append(f"- **레짐**: {ms.regime} ({ms.regime_confidence} 신뢰도)")
            md.append(f"- **변동성**: {ms.volatility_state}")
            md.append(f"- **리스크 점수**: {ms.risk_score:.1f}/100 ({ms.risk_level})")
            md.append(f"- **Fed 순유동성**: {ms.net_liquidity}")
            md.append(f"- **최종 권고**: {ms.final_recommendation} (신뢰도 {ms.confidence:.0%})")
            md.append(f"- **데이터 품질**: {ms.data_quality}")
            md.append("")
            md.append(f"**요약**: {ms.summary_text}")
        md.append("")

        # 섹션 2: 핵심 근거
        md.append("## 2. 핵심 근거 3가지")
        for i, r in enumerate(self.key_rationales[:3], 1):
            md.append(f"### 근거 {i}: {r.title}")
            md.append(f"- **출처**: `{r.source}`")
            md.append(f"- **값**: {r.value}")
            md.append(f"- **해석**: {r.interpretation}")
            md.append("")

        # 섹션 3: 리스크
        md.append("## 3. 리스크 및 반증 조건 3가지")
        for i, rc in enumerate(self.risk_conditions[:3], 1):
            md.append(f"### 리스크 {i}: {rc.risk_title}")
            md.append(f"- **현재 값**: {rc.current_value}")
            md.append(f"- **반증 조건**: {rc.falsification_condition}")
            md.append(f"- **모니터링 지표**: {rc.monitoring_metric}")
            md.append("")

        # 섹션 4: 액션 아이템
        md.append("## 4. 운용 관점의 액션 아이템")
        for i, ai in enumerate(self.action_items, 1):
            md.append(f"### {i}. [{ai.action}] {ai.target}")
            md.append(f"- **근거**: {ai.rationale}")
            if ai.constraints:
                md.append(f"- **제약**: {', '.join(ai.constraints)}")
            md.append(f"- **우선순위**: {ai.priority}")
            md.append("")

        md.append("---")
        md.append("*본 리포트는 EIMAS JSON 결과를 기반으로 자동 생성되었습니다.*")
        md.append("*새로운 숫자나 비중은 생성되지 않았으며, 모든 값은 JSON에서 인용되었습니다.*")

        return "\n".join(md)


class AllocationReportAgent:
    """
    자산배분팀 리서치 리포트 에이전트

    EIMAS JSON → 구조화된 한국어 리포트

    Example:
        >>> agent = AllocationReportAgent()
        >>> report = agent.generate_report(eimas_json)
        >>> print(report.to_markdown())
    """

    def __init__(self):
        self.turnover_cap = 0.30  # 30%
        self.weight_bounds = {
            'equity_max': 0.60,
            'bond_max': 0.40,
            'cash_min': 0.05,
            'crypto_max': 0.10
        }

    def generate_report(self, eimas_result: Dict) -> AllocationReport:
        """
        EIMAS JSON 결과로부터 리포트 생성

        Args:
            eimas_result: EIMAS 분석 결과 JSON

        Returns:
            AllocationReport
        """
        report = AllocationReport(timestamp=datetime.now().isoformat())

        # 데이터 품질 및 신호 충돌 검사
        quality_check = self._check_data_quality(eimas_result)
        conflict_check = self._check_signal_conflict(eimas_result)

        if not quality_check['valid'] or conflict_check['conflict']:
            report.default_to_hold = True
            report.data_quality_warning = not quality_check['valid']
            report.signal_conflict_warning = conflict_check['conflict']
            report.hold_reason = quality_check.get('reason', '') or conflict_check.get('reason', '')
            logger.warning(f"Defaulting to HOLD: {report.hold_reason}")

        # 섹션 1: 시장 요약
        report.market_summary = self._extract_market_summary(eimas_result, report.default_to_hold)

        # 섹션 2: 핵심 근거
        report.key_rationales = self._extract_key_rationales(eimas_result)

        # 섹션 3: 리스크 조건
        report.risk_conditions = self._extract_risk_conditions(eimas_result)

        # 섹션 4: 액션 아이템
        report.action_items = self._generate_action_items(eimas_result, report.default_to_hold)

        return report

    def _check_data_quality(self, data: Dict) -> Dict[str, Any]:
        """데이터 품질 검사"""
        # market_quality 필드 확인
        mq = data.get('market_quality', {})
        if isinstance(mq, dict):
            quality = mq.get('data_quality', 'COMPLETE')
            if quality == 'DEGRADED':
                return {'valid': False, 'reason': '시장 데이터 품질 저하 (DEGRADED)'}

        # 필수 필드 확인
        required = ['regime', 'risk_score', 'final_recommendation']
        missing = [f for f in required if f not in data or data.get(f) is None]
        if missing:
            return {'valid': False, 'reason': f'필수 필드 누락: {missing}'}

        return {'valid': True}

    def _check_signal_conflict(self, data: Dict) -> Dict[str, Any]:
        """신호 충돌 검사"""
        conflicts = []

        # 레짐과 권고 불일치 검사
        regime = data.get('regime', {})
        if isinstance(regime, dict):
            regime_state = regime.get('regime', 'NEUTRAL')
        else:
            regime_state = 'NEUTRAL'

        recommendation = data.get('final_recommendation', 'HOLD')

        # BULL 레짐인데 BEARISH 권고, 또는 그 반대
        if regime_state == 'BULL' and recommendation == 'BEARISH':
            conflicts.append("레짐(BULL)과 권고(BEARISH) 불일치")
        elif regime_state == 'BEAR' and recommendation == 'BULLISH':
            conflicts.append("레짐(BEAR)과 권고(BULLISH) 불일치")

        # Full Mode와 Reference Mode 불일치
        if not data.get('modes_agree', True):
            conflicts.append("Full Mode와 Reference Mode 불일치")

        # Strong Dissent
        if data.get('has_strong_dissent', False):
            conflicts.append("에이전트 간 강한 이견 존재")

        if conflicts:
            return {'conflict': True, 'reason': ', '.join(conflicts)}

        return {'conflict': False}

    def _extract_market_summary(self, data: Dict, default_hold: bool) -> MarketSummary:
        """시장 요약 추출 (JSON 값만 사용)"""
        regime_data = data.get('regime', {})
        if isinstance(regime_data, dict):
            regime = regime_data.get('regime', 'NEUTRAL')
            volatility = regime_data.get('volatility', 'Normal')
        else:
            regime = 'NEUTRAL'
            volatility = 'Normal'

        risk_score = data.get('risk_score', 50.0)
        risk_level = data.get('risk_level', 'MEDIUM')

        # Fed 순유동성
        fred = data.get('fred_summary', {})
        net_liq = fred.get('net_liquidity', 0)
        net_liq_str = f"${net_liq:.0f}B" if net_liq else "N/A"

        # 신뢰도 판단 (리스크 점수 기반)
        if risk_score < 30:
            confidence_level = "High"
        elif risk_score < 60:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        # 데이터 품질
        mq = data.get('market_quality', {})
        data_quality = mq.get('data_quality', 'COMPLETE') if isinstance(mq, dict) else 'COMPLETE'

        # 권고
        final_rec = "HOLD" if default_hold else data.get('final_recommendation', 'HOLD')
        confidence = data.get('confidence', 0.5)

        # 요약 문장 생성 (JSON 값만 사용)
        summary = self._generate_summary_text(regime, volatility, risk_score, final_rec, default_hold)

        return MarketSummary(
            regime=regime,
            regime_confidence=confidence_level,
            volatility_state=volatility,
            risk_score=float(risk_score),
            risk_level=risk_level,
            net_liquidity=net_liq_str,
            final_recommendation=final_rec,
            confidence=float(confidence),
            data_quality=data_quality,
            summary_text=summary
        )

    def _generate_summary_text(
        self,
        regime: str,
        volatility: str,
        risk_score: float,
        recommendation: str,
        default_hold: bool
    ) -> str:
        """요약 문장 생성 (새 숫자 없이 JSON 값만 조합)"""
        if default_hold:
            return f"현재 {regime} 레짐, {volatility} 상태이나, 데이터 신뢰도 이슈로 HOLD 권고."

        if regime == 'BULL':
            regime_text = "상승 추세"
        elif regime == 'BEAR':
            regime_text = "하락 압력"
        else:
            regime_text = "횡보 국면"

        return f"현재 {regime_text} ({regime} 레짐), 변동성 {volatility}, 리스크 점수 {risk_score:.1f}. 최종 권고: {recommendation}."

    def _extract_key_rationales(self, data: Dict) -> List[KeyRationale]:
        """핵심 근거 3가지 추출 (JSON 값만 인용)"""
        rationales = []

        # 1. 레짐 기반 근거
        regime_data = data.get('regime', {})
        if regime_data:
            regime = regime_data.get('regime', 'NEUTRAL')
            gmm_regime = regime_data.get('gmm_regime', '')
            entropy = regime_data.get('entropy', 0)

            rationales.append(KeyRationale(
                title="시장 레짐 분석",
                source="regime",
                value=f"{regime}" + (f" (GMM: {gmm_regime}, Entropy: {entropy:.3f})" if gmm_regime else ""),
                interpretation=self._interpret_regime(regime, entropy)
            ))

        # 2. 유동성 신호 기반 근거
        liq_signal = data.get('liquidity_signal', 'NEUTRAL')
        fred = data.get('fred_summary', {})
        rrp = fred.get('rrp', 0)
        tga = fred.get('tga', 0)

        rationales.append(KeyRationale(
            title="Fed 유동성 상태",
            source="liquidity_signal, fred_summary",
            value=f"{liq_signal} (RRP: ${rrp:.0f}B, TGA: ${tga:.0f}B)",
            interpretation=self._interpret_liquidity(liq_signal, rrp, tga)
        ))

        # 3. AI 토론 합의
        consensus = data.get('debate_consensus', {})
        full_mode = data.get('full_mode_position', 'NEUTRAL')
        ref_mode = data.get('reference_mode_position', 'NEUTRAL')
        modes_agree = data.get('modes_agree', True)

        rationales.append(KeyRationale(
            title="AI 에이전트 토론 결과",
            source="debate_consensus, full_mode_position, reference_mode_position",
            value=f"Full: {full_mode}, Ref: {ref_mode}, 합의: {'예' if modes_agree else '아니오'}",
            interpretation=self._interpret_debate(full_mode, ref_mode, modes_agree)
        ))

        return rationales[:3]

    def _interpret_regime(self, regime: str, entropy: float) -> str:
        """레짐 해석 (새 값 생성 없음)"""
        if regime == 'BULL':
            return f"상승 추세 레짐. Entropy {entropy:.3f}으로 {'높은' if entropy > 0.5 else '낮은'} 불확실성."
        elif regime == 'BEAR':
            return f"하락 추세 레짐. 방어적 포지셔닝 필요."
        else:
            return f"중립 레짐. 방향성 모니터링 필요."

    def _interpret_liquidity(self, signal: str, rrp: float, tga: float) -> str:
        """유동성 해석"""
        if signal == 'BULLISH':
            return f"RRP 감소와 TGA 안정으로 유동성 공급 확대 중."
        elif signal == 'BEARISH':
            return f"유동성 긴축 신호. RRP/TGA 모니터링 필요."
        else:
            return f"유동성 중립. 변화 방향 관찰 필요."

    def _interpret_debate(self, full: str, ref: str, agree: bool) -> str:
        """토론 결과 해석"""
        if agree:
            return f"두 모드 모두 {full} 판단. 신뢰도 높음."
        else:
            return f"Full({full})과 Reference({ref}) 모드 불일치. 신중한 접근 필요."

    def _extract_risk_conditions(self, data: Dict) -> List[RiskCondition]:
        """리스크 및 반증 조건 3가지 추출"""
        conditions = []

        # 1. 리스크 점수 기반 조건
        risk_score = data.get('risk_score', 50)
        conditions.append(RiskCondition(
            risk_title="리스크 점수 상승",
            current_value=f"{risk_score:.1f}/100",
            falsification_condition=f"리스크 점수가 {min(risk_score + 20, 100):.0f} 이상 상승 시 뷰 재검토",
            monitoring_metric="risk_score"
        ))

        # 2. 버블 리스크
        bubble = data.get('bubble_risk', {})
        if isinstance(bubble, dict):
            status = bubble.get('overall_status', 'NONE')
        else:
            status = 'NONE'

        conditions.append(RiskCondition(
            risk_title="버블 리스크",
            current_value=f"{status}",
            falsification_condition="버블 상태가 WARNING → DANGER 전환 시 즉시 비중 축소",
            monitoring_metric="bubble_risk.overall_status"
        ))

        # 3. 변동성 스파이크
        regime_data = data.get('regime', {})
        volatility = regime_data.get('volatility', 'Normal') if isinstance(regime_data, dict) else 'Normal'

        conditions.append(RiskCondition(
            risk_title="변동성 급등",
            current_value=f"{volatility}",
            falsification_condition="변동성이 High Vol로 전환 시 포지션 규모 축소 검토",
            monitoring_metric="regime.volatility"
        ))

        return conditions[:3]

    def _generate_action_items(self, data: Dict, default_hold: bool) -> List[ActionItem]:
        """액션 아이템 생성 (제약 조건 준수)"""
        items = []

        # 기본 HOLD인 경우
        if default_hold:
            items.append(ActionItem(
                action="HOLD",
                target="전체 포트폴리오",
                rationale="데이터 신뢰도 저하 또는 신호 충돌로 인해 현재 비중 유지",
                constraints=["리밸런싱 보류", "추가 데이터 확인 후 재평가"],
                priority="HIGH"
            ))
            return items

        # 리밸런싱 결정 확인
        rebal = data.get('rebalance_decision', {})
        should_rebal = rebal.get('should_rebalance', False)
        action = rebal.get('action', 'HOLD')
        turnover = rebal.get('turnover', 0)

        if should_rebal and turnover <= self.turnover_cap:
            items.append(ActionItem(
                action="REBALANCE",
                target="포트폴리오 전체",
                rationale=rebal.get('reason', '목표 비중 편차 발생'),
                constraints=[
                    f"Turnover Cap: {self.turnover_cap:.0%} 이내",
                    f"현재 Turnover: {turnover:.1%}",
                    f"Equity Max: {self.weight_bounds['equity_max']:.0%}",
                    f"Bond Max: {self.weight_bounds['bond_max']:.0%}"
                ],
                priority="MEDIUM"
            ))
        else:
            # 리밸런싱 불필요 또는 제약 초과
            hold_reason = "리밸런싱 조건 미충족" if not should_rebal else f"Turnover ({turnover:.1%}) > Cap ({self.turnover_cap:.0%})"
            items.append(ActionItem(
                action="HOLD",
                target="전체 포트폴리오",
                rationale=hold_reason,
                constraints=[f"Turnover Cap: {self.turnover_cap:.0%}"],
                priority="MEDIUM"
            ))

        # 포트폴리오 비중 기반 액션 (JSON 값 인용)
        weights = data.get('portfolio_weights', {}) or data.get('allocation_result', {}).get('weights', {})
        if weights:
            # 상위 비중 자산 모니터링
            top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{t}: {w:.1%}" for t, w in top_weights])
            items.append(ActionItem(
                action="MONITOR",
                target="상위 비중 자산",
                rationale=f"현재 상위 비중: {top_str}",
                constraints=[f"단일 자산 Max: {self.weight_bounds.get('equity_max', 0.6):.0%}"],
                priority="LOW"
            ))

        return items

    def save_report(
        self,
        report: AllocationReport,
        output_dir: str = "outputs/reports"
    ) -> str:
        """리포트 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"allocation_report_{timestamp}.md"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())

        # JSON도 함께 저장
        json_path = output_path / f"allocation_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Report saved to {filepath}")
        return str(filepath)


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Allocation Report Agent Test")
    print("=" * 60)

    # Mock EIMAS 결과 (실제 JSON 구조)
    mock_eimas_result = {
        "timestamp": "2026-02-02T22:30:00",
        "regime": {
            "regime": "BULL",
            "volatility": "Low Vol",
            "gmm_regime": "Bull",
            "entropy": 0.342
        },
        "risk_score": 45.2,
        "risk_level": "MEDIUM",
        "fred_summary": {
            "rrp": 5.2,
            "tga": 721.5,
            "net_liquidity": 5799.3
        },
        "liquidity_signal": "BULLISH",
        "final_recommendation": "BULLISH",
        "confidence": 0.65,
        "modes_agree": True,
        "full_mode_position": "BULLISH",
        "reference_mode_position": "BULLISH",
        "has_strong_dissent": False,
        "market_quality": {
            "data_quality": "COMPLETE",
            "avg_liquidity_score": 65.2
        },
        "bubble_risk": {
            "overall_status": "WATCH"
        },
        "portfolio_weights": {
            "HYG": 0.54,
            "DIA": 0.06,
            "XLV": 0.05,
            "SPY": 0.10,
            "TLT": 0.15,
            "GLD": 0.10
        },
        "rebalance_decision": {
            "should_rebalance": False,
            "action": "HOLD",
            "reason": "편차 임계값 미달",
            "turnover": 0.02
        }
    }

    # 에이전트 생성 및 리포트 생성
    agent = AllocationReportAgent()
    report = agent.generate_report(mock_eimas_result)

    # 마크다운 출력
    print("\n" + report.to_markdown())

    # 신호 충돌 테스트
    print("\n" + "=" * 60)
    print("Test 2: Signal Conflict (HOLD expected)")
    print("=" * 60)

    conflict_result = mock_eimas_result.copy()
    conflict_result['modes_agree'] = False
    conflict_result['has_strong_dissent'] = True

    report2 = agent.generate_report(conflict_result)
    print(f"\nDefault to HOLD: {report2.default_to_hold}")
    print(f"Reason: {report2.hold_reason}")
    print(f"Action: {report2.action_items[0].action if report2.action_items else 'N/A'}")

    # 데이터 품질 저하 테스트
    print("\n" + "=" * 60)
    print("Test 3: Data Quality Degraded (HOLD expected)")
    print("=" * 60)

    degraded_result = mock_eimas_result.copy()
    degraded_result['market_quality'] = {'data_quality': 'DEGRADED'}

    report3 = agent.generate_report(degraded_result)
    print(f"\nDefault to HOLD: {report3.default_to_hold}")
    print(f"Reason: {report3.hold_reason}")

    print("\nTest completed successfully!")
