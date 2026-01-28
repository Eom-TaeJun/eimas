"""
Economic Insight Agent - Pydantic Schemas
==========================================

agentcommand.txt 요구사항에 맞는 JSON 출력 스키마 정의
기존 EIMAS 모듈과 호환 가능하도록 설계

Output JSON (must include these top-level keys):
- meta: request_id, timestamp, frame
- phenomenon: one sentence
- causal_graph: nodes[], edges[]
- mechanisms: paths[] (each path lists nodes, edge_signs, narrative)
- hypotheses: main, rivals[], falsification_tests[]
- risk: regime_shift_risks[], data_limitations[]
- suggested_data: prioritized datasets to fetch next
- next_actions: 3-7 concrete steps
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# =============================================================================
# Enums
# =============================================================================

class AnalysisFrame(str, Enum):
    """분석 프레임 분류"""
    MACRO = "macro"              # 거시경제 (Fed, inflation, GDP)
    MARKETS = "markets"          # 시장 (equities, bonds, FX)
    CRYPTO = "crypto"            # 암호화폐 (stablecoins, DeFi)
    MIXED = "mixed"              # 복합

class EdgeSign(str, Enum):
    """인과 관계 방향"""
    POSITIVE = "+"               # 양의 영향
    NEGATIVE = "-"               # 음의 영향
    AMBIGUOUS = "?"              # 불확실

class ConfidenceLevel(str, Enum):
    """신뢰도 수준"""
    HIGH = "high"                # > 0.8
    MEDIUM = "medium"            # 0.5 - 0.8
    LOW = "low"                  # < 0.5

class RiskSeverity(str, Enum):
    """리스크 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Request Schema (Input)
# =============================================================================

class InsightRequest(BaseModel):
    """분석 요청 입력"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., description="분석 질문 (예: 'Fed가 금리를 올리면 어떻게 되나?')")
    context: Optional[Dict[str, Any]] = Field(default=None, description="추가 컨텍스트")
    frame_hint: Optional[AnalysisFrame] = Field(default=None, description="프레임 힌트 (자동 감지 가능)")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "스테이블코인 공급 증가가 국채 수요에 미치는 영향은?",
                "frame_hint": "crypto"
            }
        }


# =============================================================================
# Causal Graph Schema
# =============================================================================

class CausalNode(BaseModel):
    """인과 그래프 노드"""
    id: str = Field(..., description="노드 ID")
    name: str = Field(..., description="노드 이름 (예: 'Fed_Funds_Rate')")
    layer: Optional[str] = Field(default=None, description="계층 (POLICY/LIQUIDITY/RISK_PREMIUM/ASSET_PRICE)")
    category: Optional[str] = Field(default=None, description="카테고리 (macro/market/crypto/sector)")

    # 기존 EIMAS 모듈과 호환
    centrality: Optional[float] = Field(default=None, description="중심성 점수")
    criticality: Optional[float] = Field(default=None, description="Critical Path 기여도")

class CausalEdge(BaseModel):
    """인과 그래프 엣지"""
    source: str = Field(..., description="출발 노드 ID")
    target: str = Field(..., description="도착 노드 ID")
    sign: EdgeSign = Field(..., description="관계 방향 (+/-/?)")

    # 통계적 근거 (기존 Granger Causality와 호환)
    lag: Optional[int] = Field(default=None, description="시차 (일)")
    p_value: Optional[float] = Field(default=None, description="Granger p-value")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

    # 경제학적 근거
    mechanism: Optional[str] = Field(default=None, description="전달 메커니즘 설명")

class CausalGraph(BaseModel):
    """인과 그래프 전체"""
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)

    # 그래프 메타데이터
    has_cycles: bool = Field(default=False, description="피드백 루프 존재 여부")
    critical_path: Optional[List[str]] = Field(default=None, description="Critical Path 노드 순서")


# =============================================================================
# Mechanism Schema
# =============================================================================

class MechanismPath(BaseModel):
    """전달 메커니즘 경로"""
    path_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    nodes: List[str] = Field(..., description="경로상 노드들")
    edge_signs: List[str] = Field(..., description="각 엣지의 부호 (+/-)")
    net_effect: EdgeSign = Field(..., description="최종 효과 방향")

    # 내러티브
    narrative: str = Field(..., description="경로 설명 (한글)")
    strength: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

    # 기존 ShockPropagationGraph와 호환
    bottleneck_node: Optional[str] = Field(default=None, description="병목 노드")


# =============================================================================
# Hypothesis Schema (신규)
# =============================================================================

class FalsificationTest(BaseModel):
    """가설 반증 테스트"""
    test_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="테스트 설명")
    data_required: List[str] = Field(..., description="필요한 데이터")
    expected_if_true: str = Field(..., description="가설이 맞다면 예상 결과")
    expected_if_false: str = Field(..., description="가설이 틀리다면 예상 결과")

class Hypothesis(BaseModel):
    """가설"""
    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str = Field(..., description="가설 진술")
    supporting_evidence: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)

class HypothesesSection(BaseModel):
    """가설 섹션"""
    main: Hypothesis = Field(..., description="주요 가설")
    rivals: List[Hypothesis] = Field(default_factory=list, description="대안 가설들")
    falsification_tests: List[FalsificationTest] = Field(default_factory=list)


# =============================================================================
# Risk Schema
# =============================================================================

class RegimeShiftRisk(BaseModel):
    """레짐 변화 리스크"""
    risk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="리스크 설명")
    trigger: str = Field(..., description="트리거 조건")
    probability: Optional[float] = Field(default=None, ge=0, le=1)
    severity: RiskSeverity = Field(default=RiskSeverity.MEDIUM)

    # 기존 BubbleDetector, CriticalPath와 호환
    source_module: Optional[str] = Field(default=None, description="출처 모듈")

class DataLimitation(BaseModel):
    """데이터 한계"""
    limitation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="한계 설명")
    impact: str = Field(..., description="분석에 미치는 영향")
    mitigation: Optional[str] = Field(default=None, description="완화 방안")

class RiskSection(BaseModel):
    """리스크 섹션"""
    regime_shift_risks: List[RegimeShiftRisk] = Field(default_factory=list)
    data_limitations: List[DataLimitation] = Field(default_factory=list)


# =============================================================================
# Suggested Data & Actions
# =============================================================================

class SuggestedDataset(BaseModel):
    """추천 데이터셋"""
    name: str = Field(..., description="데이터셋 이름")
    category: str = Field(..., description="카테고리 (macro/flows/on-chain)")
    priority: int = Field(..., ge=1, le=5, description="우선순위 (1=최고)")
    rationale: str = Field(..., description="필요한 이유")
    source: Optional[str] = Field(default=None, description="데이터 소스")

class NextAction(BaseModel):
    """다음 행동"""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(..., description="행동 설명")
    category: str = Field(..., description="카테고리 (data/analysis/monitor/trade)")
    priority: int = Field(..., ge=1, le=5)
    timeframe: Optional[str] = Field(default=None, description="시간 프레임")


# =============================================================================
# Meta Schema
# =============================================================================

class InsightMeta(BaseModel):
    """메타 정보"""
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    frame: AnalysisFrame

    # 기존 EIMAS 모듈 출처 추적
    modules_used: List[str] = Field(default_factory=list, description="사용된 EIMAS 모듈")
    processing_time_ms: Optional[int] = Field(default=None)


# =============================================================================
# Final Output Schema
# =============================================================================

class EconomicInsightReport(BaseModel):
    """
    Economic Insight Agent 최종 출력

    agentcommand.txt 요구사항 충족:
    - meta: request_id, timestamp, frame
    - phenomenon: one sentence
    - causal_graph: nodes[], edges[]
    - mechanisms: paths[]
    - hypotheses: main, rivals[], falsification_tests[]
    - risk: regime_shift_risks[], data_limitations[]
    - suggested_data: prioritized datasets
    - next_actions: 3-7 concrete steps
    """
    meta: InsightMeta
    phenomenon: str = Field(..., description="현상 요약 (한 문장)")
    causal_graph: CausalGraph
    mechanisms: List[MechanismPath] = Field(..., min_length=1, max_length=5)
    hypotheses: HypothesesSection
    risk: RiskSection
    suggested_data: List[SuggestedDataset] = Field(..., max_length=10)
    next_actions: List[NextAction] = Field(..., min_length=3, max_length=7)

    # 기존 EIMAS 결과와의 호환성
    raw_eimas_data: Optional[Dict[str, Any]] = Field(default=None, description="원본 EIMAS 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "meta": {
                    "request_id": "abc123",
                    "timestamp": "2026-01-28T14:00:00",
                    "frame": "crypto",
                    "modules_used": ["genius_act_macro", "shock_propagation_graph"]
                },
                "phenomenon": "스테이블코인 공급 증가가 국채 단기물 수요를 견인하고 있다",
                "causal_graph": {
                    "nodes": [
                        {"id": "USDC_SUPPLY", "name": "USDC Supply", "layer": "LIQUIDITY"},
                        {"id": "TBILL_DEMAND", "name": "T-Bill Demand", "layer": "ASSET_PRICE"}
                    ],
                    "edges": [
                        {"source": "USDC_SUPPLY", "target": "TBILL_DEMAND", "sign": "+", "mechanism": "담보 수요"}
                    ]
                },
                "mechanisms": [
                    {
                        "nodes": ["USDC_SUPPLY", "RESERVE_DEMAND", "TBILL_DEMAND"],
                        "edge_signs": ["+", "+"],
                        "net_effect": "+",
                        "narrative": "USDC 발행 증가 → Circle 준비금 증가 → 국채 매수"
                    }
                ],
                "hypotheses": {
                    "main": {
                        "statement": "스테이블코인 성장이 국채 수요의 새로운 동인이다",
                        "confidence": "high"
                    },
                    "rivals": [
                        {"statement": "국채 수요는 Fed 정책에 의해 주도된다", "confidence": "medium"}
                    ],
                    "falsification_tests": [
                        {
                            "description": "스테이블코인 감소 시 국채 수요 감소 확인",
                            "data_required": ["stablecoin_supply", "tbill_auction_results"],
                            "expected_if_true": "양의 상관관계",
                            "expected_if_false": "무상관"
                        }
                    ]
                },
                "risk": {
                    "regime_shift_risks": [
                        {"description": "규제 변화로 스테이블코인 준비금 요건 강화", "severity": "high"}
                    ],
                    "data_limitations": [
                        {"description": "실시간 준비금 구성 비공개", "impact": "정확한 국채 비중 추정 불가"}
                    ]
                },
                "suggested_data": [
                    {"name": "Circle USDC 준비금 보고서", "category": "on-chain", "priority": 1}
                ],
                "next_actions": [
                    {"description": "월간 스테이블코인 공급 vs 국채 경매 결과 상관분석", "priority": 1},
                    {"description": "Tether 준비금 공시 모니터링", "priority": 2},
                    {"description": "Fed RRP 잔고와 스테이블코인 TVL 비교", "priority": 3}
                ]
            }
        }


# =============================================================================
# Adapter Functions (기존 EIMAS 모듈과 연결)
# =============================================================================

def from_shock_propagation(spg_result: Dict) -> CausalGraph:
    """
    기존 ShockPropagationGraph 결과를 CausalGraph로 변환

    Args:
        spg_result: ShockPropagationGraph.analyze() 결과

    Returns:
        CausalGraph 인스턴스
    """
    nodes = []
    edges = []

    # 노드 변환
    if 'node_roles' in spg_result:
        for node_id, role_info in spg_result['node_roles'].items():
            nodes.append(CausalNode(
                id=node_id,
                name=node_id.replace('_', ' ').title(),
                layer=role_info.get('layer'),
                centrality=role_info.get('centrality_score')
            ))

    # 엣지 변환 (Granger 결과)
    if 'granger_results' in spg_result:
        for result in spg_result['granger_results']:
            edges.append(CausalEdge(
                source=result.get('cause'),
                target=result.get('effect'),
                sign=EdgeSign.POSITIVE if result.get('coefficient', 0) > 0 else EdgeSign.NEGATIVE,
                lag=result.get('optimal_lag'),
                p_value=result.get('p_value'),
                confidence=ConfidenceLevel.HIGH if result.get('p_value', 1) < 0.01 else
                          ConfidenceLevel.MEDIUM if result.get('p_value', 1) < 0.05 else
                          ConfidenceLevel.LOW
            ))

    return CausalGraph(
        nodes=nodes,
        edges=edges,
        critical_path=spg_result.get('critical_path', {}).get('path')
    )


def from_critical_path(cp_result: Dict) -> List[RegimeShiftRisk]:
    """
    기존 CriticalPathAggregator 결과를 RegimeShiftRisk로 변환
    """
    risks = []

    # 레짐 전환 리스크
    if cp_result.get('transition_probability', 0) > 0.3:
        risks.append(RegimeShiftRisk(
            description=f"현재 레짐({cp_result.get('current_regime')})에서 전환 가능성",
            trigger=f"전환 확률 {cp_result.get('transition_probability'):.1%}",
            probability=cp_result.get('transition_probability'),
            severity=RiskSeverity.HIGH if cp_result.get('transition_probability', 0) > 0.5 else RiskSeverity.MEDIUM,
            source_module="critical_path"
        ))

    # 활성 경고
    for warning in cp_result.get('active_warnings', []):
        risks.append(RegimeShiftRisk(
            description=warning,
            trigger="임계값 초과",
            severity=RiskSeverity.MEDIUM,
            source_module="critical_path"
        ))

    return risks


def from_genius_act(ga_result: Dict) -> List[MechanismPath]:
    """
    기존 GeniusActMacroStrategy 결과를 MechanismPath로 변환
    """
    paths = []

    # 스테이블코인 → 유동성 경로
    if ga_result.get('stablecoin_signals'):
        for signal in ga_result['stablecoin_signals']:
            paths.append(MechanismPath(
                nodes=['Stablecoin_Supply', 'Reserve_Demand', 'Treasury_Demand', 'Net_Liquidity'],
                edge_signs=['+', '+', '+'],
                net_effect=EdgeSign.POSITIVE,
                narrative=f"스테이블코인 {signal.get('direction', 'change')} → 준비금 수요 변화 → 국채 수요 영향",
                strength=ConfidenceLevel.MEDIUM
            ))

    # M = B + S*B* 경로
    if ga_result.get('extended_liquidity'):
        el = ga_result['extended_liquidity']
        paths.append(MechanismPath(
            nodes=['Fed_Balance_Sheet', 'Net_Liquidity', 'Stablecoin_Contribution', 'Extended_M'],
            edge_signs=['+', '+', '+'],
            net_effect=EdgeSign.POSITIVE,
            narrative=f"확장 유동성: M={el.get('total_m', 0):.0f}B (기본={el.get('base_b', 0):.0f}B + 스테이블={el.get('stablecoin_sb', 0):.0f}B)",
            strength=ConfidenceLevel.HIGH
        ))

    return paths
