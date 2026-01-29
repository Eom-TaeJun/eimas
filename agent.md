# EIMAS Agent Systems Documentation

> **버전**: v2.2.0 (2026-01-28)
> **목적**: EIMAS의 두 에이전트 시스템(`agents/` 와 `agent/`) 구조와 기능 설명

---

## 목차

1. [에이전트 시스템 개요](#1-에이전트-시스템-개요)
2. [agents/ - 멀티에이전트 토론 시스템](#2-agents---멀티에이전트-토론-시스템)
3. [agent/ - Economic Insight Agent](#3-agent---economic-insight-agent)
4. [두 시스템의 관계](#4-두-시스템의-관계)
5. [실행 방법](#5-실행-방법)
6. [구현 상세](#6-구현-상세)

---

## 1. 에이전트 시스템 개요

EIMAS는 **두 개의 독립적인 에이전트 시스템**을 포함합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EIMAS Agent Systems                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────┐    │
│  │      agents/            │    │       agent/            │    │
│  │  (Multi-Agent Debate)   │    │ (Economic Insight)      │    │
│  │                         │    │                         │    │
│  │  - MetaOrchestrator     │    │  - EconomicInsight      │    │
│  │  - AnalysisAgent        │    │    Orchestrator         │    │
│  │  - ForecastAgent        │    │  - EIMASAdapter         │    │
│  │  - ResearchAgent        │    │  - Pydantic Schemas     │    │
│  │  - StrategyAgent        │    │                         │    │
│  │  - Multi-LLM Debate     │    │  6단계 추론 파이프라인   │    │
│  │                         │    │  JSON-first 출력        │    │
│  └───────────┬─────────────┘    └───────────┬─────────────┘    │
│              │                              │                   │
│              └──────────┬───────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                           │
│              │   EIMASAdapter      │                           │
│              │   (결과 변환)        │                           │
│              └─────────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 두 시스템의 차이점

| 구분 | `agents/` | `agent/` |
|------|-----------|----------|
| **목적** | AI 에이전트 토론 및 합의 도출 | 인과적, 설명 가능한 분석 |
| **방식** | Multi-LLM 토론 (Claude/GPT-4/Gemini) | 6단계 추론 파이프라인 |
| **출력** | consensus, dissent, reasoning_chain | JSON (causal_graph, mechanisms, hypotheses) |
| **실행** | main.py Phase 3에서 호출 | 독립 CLI 또는 EIMAS 통합 |
| **LLM** | 실제 LLM API 호출 | 템플릿 기반 (LLM 선택적) |

---

## 2. agents/ - 멀티에이전트 토론 시스템

### 2.1 디렉토리 구조

```
agents/
├── __init__.py              # 에이전트 export
├── base_agent.py            # BaseAgent 추상 클래스
├── orchestrator.py          # MetaOrchestrator (토론 조정)
├── analysis_agent.py        # CriticalPath 기반 분석
├── forecast_agent.py        # LASSO 예측
├── research_agent.py        # Perplexity API 연동
├── strategy_agent.py        # 포트폴리오 전략
├── verification_agent.py    # AI 출력 검증
├── interpretation_debate.py # 경제학파별 해석 토론
└── methodology_debate.py    # 방법론 선택 토론
```

### 2.2 에이전트 역할

| 에이전트 | 역할 | LLM 사용 | 출력 |
|---------|------|---------|------|
| `MetaOrchestrator` | 워크플로우 조정, 토론 실행 | X (Rule-based) | consensus, debate_results |
| `AnalysisAgent` | Critical Path 분석 | X | risk_score, regime |
| `ForecastAgent` | LASSO 예측 | X | rate_forecast |
| `ResearchAgent` | 실시간 리서치 | **Claude** | opinions, sources |
| `StrategyAgent` | 포트폴리오 전략 | X | weights, reasoning |
| `VerificationAgent` | AI 출력 검증 | X | verification_score |
| `InterpretationDebateAgent` | 경제학파 토론 | **Multi-LLM** | school_interpretations |
| `MethodologyDebateAgent` | 방법론 토론 | **Multi-LLM** | methodology_decision |

### 2.3 MetaOrchestrator 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│                 MetaOrchestrator Workflow                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: 데이터 수집
         │
         ▼
Step 2: 에이전트별 분석 실행
         │  - AnalysisAgent (risk_score)
         │  - ForecastAgent (rate_forecast)
         │  - ResearchAgent (context)
         │  - StrategyAgent (portfolio)
         ▼
Step 3: 의견 수집 (form_opinion)
         │
         ▼
Step 4: 토론 프로토콜 실행
         │  - Round 1: Initial Positions
         │  - Round 2: Cross-Examination
         │  - Round 3: Synthesis
         ▼
Step 4.5: 심화 토론 (선택)
         │  - InterpretationDebateAgent
         │  - MethodologyDebateAgent
         ▼
Step 5: 합의 도출
         │
         ▼
Step 6: 검증 (VerificationAgent)
         │
         ▼
Step 7: 최종 보고서 생성
```

### 2.4 Multi-LLM Debate Engine

```python
class MultiLLMDebate:
    """
    3개 LLM 토론 엔진

    참여자:
    - Claude (Economist): 경제학적 관점
    - GPT-4 (Devil's Advocate): 반론 제시
    - Gemini (Risk Manager): 리스크 관점
    """

    async def run_debate(self, topic: str, context: Dict) -> DebateResult:
        # Round 1: Initial Positions
        positions = await self._gather_initial_positions(topic, context)

        # Round 2: Cross-Examination
        rebuttals = await self._cross_examine(positions)

        # Round 3: Synthesis (Claude)
        synthesis = await self._synthesize(positions, rebuttals)

        return DebateResult(
            consensus_position='BULLISH' | 'BEARISH' | 'NEUTRAL',
            consensus_confidence=(60, 75),  # 범위
            dissent_points=[...],
            model_contributions={'claude': ..., 'gpt4': ..., 'gemini': ...}
        )
```

### 2.5 ReasoningChain (추론 추적)

```python
class ReasoningChain:
    """AI 의사결정 과정 추적"""

    def add_step(self, agent, input_summary, output_summary, confidence, key_factors):
        self.chain.append({
            'step': len(self.chain) + 1,
            'agent': agent,
            'input': input_summary,
            'output': output_summary,
            'confidence': confidence,
            'key_factors': key_factors
        })

# 사용 예시
reasoning = ReasoningChain()
reasoning.add_step(
    agent="AnalysisAgent",
    input_summary="Market data + FRED",
    output_summary="Risk Score: 45.2/100, Regime: Bull",
    confidence=78,
    key_factors=["VIX 14.2 < 20", "Credit Spread 285bp < 300bp"]
)
```

---

## 3. agent/ - Economic Insight Agent

### 3.1 디렉토리 구조

```
agent/
├── __init__.py              # Main exports
├── cli.py                   # CLI 인터페이스
├── README.md                # Agent 문서
├── core/
│   ├── __init__.py
│   ├── adapters.py          # EIMAS 모듈 → Schema 변환 (631줄)
│   └── orchestrator.py      # 6단계 추론 파이프라인 (830줄)
├── schemas/
│   ├── __init__.py
│   └── insight_schema.py    # Pydantic 스키마 (424줄)
├── examples/
│   ├── request_stablecoin.json
│   ├── request_fed_policy.json
│   ├── request_market_rotation.json
│   └── request_mixed.json
├── evals/
│   ├── __init__.py
│   ├── scenarios.py         # 10개 평가 시나리오
│   └── runner.py            # 평가 실행기
└── tests/
    ├── __init__.py
    ├── test_schemas.py      # 스키마 테스트
    ├── test_graph_utilities.py  # 그래프 알고리즘 테스트
    └── test_orchestrator.py # 통합 테스트
```

### 3.2 6단계 추론 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│              Economic Insight Agent Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Parse Request → Classify Frame
        │
        │  InsightRequest(question="스테이블코인 공급 증가 영향?")
        │  → Frame: CRYPTO
        ▼
Step 2: Build Initial Causal Graph Template
        │
        │  Stablecoin_Supply → Reserve_Demand → TBill_Demand
        │  (프레임별 템플릿 그래프 사용)
        ▼
Step 3: Map Evidence into Nodes/Edges
        │
        │  p-value, lag, confidence 업데이트
        │  (EIMAS 결과 또는 컨텍스트에서)
        ▼
Step 4: Generate Mechanism Paths
        │
        │  Path: [Stablecoin, Reserve, TBill]
        │  Signs: [+, +], Net: +
        │  Narrative: "스테이블코인 증가 → 담보 수요 → 국채 매수"
        ▼
Step 5: Generate Hypotheses + Falsification Tests
        │
        │  Main: "스테이블코인 성장이 국채 수요 동인"
        │  Rival: "Fed 정책이 주 동인"
        │  Test: "스테이블코인 감소 시 국채 수요 감소 확인"
        ▼
Step 6: Produce Final JSON Report
        │
        │  EconomicInsightReport {
        │    meta, phenomenon, causal_graph,
        │    mechanisms, hypotheses, risk,
        │    suggested_data, next_actions
        │  }
        ▼
```

### 3.3 분석 프레임 (Analysis Frames)

| Frame | 키워드 | 템플릿 그래프 |
|-------|--------|--------------|
| `MACRO` | Fed, 금리, inflation, GDP | Fed → Liquidity → VIX → Assets |
| `CRYPTO` | stablecoin, BTC, DeFi | Stablecoin → Reserve → Treasury |
| `MARKETS` | SPY, VIX, sector | Sentiment → Flows → Prices |
| `MIXED` | (복합) | Macro + Crypto 결합 |

### 3.4 Pydantic 스키마

```python
# 요청
class InsightRequest(BaseModel):
    request_id: str
    question: str
    frame_hint: Optional[AnalysisFrame]
    context: Optional[Dict]

# 인과 그래프
class CausalGraph(BaseModel):
    nodes: List[CausalNode]      # 노드들
    edges: List[CausalEdge]      # 엣지들
    has_cycles: bool             # 피드백 루프 존재
    critical_path: List[str]     # 핵심 경로

class CausalNode(BaseModel):
    id: str
    name: str
    layer: str      # POLICY/LIQUIDITY/RISK_PREMIUM/ASSET_PRICE
    category: str   # macro/market/crypto/sector
    centrality: Optional[float]
    criticality: Optional[float]

class CausalEdge(BaseModel):
    source: str
    target: str
    sign: EdgeSign  # +/-/?
    lag: Optional[int]
    p_value: Optional[float]
    confidence: ConfidenceLevel
    mechanism: Optional[str]

# 메커니즘 경로
class MechanismPath(BaseModel):
    nodes: List[str]
    edge_signs: List[str]
    net_effect: EdgeSign
    narrative: str
    strength: ConfidenceLevel

# 가설
class HypothesesSection(BaseModel):
    main: Hypothesis
    rivals: List[Hypothesis]
    falsification_tests: List[FalsificationTest]

# 최종 출력
class EconomicInsightReport(BaseModel):
    meta: InsightMeta
    phenomenon: str
    causal_graph: CausalGraph
    mechanisms: List[MechanismPath]   # 1-5개
    hypotheses: HypothesesSection
    risk: RiskSection
    suggested_data: List[SuggestedDataset]
    next_actions: List[NextAction]    # 3-7개
```

### 3.5 EIMASAdapter (기존 모듈 통합)

```python
class EIMASAdapter:
    """기존 EIMAS 모듈 출력 → Economic Insight Schema 변환"""

    # ShockPropagationGraph → CausalGraph
    def adapt_shock_propagation(self, spg_result: Dict) -> CausalGraph:
        """Granger 결과, Lead-Lag, shock_paths를 CausalGraph로 변환"""

    # CriticalPathAggregator → RegimeShiftRisk[]
    def adapt_critical_path(self, cp_result: Dict) -> List[RegimeShiftRisk]:
        """레짐 전환 확률, 리스크 점수, 경고를 RegimeShiftRisk로 변환"""

    # GeniusActMacroStrategy → MechanismPath[]
    def adapt_genius_act(self, ga_result: Dict) -> List[MechanismPath]:
        """확장 유동성 모델, 스테이블코인 시그널을 MechanismPath로 변환"""

    # BubbleDetector → RegimeShiftRisk[]
    def adapt_bubble_detector(self, bd_result: Dict) -> List[RegimeShiftRisk]:
        """Greenwood-Shleifer 버블 리스크를 RegimeShiftRisk로 변환"""

    # GraphClusteredPortfolio → NextAction[]
    def adapt_portfolio(self, gcp_result: Dict) -> List[NextAction]:
        """HRP 포트폴리오 결과를 NextAction으로 변환"""

    # 가설 생성 (신규 로직)
    def generate_hypotheses(self, phenomenon, graph, mechanisms) -> HypothesesSection:
        """현상과 인과 그래프를 바탕으로 가설 + 반증 테스트 생성"""
```

### 3.6 Eval Harness (10개 시나리오)

| ID | 시나리오 | Frame | 상태 |
|----|----------|-------|------|
| S01 | Stablecoin-Treasury Channel | CRYPTO | ✅ PASS |
| S02 | Fed Rate Policy Impact | MACRO | ✅ PASS |
| S03 | Liquidity Transmission Mechanism | MACRO | ✅ PASS |
| S04 | Crypto-Macro Correlation | MIXED | ✅ PASS |
| S05 | Sector Rotation Analysis | MARKETS | ✅ PASS |
| S06 | DeFi TVL and ETH | CRYPTO | ✅ PASS |
| S07 | VIX Risk Transmission | MARKETS | ✅ PASS |
| S08 | RRP Liquidity Drain | MACRO | ✅ PASS |
| S09 | Credit Spread Widening | MARKETS | ✅ PASS |
| S10 | Full Macro-Crypto Integration | MIXED | ✅ PASS |

---

## 4. 두 시스템의 관계

### 4.1 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                      Integration Flow                            │
└─────────────────────────────────────────────────────────────────┘

main.py 실행
    │
    ▼
Phase 1-2: 데이터 수집 및 분석
    │  - lib/shock_propagation_graph.py
    │  - lib/critical_path.py
    │  - lib/genius_act_macro.py
    │  - lib/bubble_detector.py
    │  - lib/graph_clustered_portfolio.py
    ▼
Phase 3: Multi-Agent Debate (agents/)
    │  - MetaOrchestrator
    │  - Multi-LLM 토론
    │  - ReasoningChain
    ▼
outputs/integrated_*.json 저장
    │
    ▼
Economic Insight Agent 연동 (선택)
    │
    │  python -m agent.cli --with-eimas --question "질문"
    │
    ▼
EIMASAdapter가 outputs/*.json 읽어서 변환
    │  - shock_propagation → CausalGraph
    │  - critical_path → RegimeShiftRisk
    │  - genius_act → MechanismPath
    ▼
EconomicInsightReport JSON 출력
```

### 4.2 사용 시나리오

**시나리오 1: EIMAS 전체 실행 후 인과 분석**
```bash
# 1. EIMAS 분석 실행
python main.py --quick

# 2. 결과 기반 인과 분석
python -m agent.cli --with-eimas --question "현재 시장 상황의 원인과 전망은?"
```

**시나리오 2: 템플릿 기반 빠른 분석**
```bash
# EIMAS 실행 없이 템플릿 기반 분석
python -m agent.cli --question "Fed 금리 인상이 시장에 미치는 영향은?"
```

**시나리오 3: Python API 통합**
```python
from agent import EconomicInsightOrchestrator, InsightRequest

# EIMAS 결과 직접 전달
orchestrator = EconomicInsightOrchestrator()
report = orchestrator.run_with_eimas_results(
    request=InsightRequest(question="분석 질문"),
    eimas_results={
        'shock_propagation': {...},
        'critical_path': {...},
        'genius_act': {...}
    }
)
```

---

## 5. 실행 방법

### 5.1 agents/ 시스템 (main.py 통합)

```bash
# 전체 파이프라인 (Phase 3 토론 포함)
python main.py

# 빠른 분석 (토론 포함)
python main.py --quick

# AI 리포트 포함
python main.py --report
```

### 5.2 agent/ 시스템 (독립 CLI)

```bash
# 템플릿 기반 분석
python -m agent.cli --question "스테이블코인 공급 증가가 국채 수요에 미치는 영향은?"

# EIMAS 결과 활용
python -m agent.cli --with-eimas --question "현재 시장 상황 분석"

# 프레임 지정
python -m agent.cli --question "분석 질문" --frame crypto

# 파일 출력
python -m agent.cli --question "분석 질문" --output report.json

# JSON 파일 입력
python -m agent.cli examples/request_fed_policy.json

# Eval 실행
python -m agent.evals.runner --verbose
```

### 5.3 Python API

```python
# agents/ 시스템
from agents import MetaOrchestrator

orchestrator = MetaOrchestrator()
result = await orchestrator.run_with_debate(
    topics=['market_outlook', 'primary_risk'],
    context={'market_data': {...}, 'regime': 'Bull'}
)

# agent/ 시스템
from agent import EconomicInsightOrchestrator, InsightRequest

orchestrator = EconomicInsightOrchestrator()
request = InsightRequest(question="Fed 금리 인상 영향?")
report = orchestrator.run(request)
print(report.model_dump_json(indent=2))
```

---

## 6. 구현 상세

### 6.1 핵심 알고리즘

**부호 합성 (Sign Composition)**
```python
# 경로의 최종 효과 계산
# + * + = +
# + * - = -
# - * - = +
def compute_net_effect(edge_signs: List[str]) -> EdgeSign:
    neg_count = sum(1 for s in edge_signs if s == "-")
    return EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
```

**사이클 감지 (DFS)**
```python
def detect_cycles(edges: List[CausalEdge]) -> bool:
    """DFS로 피드백 루프 감지"""
    adj = {e.source: [] for e in edges}
    for e in edges:
        adj[e.source].append(e.target)

    visited, rec_stack = set(), set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor): return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    return any(dfs(n) for n in adj if n not in visited)
```

**p-value → 신뢰도 변환**
```python
def p_to_confidence(p_value: float) -> ConfidenceLevel:
    if p_value < 0.01: return ConfidenceLevel.HIGH
    elif p_value < 0.05: return ConfidenceLevel.MEDIUM
    else: return ConfidenceLevel.LOW
```

### 6.2 파일별 코드 규모

| 파일 | 줄 수 | 핵심 클래스/함수 |
|------|------|-----------------|
| `agent/schemas/insight_schema.py` | 424 | InsightRequest, EconomicInsightReport, CausalGraph |
| `agent/core/adapters.py` | 631 | EIMASAdapter (6개 어댑터 메서드) |
| `agent/core/orchestrator.py` | 830 | EconomicInsightOrchestrator (6단계 파이프라인) |
| `agent/cli.py` | 150 | CLI 인터페이스 |
| `agent/evals/scenarios.py` | 120 | 10개 시나리오 정의 |
| `agent/evals/runner.py` | 200 | 평가 실행기 |
| `agents/orchestrator.py` | ~600 | MetaOrchestrator |
| `core/multi_llm_debate.py` | ~400 | MultiLLMDebate |
| `core/reasoning_chain.py` | ~100 | ReasoningChain |

### 6.3 테스트 커버리지

| 테스트 파일 | 테스트 항목 |
|------------|-------------|
| `test_schemas.py` | 스키마 유효성, JSON 직렬화, 필드 검증 |
| `test_graph_utilities.py` | 사이클 감지, 부호 합성, 카테고리 추론 |
| `test_orchestrator.py` | 템플릿 분석, EIMAS 통합, 출력 구조 |

```bash
# 테스트 실행
python -m agent.evals.runner  # 10/10 시나리오 PASS
```

---

## 부록: JSON 출력 예시

### EconomicInsightReport 샘플

```json
{
  "meta": {
    "request_id": "abc123",
    "timestamp": "2026-01-28T14:00:00",
    "frame": "crypto",
    "modules_used": ["shock_propagation_graph", "genius_act_macro"],
    "processing_time_ms": 45
  },
  "phenomenon": "스테이블코인 공급 증가가 국채 단기물 수요를 견인하고 있다",
  "causal_graph": {
    "nodes": [
      {"id": "Stablecoin_Supply", "name": "Stablecoin Supply", "layer": "LIQUIDITY", "category": "crypto"},
      {"id": "Reserve_Demand", "name": "Reserve Demand", "layer": "LIQUIDITY", "category": "macro"},
      {"id": "TBill_Demand", "name": "T-Bill Demand", "layer": "ASSET_PRICE", "category": "macro"}
    ],
    "edges": [
      {"source": "Stablecoin_Supply", "target": "Reserve_Demand", "sign": "+", "mechanism": "담보 수요"},
      {"source": "Reserve_Demand", "target": "TBill_Demand", "sign": "+", "mechanism": "국채 매수"}
    ],
    "has_cycles": false,
    "critical_path": ["Stablecoin_Supply", "Reserve_Demand", "TBill_Demand"]
  },
  "mechanisms": [
    {
      "nodes": ["Stablecoin_Supply", "Reserve_Demand", "TBill_Demand"],
      "edge_signs": ["+", "+"],
      "net_effect": "+",
      "narrative": "스테이블코인 발행 증가 → 담보 준비금 수요 증가 → 국채 매수 확대",
      "strength": "high"
    }
  ],
  "hypotheses": {
    "main": {
      "statement": "스테이블코인 성장이 국채 수요의 새로운 구조적 동인이다",
      "supporting_evidence": ["USDC 공급 +15% YoY", "Circle 준비금 80% 국채"],
      "confidence": "high"
    },
    "rivals": [
      {"statement": "Fed 정책이 국채 수요의 주 동인이다", "confidence": "medium"}
    ],
    "falsification_tests": [
      {
        "description": "스테이블코인 공급 감소 시 국채 수요 감소 확인",
        "data_required": ["stablecoin_supply", "tbill_auction_results"],
        "expected_if_true": "양의 상관관계 유지",
        "expected_if_false": "무상관 또는 역상관"
      }
    ]
  },
  "risk": {
    "regime_shift_risks": [
      {"description": "스테이블코인 규제 강화 가능성", "severity": "high", "trigger": "SEC/의회 규제"}
    ],
    "data_limitations": [
      {"description": "실시간 준비금 구성 비공개", "impact": "정확한 국채 비중 추정 불가"}
    ]
  },
  "suggested_data": [
    {"name": "Circle USDC 준비금 보고서", "category": "on-chain", "priority": 1, "rationale": "실제 국채 보유량 확인"}
  ],
  "next_actions": [
    {"description": "월간 스테이블코인 공급 vs 국채 경매 상관분석", "category": "analysis", "priority": 1},
    {"description": "Tether 준비금 공시 모니터링", "category": "monitor", "priority": 2},
    {"description": "Fed RRP 잔고와 스테이블코인 TVL 비교", "category": "analysis", "priority": 3}
  ]
}
```

---

*마지막 업데이트: 2026-01-28*
*EIMAS v2.2.0 - Multi-Agent + Economic Insight Edition*
