# EIMAS Agent Implementation Plan

> AI 에이전트의 기능이 실제 결과에 반영되도록 하는 구현 로드맵

---

## 1. 현재 상태 (2026-01-28 Phase 3 완료 + 통합)

### 활성 에이전트 (8개) - Phase 1 + 2 + 3 통합 완료
| 에이전트 | 역할 | LLM 사용 | 결과 반영 | 상태 |
|---------|------|---------|----------|------|
| `MetaOrchestrator` | 워크플로우 조정 | X (Rule-based) | 토론 합의 | **✅ 통합완료** |
| `AnalysisAgent` | Critical Path 분석 | X | risk_score | **✅ 통합완료** |
| `ForecastAgent` | LASSO 예측 | X | rate_forecast | **✅ 통합완료** |
| `ResearchAgent` | Perplexity + Claude 리서치 | **Claude** | opinions, sources | **✅ 통합완료** |
| `StrategyAgent` | 포트폴리오 전략 | X | opinions, weights | **✅ 통합완료** |
| `VerificationAgent` | AI 출력 검증 | X | verification_score | **✅ 통합완료** |
| `InterpretationDebateAgent` | 경제학파별 해석 | **Multi-LLM** | school_interpretations | **✅ 통합완료** |
| `MethodologyDebateAgent` | 방법론 선택 토론 | **Multi-LLM** | methodology_decision | **✅ 통합완료** |

### Multi-LLM Debate Engine (`core/multi_llm_debate.py`) - Phase 2 완료
- **참여 모델**: Claude (Economist), GPT-4 (Devil's Advocate), Gemini (Risk Manager)
- **토론 라운드**: Initial Position → Cross-Examination → Synthesis
- **모델 ID 수정**: `claude-sonnet-4-20250514` (최신)

### Archive 에이전트 (3개) - 향후 검토
- `VisualizationAgent`, `TopDownOrchestrator`, `RegimeChangeDetectionPipeline`

### Phase 1 + 2 완료 사항
1. **ResearchAgent**: `form_opinion()` + Claude 해석 통합
2. **StrategyAgent**: `form_opinion()` + 전략적 스탠스 로직
3. **VerificationAgent**: Step 6으로 워크플로우에 추가
4. **InterpretationDebateAgent**: 경제학파(Monetarist/Keynesian/Austrian) 토론
5. **MethodologyDebateAgent**: 방법론(LASSO/VAR/GARCH 등) 선택 토론
6. **MultiLLMDebate**: 3-모델 토론 엔진 (Claude/GPT-4/Gemini)

### Phase 3 완료 사항
1. **ReasoningChain** (`core/reasoning_chain.py`): 추론 과정 추적 시스템 구현
2. **New Schemas** (`core/schemas.py`):
   - `AgentOutputs`: 에이전트별 출력 컨테이너
   - `DebateResults`: Multi-LLM 토론 결과
   - `VerificationResults`: 검증 메트릭
3. **Report Enhancement** (`pipeline/schemas.py`):
   - Enhanced Debate (Multi-LLM) 섹션
   - Agent Contributions 섹션
   - Verification Report 섹션
   - Reasoning Chain 섹션
4. **Orchestrator Integration**: `agent_outputs` 필드 추가

### 시스템 완성도 (2026-01-28 통합 완료)
- **활성 에이전트**: 7개 + 1 Orchestrator (전체 통합 완료)
- **Multi-LLM Debate**: Claude/GPT-4/Gemini 토론 엔진 (Step 4.5)
- **Traceability**: ReasoningChain으로 전체 추론 과정 추적
- **Verification**: Step 6에서 Hallucination/Sycophancy 탐지
- **Dead Code 해소**: 모든 Phase 2-3 기능이 orchestrator.py에 통합됨

---

## 2. 목표 상태 (To-Be)

### 핵심 원칙
```
"모든 에이전트는 LLM을 호출하고, 그 결과가 최종 출력에 명시적으로 기여해야 한다"
```

### 목표 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                    MetaOrchestrator v2                       │
│                   (Multi-LLM Coordinator)                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Analysis Agent│    │ Forecast Agent│    │ Research Agent│
│   (Claude)    │    │   (Claude)    │    │ (Perplexity)  │
│ Critical Path │    │ LASSO + LLM   │    │  Real-time    │
│  Reasoning    │    │  Interpretation│   │   Context     │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────┐
                    │ Strategy Agent│
                    │   (Claude)    │
                    │  Portfolio +  │
                    │  Allocation   │
                    └───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Interpretation│    │  Methodology  │    │ Verification  │
│ Debate Agent  │    │ Debate Agent  │    │    Agent      │
│ (Multi-Model) │    │ (Multi-Model) │    │   (Claude)    │
│ School Wars   │    │ Model Choice  │    │ Fact-Check    │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │  Final Report │
                    │ (Traceable    │
                    │  Reasoning)   │
                    └───────────────┘
```

---

## 3. 구현 계획

### Phase 1: Archive 에이전트 재활성화 (1-2일)

#### Task 1.1: ResearchAgent 통합
**파일:** `agents/research_agent.py` (archive에서 복구)

**현재 상태:** Perplexity API 연동 구현됨

**수정 사항:**
```python
class ResearchAgent(BaseAgent):
    """
    실시간 뉴스/리서치 수집 및 LLM 해석

    Input: 분석 주제 (예: "Fed policy outlook")
    Output: ResearchReport with citations
    """

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        # 1. Perplexity API로 최신 정보 검색
        raw_research = await self._fetch_perplexity(request.query)

        # 2. Claude로 경제학적 맥락 해석 (NEW)
        interpreted = await self._interpret_with_claude(raw_research)

        # 3. 인용 소스와 함께 반환
        return AgentResponse(
            content=interpreted,
            sources=raw_research.citations,
            confidence=self._calculate_source_reliability()
        )
```

**결과 반영:**
```python
# main.py 수정
result.research_context = research_agent.output.content
result.research_sources = research_agent.output.sources
```

---

#### Task 1.2: StrategyAgent 통합
**파일:** `agents/strategy_agent.py` (archive에서 복구)

**수정 사항:**
```python
class StrategyAgent(BaseAgent):
    """
    포트폴리오 전략 생성 + LLM 근거 설명

    Input: market_data, risk_score, regime
    Output: PortfolioStrategy with allocation + reasoning
    """

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        # 1. GC-HRP로 최적 가중치 계산 (정량)
        weights = self._run_gc_hrp(request.market_data)

        # 2. Claude로 배분 근거 생성 (정성) - NEW
        reasoning = await self._generate_allocation_rationale(
            weights=weights,
            regime=request.context['regime'],
            risk_score=request.context['risk_score']
        )

        return AgentResponse(
            content={
                'weights': weights,
                'reasoning': reasoning,
                'rebalance_urgency': self._assess_urgency()
            }
        )
```

**결과 반영:**
```python
# main.py 수정
result.portfolio_strategy = strategy_agent.output.content
result.allocation_reasoning = strategy_agent.output.content['reasoning']
```

---

#### Task 1.3: VerificationAgent 통합
**파일:** `agents/verification_agent.py` (archive에서 복구)

**역할:** 다른 에이전트의 출력을 검증

**수정 사항:**
```python
class VerificationAgent(BaseAgent):
    """
    AI 출력 검증 (Hallucination/Sycophancy 탐지)

    Input: 다른 에이전트들의 출력
    Output: VerificationReport with flags
    """

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        agent_outputs = request.context['all_agent_outputs']

        verification_results = []
        for output in agent_outputs:
            # 1. 내부 일관성 체크
            consistency = self._check_internal_consistency(output)

            # 2. 데이터-주장 정합성 체크 (NEW)
            data_alignment = await self._verify_data_alignment(
                claims=output.claims,
                actual_data=request.market_data
            )

            # 3. Sycophancy 탐지 (과도한 낙관/비관)
            bias_score = self._detect_sycophancy(output)

            verification_results.append({
                'agent': output.agent_name,
                'consistency': consistency,
                'data_alignment': data_alignment,
                'bias_score': bias_score,
                'flags': self._generate_flags(consistency, data_alignment, bias_score)
            })

        return AgentResponse(
            content={
                'verification_results': verification_results,
                'overall_reliability': self._calculate_overall_reliability(),
                'warnings': self._extract_warnings()
            }
        )
```

**결과 반영:**
```python
# main.py 수정
result.verification_report = verification_agent.output.content
result.reliability_score = verification_agent.output.content['overall_reliability']
result.ai_warnings = verification_agent.output.content['warnings']
```

---

### Phase 2: Multi-LLM Debate 구현 (3-5일)

#### Task 2.1: 실제 LLM 토론 엔진
**파일:** `core/multi_llm_debate.py` (신규)

```python
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai

class MultiLLMDebate:
    """
    실제 LLM 간 토론을 통한 합의 도출

    방법론:
    - Round 1: 각 LLM이 독립적으로 의견 제시
    - Round 2: 상대방 의견에 대한 반론/동의
    - Round 3: Synthesis LLM이 최종 합의안 도출
    """

    def __init__(self):
        self.claude = Anthropic()
        self.openai = OpenAI()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini = genai.GenerativeModel('gemini-pro')

    async def run_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        max_rounds: int = 3
    ) -> DebateResult:
        """
        3개 LLM 토론 실행

        Args:
            topic: 토론 주제 (예: "2024 Fed Policy Direction")
            context: 시장 데이터, 분석 결과 등
            max_rounds: 최대 토론 라운드

        Returns:
            DebateResult with consensus, dissents, reasoning_chain
        """

        debate_history = []

        # Round 1: Initial Positions
        positions = await self._gather_initial_positions(topic, context)
        debate_history.append({'round': 1, 'positions': positions})

        # Round 2: Cross-Examination
        rebuttals = await self._cross_examine(positions, context)
        debate_history.append({'round': 2, 'rebuttals': rebuttals})

        # Early exit if consensus
        if self._check_early_consensus(rebuttals):
            return self._build_result(debate_history, early_exit=True)

        # Round 3: Synthesis
        synthesis = await self._synthesize(debate_history, context)
        debate_history.append({'round': 3, 'synthesis': synthesis})

        return self._build_result(debate_history)

    async def _gather_initial_positions(
        self,
        topic: str,
        context: Dict
    ) -> Dict[str, Position]:
        """각 LLM의 초기 입장 수집"""

        prompt = f"""
        Topic: {topic}

        Market Context:
        - Regime: {context['regime']}
        - Risk Score: {context['risk_score']}/100
        - Key Metrics: {context['key_metrics']}

        Provide your position with:
        1. Stance (BULLISH/BEARISH/NEUTRAL)
        2. Confidence (0-100)
        3. Key Reasoning (3 bullet points)
        4. Risk Factors (2 bullet points)
        """

        # 병렬 호출
        claude_task = self._call_claude(prompt, role="Economist")
        gpt_task = self._call_openai(prompt, role="Devil's Advocate")
        gemini_task = self._call_gemini(prompt, role="Risk Manager")

        results = await asyncio.gather(claude_task, gpt_task, gemini_task)

        return {
            'claude': self._parse_position(results[0]),
            'gpt4': self._parse_position(results[1]),
            'gemini': self._parse_position(results[2])
        }

    async def _cross_examine(
        self,
        positions: Dict[str, Position],
        context: Dict
    ) -> Dict[str, Rebuttal]:
        """상대방 의견에 대한 반론/동의"""

        cross_prompt = """
        You are reviewing other analysts' positions.

        Claude's Position: {claude_pos}
        GPT-4's Position: {gpt_pos}
        Gemini's Position: {gemini_pos}

        For each position that differs from yours:
        1. Identify the strongest counterargument
        2. Acknowledge valid points
        3. Revise your confidence if warranted
        """

        # 각 모델이 다른 모델들의 의견 검토
        rebuttals = await asyncio.gather(
            self._call_claude(cross_prompt.format(**positions)),
            self._call_openai(cross_prompt.format(**positions)),
            self._call_gemini(cross_prompt.format(**positions))
        )

        return {
            'claude': rebuttals[0],
            'gpt4': rebuttals[1],
            'gemini': rebuttals[2]
        }

    async def _synthesize(
        self,
        debate_history: List[Dict],
        context: Dict
    ) -> Synthesis:
        """최종 합의안 도출 (Claude가 Synthesizer 역할)"""

        synthesis_prompt = f"""
        You are the final synthesizer for this debate.

        Debate History:
        {json.dumps(debate_history, indent=2)}

        Your task:
        1. Identify points of consensus
        2. Document unresolved disagreements
        3. Provide final recommendation with confidence interval
        4. List key assumptions that could invalidate this conclusion

        Output format:
        - Consensus Position: [BULLISH/BEARISH/NEUTRAL]
        - Confidence: [X-Y%] (range reflecting uncertainty)
        - Consensus Points: [list]
        - Dissent Points: [list with attribution]
        - Key Assumptions: [list]
        """

        return await self._call_claude(synthesis_prompt, role="Synthesizer")
```

**결과 반영:**
```python
# EIMASResult 확장
@dataclass
class EIMASResult:
    # ... 기존 필드 ...

    # NEW: Multi-LLM Debate Results
    debate_transcript: List[Dict]           # 전체 토론 기록
    consensus_position: str                 # 합의된 입장
    consensus_confidence_range: Tuple[int, int]  # 신뢰도 범위 (예: 60-75%)
    dissent_points: List[Dict]              # 합의 못한 포인트
    model_contributions: Dict[str, str]     # 각 모델의 핵심 기여
```

---

#### Task 2.2: InterpretationDebateAgent 활성화
**파일:** `agents/interpretation_debate.py` (archive에서 복구)

**역할:** 경제학파별 관점 토론

```python
class InterpretationDebateAgent(BaseAgent):
    """
    경제학파별 해석 토론

    Schools:
    - Keynesian: 총수요, 재정정책 중시
    - Monetarist: 통화량, 인플레이션 중시
    - Austrian: 신용 사이클, 구조적 왜곡 중시
    - MMT: 국가 부채 지속가능성 관점
    """

    SCHOOLS = {
        'keynesian': {
            'focus': ['aggregate_demand', 'fiscal_policy', 'output_gap'],
            'prompt_modifier': "Focus on demand-side factors and fiscal stimulus effects"
        },
        'monetarist': {
            'focus': ['money_supply', 'inflation_expectations', 'velocity'],
            'prompt_modifier': "Focus on monetary aggregates and inflation dynamics"
        },
        'austrian': {
            'focus': ['credit_cycle', 'malinvestment', 'interest_rate_distortion'],
            'prompt_modifier': "Focus on credit-driven distortions and liquidation cycles"
        },
        'mmt': {
            'focus': ['sovereign_debt_capacity', 'sectoral_balances', 'job_guarantee'],
            'prompt_modifier': "Focus on fiscal space and debt sustainability"
        }
    }

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        market_data = request.market_data

        # 각 학파 관점에서 해석 생성
        interpretations = {}
        for school, config in self.SCHOOLS.items():
            interpretation = await self._generate_school_interpretation(
                school=school,
                config=config,
                market_data=market_data
            )
            interpretations[school] = interpretation

        # 학파 간 토론 시뮬레이션
        debate_result = await self._run_school_debate(interpretations)

        return AgentResponse(
            content={
                'interpretations': interpretations,
                'dominant_narrative': debate_result['dominant'],
                'minority_views': debate_result['minority'],
                'synthesis': debate_result['synthesis']
            }
        )
```

**결과 반영:**
```python
# Markdown 리포트에 추가
## Economic Interpretation

### Dominant Narrative (Monetarist)
Fed의 금리 정책이 핵심 동인. M2 성장률 둔화가 향후 6개월 내
디스인플레이션 압력으로 작용할 것.

### Minority View (Austrian)
신용 사이클 관점에서 현재 금리 수준은 여전히 부정적 실질금리.
자산 가격의 구조적 왜곡 지속 중.

### Synthesis
통화 긴축의 1차 효과는 확인되나, 구조적 불균형 해소에는
추가 시간 필요. 레짐 전환 리스크 상존.
```

---

#### Task 2.3: MethodologyDebateAgent 활성화
**파일:** `agents/methodology_debate.py` (archive에서 복구)

**역할:** 분석 방법론 선택의 투명성 확보

```python
class MethodologyDebateAgent(BaseAgent):
    """
    방법론 선택 토론

    Topics:
    - Feature Selection: LASSO vs Ridge vs Elastic Net
    - Regime Detection: GMM vs HMM vs Threshold
    - Risk Measure: VaR vs CVaR vs Drawdown
    - Causality: Granger vs Transfer Entropy vs PCMCI
    """

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        methodology_topic = request.context.get('methodology_topic', 'feature_selection')

        # 각 방법론의 장단점 분석
        options = await self._analyze_methodology_options(
            topic=methodology_topic,
            data_characteristics=request.context['data_characteristics']
        )

        # LLM 토론을 통한 선택
        selection = await self._debate_methodology_choice(options)

        return AgentResponse(
            content={
                'topic': methodology_topic,
                'options_analyzed': options,
                'selected_methodology': selection['choice'],
                'selection_rationale': selection['rationale'],
                'trade_offs_acknowledged': selection['trade_offs']
            }
        )
```

**결과 반영:**
```python
# 투명성 섹션 추가
## Methodology Transparency

### Feature Selection
- **Chosen:** LASSO (L1 Regularization)
- **Rationale:** 68개 변수 중 sparsity 필요, 해석 가능성 우선
- **Trade-off:** Ridge 대비 변수 제거가 과격할 수 있음
- **Alternatives Considered:** Ridge (rejected: 모든 변수 유지), Elastic Net (rejected: 추가 하이퍼파라미터 복잡성)

### Regime Detection
- **Chosen:** GMM 3-State
- **Rationale:** Bull/Neutral/Bear 명시적 분류, 확률 분포 제공
- **Trade-off:** HMM 대비 시간 의존성 미반영
```

---

### Phase 3: 결과 통합 및 Traceability (2-3일)

#### Task 3.1: EIMASResult 확장
**파일:** `core/schemas.py` 수정

```python
@dataclass
class EIMASResult:
    timestamp: str

    # ===== Phase 1: Data =====
    fred_summary: Dict
    market_data_count: int
    crypto_data_count: int

    # ===== Phase 2: Analysis =====
    regime: Dict
    risk_score: float
    events_detected: List[Dict]

    # ===== Phase 3: Agent Outputs (NEW) =====
    agent_outputs: AgentOutputs  # 새 데이터클래스

    # ===== Phase 4: Debate (NEW) =====
    debate_results: DebateResults  # 새 데이터클래스

    # ===== Phase 5: Verification (NEW) =====
    verification: VerificationResults  # 새 데이터클래스

    # ===== Final =====
    final_recommendation: str
    confidence: float
    confidence_range: Tuple[int, int]  # NEW: 범위로 표현
    reasoning_chain: List[str]  # NEW: 추론 과정 추적


@dataclass
class AgentOutputs:
    """각 에이전트의 출력 기록"""
    analysis: Dict          # AnalysisAgent 출력
    forecast: Dict          # ForecastAgent 출력
    research: Dict          # ResearchAgent 출력 (NEW)
    strategy: Dict          # StrategyAgent 출력 (NEW)
    interpretation: Dict    # InterpretationDebateAgent 출력 (NEW)
    methodology: Dict       # MethodologyDebateAgent 출력 (NEW)


@dataclass
class DebateResults:
    """Multi-LLM 토론 결과"""
    transcript: List[Dict]              # 전체 대화 기록
    consensus_position: str             # 합의 입장
    consensus_confidence: Tuple[int, int]  # 신뢰도 범위
    dissent_points: List[Dict]          # 불일치 포인트
    model_contributions: Dict[str, str] # 각 모델 기여


@dataclass
class VerificationResults:
    """검증 결과"""
    overall_reliability: float          # 0-100
    consistency_score: float            # 내부 일관성
    data_alignment_score: float         # 데이터 정합성
    bias_detected: List[str]            # 탐지된 편향
    warnings: List[str]                 # 경고 사항
```

---

#### Task 3.2: Reasoning Chain 구현
**파일:** `core/reasoning_chain.py` (신규)

```python
class ReasoningChain:
    """
    AI 의사결정 과정 추적

    목적:
    - 최종 권고가 어떤 근거로 도출되었는지 명시
    - 각 에이전트의 기여도 추적
    - 감사(Audit) 가능한 AI 시스템 구현
    """

    def __init__(self):
        self.chain = []

    def add_step(
        self,
        agent: str,
        input_summary: str,
        output_summary: str,
        confidence: float,
        key_factors: List[str]
    ):
        """추론 단계 추가"""
        self.chain.append({
            'step': len(self.chain) + 1,
            'agent': agent,
            'input': input_summary,
            'output': output_summary,
            'confidence': confidence,
            'key_factors': key_factors,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self) -> str:
        """추론 과정 요약"""
        summary_lines = ["## Reasoning Chain\n"]

        for step in self.chain:
            summary_lines.append(f"### Step {step['step']}: {step['agent']}")
            summary_lines.append(f"- **Input:** {step['input']}")
            summary_lines.append(f"- **Output:** {step['output']}")
            summary_lines.append(f"- **Confidence:** {step['confidence']}%")
            summary_lines.append(f"- **Key Factors:**")
            for factor in step['key_factors']:
                summary_lines.append(f"  - {factor}")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def to_dict(self) -> List[Dict]:
        """JSON 직렬화용"""
        return self.chain
```

**사용 예시:**
```python
# main.py에서
reasoning = ReasoningChain()

# Step 1: 데이터 수집
reasoning.add_step(
    agent="DataCollector",
    input_summary="FRED + yfinance + Binance APIs",
    output_summary="24 tickers, RRP=$5.2B, Net Liq=$5799B",
    confidence=100,
    key_factors=["API 응답 정상", "데이터 완전성 검증 통과"]
)

# Step 2: 분석
reasoning.add_step(
    agent="AnalysisAgent",
    input_summary="Market data + FRED summary",
    output_summary="Risk Score: 45.2/100, Regime: Bull (Low Vol)",
    confidence=78,
    key_factors=[
        "VIX 14.2 < 20 (Low Vol)",
        "Credit Spread 285bp < 300bp (Normal)",
        "Net Liquidity 상승 추세"
    ]
)

# Step 3: 토론
reasoning.add_step(
    agent="MultiLLMDebate",
    input_summary="Analysis results + Research context",
    output_summary="Consensus: BULLISH (65-75% confidence)",
    confidence=70,
    key_factors=[
        "Claude: 유동성 환경 긍정적 (BULLISH, 75%)",
        "GPT-4: 밸류에이션 부담 지적 (NEUTRAL, 55%)",
        "Gemini: 모멘텀 지속 전망 (BULLISH, 70%)",
        "Synthesis: 2/3 동의, 밸류에이션 리스크 인정"
    ]
)

# 최종 결과에 포함
result.reasoning_chain = reasoning.to_dict()
```

---

#### Task 3.3: Markdown Report 개선
**파일:** `main.py`의 `to_markdown()` 수정

```python
def to_markdown(self) -> str:
    """Enhanced markdown report with agent contributions"""

    md = f"""
# EIMAS Analysis Report
**Generated:** {self.timestamp}
**Version:** v2.2.0 (Multi-Agent Enhanced)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Final Recommendation** | {self.final_recommendation} |
| **Confidence** | {self.confidence_range[0]}-{self.confidence_range[1]}% |
| **Risk Level** | {self.risk_level} |
| **Regime** | {self.regime['regime']} ({self.regime['volatility']}) |

---

## Agent Contributions

### 1. Analysis Agent (Critical Path)
{self._format_agent_output(self.agent_outputs.analysis)}

### 2. Forecast Agent (LASSO)
{self._format_agent_output(self.agent_outputs.forecast)}

### 3. Research Agent (Perplexity)
{self._format_agent_output(self.agent_outputs.research)}

### 4. Strategy Agent (GC-HRP)
{self._format_agent_output(self.agent_outputs.strategy)}

---

## Multi-LLM Debate

### Participants
- **Claude (Economist):** {self.debate_results.model_contributions.get('claude', 'N/A')}
- **GPT-4 (Devil's Advocate):** {self.debate_results.model_contributions.get('gpt4', 'N/A')}
- **Gemini (Risk Manager):** {self.debate_results.model_contributions.get('gemini', 'N/A')}

### Consensus Points
{self._format_list(self.debate_results.consensus_points)}

### Dissent Points
{self._format_dissent(self.debate_results.dissent_points)}

---

## Economic Interpretation

### Dominant Narrative ({self.agent_outputs.interpretation['dominant_school']})
{self.agent_outputs.interpretation['dominant_narrative']}

### Minority Views
{self._format_minority_views(self.agent_outputs.interpretation['minority_views'])}

---

## Methodology Transparency

{self._format_methodology(self.agent_outputs.methodology)}

---

## Verification Report

| Check | Score | Status |
|-------|-------|--------|
| Internal Consistency | {self.verification.consistency_score:.1f}% | {self._score_emoji(self.verification.consistency_score)} |
| Data Alignment | {self.verification.data_alignment_score:.1f}% | {self._score_emoji(self.verification.data_alignment_score)} |
| Overall Reliability | {self.verification.overall_reliability:.1f}% | {self._score_emoji(self.verification.overall_reliability)} |

### Warnings
{self._format_list(self.verification.warnings) if self.verification.warnings else "None"}

---

## Reasoning Chain

{self._format_reasoning_chain(self.reasoning_chain)}

---

## Appendix: Raw Data

<details>
<summary>Click to expand</summary>

### FRED Summary
```json
{json.dumps(self.fred_summary, indent=2)}
```

### Debate Transcript
```json
{json.dumps(self.debate_results.transcript, indent=2)}
```

</details>

---
*Generated by EIMAS v2.2.0 | Multi-Agent Enhanced Edition*
"""
    return md
```

---

### Phase 4: 통합 테스트 및 검증 (2일)

#### Task 4.1: 통합 테스트
**파일:** `tests/test_multi_agent.py` (신규)

```python
import pytest
from agents.orchestrator import MetaOrchestrator
from core.multi_llm_debate import MultiLLMDebate

@pytest.mark.asyncio
async def test_full_agent_pipeline():
    """전체 에이전트 파이프라인 테스트"""
    orchestrator = MetaOrchestrator(verbose=True)

    # Mock market data
    market_data = {
        'SPY': {'price': 450, 'change': 0.5},
        'VIX': 14.2,
        'risk_score': 45.0
    }

    result = await orchestrator.run_full_pipeline(
        query="Current market outlook",
        market_data=market_data
    )

    # 모든 에이전트 출력 확인
    assert result.agent_outputs.analysis is not None
    assert result.agent_outputs.forecast is not None
    assert result.agent_outputs.research is not None
    assert result.agent_outputs.strategy is not None

    # 토론 결과 확인
    assert result.debate_results.consensus_position in ['BULLISH', 'BEARISH', 'NEUTRAL']
    assert len(result.debate_results.transcript) >= 3  # 최소 3라운드

    # 검증 결과 확인
    assert 0 <= result.verification.overall_reliability <= 100

    # Reasoning chain 확인
    assert len(result.reasoning_chain) >= 4  # 최소 4단계


@pytest.mark.asyncio
async def test_multi_llm_debate():
    """Multi-LLM 토론 테스트"""
    debate = MultiLLMDebate()

    result = await debate.run_debate(
        topic="Fed Policy Direction Q1 2025",
        context={
            'regime': 'Bull',
            'risk_score': 45,
            'key_metrics': {'VIX': 14.2, 'Credit_Spread': 285}
        }
    )

    # 합의 도출 확인
    assert result.consensus_position is not None
    assert result.consensus_confidence[0] <= result.consensus_confidence[1]

    # 모든 모델 참여 확인
    assert 'claude' in result.model_contributions
    assert 'gpt4' in result.model_contributions
    assert 'gemini' in result.model_contributions
```

---

## 4. 파일 변경 요약

| 파일 | 작업 | 상태 |
|------|------|------|
| `agents/orchestrator.py` | 7개 에이전트 + ReasoningChain + Step 4.5 통합 | ✅ 완료 |
| `agents/research_agent.py` | Archive에서 복구, form_opinion() → AgentOpinion | ✅ 완료 |
| `agents/strategy_agent.py` | Archive에서 복구, form_opinion() → AgentOpinion | ✅ 완료 |
| `agents/verification_agent.py` | Archive에서 복구, Step 6로 통합 | ✅ 완료 |
| `agents/interpretation_debate.py` | Archive에서 복구, Step 4.5에서 호출 | ✅ 완료 |
| `agents/methodology_debate.py` | Archive에서 복구, Step 4.5에서 호출 | ✅ 완료 |
| `core/multi_llm_debate.py` | Multi-LLM 토론 엔진 (Claude/GPT-4/Gemini) | ✅ 완료 |
| `core/reasoning_chain.py` | 추론 과정 추적 시스템 | ✅ 완료 |
| `core/schemas.py` | AgentOutputs, DebateResults, VerificationResults 추가 | ✅ 완료 |
| `pipeline/schemas.py` | to_markdown() Enhanced Debate 섹션 추가 | ✅ 완료 |
| `agents/__init__.py` | 7개 에이전트 export | ✅ 완료 |

---

## 5. 예상 일정

| Phase | 작업 | 기간 | 산출물 |
|-------|------|------|--------|
| **Phase 1** | Archive 에이전트 재활성화 | 1-2일 | 5개 에이전트 통합 |
| **Phase 2** | Multi-LLM Debate 구현 | 3-5일 | 실제 LLM 토론 엔진 |
| **Phase 3** | 결과 통합 및 Traceability | 2-3일 | 개선된 리포트 |
| **Phase 4** | 통합 테스트 | 2일 | 검증된 시스템 |
| **Total** | | **8-12일** | |

---

## 6. 성공 지표

1. **Agent Coverage:** 모든 8개 에이전트가 실행되고 결과에 기여
2. **LLM Utilization:** 최소 3개 LLM이 토론에 참여
3. **Traceability:** 최종 권고의 모든 근거가 추적 가능
4. **Verification:** 자동 팩트체킹으로 Hallucination 90% 이상 탐지
5. **Report Quality:** 투명한 방법론 + 경제학적 해석 포함

---

*Last Updated: 2026-01-28 (Phase 3 통합 완료)*
*Author: EIMAS Development Team*
