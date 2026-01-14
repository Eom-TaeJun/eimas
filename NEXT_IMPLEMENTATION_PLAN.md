# Next Implementation Plan

> Multi-AI 토론 활성화 및 Regime Change 파이프라인 통합 계획
>
> 작성일: 2025-12-27

---

## 1. Multi-AI 토론 활성화

### 1.1 현재 상태

```
현재: Mock 응답 반환 (AI API 호출 없음)
목표: 실제 Claude/OpenAI/Gemini API 호출
```

**현재 Mock 위치:**
| 파일 | 함수 | 상태 |
|-----|------|------|
| `pipeline/full_pipeline.py` | `_run_methodology_selection()` | Mock MethodologyDecision |
| `pipeline/full_pipeline.py` | `_run_interpretation()` | Mock InterpretationConsensus |
| `core/debate_framework.py` | `AIClient._call_api()` | 실제 API 호출 코드 있음 (미테스트) |

### 1.2 TODO

```
[ ] 1. API 키 검증 스크립트 작성
[ ] 2. AIClient 단일 호출 테스트
[ ] 3. DebateFramework 토론 라운드 테스트
[ ] 4. MethodologyDebate 실제 토론 테스트
[ ] 5. InterpretationDebate 실제 토론 테스트
[ ] 6. Pipeline에서 Mock → 실제 토론 전환
[ ] 7. 토론 품질 검증 및 프롬프트 튜닝
```

### 1.3 핵심 키워드

```
AIClient              # AI API 래퍼 클래스
AIProvider            # Claude/OpenAI/Gemini/Perplexity enum
DebateFramework       # 토론 프레임워크 기본 클래스
DebateParticipant     # 토론 참여자 (AI + role + system_prompt)
run_debate()          # 토론 실행 메인 함수
_collect_opinions()   # Round 1: 의견 수집
_run_critique_round() # Round 2: 상호 비판
_run_rebuttal_round() # Round 2.5: 반박
_reach_consensus()    # Round 3: 합의 도출
```

### 1.4 핵심 함수 및 위치

#### 1.4.1 AI Client (core/debate_framework.py)

```python
class AIClient:
    """AI API 통합 클라이언트"""

    async def _call_api(self, provider: AIProvider, prompt: str, system_prompt: str) -> str:
        """실제 API 호출"""
        # 위치: core/debate_framework.py:150-220

        if provider == AIProvider.CLAUDE:
            # Anthropic API 호출
            response = await self.claude.messages.create(...)

        elif provider == AIProvider.OPENAI:
            # OpenAI API 호출
            response = await self.openai.chat.completions.create(...)

        elif provider == AIProvider.GEMINI:
            # Gemini API 호출
            response = await self.gemini.generate_content_async(...)
```

#### 1.4.2 Debate Framework (core/debate_framework.py)

```python
class DebateFramework:
    """토론 프레임워크 기본 클래스"""

    async def run_debate(self, topic: str, context: Dict) -> DebateResult:
        """토론 실행 메인"""
        # 위치: core/debate_framework.py:280-350

        # Round 1: 의견 수집
        opinions = await self._collect_opinions(topic, context)

        # Round 2: 상호 비판
        critiques = await self._run_critique_round(opinions, context)

        # Round 2.5: 반박 (optional)
        if self.config.enable_rebuttal:
            rebuttals = await self._run_rebuttal_round(...)

        # Round 3: 합의 도출
        consensus = self.evaluate_consensus(opinions)

        return DebateResult(...)
```

#### 1.4.3 Pipeline 전환 (pipeline/full_pipeline.py)

```python
# 현재 (Mock):
async def _run_methodology_selection(self, ...):
    decision = MethodologyDecision(...)  # 하드코딩된 Mock
    return StageResult(result=decision, ...)

# 변경 후 (실제 토론):
async def _run_methodology_selection(self, ...):
    decision = await self.methodology.debate_methodology(
        research_question=research_question,
        research_goal=config.research_goal,
        data_summary=data_summary
    )
    return StageResult(result=decision, ...)
```

### 1.5 구현 단계

#### Step 1: API 검증 스크립트

```python
# tests/test_api_connection.py

async def test_all_apis():
    """모든 API 연결 테스트"""

    # Claude
    claude_ok = await test_claude_api()

    # OpenAI
    openai_ok = await test_openai_api()

    # Gemini
    gemini_ok = await test_gemini_api()

    # Perplexity
    perplexity_ok = await test_perplexity_api()

    return {
        "claude": claude_ok,
        "openai": openai_ok,
        "gemini": gemini_ok,
        "perplexity": perplexity_ok
    }
```

#### Step 2: 단일 AI 호출 테스트

```python
# AIClient 단독 테스트
client = AIClient()
response = await client.call(
    provider=AIProvider.CLAUDE,
    prompt="Fed 금리 인상의 영향을 분석해주세요.",
    system_prompt="당신은 경제학자입니다."
)
print(response)
```

#### Step 3: 토론 프레임워크 테스트

```python
# 간단한 토론 테스트
from core.debate_framework import DebateFramework, get_default_participants

debate = TestDebate(participants=get_default_participants())
result = await debate.run_debate(
    topic="2025년 Fed 금리 방향",
    context={"current_rate": 4.5}
)
print(result.summary)
```

#### Step 4: Pipeline 전환

```python
# pipeline/full_pipeline.py 수정

class FullPipelineRunner:
    def __init__(self, use_mock: bool = True):  # 기본값 Mock
        self.use_mock = use_mock

    async def _run_methodology_selection(self, ...):
        if self.use_mock:
            return self._mock_methodology_selection(...)
        else:
            return await self._real_methodology_selection(...)
```

### 1.6 예상 이슈 및 해결

| 이슈 | 해결 방안 |
|-----|---------|
| API 호출 시간 (느림) | 병렬 호출, 타임아웃 설정 |
| Rate Limit | 재시도 로직, 백오프 |
| 응답 파싱 실패 | JSON 추출 정규식 강화 |
| 일관성 없는 응답 | 프롬프트 튜닝, 예시 추가 |
| 비용 | Haiku/GPT-3.5 옵션 추가 |

---

## 2. Regime Change 파이프라인 통합

### 2.1 현재 상태

```
현재: RegimeChangeDetectionPipeline 독립 실행
목표: FullPipeline Stage 2.5로 통합
```

**Regime Change 파이프라인:**
```
거래량 급변 → 뉴스 검색 → 뉴스 분류 → AI 토론 → 레짐 결정
   Step 1       Step 2       Step 3      Step 4      Step 5
```

### 2.2 TODO

```
[ ] 1. RegimeChangeDetectionPipeline을 FullPipeline에서 호출 가능하게 래핑
[ ] 2. Stage 2.5 (Regime Check) 추가
[ ] 3. 레짐 변화 감지 시 분석 컨텍스트 조정 로직
[ ] 4. 레짐 변화 결과를 Interpretation 단계에 전달
[ ] 5. 테스트 및 검증
```

### 2.3 핵심 키워드

```
RegimeChangeDetectionPipeline  # 레짐 변화 탐지 통합 파이프라인
VolumeBreakoutDetector         # Step 1: 거래량 급변 탐지
NewsSearchAgent                # Step 2: 뉴스 검색 (Perplexity)
NewsClassificationAgent        # Step 3: 뉴스 분류 (Claude)
ImpactAssessmentDebate         # Step 4: 영향 평가 (Multi-AI)
RegimeChangeResult             # 최종 결과 데이터클래스
RegimeType                     # EXPANSION, CONTRACTION, TRANSITION
```

### 2.4 핵심 함수 및 위치

#### 2.4.1 Regime Change Pipeline (agents/regime_change.py)

```python
class RegimeChangeDetectionPipeline:
    """레짐 변화 탐지 통합 파이프라인"""

    async def run(self, ticker: str, data: pd.DataFrame, ...) -> List[RegimeChangeResult]:
        """전체 파이프라인 실행"""
        # 위치: agents/regime_change.py:650-750

        # Step 1: 거래량 급변 탐지
        volume_events = self.volume_detector.detect(data)

        # Step 2-5: 각 이벤트에 대해 분석
        for event in volume_events:
            # Step 2: 뉴스 검색
            news = await self.news_search.search(...)

            # Step 3: 뉴스 분류
            classified = await self.news_classifier.classify(...)

            # Step 4: 영향 평가 토론
            impact = await self.impact_debate.evaluate(...)

            # Step 5: 레짐 결정
            regime_result = self._determine_regime(...)
```

#### 2.4.2 Pipeline 통합 위치

```python
# pipeline/full_pipeline.py

class PipelineStage(Enum):
    DATA_COLLECTION = "data_collection"
    TOP_DOWN_ANALYSIS = "top_down_analysis"
    REGIME_CHECK = "regime_check"           # ⭐ NEW
    METHODOLOGY_SELECTION = "methodology_selection"
    # ...

async def run(self, ...):
    # Stage 1: Data Collection
    # Stage 2: Top-Down Analysis

    # Stage 2.5: Regime Check ⭐ NEW
    if PipelineStage.REGIME_CHECK not in config.skip_stages:
        stage_result = await self._run_regime_check(data, result.top_down)
        result.stages[PipelineStage.REGIME_CHECK] = stage_result
        result.regime_change = stage_result.result

        # 레짐 변화 시 컨텍스트 조정
        if result.regime_change and result.regime_change.is_regime_change:
            self._adjust_context_for_regime_change(result)

    # Stage 3: Methodology Selection
    # ...
```

### 2.5 통합 설계

#### 2.5.1 Stage 2.5: Regime Check

```python
async def _run_regime_check(
    self,
    data: Dict[str, Any],
    top_down: Optional[TopDownResult]
) -> StageResult:
    """Stage 2.5: 레짐 변화 확인"""

    # 가격/거래량 데이터 필요
    price_data = data.get('price_data')
    if price_data is None:
        return StageResult(
            stage=PipelineStage.REGIME_CHECK,
            status=PipelineStatus.SKIPPED,
            result=None,
            duration_seconds=0
        )

    # 레짐 변화 탐지 실행
    regime_pipeline = RegimeChangeDetectionPipeline()
    results = await regime_pipeline.run(
        ticker=data.get('ticker', 'SPY'),
        data=price_data,
        lookback_days=90
    )

    # 최근 레짐 변화 확인
    recent_change = self._find_recent_regime_change(results)

    return StageResult(
        stage=PipelineStage.REGIME_CHECK,
        status=PipelineStatus.COMPLETED,
        result=recent_change,
        duration_seconds=...
    )
```

#### 2.5.2 컨텍스트 조정 로직

```python
def _adjust_context_for_regime_change(self, result: PipelineResult):
    """레짐 변화 시 분석 컨텍스트 조정"""

    regime = result.regime_change

    if regime.after_regime == RegimeType.CONTRACTION:
        # 수축기 진입: 보수적 분석
        result.context_adjustment = {
            "risk_tolerance": "conservative",
            "methodology_bias": "defensive",
            "strategy_bias": "risk_off"
        }

    elif regime.after_regime == RegimeType.EXPANSION:
        # 확장기 진입: 공격적 분석
        result.context_adjustment = {
            "risk_tolerance": "aggressive",
            "methodology_bias": "growth",
            "strategy_bias": "risk_on"
        }

    elif regime.after_regime == RegimeType.TRANSITION:
        # 전환기: 관망
        result.context_adjustment = {
            "risk_tolerance": "neutral",
            "methodology_bias": "balanced",
            "strategy_bias": "wait_and_see"
        }
```

#### 2.5.3 Interpretation 단계 연동

```python
async def _run_interpretation(self, ...):
    """Stage 5: 경제학파별 해석"""

    # 레짐 변화 정보를 컨텍스트에 추가
    context = {
        'analysis_result': analysis_result,
        'regime_change': result.regime_change,  # ⭐ 추가
        'context_adjustment': result.context_adjustment  # ⭐ 추가
    }

    # 해석 토론 시 레짐 변화 고려
    consensus = await self.interpretation.interpret_results(
        analysis_result=analysis_result,
        additional_context=context
    )
```

### 2.6 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                    REGIME-AWARE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Data Collection                                        │
│  └── price_data, volume_data 포함                                │
│           ↓                                                      │
│  Stage 2: Top-Down Analysis                                      │
│  └── L0~L3 하향식 분석                                           │
│           ↓                                                      │
│  Stage 2.5: Regime Check ⭐                                       │
│  ├── VolumeBreakoutDetector (3σ 급변 탐지)                       │
│  ├── NewsSearchAgent (Perplexity)                                │
│  ├── NewsClassificationAgent (Claude)                            │
│  ├── ImpactAssessmentDebate (Multi-AI)                           │
│  └── RegimeChangeResult                                          │
│           ↓                                                      │
│       ┌───────────────────────────────────────┐                  │
│       │ 레짐 변화 감지?                        │                  │
│       ├── YES → 컨텍스트 조정                  │                  │
│       │         (risk_tolerance, bias 등)     │                  │
│       └── NO → 그대로 진행                     │                  │
│       └───────────────────────────────────────┘                  │
│           ↓                                                      │
│  Stage 3: Methodology Selection                                  │
│  └── 레짐 변화 고려한 방법론 선택                                │
│           ↓                                                      │
│  Stage 4: Core Analysis                                          │
│           ↓                                                      │
│  Stage 5: Interpretation                                         │
│  └── 레짐 변화 정보 포함하여 해석                                │
│           ↓                                                      │
│  Stage 6: Strategy                                               │
│  └── 레짐 변화 반영한 전략                                       │
│           ↓                                                      │
│  Stage 7: Synthesis                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.7 PipelineResult 확장

```python
@dataclass
class PipelineResult:
    # 기존 필드
    stages: Dict[PipelineStage, StageResult]
    top_down: Optional[TopDownResult] = None
    methodology: Optional[MethodologyDecision] = None
    # ...

    # 추가 필드 ⭐
    regime_change: Optional[RegimeChangeResult] = None
    context_adjustment: Optional[Dict[str, str]] = None
    regime_aware: bool = False  # 레짐 변화 반영 여부
```

---

## 3. 통합 구현 순서

### Phase 1: API 활성화 (1-2시간)

```
1. tests/test_api_connection.py 작성
2. 각 API 단독 테스트
3. AIClient 통합 테스트
4. DebateFramework 토론 테스트
```

### Phase 2: Pipeline 전환 (1시간)

```
1. FullPipelineRunner에 use_mock 플래그 추가
2. Mock/Real 분기 로직 구현
3. 실제 토론으로 전체 파이프라인 테스트
```

### Phase 3: Regime Change 통합 (1-2시간)

```
1. PipelineStage.REGIME_CHECK 추가
2. _run_regime_check() 구현
3. 컨텍스트 조정 로직 구현
4. Interpretation 단계 연동
5. 통합 테스트
```

### Phase 4: 검증 및 튜닝 (1시간)

```
1. 전체 파이프라인 E2E 테스트
2. 토론 품질 검증
3. 프롬프트 튜닝
4. 문서 업데이트
```

---

## 4. 예상 결과물

### 4.1 파일 변경

| 파일 | 변경 내용 |
|-----|---------|
| `tests/test_api_connection.py` | 신규 생성 |
| `pipeline/full_pipeline.py` | use_mock 플래그, REGIME_CHECK 추가 |
| `agents/regime_change.py` | Pipeline 연동 인터페이스 |
| `IMPLEMENTATION_PROGRESS.md` | 진행 상황 업데이트 |

### 4.2 새로운 기능

```python
# 1. Mock/Real 전환
runner = FullPipelineRunner(use_mock=False)  # 실제 AI 토론

# 2. Regime-aware 분석
config = PipelineConfig(
    enable_regime_check=True,  # 레짐 변화 확인
    price_data=df  # 가격/거래량 데이터
)

# 3. 결과 접근
result.regime_change  # RegimeChangeResult
result.context_adjustment  # {"risk_tolerance": "conservative", ...}
result.regime_aware  # True
```

---

## 5. 체크리스트

### API 활성화

- [ ] API 키 환경변수 확인 (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`)
- [ ] test_api_connection.py 작성
- [ ] Claude API 테스트
- [ ] OpenAI API 테스트
- [ ] Gemini API 테스트
- [ ] AIClient 통합 테스트
- [ ] DebateFramework 토론 테스트
- [ ] Pipeline use_mock 플래그 추가
- [ ] 실제 토론 파이프라인 테스트

### Regime Change 통합

- [ ] PipelineStage.REGIME_CHECK 추가
- [ ] _run_regime_check() 구현
- [ ] _adjust_context_for_regime_change() 구현
- [ ] PipelineResult에 regime_change 필드 추가
- [ ] Interpretation 단계 연동
- [ ] 통합 테스트
- [ ] 문서 업데이트

---

*작성: Claude Code*
*마지막 업데이트: 2025-12-27*
