# Economic AI Agent System (EAAS)

> **목표**: 경제학 연구 전 과정에 AI 에이전트가 협업하는 시스템
> **핵심 철학**: 서치 → 방법론 토론 → 실행 → 결과 해석 → 종합, 모든 단계에서 Multi-AI 토론

---

## 1. 시스템 개요

### 1.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Query (연구 주제)                           │
│                    예: "2025년 Fed 금리 전망 분석"                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT (Claude)                         │
│         주제 분해 / 에이전트 조율 / 워크플로우 결정 / 품질 검증            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   PHASE 1     │          │   PHASE 2     │          │   PHASE 3     │
│   Research    │    →     │  Methodology  │    →     │   Execution   │
│   + Data      │          │   Debate      │          │   + Analysis  │
└───────────────┘          └───────────────┘          └───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│   PHASE 4     │          │   PHASE 5     │          │   PHASE 6     │
│  Interpretation│   →     │   Synthesis   │    →     │    Output     │
│   Debate      │          │   + Report    │          │  Dashboard    │
└───────────────┘          └───────────────┘          └───────────────┘
```

### 1.2 참여 AI 모델 역할 분담

| AI Model | 주요 역할 | 강점 활용 |
|----------|----------|----------|
| **Perplexity** | Research Agent | 실시간 웹 검색, 최신 뉴스/논문 |
| **Claude** | Orchestrator, Synthesis | 복잡한 추론, 긴 컨텍스트, 코드 생성 |
| **OpenAI (GPT-4)** | Methodology Debate | 다양한 관점, 창의적 제안 |
| **Gemini** | Data Analysis, Visualization | 대용량 데이터 처리, 멀티모달 |

---

## 2. Phase별 상세 설계

### 2.1 PHASE 1: Research + Data Collection

#### 2.1.1 Research Agent (Perplexity)

```python
class ResearchAgent:
    """
    역할: 연구 주제 관련 최신 정보 수집
    도구: Perplexity API (실시간 웹 검색)
    """

    async def execute(self, query: str) -> ResearchReport:
        tasks = [
            self.search_news(query),           # 최신 뉴스
            self.search_fed_communications(),   # Fed 발언/회의록
            self.search_academic_papers(),      # 관련 논문
            self.search_market_reports(),       # 투자은행 리포트
        ]
        results = await asyncio.gather(*tasks)
        return self.compile_report(results)
```

**수집 대상**:
| 카테고리 | 소스 | 예시 |
|---------|------|------|
| Fed 정책 | FOMC 회의록, Fed 발언 | "Powell 12월 발언 요약" |
| 시장 뉴스 | Bloomberg, Reuters | "금리 선물 시장 동향" |
| 논문 | NBER, Fed Working Papers | "Yield Curve Inversion" |
| 리포트 | Goldman, JPM Research | "2025 Outlook" |

#### 2.1.2 Data Agent (Python + APIs)

```python
class DataAgent:
    """
    역할: 시장 데이터 수집 및 전처리
    도구: yfinance, FRED API, CME FedWatch
    """

    async def collect(self, config: DataConfig) -> ProcessedData:
        # 1. 병렬 데이터 수집
        yahoo_data = await self.fetch_yahoo(config.tickers)
        fred_data = await self.fetch_fred(config.indicators)
        cme_data = await self.fetch_cme_fedwatch()

        # 2. 변수 변환
        features = self.transform_variables(yahoo_data, fred_data)
        # Ret_* : 로그 수익률
        # d_*   : 일간 차분

        # 3. 이벤트 태깅
        features = self.tag_events(features, config.events)

        # 4. Treasury 제외 (Simultaneity 문제)
        features = self.exclude_endogenous(features)

        return ProcessedData(features, cme_data)
```

**데이터 소스**:
```
┌─────────────────────────────────────────────────────────────┐
│  yfinance                                                    │
│  ├── Equity: SPY, QQQ, IWM, VTI                             │
│  ├── Bonds: TLT, IEF, HYG, LQD                              │
│  ├── Commodities: GLD, SLV, USO, COPX                       │
│  ├── FX: DXY, USDKRW, USDJPY                                │
│  └── Volatility: VIX                                         │
│                                                              │
│  FRED API                                                    │
│  ├── Rates: DGS2, DGS10, FEDFUNDS                           │
│  ├── Spreads: BAA, AAA (계산: BAA-DGS10)                    │
│  ├── Inflation: T5YIE, T10YIE, CPIAUCSL                     │
│  └── Economic: UNRATE, PAYEMS, INDPRO                       │
│                                                              │
│  CME FedWatch                                                │
│  ├── exp_rate_bp: 기대 금리 (bp)                            │
│  ├── days_to_meeting: FOMC까지 남은 일수                    │
│  └── rate_uncertainty: 불확실성 지수                         │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.2 PHASE 2: Methodology Debate (Multi-AI 토론)

> **핵심**: 어떤 방법론을 쓸지 AI들이 토론하여 결정

#### 2.2.1 토론 참여자

```
┌─────────────────────────────────────────────────────────────┐
│                  METHODOLOGY DEBATE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Claude     │  │  OpenAI     │  │  Gemini     │         │
│  │             │  │             │  │             │         │
│  │ "LASSO가    │  │ "VAR/IRF로 │  │ "ML 앙상블  │         │
│  │  변수 선택에│  │  동적 관계를│  │  로 예측    │         │
│  │  적합"      │  │  봐야 한다" │  │  정확도를"  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                 ┌───────────────┐                           │
│                 │   CONSENSUS   │                           │
│                 │  또는 HYBRID  │                           │
│                 └───────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 방법론 옵션

| 방법론 | 용도 | 장점 | 단점 |
|--------|------|------|------|
| **LASSO** | 변수 선택 | Sparsity, 핵심 변수 식별 | 비선형 관계 미포착 |
| **Post-LASSO OLS** | 통계 추론 | p-value, 신뢰구간 제공 | LASSO 선택에 의존 |
| **VAR/IRF** | 동적 관계 | 충격반응 분석, 예측 | 변수 많으면 불안정 |
| **Granger Causality** | 인과성 | 선행 관계 검정 | 상관 ≠ 인과 |
| **GARCH** | 변동성 | 클러스터링 포착 | 복잡성 |
| **ML Ensemble** | 예측 | 높은 정확도 | 해석 어려움 |

#### 2.2.3 토론 프로토콜

```python
class MethodologyDebate:
    """
    Multi-AI 방법론 토론
    """

    async def run_debate(
        self,
        research_report: ResearchReport,
        data_summary: DataSummary,
        user_goal: str
    ) -> MethodologyDecision:

        # Round 1: 각 AI가 방법론 제안
        proposals = await asyncio.gather(
            self.claude.propose_methodology(context),
            self.openai.propose_methodology(context),
            self.gemini.propose_methodology(context),
        )

        # Round 2: 상호 비판 및 반박
        critiques = await self.cross_critique(proposals)

        # Round 3: 합의 도출 또는 하이브리드 제안
        if self.check_consensus(proposals):
            return MethodologyDecision(
                method=proposals[0].method,
                confidence="HIGH",
                rationale=self.merge_rationales(proposals)
            )
        else:
            return MethodologyDecision(
                method="HYBRID",
                components=[p.method for p in proposals],
                confidence="MEDIUM",
                rationale="다양한 관점 통합 필요"
            )
```

#### 2.2.4 토론 예시

**연구 질문**: "2025년 Fed 금리 변화를 예측하는 핵심 변수는?"

```yaml
Round 1 - 제안:
  Claude:
    method: "LASSO + Post-LASSO OLS"
    rationale: |
      - 변수가 50개 이상으로 많음
      - 핵심 변수 선택이 목표
      - HAC 표준오차로 시계열 특성 보정
      - Treasury 제외로 simultaneity 방지

  OpenAI:
    method: "VAR with Granger Causality"
    rationale: |
      - 동적 관계 분석이 중요
      - IRF로 충격 전파 경로 파악
      - 변수 간 상호작용 포착

  Gemini:
    method: "Gradient Boosting + SHAP"
    rationale: |
      - 비선형 관계 포착
      - SHAP으로 변수 중요도 해석
      - 예측 정확도 우선

Round 2 - 비판:
  Claude → OpenAI: "변수가 너무 많으면 VAR 불안정"
  OpenAI → Claude: "LASSO는 동적 관계 미포착"
  Gemini → Both: "선형 모델의 한계"
  Claude → Gemini: "ML은 경제학적 해석 어려움"

Round 3 - 합의:
  decision: "HYBRID"
  pipeline:
    1. LASSO로 변수 선택 (20개 → 5-7개)
    2. 선택된 변수로 VAR 추정
    3. IRF로 동적 관계 분석
    4. GB로 예측, SHAP으로 검증
  rationale: "각 방법론의 강점 결합"
```

---

### 2.3 PHASE 3: Execution + Analysis

#### 2.3.1 Execution Agent

```python
class ExecutionAgent:
    """
    역할: 선택된 방법론 실행
    """

    async def execute(
        self,
        data: ProcessedData,
        methodology: MethodologyDecision
    ) -> AnalysisResult:

        results = {}

        # 방법론별 실행
        if "LASSO" in methodology.components:
            results['lasso'] = self.run_lasso(data)
            results['post_lasso_ols'] = self.run_post_lasso_ols(
                data,
                results['lasso'].selected_vars
            )

        if "VAR" in methodology.components:
            results['var'] = self.run_var(data)
            results['irf'] = self.calculate_irf(results['var'])
            results['granger'] = self.granger_causality(data)

        if "ML" in methodology.components:
            results['ml'] = self.run_ml_ensemble(data)
            results['shap'] = self.calculate_shap(results['ml'])

        # Horizon별 분리 분석
        results['by_horizon'] = {
            'very_short': self.analyze_horizon(data, days<=30),
            'short': self.analyze_horizon(data, 30<days<=90),
            'long': self.analyze_horizon(data, days>=180),
        }

        return AnalysisResult(results)
```

#### 2.3.2 실행 결과 구조

```
AnalysisResult
├── lasso_result
│   ├── selected_vars: ["d_Spread_HighYield", "Ret_Dollar_Idx", ...]
│   ├── coefficients: {var: coef, ...}
│   └── r_squared: 0.72
│
├── post_lasso_ols
│   ├── coefficients: {var: (coef, std_err, p_value), ...}
│   ├── hac_adjusted: True
│   └── significant_vars: ["d_Spread_HighYield", ...]
│
├── var_result
│   ├── lag_order: 3
│   ├── coefficients: matrix
│   └── residuals: DataFrame
│
├── irf_result
│   ├── impulse: "d_Exp_Rate"
│   ├── responses: {var: [t0, t1, t2, ...], ...}
│   └── cumulative: {var: [cum_t0, ...], ...}
│
├── granger_result
│   ├── significant_pairs: [("X", "Y", p_value), ...]
│   └── network: adjacency_matrix
│
└── by_horizon
    ├── very_short: {...}  # VIX, 이벤트 더미 중요
    ├── short: {...}       # 크레딧, FX 중요
    └── long: {...}        # 인플레 기대 중요
```

---

### 2.4 PHASE 4: Interpretation Debate (경제학파 토론)

> **핵심**: 분석 결과를 다양한 경제학적 관점에서 해석

#### 2.4.1 경제학파 에이전트

```
┌─────────────────────────────────────────────────────────────┐
│                INTERPRETATION DEBATE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Monetarist  │  │ Keynesian   │  │ Austrian    │         │
│  │ (Claude)    │  │ (OpenAI)    │  │ (Gemini)    │         │
│  │             │  │             │  │             │         │
│  │ "M↔P 관계  │  │ "총수요     │  │ "저금리가   │         │
│  │  가 핵심,   │  │  부양이     │  │  버블을     │         │
│  │  인플레     │  │  필요하다"  │  │  만든다"    │         │
│  │  불가피"    │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│  ┌─────────────┐                                            │
│  │ Technical   │  (Perplexity - 최신 시장 시그널)           │
│  │ "Net Buy    │                                            │
│  │  5x 이상,   │                                            │
│  │  매수 신호" │                                            │
│  └─────────────┘                                            │
│                                                              │
│                    ↓ Evidence-Based Debate ↓                │
│                                                              │
│                 ┌───────────────────┐                       │
│                 │    CONSENSUS      │                       │
│                 │  + Dissent Points │                       │
│                 └───────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 학파별 해석 프레임워크

```python
class MonetaristAgent:
    """
    통화주의 관점 (Friedman)
    핵심: M↔P, 통화 중립성, 인플레는 화폐 현상
    """

    framework = """
    MV = PY (화폐수량설)

    단기: M↑ → r↓ → I↑ → Y↑
    장기: M↑ → P↑ (실질 변수 불변)

    분석 포인트:
    - M2 증가율 vs CPI
    - 통화정책 래그 (12-18개월)
    - 기대 인플레이션
    """

    def interpret(self, results: AnalysisResult) -> Opinion:
        # M2, CPI 관계 확인
        m_p_relation = results.irf['M2']['CPI']

        if m_p_relation > threshold:
            return Opinion(
                position="INFLATIONARY",
                evidence=[f"M2→CPI IRF: {m_p_relation}"],
                recommendation="인플레 헤지 자산 선호"
            )


class KeynesianAgent:
    """
    케인즈주의 관점
    핵심: Y = C + I + G + NX, 승수효과, 유동성 함정
    """

    framework = """
    총수요 관리가 핵심

    경기침체 시: G↑ → Y↑ (승수효과)
    유동성 함정: r→0 이면 통화정책 무력

    분석 포인트:
    - 소비/투자 트렌드
    - 재정 승수 추정
    - 산출갭
    """

    def interpret(self, results: AnalysisResult) -> Opinion:
        consumption_trend = results.get_trend('Ret_Consumer')
        investment_trend = results.get_trend('Ret_Industrial')

        if consumption_trend < 0 and investment_trend < 0:
            return Opinion(
                position="STIMULATE",
                evidence=["소비/투자 모두 하락 추세"],
                recommendation="재정 확대 필요"
            )


class AustrianAgent:
    """
    오스트리아 학파 관점 (Mises, Hayek)
    핵심: 인위적 저금리 → 자본 오배분 → 버블 → 붕괴
    """

    framework = """
    Austrian Business Cycle Theory

    저금리 → 과잉투자 → 버블 형성 → 청산 불가피

    분석 포인트:
    - 실질금리 vs 자연이자율
    - 신용 팽창률
    - 금/은 가격 (fiat 불신)
    - 자산 가격 버블 징후
    """

    def interpret(self, results: AnalysisResult) -> Opinion:
        gold_trend = results.get_trend('Ret_GLD')
        credit_expansion = results.get_credit_growth()

        if gold_trend > 0 and credit_expansion > threshold:
            return Opinion(
                position="BUBBLE_WARNING",
                evidence=["금 상승 + 신용 팽창"],
                recommendation="실물 자산 선호, 레버리지 축소"
            )


class TechnicalAgent:
    """
    기술적 분석 + 시그널 포착 (Perplexity)
    핵심: 시장 데이터 기반 시그널
    """

    def interpret(self, results: AnalysisResult) -> Opinion:
        net_buy_ratio = results.get_net_buy_ratio()
        volume_spike = results.get_volume_spike()

        if net_buy_ratio >= 5:
            return Opinion(
                position="STRONG_BUY",
                evidence=[f"Net Buy Ratio: {net_buy_ratio}x"],
                recommendation="매수 진입 고려"
            )
```

#### 2.4.3 토론 프로토콜

```python
class InterpretationDebate:
    """
    경제학파 간 토론
    """

    async def run_debate(
        self,
        analysis_result: AnalysisResult
    ) -> DebateResult:

        # 1. 각 학파 의견 수집
        opinions = await asyncio.gather(
            self.monetarist.interpret(analysis_result),
            self.keynesian.interpret(analysis_result),
            self.austrian.interpret(analysis_result),
            self.technical.interpret(analysis_result),
        )

        # 2. 합의점 찾기
        consensus_points = self.find_consensus(opinions)

        # 3. 분열점 기록
        dissent_points = self.find_dissent(opinions)

        # 4. 증거 가중치 평가
        weighted_conclusion = self.weight_by_evidence(opinions)

        return DebateResult(
            consensus=consensus_points,
            dissent=dissent_points,
            conclusion=weighted_conclusion,
            confidence=self.calculate_confidence(opinions)
        )
```

---

### 2.5 PHASE 5: Synthesis + Report

#### 2.5.1 Synthesis Agent (Claude)

```python
class SynthesisAgent:
    """
    역할: 모든 결과를 종합하여 최종 보고서 생성
    """

    async def synthesize(
        self,
        research: ResearchReport,
        methodology: MethodologyDecision,
        analysis: AnalysisResult,
        debate: DebateResult
    ) -> FinalReport:

        report = FinalReport()

        # 1. Executive Summary
        report.summary = self.generate_summary(
            key_findings=analysis.key_findings,
            consensus=debate.consensus,
            recommendation=debate.conclusion
        )

        # 2. 연구 배경
        report.background = research.compile()

        # 3. 방법론 설명
        report.methodology = methodology.explain()

        # 4. 분석 결과
        report.results = self.format_results(analysis)

        # 5. 경제학적 해석
        report.interpretation = self.format_debate(debate)

        # 6. 투자 전략 제안
        report.strategy = self.generate_strategy(
            analysis=analysis,
            debate=debate
        )

        # 7. 리스크 요인
        report.risks = self.identify_risks(debate.dissent)

        return report
```

#### 2.5.2 보고서 구조

```markdown
# Economic Analysis Report
## [연구 주제]

### Executive Summary
- 핵심 발견 3줄 요약
- 투자 권고
- 신뢰도 수준

### 1. 연구 배경
- 시장 현황 (Research Agent)
- 최신 뉴스/이벤트

### 2. 데이터 및 방법론
- 사용 데이터
- 선택된 방법론 (Methodology Debate 결과)
- 방법론 선택 이유

### 3. 분석 결과
#### 3.1 LASSO 결과
- 선택된 변수
- 계수 및 유의성

#### 3.2 VAR/IRF 결과
- 충격반응함수 그래프
- 동적 관계 해석

#### 3.3 Horizon별 분석
- 초단기/단기/장기 차이

### 4. 경제학적 해석 (Debate 결과)
#### 4.1 합의점
- 모든 학파가 동의하는 부분

#### 4.2 분열점
- 학파별 다른 해석
- 각 관점의 근거

### 5. 투자 전략
- 추천 포지션
- 진입/청산 기준
- 리스크 관리

### 6. 리스크 요인
- 분석의 한계
- 주의해야 할 시나리오

### Appendix
- 상세 통계표
- 코드/데이터 참조
```

---

### 2.6 PHASE 6: Visualization + Dashboard

#### 2.6.1 Visualization Agent (Gemini)

```python
class VisualizationAgent:
    """
    역할: 분석 결과 시각화 및 대시보드 생성
    """

    def generate_dashboard(
        self,
        report: FinalReport
    ) -> Dashboard:

        charts = []

        # 1. IRF 그래프
        charts.append(self.plot_irf(report.analysis.irf))

        # 2. Critical Path 네트워크
        charts.append(self.plot_network(report.analysis.granger))

        # 3. Horizon별 변수 중요도
        charts.append(self.plot_variable_importance(
            report.analysis.by_horizon
        ))

        # 4. 학파별 의견 비교
        charts.append(self.plot_debate_summary(report.debate))

        # 5. 시계열 예측
        charts.append(self.plot_forecast(report.analysis.forecast))

        return Dashboard(
            charts=charts,
            summary=report.summary,
            interactive=True
        )
```

---

## 3. 워크플로우 유형

### 3.1 Type A: Quick Analysis (빠른 분석)

```
User Query
    │
    ├── Research Agent (Perplexity) ──┐
    │                                  │
    └── Data Agent ───────────────────┼── Execution Agent
                                       │         │
                                       │         ▼
                                       └── Synthesis Agent
                                                 │
                                                 ▼
                                            Quick Report

시간: 5-10분
토론: 생략
용도: 빠른 시장 체크
```

### 3.2 Type B: Standard Research (표준 연구)

```
User Query
    │
    ├── Research Agent ───┐
    │                      │
    └── Data Agent ───────┼── Methodology Debate (3 AI)
                          │           │
                          │           ▼
                          └── Execution Agent
                                      │
                                      ▼
                          Interpretation Debate (4 School)
                                      │
                                      ▼
                              Synthesis Agent
                                      │
                                      ▼
                              Full Report + Dashboard

시간: 30분-1시간
토론: 방법론 + 해석
용도: 정기 분석 리포트
```

### 3.3 Type C: Deep Research (심층 연구)

```
User Query
    │
    ▼
Orchestrator ── 주제 분해 ── Sub-Query 1, 2, 3...
    │
    ├── [Sub-Query 1] ── Full Pipeline ──┐
    ├── [Sub-Query 2] ── Full Pipeline ──┼── Integration Debate
    └── [Sub-Query 3] ── Full Pipeline ──┘          │
                                                    ▼
                                           Meta-Synthesis
                                                    │
                                                    ▼
                                         Comprehensive Report

시간: 2-4시간
토론: 다단계
용도: 분기별 심층 리포트, 논문
```

### 3.4 Type D: Real-time Alert (실시간 알림)

```
Streaming Data
    │
    ▼
Signal Detector (Rule-based)
    │
    ├── Net Buy Ratio > 5x ──┐
    ├── Volume Spike > 2σ ───┼── Alert Generator
    └── VIX Spike > 20% ─────┘        │
                                      ▼
                               Quick Interpretation
                                      │
                                      ▼
                               Push Notification

시간: 실시간
토론: 생략 (사후 분석 별도)
용도: 시장 모니터링
```

---

## 4. Critical Path Discovery & Application

> **핵심**: 크리티컬 패스를 발견하고, 이를 통해 현재 상태 진단 → 충격 대비 → 매수/매도 추천까지 연결

### 4.1 크리티컬 패스 발견 프로세스

#### 4.1.1 발견 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL PATH DISCOVERY PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: 데이터 수집                                                      │
│  ├── 시장 데이터 (yfinance): 주가, 원자재, 환율, VIX                      │
│  ├── 경제 지표 (FRED): 금리, 스프레드, 인플레                             │
│  └── 실시간 데이터: 체결강도, Net Buy Ratio                               │
│                         │                                                │
│                         ▼                                                │
│  STEP 2: 관계 분석 (Granger Causality)                                   │
│  ├── 모든 변수 쌍에 대해 Granger 검정                                     │
│  ├── p < 0.05 → 유의미한 선행 관계                                       │
│  └── Lead time 측정 (며칠 선행하는가?)                                   │
│                         │                                                │
│                         ▼                                                │
│  STEP 3: 네트워크 구축                                                    │
│  ├── Node: 경제 변수                                                     │
│  ├── Edge: 유의미한 선행 관계                                            │
│  ├── Weight: 영향력 크기 (β 계수)                                        │
│  └── Lead Time: 선행 일수                                                │
│                         │                                                │
│                         ▼                                                │
│  STEP 4: 크리티컬 패스 추출                                               │
│  ├── 타겟 변수 정의 (Fed Rate, SPY, 등)                                  │
│  ├── 역추적: 타겟 → 선행 변수 → 선행의 선행...                           │
│  └── 가장 영향력 큰 경로 = Critical Path                                 │
│                         │                                                │
│                         ▼                                                │
│  STEP 5: 검증 및 업데이트                                                 │
│  ├── Out-of-sample 검증                                                  │
│  ├── Rolling window로 안정성 확인                                        │
│  └── 주기적 업데이트 (시장 구조 변화 반영)                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 발견된 크리티컬 패스 예시

```yaml
# 크리티컬 패스 저장소
critical_paths:

  # Path 1: Fed 금리 예측
  fed_rate_path:
    target: "Fed_Rate_Decision"
    path:
      - variable: "Gold_Silver"
        lead_time: 30  # 30일 선행
        weight: 0.72
        direction: "inverse"  # 금↑ → 금리↓ 기대
      - variable: "Dollar_Index"
        lead_time: 20
        weight: 0.65
        direction: "positive"  # 달러↓ → 금리↓ 기대
      - variable: "HY_Spread"
        lead_time: 15
        weight: 0.58
        direction: "inverse"  # 스프레드↑ → 금리↓ 기대
    confidence: 0.78
    last_validated: "2025-12-20"

  # Path 2: 주식시장 방향
  equity_path:
    target: "SPY_Direction"
    path:
      - variable: "VIX"
        lead_time: 5
        weight: -0.81
        direction: "inverse"
      - variable: "HY_Spread"
        lead_time: 10
        weight: -0.67
        direction: "inverse"
      - variable: "Copper_Gold_Ratio"
        lead_time: 15
        weight: 0.54
        direction: "positive"
    confidence: 0.72

  # Path 3: 인플레이션 예측
  inflation_path:
    target: "CPI_Direction"
    path:
      - variable: "M2_Growth"
        lead_time: 365  # 12-18개월
        weight: 0.68
        direction: "positive"
      - variable: "Oil_Price"
        lead_time: 60
        weight: 0.55
        direction: "positive"
      - variable: "Breakeven_5Y"
        lead_time: 30
        weight: 0.71
        direction: "positive"
    confidence: 0.65

  # Path 4: 경기 사이클
  business_cycle_path:
    target: "GDP_Growth"
    path:
      - variable: "Yield_Curve"
        lead_time: 365  # 12-18개월
        weight: 0.73
        direction: "positive"  # 역전 → 침체
      - variable: "Copper_Price"
        lead_time: 90
        weight: 0.61
        direction: "positive"
      - variable: "Initial_Claims"
        lead_time: 30
        weight: -0.58
        direction: "inverse"
    confidence: 0.70
```

---

### 4.2 현재 상태 진단 (State Diagnosis)

#### 4.2.1 진단 프레임워크

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       STATE DIAGNOSIS FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  각 크리티컬 패스의 현재 상태를 점검하여 시장 국면 진단                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PATH 1: Fed 금리 방향                                           │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │    │
│  │  │ Gold/Slv │ →  │ Dollar   │ →  │ HY Sprd  │ →  [Fed Rate]    │    │
│  │  │ ↑ +5.2%  │    │ ↓ -2.1%  │    │ ↓ -15bp  │                   │    │
│  │  │ BULLISH  │    │ BEARISH  │    │ RISK-ON  │                   │    │
│  │  └──────────┘    └──────────┘    └──────────┘                   │    │
│  │                                                                  │    │
│  │  진단: 3/3 지표가 "금리 인하" 방향 → DOVISH 신호 (신뢰도: HIGH)  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PATH 2: 주식시장 방향                                           │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │    │
│  │  │ VIX      │ →  │ HY Sprd  │ →  │ Cu/Au    │ →  [SPY]         │    │
│  │  │ 14.5     │    │ 340bp    │    │ ↓ -3.2%  │                   │    │
│  │  │ COMPLACENT│   │ NORMAL   │    │ CAUTIOUS │                   │    │
│  │  └──────────┘    └──────────┘    └──────────┘                   │    │
│  │                                                                  │    │
│  │  진단: 2/3 긍정, 1/3 경고 → CAUTIOUSLY BULLISH (신뢰도: MEDIUM) │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 상태 진단 코드

```python
class StateDiagnosisAgent:
    """
    크리티컬 패스 기반 현재 상태 진단
    """

    def __init__(self, critical_paths: Dict):
        self.paths = critical_paths
        self.thresholds = self.load_thresholds()

    async def diagnose(self, current_data: pd.DataFrame) -> StateDiagnosis:
        """
        모든 크리티컬 패스에 대해 현재 상태 진단
        """
        diagnoses = {}

        for path_name, path_config in self.paths.items():
            # 각 경로의 변수들 현재 상태 확인
            path_signals = []

            for var in path_config['path']:
                current_value = current_data[var['variable']].iloc[-1]
                change = self.calculate_change(current_data, var['variable'])
                signal = self.interpret_signal(var, change)
                path_signals.append(signal)

            # 경로 전체 진단
            diagnoses[path_name] = PathDiagnosis(
                target=path_config['target'],
                signals=path_signals,
                consensus=self.calculate_consensus(path_signals),
                confidence=self.calculate_confidence(path_signals),
                direction=self.determine_direction(path_signals)
            )

        # 전체 시장 상태 종합
        market_state = self.synthesize_market_state(diagnoses)

        return StateDiagnosis(
            path_diagnoses=diagnoses,
            market_state=market_state,
            timestamp=datetime.now()
        )

    def interpret_signal(self, var_config: Dict, change: float) -> Signal:
        """
        변수 변화를 신호로 해석
        """
        thresholds = self.thresholds[var_config['variable']]

        if abs(change) < thresholds['noise']:
            level = "NEUTRAL"
        elif change > thresholds['strong_positive']:
            level = "STRONG_BULLISH"
        elif change > thresholds['positive']:
            level = "BULLISH"
        elif change < thresholds['strong_negative']:
            level = "STRONG_BEARISH"
        elif change < thresholds['negative']:
            level = "BEARISH"
        else:
            level = "NEUTRAL"

        # 방향성 적용 (inverse 관계면 반전)
        if var_config['direction'] == 'inverse':
            level = self.invert_signal(level)

        return Signal(
            variable=var_config['variable'],
            current_change=change,
            level=level,
            lead_time=var_config['lead_time'],
            weight=var_config['weight']
        )

    def synthesize_market_state(self, diagnoses: Dict) -> MarketState:
        """
        모든 경로 진단을 종합하여 시장 상태 결정
        """
        states = {
            'fed_outlook': diagnoses.get('fed_rate_path'),
            'equity_outlook': diagnoses.get('equity_path'),
            'inflation_outlook': diagnoses.get('inflation_path'),
            'cycle_position': diagnoses.get('business_cycle_path')
        }

        # 종합 판단
        return MarketState(
            regime=self.determine_regime(states),
            risk_level=self.calculate_risk_level(states),
            opportunity_score=self.calculate_opportunity(states),
            primary_concern=self.identify_primary_concern(states)
        )
```

#### 4.2.3 상태 진단 출력 예시

```yaml
state_diagnosis:
  timestamp: "2025-12-27 10:30:00"

  path_diagnoses:
    fed_rate_path:
      target: "Fed_Rate_Decision"
      signals:
        - variable: "Gold_Silver"
          change: "+5.2%"
          level: "BULLISH"
          implication: "금리 인하 기대"
        - variable: "Dollar_Index"
          change: "-2.1%"
          level: "BEARISH"
          implication: "달러 약세 = dovish"
        - variable: "HY_Spread"
          change: "-15bp"
          level: "RISK_ON"
          implication: "신용 환경 양호"
      consensus: "3/3 DOVISH"
      direction: "RATE_CUT_EXPECTED"
      confidence: 0.85
      expected_timing: "30일 내"

    equity_path:
      target: "SPY_Direction"
      signals:
        - variable: "VIX"
          value: 14.5
          level: "COMPLACENT"
          warning: "과도한 안일함"
        - variable: "HY_Spread"
          value: "340bp"
          level: "NORMAL"
        - variable: "Copper_Gold_Ratio"
          change: "-3.2%"
          level: "CAUTIOUS"
          warning: "산업 활동 둔화 신호"
      consensus: "2/3 POSITIVE, 1/3 WARNING"
      direction: "CAUTIOUSLY_BULLISH"
      confidence: 0.65
      risk_flag: "VIX_COMPLACENCY"

  market_state:
    regime: "LATE_CYCLE_BULL"
    risk_level: "ELEVATED"
    opportunity_score: 0.62
    primary_concerns:
      - "VIX 과도 안일 → 조정 가능성"
      - "구리/금 비율 하락 → 성장 둔화 신호"
    primary_opportunities:
      - "Fed 금리 인하 기대 → 채권/금 유리"
      - "달러 약세 → 신흥국/원자재 유리"
```

---

### 4.3 충격 대비 (Shock Preparation)

#### 4.3.1 충격 시나리오 프레임워크

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SHOCK PREPARATION FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  크리티컬 패스가 "이상 신호"를 보내면 충격 대비 모드 활성화               │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  SHOCK DETECTION                                                 │    │
│  │                                                                  │    │
│  │  정상 범위:     [━━━━━━━━━━━━━━━━━━━━━━━━]                        │    │
│  │                        ↑ 현재값                                  │    │
│  │                                                                  │    │
│  │  경고 범위:  [━━━] ◄── 1.5σ 이탈 ──► [━━━]                       │    │
│  │                                                                  │    │
│  │  위험 범위: [━] ◄──── 2.0σ 이탈 ────► [━]                        │    │
│  │                                                                  │    │
│  │  극단 범위: ◄────── 3.0σ 이탈 ──────►                            │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  SCENARIO ANALYSIS                                               │    │
│  │                                                                  │    │
│  │  시나리오 1: VIX Spike (+50%)                                    │    │
│  │  ├── 발생 확률: 15%                                              │    │
│  │  ├── 예상 영향: SPY -8%, TLT +5%, GLD +3%                       │    │
│  │  ├── 선행 신호: HY 스프레드 확대, 거래량 급증                    │    │
│  │  └── 대응 전략: 주식 축소, 채권/금 확대                          │    │
│  │                                                                  │    │
│  │  시나리오 2: Fed Hawkish Surprise                                │    │
│  │  ├── 발생 확률: 10%                                              │    │
│  │  ├── 예상 영향: TLT -5%, DXY +3%, GLD -2%                       │    │
│  │  ├── 선행 신호: 인플레 서프라이즈, 고용 호조                     │    │
│  │  └── 대응 전략: 듀레이션 축소, 달러 롱                           │    │
│  │                                                                  │    │
│  │  시나리오 3: Credit Crisis                                       │    │
│  │  ├── 발생 확률: 5%                                               │    │
│  │  ├── 예상 영향: HYG -15%, SPY -20%, TLT +10%                    │    │
│  │  ├── 선행 신호: HY 스프레드 급등, 은행주 급락                    │    │
│  │  └── 대응 전략: 리스크 자산 청산, 국채/현금                      │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.3.2 충격 대비 에이전트

```python
class ShockPreparationAgent:
    """
    크리티컬 패스 기반 충격 감지 및 대비
    """

    # 사전 정의된 충격 시나리오
    SHOCK_SCENARIOS = {
        "vix_spike": {
            "trigger": {"VIX": {"change": ">30%", "or_level": ">30"}},
            "probability_base": 0.15,
            "impact": {"SPY": -0.08, "TLT": 0.05, "GLD": 0.03},
            "leading_signals": ["HY_Spread_widening", "Volume_spike"],
            "response": "risk_off"
        },
        "fed_hawkish": {
            "trigger": {"Fed_Tone": "hawkish_surprise"},
            "probability_base": 0.10,
            "impact": {"TLT": -0.05, "DXY": 0.03, "GLD": -0.02},
            "leading_signals": ["CPI_surprise", "Employment_strong"],
            "response": "duration_short"
        },
        "credit_crisis": {
            "trigger": {"HY_Spread": {"change": ">100bp"}},
            "probability_base": 0.05,
            "impact": {"HYG": -0.15, "SPY": -0.20, "TLT": 0.10},
            "leading_signals": ["Bank_stocks_drop", "HY_outflows"],
            "response": "full_risk_off"
        },
        "dollar_crash": {
            "trigger": {"DXY": {"change": "<-5%"}},
            "probability_base": 0.08,
            "impact": {"GLD": 0.10, "EM": 0.08, "SPY": -0.03},
            "leading_signals": ["Gold_surge", "Fed_dovish"],
            "response": "dollar_hedge"
        }
    }

    async def assess_shock_risks(
        self,
        current_data: pd.DataFrame,
        path_diagnosis: StateDiagnosis
    ) -> ShockAssessment:
        """
        현재 상태에서 각 충격 시나리오 위험도 평가
        """
        assessments = []

        for scenario_name, scenario in self.SHOCK_SCENARIOS.items():
            # 선행 신호 체크
            leading_signal_score = self.check_leading_signals(
                current_data,
                scenario['leading_signals']
            )

            # 트리거 근접도 체크
            trigger_proximity = self.check_trigger_proximity(
                current_data,
                scenario['trigger']
            )

            # 조정된 확률 계산
            adjusted_probability = (
                scenario['probability_base'] *
                (1 + leading_signal_score) *
                (1 + trigger_proximity)
            )

            # 예상 영향 계산
            expected_impact = self.calculate_expected_impact(
                scenario['impact'],
                adjusted_probability
            )

            assessments.append(ShockScenario(
                name=scenario_name,
                probability=adjusted_probability,
                impact=scenario['impact'],
                expected_impact=expected_impact,
                leading_signals=self.get_active_signals(
                    current_data,
                    scenario['leading_signals']
                ),
                trigger_proximity=trigger_proximity,
                recommended_response=scenario['response']
            ))

        # 위험도 순으로 정렬
        assessments.sort(key=lambda x: x.expected_impact, reverse=True)

        return ShockAssessment(
            scenarios=assessments,
            overall_risk_level=self.calculate_overall_risk(assessments),
            recommended_hedges=self.recommend_hedges(assessments)
        )

    def recommend_hedges(self, scenarios: List[ShockScenario]) -> List[Hedge]:
        """
        충격 시나리오에 대한 헤지 전략 추천
        """
        hedges = []

        for scenario in scenarios:
            if scenario.probability > 0.10:  # 10% 이상 확률
                if scenario.name == "vix_spike":
                    hedges.append(Hedge(
                        type="VIX_CALL",
                        size="portfolio의 2-3%",
                        rationale="VIX 급등 시 보호"
                    ))
                elif scenario.name == "credit_crisis":
                    hedges.append(Hedge(
                        type="HYG_PUT",
                        size="HY 익스포저의 10%",
                        rationale="신용 위기 시 보호"
                    ))
                elif scenario.name == "dollar_crash":
                    hedges.append(Hedge(
                        type="GLD_POSITION",
                        size="portfolio의 5-10%",
                        rationale="달러 약세 헤지"
                    ))

        return hedges
```

#### 4.3.3 충격 대비 출력 예시

```yaml
shock_assessment:
  timestamp: "2025-12-27 10:30:00"
  overall_risk_level: "ELEVATED"

  scenarios:
    - name: "vix_spike"
      probability: 0.22  # 기본 15% → 선행 신호로 상향
      trigger_proximity: 0.45
      leading_signals_active:
        - "HY_Spread widening: +20bp (경고)"
        - "Put/Call ratio elevated: 1.2"
      expected_impact:
        SPY: "-1.8%"
        TLT: "+1.1%"
        GLD: "+0.7%"
      recommended_response: "Reduce equity exposure 10%"

    - name: "fed_hawkish"
      probability: 0.08
      trigger_proximity: 0.20
      leading_signals_active: []
      expected_impact:
        TLT: "-0.4%"
        DXY: "+0.2%"
      recommended_response: "Monitor CPI release"

  recommended_hedges:
    - type: "VIX_CALL"
      strike: "VIX 20"
      size: "Portfolio 2%"
      cost: "~0.3% of portfolio"
      rationale: "VIX spike 확률 22%로 상승, 보호 필요"

    - type: "TLT_POSITION"
      action: "Increase"
      size: "+5% allocation"
      rationale: "Risk-off 시나리오 헤지"

  action_items:
    immediate:
      - "VIX 콜옵션 매수 검토"
      - "HY 익스포저 점검"
    watch:
      - "HY 스프레드 350bp 돌파 시 추가 조치"
      - "VIX 18 돌파 시 경고 업그레이드"
```

---

### 4.4 매수/매도 추천 (Trading Recommendations)

#### 4.4.1 추천 생성 프레임워크

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADING RECOMMENDATION FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT                                                                   │
│  ├── 크리티컬 패스 상태 진단                                              │
│  ├── 충격 시나리오 평가                                                   │
│  ├── 경제학파 토론 결과                                                   │
│  └── 기술적 시그널                                                       │
│                         │                                                │
│                         ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  SIGNAL AGGREGATION                                              │    │
│  │                                                                  │    │
│  │  자산별 신호 종합:                                               │    │
│  │                                                                  │    │
│  │  SPY:                                                           │    │
│  │  ├── 크리티컬 패스: CAUTIOUSLY_BULLISH (+0.3)                   │    │
│  │  ├── 충격 위험: ELEVATED (-0.2)                                 │    │
│  │  ├── 학파 토론: 3/4 POSITIVE (+0.2)                             │    │
│  │  ├── 기술적 시그널: Net Buy 3x (+0.4)                           │    │
│  │  └── 종합 점수: +0.7 → MODERATE_BUY                             │    │
│  │                                                                  │    │
│  │  GLD:                                                           │    │
│  │  ├── 크리티컬 패스: BULLISH (+0.5)                              │    │
│  │  ├── 충격 위험: HEDGE_VALUE (+0.3)                              │    │
│  │  ├── 학파 토론: Monetarist/Austrian 강력 지지 (+0.4)            │    │
│  │  ├── 기술적 시그널: Breakout (+0.3)                             │    │
│  │  └── 종합 점수: +1.5 → STRONG_BUY                               │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                         │                                                │
│                         ▼                                                │
│  OUTPUT                                                                  │
│  ├── 자산별 추천 (BUY/HOLD/SELL)                                        │
│  ├── 포지션 사이즈 제안                                                  │
│  ├── 진입/청산 가격                                                      │
│  ├── 손절/익절 레벨                                                      │
│  └── 신뢰도 및 근거                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.4.2 추천 에이전트

```python
class TradingRecommendationAgent:
    """
    크리티컬 패스 + 토론 + 시그널 종합하여 매수/매도 추천
    """

    # 추천 레벨
    LEVELS = {
        "STRONG_BUY": {"score_min": 1.2, "position_pct": 0.10},
        "MODERATE_BUY": {"score_min": 0.5, "position_pct": 0.05},
        "HOLD": {"score_min": -0.5, "position_pct": 0.00},
        "MODERATE_SELL": {"score_min": -1.2, "position_pct": -0.05},
        "STRONG_SELL": {"score_min": -999, "position_pct": -0.10},
    }

    async def generate_recommendations(
        self,
        state_diagnosis: StateDiagnosis,
        shock_assessment: ShockAssessment,
        debate_result: DebateResult,
        technical_signals: Dict
    ) -> TradingRecommendations:
        """
        모든 입력을 종합하여 자산별 추천 생성
        """
        recommendations = {}

        for asset in self.target_assets:
            # 1. 크리티컬 패스 신호
            path_score = self.get_path_score(asset, state_diagnosis)

            # 2. 충격 위험 조정
            shock_adjustment = self.get_shock_adjustment(asset, shock_assessment)

            # 3. 학파 토론 결과
            debate_score = self.get_debate_score(asset, debate_result)

            # 4. 기술적 시그널
            technical_score = self.get_technical_score(asset, technical_signals)

            # 종합 점수 (가중 평균)
            total_score = (
                path_score * 0.30 +
                shock_adjustment * 0.20 +
                debate_score * 0.25 +
                technical_score * 0.25
            )

            # 추천 레벨 결정
            level = self.determine_level(total_score)

            # 포지션 사이즈 및 가격 계산
            position_size = self.calculate_position_size(level, asset)
            entry_price = self.calculate_entry_price(asset, level)
            stop_loss = self.calculate_stop_loss(asset, level)
            take_profit = self.calculate_take_profit(asset, level)

            recommendations[asset] = Recommendation(
                asset=asset,
                action=level,
                score=total_score,
                position_size=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=self.calculate_confidence(
                    path_score, debate_score, technical_score
                ),
                rationale=self.generate_rationale(
                    asset, path_score, shock_adjustment,
                    debate_score, technical_score
                ),
                risk_reward_ratio=self.calculate_rr_ratio(
                    entry_price, stop_loss, take_profit
                )
            )

        return TradingRecommendations(
            recommendations=recommendations,
            portfolio_summary=self.generate_portfolio_summary(recommendations),
            risk_warning=self.generate_risk_warning(shock_assessment)
        )

    def generate_rationale(
        self,
        asset: str,
        path_score: float,
        shock_adj: float,
        debate_score: float,
        tech_score: float
    ) -> str:
        """
        추천 근거 생성
        """
        reasons = []

        if path_score > 0.3:
            reasons.append(f"크리티컬 패스: 선행 지표들이 {asset} 상승 시사")
        elif path_score < -0.3:
            reasons.append(f"크리티컬 패스: 선행 지표들이 {asset} 하락 시사")

        if shock_adj < -0.1:
            reasons.append("충격 위험: 헤지 필요성으로 포지션 축소 권장")

        if debate_score > 0.3:
            reasons.append("경제학파 토론: 다수 학파가 긍정적 전망")

        if tech_score > 0.3:
            reasons.append("기술적 시그널: 매수 압력 확인")

        return "; ".join(reasons)
```

#### 4.4.3 추천 출력 예시

```yaml
trading_recommendations:
  timestamp: "2025-12-27 10:30:00"
  market_context: "Late-cycle bull, elevated risk"

  recommendations:
    GLD:
      action: "STRONG_BUY"
      score: 1.52
      confidence: 0.82
      position_size: "Portfolio 10% (신규) 또는 5% 추가"
      entry_price: "$2,050 이하"
      stop_loss: "$1,980 (-3.4%)"
      take_profit: "$2,200 (+7.3%)"
      risk_reward_ratio: 2.1
      rationale:
        - "크리티컬 패스: 금↑달러↓ 패턴 진행 중 (Fed 금리 인하 선행)"
        - "충격 위험: VIX spike 시 헤지 역할"
        - "Monetarist/Austrian: 인플레 + fiat 불신으로 강력 지지"
        - "기술적: 2,020 저항선 돌파, 거래량 증가"

    SPY:
      action: "HOLD"
      score: 0.32
      confidence: 0.58
      position_size: "현 포지션 유지"
      entry_price: "N/A (신규 진입 비추천)"
      stop_loss: "현재가 -5% 하회 시 축소"
      take_profit: "N/A"
      risk_reward_ratio: N/A
      rationale:
        - "크리티컬 패스: 혼재된 신호 (VIX 낮지만 Cu/Au 하락)"
        - "충격 위험: VIX spike 확률 상승 → 신규 진입 자제"
        - "학파 분열: Keynesian 긍정 vs Austrian 버블 경고"
        - "기술적: 상승 모멘텀 있으나 과매수 근접"

    TLT:
      action: "MODERATE_BUY"
      score: 0.78
      confidence: 0.71
      position_size: "Portfolio 5% 추가"
      entry_price: "$92 이하"
      stop_loss: "$88 (-4.3%)"
      take_profit: "$98 (+6.5%)"
      risk_reward_ratio: 1.5
      rationale:
        - "크리티컬 패스: Fed dovish 신호 → 금리 하락 기대"
        - "충격 위험: Risk-off 시 수혜"
        - "Keynesian: 경기 둔화 시 채권 유리"
        - "기술적: 200일 MA 지지, RSI 중립"

    HYG:
      action: "MODERATE_SELL"
      score: -0.65
      confidence: 0.68
      position_size: "현 포지션 50% 축소"
      entry_price: "N/A"
      stop_loss: "N/A"
      take_profit: "N/A"
      rationale:
        - "크리티컬 패스: HY 스프레드 확대 조짐"
        - "충격 위험: Credit crisis 시나리오 존재"
        - "Austrian: 신용 사이클 후반, 위험 증가"
        - "기술적: 상승 모멘텀 약화"

  portfolio_summary:
    current_allocation:
      equity: 60%
      fixed_income: 25%
      commodities: 10%
      cash: 5%
    recommended_change:
      equity: -5%  # 55%로 축소
      fixed_income: +5%  # 30%로 증가
      commodities: +5%  # 15%로 증가 (금 추가)
      cash: -5%  # 0%로 축소

  risk_warnings:
    - level: "WARNING"
      message: "VIX 급등 확률 22% - 헤지 고려"
    - level: "CAUTION"
      message: "구리/금 비율 하락 - 성장 둔화 모니터링"
```

---

### 4.5 전체 프로세스 요약

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL PATH → RECOMMENDATION FLOW                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. DISCOVERY (크리티컬 패스 발견)                                        │
│     └─→ Granger Causality → 네트워크 → 핵심 경로 추출                    │
│                                                                          │
│  2. DIAGNOSIS (현재 상태 진단)                                            │
│     └─→ 각 경로 변수 체크 → 시장 국면 판단                               │
│                                                                          │
│  3. PREPARATION (충격 대비)                                               │
│     └─→ 이상 신호 감지 → 시나리오 확률 → 헤지 전략                       │
│                                                                          │
│  4. DEBATE (경제학적 해석)                                                │
│     └─→ 4개 학파 토론 → 합의/분열 도출                                   │
│                                                                          │
│  5. RECOMMENDATION (매수/매도 추천)                                       │
│     └─→ 모든 신호 종합 → 자산별 추천 + 포지션 사이즈                     │
│                                                                          │
│  6. EXECUTION (실행 지원)                                                 │
│     └─→ 진입/청산 가격, 손절/익절 레벨, 리스크 경고                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Regime Change Detection (구조 변화 탐지)

> **핵심 원칙**: 테슬라 계약 전 삼성전자 ≠ 계약 후 삼성전자
> 구조 변화 전후 데이터를 섞으면 분석이 왜곡된다.

### 5.1 구조 변화란?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     REGIME CHANGE CONCEPT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  기업/시장의 "본질적 가치 구조"가 바뀌는 이벤트                           │
│                                                                          │
│  예시:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  삼성전자                                                        │    │
│  │                                                                  │    │
│  │  Before: 메모리 반도체 + 스마트폰 기업                           │    │
│  │          → P/E 8-12배, 사이클 기업 취급                          │    │
│  │                                                                  │    │
│  │  [테슬라 계약 이벤트]  ← 거래량 급증으로 탐지                     │    │
│  │                                                                  │    │
│  │  After:  EV 생태계 핵심 파트너                                   │    │
│  │          → P/E 15-20배 리레이팅 가능                             │    │
│  │          → 새로운 밸류에이션 기준 적용 필요                       │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ⚠️ Before 데이터로 학습한 모델은 After 예측에 부적합                    │
│  ⚠️ Before + After 섞으면 노이즈만 증가                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 구조 변화 탐지 및 영향력 평가 파이프라인

> **핵심 프로세스**: 거래량 급변 탐지 → 뉴스 검색 → 좋은지/나쁜지 판단 → 영향력 크기 추정

```
┌─────────────────────────────────────────────────────────────────────────┐
│              REGIME CHANGE DETECTION & IMPACT ASSESSMENT                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: 거래량 급변 탐지                                                │
│  ├── 일일 거래량 모니터링                                                │
│  ├── Rolling 평균 대비 Z-score 계산                                     │
│  └── 3σ 이상 → 이상 신호 발생                                           │
│           │                                                              │
│           ▼                                                              │
│  STEP 2: 뉴스 검색 (Perplexity Agent)                                   │
│  ├── 해당 날짜 ±3일 뉴스 검색                                           │
│  ├── 기업명 + 키워드로 관련 뉴스 필터링                                  │
│  └── 뉴스 없으면 → 노이즈로 판단, 종료                                   │
│           │                                                              │
│           ▼                                                              │
│  STEP 3: 뉴스 분류 (Claude/OpenAI Agent)                                │
│  ├── 좋은 뉴스 vs 나쁜 뉴스 판단                                        │
│  ├── 뉴스 유형 분류 (계약, M&A, 스캔들, 실적, 규제...)                  │
│  └── 일시적 이벤트 vs 구조적 변화 구분                                   │
│           │                                                              │
│           ▼                                                              │
│  STEP 4: 영향력 크기 추정 (Multi-AI Debate)                             │
│  ├── 거래량 변화 폭 분석                                                 │
│  ├── 과거 유사 사례 검색 및 비교                                        │
│  ├── 경제학 이론으로 영향 예측                                           │
│  └── AI 토론으로 합의 도출                                               │
│           │                                                              │
│           ▼                                                              │
│  STEP 5: 레짐 변화 결정 및 데이터 분리                                   │
│  ├── 구조적 변화 확정 시 → 데이터 분리점 설정                           │
│  ├── 새로운 밸류에이션 기준 적용                                         │
│  └── 이후 분석은 새 레짐 데이터만 사용                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.1 Step 1: 거래량 급변 탐지

```python
class VolumeBreakoutDetector:
    """
    거래량 급변 탐지 - 구조 변화의 첫 번째 신호
    """

    def detect(
        self,
        data: pd.DataFrame,
        ticker: str,
        lookback: int = 60,
        threshold_sigma: float = 3.0
    ) -> List[VolumeEvent]:
        """
        거래량이 평균 대비 N시그마 이상 급증한 날 탐지

        원리:
        - 구조 변화 시 정보를 가진 투자자들이 먼저 움직임
        - 거래량 급증 = 새로운 정보가 시장에 반영 중
        - 3시그마 이상 = 통계적으로 유의미한 이벤트
        """
        volume = data['Volume']
        rolling_mean = volume.rolling(lookback).mean()
        rolling_std = volume.rolling(lookback).std()

        z_score = (volume - rolling_mean) / rolling_std

        # 급증 날짜 탐지
        breakout_dates = data.index[z_score > threshold_sigma].tolist()

        events = []
        for date in breakout_dates:
            # 가격 변화도 함께 기록
            price_change = data.loc[date, 'Close'] / data.loc[date - pd.Timedelta(days=1), 'Close'] - 1

            events.append(VolumeEvent(
                date=date,
                ticker=ticker,
                volume_zscore=z_score.loc[date],
                volume_multiple=volume.loc[date] / rolling_mean.loc[date],
                price_change=price_change,
                direction="UP" if price_change > 0 else "DOWN"
            ))

        return events
```

#### 5.2.2 Step 2: 뉴스 검색 (Perplexity Agent)

```python
class NewsSearchAgent:
    """
    Perplexity API를 활용한 뉴스 검색
    거래량 급변 날짜에 무슨 일이 있었는지 파악
    """

    def __init__(self):
        self.perplexity = PerplexityClient()

    async def search_news_for_event(
        self,
        ticker: str,
        company_name: str,
        event_date: datetime,
        days_range: int = 3
    ) -> NewsSearchResult:
        """
        특정 날짜 전후 관련 뉴스 검색

        검색 전략:
        1. 기업명 + 날짜로 직접 검색
        2. 티커 + 주요 키워드 (계약, 인수, 실적, 스캔들)
        3. 산업 전체 뉴스 (섹터 이벤트일 수 있음)
        """
        start_date = event_date - timedelta(days=days_range)
        end_date = event_date + timedelta(days=days_range)

        # 검색 쿼리 구성
        queries = [
            f"{company_name} news {event_date.strftime('%Y-%m-%d')}",
            f"{company_name} announcement {event_date.strftime('%B %Y')}",
            f"{ticker} stock surge {event_date.strftime('%Y-%m-%d')}",
            f"{company_name} contract partnership deal {event_date.strftime('%B %Y')}",
        ]

        all_news = []
        for query in queries:
            response = await self.perplexity.search(
                query=query,
                date_range=(start_date, end_date)
            )
            all_news.extend(response.results)

        # 중복 제거 및 관련성 순 정렬
        unique_news = self.deduplicate_and_rank(all_news, company_name)

        if not unique_news:
            return NewsSearchResult(
                found=False,
                message="관련 뉴스 없음 - 거래량 급증이 노이즈일 가능성"
            )

        return NewsSearchResult(
            found=True,
            news_items=unique_news,
            primary_news=unique_news[0],  # 가장 관련성 높은 뉴스
            search_date=event_date
        )
```

#### 5.2.3 Step 3: 뉴스 분류 및 성격 판단 (Claude Agent)

```python
class NewsClassificationAgent:
    """
    Claude를 활용한 뉴스 분류 및 성격 판단
    좋은 뉴스인지, 나쁜 뉴스인지, 얼마나 중요한지 판단
    """

    CLASSIFICATION_PROMPT = """
    당신은 금융 뉴스 분석 전문가입니다.

    ## 분석 대상
    기업: {company_name} ({ticker})
    날짜: {event_date}
    거래량 변화: 평균 대비 {volume_multiple:.1f}배 (Z-score: {volume_zscore:.1f})
    가격 변화: {price_change:+.1%}

    ## 관련 뉴스
    {news_content}

    ## 분석 요청

    1. **뉴스 성격 판단**
       - POSITIVE (호재) / NEGATIVE (악재) / NEUTRAL (중립)
       - 판단 근거

    2. **뉴스 유형 분류**
       - CONTRACT: 대형 계약 체결
       - PARTNERSHIP: 전략적 제휴
       - MA: 인수합병
       - EARNINGS: 실적 발표
       - PRODUCT: 신제품/기술
       - REGULATORY: 규제 변화
       - SCANDAL: 스캔들/소송
       - MANAGEMENT: 경영진 변화
       - MACRO: 거시경제 영향
       - OTHER: 기타

    3. **영향 지속성 판단**
       - TEMPORARY: 일시적 이벤트 (1-2주 내 소멸)
       - MEDIUM_TERM: 중기적 영향 (1-6개월)
       - STRUCTURAL: 구조적 변화 (기업 가치 재평가 필요)

    4. **구조적 변화 여부**
       - 이 이벤트로 인해 기업의 "본질적 가치 구조"가 변하는가?
       - 예: 삼성전자-테슬라 계약 → 메모리 기업에서 EV 생태계 기업으로

    ## 출력 형식 (YAML)
    ```yaml
    sentiment: POSITIVE | NEGATIVE | NEUTRAL
    sentiment_score: -1.0 ~ +1.0
    news_type: CONTRACT | PARTNERSHIP | ...
    duration: TEMPORARY | MEDIUM_TERM | STRUCTURAL
    is_regime_change: true | false
    regime_change_reason: "..."
    confidence: 0.0 ~ 1.0
    key_points:
      - "핵심 포인트 1"
      - "핵심 포인트 2"
    ```
    """

    async def classify_news(
        self,
        news_result: NewsSearchResult,
        volume_event: VolumeEvent,
        company_info: Dict
    ) -> NewsClassification:
        """
        뉴스를 분류하고 성격 판단
        """
        prompt = self.CLASSIFICATION_PROMPT.format(
            company_name=company_info['name'],
            ticker=company_info['ticker'],
            event_date=volume_event.date,
            volume_multiple=volume_event.volume_multiple,
            volume_zscore=volume_event.volume_zscore,
            price_change=volume_event.price_change,
            news_content=self.format_news(news_result.news_items)
        )

        response = await self.claude.complete(prompt)
        classification = self.parse_yaml_response(response)

        return NewsClassification(**classification)
```

#### 5.2.4 Step 4: 영향력 크기 추정 (Multi-AI Debate)

```python
class ImpactAssessmentDebate:
    """
    여러 AI가 토론하여 영향력 크기 추정

    판단 기준:
    1. 거래량 변화 폭 (클수록 시장 반응 강함)
    2. 과거 유사 사례 (역사적 선례)
    3. 경제학 이론 (밸류에이션, 성장률 영향)
    """

    async def assess_impact(
        self,
        volume_event: VolumeEvent,
        news_classification: NewsClassification,
        company_info: Dict
    ) -> ImpactAssessment:
        """
        Multi-AI 토론으로 영향력 추정
        """

        # 1. 과거 유사 사례 검색 (Perplexity)
        similar_cases = await self.perplexity_agent.search_similar_cases(
            news_type=news_classification.news_type,
            industry=company_info['industry'],
            market_cap=company_info['market_cap']
        )

        # 2. 각 AI가 영향력 추정
        context = {
            "volume_event": volume_event,
            "news": news_classification,
            "company": company_info,
            "similar_cases": similar_cases
        }

        opinions = await asyncio.gather(
            self.claude_assess(context),
            self.openai_assess(context),
            self.gemini_assess(context)
        )

        # 3. 토론 및 합의
        debate_result = await self.run_debate(opinions)

        return ImpactAssessment(
            magnitude=debate_result.consensus_magnitude,
            duration=debate_result.consensus_duration,
            valuation_impact=debate_result.valuation_change,
            confidence=debate_result.confidence,
            similar_cases=similar_cases,
            debate_summary=debate_result.summary
        )

    async def claude_assess(self, context: Dict) -> ImpactOpinion:
        """
        Claude의 영향력 평가 (경제학 이론 중심)
        """
        prompt = f"""
        ## 이벤트 분석
        기업: {context['company']['name']}
        뉴스 유형: {context['news'].news_type}
        뉴스 성격: {context['news'].sentiment} ({context['news'].sentiment_score:+.1f})
        거래량: 평균 대비 {context['volume_event'].volume_multiple:.1f}배

        ## 과거 유사 사례
        {self.format_similar_cases(context['similar_cases'])}

        ## 경제학적 분석 요청

        1. **밸류에이션 영향**
           - 이 이벤트가 기업 가치에 미치는 영향은?
           - P/E, P/S 배수 변화 예상?

        2. **성장률 영향**
           - 매출/이익 성장률에 미치는 영향?
           - 일시적 vs 지속적?

        3. **리스크 프리미엄 영향**
           - 기업 리스크가 증가/감소하는가?
           - 할인율 변화?

        4. **과거 사례와 비교**
           - 유사 사례에서 주가는 어떻게 움직였는가?
           - 이번 케이스와의 차이점은?

        5. **적정 주가 영향 추정**
           - 현재가 대비 +X% ~ +Y% 또는 -X% ~ -Y%

        ## 출력 (YAML)
        ```yaml
        valuation_impact:
          pe_change: "+2x ~ +4x"  # P/E 배수 변화
          reason: "EV 생태계 진입으로 리레이팅"
        growth_impact:
          revenue_boost: "+5% ~ +10% annually"
          duration: "3-5 years"
        risk_impact:
          direction: "DECREASE"  # 고객 다변화
          discount_rate_change: "-0.5%"
        price_impact:
          range: "+15% ~ +25%"
          base_case: "+20%"
          confidence: 0.7
        similar_case_comparison:
          case: "LG에너지솔루션-테슬라 계약 (2021)"
          outcome: "+35% in 3 months"
          applicability: 0.8
        ```
        """

        response = await self.claude.complete(prompt)
        return self.parse_opinion(response, agent="Claude")

    async def run_debate(
        self,
        opinions: List[ImpactOpinion]
    ) -> DebateResult:
        """
        AI들의 의견을 종합하여 합의 도출
        """

        # 의견 비교
        magnitudes = [op.price_impact.base_case for op in opinions]
        avg_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)

        # 불일치 시 추가 토론
        if std_magnitude > 0.10:  # 10%p 이상 차이
            # 각 AI에게 다른 의견 반박 요청
            critiques = await self.cross_critique(opinions)
            # 재평가
            revised_opinions = await self.revise_opinions(opinions, critiques)
            magnitudes = [op.price_impact.base_case for op in revised_opinions]
            avg_magnitude = np.mean(magnitudes)

        return DebateResult(
            consensus_magnitude=avg_magnitude,
            consensus_duration=self.aggregate_duration(opinions),
            valuation_change=self.aggregate_valuation(opinions),
            confidence=1 - (std_magnitude / avg_magnitude) if avg_magnitude != 0 else 0,
            summary=self.generate_summary(opinions)
        )
```

#### 5.2.5 Step 5: 레짐 변화 결정

```python
class RegimeChangeDecision:
    """
    최종 레짐 변화 결정 및 데이터 분리
    """

    async def decide_regime_change(
        self,
        volume_event: VolumeEvent,
        news_classification: NewsClassification,
        impact_assessment: ImpactAssessment
    ) -> RegimeChangeResult:
        """
        구조적 변화 여부 최종 결정
        """

        # 구조적 변화 조건
        is_structural = (
            news_classification.is_regime_change and
            news_classification.duration == "STRUCTURAL" and
            impact_assessment.confidence > 0.6 and
            abs(impact_assessment.magnitude) > 0.15  # 15% 이상 영향
        )

        if is_structural:
            return RegimeChangeResult(
                is_regime_change=True,
                change_date=volume_event.date,
                before_regime=self.define_before_regime(volume_event.ticker),
                after_regime=self.define_after_regime(
                    volume_event.ticker,
                    news_classification,
                    impact_assessment
                ),
                analysis_instruction=f"""
                [레짐 변화 확정]
                날짜: {volume_event.date}
                이유: {news_classification.regime_change_reason}
                영향: {impact_assessment.magnitude:+.1%}

                분석 지침:
                1. {volume_event.date} 이전 데이터는 과거 레짐으로 분류
                2. 이후 데이터만 현재 분석에 사용
                3. 밸류에이션 기준: {impact_assessment.valuation_change}
                4. 새로운 핵심 드라이버 반영
                """
            )
        else:
            return RegimeChangeResult(
                is_regime_change=False,
                reason="일시적 이벤트로 판단 - 구조적 변화 아님",
                recommendation="전체 데이터 계속 사용"
            )
```

#### 5.2.6 전체 파이프라인 통합

```python
class RegimeChangeDetectionPipeline:
    """
    거래량 급변 → 뉴스 검색 → 영향력 평가 → 레짐 변화 결정
    전체 파이프라인 통합
    """

    def __init__(self):
        self.volume_detector = VolumeBreakoutDetector()
        self.news_agent = NewsSearchAgent()  # Perplexity
        self.classification_agent = NewsClassificationAgent()  # Claude
        self.impact_debate = ImpactAssessmentDebate()  # Multi-AI
        self.regime_decision = RegimeChangeDecision()

    async def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        company_info: Dict
    ) -> List[RegimeChangeResult]:
        """
        전체 파이프라인 실행
        """
        results = []

        # Step 1: 거래량 급변 탐지
        volume_events = self.volume_detector.detect(data, ticker)
        print(f"[Step 1] {len(volume_events)}개 거래량 급변 탐지됨")

        for event in volume_events:
            print(f"\n[분석 중] {event.date} - 거래량 {event.volume_multiple:.1f}x")

            # Step 2: 뉴스 검색
            news_result = await self.news_agent.search_news_for_event(
                ticker=ticker,
                company_name=company_info['name'],
                event_date=event.date
            )

            if not news_result.found:
                print(f"  → 관련 뉴스 없음 - 노이즈로 판단")
                continue

            print(f"  → 뉴스 발견: {news_result.primary_news.headline}")

            # Step 3: 뉴스 분류
            classification = await self.classification_agent.classify_news(
                news_result=news_result,
                volume_event=event,
                company_info=company_info
            )

            print(f"  → 분류: {classification.news_type}, {classification.sentiment}")
            print(f"  → 지속성: {classification.duration}")

            if classification.duration == "TEMPORARY":
                print(f"  → 일시적 이벤트 - 레짐 변화 아님")
                continue

            # Step 4: 영향력 평가 (Multi-AI Debate)
            print(f"  → 영향력 평가 중 (AI 토론)...")
            impact = await self.impact_debate.assess_impact(
                volume_event=event,
                news_classification=classification,
                company_info=company_info
            )

            print(f"  → 예상 영향: {impact.magnitude:+.1%} (신뢰도: {impact.confidence:.0%})")

            # Step 5: 레짐 변화 결정
            regime_result = await self.regime_decision.decide_regime_change(
                volume_event=event,
                news_classification=classification,
                impact_assessment=impact
            )

            if regime_result.is_regime_change:
                print(f"  ✅ 레짐 변화 확정!")
                print(f"     Before: {regime_result.before_regime['name']}")
                print(f"     After: {regime_result.after_regime['name']}")

            results.append(regime_result)

        return results
```

#### 5.2.7 실행 예시

```yaml
# 실행 예시: 삼성전자 분석

[Step 1] 3개 거래량 급변 탐지됨

[분석 중] 2024-03-15 - 거래량 4.2x
  → 뉴스 발견: "삼성전자, 테슬라와 배터리 공급 계약 체결"
  → 분류: CONTRACT, POSITIVE
  → 지속성: STRUCTURAL
  → 영향력 평가 중 (AI 토론)...

    Claude 의견:
      - P/E 리레이팅 +4x 예상 (8x → 12x)
      - EV 매출 비중 증가로 성장률 +5%
      - 유사 사례: LG에너지솔루션 +35%

    OpenAI 의견:
      - 보수적으로 P/E +2x
      - 실행 리스크 존재
      - 유사 사례 대비 시장 환경 다름

    Gemini 의견:
      - 중간값 P/E +3x
      - 데이터 기반 유사 사례 분석

    [토론 결과]
      합의: +18% ~ +25% (base case: +22%)
      신뢰도: 75%

  → 예상 영향: +22% (신뢰도: 75%)
  ✅ 레짐 변화 확정!
     Before: traditional_semiconductor (P/E 8-12x)
     After: ev_ecosystem_partner (P/E 12-16x)

[분석 중] 2024-05-20 - 거래량 3.1x
  → 뉴스 발견: "삼성전자 1분기 실적 예상치 상회"
  → 분류: EARNINGS, POSITIVE
  → 지속성: TEMPORARY
  → 일시적 이벤트 - 레짐 변화 아님

[분석 중] 2024-08-10 - 거래량 2.8x
  → 관련 뉴스 없음 - 노이즈로 판단

[최종 결과]
레짐 변화 1건 확정: 2024-03-15 (테슬라 계약)
```

#### 5.2.2 탐지된 구조 변화 저장

```yaml
# regime_changes.yaml
regime_changes:

  samsung_tesla_partnership:
    ticker: "005930.KS"
    date: "2024-XX-XX"
    detection:
      volume_zscore: 4.2
      price_change: "+8.5%"
      validated: true
    before_regime:
      name: "traditional_semiconductor"
      valuation: "P/E 8-12x"
      key_drivers: ["memory_cycle", "smartphone_demand"]
      data_start: "2020-01-01"
      data_end: "2024-XX-XX"
    after_regime:
      name: "ev_ecosystem_partner"
      valuation: "P/E 15-20x"
      key_drivers: ["ev_penetration", "battery_demand", "tesla_revenue"]
      data_start: "2024-XX-XX"
      data_end: null  # 현재 진행 중
    analysis_instruction: |
      Before와 After 데이터는 별도 모델로 분석
      After 기간 데이터로만 현재 예측 수행

  fed_pivot_2024:
    ticker: "MACRO"
    date: "2024-09-18"
    detection:
      vix_change: "-15%"
      spy_change: "+1.5%"
      validated: true
    before_regime:
      name: "hiking_cycle"
      environment: "rates rising, inflation fighting"
    after_regime:
      name: "easing_cycle"
      environment: "rates falling, growth supporting"
    analysis_instruction: |
      금리 인상기 vs 인하기 분리 분석
      인하기 시작 이후 데이터로 현재 전망

  trump_election_2024:
    ticker: "MACRO"
    date: "2024-11-05"
    detection:
      dollar_change: "+2.3%"
      russell_change: "+5.8%"
    before_regime:
      name: "biden_policy"
      expectations: ["green_energy", "regulation"]
    after_regime:
      name: "trump_policy"
      expectations: ["tariffs", "deregulation", "tax_cuts"]
    analysis_instruction: |
      정책 기대 변화 반영
      섹터별 영향 차별화 (에너지, 금융, 제조업)
```

### 5.3 데이터 분리 학습

```python
class RegimeAwareAnalyzer:
    """
    구조 변화를 인식하는 분석기
    """

    def __init__(self, regime_changes: Dict):
        self.regime_changes = regime_changes

    def get_relevant_data(
        self,
        ticker: str,
        current_date: datetime
    ) -> pd.DataFrame:
        """
        현재 레짐에 해당하는 데이터만 반환

        핵심: 과거 레짐 데이터는 현재 분석에 포함하지 않음
        """
        changes = self.regime_changes.get(ticker, [])

        if not changes:
            # 구조 변화 없으면 전체 데이터 사용
            return self.load_full_data(ticker)

        # 가장 최근 구조 변화 찾기
        latest_change = max(
            changes,
            key=lambda x: x['date']
        )

        # 구조 변화 이후 데이터만 반환
        return self.load_data(
            ticker,
            start_date=latest_change['date'],
            end_date=current_date
        )

    def train_regime_aware_model(
        self,
        ticker: str,
        target: str
    ) -> Dict[str, Any]:
        """
        레짐별 별도 모델 학습

        예: 삼성전자
        - Model A: 반도체 사이클 기업 모델 (테슬라 전)
        - Model B: EV 생태계 기업 모델 (테슬라 후) ← 현재 사용
        """
        models = {}

        for regime_name, regime_config in self.regime_changes[ticker].items():
            data = self.load_data(
                ticker,
                start_date=regime_config['data_start'],
                end_date=regime_config['data_end']
            )

            model = self.train_model(data, target)
            models[regime_name] = model

        # 현재 활성 레짐 모델 반환
        current_regime = self.get_current_regime(ticker)
        return {
            'current_model': models[current_regime],
            'all_models': models,
            'current_regime': current_regime
        }
```

---

## 6. Top-Down Analysis Hierarchy (하향식 분석 계층)

> **핵심 원칙**: 개별 기업 분석 전에 상위 레벨 상태부터 판단
> 세계정세 → 시장 → 자산군 → 섹터 → 개별기업

### 6.1 분석 계층 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TOP-DOWN ANALYSIS HIERARCHY                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 0: GLOBAL GEOPOLITICS (세계 정세)                                 │
│  ├── 전쟁/분쟁 위험                                                      │
│  ├── 무역 관계 (관세, 제재)                                              │
│  ├── 팬데믹/자연재해                                                     │
│  └── 정치적 불확실성                                                     │
│           │                                                              │
│           ▼ 상위 레벨이 위험하면 하위 분석 보수적으로                     │
│                                                                          │
│  Level 1: MONETARY ENVIRONMENT (통화 환경)                               │
│  ├── 글로벌 유동성 (돈이 많이 풀렸는가?)                                 │
│  ├── 중앙은행 정책 방향 (긴축 vs 완화)                                   │
│  ├── 인플레이션 수준                                                     │
│  └── 금리 사이클 위치                                                    │
│           │                                                              │
│           ▼ 유동성 환경이 자산군 성과 결정                               │
│                                                                          │
│  Level 2: ASSET CLASS (자산군)                                           │
│  ├── 주식 (Equity)                                                       │
│  ├── 채권 (Fixed Income)                                                 │
│  ├── 원자재 (Commodities)                                                │
│  ├── 암호화폐 (Crypto)                                                   │
│  └── 부동산 (Real Estate)                                                │
│           │                                                              │
│           ▼ 자산군 내 섹터 선택                                          │
│                                                                          │
│  Level 3: SECTOR (섹터)                                                  │
│  ├── Technology                                                          │
│  ├── Financials                                                          │
│  ├── Healthcare                                                          │
│  ├── Energy                                                              │
│  ├── Consumer                                                            │
│  └── ...                                                                 │
│           │                                                              │
│           ▼ 섹터 내 개별 기업 선택                                       │
│                                                                          │
│  Level 4: INDIVIDUAL STOCK (개별 기업)                                   │
│  ├── 펀더멘털 분석                                                       │
│  ├── 밸류에이션                                                          │
│  ├── 기술적 분석                                                         │
│  └── 이벤트/뉴스                                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

⚠️ 상위 레벨이 부정적이면 하위 레벨 분석은 의미 감소
   예: 전쟁 발발 시 개별 기업 분석보다 리스크 관리 우선
```

### 6.2 각 레벨별 AI 토론 주제

#### 6.2.1 Level 0: 세계 정세 토론

```yaml
level_0_debate:
  topic: "Global Geopolitical Risk Assessment"
  frequency: "Weekly"

  questions:
    - "현재 세계 정세에서 가장 큰 위험 요소는?"
    - "향후 3개월 내 지정학적 충격 가능성은?"
    - "리스크 자산 투자에 적합한 환경인가?"

  data_sources:
    perplexity:
      - "Russia Ukraine war latest"
      - "China Taiwan tensions"
      - "Middle East conflict"
      - "US China trade relations"
      - "Global supply chain risks"

  agent_perspectives:
    claude:
      focus: "전쟁/분쟁의 경제적 영향 분석"
      framework: "supply chain disruption, energy prices, safe haven flows"

    openai:
      focus: "정치적 불확실성 시나리오"
      framework: "election risks, policy uncertainty, regime changes"

    gemini:
      focus: "데이터 기반 위험 지표"
      framework: "geopolitical risk indices, conflict probability models"

  output:
    risk_level: "LOW | MEDIUM | ELEVATED | HIGH | CRITICAL"
    primary_risks:
      - risk: "..."
        probability: 0.X
        impact: "..."
    recommendation:
      risk_assets: "REDUCE | MAINTAIN | INCREASE"
      safe_havens: "INCREASE | MAINTAIN"
```

#### 6.2.2 Level 1: 통화 환경 토론

```yaml
level_1_debate:
  topic: "Global Monetary Environment"
  frequency: "Weekly"

  key_questions:
    - "돈이 많이 풀린 상황인가?"
    - "중앙은행들은 긴축인가 완화인가?"
    - "인플레이션은 통제되고 있는가?"
    - "금리 사이클의 어디에 있는가?"

  data_check:
    liquidity:
      - "Global M2 growth"
      - "Central bank balance sheets"
      - "Credit growth"
    policy:
      - "Fed policy stance"
      - "ECB policy stance"
      - "BOJ policy stance"
      - "PBOC policy stance"
    inflation:
      - "US CPI, PCE"
      - "Eurozone HICP"
      - "Breakeven inflation"

  agent_perspectives:
    monetarist:
      focus: "통화량과 인플레이션"
      key_metric: "M2 growth vs nominal GDP"
      interpretation: |
        M2 증가율 > 명목 GDP 증가율 → 과잉 유동성 → 자산 인플레
        M2 증가율 < 명목 GDP 증가율 → 유동성 긴축 → 자산 디플레

    keynesian:
      focus: "금리와 총수요"
      key_metric: "Real interest rate vs natural rate"
      interpretation: |
        실질금리 < 자연이자율 → 확장적 환경 → 리스크 자산 유리
        실질금리 > 자연이자율 → 긴축적 환경 → 안전자산 유리

    austrian:
      focus: "신용 사이클"
      key_metric: "Credit growth, debt levels"
      interpretation: |
        신용 과잉 팽창 → 버블 형성 → 조정 불가피
        건전한 신용 성장 → 지속 가능한 확장

  output:
    liquidity_regime: "ABUNDANT | NORMAL | TIGHT"
    policy_stance: "VERY_DOVISH | DOVISH | NEUTRAL | HAWKISH | VERY_HAWKISH"
    inflation_trend: "ACCELERATING | STABLE | DECELERATING"
    cycle_position: "EARLY | MID | LATE | TURNING"
    asset_implication:
      equities: "FAVORABLE | NEUTRAL | UNFAVORABLE"
      bonds: "..."
      commodities: "..."
      crypto: "..."
```

#### 6.2.3 Level 2: 자산군 토론

```yaml
level_2_debate:
  topic: "Asset Class Assessment"
  frequency: "Weekly"

  key_questions:
    - "주식시장이 계속 성장할 것인가?"
    - "채권은 매력적인가?"
    - "원자재 슈퍼사이클인가?"
    - "암호화폐에 자금이 유입되고 있는가?"

  asset_classes:
    equities:
      indicators:
        - "Earnings growth"
        - "Valuation (P/E, P/B)"
        - "Risk premium"
        - "Fund flows"
      debate_topic: "현재 밸류에이션이 정당화되는가?"

    fixed_income:
      indicators:
        - "Real yields"
        - "Credit spreads"
        - "Duration risk"
        - "Default rates"
      debate_topic: "금리 하락 여지가 있는가?"

    commodities:
      indicators:
        - "Supply/demand balance"
        - "Inventory levels"
        - "Dollar correlation"
        - "Inflation hedge value"
      debate_topic: "슈퍼사이클인가 일시적 상승인가?"

    crypto:
      indicators:
        - "Bitcoin dominance"
        - "Stablecoin flows"
        - "On-chain metrics"
        - "Institutional adoption"
      debate_topic: "기관 자금 유입이 지속되는가?"

  cross_asset_signals:
    - "Stock-bond correlation"
    - "Dollar vs commodities"
    - "VIX vs credit spreads"
    - "Gold vs real yields"

  output:
    allocation_recommendation:
      equities: "OVERWEIGHT | NEUTRAL | UNDERWEIGHT"
      bonds: "..."
      commodities: "..."
      crypto: "..."
      cash: "..."
    conviction_level: "HIGH | MEDIUM | LOW"
    key_risks: [...]
```

#### 6.2.4 Level 3: 섹터 토론

```yaml
level_3_debate:
  topic: "Sector Rotation Analysis"
  frequency: "Weekly"

  key_questions:
    - "현재 경기 사이클에서 어떤 섹터가 유리한가?"
    - "금리 환경이 어떤 섹터에 영향을 주는가?"
    - "정책 변화로 수혜받는 섹터는?"

  cycle_based_rotation:
    early_cycle:
      favored: ["Consumer Discretionary", "Financials", "Real Estate"]
      avoid: ["Utilities", "Consumer Staples"]

    mid_cycle:
      favored: ["Technology", "Industrials", "Materials"]
      avoid: ["Utilities"]

    late_cycle:
      favored: ["Energy", "Materials", "Healthcare"]
      avoid: ["Consumer Discretionary", "Real Estate"]

    recession:
      favored: ["Utilities", "Consumer Staples", "Healthcare"]
      avoid: ["Consumer Discretionary", "Financials"]

  rate_sensitivity:
    rate_rise:
      negative: ["Utilities", "Real Estate", "High-Growth Tech"]
      positive: ["Financials", "Insurance"]

    rate_fall:
      positive: ["Utilities", "Real Estate", "Growth Tech"]
      negative: ["Financials"]

  policy_impact:
    trump_policies:
      positive: ["Energy", "Financials", "Defense", "Domestic Manufacturing"]
      negative: ["Clean Energy", "Import-dependent sectors"]

  output:
    top_sectors: ["Sector1", "Sector2", "Sector3"]
    avoid_sectors: ["Sector1", "Sector2"]
    sector_rotation_signal: "DEFENSIVE | NEUTRAL | CYCLICAL"
    rationale: "..."
```

### 6.3 계층적 분석 흐름

```python
class TopDownAnalyzer:
    """
    하향식 분석 계층 관리
    """

    async def run_full_analysis(self) -> TopDownResult:
        """
        Level 0부터 순차적으로 분석
        상위 레벨 결과가 하위 레벨 분석에 영향
        """

        # Level 0: 세계 정세
        geopolitical = await self.analyze_geopolitics()
        if geopolitical.risk_level == "CRITICAL":
            return TopDownResult(
                recommendation="RISK_OFF",
                reason="지정학적 위험 극심 - 상세 분석 불필요",
                action="현금/금 비중 최대화"
            )

        # Level 1: 통화 환경
        monetary = await self.analyze_monetary(
            geopolitical_context=geopolitical
        )

        # Level 2: 자산군
        asset_class = await self.analyze_asset_classes(
            geopolitical_context=geopolitical,
            monetary_context=monetary
        )

        # Level 3: 섹터
        sector = await self.analyze_sectors(
            asset_context=asset_class,
            monetary_context=monetary
        )

        # Level 4: 개별 기업 (선택된 섹터 내에서만)
        stocks = await self.analyze_stocks(
            sector_context=sector,
            regime_changes=self.regime_changes
        )

        return TopDownResult(
            geopolitical=geopolitical,
            monetary=monetary,
            asset_class=asset_class,
            sector=sector,
            stocks=stocks,
            final_recommendation=self.synthesize_recommendation()
        )
```

### 6.4 AI 토론 프로토콜 (레벨별)

```python
class LevelBasedDebate:
    """
    각 레벨에서 AI들이 생산적으로 토론
    """

    async def debate_level(
        self,
        level: int,
        context: Dict,
        data: Dict
    ) -> DebateResult:
        """
        레벨별 토론 실행
        """

        # 1. 각 AI가 의견 제시
        opinions = await asyncio.gather(
            self.claude.analyze(level, context, data),
            self.openai.analyze(level, context, data),
            self.gemini.analyze(level, context, data),
            self.perplexity.get_latest_info(level, context)
        )

        # 2. 상호 비판 라운드
        critiques = await self.cross_critique(opinions)

        # 3. 증거 기반 합의 도출
        consensus = await self.reach_consensus(
            opinions=opinions,
            critiques=critiques,
            require_evidence=True  # 주장에는 반드시 증거 필요
        )

        # 4. 분열점 기록 (모든 학파가 동의하지 않는 부분)
        dissent = self.identify_dissent(opinions)

        return DebateResult(
            level=level,
            consensus=consensus,
            dissent=dissent,
            confidence=self.calculate_confidence(opinions),
            key_evidence=self.extract_key_evidence(opinions)
        )

    async def cross_critique(self, opinions: List[Opinion]) -> List[Critique]:
        """
        AI들이 서로의 의견을 비판

        생산적 토론 규칙:
        1. 주장에는 반드시 증거 제시
        2. 반박에는 대안 제시
        3. 동의 시에도 추가 관점 제시
        """
        critiques = []

        for i, opinion in enumerate(opinions):
            for j, other in enumerate(opinions):
                if i != j:
                    critique = await self.generate_critique(
                        critic=opinion.agent,
                        target=other,
                        style="constructive"  # 건설적 비판
                    )
                    critiques.append(critique)

        return critiques
```

---

## 7. ML/DL 과적합 방지 및 위험 관리

> **핵심 문제**: ML/DL은 단순 상승/하락 비율로 과적합되기 쉬움
> 해결: 도메인 지식 + 앙상블 + 불확실성 정량화

### 7.1 과적합 위험

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ML/DL OVERFITTING RISKS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  문제 1: 상승/하락 비율 학습                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Training Data: 60% 상승, 40% 하락                               │    │
│  │  모델이 학습하는 것: "60% 확률로 상승 예측하면 정확도 60%"        │    │
│  │                                                                  │    │
│  │  ⚠️ 이건 패턴 학습이 아니라 통계적 트릭                          │    │
│  │  ⚠️ 시장 레짐이 바뀌면 완전히 무력화                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  문제 2: 과거 패턴 과적합                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  "RSI < 30이면 매수" → 과거에는 작동                             │    │
│  │  But: 시장 구조가 변하면 더 이상 작동 안 함                      │    │
│  │                                                                  │    │
│  │  ⚠️ 금융 시계열은 비정상적 (non-stationary)                      │    │
│  │  ⚠️ 과거 패턴이 미래에 반복된다는 보장 없음                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  문제 3: 블랙박스 위험                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  딥러닝 모델: "매수 신호" (이유 모름)                            │    │
│  │                                                                  │    │
│  │  ⚠️ 왜 매수인지 설명 불가 → 신뢰 불가                            │    │
│  │  ⚠️ 잘못된 이유로 맞춘 것일 수 있음                               │    │
│  │  ⚠️ 시장 변화 시 실패 예측 불가                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 과적합 방지 전략

```python
class RobustModelFramework:
    """
    과적합을 방지하는 모델 프레임워크
    """

    # 원칙 1: 도메인 지식 우선
    def apply_domain_constraints(self, predictions):
        """
        경제학적으로 말이 안 되는 예측은 필터링

        예: "금리 상승 + 채권 상승" → 경제학적 모순 → 예측 불신
        """
        constraints = [
            lambda p: not (p['rate_up'] and p['bond_up']),
            lambda p: not (p['vix_spike'] and p['risk_on']),
            lambda p: not (p['dollar_up'] and p['gold_up']),  # 일반적으로
        ]

        for constraint in constraints:
            if not constraint(predictions):
                predictions['confidence'] *= 0.5
                predictions['warning'] = "경제학적 모순 감지"

        return predictions

    # 원칙 2: 앙상블 + 불일치 경고
    def ensemble_with_disagreement_check(self, models, data):
        """
        여러 모델의 예측이 일치하는지 확인

        모델들이 불일치하면 → 불확실성 높음 → 포지션 축소
        """
        predictions = [model.predict(data) for model in models]

        agreement = self.calculate_agreement(predictions)

        if agreement < 0.6:  # 60% 미만 일치
            return {
                'prediction': 'UNCERTAIN',
                'action': 'REDUCE_POSITION',
                'reason': f'모델 일치도 {agreement:.0%} - 불확실성 높음'
            }

        return {
            'prediction': self.aggregate_predictions(predictions),
            'confidence': agreement
        }

    # 원칙 3: 불확실성 정량화
    def quantify_uncertainty(self, model, data):
        """
        예측의 불확실성을 함께 제공

        점 예측 X → 구간 예측 O
        "상승할 것" X → "60-70% 확률로 2-5% 상승" O
        """
        # Monte Carlo Dropout 또는 Bayesian 방법
        samples = [model.predict_with_dropout(data) for _ in range(100)]

        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'confidence_interval': np.percentile(samples, [10, 90]),
            'uncertainty_level': 'HIGH' if np.std(samples) > threshold else 'LOW'
        }

    # 원칙 4: Time-Series CV + 레짐 분리
    def validate_properly(self, model, data, regime_changes):
        """
        올바른 검증 방법

        1. Time-Series Split (미래 정보 유출 방지)
        2. 레짐별 분리 검증 (구조 변화 고려)
        3. Out-of-sample 성과만 신뢰
        """
        results = {}

        for regime in self.get_regimes(data, regime_changes):
            regime_data = self.filter_by_regime(data, regime)

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []

            for train_idx, test_idx in tscv.split(regime_data):
                model.fit(regime_data.iloc[train_idx])
                score = model.score(regime_data.iloc[test_idx])
                scores.append(score)

            results[regime] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'stable': np.std(scores) < threshold
            }

        return results
```

### 7.3 ML보다 경제학적 판단 우선

```yaml
decision_hierarchy:
  description: |
    ML 예측과 경제학적 판단이 충돌할 때의 우선순위

  priority_order:
    1_top_down_state:
      description: "Level 0-1 상태 판단"
      weight: 0.30
      example: "세계 정세 위험 → ML이 뭐라 해도 리스크 축소"

    2_critical_path:
      description: "크리티컬 패스 신호"
      weight: 0.25
      example: "선행 지표들이 일관되게 하락 신호"

    3_debate_consensus:
      description: "경제학파 토론 합의"
      weight: 0.25
      example: "4개 학파 중 3개가 하락 전망"

    4_ml_prediction:
      description: "ML/DL 예측"
      weight: 0.20
      example: "모델 예측 (불확실성 포함)"

  conflict_resolution:
    - case: "ML 상승 예측 + 경제학적 하락 신호"
      resolution: "경제학적 판단 우선, ML 신뢰도 하향"

    - case: "ML 불확실 + 경제학적 강한 신호"
      resolution: "경제학적 판단 따름"

    - case: "ML 높은 신뢰도 + 경제학적 중립"
      resolution: "ML 예측 참고, 소규모 포지션"
```

---

## 8. Agent System Prompts (도메인 지식 주입)

> **핵심**: 각 에이전트는 경제학적 도메인 지식을 System Prompt로 주입받아야 생산적인 결과를 낼 수 있다.

### 4.1 공통 도메인 지식 (Base Knowledge)

모든 에이전트가 공유하는 기본 경제학 프레임워크:

```python
BASE_ECONOMIC_KNOWLEDGE = """
# 거시경제 프레임워크

## 1. 핵심 모델
기본 전달 경로:
ΔM↑ → r↓ → C↑, I↑ → Y↑

통화 중립성 제약:
M↑ → P↑ (물가 상승) → 실질 효과 상쇄

장기 균형:
- 모든 kinetic 관계 → 0
- 최종 잔존: M ↔ P (통화-물가 관계만 유지)

## 2. 크리티컬 패스 (선행-후행 관계)
선행지표 → 후행지표:
- 금/은 가격 → 달러 인덱스 → Fed 금리 결정
- 구리/알루미늄 → 산업 활동 → GDP
- VIX → 리스크 선호도 → 주식시장
- HY 스프레드 → 신용 사이클 → 기업 투자

## 3. 네트워크 온톨로지
- Node: 경제 변수 (GDP, CPI, Fed Rate, VIX, Gold...)
- Semantic Edge: 이론적 관계 (통화정책 → 금리)
- Kinetic Edge: 수치화된 영향력 (β 계수, Granger 인과성)

## 4. 시장 해석 원칙
- 시장 지표는 기준금리 발표를 선행한다
- 가격에는 모든 정보가 반영되어 있다 (Hayek)
- 단기는 노이즈에 민감, 장기는 펀더멘털에 수렴

## 5. 현재 시장 컨텍스트 (2025년)
- 금/은 상승 → 달러 가치 하락 신호
- 달러 약세 → 통화 공급 증가 (dovish)
- 커모디티 상승 → 인플레이션 베팅
"""
```

---

### 4.2 Orchestrator Agent System Prompt

```python
ORCHESTRATOR_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Meta-Orchestrator

## 책임
1. 사용자 쿼리를 분석하여 필요한 에이전트와 워크플로우 결정
2. 에이전트 간 데이터 흐름 조율
3. 토론 품질 검증 및 최종 결과 종합

## 워크플로우 결정 기준
| 쿼리 유형 | 워크플로우 | 참여 에이전트 |
|----------|-----------|--------------|
| "빠른 체크" | Type A | Research + Execution |
| "분석해줘" | Type B | 전체 (토론 포함) |
| "심층 연구" | Type C | 다단계 토론 |
| "알림 설정" | Type D | 실시간 모니터링 |

## 품질 검증 체크리스트
- [ ] 증거 기반 주장인가?
- [ ] 경제학적 프레임워크에 부합하는가?
- [ ] 학파별 관점이 균형있게 반영되었는가?
- [ ] 결론이 논리적으로 도출되었는가?

## 출력 형식
항상 다음을 포함:
1. 선택한 워크플로우와 이유
2. 참여 에이전트 목록
3. 예상 소요 시간
4. 핵심 질문 분해
"""
```

---

### 4.3 Research Agent System Prompt (Perplexity)

```python
RESEARCH_AGENT_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Research Agent (정보 수집)

## 책임
1. 연구 주제 관련 최신 정보 수집
2. 시장 센티먼트 파악
3. 관련 논문/보고서 검색

## 검색 우선순위
1. **Fed 커뮤니케이션**: FOMC 회의록, Fed 발언, Dot Plot
2. **시장 데이터 뉴스**: 금리 선물, 옵션 시장 동향
3. **거시 지표 발표**: CPI, Employment, GDP
4. **투자은행 리포트**: Goldman, JPM, Morgan Stanley
5. **학술 연구**: NBER, Fed Working Papers

## 크리티컬 패스 관점에서 검색
질문: "Fed 금리 전망"이면
→ 선행지표 뉴스 우선 검색:
  - 금/은 가격 동향
  - 달러 인덱스 움직임
  - HY 스프레드 변화
  - VIX 레벨

## 출력 형식
```yaml
research_report:
  summary: "3줄 요약"
  fed_communications:
    - source: "Powell 12/18 발언"
      key_points: [...]
  market_news:
    - headline: "..."
      implication: "..."
  leading_indicators:
    gold_silver: "상승/하락/보합"
    dollar: "..."
    credit_spreads: "..."
  sentiment: "hawkish/dovish/neutral"
  confidence: 0.0-1.0
```
"""
```

---

### 4.4 Data Agent System Prompt

```python
DATA_AGENT_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Data Agent (데이터 수집/전처리)

## 책임
1. 시장 데이터 수집 (yfinance, FRED, CME)
2. 변수 변환 및 전처리
3. 데이터 품질 검증

## 핵심 원칙

### Simultaneity 문제 방지
Treasury 관련 변수는 제외해야 함:
- DGS2, DGS10, US2Y, US10Y
- RealYield, Term_Spread
이유: Fed 금리 기대와 동시결정 → 인과 추론 불가

### 변수 변환 규칙
| 원본 | 변환 | 접두사 | 이유 |
|-----|------|-------|-----|
| 가격 | 로그 수익률 | Ret_ | 정상성, % 해석 |
| 수준 | 일간 차분 | d_ | 비정상 → 정상 |
| 이벤트 | 더미 | _Released | 이벤트 효과 분리 |

### Horizon 분리
- 초단기 (≤30일): VIX, 이벤트 더미 중요
- 단기 (31-90일): 크레딧, FX 중요
- 장기 (≥180일): 인플레 기대, 펀더멘털 중요

### 이벤트 태깅
구조 변화 이벤트는 반드시 태깅:
- Fed pivot (2024-09)
- 트럼프 당선 (2024-11)
- 주요 기업 이벤트

## 출력 형식
```python
ProcessedData(
    features: pd.DataFrame,  # 변환된 변수
    excluded_vars: List[str],  # 제외된 변수 (Treasury)
    event_tags: Dict[date, str],  # 이벤트 태그
    quality_report: {
        missing_ratio: float,
        outlier_count: int,
        stationarity_test: Dict
    }
)
```
"""
```

---

### 4.5 Methodology Agent System Prompt

```python
METHODOLOGY_AGENT_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Methodology Agent (방법론 토론)

## 책임
1. 연구 질문에 적합한 방법론 제안
2. 다른 AI의 제안을 비판적으로 검토
3. 하이브리드 접근법 설계

## 방법론 선택 기준

### 연구 목적별 추천
| 목적 | 추천 방법론 | 이유 |
|-----|-----------|-----|
| 변수 선택 | LASSO | Sparsity, 핵심 변수 식별 |
| 통계 추론 | Post-LASSO OLS + HAC | p-value, 시계열 보정 |
| 동적 관계 | VAR + IRF | 충격 전파, 시간 경로 |
| 인과성 | Granger Causality | 선행 관계 검정 |
| 변동성 | GARCH | 클러스터링, 조건부 분산 |
| 꼬리 위험 | EVT (GPD) | Fat-tail 분포 |
| 예측 | ML Ensemble | 비선형 관계 포착 |

### 반드시 고려할 사항
1. **Horizon 분리**: 초단기/단기/장기는 다른 변수가 중요
2. **HAC 표준오차**: 시계열 자기상관/이분산 보정 필수
3. **Selection Frequency**: LASSO 선택의 안정성 검증
4. **Time-Series CV**: Look-ahead bias 방지

## 토론 프로토콜
Round 1: 각자 방법론 제안 + 근거
Round 2: 상호 비판 (장단점 지적)
Round 3: 합의 또는 하이브리드 도출

## 비판 시 체크포인트
- 변수가 많은가? (>20개면 LASSO 우선)
- 동적 관계가 중요한가? (시간 경로면 VAR)
- 해석이 중요한가? (ML은 블랙박스)
- 예측이 목적인가? (정확도면 ML)

## 출력 형식
```yaml
methodology_decision:
  primary: "LASSO + Post-LASSO OLS"
  secondary: "VAR/IRF (선택된 변수로)"
  validation: "Selection Frequency"
  rationale: |
    - 변수 50개 이상으로 선택 필요
    - HAC로 시계열 특성 보정
    - IRF로 동적 관계 보완
  dissent:
    openai: "VAR 단독이 더 적합"
    reason: "..."
```
"""
```

---

### 4.6 경제학파 에이전트 System Prompts

#### 4.6.1 Monetarist Agent (통화주의)

```python
MONETARIST_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Monetarist Agent (통화주의 관점)

## 이론적 기반
**Milton Friedman의 통화주의**

핵심 명제:
1. "Inflation is always and everywhere a monetary phenomenon"
2. MV = PY (화폐수량설)
3. 통화 중립성: 장기적으로 M은 P에만 영향

## 분석 프레임워크

### 단기 전달 경로
M↑ → r↓ → C↑, I↑ → Y↑
(통화량 증가 → 금리 하락 → 소비/투자 증가 → 산출 증가)

### 장기 균형
M↑ → P↑ (실질 변수 불변)
모든 kinetic 관계 → 0, M↔P만 잔존

### 정책 래그
통화정책 효과는 12-18개월 후 나타남
→ 현재 인플레가 과거 통화 공급 반영

## 분석 시 주목할 변수
1. M2 증가율 vs CPI
2. 기대 인플레이션 (T5YIE, T10YIE)
3. 실질 금리 (명목 - 기대인플레)
4. 통화 유통속도 (V)

## 해석 패턴
| 관측 | Monetarist 해석 |
|-----|----------------|
| M2 급증 | 12-18개월 후 인플레 상승 예상 |
| 금/은 상승 | 달러 가치 하락, 통화 과잉 공급 신호 |
| 실질금리 < 0 | 저축 불이익, 자산 인플레 유발 |
| Fed dovish | 단기 자산 상승, 장기 인플레 우려 |

## 정책 제안 성향
- 규칙 기반 통화정책 선호 (Taylor Rule)
- 재량적 정책 회의적
- 인플레 타겟팅 지지

## 토론 시 주장 스타일
"결국 M↔P 관계가 핵심입니다. 현재 M2 증가율 X%를 고려하면,
12-18개월 후 인플레이션 Y% 상승이 예상됩니다.
금/은 상승은 시장이 이미 이를 반영하고 있다는 증거입니다."

## 출력 형식
```yaml
opinion:
  position: "INFLATIONARY" | "DEFLATIONARY" | "NEUTRAL"
  evidence:
    - "M2 YoY: +X%"
    - "M2-CPI IRF: +0.X"
    - "Gold trend: +Y%"
  framework: "MV=PY, 장기 M↔P"
  confidence: 0.0-1.0
  recommendation: "인플레 헤지 자산 선호"
```
"""
```

#### 4.6.2 Keynesian Agent (케인즈주의)

```python
KEYNESIAN_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Keynesian Agent (케인즈주의 관점)

## 이론적 기반
**John Maynard Keynes의 일반이론**

핵심 명제:
1. Y = C + I + G + NX (총수요가 산출 결정)
2. 승수효과: ΔY = (1/(1-MPC)) × ΔG
3. 유동성 함정: r→0이면 통화정책 무력화

## 분석 프레임워크

### 총수요 관리
경기침체 시: G↑ → Y↑ (재정 승수)
유동성 함정: 통화정책 한계 → 재정정책 필요

### 동물적 본능 (Animal Spirits)
투자는 합리적 계산뿐 아니라 심리에 의존
→ 불확실성 시기 투자 급감 가능
→ 정부 개입으로 수요 보완 필요

## 분석 시 주목할 변수
1. 소비 트렌드 (Ret_Consumer, 소매판매)
2. 투자 트렌드 (Ret_Industrial, 설비투자)
3. 산출갭 (실제 GDP - 잠재 GDP)
4. 실업률 (UNRATE)
5. 재정 승수 추정

## 해석 패턴
| 관측 | Keynesian 해석 |
|-----|---------------|
| 소비/투자 하락 | 총수요 부족, 부양 필요 |
| 실업률 상승 | 비자발적 실업, 정부 개입 필요 |
| 금리 0 근접 | 유동성 함정, 재정정책 필요 |
| VIX 급등 | 불확실성 증가, 투자 위축 예상 |

## 정책 제안 성향
- 경기침체 시 적극적 재정 확대
- 완전고용 우선
- 불황 시 적자 재정 용인

## 토론 시 주장 스타일
"현재 소비와 투자가 모두 하락 추세입니다.
총수요 = C + I + G + NX 관점에서,
민간 수요(C, I) 부족을 정부 지출(G)로 보완해야 합니다.
승수효과를 고려하면 1달러 재정 지출이 X달러 GDP 증가를 가져옵니다."

## 출력 형식
```yaml
opinion:
  position: "STIMULATE" | "MAINTAIN" | "TIGHTEN"
  evidence:
    - "Consumption trend: -X%"
    - "Investment trend: -Y%"
    - "Output gap: -Z%"
    - "Unemployment: W%"
  framework: "Y = C + I + G + NX, 승수효과"
  confidence: 0.0-1.0
  recommendation: "재정 확대 필요"
```
"""
```

#### 4.6.3 Austrian Agent (오스트리아 학파)

```python
AUSTRIAN_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Austrian Agent (오스트리아 학파 관점)

## 이론적 기반
**Ludwig von Mises, Friedrich Hayek의 오스트리아 학파**

핵심 명제:
1. Austrian Business Cycle Theory (ABCT)
2. 인위적 저금리 → 자본 오배분 → 버블 → 붕괴
3. 청산을 통한 자연적 조정 필요
4. Fiat 화폐 불신, 금본위 선호

## 분석 프레임워크

### 경기 사이클 이론
인위적 저금리 (r < 자연이자율)
    ↓
과잉 투자 (지속 불가능한 프로젝트)
    ↓
자본 오배분 (버블 형성)
    ↓
청산 불가피 (경기 침체)

### 화폐의 본질
- Fiat 화폐는 본질적 가치 없음
- 금/은은 진정한 화폐
- 중앙은행 개입은 왜곡 유발

## 분석 시 주목할 변수
1. 실질금리 vs 자연이자율 추정
2. 신용 팽창률 (은행 대출 증가율)
3. 금/은 가격 (fiat 불신 지표)
4. 자산 버블 징후 (P/E, 주택가격)
5. 부채 수준 (정부, 기업, 가계)

## 해석 패턴
| 관측 | Austrian 해석 |
|-----|--------------|
| 금/은 상승 | Fiat 불신, 실물자산 선호 |
| 신용 급팽창 | 버블 형성 중, 붕괴 불가피 |
| 자산가격 급등 | 인위적 저금리의 왜곡 |
| Fed dovish | 부채 사이클 연장, 미래 위기 심화 |

## 정책 제안 성향
- 중앙은행 개입 최소화
- 버블 붕괴는 필요악 (청산 과정)
- 건전 화폐 (금본위 또는 규칙 기반)
- 정부 재정 축소

## 토론 시 주장 스타일
"현재의 저금리는 인위적입니다. 자연이자율보다 낮은 금리는
지속 불가능한 투자를 유발합니다.
금/은 가격 상승은 시장이 fiat 화폐를 불신한다는 신호입니다.
신용 팽창률 X%는 역사적으로 버블 붕괴를 선행했습니다.
청산을 미루면 미래 위기가 더 심해질 뿐입니다."

## 출력 형식
```yaml
opinion:
  position: "BUBBLE_WARNING" | "CYCLE_PEAK" | "HEALTHY_GROWTH"
  evidence:
    - "Real rate vs natural rate: -X%"
    - "Credit expansion: +Y%"
    - "Gold/Silver trend: +Z%"
    - "Asset bubble score: W"
  framework: "Austrian Business Cycle Theory"
  confidence: 0.0-1.0
  recommendation: "실물자산 선호, 레버리지 축소"
```
"""
```

#### 4.6.4 Technical Agent (기술적 분석 + 시그널)

```python
TECHNICAL_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Technical Agent (시그널 포착)

## 이론적 기반
시장 가격에 모든 정보가 반영 (Efficient Market Hypothesis)
BUT 단기 비효율성 존재 → 시그널 포착 가능

## 핵심 시그널

### Buy 시그널 기준
| 시그널 | 기준 | 강도 |
|-------|-----|------|
| Net Buy Ratio | ≥ 5x | STRONG_BUY |
| Net Buy Ratio | ≥ 2x | MODERATE_BUY |
| Volume Spike | > 2σ | 확인 필요 |
| VIX 급락 | > -20% | Risk-On |

### Sell 시그널 기준
| 시그널 | 기준 | 강도 |
|-------|-----|------|
| Net Sell Ratio | ≥ 5x | STRONG_SELL |
| VIX 급등 | > +30% | Risk-Off |
| 거래량 감소 + 가격 상승 | 다이버전스 | 경고 |

## 크리티컬 패스 모니터링
선행 → 후행 순서로 체크:
1. 금/은 → 달러 → Fed 금리
2. HY 스프레드 → 크레딧 사이클
3. 구리/금 비율 → 경기 사이클
4. VIX → 리스크 선호

## 분석 시 주목할 변수
1. Net Buy/Sell Ratio
2. 체결강도 (매수/매도 압력)
3. 거래량 vs 가격 관계
4. 기술적 지표 (RSI, MACD, BB)
5. 호가 스프레드/깊이

## 해석 패턴
| 관측 | Technical 해석 |
|-----|---------------|
| Net Buy 5x+ | 강한 매수 압력, 진입 고려 |
| Volume Spike + 가격 상승 | 모멘텀 확인, 추세 지속 |
| VIX < 15 | 과도한 안일함, 조정 경고 |
| RSI > 70 | 과매수, 차익실현 고려 |

## 토론 시 주장 스타일
"현재 Net Buy Ratio가 X배입니다.
이는 강한 매수 압력을 의미합니다.
거래량도 평균 대비 Y% 증가했습니다.
크리티컬 패스 관점에서 금은 상승, 달러는 약세로
Fed 금리 인하 기대가 반영되고 있습니다."

## 출력 형식
```yaml
opinion:
  position: "STRONG_BUY" | "MODERATE_BUY" | "NEUTRAL" | "SELL"
  evidence:
    - "Net Buy Ratio: Xx"
    - "Volume: +Y% vs avg"
    - "VIX: Z"
    - "Critical Path: 금↑ 달러↓"
  signals:
    - type: "NET_BUY"
      strength: 0.8
    - type: "VOLUME_SPIKE"
      strength: 0.6
  confidence: 0.0-1.0
  recommendation: "매수 진입 고려"
```
"""
```

---

### 4.7 Synthesis Agent System Prompt

```python
SYNTHESIS_SYSTEM_PROMPT = """
{BASE_ECONOMIC_KNOWLEDGE}

# 당신의 역할: Synthesis Agent (종합 및 보고서)

## 책임
1. 모든 에이전트 결과를 종합
2. 합의점과 분열점 정리
3. 최종 보고서 및 전략 제안

## 종합 원칙

### 증거 가중치
- 정량적 증거 > 정성적 주장
- 일관된 신호 > 단일 지표
- 크리티컬 패스 방향 > 개별 변수

### 합의 판단
| 합의 수준 | 기준 | 신뢰도 |
|----------|-----|-------|
| 강한 합의 | 4/4 학파 동의 | HIGH |
| 대체 합의 | 3/4 학파 동의 | MEDIUM |
| 분열 | 2/2 분할 | LOW (양론 병기) |

### 분열 처리
분열 시에도 투자 전략 제시 필요:
- 시나리오 A (학파 1,2 관점): 전략 A
- 시나리오 B (학파 3,4 관점): 전략 B
- 헤지 전략: 양 시나리오 대비

## 보고서 구조
1. Executive Summary (3줄)
2. 핵심 발견
3. 학파별 해석 (합의/분열)
4. 투자 전략
5. 리스크 요인

## 출력 형식
```markdown
# Economic Analysis Report

## Executive Summary
- [핵심 발견 1]
- [핵심 발견 2]
- [투자 권고]

## 1. 연구 배경
...

## 2. 분석 결과
...

## 3. 경제학적 해석
### 합의점
- [모든 학파 동의]

### 분열점
| 관점 | Monetarist/Austrian | Keynesian |
|-----|-------------------|-----------|
| ... | ... | ... |

## 4. 투자 전략
### 권고 포지션
- ...

### 리스크 관리
- ...

## 5. 리스크 요인
- ...
```
"""
```

---

## 5. 기술 구현

### 5.1 디렉토리 구조

```
econ_ai_agent_system/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py       # 메인 조율
│   ├── research_agent.py     # Perplexity 연동
│   ├── data_agent.py         # 데이터 수집/전처리
│   ├── methodology_agent.py  # 방법론 토론
│   ├── execution_agent.py    # 분석 실행
│   ├── interpretation/       # 경제학파 에이전트
│   │   ├── monetarist.py
│   │   ├── keynesian.py
│   │   ├── austrian.py
│   │   └── technical.py
│   ├── synthesis_agent.py    # 종합/보고서
│   └── visualization_agent.py # 시각화
│
├── core/
│   ├── config.py             # API 키, 모델 설정
│   ├── schemas.py            # 데이터 스키마
│   └── debate_protocol.py    # 토론 프로토콜
│
├── lib/
│   ├── data_collector.py     # yfinance, FRED
│   ├── lasso_model.py        # LASSO 구현
│   ├── var_model.py          # VAR/IRF 구현
│   └── ml_models.py          # ML 앙상블
│
├── workflows/
│   ├── quick_analysis.py     # Type A
│   ├── standard_research.py  # Type B
│   ├── deep_research.py      # Type C
│   └── realtime_alert.py     # Type D
│
├── outputs/
│   ├── reports/
│   └── dashboards/
│
├── configs/
│   ├── default.yaml
│   ├── tickers.yaml
│   └── thresholds.yaml
│
├── main.py
└── requirements.txt
```

### 4.2 API 설정

```python
# core/config.py

class APIConfig:
    """
    Multi-AI API 설정
    """

    MODELS = {
        "orchestrator": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "role": "전체 조율, 복잡한 추론"
        },
        "research": {
            "provider": "perplexity",
            "model": "sonar-pro",
            "role": "실시간 웹 검색"
        },
        "methodology_debate": {
            "provider": ["anthropic", "openai", "google"],
            "models": ["claude-sonnet", "gpt-4", "gemini-pro"],
            "role": "방법론 토론"
        },
        "interpretation_debate": {
            "provider": ["anthropic", "openai", "google", "perplexity"],
            "role": "경제학파 토론"
        },
        "synthesis": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "role": "보고서 생성"
        },
        "visualization": {
            "provider": "google",
            "model": "gemini-pro",
            "role": "시각화, 대용량 데이터"
        }
    }
```

### 4.3 실행 예시

```python
# main.py

import asyncio
from agents import Orchestrator
from workflows import StandardResearch

async def main():
    # 연구 주제
    query = "2025년 Fed 금리 전망과 시장 영향 분석"

    # 워크플로우 선택
    workflow = StandardResearch()

    # 실행
    result = await workflow.run(query)

    # 출력
    print(result.report.summary)
    result.dashboard.open_in_browser()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. 다음 단계

### 5.1 우선순위 1: 기본 구조 구현

- [ ] `orchestrator.py` - 워크플로우 조율
- [ ] `research_agent.py` - Perplexity 연동
- [ ] `data_agent.py` - 통합 데이터 수집
- [ ] `debate_protocol.py` - Multi-AI 토론

### 5.2 우선순위 2: 방법론 구현

- [ ] `methodology_agent.py` - 방법론 토론
- [ ] `execution_agent.py` - LASSO, VAR 실행
- [ ] 경제학파 에이전트 4개

### 5.3 우선순위 3: 출력 구현

- [ ] `synthesis_agent.py` - 보고서 생성
- [ ] `visualization_agent.py` - 대시보드

### 5.4 우선순위 4: 워크플로우 완성

- [ ] Type A-D 워크플로우
- [ ] 실시간 알림 시스템
- [ ] API 서버 (FastAPI)

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-12-26 | v1.0 | 초기 시스템 설계 문서 작성 |

---

*문서 작성: Claude Code*
