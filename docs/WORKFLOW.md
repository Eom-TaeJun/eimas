# EIMAS Workflow & Architecture Guide

> **버전**: v2.2.0 (2026-01-28)
> **목적**: 이 문서 하나만 읽으면 EIMAS 전체 구조와 기능을 파악할 수 있도록 작성

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [데이터 파이프라인](#3-데이터-파이프라인)
4. [핵심 모듈 상세](#4-핵심-모듈-상세)
5. [AI 에이전트 시스템](#5-ai-에이전트-시스템)
6. [Economic Insight Agent](#6-economic-insight-agent)
7. [API 및 프론트엔드](#7-api-및-프론트엔드)
8. [실행 방법](#8-실행-방법)
9. [디렉토리 구조](#9-디렉토리-구조)
10. [경제학적 방법론](#10-경제학적-방법론)

---

## 1. 프로젝트 개요

### EIMAS란?

**Economic Intelligence Multi-Agent System**의 약자로, 다음 기능을 수행하는 AI 기반 경제 분석 시스템입니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                         EIMAS 핵심 기능                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 데이터 수집    │ FRED(연준), yfinance(시장), 크립토/RWA      │
│  2. 레짐 탐지      │ Bull/Bear/Neutral 시장 상태 판단            │
│  3. 리스크 분석    │ 유동성, 버블, 미세구조 다차원 평가          │
│  4. AI 토론       │ Claude/GPT-4/Gemini 멀티에이전트 합의       │
│  5. 권고 생성      │ BULLISH/BEARISH/NEUTRAL + 신뢰도           │
│  6. 인과 분석      │ 전달 메커니즘과 반증 가설 생성 (NEW)        │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 철학

1. **Causality-first**: 단순 예측보다 "왜" 그런 결과가 나오는지 설명
2. **JSON-first**: 모든 출력은 구조화된 JSON으로 재현 가능
3. **Multi-Agent Debate**: 단일 AI 의견이 아닌 여러 관점의 토론과 합의
4. **Economic Rigor**: 학술적으로 검증된 경제학 방법론 적용

---

## 2. 전체 아키텍처

### 2.1 High-Level 데이터 흐름

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           EIMAS Data Flow                                 │
└──────────────────────────────────────────────────────────────────────────┘

  [External Data Sources]
         │
         ▼
┌─────────────────┐
│  Phase 1: DATA  │  FRED API, yfinance, CoinGecko
│  COLLECTION     │  → fred_summary, market_data (24 tickers + crypto + RWA)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 2:       │  RegimeDetector, GMM, LiquidityAnalyzer,
│  ANALYSIS       │  CriticalPath, Microstructure, BubbleDetector,
│  (10개 서브모듈) │  ShockPropagation, GC-HRP Portfolio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 3:       │  MetaOrchestrator
│  MULTI-AGENT    │  → Claude/GPT-4/Gemini 토론
│  DEBATE         │  → FULL mode vs REFERENCE mode 비교
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 4:       │  BinanceStreamer (WebSocket)
│  REALTIME       │  → OFI, VPIN 실시간 계산
│  (선택)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 5:       │  SQLite (events.db, signals.db)
│  STORAGE        │  → JSON, Markdown 저장
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Phase 6-7:     │  AIReportGenerator (Claude/Perplexity)
│  AI REPORT &    │  → WhiteningEngine (경제학적 해석)
│  VALIDATION     │  → AutonomousFactChecker (팩트체킹)
└─────────────────┘
```

### 2.2 모듈 의존성 그래프

```
lib/fred_collector.py ──┐
lib/data_collector.py ──┼──▶ Phase 1: Data
lib/data_loader.py ─────┘

lib/regime_detector.py ────┐
lib/regime_analyzer.py ────┤
lib/event_framework.py ────┤
lib/liquidity_analysis.py ─┤
lib/critical_path.py ──────┼──▶ Phase 2: Analysis
lib/microstructure.py ─────┤
lib/bubble_detector.py ────┤
lib/shock_propagation.py ──┤
lib/graph_clustered_portfolio.py ─┤
lib/integrated_strategy.py ┘

agents/orchestrator.py ────┬──▶ Phase 3: Debate
agents/analysis_agent.py ──┤
pipeline/debate.py ────────┘

lib/binance_stream.py ─────▶ Phase 4: Realtime

lib/ai_report_generator.py ┬──▶ Phase 6-7: Report
lib/whitening_engine.py ───┤
lib/autonomous_agent.py ───┘

agent/core/orchestrator.py ─▶ Economic Insight Agent (독립 실행 가능)
```

---

## 3. 데이터 파이프라인

### 3.1 Phase 1: 데이터 수집

| 소스 | 모듈 | 수집 데이터 |
|------|------|-------------|
| **FRED** | `lib/fred_collector.py` | RRP, TGA, Fed Balance Sheet, Fed Funds Rate |
| **yfinance** | `lib/data_collector.py` | SPY, QQQ, TLT, GLD, VIX 등 24개 티커 |
| **Crypto** | `lib/data_loader.py` | BTC-USD, ETH-USD |
| **RWA** | `lib/data_loader.py` | ONDO-USD, PAXG-USD, COIN (토큰화 자산) |

**핵심 지표 계산**:
```python
# 순유동성 (Fed Liquidity)
Net_Liquidity = Fed_Balance_Sheet - RRP - TGA

# 확장 유동성 (Genius Act Model)
M = B + S·B*  # 순유동성 + 스테이블코인 기여도
```

### 3.2 Phase 2: 분석 (10개 서브모듈)

| Phase | 모듈 | 출력 | 설명 |
|-------|------|------|------|
| 2.1 | `RegimeDetector` | regime (Bull/Bear/Neutral) | 시장 상태 판단 |
| 2.1.1 | `GMMRegimeAnalyzer` | GMM 확률 + Entropy | 3-state 분류 + 불확실성 |
| 2.2 | `QuantitativeEventDetector` | events[] | 이벤트 탐지 |
| 2.3 | `LiquidityMarketAnalyzer` | liquidity_signal | Granger Causality |
| 2.4 | `CriticalPathAggregator` | base_risk_score | 리스크 점수 (0-100) |
| 2.4.1 | `DailyMicrostructureAnalyzer` | market_quality | Amihud, VPIN |
| 2.4.2 | `BubbleDetector` | bubble_risk | Greenwood-Shleifer |
| 2.5 | `ETFFlowAnalyzer` | sector_rotation | 섹터 자금흐름 |
| 2.6 | `GeniusActMacroStrategy` | genius_act_regime | 스테이블코인-유동성 |
| 2.7 | `CustomETFBuilder` | theme_etf | 테마 ETF 분석 |
| 2.8 | `ShockPropagationGraph` | shock_paths[] | 충격 전파 경로 |
| 2.9 | `GraphClusteredPortfolio` | portfolio_weights | HRP + MST 최적화 |
| 2.10 | `IntegratedStrategy` | integrated_signals | 통합 전략 시그널 |

**리스크 점수 계산 (v2.1.1)**:
```python
Final_Risk = Base_Risk + Microstructure_Adj + Bubble_Adj

# Base_Risk: CriticalPathAggregator (0-100)
# Microstructure_Adj: (50 - avg_liquidity) / 5, clamped ±10
# Bubble_Adj: NONE=0, WATCH=+5, WARNING=+10, DANGER=+15
```

### 3.3 Phase 3: AI 토론

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Debate System                     │
└─────────────────────────────────────────────────────────────────┘

                    MetaOrchestrator
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │  Claude   │   │  GPT-4    │   │  Gemini   │
   │  Agent    │   │  Agent    │   │  Agent    │
   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │   Debate Protocol   │
              │   (최대 3라운드)     │
              │   85% 일관성 임계값  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │     Consensus       │
              │  BULLISH/BEARISH/   │
              │     NEUTRAL         │
              └─────────────────────┘
```

**Dual Mode 분석**:
- **FULL Mode**: 365일 데이터 기반 장기 분석
- **REFERENCE Mode**: 90일 데이터 기반 단기 분석
- **비교**: 두 모드의 합의 여부 및 dissent 기록

### 3.4 Phase 4-7: 실시간, 저장, 리포트

| Phase | 기능 | 트리거 |
|-------|------|--------|
| 4 | 실시간 WebSocket (Binance) | `--realtime` |
| 5 | DB 저장 (SQLite) | 항상 |
| 6 | AI 리포트 생성 | `--report` |
| 7 | Whitening + Fact Check | `--report` |

---

## 4. 핵심 모듈 상세

### 4.1 레짐 분석 (`lib/regime_analyzer.py`)

```python
class GMMRegimeAnalyzer:
    """GMM 3-State 레짐 분류 + Shannon Entropy"""

    def analyze(self, returns: pd.Series) -> Dict:
        # GMM 3개 상태: Bull(고수익/저변동), Neutral, Bear(저수익/고변동)
        gmm = GaussianMixture(n_components=3)
        states = gmm.fit_predict(returns)

        # Shannon Entropy로 불확실성 측정
        probs = gmm.predict_proba(returns[-1:])
        entropy = -sum(p * log(p) for p in probs if p > 0)

        return {
            'regime': 'BULL' | 'NEUTRAL' | 'BEAR',
            'probability': float,
            'entropy': float,  # 낮을수록 확신
            'entropy_level': 'Very Low' | 'Low' | 'Medium' | 'High' | 'Very High'
        }
```

### 4.2 버블 탐지 (`lib/bubble_detector.py`)

```python
class BubbleDetector:
    """Greenwood-Shleifer (2019) 버블 탐지"""

    def detect(self, prices: pd.DataFrame) -> Dict:
        # 2년 누적 수익률 > 100% → 버블 위험
        two_year_return = (price_now / price_2y_ago - 1) * 100

        # Volatility Spike (Z-score > 2)
        vol_zscore = (current_vol - mean_vol) / std_vol

        return {
            'overall_status': 'NONE' | 'WATCH' | 'WARNING' | 'DANGER',
            'risk_tickers': [{'ticker': str, 'run_up_pct': float}],
            'highest_risk_ticker': str,
            'highest_risk_score': float
        }
```

### 4.3 포트폴리오 최적화 (`lib/graph_clustered_portfolio.py`)

```python
class GraphClusteredPortfolio:
    """HRP + MST 기반 포트폴리오 최적화"""

    def optimize(self, returns: pd.DataFrame) -> Dict:
        # 1. 상관관계 → 거리 변환 (Mantegna 1999)
        distance = sqrt(2 * (1 - correlation))

        # 2. MST (Minimum Spanning Tree) 구축
        mst = minimum_spanning_tree(distance_matrix)

        # 3. 중심성 분석 (시스템 리스크 허브 식별)
        centrality = {
            'betweenness': 0.45,  # 충격 전파 핵심
            'degree': 0.35,       # 허브 식별
            'closeness': 0.20    # 정보 흐름
        }

        # 4. HRP (Hierarchical Risk Parity)
        weights = hrp_allocation(returns)

        return {
            'weights': Dict[str, float],  # 합계 = 1.0
            'systemic_risk_nodes': List[str],
            'clusters': Dict[int, List[str]]
        }
```

### 4.4 충격 전파 분석 (`lib/shock_propagation_graph.py`)

```python
class ShockPropagationGraph:
    """Granger Causality 기반 충격 전파 경로 분석"""

    def analyze(self, data: pd.DataFrame) -> Dict:
        # 1. Granger Causality 테스트
        for pair in asset_pairs:
            p_value = granger_test(cause, effect, lag)
            if p_value < 0.05:
                edges.append((cause, effect, lag, p_value))

        # 2. Lead-Lag 관계 추출
        lead_lag = cross_correlation_analysis(data)

        # 3. Critical Path 식별
        critical_path = dijkstra(graph, source='Fed_Policy')

        return {
            'granger_results': List[Dict],
            'lead_lag_results': List[Dict],
            'shock_paths': [{'path': List[str], 'strength': float}],
            'node_analysis': Dict[str, Dict]
        }
```

---

## 5. AI 에이전트 시스템

### 5.1 에이전트 구조 (`agents/`)

```
agents/
├── base_agent.py           # BaseAgent 추상 클래스
├── orchestrator.py         # MetaOrchestrator (토론 조정)
├── analysis_agent.py       # CriticalPath 기반 분석
├── forecast_agent.py       # LASSO 예측
├── research_agent.py       # Perplexity API 연동
├── strategy_agent.py       # 전략 권고
├── visualization_agent.py  # 시각화
├── methodology_debate.py   # 방법론 토론
└── interpretation_debate.py # 해석 토론
```

### 5.2 MetaOrchestrator 워크플로우

```python
class MetaOrchestrator:
    """멀티에이전트 토론 조정"""

    async def run_with_debate(self, market_data: Dict) -> Dict:
        # 1. 토픽 자동 감지
        topics = self.auto_detect_topics(market_data)
        # ['market_outlook', 'primary_risk', 'regime_stability', 'crypto_correlation']

        # 2. 에이전트별 의견 수집
        opinions = {}
        for agent in [claude_agent, gpt4_agent, gemini_agent]:
            opinions[agent.name] = await agent.form_opinion(topics, market_data)

        # 3. 토론 프로토콜 실행
        consensus = self.debate_protocol.run_debate(
            opinions,
            max_rounds=3,
            consistency_threshold=0.85
        )

        # 4. 최종 보고서 종합
        return self.synthesize_report(consensus, opinions)
```

### 5.3 토론 프로토콜 (`core/debate.py`)

```python
class DebateProtocol:
    """Rule-based 토론 (LLM 호출 없이 빠른 합의)"""

    def run_debate(self, opinions: Dict, max_rounds: int = 3) -> Consensus:
        for round in range(max_rounds):
            # 1. 의견 일관성 계산
            consistency = self.calculate_consistency(opinions)

            # 2. 조기 종료 조건
            if consistency >= 0.85:
                return Consensus(agreed=True, position=majority_position)

            # 3. 갈등 식별 및 중재
            conflicts = self.identify_conflicts(opinions)
            opinions = self.mediate(opinions, conflicts)

        # 최대 라운드 후 다수결
        return Consensus(agreed=False, position=majority_position)
```

---

## 6. Economic Insight Agent

### 6.1 개요

`agent/` 디렉토리에 위치한 **독립적인 인과 분석 에이전트**입니다. 기존 EIMAS 모듈과 통합되거나 독립 실행 가능합니다.

```
기존 agents/ (토론 시스템)  ←→  agent/ (인과 분석)
         │                            │
         │    EIMASAdapter로 연결      │
         └─────────────────────────────┘
```

### 6.2 6단계 추론 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│              Economic Insight Agent Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Parse Request → Classify Frame
        │
        │  "스테이블코인 공급 증가가 국채 수요에 미치는 영향?"
        │  → Frame: CRYPTO
        ▼
Step 2: Build Initial Causal Graph Template
        │
        │  Stablecoin_Supply → Reserve_Demand → TBill_Demand
        ▼
Step 3: Map Evidence into Nodes/Edges
        │
        │  p-value, lag, confidence 업데이트
        ▼
Step 4: Generate Mechanism Paths
        │
        │  Path: [Stablecoin, Reserve, TBill], Signs: [+, +], Net: +
        ▼
Step 5: Generate Hypotheses + Falsification Tests
        │
        │  Main: "스테이블코인이 국채 수요 동인"
        │  Rival: "Fed 정책이 주 동인"
        │  Test: "스테이블코인 감소 시 국채 수요 감소 확인"
        ▼
Step 6: Produce Final JSON Report
        │
        │  {meta, phenomenon, causal_graph, mechanisms,
        │   hypotheses, risk, suggested_data, next_actions}
        ▼
```

### 6.3 EIMAS 어댑터

```python
class EIMASAdapter:
    """기존 EIMAS 모듈 출력 → Economic Insight Schema 변환"""

    # ShockPropagationGraph → CausalGraph
    def adapt_shock_propagation(self, spg_result: Dict) -> CausalGraph

    # CriticalPathAggregator → RegimeShiftRisk[]
    def adapt_critical_path(self, cp_result: Dict) -> List[RegimeShiftRisk]

    # GeniusActMacroStrategy → MechanismPath[]
    def adapt_genius_act(self, ga_result: Dict) -> List[MechanismPath]

    # BubbleDetector → RegimeShiftRisk[]
    def adapt_bubble_detector(self, bd_result: Dict) -> List[RegimeShiftRisk]

    # GraphClusteredPortfolio → NextAction[]
    def adapt_portfolio(self, gcp_result: Dict) -> List[NextAction]
```

### 6.4 JSON 출력 구조

```json
{
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO8601",
    "frame": "macro|markets|crypto|mixed",
    "modules_used": ["shock_propagation_graph", "critical_path"]
  },
  "phenomenon": "현상 요약 (한 문장)",
  "causal_graph": {
    "nodes": [{"id", "name", "layer", "category", "centrality"}],
    "edges": [{"source", "target", "sign", "lag", "p_value", "mechanism"}],
    "has_cycles": false,
    "critical_path": ["Fed_Policy", "Net_Liquidity", "SPY"]
  },
  "mechanisms": [
    {
      "nodes": ["A", "B", "C"],
      "edge_signs": ["+", "-"],
      "net_effect": "-",
      "narrative": "A 증가 → B 감소 → C 감소"
    }
  ],
  "hypotheses": {
    "main": {"statement": "...", "confidence": "high"},
    "rivals": [...],
    "falsification_tests": [
      {"description": "...", "data_required": [...], "expected_if_true": "..."}
    ]
  },
  "risk": {
    "regime_shift_risks": [...],
    "data_limitations": [...]
  },
  "suggested_data": [{"name", "category", "priority", "rationale"}],
  "next_actions": [{"description", "category", "priority", "timeframe"}]
}
```

### 6.5 사용법

```bash
# CLI
python -m agent.cli --question "Fed 금리 인상 영향은?"
python -m agent.cli --with-eimas --question "현재 시장 분석"

# Eval (10개 시나리오)
python -m agent.evals.runner  # ALL PASS
```

```python
# Python API
from agent import EconomicInsightOrchestrator, InsightRequest

orchestrator = EconomicInsightOrchestrator()
request = InsightRequest(question="스테이블코인 공급 증가 영향?")
report = orchestrator.run(request)
print(report.model_dump_json(indent=2))
```

---

## 7. API 및 프론트엔드

### 7.1 FastAPI 엔드포인트 (`api/`)

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 헬스체크 |
| GET | `/latest` | 최신 integrated JSON 반환 (대시보드용) |
| POST | `/analysis/run` | 분석 실행 |
| GET | `/regime/current` | 현재 레짐 |
| POST | `/debate/run` | 토론 실행 |
| POST | `/report/generate` | AI 리포트 생성 |

```bash
# 서버 실행
uvicorn api.main:app --reload --port 8000
```

### 7.2 프론트엔드 (`frontend/`)

**기술 스택**: Next.js 16, React 19, Tailwind CSS 4, SWR

```
frontend/
├── app/
│   ├── page.tsx          # 메인 대시보드
│   └── layout.tsx
├── components/
│   ├── MetricsGrid.tsx   # 5초 자동 폴링 메트릭 카드
│   ├── SignalsTable.tsx  # 시그널 테이블
│   └── ui/               # shadcn/ui 컴포넌트
└── lib/
    ├── api.ts            # fetchLatestAnalysis()
    └── types.ts          # TypeScript 인터페이스
```

```bash
# 프론트엔드 실행
cd frontend && npm run dev
# → http://localhost:3000
```

---

## 8. 실행 방법

### 8.1 설치

```bash
# 1. Python 의존성
pip install -r requirements.txt

# 2. 환경변수 설정
export ANTHROPIC_API_KEY="sk-ant-..."
export FRED_API_KEY="your-fred-key"
export OPENAI_API_KEY="sk-..."      # 선택
export GOOGLE_API_KEY="..."         # 선택
export PERPLEXITY_API_KEY="pplx-..." # 선택

# 3. API 키 검증
python -c "from core.config import APIConfig; print(APIConfig.validate())"
```

### 8.2 main.py 실행 옵션

```bash
# 기본 실행 (전체 파이프라인, ~40초)
python main.py

# 빠른 분석 (~16초, Phase 2.3-2.10 스킵)
python main.py --quick

# AI 리포트 포함
python main.py --report

# 실시간 스트리밍
python main.py --realtime --duration 60

# 서버/크론용 (최소 출력)
python main.py --cron --output /path/to/outputs
```

### 8.3 Economic Insight Agent 실행

```bash
# 템플릿 기반 분석
python -m agent.cli --question "질문"

# EIMAS 결과 활용
python -m agent.cli --with-eimas --question "질문"

# 평가 실행
python -m agent.evals.runner --verbose
```

### 8.4 대시보드 실행 (3개 터미널)

```bash
# Terminal 1: FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2: EIMAS 분석 (최소 1회)
python main.py --quick

# Terminal 3: Frontend
cd frontend && npm run dev

# → 브라우저: http://localhost:3000
```

---

## 9. 디렉토리 구조

```
eimas/
├── main.py                 # 메인 파이프라인 (~1100줄)
├── CLAUDE.md               # Claude Code용 요약
├── WORKFLOW.md             # 이 파일 (상세 문서)
├── README.md               # 간략 소개
│
├── agents/                 # AI 에이전트 (토론 시스템)
│   ├── base_agent.py       # BaseAgent 추상 클래스
│   ├── orchestrator.py     # MetaOrchestrator
│   ├── analysis_agent.py   # CriticalPath 분석
│   ├── forecast_agent.py   # LASSO 예측
│   ├── research_agent.py   # Perplexity 연동
│   └── ...
│
├── agent/                  # Economic Insight Agent (인과 분석)
│   ├── cli.py              # CLI 인터페이스
│   ├── core/
│   │   ├── adapters.py     # EIMAS → Schema 변환
│   │   └── orchestrator.py # 6단계 추론 파이프라인
│   ├── schemas/
│   │   └── insight_schema.py  # Pydantic 스키마
│   ├── evals/              # 10개 시나리오 평가
│   └── tests/              # 단위 테스트
│
├── core/                   # 핵심 프레임워크
│   ├── config.py           # API 설정
│   ├── schemas.py          # 데이터 스키마
│   └── debate.py           # 토론 프로토콜
│
├── lib/                    # 기능 모듈 (80개+)
│   ├── fred_collector.py   # FRED 데이터
│   ├── data_collector.py   # 시장 데이터
│   ├── regime_detector.py  # 레짐 탐지
│   ├── regime_analyzer.py  # GMM & Entropy
│   ├── critical_path.py    # 리스크 점수
│   ├── microstructure.py   # 미세구조 분석
│   ├── bubble_detector.py  # 버블 탐지
│   ├── shock_propagation_graph.py  # 충격 전파
│   ├── graph_clustered_portfolio.py  # HRP 포트폴리오
│   ├── genius_act_macro.py # 스테이블코인-유동성
│   ├── whitening_engine.py # 경제학적 해석
│   ├── autonomous_agent.py # 팩트체킹
│   └── ...
│
├── api/                    # FastAPI 서버
│   ├── main.py
│   └── routes/
│       ├── analysis.py     # /latest 엔드포인트
│       └── ...
│
├── frontend/               # Next.js 대시보드
│   ├── app/page.tsx
│   ├── components/
│   └── lib/
│
├── pipeline/               # 파이프라인 모듈
│   ├── schemas.py          # EIMASResult, DebateResult
│   ├── debate.py           # Dual Mode 토론
│   └── full_pipeline.py
│
├── outputs/                # 결과 저장
│   └── integrated_YYYYMMDD_HHMMSS.json
│
└── data/                   # 데이터베이스
    └── events.db
```

---

## 10. 경제학적 방법론

### 10.1 사용된 학술 방법론

| 방법론 | 출처 | 사용처 |
|--------|------|--------|
| LASSO (L1 정규화) | Tibshirani (1996) | ForecastAgent |
| Granger Causality | Granger (1969) | ShockPropagation |
| GMM 3-State | Hamilton (1989) | RegimeAnalyzer |
| Shannon Entropy | Shannon (1948) | 불확실성 측정 |
| Bekaert VIX 분해 | Bekaert et al. (2013) | CriticalPath |
| Greenwood-Shleifer 버블 | Greenwood-Shleifer (2019) | BubbleDetector |
| Amihud Lambda | Amihud (2002) | Microstructure |
| VPIN | Easley et al. (2012) | 정보 비대칭 |
| MST | Mantegna (1999) | 상관관계 네트워크 |
| HRP | De Prado (2016) | 포트폴리오 최적화 |

### 10.2 핵심 수식

```
# 순유동성
Net_Liquidity = Fed_Balance_Sheet - RRP - TGA

# 확장 유동성 (Genius Act)
M = B + S·B*

# 리스크 점수 (v2.1.1)
Final_Risk = Base + Micro_Adj(±10) + Bubble_Adj(0~15)

# MST 거리 (Mantegna)
d(i,j) = sqrt(2 * (1 - ρ_ij))

# 부호 합성
Net_Effect = Π(edge_signs)  # 홀수 음수면 음수
```

### 10.3 인과 그래프 레이어

```
Layer 1: POLICY        Fed_Policy, Fed_Funds_Rate
           ↓
Layer 2: LIQUIDITY     Net_Liquidity, RRP, TGA, Stablecoin
           ↓
Layer 3: RISK_PREMIUM  VIX, Credit_Spread
           ↓
Layer 4: ASSET_PRICE   SPY, TLT, BTC
```

---

## 부록: 자주 묻는 질문

### Q: agents/와 agent/의 차이는?

- `agents/`: 기존 토론 시스템 (MetaOrchestrator, Claude/GPT-4/Gemini 토론)
- `agent/`: Economic Insight Agent (인과 그래프, JSON-first, 독립 실행 가능)

### Q: --quick과 기본 실행의 차이는?

- `--quick`: Phase 2.3-2.10 스킵, ~16초
- 기본: 전체 파이프라인, ~40초

### Q: EIMAS 결과를 Economic Insight Agent에서 사용하려면?

```bash
python -m agent.cli --with-eimas --question "질문"
```
→ `outputs/integrated_*.json`에서 최신 결과를 자동으로 읽어 변환합니다.

---

*마지막 업데이트: 2026-01-28*
