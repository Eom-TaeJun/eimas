# EIMAS - Economic Intelligence Multi-Agent System

> Claude Code가 프로젝트를 빠르게 파악하기 위한 요약 문서입니다.
> main.py를 매번 읽지 않아도 됩니다.

---

## 프로젝트 개요

**EIMAS**는 거시경제 데이터와 시장 데이터를 수집하여 **AI 멀티에이전트 토론**을 통해 시장 전망과 투자 권고를 생성하는 시스템입니다.

### 무엇을 하나요?

1. **데이터 수집**: FRED(연준 데이터), yfinance(시장 데이터), 크립토/RWA 자산
2. **레짐 탐지**: 현재 시장이 Bull/Bear/Neutral 중 어디인지 판단
3. **리스크 분석**: 유동성, 버블, 시장 미세구조 등 다차원 리스크 평가
4. **AI 토론**: Claude 기반 에이전트들이 서로 다른 관점에서 토론 후 합의
5. **권고 생성**: 최종 투자 방향(BULLISH/BEARISH/NEUTRAL)과 신뢰도 제공

### 누가 사용하나요?

- 거시경제 기반 투자 의사결정이 필요한 개인/기관
- 정량적 시장 분석을 자동화하려는 퀀트 리서처
- AI 에이전트 시스템을 연구하는 개발자

---

## 경제학적 방법론

| 방법론 | 사용처 | 설명 |
|--------|--------|------|
| **LASSO (L1 정규화)** | ForecastAgent | 변수 선택 (sparsity), 과적합 방지 |
| **Granger Causality** | LiquidityAnalyzer | 시계열 간 인과관계 테스트 |
| **GMM 3-State** | RegimeAnalyzer | Bull/Neutral/Bear 상태 분류 |
| **Shannon Entropy** | RegimeAnalyzer | 시장 불확실성 정량화 |
| **Bekaert VIX 분해** | CriticalPath | VIX = Uncertainty + Risk Appetite |
| **Greenwood-Shleifer** | BubbleDetector | 2년 100% run-up → 버블 위험 |
| **Amihud Lambda** | Microstructure | 비유동성 측정 (가격 충격/거래량) |
| **VPIN** | Microstructure | 정보 비대칭/독성 주문 흐름 |
| **MST (Mantegna 1999)** | GraphClusteredPortfolio | 상관관계 기반 최소신장트리 |
| **HRP (De Prado)** | GraphClusteredPortfolio | 계층적 리스크 패리티 포트폴리오 |

### 핵심 수식

```
# 순 유동성 (Fed 유동성)
Net Liquidity = Fed Balance Sheet - RRP - TGA

# Genius Act 확장 유동성
M = B + S·B*  (순유동성 + 스테이블코인 기여도)

# 리스크 점수 (v2.1.1)
Final Risk = Base(CriticalPath) + Micro Adj(±10) + Bubble Adj(+0~15)

# MST 거리 공식
d(i,j) = sqrt(2 * (1 - ρ_ij))
```

---

## 설치 및 환경 설정

### 1. 의존성 설치

```bash
cd eimas
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# .env 파일 생성 또는 환경변수 설정
export ANTHROPIC_API_KEY="sk-ant-..."      # Claude (필수)
export FRED_API_KEY="your-fred-key"        # FRED 데이터 (필수)
export PERPLEXITY_API_KEY="pplx-..."       # Perplexity (선택)
export OPENAI_API_KEY="sk-..."             # OpenAI (선택)
export GOOGLE_API_KEY="..."                # Gemini (선택)
```

### 3. API 키 검증

```bash
python -c "from core.config import APIConfig; print(APIConfig.validate())"
# 예상 출력: {'anthropic': True, 'perplexity': True, 'openai': True, ...}
```

---

## Quick Start (처음 사용자용)

### Step 1: 빠른 분석 실행

```bash
python main.py --quick
```

예상 출력:
```
[1.1] Collecting FRED data...
      ✓ RRP: $5.2B, TGA: $721.5B, Net Liquidity: $5799.3B
[1.2] Collecting market data...
      ✓ Collected 24 tickers
...
[2.4.1] Microstructure risk enhancement...
      ✓ Avg Liquidity Score: 65.2/100
[2.4.2] Bubble risk overlay...
      ✓ Overall Bubble Status: WATCH
...
============================================================
                    FINAL SUMMARY
============================================================
📊 DATA: FRED RRP=$5B, Net Liq=$5799B, Market 24 tickers
📈 REGIME: Bull (Low Vol), Risk 45.2/100
🤖 DEBATE: FULL=BULLISH, REF=BULLISH (Agree ✓)
🎯 FINAL: BULLISH, Confidence 65%, Risk MEDIUM
```

### Step 2: 결과 확인

```bash
# JSON 결과
cat outputs/integrated_YYYYMMDD_HHMMSS.json

# 마크다운 리포트
cat outputs/integrated_YYYYMMDD_HHMMSS.md
```

### Step 3: AI 리포트 생성 (선택)

```bash
python main.py --report
# Claude/Perplexity가 분석 결과를 자연어로 해석
```

### Step 4: 실시간 대시보드 (NEW - 2026-01-11)

**3개 터미널로 실시간 UI 실행:**

```bash
# 터미널 1: FastAPI 서버
uvicorn api.main:app --reload --port 8000

# 터미널 2: EIMAS 분석 (최소 1회)
python main.py --quick

# 터미널 3: 프론트엔드
cd frontend
npm install  # 최초 1회
npm run dev
```

브라우저: **http://localhost:3000**

**기능:**
- 5초 자동 폴링으로 최신 분석 결과 실시간 업데이트
- 시장 레짐, 리스크 점수, AI 합의 결과 시각화
- v2.1.1 Market Quality & Bubble Risk 메트릭 포함
- GitHub 스타일 다크 테마 UI

**상세 가이드:** `DASHBOARD_QUICKSTART.md` 참조

---

## Quick Reference

```bash
# 실행 명령어 (v2.1.0 Real-World Agent Edition)
python main.py                    # 전체 파이프라인 (~40초)
python main.py --quick            # 빠른 분석 (~16초, Phase 2.3-2.10 스킵)
python main.py --report           # AI 리포트 포함
python main.py --realtime         # 실시간 스트리밍 포함
python main.py --realtime --duration 60  # 60초 스트리밍

# CLI 자동화 옵션 (2026-01-08 추가)
python main.py --mode full        # 전체 분석 (기본값)
python main.py --mode quick       # 빠른 분석 (--quick과 동일)
python main.py --mode report      # AI 리포트 포함

python main.py --cron             # 크론/서버용 (최소 출력)
python main.py --output /path     # 출력 디렉토리 지정
python main.py --version          # v2.1.0 (Real-World Agent Edition)
```

## main.py 파이프라인 구조

```
Phase 1: DATA COLLECTION
|-- [1.1] FREDCollector          -> RRP, TGA, Net Liquidity, Fed Funds
|-- [1.2] DataManager            -> 시장 데이터 (SPY, QQQ, TLT, GLD 등 24개)
|-- [1.3] Crypto & RWA data      -> BTC-USD, ETH-USD + ONDO-USD, PAXG-USD, COIN
|-- [1.4] MarketIndicatorsCollector -> VIX, Fear & Greed
+-- 출력: fred_summary, market_data (24 tickers + 2 crypto + 3 RWA)

Phase 2: ANALYSIS
|-- [2.1] RegimeDetector           -> 시장 레짐 (BULL/BEAR/NEUTRAL)
|-- [2.1.1] GMMRegimeAnalyzer      -> GMM 3-state + Shannon Entropy
|-- [2.2] QuantitativeEventDetector -> 이벤트 탐지
|-- [2.3] LiquidityMarketAnalyzer  -> Granger Causality
|-- [2.4] CriticalPathAggregator   -> 리스크 스코어 (Base)
|-- [2.4.1] DailyMicrostructureAnalyzer -> 시장 미세구조 품질 (NEW v2.1.1)
|-- [2.4.2] BubbleDetector         -> 버블 리스크 오버레이 (NEW v2.1.1)
|-- [2.5] ETFFlowAnalyzer          -> 섹터 로테이션
|-- [2.6] GeniusActMacroStrategy   -> 스테이블코인-유동성 분석
|-- [2.7] CustomETFBuilder         -> 테마 ETF 분석
|-- [2.8] ShockPropagationGraph    -> 충격 전파 그래프
|-- [2.9] GraphClusteredPortfolio  -> GC-HRP 포트폴리오 최적화
|-- [2.10] IntegratedStrategy      -> 통합 전략 (Portfolio + Causality)
+-- 출력: regime, events, risk_score (adjusted), market_quality, bubble_risk

Phase 3: MULTI-AGENT DEBATE
|-- [3.1] MetaOrchestrator (FULL mode, 365일)
|-- [3.2] MetaOrchestrator (REFERENCE mode, 90일)
|-- [3.3] DualModeAnalyzer       -> 모드 비교
+-- 출력: final_recommendation, confidence

Phase 4: REAL-TIME (--realtime 옵션)
|-- [4.1] BinanceStreamer        -> WebSocket
|-- MicrostructureAnalyzer       -> OFI, VPIN
+-- 출력: realtime_signals

Phase 5: DATABASE STORAGE
|-- [5.1] EventDatabase          -> data/events.db
|-- [5.2] SignalDatabase         -> outputs/realtime_signals.db
|-- [5.3] Results 저장           -> outputs/integrated_YYYYMMDD_HHMMSS.json
                                 -> outputs/integrated_YYYYMMDD_HHMMSS.md

Phase 6: AI REPORT (--report 옵션)
|-- [6.1] AIReportGenerator      -> Claude/Perplexity 기반
|-- [6.2] Report Save
+-- 출력: outputs/ai_report_YYYYMMDD.json

Phase 7: WHITENING & FACT CHECK (--report 옵션)
|-- [7.1] WhiteningEngine        -> 결과 경제학적 해석
|-- [7.2] AutonomousFactChecker  -> AI 출력 팩트체킹
+-- 출력: whitening_summary, fact_check_grade
```

## 신규 모듈 통합 상태 (16개)

| # | 모듈 | 통합 위치 | 상태 | 설명 |
|---|------|----------|------|------|
| 1 | `genius_act_macro.py` | Phase 2.6 | ✅ | 스테이블코인-유동성 + 크립토 리스크 |
| 2 | `custom_etf_builder.py` | Phase 2.7 | ✅ | 테마 ETF 분석 |
| 3 | `shock_propagation_graph.py` | Phase 2.8 | ✅ | 충격 전파 인과관계 |
| 4 | `graph_clustered_portfolio.py` | Phase 2.9 | ✅ | GC-HRP + MST v2 (Eigenvector 제거) |
| 5 | `integrated_strategy.py` | Phase 2.10 | ✅ | 통합 전략 엔진 |
| 6 | `whitening_engine.py` | Phase 7.1 | ✅ | 경제학적 해석 |
| 7 | `autonomous_agent.py` | Phase 7.2 | ✅ | AI 팩트체킹 |
| 8 | `data_loader.py` | Phase 1.3 | ✅ | RWA 자산 확장 (2026-01-08) |
| 9 | `regime_analyzer.py` | Phase 2.1.1 | ✅ | GMM & Entropy 레짐 (2026-01-08) |
| 10 | `causality_graph.py` | Phase 2.8 | ✅ | 인과관계 Narrative (2026-01-08) |
| 11 | `microstructure.py` | Phase 2.4.1 | ✅ | **Risk Enhancement Layer** (2026-01-09) |
| 12 | `bubble_detector.py` | Phase 2.4.2 | ✅ | **Bubble Risk Overlay** (2026-01-09) |
| 13 | `validate_methodology.py` | scripts/ | ✅ | API 방법론 검증 (2026-01-09) |
| 14 | `validate_integration_design.py` | scripts/ | ✅ | 아키텍처 통합 설계 검증 (2026-01-09) |
| 15 | MarketQualityMetrics | main.py | ✅ | 시장 미세구조 메트릭 (2026-01-09) |
| 16 | **Economic Insight Agent** | `agent/` | ✅ | **인과적 분석 에이전트 (2026-01-28)** |

## 핵심 데이터 클래스

```python
@dataclass
class EIMASResult:
    timestamp: str

    # Phase 1: 데이터 수집
    fred_summary: Dict           # RRP, TGA, Net Liquidity
    market_data_count: int
    crypto_data_count: int

    # Phase 2: 분석
    regime: Dict                 # regime, trend, volatility
    events_detected: List[Dict]
    liquidity_signal: str
    risk_score: float            # 최종 조정된 리스크 점수
    genius_act_regime: str       # expansion/contraction/neutral
    genius_act_signals: List[Dict]
    theme_etf_analysis: Dict
    shock_propagation: Dict
    portfolio_weights: Dict[str, float]      # GC-HRP 결과
    integrated_signals: List[Dict]           # 통합 전략 시그널

    # Phase 2.4.1-2.4.2: Risk Enhancement (v2.1.1 NEW)
    market_quality: MarketQualityMetrics     # 시장 미세구조 품질
    bubble_risk: BubbleRiskMetrics           # 버블 리스크 메트릭
    base_risk_score: float                   # CriticalPath 기본 점수
    microstructure_adjustment: float         # ±10 범위 조정
    bubble_risk_adjustment: float            # 버블 리스크 가산

    # Phase 3: 토론
    full_mode_position: str      # BULLISH/BEARISH/NEUTRAL
    reference_mode_position: str
    modes_agree: bool
    dissent_records: List[Dict]
    has_strong_dissent: bool

    # 최종 결과
    final_recommendation: str    # HOLD/BUY/SELL/BULLISH/BEARISH
    confidence: float
    risk_level: str              # LOW/MEDIUM/HIGH
    warnings: List[str]

    # Phase 4 (--realtime 옵션)
    realtime_signals: List[Dict]

    # Phase 7 (--report 옵션)
    whitening_summary: str
    fact_check_grade: str

# v2.1.1 NEW: 시장 미세구조 품질 메트릭
@dataclass
class MarketQualityMetrics:
    avg_liquidity_score: float       # 0-100 스케일
    liquidity_scores: Dict[str, float]
    high_toxicity_tickers: List[str]  # VPIN > 50%
    illiquid_tickers: List[str]       # 유동성 < 30
    data_quality: str                 # COMPLETE/PARTIAL/DEGRADED

# v2.1.1 NEW: 버블 리스크 메트릭 (Greenwood-Shleifer)
@dataclass
class BubbleRiskMetrics:
    overall_status: str              # NONE/WATCH/WARNING/DANGER
    risk_tickers: List[Dict]         # Top 5 위험 종목
    highest_risk_ticker: str
    highest_risk_score: float
    methodology_notes: str
```

## 디렉토리 구조

```
eimas/
|-- main.py              # 메인 파이프라인 (~1088줄)
|-- CLAUDE.md            # 이 파일 (요약)
|-- ARCHITECTURE.md      # 상세 아키텍처
|-- agents/              # 에이전트 모듈 (14개 파일)
|   |-- __init__.py
|   |-- base_agent.py         # BaseAgent 추상 클래스
|   |-- orchestrator.py       # MetaOrchestrator
|   |-- analysis_agent.py     # CriticalPath 분석
|   |-- forecast_agent.py     # LASSO 예측
|   |-- research_agent.py     # Perplexity 연동
|   |-- strategy_agent.py     # 전략 권고
|   |-- visualization_agent.py # 시각화
|   |-- top_down_orchestrator.py  # Top-Down 분석
|   |-- regime_change.py      # 레짐 변화 감지
|   |-- methodology_debate.py # 방법론 토론
|   +-- interpretation_debate.py # 해석 토론
|-- agent/              # Economic Insight Agent (NEW 2026-01-28)
|   |-- __init__.py          # Main exports
|   |-- cli.py               # CLI interface
|   |-- README.md            # Agent 문서
|   |-- core/
|   |   |-- adapters.py      # EIMAS → Schema 변환
|   |   +-- orchestrator.py  # 6단계 추론 파이프라인
|   |-- schemas/
|   |   +-- insight_schema.py  # Pydantic JSON 스키마
|   |-- examples/            # JSON 요청 예제
|   |-- evals/               # 10개 시나리오 평가
|   +-- tests/               # 단위 테스트
|-- core/                # 핵심 프레임워크
|   |-- __init__.py
|   |-- schemas.py       # 데이터 스키마
|   |-- config.py        # API 설정
|   |-- debate.py        # 토론 프로토콜
|   |-- database.py      # DB 설정
|   |-- signal_action.py # 시그널 액션
|   |-- logging_config.py
|   +-- health_check.py
|-- lib/                 # 기능 모듈 (80개 파일)
|   |-- fred_collector.py
|   |-- data_collector.py
|   |-- data_loader.py           # RWA 자산 (NEW)
|   |-- regime_detector.py
|   |-- regime_analyzer.py       # GMM & Entropy (NEW)
|   |-- event_framework.py
|   |-- liquidity_analysis.py
|   |-- critical_path.py
|   |-- etf_flow_analyzer.py
|   |-- graph_clustered_portfolio.py  # GC-HRP
|   |-- integrated_strategy.py        # 통합 전략
|   |-- shock_propagation_graph.py
|   |-- causality_graph.py       # 인과관계 Narrative (NEW)
|   |-- genius_act_macro.py
|   |-- whitening_engine.py
|   |-- autonomous_agent.py
|   |-- ai_report_generator.py
|   |-- binance_stream.py
|   |-- microstructure.py
|   |-- realtime_pipeline.py
|   |-- dual_mode_analyzer.py
|   +-- ... (기타 77개)
|-- api/                 # FastAPI 서버
|   |-- server.py
|   |-- main.py
|   |-- routes/
|   |   |-- health.py
|   |   |-- analysis.py      # /latest 엔드포인트 (NEW 2026-01-11)
|   |   |-- regime.py
|   |   |-- debate.py
|   |   +-- report.py
|   +-- models/
|       |-- requests.py
|       +-- responses.py
|-- frontend/            # 실시간 대시보드 (NEW 2026-01-11)
|   |-- app/             # Next.js 16 App Router
|   |   |-- page.tsx     # 메인 대시보드 페이지
|   |   |-- layout.tsx
|   |   +-- globals.css
|   |-- components/      # React 컴포넌트
|   |   |-- MetricsGrid.tsx   # 메트릭 카드 (5초 폴링)
|   |   |-- SignalsTable.tsx  # 시그널 테이블
|   |   |-- Navbar.tsx
|   |   +-- ui/          # shadcn/ui 컴포넌트
|   |-- lib/             # 유틸리티
|   |   |-- api.ts       # fetchLatestAnalysis()
|   |   +-- types.ts     # EIMASAnalysis 인터페이스
|   |-- package.json     # Next.js 16, React 19, SWR
|   |-- tsconfig.json
|   +-- README.md        # 프론트엔드 상세 가이드
|-- cli/                 # CLI 인터페이스
|   +-- eimas.py
|-- pipeline/            # 파이프라인 모듈
|   +-- full_pipeline.py
|-- scripts/             # 스크립트
|   |-- daily_collector.py
|   |-- daily_analysis.py
|   |-- run_backtest.py
|   +-- scheduler.py
|-- tests/               # 테스트
|   |-- test_integration.py
|   |-- test_lasso_forecast.py
|   |-- test_api_connection.py
|   |-- test_signal_action.py
|   +-- test_lib.py
|-- data/                # 데이터베이스
|   |-- cache.py
|   +-- pipeline.py
|-- outputs/             # 결과 JSON
+-- configs/             # YAML 설정
```

## Phase별 실행 조건

| Phase | --quick | 기본 | --report | --realtime |
|-------|---------|------|----------|------------|
| 1. Data Collection (RWA 포함) | O | O | O | O |
| 2.1 RegimeDetector | O | O | O | O |
| 2.1.1 GMM & Entropy | X | O | O | O |
| 2.2 EventDetector | O | O | O | O |
| 2.3-2.10 Advanced | X | O | O | O |
| 3. Debate | O | O | O | O |
| 4. Realtime | X | X | X | O |
| 5. DB Storage | O | O | O | O |
| 6. AI Report | X | X | O | X |
| 7. Whitening | X | X | O | X |

## 새 모듈 추가 시 체크리스트

1. `lib/` 에 모듈 생성
2. `if __name__ == "__main__"` 테스트 코드 포함
3. **main.py에 import 추가** (line 45-86)
4. **적절한 Phase에 호출 코드 추가**
5. **EIMASResult에 필요한 필드 추가** (line 100-146)
6. **Summary 출력에 결과 추가** (line 958-1014)
7. 이 파일(CLAUDE.md) 업데이트

## API 키 (환경변수)

- `ANTHROPIC_API_KEY` - Claude
- `PERPLEXITY_API_KEY` - Perplexity
- `OPENAI_API_KEY` - OpenAI
- `GOOGLE_API_KEY` - Gemini
- `FRED_API_KEY` - FRED

## API 서버

```bash
# FastAPI 서버 실행
uvicorn api.main:app --reload --port 8000

# 엔드포인트
GET  /health           # 헬스 체크
POST /analysis/run     # 분석 실행
GET  /regime/current   # 현재 레짐
POST /debate/run       # 토론 실행
POST /report/generate  # 리포트 생성
GET  /latest           # 최신 integrated JSON 반환 (NEW 2026-01-11, 대시보드용)
```

## CLI 사용법

```bash
# CLI 도움말
python -m cli.eimas --help

# 분석 실행
python -m cli.eimas analyze --quick
python -m cli.eimas analyze --report
```

## 최근 업데이트 (Changelog)

### v2.1.1 (2026-01-09) - Risk Analytics Enhancement

**Task 1: 시장 미세구조 모듈 강화** (2026-01-08)
- `lib/microstructure.py`에 AMFL Chapter 19 기반 지표 추가
  - Amihud Lambda (비유동성 측정)
  - Roll Spread (Bid-Ask 추정)
  - VPIN Approximation (일별 데이터용)
- `lib/bubble_detector.py` 신규 생성 (570+ lines)
  - "Bubbles for Fama" 논문 기반 버블 탐지
  - Run-up Check (2년 누적 수익률 > 100%)
  - Volatility Spike (Z-score > 2)
  - Share Issuance 증가 확인
  - 테스트: NVDA 1094.6% run-up → WARNING level

**Task 2: 크립토 리스크 평가** (2026-01-09)
- `lib/genius_act_macro.py`에 `CryptoRiskEvaluator` 추가 (320+ lines)
  - 스테이블코인 담보 유형 분류:
    - TREASURY_CASH (USDC): 15점 - 국채/현금 담보
    - MIXED_RESERVE (USDT): 35점 - 혼합 준비금
    - CRYPTO_BACKED (DAI): 40점 - 암호화폐 담보
    - DERIVATIVE_HEDGE (USDe): 50점 - 파생상품 헤지
    - ALGORITHMIC: 80점 - 알고리즘
  - 이자 지급 스테이블코인 +15점 페널티 (SEC 증권 분류 리스크)

**Task 3: MST 시스템 리스크 분석** (2026-01-09)
- `lib/graph_clustered_portfolio.py`에 MST 분석 추가 (150+ lines)
  - 거리 공식: `d = sqrt(2 * (1 - rho))` (Mantegna 1999)
  - MST 기반 중심성 분석 (v2 - API 검증 후 조정):
    - Betweenness Centrality (45%) - 충격 전파 핵심
    - Degree Centrality (35%) - 허브 식별
    - Closeness Centrality (20%) - 정보 흐름 속도
    - ~~Eigenvector Centrality~~ - 트리 구조에서 비효율적 → **제거됨**
  - `_adaptive_node_selection()`: sqrt(N) 기반 자동 노드 선택
  - `rolling_mst_analysis()`: 시계열 시스템 리스크 추적

**Task 4: Risk Enhancement Layer 통합** (2026-01-09)
- API 아키텍처 설계 검증: **Option C (Risk Enhancement Layer)** 선택
- `main.py`에 Phase 2.4.1, 2.4.2 통합 (140+ lines)
  - Phase 2.4.1: `DailyMicrostructureAnalyzer` - 시장 미세구조 품질
    - `MarketQualityMetrics` 데이터클래스 추가
    - 유동성 점수 기반 리스크 조정 (±10 범위)
  - Phase 2.4.2: `BubbleDetector` - 버블 리스크 오버레이
    - `BubbleRiskMetrics` 데이터클래스 추가
    - 버블 레벨별 리스크 가산 (+5/+10/+15)
  - 최종 리스크 점수 = Base + Microstructure Adj. + Bubble Adj.
- `to_markdown()` 리포트 업데이트:
  - Risk Score Breakdown 테이블 추가
  - Market Quality & Bubble Risk 섹션 추가 (이모지 지표)

---

## API 방법론 검증 (2026-01-09)

### 검증 결과 요약

| 모듈 | 평가 | Claude | Perplexity |
|------|------|--------|------------|
| Stablecoin Risk | PARTIALLY_CORRECT | 기본 리스크 순서 적절, 이자 페널티 세분화 필요 | 은행급 리스크 프레임워크 권장 |
| MST Systemic Risk | PARTIALLY_CORRECT | 거리 공식 학술적으로 정확, 중심성 가중치 조정 권장 | Mantegna (1999) 인용 확인 |

### Stablecoin Risk 검증 상세

**Claude 평가:**
- ✅ 기본 리스크 순서 (USDC < USDT < DAI < USDe) 합리적
- ⚠️ +15 이자 페널티는 과도하게 단순화됨
- ❌ 누락된 리스크 요소: 유동성, 거버넌스, 기술적 리스크

**Perplexity 리서치:**
- USDC: 투자등급급 취급, 규제 친화적
- USDe: 합성/파생상품 기반, 기관 투자자 경계
- 권장: 다차원 스코어링 (신용, 유동성, 규제, 기술)

**개선 권고사항:** ✅ 구현 완료 (2026-01-09)
```python
# 구현된 다차원 리스크 평가 (v2)
class MultiDimensionalRiskScore:
    WEIGHTS = {
        'credit': 0.30,      # 신용/담보 리스크
        'liquidity': 0.25,   # 유동성 리스크
        'regulatory': 0.25,  # 규제 리스크 (이자 차등 페널티)
        'technical': 0.20    # 기술/스마트컨트랙트 리스크
    }

# 결과 예시 (v2):
# USDC:  10.2점 (A) - Genius Act 준수
# USDe:  50.7점 (D) - 이자 지급 + 파생상품 → 높은 규제 리스크
```

### MST Systemic Risk 검증 상세

**Claude 평가:**
- ✅ 거리 공식 `d = sqrt(2*(1-rho))` 학술적으로 정확
- ⚠️ Eigenvector centrality는 트리 구조에서 비효율적 (제거 권장)
- ❌ 누락된 요소: 동적 분석, 섹터 클러스터링, 레짐별 MST

**Perplexity 리서치:**
- Mantegna (1999) 거리 공식 = 정석 (canonical)
- 최근 트렌드: PMFG, TMFG 등 보완 필터 사용
- 동적 분석 (rolling window) 필수

**개선 권고사항:** ✅ 구현 완료 (2026-01-09)
```python
# 구현된 중심성 가중치 (v2)
CENTRALITY_WEIGHTS = {
    'betweenness': 0.45,  # 충격 전파 핵심
    'degree': 0.35,       # 허브 식별
    'closeness': 0.20,    # 정보 흐름
    # eigenvector: 제거됨 (트리 구조에서 비효율적)
}

# 구현된 동적 분석
def rolling_mst_analysis(returns, window=252, step=21):
    """시간에 따른 시스템 리스크 노드 변화 추적"""
    # 구현 완료 - 노드 지속성 분석 포함

# 적응형 노드 선택 구현
def _adaptive_node_selection(n_assets):
    # sqrt(N) 기반 휴리스틱
    # 10자산: 3개, 100자산: 5개, 500자산: 8개
```

---

## 아키텍처 통합 설계 검증 (2026-01-09)

### 설계 옵션 비교

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A: Sequential** | Phase 2.2 → 2.2.1 (Micro) → 2.2.2 (Bubble) | 단순한 의존성 | 실행 시간 증가 |
| **B: Parallel** | Phase 2.2a, 2.2b, 2.2c 병렬 실행 | 빠른 실행 | 복잡한 동기화 |
| **C: Risk Layer** | Phase 2.4.1, 2.4.2로 CriticalPath 후 실행 | 리스크 통합 용이 | 추가 Phase |

### API 검증 결과

**선택: Option C (Risk Enhancement Layer)** ✅

- **Claude 권고**: "CriticalPathAggregator 이후에 위치하면 기본 리스크 점수 위에 추가 분석 레이어를 쌓을 수 있어 리스크 통합이 용이"
- **Perplexity 리서치**: "헤지펀드 시스템에서 일반적으로 사용하는 패턴. 기본 리스크 → 시장 미세구조 조정 → 버블/테일 리스크 오버레이"

### 구현된 리스크 점수 공식

```python
# Risk Enhancement Layer (Option C)
final_risk = base_risk + microstructure_adj + bubble_adj

# Base Risk: CriticalPathAggregator (0-100)
# Microstructure Adj: (50 - avg_liquidity) / 5, clamped ±10
# Bubble Adj: NONE=0, WATCH=+5, WARNING=+10, DANGER=+15

# 예시:
# Base=45.0, Micro=-4.0 (유동성 우수), Bubble=+10 (WARNING)
# Final = 45.0 - 4.0 + 10.0 = 51.0
```

---

### v2.1.0 (2026-01-08) - Real-World Agent Edition

**Task 1: RWA 자산 확장**
- `lib/data_loader.py` 신규 생성 (350+ lines)
- 토큰화 자산 지원: ONDO-USD (US Treasury), PAXG-USD (Gold), COIN (Exchange)
- 경제학적 근거: "Asset이 infinite... 모든 거래 가능한 걸 토큰화"
- 테스트 결과: PAXG-USD $4438, ONDO-USD $0.40, COIN $245

**Task 2: GMM & Entropy 레짐 분석**
- `lib/regime_analyzer.py` 신규 생성 (450+ lines)
- GMM 3-state 분류: Bull / Neutral / Bear
- Shannon Entropy로 불확실성 측정
- Entropy Level: Very Low (확신) ~ Very High (불확실)
- 경제학적 근거: "GMM을 써야 함", "엔트로피로 불확실성 측정"

**Task 3: CLI 자동화**
- `--mode` (full/quick/report)
- `--cron` (서버 배포용, 최소 출력)
- `--output` (출력 디렉토리 지정)
- `--version` (v2.1.0 표시)
- 경제학적 근거: "목표는 터미널을 통해서 작업할 수 있을 정도"

**Causality Narrative (2026-01-08)**
- `lib/causality_graph.py`에 `generate_report_narrative()` 추가
- Critical Path + Shock Propagation → 자연어 변환
- MD 리포트에 인과관계 분석 결과 포함

### 실행 결과 예시 (2026-01-08)
```
📊 DATA: FRED RRP=$5B, Net Liq=$5799B, Market 24 tickers
📈 REGIME: Bull (Low Vol), Risk 10.6/100
🤖 DEBATE: FULL=BULLISH, REF=BULLISH (Agree ✓)
🎯 FINAL: BULLISH, Confidence 65%, Risk MEDIUM
⏱️ TIME: --quick ~16초
```

---

### v2.1.2 (2026-01-11) - Real-Time Dashboard

**Task 5: 실시간 대시보드 UI 구현** (2026-01-11)
- **프론트엔드**: Next.js 16 기반 실시간 대시보드
  - `frontend/` 디렉토리: v0 MCP로 생성된 React 앱 활용
  - `components/MetricsGrid.tsx`: 5초 자동 폴링 (SWR 사용)
  - `lib/api.ts`: `fetchLatestAnalysis()` API 클라이언트
  - `lib/types.ts`: `EIMASAnalysis` TypeScript 인터페이스
  - 다크 테마 (GitHub 스타일), Tailwind CSS 4
  - Radix UI 컴포넌트, Lucide React 아이콘

- **백엔드**: FastAPI 엔드포인트 추가
  - `api/routes/analysis.py`에 `GET /latest` 추가
  - outputs 디렉토리에서 최신 `integrated_*.json` 자동 선택
  - 파일 메타데이터 포함 (수정 시간, 파일명)

- **화면 구성**:
  - Main Status Banner: 최종 권고 (BULLISH/BEARISH/NEUTRAL) + 신뢰도 + 리스크
  - Metrics Grid (4 cards):
    1. Market Regime (Bull/Bear/Neutral + 아이콘)
    2. AI Consensus (Full Mode vs Reference Mode 비교)
    3. Data Collection (Market tickers + Crypto assets)
    4. Market Quality (v2.1.1 메트릭)
  - Warnings 섹션 (있을 경우)

- **실행 방법**:
  ```bash
  # 터미널 1: FastAPI
  uvicorn api.main:app --reload --port 8000

  # 터미널 2: EIMAS 분석
  python main.py --quick

  # 터미널 3: 프론트엔드
  cd frontend && npm run dev
  ```
  브라우저: http://localhost:3000

- **문서화**:
  - `frontend/README.md`: 프론트엔드 상세 가이드
  - `DASHBOARD_QUICKSTART.md`: 3분 빠른 시작 가이드
  - `CLAUDE.md`: 디렉토리 구조 + API 엔드포인트 업데이트

---

## 현재 상태 (2026-01-11 18:00 KST)

### ✅ 작동 중
- **FastAPI 서버** (포트 8000): `/latest` 엔드포인트 정상 작동
- **Next.js 프론트엔드** (포트 3002): 기본 대시보드 렌더링
- **데이터 수집**: integrated_*.json 파일 생성 중
- **5초 자동 폴링**: SWR로 최신 데이터 갱신

### ⚠️ 알려진 이슈

**1. 차트/그래프 미구현**
- 현재 상태: 텍스트 메트릭만 표시 (카드 4개)
- 누락된 시각화:
  - 포트폴리오 가중치 파이 차트 (HYG 54%, DIA 6%, XLV 5%, ...)
  - 상관관계 히트맵 (24개 자산)
  - 리스크 점수 타임라인
  - GMM 확률 분포 차트
  - 섹터 로테이션 바 차트
- 필요 라이브러리: Recharts (이미 설치됨, `package.json` 확인 필요)

**2. 시그널 테이블 데이터 소스 불일치**
- `SignalsTable.tsx`: `/api/signals` 엔드포인트 호출 (기존 시그널 시스템)
- `MetricsGrid.tsx`: `/latest` 엔드포인트 호출 (integrated 결과)
- 문제: 두 데이터 소스가 다름
- 해결책: SignalsTable도 `/latest`의 `integrated_signals` 사용하도록 수정 필요

**3. 실시간 WebSocket 미연동**
- 현재: HTTP 폴링 (5초마다)
- Phase 4 (--realtime) 결과가 대시보드에 미반영
- BinanceStreamer 데이터 시각화 없음

### 📋 다음 작업 우선순위

**Priority 1: 차트 추가 (2-3시간)**
1. 포트폴리오 파이 차트 컴포넌트 (`PortfolioChart.tsx`)
2. 리스크 점수 라인 차트 (히스토리 API 추가 필요)
3. GMM 확률 바 차트

**Priority 2: 데이터 통합 (1시간)**
1. SignalsTable을 `/latest` 기반으로 수정
2. `integrated_signals` 필드 활용

**Priority 3: 실시간 기능 (4-5시간)**
1. WebSocket 연결 (`useWebSocket` hook)
2. Phase 4 결과 실시간 업데이트
3. 실시간 차트 애니메이션

### 🔧 환경 요구사항 확인

**프론트엔드 의존성:**
```bash
cd /home/tj/projects/autoai/eimas/frontend
npm list recharts  # 차트 라이브러리 확인
npm list swr       # ✅ 이미 사용 중
```

**백엔드 패키지:**
```bash
pip list | grep -E "fastapi|uvicorn|yfinance|anthropic"
```

모든 필수 패키지는 이미 설치됨 (requirements.txt 기반)

---

### v2.2.0 (2026-01-28) - Economic Insight Agent

**Task: agentcommand.txt 기반 인과적 분석 에이전트 구현**

- **`agent/` 모듈 신규 생성** (~2000 lines)
  - `agent/schemas/insight_schema.py`: Pydantic JSON 스키마 (424 lines)
  - `agent/core/adapters.py`: EIMAS 모듈 → Schema 변환 (631 lines)
  - `agent/core/orchestrator.py`: 6단계 추론 파이프라인 (830 lines)
  - `agent/cli.py`: CLI 인터페이스
  - `agent/evals/`: 10개 시나리오 평가 (ALL PASS)
  - `agent/tests/`: 단위 테스트 (스키마, 그래프 유틸리티, 통합)

- **핵심 기능**
  - Causality-first Analysis: 인과 그래프 + 메커니즘 경로 + 반증 가설
  - JSON-first Output: `meta`, `phenomenon`, `causal_graph`, `mechanisms`, `hypotheses`, `risk`, `suggested_data`, `next_actions`
  - EIMAS 통합: ShockPropagation, CriticalPath, GeniusAct, BubbleDetector 결과 자동 변환
  - 4개 프레임: MACRO, CRYPTO, MARKETS, MIXED (템플릿 그래프 제공)

- **EIMAS 어댑터 매핑**
  | EIMAS 모듈 | 변환 메서드 | 출력 |
  |-----------|------------|------|
  | ShockPropagationGraph | `adapt_shock_propagation()` | CausalGraph |
  | CriticalPathAggregator | `adapt_critical_path()` | RegimeShiftRisk[] |
  | GeniusActMacroStrategy | `adapt_genius_act()` | MechanismPath[] |
  | BubbleDetector | `adapt_bubble_detector()` | RegimeShiftRisk[] |
  | GraphClusteredPortfolio | `adapt_portfolio()` | NextAction[] |

- **사용법**
  ```bash
  # CLI
  python -m agent.cli --question "Fed 금리 인상 영향은?"
  python -m agent.cli --with-eimas --question "현재 시장 분석"

  # Eval
  python -m agent.evals.runner  # 10/10 시나리오 통과
  ```

- **Python API**
  ```python
  from agent import EconomicInsightOrchestrator, InsightRequest

  orchestrator = EconomicInsightOrchestrator()
  request = InsightRequest(question="스테이블코인 공급 증가 영향?")
  report = orchestrator.run(request)
  print(report.model_dump_json(indent=2))
  ```

---
*마지막 업데이트: 2026-01-28 14:40 KST*
