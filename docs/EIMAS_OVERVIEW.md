# EIMAS - Economic Intelligence Multi-Agent System
## 하향식 구조 및 최근 업데이트 (2026-02-04)

---

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [main.py 파이프라인 구조](#mainpy-파이프라인-구조)
3. [Phase별 상세 기능](#phase별-상세-기능)
4. [패키지 구조](#패키지-구조)
5. [최근 리팩토링 결과](#최근-리팩토링-결과)

---

## 시스템 개요

EIMAS는 거시경제 데이터와 시장 데이터를 수집하여 **AI 멀티에이전트 토론**을 통해 시장 전망과 투자 권고를 생성하는 시스템입니다.

### 핵심 특징
- 📊 **다층 분석**: 레짐 탐지 → 리스크 평가 → AI 토론 → 최종 권고
- 🤖 **멀티에이전트**: Claude 기반 에이전트들의 협업 및 토론
- 📈 **경제학 기반**: 학술 논문 기반 방법론 (Granger, Fama, Bekaert 등)
- 🔄 **실시간 대응**: 시장 변화에 따른 동적 전략 조정

### 실행 방법
```bash
# 전체 파이프라인 (~4-5분)
python main.py

# 빠른 분석 (~30초)
python main.py --quick

# AI 리포트 포함
python main.py --report
```

---

## main.py 파이프라인 구조

```
main.py (1088 lines)
│
├─ Phase 1: DATA COLLECTION
│  └─ 데이터 수집 (FRED, 시장, 크립토, RWA)
│
├─ Phase 2: ANALYSIS (12단계)
│  ├─ 2.1   Regime Detection
│  ├─ 2.1.1 GMM & Entropy
│  ├─ 2.2   Event Detection
│  ├─ 2.3   Liquidity Analysis
│  ├─ 2.4   Risk Scoring
│  ├─ 2.4.1 Microstructure
│  ├─ 2.4.2 Bubble Detection
│  ├─ 2.5   ETF Flow
│  ├─ 2.6   Genius Act Macro
│  ├─ 2.7   Custom ETF
│  ├─ 2.8   Shock Propagation
│  ├─ 2.9   Portfolio Optimization
│  ├─ 2.10  Integrated Strategy
│  ├─ 2.11  Allocation
│  └─ 2.12  Rebalancing
│
├─ Phase 3: MULTI-AGENT DEBATE
│  ├─ 3.1 FULL Mode (365일)
│  ├─ 3.2 REFERENCE Mode (90일)
│  └─ 3.3 Mode Comparison
│
├─ Phase 5: DATABASE STORAGE
│
├─ Phase 6: AI REPORT (--report)
│
└─ Phase 7: VALIDATION (--report)
```

---

## Phase별 상세 기능

### Phase 1: DATA COLLECTION

```
main.py
├─ [1.1] FRED Data
│  └─ lib/collectors/fred.py (FREDCollector)
│      → RRP, TGA, Fed Balance Sheet, Fed Funds Rate
│
├─ [1.2] Market Data
│  └─ lib/data_collector.py (DataManager)
│      → SPY, QQQ, TLT, GLD 등 24개 ETF
│
├─ [1.3] Crypto & RWA
│  └─ lib/data_loader.py (RWADataLoader)
│      → BTC-USD, ETH-USD, ONDO-USD, PAXG-USD, COIN
│
└─ [1.4] Market Indicators
   └─ lib/market_indicators.py (MarketIndicatorsCollector)
       → VIX, Fear & Greed Index
```

**출력:**
- `fred_summary`: RRP, TGA, Net Liquidity
- `market_data`: 24 tickers + 5 crypto/RWA
- `market_indicators`: VIX, sentiment

---

### Phase 2: ANALYSIS (핵심 분석 엔진)

#### 2.1 Regime Detection
```
lib/regime_detector.py (RegimeDetector)
├─ Input: SPY, QQQ 가격 데이터
├─ Method: GMM 3-state (Bull/Neutral/Bear)
└─ Output: regime, trend, volatility_cluster

└─ [2.1.1] GMM & Entropy
   lib/regime_analyzer.py (GMMRegimeAnalyzer)
   ├─ Method: Gaussian Mixture Model + Shannon Entropy
   └─ Output: regime_probs, entropy_level
```

#### 2.2 Event Detection
```
lib/event_framework/ ✨ 패키지 (리팩토링 완료)
├─ detector.py (QuantitativeEventDetector)
│  └─ 통계적 이벤트 탐지 (변동성, 거래량 급등)
├─ calendar.py (CalendarEventManager)
│  └─ FOMC, CPI, NFP 일정 관리
└─ framework.py (EventFramework)
   └─ 이벤트 통합 분석

경제학 기반: Fama et al. (1969) Event Study
```

#### 2.3 Liquidity Analysis
```
lib/analyzers/liquidity/ ✨ 패키지 (리팩토링 완료)
├─ analyzer.py
│  ├─ LiquidityMarketAnalyzer: 유동성-시장 상관관계
│  └─ DynamicLagAnalyzer: 시차 분석
└─ schemas.py
   ├─ LiquidityImpactResult
   └─ DynamicLagResult

경제학 기반: Granger (1969) Causality
Method: Fed Liquidity = Balance Sheet - RRP - TGA
```

#### 2.4 Risk Scoring (3단계)
```
[2.4] CriticalPathAggregator
lib/critical_path/aggregator.py
├─ Input: 유동성, 레짐, 이벤트
└─ Output: base_risk_score (0-100)

[2.4.1] Microstructure Quality
lib/microstructure/ ✨ 패키지
├─ DailyMicrostructureAnalyzer
│  ├─ Amihud Lambda (비유동성)
│  ├─ Roll Spread (Bid-Ask 추정)
│  └─ VPIN (정보 비대칭)
└─ Output: MarketQualityMetrics
   └─ microstructure_adjustment: ±10

[2.4.2] Bubble Risk Overlay
lib/bubble/ ✨ 패키지 (리팩토링 완료)
├─ detector.py (BubbleDetector)
│  ├─ Run-up Check: 2년 수익률 > 100%
│  ├─ Volatility Spike: Z-score > 2σ
│  └─ Share Issuance: 증가율 > 5%
├─ framework.py (FiveStageBubbleFramework)
│  └─ JP Morgan 5-Stage: Paradigm → Credit → Leverage → Speculation → Collapse
└─ Output: BubbleRiskMetrics
   └─ bubble_risk_adjustment: +0~15

경제학 기반: 
- "Bubbles for Fama" (Greenwood et al. 2019)
- JP Morgan Bubble Framework

최종 리스크 점수:
final_risk = base_risk + microstructure_adj + bubble_adj
```

#### 2.5 ETF Flow Analysis
```
lib/analyzers/etf/ ✨ 패키지 (리팩토링 완료)
├─ flow_analyzer.py (ETFFlowAnalyzer)
│  ├─ 섹터별 자금 흐름 추적
│  └─ Growth/Value 로테이션 탐지
└─ Output: SectorRotationResult

Phase에서 사용:
- main.py Line 620: ETFFlowAnalyzer 실행
```

#### 2.6 Genius Act Macro
```
lib/genius_act/ ✨ 패키지 (리팩토링 완료)
├─ strategy.py (GeniusActMacroStrategy)
│  └─ 확장 유동성 = M + S·B*
├─ crypto_risk.py (CryptoRiskEvaluator)
│  ├─ 스테이블코인 담보 분류
│  └─ Risk Score: USDC(15) < USDT(35) < DAI(40) < USDe(50)
└─ Output: genius_act_regime, signals

경제학 기반: Genius Act Liquidity Model
```

#### 2.7 Custom ETF Builder
```
lib/strategies/etf/ ✨ 패키지 (리팩토링 완료)
├─ builder.py (CustomETFBuilder)
│  ├─ SupplyChainGraph: 공급망 분석
│  └─ ThemeDetector: AI, EV, Biotech 등
└─ Output: ThemeETF, ThemeAllocation
```

#### 2.8 Shock Propagation
```
lib/shock_propagation/ ✨ 패키지 (리팩토링 완료)
├─ graph.py (ShockPropagationGraph)
│  ├─ Lead-Lag 분석
│  └─ Granger Causality
├─ granger.py (GrangerCausalityAnalyzer)
└─ Output: ShockPath, PropagationAnalysis

lib/causality/ ✨ 패키지 (리팩토링 완료)
├─ graph.py (CausalityGraphEngine)
├─ builder.py (CausalNetworkBuilder)
└─ analyzer.py (CausalNetworkAnalyzer)

경제학 기반: Granger (1969) Causality
```

#### 2.9 Portfolio Optimization
```
lib/graph_portfolio/ ✨ 패키지
├─ mst_analyzer.py (MSTSystemRiskAnalyzer)
│  └─ MST 기반 중심성 분석
└─ hrp_optimizer.py (HRPOptimizer)
   └─ Hierarchical Risk Parity

경제학 기반:
- MST: Mantegna (1999)
- HRP: De Prado (2016)

Output: portfolio_weights
```

#### 2.10 Integrated Strategy
```
lib/integrated_strategy.py
└─ IntegratedStrategy
   ├─ Portfolio 결과 + Causality 결과 통합
   └─ 진입/청산 시그널 생성
```

#### 2.11-2.12 Allocation & Rebalancing
```
lib/strategies/allocation/ ✨ 패키지 (리팩토링 완료)
├─ engine.py (AllocationEngine)
│  ├─ MVO (Mean-Variance Optimization)
│  ├─ Risk Parity
│  ├─ HRP (Hierarchical Risk Parity)
│  └─ Black-Litterman
└─ Output: AllocationResult

lib/strategies/rebalancing/ ✨ 패키지 (리팩토링 완료)
├─ policy.py (RebalancingPolicy)
│  ├─ Periodic (주간/월간)
│  ├─ Threshold (편차 임계값)
│  └─ Hybrid (정기 + 임계값)
└─ Output: RebalanceDecision

경제학 기반:
- Markowitz (1952) Portfolio Theory
- Black-Litterman (1992)
```

---

### Phase 3: MULTI-AGENT DEBATE

```
agents/orchestrator.py (MetaOrchestrator)
│
├─ [3.1] FULL Mode (365일 데이터)
│  ├─ CriticalPathAnalyst
│  ├─ ForecastAgent (LASSO)
│  ├─ ResearchAgent (Perplexity)
│  ├─ StrategyAgent
│  └─ VerificationAgent
│  
├─ [3.2] REFERENCE Mode (90일 데이터)
│  └─ 동일 에이전트, 짧은 기간
│
└─ [3.3] Mode Comparison
   lib/dual_mode_analyzer.py (DualModeAnalyzer)
   └─ FULL vs REFERENCE 비교 및 최종 결정

경제학 기반:
- LASSO (L1 Regularization)
- Multi-Agent Consensus
```

**에이전트 역할:**
- **CriticalPathAnalyst**: 리스크 경로 분석
- **ForecastAgent**: LASSO 기반 Fed 금리 예측
- **ResearchAgent**: Perplexity로 최신 뉴스/리서치 수집
- **StrategyAgent**: 포트폴리오 전략 권고
- **VerificationAgent**: 다른 에이전트 검증

---

### Phase 5: DATABASE STORAGE

```
core/database.py
├─ EventDatabase (data/events.db)
│  └─ 탐지된 이벤트 저장
├─ SignalDatabase (outputs/realtime_signals.db)
│  └─ 실시간 시그널 저장
└─ JSON 결과 저장
   ├─ outputs/eimas_YYYYMMDD_HHMMSS.json
   └─ outputs/eimas_YYYYMMDD_HHMMSS.md
```

---

### Phase 6: AI REPORT (--report 옵션)

```
lib/ai_report_generator.py (AIReportGenerator)
│
├─ Technical Indicators
│  └─ RSI, MACD, 볼린저밴드
│
├─ Scenario Analysis
│  ├─ Bull Case
│  ├─ Base Case
│  └─ Bear Case
│
├─ News & Sentiment
│  ├─ Perplexity API (최신 뉴스 20개)
│  ├─ Fear & Greed Index
│  └─ VIX 구조
│
└─ IB-style Memorandum
   └─ Investment Banking 스타일 보고서

Output:
- outputs/ai_report_YYYYMMDD.md (19KB)
- outputs/ib_memorandum_YYYYMMDD.md (3.6KB)
```

---

### Phase 7: VALIDATION (--report 옵션)

```
lib/whitening_engine.py (WhiteningEngine)
└─ 경제학적 해석 및 설명

lib/autonomous_agent.py (AutonomousFactChecker)
└─ AI 출력 팩트체킹
```

---

## 패키지 구조

### 리팩토링 완료 패키지 (2026-02-04) ✨

#### 1. 분석 패키지 (Analyzers)
```
lib/analyzers/
├─ etf/                      ✨ NEW (1059 lines)
│  ├─ flow_analyzer.py       ETF 자금 흐름
│  ├─ enums.py               MarketSentiment, StyleRotation
│  └─ schemas.py             ETFData, SectorRotationResult
│
└─ liquidity/                ✨ NEW (960 lines)
   ├─ analyzer.py            LiquidityMarketAnalyzer
   ├─ analyzer.py            DynamicLagAnalyzer
   └─ schemas.py             LiquidityImpactResult
```

#### 2. 전략 패키지 (Strategies)
```
lib/strategies/
├─ etf/                      ✨ NEW (956 lines)
│  ├─ builder.py             CustomETFBuilder
│  ├─ builder.py             SupplyChainGraph
│  ├─ enums.py               ThemeCategory
│  └─ schemas.py             ThemeStock, ThemeETF
│
├─ rebalancing/              ✨ NEW (894 lines)
│  ├─ policy.py              RebalancingPolicy
│  ├─ enums.py               RebalanceFrequency
│  └─ schemas.py             RebalanceDecision, TradingCostModel
│
└─ allocation/               ✨ NEW (886 lines)
   ├─ engine.py              AllocationEngine
   ├─ enums.py               AllocationStrategy
   └─ schemas.py             AllocationResult, AllocationConstraints
```

#### 3. 분석 프레임워크 (Analysis Frameworks)
```
lib/bubble/                  ✨ NEW (1727 lines)
├─ detector.py               BubbleDetector
├─ framework.py              FiveStageBubbleFramework
├─ enums.py                  BubbleWarningLevel, JPMorganBubbleStage
└─ schemas.py                BubbleDetectionResult

lib/causality/               ✨ NEW (1851 lines)
├─ graph.py                  CausalityGraphEngine
├─ granger.py                GrangerCausalityAnalyzer
├─ builder.py                CausalNetworkBuilder
├─ analyzer.py               CausalNetworkAnalyzer
├─ enums.py                  EdgeType, NodeType
└─ schemas.py                CausalEdge, CausalityPath

lib/shock_propagation/       ✅ (1277 lines)
├─ graph.py                  ShockPropagationGraph
├─ granger.py                GrangerCausalityAnalyzer
├─ lead_lag.py               LeadLagAnalyzer
├─ enums.py                  NodeLayer, CausalityStrength
└─ schemas.py                ShockPath, PropagationAnalysis

lib/event_framework/         ✅ (1372 lines)
├─ detector.py               QuantitativeEventDetector
├─ calendar.py               CalendarEventManager
├─ framework.py              EventFramework
├─ impact.py                 EventImpactAnalyzer
├─ enums.py                  EventType, EventImportance
└─ schemas.py                Event, EventImpact
```

#### 4. 기타 주요 패키지
```
lib/genius_act/              ✅ (1600 lines)
├─ strategy.py               GeniusActMacroStrategy
├─ crypto_risk.py            CryptoRiskEvaluator
├─ liquidity.py              ExtendedLiquidityModel
└─ stablecoin_risk.py        MultiDimensionalRiskScore

lib/validation/              ✅ (1482 lines)
├─ manager.py                ValidationAgentManager
├─ consensus.py              ConsensusEngine
├─ claude.py                 ClaudeValidationAgent
└─ perplexity.py             PerplexityValidationAgent

lib/microstructure/          ✅ (2136 lines)
├─ analyzer.py               MicrostructureAnalyzer
├─ daily_analyzer.py         DailyMicrostructureAnalyzer
└─ metrics.py                Amihud Lambda, VPIN

lib/graph_portfolio/         ✅ (1823 lines)
├─ mst_analyzer.py           MSTSystemRiskAnalyzer
└─ hrp_optimizer.py          HRPOptimizer

lib/operational/             ✅ (3745 lines)
├─ engine.py                 OperationalEngine
└─ monitor.py                PortfolioMonitor

lib/critical_path/           ✅ (3389 lines)
├─ aggregator.py             CriticalPathAggregator
└─ crypto_sentiment.py       CryptoSentimentAnalyzer
```

---

## 최근 리팩토링 결과

### 2026-02-04: 분석 + 전략 패키지 리팩토링 (7개)

#### ✅ 완료된 작업

**리팩토링된 모듈:**
1. **bubble_detector.py** (1186줄) + **bubble_framework.py** (541줄)
   → **lib/bubble/** (6 files, 1727 lines)

2. **causality_graph.py** (1099줄) + **causal_network.py** (752줄)
   → **lib/causality/** (7 files, 1851 lines)

3. **etf_flow_analyzer.py** (1059줄)
   → **lib/analyzers/etf/** (4 files, 1059 lines)

4. **liquidity_analysis.py** (960줄)
   → **lib/analyzers/liquidity/** (3 files, 960 lines)

5. **custom_etf_builder.py** (956줄)
   → **lib/strategies/etf/** (4 files, 956 lines)

6. **rebalancing_policy.py** (894줄)
   → **lib/strategies/rebalancing/** (4 files, 894 lines)

7. **allocation_engine.py** (886줄)
   → **lib/strategies/allocation/** (4 files, 886 lines)

#### 📊 통계

| 항목 | 수치 |
|------|------|
| 원본 파일 | 7개 |
| 총 라인 수 | 8,333줄 |
| 생성된 패키지 | 7개 |
| 생성된 모듈 | ~45개 |
| Git 커밋 | 7개 |
| GitHub Push | ✅ 완료 |

#### 🎯 개선 효과

1. **모듈화**: 대형 파일 → 기능별 패키지
   - bubble_detector.py (1186줄) → 6개 파일로 분산
   - causality_graph.py (1099줄) → 7개 파일로 분산

2. **구조 개선**:
   ```
   Before: lib/bubble_detector.py (1186 lines)
   
   After:  lib/bubble/
           ├─ enums.py          (3 Enums)
           ├─ schemas.py        (8 Dataclasses)
           ├─ detector.py       (~830 lines)
           ├─ framework.py      (~470 lines)
           ├─ utils.py          (2 functions)
           └─ __init__.py       (Public API)
   ```

3. **경제학적 근거 문서화**:
   - 각 모듈 헤더에 학술 논문 인용
   - 경제학 방법론 명시

4. **하위 호환성 유지**:
   - 기존 import 경로 유지
   - `from lib.bubble_detector import BubbleDetector` → 여전히 작동
   - `from lib.bubble import BubbleDetector` → 새 경로도 가능

5. **테스트 용이성**:
   - 모듈별 독립 테스트 가능
   - Mock 데이터 주입 간편

---

## 경제학 방법론 요약

| 방법론 | 사용처 | 참고 문헌 |
|--------|--------|-----------|
| **LASSO** | ForecastAgent | Tibshirani (1996) |
| **Granger Causality** | Liquidity, Shock | Granger (1969) |
| **GMM 3-State** | RegimeAnalyzer | Hamilton (1989) |
| **Shannon Entropy** | RegimeAnalyzer | Shannon (1948) |
| **Event Study** | EventFramework | Fama et al. (1969) |
| **Bubble Detection** | BubbleDetector | Greenwood et al. (2019) |
| **JP Morgan 5-Stage** | BubbleFramework | JP Morgan (2021) |
| **Amihud Lambda** | Microstructure | Amihud (2002) |
| **VPIN** | Microstructure | Easley et al. (2012) |
| **MST** | Portfolio | Mantegna (1999) |
| **HRP** | Portfolio | De Prado (2016) |
| **Black-Litterman** | Allocation | Black & Litterman (1992) |

---

## 실행 결과 예시 (2026-02-04)

```bash
$ python main.py
```

**출력:**
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
⏱️ TIME: 249.4s

Output: outputs/eimas_20260204_183853.json
```

**생성된 파일:**
- `outputs/eimas_20260204_183853.json` (96KB)
- `outputs/eimas_20260204_183853.md` (Markdown)
- `outputs/ai_report_20260204_183825.md` (19KB, --report 옵션)
- `outputs/ib_memorandum_20260204_183853.md` (3.6KB, --report 옵션)

---

## 다음 단계

### 추가 리팩토링 대상 (TOP 3)

1. **trading_db.py** (1204 lines)
   → `lib/db/trading/`

2. **data_collector.py** (858 lines)
   → `lib/collectors/market/`

3. **market_indicators.py** (1021 lines)
   → `lib/collectors/indicators/`

### 기능 개선 계획

1. **실시간 대시보드**
   - Frontend: Next.js 16 (이미 구현됨)
   - 5초 자동 폴링
   - WebSocket 연결 추가

2. **백테스트 엔진**
   - 과거 데이터로 전략 검증
   - 성과 측정 및 보고서

3. **알림 시스템**
   - 중요 이벤트 발생 시 알림
   - Slack/Discord 연동

---

## 참고 자료

- **프로젝트 루트**: `/home/tj/projects/autoai/eimas/`
- **메인 문서**: `CLAUDE.md` (사용자 가이드)
- **아키텍처**: `ARCHITECTURE.md` (상세 설계)
- **대시보드 가이드**: `DASHBOARD_QUICKSTART.md`
- **GitHub**: 최신 커밋 87ff936

---

*Last Updated: 2026-02-04 19:00 KST*
*Version: 2.2.2 (Refactoring Edition)*
