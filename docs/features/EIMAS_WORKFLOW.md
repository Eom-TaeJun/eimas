# EIMAS Workflow Guide
## Economic Intelligence Multi-Agent System

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EIMAS Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [Data Layer]          [Analysis Layer]         [Agent Layer]          │
│   ┌──────────┐          ┌─────────────┐         ┌─────────────┐        │
│   │ FRED API │────┐     │ Critical    │         │ Forecast    │        │
│   │ yfinance │    │     │ Path        │────────▶│ Agent       │        │
│   │ Binance  │────┼────▶│ Analyzer    │         ├─────────────┤        │
│   │ ETF Data │    │     ├─────────────┤         │ Analysis    │        │
│   │ News API │────┘     │ Granger     │────────▶│ Agent       │        │
│   └──────────┘          │ Causality   │         ├─────────────┤        │
│                         ├─────────────┤         │ Research    │        │
│   [Micro Layer]         │ LASSO       │────────▶│ Agent       │        │
│   ┌──────────┐          │ Regression  │         ├─────────────┤        │
│   │ OFI/VPIN │          ├─────────────┤         │ Strategy    │        │
│   │ Orderbook│──────────│ Regime      │────────▶│ Agent       │        │
│   │ Liquidity│          │ Detection   │         └──────┬──────┘        │
│   └──────────┘          └─────────────┘                │               │
│                                                        ▼               │
│                         [Debate Layer]          ┌─────────────┐        │
│                         ┌─────────────┐         │ Orchestrator│        │
│                         │ Methodology │◀────────│ (Claude)    │        │
│                         │ Debate      │         └──────┬──────┘        │
│                         ├─────────────┤                │               │
│                         │Interpretation│               ▼               │
│                         │ Debate      │         ┌─────────────┐        │
│                         └─────────────┘         │ Consensus   │        │
│                                                 │ Report      │        │
│                                                 └──────┬──────┘        │
│                                                        │               │
│   [Output Layer]                                       ▼               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐  ┌──────────┐          │
│   │Dashboard │    │ Alerts   │    │ Signals  │  │ Reports  │          │
│   │ (HTML)   │    │(Telegram)│    │ (DB)     │  │ (HTML)   │          │
│   └──────────┘    └──────────┘    └──────────┘  └──────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Collection Layer

### 2.1 FRED Collector (`lib/fred_collector.py`)

**목적**: 미국 연준 경제 데이터 수집

**수집 데이터**:
| 카테고리 | 시리즈 | 설명 |
|---------|-------|------|
| Rates | DFF, DGS2, DGS10, DGS30 | Fed Funds, 국채 금리 |
| Spreads | T10Y2Y, T10Y3M, BAMLH0A0HYM2 | 수익률 곡선, 하이일드 스프레드 |
| Inflation | CPIAUCSL, PCEPILFE, T5YIE | CPI, Core PCE, Breakeven |
| Employment | UNRATE, ICSA | 실업률, 신규 실업수당 |
| **Liquidity** | **RRPONTSYD, WTREGEN, WALCL** | **RRP, TGA, Fed Assets** |

**구현**:
```python
from lib.fred_collector import FREDCollector
collector = FREDCollector()
summary = collector.collect_all()

# 결과
# - RRP: $500B, TGA: $700B, Fed Assets: $7000B
# - Net Liquidity = Fed Assets - RRP - TGA = $5800B
# - Regime: Abundant/Normal/Tight/Stressed
```

**결과**: FREDSummary 객체 (금리, 스프레드, 유동성 지표 포함)

---

### 2.2 Market Data Collector (`lib/data_collector.py`)

**목적**: 주식/ETF/암호화폐 시장 데이터 수집

**수집 대상**:
- **주요 지수**: SPY, QQQ, IWM, DIA
- **섹터 ETF**: XLF, XLE, XLK, XLV, XLI, XLY, XLP, XLU, XLB, XLRE
- **스타일 ETF**: IWD (Value), IWF (Growth), SIZE
- **채권 ETF**: TLT, IEF, HYG, LQD, TIP
- **암호화폐**: BTC, ETH (via yfinance)
- **변동성**: VIX

**구현**:
```python
from lib.data_collector import DataManager
dm = DataManager(lookback_days=365)
data = dm.collect_all()
```

---

### 2.3 ETF Flow Analyzer (`lib/etf_flow_analyzer.py`)

**목적**: ETF 자금 흐름으로 시장 센티먼트 분석

**분석 항목**:
- Risk-On vs Risk-Off 자금 흐름
- 섹터 로테이션
- Value vs Growth 선호도
- 경기 사이클 단계 (Early/Mid/Late/Recession)

**구현**:
```python
from lib.etf_flow_analyzer import ETFFlowAnalyzer
analyzer = ETFFlowAnalyzer()
result = analyzer.analyze()

# 결과: SectorRotationResult, MarketRegimeResult
```

---

### 2.4 Real-time Streaming (`lib/binance_stream.py`)

**목적**: Binance WebSocket으로 실시간 호가/체결 수신

**기능**:
- Depth (호가창) 실시간 업데이트
- Trade (체결) 스트리밍
- 100ms 업데이트 속도

**구현**:
```python
from lib.binance_stream import BinanceStreamer, StreamConfig
config = StreamConfig(symbols=['BTCUSDT'])
streamer = BinanceStreamer(config, on_metrics=callback)
await streamer.start()
```

**결과**: 초당 10-15개 호가 업데이트, 실시간 OFI/VPIN 계산

---

## 3. Analysis Layer

### 3.1 Critical Path Analyzer (`lib/critical_path.py`)

**목적**: Bekaert et al. 기반 리스크 전이 경로 분석

**핵심 개념**:
- VIX = Uncertainty + Risk Aversion
- 리스크 전파 경로 추적
- Critical Path (위험 전이 경로) 식별

**주요 경로**:
| 경로 | 설명 | 해석 |
|-----|------|-----|
| VIX → HY Spread | 변동성 → 신용 위험 | 리스크오프 확산 |
| Yield Curve → Bank ETF | 금리 구조 → 은행 수익성 | 경기 순환 |
| Oil → Inflation | 에너지 → 물가 | 비용 인플레이션 |
| Dollar → EM | 달러 강세 → 신흥국 압박 | 글로벌 자금 흐름 |

**구현**:
```python
from lib.critical_path import CriticalPathAggregator
analyzer = CriticalPathAggregator()
result = analyzer.analyze(market_data)

# 결과: 리스크 레벨, 전이 경로, 경고 신호
```

---

### 3.2 Granger Causality Network (`lib/causal_network.py`)

**목적**: 시계열 간 선행-후행 관계 분석

**방법론**:
- Granger Causality Test
- DAG (Directed Acyclic Graph) 구축
- 인과관계 강도 측정

**구현**:
```python
from lib.causal_network import CausalNetworkAnalyzer
analyzer = CausalNetworkAnalyzer(max_lag=5)
network = analyzer.analyze(dataframe)

# 결과: edges (인과관계), paths (전이 경로)
```

---

### 3.3 LASSO Forecast (`lib/lasso_model.py`)

**목적**: Fed 금리 예측 (Sparsity 기반)

**방법론**:
- L1 정규화 (변수 선택)
- Treasury 변수 제외 (Simultaneity 방지)
- Horizon 분리 (단기/장기)

**구현**:
```python
from lib.lasso_model import LASSOForecastModel
model = LASSOForecastModel()
forecast = model.fit_predict(data, horizon_days=90)

# 결과: 금리 예측, 주요 변수 (feature importance)
```

---

### 3.4 Regime Detection (`lib/regime_detector.py`)

**목적**: 시장 국면 (Regime) 탐지

**Regime 유형**:
| Regime | 특징 | 전략 |
|--------|-----|------|
| Bull-LowVol | 상승 + 저변동성 | Buy & Hold |
| Bull-HighVol | 상승 + 고변동성 | 조심스러운 매수 |
| Bear-LowVol | 하락 + 저변동성 | 숏 or 현금 |
| Bear-HighVol | 하락 + 고변동성 | 방어적 |
| Sideways | 횡보 | Mean Reversion |

**구현**:
```python
from lib.regime_detector import RegimeDetector
detector = RegimeDetector()
regime = detector.detect(price_data, vix_data)

# 결과: MarketRegime, 확률, 전환 신호
```

---

### 3.5 Liquidity Analysis (`lib/liquidity_analysis.py`)

**목적**: 유동성 → 시장 Granger 인과관계 분석

**핵심 공식**:
```
Net Liquidity = Fed Assets - RRP - TGA
```

**해석**:
- RRP 감소 → 유동성 시장 유입 → 위험자산 상승
- TGA 증가 → 유동성 흡수 → 위험자산 하락

**구현**:
```python
from lib.liquidity_analysis import LiquidityMarketAnalyzer
analyzer = LiquidityMarketAnalyzer()
signals = analyzer.generate_signals()

# 결과: BULLISH/BEARISH/NEUTRAL, 인과관계 강도
```

---

## 4. Microstructure Layer

### 4.1 OFI Calculator (`lib/microstructure.py`)

**목적**: Order Flow Imbalance 계산

**공식**:
```
OFI = Σ(ΔBid_qty - ΔAsk_qty) / Σ(ΔBid_qty + ΔAsk_qty)
```

**해석**:
| OFI 값 | 해석 | 액션 |
|--------|-----|------|
| > +0.5 | 강한 매수 압력 | Bullish |
| +0.2 ~ +0.5 | 약한 매수 압력 | Slightly Bullish |
| -0.2 ~ +0.2 | 균형 | Neutral |
| -0.5 ~ -0.2 | 약한 매도 압력 | Slightly Bearish |
| < -0.5 | 강한 매도 압력 | Bearish |

**구현**:
```python
from lib.microstructure import OFICalculator
ofi_calc = OFICalculator(levels=5)
ofi = ofi_calc.calculate(orderbook)
```

---

### 4.2 VPIN Calculator (`lib/microstructure.py`)

**목적**: Volume-Synchronized Probability of Informed Trading

**공식**:
```
VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
```

**참고**: Easley et al. (2012) "Flow Toxicity"

**해석**:
| VPIN 값 | 해석 |
|---------|-----|
| > 0.7 | 정보 비대칭 높음 (위험) |
| 0.4 ~ 0.7 | 보통 |
| < 0.4 | 유동성 양호 |

**구현**:
```python
from lib.microstructure import VPINCalculator
vpin_calc = VPINCalculator(bucket_size=1.0, n_buckets=50)
for trade in trades:
    vpin = vpin_calc.add_trade(trade)
```

---

### 4.3 Real-time Pipeline (`lib/realtime_pipeline.py`)

**목적**: FRED + Binance 통합 실시간 분석

**데이터 흐름**:
```
FRED (1시간) → Macro Signal (40% weight)
                         ↓
                   Combined Signal → BUY/HOLD/SELL
                         ↑
Binance (100ms) → Micro Signal (60% weight)
```

**구현**:
```python
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig
config = PipelineConfig(symbols=['BTCUSDT', 'ETHUSDT'])
pipeline = RealtimePipeline(config)
await pipeline.start()
```

**결과**: IntegratedSignal (macro + micro 통합 신호)

---

## 5. Agent Layer

### 5.1 Base Agent (`agents/base_agent.py`)

**목적**: 모든 에이전트의 추상 베이스 클래스

**인터페이스**:
```python
class BaseAgent(ABC):
    async def execute(self, request: AgentRequest) -> AgentResponse
    async def form_opinion(self, topic: str, context: Dict) -> AgentOpinion
```

---

### 5.2 Forecast Agent (`agents/forecast_agent.py`)

**목적**: LASSO 기반 Fed 금리 예측

**워크플로우**:
1. 데이터 수집 (FRED + Market)
2. LASSO 모델 학습
3. Horizon별 예측 (30일, 90일, 180일)
4. 주요 변수 식별

**결과**: ForecastResult (예측값, 신뢰구간, 주요 변수)

---

### 5.3 Analysis Agent (`agents/analysis_agent.py`)

**목적**: Critical Path 분석 래핑

**토픽**:
- `market_outlook`: 시장 전망
- `primary_risk`: 주요 리스크
- `regime_stability`: 레짐 안정성
- `crypto_correlation`: 암호화폐 상관관계

---

### 5.4 Research Agent (`agents/research_agent.py`)

**목적**: Perplexity API로 실시간 리서치

**기능**:
- 뉴스 검색 및 요약
- 경제 이벤트 분석
- 시장 컨센서스 파악

---

### 5.5 Strategy Agent (`agents/strategy_agent.py`)

**목적**: 트레이딩 전략 권고

**출력**:
- 포지션 방향 (Long/Short/Neutral)
- 비중 (%)
- 손절/익절 레벨
- 근거

---

### 5.6 Meta Orchestrator (`agents/orchestrator.py`)

**목적**: 멀티 에이전트 조율 및 합의 도출

**워크플로우**:
```
1. 토픽 자동 감지
2. 에이전트별 의견 수집
3. 토론 진행 (최대 3라운드)
4. 합의 도출 (85% 일관성)
5. 최종 보고서 생성
```

---

## 6. Debate Layer

### 6.1 Methodology Debate (`agents/methodology_debate.py`)

**목적**: 분석 방법론 토론

**참가자**:
- Technical Analyst
- Fundamental Analyst
- Quantitative Analyst

**토론 주제**: 어떤 분석 방법이 현재 시장에 적합한가?

---

### 6.2 Interpretation Debate (`agents/interpretation_debate.py`)

**목적**: 데이터 해석 토론

**예시**:
- VIX 상승의 의미?
- 수익률 곡선 역전의 시사점?
- 유동성 감소의 영향?

---

## 7. Event Framework (`lib/event_framework.py`)

### 7.1 Event Types

| 이벤트 | 설명 | 임팩트 |
|--------|-----|--------|
| `YIELD_CURVE_INVERSION` | 수익률 곡선 역전 | 경기침체 선행 |
| `VIX_SPIKE` | VIX 급등 | 리스크오프 |
| `CREDIT_SPREAD_BLOWOUT` | 신용 스프레드 급등 | 신용 위기 |
| `RRP_SURGE` | RRP 급등 | 유동성 흡수 |
| `RRP_DRAIN` | RRP 급감 | 유동성 방출 (Bullish) |
| `TGA_BUILDUP` | TGA 증가 | 유동성 흡수 (Bearish) |
| `TGA_DRAWDOWN` | TGA 감소 | 유동성 방출 (Bullish) |
| `LIQUIDITY_STRESS` | Net Liquidity 급감 | 시장 스트레스 |

### 7.2 Event Detection

```python
from lib.event_framework import QuantitativeEventDetector
detector = QuantitativeEventDetector()
events = detector.detect_all(market_data, fred_data, liquidity_data)
```

---

## 8. Output Layer

### 8.1 Dashboard Generator (`lib/dashboard_generator.py`)

**목적**: HTML 대시보드 생성

**섹션**:
- 시장 개요
- 리스크 지표
- 예측 결과
- 에이전트 의견

---

### 8.2 Alert Manager (`lib/alert_manager.py`)

**목적**: 알림 통합 관리

**채널**:
- Telegram
- Slack
- Email (옵션)

**알림 레벨**:
- INFO: 정보성
- WARNING: 주의
- CRITICAL: 긴급

---

### 8.3 Signal Database (`core/database.py`)

**목적**: 신호 및 분석 결과 저장

**테이블**:
- `signals`: 트레이딩 신호
- `events`: 감지된 이벤트
- `forecasts`: 예측 결과
- `agent_opinions`: 에이전트 의견

---

## 9. Complete Workflow Example

```python
#!/usr/bin/env python3
"""EIMAS Complete Workflow"""

import asyncio
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.regime_detector import RegimeDetector
from lib.liquidity_analysis import LiquidityMarketAnalyzer
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig
from agents.orchestrator import MetaOrchestrator

async def main():
    # 1. 데이터 수집
    fred = FREDCollector()
    fred_summary = fred.collect_all()

    dm = DataManager(lookback_days=365)
    market_data = dm.collect_all()

    # 2. 분석
    regime = RegimeDetector().detect(market_data)
    liquidity = LiquidityMarketAnalyzer().generate_signals()

    # 3. 에이전트 토론
    orchestrator = MetaOrchestrator()
    consensus = await orchestrator.run_with_debate(
        topic="market_outlook",
        context={"regime": regime, "liquidity": liquidity}
    )

    # 4. 실시간 파이프라인 (옵션)
    config = PipelineConfig(symbols=['BTCUSDT'])
    pipeline = RealtimePipeline(config)
    await pipeline.start(duration_seconds=60)

    # 5. 결과 출력
    print(f"Regime: {regime}")
    print(f"Liquidity Signal: {liquidity}")
    print(f"Consensus: {consensus}")

asyncio.run(main())
```

---

## 10. Test Results Summary

### 10.1 FRED Liquidity Test
```
RRP: $6B (Δ +0.8B)
TGA: $837B (Δ +0.2B)
Net Liquidity: $5,797B
Regime: Abundant
```

### 10.2 Binance WebSocket Test (10초)
```
Depth updates: 91
Trades processed: 55
Signals: 87.9% neutral, 11% bearish, 1.1% bullish
```

### 10.3 Integration Test
```
✓ FRED API (RRP/TGA/Fed Assets)
✓ Binance WebSocket (orderbook/trades)
✓ OFI calculation
✓ VPIN calculation
✓ Macro-Micro signal integration
```

---

## 11. File Structure

```
eimas/
├── main.py                    # 메인 엔트리포인트
├── scheduler.py               # 스케줄러
│
├── agents/                    # 멀티 에이전트
│   ├── base_agent.py         # 베이스 클래스
│   ├── forecast_agent.py     # LASSO 예측
│   ├── analysis_agent.py     # Critical Path
│   ├── research_agent.py     # Perplexity 리서치
│   ├── strategy_agent.py     # 전략 권고
│   ├── orchestrator.py       # 조율자
│   ├── methodology_debate.py # 방법론 토론
│   └── interpretation_debate.py # 해석 토론
│
├── core/                      # 코어 인프라
│   ├── config.py             # 설정
│   ├── schemas.py            # 데이터 스키마
│   ├── database.py           # DB 연결
│   ├── debate.py             # 토론 프로토콜
│   └── signal_action.py      # 신호 → 액션
│
├── lib/                       # 라이브러리
│   ├── fred_collector.py     # FRED 데이터
│   ├── data_collector.py     # 시장 데이터
│   ├── critical_path.py      # 리스크 분석
│   ├── causal_network.py     # Granger 인과
│   ├── lasso_model.py        # LASSO 예측
│   ├── regime_detector.py    # 레짐 탐지
│   ├── liquidity_analysis.py # 유동성 분석
│   ├── microstructure.py     # OFI/VPIN
│   ├── binance_stream.py     # WebSocket
│   ├── realtime_pipeline.py  # 실시간 통합
│   ├── event_framework.py    # 이벤트 감지
│   ├── etf_flow_analyzer.py  # ETF 흐름
│   └── ...                   # 기타 모듈
│
├── api/                       # FastAPI
│   ├── server.py             # 서버
│   └── routes/               # 라우트
│
└── outputs/                   # 결과물
    ├── dashboards/           # HTML 대시보드
    ├── signals.db            # 신호 DB
    └── reports/              # 리포트
```

---

## 12. API Endpoints

| Endpoint | Method | 설명 |
|----------|--------|-----|
| `/api/health` | GET | 헬스 체크 |
| `/api/analysis/regime` | GET | 현재 레짐 |
| `/api/analysis/forecast` | GET | 금리 예측 |
| `/api/analysis/signals` | GET | 최근 신호 |
| `/api/debate/start` | POST | 토론 시작 |
| `/api/report/generate` | POST | 리포트 생성 |

---

*Last Updated: 2025-01-06*
*Version: 2.0*
