# Market Anomaly Detector v2.2 - Architecture Guide

> 경제 AI 에이전트 개발 참고용 아키텍처 문서

---

## 1. 프로젝트 개요

시장 이상 징후 탐지 시스템. yfinance/FRED에서 데이터를 수집하고, 다양한 지표를 계산한 후, Rule-based + ML 기반으로 이상 신호를 탐지하여 대시보드로 시각화한다.

---

## 2. 파일별 역할 및 함수 정리

### 2.1 `collectors.py` - 데이터 수집기

**역할**: yfinance와 FRED API에서 시장 데이터를 수집

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `DataManager` | 전체 데이터 수집을 관리하는 메인 클래스 |
| `collect_all()` | 모든 자산군 데이터 일괄 수집 |
| `_download_with_retry()` | yfinance 다운로드 + 재시도 로직 |
| `_collect_fred_data()` | FRED API에서 경제 지표 수집 (금리, 인플레 등) |
| `_collect_crypto_data()` | 암호화폐 데이터 수집 (fallback 메커니즘 포함) |

**반환 형태**: `Dict[str, pd.DataFrame]` (ticker → OHLCV DataFrame)

---

### 2.2 `processors.py` - 지표 계산기

**역할**: 수집된 데이터에서 기술적 지표 및 통계 지표 계산

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `DataProcessor` | 지표 계산 메인 클래스 |
| `process_all()` | 모든 티커에 대해 지표 일괄 계산 |
| `_calculate_indicators()` | 개별 티커 지표 계산 (MA, RSI, BB, Z-score 등) |
| `calculate_ratios()` | 자산 비율 계산 (예: IWM/SPY, GLD/SPY) |
| `get_latest_indicators()` | 최신 시점 지표 딕셔너리 반환 |
| `get_latest_ratios()` | 최신 시점 비율 딕셔너리 반환 |

**계산 지표**:
- **이동평균**: MA5, MA20, MA50, MA200
- **모멘텀**: RSI, Return_1d, Return_5d
- **변동성**: Volatility_5d, Volatility_20d, Bollinger Band
- **Z-score**: Close_Z, Return_Z, Volume_Z
- **볼륨**: Volume_Ratio, Volume_Spike

---

### 2.3 `detectors.py` - 이상 탐지기 (Rule-based)

**역할**: 규칙 기반으로 통계적 이상, Cross-Asset 이상, 선행 신호 탐지

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `Signal` | 이상 신호 데이터클래스 (type, ticker, level, description 등) |
| `StatisticalDetector` | Z-score 기반 통계적 이상 탐지 (가격/수익률/거래량/RSI/BB) |
| `CrossAssetAnomalyDetector` | 자산간 상관관계 이상 탐지 (금리-금, VIX-주식, 리스크패리티) |
| `EarlyWarningDetector` | 선행 신호 탐지 (거래량 축적, 소형주 약세, 안전자산 선호) |
| `AnomalyDetector` | 3개 탐지기 통합, `detect_all()` 제공 |
| `merge_ml_risk_into_signals()` | ML risk_prob를 Rule-based 신호에 병합 |

**신호 레벨**: `NORMAL` < `WARNING` < `ALERT` < `CRITICAL`

---

### 2.4 `risk_model.py` - ML 기반 위험 확률 추정

**역할**: 향후 N일 내 Max Drawdown 확률을 ML로 예측

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `RiskLabelGenerator` | 미래 N일 Max Drawdown 기준 이진 레이블 생성 |
| `FeatureMatrixBuilder` | Risk Model용 Feature Matrix 구성 |
| `RiskModel` | Logistic/RF/GB 모델 학습 및 예측 |
| `estimate_risk_probability()` | **메인 API** - 현재 시점 위험 확률 추정 |
| `calculate_garch_volatility()` | GARCH(1,1) 조건부 변동성 계산 |
| `calculate_tail_risk()` | EVT 기반 Tail Risk 측정 (GPD) |

**Feature 목록**:
- 이동평균: `Price_vs_MA5`, `Price_vs_MA20`, `MA5_Slope`, `MA20_Slope`
- 거래량: `Volume_Spike_5d`, `Volume_Spike_20d`, `Volume_Trend`
- 변동성: `Volatility_5d`, `Volatility_20d`, `Vol_Ratio`
- GARCH: `GARCH_Vol`, `Vol_Persistence`, `Vol_Half_Life`
- Tail Risk: `Tail_Index`, `ES_99`, `Extreme_Event_Prob`

---

### 2.5 `regime_detector.py` - 시장 국면 탐지

**역할**: 시장 Regime(BULL/BEAR/NEUTRAL) 판단

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `RegimeDetector` | MA 기반 또는 Markov Switching 기반 Regime 판단 |
| `MarkovSwitchingRegime` | Hamilton(1989) 2-state/3-state Markov 모델 |
| `fetch_data()` | Regime 분석용 데이터 수집 (SPY, 섹터 ETF 등) |
| `detect_regime()` | 단일 자산 Regime 판단 |
| `analyze_all()` | 모든 자산 Regime 분석 + 글로벌 Regime 결정 |
| `get_portfolio_recommendation()` | Regime 기반 포트폴리오 배분 추천 |

**Regime 결정 로직 (MA 기반)**:
- Price > MA200 + MA50 > MA200 + MA 기울기 양수 → `BULL`
- 반대 조건 → `BEAR`
- 혼합 → `NEUTRAL`

---

### 2.6 `critical_path_analyzer.py` - 리스크/불확실성 분석

**역할**: Bekaert et al. 연구 기반 Risk Appetite와 Uncertainty 분리 측정

**핵심 클래스/함수**:
| 클래스/함수 | 기능 |
|------------|------|
| `RiskAppetiteUncertaintyIndex` | Risk Appetite와 Uncertainty 지수 계산 |
| `calculate_uncertainty_index()` | 불확실성 지수 (VIX, 실현변동성, 괴리, 섹터상관) |
| `calculate_risk_appetite_index()` | 리스크 선호 지수 (HYG/LQD, XLY/XLP, IWM/SPY, VRP) |
| `determine_market_state()` | 두 지수 조합으로 시장 상태 결정 |
| `calculate_rolling_zscore()` | 롤링 Z-score 계산 유틸리티 |
| `calculate_realized_volatility()` | 실현 변동성 계산 (연율화) |

**시장 상태**:
| 상태 | Risk Appetite | Uncertainty |
|------|---------------|-------------|
| NORMAL | 중간 | 낮음 |
| SPECULATIVE | 높음 | 낮음 |
| STAGNANT | 낮음 | 높음 |
| CRISIS | 낮음 | 매우 높음 |

---

### 2.7 `dashboard_generator.py` - 대시보드 생성

**역할**: 분석 결과를 HTML 대시보드로 시각화

**핵심 함수**:
| 함수 | 기능 |
|------|------|
| `generate_dashboard()` | 메인 대시보드 HTML 생성 |
| `generate_asset_risk_section()` | 자산군별 위험 현황 카드 생성 |
| `generate_regime_display()` | Regime 표시 (게이지 바 포함) |
| `generate_crypto_panel_html()` | 암호화폐 전용 패널 생성 |
| `generate_signal_table()` | 신호 테이블 HTML 생성 |

---

### 2.8 `config_loader.py` - 설정 관리

**역할**: YAML 설정 파일 로드 및 중앙화된 설정 관리

**핵심 함수**:
| 함수 | 기능 |
|------|------|
| `get_risk_bucket()` | 확률 → 위험 레벨 변환 (LOW/MEDIUM/HIGH/CRITICAL) |
| `get_asset_class()` | 티커 → 자산 클래스 변환 (equity/bond/commodity/fx/crypto) |
| `RiskThresholds` | Z-score, Volume 임계값 설정 클래스 |
| `AssetMetadata` | 자산 메타데이터 컨테이너 |

---

### 2.9 `core/config.py` - API 및 모델 설정

**역할**: API 키 관리 및 LLM 모델 설정

**핵심 클래스**:
| 클래스 | 기능 |
|--------|------|
| `APIConfig` | API 키 관리 (OpenAI, Anthropic, Perplexity, Gemini, FRED) |
| `get_client()` | API 클라이언트 싱글톤 반환 |
| `MODELS` | 용도별 모델 설정 (orchestrator, code_gen, analysis 등) |
| `AGENT_CONFIG` | 에이전트별 설정 (model, max_tokens, temperature) |

---

### 2.10 `main.py` - 메인 파이프라인

**역할**: 전체 파이프라인 실행 오케스트레이션

**핵심 함수**:
| 함수 | 기능 |
|------|------|
| `run_pipeline()` | 메인 실행 함수 |
| `run_analysis()` | 데이터 수집 → 지표 계산 → 탐지 → ML → 대시보드 |
| `merge_risk_probability()` | ML risk_prob를 신호에 병합 |
| `generate_ai_interpretation()` | Claude API로 분석 결과 해석 생성 |

---

## 3. 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (Pipeline)                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: 데이터 수집                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ collectors.py → DataManager.collect_all()                    │   │
│  │   - yfinance: 주식, ETF, 원자재, 환율, 암호화폐              │   │
│  │   - FRED API: 금리, 인플레이션, 경제지표                      │   │
│  │   → Dict[ticker, DataFrame] 반환                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: 지표 계산                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ processors.py → DataProcessor.process_all()                  │   │
│  │   - 이동평균 (MA5, MA20, MA50, MA200)                         │   │
│  │   - 모멘텀 (RSI, Returns)                                     │   │
│  │   - 변동성 (Volatility, Bollinger Band)                       │   │
│  │   - Z-score (Close_Z, Return_Z, Volume_Z)                     │   │
│  │   - 자산 비율 (IWM/SPY, GLD/SPY 등)                           │   │
│  │   → indicators Dict, ratios Dict 반환                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  Step 3a: Rule-based 탐지       │  │  Step 3b: Regime 분석           │
│  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │
│  │ detectors.py              │  │  │  │ regime_detector.py        │  │
│  │  - StatisticalDetector    │  │  │  │  - MA 기반 Regime         │  │
│  │  - CrossAssetAnomalyDet.  │  │  │  │  - Markov Switching       │  │
│  │  - EarlyWarningDetector   │  │  │  │  - 섹터별 분석            │  │
│  │  → List[Signal] 반환      │  │  │  │  → regime_info 반환       │  │
│  └───────────────────────────┘  │  │  └───────────────────────────┘  │
└─────────────────────────────────┘  └─────────────────────────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: ML 기반 위험 확률 추정                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ risk_model.py → estimate_risk_probability()                  │   │
│  │   - Feature 구성 (MA, Volume, Volatility, GARCH, Tail Risk)  │   │
│  │   - Logistic / RF / GB 모델 예측                              │   │
│  │   - 자산 클래스별 개별 모델 지원                               │   │
│  │   → risk_prob (0~1), risk_level 반환                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 신호 병합: merge_ml_risk_into_signals()                       │   │
│  │   - Rule-based 신호에 ML risk_prob 추가                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: Risk Appetite / Uncertainty 분석                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ critical_path_analyzer.py → RiskAppetiteUncertaintyIndex     │   │
│  │   - Uncertainty: VIX, 실현변동성, VIX-RV 괴리, 섹터상관       │   │
│  │   - Risk Appetite: HYG/LQD, XLY/XLP, IWM/SPY, VRP             │   │
│  │   → market_state (NORMAL/SPECULATIVE/STAGNANT/CRISIS) 반환    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6: AI 해석 생성 (Optional)                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Claude API → generate_ai_interpretation()                     │   │
│  │   - 신호 + Regime + Risk Appetite 종합 해석                   │   │
│  │   - 투자 전략 제안                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 7: 대시보드 생성                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ dashboard_generator.py → generate_dashboard()                 │   │
│  │   - 자산군별 위험 현황                                         │   │
│  │   - Regime 표시 (게이지 바)                                    │   │
│  │   - 신호 테이블                                                │   │
│  │   - 암호화폐 패널                                              │   │
│  │   → HTML 파일 출력                                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 경제학적 방법론 정리

| 개념 | 구현 위치 | 설명 |
|------|----------|------|
| **Z-score 기반 이상 탐지** | detectors.py | 정규분포 가정, \|Z\| > 2는 95% 신뢰구간 이탈 |
| **Cross-Asset Correlation** | detectors.py | 자산간 이론적 상관관계 이탈 탐지 |
| **GARCH(1,1)** | risk_model.py | 변동성 클러스터링 모델링 (Engle 1982) |
| **Extreme Value Theory** | risk_model.py | Fat-tail 분포의 꼬리 위험 측정 (GPD) |
| **Markov Switching** | regime_detector.py | 시장 국면 전환 확률 모델 (Hamilton 1989) |
| **Bekaert VIX 분해** | critical_path_analyzer.py | VIX = Uncertainty + Risk Appetite |
| **Variance Risk Premium** | critical_path_analyzer.py | VRP = VIX² - 실현분산 |

---

## 5. 새 경제 AI 에이전트 개발 시 참고사항

### 데이터 수집 패턴
```python
from collectors import DataManager

manager = DataManager(lookback_days=60)
market_data, collection_status = manager.collect_all(tickers_config)
```

### 지표 계산 패턴
```python
from processors import DataProcessor

processor = DataProcessor(market_data)
processor.process_all()
indicators = processor.get_latest_indicators()
ratios = processor.get_latest_ratios()
```

### 이상 탐지 패턴
```python
from detectors import AnomalyDetector

detector = AnomalyDetector(thresholds_config)
signals = detector.detect_all(indicators, ratios, regime_info)
```

### ML 위험 확률 패턴
```python
from risk_model import estimate_risk_probability

risk_df = estimate_risk_probability(snapshot_df, model=trained_model)
```

---

## 6. 디렉토리 구조

```
market_anomaly_detector/
├── main.py                    # 메인 파이프라인
├── collectors.py              # 데이터 수집
├── processors.py              # 지표 계산
├── detectors.py               # Rule-based 탐지
├── risk_model.py              # ML 위험 예측
├── regime_detector.py         # 시장 국면 판단
├── critical_path_analyzer.py  # Risk/Uncertainty 분석
├── dashboard_generator.py     # HTML 대시보드
├── config_loader.py           # 설정 관리
├── config/                    # YAML 설정 파일
│   ├── tickers.yaml
│   └── thresholds.yaml
├── core/                      # 핵심 모듈
│   └── config.py              # API 설정
├── models/                    # 학습된 모델 저장
├── outputs/                   # 출력 파일
└── dashboard/                 # 대시보드 관련
```

---

---

# Part 2: Graph-Based Portfolio & Causality Engine (NEW)

## 개요

무한 에셋(N → ∞) 환경에서의 포트폴리오 최적화와 인과관계 기반 투자 전략 시스템

### 핵심 철학

- **Whitebox AI**: 단순 예측이 아닌 설명 가능한 인과관계 규명
- **Volume > Price**: 거래량 급증 = 정보 비대칭 신호
- **M = B + S·B***: 스테이블코인 포함 확장 유동성 공식
- **Impulse Response**: 충격 전파 경로 기반 전략

---

## 신규 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATED STRATEGY                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │   FOUNDATION    │         │  INTELLIGENCE   │           │
│  │  (GC-HRP.py)    │         │    (SPG.py)     │           │
│  ├─────────────────┤         ├─────────────────┤           │
│  │ • Clustering    │         │ • Lead-Lag      │           │
│  │ • HRP Weights   │    +    │ • Granger       │           │
│  │ • Risk Parity   │         │ • Critical Path │           │
│  └────────┬────────┘         └────────┬────────┘           │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       ▼                                     │
│           ┌─────────────────────┐                          │
│           │   APPLICATION       │                          │
│           │ (integrated.py)     │                          │
│           ├─────────────────────┤                          │
│           │ • Leading Tilt      │                          │
│           │ • Shock Warning     │                          │
│           │ • Volume Anomaly    │                          │
│           │ • Risk Metrics      │                          │
│           └─────────────────────┘                          │
│                       │                                     │
│                       ▼                                     │
│           [Strategy Recommendation]                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 신규 모듈 상세

### 1. Foundation: `lib/graph_clustered_portfolio.py`

**목적**: 무한 에셋(N → ∞) 환경에서 공분산 행렬 특이점 문제 해결

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `CorrelationNetwork` | 상관관계 기반 그래프 구축, PageRank/Eigenvector 중심성 |
| `AssetClusterer` | K-means, Louvain, GMM, Hierarchical 클러스터링 |
| `RepresentativeSelector` | 클러스터당 대표 자산 선정 (Centrality, Volume, Sharpe) |
| `HierarchicalRiskParity` | 공분산 역행렬 없이 계층적 리스크 패리티 가중치 계산 |
| `GraphClusteredPortfolio` | 전체 파이프라인 통합 |

#### 알고리즘 흐름

```
[N개 자산 유니버스]
        ↓
Step 1: Correlation Network 구축
        - 상관관계 > threshold → 엣지 연결
        - 거래량 가중치 적용
        ↓
Step 2: Community Detection (Louvain/K-means)
        - 유사 자산 클러스터링
        - N → K개 클러스터
        ↓
Step 3: Representative Selection
        - 클러스터당 1-3개 대표 선정
        - Eigenvector Centrality 기반
        ↓
Step 4: HRP on Reduced Universe
        - Ward Linkage 계층적 클러스터링
        - Quasi-diagonal 정렬
        - Recursive Bisection 가중치 배분
        ↓
[최종 포트폴리오 가중치]
```

#### 주요 메트릭

- **Diversification Ratio**: 개별 변동성 가중평균 / 포트폴리오 변동성
- **Effective N**: 1 / HHI (Herfindahl-Hirschman Index)
- **Risk Contribution**: 자산별 리스크 기여도

---

### 2. Intelligence: `lib/shock_propagation_graph.py`

**목적**: 거시지표/자산 간 인과관계 규명 및 충격 전파 경로 분석

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `LeadLagAnalyzer` | Cross-correlation at lags로 최적 시차 탐색 |
| `GrangerCausalityAnalyzer` | Granger Causality 통계적 검정 |
| `ShockPropagationGraph` | 방향성 그래프(DAG) 구축 및 분석 |

#### 경제학적 레이어 구조

```
LAYER 1: POLICY (정책)
├── Fed Funds Rate
├── ECB Rate
└── ...

LAYER 2: LIQUIDITY (유동성)
├── RRP (Reverse Repo)
├── TGA (Treasury General Account)
├── M2
└── Stablecoin Supply (USDT, USDC)

LAYER 3: RISK_PREMIUM (리스크 프리미엄)
├── VIX
├── HY Spread
└── Credit Spread

LAYER 4: ASSET_PRICE (자산 가격)
├── SPY, QQQ
├── TLT, GLD
└── BTC, ETH
```

#### 충격 전파 예시

```
FED_FUNDS (POLICY)
    ↓ [Lag 2d, r=0.99]
DXY (ASSET_PRICE)
    ↓ [Lag 3d, r=0.63]
VIX (RISK_PREMIUM)
    ↓ [Lag 1d, r=-0.07]
SPY (ASSET_PRICE)
    ↓ [Lag 10d, r=0.97]
TLT (ASSET_PRICE)
    ↓ [Lag 1d, r=0.74]
GLD (ASSET_PRICE)

Total Propagation: 15 days
```

#### 노드 역할 분류

- **LEADING**: Out-degree > In-degree (선행 지표)
- **LAGGING**: In-degree > Out-degree (후행 지표)
- **BRIDGE**: Betweenness Centrality 높음 (전파 중개자)
- **ISOLATED**: 연결 없음

---

### 3. Application: `lib/integrated_strategy.py`

**목적**: Foundation + Intelligence를 통합한 실제 투자 전략

#### 핵심 기능

| 기능 | 설명 |
|------|------|
| **Leading Tilt** | 선행지표에 +15% 가중치 부여 |
| **Shock Warning** | 상위 레이어 충격 감지 → 하위 레이어 경고 |
| **Volume Anomaly** | 거래량 급증(MA20 대비 3x+) 탐지 |
| **Risk Metrics** | VaR, CVaR, Max Drawdown 계산 |

#### 시그널 유형

```python
class SignalType(Enum):
    LEADING_TILT = "leading_tilt"       # 선행지표 기반 틸팅
    SHOCK_WARNING = "shock_warning"     # 충격 전파 경고
    VOLUME_SPIKE = "volume_spike"       # 거래량 급증
    REGIME_SHIFT = "regime_shift"       # 레짐 변화
    REBALANCE = "rebalance"             # 리밸런싱 필요
```

#### 거래량 해석 (정보 비대칭 이론)

| 조건 | 해석 |
|------|------|
| 거래량↑ + 가격↑ | NEW_INFORMATION (새로운 정보 유입, 강한 매수) |
| 거래량↑ + 가격↓ | EXHAUSTION (패닉 매도 또는 고점 신호) |
| 거래량↑ + 가격→ | ACCUMULATION (축적 또는 분배) |

#### 출력 구조

```python
@dataclass
class StrategyRecommendation:
    portfolio_weights: Dict[str, float]      # 기본 가중치
    tilted_weights: Dict[str, float]         # 틸팅 적용 가중치
    tilt_factors: Dict[str, float]           # 틸팅 팩터
    signals: List[Signal]                    # 생성된 시그널
    risk_metrics: Dict[str, float]           # 리스크 메트릭

    leading_exposure: float                  # 선행지표 노출도
    lagging_exposure: float                  # 후행지표 노출도
    shock_vulnerability: float               # 충격 취약도

    actions: List[Dict]                      # 실행 액션
    warnings: List[str]                      # 경고 메시지
```

---

## 사용 예시

```python
from lib.integrated_strategy import IntegratedStrategy, ClusteringMethod

# 전략 엔진 초기화
strategy = IntegratedStrategy(
    correlation_threshold=0.3,
    clustering_method=ClusteringMethod.KMEANS,
    leading_tilt_factor=0.15,
    volume_surge_threshold=3.0
)

# 전략 수립
recommendation = strategy.fit(
    returns=asset_returns_df,      # 자산 수익률
    macro_data=macro_df,           # 거시지표 (Fed Funds, VIX, etc.)
    volumes=volume_df              # 거래량 (선택)
)

# 결과 확인
print(recommendation.tilted_weights)  # 최종 가중치
print(recommendation.signals)         # 시그널
print(recommendation.warnings)        # 경고
```

---

## 테스트 결과 (샘플 데이터)

### Foundation 결과
```
Input:  100개 자산, 252일 데이터
Output: 7개 클러스터, 14개 대표 자산
        Diversification Ratio: 6.62
        Effective N: 52.9
        Expected Volatility: 6.42%
```

### Intelligence 결과
```
Input:  6개 거시지표 (FED_FUNDS, DXY, VIX, SPY, TLT, GLD)
Output: 25개 유의미한 인과관계 엣지
        Critical Path: FED_FUNDS → DXY → VIX → SPY → TLT → GLD
        Total Lag: 15 days
```

### Application 결과
```
Signals Generated: 3
  - [MEDIUM] shock_warning: VIX 5일간 -4.9% 하락
  - [MEDIUM] volume_spike: ASSET_21 거래량 급증 (3.6x) - EXHAUSTION
  - [MEDIUM] volume_spike: ASSET_44 거래량 급증 (4.0x) - NEW_INFORMATION

Risk Metrics:
  - Volatility: 5.53%
  - VaR (95%): -9.36%
  - Max Drawdown: -11.15%
```

---

## 향후 개발 계획

### Phase 1: 실제 데이터 연동
- [ ] yfinance로 실시간 시장 데이터 수집
- [ ] FRED API 거시지표 연동
- [ ] 스테이블코인 공급량 데이터 (CoinGecko/DefiLlama)

### Phase 2: 확장 유동성 공식 구현
- [ ] M = B + S·B* 모델링
- [ ] 스테이블코인 시가총액 → 달러 유동성 프록시
- [ ] DeFi TVL 데이터 통합

### Phase 3: 대시보드 통합
- [ ] Streamlit 네트워크 시각화 (pyvis)
- [ ] Critical Path 실시간 모니터링
- [ ] 시그널 알림 시스템

### Phase 4: 백테스팅
- [ ] 과거 데이터로 전략 검증
- [ ] Walk-forward 최적화
- [ ] 트랜잭션 비용 고려

---

## 참고 문헌

1. **HRP**: Lopez de Prado (2016) - "Building Diversified Portfolios that Outperform Out-of-Sample"
2. **Granger Causality**: Granger (1969) - "Investigating Causal Relations by Econometric Models"
3. **VIX Decomposition**: Bekaert et al. - Risk Appetite vs Uncertainty
4. **Louvain Algorithm**: Blondel et al. (2008) - Community Detection
5. **Volume Analysis**: Kyle (1985) - Market Microstructure

---

*문서 작성일: 2025-12-26*
*Part 2 추가: 2026-01-07*

---

# Part 3: Advanced Strategy & Verification Engine (NEW)

## 개요

4단계 고급 전략 및 검증 시스템:
1. **Whitening Engine**: AI 결과 경제학적 역설계
2. **Custom ETF Builder**: 테마/공급망 기반 ETF
3. **Genius Act Macro**: 스테이블코인-국채 연계 전략
4. **Autonomous Agent**: 자율 팩트체킹

---

## 신규 모듈 상세

### 1. `lib/whitening_engine.py` - 경제학적 역설계

**목적**: AI 결과를 경제학적으로 해석하고 인과관계 검증

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `WhiteningEngine` | 포트폴리오 결과 화이트박스 해석 |
| `EconomicFactor` | 경제 팩터 열거형 (금리, 유동성, 달러 등) |
| `FactorAttribution` | 팩터별 기여도 귀인 |
| `CausalValidation` | 인과관계 가설 검증 |
| `EconomicNarrative` | 종합 서사 생성 |

#### 팩터-자산 매핑

```python
FACTOR_ASSET_MAPPING = {
    "interest_rate": ["TLT", "IEF", "SHY"],
    "liquidity": ["BTC-USD", "QQQ", "IWM"],
    "dollar_strength": ["EEM", "GLD", "DXY"],
    "stablecoin_flow": ["BTC-USD", "ETH-USD", "COIN"],
    "tech_momentum": ["QQQ", "SMH", "SOXX"],
    "risk_appetite": ["HYG", "XLY", "IWM"],
    "geopolitical": ["GLD", "XAR", "ITA"],
    # ...
}
```

#### 인과관계 가설 템플릿

| 가설 | 경로 |
|------|------|
| 금리 인하 기대 → 성장주 선호 | FED_FUNDS → DGS10 → QQQ |
| 유동성 증가 → 위험자산 선호 | M2 → NET_LIQUIDITY → SPY → BTC-USD |
| 달러 약세 → 신흥시장/금 강세 | DXY → EEM → GLD |
| 스테이블코인 유입 → 크립토 강세 | USDT_SUPPLY → BTC-USD |

---

### 2. `lib/custom_etf_builder.py` - 테마 ETF 빌더

**목적**: 공급망 그래프 기반 테마 ETF 구축

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `ThemeCategory` | 테마 열거형 (AI, EV, Clean Energy 등) |
| `SupplyChainLayer` | 공급망 레이어 (원자재 → 제조 → 최종사용자) |
| `SupplyChainGraph` | NetworkX 기반 공급망 DAG |
| `CustomETFBuilder` | ETF 생성 및 비중 계산 |
| `ETFComparator` | 다중 테마 비교 분석 |

#### 사전 정의 테마

```
AI_SEMICONDUCTOR:   ASML → TSM → NVDA → MSFT
ELECTRIC_VEHICLE:   ALB → Panasonic → TSLA → ChargePoint
CLEAN_ENERGY:       FSLR → ENPH → NEE → Sunrun
DEFENSE:            Component → LMT → RTX → PLTR
CYBERSECURITY:      Cloudflare → PANW → CRWD
BLOCKCHAIN:         BTC → MSTR → COIN → HOOD
```

#### 공급망 레이어 구조

```
[RAW_MATERIAL] → [COMPONENT] → [EQUIPMENT]
                                   ↓
                            [MANUFACTURER]
                                   ↓
                            [INTEGRATOR]
                                   ↓
                            [DISTRIBUTION] → [END_USER]
```

#### 분석 기능

- **중심성 점수**: PageRank + Betweenness Centrality
- **병목 탐지**: 단일 공급원 의존도
- **충격 전파 분석**: 특정 노드 disruption 시 영향
- **리밸런싱 시그널**: 현재 vs 목표 비중 차이

---

### 3. `lib/genius_act_macro.py` - 스테이블코인 매크로 전략

**목적**: Genius Act(스테이블코인 규제법) 기반 매크로 전략

#### 핵심 공식: M = B + S·B*

```
M = 총 유효 유동성
B = 기본 유동성 (Fed BS - RRP - TGA)
S = 스테이블코인 승수
B* = 스테이블코인 담보 자산 (국채)
```

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `ExtendedLiquidityModel` | M = B + S·B* 계산 |
| `GeniusActRules` | 규칙 기반 시그널 생성 |
| `GeniusActMacroStrategy` | 전략 통합 엔진 |
| `LiquidityMonitor` | 실시간 모니터링 |

#### 시그널 유형

| 시그널 | 트리거 | 영향 자산 |
|--------|--------|----------|
| `STABLECOIN_SURGE` | USDT+USDC >5% 주간 증가 | TLT, BTC-USD |
| `RRP_DRAIN` | 역레포 >10% 월간 감소 | SPY, QQQ |
| `TGA_DRAIN` | TGA >15% 월간 감소 | SPY, IWM |
| `LIQUIDITY_INJECTION` | 순유동성 >2% 증가 | 위험자산 전반 |
| `TREASURY_DEMAND` | 스테이블코인 담보 수요 | SHY, BIL |

#### 유동성 레짐

```
EXPANSION:    순유동성↑ + 스테이블코인↑ → Risk-On
CONTRACTION:  순유동성↓ + RRP↑ → Risk-Off
TRANSITION:   혼합 신호 → 주의
NEUTRAL:      안정 상태
```

---

### 4. `lib/autonomous_agent.py` - 자율 팩트체킹

**목적**: AI 출력 결과에 대한 자동 사실 검증

#### 핵심 컴포넌트

| 클래스 | 기능 |
|--------|------|
| `AutonomousFactChecker` | 메인 에이전트 |
| `NumericVerifier` | 수치 데이터 검증 |
| `TrendVerifier` | 추세 주장 검증 |
| `CausalVerifier` | 인과관계 검증 |
| `PerplexityVerifier` | Perplexity API 실시간 검증 |
| `AIOutputVerifier` | JSON 출력 전문 검증 |

#### 주장 유형 분류

```python
class ClaimType(Enum):
    NUMERIC = "numeric"         # 수치 (예: "금리 5.25%")
    TREND = "trend"             # 추세 (예: "상승세")
    CAUSAL = "causal"           # 인과 (예: "금리 인상 → 주가 하락")
    PREDICTION = "prediction"   # 예측
    FACT = "fact"               # 사실 진술
```

#### 검증 상태

| 상태 | 설명 |
|------|------|
| `VERIFIED` | 완전 검증됨 |
| `PARTIALLY_VERIFIED` | 부분 확인 |
| `CONTRADICTED` | 반박됨 (수정 필요) |
| `OUTDATED` | 오래된 정보 |
| `UNABLE_TO_VERIFY` | 검증 불가 |

#### 검증 파이프라인

```
[AI 출력 텍스트]
        ↓
Step 1: 문장 분리 & 주장 추출
        ↓
Step 2: 주장 유형 분류 (NUMERIC/TREND/CAUSAL)
        ↓
Step 3: 전문 도구로 1차 검증
        ↓
Step 4: Perplexity API로 실시간 크로스체크
        ↓
Step 5: 결과 종합 & 등급 결정 (A/B/C/D)
        ↓
[검증 보고서 + 수정 제안]
```

---

## 통합 사용 예시

```python
# 1. 포트폴리오 화이트닝
from lib.whitening_engine import WhiteningEngine

whitening = WhiteningEngine()
explanation = whitening.explain_allocation({
    "allocation": {"SPY": 0.3, "QQQ": 0.2, "TLT": 0.15, ...},
    "changes": {"SPY": 0.1, "GLD": -0.05}
})
print(explanation.summary)
print(explanation.key_drivers)

# 2. 테마 ETF 생성
from lib.custom_etf_builder import CustomETFBuilder, ThemeCategory

builder = CustomETFBuilder()
ai_etf = builder.create_etf(ThemeCategory.AI_SEMICONDUCTOR)
print(ai_etf.target_weight)

shock = builder.get_shock_impact_analysis(ai_etf, "ASML")
print(shock['propagation_path'])

# 3. 매크로 전략
from lib.genius_act_macro import GeniusActMacroStrategy, LiquidityIndicators

strategy = GeniusActMacroStrategy()
result = strategy.analyze(current_indicators, previous_indicators)
print(result['regime'])
print(result['signals'])

# 4. 팩트체킹
from lib.autonomous_agent import AutonomousFactChecker

checker = AutonomousFactChecker()
report = await checker.verify_document(ai_generated_text)
print(report['summary']['grade'])
```

---

## 테스트 결과

### Whitening Engine
```
팩터 귀인: tech_momentum(33%), liquidity(20%), dollar(-10%)
인과관계 검증: 5개 가설 테스트
전체 신뢰도: 35% (추가 데이터 필요)
```

### Custom ETF Builder
```
AI Semiconductor ETF: 13개 종목
Top 5 Concentration: 53.7%
병목 노출: ASML 충격 시 80% 영향
```

### Genius Act Macro
```
레짐: EXPANSION (Risk-On)
시그널: STABLECOIN_SURGE, RRP_DRAIN, LIQUIDITY_INJECTION
포지션: LONG SPY(30%), LONG BTC-USD(15%), LONG QQQ(10%)
```

### Autonomous Agent
```
총 검증 주장: 4개
검증됨: 1, 부분 검증: 2
등급: C (추가 검증 필요)
```

---

*Part 3 추가: 2026-01-07*
