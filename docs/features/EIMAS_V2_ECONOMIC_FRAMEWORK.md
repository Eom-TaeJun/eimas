# EIMAS v2.0 경제학 프레임워크

> **핵심 철학**: 시장이 가는 방향을 쫓아가되, 경제학적 해석을 더한다.
> **목표**: 선행지표 → 크리티컬 패스 → 네트워크 온톨로지 → AI 토론 → 경제학적 결론

---

## 1. 이론적 기반

### 1.1 핵심 명제

```
시장 지표는 기준금리 발표를 선행한다.
→ 크리티컬 패스를 통해 몇 가지 지표로 시장 예측 가능
```

### 1.2 거시경제 프레임워크

```
기본 모델:
ΔM↑ → r↓ → C↑, I↑ → Y↑

통화 중립성 제약:
M↑ → P↑ (물가 상승) → 실질 효과 상쇄

장기 균형:
모든 kinetic 관계 → 0
최종 잔존: M ↔ P (통화-물가 관계만 유지)
```

### 1.3 현재 시장 해석 (2025년 기준)

| 관측 | 해석 |
|------|------|
| 금, 은 상승 | 달러 가치 하락 신호 |
| 달러 약세 | 공급 증가 (dovish 확정) |
| 금, 은, 알루미늄, 구리, 백금 | 버블 조짐 |
| 저금리 환경 | 커모디티 상승, 바이오(재정적자 해소) |

**결론**: 시장은 인플레이션에 베팅 중

---

## 2. 크리티컬 패스 분석

### 2.1 선행-후행 관계

```
┌─────────────────────────────────────────────────────────┐
│                    선행 지표 그룹                        │
├─────────────────────────────────────────────────────────┤
│  금/은 가격 ──┐                                          │
│  구리/알루미늄 ─┼──→ 달러 인덱스 ──→ Fed 금리 결정       │
│  VIX ─────────┘                                          │
│                                                          │
│  Net Buy 비율 (5-10x) ──→ Buy 시그널                    │
│  체결강도/거래량 ──→ 모멘텀 확인                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 이벤트 기반 데이터 분리

```python
# 예시: 삼성전자-테슬라 계약
events = {
    "samsung_tesla": {
        "date": "2024-XX-XX",
        "before_regime": "traditional_valuation",
        "after_regime": "ev_ecosystem_premium",
        "data_treatment": "separate_models"
    }
}

# 이벤트 전후 데이터는 다른 모델로 학습
# 단기 이벤트 대응 vs 장기 구조 분석 분리
```

### 2.3 인플레이션-고용 트레이드오프

```
Phillips Curve 현대적 해석:
┌──────────────────────────────────────┐
│ 금리 인상 → 인플레이션↓, 실업↑       │
│ 금리 인하 → 인플레이션↑, 실업↓       │
│                                       │
│ 2025년 9월 이후:                      │
│ - 금, 은 상승                         │
│ - 달러 가치 하락                      │
│ → 공급 증가 신호 (dovish)             │
│ → 인플레이션 베팅이 유리              │
└──────────────────────────────────────┘
```

---

## 3. 네트워크 온톨로지 (Palantir 스타일)

### 3.1 핵심 개념

| 용어 | 정의 | 경제학 대응 |
|------|------|------------|
| **Node** | 경제 변수/지표 | GDP, CPI, Fed Rate, VIX... |
| **Semantic Edge** | 분야 특수 연결 | 통화정책 → 금리, 금리 → 주가 |
| **Kinetic Edge** | 수치화된 영향력 | β 계수, Granger 인과성 |

### 3.2 경제학자에게 쉬운 이유

```
Palantir Ontology ≈ VAR 충격반응함수 (IRF)

Vector Autoregression:
Y_t = A₁Y_{t-1} + A₂Y_{t-2} + ... + ε_t

충격반응함수:
∂Y_{t+h}/∂ε_t = IRF(h)

이것이 곧:
- Node: 변수 (Y의 각 요소)
- Kinetic: A 행렬의 계수
- Dynamic: 시간에 따른 충격 전파
```

### 3.3 장기 균형 수렴

```
시간 → ∞:
대부분의 kinetic 관계 → 0
잔존 관계: M ↔ P (통화-물가)

단기:
모든 관계가 유의미 (복잡한 네트워크)

장기:
통화 중립성 → 실질 변수 영향 無
```

### 3.4 네트워크 시각화 구조

```python
# 학습 가능한 네트워크 구조
class EconomicNetwork:
    """
    - 관계 있는 선 (Significant): Granger p < 0.05
    - 관계 없는 선 (Insignificant): Granger p >= 0.05
    - 학습 대상: edge weights (kinetic values)
    """

    nodes = ["M", "r", "C", "I", "Y", "P", "VIX", "Gold", ...]

    edges = {
        ("M", "r"): {"weight": -0.8, "significant": True},
        ("r", "C"): {"weight": -0.3, "significant": True},
        ("r", "I"): {"weight": -0.5, "significant": True},
        ("C", "Y"): {"weight": 0.6, "significant": True},
        ("I", "Y"): {"weight": 0.4, "significant": True},
        ("M", "P"): {"weight": 0.9, "significant": True},  # 장기 유지
        ...
    }
```

---

## 4. AI 시그널 포착

### 4.1 기존 이론과의 차이

```
전통 이론: 가격 상승 → 공급 > 수요
실제 관측: 급등 시 공급이 더 많음?

해석:
- 모멘텀 트레이딩
- 정보 비대칭
- 기대 형성 (자기실현적 예언)

AI 역할:
- 이상 패턴 포착
- 전통 이론과 괴리 시 알림
```

### 4.2 Buy 시그널 로직

```python
def detect_buy_signal(data):
    """
    Net Buy가 5-10배 많으면 Buy 타이밍
    AI가 캡처하고, 인간이 해석
    """
    net_buy_ratio = data['buy_volume'] / data['sell_volume']

    if net_buy_ratio >= 5:
        return {
            "signal": "STRONG_BUY",
            "ratio": net_buy_ratio,
            "confidence": min(net_buy_ratio / 10, 1.0)
        }
    elif net_buy_ratio >= 2:
        return {"signal": "MODERATE_BUY", ...}
    else:
        return {"signal": "NEUTRAL", ...}
```

### 4.3 포착 대상 지표

| 지표 | 용도 | 소스 |
|------|------|------|
| 체결강도 | 매수/매도 압력 | 거래소 API |
| 거래량 | 관심도/유동성 | yfinance |
| Net Buy Ratio | 매수 우위 | 계산 |
| 호가 스프레드 | 유동성 품질 | 실시간 API |
| 대량 체결 | 기관 움직임 | 거래소 API |

---

## 5. 에이전트 토론 프레임워크

### 5.1 경제학 도메인 에이전트

```
┌─────────────────────────────────────────────────────────┐
│                    MetaOrchestrator                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Monetarist  │  │ Keynesian   │  │ Austrian    │     │
│  │ Agent       │  │ Agent       │  │ Agent       │     │
│  │             │  │             │  │             │     │
│  │ M↔P 중시    │  │ Y,C,I 중시  │  │ 금본위,     │     │
│  │ 통화중립성  │  │ 승수효과    │  │ 사이클 중시 │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                │              │
│         └────────────────┼────────────────┘              │
│                          ▼                               │
│                    ┌───────────┐                        │
│                    │  Debate   │                        │
│                    │  Protocol │                        │
│                    └─────┬─────┘                        │
│                          ▼                               │
│                  Economic Consensus                      │
└─────────────────────────────────────────────────────────┘
```

### 5.2 토론 주제 예시

```yaml
debate_topics:
  - topic: "current_market_regime"
    question: "현재 시장은 어떤 국면인가?"
    perspectives:
      monetarist: "통화 확장기 → 인플레이션 우려"
      keynesian: "총수요 부족 → 부양 필요"
      austrian: "인위적 저금리 → 버블 형성"

  - topic: "gold_silver_surge"
    question: "금/은 가격 상승의 의미는?"
    perspectives:
      monetarist: "달러 가치 하락 반영"
      keynesian: "안전자산 선호 (불확실성)"
      austrian: "fiat 불신, 실물자산 회귀"

  - topic: "fed_policy_impact"
    question: "Fed dovish 정책의 영향은?"
    perspectives:
      monetarist: "장기 인플레이션, 단기 자산 상승"
      keynesian: "경기 부양, 고용 개선"
      austrian: "부채 사이클 연장, 미래 위기 심화"
```

### 5.3 충격반응 기반 토론

```python
async def irf_based_debate(shock: str, horizon: int):
    """
    VAR 충격반응함수 기반 에이전트 토론

    Args:
        shock: 충격 변수 (e.g., "fed_rate", "oil_price")
        horizon: 예측 기간 (일/주/월)
    """
    # 1. VAR 모델에서 IRF 계산
    irf_results = calculate_irf(shock, horizon)

    # 2. 각 에이전트가 IRF 해석
    interpretations = []
    for agent in [monetarist, keynesian, austrian]:
        opinion = await agent.interpret_irf(irf_results)
        interpretations.append(opinion)

    # 3. 토론 및 합의
    consensus = await debate_protocol.reach_consensus(interpretations)

    return {
        "irf": irf_results,
        "interpretations": interpretations,
        "consensus": consensus
    }
```

---

## 6. 실행 워크플로우

### 6.1 프롬프트 구조

```
1. TO DO (목표)
   - 무엇을 분석할 것인가?
   - 어떤 결론을 도출할 것인가?

2. 프롬프트 & 키워드 & 핵심 함수
   - Perplexity: 최신 뉴스/데이터 수집
   - Claude: 경제학적 해석
   - 핵심 함수: calculate_irf(), detect_signal()

3. 실행
   - 순차적 또는 병렬 실행
   - 결과 종합 및 보고서 생성
```

### 6.2 단계별 프로세스

```
Phase 1: 데이터 수집 (Perplexity)
┌─────────────────────────────────────┐
│ - 최신 경제 지표                     │
│ - 시장 뉴스 및 센티먼트              │
│ - Fed 발언/정책 동향                 │
│ - 원자재 가격 동향                   │
└─────────────────────────────────────┘
              ↓
Phase 2: 시그널 포착 (AI)
┌─────────────────────────────────────┐
│ - Net Buy Ratio 계산                 │
│ - 체결강도/거래량 분석               │
│ - 이상 패턴 탐지                     │
│ - 크리티컬 패스 업데이트             │
└─────────────────────────────────────┘
              ↓
Phase 3: 경제학적 해석 (Agent Debate)
┌─────────────────────────────────────┐
│ - 각 학파별 해석                     │
│ - IRF 기반 예측                      │
│ - 토론 및 합의 도출                  │
└─────────────────────────────────────┘
              ↓
Phase 4: 결론 도출
┌─────────────────────────────────────┐
│ - 시장 방향 예측                     │
│ - 리스크 평가                        │
│ - 투자 전략 제안                     │
└─────────────────────────────────────┘
```

### 6.3 버전 업그레이드 체크리스트

```yaml
version_upgrade:
  research_economics:
    - [ ] 새로운 경제학 이론 반영
    - [ ] VAR/IRF 모델 개선
    - [ ] 에이전트 학파 추가

  system_improvement:
    - [ ] 데이터 수집 소스 확장
    - [ ] 시그널 포착 정확도 개선
    - [ ] 실시간 처리 성능 향상

  rwa_integration:
    - [ ] Real World Asset 데이터 연동
    - [ ] 토큰화 자산 분석
    - [ ] 온체인 데이터 통합

  db_construction:
    - [ ] 시계열 DB 구축 (TimescaleDB)
    - [ ] 이벤트 DB (분리 학습용)
    - [ ] 지식 베이스 (Elicit 연동)
```

---

## 7. 핵심 함수 정의

### 7.1 크리티컬 패스 분석

```python
async def analyze_critical_path(
    indicators: List[str],
    target: str = "fed_rate",
    method: str = "granger"
) -> CriticalPathResult:
    """
    선행지표 → 타겟 변수 경로 분석

    Args:
        indicators: 분석할 지표 리스트
        target: 예측 대상
        method: granger, var, network

    Returns:
        significant_paths: 유의미한 경로
        lead_times: 선행 시간
        strength: 영향력 수치
    """
    pass
```

### 7.2 네트워크 온톨로지 구축

```python
def build_economic_ontology(
    data: pd.DataFrame,
    significance_level: float = 0.05
) -> EconomicNetwork:
    """
    Palantir 스타일 경제 네트워크 구축

    Nodes: 경제 변수
    Semantic Edges: 이론적 관계
    Kinetic Edges: 실증적 관계 (계수)
    """
    pass
```

### 7.3 시그널 포착

```python
def capture_market_signals(
    real_time_data: Dict,
    thresholds: Dict = {"net_buy_ratio": 5, "volume_spike": 2}
) -> List[Signal]:
    """
    AI 기반 시장 시그널 포착

    - Net Buy Ratio
    - 체결강도
    - 거래량 스파이크
    - 호가 불균형
    """
    pass
```

### 7.4 경제학파 토론

```python
async def economic_school_debate(
    topic: str,
    market_data: Dict,
    irf_results: Optional[Dict] = None
) -> DebateResult:
    """
    경제학파 간 토론

    Agents:
    - Monetarist: 통화량 중시
    - Keynesian: 총수요 중시
    - Austrian: 사이클/버블 중시

    Returns:
        consensus: 합의된 견해
        dissent: 이견
        confidence: 합의 수준
    """
    pass
```

---

## 8. RWA (Real World Asset) 통합

### 8.1 개념

```
Real World Asset = 실물 자산의 토큰화

예시:
- 부동산 토큰
- 원자재 담보 토큰
- 국채 토큰
- 탄소 크레딧

EIMAS 활용:
- 온체인 데이터로 실물 경제 추적
- 토큰 가격 ↔ 실물 가격 괴리 분석
- 유동성 흐름 모니터링
```

### 8.2 데이터 소스

```yaml
rwa_data_sources:
  on_chain:
    - ethereum: "MakerDAO RWA vaults"
    - polygon: "RealT properties"
    - stellar: "Franklin Templeton MMF"

  off_chain:
    - fred: "Treasury yields, GDP"
    - bls: "Employment data"
    - commodity_exchanges: "Gold, Silver, Copper"

  integration:
    - oracle_feeds: "Chainlink price feeds"
    - cross_validation: "On-chain vs Off-chain"
```

---

## 9. 다음 단계

### 9.1 즉시 실행 가능

```
1. Perplexity로 현재 시장 상황 정리
   → 금/은/달러 동향, Fed 발언 요약

2. 기존 LASSO 결과 재해석
   → 경제학적 의미 부여

3. 크리티컬 패스 시각화
   → 네트워크 그래프 생성
```

### 9.2 단기 개발 (1-2주)

```
1. 경제학파 에이전트 구현
   - MonetaristAgent
   - KeynesianAgent
   - AustrianAgent

2. IRF 기반 토론 프로토콜

3. 시그널 포착 모듈
   - Net Buy Ratio
   - 체결강도
```

### 9.3 중기 개발 (1개월)

```
1. RWA 데이터 연동
2. 실시간 대시보드
3. 알림 시스템 (Discord/Slack)
```

---

## 10. 프롬프트 템플릿

### 10.1 Perplexity 정리용

```
TO DO:
현재 시장 상황 요약 (2025년 12월 기준)

KEYWORDS:
- Fed policy, interest rate outlook
- Gold, Silver, Copper prices
- Dollar index trend
- Inflation expectations
- Employment data

OUTPUT:
1. 주요 지표 현황 (표)
2. 시장 센티먼트 요약
3. 단기 전망 (1-3개월)
```

### 10.2 경제학적 해석용

```
TO DO:
LASSO 선택 변수의 경제학적 의미 해석

CONTEXT:
- Selected: d_Spread_HighYield (-3.58), Ret_HighYield_ETF (-1.33)
- Regime: BULL (50% confidence)
- Market: 금/은 상승, 달러 약세

QUESTIONS:
1. High Yield 스프레드 축소가 의미하는 바는?
2. 이것이 Fed 금리 결정과 어떤 관계?
3. 각 경제학파는 이를 어떻게 해석하는가?
```

### 10.3 시그널 포착용

```
TO DO:
Buy/Sell 시그널 포착 및 검증

DATA:
- Net Buy Ratio: [실시간 데이터]
- Volume: [거래량]
- Price Action: [가격 움직임]

RULES:
- Net Buy >= 5x → STRONG_BUY
- Net Buy >= 2x → MODERATE_BUY
- Volume Spike > 2σ → 확인 필요

OUTPUT:
1. 시그널 리스트
2. 신뢰도 점수
3. 추천 액션
```

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-12-25 | v0.1 | 초기 프레임워크 문서 작성 |

---

---

## 11. 현재 시스템 분석 (main.py 기반)

### 11.1 데이터 수집 개요

현재 `main.py`는 두 가지 데이터 소스를 사용한다:

```
┌─────────────────────────────────────────────────────────────┐
│                    현재 데이터 파이프라인                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DataManager (lib/data_collector.py)                        │
│  ├── collect_all(tickers_config)                            │
│  │   ├── yfinance: SPY, QQQ, TLT, GLD, VIX...              │
│  │   └── FRED: DGS10, BAA, T5YIE, DTWEXB...                 │
│  │                                                          │
│  └── 문제점:                                                 │
│      - tickers.yaml 형식 불일치                              │
│      - 중복 컬럼 처리 미흡                                   │
│      - 변수 변환 로직 분산                                   │
│                                                              │
│  UnifiedDataCollector (수동 추가)                           │
│  ├── collect_all()                                          │
│  │   ├── fetch_yahoo(): 주가, 원자재, 환율                  │
│  │   ├── fetch_fred_rates(): 금리, 스프레드                 │
│  │   └── 변환: Ret_*, d_* 접두사                            │
│  │                                                          │
│  └── 장점:                                                   │
│      - 일관된 변수 명명                                      │
│      - 스프레드 자동 계산                                    │
│      - Term_Spread, Spread_Baa 등 파생변수                  │
│                                                              │
│  CME Panel Data (plus/complete_cme_panel_history_*.csv)     │
│  ├── 종속변수: d_Exp_Rate (일별 기대금리 변화)              │
│  ├── days_to_meeting: FOMC까지 남은 일수                    │
│  └── Horizon 분류 기준                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 분석 파이프라인 현황

```
현재 실행 흐름 (run_full_pipeline):

[1/5] Config 로드
      ↓
[2/5] DataManager.collect_all()
      - 문제: 실제로 UnifiedDataCollector 사용
      - DataManager는 형식적 호출만
      ↓
[3/5] ForecastAgent (LASSO)
      - UnifiedDataCollector → market_features
      - CME Panel → d_Exp_Rate, days_to_meeting
      - Horizon별 분리: VeryShort/Short/Long
      - 결과: R² 0.68~0.76
      ↓
[4/5] MetaOrchestrator.run_with_debate()
      - AnalysisAgent: CriticalPath 분석
      - 토론 프로토콜: 합의 도출
      - 문제: opinions 구조 불일치
      ↓
[5/5] VisualizationAgent
      - HTML 대시보드 생성
      - Chart.js 시각화
```

### 11.3 현재 시스템의 한계

#### 데이터 수집 관련

| 문제 | 현상 | 영향 |
|------|------|------|
| **이중 수집 구조** | DataManager + UnifiedDataCollector 혼용 | 코드 복잡성, 유지보수 어려움 |
| **CME 의존성** | CME 패널 없으면 proxy 사용 | 예측 정확도 저하 |
| **실시간 미지원** | 배치 처리만 가능 | 시장 변화 대응 불가 |
| **이벤트 미분리** | 삼성-테슬라 같은 구조변화 미고려 | 모델 정확도 저하 |

#### 분석 관련

| 문제 | 현상 | 영향 |
|------|------|------|
| **단순 LASSO** | 선형 관계만 포착 | 비선형 패턴 누락 |
| **Treasury 제외** | simultaneity 문제로 제외 | 정보 손실 가능성 |
| **단일 종속변수** | d_Exp_Rate만 예측 | 다양한 시나리오 미탐색 |
| **IRF 미구현** | 충격반응함수 없음 | 동적 관계 미파악 |

#### 에이전트 관련

| 문제 | 현상 | 영향 |
|------|------|------|
| **학파 부재** | 경제학 관점 미분화 | 피상적 토론 |
| **증거 미약** | opinion에 숫자 근거 부족 | 신뢰도 저하 |
| **합의 알고리즘** | Rule-based만 사용 | LLM 활용 미흡 |

### 11.4 main2.py 방향성

#### 핵심 설계 원칙

```
1. 단일 데이터 소스
   - UnifiedDataCollector v2로 통합
   - 실시간 + 배치 모드 지원
   - 이벤트 태깅 시스템

2. 네트워크 기반 분석
   - Granger Causality → 네트워크 구축
   - VAR/IRF 통합
   - 동적 관계 시각화

3. 경제학파 에이전트
   - Monetarist, Keynesian, Austrian
   - 학파별 해석 프레임워크
   - 증거 기반 토론

4. 시그널 포착 모듈
   - Net Buy Ratio
   - 체결강도/거래량
   - 이상 패턴 탐지
```

#### 제안 아키텍처

```python
# main2.py 구조 (제안)

class EIMASv2:
    """
    EIMAS v2.0 - 경제학 기반 멀티에이전트 시스템
    """

    def __init__(self):
        # 1. 통합 데이터 수집기
        self.collector = UnifiedDataCollectorV2(
            mode='hybrid',  # batch + realtime
            event_tagging=True
        )

        # 2. 네트워크 분석기
        self.network = EconomicNetworkBuilder(
            method='granger+var',
            significance=0.05
        )

        # 3. 경제학파 에이전트
        self.agents = {
            'monetarist': MonetaristAgent(),
            'keynesian': KeynesianAgent(),
            'austrian': AustrianAgent(),
            'technical': TechnicalAgent()  # 시그널 포착
        }

        # 4. 토론 오케스트레이터
        self.orchestrator = EconomicDebateOrchestrator(
            use_llm=True,
            evidence_required=True
        )

    async def run(self, query: str) -> Dict:
        """
        전체 파이프라인 실행
        """
        # Phase 1: 데이터 수집
        data = await self.collector.collect()

        # Phase 2: 네트워크 구축 (IRF 포함)
        network = self.network.build(data)
        irf = self.network.calculate_irf(data)

        # Phase 3: 에이전트 분석
        opinions = []
        for name, agent in self.agents.items():
            opinion = await agent.analyze(
                data=data,
                network=network,
                irf=irf
            )
            opinions.append(opinion)

        # Phase 4: 토론 및 합의
        consensus = await self.orchestrator.debate(
            opinions=opinions,
            evidence=network.get_evidence()
        )

        # Phase 5: 시그널 포착
        signals = self.agents['technical'].detect_signals(data)

        return {
            'network': network.to_dict(),
            'irf': irf,
            'opinions': opinions,
            'consensus': consensus,
            'signals': signals,
            'recommendations': self.generate_recommendations(consensus, signals)
        }
```

### 11.5 데이터 수집 개선안

#### UnifiedDataCollectorV2 설계

```python
class UnifiedDataCollectorV2:
    """
    개선된 통합 데이터 수집기

    개선점:
    1. 이벤트 태깅 (구조변화 분리)
    2. 실시간 모드 지원
    3. 캐싱 및 증분 업데이트
    4. 데이터 품질 검증
    """

    def __init__(
        self,
        mode: str = 'batch',  # batch | realtime | hybrid
        event_tagging: bool = True,
        cache_dir: str = 'data/cache'
    ):
        self.mode = mode
        self.event_tagging = event_tagging
        self.cache_dir = Path(cache_dir)

        # 이벤트 레지스트리
        self.events = EventRegistry()

    async def collect(self) -> pd.DataFrame:
        """통합 데이터 수집"""

        # 1. 캐시 확인
        if self._cache_valid():
            return self._load_cache()

        # 2. 병렬 수집
        yahoo_data, fred_data, cme_data = await asyncio.gather(
            self._fetch_yahoo_async(),
            self._fetch_fred_async(),
            self._fetch_cme_async()
        )

        # 3. 변수 변환
        features = self._transform_variables(yahoo_data, fred_data)

        # 4. CME 병합
        if cme_data is not None:
            features = self._merge_cme(features, cme_data)

        # 5. 이벤트 태깅
        if self.event_tagging:
            features = self._tag_events(features)

        # 6. 품질 검증
        features = self._validate_quality(features)

        # 7. 캐시 저장
        self._save_cache(features)

        return features

    def _tag_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        구조변화 이벤트 태깅

        예시:
        - 삼성-테슬라 계약: 2024-XX-XX
        - Fed pivot: 2024-09-18
        - 트럼프 당선: 2024-11-05
        """
        df['event_regime'] = 'normal'

        for event in self.events.get_all():
            mask = (df.index >= event.start) & (df.index <= event.end)
            df.loc[mask, 'event_regime'] = event.name

        return df
```

#### 이벤트 레지스트리

```python
class EventRegistry:
    """
    구조변화 이벤트 관리

    용도:
    - 이벤트 전후 데이터 분리 학습
    - 이벤트 효과 분석
    - 레짐 전환 탐지
    """

    def __init__(self):
        self.events = [
            Event(
                name='fed_pivot_2024',
                start='2024-09-01',
                end='2024-10-31',
                description='Fed 금리 인하 시작',
                impact={'rate': 'dovish', 'equity': 'bullish'}
            ),
            Event(
                name='trump_election_2024',
                start='2024-11-01',
                end='2024-12-31',
                description='트럼프 당선 및 정책 기대',
                impact={'fiscal': 'expansionary', 'tariff': 'uncertain'}
            ),
            Event(
                name='samsung_tesla_contract',
                start='2024-XX-XX',  # 실제 날짜로 업데이트
                end='2024-XX-XX',
                description='삼성전자-테슬라 계약',
                impact={'samsung': 'revaluation', 'ev_sector': 'positive'}
            ),
        ]
```

### 11.6 분석 개선안

#### VAR/IRF 통합

```python
class EconomicNetworkBuilder:
    """
    경제 네트워크 구축 및 IRF 분석

    Palantir Ontology 개념 적용:
    - Node: 경제 변수
    - Semantic Edge: 이론적 관계
    - Kinetic Edge: 실증적 관계 (VAR 계수)
    """

    def build(self, data: pd.DataFrame) -> EconomicNetwork:
        """
        Granger 인과성 + VAR 기반 네트워크 구축
        """
        # 1. 변수 선택 (LASSO로 사전 필터링)
        selected_vars = self._lasso_select(data)

        # 2. Granger 인과성 테스트
        granger_matrix = self._granger_causality(data[selected_vars])

        # 3. VAR 모델 추정
        var_model = VAR(data[selected_vars]).fit(maxlags=5, ic='aic')

        # 4. 네트워크 구축
        network = EconomicNetwork()

        for i, var_i in enumerate(selected_vars):
            network.add_node(var_i, type=self._classify_var(var_i))

            for j, var_j in enumerate(selected_vars):
                if granger_matrix[i, j] < 0.05:  # 유의미한 관계
                    network.add_edge(
                        source=var_j,
                        target=var_i,
                        weight=var_model.coefs[0][i, j],  # 첫 번째 래그 계수
                        p_value=granger_matrix[i, j]
                    )

        return network

    def calculate_irf(
        self,
        data: pd.DataFrame,
        shock_var: str = 'd_Exp_Rate',
        periods: int = 20
    ) -> Dict:
        """
        충격반응함수 계산

        경제학적 해석:
        - Fed 금리 충격 → 각 변수 반응
        - 시간에 따른 반응 감쇠
        - 장기 균형으로 수렴 (M↔P만 잔존)
        """
        var_model = VAR(data).fit(maxlags=5, ic='aic')
        irf = var_model.irf(periods)

        return {
            'impulse': shock_var,
            'periods': periods,
            'responses': {
                var: irf.irfs[:, i, var_model.names.index(shock_var)].tolist()
                for i, var in enumerate(var_model.names)
            },
            'cumulative': {
                var: irf.cum_effects[:, i, var_model.names.index(shock_var)].tolist()
                for i, var in enumerate(var_model.names)
            }
        }
```

### 11.7 에이전트 개선안

#### 경제학파 에이전트 구현

```python
class MonetaristAgent(BaseAgent):
    """
    통화주의 관점 에이전트

    핵심 명제:
    - M↔P 장기 관계가 가장 중요
    - 통화 중립성: 실질 변수에 장기 영향 없음
    - 인플레이션은 항상 화폐적 현상
    """

    async def analyze(self, data, network, irf) -> AgentOpinion:
        # 1. M2, CPI 관계 확인
        m_p_relation = network.get_edge_weight('M2', 'CPI')

        # 2. IRF에서 통화 충격 분석
        monetary_shock_effect = irf['responses'].get('CPI', [])

        # 3. 의견 형성
        if m_p_relation > 0.5:
            position = "INFLATIONARY"
            reasoning = "M2 증가가 물가 상승으로 이어질 것"
        else:
            position = "NEUTRAL"
            reasoning = "통화-물가 연결고리 약화"

        return AgentOpinion(
            agent_role='monetarist',
            topic='inflation_outlook',
            position=position,
            confidence=abs(m_p_relation),
            evidence=[
                f"M2-CPI 관계: {m_p_relation:.3f}",
                f"통화충격 누적효과: {sum(monetary_shock_effect):.3f}"
            ],
            economic_framework="MV=PY, 장기 M↔P"
        )


class KeynesianAgent(BaseAgent):
    """
    케인즈주의 관점 에이전트

    핵심 명제:
    - 총수요(C+I+G+NX) 관리가 핵심
    - 승수효과: 지출 → 소득 → 지출 순환
    - 유동성 함정 가능성
    """

    async def analyze(self, data, network, irf) -> AgentOpinion:
        # 1. 소비, 투자 동향 확인
        consumption_trend = self._analyze_trend(data, 'Ret_Consumer')
        investment_trend = self._analyze_trend(data, 'Ret_Industrial')

        # 2. 승수 효과 추정
        multiplier = network.get_edge_weight('Gov_Spending', 'GDP')

        # 3. 의견 형성
        if consumption_trend < 0 and investment_trend < 0:
            position = "STIMULATE"
            reasoning = "총수요 부족, 재정 확대 필요"
        else:
            position = "MAINTAIN"
            reasoning = "총수요 안정, 현 정책 유지"

        return AgentOpinion(
            agent_role='keynesian',
            topic='fiscal_policy',
            position=position,
            confidence=0.7,
            evidence=[
                f"소비 트렌드: {consumption_trend:.2%}",
                f"투자 트렌드: {investment_trend:.2%}",
                f"재정 승수: {multiplier:.2f}"
            ],
            economic_framework="Y = C + I + G + NX"
        )


class AustrianAgent(BaseAgent):
    """
    오스트리아 학파 관점 에이전트

    핵심 명제:
    - 인위적 저금리 → 자본 오배분 → 버블
    - 경기 사이클은 신용 팽창의 결과
    - 청산을 통한 자연적 조정 필요
    """

    async def analyze(self, data, network, irf) -> AgentOpinion:
        # 1. 신용 사이클 분석
        credit_expansion = self._analyze_credit(data)

        # 2. 자산 버블 징후
        bubble_indicators = self._detect_bubble(data)

        # 3. 금/은 가격 (실물 자산 선호)
        gold_silver_trend = self._precious_metals_trend(data)

        # 4. 의견 형성
        if bubble_indicators['score'] > 0.7:
            position = "BUBBLE_WARNING"
            reasoning = "인위적 저금리로 자산 버블 형성 중"
        elif gold_silver_trend > 0:
            position = "INFLATION_HEDGE"
            reasoning = "실물 자산 선호 증가, fiat 불신"
        else:
            position = "CAUTIOUS"
            reasoning = "사이클 후반, 조정 대비 필요"

        return AgentOpinion(
            agent_role='austrian',
            topic='cycle_analysis',
            position=position,
            confidence=bubble_indicators['confidence'],
            evidence=[
                f"신용 팽창률: {credit_expansion:.2%}",
                f"버블 점수: {bubble_indicators['score']:.2f}",
                f"금/은 트렌드: {gold_silver_trend:.2%}"
            ],
            economic_framework="Austrian Business Cycle Theory"
        )
```

### 11.8 main2.py 구현 로드맵

```
Phase 1: 데이터 기반 (1주)
├── [ ] UnifiedDataCollectorV2 구현
├── [ ] EventRegistry 설정
├── [ ] 캐싱 시스템
└── [ ] 품질 검증 로직

Phase 2: 네트워크 분석 (1주)
├── [ ] EconomicNetworkBuilder 구현
├── [ ] Granger Causality 테스트
├── [ ] VAR 모델 추정
└── [ ] IRF 계산 및 시각화

Phase 3: 에이전트 고도화 (1주)
├── [ ] MonetaristAgent 구현
├── [ ] KeynesianAgent 구현
├── [ ] AustrianAgent 구현
└── [ ] TechnicalAgent (시그널 포착)

Phase 4: 토론 및 통합 (1주)
├── [ ] EconomicDebateOrchestrator
├── [ ] LLM 기반 합의 도출
├── [ ] 대시보드 v2
└── [ ] API 서버 (FastAPI)
```

---

## 참고 문헌

- Bekaert, G., Hoerova, M. (2014). "The VIX, the variance premium and stock market volatility"
- Palantir Foundry Ontology Documentation
- Federal Reserve Economic Data (FRED)
- VAR/IRF: Hamilton (1994) "Time Series Analysis"
- Austrian Business Cycle Theory: Mises, Hayek
- Keynesian Economics: Keynes (1936) "General Theory"
- Monetarism: Friedman (1968) "The Role of Monetary Policy"
