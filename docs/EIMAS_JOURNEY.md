# EIMAS: 프로젝트 여정과 기술적 의사결정

> 문제 정의부터 실패, 학습, 개선까지의 전체 기록

---

## 1. 문제 정의: 왜 EIMAS를 만들었는가?

### 1.1 기존 시스템의 한계

**개인 투자자의 현실:**
- 거시경제 데이터(FRED, Fed 정책)와 시장 데이터(가격, 거래량)가 분리됨
- 정보는 넘쳐나지만 **인과관계 파악**이 어려움
- 기관 리서치(JP Morgan, Goldman Sachs)의 방법론은 공개되지만 **실행 가능한 시스템**이 없음

**기존 도구의 문제:**
| 도구 | 한계 |
|------|------|
| TradingView | 기술적 분석 중심, 거시경제 통합 부재 |
| Bloomberg Terminal | 고가, 개인 접근 어려움 |
| Python 스크립트 | 파편화, 유지보수 어려움 |
| ChatGPT/Claude | 일회성 분석, 시스템화 안 됨 |

### 1.2 목표 설정

**핵심 질문:**
> "오늘 시장에 들어가야 하는가? 얼마나 확신할 수 있는가?"

**구체적 목표:**
1. **데이터 통합**: FRED + 시장 + 크립토 + 대체자산을 하나의 파이프라인으로
2. **인과관계 추론**: 단순 상관관계가 아닌 Granger Causality 기반 분석
3. **레짐 탐지**: Bull/Bear/Neutral 상태를 정량적으로 판단
4. **리스크 계층화**: 유동성 → 버블 → 미세구조 다단계 리스크
5. **AI 합의**: 단일 모델이 아닌 다중 에이전트 토론으로 편향 감소

---

## 2. 접근 방식: 기관 방법론의 개인화

### 2.1 벤치마크: 기관 투자자들은 어떻게 하는가?

**docx2 레퍼런스 분석 결과:**

| 기관 | 핵심 방법론 | EIMAS 적용 |
|------|------------|-----------|
| **JP Morgan** | 5단계 버블 평가, K-shaped 소비 모델 | BubbleDetector, RegimeAnalyzer |
| **Goldman Sachs** | 상품 통제 사이클, 지정학적 제약 | ShockPropagationGraph, CausalityGraph |
| **Berkshire** | 보수적 회계, 내재가치 프레임워크 | Risk Enhancement Layer |

**공통 패턴 발견:**
1. **정량적 검증**: 정성적 서사를 반드시 데이터로 뒷받침
2. **구조적 vs 순환적 분리**: 일시적 충격과 영구적 레짐 변화 구분
3. **가치사슬 매핑**: 어디서 가치가 축적되는지 추적
4. **리스크 프리미엄 정량화**: 불확실성을 숫자로 변환

### 2.2 EIMAS 아키텍처 설계

```
┌─────────────────────────────────────────────────────────────────┐
│                     EIMAS Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: DATA COLLECTION                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │  FRED   │ │ yfinance│ │ Crypto  │ │   RWA   │              │
│  │(매크로) │ │ (시장)  │ │(BTC/ETH)│ │(ONDO등) │              │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│       └──────────┬┴──────────┬┴───────────┘                    │
│                  ▼                                              │
│  Phase 2: ANALYSIS                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RegimeDetector → CriticalPath → Microstructure         │   │
│  │       ↓              ↓               ↓                   │   │
│  │  GMM 3-State    Granger Causal   Amihud/VPIN            │   │
│  │       ↓              ↓               ↓                   │   │
│  │  BubbleDetector ← ShockPropagation ← Risk Enhancement   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                  ▼                                              │
│  Phase 3: MULTI-AGENT DEBATE                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AnalysisAgent ←→ ForecastAgent ←→ StrategyAgent        │   │
│  │       ↓                                                  │   │
│  │  MetaOrchestrator → Consensus → Final Recommendation    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                  ▼                                              │
│  Phase 4-7: OUTPUT                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │   DB    │ │  JSON   │ │   MD    │ │  HTML   │              │
│  │ Storage │ │ Export  │ │ Report  │ │Dashboard│              │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 핵심 기술 선택과 Trade-off

| 선택 | 대안 | 선택 이유 | Trade-off |
|------|------|----------|-----------|
| **LASSO** | Ridge, ElasticNet | Sparsity로 핵심 변수만 선택 | 상관된 변수 중 하나만 선택됨 |
| **GMM 3-State** | HMM, K-means | 확률적 레짐 분류, 불확실성 정량화 | 초기값 민감, 수렴 불안정 |
| **Granger Causality** | VAR, SVAR | 시계열 인과관계의 표준 | 선형 관계만 포착 |
| **Shannon Entropy** | Gini, Variance | 정보이론 기반 불확실성 | 해석 직관성 낮음 |
| **SQLite** | PostgreSQL, MongoDB | 단일 파일, 배포 용이 | 동시성 제한 |
| **Claude API** | GPT-4, Gemini | 긴 컨텍스트, 추론 품질 | 비용, 속도 |

---

## 3. 실패한 시도와 학습

### 3.1 백테스팅 엔진의 복리 버그

**문제 발견 (2026-01-31):**
```
EIMAS_Regime 전략 백테스트 결과:
- 수익률: +8,360% (5년)
- Sharpe: 1.85
- 승률: 39%

→ 39% 승률로 8,360% 수익은 통계적으로 불가능
```

**원인 분석:**
```python
# 버그 코드 (Short Entry)
capital -= commission  # 원금 차감 누락!

# 버그 코드 (Exit)
capital += shares * entry_price + pnl  # 원금 + 손익 추가

# 결과: Short 거래마다 $30,000가 "무에서 생성"
```

**수정 후 결과:**
| 지표 | 버그 | 수정 후 | 변화 |
|------|------|--------|------|
| 총 수익률 | +8,360% | +1.14% | -8,359pp |
| Sharpe | 1.85 | 0.19 | -1.66 |
| Alpha | +8,265% | -26.5% | 역전 |

**학습:**
- 백테스트 결과가 "너무 좋으면" 버그를 의심할 것
- 자본 관리 로직은 Long/Short 대칭성 검증 필수
- Position sizing 모드 분리 (FIXED vs PERCENTAGE)

### 3.2 Walk-Forward Validation 실패

**설정:**
- 기간: 2022-01-01 ~ 2025-12-31 (4년)
- Train: 12개월, Test: 3개월
- 총 12 Folds

**결과:**
| Fold | Train Period | IS Sharpe | OOS Sharpe | Degradation |
|------|-------------|-----------|------------|-------------|
| 6 | 2023-03 ~ 2024-03 | 2.10 | 4.09 | -94% (Good) |
| 9 | 2023-12 ~ 2024-12 | 0.23 | -2.07 | +1007% (Bad) |
| **평균** | - | **0.18** | **0.40** | **+69%** |

**진단:**
- Avg OOS Sharpe 0.40 < 기준 0.5 → **FAIL**
- Degradation 69% > 기준 30% → **과적합**
- 2023-2024 상승장에서만 작동, 다른 레짐에서 붕괴

**학습:**
- In-Sample 성과는 무의미, OOS만 신뢰
- 레짐 의존적 전략은 레짐 필터링 필요
- 단일 전략 < 앙상블

### 3.3 API 방법론 검증에서 발견된 문제

**Stablecoin Risk 평가 (2026-01-09):**

| 평가 | Claude | Perplexity |
|------|--------|------------|
| 기본 순서 (USDC < USDT < DAI < USDe) | 적절 | 적절 |
| 이자 페널티 +15점 | 과도하게 단순화 | 세분화 필요 |
| 누락된 요소 | 유동성, 거버넌스, 기술 | 은행급 프레임워크 |

**수정:** 다차원 리스크 평가로 전환
```python
WEIGHTS = {
    'credit': 0.30,      # 신용/담보
    'liquidity': 0.25,   # 유동성
    'regulatory': 0.25,  # 규제 (이자 차등)
    'technical': 0.20    # 스마트컨트랙트
}
```

**MST Systemic Risk (2026-01-09):**

| 평가 | Claude | Perplexity |
|------|--------|------------|
| 거리 공식 √(2(1-ρ)) | 학술적 정확 | Mantegna 1999 정석 |
| Eigenvector Centrality | 트리에서 비효율 | 제거 권장 |

**수정:** Eigenvector 제거, 중심성 가중치 조정
```python
# v1: Eigenvector 포함 (비효율)
# v2: 제거
CENTRALITY_WEIGHTS = {
    'betweenness': 0.45,
    'degree': 0.35,
    'closeness': 0.20,
}
```

---

## 4. 기술적 의사결정 상세

### 4.1 Risk Enhancement Layer 선택

**3가지 옵션 검토:**

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| A: Sequential | Phase 2.2 → Micro → Bubble | 단순 | 실행시간 증가 |
| B: Parallel | 병렬 실행 | 빠름 | 동기화 복잡 |
| **C: Layer** | CriticalPath 후 조정 | 통합 용이 | 추가 Phase |

**선택: Option C**

**이유:**
- JP Morgan/Goldman 방식: 기본 리스크 → 추가 레이어 적층
- 공식: `Final Risk = Base + Micro Adj(±10) + Bubble Adj(+0~15)`

### 4.2 DB 스키마 설계

**초기 설계 (v1):**
```sql
signals, portfolio_candidates, executions,
performance_tracking, signal_performance, session_analysis
```

**확장 (v2, 2026-01-31):**
```sql
+ backtest_runs           -- 백테스트 실행 결과
+ backtest_trades         -- 개별 거래 기록
+ walk_forward_results    -- OOS 검증 결과
```

**설계 원칙:**
- 모든 결과는 재현 가능해야 함 (파라미터 저장)
- 시계열 추적 가능해야 함 (timestamp 필수)
- JSON 필드로 유연성 확보 (metadata, parameters)

### 4.3 Multi-Agent Debate 설계

**단일 LLM vs Multi-Agent:**

| 접근 | 장점 | 단점 |
|------|------|------|
| 단일 LLM | 빠름, 저비용 | 편향, 환각 |
| **Multi-Agent** | 다양한 관점, 교차 검증 | 비용, 복잡성 |

**EIMAS 토론 프로토콜:**
1. 각 Agent가 독립적으로 분석 (AnalysisAgent, ForecastAgent)
2. MetaOrchestrator가 의견 수집
3. 충돌 시 Rule-based 조정 (85% 일관성 임계값)
4. 합의 도출 또는 불일치 기록

---

## 5. 기관 방법론과의 Gap 분석

### 5.1 현재 구현 vs 기관 수준

| 영역 | 기관 (JP Morgan, GS) | EIMAS 현재 | Gap |
|------|---------------------|-----------|-----|
| **데이터** | 독점 데이터, 대안 데이터 | FRED, yfinance, 공개 API | 대안 데이터 부재 |
| **모델** | 전담 퀀트팀, 검증된 모델 | 학술 논문 기반 | 실전 검증 부족 |
| **백테스트** | Walk-Forward, Monte Carlo, Stress Test | Walk-Forward 구현 | Monte Carlo 미구현 |
| **실행** | 알고 트레이딩, 최적 집행 | Paper Trading만 | 실거래 미연동 |
| **리스크** | VaR, CVaR, Stress Test | CriticalPath + Micro + Bubble | Tail Risk 미흡 |

### 5.2 채워야 할 Gap (우선순위)

**P0 (Critical):**
1. ~~백테스트 버그 수정~~ ✅ 완료
2. ~~Walk-Forward Validation~~ ✅ 완료
3. Monte Carlo Simulation 추가

**P1 (High):**
4. Tail Risk (VaR, CVaR) 모듈
5. 앙상블 전략 (Multi-Strategy)
6. 실거래 연동 (Alpaca, IBKR)

**P2 (Medium):**
7. 대안 데이터 (위성, 신용카드, 소셜)
8. NLP 뉴스 감성 분석
9. 옵션 내재 변동성 분석

---

## 6. 개선 로드맵

### 6.1 단기 (1-2주)

| 작업 | 목표 | 예상 효과 |
|------|------|----------|
| Monte Carlo Simulation | 1,000회 시뮬레이션 | 수익률 분포 파악 |
| 전략 단순화 | 파라미터 4→2개 | 과적합 감소 |
| 레짐 필터 | Bear 시 현금 보유 | MDD 감소 |

### 6.2 중기 (1-2개월)

| 작업 | 목표 | 예상 효과 |
|------|------|----------|
| VaR/CVaR 모듈 | 일일 리스크 한도 | Tail Risk 관리 |
| 앙상블 전략 | 3개 전략 결합 | Sharpe 안정화 |
| Alpaca 연동 | Paper → Live | 실전 검증 |

### 6.3 장기 (3-6개월)

| 작업 | 목표 | 예상 효과 |
|------|------|----------|
| 대안 데이터 | 위성/신용카드 | 정보 우위 |
| 옵션 분석 | IV Surface | 시장 기대 파악 |
| 자동 리밸런싱 | 월간 자동 조정 | 운영 효율 |

---

## 7. 핵심 교훈

### 7.1 기술적 교훈

1. **"너무 좋은 결과"는 버그다**
   - 8,360% 수익률 → Short 버그
   - 항상 sanity check 수행

2. **In-Sample은 무의미하다**
   - OOS Sharpe만 신뢰
   - Walk-Forward는 필수

3. **단순함이 견고함이다**
   - 파라미터 많을수록 과적합
   - 전략 단순화가 성과 개선

### 7.2 프로세스 교훈

1. **기관 방법론을 참고하되 맹신하지 말 것**
   - 공개 자료는 "무엇을"만 알려줌
   - "어떻게"는 직접 구현하며 학습

2. **실패를 문서화할 것**
   - 같은 실수 반복 방지
   - 의사결정 근거 추적

3. **점진적 검증**
   - 작은 단위로 테스트
   - 통합 전 개별 모듈 검증

---

## 8. 결론

EIMAS는 "개인 투자자를 위한 기관급 분석 시스템"을 목표로 시작했다. 현재까지:

**달성:**
- 7개 Phase 통합 파이프라인
- 10+ 경제학적 방법론 구현
- Multi-Agent 토론 시스템
- 실시간 대시보드

**실패 후 학습:**
- 백테스트 복리 버그 → 고정 포지션 사이징
- 과적합 → Walk-Forward 필수화
- 단일 전략 한계 → 앙상블 필요

**남은 과제:**
- Monte Carlo, VaR/CVaR
- 실거래 연동
- 대안 데이터 통합

> "시스템은 완성되지 않는다. 계속 진화할 뿐이다."

---

*작성일: 2026-01-31*
*버전: v1.0*
