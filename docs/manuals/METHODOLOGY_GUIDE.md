# Fed Rate Forecasting - Methodology Guide

> 경제 AI 에이전트 개발을 위한 방법론 참고 문서
> "무엇을 했는지"보다 "왜 그렇게 했는지"에 초점

---

## 1. 프로젝트 목표

**CME FedWatch 데이터를 활용하여 시장의 Fed 금리 기대 변화를 설명하는 변수 식별**

- 종속변수: `d_Exp_Rate` (기대금리 일별 변화, bp)
- 핵심 질문: "어떤 시장 변수가 금리 기대 변화를 설명하는가?"

---

## 2. 핵심 방법론과 선택 이유

### 2.1 LASSO 회귀 (L1 정규화)

#### 왜 LASSO인가?

| 문제 상황 | LASSO가 해결하는 방법 |
|----------|---------------------|
| **변수가 너무 많음** (50개+) | Sparsity - 대부분 계수를 0으로 축소하여 핵심 변수만 선택 |
| **다중공선성** | 상관관계 높은 변수 중 하나만 선택 |
| **과적합 방지** | 정규화로 모델 복잡도 제어 |
| **해석 용이성** | 선택된 변수만 보면 됨 (vs Ridge의 모든 변수 포함) |

```python
# LASSO: 변수 선택용 (p-value 없음)
model = LassoCV(cv=TimeSeriesSplit(n_splits=5))
```

#### 왜 Ridge가 아닌가?

- Ridge(L2)는 계수를 0에 가깝게 축소하지만 완전히 0으로 만들지 않음
- 변수 선택이 아닌 예측 정확도가 목적일 때는 Ridge가 적합
- 본 연구는 **"어떤 변수가 중요한가"**가 목적 → LASSO 선택

---

### 2.2 Treasury 변수 제외

#### 왜 제외하는가? - Simultaneity (동시결정) 문제

```
문제: Treasury 금리 ↔ Fed 금리 기대
      (서로가 서로에게 영향)
```

**경제학적 설명**:
- Fed 금리 기대가 변하면 → Treasury 금리도 변함
- Treasury 금리 변화를 보면 → Fed 금리 기대 변화를 알 수 있음
- 즉, **인과관계가 아닌 동시결정 관계**

**인과 추론의 원칙**:
- X → Y 를 분석하려면 X가 외생적(exogenous)이어야 함
- Treasury 금리는 Y(기대금리)와 동시결정 → 내생적(endogenous)
- 포함하면 **인과관계를 오해석**할 위험

```python
# 제외 변수 패턴
EXCLUDE_PATTERNS = [
    'Treasury',      # Ret_Treasury_*, d_Treasury_*
    'US2Y', 'US10Y', # 국채금리
    'RealYield',     # 실질금리 (TIPS)
    'Term_Spread'    # 장단기 스프레드
]
```

---

### 2.3 Horizon 분리 (초단기/단기/장기)

#### 왜 기간을 분리하는가?

| Horizon | 기간 | 특성 | 주요 영향 변수 |
|---------|------|------|---------------|
| **초단기** | ≤30일 | 뉴스/이벤트 민감 | VIX, 이벤트 더미 |
| **단기** | 31-90일 | 중간 | 달러, 크레딧 스프레드 |
| **장기** | ≥180일 | 펀더멘털 중심 | 인플레 기대, 거시 지표 |

**경제학적 배경**:
- 단기 기대는 노이즈에 민감 (공포, 뉴스)
- 장기 기대는 펀더멘털에 수렴 (인플레, 성장)
- **동일 모델로 모든 기간을 설명하면 해석 오류 발생**

```python
# Horizon 분리
very_short_term = df[df['days_to_meeting'] <= 30]  # 공포 지표 중요
short_term = df[(df['days_to_meeting'] >= 31) & (df['days_to_meeting'] <= 90)]
long_term = df[df['days_to_meeting'] >= 180]  # 펀더멘털 중요
```

---

### 2.4 Post-LASSO OLS with HAC

#### 왜 2단계 접근인가?

```
1단계: LASSO → 변수 선택 (p-value 없음)
2단계: OLS → 통계적 추론 (p-value, 신뢰구간)
```

**LASSO의 한계**:
- 정규화 때문에 계수가 축소됨
- p-value를 제공하지 않음
- 신뢰구간 계산이 어려움

**Post-LASSO OLS**:
- LASSO가 선택한 변수만으로 OLS 수행
- 정규화 없이 실제 관계 추정
- 통계적 유의성 검정 가능

#### 왜 HAC (Newey-West) 표준오차인가?

| 문제 | HAC가 해결하는 방법 |
|------|---------------------|
| **자기상관** (Autocorrelation) | lag 5까지의 상관 보정 |
| **이분산** (Heteroskedasticity) | 분산 변화 보정 |

**시계열 데이터의 특성**:
- 오늘 오차와 내일 오차는 상관됨 (자기상관)
- 변동성이 큰 시기와 작은 시기가 있음 (이분산)
- 일반 OLS는 이를 무시 → **표준오차 과소추정** → p-value 왜곡

```python
# HAC (Newey-West) 표준오차
model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
```

---

### 2.5 Time-Series CV (시계열 교차검증)

#### 왜 일반 K-Fold가 아닌가?

```
일반 K-Fold:  [1][2][3][4][5] → 랜덤 분할
              과거 데이터로 미래 예측 가능 (Data Leakage)

Time-Series CV: [Train][Test]
               [Train    ][Test]
               [Train        ][Test]
               항상 과거로 미래 예측 (No Leakage)
```

**시계열 데이터의 특성**:
- 시간 순서가 중요
- 미래 데이터로 과거를 예측하면 안 됨
- 일반 CV는 **미래 정보 유출 (Look-ahead Bias)**

```python
# Time-Series Split
tscv = TimeSeriesSplit(n_splits=5)
model = LassoCV(cv=tscv)
```

---

### 2.6 Selection Frequency Analysis

#### 왜 Rolling Window로 빈도를 계산하는가?

**목적**: Post-Selection Inference Robustness

| 문제 | 해결 방법 |
|------|----------|
| LASSO 선택이 **불안정**할 수 있음 | 여러 window에서 선택 빈도 확인 |
| 특정 기간에만 유의한 변수 | 빈도 낮음 → 신뢰도 낮음 |
| 일관되게 선택되는 변수 | 빈도 높음 → 신뢰도 높음 |

**해석 기준**:
- Selection Frequency > 50%: 안정적으로 중요한 변수
- Sign Consistency > 80%: 부호가 일관됨 (해석 신뢰)

```python
# Rolling window에서 선택 빈도 계산
for start in range(0, len(data) - window_size, step):
    window = data[start:start+window_size]
    lasso.fit(window)
    # 선택 여부 기록
```

---

### 2.7 ADF 검정 (단위근 검정)

#### 왜 수행하는가?

**문제**: 비정상(non-stationary) 시계열로 회귀하면 **허구적 회귀**

```
비정상 시계열: 평균/분산이 시간에 따라 변함
              → 관계 없는 두 변수도 유의하게 나옴 (Spurious Regression)

정상 시계열: 평균/분산이 일정
            → 진짜 관계만 유의함
```

**해결책**:
- 수준(level) 변수 → 차분(difference)으로 변환
- `d_Exp_Rate` = `exp_rate_bp(t)` - `exp_rate_bp(t-1)`
- 차분하면 대부분 정상 시계열이 됨

```python
# ADF 검정
adf_result = adfuller(df['d_Exp_Rate'])
# p-value < 0.05 → 정상(stationary)
```

---

## 3. 데이터 처리 파이프라인

### 3.1 데이터 구조

```
CME Panel Data (FedWatch)
├── meeting_date: FOMC 회의일
├── asof_date: 관측일
├── exp_rate_bp: 기대금리 (bp)
├── days_to_meeting: 회의까지 남은 일수
└── rate_uncertainty: 불확실성 (분산 기반)

Market Data
├── Ret_* : 일간 로그 수익률 (%)
├── d_* : 일간 차분 (level 변화)
└── *_Released : 이벤트 더미 (0/1)
```

### 3.2 변수 변환 이유

| 변환 | 원본 | 결과 | 이유 |
|------|------|------|------|
| **로그 수익률** | 가격 | `Ret_` | 정상성, 해석 용이 |
| **차분** | 수준 | `d_` | 비정상 → 정상 |
| **이벤트 더미** | 발표일 | `_Released` | 이벤트 효과 분리 |

---

## 4. 주요 변수 카테고리와 경제학적 의미

### 4.1 Credit 관련

| 변수 | 의미 | Fed 금리와의 관계 |
|------|------|------------------|
| `d_Spread_Baa` | 신용 스프레드 변화 | 스프레드 ↑ → 금융 스트레스 ↑ → Fed 완화 기대 ↑ |
| `Ret_HighYield_ETF` | 하이일드 채권 수익률 | HY ↓ → 리스크 오프 → Fed 완화 기대 ↑ |

### 4.2 FX 관련

| 변수 | 의미 | Fed 금리와의 관계 |
|------|------|------------------|
| `Ret_Dollar_Idx` | 달러 인덱스 수익률 | 달러 ↑ → 긴축 기대 ↑ |
| `Ret_USDKRW` | 원/달러 환율 | 신흥국 리스크 프록시 |

### 4.3 Equity/Risk 관련

| 변수 | 의미 | Fed 금리와의 관계 |
|------|------|------------------|
| `Ret_SP500` | 주식시장 수익률 | 주식 ↓ → Fed Put 기대 → 완화 기대 ↑ |
| `d_VIX` | 변동성 지수 변화 | VIX ↑ → 공포 ↑ → 단기적 완화 기대 ↑ |

### 4.4 Inflation 관련

| 변수 | 의미 | Fed 금리와의 관계 |
|------|------|------------------|
| `d_Breakeven5Y` | 5년 기대인플레 변화 | BEI ↑ → 인플레 기대 ↑ → 긴축 기대 ↑ |
| `CPI_Released` | CPI 발표일 더미 | 이벤트 효과 |

---

## 5. 분석 결과 해석 가이드

### 5.1 LASSO 계수 해석

```
양의 계수 (+): X ↑ → Fed 금리 기대 ↑ (긴축 방향)
음의 계수 (-): X ↑ → Fed 금리 기대 ↓ (완화 방향)
```

**주의**: 표준화된 계수이므로 단위가 다른 변수 간 비교 가능

### 5.2 Horizon별 해석

| 결과 패턴 | 해석 |
|----------|------|
| 초단기만 유의 | 단기 노이즈/공포 지표 |
| 장기만 유의 | 펀더멘털 변수 |
| 전 구간 유의 | 핵심 설명 변수 |
| 부호 반전 | 기간에 따라 관계가 다름 (구조적 변화 가능성) |

### 5.3 부호 불일치 대응

```
문제: Selection Frequency에서 음수, HAC에서 양수

가능한 원인:
1. 데이터 기간 차이 (전체 vs 최근)
2. 표준화 기준 차이 (window별 vs 전체)
3. 변수 집합 차이 (omitted variable bias)

대응:
- 두 결과를 모두 보고
- 시간에 따른 관계 변화로 해석
```

---

## 6. 새 에이전트 개발 시 적용 가이드

### 6.1 데이터 수집 단계

```python
# CME FedWatch 데이터
panel = collect_fedwatch_panel()

# 시장 데이터 (Treasury 제외!)
market = collect_market_data(exclude=['Treasury', 'US2Y', 'US10Y'])

# 병합
df = panel.merge(market, on='date')
```

### 6.2 전처리 단계

```python
# 종속변수: 차분
df['d_Exp_Rate'] = df.groupby('meeting_date')['exp_rate_bp'].diff()

# 이상치 제거 (±3σ)
df = remove_outliers(df, 'd_Exp_Rate', n_std=3)

# Horizon 분리
very_short = df[df['days_to_meeting'] <= 30]
short = df[(df['days_to_meeting'] > 30) & (df['days_to_meeting'] <= 90)]
long = df[df['days_to_meeting'] >= 180]
```

### 6.3 분석 단계

```python
# 1단계: LASSO로 변수 선택
lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5))
lasso.fit(X_scaled, y)
selected_vars = X.columns[lasso.coef_ != 0]

# 2단계: Post-LASSO OLS with HAC
ols = sm.OLS(y, X[selected_vars]).fit(
    cov_type='HAC',
    cov_kwds={'maxlags': 5}
)

# 3단계: Robustness Check
selection_freq = compute_rolling_selection_frequency(data, X_vars)
```

---

## 7. 파일 참조

| 파일 | 용도 |
|------|------|
| `forecasting_20251218.py` | 메인 분석 코드 (Part 1: 데이터~LASSO~HAC) |
| `forecasting_20251218_post_ppt.py` | Part 2: Selection Frequency, Robustness |
| `data_pipeline.py` | CME 패널 데이터 생성 |
| `actual_fed_rates.py` | 실제 FOMC 결정 금리 데이터 |
| `collect_macro_finance.py` | 시장 데이터 수집 |

---

## 8. 핵심 요약

### 왜 LASSO?
→ 많은 변수 중 핵심만 선택 (Sparsity)

### 왜 Treasury 제외?
→ 동시결정 문제 (Simultaneity) - 인과 추론 불가

### 왜 Horizon 분리?
→ 단기는 노이즈, 장기는 펀더멘털 - 다른 변수가 중요

### 왜 Post-LASSO OLS?
→ LASSO는 선택용, OLS는 추론용 (p-value)

### 왜 HAC 표준오차?
→ 시계열의 자기상관/이분산 보정 - 올바른 p-value

### 왜 Selection Frequency?
→ LASSO 선택의 안정성 검증 - Robustness

---

*문서 작성일: 2025-12-26*
