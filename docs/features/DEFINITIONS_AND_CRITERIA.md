# EIMAS 정의, 가정, 판정 기준

> Economic Intelligence Multi-Agent System
> Version: v2.1.2
> Date: 2026-01-12

---

## 1. 핵심 개념 정의

### 1.1 시장 레짐 (Market Regime)

**정의**: 시장의 현재 상태를 Bull/Neutral/Bear 3개 범주로 분류한 것

**판정 기준** (Rule-based RegimeDetector):
```
Bull Regime:
  - SPY 20일 이동평균 > SPY 50일 이동평균
  - SPY 현재가 > SPY 20일 이동평균
  - VIX < 20

Bear Regime:
  - SPY 20일 이동평균 < SPY 50일 이동평균
  - SPY 현재가 < SPY 50일 이동평균
  - VIX > 30

Neutral Regime:
  - 위 두 조건에 해당하지 않는 경우
```

**변동성 분류**:
- Low Vol: VIX < 15
- Medium Vol: 15 ≤ VIX < 25
- High Vol: VIX ≥ 25

**가정**:
1. SPY는 미국 주식시장 전체를 대표한다
2. VIX는 시장 불확실성의 대리변수다
3. 이동평균 교차는 추세 전환의 신호다

### 1.2 GMM 레짐 (Gaussian Mixture Model Regime)

**정의**: 수익률 분포를 3개 가우시안으로 분해하여 확률적으로 레짐을 분류

**모델 구조** (Hamilton 1989):
```python
Returns ~ Σ(w_i · N(μ_i, σ_i²))  # i = Bull, Neutral, Bear
```

**판정 기준**:
- Bull: μ > 0.0005, σ < 0.015
- Neutral: -0.0002 < μ < 0.0005
- Bear: μ < -0.0002, σ > 0.02

**현재 레짐**: 가장 높은 확률을 가진 상태

**불확실성 측정** (Shannon Entropy):
```
H = -Σ p_i · log_2(p_i)
H = 0: 완전 확신 (한 상태가 100%)
H = 1.58: 최대 불확실성 (균등 분포)
```

**가정**:
1. 수익률은 여러 숨은 상태(hidden states)의 혼합으로 설명된다
2. 각 상태는 정규분포를 따른다
3. 상태 간 전이는 마코프 과정이다

### 1.3 리스크 점수 (Risk Score)

**정의**: 시장의 종합 리스크를 0-100 스케일로 정량화한 지표

**계산식** (v2.1.1 Risk Enhancement Layer):
```
Final Risk Score = Base Risk + Microstructure Adj + Bubble Adj

Base Risk (0-100):
  - CriticalPathAggregator 출력
  - Bekaert VIX 분해 (Uncertainty + Risk Appetite)
  - 유동성 리스크
  - 크립토 리스크

Microstructure Adjustment (±10):
  = (50 - avg_liquidity_score) / 5
  = (50 - 82.2) / 5 = -6.4  (예시: 유동성 우수)

Bubble Risk Adjustment (0-15):
  - NONE: +0
  - WATCH: +5
  - WARNING: +10
  - DANGER: +15
```

**리스크 레벨 분류**:
- LOW: Risk < 30
- MEDIUM: 30 ≤ Risk < 60
- HIGH: Risk ≥ 60

**가정**:
1. 리스크는 여러 독립적인 요소들의 합으로 표현된다
2. 시장 미세구조 품질은 리스크를 조정한다
3. 버블 리스크는 가산적으로 작용한다

---

## 2. 시그널 정의

### 2.1 투자 권고 시그널 (Final Recommendation)

**정의**: AI 멀티에이전트 토론 결과로 도출된 최종 투자 방향

**가능한 값**:
- BULLISH: 매수 또는 롱 포지션 유지
- BEARISH: 매도 또는 숏 포지션 고려
- NEUTRAL/HOLD: 현금 또는 중립 포지션 유지

**판정 프로세스**:
```
1. FULL Mode (365일 데이터) → Position_FULL + Confidence_FULL
2. REF Mode (90일 데이터) → Position_REF + Confidence_REF
3. DualModeAnalyzer:
   - If Position_FULL == Position_REF → Final = FULL Position
   - If Position_FULL ≠ Position_REF → Final = NEUTRAL (불일치)
   - Final Confidence = (Confidence_FULL + Confidence_REF) / 2
```

**신뢰도 (Confidence)**:
- 0-100% 스케일
- > 70%: High Confidence
- 50-70%: Medium Confidence
- < 50%: Low Confidence

**가정**:
1. 장기(365일)와 단기(90일) 관점이 일치하면 신호가 강하다
2. 불일치 시 보수적으로 NEUTRAL 선택
3. Claude API 기반 멀티에이전트는 합리적 판단을 한다

### 2.2 포트폴리오 시그널 (Portfolio Weights)

**정의**: GC-HRP 알고리즘으로 계산된 자산별 최적 가중치

**알고리즘** (De Prado 2016):
```
1. 상관관계 행렬 계산 (252일 rolling)
2. MST 거리 변환: d(i,j) = sqrt(2 * (1 - ρ_ij))
3. 계층적 클러스터링 (Ward linkage)
4. 재귀적 리스크 패리티 할당
```

**가중치 제약**:
- 최소 가중치: 1%
- 최대 가중치: 60% (집중도 제한)
- 합: 100%

**시스템 리스크 노드** (MST 중심성 기반):
- Betweenness Centrality: 45%
- Degree Centrality: 35%
- Closeness Centrality: 20%
- 상위 sqrt(N)개 노드 선택

**가정**:
1. 과거 상관관계는 미래에도 지속된다 (단, 적응형 window 사용)
2. 계층적 구조는 리스크 분산에 효과적이다
3. MST는 가장 중요한 연결만 포착한다

### 2.3 통합 전략 시그널 (Integrated Signals)

**정의**: Portfolio + Causality 분석을 결합한 실행 가능한 액션

**생성 조건**:
1. **Strong Buy**:
   - Risk Score < 20 AND
   - Final = BULLISH (Conf > 70%) AND
   - Granger Causality: Liquidity → Market (p < 0.05)

2. **Buy**:
   - Risk Score < 40 AND
   - Final = BULLISH (Conf > 50%)

3. **Sell**:
   - Risk Score > 60 OR
   - Final = BEARISH (Conf > 50%)

4. **Reduce Exposure**:
   - Risk Score > 40 AND
   - Bubble Status = WARNING/DANGER

**가정**:
1. 리스크와 시그널이 일치할 때 신호가 강하다
2. 인과관계가 확인되면 신뢰도가 증가한다
3. 버블 리스크는 무조건 포지션 축소를 요구한다

---

## 3. 이상 탐지 기준

### 3.1 시장 이벤트 (Market Events)

**유동성 이벤트**:
```
Net Liquidity 변화 > 2σ (20일 rolling std)
예: TGA 급격한 감소 → 유동성 증가
```

**변동성 스파이크**:
```
VIX 변화 > 50% (1일)
예: VIX 14 → 21 (50% 증가)
```

**섹터 로테이션**:
```
상위 3개 섹터 수익률 차이 > 5% (5일)
예: Tech +8%, Finance -3% → 11% spread
```

### 3.2 암호화폐 이상 (Crypto Anomalies)

**거래량 이상** (24시간 rolling):
```
Volume_t / MA_7d(Volume) > 2.0
예: BTC 거래량 3.7배 폭발
```

**변동성 이상**:
```
Volatility Z-score > 2.0
예: ETH 변동성 4.1σ 급등
```

**가격 급등/급락**:
```
|Return_24h| > 10%
예: ETH +15% in 24h
```

**스테이블코인 De-peg**:
```
|Price - $1.00| > $0.02
예: USDT $0.985 → 1.5% de-peg
```

### 3.3 버블 리스크 (Bubble Risk)

**Greenwood-Shleifer 기준** (2019):

**Run-up Check**:
```
Cumulative Return (2 years) > 100%
예: NVDA +1094% (2022-2024) → WARNING
```

**Volatility Spike**:
```
(Vol_t - μ_vol) / σ_vol > 2.0
예: 변동성 2σ 초과
```

**Share Issuance** (선택):
```
Shares Outstanding 증가율 > 10% (1년)
```

**판정 레벨**:
- NONE: Run-up < 50%
- WATCH: 50% < Run-up < 100%
- WARNING: Run-up > 100% OR Vol Z-score > 2
- DANGER: Run-up > 150% AND Vol Z-score > 2.5

---

## 4. 시장 미세구조 기준

### 4.1 유동성 측정 (Amihud Lambda)

**정의**: 가격 충격당 거래량 비율
```
Lambda = (1/D) · Σ |Return_d| / Volume_d
Liquidity Score = 100 - Lambda (정규화)
```

**판정 기준**:
- Highly Liquid: Score > 70
- Moderately Liquid: 40 < Score ≤ 70
- Illiquid: Score ≤ 40

**가정**:
1. 유동성은 가격 충격과 반비례한다
2. 일별 데이터로도 유동성 추정이 가능하다

### 4.2 독성 주문 흐름 (VPIN)

**정의**: Volume-synchronized Probability of Informed Trading
```
VPIN = |Buy Volume - Sell Volume| / Total Volume
```

**일별 근사** (Easley et al. 2012):
```
OFI = (Up Volume - Down Volume) / Total Volume
VPIN ≈ |EMA(OFI, 20일)|
```

**판정 기준**:
- Low Toxicity: VPIN < 0.3
- Medium Toxicity: 0.3 ≤ VPIN < 0.5
- High Toxicity: VPIN ≥ 0.5

**가정**:
1. 정보 거래자는 방향성 있는 주문을 낸다
2. OFI는 정보 비대칭의 대리변수다

---

## 5. 데이터 가정

### 5.1 시계열 가정

1. **Stationarity**: 수익률은 약정상성(weak stationarity)을 만족한다
2. **Ergodicity**: 시간 평균 = 앙상블 평균
3. **No Look-ahead Bias**: 미래 정보를 사용하지 않는다
4. **Survivorship Bias**: 상장폐지 종목은 제외됨 (인정)

### 5.2 데이터 품질

**FRED 데이터**:
- 업데이트 주기: 일별/주별
- 지연: T+1 ~ T+3
- 신뢰도: 99.9% (연준 공식 데이터)

**yfinance 데이터**:
- 업데이트 주기: 실시간 (15분 지연)
- 조정: 액면분할/배당 조정됨
- 신뢰도: 95% (Yahoo Finance API)

**CryptoCompare 데이터**:
- 업데이트 주기: 1분
- 거래소: 다중 거래소 평균
- 신뢰도: 90% (API 안정성)

### 5.3 결측치 처리

1. **Forward Fill**: 최대 5일
2. **Drop**: 5일 이상 결측 시 해당 기간 제외
3. **Interpolation**: 선형 보간 (금리 데이터만)

---

## 6. 모델 가정 및 한계

### 6.1 GMM 레짐 모델

**가정**:
- 레짐 수 K=3 (고정)
- 전이 확률은 시간에 따라 변하지 않음 (stationary)
- 각 레짐은 정규분포

**한계**:
- 극단적 사건(fat tail)은 포착 못함
- 레짐 수를 자동으로 선택하지 않음
- 실시간 레짐 전환 감지는 1-2일 지연

### 6.2 Granger Causality

**가정**:
- 선형 관계
- Lag 수는 최적 (AIC/BIC 기준)
- 외생 변수 없음

**한계**:
- 인과관계 ≠ 진정한 원인 (상관관계)
- 비선형 관계는 포착 못함
- 구조적 변화(structural break) 시 실패

### 6.3 GC-HRP 포트폴리오

**가정**:
- 상관관계는 안정적 (rolling window 사용)
- 수익률은 정규분포 (실제로는 t-분포)
- 거래비용 무시

**한계**:
- 급격한 상관관계 변화에 느림
- 극단적 시장 상황에서 실패 가능
- 리밸런싱 비용 미고려

---

## 7. 백테스트 가정

### 7.1 거래 가정

- **슬리피지**: 0 (이상적 조건)
- **거래비용**: 0 (기관투자자 가정)
- **유동성**: 무한 (ETF 가정)
- **실행가**: 종가 (당일 종가에 실행 가능)

### 7.2 리밸런싱

- **주기**: 매일 (시그널 발생 시)
- **방법**: 전량 매도 후 재매수
- **최소 보유기간**: 없음

### 7.3 데이터 기간

- **학습 기간**: 2015-2019 (5년)
- **테스트 기간**: 2020-2024 (5년)
- **총 기간**: 10년

---

## 8. 판정 기준 요약표

| 항목 | 기준 | 출처 |
|------|------|------|
| Bull Regime | MA20 > MA50, Price > MA20, VIX < 20 | Rule-based |
| Bear Regime | MA20 < MA50, Price < MA50, VIX > 30 | Rule-based |
| GMM Bull | μ > 0.0005, σ < 0.015 | Hamilton 1989 |
| GMM Bear | μ < -0.0002, σ > 0.02 | Hamilton 1989 |
| High Risk | Risk Score > 60 | CriticalPath |
| High Uncertainty | Shannon Entropy > 1.0 | Shannon 1948 |
| Liquidity Event | Net Liq 변화 > 2σ | Custom |
| Volume Anomaly | Volume > 2.0 × MA7d | CryptoCompare |
| Volatility Spike | Vol Z-score > 2.0 | Standard |
| Bubble Warning | Run-up > 100% | Greenwood 2019 |
| Illiquid | Amihud Lambda > 60 | Amihud 2002 |
| High Toxicity | VPIN > 0.5 | Easley 2012 |
| Strong Buy | Risk < 20, BULLISH, Conf > 70% | Integrated |
| Sell | Risk > 60 OR BEARISH | Integrated |

---

**문서 작성**: 2026-01-12
**버전**: v2.1.2
**상태**: Final
