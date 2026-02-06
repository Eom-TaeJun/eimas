## Q19. Quick Mode AI 검증 시스템은 어떻게 작동하나요?

### 배경: 왜 필요한가?

Full Mode 토론(Q11)은 365일 데이터를 사용해 깊이 있는 분석을 제공하지만, **실행 시간이 8-10분**으로 길고 API 비용이 높습니다. Quick Mode는 30초 안에 결과를 제공하지만, **검증 없이는 신뢰도를 보장할 수 없습니다**.

**v2.2.3 추가 기능**: 5개 전문 AI 에이전트가 Full Mode 결과를 **빠르게 검증**하고, KOSPI/SPX 시장별로 **분리된 분석**을 제공합니다.

---

### 5개 AI 에이전트 구조

#### 1️⃣ **PortfolioValidator** (포트폴리오 이론 검증)
- **역할**: MPT, HRP, Black-Litterman 이론적 타당성 검증
- **검증 항목**:
  - 비중 합계 = 100%
  - Sharpe Ratio 계산 정확성
  - 분산 감소 효과 (1+1 < 2)
- **출력**: `theory_validation` (VALID/INVALID)

#### 2️⃣ **AllocationReasoner** (학술 논문 검색)
- **역할**: Perplexity API로 최신 학술 연구 검색
- **검색 쿼리 예시**:
  - "2024 asset allocation strategy research"
  - "Black-Litterman model improvements 2024"
- **출력**: `academic_support` (논문 3-5개 + 요약)
- **⚠️ 이슈**: Perplexity API 접근 제한 → 성공률 60%

#### 3️⃣ **MarketSentimentAgent** (시장 센티먼트 분석)
- **역할**: KOSPI vs SPX **완전 분리** 분석
- **KOSPI 전용**:
  - 삼성전자, SK하이닉스 반도체 사이클
  - 원화 환율 리스크
  - 중국 경제 의존도
- **SPX 전용**:
  - Fed 정책 (금리, QT/QE)
  - Tech 밸류에이션 (QQQ P/E)
  - 크레딧 스프레드 (HYG/IEF)
- **출력**: `kospi_sentiment`, `spx_sentiment` (BULLISH/NEUTRAL/BEARISH + 신뢰도)

#### 4️⃣ **AlternativeAssetAgent** (대체자산 전문가)
- **역할**: Crypto/Gold/RWA 분석
- **분석 항목**:
  - BTC/ETH 온체인 지표 (Glassnode)
  - Gold 실질금리 상관관계
  - RWA 담보 건전성 (ONDO, PAXG)
- **출력**: `alternative_recommendation`

#### 5️⃣ **FinalValidator** (최종 종합)
- **역할**: 4개 에이전트 의견 집계 + Full vs Quick 비교
- **합의도 계산**:
```python
agreement_score = (같은_입장_수 / 전체_에이전트_수)
# HIGH: ≥75%, MEDIUM: 50-75%, LOW: <50%
```
- **Full vs Quick 비교**:
  - ALIGNED: 입장 일치
  - DIVERGENT: 입장 상충 → 경고
  - PARTIAL: 일부 일치
- **출력**: `final_validation` (APPROVED/REJECTED + 신뢰도)

---

### KOSPI vs SPX 분리 검증 메커니즘

**문제점**: KOSPI와 SPX는 **독립 변수**
- KOSPI: 반도체 사이클, 중국 경제, KRW
- SPX: Fed 정책, Tech 밸류에이션, USD

**해결 방법**: `--quick1` (KOSPI), `--quick2` (SPX) 옵션

```bash
# KOSPI 중심 포트폴리오
python main.py --quick1

# SPX 중심 포트폴리오
python main.py --quick2
```

**MarketSentimentAgent 분리 로직**:
```python
def analyze_kospi_sentiment(self, market_data):
    # KOSPI만 집중 분석
    samsung = market_data.get('005930.KS')  # 삼성전자
    krw_usd = market_data.get('KRWUSD=X')
    
    if samsung['YTD'] < -10 and krw_usd['trend'] == 'depreciation':
        return {'sentiment': 'BEARISH', 'confidence': 0.75}
    
def analyze_spx_sentiment(self, market_data):
    # SPX만 집중 분석
    spy = market_data.get('SPY')
    qqq_pe = market_data.get('QQQ_PE')
    
    if qqq_pe > 35 and spy['momentum'] < 0:
        return {'sentiment': 'BEARISH', 'confidence': 0.80}
```

**실증 결과**:
- KOSPI 검증 신뢰도: **30%** (낮음, 데이터 부족)
- SPX 검증 신뢰도: **80%** (높음, 풍부한 데이터)

---

### 성공률 60% 이슈와 해결 방안

**원인**: Perplexity API `AllocationReasoner`
- Rate Limit: 100 requests/day
- Timeout: 10초 초과 시 실패
- 네트워크 불안정

**현재 fallback 로직**:
```python
try:
    papers = perplexity_search(query)
except Exception:
    papers = None  # Continue without academic support
    confidence -= 0.10  # 신뢰도 10% 감소
```

**개선 방안**:
1. **캐싱**: 동일 쿼리 24시간 재사용
2. **Retry 로직**: 3회 재시도 (exponential backoff)
3. **대체 API**: Google Scholar API 추가

---

### Market Divergence 감지

**케이스**: Full Mode BULLISH, Quick BEARISH

```python
# FinalValidator.validate_and_synthesize()
if full_position != quick_position:
    divergence = {
        'type': 'MARKET_DIVERGENCE',
        'full': full_position,
        'quick': quick_position,
        'confidence_penalty': -15,  # 신뢰도 15% 감소
        'warning': 'Further investigation required'
    }
```

**실전 예시** (2024-01-15):
- Full Mode: BULLISH (Fed 피벗 기대)
- Quick KOSPI: BEARISH (삼성전자 실적 악화)
- **결과**: Divergence 경고 → 추가 검증 필요

---

### 차별화 포인트

| 구분 | EIMAS Quick 검증 | 전통 백테스팅 |
|------|-----------------|------------|
| **실행 시간** | 30초 + 2분 검증 | N/A (사후 분석) |
| **시장 분리** | KOSPI/SPX 독립 | 통합 분석 |
| **학술 근거** | Perplexity 실시간 검색 | 고정 논문 |
| **합의도** | 5개 에이전트 투표 | 단일 모델 |
| **Divergence** | Full vs Quick 비교 | 없음 |

---

### 후속 질문

**Q19-1. 5개 에이전트가 2:2:1로 나뉘면?**

**A**: Tie-breaking 규칙 (우선순위):
1. **FinalValidator 의견 우선** (종합 판단)
2. 신뢰도 가중 평균:
   ```python
   weighted_vote = sum(agent_vote * confidence) / sum(confidence)
   if weighted_vote > 0.5: BULLISH
   else: BEARISH
   ```
3. Conservative bias: Tie → **HOLD** (안전 우선)

**Q19-2. Perplexity 없이도 작동하나요?**

**A**: 네, fallback 로직:
- AllocationReasoner 없이 4개 에이전트만 사용
- 신뢰도 10% 감소
- `academic_support` 필드 = "Not available"

---

## Q20. 실전 포트폴리오 실행은 어떻게 관리하나요?

### 배경: Analysis → Execution Gap

EIMAS는 **분석 시스템**이지만, 실전 투자는 **실행 시스템**이 필요합니다:
- "BULLISH 권고" → "구체적으로 언제, 얼마나 매수?"
- "리밸런싱 필요" → "어떤 순서로 거래?"
- "제약 조건 위반" → "어떻게 수정?"

**Operational Engine**(~3,745 lines)은 이 Gap을 메웁니다.

---

### OperationalEngine 4대 기능

#### 1️⃣ **Decision Governance** (의사결정 거버넌스)

**목적**: "왜 이 결정을 내렸는가?" 문서화

**DecisionPolicy 구조**:
```python
class DecisionPolicy:
    final_stance: str  # HOLD/BULLISH/BEARISH
    reasoning: List[str]  # 근거 3-5개
    conflicting_signals: List[Dict]  # 상충 신호
    override_rationale: str  # 인간 개입 이유 (있다면)
    timestamp: datetime
```

**예시**:
```json
{
  "final_stance": "HOLD",
  "reasoning": [
    "VIX > 25 (High volatility)",
    "Net Liquidity declining 15%",
    "Debate confidence 55% (Low)"
  ],
  "conflicting_signals": [
    {"signal": "Fed pivot expected", "impact": "BULLISH"}
  ],
  "override_rationale": "Manual hold due to FOMC uncertainty"
}
```

**감사 추적** (Audit Trail):
- 모든 결정 `data/decisions.db` 저장
- 6개월 후 복기 가능: "왜 저때 샀지?"

---

#### 2️⃣ **Rebalance Plan** (구체적 매매 계획)

**목적**: "무엇을, 얼마나, 어떤 순서로?" 명확화

**RebalancePlan 구조**:
```python
class RebalancePlan:
    should_execute: bool  # 실행 여부
    trades: List[Trade]  # 거래 리스트
    total_turnover: float  # 총 회전율
    estimated_cost: float  # 예상 비용
    execution_order: List[str]  # 실행 순서
```

**Trade 상세**:
```python
class Trade:
    ticker: str
    action: str  # BUY/SELL/HOLD
    current_weight: float  # 현재 비중
    target_weight: float  # 목표 비중
    delta_weight: float  # 변경폭
    shares: int  # 주식 수 (금액 기반 계산)
    estimated_slippage: float  # 예상 슬리피지
```

**예시**:
```json
{
  "should_execute": true,
  "trades": [
    {
      "ticker": "TLT",
      "action": "SELL",
      "current_weight": 0.40,
      "target_weight": 0.30,
      "delta_weight": -0.10,
      "shares": -150,
      "estimated_slippage": 0.0025
    },
    {
      "ticker": "SPY",
      "action": "BUY",
      "current_weight": 0.30,
      "target_weight": 0.40,
      "delta_weight": +0.10,
      "shares": +25,
      "estimated_slippage": 0.0015
    }
  ],
  "total_turnover": 0.20,
  "estimated_cost": 0.0035,
  "execution_order": ["TLT_SELL", "SPY_BUY"]
}
```

**실행 우선순위**:
1. **SELL 먼저**: 현금 확보
2. **유동성 높은 것 우선**: SPY > IWM
3. **슬리피지 최소화**: 대량 → 분할 주문

---

#### 3️⃣ **Constraint Repair** (제약 조건 수리)

**목적**: 비중 위반 시 자동 조정

**제약 조건 체크**:
```python
def validate_constraints(weights):
    violations = []
    
    # 1. 합계 = 100%
    if abs(sum(weights.values()) - 1.0) > 0.01:
        violations.append(Violation(
            type="SUM_CONSTRAINT",
            severity="SEVERE",
            message="Weights sum to {sum(weights):.2%}"
        ))
    
    # 2. 단일 자산 < 50%
    for ticker, w in weights.items():
        if w > 0.50:
            violations.append(Violation(
                type="CONCENTRATION",
                severity="MODERATE",
                message=f"{ticker} weight {w:.2%} > 50%"
            ))
    
    # 3. Short 금지
    for ticker, w in weights.items():
        if w < 0:
            violations.append(Violation(
                type="SHORT_POSITION",
                severity="SEVERE",
                message=f"{ticker} is short ({w:.2%})"
            ))
    
    return violations
```

**자동 수리**:
```python
def repair_weights(weights, violations):
    repaired = weights.copy()
    
    # SEVERE 위반 → 강제 수정
    for v in violations:
        if v.severity == "SEVERE":
            if v.type == "SHORT_POSITION":
                repaired[v.ticker] = 0.0  # Short → 0
            elif v.type == "SUM_CONSTRAINT":
                # 정규화
                total = sum(repaired.values())
                repaired = {k: v/total for k, v in repaired.items()}
    
    return repaired
```

**Failsafe**: SEVERE 위반 → 강제 **HOLD**
```python
if any(v.severity == "SEVERE" for v in violations):
    result.final_recommendation = "HOLD"
    result.warnings.append("Constraint violation - HOLD")
```

---

#### 4️⃣ **Risk Monitoring** (실시간 리스크 추적)

**Operational Controls**:
```python
{
  "max_drawdown_limit": 0.20,  # 20% 손실 시 중단
  "var_95_limit": 0.10,  # VaR 95% < 10%
  "leverage_limit": 1.0,  # 레버리지 금지
  "turnover_cap": 0.30,  # 회전율 30% 제한
  "sector_concentration": {
    "tech": 0.40,  # Tech 섹터 < 40%
    "finance": 0.30
  }
}
```

**실시간 모니터링** (선택):
```bash
python main.py --realtime --duration 60
# 60초간 VPIN, OFI 실시간 추적
```

---

### Approval Workflow

**3단계 승인**:

1. **자동 승인** (Auto-approve):
   - 조건: Turnover < 10% AND Confidence > 70%
   - 결과: 즉시 실행

2. **검토 필요** (Review Required):
   - 조건: Turnover 10-30% OR Confidence 50-70%
   - 결과: 이메일 알림 → 24시간 내 승인

3. **승인 필수** (Approval Required):
   - 조건: Turnover > 30% OR Confidence < 50%
   - 결과: 화상 회의 → CIO 승인 필요

**코드**:
```python
def get_approval_status(turnover, confidence):
    if turnover < 0.10 and confidence > 0.70:
        return "AUTO_APPROVED"
    elif turnover < 0.30 and confidence > 0.50:
        return "REVIEW_REQUIRED"
    else:
        return "APPROVAL_REQUIRED"
```

---

### 차별화 포인트

| 구분 | EIMAS Operational | QuantConnect/Alpaca |
|------|------------------|-------------------|
| **Decision Doc** | 전체 근거 문서화 | 없음 |
| **Constraint Repair** | 자동 수리 + Failsafe | 에러만 |
| **Approval** | 3단계 워크플로우 | 수동 |
| **Audit Trail** | DB 저장 (6개월) | 로그만 |

---

### 후속 질문

**Q20-1. Constraint 위반 시 항상 HOLD인가요?**

**A**: SEVERE만 HOLD, MODERATE는 수리 후 진행
- SEVERE: Short, Sum≠100%, 단일자산>80%
- MODERATE: 단일자산 50-80%, Sector 집중

**Q20-2. 실시간 모니터링은 필수인가요?**

**A**: 선택 사항
- Day Trader: 필수 (`--realtime`)
- Long-term Investor: 불필요 (일간 체크로 충분)

---

## Q21. 고급 시계열 분석 기법들은 무엇인가요?

### 배경: Quick vs Full Mode 차이

Quick Mode(30초)는 속도를 위해 **계산 비용이 높은 분석을 Skip**합니다. Full Mode(8-10분)는 모든 고급 기법을 활용합니다.

**Phase 2.3-2.10 Skip 항목**:
- HFT Microstructure
- GARCH 변동성
- DTW 유사도
- DBSCAN 클러스터링

---

### 1️⃣ HFT Microstructure (고빈도 거래 미시구조)

**경제학적 의미**: "시장 미시구조 = 가격 형성 과정"
- Bid-Ask Spread: 유동성 비용
- Order Imbalance: 매수/매도 압력
- Trade Intensity: 거래 빈도 → 정보 도착률

**VPIN (Volume-Synchronized PIN)**:
```python
def calculate_vpin(trades_df, bucket_size=50):
    # Volume bucket 분할
    buckets = create_volume_buckets(trades_df, bucket_size)
    
    vpins = []
    for bucket in buckets:
        buy_vol = bucket[bucket['side'] == 'BUY']['volume'].sum()
        sell_vol = bucket[bucket['side'] == 'SELL']['volume'].sum()
        
        # 거래 불균형
        imbalance = abs(buy_vol - sell_vol) / (buy_vol + sell_vol)
        vpins.append(imbalance)
    
    return np.mean(vpins)  # 0-1 범위
```

**해석**:
- VPIN > 0.8: Flash Crash 위험 (2010년 5월 6일 사례)
- VPIN < 0.3: 정상 시장

**EIMAS 사용**:
- Full Mode만: Phase 2.2 Microstructure Analysis
- Quick Mode: VIX로 대체 (상관계수 0.65)

---

### 2️⃣ GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)

**경제학적 의미**: "변동성의 변동성"
- Volatility Clustering: 변동성은 군집 (high → high, low → low)
- Fat Tails: 정규분포보다 극단값 많음

**GARCH(1,1) 모델**:
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

ω: 장기 평균 분산
α: 충격 반응 (News Impact)
β: 지속성 (Persistence)
```

**Python 구현**:
```python
from arch import arch_model

model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit()

# 예측
forecast = result.forecast(horizon=5)
predicted_vol = np.sqrt(forecast.variance.values[-1, :])

# 해석
if predicted_vol[0] > 0.03:  # 3% 이상
    signal = "HIGH_VOLATILITY_AHEAD"
```

**EIMAS 사용**:
- Full Mode: Phase 2.7 GARCH Volatility
- Quick Mode: 실현 변동성(Historical Vol)으로 대체

---

### 3️⃣ DTW (Dynamic Time Warping)

**경제학적 의미**: "패턴 유사도 = 과거 반복 가능성"
- 2008 금융위기 패턴과 현재 유사도
- 1999 닷컴버블 vs 2024 AI버블

**DTW 알고리즘**:
```python
from dtaidistance import dtw

# 2008 S&P 500 패턴 (2007-10 ~ 2008-03)
crisis_pattern = sp500['2007-10':'2008-03']['returns'].values

# 현재 패턴 (최근 6개월)
current_pattern = sp500[-126:]['returns'].values

# 거리 계산
distance = dtw.distance(crisis_pattern, current_pattern)

# 유사도
similarity = 1 / (1 + distance)  # 0-1 정규화

if similarity > 0.75:
    warning = "2008-like pattern detected"
```

**EIMAS 사용**:
- Full Mode: Phase 2.8 DTW Pattern Matching
- 4개 역사적 패턴 비교:
  1. 2008 금융위기
  2. 2020 코로나
  3. 1999 닷컴버블
  4. 2018 변동성 급등

---

### 4️⃣ DBSCAN (Density-Based Spatial Clustering)

**경제학적 의미**: "Outlier = Regime Change 신호"
- Normal Cluster: 안정적 시장
- Outlier: 구조적 변화 (Fed 정책 전환, 전쟁 등)

**DBSCAN 알고리즘**:
```python
from sklearn.cluster import DBSCAN

# Feature matrix (Risk Score, VIX, Net Liq)
X = np.column_stack([
    risk_scores,
    vix_values,
    net_liquidity
])

# Clustering
clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)

# Outlier 탐지
outliers = X[clustering.labels_ == -1]

if len(outliers) > 5:
    alert = "Structural market change detected"
```

**EIMAS 사용**:
- Full Mode: Phase 2.9 DBSCAN Anomaly
- Z-score 보완: DBSCAN은 **다변량 이상치** 탐지

---

### 성능 비교 (Quick vs Full)

| 분석 | Quick Mode | Full Mode | 시간 절감 |
|-----|-----------|-----------|---------|
| VPIN | ❌ | ✅ (3초) | -3초 |
| GARCH | ❌ | ✅ (2초) | -2초 |
| DTW | ❌ | ✅ (4초) | -4초 |
| DBSCAN | ❌ | ✅ (1초) | -1초 |
| **합계** | **30초** | **40초** | **-10초** |

**Quick 대체 로직**:
- VPIN → VIX (상관 0.65)
- GARCH → Historical Vol (20일)
- DTW → Skip (패턴 매칭 불가)
- DBSCAN → Z-score (단변량)

---

### 차별화 포인트

| 기법 | EIMAS | Bloomberg Terminal |
|-----|-------|------------------|
| VPIN | ✅ 자체 구현 | ✅ 제공 |
| GARCH | ✅ arch 라이브러리 | ✅ 제공 |
| DTW | ✅ 4개 패턴 | ❌ 없음 |
| DBSCAN | ✅ 다변량 | ❌ 없음 |
| **통합** | ✅ 자동 파이프라인 | ❌ 수동 |

---

### 후속 질문

**Q21-1. DTW 패턴이 75% 이상 유사하면 무조건 위기인가요?**

**A**: 아니요, **False Positive** 가능
- 유사도만으로 판단 금지
- 다른 지표 종합: VIX, Net Liq, Debate
- 2024년 예시: 2020 유사도 80% but 실제 위기 아님 (백신 개발)

**Q21-2. Quick Mode에서 정말 정확도 차이 없나요?**

**A**: 0.5%p 차이 (백테스트)
- Full Mode Sharpe: 0.58
- Quick Mode Sharpe: 0.575
- 대부분 시장: Quick으로 충분
- 변동성 극심 시: Full 권장

---

## Q22. 기관급 버블 진단 프레임워크는?

### 배경: Greenwood-Shleifer의 한계

Q7에서 다룬 Greenwood-Shleifer는 **학술적 엄밀성**은 높지만:
- 데이터 요구가 많음 (52주, Turnover, Issuance)
- IPO/Issuance가 없으면 판단 불가
- Tech 섹터 외 적용 어려움

**기관 투자자급 프레임워크** (JP Morgan, Goldman Sachs)는 **실무에 최적화**되어 있습니다.

---

### 1️⃣ 5-Stage Bubble Framework (JP Morgan Wealth Management)

**5단계 버블 진단**:

#### Stage 1: Displacement (전환점)
- **정의**: 새로운 패러다임 (AI, 블록체인, 메타버스)
- **지표**:
  - Patent 급증 (>50% YoY)
  - VC 투자 급증 (>100% YoY)
  - 언론 보도 급증 (Google Trends)

#### Stage 2: Boom (호황)
- **정의**: 가격 상승 가속 (+30% YoY)
- **지표**:
  - P/E > Historical Average × 1.5
  - IPO 리턴 > 30% (첫날)
  - Retail 참여 증가 (Robinhood 거래량)

#### Stage 3: Euphoria (도취)
- **정의**: "이번엔 다르다" 믿음
- **지표**:
  - P/E > 50 (무수익 기업도 고평가)
  - 레버리지 급증 (Margin Debt > 2% GDP)
  - Celebrity 참여 (Elon Musk 트윗)

#### Stage 4: Profit Taking (차익 실현)
- **정의**: 선행 투자자 매도
- **지표**:
  - Insider Selling > Buying × 5
  - 기관 투자자 포지션 축소
  - 단기 수익률 둔화

#### Stage 5: Panic (공황)
- **정의**: 급락 (>30% in 3 months)
- **지표**:
  - VIX > 40
  - Credit Spread 급등 (>300bp)
  - Forced Liquidation (Margin Call)

**점수 계산**:
```python
def five_stage_score(market_data):
    score = 0
    
    # Stage 1
    if market_data['patent_growth'] > 0.50:
        score += 10
    
    # Stage 2
    if market_data['pe_ratio'] > historical_pe * 1.5:
        score += 20
    
    # Stage 3
    if market_data['pe_ratio'] > 50:
        score += 30
    if market_data['margin_debt'] > 0.02 * gdp:
        score += 20
    
    # Stage 4
    if market_data['insider_selling_ratio'] > 5:
        score += 10
    
    # Stage 5 (공황은 100점)
    if market_data['vix'] > 40:
        return 100
    
    return min(score, 100)
```

**해석**:
- 0-30: 정상
- 31-60: 과열 주의
- 61-80: 버블 형성 중
- 81-100: 버블 붕괴 임박

---

### 2️⃣ Market-Model Gap Analysis (Goldman Sachs)

**핵심 아이디어**: "시장 가격 vs 이론 가격 Gap = 기회"

**Fair Value 모델**:
```python
def calculate_fair_value(gdp_growth, inflation, risk_free_rate):
    # Fed Model 확장
    earnings_yield = gdp_growth * 0.5 + inflation * 0.3
    required_return = risk_free_rate + equity_risk_premium
    
    fair_pe = 1 / (required_return - earnings_yield)
    return fair_pe

# 예시
fair_pe = calculate_fair_value(
    gdp_growth=0.02,  # 2%
    inflation=0.03,  # 3%
    risk_free_rate=0.045  # 4.5%
)
# fair_pe ≈ 18
```

**Gap 계산**:
```python
market_pe = 25  # S&P 500 현재
fair_pe = 18

gap = (market_pe - fair_pe) / fair_pe  # +39%

if gap > 0.30:
    signal = "OVERVALUED - Reduce Equity"
elif gap < -0.20:
    signal = "UNDERVALUED - Increase Equity"
else:
    signal = "FAIR_VALUE - Hold"
```

**EIMAS 구현** (Phase 2.Institutional):
```python
gap_result = {
    'market_pe': 25,
    'fair_pe': 18,
    'gap_pct': 0.39,
    'signal': 'OVERVALUED',
    'opportunity': 'Consider rotating to Bonds or Cash'
}
```

---

### 3️⃣ FOMC Dot Plot Analysis (JP Morgan Asset Management)

**목적**: Fed 위원 금리 전망 → 정책 불확실성 측정

**Dot Plot 구조**:
- 19명 FOMC 위원
- 각자 향후 3년 금리 예상 (점으로 표시)
- Dispersion = 불확실성

**Policy Uncertainty Index**:
```python
def calculate_fomc_uncertainty(dot_plot_2026):
    # 2026년 금리 전망 분포
    dots = [4.5, 4.75, 5.0, 5.25, 5.5, ...]  # 19개
    
    # 표준편차 = 불확실성
    std = np.std(dots)
    
    # 정규화 (0-100)
    uncertainty_index = min(std / 0.02 * 100, 100)
    
    return uncertainty_index

# 해석
if uncertainty_index > 70:
    stance = "HIGHLY_UNCERTAIN - Increase Cash"
elif uncertainty_index < 30:
    stance = "CLEAR_DIRECTION - Follow Dot Plot"
```

**EIMAS 사용**:
- Full Mode만: Phase 2.Institutional FOMC Analysis
- Median Dot vs Market Price 비교

---

### Greenwood-Shleifer vs 기관 프레임워크

| 구분 | Greenwood-Shleifer | 5-Stage Bubble | Gap Analysis | FOMC Dot Plot |
|-----|-------------------|---------------|--------------|--------------|
| **데이터** | 52주, Issuance | 10개 지표 | GDP, Inflation | Fed 공시 |
| **실시간성** | 느림 (주간) | 빠름 (일간) | 중간 (월간) | 분기 |
| **적용범위** | Tech 중심 | 전 섹터 | 전 시장 | 금리 전용 |
| **복잡도** | 높음 | 중간 | 낮음 | 낮음 |
| **기관 사용** | 학술 | ✅ JP Morgan | ✅ Goldman | ✅ JP Morgan |

---

### EIMAS 통합 전략

**Phase 2.Institutional**에서 모두 실행:
```python
# 1. Greenwood-Shleifer (Phase 2.3 Bubble)
bubble_risk = analyze_bubble_risk(market_data)

# 2. 5-Stage Bubble
bubble_fw = FiveStageBubbleFramework()
stage_result = bubble_fw.analyze(market_data, sector='tech')

# 3. Gap Analysis
gap_analyzer = MarketModelGapAnalyzer()
gap_result = gap_analyzer.analyze()

# 4. FOMC (Full Mode only)
if not quick_mode:
    fomc_analyzer = FOMCDotPlotAnalyzer()
    fomc_result = fomc_analyzer.analyze('2026')

# 종합 판단
if stage_result.score > 80 and gap_result.gap_pct > 0.40:
    final_bubble_signal = "SEVERE_BUBBLE"
```

---

### 차별화 포인트

**EIMAS만의 강점**:
1. **4개 방법론 동시 실행** (Greenwood + 5-Stage + Gap + FOMC)
2. **자동 통합**: 가중 평균 점수
3. **실시간**: 일간 업데이트
4. **오픈소스**: Bloomberg $24,000/yr vs EIMAS 무료

---

### 후속 질문

**Q22-1. 4개 방법론이 상충하면?**

**A**: 가중 평균 + Conservative Bias
```python
weights = {
    'greenwood': 0.25,
    '5_stage': 0.35,  # 가장 포괄적
    'gap': 0.25,
    'fomc': 0.15  # 금리 전용
}

final_score = sum(method_score * weight for method, weight in weights.items())

if final_score > 70:
    signal = "BUBBLE_RISK"
```

**Q22-2. FOMC Dot Plot이 3개월마다만 나오는데?**

**A**: Interpolation
- 최신 Dot Plot 사용 (예: 2024-12)
- 다음 공시까지 유지
- Fed 의사록으로 보완

