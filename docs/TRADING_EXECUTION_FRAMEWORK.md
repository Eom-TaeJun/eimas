# EIMAS Trading Execution Framework
## 트레이딩 실행 및 포트폴리오 피드백 시스템

> **목표**: 시장 세션별 전략 실행 → AI 시그널 기반 포트폴리오 → DB 기록 → 실제값 피드백

---

## 1. 시장 세션 개요

### 1.1 미국 시장 시간대 (ET 기준)

| 세션 | 시간 (ET) | 한국 시간 (KST) | 특징 |
|------|-----------|-----------------|------|
| **Pre-Market** | 04:00 - 09:30 | 18:00 - 23:30 (전일) | 유동성 낮음, 스프레드 큼, 뉴스 반응 |
| **Regular Hours** | 09:30 - 16:00 | 23:30 - 06:00 | 최대 유동성, 기관 참여 |
| **After-Hours** | 16:00 - 20:00 | 06:00 - 10:00 | 실적 발표 반응, 변동성 높음 |
| **Overnight** | 20:00 - 04:00 | 10:00 - 18:00 | 선물만, 아시아/유럽 영향 |

### 1.2 세션별 특성

```
Pre-Market (프리마켓)
├── 장점: 뉴스/실적에 선제 대응 가능
├── 단점: 유동성 부족, 슬리피지 큼
├── 적합: 확실한 촉매가 있는 경우만
└── 데이터: 거래량 급증 = 강한 시그널

Regular Hours (정규장)
├── 09:30-10:30: Opening Range (방향성 결정)
├── 10:30-14:00: Mid-day (횡보, 기관 축적)
├── 14:00-15:00: Power Hour 전 (포지션 정리)
└── 15:00-16:00: Power Hour (최종 방향성)

After-Hours (애프터)
├── 16:00-17:00: 실적 발표 집중 시간
├── 변동성 극심, 리스크 높음
└── 적합: 실적 플레이 또는 헤지
```

---

## 2. 세션별 전략 실행 가이드라인

### 2.1 전략 유형별 최적 실행 시간

| 전략 | Pre-Market | Open (09:30-10:30) | Mid-Day | Close (15:00-16:00) | After |
|------|------------|---------------------|---------|---------------------|-------|
| **모멘텀** | ⚠️ 제한적 | ✅ 최적 | ❌ | ⚠️ | ❌ |
| **역발상** | ❌ | ⚠️ | ✅ 최적 | ✅ | ❌ |
| **스윙** | ⚠️ | ✅ | ✅ | ✅ 최적 | ⚠️ |
| **이벤트** | ✅ 실적 전 | ⚠️ | ⚠️ | ⚠️ | ✅ 실적 후 |
| **레짐 기반** | ❌ | ⚠️ 관찰 | ✅ 확인 후 | ✅ 최적 | ❌ |

### 2.2 실행 규칙

```python
EXECUTION_RULES = {
    "pre_market": {
        "condition": "major_catalyst_exists",  # 실적, M&A, FDA 등
        "position_size": 0.5,  # 정규장의 50%
        "stop_loss": 0.03,     # 3% (타이트)
        "order_type": "limit", # 시장가 금지
    },
    "opening_range": {
        "wait_minutes": 15,    # 첫 15분 관찰
        "confirm_volume": True,
        "position_size": 0.7,
        "stop_loss": 0.05,
    },
    "mid_day": {
        "avoid_chop": True,    # 횡보장 진입 자제
        "scale_in": True,      # 분할 매수
        "position_size": 1.0,
    },
    "power_hour": {
        "best_for": ["swing", "regime"],
        "position_size": 1.0,
        "capture_close": True,  # 종가 기준 진입
    },
    "after_hours": {
        "condition": "earnings_play_only",
        "position_size": 0.3,
        "hedge_required": True,
    }
}
```

### 2.3 시간대별 슬리피지 예상

| 세션 | 예상 슬리피지 | 스프레드 | 유동성 점수 |
|------|--------------|----------|------------|
| Pre-Market | 0.3-1.0% | 0.2-0.5% | 2/10 |
| Open (첫 15분) | 0.1-0.3% | 0.05-0.1% | 7/10 |
| Mid-Day | 0.05-0.1% | 0.02-0.05% | 10/10 |
| Power Hour | 0.05-0.15% | 0.03-0.07% | 9/10 |
| After-Hours | 0.5-1.5% | 0.3-0.8% | 1/10 |

---

## 3. AI 시그널 기반 포트폴리오 설계

### 3.1 투자자 프로파일

```python
INVESTOR_PROFILES = {
    "conservative": {
        "name": "보수적 투자자",
        "risk_tolerance": 0.3,      # 최대 30% 손실 감내
        "target_return": 0.08,      # 연 8% 목표
        "max_drawdown": 0.10,       # MDD 10% 제한
        "position_limit": 0.20,     # 종목당 최대 20%
        "cash_minimum": 0.30,       # 최소 현금 30%
        "rebalance_freq": "monthly",
        "preferred_signals": ["regime_low_vol", "quality_factor"],
    },
    "moderate": {
        "name": "중립적 투자자",
        "risk_tolerance": 0.5,
        "target_return": 0.15,
        "max_drawdown": 0.20,
        "position_limit": 0.25,
        "cash_minimum": 0.15,
        "rebalance_freq": "weekly",
        "preferred_signals": ["multi_factor", "momentum"],
    },
    "aggressive": {
        "name": "공격적 투자자",
        "risk_tolerance": 0.7,
        "target_return": 0.25,
        "max_drawdown": 0.35,
        "position_limit": 0.40,
        "cash_minimum": 0.05,
        "rebalance_freq": "daily",
        "preferred_signals": ["momentum", "vix_reversal", "leverage"],
    },
    "tactical": {
        "name": "전술적 투자자",
        "risk_tolerance": 0.6,
        "target_return": 0.20,
        "max_drawdown": 0.25,
        "position_limit": 0.30,
        "cash_minimum": 0.10,
        "rebalance_freq": "signal_based",
        "preferred_signals": ["regime_change", "critical_path", "event"],
    }
}
```

### 3.2 AI 에이전트별 시그널 가중치

```python
AGENT_SIGNAL_WEIGHTS = {
    "regime_detector": {
        "bull_low_vol": {"action": "long", "conviction": 0.8},
        "bull_high_vol": {"action": "reduce", "conviction": 0.6},
        "bear_low_vol": {"action": "hedge", "conviction": 0.5},
        "bear_high_vol": {"action": "defensive", "conviction": 0.9},
        "transition": {"action": "wait", "conviction": 0.3},
    },
    "critical_path_monitor": {
        "yield_curve_inversion": {"action": "defensive", "lead_days": 180},
        "credit_spread_widen": {"action": "reduce_risk", "lead_days": 60},
        "vix_spike": {"action": "contrarian_buy", "lead_days": 5},
        "breakeven_drop": {"action": "deflation_hedge", "lead_days": 90},
    },
    "etf_flow_analyzer": {
        "sector_rotation": {"action": "follow", "conviction": 0.7},
        "risk_on_flow": {"action": "long_beta", "conviction": 0.6},
        "risk_off_flow": {"action": "long_quality", "conviction": 0.6},
    },
    "fear_greed_index": {
        "extreme_fear": {"action": "contrarian_long", "conviction": 0.8},
        "extreme_greed": {"action": "reduce", "conviction": 0.7},
    }
}
```

### 3.3 포트폴리오 후보 생성

```python
def generate_portfolio_candidates(
    profile: str,
    signals: Dict[str, Any],
    market_regime: str
) -> List[Portfolio]:
    """
    투자자 프로파일 + AI 시그널 → 포트폴리오 후보 3-5개 생성

    Returns:
        List of Portfolio candidates with:
        - allocations: Dict[str, float]
        - expected_return: float
        - expected_risk: float
        - confidence: float
        - reasoning: str
    """
    candidates = []

    # 후보 1: 시그널 기반 최적화
    # 후보 2: 벤치마크 추종 + 알파
    # 후보 3: 리스크 패리티
    # 후보 4: 모멘텀 기반
    # 후보 5: 방어적 배분

    return rank_by_sharpe(candidates)
```

---

## 4. 데이터베이스 스키마

### 4.1 핵심 테이블

```sql
-- 1. 시그널 기록
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    signal_source VARCHAR(50) NOT NULL,  -- 'regime_detector', 'critical_path', etc.
    signal_type VARCHAR(50) NOT NULL,    -- 'BUY', 'SELL', 'HOLD'
    ticker VARCHAR(10),
    conviction FLOAT,                     -- 0.0 - 1.0
    reasoning TEXT,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. 포트폴리오 후보
CREATE TABLE portfolio_candidates (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    profile_type VARCHAR(20) NOT NULL,   -- 'conservative', 'aggressive', etc.
    candidate_rank INTEGER,               -- 1, 2, 3...
    allocations JSON,                     -- {"SPY": 0.4, "TLT": 0.3, ...}
    expected_return FLOAT,
    expected_risk FLOAT,
    expected_sharpe FLOAT,
    signals_used JSON,                    -- 사용된 시그널 ID 목록
    reasoning TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 3. 실행 기록
CREATE TABLE executions (
    id INTEGER PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolio_candidates(id),
    execution_time DATETIME NOT NULL,
    session_type VARCHAR(20),            -- 'pre_market', 'regular', 'after_hours'
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(10) NOT NULL,         -- 'BUY', 'SELL'
    target_price FLOAT,
    executed_price FLOAT,
    slippage FLOAT,
    shares FLOAT,
    commission FLOAT,
    status VARCHAR(20),                  -- 'pending', 'filled', 'cancelled'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 4. 성과 추적 (핵심!)
CREATE TABLE performance_tracking (
    id INTEGER PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolio_candidates(id),
    date DATE NOT NULL,
    -- 예측값 (포트폴리오 생성 시점)
    predicted_return_1d FLOAT,
    predicted_return_1w FLOAT,
    predicted_return_1m FLOAT,
    predicted_volatility FLOAT,
    -- 실제값 (시간이 지난 후 기록)
    actual_return_1d FLOAT,
    actual_return_1w FLOAT,
    actual_return_1m FLOAT,
    actual_volatility FLOAT,
    -- 평가
    prediction_error_1d FLOAT,           -- actual - predicted
    prediction_error_1w FLOAT,
    prediction_error_1m FLOAT,
    mape FLOAT,                          -- Mean Absolute Percentage Error
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

-- 5. 시그널 성과 (피드백 루프의 핵심)
CREATE TABLE signal_performance (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(id),
    evaluation_date DATE,
    -- 시그널 발생 후 성과
    return_1d FLOAT,                     -- 1일 후 수익률
    return_5d FLOAT,                     -- 5일 후
    return_20d FLOAT,                    -- 20일 후
    return_60d FLOAT,                    -- 60일 후
    max_gain FLOAT,                      -- 기간 내 최대 상승
    max_loss FLOAT,                      -- 기간 내 최대 하락
    -- 시그널 품질 평가
    signal_accuracy BOOLEAN,             -- 방향 맞았는지
    profit_factor FLOAT,
    information_ratio FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 6. 세션별 실행 분석
CREATE TABLE session_analysis (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    -- 세션별 수익률
    pre_market_return FLOAT,
    opening_hour_return FLOAT,           -- 09:30-10:30
    mid_day_return FLOAT,                -- 10:30-14:00
    power_hour_return FLOAT,             -- 15:00-16:00
    after_hours_return FLOAT,
    overnight_return FLOAT,
    -- 최적 실행 시간
    best_buy_time TIME,
    best_sell_time TIME,
    volume_distribution JSON,            -- 시간대별 거래량
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 인덱스

```sql
CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_source ON signals(signal_source);
CREATE INDEX idx_performance_date ON performance_tracking(date);
CREATE INDEX idx_signal_perf_date ON signal_performance(evaluation_date);
CREATE INDEX idx_session_date_ticker ON session_analysis(date, ticker);
```

---

## 5. 선행지표 탐색 프레임워크

### 5.1 현재 확인된 선행지표

| 지표 | 선행 기간 | 대상 | 신뢰도 | 출처 |
|------|----------|------|--------|------|
| **Yield Curve (10Y-2Y)** | 12-18개월 | 경기침체 | 높음 | Fed, 학술 |
| **Copper/Gold Ratio** | 3-6개월 | 경기방향 | 중간 | 경험적 |
| **Initial Claims** | 1-3개월 | 고용시장 | 높음 | FRED |
| **ISM New Orders** | 2-4개월 | 제조업 | 중간 | ISM |
| **Credit Spreads** | 1-3개월 | 리스크 | 높음 | FRED |
| **MOVE Index** | 1-2주 | 채권 변동성 | 중간 | Bloomberg |
| **Fear & Greed** | 1-2주 | 단기 반전 | 중간 | CNN |
| **Put/Call Ratio** | 1-5일 | 단기 반전 | 낮음 | CBOE |
| **VIX Term Structure** | 1-2주 | 변동성 방향 | 중간 | CBOE |

### 5.2 탐색 중인 잠재적 선행지표

```python
EXPERIMENTAL_LEADING_INDICATORS = {
    # === 거시 경제 ===
    "fed_balance_sheet": {
        "source": "FRED",
        "hypothesis": "자산 증가 → 유동성 증가 → 주식 상승",
        "lead_time": "3-6개월",
        "status": "testing",
    },
    "m2_growth": {
        "source": "FRED",
        "hypothesis": "M2 증가율 둔화 → 유동성 축소 → 주식 약세",
        "lead_time": "6-12개월",
        "status": "testing",
    },
    "china_pmi": {
        "source": "NBS",
        "hypothesis": "중국 PMI → 글로벌 경기 → 미국 주식",
        "lead_time": "1-3개월",
        "status": "testing",
    },

    # === 시장 구조 ===
    "dealer_gamma": {
        "source": "Options Flow",
        "hypothesis": "딜러 감마 포지션 → 시장 방향성",
        "lead_time": "1-5일",
        "status": "research",
    },
    "dark_pool_activity": {
        "source": "FINRA",
        "hypothesis": "다크풀 거래량 증가 → 기관 축적 → 상승",
        "lead_time": "1-2주",
        "status": "research",
    },
    "repo_rate_stress": {
        "source": "NY Fed",
        "hypothesis": "레포 금리 급등 → 유동성 위기",
        "lead_time": "1-3일",
        "status": "monitoring",
    },

    # === 대안 데이터 ===
    "satellite_retail_traffic": {
        "source": "Alternative Data",
        "hypothesis": "소매 트래픽 → 소비 → 리테일 실적",
        "lead_time": "1-2개월",
        "status": "unavailable",
    },
    "job_postings": {
        "source": "Indeed/LinkedIn",
        "hypothesis": "채용공고 → 고용 → 경기",
        "lead_time": "2-4개월",
        "status": "research",
    },
    "shipping_rates": {
        "source": "Baltic Dry Index",
        "hypothesis": "운임 → 글로벌 무역 → 경기",
        "lead_time": "2-3개월",
        "status": "testing",
    },

    # === 심리 지표 ===
    "social_sentiment": {
        "source": "Twitter/Reddit",
        "hypothesis": "소셜 심리 → 개인 투자자 행동",
        "lead_time": "1-3일",
        "status": "noisy",  # 뉴스 왜곡 주의
        "caution": "의도적 왜곡 가능성 높음",
    },
    "insider_trading": {
        "source": "SEC Form 4",
        "hypothesis": "내부자 매수 → 주가 상승",
        "lead_time": "1-3개월",
        "status": "testing",
    },
}
```

### 5.3 선행지표 검증 프로세스

```
1. 가설 수립
   └── "X 지표가 Y일 후 Z 자산을 예측한다"

2. 데이터 수집
   ├── 최소 5년 히스토리
   ├── 일별/주별/월별 빈도
   └── 결측치 처리

3. 상관관계 분석
   ├── Rolling Correlation
   ├── Lead-Lag Analysis
   └── Cross-Correlation

4. Granger Causality Test
   ├── H0: X가 Y를 Granger-cause 하지 않음
   ├── p-value < 0.05 → 선행 관계 존재
   └── 최적 lag 결정

5. 실제 예측력 테스트
   ├── Out-of-sample backtest
   ├── Walk-forward validation
   └── 다양한 시장 국면에서 검증

6. 결론
   ├── 선행지표로 채택 → 시스템 통합
   ├── 추가 연구 필요 → 모니터링 지속
   └── 기각 → 폐기 또는 조건부 사용
```

### 5.4 뉴스/칼럼 데이터 사용 시 주의사항

> **User Insight**: "뉴스나 칼럼의 경우 시장을 왜곡 해석하거나
> 의도를 가지고 긍정, 부정을 부풀릴 수 있다"

```python
NEWS_SENTIMENT_CAUTION = {
    "biases": [
        "confirmation_bias",      # 기존 관점 확인 편향
        "recency_bias",           # 최근 사건 과대 해석
        "narrative_fallacy",      # 스토리에 맞춰 해석
        "sponsored_content",      # 광고성 기사
        "short_seller_reports",   # 공매도 세력 리포트
    ],

    "mitigation_strategies": {
        "use_price_first": True,              # 가격 데이터 우선
        "sentiment_as_contrarian": True,      # 극단적 심리 = 역발상 시그널
        "cross_validate_sources": True,       # 다중 소스 검증
        "weight_by_track_record": True,       # 과거 정확도로 가중치
        "delay_reaction": True,               # 즉각 반응 자제
    },

    "preferred_data_hierarchy": [
        "price_volume_data",      # 1순위: 가격/거래량
        "flow_data",              # 2순위: 자금 흐름
        "positioning_data",       # 3순위: 포지션 데이터
        "economic_data",          # 4순위: 경제 지표
        "sentiment_contrarian",   # 5순위: 심리 (역발상만)
        "news_as_filter",         # 최하위: 뉴스 (필터용만)
    ]
}
```

---

## 6. 피드백 루프 시스템

### 6.1 아키텍처

```
[시그널 생성]     [포트폴리오 생성]     [실행]     [성과 측정]
     │                  │                │              │
     ▼                  ▼                ▼              ▼
 ┌─────────┐      ┌──────────┐      ┌────────┐    ┌──────────┐
 │ Signals │ ──→  │ Portfolio│ ──→  │Execute │ →  │ Tracking │
 │   DB    │      │Candidates│      │   DB   │    │    DB    │
 └─────────┘      └──────────┘      └────────┘    └──────────┘
     │                  │                              │
     │                  │                              │
     │                  ▼                              ▼
     │            ┌──────────┐                  ┌──────────┐
     │            │ Actual   │ ←───────────────│ Compare  │
     │            │ Returns  │                  │ Pred/Act │
     │            └──────────┘                  └──────────┘
     │                                                │
     │                                                │
     ▼                                                ▼
┌─────────────────────────────────────────────────────────┐
│                   FEEDBACK LOOP                          │
│  - 시그널 정확도 업데이트                                │
│  - 가중치 조정                                           │
│  - 선행지표 유효성 재평가                                │
│  - 포트폴리오 전략 개선                                  │
└─────────────────────────────────────────────────────────┘
```

### 6.2 일일 프로세스

```
[장 시작 전] 18:00-22:00 KST
├── 1. 전일 성과 평가
│   ├── 예측 vs 실제 비교
│   ├── 시그널별 정확도 업데이트
│   └── DB 기록
│
├── 2. 금일 시그널 수집
│   ├── Regime Detector 실행
│   ├── Critical Path 확인
│   ├── FRED 데이터 업데이트
│   └── 시그널 DB 저장
│
├── 3. 포트폴리오 후보 생성
│   ├── 투자자 프로파일별 3-5개
│   ├── 예상 수익률/리스크 계산
│   └── 후보 DB 저장
│
└── 4. Pre-Market 모니터링
    └── 중요 뉴스/실적 확인

[정규장] 23:30-06:00 KST
├── 1. Opening Range 관찰 (첫 30분)
├── 2. 시그널 확인 후 실행 결정
├── 3. 실행 기록 DB 저장
└── 4. Power Hour 재평가

[장 마감 후] 06:00-10:00 KST
├── 1. 종가 기록
├── 2. 일일 성과 계산
├── 3. 세션별 분석
└── 4. 피드백 루프 실행
```

### 6.3 주간/월간 프로세스

```
[주간 리뷰] 매주 일요일
├── 시그널 정확도 주간 집계
├── 포트폴리오 성과 vs 벤치마크
├── 선행지표 유효성 체크
├── 가중치 미세 조정
└── 주간 리포트 생성

[월간 리뷰] 매월 1일
├── 전략별 성과 분석
├── 시장 국면별 분석
├── 선행지표 Granger 테스트 재실행
├── 모델 파라미터 최적화
└── 월간 리포트 생성
```

---

## 7. 구현 우선순위

### Phase 1: 기초 인프라 (1-2주)
- [ ] DB 테이블 생성 (`signals`, `portfolio_candidates`, `executions`)
- [ ] 시그널 저장 파이프라인
- [ ] 기본 성과 추적

### Phase 2: 세션별 분석 (1-2주)
- [ ] 시간대별 가격 데이터 수집
- [ ] `session_analysis` 테이블 구현
- [ ] 최적 실행 시간 분석

### Phase 3: 포트폴리오 생성 (2주)
- [ ] 투자자 프로파일 구현
- [ ] AI 시그널 → 포트폴리오 변환
- [ ] 후보 순위화 알고리즘

### Phase 4: 피드백 루프 (2주)
- [ ] 예측-실제 비교 자동화
- [ ] 시그널 정확도 추적
- [ ] 가중치 자동 조정

### Phase 5: 선행지표 연구 (지속)
- [ ] Granger Causality 자동 테스트
- [ ] 신규 지표 탐색
- [ ] 리포트 자동화

---

## 8. 참고: 데이터 소스

| 데이터 | 소스 | 빈도 | 비용 |
|--------|------|------|------|
| 주가/ETF | yfinance | 분봉/일봉 | 무료 |
| 경제지표 | FRED API | 일/주/월 | 무료 |
| VIX/옵션 | CBOE | 일봉 | 무료 |
| Fear & Greed | CNN/Alternative.me | 일봉 | 무료 |
| ARK Holdings | ARK 공식 | 일봉 | 무료 |
| 13F Filings | SEC EDGAR | 분기 | 무료 |
| Insider Trading | SEC Form 4 | 일봉 | 무료 |
| Pre/After Market | yfinance (제한적) | 분봉 | 무료 |
| Credit Spreads | FRED (ICE) | 일봉 | 무료 |
| Repo Rates | NY Fed | 일봉 | 무료 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-31 | 1.0 | 초안 작성 |

---

> **Next Step**: Phase 1 DB 테이블 생성 및 시그널 저장 파이프라인 구현
