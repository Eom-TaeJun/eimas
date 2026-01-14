# EIMAS 예측 검증 데이터베이스

> 실시간에서만 수집 가능한 데이터와 예측을 저장하여 정확도/경향을 분석

---

## 설계 원칙

### 저장하는 것
- 실행 시점에만 얻을 수 있는 데이터 (호가, VPIN, 실시간 괴리율)
- 모델/에이전트의 예측값 (레짐, 리스크, 포지션)
- 예측 시점의 컨텍스트 (신뢰도, 합의율)

### 저장하지 않는 것
- 종가, OHLCV (나중에 조회 가능)
- 확정된 경제지표 (FRED에서 재조회 가능)
- 정적 설정값

---

## 데이터베이스 구조

**파일**: `data/predictions.db`

```
predictions.db
├── regime_predictions      # 레짐 예측
├── risk_predictions        # 리스크 점수 예측
├── debate_predictions      # 멀티에이전트 토론 결과
├── vpin_snapshots          # 실시간 VPIN
├── stablecoin_snapshots    # 스테이블코인 상태
├── bubble_alerts           # 버블 경고
├── portfolio_snapshots     # 포트폴리오 추천
└── validation_log          # 검증 결과 로그
```

---

## 테이블 상세

### 1. regime_predictions (레짐 예측)

**왜 저장하나?**
- GMM 확률 분포는 실행 시점에만 계산됨
- Shannon Entropy는 당시 불확실성을 반영
- 나중에 "Bull 예측이 맞았는가?" 검증 필요

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 예측 시점 |
| predicted_regime | TEXT | Bull / Neutral / Bear |
| confidence | REAL | 0.0 ~ 1.0 |
| gmm_bull_prob | REAL | GMM Bull 확률 |
| gmm_neutral_prob | REAL | GMM Neutral 확률 |
| gmm_bear_prob | REAL | GMM Bear 확률 |
| shannon_entropy | REAL | 불확실성 지표 |
| trend | TEXT | Uptrend / Downtrend / Sideways |
| volatility | TEXT | Low / Medium / High |
| spy_price_at_prediction | REAL | 예측 시점 SPY 가격 |
| validated | INTEGER | 0=미검증, 1=검증완료 |
| actual_return_1d | REAL | T+1 실제 수익률 |
| actual_return_5d | REAL | T+5 실제 수익률 |
| actual_return_20d | REAL | T+20 실제 수익률 |
| validated_at | DATETIME | 검증 완료 시점 |

**검증 가능한 분석:**
- Bull 예측 후 5일 수익률 분포
- Entropy 높을 때 vs 낮을 때 예측 정확도 차이
- GMM 확률 > 80% vs < 60% 정확도 비교

---

### 2. risk_predictions (리스크 점수 예측)

**왜 저장하나?**
- 리스크 점수는 여러 컴포넌트의 조합 (Base + Micro + Bubble)
- 각 컴포넌트별 기여도 추적 필요
- "높은 리스크 예측 → 실제 낙폭" 캘리브레이션

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 예측 시점 |
| final_risk_score | REAL | 최종 리스크 (0-100) |
| base_score | REAL | CriticalPath 기본 점수 |
| microstructure_adj | REAL | 미세구조 조정 (±10) |
| bubble_adj | REAL | 버블 가산 (+0~15) |
| risk_level | TEXT | LOW / MEDIUM / HIGH |
| avg_liquidity_score | REAL | 유동성 점수 |
| bubble_status | TEXT | NONE / WATCH / WARNING / DANGER |
| vix_at_prediction | REAL | 예측 시점 VIX |
| validated | INTEGER | 0=미검증, 1=검증완료 |
| actual_max_drawdown_5d | REAL | T+5 최대 낙폭 |
| actual_max_drawdown_20d | REAL | T+20 최대 낙폭 |
| validated_at | DATETIME | 검증 완료 시점 |

**검증 가능한 분석:**
- 리스크 80+ 예측 후 실제 낙폭 평균
- Microstructure 조정이 예측력에 기여하는가?
- Bubble 경고 가산이 실제 하락과 상관있는가?

---

### 3. debate_predictions (토론 결과)

**왜 저장하나?**
- 멀티에이전트 합의 과정은 재현 불가 (LLM 확률성)
- FULL vs REF 모드 비교 데이터 축적
- 합의율과 예측 정확도 관계 분석

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 예측 시점 |
| full_mode_position | TEXT | BULLISH / BEARISH / NEUTRAL |
| ref_mode_position | TEXT | BULLISH / BEARISH / NEUTRAL |
| modes_agree | INTEGER | 1=일치, 0=불일치 |
| final_recommendation | TEXT | 최종 권고 |
| confidence | REAL | 0.0 ~ 1.0 |
| dissent_count | INTEGER | 반대 의견 수 |
| devils_advocate_1 | TEXT | 반대 논거 1 |
| devils_advocate_2 | TEXT | 반대 논거 2 |
| devils_advocate_3 | TEXT | 반대 논거 3 |
| spy_price_at_prediction | REAL | 예측 시점 SPY 가격 |
| validated | INTEGER | 0=미검증, 1=검증완료 |
| actual_direction_1d | TEXT | T+1 실제 방향 (UP/DOWN) |
| actual_direction_5d | TEXT | T+5 실제 방향 |
| actual_return_5d | REAL | T+5 실제 수익률 |
| validated_at | DATETIME | 검증 완료 시점 |

**검증 가능한 분석:**
- FULL/REF 일치 시 vs 불일치 시 정확도
- 신뢰도 80%+ 예측의 실제 적중률
- Devil's Advocate가 맞았던 비율

---

### 4. vpin_snapshots (실시간 VPIN)

**왜 저장하나?**
- VPIN은 체결 순서 기반 → 실시간만 가능
- Order flow 불균형은 나중에 재계산 불가
- VPIN 급등 후 가격 변동 패턴 분석

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 캡처 시점 |
| symbol | TEXT | BTCUSDT / ETHUSDT |
| vpin_1m | REAL | 1분 VPIN |
| vpin_5m | REAL | 5분 VPIN |
| vpin_15m | REAL | 15분 VPIN |
| alert_level | TEXT | normal / elevated / high / extreme |
| buy_volume | REAL | 매수 체결량 |
| sell_volume | REAL | 매도 체결량 |
| imbalance_ratio | REAL | (buy-sell)/(buy+sell) |
| price_at_capture | REAL | 캡처 시점 가격 |
| price_1h_later | REAL | 1시간 후 가격 (검증용) |
| price_change_1h | REAL | 1시간 수익률 |
| validated | INTEGER | 0=미검증, 1=검증완료 |

**검증 가능한 분석:**
- VPIN > 0.6 후 1시간 내 가격 변동 분포
- Alert level별 실제 급변 발생 빈도
- 매수/매도 불균형과 가격 방향 상관관계

---

### 5. stablecoin_snapshots (스테이블코인 상태)

**왜 저장하나?**
- 실시간 괴리율은 초 단위로 변동
- De-peg 사전 징후 패턴 학습
- Stress Test 예측 검증

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 캡처 시점 |
| usdt_price | REAL | Tether 가격 |
| usdc_price | REAL | USDC 가격 |
| dai_price | REAL | DAI 가격 |
| usdt_deviation | REAL | USDT 괴리율 (%) |
| usdc_deviation | REAL | USDC 괴리율 (%) |
| dai_deviation | REAL | DAI 괴리율 (%) |
| max_deviation | REAL | 최대 괴리율 |
| depeg_alert | INTEGER | 1=경고 발령 |
| stress_test_depeg_prob | REAL | Stress Test 예측 확률 |
| stress_test_expected_loss | REAL | 예상 손실액 |
| actual_depeg_24h | INTEGER | 24시간 내 실제 De-peg 발생 |
| validated | INTEGER | 0=미검증, 1=검증완료 |

**검증 가능한 분석:**
- Stress Test 확률과 실제 De-peg 상관관계
- 괴리율 0.5%+ 발생 후 24시간 내 추가 이탈 빈도
- USDT vs USDC vs DAI 안정성 비교

---

### 6. bubble_alerts (버블 경고)

**왜 저장하나?**
- Greenwood-Shleifer 신호는 당시 계산값
- Run-up, 변동성 Z-score는 윈도우 기준 변동
- "WARNING 후 실제 붕괴" 빈도 분석

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 경고 시점 |
| ticker | TEXT | 종목 |
| alert_level | TEXT | WATCH / WARNING / DANGER |
| runup_2y_pct | REAL | 2년 누적 수익률 (%) |
| volatility_zscore | REAL | 변동성 Z-score |
| issuance_growth | REAL | 주식 발행 증가율 |
| bubble_score | REAL | 종합 버블 점수 |
| price_at_alert | REAL | 경고 시점 가격 |
| validated | INTEGER | 0=미검증, 1=검증완료 |
| max_drawdown_30d | REAL | 30일 내 최대 낙폭 |
| max_drawdown_90d | REAL | 90일 내 최대 낙폭 |
| crash_occurred | INTEGER | 1=-20% 이상 하락 발생 |
| validated_at | DATETIME | 검증 완료 시점 |

**검증 가능한 분석:**
- WARNING 발령 후 30일 내 -20% 발생률
- Run-up > 100% 종목의 평균 후속 낙폭
- 버블 점수와 실제 낙폭 상관계수

---

### 7. portfolio_snapshots (포트폴리오 추천)

**왜 저장하나?**
- GC-HRP 비중은 당시 상관관계 기반
- 추천 포트폴리오 vs 벤치마크 성과 비교
- 클러스터링 효과 검증

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| timestamp | DATETIME | 추천 시점 |
| portfolio_json | TEXT | {ticker: weight} JSON |
| top5_tickers | TEXT | 상위 5개 종목 |
| top5_weights | TEXT | 상위 5개 비중 |
| cluster_count | INTEGER | 클러스터 수 |
| allocation_rationale | TEXT | 배분 근거 |
| spy_weight | REAL | SPY 비중 |
| tlt_weight | REAL | TLT 비중 |
| validated | INTEGER | 0=미검증, 1=검증완료 |
| portfolio_return_5d | REAL | 5일 포트폴리오 수익률 |
| portfolio_return_20d | REAL | 20일 포트폴리오 수익률 |
| spy_return_5d | REAL | 5일 SPY 수익률 (벤치마크) |
| spy_return_20d | REAL | 20일 SPY 수익률 |
| outperformed_5d | INTEGER | 1=SPY 초과 |
| outperformed_20d | INTEGER | 1=SPY 초과 |
| validated_at | DATETIME | 검증 완료 시점 |

**검증 가능한 분석:**
- GC-HRP가 SPY 대비 초과수익 달성 빈도
- 클러스터 수와 성과 관계
- 특정 종목 과대비중 시 성과 영향

---

### 8. validation_log (검증 로그)

**왜 저장하나?**
- 모든 검증 이력 중앙 관리
- 검증 배치 실행 추적
- 예측 유형별 정확도 집계

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | PK |
| validated_at | DATETIME | 검증 실행 시점 |
| prediction_type | TEXT | regime / risk / debate / ... |
| prediction_id | INTEGER | 원본 예측 ID |
| prediction_timestamp | DATETIME | 예측 시점 |
| predicted_value | TEXT | 예측값 |
| actual_value | TEXT | 실제값 |
| is_correct | INTEGER | 1=정확, 0=오류 |
| accuracy_score | REAL | 정확도 점수 (연속형) |
| horizon_days | INTEGER | 검증 기간 (1/5/20일) |
| notes | TEXT | 비고 |

**검증 가능한 분석:**
- 예측 유형별 전체 정확도
- 시간에 따른 정확도 추이 (개선/악화)
- 특정 시장 환경에서의 정확도 변화

---

## 검증 주기

| 예측 유형 | 검증 시점 | 검증 방법 |
|----------|----------|----------|
| 레짐 예측 | T+1, T+5, T+20 | SPY 수익률 부호 비교 |
| 리스크 점수 | T+5, T+20 | 최대 낙폭과 점수 상관 |
| 토론 결과 | T+1, T+5 | 방향 적중 여부 |
| VPIN | T+1h | 가격 변동폭 비교 |
| 스테이블코인 | T+24h | De-peg 발생 여부 |
| 버블 경고 | T+30d, T+90d | -20% 하락 발생 여부 |
| 포트폴리오 | T+5, T+20 | 수익률 vs SPY |

---

## 활용 예시

### 1. 레짐 예측 정확도 리포트

```sql
SELECT
    predicted_regime,
    COUNT(*) as total,
    SUM(CASE WHEN actual_return_5d > 0 AND predicted_regime = 'Bull' THEN 1
             WHEN actual_return_5d < 0 AND predicted_regime = 'Bear' THEN 1
             ELSE 0 END) as correct,
    ROUND(100.0 * correct / total, 1) as accuracy_pct
FROM regime_predictions
WHERE validated = 1
GROUP BY predicted_regime;
```

### 2. 리스크 캘리브레이션

```sql
SELECT
    CASE
        WHEN final_risk_score < 30 THEN 'LOW (0-30)'
        WHEN final_risk_score < 60 THEN 'MEDIUM (30-60)'
        ELSE 'HIGH (60+)'
    END as risk_bucket,
    AVG(actual_max_drawdown_5d) as avg_drawdown,
    COUNT(*) as samples
FROM risk_predictions
WHERE validated = 1
GROUP BY risk_bucket;
```

### 3. VPIN 유효성 검증

```sql
SELECT
    alert_level,
    AVG(ABS(price_change_1h)) as avg_price_move,
    COUNT(*) as samples
FROM vpin_snapshots
WHERE validated = 1
GROUP BY alert_level
ORDER BY avg_price_move DESC;
```

---

## 구현 우선순위

| 순위 | 테이블 | 이유 |
|------|--------|------|
| 1 | regime_predictions | 핵심 예측, 매일 생성 |
| 2 | debate_predictions | 토론 결과 추적 필수 |
| 3 | risk_predictions | 리스크 캘리브레이션 중요 |
| 4 | vpin_snapshots | 실시간 모니터링 시 생성 |
| 5 | portfolio_snapshots | 포트폴리오 성과 추적 |
| 6 | bubble_alerts | 장기 검증 (30-90일) |
| 7 | stablecoin_snapshots | 크립토 리스크 추적 |
| 8 | validation_log | 마지막 통합 |

---

## 예상 데이터 증가량

| 테이블 | 생성 빈도 | 월간 예상 레코드 |
|--------|----------|-----------------|
| regime_predictions | 1회/일 | ~22 |
| debate_predictions | 1회/일 | ~22 |
| risk_predictions | 1회/일 | ~22 |
| vpin_snapshots | 1회/분 (실시간 시) | ~3,600/시간 |
| portfolio_snapshots | 1회/일 | ~22 |
| bubble_alerts | 이벤트 기반 | ~5-10 |
| stablecoin_snapshots | 1회/시간 | ~720 |

---

*EIMAS v2.1.2 - Prediction Validation Database*
