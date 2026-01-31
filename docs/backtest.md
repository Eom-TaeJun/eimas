# EIMAS Backtesting Engine - Analysis & Improvement Plan

> 작성일: 2026-01-31
> 버전: v1.0

---

## 1. 현재 시스템 구조

### 1.1 파일 구성

| 파일 | 상태 | 줄수 | 역할 |
|------|------|------|------|
| `lib/event_backtester.py` | Active | 636 | 경제 이벤트(FOMC/CPI/NFP) 전후 가격 분석 |
| `scripts/run_backtest.py` | Active | 101 | CLI 백테스트 실행기 |
| `lib/trading_db.py` | Active | 871 | 시그널/포트폴리오/실행 DB |
| `archive/deprecated/backtester.py` | Deprecated | 1037 | 전체 백테스팅 엔진 (버그 있음) |
| `archive/deprecated/backtest_engine.py` | Deprecated | 690 | Walk-Forward 검증 엔진 |

### 1.2 DB 스키마 (trading_db.py)

```sql
-- 현재 테이블
signals              -- 시그널 (source, action, conviction)
portfolio_candidates -- 포트폴리오 후보 (allocations, expected_return)
executions           -- 실행 기록 (price, slippage, commission)
performance_tracking -- 예측/실제 비교
signal_performance   -- 시그널별 성과
session_analysis     -- 세션별 분석 (pre-market, power-hour 등)
```

---

## 2. 발견된 문제점

### 2.1 금융적 문제 (Financial Issues)

#### 2.1.1 복리 계산 버그 (Critical)

**위치:** `archive/deprecated/backtester.py:452, 483`

```python
# Line 483: Entry 시
capital -= invest_amount + cost  # 자본에서 투자금 차감

# Line 452: Exit 시
capital += shares * entry_price + pnl  # 원금 + 손익 추가
```

**문제점:**
- 수익 발생 → `capital` 증가 → 다음 거래에서 `capital * position_size`로 더 큰 포지션
- 기하급수적 복리 효과 → 비현실적 수익률 (8,360%)
- 39% 승률로 8,360% 수익은 **통계적으로 불가능**

**수정 방향:**
```python
# Option A: 고정 포지션 (Fixed Position)
initial_capital = 100000
position_value = initial_capital * position_size  # 항상 동일

# Option B: Kelly Criterion
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = min(kelly_fraction, 0.25)  # 최대 25% 제한
```

#### 2.1.2 Look-ahead Bias

**위치:** `signal_func(data: pd.DataFrame, idx: int)`

```python
# 현재: 전체 DataFrame을 전달
def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
    # idx 이후 데이터에 접근 가능 (실수 유발)
    close = data['Close'].values
```

**수정 방향:**
```python
# data[:idx+1]만 전달하여 미래 데이터 접근 원천 차단
def signal_func(data: pd.DataFrame) -> SignalType:
    # data는 현재까지의 데이터만 포함
```

#### 2.1.3 Short Selling 비용 누락

| 비용 항목 | 현재 반영 | 현실 |
|----------|----------|------|
| Borrow Fee | X | 연 0.3% ~ 50%+ (종목별) |
| Hard-to-Borrow | X | 일부 종목 숏 불가 |
| Short Squeeze | X | 급격한 손실 가능 |
| Margin Interest | X | 증거금 이자 |

#### 2.1.4 Transaction Cost 과소평가

```python
# 현재 설정
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005   # 0.05%

# 누락된 비용
- Market Impact (대량 주문 시 가격 영향)
- 급등/급락 시 슬리피지 증가
- 유동성 낮은 시간대 스프레드 확대
```

---

### 2.2 경제학적 문제 (Econometric Issues)

#### 2.2.1 In-Sample Overfitting

**현재 상태:**
- 2020-2024 전체 기간으로 훈련 + 테스트 동시 수행
- 동일 데이터로 파라미터 최적화 → 과적합

**올바른 방법:**
```
Walk-Forward Validation:

Period 1: Train 2020-2021, Test 2022-Q1
Period 2: Train 2020-2022Q1, Test 2022-Q2
Period 3: Train 2020-2022Q2, Test 2022-Q3
...
```

#### 2.2.2 레짐 의존성 (Regime Dependence)

**2020-2024 시장 특성:**
| 기간 | 이벤트 | 수익률 |
|------|--------|--------|
| 2020.03 | COVID 폭락 | -34% |
| 2020.04-2021.12 | 유동성 랠리 | +120% |
| 2022 | 금리 인상 | -20% |
| 2023-2024 | AI 랠리 | +50% |

이 기간에 최적화된 전략은 다른 시장 환경에서 실패 가능성 높음.

#### 2.2.3 Survivorship Bias

- SPY만 테스트 (생존한 대형 ETF)
- 상폐된 종목, 실패한 ETF 미포함
- 실제 투자 가능 유니버스 미반영

---

### 2.3 논리적 문제 (Logical Issues)

#### 2.3.1 Short PnL 계산 단순화

```python
# 현재 (Line 394)
position_value = shares * (2 * entry_price - close)

# 정확한 계산
initial_margin = shares * entry_price * margin_rate
unrealized_pnl = (entry_price - current_price) * shares
position_value = initial_margin + unrealized_pnl
```

#### 2.3.2 이벤트-전략 분리

- `event_backtester.py`: FOMC/CPI/NFP 이벤트 분석
- `backtester.py`: 기술적 전략 백테스트
- **두 시스템이 연결되지 않음**

#### 2.3.3 DB 저장 미흡

- 백테스트 결과가 JSON 파일로만 저장
- 파라미터 조합 기록 없음
- Walk-forward 검증 결과 저장 없음

---

## 3. 백테스트 결과 분석

### 3.1 기존 결과 (2020-2024, 버그 포함)

| 전략 | 총 수익률 | 연 수익률 | Sharpe | MDD | 승률 |
|------|----------|----------|--------|-----|------|
| EIMAS_Regime | +8,360% | +143% | 1.85 | 3.5% | 39% |
| Multi_Factor | - | - | - | - | - |
| MA_Crossover | - | - | - | - | - |

**분석:**
- 39% 승률, 1.26 Profit Factor로 8,360% 수익은 불가능
- 복리 버그로 인한 비현실적 결과

### 3.2 예상 수정 후 결과

현실적인 기대값 (고정 포지션, 0.3 사이즈):
- 연 수익률: 10-25%
- Sharpe: 0.5-1.5
- MDD: 10-20%

---

## 4. 개선 계획

### 4.1 Phase 1: 핵심 버그 수정 (Priority: Critical)

- [ ] 복리 계산 버그 수정 (고정 포지션)
- [ ] Look-ahead bias 방지 (데이터 슬라이싱)
- [ ] 백테스트 기간 변경 (2024-09 ~ 현재)

### 4.2 Phase 2: DB 스키마 확장

```sql
-- 추가할 테이블
CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY,
    strategy_name TEXT,
    start_date DATE,
    end_date DATE,
    initial_capital REAL,
    final_capital REAL,
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    parameters JSON,
    created_at DATETIME
);

CREATE TABLE backtest_trades (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    entry_date DATE,
    exit_date DATE,
    ticker TEXT,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE walk_forward_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    fold_number INTEGER,
    train_start DATE,
    train_end DATE,
    test_start DATE,
    test_end DATE,
    in_sample_sharpe REAL,
    out_sample_sharpe REAL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);
```

### 4.3 Phase 3: Transaction Cost 개선

```python
class TransactionCostModel:
    def __init__(self):
        self.base_commission = 0.001  # 0.1%
        self.base_slippage = 0.0005   # 0.05%
        self.short_borrow_rate = 0.003  # 연 0.3% (기본)

    def calculate_slippage(self, volume_ratio: float) -> float:
        """거래량 비율에 따른 슬리피지"""
        if volume_ratio < 0.01:
            return self.base_slippage
        elif volume_ratio < 0.05:
            return self.base_slippage * 2
        else:
            return self.base_slippage * 5  # 대량 주문

    def calculate_short_cost(self, days: int, hard_to_borrow: bool) -> float:
        """숏 비용 계산"""
        rate = self.short_borrow_rate * 10 if hard_to_borrow else self.short_borrow_rate
        return rate * days / 365
```

### 4.4 Phase 4: Walk-Forward Validation

```python
def walk_forward_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    train_window: int = 252,  # 1년
    test_window: int = 63,    # 3개월
    step: int = 21            # 1개월
) -> List[WalkForwardResult]:
    results = []

    for i in range(train_window, len(data) - test_window, step):
        train_data = data.iloc[i-train_window:i]
        test_data = data.iloc[i:i+test_window]

        # Train (파라미터 최적화)
        optimized_params = optimize_strategy(strategy, train_data)

        # Test (OOS 검증)
        result = backtest(strategy, test_data, optimized_params)
        results.append(result)

    return results
```

---

## 5. 테스트 기간 정책

### 5.1 기본 기간: 2024-09-01 ~ 현재

**선택 이유:**
- Fed 금리 인하 시작 (2024-09-18: -50bp)
- 현재 시장 레짐에 가장 관련성 높음
- 충분한 데이터 (~5개월)

### 5.2 확장 테스트

| 기간 | 목적 | 특성 |
|------|------|------|
| 2024-09 ~ 현재 | 기본 테스트 | 금리 인하 사이클 |
| 2023-01 ~ 2024-08 | 검증 | 금리 동결 기간 |
| 2022-01 ~ 2022-12 | 스트레스 | 금리 인상 기간 |
| 2020-03 ~ 2020-12 | 극단 | COVID 폭락/회복 |

---

## 6. 성공 기준

### 6.1 수익률 목표 (현실적)

| 지표 | 목표 | 허용 범위 |
|------|------|----------|
| 연 수익률 | 12-20% | 8-30% |
| Sharpe Ratio | 1.0 | 0.7-1.5 |
| Max Drawdown | < 15% | < 20% |
| Win Rate | 45-55% | 40-60% |

### 6.2 검증 기준

- [ ] In-Sample vs Out-of-Sample 성과 차이 < 30%
- [ ] 최소 3개 다른 시장 레짐에서 수익
- [ ] 거래 비용 반영 후에도 수익

---

## 7. 실행 결과 (2026-01-31)

### 7.1 Short Position 버그 수정 결과

| 지표 | 수정 전 (버그) | 수정 후 | 비고 |
|------|---------------|---------|------|
| 기간 | 2024-09 ~ 현재 | 2024-09 ~ 현재 | 동일 |
| 총 수익률 | +91.1% | **+1.14%** | 버그로 89.9%p 과대평가 |
| 연 수익률 | +58.4% | **+0.80%** | 현실적 |
| Alpha | +63.5% | **-26.5%** | 벤치마크 대비 저조 |
| Sharpe | 1.43 | **0.19** | 현실적 |
| MDD | 2.67% | **5.92%** | 현실적 |

**버그 원인**: Short position Entry에서 `capital -= commission`만 차감하고 원금(invest_amount)을 차감하지 않음. Exit에서 원금 + PnL이 추가되어 Short 거래마다 "무에서 유"가 발생.

### 7.2 Walk-Forward Validation 결과

**설정**:
- 기간: 2022-01-01 ~ 2025-12-31 (4년)
- Train: 12개월, Test: 3개월
- 총 12 Folds

**결과**:

| 지표 | 값 | 기준 | 판정 |
|------|-----|------|------|
| Avg In-Sample Sharpe | 0.18 | - | - |
| Avg Out-of-Sample Sharpe | 0.40 | > 0.5 | FAIL |
| Avg Degradation | +69.4% | < 30% | FAIL |
| OOS Sharpe Range | [-3.02, 4.09] | - | 매우 불안정 |
| Cumulative OOS Return | +1.9% | - | 4년간 미미 |
| **Robustness Check** | **FAIL** | - | 과적합 |

**Fold 상세**:

| Fold | Train Period | IS Sharpe | OOS Sharpe | Degradation |
|------|-------------|-----------|------------|-------------|
| 5 | 2022-12 ~ 2023-12 | 1.04 | 1.45 | -39.8% (Good) |
| 6 | 2023-03 ~ 2024-03 | 2.10 | 4.09 | -94.4% (Good) |
| 8 | 2023-09 ~ 2024-09 | 0.55 | 0.56 | -1.2% (Good) |
| 1 | 2022-01 ~ 2022-12 | -0.38 | -2.54 | +561.7% (Bad) |
| 9 | 2023-12 ~ 2024-12 | 0.23 | -2.07 | +1006.8% (Bad) |

**결론**: EIMAS_Regime 전략은 **과적합** 상태. 특정 시장 레짐(2023-2024 상승장)에서만 잘 작동하며, 다른 레짐에서는 성과 불안정.

### 7.3 권장 개선 방향

1. **전략 단순화**: 파라미터 수 줄이기 (현재 4개 → 2개)
2. **레짐 필터링**: Bear 시장에서는 현금 보유
3. **앙상블**: Multi_Factor + MA_Crossover 결합
4. **자산 다변화**: SPY 외 QQQ, TLT 추가 테스트

---

## 8. 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-31 | v1.0 | 초기 분석 및 계획 수립 |
| 2026-01-31 | v1.1 | Short 버그 수정, Walk-Forward 구현 및 실행 |

---

*마지막 업데이트: 2026-01-31*
