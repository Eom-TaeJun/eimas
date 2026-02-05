# EIMAS Backtest & Database Architecture

> 백테스트 엔진과 데이터베이스 통합 구조 문서
> **Version**: v1.0 (2026-02-05)

---

## 📊 전체 데이터 플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EIMAS Pipeline                              │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: Data Collection
  ├─ FRED API → fred_data
  ├─ yfinance → market_data (24 tickers)
  ├─ Crypto/RWA → crypto_data
  └─ Korea → korea_data

Phase 2: Analysis
  ├─ RegimeAnalyzer → regime
  ├─ CriticalPathAnalyzer → risk_score
  ├─ Microstructure → liquidity_adjustment
  ├─ BubbleDetector → bubble_adjustment
  ├─ GeniusActMacro → genius_act_regime
  ├─ GraphClusteredPortfolio → portfolio_weights
  ├─ AllocationEngine → allocation_result
  └─ RebalancingPolicy → rebalance_decision

Phase 3: AI Debate
  ├─ Full Mode (365d) → full_mode_position
  ├─ Reference Mode (90d) → reference_mode_position
  └─ Dual Mode Analyzer → consensus

Phase 5: Storage
  ├─ JSON → outputs/eimas_YYYYMMDD_HHMMSS.json
  ├─ Markdown → outputs/eimas_YYYYMMDD_HHMMSS.md
  └─ **Database** ↓

Phase 6.1: Backtest (--backtest) ← 여기!
  ├─ BacktestEngine.run() → BacktestResult
  └─ TradingDB.save_backtest_run() → trading.db
```

---

## 🗄️ 데이터베이스 구조

### 1. **eimas.db** (Core Database)

**위치**: `data/eimas.db`
**관리**: `core/database.py` (DatabaseManager)

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|----------|
| `ark_holdings` | ARK ETF 보유종목 | date, etf, ticker, weight, shares |
| `ark_weight_changes` | 비중 변화 이력 | date, ticker, weight_change, change_type |
| `etf_analysis` | ETF 분석 결과 | date, analysis_type, data (JSON) |
| `market_regime` | 시장 레짐 이력 | date, sentiment, cycle_phase, risk_appetite_score |
| `signals` | 생성된 신호 | date, signal_type, strength, ticker |
| `actions` | 권고 액션 | date, action_type, portfolio_id |

---

### 2. **events.db** (Event Database)

**위치**: `data/events.db`
**관리**: `lib/event_db.py` (EventDatabase)

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|----------|
| `detected_events` | 감지된 시장 이벤트 | event_date, event_type, ticker, importance, value |
| `event_predictions` | 이벤트 예측 | prediction_id, event_type, event_date, predicted_impact |
| `market_snapshots` | 시장 상태 스냅샷 | snapshot_date, vix, liquidity, regime |
| `prediction_outcomes` | 예측 정확도 추적 | prediction_id, actual_outcome, accuracy |

---

### 3. **trading.db** (Trading & Backtest) ⭐

**위치**: `data/trading.db`
**관리**: `lib/trading_db.py` (TradingDB)

#### 3.1 실시간 트레이딩 테이블

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|----------|
| `signals` | 시그널 기록 | timestamp, signal_source, signal_action, conviction |
| `portfolio_candidates` | 포트폴리오 후보 | profile_type, allocations (JSON), expected_sharpe |
| `executions` | 실행 기록 | portfolio_id, ticker, executed_price, commission |
| `performance_tracking` | 성과 추적 | portfolio_id, date, predicted/actual returns, mape |
| `signal_performance` | 시그널 성과 | signal_id, return_1d/5d/20d, signal_accuracy |
| `session_analysis` | 세션별 분석 | date, pre_market_return, power_hour_return |

#### 3.2 백테스트 테이블 ⭐⭐

| 테이블 | 설명 | 주요 컬럼 |
|--------|------|----------|
| **`backtest_runs`** | 백테스트 실행 결과 | strategy_name, start/end_date, sharpe, max_dd, win_rate |
| **`backtest_trades`** | 백테스트 거래 내역 | run_id, entry/exit_date, pnl, holding_days |
| **`walk_forward_results`** | Walk-Forward 검증 | run_id, fold_number, in/out_sample_sharpe |

---

## 🔧 백테스트 엔진 구조

### BacktestEngine 클래스

**파일**: `lib/backtest.py` (~529 lines)

```python
@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "monthly"  # daily/weekly/monthly/quarterly
    transaction_cost_bps: float = 10.0   # 거래비용 10bp
    slippage_bps: float = 5.0            # 슬리피지 5bp
    train_period_days: int = 252
    test_period_days: int = 63
    use_rolling_window: bool = True

@dataclass
class BacktestMetrics:
    # Returns
    total_return: float
    annualized_return: float

    # Risk
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Downside
    var_95: float
    cvar_95: float
    downside_deviation: float

    # Win rate
    win_rate: float
    profit_factor: float

    # Trading
    num_trades: int
    turnover_annual: float
    total_transaction_costs: float

class BacktestEngine:
    def run(
        self,
        prices: pd.DataFrame,
        allocation_func: Callable,
        regime_func: Optional[Callable] = None
    ) -> BacktestResult:
        # 1. 리밸런싱 날짜 계산
        # 2. 각 리밸런싱 시점에서:
        #    a. allocation_func 호출 → 새 비중
        #    b. 거래비용 계산 (turnover × 15bp)
        #    c. 포트폴리오 가치 업데이트
        # 3. 성과 지표 계산
        # 4. BacktestResult 반환
```

---

## 🔗 통합 흐름도

### main.py 백테스트 실행

```python
# main.py line 844-918
def _run_backtest(result: EIMASResult, market_data: Dict, enable: bool):
    if not enable:
        return

    # 1. 가격 데이터 준비
    prices = pd.DataFrame({
        ticker: market_data[ticker]['close']
        for ticker in market_data.keys()
    })

    # 2. 백테스트 설정
    config = BacktestConfig(
        start_date=str(prices.index[252]),  # 1년 후부터
        end_date=str(prices.index[-1]),
        rebalance_frequency='monthly',
        transaction_cost_bps=10,
        initial_capital=1_000_000
    )

    # 3. 배분 전략 (현재 포트폴리오 비중 또는 동일가중)
    def allocation_strategy(prices_window):
        if result.portfolio_weights:
            return result.portfolio_weights
        else:
            n = len(prices_window.columns)
            return {ticker: 1/n for ticker in prices_window.columns}

    # 4. 백테스트 실행
    engine = BacktestEngine(config)
    backtest_result = engine.run(prices, allocation_strategy)

    # 5. 결과 저장 (EIMASResult에 임베드)
    result.backtest_metrics = backtest_result.metrics.to_dict()

    # ⚠️ 현재 누락: DB 저장!
```

### ⚠️ 현재 문제: DB 저장 누락

**main.py는 백테스트 결과를 `result.backtest_metrics`에만 저장**
→ JSON/Markdown에는 포함되지만 **DB에는 저장 안 됨**

---

## 🛠️ 개선 제안

### 1. main.py에 DB 저장 추가

```python
# main.py _run_backtest() 함수 수정 (line 915 이후)
def _run_backtest(result: EIMASResult, market_data: Dict, enable: bool):
    # ... (기존 코드) ...

    backtest_result = engine.run(prices, allocation_strategy)
    result.backtest_metrics = backtest_result.metrics.to_dict()

    # ✅ 추가: DB 저장
    from lib.trading_db import TradingDB

    db = TradingDB()

    # BacktestResult → Dict 변환
    backtest_dict = {
        'strategy_name': 'EIMAS_Portfolio',
        'start_date': config.start_date,
        'end_date': config.end_date,
        'initial_capital': config.initial_capital,
        'final_capital': backtest_result.portfolio_values.iloc[-1],
        'total_return': backtest_result.metrics.total_return,
        'annual_return': backtest_result.metrics.annualized_return,
        'benchmark_return': 0.0,  # TODO: SPY 벤치마크 추가
        'alpha': 0.0,
        'volatility': backtest_result.metrics.annualized_volatility,
        'max_drawdown': backtest_result.metrics.max_drawdown,
        'max_drawdown_duration': backtest_result.metrics.max_drawdown_duration,
        'sharpe_ratio': backtest_result.metrics.sharpe_ratio,
        'sortino_ratio': backtest_result.metrics.sortino_ratio,
        'calmar_ratio': backtest_result.metrics.calmar_ratio,
        'total_trades': backtest_result.metrics.num_trades,
        'winning_trades': 0,  # TODO: 승/패 분리 로직 추가
        'losing_trades': 0,
        'win_rate': backtest_result.metrics.win_rate,
        'avg_win': backtest_result.metrics.avg_win,
        'avg_loss': backtest_result.metrics.avg_loss,
        'profit_factor': backtest_result.metrics.profit_factor,
        'avg_holding_days': 30,  # monthly rebalance
        'total_commission': backtest_result.metrics.total_transaction_costs,
        'total_slippage': 0.0,
        'total_short_cost': 0.0,
        'parameters': {
            'rebalance_frequency': config.rebalance_frequency,
            'transaction_cost_bps': config.transaction_cost_bps,
            'initial_capital': config.initial_capital
        },
        'trades': []  # TODO: 개별 거래 기록 추가
    }

    run_id = db.save_backtest_run(backtest_dict)
    print(f"     DB Saved: Run ID {run_id}")
```

---

### 2. EIMASResult에 run_id 추가

```python
@dataclass
class EIMASResult:
    # ... (기존 필드) ...

    backtest_metrics: Optional[Dict] = None
    backtest_run_id: Optional[int] = None  # ✅ 추가
```

---

### 3. 백테스트 조회 API 추가

```python
# api/routes/backtest.py (신규 생성)
from fastapi import APIRouter
from lib.trading_db import TradingDB

router = APIRouter(prefix="/backtest", tags=["backtest"])

@router.get("/runs")
async def get_backtest_runs(strategy: str = None, limit: int = 50):
    db = TradingDB()
    runs = db.get_backtest_runs(strategy_name=strategy, limit=limit)
    return runs

@router.get("/runs/{run_id}")
async def get_backtest_detail(run_id: int):
    db = TradingDB()
    runs = db.get_backtest_runs()
    run = next((r for r in runs if r['id'] == run_id), None)

    if not run:
        raise HTTPException(404, "Run not found")

    trades = db.get_backtest_trades(run_id)
    return {
        "run": run,
        "trades": trades
    }

@router.get("/performance/{strategy}")
async def get_strategy_performance(strategy: str):
    db = TradingDB()
    history = db.get_strategy_performance_history(strategy)
    return history
```

---

## 📈 데이터베이스 ERD

```
┌────────────────────┐
│  backtest_runs     │
├────────────────────┤
│ id (PK)            │
│ strategy_name      │◄─────┐
│ start_date         │      │
│ sharpe_ratio       │      │ 1:N
│ max_drawdown       │      │
│ ...                │      │
└────────────────────┘      │
                            │
                            │
┌────────────────────┐      │
│ backtest_trades    │      │
├────────────────────┤      │
│ id (PK)            │      │
│ run_id (FK) ───────┼──────┘
│ entry_date         │
│ exit_date          │
│ pnl                │
│ ...                │
└────────────────────┘

┌────────────────────┐
│ walk_forward_results│
├────────────────────┤
│ id (PK)            │
│ run_id (FK) ───────┼──────┐
│ fold_number        │      │
│ in_sample_sharpe   │      │ 1:N
│ out_sample_sharpe  │      │
│ ...                │      │
└────────────────────┘      │
                            │
                            └──(backtest_runs)
```

---

## 🎯 사용 예시

### CLI로 백테스트 실행 및 조회

```bash
# 1. 백테스트 실행
python main.py --backtest

# 출력:
# [Phase 6.1] Running Backtest Engine...
#   ✅ Backtest Complete:
#      Sharpe: 1.45
#      Max DD: -12.3%
#      VaR 95%: -1.82%
#      DB Saved: Run ID 17

# 2. Python으로 DB 조회
python -c "
from lib.trading_db import TradingDB
db = TradingDB()

# 최근 실행 조회
runs = db.get_backtest_runs(limit=5)
for run in runs:
    print(f'{run[\"id\"]}: {run[\"strategy_name\"]} - Sharpe {run[\"sharpe_ratio\"]:.2f}')

# 특정 실행의 거래 내역
trades = db.get_backtest_trades(run_id=17)
print(f'Total trades: {len(trades)}')
"

# 3. API로 조회 (FastAPI 서버 필요)
curl http://localhost:8000/backtest/runs
curl http://localhost:8000/backtest/runs/17
```

---

## 📋 체크리스트

### ✅ 완료
- [x] BacktestEngine 클래스 구현 (`lib/backtest.py`)
- [x] DB 스키마 정의 (backtest_runs, backtest_trades)
- [x] TradingDB.save_backtest_run() 메서드
- [x] main.py _run_backtest() 함수

### ⚠️ 누락
- [ ] main.py에서 DB 저장 호출 (line 915)
- [ ] EIMASResult.backtest_run_id 필드 추가
- [ ] 개별 거래 기록 (backtest_trades) 저장
- [ ] SPY 벤치마크 비교 로직
- [ ] 승/패 거래 분리 로직

### 🔮 향후 개선
- [ ] FastAPI 백테스트 엔드포인트 (`api/routes/backtest.py`)
- [ ] 대시보드 백테스트 차트 (frontend)
- [ ] Walk-Forward Validation 저장
- [ ] Regime별 성과 분해 저장
- [ ] 백테스트 비교 UI (여러 전략 비교)

---

## 🔍 핵심 포인트

1. **백테스트 엔진은 완성** (`lib/backtest.py`)
2. **DB 테이블도 준비** (`lib/trading_db.py`)
3. **main.py에서 DB 저장만 추가하면 완성** (15줄 코드)
4. **API/대시보드 연동은 선택 사항** (나중에 추가 가능)

---

*마지막 업데이트: 2026-02-05*
*문의: EIMAS 프로젝트 담당자*
