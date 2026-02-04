# EIMAS Backtest System Guide

## 개요

EIMAS v2.2.2의 새로운 모듈형 백테스트 시스템 사용 가이드입니다.

## 시스템 구조

```
lib/backtest/
├── enums.py          # RebalanceFrequency, BacktestMode
├── schemas.py        # BacktestConfig, BacktestMetrics, BacktestResult
├── metrics.py        # 성과 지표 계산 함수
├── engine.py         # BacktestEngine (메인 엔진)
├── utils.py          # compare_strategies, generate_report
└── __init__.py       # 패키지 export
```

## 주요 기능

### 1. **Out-of-Sample Testing**
- Train/Test split 지원
- Rolling window analysis
- Walk-forward validation

### 2. **거래 비용 반영**
- Transaction cost (기본 10bp)
- Slippage (기본 5bp)
- Turnover 계산 및 제약

### 3. **성과 지표 (15개)**
- **Returns**: Total, Annualized, Cumulative
- **Risk**: Volatility, Max Drawdown, Downside Deviation
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega
- **Downside Risk**: VaR 95%, CVaR 95%
- **Trading**: Win Rate, Profit Factor, Num Trades, Turnover

### 4. **Regime별 성과 분해**
- Bull/Neutral/Bear 레짐별 수익률 분리 분석

## 사용법

### Step 1: 과거 데이터 수집 (12개월)

```bash
# 기본 실행 (2025-02-04 ~ 2026-02-04)
python scripts/prepare_historical_data.py

# 커스텀 기간
python scripts/prepare_historical_data.py \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --output data/backtest_custom.parquet
```

**출력**: `data/backtest_historical.parquet`
- FRED 데이터: RRP, TGA, WRESBAL, EFFR
- 시장 데이터: 24개 티커 (SPY, QQQ, TLT, GLD 등)
- 크립토 & RWA: BTC-USD, ETH-USD, ONDO-USD, PAXG-USD, COIN

### Step 2: 백테스트 실행

#### 2.1 기본 예제 (Equa Weight)

```python
from lib.backtest import BacktestEngine, BacktestConfig
import pandas as pd

# Load data
data = pd.read_parquet('data/backtest_historical.parquet')

# Remove FRED columns
market_prices = data.drop(columns=['RRP', 'TGA', 'WRESBAL', 'EFFR'], errors='ignore')

# Config
config = BacktestConfig(
    start_date='2025-06-01',  # Skip first 4 months for warmup
    end_date='2026-02-04',
    rebalance_frequency='monthly',
    transaction_cost_bps=10.0
)

# Strategy
def equal_weight(prices: pd.DataFrame):
    n = len(prices.columns)
    return {ticker: 1/n for ticker in prices.columns}

# Run
engine = BacktestEngine(config)
result = engine.run(market_prices, equal_weight)

# Print
print(result.summary())
print(f"Target Met: {result.metrics.meets_targets()}")
```

#### 2.2 EIMAS 전략 (Risk Parity)

```python
from lib.strategies.allocation import AllocationEngine, AllocationStrategy

def eimas_risk_parity(prices: pd.DataFrame):
    """EIMAS Risk Parity 배분"""
    returns = prices.pct_change().dropna()

    if len(returns) < 20:
        # Fallback to equal weight
        n = len(prices.columns)
        return {ticker: 1/n for ticker in prices.columns}

    # Use AllocationEngine
    engine = AllocationEngine()
    result = engine.allocate(
        returns=returns,
        strategy=AllocationStrategy.RISK_PARITY
    )

    return result.weights

# Run
result = engine.run(market_prices, eimas_risk_parity)
```

#### 2.3 여러 전략 비교

```python
from lib.backtest import compare_strategies

# Run multiple strategies
strategies = {
    'Equal Weight': lambda p: {t: 1/len(p.columns) for t in p.columns},
    'Risk Parity': eimas_risk_parity,
    # Add more...
}

results = {}
for name, func in strategies.items():
    engine = BacktestEngine(config)
    results[name] = engine.run(market_prices, func)

# Compare
comparison = compare_strategies(results)
print(comparison)
```

### Step 3: 결과 검증 (Target Metrics)

TODO.md 목표 지표 달성 확인:

```python
m = result.metrics

# Check targets
print("=== Target Achievement ===")
print(f"Sharpe >= 1.0:    {m.sharpe_ratio:.2f} {'✓' if m.sharpe_ratio >= 1.0 else '✗'}")
print(f"Max DD <= 20%:    {abs(m.max_drawdown)*100:.1f}% {'✓' if abs(m.max_drawdown) <= 0.20 else '✗'}")
print(f"Win Rate >= 55%:  {m.win_rate*100:.1f}% {'✓' if m.win_rate >= 0.55 else '✗'}")

meets_targets = m.meets_targets()
print(f"\nOverall: {'✓ PASS' if meets_targets else '✗ FAIL'}")
```

## 고급 기능

### Regime별 성과 분석

```python
from lib.regime_analyzer import GMMRegimeAnalyzer

def regime_func(prices: pd.DataFrame) -> str:
    """레짐 판단 함수"""
    returns = prices['SPY'].pct_change().dropna()

    analyzer = GMMRegimeAnalyzer()
    result = analyzer.classify_regime(returns)

    return result.regime  # "Bull", "Neutral", "Bear"

# Run with regime tracking
result = engine.run(
    prices=market_prices,
    allocation_func=eimas_risk_parity,
    regime_func=regime_func
)

# Regime breakdown
print("\n=== Regime Returns ===")
for regime, ret in result.metrics.regime_returns.items():
    print(f"{regime:>10}: {ret*100:>8.2f}%")
```

### Over-fitting 체크

```python
from lib.backtest.utils import check_overfitting

# In-sample (2025-02 ~ 2025-08)
in_sample_config = BacktestConfig(
    start_date='2025-02-04',
    end_date='2025-08-31',
    ...
)
in_sample_result = engine.run(market_prices, strategy_func)

# Out-of-sample (2025-09 ~ 2026-02)
out_sample_config = BacktestConfig(
    start_date='2025-09-01',
    end_date='2026-02-04',
    ...
)
out_sample_result = engine.run(market_prices, strategy_func)

# Check
overfitting_check = check_overfitting(
    in_sample_result,
    out_sample_result,
    tolerance=0.3  # 30% 성과 감소 허용
)

print(f"Overfitted: {overfitting_check['overfitted']}")
```

## 경제학적 방법론

### Sharpe Ratio (Sharpe 1966)
$$\text{Sharpe} = \frac{\mu - r_f}{\sigma}$$

### Sortino Ratio (Sortino & van der Meer 1991)
$$\text{Sortino} = \frac{\mu - \text{MAR}}{\sigma_{\text{downside}}}$$

### Calmar Ratio (Young 1991)
$$\text{Calmar} = \frac{\text{Annualized Return}}{|\text{Max Drawdown}|}$$

### Omega Ratio (Keating & Shadwick 2002)
$$\Omega(\tau) = \frac{\int_{\tau}^{\infty} (1 - F(x)) dx}{\int_{-\infty}^{\tau} F(x) dx}$$

### CVaR (Rockafellar & Uryasev 2000)
$$\text{CVaR}_{\alpha} = \mathbb{E}[X | X \leq \text{VaR}_{\alpha}]$$

## 참고 문헌

1. **Prado (2018)**: "Advances in Financial Machine Learning" - Chapter 7 (Cross-Validation)
2. **Bailey et al. (2014)**: "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
3. **Harvey, Liu, Zhu (2016)**: "...and the Cross-Section of Expected Returns"
4. **Sharpe (1966)**: "Mutual Fund Performance"
5. **Sortino & van der Meer (1991)**: "Downside Risk"

## 다음 단계

TODO.md Priority 1 완료 후:

1. **Priority 2**: 성능 최적화
   - 데이터 수집 병렬화 (75초 → 30초)
   - 분석 모듈 캐싱 (120초 → 60초)
   - AI 호출 최적화 (30초 → 15초)

2. **Priority 3**: 대시보드 개선
   - 백테스트 결과 차트 추가
   - 성과 지표 실시간 모니터링

3. **Priority 4**: 알림 시스템
   - 백테스트 목표 미달성 시 알림
   - Sharpe Ratio 임계값 하회 경고

---

*Last Updated: 2026-02-04*
*Version: v2.2.2 (Backtest Module)*
