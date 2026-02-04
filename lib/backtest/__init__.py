"""
Backtest Package
================
백테스트 시스템 - 과거 데이터 기반 포트폴리오 전략 검증

Package Structure:
- enums.py: RebalanceFrequency, BacktestMode
- schemas.py: BacktestConfig, BacktestMetrics, BacktestResult
- metrics.py: Metric calculation functions
- engine.py: BacktestEngine
- utils.py: compare_strategies, generate_report

Economic Foundation:
- Prado (2018): "Advances in Financial Machine Learning"
- Bailey et al. (2014): "The Deflated Sharpe Ratio"
- Harvey, Liu, Zhu (2016): "...and the Cross-Section of Expected Returns"
- Sharpe (1966): "Mutual Fund Performance"
- Sortino & van der Meer (1991): "Downside Risk"

Example:
    ```python
    from lib.backtest import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_frequency='monthly'
    )

    engine = BacktestEngine(config)
    result = engine.run(prices, allocation_func)

    print(result.metrics.sharpe_ratio)
    print(result.summary())
    ```
"""

from __future__ import annotations

# Enums
from .enums import (
    RebalanceFrequency,
    BacktestMode
)

# Schemas
from .schemas import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult
)

# Engine
from .engine import BacktestEngine

# Metrics
from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_var_cvar,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_drawdown_series,
    calculate_regime_returns,
    calculate_turnover,
    annualize_turnover
)

# Utils
from .utils import (
    compare_strategies,
    rank_strategies,
    check_overfitting,
    generate_report
)

__all__ = [
    # Enums
    'RebalanceFrequency',
    'BacktestMode',

    # Schemas
    'BacktestConfig',
    'BacktestMetrics',
    'BacktestResult',

    # Engine
    'BacktestEngine',

    # Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_omega_ratio',
    'calculate_var_cvar',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_drawdown_series',
    'calculate_regime_returns',
    'calculate_turnover',
    'annualize_turnover',

    # Utils
    'compare_strategies',
    'rank_strategies',
    'check_overfitting',
    'generate_report',
]
