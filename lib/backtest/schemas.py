"""
Backtest Schemas
================
백테스트 데이터 스키마 정의

Economic Foundation:
- Bailey et al. (2014): "The Deflated Sharpe Ratio"
- Prado (2018): "Advances in Financial Machine Learning"
- Harvey, Liu, Zhu (2016): Multiple testing problem
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pandas as pd


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    transaction_cost_bps: float = 10.0  # 기본 10bp
    slippage_bps: float = 5.0
    train_period_days: int = 252  # 1년 학습 기간
    test_period_days: int = 63    # 1분기 테스트
    use_rolling_window: bool = True
    min_history_days: int = 252   # 최소 히스토리


@dataclass
class BacktestMetrics:
    """
    백테스트 성과 지표

    References:
    - Sharpe Ratio: Sharpe (1966)
    - Sortino Ratio: Sortino & van der Meer (1991)
    - Calmar Ratio: Young (1991)
    - Omega Ratio: Keating & Shadwick (2002)
    - CVaR: Rockafellar & Uryasev (2000)
    """
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # Return / Max DD
    omega_ratio: float

    # Downside risk
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    downside_deviation: float

    # Win rate
    win_rate: float  # % of positive periods
    profit_factor: float  # Gross profit / Gross loss
    avg_win: float
    avg_loss: float

    # Trading
    num_trades: int
    turnover_annual: float
    total_transaction_costs: float

    # Time periods
    num_periods: int
    start_date: str
    end_date: str

    # Regime breakdown
    regime_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'annualized_volatility': self.annualized_volatility,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'downside_deviation': self.downside_deviation,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'num_trades': self.num_trades,
            'turnover_annual': self.turnover_annual,
            'total_transaction_costs': self.total_transaction_costs,
            'num_periods': self.num_periods,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'regime_returns': self.regime_returns
        }

    def meets_targets(
        self,
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.20,
        min_win_rate: float = 0.55
    ) -> bool:
        """목표 지표 달성 여부 확인"""
        return (
            self.sharpe_ratio >= min_sharpe and
            abs(self.max_drawdown) <= max_drawdown and
            self.win_rate >= min_win_rate
        )


@dataclass
class BacktestResult:
    """백테스트 결과"""
    config: BacktestConfig
    metrics: BacktestMetrics

    # Time series
    portfolio_values: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    drawdowns: pd.Series

    # Trade log
    trades: List[Dict[str, Any]] = field(default_factory=list)

    # Regime breakdown
    regime_history: Optional[pd.Series] = None

    # v2.1: 세부 기록 (DB 저장용)
    daily_nav_records: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_records: List[Dict[str, Any]] = field(default_factory=list)
    period_metrics: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_series: Optional[pd.Series] = None

    def save(self, output_path: Path):
        """결과 저장"""
        result = {
            'config': self.config.__dict__,
            'metrics': self.metrics.to_dict(),
            'portfolio_values': self.portfolio_values.to_dict(),
            'returns': self.returns.to_dict(),
            'weights_history': self.weights_history.to_dict(),
            'drawdowns': self.drawdowns.to_dict(),
            'trades': self.trades,
            'regime_history': self.regime_history.to_dict() if self.regime_history is not None else None
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def summary(self) -> str:
        """결과 요약 문자열"""
        m = self.metrics
        return f"""
=== Backtest Summary ===
Period: {m.start_date} to {m.end_date} ({m.num_periods} periods)

Returns:
  Total Return:       {m.total_return*100:>8.2f}%
  Annualized Return:  {m.annualized_return*100:>8.2f}%
  Volatility:         {m.annualized_volatility*100:>8.2f}%

Risk-Adjusted:
  Sharpe Ratio:       {m.sharpe_ratio:>8.2f}
  Sortino Ratio:      {m.sortino_ratio:>8.2f}
  Calmar Ratio:       {m.calmar_ratio:>8.2f}

Risk:
  Max Drawdown:       {m.max_drawdown*100:>8.2f}%
  DD Duration:        {m.max_drawdown_duration:>8} days
  VaR 95%:            {m.var_95*100:>8.2f}%
  CVaR 95%:           {m.cvar_95*100:>8.2f}%

Trading:
  Win Rate:           {m.win_rate*100:>8.1f}%
  Profit Factor:      {m.profit_factor:>8.2f}
  Num Trades:         {m.num_trades:>8}
  Turnover (Annual):  {m.turnover_annual*100:>8.0f}%
  Total Costs:        ${m.total_transaction_costs:>8,.0f}
"""
