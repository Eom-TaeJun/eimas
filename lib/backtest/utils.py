"""
Backtest Utilities
==================
백테스트 유틸리티 함수

Economic Foundation:
- Multiple testing adjustment: Harvey, Liu, Zhu (2016)
- Strategy comparison: Prado (2018) Chapter 7
"""

from __future__ import annotations
import pandas as pd
from typing import Dict
from .schemas import BacktestResult


def compare_strategies(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    여러 전략 비교

    Args:
        results: {strategy_name: BacktestResult}

    Returns:
        비교 DataFrame
    """
    comparison = []

    for name, result in results.items():
        m = result.metrics
        comparison.append({
            'Strategy': name,
            'Total Return': f"{m.total_return*100:.2f}%",
            'Ann. Return': f"{m.annualized_return*100:.2f}%",
            'Ann. Vol': f"{m.annualized_volatility*100:.2f}%",
            'Sharpe': f"{m.sharpe_ratio:.2f}",
            'Sortino': f"{m.sortino_ratio:.2f}",
            'Max DD': f"{m.max_drawdown*100:.2f}%",
            'Calmar': f"{m.calmar_ratio:.2f}",
            'Win Rate': f"{m.win_rate*100:.1f}%",
            'Turnover': f"{m.turnover_annual*100:.0f}%"
        })

    return pd.DataFrame(comparison)


def rank_strategies(
    results: Dict[str, BacktestResult],
    metric: str = 'sharpe_ratio'
) -> pd.DataFrame:
    """
    전략 랭킹

    Args:
        results: {strategy_name: BacktestResult}
        metric: 랭킹 기준 (sharpe_ratio, calmar_ratio, sortino_ratio 등)

    Returns:
        랭킹 DataFrame
    """
    scores = []

    for name, result in results.items():
        m = result.metrics
        metric_value = getattr(m, metric, None)

        if metric_value is not None:
            scores.append({
                'Strategy': name,
                'Metric': metric,
                'Score': metric_value
            })

    df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    return df[['Rank', 'Strategy', 'Metric', 'Score']]


def check_overfitting(
    in_sample_result: BacktestResult,
    out_of_sample_result: BacktestResult,
    tolerance: float = 0.3
) -> Dict[str, bool]:
    """
    과적합 체크

    Harvey, Liu, Zhu (2016) 방법론:
    Out-of-sample Sharpe가 In-sample의 70% 이상이면 통과

    Args:
        in_sample_result: 학습 기간 결과
        out_of_sample_result: 검증 기간 결과
        tolerance: 허용 감소율 (기본 30%)

    Returns:
        {metric: is_overfitted}
    """
    in_sample = in_sample_result.metrics
    out_sample = out_of_sample_result.metrics

    checks = {}

    # Sharpe Ratio
    if in_sample.sharpe_ratio > 0:
        sharpe_degradation = 1 - (out_sample.sharpe_ratio / in_sample.sharpe_ratio)
        checks['sharpe_overfitted'] = sharpe_degradation > tolerance
    else:
        checks['sharpe_overfitted'] = None

    # Win Rate
    if in_sample.win_rate > 0:
        win_rate_degradation = 1 - (out_sample.win_rate / in_sample.win_rate)
        checks['win_rate_overfitted'] = win_rate_degradation > tolerance
    else:
        checks['win_rate_overfitted'] = None

    # Overall
    checks['overfitted'] = any(v for v in checks.values() if v is not None and v)

    return checks


def generate_report(result: BacktestResult) -> str:
    """
    백테스트 리포트 생성

    Args:
        result: BacktestResult

    Returns:
        텍스트 리포트
    """
    m = result.metrics

    report = f"""
{'='*80}
BACKTEST REPORT
{'='*80}

Configuration:
  Period:           {result.config.start_date} to {result.config.end_date}
  Initial Capital:  ${result.config.initial_capital:,.0f}
  Rebalance Freq:   {result.config.rebalance_frequency}
  Transaction Cost: {result.config.transaction_cost_bps:.1f} bps
  Slippage:         {result.config.slippage_bps:.1f} bps

{'='*80}
PERFORMANCE METRICS
{'='*80}

Returns:
  Total Return:       {m.total_return*100:>10.2f}%
  Annualized Return:  {m.annualized_return*100:>10.2f}%
  Cumulative Return:  {m.cumulative_return*100:>10.2f}%

Risk:
  Annualized Vol:     {m.annualized_volatility*100:>10.2f}%
  Max Drawdown:       {m.max_drawdown*100:>10.2f}%
  DD Duration:        {m.max_drawdown_duration:>10} days
  Downside Deviation: {m.downside_deviation*100:>10.2f}%

Risk-Adjusted Returns:
  Sharpe Ratio:       {m.sharpe_ratio:>10.2f}
  Sortino Ratio:      {m.sortino_ratio:>10.2f}
  Calmar Ratio:       {m.calmar_ratio:>10.2f}
  Omega Ratio:        {m.omega_ratio:>10.2f}

Downside Risk:
  VaR 95%:            {m.var_95*100:>10.2f}%
  CVaR 95%:           {m.cvar_95*100:>10.2f}%

Trading Statistics:
  Win Rate:           {m.win_rate*100:>10.1f}%
  Profit Factor:      {m.profit_factor:>10.2f}
  Avg Win:            {m.avg_win*100:>10.2f}%
  Avg Loss:           {m.avg_loss*100:>10.2f}%

Transaction Costs:
  Num Trades:         {m.num_trades:>10}
  Total Costs:        ${m.total_transaction_costs:>10,.0f}
  Annual Turnover:    {m.turnover_annual*100:>10.0f}%

{'='*80}
REGIME BREAKDOWN
{'='*80}
"""

    if m.regime_returns:
        for regime, ret in m.regime_returns.items():
            report += f"  {regime:>15}: {ret*100:>10.2f}%\n"
    else:
        report += "  (No regime data available)\n"

    report += f"\n{'='*80}\n"

    # Target achievement
    meets_targets = m.meets_targets()
    report += f"\nTarget Achievement: {'✓ PASS' if meets_targets else '✗ FAIL'}\n"
    report += f"  Sharpe >= 1.0:     {m.sharpe_ratio:.2f} {'✓' if m.sharpe_ratio >= 1.0 else '✗'}\n"
    report += f"  Max DD <= 20%:     {abs(m.max_drawdown)*100:.1f}% {'✓' if abs(m.max_drawdown) <= 0.20 else '✗'}\n"
    report += f"  Win Rate >= 55%:   {m.win_rate*100:.1f}% {'✓' if m.win_rate >= 0.55 else '✗'}\n"

    return report
