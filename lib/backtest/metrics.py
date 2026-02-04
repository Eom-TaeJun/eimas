"""
Backtest Metrics Calculation
=============================
백테스트 성과 지표 계산 함수

Economic Foundation:
- Sharpe (1966): "Mutual Fund Performance"
- Sortino & van der Meer (1991): "Downside Risk"
- Keating & Shadwick (2002): "A Universal Performance Measure"
- Rockafellar & Uryasev (2000): "Optimization of Conditional Value-at-Risk"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    샤프 비율 계산

    Sharpe Ratio = (Return - Risk Free Rate) / Volatility

    Args:
        returns: 수익률 시계열
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 기간 수

    Returns:
        샤프 비율
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess_returns.mean() * periods_per_year
    vol = returns.std() * np.sqrt(periods_per_year)

    if vol == 0:
        return 0.0

    return mean_excess / vol


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    소르티노 비율 계산

    Sortino Ratio = (Return - Target) / Downside Deviation

    Args:
        returns: 수익률 시계열
        target_return: 목표 수익률
        periods_per_year: 연간 기간 수

    Returns:
        소르티노 비율
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return float('inf')

    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_dev == 0:
        return 0.0

    mean_excess = excess_returns.mean() * periods_per_year
    return mean_excess / downside_dev


def calculate_max_drawdown(portfolio_values: pd.Series) -> Tuple[float, int]:
    """
    최대 낙폭 및 기간 계산

    Args:
        portfolio_values: 포트폴리오 가치 시계열

    Returns:
        (max_drawdown, max_drawdown_duration_days)
    """
    if len(portfolio_values) == 0:
        return 0.0, 0

    # Drawdown series
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max

    max_dd = drawdown.min()

    # Duration
    in_drawdown = drawdown < 0
    dd_periods = []
    start = None

    for i, is_dd in enumerate(in_drawdown):
        if is_dd and start is None:
            start = i
        elif not is_dd and start is not None:
            dd_periods.append(i - start)
            start = None

    if start is not None:
        dd_periods.append(len(in_drawdown) - start)

    max_dd_duration = max(dd_periods) if dd_periods else 0

    return max_dd, max_dd_duration


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """
    칼마 비율 계산

    Calmar Ratio = Annualized Return / |Max Drawdown|

    Args:
        annualized_return: 연율화 수익률
        max_drawdown: 최대 낙폭 (음수)

    Returns:
        칼마 비율
    """
    if max_drawdown == 0:
        return 0.0

    return annualized_return / abs(max_drawdown)


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    오메가 비율 계산

    Omega Ratio = Gains above threshold / Losses below threshold

    Args:
        returns: 수익률 시계열
        threshold: 임계값 수익률

    Returns:
        오메가 비율
    """
    if len(returns) == 0:
        return 0.0

    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0

    return gains / losses


def calculate_var_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    VaR 및 CVaR 계산

    VaR: Value at Risk (분위수)
    CVaR: Conditional VaR (Expected Shortfall)

    Args:
        returns: 수익률 시계열
        confidence_level: 신뢰수준 (예: 0.95)

    Returns:
        (var, cvar)
    """
    if len(returns) == 0:
        return 0.0, 0.0

    alpha = 1 - confidence_level
    var = returns.quantile(alpha)

    # CVaR: 평균 손실 (VaR 이하)
    tail_losses = returns[returns <= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else var

    return var, cvar


def calculate_win_rate(returns: pd.Series) -> float:
    """
    승률 계산

    Args:
        returns: 수익률 시계열

    Returns:
        승률 (0~1)
    """
    if len(returns) == 0:
        return 0.0

    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    수익 팩터 계산

    Profit Factor = Total Gains / Total Losses

    Args:
        returns: 수익률 시계열

    Returns:
        수익 팩터
    """
    if len(returns) == 0:
        return 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    total_gains = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0

    if total_losses == 0:
        return float('inf') if total_gains > 0 else 0.0

    return total_gains / total_losses


def calculate_drawdown_series(portfolio_values: pd.Series) -> pd.Series:
    """
    낙폭 시계열 계산

    Args:
        portfolio_values: 포트폴리오 가치 시계열

    Returns:
        낙폭 시계열 (음수)
    """
    if len(portfolio_values) == 0:
        return pd.Series()

    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown


def calculate_regime_returns(
    returns: pd.Series,
    regime_df: pd.DataFrame
) -> Dict[str, float]:
    """
    레짐별 수익률 분해

    Args:
        returns: 수익률 시계열
        regime_df: 레짐 데이터프레임 (index=date, columns=['regime'])

    Returns:
        {regime: cumulative_return}
    """
    regime_returns = {}

    for regime in regime_df['regime'].unique():
        regime_dates = regime_df[regime_df['regime'] == regime].index
        regime_ret = returns[returns.index.isin(regime_dates)]

        if len(regime_ret) > 0:
            total_ret = (1 + regime_ret).prod() - 1
            regime_returns[regime] = total_ret

    return regime_returns


def calculate_turnover(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float]
) -> float:
    """
    턴오버 계산

    Turnover = Sum of |weight_new - weight_old|

    Args:
        old_weights: 이전 비중
        new_weights: 새 비중

    Returns:
        턴오버 (0~2, 일반적으로 0~1)
    """
    all_tickers = set(list(old_weights.keys()) + list(new_weights.keys()))

    turnover = 0.0
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0)
        new_w = new_weights.get(ticker, 0)
        turnover += abs(new_w - old_w)

    return turnover


def annualize_turnover(
    avg_turnover: float,
    rebalance_frequency: str
) -> float:
    """
    연율화 턴오버 계산

    Args:
        avg_turnover: 평균 턴오버
        rebalance_frequency: 리밸런싱 주기

    Returns:
        연간 턴오버
    """
    rebalances_per_year = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4
    }.get(rebalance_frequency, 12)

    return avg_turnover * rebalances_per_year
