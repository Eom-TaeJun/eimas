"""
Backtest Engine
===============
백테스팅 엔진 - 과거 데이터 기반 포트폴리오 전략 백테스팅

Features:
1. Out-of-sample testing (train/test split)
2. Rolling window analysis
3. Regime별 성과 분해
4. Transaction cost simulation
5. Performance metrics (Sharpe, Sortino, Max DD, Calmar)

Economic Foundation:
- Prado (2018): "Advances in Financial Machine Learning"
- Bailey et al. (2014): "The Deflated Sharpe Ratio"
- Harvey, Liu, Zhu (2016): "...and the Cross-Section of Expected Returns"
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

from .schemas import BacktestConfig, BacktestMetrics, BacktestResult
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

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    백테스팅 엔진

    주요 기능:
    1. 과거 데이터 로드 및 전처리
    2. 리밸런싱 시뮬레이션
    3. 거래비용 반영
    4. 성과 지표 계산
    5. Regime별 성과 분해
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.weights_history = []
        self.trades = []
        self.portfolio_values = []
        self.dates = []

    def run(
        self,
        prices: pd.DataFrame,
        allocation_func: Callable[[pd.DataFrame], Dict[str, float]],
        regime_func: Optional[Callable[[pd.DataFrame], str]] = None
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            prices: 가격 데이터 (DataFrame, columns=tickers, index=dates)
            allocation_func: 배분 함수 (prices -> weights)
            regime_func: 레짐 판단 함수 (prices -> regime)

        Returns:
            BacktestResult
        """
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")

        # Filter dates
        prices = prices.loc[self.config.start_date:self.config.end_date]

        if len(prices) < self.config.min_history_days:
            raise ValueError(f"Insufficient data: {len(prices)} < {self.config.min_history_days}")

        # Rebalance dates
        rebalance_dates = self._get_rebalance_dates(prices.index)

        # Initialize
        current_weights = None
        regime_history = []

        for i, date in enumerate(prices.index):
            # Check if rebalance needed
            if date in rebalance_dates:
                # Get historical data up to this point
                lookback_prices = prices.iloc[max(0, i - self.config.train_period_days):i+1]

                if len(lookback_prices) >= self.config.min_history_days:
                    # Compute new weights
                    new_weights = allocation_func(lookback_prices)

                    # Execute trades
                    if current_weights is not None:
                        self._execute_rebalance(date, current_weights, new_weights, prices.loc[date])

                    current_weights = new_weights

            # Record regime
            if regime_func and i > self.config.min_history_days:
                lookback_prices = prices.iloc[max(0, i - 252):i+1]
                regime = regime_func(lookback_prices)
                regime_history.append({'date': date, 'regime': regime})

            # Update portfolio value
            if current_weights is not None and i > 0:
                returns = prices.loc[date] / prices.iloc[i-1] - 1
                portfolio_return = sum(current_weights.get(ticker, 0) * returns[ticker]
                                     for ticker in returns.index if ticker in current_weights)
                self.portfolio_value *= (1 + portfolio_return)

            # Record
            self.portfolio_values.append(self.portfolio_value)
            self.dates.append(date)
            if current_weights:
                self.weights_history.append({'date': date, **current_weights})

        # Compute metrics
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns_series = portfolio_series.pct_change().dropna()
        weights_df = pd.DataFrame(self.weights_history).set_index('date')

        metrics = self._compute_metrics(portfolio_series, returns_series, weights_df)

        # Regime breakdown
        if regime_history:
            regime_df = pd.DataFrame(regime_history).set_index('date')
            metrics.regime_returns = calculate_regime_returns(returns_series, regime_df)
        else:
            regime_df = None

        # Drawdown series
        dd_series = calculate_drawdown_series(portfolio_series)

        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            portfolio_values=portfolio_series,
            returns=returns_series,
            weights_history=weights_df,
            drawdowns=dd_series,
            trades=self.trades,
            regime_history=regime_df['regime'] if regime_df is not None else None
        )

        logger.info(f"Backtest complete. Sharpe: {metrics.sharpe_ratio:.2f}, "
                   f"Max DD: {metrics.max_drawdown*100:.1f}%")

        return result

    def _get_rebalance_dates(self, all_dates: pd.DatetimeIndex) -> List:
        """리밸런싱 날짜 계산"""
        freq = self.config.rebalance_frequency

        if freq == "daily":
            return all_dates.tolist()
        elif freq == "weekly":
            return all_dates[all_dates.weekday == 4].tolist()  # Fridays
        elif freq == "monthly":
            return all_dates[all_dates.is_month_end].tolist()
        elif freq == "quarterly":
            return all_dates[(all_dates.month % 3 == 0) & all_dates.is_month_end].tolist()
        else:
            return all_dates[all_dates.is_month_end].tolist()

    def _execute_rebalance(self, date, old_weights: Dict, new_weights: Dict, prices: pd.Series):
        """리밸런싱 실행 및 거래비용 계산"""
        # Calculate turnover
        turnover = calculate_turnover(old_weights, new_weights)

        # Transaction cost
        cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        cost = turnover * cost_bps / 10000.0

        # Apply cost
        self.portfolio_value *= (1 - cost)

        # Record trade
        self.trades.append({
            'date': str(date),
            'turnover': turnover,
            'cost': cost,
            'cost_bps': cost * 10000
        })

    def _compute_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        weights: pd.DataFrame
    ) -> BacktestMetrics:
        """성과 지표 계산"""

        # Basic returns
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        num_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / num_years) - 1

        # Risk
        annualized_vol = returns.std() * np.sqrt(252)

        # Drawdown
        max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)

        # Sharpe
        sharpe = calculate_sharpe_ratio(returns)

        # Sortino
        sortino = calculate_sortino_ratio(returns)

        # Calmar
        calmar = calculate_calmar_ratio(annualized_return, max_dd)

        # Omega
        omega = calculate_omega_ratio(returns)

        # VaR & CVaR
        var_95, cvar_95 = calculate_var_cvar(returns, confidence_level=0.95)

        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)

        # Win rate
        win_rate = calculate_win_rate(returns)

        # Profit factor
        profit_factor = calculate_profit_factor(returns)

        # Avg win/loss
        wins = returns[returns > 0]
        losses_series = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses_series.mean() if len(losses_series) > 0 else 0

        # Trading
        num_trades = len(self.trades)
        total_costs = sum(t['cost'] for t in self.trades)

        # Turnover
        turnovers = [t['turnover'] for t in self.trades]
        avg_turnover = np.mean(turnovers) if turnovers else 0
        turnover_annual = annualize_turnover(avg_turnover, self.config.rebalance_frequency)

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=total_return,
            annualized_volatility=annualized_vol,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            var_95=var_95,
            cvar_95=cvar_95,
            downside_deviation=downside_dev,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            num_trades=num_trades,
            turnover_annual=turnover_annual,
            total_transaction_costs=total_costs,
            num_periods=len(returns),
            start_date=str(portfolio_values.index[0]),
            end_date=str(portfolio_values.index[-1])
        )
