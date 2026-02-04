"""
Backtesting Engine
==================
과거 데이터 기반 포트폴리오 전략 백테스팅

Features:
1. Out-of-sample testing (train/test split)
2. Rolling window analysis
3. Regime별 성과 분해
4. Transaction cost simulation
5. Performance metrics (Sharpe, Sortino, Max DD, Calmar)

References:
- Prado (2018): "Advances in Financial Machine Learning"
- Bailey et al. (2014): "The Deflated Sharpe Ratio"
- Harvey, Liu, Zhu (2016): "...and the Cross-Section of Expected Returns"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


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
    """백테스트 성과 지표"""
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
        result = {
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
        return result


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

        logger.info(f"Backtest result saved to {output_path}")


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
            metrics.regime_returns = self._compute_regime_returns(returns_series, regime_df)
        else:
            regime_df = None

        # Drawdown series
        dd_series = self._compute_drawdown_series(portfolio_series)

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
        turnover = 0.0
        for ticker in set(list(old_weights.keys()) + list(new_weights.keys())):
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            turnover += abs(new_w - old_w)

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
        dd_series = self._compute_drawdown_series(portfolio_values)
        max_dd = dd_series.min()
        max_dd_duration = self._compute_max_dd_duration(dd_series)

        # Sharpe
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Sortino
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)
        sortino = annualized_return / downside_dev if downside_dev > 0 else 0

        # Calmar
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Omega
        threshold = 0
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega = gains / losses if losses > 0 else float('inf')

        # VaR & CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        wins = returns[returns > 0]
        losses_series = returns[returns < 0]
        profit_factor = wins.sum() / abs(losses_series.sum()) if len(losses_series) > 0 else float('inf')
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses_series.mean() if len(losses_series) > 0 else 0

        # Trading
        num_trades = len(self.trades)
        total_costs = sum(t['cost'] for t in self.trades)

        # Turnover
        turnovers = [t['turnover'] for t in self.trades]
        avg_turnover = np.mean(turnovers) if turnovers else 0

        # Annualize turnover
        rebalances_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }.get(self.config.rebalance_frequency, 12)

        turnover_annual = avg_turnover * rebalances_per_year

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

    def _compute_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """낙폭 시계열 계산"""
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown

    def _compute_max_dd_duration(self, drawdown_series: pd.Series) -> int:
        """최대 낙폭 기간 계산 (일수)"""
        in_drawdown = drawdown_series < 0

        # Find drawdown periods
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

        return max(dd_periods) if dd_periods else 0

    def _compute_regime_returns(self, returns: pd.Series, regime_df: pd.DataFrame) -> Dict[str, float]:
        """레짐별 수익률 분해"""
        regime_returns = {}

        for regime in regime_df['regime'].unique():
            regime_dates = regime_df[regime_df['regime'] == regime].index
            regime_ret = returns[returns.index.isin(regime_dates)]

            if len(regime_ret) > 0:
                total_ret = (1 + regime_ret).prod() - 1
                regime_returns[regime] = total_ret

        return regime_returns


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
            'Max DD': f"{m.max_dd*100:.2f}%",
            'Calmar': f"{m.calmar_ratio:.2f}",
            'Win Rate': f"{m.win_rate*100:.1f}%",
            'Turnover': f"{m.turnover_annual*100:.0f}%"
        })

    return pd.DataFrame(comparison)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Simple Equal Weight Strategy
    import yfinance as yf

    # Download data
    tickers = ['SPY', 'TLT', 'GLD', 'DBC']
    data = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Adj Close']

    # Define strategy
    def equal_weight_strategy(prices: pd.DataFrame) -> Dict[str, float]:
        """균등 비중 전략"""
        n = len(prices.columns)
        return {ticker: 1/n for ticker in prices.columns}

    # Backtest config
    config = BacktestConfig(
        start_date='2016-01-01',
        end_date='2023-12-31',
        initial_capital=1_000_000,
        rebalance_frequency='quarterly',
        transaction_cost_bps=10,
        slippage_bps=5
    )

    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run(data, equal_weight_strategy)

    # Print results
    print("\n=== Backtest Results ===")
    print(f"Total Return: {result.metrics.total_return*100:.2f}%")
    print(f"Annualized Return: {result.metrics.annualized_return*100:.2f}%")
    print(f"Volatility: {result.metrics.annualized_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown*100:.2f}%")
    print(f"Calmar Ratio: {result.metrics.calmar_ratio:.2f}")
    print(f"VaR 95%: {result.metrics.var_95*100:.2f}%")
    print(f"CVaR 95%: {result.metrics.cvar_95*100:.2f}%")

    # Save result
    output_path = Path("outputs/backtest_equal_weight.json")
    result.save(output_path)
    print(f"\nResults saved to {output_path}")
