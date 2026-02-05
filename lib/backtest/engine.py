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
    백테스팅 엔진 (v2.1 — drift tracking, benchmark, per-ticker attribution)

    주요 기능:
    1. 과거 데이터 로드 및 전처리
    2. 리밸런싱 시뮬레이션 (실제 drift 추적)
    3. 거래비용 반영 (TradingCostModel 지원)
    4. 성과 지표 계산 (benchmark-relative alpha/beta)
    5. Regime별 성과 분해 + 기간별 메트릭
    6. 일별 NAV / 스냅샷 기록 (DB 저장용)
    """

    def __init__(self, config: BacktestConfig, cost_model=None):
        """
        Args:
            config: BacktestConfig
            cost_model: TradingCostModel 인스턴스 (None이면 simple bps 사용)
        """
        self.config = config
        self.cost_model = cost_model
        self.portfolio_value = config.initial_capital
        self.weights_history = []
        self.trades = []
        self.portfolio_values = []
        self.dates = []

    def run(
        self,
        prices: pd.DataFrame,
        allocation_func: Callable[[pd.DataFrame], Dict[str, float]],
        regime_func: Optional[Callable[[pd.DataFrame], str]] = None,
        benchmark: str = 'SPY'
    ) -> BacktestResult:
        """
        백테스트 실행

        Args:
            prices: 가격 데이터 (DataFrame, columns=tickers, index=dates)
            allocation_func: 배분 함수 (prices -> weights)
            regime_func: 레짐 판단 함수 (prices -> regime)
            benchmark: 벌치마크 ticker (기본 'SPY')

        Returns:
            BacktestResult
        """
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")

        # Filter dates
        prices = prices.loc[self.config.start_date:self.config.end_date]

        if len(prices) < self.config.min_history_days:
            raise ValueError(f"Insufficient data: {len(prices)} < {self.config.min_history_days}")

        # Rebalance dates (set for O(1) lookup)
        rebalance_dates = set(self._get_rebalance_dates(prices.index))

        # Benchmark NAV tracking
        benchmark_col = benchmark if benchmark in prices.columns else None
        benchmark_nav = self.config.initial_capital
        benchmark_navs = []

        # Drift-tracking state
        actual_weights: Optional[Dict[str, float]] = None  # drifted actual weights
        target_weights: Optional[Dict[str, float]] = None  # last allocation target

        # v2.1 기록
        daily_nav_records: List[Dict] = []
        snapshot_records: List[Dict] = []
        regime_history: List[Dict] = []
        current_regime: Optional[str] = None

        for i, date in enumerate(prices.index):
            is_rebalance_day = False
            rebalance_cost = 0.0

            # --- Regime ---
            if regime_func and i > self.config.min_history_days:
                lookback_prices = prices.iloc[max(0, i - 252):i+1]
                current_regime = regime_func(lookback_prices)
                regime_history.append({'date': date, 'regime': current_regime})

            # --- Rebalance check ---
            if date in rebalance_dates:
                lookback_prices = prices.iloc[max(0, i - self.config.train_period_days):i+1]

                if len(lookback_prices) >= self.config.min_history_days:
                    information_cutoff = lookback_prices.index[-1]
                    new_weights = allocation_func(lookback_prices)

                    if actual_weights is not None:
                        nav_before = self.portfolio_value
                        turnover, cost, cost_per_ticker = self._execute_rebalance(
                            date, actual_weights, new_weights, prices.loc[date]
                        )
                        rebalance_cost = cost
                        nav_after = self.portfolio_value
                        is_rebalance_day = True

                        snapshot_records.append({
                            'snapshot_date': str(date.date()) if hasattr(date, 'date') else str(date),
                            'information_cutoff': str(information_cutoff.date()) if hasattr(information_cutoff, 'date') else str(information_cutoff),
                            'weights_drifted': {k: round(v, 6) for k, v in actual_weights.items()},
                            'weights_target': {k: round(v, 6) for k, v in new_weights.items()},
                            'weights_executed': {k: round(v, 6) for k, v in new_weights.items()},
                            'turnover': round(turnover, 6),
                            'transaction_cost': round(cost, 6),
                            'cost_per_ticker': {k: round(v, 6) for k, v in cost_per_ticker.items()},
                            'nav_before': round(nav_before, 2),
                            'nav_after': round(nav_after, 2),
                            'regime': current_regime,
                        })

                    target_weights = new_weights
                    actual_weights = dict(new_weights)

            # --- Portfolio update with drift ---
            ticker_returns: Dict[str, float] = {}
            ticker_pnl: Dict[str, float] = {}
            portfolio_return = 0.0

            if actual_weights is not None and i > 0:
                prev_prices = prices.iloc[i-1]
                curr_prices = prices.loc[date]

                new_actual = {}
                for ticker in list(actual_weights.keys()):
                    w = actual_weights[ticker]
                    if ticker in curr_prices.index and ticker in prev_prices.index and prev_prices[ticker] != 0:
                        ret = float((curr_prices[ticker] / prev_prices[ticker]) - 1)
                    else:
                        ret = 0.0
                    ticker_returns[ticker] = round(ret, 8)
                    pnl = w * ret
                    ticker_pnl[ticker] = round(pnl, 8)
                    portfolio_return += pnl
                    new_actual[ticker] = w * (1 + ret)

                # 정규화
                total_weight = sum(new_actual.values())
                if total_weight > 0:
                    actual_weights = {t: w / total_weight for t, w in new_actual.items()}
                else:
                    actual_weights = new_actual

                self.portfolio_value *= (1 + portfolio_return)

            # --- Benchmark NAV ---
            bm_return = 0.0
            if benchmark_col and i > 0:
                prev_bm = prices.iloc[i-1][benchmark_col]
                curr_bm = prices.loc[date][benchmark_col]
                if prev_bm != 0:
                    bm_return = float((curr_bm / prev_bm) - 1)
                benchmark_nav *= (1 + bm_return)
            benchmark_navs.append(benchmark_nav)

            # --- Record ---
            self.portfolio_values.append(self.portfolio_value)
            self.dates.append(date)
            if target_weights:
                self.weights_history.append({'date': date, **target_weights})

            peak = max(self.portfolio_values)
            drawdown = (self.portfolio_value - peak) / peak if peak > 0 else 0.0
            cum_return = (self.portfolio_value / self.config.initial_capital) - 1
            date_str = str(date.date()) if hasattr(date, 'date') else str(date)

            daily_nav_records.append({
                'trade_date': date_str,
                'nav': round(self.portfolio_value, 2),
                'daily_return': round(float(portfolio_return), 8),
                'cum_return': round(float(cum_return), 6),
                'drawdown': round(float(drawdown), 6),
                'weights': {k: round(v, 6) for k, v in (actual_weights or {}).items()},
                'ticker_returns': ticker_returns,
                'ticker_pnl': ticker_pnl,
                'benchmark_nav': round(benchmark_nav, 2),
                'benchmark_return': round(float(bm_return), 8),
                'regime': current_regime,
                'is_rebalance_day': 1 if is_rebalance_day else 0,
                'rebalance_cost': round(rebalance_cost, 6) if is_rebalance_day else None,
            })

        # --- Final metrics ---
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns_series = portfolio_series.pct_change().dropna()
        weights_df = pd.DataFrame(self.weights_history).set_index('date') if self.weights_history else pd.DataFrame()
        benchmark_series = pd.Series(benchmark_navs, index=self.dates) if benchmark_navs else None

        metrics = self._compute_metrics(portfolio_series, returns_series, weights_df)

        if regime_history:
            regime_df = pd.DataFrame(regime_history).set_index('date')
            metrics.regime_returns = calculate_regime_returns(returns_series, regime_df)
        else:
            regime_df = None

        dd_series = calculate_drawdown_series(portfolio_series)
        period_metrics = self._compute_period_metrics(portfolio_series, returns_series, benchmark_series, regime_df)

        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            portfolio_values=portfolio_series,
            returns=returns_series,
            weights_history=weights_df,
            drawdowns=dd_series,
            trades=self.trades,
            regime_history=regime_df['regime'] if regime_df is not None else None,
            daily_nav_records=daily_nav_records,
            snapshot_records=snapshot_records,
            period_metrics=period_metrics,
            benchmark_series=benchmark_series,
        )

        logger.info(f"Backtest complete. Sharpe: {metrics.sharpe_ratio:.2f}, "
                   f"Max DD: {metrics.max_drawdown*100:.1f}%, "
                   f"Snapshots: {len(snapshot_records)}")

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

    def _execute_rebalance(
        self,
        date,
        drifted_weights: Dict[str, float],
        new_weights: Dict[str, float],
        prices: pd.Series
    ):
        """리밸런싱 실행 및 거래비용 계산 (drift-aware)

        Returns:
            (turnover, total_cost, cost_per_ticker)
        """
        all_tickers = set(list(drifted_weights.keys()) + list(new_weights.keys()))
        turnover = 0.0
        cost_per_ticker: Dict[str, float] = {}

        if self.cost_model is not None:
            for ticker in all_tickers:
                trade_weight = abs(new_weights.get(ticker, 0.0) - drifted_weights.get(ticker, 0.0))
                turnover += trade_weight
                trade_value = trade_weight * self.portfolio_value
                cost_per_ticker[ticker] = self.cost_model.calculate_cost(trade_value, self.portfolio_value)
            total_cost = sum(cost_per_ticker.values())
        else:
            cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
            for ticker in all_tickers:
                trade_weight = abs(new_weights.get(ticker, 0.0) - drifted_weights.get(ticker, 0.0))
                turnover += trade_weight
                cost_per_ticker[ticker] = trade_weight * cost_bps / 10000.0 * self.portfolio_value
            total_cost = turnover * cost_bps / 10000.0 * self.portfolio_value

        self.portfolio_value -= total_cost

        cost_ratio = total_cost / (self.portfolio_value + total_cost) if (self.portfolio_value + total_cost) > 0 else 0
        self.trades.append({
            'date': str(date),
            'turnover': round(turnover, 6),
            'cost': round(total_cost, 2),
            'cost_bps': round(cost_ratio * 10000, 2)
        })

        return turnover, total_cost, cost_per_ticker

    def _compute_period_metrics(
        self,
        portfolio_series: pd.Series,
        returns_series: pd.Series,
        benchmark_series,
        regime_df
    ) -> List[Dict]:
        """기간별 성과 메트릭 (OVERALL, YEARLY, QUARTERLY, REGIME)"""
        periods: List[Dict] = []

        def _calc(start, end, label, ptype):
            mask = (returns_series.index >= start) & (returns_series.index <= end)
            rets = returns_series[mask]
            if len(rets) == 0:
                return None

            nav_mask = (portfolio_series.index >= start) & (portfolio_series.index <= end)
            nav_sub = portfolio_series[nav_mask]
            total_ret = float((nav_sub.iloc[-1] / nav_sub.iloc[0]) - 1) if len(nav_sub) > 1 else 0.0

            num_days = len(rets)
            num_years = num_days / 252.0
            ann_ret = float((1 + total_ret) ** (1 / num_years) - 1) if num_years > 0 else 0.0
            ann_vol = float(rets.std() * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

            downside = rets[rets < 0]
            down_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0
            sortino = ann_ret / down_dev if down_dev > 0 else 0.0

            dd_s = calculate_drawdown_series(nav_sub)
            max_dd = float(dd_s.min()) if len(dd_s) > 0 else 0.0

            bm_ret, alpha, beta, te, ir = None, None, None, None, None
            if benchmark_series is not None:
                bm_mask = (benchmark_series.index >= start) & (benchmark_series.index <= end)
                bm_sub = benchmark_series[bm_mask]
                if len(bm_sub) > 1:
                    bm_total_ret = float((bm_sub.iloc[-1] / bm_sub.iloc[0]) - 1)
                    bm_ret = bm_total_ret
                    bm_rets = bm_sub.pct_change().dropna()
                    common_idx = rets.index.intersection(bm_rets.index)
                    if len(common_idx) > 1:
                        s_a = rets[common_idx].values
                        b_a = bm_rets[common_idx].values
                        cov_mat = np.cov(s_a, b_a)
                        beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] != 0 else 0.0
                        alpha = float(total_ret - beta * bm_total_ret)
                        te = float(np.std(s_a - b_a) * np.sqrt(252))
                        ir = float(alpha / te) if te > 0 else 0.0

            return {
                'period_type': ptype,
                'period_label': label,
                'period_start': str(start.date()) if hasattr(start, 'date') else str(start),
                'period_end': str(end.date()) if hasattr(end, 'date') else str(end),
                'total_return': round(total_ret, 6),
                'annualized_return': round(ann_ret, 6),
                'annualized_vol': round(ann_vol, 6),
                'sharpe_ratio': round(sharpe, 4),
                'sortino_ratio': round(sortino, 4),
                'max_drawdown': round(max_dd, 6),
                'num_trading_days': num_days,
                'benchmark_return': round(bm_ret, 6) if bm_ret is not None else None,
                'alpha': round(alpha, 6) if alpha is not None else None,
                'beta': round(beta, 4) if beta is not None else None,
                'tracking_error': round(te, 6) if te is not None else None,
                'information_ratio': round(ir, 4) if ir is not None else None,
            }

        # OVERALL
        p = _calc(returns_series.index[0], returns_series.index[-1], 'FULL', 'OVERALL')
        if p:
            periods.append(p)

        # YEARLY
        for year in sorted(returns_series.index.year.unique()):
            year_rets = returns_series[returns_series.index.year == year]
            if len(year_rets) > 0:
                p = _calc(year_rets.index[0], year_rets.index[-1], str(year), 'YEARLY')
                if p:
                    periods.append(p)

        # QUARTERLY
        for year in sorted(returns_series.index.year.unique()):
            for q in range(1, 5):
                months = [(q-1)*3 + 1, (q-1)*3 + 2, (q-1)*3 + 3]
                q_mask = (returns_series.index.year == year) & (returns_series.index.month.isin(months))
                q_rets = returns_series[q_mask]
                if len(q_rets) > 0:
                    p = _calc(q_rets.index[0], q_rets.index[-1], f"Q{q}-{year}", 'QUARTERLY')
                    if p:
                        periods.append(p)

        # REGIME
        if regime_df is not None and 'regime' in regime_df.columns:
            for regime_name in regime_df['regime'].unique():
                regime_dates = regime_df[regime_df['regime'] == regime_name].index
                common = returns_series.index.intersection(regime_dates)
                if len(common) > 1:
                    p = _calc(common[0], common[-1], str(regime_name), 'REGIME')
                    if p:
                        periods.append(p)

        return periods

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
