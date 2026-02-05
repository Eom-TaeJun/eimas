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

        logger.info(f"Backtest result saved to {output_path}")


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
        self.cost_model = cost_model  # lib.rebalancing_policy.TradingCostModel
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

        # Initialize drift-tracking state
        actual_weights: Optional[Dict[str, float]] = None  # drifted actual weights (normalized)
        target_weights: Optional[Dict[str, float]] = None  # last target weights (from allocation_func)

        # v2.1 기록 저장소
        daily_nav_records: List[Dict[str, Any]] = []
        snapshot_records: List[Dict[str, Any]] = []
        regime_history: List[Dict] = []
        current_regime: Optional[str] = None

        for i, date in enumerate(prices.index):
            is_rebalance_day = False
            rebalance_cost = 0.0

            # --- Regime 판단 ---
            if regime_func and i > self.config.min_history_days:
                lookback_prices = prices.iloc[max(0, i - 252):i+1]
                current_regime = regime_func(lookback_prices)
                regime_history.append({'date': date, 'regime': current_regime})

            # --- 리밸런싱 체크 ---
            if date in rebalance_dates:
                lookback_prices = prices.iloc[max(0, i - self.config.train_period_days):i+1]

                if len(lookback_prices) >= self.config.min_history_days:
                    # information_cutoff = 이 시점까지 allocation_func이 본 마지막 날짜
                    information_cutoff = lookback_prices.index[-1]

                    new_weights = allocation_func(lookback_prices)

                    if actual_weights is not None:
                        # 스냅샷 기록: 리밸런싱 전 drifted weights
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

                    # 새 target으로 actual 설정
                    target_weights = new_weights
                    actual_weights = dict(new_weights)

            # --- 포트폴리오 가치 업데이트 (drift 추적) ---
            ticker_returns: Dict[str, float] = {}
            ticker_pnl: Dict[str, float] = {}
            portfolio_return = 0.0

            if actual_weights is not None and i > 0:
                prev_prices = prices.iloc[i-1]
                curr_prices = prices.loc[date]

                # 각 ticker의 수익률 계산 및 drift 적용
                new_actual = {}
                for ticker in list(actual_weights.keys()):
                    w = actual_weights[ticker]
                    if ticker in curr_prices.index and ticker in prev_prices.index:
                        if prev_prices[ticker] != 0:
                            ret = (curr_prices[ticker] / prev_prices[ticker]) - 1
                        else:
                            ret = 0.0
                        ticker_returns[ticker] = round(float(ret), 8)
                        pnl = w * ret
                        ticker_pnl[ticker] = round(float(pnl), 8)
                        portfolio_return += pnl
                        # drift: 비중이 수익률에 따라 변화
                        new_actual[ticker] = w * (1 + ret)
                    else:
                        ticker_returns[ticker] = 0.0
                        ticker_pnl[ticker] = 0.0
                        new_actual[ticker] = w

                # 정규화 (포트폴리오 수익률 반영 후 비중 합 = 1)
                total_weight = sum(new_actual.values())
                if total_weight > 0:
                    actual_weights = {t: w / total_weight for t, w in new_actual.items()}
                else:
                    actual_weights = new_actual

                self.portfolio_value *= (1 + portfolio_return)

            # --- Benchmark NAV 업데이트 ---
            bm_return = 0.0
            if benchmark_col and i > 0:
                prev_bm = prices.iloc[i-1][benchmark_col]
                curr_bm = prices.loc[date][benchmark_col]
                if prev_bm != 0:
                    bm_return = float((curr_bm / prev_bm) - 1)
                benchmark_nav *= (1 + bm_return)

            benchmark_navs.append(benchmark_nav)

            # --- 기록 ---
            self.portfolio_values.append(self.portfolio_value)
            self.dates.append(date)
            if target_weights:
                self.weights_history.append({'date': date, **target_weights})

            # 낙폭 계산
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

        # --- 최종 계산 ---
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns_series = portfolio_series.pct_change().dropna()
        weights_df = pd.DataFrame(self.weights_history).set_index('date') if self.weights_history else pd.DataFrame()

        benchmark_series = pd.Series(benchmark_navs, index=self.dates) if benchmark_navs else None

        metrics = self._compute_metrics(portfolio_series, returns_series, weights_df)

        # Regime breakdown
        if regime_history:
            regime_df = pd.DataFrame(regime_history).set_index('date')
            metrics.regime_returns = self._compute_regime_returns(returns_series, regime_df)
        else:
            regime_df = None

        # Drawdown series
        dd_series = self._compute_drawdown_series(portfolio_series)

        # --- Period metrics 계산 ---
        period_metrics = self._compute_period_metrics(
            portfolio_series, returns_series, benchmark_series, regime_df
        )

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
    ) -> Tuple[float, float, Dict[str, float]]:
        """리밸런싱 실행 및 거래비용 계산 (drift-aware)

        Args:
            drifted_weights: 실제 drift된 현재 비중
            new_weights: 새 target 비중
            prices: 현재 날짜 가격

        Returns:
            (turnover, total_cost, cost_per_ticker)
        """
        all_tickers = set(list(drifted_weights.keys()) + list(new_weights.keys()))

        # Turnover = sum(|new_target - drifted_actual|)
        turnover = 0.0
        cost_per_ticker: Dict[str, float] = {}

        if self.cost_model is not None:
            # TradingCostModel 사용
            for ticker in all_tickers:
                old_w = drifted_weights.get(ticker, 0.0)
                new_w = new_weights.get(ticker, 0.0)
                trade_weight = abs(new_w - old_w)
                turnover += trade_weight
                trade_value = trade_weight * self.portfolio_value
                cost_per_ticker[ticker] = self.cost_model.calculate_cost(trade_value, self.portfolio_value)

            total_cost = sum(cost_per_ticker.values())
        else:
            # Simple bps 모델
            cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
            for ticker in all_tickers:
                old_w = drifted_weights.get(ticker, 0.0)
                new_w = new_weights.get(ticker, 0.0)
                trade_weight = abs(new_w - old_w)
                turnover += trade_weight
                cost_per_ticker[ticker] = trade_weight * cost_bps / 10000.0 * self.portfolio_value

            total_cost = turnover * cost_bps / 10000.0 * self.portfolio_value

        # Apply cost (비율로 차감)
        cost_ratio = total_cost / self.portfolio_value if self.portfolio_value > 0 else 0
        self.portfolio_value -= total_cost

        # Record trade
        self.trades.append({
            'date': str(date),
            'turnover': round(turnover, 6),
            'cost': round(total_cost, 2),
            'cost_bps': round(cost_ratio * 10000, 2)
        })

        return turnover, total_cost, cost_per_ticker

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

    def _compute_period_metrics(
        self,
        portfolio_series: pd.Series,
        returns_series: pd.Series,
        benchmark_series: Optional[pd.Series],
        regime_df: Optional[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """기간별 성과 메트릭 계산 (OVERALL, YEARLY, QUARTERLY, REGIME)"""
        periods: List[Dict[str, Any]] = []

        def _calc_period(start, end, label, period_type) -> Dict[str, Any]:
            """단일 기간의 메트릭 계산"""
            mask = (returns_series.index >= start) & (returns_series.index <= end)
            rets = returns_series[mask]

            if len(rets) == 0:
                return {}

            nav_mask = (portfolio_series.index >= start) & (portfolio_series.index <= end)
            nav_sub = portfolio_series[nav_mask]
            total_ret = (nav_sub.iloc[-1] / nav_sub.iloc[0]) - 1 if len(nav_sub) > 1 else 0.0

            num_days = len(rets)
            num_years = num_days / 252.0
            ann_ret = (1 + total_ret) ** (1 / num_years) - 1 if num_years > 0 else 0.0
            ann_vol = rets.std() * np.sqrt(252)

            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

            downside = rets[rets < 0]
            down_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
            sortino = ann_ret / down_dev if down_dev > 0 else 0.0

            dd_series = self._compute_drawdown_series(nav_sub)
            max_dd = dd_series.min() if len(dd_series) > 0 else 0.0

            # Benchmark relative metrics
            bm_ret, alpha, beta, tracking_err, info_ratio = None, None, None, None, None
            if benchmark_series is not None:
                bm_mask = (benchmark_series.index >= start) & (benchmark_series.index <= end)
                bm_sub = benchmark_series[bm_mask]
                if len(bm_sub) > 1:
                    bm_total_ret = (bm_sub.iloc[-1] / bm_sub.iloc[0]) - 1
                    bm_ret = float(bm_total_ret)
                    bm_rets = bm_sub.pct_change().dropna()
                    # Align indices
                    common_idx = rets.index.intersection(bm_rets.index)
                    if len(common_idx) > 1:
                        s_aligned = rets[common_idx]
                        b_aligned = bm_rets[common_idx]
                        cov_mat = np.cov(s_aligned, b_aligned)
                        beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] != 0 else 0.0
                        alpha = float(total_ret - beta * bm_total_ret)
                        excess = s_aligned - b_aligned
                        tracking_err = float(excess.std() * np.sqrt(252))
                        info_ratio = float(alpha / tracking_err) if tracking_err > 0 else 0.0

            return {
                'period_type': period_type,
                'period_label': label,
                'period_start': str(start.date()) if hasattr(start, 'date') else str(start),
                'period_end': str(end.date()) if hasattr(end, 'date') else str(end),
                'total_return': round(float(total_ret), 6),
                'annualized_return': round(float(ann_ret), 6),
                'annualized_vol': round(float(ann_vol), 6),
                'sharpe_ratio': round(float(sharpe), 4),
                'sortino_ratio': round(float(sortino), 4),
                'max_drawdown': round(float(max_dd), 6),
                'num_trading_days': num_days,
                'benchmark_return': round(bm_ret, 6) if bm_ret is not None else None,
                'alpha': round(alpha, 6) if alpha is not None else None,
                'beta': round(beta, 4) if beta is not None else None,
                'tracking_error': round(tracking_err, 6) if tracking_err is not None else None,
                'information_ratio': round(info_ratio, 4) if info_ratio is not None else None,
            }

        # --- OVERALL ---
        overall = _calc_period(returns_series.index[0], returns_series.index[-1], 'FULL', 'OVERALL')
        if overall:
            periods.append(overall)

        # --- YEARLY ---
        for year in sorted(returns_series.index.year.unique()):
            year_mask = returns_series.index.year == year
            year_rets = returns_series[year_mask]
            if len(year_rets) > 0:
                p = _calc_period(year_rets.index[0], year_rets.index[-1], str(year), 'YEARLY')
                if p:
                    periods.append(p)

        # --- QUARTERLY ---
        for year in sorted(returns_series.index.year.unique()):
            for q in range(1, 5):
                months = [(q-1)*3 + 1, (q-1)*3 + 2, (q-1)*3 + 3]
                q_mask = (returns_series.index.year == year) & (returns_series.index.month.isin(months))
                q_rets = returns_series[q_mask]
                if len(q_rets) > 0:
                    label = f"Q{q}-{year}"
                    p = _calc_period(q_rets.index[0], q_rets.index[-1], label, 'QUARTERLY')
                    if p:
                        periods.append(p)

        # --- REGIME ---
        if regime_df is not None and 'regime' in regime_df.columns:
            for regime_name in regime_df['regime'].unique():
                regime_dates = regime_df[regime_df['regime'] == regime_name].index
                common = returns_series.index.intersection(regime_dates)
                if len(common) > 1:
                    p = _calc_period(common[0], common[-1], regime_name, 'REGIME')
                    if p:
                        # REGIME period_label이 실제 regime 이름
                        periods.append(p)

        return periods

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
            'Max DD': f"{m.max_drawdown*100:.2f}%",
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
    data = yf.download(tickers, start='2015-01-01', end='2024-01-01', auto_adjust=True)['Close']

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
