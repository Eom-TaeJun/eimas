#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 6: Portfolio Theory

Purpose:
    Portfolio theory operations including backtest, attribution, and stress testing
"""

import subprocess
import sqlite3
from pathlib import Path
from typing import Dict, Any

from lib.adapters import StressTestEngine, TacticalAssetAllocator, TradingCostModel
from lib.backtest import BacktestEngine, BacktestConfig
from lib.performance_attribution import BrinsonAttribution, InformationRatio, ActiveShare
from lib.trading_db import TradingDB
from pipeline.schemas import EIMASResult


def _extract_close_series(df):
    """Return a close-price series across common OHLC naming variants."""
    if df is None:
        return None

    for key in ("Close", "close", "Adj Close", "adj_close"):
        if key in df.columns:
            return df[key]

    lowered = {str(col).lower(): col for col in df.columns}
    if "close" in lowered:
        return df[lowered["close"]]
    return None


def run_backtest(result: EIMASResult, market_data: Dict, enable: bool):
    """[Phase 6.1] 백테스팅 엔진 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.1] Running Backtest Engine...")

    try:
        import pandas as pd
        from datetime import timedelta

        # Convert market_data to prices DataFrame
        if not market_data or len(market_data) == 0:
            print("⚠️ No market data for backtest")
            return

        # Get price data from market_data dict
        tickers = list(market_data.keys())
        prices_dict = {}

        for ticker in tickers:
            data = market_data[ticker]
            if isinstance(data, pd.DataFrame):
                close_series = _extract_close_series(data)
                if close_series is not None:
                    prices_dict[ticker] = close_series

        if not prices_dict:
            print("⚠️ No valid price data for backtest")
            return

        prices = pd.DataFrame(prices_dict)
        prices = prices.dropna()

        if len(prices) < 252:
            print(f"⚠️ Insufficient data for backtest: {len(prices)} days")
            return

        # Configure backtest (5 years or all available)
        start_date = str(prices.index[252])  # Skip first year for indicators
        end_date = str(prices.index[-1])

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency='monthly',
            transaction_cost_bps=10,
            initial_capital=1_000_000
        )

        # Define allocation strategy (use current portfolio weights or equal weight)
        def allocation_strategy(prices_window):
            if result.portfolio_weights:
                # Use existing portfolio weights
                return result.portfolio_weights
            else:
                # Equal weight fallback
                n = len(prices_window.columns)
                return {ticker: 1/n for ticker in prices_window.columns}

        # TradingCostModel 생성
        cost_model = TradingCostModel()

        # Run backtest (v2.1: drift tracking + benchmark + TradingCostModel)
        engine = BacktestEngine(config, cost_model=cost_model)
        backtest_result = engine.run(prices, allocation_strategy, benchmark='SPY')

        # Store metrics
        metrics = backtest_result.metrics
        result.backtest_metrics = metrics.to_dict()

        # --- Period metrics에서 OVERALL alpha/beta 추출 ---
        overall_alpha = 0.0
        overall_bm_return = 0.0
        for pm in backtest_result.period_metrics:
            if pm['period_type'] == 'OVERALL':
                overall_alpha = pm.get('alpha') or 0.0
                overall_bm_return = pm.get('benchmark_return') or 0.0
                break

        # DB 저장
        db = TradingDB()

        # 1. backtest_runs (기존 테이블 + v2.1 컬럼)
        db_payload = {
            'strategy_name': 'EIMAS_Portfolio',
            'start_date': metrics.start_date,
            'end_date': metrics.end_date,
            'initial_capital': config.initial_capital,
            'final_capital': config.initial_capital * (1 + metrics.total_return),
            'total_return': metrics.total_return,
            'annual_return': metrics.annualized_return,
            'benchmark_return': overall_bm_return,
            'alpha': overall_alpha,
            'volatility': metrics.annualized_volatility,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'total_trades': metrics.num_trades,
            'winning_trades': int(metrics.win_rate * metrics.num_periods),
            'losing_trades': metrics.num_periods - int(metrics.win_rate * metrics.num_periods),
            'win_rate': metrics.win_rate,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'profit_factor': metrics.profit_factor,
            'avg_holding_days': 30,  # monthly rebalance
            'total_commission': metrics.total_transaction_costs,
            'total_slippage': 0.0,
            'total_short_cost': 0.0,
            'parameters': {
                'rebalance_frequency': config.rebalance_frequency,
                'transaction_cost_bps': config.transaction_cost_bps,
                'initial_capital': config.initial_capital,
                'benchmark': 'SPY',
                'cost_model': 'TradingCostModel',
            },
            'trades': []  # ticker-level entry/exit는 snapshots에서 복원 가능
        }
        result.backtest_run_id = db.save_backtest_run(db_payload)
        run_id = result.backtest_run_id

        # 2. backtest_daily_nav (일별 NAV + ticker attribution)
        db.save_backtest_daily_nav(run_id, backtest_result.daily_nav_records)

        # 3. backtest_snapshots (리밸런싱 스냅샷)
        db.save_backtest_snapshots(run_id, backtest_result.snapshot_records)

        # 4. backtest_period_metrics (기간별 성과 분해)
        db.save_backtest_period_metrics(run_id, backtest_result.period_metrics)

        # v2.1 컬럼 업데이트 (git_commit, benchmark, cost_model)
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=str(Path(__file__).parent),
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = None

        conn = sqlite3.connect(db.db_path)
        conn.execute("""
            UPDATE backtest_runs
            SET git_commit = ?, benchmark = 'SPY', cost_model = 'TradingCostModel'
            WHERE id = ?
        """, (git_hash, run_id))
        conn.commit()
        conn.close()

        print(f"  ✅ Backtest Complete:")
        print(f"     Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"     Max DD: {metrics.max_drawdown*100:.1f}%")
        print(f"     VaR 95%: {metrics.var_95*100:.2f}%")
        print(f"     Alpha: {overall_alpha*100:.2f}%")
        print(f"     Snapshots: {len(backtest_result.snapshot_records)}")
        print(f"     DB Saved: Run ID {run_id}")

    except Exception as e:
        print(f"⚠️ Backtest Error: {e}")
        import traceback
        traceback.print_exc()


def run_performance_attribution(result: EIMASResult, enable: bool):
    """[Phase 6.2] 성과 귀속 분석 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.2] Running Performance Attribution...")

    try:
        # Need portfolio weights and benchmark weights
        portfolio_weights = result.portfolio_weights
        if not portfolio_weights:
            print("⚠️ No portfolio weights for attribution")
            return

        # Use 60/40 as default benchmark
        benchmark_weights = {}
        equity_tickers = ['SPY', 'QQQ', 'IWM']
        bond_tickers = ['TLT', 'IEF']

        # Distribute 60% to equities, 40% to bonds
        for ticker in portfolio_weights.keys():
            if ticker in equity_tickers:
                benchmark_weights[ticker] = 0.60 / len([t for t in portfolio_weights if t in equity_tickers])
            elif ticker in bond_tickers:
                benchmark_weights[ticker] = 0.40 / len([t for t in portfolio_weights if t in bond_tickers])
            else:
                benchmark_weights[ticker] = 0.0

        # Normalize
        total = sum(benchmark_weights.values())
        if total > 0:
            benchmark_weights = {k: v/total for k, v in benchmark_weights.items()}

        # Use assumed returns (would need actual returns from market data)
        # This is a simplified example - real implementation would calculate from prices
        portfolio_returns = {ticker: 0.10 for ticker in portfolio_weights.keys()}
        benchmark_returns = {ticker: 0.08 for ticker in benchmark_weights.keys()}

        # Brinson Attribution
        brinson = BrinsonAttribution()
        attribution = brinson.compute(
            portfolio_weights, portfolio_returns,
            benchmark_weights, benchmark_returns
        )

        result.performance_attribution = attribution.to_dict()

        # Active Share
        active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
        result.performance_attribution['active_share'] = active_share

        print(f"  ✅ Attribution Complete:")
        print(f"     Excess Return: {attribution.excess_return*100:.2f}%")
        print(f"     Allocation Effect: {attribution.allocation_effect*100:.2f}%")
        print(f"     Active Share: {active_share*100:.1f}%")

    except Exception as e:
        print(f"⚠️ Attribution Error: {e}")
        import traceback
        traceback.print_exc()


def run_tactical_allocation(result: EIMASResult):
    """[Phase 2.11] 전술적 자산배분 (Always run if portfolio exists)"""
    if not result.portfolio_weights:
        return

    print("\n[Phase 2.11] Running Tactical Asset Allocation...")

    try:
        # Asset class mapping
        asset_class_mapping = {
            'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity', 'DIA': 'equity',
            'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond', 'HYG': 'bond',
            'GLD': 'commodity', 'DBC': 'commodity',
            'BTC-USD': 'crypto', 'ETH-USD': 'crypto'
        }

        # Get current regime
        regime = result.regime.get('regime', 'Neutral')
        regime_confidence = result.regime.get('confidence', 0.5)

        # Tactical allocator
        taa = TacticalAssetAllocator(
            strategic_weights=result.portfolio_weights,
            asset_class_mapping=asset_class_mapping,
            max_tilt_pct=0.15
        )

        # Compute tactical weights
        tactical_weights = taa.compute_tactical_weights(
            regime=regime,
            confidence=regime_confidence
        )

        result.tactical_weights = tactical_weights

        # Calculate adjustment
        total_adjustment = sum(abs(tactical_weights[t] - result.portfolio_weights[t])
                              for t in tactical_weights)

        print(f"  ✅ Tactical Allocation Complete:")
        print(f"     Regime: {regime}")
        print(f"     Total Adjustment: {total_adjustment*100:.1f}%")

    except Exception as e:
        print(f"⚠️ Tactical Allocation Error: {e}")
        import traceback
        traceback.print_exc()


def run_stress_test(result: EIMASResult, enable: bool):
    """[Phase 6.3] 스트레스 테스트 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.3] Running Stress Testing...")

    try:
        portfolio_weights = result.portfolio_weights
        if not portfolio_weights:
            print("⚠️ No portfolio weights for stress test")
            return

        # Stress test engine
        engine = StressTestEngine(
            portfolio_weights=portfolio_weights,
            portfolio_value=1_000_000
        )

        # Run historical scenarios
        historical_results = engine.run_all_historical()

        # Run hypothetical scenarios
        hypothetical_results = engine.run_all_hypothetical()

        # Extreme scenario
        extreme_result = engine.extreme_scenario("severe")

        # Store results
        result.stress_test_results = {
            'historical': [r.to_dict() for r in historical_results],
            'hypothetical': [r.to_dict() for r in hypothetical_results],
            'extreme': extreme_result.to_dict()
        }

        # Find worst case
        all_results = historical_results + hypothetical_results + [extreme_result]
        worst_case = max(all_results, key=lambda r: r.loss_pct)

        print(f"  ✅ Stress Test Complete:")
        print(f"     Scenarios Tested: {len(all_results)}")
        print(f"     Worst Case: {worst_case.scenario_name} ({worst_case.loss_pct*100:.1f}%)")

    except Exception as e:
        print(f"⚠️ Stress Test Error: {e}")
        import traceback
        traceback.print_exc()
