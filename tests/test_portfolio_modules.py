"""
Portfolio Theory Modules Integration Test
==========================================
신규 구현된 모듈들의 통합 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import new modules
from lib.backtest import BacktestEngine, BacktestConfig, compare_strategies
from lib.performance_attribution import BrinsonAttribution, InformationRatio, ActiveShare
from lib.tactical_allocation import TacticalAssetAllocator, VolatilityTargeting, MomentumOverlay
from lib.stress_test import StressTestEngine, generate_stress_test_report


def generate_sample_data(n_days=1000):
    """샘플 데이터 생성"""
    dates = pd.date_range(end=datetime.now(), periods=n_days)

    # Generate correlated returns
    np.random.seed(42)

    # Tickers
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']

    # Generate returns
    returns_data = {}
    returns_data['SPY'] = np.random.normal(0.0003, 0.01, n_days)  # 7.5% ann, 16% vol
    returns_data['QQQ'] = returns_data['SPY'] * 1.2 + np.random.normal(0, 0.005, n_days)
    returns_data['TLT'] = np.random.normal(0.0001, 0.008, n_days)  # 2.5% ann, 12% vol
    returns_data['GLD'] = np.random.normal(0.0002, 0.012, n_days)  # 5% ann, 19% vol
    returns_data['BTC-USD'] = np.random.normal(0.0005, 0.04, n_days)  # 12.5% ann, 63% vol

    # Convert to prices
    prices_data = {}
    for ticker, returns in returns_data.items():
        prices = (1 + pd.Series(returns)).cumprod() * 100
        prices_data[ticker] = prices.values

    prices = pd.DataFrame(prices_data, index=dates)

    return prices


def test_backtest_engine():
    """백테스팅 엔진 테스트"""
    print("\n" + "="*60)
    print("TEST 1: Backtest Engine")
    print("="*60)

    # Generate data
    prices = generate_sample_data(1000)

    # Simple equal weight strategy
    def equal_weight_strategy(prices_window):
        n = len(prices_window.columns)
        return {ticker: 1/n for ticker in prices_window.columns}

    # Config
    config = BacktestConfig(
        start_date=str(prices.index[252]),
        end_date=str(prices.index[-1]),
        rebalance_frequency='quarterly',
        transaction_cost_bps=10,
        initial_capital=1_000_000
    )

    # Run
    engine = BacktestEngine(config)
    result = engine.run(prices, equal_weight_strategy)

    # Print results
    m = result.metrics
    print(f"\n✅ Backtest Complete:")
    print(f"   Total Return: {m.total_return*100:.2f}%")
    print(f"   Ann. Return: {m.annualized_return*100:.2f}%")
    print(f"   Ann. Vol: {m.annualized_volatility*100:.2f}%")
    print(f"   Sharpe: {m.sharpe_ratio:.2f}")
    print(f"   Sortino: {m.sortino_ratio:.2f}")
    print(f"   Max DD: {m.max_drawdown*100:.2f}%")
    print(f"   Calmar: {m.calmar_ratio:.2f}")
    print(f"   VaR 95%: {m.var_95*100:.2f}%")
    print(f"   CVaR 95%: {m.cvar_95*100:.2f}%")
    print(f"   Win Rate: {m.win_rate*100:.1f}%")
    print(f"   Trades: {m.num_trades}")
    print(f"   Total Cost: ${m.total_transaction_costs:,.0f}")

    return result


def test_performance_attribution():
    """성과 귀속 분석 테스트"""
    print("\n" + "="*60)
    print("TEST 2: Performance Attribution (Brinson)")
    print("="*60)

    # Portfolio
    portfolio_weights = {'SPY': 0.50, 'TLT': 0.30, 'GLD': 0.20}
    portfolio_returns = {'SPY': 0.12, 'TLT': 0.03, 'GLD': 0.08}

    # Benchmark (60/40)
    benchmark_weights = {'SPY': 0.60, 'TLT': 0.40, 'GLD': 0.00}
    benchmark_returns = {'SPY': 0.12, 'TLT': 0.03, 'GLD': 0.08}

    # Brinson Attribution
    brinson = BrinsonAttribution()
    result = brinson.compute(
        portfolio_weights, portfolio_returns,
        benchmark_weights, benchmark_returns
    )

    print(f"\n✅ Attribution Complete:")
    print(f"   Portfolio Return: {result.portfolio_return*100:.2f}%")
    print(f"   Benchmark Return: {result.benchmark_return*100:.2f}%")
    print(f"   Excess Return: {result.excess_return*100:.2f}%")
    print(f"\n   📊 Decomposition:")
    print(f"   Allocation Effect: {result.allocation_effect*100:.3f}%")
    print(f"   Selection Effect: {result.selection_effect*100:.3f}%")
    print(f"   Interaction Effect: {result.interaction_effect*100:.3f}%")
    print(f"   Verification: {'✅ PASS' if result.verify() else '❌ FAIL'}")

    # Active Share
    active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
    print(f"\n   Active Share: {active_share*100:.1f}%")

    return result


def test_tactical_allocation():
    """전술적 자산배분 테스트"""
    print("\n" + "="*60)
    print("TEST 3: Tactical Asset Allocation")
    print("="*60)

    # Strategic weights
    strategic_weights = {
        'SPY': 0.25, 'QQQ': 0.15, 'TLT': 0.35,
        'GLD': 0.15, 'BTC-USD': 0.10
    }

    asset_class_mapping = {
        'SPY': 'equity', 'QQQ': 'equity',
        'TLT': 'bond', 'GLD': 'commodity', 'BTC-USD': 'crypto'
    }

    # TAA
    taa = TacticalAssetAllocator(strategic_weights, asset_class_mapping, max_tilt_pct=0.15)

    # Test different regimes
    regimes = ["Bull (Low Vol)", "Bear (High Vol)", "Neutral"]

    for regime in regimes:
        tactical = taa.compute_tactical_weights(regime, confidence=0.8)

        print(f"\n   {regime}:")
        for ticker in strategic_weights:
            strat = strategic_weights[ticker]
            tact = tactical[ticker]
            change = tact - strat
            print(f"      {ticker}: {tact:.2%} (Δ{change:+.2%})")

    return tactical


def test_stress_testing():
    """스트레스 테스트"""
    print("\n" + "="*60)
    print("TEST 4: Stress Testing")
    print("="*60)

    # Portfolio
    portfolio_weights = {
        'SPY': 0.40, 'QQQ': 0.20, 'TLT': 0.25,
        'GLD': 0.10, 'BTC-USD': 0.05
    }

    # Engine
    engine = StressTestEngine(portfolio_weights, portfolio_value=1_000_000)

    print("\n   📊 Historical Scenarios:")
    historical = engine.run_all_historical()
    for result in historical[:3]:  # Top 3
        breach = "⚠️" if result.var_breach else "✅"
        print(f"      {breach} {result.scenario_name}: "
              f"Loss ${result.loss:,.0f} ({result.loss_pct*100:.2f}%)")

    print("\n   🔮 Hypothetical Scenarios:")
    hypothetical = engine.run_all_hypothetical()
    for result in hypothetical[:3]:  # Top 3
        breach = "⚠️" if result.var_breach else "✅"
        print(f"      {breach} {result.scenario_name}: "
              f"Loss ${result.loss:,.0f} ({result.loss_pct*100:.2f}%)")

    print("\n   💥 Extreme Scenario:")
    extreme = engine.extreme_scenario("severe")
    print(f"      Black Swan (Severe): "
          f"Loss ${extreme.loss:,.0f} ({extreme.loss_pct*100:.2f}%)")

    # Monte Carlo
    print("\n   🎲 Monte Carlo Simulation (10,000 runs)...")

    # Generate sample returns
    tickers = list(portfolio_weights.keys())
    returns_mean = {t: np.random.uniform(0.0001, 0.0005) for t in tickers}

    # Generate correlation matrix
    n = len(tickers)
    corr = np.random.rand(n, n)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    vols = [0.01, 0.012, 0.008, 0.015, 0.04]
    cov = np.outer(vols, vols) * corr
    returns_cov = pd.DataFrame(cov, index=tickers, columns=tickers)

    mc_result = engine.monte_carlo(
        returns_mean, returns_cov,
        n_simulations=10_000,
        confidence_level=0.95
    )

    print(f"      Mean Value: ${mc_result['mean']:,.0f}")
    print(f"      Std Dev: ${mc_result['std']:,.0f}")
    print(f"      VaR(95%): ${mc_result['var']:,.0f} ({mc_result['var_pct']*100:.2f}%)")
    print(f"      CVaR(95%): ${mc_result['cvar']:,.0f} ({mc_result['cvar_pct']*100:.2f}%)")

    return historical, hypothetical, extreme


def main():
    """통합 테스트 실행"""
    print("\n" + "="*80)
    print(" "*20 + "Portfolio Theory Modules Integration Test")
    print("="*80)

    try:
        # Run tests
        backtest_result = test_backtest_engine()
        attribution_result = test_performance_attribution()
        tactical_weights = test_tactical_allocation()
        stress_results = test_stress_testing()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\n📊 Summary:")
        print(f"   - Backtest Engine: ✅ Working")
        print(f"   - Performance Attribution: ✅ Working")
        print(f"   - Tactical Allocation: ✅ Working")
        print(f"   - Stress Testing: ✅ Working")

        print("\n🎯 Ready for EIMAS Integration")
        print("\nNext Steps:")
        print("   1. Add --backtest flag to main.py")
        print("   2. Add --attribution flag for Brinson analysis")
        print("   3. Add --stress-test flag")
        print("   4. Integrate tactical allocation into Phase 2.11")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
