#!/usr/bin/env python3
"""
EIMAS Backtest Runner
=====================
EIMAS 전략 백테스트 실행 및 리포트 생성
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from datetime import datetime
import json
from lib.backtester import (
    Backtester,
    create_regime_based_strategy,
    create_vix_mean_reversion_strategy,
    create_multi_factor_strategy,
    create_yield_curve_strategy,
    create_copper_gold_strategy,
    create_ma_crossover_strategy,
)


def run_conservative_backtest():
    """보수적 설정으로 백테스트 실행"""
    print("=" * 70)
    print("EIMAS Conservative Backtest")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Settings: position_size=0.3 (30% of capital per trade)")
    
    # 전략 생성 (position_size를 낮게 설정)
    strategies = []
    
    # EIMAS Regime (position 30%)
    regime = create_regime_based_strategy("SPY")
    regime.position_size = 0.3
    strategies.append(regime)
    
    # Multi-Factor (position 30%)  
    multi = create_multi_factor_strategy("SPY")
    multi.position_size = 0.3
    strategies.append(multi)
    
    # MA Crossover 비교용
    ma = create_ma_crossover_strategy(20, 50, "SPY")
    ma.position_size = 0.3
    strategies.append(ma)
    
    results = []
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        bt = Backtester(
            strategy=strategy,
            start_date="2020-01-01",
            end_date="2024-12-31",
            initial_capital=100000,
        )
        
        try:
            result = bt.run()
            bt.print_report(result)
            results.append(result)
        except Exception as e:
            print(f"Error: {strategy.name}: {e}")
    
    # 비교 테이블
    if results:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON (30% position)")
        print("=" * 70)
        print(f"\n{'Strategy':<25} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.strategy_name:<25} {r.total_return:>+9.1f}% {r.annual_return:>+9.1f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown:>7.1f}% {r.win_rate:>7.1f}%")
        
        # JSON 저장
        output = {
            'run_date': datetime.now().isoformat(),
            'settings': {
                'position_size': 0.3,
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'initial_capital': 100000
            },
            'results': [r.to_dict() for r in results]
        }
        
        with open('/home/tj/projects/autoai/eimas/outputs/backtest_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: outputs/backtest_results.json")
    
    print("\n" + "=" * 70)
    print("Backtest Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_conservative_backtest()
