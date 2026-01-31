#!/usr/bin/env python3
"""
EIMAS Backtest Runner v2.0
==========================
EIMAS 전략 백테스트 실행 및 리포트 생성

v2.0 변경사항 (2026-01-31):
- 복리 버그 수정된 backtester 사용
- 기본 기간: 2024-09-01 ~ 현재
- DB 저장 기능 추가
- 고정 포지션 사이징 (30%)

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --period 2023-01-01 2024-12-31
    python scripts/run_backtest.py --strategy EIMAS_Regime
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import argparse
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
    DEFAULT_START_DATE,
)


def run_backtest(
    start_date: str = None,
    end_date: str = None,
    strategies_to_run: list = None,
    save_to_db: bool = True,
    save_to_json: bool = True,
):
    """백테스트 실행

    Args:
        start_date: 시작일 (기본: 2024-09-01)
        end_date: 종료일 (기본: 오늘)
        strategies_to_run: 실행할 전략 이름 리스트 (기본: 전체)
        save_to_db: DB에 결과 저장 여부
        save_to_json: JSON 파일 저장 여부
    """
    print("=" * 70)
    print("EIMAS Backtest Runner v2.0")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Period: {start_date or DEFAULT_START_DATE} ~ {end_date or 'today'}")
    print(f"Position Sizing: FIXED 30% (no compounding)")
    print("=" * 70)

    # 전략 생성
    all_strategies = {
        'EIMAS_Regime': create_regime_based_strategy("SPY"),
        'Multi_Factor': create_multi_factor_strategy("SPY"),
        'MA_Crossover': create_ma_crossover_strategy(20, 50, "SPY"),
        'VIX_Mean_Reversion': create_vix_mean_reversion_strategy("SPY"),
        'Yield_Curve_Proxy': create_yield_curve_strategy("SPY"),
        'Copper_Gold_Proxy': create_copper_gold_strategy("SPY"),
    }

    # 실행할 전략 선택
    if strategies_to_run:
        strategies = {k: v for k, v in all_strategies.items() if k in strategies_to_run}
        if not strategies:
            print(f"Error: No matching strategies found. Available: {list(all_strategies.keys())}")
            return []
    else:
        # 기본: 주요 3개 전략만
        strategies = {
            'EIMAS_Regime': all_strategies['EIMAS_Regime'],
            'Multi_Factor': all_strategies['Multi_Factor'],
            'MA_Crossover': all_strategies['MA_Crossover'],
        }

    results = []

    for name, strategy in strategies.items():
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")

        bt = Backtester(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
        )

        try:
            result = bt.run()
            bt.print_report(result)

            # DB 저장
            if save_to_db:
                try:
                    bt.save_to_db(result)
                except Exception as e:
                    print(f"Warning: DB save failed: {e}")

            # JSON 저장
            if save_to_json:
                bt.save_to_json(result)

            results.append(result)

        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()

    # 비교 테이블
    if results:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON (Fixed Position 30%)")
        print("=" * 70)
        print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Trades':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.strategy_name:<25} {r.total_return:>+9.1f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown:>7.1f}% {r.win_rate:>7.1f}% {r.total_trades:>8}")

        # 최고 성과 전략
        if len(results) > 1:
            best = max(results, key=lambda x: x.sharpe_ratio)
            print(f"\nBest Sharpe: {best.strategy_name} ({best.sharpe_ratio:.2f})")

        # 종합 JSON 저장
        output = {
            'run_date': datetime.now().isoformat(),
            'version': '2.0',
            'settings': {
                'position_size': 0.3,
                'sizing_mode': 'fixed',
                'start_date': start_date or DEFAULT_START_DATE,
                'end_date': end_date or datetime.now().strftime("%Y-%m-%d"),
                'initial_capital': 100000
            },
            'results': [r.to_dict() for r in results]
        }

        output_path = '/home/tj/projects/autoai/eimas/outputs/backtest_results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSummary saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Backtest Complete!")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='EIMAS Backtest Runner v2.0')
    parser.add_argument(
        '--period',
        nargs=2,
        metavar=('START', 'END'),
        help='Backtest period (e.g., --period 2024-09-01 2025-01-31)'
    )
    parser.add_argument(
        '--strategy',
        nargs='+',
        help='Strategies to run (e.g., --strategy EIMAS_Regime Multi_Factor)'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip DB save'
    )
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip JSON save'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available strategies'
    )

    args = parser.parse_args()

    start_date = args.period[0] if args.period else None
    end_date = args.period[1] if args.period else None

    strategies = None
    if args.all:
        strategies = [
            'EIMAS_Regime', 'Multi_Factor', 'MA_Crossover',
            'VIX_Mean_Reversion', 'Yield_Curve_Proxy', 'Copper_Gold_Proxy'
        ]
    elif args.strategy:
        strategies = args.strategy

    run_backtest(
        start_date=start_date,
        end_date=end_date,
        strategies_to_run=strategies,
        save_to_db=not args.no_db,
        save_to_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
