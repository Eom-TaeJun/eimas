#!/usr/bin/env python3
"""
EIMAS CLI Tool
==============
명령줄 인터페이스 도구

Usage:
    python cli/eimas.py signal list
    python cli/eimas.py portfolio show
    python cli/eimas.py risk check
    python cli/eimas.py optimize --method sharpe
    python cli/eimas.py trade buy SPY 10
    python cli/eimas.py report daily
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import argparse
from datetime import datetime, date
from typing import Optional

# Rich for beautiful output (optional, falls back to plain text)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# ============================================================================
# Output Helpers
# ============================================================================

def print_header(title: str):
    """헤더 출력"""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{title}[/bold blue]")
        console.print("=" * 50)
    else:
        print(f"\n{title}")
        print("=" * 50)


def print_table(headers: list, rows: list, title: str = ""):
    """테이블 출력"""
    if RICH_AVAILABLE:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*[str(x) for x in row])
        console.print(table)
    else:
        if title:
            print(f"\n{title}")
        print("-" * 70)
        print("  ".join(f"{h:<12}" for h in headers))
        print("-" * 70)
        for row in rows:
            print("  ".join(f"{str(x):<12}" for x in row))


def print_success(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[green]✅ {msg}[/green]")
    else:
        print(f"✅ {msg}")


def print_error(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[red]❌ {msg}[/red]")
    else:
        print(f"❌ {msg}")


def print_warning(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[yellow]⚠️ {msg}[/yellow]")
    else:
        print(f"⚠️ {msg}")


# ============================================================================
# Signal Commands
# ============================================================================

def cmd_signal_list(args):
    """시그널 목록"""
    from lib.signal_pipeline import SignalPipeline

    print_header("EIMAS Signals")

    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()

    print(f"\nConsensus: {consensus['action'].upper()} ({consensus['conviction']:.0%})")
    print(f"Total Signals: {len(signals)}")

    if signals:
        headers = ["Source", "Action", "Ticker", "Conviction", "Time"]
        rows = [
            [s.source.value, s.action.value.upper(), s.ticker,
             f"{s.conviction:.0%}", s.timestamp.strftime("%H:%M")]
            for s in signals[:args.limit]
        ]
        print_table(headers, rows, "Recent Signals")


def cmd_signal_generate(args):
    """시그널 생성"""
    from lib.signal_pipeline import SignalPipeline

    print_header("Generating Signals")

    pipeline = SignalPipeline()
    signals = pipeline.run()

    print_success(f"Generated {len(signals)} signals")

    consensus = pipeline.get_consensus()
    print(f"Consensus: {consensus['action'].upper()} ({consensus['conviction']:.0%})")


# ============================================================================
# Portfolio Commands
# ============================================================================

def cmd_portfolio_show(args):
    """포트폴리오 조회"""
    from lib.paper_trader import PaperTrader

    print_header(f"Portfolio: {args.account}")

    trader = PaperTrader(account_name=args.account)
    summary = trader.get_portfolio_summary()

    print(f"\nCash: ${summary.cash:,.2f}")
    print(f"Positions: ${summary.positions_value:,.2f}")
    print(f"Total: ${summary.total_value:,.2f}")

    pnl_color = "green" if summary.total_pnl >= 0 else "red"
    if RICH_AVAILABLE:
        console.print(f"P&L: [{pnl_color}]${summary.total_pnl:,.2f} ({summary.total_pnl_pct:+.2f}%)[/{pnl_color}]")
    else:
        print(f"P&L: ${summary.total_pnl:,.2f} ({summary.total_pnl_pct:+.2f}%)")

    if summary.positions:
        headers = ["Ticker", "Qty", "Avg Cost", "Current", "P&L"]
        rows = [
            [ticker, f"{pos.quantity:.1f}", f"${pos.avg_cost:.2f}",
             f"${pos.current_price:.2f}", f"${pos.unrealized_pnl:+.2f}"]
            for ticker, pos in summary.positions.items()
        ]
        print_table(headers, rows, "\nPositions")


def cmd_portfolio_optimize(args):
    """포트폴리오 최적화"""
    from lib.portfolio_optimizer import PortfolioOptimizer

    print_header(f"Portfolio Optimization: {args.method.upper()}")

    assets = args.assets.split(",") if args.assets else ["SPY", "TLT", "GLD", "QQQ", "IWM"]
    print(f"Assets: {', '.join(assets)}")

    optimizer = PortfolioOptimizer(assets)
    optimizer.fetch_data()

    if args.method == "sharpe":
        result = optimizer.optimize_sharpe()
    elif args.method == "min_var":
        result = optimizer.optimize_min_variance()
    elif args.method == "risk_parity":
        result = optimizer.optimize_risk_parity()
    else:
        print_error(f"Unknown method: {args.method}")
        return

    print(f"\nExpected Return: {result.expected_return:.2%}")
    print(f"Expected Volatility: {result.expected_volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

    headers = ["Asset", "Weight"]
    rows = [[asset, f"{weight:.1%}"]
            for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1])
            if weight > 0.01]
    print_table(headers, rows, "\nOptimal Weights")


# ============================================================================
# Risk Commands
# ============================================================================

def cmd_risk_check(args):
    """리스크 체크"""
    from lib.paper_trader import PaperTrader
    from lib.risk_manager import RiskManager

    print_header("Risk Analysis")

    trader = PaperTrader(account_name=args.account)
    summary = trader.get_portfolio_summary()

    if not summary.positions:
        print_warning("No positions to analyze")
        return

    holdings = {t: p.market_value / summary.total_value for t, p in summary.positions.items()}

    rm = RiskManager()
    risk = rm.calculate_portfolio_risk(holdings, summary.total_value)

    risk_color = "green" if risk.risk_level.value == "low" else \
                 "red" if risk.risk_level.value == "high" else "yellow"

    if RICH_AVAILABLE:
        console.print(f"Risk Level: [{risk_color}]{risk.risk_level.value.upper()}[/{risk_color}]")
    else:
        print(f"Risk Level: {risk.risk_level.value.upper()}")

    print(f"\nVaR 95%: {risk.var_95:.2%}")
    print(f"VaR 99%: {risk.var_99:.2%}")
    print(f"CVaR 95%: {risk.cvar_95:.2%}")
    print(f"Max Drawdown: {risk.max_drawdown:.2%}")
    print(f"Volatility: {risk.annual_vol:.2%}")
    print(f"Sharpe Ratio: {risk.sharpe_estimate:.2f}")


# ============================================================================
# Trade Commands
# ============================================================================

def cmd_trade(args):
    """트레이드 실행"""
    from lib.paper_trader import PaperTrader

    print_header(f"Paper Trade: {args.side.upper()} {args.quantity} {args.ticker}")

    trader = PaperTrader(account_name=args.account)
    order = trader.execute_order(
        ticker=args.ticker,
        side=args.side,
        quantity=float(args.quantity),
    )

    if order.status.value == "filled":
        print_success(f"Order filled at ${order.filled_price:.2f}")
        print(f"Commission: ${order.commission:.2f}")
    else:
        print_error(f"Order {order.status.value}")


# ============================================================================
# Analysis Commands
# ============================================================================

def cmd_sectors(args):
    """섹터 분석"""
    from lib.sector_rotation import SectorRotationModel

    print_header("Sector Rotation Analysis")

    model = SectorRotationModel()
    result = model.analyze()

    print(f"\nEconomic Cycle: {result.cycle.current_cycle.value.replace('_', ' ').title()}")
    print(f"Confidence: {result.cycle.confidence:.0%}")

    print(f"\n🟢 Overweight: {', '.join(result.rotation_signal.overweight[:3])}")
    print(f"🔴 Underweight: {', '.join(result.rotation_signal.underweight[:3])}")

    headers = ["Rank", "Ticker", "Sector", "3M Mom", "RS", "Signal"]
    rows = [
        [s.rank, s.ticker, s.name[:20], f"{s.momentum_3m:.1f}%",
         f"{s.relative_strength:.1f}%", s.signal.value]
        for s in result.sector_stats[:args.limit]
    ]
    print_table(headers, rows, "\nSector Rankings")


def cmd_correlation(args):
    """상관관계 분석"""
    from lib.correlation_monitor import CorrelationMonitor

    print_header("Correlation Analysis")

    assets = args.assets.split(",") if args.assets else ["SPY", "TLT", "GLD", "QQQ"]
    print(f"Assets: {', '.join(assets)}")

    cm = CorrelationMonitor(assets)
    result = cm.analyze()

    print(f"\nRegime: {result.regime.value.upper()}")
    print(f"Avg Correlation: {result.diversification.average_correlation:.2f}")
    print(f"Diversification Ratio: {result.diversification.diversification_ratio:.2f}")

    if result.alerts:
        print(f"\n⚠️ Alerts:")
        for alert in result.alerts:
            print(f"  - {alert.message}")


def cmd_regime(args):
    """레짐 분석"""
    from lib.regime_detector import RegimeDetector

    print_header(f"Market Regime: {args.ticker}")

    detector = RegimeDetector(ticker=args.ticker)
    result = detector.detect()

    regime_color = "green" if "bull" in result.regime.value.lower() else \
                   "red" if "bear" in result.regime.value.lower() else "yellow"

    if RICH_AVAILABLE:
        console.print(f"Regime: [{regime_color}]{result.regime.value}[/{regime_color}]")
    else:
        print(f"Regime: {result.regime.value}")

    print(f"Trend: {result.trend_state.value}")
    print(f"Volatility: {result.volatility_state.value}")
    print(f"Confidence: {result.confidence:.0%}")


# ============================================================================
# Report Commands
# ============================================================================

def cmd_report_daily(args):
    """일일 리포트"""
    from lib.report_generator import ReportGenerator

    print_header("Daily Report Generation")

    generator = ReportGenerator()
    path = generator.generate_daily_report()

    print_success(f"Report saved: {path}")


def cmd_run(args):
    """통합 파이프라인 실행"""
    # Import from the new main_integrated which has the refactored pipeline
    from main_integrated import run_integrated_pipeline
    import asyncio

    print_header("Running Integrated Pipeline")
    
    asyncio.run(run_integrated_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=args.quick,
        generate_report=False # CLI doesn't have explicit report flag in run command yet, or use args if added
    ))


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="eimas",
        description="EIMAS - Economic Intelligence Multi-Agent System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run integrated pipeline")
    run_parser.add_argument("--realtime", action="store_true", help="Enable realtime monitoring")
    run_parser.add_argument("--duration", type=int, default=30, help="Realtime duration in seconds")
    run_parser.add_argument("--quick", action="store_true", help="Quick mode")
    run_parser.add_argument("--full", action="store_true", help="Full mode (standalone scripts)")
    run_parser.add_argument("--output", default="outputs", help="Output directory")
    run_parser.set_defaults(func=cmd_run)

    # Signal commands
    signal_parser = subparsers.add_parser("signal", help="Signal management")
    signal_sub = signal_parser.add_subparsers(dest="subcommand")

    signal_list = signal_sub.add_parser("list", help="List signals")
    signal_list.add_argument("--limit", type=int, default=10, help="Max signals")
    signal_list.set_defaults(func=cmd_signal_list)

    signal_gen = signal_sub.add_parser("generate", help="Generate signals")
    signal_gen.set_defaults(func=cmd_signal_generate)

    # Portfolio commands
    port_parser = subparsers.add_parser("portfolio", help="Portfolio management")
    port_sub = port_parser.add_subparsers(dest="subcommand")

    port_show = port_sub.add_parser("show", help="Show portfolio")
    port_show.add_argument("--account", default="default", help="Account name")
    port_show.set_defaults(func=cmd_portfolio_show)

    port_opt = port_sub.add_parser("optimize", help="Optimize portfolio")
    port_opt.add_argument("--method", default="sharpe", choices=["sharpe", "min_var", "risk_parity"])
    port_opt.add_argument("--assets", help="Comma-separated assets")
    port_opt.set_defaults(func=cmd_portfolio_optimize)

    # Risk commands
    risk_parser = subparsers.add_parser("risk", help="Risk analysis")
    risk_sub = risk_parser.add_subparsers(dest="subcommand")

    risk_check = risk_sub.add_parser("check", help="Check risk")
    risk_check.add_argument("--account", default="default", help="Account name")
    risk_check.set_defaults(func=cmd_risk_check)

    # Trade commands
    trade_parser = subparsers.add_parser("trade", help="Paper trading")
    trade_parser.add_argument("side", choices=["buy", "sell"])
    trade_parser.add_argument("ticker")
    trade_parser.add_argument("quantity", type=float)
    trade_parser.add_argument("--account", default="default", help="Account name")
    trade_parser.set_defaults(func=cmd_trade)

    # Analysis commands
    sectors_parser = subparsers.add_parser("sectors", help="Sector analysis")
    sectors_parser.add_argument("--limit", type=int, default=11, help="Max sectors")
    sectors_parser.set_defaults(func=cmd_sectors)

    corr_parser = subparsers.add_parser("correlation", help="Correlation analysis")
    corr_parser.add_argument("--assets", help="Comma-separated assets")
    corr_parser.set_defaults(func=cmd_correlation)

    regime_parser = subparsers.add_parser("regime", help="Market regime")
    regime_parser.add_argument("--ticker", default="SPY")
    regime_parser.set_defaults(func=cmd_regime)

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_sub = report_parser.add_subparsers(dest="subcommand")

    report_daily = report_sub.add_parser("daily", help="Daily report")
    report_daily.set_defaults(func=cmd_report_daily)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            print_error(str(e))
            if args.command == "signal":  # Debug mode
                raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
