#!/usr/bin/env python3
"""
Auto paper execution runner.

Usage:
  python scripts/auto_paper_execution.py
  python scripts/auto_paper_execution.py --input outputs/eimas_20260209_003303.json --run-backtest
  python scripts/auto_paper_execution.py --poll-only --account ra_auto
"""

try:
    from _project_bootstrap import ensure_project_root
except ImportError:
    from scripts._project_bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(__file__)

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from lib.auto_paper_execution import (
    AutoPaperExecutionConfig,
    poll_pending_paper_orders,
    run_auto_paper_execution,
)


def _latest_eimas_json() -> Path:
    output_dir = PROJECT_ROOT / "outputs"
    candidates = sorted(output_dir.glob("eimas_*.json"))
    if not candidates:
        raise FileNotFoundError("No eimas_*.json found under outputs/")
    return candidates[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"auto_paper_execution_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run EIMAS auto paper execution")
    parser.add_argument("--input", help="EIMAS json path (default: latest outputs/eimas_*.json)")
    parser.add_argument("--broker", default="ibkr", help="Execution broker (default: ibkr)")
    parser.add_argument("--account", default="ra_auto", help="Paper account name")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial paper capital")
    parser.add_argument("--poll-only", action="store_true", help="Only poll pending orders")
    parser.add_argument("--run-backtest", action="store_true", help="Run allocation backtest")
    parser.add_argument(
        "--backtest-require-market-data",
        action="store_true",
        help="Disable synthetic fallback for backtest prices",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not place orders")
    parser.add_argument("--enforce-approval", action="store_true", help="Block when human approval is required")
    parser.add_argument("--max-orders", type=int, default=12, help="Maximum orders per run")
    parser.add_argument("--buy-buffer-bps", type=float, default=40.0, help="BUY limit buffer in bps")
    parser.add_argument("--sell-buffer-bps", type=float, default=40.0, help="SELL limit buffer in bps")
    parser.add_argument(
        "--max-order-notional-pct",
        type=float,
        default=0.20,
        help="Global max notional cap per order as portfolio ratio (default: 0.20)",
    )
    parser.add_argument(
        "--disable-asset-class",
        action="append",
        default=[],
        help="Disable order placement for an asset class (repeatable)",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "reports"), help="Summary output directory")
    args = parser.parse_args()

    if args.poll_only:
        summary = poll_pending_paper_orders(
            broker=args.broker,
            account_name=args.account,
            initial_capital=args.capital,
        )
        out_path = _save_summary(summary, Path(args.output_dir))
        print(f"Poll complete. Pending: {summary.get('pending_count', 0)}")
        print(f"Saved: {out_path}")
        return

    input_path = Path(args.input) if args.input else _latest_eimas_json()
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    result_data = _load_json(input_path)

    cfg = AutoPaperExecutionConfig(
        broker=args.broker,
        account_name=args.account,
        initial_capital=args.capital,
        buy_limit_buffer_bps=args.buy_buffer_bps,
        sell_limit_buffer_bps=args.sell_buffer_bps,
        max_orders=args.max_orders,
        run_backtest=args.run_backtest,
        allow_synthetic_backtest_fallback=not args.backtest_require_market_data,
        dry_run=args.dry_run,
        enforce_human_approval=args.enforce_approval,
        max_order_notional_pct=args.max_order_notional_pct,
        disabled_asset_classes=tuple(args.disable_asset_class),
    )
    summary = run_auto_paper_execution(result_data, config=cfg)
    out_path = _save_summary(summary, Path(args.output_dir))

    print(f"Registered orders: {len(summary.get('registered_orders', []))}")
    print(f"Pending orders: {summary.get('pending_count', 0)}")
    if isinstance(summary.get("backtest"), dict) and summary["backtest"].get("ok"):
        print(f"Backtest run_id: {summary['backtest'].get('run_id')}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
