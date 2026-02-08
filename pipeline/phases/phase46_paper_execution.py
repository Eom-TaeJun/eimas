#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 4.6: Auto Paper Execution

Purpose:
    EIMAS 결과(trade_plan/AI 결론)를 목표가 기반 모의 주문으로 연결
"""

from __future__ import annotations

from lib.auto_paper_execution import (
    AutoPaperExecutionConfig,
    poll_pending_paper_orders,
    run_auto_paper_execution,
)
from pipeline.schemas import EIMASResult


def run_paper_execution(
    result: EIMASResult,
    enable: bool = False,
    account_name: str = "ra_auto",
    initial_capital: float = 100_000.0,
    poll_only: bool = False,
    run_backtest: bool = False,
    enforce_human_approval: bool = False,
):
    """[Phase 4.6] Auto LIMIT 주문 등록 + 대기 주문 폴링."""
    if not enable and not poll_only:
        return

    print("\n[Phase 4.6] Auto Paper Execution...")
    try:
        if poll_only:
            poll_result = poll_pending_paper_orders(
                account_name=account_name,
                initial_capital=initial_capital,
            )
            result.paper_execution = {
                "mode": "poll_only",
                **poll_result,
            }
            print(
                "      ✓ Poll complete: "
                f"filled={poll_result.get('poll_result', {}).get('filled', 0)}, "
                f"pending={poll_result.get('pending_count', 0)}"
            )
            return

        cfg = AutoPaperExecutionConfig(
            account_name=account_name,
            initial_capital=initial_capital,
            run_backtest=run_backtest,
            enforce_human_approval=enforce_human_approval,
        )
        execution_summary = run_auto_paper_execution(result.to_dict(), config=cfg)
        result.paper_execution = execution_summary
        if isinstance(execution_summary.get("backtest"), dict):
            result.paper_execution_backtest = execution_summary.get("backtest", {})

        print(
            "      ✓ Auto execution summary: "
            f"orders={len(execution_summary.get('registered_orders', []))}, "
            f"pending={execution_summary.get('pending_count', 0)}"
        )
    except Exception as e:
        print(f"      ⚠️ Auto paper execution error: {e}")
        result.paper_execution = {"error": str(e)}
