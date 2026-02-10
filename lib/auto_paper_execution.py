#!/usr/bin/env python3
"""
EIMAS Auto Paper Execution
==========================
거시/기업/AI 결론 + trade_plan을 기반으로 목표가(LIMIT) 모의 주문을 자동 등록하고,
목표가 도달 시 체결을 폴링한다. 체결/대기/거절 내역은 TradingDB에 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path
import hashlib
import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf

from lib.broker_execution import BrokerOrderRequest, build_ibkr_paper_router
from lib.paper_trader import PaperTrader
from lib.backtest import BacktestConfig, BacktestEngine
from lib.ra_sql_store import save_backtest_metrics_to_sql
from lib.trading_db import (
    SessionType,
    Signal,
    SignalAction,
    SignalSource,
    TradingDB,
)


def _configure_yfinance_cache_dir() -> None:
    cache_dir = os.getenv("EIMAS_YFINANCE_CACHE_DIR", "/tmp/eimas_yfinance_cache").strip()
    if not cache_dir or cache_dir.lower() in {"off", "none", "disable", "false", "0"}:
        return
    target = Path(cache_dir).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(target))
    except Exception:
        return


_configure_yfinance_cache_dir()


@dataclass
class AutoPaperExecutionConfig:
    broker: str = "ibkr"
    account_name: str = "ra_auto"
    initial_capital: float = 100_000.0
    buy_limit_buffer_bps: float = 40.0
    sell_limit_buffer_bps: float = 40.0
    min_order_notional: float = 100.0
    min_delta_weight: float = 0.005
    max_orders: int = 12
    enforce_human_approval: bool = False
    allow_buys_when_hold: bool = True
    poll_pending: bool = True
    run_backtest: bool = False
    backtest_lookback_days: int = 756
    allow_synthetic_backtest_fallback: bool = True
    dry_run: bool = False
    idempotency_scope: str = "daily"
    strategy_tag: str = "eimas.auto_paper_execution"
    max_order_notional_pct: float = 0.20
    disabled_asset_classes: Tuple[str, ...] = ()
    asset_policy_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetClassPolicy:
    asset_class: str
    min_notional: float
    max_notional_pct: float
    quantity_precision: int
    allow_fractional: bool
    tradable: bool = True


ASSET_POLICY_VERSION = "us-trader-v1.1"


_CRYPTO_TICKERS = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "DOT",
    "MATIC",
}


_US_BOND_ETF_TICKERS = {
    "TLT",
    "IEF",
    "SHY",
    "BND",
    "AGG",
    "LQD",
    "HYG",
    "BIL",
    "SGOV",
}


_US_COMMODITY_ETF_TICKERS = {
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "DBA",
    "DBC",
}


_US_ETF_TICKERS = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VEA",
    "VWO",
    "EEM",
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLI",
    "XLP",
    "XLU",
    "XLY",
    "XLC",
    "XLB",
    "UUP",
}


_DEFAULT_ASSET_CLASS_POLICIES: Dict[str, AssetClassPolicy] = {
    "us_equity": AssetClassPolicy(
        asset_class="us_equity",
        min_notional=100.0,
        max_notional_pct=0.10,
        quantity_precision=0,
        allow_fractional=False,
    ),
    "us_etf": AssetClassPolicy(
        asset_class="us_etf",
        min_notional=100.0,
        max_notional_pct=0.15,
        quantity_precision=0,
        allow_fractional=False,
    ),
    "us_bond_etf": AssetClassPolicy(
        asset_class="us_bond_etf",
        min_notional=100.0,
        max_notional_pct=0.20,
        quantity_precision=0,
        allow_fractional=False,
    ),
    "us_commodity_etf": AssetClassPolicy(
        asset_class="us_commodity_etf",
        min_notional=100.0,
        max_notional_pct=0.12,
        quantity_precision=0,
        allow_fractional=False,
    ),
    "korea_equity": AssetClassPolicy(
        asset_class="korea_equity",
        min_notional=100.0,
        max_notional_pct=0.10,
        quantity_precision=0,
        allow_fractional=False,
    ),
    "crypto_spot": AssetClassPolicy(
        asset_class="crypto_spot",
        min_notional=50.0,
        max_notional_pct=0.12,
        quantity_precision=4,
        allow_fractional=True,
    ),
    "futures": AssetClassPolicy(
        asset_class="futures",
        min_notional=0.0,
        max_notional_pct=0.0,
        quantity_precision=0,
        allow_fractional=False,
        tradable=False,
    ),
    "index": AssetClassPolicy(
        asset_class="index",
        min_notional=0.0,
        max_notional_pct=0.0,
        quantity_precision=0,
        allow_fractional=False,
        tradable=False,
    ),
    "unknown": AssetClassPolicy(
        asset_class="unknown",
        min_notional=100.0,
        max_notional_pct=0.05,
        quantity_precision=0,
        allow_fractional=False,
    ),
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _priority_value(raw: Any) -> int:
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        upper = raw.strip().upper()
        mapping = {"HIGH": 1, "MEDIUM": 5, "LOW": 9}
        if upper in mapping:
            return mapping[upper]
        if raw.isdigit():
            return int(raw)
    return 99


def _infer_session_type(now: datetime) -> SessionType:
    hour = now.hour
    minute = now.minute
    hm = hour * 100 + minute
    if 400 <= hm < 930:
        return SessionType.PRE_MARKET
    if 930 <= hm < 1100:
        return SessionType.OPENING
    if 1100 <= hm < 1500:
        return SessionType.MID_DAY
    if 1500 <= hm < 1600:
        return SessionType.POWER_HOUR
    return SessionType.AFTER_HOURS


def _extract_trade_candidates(result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    trade_plan = result_data.get("trade_plan")
    if not isinstance(trade_plan, list) or not trade_plan:
        rd = result_data.get("rebalance_decision", {})
        trade_plan = rd.get("trade_plan", [])
    if not isinstance(trade_plan, list) or not trade_plan:
        op = result_data.get("operational_report", {})
        trade_plan = ((op.get("rebalance_plan") or {}).get("trades")) or []
    if not isinstance(trade_plan, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in trade_plan:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).upper().strip()
        if action not in {"BUY", "SELL"}:
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        normalized.append({**item, "ticker": ticker, "action": action})

    normalized.sort(key=lambda x: _priority_value(x.get("priority")))
    return normalized


def _extract_delta_weight(item: Dict[str, Any]) -> float:
    if "delta_weight" in item:
        return _safe_float(item.get("delta_weight"))
    current_w = _safe_float(item.get("current_weight"))
    target_w = _safe_float(item.get("target_weight"))
    return target_w - current_w


def _is_ascii_upper_numeric(value: str) -> bool:
    return value.isascii() and value.replace(".", "").replace("-", "").isalnum()


def _classify_asset_class(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return "unknown"
    if t.startswith("^"):
        return "index"
    if t.endswith("=F"):
        return "futures"
    if t.endswith("-USD") or t in _CRYPTO_TICKERS:
        return "crypto_spot"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "korea_equity"
    if t in _US_BOND_ETF_TICKERS:
        return "us_bond_etf"
    if t in _US_COMMODITY_ETF_TICKERS:
        return "us_commodity_etf"
    if t in _US_ETF_TICKERS:
        return "us_etf"
    if _is_ascii_upper_numeric(t) and len(t) <= 5:
        return "us_equity"
    return "unknown"


def _resolve_asset_policy(
    cfg: AutoPaperExecutionConfig,
    asset_class: str,
) -> AssetClassPolicy:
    base = _DEFAULT_ASSET_CLASS_POLICIES.get(
        asset_class,
        _DEFAULT_ASSET_CLASS_POLICIES["unknown"],
    )
    overrides = cfg.asset_policy_overrides if isinstance(cfg.asset_policy_overrides, dict) else {}
    raw_override = overrides.get(asset_class)
    if not isinstance(raw_override, dict):
        return base

    quantity_precision = int(
        _safe_float(
            raw_override.get("quantity_precision", base.quantity_precision),
            float(base.quantity_precision),
        )
    )
    quantity_precision = max(0, min(quantity_precision, 8))
    return AssetClassPolicy(
        asset_class=base.asset_class,
        min_notional=max(0.0, _safe_float(raw_override.get("min_notional", base.min_notional), base.min_notional)),
        max_notional_pct=max(
            0.0,
            _safe_float(raw_override.get("max_notional_pct", base.max_notional_pct), base.max_notional_pct),
        ),
        quantity_precision=quantity_precision,
        allow_fractional=_safe_bool(raw_override.get("allow_fractional", base.allow_fractional), base.allow_fractional),
        tradable=_safe_bool(raw_override.get("tradable", base.tradable), base.tradable),
    )


def _normalized_disabled_asset_classes(cfg: AutoPaperExecutionConfig) -> set[str]:
    raw = cfg.disabled_asset_classes
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (tuple, list, set)):
        values = list(raw)
    else:
        values = []
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _round_quantity(quantity: float, *, allow_fractional: bool, precision: int) -> float:
    if quantity <= 0:
        return 0.0
    if not allow_fractional:
        return float(int(quantity))
    bounded = max(0, min(int(precision), 8))
    return float(round(quantity, bounded))


def _build_research_signal(result_data: Dict[str, Any]) -> Signal:
    final_rec = str(result_data.get("final_recommendation", "HOLD")).upper()
    action_map = {
        "BULLISH": SignalAction.BUY,
        "BEARISH": SignalAction.SELL,
        "HOLD": SignalAction.HOLD,
        "NEUTRAL": SignalAction.HOLD,
    }
    action = action_map.get(final_rec, SignalAction.HOLD)
    confidence = _safe_float(result_data.get("confidence", 0.5), 0.5)
    regime = (
        result_data.get("regime", {}).get("regime")
        if isinstance(result_data.get("regime"), dict)
        else "UNKNOWN"
    )
    risk = _safe_float(result_data.get("risk_score", 50.0), 50.0)
    company_count = len((result_data.get("company_ra_analysis", {}) or {}).get("companies", []) or [])

    reasoning = (
        f"EIMAS auto execution | regime={regime} | risk={risk:.1f} | "
        f"recommendation={final_rec} | company_coverage={company_count}"
    )
    metadata = {
        "regime": regime,
        "risk_score": risk,
        "final_recommendation": final_rec,
        "company_coverage_count": company_count,
        "macro_liquidity_regime": (result_data.get("fred_summary", {}) or {}).get("liquidity_regime", ""),
    }

    return Signal(
        source=SignalSource.EIMAS_PIPELINE,
        action=action,
        ticker="SPY",
        conviction=max(0.0, min(confidence, 1.0)),
        reasoning=reasoning,
        metadata=metadata,
    )


def _resolve_idempotency_scope_date(
    scope: str,
    result_data: Dict[str, Any],
) -> str:
    normalized = (scope or "").strip().lower()
    if normalized == "run":
        ts = str(result_data.get("timestamp", "")).strip()
        if ts:
            return ts[:19]
    return datetime.now().strftime("%Y-%m-%d")


def _build_order_explainability(
    result_data: Dict[str, Any],
    item: Dict[str, Any],
    *,
    ticker: str,
    side: str,
    asset_class: str,
    delta_weight: float,
    current_price: float,
    limit_price: float,
    target_notional_requested: float,
    target_notional_effective: float,
    min_notional_threshold: float,
    max_notional_cap: float,
    quantity_precision: int,
    allow_fractional: bool,
    notional_capped: bool,
) -> Dict[str, Any]:
    regime = (result_data.get("regime") or {})
    op_report = (result_data.get("operational_report") or {})
    rebalance_plan = (op_report.get("rebalance_plan") or {})
    approval = rebalance_plan.get("approval") or {}

    return {
        "source": "trade_plan",
        "ticker": ticker,
        "side": side,
        "asset_class": asset_class,
        "asset_policy_version": ASSET_POLICY_VERSION,
        "delta_weight": round(delta_weight, 6),
        "current_price": round(current_price, 6),
        "limit_price": round(limit_price, 6),
        "target_notional_requested": round(target_notional_requested, 6),
        "target_notional_effective": round(target_notional_effective, 6),
        "min_notional_threshold": round(min_notional_threshold, 6),
        "max_notional_cap": round(max_notional_cap, 6),
        "quantity_precision": int(quantity_precision),
        "allow_fractional": bool(allow_fractional),
        "notional_capped": bool(notional_capped),
        "trade_priority": item.get("priority"),
        "trade_reason": item.get("reason", ""),
        "final_recommendation": str(result_data.get("final_recommendation", "HOLD")).upper(),
        "confidence": round(_safe_float(result_data.get("confidence", 0.0)), 4),
        "risk_score": round(_safe_float(result_data.get("risk_score", 50.0)), 2),
        "risk_level": str(result_data.get("risk_level", "")),
        "regime": regime.get("regime") if isinstance(regime, dict) else "",
        "regime_confidence": _safe_float(regime.get("confidence"), 0.0) if isinstance(regime, dict) else 0.0,
        "requires_human_approval": bool(approval.get("requires_human_approval", False)),
        "approval_reason": approval.get("approval_reason", ""),
    }


def run_auto_paper_execution(
    result_data: Dict[str, Any],
    config: Optional[AutoPaperExecutionConfig] = None,
) -> Dict[str, Any]:
    """
    EIMAS 결과 기반 자동 목표가 모의 주문 실행.

    Returns:
        실행/대기/체결/백테스트 요약 딕셔너리
    """
    cfg = config or AutoPaperExecutionConfig()
    summary: Dict[str, Any] = {
        "enabled": True,
        "timestamp": datetime.now().isoformat(),
        "broker": cfg.broker,
        "asset_policy_version": ASSET_POLICY_VERSION,
        "account": cfg.account_name,
        "dry_run": cfg.dry_run,
        "registered_orders": [],
        "skipped": [],
        "poll_result": {},
        "pending_count": 0,
        "signal_id": None,
        "approval_gate_blocked": False,
        "backtest": {},
    }

    if not isinstance(result_data, dict):
        summary["enabled"] = False
        summary["error"] = "invalid_result_data"
        return summary

    trade_candidates = _extract_trade_candidates(result_data)
    if not trade_candidates:
        summary["skipped"].append({"reason": "no_trade_candidates"})
        if not cfg.run_backtest:
            summary["enabled"] = False
            summary["error"] = "no_trade_candidates"
            return summary
        summary["error"] = "no_trade_candidates_backtest_only"

    requires_approval = bool(
        (
            ((result_data.get("operational_report") or {}).get("rebalance_plan") or {})
            .get("approval", {})
            .get("requires_human_approval", False)
        )
    )
    if trade_candidates and requires_approval and cfg.enforce_human_approval:
        summary["approval_gate_blocked"] = True
        summary["enabled"] = False
        summary["error"] = "human_approval_required"
        return summary

    if (cfg.broker or "").strip().lower() != "ibkr":
        summary["enabled"] = False
        summary["error"] = f"unsupported_broker:{cfg.broker}"
        return summary

    db = TradingDB()
    signal = _build_research_signal(result_data)
    summary["signal_id"] = db.save_signal(signal)

    trader = PaperTrader(
        initial_capital=cfg.initial_capital,
        account_name=cfg.account_name,
    )
    account_summary = trader.get_portfolio_summary()
    portfolio_value = max(account_summary.total_value, 1.0)
    router = build_ibkr_paper_router(
        account_name=cfg.account_name,
        initial_capital=cfg.initial_capital,
        db=db,
        trader=trader,
        dry_run=cfg.dry_run,
    )
    idempotency_scope_date = _resolve_idempotency_scope_date(
        cfg.idempotency_scope,
        result_data,
    )
    summary["idempotency_scope_date"] = idempotency_scope_date

    final_rec = str(result_data.get("final_recommendation", "HOLD")).upper()
    selected = trade_candidates[: max(1, int(cfg.max_orders))]
    disabled_asset_classes = _normalized_disabled_asset_classes(cfg)
    summary["disabled_asset_classes"] = sorted(disabled_asset_classes)
    prefetch_tickers = set()
    for item in selected:
        ticker = str(item.get("ticker", "")).upper()
        if not ticker:
            continue
        asset_class = _classify_asset_class(ticker)
        policy = _resolve_asset_policy(cfg, asset_class)
        if asset_class in disabled_asset_classes or not policy.tradable:
            continue
        prefetch_tickers.add(ticker)
    price_map = trader.get_current_prices(sorted(prefetch_tickers))
    session = _infer_session_type(datetime.now())

    for item in selected:
        ticker = str(item.get("ticker", "")).upper()
        side = "buy" if str(item.get("action", "")).upper() == "BUY" else "sell"
        asset_class = _classify_asset_class(ticker)
        policy = _resolve_asset_policy(cfg, asset_class)

        if asset_class in disabled_asset_classes:
            summary["skipped"].append(
                {
                    "ticker": ticker,
                    "asset_class": asset_class,
                    "reason": "asset_class_disabled",
                }
            )
            continue
        if not policy.tradable:
            summary["skipped"].append(
                {
                    "ticker": ticker,
                    "asset_class": asset_class,
                    "reason": "asset_class_not_tradable",
                }
            )
            continue

        if final_rec == "HOLD" and side == "buy" and not cfg.allow_buys_when_hold:
            summary["skipped"].append(
                {"ticker": ticker, "asset_class": asset_class, "reason": "hold_guard"}
            )
            continue

        current_price = _safe_float(price_map.get(ticker), 0.0)
        if current_price <= 0:
            current_price = trader.get_current_price(ticker)
        if current_price <= 0:
            summary["skipped"].append(
                {"ticker": ticker, "asset_class": asset_class, "reason": "price_unavailable"}
            )
            continue

        delta_weight = abs(_extract_delta_weight(item))
        if delta_weight < cfg.min_delta_weight:
            summary["skipped"].append(
                {"ticker": ticker, "asset_class": asset_class, "reason": "delta_below_threshold"}
            )
            continue

        target_notional_requested = portfolio_value * delta_weight
        min_notional_threshold = max(cfg.min_order_notional, policy.min_notional)
        if target_notional_requested < min_notional_threshold:
            summary["skipped"].append(
                {
                    "ticker": ticker,
                    "asset_class": asset_class,
                    "reason": "notional_too_small",
                    "requested_notional": target_notional_requested,
                    "min_notional_threshold": min_notional_threshold,
                }
            )
            continue

        policy_cap = portfolio_value * max(policy.max_notional_pct, 0.0)
        global_cap = portfolio_value * max(cfg.max_order_notional_pct, 0.0)
        if global_cap > 0:
            max_notional_cap = policy_cap if policy_cap > 0 else global_cap
            max_notional_cap = min(max_notional_cap, global_cap)
        else:
            max_notional_cap = policy_cap

        target_notional_effective = target_notional_requested
        notional_capped = False
        if max_notional_cap > 0 and target_notional_effective > max_notional_cap:
            target_notional_effective = max_notional_cap
            notional_capped = True

        if target_notional_effective < min_notional_threshold:
            summary["skipped"].append(
                {
                    "ticker": ticker,
                    "asset_class": asset_class,
                    "reason": "cap_below_min_notional",
                    "effective_notional": target_notional_effective,
                    "min_notional_threshold": min_notional_threshold,
                }
            )
            continue

        quantity = _round_quantity(
            target_notional_effective / current_price,
            allow_fractional=policy.allow_fractional,
            precision=policy.quantity_precision,
        )
        if quantity <= 0:
            summary["skipped"].append(
                {"ticker": ticker, "asset_class": asset_class, "reason": "quantity_zero"}
            )
            continue

        if side == "sell":
            current_qty = _safe_float(trader.positions.get(ticker, {}).get("quantity", 0.0), 0.0)
            quantity = min(quantity, current_qty)
            if quantity <= 0:
                summary["skipped"].append(
                    {"ticker": ticker, "asset_class": asset_class, "reason": "no_position_to_sell"}
                )
                continue

        if side == "buy":
            limit_price = current_price * (1.0 - cfg.buy_limit_buffer_bps / 10000.0)
        else:
            limit_price = current_price * (1.0 + cfg.sell_limit_buffer_bps / 10000.0)
        target_notional_effective = quantity * current_price
        explainability = _build_order_explainability(
            result_data,
            item,
            ticker=ticker,
            side=side,
            asset_class=asset_class,
            delta_weight=delta_weight,
            current_price=current_price,
            limit_price=limit_price,
            target_notional_requested=target_notional_requested,
            target_notional_effective=target_notional_effective,
            min_notional_threshold=min_notional_threshold,
            max_notional_cap=max_notional_cap,
            quantity_precision=policy.quantity_precision,
            allow_fractional=policy.allow_fractional,
            notional_capped=notional_capped,
        )
        submit_result = router.submit_limit_order(
            BrokerOrderRequest(
                ticker=ticker,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                session=session,
                reference_price=current_price,
                strategy_tag=cfg.strategy_tag,
                explainability=explainability,
            ),
            scope_date=idempotency_scope_date,
        )

        summary["registered_orders"].append(
            {
                "execution_id": submit_result.get("execution_id"),
                "external_order_id": submit_result.get("external_order_id"),
                "idempotency_key": submit_result.get("idempotency_key"),
                "deduplicated": bool(submit_result.get("deduplicated", False)),
                "broker": submit_result.get("broker", cfg.broker),
                "ticker": ticker,
                "asset_class": asset_class,
                "side": side,
                "quantity": quantity,
                "current_price": current_price,
                "limit_price": limit_price,
                "target_notional_requested": target_notional_requested,
                "target_notional_effective": target_notional_effective,
                "notional_capped": notional_capped,
                "executed_price": _safe_float(submit_result.get("executed_price"), 0.0),
                "order_state": submit_result.get("order_state"),
                "status": submit_result.get("status"),
            }
        )

    if cfg.poll_pending and not cfg.dry_run:
        poll_result = router.poll_pending_orders()
        summary["poll_result"] = poll_result

    summary["pending_count"] = router.pending_order_count()
    if cfg.run_backtest:
        summary["backtest"] = run_allocation_backtest_from_result(
            result_data=result_data,
            db=db,
            lookback_days=cfg.backtest_lookback_days,
            allow_synthetic_fallback=cfg.allow_synthetic_backtest_fallback,
        )
    return summary


def _download_close_panel(
    tickers: List[str],
    lookback_days: int,
    allow_synthetic_fallback: Optional[bool] = None,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    source_tags: List[str] = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{int(lookback_days)}d", interval="1d", progress=False)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if close_col is None:
            continue
        series_map[ticker] = df[close_col]
    if series_map:
        source_tags.append("yfinance")

    missing_tickers = [ticker for ticker in tickers if ticker not in series_map]
    if missing_tickers:
        local_panel = _load_close_panel_from_local_market_sources(
            tickers=missing_tickers,
            lookback_days=lookback_days,
        )
        if not local_panel.empty:
            for ticker in local_panel.columns:
                if ticker in series_map:
                    continue
                series_map[ticker] = local_panel[ticker]
            source_tags.append("local_market_db")

    if allow_synthetic_fallback is None:
        allow_synthetic = os.getenv("EIMAS_BACKTEST_ALLOW_SYNTHETIC_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
    else:
        allow_synthetic = bool(allow_synthetic_fallback)

    if not series_map:
        if allow_synthetic:
            synthetic = _build_synthetic_close_panel(tickers, lookback_days)
            if not synthetic.empty:
                synthetic.attrs["synthetic_fallback"] = True
                synthetic.attrs["market_data_source"] = "synthetic_fallback"
                return synthetic
        return pd.DataFrame()
    panel = pd.DataFrame(series_map).sort_index()
    panel = panel.ffill().dropna(how="all")
    panel = panel.dropna(how="any")
    panel.attrs["synthetic_fallback"] = False
    panel.attrs["market_data_source"] = ",".join(source_tags) if source_tags else "unknown"
    return panel


def _infer_backtest_weights_from_result(result_data: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
    """
    Infer backtest target weights when explicit allocation weights are missing.
    Priority:
    1) allocation_result.weights
    2) portfolio_weights
    3) deterministic RA macro ETF basket fallback
    """
    alloc = (result_data.get("allocation_result") or {}).get("weights")
    if isinstance(alloc, dict) and alloc:
        return dict(alloc), "allocation_result.weights"

    portfolio = result_data.get("portfolio_weights")
    if isinstance(portfolio, dict) and portfolio:
        return dict(portfolio), "portfolio_weights"

    # Fallback: balanced macro/ETF RA reference basket.
    fallback = {
        "SPY": 0.20,
        "QQQ": 0.12,
        "IWM": 0.08,
        "DIA": 0.10,
        "TLT": 0.20,
        "GLD": 0.15,
        "USO": 0.08,
        "UUP": 0.07,
    }
    return fallback, "ra_macro_fallback_basket"


def _query_sqlite_price_panel(
    db_path: Path,
    table: str,
    date_col: str,
    ticker_col: str,
    close_col: str,
    tickers: List[str],
    lookback_days: int,
) -> pd.DataFrame:
    if not db_path.exists() or not tickers:
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(tickers))
    sql = (
        f"SELECT {date_col} AS dt, {ticker_col} AS ticker, {close_col} AS close "
        f"FROM {table} "
        f"WHERE {ticker_col} IN ({placeholders}) "
        f"ORDER BY {date_col} ASC"
    )
    try:
        with sqlite3.connect(str(db_path)) as conn:
            raw = pd.read_sql_query(sql, conn, params=tickers)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()
    raw["dt"] = pd.to_datetime(raw["dt"], errors="coerce")
    raw = raw.dropna(subset=["dt", "ticker", "close"])
    if raw.empty:
        return pd.DataFrame()
    panel = raw.pivot_table(index="dt", columns="ticker", values="close", aggfunc="last").sort_index()
    rows = max(260, min(int(lookback_days), 756))
    panel = panel.tail(rows)
    panel = panel.ffill().dropna(how="all")
    return panel


def _load_close_panel_from_local_market_sources(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """
    네트워크 실패 시 로컬 시세 DB에서 종가 패널을 로드한다.
    1순위: financial_indicators DB (market_data)
    2순위: eimas stable market DB (daily_prices)
    """
    fi_default = Path(__file__).resolve().parent.parent.parent / "financial_indicators" / "data" / "financial_indicators.db"
    fi_path = Path(os.getenv("EIMAS_BACKTEST_FI_DB_PATH", str(fi_default))).expanduser()
    if not fi_path.is_absolute():
        fi_path = (Path.cwd() / fi_path).resolve()
    panel = _query_sqlite_price_panel(
        db_path=fi_path,
        table="market_data",
        date_col="date",
        ticker_col="ticker",
        close_col="close",
        tickers=tickers,
        lookback_days=lookback_days,
    )
    if not panel.empty:
        panel.attrs["source_db"] = str(fi_path)
        return panel

    stable_default = Path(__file__).resolve().parent.parent / "data" / "stable" / "market.db"
    stable_path = Path(os.getenv("EIMAS_BACKTEST_STABLE_MARKET_DB_PATH", str(stable_default))).expanduser()
    if not stable_path.is_absolute():
        stable_path = (Path.cwd() / stable_path).resolve()
    panel = _query_sqlite_price_panel(
        db_path=stable_path,
        table="daily_prices",
        date_col="date",
        ticker_col="ticker",
        close_col="close",
        tickers=tickers,
        lookback_days=lookback_days,
    )
    if not panel.empty:
        panel.attrs["source_db"] = str(stable_path)
    return panel


def _seed_for_ticker(ticker: str) -> int:
    digest = hashlib.sha256(ticker.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _base_price_for_ticker(ticker: str) -> float:
    defaults = {
        "SPY": 520.0,
        "QQQ": 450.0,
        "IWM": 205.0,
        "DIA": 390.0,
        "TLT": 95.0,
        "GLD": 210.0,
        "USO": 78.0,
        "UUP": 29.0,
        "^VIX": 18.0,
        "BTC-USD": 65000.0,
        "ETH-USD": 3500.0,
        "SOL-USD": 140.0,
    }
    return defaults.get(ticker, 100.0)


def _build_synthetic_close_panel(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    points = max(260, min(int(lookback_days), 756))
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=points)
    series_map: Dict[str, pd.Series] = {}

    for ticker in tickers:
        rng = np.random.default_rng(_seed_for_ticker(ticker))
        drift = 0.00025
        vol = 0.012
        if ticker in {"TLT", "GLD"}:
            drift = 0.00010
            vol = 0.008
        elif ticker in {"BTC-USD", "ETH-USD", "SOL-USD"}:
            drift = 0.00055
            vol = 0.026
        elif ticker in {"^VIX"}:
            drift = 0.0
            vol = 0.03

        returns = drift + rng.normal(0, vol, size=points)
        close = _base_price_for_ticker(ticker) * np.exp(np.cumsum(returns))
        if ticker == "^VIX":
            close = np.clip(close, 10.0, 80.0)
        series_map[ticker] = pd.Series(close, index=dates, name=ticker)

    panel = pd.DataFrame(series_map).dropna()
    return panel


def run_allocation_backtest_from_result(
    result_data: Dict[str, Any],
    db: Optional[TradingDB] = None,
    lookback_days: int = 756,
    allow_synthetic_fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    result의 배분 비중으로 간단 백테스트를 수행하고 TradingDB에 저장.
    """
    db = db or TradingDB()
    weights, weight_source = _infer_backtest_weights_from_result(result_data)
    if not isinstance(weights, dict) or not weights:
        return {"ok": False, "error": "missing_weights"}

    filtered = {k: _safe_float(v) for k, v in weights.items() if _safe_float(v) > 0}
    if len(filtered) < 2:
        return {"ok": False, "error": "insufficient_weight_universe"}

    total = sum(filtered.values())
    filtered = {k: v / total for k, v in filtered.items()} if total > 0 else filtered

    tickers = sorted(set(filtered.keys()) | {"SPY"})
    prices = _download_close_panel(
        tickers,
        lookback_days=lookback_days,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    min_required_rows_raw = os.getenv("EIMAS_BACKTEST_MIN_REQUIRED_ROWS", "200").strip()
    try:
        min_required_rows = max(120, int(min_required_rows_raw))
    except Exception:
        min_required_rows = 200
    if prices.empty or len(prices) < min_required_rows:
        return {
            "ok": False,
            "error": "insufficient_price_history",
            "rows": len(prices),
            "required_rows": min_required_rows,
        }
    synthetic_fallback = bool(prices.attrs.get("synthetic_fallback", False))
    market_data_source = str(prices.attrs.get("market_data_source", "unknown"))

    available_assets = sorted([ticker for ticker in filtered if ticker in prices.columns])
    dropped_assets = sorted([ticker for ticker in filtered if ticker not in prices.columns])
    if len(available_assets) < 2:
        return {
            "ok": False,
            "error": "insufficient_weight_universe_after_price_filter",
            "available_assets": available_assets,
            "dropped_assets": dropped_assets,
        }
    if dropped_assets:
        filtered = {ticker: weight for ticker, weight in filtered.items() if ticker in available_assets}
        norm = sum(filtered.values())
        if norm <= 0:
            return {"ok": False, "error": "invalid_filtered_weights"}
        filtered = {ticker: weight / norm for ticker, weight in filtered.items()}

    if "SPY" not in prices.columns:
        return {"ok": False, "error": "missing_benchmark_price", "benchmark": "SPY"}

    # Start from first available date so monthly rebalance can actually trigger.
    warmup_bars = 0
    adaptive_min_history = max(90, min(180, len(prices) // 2))
    if len(prices) <= adaptive_min_history + 5:
        adaptive_min_history = max(60, min(120, max(30, len(prices) // 3)))

    start_date = str(prices.index[0].date())
    end_date = str(prices.index[-1].date())
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency="monthly",
        transaction_cost_bps=10,
        initial_capital=1_000_000,
        train_period_days=adaptive_min_history,
        min_history_days=adaptive_min_history,
    )

    def static_alloc(_: pd.DataFrame) -> Dict[str, float]:
        return filtered

    engine = BacktestEngine(config)
    bt_result = engine.run(prices, static_alloc, benchmark="SPY")
    metrics = bt_result.metrics

    overall_alpha = 0.0
    overall_benchmark_return = 0.0
    for pm in bt_result.period_metrics:
        if pm.get("period_type") == "OVERALL":
            overall_alpha = _safe_float(pm.get("alpha"), 0.0)
            overall_benchmark_return = _safe_float(pm.get("benchmark_return"), 0.0)
            break

    payload = {
        "strategy_name": "EIMAS_AutoPaper_TargetPrice",
        "start_date": metrics.start_date,
        "end_date": metrics.end_date,
        "initial_capital": config.initial_capital,
        "final_capital": config.initial_capital * (1 + metrics.total_return),
        "total_return": metrics.total_return,
        "annual_return": metrics.annualized_return,
        "benchmark_return": overall_benchmark_return,
        "alpha": overall_alpha,
        "volatility": metrics.annualized_volatility,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_duration": metrics.max_drawdown_duration,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "total_trades": metrics.num_trades,
        "winning_trades": int(metrics.win_rate * metrics.num_periods),
        "losing_trades": metrics.num_periods - int(metrics.win_rate * metrics.num_periods),
        "win_rate": metrics.win_rate,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "profit_factor": metrics.profit_factor,
        "avg_holding_days": 30,
        "total_commission": metrics.total_transaction_costs,
        "total_slippage": 0.0,
        "total_short_cost": 0.0,
        "parameters": {
            "source": "auto_paper_execution",
            "lookback_days": int(lookback_days),
            "universe_size": len(filtered),
            "benchmark": "SPY",
            "synthetic_price_fallback": synthetic_fallback,
            "market_data_source": market_data_source,
            "dropped_assets": dropped_assets,
            "price_rows": int(len(prices)),
            "warmup_bars": int(warmup_bars),
            "adaptive_min_history": int(adaptive_min_history),
            "weight_source": weight_source,
        },
        "trades": [],
    }
    run_id = db.save_backtest_run(payload)
    db.save_backtest_daily_nav(run_id, bt_result.daily_nav_records)
    db.save_backtest_snapshots(run_id, bt_result.snapshot_records)
    db.save_backtest_period_metrics(run_id, bt_result.period_metrics)

    sql_evidence = save_backtest_metrics_to_sql(
        metrics={
            **metrics.to_dict(),
            "alpha": overall_alpha,
            "benchmark_return": overall_benchmark_return,
        },
        source="eimas.auto_paper_execution.backtest",
        strategy_name="EIMAS_AutoPaper_TargetPrice",
        start_date=metrics.start_date,
        end_date=metrics.end_date,
        linked_run_id=run_id,
        notes={
            "lookback_days": int(lookback_days),
            "benchmark": "SPY",
            "universe_size": len(filtered),
            "market_data_source": market_data_source,
            "dropped_assets": dropped_assets,
            "price_rows": int(len(prices)),
            "warmup_bars": int(warmup_bars),
            "adaptive_min_history": int(adaptive_min_history),
            "weight_source": weight_source,
        },
    )

    return {
        "ok": True,
        "run_id": run_id,
        "metrics": metrics.to_dict(),
        "start_date": start_date,
        "end_date": end_date,
        "tickers": sorted(filtered.keys()),
        "ra_sql": sql_evidence,
        "synthetic_price_fallback": synthetic_fallback,
        "market_data_source": market_data_source,
        "dropped_assets": dropped_assets,
        "price_rows": int(len(prices)),
        "warmup_bars": int(warmup_bars),
        "adaptive_min_history": int(adaptive_min_history),
        "weight_source": weight_source,
    }


def poll_pending_paper_orders(
    broker: str = "ibkr",
    account_name: str = "ra_auto",
    initial_capital: float = 100_000.0,
) -> Dict[str, Any]:
    """대기 주문만 폴링/체결하고 TradingDB 상태를 동기화."""
    if (broker or "").strip().lower() != "ibkr":
        return {
            "account": account_name,
            "broker": broker,
            "poll_result": {},
            "pending_count": 0,
            "error": f"unsupported_broker:{broker}",
        }

    trader = PaperTrader(initial_capital=initial_capital, account_name=account_name)
    db = TradingDB()
    router = build_ibkr_paper_router(
        account_name=account_name,
        initial_capital=initial_capital,
        db=db,
        trader=trader,
        dry_run=False,
    )
    poll_result = router.poll_pending_orders()

    return {
        "account": account_name,
        "broker": broker,
        "poll_result": poll_result,
        "pending_count": router.pending_order_count(),
    }
