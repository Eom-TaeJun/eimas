#!/usr/bin/env python3
"""
EIMAS RA SQL Store
==================
Internal SQL evidence layer for RA workflow.

Purpose:
- Persist company fundamentals/valuation snapshots with SQL upsert.
- Persist backtest summary rows with SQL insert.
- Provide reusable validation/summary queries for report evidence.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _resolve_db_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser()
    else:
        configured = os.getenv("EIMAS_RA_SQLITE_PATH", "data/ra_research.db").strip()
        resolved = Path(configured).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


class RAResearchSQLStore:
    """SQLite-backed internal SQL storage for RA evidence."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = _resolve_db_path(db_path)
        self._initialize_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ra_company_fundamentals (
                    as_of_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    currency TEXT,
                    last_close REAL,
                    market_cap REAL,
                    trailing_pe REAL,
                    forward_pe REAL,
                    price_to_book REAL,
                    ev_to_ebitda REAL,
                    revenue REAL,
                    operating_income REAL,
                    net_income REAL,
                    operating_cash_flow REAL,
                    free_cash_flow REAL,
                    total_assets REAL,
                    total_liabilities REAL,
                    total_equity REAL,
                    roe REAL,
                    roa REAL,
                    net_margin REAL,
                    debt_to_equity REAL,
                    current_ratio REAL,
                    ret_5d_pct REAL,
                    ret_20d_pct REAL,
                    momentum_label TEXT,
                    valuation_signal TEXT,
                    peer_pe_median REAL,
                    ra_takeaway TEXT,
                    source TEXT,
                    revision_tag TEXT,
                    quality_flag TEXT DEFAULT 'OK',
                    ingested_at TEXT NOT NULL,
                    PRIMARY KEY (as_of_date, ticker)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ra_backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    linked_run_id INTEGER,
                    strategy_name TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    total_return REAL,
                    annualized_return REAL,
                    annualized_volatility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    alpha REAL,
                    benchmark_return REAL,
                    notes_json TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ra_etf_snapshot (
                    as_of_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    etf_name TEXT,
                    category TEXT,
                    asset_role TEXT,
                    factor_exposure TEXT,
                    sector_or_theme TEXT,
                    duration_profile TEXT,
                    last_close REAL,
                    ret_1d_pct REAL,
                    ret_5d_pct REAL,
                    ret_20d_pct REAL,
                    ret_60d_pct REAL,
                    relative_strength_20d REAL,
                    volume_ratio_20d REAL,
                    rsi_14 REAL,
                    expense_ratio_pct REAL,
                    dividend_yield_pct REAL,
                    total_assets_usd_bn REAL,
                    holdings_count INTEGER,
                    top_holdings_json TEXT,
                    sector_weights_json TEXT,
                    momentum_label TEXT,
                    data_source TEXT,
                    quality_flag TEXT DEFAULT 'OK',
                    source TEXT,
                    revision_tag TEXT,
                    ingested_at TEXT NOT NULL,
                    PRIMARY KEY (as_of_date, ticker)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ra_commentary_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    commentary_source TEXT NOT NULL,
                    model_name TEXT,
                    is_fallback INTEGER NOT NULL DEFAULT 0,
                    prompt_text TEXT,
                    snapshot_json TEXT,
                    response_text TEXT,
                    response_json TEXT,
                    error_tag TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_fund_ticker_date ON ra_company_fundamentals(ticker, as_of_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_fund_date ON ra_company_fundamentals(as_of_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_backtest_created ON ra_backtest_runs(created_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_etf_ticker_date ON ra_etf_snapshot(ticker, as_of_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_etf_date ON ra_etf_snapshot(as_of_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_commentary_created ON ra_commentary_audit_log(created_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ra_commentary_source ON ra_commentary_audit_log(commentary_source)"
            )
            cur.execute(
                """
                CREATE VIEW IF NOT EXISTS vw_ra_valuation_snapshot AS
                SELECT
                    f.as_of_date,
                    f.ticker,
                    f.sector,
                    f.trailing_pe,
                    f.price_to_book,
                    f.roe,
                    f.net_margin,
                    f.ret_20d_pct,
                    f.valuation_signal,
                    CASE
                        WHEN f.trailing_pe IS NULL OR f.trailing_pe <= 0 THEN 'N/A'
                        WHEN f.trailing_pe < 15 THEN 'LOW_PE'
                        WHEN f.trailing_pe <= 30 THEN 'MID_PE'
                        ELSE 'HIGH_PE'
                    END AS pe_bucket
                FROM ra_company_fundamentals f
                JOIN (
                    SELECT MAX(as_of_date) AS max_date FROM ra_company_fundamentals
                ) d ON f.as_of_date = d.max_date
                """
            )
            cur.execute(
                """
                CREATE VIEW IF NOT EXISTS vw_ra_etf_momentum_snapshot AS
                SELECT
                    e.as_of_date,
                    e.ticker,
                    e.etf_name,
                    e.category,
                    e.sector_or_theme,
                    e.ret_5d_pct,
                    e.ret_20d_pct,
                    e.ret_60d_pct,
                    e.relative_strength_20d,
                    e.momentum_label,
                    RANK() OVER (
                        ORDER BY CASE WHEN e.ret_20d_pct IS NULL THEN -999999.0 ELSE e.ret_20d_pct END DESC
                    ) AS rank_ret_20d
                FROM ra_etf_snapshot e
                JOIN (
                    SELECT MAX(as_of_date) AS max_date FROM ra_etf_snapshot
                ) d ON e.as_of_date = d.max_date
                """
            )
            cur.execute(
                """
                CREATE VIEW IF NOT EXISTS vw_ra_backtest_compare AS
                SELECT
                    id,
                    created_at,
                    source,
                    strategy_name,
                    start_date,
                    end_date,
                    total_return,
                    annualized_return,
                    sharpe_ratio,
                    max_drawdown,
                    win_rate,
                    RANK() OVER (
                        ORDER BY CASE WHEN sharpe_ratio IS NULL THEN -999999.0 ELSE sharpe_ratio END DESC
                    ) AS sharpe_rank,
                    AVG(total_return) OVER (PARTITION BY strategy_name) AS strategy_avg_total_return
                FROM ra_backtest_runs
                """
            )
            cur.execute(
                """
                CREATE VIEW IF NOT EXISTS vw_ra_allocation_signal AS
                WITH latest_company AS (
                    SELECT *
                    FROM ra_company_fundamentals
                    WHERE as_of_date = (SELECT MAX(as_of_date) FROM ra_company_fundamentals)
                ),
                latest_etf AS (
                    SELECT *
                    FROM ra_etf_snapshot
                    WHERE as_of_date = (SELECT MAX(as_of_date) FROM ra_etf_snapshot)
                ),
                company_agg AS (
                    SELECT
                        COUNT(*) AS n_companies,
                        SUM(CASE WHEN valuation_signal = 'UNDERVALUED' THEN 1 ELSE 0 END) AS undervalued_cnt,
                        SUM(CASE WHEN valuation_signal = 'OVERVALUED' THEN 1 ELSE 0 END) AS overvalued_cnt,
                        AVG(CASE WHEN ret_20d_pct IS NULL THEN NULL ELSE ret_20d_pct END) AS avg_company_ret20d
                    FROM latest_company
                ),
                etf_agg AS (
                    SELECT
                        COUNT(*) AS n_etf,
                        SUM(CASE WHEN ret_20d_pct > 0 THEN 1 ELSE 0 END) AS positive_ret20d_cnt,
                        AVG(CASE WHEN ret_20d_pct IS NULL THEN NULL ELSE ret_20d_pct END) AS avg_etf_ret20d,
                        MAX(CASE WHEN ticker = 'TLT' THEN ret_20d_pct END) AS tlt_ret20d,
                        MAX(CASE WHEN ticker = 'UUP' THEN ret_20d_pct END) AS uup_ret20d,
                        MAX(CASE WHEN ticker = 'GLD' THEN ret_20d_pct END) AS gld_ret20d
                    FROM latest_etf
                ),
                score_base AS (
                    SELECT
                        date('now') AS as_of_date,
                        c.n_companies,
                        c.undervalued_cnt,
                        c.overvalued_cnt,
                        e.n_etf,
                        e.positive_ret20d_cnt,
                        c.avg_company_ret20d,
                        e.avg_etf_ret20d,
                        e.tlt_ret20d,
                        e.uup_ret20d,
                        e.gld_ret20d,
                        CASE
                            WHEN c.n_companies > 0 THEN
                                ((COALESCE(c.undervalued_cnt, 0) - COALESCE(c.overvalued_cnt, 0)) * 10.0)
                                / CAST(c.n_companies AS REAL)
                            ELSE 0.0
                        END AS valuation_score_raw,
                        CASE
                            WHEN e.n_etf > 0 THEN
                                ((COALESCE(e.positive_ret20d_cnt, 0) * 100.0) / CAST(e.n_etf AS REAL)) - 50.0
                            ELSE 0.0
                        END AS etf_breadth_score_raw,
                        (
                            COALESCE(e.tlt_ret20d, 0.0) * 0.35
                            + COALESCE(e.uup_ret20d, 0.0) * 0.35
                            + COALESCE(e.gld_ret20d, 0.0) * 0.30
                        ) AS macro_proxy_score_raw
                    FROM company_agg c
                    CROSS JOIN etf_agg e
                )
                SELECT
                    as_of_date,
                    n_companies,
                    undervalued_cnt,
                    overvalued_cnt,
                    n_etf,
                    positive_ret20d_cnt,
                    ROUND(COALESCE(avg_company_ret20d, 0.0), 3) AS avg_company_ret20d,
                    ROUND(COALESCE(avg_etf_ret20d, 0.0), 3) AS avg_etf_ret20d,
                    ROUND(COALESCE(tlt_ret20d, 0.0), 3) AS tlt_ret20d,
                    ROUND(COALESCE(uup_ret20d, 0.0), 3) AS uup_ret20d,
                    ROUND(COALESCE(gld_ret20d, 0.0), 3) AS gld_ret20d,
                    ROUND(valuation_score_raw, 3) AS valuation_score,
                    ROUND(etf_breadth_score_raw, 3) AS etf_breadth_score,
                    ROUND(macro_proxy_score_raw, 3) AS macro_proxy_score,
                    ROUND(
                        (valuation_score_raw * 0.45)
                        + (etf_breadth_score_raw * 0.35)
                        + (macro_proxy_score_raw * 0.20),
                        3
                    ) AS composite_score,
                    CASE
                        WHEN (
                            (valuation_score_raw * 0.45)
                            + (etf_breadth_score_raw * 0.35)
                            + (macro_proxy_score_raw * 0.20)
                        ) >= 10 THEN 'RISK_ON'
                        WHEN (
                            (valuation_score_raw * 0.45)
                            + (etf_breadth_score_raw * 0.35)
                            + (macro_proxy_score_raw * 0.20)
                        ) <= -10 THEN 'RISK_OFF'
                        ELSE 'NEUTRAL'
                    END AS signal_label
                FROM score_base
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_company_rows(
        self,
        companies: List[Dict[str, Any]],
        as_of_date: Optional[str] = None,
        source: str = "eimas_company_ra_analysis",
        revision_tag: str = "",
    ) -> int:
        if not companies:
            return 0

        target_date = (as_of_date or _today_iso()).strip() or _today_iso()
        now_iso = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            rows = 0
            for item in companies:
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker", "")).upper().strip()
                if not ticker:
                    continue

                valuation = item.get("valuation", {})
                accounting = item.get("accounting", {})
                ratios = item.get("ratios", {})
                momentum = item.get("price_momentum", {})

                missing_core = 0
                core_values = (
                    _safe_float((valuation or {}).get("trailing_pe")),
                    _safe_float((valuation or {}).get("price_to_book")),
                    _safe_float((accounting or {}).get("revenue")),
                    _safe_float((accounting or {}).get("net_income")),
                )
                for value in core_values:
                    if value is None:
                        missing_core += 1
                quality_flag = "OK" if missing_core == 0 else ("PARTIAL" if missing_core <= 2 else "DEGRADED")

                cur.execute(
                    """
                    INSERT INTO ra_company_fundamentals (
                        as_of_date, ticker, company_name, sector, industry, currency,
                        last_close, market_cap, trailing_pe, forward_pe, price_to_book, ev_to_ebitda,
                        revenue, operating_income, net_income, operating_cash_flow, free_cash_flow,
                        total_assets, total_liabilities, total_equity,
                        roe, roa, net_margin, debt_to_equity, current_ratio,
                        ret_5d_pct, ret_20d_pct, momentum_label,
                        valuation_signal, peer_pe_median, ra_takeaway,
                        source, revision_tag, quality_flag, ingested_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    ON CONFLICT(as_of_date, ticker) DO UPDATE SET
                        company_name=excluded.company_name,
                        sector=excluded.sector,
                        industry=excluded.industry,
                        currency=excluded.currency,
                        last_close=excluded.last_close,
                        market_cap=excluded.market_cap,
                        trailing_pe=excluded.trailing_pe,
                        forward_pe=excluded.forward_pe,
                        price_to_book=excluded.price_to_book,
                        ev_to_ebitda=excluded.ev_to_ebitda,
                        revenue=excluded.revenue,
                        operating_income=excluded.operating_income,
                        net_income=excluded.net_income,
                        operating_cash_flow=excluded.operating_cash_flow,
                        free_cash_flow=excluded.free_cash_flow,
                        total_assets=excluded.total_assets,
                        total_liabilities=excluded.total_liabilities,
                        total_equity=excluded.total_equity,
                        roe=excluded.roe,
                        roa=excluded.roa,
                        net_margin=excluded.net_margin,
                        debt_to_equity=excluded.debt_to_equity,
                        current_ratio=excluded.current_ratio,
                        ret_5d_pct=excluded.ret_5d_pct,
                        ret_20d_pct=excluded.ret_20d_pct,
                        momentum_label=excluded.momentum_label,
                        valuation_signal=excluded.valuation_signal,
                        peer_pe_median=excluded.peer_pe_median,
                        ra_takeaway=excluded.ra_takeaway,
                        source=excluded.source,
                        revision_tag=excluded.revision_tag,
                        quality_flag=excluded.quality_flag,
                        ingested_at=excluded.ingested_at
                    """,
                    (
                        target_date,
                        ticker,
                        str(item.get("company_name", "")),
                        str(item.get("sector", "")),
                        str(item.get("industry", "")),
                        str(item.get("currency", "")),
                        _safe_float(item.get("last_close")),
                        _safe_float(item.get("market_cap")),
                        _safe_float((valuation or {}).get("trailing_pe")),
                        _safe_float((valuation or {}).get("forward_pe")),
                        _safe_float((valuation or {}).get("price_to_book")),
                        _safe_float((valuation or {}).get("ev_to_ebitda")),
                        _safe_float((accounting or {}).get("revenue")),
                        _safe_float((accounting or {}).get("operating_income")),
                        _safe_float((accounting or {}).get("net_income")),
                        _safe_float((accounting or {}).get("operating_cash_flow")),
                        _safe_float((accounting or {}).get("free_cash_flow")),
                        _safe_float((accounting or {}).get("total_assets")),
                        _safe_float((accounting or {}).get("total_liabilities")),
                        _safe_float((accounting or {}).get("total_equity")),
                        _safe_float((ratios or {}).get("roe")),
                        _safe_float((ratios or {}).get("roa")),
                        _safe_float((ratios or {}).get("net_margin")),
                        _safe_float((ratios or {}).get("debt_to_equity")),
                        _safe_float((ratios or {}).get("current_ratio")),
                        _safe_float((momentum or {}).get("ret_5d_pct")),
                        _safe_float((momentum or {}).get("ret_20d_pct")),
                        str((momentum or {}).get("momentum_label", "")),
                        str(item.get("valuation_signal", "")),
                        _safe_float(item.get("peer_pe_median")),
                        str(item.get("ra_takeaway", "")),
                        source,
                        revision_tag,
                        quality_flag,
                        now_iso,
                    ),
                )
                rows += 1
            conn.commit()
            return rows
        finally:
            conn.close()

    def upsert_etf_rows(
        self,
        etf_rows: List[Dict[str, Any]],
        as_of_date: Optional[str] = None,
        source: str = "eimas_company_ra_analysis",
        revision_tag: str = "",
    ) -> int:
        if not etf_rows:
            return 0

        target_date = (as_of_date or _today_iso()).strip() or _today_iso()
        now_iso = datetime.now().isoformat()

        conn = self._get_conn()
        try:
            cur = conn.cursor()
            rows = 0
            for item in etf_rows:
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker", "")).upper().strip()
                if not ticker:
                    continue

                top_holdings = item.get("top_holdings", [])
                sector_weights = item.get("sector_weights", [])
                quality_flag = str(item.get("quality_flag", "")).strip().upper() or "OK"

                cur.execute(
                    """
                    INSERT INTO ra_etf_snapshot (
                        as_of_date, ticker, etf_name, category, asset_role, factor_exposure,
                        sector_or_theme, duration_profile,
                        last_close, ret_1d_pct, ret_5d_pct, ret_20d_pct, ret_60d_pct,
                        relative_strength_20d, volume_ratio_20d, rsi_14,
                        expense_ratio_pct, dividend_yield_pct, total_assets_usd_bn, holdings_count,
                        top_holdings_json, sector_weights_json, momentum_label, data_source,
                        quality_flag, source, revision_tag, ingested_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    ON CONFLICT(as_of_date, ticker) DO UPDATE SET
                        etf_name=excluded.etf_name,
                        category=excluded.category,
                        asset_role=excluded.asset_role,
                        factor_exposure=excluded.factor_exposure,
                        sector_or_theme=excluded.sector_or_theme,
                        duration_profile=excluded.duration_profile,
                        last_close=excluded.last_close,
                        ret_1d_pct=excluded.ret_1d_pct,
                        ret_5d_pct=excluded.ret_5d_pct,
                        ret_20d_pct=excluded.ret_20d_pct,
                        ret_60d_pct=excluded.ret_60d_pct,
                        relative_strength_20d=excluded.relative_strength_20d,
                        volume_ratio_20d=excluded.volume_ratio_20d,
                        rsi_14=excluded.rsi_14,
                        expense_ratio_pct=excluded.expense_ratio_pct,
                        dividend_yield_pct=excluded.dividend_yield_pct,
                        total_assets_usd_bn=excluded.total_assets_usd_bn,
                        holdings_count=excluded.holdings_count,
                        top_holdings_json=excluded.top_holdings_json,
                        sector_weights_json=excluded.sector_weights_json,
                        momentum_label=excluded.momentum_label,
                        data_source=excluded.data_source,
                        quality_flag=excluded.quality_flag,
                        source=excluded.source,
                        revision_tag=excluded.revision_tag,
                        ingested_at=excluded.ingested_at
                    """,
                    (
                        target_date,
                        ticker,
                        str(item.get("etf_name", "")),
                        str(item.get("category", "")),
                        str(item.get("asset_role", "")),
                        str(item.get("factor_exposure", "")),
                        str(item.get("sector_or_theme", "")),
                        str(item.get("duration_profile", "")),
                        _safe_float(item.get("last_close")),
                        _safe_float(item.get("ret_1d_pct")),
                        _safe_float(item.get("ret_5d_pct")),
                        _safe_float(item.get("ret_20d_pct")),
                        _safe_float(item.get("ret_60d_pct")),
                        _safe_float(item.get("relative_strength_20d")),
                        _safe_float(item.get("volume_ratio_20d")),
                        _safe_float(item.get("rsi_14")),
                        _safe_float(item.get("expense_ratio_pct")),
                        _safe_float(item.get("dividend_yield_pct")),
                        _safe_float(item.get("total_assets_usd_bn")),
                        int(item.get("holdings_count", 0) or 0),
                        json.dumps(top_holdings if isinstance(top_holdings, list) else [], ensure_ascii=False),
                        json.dumps(sector_weights if isinstance(sector_weights, list) else [], ensure_ascii=False),
                        str(item.get("momentum_label", "")),
                        str(item.get("data_source", "")),
                        quality_flag,
                        source,
                        revision_tag,
                        now_iso,
                    ),
                )
                rows += 1
            conn.commit()
            return rows
        finally:
            conn.close()

    def summarize_etf_table(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT ticker) AS distinct_tickers,
                    MIN(as_of_date) AS min_date,
                    MAX(as_of_date) AS max_date
                FROM ra_etf_snapshot
                """
            )
            row = cur.fetchone()
            summary = {
                "total_rows": int((row["total_rows"] if row else 0) or 0),
                "distinct_tickers": int((row["distinct_tickers"] if row else 0) or 0),
                "min_date": (row["min_date"] if row else "") or "",
                "max_date": (row["max_date"] if row else "") or "",
            }
            cur.execute(
                """
                SELECT data_source, COUNT(*) AS cnt
                FROM ra_etf_snapshot
                GROUP BY data_source
                ORDER BY cnt DESC
                LIMIT 5
                """
            )
            source_rows = cur.fetchall() or []
            summary["source_mix"] = [
                {"data_source": str(r["data_source"] or ""), "count": int(r["cnt"] or 0)}
                for r in source_rows
            ]
            return summary
        finally:
            conn.close()

    def run_etf_quality_checks(self, as_of_date: Optional[str] = None) -> Dict[str, int]:
        target_date = (as_of_date or "").strip()
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            predicate = "as_of_date = ?" if target_date else "1=1"
            params = (target_date,) if target_date else ()

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_etf_snapshot
                WHERE {predicate}
                  AND (ret_5d_pct IS NULL OR ret_20d_pct IS NULL)
                """,
                params,
            )
            missing_returns = int((cur.fetchone() or {"cnt": 0})["cnt"])

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_etf_snapshot
                WHERE {predicate}
                  AND (top_holdings_json IS NULL OR top_holdings_json = '[]')
                """,
                params,
            )
            missing_holdings = int((cur.fetchone() or {"cnt": 0})["cnt"])

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_etf_snapshot
                WHERE {predicate}
                  AND quality_flag NOT IN ('OK', 'PARTIAL', 'COMPLETE', 'FALLBACK', 'SYNTHETIC')
                """,
                params,
            )
            unexpected_quality = int((cur.fetchone() or {"cnt": 0})["cnt"])

            return {
                "missing_return_rows": missing_returns,
                "missing_holdings_rows": missing_holdings,
                "unexpected_quality_rows": unexpected_quality,
            }
        finally:
            conn.close()

    def refresh_sql_artifacts(self) -> Dict[str, Any]:
        """
        Refresh materialized snapshots used by RA report.
        SQLite has no native materialized view, so we maintain table snapshots.
        """
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            materialized_specs = [
                ("ra_valuation_snapshot_mv", "SELECT * FROM vw_ra_valuation_snapshot"),
                ("ra_etf_momentum_snapshot_mv", "SELECT * FROM vw_ra_etf_momentum_snapshot"),
                ("ra_backtest_compare_mv", "SELECT * FROM vw_ra_backtest_compare"),
                ("ra_allocation_signal_mv", "SELECT * FROM vw_ra_allocation_signal"),
            ]
            row_counts: Dict[str, int] = {}
            for table_name, sql in materialized_specs:
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                cur.execute(f"CREATE TABLE {table_name} AS {sql}")
                cur.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
                row = cur.fetchone()
                row_counts[table_name] = int((row["cnt"] if row else 0) or 0)
            conn.commit()
            return {
                "materialized_tables": list(row_counts.keys()),
                "row_counts": row_counts,
                "refreshed_at": datetime.now().isoformat(),
            }
        finally:
            conn.close()

    def _query_preview(self, sql: str, params: tuple = (), limit: int = 5) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall() or []
                output: List[Dict[str, Any]] = []
                for idx, row in enumerate(rows):
                    if idx >= limit:
                        break
                    output.append({k: row[k] for k in row.keys()})
                return output
            except sqlite3.Error:
                return []
        finally:
            conn.close()

    def preview_sql_artifacts(self, limit: int = 5) -> Dict[str, Any]:
        return {
            "valuation_snapshot_mv": self._query_preview(
                """
                SELECT ticker, sector, trailing_pe, price_to_book, ret_20d_pct, valuation_signal, pe_bucket
                FROM ra_valuation_snapshot_mv
                ORDER BY CASE WHEN trailing_pe IS NULL THEN 999999.0 ELSE trailing_pe END ASC, ticker
                """,
                limit=limit,
            ),
            "etf_momentum_snapshot_mv": self._query_preview(
                """
                SELECT ticker, category, ret_5d_pct, ret_20d_pct, rank_ret_20d, momentum_label
                FROM ra_etf_momentum_snapshot_mv
                ORDER BY rank_ret_20d ASC, ticker
                """,
                limit=limit,
            ),
            "backtest_compare_mv": self._query_preview(
                """
                SELECT strategy_name, total_return, annualized_return, sharpe_ratio, max_drawdown, sharpe_rank
                FROM ra_backtest_compare_mv
                ORDER BY sharpe_rank ASC, id DESC
                """,
                limit=limit,
            ),
            "allocation_signal_mv": self._query_preview(
                """
                SELECT
                    as_of_date,
                    valuation_score,
                    etf_breadth_score,
                    macro_proxy_score,
                    composite_score,
                    signal_label,
                    n_companies,
                    n_etf
                FROM ra_allocation_signal_mv
                ORDER BY as_of_date DESC
                """,
                limit=limit,
            ),
        }

    def summarize_company_table(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT ticker) AS distinct_tickers,
                    MIN(as_of_date) AS min_date,
                    MAX(as_of_date) AS max_date
                FROM ra_company_fundamentals
                """
            )
            row = cur.fetchone()
            if row is None:
                return {
                    "total_rows": 0,
                    "distinct_tickers": 0,
                    "min_date": "",
                    "max_date": "",
                }
            return {
                "total_rows": int(row["total_rows"] or 0),
                "distinct_tickers": int(row["distinct_tickers"] or 0),
                "min_date": row["min_date"] or "",
                "max_date": row["max_date"] or "",
            }
        finally:
            conn.close()

    def run_company_quality_checks(self, as_of_date: Optional[str] = None) -> Dict[str, int]:
        target_date = (as_of_date or "").strip()
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            predicate = "as_of_date = ?" if target_date else "1=1"
            params = (target_date,) if target_date else ()

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_company_fundamentals
                WHERE {predicate}
                  AND (trailing_pe IS NULL OR price_to_book IS NULL)
                """,
                params,
            )
            missing_valuation = int((cur.fetchone() or {"cnt": 0})["cnt"])

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_company_fundamentals
                WHERE {predicate}
                  AND (revenue IS NULL OR net_income IS NULL)
                """,
                params,
            )
            missing_financial = int((cur.fetchone() or {"cnt": 0})["cnt"])

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM ra_company_fundamentals
                WHERE {predicate}
                  AND quality_flag != 'OK'
                """,
                params,
            )
            flagged_rows = int((cur.fetchone() or {"cnt": 0})["cnt"])

            return {
                "missing_valuation_rows": missing_valuation,
                "missing_financial_rows": missing_financial,
                "quality_flagged_rows": flagged_rows,
            }
        finally:
            conn.close()

    def insert_backtest_run(
        self,
        source: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        metrics: Dict[str, Any],
        linked_run_id: Optional[int] = None,
        notes: Optional[Dict[str, Any]] = None,
    ) -> int:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ra_backtest_runs (
                    created_at, source, linked_run_id, strategy_name,
                    start_date, end_date,
                    total_return, annualized_return, annualized_volatility,
                    sharpe_ratio, max_drawdown, win_rate,
                    alpha, benchmark_return, notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    source,
                    linked_run_id,
                    strategy_name,
                    start_date,
                    end_date,
                    _safe_float(metrics.get("total_return")),
                    _safe_float(metrics.get("annualized_return")),
                    _safe_float(metrics.get("annualized_volatility")),
                    _safe_float(metrics.get("sharpe_ratio")),
                    _safe_float(metrics.get("max_drawdown")),
                    _safe_float(metrics.get("win_rate")),
                    _safe_float(metrics.get("alpha")),
                    _safe_float(metrics.get("benchmark_return")),
                    json.dumps(notes or {}, ensure_ascii=False),
                ),
            )
            run_id = int(cur.lastrowid)
            conn.commit()
            return run_id
        finally:
            conn.close()

    def summarize_backtest_table(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_runs,
                    AVG(sharpe_ratio) AS avg_sharpe,
                    MAX(created_at) AS last_created_at
                FROM ra_backtest_runs
                """
            )
            row = cur.fetchone()
            return {
                "total_runs": int((row["total_runs"] if row else 0) or 0),
                "avg_sharpe": float((row["avg_sharpe"] if row else 0.0) or 0.0),
                "last_created_at": (row["last_created_at"] if row else "") or "",
            }
        finally:
            conn.close()

    def insert_commentary_audit_log(
        self,
        commentary_source: str,
        model_name: str,
        snapshot: Dict[str, Any],
        response_payload: Dict[str, Any],
        prompt_text: str = "",
        response_text: str = "",
        error_tag: str = "",
    ) -> int:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ra_commentary_audit_log (
                    created_at, commentary_source, model_name, is_fallback,
                    prompt_text, snapshot_json, response_text, response_json, error_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    str(commentary_source or "rule_based"),
                    str(model_name or ""),
                    0 if str(commentary_source or "").lower() == "openai" else 1,
                    str(prompt_text or ""),
                    json.dumps(snapshot or {}, ensure_ascii=False),
                    str(response_text or ""),
                    json.dumps(response_payload or {}, ensure_ascii=False),
                    str(error_tag or ""),
                ),
            )
            log_id = int(cur.lastrowid)
            conn.commit()
            return log_id
        finally:
            conn.close()

    def summarize_commentary_audit_table(self) -> Dict[str, Any]:
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_logs,
                    MAX(created_at) AS last_created_at
                FROM ra_commentary_audit_log
                """
            )
            row = cur.fetchone()
            cur.execute(
                """
                SELECT commentary_source, COUNT(*) AS cnt
                FROM ra_commentary_audit_log
                GROUP BY commentary_source
                ORDER BY cnt DESC
                LIMIT 3
                """
            )
            source_rows = cur.fetchall() or []
            return {
                "total_logs": int((row["total_logs"] if row else 0) or 0),
                "last_created_at": (row["last_created_at"] if row else "") or "",
                "source_mix": [
                    {"commentary_source": str(r["commentary_source"] or ""), "count": int(r["cnt"] or 0)}
                    for r in source_rows
                ],
            }
        finally:
            conn.close()


def ingest_company_ra_analysis_to_sql(
    company_ra_analysis: Dict[str, Any],
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upsert company RA payload into internal SQL store and return evidence summary.
    """
    evidence: Dict[str, Any] = {
        "enabled": False,
        "backend": "sqlite",
        "db_path": "",
        "table": "ra_company_fundamentals",
        "upserted_rows": 0,
        "total_rows": 0,
        "distinct_tickers": 0,
        "min_date": "",
        "max_date": "",
        "quality_checks": {},
        "error": "",
    }

    if not _env_flag("EIMAS_RA_SQL_ENABLED", default=True):
        evidence["error"] = "disabled_by_env"
        return evidence

    if not isinstance(company_ra_analysis, dict):
        evidence["error"] = "invalid_company_ra_analysis"
        return evidence

    companies = company_ra_analysis.get("companies", [])
    if not isinstance(companies, list):
        evidence["error"] = "invalid_companies_payload"
        return evidence
    etf_rows = company_ra_analysis.get("etf_strategy_snapshot", [])
    if not isinstance(etf_rows, list):
        etf_rows = []

    try:
        revision_tag = os.getenv("EIMAS_RA_SQL_REVISION_TAG", "").strip()
        store = RAResearchSQLStore()
        company_upserted = store.upsert_company_rows(
            companies=companies,
            as_of_date=as_of_date,
            source="eimas.phase1.company_ra_analysis",
            revision_tag=revision_tag,
        )
        etf_upserted = store.upsert_etf_rows(
            etf_rows=etf_rows,
            as_of_date=as_of_date,
            source="eimas.phase1.etf_snapshot",
            revision_tag=revision_tag,
        )
        company_summary = store.summarize_company_table()
        etf_summary = store.summarize_etf_table()
        quality = store.run_company_quality_checks(as_of_date=as_of_date)
        etf_quality = store.run_etf_quality_checks(as_of_date=as_of_date)
        artifacts = store.refresh_sql_artifacts()
        preview = store.preview_sql_artifacts(limit=8)
        evidence.update(
            {
                "enabled": True,
                "db_path": str(store.db_path),
                "upserted_rows": company_upserted,
                "total_rows": int(company_summary.get("total_rows", 0)),
                "distinct_tickers": int(company_summary.get("distinct_tickers", 0)),
                "min_date": str(company_summary.get("min_date", "")),
                "max_date": str(company_summary.get("max_date", "")),
                "etf_table": "ra_etf_snapshot",
                "etf_upserted_rows": int(etf_upserted),
                "etf_total_rows": int(etf_summary.get("total_rows", 0)),
                "etf_distinct_tickers": int(etf_summary.get("distinct_tickers", 0)),
                "etf_source_mix": etf_summary.get("source_mix", []),
                "quality_checks": quality,
                "etf_quality_checks": etf_quality,
                "sql_artifacts": artifacts,
                "sql_preview_tables": preview,
                "validation_sql_examples": [
                    "SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS distinct_tickers, MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date FROM ra_company_fundamentals;",
                    "SELECT ticker, trailing_pe, price_to_book, revenue, net_income, quality_flag FROM ra_company_fundamentals ORDER BY as_of_date DESC, ticker LIMIT 10;",
                    "SELECT ticker, category, ret_20d_pct, momentum_label FROM ra_etf_snapshot ORDER BY as_of_date DESC, ret_20d_pct DESC LIMIT 10;",
                    "SELECT ticker, pe_bucket, trailing_pe, ret_20d_pct FROM ra_valuation_snapshot_mv ORDER BY trailing_pe LIMIT 10;",
                    "SELECT as_of_date, valuation_score, etf_breadth_score, macro_proxy_score, composite_score, signal_label FROM ra_allocation_signal_mv ORDER BY as_of_date DESC LIMIT 10;",
                ],
            }
        )
    except Exception as e:
        evidence["error"] = str(e)

    return evidence


def save_backtest_metrics_to_sql(
    metrics: Dict[str, Any],
    source: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    linked_run_id: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist backtest summary metrics to internal SQL store.
    """
    evidence: Dict[str, Any] = {
        "enabled": False,
        "backend": "sqlite",
        "db_path": "",
        "table": "ra_backtest_runs",
        "saved_run_id": None,
        "total_runs": 0,
        "avg_sharpe": 0.0,
        "error": "",
    }

    if not _env_flag("EIMAS_RA_SQL_ENABLED", default=True):
        evidence["error"] = "disabled_by_env"
        return evidence

    if not isinstance(metrics, dict):
        evidence["error"] = "invalid_metrics"
        return evidence

    try:
        store = RAResearchSQLStore()
        saved_id = store.insert_backtest_run(
            source=source,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            linked_run_id=linked_run_id,
            notes=notes,
        )
        summary = store.summarize_backtest_table()
        artifacts = store.refresh_sql_artifacts()
        preview = store.preview_sql_artifacts(limit=8)
        evidence.update(
            {
                "enabled": True,
                "db_path": str(store.db_path),
                "saved_run_id": saved_id,
                "total_runs": int(summary.get("total_runs", 0)),
                "avg_sharpe": float(summary.get("avg_sharpe", 0.0)),
                "last_created_at": str(summary.get("last_created_at", "")),
                "sql_artifacts": artifacts,
                "sql_preview_tables": preview,
                "validation_sql_examples": [
                    "SELECT COUNT(*) AS total_runs, AVG(sharpe_ratio) AS avg_sharpe FROM ra_backtest_runs;",
                    "SELECT source, strategy_name, start_date, end_date, total_return, sharpe_ratio FROM ra_backtest_runs ORDER BY id DESC LIMIT 10;",
                    "SELECT strategy_name, sharpe_ratio, sharpe_rank FROM ra_backtest_compare_mv ORDER BY sharpe_rank ASC LIMIT 10;",
                ],
            }
        )
    except Exception as e:
        evidence["error"] = str(e)

    return evidence


def save_ra_commentary_audit_log(
    snapshot: Dict[str, Any],
    commentary_payload: Dict[str, Any],
    prompt_text: str = "",
    response_text: str = "",
    error_tag: str = "",
) -> Dict[str, Any]:
    """
    Persist RA commentary generation audit trail (snapshot/prompt/response).
    """
    evidence: Dict[str, Any] = {
        "enabled": False,
        "backend": "sqlite",
        "db_path": "",
        "table": "ra_commentary_audit_log",
        "saved_id": None,
        "total_logs": 0,
        "last_created_at": "",
        "source_mix": [],
        "error": "",
    }

    if not _env_flag("EIMAS_RA_SQL_ENABLED", default=True):
        evidence["error"] = "disabled_by_env"
        return evidence

    if not _env_flag("EIMAS_RA_COMMENTARY_AUDIT_ENABLED", default=True):
        evidence["error"] = "disabled_by_env_commentary_audit"
        return evidence

    if not isinstance(snapshot, dict):
        evidence["error"] = "invalid_snapshot"
        return evidence
    if not isinstance(commentary_payload, dict):
        evidence["error"] = "invalid_commentary_payload"
        return evidence

    commentary_source = str(commentary_payload.get("source", "rule_based"))
    model_name = str(commentary_payload.get("model", ""))

    try:
        store = RAResearchSQLStore()
        saved_id = store.insert_commentary_audit_log(
            commentary_source=commentary_source,
            model_name=model_name,
            snapshot=snapshot,
            response_payload=commentary_payload,
            prompt_text=prompt_text,
            response_text=response_text,
            error_tag=error_tag,
        )
        summary = store.summarize_commentary_audit_table()
        evidence.update(
            {
                "enabled": True,
                "db_path": str(store.db_path),
                "saved_id": saved_id,
                "total_logs": int(summary.get("total_logs", 0)),
                "last_created_at": str(summary.get("last_created_at", "")),
                "source_mix": summary.get("source_mix", []),
                "validation_sql_examples": [
                    "SELECT commentary_source, model_name, is_fallback, created_at FROM ra_commentary_audit_log ORDER BY id DESC LIMIT 20;",
                    "SELECT commentary_source, COUNT(*) AS cnt FROM ra_commentary_audit_log GROUP BY commentary_source;",
                ],
            }
        )
    except Exception as e:
        evidence["error"] = str(e)

    return evidence
