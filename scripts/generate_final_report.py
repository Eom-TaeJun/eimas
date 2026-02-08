#!/usr/bin/env python3
"""
EIMAS Final Report Generator
=============================
실행된 분석 결과(JSON)를 바탕으로 AI 기반 심층 리포트를 생성합니다.

기능:
1. 최신 eimas_*.json 로드 (legacy integrated_*.json fallback)
2. RA 스타일(자산배분팀 리서치) 리포트 생성
3. 선택적으로 IB 스타일 Memorandum 생성 (--style ib)
4. 결과 저장
"""

import json
import asyncio
import argparse
import shutil
import subprocess
import re
from datetime import datetime
from pathlib import Path

try:
    from _project_bootstrap import ensure_project_root
except ImportError:
    from scripts._project_bootstrap import ensure_project_root

PROJECT_ROOT = ensure_project_root(__file__)

from lib.ai_report_generator import AIReportGenerator
from lib.allocation_report_agent import AllocationReportAgent

try:
    from scripts.convert_md_to_html import convert_md_to_html
except ImportError:
    from convert_md_to_html import convert_md_to_html


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _find_latest_json(output_dir: Path) -> Path | None:
    json_files = sorted(output_dir.glob("eimas_*.json"), reverse=True)
    if not json_files:
        json_files = sorted(output_dir.glob("integrated_*.json"), reverse=True)
    return json_files[0] if json_files else None


def _load_analysis_result(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()


def _append_visual_appendix(report_content: str, visual_assets: list[dict]) -> str:
    if not visual_assets:
        return report_content

    lines = [report_content, "", "## 부록 A. PDF 시각자료", ""]
    for idx, asset in enumerate(visual_assets, 1):
        title = asset.get("title", f"Figure {idx}")
        rel_path = asset.get("rel_path", "")
        source = asset.get("source", "EIMAS JSON")
        note = asset.get("note", "")
        lines.append(f"### Figure {idx}. {title}")
        if rel_path:
            lines.append(f"![{title}]({rel_path})")
        if note:
            lines.append(f"- 설명: {note}")
        lines.append(f"- 출처: `{source}`")
        lines.append("")
    return "\n".join(lines)


def _generate_ra_visual_assets(analysis_result: dict, markdown_path: Path) -> list[dict]:
    """
    RA 보고서용 PNG 도표 생성.
    실패해도 리포트 생성은 계속 진행한다.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figure_root = markdown_path.parent / "figures" / _safe_slug(markdown_path.stem)
    figure_root.mkdir(parents=True, exist_ok=True)

    assets: list[dict] = []

    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _fmt_signed_pct(value: float) -> str:
        return f"{value:+.2f}%"

    def _fmt_pct(value: float) -> str:
        return f"{value:.1f}%"

    def _resolve_monitor_metric(primary, fallbacks):
        """Pick first positive numeric metric from primary + fallback candidates."""
        primary_val = _safe_float(primary, default=0.0)
        if primary_val > 0:
            return primary_val
        for fb in fallbacks:
            val = _safe_float(fb, default=0.0)
            if val > 0:
                return val
        return primary_val

    # Figure 1: Portfolio weights
    weights = (
        (analysis_result.get("allocation_result") or {}).get("weights")
        or analysis_result.get("portfolio_weights")
        or {}
    )
    if isinstance(weights, dict) and weights:
        top = sorted(
            ((str(k), float(v)) for k, v in weights.items() if isinstance(v, (int, float))),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        if top:
            tickers = [x[0] for x in top]
            values = [x[1] * 100 for x in top]
            fig_path = figure_root / "portfolio_weights_top10.png"
            plt.figure(figsize=(9, 4.8))
            bars = plt.bar(tickers, values, color="#58a6ff")
            plt.title("Portfolio Target Weights (Top 10)")
            plt.ylabel("Weight (%)")
            plt.xticks(rotation=30, ha="right")
            for bar, val in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            plt.tight_layout()
            plt.savefig(fig_path, dpi=180)
            plt.close()
            assets.append(
                {
                    "title": "포트폴리오 목표비중 Top 10",
                    "path": str(fig_path),
                    "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                    "source": "allocation_result.weights / portfolio_weights",
                    "note": (
                        f"상위 비중은 {tickers[0]} {_fmt_pct(values[0])}로 가장 크며, "
                        f"Top3 합산 비중은 {_fmt_pct(sum(values[:3]))}."
                    ),
                }
            )

    # Figure 2: Risk score decomposition snapshot
    risk_score = analysis_result.get("risk_score", 0.0)
    base = analysis_result.get("base_risk_score", 0.0)
    micro = analysis_result.get("microstructure_adjustment", 0.0)
    bubble = analysis_result.get("bubble_risk_adjustment", 0.0)
    extended = analysis_result.get("extended_data_adjustment", 0.0)
    fig_path = figure_root / "risk_score_decomposition.png"
    labels = ["Base", "Micro Adj", "Bubble Adj", "Extended Adj", "Final Risk"]
    vals = [base, micro, bubble, extended, risk_score]
    colors = ["#3fb950", "#d29922", "#f85149", "#58a6ff", "#c9d1d9"]
    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.title("Risk Score Decomposition Snapshot")
    plt.ylabel("Score")
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -0.8),
            f"{val:.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    assets.append(
        {
            "title": "리스크 점수 분해",
            "path": str(fig_path),
            "rel_path": str(fig_path.relative_to(markdown_path.parent)),
            "source": "risk_score / base_risk_score / *_adjustment",
            "note": (
                f"최종 리스크 {risk_score:.1f}는 Base {base:.1f}, "
                f"Micro {micro:+.1f}, Bubble {bubble:+.1f}, Extended {extended:+.1f} 조합으로 산출."
            ),
        }
    )

    # Figure 3: Decision snapshot
    final_rec = str(analysis_result.get("final_recommendation", "N/A"))
    confidence = float(analysis_result.get("confidence", 0.0) or 0.0)
    fig_path = figure_root / "decision_snapshot.png"
    plt.figure(figsize=(9, 2.8))
    plt.barh(["Confidence"], [confidence * 100], color="#3fb950")
    plt.xlim(0, 100)
    plt.title(f"Decision Snapshot: {final_rec}")
    plt.xlabel("Confidence (%)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    assets.append(
            {
                "title": f"최종 권고/신뢰도 스냅샷 ({final_rec})",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "final_recommendation / confidence",
                "note": f"최종 권고는 {final_rec}, 신뢰도는 {confidence * 100:.1f}% 수준.",
            }
        )

    # Figure 4: Macro snapshot (rates / inflation / credit)
    fred = analysis_result.get("fred_summary", {}) or {}
    macro_labels = [
        ("Fed Funds", _safe_float(fred.get("fed_funds"))),
        ("2Y", _safe_float(fred.get("treasury_2y"))),
        ("10Y", _safe_float(fred.get("treasury_10y"))),
        ("HY OAS", _safe_float(fred.get("hy_oas"))),
        ("CPI YoY", _safe_float(fred.get("cpi_yoy"))),
    ]
    macro_labels = [(k, v) for k, v in macro_labels if v != 0.0]
    if macro_labels:
        fig_path = figure_root / "macro_snapshot_rates_credit.png"
        labels = [x[0] for x in macro_labels]
        vals = [x[1] for x in macro_labels]
        plt.figure(figsize=(9, 4.8))
        bars = plt.bar(labels, vals, color=["#58a6ff", "#3fb950", "#3fb950", "#d29922", "#f85149"][: len(vals)])
        plt.title("Macro Snapshot (Rates / Credit / Inflation)")
        plt.ylabel("%")
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "거시 스냅샷(금리/신용/물가)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "fred_summary",
                "note": (
                    f"Fed Funds {fred.get('fed_funds', 'N/A')}, 10Y {fred.get('treasury_10y', 'N/A')}, "
                    f"HY OAS {fred.get('hy_oas', 'N/A')} 기준 매크로/신용 환경을 동시 점검."
                ),
            }
        )

    # Figure 5: Company valuation map (P/E vs P/B)
    companies = ((analysis_result.get("company_ra_analysis") or {}).get("companies")) or []
    points = []
    for item in companies:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        valuation = item.get("valuation", {}) or {}
        pe = _safe_float(valuation.get("trailing_pe"), default=-1)
        pb = _safe_float(valuation.get("price_to_book"), default=-1)
        if ticker and pe > 0 and pb > 0:
            points.append((ticker, pe, pb))
    if points:
        fig_path = figure_root / "company_valuation_map.png"
        plt.figure(figsize=(9, 5.2))
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        plt.scatter(x, y, color="#58a6ff", s=80, alpha=0.9)
        for ticker, pe, pb in points:
            plt.text(pe * 1.01, pb * 1.01, ticker, fontsize=8)
        plt.xlabel("Trailing P/E")
        plt.ylabel("Price-to-Book (P/B)")
        plt.title("Company Valuation Map (Coverage Universe)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "기업 밸류에이션 맵(P/E vs P/B)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "company_ra_analysis.companies[].valuation",
                "note": (
                    f"커버리지 {len(points)}종목 기준 P/E {min(x):.1f}~{max(x):.1f}, "
                    f"P/B {min(y):.1f}~{max(y):.1f} 분포."
                ),
            }
        )

    # Figure 5b: ETF momentum snapshot
    etf_snapshot = ((analysis_result.get("company_ra_analysis") or {}).get("etf_strategy_snapshot")) or []
    etf_rows = []
    for item in etf_snapshot:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        ret_20_raw = item.get("ret_20d_pct")
        if ret_20_raw is None:
            continue
        ret_20 = _safe_float(ret_20_raw, default=0.0)
        etf_rows.append((ticker, ret_20))
    if etf_rows:
        etf_rows = sorted(etf_rows, key=lambda x: x[1], reverse=True)[:12]
        fig_path = figure_root / "etf_momentum_snapshot.png"
        labels = [x[0] for x in etf_rows]
        vals = [x[1] for x in etf_rows]
        colors = ["#3fb950" if v >= 0 else "#f85149" for v in vals]
        plt.figure(figsize=(9, 4.8))
        bars = plt.bar(labels, vals, color=colors)
        plt.title("ETF Momentum Snapshot (20D Return %)")
        plt.ylabel("20D Return (%)")
        plt.xticks(rotation=20, ha="right")
        for bar, val in zip(bars, vals):
            offset = 0.2 if val >= 0 else -0.4
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "ETF 모멘텀 스냅샷(20일 수익률)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "company_ra_analysis.etf_strategy_snapshot",
                "note": (
                    f"상위 {labels[0]} {_fmt_signed_pct(vals[0])}, "
                    f"하위 {labels[-1]} {_fmt_signed_pct(vals[-1])}로 모멘텀 스프레드 {vals[0]-vals[-1]:.2f}%p."
                ),
            }
        )

    # Figure 6: SQL evidence dashboard
    ra = analysis_result.get("company_ra_analysis", {}) or {}
    pg = ra.get("postgresql", {}) or {}
    internal = ra.get("internal_sql", {}) or {}
    company_sql = internal if isinstance(internal, dict) and "upserted_rows" in internal else (internal.get("company", {}) or {})
    phase6_sql = (internal.get("phase6_backtest", {}) or {}) if isinstance(internal, dict) else {}
    paper_bt_sql = ((analysis_result.get("paper_execution_backtest") or {}).get("ra_sql")) or {}
    total_bt_runs = _safe_float(phase6_sql.get("total_runs"), default=_safe_float(paper_bt_sql.get("total_runs"), 0.0))

    row_counts = (company_sql.get("sql_artifacts", {}) or {}).get("row_counts", {}) if isinstance(company_sql, dict) else {}
    sql_metrics = [
        ("PG stored_rows", _safe_float(pg.get("stored_rows"), 0.0)),
        ("Internal upserted", _safe_float(company_sql.get("upserted_rows"), 0.0)),
        ("Internal total_rows", _safe_float(company_sql.get("total_rows"), 0.0)),
        ("ETF rows", _safe_float(company_sql.get("etf_total_rows"), 0.0)),
        ("Backtest runs", total_bt_runs),
        ("Valuation MV", _safe_float(row_counts.get("ra_valuation_snapshot_mv"), 0.0)),
        ("ETF MV", _safe_float(row_counts.get("ra_etf_momentum_snapshot_mv"), 0.0)),
        ("Allocation MV", _safe_float(row_counts.get("ra_allocation_signal_mv"), 0.0)),
    ]
    if any(v > 0 for _, v in sql_metrics):
        fig_path = figure_root / "sql_evidence_dashboard.png"
        labels = [x[0] for x in sql_metrics]
        vals = [x[1] for x in sql_metrics]
        plt.figure(figsize=(9, 4.8))
        color_cycle = ["#3fb950", "#58a6ff", "#58a6ff", "#58a6ff", "#d29922", "#a371f7", "#a371f7", "#2ea043"]
        bars = plt.bar(labels, vals, color=color_cycle[: len(labels)])
        plt.title("SQL Evidence Dashboard (Ingestion + Backtest)")
        plt.ylabel("Count")
        plt.xticks(rotation=15, ha="right")
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "SQL 증빙 대시보드(적재 + 백테스트)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "company_ra_analysis.postgresql / company_ra_analysis.internal_sql / paper_execution_backtest.ra_sql",
                "note": (
                    f"적재 {sql_metrics[2][1]:.0f}건, ETF {sql_metrics[3][1]:.0f}건, "
                    f"Backtest {sql_metrics[4][1]:.0f}건, Allocation MV {sql_metrics[7][1]:.0f}건."
                ),
            }
        )

    # Figure 7: Monitoring dashboard snapshot
    indicators = analysis_result.get("market_indicators", {}) or {}
    sentiment = analysis_result.get("sentiment_analysis", {}) or {}
    vix_struct = sentiment.get("vix_structure", {}) if isinstance(sentiment.get("vix_structure"), dict) else {}
    gap_analysis = analysis_result.get("gap_analysis", {}) or {}
    confidence = _safe_float(analysis_result.get("confidence"), 0.0)
    gap_confidence = _safe_float(gap_analysis.get("confidence"), 0.0)
    risk_score_pipeline = _safe_float(analysis_result.get("risk_score"), 0.0)

    vix_value = _resolve_monitor_metric(
        indicators.get("vix_current"),
        [vix_struct.get("vix_spot")],
    )
    market_risk_value = _resolve_monitor_metric(
        indicators.get("risk_score"),
        [risk_score_pipeline],
    )
    # opportunity_score가 비어있을 때는 gap confidence(%) -> decision confidence(%) 순으로 대체
    opportunity_value = _resolve_monitor_metric(
        indicators.get("opportunity_score"),
        [gap_confidence * 100.0, confidence * 100.0],
    )

    if _safe_float(indicators.get("vix_current"), 0.0) > 0:
        vix_source = "market_indicators.vix_current"
    elif _safe_float(vix_struct.get("vix_spot"), 0.0) > 0:
        vix_source = "sentiment_analysis.vix_structure.vix_spot"
    else:
        vix_source = "market_indicators.vix_current"

    if _safe_float(indicators.get("risk_score"), 0.0) > 0:
        market_risk_source = "market_indicators.risk_score"
    else:
        market_risk_source = "risk_score"

    if _safe_float(indicators.get("opportunity_score"), 0.0) > 0:
        opportunity_source = "market_indicators.opportunity_score"
    elif gap_confidence > 0:
        opportunity_source = "gap_analysis.confidence(%)"
    else:
        opportunity_source = "confidence(%)"

    monitor_metrics = [
        ("VIX", vix_value),
        ("Market Risk", market_risk_value),
        ("Opportunity", opportunity_value),
        ("Pipeline Risk", risk_score_pipeline),
    ]
    if any(v > 0 for _, v in monitor_metrics):
        fig_path = figure_root / "monitoring_dashboard_snapshot.png"
        labels = [x[0] for x in monitor_metrics]
        vals = [x[1] for x in monitor_metrics]
        plt.figure(figsize=(9, 4.8))
        bars = plt.bar(labels, vals, color=["#f85149", "#d29922", "#3fb950", "#58a6ff"])
        plt.title("Monitoring Dashboard Snapshot")
        plt.ylabel("Score / Level")
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "모니터링 대시보드 스냅샷(VIX/리스크/기회)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": f"{vix_source} + {market_risk_source} + {opportunity_source} + risk_score",
                "note": (
                    f"VIX {monitor_metrics[0][1]:.1f}, Market Risk {monitor_metrics[1][1]:.1f}, "
                    f"Opportunity {monitor_metrics[2][1]:.1f}, Pipeline Risk {monitor_metrics[3][1]:.1f}."
                ),
            }
        )

    # Figure 8: Backtest result metrics
    bt_metrics = analysis_result.get("backtest_metrics", {}) or {}
    if not bt_metrics:
        paper_bt = analysis_result.get("paper_execution_backtest", {}) or {}
        bt_metrics = paper_bt.get("metrics", {}) if isinstance(paper_bt, dict) else {}

    if isinstance(bt_metrics, dict) and bt_metrics:
        total_return_pct = _safe_float(bt_metrics.get("total_return"), 0.0) * 100.0
        annual_return_pct = _safe_float(bt_metrics.get("annualized_return"), 0.0) * 100.0
        max_dd_pct = abs(_safe_float(bt_metrics.get("max_drawdown"), 0.0)) * 100.0
        sharpe = _safe_float(bt_metrics.get("sharpe_ratio"), 0.0)
        win_rate_pct = _safe_float(bt_metrics.get("win_rate"), 0.0) * 100.0
        synthetic_fallback = bool(
            ((analysis_result.get("paper_execution_backtest") or {}).get("synthetic_price_fallback", False))
        )
        perf_vals = [
            ("Total Return %", total_return_pct),
            ("Ann Return %", annual_return_pct),
            ("MaxDD %", max_dd_pct),
            ("Sharpe", sharpe),
            ("WinRate %", win_rate_pct),
        ]
        fig_path = figure_root / "backtest_metrics_snapshot.png"
        plt.figure(figsize=(9, 4.8))
        bars = plt.bar([x[0] for x in perf_vals], [x[1] for x in perf_vals], color="#58a6ff")
        plt.title("Backtest Metrics Snapshot")
        for bar, (_, val) in zip(bars, perf_vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        assets.append(
            {
                "title": "백테스트 성과 스냅샷(수익률/MDD/Sharpe/승률)",
                "path": str(fig_path),
                "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                "source": "backtest_metrics / paper_execution_backtest.metrics",
                "note": (
                    f"총수익률 {total_return_pct:+.2f}%, Sharpe {sharpe:.2f}, MaxDD {max_dd_pct:.2f}%."
                    + (" (네트워크 제약으로 synthetic price fallback 사용)" if synthetic_fallback else "")
                ),
            }
        )
    else:
        ra = analysis_result.get("company_ra_analysis", {}) or {}
        internal = ra.get("internal_sql", {}) or {}
        phase6_sql = internal.get("phase6_backtest", {}) if isinstance(internal, dict) else {}
        if isinstance(phase6_sql, dict) and phase6_sql:
            vals = [
                ("Backtest Runs", _safe_float(phase6_sql.get("total_runs"), 0.0)),
                ("Avg Sharpe", _safe_float(phase6_sql.get("avg_sharpe"), 0.0)),
            ]
            fig_path = figure_root / "backtest_sql_summary.png"
            plt.figure(figsize=(7.5, 4.4))
            bars = plt.bar([x[0] for x in vals], [x[1] for x in vals], color=["#d29922", "#58a6ff"])
            plt.title("Backtest SQL Summary")
            for bar, (_, val) in zip(bars, vals):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            plt.tight_layout()
            plt.savefig(fig_path, dpi=180)
            plt.close()
            assets.append(
                {
                    "title": "백테스트 SQL 요약(실행 건수/평균 Sharpe)",
                    "path": str(fig_path),
                    "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                    "source": "company_ra_analysis.internal_sql.phase6_backtest",
                    "note": (
                        f"SQL 저장 기준 실행 {vals[0][1]:.0f}건, 평균 Sharpe {vals[1][1]:.2f}."
                    ),
                }
            )
        else:
            fig_path = figure_root / "backtest_placeholder.png"
            plt.figure(figsize=(8.2, 3.8))
            plt.axis("off")
            plt.text(
                0.5,
                0.62,
                "Backtest Metrics Not Available",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="#f85149",
            )
            plt.text(
                0.5,
                0.40,
                "Run: python main.py --full --backtest",
                ha="center",
                va="center",
                fontsize=11,
                color="#c9d1d9",
            )
            plt.text(
                0.5,
                0.26,
                "or python main.py --full --paper-auto --paper-backtest",
                ha="center",
                va="center",
                fontsize=10,
                color="#8b949e",
            )
            plt.tight_layout()
            plt.savefig(fig_path, dpi=180)
            plt.close()
            assets.append(
                {
                    "title": "백테스트 결과(실행 전 상태 안내)",
                    "path": str(fig_path),
                    "rel_path": str(fig_path.relative_to(markdown_path.parent)),
                    "source": "backtest_metrics / paper_execution_backtest",
                    "note": "백테스트 미실행 시에도 보고서에 상태를 명시",
                }
            )

    return assets


def _generate_ra_report(analysis_result: dict, output_dir: Path) -> tuple[Path, str]:
    print("\n🚀 Generating RA-style allocation research report...")
    report_agent = AllocationReportAgent()
    initial_report = report_agent.generate_report(analysis_result)
    saved_path = Path(report_agent.save_report(initial_report, output_dir=str(output_dir / "reports")))

    visual_assets = _generate_ra_visual_assets(analysis_result, saved_path)
    analysis_result["generated_visual_assets"] = visual_assets

    refreshed_report = report_agent.generate_report(analysis_result)
    report_content = _append_visual_appendix(refreshed_report.to_markdown(), visual_assets)

    with open(saved_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    json_path = saved_path.with_suffix(".json")
    try:
        payload = refreshed_report.to_dict()
        payload["generated_visual_assets"] = visual_assets
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    analysis_result.pop("generated_visual_assets", None)

    return saved_path, report_content


async def _generate_ib_report(analysis_result: dict, output_dir: Path) -> tuple[Path, str]:
    print("\n🚀 Generating Investment Banking Memorandum...")
    generator = AIReportGenerator(verbose=True)
    report_content = await generator.generate_ib_report(analysis_result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"EIMAS_IB_Memorandum_{timestamp}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content or "")

    return output_file, report_content or ""


def _export_markdown_pdf(markdown_path: Path) -> tuple[Path | None, Path | None]:
    """Markdown -> HTML -> PDF export using wkhtmltopdf."""
    if not markdown_path.exists():
        print(f"⚠️ PDF export skipped: markdown missing ({markdown_path})")
        return None, None

    html_path = markdown_path.with_suffix(".html")
    pdf_path = markdown_path.with_suffix(".pdf")

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    convert_md_to_html(md_content, str(html_path))

    wkhtmltopdf = shutil.which("wkhtmltopdf")
    if not wkhtmltopdf:
        print("⚠️ PDF export skipped: wkhtmltopdf not installed")
        return html_path, None

    result = subprocess.run(
        [
            wkhtmltopdf,
            "--enable-local-file-access",
            "--encoding",
            "utf-8",
            "--page-size",
            "A4",
            str(html_path),
            str(pdf_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 and not pdf_path.exists():
        print(f"⚠️ PDF export failed: exit={result.returncode}, stderr={result.stderr.strip()[:200]}")
        return html_path, None

    return html_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EIMAS final report")
    parser.add_argument(
        "--style",
        choices=["ra", "ib"],
        default="ra",
        help="Report style: ra (default) or ib",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input JSON path (default: latest outputs/eimas_*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for report artifacts",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export Markdown report to HTML/PDF (wkhtmltopdf required)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    print("=" * 60)
    print("EIMAS AI Report Generator")
    print("=" * 60)

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 분석 결과 로드
    if args.input:
        latest_file = _resolve_path(args.input)
    else:
        latest_file = _find_latest_json(output_dir)

    if latest_file is None or not latest_file.exists():
        print("❌ 분석 결과 파일(eimas_*.json)을 찾을 수 없습니다.")
        print("먼저 'python main.py --full'를 실행해주세요.")
        return

    print(f"📂 Loading analysis: {latest_file}")

    try:
        analysis_result = _load_analysis_result(latest_file)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # 2. 스타일별 리포트 생성
    try:
        if args.style == "ib":
            output_file, report_content = await _generate_ib_report(analysis_result, output_dir)
        else:
            output_file, report_content = _generate_ra_report(analysis_result, output_dir)

        if not report_content.strip():
            print("❌ 리포트 내용이 비어있습니다.")
            return

        print(f"\n✅ Report generated successfully!")
        print(f"📄 Style: {args.style.upper()}")
        print(f"📄 Saved to: {output_file}")

        print("\n" + "=" * 60)
        print("REPORT PREVIEW (First 500 chars)")
        print("=" * 60)
        print(report_content[:500] + "...")
        print("=" * 60)

        if args.pdf and output_file.suffix.lower() == ".md":
            print("\n🧾 Exporting report to HTML/PDF...")
            html_path, pdf_path = _export_markdown_pdf(output_file)
            if html_path:
                print(f"📄 HTML Saved to: {html_path}")
            if pdf_path:
                print(f"📄 PDF Saved to: {pdf_path}")
            else:
                print("⚠️ PDF file was not generated.")

    except Exception as e:
        print(f"❌ 리포트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
