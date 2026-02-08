#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 7: AI Report Generation

Purpose:
    AI report generation and report validation workflow

Input:
    - result: EIMASResult
    - market_data: Dict[str, Any]
    - generate: bool
    - output_dir: Path
    - output_file: str (optional, canonical artifact path)

Output:
    - report_content: str

Functions:
    - generate_report: AI report generator
    - validate_report: Whitening + fact-check validator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pipeline.schemas import EIMASResult
from pipeline.report import generate_ai_report, run_whitening_check, run_fact_check
from pipeline.storage import save_result_json


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_ark_effectively_empty(ark_analysis: Any) -> bool:
    if not isinstance(ark_analysis, dict) or not ark_analysis:
        return True
    signal_keys = (
        "consensus_buys",
        "consensus_sells",
        "new_positions",
        "top_increases",
        "top_decreases",
        "signals",
    )
    return all(not ark_analysis.get(key) for key in signal_keys)


def _extract_report_data(ai_report_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ai_report_payload, dict):
        return {}
    report_data = ai_report_payload.get("report_data")
    return report_data if isinstance(report_data, dict) else {}


def _derive_ark_from_ai_report(ai_report_payload: Dict[str, Any]) -> Dict[str, Any]:
    report_data = _extract_report_data(ai_report_payload)
    notable = report_data.get("notable_stocks", [])
    if not isinstance(notable, list):
        return {}

    consensus_buys: List[str] = []
    consensus_sells: List[str] = []
    top_increases: List[Dict[str, Any]] = []
    top_decreases: List[Dict[str, Any]] = []
    signals: List[Dict[str, Any]] = []

    for item in notable[:10]:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip()
        if not ticker:
            continue

        reason = str(
            item.get("notable_reason")
            or item.get("news_summary")
            or "AI notable stock signal"
        ).strip()
        if not reason:
            reason = "AI notable stock signal"

        signals.append(
            {
                "ticker": ticker,
                "action": "WATCH",
                "reason": reason,
                "source": "ai_report.notable_stocks",
            }
        )

        change_1d = _safe_float(item.get("change_1d"))
        if change_1d is not None:
            if change_1d >= 2.0 and ticker not in consensus_buys:
                consensus_buys.append(ticker)
            elif change_1d <= -2.0 and ticker not in consensus_sells:
                consensus_sells.append(ticker)

        change_5d = _safe_float(item.get("change_5d"))
        if change_5d is not None:
            target = top_increases if change_5d > 0 else top_decreases
            target.append({"ticker": ticker, "change_5d": round(change_5d, 2)})

    if not signals:
        return {}

    top_increases.sort(key=lambda row: row.get("change_5d", 0.0), reverse=True)
    top_decreases.sort(key=lambda row: row.get("change_5d", 0.0))

    return {
        "timestamp": datetime.now().isoformat(),
        "consensus_buys": consensus_buys[:5],
        "consensus_sells": consensus_sells[:5],
        "new_positions": [],
        "top_increases": top_increases[:5],
        "top_decreases": top_decreases[:5],
        "signals": signals[:10],
        "derived": True,
        "derived_from": "ai_report.notable_stocks",
    }


def _derive_news_from_ai_report(ai_report_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    report_data = _extract_report_data(ai_report_payload)
    news_text = report_data.get("perplexity_news")
    if not isinstance(news_text, str):
        return []

    text = news_text.strip()
    if not text:
        return []

    entries: List[Dict[str, Any]] = []
    bullet_pattern = re.compile(r"^\s*-\s*\*\*([^*]+)\*\*:\s*(.+)$", re.MULTILINE)
    for match in bullet_pattern.finditer(text):
        asset = " ".join(match.group(1).split())
        summary = " ".join(match.group(2).split())
        if not asset or not summary:
            continue
        entries.append(
            {
                "asset": asset,
                "summary": summary[:400],
                "source": "ai_report.perplexity_news",
                "derived": True,
            }
        )

    if entries:
        return entries[:10]

    paragraphs = [chunk.strip() for chunk in re.split(r"\n{2,}", text) if chunk.strip()]
    for chunk in paragraphs[:3]:
        compact = " ".join(chunk.split())
        if not compact:
            continue
        entries.append(
            {
                "asset": "MARKET",
                "summary": compact[:400],
                "source": "ai_report.perplexity_news",
                "derived": True,
            }
        )

    return entries


def _apply_ai_report_fallback_enrichment(
    result: EIMASResult,
    ai_report_payload: Dict[str, Any],
) -> Dict[str, Any]:
    enrichment_meta: Dict[str, Any] = {}

    if _is_ark_effectively_empty(result.ark_analysis):
        derived_ark = _derive_ark_from_ai_report(ai_report_payload)
        if derived_ark:
            result.ark_analysis = derived_ark
            enrichment_meta["ark_analysis_source"] = "ai_report.notable_stocks"
            enrichment_meta["ark_signal_count"] = len(derived_ark.get("signals", []))

    if not result.news_correlations:
        derived_news = _derive_news_from_ai_report(ai_report_payload)
        if derived_news:
            result.news_correlations = derived_news
            enrichment_meta["news_source"] = "ai_report.perplexity_news"
            enrichment_meta["news_item_count"] = len(derived_news)

    return enrichment_meta


async def generate_report(
    result: EIMASResult,
    market_data: Dict[str, Any],
    generate: bool,
    output_dir: Path,
    output_file: str = "",
) -> str:
    """[Phase 7] Generate AI report and persist updated result."""
    if not generate:
        return ""

    print("\n[Phase 7] Generating Report...")
    try:
        ai_report = await generate_ai_report(result, market_data)
        result.ai_report = ai_report.to_dict() if hasattr(ai_report, "to_dict") else ai_report.__dict__
        enrichment_meta = _apply_ai_report_fallback_enrichment(result, result.ai_report)
        if enrichment_meta:
            result.audit_metadata["phase7_enrichment"] = enrichment_meta
            print(
                "      ✓ AI report fallback enrichment: "
                f"{enrichment_meta}"
            )
        save_result_json(result, output_dir=output_dir, output_file=output_file or None)
        return ai_report.content
    except Exception as e:
        print(f"⚠️ Report Generation Error: {e}")
        return ""


async def validate_report(
    result: EIMASResult,
    report_content: str,
    generate: bool,
    output_dir: Path,
    output_file: str = "",
):
    """[Phase 7.x] Validate generated report and persist updated result."""
    if not generate:
        return

    print("\n" + "=" * 50)
    print("PHASE 7: VALIDATION")
    print("=" * 50)

    try:
        result.whitening_summary = run_whitening_check(result)
        if report_content:
            result.fact_check_grade = await run_fact_check(report_content)
        save_result_json(result, output_dir=output_dir, output_file=output_file or None)
    except Exception as e:
        print(f"⚠️ Validation Error: {e}")
