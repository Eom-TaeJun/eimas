#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 2: Risk Adjustment

Purpose:
    Risk adjustment based on sentiment, bubble, and microstructure analysis

Input:
    - market_data: Dict[str, Any]
    - result: EIMASResult

Output:
    - result: EIMASResult (adjusted risk_score)

Functions:
    - analyze_sentiment_bubble: Sentiment and bubble analysis
    - apply_extended_data_adjustment: Apply extended data adjustments
    - analyze_institutional_frameworks: Analyze institutional frameworks
    - run_adaptive_portfolio_phase: Run adaptive portfolio optimization

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

import os
import socket
from datetime import datetime
from time import perf_counter
from typing import Any, Dict

from lib.bubble_framework import FiveStageBubbleFramework
from lib.fomc_analyzer import FOMCDotPlotAnalyzer
from lib.gap_analyzer import MarketModelGapAnalyzer
from pipeline.analyzers import analyze_bubble_risk, analyze_sentiment, run_adaptive_portfolio
from pipeline.risk_config import get_risk_config
from pipeline.risk_utils import derive_risk_level
from pipeline.schemas import BubbleRiskMetrics, EIMASResult


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _network_probe_hosts() -> list[str]:
    raw = os.getenv(
        "EIMAS_INSTITUTIONAL_NETWORK_PROBE_HOSTS",
        "guce.yahoo.com,query1.finance.yahoo.com",
    )
    hosts = [item.strip() for item in raw.split(",") if item.strip()]
    return hosts or ["guce.yahoo.com", "query1.finance.yahoo.com"]


def _resolve_hosts(value: str, fallback: list[str]) -> list[str]:
    hosts = [item.strip() for item in value.split(",") if item.strip()]
    return hosts or fallback


def _is_network_available(hosts: list[str]) -> bool:
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            continue
    return False


def analyze_sentiment_bubble(
    result: EIMASResult,
    market_data: Dict[str, Any],
    quick_mode: bool,
    *,
    skip_bubble: bool = False,
    skip_sentiment: bool = False,
):
    """[Phase 2.3] Run bubble risk (full) and sentiment (always)."""
    if not quick_mode:
        skip_bubble_env = _env_flag("EIMAS_SKIP_BUBBLE_ANALYSIS", default=False)
        bubble_fail_fast = _env_flag("EIMAS_BUBBLE_FAIL_FAST_NETWORK", default=False)
        bubble_reason = ""

        if skip_bubble:
            bubble_reason = "pipeline_profile_skip_bubble"
        elif skip_bubble_env:
            bubble_reason = "EIMAS_SKIP_BUBBLE_ANALYSIS"
        elif bubble_fail_fast:
            bubble_hosts = _resolve_hosts(
                os.getenv(
                    "EIMAS_BUBBLE_NETWORK_PROBE_HOSTS",
                    "guce.yahoo.com,query1.finance.yahoo.com",
                ),
                ["guce.yahoo.com", "query1.finance.yahoo.com"],
            )
            if not _is_network_available(bubble_hosts):
                bubble_reason = f"dns_unavailable:{','.join(bubble_hosts)}"

        if bubble_reason:
            result.bubble_risk = BubbleRiskMetrics(
                overall_status="SKIPPED",
                methodology_notes=f"Skipped: {bubble_reason}",
            )
            print(f"      i Bubble analysis skip ({bubble_reason})")
        else:
            try:
                bubble_res = analyze_bubble_risk(market_data)
                if bubble_res:
                    result.bubble_risk = BubbleRiskMetrics(**bubble_res)
            except Exception as e:
                print(f"⚠️ Bubble Risk Error: {e}")

    skip_sentiment_env = _env_flag("EIMAS_SKIP_SENTIMENT_ANALYSIS", default=False)
    sentiment_fail_fast = _env_flag("EIMAS_SENTIMENT_FAIL_FAST_NETWORK", default=False)
    if skip_sentiment:
        result.sentiment_analysis = {
            "skipped": True,
            "reason": "pipeline_profile_skip_sentiment",
        }
        print("      i Sentiment analysis skipped by pipeline profile")
        return

    if skip_sentiment_env:
        result.sentiment_analysis = {
            "skipped": True,
            "reason": "EIMAS_SKIP_SENTIMENT_ANALYSIS",
        }
        print("      i Sentiment analysis skipped by EIMAS_SKIP_SENTIMENT_ANALYSIS")
        return

    if sentiment_fail_fast:
        hosts = _resolve_hosts(
            os.getenv(
                "EIMAS_SENTIMENT_NETWORK_PROBE_HOSTS",
                "production.dataviz.cnn.io,guce.yahoo.com",
            ),
            ["production.dataviz.cnn.io", "guce.yahoo.com"],
        )
        if not _is_network_available(hosts):
            reason = f"dns_unavailable:{','.join(hosts)}"
            result.sentiment_analysis = {
                "skipped": True,
                "reason": reason,
            }
            print(f"      i Sentiment analysis fail-fast skip ({reason})")
            return

    try:
        result.sentiment_analysis = analyze_sentiment()
    except Exception as e:
        print(f"⚠️ Sentiment Error: {e}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_microstructure_adjustment(result: EIMASResult) -> None:
    """
    Apply risk adjustment from HFT microstructure signals.
    Design goal: avoid persistent 0.0 when valid microstructure evidence exists.
    Also populate MarketQualityMetrics for visualization.
    """
    hft = getattr(result, "hft_microstructure", {})
    if not isinstance(hft, dict) or not hft:
        return

    tick = hft.get("tick_rule", {}) if isinstance(hft.get("tick_rule"), dict) else {}
    kyle = hft.get("kyles_lambda", {}) if isinstance(hft.get("kyles_lambda"), dict) else {}

    buy_ratio = _safe_float(tick.get("buy_ratio"), default=0.5)
    impact_label = str(kyle.get("interpretation", "NEUTRAL")).upper()
    r_squared = _safe_float(kyle.get("r_squared"), default=0.0)

    # Continuous buy/sell pressure component around 0.5 neutral line.
    pressure_adj = -(buy_ratio - 0.5) * 10.0

    impact_adj = 0.0
    if "HIGH_IMPACT" in impact_label:
        impact_adj = +0.9
    elif "MEDIUM_IMPACT" in impact_label:
        impact_adj = +0.3
    elif "LOW_IMPACT" in impact_label:
        impact_adj = -0.4

    quality_adj = 0.0
    if r_squared >= 0.5:
        quality_adj = -0.2
    elif 0.0 < r_squared < 0.2:
        quality_adj = +0.3

    raw_adj = pressure_adj + impact_adj + quality_adj
    micro_adj = round(max(-3.0, min(3.0, raw_adj)), 1)

    # If microstructure block has valid signals but near-zero due cancellation,
    # keep minimal non-zero contribution to avoid losing evidence in report.
    has_signal = ("buy_ratio" in tick) or ("interpretation" in kyle)
    if has_signal and abs(micro_adj) < 0.1:
        micro_adj = -0.5 if buy_ratio >= 0.5 else +0.5

    # Populate MarketQualityMetrics
    from pipeline.schemas import MarketQualityMetrics

    # Calculate liquidity score (0-100 scale, higher = better)
    # Based on r_squared (data quality) and impact (liquidity)
    base_liquidity = 50.0
    if "LOW_IMPACT" in impact_label:
        base_liquidity += 20.0  # Low impact = high liquidity
    elif "HIGH_IMPACT" in impact_label:
        base_liquidity -= 20.0  # High impact = low liquidity

    if r_squared >= 0.5:
        base_liquidity += 10.0  # High quality
    elif r_squared < 0.2:
        base_liquidity -= 10.0  # Low quality

    avg_liquidity = max(0.0, min(100.0, base_liquidity))

    # Data quality assessment
    if r_squared >= 0.5:
        data_quality = "COMPLETE"
    elif r_squared >= 0.2:
        data_quality = "PARTIAL"
    else:
        data_quality = "DEGRADED"

    result.market_quality = MarketQualityMetrics(
        avg_liquidity_score=avg_liquidity,
        liquidity_scores={"SPY": avg_liquidity},  # Simplified - can expand per ticker
        high_toxicity_tickers=[],  # Placeholder - can add VPIN analysis
        illiquid_tickers=[] if avg_liquidity >= 50 else ["SPY"],
        data_quality=data_quality
    )

    if abs(micro_adj) < 0.1:
        return

    old_risk = result.risk_score
    result.microstructure_adjustment = micro_adj
    result.risk_score = max(1.0, min(100.0, result.risk_score + micro_adj))
    result.risk_level = derive_risk_level(result.risk_score)
    print(
        f"      ✓ Microstructure Adjustment: {micro_adj:+.1f} "
        f"({old_risk:.1f} -> {result.risk_score:.1f})"
    )
    print(
        f"        Details: buy_ratio={buy_ratio:.3f}, impact={impact_label or 'N/A'}, "
        f"r2={r_squared:.3f}, liquidity={avg_liquidity:.0f}"
    )


def _apply_bubble_adjustment(result: EIMASResult) -> None:
    bubble = getattr(result, "bubble_risk", {})
    if not isinstance(bubble, dict) or not bubble:
        return

    status = str(bubble.get("overall_status", "NONE")).upper()
    highest_score = _safe_float(bubble.get("highest_risk_score"), default=0.0)

    status_map = {
        "NONE": 0.0,
        "LOW": -0.5,
        "WATCH": +1.5,
        "MODERATE": +2.5,
        "HIGH": +4.0,
        "EXTREME": +6.0,
        "SKIPPED": 0.0,
    }
    bubble_adj = status_map.get(status, 0.0)
    if highest_score >= 70:
        bubble_adj += 1.0
    elif highest_score >= 50:
        bubble_adj += 0.5

    bubble_adj = round(max(-5.0, min(8.0, bubble_adj)), 1)
    if abs(bubble_adj) < 0.1:
        return

    old_risk = result.risk_score
    result.bubble_risk_adjustment = bubble_adj
    result.risk_score = max(1.0, min(100.0, result.risk_score + bubble_adj))
    result.risk_level = derive_risk_level(result.risk_score)
    print(
        f"      ✓ Bubble Adjustment: {bubble_adj:+.1f} "
        f"({old_risk:.1f} -> {result.risk_score:.1f})"
    )
    print(f"        Details: status={status}, highest_risk_score={highest_score:.1f}")


def apply_extended_data_adjustment(result: EIMASResult):
    """Apply risk score adjustments from microstructure/bubble + extended overlays."""
    # 1) Quant overlays first (HFT microstructure + bubble state)
    _apply_microstructure_adjustment(result)
    _apply_bubble_adjustment(result)

    # 2) Extended overlays (PCR/F&G/news/credit/KRW)
    if not result.extended_data:
        return

    # Load configuration from YAML (cached)
    config = get_risk_config()

    ext = result.extended_data
    adjustment = 0.0
    details = []

    # PCR (Put/Call Ratio) adjustment
    pcr = ext.get("put_call_ratio", {})
    if pcr.get("ratio", 0) > 0:
        ratio = pcr["ratio"]
        if ratio > config.pcr.high_threshold:
            adjustment += config.pcr.high_adjustment
            details.append(f"PCR={ratio:.2f} (Fear) -> {config.pcr.high_adjustment:+.0f}")
        elif ratio < config.pcr.low_threshold:
            adjustment += config.pcr.low_adjustment
            details.append(f"PCR={ratio:.2f} (Greed) -> {config.pcr.low_adjustment:+.0f}")

    # Crypto Fear & Greed Index adjustment
    fng = ext.get("crypto_fng", {})
    if fng.get("value", 0) > 0:
        val = fng["value"]
        if val < config.crypto_fng.fear_threshold:
            adjustment += config.crypto_fng.fear_adjustment
            details.append(f"Crypto F&G={val} (Fear) -> {config.crypto_fng.fear_adjustment:+.0f}")
        elif val > config.crypto_fng.greed_threshold:
            adjustment += config.crypto_fng.greed_adjustment
            details.append(f"Crypto F&G={val} (Greed) -> {config.crypto_fng.greed_adjustment:+.0f}")

    # News Sentiment adjustment
    news = ext.get("news_sentiment", {})
    label = news.get("label", "")
    if label == "Bearish":
        adjustment += config.news_sentiment.bearish_adjustment
        details.append(f"News=Bearish -> {config.news_sentiment.bearish_adjustment:+.0f}")
    elif label == "Bullish":
        adjustment += config.news_sentiment.bullish_adjustment
        details.append(f"News=Bullish -> {config.news_sentiment.bullish_adjustment:+.0f}")

    # Credit Spreads adjustment
    credit = ext.get("credit_spreads", {})
    interp = credit.get("interpretation", "")
    if interp == "Risk OFF":
        adjustment += config.credit_spreads.risk_off_adjustment
        details.append(f"Credit=Risk OFF -> {config.credit_spreads.risk_off_adjustment:+.0f}")
    elif interp == "Risk ON":
        adjustment += config.credit_spreads.risk_on_adjustment
        details.append(f"Credit=Risk ON -> {config.credit_spreads.risk_on_adjustment:+.0f}")

    # Korea Risk (KRW) adjustment
    krw = ext.get("korea_risk", {})
    status = krw.get("status", "")
    if "Overheated" in status:
        adjustment += config.korea_risk.overheated_adjustment
        details.append(f"KRW=Overheated -> {config.korea_risk.overheated_adjustment:+.0f}")
    elif "Volatile" in status:
        adjustment += config.korea_risk.volatile_adjustment
        details.append(f"KRW=Volatile -> {config.korea_risk.volatile_adjustment:+.0f}")

    # Apply global constraints
    adjustment = max(config.constraints.min_adjustment, min(config.constraints.max_adjustment, adjustment))

    if adjustment != 0:
        result.extended_data_adjustment = adjustment
        old_risk = result.risk_score
        result.risk_score = max(1.0, min(100, result.risk_score + adjustment))
        result.risk_level = derive_risk_level(result.risk_score)
        print(f"      ✓ Extended Data Adjustment: {adjustment:+.0f} ({old_risk:.1f} -> {result.risk_score:.1f})")
        if details:
            print(f"        Details: {', '.join(details)}")
        if result.risk_score < 5:
            warning = f"⚠️ Extremely Low Risk ({result.risk_score:.1f}/100) - Verify market conditions"
            result.warnings.append(warning)
            print(f"      {warning}")


def analyze_institutional_frameworks(result: EIMASResult, market_data: Dict[str, Any], quick_mode: bool):
    """Run institutional framework analyses (Bubble/Gap/FOMC)."""
    print("\n[Phase 2.Institutional] Running Institutional Frameworks...")
    component_timings: Dict[str, Dict[str, Any]] = {}

    def _record_component_timing(
        name: str,
        started_at: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        duration = perf_counter() - started_at
        entry: Dict[str, Any] = {
            "duration_sec": round(duration, 3),
            "status": status,
        }
        if error:
            entry["error"] = error[:200]
        component_timings[name] = entry

    skip_all = _env_flag("EIMAS_SKIP_INSTITUTIONAL_FRAMEWORKS", default=False)
    skip_network = _env_flag("EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS", default=False)
    fail_fast_network = _env_flag("EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK", default=False)
    network_available = True
    network_reason = ""
    if skip_all:
        network_available = False
        network_reason = "EIMAS_SKIP_INSTITUTIONAL_FRAMEWORKS"
    elif skip_network:
        network_available = False
        network_reason = "EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS"
    elif fail_fast_network:
        hosts = _network_probe_hosts()
        network_available = _is_network_available(hosts)
        if not network_available:
            network_reason = f"dns_unavailable:{','.join(hosts)}"

    if network_available and not market_data:
        network_available = False
        network_reason = "market_data_unavailable"

    if not network_available:
        print(f"      i Institutional network analysis skipped ({network_reason})")
        now_iso = datetime.now().isoformat()
        result.bubble_framework = {
            "timestamp": now_iso,
            "sector": "tech",
            "stage": "SKIPPED_NETWORK",
            "total_score": 0.0,
            "stage_results": [],
            "warning_flags": [],
            "skipped": True,
            "skip_reason": network_reason,
        }
        result.gap_analysis = {
            "timestamp": now_iso,
            "overall_signal": "NEUTRAL",
            "opportunity": "Institutional network analysis skipped",
            "market_too_pessimistic": False,
            "market_too_optimistic": False,
            "confidence": 0.0,
            "gaps": [],
            "skipped": True,
            "skip_reason": network_reason,
        }
        component_timings["bubble_framework"] = {"duration_sec": 0.0, "status": "skipped_network"}
        component_timings["gap_analysis"] = {"duration_sec": 0.0, "status": "skipped_network"}

    if network_available:
        bubble_started = perf_counter()
        try:
            bubble_fw = FiveStageBubbleFramework()
            bubble_result = bubble_fw.analyze(market_data, sector="tech")
            result.bubble_framework = bubble_result.to_dict()
            print(f"      ✓ 5-Stage Bubble: {bubble_result.stage} (Score: {bubble_result.total_score:.1f}/100)")
            _record_component_timing("bubble_framework", bubble_started, status="ok")
        except Exception as e:
            print(f"      ⚠️ 5-Stage Bubble Error: {e}")
            _record_component_timing(
                "bubble_framework",
                bubble_started,
                status="error",
                error=str(e),
            )

        gap_started = perf_counter()
        try:
            gap_analyzer = MarketModelGapAnalyzer()
            gap_result = gap_analyzer.analyze()
            result.gap_analysis = gap_result.to_dict()
            print(f"      ✓ Market-Model Gap: {gap_result.overall_signal} ({gap_result.opportunity[:40]}...)")
            _record_component_timing("gap_analysis", gap_started, status="ok")
        except Exception as e:
            print(f"      ⚠️ Gap Analysis Error: {e}")
            _record_component_timing(
                "gap_analysis",
                gap_started,
                status="error",
                error=str(e),
            )

    if not quick_mode:
        fomc_started = perf_counter()
        try:
            fomc_analyzer = FOMCDotPlotAnalyzer()
            fomc_result = fomc_analyzer.analyze("2026")
            result.fomc_analysis = fomc_result.to_dict()
            print(
                f"      ✓ FOMC Analysis: {fomc_result.stance} "
                f"(Uncertainty: {fomc_result.policy_uncertainty_index:.0f}/100)"
            )
            _record_component_timing("fomc_analysis", fomc_started, status="ok")
        except Exception as e:
            print(f"      ⚠️ FOMC Analysis Error: {e}")
            _record_component_timing(
                "fomc_analysis",
                fomc_started,
                status="error",
                error=str(e),
            )
    else:
        component_timings["fomc_analysis"] = {"duration_sec": 0.0, "status": "skipped_quick_mode"}

    result.audit_metadata["phase2_institutional_components"] = component_timings


def run_adaptive_portfolio_phase(result: EIMASResult, regime_res: Any, quick_mode: bool):
    """[Phase 2.4] Run adaptive portfolio only in full mode."""
    if not quick_mode:
        try:
            result.adaptive_portfolios = run_adaptive_portfolio(regime_res)
        except Exception as e:
            print(f"⚠️ Adaptive Portfolio Error: {e}")
