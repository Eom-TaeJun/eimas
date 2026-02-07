#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 2: Enhanced Analysis

Purpose:
    Enhanced analysis including HFT, GARCH, portfolio, and allocation

Input:
    - market_data: Dict[str, Any]
    - result: EIMASResult

Output:
    - result: EIMASResult (allocation, portfolio_weights)

Functions:
    - analyze_enhanced: Enhanced analysis orchestrator
    - set_allocation_result: Set allocation results
    - set_liquidity: Set liquidity metrics
    - set_genius_act: Set Genius Act macro regime
    - calculate_strategic_allocation_analysis: Calculate strategic allocation

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

import os
import socket
from typing import Any, Callable, Dict, Optional

import numpy as np

from pipeline.analyzers import (
    analyze_ark_trades,
    analyze_dtw_similarity,
    analyze_etf_flow,
    analyze_genius_act,
    analyze_hft_microstructure,
    analyze_information_flow,
    analyze_liquidity,
    analyze_shock_propagation,
    analyze_theme_etf,
    analyze_volume_anomalies,
    analyze_volatility_garch,
    calculate_proof_of_index,
    detect_outliers_with_dbscan,
    enhance_portfolio_with_systemic_similarity,
    optimize_portfolio_mst,
    run_allocation_engine,
    run_rebalancing_policy,
)
from pipeline.korea_integration import (
    calculate_fair_values,
    calculate_strategic_allocation,
    generate_allocation_summary,
)
from pipeline.phases.phase_cache import fetch_with_file_cache
from pipeline.phases.common import safe_call
from pipeline.schemas import EIMASResult


def _extract_close_series(data):
    """Return a close series from a DataFrame with flexible column naming."""
    if data is None or getattr(data, "empty", True):
        return None

    for key in ("Close", "close", "Adj Close", "adj_close"):
        if key in data.columns:
            return data[key]

    lowered = {str(col).lower(): col for col in data.columns}
    if "close" in lowered:
        return data[lowered["close"]]
    return None


def _count_assets_with_close_history(market_data: Dict[str, Any], min_points: int = 2) -> int:
    """Count assets that have enough close-price observations for return analytics."""
    count = 0
    for data in market_data.values():
        close_series = _extract_close_series(data)
        if close_series is None:
            continue
        try:
            if len(close_series.dropna()) >= min_points:
                count += 1
        except Exception:
            continue
    return count


def _get_metric(obj: Any, primary_key: str, legacy_key: Optional[str] = None) -> Optional[Any]:
    """Read value from dict/object with primary -> legacy fallback."""
    if obj is None:
        return None

    if isinstance(obj, dict):
        if primary_key in obj:
            return obj[primary_key]
        if legacy_key and legacy_key in obj:
            return obj[legacy_key]
        return None

    if hasattr(obj, primary_key):
        return getattr(obj, primary_key)
    if legacy_key and hasattr(obj, legacy_key):
        return getattr(obj, legacy_key)
    return None


def _phase2_cache_enabled() -> bool:
    value = os.getenv("EIMAS_PHASE2_CACHE_ENABLED", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _phase2_cache_ttl_seconds() -> int:
    raw = os.getenv("EIMAS_PHASE2_CACHE_TTL", "3600").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 3600


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _new_phase2_cache_stats() -> Dict[str, Any]:
    return {
        "enabled": _phase2_cache_enabled(),
        "ttl_seconds": _phase2_cache_ttl_seconds(),
        "hits": 0,
        "misses": 0,
        "bypassed": 0,
        "keys": {},
    }


def _record_phase2_cache_event(
    cache_stats: Optional[Dict[str, Any]],
    key: str,
    state: str,
    age_seconds: Optional[float] = None,
) -> None:
    if cache_stats is None:
        return
    keys = cache_stats.setdefault("keys", {})
    key_stats = keys.setdefault(key, {"hits": 0, "misses": 0, "bypassed": 0})

    if state == "hit":
        cache_stats["hits"] = cache_stats.get("hits", 0) + 1
        key_stats["hits"] += 1
        if age_seconds is not None:
            key_stats["last_hit_age_seconds"] = round(float(age_seconds), 3)
        return

    if state == "miss":
        cache_stats["misses"] = cache_stats.get("misses", 0) + 1
        key_stats["misses"] += 1
        return

    cache_stats["bypassed"] = cache_stats.get("bypassed", 0) + 1
    key_stats["bypassed"] += 1


def _cached_phase2_fetch(
    key: str,
    fetch_fn: Callable[[], Any],
    cache_stats: Optional[Dict[str, Any]] = None,
) -> Any:
    if not _phase2_cache_enabled():
        _record_phase2_cache_event(cache_stats, key, "bypass")
        return fetch_fn()

    ttl_seconds = _phase2_cache_ttl_seconds()
    metrics: Dict[str, Any] = {}
    value = fetch_with_file_cache(
        namespace="phase2_enhanced_v1",
        key=key,
        fetch_fn=fetch_fn,
        ttl_seconds=ttl_seconds,
        metrics=metrics,
    )
    if metrics.get("hit"):
        _record_phase2_cache_event(
            cache_stats,
            key,
            "hit",
            age_seconds=metrics.get("age_seconds"),
        )
    else:
        _record_phase2_cache_event(cache_stats, key, "miss")
    return value


def analyze_enhanced(
    result: EIMASResult,
    market_data: Dict[str, Any],
    quick_mode: bool,
    run_tactical_allocation_fn: Optional[Callable[[EIMASResult], None]] = None,
):
    """[Phase 2.2] Run enhanced metrics and optional full-mode heavy analytics."""
    print("\n[Phase 2.Enhanced] Running Advanced Metrics...")
    cache_stats = _new_phase2_cache_stats()
    result.phase2_cache_stats = cache_stats

    # Always run.
    safe_call(
        lambda: setattr(result, "hft_microstructure", analyze_hft_microstructure(market_data)),
        error_msg="HFT Error",
    )
    safe_call(
        lambda: setattr(result, "garch_volatility", analyze_volatility_garch(market_data)),
        error_msg="GARCH Error",
    )
    safe_call(
        lambda: setattr(result, "information_flow", analyze_information_flow(market_data)),
        error_msg="Info Flow Error",
    )
    safe_call(
        lambda: setattr(result, "proof_of_index", calculate_proof_of_index(market_data)),
        error_msg="PoI Error",
    )
    safe_call(
        lambda: setattr(
            result,
            "ark_analysis",
            _cached_phase2_fetch("ark_analysis", analyze_ark_trades, cache_stats=cache_stats),
        ),
        error_msg="ARK Error",
    )
    safe_call(
        lambda: enhance_portfolio_with_systemic_similarity(market_data),
        error_msg="Systemic Similarity Error",
    )
    safe_call(
        lambda: calculate_strategic_allocation_analysis(result, market_data, quick_mode),
        error_msg="Strategic Allocation Error",
    )

    # Full mode only.
    if not quick_mode:
        analytics_asset_count = _count_assets_with_close_history(market_data, min_points=2)
        enhanced_network_skip_reason = ""
        if _env_flag("EIMAS_SKIP_ENHANCED_NETWORK_ANALYTICS", default=False):
            enhanced_network_skip_reason = "EIMAS_SKIP_ENHANCED_NETWORK_ANALYTICS"
        elif not market_data:
            enhanced_network_skip_reason = "market_data_unavailable"
        elif _env_flag("EIMAS_ENHANCED_FAIL_FAST_NETWORK", default=False):
            hosts = _resolve_hosts(
                os.getenv(
                    "EIMAS_ENHANCED_NETWORK_PROBE_HOSTS",
                    "guce.yahoo.com,query1.finance.yahoo.com",
                ),
                ["guce.yahoo.com", "query1.finance.yahoo.com"],
            )
            if not _is_network_available(hosts):
                enhanced_network_skip_reason = f"dns_unavailable:{','.join(hosts)}"

        safe_call(
            lambda: setattr(result, "dtw_similarity", analyze_dtw_similarity(market_data)),
            error_msg="DTW Error",
        )
        safe_call(
            lambda: setattr(result, "dbscan_outliers", detect_outliers_with_dbscan(market_data)),
            error_msg="DBSCAN Error",
        )
        if enhanced_network_skip_reason:
            print(
                "      i Skipping enhanced network analytics "
                f"({enhanced_network_skip_reason})"
            )
            result.liquidity_signal = "NEUTRAL"
            result.liquidity_analysis = {
                "skipped": True,
                "reason": enhanced_network_skip_reason,
            }
            result.etf_flow_result = {
                "skipped": True,
                "reason": enhanced_network_skip_reason,
            }
            result.genius_act_signals = []
            result.genius_act_regime = "NEUTRAL"
            result.theme_etf_analysis = {
                "skipped": True,
                "reason": enhanced_network_skip_reason,
            }
        else:
            safe_call(lambda: set_liquidity(result, cache_stats=cache_stats), error_msg="Liquidity Error")
            safe_call(
                lambda: setattr(
                    result,
                    "etf_flow_result",
                    _cached_phase2_fetch(
                        "etf_flow_result",
                        lambda: analyze_etf_flow().to_dict(),
                        cache_stats=cache_stats,
                    ),
                ),
                error_msg="ETF Flow Error",
            )
            safe_call(lambda: set_genius_act(result, cache_stats=cache_stats), error_msg="Genius Act Error")
            safe_call(
                lambda: setattr(
                    result,
                    "theme_etf_analysis",
                    _cached_phase2_fetch(
                        "theme_etf_analysis",
                        lambda: analyze_theme_etf().to_dict(),
                        cache_stats=cache_stats,
                    ),
                ),
                error_msg="Theme ETF Error",
            )
        if analytics_asset_count >= 3:
            safe_call(
                lambda: setattr(result, "shock_propagation", analyze_shock_propagation(market_data).to_dict()),
                error_msg="Shock Error",
            )
            safe_call(
                lambda: setattr(result, "portfolio_weights", optimize_portfolio_mst(market_data).weights),
                error_msg="Portfolio Error",
            )
        else:
            skip_reason = f"insufficient_assets:{analytics_asset_count}"
            print(f"      i Skipping shock/portfolio analytics ({skip_reason})")
            result.shock_propagation = {
                "impact_score": 0.0,
                "contagion_path": [],
                "vulnerable_assets": [],
                "details": {"skipped": True, "reason": skip_reason},
            }
            result.portfolio_weights = {}
        safe_call(
            lambda: setattr(result, "volume_anomalies", analyze_volume_anomalies(market_data)),
            error_msg="Volume Error",
        )
        safe_call(lambda: set_allocation_result(result, market_data), error_msg="Allocation Engine Error")
        if run_tactical_allocation_fn is not None:
            safe_call(lambda: run_tactical_allocation_fn(result), error_msg="Tactical Allocation Error")


def set_allocation_result(result: EIMASResult, market_data: Dict[str, Any]):
    """[Phase 2.11-2.12] Run allocation engine and rebalancing decision."""
    current_weights = result.portfolio_weights if result.portfolio_weights else None

    alloc_result = run_allocation_engine(
        market_data=market_data,
        strategy="risk_parity",
        current_weights=current_weights,
    )

    result.allocation_result = alloc_result.get("allocation_result", {})
    result.allocation_strategy = alloc_result.get("allocation_strategy", "risk_parity")
    result.allocation_config = alloc_result.get("allocation_config", {})

    if alloc_result.get("rebalance_decision"):
        result.rebalance_decision = alloc_result["rebalance_decision"]
    elif current_weights and alloc_result.get("allocation_result", {}).get("weights"):
        result.rebalance_decision = run_rebalancing_policy(
            current_weights=current_weights,
            target_weights=alloc_result["allocation_result"]["weights"],
        )

    if alloc_result.get("warnings"):
        result.warnings.extend(alloc_result["warnings"])


def set_liquidity(result: EIMASResult, cache_stats: Optional[Dict[str, Any]] = None):
    """Populate liquidity signal fields."""
    payload = _cached_phase2_fetch(
        "liquidity_signal",
        lambda: analyze_liquidity().to_dict(),
        cache_stats=cache_stats,
    )
    if not isinstance(payload, dict):
        payload = {}
    result.liquidity_signal = payload.get("signal", "NEUTRAL")
    result.liquidity_analysis = payload


def set_genius_act(result: EIMASResult, cache_stats: Optional[Dict[str, Any]] = None):
    """Populate Genius Act fields."""
    payload = _cached_phase2_fetch(
        "genius_act",
        lambda: analyze_genius_act().to_dict(),
        cache_stats=cache_stats,
    )
    if not isinstance(payload, dict):
        payload = {}
    result.genius_act_signals = payload.get("signals", [])
    result.genius_act_regime = payload.get("regime", "N/A")


def calculate_strategic_allocation_analysis(
    result: EIMASResult,
    market_data: Dict[str, Any],
    quick_mode: bool,
):
    """[Phase 2.13] Fair value + strategic allocation analysis."""
    print("\n[Phase 2.13] Calculating Fair Values & Strategic Allocation...")

    try:
        bond_yields = {"us_10y": 0.042, "korea_10y": 0.035}
        us_10y = _get_metric(getattr(result, "fred_summary", None), "treasury_10y", "DGS10")
        if us_10y is not None:
            try:
                us_10y_value = float(us_10y)
                if us_10y_value > 0:
                    bond_yields["us_10y"] = us_10y_value / 100.0
            except (TypeError, ValueError):
                pass

        mode = "quick" if quick_mode else "comprehensive"
        fair_value_results = calculate_fair_values(
            market_data=market_data,
            bond_yields=bond_yields,
            mode=mode,
        )

        result.fair_value_analysis = fair_value_results
        print(
            f"  ✓ Fair Value: SPX {'✓' if 'spx' in fair_value_results else '✗'}, "
            f"KOSPI {'✓' if 'kospi' in fair_value_results else '✗'}"
        )

        market_stats = {
            "stock_return": 0.08,
            "bond_return": 0.04,
            "stock_vol": 0.16,
            "bond_vol": 0.06,
            "correlation": 0.1,
            "kospi_return": 0.06,
            "kospi_vol": 0.20,
            "us_korea_corr": 0.6,
        }

        if "SPY" in market_data:
            spy_close = _extract_close_series(market_data["SPY"])
            if spy_close is not None:
                spy_returns = spy_close.pct_change().dropna()
                if len(spy_returns) > 20:
                    market_stats["stock_return"] = spy_returns.mean() * 252
                    market_stats["stock_vol"] = spy_returns.std() * np.sqrt(252)

        include_korea = "korea_data" in market_data and market_data["korea_data"]
        allocation_results = calculate_strategic_allocation(
            fair_value_results=fair_value_results,
            market_stats=market_stats,
            risk_tolerance="moderate",
            include_korea=include_korea,
            use_evidence_based=not quick_mode,
        )

        result.strategic_allocation = allocation_results
        summary = generate_allocation_summary(allocation_results)
        print(summary)
    except Exception as e:
        print(f"  ✗ Strategic Allocation Error: {e}")
        import traceback

        traceback.print_exc()
