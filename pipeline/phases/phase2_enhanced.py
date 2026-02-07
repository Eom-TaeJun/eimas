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


def analyze_enhanced(
    result: EIMASResult,
    market_data: Dict[str, Any],
    quick_mode: bool,
    run_tactical_allocation_fn: Optional[Callable[[EIMASResult], None]] = None,
):
    """[Phase 2.2] Run enhanced metrics and optional full-mode heavy analytics."""
    print("\n[Phase 2.Enhanced] Running Advanced Metrics...")

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
    safe_call(lambda: setattr(result, "ark_analysis", analyze_ark_trades()), error_msg="ARK Error")
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
        safe_call(
            lambda: setattr(result, "dtw_similarity", analyze_dtw_similarity(market_data)),
            error_msg="DTW Error",
        )
        safe_call(
            lambda: setattr(result, "dbscan_outliers", detect_outliers_with_dbscan(market_data)),
            error_msg="DBSCAN Error",
        )
        safe_call(lambda: set_liquidity(result), error_msg="Liquidity Error")
        safe_call(
            lambda: setattr(result, "etf_flow_result", analyze_etf_flow().to_dict()),
            error_msg="ETF Flow Error",
        )
        safe_call(lambda: set_genius_act(result), error_msg="Genius Act Error")
        safe_call(
            lambda: setattr(result, "theme_etf_analysis", analyze_theme_etf().to_dict()),
            error_msg="Theme ETF Error",
        )
        safe_call(
            lambda: setattr(result, "shock_propagation", analyze_shock_propagation(market_data).to_dict()),
            error_msg="Shock Error",
        )
        safe_call(
            lambda: setattr(result, "portfolio_weights", optimize_portfolio_mst(market_data).weights),
            error_msg="Portfolio Error",
        )
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


def set_liquidity(result: EIMASResult):
    """Populate liquidity signal fields."""
    liq_res = analyze_liquidity()
    result.liquidity_signal = liq_res.signal
    result.liquidity_analysis = liq_res.to_dict()


def set_genius_act(result: EIMASResult):
    """Populate Genius Act fields."""
    ga_res = analyze_genius_act()
    result.genius_act_signals = ga_res.signals
    result.genius_act_regime = ga_res.regime


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
