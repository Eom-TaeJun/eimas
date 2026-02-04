"""
Korea Market & Fair Value Integration for Pipeline
===================================================
한국 자산 + Fair Value + 전략적 자산배분 파이프라인 통합

Usage in main.py:
    from pipeline.korea_integration import (
        collect_korea_assets,
        calculate_fair_values,
        calculate_strategic_allocation
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from lib.korea_data_collector import KoreaDataCollector, ParallelKoreaCollector
from lib.fair_value_calculator import FairValueCalculator, compare_fair_values
from lib.strategic_allocation import StrategicAssetAllocator, GlobalAssetAllocator
from lib.evidence_based_allocator import EvidenceBasedAllocator

logger = logging.getLogger(__name__)


def collect_korea_assets(
    lookback_days: int = 365,
    use_parallel: bool = True
) -> Dict:
    """
    한국 자산 데이터 수집

    Args:
        lookback_days: 과거 데이터 기간
        use_parallel: 병렬 수집 사용 여부

    Returns:
        {
            'data': {category: {asset: DataFrame}},
            'summary': {...},
            'latest_prices': {...}
        }
    """
    try:
        if use_parallel:
            collector = ParallelKoreaCollector(
                lookback_days=lookback_days,
                max_workers=8
            )
        else:
            collector = KoreaDataCollector(lookback_days=lookback_days)

        # Collect all data
        data = collector.collect_all()

        # Get latest prices
        latest_prices = {}
        for category, assets in data.items():
            for name, df in assets.items():
                if not df.empty and 'Close' in df.columns:
                    latest_prices[f"{category}_{name}"] = float(df['Close'].iloc[-1])

        # KOSPI statistics
        kospi_stats = {}
        if 'indices' in data and 'KOSPI' in data['indices']:
            kospi_df = data['indices']['KOSPI']
            if not kospi_df.empty:
                prices = kospi_df['Close']
                returns = prices.pct_change().dropna()

                kospi_stats = {
                    'current_price': float(prices.iloc[-1]),
                    'ytd_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                    'volatility': returns.std() * np.sqrt(252) * 100,
                    'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                }

        summary = {
            'total_assets': sum(len(v) for v in data.values()),
            'categories': list(data.keys()),
            'kospi_stats': kospi_stats,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        return {
            'data': data,
            'summary': summary,
            'latest_prices': latest_prices
        }

    except Exception as e:
        logger.error(f"Korea asset collection failed: {e}")
        return {
            'data': {},
            'summary': {'error': str(e)},
            'latest_prices': {}
        }


def calculate_fair_values(
    market_data: Dict,
    bond_yields: Dict[str, float],
    mode: str = 'comprehensive'
) -> Dict:
    """
    Fair Value 계산 (SPX, KOSPI)

    Args:
        market_data: 시장 데이터
        bond_yields: {'us_10y': 0.042, 'korea_10y': 0.035}
        mode: 'comprehensive' or 'quick'

    Returns:
        {
            'spx': {...},
            'kospi': {...},
            'comparison': {...}
        }
    """
    calculator = FairValueCalculator()
    results = {}

    try:
        # SPX Fair Value (if data available)
        if 'SPY' in market_data:
            spy_data = market_data['SPY']
            current_price = float(spy_data['Close'].iloc[-1])

            # Estimate EPS (proxy: P/E = 20)
            eps_estimate = current_price / 20

            if mode == 'comprehensive':
                spx_analysis = calculator.calculate_comprehensive_fair_value(
                    current_price=current_price,
                    eps=eps_estimate,
                    average_earnings_10y=eps_estimate * 0.95,  # Proxy
                    dividend=eps_estimate * 0.4,  # 40% payout ratio
                    bond_yield=bond_yields.get('us_10y', 0.042),
                    earnings_growth=0.07,
                    market='spx'
                )
                results['spx'] = spx_analysis
            else:
                # Quick mode: Fed Model only
                fed_result = calculator.calculate_fed_model_fair_value(
                    current_price=current_price,
                    earnings_per_share=eps_estimate,
                    bond_yield_10y=bond_yields.get('us_10y', 0.042),
                    market='spx'
                )
                results['spx'] = {'fed_model': fed_result, 'current_price': current_price}

        # KOSPI Fair Value (if data available)
        if 'korea_data' in market_data and 'indices' in market_data['korea_data']:
            kospi_indices = market_data['korea_data']['indices']
            if 'KOSPI' in kospi_indices:
                kospi_data = kospi_indices['KOSPI']
                current_price = float(kospi_data['Close'].iloc[-1])

                eps_estimate = current_price / 12  # KOSPI P/E typically lower

                if mode == 'comprehensive':
                    kospi_analysis = calculator.calculate_comprehensive_fair_value(
                        current_price=current_price,
                        eps=eps_estimate,
                        average_earnings_10y=eps_estimate * 0.9,
                        dividend=eps_estimate * 0.5,  # Higher payout in Korea
                        bond_yield=bond_yields.get('korea_10y', 0.035),
                        earnings_growth=0.05,
                        market='kospi'
                    )
                    results['kospi'] = kospi_analysis
                else:
                    # Quick mode
                    fed_result = calculator.calculate_fed_model_fair_value(
                        current_price=current_price,
                        earnings_per_share=eps_estimate,
                        bond_yield_10y=bond_yields.get('korea_10y', 0.035),
                        market='kospi'
                    )
                    results['kospi'] = {'fed_model': fed_result, 'current_price': current_price}

        # Comparison (if both available)
        if 'spx' in results and 'kospi' in results and mode == 'comprehensive':
            comparison = compare_fair_values(results['spx'], results['kospi'])
            results['comparison'] = comparison

    except Exception as e:
        logger.error(f"Fair value calculation failed: {e}")
        results['error'] = str(e)

    return results


def calculate_strategic_allocation(
    fair_value_results: Dict,
    market_stats: Dict,
    risk_tolerance: str = 'moderate',
    include_korea: bool = True,
    use_evidence_based: bool = True
) -> Dict:
    """
    전략적 자산배분 계산

    Args:
        fair_value_results: Fair value 분석 결과
        market_stats: 시장 통계 (return, vol, correlation)
        risk_tolerance: 'conservative', 'moderate', 'growth', 'aggressive'
        include_korea: 한국 자산 포함 여부
        use_evidence_based: 증거 기반 계산 사용 (하드코딩 제거)

    Returns:
        {
            'stock_bond_allocation': {...},
            'global_allocation': {...} (if include_korea),
            'tactical_signals': {...},
            'calculation_evidence': [...] (if use_evidence_based)
        }
    """
    results = {}

    try:
        # NEW: Evidence-Based Allocation (Full 모드 JSON 활용)
        if use_evidence_based:
            logger.info("Using Evidence-Based Allocator (Full JSON results)")

            evidence_allocator = EvidenceBasedAllocator()
            full_analysis = evidence_allocator.load_latest_full_analysis()

            if full_analysis:
                # Extract market data from Full JSON
                market_data = evidence_allocator.extract_market_data(full_analysis)

                # Calculate allocation with evidence
                allocation_result = evidence_allocator.calculate_risk_adjusted_allocation(
                    market_data,
                    risk_tolerance=risk_tolerance
                )

                # Format as stock_bond_allocation
                stock_bond_result = {
                    'risk_tolerance': risk_tolerance,
                    'tactical_allocation': {
                        'stock': allocation_result['stock'],
                        'bond': allocation_result['bond']
                    },
                    'portfolio_metrics': {
                        'expected_return': allocation_result['expected_return'],
                        'volatility': allocation_result['expected_volatility'],
                        'sharpe_ratio': allocation_result['sharpe_ratio']
                    },
                    'evidence_based': True,
                    'calculation_details': allocation_result['calculation_details'],
                    'market_data_used': allocation_result['market_data_used']
                }

                results['stock_bond_allocation'] = stock_bond_result
                results['calculation_evidence'] = allocation_result['calculation_details']

                logger.info(f"Evidence-based allocation: Stock {allocation_result['stock']*100:.1f}%, Bond {allocation_result['bond']*100:.1f}%")

            else:
                logger.warning("No Full JSON found, falling back to traditional method")
                use_evidence_based = False

        # FALLBACK: Traditional method (if no Full JSON or disabled)
        if not use_evidence_based:
            allocator = StrategicAssetAllocator()

            # 1. Stock/Bond Allocation
            us_fair_gap = 0
            if 'spx' in fair_value_results:
                if 'consensus' in fair_value_results['spx']:
                    us_fair_gap = fair_value_results['spx']['consensus']['valuation_gap_pct']
                elif 'fed_model' in fair_value_results['spx']:
                    us_fair_gap = fair_value_results['spx']['fed_model']['valuation_gap_pct']

            stock_bond_result = allocator.calculate_stock_bond_allocation(
                expected_stock_return=market_stats.get('stock_return', 0.08),
                expected_bond_return=market_stats.get('bond_return', 0.04),
                stock_vol=market_stats.get('stock_vol', 0.16),
                bond_vol=market_stats.get('bond_vol', 0.06),
                correlation=market_stats.get('correlation', 0.1),
                risk_tolerance=risk_tolerance,
                fair_value_adjustment=us_fair_gap
            )

            results['stock_bond_allocation'] = stock_bond_result

        # 2. Global Allocation (US + Korea)
        if include_korea and 'kospi' in fair_value_results:
            korea_fair_gap = 0
            if 'consensus' in fair_value_results['kospi']:
                korea_fair_gap = fair_value_results['kospi']['consensus']['valuation_gap_pct']
            elif 'fed_model' in fair_value_results['kospi']:
                korea_fair_gap = fair_value_results['kospi']['fed_model']['valuation_gap_pct']

            global_allocator = GlobalAssetAllocator()
            global_result = global_allocator.allocate_global_portfolio(
                us_expected_return=market_stats.get('stock_return', 0.08),
                us_volatility=market_stats.get('stock_vol', 0.16),
                korea_expected_return=market_stats.get('kospi_return', 0.06),
                korea_volatility=market_stats.get('kospi_vol', 0.20),
                correlation=market_stats.get('us_korea_corr', 0.6),
                us_fair_value_gap=us_fair_gap,
                korea_fair_value_gap=korea_fair_gap,
                home_bias=0.4  # 60% US, 40% Korea baseline
            )

            results['global_allocation'] = global_result

        # 3. Tactical Signals
        tactical_signals = {
            'us_equity': 'NEUTRAL',
            'korea_equity': 'NEUTRAL',
            'bonds': 'NEUTRAL'
        }

        if us_fair_gap < -10:
            tactical_signals['us_equity'] = 'OVERWEIGHT'
        elif us_fair_gap > 10:
            tactical_signals['us_equity'] = 'UNDERWEIGHT'

        if include_korea and 'kospi' in fair_value_results:
            if korea_fair_gap < -10:
                tactical_signals['korea_equity'] = 'OVERWEIGHT'
            elif korea_fair_gap > 10:
                tactical_signals['korea_equity'] = 'UNDERWEIGHT'

        results['tactical_signals'] = tactical_signals

    except Exception as e:
        logger.error(f"Strategic allocation failed: {e}")
        results['error'] = str(e)

    return results


# ============================================================================
# Summary Generator
# ============================================================================

def generate_allocation_summary(allocation_results: Dict) -> str:
    """자산배분 결과 요약 텍스트 생성"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("STRATEGIC ASSET ALLOCATION SUMMARY")
    lines.append("="*80)

    # Evidence-Based Calculation Details (if available)
    if 'calculation_evidence' in allocation_results:
        lines.append("\n[Evidence-Based Calculation]")
        lines.append("Using Full mode JSON results with quantitative formulas")
        lines.append("")

    # Stock/Bond Allocation
    if 'stock_bond_allocation' in allocation_results:
        sb = allocation_results['stock_bond_allocation']
        tactical = sb['tactical_allocation']

        # Check if evidence-based
        is_evidence = sb.get('evidence_based', False)
        method_label = "Evidence-Based" if is_evidence else "Traditional"

        lines.append(f"\n[1] Stock/Bond Allocation ({method_label}):")
        lines.append(f"  Risk Tolerance: {sb['risk_tolerance']}")
        lines.append(f"  Stock: {tactical['stock']*100:.1f}%")
        lines.append(f"  Bond:  {tactical['bond']*100:.1f}%")

        metrics = sb['portfolio_metrics']
        lines.append(f"\n  Portfolio Metrics:")
        lines.append(f"    Expected Return: {metrics['expected_return']*100:.2f}%")
        lines.append(f"    Volatility:      {metrics['volatility']*100:.2f}%")
        lines.append(f"    Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")

        # Show market data used (if evidence-based)
        if is_evidence and 'market_data_used' in sb:
            md = sb['market_data_used']
            lines.append(f"\n  Data Sources Used:")
            lines.append(f"    Regime: {md.get('regime', 'N/A')} ({md.get('regime_confidence', 0)*100:.0f}% confidence)")
            lines.append(f"    Risk Score: {md.get('risk_score', 0):.1f}/100")
            if 'volatility' in md:
                lines.append(f"    Volatility: {md['volatility']*100:.2f}% (GARCH model)")
            if 'bubble_status' in md:
                lines.append(f"    Bubble Risk: {md['bubble_status']}")

    # Global Allocation
    if 'global_allocation' in allocation_results:
        ga = allocation_results['global_allocation']
        tactical = ga['tactical_weights']

        lines.append("\n[2] Global Allocation (US + Korea):")
        lines.append(f"  US:    {tactical['us']*100:.1f}%")
        lines.append(f"  Korea: {tactical['korea']*100:.1f}%")
        lines.append(f"  Recommendation: {ga['recommendation']}")

    # Tactical Signals
    if 'tactical_signals' in allocation_results:
        signals = allocation_results['tactical_signals']

        lines.append("\n[3] Tactical Signals:")
        for asset, signal in signals.items():
            emoji = '↑' if signal == 'OVERWEIGHT' else ('↓' if signal == 'UNDERWEIGHT' else '→')
            lines.append(f"  {emoji} {asset.replace('_', ' ').title()}: {signal}")

    lines.append("\n" + "="*80)

    return "\n".join(lines)
