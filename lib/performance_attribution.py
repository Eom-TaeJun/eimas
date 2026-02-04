"""
Performance Attribution
=======================
Brinson-Hood-Beebower 성과 귀속 분석

References:
- Brinson, Hood, Beebower (1986): "Determinants of Portfolio Performance"
- Brinson, Singer, Beebower (1991): "Determinants of Portfolio Performance II"
- Karnosky, Singer (1994): "Global Asset Management and Performance Attribution"

Key Finding:
"93.6% of return variation is explained by asset allocation policy"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """
    귀속 분석 결과

    Brinson Attribution:
    Total Excess Return = Allocation Effect + Selection Effect + Interaction Effect
    """
    # Total
    portfolio_return: float
    benchmark_return: float
    excess_return: float  # Active return

    # Effects
    allocation_effect: float      # 자산배분 효과
    selection_effect: float        # 종목선택 효과
    interaction_effect: float      # 상호작용 효과

    # Breakdown by asset class
    allocation_breakdown: Dict[str, float] = field(default_factory=dict)
    selection_breakdown: Dict[str, float] = field(default_factory=dict)

    # Additional metrics
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    active_share: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'portfolio_return': self.portfolio_return,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'allocation_effect': self.allocation_effect,
            'selection_effect': self.selection_effect,
            'interaction_effect': self.interaction_effect,
            'allocation_breakdown': self.allocation_breakdown,
            'selection_breakdown': self.selection_breakdown,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'active_share': self.active_share
        }

    def verify(self) -> bool:
        """검증: Allocation + Selection + Interaction = Excess Return"""
        total_effect = self.allocation_effect + self.selection_effect + self.interaction_effect
        return abs(total_effect - self.excess_return) < 1e-6


class BrinsonAttribution:
    """
    Brinson-Hood-Beebower 귀속 분석

    공식:
    1. Allocation Effect = Σ (w_p - w_b) * R_b
       "포트폴리오 비중과 벤치마크 비중 차이가 벤치마크 수익률에 미치는 영향"

    2. Selection Effect = Σ w_b * (R_p - R_b)
       "벤치마크 비중으로 포트폴리오 수익률과 벤치마크 수익률 차이"

    3. Interaction Effect = Σ (w_p - w_b) * (R_p - R_b)
       "비중 차이와 수익률 차이의 상호작용"

    where:
    - w_p: 포트폴리오 비중
    - w_b: 벤치마크 비중
    - R_p: 포트폴리오 수익률
    - R_b: 벤치마크 수익률
    """

    def __init__(self):
        pass

    def compute(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_weights: Dict[str, float],
        benchmark_returns: Dict[str, float]
    ) -> AttributionResult:
        """
        단일 기간 귀속 분석

        Args:
            portfolio_weights: {asset: weight}
            portfolio_returns: {asset: return}
            benchmark_weights: {asset: weight}
            benchmark_returns: {asset: return}

        Returns:
            AttributionResult
        """

        # Get all assets
        all_assets = set(
            list(portfolio_weights.keys()) +
            list(benchmark_weights.keys()) +
            list(portfolio_returns.keys()) +
            list(benchmark_returns.keys())
        )

        # Initialize effects
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0

        allocation_breakdown = {}
        selection_breakdown = {}

        for asset in all_assets:
            w_p = portfolio_weights.get(asset, 0.0)
            w_b = benchmark_weights.get(asset, 0.0)
            r_p = portfolio_returns.get(asset, 0.0)
            r_b = benchmark_returns.get(asset, 0.0)

            # Allocation effect for this asset
            alloc = (w_p - w_b) * r_b
            allocation_effect += alloc
            allocation_breakdown[asset] = alloc

            # Selection effect for this asset
            select = w_b * (r_p - r_b)
            selection_effect += select
            selection_breakdown[asset] = select

            # Interaction effect for this asset
            interact = (w_p - w_b) * (r_p - r_b)
            interaction_effect += interact

        # Total returns
        portfolio_return = sum(portfolio_weights.get(a, 0) * portfolio_returns.get(a, 0)
                              for a in all_assets)
        benchmark_return = sum(benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0)
                             for a in all_assets)
        excess_return = portfolio_return - benchmark_return

        result = AttributionResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            allocation_breakdown=allocation_breakdown,
            selection_breakdown=selection_breakdown
        )

        # Verify
        if not result.verify():
            logger.warning(f"Attribution verification failed. "
                          f"Sum of effects: {allocation_effect + selection_effect + interaction_effect:.6f}, "
                          f"Excess return: {excess_return:.6f}")

        return result

    def compute_multi_period(
        self,
        portfolio_weights_series: pd.DataFrame,
        portfolio_returns_series: pd.DataFrame,
        benchmark_weights_series: pd.DataFrame,
        benchmark_returns_series: pd.DataFrame
    ) -> pd.DataFrame:
        """
        다기간 귀속 분석

        Args:
            *_series: DataFrame with dates as index, assets as columns

        Returns:
            DataFrame with attribution effects over time
        """
        results = []

        for date in portfolio_returns_series.index:
            pw = portfolio_weights_series.loc[date].to_dict()
            pr = portfolio_returns_series.loc[date].to_dict()
            bw = benchmark_weights_series.loc[date].to_dict()
            br = benchmark_returns_series.loc[date].to_dict()

            result = self.compute(pw, pr, bw, br)

            results.append({
                'date': date,
                'portfolio_return': result.portfolio_return,
                'benchmark_return': result.benchmark_return,
                'excess_return': result.excess_return,
                'allocation_effect': result.allocation_effect,
                'selection_effect': result.selection_effect,
                'interaction_effect': result.interaction_effect
            })

        return pd.DataFrame(results).set_index('date')


class InformationRatio:
    """
    정보비율 (Information Ratio) 계산

    IR = E[R_p - R_b] / TE
    where TE = std(R_p - R_b) is Tracking Error

    Interpretation:
    - IR > 0.5: Good active management
    - IR > 1.0: Excellent active management
    - IR < 0: Underperformance
    """

    @staticmethod
    def compute(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        정보비율 계산

        Returns:
            (information_ratio, tracking_error, active_return)
        """
        # Active returns
        active_returns = portfolio_returns - benchmark_returns

        # Active return (mean)
        active_return = active_returns.mean()

        # Tracking error (std)
        tracking_error = active_returns.std()

        # Information ratio
        if tracking_error > 0:
            information_ratio = active_return / tracking_error
        else:
            information_ratio = 0.0

        return information_ratio, tracking_error, active_return

    @staticmethod
    def annualize(
        information_ratio: float,
        tracking_error: float,
        active_return: float,
        periods_per_year: int = 252
    ) -> Tuple[float, float, float]:
        """
        연환산

        Returns:
            (ann_ir, ann_te, ann_active_return)
        """
        ann_active_return = active_return * periods_per_year
        ann_tracking_error = tracking_error * np.sqrt(periods_per_year)

        if ann_tracking_error > 0:
            ann_ir = ann_active_return / ann_tracking_error
        else:
            ann_ir = 0.0

        return ann_ir, ann_tracking_error, ann_active_return


class ActiveShare:
    """
    Active Share 계산

    AS = 0.5 * Σ |w_p - w_b|

    Interpretation:
    - AS = 0%: 완전 패시브 (인덱스 펀드)
    - AS = 100%: 완전 액티브 (벤치마크와 겹치는 종목 없음)
    - AS > 60%: Concentrated/High conviction
    - AS 20-60%: Moderate active
    - AS < 20%: Closet indexer
    """

    @staticmethod
    def compute(
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float]
    ) -> float:
        """Active Share 계산"""

        all_assets = set(list(portfolio_weights.keys()) + list(benchmark_weights.keys()))

        active_share = 0.0
        for asset in all_assets:
            w_p = portfolio_weights.get(asset, 0.0)
            w_b = benchmark_weights.get(asset, 0.0)
            active_share += abs(w_p - w_b)

        return active_share / 2.0


class CaptureRatios:
    """
    Up/Down Capture Ratios

    Up Capture = (R_p when R_b > 0) / (R_b when R_b > 0)
    Down Capture = (R_p when R_b < 0) / (R_b when R_b < 0)

    Ideal:
    - Up Capture > 100%: 상승장에서 시장 초과
    - Down Capture < 100%: 하락장에서 손실 제한
    """

    @staticmethod
    def compute(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Up/Down Capture Ratios 계산

        Returns:
            (up_capture, down_capture, capture_ratio)
        """
        # Up periods
        up_mask = benchmark_returns > 0
        up_portfolio = portfolio_returns[up_mask].mean()
        up_benchmark = benchmark_returns[up_mask].mean()
        up_capture = (up_portfolio / up_benchmark) if up_benchmark != 0 else 0

        # Down periods
        down_mask = benchmark_returns < 0
        down_portfolio = portfolio_returns[down_mask].mean()
        down_benchmark = benchmark_returns[down_mask].mean()
        down_capture = (down_portfolio / down_benchmark) if down_benchmark != 0 else 0

        # Capture ratio (Up / Down)
        capture_ratio = (up_capture / down_capture) if down_capture != 0 else 0

        return up_capture, down_capture, capture_ratio


def generate_attribution_report(
    attribution_result: AttributionResult
) -> str:
    """귀속 분석 리포트 생성"""

    report = f"""
# Performance Attribution Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Portfolio Return: {attribution_result.portfolio_return*100:.2f}%
- Benchmark Return: {attribution_result.benchmark_return*100:.2f}%
- **Excess Return: {attribution_result.excess_return*100:.2f}%**

## Brinson Attribution
| Effect | Value | % of Excess |
|--------|-------|-------------|
| **Allocation Effect** | {attribution_result.allocation_effect*100:.2f}% | {attribution_result.allocation_effect/attribution_result.excess_return*100:.1f}% |
| **Selection Effect** | {attribution_result.selection_effect*100:.2f}% | {attribution_result.selection_effect/attribution_result.excess_return*100:.1f}% |
| **Interaction Effect** | {attribution_result.interaction_effect*100:.2f}% | {attribution_result.interaction_effect/attribution_result.excess_return*100:.1f}% |

### Interpretation
"""

    if attribution_result.allocation_effect > attribution_result.selection_effect:
        report += "- Asset allocation contributed more than security selection\n"
    else:
        report += "- Security selection contributed more than asset allocation\n"

    if attribution_result.information_ratio:
        report += f"\n## Additional Metrics\n"
        report += f"- Information Ratio: {attribution_result.information_ratio:.2f}\n"
        report += f"- Tracking Error: {attribution_result.tracking_error*100:.2f}%\n"
        report += f"- Active Share: {attribution_result.active_share*100:.1f}%\n"

    return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example
    portfolio_weights = {'SPY': 0.6, 'TLT': 0.3, 'GLD': 0.1}
    portfolio_returns = {'SPY': 0.10, 'TLT': 0.03, 'GLD': 0.08}

    benchmark_weights = {'SPY': 0.6, 'TLT': 0.4, 'GLD': 0.0}
    benchmark_returns = {'SPY': 0.10, 'TLT': 0.03, 'GLD': 0.08}

    # Brinson Attribution
    brinson = BrinsonAttribution()
    result = brinson.compute(
        portfolio_weights, portfolio_returns,
        benchmark_weights, benchmark_returns
    )

    print("=== Brinson Attribution ===")
    print(f"Portfolio Return: {result.portfolio_return*100:.2f}%")
    print(f"Benchmark Return: {result.benchmark_return*100:.2f}%")
    print(f"Excess Return: {result.excess_return*100:.2f}%")
    print(f"\nAllocation Effect: {result.allocation_effect*100:.4f}%")
    print(f"Selection Effect: {result.selection_effect*100:.4f}%")
    print(f"Interaction Effect: {result.interaction_effect*100:.4f}%")
    print(f"\nVerification: {result.verify()}")

    # Active Share
    active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
    print(f"\nActive Share: {active_share*100:.1f}%")
