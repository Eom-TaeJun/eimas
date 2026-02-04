#!/usr/bin/env python3
"""
Strategic Asset Allocation Engine
==================================
전략적 자산배분 - 고전/최신 포트폴리오 이론 통합

Purpose:
- 주식/채권 비중 결정
- Fair Value 기반 전술적 조정
- 다양한 자산배분 이론 통합 실행

Economic Foundation:
===================
[고전 이론]
- Markowitz (1952): "Portfolio Selection" - Mean-Variance Optimization
- Sharpe (1964): "Capital Asset Pricing Model" - Market Portfolio
- Black-Litterman (1992): "Global Portfolio Optimization" - Bayesian Approach
- Treynor-Black (1973): "How to Use Security Analysis" - Active/Passive Mix

[최신 이론]
- Risk Parity: Qian (2005), Bridgewater Associates
- HRP: López de Prado (2016) - Machine Learning Portfolio
- Factor-Based: Fama-French (2015) 5-Factor Model
- Tactical Overlay: Ilmanen (2011) "Expected Returns"

[실무 프레임워크]
- 60/40 Portfolio: Swensen (2009) "Pioneering Portfolio Management"
- All Weather: Dalio (2012) Risk Parity approach
- Global Tactical: Asness et al. (2013) "Value and Momentum Everywhere"
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StrategicAssetAllocator:
    """전략적 자산배분 엔진"""

    def __init__(self):
        """Initialize Strategic Asset Allocator"""
        # Classic allocation templates
        self.CLASSIC_ALLOCATIONS = {
            'conservative': {'equity': 0.30, 'bond': 0.60, 'alternative': 0.10},
            'moderate': {'equity': 0.50, 'bond': 0.40, 'alternative': 0.10},
            'balanced': {'equity': 0.60, 'bond': 0.35, 'alternative': 0.05},
            'growth': {'equity': 0.75, 'bond': 0.20, 'alternative': 0.05},
            'aggressive': {'equity': 0.85, 'bond': 0.10, 'alternative': 0.05},
        }

    def markowitz_mvo(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.02,
        target_return: Optional[float] = None
    ) -> Dict:
        """
        Markowitz Mean-Variance Optimization (1952)

        Objective: Maximize Sharpe Ratio or Target Return

        Args:
            expected_returns: 예상 수익률 벡터
            cov_matrix: 공분산 행렬
            risk_free_rate: 무위험 수익률
            target_return: 목표 수익률 (None이면 Max Sharpe)

        Returns:
            Optimal weights and metrics
        """
        n_assets = len(expected_returns)

        # Objective: Minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        # Sharpe ratio (negative for minimization)
        def negative_sharpe(weights):
            port_return = weights @ expected_returns
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns - target_return
            })

        # Bounds: 0 <= weight <= 1
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weight
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        if target_return is None:
            # Maximize Sharpe
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        else:
            # Minimize variance with target return
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")

        weights = result.x
        port_return = weights @ expected_returns
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        return {
            'method': 'Markowitz MVO',
            'weights': weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }

    def black_litterman(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        views: Dict[int, float],
        view_confidences: Dict[int, float],
        risk_aversion: float = 2.5,
        tau: float = 0.025
    ) -> Dict:
        """
        Black-Litterman Model (1992)

        Combines market equilibrium with investor views

        Args:
            market_weights: 시장 균형 비중
            cov_matrix: 공분산 행렬
            views: {asset_index: expected_return} 투자 의견
            view_confidences: {asset_index: confidence} 확신도 (0~1)
            risk_aversion: 위험회피 계수 (일반적으로 2~4)
            tau: 추정 불확실성 (0.01~0.05)

        Returns:
            Posterior weights and returns
        """
        n_assets = len(market_weights)

        # 1. Market equilibrium returns (implied by market weights)
        market_cap_returns = risk_aversion * (cov_matrix @ market_weights)

        # 2. Views matrix
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        omega_diag = []

        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1
            Q[i] = view_return
            # View uncertainty (inversely proportional to confidence)
            confidence = view_confidences.get(asset_idx, 0.5)
            omega_diag.append(tau * cov_matrix[asset_idx, asset_idx] / confidence)

        Omega = np.diag(omega_diag) if omega_diag else np.eye(len(views))

        # 3. Black-Litterman formula
        # Posterior covariance
        tau_cov = tau * cov_matrix
        M_inv = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P
        posterior_cov = np.linalg.inv(M_inv)

        # Posterior returns
        posterior_returns = posterior_cov @ (
            np.linalg.inv(tau_cov) @ market_cap_returns +
            P.T @ np.linalg.inv(Omega) @ Q
        )

        # 4. Optimal weights (maximize utility)
        optimal_weights = (1 / risk_aversion) * np.linalg.inv(cov_matrix) @ posterior_returns

        # Normalize to sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        optimal_weights = np.maximum(optimal_weights, 0)  # No short selling
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        port_return = optimal_weights @ posterior_returns
        port_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)

        return {
            'method': 'Black-Litterman',
            'weights': optimal_weights,
            'posterior_returns': posterior_returns,
            'expected_return': port_return,
            'volatility': port_vol,
            'market_weights': market_weights
        }

    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        target_risk_contributions: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Risk Parity Allocation (Qian 2005, Bridgewater)

        Equalizes risk contribution from each asset

        Args:
            cov_matrix: 공분산 행렬
            target_risk_contributions: 목표 리스크 기여도 (기본: equal)

        Returns:
            Risk parity weights
        """
        n_assets = cov_matrix.shape[0]

        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets

        # Risk contribution function
        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / port_vol
            return risk_contrib

        # Objective: Minimize squared deviation from target risk contributions
        def objective(weights):
            risk_contrib = risk_contribution(weights)
            target_contrib = target_risk_contributions * np.sum(risk_contrib)
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple((0.001, 1) for _ in range(n_assets))
        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        risk_contrib = risk_contribution(weights)

        return {
            'method': 'Risk Parity',
            'weights': weights,
            'volatility': port_vol,
            'risk_contributions': risk_contrib,
            'risk_contribution_pct': risk_contrib / np.sum(risk_contrib) * 100,
            'success': result.success
        }

    def tactical_overlay(
        self,
        strategic_weights: np.ndarray,
        fair_value_signals: Dict[int, float],
        max_deviation: float = 0.15
    ) -> Dict:
        """
        Tactical Asset Allocation Overlay

        전략적 비중 + Fair Value 기반 전술적 조정

        Args:
            strategic_weights: 전략적 비중 (baseline)
            fair_value_signals: {asset_idx: valuation_gap_pct}
            max_deviation: 최대 이탈 허용 (예: 0.15 = ±15%)

        Returns:
            Tactical weights
        """
        tactical_weights = strategic_weights.copy()

        for asset_idx, valuation_gap in fair_value_signals.items():
            # Undervalued (negative gap) → Increase weight
            # Overvalued (positive gap) → Decrease weight
            adjustment_factor = -valuation_gap / 100  # Convert % to decimal

            # Limit adjustment
            adjustment = np.clip(adjustment_factor, -max_deviation, max_deviation)

            # Apply adjustment
            new_weight = strategic_weights[asset_idx] * (1 + adjustment)
            tactical_weights[asset_idx] = new_weight

        # Renormalize
        tactical_weights = np.maximum(tactical_weights, 0)
        tactical_weights = tactical_weights / np.sum(tactical_weights)

        return {
            'method': 'Tactical Overlay',
            'strategic_weights': strategic_weights,
            'tactical_weights': tactical_weights,
            'adjustments': tactical_weights - strategic_weights,
            'valuation_signals': fair_value_signals
        }

    def calculate_stock_bond_allocation(
        self,
        expected_stock_return: float,
        expected_bond_return: float,
        stock_vol: float,
        bond_vol: float,
        correlation: float,
        risk_tolerance: str = 'moderate',
        fair_value_adjustment: Optional[float] = None
    ) -> Dict:
        """
        주식/채권 비중 결정

        Args:
            expected_stock_return: 주식 예상 수익률
            expected_bond_return: 채권 예상 수익률
            stock_vol: 주식 변동성
            bond_vol: 채권 변동성
            correlation: 주식-채권 상관관계
            risk_tolerance: 'conservative', 'moderate', 'growth', 'aggressive'
            fair_value_adjustment: Fair Value 기반 조정 (%)

        Returns:
            Optimal stock/bond weights
        """
        # Base allocation from risk tolerance
        base_allocation = self.CLASSIC_ALLOCATIONS.get(
            risk_tolerance,
            self.CLASSIC_ALLOCATIONS['moderate']
        )

        stock_weight = base_allocation['equity']
        bond_weight = base_allocation['bond']

        # MVO optimization (2 assets)
        returns = np.array([expected_stock_return, expected_bond_return])
        cov_matrix = np.array([
            [stock_vol**2, correlation * stock_vol * bond_vol],
            [correlation * stock_vol * bond_vol, bond_vol**2]
        ])

        mvo_result = self.markowitz_mvo(returns, cov_matrix)

        # Fair Value tactical adjustment
        if fair_value_adjustment is not None:
            # Negative gap (undervalued) → increase stock weight
            adjustment = -fair_value_adjustment / 100 * 0.5  # 50% of signal
            stock_weight = np.clip(stock_weight + adjustment, 0.2, 0.9)
            bond_weight = 1 - stock_weight

        # Portfolio metrics
        port_return = stock_weight * expected_stock_return + bond_weight * expected_bond_return
        port_vol = np.sqrt(
            stock_weight**2 * stock_vol**2 +
            bond_weight**2 * bond_vol**2 +
            2 * stock_weight * bond_weight * correlation * stock_vol * bond_vol
        )

        return {
            'risk_tolerance': risk_tolerance,
            'base_allocation': base_allocation,
            'optimal_mvo': {
                'stock': mvo_result['weights'][0],
                'bond': mvo_result['weights'][1]
            },
            'tactical_allocation': {
                'stock': stock_weight,
                'bond': bond_weight
            },
            'portfolio_metrics': {
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': port_return / port_vol if port_vol > 0 else 0
            },
            'fair_value_adjustment': fair_value_adjustment
        }


class GlobalAssetAllocator:
    """글로벌 자산배분 (US + Korea)"""

    def __init__(self):
        self.allocator = StrategicAssetAllocator()

    def allocate_global_portfolio(
        self,
        us_expected_return: float,
        us_volatility: float,
        korea_expected_return: float,
        korea_volatility: float,
        correlation: float,
        us_fair_value_gap: float,
        korea_fair_value_gap: float,
        home_bias: float = 0.6
    ) -> Dict:
        """
        US + Korea 글로벌 포트폴리오 배분

        Args:
            us_expected_return: US 주식 예상 수익률
            us_volatility: US 변동성
            korea_expected_return: Korea 주식 예상 수익률
            korea_volatility: Korea 변동성
            correlation: US-Korea 상관관계
            us_fair_value_gap: US Fair Value gap (%)
            korea_fair_value_gap: Korea Fair Value gap (%)
            home_bias: 국내 편향 (0.6 = 60% 한국 선호)

        Returns:
            Global allocation
        """
        # Market cap weights (proxy)
        market_weights = np.array([1 - home_bias, home_bias])  # US, Korea

        # MVO optimization
        returns = np.array([us_expected_return, korea_expected_return])
        cov_matrix = np.array([
            [us_volatility**2, correlation * us_volatility * korea_volatility],
            [correlation * us_volatility * korea_volatility, korea_volatility**2]
        ])

        mvo_result = self.allocator.markowitz_mvo(returns, cov_matrix)

        # Fair Value tactical overlay
        fair_value_signals = {
            0: us_fair_value_gap,
            1: korea_fair_value_gap
        }

        tactical_result = self.allocator.tactical_overlay(
            mvo_result['weights'],
            fair_value_signals,
            max_deviation=0.20
        )

        return {
            'market_cap_weights': {
                'us': market_weights[0],
                'korea': market_weights[1]
            },
            'mvo_weights': {
                'us': mvo_result['weights'][0],
                'korea': mvo_result['weights'][1]
            },
            'tactical_weights': {
                'us': tactical_result['tactical_weights'][0],
                'korea': tactical_result['tactical_weights'][1]
            },
            'recommendation': self._get_recommendation(tactical_result['tactical_weights']),
            'portfolio_metrics': {
                'expected_return': mvo_result['expected_return'],
                'volatility': mvo_result['volatility'],
                'sharpe_ratio': mvo_result['sharpe_ratio']
            }
        }

    def _get_recommendation(self, weights: np.ndarray) -> str:
        """추천 메시지 생성"""
        us_weight, korea_weight = weights

        if us_weight > 0.7:
            return "Overweight US equities"
        elif korea_weight > 0.7:
            return "Overweight Korea equities"
        elif abs(us_weight - korea_weight) < 0.1:
            return "Balanced US-Korea allocation"
        elif us_weight > korea_weight:
            return "Tilt towards US"
        else:
            return "Tilt towards Korea"


if __name__ == "__main__":
    # Example: Stock/Bond allocation
    allocator = StrategicAssetAllocator()

    result = allocator.calculate_stock_bond_allocation(
        expected_stock_return=0.08,
        expected_bond_return=0.04,
        stock_vol=0.16,
        bond_vol=0.06,
        correlation=0.1,
        risk_tolerance='moderate',
        fair_value_adjustment=-10.0  # Stocks 10% undervalued
    )

    print("\n" + "="*80)
    print("STRATEGIC ASSET ALLOCATION")
    print("="*80)
    print(f"\nRisk Tolerance: {result['risk_tolerance']}")
    print(f"\nTactical Allocation:")
    print(f"  Stock: {result['tactical_allocation']['stock']*100:.1f}%")
    print(f"  Bond:  {result['tactical_allocation']['bond']*100:.1f}%")
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {result['portfolio_metrics']['expected_return']*100:.2f}%")
    print(f"  Volatility:      {result['portfolio_metrics']['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio:    {result['portfolio_metrics']['sharpe_ratio']:.2f}")
