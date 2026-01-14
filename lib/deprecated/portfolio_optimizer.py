#!/usr/bin/env python3
"""
EIMAS Portfolio Optimizer
=========================
현대 포트폴리오 이론 기반 최적화

주요 기능:
1. Mean-Variance Optimization (Markowitz)
2. Black-Litterman Model
3. Risk Parity
4. Efficient Frontier

Usage:
    from lib.portfolio_optimizer import PortfolioOptimizer

    optimizer = PortfolioOptimizer(['SPY', 'TLT', 'GLD', 'QQQ'])
    result = optimizer.optimize_sharpe()
    print(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, Bounds, LinearConstraint
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


# ============================================================================
# Constants
# ============================================================================

RISK_FREE_RATE = 0.045  # 4.5% 무위험수익률 (2025년 기준)
TRADING_DAYS = 252


# ============================================================================
# Data Classes
# ============================================================================

class OptimizationType(str, Enum):
    """최적화 유형"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    RISK_PARITY = "risk_parity"
    MAX_RETURN = "max_return"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"


@dataclass
class OptimizationConstraints:
    """최적화 제약조건"""
    min_weight: float = 0.0          # 최소 비중
    max_weight: float = 1.0          # 최대 비중
    max_turnover: float = 1.0        # 최대 회전율
    sector_limits: Dict[str, float] = field(default_factory=dict)
    asset_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_type: OptimizationType
    converged: bool
    iterations: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'optimization_type': self.optimization_type.value,
            'converged': self.converged,
        }


@dataclass
class EfficientFrontierPoint:
    """효율적 프론티어 포인트"""
    return_target: float
    volatility: float
    sharpe: float
    weights: Dict[str, float]


@dataclass
class BlackLittermanResult:
    """Black-Litterman 결과"""
    prior_returns: Dict[str, float]
    posterior_returns: Dict[str, float]
    optimal_weights: Dict[str, float]
    views_incorporated: int
    confidence_adjustment: float


# ============================================================================
# Portfolio Optimizer
# ============================================================================

class PortfolioOptimizer:
    """포트폴리오 최적화"""

    def __init__(
        self,
        assets: List[str],
        risk_free_rate: float = RISK_FREE_RATE,
        lookback_days: int = 252,
    ):
        self.assets = assets
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        self.returns: Optional[pd.DataFrame] = None
        self.mean_returns: Optional[pd.Series] = None
        self.cov_matrix: Optional[pd.DataFrame] = None

    def fetch_data(self, period: str = None) -> bool:
        """가격 데이터 수집"""
        try:
            period = period or f"{self.lookback_days}d"
            df = yf.download(
                self.assets,
                period=period,
                progress=False,
            )['Close']

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 수익률 계산
            self.returns = df.pct_change().dropna()

            # 연율화된 평균 수익률
            self.mean_returns = self.returns.mean() * TRADING_DAYS

            # 연율화된 공분산
            self.cov_matrix = self.returns.cov() * TRADING_DAYS

            print(f"Data loaded: {len(self.assets)} assets, {len(self.returns)} days")
            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """포트폴리오 통계 계산"""
        weights = np.array(weights)
        returns = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0
        return returns, volatility, sharpe

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        """음의 샤프비율 (최소화용)"""
        _, _, sharpe = self._portfolio_stats(weights)
        return -sharpe

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """포트폴리오 변동성"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """포트폴리오 수익률"""
        return np.dot(weights, self.mean_returns)

    # ========================================================================
    # Optimization Methods
    # ========================================================================

    def optimize_sharpe(
        self,
        constraints: OptimizationConstraints = None,
    ) -> OptimizationResult:
        """샤프비율 최대화"""
        if self.returns is None:
            self.fetch_data()

        constraints = constraints or OptimizationConstraints()
        n = len(self.assets)

        # 초기 가중치 (동일 가중)
        init_weights = np.array([1/n] * n)

        # 제약조건
        bounds = Bounds(constraints.min_weight, constraints.max_weight)

        # 비중 합 = 1
        eq_constraint = LinearConstraint(np.ones(n), 1, 1)

        # 최적화
        result = minimize(
            self._neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)

        return OptimizationResult(
            weights={asset: float(w) for asset, w in zip(self.assets, weights)},
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_type=OptimizationType.MAX_SHARPE,
            converged=result.success,
            iterations=result.nit,
        )

    def optimize_min_variance(
        self,
        constraints: OptimizationConstraints = None,
    ) -> OptimizationResult:
        """최소 분산 포트폴리오"""
        if self.returns is None:
            self.fetch_data()

        constraints = constraints or OptimizationConstraints()
        n = len(self.assets)

        init_weights = np.array([1/n] * n)
        bounds = Bounds(constraints.min_weight, constraints.max_weight)

        result = minimize(
            self._portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)

        return OptimizationResult(
            weights={asset: float(w) for asset, w in zip(self.assets, weights)},
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_type=OptimizationType.MIN_VARIANCE,
            converged=result.success,
            iterations=result.nit,
        )

    def optimize_risk_parity(self) -> OptimizationResult:
        """리스크 패리티 포트폴리오"""
        if self.returns is None:
            self.fetch_data()

        n = len(self.assets)

        def risk_parity_objective(weights):
            """리스크 기여도 균등화"""
            weights = np.array(weights)
            port_vol = self._portfolio_volatility(weights)

            # 한계 리스크 기여도
            marginal_risk = np.dot(self.cov_matrix, weights) / port_vol
            risk_contribution = weights * marginal_risk

            # 목표: 모든 자산의 리스크 기여도 동일
            target_risk = port_vol / n
            return np.sum((risk_contribution - target_risk) ** 2)

        init_weights = np.array([1/n] * n)
        bounds = Bounds(0.01, 1.0)  # 최소 1%

        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)

        return OptimizationResult(
            weights={asset: float(w) for asset, w in zip(self.assets, weights)},
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_type=OptimizationType.RISK_PARITY,
            converged=result.success,
            iterations=result.nit,
        )

    def optimize_target_return(
        self,
        target_return: float,
        constraints: OptimizationConstraints = None,
    ) -> OptimizationResult:
        """목표 수익률에서 최소 변동성"""
        if self.returns is None:
            self.fetch_data()

        constraints = constraints or OptimizationConstraints()
        n = len(self.assets)

        init_weights = np.array([1/n] * n)
        bounds = Bounds(constraints.min_weight, constraints.max_weight)

        result = minimize(
            self._portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: self._portfolio_return(w) - target_return},
            ],
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)

        return OptimizationResult(
            weights={asset: float(w) for asset, w in zip(self.assets, weights)},
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_type=OptimizationType.TARGET_RETURN,
            converged=result.success,
            iterations=result.nit,
        )

    def optimize_target_volatility(
        self,
        target_volatility: float,
        constraints: OptimizationConstraints = None,
    ) -> OptimizationResult:
        """목표 변동성에서 최대 수익률"""
        if self.returns is None:
            self.fetch_data()

        constraints = constraints or OptimizationConstraints()
        n = len(self.assets)

        init_weights = np.array([1/n] * n)
        bounds = Bounds(constraints.min_weight, constraints.max_weight)

        def neg_return(w):
            return -self._portfolio_return(w)

        result = minimize(
            neg_return,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: target_volatility - self._portfolio_volatility(w)},
            ],
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self._portfolio_stats(weights)

        return OptimizationResult(
            weights={asset: float(w) for asset, w in zip(self.assets, weights)},
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_type=OptimizationType.TARGET_RISK,
            converged=result.success,
            iterations=result.nit,
        )

    # ========================================================================
    # Efficient Frontier
    # ========================================================================

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: OptimizationConstraints = None,
    ) -> List[EfficientFrontierPoint]:
        """효율적 프론티어 계산"""
        if self.returns is None:
            self.fetch_data()

        # 최소/최대 수익률 구간
        min_ret = float(self.mean_returns.min())
        max_ret = float(self.mean_returns.max())

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier = []

        for target in target_returns:
            try:
                result = self.optimize_target_return(target, constraints)
                if result.converged:
                    frontier.append(EfficientFrontierPoint(
                        return_target=target,
                        volatility=result.expected_volatility,
                        sharpe=result.sharpe_ratio,
                        weights=result.weights,
                    ))
            except Exception:
                continue

        return frontier

    # ========================================================================
    # Black-Litterman Model
    # ========================================================================

    def black_litterman(
        self,
        views: Dict[str, float],
        view_confidences: Dict[str, float] = None,
        tau: float = 0.05,
        market_weights: Dict[str, float] = None,
    ) -> BlackLittermanResult:
        """
        Black-Litterman 모델

        Args:
            views: 투자자 전망 {asset: expected_return}
            view_confidences: 전망 신뢰도 {asset: confidence 0-1}
            tau: 불확실성 스케일링 파라미터
            market_weights: 시장 균형 가중치
        """
        if self.returns is None:
            self.fetch_data()

        n = len(self.assets)

        # 시장 균형 가중치 (없으면 동일 가중)
        if market_weights:
            w_mkt = np.array([market_weights.get(a, 1/n) for a in self.assets])
        else:
            w_mkt = np.array([1/n] * n)
        w_mkt = w_mkt / w_mkt.sum()

        # 사전 기대수익률 (균형 수익률)
        # pi = lambda * Sigma * w_mkt
        risk_aversion = 2.5  # 위험회피계수
        pi = risk_aversion * np.dot(self.cov_matrix, w_mkt)
        prior_returns = {a: float(r) for a, r in zip(self.assets, pi)}

        # 전망 행렬 P와 Q
        view_assets = [a for a in views.keys() if a in self.assets]
        if not view_assets:
            # 전망이 없으면 사전 기대수익률 반환
            result = self.optimize_sharpe()
            return BlackLittermanResult(
                prior_returns=prior_returns,
                posterior_returns=prior_returns,
                optimal_weights=result.weights,
                views_incorporated=0,
                confidence_adjustment=1.0,
            )

        k = len(view_assets)
        P = np.zeros((k, n))
        Q = np.zeros(k)

        for i, asset in enumerate(view_assets):
            asset_idx = self.assets.index(asset)
            P[i, asset_idx] = 1
            Q[i] = views[asset]

        # 전망 불확실성 Omega
        if view_confidences:
            confidences = [view_confidences.get(a, 0.5) for a in view_assets]
        else:
            confidences = [0.5] * k

        # Omega = diag(1/confidence * P * tau * Sigma * P')
        omega_diag = []
        for i, conf in enumerate(confidences):
            var = float(np.dot(P[i], np.dot(tau * self.cov_matrix, P[i].T)))
            omega_diag.append(var / max(conf, 0.01))
        Omega = np.diag(omega_diag)

        # 사후 기대수익률
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        tau_sigma_inv = np.linalg.inv(tau * self.cov_matrix)
        omega_inv = np.linalg.inv(Omega)

        M = np.linalg.inv(tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P)))
        posterior = np.dot(M, np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q)))

        posterior_returns = {a: float(r) for a, r in zip(self.assets, posterior)}

        # 사후 기대수익률로 최적화
        self.mean_returns = pd.Series(posterior, index=self.assets)
        result = self.optimize_sharpe()

        return BlackLittermanResult(
            prior_returns=prior_returns,
            posterior_returns=posterior_returns,
            optimal_weights=result.weights,
            views_incorporated=len(view_assets),
            confidence_adjustment=np.mean(confidences),
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_asset_stats(self) -> pd.DataFrame:
        """개별 자산 통계"""
        if self.returns is None:
            self.fetch_data()

        stats = pd.DataFrame({
            'Annual Return': self.mean_returns,
            'Annual Volatility': self.returns.std() * np.sqrt(TRADING_DAYS),
            'Sharpe Ratio': self.mean_returns / (self.returns.std() * np.sqrt(TRADING_DAYS)),
        })
        return stats.round(4)

    def correlation_matrix(self) -> pd.DataFrame:
        """상관관계 행렬"""
        if self.returns is None:
            self.fetch_data()
        return self.returns.corr()

    def print_result(self, result: OptimizationResult):
        """결과 출력"""
        print("\n" + "=" * 50)
        print(f"Optimization: {result.optimization_type.value.upper()}")
        print("=" * 50)

        print(f"\nExpected Return: {result.expected_return:.2%}")
        print(f"Expected Volatility: {result.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Converged: {result.converged}")

        print("\nOptimal Weights:")
        for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
            if weight > 0.001:
                print(f"  {asset}: {weight:.1%}")

        print("=" * 50)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_optimize(
    assets: List[str] = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM'],
    method: str = 'sharpe',
) -> OptimizationResult:
    """빠른 최적화"""
    optimizer = PortfolioOptimizer(assets)
    optimizer.fetch_data()

    if method == 'sharpe':
        return optimizer.optimize_sharpe()
    elif method == 'min_var':
        return optimizer.optimize_min_variance()
    elif method == 'risk_parity':
        return optimizer.optimize_risk_parity()
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_optimizations(
    assets: List[str] = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM'],
) -> pd.DataFrame:
    """최적화 방법 비교"""
    optimizer = PortfolioOptimizer(assets)
    optimizer.fetch_data()

    results = {
        'Max Sharpe': optimizer.optimize_sharpe(),
        'Min Variance': optimizer.optimize_min_variance(),
        'Risk Parity': optimizer.optimize_risk_parity(),
    }

    comparison = []
    for name, result in results.items():
        comparison.append({
            'Method': name,
            'Return': result.expected_return,
            'Volatility': result.expected_volatility,
            'Sharpe': result.sharpe_ratio,
            **result.weights,
        })

    return pd.DataFrame(comparison).set_index('Method')


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Portfolio Optimizer Test")
    print("=" * 60)

    # 자산 설정
    assets = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'EFA']

    # 옵티마이저 생성
    optimizer = PortfolioOptimizer(assets)
    optimizer.fetch_data()

    # 개별 자산 통계
    print("\nAsset Statistics:")
    print(optimizer.get_asset_stats())

    # 샤프비율 최대화
    print("\n" + "-" * 40)
    result_sharpe = optimizer.optimize_sharpe()
    optimizer.print_result(result_sharpe)

    # 최소 분산
    print("\n" + "-" * 40)
    result_minvar = optimizer.optimize_min_variance()
    optimizer.print_result(result_minvar)

    # 리스크 패리티
    print("\n" + "-" * 40)
    result_rp = optimizer.optimize_risk_parity()
    optimizer.print_result(result_rp)

    # Black-Litterman
    print("\n" + "-" * 40)
    print("Black-Litterman Model")
    views = {
        'SPY': 0.12,  # SPY 12% 기대
        'TLT': 0.05,  # TLT 5% 기대
        'GLD': 0.08,  # GLD 8% 기대
    }
    confidences = {
        'SPY': 0.7,
        'TLT': 0.5,
        'GLD': 0.6,
    }
    bl_result = optimizer.black_litterman(views, confidences)
    print(f"\nViews incorporated: {bl_result.views_incorporated}")
    print(f"Confidence adjustment: {bl_result.confidence_adjustment:.0%}")
    print("\nPosterior Returns:")
    for asset, ret in sorted(bl_result.posterior_returns.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {ret:.2%}")
    print("\nOptimal Weights:")
    for asset, w in sorted(bl_result.optimal_weights.items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {asset}: {w:.1%}")

    # 최적화 비교
    print("\n" + "-" * 40)
    print("Optimization Comparison:")
    comparison = compare_optimizations(assets)
    print(comparison.round(3))
