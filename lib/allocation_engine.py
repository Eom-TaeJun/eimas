"""
Allocation Engine
=================
포트폴리오 비중 산출 엔진

지원 전략:
1. MVO (Mean-Variance Optimization) - Markowitz 최적화
2. Risk Parity - 동일 리스크 기여도 배분
3. HRP (Hierarchical Risk Parity) - 계층적 리스크 패리티
4. Equal Weight - 균등 배분
5. Inverse Volatility - 변동성 역수 비중

References:
- Markowitz (1952): "Portfolio Selection"
- Maillard, Roncalli, Teiletche (2010): "The Properties of Equally Weighted Risk Contribution Portfolios"
- Lopez de Prado (2016): "Building Diversified Portfolios that Outperform Out-of-Sample"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import warnings
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    from scipy.optimize import minimize, OptimizeResult
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. MVO will use analytical solutions only.")

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_CLUSTER_AVAILABLE = True
except ImportError:
    SCIPY_CLUSTER_AVAILABLE = False


class AllocationStrategy(Enum):
    """배분 전략 유형"""
    MVO_MAX_SHARPE = "mvo_max_sharpe"       # 샤프 비율 최대화
    MVO_MIN_VARIANCE = "mvo_min_variance"   # 최소 분산
    MVO_MAX_RETURN = "mvo_max_return"       # 수익률 최대화 (주어진 변동성)
    RISK_PARITY = "risk_parity"             # 동일 리스크 기여도
    HRP = "hrp"                             # 계층적 리스크 패리티
    EQUAL_WEIGHT = "equal_weight"           # 균등 배분
    INVERSE_VOLATILITY = "inverse_vol"      # 변동성 역수 비중
    BLACK_LITTERMAN = "black_litterman"     # Black-Litterman (views 필요)


@dataclass
class AllocationConstraints:
    """
    배분 제약 조건

    Attributes:
        min_weight: 최소 비중 (기본 0)
        max_weight: 최대 비중 (기본 1)
        sum_to_one: 비중 합 = 1 강제 (기본 True)
        long_only: 공매도 금지 (기본 True)
        max_turnover: 최대 회전율 (리밸런싱용)
        sector_limits: 섹터별 최대 비중 {섹터명: 최대비중}
        asset_limits: 자산별 비중 제한 {자산명: (min, max)}
    """
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_to_one: bool = True
    long_only: bool = True
    max_turnover: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    asset_limits: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class AllocationResult:
    """
    배분 결과

    Attributes:
        weights: 자산별 비중 {ticker: weight}
        strategy: 사용된 전략
        expected_return: 기대 수익률 (연환산)
        expected_volatility: 기대 변동성 (연환산)
        sharpe_ratio: 샤프 비율
        risk_contributions: 자산별 리스크 기여도
        diversification_ratio: 분산화 비율
        effective_n: 실효 자산 수 (1/sum(w^2))
        metadata: 추가 메타데이터
    """
    weights: Dict[str, float]
    strategy: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights)


class AllocationEngine:
    """
    포트폴리오 비중 산출 엔진

    MVO, Risk Parity, HRP 등 다양한 배분 전략 지원

    Example:
        >>> engine = AllocationEngine()
        >>> result = engine.allocate(
        ...     returns=returns_df,
        ...     strategy=AllocationStrategy.RISK_PARITY,
        ...     constraints=AllocationConstraints(max_weight=0.3)
        ... )
        >>> print(result.weights)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,  # 현재 Fed Funds Rate ~4.5%
        annualization_factor: int = 252  # 일간 데이터 가정
    ):
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def allocate(
        self,
        returns: pd.DataFrame,
        strategy: Union[AllocationStrategy, str] = AllocationStrategy.RISK_PARITY,
        constraints: Optional[AllocationConstraints] = None,
        **kwargs
    ) -> AllocationResult:
        """
        포트폴리오 비중 산출

        Args:
            returns: 자산별 수익률 DataFrame (일별/주별)
            strategy: 배분 전략
            constraints: 배분 제약 조건
            **kwargs: 전략별 추가 파라미터

        Returns:
            AllocationResult
        """
        if isinstance(strategy, str):
            strategy = AllocationStrategy(strategy)

        if constraints is None:
            constraints = AllocationConstraints()

        # 데이터 검증
        if returns.empty:
            raise ValueError("Empty returns DataFrame")
        if returns.isnull().all().any():
            logger.warning("Some assets have all NaN returns, dropping them")
            returns = returns.dropna(axis=1, how='all')

        # 공분산/상관행렬 계산
        cov_matrix = returns.cov() * self.annualization_factor
        mean_returns = returns.mean() * self.annualization_factor

        # 전략별 최적화 실행
        strategy_map = {
            AllocationStrategy.MVO_MAX_SHARPE: self._mvo_max_sharpe,
            AllocationStrategy.MVO_MIN_VARIANCE: self._mvo_min_variance,
            AllocationStrategy.MVO_MAX_RETURN: self._mvo_max_return,
            AllocationStrategy.RISK_PARITY: self._risk_parity,
            AllocationStrategy.HRP: self._hrp,
            AllocationStrategy.EQUAL_WEIGHT: self._equal_weight,
            AllocationStrategy.INVERSE_VOLATILITY: self._inverse_volatility,
            AllocationStrategy.BLACK_LITTERMAN: self._black_litterman,
        }

        optimizer = strategy_map.get(strategy)
        if optimizer is None:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 최적화 실행
        weights = optimizer(mean_returns, cov_matrix, constraints, **kwargs)

        # 제약 조건 적용 후 정규화
        weights = self._apply_constraints(weights, constraints)

        # 결과 계산
        result = self._compute_result(
            weights=weights,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            strategy=strategy.value
        )

        return result

    # =========================================================================
    # MVO (Mean-Variance Optimization)
    # =========================================================================

    def _mvo_max_sharpe(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """
        샤프 비율 최대화 (Tangency Portfolio)

        max (w'μ - rf) / sqrt(w'Σw)
        s.t. sum(w) = 1, w >= 0
        """
        n = len(mean_returns)
        assets = mean_returns.index.tolist()

        if not SCIPY_AVAILABLE:
            # Fallback: 분석적 해 (제약 없는 경우)
            return self._analytical_tangency(mean_returns, cov_matrix, constraints)

        # 초기값
        w0 = np.array([1.0/n] * n)

        # 목적 함수: 음의 샤프 비율 (최소화)
        def neg_sharpe(w):
            port_return = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol

        # 제약 조건
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # 경계 조건
        bounds = self._get_bounds(assets, constraints)

        # 최적화
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        if result.success:
            return dict(zip(assets, result.x))
        else:
            logger.warning(f"MVO optimization failed: {result.message}. Using equal weight.")
            return dict(zip(assets, w0))

    def _mvo_min_variance(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """
        최소 분산 포트폴리오 (Global Minimum Variance)

        min w'Σw
        s.t. sum(w) = 1, w >= 0
        """
        n = len(mean_returns)
        assets = mean_returns.index.tolist()

        if not SCIPY_AVAILABLE:
            # Fallback: 분석적 해
            return self._analytical_gmv(cov_matrix, constraints)

        w0 = np.array([1.0/n] * n)

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = self._get_bounds(assets, constraints)

        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        if result.success:
            return dict(zip(assets, result.x))
        else:
            logger.warning(f"Min variance optimization failed: {result.message}")
            return dict(zip(assets, w0))

    def _mvo_max_return(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        target_vol: float = 0.15,
        **kwargs
    ) -> Dict[str, float]:
        """
        목표 변동성에서 수익률 최대화

        max w'μ
        s.t. sqrt(w'Σw) <= target_vol, sum(w) = 1
        """
        n = len(mean_returns)
        assets = mean_returns.index.tolist()

        if not SCIPY_AVAILABLE:
            return self._equal_weight(mean_returns, cov_matrix, constraints)

        w0 = np.array([1.0/n] * n)

        def neg_return(w):
            return -np.dot(w, mean_returns)

        def vol_constraint(w):
            return target_vol**2 - np.dot(w.T, np.dot(cov_matrix, w))

        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': vol_constraint}
        ]
        bounds = self._get_bounds(assets, constraints)

        result = minimize(
            neg_return,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        if result.success:
            return dict(zip(assets, result.x))
        else:
            return dict(zip(assets, w0))

    def _analytical_tangency(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> Dict[str, float]:
        """분석적 접선 포트폴리오 (제약 없는 경우)"""
        try:
            excess_returns = mean_returns - self.risk_free_rate
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(mean_returns))

            w = np.dot(inv_cov, excess_returns)
            w = w / np.sum(w)

            # Long-only 제약
            if constraints.long_only:
                w = np.maximum(w, 0)
                w = w / np.sum(w)

            return dict(zip(mean_returns.index, w))
        except np.linalg.LinAlgError:
            return self._equal_weight(mean_returns, cov_matrix, constraints)

    def _analytical_gmv(
        self,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints
    ) -> Dict[str, float]:
        """분석적 최소 분산 포트폴리오"""
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(cov_matrix))

            w = np.dot(inv_cov, ones)
            w = w / np.sum(w)

            if constraints.long_only:
                w = np.maximum(w, 0)
                w = w / np.sum(w)

            return dict(zip(cov_matrix.index, w))
        except np.linalg.LinAlgError:
            n = len(cov_matrix)
            return dict(zip(cov_matrix.index, [1/n]*n))

    # =========================================================================
    # Risk Parity
    # =========================================================================

    def _risk_parity(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """
        Risk Parity (Equal Risk Contribution)

        각 자산의 리스크 기여도가 동일하도록 배분
        RC_i = w_i * (Σw)_i / σ_p = 1/n

        Reference: Maillard, Roncalli, Teiletche (2010)
        """
        n = len(mean_returns)
        assets = mean_returns.index.tolist()
        target_rc = 1.0 / n  # 동일 리스크 기여도

        if not SCIPY_AVAILABLE:
            # Fallback: Inverse Volatility (근사)
            return self._inverse_volatility(mean_returns, cov_matrix, constraints)

        # 초기값: 역분산 비중
        vols = np.sqrt(np.diag(cov_matrix))
        w0 = (1.0 / vols) / np.sum(1.0 / vols)

        def risk_contribution_error(w):
            """리스크 기여도 편차의 제곱합"""
            w = np.array(w)
            sigma = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            if sigma < 1e-10:
                return 1e10

            # Marginal Risk Contribution
            mrc = np.dot(cov_matrix, w) / sigma

            # Risk Contribution
            rc = w * mrc / sigma

            # 목표 RC와의 편차
            return np.sum((rc - target_rc) ** 2)

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = self._get_bounds(assets, constraints)

        result = minimize(
            risk_contribution_error,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        if result.success:
            return dict(zip(assets, result.x))
        else:
            logger.warning("Risk parity optimization failed, using inverse vol")
            return self._inverse_volatility(mean_returns, cov_matrix, constraints)

    # =========================================================================
    # HRP (Hierarchical Risk Parity)
    # =========================================================================

    def _hrp(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP)

        1. Tree Clustering: 상관계수 거리 기반 계층적 군집화
        2. Quasi-Diagonalization: 공분산 행렬 재정렬
        3. Recursive Bisection: 역분산 비중 배분

        Reference: Lopez de Prado (2016)
        """
        assets = mean_returns.index.tolist()
        n = len(assets)

        # 상관행렬 계산
        corr = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))
        corr = pd.DataFrame(corr, index=assets, columns=assets)

        # 거리 행렬: d = sqrt(2 * (1 - rho))
        dist = np.sqrt(2 * (1 - corr))
        dist = dist.fillna(0)

        # Quasi-Diagonalization
        if SCIPY_CLUSTER_AVAILABLE:
            condensed_dist = squareform(dist.values, checks=False)
            link = linkage(condensed_dist, method='ward')
            sorted_indices = self._get_linkage_order(link, n)
        else:
            sorted_indices = list(range(n))

        sorted_assets = [assets[i] for i in sorted_indices]

        # Recursive Bisection
        weights = self._recursive_bisection(
            cov_matrix.loc[sorted_assets, sorted_assets],
            sorted_assets
        )

        return weights

    def _get_linkage_order(self, link: np.ndarray, n: int) -> List[int]:
        """Linkage 매트릭스에서 정렬 순서 추출"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = n

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_assets: List[str]
    ) -> Dict[str, float]:
        """재귀적 이분법"""
        weights = {asset: 1.0 for asset in sorted_assets}
        clusters = [sorted_assets]

        while clusters:
            cluster = clusters.pop(0)

            if len(cluster) <= 1:
                continue

            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = self._get_cluster_var(cov, left)
            var_right = self._get_cluster_var(cov, right)

            alpha = 1 - var_left / (var_left + var_right + 1e-10)

            for asset in left:
                weights[asset] *= alpha
            for asset in right:
                weights[asset] *= (1 - alpha)

            if len(left) > 1:
                clusters.append(left)
            if len(right) > 1:
                clusters.append(right)

        return weights

    def _get_cluster_var(self, cov: pd.DataFrame, assets: List[str]) -> float:
        """클러스터의 역분산 가중 분산"""
        cov_slice = cov.loc[assets, assets]
        inv_diag = 1 / np.diag(cov_slice)
        parity_w = inv_diag / inv_diag.sum()
        return np.dot(parity_w, np.dot(cov_slice, parity_w))

    # =========================================================================
    # Simple Strategies
    # =========================================================================

    def _equal_weight(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """균등 배분 (1/N)"""
        n = len(mean_returns)
        return dict(zip(mean_returns.index, [1.0/n] * n))

    def _inverse_volatility(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        **kwargs
    ) -> Dict[str, float]:
        """변동성 역수 비중"""
        vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (vols + 1e-10)
        weights = inv_vols / np.sum(inv_vols)
        return dict(zip(mean_returns.index, weights))

    def _black_litterman(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: AllocationConstraints,
        views: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        **kwargs
    ) -> Dict[str, float]:
        """
        Black-Litterman Model

        Args:
            views: 투자자 뷰 {asset: expected_return}
            tau: 불확실성 스케일링 (기본 0.05)
        """
        if views is None:
            logger.warning("No views provided for Black-Litterman, using equilibrium returns")
            return self._mvo_max_sharpe(mean_returns, cov_matrix, constraints)

        # 간소화된 BL 구현
        # 실제로는 P, Q 행렬과 Omega 필요
        n = len(mean_returns)
        assets = mean_returns.index.tolist()

        # 균형 수익률 (CAPM)
        market_weights = np.array([1.0/n] * n)
        lambda_param = (mean_returns.mean() - self.risk_free_rate) / (
            np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        )
        pi = lambda_param * np.dot(cov_matrix, market_weights)

        # 뷰 반영 (간소화)
        adjusted_returns = pi.copy()
        for asset, view in views.items():
            if asset in assets:
                idx = assets.index(asset)
                adjusted_returns[idx] = (1 - tau) * pi[idx] + tau * view

        # MVO with adjusted returns
        mean_returns_adj = pd.Series(adjusted_returns, index=assets)
        return self._mvo_max_sharpe(mean_returns_adj, cov_matrix, constraints)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_bounds(
        self,
        assets: List[str],
        constraints: AllocationConstraints
    ) -> List[Tuple[float, float]]:
        """자산별 경계 조건 생성"""
        bounds = []
        for asset in assets:
            if constraints.asset_limits and asset in constraints.asset_limits:
                min_w, max_w = constraints.asset_limits[asset]
            else:
                min_w = constraints.min_weight if constraints.long_only else -1.0
                max_w = constraints.max_weight
            bounds.append((min_w, max_w))
        return bounds

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        constraints: AllocationConstraints
    ) -> Dict[str, float]:
        """제약 조건 적용"""
        w = np.array(list(weights.values()))
        assets = list(weights.keys())

        # Long-only
        if constraints.long_only:
            w = np.maximum(w, 0)

        # Min/Max weight
        w = np.clip(w, constraints.min_weight, constraints.max_weight)

        # 정규화
        if constraints.sum_to_one and np.sum(w) > 0:
            w = w / np.sum(w)

        return dict(zip(assets, w))

    def _compute_result(
        self,
        weights: Dict[str, float],
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        strategy: str
    ) -> AllocationResult:
        """최적화 결과 계산"""
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])

        # 기대 수익률 & 변동성
        exp_return = np.dot(w, mean_returns.loc[assets])
        exp_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.loc[assets, assets], w)))

        # 샤프 비율
        sharpe = (exp_return - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0

        # 리스크 기여도
        if exp_vol > 0:
            mrc = np.dot(cov_matrix.loc[assets, assets], w) / exp_vol
            rc = w * mrc / exp_vol
            risk_contributions = dict(zip(assets, rc))
        else:
            risk_contributions = dict(zip(assets, [0]*len(assets)))

        # 분산화 비율
        weighted_vols = w * np.sqrt(np.diag(cov_matrix.loc[assets, assets]))
        div_ratio = np.sum(weighted_vols) / exp_vol if exp_vol > 0 else 1.0

        # 실효 자산 수
        effective_n = 1.0 / np.sum(w**2) if np.sum(w**2) > 0 else 1.0

        return AllocationResult(
            weights=weights,
            strategy=strategy,
            expected_return=float(exp_return),
            expected_volatility=float(exp_vol),
            sharpe_ratio=float(sharpe),
            risk_contributions=risk_contributions,
            diversification_ratio=float(div_ratio),
            effective_n=float(effective_n)
        )

    def compare_strategies(
        self,
        returns: pd.DataFrame,
        strategies: Optional[List[AllocationStrategy]] = None,
        constraints: Optional[AllocationConstraints] = None
    ) -> pd.DataFrame:
        """
        여러 전략 비교

        Returns:
            DataFrame with strategy comparison metrics
        """
        if strategies is None:
            strategies = [
                AllocationStrategy.MVO_MAX_SHARPE,
                AllocationStrategy.MVO_MIN_VARIANCE,
                AllocationStrategy.RISK_PARITY,
                AllocationStrategy.HRP,
                AllocationStrategy.EQUAL_WEIGHT,
                AllocationStrategy.INVERSE_VOLATILITY,
            ]

        results = []
        for strategy in strategies:
            try:
                result = self.allocate(returns, strategy, constraints)
                results.append({
                    'strategy': strategy.value,
                    'expected_return': result.expected_return,
                    'expected_volatility': result.expected_volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'diversification_ratio': result.diversification_ratio,
                    'effective_n': result.effective_n,
                    'max_weight': max(result.weights.values()),
                    'min_weight': min(result.weights.values()),
                })
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")

        return pd.DataFrame(results)


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Allocation Engine Test")
    print("=" * 60)

    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252)

    # 상관된 자산 수익률 생성
    n_assets = 5
    assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']

    # 상관 구조 생성
    corr_matrix = np.array([
        [1.0, 0.8, -0.2, 0.1, 0.5],
        [0.8, 1.0, -0.3, 0.0, 0.4],
        [-0.2, -0.3, 1.0, 0.3, 0.0],
        [0.1, 0.0, 0.3, 1.0, 0.1],
        [0.5, 0.4, 0.0, 0.1, 1.0]
    ])
    vols = np.array([0.15, 0.20, 0.10, 0.12, 0.18]) / np.sqrt(252)

    # Cholesky 분해로 상관된 수익률 생성
    L = np.linalg.cholesky(corr_matrix)
    random_returns = np.random.randn(252, 5)
    returns_data = np.dot(random_returns, L.T) * vols

    returns_df = pd.DataFrame(returns_data, index=dates, columns=assets)

    print(f"Sample Data: {returns_df.shape}")
    print(f"Assets: {assets}")

    # 엔진 초기화
    engine = AllocationEngine(risk_free_rate=0.045)

    # 전략 비교
    print("\n[1] Strategy Comparison")
    comparison = engine.compare_strategies(returns_df)
    print(comparison.to_string())

    # Risk Parity 상세
    print("\n[2] Risk Parity Details")
    rp_result = engine.allocate(
        returns_df,
        AllocationStrategy.RISK_PARITY,
        AllocationConstraints(max_weight=0.4)
    )
    print(f"  Weights: {rp_result.weights}")
    print(f"  Expected Return: {rp_result.expected_return:.2%}")
    print(f"  Expected Vol: {rp_result.expected_volatility:.2%}")
    print(f"  Sharpe: {rp_result.sharpe_ratio:.2f}")
    print(f"  Risk Contributions: {rp_result.risk_contributions}")

    # MVO Max Sharpe
    print("\n[3] MVO Max Sharpe Details")
    mvo_result = engine.allocate(
        returns_df,
        AllocationStrategy.MVO_MAX_SHARPE
    )
    print(f"  Weights: {mvo_result.weights}")
    print(f"  Sharpe: {mvo_result.sharpe_ratio:.2f}")

    print("\nTest completed successfully!")
