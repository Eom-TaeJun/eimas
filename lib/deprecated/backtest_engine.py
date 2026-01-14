#!/usr/bin/env python3
"""
Walk-Forward Backtesting Engine (Elicit Report Enhancement)
============================================================
HRP 및 포트폴리오 최적화 전략의 Out-of-Sample 검증 모듈

학술적 근거:
- Lopez de Prado (2016): HRP out-of-sample outperformance 주장
- Bailey et al. (2014): Walk-forward validation 중요성
- Perplexity 검증: 1/N 벤치마크 대비 혼합 결과 (로버스트니스 필요)

주요 기능:
1. Walk-Forward 검증 프레임워크
2. 다중 벤치마크 비교 (1/N, Mean-Variance, Risk Parity)
3. 성과 지표 계산 (Sharpe, Sortino, Max Drawdown, Calmar)
4. 통계적 유의성 테스트
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings('ignore')

# scipy 통계
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some features will be limited.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PerformanceMetrics:
    """포트폴리오 성과 지표"""
    total_return: float              # 총 수익률
    annualized_return: float         # 연율화 수익률
    annualized_volatility: float     # 연율화 변동성
    sharpe_ratio: float              # Sharpe Ratio (RF=0 가정)
    sortino_ratio: float             # Sortino Ratio (하방 위험만 고려)
    max_drawdown: float              # 최대 낙폭
    calmar_ratio: float              # Calmar Ratio (연율 수익/MDD)
    win_rate: float                  # 양수 수익률 비율
    avg_win: float                   # 평균 수익 (양수 기간)
    avg_loss: float                  # 평균 손실 (음수 기간)
    profit_factor: float             # 총 이익 / 총 손실
    var_95: float                    # 95% VaR
    cvar_95: float                   # 95% CVaR (Expected Shortfall)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WalkForwardResult:
    """Walk-Forward 검증 결과"""
    strategy_name: str
    train_periods: int
    test_periods: int
    total_periods: int

    # 성과 지표
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics

    # 기간별 결과
    period_returns: List[float]
    period_weights: List[Dict[str, float]]

    # 통계적 유의성
    oos_sharpe_tstat: float
    oos_sharpe_pvalue: float
    is_significant: bool

    # 벤치마크 대비
    excess_return_vs_benchmark: float
    information_ratio: float

    # 메타데이터
    methodology_notes: str
    academic_references: List[str]


@dataclass
class BacktestComparison:
    """전략 비교 결과"""
    timestamp: str
    strategies: Dict[str, WalkForwardResult]
    best_strategy: str
    best_sharpe: float
    ranking: List[Tuple[str, float]]  # (strategy_name, sharpe)

    # 통계적 비교
    sharpe_diff_significant: bool
    anova_pvalue: Optional[float]

    interpretation: str


# ============================================================================
# Performance Calculator
# ============================================================================

class PerformanceCalculator:
    """포트폴리오 성과 지표 계산기"""

    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """
        포트폴리오 수익률 시계열에서 성과 지표 계산

        Parameters:
        -----------
        returns : Series
            일별 수익률 시계열

        Returns:
        --------
        PerformanceMetrics
        """
        if len(returns) == 0:
            return self._empty_metrics()

        returns = returns.dropna()
        if len(returns) < 2:
            return self._empty_metrics()

        # 기본 통계
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        ann_factor = self.trading_days / n_periods if n_periods > 0 else 1

        annualized_return = (1 + total_return) ** ann_factor - 1
        annualized_vol = returns.std() * np.sqrt(self.trading_days)

        # Sharpe Ratio
        if annualized_vol != 0:
            sharpe = (annualized_return - self.risk_free_rate) / annualized_vol
        else:
            sharpe = 0.0

        # Sortino Ratio (하방 위험만)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(self.trading_days)
            if downside_vol != 0:
                sortino = (annualized_return - self.risk_free_rate) / downside_vol
            else:
                sortino = 0.0
        else:
            sortino = sharpe * 2  # 하방 없으면 Sharpe 2배로 근사

        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        if max_drawdown != 0:
            calmar = annualized_return / abs(max_drawdown)
        else:
            calmar = 0.0

        # Win/Loss 통계
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        # Profit Factor
        total_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
        total_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0001
        profit_factor = total_profit / total_loss

        # VaR & CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """빈 결과 반환"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, annualized_volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            var_95=0.0, cvar_95=0.0
        )


# ============================================================================
# Portfolio Strategies (Benchmarks)
# ============================================================================

class PortfolioBenchmarks:
    """벤치마크 전략 모음"""

    @staticmethod
    def equal_weight(returns_df: pd.DataFrame) -> np.ndarray:
        """1/N 동일가중 전략"""
        n_assets = returns_df.shape[1]
        return np.ones(n_assets) / n_assets

    @staticmethod
    def inverse_volatility(returns_df: pd.DataFrame) -> np.ndarray:
        """역변동성 가중 전략"""
        vols = returns_df.std()
        inv_vols = 1 / (vols + 1e-10)
        return (inv_vols / inv_vols.sum()).values

    @staticmethod
    def risk_parity(returns_df: pd.DataFrame) -> np.ndarray:
        """리스크 패리티 전략 (균등 위험 기여도)"""
        cov = returns_df.cov()
        n_assets = len(cov)

        # 초기값: 동일가중
        x0 = np.ones(n_assets) / n_assets

        def risk_budget_objective(weights):
            """리스크 기여도 균등화 목적함수"""
            port_vol = np.sqrt(weights @ cov @ weights)
            if port_vol == 0:
                return 1e10
            marginal_contrib = cov @ weights
            risk_contrib = weights * marginal_contrib / port_vol
            target_risk = port_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]

        if SCIPY_AVAILABLE:
            result = minimize(risk_budget_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else x0
        return x0

    @staticmethod
    def min_variance(returns_df: pd.DataFrame) -> np.ndarray:
        """최소분산 전략"""
        cov = returns_df.cov().values
        n_assets = len(cov)

        def portfolio_variance(weights):
            return weights @ cov @ weights

        x0 = np.ones(n_assets) / n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]

        if SCIPY_AVAILABLE:
            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else x0
        return x0


# ============================================================================
# Walk-Forward Engine
# ============================================================================

class WalkForwardEngine:
    """
    Walk-Forward 백테스팅 엔진

    Lopez de Prado (2016) 방법론 기반:
    - 훈련 기간: 과거 N일 데이터로 포트폴리오 최적화
    - 테스트 기간: 향후 M일 동안 out-of-sample 성과 측정
    - 롤링 방식: step일씩 이동하며 반복

    Perplexity 검증 결과:
    - HRP가 항상 우월하지 않음 (2025 연구에서 1/N이 우세한 경우 존재)
    - 다중 벤치마크 비교 필수
    - 통계적 유의성 테스트 권장
    """

    def __init__(
        self,
        train_window: int = 252,    # 훈련 기간 (1년)
        test_window: int = 21,      # 테스트 기간 (1개월)
        step: int = 21,             # 스텝 크기 (1개월)
        min_train_samples: int = 100
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.min_train_samples = min_train_samples
        self.perf_calc = PerformanceCalculator()

    def run_walk_forward(
        self,
        returns_df: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], np.ndarray],
        strategy_name: str = "Custom"
    ) -> WalkForwardResult:
        """
        Walk-Forward 검증 실행

        Parameters:
        -----------
        returns_df : DataFrame
            자산별 수익률 (columns=자산, index=날짜)
        strategy_func : Callable
            가중치 산출 함수: (train_returns) -> weights
        strategy_name : str
            전략 이름

        Returns:
        --------
        WalkForwardResult
        """
        n_periods = len(returns_df)
        total_steps = 0

        # 결과 저장
        is_returns = []    # In-sample 수익률
        oos_returns = []   # Out-of-sample 수익률
        period_weights = []

        # Walk-Forward 루프
        start_idx = self.train_window
        while start_idx + self.test_window <= n_periods:
            # 훈련 데이터
            train_start = start_idx - self.train_window
            train_end = start_idx
            train_data = returns_df.iloc[train_start:train_end]

            # 테스트 데이터
            test_start = start_idx
            test_end = min(start_idx + self.test_window, n_periods)
            test_data = returns_df.iloc[test_start:test_end]

            # 가중치 계산 (훈련 데이터 기반)
            try:
                weights = strategy_func(train_data)
                weights = np.array(weights)
                # 음수 가중치 제거, 정규화
                weights = np.maximum(weights, 0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(weights)) / len(weights)
            except Exception as e:
                # 실패 시 동일가중
                weights = np.ones(train_data.shape[1]) / train_data.shape[1]

            # 가중치 저장
            weight_dict = {col: w for col, w in zip(returns_df.columns, weights)}
            period_weights.append(weight_dict)

            # In-sample 포트폴리오 수익률
            is_port_ret = (train_data @ weights).values
            is_returns.extend(is_port_ret)

            # Out-of-sample 포트폴리오 수익률
            oos_port_ret = (test_data @ weights).values
            oos_returns.extend(oos_port_ret)

            # 다음 스텝
            start_idx += self.step
            total_steps += 1

        if total_steps == 0:
            raise ValueError("데이터 부족: Walk-Forward 실행 불가")

        # 성과 지표 계산
        is_metrics = self.perf_calc.calculate_metrics(pd.Series(is_returns))
        oos_metrics = self.perf_calc.calculate_metrics(pd.Series(oos_returns))

        # 통계적 유의성 (Sharpe > 0 테스트)
        oos_returns_arr = np.array(oos_returns)
        if len(oos_returns_arr) > 30 and oos_returns_arr.std() > 0:
            tstat = (oos_returns_arr.mean() / oos_returns_arr.std()) * np.sqrt(len(oos_returns_arr))
            pvalue = 1 - stats.t.cdf(tstat, len(oos_returns_arr) - 1) if SCIPY_AVAILABLE else 0.5
        else:
            tstat, pvalue = 0.0, 1.0

        is_significant = pvalue < 0.05 and oos_metrics.sharpe_ratio > 0

        return WalkForwardResult(
            strategy_name=strategy_name,
            train_periods=total_steps * self.train_window,
            test_periods=total_steps * self.test_window,
            total_periods=n_periods,
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            period_returns=oos_returns,
            period_weights=period_weights,
            oos_sharpe_tstat=tstat,
            oos_sharpe_pvalue=pvalue,
            is_significant=is_significant,
            excess_return_vs_benchmark=0.0,  # 나중에 계산
            information_ratio=0.0,           # 나중에 계산
            methodology_notes=f"Train: {self.train_window}d, Test: {self.test_window}d, Step: {self.step}d",
            academic_references=[
                "Lopez de Prado (2016): Building Diversified Portfolios that Outperform",
                "Bailey et al. (2014): Pseudo-Mathematics and Financial Charlatanism"
            ]
        )

    def compare_strategies(
        self,
        returns_df: pd.DataFrame,
        hrp_func: Optional[Callable] = None,
        include_benchmarks: List[str] = None
    ) -> BacktestComparison:
        """
        HRP vs 벤치마크 전략 비교

        Parameters:
        -----------
        returns_df : DataFrame
            자산별 수익률
        hrp_func : Callable (optional)
            HRP 가중치 함수 (없으면 스킵)
        include_benchmarks : List[str]
            비교할 벤치마크 목록 ['equal_weight', 'inverse_vol', 'risk_parity', 'min_variance']

        Returns:
        --------
        BacktestComparison
        """
        if include_benchmarks is None:
            include_benchmarks = ['equal_weight', 'inverse_vol', 'risk_parity']

        strategies = {}

        # 벤치마크 전략들
        benchmark_funcs = {
            'equal_weight': ('1/N Equal Weight', PortfolioBenchmarks.equal_weight),
            'inverse_vol': ('Inverse Volatility', PortfolioBenchmarks.inverse_volatility),
            'risk_parity': ('Risk Parity', PortfolioBenchmarks.risk_parity),
            'min_variance': ('Minimum Variance', PortfolioBenchmarks.min_variance)
        }

        for bench_key in include_benchmarks:
            if bench_key in benchmark_funcs:
                name, func = benchmark_funcs[bench_key]
                try:
                    result = self.run_walk_forward(returns_df, func, name)
                    strategies[name] = result
                except Exception as e:
                    print(f"Warning: {name} failed: {e}")

        # HRP 전략
        if hrp_func is not None:
            try:
                hrp_result = self.run_walk_forward(returns_df, hrp_func, "HRP")
                strategies["HRP"] = hrp_result
            except Exception as e:
                print(f"Warning: HRP failed: {e}")

        if not strategies:
            raise ValueError("모든 전략 실행 실패")

        # 순위 결정 (OOS Sharpe 기준)
        ranking = sorted(
            [(name, r.out_of_sample_metrics.sharpe_ratio) for name, r in strategies.items()],
            key=lambda x: x[1],
            reverse=True
        )

        best_strategy, best_sharpe = ranking[0]

        # 벤치마크 대비 초과 수익 계산 (1/N 기준)
        if '1/N Equal Weight' in strategies:
            benchmark_result = strategies['1/N Equal Weight']
            for name, result in strategies.items():
                if name != '1/N Equal Weight':
                    excess = (
                        result.out_of_sample_metrics.annualized_return -
                        benchmark_result.out_of_sample_metrics.annualized_return
                    )
                    # Information Ratio 계산
                    excess_returns = np.array(result.period_returns) - np.array(benchmark_result.period_returns)
                    if len(excess_returns) > 0 and np.std(excess_returns) > 0:
                        ir = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 / self.test_window)
                    else:
                        ir = 0.0

                    # 결과 업데이트
                    result.excess_return_vs_benchmark = excess
                    result.information_ratio = ir

        # ANOVA 테스트 (전략 간 유의한 차이 여부)
        if len(strategies) >= 3 and SCIPY_AVAILABLE:
            returns_lists = [r.period_returns for r in strategies.values()]
            # 길이 맞추기
            min_len = min(len(r) for r in returns_lists)
            returns_lists = [r[:min_len] for r in returns_lists]
            try:
                f_stat, anova_p = stats.f_oneway(*returns_lists)
            except:
                anova_p = None
        else:
            anova_p = None

        sharpe_diff_significant = anova_p is not None and anova_p < 0.05

        # 해석 생성
        interpretation = self._generate_interpretation(
            strategies, best_strategy, best_sharpe, sharpe_diff_significant
        )

        return BacktestComparison(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            strategies=strategies,
            best_strategy=best_strategy,
            best_sharpe=best_sharpe,
            ranking=ranking,
            sharpe_diff_significant=sharpe_diff_significant,
            anova_pvalue=anova_p,
            interpretation=interpretation
        )

    def _generate_interpretation(
        self,
        strategies: Dict,
        best: str,
        best_sharpe: float,
        significant: bool
    ) -> str:
        """비교 결과 해석 생성"""
        lines = []

        lines.append(f"최우수 전략: {best} (OOS Sharpe: {best_sharpe:.3f})")

        if significant:
            lines.append("전략 간 성과 차이가 통계적으로 유의함 (p < 0.05)")
        else:
            lines.append("전략 간 성과 차이가 통계적으로 유의하지 않음")

        # HRP vs 1/N 비교
        if 'HRP' in strategies and '1/N Equal Weight' in strategies:
            hrp = strategies['HRP']
            equal = strategies['1/N Equal Weight']

            hrp_sharpe = hrp.out_of_sample_metrics.sharpe_ratio
            equal_sharpe = equal.out_of_sample_metrics.sharpe_ratio

            if hrp_sharpe > equal_sharpe:
                lines.append(
                    f"HRP가 1/N 대비 우월 (Sharpe: {hrp_sharpe:.3f} vs {equal_sharpe:.3f})"
                )
            else:
                lines.append(
                    f"주의: 1/N이 HRP 대비 우월 (Sharpe: {equal_sharpe:.3f} vs {hrp_sharpe:.3f})"
                    " - Perplexity 검증 결과와 일치"
                )

        return " | ".join(lines)


# ============================================================================
# HRP Integration Helper
# ============================================================================

def create_hrp_strategy_func(hrp_portfolio_instance) -> Callable:
    """
    HRP 포트폴리오 인스턴스를 Walk-Forward 호환 함수로 변환

    Parameters:
    -----------
    hrp_portfolio_instance : GraphClusteredPortfolio
        lib/graph_clustered_portfolio.py의 인스턴스

    Returns:
    --------
    Callable: (returns_df) -> weights
    """
    def hrp_func(train_returns: pd.DataFrame) -> np.ndarray:
        """HRP 가중치 계산 래퍼"""
        try:
            # HRP 인스턴스의 optimize 메서드 호출
            hrp_portfolio_instance.build_from_returns(train_returns)
            weights_dict = hrp_portfolio_instance.optimize()

            # DataFrame 컬럼 순서에 맞게 가중치 배열 생성
            weights = np.array([
                weights_dict.get(col, 0.0) for col in train_returns.columns
            ])
            return weights
        except Exception as e:
            # 실패 시 동일가중
            print(f"HRP Warning: {e}")
            return np.ones(train_returns.shape[1]) / train_returns.shape[1]

    return hrp_func


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Walk-Forward Backtesting Engine Test")
    print("=" * 60)

    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_days = 504  # 2년
    n_assets = 10

    # 상관관계 있는 수익률 생성
    base_return = np.random.randn(n_days) * 0.01
    returns_data = {}
    for i in range(n_assets):
        asset_specific = np.random.randn(n_days) * 0.015
        returns_data[f'Asset_{i}'] = base_return * 0.5 + asset_specific

    returns_df = pd.DataFrame(
        returns_data,
        index=pd.date_range('2024-01-01', periods=n_days, freq='D')
    )

    print(f"\n[Simulated Data]")
    print(f"  Shape: {returns_df.shape}")
    print(f"  Period: {returns_df.index[0]} ~ {returns_df.index[-1]}")

    # Walk-Forward 엔진 생성
    engine = WalkForwardEngine(
        train_window=252,
        test_window=21,
        step=21
    )

    # 전략 비교
    print(f"\n[Running Walk-Forward Comparison...]")
    comparison = engine.compare_strategies(
        returns_df,
        hrp_func=None,  # HRP 없이 벤치마크만
        include_benchmarks=['equal_weight', 'inverse_vol', 'risk_parity']
    )

    print(f"\n[Results]")
    print(f"  Best Strategy: {comparison.best_strategy}")
    print(f"  Best OOS Sharpe: {comparison.best_sharpe:.3f}")
    print(f"  ANOVA p-value: {comparison.anova_pvalue:.4f}" if comparison.anova_pvalue else "  ANOVA: N/A")
    print(f"  Significant Difference: {comparison.sharpe_diff_significant}")

    print(f"\n[Ranking]")
    for rank, (name, sharpe) in enumerate(comparison.ranking, 1):
        result = comparison.strategies[name]
        oos = result.out_of_sample_metrics
        print(f"  {rank}. {name}")
        print(f"     OOS Sharpe: {oos.sharpe_ratio:.3f}, Return: {oos.annualized_return*100:.1f}%")
        print(f"     Max DD: {oos.max_drawdown*100:.1f}%, Significant: {result.is_significant}")

    print(f"\n[Interpretation]")
    print(f"  {comparison.interpretation}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
