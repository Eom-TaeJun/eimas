"""
Stress Testing Framework
=========================
포트폴리오 스트레스 테스트

테스트 유형:
1. Historical Scenario: 과거 위기 재현 (2008, 2020, etc.)
2. Hypothetical Scenario: 가상 시나리오 (금리 급등, 신용경색)
3. Factor Shock: 리스크 팩터 충격
4. Monte Carlo: 확률적 시뮬레이션
5. Extreme Scenario: Black Swan (꼬리 위험)

References:
- Basel III: Stress Testing Principles
- Breeden, Litt (2017): "Stress Testing in Non-Normal Markets"
- Rebonato (2010): "Coherent Stress Testing"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """스트레스 시나리오 정의"""
    name: str
    description: str
    asset_shocks: Dict[str, float]  # {asset: return shock}
    probability: Optional[float] = None  # 발생 확률
    duration_days: Optional[int] = None  # 지속 기간


@dataclass
class StressTestResult:
    """스트레스 테스트 결과"""
    scenario_name: str
    initial_value: float
    stressed_value: float
    loss: float
    loss_pct: float
    var_breach: bool  # VaR 위반 여부
    time_to_recover_days: Optional[int] = None

    # Component breakdown
    asset_contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'scenario_name': self.scenario_name,
            'initial_value': self.initial_value,
            'stressed_value': self.stressed_value,
            'loss': self.loss,
            'loss_pct': self.loss_pct,
            'var_breach': self.var_breach,
            'time_to_recover_days': self.time_to_recover_days,
            'asset_contributions': self.asset_contributions
        }


# ============================================================================
# HISTORICAL SCENARIOS (실제 역사적 사건)
# ============================================================================

HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        description="Sep-Nov 2008: Lehman Brothers collapse",
        asset_shocks={
            'SPY': -0.35,   # S&P 500
            'QQQ': -0.40,   # Nasdaq
            'IWM': -0.42,   # Russell 2000
            'TLT': +0.15,   # Long-term Treasuries (flight to safety)
            'GLD': +0.05,   # Gold
            'DBC': -0.50,   # Commodities
            'BTC-USD': 0.0  # N/A (didn't exist)
        },
        duration_days=60
    ),

    "2020_covid_crash": StressScenario(
        name="2020 COVID-19 Crash",
        description="Feb-Mar 2020: Pandemic panic",
        asset_shocks={
            'SPY': -0.34,
            'QQQ': -0.27,   # Tech more resilient
            'IWM': -0.42,
            'TLT': +0.20,
            'GLD': +0.03,
            'DBC': -0.30,
            'BTC-USD': -0.50  # Crypto sold off
        },
        duration_days=30
    ),

    "2022_rate_hike": StressScenario(
        name="2022 Rate Hike Cycle",
        description="2022: Fed aggressive tightening",
        asset_shocks={
            'SPY': -0.19,
            'QQQ': -0.33,   # Growth stocks hit hard
            'IWM': -0.21,
            'TLT': -0.30,   # Bonds crushed
            'GLD': -0.01,
            'DBC': +0.15,   # Commodities up (inflation)
            'BTC-USD': -0.65  # Crypto winter
        },
        duration_days=250
    ),

    "1987_black_monday": StressScenario(
        name="1987 Black Monday",
        description="Oct 19, 1987: Single day crash",
        asset_shocks={
            'SPY': -0.20,
            'QQQ': -0.20,
            'IWM': -0.20,
            'TLT': +0.10,
            'GLD': +0.02,
            'DBC': -0.10,
            'BTC-USD': 0.0
        },
        duration_days=1
    )
}


# ============================================================================
# HYPOTHETICAL SCENARIOS (가상 시나리오)
# ============================================================================

HYPOTHETICAL_SCENARIOS = {
    "sudden_rate_spike": StressScenario(
        name="Sudden Rate Spike",
        description="Fed raises rates 200bp unexpectedly",
        asset_shocks={
            'SPY': -0.25,
            'QQQ': -0.30,
            'IWM': -0.28,
            'TLT': -0.20,
            'GLD': +0.05,
            'DBC': -0.15,
            'BTC-USD': -0.40
        }
    ),

    "credit_freeze": StressScenario(
        name="Credit Market Freeze",
        description="Credit spreads widen 500bp",
        asset_shocks={
            'SPY': -0.30,
            'QQQ': -0.35,
            'IWM': -0.40,
            'TLT': +0.10,   # Flight to Treasuries
            'GLD': +0.15,
            'DBC': -0.25,
            'BTC-USD': -0.30
        }
    ),

    "crypto_collapse": StressScenario(
        name="Crypto Market Collapse",
        description="Major crypto exchange failure",
        asset_shocks={
            'SPY': -0.05,
            'QQQ': -0.08,
            'IWM': -0.05,
            'TLT': +0.02,
            'GLD': +0.01,
            'DBC': 0.0,
            'BTC-USD': -0.80,  # 80% crash
            'ETH-USD': -0.85
        }
    ),

    "stagflation": StressScenario(
        name="Stagflation",
        description="High inflation + Low growth",
        asset_shocks={
            'SPY': -0.15,
            'QQQ': -0.20,
            'IWM': -0.15,
            'TLT': -0.15,   # Bonds hurt by inflation
            'GLD': +0.25,   # Gold benefits
            'DBC': +0.30,   # Commodities surge
            'BTC-USD': -0.20  # Risk-off
        }
    )
}


class StressTestEngine:
    """
    스트레스 테스트 엔진

    주요 기능:
    1. Historical scenario replay
    2. Hypothetical scenario analysis
    3. Factor shock testing
    4. Monte Carlo simulation
    5. Tail risk analysis (CVaR, Expected Shortfall)
    """

    def __init__(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float = 1_000_000.0
    ):
        """
        Args:
            portfolio_weights: {ticker: weight}
            portfolio_value: 포트폴리오 평가액
        """
        self.portfolio_weights = portfolio_weights
        self.portfolio_value = portfolio_value

    def run_scenario(
        self,
        scenario: StressScenario,
        var_threshold: float = 0.05  # 95% VaR
    ) -> StressTestResult:
        """
        단일 시나리오 스트레스 테스트

        Args:
            scenario: StressScenario 객체
            var_threshold: VaR 임계값 (기본 5% = 95% VaR)

        Returns:
            StressTestResult
        """
        # Compute stressed value
        portfolio_return = 0.0
        asset_contributions = {}

        for ticker, weight in self.portfolio_weights.items():
            shock = scenario.asset_shocks.get(ticker, 0.0)
            contribution = weight * shock
            portfolio_return += contribution
            asset_contributions[ticker] = contribution * self.portfolio_value

        stressed_value = self.portfolio_value * (1 + portfolio_return)
        loss = self.portfolio_value - stressed_value
        loss_pct = -portfolio_return

        # Check VaR breach
        var_breach = loss_pct > var_threshold

        result = StressTestResult(
            scenario_name=scenario.name,
            initial_value=self.portfolio_value,
            stressed_value=stressed_value,
            loss=loss,
            loss_pct=loss_pct,
            var_breach=var_breach,
            asset_contributions=asset_contributions
        )

        logger.info(f"Stress test '{scenario.name}': Loss {loss_pct*100:.2f}%, "
                   f"VaR breach: {var_breach}")

        return result

    def run_all_historical(self) -> List[StressTestResult]:
        """모든 역사적 시나리오 실행"""
        results = []

        for scenario_name, scenario in HISTORICAL_SCENARIOS.items():
            result = self.run_scenario(scenario)
            results.append(result)

        return results

    def run_all_hypothetical(self) -> List[StressTestResult]:
        """모든 가상 시나리오 실행"""
        results = []

        for scenario_name, scenario in HYPOTHETICAL_SCENARIOS.items():
            result = self.run_scenario(scenario)
            results.append(result)

        return results

    def factor_shock(
        self,
        factor_shocks: Dict[str, float],
        factor_exposures: Dict[str, Dict[str, float]]
    ) -> StressTestResult:
        """
        팩터 충격 테스트

        Args:
            factor_shocks: {factor: shock}, e.g., {'equity': -0.2, 'bond': -0.1}
            factor_exposures: {ticker: {factor: exposure}}

        Returns:
            StressTestResult
        """
        # Compute asset-level shocks from factor shocks
        asset_shocks = {}

        for ticker in self.portfolio_weights:
            exposures = factor_exposures.get(ticker, {})
            shock = sum(exposures.get(factor, 0) * factor_shocks.get(factor, 0)
                       for factor in factor_shocks)
            asset_shocks[ticker] = shock

        # Create scenario
        scenario = StressScenario(
            name="Factor Shock",
            description=f"Factors: {factor_shocks}",
            asset_shocks=asset_shocks
        )

        return self.run_scenario(scenario)

    def monte_carlo(
        self,
        returns_mean: Dict[str, float],
        returns_cov: pd.DataFrame,
        n_simulations: int = 10_000,
        horizon_days: int = 1,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Monte Carlo 시뮬레이션

        Args:
            returns_mean: {ticker: expected_daily_return}
            returns_cov: Covariance matrix (DataFrame)
            n_simulations: 시뮬레이션 횟수
            horizon_days: 기간 (일)
            confidence_level: 신뢰수준 (기본 95%)

        Returns:
            {
                'mean': 평균 포트폴리오 가치,
                'std': 표준편차,
                'var': VaR,
                'cvar': CVaR (Expected Shortfall),
                'distribution': 시뮬레이션 결과 분포
            }
        """
        # Align tickers
        tickers = list(self.portfolio_weights.keys())
        mean_vector = np.array([returns_mean.get(t, 0) for t in tickers])
        cov_matrix = returns_cov.loc[tickers, tickers].values

        # Simulate returns (multivariate normal)
        simulated_returns = np.random.multivariate_normal(
            mean_vector * horizon_days,
            cov_matrix * horizon_days,
            size=n_simulations
        )

        # Portfolio returns
        weights_array = np.array([self.portfolio_weights[t] for t in tickers])
        portfolio_returns = simulated_returns @ weights_array

        # Portfolio values
        portfolio_values = self.portfolio_value * (1 + portfolio_returns)

        # Statistics
        mean_value = portfolio_values.mean()
        std_value = portfolio_values.std()

        # VaR & CVaR
        losses = self.portfolio_value - portfolio_values
        var_quantile = np.quantile(losses, confidence_level)
        cvar = losses[losses >= var_quantile].mean()

        result = {
            'mean': mean_value,
            'std': std_value,
            'var': var_quantile,
            'cvar': cvar,
            'var_pct': var_quantile / self.portfolio_value,
            'cvar_pct': cvar / self.portfolio_value,
            'distribution': portfolio_values
        }

        logger.info(f"Monte Carlo ({n_simulations} sims): "
                   f"VaR({confidence_level*100:.0f}%) = ${var_quantile:,.0f} ({result['var_pct']*100:.2f}%), "
                   f"CVaR = ${cvar:,.0f} ({result['cvar_pct']*100:.2f}%)")

        return result

    def extreme_scenario(
        self,
        severity: str = "severe"  # moderate, severe, extreme
    ) -> StressTestResult:
        """
        극한 시나리오 (Black Swan)

        Args:
            severity: 심각도 (moderate, severe, extreme)

        Returns:
            StressTestResult
        """
        # Define severity levels
        severity_shocks = {
            'moderate': {
                'SPY': -0.30, 'QQQ': -0.35, 'IWM': -0.40,
                'TLT': +0.15, 'GLD': +0.10, 'DBC': -0.20,
                'BTC-USD': -0.50
            },
            'severe': {
                'SPY': -0.50, 'QQQ': -0.55, 'IWM': -0.60,
                'TLT': +0.20, 'GLD': +0.20, 'DBC': -0.40,
                'BTC-USD': -0.70
            },
            'extreme': {
                'SPY': -0.70, 'QQQ': -0.75, 'IWM': -0.80,
                'TLT': +0.25, 'GLD': +0.30, 'DBC': -0.60,
                'BTC-USD': -0.90
            }
        }

        shocks = severity_shocks.get(severity, severity_shocks['severe'])

        scenario = StressScenario(
            name=f"Extreme Scenario ({severity.upper()})",
            description="Black Swan event",
            asset_shocks=shocks
        )

        return self.run_scenario(scenario)


def generate_stress_test_report(
    results: List[StressTestResult]
) -> str:
    """스트레스 테스트 리포트 생성"""

    report = f"""
# Stress Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
Total Scenarios Tested: {len(results)}

## Results

| Scenario | Loss | Loss % | VaR Breach |
|----------|------|--------|------------|
"""

    for result in results:
        breach_icon = "⚠️" if result.var_breach else "✅"
        report += f"| {result.scenario_name} | ${result.loss:,.0f} | {result.loss_pct*100:.2f}% | {breach_icon} |\n"

    # Worst case
    worst = max(results, key=lambda r: r.loss_pct)
    report += f"""
## Worst Case Scenario
- **Scenario:** {worst.scenario_name}
- **Loss:** ${worst.loss:,.0f} ({worst.loss_pct*100:.2f}%)
- **Stressed Value:** ${worst.stressed_value:,.0f}

### Asset Contributions (Worst Case)
"""

    for ticker, contribution in sorted(worst.asset_contributions.items(),
                                      key=lambda x: x[1]):
        report += f"- {ticker}: ${contribution:,.0f}\n"

    return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example portfolio
    portfolio_weights = {
        'SPY': 0.40,
        'QQQ': 0.20,
        'TLT': 0.25,
        'GLD': 0.10,
        'BTC-USD': 0.05
    }

    # Stress test engine
    engine = StressTestEngine(portfolio_weights, portfolio_value=1_000_000)

    print("=== Historical Scenarios ===")
    historical_results = engine.run_all_historical()
    for result in historical_results:
        print(f"{result.scenario_name}: "
              f"Loss ${result.loss:,.0f} ({result.loss_pct*100:.2f}%), "
              f"VaR Breach: {result.var_breach}")

    print("\n=== Hypothetical Scenarios ===")
    hypothetical_results = engine.run_all_hypothetical()
    for result in hypothetical_results:
        print(f"{result.scenario_name}: "
              f"Loss ${result.loss:,.0f} ({result.loss_pct*100:.2f}%), "
              f"VaR Breach: {result.var_breach}")

    print("\n=== Extreme Scenario ===")
    extreme_result = engine.extreme_scenario("severe")
    print(f"Extreme (Severe): "
          f"Loss ${extreme_result.loss:,.0f} ({extreme_result.loss_pct*100:.2f}%)")

    # Generate report
    all_results = historical_results + hypothetical_results + [extreme_result]
    report = generate_stress_test_report(all_results)
    print(report)
