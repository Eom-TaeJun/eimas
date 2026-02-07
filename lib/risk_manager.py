#!/usr/bin/env python3
"""
EIMAS Risk Manager
==================
포지션 사이징, 리스크 관리, 포트폴리오 리밸런싱

주요 기능:
1. Position Sizing (Kelly, Fixed Fractional, Risk Parity)
2. Portfolio Risk 계산 (VaR, CVaR, Max Drawdown)
3. Rebalancing Triggers (Drift, Calendar, Signal)
4. Risk Limits & Alerts

Usage:
    from lib.risk_manager import RiskManager, PositionSizer

    rm = RiskManager()
    risk = rm.calculate_portfolio_risk(holdings)

    sizer = PositionSizer()
    size = sizer.kelly_criterion(win_rate, avg_win, avg_loss)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Constants & Enums
# ============================================================================

class RiskLevel(str, Enum):
    """리스크 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class RebalanceTrigger(str, Enum):
    """리밸런싱 트리거"""
    DRIFT = "drift"           # 비중 이탈
    CALENDAR = "calendar"     # 정기 리밸런싱
    SIGNAL = "signal"         # 시그널 기반
    RISK = "risk"             # 리스크 한도 초과
    NONE = "none"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PortfolioRisk:
    """포트폴리오 리스크 지표"""
    total_value: float
    daily_vol: float          # 일간 변동성
    annual_vol: float         # 연환산 변동성
    var_95: float             # 95% VaR (일간)
    var_99: float             # 99% VaR (일간)
    cvar_95: float            # 95% CVaR (Expected Shortfall)
    max_drawdown: float       # 최대 낙폭
    sharpe_estimate: float    # 추정 샤프 비율
    risk_level: RiskLevel
    beta: float = 1.0         # 시장 베타
    correlation_to_spy: float = 0.0


@dataclass
class PositionLimit:
    """포지션 한도"""
    ticker: str
    max_weight: float         # 최대 비중
    max_dollar: float         # 최대 금액
    current_weight: float
    current_dollar: float
    is_exceeded: bool
    excess_amount: float = 0.0


@dataclass
class RebalanceRecommendation:
    """리밸런싱 권고"""
    trigger: RebalanceTrigger
    urgency: RiskLevel
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trades_needed: Dict[str, float]  # 양수=매수, 음수=매도
    estimated_cost: float
    reasoning: str


# ============================================================================
# Position Sizer
# ============================================================================

class PositionSizer:
    """포지션 사이징 계산기"""

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5  # Half Kelly (보수적)
    ) -> float:
        """
        켈리 기준
        f* = (p * b - q) / b
        p = 승률, q = 1-p, b = 평균이익/평균손실
        """
        if avg_loss == 0:
            return 0.0

        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win / avg_loss)

        kelly = (p * b - q) / b

        # 음수면 베팅하지 않음
        if kelly <= 0:
            return 0.0

        # Fraction Kelly (보통 0.25 ~ 0.5 사용)
        return min(kelly * fraction, 1.0)

    def fixed_fractional(
        self,
        capital: float,
        risk_per_trade: float,  # 거래당 리스크 (예: 0.02 = 2%)
        stop_loss_pct: float,   # 스탑로스 %
        price: float
    ) -> Tuple[float, int]:
        """
        고정 비율 사이징
        Returns: (position_value, shares)
        """
        risk_amount = capital * risk_per_trade
        position_value = risk_amount / stop_loss_pct
        shares = int(position_value / price)

        return position_value, shares

    def volatility_adjusted(
        self,
        capital: float,
        target_vol: float,      # 목표 변동성 (예: 0.15 = 15%)
        asset_vol: float,       # 자산 변동성
        max_weight: float = 0.25
    ) -> float:
        """
        변동성 조정 사이징
        높은 변동성 자산 = 낮은 비중
        """
        if asset_vol == 0:
            return 0.0

        weight = target_vol / asset_vol
        return min(weight, max_weight)

    def risk_parity(
        self,
        volatilities: Dict[str, float],
        correlations: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        리스크 패리티 (각 자산의 리스크 기여도 균등)
        단순화: 변동성의 역수에 비례
        """
        if not volatilities:
            return {}

        # 변동성 역수
        inv_vols = {k: 1/v if v > 0 else 0 for k, v in volatilities.items()}
        total = sum(inv_vols.values())

        if total == 0:
            return {k: 1/len(volatilities) for k in volatilities}

        return {k: v/total for k, v in inv_vols.items()}

    def max_drawdown_constrained(
        self,
        capital: float,
        max_dd_limit: float,    # 최대 허용 DD (예: 0.10 = 10%)
        expected_vol: float,
        confidence: float = 0.95
    ) -> float:
        """
        최대 낙폭 제한 사이징
        """
        # VaR 기반 추정 (정규분포 가정)
        from scipy.stats import norm
        z = norm.ppf(confidence)

        # 일간 VaR에서 월간 추정
        daily_var = expected_vol / np.sqrt(252)
        monthly_var = daily_var * np.sqrt(21) * z

        if monthly_var == 0:
            return 1.0

        # MDD 제한에 맞는 레버리지
        max_leverage = max_dd_limit / monthly_var
        return min(max_leverage, 1.0)


# ============================================================================
# Risk Manager
# ============================================================================

class RiskManager:
    """리스크 관리자"""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_prices(self, ticker: str) -> pd.DataFrame:
        """가격 데이터 로드"""
        if ticker not in self._price_cache:
            end = datetime.now()
            start = end - timedelta(days=self.lookback_days + 30)

            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self._price_cache[ticker] = df

        return self._price_cache[ticker]

    def _get_returns(self, ticker: str) -> pd.Series:
        """수익률 계산"""
        df = self._get_prices(ticker)
        if df.empty:
            return pd.Series()
        return df['Close'].pct_change().dropna()

    def calculate_asset_risk(self, ticker: str) -> Dict[str, float]:
        """개별 자산 리스크"""
        returns = self._get_returns(ticker)
        if returns.empty:
            return {}

        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # VaR 계산 (Historical)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()

        # Max Drawdown
        prices = self._get_prices(ticker)['Close']
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        return {
            'ticker': ticker,
            'daily_vol': round(daily_vol, 4),
            'annual_vol': round(annual_vol, 4),
            'var_95': round(var_95 * 100, 2),
            'var_99': round(var_99 * 100, 2),
            'cvar_95': round(cvar_95 * 100, 2) if not np.isnan(cvar_95) else 0,
            'max_drawdown': round(max_dd * 100, 2),
        }

    def calculate_portfolio_risk(
        self,
        holdings: Dict[str, float],  # {ticker: weight}
        total_value: float = 100000
    ) -> PortfolioRisk:
        """포트폴리오 전체 리스크"""
        if not holdings:
            return PortfolioRisk(
                total_value=total_value,
                daily_vol=0, annual_vol=0,
                var_95=0, var_99=0, cvar_95=0,
                max_drawdown=0, sharpe_estimate=0,
                risk_level=RiskLevel.LOW
            )

        # 각 자산 수익률
        returns_dict = {}
        for ticker in holdings:
            if ticker == 'CASH':
                continue
            ret = self._get_returns(ticker)
            if not ret.empty:
                returns_dict[ticker] = ret

        if not returns_dict:
            return PortfolioRisk(
                total_value=total_value,
                daily_vol=0, annual_vol=0,
                var_95=0, var_99=0, cvar_95=0,
                max_drawdown=0, sharpe_estimate=0,
                risk_level=RiskLevel.LOW
            )

        # 포트폴리오 수익률 (가중평균)
        common_idx = None
        for ret in returns_dict.values():
            if common_idx is None:
                common_idx = ret.index
            else:
                common_idx = common_idx.intersection(ret.index)

        portfolio_returns = pd.Series(0, index=common_idx)
        for ticker, weight in holdings.items():
            if ticker in returns_dict and ticker != 'CASH':
                portfolio_returns += returns_dict[ticker].loc[common_idx] * weight

        # 리스크 지표
        daily_vol = portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Max Drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        # Sharpe 추정 (rf=0)
        mean_return = portfolio_returns.mean() * 252
        sharpe = mean_return / annual_vol if annual_vol > 0 else 0

        # SPY 베타
        spy_returns = self._get_returns("SPY")
        if not spy_returns.empty:
            common = portfolio_returns.index.intersection(spy_returns.index)
            if len(common) > 30:
                cov = portfolio_returns.loc[common].cov(spy_returns.loc[common])
                var_spy = spy_returns.loc[common].var()
                beta = cov / var_spy if var_spy > 0 else 1.0
                corr = portfolio_returns.loc[common].corr(spy_returns.loc[common])
            else:
                beta = 1.0
                corr = 0.0
        else:
            beta = 1.0
            corr = 0.0

        # 리스크 레벨
        if annual_vol < 0.10:
            risk_level = RiskLevel.LOW
        elif annual_vol < 0.20:
            risk_level = RiskLevel.MEDIUM
        elif annual_vol < 0.30:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME

        return PortfolioRisk(
            total_value=total_value,
            daily_vol=round(daily_vol * 100, 2),
            annual_vol=round(annual_vol * 100, 2),
            var_95=round(abs(var_95) * total_value, 2),
            var_99=round(abs(var_99) * total_value, 2),
            cvar_95=round(abs(cvar_95) * total_value, 2) if not np.isnan(cvar_95) else 0,
            max_drawdown=round(max_dd * 100, 2),
            sharpe_estimate=round(sharpe, 2),
            risk_level=risk_level,
            beta=round(beta, 2),
            correlation_to_spy=round(corr, 2),
        )

    def check_position_limits(
        self,
        holdings: Dict[str, float],
        total_value: float,
        max_single_position: float = 0.25,
        max_sector_exposure: float = 0.40
    ) -> List[PositionLimit]:
        """포지션 한도 확인"""
        limits = []

        for ticker, weight in holdings.items():
            max_dollar = total_value * max_single_position
            current_dollar = total_value * weight
            is_exceeded = weight > max_single_position

            limits.append(PositionLimit(
                ticker=ticker,
                max_weight=max_single_position,
                max_dollar=max_dollar,
                current_weight=weight,
                current_dollar=current_dollar,
                is_exceeded=is_exceeded,
                excess_amount=current_dollar - max_dollar if is_exceeded else 0,
            ))

        return limits

    def check_rebalance_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        drift_threshold: float = 0.05,  # 5% 이탈
        last_rebalance: date = None,
        rebalance_frequency_days: int = 30
    ) -> RebalanceRecommendation:
        """리밸런싱 필요 여부 확인"""
        # Drift 계산
        max_drift = 0
        drifts = {}
        for ticker in set(current_weights.keys()) | set(target_weights.keys()):
            curr = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            drift = abs(curr - target)
            drifts[ticker] = drift
            max_drift = max(max_drift, drift)

        # 거래 필요량
        trades_needed = {}
        for ticker in set(current_weights.keys()) | set(target_weights.keys()):
            curr = current_weights.get(ticker, 0)
            target = target_weights.get(ticker, 0)
            trades_needed[ticker] = round(target - curr, 4)

        # 트리거 결정
        trigger = RebalanceTrigger.NONE
        urgency = RiskLevel.LOW
        reasoning = "No rebalancing needed"

        # Drift 기반
        if max_drift > drift_threshold:
            trigger = RebalanceTrigger.DRIFT
            urgency = RiskLevel.MEDIUM if max_drift > drift_threshold * 2 else RiskLevel.LOW
            reasoning = f"Max drift {max_drift:.1%} exceeds threshold {drift_threshold:.1%}"

        # Calendar 기반
        if last_rebalance:
            days_since = (date.today() - last_rebalance).days
            if days_since >= rebalance_frequency_days:
                trigger = RebalanceTrigger.CALENDAR
                reasoning = f"{days_since} days since last rebalance"

        # 추정 비용 (슬리피지 + 수수료)
        total_turnover = sum(abs(v) for v in trades_needed.values()) / 2
        estimated_cost = total_turnover * 0.002  # 0.2% 가정

        return RebalanceRecommendation(
            trigger=trigger,
            urgency=urgency,
            current_weights=current_weights,
            target_weights=target_weights,
            trades_needed=trades_needed,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
        )

    def calculate_optimal_weights(
        self,
        tickers: List[str],
        method: str = "risk_parity"  # "risk_parity", "equal", "min_variance"
    ) -> Dict[str, float]:
        """최적 비중 계산"""
        if method == "equal":
            return {t: 1/len(tickers) for t in tickers}

        # 변동성 계산
        volatilities = {}
        for ticker in tickers:
            risk = self.calculate_asset_risk(ticker)
            if risk:
                volatilities[ticker] = risk['annual_vol'] / 100

        if not volatilities:
            return {t: 1/len(tickers) for t in tickers}

        if method == "risk_parity":
            sizer = PositionSizer()
            return sizer.risk_parity(volatilities)

        # Default
        return {t: 1/len(tickers) for t in tickers}

    def print_risk_report(self, risk: PortfolioRisk, limits: List[PositionLimit] = None):
        """리스크 리포트 출력"""
        print("\n" + "=" * 60)
        print("Portfolio Risk Report")
        print("=" * 60)

        print(f"\nPortfolio Value: ${risk.total_value:,.0f}")
        print(f"Risk Level: {risk.risk_level.value.upper()}")

        print(f"\n[Volatility]")
        print(f"  Daily:  {risk.daily_vol:.2f}%")
        print(f"  Annual: {risk.annual_vol:.2f}%")

        print(f"\n[Value at Risk (1-day)]")
        print(f"  VaR 95%: ${risk.var_95:,.0f}")
        print(f"  VaR 99%: ${risk.var_99:,.0f}")
        print(f"  CVaR 95%: ${risk.cvar_95:,.0f}")

        print(f"\n[Other Metrics]")
        print(f"  Max Drawdown: {risk.max_drawdown:.1f}%")
        print(f"  Sharpe (Est): {risk.sharpe_estimate:.2f}")
        print(f"  Beta: {risk.beta:.2f}")
        print(f"  Corr to SPY: {risk.correlation_to_spy:.2f}")

        if limits:
            exceeded = [l for l in limits if l.is_exceeded]
            if exceeded:
                print(f"\n⚠️ Position Limit Exceeded:")
                for l in exceeded:
                    print(f"  {l.ticker}: {l.current_weight:.1%} > {l.max_weight:.1%} (excess ${l.excess_amount:,.0f})")

        print("=" * 60)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Risk Manager Test")
    print("=" * 60)

    # 테스트 포트폴리오
    holdings = {
        "SPY": 0.40,
        "QQQ": 0.20,
        "TLT": 0.25,
        "GLD": 0.10,
        "CASH": 0.05,
    }

    rm = RiskManager()

    # 포트폴리오 리스크
    print("\nCalculating portfolio risk...")
    risk = rm.calculate_portfolio_risk(holdings, total_value=100000)
    limits = rm.check_position_limits(holdings, 100000)
    rm.print_risk_report(risk, limits)

    # 리밸런싱 체크
    print("\nChecking rebalance trigger...")
    target = {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.20, "GLD": 0.15, "CASH": 0.05}
    rebal = rm.check_rebalance_trigger(holdings, target)
    print(f"  Trigger: {rebal.trigger.value}")
    print(f"  Urgency: {rebal.urgency.value}")
    print(f"  Reasoning: {rebal.reasoning}")

    # Position Sizer
    print("\nPosition Sizing Examples:")
    sizer = PositionSizer()

    kelly = sizer.kelly_criterion(win_rate=0.55, avg_win=0.10, avg_loss=0.05)
    print(f"  Kelly (half): {kelly:.1%}")

    pos_value, shares = sizer.fixed_fractional(
        capital=100000, risk_per_trade=0.02,
        stop_loss_pct=0.05, price=450
    )
    print(f"  Fixed Fractional: ${pos_value:,.0f} ({shares} shares)")

    # Risk Parity
    print("\nRisk Parity Weights:")
    optimal = rm.calculate_optimal_weights(["SPY", "TLT", "GLD"], method="risk_parity")
    for ticker, weight in optimal.items():
        print(f"  {ticker}: {weight:.1%}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
