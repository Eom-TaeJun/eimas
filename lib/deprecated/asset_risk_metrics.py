#!/usr/bin/env python3
"""
Asset Risk Metrics Module
=========================
자산별 위험조정수익률 및 리스크 메트릭 계산

주요 메트릭:
1. Sharpe Ratio - 위험조정수익률
2. Sortino Ratio - 하방위험조정수익률
3. VaR (Value at Risk) 95% - 최대손실 추정
4. CVaR (Conditional VaR) 95% - 꼬리위험
5. Maximum Drawdown - 최대 낙폭
6. Calmar Ratio - MDD 대비 수익률

경제학적 방법론:
- Sharpe (1966): Risk-adjusted return의 표준
- Sortino & Price (1994): Downside risk에 집중
- VaR: JP Morgan RiskMetrics (1994)
- CVaR: Artzner et al. (1999) Coherent risk measure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AssetRiskMetrics:
    """개별 자산의 리스크 메트릭"""
    ticker: str
    name: str
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    calmar_ratio: float
    max_dd_peak_date: str
    max_dd_trough_date: str
    max_dd_duration_days: int
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskMetricsResult:
    """전체 리스크 메트릭 결과"""
    timestamp: str
    metrics: Dict[str, AssetRiskMetrics]
    summary: str
    high_risk_assets: List[str]
    low_risk_assets: List[str]
    interpretation: str

    def to_dict(self) -> Dict:
        result = {
            'timestamp': self.timestamp,
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'summary': self.summary,
            'high_risk_assets': self.high_risk_assets,
            'low_risk_assets': self.low_risk_assets,
            'interpretation': self.interpretation,
        }
        return result


class AssetRiskCalculator:
    """
    자산별 리스크 메트릭 계산기

    Features:
    - Sharpe/Sortino Ratio
    - VaR/CVaR (Historical Simulation)
    - Maximum Drawdown with dates
    - Calmar Ratio
    - 분포 특성 (Skewness, Kurtosis)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 연간 무위험수익률 (5%)
        lookback_days: int = 60,
        var_confidence: float = 0.95,
    ):
        """
        Args:
            risk_free_rate: 연간 무위험수익률 (default: 5%)
            lookback_days: 분석 기간 (default: 60일)
            var_confidence: VaR 신뢰수준 (default: 95%)
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        self.var_confidence = var_confidence
        self.daily_rf = risk_free_rate / 252  # 일간 무위험수익률

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Sharpe Ratio 계산

        Sharpe = (Rp - Rf) / σp

        경제학적 의미:
        - 위험 1단위당 초과수익률
        - Sharpe > 1: 좋음, > 2: 매우 좋음, > 3: 탁월
        - 음수: 무위험 대비 손실
        """
        if len(returns) < 5 or returns.std() == 0:
            return 0.0

        excess_return = returns.mean() - self.daily_rf
        daily_sharpe = excess_return / returns.std()
        annualized_sharpe = daily_sharpe * np.sqrt(252)

        return annualized_sharpe

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Sortino Ratio 계산

        Sortino = (Rp - Rf) / σd (σd = 하방 표준편차)

        경제학적 의미:
        - 하방 위험만 고려한 위험조정수익률
        - 상승 변동성은 투자자에게 좋은 것이므로 제외
        - Sharpe 대비 더 정확한 리스크 측정
        """
        if len(returns) < 5:
            return 0.0

        excess_return = returns.mean() - self.daily_rf
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_return > 0 else 0.0

        downside_std = downside_returns.std()
        daily_sortino = excess_return / downside_std
        annualized_sortino = daily_sortino * np.sqrt(252)

        return annualized_sortino

    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """
        VaR (Value at Risk) 계산 - Historical Simulation

        경제학적 의미:
        - 특정 신뢰수준에서 예상되는 최대 손실
        - VaR 95% = -2%: 95% 확률로 일일 손실이 2% 이하
        - 한계: 꼬리 리스크 과소평가 가능
        """
        confidence = confidence or self.var_confidence

        if len(returns) < 5:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        return var

    def calculate_cvar(self, returns: pd.Series, confidence: float = None) -> float:
        """
        CVaR (Conditional VaR / Expected Shortfall) 계산

        경제학적 의미:
        - VaR를 초과하는 손실의 평균
        - 꼬리 리스크를 더 잘 포착
        - Coherent risk measure (일관된 위험 측정)
        """
        confidence = confidence or self.var_confidence

        if len(returns) < 5:
            return 0.0

        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()

        return cvar if not pd.isna(cvar) else var

    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, str, str, int]:
        """
        Maximum Drawdown 계산

        경제학적 의미:
        - 고점 대비 최대 하락폭
        - 투자자가 경험할 수 있는 최악의 시나리오
        - MDD > 20%: 고위험, MDD > 50%: 극단적 위험

        Returns:
            (MDD 비율, 고점 날짜, 저점 날짜, 회복 기간)
        """
        if len(prices) < 2:
            return 0.0, "", "", 0

        # Running maximum
        running_max = prices.expanding().max()

        # Drawdown series
        drawdown = (prices - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # 고점/저점 날짜
        trough_idx = drawdown.idxmin()
        peak_idx = prices[:trough_idx].idxmax() if trough_idx in prices.index else prices.index[0]

        # 기간 계산
        if isinstance(peak_idx, pd.Timestamp) and isinstance(trough_idx, pd.Timestamp):
            duration = (trough_idx - peak_idx).days
        else:
            duration = 0

        peak_date = str(peak_idx)[:10] if hasattr(peak_idx, 'isoformat') else str(peak_idx)
        trough_date = str(trough_idx)[:10] if hasattr(trough_idx, 'isoformat') else str(trough_idx)

        return max_dd, peak_date, trough_date, duration

    def calculate_calmar_ratio(self, returns: pd.Series, max_dd: float) -> float:
        """
        Calmar Ratio 계산

        Calmar = 연간수익률 / |MDD|

        경제학적 의미:
        - 최대 낙폭 대비 수익률
        - 헤지펀드 성과 평가에 주로 사용
        - Calmar > 1: 수익률이 MDD보다 큼
        """
        if max_dd == 0 or len(returns) < 5:
            return 0.0

        annualized_return = returns.mean() * 252
        calmar = annualized_return / abs(max_dd)

        return calmar

    def calculate_distribution_stats(self, returns: pd.Series) -> Tuple[float, float]:
        """
        분포 특성 계산 (Skewness, Kurtosis)

        경제학적 의미:
        - Skewness < 0: 왼쪽 꼬리가 긴 분포 (급락 위험)
        - Kurtosis > 3: 정규분포보다 두꺼운 꼬리 (극단 이벤트)
        """
        if len(returns) < 10:
            return 0.0, 0.0

        skew = returns.skew()
        kurt = returns.kurtosis()

        return skew, kurt

    def calculate_asset_metrics(
        self,
        ticker: str,
        df: pd.DataFrame,
        name: str = None
    ) -> AssetRiskMetrics:
        """개별 자산 메트릭 계산"""
        name = name or ticker

        # 가격 추출
        if 'Close' in df.columns:
            prices = df['Close']
        elif 'Adj Close' in df.columns:
            prices = df['Adj Close']
        else:
            raise ValueError(f"No price column found for {ticker}")

        # lookback 적용
        if len(prices) > self.lookback_days:
            prices = prices.tail(self.lookback_days)

        # 수익률 계산
        returns = prices.pct_change().dropna()

        if len(returns) < 5:
            return AssetRiskMetrics(
                ticker=ticker,
                name=name,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                max_dd_peak_date="",
                max_dd_trough_date="",
                max_dd_duration_days=0,
            )

        # 메트릭 계산
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        var_95 = self.calculate_var(returns)
        cvar_95 = self.calculate_cvar(returns)
        max_dd, peak_date, trough_date, duration = self.calculate_max_drawdown(prices)
        calmar = self.calculate_calmar_ratio(returns, max_dd)
        skew, kurt = self.calculate_distribution_stats(returns)

        # 연환산 수익률/변동성
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)

        return AssetRiskMetrics(
            ticker=ticker,
            name=name,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            max_dd_peak_date=peak_date,
            max_dd_trough_date=trough_date,
            max_dd_duration_days=duration,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            skewness=skew,
            kurtosis=kurt,
        )

    def analyze_all(
        self,
        market_data: Dict[str, pd.DataFrame],
        asset_names: Dict[str, str] = None
    ) -> RiskMetricsResult:
        """전체 시장 데이터 분석"""
        asset_names = asset_names or {}
        metrics = {}
        timestamp = datetime.now().isoformat()

        for ticker, df in market_data.items():
            try:
                name = asset_names.get(ticker, ticker)
                metrics[ticker] = self.calculate_asset_metrics(ticker, df, name)
            except Exception as e:
                print(f"Warning: Error calculating metrics for {ticker}: {e}")
                continue

        # 고위험/저위험 자산 분류
        high_risk = []
        low_risk = []

        for ticker, m in metrics.items():
            # MDD > 10% or Sharpe < 0
            if m.max_drawdown < -0.10 or m.sharpe_ratio < 0:
                high_risk.append(ticker)
            # Sharpe > 1 and MDD > -5%
            elif m.sharpe_ratio > 1 and m.max_drawdown > -0.05:
                low_risk.append(ticker)

        # 요약 생성
        summary = self._generate_summary(metrics)
        interpretation = self._generate_interpretation(metrics, high_risk, low_risk)

        return RiskMetricsResult(
            timestamp=timestamp,
            metrics=metrics,
            summary=summary,
            high_risk_assets=high_risk,
            low_risk_assets=low_risk,
            interpretation=interpretation,
        )

    def _generate_summary(self, metrics: Dict[str, AssetRiskMetrics]) -> str:
        """요약 텍스트 생성"""
        if not metrics:
            return "분석된 자산이 없습니다."

        sharpes = [m.sharpe_ratio for m in metrics.values() if m.sharpe_ratio != 0]
        mdds = [m.max_drawdown for m in metrics.values() if m.max_drawdown != 0]

        avg_sharpe = np.mean(sharpes) if sharpes else 0
        avg_mdd = np.mean(mdds) if mdds else 0

        best_sharpe = max(metrics.items(), key=lambda x: x[1].sharpe_ratio)
        worst_mdd = min(metrics.items(), key=lambda x: x[1].max_drawdown)

        summary = f"""리스크 메트릭 요약 ({len(metrics)}개 자산 분석):
- 평균 Sharpe Ratio: {avg_sharpe:.2f}
- 평균 Maximum Drawdown: {avg_mdd:.1%}
- 최고 Sharpe: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})
- 최대 MDD: {worst_mdd[0]} ({worst_mdd[1].max_drawdown:.1%})"""

        return summary

    def _generate_interpretation(
        self,
        metrics: Dict[str, AssetRiskMetrics],
        high_risk: List[str],
        low_risk: List[str]
    ) -> str:
        """해석 텍스트 생성"""
        parts = []

        if high_risk:
            parts.append(f"고위험 자산 ({len(high_risk)}개): {', '.join(high_risk[:5])}")
            parts.append("  → 포지션 축소 또는 헷지 검토 필요")

        if low_risk:
            parts.append(f"저위험 고수익 자산 ({len(low_risk)}개): {', '.join(low_risk[:5])}")
            parts.append("  → 비중 확대 검토 가능")

        # 분포 이상 체크
        fat_tail_assets = [
            t for t, m in metrics.items()
            if m.kurtosis > 5  # 정규분포보다 두꺼운 꼬리
        ]
        if fat_tail_assets:
            parts.append(f"꼬리 위험 주의 자산: {', '.join(fat_tail_assets[:3])}")
            parts.append("  → 극단적 이벤트 발생 가능성 높음 (Kurtosis > 5)")

        left_skew_assets = [
            t for t, m in metrics.items()
            if m.skewness < -0.5  # 왼쪽 꼬리
        ]
        if left_skew_assets:
            parts.append(f"급락 위험 자산: {', '.join(left_skew_assets[:3])}")
            parts.append("  → 수익률 분포가 왼쪽으로 치우침 (Skewness < -0.5)")

        if not parts:
            parts.append("전반적으로 위험 수준이 정상 범위입니다.")

        return '\n'.join(parts)

    def get_top_performers(
        self,
        metrics: Dict[str, AssetRiskMetrics],
        metric_name: str = 'sharpe_ratio',
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """특정 메트릭 기준 상위 자산"""
        sorted_assets = sorted(
            metrics.items(),
            key=lambda x: getattr(x[1], metric_name),
            reverse=True
        )
        return [(t, getattr(m, metric_name)) for t, m in sorted_assets[:top_n]]

    def get_worst_performers(
        self,
        metrics: Dict[str, AssetRiskMetrics],
        metric_name: str = 'max_drawdown',
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """특정 메트릭 기준 하위 자산"""
        sorted_assets = sorted(
            metrics.items(),
            key=lambda x: getattr(x[1], metric_name)
        )
        return [(t, getattr(m, metric_name)) for t, m in sorted_assets[:top_n]]


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=== Asset Risk Metrics Test ===\n")

    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

    # 다양한 특성의 자산 시뮬레이션
    # SPY: 안정적 상승
    spy_returns = np.random.randn(100) * 0.01 + 0.0005
    spy_prices = 100 * np.exp(np.cumsum(spy_returns))

    # QQQ: 변동성 높음
    qqq_returns = np.random.randn(100) * 0.02 + 0.0003
    qqq_prices = 100 * np.exp(np.cumsum(qqq_returns))

    # TLT: 안정적
    tlt_returns = np.random.randn(100) * 0.005 + 0.0002
    tlt_prices = 100 * np.exp(np.cumsum(tlt_returns))

    # BTC: 고변동성, 급락 포함
    btc_returns = np.random.randn(100) * 0.04
    btc_returns[50:55] = -0.05  # 급락 기간
    btc_prices = 100 * np.exp(np.cumsum(btc_returns))

    market_data = {
        'SPY': pd.DataFrame({'Close': spy_prices}, index=dates),
        'QQQ': pd.DataFrame({'Close': qqq_prices}, index=dates),
        'TLT': pd.DataFrame({'Close': tlt_prices}, index=dates),
        'BTC-USD': pd.DataFrame({'Close': btc_prices}, index=dates),
    }

    asset_names = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'TLT': 'Long Treasury',
        'BTC-USD': 'Bitcoin',
    }

    # 분석
    calculator = AssetRiskCalculator(lookback_days=60)
    result = calculator.analyze_all(market_data, asset_names)

    print(result.summary)
    print("\n" + "=" * 50 + "\n")

    print("자산별 상세 메트릭:\n")
    for ticker, m in result.metrics.items():
        print(f"{ticker} ({m.name}):")
        print(f"  Sharpe: {m.sharpe_ratio:.2f}")
        print(f"  Sortino: {m.sortino_ratio:.2f}")
        print(f"  VaR 95%: {m.var_95:.2%}")
        print(f"  CVaR 95%: {m.cvar_95:.2%}")
        print(f"  Max DD: {m.max_drawdown:.2%} ({m.max_dd_peak_date} → {m.max_dd_trough_date})")
        print(f"  Calmar: {m.calmar_ratio:.2f}")
        print(f"  Skewness: {m.skewness:.2f}, Kurtosis: {m.kurtosis:.2f}")
        print()

    print("해석:")
    print(result.interpretation)
