"""
Tactical Asset Allocation (TAA)
================================
레짐 기반 동적 자산배분

전략:
1. Regime-Based Tilting: 시장 레짐에 따른 배분 조정
2. Dynamic Risk Budgeting: 변동성에 따른 리스크 예산 조정
3. Momentum Overlay: 모멘텀 기반 오버레이
4. Volatility Targeting: 목표 변동성 유지

References:
- Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
- Asness, Moskowitz, Pedersen (2013): "Value and Momentum Everywhere"
- Moreira, Muir (2017): "Volatility-Managed Portfolios"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """시장 레짐 분류"""
    BULL_LOW_VOL = "Bull (Low Vol)"      # 최적 환경
    BULL_HIGH_VOL = "Bull (High Vol)"    # 조심스러운 상승
    BEAR_HIGH_VOL = "Bear (High Vol)"    # 위기
    BEAR_LOW_VOL = "Bear (Low Vol)"      # 조정
    NEUTRAL = "Neutral"                   # 횡보


@dataclass
class RegimeProfile:
    """레짐별 자산배분 프로파일"""
    equity_min: float
    equity_max: float
    equity_target: float

    bond_min: float
    bond_max: float
    bond_target: float

    alternative_min: float
    alternative_max: float
    alternative_target: float  # Commodity + Crypto

    cash_target: float

    def to_dict(self) -> Dict:
        return {
            'equity': {'min': self.equity_min, 'max': self.equity_max, 'target': self.equity_target},
            'bond': {'min': self.bond_min, 'max': self.bond_max, 'target': self.bond_target},
            'alternative': {'min': self.alternative_min, 'max': self.alternative_max, 'target': self.alternative_target},
            'cash': {'target': self.cash_target}
        }


# Regime별 프로파일 정의 (학술 연구 기반)
REGIME_PROFILES = {
    MarketRegime.BULL_LOW_VOL: RegimeProfile(
        equity_min=0.5, equity_max=0.8, equity_target=0.65,
        bond_min=0.15, bond_max=0.35, bond_target=0.25,
        alternative_min=0.05, alternative_max=0.15, alternative_target=0.10,
        cash_target=0.0
    ),
    MarketRegime.BULL_HIGH_VOL: RegimeProfile(
        equity_min=0.4, equity_max=0.6, equity_target=0.50,
        bond_min=0.25, bond_max=0.45, bond_target=0.35,
        alternative_min=0.10, alternative_max=0.20, alternative_target=0.15,
        cash_target=0.0
    ),
    MarketRegime.NEUTRAL: RegimeProfile(
        equity_min=0.3, equity_max=0.5, equity_target=0.40,
        bond_min=0.35, bond_max=0.55, bond_target=0.45,
        alternative_min=0.10, alternative_max=0.20, alternative_target=0.15,
        cash_target=0.0
    ),
    MarketRegime.BEAR_LOW_VOL: RegimeProfile(
        equity_min=0.2, equity_max=0.4, equity_target=0.30,
        bond_min=0.45, bond_max=0.65, bond_target=0.55,
        alternative_min=0.10, alternative_max=0.20, alternative_target=0.15,
        cash_target=0.0
    ),
    MarketRegime.BEAR_HIGH_VOL: RegimeProfile(
        equity_min=0.10, equity_max=0.25, equity_target=0.15,
        bond_min=0.50, bond_max=0.70, bond_target=0.60,
        alternative_min=0.15, alternative_max=0.30, alternative_target=0.20,
        cash_target=0.05
    )
}


class TacticalAssetAllocator:
    """
    전술적 자산배분 엔진

    전략:
    1. 현재 레짐 판단
    2. 레짐별 목표 비중 설정
    3. Strategic 비중에서 Tactical 조정
    4. 제약조건 준수 확인
    """

    def __init__(
        self,
        strategic_weights: Dict[str, float],
        asset_class_mapping: Dict[str, str],
        max_tilt_pct: float = 0.15  # 최대 15% 틸트
    ):
        """
        Args:
            strategic_weights: 전략적 배분 (베이스라인)
            asset_class_mapping: {ticker: asset_class}
            max_tilt_pct: 최대 틸트 비율
        """
        self.strategic_weights = strategic_weights
        self.asset_class_mapping = asset_class_mapping
        self.max_tilt_pct = max_tilt_pct

    def compute_tactical_weights(
        self,
        regime: str,
        confidence: float = 1.0
    ) -> Dict[str, float]:
        """
        전술적 비중 계산

        Args:
            regime: 현재 레짐 (MarketRegime enum 또는 string)
            confidence: 레짐 신뢰도 (0-1), 낮으면 틸트 축소

        Returns:
            {ticker: tactical_weight}
        """
        # Parse regime
        if isinstance(regime, str):
            regime_key = self._parse_regime_string(regime)
        else:
            regime_key = regime

        # Get target profile
        profile = REGIME_PROFILES.get(regime_key)

        if profile is None:
            logger.warning(f"Unknown regime: {regime}. Using strategic weights.")
            return self.strategic_weights.copy()

        # Compute current asset class weights
        current_ac_weights = self._compute_asset_class_weights(self.strategic_weights)

        # Compute tilts
        tilts = {}

        # Equity tilt
        equity_tilt = (profile.equity_target - current_ac_weights.get('equity', 0.4)) * confidence
        equity_tilt = np.clip(equity_tilt, -self.max_tilt_pct, self.max_tilt_pct)
        tilts['equity'] = equity_tilt

        # Bond tilt
        bond_tilt = (profile.bond_target - current_ac_weights.get('bond', 0.4)) * confidence
        bond_tilt = np.clip(bond_tilt, -self.max_tilt_pct, self.max_tilt_pct)
        tilts['bond'] = bond_tilt

        # Alternative tilt (Commodity + Crypto)
        alt_current = current_ac_weights.get('commodity', 0) + current_ac_weights.get('crypto', 0)
        alt_tilt = (profile.alternative_target - alt_current) * confidence
        alt_tilt = np.clip(alt_tilt, -self.max_tilt_pct, self.max_tilt_pct)
        tilts['alternative'] = alt_tilt

        # Apply tilts to strategic weights
        tactical_weights = self._apply_tilts(self.strategic_weights, tilts)

        logger.info(f"Tactical allocation for regime {regime}: Equity tilt {equity_tilt:+.2%}, "
                   f"Bond tilt {bond_tilt:+.2%}, Alt tilt {alt_tilt:+.2%}")

        return tactical_weights

    def _parse_regime_string(self, regime_str: str) -> MarketRegime:
        """레짐 문자열 파싱"""
        regime_lower = regime_str.lower()

        if 'bull' in regime_lower and 'low' in regime_lower:
            return MarketRegime.BULL_LOW_VOL
        elif 'bull' in regime_lower and 'high' in regime_lower:
            return MarketRegime.BULL_HIGH_VOL
        elif 'bear' in regime_lower and 'high' in regime_lower:
            return MarketRegime.BEAR_HIGH_VOL
        elif 'bear' in regime_lower and 'low' in regime_lower:
            return MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.NEUTRAL

    def _compute_asset_class_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """자산군별 비중 계산"""
        ac_weights = {}

        for ticker, weight in weights.items():
            asset_class = self.asset_class_mapping.get(ticker, 'other')

            if asset_class not in ac_weights:
                ac_weights[asset_class] = 0.0
            ac_weights[asset_class] += weight

        return ac_weights

    def _apply_tilts(
        self,
        strategic_weights: Dict[str, float],
        tilts: Dict[str, float]
    ) -> Dict[str, float]:
        """틸트를 strategic 비중에 적용"""

        tactical_weights = {}

        # Compute asset class counts
        ac_counts = {}
        for ticker in strategic_weights:
            ac = self.asset_class_mapping.get(ticker, 'other')
            if ac not in ac_counts:
                ac_counts[ac] = 0
            ac_counts[ac] += 1

        # Apply tilts proportionally
        for ticker, strategic_weight in strategic_weights.items():
            ac = self.asset_class_mapping.get(ticker, 'other')

            # Map alternative to commodity/crypto
            if ac in ['commodity', 'crypto']:
                tilt = tilts.get('alternative', 0) / 2  # Split evenly
            else:
                tilt = tilts.get(ac, 0)

            # Distribute tilt proportionally within asset class
            count = ac_counts.get(ac, 1)
            tilt_per_asset = tilt / count

            tactical_weight = strategic_weight + tilt_per_asset
            tactical_weight = max(0, tactical_weight)  # Non-negative

            tactical_weights[ticker] = tactical_weight

        # Renormalize to sum to 1
        total = sum(tactical_weights.values())
        if total > 0:
            tactical_weights = {k: v/total for k, v in tactical_weights.items()}

        return tactical_weights


class VolatilityTargeting:
    """
    변동성 타겟팅 전략

    목표 변동성을 유지하도록 레버리지 조정

    Reference:
    Moreira, Muir (2017): "Volatility-Managed Portfolios"
    - 변동성이 낮을 때 레버리지 증가
    - 변동성이 높을 때 레버리지 감소
    """

    def __init__(self, target_volatility: float = 0.10):
        """
        Args:
            target_volatility: 목표 연환산 변동성 (기본 10%)
        """
        self.target_volatility = target_volatility

    def compute_leverage(
        self,
        returns: pd.Series,
        lookback_days: int = 60
    ) -> float:
        """
        레버리지 계산

        Leverage = Target Vol / Realized Vol

        Args:
            returns: 수익률 시계열
            lookback_days: 변동성 측정 기간

        Returns:
            레버리지 (1.0 = 100%, 0.5 = 50%, 1.5 = 150%)
        """
        # Realized volatility
        realized_vol = returns.iloc[-lookback_days:].std() * np.sqrt(252)

        if realized_vol == 0:
            return 1.0

        # Leverage
        leverage = self.target_volatility / realized_vol

        # Cap leverage (0.5x ~ 1.5x)
        leverage = np.clip(leverage, 0.5, 1.5)

        return leverage

    def adjust_weights(
        self,
        weights: Dict[str, float],
        leverage: float,
        cash_ticker: str = 'CASH'
    ) -> Dict[str, float]:
        """
        레버리지에 따른 비중 조정

        Args:
            weights: 원래 비중
            leverage: 레버리지 배수
            cash_ticker: 현금 티커명

        Returns:
            조정된 비중
        """
        adjusted = {}

        # Scale risky assets
        for ticker, weight in weights.items():
            if ticker != cash_ticker:
                adjusted[ticker] = weight * leverage

        # Residual to cash
        total_risky = sum(adjusted.values())
        adjusted[cash_ticker] = 1.0 - total_risky

        # Ensure non-negative
        if adjusted[cash_ticker] < 0:
            # Renormalize (no shorting cash)
            adjusted[cash_ticker] = 0
            total = sum(adjusted.values())
            adjusted = {k: v/total for k, v in adjusted.items()}

        return adjusted


class MomentumOverlay:
    """
    모멘텀 오버레이 전략

    Reference:
    Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
    - 10개월(200일) 이동평균선 기준
    - MA 상단: 매수, 하단: 현금/채권
    """

    def __init__(self, lookback_days: int = 200):
        """
        Args:
            lookback_days: 모멘텀 측정 기간 (기본 200일 = 10개월)
        """
        self.lookback_days = lookback_days

    def compute_signals(
        self,
        prices: pd.DataFrame
    ) -> Dict[str, int]:
        """
        모멘텀 시그널 계산

        Args:
            prices: 가격 DataFrame (columns=tickers)

        Returns:
            {ticker: signal}, signal = 1 (매수) or 0 (현금)
        """
        signals = {}

        for ticker in prices.columns:
            price_series = prices[ticker].dropna()

            if len(price_series) < self.lookback_days:
                signals[ticker] = 1  # Insufficient data -> hold
                continue

            # Moving average
            ma = price_series.rolling(self.lookback_days).mean()
            current_price = price_series.iloc[-1]
            current_ma = ma.iloc[-1]

            # Signal
            if current_price > current_ma:
                signals[ticker] = 1  # Buy
            else:
                signals[ticker] = 0  # Cash

        return signals

    def apply_overlay(
        self,
        weights: Dict[str, float],
        signals: Dict[str, int],
        defensive_tickers: List[str] = ['TLT', 'IEF', 'SHY']
    ) -> Dict[str, float]:
        """
        모멘텀 시그널을 비중에 적용

        Args:
            weights: 원래 비중
            signals: 모멘텀 시그널
            defensive_tickers: 방어 자산 (채권)

        Returns:
            조정된 비중
        """
        adjusted = {}

        # Reallocate negative momentum assets to defensive
        total_reallocate = 0.0

        for ticker, weight in weights.items():
            signal = signals.get(ticker, 1)

            if signal == 1:
                adjusted[ticker] = weight
            else:
                # Zero out negative momentum
                total_reallocate += weight
                adjusted[ticker] = 0.0

        # Distribute to defensive assets
        if total_reallocate > 0 and defensive_tickers:
            defensive_weight = total_reallocate / len(defensive_tickers)

            for def_ticker in defensive_tickers:
                if def_ticker in adjusted:
                    adjusted[def_ticker] += defensive_weight
                else:
                    adjusted[def_ticker] = defensive_weight

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}

        return adjusted


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Tactical Allocation
    strategic_weights = {
        'SPY': 0.25,
        'QQQ': 0.15,
        'IWM': 0.10,
        'TLT': 0.30,
        'GLD': 0.10,
        'BTC-USD': 0.05,
        'DBC': 0.05
    }

    asset_class_mapping = {
        'SPY': 'equity',
        'QQQ': 'equity',
        'IWM': 'equity',
        'TLT': 'bond',
        'GLD': 'commodity',
        'BTC-USD': 'crypto',
        'DBC': 'commodity'
    }

    # Tactical allocation
    taa = TacticalAssetAllocator(strategic_weights, asset_class_mapping, max_tilt_pct=0.15)

    # Bull Low Vol regime
    tactical_bull = taa.compute_tactical_weights("Bull (Low Vol)", confidence=0.8)
    print("=== Bull (Low Vol) Tactical Allocation ===")
    for ticker, weight in tactical_bull.items():
        strategic = strategic_weights[ticker]
        change = weight - strategic
        print(f"{ticker}: {weight:.2%} (strategic: {strategic:.2%}, change: {change:+.2%})")

    print("\n=== Bear (High Vol) Tactical Allocation ===")
    tactical_bear = taa.compute_tactical_weights("Bear (High Vol)", confidence=0.8)
    for ticker, weight in tactical_bear.items():
        strategic = strategic_weights[ticker]
        change = weight - strategic
        print(f"{ticker}: {weight:.2%} (strategic: {strategic:.2%}, change: {change:+.2%})")
