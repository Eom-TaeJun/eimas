#!/usr/bin/env python3
"""
Signal Analyzer Module
======================
다양한 기술적/통계적 지표 기반 시그널 탐지 + 행동 가이드 + 학술적 근거

주요 기능:
1. Z-score 기반 이상치 탐지 (가격, 수익률)
2. RSI 과매수/과매도
3. Bollinger Band 돌파
4. 거래량 이상 탐지
5. 각 시그널에 대한 action_guide와 theory_note 제공

경제학적 방법론:
- Mean Reversion: Poterba & Summers(1988)
- Momentum: Jegadeesh & Titman(1993)
- Volume-Price Relationship: Karpoff(1987)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class SignalLevel(Enum):
    CRITICAL = "CRITICAL"  # Z > 3 or 심각한 이탈
    ALERT = "ALERT"        # Z > 2 or 주의 필요
    WARNING = "WARNING"    # Z > 1.5 or 관찰 필요


class SignalType(Enum):
    STATISTICAL = "statistical"
    EARLY_WARNING = "early_warning"
    TECHNICAL = "technical"


@dataclass
class Signal:
    """시그널 데이터 클래스"""
    type: str
    ticker: str
    name: str
    indicator: str
    value: float
    threshold: float
    z_score: float
    level: str
    description: str
    action_guide: str
    theory_note: str
    timestamp: str
    risk_prob: float = 0.0
    risk_level_ml: str = "LOW"
    risk_model_type: str = "heuristic"

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 자산군별 이론적 근거 (Theory Notes)
# ============================================================================

THEORY_NOTES = {
    # 주식
    'equity': {
        'mean_reversion': "평균회귀(Mean Reversion): 가격이 장기 평균에서 크게 이탈 시 회귀 경향. Poterba & Summers(1988): 주가는 장기적으로 평균회귀.",
        'momentum': "단기 과잉반응: Jegadeesh & Titman(1993) 모멘텀 연구 - 3-12개월은 추세 지속, 이후 반전. 극단적 급등은 단기 조정 가능성.",
        'overbought': "기술적 과매수: RSI 70+ 구간에서 통계적으로 단기 조정 확률 증가. 단, 강한 추세장에서는 지속 가능.",
        'oversold': "기술적 과매도: RSI 30- 구간은 반등 가능성 증가. Contrarian 전략의 근거.",
        'bollinger': "볼린저밴드 상단돌파: 강한 모멘텀 또는 과열 신호. 추세 지속과 평균회귀 중 판단 필요.",
        'volume': "거래량-수익률 관계: Karpoff(1987) 메타분석 - 거래량은 정보비대칭 해소 신호. 방향성과 함께 해석 필요.",
    },
    # 채권
    'bond': {
        'credit_spread': "신용스프레드와 경기: 하이일드 스프레드 확대는 경기침체 선행지표. Gilchrist & Zakrajsek(2012): 신용스프레드의 거시경제 예측력.",
        'term_structure': "기간구조 이론(Term Structure): 단기 금리는 중앙은행 정책에 민감. 기대이론과 유동성프리미엄 이론의 결합.",
        'duration': "듀레이션 리스크: 금리 1% 변화 시 채권 가격 변화율. 장기채일수록 금리 민감도 증가.",
    },
    # 원자재
    'commodity': {
        'gold': "금의 이중적 성격: 인플레이션 헷지 + 안전자산. Erb & Harvey(2013): 장기 금 수익률은 인플레이션 대비 0에 수렴, 단기는 심리와 달러에 좌우.",
        'silver': "금-은 비율(Gold-Silver Ratio): 역사적 평균 60-70배. 비율 급등 시 경기침체 우려, 급락 시 인플레이션 기대. 은은 금 대비 산업수요(태양광, 전자) 비중 높아 경기민감.",
        'copper': "구리는 경기선행지표(Dr. Copper): 산업 수요의 바로미터. 중국 건설/제조업과 높은 상관관계.",
        'oil': "유가와 인플레이션: 유가 10% 상승 시 CPI 0.2-0.4%p 상승 압력. 수요 견인(긍정) vs 공급 충격(부정) 구분 필요.",
    },
    # 환율
    'fx': {
        'dollar': "환율 결정이론: 구매력평가(PPP), 금리평가(IRP), 자산접근법의 복합 작용. 단기는 금리차, 장기는 인플레차에 수렴.",
        'em_currency': "금리평가설과 자본흐름: 신흥국 통화는 금리차와 자본유출입에 민감. Forbes & Warnock(2012) 연구 참조.",
    },
    # 암호화폐
    'crypto': {
        'btc': "비트코인 특성: 디지털 금 vs 투기 자산. 반감기 사이클, 고래 동향, 규제 불확실성이 주요 변수.",
        'altcoin': "알트코인 베타: 비트코인 대비 높은 변동성. 기술적 업데이트, 네트워크 성장, 유동성이 핵심.",
    },
    # VIX
    'vix': {
        'spike': "VIX 스파이크: 공포의 정량화. VIX 30+ 시 시장 스트레스, VIX 20 이하는 안정. Whaley(2000): VIX는 투자자 공포의 척도.",
        'term_structure': "VIX 기간구조: 콘탱고(정상) vs 백워데이션(위기). 백워데이션은 단기 불확실성 급등 신호.",
    },
}

# ============================================================================
# 지표별 행동 가이드 (Action Guides)
# ============================================================================

ACTION_GUIDES = {
    'price_surge': "급등 후 차익실현 매물 출회 가능. 신규 매수보다 보유 물량 일부 이익실현 고려.",
    'price_plunge': "급락 시 패닉셀 주의. 펀더멘털 확인 후 분할 매수 기회 탐색 가능.",
    'return_surge': "단기 과열 신호. 추격 매수 자제, 되돌림 시 진입 검토.",
    'return_plunge': "단기 과매도 가능성. 반등 시 단기 트레이딩 기회, 단 추세 확인 필요.",
    'rsi_overbought': "과매수 구간. 신규 매수 자제, 보유 시 일부 차익실현 고려.",
    'rsi_oversold': "과매도 구간. 반등 가능성 있으나 하락 추세 확인 필요. 분할 매수 검토.",
    'bollinger_upper': "상단 돌파는 강한 상승 모멘텀이나 과열 신호. 추격 매수 주의.",
    'bollinger_lower': "하단 이탈은 강한 하락 또는 반등 임박. 손절/분할매수 결정 필요.",
    'volume_spike': "대량 거래는 세력/기관 움직임 시사. 방향성 확인 후 추종 또는 관망.",
    'volume_buildup': "가격 보합+거래량 증가는 방향 전환 임박. 돌파 방향 확인 후 대응.",
}


# ============================================================================
# 자산군 분류
# ============================================================================

ASSET_CATEGORIES = {
    # 주식 ETF
    'SPY': ('equity', 'S&P 500'),
    'QQQ': ('equity', 'Nasdaq 100'),
    'IWM': ('equity', 'Russell 2000'),
    'DIA': ('equity', 'Dow Jones'),
    'EEM': ('equity', 'Emerging Markets'),
    'IWF': ('equity', 'Growth'),
    'IWD': ('equity', 'Value'),
    # 섹터 ETF
    'XLK': ('equity', 'Technology'),
    'XLF': ('equity', 'Financial'),
    'XLV': ('equity', 'Healthcare'),
    'XLE': ('equity', 'Energy'),
    'XLI': ('equity', 'Industrial'),
    'XLY': ('equity', 'Consumer Discretionary'),
    'XLP': ('equity', 'Consumer Staples'),
    'XLU': ('equity', 'Utilities'),
    'XLB': ('equity', 'Materials'),
    'XLC': ('equity', 'Communication'),
    'XLRE': ('equity', 'Real Estate'),
    # 채권 ETF
    'TLT': ('bond', 'Long-Term Treasury'),
    'IEF': ('bond', 'Intermediate Treasury'),
    'SHY': ('bond', 'Short-Term Treasury'),
    'HYG': ('bond', 'High Yield Corporate'),
    'LQD': ('bond', 'Investment Grade Corporate'),
    'TIP': ('bond', 'TIPS'),
    # 원자재
    'GLD': ('commodity', 'Gold ETF'),
    'GC=F': ('commodity', 'Gold Futures'),
    'SLV': ('commodity', 'Silver ETF'),
    'SI=F': ('commodity', 'Silver Futures'),
    'CL=F': ('commodity', 'WTI Crude'),
    'BZ=F': ('commodity', 'Brent Crude'),
    'NG=F': ('commodity', 'Natural Gas'),
    'HG=F': ('commodity', 'Copper'),
    'ZW=F': ('commodity', 'Wheat'),
    'ZC=F': ('commodity', 'Corn'),
    'ZS=F': ('commodity', 'Soybean'),
    'DBC': ('commodity', 'Commodities ETF'),
    'DBA': ('commodity', 'Agriculture ETF'),
    # 환율
    'DX-Y.NYB': ('fx', 'Dollar Index'),
    'EURUSD=X': ('fx', 'EUR/USD'),
    'GBPUSD=X': ('fx', 'GBP/USD'),
    'USDJPY=X': ('fx', 'USD/JPY'),
    'USDCNY=X': ('fx', 'USD/CNY'),
    'USDKRW=X': ('fx', 'USD/KRW'),
    # 암호화폐
    'BTC-USD': ('crypto', 'Bitcoin'),
    'ETH-USD': ('crypto', 'Ethereum'),
    'SOL-USD': ('crypto', 'Solana'),
    # VIX
    '^VIX': ('vix', 'VIX'),
    '^VIX3M': ('vix', 'VIX 3-Month'),
    # 부동산
    'VNQ': ('equity', 'Real Estate'),
    'IYR': ('equity', 'Real Estate'),
}


def get_asset_info(ticker: str) -> tuple:
    """자산 정보 반환 (카테고리, 이름)"""
    if ticker in ASSET_CATEGORIES:
        return ASSET_CATEGORIES[ticker]
    return ('equity', ticker)


def get_theory_note(ticker: str, indicator: str) -> str:
    """해당 자산과 지표에 맞는 이론적 근거 반환"""
    category, _ = get_asset_info(ticker)

    # 지표별 이론 선택
    if indicator in ['price_z', 'return_z']:
        if category == 'commodity' and 'GLD' in ticker or 'GC' in ticker:
            return THEORY_NOTES['commodity']['gold']
        elif category == 'commodity' and ('SLV' in ticker or 'SI' in ticker):
            return THEORY_NOTES['commodity']['silver']
        elif category == 'fx':
            if 'DX' in ticker:
                return THEORY_NOTES['fx']['dollar']
            return THEORY_NOTES['fx']['em_currency']
        elif category == 'bond':
            if 'HYG' in ticker:
                return THEORY_NOTES['bond']['credit_spread']
            return THEORY_NOTES['bond']['term_structure']
        elif category == 'vix':
            return THEORY_NOTES['vix']['spike']
        else:
            return THEORY_NOTES['equity']['mean_reversion']
    elif indicator == 'rsi':
        return THEORY_NOTES['equity']['overbought']
    elif indicator == 'bollinger':
        return THEORY_NOTES['equity']['bollinger']
    elif indicator == 'volume':
        return THEORY_NOTES['equity']['volume']
    elif indicator == 'volume_buildup':
        return "축적/분산(Accumulation/Distribution) 이론: 가격 보합 중 거래량 증가는 세력의 포지션 구축. 돌파 방향에 큰 움직임 예상."

    return THEORY_NOTES['equity']['mean_reversion']


# ============================================================================
# Signal Analyzer Class
# ============================================================================

class SignalAnalyzer:
    """
    다양한 지표 기반 시그널 분석기

    Features:
    - Z-score 기반 이상치 탐지
    - RSI 과매수/과매도
    - Bollinger Band
    - 거래량 이상
    - 각 시그널에 action_guide와 theory_note 제공
    """

    def __init__(
        self,
        lookback: int = 60,
        z_threshold_warning: float = 1.5,
        z_threshold_alert: float = 2.0,
        z_threshold_critical: float = 3.0,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        volume_threshold: float = 1.5,
    ):
        self.lookback = lookback
        self.z_threshold_warning = z_threshold_warning
        self.z_threshold_alert = z_threshold_alert
        self.z_threshold_critical = z_threshold_critical
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_threshold = volume_threshold

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_bollinger(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
        """볼린저 밴드 계산 - (현재가격, 상단, 하단, %B)"""
        if len(prices) < window:
            return prices.iloc[-1], prices.iloc[-1], prices.iloc[-1], 0.5

        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper = sma + (num_std * std)
        lower = sma - (num_std * std)

        current = prices.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]

        # %B = (가격 - 하단) / (상단 - 하단)
        band_width = upper_val - lower_val
        if band_width > 0:
            percent_b = (current - lower_val) / band_width
        else:
            percent_b = 0.5

        return current, upper_val, lower_val, percent_b

    def calculate_z_score(self, series: pd.Series, window: int = None) -> float:
        """Z-score 계산"""
        window = window or self.lookback
        if len(series) < window:
            window = len(series)
        if window < 2:
            return 0.0

        recent = series.tail(window)
        mean = recent.mean()
        std = recent.std()

        if std == 0 or pd.isna(std):
            return 0.0

        return (series.iloc[-1] - mean) / std

    def get_risk_prob(self, ticker: str, z_score: float, indicator: str) -> tuple:
        """ML/휴리스틱 기반 리스크 확률 추정"""
        category, _ = get_asset_info(ticker)

        # 기본 휴리스틱 리스크 추정
        base_prob = min(0.05 + abs(z_score) * 0.05, 0.5)

        # 자산군별 조정
        if category == 'vix':
            base_prob = min(base_prob * 1.5, 0.5)
            model_type = 'heuristic_vix'
        elif category == 'fx':
            base_prob = min(base_prob * 1.2, 0.5)
            model_type = 'heuristic_fx'
        elif category == 'crypto':
            base_prob = min(base_prob * 1.3, 0.5)
            model_type = 'heuristic_crypto'
        else:
            model_type = 'heuristic'

        # 리스크 레벨
        if base_prob >= 0.4:
            risk_level = 'HIGH'
        elif base_prob >= 0.2:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return base_prob, risk_level, model_type

    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> List[Signal]:
        """개별 티커 분석"""
        signals = []
        timestamp = datetime.now().isoformat()
        category, name = get_asset_info(ticker)

        if df is None or df.empty or len(df) < 5:
            return signals

        # Close 가격 추출
        if 'Close' in df.columns:
            prices = df['Close']
        elif 'Adj Close' in df.columns:
            prices = df['Adj Close']
        else:
            return signals

        # 수익률 계산
        returns = prices.pct_change().dropna() * 100

        # 1. 가격 Z-score
        price_z = self.calculate_z_score(prices)
        if abs(price_z) >= self.z_threshold_warning:
            level = self._get_level(abs(price_z))
            is_surge = price_z > 0

            risk_prob, risk_level, model_type = self.get_risk_prob(ticker, price_z, 'price_z')

            signals.append(Signal(
                type=SignalType.STATISTICAL.value,
                ticker=ticker,
                name=name,
                indicator='price_z',
                value=prices.iloc[-1],
                threshold=self.z_threshold_warning,
                z_score=price_z,
                level=level.value,
                description=f"{name} 가격 {'급등' if is_surge else '급락'} (Z={price_z:.2f})",
                action_guide=ACTION_GUIDES['price_surge' if is_surge else 'price_plunge'],
                theory_note=get_theory_note(ticker, 'price_z'),
                timestamp=timestamp,
                risk_prob=risk_prob,
                risk_level_ml=risk_level,
                risk_model_type=model_type,
            ))

        # 2. 수익률 Z-score
        if len(returns) >= 5:
            return_z = self.calculate_z_score(returns)
            if abs(return_z) >= self.z_threshold_warning:
                level = self._get_level(abs(return_z))
                is_surge = return_z > 0

                risk_prob, risk_level, model_type = self.get_risk_prob(ticker, return_z, 'return_z')

                signals.append(Signal(
                    type=SignalType.STATISTICAL.value,
                    ticker=ticker,
                    name=name,
                    indicator='return_z',
                    value=returns.iloc[-1],
                    threshold=self.z_threshold_warning,
                    z_score=return_z,
                    level=level.value,
                    description=f"{name} 일일 수익률 이상 (Z={return_z:.2f})",
                    action_guide=ACTION_GUIDES['return_surge' if is_surge else 'return_plunge'],
                    theory_note=get_theory_note(ticker, 'return_z'),
                    timestamp=timestamp,
                    risk_prob=risk_prob,
                    risk_level_ml=risk_level,
                    risk_model_type=model_type,
                ))

        # 3. RSI
        rsi = self.calculate_rsi(prices)
        if rsi >= self.rsi_overbought:
            risk_prob, risk_level, model_type = self.get_risk_prob(ticker, (rsi - 50) / 10, 'rsi')
            signals.append(Signal(
                type=SignalType.STATISTICAL.value,
                ticker=ticker,
                name=name,
                indicator='rsi',
                value=rsi,
                threshold=self.rsi_overbought,
                z_score=0,
                level=SignalLevel.ALERT.value if rsi >= 80 else SignalLevel.WARNING.value,
                description=f"{name} RSI 과매수 ({rsi:.1f})",
                action_guide=ACTION_GUIDES['rsi_overbought'],
                theory_note=get_theory_note(ticker, 'rsi'),
                timestamp=timestamp,
                risk_prob=risk_prob,
                risk_level_ml=risk_level,
                risk_model_type=model_type,
            ))
        elif rsi <= self.rsi_oversold:
            risk_prob, risk_level, model_type = self.get_risk_prob(ticker, (50 - rsi) / 10, 'rsi')
            signals.append(Signal(
                type=SignalType.STATISTICAL.value,
                ticker=ticker,
                name=name,
                indicator='rsi',
                value=rsi,
                threshold=self.rsi_oversold,
                z_score=0,
                level=SignalLevel.ALERT.value if rsi <= 20 else SignalLevel.WARNING.value,
                description=f"{name} RSI 과매도 ({rsi:.1f})",
                action_guide=ACTION_GUIDES['rsi_oversold'],
                theory_note=get_theory_note(ticker, 'rsi'),
                timestamp=timestamp,
                risk_prob=risk_prob,
                risk_level_ml=risk_level,
                risk_model_type=model_type,
            ))

        # 4. Bollinger Band
        _, upper, lower, percent_b = self.calculate_bollinger(prices)
        if percent_b > 1.0:  # 상단 돌파
            risk_prob, risk_level, model_type = self.get_risk_prob(ticker, percent_b - 0.5, 'bollinger')
            signals.append(Signal(
                type=SignalType.STATISTICAL.value,
                ticker=ticker,
                name=name,
                indicator='bollinger',
                value=percent_b,
                threshold=1.0,
                z_score=0,
                level=SignalLevel.ALERT.value if percent_b > 1.2 else SignalLevel.WARNING.value,
                description=f"{name} 볼린저 밴드 상단 돌파 (%B={percent_b:.2f})",
                action_guide=ACTION_GUIDES['bollinger_upper'],
                theory_note=get_theory_note(ticker, 'bollinger'),
                timestamp=timestamp,
                risk_prob=risk_prob,
                risk_level_ml=risk_level,
                risk_model_type=model_type,
            ))
        elif percent_b < 0:  # 하단 이탈
            risk_prob, risk_level, model_type = self.get_risk_prob(ticker, abs(percent_b), 'bollinger')
            signals.append(Signal(
                type=SignalType.STATISTICAL.value,
                ticker=ticker,
                name=name,
                indicator='bollinger',
                value=percent_b,
                threshold=0,
                z_score=0,
                level=SignalLevel.ALERT.value if percent_b < -0.2 else SignalLevel.WARNING.value,
                description=f"{name} 볼린저 밴드 하단 이탈 (%B={percent_b:.2f})",
                action_guide=ACTION_GUIDES['bollinger_lower'],
                theory_note=get_theory_note(ticker, 'bollinger'),
                timestamp=timestamp,
                risk_prob=risk_prob,
                risk_level_ml=risk_level,
                risk_model_type=model_type,
            ))

        # 5. 거래량 이상
        if 'Volume' in df.columns:
            volume = df['Volume']
            if len(volume) >= 20 and volume.iloc[-1] > 0:
                avg_volume = volume.tail(20).mean()
                if avg_volume > 0:
                    volume_ratio = volume.iloc[-1] / avg_volume
                    if volume_ratio >= self.volume_threshold:
                        vol_z = self.calculate_z_score(volume)
                        risk_prob, risk_level, model_type = self.get_risk_prob(ticker, vol_z, 'volume')
                        signals.append(Signal(
                            type=SignalType.STATISTICAL.value,
                            ticker=ticker,
                            name=name,
                            indicator='volume',
                            value=volume_ratio,
                            threshold=self.volume_threshold,
                            z_score=vol_z,
                            level=SignalLevel.ALERT.value if volume_ratio >= 2.5 else SignalLevel.WARNING.value,
                            description=f"{name} 거래량 급증 (평균 대비 {volume_ratio:.1f}배)",
                            action_guide=ACTION_GUIDES['volume_spike'],
                            theory_note=get_theory_note(ticker, 'volume'),
                            timestamp=timestamp,
                            risk_prob=risk_prob,
                            risk_level_ml=risk_level,
                            risk_model_type=model_type,
                        ))

                        # 가격 보합 + 거래량 증가 (방향 전환 신호)
                        if len(returns) >= 5:
                            recent_return = abs(returns.tail(5).mean())
                            if recent_return < 0.5 and volume_ratio >= 1.3:
                                signals.append(Signal(
                                    type=SignalType.EARLY_WARNING.value,
                                    ticker=ticker,
                                    name=name,
                                    indicator='volume_buildup',
                                    value=volume_ratio,
                                    threshold=1.3,
                                    z_score=0,
                                    level=SignalLevel.WARNING.value,
                                    description=f"{name} 가격 보합 + 거래량 증가 (방향 전환 임박 가능)",
                                    action_guide=ACTION_GUIDES['volume_buildup'],
                                    theory_note=get_theory_note(ticker, 'volume_buildup'),
                                    timestamp=timestamp,
                                    risk_prob=risk_prob,
                                    risk_level_ml=risk_level,
                                    risk_model_type=model_type,
                                ))

        return signals

    def _get_level(self, abs_z: float) -> SignalLevel:
        """Z-score 절대값으로 레벨 결정"""
        if abs_z >= self.z_threshold_critical:
            return SignalLevel.CRITICAL
        elif abs_z >= self.z_threshold_alert:
            return SignalLevel.ALERT
        else:
            return SignalLevel.WARNING

    def analyze_all(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """전체 시장 데이터 분석"""
        all_signals = []

        for ticker, df in market_data.items():
            try:
                signals = self.analyze_ticker(ticker, df)
                all_signals.extend(signals)
            except Exception as e:
                print(f"Warning: Error analyzing {ticker}: {e}")
                continue

        # 레벨 순으로 정렬 (CRITICAL > ALERT > WARNING)
        level_order = {'CRITICAL': 0, 'ALERT': 1, 'WARNING': 2}
        all_signals.sort(key=lambda s: level_order.get(s.level, 3))

        return all_signals

    def generate_summary(self, signals: List[Signal]) -> str:
        """시그널 요약 생성"""
        if not signals:
            return "현재 탐지된 이상 시그널이 없습니다. 시장이 안정적입니다."

        critical = [s for s in signals if s.level == 'CRITICAL']
        alert = [s for s in signals if s.level == 'ALERT']
        warning = [s for s in signals if s.level == 'WARNING']

        summary_parts = [
            f"총 {len(signals)}개 시그널 탐지:",
            f"- CRITICAL: {len(critical)}개",
            f"- ALERT: {len(alert)}개",
            f"- WARNING: {len(warning)}개",
        ]

        if critical:
            summary_parts.append("\n주요 CRITICAL 시그널:")
            for s in critical[:3]:
                summary_parts.append(f"  - {s.description}")

        return '\n'.join(summary_parts)


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=== Signal Analyzer Test ===\n")

    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')

    # SPY 시뮬레이션 (상승 추세 + 최근 급등)
    spy_prices = 400 + np.cumsum(np.random.randn(100) * 2) + np.linspace(0, 50, 100)
    spy_prices[-5:] += 20  # 최근 급등
    spy_volume = np.random.randint(50000000, 100000000, 100)
    spy_volume[-1] = 200000000  # 마지막 날 거래량 급증

    spy_df = pd.DataFrame({
        'Close': spy_prices,
        'Volume': spy_volume,
    }, index=dates)

    # GLD 시뮬레이션 (과매수)
    gld_prices = 180 + np.cumsum(np.random.randn(100) * 0.5) + np.linspace(0, 20, 100)
    gld_df = pd.DataFrame({
        'Close': gld_prices,
        'Volume': np.random.randint(5000000, 10000000, 100),
    }, index=dates)

    market_data = {
        'SPY': spy_df,
        'GLD': gld_df,
    }

    # 분석
    analyzer = SignalAnalyzer()
    signals = analyzer.analyze_all(market_data)

    print(f"탐지된 시그널: {len(signals)}개\n")

    for signal in signals[:5]:
        print(f"[{signal.level}] {signal.ticker} - {signal.indicator}")
        print(f"  설명: {signal.description}")
        print(f"  행동 가이드: {signal.action_guide}")
        print(f"  이론적 근거: {signal.theory_note[:80]}...")
        print(f"  리스크: {signal.risk_prob:.1%} ({signal.risk_level_ml})")
        print()

    print("\n=== Summary ===")
    print(analyzer.generate_summary(signals))
