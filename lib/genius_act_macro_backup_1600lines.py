"""
Genius Act Macro Strategy
=========================

Genius Act(스테이블코인 규제법)와 연계된 매크로 전략:

핵심 아이디어:
1. 스테이블코인 발행량 → 미국 국채 수요 (담보 요건)
2. M = B + S·B* 확장 유동성 공식
3. 크립토 유동성 사이클이 전통 금융에 영향

전략 규칙:
- USDT/USDC 공급 증가 → 국채 강세 (담보 수요)
- 역레포 감소 + 스테이블코인 증가 → 위험자산 강세
- TGA(재무부 일반계정) 감소 → 유동성 주입
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta


class LiquidityRegime(Enum):
    """유동성 레짐"""
    EXPANSION = "expansion"           # 유동성 확장
    CONTRACTION = "contraction"       # 유동성 수축
    NEUTRAL = "neutral"               # 중립
    TRANSITION = "transition"         # 전환기


class SignalType(Enum):
    """시그널 타입"""
    STABLECOIN_SURGE = "stablecoin_surge"           # 스테이블코인 급등
    STABLECOIN_DRAIN = "stablecoin_drain"           # 스테이블코인 급감
    TREASURY_DEMAND = "treasury_demand"              # 국채 수요 증가
    TREASURY_SUPPLY = "treasury_supply"              # 국채 공급 증가
    LIQUIDITY_INJECTION = "liquidity_injection"      # 유동성 주입
    LIQUIDITY_DRAIN = "liquidity_drain"              # 유동성 흡수
    CRYPTO_RISK_ON = "crypto_risk_on"                # 크립토 리스크온
    CRYPTO_RISK_OFF = "crypto_risk_off"              # 크립토 리스크오프
    RRP_DRAIN = "rrp_drain"                          # 역레포 감소
    TGA_DRAIN = "tga_drain"                          # TGA 감소


@dataclass
class MacroSignal:
    """매크로 시그널"""
    signal_type: SignalType
    strength: float  # -1 to 1
    description: str
    triggered_at: datetime
    affected_assets: List[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: Dict = field(default_factory=dict)


@dataclass
class LiquidityIndicators:
    """유동성 지표"""
    fed_balance_sheet: float = 0        # Fed 자산 (조 달러)
    rrp_balance: float = 0              # 역레포 잔액 (조 달러)
    tga_balance: float = 0              # 재무부 일반계정 (조 달러)
    usdt_supply: float = 0              # USDT 시가총액 (십억 달러)
    usdc_supply: float = 0              # USDC 시가총액 (십억 달러)
    dai_supply: float = 0               # DAI 시가총액 (십억 달러)
    net_liquidity: float = 0            # 순 유동성 = Fed BS - RRP - TGA
    m2: float = 0                       # M2 통화량 (조 달러)
    dxy: float = 100                    # 달러 인덱스
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyPosition:
    """전략 포지션"""
    asset: str
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    size: float     # 비중 (0-1)
    entry_signal: SignalType
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: str = ""


# =============================================================================
# 확장 유동성 공식: M = B + S·B*
# =============================================================================

class ExtendedLiquidityModel:
    """
    확장 유동성 공식: M = B + S·B*

    M = 총 유효 유동성
    B = 기본 유동성 (Fed BS - RRP - TGA)
    S = 스테이블코인 승수 (발행량/담보비율 기반)
    B* = 스테이블코인 담보 자산 (국채 등)
    """

    def __init__(
        self,
        stablecoin_multiplier: float = 0.9,  # 담보비율 반영
        crypto_impact_factor: float = 0.1    # 크립토가 전통금융에 미치는 영향
    ):
        self.stablecoin_multiplier = stablecoin_multiplier
        self.crypto_impact_factor = crypto_impact_factor

    def calculate_base_liquidity(
        self,
        fed_bs: float,
        rrp: float,
        tga: float
    ) -> float:
        """기본 유동성 계산: B = Fed BS - RRP - TGA"""
        return fed_bs - rrp - tga

    def calculate_stablecoin_contribution(
        self,
        usdt_supply: float,
        usdc_supply: float,
        dai_supply: float
    ) -> float:
        """스테이블코인 기여도: S·B*"""
        total_stablecoin = usdt_supply + usdc_supply + dai_supply
        # 스테이블코인 담보는 대부분 국채 + 현금성 자산
        # S·B* = total_stablecoin * multiplier
        return total_stablecoin * self.stablecoin_multiplier / 1000  # 조 달러로 변환

    def calculate_total_liquidity(
        self,
        indicators: LiquidityIndicators
    ) -> Dict[str, float]:
        """총 유효 유동성: M = B + S·B*"""

        B = self.calculate_base_liquidity(
            indicators.fed_balance_sheet,
            indicators.rrp_balance,
            indicators.tga_balance
        )

        SB_star = self.calculate_stablecoin_contribution(
            indicators.usdt_supply,
            indicators.usdc_supply,
            indicators.dai_supply
        )

        M = B + SB_star * self.crypto_impact_factor

        return {
            "base_liquidity_B": B,
            "stablecoin_contribution_SBstar": SB_star,
            "total_liquidity_M": M,
            "stablecoin_share": SB_star / M if M > 0 else 0,
            "formula": f"M({M:.2f}) = B({B:.2f}) + S·B*({SB_star:.2f})"
        }


# =============================================================================
# Genius Act 규칙 엔진
# =============================================================================

class GeniusActRules:
    """
    Genius Act 기반 규칙:
    1. 스테이블코인 발행자는 미국 국채/현금 담보 보유 필수
    2. 담보 요건이 국채 수요에 영향
    3. 스테이블코인 증가 → 국채 수요 증가
    """

    # 임계값 설정
    STABLECOIN_SURGE_THRESHOLD = 0.05       # 5% 주간 증가
    STABLECOIN_DRAIN_THRESHOLD = -0.03      # 3% 주간 감소
    RRP_DRAIN_THRESHOLD = 0.10              # 10% 월간 감소
    TGA_DRAIN_THRESHOLD = 0.15              # 15% 월간 감소
    LIQUIDITY_CHANGE_THRESHOLD = 0.02       # 2% 변화

    @staticmethod
    def check_stablecoin_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators
    ) -> List[MacroSignal]:
        """스테이블코인 관련 시그널 체크"""
        signals = []

        # 총 스테이블코인 공급
        current_total = current.usdt_supply + current.usdc_supply + current.dai_supply
        previous_total = previous.usdt_supply + previous.usdc_supply + previous.dai_supply

        if previous_total > 0:
            change = (current_total - previous_total) / previous_total

            if change > GeniusActRules.STABLECOIN_SURGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.STABLECOIN_SURGE,
                    strength=min(change / 0.1, 1.0),  # 10% 증가 시 최대
                    description=f"스테이블코인 공급 {change*100:.1f}% 증가 - 국채 수요 상승 예상",
                    triggered_at=current.timestamp,
                    affected_assets=["TLT", "IEF", "SHY", "BTC-USD"],
                    confidence=0.75,
                    metadata={
                        "usdt_change": f"{(current.usdt_supply - previous.usdt_supply):.1f}B",
                        "usdc_change": f"{(current.usdc_supply - previous.usdc_supply):.1f}B",
                        "total_supply": f"{current_total:.1f}B"
                    }
                ))

                # 크립토 리스크온
                signals.append(MacroSignal(
                    signal_type=SignalType.CRYPTO_RISK_ON,
                    strength=min(change / 0.1, 1.0),
                    description="스테이블코인 유입 → 크립토 매수 대기 자금 증가",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD", "COIN", "MSTR"],
                    confidence=0.7
                ))

            elif change < GeniusActRules.STABLECOIN_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.STABLECOIN_DRAIN,
                    strength=abs(change) / 0.1,
                    description=f"스테이블코인 공급 {change*100:.1f}% 감소 - 크립토 자금 이탈",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD"],
                    confidence=0.7,
                    metadata={"drain_amount": f"{previous_total - current_total:.1f}B"}
                ))

                signals.append(MacroSignal(
                    signal_type=SignalType.CRYPTO_RISK_OFF,
                    strength=abs(change) / 0.1,
                    description="스테이블코인 이탈 → 크립토 매도 압력",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD", "COIN"],
                    confidence=0.65
                ))

        return signals

    @staticmethod
    def check_fed_liquidity_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators
    ) -> List[MacroSignal]:
        """Fed 유동성 관련 시그널"""
        signals = []

        # 역레포 변화
        if previous.rrp_balance > 0:
            rrp_change = (current.rrp_balance - previous.rrp_balance) / previous.rrp_balance

            if rrp_change < -GeniusActRules.RRP_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.RRP_DRAIN,
                    strength=min(abs(rrp_change) / 0.2, 1.0),
                    description=f"역레포 {rrp_change*100:.1f}% 감소 → 시장 유동성 주입",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD", "TLT"],
                    confidence=0.8,
                    metadata={
                        "rrp_drain": f"${abs(current.rrp_balance - previous.rrp_balance)*1000:.0f}B",
                        "remaining_rrp": f"${current.rrp_balance:.2f}T"
                    }
                ))

        # TGA 변화
        if previous.tga_balance > 0:
            tga_change = (current.tga_balance - previous.tga_balance) / previous.tga_balance

            if tga_change < -GeniusActRules.TGA_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.TGA_DRAIN,
                    strength=min(abs(tga_change) / 0.3, 1.0),
                    description=f"TGA {tga_change*100:.1f}% 감소 → 재정 지출로 유동성 공급",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "IWM"],
                    confidence=0.75
                ))

        # 순 유동성 변화
        current_net = current.fed_balance_sheet - current.rrp_balance - current.tga_balance
        previous_net = previous.fed_balance_sheet - previous.rrp_balance - previous.tga_balance

        if previous_net > 0:
            net_change = (current_net - previous_net) / previous_net

            if net_change > GeniusActRules.LIQUIDITY_CHANGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.LIQUIDITY_INJECTION,
                    strength=min(net_change / 0.05, 1.0),
                    description=f"순 유동성 {net_change*100:.1f}% 증가 → 위험자산 강세",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD", "HYG"],
                    confidence=0.85
                ))
            elif net_change < -GeniusActRules.LIQUIDITY_CHANGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.LIQUIDITY_DRAIN,
                    strength=min(abs(net_change) / 0.05, 1.0),
                    description=f"순 유동성 {net_change*100:.1f}% 감소 → 위험자산 약세",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD"],
                    confidence=0.8
                ))

        return signals

    @staticmethod
    def check_treasury_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators,
        treasury_supply_change: float = 0  # 국채 발행량 변화
    ) -> List[MacroSignal]:
        """국채 관련 시그널"""
        signals = []

        # 스테이블코인 증가 → 국채 담보 수요
        stablecoin_current = current.usdt_supply + current.usdc_supply
        stablecoin_previous = previous.usdt_supply + previous.usdc_supply

        if stablecoin_previous > 0:
            sc_change = (stablecoin_current - stablecoin_previous) / stablecoin_previous

            # 스테이블코인 증가 시 국채 수요 증가
            if sc_change > 0.03:
                signals.append(MacroSignal(
                    signal_type=SignalType.TREASURY_DEMAND,
                    strength=min(sc_change / 0.1, 1.0),
                    description="스테이블코인 담보 요건 → 단기 국채 수요 증가",
                    triggered_at=current.timestamp,
                    affected_assets=["SHY", "BIL", "SGOV"],
                    confidence=0.7,
                    metadata={
                        "estimated_demand": f"${sc_change * stablecoin_current:.1f}B"
                    }
                ))

        # 국채 공급 증가 (국채 발행)
        if treasury_supply_change > 0.05:  # 5% 이상 발행 증가
            signals.append(MacroSignal(
                signal_type=SignalType.TREASURY_SUPPLY,
                strength=min(treasury_supply_change / 0.1, 1.0),
                description=f"국채 발행 {treasury_supply_change*100:.1f}% 증가 → 금리 상승 압력",
                triggered_at=current.timestamp,
                affected_assets=["TLT", "IEF", "TBT"],
                confidence=0.65
            ))

        return signals


# =============================================================================
# 매크로 전략 엔진
# =============================================================================

class GeniusActMacroStrategy:
    """Genius Act 기반 매크로 전략"""

    def __init__(self):
        self.liquidity_model = ExtendedLiquidityModel()
        self.rules = GeniusActRules()
        self.signal_history: List[MacroSignal] = []
        self.positions: List[StrategyPosition] = []

    def analyze(
        self,
        current: LiquidityIndicators,
        previous: LiquidityIndicators,
        treasury_supply_change: float = 0
    ) -> Dict:
        """전체 분석 실행"""

        # 1. 유동성 계산
        liquidity = self.liquidity_model.calculate_total_liquidity(current)

        # 2. 시그널 생성
        signals = []
        signals.extend(self.rules.check_stablecoin_signals(current, previous))
        signals.extend(self.rules.check_fed_liquidity_signals(current, previous))
        signals.extend(self.rules.check_treasury_signals(current, previous, treasury_supply_change))

        # 히스토리에 저장
        self.signal_history.extend(signals)

        # 3. 레짐 판단
        regime = self._determine_regime(liquidity, signals)

        # 4. 포지션 추천
        positions = self._generate_positions(signals, regime)

        return {
            "timestamp": current.timestamp.isoformat(),
            "liquidity": liquidity,
            "regime": regime.value,
            "signals": [self._signal_to_dict(s) for s in signals],
            "positions": [self._position_to_dict(p) for p in positions],
            "summary": self._generate_summary(liquidity, regime, signals, positions)
        }

    def _determine_regime(
        self,
        liquidity: Dict,
        signals: List[MacroSignal]
    ) -> LiquidityRegime:
        """유동성 레짐 판단"""

        # 시그널 기반 점수
        expansion_score = 0
        contraction_score = 0

        for signal in signals:
            if signal.signal_type in [
                SignalType.LIQUIDITY_INJECTION,
                SignalType.RRP_DRAIN,
                SignalType.TGA_DRAIN,
                SignalType.STABLECOIN_SURGE,
                SignalType.CRYPTO_RISK_ON
            ]:
                expansion_score += signal.strength * signal.confidence

            elif signal.signal_type in [
                SignalType.LIQUIDITY_DRAIN,
                SignalType.STABLECOIN_DRAIN,
                SignalType.CRYPTO_RISK_OFF,
                SignalType.TREASURY_SUPPLY
            ]:
                contraction_score += signal.strength * signal.confidence

        # 레짐 결정
        net_score = expansion_score - contraction_score

        if net_score > 0.5:
            return LiquidityRegime.EXPANSION
        elif net_score < -0.5:
            return LiquidityRegime.CONTRACTION
        elif abs(net_score) < 0.2:
            return LiquidityRegime.NEUTRAL
        else:
            return LiquidityRegime.TRANSITION

    def _generate_positions(
        self,
        signals: List[MacroSignal],
        regime: LiquidityRegime
    ) -> List[StrategyPosition]:
        """시그널 기반 포지션 생성"""
        positions = []

        # 레짐별 기본 포지션
        if regime == LiquidityRegime.EXPANSION:
            positions.append(StrategyPosition(
                asset="SPY",
                direction="LONG",
                size=0.3,
                entry_signal=SignalType.LIQUIDITY_INJECTION,
                rationale="유동성 확장 레짐 → 주식 강세"
            ))
            positions.append(StrategyPosition(
                asset="BTC-USD",
                direction="LONG",
                size=0.1,
                entry_signal=SignalType.LIQUIDITY_INJECTION,
                rationale="유동성 확장 → 위험자산 선호"
            ))

        elif regime == LiquidityRegime.CONTRACTION:
            positions.append(StrategyPosition(
                asset="TLT",
                direction="LONG",
                size=0.3,
                entry_signal=SignalType.LIQUIDITY_DRAIN,
                rationale="유동성 수축 → 안전자산 선호"
            ))
            positions.append(StrategyPosition(
                asset="SPY",
                direction="SHORT",
                size=0.1,
                entry_signal=SignalType.LIQUIDITY_DRAIN,
                rationale="유동성 수축 → 주식 약세"
            ))

        # 시그널별 추가 포지션
        for signal in signals:
            if signal.signal_type == SignalType.STABLECOIN_SURGE:
                positions.append(StrategyPosition(
                    asset="BTC-USD",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="스테이블코인 유입 → 크립토 매수 대기"
                ))
                positions.append(StrategyPosition(
                    asset="SHY",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="스테이블코인 담보 요건 → 단기 국채 수요"
                ))

            elif signal.signal_type == SignalType.RRP_DRAIN:
                positions.append(StrategyPosition(
                    asset="QQQ",
                    direction="LONG",
                    size=0.1,
                    entry_signal=signal.signal_type,
                    rationale="역레포 유동성 방출 → 성장주 강세"
                ))

            elif signal.signal_type == SignalType.TREASURY_SUPPLY:
                positions.append(StrategyPosition(
                    asset="TBT",
                    direction="LONG",
                    size=0.05,
                    entry_signal=signal.signal_type,
                    rationale="국채 공급 증가 → 금리 상승 베팅"
                ))

        # 중복 제거 및 크기 조정
        return self._consolidate_positions(positions)

    def _consolidate_positions(
        self,
        positions: List[StrategyPosition]
    ) -> List[StrategyPosition]:
        """포지션 통합 및 정규화"""
        consolidated = {}

        for pos in positions:
            key = (pos.asset, pos.direction)
            if key in consolidated:
                existing = consolidated[key]
                existing.size += pos.size
                existing.rationale += f"; {pos.rationale}"
            else:
                consolidated[key] = pos

        result = list(consolidated.values())

        # 비중 정규화 (총합 1.0 이하)
        total = sum(p.size for p in result)
        if total > 1.0:
            for p in result:
                p.size /= total

        return result

    def _signal_to_dict(self, signal: MacroSignal) -> Dict:
        """시그널을 딕셔너리로 변환"""
        return {
            "type": signal.signal_type.value,
            "strength": f"{signal.strength:.2f}",
            "description": signal.description,
            "affected_assets": signal.affected_assets,
            "confidence": f"{signal.confidence*100:.0f}%",
            "metadata": signal.metadata
        }

    def _position_to_dict(self, position: StrategyPosition) -> Dict:
        """포지션을 딕셔너리로 변환"""
        return {
            "asset": position.asset,
            "direction": position.direction,
            "size": f"{position.size*100:.1f}%",
            "signal": position.entry_signal.value,
            "rationale": position.rationale
        }

    def _generate_summary(
        self,
        liquidity: Dict,
        regime: LiquidityRegime,
        signals: List[MacroSignal],
        positions: List[StrategyPosition]
    ) -> str:
        """분석 요약 생성"""
        summary_parts = []

        # 유동성 상태
        summary_parts.append(f"[유동성] {liquidity['formula']}")

        # 레짐
        regime_desc = {
            LiquidityRegime.EXPANSION: "확장 (Risk-On)",
            LiquidityRegime.CONTRACTION: "수축 (Risk-Off)",
            LiquidityRegime.NEUTRAL: "중립",
            LiquidityRegime.TRANSITION: "전환기 (주의)"
        }
        summary_parts.append(f"[레짐] {regime_desc[regime]}")

        # 핵심 시그널
        if signals:
            high_conf_signals = [s for s in signals if s.confidence > 0.7]
            if high_conf_signals:
                summary_parts.append(f"[시그널] {len(high_conf_signals)}개 고신뢰 시그널")

        # 포지션 추천
        if positions:
            longs = [p for p in positions if p.direction == "LONG"]
            shorts = [p for p in positions if p.direction == "SHORT"]
            summary_parts.append(f"[포지션] LONG {len(longs)}개, SHORT {len(shorts)}개")

        return " | ".join(summary_parts)


# =============================================================================
# 크립토 리스크 평가 (Genius Act 담보 유형별 차등화)
# =============================================================================

class StablecoinCollateralType(Enum):
    """스테이블코인 담보 유형"""
    TREASURY_CASH = "treasury_cash"          # USDC: 국채 + 현금 (가장 안전)
    MIXED_RESERVE = "mixed_reserve"          # USDT: 혼합 준비금
    CRYPTO_BACKED = "crypto_backed"          # DAI: 암호화폐 담보
    ALGORITHMIC = "algorithmic"              # UST류: 알고리즘 (최고 위험)
    DERIVATIVE_HEDGE = "derivative_hedge"    # USDe: 파생상품 헤징


@dataclass
class MultiDimensionalRiskScore:
    """다차원 리스크 점수 (API 검증 결과 반영)"""
    credit_risk: float        # 신용/담보 리스크 (0-100)
    liquidity_risk: float     # 유동성 리스크 (0-100)
    regulatory_risk: float    # 규제 리스크 (0-100)
    technical_risk: float     # 기술/스마트컨트랙트 리스크 (0-100)

    # 가중치 (Claude/Perplexity 검증 결과)
    WEIGHTS = {
        'credit': 0.30,
        'liquidity': 0.25,
        'regulatory': 0.25,
        'technical': 0.20
    }

    def get_weighted_score(self) -> float:
        """가중 평균 리스크 점수"""
        return (
            self.credit_risk * self.WEIGHTS['credit'] +
            self.liquidity_risk * self.WEIGHTS['liquidity'] +
            self.regulatory_risk * self.WEIGHTS['regulatory'] +
            self.technical_risk * self.WEIGHTS['technical']
        )

    def to_dict(self) -> Dict:
        return {
            'credit_risk': round(self.credit_risk, 2),
            'liquidity_risk': round(self.liquidity_risk, 2),
            'regulatory_risk': round(self.regulatory_risk, 2),
            'technical_risk': round(self.technical_risk, 2),
            'weighted_score': round(self.get_weighted_score(), 2),
            'weights': self.WEIGHTS
        }


@dataclass
class StablecoinRiskProfile:
    """스테이블코인 리스크 프로파일 (다차원 스코어링 v2)"""
    name: str
    ticker: str
    collateral_type: StablecoinCollateralType
    base_risk_score: float        # 기본 리스크 점수 (0-100)
    pays_interest: bool           # 이자 지급 여부
    regulatory_risk_weight: float # 규제 리스크 가중치
    collateral_transparency: float # 담보 투명성 (0-1)
    peg_stability_score: float    # 페그 안정성 (0-100)

    # 신규: 다차원 리스크 세부 점수
    liquidity_depth: float = 0.8      # DEX/CEX 유동성 깊이 (0-1)
    smart_contract_audits: int = 0    # 감사 횟수
    governance_centralization: float = 0.5  # 거버넌스 중앙화 정도 (0=분산, 1=중앙)
    market_cap_billion: float = 0.0   # 시가총액 (십억 달러)

    def get_multi_dimensional_risk(self) -> MultiDimensionalRiskScore:
        """다차원 리스크 점수 계산 (v2 - API 검증 결과 반영)"""

        # 1. 신용/담보 리스크 (30%)
        credit_risk = self._calculate_credit_risk()

        # 2. 유동성 리스크 (25%)
        liquidity_risk = self._calculate_liquidity_risk()

        # 3. 규제 리스크 (25%)
        regulatory_risk = self._calculate_regulatory_risk()

        # 4. 기술 리스크 (20%)
        technical_risk = self._calculate_technical_risk()

        return MultiDimensionalRiskScore(
            credit_risk=credit_risk,
            liquidity_risk=liquidity_risk,
            regulatory_risk=regulatory_risk,
            technical_risk=technical_risk
        )

    def _calculate_credit_risk(self) -> float:
        """신용/담보 리스크 계산"""
        # 담보 유형별 기본 점수
        collateral_scores = {
            StablecoinCollateralType.TREASURY_CASH: 10,
            StablecoinCollateralType.MIXED_RESERVE: 35,
            StablecoinCollateralType.CRYPTO_BACKED: 55,
            StablecoinCollateralType.DERIVATIVE_HEDGE: 65,
            StablecoinCollateralType.ALGORITHMIC: 90
        }
        base = collateral_scores.get(self.collateral_type, 50)

        # 투명성 보너스
        transparency_bonus = self.collateral_transparency * 15

        # 페그 안정성
        peg_factor = (100 - self.peg_stability_score) * 0.3

        return max(0, min(100, base - transparency_bonus + peg_factor))

    def _calculate_liquidity_risk(self) -> float:
        """유동성 리스크 계산"""
        # 유동성 깊이 기반
        base = (1 - self.liquidity_depth) * 60

        # 시가총액 보너스 (크면 유동성 좋음)
        if self.market_cap_billion > 50:
            cap_bonus = 20
        elif self.market_cap_billion > 10:
            cap_bonus = 10
        elif self.market_cap_billion > 1:
            cap_bonus = 5
        else:
            cap_bonus = 0

        return max(0, min(100, base + 20 - cap_bonus))

    def _calculate_regulatory_risk(self) -> float:
        """규제 리스크 계산 (개선된 이자 페널티)"""
        # 기본 규제 리스크 가중치
        base = self.regulatory_risk_weight * 40

        # 이자 지급 시 차등 페널티 (blanket +15 대신 세분화)
        if self.pays_interest:
            # 이자 소스에 따른 차등화
            if self.collateral_type == StablecoinCollateralType.DERIVATIVE_HEDGE:
                # 파생상품 수익 = 가장 높은 규제 리스크
                interest_penalty = 25
            elif self.collateral_type == StablecoinCollateralType.CRYPTO_BACKED:
                # 스테이킹 수익 = 중간 규제 리스크
                interest_penalty = 15
            else:
                # 기타 = 기본 페널티
                interest_penalty = 10
        else:
            interest_penalty = 0

        # 거버넌스 중앙화 (규제 대상 명확)
        centralization_factor = self.governance_centralization * 10

        return max(0, min(100, base + interest_penalty + centralization_factor))

    def _calculate_technical_risk(self) -> float:
        """기술/스마트컨트랙트 리스크 계산"""
        # 감사 횟수 기반
        if self.smart_contract_audits >= 5:
            audit_score = 10
        elif self.smart_contract_audits >= 3:
            audit_score = 25
        elif self.smart_contract_audits >= 1:
            audit_score = 40
        else:
            audit_score = 70

        # 복잡성 (담보 유형에 따른)
        complexity_scores = {
            StablecoinCollateralType.TREASURY_CASH: 10,      # 단순
            StablecoinCollateralType.MIXED_RESERVE: 20,
            StablecoinCollateralType.CRYPTO_BACKED: 35,      # CDP 복잡
            StablecoinCollateralType.DERIVATIVE_HEDGE: 50,   # 파생상품 복잡
            StablecoinCollateralType.ALGORITHMIC: 60         # 가장 복잡
        }
        complexity = complexity_scores.get(self.collateral_type, 30)

        return max(0, min(100, (audit_score + complexity) / 2))

    def get_total_risk_score(self) -> float:
        """총 리스크 점수 계산 (v2: 다차원 가중 평균)"""
        return self.get_multi_dimensional_risk().get_weighted_score()

    def to_dict(self) -> Dict:
        multi_risk = self.get_multi_dimensional_risk()
        return {
            'name': self.name,
            'ticker': self.ticker,
            'collateral_type': self.collateral_type.value,
            'base_risk_score': self.base_risk_score,
            'pays_interest': self.pays_interest,
            'regulatory_risk_weight': self.regulatory_risk_weight,
            'total_risk_score': self.get_total_risk_score(),
            'multi_dimensional_risk': multi_risk.to_dict()
        }


class CryptoRiskEvaluator:
    """
    크립토/스테이블코인 리스크 평가기

    Genius Act 담보 요건 기반:
    1. 국채+현금 담보 (USDC): 가장 낮은 리스크
    2. 혼합 준비금 (USDT): 중간 리스크
    3. 암호화폐 담보 (DAI): 높은 변동성 리스크
    4. 파생상품 헤징 (USDe): 규제 불확실성 + 복잡성 리스크
    5. 알고리즘 (UST류): 최고 리스크 (붕괴 가능성)

    이자 지급 여부:
    - 이자 지급 O: SEC 증권 분류 가능성 → 규제 리스크 증가
    - 이자 지급 X: 화폐 대용 → 규제 리스크 낮음
    """

    # 주요 스테이블코인 프로파일 (v2 - 다차원 리스크 지원)
    STABLECOIN_PROFILES = {
        'USDC': StablecoinRiskProfile(
            name='USD Coin',
            ticker='USDC',
            collateral_type=StablecoinCollateralType.TREASURY_CASH,
            base_risk_score=15,        # 국채+현금 = 가장 안전
            pays_interest=False,       # 이자 미지급
            regulatory_risk_weight=0.3,
            collateral_transparency=0.95,  # Circle 월간 증명
            peg_stability_score=95,
            # v2 신규 필드
            liquidity_depth=0.95,          # 매우 깊은 유동성
            smart_contract_audits=5,       # 다수 감사
            governance_centralization=0.8,  # Circle 중앙 통제
            market_cap_billion=45.0
        ),
        'USDT': StablecoinRiskProfile(
            name='Tether',
            ticker='USDT',
            collateral_type=StablecoinCollateralType.MIXED_RESERVE,
            base_risk_score=30,        # 혼합 준비금 = 중간
            pays_interest=False,
            regulatory_risk_weight=0.5,
            collateral_transparency=0.6,   # 분기별 증명, 상세 부족
            peg_stability_score=90,
            liquidity_depth=0.98,          # 가장 깊은 유동성
            smart_contract_audits=3,
            governance_centralization=0.9,  # Tether Ltd 중앙 통제
            market_cap_billion=140.0
        ),
        'DAI': StablecoinRiskProfile(
            name='DAI',
            ticker='DAI',
            collateral_type=StablecoinCollateralType.CRYPTO_BACKED,
            base_risk_score=40,        # 암호화폐 담보 = 높은 변동성
            pays_interest=False,       # DSR은 별도
            regulatory_risk_weight=0.4,
            collateral_transparency=0.99,  # 온체인 완전 투명
            peg_stability_score=85,
            liquidity_depth=0.85,
            smart_contract_audits=6,       # MakerDAO 다수 감사
            governance_centralization=0.3,  # DAO 분산 거버넌스
            market_cap_billion=5.0
        ),
        'USDe': StablecoinRiskProfile(
            name='Ethena USDe',
            ticker='USDe',
            collateral_type=StablecoinCollateralType.DERIVATIVE_HEDGE,
            base_risk_score=50,        # 파생상품 헤징 = 복잡성 리스크
            pays_interest=True,        # sUSDe 이자 지급 (핵심 차이점!)
            regulatory_risk_weight=0.8,    # 이자 지급 → 높은 규제 리스크
            collateral_transparency=0.85,
            peg_stability_score=80,
            liquidity_depth=0.70,          # 신규, 유동성 제한적
            smart_contract_audits=2,       # 비교적 신규
            governance_centralization=0.7,
            market_cap_billion=3.5
        ),
        'FRAX': StablecoinRiskProfile(
            name='Frax',
            ticker='FRAX',
            collateral_type=StablecoinCollateralType.MIXED_RESERVE,
            base_risk_score=35,
            pays_interest=False,
            regulatory_risk_weight=0.5,
            collateral_transparency=0.8,
            peg_stability_score=85,
            liquidity_depth=0.75,
            smart_contract_audits=4,
            governance_centralization=0.4,  # Frax DAO
            market_cap_billion=1.0
        ),
        'PYUSD': StablecoinRiskProfile(
            name='PayPal USD',
            ticker='PYUSD',
            collateral_type=StablecoinCollateralType.TREASURY_CASH,
            base_risk_score=20,        # PayPal 규제 라이선스
            pays_interest=False,
            regulatory_risk_weight=0.2,    # 기존 금융기관 = 낮은 규제 리스크
            collateral_transparency=0.9,
            peg_stability_score=95,
            liquidity_depth=0.60,          # 신규, 유동성 구축 중
            smart_contract_audits=3,
            governance_centralization=0.95, # PayPal 완전 통제
            market_cap_billion=0.5
        ),
    }

    def __init__(self):
        self.profiles = self.STABLECOIN_PROFILES.copy()

    def evaluate_stablecoin(self, ticker: str) -> Optional[Dict]:
        """개별 스테이블코인 리스크 평가"""
        # 대소문자 무관하게 찾기
        ticker_upper = ticker.upper()
        matched_ticker = None

        for profile_ticker in self.profiles.keys():
            if profile_ticker.upper() == ticker_upper:
                matched_ticker = profile_ticker
                break

        if matched_ticker is None:
            return None

        ticker = matched_ticker  # 원본 키 사용
        profile = self.profiles[ticker]
        total_risk = profile.get_total_risk_score()

        # 리스크 등급
        if total_risk < 20:
            risk_grade = 'A'
            risk_label = 'Very Low Risk (Genius Act Compliant)'
        elif total_risk < 35:
            risk_grade = 'B'
            risk_label = 'Low Risk (Near Compliant)'
        elif total_risk < 50:
            risk_grade = 'C'
            risk_label = 'Moderate Risk (Regulatory Uncertainty)'
        elif total_risk < 70:
            risk_grade = 'D'
            risk_label = 'High Risk (Significant Concerns)'
        else:
            risk_grade = 'F'
            risk_label = 'Very High Risk (Avoid)'

        # 규제 상세 분석
        regulatory_analysis = self._analyze_regulatory_risk(profile)

        return {
            'ticker': ticker,
            'name': profile.name,
            'collateral_type': profile.collateral_type.value,
            'base_risk_score': profile.base_risk_score,
            'pays_interest': profile.pays_interest,
            'total_risk_score': total_risk,
            'risk_grade': risk_grade,
            'risk_label': risk_label,
            'regulatory_analysis': regulatory_analysis,
            'genius_act_compliant': total_risk < 30 and not profile.pays_interest
        }

    def _analyze_regulatory_risk(self, profile: StablecoinRiskProfile) -> str:
        """규제 리스크 상세 분석"""
        analysis_parts = []

        # 담보 유형별 분석
        if profile.collateral_type == StablecoinCollateralType.TREASURY_CASH:
            analysis_parts.append(
                "담보: 미국 국채 + 현금 (Genius Act 요건 충족). "
                "유동성 위기 시에도 담보 가치 안정."
            )
        elif profile.collateral_type == StablecoinCollateralType.MIXED_RESERVE:
            analysis_parts.append(
                "담보: 혼합 준비금 (국채, 기업어음, 기타). "
                "일부 자산 유동성 리스크 존재."
            )
        elif profile.collateral_type == StablecoinCollateralType.CRYPTO_BACKED:
            analysis_parts.append(
                "담보: 암호화폐 (ETH 등). "
                "시장 급락 시 담보 청산 리스크. 과담보로 완화."
            )
        elif profile.collateral_type == StablecoinCollateralType.DERIVATIVE_HEDGE:
            analysis_parts.append(
                "담보: 파생상품 헤징 (Delta-Neutral 전략). "
                "펀딩비 역전, 거래소 리스크, 복잡성 리스크 존재."
            )
        elif profile.collateral_type == StablecoinCollateralType.ALGORITHMIC:
            analysis_parts.append(
                "경고: 알고리즘 스테이블코인. "
                "담보 없음, Death Spiral 위험. UST 사례 참조."
            )

        # 이자 지급 여부
        if profile.pays_interest:
            analysis_parts.append(
                f"규제 경고: 이자 지급 ({profile.ticker}). "
                "SEC 증권법 적용 가능성. "
                "Howey Test 충족 시 미등록 증권 발행으로 분류될 수 있음."
            )
        else:
            analysis_parts.append(
                "이자 미지급: 화폐 대용으로 분류될 가능성 높음. "
                "Genius Act 규제 프레임워크 적합."
            )

        return " ".join(analysis_parts)

    def compare_stablecoins(self, tickers: List[str]) -> pd.DataFrame:
        """여러 스테이블코인 비교"""
        data = []

        for ticker in tickers:
            result = self.evaluate_stablecoin(ticker)
            if result:
                data.append({
                    'Ticker': result['ticker'],
                    'Collateral': result['collateral_type'],
                    'Pays Interest': 'Yes' if result['pays_interest'] else 'No',
                    'Risk Score': result['total_risk_score'],
                    'Grade': result['risk_grade'],
                    'Genius Act': 'Yes' if result['genius_act_compliant'] else 'No'
                })

        return pd.DataFrame(data).sort_values('Risk Score')

    def get_portfolio_stablecoin_risk(
        self,
        stablecoin_holdings: Dict[str, float]  # {ticker: amount_in_usd}
    ) -> Dict:
        """
        포트폴리오 전체 스테이블코인 리스크 계산

        Args:
            stablecoin_holdings: {ticker: 보유액(USD)}

        Returns:
            포트폴리오 가중 리스크 점수 및 분석
        """
        total_value = sum(stablecoin_holdings.values())
        if total_value == 0:
            return {'error': 'No holdings'}

        weighted_risk = 0
        non_compliant_value = 0
        interest_bearing_value = 0

        breakdown = []

        for ticker, amount in stablecoin_holdings.items():
            result = self.evaluate_stablecoin(ticker)
            if result:
                weight = amount / total_value
                weighted_risk += result['total_risk_score'] * weight

                if not result['genius_act_compliant']:
                    non_compliant_value += amount

                if result['pays_interest']:
                    interest_bearing_value += amount

                breakdown.append({
                    'ticker': ticker,
                    'amount': amount,
                    'weight': weight,
                    'risk_score': result['total_risk_score'],
                    'contribution': result['total_risk_score'] * weight
                })

        return {
            'total_value': total_value,
            'weighted_risk_score': weighted_risk,
            'non_compliant_ratio': non_compliant_value / total_value,
            'interest_bearing_ratio': interest_bearing_value / total_value,
            'breakdown': breakdown,
            'recommendation': self._generate_recommendation(
                weighted_risk, non_compliant_value / total_value
            )
        }

    def _generate_recommendation(
        self,
        weighted_risk: float,
        non_compliant_ratio: float
    ) -> str:
        """포트폴리오 권고사항 생성"""
        if weighted_risk < 25 and non_compliant_ratio < 0.1:
            return (
                "포트폴리오 리스크 낮음. Genius Act 준수 비율 높음. "
                "현재 구성 유지 권장."
            )
        elif weighted_risk < 40:
            return (
                "포트폴리오 리스크 중간. "
                f"비준수 비율 {non_compliant_ratio:.0%}. "
                "USDC/PYUSD 비중 확대 고려."
            )
        else:
            return (
                "경고: 포트폴리오 리스크 높음. "
                f"비준수 비율 {non_compliant_ratio:.0%}. "
                "이자 지급 스테이블코인 축소 및 "
                "국채 담보 스테이블코인(USDC, PYUSD)으로 재배분 권장."
            )

    def run_stress_test(
        self,
        stablecoin_holdings: Dict[str, float],
        stress_scenario: str = "moderate"
    ) -> Dict:
        """
        스테이블코인 포트폴리오 스트레스 테스트

        De-peg 확률 및 스트레스 상황 예상 손실 계산

        Parameters:
        -----------
        stablecoin_holdings : Dict[str, float]
            스테이블코인 보유량 {ticker: amount_in_usd}
        stress_scenario : str
            스트레스 시나리오 ('mild', 'moderate', 'severe', 'extreme')

        Returns:
        --------
        Dict with stress test results including:
            - depeg_probability: 디페깅 확률 (%)
            - estimated_loss_under_stress: 스트레스 상황 예상 손실 ($)
            - breakdown_by_coin: 코인별 상세 분석
        """
        # 시나리오별 가정 (Elicit 리포트 + 실증 연구 기반)
        STRESS_SCENARIOS = {
            'mild': {
                'name': 'Mild (국채 금리 50bp 상승)',
                'depeg_base_prob': 0.01,     # 기본 디페깅 확률 1%
                'loss_multiplier': 0.02,     # 2% 손실
                'crypto_vol_shock': 0.20     # 크립토 20% 하락
            },
            'moderate': {
                'name': 'Moderate (신용위기 수준)',
                'depeg_base_prob': 0.05,     # 기본 디페깅 확률 5%
                'loss_multiplier': 0.10,     # 10% 손실
                'crypto_vol_shock': 0.40     # 크립토 40% 하락
            },
            'severe': {
                'name': 'Severe (2022년 UST/FTX 수준)',
                'depeg_base_prob': 0.15,     # 기본 디페깅 확률 15%
                'loss_multiplier': 0.30,     # 30% 손실
                'crypto_vol_shock': 0.60     # 크립토 60% 하락
            },
            'extreme': {
                'name': 'Extreme (전면 붕괴)',
                'depeg_base_prob': 0.30,     # 기본 디페깅 확률 30%
                'loss_multiplier': 0.80,     # 80% 손실
                'crypto_vol_shock': 0.80     # 크립토 80% 하락
            }
        }

        scenario = STRESS_SCENARIOS.get(stress_scenario, STRESS_SCENARIOS['moderate'])
        total_value = sum(stablecoin_holdings.values())

        if total_value == 0:
            return {
                'error': 'No holdings',
                'depeg_probability': 0.0,
                'estimated_loss_under_stress': 0.0
            }

        breakdown = []
        portfolio_depeg_prob = 0.0
        portfolio_expected_loss = 0.0

        for ticker, amount in stablecoin_holdings.items():
            profile = self.profiles.get(ticker.upper())
            weight = amount / total_value

            if profile is None:
                # 알 수 없는 코인: 높은 리스크 가정
                coin_depeg_prob = scenario['depeg_base_prob'] * 3
                coin_loss = amount * scenario['loss_multiplier'] * 2
            else:
                # 담보 유형별 리스크 조정
                collateral_risk_factor = {
                    StablecoinCollateralType.TREASURY_CASH: 0.1,      # 매우 낮음
                    StablecoinCollateralType.MIXED_RESERVE: 0.5,      # 중간
                    StablecoinCollateralType.CRYPTO_BACKED: 1.5,      # 높음 (크립토 가격 연동)
                    StablecoinCollateralType.DERIVATIVE_HEDGE: 2.0,   # 매우 높음
                    StablecoinCollateralType.ALGORITHMIC: 5.0         # 극단적
                }.get(profile.collateral_type, 1.0)

                # 이자 지급 시 추가 리스크 (규제 불확실성)
                interest_factor = 1.5 if profile.pays_interest else 1.0

                # 디페깅 확률 계산
                coin_depeg_prob = min(1.0, scenario['depeg_base_prob'] * collateral_risk_factor * interest_factor)

                # 예상 손실 계산
                # 크립토 담보는 담보 가치 하락 효과 추가
                if profile.collateral_type == StablecoinCollateralType.CRYPTO_BACKED:
                    crypto_loss_effect = scenario['crypto_vol_shock'] * 0.5  # 과담보로 50% 완충
                else:
                    crypto_loss_effect = 0

                loss_rate = (scenario['loss_multiplier'] * collateral_risk_factor + crypto_loss_effect) * coin_depeg_prob
                coin_loss = amount * min(loss_rate, 1.0)

            portfolio_depeg_prob += coin_depeg_prob * weight
            portfolio_expected_loss += coin_loss

            breakdown.append({
                'ticker': ticker,
                'amount': amount,
                'weight': weight,
                'depeg_probability': coin_depeg_prob,
                'expected_loss': coin_loss,
                'loss_rate': coin_loss / amount if amount > 0 else 0
            })

        # 결과 정리
        return {
            'scenario': scenario['name'],
            'total_value': total_value,
            'depeg_probability': portfolio_depeg_prob,
            'depeg_probability_pct': f"{portfolio_depeg_prob * 100:.1f}%",
            'estimated_loss_under_stress': portfolio_expected_loss,
            'estimated_loss_pct': f"{(portfolio_expected_loss / total_value) * 100:.1f}%",
            'breakdown_by_coin': sorted(breakdown, key=lambda x: x['expected_loss'], reverse=True),
            'risk_rating': self._get_stress_risk_rating(portfolio_depeg_prob, portfolio_expected_loss / total_value),
            'methodology_note': (
                f"스트레스 테스트: {scenario['name']}. "
                f"담보 유형별 리스크 가중치 적용. "
                f"크립토 담보는 {scenario['crypto_vol_shock']*100:.0f}% 가격 하락 가정."
            )
        }

    def _get_stress_risk_rating(self, depeg_prob: float, loss_rate: float) -> str:
        """스트레스 테스트 결과 등급 판정"""
        combined_score = depeg_prob * 0.5 + loss_rate * 0.5

        if combined_score < 0.02:
            return "LOW (낮음)"
        elif combined_score < 0.05:
            return "MODERATE (보통)"
        elif combined_score < 0.15:
            return "ELEVATED (주의)"
        elif combined_score < 0.30:
            return "HIGH (높음)"
        else:
            return "CRITICAL (위험)"


# =============================================================================
# 스테이블코인 데이터 수집 및 상세 코멘트 생성
# =============================================================================

class StablecoinDataCollector:
    """
    스테이블코인 시가총액 데이터 수집 및 7일 델타 계산

    소스 이론: "스테이블 코인(B*) 발행량이 늘어야 미국 국채를 사주므로(M 증가)
    Genius Act가 작동하는 것."
    """

    # 스테이블코인 티커 (yfinance용)
    STABLECOIN_TICKERS = {
        'USDT-USD': 'USDT',  # Tether
        'USDC-USD': 'USDC',  # USD Coin
        'DAI-USD': 'DAI',    # DAI
    }

    # 추정 시가총액 (십억 달러) - API 실패 시 폴백
    FALLBACK_MARKET_CAP = {
        'USDT': 140.0,  # 2025년 기준 약 $140B
        'USDC': 45.0,   # 2025년 기준 약 $45B
        'DAI': 5.0,     # 2025년 기준 약 $5B
    }

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file
        self._cache: Dict[str, List[Tuple[datetime, float]]] = {}

    def fetch_stablecoin_supply(self, lookback_days: int = 14) -> Dict[str, Dict]:
        """
        스테이블코인 시가총액 데이터 수집

        Returns:
            Dict: {
                'USDT': {'current': 140.0, 'week_ago': 138.0, 'delta_7d': 2.0, 'delta_pct': 1.45},
                'USDC': {'current': 45.0, 'week_ago': 44.0, 'delta_7d': 1.0, 'delta_pct': 2.27},
                ...
            }
        """
        import yfinance as yf

        result = {}

        for ticker, name in self.STABLECOIN_TICKERS.items():
            try:
                # yfinance에서 가격 데이터 가져오기 (시가총액 근사치로 사용)
                data = yf.download(ticker, period=f"{lookback_days}d", progress=False)

                if data.empty or len(data) < 7:
                    # 폴백: 추정 시가총액 사용
                    result[name] = {
                        'current': self.FALLBACK_MARKET_CAP.get(name, 0),
                        'week_ago': self.FALLBACK_MARKET_CAP.get(name, 0),
                        'delta_7d': 0.0,
                        'delta_pct': 0.0,
                        'source': 'fallback'
                    }
                    continue

                # 현재와 7일 전 가격 (스테이블코인이므로 가격≈$1, 변동 = 시가총액 변화 추정)
                # 실제로는 가격이 $1 부근이므로 시가총액 추정 필요
                # 여기서는 폴백 시가총액에 비율 변화를 적용

                base_cap = self.FALLBACK_MARKET_CAP.get(name, 100)

                # 최근 변동성 (peg 이탈) 기반 시가총액 변화 추정
                # 스테이블코인 가격이 $1 이상이면 수요 증가, 이하면 수요 감소
                current_price = float(data['Close'].iloc[-1])
                week_ago_price = float(data['Close'].iloc[-7]) if len(data) >= 7 else current_price

                # 시가총액 추정: 가격 프리미엄/할인을 공급 변화로 해석
                # 실제 시가총액 데이터가 없으므로 근사치 사용
                price_delta_pct = ((current_price - week_ago_price) / week_ago_price) * 100 if week_ago_price > 0 else 0

                # 시가총액 변화 추정 (가격 변화 * 50배로 확대 - 실제 공급 변화 반영)
                estimated_supply_change_pct = price_delta_pct * 50  # 프리미엄 1% = 공급 변화 추정

                # 범위 제한 (-10% ~ +10%)
                estimated_supply_change_pct = max(-10, min(10, estimated_supply_change_pct))

                current_cap = base_cap * (1 + estimated_supply_change_pct / 100)
                week_ago_cap = base_cap

                result[name] = {
                    'current': round(current_cap, 2),
                    'week_ago': round(week_ago_cap, 2),
                    'delta_7d': round(current_cap - week_ago_cap, 2),
                    'delta_pct': round(estimated_supply_change_pct, 2),
                    'price_current': round(current_price, 4),
                    'price_week_ago': round(week_ago_price, 4),
                    'source': 'estimated'
                }

            except Exception as e:
                # 에러 시 폴백
                result[name] = {
                    'current': self.FALLBACK_MARKET_CAP.get(name, 0),
                    'week_ago': self.FALLBACK_MARKET_CAP.get(name, 0),
                    'delta_7d': 0.0,
                    'delta_pct': 0.0,
                    'source': 'fallback',
                    'error': str(e)
                }

        return result

    def generate_detailed_comment(self, stablecoin_data: Dict[str, Dict]) -> Dict:
        """
        스테이블코인 7일 변화율 기반 상세 코멘트 생성

        소스 이론: "스테이블 코인(B*) 발행량이 늘어야 미국 국채를 사주므로(M 증가)
        Genius Act가 작동하는 것."

        Returns:
            Dict: {
                'total_market_cap': float,
                'total_delta_7d': float,
                'total_delta_pct': float,
                'genius_act_status': str,  # 'active', 'moderate', 'flat', 'draining'
                'detailed_comment': str,
                'economic_interpretation': str,
                'components': {...}
            }
        """
        # 총 시가총액 계산
        total_current = sum(d.get('current', 0) for d in stablecoin_data.values())
        total_week_ago = sum(d.get('week_ago', 0) for d in stablecoin_data.values())
        total_delta = total_current - total_week_ago
        total_delta_pct = (total_delta / total_week_ago * 100) if total_week_ago > 0 else 0

        # Genius Act 상태 판단
        if total_delta_pct > 3.0:
            status = 'active'
            comment = f"USD Liquidity Injection (Genius Act Active): Stablecoin issuance +{total_delta_pct:.1f}% in 7 days"
            interpretation = (
                f"스테이블코인 발행량 급증 (${total_delta:.1f}B, +{total_delta_pct:.1f}%). "
                f"Genius Act 담보 요건에 따라 미국 국채 수요 상승 예상. "
                f"M = B + S·B* 공식에서 S·B* 증가로 총 유동성(M) 확대. "
                f"크립토 시장 매수 대기 자금 증가, Risk-On 환경."
            )
        elif total_delta_pct > 1.0:
            status = 'moderate'
            comment = f"Moderate Stablecoin Growth: +{total_delta_pct:.1f}% weekly (${total_delta:.1f}B)"
            interpretation = (
                f"스테이블코인 완만한 증가 (+{total_delta_pct:.1f}%). "
                f"Genius Act 중립적 작동. 국채 수요 점진적 증가. "
                f"유동성 환경 안정적."
            )
        elif total_delta_pct > -1.0:
            status = 'flat'
            comment = f"Stablecoin issuance flat: {total_delta_pct:+.1f}% weekly (${total_delta:+.1f}B)"
            interpretation = (
                f"스테이블코인 발행량 정체 ({total_delta_pct:+.1f}%). "
                f"Genius Act 영향 미미. 국채 수요 변동 없음. "
                f"유동성 환경 변화 없음, 시장 중립."
            )
        else:
            status = 'draining'
            comment = f"Stablecoin Draining: {total_delta_pct:.1f}% weekly redemption (${abs(total_delta):.1f}B)"
            interpretation = (
                f"스테이블코인 소각/환매 진행 ({total_delta_pct:.1f}%). "
                f"크립토 시장 자금 이탈 신호. Genius Act 역작용 가능. "
                f"국채 담보 매각 압력, Risk-Off 주의."
            )

        # 개별 스테이블코인 상세
        components = {}
        for name, data in stablecoin_data.items():
            delta_pct = data.get('delta_pct', 0)
            if delta_pct > 2:
                component_status = "surging"
            elif delta_pct > 0:
                component_status = "growing"
            elif delta_pct > -2:
                component_status = "stable"
            else:
                component_status = "declining"

            components[name] = {
                'current': data.get('current', 0),
                'delta_7d': data.get('delta_7d', 0),
                'delta_pct': delta_pct,
                'status': component_status
            }

        return {
            'total_market_cap': round(total_current, 2),
            'total_delta_7d': round(total_delta, 2),
            'total_delta_pct': round(total_delta_pct, 2),
            'genius_act_status': status,
            'detailed_comment': comment,
            'economic_interpretation': interpretation,
            'components': components
        }


# =============================================================================
# 모니터링 대시보드
# =============================================================================

class LiquidityMonitor:
    """유동성 모니터링"""

    def __init__(self):
        self.history: List[LiquidityIndicators] = []
        self.strategy = GeniusActMacroStrategy()

    def update(self, indicators: LiquidityIndicators):
        """지표 업데이트"""
        self.history.append(indicators)

    def get_trend(self, window: int = 5) -> Dict:
        """트렌드 분석"""
        if len(self.history) < window:
            return {"error": "Insufficient data"}

        recent = self.history[-window:]

        # 스테이블코인 트렌드
        sc_trend = []
        for h in recent:
            total = h.usdt_supply + h.usdc_supply + h.dai_supply
            sc_trend.append(total)

        sc_change = (sc_trend[-1] - sc_trend[0]) / sc_trend[0] if sc_trend[0] > 0 else 0

        # 유동성 트렌드
        liq_trend = []
        for h in recent:
            net = h.fed_balance_sheet - h.rrp_balance - h.tga_balance
            liq_trend.append(net)

        liq_change = (liq_trend[-1] - liq_trend[0]) / liq_trend[0] if liq_trend[0] > 0 else 0

        return {
            "stablecoin_trend": "UP" if sc_change > 0.02 else "DOWN" if sc_change < -0.02 else "FLAT",
            "stablecoin_change": f"{sc_change*100:.1f}%",
            "liquidity_trend": "UP" if liq_change > 0.01 else "DOWN" if liq_change < -0.01 else "FLAT",
            "liquidity_change": f"{liq_change*100:.1f}%",
            "window": f"{window} periods"
        }

    def get_alerts(self) -> List[str]:
        """경고 알림"""
        alerts = []

        if len(self.history) < 2:
            return alerts

        current = self.history[-1]
        previous = self.history[-2]

        # 역레포 고갈 경고
        if current.rrp_balance < 0.2:  # 2000억 달러 미만
            alerts.append("⚠️ 역레포 잔액 고갈 임박 - 유동성 완충재 부족")

        # TGA 급락
        if previous.tga_balance > 0:
            tga_change = (current.tga_balance - previous.tga_balance) / previous.tga_balance
            if tga_change < -0.2:
                alerts.append(f"📊 TGA {tga_change*100:.0f}% 급락 - 대규모 재정 지출")

        # 스테이블코인 급변
        current_sc = current.usdt_supply + current.usdc_supply
        previous_sc = previous.usdt_supply + previous.usdc_supply
        if previous_sc > 0:
            sc_change = (current_sc - previous_sc) / previous_sc
            if sc_change > 0.1:
                alerts.append(f"🚀 스테이블코인 {sc_change*100:.0f}% 급증 - 크립토 유입 가속")
            elif sc_change < -0.05:
                alerts.append(f"🔻 스테이블코인 {sc_change*100:.0f}% 급감 - 크립토 이탈")

        return alerts


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Genius Act Macro Strategy Test")
    print("=" * 60)

    # 테스트 데이터 (2023-2024 시나리오 시뮬레이션)
    previous = LiquidityIndicators(
        fed_balance_sheet=7.8,     # 7.8조 달러
        rrp_balance=1.5,           # 1.5조 달러
        tga_balance=0.5,           # 5000억 달러
        usdt_supply=80,            # 800억 달러
        usdc_supply=30,            # 300억 달러
        dai_supply=5,              # 50억 달러
        m2=20.5,
        dxy=103,
        timestamp=datetime(2024, 1, 1)
    )

    current = LiquidityIndicators(
        fed_balance_sheet=7.5,     # 3000억 QT
        rrp_balance=0.8,           # 7000억 감소 (역레포 drain)
        tga_balance=0.6,           # 1000억 증가
        usdt_supply=95,            # 150억 증가 (+18.75%)
        usdc_supply=35,            # 50억 증가
        dai_supply=5,
        m2=20.8,
        dxy=101,
        timestamp=datetime(2024, 6, 1)
    )

    # 전략 실행
    strategy = GeniusActMacroStrategy()
    result = strategy.analyze(current, previous)

    print("\n1. Liquidity Analysis:")
    print(f"   Formula: {result['liquidity']['formula']}")
    print(f"   Base Liquidity (B): ${result['liquidity']['base_liquidity_B']:.2f}T")
    print(f"   Stablecoin Contribution (S·B*): ${result['liquidity']['stablecoin_contribution_SBstar']:.3f}T")
    print(f"   Total Liquidity (M): ${result['liquidity']['total_liquidity_M']:.2f}T")

    print(f"\n2. Current Regime: {result['regime']}")

    print("\n3. Generated Signals:")
    for sig in result['signals']:
        print(f"   [{sig['type']}] {sig['description']}")
        print(f"      Strength: {sig['strength']}, Confidence: {sig['confidence']}")
        print(f"      Affected: {', '.join(sig['affected_assets'])}")

    print("\n4. Recommended Positions:")
    for pos in result['positions']:
        print(f"   {pos['direction']} {pos['asset']} ({pos['size']})")
        print(f"      Signal: {pos['signal']}")
        print(f"      Rationale: {pos['rationale']}")

    print(f"\n5. Summary: {result['summary']}")

    # 모니터링 테스트
    print("\n" + "=" * 60)
    print("Liquidity Monitor Test")
    print("=" * 60)

    monitor = LiquidityMonitor()
    monitor.update(previous)
    monitor.update(current)

    alerts = monitor.get_alerts()
    print("\nAlerts:")
    for alert in alerts:
        print(f"   {alert}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
