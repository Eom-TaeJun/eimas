from __future__ import annotations
#!/usr/bin/env python3
"""
Genius Act - Data Schemas
============================================================

Data classes for Genius Act macro strategy

Economic Foundation:
    - Genius Act: Fed liquidity framework
    - Stablecoin risk: Collateral-based risk assessment
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .enums import LiquidityRegime, SignalType, StablecoinCollateralType


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


