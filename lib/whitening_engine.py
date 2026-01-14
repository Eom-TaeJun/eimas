"""
Whitening Engine - Economic Reverse Engineering
================================================
AI 결과에 대한 경제학적 해석 및 역추적 시스템

핵심 철학:
- "왜 이 결과가 나왔는가?"에 답할 수 없으면 블랙박스
- 결과 → 원인으로 역추적 (Reverse Engineering)
- 모든 의사결정에 경제학적 근거 부여

기능:
1. Factor Attribution: 어떤 팩터가 결과에 기여했는가?
2. Causal Validation: 인과관계 경로가 실제로 작동하는가?
3. Economic Narrative: 결과를 경제학적으로 설명
4. Anomaly Explanation: 이상치가 왜 발생했는가?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# Local imports
try:
    from lib.shock_propagation_graph import (
        ShockPropagationGraph,
        NodeLayer,
        get_node_layer
    )
    SPG_AVAILABLE = True
except ImportError:
    SPG_AVAILABLE = False


class EconomicFactor(Enum):
    """경제학적 팩터"""
    # Macro Factors
    INTEREST_RATE = "interest_rate"           # 금리 (R)
    INFLATION = "inflation"                   # 인플레이션 (π)
    LIQUIDITY = "liquidity"                   # 유동성 (M)
    DOLLAR_STRENGTH = "dollar_strength"       # 달러 강세 (DXY)

    # Risk Factors
    GEOPOLITICAL = "geopolitical"             # 지정학적 리스크
    CREDIT_RISK = "credit_risk"               # 신용 리스크
    VOLATILITY = "volatility"                 # 변동성

    # Sector Factors
    TECH_MOMENTUM = "tech_momentum"           # 기술주 모멘텀
    ENERGY_CYCLE = "energy_cycle"             # 에너지 사이클
    DEFENSIVE_ROTATION = "defensive_rotation" # 방어주 로테이션

    # Flow Factors
    INSTITUTIONAL_FLOW = "institutional_flow" # 기관 자금 흐름
    RETAIL_SENTIMENT = "retail_sentiment"     # 개인 투자 심리
    STABLECOIN_FLOW = "stablecoin_flow"       # 스테이블코인 유입


@dataclass
class FactorAttribution:
    """팩터 기여도 분석"""
    factor: EconomicFactor
    contribution: float          # 기여도 (%)
    direction: str              # "POSITIVE", "NEGATIVE"
    confidence: float           # 신뢰도
    evidence: List[str]         # 근거
    data_points: Dict[str, float] = field(default_factory=dict)


@dataclass
class CausalValidation:
    """인과관계 검증 결과"""
    hypothesis: str             # 가설
    path: List[str]             # 인과 경로
    is_valid: bool              # 검증 결과
    correlation: float          # 상관관계
    lag_days: int               # 시차
    p_value: float              # 통계적 유의성
    counter_evidence: List[str] # 반증


@dataclass
class EconomicNarrative:
    """경제학적 내러티브"""
    timestamp: str
    summary: str                # 핵심 요약
    key_drivers: List[FactorAttribution]
    causal_paths: List[CausalValidation]
    risk_factors: List[str]
    opportunities: List[str]
    confidence: float
    caveats: List[str]          # 주의사항


class WhiteningEngine:
    """
    Whitening Engine: 결과를 경제학적으로 설명

    "바이오 섹터 비중 증가" → "왜?"
    → "금리 인하 기대 (R↓) + 장수(Longevity) 테마 부상"
    """

    # 팩터-자산 매핑 (경제학적 도메인 지식)
    FACTOR_ASSET_MAPPING = {
        EconomicFactor.INTEREST_RATE: {
            'positive': ['TLT', 'XLU', 'XLRE'],  # 금리 하락 수혜
            'negative': ['XLF', 'KRE'],           # 금리 하락 피해
            'proxy': ['DGS10', 'DGS2', 'T10Y2Y']
        },
        EconomicFactor.INFLATION: {
            'positive': ['GLD', 'TIP', 'DBC'],    # 인플레 헤지
            'negative': ['TLT', 'VCIT'],          # 인플레 피해
            'proxy': ['CPIAUCSL', 'PCEPILFE']
        },
        EconomicFactor.LIQUIDITY: {
            'positive': ['BTC-USD', 'QQQ', 'ARKK'],  # 유동성 수혜
            'negative': ['SHY'],                      # 유동성 중립
            'proxy': ['RRP', 'TGA', 'M2', 'USDT_SUPPLY']
        },
        EconomicFactor.DOLLAR_STRENGTH: {
            'positive': ['UUP', 'DXY'],           # 달러 강세 수혜
            'negative': ['EEM', 'GLD', 'FXE'],    # 달러 강세 피해
            'proxy': ['DXY', 'DX-Y.NYB']
        },
        EconomicFactor.GEOPOLITICAL: {
            'positive': ['XAR', 'ITA', 'LMT', 'RTX'],  # 방산
            'negative': ['EEM', 'VWO'],                 # 신흥국
            'proxy': ['VIX', 'GLD']
        },
        EconomicFactor.TECH_MOMENTUM: {
            'positive': ['QQQ', 'XLK', 'SMH', 'SOXX'],
            'negative': ['XLU', 'XLP'],
            'proxy': ['QQQ', 'ARKK']
        },
        EconomicFactor.STABLECOIN_FLOW: {
            'positive': ['BTC-USD', 'ETH-USD'],   # 크립토
            'negative': [],
            'proxy': ['USDT_SUPPLY', 'USDC_SUPPLY']
        }
    }

    # 클러스터-테마 매핑
    CLUSTER_THEME_MAPPING = {
        'tech': ['AI', '반도체', '클라우드', '소프트웨어'],
        'healthcare': ['바이오', '제약', '의료기기', 'Longevity'],
        'defense': ['방산', '사이버보안', '우주항공'],
        'energy': ['석유', '천연가스', '신재생', '원자력'],
        'financial': ['은행', '보험', '핀테크'],
        'consumer': ['소비재', '유통', 'e커머스']
    }

    def __init__(self, macro_data: Optional[pd.DataFrame] = None):
        self.macro_data = macro_data
        self.spg = ShockPropagationGraph() if SPG_AVAILABLE else None

    def explain_allocation(
        self,
        weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]] = None,
        returns: Optional[pd.DataFrame] = None,
        macro_data: Optional[pd.DataFrame] = None
    ) -> EconomicNarrative:
        """
        포트폴리오 배분 결과를 경제학적으로 설명

        Args:
            weights: 현재 가중치
            previous_weights: 이전 가중치 (변화 분석용)
            returns: 수익률 데이터
            macro_data: 거시 데이터

        Returns:
            EconomicNarrative
        """
        if macro_data is not None:
            self.macro_data = macro_data

        # 1. 가중치 변화 분석
        weight_changes = self._analyze_weight_changes(weights, previous_weights)

        # 2. 팩터 기여도 분석
        factor_attributions = self._attribute_factors(weights, weight_changes, returns)

        # 3. 인과관계 검증
        causal_validations = self._validate_causality(factor_attributions)

        # 4. 리스크 요인 식별
        risk_factors = self._identify_risks(weights, factor_attributions)

        # 5. 기회 요인 식별
        opportunities = self._identify_opportunities(weights, factor_attributions)

        # 6. 내러티브 생성
        summary = self._generate_summary(factor_attributions, weight_changes)

        # 7. 주의사항
        caveats = self._generate_caveats(factor_attributions, causal_validations)

        # 신뢰도 계산
        confidence = self._calculate_confidence(factor_attributions, causal_validations)

        return EconomicNarrative(
            timestamp=datetime.now().isoformat(),
            summary=summary,
            key_drivers=factor_attributions[:5],  # 상위 5개
            causal_paths=causal_validations,
            risk_factors=risk_factors,
            opportunities=opportunities,
            confidence=confidence,
            caveats=caveats
        )

    def _analyze_weight_changes(
        self,
        weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Dict]:
        """가중치 변화 분석"""
        changes = {}

        for asset, weight in weights.items():
            prev = previous_weights.get(asset, 0) if previous_weights else 0
            change = weight - prev

            if abs(change) > 0.005:  # 0.5% 이상 변화만
                changes[asset] = {
                    'current': weight,
                    'previous': prev,
                    'change': change,
                    'direction': 'INCREASE' if change > 0 else 'DECREASE',
                    'magnitude': abs(change)
                }

        return changes

    def _attribute_factors(
        self,
        weights: Dict[str, float],
        weight_changes: Dict[str, Dict],
        returns: Optional[pd.DataFrame]
    ) -> List[FactorAttribution]:
        """팩터 기여도 분석"""
        attributions = []

        for factor, mapping in self.FACTOR_ASSET_MAPPING.items():
            positive_assets = mapping.get('positive', [])
            negative_assets = mapping.get('negative', [])
            proxy_assets = mapping.get('proxy', [])

            # 해당 팩터와 관련된 자산의 가중치 합계
            positive_weight = sum(weights.get(a, 0) for a in positive_assets)
            negative_weight = sum(weights.get(a, 0) for a in negative_assets)

            # 순 노출도
            net_exposure = positive_weight - negative_weight

            if abs(net_exposure) < 0.01:
                continue

            # 방향 결정
            direction = "POSITIVE" if net_exposure > 0 else "NEGATIVE"

            # 근거 수집
            evidence = []
            for asset in positive_assets:
                if asset in weights and weights[asset] > 0.01:
                    evidence.append(f"{asset}: {weights[asset]:.1%}")

            # 거시 데이터에서 프록시 값 추출
            data_points = {}
            if self.macro_data is not None:
                for proxy in proxy_assets:
                    if proxy in self.macro_data.columns:
                        recent = self.macro_data[proxy].dropna()
                        if len(recent) > 0:
                            data_points[proxy] = recent.iloc[-1]

            # 신뢰도 계산
            confidence = min(0.9, abs(net_exposure) * 5 + len(evidence) * 0.1)

            attributions.append(FactorAttribution(
                factor=factor,
                contribution=net_exposure * 100,
                direction=direction,
                confidence=confidence,
                evidence=evidence,
                data_points=data_points
            ))

        # 기여도 순 정렬
        attributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return attributions

    def _validate_causality(
        self,
        factor_attributions: List[FactorAttribution]
    ) -> List[CausalValidation]:
        """인과관계 검증"""
        validations = []

        # 주요 인과관계 가설
        hypotheses = [
            {
                'hypothesis': '금리 인하 기대 → 성장주 선호',
                'path': ['FED_FUNDS', 'DGS10', 'QQQ'],
                'factor': EconomicFactor.INTEREST_RATE
            },
            {
                'hypothesis': '유동성 증가 → 위험자산 선호',
                'path': ['M2', 'NET_LIQUIDITY', 'SPY', 'BTC-USD'],
                'factor': EconomicFactor.LIQUIDITY
            },
            {
                'hypothesis': '달러 약세 → 신흥시장/금 강세',
                'path': ['DXY', 'EEM', 'GLD'],
                'factor': EconomicFactor.DOLLAR_STRENGTH
            },
            {
                'hypothesis': '스테이블코인 유입 → 크립토 강세',
                'path': ['USDT_SUPPLY', 'BTC-USD'],
                'factor': EconomicFactor.STABLECOIN_FLOW
            },
            {
                'hypothesis': '지정학적 긴장 → 방산/금 강세',
                'path': ['VIX', 'GLD', 'XAR'],
                'factor': EconomicFactor.GEOPOLITICAL
            }
        ]

        for hyp in hypotheses:
            # 해당 팩터가 활성화되었는지 확인
            relevant_attr = next(
                (a for a in factor_attributions if a.factor == hyp['factor']),
                None
            )

            if relevant_attr is None or abs(relevant_attr.contribution) < 1:
                continue

            # 인과관계 검증 (거시 데이터 있을 때)
            is_valid = False
            correlation = 0.0
            lag_days = 0
            p_value = 1.0
            counter_evidence = []

            if self.macro_data is not None and self.spg is not None:
                # SPG를 통한 인과관계 검증
                path = hyp['path']
                available_path = [p for p in path if p in self.macro_data.columns]

                if len(available_path) >= 2:
                    # Granger causality 검증
                    source = self.macro_data[available_path[0]].dropna()
                    target = self.macro_data[available_path[-1]].dropna()

                    if len(source) > 30 and len(target) > 30:
                        # 간단한 상관관계 검증
                        aligned = pd.concat([source, target], axis=1).dropna()
                        if len(aligned) > 20:
                            correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                            is_valid = abs(correlation) > 0.3
                            p_value = 0.05 if is_valid else 0.5

            # 반증 수집
            if not is_valid:
                counter_evidence.append("상관관계가 약하거나 데이터 부족")

            validations.append(CausalValidation(
                hypothesis=hyp['hypothesis'],
                path=hyp['path'],
                is_valid=is_valid,
                correlation=correlation,
                lag_days=lag_days,
                p_value=p_value,
                counter_evidence=counter_evidence
            ))

        return validations

    def _identify_risks(
        self,
        weights: Dict[str, float],
        factor_attributions: List[FactorAttribution]
    ) -> List[str]:
        """리스크 요인 식별"""
        risks = []

        # 집중도 리스크
        top_5_weight = sum(sorted(weights.values(), reverse=True)[:5])
        if top_5_weight > 0.4:
            risks.append(f"상위 5개 자산 집중도 {top_5_weight:.1%} - 분산 부족")

        # 팩터 쏠림 리스크
        for attr in factor_attributions[:3]:
            if abs(attr.contribution) > 15:
                risks.append(
                    f"{attr.factor.value} 팩터 노출 {attr.contribution:.1f}% - "
                    f"{'과도한 베팅' if attr.direction == 'POSITIVE' else '역방향 리스크'}"
                )

        # 유동성 리스크
        if any(a.factor == EconomicFactor.LIQUIDITY and a.direction == "NEGATIVE"
               for a in factor_attributions):
            risks.append("유동성 축소 환경에서 위험자산 노출")

        # 금리 리스크
        rate_attr = next(
            (a for a in factor_attributions if a.factor == EconomicFactor.INTEREST_RATE),
            None
        )
        if rate_attr and rate_attr.direction == "POSITIVE" and rate_attr.contribution > 10:
            risks.append("금리 인상 시 듀레이션 리스크 노출")

        return risks

    def _identify_opportunities(
        self,
        weights: Dict[str, float],
        factor_attributions: List[FactorAttribution]
    ) -> List[str]:
        """기회 요인 식별"""
        opportunities = []

        for attr in factor_attributions[:3]:
            if attr.direction == "POSITIVE" and attr.confidence > 0.6:
                if attr.factor == EconomicFactor.LIQUIDITY:
                    opportunities.append(
                        "유동성 확장 환경 - 위험자산 추가 비중 확대 기회"
                    )
                elif attr.factor == EconomicFactor.TECH_MOMENTUM:
                    opportunities.append(
                        "기술주 모멘텀 지속 - AI/반도체 익스포저 유효"
                    )
                elif attr.factor == EconomicFactor.STABLECOIN_FLOW:
                    opportunities.append(
                        "스테이블코인 유입 증가 - 크립토 추가 비중 고려"
                    )
                elif attr.factor == EconomicFactor.GEOPOLITICAL:
                    opportunities.append(
                        "지정학적 긴장 지속 - 방산/금 헤지 효과 기대"
                    )

        return opportunities

    def _generate_summary(
        self,
        factor_attributions: List[FactorAttribution],
        weight_changes: Dict[str, Dict]
    ) -> str:
        """핵심 요약 생성"""
        if not factor_attributions:
            return "팩터 기여도 분석 불가 - 데이터 부족"

        top_factor = factor_attributions[0]

        # 주요 변화 자산
        major_changes = sorted(
            weight_changes.items(),
            key=lambda x: abs(x[1]['change']),
            reverse=True
        )[:3]

        change_summary = ""
        if major_changes:
            changes = [f"{a}({c['direction'][:1]}{abs(c['change']):.1%})"
                      for a, c in major_changes]
            change_summary = f" 주요 변화: {', '.join(changes)}."

        return (
            f"포트폴리오는 {top_factor.factor.value} 팩터에 "
            f"{abs(top_factor.contribution):.1f}% {top_factor.direction} 노출."
            f"{change_summary}"
        )

    def _generate_caveats(
        self,
        factor_attributions: List[FactorAttribution],
        causal_validations: List[CausalValidation]
    ) -> List[str]:
        """주의사항 생성"""
        caveats = []

        # 낮은 신뢰도 팩터
        low_conf = [a for a in factor_attributions if a.confidence < 0.5]
        if low_conf:
            caveats.append(
                f"{len(low_conf)}개 팩터의 신뢰도가 낮음 - 추가 검증 필요"
            )

        # 검증 실패 인과관계
        invalid_causal = [c for c in causal_validations if not c.is_valid]
        if invalid_causal:
            caveats.append(
                f"{len(invalid_causal)}개 인과관계 가설이 데이터로 검증되지 않음"
            )

        # 데이터 부족
        if self.macro_data is None or len(self.macro_data) < 60:
            caveats.append("거시 데이터 부족 - 팩터 분석의 신뢰도 제한")

        return caveats

    def _calculate_confidence(
        self,
        factor_attributions: List[FactorAttribution],
        causal_validations: List[CausalValidation]
    ) -> float:
        """전체 신뢰도 계산"""
        if not factor_attributions:
            return 0.3

        # 팩터 신뢰도 평균
        factor_conf = np.mean([a.confidence for a in factor_attributions])

        # 인과관계 검증률
        if causal_validations:
            valid_rate = sum(1 for c in causal_validations if c.is_valid) / len(causal_validations)
        else:
            valid_rate = 0.5

        return (factor_conf * 0.6 + valid_rate * 0.4)

    def explain_cluster(
        self,
        cluster_id: int,
        cluster_assets: List[str],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        클러스터를 경제학적으로 설명

        "왜 이 자산들이 묶였는가?"
        """
        # 클러스터 내 가중치 합계
        cluster_weight = sum(weights.get(a, 0) for a in cluster_assets)

        # 테마 추론
        themes = []
        for theme, keywords in self.CLUSTER_THEME_MAPPING.items():
            # 자산명에서 테마 키워드 매칭 (간단화된 버전)
            matches = sum(1 for a in cluster_assets
                         if any(k.lower() in a.lower() for k in keywords))
            if matches > 0:
                themes.append(theme)

        # 팩터 노출도
        factor_exposures = {}
        for factor, mapping in self.FACTOR_ASSET_MAPPING.items():
            positive = [a for a in cluster_assets if a in mapping.get('positive', [])]
            negative = [a for a in cluster_assets if a in mapping.get('negative', [])]
            if positive or negative:
                factor_exposures[factor.value] = {
                    'positive': positive,
                    'negative': negative,
                    'net_count': len(positive) - len(negative)
                }

        return {
            'cluster_id': cluster_id,
            'assets': cluster_assets,
            'total_weight': cluster_weight,
            'inferred_themes': themes if themes else ['General'],
            'factor_exposures': factor_exposures,
            'interpretation': self._interpret_cluster(themes, factor_exposures, cluster_weight)
        }

    def _interpret_cluster(
        self,
        themes: List[str],
        factor_exposures: Dict,
        weight: float
    ) -> str:
        """클러스터 해석"""
        if not themes:
            return f"범용 클러스터 (비중: {weight:.1%})"

        theme_str = '/'.join(themes)

        # 주요 팩터
        if factor_exposures:
            top_factor = max(
                factor_exposures.items(),
                key=lambda x: abs(x[1]['net_count'])
            )
            return (
                f"{theme_str} 테마 클러스터 (비중: {weight:.1%}). "
                f"주요 팩터: {top_factor[0]}"
            )

        return f"{theme_str} 테마 클러스터 (비중: {weight:.1%})"

    def reverse_engineer(
        self,
        observation: str,
        weights: Dict[str, float],
        macro_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        관찰된 현상을 역추적

        예: "바이오 섹터 비중이 높다" → "왜?"

        Args:
            observation: 관찰된 현상 (텍스트)
            weights: 포트폴리오 가중치
            macro_data: 거시 데이터

        Returns:
            역추적 분석 결과
        """
        if macro_data is not None:
            self.macro_data = macro_data

        # 관찰 파싱 (키워드 기반)
        keywords = observation.lower().split()

        # 관련 팩터 식별
        related_factors = []
        for factor, mapping in self.FACTOR_ASSET_MAPPING.items():
            all_assets = mapping.get('positive', []) + mapping.get('negative', [])
            if any(k in ' '.join(all_assets).lower() for k in keywords):
                related_factors.append(factor)

        # 가능한 원인 목록
        possible_causes = []

        for factor in related_factors:
            mapping = self.FACTOR_ASSET_MAPPING[factor]
            proxy_assets = mapping.get('proxy', [])

            # 거시 데이터에서 최근 트렌드 확인
            trends = {}
            if self.macro_data is not None:
                for proxy in proxy_assets:
                    if proxy in self.macro_data.columns:
                        series = self.macro_data[proxy].dropna()
                        if len(series) >= 20:
                            recent_change = (series.iloc[-1] / series.iloc[-20] - 1) * 100
                            trends[proxy] = recent_change

            possible_causes.append({
                'factor': factor.value,
                'explanation': self._factor_to_explanation(factor),
                'supporting_data': trends,
                'confidence': 0.7 if trends else 0.4
            })

        return {
            'observation': observation,
            'timestamp': datetime.now().isoformat(),
            'possible_causes': possible_causes,
            'recommendation': self._generate_recommendation(possible_causes)
        }

    def _factor_to_explanation(self, factor: EconomicFactor) -> str:
        """팩터를 경제학적 설명으로 변환"""
        explanations = {
            EconomicFactor.INTEREST_RATE: "금리 변화로 인한 할인율 효과 (R↓ → Growth↑)",
            EconomicFactor.INFLATION: "인플레이션 헤지 수요 또는 실질수익률 변화",
            EconomicFactor.LIQUIDITY: "유동성 환경 변화 (M↑ → Risk Asset↑)",
            EconomicFactor.DOLLAR_STRENGTH: "달러 강/약세에 따른 상대 가치 변화",
            EconomicFactor.GEOPOLITICAL: "지정학적 리스크에 따른 안전자산 선호",
            EconomicFactor.TECH_MOMENTUM: "기술주 모멘텀 및 AI 투자 사이클",
            EconomicFactor.STABLECOIN_FLOW: "스테이블코인 유입에 따른 크립토 유동성"
        }
        return explanations.get(factor, "알 수 없는 팩터")

    def _generate_recommendation(self, possible_causes: List[Dict]) -> str:
        """권고사항 생성"""
        if not possible_causes:
            return "추가 데이터 필요"

        top_cause = max(possible_causes, key=lambda x: x['confidence'])

        if top_cause['confidence'] > 0.6:
            return f"주요 원인: {top_cause['factor']}. {top_cause['explanation']}"
        else:
            return "복합적 요인으로 판단됨. 추가 분석 필요."


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Whitening Engine Test")
    print("=" * 60)

    # 샘플 데이터
    weights = {
        'QQQ': 0.15, 'SMH': 0.10, 'SOXX': 0.08,  # Tech
        'XLV': 0.12, 'IBB': 0.08,                 # Healthcare
        'XAR': 0.05, 'ITA': 0.05,                 # Defense
        'GLD': 0.10, 'TLT': 0.07,                 # Safe haven
        'BTC-USD': 0.05,                          # Crypto
        'SPY': 0.15                               # Broad market
    }

    previous_weights = {
        'QQQ': 0.12, 'SMH': 0.08, 'SOXX': 0.05,
        'XLV': 0.10, 'IBB': 0.05,
        'XAR': 0.03, 'ITA': 0.03,
        'GLD': 0.15, 'TLT': 0.12,
        'BTC-USD': 0.02,
        'SPY': 0.25
    }

    # Whitening Engine 실행
    engine = WhiteningEngine()
    narrative = engine.explain_allocation(weights, previous_weights)

    print("\n1. Summary:")
    print(f"   {narrative.summary}")

    print("\n2. Key Drivers:")
    for driver in narrative.key_drivers:
        print(f"   [{driver.factor.value}] {driver.contribution:.1f}% {driver.direction}")
        print(f"      Evidence: {', '.join(driver.evidence[:3])}")
        print(f"      Confidence: {driver.confidence:.0%}")

    print("\n3. Causal Validations:")
    for cv in narrative.causal_paths:
        status = "✓" if cv.is_valid else "✗"
        print(f"   {status} {cv.hypothesis}")
        print(f"      Path: {' → '.join(cv.path)}")

    print("\n4. Risks:")
    for risk in narrative.risk_factors:
        print(f"   ⚠️ {risk}")

    print("\n5. Opportunities:")
    for opp in narrative.opportunities:
        print(f"   ✨ {opp}")

    print("\n6. Caveats:")
    for caveat in narrative.caveats:
        print(f"   📝 {caveat}")

    print(f"\n7. Overall Confidence: {narrative.confidence:.0%}")

    # Reverse Engineering 테스트
    print("\n" + "=" * 60)
    print("Reverse Engineering Test")
    print("=" * 60)

    result = engine.reverse_engineer(
        "기술주와 반도체 비중이 높다",
        weights
    )

    print(f"\nObservation: {result['observation']}")
    print("\nPossible Causes:")
    for cause in result['possible_causes']:
        print(f"   • {cause['factor']}: {cause['explanation']}")
        print(f"     Confidence: {cause['confidence']:.0%}")

    print(f"\nRecommendation: {result['recommendation']}")

    print("\n" + "=" * 60)
    print("Test completed!")
