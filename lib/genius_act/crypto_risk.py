#!/usr/bin/env python3
"""
Genius Act - Crypto Risk Evaluator
============================================================

Stablecoin-specific risk evaluation

Economic Foundation:
    - Collateral backing analysis
    - Interest-bearing risk premium
    - Multi-dimensional risk scoring

Class:
    - CryptoRiskEvaluator: Crypto and stablecoin risk assessment
"""

from typing import Dict, List, Optional
import logging
import pandas as pd

from .enums import StablecoinCollateralType
from .stablecoin_risk import MultiDimensionalRiskScore, StablecoinRiskProfile

logger = logging.getLogger(__name__)


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

