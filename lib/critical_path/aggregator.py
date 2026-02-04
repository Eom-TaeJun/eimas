#!/usr/bin/env python3
"""
Critical Path - Main Aggregator
================================

모든 모듈을 통합하여 종합 리스크 분석을 수행하는 메인 클래스

Architecture:
    CriticalPathAggregator가 다음 모듈들을 조합:
    1. RiskAppetiteUncertaintyIndex - VIX 분해 (Bekaert et al.)
    2. EnhancedRegimeDetector - 레짐 탐지 (Maheu & McCurdy)
    3. SpilloverNetwork - 충격 전이 (Boeckelmann)
    4. CryptoSentimentBlock - 암호화폐 심리 (IMF)
    5. StressRegimeMultiplier - 스트레스 승수 (Longin-Solnik)

Economic Foundation:
    각 경로(path)의 위험 기여도를 합산하여 전체 리스크 점수 도출

Classes:
    - CriticalPathAggregator: 종합 분석 및 리스크 점수 계산

Returns:
    CriticalPathResult: 전체 위험도, 경로별 기여도, 하위 모듈 결과
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import from same package
from .schemas import (
    CriticalPathResult,
    RiskAppetiteUncertaintyResult,
    RegimeResult,
    SpilloverResult,
    CryptoSentimentResult
)
from .risk_appetite import RiskAppetiteUncertaintyIndex
from .regime import EnhancedRegimeDetector
from .spillover import SpilloverNetwork
from .crypto_sentiment import CryptoSentimentBlock
from .stress import StressRegimeMultiplier


class CriticalPathAggregator:
    """
    Critical Path 분석 통합 모듈
    
    4개 하위 모듈을 조율하고 최종 결과 산출
    
    하위 모듈:
    1. RiskAppetiteUncertaintyIndex: 리스크 선호도와 불확실성 분리 측정
    2. EnhancedRegimeDetector: 레짐 탐지 및 레짐별 임계값 제공
    3. SpilloverNetwork: 자산간 충격 전이 네트워크
    4. CryptoSentimentBlock: 암호화폐 심리 지표 블록
    """
    
    # 경로별 기본 가중치
    BASE_PATH_WEIGHTS = {
        'liquidity': 0.25,       # 유동성/금리 경로
        'concentration': 0.25,   # AI/빅테크 집중 경로
        'credit': 0.20,          # 신용 스트레스 경로
        'volatility': 0.15,      # 변동성/공포 경로
        'rotation': 0.10,        # 섹터 로테이션 경로
        'crypto': 0.05,          # 암호화폐 (기본값, 동적 조정)
    }
    
    # Perplexity 제안 기반 검증된 임계값 상수
    THRESHOLDS = {
        'zscore': {'warning': 1.5, 'alert': 2.0, 'critical': 2.5},
        'ml_prob': {'warning': 0.20, 'alert': 0.40, 'critical': 0.60},
        'rsi': {'overbought': 70, 'oversold': 30, 'extreme': {'ob': 80, 'os': 20}},
        'drawdown': {'days': 10, 'threshold': -0.05},
        'vix': {'normal': (15, 25), 'stress': 30, 'complacency': 12},
        'bb': {'window': 20, 'std': 2.0, 'compression_ratio': 0.5},
    }
    
    # Risk Appetite 가중합 (Bekaert 기반)
    RISK_APPETITE_WEIGHTS = {'HYG_LQD': 0.4, 'XLY_XLP': 0.3, 'IWM_SPY': 0.3}
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리 (선택적)
        """
        # 하위 모듈 초기화
        self.ra_uncertainty = RiskAppetiteUncertaintyIndex(lookback=20)
        self.regime_detector = EnhancedRegimeDetector(short_ma=20, long_ma=120)
        self.spillover_network = SpilloverNetwork(lookback=20)
        self.crypto_sentiment = CryptoSentimentBlock(lookback=20, correlation_window=20)
        
        # 설정 로드
        self.config = config or {}
        self.path_weights = self.BASE_PATH_WEIGHTS.copy()
    
    def collect_required_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        필요한 데이터가 모두 있는지 확인하고 정리
        
        필수 티커:
        - SPY, QQQ, TLT, GLD, VIX (기본)
        - HYG, LQD, XLY, XLP, IWM, XLF (RA/Spillover)
        - BTC-USD (Crypto)
        - SMH, NVDA, DXY (추가 경로)
        
        Returns:
            정리된 데이터 딕셔너리 및 누락된 티커 목록
        """
        required_tickers = [
            'SPY', 'QQQ', 'TLT', 'GLD', '^VIX', 'VIX',
            'HYG', 'LQD', 'XLY', 'XLP', 'IWM', 'XLF',
            'BTC-USD', 'SMH', 'NVDA', 'DXY', 'DX-Y.NYB', 'EEM'
        ]
        
        collected = {}
        missing = []
        
        for ticker in required_tickers:
            # 티커명 변형 시도
            data = market_data.get(ticker)
            if data is None:
                # 대체 티커 시도
                alt_tickers = {
                    '^VIX': 'VIX',
                    'DX-Y.NYB': 'DXY',
                }
                alt_ticker = alt_tickers.get(ticker)
                if alt_ticker:
                    data = market_data.get(alt_ticker)
            
            if data is not None and not data.empty:
                collected[ticker] = data
            else:
                missing.append(ticker)
        
        return {
            'data': collected,
            'missing': missing
        }
    
    def run_submodules(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        4개 하위 모듈 순차 실행
        
        실행 순서:
        1. Regime Detector (임계값 결정에 필요)
        2. Risk Appetite & Uncertainty
        3. Spillover Network (레짐 정보 활용)
        4. Crypto Sentiment
        
        Returns:
            Dict with all submodule results
        """
        results = {}
        
        # 1. Regime Detector (먼저 실행 - 다른 모듈에서 레짐 정보 필요)
        spy_data = market_data.get('SPY')
        vix_data = market_data.get('^VIX')
        if vix_data is None or (hasattr(vix_data, 'empty') and vix_data.empty):
            vix_data = market_data.get('VIX')
        
        if spy_data is not None and vix_data is not None:
            if hasattr(spy_data, 'empty') and spy_data.empty:
                spy_data = None
            if hasattr(vix_data, 'empty') and vix_data.empty:
                vix_data = None
        
        if spy_data is not None and vix_data is not None:
            regime_result = self.regime_detector.analyze(spy_data, vix_data)
            results['regime'] = regime_result
            current_regime = regime_result.current_regime
        else:
            current_regime = "TRANSITION"
            # 기본 RegimeResult 생성 (에러 방지)
            results['regime'] = None
        
        # 2. Risk Appetite & Uncertainty
        try:
            ra_result = self.ra_uncertainty.analyze(market_data)
            results['risk_appetite'] = ra_result
        except Exception as e:
            print(f"Warning: Risk Appetite analysis failed: {e}")
            results['risk_appetite'] = None
        
        # 3. Spillover Network (레짐 정보 활용)
        try:
            spillover_result = self.spillover_network.analyze(market_data, current_regime)
            results['spillover'] = spillover_result
        except Exception as e:
            print(f"Warning: Spillover analysis failed: {e}")
            results['spillover'] = None
        
        # 4. Crypto Sentiment
        btc_data = market_data.get('BTC-USD')
        spy_data = market_data.get('SPY')
        gld_data = market_data.get('GLD')
        
        if btc_data is not None and spy_data is not None and gld_data is not None:
            try:
                crypto_result = self.crypto_sentiment.analyze(btc_data, spy_data, gld_data)
                results['crypto'] = crypto_result
            except Exception as e:
                print(f"Warning: Crypto sentiment analysis failed: {e}")
                results['crypto'] = None
        else:
            results['crypto'] = None
        
        return results
    
    def calculate_path_contributions(
        self, 
        submodule_results: Dict,
        regime: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        경로별 위험 기여도 계산
        
        로직:
        1. 각 경로별 원시 점수(raw score) 계산
        2. 각 경로별 점수를 0-100 범위로 클리핑
        3. 시각화용 분포(path_distribution) 별도 계산 (100% 정규화)
        
        경로별 점수 산출:
        - liquidity: TLT 신호 + DXY 신호 + 수익률곡선
        - concentration: QQQ/SPY + RSP/SPY + NVDA/SMH
        - credit: HYG/LQD + XLF 신호
        - volatility: VIX 레벨 + 불확실성 지수
        - rotation: XLY/XLP + IWM/SPY
        - crypto: CryptoSentimentResult의 risk_contribution
        
        Returns:
            Tuple[path_contributions (raw scores), path_distribution (100% 정규화)]
        """
        contributions = {}
        
        # 1. Liquidity 경로 (유동성/금리)
        liquidity_score = 0.0
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            # TLT 관련 경로 찾기
            for edge in spillover.active_paths:
                if edge.source in ['TLT', 'DXY'] and edge.category == 'liquidity':
                    liquidity_score += edge.signal_strength * 0.5
        
        # Risk Appetite에서 불확실성 점수 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 불확실성이 높으면 유동성 경로 위험 증가
            liquidity_score += ra.uncertainty_score * 0.3
        
        # 0-100 범위로 클리핑
        contributions['liquidity'] = max(0.0, min(100.0, liquidity_score))
        
        # 2. Concentration 경로 (AI/빅테크 집중) - 개선
        concentration_score = 0.0
        
        # Spillover 경로에서 집중도 신호 확인
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'concentration':
                    concentration_score += edge.signal_strength * 0.6
        
        # 직접 계산 추가: Risk Appetite의 components 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            components = ra.components
            
            # HYG/LQD zscore가 높으면 신용 선호 (집중 위험 증가)
            hyg_lqd_z = abs(components.get('hyg_lqd_zscore', 0))
            if hyg_lqd_z > 1.0:  # 임계값 완화: 1.5 -> 1.0
                concentration_score += hyg_lqd_z * 6  # 가중치 조정
            
            # XLY/XLP zscore로 경기민감주 선호도 확인
            xly_xlp_z = components.get('xly_xlp_zscore', 0)
            if xly_xlp_z > 1.0:  # 양수일 때만 성장주 집중 (임계값 완화)
                concentration_score += xly_xlp_z * 5
            
            # 상관관계 분산이 낮으면 동조화 (집중 위험 증가)
            corr_var = components.get('corr_variance_score', 50)
            if corr_var < 30:  # 낮은 분산 = 높은 상관 = 동조화
                concentration_score += (30 - corr_var) * 0.5
        
        # 0-100 범위로 클리핑
        contributions['concentration'] = max(0.0, min(100.0, concentration_score))
        
        # 3. Credit 경로 (신용 스트레스)
        credit_score = 0.0
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'credit':
                    credit_score += edge.signal_strength * 0.5
        
        # Risk Appetite에서 리스크 선호도 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 리스크 선호도가 낮으면 신용 경로 위험 증가
            credit_score += (100 - ra.risk_appetite_score) * 0.3
        
        # 0-100 범위로 클리핑
        contributions['credit'] = max(0.0, min(100.0, credit_score))
        
        # 4. Volatility 경로 (변동성/공포)
        volatility_score = 0.0
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            # 불확실성 점수 활용
            volatility_score = ra.uncertainty_score * 0.5
        
        # Spillover에서 VIX 관련 경로
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'volatility':
                    volatility_score += edge.signal_strength * 0.5
        
        # 0-100 범위로 클리핑
        contributions['volatility'] = max(0.0, min(100.0, volatility_score))
        
        # 5. Rotation 경로 (섹터 로테이션) - 개선
        rotation_score = 0.0
        
        # Spillover 경로에서 로테이션 신호 확인
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.category == 'rotation':
                    rotation_score += edge.signal_strength * 0.5
        
        # 직접 계산 추가: Risk Appetite의 components 활용
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            components = ra.components
            
            # IWM/SPY zscore로 소형주 로테이션 확인 (임계값 완화)
            iwm_spy_z = abs(components.get('iwm_spy_zscore', 0))
            if iwm_spy_z > 0.8:  # 임계값 완화: 1.0 -> 0.8
                rotation_score += iwm_spy_z * 10
            
            # XLY/XLP zscore로 섹터 로테이션 확인 (임계값 완화)
            xly_xlp_z = abs(components.get('xly_xlp_zscore', 0))
            if xly_xlp_z > 0.8:  # 임계값 완화: 1.0 -> 0.8
                rotation_score += xly_xlp_z * 8
            
            # VRP(Variance Risk Premium)가 높으면 로테이션 신호
            vrp_score = components.get('vrp_score', 50)
            if vrp_score > 60:
                rotation_score += (vrp_score - 60) * 0.3
        
        # 0-100 범위로 클리핑
        contributions['rotation'] = max(0.0, min(100.0, rotation_score))
        
        # 6. Crypto 경로
        crypto_score = 0.0
        if submodule_results.get('crypto'):
            crypto = submodule_results['crypto']
            # 위험 기여도 활용 (0-0.2 범위를 0-40으로 변환, 상한선 설정)
            crypto_score = min(crypto.risk_contribution * 200, 40)  # 최대 40점
            # 심리 점수도 반영 (극단적 상태일 때)
            if crypto.sentiment_level in ['EXTREME_FEAR', 'EXTREME_GREED']:
                crypto_score += 20
        
        # 0-100 범위로 클리핑
        contributions['crypto'] = max(0.0, min(100.0, crypto_score))
        
        # 시각화용 분포 계산 (100% 정규화)
        path_distribution = {}
        total_raw = sum(contributions.values())
        if total_raw > 0:
            for path_name, score in contributions.items():
                path_distribution[path_name] = (score / total_raw) * 100.0
        else:
            # 모든 경로가 0인 경우 균등 분배
            num_paths = len(contributions)
            if num_paths > 0:
                equal_share = 100.0 / num_paths
                path_distribution = {path_name: equal_share for path_name in contributions.keys()}
            else:
                path_distribution = {}
        
        return contributions, path_distribution
    
    def adjust_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        레짐에 따라 경로 가중치 조정
        
        BULL: 집중도 경로 가중치 증가 (균열 감지 중요)
        BEAR: 신용, 유동성 경로 가중치 증가
        CRISIS: 변동성 경로 가중치 증가
        
        Returns:
            조정된 가중치 딕셔너리
        """
        weights = self.BASE_PATH_WEIGHTS.copy()
        
        if regime == "BULL":
            # 집중도 경로 가중치 증가
            weights['concentration'] *= 1.3
            weights['liquidity'] *= 0.9
            weights['credit'] *= 0.8
        elif regime == "BEAR":
            # 신용, 유동성 경로 가중치 증가
            weights['credit'] *= 1.4
            weights['liquidity'] *= 1.2
            weights['volatility'] *= 1.1
        elif regime == "CRISIS":
            # 변동성 경로 가중치 증가
            weights['volatility'] *= 1.5
            weights['credit'] *= 1.3
            weights['liquidity'] *= 1.2
        # TRANSITION은 기본 가중치 유지
        
        # 합계를 1.0으로 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_total_risk(
        self, 
        path_contributions: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        전체 위험도 계산 (가중평균 방식)
        
        각 경로의 raw score에 BASE_PATH_WEIGHTS 가중치를 곱한 가중평균을 계산합니다.
        이는 포트폴리오 위험 측정의 표준 방법론입니다.
        
        Returns:
            Tuple[score (0-100), level]
        
        Level 정의:
        - 0-25: LOW
        - 25-50: MEDIUM
        - 50-75: HIGH
        - 75-100: CRITICAL
        """
        # 가중평균 계산: 각 경로의 raw score × 기본 가중치
        weighted_sum = 0.0
        total_weight = 0.0
        
        for path_name, raw_score in path_contributions.items():
            weight = self.BASE_PATH_WEIGHTS.get(path_name, 0.0)
            weighted_sum += raw_score * weight
            total_weight += weight
        
        # 가중평균 계산
        if total_weight > 0:
            total_score = weighted_sum / total_weight
        else:
            total_score = 0.0
        
        # 0-100 범위로 클리핑
        total_score = max(0.0, min(100.0, total_score))
        
        if total_score < 25:
            level = "LOW"
        elif total_score < 50:
            level = "MEDIUM"
        elif total_score < 75:
            level = "HIGH"
        else:
            level = "CRITICAL"
        
        return total_score, level
    
    def generate_warnings(self, submodule_results: Dict) -> List[str]:
        """
        활성화된 경고 목록 생성
        
        경고 예시:
        - "TLT 급락 중: 금리 상승 압력, QQQ 영향 예상 (3일 시차)"
        - "BTC 선행 하락 감지: RISK_OFF_WARNING"
        - "레짐 전환 징후: BULL → TRANSITION (확률 65%)"
        - "VIX-실현변동성 괴리 확대: 불확실성 프리미엄 증가"
        
        Returns:
            List of warning strings
        """
        warnings = []
        
        # 1. Spillover 경로 경고
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            for edge in spillover.active_paths:
                if edge.signal_strength >= 70:
                    direction = "하락" if edge.expected_target_move == "DOWN" else "상승"
                    warnings.append(
                        f"{edge.source} 급변 중: {edge.theory_note}, "
                        f"{edge.target} {direction} 압력 예상 ({edge.adjusted_lag}일 시차)"
                    )
        
        # 2. 레짐 전환 경고
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            if regime.transition_probability >= 50:
                warnings.append(
                    f"레짐 전환 징후: {regime.current_regime} → "
                    f"{regime.transition_direction} (확률 {regime.transition_probability:.0f}%)"
                )
        
        # 3. Crypto 선행지표 경고
        if submodule_results.get('crypto'):
            crypto = submodule_results['crypto']
            if crypto.is_leading_indicator and crypto.leading_signal:
                if crypto.leading_signal == "RISK_OFF_WARNING":
                    warnings.append("BTC 선행 하락 감지: RISK_OFF_WARNING - 주식 시장 하락 선행 가능")
                elif crypto.leading_signal == "RISK_ON_SIGNAL":
                    warnings.append("BTC 선행 상승 감지: RISK_ON_SIGNAL - 주식 시장 상승 선행 가능")
        
        # 4. 불확실성 프리미엄 경고
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            if 'vix_realized_gap' in ra.components:
                gap = ra.components['vix_realized_gap']
                if gap > 10:
                    warnings.append(
                        f"VIX-실현변동성 괴리 확대 ({gap:.1f}%p): 불확실성 프리미엄 증가"
                    )
        
        # 5. 위기 상태 경고
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            if regime.current_regime == "CRISIS":
                warnings.append("🚨 위기 레짐 감지: 유동성 확보 및 방어적 포지션 권장")
        
        return warnings
    
    def generate_interpretation(
        self, 
        total_risk: float,
        risk_level: str,
        path_contributions: Dict,
        submodule_results: Dict
    ) -> str:
        """
        종합 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        # 기본 위험도 설명
        interpretation = f"현재 시장 위험도는 {total_risk:.1f}% ({risk_level}) 수준입니다. "
        
        # 주요 위험 경로
        sorted_paths = sorted(
            path_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if sorted_paths:
            top_path = sorted_paths[0]
            path_names = {
                'liquidity': '유동성/금리',
                'concentration': 'AI/빅테크 집중',
                'credit': '신용 스트레스',
                'volatility': '변동성/공포',
                'rotation': '섹터 로테이션',
                'crypto': '암호화폐'
            }
            top_path_name = path_names.get(top_path[0], top_path[0])
            interpretation += f"주요 위험 요인은 {top_path_name} 경로({top_path[1]:.1f}%)입니다. "
        
        # 레짐 정보
        if submodule_results.get('regime'):
            regime = submodule_results['regime']
            interpretation += f"현재 {regime.current_regime} 레짐이며, "
            if regime.transition_probability >= 50:
                interpretation += f"레짐 전환 확률이 {regime.transition_probability:.0f}%로 상승 중입니다. "
            else:
                interpretation += f"레짐 안정도는 {regime.regime_confidence:.0f}%입니다. "
        
        # Risk Appetite 정보
        if submodule_results.get('risk_appetite'):
            ra = submodule_results['risk_appetite']
            interpretation += (
                f"리스크 선호도는 {ra.risk_appetite_score:.0f}점({ra.risk_appetite_level}), "
                f"불확실성은 {ra.uncertainty_score:.0f}점({ra.uncertainty_level})입니다. "
            )
        
        # Spillover 정보
        if submodule_results.get('spillover'):
            spillover = submodule_results['spillover']
            if spillover.active_paths:
                interpretation += (
                    f"활성화된 충격 전이 경로는 {len(spillover.active_paths)}개이며, "
                    f"주요 위험 진원지는 {spillover.primary_risk_source}입니다. "
                )
        
        return interpretation
    
    def analyze(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> CriticalPathResult:
        """
        전체 분석 파이프라인 실행
        
        1. 데이터 검증 및 정리
        2. 하위 모듈 실행
        3. 경로별 기여도 계산
        4. 전체 위험도 산출
        5. 경고 및 해석 생성
        6. 결과 객체 반환
        
        Args:
            market_data: 티커별 가격 데이터 딕셔너리
        
        Returns:
            CriticalPathResult 객체
        """
        # 1. 데이터 검증
        data_check = self.collect_required_data(market_data)
        if len(data_check['missing']) > 5:
            print(f"Warning: {len(data_check['missing'])} required tickers missing")
        
        # 2. 하위 모듈 실행
        submodule_results = self.run_submodules(market_data)
        
        # 레짐 정보 추출
        if submodule_results.get('regime'):
            current_regime = submodule_results['regime'].current_regime
            regime_confidence = submodule_results['regime'].regime_confidence
            transition_prob = submodule_results['regime'].transition_probability
        else:
            current_regime = "TRANSITION"
            regime_confidence = 50.0
            transition_prob = 0.0
        
        # 3. 경로별 기여도 계산
        path_contributions, path_distribution = self.calculate_path_contributions(
            submodule_results,
            current_regime
        )
        
        # 4. 전체 위험도 산출
        total_risk, risk_level = self.calculate_total_risk(path_contributions)
        
        # 5. 주요 위험 경로 식별
        if path_contributions:
            primary_risk_path = max(path_contributions.items(), key=lambda x: x[1])[0]
        else:
            primary_risk_path = "NONE"
        
        # 6. 경고 생성
        active_warnings = self.generate_warnings(submodule_results)
        
        # 7. 해석 생성
        interpretation = self.generate_interpretation(
            total_risk,
            risk_level,
            path_contributions,
            submodule_results
        )
        
        # 8. 결과 객체 생성 (None 값 처리)
        risk_appetite_result = submodule_results.get('risk_appetite')
        regime_result = submodule_results.get('regime')
        spillover_result = submodule_results.get('spillover')
        crypto_result = submodule_results.get('crypto')
        
        # 기본값 생성 (None인 경우)
        if risk_appetite_result is None:
            risk_appetite_result = RiskAppetiteUncertaintyResult(
                timestamp=datetime.now().isoformat(),
                risk_appetite_score=50.0,
                uncertainty_score=50.0,
                risk_appetite_level="MEDIUM",
                uncertainty_level="MEDIUM",
                market_state="MIXED",
                components={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if regime_result is None:
            regime_result = RegimeResult(
                timestamp=datetime.now().isoformat(),
                current_regime="TRANSITION",
                regime_confidence=50.0,
                transition_probability=0.0,
                transition_direction="STABLE",
                thresholds={},
                ma_status={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if spillover_result is None:
            spillover_result = SpilloverResult(
                timestamp=datetime.now().isoformat(),
                active_paths=[],
                risk_score=0.0,
                primary_risk_source="NONE",
                expected_impacts={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        if crypto_result is None:
            crypto_result = CryptoSentimentResult(
                timestamp=datetime.now().isoformat(),
                sentiment_score=50.0,
                sentiment_level="NEUTRAL",
                btc_spy_correlation=0.0,
                correlation_regime="DECOUPLED",
                is_leading_indicator=False,
                leading_signal=None,
                risk_contribution=0.05,
                components={},
                interpretation="데이터 부족으로 분석 불가"
            )
        
        return CriticalPathResult(
            timestamp=datetime.now().isoformat(),
            total_risk_score=total_risk,
            risk_level=risk_level,
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            transition_probability=transition_prob,
            path_contributions=path_contributions,
            path_distribution=path_distribution,
            risk_appetite_result=risk_appetite_result,
            regime_result=regime_result,
            spillover_result=spillover_result,
            crypto_result=crypto_result,
            primary_risk_path=primary_risk_path,
            active_warnings=active_warnings,
            interpretation=interpretation
        )


# ============================================================
# 통합 함수 (main.py에서 호출)
# ============================================================

def run_critical_path_analysis(
    market_data: Dict[str, pd.DataFrame]
) -> CriticalPathResult:
    """
    Critical Path 분석 실행 (편의 함수)
    
    Usage:
        from critical_path_analyzer import run_critical_path_analysis
        result = run_critical_path_analysis(market_data)
        print(f"Total Risk: {result.total_risk_score}%")
    
    Args:
        market_data: 티커별 가격 데이터 딕셔너리
    
    Returns:
        CriticalPathResult 객체
    """
    aggregator = CriticalPathAggregator()
