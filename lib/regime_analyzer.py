#!/usr/bin/env python3
"""
EIMAS Regime Analyzer - GMM & Entropy
======================================
가우시안 혼합 모델(GMM)과 섀넌 엔트로피 기반 레짐 분석

경제학적 근거:
- "지금 현재 값들이 여러 개가 믹스된 값... GMM(Gaussian Mixture Model)을 써야 함"
- "엔트로피(Entropy): 데이터가 얼마나 섞여 있는지를 나타내는 지표... 불순할수록 엔트로피 ↑"

핵심 기능:
1. GMM 기반 3-상태 분류 (Bull/Neutral/Bear)
2. 각 상태 소속 확률 계산
3. 섀넌 엔트로피로 시장 불확실성 정량화
4. 리포트용 해석 텍스트 생성

Usage:
    from lib.regime_analyzer import GMMRegimeAnalyzer

    analyzer = GMMRegimeAnalyzer()
    result = analyzer.analyze(spy_returns, vix_data)
    print(result.interpretation)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.regime_analyzer')

# Optional: sklearn
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not installed. GMM analysis will be disabled.")


# =============================================================================
# Enums & Constants
# =============================================================================

class GMMRegimeState(str, Enum):
    """GMM 기반 레짐 상태"""
    BULL = "Bull"               # 강세 (높은 수익률, 낮은 변동성)
    NEUTRAL = "Neutral"         # 중립 (평균 수익률)
    BEAR = "Bear"               # 약세 (낮은 수익률, 높은 변동성)
    UNKNOWN = "Unknown"         # 미확인


class EntropyLevel(str, Enum):
    """엔트로피 수준"""
    VERY_LOW = "Very Low"       # < 0.3 (매우 명확한 신호)
    LOW = "Low"                 # 0.3 - 0.5 (명확한 신호)
    MEDIUM = "Medium"           # 0.5 - 0.7 (보통)
    HIGH = "High"               # 0.7 - 0.9 (혼재)
    VERY_HIGH = "Very High"     # > 0.9 (매우 혼재/혼란)


# 엔트로피 수준별 해석
ENTROPY_INTERPRETATIONS = {
    EntropyLevel.VERY_LOW: {
        "message": "Strong Regime Signal",
        "description": "시장이 한 방향으로 명확하게 기울어 있음. 신뢰도 높은 신호.",
        "action": "레짐에 따른 포지션 적극 구축 가능",
    },
    EntropyLevel.LOW: {
        "message": "Clear Regime Signal",
        "description": "시장 방향성이 비교적 명확함.",
        "action": "레짐 방향으로 점진적 포지션 구축",
    },
    EntropyLevel.MEDIUM: {
        "message": "Moderate Uncertainty",
        "description": "시장에 약간의 불확실성 존재.",
        "action": "분할 진입, 헤지 고려",
    },
    EntropyLevel.HIGH: {
        "message": "Market is Mixed",
        "description": "시장 신호가 혼재되어 있음. 레짐 전환 가능성.",
        "action": "포지션 축소, 관망 권장",
    },
    EntropyLevel.VERY_HIGH: {
        "message": "Market is Confused",
        "description": "시장이 매우 불확실함. 여러 레짐이 비슷한 확률로 혼재.",
        "action": "현금 비중 확대, 명확한 신호 대기",
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GMMResult:
    """GMM 분석 결과"""
    timestamp: str

    # 레짐 분류
    current_regime: GMMRegimeState
    regime_probabilities: Dict[str, float]      # {'Bull': 0.7, 'Neutral': 0.2, 'Bear': 0.1}

    # 엔트로피
    shannon_entropy: float                       # 0-1 (정규화)
    entropy_level: EntropyLevel
    entropy_interpretation: str

    # GMM 파라미터
    n_components: int = 3
    converged: bool = True
    bic: float = 0.0                            # Bayesian Information Criterion

    # 클러스터 특성
    cluster_means: Dict[str, float] = field(default_factory=dict)
    cluster_stds: Dict[str, float] = field(default_factory=dict)

    # 해석
    interpretation: str = ""
    action_recommendation: str = ""

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'current_regime': self.current_regime.value,
            'regime_probabilities': {k: round(v, 3) for k, v in self.regime_probabilities.items()},
            'shannon_entropy': round(self.shannon_entropy, 3),
            'entropy_level': self.entropy_level.value,
            'entropy_interpretation': self.entropy_interpretation,
            'interpretation': self.interpretation,
            'action': self.action_recommendation,
            'gmm_converged': self.converged,
            'gmm_bic': round(self.bic, 2),
        }

    def to_report_line(self) -> str:
        """리포트용 한 줄 요약"""
        probs = self.regime_probabilities
        prob_str = f"Bull:{probs.get('Bull', 0):.0%}/Neutral:{probs.get('Neutral', 0):.0%}/Bear:{probs.get('Bear', 0):.0%}"
        return (
            f"GMM Regime: **{self.current_regime.value}** ({prob_str}) | "
            f"Entropy: {self.shannon_entropy:.2f} ({self.entropy_level.value}) - "
            f"{self.entropy_interpretation}"
        )


# =============================================================================
# GMM Regime Analyzer
# =============================================================================

class GMMRegimeAnalyzer:
    """
    가우시안 혼합 모델 기반 레짐 분석기

    Features:
    - 3-상태 GMM (Bull/Neutral/Bear)
    - 섀넌 엔트로피 계산
    - 불확실성 해석
    """

    def __init__(
        self,
        n_components: int = 3,
        lookback_days: int = 252,
        random_state: int = 42
    ):
        """
        Args:
            n_components: GMM 클러스터 수 (기본 3: Bull/Neutral/Bear)
            lookback_days: 분석 기간 (기본 252일 = 1년)
            random_state: 재현성을 위한 시드
        """
        self.n_components = n_components
        self.lookback_days = lookback_days
        self.random_state = random_state
        self.gmm = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None

    def _prepare_features(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        GMM 입력 피처 준비

        Features:
        1. 일간 수익률
        2. 20일 롤링 수익률
        3. VIX (있을 경우)
        4. 실현 변동성
        """
        features = []

        # 1. 일간 수익률
        daily_ret = returns.dropna().values[-self.lookback_days:]
        features.append(daily_ret)

        # 2. 20일 롤링 수익률
        rolling_ret = returns.rolling(20).mean().dropna().values[-self.lookback_days:]
        if len(rolling_ret) < len(daily_ret):
            rolling_ret = np.pad(rolling_ret, (len(daily_ret) - len(rolling_ret), 0), 'edge')
        features.append(rolling_ret[:len(daily_ret)])

        # 3. 실현 변동성 (20일)
        realized_vol = returns.rolling(20).std().dropna().values[-self.lookback_days:]
        if len(realized_vol) < len(daily_ret):
            realized_vol = np.pad(realized_vol, (len(daily_ret) - len(realized_vol), 0), 'edge')
        features.append(realized_vol[:len(daily_ret)])

        # 4. VIX (있을 경우)
        if vix is not None and len(vix) > 0:
            vix_values = vix.dropna().values[-self.lookback_days:]
            if len(vix_values) < len(daily_ret):
                vix_values = np.pad(vix_values, (len(daily_ret) - len(vix_values), 0), 'edge')
            features.append(vix_values[:len(daily_ret)])

        # 피처 행렬 구성
        X = np.column_stack(features)

        # 표준화
        if self.scaler:
            X = self.scaler.fit_transform(X)

        return X

    def _calculate_shannon_entropy(self, probabilities: np.ndarray) -> float:
        """
        섀넌 엔트로피 계산

        H = -Σ p(i) * log2(p(i))

        정규화: H / log2(n_components) -> 0~1 범위

        Args:
            probabilities: 각 클러스터 소속 확률

        Returns:
            정규화된 엔트로피 (0=완전 순수, 1=완전 혼합)
        """
        # 0 확률 처리
        probs = probabilities[probabilities > 1e-10]

        if len(probs) == 0:
            return 0.0

        # 섀넌 엔트로피
        entropy = -np.sum(probs * np.log2(probs))

        # 정규화 (최대 엔트로피 = log2(n_components))
        max_entropy = np.log2(self.n_components)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(np.clip(normalized_entropy, 0, 1))

    def _get_entropy_level(self, entropy: float) -> EntropyLevel:
        """엔트로피 수준 분류"""
        if entropy < 0.3:
            return EntropyLevel.VERY_LOW
        elif entropy < 0.5:
            return EntropyLevel.LOW
        elif entropy < 0.7:
            return EntropyLevel.MEDIUM
        elif entropy < 0.9:
            return EntropyLevel.HIGH
        else:
            return EntropyLevel.VERY_HIGH

    def _classify_clusters(
        self,
        cluster_means: np.ndarray
    ) -> Dict[int, GMMRegimeState]:
        """
        클러스터를 레짐으로 분류

        수익률 평균 기준:
        - 가장 높은 평균 → Bull
        - 중간 평균 → Neutral
        - 가장 낮은 평균 → Bear
        """
        # 첫 번째 피처(일간 수익률)의 평균으로 정렬
        means = cluster_means[:, 0]  # 일간 수익률 컬럼
        sorted_indices = np.argsort(means)  # 오름차순

        mapping = {}
        mapping[sorted_indices[0]] = GMMRegimeState.BEAR      # 가장 낮은 수익률
        mapping[sorted_indices[1]] = GMMRegimeState.NEUTRAL   # 중간
        mapping[sorted_indices[2]] = GMMRegimeState.BULL      # 가장 높은 수익률

        return mapping

    def analyze(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> GMMResult:
        """
        GMM 기반 레짐 분석 수행

        Args:
            returns: SPY 또는 시장 지수 수익률 시리즈
            vix: VIX 시리즈 (선택)
            market_data: 시장 데이터 딕셔너리 (선택, SPY/VIX 자동 추출)

        Returns:
            GMMResult 객체
        """
        timestamp = datetime.now().isoformat()

        # sklearn 체크
        if not HAS_SKLEARN:
            return GMMResult(
                timestamp=timestamp,
                current_regime=GMMRegimeState.UNKNOWN,
                regime_probabilities={'Bull': 0.33, 'Neutral': 0.34, 'Bear': 0.33},
                shannon_entropy=1.0,
                entropy_level=EntropyLevel.VERY_HIGH,
                entropy_interpretation="sklearn not installed",
                interpretation="GMM analysis requires sklearn",
                action_recommendation="Install sklearn: pip install scikit-learn"
            )

        # market_data에서 SPY/VIX 추출
        if market_data and returns is None:
            if 'SPY' in market_data:
                spy_df = market_data['SPY']
                returns = spy_df['Close'].pct_change().dropna()
            if '^VIX' in market_data and vix is None:
                vix_df = market_data['^VIX']
                vix = vix_df['Close']

        if returns is None or len(returns) < 30:
            return GMMResult(
                timestamp=timestamp,
                current_regime=GMMRegimeState.UNKNOWN,
                regime_probabilities={'Bull': 0.33, 'Neutral': 0.34, 'Bear': 0.33},
                shannon_entropy=1.0,
                entropy_level=EntropyLevel.VERY_HIGH,
                entropy_interpretation="Insufficient data",
                interpretation="Not enough data for GMM analysis (min 30 days required)",
                action_recommendation="Collect more historical data"
            )

        try:
            # 피처 준비
            X = self._prepare_features(returns, vix)

            if len(X) < 30:
                raise ValueError("Insufficient observations for GMM")

            # GMM 학습
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                max_iter=200,
                random_state=self.random_state,
                n_init=5
            )
            self.gmm.fit(X)

            # 최근 관측치의 확률 예측
            latest_obs = X[-1:, :]
            cluster_probs = self.gmm.predict_proba(latest_obs)[0]
            predicted_cluster = self.gmm.predict(latest_obs)[0]

            # 클러스터 → 레짐 매핑
            cluster_to_regime = self._classify_clusters(self.gmm.means_)

            # 레짐별 확률 계산
            regime_probs = {}
            for cluster_idx, regime in cluster_to_regime.items():
                regime_probs[regime.value] = float(cluster_probs[cluster_idx])

            # 현재 레짐
            current_regime = cluster_to_regime[predicted_cluster]

            # 섀넌 엔트로피 계산
            entropy = self._calculate_shannon_entropy(cluster_probs)
            entropy_level = self._get_entropy_level(entropy)
            entropy_info = ENTROPY_INTERPRETATIONS[entropy_level]

            # 클러스터 특성 저장
            cluster_means = {}
            cluster_stds = {}
            for cluster_idx, regime in cluster_to_regime.items():
                cluster_means[regime.value] = float(self.gmm.means_[cluster_idx, 0])
                cluster_stds[regime.value] = float(np.sqrt(self.gmm.covariances_[cluster_idx, 0, 0]))

            # 해석 생성
            max_prob = max(regime_probs.values())
            if max_prob > 0.7:
                confidence_text = "높은 신뢰도로"
            elif max_prob > 0.5:
                confidence_text = "보통 신뢰도로"
            else:
                confidence_text = "낮은 신뢰도로"

            interpretation = (
                f"GMM 분석 결과: 시장은 {confidence_text} **{current_regime.value}** 레짐에 있음. "
                f"(확률: {regime_probs.get(current_regime.value, 0):.1%}). "
                f"엔트로피 {entropy:.2f}로 {entropy_info['message']}."
            )

            return GMMResult(
                timestamp=timestamp,
                current_regime=current_regime,
                regime_probabilities=regime_probs,
                shannon_entropy=entropy,
                entropy_level=entropy_level,
                entropy_interpretation=entropy_info['message'],
                n_components=self.n_components,
                converged=self.gmm.converged_,
                bic=self.gmm.bic(X),
                cluster_means=cluster_means,
                cluster_stds=cluster_stds,
                interpretation=interpretation,
                action_recommendation=entropy_info['action']
            )

        except Exception as e:
            logger.error(f"GMM analysis failed: {e}")
            return GMMResult(
                timestamp=timestamp,
                current_regime=GMMRegimeState.UNKNOWN,
                regime_probabilities={'Bull': 0.33, 'Neutral': 0.34, 'Bear': 0.33},
                shannon_entropy=1.0,
                entropy_level=EntropyLevel.VERY_HIGH,
                entropy_interpretation=f"Analysis error: {str(e)}",
                interpretation=f"GMM analysis failed: {str(e)}",
                action_recommendation="Check data quality and retry"
            )

    def analyze_from_market_data(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> GMMResult:
        """
        시장 데이터 딕셔너리에서 직접 분석

        Args:
            market_data: {'SPY': DataFrame, '^VIX': DataFrame, ...}

        Returns:
            GMMResult
        """
        returns = None
        vix = None

        if 'SPY' in market_data:
            spy_df = market_data['SPY']
            if 'Close' in spy_df.columns:
                returns = spy_df['Close'].pct_change().dropna()

        if '^VIX' in market_data:
            vix_df = market_data['^VIX']
            if 'Close' in vix_df.columns:
                vix = vix_df['Close']

        return self.analyze(returns, vix)


# =============================================================================
# Integration Helper
# =============================================================================

def get_gmm_regime_summary(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    main.py 통합용 간단 함수

    Returns:
        {
            'regime': 'Bull',
            'probabilities': {'Bull': 0.7, ...},
            'entropy': 0.45,
            'entropy_level': 'Low',
            'interpretation': '...',
            'report_line': '...'
        }
    """
    analyzer = GMMRegimeAnalyzer()
    result = analyzer.analyze_from_market_data(market_data)

    return {
        'regime': result.current_regime.value,
        'probabilities': result.regime_probabilities,
        'entropy': result.shannon_entropy,
        'entropy_level': result.entropy_level.value,
        'interpretation': result.entropy_interpretation,
        'action': result.action_recommendation,
        'report_line': result.to_report_line(),
        'full_result': result.to_dict()
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("EIMAS GMM Regime Analyzer Test")
    print("=" * 60)

    if not HAS_SKLEARN:
        print("\n[ERROR] sklearn not installed. Run: pip install scikit-learn")
        exit(1)

    # 데이터 수집
    print("\n[1] Fetching SPY & VIX data...")
    spy = yf.download('SPY', period='1y', progress=False)
    vix = yf.download('^VIX', period='1y', progress=False)

    spy_returns = spy['Close'].pct_change().dropna()
    vix_close = vix['Close']

    print(f"    SPY observations: {len(spy_returns)}")
    print(f"    VIX observations: {len(vix_close)}")

    # GMM 분석
    print("\n[2] Running GMM Analysis...")
    analyzer = GMMRegimeAnalyzer(n_components=3, lookback_days=252)
    result = analyzer.analyze(spy_returns, vix_close)

    print(f"\n[3] Results:")
    print(f"    Current Regime: {result.current_regime.value}")
    print(f"    Probabilities:")
    for regime, prob in sorted(result.regime_probabilities.items(), key=lambda x: -x[1]):
        print(f"      → {regime}: {prob:.1%}")

    print(f"\n    Shannon Entropy: {result.shannon_entropy:.3f}")
    print(f"    Entropy Level: {result.entropy_level.value}")
    print(f"    Interpretation: {result.entropy_interpretation}")

    print(f"\n    GMM Converged: {result.converged}")
    print(f"    BIC Score: {result.bic:.2f}")

    print(f"\n[4] Report Line:")
    print(f"    {result.to_report_line()}")

    print(f"\n[5] Full Interpretation:")
    print(f"    {result.interpretation}")
    print(f"    Action: {result.action_recommendation}")

    print("\n" + "=" * 60)
    print("GMM Regime Analyzer Test Complete!")
    print("=" * 60)
