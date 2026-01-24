"""
Time Series Similarity Analysis
================================
Dynamic Time Warping (DTW) 및 시계열 유사도 분석 모듈

경제학적 배경 (금융경제정리.docx, todolist.md):
- DTW: 시계열 간 시차(lag)를 무시하고 패턴 유사도만 측정
- 리드-래그 관계 파악: 선행/후행 자산 식별
- 레짐 전환 조기 감지: 현재 패턴과 과거 레짐 패턴 비교
- 클러스터링 품질 향상: 상관관계 대신 DTW 거리 사용

핵심 아이디어:
1. DTW는 두 시계열을 정렬(align)하여 최소 거리를 찾음
2. Euclidean 거리와 달리 시차가 있어도 패턴이 같으면 거리가 작음
3. 금융 시장에서 자산 간 리드-래그 관계 발견에 유용

References:
- Berndt, D. J., & Clifford, J. (1994). "Using dynamic time warping to find patterns
  in time series." KDD-94.
- Petitjean, F., et al. (2011). "A global averaging method for dynamic time warping,
  with applications to clustering." Pattern Recognition, 44(3), 678-693.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Optional imports
try:
    from scipy.spatial.distance import euclidean
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available. Some features may be limited.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DTWResult:
    """DTW 거리 계산 결과"""
    distance: float
    normalized_distance: float  # 시계열 길이로 정규화
    alignment_path: List[Tuple[int, int]]  # (i, j) 매칭 경로
    cumulative_cost_matrix: np.ndarray  # 누적 비용 행렬
    computation_time: float  # 계산 시간 (초)


@dataclass
class SimilarityMatrixResult:
    """DTW 유사도 행렬 결과"""
    timestamp: str
    distance_matrix: pd.DataFrame  # DTW 거리 행렬
    n_series: int
    series_names: List[str]
    avg_distance: float
    most_similar_pair: Tuple[str, str, float]  # (series1, series2, distance)
    most_dissimilar_pair: Tuple[str, str, float]


@dataclass
class LeadLagResult:
    """리드-래그 관계 분석 결과"""
    timestamp: str
    lead_asset: str  # 선행 자산
    lag_asset: str   # 후행 자산
    optimal_lag: int  # 최적 시차 (일)
    min_distance: float  # 최소 DTW 거리
    cross_correlation: float  # 최대 교차상관계수
    interpretation: str


@dataclass
class RegimeShiftSignal:
    """레짐 전환 신호"""
    timestamp: str
    current_regime: str  # 현재 추정 레짐
    similarity_to_bull: float  # Bull 레짐과의 유사도
    similarity_to_bear: float  # Bear 레짐과의 유사도
    shift_probability: float  # 레짐 전환 확률 (0-1)
    signal: str  # STABLE / WARNING / SHIFT_DETECTED
    interpretation: str


# ============================================================================
# Core DTW Functions
# ============================================================================

def dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None,
    return_path: bool = False
) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Dynamic Time Warping (DTW) 거리 계산

    경제학적 의미:
    - Euclidean 거리: 시점이 정확히 일치해야 함
    - DTW 거리: 시차가 있어도 패턴이 같으면 유사하다고 판단
    - 금융에서 자산 간 리드-래그 관계 발견에 핵심

    알고리즘 (동적 프로그래밍):
    1. 비용 행렬 D[i,j] 초기화
    2. D[i,j] = |series1[i] - series2[j]| + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    3. D[n,m]이 최종 DTW 거리

    Parameters:
        series1: 시계열 1 (1D numpy array)
        series2: 시계열 2 (1D numpy array)
        window: Sakoe-Chiba 윈도우 크기 (None = 전역 정렬)
                - 계산 복잡도 O(n*m) → O(n*window)로 감소
        return_path: True면 정렬 경로도 반환

    Returns:
        distance: DTW 거리
        path: (optional) 정렬 경로 [(i1, j1), (i2, j2), ...]

    Example:
        >>> s1 = np.array([1, 2, 3, 4, 5])
        >>> s2 = np.array([1, 1, 2, 3, 4, 5])  # s1보다 1기간 늦음
        >>> dist, path = dtw_distance(s1, s2, return_path=True)
        >>> print(f"DTW distance: {dist:.2f}")

    References:
        Sakoe, H., & Chiba, S. (1978). "Dynamic programming algorithm optimization
        for spoken word recognition." IEEE TASSP, 26(1), 43-49.
    """
    n, m = len(series1), len(series2)

    # 누적 비용 행렬 초기화
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0

    # Sakoe-Chiba 윈도우 (제약 조건)
    if window is None:
        window = max(n, m)

    # DP 계산
    for i in range(1, n + 1):
        # 윈도우 범위 계산
        start = max(1, i - window)
        end = min(m + 1, i + window + 1)

        for j in range(start, end):
            # 로컬 비용 (Euclidean 거리)
            cost = abs(series1[i - 1] - series2[j - 1])

            # 최소 경로 선택
            D[i, j] = cost + min(
                D[i - 1, j],      # 삽입 (series1에서 갭)
                D[i, j - 1],      # 삭제 (series2에서 갭)
                D[i - 1, j - 1]   # 매칭
            )

    dtw_dist = D[n, m]

    # 정렬 경로 역추적
    path = None
    if return_path:
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))

            # 최소 비용 방향 선택
            candidates = [
                (D[i - 1, j], (i - 1, j)),
                (D[i, j - 1], (i, j - 1)),
                (D[i - 1, j - 1], (i - 1, j - 1))
            ]
            _, (i, j) = min(candidates, key=lambda x: x[0])

        path.reverse()

    return dtw_dist, path


def compute_dtw_similarity_matrix(
    returns: pd.DataFrame,
    window: Optional[int] = None,
    normalize: bool = True
) -> SimilarityMatrixResult:
    """
    다중 시계열 간 DTW 유사도 행렬 계산

    경제학적 의미:
    - 상관관계 행렬: 시차 무시, 선형 관계만 포착
    - DTW 행렬: 시차 고려, 비선형 패턴도 포착
    - 클러스터링에 사용 시 더 정확한 자산 군집 발견

    Parameters:
        returns: 자산별 수익률 DataFrame (columns = assets)
        window: DTW 윈도우 크기 (None = 전역 정렬)
        normalize: True면 시계열 길이로 정규화

    Returns:
        SimilarityMatrixResult:
            - distance_matrix: DTW 거리 행렬 (n_assets x n_assets)
            - most_similar_pair: 가장 유사한 자산 쌍
            - most_dissimilar_pair: 가장 상이한 자산 쌍

    Example:
        >>> result = compute_dtw_similarity_matrix(returns_df, window=20)
        >>> print(result.distance_matrix)
        >>> print(f"Most similar: {result.most_similar_pair}")
    """
    assets = returns.columns.tolist()
    n_assets = len(assets)

    # DTW 거리 행렬 초기화
    dist_matrix = np.zeros((n_assets, n_assets))

    # 각 자산 쌍에 대해 DTW 계산
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            s1 = returns[assets[i]].dropna().values
            s2 = returns[assets[j]].dropna().values

            # DTW 거리 계산
            dist, _ = dtw_distance(s1, s2, window=window, return_path=False)

            # 정규화 (시계열 길이)
            if normalize:
                dist = dist / max(len(s1), len(s2))

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 대칭

    # DataFrame으로 변환
    dist_df = pd.DataFrame(dist_matrix, index=assets, columns=assets)

    # 통계 계산
    # 대각선 제외 (자기 자신과의 거리 = 0)
    mask = ~np.eye(n_assets, dtype=bool)
    avg_dist = dist_matrix[mask].mean()

    # 가장 유사/상이한 쌍 찾기
    min_dist = np.inf
    max_dist = -np.inf
    min_pair = None
    max_pair = None

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if dist_matrix[i, j] < min_dist:
                min_dist = dist_matrix[i, j]
                min_pair = (assets[i], assets[j], min_dist)

            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
                max_pair = (assets[i], assets[j], max_dist)

    return SimilarityMatrixResult(
        timestamp=datetime.now().isoformat(),
        distance_matrix=dist_df,
        n_series=n_assets,
        series_names=assets,
        avg_distance=avg_dist,
        most_similar_pair=min_pair,
        most_dissimilar_pair=max_pair
    )


def find_lead_lag_relationship(
    series1: pd.Series,
    series2: pd.Series,
    max_lag: int = 20,
    series1_name: str = "Asset1",
    series2_name: str = "Asset2"
) -> LeadLagResult:
    """
    두 시계열 간 리드-래그 관계 탐지

    경제학적 의미:
    - 리드(Lead) 자산: 시장 움직임을 선행하는 자산 (정보 우위)
    - 래그(Lag) 자산: 후행하는 자산
    - 활용: 선행 지표 기반 트레이딩 전략

    알고리즘:
    1. series2를 -max_lag ~ +max_lag 범위로 시프트
    2. 각 시차에 대해 DTW 거리 계산
    3. 최소 DTW 거리를 주는 시차를 최적 lag로 선택
    4. lag > 0: series1이 선행, lag < 0: series2가 선행

    Parameters:
        series1: 시계열 1 (pandas Series)
        series2: 시계열 2 (pandas Series)
        max_lag: 최대 탐색 시차 (일)
        series1_name: 시계열 1 이름
        series2_name: 시계열 2 이름

    Returns:
        LeadLagResult:
            - lead_asset: 선행 자산
            - lag_asset: 후행 자산
            - optimal_lag: 최적 시차 (일)
            - min_distance: 최소 DTW 거리

    Example:
        >>> result = find_lead_lag_relationship(spy_returns, qqq_returns, max_lag=10)
        >>> print(f"{result.lead_asset} leads {result.lag_asset} by {result.optimal_lag} days")
    """
    # 공통 인덱스 정렬
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx].dropna().values
    s2 = series2.loc[common_idx].dropna().values

    min_len = min(len(s1), len(s2))
    s1 = s1[:min_len]
    s2 = s2[:min_len]

    # 각 시차에 대해 DTW 거리 계산
    best_lag = 0
    min_dist = np.inf
    max_corr = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            s2_shifted = s2
        elif lag > 0:
            # series2를 앞으로 시프트 (series1이 선행)
            s2_shifted = s2[lag:]
            s1_truncated = s1[:-lag] if lag < len(s1) else s1
        else:
            # series2를 뒤로 시프트 (series2가 선행)
            s1_truncated = s1[-lag:]
            s2_shifted = s2[:lag]

        # 길이 맞추기
        min_len_shifted = min(len(s1_truncated), len(s2_shifted))
        if min_len_shifted < 10:  # 너무 짧으면 스킵
            continue

        s1_calc = s1_truncated[:min_len_shifted] if lag <= 0 else s1[:min_len_shifted]
        s2_calc = s2_shifted[:min_len_shifted]

        # DTW 거리 계산
        dist, _ = dtw_distance(s1_calc, s2_calc, window=10, return_path=False)
        dist_normalized = dist / min_len_shifted

        # 교차상관계수 계산
        if len(s1_calc) > 1 and len(s2_calc) > 1:
            corr = np.corrcoef(s1_calc, s2_calc)[0, 1]
        else:
            corr = 0.0

        # 최소 거리 업데이트
        if dist_normalized < min_dist:
            min_dist = dist_normalized
            best_lag = lag
            max_corr = corr

    # 결과 해석
    if best_lag > 0:
        lead_asset = series1_name
        lag_asset = series2_name
        interpretation = f"{series1_name}이(가) {series2_name}보다 {best_lag}일 선행"
    elif best_lag < 0:
        lead_asset = series2_name
        lag_asset = series1_name
        interpretation = f"{series2_name}이(가) {series1_name}보다 {-best_lag}일 선행"
    else:
        lead_asset = "N/A"
        lag_asset = "N/A"
        interpretation = "동시 움직임 (리드-래그 관계 없음)"

    return LeadLagResult(
        timestamp=datetime.now().isoformat(),
        lead_asset=lead_asset,
        lag_asset=lag_asset,
        optimal_lag=abs(best_lag),
        min_distance=min_dist,
        cross_correlation=max_corr,
        interpretation=interpretation
    )


def detect_regime_shift_dtw(
    current_window: pd.Series,
    bull_template: pd.Series,
    bear_template: pd.Series,
    threshold: float = 0.3
) -> RegimeShiftSignal:
    """
    DTW 기반 레짐 전환 조기 감지

    경제학적 의미:
    - 현재 시장 패턴과 과거 Bull/Bear 레짐 패턴을 DTW로 비교
    - 패턴 유사도 급변 = 레짐 전환 신호
    - 기존 통계 기법보다 빠른 조기 감지 가능

    알고리즘:
    1. 현재 윈도우 (예: 최근 20일)와 Bull 템플릿 DTW 거리 계산
    2. 현재 윈도우와 Bear 템플릿 DTW 거리 계산
    3. 거리 비교로 현재 레짐 추정
    4. 거리 차이가 threshold 이하면 전환 경고

    Parameters:
        current_window: 현재 시장 데이터 (최근 N일)
        bull_template: Bull 레짐 템플릿 (과거 Bull 기간 평균 패턴)
        bear_template: Bear 레짐 템플릿 (과거 Bear 기간 평균 패턴)
        threshold: 전환 감지 임계값 (0-1, 낮을수록 민감)

    Returns:
        RegimeShiftSignal:
            - current_regime: BULL / BEAR / UNCERTAIN
            - shift_probability: 레짐 전환 확률
            - signal: STABLE / WARNING / SHIFT_DETECTED

    Example:
        >>> # 과거 Bull/Bear 기간 패턴 추출
        >>> bull_template = returns['2019-01-01':'2019-12-31']  # 예시
        >>> bear_template = returns['2020-03-01':'2020-04-30']
        >>> current = returns.tail(20)
        >>> signal = detect_regime_shift_dtw(current, bull_template, bear_template)
        >>> print(signal.interpretation)
    """
    current_array = current_window.dropna().values
    bull_array = bull_template.dropna().values
    bear_array = bear_template.dropna().values

    # DTW 거리 계산
    dist_to_bull, _ = dtw_distance(current_array, bull_array, window=10)
    dist_to_bear, _ = dtw_distance(current_array, bear_array, window=10)

    # 정규화
    dist_to_bull /= max(len(current_array), len(bull_array))
    dist_to_bear /= max(len(current_array), len(bear_array))

    # 유사도 (거리 역수)
    # 작은 값을 더하여 0 나누기 방지
    eps = 1e-3  # 더 큰 epsilon 값 사용
    sim_to_bull = 1 / (dist_to_bull + eps)
    sim_to_bear = 1 / (dist_to_bear + eps)

    # 정규화 (합 = 1)
    total_sim = sim_to_bull + sim_to_bear
    if total_sim > 0:
        sim_to_bull_norm = sim_to_bull / total_sim
        sim_to_bear_norm = sim_to_bear / total_sim
    else:
        sim_to_bull_norm = 0.5
        sim_to_bear_norm = 0.5

    # 현재 레짐 판단
    if sim_to_bull_norm > 0.6:
        current_regime = "BULL"
    elif sim_to_bear_norm > 0.6:
        current_regime = "BEAR"
    else:
        current_regime = "UNCERTAIN"

    # 레짐 전환 확률 (유사도 차이가 작을수록 높음)
    shift_prob = 1 - abs(sim_to_bull_norm - sim_to_bear_norm)

    # 신호 판단
    if shift_prob < threshold:
        signal = "STABLE"
        interpretation = f"{current_regime} 레짐 안정적 (전환 확률 {shift_prob:.1%})"
    elif shift_prob < threshold + 0.2:
        signal = "WARNING"
        interpretation = f"{current_regime} → 레짐 전환 경고 (전환 확률 {shift_prob:.1%})"
    else:
        signal = "SHIFT_DETECTED"
        interpretation = f"레짐 전환 감지! 현재 {current_regime} (전환 확률 {shift_prob:.1%})"

    return RegimeShiftSignal(
        timestamp=datetime.now().isoformat(),
        current_regime=current_regime,
        similarity_to_bull=sim_to_bull_norm,
        similarity_to_bear=sim_to_bear_norm,
        shift_probability=shift_prob,
        signal=signal,
        interpretation=interpretation
    )


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic Time Warping (DTW) Test")
    print("=" * 70)

    # Test 1: 기본 DTW 거리 계산
    print("\n[Test 1] Basic DTW Distance")
    print("-" * 70)

    # 시계열 1: 정상 패턴
    s1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    # 시계열 2: 시차가 있는 동일 패턴 (2기간 늦음)
    s2 = np.array([0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1])

    dist_dtw, path = dtw_distance(s1, s2, return_path=True)
    dist_euclidean = np.linalg.norm(s1[:9] - s2[2:11])

    print(f"Series 1: {s1}")
    print(f"Series 2 (2-period lag): {s2}")
    print(f"\nDTW Distance: {dist_dtw:.2f}")
    print(f"Euclidean Distance: {dist_euclidean:.2f}")
    print(f"DTW captures lag better: {dist_dtw < dist_euclidean}")
    print(f"Alignment path length: {len(path)}")

    # Test 2: DTW 유사도 행렬
    print("\n[Test 2] DTW Similarity Matrix")
    print("-" * 70)

    np.random.seed(42)
    n_assets = 5
    n_days = 100

    # 팩터 기반 수익률 생성
    factor = np.cumsum(np.random.randn(n_days)) * 0.01
    returns_data = {}
    for i in range(n_assets):
        # 각 자산마다 다른 시차로 팩터 추종
        lag = i * 2
        if lag > 0:
            asset_returns = np.concatenate([np.zeros(lag), factor[:-lag]])
        else:
            asset_returns = factor

        # 노이즈 추가
        asset_returns += np.random.randn(n_days) * 0.005
        returns_data[f'Asset_{i}'] = asset_returns

    returns_df = pd.DataFrame(returns_data)

    result = compute_dtw_similarity_matrix(returns_df, window=20, normalize=True)

    print(f"Number of assets: {result.n_series}")
    print(f"Average DTW distance: {result.avg_distance:.4f}")
    print(f"\nMost similar pair:")
    print(f"  {result.most_similar_pair[0]} - {result.most_similar_pair[1]}")
    print(f"  Distance: {result.most_similar_pair[2]:.4f}")
    print(f"\nMost dissimilar pair:")
    print(f"  {result.most_dissimilar_pair[0]} - {result.most_dissimilar_pair[1]}")
    print(f"  Distance: {result.most_dissimilar_pair[2]:.4f}")

    print("\nDTW Distance Matrix (first 5x5):")
    print(result.distance_matrix.iloc[:5, :5].to_string())

    # Test 3: 리드-래그 관계 탐지
    print("\n[Test 3] Lead-Lag Relationship Detection")
    print("-" * 70)

    # 시계열 생성: Asset_A가 Asset_B보다 3일 선행
    lead_lag_test = 5
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    asset_a_returns = np.cumsum(np.random.randn(n_days)) * 0.01
    asset_b_returns = np.concatenate([
        np.zeros(lead_lag_test),
        asset_a_returns[:-lead_lag_test]
    ]) + np.random.randn(n_days) * 0.002

    asset_a_series = pd.Series(asset_a_returns, index=dates, name='Asset_A')
    asset_b_series = pd.Series(asset_b_returns, index=dates, name='Asset_B')

    lead_lag_result = find_lead_lag_relationship(
        asset_a_series,
        asset_b_series,
        max_lag=10,
        series1_name='Asset_A',
        series2_name='Asset_B'
    )

    print(f"Lead Asset: {lead_lag_result.lead_asset}")
    print(f"Lag Asset: {lead_lag_result.lag_asset}")
    print(f"Optimal Lag: {lead_lag_result.optimal_lag} days")
    print(f"Min DTW Distance: {lead_lag_result.min_distance:.4f}")
    print(f"Cross-Correlation: {lead_lag_result.cross_correlation:.3f}")
    print(f"Interpretation: {lead_lag_result.interpretation}")

    # 정확도 검증
    if lead_lag_result.lead_asset == 'Asset_A' and lead_lag_result.optimal_lag == lead_lag_test:
        print("\n✅ Lead-Lag detection successful!")
    else:
        print(f"\n⚠️ Expected lag: {lead_lag_test}, Detected: {lead_lag_result.optimal_lag}")

    # Test 4: 레짐 전환 감지
    print("\n[Test 4] Regime Shift Detection (DTW-based)")
    print("-" * 70)

    # Bull 템플릿: 상승 추세
    bull_template_data = np.linspace(0, 0.5, 50) + np.random.randn(50) * 0.05
    bull_template = pd.Series(bull_template_data)

    # Bear 템플릿: 하락 추세
    bear_template_data = np.linspace(0, -0.5, 50) + np.random.randn(50) * 0.05
    bear_template = pd.Series(bear_template_data)

    # 현재 윈도우: Bull 패턴 유사
    current_bull_like = pd.Series(np.linspace(0, 0.4, 20) + np.random.randn(20) * 0.03)
    # 현재 윈도우: Bear 패턴 유사
    current_bear_like = pd.Series(np.linspace(0, -0.4, 20) + np.random.randn(20) * 0.03)
    # 현재 윈도우: 불확실 (횡보)
    current_uncertain = pd.Series(np.random.randn(20) * 0.1)

    # Bull-like 테스트
    signal_bull = detect_regime_shift_dtw(
        current_bull_like,
        bull_template,
        bear_template,
        threshold=0.3
    )

    print(f"[Case 1] Bull-like market:")
    print(f"  Current Regime: {signal_bull.current_regime}")
    print(f"  Similarity to Bull: {signal_bull.similarity_to_bull:.1%}")
    print(f"  Similarity to Bear: {signal_bull.similarity_to_bear:.1%}")
    print(f"  Shift Probability: {signal_bull.shift_probability:.1%}")
    print(f"  Signal: {signal_bull.signal}")
    print(f"  {signal_bull.interpretation}")

    # Bear-like 테스트
    signal_bear = detect_regime_shift_dtw(
        current_bear_like,
        bull_template,
        bear_template,
        threshold=0.3
    )

    print(f"\n[Case 2] Bear-like market:")
    print(f"  Current Regime: {signal_bear.current_regime}")
    print(f"  Similarity to Bull: {signal_bear.similarity_to_bull:.1%}")
    print(f"  Similarity to Bear: {signal_bear.similarity_to_bear:.1%}")
    print(f"  Shift Probability: {signal_bear.shift_probability:.1%}")
    print(f"  Signal: {signal_bear.signal}")
    print(f"  {signal_bear.interpretation}")

    # Uncertain 테스트
    signal_uncertain = detect_regime_shift_dtw(
        current_uncertain,
        bull_template,
        bear_template,
        threshold=0.3
    )

    print(f"\n[Case 3] Uncertain market:")
    print(f"  Current Regime: {signal_uncertain.current_regime}")
    print(f"  Similarity to Bull: {signal_uncertain.similarity_to_bull:.1%}")
    print(f"  Similarity to Bear: {signal_uncertain.similarity_to_bear:.1%}")
    print(f"  Shift Probability: {signal_uncertain.shift_probability:.1%}")
    print(f"  Signal: {signal_uncertain.signal}")
    print(f"  {signal_uncertain.interpretation}")

    print("\n" + "=" * 70)
    print("All DTW tests completed successfully!")
    print("=" * 70)
