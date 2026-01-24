# EIMAS GAP Analysis Report

> 기존 EIMAS 시스템 vs. DOCX 파일 분석(todolist.md) 비교 분석
>
> 마지막 업데이트: 2026-01-24

---

## 📊 Executive Summary

**전체 구현도: 52%** (100점 만점)

| 상태 | 비율 | 설명 |
|------|------|------|
| ✅ 완전 구현 | 52% | 그대로 사용 가능 |
| ⚠️ 부분 구현 | 20% | 보완 필요 |
| ❌ 미구현 | 28% | 신규 생성 필요 |

**핵심 발견사항:**
- 🟢 **강점**: 경제학 통합 (Whitening, 인과관계, 팩트체킹) - 95% 완성도
- 🟡 **보완 필요**: HFT 미세구조, HRP 고도화 - 40-70% 완성도
- 🔴 **신규 필요**: 블록체인 PoI, CNN 패턴 탐지 - 0% 완성도

---

## 🎯 카테고리별 상세 분석

### 1. 포트폴리오 최적화 및 자산 배분 (평균 65% 구현)

#### 1.1 Hierarchical Risk Parity (HRP) 고도화

**평가:** ⚠️ **70% 구현 (Systemic Similarity 누락)**

**현재 구현된 기능:**
- ✅ Correlation-Distance 변환 (`graph_clustered_portfolio.py:344-376`)
- ✅ Hierarchical Clustering (scipy linkage/dendrogram)
- ✅ Recursive Bisection (`_recursive_bisection()`, 라인 1084-1127)
- ✅ 가중치 검증

**누락된 핵심 기능:**
```python
# 🔴 MISSING: Systemic Similarity 계산
# lib/graph_clustered_portfolio.py에 추가 필요
def compute_systemic_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """
    D_bar[i,j] = sqrt(sum((D[k,i] - D[k,j])²))

    단순 correlation 초과 → 자산 간 상호작용 강도 정량화
    """
    n = distance_matrix.shape[0]
    d_bar = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_bar[i, j] = np.sqrt(np.sum((distance_matrix[:, i] - distance_matrix[:, j])**2))
    return d_bar
```

**보완 작업:**
- Systemic Similarity 로직 구현 (3-5일)
- Seriation 최적화 알고리즘 개선 (옵션)
- HRP vs. CLA 벤치마킹 테스트 추가

**우선순위:** ⭐⭐⭐ (단기 - 1주 이내)

---

#### 1.2 클러스터링 기반 포트폴리오 최적화

**평가:** ⚠️ **60% 구현 (DBSCAN, DTW 누락)**

**현재 구현된 기능:**
- ✅ K-means 클러스터링 (`_kmeans_clustering()`, 라인 716-769)
- ✅ Hierarchical Clustering (`_hierarchical_clustering()`, 라인 803-829)
- ✅ MST (Minimum Spanning Tree) (`build_mst()`, 라인 315-413)
- ⚠️ LASSO (부분 구현, 정규화 강도 자동 선택 미흡)

**누락된 핵심 기능:**
```python
# 🔴 MISSING: DBSCAN (이상치 탐지)
# lib/graph_clustered_portfolio.py에 추가
from sklearn.cluster import DBSCAN

def dbscan_outlier_detection(returns: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> List[str]:
    """
    밀도 기반 이상 자산 탐지
    Returns: 이상치 ticker 리스트
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(returns.T)
    outliers = [ticker for ticker, label in zip(returns.columns, labels) if label == -1]
    return outliers

# 🔴 MISSING: Dynamic Time Warping
# lib/time_series_similarity.py (신규 파일)
def dtw_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """시계열 리드-래그 관계 파악"""
    from dtaidistance import dtw
    return dtw.distance(series1, series2)

# 🔴 MISSING: 클러스터링 품질 평가
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering_quality(returns, labels):
    """Silhouette, Davies-Bouldin 점수 계산"""
    silhouette = silhouette_score(returns.T, labels)
    davies_bouldin = davies_bouldin_score(returns.T, labels)
    return {"silhouette": silhouette, "davies_bouldin": davies_bouldin}
```

**보완 작업:**
- DBSCAN 이상치 탐지 구현 (2-3일)
- DTW 신규 모듈 생성 (`lib/time_series_similarity.py`) (3-5일)
- Silhouette/Davies-Bouldin 평가 메트릭 추가 (1-2일)

**우선순위:** ⭐⭐ (중기 - 2-3주)

---

### 2. 시장 미세구조 및 거래 메커니즘 (평균 40% 구현)

#### 2.1 High-Frequency Trading (HFT) 환경 지표

**평가:** ⚠️ **40% 구현 (Tick Rule, Kyle's Lambda 누락)**

**현재 구현된 기능:**
- ✅ Roll's Measure (`calculate_roll_spread()`, 라인 1223-1290)
- ✅ Amihud's Illiquidity (`calculate_amihud_lambda()`, 라인 1138-1222)
- ⚠️ VPIN Approximation (일별 데이터 근사, 정확한 Volume Clock 미구현)

**누락된 핵심 기능:**
```python
# 🔴 MISSING: Tick Rule (거래 방향 분류)
# lib/microstructure.py에 추가 (1749줄)
def tick_rule_classification(prices: pd.Series) -> pd.Series:
    """
    거래 방향 분류 (Buy/Sell/Neutral)

    Rule:
    - p[t] > p[t-1]: b[t] = 1 (Buy)
    - p[t] < p[t-1]: b[t] = -1 (Sell)
    - p[t] = p[t-1]: b[t] = b[t-1] (이전 방향 유지)
    """
    b = pd.Series(index=prices.index, dtype=int)
    b.iloc[0] = 1  # 초기값 = Buy

    for i in range(1, len(prices)):
        if prices.iloc[i] > prices.iloc[i-1]:
            b.iloc[i] = 1
        elif prices.iloc[i] < prices.iloc[i-1]:
            b.iloc[i] = -1
        else:
            b.iloc[i] = b.iloc[i-1]

    return b

# 🔴 MISSING: Kyle's Lambda (Market Impact)
def kyles_lambda(price_changes: pd.Series, signed_volume: pd.Series) -> float:
    """
    Kyle's Lambda = Market Impact 계수

    모델: delta_p[t] = Lambda * (b[t] * V[t]) + error[t]
    OLS 회귀로 Lambda 추정
    """
    from sklearn.linear_model import LinearRegression

    X = signed_volume.values.reshape(-1, 1)
    y = price_changes.values

    model = LinearRegression()
    model.fit(X, y)

    lambda_value = model.coef_[0]
    r_squared = model.score(X, y)

    return {"lambda": lambda_value, "r_squared": r_squared}

# 🔴 MISSING: Volume Clock Sampling
def volume_clock_sampling(df: pd.DataFrame, volume_bucket: float) -> pd.DataFrame:
    """
    Volume 기준 동기화 샘플링 (VPIN 정확도 향상)

    Args:
        df: OHLCV 데이터프레임 (columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        volume_bucket: 각 버킷의 누적 거래량 (예: 1,000,000)

    Returns:
        Volume 기준으로 샘플링된 데이터프레임
    """
    df = df.copy()
    df['cumulative_volume'] = df['volume'].cumsum()
    df['bucket_id'] = (df['cumulative_volume'] / volume_bucket).astype(int)

    # 각 버킷의 마지막 거래 선택
    sampled = df.groupby('bucket_id').last().reset_index(drop=True)

    return sampled

# 🔴 MISSING: Quote Stuffing 탐지
def detect_quote_stuffing(order_data: pd.DataFrame, cancel_threshold: float = 0.9) -> Dict:
    """
    Quote Stuffing 탐지 (주문 취소율 > 90%)

    Args:
        order_data: 주문 데이터 (columns: ['order_id', 'action', 'timestamp'])
        cancel_threshold: 주문 취소율 임계값

    Returns:
        {'is_stuffing': bool, 'cancel_rate': float}
    """
    total_orders = len(order_data)
    canceled_orders = len(order_data[order_data['action'] == 'cancel'])
    cancel_rate = canceled_orders / total_orders

    return {
        "is_stuffing": cancel_rate > cancel_threshold,
        "cancel_rate": cancel_rate,
        "total_orders": total_orders,
        "canceled_orders": canceled_orders
    }
```

**보완 작업:**
1. Tick Rule 구현 (1-2일)
2. Kyle's Lambda 구현 (2-3일)
3. Volume Clock Sampling 구현 (2-3일)
4. Quote Stuffing 탐지 (옵션, 1-2일)
5. VPIN 정확도 개선 (Volume Clock 기반)

**우선순위:** ⭐⭐⭐⭐ (최우선 - 1주 이내)

**파일 위치:** `/home/tj/projects/autoai/eimas/lib/microstructure.py` (1749줄)

---

### 3. 블록체인 기반 인덱스 & 스마트 거래 (평균 30% 구현)

#### 3.1 Proof-of-Index (PoI) 및 온체인 퀀트 전략

**평가:** ❌ **30% 구현 (PoI 모듈 신규 필요)**

**현재 구현된 기능:**
- ✅ Stablecoin 리스크 평가 (`genius_act_macro.py`, CryptoRiskEvaluator)
- ✅ Multi-dimensional Risk Scoring (신용, 유동성, 규제, 기술)
- ⚠️ Mean Reversion Signal (부분 구현, `integrated_strategy.py`)

**누락된 핵심 기능:**
```python
# 🔴 MISSING: Proof-of-Index 전체 모듈
# lib/proof_of_index.py (신규 생성, ~400줄)

import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List

class ProofOfIndex:
    """
    Proof-of-Index (PoI) 및 온체인 퀀트 전략

    배경:
    - 기존 금융지수: 계산 블랙박스, 정산 지연 (T+2)
    - 블록체인 기반 투명성 및 실시간 검증
    """

    def __init__(self, divisor: float = 1.0):
        self.divisor = divisor
        self.index_history = []

    def calculate_index(self, prices: Dict[str, float], quantities: Dict[str, float]) -> float:
        """
        인덱스 계산: I_t = sum(P_i_t * Q_i_t) / D_t

        Args:
            prices: {ticker: price}
            quantities: {ticker: quantity}

        Returns:
            index_value: 계산된 인덱스 값
        """
        total_market_cap = sum(prices[ticker] * quantities[ticker]
                               for ticker in prices.keys())
        index_value = total_market_cap / self.divisor

        self.index_history.append({
            "timestamp": pd.Timestamp.now(),
            "value": index_value,
            "components": prices.copy()
        })

        return index_value

    def hash_index_weights(self, weights: Dict[str, float]) -> str:
        """
        SHA-256 기반 가중치 해시 생성 (On-chain 검증용)

        Args:
            weights: {ticker: weight}

        Returns:
            hash_value: SHA-256 해시 문자열
        """
        # 사전 순서로 정렬하여 재현 가능성 보장
        sorted_weights = {k: weights[k] for k in sorted(weights.keys())}
        weights_str = str(sorted_weights).encode('utf-8')

        hash_object = hashlib.sha256(weights_str)
        return hash_object.hexdigest()

    def verify_on_chain(self, hash_value: str, reference_hash: str) -> bool:
        """
        Smart Contract 기반 해시 검증

        Args:
            hash_value: 계산된 해시
            reference_hash: On-chain 참조 해시

        Returns:
            is_valid: 검증 결과
        """
        return hash_value == reference_hash

    def mean_reversion_signal(self,
                              prices: pd.Series,
                              window: int = 20,
                              threshold: float = 2.0) -> str:
        """
        Mean Reversion 퀀트 신호 생성

        Args:
            prices: 가격 시계열
            window: 이동평균 윈도우
            threshold: Z-score 임계값 (예: ±2.0)

        Returns:
            signal: 'BUY' / 'SELL' / 'HOLD'
        """
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        z_score = (prices - mean) / std

        latest_z = z_score.iloc[-1]

        if latest_z < -threshold:
            return "BUY"
        elif latest_z > threshold:
            return "SELL"
        else:
            return "HOLD"

    def backtest_strategy(self,
                         prices: pd.DataFrame,
                         initial_capital: float = 100000) -> Dict:
        """
        Mean Reversion 전략 백테스트

        Returns:
            results: {'total_return': float, 'sharpe_ratio': float, 'max_drawdown': float}
        """
        # 백테스트 로직 구현
        pass

# 사용 예시:
# poi = ProofOfIndex(divisor=100.0)
# index_value = poi.calculate_index(prices={'BTC': 50000, 'ETH': 3000},
#                                    quantities={'BTC': 1.0, 'ETH': 10.0})
# hash_val = poi.hash_index_weights({'BTC': 0.6, 'ETH': 0.4})
# signal = poi.mean_reversion_signal(btc_prices)
```

**보완 작업:**
1. `lib/proof_of_index.py` 신규 생성 (3-5일)
2. Mean Reversion 백테스트 엔진 (3-5일)
3. On-chain 데이터 수집 (Chainlink/Pyth 오라클) (5-7일)
4. Smart Contract 검증 로직 (옵션, 시뮬레이션)

**우선순위:** ⭐⭐ (중기 - 2-3주)

---

### 4. AI/ML 기술 기초

#### 4.1 Convolution 기반 시계열 패턴 탐지

**평가:** ❌ **0% 구현 (CNN 모듈 신규 필요)**

**누락된 전체 기능:**
```python
# 🔴 MISSING: CNN 패턴 탐지 모듈
# lib/cnn_pattern_detector.py (신규 생성, ~500줄)

import numpy as np
import pandas as pd
from typing import Tuple, List

class CNNPatternDetector:
    """
    Convolution 기반 시계열 패턴 탐지

    배경:
    - 주식 가격 heatmap에서 패턴 자동 추출
    - 기술적 지표 자동화 (헤드앤숄더, 삼각 수렴 등)
    """

    def __init__(self, filter_size: Tuple[int, int] = (3, 3)):
        self.filter_size = filter_size
        self.filters = self._initialize_filters()

    def _initialize_filters(self) -> Dict[str, np.ndarray]:
        """
        필터 초기화 (Edge Detection, Momentum 등)

        Returns:
            filters: {'edge': array, 'momentum': array, ...}
        """
        filters = {}

        # Edge Detection (Sobel)
        filters['edge_x'] = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        filters['edge_y'] = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        # Momentum Filter (가격 상승 패턴)
        filters['momentum'] = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ])

        return filters

    def conv2d(self,
               input_grid: np.ndarray,
               filter_kernel: np.ndarray,
               stride: int = 1) -> np.ndarray:
        """
        2D Convolution 연산

        Args:
            input_grid: 입력 이미지 (H × W)
            filter_kernel: 필터 (F_h × F_w)
            stride: 슬라이딩 간격

        Returns:
            output_map: Feature Map ((H-F_h)/stride+1 × (W-F_w)/stride+1)
        """
        H, W = input_grid.shape
        F_h, F_w = filter_kernel.shape

        # 출력 크기 계산
        out_H = (H - F_h) // stride + 1
        out_W = (W - F_w) // stride + 1

        output_map = np.zeros((out_H, out_W))

        for r in range(out_H):
            for c in range(out_W):
                # 윈도우 추출
                r_start = r * stride
                c_start = c * stride
                window = input_grid[r_start:r_start+F_h, c_start:c_start+F_w]

                # Element-wise 곱셈 및 합산
                output_map[r, c] = np.sum(window * filter_kernel)

        return output_map

    def generate_heatmap(self,
                        prices: pd.DataFrame,
                        window: int = 20) -> np.ndarray:
        """
        가격 시계열 → 2D Heatmap 변환

        Args:
            prices: 가격 데이터 (columns: tickers, index: dates)
            window: 시간 윈도우

        Returns:
            heatmap: 2D 이미지 (tickers × time)
        """
        # 정규화 (0-255 범위)
        normalized = (prices - prices.min()) / (prices.max() - prices.min()) * 255
        heatmap = normalized.values.T  # Transpose (tickers as rows)

        return heatmap

    def detect_patterns(self, prices: pd.DataFrame) -> Dict[str, List]:
        """
        패턴 탐지 (헤드앤숄더, 삼각 수렴 등)

        Returns:
            patterns: {'head_and_shoulders': [...], 'triangle': [...]}
        """
        heatmap = self.generate_heatmap(prices)

        patterns = {}

        # Edge Detection
        edge_x = self.conv2d(heatmap, self.filters['edge_x'])
        edge_y = self.conv2d(heatmap, self.filters['edge_y'])
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Momentum Detection
        momentum = self.conv2d(heatmap, self.filters['momentum'])

        # 패턴 식별 (단순 임계값 기반)
        patterns['strong_edges'] = np.where(edge_magnitude > 200)
        patterns['momentum_zones'] = np.where(momentum > 100)

        return patterns

    def validate_output_size(self,
                            input_shape: Tuple[int, int],
                            filter_shape: Tuple[int, int],
                            stride: int) -> Tuple[int, int]:
        """
        출력 크기 검증

        Formula: output = (input - filter) / stride + 1
        """
        H, W = input_shape
        F_h, F_w = filter_shape

        out_H = (H - F_h) // stride + 1
        out_W = (W - F_w) // stride + 1

        return (out_H, out_W)

# 사용 예시:
# detector = CNNPatternDetector()
# heatmap = detector.generate_heatmap(prices_df)
# patterns = detector.detect_patterns(prices_df)
# edge_map = detector.conv2d(heatmap, detector.filters['edge_x'])
```

**보완 작업:**
1. `lib/cnn_pattern_detector.py` 신규 생성 (5-7일)
2. 기술적 지표 패턴 라이브러리 구축 (7-10일)
3. 백테스트 통합 (3-5일)

**우선순위:** ⭐ (장기 - 3-4주)

---

#### 4.2 LLM 도메인 특화

**평가:** ⚠️ **50% 구현 (Fine-tuning, Multimodal 미구현)**

**현재 구현된 기능:**
- ✅ Claude/Perplexity API 사용 (`agents/orchestrator.py`)
- ✅ 토론 프로토콜 (`core/debate.py`)
- ⚠️ 팩트체킹 (`autonomous_agent.py`, 편향성 탐지 미흡)

**보완 작업:**
1. 경제학/금융 도메인 Fine-tuning 데이터셋 수집 (10K+ 샘플) (2-3주)
2. Supervised Fine-Tuning (SFT) 파이프라인 구축 (3-4주)
3. Vision Transformer 기반 차트 해석 모듈 (4-6주)
4. Bias Detection 자동화 (2-3주)

**우선순위:** ⭐⭐ (중기 - 2-3개월)

---

### 5. 경제학 통합 및 인과관계 분석 (평균 80% 구현)

#### 5.1 Causality vs. Correlation: 인과관계 네트워크

**평가:** ⚠️ **65% 구현 (GARCH, Private Info Score 누락)**

**현재 구현된 기능:**
- ✅ Granger Causality (`causality_graph.py`, 라인 1-1099)
- ✅ Shock Propagation (`shock_propagation_graph.py`, 라인 1-897)
- ✅ Sector Rotation (GMM) (`regime_analyzer.py`)
- ⚠️ Information Flow (부분, `etf_flow_analyzer.py`)

**누락된 핵심 기능:**
```python
# 🔴 MISSING: GARCH Model
# lib/regime_analyzer.py에 추가 (~450줄 → ~600줄)

from arch import arch_model

class GARCHModel:
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
    시변 변동성 모델링
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Args:
            p: ARCH 항 차수
            q: GARCH 항 차수

        Model:
            sigma_t² = ω + α·ε²_{t-1} + β·σ²_{t-1}
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None

    def fit(self, returns: pd.Series) -> Dict:
        """
        GARCH 모델 피팅

        Returns:
            params: {'omega': float, 'alpha': float, 'beta': float}
        """
        # GARCH(p,q) 모델
        self.model = arch_model(returns, vol='Garch', p=self.p, q=self.q)
        self.fitted_model = self.model.fit(disp='off')

        params = {
            'omega': self.fitted_model.params['omega'],
            'alpha': self.fitted_model.params['alpha[1]'] if self.p > 0 else 0,
            'beta': self.fitted_model.params['beta[1]'] if self.q > 0 else 0
        }

        return params

    def forecast(self, horizon: int = 20) -> pd.Series:
        """
        다중 기간 변동성 예측

        Returns:
            volatility_forecast: 예측된 조건부 분산
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        forecast = self.fitted_model.forecast(horizon=horizon)
        volatility = np.sqrt(forecast.variance.values[-1, :])

        return pd.Series(volatility, index=range(1, horizon+1))

# 🔴 MISSING: Information Flow 모듈
# lib/information_flow.py (신규 생성, ~300줄)

class InformationFlowAnalyzer:
    """
    정보 플로우 분석 (거래량 이상 탐지, Private Information Score)
    """

    def __init__(self, ma_window: int = 20, threshold: float = 5.0):
        self.ma_window = ma_window
        self.threshold = threshold

    def detect_abnormal_volume(self, volume: pd.Series) -> pd.DataFrame:
        """
        거래량 이상 탐지

        Rule: volume[t] > MA(volume, 20) * 5

        Returns:
            abnormal_dates: DataFrame with columns ['date', 'volume', 'ma', 'ratio']
        """
        ma = volume.rolling(self.ma_window).mean()
        ratio = volume / ma

        abnormal = ratio > self.threshold

        results = pd.DataFrame({
            'date': volume.index[abnormal],
            'volume': volume[abnormal].values,
            'ma': ma[abnormal].values,
            'ratio': ratio[abnormal].values
        })

        return results

    def calculate_private_info_score(self,
                                     buy_volume: pd.Series,
                                     sell_volume: pd.Series) -> pd.Series:
        """
        Private Information Extraction Score

        Formula: (volume_buy - volume_sell) / total_volume

        Interpretation:
        - > 0: Buy pressure (정보 우위 매수세)
        - < 0: Sell pressure (정보 우위 매도세)
        """
        total_volume = buy_volume + sell_volume
        score = (buy_volume - sell_volume) / total_volume

        return score

    def estimate_capm(self,
                     asset_returns: pd.Series,
                     market_returns: pd.Series) -> Dict:
        """
        CAPM Regression: E[R_i] = Alpha + Beta * E[R_m]

        Returns:
            {'alpha': float, 'beta': float, 'r_squared': float}
        """
        from sklearn.linear_model import LinearRegression

        # NaN 제거
        mask = ~(asset_returns.isna() | market_returns.isna())
        X = market_returns[mask].values.reshape(-1, 1)
        y = asset_returns[mask].values

        model = LinearRegression()
        model.fit(X, y)

        alpha = model.intercept_
        beta = model.coef_[0]
        r_squared = model.score(X, y)

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared
        }
```

**보완 작업:**
1. GARCH 모델 구현 (`regime_analyzer.py`) (3-5일)
2. Information Flow 모듈 신규 생성 (`lib/information_flow.py`) (3-5일)
3. CAPM Alpha/Beta 자동 계산 통합 (2-3일)

**우선순위:** ⭐⭐⭐ (단기 - 1-2주)

---

#### 5.2 Whitening (Explainability) 강화

**평가:** ✅ **95% 구현 (거의 완성)**

**현재 구현된 기능:**
- ✅ 경제학적 해석 레이어 (`whitening_engine.py`, 1000+ 줄)
- ✅ 인과관계 체인 추적 (`causality_graph.py` + `shock_propagation_graph.py`)
- ✅ 팩트체킹 통합 (`autonomous_agent.py`)

**소폭 보완 작업 (옵션):**
- 그래프 시각화 개선 (D3.js/Graphviz 렌더링) (3-5일)
- 실시간 Whitening (스트리밍 데이터) (5-7일)

**우선순위:** ⭐ (완료, 옵션 개선만)

---

## 🔥 즉시 실행 가능한 작업 리스트 (1주 이내)

### Week 1: HFT 미세구조 강화 (최우선)

**파일:** `/home/tj/projects/autoai/eimas/lib/microstructure.py` (1749줄)

**작업:**
1. ✅ Tick Rule 구현 (1-2일)
   ```python
   def tick_rule_classification(prices: pd.Series) -> pd.Series:
       # Buy/Sell/Neutral 분류
   ```

2. ✅ Kyle's Lambda 구현 (2-3일)
   ```python
   def kyles_lambda(price_changes: pd.Series, signed_volume: pd.Series) -> float:
       # OLS 회귀로 Lambda 추정
   ```

3. ✅ Volume Clock Sampling 구현 (2-3일)
   ```python
   def volume_clock_sampling(df: pd.DataFrame, volume_bucket: float) -> pd.DataFrame:
       # VPIN 정확도 향상
   ```

**예상 추가 코드:** ~200-300줄
**우선순위:** ⭐⭐⭐⭐

---

### Week 1-2: HRP Systemic Similarity (단기)

**파일:** `/home/tj/projects/autoai/eimas/lib/graph_clustered_portfolio.py` (1524줄)

**작업:**
```python
def compute_systemic_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """D_bar[i,j] = sqrt(sum((D[k,i] - D[k,j])²))"""
    # 자산 간 상호작용 강도 정량화
```

**예상 추가 코드:** ~50-100줄
**우선순위:** ⭐⭐⭐

---

### Week 1-2: GARCH + Information Flow (단기)

**파일 1:** `/home/tj/projects/autoai/eimas/lib/regime_analyzer.py` (~450줄)
**작업:** GARCH 클래스 추가 (~100-150줄)

**파일 2:** `/home/tj/projects/autoai/eimas/lib/information_flow.py` (신규)
**작업:** InformationFlowAnalyzer 클래스 생성 (~300줄)

**우선순위:** ⭐⭐⭐

---

## 🟡 중기 실행 작업 리스트 (2-4주)

### Month 1: 클러스터링 보완

**파일:** `/home/tj/projects/autoai/eimas/lib/graph_clustered_portfolio.py`

**작업:**
1. DBSCAN 이상치 탐지 (2-3일)
2. DTW 신규 모듈 (`lib/time_series_similarity.py`) (3-5일)
3. Silhouette/Davies-Bouldin 평가 (1-2일)

**예상 추가 코드:** ~200-300줄

**우선순위:** ⭐⭐

---

### Month 1-2: Proof-of-Index 모듈

**파일:** `/home/tj/projects/autoai/eimas/lib/proof_of_index.py` (신규, ~400줄)

**작업:**
1. ProofOfIndex 클래스 구현 (3-5일)
2. Mean Reversion 백테스트 (3-5일)
3. On-chain 데이터 연동 (5-7일)

**우선순위:** ⭐⭐

---

## 🟠 장기 실행 작업 리스트 (1-3개월)

### Month 2-3: CNN 패턴 탐지

**파일:** `/home/tj/projects/autoai/eimas/lib/cnn_pattern_detector.py` (신규, ~500줄)

**작업:**
1. CNNPatternDetector 클래스 (5-7일)
2. 기술적 지표 패턴 라이브러리 (7-10일)
3. 백테스트 통합 (3-5일)

**우선순위:** ⭐

---

### Month 2-3: LLM Fine-tuning

**작업:**
1. 경제학/금융 데이터셋 수집 (2-3주)
2. SFT 파이프라인 구축 (3-4주)
3. Vision Transformer 차트 해석 (4-6주)

**우선순위:** ⭐⭐

---

## 📊 최종 요약표

| 카테고리 | 구현도 | 즉시 필요 (1주) | 단기 (2-4주) | 중기 (1-2개월) | 장기 (3-6개월) |
|---------|-------|---------------|-------------|---------------|---------------|
| **포트폴리오 최적화** | 65% | Systemic Similarity | DBSCAN, DTW | - | - |
| **시장 미세구조** | 40% | Tick Rule, Kyle's Lambda, Volume Clock | - | - | - |
| **블록체인** | 30% | - | - | Proof-of-Index | - |
| **AI/ML** | 25% | - | - | LLM Fine-tuning | CNN 패턴 탐지 |
| **경제학 통합** | 80% | GARCH, Info Flow | - | - | - |
| **Whitening** | 95% | (완료) | - | - | - |

---

## 📁 신규 생성 필요 파일

| 파일명 | 예상 크기 | 우선순위 | 예상 작업 시간 |
|-------|---------|---------|----------------|
| `lib/information_flow.py` | ~300줄 | ⭐⭐⭐ | 3-5일 |
| `lib/proof_of_index.py` | ~400줄 | ⭐⭐ | 2-3주 |
| `lib/time_series_similarity.py` | ~200줄 | ⭐⭐ | 3-5일 |
| `lib/cnn_pattern_detector.py` | ~500줄 | ⭐ | 3-4주 |

---

## 🎯 권장 실행 순서

### Phase 1 (Week 1-2): 핵심 보완
1. ✅ Tick Rule + Kyle's Lambda + Volume Clock (`microstructure.py`)
2. ✅ Systemic Similarity (`graph_clustered_portfolio.py`)
3. ✅ GARCH 모델 (`regime_analyzer.py`)
4. ✅ Information Flow 모듈 신규 생성

**예상 총 작업 시간:** 10-14일

---

### Phase 2 (Week 3-6): 기능 확장
5. DBSCAN + DTW + 클러스터링 평가
6. Proof-of-Index 모듈 신규 생성
7. LLM Fine-tuning 데이터셋 수집 시작

**예상 총 작업 시간:** 4-6주

---

### Phase 3 (Month 3-6): 고급 기능
8. CNN 패턴 탐지 모듈
9. LLM Fine-tuning 파이프라인
10. Vision Transformer 차트 해석

**예상 총 작업 시간:** 2-3개월

---

## 📌 핵심 발견사항

### 강점 (Keep)
- ✅ **Whitening & 인과관계 분석**: 95% 완성도 (세계적 수준)
- ✅ **Stablecoin 리스크 평가**: 다차원 평가 완비
- ✅ **MST & HRP**: 기본 골격 완성

### 보완 필요 (Improve)
- ⚠️ **HFT 미세구조**: Tick Rule, Kyle's Lambda 누락
- ⚠️ **HRP**: Systemic Similarity 미구현
- ⚠️ **GARCH**: 시변 변동성 모델 부재

### 신규 필요 (Add)
- ❌ **Proof-of-Index**: 블록체인 인덱스 전체 모듈
- ❌ **CNN 패턴 탐지**: 딥러닝 기반 기술적 분석
- ❌ **LLM Fine-tuning**: 경제학 도메인 특화

---

**문서 작성:** Claude Code (Explore 에이전트)
**마지막 업데이트:** 2026-01-24 23:30 KST
**소스:** EIMAS 시스템 (v2.1.2) + todolist.md 비교 분석
