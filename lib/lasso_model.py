#!/usr/bin/env python3
"""
LASSO Forecaster Module
========================
LASSO (L1 정규화) 기반 Fed 금리 예측 모델 래퍼 클래스

경제학적 배경:
- LASSO는 고차원 변수 선택에 효과적 (Tibshirani, 1996)
- Post-LASSO OLS로 편향 없는 추정 (Belloni & Chernozhukov, 2013)
- HAC 표준오차로 시계열 자기상관 처리 (Newey-West)

참고: Treasury 관련 변수는 Simultaneity bias 방지를 위해 제외
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# statsmodels 관련 import
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. HAC and VIF calculations will be disabled.")

# 로거 설정
logger = logging.getLogger('eimas.lasso')


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class LASSOConfig:
    """
    LASSO 모델 설정
    
    Attributes:
        n_splits: TimeSeriesSplit의 fold 수
        max_iter: LASSO 최대 반복 횟수
        tol: 수렴 허용오차
        hac_lag: Newey-West HAC 표준오차의 lag 수 (1주일 거래일 = 5)
        excluded_prefixes: 제외할 변수 접두사 목록
        random_state: 재현성을 위한 랜덤 시드
    """
    n_splits: int = 5
    max_iter: int = 10000
    tol: float = 1e-4
    hac_lag: int = 5
    excluded_prefixes: List[str] = field(default_factory=list)
    random_state: Optional[int] = 42

    def __post_init__(self):
        """기본 제외 변수 설정"""
        if not self.excluded_prefixes:
            self.excluded_prefixes = []


# 기본 제외 변수 (Treasury 관련 - Simultaneity bias 방지)
DEFAULT_EXCLUDED_VARS = [
    'd_US10Y', 'd_US2Y', 'd_RealYield10Y', 'd_Term_Spread',
    'Ret_Treasury_7_10Y', 'Ret_Treasury_1_3Y', 'Ret_Treasury_20Y'
]


@dataclass
class LASSOResult:
    """
    LASSO 학습 결과
    
    Attributes:
        horizon: 예측 horizon ("VeryShort" / "Short" / "Long")
        lambda_optimal: CV로 선택된 최적 lambda (alpha)
        selected_variables: 계수 != 0인 변수 목록
        coefficients: {변수명: 표준화 계수}
        r_squared: 결정계수 (0~1)
        n_observations: 관측치 수
        n_selected: 선택된 변수 수
    """
    horizon: str
    lambda_optimal: float
    selected_variables: List[str]
    coefficients: Dict[str, float]
    r_squared: float
    n_observations: int
    n_selected: int


# ============================================================================
# Main LASSO Forecaster Class
# ============================================================================

class LASSOForecaster:
    """
    LASSO 기반 Fed 금리 예측 모델 래퍼 클래스
    
    Fed 금리 기대 변화를 예측하기 위한 LASSO 모델을 래핑.
    TimeSeriesSplit을 사용한 교차검증으로 최적 정규화 파라미터 선택.
    
    Example:
        >>> config = LASSOConfig(n_splits=5)
        >>> forecaster = LASSOForecaster(config)
        >>> result = forecaster.fit(X, y, horizon='Long')
        >>> print(f"R²: {result.r_squared:.4f}")
        >>> print(f"Selected: {result.selected_variables}")
    """
    
    def __init__(self, config: Optional[LASSOConfig] = None):
        """
        LASSOForecaster 초기화
        
        Args:
            config: LASSO 설정 객체. None이면 기본값 사용.
        """
        self.config = config or LASSOConfig()
        
        # TimeSeriesSplit 설정
        self._cv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        # LASSO 모델 (fit 시 초기화)
        self._fitted_model: Optional[LassoCV] = None
        
        # StandardScaler
        self._scaler = StandardScaler()
        
        # 내부 상태
        self._selected_vars: List[str] = []
        self._feature_names: List[str] = []
        self._is_fitted: bool = False
        
        logger.info(f"LASSOForecaster initialized with config: n_splits={self.config.n_splits}, "
                   f"max_iter={self.config.max_iter}")
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        horizon: str
    ) -> LASSOResult:
        """
        LASSO 모델 학습 및 변수 선택
        
        Args:
            X: 설명변수 DataFrame (N x P), 컬럼명 = 변수명
            y: 종속변수 Series (d_Exp_Rate, 기대금리 일별 변화)
            horizon: 예측 horizon ("VeryShort" / "Short" / "Long")
            
        Returns:
            LASSOResult: 학습 결과
            
        Raises:
            ValueError: X가 비어있는 경우
        """
        # 1. 입력 검증
        if X.empty:
            raise ValueError("Empty feature matrix")
        
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        logger.info(f"Starting LASSO fit: horizon={horizon}, n_obs={len(X)}, n_vars={X.shape[1]}")
        
        # 2. Treasury 변수 필터링
        X_filtered = self._filter_treasury_variables(X)
        logger.info(f"After Treasury filtering: {X_filtered.shape[1]} variables remain")
        
        # 3. 결측치 처리
        mask = ~(X_filtered.isna().any(axis=1) | y.isna())
        X_clean = X_filtered[mask]
        y_clean = y[mask]
        
        if len(X_clean) < self.config.n_splits + 1:
            logger.warning(f"Insufficient data for CV: {len(X_clean)} observations")
            return LASSOResult(
                horizon=horizon,
                lambda_optimal=0.0,
                selected_variables=[],
                coefficients={},
                r_squared=0.0,
                n_observations=len(X_clean),
                n_selected=0
            )
        
        # 4. 특성명 저장
        self._feature_names = X_clean.columns.tolist()
        
        # 5. 표준화 (y는 표준화하지 않음 - 해석 용이성)
        X_scaled = self._scaler.fit_transform(X_clean)
        
        # 6. LASSO 학습 (재시도 로직 포함)
        max_retries = 3
        current_max_iter = self.config.max_iter
        
        for attempt in range(max_retries):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    
                    self._fitted_model = LassoCV(
                        cv=self._cv,
                        max_iter=current_max_iter,
                        tol=self.config.tol,
                        random_state=self.config.random_state,
                        n_jobs=-1  # 병렬 처리
                    )
                    self._fitted_model.fit(X_scaled, y_clean)
                    break
                    
            except Exception as e:
                logger.warning(f"LASSO fit attempt {attempt + 1} failed: {e}")
                current_max_iter *= 2
                
                if attempt == max_retries - 1:
                    logger.error(f"LASSO convergence failed after {max_retries} attempts")
                    raise
        
        # 7. 결과 추출
        lambda_optimal = self._fitted_model.alpha_
        coefficients = self._fitted_model.coef_
        
        # 선택된 변수 (계수 != 0)
        selected_idx = np.where(coefficients != 0)[0]
        self._selected_vars = [self._feature_names[i] for i in selected_idx]
        
        # 계수 딕셔너리
        coef_dict = {
            self._feature_names[i]: float(coefficients[i])
            for i in selected_idx
        }
        
        # 8. R² 계산
        y_pred = self._fitted_model.predict(X_scaled)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # 0-1 범위로 제한
        
        self._is_fitted = True
        
        # 9. 결과 로깅
        n_selected = len(self._selected_vars)
        if n_selected == 0:
            logger.warning(f"No variables selected for horizon={horizon} (expected for VeryShort)")
        else:
            logger.info(f"LASSO fit completed: horizon={horizon}, R²={r_squared:.4f}, "
                       f"selected={n_selected}, λ={lambda_optimal:.6f}")
        
        return LASSOResult(
            horizon=horizon,
            lambda_optimal=lambda_optimal,
            selected_variables=self._selected_vars,
            coefficients=coef_dict,
            r_squared=r_squared,
            n_observations=len(y_clean),
            n_selected=n_selected
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        학습된 모델로 예측 수행
        
        Args:
            X: 설명변수 DataFrame
            
        Returns:
            예측값 배열
            
        Raises:
            ValueError: 모델이 학습되지 않은 경우
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Treasury 변수 필터링
        X_filtered = self._filter_treasury_variables(X)
        
        # 학습 시 사용된 변수만 선택
        X_aligned = X_filtered.reindex(columns=self._feature_names, fill_value=0)
        
        # 표준화
        X_scaled = self._scaler.transform(X_aligned)
        
        return self._fitted_model.predict(X_scaled)
    
    def get_selected_variables(self) -> List[str]:
        """선택된 변수 목록 반환"""
        return self._selected_vars.copy()
    
    def get_coefficients(self) -> Dict[str, float]:
        """선택된 변수의 계수 반환"""
        if not self._is_fitted or self._fitted_model is None:
            return {}
        
        coefficients = self._fitted_model.coef_
        return {
            self._feature_names[i]: float(coefficients[i])
            for i in range(len(coefficients))
            if coefficients[i] != 0
        }
    
    def compute_hac_standard_errors(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Post-LASSO OLS의 HAC(Newey-West) 표준오차 계산
        
        LASSO로 선택된 변수만 사용하여 OLS 회귀 후
        Newey-West HAC 표준오차 계산.
        
        Args:
            X: LASSO 선택된 변수만 포함한 DataFrame
            y: 종속변수 Series
            
        Returns:
            {변수명: HAC 표준오차} 딕셔너리
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Returning empty dict.")
            return {}
        
        if X.empty or len(self._selected_vars) == 0:
            return {}
        
        try:
            # 선택된 변수만 사용
            X_selected = X[self._selected_vars] if all(v in X.columns for v in self._selected_vars) else X
            
            # 결측치 제거
            mask = ~(X_selected.isna().any(axis=1) | y.isna())
            X_clean = X_selected[mask]
            y_clean = y[mask]
            
            if len(X_clean) < len(self._selected_vars) + 2:
                logger.warning("Insufficient observations for HAC estimation")
                return {}
            
            # OLS with HAC standard errors
            X_with_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_with_const)
            results = model.fit(
                cov_type='HAC', 
                cov_kwds={'maxlags': self.config.hac_lag}
            )
            
            # 표준오차 추출 (상수항 제외)
            std_errors = results.bse
            return {
                var: float(std_errors.get(var, 0.0))
                for var in X_clean.columns
            }
            
        except Exception as e:
            logger.error(f"HAC standard error computation failed: {e}")
            return {}
    
    def compute_vif_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        다중공선성 진단 (VIF 점수 계산)
        
        VIF > 10이면 다중공선성 주의 필요.
        
        Args:
            X: 설명변수 DataFrame
            
        Returns:
            {변수명: VIF 점수} 딕셔너리
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Returning empty dict.")
            return {}
        
        if X.empty or X.shape[1] < 2:
            return {}
        
        try:
            # 결측치 제거
            X_clean = X.dropna()
            
            if len(X_clean) < X.shape[1] + 2:
                logger.warning("Insufficient observations for VIF calculation")
                return {}
            
            # 상수항 추가
            X_with_const = sm.add_constant(X_clean)
            
            # 각 변수별 VIF 계산
            vif_scores = {}
            for i, col in enumerate(X_clean.columns):
                try:
                    vif = variance_inflation_factor(X_with_const.values, i + 1)
                    vif_scores[col] = float(vif) if np.isfinite(vif) else 999.0
                except Exception:
                    vif_scores[col] = 999.0  # 계산 불가 시 높은 값
            
            # 높은 VIF 경고
            high_vif = {k: v for k, v in vif_scores.items() if v > 10}
            if high_vif:
                logger.warning(f"High VIF detected (>10): {list(high_vif.keys())}")
            
            return vif_scores
            
        except Exception as e:
            logger.error(f"VIF computation failed: {e}")
            return {}
    
    def _filter_treasury_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simultaneity bias 방지를 위해 Treasury 변수 제외
        
        Treasury 변수들은 금리 기대와 동시적으로 결정되므로
        예측 모델에 포함하면 내생성 문제 발생.
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            Treasury 변수가 제외된 DataFrame
        """
        # 제외할 변수 목록
        exclude_list = DEFAULT_EXCLUDED_VARS + self.config.excluded_prefixes
        
        # 제외 대상 컬럼 찾기
        cols_to_drop = [
            col for col in df.columns
            if any(excl in col for excl in exclude_list)
        ]
        
        if cols_to_drop:
            logger.debug(f"Excluding Treasury variables: {cols_to_drop}")
        
        return df.drop(columns=cols_to_drop, errors='ignore')


# ============================================================================
# Utility Functions
# ============================================================================

def classify_horizon(days_to_meeting: int) -> Optional[str]:
    """
    FOMC 회의까지 남은 일수를 기준으로 horizon 분류
    
    Args:
        days_to_meeting: FOMC 회의까지 남은 거래일 수
        
    Returns:
        "VeryShort", "Short", "Long", 또는 None (제외 구간)
    """
    if days_to_meeting <= 30:
        return "VeryShort"
    elif 31 <= days_to_meeting <= 90:
        return "Short"
    elif days_to_meeting >= 180:
        return "Long"
    else:
        return None  # 91~179일은 제외


def get_horizon_mask(
    days_series: pd.Series, 
    horizon: str
) -> pd.Series:
    """
    주어진 horizon에 해당하는 데이터 마스크 생성
    
    Args:
        days_series: FOMC 회의까지 남은 일수 Series
        horizon: 대상 horizon ("VeryShort" / "Short" / "Long")
        
    Returns:
        Boolean mask Series
    """
    if horizon == "VeryShort":
        return days_series <= 30
    elif horizon == "Short":
        return (days_series >= 31) & (days_series <= 90)
    elif horizon == "Long":
        return days_series >= 180
    else:
        raise ValueError(f"Unknown horizon: {horizon}")


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    # 테스트
    print("=== LASSOForecaster Test ===\n")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    # 특성 데이터
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'var_{i}' for i in range(n_features)]
    )
    
    # Treasury 변수 추가 (필터링 테스트용)
    X['d_US10Y'] = np.random.randn(n_samples)
    X['d_Term_Spread'] = np.random.randn(n_samples)
    
    # 종속변수 (일부 변수와 상관관계)
    y = pd.Series(
        0.5 * X['var_0'] + 0.3 * X['var_1'] - 0.2 * X['var_2'] + np.random.randn(n_samples) * 0.5
    )
    
    # 모델 학습
    config = LASSOConfig(n_splits=3)
    forecaster = LASSOForecaster(config)
    
    print("1. Testing Treasury variable filtering...")
    X_filtered = forecaster._filter_treasury_variables(X)
    assert 'd_US10Y' not in X_filtered.columns
    assert 'd_Term_Spread' not in X_filtered.columns
    print(f"   ✓ Filtered columns: {X.shape[1]} → {X_filtered.shape[1]}")
    
    print("\n2. Testing LASSO fit...")
    result = forecaster.fit(X, y, horizon='Long')
    print(f"   ✓ R²: {result.r_squared:.4f}")
    print(f"   ✓ Selected variables: {result.selected_variables}")
    print(f"   ✓ Lambda optimal: {result.lambda_optimal:.6f}")
    
    print("\n3. Testing prediction...")
    y_pred = forecaster.predict(X)
    print(f"   ✓ Prediction shape: {y_pred.shape}")
    
    print("\n4. Testing HAC standard errors...")
    if STATSMODELS_AVAILABLE:
        hac_errors = forecaster.compute_hac_standard_errors(X_filtered, y)
        print(f"   ✓ HAC errors computed for {len(hac_errors)} variables")
    else:
        print("   ⚠ statsmodels not available, skipping")
    
    print("\n5. Testing VIF scores...")
    if STATSMODELS_AVAILABLE:
        vif_scores = forecaster.compute_vif_scores(X_filtered[forecaster.get_selected_variables()])
        high_vif = [k for k, v in vif_scores.items() if v > 10]
        print(f"   ✓ VIF computed, high VIF warnings: {high_vif}")
    else:
        print("   ⚠ statsmodels not available, skipping")
    
    print("\n6. Testing horizon classification...")
    assert classify_horizon(15) == 'VeryShort'
    assert classify_horizon(30) == 'VeryShort'
    assert classify_horizon(31) == 'Short'
    assert classify_horizon(90) == 'Short'
    assert classify_horizon(120) is None
    assert classify_horizon(180) == 'Long'
    print("   ✓ All horizon classifications correct")
    
    print("\n=== All tests passed! ===")

