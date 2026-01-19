"""
XAI Explainer Module
====================
SHAP (SHapley Additive exPlanations) 기반 모델 설명 모듈

이 모듈은 "결과값만 내놓는 AI"가 아닌 "근거를 설명하는 AI"를 구현합니다.
금융 모델의 예측(예: Fed 금리 결정, 시장 방향성)에 대해 각 변수가
얼마나 기여했는지 정량적으로 설명합니다.

기능:
1. Global Importance: 모델 전체에서 가장 중요한 변수 순위
2. Local Explanation: 특정 예측(오늘의 시장)에 대한 변수별 기여도
3. Dependency Analysis: 변수 변화에 따른 예측값 변화 패턴

References:
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not found. XAI features will be disabled. Install with: pip install shap")

# 로거 설정
logger = logging.getLogger('eimas.xai')


@dataclass
class XAIExplanation:
    """단일 예측에 대한 설명 결과"""
    prediction: float
    base_value: float  # 모델의 평균 예측값
    contributions: Dict[str, float]  # {변수명: SHAP value}
    top_drivers: List[Tuple[str, float]]  # 상위 기여 변수 [(변수명, SHAP)]
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'prediction': float(self.prediction),
            'base_value': float(self.base_value),
            'contributions': self.contributions,
            'top_drivers': [{'feature': f, 'impact': v} for f, v in self.top_drivers],
            'timestamp': self.timestamp
        }


@dataclass
class GlobalImportance:
    """전역 변수 중요도 결과"""
    feature_importance: Dict[str, float]  # {변수명: mean(|SHAP|)}
    top_features: List[str]
    summary_plot: Optional[Any] = None  # matplotlib figure or similar (optional)

    def to_dict(self) -> Dict:
        return {
            'feature_importance': self.feature_importance,
            'top_features': self.top_features
        }


class ModelExplainer:
    """
    SHAP 기반 모델 설명기
    
    지원 모델:
    - Linear Models (Lasso, Ridge, LogisticRegression 등)
    - Tree Models (RandomForest, XGBoost, LightGBM 등)
    - Generic Blackbox (KernelExplainer 사용 - 느림)
    """

    def __init__(self, model: Any, X_train: pd.DataFrame, model_type: str = 'auto'):
        """
        Args:
            model: 학습된 sklearn/xgboost/lightgbm 모델 객체
            X_train: 학습에 사용된 데이터 (Background distribution용)
            model_type: 'linear', 'tree', 'kernel', or 'auto'
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for ModelExplainer")

        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.explainer = None
        
        self._initialize_explainer(model_type)

    def _initialize_explainer(self, model_type: str):
        """적절한 SHAP Explainer 초기화"""
        try:
            if model_type == 'linear':
                # 선형 모델용 (빠름)
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                logger.info("Initialized SHAP LinearExplainer")
                
            elif model_type == 'tree':
                # 트리 모델용 (빠름)
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized SHAP TreeExplainer")
                
            elif model_type == 'kernel':
                # 일반 모델용 (느림, 근사치)
                # K-means로 배경 데이터 요약하여 속도 향상
                X_summary = shap.kmeans(self.X_train, min(50, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict, X_summary)
                logger.info("Initialized SHAP KernelExplainer")
                
            else: # 'auto'
                # 자동 감지 시도
                # 1. 선형 모델 체크
                if hasattr(self.model, 'coef_') and not hasattr(self.model, 'feature_importances_'):
                    self.explainer = shap.LinearExplainer(self.model, self.X_train)
                    logger.info("Auto-detected Linear Model -> Using LinearExplainer")
                    
                # 2. 트리 모델 체크 (sklearn, xgboost, lightgbm)
                elif hasattr(self.model, 'feature_importances_') or \
                     type(self.model).__name__.lower().find('xgb') >= 0 or \
                     type(self.model).__name__.lower().find('lgbm') >= 0:
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("Auto-detected Tree Model -> Using TreeExplainer")
                
                # 3. 나머지는 Kernel
                else:
                    logger.warning("Could not detect model type. Falling back to KernelExplainer (slow).")
                    X_summary = shap.kmeans(self.X_train, min(50, len(self.X_train)))
                    self.explainer = shap.KernelExplainer(self.model.predict, X_summary)
        
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            raise

    def explain_local(self, X_instance: pd.DataFrame, top_n: int = 5) -> XAIExplanation:
        """
        단일 샘플에 대한 지역적 설명 (Local Explanation)
        
        Args:
            X_instance: 설명할 샘플 데이터 (1 row DataFrame)
            top_n: 반환할 상위 기여 변수 개수
            
        Returns:
            XAIExplanation 객체
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        try:
            # SHAP 값 계산
            shap_values = self.explainer.shap_values(X_instance)
            
            # 리스트/배열 구조 처리 (모델마다 다름)
            if isinstance(shap_values, list):
                # 분류 모델의 경우 클래스별로 리스트 반환됨 -> 첫 번째(또는 양성) 클래스 선택
                # 이진 분류라고 가정하고 1번 인덱스 사용 (또는 회귀면 그냥 값)
                if len(shap_values) == 2:
                    vals = shap_values[1][0]
                else:
                    vals = shap_values[0][0]
            else:
                # 회귀 모델 등
                vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Base value (Expected value)
            base_val = self.explainer.expected_value
            if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
                base_val = base_val[-1] # 마지막 값 사용 (단순화)
            
            # 예측값
            prediction = base_val + np.sum(vals)
            
            # 기여도 딕셔너리
            contributions = {
                feat: float(val) 
                for feat, val in zip(self.feature_names, vals)
            }
            
            # 상위 기여 변수 추출 (절대값 기준 정렬)
            sorted_feats = sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            top_drivers = sorted_feats[:top_n]
            
            return XAIExplanation(
                prediction=float(prediction),
                base_value=float(base_val),
                contributions=contributions,
                top_drivers=top_drivers,
                timestamp=X_instance.index[0].isoformat() if hasattr(X_instance.index, 'isoformat') else None
            )

        except Exception as e:
            logger.error(f"Local explanation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def explain_global(self, X_sample: Optional[pd.DataFrame] = None) -> GlobalImportance:
        """
        전체 모델에 대한 전역적 중요도 (Global Importance)
        
        Args:
            X_sample: 설명할 데이터 샘플 (None이면 학습 데이터 전체/일부 사용)
            
        Returns:
            GlobalImportance 객체
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        target_X = X_sample if X_sample is not None else self.X_train
        
        try:
            # 전체 데이터에 대한 SHAP 값
            shap_values = self.explainer.shap_values(target_X)
            
            if isinstance(shap_values, list):
                vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                vals = shap_values

            # Global Importance: Mean(|SHAP value|)
            feature_importance = np.abs(vals).mean(axis=0)
            
            importance_dict = {
                feat: float(imp)
                for feat, imp in zip(self.feature_names, feature_importance)
            }
            
            # 중요도 순 정렬
            sorted_features = sorted(
                importance_dict.keys(),
                key=lambda k: importance_dict[k],
                reverse=True
            )
            
            return GlobalImportance(
                feature_importance=importance_dict,
                top_features=sorted_features
            )

        except Exception as e:
            logger.error(f"Global explanation failed: {e}")
            return None

    def plot_waterfall(self, X_instance: pd.DataFrame, max_display: int = 10, save_path: Optional[str] = None):
        """
        특정 예측에 대한 Waterfall Plot 생성 및 저장
        
        Args:
            X_instance: 설명할 샘플 (1 row)
            max_display: 표시할 최대 변수 수
            save_path: 저장할 파일 경로 (None이면 화면 표시)
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return

        try:
            # SHAP Explanation 객체 생성 (최신 SHAP 버전에 적합)
            shap_values = self.explainer(X_instance)
            
            # 첫 번째 샘플 선택
            sample_shap = shap_values[0]
            
            plt.figure()
            shap.plots.waterfall(sample_shap, max_display=max_display, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Waterfall plot failed: {e}")

    def plot_summary(self, X_sample: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        """
        전체 데이터에 대한 Summary Plot (Beeswarm) 생성 및 저장
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return

        target_X = X_sample if X_sample is not None else self.X_train

        try:
            shap_values = self.explainer.shap_values(target_X)
            
            if isinstance(shap_values, list):
                vals = shap_values[1] # 이진 분류의 양성 클래스
            else:
                vals = shap_values

            plt.figure()
            shap.summary_plot(vals, target_X, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Summary plot failed: {e}")


# ============================================================================ 
# Wrapper Function for Easy Integration
# ============================================================================ 

def explain_lasso_prediction(
    lasso_forecaster_model: Any,  # sklearn LassoCV object
    X_train: pd.DataFrame,
    X_current: pd.DataFrame
) -> Dict[str, Any]:
    """
    LASSO 모델 예측을 설명하는 간편 함수 (EIMAS 파이프라인 연동용)
    
    Args:
        lasso_forecaster_model: 학습된 LassoCV 모델 (LASSOForecaster._fitted_model)
        X_train: 학습 데이터
        X_current: 설명할 현재 데이터 (1 row)
        
    Returns:
        분석 결과 딕셔너리
    """
    if lasso_forecaster_model is None:
        return {"error": "Model not fitted"}

    explainer = ModelExplainer(lasso_forecaster_model, X_train, model_type='linear')
    
    # 1. Local Explanation
    local_exp = explainer.explain_local(X_current)
    
    # 2. Global Importance (Top 5)
    global_exp = explainer.explain_global()
    
    if local_exp and global_exp:
        # 텍스트 설명 생성
        top_factor = local_exp.top_drivers[0]
        direction = "상승" if top_factor[1] > 0 else "하락"
        
        narrative = (
            f"모델은 예측값 {local_exp.prediction:.4f}를 제시했습니다 (기준값 {local_exp.base_value:.4f}). "
            f"가장 큰 영향을 미친 요인은 '{top_factor[0]}'이며, "
            f"이로 인해 예측값이 {abs(top_factor[1]):.4f}만큼 {direction}했습니다."
        )
        
        return {
            'prediction': local_exp.prediction,
            'base_value': local_exp.base_value,
            'narrative': narrative,
            'top_contributors': [
                {'name': name, 'impact': val} 
                for name, val in local_exp.top_drivers
            ],
            'global_top_features': global_exp.top_features[:5]
        }
    
    return {"error": "Explanation failed"}


# ============================================================================ 
# Test
# ============================================================================ 

if __name__ == "__main__":
    print("=" * 60)
    print("XAI Explainer Module Test")
    print("=" * 60)

    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping test.")
        exit(0)

    try:
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.ensemble import RandomForestRegressor
        
        # 샘플 데이터 생성
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.rand(100, 5), columns=['F1', 'F2', 'F3', 'F4', 'F5'])
        # y = 2*F1 - 1*F2 + noise (F3, F4, F5는 관계 없음)
        y_train = 2 * X_train['F1'] - 1 * X_train['F2'] + np.random.normal(0, 0.1, 100)
        
        X_test = pd.DataFrame(np.random.rand(1, 5), columns=['F1', 'F2', 'F3', 'F4', 'F5'])
        
        # 1. Linear Model Test
        print("\n[1] Linear Model Test (Lasso)")
        model_linear = Lasso(alpha=0.01)
        model_linear.fit(X_train, y_train)
        
        explainer_linear = ModelExplainer(model_linear, X_train, model_type='linear')
        local_res = explainer_linear.explain_local(X_test)
        
        print(f"   Prediction: {local_res.prediction:.4f}")
        print(f"   Base Value: {local_res.base_value:.4f}")
        print("   Top Drivers:")
        for name, imp in local_res.top_drivers:
            print(f"     - {name}: {imp:+.4f}")
            
        # 2. Tree Model Test
        print("\n[2] Tree Model Test (RandomForest)")
        model_tree = RandomForestRegressor(n_estimators=10, random_state=42)
        model_tree.fit(X_train, y_train)
        
        explainer_tree = ModelExplainer(model_tree, X_train, model_type='tree')
        local_res_tree = explainer_tree.explain_local(X_test)
        
        print(f"   Prediction: {local_res_tree.prediction:.4f}")
        print("   Top Drivers:")
        for name, imp in local_res_tree.top_drivers:
            print(f"     - {name}: {imp:+.4f}")

        # 3. Wrapper Function Test
        print("\n[3] Wrapper Function Test")
        wrapper_res = explain_lasso_prediction(model_linear, X_train, X_test)
        if 'error' not in wrapper_res:
            print(f"   Narrative: {wrapper_res['narrative']}")
        else:
            print(f"   Error: {wrapper_res['error']}")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
