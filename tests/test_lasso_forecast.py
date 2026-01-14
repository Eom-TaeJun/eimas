#!/usr/bin/env python3
"""
LASSO Forecast Tests
====================
LASSOForecaster 및 ForecastAgent 테스트

테스트 범위:
- LASSOForecaster 단위 테스트
- ForecastAgent 통합 테스트
- Horizon 분류 테스트
- HAC/VIF 계산 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime

# 테스트 대상 임포트
from lib.lasso_model import (
    LASSOForecaster,
    LASSOConfig,
    LASSOResult,
    classify_horizon,
    get_horizon_mask,
    DEFAULT_EXCLUDED_VARS
)
from core.schemas import (
    ForecastResult,
    LASSODiagnostics,
    HorizonConfig,
    AgentRequest,
    AgentRole
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_market_data():
    """테스트용 시장 데이터 생성"""
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='B')
    
    # 특성 생성
    data = {
        # 신용 시장
        'd_Baa_Yield': np.random.randn(n_samples) * 0.1,
        'd_Spread_Baa': np.random.randn(n_samples) * 0.05,
        'd_HighYield_Rate': np.random.randn(n_samples) * 0.08,
        
        # 달러
        'Ret_Dollar_Idx': np.random.randn(n_samples) * 0.01,
        'd_Dollar_Idx': np.random.randn(n_samples) * 0.02,
        
        # 인플레이션
        'd_Breakeven5Y': np.random.randn(n_samples) * 0.02,
        'd_Breakeven10Y': np.random.randn(n_samples) * 0.015,
        
        # 리스크
        'Ret_VIX': np.random.randn(n_samples) * 0.05,
        'd_VIX': np.random.randn(n_samples) * 0.03,
        
        # 주식
        'Ret_SP500': np.random.randn(n_samples) * 0.01,
        'Ret_NASDAQ': np.random.randn(n_samples) * 0.015,
        
        # Treasury (제외 대상)
        'd_US10Y': np.random.randn(n_samples) * 0.05,
        'd_US2Y': np.random.randn(n_samples) * 0.04,
        'd_Term_Spread': np.random.randn(n_samples) * 0.03,
        
        # 타겟
        'd_Exp_Rate': (
            0.5 * np.random.randn(n_samples) * 0.1 +
            0.3 * np.random.randn(n_samples) * 0.05
        ),
        
        # days_to_meeting (다양한 horizon 포함)
        'days_to_meeting': np.random.randint(1, 400, n_samples)
    }
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def lasso_config():
    """테스트용 LASSO 설정"""
    return LASSOConfig(
        n_splits=3,  # 빠른 테스트를 위해 줄임
        max_iter=5000,
        tol=1e-4,
        hac_lag=5
    )


@pytest.fixture
def forecaster(lasso_config):
    """테스트용 LASSOForecaster 인스턴스"""
    return LASSOForecaster(lasso_config)


# ============================================================================
# TestLASSOForecaster - 단위 테스트
# ============================================================================

class TestLASSOForecaster:
    """LASSOForecaster 클래스 테스트"""
    
    def test_treasury_filter(self, forecaster, sample_market_data):
        """Treasury 변수가 정상적으로 제외되는지 확인"""
        # Given: Treasury 변수 포함된 DataFrame
        df = sample_market_data.copy()
        
        # When: 필터링 적용
        result = forecaster._filter_treasury_variables(df)
        
        # Then: Treasury 변수 제외됨
        assert 'd_US10Y' not in result.columns, "d_US10Y should be excluded"
        assert 'd_US2Y' not in result.columns, "d_US2Y should be excluded"
        assert 'd_Term_Spread' not in result.columns, "d_Term_Spread should be excluded"
        
        # 다른 변수는 유지됨
        assert 'd_Baa_Yield' in result.columns, "d_Baa_Yield should remain"
        assert 'd_Spread_Baa' in result.columns, "d_Spread_Baa should remain"
        assert 'Ret_VIX' in result.columns, "Ret_VIX should remain"
    
    def test_horizon_classification(self):
        """Horizon 분류가 정확한지 확인"""
        # VeryShort
        assert classify_horizon(1) == 'VeryShort', "1 day should be VeryShort"
        assert classify_horizon(15) == 'VeryShort', "15 days should be VeryShort"
        assert classify_horizon(30) == 'VeryShort', "30 days should be VeryShort"
        
        # Short
        assert classify_horizon(31) == 'Short', "31 days should be Short"
        assert classify_horizon(60) == 'Short', "60 days should be Short"
        assert classify_horizon(90) == 'Short', "90 days should be Short"
        
        # Excluded (91-179)
        assert classify_horizon(91) is None, "91 days should be excluded"
        assert classify_horizon(120) is None, "120 days should be excluded"
        assert classify_horizon(179) is None, "179 days should be excluded"
        
        # Long
        assert classify_horizon(180) == 'Long', "180 days should be Long"
        assert classify_horizon(250) == 'Long', "250 days should be Long"
        assert classify_horizon(365) == 'Long', "365 days should be Long"
    
    def test_horizon_config_classify(self):
        """HorizonConfig.classify 메서드 테스트"""
        config = HorizonConfig()
        
        assert config.classify(15) == 'VeryShort'
        assert config.classify(50) == 'Short'
        assert config.classify(100) is None  # 제외 구간
        assert config.classify(200) == 'Long'
    
    def test_horizon_mask(self, sample_market_data):
        """Horizon 마스크 생성 테스트"""
        days = sample_market_data['days_to_meeting']
        
        # VeryShort mask
        mask_vs = get_horizon_mask(days, 'VeryShort')
        assert mask_vs.sum() > 0, "VeryShort mask should have some True values"
        assert (days[mask_vs] <= 30).all(), "All VeryShort days should be <= 30"
        
        # Short mask
        mask_s = get_horizon_mask(days, 'Short')
        assert mask_s.sum() > 0, "Short mask should have some True values"
        assert ((days[mask_s] >= 31) & (days[mask_s] <= 90)).all()
        
        # Long mask
        mask_l = get_horizon_mask(days, 'Long')
        assert mask_l.sum() > 0, "Long mask should have some True values"
        assert (days[mask_l] >= 180).all(), "All Long days should be >= 180"
    
    def test_lasso_fit_returns_result(self, forecaster, sample_market_data):
        """LASSO fit이 LASSOResult를 반환하는지 확인"""
        # Given: 샘플 데이터
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting'])
        y = sample_market_data['d_Exp_Rate']
        
        # When: fit 실행
        result = forecaster.fit(X, y, 'Long')
        
        # Then: LASSOResult 타입, 필수 필드 존재
        assert isinstance(result, LASSOResult), "Result should be LASSOResult"
        assert result.horizon == 'Long', "Horizon should be 'Long'"
        assert 0 <= result.r_squared <= 1, "R² should be between 0 and 1"
        assert isinstance(result.selected_variables, list), "selected_variables should be list"
        assert isinstance(result.coefficients, dict), "coefficients should be dict"
        assert result.n_observations > 0, "n_observations should be positive"
    
    def test_empty_dataframe_raises_error(self, forecaster):
        """빈 DataFrame 입력 시 ValueError 발생"""
        X = pd.DataFrame()
        y = pd.Series([])
        
        with pytest.raises(ValueError, match="Empty feature matrix"):
            forecaster.fit(X, y, 'Long')
    
    def test_empty_selection_very_short(self, forecaster):
        """VeryShort horizon에서 변수 선택이 없어도 에러 없이 동작"""
        # Given: 노이즈 데이터 (설명력 없음)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'var_{i}' for i in range(5)])
        y = pd.Series(np.random.randn(50))
        
        # When: VeryShort fit
        result = forecaster.fit(X, y, 'VeryShort')
        
        # Then: 에러 없이 완료, 빈 선택 허용
        assert result is not None, "Result should not be None"
        # VeryShort에서는 변수가 거의 선택되지 않을 수 있음
        assert result.r_squared >= 0, "R² should be non-negative"
    
    def test_predict_after_fit(self, forecaster, sample_market_data):
        """fit 후 predict가 정상 동작하는지 확인"""
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting'])
        y = sample_market_data['d_Exp_Rate']
        
        # fit
        forecaster.fit(X, y, 'Long')
        
        # predict
        predictions = forecaster.predict(X)
        
        assert len(predictions) == len(X), "Predictions length should match input"
        assert not np.isnan(predictions).all(), "Predictions should not all be NaN"
    
    def test_predict_before_fit_raises_error(self, forecaster, sample_market_data):
        """fit 전에 predict 호출 시 에러"""
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting'])
        
        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.predict(X)
    
    def test_get_selected_variables(self, forecaster, sample_market_data):
        """get_selected_variables 메서드 테스트"""
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting'])
        y = sample_market_data['d_Exp_Rate']
        
        result = forecaster.fit(X, y, 'Long')
        selected = forecaster.get_selected_variables()
        
        assert selected == result.selected_variables
        assert isinstance(selected, list)
    
    def test_get_coefficients(self, forecaster, sample_market_data):
        """get_coefficients 메서드 테스트"""
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting'])
        y = sample_market_data['d_Exp_Rate']
        
        result = forecaster.fit(X, y, 'Long')
        coefficients = forecaster.get_coefficients()
        
        assert coefficients == result.coefficients
        assert isinstance(coefficients, dict)


class TestLASSODiagnostics:
    """LASSO 진단 관련 테스트"""
    
    def test_vif_calculation(self, forecaster, sample_market_data):
        """VIF 계산 정상 동작 확인"""
        X = sample_market_data[['d_Baa_Yield', 'd_Spread_Baa', 'Ret_VIX']].dropna()
        
        vif_scores = forecaster.compute_vif_scores(X)
        
        # VIF가 계산되어야 함 (statsmodels 있을 경우)
        if vif_scores:
            assert isinstance(vif_scores, dict)
            assert len(vif_scores) == 3
            assert all(isinstance(v, float) for v in vif_scores.values())
    
    def test_hac_standard_errors(self, forecaster, sample_market_data):
        """HAC 표준오차 계산 테스트"""
        X = sample_market_data.drop(columns=['d_Exp_Rate', 'days_to_meeting', 
                                              'd_US10Y', 'd_US2Y', 'd_Term_Spread'])
        y = sample_market_data['d_Exp_Rate']
        
        # fit first
        forecaster.fit(X, y, 'Long')
        
        # HAC errors for selected variables
        selected_vars = forecaster.get_selected_variables()
        if selected_vars:
            X_selected = X[selected_vars]
            hac_errors = forecaster.compute_hac_standard_errors(X_selected, y)
            
            if hac_errors:  # statsmodels available
                assert isinstance(hac_errors, dict)
                assert all(isinstance(v, float) for v in hac_errors.values())


# ============================================================================
# TestForecastAgentIntegration - 통합 테스트
# ============================================================================

class TestForecastAgentIntegration:
    """ForecastAgent 통합 테스트"""
    
    @pytest.fixture
    def forecast_agent(self):
        """테스트용 ForecastAgent"""
        from agents.forecast_agent import ForecastAgent
        return ForecastAgent()
    
    @pytest.mark.asyncio
    async def test_execute_returns_forecasts(self, forecast_agent, sample_market_data):
        """_execute가 Horizon별 예측을 반환하는지 확인"""
        # Given
        days_to_meeting = sample_market_data['days_to_meeting']
        market_data = sample_market_data.drop(columns=['days_to_meeting'])
        
        request = AgentRequest(
            task_id="test_forecast",
            role=AgentRole.FORECAST,
            instruction="Test forecast",
            context={
                'market_data': market_data,
                'days_to_meeting': days_to_meeting
            }
        )
        
        # When
        response = await forecast_agent._execute(request)
        
        # Then
        assert 'forecasts' in response, "Response should contain 'forecasts'"
        forecasts = response['forecasts']
        assert len(forecasts) == 3, "Should have 3 horizons (VeryShort, Short, Long)"
        
        # 각 horizon 확인
        horizons = [f['horizon'] if isinstance(f, dict) else f.horizon for f in forecasts]
        assert 'VeryShort' in horizons
        assert 'Short' in horizons
        assert 'Long' in horizons
    
    @pytest.mark.asyncio
    async def test_execute_includes_diagnostics(self, forecast_agent, sample_market_data):
        """_execute가 진단 정보를 포함하는지 확인"""
        days_to_meeting = sample_market_data['days_to_meeting']
        market_data = sample_market_data.drop(columns=['days_to_meeting'])
        
        request = AgentRequest(
            task_id="test_forecast",
            role=AgentRole.FORECAST,
            instruction="Test forecast",
            context={
                'market_data': market_data,
                'days_to_meeting': days_to_meeting
            }
        )
        
        response = await forecast_agent._execute(request)
        
        assert 'diagnostics' in response, "Response should contain 'diagnostics'"
        assert 'interpretation' in response, "Response should contain 'interpretation'"
        assert 'confidence' in response, "Response should contain 'confidence'"
    
    @pytest.mark.asyncio
    async def test_form_opinion_rate_direction(self, forecast_agent, sample_market_data):
        """rate_direction 토픽에 대한 의견 형성"""
        # Given: 미리 계산된 결과
        context = {
            'forecasts': [
                ForecastResult(
                    horizon='VeryShort',
                    selected_variables=[],
                    coefficients={},
                    r_squared=0.02,
                    n_observations=100,
                    lambda_optimal=0.1
                ),
                ForecastResult(
                    horizon='Short',
                    selected_variables=['d_Spread_Baa'],
                    coefficients={'d_Spread_Baa': -0.5},
                    r_squared=0.35,
                    n_observations=150,
                    lambda_optimal=0.05
                ),
                ForecastResult(
                    horizon='Long',
                    selected_variables=['d_Baa_Yield', 'd_Spread_Baa', 'Ret_Dollar_Idx'],
                    coefficients={
                        'd_Baa_Yield': 2.0,
                        'd_Spread_Baa': -1.5,
                        'Ret_Dollar_Idx': 1.0
                    },
                    r_squared=0.60,
                    n_observations=200,
                    lambda_optimal=0.03
                )
            ]
        }
        
        # When
        opinion = await forecast_agent.form_opinion('rate_direction', context)
        
        # Then
        assert opinion.topic == 'rate_direction'
        assert opinion.position in ['UP', 'DOWN', 'HOLD']
        assert 0 <= opinion.confidence <= 1
        assert len(opinion.evidence) > 0
    
    @pytest.mark.asyncio
    async def test_form_opinion_confidence_bounds(self, forecast_agent):
        """confidence 범위가 0-1 사이인지 확인"""
        context = {
            'forecasts': [
                ForecastResult(
                    horizon='Long',
                    selected_variables=['d_Baa_Yield'],
                    coefficients={'d_Baa_Yield': 1.0},
                    r_squared=0.85,  # 높은 R²
                    n_observations=200,
                    lambda_optimal=0.01
                )
            ]
        }
        
        opinion = await forecast_agent.form_opinion('forecast_confidence', context)
        
        assert 0 <= opinion.confidence <= 1, f"Confidence {opinion.confidence} out of bounds"
        # R²가 0.85여도 confidence는 0.95로 제한됨
        assert opinion.confidence <= 0.95


# ============================================================================
# TestSchemas - 스키마 테스트
# ============================================================================

class TestSchemas:
    """스키마 관련 테스트"""
    
    def test_forecast_result_to_dict(self):
        """ForecastResult.to_dict 테스트"""
        result = ForecastResult(
            horizon='Long',
            selected_variables=['var1', 'var2'],
            coefficients={'var1': 0.5, 'var2': -0.3},
            r_squared=0.64,
            n_observations=200,
            lambda_optimal=0.01
        )
        
        d = result.to_dict()
        
        assert d['horizon'] == 'Long'
        assert d['r_squared'] == 0.64
        assert 'selected_variables' in d
    
    def test_forecast_result_get_top_variables(self):
        """ForecastResult.get_top_variables 테스트"""
        result = ForecastResult(
            horizon='Long',
            selected_variables=['var1', 'var2', 'var3'],
            coefficients={'var1': 0.5, 'var2': -0.8, 'var3': 0.3},
            r_squared=0.64,
            n_observations=200,
            lambda_optimal=0.01
        )
        
        top = result.get_top_variables(2)
        
        assert len(top) == 2
        assert top[0][0] == 'var2'  # 가장 큰 절대값
        assert abs(top[0][1]) == 0.8
    
    def test_forecast_result_high_vif_warnings(self):
        """ForecastResult.has_high_vif_warnings 테스트"""
        result = ForecastResult(
            horizon='Long',
            selected_variables=['var1', 'var2'],
            coefficients={'var1': 0.5, 'var2': -0.3},
            r_squared=0.64,
            n_observations=200,
            lambda_optimal=0.01,
            vif_scores={'var1': 5.0, 'var2': 15.0}  # var2 > 10
        )
        
        warnings = result.has_high_vif_warnings(10.0)
        
        assert 'var2' in warnings
        assert 'var1' not in warnings
    
    def test_lasso_diagnostics_is_all_converged(self):
        """LASSODiagnostics.is_all_converged 테스트"""
        # 모두 수렴
        diag1 = LASSODiagnostics(
            total_candidate_vars=50,
            excluded_vars=['d_US10Y'],
            high_vif_warnings=[],
            convergence_info={'VeryShort': True, 'Short': True, 'Long': True},
            computation_time=5.0
        )
        assert diag1.is_all_converged() is True
        
        # 일부 실패
        diag2 = LASSODiagnostics(
            total_candidate_vars=50,
            excluded_vars=['d_US10Y'],
            high_vif_warnings=[],
            convergence_info={'VeryShort': True, 'Short': False, 'Long': True},
            computation_time=5.0
        )
        assert diag2.is_all_converged() is False


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

