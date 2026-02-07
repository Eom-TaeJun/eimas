#!/usr/bin/env python3
"""
Forecast Agent - LASSO 기반 Fed 금리 예측
==========================================
멀티에이전트 시스템의 예측 에이전트.
LASSO 회귀를 사용하여 Fed 금리 기대 변화를 예측.

경제학적 의미:
- LASSO: 고차원 변수에서 핵심 예측 변수 선택
- Horizon 분리: 시장 효율성 가설에 따른 정보 반영 속도 차이
- Treasury 제외: Simultaneity bias 방지
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from core.schemas import (
    AgentRequest, 
    AgentResponse, 
    AgentOpinion, 
    AgentRole, 
    OpinionStrength,
    ForecastResult,
    LASSODiagnostics,
    HorizonConfig
)
from agents.base_agent import BaseAgent, AgentConfig

# LASSOForecaster import
from lib.lasso_model import (
    LASSOForecaster, 
    LASSOConfig, 
    LASSOResult,
    classify_horizon,
    get_horizon_mask
)

# 로거 설정
logger = logging.getLogger('eimas.forecast_agent')


class ForecastAgent(BaseAgent):
    """
    LASSO 기반 Fed 금리 예측 에이전트
    
    Fed 금리 기대 변화(d_Exp_Rate)를 예측하기 위한 에이전트.
    Horizon별로 별도의 LASSO 모델을 학습하고 변수 선택 수행.
    
    Horizons:
        - VeryShort: ≤30일 (거의 확정된 정보, R² ≈ 0)
        - Short: 31-90일 (신용시장/인플레이션 기대 중심)
        - Long: ≥180일 (광범위 거시변수, R² ≈ 0.64)
    
    Example:
        >>> agent = ForecastAgent()
        >>> request = AgentRequest(
        ...     task_id="forecast_001",
        ...     role=AgentRole.FORECAST,
        ...     instruction="Predict Fed rate expectations",
        ...     context={'market_data': df, 'days_to_meeting': days_series}
        ... )
        >>> response = await agent.execute(request)
    """
    
    # 핵심 예측 변수 그룹 (해석용)
    KEY_VARIABLE_GROUPS = {
        'credit': ['d_Baa_Yield', 'd_Spread_Baa', 'd_HighYield_Rate'],
        'dollar': ['Ret_Dollar_Idx', 'd_Dollar_Idx'],
        'inflation': ['d_Breakeven5Y', 'd_Breakeven10Y'],
        'risk': ['Ret_VIX', 'd_VIX'],
        'equity': ['Ret_SP500', 'Ret_NASDAQ']
    }
    
    # 방향 결정 임계값
    DIRECTION_THRESHOLD = 0.001  # 1bp
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        ForecastAgent 초기화
        
        Args:
            config: 에이전트 설정. None이면 기본값 사용.
        """
        if config is None:
            config = AgentConfig(
                name="ForecastAgent",
                role=AgentRole.FORECAST,
                timeout=180,
                verbose=True
            )
        super().__init__(config)
        
        # LASSO 설정
        self.lasso_config = LASSOConfig(
            n_splits=5,
            max_iter=10000,
            tol=1e-4,
            hac_lag=5
        )
        
        # Horizon 설정
        self.horizon_config = HorizonConfig()
        
        # 내부 상태
        self._last_results: List[ForecastResult] = []
        self._last_diagnostics: Optional[LASSODiagnostics] = None
        
        logger.info("ForecastAgent initialized with LASSO configuration")
    
    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """
        LASSO 분석 실행 및 예측 결과 반환
        
        Args:
            request: AgentRequest with context containing:
                - market_data: pd.DataFrame (일별 금융/거시 데이터)
                - days_to_meeting: pd.Series (선택, FOMC 회의까지 남은 일수)
                - current_date: str (선택, 'YYYY-MM-DD')
        
        Returns:
            Dict containing:
                - forecasts: List[ForecastResult]
                - diagnostics: LASSODiagnostics
                - interpretation: str
                - confidence: float
        """
        start_time = datetime.now()
        
        # 1. 데이터 준비
        market_data = request.context.get('market_data')
        if market_data is None:
            raise ValueError("market_data is required in request context")
        
        # DataFrame 변환 (dict of DataFrames인 경우)
        if isinstance(market_data, dict):
            market_data = self._prepare_features(market_data)
        
        # 종속변수 추출 또는 생성
        if 'd_Exp_Rate' in market_data.columns:
            y = market_data['d_Exp_Rate']
        else:
            # 폴백: 임의의 종속변수 생성 (실제 데이터가 없는 경우)
            logger.warning("d_Exp_Rate not found, using placeholder")
            y = pd.Series(np.random.randn(len(market_data)) * 0.01, index=market_data.index)
        
        # days_to_meeting 추출
        days_to_meeting = request.context.get('days_to_meeting')
        if days_to_meeting is None:
            # 폴백: 임의의 days_to_meeting 생성
            self.logger.warning("days_to_meeting not provided, using random values")
            days_to_meeting = pd.Series(
                np.random.randint(1, 400, len(market_data)),
                index=market_data.index
            )
        
        # 인덱스 정렬: days_to_meeting을 market_data 인덱스에 맞춤
        if not days_to_meeting.index.equals(market_data.index):
            self.logger.info(f"Aligning days_to_meeting index: {len(days_to_meeting)} -> {len(market_data)}")
            days_to_meeting = days_to_meeting.reindex(market_data.index).ffill().bfill()
        
        # X 변수 (종속변수와 days_to_meeting 제외)
        exclude_cols = ['d_Exp_Rate', 'days_to_meeting', 'Exp_Rate_Level']
        X = market_data.drop(columns=[c for c in exclude_cols if c in market_data.columns], errors='ignore')
        
        # 디버그 로그
        self.logger.info(f"Data prepared: {len(X)} observations, {X.shape[1]} features")
        self.logger.info(f"days_to_meeting range: {days_to_meeting.min():.0f} - {days_to_meeting.max():.0f}")
        self.logger.info(f"d_Exp_Rate: valid={y.notna().sum()}, NaN={y.isna().sum()}")
        
        # Horizon별 관측치 수 미리 확인
        from lib.lasso_model import get_horizon_mask
        for h in ['VeryShort', 'Short', 'Long']:
            mask = get_horizon_mask(days_to_meeting, h)
            self.logger.info(f"  {h}: {mask.sum()} observations")
        
        # 2. Horizon별 분석
        forecasts: List[ForecastResult] = []
        convergence_info: Dict[str, bool] = {}
        high_vif_all: List[str] = []
        
        for horizon in ['VeryShort', 'Short', 'Long']:
            try:
                result = await self._fit_horizon(X, y, days_to_meeting, horizon)
                
                # ForecastResult로 변환
                forecast = ForecastResult(
                    horizon=result.horizon,
                    selected_variables=result.selected_variables,
                    coefficients=result.coefficients,
                    r_squared=result.r_squared,
                    n_observations=result.n_observations,
                    lambda_optimal=result.lambda_optimal,
                    hac_std_errors={},  # 별도 계산 필요
                    vif_scores={},  # 별도 계산 필요
                    predicted_change=None,
                    confidence_interval=None
                )
                
                forecasts.append(forecast)
                convergence_info[horizon] = True
                
                self.logger.info(
                    f"Horizon {horizon}: R²={result.r_squared:.4f}, "
                    f"selected={result.n_selected}"
                )
                
            except Exception as e:
                self.logger.error(f"Horizon {horizon} failed: {e}")
                convergence_info[horizon] = False
                
                # 빈 결과 추가
                forecasts.append(ForecastResult(
                    horizon=horizon,
                    selected_variables=[],
                    coefficients={},
                    r_squared=0.0,
                    n_observations=0,
                    lambda_optimal=0.0
                ))
        
        # 3. 진단 정보 생성
        diagnostics = LASSODiagnostics(
            total_candidate_vars=X.shape[1],
            excluded_vars=self._get_excluded_vars(market_data),
            high_vif_warnings=high_vif_all,
            convergence_info=convergence_info,
            computation_time=(datetime.now() - start_time).total_seconds()
        )
        
        # 4. 해석 생성
        interpretation = self._interpret_coefficients(forecasts)
        
        # 5. 내부 상태 저장
        self._last_results = forecasts
        self._last_diagnostics = diagnostics
        
        # 6. 신뢰도 계산 (Long horizon R² 기반)
        long_result = forecasts[2] if len(forecasts) > 2 else forecasts[-1]
        confidence = min(long_result.r_squared, 0.95) if long_result.r_squared > 0 else 0.1
        
        return {
            'forecasts': [f.to_dict() for f in forecasts],
            'diagnostics': diagnostics.to_dict(),
            'interpretation': interpretation,
            'confidence': confidence,
            'reasoning': interpretation
        }
    
    async def _fit_horizon(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        days_to_meeting: pd.Series,
        horizon: str
    ) -> LASSOResult:
        """
        특정 horizon에 대해 LASSO 모델 학습
        
        Args:
            X: 설명변수 DataFrame
            y: 종속변수 Series
            days_to_meeting: FOMC까지 남은 일수
            horizon: 대상 horizon
            
        Returns:
            LASSOResult
        """
        # Horizon 마스크 생성
        mask = get_horizon_mask(days_to_meeting, horizon)
        
        X_h = X[mask].copy()
        y_h = y[mask].copy()
        
        if len(X_h) < 20:  # 최소 관측치 확인
            logger.warning(f"Insufficient observations for {horizon}: {len(X_h)}")
            return LASSOResult(
                horizon=horizon,
                lambda_optimal=0.0,
                selected_variables=[],
                coefficients={},
                r_squared=0.0,
                n_observations=len(X_h),
                n_selected=0
            )
        
        # LASSO 학습
        forecaster = LASSOForecaster(self.lasso_config)
        result = forecaster.fit(X_h, y_h, horizon)
        
        return result
    
    async def form_opinion(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """
        멀티에이전트 토론에서 특정 토픽에 대한 의견 형성
        
        Args:
            topic: 토론 주제 (rate_direction, rate_magnitude, 
                   forecast_confidence, key_drivers)
            context: 분석 결과 포함 컨텍스트
        
        Returns:
            AgentOpinion
        """
        # 저장된 결과 또는 context에서 가져오기
        forecasts = context.get('forecasts', [])
        if not forecasts and self._last_results:
            forecasts = self._last_results
        
        if not forecasts:
            return AgentOpinion(
                agent_role=self.config.role,
                topic=topic,
                position="HOLD",
                strength=OpinionStrength.NEUTRAL,
                confidence=0.0,
                evidence=["No forecast results available"],
                caveats=["Analysis not completed"]
            )
        
        # ForecastResult 객체로 변환 (dict인 경우)
        if isinstance(forecasts[0], dict):
            forecasts = [self._dict_to_forecast_result(f) for f in forecasts]
        
        # Long horizon 결과 (주요 분석 대상)
        long_result = forecasts[2] if len(forecasts) > 2 else forecasts[-1]
        
        if topic == "rate_direction":
            return self._form_rate_direction_opinion(long_result, context)
        
        elif topic == "rate_magnitude":
            return self._form_rate_magnitude_opinion(long_result, context)
        
        elif topic == "forecast_confidence":
            return self._form_confidence_opinion(long_result, context)
        
        elif topic == "key_drivers":
            return self._form_key_drivers_opinion(long_result, context)
        
        else:
            return AgentOpinion(
                agent_role=self.config.role,
                topic=topic,
                position=f"Unknown topic: {topic}",
                strength=OpinionStrength.NEUTRAL,
                confidence=0.5,
                evidence=[]
            )
    
    def _form_rate_direction_opinion(
        self, 
        result: ForecastResult, 
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """rate_direction 토픽에 대한 의견 형성"""
        
        # 핵심 변수 계수 분석
        key_vars = ['d_Spread_Baa', 'Ret_Dollar_Idx', 'd_Breakeven5Y', 'd_VIX']
        signals: Dict[str, float] = {}
        
        for var in key_vars:
            if var in result.coefficients:
                signals[var] = result.coefficients[var]
        
        # 가중 신호 계산
        # d_Spread_Baa 음(-): 스프레드 확대 → 인하 기대 감소 (UP)
        # Ret_Dollar_Idx 양(+): 달러 강세 → 인하 기대 감소 (UP)
        weighted_signal = 0.0
        
        if 'd_Spread_Baa' in signals:
            weighted_signal -= signals['d_Spread_Baa']  # 음의 계수가 UP
        if 'Ret_Dollar_Idx' in signals:
            weighted_signal += signals['Ret_Dollar_Idx']  # 양의 계수가 UP
        if 'd_Breakeven5Y' in signals:
            weighted_signal += signals['d_Breakeven5Y'] * 0.5
        
        # 방향 결정
        if weighted_signal > self.DIRECTION_THRESHOLD:
            position = "UP"
            strength = OpinionStrength.AGREE if weighted_signal > 0.01 else OpinionStrength.NEUTRAL
        elif weighted_signal < -self.DIRECTION_THRESHOLD:
            position = "DOWN"
            strength = OpinionStrength.DISAGREE if weighted_signal < -0.01 else OpinionStrength.NEUTRAL
        else:
            position = "HOLD"
            strength = OpinionStrength.NEUTRAL
        
        # 신뢰도 (R² 기반)
        confidence = min(result.r_squared, 0.95)
        
        # 근거
        evidence = [
            f"Long horizon R²: {result.r_squared:.4f}",
            f"Selected variables: {len(result.selected_variables)}",
        ]
        for var, coef in signals.items():
            evidence.append(f"{var}: {coef:+.4f}")
        
        # 주의사항
        caveats = []
        if result.r_squared < 0.3:
            caveats.append("Low model explanatory power")
        if len(result.selected_variables) < 5:
            caveats.append("Few variables selected, limited signal")
        
        return AgentOpinion(
            agent_role=self.config.role,
            topic="rate_direction",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            caveats=caveats
        )
    
    def _form_rate_magnitude_opinion(
        self, 
        result: ForecastResult, 
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """rate_magnitude 토픽에 대한 의견 형성"""
        
        # 예측된 변화 (bp 단위)
        predicted_change = result.predicted_change or 0.0
        abs_change = abs(predicted_change)
        
        if abs_change > 50:
            position = f"Large change expected: {predicted_change:+.1f}bp"
            strength = OpinionStrength.STRONG_AGREE
        elif abs_change > 25:
            position = f"Moderate change expected: {predicted_change:+.1f}bp"
            strength = OpinionStrength.AGREE
        else:
            position = f"Minimal change expected: {predicted_change:+.1f}bp"
            strength = OpinionStrength.NEUTRAL
        
        return AgentOpinion(
            agent_role=self.config.role,
            topic="rate_magnitude",
            position=position,
            strength=strength,
            confidence=min(result.r_squared, 0.95),
            evidence=[
                f"Predicted change: {predicted_change:+.2f}bp",
                f"Model R²: {result.r_squared:.4f}"
            ],
            caveats=["Point estimate only, confidence interval not computed"]
        )
    
    def _form_confidence_opinion(
        self, 
        result: ForecastResult, 
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """forecast_confidence 토픽에 대한 의견 형성"""
        
        r_squared = result.r_squared
        
        if r_squared > 0.6:
            position = "High confidence"
            strength = OpinionStrength.STRONG_AGREE
        elif r_squared > 0.4:
            position = "Moderate confidence"
            strength = OpinionStrength.AGREE
        elif r_squared > 0.2:
            position = "Low confidence"
            strength = OpinionStrength.DISAGREE
        else:
            position = "Very low confidence"
            strength = OpinionStrength.STRONG_DISAGREE
        
        return AgentOpinion(
            agent_role=self.config.role,
            topic="forecast_confidence",
            position=position,
            strength=strength,
            confidence=r_squared,
            evidence=[
                f"R²: {r_squared:.4f}",
                f"N observations: {result.n_observations}",
                f"Variables selected: {len(result.selected_variables)}"
            ]
        )
    
    def _form_key_drivers_opinion(
        self, 
        result: ForecastResult, 
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """key_drivers 토픽에 대한 의견 형성"""
        
        # 상위 5개 변수 추출
        top_vars = result.get_top_variables(5) if hasattr(result, 'get_top_variables') else []
        if not top_vars and result.coefficients:
            sorted_coefs = sorted(
                result.coefficients.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            top_vars = sorted_coefs
        
        if not top_vars:
            return AgentOpinion(
                agent_role=self.config.role,
                topic="key_drivers",
                position="No key drivers identified",
                strength=OpinionStrength.NEUTRAL,
                confidence=0.0,
                evidence=[]
            )
        
        # 변수 그룹 분류
        driver_groups = self._classify_drivers(top_vars)
        
        position = f"Top drivers: {', '.join([v[0] for v in top_vars[:3]])}"
        
        evidence = [f"{var}: {coef:+.4f}" for var, coef in top_vars]
        evidence.append(f"Dominant groups: {', '.join(driver_groups)}")
        
        return AgentOpinion(
            agent_role=self.config.role,
            topic="key_drivers",
            position=position,
            strength=OpinionStrength.AGREE,
            confidence=min(result.r_squared, 0.95),
            evidence=evidence
        )
    
    def _classify_drivers(self, top_vars: List[Tuple[str, float]]) -> List[str]:
        """변수들을 그룹으로 분류"""
        groups = set()
        for var, _ in top_vars:
            for group, vars_list in self.KEY_VARIABLE_GROUPS.items():
                if any(v in var for v in vars_list):
                    groups.add(group)
        return list(groups)
    
    def _prepare_features(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Raw market_data를 분석용 단일 DataFrame으로 변환
        
        Args:
            market_data: {ticker: DataFrame} 형식의 시장 데이터
            
        Returns:
            통합된 DataFrame
        """
        combined_df = pd.DataFrame()
        
        for ticker, df in market_data.items():
            if df.empty:
                continue
            
            # 종가 추출
            if 'Close' in df.columns:
                series = df['Close']
            elif 'Adj Close' in df.columns:
                series = df['Adj Close']
            else:
                series = df.iloc[:, 0]
            
            series.name = ticker
            
            # 변수 타입에 따른 변환
            is_rate = any(x in ticker.upper() for x in ['YIELD', 'RATE', 'VIX', 'SPREAD'])
            
            if is_rate:
                # 차분 (금리, 변동성)
                diff_series = series.diff()
                diff_series.name = f"d_{ticker}"
                combined_df = combined_df.join(diff_series, how='outer')
            else:
                # 로그 수익률 (주식, 암호화폐, 원자재)
                ret_series = np.log(series / series.shift(1))
                ret_series.name = f"Ret_{ticker}"
                combined_df = combined_df.join(ret_series, how='outer')
        
        # 결측치 처리
        combined_df = combined_df.dropna(how='all')
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        return combined_df
    
    def _classify_horizon(self, days_to_meeting: int) -> Optional[str]:
        """FOMC 회의까지 남은 일수로 horizon 분류"""
        return self.horizon_config.classify(days_to_meeting)
    
    def _interpret_coefficients(self, results: List[ForecastResult]) -> str:
        """LASSO 결과에 대한 자연어 해석 생성"""
        
        interpretations = []
        
        for result in results:
            if not result.selected_variables:
                interpretations.append(
                    f"{result.horizon}: 유의미한 예측 변수 없음 (R²={result.r_squared:.4f})"
                )
                continue
            
            top_vars = sorted(
                result.coefficients.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            var_str = ", ".join([f"{v}({c:+.3f})" for v, c in top_vars])
            interpretations.append(
                f"{result.horizon}: R²={result.r_squared:.4f}, "
                f"주요 변수: {var_str}"
            )
        
        summary = "\n".join(interpretations)
        
        # Long horizon 특별 해석
        if len(results) >= 3 and results[2].r_squared > 0.3:
            long_result = results[2]
            summary += f"\n\n[분석] Long horizon에서 설명력 {long_result.r_squared:.1%} 달성. "
            
            # 신용시장 영향
            credit_vars = [v for v in long_result.selected_variables 
                          if any(c in v for c in ['Baa', 'Spread', 'HighYield'])]
            if credit_vars:
                summary += f"신용시장 변수({', '.join(credit_vars[:2])})가 핵심 동인."
        
        return summary
    
    def _get_excluded_vars(self, df: pd.DataFrame) -> List[str]:
        """제외된 Treasury 변수 목록"""
        from lib.lasso_model import DEFAULT_EXCLUDED_VARS
        return [col for col in df.columns 
                if any(excl in col for excl in DEFAULT_EXCLUDED_VARS)]
    
    def _dict_to_forecast_result(self, d: Dict) -> ForecastResult:
        """딕셔너리를 ForecastResult로 변환"""
        return ForecastResult(
            horizon=d.get('horizon', 'Unknown'),
            selected_variables=d.get('selected_variables', []),
            coefficients=d.get('coefficients', {}),
            r_squared=d.get('r_squared', 0.0),
            n_observations=d.get('n_observations', 0),
            lambda_optimal=d.get('lambda_optimal', 0.0),
            hac_std_errors=d.get('hac_std_errors', {}),
            vif_scores=d.get('vif_scores', {}),
            predicted_change=d.get('predicted_change'),
            confidence_interval=d.get('confidence_interval')
        )


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        """ForecastAgent 테스트"""
        print("=" * 60)
        print("ForecastAgent Test")
        print("=" * 60)
        
        # 에이전트 초기화
        agent = ForecastAgent()
        print(f"\n1. Initialized: {agent}")
        
        # Mock 데이터 생성
        np.random.seed(42)
        n_samples = 300
        dates = pd.date_range(start='2023-01-01', periods=n_samples)
        
        # 시장 데이터
        market_data = pd.DataFrame({
            'd_Baa_Yield': np.random.randn(n_samples) * 0.1,
            'd_Spread_Baa': np.random.randn(n_samples) * 0.05,
            'Ret_Dollar_Idx': np.random.randn(n_samples) * 0.01,
            'd_Breakeven5Y': np.random.randn(n_samples) * 0.02,
            'Ret_VIX': np.random.randn(n_samples) * 0.05,
            'Ret_SP500': np.random.randn(n_samples) * 0.01,
            'd_Exp_Rate': 0.5 * np.random.randn(n_samples) * 0.1  # 타겟
        }, index=dates)
        
        # days_to_meeting
        days_to_meeting = pd.Series(
            np.random.randint(1, 400, n_samples),
            index=dates
        )
        
        # 요청 생성
        request = AgentRequest(
            task_id="test_forecast_001",
            role=AgentRole.FORECAST,
            instruction="Predict Fed rate expectations",
            context={
                'market_data': market_data,
                'days_to_meeting': days_to_meeting
            }
        )
        
        # 실행
        print("\n2. Testing _execute...")
        try:
            result = await agent._execute(request)
            print(f"   Status: Success")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Forecasts: {len(result['forecasts'])} horizons")
            print(f"   Interpretation:\n   {result['interpretation'][:200]}...")
        except Exception as e:
            print(f"   Failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 의견 형성 테스트
        print("\n3. Testing form_opinion...")
        topics = ['rate_direction', 'rate_magnitude', 'forecast_confidence', 'key_drivers']
        
        for topic in topics:
            try:
                opinion = await agent.form_opinion(topic, {'forecasts': agent._last_results})
                print(f"   {topic}: {opinion.position} (conf: {opinion.confidence:.2f})")
            except Exception as e:
                print(f"   {topic}: Failed - {e}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
    
    asyncio.run(test())
