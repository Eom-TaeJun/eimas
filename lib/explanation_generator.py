#!/usr/bin/env python3
"""
Explanation Generator
=====================
SHAP 기반의 시장 예측 근거 생성 모듈.

이 모듈은 "왜?"에 대답하기 위해 별도의 머신러닝 모델(Random Forest)을 학습시키고,
XAI(Explainable AI) 기술을 사용하여 현재 시장 예측(또는 상태)의 주요 원인을 분석합니다.

주요 기능:
1. 시장 데이터 전처리 및 특성 공학 (Feature Engineering)
2. Proxy Model 학습 (SPY 익일 수익률 예측)
3. SHAP Local Explanation 생성 (현재 시점의 예측 근거)
4. 자연어 설명 생성 (LLM 프롬프트용)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from lib.xai_explainer import ModelExplainer, XAIExplanation

logger = logging.getLogger('eimas.explanation')

class MarketExplanationGenerator:
    """시장 상황에 대한 인과관계 설명 생성기"""

    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.explainer: Optional[ModelExplainer] = None
        self.feature_names: List[str] = []

    def prepare_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        시장 데이터를 통합하여 모델 학습용 데이터셋 생성
        
        Args:
            market_data: {ticker: DataFrame} 딕셔너리
        
        Returns:
            학습용 DataFrame (특성 + 타겟)
        """
        # 주요 자산 데이터 추출
        spy = market_data.get('SPY')
        vix = market_data.get('^VIX')
        
        if spy is None or spy.empty:
            logger.error("SPY data missing for explanation generation")
            return pd.DataFrame()

        # 기본 데이터 프레임 (SPY 기준)
        df = pd.DataFrame(index=spy.index)
        
        # 1. SPY 특성
        df['SPY_Close'] = spy['Close']
        df['SPY_Ret_1d'] = spy['Close'].pct_change()
        df['SPY_Ret_5d'] = spy['Close'].pct_change(5)
        df['SPY_Vol_20d'] = spy['Close'].pct_change().rolling(20).std()
        
        # 2. VIX 특성 (공포 지수)
        if vix is not None:
            # 인덱스 정렬
            vix = vix.reindex(df.index).ffill()
            df['VIX_Close'] = vix['Close']
            df['VIX_Change'] = vix['Close'].diff()
        
        # 3. 매크로/섹터 ETF (데이터에 있는 경우)
        # 10년물 국채 금리 (TLT 역수로 대용하거나 별도 매크로 데이터 사용)
        # 여기서는 market_data에 TLT가 있다고 가정
        tlt = market_data.get('TLT')
        if tlt is not None:
            tlt = tlt.reindex(df.index).ffill()
            df['TLT_Ret_1d'] = tlt['Close'].pct_change() # 금리와 역의 관계
            
        # 달러 (UUP)
        uup = market_data.get('UUP')
        if uup is not None:
            uup = uup.reindex(df.index).ffill()
            df['USD_Ret_1d'] = uup['Close'].pct_change()

        # 유가 (USO)
        uso = market_data.get('USO')
        if uso is not None:
            uso = uso.reindex(df.index).ffill()
            df['Oil_Ret_1d'] = uso['Close'].pct_change()

        # 4. 기술적 지표
        # RSI
        delta = df['SPY_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 이동평균 괴리율
        df['MA50_Div'] = (df['SPY_Close'] / df['SPY_Close'].rolling(50).mean()) - 1
        
        # 5. 타겟 변수: 익일 SPY 수익률 (Shift -1)
        # 우리가 설명하고 싶은 것은 "현재의 시장 상황이 내일(미래)에 어떤 영향을 미칠 것으로 예측되는가?"
        # 또는 "오늘 왜 올랐는가?"를 분석하려면 타겟을 '당일 수익률'로 하고 동시대 변수(Ex-post)를 써야 함.
        # IB 리포트의 목적은 "전망"과 "현재 판단의 근거"이므로, 미래 예측 모델의 기여도를 보는 것이 적절함.
        df['Target_Next_Ret'] = df['SPY_Ret_1d'].shift(-1) * 100 # % 단위
        
        # 결측치 제거
        df = df.dropna()
        
        # 최근 N일 데이터만 사용
        if len(df) > self.lookback_days:
            df = df.tail(self.lookback_days)
            
        return df

    def generate_explanation(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        현재 시장 상황에 대한 예측과 그 근거(SHAP)를 생성
        """
        try:
            # 1. 데이터 준비
            df = self.prepare_data(market_data)
            if df.empty or len(df) < 30:
                return {"error": f"Insufficient data: {len(df)} rows (min 30)"}

            # 특성과 타겟 분리
            # 마지막 행은 타겟(미래)이 없으므로 학습에서는 제외, 설명(테스트)에서는 사용
            feature_cols = [c for c in df.columns if c != 'Target_Next_Ret']
            self.feature_names = feature_cols
            
            # 학습 데이터 (마지막 행 제외)
            X_train = df.iloc[:-1][feature_cols]
            y_train = df.iloc[:-1]['Target_Next_Ret']
            
            # 설명 대상 (마지막 행 = 현재 시점)
            X_current = df.iloc[[-1]][feature_cols]
            
            # 2. 모델 학습
            self.model.fit(X_train, y_train)
            
            # 3. Explainer 초기화
            self.explainer = ModelExplainer(self.model, X_train, model_type='tree')
            
            # 4. SHAP 값 계산 (Local Explanation)
            explanation = self.explainer.explain_local(X_current, top_n=5)
            
            if not explanation:
                return {"error": "Explanation generation failed"}

            # 5. 결과 포맷팅
            result = {
                "prediction": explanation.prediction, # 예상 등락률 (%)
                "base_value": explanation.base_value, # 평균적인 등락률
                "drivers": [],
                "narrative": ""
            }
            
            # 주요 요인 추출
            narrative_parts = []
            for feature, impact in explanation.top_drivers[:3]:
                direction = "상승" if impact > 0 else "하락"
                
                # 변수명 한글화 및 설명
                desc = self._get_feature_description(feature)
                
                result["drivers"].append({
                    "name": feature,
                    "description": desc,
                    "impact": impact,
                    "direction": direction
                })
                
                narrative_parts.append(f"**{desc}**의 영향({impact:+.2f}%)")
            
            pred_direction = "상승" if explanation.prediction > 0 else "하락"
            
            result["narrative"] = (
                f"모델은 향후 시장의 **{pred_direction}**({explanation.prediction:+.2f}%)을 예측합니다. "
                f"주요 원인은 {', '.join(narrative_parts)}입니다."
            )
            
            return result

        except Exception as e:
            logger.error(f"Error in generate_explanation: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _get_feature_description(self, feature_name: str) -> str:
        """변수명을 읽기 쉬운 설명으로 변환"""
        map_desc = {
            'SPY_Ret_1d': '최근 주가 모멘텀',
            'SPY_Ret_5d': '단기 추세',
            'SPY_Vol_20d': '시장 변동성(20일)',
            'VIX_Close': '공포 지수 레벨',
            'VIX_Change': '공포 지수 변화',
            'TLT_Ret_1d': '채권 가격 변화(금리)',
            'USD_Ret_1d': '달러 가치 변화',
            'Oil_Ret_1d': '유가 변화',
            'RSI': '상대적 강도(RSI)',
            'MA50_Div': '이동평균 이격도'
        }
        return map_desc.get(feature_name, feature_name)

# 테스트 코드
if __name__ == "__main__":
    import yfinance as yf
    
    print("Fetching test data...")
    tickers = ['SPY', '^VIX', 'TLT', 'UUP', 'USO']
    data = {}
    for t in tickers:
        data[t] = yf.download(t, period='1y', progress=False)
        
    generator = MarketExplanationGenerator()
    result = generator.generate_explanation(data)
    
    print("\nExplanation Result:")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Prediction: {result['prediction']:.2f}%")
        print(f"Narrative: {result['narrative']}")
        print("\nTop Drivers:")
        for d in result['drivers']:
            print(f"- {d['description']}: {d['impact']:+.4f}%")
