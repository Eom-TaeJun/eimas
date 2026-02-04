#!/usr/bin/env python3
"""
Critical Path - Regime Detector
================================

시장 레짐 탐지 및 레짐별 임계값 제공 모듈

Economic Foundation:
    Maheu & McCurdy: Markov Switching Models
    Hamilton (1989): "A New Approach to the Economic Analysis of Nonstationary Time Series"

    핵심 개념:
    - Bull/Bear/Transition/Crisis 4가지 레짐
    - 레짐별 동적 임계값 (위기 시 더 엄격한 기준)
    - MA(20/50/200) 기반 추세 판단
    - 레짐 전환 확률 추정

Classes:
    - EnhancedRegimeDetector: 레짐 탐지 및 임계값 제공

Returns:
    RegimeResult: 현재 레짐, 확신도, 전환 확률, 동적 임계값
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime

# Import schemas from same package
from .schemas import RegimeResult, normalize_to_score


class EnhancedRegimeDetector:
    """
    레짐 탐지 및 레짐별 임계값 제공
    
    Maheu & McCurdy 연구에 기반하여 Bull/Bear/Transition 레짐을 탐지하고,
    각 레짐별로 다른 임계값 세트를 제공합니다. 레짐 전환 감지가 핵심입니다.
    
    경제학적 배경:
    - Bull과 Bear 시장은 수익률 분포 자체가 다름 (Maheu & McCurdy)
    - Bull: 낮은 변동성, 양의 평균, 정규분포에 가까움
    - Bear: 높은 변동성, 음의 평균, fat tail
    - 같은 -3% 하락도 Bull에서는 2σ 이벤트, Bear에서는 1σ 이벤트
    - 레짐 전환 초기에 신호가 가장 가치 있음
    """
    
    # 레짐별 임계값 정의
    REGIME_THRESHOLDS = {
        'BULL': {
            'volume_spike': 2.5,      # 거래량 급증 기준 (평균 대비 배수)
            'ma_deviation': -2.5,     # MA 이탈 기준 (%)
            'zscore_alert': 2.5,      # Z-score 경고 기준
            'vix_warning': 22,        # VIX 경고 레벨
            'return_alert': -2.0,     # 일간 수익률 경고 (%)
        },
        'TRANSITION': {
            'volume_spike': 2.0,
            'ma_deviation': -2.0,
            'zscore_alert': 2.0,
            'vix_warning': 25,
            'return_alert': -1.5,
        },
        'BEAR': {
            'volume_spike': 1.8,
            'ma_deviation': -1.5,
            'zscore_alert': 1.5,
            'vix_warning': 30,
            'return_alert': -1.0,
        },
        'CRISIS': {
            'volume_spike': 1.5,
            'ma_deviation': -1.0,
            'zscore_alert': 1.0,
            'vix_warning': 35,
            'return_alert': -0.5,
        }
    }
    
    def __init__(self, short_ma: int = 20, long_ma: int = 120, crisis_vix: float = 30):
        """
        Args:
            short_ma: 단기 이동평균 기간 (기본값 20일)
            long_ma: 장기 이동평균 기간 (기본값 120일)
            crisis_vix: 위기 판단 VIX 임계값 (기본값 30)
        """
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.crisis_vix = crisis_vix
        
        # 레짐 히스토리 저장 (최근 20일)
        self.regime_history: List[str] = []
        self.max_history = 20
    
    def detect_regime(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
        """
        현재 레짐 판단
        
        로직:
        1. CRISIS 체크 (우선순위 최고)
        2. BULL 조건
        3. BEAR 조건
        4. TRANSITION
        
        Returns:
            str: 레짐 이름
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return "TRANSITION"
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_short = float(ma_short.iloc[-1])
        current_ma_long = float(ma_long.iloc[-1])
        
        # 1. CRISIS 체크 (우선순위 최고)
        vix_value = None
        vix_empty = hasattr(vix_data, 'empty') and vix_data.empty if vix_data is not None else True
        if vix_data is not None and not vix_empty and 'Close' in vix_data.columns:
            vix_value = float(vix_data['Close'].iloc[-1])
        
        if len(close) >= 5:
            return_5d = (current_price / close.iloc[-5] - 1) * 100
        else:
            return_5d = 0.0
        
        if (vix_value is not None and vix_value >= self.crisis_vix) or return_5d < -5.0:
            return "CRISIS"
        
        # 2. BULL 조건
        if not pd.isna(current_ma_long):
            price_above_long = current_price > current_ma_long
            ma_short_above_long = current_ma_short > current_ma_long if not pd.isna(current_ma_short) else False
            
            if price_above_long and ma_short_above_long:
                return "BULL"
        
        # 3. BEAR 조건
        if not pd.isna(current_ma_long):
            price_below_long = current_price < current_ma_long
            ma_short_below_long = current_ma_short < current_ma_long if not pd.isna(current_ma_short) else False
            
            if price_below_long and ma_short_below_long:
                return "BEAR"
        
        # 4. TRANSITION (그 외 모든 경우)
        return "TRANSITION"
    
    def calculate_regime_confidence(self, spy_data: pd.DataFrame) -> float:
        """
        레짐 확신도 계산
        
        Returns:
            float: 0-100 사이 확신도
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return 50.0
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_short = float(ma_short.iloc[-1])
        current_ma_long = float(ma_long.iloc[-1])
        
        if pd.isna(current_ma_long) or current_ma_long == 0:
            return 50.0
        
        # 1. 현재가와 120일 MA의 거리 (0-50점)
        price_distance = abs((current_price / current_ma_long - 1) * 100)
        price_score = min(50.0, normalize_to_score(price_distance, min_val=0.0, max_val=5.0) * 0.5)
        
        # 2. 20일 MA와 120일 MA의 거리 (0-30점)
        if not pd.isna(current_ma_short) and current_ma_long != 0:
            ma_distance = abs((current_ma_short / current_ma_long - 1) * 100)
            ma_score = min(30.0, normalize_to_score(ma_distance, min_val=0.0, max_val=3.0) * 0.3)
        else:
            ma_score = 15.0
        
        # 3. 최근 N일간 레짐 일관성 (0-20점)
        if len(self.regime_history) >= 5:
            from collections import Counter
            recent_regimes = self.regime_history[-5:]
            counter = Counter(recent_regimes)
            most_common_count = counter.most_common(1)[0][1] if counter else 0
            consistency_ratio = most_common_count / len(recent_regimes)
            consistency_score = consistency_ratio * 20.0
        else:
            consistency_score = 10.0
        
        total_confidence = price_score + ma_score + consistency_score
        return min(100.0, max(0.0, total_confidence))
    
    def calculate_transition_probability(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> Tuple[float, str]:
        """
        레짐 전환 확률 계산
        
        Returns:
            Tuple[확률, 방향]
        """
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return 0.0, "STABLE"
        
        close = spy_data['Close']
        ma_short = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_long = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        signals = []
        
        # 1. MA 근접도 체크
        if len(ma_short) > 0 and len(ma_long) > 0:
            current_ma_short = float(ma_short.iloc[-1])
            current_ma_long = float(ma_long.iloc[-1])
            
            if not pd.isna(current_ma_short) and not pd.isna(current_ma_long) and current_ma_long != 0:
                ma_distance_pct = abs((current_ma_short / current_ma_long - 1) * 100)
                if ma_distance_pct < 3.0:
                    signals.append(('ma_proximity', 30.0))
        
        # 2. MA 기울기 변화 체크
        if len(ma_short) >= 10:
            recent_slope = (float(ma_short.iloc[-1]) / float(ma_short.iloc[-5]) - 1) * 100 if len(ma_short) >= 5 else 0
            if len(ma_short) >= 10:
                prev_slope = (float(ma_short.iloc[-5]) / float(ma_short.iloc[-10]) - 1) * 100
                if (recent_slope > 0 and prev_slope < 0) or (recent_slope < 0 and prev_slope > 0):
                    signals.append(('ma_slope_change', 25.0))
        
        # 3. 거래량 증가 + 가격 역방향
        if 'Volume' in spy_data.columns and len(spy_data) >= 20:
            volume = spy_data['Volume']
            volume_ma = volume.rolling(window=20, min_periods=1).mean()
            
            if len(volume) > 0 and len(volume_ma) > 0:
                current_volume = float(volume.iloc[-1])
                avg_volume = float(volume_ma.iloc[-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                if len(close) >= 3:
                    return_3d = (float(close.iloc[-1]) / float(close.iloc[-3]) - 1) * 100
                    
                    if volume_ratio > 1.3 and return_3d < -1.0:
                        signals.append(('volume_price_divergence', 20.0))
        
        # 4. VIX 추세 체크
        vix_empty = hasattr(vix_data, 'empty') and vix_data.empty if vix_data is not None else True
        if vix_data is not None and not vix_empty and 'Close' in vix_data.columns:
            vix_close = vix_data['Close']
            if len(vix_close) >= 5:
                vix_trend = []
                for i in range(len(vix_close) - 4, len(vix_close)):
                    if i > 0:
                        change = (float(vix_close.iloc[i]) / float(vix_close.iloc[i-1]) - 1) * 100
                        vix_trend.append(change)
                
                if len(vix_trend) == 4:
                    all_positive = all(x > 0 for x in vix_trend)
                    all_negative = all(x < 0 for x in vix_trend)
                    if all_positive or all_negative:
                        signals.append(('vix_trend', 25.0))
        
        total_probability = min(100.0, sum(prob for _, prob in signals))
        
        current_regime = self.detect_regime(spy_data, vix_data)
        
        if total_probability < 30.0:
            direction = "STABLE"
        elif current_regime == "BULL":
            direction = "BULL_TO_BEAR"
        elif current_regime == "BEAR":
            direction = "BEAR_TO_BULL"
        else:
            direction = "UNCERTAIN"
        
        return total_probability, direction
    
    def get_thresholds_for_regime(self, regime: str) -> Dict:
        """레짐에 맞는 임계값 세트 반환"""
        return self.REGIME_THRESHOLDS.get(regime, self.REGIME_THRESHOLDS['TRANSITION'])
    
    def get_ma_status(self, spy_data: pd.DataFrame) -> Dict:
        """이동평균 상태 정보 반환"""
        if spy_data is None or (hasattr(spy_data, 'empty') and spy_data.empty) or (spy_data is not None and 'Close' not in spy_data.columns):
            return {}
        
        close = spy_data['Close']
        ma_5 = close.rolling(window=5, min_periods=1).mean()
        ma_20 = close.rolling(window=self.short_ma, min_periods=1).mean()
        ma_120 = close.rolling(window=self.long_ma, min_periods=1).mean()
        
        current_price = float(close.iloc[-1])
        current_ma_5 = float(ma_5.iloc[-1]) if not ma_5.empty else None
        current_ma_20 = float(ma_20.iloc[-1]) if not ma_20.empty else None
        current_ma_120 = float(ma_120.iloc[-1]) if not ma_120.empty else None
        
        price_vs_ma20 = ((current_price / current_ma_20 - 1) * 100) if current_ma_20 and current_ma_20 != 0 else None
        price_vs_ma120 = ((current_price / current_ma_120 - 1) * 100) if current_ma_120 and current_ma_120 != 0 else None
        ma20_vs_ma120 = ((current_ma_20 / current_ma_120 - 1) * 100) if current_ma_20 and current_ma_120 and current_ma_120 != 0 else None
        
        ma20_slope = None
        if len(ma_20) >= 5:
            ma20_slope = ((float(ma_20.iloc[-1]) / float(ma_20.iloc[-5]) - 1) * 100) if len(ma_20) >= 5 else None
        
        ma120_slope = None
        if len(ma_120) >= 20:
            ma120_slope = ((float(ma_120.iloc[-1]) / float(ma_120.iloc[-20]) - 1) * 100) if len(ma_120) >= 20 else None
        
        return {
            'ma_5': current_ma_5,
            'ma_20': current_ma_20,
            'ma_120': current_ma_120,
            'price_vs_ma20': price_vs_ma20,
            'price_vs_ma120': price_vs_ma120,
            'ma20_vs_ma120': ma20_vs_ma120,
            'ma20_slope': ma20_slope,
            'ma120_slope': ma120_slope,
        }
    
    def _apply_regime_buffer(self, new_regime: str) -> str:
        """레짐 전환 버퍼 적용 (급격한 스위칭 방지)"""
        if len(self.regime_history) == 0:
            return new_regime
        
        last_regime = self.regime_history[-1]
        
        if new_regime == last_regime:
            return new_regime
        
        if new_regime == "CRISIS":
            return new_regime
        
        if len(self.regime_history) >= 2:
            recent_regimes = self.regime_history[-2:]
            if new_regime in recent_regimes:
                return new_regime
        
        return last_regime
    
    def generate_interpretation(self, regime: str, confidence: float, transition_prob: float, transition_dir: str) -> str:
        """해석 텍스트 생성"""
        regime_names = {
            "BULL": "강세장",
            "BEAR": "약세장",
            "TRANSITION": "전환기",
            "CRISIS": "위기"
        }
        
        regime_name = regime_names.get(regime, regime)
        base_text = f"현재 시장은 {regime_name} 국면입니다. "
        
        if confidence >= 70:
            base_text += f"레짐 판단 확신도가 높습니다 ({confidence:.1f}%). "
        elif confidence >= 50:
            base_text += f"레짐 판단 확신도가 보통입니다 ({confidence:.1f}%). "
        else:
            base_text += f"레짐 판단 확신도가 낮습니다 ({confidence:.1f}%). "
        
        if transition_prob >= 70:
            base_text += f"⚠️ 레짐 전환 가능성이 높습니다 ({transition_prob:.1f}%). "
            if transition_dir == "BULL_TO_BEAR":
                base_text += "강세장에서 약세장으로 전환될 가능성이 있습니다."
            elif transition_dir == "BEAR_TO_BULL":
                base_text += "약세장에서 강세장으로 전환될 가능성이 있습니다."
            else:
                base_text += f"전환 방향: {transition_dir}"
        elif transition_prob >= 50:
            base_text += f"레짐 전환 가능성이 있습니다 ({transition_prob:.1f}%). "
        else:
            base_text += f"현재 레짐이 안정적입니다 (전환 확률 {transition_prob:.1f}%). "
        
        return base_text
    
    def analyze(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> RegimeResult:
        """전체 분석 실행"""
        detected_regime = self.detect_regime(spy_data, vix_data)
        final_regime = self._apply_regime_buffer(detected_regime)
        
        self.regime_history.append(final_regime)
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
        
        confidence = self.calculate_regime_confidence(spy_data)
        transition_prob, transition_dir = self.calculate_transition_probability(spy_data, vix_data)
        thresholds = self.get_thresholds_for_regime(final_regime)
        ma_status = self.get_ma_status(spy_data)
        interpretation = self.generate_interpretation(final_regime, confidence, transition_prob, transition_dir)
        
        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            current_regime=final_regime,
            regime_confidence=confidence,
            transition_probability=transition_prob,
            transition_direction=transition_dir,
            thresholds=thresholds,
            ma_status=ma_status,
            interpretation=interpretation
        )


