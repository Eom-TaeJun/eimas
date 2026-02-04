#!/usr/bin/env python3
"""
Critical Path - Spillover Network
==================================

자산간 충격 전이(spillover) 네트워크 분석 모듈

Economic Foundation:
    Boeckelmann: Spillover Networks in Financial Markets
    Diebold & Yilmaz (2012): "Better to Give than to Receive"

    핵심 개념:
    - 자산간 충격 전이를 그래프 구조로 모델링
    - 경로별 전이 신호 탐지 (유동성, 변동성, 신용, 집중도, 로테이션)
    - 레짐별 시차 조정 (위기 시 전이 속도 빨라짐)
    - 위험 진원지 식별 (가장 많은 경로의 소스)

Classes:
    - SpilloverNetwork: 충격 전이 네트워크 분석

Returns:
    SpilloverResult: 활성 경로, 리스크 점수, 주요 위험 진원지
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import schemas from same package
from .schemas import SpilloverEdge, SpilloverResult, normalize_to_score


class SpilloverNetwork:
    """
    자산간 충격 전이(spillover) 네트워크
    
    Boeckelmann 연구에 기반하여 자산간 충격 전이를 그래프 구조로 모델링하고,
    경로별 전이 신호를 탐지합니다.
    
    경제학적 배경:
    - 자산간 spillover 강도와 시차가 시간에 따라 변함
    - 위기 시: spillover 강도 증가, 시차 단축 (빠른 전이)
    - 평시: spillover 약함, 시차 김 (느린 전이)
    - 금융 경로(유동성) vs 실물 경로(공급망)는 시차가 다름
    
    노드: 자산/자산군
    엣지: 경제학적 인과관계
    """
    
    # 경로 정의 (경제학 이론 기반)
    SPILLOVER_PATHS = [
        # === 유동성/금리 경로 ===
        {
            'source': 'TLT',
            'target': 'QQQ', 
            'edge_type': 'POSITIVE',      # TLT↓(금리↑) → QQQ↓
            'base_lag': 3,
            'category': 'liquidity',
            'theory': '금리 상승 시 성장주 할인율 증가로 밸류에이션 압박'
        },
        {
            'source': 'DXY',
            'target': 'GLD',
            'edge_type': 'NEGATIVE',      # DXY↑ → GLD↓
            'base_lag': 1,
            'category': 'liquidity',
            'theory': '달러 강세 시 달러 표시 금 가격 하락 압력'
        },
        {
            'source': 'DXY',
            'target': 'EEM',
            'edge_type': 'NEGATIVE',      # DXY↑ → EEM↓
            'base_lag': 3,
            'category': 'liquidity',
            'theory': '달러 강세 시 신흥국 자금 유출, 달러 부채 부담 증가'
        },
        
        # === 변동성/공포 경로 ===
        {
            'source': '^VIX',
            'target': 'SPY',
            'edge_type': 'NEGATIVE',      # VIX↑ → SPY↓
            'base_lag': 1,
            'category': 'volatility',
            'theory': 'VIX 급등 시 옵션 딜러 감마 헷지 매도, 하락 가속'
        },
        {
            'source': 'VIX',
            'target': 'SPY',
            'edge_type': 'NEGATIVE',      # VIX↑ → SPY↓ (대체 티커)
            'base_lag': 1,
            'category': 'volatility',
            'theory': 'VIX 급등 시 옵션 딜러 감마 헷지 매도, 하락 가속'
        },
        
        # === 신용 경로 ===
        {
            'source': 'HYG',
            'target': 'XLF',
            'edge_type': 'POSITIVE',      # HYG↓ → XLF↓
            'base_lag': 3,
            'category': 'credit',
            'theory': '하이일드 스프레드 확대 시 금융섹터 신용 우려'
        },
        {
            'source': 'HYG',
            'target': 'IWM',
            'edge_type': 'POSITIVE',      # HYG↓ → IWM↓
            'base_lag': 5,
            'category': 'credit',
            'theory': '신용 경색 시 소형주 자금조달 어려움'
        },
        
        # === 빅테크/집중도 경로 ===
        {
            'source': 'QQQ',
            'target': 'SPY',
            'edge_type': 'POSITIVE',      # QQQ↓ → SPY↓
            'base_lag': 1,
            'category': 'concentration',
            'theory': 'MAG7이 SPY의 30% 차지, 빅테크 하락이 지수 끌어내림'
        },
        {
            'source': 'NVDA',
            'target': 'SMH',
            'edge_type': 'POSITIVE',      # NVDA↓ → SMH↓
            'base_lag': 1,
            'category': 'concentration',
            'theory': 'AI 대장주 균열이 반도체 섹터 전체로 전이'
        },
        
        # === 섹터 로테이션 경로 ===
        {
            'source': 'XLY',
            'target': 'SPY',
            'edge_type': 'POSITIVE',      # XLY↓ → SPY↓ (with lag)
            'base_lag': 5,
            'category': 'rotation',
            'theory': '경기민감 섹터 약세가 전체 시장 약세로 확산'
        },
    ]
    
    # 레짐별 시차 조정 계수
    LAG_ADJUSTMENTS = {
        'BULL': 1.0,        # 기본 시차 유지
        'TRANSITION': 0.8,  # 20% 단축
        'BEAR': 0.7,        # 30% 단축
        'CRISIS': 0.5,      # 50% 단축 (빠른 전이)
    }
    
    # 활성화 임계값 (레짐별) - 완화된 임계값
    ACTIVATION_THRESHOLDS = {
        'BULL': {
            'min_move': 1.2,       # 최소 1.2% 움직임 (기존 2.0에서 완화)
            'volume_ratio': 1.3,   # 거래량 1.3배 (기존 1.5에서 완화)
        },
        'BEAR': {
            'min_move': 1.0,       # 최소 1.0% 움직임 (기존 1.5에서 완화)
            'volume_ratio': 1.2,   # 거래량 1.2배 (기존 1.3에서 완화)
        },
        'TRANSITION': {
            'min_move': 1.0,       # 최소 1.0% 움직임 (기존 1.8에서 완화)
            'volume_ratio': 1.2,   # 거래량 1.2배 (기존 1.4에서 완화)
        },
        'CRISIS': {
            'min_move': 0.8,       # 최소 0.8% 움직임 (기존 1.0에서 완화)
            'volume_ratio': 1.1,   # 거래량 1.1배 (기존 1.2에서 완화)
        }
    }
    
    def __init__(self, lookback: int = 20):
        """
        Args:
            lookback: 롤링 윈도우 기간 (기본값 20일)
        """
        self.lookback = lookback
        self.paths = self.SPILLOVER_PATHS
    
    def adjust_lag_for_regime(self, base_lag: int, regime: str) -> int:
        """
        레짐에 따라 시차 조정
        
        경제학적 의미:
        - 위기 시 시차가 단축됨 (빠른 전이)
        - 평시에는 시차가 길어짐 (느린 전이)
        
        Returns:
            int: 조정된 시차 (최소 1일)
        """
        adjustment = self.LAG_ADJUSTMENTS.get(regime, 1.0)
        adjusted = max(1, int(base_lag * adjustment))
        return adjusted
    
    def calculate_source_signal(
        self, 
        source_data: pd.DataFrame, 
        lag: int
    ) -> Tuple[float, float]:
        """
        소스 자산의 신호 계산
        
        로직:
        - lag일 전 대비 현재 수익률 계산
        - 거래량 대비 이상 여부 확인
        - 신호 강도 = |수익률| × 거래량비율 (정규화)
        
        Returns:
            Tuple[움직임(%), 신호강도(0-100)]
        """
        if source_data.empty or 'Close' not in source_data.columns:
            return 0.0, 0.0
        
        close = source_data['Close']
        
        if len(close) < lag + 1:
            return 0.0, 0.0
        
        # lag일 전 대비 현재 수익률
        move_pct = (float(close.iloc[-1]) / float(close.iloc[-lag-1]) - 1) * 100
        
        # 거래량 비율 계산
        volume_ratio = 1.0
        if 'Volume' in source_data.columns and len(source_data) >= self.lookback:
            volume = source_data['Volume']
            current_volume = float(volume.iloc[-1]) if len(volume) > 0 else 0
            avg_volume = float(volume.tail(self.lookback).mean()) if len(volume) >= self.lookback else current_volume
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
        
        # 신호 강도 계산: |움직임| × 거래량비율 (정규화)
        # 움직임 0-5% 범위를 0-50점으로, 거래량비율 1.0-2.0 범위를 0-50점으로
        move_score = min(50.0, abs(move_pct) / 5.0 * 50.0)
        volume_score = min(50.0, (volume_ratio - 1.0) / 1.0 * 50.0) if volume_ratio >= 1.0 else 0.0
        
        signal_strength = min(100.0, move_score + volume_score)
        
        return move_pct, signal_strength
    
    def check_path_activation(
        self, 
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        path: Dict,
        regime: str
    ) -> Optional[SpilloverEdge]:
        """
        개별 경로 활성화 여부 확인
        
        활성화 조건:
        1. 소스 자산이 임계값 이상 움직임 (예: 3일간 ±2%)
        2. 거래량이 평균 대비 1.5배 이상
        
        신호 강도 계산:
        - 움직임 크기 × 거래량 비율 × 레짐 가중치
        
        Returns:
            SpilloverEdge 객체 (활성화되지 않으면 None)
        """
        if source_data.empty or 'Close' not in source_data.columns:
            return None
        
        # 레짐별 임계값 가져오기
        thresholds = self.ACTIVATION_THRESHOLDS.get(regime, self.ACTIVATION_THRESHOLDS['BEAR'])
        min_move = thresholds['min_move']
        min_volume_ratio = thresholds['volume_ratio']
        
        # 시차 조정
        base_lag = path.get('base_lag', 3)
        adjusted_lag = self.adjust_lag_for_regime(base_lag, regime)
        
        # 소스 신호 계산
        source_move, signal_strength = self.calculate_source_signal(source_data, adjusted_lag)
        
        # 거래량 체크
        volume_ratio = 1.0
        if 'Volume' in source_data.columns and len(source_data) >= self.lookback:
            volume = source_data['Volume']
            current_volume = float(volume.iloc[-1]) if len(volume) > 0 else 0
            avg_volume = float(volume.tail(self.lookback).mean()) if len(volume) >= self.lookback else current_volume
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
        
        # 활성화 조건 체크
        abs_move = abs(source_move)
        is_active = (abs_move >= min_move) and (volume_ratio >= min_volume_ratio)
        
        if not is_active:
            return None
        
        # 레짐 가중치 적용 (위기 시 신호 강도 증가)
        regime_weights = {
            'BULL': 1.0,
            'TRANSITION': 1.1,
            'BEAR': 1.2,
            'CRISIS': 1.5
        }
        weight = regime_weights.get(regime, 1.0)
        final_signal_strength = min(100.0, signal_strength * weight)
        
        # 예상 타겟 방향 결정
        edge_type = path.get('edge_type', 'POSITIVE')
        if edge_type == 'POSITIVE':
            # 같은 방향: source가 하락하면 target도 하락
            expected_direction = "DOWN" if source_move < 0 else "UP"
        else:  # NEGATIVE
            # 반대 방향: source가 상승하면 target은 하락
            expected_direction = "DOWN" if source_move > 0 else "UP"
        
        return SpilloverEdge(
            source=path['source'],
            target=path['target'],
            edge_type=edge_type,
            base_lag=base_lag,
            adjusted_lag=adjusted_lag,
            signal_strength=final_signal_strength,
            is_active=True,
            source_move=source_move,
            expected_target_move=expected_direction,
            theory_note=path.get('theory', ''),
            category=path.get('category', '')
        )
    
    def get_expected_impacts(self, active_paths: List[SpilloverEdge]) -> Dict[str, str]:
        """
        활성화된 경로 기반으로 각 자산 예상 영향
        
        여러 경로가 같은 타겟을 가리킬 경우, 신호 강도가 높은 경로 우선
        
        Returns:
            Dict[ticker, expected_direction]
            예: {"QQQ": "DOWN", "GLD": "DOWN", "SPY": "DOWN"}
        """
        impacts = {}
        
        # 타겟별로 신호 강도가 가장 높은 경로 선택
        target_paths = {}
        for edge in active_paths:
            target = edge.target
            if target not in target_paths or edge.signal_strength > target_paths[target].signal_strength:
                target_paths[target] = edge
        
        # 예상 영향 결정
        for target, edge in target_paths.items():
            impacts[target] = edge.expected_target_move
        
        return impacts
    
    def identify_risk_source(self, active_paths: List[SpilloverEdge]) -> str:
        """
        가장 많은 경로의 소스가 되는 자산 = 위험 진원지
        
        Returns:
            str: 티커 이름 (활성 경로가 없으면 "NONE")
        """
        if not active_paths:
            return "NONE"
        
        # 소스별 경로 수 집계
        from collections import Counter
        source_counts = Counter(edge.source for edge in active_paths)
        
        if not source_counts:
            return "NONE"
        
        # 가장 많은 경로를 가진 소스 반환
        primary_source = source_counts.most_common(1)[0][0]
        return primary_source
    
    def calculate_risk_score(self, active_paths: List[SpilloverEdge], regime: str) -> float:
        """
        전이 위험 점수 계산 (0-100)
        
        로직:
        - 활성 경로 수 (최대 50점)
        - 평균 신호 강도 (최대 50점)
        - 레짐 가중치 적용
        
        Returns:
            float: 위험 점수 (0-100)
        """
        if not active_paths:
            return 0.0
        
        # 활성 경로 수 점수 (최대 50점)
        num_paths = len(active_paths)
        path_score = min(50.0, num_paths / 10.0 * 50.0)  # 10개 경로 = 50점
        
        # 평균 신호 강도 점수 (최대 50점)
        avg_strength = sum(edge.signal_strength for edge in active_paths) / len(active_paths)
        strength_score = avg_strength * 0.5  # 최대 50점
        
        base_score = path_score + strength_score
        
        # 레짐 가중치
        regime_weights = {
            'BULL': 0.8,
            'TRANSITION': 1.0,
            'BEAR': 1.2,
            'CRISIS': 1.5
        }
        weight = regime_weights.get(regime, 1.0)
        
        final_score = min(100.0, base_score * weight)
        return final_score
    
    def generate_interpretation(
        self,
        active_paths: List[SpilloverEdge],
        risk_score: float,
        primary_source: str,
        expected_impacts: Dict[str, str]
    ) -> str:
        """
        분석 결과 해석 텍스트 생성
        
        Returns:
            str: 해석 텍스트
        """
        if not active_paths:
            return "현재 활성화된 충격 전이 경로가 없습니다. 시장이 상대적으로 안정적입니다."
        
        base_text = f"총 {len(active_paths)}개의 충격 전이 경로가 활성화되어 있습니다. "
        
        # 위험 점수 해석
        if risk_score >= 70:
            base_text += f"⚠️ 전이 위험 점수가 높습니다 ({risk_score:.1f}점). "
        elif risk_score >= 50:
            base_text += f"전이 위험 점수가 보통입니다 ({risk_score:.1f}점). "
        else:
            base_text += f"전이 위험 점수가 낮습니다 ({risk_score:.1f}점). "
        
        # 주요 위험 진원지
        if primary_source != "NONE":
            base_text += f"주요 위험 진원지는 {primary_source}입니다. "
        
        # 예상 영향
        if expected_impacts:
            down_assets = [ticker for ticker, direction in expected_impacts.items() if direction == "DOWN"]
            up_assets = [ticker for ticker, direction in expected_impacts.items() if direction == "UP"]
            
            if down_assets:
                base_text += f"하락 압력이 예상되는 자산: {', '.join(down_assets)}. "
            if up_assets:
                base_text += f"상승 압력이 예상되는 자산: {', '.join(up_assets)}. "
        
        return base_text
    
    def analyze(
        self, 
        market_data: Dict[str, pd.DataFrame],
        regime: str
    ) -> SpilloverResult:
        """
        전체 네트워크 분석
        
        1. 각 경로별 활성화 여부 확인
        2. 활성화된 경로들의 위험도 합산
        3. 주요 위험 진원지 식별
        4. 타겟 자산별 예상 영향 정리
        
        Args:
            market_data: 티커별 가격 데이터 딕셔너리
            regime: 현재 레짐 ("BULL", "BEAR", "TRANSITION", "CRISIS")
        
        Returns:
            SpilloverResult 객체
        """
        active_paths = []
        
        # 각 경로별 활성화 여부 확인
        for path in self.paths:
            source_ticker = path['source']
            target_ticker = path['target']
            
            # 소스 데이터 가져오기 (티커명 변형 시도)
            source_data = market_data.get(source_ticker)
            if source_data is None:
                # 대체 티커 시도 (예: ^VIX -> VIX)
                alt_source = source_ticker.replace('^', '')
                source_data = market_data.get(alt_source)
            
            # 타겟 데이터 가져오기
            target_data = market_data.get(target_ticker)
            
            if source_data is None or target_data is None:
                continue
            
            # 경로 활성화 체크
            edge = self.check_path_activation(source_data, target_data, path, regime)
            if edge is not None:
                active_paths.append(edge)
        
        # 위험 점수 계산
        risk_score = self.calculate_risk_score(active_paths, regime)
        
        # 주요 위험 진원지 식별
        primary_source = self.identify_risk_source(active_paths)
        
        # 예상 영향 계산
        expected_impacts = self.get_expected_impacts(active_paths)
        
        # 해석 텍스트 생성
        interpretation = self.generate_interpretation(
            active_paths,
            risk_score,
            primary_source,
            expected_impacts
        )
        
        return SpilloverResult(
            timestamp=datetime.now().isoformat(),
            active_paths=active_paths,
            risk_score=risk_score,
            primary_risk_source=primary_source,
            expected_impacts=expected_impacts,
            interpretation=interpretation
        )

# critical_path_analyzer.py 파일 끝에 추가

