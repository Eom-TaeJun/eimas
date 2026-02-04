#!/usr/bin/env python3
"""
Microstructure - Volume Anomaly Detection
============================================================

거래량 이상 탐지

Economic Foundation:
    - Abnormal volume → potential informed trading
    - Z-score based anomaly detection
"""

from typing import List, Dict, Tuple
from collections import deque
import numpy as np
import logging

from .schemas import Trade

logger = logging.getLogger(__name__)


class VolumeAnomalyDetector:
    """
    이상 거래량(Anomaly Volume) 감지기
    
    Rule: 현재 거래량이 20일(또는 20주기) 이동평균 대비 
    3표준편차(3-sigma) 이상 급증 시 True 반환
    """
    
    def __init__(self, window: int = 20, threshold_sigma: float = 3.0):
        self.window = window
        self.threshold_sigma = threshold_sigma
        self.volume_history: deque = deque(maxlen=window + 1)
        
    def add_volume(self, volume: float) -> Tuple[bool, float, float]:
        """
        거래량 추가 및 이상 감지
        
        Returns:
            (is_anomaly, z_score, mean_volume)
        """
        self.volume_history.append(volume)
        
        if len(self.volume_history) < self.window:
            return False, 0.0, 0.0
            
        # 최근 volume을 제외한 이전 window개 데이터로 통계 계산
        recent_history = list(self.volume_history)[:-1]
        mean = np.mean(recent_history)
        std = np.std(recent_history)
        
        if std == 0:
            return False, 0.0, mean
            
        z_score = (volume - mean) / std
        is_anomaly = z_score > self.threshold_sigma
        
        return is_anomaly, z_score, mean


