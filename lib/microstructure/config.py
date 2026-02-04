#!/usr/bin/env python3
"""
Microstructure - Configuration
============================================================

Rolling window 설정 및 파라미터
"""

from dataclasses import dataclass


@dataclass
class RollingWindowConfig:
    """
    Rolling Window 설정
    
    경제학적 근거:
    - window_size: 충분한 샘플 크기 (n > 30)
    - bucket_size: VPIN 계산을 위한 volume bucket
    """
    window_size: int = 50
    bucket_size: int = 50
    min_samples: int = 30
    
    # OFI configuration
    ofi_normalize: bool = True
    ofi_window: int = 20
    
    # VPIN configuration
    vpin_buckets: int = 50
    vpin_samples: int = 50
    
    # Depth configuration
    depth_levels: int = 10
    depth_threshold: float = 0.1
    
    # Volume anomaly
    volume_threshold: float = 3.0  # z-score threshold
