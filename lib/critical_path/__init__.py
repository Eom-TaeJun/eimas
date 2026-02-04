#!/usr/bin/env python3
"""
Critical Path Package
=====================

EIMAS Critical Path 분석 패키지

모듈 구조:
    - schemas.py: 데이터 클래스 및 헬퍼 함수
    - risk_appetite.py: 리스크 선호도/불확실성 분리 측정
    - regime.py: 시장 레짐 탐지
    - spillover.py: 자산간 충격 전이 네트워크
    - crypto_sentiment.py: 암호화폐 심리 분석
    - stress.py: 스트레스 레짐 승수
    - aggregator.py: 모든 모듈 통합 (메인 클래스)

Public API:
    - CriticalPathAggregator: 메인 분석 클래스
    - CriticalPathResult: 분석 결과 데이터클래스

Usage:
    ```python
    from lib.critical_path import CriticalPathAggregator

    aggregator = CriticalPathAggregator()
    result = aggregator.analyze(market_data)
    print(f"Risk Score: {result.total_risk_score}")
    ```
"""

# Import main aggregator class (only public API)
from .aggregator import CriticalPathAggregator

# Import result schema
from .schemas import CriticalPathResult

# Package metadata
__version__ = "2.0.0"
__author__ = "EIMAS Team"
__all__ = ["CriticalPathAggregator", "CriticalPathResult"]
