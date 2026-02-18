"""
EIMAS Realtime Intelligence — 실시간 분석 패키지
==================================================
실시간 스트리밍 및 마이크로구조 분석 모듈의 단일 진입점.

기존 경로(from pipeline.realtime import ..., from lib.binance_stream import ...)는 그대로 유지됨.
신규 권장 경로: from lib.realtime_intelligence import BinanceStreamer
"""

from lib.binance_stream import BinanceStreamer, StreamConfig
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig

__all__ = [
    'BinanceStreamer',
    'StreamConfig',
    'RealtimePipeline',
    'PipelineConfig',
]
