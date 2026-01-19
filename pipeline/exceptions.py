#!/usr/bin/env python3
"""
EIMAS Pipeline - Exceptions & Logging
======================================

Purpose:
    파이프라인 전용 예외 클래스 및 로깅 유틸리티

Functions:
    - log_error(logger, msg, exc)
    - log_warning(logger, msg)

Classes:
    - PipelineError
    - CollectionError
    - AnalysisError
"""

import logging
from typing import Optional

# 기본 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"eimas.pipeline.{name}")

def log_error(logger: logging.Logger, message: str, exception: Optional[Exception] = None):
    """표준화된 에러 로깅 (Console에는 간단히, Log에는 상세히)"""
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
        print(f"      ✗ {message}: {str(exception)}")
    else:
        logger.error(message)
        print(f"      ✗ {message}")

def log_warning(logger: logging.Logger, message: str):
    """경고 로깅"""
    logger.warning(message)
    print(f"      ⚠ {message}")

class PipelineError(Exception):
    """Base class for pipeline exceptions"""
    pass

class CollectionError(PipelineError):
    """Data collection failed"""
    pass

class AnalysisError(PipelineError):
    """Analysis logic failed"""
    pass

class DebateError(PipelineError):
    """Agent debate failed"""
    pass
