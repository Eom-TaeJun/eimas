#!/usr/bin/env python3
"""
EIMAS Logging Configuration
============================
구조화된 로깅 시스템

주요 기능:
1. 구조화 로깅 (JSON 포맷)
2. 파일 로테이션
3. 레벨별 핸들러
4. 컨텍스트 추적

Usage:
    from core.logging_config import setup_logging, get_logger

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Hello", extra={'user_id': 123})
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback


# ============================================================================
# Constants
# ============================================================================

LOG_DIR = Path('/home/tj/projects/autoai/eimas/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 로그 파일 경로
LOG_FILES = {
    'main': LOG_DIR / 'eimas.log',
    'error': LOG_DIR / 'error.log',
    'debug': LOG_DIR / 'debug.log',
    'api': LOG_DIR / 'api.log',
    'trading': LOG_DIR / 'trading.log',
}

# 기본 설정
DEFAULT_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
JSON_FORMAT = True
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


# ============================================================================
# Enums and Data Classes
# ============================================================================

class LogLevel(str, Enum):
    """로그 레벨"""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


@dataclass
class LogContext:
    """로그 컨텍스트"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogEntry:
    """구조화된 로그 엔트리"""
    timestamp: str
    level: str
    logger: str
    message: str
    context: Dict[str, Any]
    exception: Optional[str] = None
    traceback: Optional[str] = None

    def to_json(self) -> str:
        data = asdict(self)
        # None 값 제거
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, ensure_ascii=False)


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """JSON 포맷터"""

    def format(self, record: logging.LogRecord) -> str:
        # 기본 필드
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            context={},
        )

        # extra 필드 추가
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'taskName', 'message',
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                try:
                    # JSON 직렬화 가능한지 확인
                    json.dumps(value)
                    entry.context[key] = value
                except (TypeError, ValueError):
                    entry.context[key] = str(value)

        # 예외 정보
        if record.exc_info:
            entry.exception = str(record.exc_info[1])
            entry.traceback = ''.join(traceback.format_exception(*record.exc_info))

        return entry.to_json()


class ColorFormatter(logging.Formatter):
    """컬러 콘솔 포맷터"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # 레벨 컬러링
        record.levelname = f"{color}{record.levelname:8s}{reset}"

        return super().format(record)


# ============================================================================
# Context Manager
# ============================================================================

class LogContextManager:
    """로그 컨텍스트 관리"""

    _context = threading.local()

    @classmethod
    def set_context(cls, **kwargs):
        """컨텍스트 설정"""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        cls._context.data.update(kwargs)

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """컨텍스트 조회"""
        if not hasattr(cls._context, 'data'):
            return {}
        return cls._context.data.copy()

    @classmethod
    def clear_context(cls):
        """컨텍스트 초기화"""
        if hasattr(cls._context, 'data'):
            cls._context.data = {}


class ContextFilter(logging.Filter):
    """컨텍스트 필터"""

    def filter(self, record: logging.LogRecord) -> bool:
        context = LogContextManager.get_context()
        for key, value in context.items():
            setattr(record, key, value)
        return True


# ============================================================================
# Structured Logger
# ============================================================================

class StructuredLogger:
    """구조화된 로거 래퍼"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log(self, level: int, msg: str, *args, **kwargs):
        """로그 기록"""
        extra = kwargs.pop('extra', {})

        # 컨텍스트 추가
        context = LogContextManager.get_context()
        extra.update(context)

        # 추가 필드
        for key in list(kwargs.keys()):
            if key not in ('exc_info', 'stack_info', 'stacklevel'):
                extra[key] = kwargs.pop(key)

        kwargs['extra'] = extra
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)

    # 편의 메서드
    def signal(self, ticker: str, action: str, confidence: float, **kwargs):
        """시그널 로깅"""
        self.info(
            f"Signal: {ticker} {action}",
            ticker=ticker,
            action=action,
            confidence=confidence,
            log_type='signal',
            **kwargs
        )

    def trade(self, ticker: str, side: str, quantity: int, price: float, **kwargs):
        """거래 로깅"""
        self.info(
            f"Trade: {side} {quantity} {ticker} @ {price}",
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
            log_type='trade',
            **kwargs
        )

    def risk(self, metric: str, value: float, threshold: float, **kwargs):
        """리스크 로깅"""
        level = logging.WARNING if value > threshold else logging.INFO
        self._log(
            level,
            f"Risk: {metric} = {value:.2%} (threshold: {threshold:.2%})",
            metric=metric,
            value=value,
            threshold=threshold,
            log_type='risk',
            **kwargs
        )

    def performance(self, metric: str, value: float, **kwargs):
        """성과 로깅"""
        self.info(
            f"Performance: {metric} = {value:.2%}",
            metric=metric,
            value=value,
            log_type='performance',
            **kwargs
        )


# ============================================================================
# Setup Functions
# ============================================================================

def setup_logging(
    level: str = 'INFO',
    json_format: bool = JSON_FORMAT,
    console: bool = True,
    file_logging: bool = True,
):
    """로깅 설정"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 컨텍스트 필터
    context_filter = ContextFilter()

    # 콘솔 핸들러
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColorFormatter(DEFAULT_FORMAT))

        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

    # 파일 핸들러
    if file_logging:
        # 메인 로그 (INFO+)
        main_handler = logging.handlers.RotatingFileHandler(
            LOG_FILES['main'],
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
        )
        main_handler.setLevel(logging.INFO)
        main_handler.setFormatter(JSONFormatter() if json_format else logging.Formatter(DEFAULT_FORMAT))
        main_handler.addFilter(context_filter)
        root_logger.addHandler(main_handler)

        # 에러 로그 (ERROR+)
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_FILES['error'],
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if json_format else logging.Formatter(DEFAULT_FORMAT))
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)

        # 디버그 로그 (DEBUG+)
        if level == 'DEBUG':
            debug_handler = logging.handlers.RotatingFileHandler(
                LOG_FILES['debug'],
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(JSONFormatter() if json_format else logging.Formatter(DEFAULT_FORMAT))
            debug_handler.addFilter(context_filter)
            root_logger.addHandler(debug_handler)

    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """구조화된 로거 조회"""
    logger = logging.getLogger(name)
    return StructuredLogger(logger)


def set_context(**kwargs):
    """로그 컨텍스트 설정"""
    LogContextManager.set_context(**kwargs)


def clear_context():
    """로그 컨텍스트 초기화"""
    LogContextManager.clear_context()


# ============================================================================
# Convenience Functions
# ============================================================================

def log_function_call(func):
    """함수 호출 로깅 데코레이터"""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(
            f"Calling {func.__name__}",
            function=func.__name__,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(
                f"Completed {func.__name__}",
                function=func.__name__,
                success=True,
            )
            return result
        except Exception as e:
            logger.exception(
                f"Error in {func.__name__}: {e}",
                function=func.__name__,
                success=False,
            )
            raise

    return wrapper


def log_execution_time(func):
    """실행 시간 로깅 데코레이터"""
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(
                f"{func.__name__} completed in {elapsed:.2f}s",
                function=func.__name__,
                execution_time=elapsed,
            )
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.exception(
                f"{func.__name__} failed after {elapsed:.2f}s: {e}",
                function=func.__name__,
                execution_time=elapsed,
            )
            raise

    return wrapper


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 로깅 설정
    setup_logging(level='DEBUG', json_format=False, console=True, file_logging=True)

    # 기본 로거
    logger = get_logger(__name__)

    print("=== Basic Logging ===")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print("\n=== Structured Logging ===")
    logger.info("User action", user_id="123", action="login", ip="192.168.1.1")

    print("\n=== Context Logging ===")
    set_context(request_id="req-123", session_id="sess-456")
    logger.info("With context")
    clear_context()

    print("\n=== Domain Specific ===")
    logger.signal("AAPL", "BUY", 0.85, reason="momentum")
    logger.trade("AAPL", "BUY", 100, 150.25)
    logger.risk("var_95", 0.08, 0.05)
    logger.performance("sharpe_ratio", 1.85)

    print("\n=== Exception Logging ===")
    try:
        1 / 0
    except Exception:
        logger.exception("Division error")

    print("\n=== Decorator Test ===")

    @log_execution_time
    def slow_function():
        import time
        time.sleep(0.1)
        return "done"

    slow_function()

    print("\nLogging test complete!")
    print(f"Log files in: {LOG_DIR}")
