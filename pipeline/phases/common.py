#!/usr/bin/env python3
"""
EIMAS Pipeline - Common Utilities

Purpose:
    Shared helper functions for all phase modules

Functions:
    - safe_call: Error-safe function wrapper with fallback
"""

from typing import Callable, Any, Optional
import logging

logger = logging.getLogger("pipeline.phases")

def safe_call(
    func: Callable,
    *args,
    fallback: Any = None,
    error_msg: str = "Operation failed",
    **kwargs
) -> Any:
    """
    Error-safe function wrapper with fallback value.

    Args:
        func: Function to call
        *args: Positional arguments
        fallback: Return value on error
        error_msg: Error log message
        **kwargs: Keyword arguments

    Returns:
        Function result or fallback value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return fallback
