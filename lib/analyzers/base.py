"""
EIMAS Analyzers - Base Interface
================================
Abstract base class for all market analyzers.

Design Pattern: Strategy Pattern
- Define family of analysis algorithms
- Make them interchangeable at runtime
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging


class BaseAnalyzer(ABC):
    """
    Base interface for all market analyzers.
    기본 분석 엔진 인터페이스.
    
    All analyzers should implement:
        - analyze(): Main analysis logic
        - get_summary(): Human-readable summary
        - get_analyzer_name(): Return analyzer identifier
    
    Attributes:
        logger: Logger instance for this analyzer
        last_analysis_time: Timestamp of last analysis
        config: Analyzer configuration
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the analyzer.
        
        Args:
            name: Optional name for the analyzer (defaults to class name)
            config: Optional configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f'eimas.analyzers.{self.name}')
        self.last_analysis_time: Optional[datetime] = None
        self.config = config or {}
        self._last_result: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on input data.
        입력 데이터에 대한 분석 수행.
        
        Args:
            data: Input data for analysis
            **kwargs: Additional analysis parameters
            
        Returns:
            Dict containing analysis results
        """
        pass
    
    @abstractmethod
    def get_summary(self) -> str:
        """
        Return human-readable summary of last analysis.
        마지막 분석 결과의 요약 반환.
        
        Returns:
            String summary of analysis results
        """
        pass
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """
        Return the analyzer identifier.
        분석기 식별자 반환.
        
        Returns:
            String identifier for this analyzer
        """
        pass
    
    def analyze_with_logging(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze with automatic logging and timing.
        
        Args:
            data: Input data for analysis
            **kwargs: Additional parameters
            
        Returns:
            Analysis result dictionary
        """
        self.logger.info(f"Starting analysis: {self.get_analyzer_name()}")
        start_time = datetime.now()
        
        try:
            result = self.analyze(data, **kwargs)
            self._last_result = result
            self.last_analysis_time = datetime.now()
            
            elapsed = (self.last_analysis_time - start_time).total_seconds()
            self.logger.info(f"Analysis complete in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return analyzer status information.
        
        Returns:
            Dict with status info
        """
        return {
            'name': self.name,
            'analyzer': self.get_analyzer_name(),
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'has_result': self._last_result is not None,
            'config': self.config
        }
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self._last_result = None
        self.last_analysis_time = None
