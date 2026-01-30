"""
EIMAS Collectors - Base Interface
=================================
Abstract base class for all data collectors.

Design Pattern: Template Method
- Define skeleton of collection algorithm
- Subclasses implement specific data source logic
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging


class BaseCollector(ABC):
    """
    Base interface for all data collectors.
    기본 데이터 수집기 인터페이스.
    
    All collectors should implement:
        - collect(): Main data collection logic
        - validate(): Data validation
        - get_source_name(): Return data source identifier
    
    Attributes:
        logger: Logger instance for this collector
        last_collection_time: Timestamp of last successful collection
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            name: Optional name for the collector (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f'eimas.collectors.{self.name}')
        self.last_collection_time: Optional[datetime] = None
    
    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Collect data from the source.
        데이터 소스에서 데이터 수집.
        
        Returns:
            Dict containing collected data
            
        Raises:
            CollectionError: If data collection fails
        """
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate collected data.
        수집된 데이터 검증.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the data source identifier.
        데이터 소스 식별자 반환.
        
        Returns:
            String identifier for this data source
        """
        pass
    
    def collect_with_validation(self, **kwargs) -> Dict[str, Any]:
        """
        Collect and validate data (Template Method).
        
        This method orchestrates the collection workflow:
        1. Call collect() to get data
        2. Call validate() to ensure data quality
        3. Update collection timestamp
        
        Returns:
            Validated data dictionary
            
        Raises:
            ValueError: If validation fails
        """
        self.logger.info(f"Starting collection from {self.get_source_name()}")
        
        data = self.collect(**kwargs)
        
        if not self.validate(data):
            raise ValueError(f"Data validation failed for {self.get_source_name()}")
        
        self.last_collection_time = datetime.now()
        self.logger.info(f"Collection complete: {len(data)} items")
        
        return data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Return collector status information.
        
        Returns:
            Dict with status info including last collection time
        """
        return {
            'name': self.name,
            'source': self.get_source_name(),
            'last_collection': self.last_collection_time.isoformat() if self.last_collection_time else None,
            'status': 'ready'
        }
