"""
Backtest Enumerations
=====================
백테스트 관련 열거형 정의

Economic Foundation:
- Prado (2018): "Advances in Financial Machine Learning" - Chapter 7 (Cross-Validation)
- Harvey, Liu, Zhu (2016): "...and the Cross-Section of Expected Returns"
"""

from enum import Enum


class RebalanceFrequency(str, Enum):
    """리밸런싱 주기"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @property
    def periods_per_year(self) -> int:
        """연간 리밸런싱 횟수"""
        return {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }[self.value]


class BacktestMode(str, Enum):
    """백테스트 모드"""
    WALKFORWARD = "walkforward"      # Walk-forward analysis
    ROLLING = "rolling"              # Rolling window
    EXPANDING = "expanding"          # Expanding window
    FIXED_SPLIT = "fixed_split"      # Single train/test split
