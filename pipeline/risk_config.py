"""
Risk Adjustment Configuration Module

This module provides configuration classes for extended data risk adjustments.
It loads threshold values from YAML configuration files to avoid hardcoded values.

Usage:
    from pipeline.risk_config import load_risk_config

    config = load_risk_config()
    if pcr_ratio > config.pcr.high_threshold:
        adjustment = config.pcr.high_adjustment
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PCRConfig:
    """Put/Call Ratio adjustment configuration."""
    high_threshold: float
    high_adjustment: float
    low_threshold: float
    low_adjustment: float


@dataclass
class CryptoFNGConfig:
    """Crypto Fear & Greed Index adjustment configuration."""
    fear_threshold: int
    fear_adjustment: float
    greed_threshold: int
    greed_adjustment: float


@dataclass
class NewsSentimentConfig:
    """News sentiment adjustment configuration."""
    bearish_adjustment: float
    bullish_adjustment: float


@dataclass
class CreditSpreadsConfig:
    """Credit spreads adjustment configuration."""
    risk_off_adjustment: float
    risk_on_adjustment: float


@dataclass
class KoreaRiskConfig:
    """Korea risk (KRW) adjustment configuration."""
    overheated_adjustment: float
    volatile_adjustment: float


@dataclass
class ConstraintsConfig:
    """Global adjustment constraints."""
    min_adjustment: float
    max_adjustment: float


@dataclass
class RiskAdjustmentConfig:
    """
    Complete risk adjustment configuration.

    Attributes:
        pcr: Put/Call Ratio configuration
        crypto_fng: Crypto Fear & Greed configuration
        news_sentiment: News sentiment configuration
        credit_spreads: Credit spreads configuration
        korea_risk: Korea risk configuration
        constraints: Global constraints
    """
    pcr: PCRConfig
    crypto_fng: CryptoFNGConfig
    news_sentiment: NewsSentimentConfig
    credit_spreads: CreditSpreadsConfig
    korea_risk: KoreaRiskConfig
    constraints: ConstraintsConfig


def load_risk_config(config_path: Optional[Path] = None) -> RiskAdjustmentConfig:
    """
    Load risk adjustment configuration from YAML file.

    Args:
        config_path: Optional custom path to config file.
                    If None, uses default path: configs/risk_adjustments.yaml

    Returns:
        RiskAdjustmentConfig object with all threshold values

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file is malformed
        KeyError: If required configuration keys are missing

    Example:
        >>> config = load_risk_config()
        >>> print(config.pcr.high_threshold)
        1.0
        >>> print(config.constraints.max_adjustment)
        15.0
    """
    if config_path is None:
        # Default path relative to this file
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "configs" / "risk_adjustments.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Risk adjustment config file not found: {config_path}"
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Parse nested configurations
    try:
        config = RiskAdjustmentConfig(
            pcr=PCRConfig(**data['pcr']),
            crypto_fng=CryptoFNGConfig(**data['crypto_fng']),
            news_sentiment=NewsSentimentConfig(**data['news_sentiment']),
            credit_spreads=CreditSpreadsConfig(**data['credit_spreads']),
            korea_risk=KoreaRiskConfig(**data['korea_risk']),
            constraints=ConstraintsConfig(**data['constraints'])
        )
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    return config


# Module-level cache for configuration
_config_cache: Optional[RiskAdjustmentConfig] = None


def get_risk_config(reload: bool = False) -> RiskAdjustmentConfig:
    """
    Get cached risk adjustment configuration.

    This is the recommended way to access the configuration in production code.
    The configuration is loaded once and cached for subsequent calls.

    Args:
        reload: If True, force reload from file (useful for testing)

    Returns:
        Cached RiskAdjustmentConfig object

    Example:
        >>> from pipeline.risk_config import get_risk_config
        >>> config = get_risk_config()
        >>> if value > config.pcr.high_threshold:
        ...     adjustment = config.pcr.high_adjustment
    """
    global _config_cache

    if _config_cache is None or reload:
        _config_cache = load_risk_config()

    return _config_cache


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Risk Adjustment Configuration...")
    print("-" * 60)

    try:
        config = load_risk_config()

        print("✅ Configuration loaded successfully!")
        print()

        print("PCR Configuration:")
        print(f"  High threshold: {config.pcr.high_threshold}")
        print(f"  High adjustment: {config.pcr.high_adjustment}")
        print(f"  Low threshold: {config.pcr.low_threshold}")
        print(f"  Low adjustment: {config.pcr.low_adjustment}")
        print()

        print("Crypto F&G Configuration:")
        print(f"  Fear threshold: {config.crypto_fng.fear_threshold}")
        print(f"  Fear adjustment: {config.crypto_fng.fear_adjustment}")
        print(f"  Greed threshold: {config.crypto_fng.greed_threshold}")
        print(f"  Greed adjustment: {config.crypto_fng.greed_adjustment}")
        print()

        print("News Sentiment Configuration:")
        print(f"  Bearish adjustment: {config.news_sentiment.bearish_adjustment}")
        print(f"  Bullish adjustment: {config.news_sentiment.bullish_adjustment}")
        print()

        print("Credit Spreads Configuration:")
        print(f"  Risk OFF adjustment: {config.credit_spreads.risk_off_adjustment}")
        print(f"  Risk ON adjustment: {config.credit_spreads.risk_on_adjustment}")
        print()

        print("Korea Risk Configuration:")
        print(f"  Overheated adjustment: {config.korea_risk.overheated_adjustment}")
        print(f"  Volatile adjustment: {config.korea_risk.volatile_adjustment}")
        print()

        print("Constraints:")
        print(f"  Min adjustment: {config.constraints.min_adjustment}")
        print(f"  Max adjustment: {config.constraints.max_adjustment}")
        print()

        print("-" * 60)
        print("✅ All configurations valid!")

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        raise
