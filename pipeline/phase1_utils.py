"""
Shared Phase 1 utilities used by both the sequential path (phase1_collect.py)
and the parallel path (team_coordinator.py).

Extracted to avoid circular / private-function imports between modules.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OFFLINE_FALLBACK_TICKERS = (
    "SPY,QQQ,TLT,GLD,^VIX,HYG,LQD,XLY,XLP,IWM,XLF,"
    "BTC-USD,SMH,NVDA,DXY,DX-Y.NYB,EEM"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def env_flag(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def count_dataframe_assets(market_data: Dict[str, Any]) -> int:
    """Count entries in *market_data* whose value is a non-empty DataFrame."""
    count = 0
    for payload in market_data.values():
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            count += 1
    return count


def inject_offline_fallback_market_data(
    market_data: Dict[str, Any],
    lookback_days: int,
) -> tuple[int, int]:
    """Inject synthetic OHLCV data for missing fallback tickers.

    Returns (injected_count, total_fallback_tickers).
    """
    fallback_tickers = _resolve_fallback_tickers()
    missing_tickers: list[str] = []

    for ticker in fallback_tickers:
        payload = market_data.get(ticker)
        if isinstance(payload, pd.DataFrame) and not payload.empty:
            continue
        missing_tickers.append(ticker)

    if not missing_tickers:
        return 0, len(fallback_tickers)

    synthetic_bundle = _build_correlated_synthetic_bundle(
        missing_tickers, lookback_days=lookback_days,
    )
    for ticker, frame in synthetic_bundle.items():
        market_data[ticker] = frame

    return len(missing_tickers), len(fallback_tickers)


# ---------------------------------------------------------------------------
# Internal helpers (used by inject_offline_fallback_market_data)
# ---------------------------------------------------------------------------

def _resolve_fallback_tickers() -> list[str]:
    raw = os.getenv("EIMAS_OFFLINE_MARKET_FALLBACK_TICKERS", _OFFLINE_FALLBACK_TICKERS)
    tickers = [item.strip() for item in raw.split(",") if item.strip()]
    return tickers


def _seed_for_ticker(ticker: str) -> int:
    digest = hashlib.sha256(ticker.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _base_price_for_ticker(ticker: str) -> float:
    defaults = {
        "SPY": 520.0,
        "QQQ": 450.0,
        "TLT": 95.0,
        "GLD": 210.0,
        "^VIX": 18.0,
        "VIX": 18.0,
        "HYG": 78.0,
        "LQD": 108.0,
        "XLY": 180.0,
        "XLP": 76.0,
        "IWM": 205.0,
        "XLF": 43.0,
        "BTC-USD": 65000.0,
        "SMH": 250.0,
        "NVDA": 900.0,
        "DXY": 104.0,
        "DX-Y.NYB": 104.0,
        "EEM": 43.0,
    }
    return defaults.get(ticker, 100.0)


def _synthetic_return_profile(ticker: str) -> Dict[str, float]:
    """Return drift/vol and factor betas for synthetic correlated market generation."""
    profiles = {
        "SPY": {"drift": 0.00030, "vol": 0.011, "beta_eq": 1.00, "beta_rates": -0.10, "beta_risk": -0.28, "beta_dxy": -0.12, "beta_crypto": 0.00, "beta_comd": 0.05},
        "QQQ": {"drift": 0.00045, "vol": 0.014, "beta_eq": 1.18, "beta_rates": -0.15, "beta_risk": -0.34, "beta_dxy": -0.10, "beta_crypto": 0.05, "beta_comd": 0.00},
        "IWM": {"drift": 0.00036, "vol": 0.014, "beta_eq": 1.10, "beta_rates": -0.08, "beta_risk": -0.25, "beta_dxy": -0.16, "beta_crypto": 0.00, "beta_comd": 0.04},
        "XLF": {"drift": 0.00031, "vol": 0.012, "beta_eq": 0.92, "beta_rates": 0.20, "beta_risk": -0.18, "beta_dxy": -0.08, "beta_crypto": 0.00, "beta_comd": 0.00},
        "XLY": {"drift": 0.00033, "vol": 0.012, "beta_eq": 1.03, "beta_rates": -0.08, "beta_risk": -0.20, "beta_dxy": -0.10, "beta_crypto": 0.00, "beta_comd": 0.00},
        "XLP": {"drift": 0.00018, "vol": 0.008, "beta_eq": 0.55, "beta_rates": -0.03, "beta_risk": 0.05, "beta_dxy": -0.05, "beta_crypto": 0.00, "beta_comd": 0.02},
        "SMH": {"drift": 0.00055, "vol": 0.018, "beta_eq": 1.35, "beta_rates": -0.16, "beta_risk": -0.38, "beta_dxy": -0.12, "beta_crypto": 0.06, "beta_comd": 0.00},
        "NVDA": {"drift": 0.00062, "vol": 0.022, "beta_eq": 1.75, "beta_rates": -0.12, "beta_risk": -0.40, "beta_dxy": -0.12, "beta_crypto": 0.10, "beta_comd": 0.00},
        "TLT": {"drift": 0.00010, "vol": 0.007, "beta_eq": -0.12, "beta_rates": -0.85, "beta_risk": 0.22, "beta_dxy": 0.10, "beta_crypto": 0.00, "beta_comd": -0.04},
        "LQD": {"drift": 0.00012, "vol": 0.006, "beta_eq": 0.18, "beta_rates": -0.45, "beta_risk": 0.10, "beta_dxy": 0.08, "beta_crypto": 0.00, "beta_comd": 0.00},
        "HYG": {"drift": 0.00016, "vol": 0.008, "beta_eq": 0.52, "beta_rates": -0.35, "beta_risk": -0.12, "beta_dxy": -0.04, "beta_crypto": 0.00, "beta_comd": 0.00},
        "GLD": {"drift": 0.00018, "vol": 0.009, "beta_eq": 0.08, "beta_rates": -0.20, "beta_risk": 0.30, "beta_dxy": -0.62, "beta_crypto": 0.00, "beta_comd": 0.30},
        "BTC-USD": {"drift": 0.00070, "vol": 0.024, "beta_eq": 0.35, "beta_rates": -0.20, "beta_risk": -0.55, "beta_dxy": -0.32, "beta_crypto": 1.30, "beta_comd": 0.00},
        "DXY": {"drift": 0.00008, "vol": 0.004, "beta_eq": -0.20, "beta_rates": 0.24, "beta_risk": 0.18, "beta_dxy": 1.00, "beta_crypto": -0.10, "beta_comd": -0.12},
        "DX-Y.NYB": {"drift": 0.00008, "vol": 0.004, "beta_eq": -0.20, "beta_rates": 0.24, "beta_risk": 0.18, "beta_dxy": 1.00, "beta_crypto": -0.10, "beta_comd": -0.12},
        "EEM": {"drift": 0.00024, "vol": 0.012, "beta_eq": 0.84, "beta_rates": -0.05, "beta_risk": -0.16, "beta_dxy": -0.52, "beta_crypto": 0.00, "beta_comd": 0.10},
        "^VIX": {"drift": 0.00000, "vol": 0.018, "beta_eq": -1.55, "beta_rates": 0.10, "beta_risk": 1.70, "beta_dxy": 0.12, "beta_crypto": 0.00, "beta_comd": 0.00},
        "VIX": {"drift": 0.00000, "vol": 0.018, "beta_eq": -1.55, "beta_rates": 0.10, "beta_risk": 1.70, "beta_dxy": 0.12, "beta_crypto": 0.00, "beta_comd": 0.00},
    }
    return profiles.get(
        ticker,
        {"drift": 0.00025, "vol": 0.011, "beta_eq": 0.75, "beta_rates": -0.05, "beta_risk": -0.10, "beta_dxy": -0.05, "beta_crypto": 0.00, "beta_comd": 0.00},
    )


def _build_synthetic_ohlcv(
    ticker: str,
    lookback_days: int,
    dates: pd.DatetimeIndex | None = None,
    shared_factors: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    points = max(180, min(lookback_days, 260))
    if dates is None:
        dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=points)
    else:
        points = len(dates)
    rng = np.random.default_rng(_seed_for_ticker(ticker))

    profile = _synthetic_return_profile(ticker)
    drift = profile["drift"]
    vol = profile["vol"]
    base_price = _base_price_for_ticker(ticker)

    if shared_factors:
        eq = shared_factors["eq"]
        rates = shared_factors["rates"]
        risk = shared_factors["risk"]
        dxy = shared_factors["dxy"]
        crypto = shared_factors["crypto"]
        commodity = shared_factors["commodity"]
        idio = rng.normal(0, vol * 0.55, size=points)
        returns = (
            drift
            + profile["beta_eq"] * eq
            + profile["beta_rates"] * rates
            + profile["beta_risk"] * risk
            + profile["beta_dxy"] * dxy
            + profile["beta_crypto"] * crypto
            + profile["beta_comd"] * commodity
            + idio
        )
    else:
        returns = drift + rng.normal(0, vol, size=points)

    if ticker in {"^VIX", "VIX"} and shared_factors:
        level = np.empty(points, dtype=float)
        level[0] = base_price
        for i in range(1, points):
            level[i] = max(10.0, level[i - 1] + 0.18 * (18.0 - level[i - 1]) + returns[i] * 7.5)
        close = level
    else:
        close = base_price * np.exp(np.cumsum(returns))

    open_ = close * (1 + rng.normal(0, vol * 0.2, size=points))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.18, size=points)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.18, size=points)))
    low = np.maximum(low, 0.01)
    volume_scale = 1.0 + np.clip(np.abs(returns) * 12.0, 0.0, 2.5)
    raw_volume = rng.integers(1_000_000, 9_000_000, size=points).astype(float)
    volume = (raw_volume * volume_scale).astype(int)

    frame = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def _build_correlated_synthetic_bundle(
    tickers: list[str],
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    points = max(180, min(lookback_days, 260))
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=points)
    rng = np.random.default_rng(20260208)

    shared_factors = {
        "eq": rng.normal(0, 0.0075, size=points),
        "rates": rng.normal(0, 0.0035, size=points),
        "risk": rng.normal(0, 0.0050, size=points),
        "dxy": rng.normal(0, 0.0028, size=points),
        "crypto": rng.normal(0, 0.0120, size=points),
        "commodity": rng.normal(0, 0.0042, size=points),
    }

    bundle: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        bundle[ticker] = _build_synthetic_ohlcv(
            ticker=ticker,
            lookback_days=lookback_days,
            dates=dates,
            shared_factors=shared_factors,
        )
    return bundle
