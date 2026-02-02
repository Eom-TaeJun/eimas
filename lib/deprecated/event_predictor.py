#!/usr/bin/env python3
"""
EIMAS Event Predictor
=====================
과거 연구 및 백테스트 결과를 바탕으로 이벤트 전후 가격 예측

주요 기능:
1. 과거 이벤트 패턴 분석
2. 현재 시장 상태 평가
3. 이벤트 전(T-5~T-1) 예상 움직임
4. 이벤트 후(T+1~T+5) 시나리오별 예측
5. 신뢰구간 및 확률 제공

사용법:
    from lib.event_predictor import EventPredictor

    predictor = EventPredictor()
    predictions = predictor.predict_upcoming_events()
    predictor.print_predictions(predictions)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Historical Research Data (연구 기반 통계)
# ============================================================================

# 이벤트별 과거 통계 (2020-2025 데이터 기반)
HISTORICAL_PATTERNS = {
    "fomc": {
        "name": "FOMC Rate Decision",
        # 이벤트 전 패턴
        "pre_event": {
            "t_minus_5_to_1": {
                "avg_return": 0.15,  # %
                "std": 0.8,
                "win_rate": 0.55,
                "pattern": "Cautious positioning, slight drift up"
            },
            "t_minus_1": {
                "avg_return": 0.05,
                "std": 0.5,
                "pattern": "Consolidation before announcement"
            }
        },
        # 이벤트 후 패턴 (시나리오별)
        "post_event": {
            "hawkish_surprise": {
                "t_plus_1": {"avg": -1.2, "std": 0.8},
                "t_plus_5": {"avg": -0.5, "std": 1.5},
                "probability": 0.15
            },
            "hawkish_inline": {
                "t_plus_1": {"avg": -0.3, "std": 0.6},
                "t_plus_5": {"avg": 0.2, "std": 1.0},
                "probability": 0.25
            },
            "dovish_inline": {
                "t_plus_1": {"avg": 0.4, "std": 0.6},
                "t_plus_5": {"avg": 0.8, "std": 1.0},
                "probability": 0.35
            },
            "dovish_surprise": {
                "t_plus_1": {"avg": 1.5, "std": 0.8},
                "t_plus_5": {"avg": 2.0, "std": 1.5},
                "probability": 0.25
            }
        },
        # VIX 반응
        "vix_reaction": {
            "before": "Usually rises 5-15% in week before",
            "after": "Typically drops 10-20% post-announcement"
        },
        # 최근 백테스트 결과 (우리 시스템)
        "backtest_2024": {
            "avg_t1": 0.25,
            "avg_t5": 1.21,
            "win_rate_t5": 0.81
        }
    },

    "cpi": {
        "name": "CPI Release",
        "pre_event": {
            "t_minus_5_to_1": {
                "avg_return": 0.10,
                "std": 0.7,
                "win_rate": 0.52,
                "pattern": "Range-bound, waiting mode"
            },
            "t_minus_1": {
                "avg_return": -0.05,
                "std": 0.6,
                "pattern": "Slight risk-off ahead of data"
            }
        },
        "post_event": {
            "hot_surprise": {  # CPI > Expected (bad for stocks)
                "t_plus_1": {"avg": -1.0, "std": 0.9},
                "t_plus_5": {"avg": -0.8, "std": 1.2},
                "probability": 0.20
            },
            "slightly_hot": {
                "t_plus_1": {"avg": -0.3, "std": 0.5},
                "t_plus_5": {"avg": 0.0, "std": 0.8},
                "probability": 0.25
            },
            "inline": {
                "t_plus_1": {"avg": 0.3, "std": 0.4},
                "t_plus_5": {"avg": 0.2, "std": 0.6},
                "probability": 0.30
            },
            "cool_surprise": {  # CPI < Expected (good for stocks)
                "t_plus_1": {"avg": 1.2, "std": 0.7},
                "t_plus_5": {"avg": 1.0, "std": 1.0},
                "probability": 0.25
            }
        },
        "vix_reaction": {
            "before": "Elevated uncertainty",
            "after": "Quick resolution, VIX drops if inline/cool"
        },
        "backtest_2024": {
            "avg_t1": 0.35,
            "avg_t5": 0.17,
            "win_rate_t5": 0.67
        }
    },

    "nfp": {
        "name": "Non-Farm Payrolls",
        "pre_event": {
            "t_minus_5_to_1": {
                "avg_return": 0.08,
                "std": 0.6,
                "win_rate": 0.50,
                "pattern": "Neutral, some pre-positioning"
            },
            "t_minus_1": {
                "avg_return": 0.0,
                "std": 0.5,
                "pattern": "Flat ahead of Friday release"
            }
        },
        "post_event": {
            "strong_jobs": {  # High NFP (good economy, but rate hike fear)
                "t_plus_1": {"avg": -0.2, "std": 0.7},
                "t_plus_5": {"avg": 0.3, "std": 1.0},
                "probability": 0.30
            },
            "goldilocks": {  # Moderate NFP (just right)
                "t_plus_1": {"avg": 0.5, "std": 0.5},
                "t_plus_5": {"avg": 0.8, "std": 0.8},
                "probability": 0.40
            },
            "weak_jobs": {  # Low NFP (recession fear)
                "t_plus_1": {"avg": -0.5, "std": 0.8},
                "t_plus_5": {"avg": 0.2, "std": 1.2},
                "probability": 0.30
            }
        },
        "vix_reaction": {
            "before": "Slight increase",
            "after": "Quick reversal typical"
        },
        "backtest_2024": {
            "avg_t1": -0.03,
            "avg_t5": 0.98,
            "win_rate_t5": 0.62
        }
    }
}

# 현재 시장 상태 기준값
MARKET_STATE_THRESHOLDS = {
    "vix": {
        "low": 12,
        "normal": 16,
        "elevated": 20,
        "high": 25,
        "extreme": 30
    },
    "rsi": {
        "oversold": 30,
        "neutral_low": 45,
        "neutral_high": 55,
        "overbought": 70
    },
    "trend": {
        "strong_up": 2.0,  # % above 20-day MA
        "up": 0.5,
        "neutral": -0.5,
        "down": -2.0
    }
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MarketState:
    """현재 시장 상태"""
    timestamp: str
    spy_price: float
    spy_change_1d: float
    spy_change_5d: float
    spy_vs_ma20: float  # % above/below 20-day MA
    vix_level: float
    vix_percentile: float  # 0-100
    rsi_14: float
    trend: str  # "strong_up", "up", "neutral", "down", "strong_down"
    volatility_regime: str  # "low", "normal", "elevated", "high"


@dataclass
class ScenarioPrediction:
    """시나리오별 예측"""
    scenario_name: str
    probability: float
    t_plus_1_return: float
    t_plus_1_range: Tuple[float, float]  # (low, high)
    t_plus_5_return: float
    t_plus_5_range: Tuple[float, float]
    description: str


@dataclass
class EventPrediction:
    """이벤트 예측"""
    event_type: str
    event_name: str
    event_date: str
    days_until: int

    # 현재 상태
    current_price: float
    market_state: MarketState

    # 이벤트 전 예측
    pre_event_forecast: Dict[str, Any]

    # 이벤트 후 시나리오
    scenarios: List[ScenarioPrediction]

    # 가중 평균 예측
    weighted_t1_return: float
    weighted_t5_return: float

    # 권고
    recommendation: str
    confidence: float


# ============================================================================
# Event Predictor
# ============================================================================

class EventPredictor:
    """이벤트 기반 예측기"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.patterns = HISTORICAL_PATTERNS
        self._cache: Dict[str, pd.DataFrame] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventPredictor] {msg}")

    def _get_prices(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """가격 데이터 조회"""
        if ticker not in self._cache:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            self._cache[ticker] = df
        return self._cache[ticker]

    def get_market_state(self) -> MarketState:
        """현재 시장 상태 평가"""
        self._log("Evaluating current market state...")

        # SPY 데이터
        spy = self._get_prices("SPY")
        vix = self._get_prices("^VIX")

        if spy.empty:
            raise ValueError("Could not fetch SPY data")

        # 가격 정보
        spy_price = spy['Close'].iloc[-1]
        spy_change_1d = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2] - 1) * 100
        spy_change_5d = (spy['Close'].iloc[-1] / spy['Close'].iloc[-6] - 1) * 100 if len(spy) > 5 else 0

        # 20일 이동평균 대비
        ma_20 = spy['Close'].rolling(20).mean().iloc[-1]
        spy_vs_ma20 = (spy_price / ma_20 - 1) * 100

        # VIX
        vix_level = vix['Close'].iloc[-1] if not vix.empty else 15
        vix_1y = vix['Close'].iloc[-252:] if len(vix) > 252 else vix['Close']
        vix_percentile = (vix_level <= vix_1y).sum() / len(vix_1y) * 100

        # RSI
        delta = spy['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

        # 트렌드 판단
        if spy_vs_ma20 > 2.0:
            trend = "strong_up"
        elif spy_vs_ma20 > 0.5:
            trend = "up"
        elif spy_vs_ma20 > -0.5:
            trend = "neutral"
        elif spy_vs_ma20 > -2.0:
            trend = "down"
        else:
            trend = "strong_down"

        # 변동성 레짐
        if vix_level < 12:
            vol_regime = "low"
        elif vix_level < 18:
            vol_regime = "normal"
        elif vix_level < 25:
            vol_regime = "elevated"
        else:
            vol_regime = "high"

        return MarketState(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            spy_price=round(spy_price, 2),
            spy_change_1d=round(spy_change_1d, 2),
            spy_change_5d=round(spy_change_5d, 2),
            spy_vs_ma20=round(spy_vs_ma20, 2),
            vix_level=round(vix_level, 2),
            vix_percentile=round(vix_percentile, 1),
            rsi_14=round(rsi_14, 1),
            trend=trend,
            volatility_regime=vol_regime
        )

    def _adjust_for_market_state(
        self,
        base_return: float,
        market_state: MarketState,
        event_type: str
    ) -> float:
        """현재 시장 상태에 따른 예측 조정"""
        adjusted = base_return

        # VIX 레벨에 따른 조정
        if market_state.volatility_regime == "high":
            # 높은 변동성 → 평균 회귀 경향
            adjusted *= 0.7
        elif market_state.volatility_regime == "low":
            # 낮은 변동성 → 콤플레이슨시 위험
            adjusted *= 1.1

        # 추세에 따른 조정
        if market_state.trend == "strong_up" and base_return > 0:
            adjusted *= 1.15  # 추세 추종
        elif market_state.trend == "strong_down" and base_return < 0:
            adjusted *= 1.15

        # RSI 과매수/과매도
        if market_state.rsi_14 > 70 and base_return > 0:
            adjusted *= 0.8  # 과매수에서 상승 제한
        elif market_state.rsi_14 < 30 and base_return < 0:
            adjusted *= 0.8  # 과매도에서 하락 제한

        return round(adjusted, 2)

    def predict_event(
        self,
        event_type: str,
        event_date: str,
        market_state: MarketState = None
    ) -> EventPrediction:
        """단일 이벤트 예측"""
        if event_type not in self.patterns:
            raise ValueError(f"Unknown event type: {event_type}")

        pattern = self.patterns[event_type]
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        days_until = (event_dt - datetime.now()).days

        if market_state is None:
            market_state = self.get_market_state()

        # 이벤트 전 예측
        pre_event = pattern["pre_event"]
        pre_forecast = {
            "t_minus_5_to_1": {
                "expected_return": self._adjust_for_market_state(
                    pre_event["t_minus_5_to_1"]["avg_return"],
                    market_state,
                    event_type
                ),
                "range": (
                    pre_event["t_minus_5_to_1"]["avg_return"] - pre_event["t_minus_5_to_1"]["std"],
                    pre_event["t_minus_5_to_1"]["avg_return"] + pre_event["t_minus_5_to_1"]["std"]
                ),
                "pattern": pre_event["t_minus_5_to_1"]["pattern"]
            },
            "t_minus_1": {
                "expected_return": pre_event["t_minus_1"]["avg_return"],
                "pattern": pre_event["t_minus_1"]["pattern"]
            }
        }

        # 이벤트 후 시나리오
        scenarios = []
        weighted_t1 = 0
        weighted_t5 = 0

        for scenario_name, scenario_data in pattern["post_event"].items():
            prob = scenario_data["probability"]

            # 시장 상태에 따른 조정
            t1_adj = self._adjust_for_market_state(
                scenario_data["t_plus_1"]["avg"],
                market_state,
                event_type
            )
            t5_adj = self._adjust_for_market_state(
                scenario_data["t_plus_5"]["avg"],
                market_state,
                event_type
            )

            t1_std = scenario_data["t_plus_1"]["std"]
            t5_std = scenario_data["t_plus_5"]["std"]

            # 시나리오 설명
            if "surprise" in scenario_name or "hot" in scenario_name or "strong" in scenario_name:
                desc = "High impact scenario"
            elif "inline" in scenario_name or "goldilocks" in scenario_name:
                desc = "Base case scenario"
            else:
                desc = "Alternative scenario"

            scenarios.append(ScenarioPrediction(
                scenario_name=scenario_name,
                probability=prob,
                t_plus_1_return=t1_adj,
                t_plus_1_range=(round(t1_adj - t1_std, 2), round(t1_adj + t1_std, 2)),
                t_plus_5_return=t5_adj,
                t_plus_5_range=(round(t5_adj - t5_std, 2), round(t5_adj + t5_std, 2)),
                description=desc
            ))

            weighted_t1 += t1_adj * prob
            weighted_t5 += t5_adj * prob

        # 권고 생성
        if weighted_t5 > 0.5 and market_state.trend in ["up", "strong_up"]:
            recommendation = "LONG bias - Trend and event favor upside"
        elif weighted_t5 < -0.5 and market_state.trend in ["down", "strong_down"]:
            recommendation = "SHORT bias - Trend and event favor downside"
        elif abs(weighted_t5) < 0.3:
            recommendation = "NEUTRAL - Wait for event resolution"
        else:
            recommendation = "CAUTIOUS - Mixed signals, reduce position size"

        # 신뢰도 (백테스트 승률 기반)
        confidence = pattern.get("backtest_2024", {}).get("win_rate_t5", 0.5)

        return EventPrediction(
            event_type=event_type,
            event_name=pattern["name"],
            event_date=event_date,
            days_until=days_until,
            current_price=market_state.spy_price,
            market_state=market_state,
            pre_event_forecast=pre_forecast,
            scenarios=scenarios,
            weighted_t1_return=round(weighted_t1, 2),
            weighted_t5_return=round(weighted_t5, 2),
            recommendation=recommendation,
            confidence=confidence
        )

    def predict_upcoming_events(self) -> List[EventPrediction]:
        """다가오는 이벤트들 예측"""
        self._log("Predicting upcoming events...")

        # 다가오는 이벤트 목록
        from lib.event_framework import CalendarEventManager

        calendar = CalendarEventManager()
        upcoming = calendar.get_upcoming_events(days_ahead=30)

        # 시장 상태 (한 번만 조회)
        market_state = self.get_market_state()

        predictions = []
        seen_types = set()

        for event in upcoming:
            event_type = event.event_type.value
            if event_type in self.patterns and event_type not in seen_types:
                seen_types.add(event_type)
                pred = self.predict_event(
                    event_type=event_type,
                    event_date=event.timestamp.strftime("%Y-%m-%d"),
                    market_state=market_state
                )
                predictions.append(pred)

        return predictions

    def generate_price_targets(
        self,
        prediction: EventPrediction
    ) -> Dict[str, Dict[str, float]]:
        """가격 목표 생성"""
        current = prediction.current_price

        targets = {
            "pre_event": {
                "t_minus_1_expected": round(current * (1 + prediction.pre_event_forecast["t_minus_5_to_1"]["expected_return"] / 100), 2),
                "t_minus_1_range": (
                    round(current * (1 + prediction.pre_event_forecast["t_minus_5_to_1"]["range"][0] / 100), 2),
                    round(current * (1 + prediction.pre_event_forecast["t_minus_5_to_1"]["range"][1] / 100), 2)
                )
            },
            "post_event": {}
        }

        for scenario in prediction.scenarios:
            targets["post_event"][scenario.scenario_name] = {
                "t_plus_1": round(current * (1 + scenario.t_plus_1_return / 100), 2),
                "t_plus_5": round(current * (1 + scenario.t_plus_5_return / 100), 2),
                "probability": scenario.probability
            }

        # 가중 평균 목표
        targets["weighted"] = {
            "t_plus_1": round(current * (1 + prediction.weighted_t1_return / 100), 2),
            "t_plus_5": round(current * (1 + prediction.weighted_t5_return / 100), 2)
        }

        return targets

    def print_predictions(self, predictions: List[EventPrediction]):
        """예측 결과 출력"""
        print("\n" + "=" * 80)
        print("EIMAS EVENT PREDICTIONS")
        print("=" * 80)

        if not predictions:
            print("\nNo upcoming events to predict.")
            return

        # 시장 상태 출력
        ms = predictions[0].market_state
        print(f"\n📊 Current Market State ({ms.timestamp})")
        print("-" * 60)
        print(f"  SPY: ${ms.spy_price} ({ms.spy_change_1d:+.2f}% 1D, {ms.spy_change_5d:+.2f}% 5D)")
        print(f"  vs 20-day MA: {ms.spy_vs_ma20:+.2f}%")
        print(f"  VIX: {ms.vix_level} (Percentile: {ms.vix_percentile:.0f}%)")
        print(f"  RSI(14): {ms.rsi_14:.0f}")
        print(f"  Trend: {ms.trend.upper()} | Volatility: {ms.volatility_regime.upper()}")

        for pred in predictions:
            print("\n" + "=" * 80)
            print(f"📅 {pred.event_name}")
            print(f"   Date: {pred.event_date} (D{pred.days_until:+d})")
            print("=" * 80)

            # 이벤트 전 예측
            pre = pred.pre_event_forecast
            print(f"\n🔮 PRE-EVENT FORECAST (T-5 to T-1)")
            print("-" * 50)
            print(f"  Expected Return: {pre['t_minus_5_to_1']['expected_return']:+.2f}%")
            print(f"  Range: {pre['t_minus_5_to_1']['range'][0]:+.1f}% to {pre['t_minus_5_to_1']['range'][1]:+.1f}%")
            print(f"  Pattern: {pre['t_minus_5_to_1']['pattern']}")

            # 이벤트 후 시나리오
            print(f"\n🎯 POST-EVENT SCENARIOS")
            print("-" * 50)
            print(f"{'Scenario':<20} {'Prob':>6} {'T+1':>10} {'T+5':>10}")
            print("-" * 50)

            for s in pred.scenarios:
                print(f"{s.scenario_name:<20} {s.probability*100:>5.0f}% {s.t_plus_1_return:>+9.2f}% {s.t_plus_5_return:>+9.2f}%")

            print("-" * 50)
            print(f"{'WEIGHTED AVERAGE':<20} {'':>6} {pred.weighted_t1_return:>+9.2f}% {pred.weighted_t5_return:>+9.2f}%")

            # 가격 목표
            targets = self.generate_price_targets(pred)
            print(f"\n💰 PRICE TARGETS (Current: ${pred.current_price})")
            print("-" * 50)
            print(f"  Pre-Event (T-1): ${targets['pre_event']['t_minus_1_expected']}")
            print(f"  Post-Event Weighted:")
            print(f"    T+1: ${targets['weighted']['t_plus_1']}")
            print(f"    T+5: ${targets['weighted']['t_plus_5']}")

            # 권고
            print(f"\n📋 RECOMMENDATION")
            print("-" * 50)
            print(f"  {pred.recommendation}")
            print(f"  Confidence: {pred.confidence*100:.0f}% (based on historical win rate)")

        print("\n" + "=" * 80)
        print("⚠️  DISCLAIMER: Predictions based on historical patterns.")
        print("   Actual outcomes may differ. Not financial advice.")
        print("=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    predictor = EventPredictor(verbose=True)

    # 다가오는 이벤트 예측
    predictions = predictor.predict_upcoming_events()
    predictor.print_predictions(predictions)
