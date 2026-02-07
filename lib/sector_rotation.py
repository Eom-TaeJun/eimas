#!/usr/bin/env python3
"""
EIMAS Sector Rotation Model
============================
경기 사이클 기반 섹터 로테이션 전략

주요 기능:
1. 섹터 모멘텀 계산
2. 경기 사이클 매핑
3. 상대 강도 분석
4. 최적 섹터 비중 도출

Usage:
    from lib.sector_rotation import SectorRotationModel

    model = SectorRotationModel()
    result = model.analyze()
    print(result.recommended_sectors)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

# SPDR Sector ETFs
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}

# 경기 사이클별 선호 섹터
CYCLE_SECTOR_MAP = {
    'early_recovery': ['XLY', 'XLF', 'XLI', 'XLB'],     # 초기 회복
    'mid_expansion': ['XLK', 'XLI', 'XLB', 'XLE'],      # 중반 확장
    'late_expansion': ['XLE', 'XLB', 'XLV', 'XLP'],     # 후반 확장
    'recession': ['XLU', 'XLP', 'XLV', 'XLRE'],         # 경기 침체
}

# 모멘텀 기간
MOMENTUM_PERIODS = {
    'short': 21,     # 1개월
    'medium': 63,    # 3개월
    'long': 252,     # 1년
}


# ============================================================================
# Data Classes
# ============================================================================

class EconomicCycle(str, Enum):
    """경기 사이클 단계"""
    EARLY_RECOVERY = "early_recovery"
    MID_EXPANSION = "mid_expansion"
    LATE_EXPANSION = "late_expansion"
    RECESSION = "recession"


class SectorSignal(str, Enum):
    """섹터 시그널"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class SectorStats:
    """섹터 통계"""
    ticker: str
    name: str
    momentum_1m: float
    momentum_3m: float
    momentum_12m: float
    relative_strength: float
    volatility: float
    rank: int
    signal: SectorSignal


@dataclass
class CycleDetection:
    """경기 사이클 감지 결과"""
    current_cycle: EconomicCycle
    confidence: float
    indicators: Dict[str, float]
    preferred_sectors: List[str]


@dataclass
class RotationSignal:
    """로테이션 시그널"""
    overweight: List[str]    # 비중 확대
    underweight: List[str]   # 비중 축소
    neutral: List[str]       # 유지
    changes: Dict[str, float]  # 비중 변화량


@dataclass
class SectorRotationResult:
    """섹터 로테이션 분석 결과"""
    timestamp: datetime
    cycle: CycleDetection
    sector_stats: List[SectorStats]
    rotation_signal: RotationSignal
    recommended_weights: Dict[str, float]
    top_sectors: List[str]
    bottom_sectors: List[str]
    summary: str


# ============================================================================
# Sector Rotation Model
# ============================================================================

class SectorRotationModel:
    """섹터 로테이션 모델"""

    def __init__(
        self,
        sectors: Dict[str, str] = None,
        benchmark: str = 'SPY',
    ):
        self.sectors = sectors or SECTOR_ETFS
        self.benchmark = benchmark
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def fetch_data(self, period: str = "2y") -> bool:
        """데이터 수집"""
        try:
            tickers = list(self.sectors.keys()) + [self.benchmark]
            df = yf.download(tickers, period=period, progress=False)['Close']

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self.data = df.dropna()
            self.returns = self.data.pct_change().dropna()

            print(f"Loaded {len(self.sectors)} sectors, {len(self.data)} days")
            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def calculate_momentum(self, period: int) -> pd.Series:
        """모멘텀 계산 (기간 수익률)"""
        if self.data is None:
            self.fetch_data()
        return (self.data.iloc[-1] / self.data.iloc[-period] - 1) * 100

    def calculate_relative_strength(self) -> pd.Series:
        """벤치마크 대비 상대 강도"""
        if self.data is None:
            self.fetch_data()

        sector_tickers = list(self.sectors.keys())

        # 3개월 상대 수익률
        sector_returns = self.calculate_momentum(63)[sector_tickers]
        benchmark_return = self.calculate_momentum(63)[self.benchmark]

        return sector_returns - benchmark_return

    def detect_cycle(self) -> CycleDetection:
        """경기 사이클 감지"""
        if self.data is None:
            self.fetch_data()

        # 여러 지표로 사이클 판단
        indicators = {}

        # 1. 방어주 vs 경기민감주 비율
        defensive = ['XLU', 'XLP', 'XLV']
        cyclical = ['XLY', 'XLI', 'XLB']

        def_mom = np.mean([self.calculate_momentum(63).get(s, 0) for s in defensive if s in self.data.columns])
        cyc_mom = np.mean([self.calculate_momentum(63).get(s, 0) for s in cyclical if s in self.data.columns])
        indicators['defensive_vs_cyclical'] = def_mom - cyc_mom

        # 2. 금융 섹터 모멘텀 (금리 방향 proxy)
        if 'XLF' in self.data.columns:
            indicators['financials_momentum'] = self.calculate_momentum(63).get('XLF', 0)

        # 3. 에너지 섹터 모멘텀 (인플레이션 proxy)
        if 'XLE' in self.data.columns:
            indicators['energy_momentum'] = self.calculate_momentum(63).get('XLE', 0)

        # 4. 기술 섹터 모멘텀 (성장 proxy)
        if 'XLK' in self.data.columns:
            indicators['tech_momentum'] = self.calculate_momentum(63).get('XLK', 0)

        # 사이클 판단 로직
        def_cyc_ratio = indicators['defensive_vs_cyclical']
        fin_mom = indicators.get('financials_momentum', 0)
        tech_mom = indicators.get('tech_momentum', 0)

        if def_cyc_ratio > 10:
            # 방어주 강세 → 경기 둔화/침체
            cycle = EconomicCycle.RECESSION
            confidence = min(0.9, 0.5 + def_cyc_ratio / 30)
        elif fin_mom > 15 and tech_mom > 10:
            # 금융/기술 강세 → 초기 회복
            cycle = EconomicCycle.EARLY_RECOVERY
            confidence = min(0.9, 0.5 + (fin_mom + tech_mom) / 60)
        elif indicators.get('energy_momentum', 0) > 20:
            # 에너지 강세 → 후반 확장
            cycle = EconomicCycle.LATE_EXPANSION
            confidence = 0.7
        else:
            # 기본: 중반 확장
            cycle = EconomicCycle.MID_EXPANSION
            confidence = 0.6

        preferred = CYCLE_SECTOR_MAP.get(cycle.value, [])

        return CycleDetection(
            current_cycle=cycle,
            confidence=confidence,
            indicators=indicators,
            preferred_sectors=preferred,
        )

    def analyze_sectors(self) -> List[SectorStats]:
        """모든 섹터 분석"""
        if self.data is None:
            self.fetch_data()

        mom_1m = self.calculate_momentum(21)
        mom_3m = self.calculate_momentum(63)
        mom_12m = self.calculate_momentum(252)
        rel_strength = self.calculate_relative_strength()
        volatility = self.returns.std() * np.sqrt(252) * 100

        stats = []
        for ticker, name in self.sectors.items():
            if ticker not in self.data.columns:
                continue

            # 종합 점수 계산
            score = (
                mom_1m.get(ticker, 0) * 0.2 +
                mom_3m.get(ticker, 0) * 0.3 +
                mom_12m.get(ticker, 0) * 0.3 +
                rel_strength.get(ticker, 0) * 0.2
            )

            # 시그널 결정
            if score > 20:
                signal = SectorSignal.STRONG_BUY
            elif score > 10:
                signal = SectorSignal.BUY
            elif score > -10:
                signal = SectorSignal.HOLD
            elif score > -20:
                signal = SectorSignal.SELL
            else:
                signal = SectorSignal.STRONG_SELL

            stats.append(SectorStats(
                ticker=ticker,
                name=name,
                momentum_1m=float(mom_1m.get(ticker, 0)),
                momentum_3m=float(mom_3m.get(ticker, 0)),
                momentum_12m=float(mom_12m.get(ticker, 0)),
                relative_strength=float(rel_strength.get(ticker, 0)),
                volatility=float(volatility.get(ticker, 0)),
                rank=0,
                signal=signal,
            ))

        # 순위 매기기 (종합 점수 기준)
        stats.sort(key=lambda x: -(x.momentum_3m * 0.5 + x.relative_strength * 0.5))
        for i, s in enumerate(stats):
            s.rank = i + 1

        return stats

    def generate_rotation_signal(
        self,
        sector_stats: List[SectorStats],
        cycle: CycleDetection,
    ) -> RotationSignal:
        """로테이션 시그널 생성"""
        overweight = []
        underweight = []
        neutral = []
        changes = {}

        n_sectors = len(sector_stats)
        top_n = max(3, n_sectors // 3)

        for s in sector_stats:
            # 상위 1/3 + 경기 사이클 선호 → 비중 확대
            is_preferred = s.ticker in cycle.preferred_sectors
            is_top = s.rank <= top_n
            is_bottom = s.rank > n_sectors - top_n

            if is_top and is_preferred:
                overweight.append(s.ticker)
                changes[s.ticker] = 0.05  # +5%
            elif is_top or is_preferred:
                overweight.append(s.ticker)
                changes[s.ticker] = 0.03  # +3%
            elif is_bottom:
                underweight.append(s.ticker)
                changes[s.ticker] = -0.03  # -3%
            else:
                neutral.append(s.ticker)
                changes[s.ticker] = 0.0

        return RotationSignal(
            overweight=overweight,
            underweight=underweight,
            neutral=neutral,
            changes=changes,
        )

    def calculate_recommended_weights(
        self,
        sector_stats: List[SectorStats],
        rotation_signal: RotationSignal,
    ) -> Dict[str, float]:
        """추천 섹터 비중 계산"""
        n = len(sector_stats)
        base_weight = 1.0 / n

        weights = {}
        for s in sector_stats:
            adjustment = rotation_signal.changes.get(s.ticker, 0)
            weights[s.ticker] = max(0.02, base_weight + adjustment)

        # 정규화
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def analyze(self) -> SectorRotationResult:
        """전체 분석 실행"""
        print("\n" + "=" * 60)
        print("EIMAS Sector Rotation Analysis")
        print("=" * 60)

        if self.data is None:
            self.fetch_data()

        # 경기 사이클 감지
        cycle = self.detect_cycle()
        print(f"\nEconomic Cycle: {cycle.current_cycle.value} ({cycle.confidence:.0%})")

        # 섹터 분석
        sector_stats = self.analyze_sectors()

        # 로테이션 시그널
        rotation = self.generate_rotation_signal(sector_stats, cycle)

        # 추천 비중
        weights = self.calculate_recommended_weights(sector_stats, rotation)

        # 상위/하위 섹터
        top_sectors = [s.ticker for s in sector_stats[:3]]
        bottom_sectors = [s.ticker for s in sector_stats[-3:]]

        # 요약
        summary = self._generate_summary(cycle, sector_stats, rotation)

        return SectorRotationResult(
            timestamp=datetime.now(),
            cycle=cycle,
            sector_stats=sector_stats,
            rotation_signal=rotation,
            recommended_weights=weights,
            top_sectors=top_sectors,
            bottom_sectors=bottom_sectors,
            summary=summary,
        )

    def _generate_summary(
        self,
        cycle: CycleDetection,
        stats: List[SectorStats],
        rotation: RotationSignal,
    ) -> str:
        """요약 생성"""
        lines = [
            f"Economic Cycle: {cycle.current_cycle.value.replace('_', ' ').title()}",
            f"Confidence: {cycle.confidence:.0%}",
            "",
            "Top Sectors (Overweight):",
        ]

        for ticker in rotation.overweight[:3]:
            s = next((x for x in stats if x.ticker == ticker), None)
            if s:
                lines.append(f"  {ticker} ({s.name}): 3M Mom {s.momentum_3m:.1f}%")

        lines.append("")
        lines.append("Bottom Sectors (Underweight):")
        for ticker in rotation.underweight[:3]:
            s = next((x for x in stats if x.ticker == ticker), None)
            if s:
                lines.append(f"  {ticker} ({s.name}): 3M Mom {s.momentum_3m:.1f}%")

        return "\n".join(lines)

    def print_result(self, result: SectorRotationResult):
        """결과 출력"""
        print("\n" + result.summary)

        print("\n" + "-" * 40)
        print("Sector Rankings:")
        print(f"{'Rank':<5} {'Ticker':<6} {'Name':<25} {'3M Mom':<10} {'RS':<10} {'Signal'}")
        print("-" * 70)

        for s in result.sector_stats:
            print(f"{s.rank:<5} {s.ticker:<6} {s.name:<25} {s.momentum_3m:>7.1f}% {s.relative_strength:>7.1f}% {s.signal.value}")

        print("\n" + "-" * 40)
        print("Recommended Weights:")
        for ticker, weight in sorted(result.recommended_weights.items(), key=lambda x: -x[1]):
            name = self.sectors.get(ticker, '')
            print(f"  {ticker} ({name}): {weight:.1%}")

        print("=" * 60)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_sector_analysis() -> SectorRotationResult:
    """빠른 섹터 분석"""
    model = SectorRotationModel()
    return model.analyze()


def get_top_sectors(n: int = 3) -> List[str]:
    """상위 N개 섹터 반환"""
    model = SectorRotationModel()
    result = model.analyze()
    return result.top_sectors[:n]


def sector_momentum_ranking() -> pd.DataFrame:
    """섹터 모멘텀 순위"""
    model = SectorRotationModel()
    model.fetch_data()

    stats = model.analyze_sectors()
    data = [{
        'Ticker': s.ticker,
        'Sector': s.name,
        '1M Mom': s.momentum_1m,
        '3M Mom': s.momentum_3m,
        '12M Mom': s.momentum_12m,
        'RS': s.relative_strength,
        'Rank': s.rank,
    } for s in stats]

    return pd.DataFrame(data).set_index('Ticker')


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    model = SectorRotationModel()
    result = model.analyze()
    model.print_result(result)

    print("\n\nSector Momentum Ranking:")
    print(sector_momentum_ranking().round(1))
