#!/usr/bin/env python3
"""
ARK ETF Holdings Analyzer
==========================
ARK ETF 보유 종목 변화 분석을 통한 선행 지표 생성

ETF_HOLDINGS_ANALYSIS.md 기반 구현:
- ARK ETF 일간 holdings 수집 (arkfunds.io API)
- 비중 변화 분석 (일간/주간)
- 종목별/섹터별 신호 생성
- Signal-Action Framework 연동

경제학적 배경:
- 액티브 ETF 비중 변화 = 펀드 매니저의 의도적 판단
- 다수 ETF가 같은 방향 → 강한 확신 신호
- 패시브 ETF와 달리 "정보 가치" 있음
"""

import json
import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from core.signal_action import (
    EnhancedSignal,
    PositionDirection,
)
from core.database import DatabaseManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# Constants
# ============================================================================

# ARK ETF 목록
ARK_ETFS = {
    "ARKK": "ARK Innovation ETF",
    "ARKW": "ARK Next Generation Internet ETF",
    "ARKG": "ARK Genomic Revolution ETF",
    "ARKF": "ARK Fintech Innovation ETF",
    "ARKQ": "ARK Autonomous Tech & Robotics ETF",
    "ARKX": "ARK Space Exploration & Innovation ETF",
}

# 섹터 분류 (종목별)
SECTOR_MAPPING = {
    # Technology
    "TSLA": "EV/Auto", "NVDA": "Semiconductor", "AMD": "Semiconductor",
    "TSM": "Semiconductor", "GOOG": "Tech/Internet", "AMZN": "Tech/Internet",
    "META": "Tech/Internet", "BIDU": "Tech/Internet", "SHOP": "E-commerce",
    "PLTR": "Software/AI", "RBLX": "Gaming/Metaverse", "ROKU": "Streaming",
    "TTD": "AdTech", "DKNG": "Gaming", "ABNB": "Travel/Sharing",

    # Fintech/Crypto
    "COIN": "Crypto", "HOOD": "Fintech", "XYZ": "Fintech", "SOFI": "Fintech",
    "CRCL": "Crypto", "BLSH": "Crypto", "BMNR": "Crypto",

    # Biotech/Healthcare
    "CRSP": "Biotech/Gene", "BEAM": "Biotech/Gene", "NTLA": "Biotech/Gene",
    "ILMN": "Genomics", "TXG": "Genomics", "TWST": "Genomics",
    "VCYT": "Diagnostics", "NTRA": "Diagnostics", "PACB": "Genomics",
    "RXRX": "Biotech/AI",

    # Industrial/Defense
    "TER": "Semiconductor Equipment", "DE": "Industrial/Agri",
    "KTOS": "Defense", "BWXT": "Defense/Nuclear", "ACHR": "Aerospace/eVTOL",

    # AI/Software
    "TEM": "AI/Healthcare", "PD": "Software",

    # Other
    "WGS": "Diagnostics", "CERS": "Biotech",
}

# API 설정
ARKFUNDS_API_BASE = "https://arkfunds.io/api/v2/etf/holdings"


# ============================================================================
# Data Classes
# ============================================================================

class SignalType(str, Enum):
    """ARK 신호 유형"""
    WEIGHT_INCREASE = "weight_increase"    # 비중 증가
    WEIGHT_DECREASE = "weight_decrease"    # 비중 감소
    NEW_POSITION = "new_position"          # 신규 편입
    EXIT_POSITION = "exit_position"        # 완전 매도
    CONSENSUS_BUY = "consensus_buy"        # 다수 ETF 매수
    CONSENSUS_SELL = "consensus_sell"      # 다수 ETF 매도


@dataclass
class HoldingData:
    """개별 보유 종목 데이터"""
    fund: str                    # ETF 심볼
    date: str                    # 날짜
    ticker: str                  # 종목 티커
    company: str                 # 회사명
    shares: int                  # 주식 수
    market_value: float          # 시장 가치
    weight: float                # 비중 (%)
    weight_rank: int             # 비중 순위
    sector: str = ""             # 섹터

    def __post_init__(self):
        if not self.sector and self.ticker:
            self.sector = SECTOR_MAPPING.get(self.ticker, "Other")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WeightChange:
    """비중 변화 데이터"""
    ticker: str
    company: str
    sector: str

    # 현재 상태
    current_weight: float        # 현재 비중
    current_shares: int          # 현재 주식 수
    etf_count: int               # 보유 ETF 수

    # 변화량
    weight_change_1d: float      # 1일 비중 변화 (%p)
    weight_change_5d: float      # 5일 비중 변화 (%p)
    shares_change_1d: int        # 1일 주식 수 변화
    shares_change_5d: int        # 5일 주식 수 변화

    # ETF별 방향
    etfs_increasing: List[str] = field(default_factory=list)   # 비중 늘린 ETF
    etfs_decreasing: List[str] = field(default_factory=list)   # 비중 줄인 ETF

    # 신호
    signal_type: Optional[SignalType] = None
    signal_strength: float = 0.0  # 0-1

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.signal_type:
            data['signal_type'] = self.signal_type.value
        return data


@dataclass
class SectorSummary:
    """섹터별 요약"""
    sector: str
    total_weight: float          # 총 비중
    stock_count: int             # 종목 수
    weight_change_1d: float      # 1일 비중 변화
    weight_change_5d: float      # 5일 비중 변화
    top_holdings: List[str] = field(default_factory=list)  # 상위 종목

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ARKAnalysisResult:
    """ARK 분석 결과"""
    timestamp: str
    etfs_analyzed: List[str]
    total_holdings: int

    # 주요 변화
    top_increases: List[WeightChange]
    top_decreases: List[WeightChange]
    new_positions: List[str]
    exited_positions: List[str]

    # 섹터 분석
    sector_summary: List[SectorSummary]

    # 컨센서스
    consensus_buys: List[str]    # 다수 ETF가 비중 증가
    consensus_sells: List[str]   # 다수 ETF가 비중 감소

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'etfs_analyzed': self.etfs_analyzed,
            'total_holdings': self.total_holdings,
            'top_increases': [w.to_dict() for w in self.top_increases],
            'top_decreases': [w.to_dict() for w in self.top_decreases],
            'new_positions': self.new_positions,
            'exited_positions': self.exited_positions,
            'sector_summary': [s.to_dict() for s in self.sector_summary],
            'consensus_buys': self.consensus_buys,
            'consensus_sells': self.consensus_sells,
        }


# ============================================================================
# ARK Holdings Collector
# ============================================================================

class ARKHoldingsCollector:
    """
    ARK ETF Holdings 수집기

    arkfunds.io API를 통해 일간 holdings 데이터 수집
    """

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        if data_dir is None:
            data_dir = PROJECT_ROOT / "data" / "ark_holdings"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_holdings(self, symbol: str, date_from: str = None, date_to: str = None) -> List[HoldingData]:
        """
        특정 ETF의 holdings 데이터 수집

        Args:
            symbol: ETF 심볼 (예: ARKK)
            date_from: 시작 날짜 (YYYY-MM-DD)
            date_to: 종료 날짜 (YYYY-MM-DD)

        Returns:
            HoldingData 리스트
        """
        params = {"symbol": symbol}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        try:
            response = requests.get(ARKFUNDS_API_BASE, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            holdings = []
            for h in data.get("holdings", []):
                if h.get("ticker"):  # ticker가 있는 경우만
                    holding = HoldingData(
                        fund=h.get("fund", symbol),
                        date=h.get("date", ""),
                        ticker=h.get("ticker", ""),
                        company=h.get("company", ""),
                        shares=int(h.get("shares", 0)),
                        market_value=float(h.get("market_value", 0)),
                        weight=float(h.get("weight", 0)),
                        weight_rank=int(h.get("weight_rank", 0)),
                    )
                    holdings.append(holding)

            return holdings

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return []

    def fetch_all_etfs(self) -> Dict[str, List[HoldingData]]:
        """모든 ARK ETF holdings 수집"""
        all_holdings = {}

        for symbol in ARK_ETFS.keys():
            print(f"  Fetching {symbol}...")
            holdings = self.fetch_holdings(symbol)
            if holdings:
                all_holdings[symbol] = holdings
                print(f"    → {len(holdings)} holdings")

        return all_holdings

    def save_snapshot(self, holdings: Dict[str, List[HoldingData]], date: str = None):
        """일간 스냅샷 저장"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        snapshot_dir = self.data_dir / date
        snapshot_dir.mkdir(exist_ok=True)

        for symbol, holding_list in holdings.items():
            filepath = snapshot_dir / f"{symbol}.json"
            data = [h.to_dict() for h in holding_list]

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Saved snapshot to {snapshot_dir}")

    def load_snapshot(self, date: str) -> Dict[str, List[HoldingData]]:
        """저장된 스냅샷 로드"""
        snapshot_dir = self.data_dir / date

        if not snapshot_dir.exists():
            return {}

        holdings = {}
        for filepath in snapshot_dir.glob("*.json"):
            symbol = filepath.stem
            with open(filepath, 'r') as f:
                data = json.load(f)
                holdings[symbol] = [HoldingData(**h) for h in data]

        return holdings

    def get_available_dates(self) -> List[str]:
        """저장된 날짜 목록"""
        dates = []
        for d in self.data_dir.iterdir():
            if d.is_dir() and len(d.name) == 10:  # YYYY-MM-DD
                dates.append(d.name)
        return sorted(dates)


# ============================================================================
# ARK Holdings Analyzer
# ============================================================================

class ARKHoldingsAnalyzer:
    """
    ARK ETF Holdings 변화 분석기
    """

    def __init__(self, collector: ARKHoldingsCollector = None):
        """
        Args:
            collector: ARKHoldingsCollector 인스턴스
        """
        self.collector = collector or ARKHoldingsCollector()
        self.current_holdings: Dict[str, List[HoldingData]] = {}
        self.historical_holdings: Dict[str, Dict[str, List[HoldingData]]] = {}  # {date: {symbol: [holdings]}}

    def load_current_holdings(self) -> Dict[str, List[HoldingData]]:
        """현재 holdings 로드 (API에서)"""
        print("Fetching current ARK holdings...")
        self.current_holdings = self.collector.fetch_all_etfs()
        return self.current_holdings

    def load_historical_holdings(self, days: int = 5) -> Dict[str, Dict[str, List[HoldingData]]]:
        """히스토리 로드 (저장된 데이터에서)"""
        available_dates = self.collector.get_available_dates()

        for date in available_dates[-days:]:
            snapshot = self.collector.load_snapshot(date)
            if snapshot:
                self.historical_holdings[date] = snapshot

        return self.historical_holdings

    def _aggregate_by_ticker(self, holdings_dict: Dict[str, List[HoldingData]]) -> Dict[str, Dict]:
        """종목별로 데이터 집계"""
        ticker_data = {}

        for symbol, holdings in holdings_dict.items():
            for h in holdings:
                if not h.ticker:
                    continue

                if h.ticker not in ticker_data:
                    ticker_data[h.ticker] = {
                        'company': h.company,
                        'sector': h.sector,
                        'total_weight': 0,
                        'total_shares': 0,
                        'etfs': {},  # {etf: weight}
                    }

                ticker_data[h.ticker]['total_weight'] += h.weight
                ticker_data[h.ticker]['total_shares'] += h.shares
                ticker_data[h.ticker]['etfs'][symbol] = h.weight

        return ticker_data

    def calculate_weight_changes(self) -> List[WeightChange]:
        """비중 변화 계산"""
        if not self.current_holdings:
            return []

        # 현재 데이터 집계
        current_data = self._aggregate_by_ticker(self.current_holdings)

        # 히스토리가 있으면 비교
        dates = sorted(self.historical_holdings.keys())

        if dates:
            # 1일 전 데이터
            prev_1d = self._aggregate_by_ticker(self.historical_holdings.get(dates[-1], {})) if dates else {}
            # 5일 전 데이터
            prev_5d = self._aggregate_by_ticker(self.historical_holdings.get(dates[0], {})) if len(dates) >= 5 else prev_1d
        else:
            prev_1d = {}
            prev_5d = {}

        changes = []

        for ticker, data in current_data.items():
            prev_1d_data = prev_1d.get(ticker, {})
            prev_5d_data = prev_5d.get(ticker, {})

            # 비중 변화 계산
            weight_1d = data['total_weight'] - prev_1d_data.get('total_weight', data['total_weight'])
            weight_5d = data['total_weight'] - prev_5d_data.get('total_weight', data['total_weight'])
            shares_1d = data['total_shares'] - prev_1d_data.get('total_shares', data['total_shares'])
            shares_5d = data['total_shares'] - prev_5d_data.get('total_shares', data['total_shares'])

            # ETF별 방향 분석
            etfs_increasing = []
            etfs_decreasing = []

            for etf, weight in data['etfs'].items():
                prev_weight = prev_1d_data.get('etfs', {}).get(etf, 0)
                if weight > prev_weight + 0.1:  # 0.1%p 이상 증가
                    etfs_increasing.append(etf)
                elif weight < prev_weight - 0.1:  # 0.1%p 이상 감소
                    etfs_decreasing.append(etf)

            # 신호 유형 결정
            signal_type = None
            signal_strength = 0.0

            if ticker not in prev_1d:
                signal_type = SignalType.NEW_POSITION
                signal_strength = 0.8
            elif len(etfs_increasing) >= 3:
                signal_type = SignalType.CONSENSUS_BUY
                signal_strength = len(etfs_increasing) / len(ARK_ETFS)
            elif len(etfs_decreasing) >= 3:
                signal_type = SignalType.CONSENSUS_SELL
                signal_strength = len(etfs_decreasing) / len(ARK_ETFS)
            elif weight_1d > 0.5:
                signal_type = SignalType.WEIGHT_INCREASE
                signal_strength = min(weight_1d / 2, 1.0)
            elif weight_1d < -0.5:
                signal_type = SignalType.WEIGHT_DECREASE
                signal_strength = min(abs(weight_1d) / 2, 1.0)

            change = WeightChange(
                ticker=ticker,
                company=data['company'],
                sector=data['sector'],
                current_weight=data['total_weight'],
                current_shares=data['total_shares'],
                etf_count=len(data['etfs']),
                weight_change_1d=round(weight_1d, 2),
                weight_change_5d=round(weight_5d, 2),
                shares_change_1d=shares_1d,
                shares_change_5d=shares_5d,
                etfs_increasing=etfs_increasing,
                etfs_decreasing=etfs_decreasing,
                signal_type=signal_type,
                signal_strength=round(signal_strength, 2),
            )
            changes.append(change)

        # 이탈 종목 확인
        for ticker in prev_1d:
            if ticker not in current_data:
                change = WeightChange(
                    ticker=ticker,
                    company=prev_1d[ticker]['company'],
                    sector=prev_1d[ticker].get('sector', 'Other'),
                    current_weight=0,
                    current_shares=0,
                    etf_count=0,
                    weight_change_1d=-prev_1d[ticker]['total_weight'],
                    weight_change_5d=-prev_1d[ticker]['total_weight'],
                    shares_change_1d=-prev_1d[ticker]['total_shares'],
                    shares_change_5d=-prev_1d[ticker]['total_shares'],
                    signal_type=SignalType.EXIT_POSITION,
                    signal_strength=0.9,
                )
                changes.append(change)

        return changes

    def calculate_sector_summary(self) -> List[SectorSummary]:
        """섹터별 요약"""
        if not self.current_holdings:
            return []

        sector_data = {}

        for symbol, holdings in self.current_holdings.items():
            for h in holdings:
                sector = h.sector or "Other"

                if sector not in sector_data:
                    sector_data[sector] = {
                        'total_weight': 0,
                        'stocks': set(),
                        'holdings': [],
                    }

                sector_data[sector]['total_weight'] += h.weight / len(self.current_holdings)  # 평균
                sector_data[sector]['stocks'].add(h.ticker)
                sector_data[sector]['holdings'].append((h.ticker, h.weight))

        summaries = []
        for sector, data in sector_data.items():
            # 상위 종목
            top = sorted(data['holdings'], key=lambda x: x[1], reverse=True)[:3]
            top_holdings = [t[0] for t in top]

            summaries.append(SectorSummary(
                sector=sector,
                total_weight=round(data['total_weight'], 2),
                stock_count=len(data['stocks']),
                weight_change_1d=0,  # 히스토리 있으면 계산
                weight_change_5d=0,
                top_holdings=top_holdings,
            ))

        # 비중 순 정렬
        summaries.sort(key=lambda x: x.total_weight, reverse=True)
        return summaries

    def run_analysis(self) -> ARKAnalysisResult:
        """전체 분석 실행"""
        # 데이터 로드
        self.load_current_holdings()
        self.load_historical_holdings()

        # 비중 변화 계산
        changes = self.calculate_weight_changes()

        # 정렬
        increases = sorted([c for c in changes if c.weight_change_1d > 0.1],
                          key=lambda x: x.weight_change_1d, reverse=True)
        decreases = sorted([c for c in changes if c.weight_change_1d < -0.1],
                          key=lambda x: x.weight_change_1d)

        # 신규/이탈
        new_positions = [c.ticker for c in changes if c.signal_type == SignalType.NEW_POSITION]
        exited = [c.ticker for c in changes if c.signal_type == SignalType.EXIT_POSITION]

        # 컨센서스
        consensus_buys = [c.ticker for c in changes if c.signal_type == SignalType.CONSENSUS_BUY]
        consensus_sells = [c.ticker for c in changes if c.signal_type == SignalType.CONSENSUS_SELL]

        # 섹터 요약
        sector_summary = self.calculate_sector_summary()

        return ARKAnalysisResult(
            timestamp=datetime.now().isoformat(),
            etfs_analyzed=list(self.current_holdings.keys()),
            total_holdings=sum(len(h) for h in self.current_holdings.values()),
            top_increases=increases[:10],
            top_decreases=decreases[:10],
            new_positions=new_positions,
            exited_positions=exited,
            sector_summary=sector_summary,
            consensus_buys=consensus_buys,
            consensus_sells=consensus_sells,
        )

    def save_to_db(self, result: ARKAnalysisResult, signals: List[EnhancedSignal] = None,
                   db: DatabaseManager = None) -> Dict[str, int]:
        """
        분석 결과를 DB에 저장

        Args:
            result: 분석 결과
            signals: 생성된 신호 (없으면 자동 생성)
            db: DatabaseManager 인스턴스 (없으면 기본값 사용)

        Returns:
            저장 통계 {holdings: N, signals: N, ...}
        """
        if db is None:
            db = DatabaseManager()

        today = datetime.now().strftime("%Y-%m-%d")
        stats = {'holdings': 0, 'weight_changes': 0, 'signals': 0}

        # 1. Holdings 저장
        holdings_list = []
        for symbol, holdings in self.current_holdings.items():
            for h in holdings:
                holdings_list.append({
                    'etf': h.fund,
                    'ticker': h.ticker,
                    'company': h.company,
                    'cusip': '',
                    'shares': h.shares,
                    'market_value': h.market_value,
                    'weight': h.weight,
                })

        if holdings_list:
            stats['holdings'] = db.save_ark_holdings(holdings_list, today)

        # 2. Weight Changes 저장
        changes = self.calculate_weight_changes()
        changes_list = []
        for c in changes:
            if c.weight_change_1d != 0 or c.signal_type:
                # 각 ETF별로 저장
                for etf in c.etfs_increasing + c.etfs_decreasing:
                    changes_list.append({
                        'ticker': c.ticker,
                        'etf': etf,
                        'prev_weight': c.current_weight - c.weight_change_1d,
                        'curr_weight': c.current_weight,
                        'weight_change': c.weight_change_1d,
                        'change_type': c.signal_type.value if c.signal_type else 'CHANGE',
                        'prev_shares': c.current_shares - c.shares_change_1d,
                        'curr_shares': c.current_shares,
                        'share_change': c.shares_change_1d,
                    })

        if changes_list:
            stats['weight_changes'] = db.save_ark_weight_changes(changes_list, today)

        # 3. Signals 저장
        if signals is None:
            signals = self.generate_signals(result)

        if signals:
            signal_dicts = []
            for sig in signals:
                signal_dicts.append({
                    'type': sig.type,
                    'ticker': sig.ticker,
                    'name': sig.name,
                    'indicator': sig.indicator,
                    'value': sig.value,
                    'threshold': sig.threshold,
                    'z_score': sig.z_score,
                    'level': sig.level,
                    'description': sig.description,
                    'confidence': sig.confidence,
                    'direction': sig.direction.value if hasattr(sig.direction, 'value') else sig.direction,
                    'horizon': sig.horizon,
                    'source': sig.source,
                    'regime_aligned': getattr(sig, 'regime_aligned', False),
                    'metadata': sig.metadata if hasattr(sig, 'metadata') else {},
                })
            db.save_signals(signal_dicts, today)
            stats['signals'] = len(signal_dicts)

        # 4. 분석 로그 저장
        db.log_analysis(
            analysis_type='ark_holdings',
            status='SUCCESS',
            records=stats['holdings'],
            date_str=today
        )

        return stats

    def generate_signals(self, result: ARKAnalysisResult) -> List[EnhancedSignal]:
        """분석 결과를 EnhancedSignal로 변환"""
        signals = []

        # 1. 컨센서스 매수 신호
        for ticker in result.consensus_buys:
            change = next((c for c in result.top_increases if c.ticker == ticker), None)
            if change:
                signal = EnhancedSignal(
                    signal_id="",
                    type="ark_consensus",
                    ticker=ticker,
                    name=f"ARK Consensus Buy - {change.company}",
                    indicator="weight_change",
                    value=change.weight_change_1d,
                    threshold=0.5,
                    z_score=change.weight_change_1d / 0.5,
                    level="ALERT",
                    description=f"{len(change.etfs_increasing)}/{len(ARK_ETFS)} ARK ETF가 비중 증가",
                    confidence=0.70 + change.signal_strength * 0.2,
                    direction=PositionDirection.LONG,
                    horizon="short",
                    source="ark_holdings",
                    metadata={
                        "etfs_increasing": change.etfs_increasing,
                        "current_weight": change.current_weight,
                        "sector": change.sector,
                    }
                )
                signals.append(signal)

        # 2. 컨센서스 매도 신호
        for ticker in result.consensus_sells:
            change = next((c for c in result.top_decreases if c.ticker == ticker), None)
            if change:
                signal = EnhancedSignal(
                    signal_id="",
                    type="ark_consensus",
                    ticker=ticker,
                    name=f"ARK Consensus Sell - {change.company}",
                    indicator="weight_change",
                    value=change.weight_change_1d,
                    threshold=-0.5,
                    z_score=change.weight_change_1d / 0.5,
                    level="ALERT",
                    description=f"{len(change.etfs_decreasing)}/{len(ARK_ETFS)} ARK ETF가 비중 감소",
                    confidence=0.65 + change.signal_strength * 0.2,
                    direction=PositionDirection.SHORT,
                    horizon="short",
                    source="ark_holdings",
                    metadata={
                        "etfs_decreasing": change.etfs_decreasing,
                        "current_weight": change.current_weight,
                        "sector": change.sector,
                    }
                )
                signals.append(signal)

        # 3. 신규 편입 신호
        for ticker in result.new_positions:
            signal = EnhancedSignal(
                signal_id="",
                type="ark_new_position",
                ticker=ticker,
                name=f"ARK New Position - {ticker}",
                indicator="new_entry",
                value=1.0,
                threshold=0.0,
                z_score=2.0,
                level="ALERT",
                description=f"ARK ETF 신규 편입",
                confidence=0.75,
                direction=PositionDirection.LONG,
                horizon="short",
                source="ark_holdings",
            )
            signals.append(signal)

        # 4. 주요 비중 증가 (상위 3개)
        for change in result.top_increases[:3]:
            if change.ticker not in result.consensus_buys:
                signal = EnhancedSignal(
                    signal_id="",
                    type="ark_weight_increase",
                    ticker=change.ticker,
                    name=f"ARK Weight Increase - {change.company}",
                    indicator="weight_change",
                    value=change.weight_change_1d,
                    threshold=0.5,
                    z_score=change.weight_change_1d / 0.5,
                    level="WARNING",
                    description=f"비중 +{change.weight_change_1d:.2f}%p ({change.etf_count} ETF)",
                    confidence=0.60 + min(change.weight_change_1d * 0.1, 0.2),
                    direction=PositionDirection.LONG,
                    horizon="short",
                    source="ark_holdings",
                    metadata={"sector": change.sector}
                )
                signals.append(signal)

        return signals

    def print_report(self, result: ARKAnalysisResult):
        """분석 결과 출력"""
        print("\n" + "=" * 70)
        print("ARK ETF HOLDINGS ANALYSIS REPORT")
        print(f"Generated: {result.timestamp[:19]}")
        print("=" * 70)

        print(f"\n[Summary]")
        print(f"  ETFs Analyzed: {', '.join(result.etfs_analyzed)}")
        print(f"  Total Holdings: {result.total_holdings}")

        if result.consensus_buys:
            print(f"\n[🟢 Consensus Buys] (다수 ETF 비중 증가)")
            for ticker in result.consensus_buys:
                change = next((c for c in result.top_increases if c.ticker == ticker), None)
                if change:
                    print(f"  {ticker:6s} {change.company[:25]:25s} +{change.weight_change_1d:+.2f}%p "
                          f"({len(change.etfs_increasing)}/{len(ARK_ETFS)} ETF)")

        if result.consensus_sells:
            print(f"\n[🔴 Consensus Sells] (다수 ETF 비중 감소)")
            for ticker in result.consensus_sells:
                change = next((c for c in result.top_decreases if c.ticker == ticker), None)
                if change:
                    print(f"  {ticker:6s} {change.company[:25]:25s} {change.weight_change_1d:+.2f}%p "
                          f"({len(change.etfs_decreasing)}/{len(ARK_ETFS)} ETF)")

        if result.new_positions:
            print(f"\n[🆕 New Positions]")
            for ticker in result.new_positions:
                print(f"  {ticker}")

        if result.exited_positions:
            print(f"\n[🚪 Exited Positions]")
            for ticker in result.exited_positions:
                print(f"  {ticker}")

        print(f"\n[Top Weight Increases]")
        for change in result.top_increases[:5]:
            print(f"  {change.ticker:6s} {change.company[:25]:25s} {change.weight_change_1d:+.2f}%p "
                  f"(현재 {change.current_weight:.1f}%)")

        print(f"\n[Top Weight Decreases]")
        for change in result.top_decreases[:5]:
            print(f"  {change.ticker:6s} {change.company[:25]:25s} {change.weight_change_1d:+.2f}%p "
                  f"(현재 {change.current_weight:.1f}%)")

        print(f"\n[Sector Summary]")
        for sector in result.sector_summary[:8]:
            print(f"  {sector.sector:20s} {sector.total_weight:5.1f}%  ({sector.stock_count} stocks)")

        print("\n" + "=" * 70)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ARK ETF Holdings Analyzer Test")
    print("=" * 70)

    # 1. 데이터 수집
    collector = ARKHoldingsCollector()

    # 2. 분석기 생성 및 실행
    analyzer = ARKHoldingsAnalyzer(collector)
    result = analyzer.run_analysis()

    # 3. 리포트 출력
    analyzer.print_report(result)

    # 4. 오늘 스냅샷 저장 (히스토리 구축 시작)
    today = datetime.now().strftime("%Y-%m-%d")
    collector.save_snapshot(analyzer.current_holdings, today)

    # 5. 신호 생성
    print("\n[Generated Signals]")
    signals = analyzer.generate_signals(result)
    for sig in signals[:5]:
        print(f"  {sig.ticker:6s} {sig.direction.value:5s} Conf:{sig.confidence:.0%} - {sig.description[:40]}...")

    # 6. DB 저장
    print("\n[Saving to Database]")
    db = DatabaseManager()
    save_stats = analyzer.save_to_db(result, signals, db)
    print(f"  Holdings saved: {save_stats['holdings']}")
    print(f"  Weight changes: {save_stats['weight_changes']}")
    print(f"  Signals saved:  {save_stats['signals']}")

    # 7. DB 통계 확인
    print("\n[Database Stats]")
    db_stats = db.get_stats()
    for table, info in db_stats['tables'].items():
        if info['count'] > 0:
            print(f"  {table:20s}: {info['count']:5d} records ({info['min_date']} ~ {info['max_date']})")

    print("\n" + "=" * 70)
    print("Test Complete!")
