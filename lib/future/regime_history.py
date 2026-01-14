#!/usr/bin/env python3
"""
Regime History Analyzer
=======================
과거 유사 레짐 검색 및 수익률 분석

기능:
- 레짐 히스토리 저장 및 로드
- 현재 레짐과 유사한 과거 레짐 검색
- 유사 레짐 이후 수익률 분석
- 리포트 섹션 생성
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict


@dataclass
class RegimeSnapshot:
    """레짐 스냅샷"""
    timestamp: str
    regime: str              # BULLISH, NEUTRAL, BEARISH
    confidence: float        # 0-1
    risk_score: float        # 0-100
    vix: float
    rsi: float
    recommendation: str

    # 추가 컨텍스트
    dxy: Optional[float] = None
    gold_change: Optional[float] = None
    sector_rotation: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'RegimeSnapshot':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SimilarRegime:
    """유사 레짐 정보"""
    snapshot: RegimeSnapshot
    similarity_score: float
    days_later_returns: Dict[int, float]  # {5: 0.02, 20: 0.05, 60: 0.08}
    outcome: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"

    def to_dict(self) -> Dict:
        return {
            'snapshot': self.snapshot.to_dict(),
            'similarity_score': self.similarity_score,
            'days_later_returns': self.days_later_returns,
            'outcome': self.outcome
        }


@dataclass
class BacktestSection:
    """백테스팅 결과 섹션"""
    similar_regimes_count: int
    avg_returns: Dict[int, float]  # {5: 0.01, 20: 0.03, 60: 0.05}
    win_rate: float
    best_outcome: Optional[SimilarRegime] = None
    worst_outcome: Optional[SimilarRegime] = None
    similar_regimes: List[SimilarRegime] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        return {
            'similar_regimes_count': self.similar_regimes_count,
            'avg_returns': self.avg_returns,
            'win_rate': self.win_rate,
            'best_outcome': self.best_outcome.to_dict() if self.best_outcome else None,
            'worst_outcome': self.worst_outcome.to_dict() if self.worst_outcome else None,
            'similar_regimes': [r.to_dict() for r in self.similar_regimes[:5]],
            'summary': self.summary
        }

    def to_markdown(self) -> str:
        """마크다운 형식으로 변환"""
        md = []
        md.append("## 📊 백테스팅: 유사 레짐 분석")
        md.append("")

        if self.similar_regimes_count == 0:
            md.append("*유사한 과거 레짐을 찾을 수 없습니다.*")
            return "\n".join(md)

        md.append(f"**분석된 유사 레짐**: {self.similar_regimes_count}건")
        md.append(f"**승률** (20일 기준): {self.win_rate:.0%}")
        md.append("")

        # 평균 수익률 테이블
        md.append("### 예상 수익률 (유사 레짐 평균)")
        md.append("| 기간 | 평균 수익률 |")
        md.append("|------|------------|")
        for days, ret in sorted(self.avg_returns.items()):
            emoji = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"
            md.append(f"| {days}일 후 | {emoji} {ret:+.2%} |")
        md.append("")

        # 유사 레짐 상세
        if self.similar_regimes:
            md.append("### 유사 레짐 히스토리")
            md.append("| 날짜 | 레짐 | 유사도 | 20일 후 수익률 |")
            md.append("|------|------|--------|--------------|")
            for sr in self.similar_regimes[:5]:
                date = sr.snapshot.timestamp[:10]
                regime = sr.snapshot.regime
                sim = sr.similarity_score
                ret_20d = sr.days_later_returns.get(20, 0)
                outcome_emoji = "🟢" if sr.outcome == "POSITIVE" else "🔴" if sr.outcome == "NEGATIVE" else "⚪"
                md.append(f"| {date} | {regime} | {sim:.0%} | {outcome_emoji} {ret_20d:+.2%} |")
            md.append("")

        # 요약
        if self.summary:
            md.append(f"**요약**: {self.summary}")

        return "\n".join(md)


class RegimeHistoryAnalyzer:
    """
    레짐 히스토리 분석기

    사용법:
        analyzer = RegimeHistoryAnalyzer()

        # 현재 레짐 저장
        analyzer.save_regime(current_regime_snapshot)

        # 유사 레짐 검색
        backtest = analyzer.find_similar_regimes(current_regime_snapshot)
    """

    def __init__(self, history_file: str = "outputs/regime_history.json"):
        self.history_file = Path(history_file)
        self.history: List[RegimeSnapshot] = []
        self.load_history()

    def load_history(self):
        """히스토리 로드"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = [RegimeSnapshot.from_dict(d) for d in data]
                print(f"[RegimeHistory] Loaded {len(self.history)} historical regimes")
            except Exception as e:
                print(f"[RegimeHistory] Failed to load history: {e}")
                self.history = []
        else:
            self.history = []

    def save_history(self):
        """히스토리 저장"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump([s.to_dict() for s in self.history], f, indent=2, ensure_ascii=False)

    def save_regime(self, snapshot: RegimeSnapshot):
        """새 레짐 스냅샷 저장"""
        # 같은 날 데이터가 있으면 업데이트
        today = snapshot.timestamp[:10]
        self.history = [s for s in self.history if s.timestamp[:10] != today]
        self.history.append(snapshot)

        # 시간순 정렬
        self.history.sort(key=lambda x: x.timestamp)
        self.save_history()

    def calculate_similarity(self, current: RegimeSnapshot, historical: RegimeSnapshot) -> float:
        """
        두 레짐 간 유사도 계산 (0-1)

        가중치:
        - 레짐 일치: 40%
        - 신뢰도 차이: 20%
        - VIX 차이: 20%
        - RSI 차이: 20%
        """
        score = 0.0

        # 1. 레짐 일치 (40%)
        if current.regime == historical.regime:
            score += 0.4
        elif (current.regime in ["BULLISH", "NEUTRAL"] and historical.regime in ["BULLISH", "NEUTRAL"]) or \
             (current.regime in ["BEARISH", "NEUTRAL"] and historical.regime in ["BEARISH", "NEUTRAL"]):
            score += 0.2  # 부분 일치

        # 2. 신뢰도 차이 (20%)
        conf_diff = abs(current.confidence - historical.confidence)
        score += 0.2 * max(0, 1 - conf_diff * 2)

        # 3. VIX 차이 (20%)
        vix_diff = abs(current.vix - historical.vix)
        score += 0.2 * max(0, 1 - vix_diff / 20)

        # 4. RSI 차이 (20%)
        rsi_diff = abs(current.rsi - historical.rsi)
        score += 0.2 * max(0, 1 - rsi_diff / 30)

        return min(1.0, score)

    def get_future_returns(self, date_str: str, days_list: List[int] = [5, 20, 60]) -> Dict[int, float]:
        """특정 날짜 이후 SPY 수익률 계산"""
        try:
            start_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            end_date = start_date + timedelta(days=max(days_list) + 5)

            df = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"), progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 2:
                return {}

            start_price = df['Close'].iloc[0]
            returns = {}

            for days in days_list:
                if len(df) > days:
                    end_price = df['Close'].iloc[days]
                    returns[days] = (end_price - start_price) / start_price

            return returns

        except Exception as e:
            print(f"[RegimeHistory] Failed to get returns for {date_str}: {e}")
            return {}

    def find_similar_regimes(
        self,
        current: RegimeSnapshot,
        min_similarity: float = 0.6,
        max_results: int = 10,
        min_days_ago: int = 30  # 최소 30일 전 레짐만
    ) -> BacktestSection:
        """
        현재 레짐과 유사한 과거 레짐 검색
        """
        cutoff_date = datetime.now() - timedelta(days=min_days_ago)

        similar = []
        for hist in self.history:
            hist_date = datetime.strptime(hist.timestamp[:10], "%Y-%m-%d")

            # 너무 최근 데이터 제외
            if hist_date >= cutoff_date:
                continue

            similarity = self.calculate_similarity(current, hist)

            if similarity >= min_similarity:
                # 미래 수익률 계산
                returns = self.get_future_returns(hist.timestamp)

                if returns:
                    ret_20d = returns.get(20, 0)
                    outcome = "POSITIVE" if ret_20d > 0.01 else "NEGATIVE" if ret_20d < -0.01 else "NEUTRAL"

                    similar.append(SimilarRegime(
                        snapshot=hist,
                        similarity_score=similarity,
                        days_later_returns=returns,
                        outcome=outcome
                    ))

        # 유사도 순 정렬
        similar.sort(key=lambda x: x.similarity_score, reverse=True)
        similar = similar[:max_results]

        if not similar:
            return BacktestSection(
                similar_regimes_count=0,
                avg_returns={},
                win_rate=0.0,
                summary="유사한 과거 레짐 데이터가 부족합니다."
            )

        # 통계 계산
        all_returns = {5: [], 20: [], 60: []}
        for sr in similar:
            for days, ret in sr.days_later_returns.items():
                if days in all_returns:
                    all_returns[days].append(ret)

        avg_returns = {}
        for days, rets in all_returns.items():
            if rets:
                avg_returns[days] = np.mean(rets)

        # 승률 (20일 기준)
        wins = sum(1 for sr in similar if sr.days_later_returns.get(20, 0) > 0)
        win_rate = wins / len(similar) if similar else 0

        # 최고/최악 케이스
        sorted_by_return = sorted(similar, key=lambda x: x.days_later_returns.get(20, 0), reverse=True)
        best = sorted_by_return[0] if sorted_by_return else None
        worst = sorted_by_return[-1] if sorted_by_return else None

        # 요약 생성
        avg_20d = avg_returns.get(20, 0)
        if avg_20d > 0.02:
            summary = f"과거 유사 레짐에서 평균 {avg_20d:.1%} 상승했습니다. 긍정적 시나리오 우세."
        elif avg_20d < -0.02:
            summary = f"과거 유사 레짐에서 평균 {avg_20d:.1%} 하락했습니다. 주의가 필요합니다."
        else:
            summary = f"과거 유사 레짐에서 평균 {avg_20d:.1%} 변동했습니다. 방향성 불명확."

        return BacktestSection(
            similar_regimes_count=len(similar),
            avg_returns=avg_returns,
            win_rate=win_rate,
            best_outcome=best,
            worst_outcome=worst,
            similar_regimes=similar,
            summary=summary
        )

    def create_snapshot_from_report(self, report_data: Dict) -> RegimeSnapshot:
        """AI 리포트 데이터에서 스냅샷 생성"""
        import re

        # 레짐 추출
        regime = "NEUTRAL"
        regime_str = report_data.get('regime_analysis', '')
        if 'BULLISH' in regime_str.upper() or 'Bull' in regime_str:
            regime = "BULLISH"
        elif 'BEARISH' in regime_str.upper() or 'Bear' in regime_str:
            regime = "BEARISH"

        # 신뢰도 추출
        confidence = 0.5
        conf_str = report_data.get('confidence_analysis', '')
        if conf_str:
            match = re.search(r'(\d+)%', conf_str)
            if match:
                confidence = float(match.group(1)) / 100

        # 기술 지표
        tech = report_data.get('technical_indicators', {})
        vix = tech.get('vix', 20)
        rsi = tech.get('rsi', 50)

        # 리스크 점수
        risk_str = report_data.get('risk_assessment', '')
        risk_score = 50
        match = re.search(r'(\d+\.?\d*)/100', risk_str)
        if match:
            risk_score = float(match.group(1))

        # 권고
        rec_str = report_data.get('final_recommendation', '')
        if '매수' in rec_str or 'BUY' in rec_str.upper():
            recommendation = "BUY"
        elif '매도' in rec_str or 'SELL' in rec_str.upper():
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # 추가 컨텍스트 (DXY, Gold, Sector)
        dxy, gold_change, sector_rotation = self._fetch_additional_context(report_data)

        return RegimeSnapshot(
            timestamp=report_data.get('timestamp', datetime.now().isoformat()),
            regime=regime,
            confidence=confidence,
            risk_score=risk_score,
            vix=vix,
            rsi=rsi,
            recommendation=recommendation,
            dxy=dxy,
            gold_change=gold_change,
            sector_rotation=sector_rotation
        )

    def _fetch_additional_context(self, report_data: Dict) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """DXY, Gold 변화율, 섹터 로테이션 데이터 수집"""
        dxy = None
        gold_change = None
        sector_rotation = None

        try:
            # DXY (US Dollar Index) - UUP ETF 사용
            uup = yf.download('UUP', period='5d', progress=False)
            if len(uup) >= 2:
                dxy = float(uup['Close'].iloc[-1])

            # Gold 변화율 - GLD ETF
            gld = yf.download('GLD', period='5d', progress=False)
            if len(gld) >= 2:
                gold_change = float((gld['Close'].iloc[-1] / gld['Close'].iloc[-2] - 1) * 100)

            # 섹터 로테이션 - report_data에서 추출 또는 ETF 분석
            if 'etf_analysis' in report_data:
                etf = report_data['etf_analysis']
                if isinstance(etf, dict) and 'leading_sector' in etf:
                    sector_rotation = etf['leading_sector']
            elif 'sector_rotation' in report_data:
                sector_rotation = report_data['sector_rotation']

            # 섹터 로테이션 fallback: XLK vs XLU 비율
            if sector_rotation is None:
                xlk = yf.download('XLK', period='5d', progress=False)
                xlu = yf.download('XLU', period='5d', progress=False)
                if len(xlk) >= 2 and len(xlu) >= 2:
                    xlk_ret = float((xlk['Close'].iloc[-1] / xlk['Close'].iloc[-2] - 1) * 100)
                    xlu_ret = float((xlu['Close'].iloc[-1] / xlu['Close'].iloc[-2] - 1) * 100)
                    if xlk_ret > xlu_ret + 0.5:
                        sector_rotation = "RISK_ON"
                    elif xlu_ret > xlk_ret + 0.5:
                        sector_rotation = "RISK_OFF"
                    else:
                        sector_rotation = "NEUTRAL"

        except Exception as e:
            print(f"[RegimeHistory] Warning: Failed to fetch additional context: {e}")

        return dxy, gold_change, sector_rotation


# ============================================================================
# Integration with AI Report Generator
# ============================================================================

def add_backtest_section_to_report(report_data: Dict) -> str:
    """
    AI 리포트에 백테스팅 섹션 추가

    Args:
        report_data: AI 리포트 JSON 데이터

    Returns:
        백테스팅 섹션 마크다운
    """
    analyzer = RegimeHistoryAnalyzer()

    # 현재 스냅샷 생성
    current_snapshot = analyzer.create_snapshot_from_report(report_data)

    # 현재 스냅샷 저장
    analyzer.save_regime(current_snapshot)

    # 유사 레짐 검색
    backtest = analyzer.find_similar_regimes(current_snapshot)

    return backtest.to_markdown()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Regime History Analyzer Test")
    print("=" * 70)

    analyzer = RegimeHistoryAnalyzer()

    # 테스트 스냅샷 생성
    test_snapshot = RegimeSnapshot(
        timestamp=datetime.now().isoformat(),
        regime="BULLISH",
        confidence=0.75,
        risk_score=35,
        vix=18.5,
        rsi=62,
        recommendation="BUY"
    )

    print(f"\n[Test Snapshot]")
    print(f"  Regime: {test_snapshot.regime}")
    print(f"  Confidence: {test_snapshot.confidence:.0%}")
    print(f"  VIX: {test_snapshot.vix}")
    print(f"  RSI: {test_snapshot.rsi}")

    # 유사 레짐 검색
    print(f"\n[Searching Similar Regimes...]")
    backtest = analyzer.find_similar_regimes(test_snapshot)

    print(f"\n[Results]")
    print(f"  Similar Regimes Found: {backtest.similar_regimes_count}")
    print(f"  Win Rate (20d): {backtest.win_rate:.0%}")

    if backtest.avg_returns:
        print(f"  Avg Returns:")
        for days, ret in sorted(backtest.avg_returns.items()):
            print(f"    {days}d: {ret:+.2%}")

    print(f"\n[Markdown Output]")
    print(backtest.to_markdown())

    print("\nTest Complete!")
