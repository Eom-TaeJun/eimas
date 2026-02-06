#!/usr/bin/env python3
"""
ETF Flow Analyzer - Main Analyzer
============================================================

Analyzes ETF fund flows and sector rotation

Class:
    - ETFFlowAnalyzer: Main ETF flow analysis engine
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from .enums import MarketSentiment, StyleRotation, CyclePhase
from .schemas import ETFData, FlowComparison, SectorRotationResult, MarketRegimeResult

logger = logging.getLogger(__name__)


class ETFFlowAnalyzer:
    """
    ETF 비중/자금흐름 기반 시장 분석기

    주요 기능:
    1. ETF 데이터 수집 (yfinance)
    2. Cross-Sectional 비교 분석
    3. 섹터 로테이션 분석
    4. 시장 레짐 판단
    """

    # 섹터 분류 (경기 민감도 기준)
    OFFENSIVE_SECTORS = ["XLK", "XLY", "XLF", "XLI", "XLB", "XLC"]  # 공격적
    DEFENSIVE_SECTORS = ["XLP", "XLU", "XLV", "XLRE"]               # 방어적

    # 비교 쌍 정의
    COMPARISON_PAIRS = [
        ("Growth vs Value", "VUG", "VTV"),
        ("Large vs Small", "SPY", "IWM"),
        ("US vs Global", "SPY", "EFA"),
        ("Equity vs Bond", "SPY", "AGG"),
        ("HY vs Treasury", "HYG", "TLT"),
        ("Tech vs Energy", "XLK", "XLE"),
        ("Discretionary vs Staples", "XLY", "XLP"),
        ("Growth vs Dividend", "QQQ", "VTV"),
    ]

    def __init__(self, lookback_days: int = 120):
        """
        Args:
            lookback_days: 데이터 수집 기간 (기본 120일)
        """
        self.lookback_days = lookback_days
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.etf_data: Dict[str, ETFData] = {}
        self.last_update: Optional[datetime] = None

    def fetch_etf_data(self, tickers: List[str] = None, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        ETF 데이터 수집

        Args:
            tickers: 수집할 티커 목록 (None이면 전체)
            force_refresh: 캐시 무시하고 새로 수집

        Returns:
            {ticker: DataFrame} 형태의 가격 데이터
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not available")

        # 티커 목록 준비
        if tickers is None:
            tickers = []
            for category in ETF_UNIVERSE.values():
                tickers.extend(category.keys())
            tickers = list(set(tickers))

        # 캐시 확인
        if not force_refresh and self.data_cache and self.last_update:
            cache_age = datetime.now() - self.last_update
            if cache_age < timedelta(hours=1):
                return self.data_cache

        # 데이터 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        print(f"Fetching {len(tickers)} ETFs...")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker'
        )

        # 개별 DataFrame으로 분리
        result = {}
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()

                if not df.empty and 'Close' in df.columns:
                    result[ticker] = df
            except Exception as e:
                print(f"Warning: Failed to process {ticker}: {e}")

        self.data_cache = result
        self.last_update = datetime.now()
        print(f"Fetched {len(result)} ETFs successfully")

        return result

    def calculate_etf_metrics(self, ticker: str, df: pd.DataFrame, category: str = "unknown") -> Optional[ETFData]:
        """개별 ETF 지표 계산"""
        try:
            if df.empty or len(df) < 20:
                return None

            close = df['Close']
            volume = df['Volume'] if 'Volume' in df.columns else pd.Series([0] * len(df))

            current_price = float(close.iloc[-1])

            # 수익률 계산
            def safe_return(days):
                if len(close) > days:
                    return float((close.iloc[-1] / close.iloc[-days-1] - 1) * 100)
                return 0.0

            change_1d = safe_return(1)
            change_5d = safe_return(5)
            change_20d = safe_return(20)
            change_60d = safe_return(60)

            # 거래량 비율
            if len(volume) >= 20 and volume.iloc[-20:].mean() > 0:
                volume_ratio = float(volume.iloc[-1] / volume.iloc[-20:].mean())
            else:
                volume_ratio = 1.0

            # RSI 계산
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not pd.isna(rs.iloc[-1]) else 50.0

            # 상대 강도 (vs SPY) - SPY 데이터가 있는 경우
            relative_strength = 0.0
            if 'SPY' in self.data_cache and ticker != 'SPY':
                spy_close = self.data_cache['SPY']['Close']
                if len(spy_close) >= 20 and len(close) >= 20:
                    etf_ret = (close.iloc[-1] / close.iloc[-20] - 1) * 100
                    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-20] - 1) * 100
                    relative_strength = float(etf_ret - spy_ret)

            # ETF 이름 찾기
            name = ticker
            for cat, etfs in ETF_UNIVERSE.items():
                if ticker in etfs:
                    name = etfs[ticker]
                    if category == "unknown":
                        category = cat
                    break

            # ETF 상세 정보 수집 (yfinance .info 사용)
            expense_ratio = None
            dividend_yield = None
            total_assets = None
            holdings_count = None
            pe_ratio = None
            beta = None
            ytd_return = None
            three_year_return = None

            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info

                # 비용비율 (퍼센트로 저장)
                if 'annualReportExpenseRatio' in info and info['annualReportExpenseRatio']:
                    expense_ratio = float(info['annualReportExpenseRatio']) * 100
                elif 'expenseRatio' in info and info['expenseRatio']:
                    expense_ratio = float(info['expenseRatio']) * 100

                # 배당수익률
                if 'yield' in info and info['yield']:
                    dividend_yield = float(info['yield']) * 100
                elif 'dividendYield' in info and info['dividendYield']:
                    dividend_yield = float(info['dividendYield']) * 100
                elif 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
                    dividend_yield = float(info['trailingAnnualDividendYield']) * 100

                # AUM (총 자산) - 십억 달러 단위로 변환
                if 'totalAssets' in info and info['totalAssets']:
                    total_assets = float(info['totalAssets']) / 1e9

                # 보유 종목 수
                if 'holdings' in info and info['holdings']:
                    holdings_count = len(info['holdings'])

                # P/E Ratio
                if 'trailingPE' in info and info['trailingPE']:
                    pe_ratio = float(info['trailingPE'])

                # Beta
                if 'beta3Year' in info and info['beta3Year']:
                    beta = float(info['beta3Year'])
                elif 'beta' in info and info['beta']:
                    beta = float(info['beta'])

                # YTD 수익률
                if 'ytdReturn' in info and info['ytdReturn']:
                    ytd_return = float(info['ytdReturn']) * 100

                # 3년 수익률
                if 'threeYearAverageReturn' in info and info['threeYearAverageReturn']:
                    three_year_return = float(info['threeYearAverageReturn']) * 100

            except Exception as e:
                # 상세 정보 수집 실패는 무시 (핵심 기능에 영향 없음)
                pass

            return ETFData(
                ticker=ticker,
                name=name,
                category=category,
                current_price=current_price,
                change_1d=change_1d,
                change_5d=change_5d,
                change_20d=change_20d,
                change_60d=change_60d,
                volume_ratio=volume_ratio,
                rsi=rsi,
                relative_strength=relative_strength,
                expense_ratio=expense_ratio,
                dividend_yield=dividend_yield,
                total_assets=total_assets,
                holdings_count=holdings_count,
                pe_ratio=pe_ratio,
                beta=beta,
                ytd_return=ytd_return,
                three_year_return=three_year_return
            )

        except Exception as e:
            print(f"Error calculating metrics for {ticker}: {e}")
            return None

    def compare_etf_pair(self, etf_a: str, etf_b: str, pair_name: str) -> Optional[FlowComparison]:
        """
        ETF 쌍 비교 분석

        Args:
            etf_a: 첫 번째 ETF (분자)
            etf_b: 두 번째 ETF (분모)
            pair_name: 비교 이름

        Returns:
            FlowComparison 결과
        """
        if etf_a not in self.data_cache or etf_b not in self.data_cache:
            return None

        df_a = self.data_cache[etf_a]['Close']
        df_b = self.data_cache[etf_b]['Close']

        # 인덱스 정렬
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < 20:
            return None

        a = df_a.loc[common_idx]
        b = df_b.loc[common_idx]

        # 비율 시계열
        ratio = a / b

        # 현재 비율 및 통계
        ratio_current = float(ratio.iloc[-1])
        ratio_20d_avg = float(ratio.iloc[-20:].mean())
        ratio_std = float(ratio.iloc[-60:].std()) if len(ratio) >= 60 else float(ratio.std())

        # Z-score
        ratio_z_score = (ratio_current - ratio_20d_avg) / ratio_std if ratio_std > 0 else 0.0

        # 수익률 스프레드
        def spread(days):
            if len(a) > days and len(b) > days:
                ret_a = (a.iloc[-1] / a.iloc[-days-1] - 1) * 100
                ret_b = (b.iloc[-1] / b.iloc[-days-1] - 1) * 100
                return float(ret_a - ret_b)
            return 0.0

        spread_1d = spread(1)
        spread_5d = spread(5)
        spread_20d = spread(20)

        # 신호 결정
        if ratio_z_score > 1.0 or spread_20d > 3.0:
            signal = "A_LEADING"
            strength = min(abs(ratio_z_score) / 2, 1.0)
        elif ratio_z_score < -1.0 or spread_20d < -3.0:
            signal = "B_LEADING"
            strength = min(abs(ratio_z_score) / 2, 1.0)
        else:
            signal = "NEUTRAL"
            strength = 0.0

        # 해석 생성
        interpretation = self._interpret_pair_comparison(pair_name, signal, spread_20d, ratio_z_score)

        return FlowComparison(
            pair_name=pair_name,
            etf_a=etf_a,
            etf_b=etf_b,
            ratio_current=round(ratio_current, 4),
            ratio_20d_avg=round(ratio_20d_avg, 4),
            ratio_z_score=round(ratio_z_score, 2),
            spread_1d=round(spread_1d, 2),
            spread_5d=round(spread_5d, 2),
            spread_20d=round(spread_20d, 2),
            signal=signal,
            strength=round(strength, 2),
            interpretation=interpretation
        )

    def _interpret_pair_comparison(self, pair_name: str, signal: str, spread: float, z_score: float) -> str:
        """쌍 비교 결과 해석"""
        interpretations = {
            "Growth vs Value": {
                "A_LEADING": "성장주 선호 → 금리 하락 기대 또는 성장 모멘텀",
                "B_LEADING": "가치주 선호 → 금리 상승 또는 밸류에이션 회귀",
            },
            "Large vs Small": {
                "A_LEADING": "대형주 선호 → 방어적 포지셔닝, 경기 둔화 우려",
                "B_LEADING": "소형주 선호 → 위험 선호, 경기 확장 기대",
            },
            "US vs Global": {
                "A_LEADING": "미국 선호 → 달러 강세, 미국 예외론",
                "B_LEADING": "글로벌 선호 → 달러 약세 기대, 분산 투자",
            },
            "Equity vs Bond": {
                "A_LEADING": "주식 선호 → Risk-on, 성장 기대",
                "B_LEADING": "채권 선호 → Risk-off, 안전자산 선호",
            },
            "HY vs Treasury": {
                "A_LEADING": "하이일드 선호 → 신용 위험 감내, 스프레드 축소",
                "B_LEADING": "국채 선호 → 신용 위험 회피, 안전자산 선호",
            },
            "Tech vs Energy": {
                "A_LEADING": "기술주 선호 → 성장 테마, 금리 민감",
                "B_LEADING": "에너지 선호 → 인플레이션 헤지, 원자재 강세",
            },
            "Discretionary vs Staples": {
                "A_LEADING": "소비재(경기민감) 선호 → 소비 강세 기대",
                "B_LEADING": "필수소비재 선호 → 방어적 포지셔닝",
            },
        }

        base = interpretations.get(pair_name, {}).get(signal, "")
        if signal == "NEUTRAL":
            return "중립적 상태, 명확한 방향성 없음"

        strength_text = "강한" if abs(z_score) > 1.5 else "완만한"
        return f"{strength_text} {base} (20일 스프레드: {spread:+.1f}%)"

    def analyze_sector_rotation(self) -> SectorRotationResult:
        """
        섹터 로테이션 분석

        섹터별 상대 성과를 분석하여 경기 사이클 단계 추정
        """
        sector_returns = {}

        # 섹터별 20일 수익률 계산
        for ticker in ETF_UNIVERSE['sector'].keys():
            if ticker in self.etf_data:
                sector_returns[ticker] = self.etf_data[ticker].change_20d

        if len(sector_returns) < 5:
            return SectorRotationResult(
                cycle_phase=CyclePhase.UNCERTAIN,
                leading_sectors=[],
                lagging_sectors=[],
                sector_rankings={},
                offensive_score=50.0,
                defensive_score=50.0,
                confidence=0.0,
                interpretation="데이터 부족"
            )

        # 순위 계산
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        rankings = {ticker: rank + 1 for rank, (ticker, _) in enumerate(sorted_sectors)}

        leading = [t for t, _ in sorted_sectors[:3]]
        lagging = [t for t, _ in sorted_sectors[-3:]]

        # 공격적/방어적 점수 계산
        offensive_returns = [sector_returns.get(t, 0) for t in self.OFFENSIVE_SECTORS if t in sector_returns]
        defensive_returns = [sector_returns.get(t, 0) for t in self.DEFENSIVE_SECTORS if t in sector_returns]

        offensive_avg = np.mean(offensive_returns) if offensive_returns else 0
        defensive_avg = np.mean(defensive_returns) if defensive_returns else 0

        # 0-100 점수로 변환 (상대적)
        total_avg = np.mean(list(sector_returns.values()))
        offensive_score = 50 + (offensive_avg - total_avg) * 10
        defensive_score = 50 + (defensive_avg - total_avg) * 10
        offensive_score = max(0, min(100, offensive_score))
        defensive_score = max(0, min(100, defensive_score))

        # 경기 사이클 단계 추정
        cycle_phase, interpretation = self._estimate_cycle_phase(leading, lagging, offensive_score, defensive_score)

        confidence = 0.7 if abs(offensive_score - defensive_score) > 10 else 0.4

        return SectorRotationResult(
            cycle_phase=cycle_phase,
            leading_sectors=leading,
            lagging_sectors=lagging,
            sector_rankings=rankings,
            offensive_score=round(offensive_score, 1),
            defensive_score=round(defensive_score, 1),
            confidence=confidence,
            interpretation=interpretation
        )

    def _estimate_cycle_phase(self, leading: List[str], lagging: List[str],
                               offensive: float, defensive: float) -> Tuple[CyclePhase, str]:
        """경기 사이클 단계 추정"""

        # 선도 섹터 기반 판단
        if "XLF" in leading or "XLY" in leading:
            if offensive > defensive + 5:
                return CyclePhase.EARLY_EXPANSION, "초기 확장: 금융/소비재 강세, 경기 회복 신호"

        if "XLK" in leading or "XLI" in leading:
            if offensive > defensive:
                return CyclePhase.MID_EXPANSION, "중기 확장: 기술/산업재 강세, 경기 확장 지속"

        if "XLE" in leading or "XLB" in leading:
            return CyclePhase.LATE_EXPANSION, "후기 확장: 에너지/원자재 강세, 인플레이션 우려"

        if "XLP" in leading or "XLU" in leading:
            if defensive > offensive + 5:
                return CyclePhase.RECESSION, "경기 둔화: 방어주 강세, 위험 회피"

        return CyclePhase.UNCERTAIN, "불확실: 명확한 사이클 신호 없음"

    def analyze_market_regime(self) -> MarketRegimeResult:
        """
        종합 시장 레짐 분석

        모든 비교 지표를 종합하여 시장 상태 판단
        """
        # 1. 모든 비교 쌍 분석
        comparisons = {}
        for pair_name, etf_a, etf_b in self.COMPARISON_PAIRS:
            comp = self.compare_etf_pair(etf_a, etf_b, pair_name)
            if comp:
                comparisons[pair_name] = comp

        # 2. 섹터 로테이션 분석
        sector_result = self.analyze_sector_rotation()

        # 3. 핵심 스프레드 추출
        growth_value = comparisons.get("Growth vs Value")
        large_small = comparisons.get("Large vs Small")
        us_global = comparisons.get("US vs Global")
        equity_bond = comparisons.get("Equity vs Bond")
        hy_treasury = comparisons.get("HY vs Treasury")

        growth_value_spread = growth_value.spread_20d if growth_value else 0
        large_small_spread = large_small.spread_20d if large_small else 0
        us_global_spread = us_global.spread_20d if us_global else 0
        equity_bond_spread = equity_bond.spread_20d if equity_bond else 0
        hy_treasury_spread = hy_treasury.spread_20d if hy_treasury else 0

        # 4. 시장 심리 판단
        risk_signals = []

        # Risk-on 신호
        if equity_bond and equity_bond.signal == "A_LEADING":
            risk_signals.append(1)  # 주식 선호
        if hy_treasury and hy_treasury.signal == "A_LEADING":
            risk_signals.append(1)  # 하이일드 선호
        if large_small and large_small.signal == "B_LEADING":
            risk_signals.append(1)  # 소형주 선호

        # Risk-off 신호
        if equity_bond and equity_bond.signal == "B_LEADING":
            risk_signals.append(-1)  # 채권 선호
        if hy_treasury and hy_treasury.signal == "B_LEADING":
            risk_signals.append(-1)  # 국채 선호
        if large_small and large_small.signal == "A_LEADING":
            risk_signals.append(-1)  # 대형주 선호

        # 심리 점수 계산
        if risk_signals:
            risk_score = sum(risk_signals) / len(risk_signals)
        else:
            risk_score = 0

        if risk_score > 0.3:
            sentiment = MarketSentiment.RISK_ON
        elif risk_score < -0.3:
            sentiment = MarketSentiment.RISK_OFF
        elif risk_signals and max(risk_signals) != min(risk_signals):
            sentiment = MarketSentiment.MIXED
        else:
            sentiment = MarketSentiment.NEUTRAL

        # Risk Appetite Score (0-100)
        risk_appetite_score = 50 + risk_score * 30
        risk_appetite_score += equity_bond_spread * 2  # 주식 초과수익 반영
        risk_appetite_score = max(0, min(100, risk_appetite_score))

        # 5. 스타일 로테이션 판단
        if growth_value and growth_value.signal == "A_LEADING":
            style_rotation = StyleRotation.GROWTH_LEADING
        elif growth_value and growth_value.signal == "B_LEADING":
            style_rotation = StyleRotation.VALUE_LEADING
        elif large_small and large_small.signal == "B_LEADING":
            style_rotation = StyleRotation.SMALL_CAP_LEADING
        elif large_small and large_small.signal == "A_LEADING":
            style_rotation = StyleRotation.LARGE_CAP_LEADING
        else:
            style_rotation = StyleRotation.NEUTRAL

        # 6. 시장 폭 점수 (상승 ETF 비율)
        if self.etf_data:
            positive_count = sum(1 for e in self.etf_data.values() if e.change_20d > 0)
            breadth_score = (positive_count / len(self.etf_data)) * 100
        else:
            breadth_score = 50.0

        # 7. 신호 및 경고 생성
        signals = []
        warnings = []

        for name, comp in comparisons.items():
            if comp.strength > 0.5:
                signals.append(f"{name}: {comp.interpretation}")

        if sentiment == MarketSentiment.MIXED:
            warnings.append("혼조 신호: 방향성 불명확, 변동성 증가 가능")

        if abs(hy_treasury_spread) > 5:
            if hy_treasury_spread < -5:
                warnings.append("신용 스프레드 확대: 신용 위험 상승 신호")

        if breadth_score < 40:
            warnings.append(f"시장 폭 약화: {breadth_score:.0f}%만 상승 중")

        # 신뢰도
        confidence = 0.5 + len(signals) * 0.1
        confidence = min(0.9, confidence)

        return MarketRegimeResult(
            sentiment=sentiment,
            style_rotation=style_rotation,
            cycle_phase=sector_result.cycle_phase,
            risk_appetite_score=round(risk_appetite_score, 1),
            breadth_score=round(breadth_score, 1),
            growth_value_spread=round(growth_value_spread, 2),
            large_small_spread=round(large_small_spread, 2),
            us_global_spread=round(us_global_spread, 2),
            equity_bond_spread=round(equity_bond_spread, 2),
            hy_treasury_spread=round(hy_treasury_spread, 2),
            signals=signals,
            warnings=warnings,
            confidence=round(confidence, 2)
        )

    def run_full_analysis(self, tickers: List[str] = None) -> Dict[str, Any]:
        """
        전체 분석 실행

        Returns:
            종합 분석 결과 딕셔너리
        """
        # 1. 데이터 수집
        self.fetch_etf_data(tickers)

        # 2. 개별 ETF 지표 계산
        for category, etfs in ETF_UNIVERSE.items():
            for ticker, name in etfs.items():
                if ticker in self.data_cache:
                    etf_data = self.calculate_etf_metrics(ticker, self.data_cache[ticker], category)
                    if etf_data:
                        self.etf_data[ticker] = etf_data

        print(f"Calculated metrics for {len(self.etf_data)} ETFs")

        # 3. 비교 분석
        comparisons = []
        for pair_name, etf_a, etf_b in self.COMPARISON_PAIRS:
            comp = self.compare_etf_pair(etf_a, etf_b, pair_name)
            if comp:
                comparisons.append(comp)

        # 4. 섹터 로테이션
        sector_rotation = self.analyze_sector_rotation()

        # 5. 종합 레짐 분석
        market_regime = self.analyze_market_regime()

        return {
            "timestamp": datetime.now().isoformat(),
            "etf_count": len(self.etf_data),
            "market_regime": market_regime.to_dict(),
            "sector_rotation": sector_rotation.to_dict(),
            "comparisons": [c.to_dict() for c in comparisons],
            "top_performers": self._get_top_performers(5),
            "worst_performers": self._get_worst_performers(5),
            "etf_details": self._get_etf_details(),
        }

    def _get_etf_details(self) -> List[Dict]:
        """ETF 상세 정보 목록 (AUM 기준 정렬)"""
        etf_list = []
        for etf in self.etf_data.values():
            etf_list.append({
                "ticker": etf.ticker,
                "name": etf.name,
                "category": etf.category,
                "current_price": etf.current_price,
                "expense_ratio": etf.expense_ratio,
                "dividend_yield": etf.dividend_yield,
                "total_assets": etf.total_assets,
                "holdings_count": etf.holdings_count,
                "pe_ratio": etf.pe_ratio,
                "beta": etf.beta,
                "ytd_return": etf.ytd_return,
                "three_year_return": etf.three_year_return,
                "change_20d": etf.change_20d,
                "rsi": etf.rsi
            })
        # AUM이 있는 것 우선, 그 다음 AUM 크기순
        etf_list.sort(key=lambda x: (x['total_assets'] is None, -(x['total_assets'] or 0)))
        return etf_list

    def _get_top_performers(self, n: int = 5) -> List[Dict]:
        """상위 성과 ETF"""
        sorted_etfs = sorted(self.etf_data.values(), key=lambda x: x.change_20d, reverse=True)
        return [{"ticker": e.ticker, "name": e.name, "return_20d": e.change_20d} for e in sorted_etfs[:n]]

    def _get_worst_performers(self, n: int = 5) -> List[Dict]:
        """하위 성과 ETF"""
        sorted_etfs = sorted(self.etf_data.values(), key=lambda x: x.change_20d)
        return [{"ticker": e.ticker, "name": e.name, "return_20d": e.change_20d} for e in sorted_etfs[:n]]

    def print_summary(self, result: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "=" * 70)
        print("ETF Flow Analysis Summary")
        print("=" * 70)

        regime = result['market_regime']
        print(f"\n[Market Regime]")
        print(f"  Sentiment: {regime['sentiment']}")
        print(f"  Style Rotation: {regime['style_rotation']}")
        print(f"  Cycle Phase: {regime['cycle_phase']}")
        print(f"  Risk Appetite: {regime['risk_appetite_score']:.1f}/100")
        print(f"  Market Breadth: {regime['breadth_score']:.1f}%")

        print(f"\n[Key Spreads (20D)]")
        print(f"  Growth vs Value: {regime['growth_value_spread']:+.2f}%")
        print(f"  Large vs Small: {regime['large_small_spread']:+.2f}%")
        print(f"  Equity vs Bond: {regime['equity_bond_spread']:+.2f}%")
        print(f"  HY vs Treasury: {regime['hy_treasury_spread']:+.2f}%")

        sector = result['sector_rotation']
        print(f"\n[Sector Rotation]")
        print(f"  Phase: {sector['cycle_phase']}")
        print(f"  Leading: {', '.join(sector['leading_sectors'])}")
        print(f"  Lagging: {', '.join(sector['lagging_sectors'])}")
        print(f"  Offensive Score: {sector['offensive_score']:.1f}")
        print(f"  Defensive Score: {sector['defensive_score']:.1f}")

        if regime['signals']:
            print(f"\n[Signals]")
            for sig in regime['signals'][:5]:
                print(f"  • {sig}")

        if regime['warnings']:
            print(f"\n[Warnings]")
            for warn in regime['warnings']:
                print(f"  ⚠ {warn}")

        print(f"\n[Top Performers (20D)]")
        for p in result['top_performers']:
            print(f"  {p['ticker']:6s} {p['name'][:25]:25s} {p['return_20d']:+.2f}%")

        print(f"\n[Worst Performers (20D)]")
        for p in result['worst_performers']:
            print(f"  {p['ticker']:6s} {p['name'][:25]:25s} {p['return_20d']:+.2f}%")

        # ETF 상세 정보 출력
        if 'etf_details' in result and result['etf_details']:
            print(f"\n[ETF Details (Top 5)]")
            print(f"  {'Ticker':<6} {'Name':<20} {'Expense':<8} {'Yield':<8} {'AUM':<10} {'P/E':<8} {'Beta':<6}")
            print(f"  {'-'*6:<6} {'-'*20:<20} {'-'*8:<8} {'-'*8:<8} {'-'*10:<10} {'-'*8:<8} {'-'*6:<6}")
            for detail in result['etf_details'][:5]:
                exp = f"{detail.get('expense_ratio', 'N/A'):.2f}%" if detail.get('expense_ratio') else "N/A"
                yld = f"{detail.get('dividend_yield', 'N/A'):.2f}%" if detail.get('dividend_yield') else "N/A"
                aum = f"${detail.get('total_assets', 'N/A'):.1f}B" if detail.get('total_assets') else "N/A"
                pe = f"{detail.get('pe_ratio', 'N/A'):.1f}" if detail.get('pe_ratio') else "N/A"
                beta = f"{detail.get('beta', 'N/A'):.2f}" if detail.get('beta') else "N/A"
                print(f"  {detail['ticker']:<6} {detail['name'][:20]:<20} {exp:<8} {yld:<8} {aum:<10} {pe:<8} {beta:<6}")

        print("\n" + "=" * 70)

    def analyze(self) -> Dict[str, Any]:
        """
        간략한 ETF 분석 실행 (main.py 파이프라인 호출용)

        Returns:
            Dict with:
                - rotation_signal: 섹터 로테이션 신호
                - style_signal: 스타일 로테이션 신호
                - sentiment: 시장 심리
                - risk_appetite: 위험 선호도 점수
        """
        try:
            # 핵심 ETF만 빠르게 수집
            core_tickers = ['SPY', 'QQQ', 'IWM', 'VUG', 'VTV', 'AGG', 'TLT', 'HYG', 'XLK', 'XLE', 'XLF', 'XLV']
            self.fetch_etf_data(core_tickers)

            # 지표 계산
            for ticker in core_tickers:
                if ticker in self.data_cache:
                    etf_data = self.calculate_etf_metrics(ticker, self.data_cache[ticker])
                    if etf_data:
                        self.etf_data[ticker] = etf_data

            # 핵심 비교 분석
            growth_value = self.compare_etf_pair('VUG', 'VTV', 'Growth vs Value')
            large_small = self.compare_etf_pair('SPY', 'IWM', 'Large vs Small')
            equity_bond = self.compare_etf_pair('SPY', 'AGG', 'Equity vs Bond')

            # 섹터 로테이션 신호
            sector_rotation = self.analyze_sector_rotation()
            rotation_signal = sector_rotation.cycle_phase.value.replace('_', ' ').title()

            # 스타일 신호
            if growth_value and growth_value.signal == "A_LEADING":
                style_signal = "Growth Leading"
            elif growth_value and growth_value.signal == "B_LEADING":
                style_signal = "Value Leading"
            elif large_small and large_small.signal == "B_LEADING":
                style_signal = "Small Cap Leading"
            elif large_small and large_small.signal == "A_LEADING":
                style_signal = "Large Cap Leading"
            else:
                style_signal = "Neutral"

            # 시장 심리
            if equity_bond and equity_bond.signal == "A_LEADING":
                sentiment = "Risk-On"
            elif equity_bond and equity_bond.signal == "B_LEADING":
                sentiment = "Risk-Off"
            else:
                sentiment = "Neutral"

            return {
                'rotation_signal': rotation_signal,
                'style_signal': style_signal,
                'sentiment': sentiment,
                'leading_sectors': sector_rotation.leading_sectors,
                'lagging_sectors': sector_rotation.lagging_sectors,
                'offensive_score': sector_rotation.offensive_score,
                'defensive_score': sector_rotation.defensive_score,
                'confidence': sector_rotation.confidence
            }

        except Exception as e:
            return {
                'rotation_signal': 'N/A',
                'style_signal': 'N/A',
                'sentiment': 'N/A',
                'error': str(e)
            }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ETF Flow Analyzer Test")
    print("=" * 70)

    if not YFINANCE_AVAILABLE:
        print("Error: yfinance not available. Install with: pip install yfinance")
        exit(1)

    # 분석기 생성
    analyzer = ETFFlowAnalyzer(lookback_days=90)

    # 전체 분석 실행
    result = analyzer.run_full_analysis()

    # 요약 출력
    analyzer.print_summary(result)

    print("\nTest Complete!")
