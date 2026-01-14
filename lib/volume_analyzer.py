#!/usr/bin/env python3
"""
Volume Anomaly Analyzer - 거래량 기반 정보 비대칭 탐지
========================================================

경제학적 근거:
1. 합리적 기대 가설 (Rational Expectations Hypothesis):
   - 거래량 폭발은 '참여자 간의 기대 불일치' 또는 '사적 정보(Private Information) 유입'을 의미
   - Kyle (1985): 정보 거래자(Informed Trader)가 시장에 진입하면 거래량이 먼저 반응
   - 가격 변동보다 거래량이 선행 지표로 작동

2. Volume-Price Divergence:
   - 거래량 급증 + 가격 정체 = 정보 비대칭 존재
   - 거래량 급증 + 가격 상승 = 매집(Accumulation)
   - 거래량 급증 + 가격 하락 = 분배(Distribution)

3. Abnormal Volume Detection:
   - 20일 이동평균 대비 3배~5배 이상 = 비정상 거래량
   - Z-score 기반 통계적 유의성 검정

Usage:
    analyzer = VolumeAnalyzer()
    results = analyzer.detect_anomalies(market_data)
    for r in results:
        print(r.alert_message)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger('eimas.volume_analyzer')


# =============================================================================
# Enums & Data Classes
# =============================================================================

class AnomalyType(Enum):
    """거래량 이상 유형"""
    ABNORMAL_SURGE = "abnormal_surge"           # 비정상 급증 (3x+)
    EXTREME_SURGE = "extreme_surge"             # 극심한 급증 (5x+)
    VOLUME_PRICE_DIVERGENCE = "divergence"      # 거래량-가격 괴리
    ACCUMULATION = "accumulation"               # 매집 신호
    DISTRIBUTION = "distribution"               # 분배 신호
    SILENT_VOLUME = "silent_volume"             # 저거래량 이상


class InformationType(Enum):
    """정보 유형 추정"""
    PRIVATE_INFO = "private_information"        # 사적 정보 유입
    PUBLIC_NEWS = "public_news"                 # 공개 뉴스 반응
    INSTITUTIONAL = "institutional"             # 기관 매매
    RETAIL_FOMO = "retail_fomo"                 # 개인 FOMO
    ACCUMULATION = "accumulation"               # 누적 매수 (기관)
    DISTRIBUTION = "distribution"               # 분산 매도 (기관)
    UNKNOWN = "unknown"                         # 불명


@dataclass
class VolumeAnomaly:
    """
    거래량 이상 감지 결과

    경제학적 의미:
    - volume_ratio: 평균 대비 배수 (3x = 정보 비대칭 가능성)
    - z_score: 통계적 유의성 (|z| > 2 = 95% 신뢰수준)
    - price_volume_correlation: 가격-거래량 상관관계
    """
    ticker: str
    timestamp: datetime

    # 거래량 지표
    current_volume: float
    avg_volume_20d: float
    volume_ratio: float              # 현재/평균
    z_score: float                   # 표준화 점수

    # 가격 지표
    price_change_1d: float           # 당일 가격 변동률
    price_change_5d: float           # 5일 가격 변동률

    # 분류
    anomaly_type: AnomalyType
    information_type: InformationType

    # 경고
    severity: str                    # LOW, MEDIUM, HIGH, CRITICAL
    alert_message: str

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'current_volume': self.current_volume,
            'avg_volume_20d': self.avg_volume_20d,
            'volume_ratio': self.volume_ratio,
            'z_score': self.z_score,
            'price_change_1d': self.price_change_1d,
            'price_change_5d': self.price_change_5d,
            'anomaly_type': self.anomaly_type.value,
            'information_type': self.information_type.value,
            'severity': self.severity,
            'alert_message': self.alert_message
        }


@dataclass
class TopMover:
    """거래량 상위 종목 (강제 감지용)"""
    ticker: str
    volume_ratio: float          # MA20 대비 거래량 비율
    price_change_1d: float       # 당일 가격 변동률
    current_volume: float        # 현재 거래량
    avg_volume_20d: float        # 20일 평균 거래량

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'volume_ratio': self.volume_ratio,
            'price_change_1d': self.price_change_1d,
            'current_volume': self.current_volume,
            'avg_volume_20d': self.avg_volume_20d
        }


@dataclass
class VolumeAnalysisResult:
    """거래량 분석 전체 결과"""
    timestamp: str
    total_tickers_analyzed: int
    anomalies_detected: int
    high_severity_count: int

    anomalies: List[VolumeAnomaly] = field(default_factory=list)

    # Top Movers (강제 감지 - 이상이 없어도 상위 3개 표시)
    top_movers: List[TopMover] = field(default_factory=list)
    top_movers_summary: str = ""

    # 시장 전체 거래량 지표
    market_volume_percentile: float = 50.0   # 시장 전체 거래량 백분위
    breadth_ratio: float = 0.0               # 이상 종목 비율

    # 요약
    summary: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'total_tickers_analyzed': self.total_tickers_analyzed,
            'anomalies_detected': self.anomalies_detected,
            'high_severity_count': self.high_severity_count,
            'anomalies': [a.to_dict() for a in self.anomalies],
            'top_movers': [m.to_dict() for m in self.top_movers],
            'top_movers_summary': self.top_movers_summary,
            'market_volume_percentile': self.market_volume_percentile,
            'breadth_ratio': self.breadth_ratio,
            'summary': self.summary,
            'warnings': self.warnings
        }


# =============================================================================
# Volume Analyzer
# =============================================================================

class VolumeAnalyzer:
    """
    거래량 기반 정보 비대칭 탐지기

    경제학적 근거:
    - Kyle (1985): 사적 정보(Private Information)가 유입되면
      정보 거래자가 시장에 진입하여 거래량이 먼저 반응
    - 합리적 기대 가설: 거래량 폭발은 참여자 간 기대 불일치 신호
    - 가격보다 거래량이 선행 지표로 작동

    탐지 기준:
    - 3x 이상: MEDIUM severity (Private Information 가능성)
    - 5x 이상: HIGH severity (강한 정보 비대칭)
    - 10x 이상: CRITICAL severity (극심한 이상)
    """

    def __init__(
        self,
        lookback_period: int = 20,           # 이동평균 기간
        surge_threshold_medium: float = 3.0,  # 3배 = MEDIUM
        surge_threshold_high: float = 5.0,    # 5배 = HIGH
        surge_threshold_critical: float = 10.0,  # 10배 = CRITICAL
        z_score_threshold: float = 2.0,       # 통계적 유의성
        verbose: bool = False
    ):
        """
        Args:
            lookback_period: 이동평균 계산 기간 (기본 20일)
            surge_threshold_medium: MEDIUM severity 임계값 (기본 3x)
            surge_threshold_high: HIGH severity 임계값 (기본 5x)
            surge_threshold_critical: CRITICAL severity 임계값 (기본 10x)
            z_score_threshold: Z-score 유의성 임계값 (기본 2.0 = 95%)
            verbose: 상세 로깅
        """
        self.lookback = lookback_period
        self.threshold_medium = surge_threshold_medium
        self.threshold_high = surge_threshold_high
        self.threshold_critical = surge_threshold_critical
        self.z_threshold = z_score_threshold
        self.verbose = verbose

    def _log(self, msg: str):
        """로깅"""
        if self.verbose:
            logger.info(msg)
            print(f"[VolumeAnalyzer] {msg}")

    def detect_anomalies(
        self,
        market_data: Dict[str, pd.DataFrame],
        include_crypto: bool = True
    ) -> VolumeAnalysisResult:
        """
        시장 데이터에서 거래량 이상 감지

        경제학적 근거:
        - 거래량 급증은 사적 정보(Private Information) 유입 신호
        - Kyle 모델: 정보 거래자 진입 시 거래량이 가격보다 먼저 반응

        Args:
            market_data: {ticker: DataFrame with 'Volume', 'Close' columns}
            include_crypto: 암호화폐 포함 여부

        Returns:
            VolumeAnalysisResult: 분석 결과
        """
        self._log(f"Analyzing {len(market_data)} tickers for volume anomalies...")

        anomalies = []
        all_volume_stats = []  # 모든 종목의 거래량 통계 (Top Movers용)
        analyzed_count = 0

        for ticker, df in market_data.items():
            # 암호화폐 제외 옵션
            if not include_crypto and ticker.endswith('-USD'):
                continue

            # 데이터 유효성 검사
            if not self._validate_data(df):
                continue

            analyzed_count += 1

            # 거래량 통계 계산 (모든 종목)
            vol_stats = self._calculate_volume_stats(ticker, df)
            if vol_stats:
                all_volume_stats.append(vol_stats)

            # 거래량 이상 감지 (임계값 이상만)
            anomaly = self._analyze_ticker(ticker, df)
            if anomaly:
                anomalies.append(anomaly)
                self._log(f"  ⚠ {ticker}: {anomaly.alert_message}")

        # 결과 정리
        high_severity = [a for a in anomalies if a.severity in ['HIGH', 'CRITICAL']]

        # 시장 전체 거래량 분석
        market_vol_percentile = self._calculate_market_volume_percentile(market_data)
        breadth_ratio = len(anomalies) / analyzed_count if analyzed_count > 0 else 0

        # Top Movers 계산 (항상 상위 3개, 이상이 없어도 표시)
        top_movers, top_movers_summary = self._calculate_top_movers(
            all_volume_stats, has_anomalies=len(anomalies) > 0
        )

        # 요약 생성
        summary = self._generate_summary(anomalies, analyzed_count, market_vol_percentile)

        # 경고 생성
        warnings = self._generate_warnings(anomalies, breadth_ratio)

        result = VolumeAnalysisResult(
            timestamp=datetime.now().isoformat(),
            total_tickers_analyzed=analyzed_count,
            anomalies_detected=len(anomalies),
            high_severity_count=len(high_severity),
            anomalies=sorted(anomalies, key=lambda x: x.volume_ratio, reverse=True),
            top_movers=top_movers,
            top_movers_summary=top_movers_summary,
            market_volume_percentile=market_vol_percentile,
            breadth_ratio=breadth_ratio,
            summary=summary,
            warnings=warnings
        )

        self._log(f"Analysis complete: {len(anomalies)} anomalies from {analyzed_count} tickers")

        return result

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """데이터 유효성 검사"""
        if df is None or len(df) < self.lookback + 5:
            return False
        if 'Volume' not in df.columns or 'Close' not in df.columns:
            return False
        return True

    def _calculate_volume_stats(
        self,
        ticker: str,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        개별 종목 거래량 통계 계산 (Top Movers용)

        모든 종목에 대해 거래량 비율을 계산하여 반환
        (이상 탐지와 별개로 Top Movers 표시에 사용)
        """
        try:
            volumes = df['Volume'].dropna()
            if len(volumes) < self.lookback:
                return None

            current_volume = volumes.iloc[-1]
            if hasattr(current_volume, 'item'):
                current_volume = current_volume.item()
            current_volume = float(current_volume)

            if current_volume <= 0:
                return None

            # 20일 이동평균
            vol_20d = volumes.iloc[-self.lookback-1:-1]
            avg_volume = vol_20d.mean()
            if hasattr(avg_volume, 'item'):
                avg_volume = avg_volume.item()
            avg_volume = float(avg_volume)

            if avg_volume <= 0:
                return None

            volume_ratio = current_volume / avg_volume

            # 가격 변동률
            prices = df['Close'].dropna()
            price_change_1d = 0.0
            if len(prices) >= 2:
                try:
                    price_change_1d = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100)
                except:
                    pass

            return {
                'ticker': ticker,
                'volume_ratio': round(volume_ratio, 3),
                'price_change_1d': round(price_change_1d, 2),
                'current_volume': current_volume,
                'avg_volume_20d': avg_volume
            }

        except Exception as e:
            logger.warning(f"Error calculating volume stats for {ticker}: {e}")
            return None

    def _calculate_top_movers(
        self,
        all_stats: List[Dict],
        has_anomalies: bool,
        top_n: int = 3
    ) -> Tuple[List[TopMover], str]:
        """
        Top Movers 계산 (강제 감지)

        이상이 감지되지 않아도 거래량 증가율 상위 종목을 표시

        소스 이론: "20일 이평선 대비 3~5배 거래량은 사적 정보 유입이다."
        -> 이상이 없어도 상위 종목은 모니터링 대상
        """
        if not all_stats:
            return [], "No volume data available"

        # 거래량 비율 기준 정렬 (^VIX 같은 거래량 0인 종목 제외)
        valid_stats = [s for s in all_stats if s['volume_ratio'] > 0]
        sorted_stats = sorted(valid_stats, key=lambda x: x['volume_ratio'], reverse=True)

        top_movers = []
        for stat in sorted_stats[:top_n]:
            top_movers.append(TopMover(
                ticker=stat['ticker'],
                volume_ratio=stat['volume_ratio'],
                price_change_1d=stat['price_change_1d'],
                current_volume=stat['current_volume'],
                avg_volume_20d=stat['avg_volume_20d']
            ))

        # 요약 메시지 생성
        if has_anomalies:
            summary = f"Significant anomalies detected. Top {len(top_movers)} by volume ratio shown."
        else:
            # 이상이 없을 때 강제 표시
            if top_movers:
                top_ticker = top_movers[0].ticker
                top_ratio = top_movers[0].volume_ratio
                summary = (
                    f"No significant anomaly, but watching top movers: "
                    f"{top_ticker} ({top_ratio:.2f}x), "
                    f"{top_movers[1].ticker if len(top_movers) > 1 else 'N/A'} "
                    f"({top_movers[1].volume_ratio:.2f}x)" if len(top_movers) > 1 else ""
                )
                summary = summary.rstrip(", ")
            else:
                summary = "No volume data to analyze"

        return top_movers, summary

    def _analyze_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[VolumeAnomaly]:
        """
        개별 종목 거래량 분석

        경제학적 근거:
        - 20일 이동평균 대비 현재 거래량 비율 계산
        - 3x 이상 = 정보 비대칭 가능성 (Private Information Inflow)
        - Z-score로 통계적 유의성 검정
        """
        try:
            # 거래량 데이터
            volumes = df['Volume'].dropna()
            if len(volumes) < self.lookback:
                return None

            current_volume = volumes.iloc[-1]
            if hasattr(current_volume, 'item'):
                current_volume = current_volume.item()
            current_volume = float(current_volume)

            # 0 거래량 무시
            if current_volume <= 0:
                return None

            # 20일 이동평균 및 표준편차
            vol_20d = volumes.iloc[-self.lookback-1:-1]  # 오늘 제외 직전 20일
            avg_volume = vol_20d.mean()
            std_volume = vol_20d.std()
            if hasattr(avg_volume, 'item'):
                avg_volume = avg_volume.item()
            if hasattr(std_volume, 'item'):
                std_volume = std_volume.item()
            avg_volume = float(avg_volume)
            std_volume = float(std_volume)

            if avg_volume <= 0:
                return None

            # 거래량 비율 및 Z-score
            volume_ratio = current_volume / avg_volume
            z_score = (current_volume - avg_volume) / std_volume if std_volume > 0 else 0

            # 임계값 미달 시 None 반환
            if volume_ratio < self.threshold_medium and abs(z_score) < self.z_threshold:
                return None

            # 가격 데이터
            prices = df['Close'].dropna()
            price_change_1d = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100) if len(prices) >= 2 else 0
            price_change_5d = float((prices.iloc[-1] / prices.iloc[-6] - 1) * 100) if len(prices) >= 6 else 0

            # 이상 유형 및 심각도 결정
            anomaly_type, severity = self._classify_anomaly(
                volume_ratio, z_score, price_change_1d, price_change_5d
            )

            # 정보 유형 추정
            info_type = self._estimate_information_type(
                volume_ratio, price_change_1d, price_change_5d
            )

            # 경고 메시지 생성
            alert_message = self._create_alert_message(
                ticker, volume_ratio, price_change_1d, info_type, severity
            )

            return VolumeAnomaly(
                ticker=ticker,
                timestamp=datetime.now(),
                current_volume=current_volume,
                avg_volume_20d=avg_volume,
                volume_ratio=round(volume_ratio, 2),
                z_score=round(z_score, 2),
                price_change_1d=round(price_change_1d, 2),
                price_change_5d=round(price_change_5d, 2),
                anomaly_type=anomaly_type,
                information_type=info_type,
                severity=severity,
                alert_message=alert_message
            )

        except Exception as e:
            logger.warning(f"Error analyzing {ticker}: {e}")
            return None

    def _classify_anomaly(
        self,
        volume_ratio: float,
        z_score: float,
        price_change_1d: float,
        price_change_5d: float
    ) -> Tuple[AnomalyType, str]:
        """
        이상 유형 및 심각도 분류

        경제학적 기준:
        - 3x~5x: MEDIUM - 잠재적 정보 비대칭
        - 5x~10x: HIGH - 강한 정보 비대칭
        - 10x+: CRITICAL - 극심한 이상 (긴급 주시)
        """
        # 심각도 결정
        if volume_ratio >= self.threshold_critical:
            severity = "CRITICAL"
        elif volume_ratio >= self.threshold_high:
            severity = "HIGH"
        elif volume_ratio >= self.threshold_medium:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # 이상 유형 결정
        if volume_ratio >= self.threshold_high:
            if abs(price_change_1d) < 0.5:
                # 거래량 급증 + 가격 정체 = 정보 비대칭
                anomaly_type = AnomalyType.VOLUME_PRICE_DIVERGENCE
            elif price_change_1d > 2:
                # 거래량 급증 + 가격 상승 = 매집
                anomaly_type = AnomalyType.ACCUMULATION
            elif price_change_1d < -2:
                # 거래량 급증 + 가격 하락 = 분배
                anomaly_type = AnomalyType.DISTRIBUTION
            else:
                anomaly_type = AnomalyType.EXTREME_SURGE
        elif volume_ratio >= self.threshold_medium:
            anomaly_type = AnomalyType.ABNORMAL_SURGE
        else:
            anomaly_type = AnomalyType.ABNORMAL_SURGE

        return anomaly_type, severity

    def _estimate_information_type(
        self,
        volume_ratio: float,
        price_change_1d: float,
        price_change_5d: float
    ) -> InformationType:
        """
        정보 유형 추정

        경제학적 근거:
        - 거래량 급증 + 가격 선행 = 사적 정보
        - 거래량 급증 + 가격 동시 반응 = 공개 뉴스
        - 지속적 거래량 = 기관 매매
        """
        if volume_ratio >= self.threshold_high:
            if abs(price_change_1d) < 1 and abs(price_change_5d) > 3:
                # 가격 선행 후 거래량 폭발 = 사적 정보
                return InformationType.PRIVATE_INFO
            elif abs(price_change_1d) > 3:
                # 동시 반응 = 공개 뉴스
                return InformationType.PUBLIC_NEWS

        if volume_ratio >= 3 and volume_ratio < 5:
            if price_change_1d > 2:
                return InformationType.RETAIL_FOMO

        if volume_ratio >= self.threshold_critical:
            return InformationType.INSTITUTIONAL

        return InformationType.UNKNOWN

    def _create_alert_message(
        self,
        ticker: str,
        volume_ratio: float,
        price_change_1d: float,
        info_type: InformationType,
        severity: str
    ) -> str:
        """경고 메시지 생성"""
        if severity in ['HIGH', 'CRITICAL']:
            base_msg = f"[{severity}] Private Information Inflow Detected"
        else:
            base_msg = f"[{severity}] Abnormal Volume Detected"

        detail_msg = f"{ticker}: {volume_ratio:.1f}x avg volume, price {price_change_1d:+.1f}%"

        if info_type == InformationType.PRIVATE_INFO:
            info_msg = "- Potential insider activity"
        elif info_type == InformationType.INSTITUTIONAL:
            info_msg = "- Large institutional order flow"
        elif info_type == InformationType.ACCUMULATION:
            info_msg = "- Smart money accumulation pattern"
        elif info_type == InformationType.DISTRIBUTION:
            info_msg = "- Distribution/profit-taking pattern"
        else:
            info_msg = ""

        return f"{base_msg}: {detail_msg} {info_msg}".strip()

    def _calculate_market_volume_percentile(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """시장 전체 거래량 백분위 계산"""
        try:
            total_volumes = []
            for ticker, df in market_data.items():
                if 'Volume' in df.columns and len(df) >= self.lookback:
                    # 최근 거래량 / 평균 거래량
                    current = df['Volume'].iloc[-1]
                    avg = df['Volume'].iloc[-self.lookback:].mean()

                    # Series를 scalar로 변환
                    if hasattr(current, 'item'):
                        current = current.item()
                    if hasattr(avg, 'item'):
                        avg = avg.item()

                    current = float(current)
                    avg = float(avg)

                    if avg > 0:
                        total_volumes.append(current / avg)

            if total_volumes:
                median_ratio = np.median(total_volumes)
                # 중위수 비율을 백분위로 변환 (1.0 = 50%)
                percentile = min(100, max(0, float(median_ratio) * 50))
                return round(percentile, 1)
        except Exception as e:
            logger.warning(f"Error calculating market volume: {e}")

        return 50.0

    def _generate_summary(
        self,
        anomalies: List[VolumeAnomaly],
        analyzed_count: int,
        market_percentile: float
    ) -> str:
        """
        분석 요약 생성

        소스 이론: "20일 이평선 대비 3~5배 거래량은 사적 정보(Private Info) 유입이다."
        (Kyle, 1985)
        """
        if not anomalies:
            # 명시적 메시지: 정상적인 거래량 프로필
            return (
                f"Volume profile is normal (No asymmetric info detected). "
                f"Analyzed {analyzed_count} tickers, all within normal range (<{self.threshold_medium}x MA20). "
                f"Market volume at {market_percentile:.0f}th percentile. "
                f"No evidence of Private Information Inflow (Kyle, 1985)."
            )

        high_count = len([a for a in anomalies if a.severity in ['HIGH', 'CRITICAL']])
        medium_count = len([a for a in anomalies if a.severity == 'MEDIUM'])
        top_anomaly = max(anomalies, key=lambda x: x.volume_ratio)

        # 경제학적 해석 추가
        if high_count > 0:
            economic_note = (
                f"⚠ Private Information Detected: {high_count} ticker(s) show {self.threshold_high}x+ volume surge. "
                f"Kyle(1985): Informed traders entering market."
            )
        elif medium_count > 0:
            economic_note = (
                f"📊 Potential Information Asymmetry: {medium_count} ticker(s) show {self.threshold_medium}x+ volume. "
                f"Monitor for price discovery."
            )
        else:
            economic_note = "Volume profile within normal bounds."

        return (
            f"Detected {len(anomalies)} volume anomalies in {analyzed_count} tickers. "
            f"{high_count} high-severity, {medium_count} medium-severity alerts. "
            f"Top: {top_anomaly.ticker} at {top_anomaly.volume_ratio:.1f}x average. "
            f"Market volume at {market_percentile:.0f}th percentile. "
            f"{economic_note}"
        )

    def _generate_warnings(
        self,
        anomalies: List[VolumeAnomaly],
        breadth_ratio: float
    ) -> List[str]:
        """경고 목록 생성"""
        warnings = []

        # 광범위한 이상 (10% 이상 종목에서 이상 발생)
        if breadth_ratio > 0.1:
            warnings.append(
                f"Broad market volume anomaly: {breadth_ratio:.1%} of tickers affected. "
                "Possible systematic event or index rebalancing."
            )

        # CRITICAL 경고
        critical = [a for a in anomalies if a.severity == 'CRITICAL']
        for c in critical[:3]:  # 최대 3개
            warnings.append(
                f"CRITICAL: {c.ticker} - {c.volume_ratio:.1f}x volume surge. "
                f"Private Information Inflow Detected. Immediate attention required."
            )

        # 정보 비대칭 경고
        private_info = [a for a in anomalies if a.information_type == InformationType.PRIVATE_INFO]
        if len(private_info) >= 3:
            tickers = ', '.join([a.ticker for a in private_info[:5]])
            warnings.append(
                f"Multiple potential insider activity detected: {tickers}"
            )

        return warnings

    def get_top_anomalies(
        self,
        result: VolumeAnalysisResult,
        n: int = 10
    ) -> List[VolumeAnomaly]:
        """
        상위 N개 이상 종목 반환

        경제학적 의미:
        - 거래량 비율 기준 정렬
        - 가장 심한 정보 비대칭 가능성 종목
        """
        return result.anomalies[:n]

    def filter_by_severity(
        self,
        result: VolumeAnalysisResult,
        min_severity: str = "HIGH"
    ) -> List[VolumeAnomaly]:
        """
        심각도 기준 필터링

        Args:
            min_severity: "LOW", "MEDIUM", "HIGH", "CRITICAL"
        """
        severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        min_level = severity_order.get(min_severity, 2)

        return [
            a for a in result.anomalies
            if severity_order.get(a.severity, 0) >= min_level
        ]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    from datetime import timedelta

    print("=" * 60)
    print("Volume Analyzer Test")
    print("=" * 60)

    # 테스트 데이터 수집
    tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'GOOGL']

    print("\n1. Fetching test data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    market_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                market_data[ticker] = data
                print(f"   {ticker}: {len(data)} days")
        except Exception as e:
            print(f"   {ticker}: Failed - {e}")

    # 분석 실행
    print("\n2. Running volume analysis...")
    analyzer = VolumeAnalyzer(verbose=True)
    result = analyzer.detect_anomalies(market_data)

    # 결과 출력
    print("\n3. Results:")
    print("-" * 50)
    print(f"Tickers analyzed: {result.total_tickers_analyzed}")
    print(f"Anomalies detected: {result.anomalies_detected}")
    print(f"High severity: {result.high_severity_count}")
    print(f"Market volume percentile: {result.market_volume_percentile:.1f}")
    print(f"Breadth ratio: {result.breadth_ratio:.1%}")
    print()
    print(f"Summary: {result.summary}")

    if result.warnings:
        print("\n4. Warnings:")
        for w in result.warnings:
            print(f"   ⚠ {w}")

    if result.anomalies:
        print("\n5. Top Anomalies:")
        for a in result.anomalies[:5]:
            print(f"   {a.ticker}: {a.volume_ratio:.1f}x, {a.severity}, {a.alert_message[:60]}...")

    print("\n" + "=" * 60)
    print("Test complete")
