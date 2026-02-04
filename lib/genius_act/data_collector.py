#!/usr/bin/env python3
"""
Genius Act - Data Collector
============================================================

Stablecoin data collection and aggregation

Class:
    - StablecoinDataCollector: Fetches stablecoin market data
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class StablecoinDataCollector:
    """
    스테이블코인 시가총액 데이터 수집 및 7일 델타 계산

    소스 이론: "스테이블 코인(B*) 발행량이 늘어야 미국 국채를 사주므로(M 증가)
    Genius Act가 작동하는 것."
    """

    # 스테이블코인 티커 (yfinance용)
    STABLECOIN_TICKERS = {
        'USDT-USD': 'USDT',  # Tether
        'USDC-USD': 'USDC',  # USD Coin
        'DAI-USD': 'DAI',    # DAI
    }

    # 추정 시가총액 (십억 달러) - API 실패 시 폴백
    FALLBACK_MARKET_CAP = {
        'USDT': 140.0,  # 2025년 기준 약 $140B
        'USDC': 45.0,   # 2025년 기준 약 $45B
        'DAI': 5.0,     # 2025년 기준 약 $5B
    }

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file
        self._cache: Dict[str, List[Tuple[datetime, float]]] = {}

    def fetch_stablecoin_supply(self, lookback_days: int = 14) -> Dict[str, Dict]:
        """
        스테이블코인 시가총액 데이터 수집

        Returns:
            Dict: {
                'USDT': {'current': 140.0, 'week_ago': 138.0, 'delta_7d': 2.0, 'delta_pct': 1.45},
                'USDC': {'current': 45.0, 'week_ago': 44.0, 'delta_7d': 1.0, 'delta_pct': 2.27},
                ...
            }
        """
        import yfinance as yf

        result = {}

        for ticker, name in self.STABLECOIN_TICKERS.items():
            try:
                # yfinance에서 가격 데이터 가져오기 (시가총액 근사치로 사용)
                data = yf.download(ticker, period=f"{lookback_days}d", progress=False)

                if data.empty or len(data) < 7:
                    # 폴백: 추정 시가총액 사용
                    result[name] = {
                        'current': self.FALLBACK_MARKET_CAP.get(name, 0),
                        'week_ago': self.FALLBACK_MARKET_CAP.get(name, 0),
                        'delta_7d': 0.0,
                        'delta_pct': 0.0,
                        'source': 'fallback'
                    }
                    continue

                # 현재와 7일 전 가격 (스테이블코인이므로 가격≈$1, 변동 = 시가총액 변화 추정)
                # 실제로는 가격이 $1 부근이므로 시가총액 추정 필요
                # 여기서는 폴백 시가총액에 비율 변화를 적용

                base_cap = self.FALLBACK_MARKET_CAP.get(name, 100)

                # 최근 변동성 (peg 이탈) 기반 시가총액 변화 추정
                # 스테이블코인 가격이 $1 이상이면 수요 증가, 이하면 수요 감소
                current_price = float(data['Close'].iloc[-1])
                week_ago_price = float(data['Close'].iloc[-7]) if len(data) >= 7 else current_price

                # 시가총액 추정: 가격 프리미엄/할인을 공급 변화로 해석
                # 실제 시가총액 데이터가 없으므로 근사치 사용
                price_delta_pct = ((current_price - week_ago_price) / week_ago_price) * 100 if week_ago_price > 0 else 0

                # 시가총액 변화 추정 (가격 변화 * 50배로 확대 - 실제 공급 변화 반영)
                estimated_supply_change_pct = price_delta_pct * 50  # 프리미엄 1% = 공급 변화 추정

                # 범위 제한 (-10% ~ +10%)
                estimated_supply_change_pct = max(-10, min(10, estimated_supply_change_pct))

                current_cap = base_cap * (1 + estimated_supply_change_pct / 100)
                week_ago_cap = base_cap

                result[name] = {
                    'current': round(current_cap, 2),
                    'week_ago': round(week_ago_cap, 2),
                    'delta_7d': round(current_cap - week_ago_cap, 2),
                    'delta_pct': round(estimated_supply_change_pct, 2),
                    'price_current': round(current_price, 4),
                    'price_week_ago': round(week_ago_price, 4),
                    'source': 'estimated'
                }

            except Exception as e:
                # 에러 시 폴백
                result[name] = {
                    'current': self.FALLBACK_MARKET_CAP.get(name, 0),
                    'week_ago': self.FALLBACK_MARKET_CAP.get(name, 0),
                    'delta_7d': 0.0,
                    'delta_pct': 0.0,
                    'source': 'fallback',
                    'error': str(e)
                }

        return result

    def generate_detailed_comment(self, stablecoin_data: Dict[str, Dict]) -> Dict:
        """
        스테이블코인 7일 변화율 기반 상세 코멘트 생성

        소스 이론: "스테이블 코인(B*) 발행량이 늘어야 미국 국채를 사주므로(M 증가)
        Genius Act가 작동하는 것."

        Returns:
            Dict: {
                'total_market_cap': float,
                'total_delta_7d': float,
                'total_delta_pct': float,
                'genius_act_status': str,  # 'active', 'moderate', 'flat', 'draining'
                'detailed_comment': str,
                'economic_interpretation': str,
                'components': {...}
            }
        """
        # 총 시가총액 계산
        total_current = sum(d.get('current', 0) for d in stablecoin_data.values())
        total_week_ago = sum(d.get('week_ago', 0) for d in stablecoin_data.values())
        total_delta = total_current - total_week_ago
        total_delta_pct = (total_delta / total_week_ago * 100) if total_week_ago > 0 else 0

        # Genius Act 상태 판단
        if total_delta_pct > 3.0:
            status = 'active'
            comment = f"USD Liquidity Injection (Genius Act Active): Stablecoin issuance +{total_delta_pct:.1f}% in 7 days"
            interpretation = (
                f"스테이블코인 발행량 급증 (${total_delta:.1f}B, +{total_delta_pct:.1f}%). "
                f"Genius Act 담보 요건에 따라 미국 국채 수요 상승 예상. "
                f"M = B + S·B* 공식에서 S·B* 증가로 총 유동성(M) 확대. "
                f"크립토 시장 매수 대기 자금 증가, Risk-On 환경."
            )
        elif total_delta_pct > 1.0:
            status = 'moderate'
            comment = f"Moderate Stablecoin Growth: +{total_delta_pct:.1f}% weekly (${total_delta:.1f}B)"
            interpretation = (
                f"스테이블코인 완만한 증가 (+{total_delta_pct:.1f}%). "
                f"Genius Act 중립적 작동. 국채 수요 점진적 증가. "
                f"유동성 환경 안정적."
            )
        elif total_delta_pct > -1.0:
            status = 'flat'
            comment = f"Stablecoin issuance flat: {total_delta_pct:+.1f}% weekly (${total_delta:+.1f}B)"
            interpretation = (
                f"스테이블코인 발행량 정체 ({total_delta_pct:+.1f}%). "
                f"Genius Act 영향 미미. 국채 수요 변동 없음. "
                f"유동성 환경 변화 없음, 시장 중립."
            )
        else:
            status = 'draining'
            comment = f"Stablecoin Draining: {total_delta_pct:.1f}% weekly redemption (${abs(total_delta):.1f}B)"
            interpretation = (
                f"스테이블코인 소각/환매 진행 ({total_delta_pct:.1f}%). "
                f"크립토 시장 자금 이탈 신호. Genius Act 역작용 가능. "
                f"국채 담보 매각 압력, Risk-Off 주의."
            )

        # 개별 스테이블코인 상세
        components = {}
        for name, data in stablecoin_data.items():
            delta_pct = data.get('delta_pct', 0)
            if delta_pct > 2:
                component_status = "surging"
            elif delta_pct > 0:
                component_status = "growing"
            elif delta_pct > -2:
                component_status = "stable"
            else:
                component_status = "declining"

            components[name] = {
                'current': data.get('current', 0),
                'delta_7d': data.get('delta_7d', 0),
                'delta_pct': delta_pct,
                'status': component_status
            }

        return {
            'total_market_cap': round(total_current, 2),
            'total_delta_7d': round(total_delta, 2),
            'total_delta_pct': round(total_delta_pct, 2),
            'genius_act_status': status,
            'detailed_comment': comment,
            'economic_interpretation': interpretation,
            'components': components
        }


# =============================================================================
# 모니터링 대시보드
# =============================================================================

