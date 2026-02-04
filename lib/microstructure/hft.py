#!/usr/bin/env python3
"""
Microstructure - HFT Indicators (eco3.docx)
============================================================

고빈도 거래 미세구조 지표

Economic Foundation:
    - Tick Rule: Lee & Ready (1991) trade classification
    - Kyle's Lambda: Kyle (1985) price impact measure
    - Volume Clock: OHLC volatility adjustment
    - Quote Stuffing: Market manipulation detection (O'Hara 2015)

Functions:
    - tick_rule_classification: Tick rule for trade direction
    - kyles_lambda: Kyle's lambda (price impact)
    - volume_clock_sampling: Volume-based time bars
    - detect_quote_stuffing: Quote stuffing detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HFT Microstructure Indicators (eco3.docx)
# ============================================================================

def tick_rule_classification(prices: pd.Series) -> pd.Series:
    """
    Tick Rule: 거래 방향 분류 (Buy/Sell/Neutral)

    경제학적 배경:
    - Lee & Ready (1991) 알고리즘
    - 체결가격 변화로 매수/매도 압력 추정
    - HFT 환경에서 Order Flow 방향 파악

    Rule:
    - p[t] > p[t-1]: b[t] = 1 (Buy-initiated)
    - p[t] < p[t-1]: b[t] = -1 (Sell-initiated)
    - p[t] = p[t-1]: b[t] = b[t-1] (이전 방향 유지)

    Args:
        prices: 체결가격 시계열 (pd.Series)

    Returns:
        b: 거래 방향 시계열 (1=Buy, -1=Sell, 0=Unknown)

    References:
        Lee, C. M. C., & Ready, M. J. (1991). Inferring Trade Direction from
        Intraday Data. The Journal of Finance, 46(2), 733-746.

    Example:
        >>> prices = pd.Series([100, 101, 101, 100, 99])
        >>> directions = tick_rule_classification(prices)
        >>> print(directions)
        0     1  (첫 거래는 Buy 가정)
        1     1  (100 → 101: Buy)
        2     1  (101 → 101: 이전 유지)
        3    -1  (101 → 100: Sell)
        4    -1  (100 → 99: Sell)
    """
    b = pd.Series(index=prices.index, dtype=int)

    # 초기값: Buy 방향 가정 (중립적 가정)
    b.iloc[0] = 1

    # Tick Rule 적용
    # prices를 numpy array로 변환 (MultiIndex 문제 방지)
    prices_values = prices.values if hasattr(prices, 'values') else prices

    for i in range(1, len(prices)):
        curr_price = prices_values[i] if isinstance(prices_values[i], (int, float)) else prices_values[i].item() if hasattr(prices_values[i], 'item') else float(prices_values[i])
        prev_price = prices_values[i-1] if isinstance(prices_values[i-1], (int, float)) else prices_values[i-1].item() if hasattr(prices_values[i-1], 'item') else float(prices_values[i-1])

        if curr_price > prev_price:
            b.iloc[i] = 1  # Buy
        elif curr_price < prev_price:
            b.iloc[i] = -1  # Sell
        else:
            b.iloc[i] = b.iloc[i-1]  # 이전 방향 유지

    return b


def kyles_lambda(price_changes: pd.Series,
                 signed_volume: pd.Series,
                 return_details: bool = False) -> Dict[str, float]:
    """
    Kyle's Lambda: Market Impact 계수 추정

    경제학적 배경:
    - Kyle (1985) 모델: 정보거래자의 시장 충격 측정
    - Lambda = 단위 주문 플로우당 가격 변화
    - 높은 Lambda = 시장 깊이 얕음, 높은 가격 충격

    모델:
        ΔP[t] = λ × (b[t] × V[t]) + ε[t]

        where:
        - ΔP[t]: 가격 변화
        - b[t]: 거래 방향 (+1/-1, tick_rule_classification 사용)
        - V[t]: 거래량
        - λ: Kyle's Lambda (추정 대상)
        - ε[t]: 오차항

    추정 방법: OLS 회귀 (Ordinary Least Squares)

    Args:
        price_changes: 가격 변화율 또는 절대 변화 (pd.Series)
        signed_volume: 부호화된 거래량 = b[t] × V[t] (pd.Series)
        return_details: True시 회귀 상세 결과 반환

    Returns:
        결과 딕셔너리:
        - lambda: Kyle's Lambda 값
        - r_squared: 결정계수 (모델 설명력)
        - n_observations: 관측치 수
        - interpretation: 해석 (HIGH/MEDIUM/LOW impact)

        (return_details=True인 경우 추가)
        - intercept: 절편 (정보 없는 가격 드리프트)
        - std_error: 표준오차

    References:
        Kyle, A. S. (1985). Continuous Auctions and Insider Trading.
        Econometrica, 53(6), 1315-1335.

    Example:
        >>> prices = pd.Series([100, 101, 102, 101, 100])
        >>> volumes = pd.Series([10, 15, 20, 10, 5])
        >>> directions = tick_rule_classification(prices)
        >>> price_changes = prices.diff()
        >>> signed_vol = directions * volumes
        >>> result = kyles_lambda(price_changes, signed_vol)
        >>> print(f"Lambda: {result['lambda']:.6f}")
        >>> print(f"R²: {result['r_squared']:.3f}")
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # NaN 제거 (DataFrame/Series 호환)
    # Series를 1D array로 변환 후 결합
    pc_values = price_changes.values.flatten() if hasattr(price_changes, 'values') else np.array(price_changes).flatten()
    sv_values = signed_volume.values.flatten() if hasattr(signed_volume, 'values') else np.array(signed_volume).flatten()

    # NaN mask 생성
    mask = ~(np.isnan(pc_values) | np.isnan(sv_values))

    X = sv_values[mask].reshape(-1, 1)
    y = pc_values[mask]

    if len(X) < 10:
        # 관측치 부족
        return {
            'lambda': np.nan,
            'r_squared': np.nan,
            'n_observations': len(X),
            'interpretation': 'INSUFFICIENT_DATA'
        }

    # OLS 회귀
    model = LinearRegression()
    model.fit(X, y)

    lambda_value = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # 해석
    abs_lambda = abs(lambda_value)
    if abs_lambda > 0.001:
        interpretation = 'HIGH_IMPACT'  # 높은 시장 충격
    elif abs_lambda > 0.0001:
        interpretation = 'MEDIUM_IMPACT'
    else:
        interpretation = 'LOW_IMPACT'  # 깊은 유동성

    result = {
        'lambda': float(lambda_value),
        'r_squared': float(r_squared),
        'n_observations': len(X),
        'interpretation': interpretation
    }

    if return_details:
        # 표준오차 계산
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse / len(X))

        result.update({
            'intercept': float(intercept),
            'std_error': float(std_error),
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals))
        })

    return result


def volume_clock_sampling(df: pd.DataFrame,
                          volume_bucket: float,
                          volume_col: str = 'volume',
                          timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Volume Clock Sampling: 거래량 기준 동기화 샘플링

    경제학적 배경:
    - 시간 기준(Time bars) → 거래 활동 무시
    - Volume bars → 시장 활동 동기화
    - VPIN 정확도 향상 (Easley et al., 2012)

    원리:
    - 누적 거래량이 volume_bucket에 도달할 때마다 샘플링
    - 거래량 많은 시기 → 샘플 빈도 증가
    - 거래량 적은 시기 → 샘플 빈도 감소

    Args:
        df: OHLCV 데이터프레임 (시간 순 정렬)
            필수 컬럼: timestamp, volume, (open, high, low, close)
        volume_bucket: 각 버킷의 누적 거래량 (예: 1,000,000 주)
        volume_col: 거래량 컬럼명 (기본: 'volume')
        timestamp_col: 타임스탬프 컬럼명 (기본: 'timestamp')

    Returns:
        sampled_df: Volume 기준으로 샘플링된 데이터프레임
            추가 컬럼:
            - bucket_id: 버킷 번호 (0, 1, 2, ...)
            - cumulative_volume: 누적 거래량

    References:
        Easley, D., López de Prado, M. M., & O'Hara, M. (2012).
        Flow Toxicity and Liquidity in a High-Frequency World.
        The Review of Financial Studies, 25(5), 1457-1493.

    Example:
        >>> # 시간 기준 데이터
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        ...     'close': np.random.randn(100).cumsum() + 100,
        ...     'volume': np.random.randint(100, 1000, 100)
        ... })
        >>>
        >>> # Volume Clock 샘플링 (10,000 주 단위)
        >>> sampled = volume_clock_sampling(df, volume_bucket=10000)
        >>> print(f"Original samples: {len(df)}")
        >>> print(f"Volume samples: {len(sampled)}")
        >>> print(f"Buckets: {sampled['bucket_id'].max() + 1}")
    """
    df = df.copy()
    
    # 결측치 및 무한대 처리 (Volume)
    if volume_col in df.columns:
        df[volume_col] = df[volume_col].fillna(0)
        df = df[np.isfinite(df[volume_col])]

    # 누적 거래량 계산
    df['cumulative_volume'] = df[volume_col].cumsum()

    # 버킷 ID 할당
    df['bucket_id'] = (df['cumulative_volume'] / volume_bucket).fillna(0).astype(int)

    # 각 버킷의 마지막 거래 선택 (시간상 가장 최근)
    # groupby().last()는 각 그룹의 마지막 행 선택
    sampled = df.groupby('bucket_id').last().reset_index(drop=False)

    # bucket_id를 인덱스로 유지하지 않고 컬럼으로 변환
    sampled = sampled.reset_index(drop=True)

    return sampled


def detect_quote_stuffing(order_data: pd.DataFrame,
                          cancel_threshold: float = 0.9,
                          action_col: str = 'action',
                          window: Optional[str] = None) -> Dict[str, Any]:
    """
    Quote Stuffing 탐지: 주문 취소율 기반 이상 거래 식별

    경제학적 배경:
    - Quote Stuffing: 대량 주문 제출 후 즉시 취소 (시장 교란)
    - HFT 전략: 경쟁자 지연 유도, 정보 왜곡
    - SEC 규제 대상 (Market Manipulation)

    탐지 기준:
    - 주문 취소율 > 90% → Quote Stuffing 의심
    - Flash Crash (2010) 전 Quote Stuffing 급증 관찰

    Args:
        order_data: 주문 데이터 (pd.DataFrame)
            필수 컬럼:
            - order_id: 주문 고유 ID
            - action: 'submit', 'cancel', 'execute' 등
            - timestamp: 주문 시각 (옵션, window 사용 시 필수)
        cancel_threshold: 주문 취소율 임계값 (기본: 0.9 = 90%)
        action_col: 주문 액션 컬럼명 (기본: 'action')
        window: 시간 윈도우 (예: '1min', '5min', None=전체)

    Returns:
        탐지 결과 딕셔너리:
        - is_stuffing: Quote Stuffing 여부 (bool)
        - cancel_rate: 주문 취소율 (0~1)
        - total_orders: 총 주문 수
        - canceled_orders: 취소된 주문 수
        - executed_orders: 체결된 주문 수
        - cancel_execute_ratio: 취소/체결 비율
        - severity: 심각도 (NONE/LOW/MEDIUM/HIGH/CRITICAL)

    References:
        Egginton, J. F., Van Ness, B. F., & Van Ness, R. A. (2016).
        Quote stuffing. Financial Management, 45(3), 583-608.

    Example:
        >>> # 시뮬레이션 주문 데이터
        >>> orders = pd.DataFrame({
        ...     'order_id': range(1000),
        ...     'action': ['submit'] * 500 + ['cancel'] * 450 + ['execute'] * 50,
        ...     'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1s')
        ... })
        >>>
        >>> result = detect_quote_stuffing(orders)
        >>> print(f"Is Quote Stuffing: {result['is_stuffing']}")
        >>> print(f"Cancel Rate: {result['cancel_rate']:.1%}")
        >>> print(f"Severity: {result['severity']}")
    """
    if window is not None:
        # 시간 윈도우별로 분석
        if 'timestamp' not in order_data.columns:
            raise ValueError("timestamp column required when window is specified")

        # 시간 윈도우 그룹화
        order_data = order_data.copy()
        order_data['time_bucket'] = pd.to_datetime(order_data['timestamp']).dt.floor(window)

        results = []
        for bucket, group in order_data.groupby('time_bucket'):
            bucket_result = _detect_stuffing_single(group, cancel_threshold, action_col)
            bucket_result['time_bucket'] = bucket
            results.append(bucket_result)

        # 전체 결과 + 윈도우별 결과
        overall = _detect_stuffing_single(order_data, cancel_threshold, action_col)
        overall['window_results'] = results
        return overall

    else:
        # 전체 데이터 분석
        return _detect_stuffing_single(order_data, cancel_threshold, action_col)


def _detect_stuffing_single(order_data: pd.DataFrame,
                            cancel_threshold: float,
                            action_col: str) -> Dict[str, Any]:
    """Quote Stuffing 탐지 (단일 데이터셋)"""

    total_orders = len(order_data)

    if total_orders == 0:
        return {
            'is_stuffing': False,
            'cancel_rate': 0.0,
            'total_orders': 0,
            'canceled_orders': 0,
            'executed_orders': 0,
            'cancel_execute_ratio': np.nan,
            'severity': 'NONE'
        }

    # 액션별 카운트
    action_counts = order_data[action_col].value_counts()

    canceled_orders = action_counts.get('cancel', 0)
    executed_orders = action_counts.get('execute', 0)

    # 취소율 계산
    cancel_rate = canceled_orders / total_orders

    # 취소/체결 비율
    if executed_orders > 0:
        cancel_execute_ratio = canceled_orders / executed_orders
    else:
        cancel_execute_ratio = np.inf if canceled_orders > 0 else 0.0

    # Quote Stuffing 판정
    is_stuffing = cancel_rate > cancel_threshold

    # 심각도 평가
    if cancel_rate >= 0.95:
        severity = 'CRITICAL'  # 95%+ 취소
    elif cancel_rate >= 0.90:
        severity = 'HIGH'  # 90-95% 취소
    elif cancel_rate >= 0.80:
        severity = 'MEDIUM'  # 80-90% 취소
    elif cancel_rate >= 0.70:
        severity = 'LOW'  # 70-80% 취소
    else:
        severity = 'NONE'  # < 70% 취소

    return {
        'is_stuffing': bool(is_stuffing),
        'cancel_rate': float(cancel_rate),
        'total_orders': int(total_orders),
        'canceled_orders': int(canceled_orders),
        'executed_orders': int(executed_orders),
        'cancel_execute_ratio': float(cancel_execute_ratio) if np.isfinite(cancel_execute_ratio) else None,
        'severity': severity
    }


