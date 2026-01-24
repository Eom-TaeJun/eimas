"""
Proof-of-Index (PoI) Module
============================
블록체인 기반 투명한 금융 지수 시스템

경제학적 배경 (eco4.docx):
- 기존 금융지수 문제점:
  1. 계산 블랙박스 (S&P 500, NASDAQ 등)
  2. 정산 지연 (T+2)
  3. 신뢰성 검증 불가

- 블록체인 해결책:
  1. Proof-of-Index: 계산 과정 투명화
  2. SHA-256 해시: 데이터 무결성 검증
  3. Smart Contract: 자동 정산 (실시간)

핵심 개념:
1. Index Calculation: I_t = sum(P_i_t * Q_i_t) / D_t
2. On-chain Verification: Hash(weights) → Blockchain
3. Mean Reversion Strategy: Z-score 기반 퀀트 전략

활용:
- 탈중앙화 인덱스 펀드 (DeFi)
- 실시간 리밸런싱
- 거래소 간 차익거래 (Arbitrage)
- 신흥국 시장 접근성 개선

Author: EIMAS Team
References: eco4.docx
"""

import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IndexSnapshot:
    """인덱스 스냅샷 (특정 시점)"""
    timestamp: datetime
    index_value: float
    components: Dict[str, float]  # {ticker: price}
    weights: Dict[str, float]  # {ticker: weight}
    hash_value: str  # SHA-256 해시

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'index_value': self.index_value,
            'components': self.components,
            'weights': self.weights,
            'hash': self.hash_value
        }


@dataclass
class MeanReversionSignal:
    """Mean Reversion 신호"""
    timestamp: datetime
    current_price: float
    mean: float
    std: float
    z_score: float
    signal: str  # BUY/SELL/HOLD
    strength: float  # 0~1 (신호 강도)
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'mean': self.mean,
            'std': self.std,
            'z_score': self.z_score,
            'signal': self.signal,
            'strength': self.strength,
            'interpretation': self.interpretation
        }


@dataclass
class BacktestResult:
    """백테스트 결과"""
    initial_capital: float
    final_capital: float
    total_return: float  # 총 수익률
    annualized_return: float  # 연간 수익률
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float  # 승률
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }

    def summary(self) -> str:
        """요약 리포트"""
        return f"""
Proof-of-Index Backtest Summary
{'=' * 60}
Performance:
  Initial Capital:    ${self.initial_capital:,.2f}
  Final Capital:      ${self.final_capital:,.2f}
  Total Return:       {self.total_return:+.2%}
  Annualized Return:  {self.annualized_return:+.2%}
  Sharpe Ratio:       {self.sharpe_ratio:.2f}
  Max Drawdown:       {self.max_drawdown:.2%}

Trading:
  Total Trades:       {self.total_trades}
  Win Rate:           {self.win_rate:.1%}
  Winning Trades:     {self.winning_trades}
  Losing Trades:      {self.losing_trades}
  Avg Win:            ${self.avg_win:,.2f}
  Avg Loss:           ${self.avg_loss:,.2f}
  Win/Loss Ratio:     {abs(self.avg_win/self.avg_loss) if self.avg_loss != 0 else np.inf:.2f}
"""


# =============================================================================
# Proof-of-Index (PoI) Core
# =============================================================================

class ProofOfIndex:
    """
    Proof-of-Index (PoI): 블록체인 기반 투명한 인덱스 시스템

    경제학적 의미:
    - 전통 금융: S&P 500, NASDAQ (계산 불투명)
    - PoI: 모든 계산 과정을 블록체인에 기록 → 검증 가능

    장점:
    1. 투명성: 누구나 인덱스 계산 검증 가능
    2. 실시간: T+2 정산 → 즉시 정산
    3. 탈중앙화: 중앙 기관 없이 운영 가능
    4. 글로벌 접근성: 국경/통화 제약 없음

    사용 예시:
    - 크립토 시장 지수 (BTC-Dominance Index)
    - 토큰화 자산 지수 (RWA Index)
    - 신흥국 주식 지수 (접근성 개선)
    """

    def __init__(self, divisor: float = 1.0, name: str = "PoI Index"):
        """
        Args:
            divisor: 인덱스 제수 (Index Divisor)
                     - 주식 분할/배당 등으로 인한 조정
                     - 일반적으로 100.0 (S&P 500 스타일)
            name: 인덱스 이름
        """
        self.divisor = divisor
        self.name = name
        self.index_history: List[IndexSnapshot] = []

    def calculate_index(
        self,
        prices: Dict[str, float],
        quantities: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> IndexSnapshot:
        """
        인덱스 계산: I_t = sum(P_i_t * Q_i_t) / D_t

        경제학적 의미:
        - P_i_t: 자산 i의 가격 (시점 t)
        - Q_i_t: 자산 i의 수량 (시가총액 가중 or 동일 가중)
        - D_t: 제수 (Divisor, 조정용)

        Args:
            prices: {ticker: price}
            quantities: {ticker: quantity} (시가총액 가중용)
            timestamp: 계산 시점 (기본: 현재)

        Returns:
            IndexSnapshot: 인덱스 스냅샷 (값, 구성요소, 해시)

        Example:
            >>> poi = ProofOfIndex(divisor=100.0)
            >>> prices = {'BTC': 50000, 'ETH': 3000}
            >>> quantities = {'BTC': 1.0, 'ETH': 10.0}
            >>> snapshot = poi.calculate_index(prices, quantities)
            >>> print(f"Index: {snapshot.index_value:.2f}")
            >>> print(f"Hash: {snapshot.hash_value[:16]}...")
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 시가총액 계산
        market_caps = {
            ticker: prices[ticker] * quantities[ticker]
            for ticker in prices.keys()
        }

        total_market_cap = sum(market_caps.values())

        # 인덱스 값 = 총 시가총액 / 제수
        index_value = total_market_cap / self.divisor

        # 가중치 계산 (시가총액 비중)
        weights = {
            ticker: mc / total_market_cap
            for ticker, mc in market_caps.items()
        }

        # SHA-256 해시 생성 (검증용)
        hash_value = self.hash_index_weights(weights, timestamp)

        # 스냅샷 생성
        snapshot = IndexSnapshot(
            timestamp=timestamp,
            index_value=index_value,
            components=prices.copy(),
            weights=weights,
            hash_value=hash_value
        )

        # 히스토리 저장
        self.index_history.append(snapshot)

        return snapshot

    def hash_index_weights(
        self,
        weights: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        SHA-256 기반 가중치 해시 생성 (On-chain 검증용)

        경제학적 의미:
        - 블록체인에 가중치 해시를 기록 → 변조 방지
        - Smart Contract가 해시 검증 → 자동 정산

        Args:
            weights: {ticker: weight}
            timestamp: 계산 시점 (기본: 현재)

        Returns:
            hash_value: SHA-256 해시 문자열 (64자리 hex)

        Example:
            >>> weights = {'BTC': 0.6, 'ETH': 0.4}
            >>> hash_val = poi.hash_index_weights(weights)
            >>> print(f"Hash: {hash_val}")
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 사전 순서로 정렬하여 재현 가능성 보장
        sorted_weights = {k: weights[k] for k in sorted(weights.keys())}

        # JSON 직렬화 (타임스탬프 포함)
        data = {
            'timestamp': timestamp.isoformat(),
            'weights': sorted_weights,
            'name': self.name,
            'divisor': self.divisor
        }

        # JSON 문자열 → bytes
        data_str = json.dumps(data, sort_keys=True)
        data_bytes = data_str.encode('utf-8')

        # SHA-256 해시
        hash_object = hashlib.sha256(data_bytes)
        hash_value = hash_object.hexdigest()

        return hash_value

    def verify_on_chain(
        self,
        hash_value: str,
        reference_hash: str
    ) -> Dict[str, Any]:
        """
        Smart Contract 기반 해시 검증

        경제학적 의미:
        - On-chain 해시와 Off-chain 계산 비교
        - 일치 → 인덱스 계산 정확 (자동 정산)
        - 불일치 → 계산 오류 or 변조 (거래 차단)

        Args:
            hash_value: 계산된 해시
            reference_hash: On-chain 참조 해시 (블록체인 기록)

        Returns:
            verification_result:
                - is_valid: 검증 결과 (bool)
                - hash_calculated: 계산된 해시
                - hash_reference: 참조 해시
                - message: 메시지

        Example:
            >>> snapshot = poi.calculate_index(prices, quantities)
            >>> result = poi.verify_on_chain(snapshot.hash_value, on_chain_hash)
            >>> print(f"Valid: {result['is_valid']}")
        """
        is_valid = (hash_value == reference_hash)

        if is_valid:
            message = "✅ Hash verified. Index calculation is correct."
        else:
            message = "❌ Hash mismatch. Possible calculation error or tampering."

        return {
            'is_valid': is_valid,
            'hash_calculated': hash_value,
            'hash_reference': reference_hash,
            'message': message
        }

    def mean_reversion_signal(
        self,
        prices: pd.Series,
        window: int = 20,
        threshold: float = 2.0
    ) -> MeanReversionSignal:
        """
        Mean Reversion 퀀트 신호 생성

        경제학적 배경:
        - Mean Reversion: 가격은 평균으로 회귀한다
        - Z-score: 표준편차 기준 이탈 정도
        - Threshold: ±2σ → 95% 신뢰구간 벗어남

        전략:
        - Z < -threshold (예: -2): BUY (저평가)
        - Z > +threshold (예: +2): SELL (고평가)
        - |Z| < threshold: HOLD (정상 범위)

        Args:
            prices: 가격 시계열 (pd.Series)
            window: 이동평균 윈도우 (기본: 20일)
            threshold: Z-score 임계값 (기본: ±2.0)

        Returns:
            MeanReversionSignal: 신호 결과

        Example:
            >>> prices = pd.Series([100, 102, 98, 95, 105, 110])
            >>> signal = poi.mean_reversion_signal(prices, window=5)
            >>> print(f"Signal: {signal.signal}")
            >>> print(f"Z-score: {signal.z_score:.2f}")
        """
        # 이동평균 및 표준편차
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()

        # Z-score 계산
        z_score = (prices - mean) / std

        # 최신 값 (스칼라로 변환)
        def to_scalar(val):
            return float(val.item() if hasattr(val, 'item') else val)

        latest_price = to_scalar(prices.iloc[-1])
        latest_mean = to_scalar(mean.iloc[-1])
        latest_std = to_scalar(std.iloc[-1])
        latest_z = to_scalar(z_score.iloc[-1])

        # NaN 처리
        if np.isnan(latest_z):
            return MeanReversionSignal(
                timestamp=datetime.now(),
                current_price=latest_price,
                mean=np.nan,
                std=np.nan,
                z_score=np.nan,
                signal='HOLD',
                strength=0.0,
                interpretation="INSUFFICIENT_DATA: Window too short"
            )

        # 신호 생성
        if latest_z < -threshold:
            signal = 'BUY'
            strength = min(abs(latest_z) / threshold - 1.0, 1.0)  # 초과분 정규화
            interpretation = f"UNDERVALUED: Z={latest_z:.2f} (< -{threshold})"
        elif latest_z > threshold:
            signal = 'SELL'
            strength = min(abs(latest_z) / threshold - 1.0, 1.0)
            interpretation = f"OVERVALUED: Z={latest_z:.2f} (> +{threshold})"
        else:
            signal = 'HOLD'
            strength = abs(latest_z) / threshold  # 0~1 범위
            interpretation = f"NORMAL: Z={latest_z:.2f} (within ±{threshold})"

        return MeanReversionSignal(
            timestamp=datetime.now(),
            current_price=latest_price,
            mean=latest_mean,
            std=latest_std,
            z_score=latest_z,
            signal=signal,
            strength=strength,
            interpretation=interpretation
        )

    def backtest_strategy(
        self,
        prices: pd.Series,
        initial_capital: float = 100000,
        window: int = 20,
        threshold: float = 2.0,
        position_size: float = 1.0,
        transaction_cost: float = 0.001  # 0.1% 거래비용
    ) -> BacktestResult:
        """
        Mean Reversion 전략 백테스트

        전략 로직:
        1. Z-score < -threshold → BUY (전체 자금)
        2. Z-score > +threshold → SELL (포지션 청산)
        3. |Z-score| < threshold → HOLD

        Args:
            prices: 가격 시계열
            initial_capital: 초기 자본 ($)
            window: MA 윈도우
            threshold: Z-score 임계값
            position_size: 포지션 크기 (1.0 = 100%)
            transaction_cost: 거래비용 (0.001 = 0.1%)

        Returns:
            BacktestResult: 백테스트 결과 (수익률, Sharpe, 거래 내역)

        Example:
            >>> result = poi.backtest_strategy(btc_prices, initial_capital=10000)
            >>> print(result.summary())
        """
        # 이동평균 및 Z-score
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        z_score = (prices - mean) / std

        # 백테스트 변수
        capital = initial_capital
        position = 0.0  # 보유 수량
        cash = initial_capital
        equity_curve = []
        trades = []

        for i in range(window, len(prices)):
            current_price = prices.iloc[i]
            current_z = z_score.iloc[i]
            current_date = prices.index[i]

            # 포지션 가치
            position_value = position * current_price
            total_equity = cash + position_value
            equity_curve.append(total_equity)

            # NaN 스킵
            if np.isnan(current_z):
                continue

            # BUY 신호 (Z < -threshold, 현재 포지션 없음)
            if current_z < -threshold and position == 0:
                # 전체 자금으로 매수
                buy_amount = cash * position_size
                buy_quantity = buy_amount / current_price
                cost = buy_amount * transaction_cost

                position += buy_quantity
                cash -= (buy_amount + cost)

                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': buy_quantity,
                    'z_score': current_z,
                    'cash': cash,
                    'position': position
                })

            # SELL 신호 (Z > +threshold, 포지션 있음)
            elif current_z > threshold and position > 0:
                # 전체 포지션 청산
                sell_amount = position * current_price
                cost = sell_amount * transaction_cost

                cash += (sell_amount - cost)
                position = 0

                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'z_score': current_z,
                    'cash': cash,
                    'position': position
                })

        # 최종 정산 (포지션 청산)
        if position > 0:
            final_price = prices.iloc[-1]
            sell_amount = position * final_price
            cost = sell_amount * transaction_cost
            cash += (sell_amount - cost)
            position = 0

        final_capital = cash
        equity_series = pd.Series(equity_curve, index=prices.index[window:])

        # 성과 지표 계산
        total_return = (final_capital - initial_capital) / initial_capital

        # 연간 수익률 (일별 데이터 가정)
        n_days = len(prices) - window
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe Ratio
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Max Drawdown
        cumulative = equity_series / equity_series.expanding().max()
        max_drawdown = (cumulative.min() - 1.0)

        # 승률
        winning_trades = sum(1 for t in trades if t['action'] == 'SELL' and len([t2 for t2 in trades if t2['action'] == 'BUY' and t2['date'] < t['date']]) > 0)
        total_trades = len([t for t in trades if t['action'] == 'SELL'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 평균 승/패
        profits = []
        losses = []
        for i, trade in enumerate(trades):
            if trade['action'] == 'SELL' and i > 0:
                prev_buy = [t for t in trades[:i] if t['action'] == 'BUY']
                if prev_buy:
                    buy_price = prev_buy[-1]['price']
                    sell_price = trade['price']
                    pnl = sell_price - buy_price
                    if pnl > 0:
                        profits.append(pnl)
                    else:
                        losses.append(pnl)

        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0

        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_series,
            trades=trades
        )

    def get_index_history(self, as_dataframe: bool = True) -> pd.DataFrame:
        """
        인덱스 히스토리 반환

        Args:
            as_dataframe: True시 DataFrame 반환

        Returns:
            history: 인덱스 시계열 데이터
        """
        if not self.index_history:
            return pd.DataFrame()

        if as_dataframe:
            data = {
                'timestamp': [s.timestamp for s in self.index_history],
                'index_value': [s.index_value for s in self.index_history],
                'hash': [s.hash_value for s in self.index_history]
            }
            return pd.DataFrame(data).set_index('timestamp')
        else:
            return self.index_history


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_poi_backtest(
    prices: pd.Series,
    initial_capital: float = 100000,
    window: int = 20,
    threshold: float = 2.0
) -> Dict[str, Any]:
    """PoI 백테스트 간편 함수"""
    poi = ProofOfIndex()
    result = poi.backtest_strategy(prices, initial_capital, window, threshold)
    return result.to_dict()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Proof-of-Index (PoI) Module Test")
    print("=" * 60)

    # 1. Index Calculation Test
    print("\n[1] Index Calculation Test")
    print("-" * 40)

    poi = ProofOfIndex(divisor=100.0, name="Crypto Index")

    prices = {
        'BTC': 50000,
        'ETH': 3000,
        'SOL': 100,
        'AVAX': 30
    }

    quantities = {
        'BTC': 1.0,
        'ETH': 10.0,
        'SOL': 100.0,
        'AVAX': 1000.0
    }

    snapshot = poi.calculate_index(prices, quantities)

    print(f"  Index Name: {poi.name}")
    print(f"  Index Value: {snapshot.index_value:.2f}")
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"\n  Weights:")
    for ticker, weight in sorted(snapshot.weights.items(), key=lambda x: -x[1]):
        print(f"    {ticker}: {weight:.1%}")
    print(f"\n  Hash (SHA-256): {snapshot.hash_value[:32]}...")

    # 2. Hash Verification Test
    print("\n[2] Hash Verification Test")
    print("-" * 40)

    # 동일한 가중치로 해시 재생성
    reference_hash = poi.hash_index_weights(snapshot.weights, snapshot.timestamp)
    verification = poi.verify_on_chain(snapshot.hash_value, reference_hash)

    print(f"  {verification['message']}")
    print(f"  Calculated: {verification['hash_calculated'][:32]}...")
    print(f"  Reference:  {verification['hash_reference'][:32]}...")

    # 변조된 해시 테스트
    tampered_hash = "0" * 64
    verification2 = poi.verify_on_chain(snapshot.hash_value, tampered_hash)
    print(f"\n  Tampered Test: {verification2['message']}")

    # 3. Mean Reversion Signal Test
    print("\n[3] Mean Reversion Signal Test")
    print("-" * 40)

    # 시뮬레이션 가격 데이터 (평균 회귀 패턴)
    np.random.seed(42)
    n = 100
    base_price = 100
    noise = np.random.randn(n) * 5
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 20
    prices_series = pd.Series(base_price + trend + noise)

    signal = poi.mean_reversion_signal(prices_series, window=20, threshold=2.0)

    print(f"  Current Price: ${signal.current_price:.2f}")
    print(f"  Mean (20-day): ${signal.mean:.2f}")
    print(f"  Std Dev: ${signal.std:.2f}")
    print(f"  Z-score: {signal.z_score:.2f}")
    print(f"  Signal: {signal.signal}")
    print(f"  Strength: {signal.strength:.1%}")
    print(f"  Interpretation: {signal.interpretation}")

    # 4. Backtest Test
    print("\n[4] Backtest Strategy Test")
    print("-" * 40)

    # 더 긴 시뮬레이션 (1년)
    n = 252
    trend = np.sin(np.linspace(0, 8 * np.pi, n)) * 30
    noise = np.random.randn(n) * 10
    prices_long = pd.Series(100 + trend + noise, index=pd.date_range('2024-01-01', periods=n))

    result = poi.backtest_strategy(
        prices_long,
        initial_capital=100000,
        window=20,
        threshold=2.0
    )

    print(result.summary())

    # 5. Index History Test
    print("\n[5] Index History Test")
    print("-" * 40)

    # 여러 시점 인덱스 계산
    for i in range(5):
        # 가격 변동 시뮬레이션
        prices_updated = {k: v * (1 + np.random.randn() * 0.05) for k, v in prices.items()}
        snapshot = poi.calculate_index(prices_updated, quantities)

    history_df = poi.get_index_history()
    print(f"  Total snapshots: {len(history_df)}")
    print(f"\n  Latest 3 snapshots:")
    print(history_df.tail(3))

    print("\n" + "=" * 60)
    print("Proof-of-Index Test Complete!")
    print("=" * 60)
