#!/usr/bin/env python3
"""
EIMAS Backtesting Engine
========================
시그널 기반 전략의 과거 성과 검증

주요 기능:
1. 시그널 → 포지션 → 수익률 계산
2. 성과 지표 (Sharpe, MDD, Win Rate)
3. 다양한 전략 지원
4. 벤치마크 대비 분석

사용법:
    from lib.backtester import Backtester, Strategy

    # 전략 정의
    strategy = Strategy(
        name="VIX_Mean_Reversion",
        signal_func=my_signal_function,
        position_size=1.0
    )

    # 백테스트 실행
    bt = Backtester(strategy, start_date="2023-01-01")
    result = bt.run()
    bt.print_report(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Enums & Constants
# ============================================================================

class PositionType(str, Enum):
    """포지션 유형"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalType(str, Enum):
    """시그널 유형"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# 기본 수수료/슬리피지
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005   # 0.05%


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Trade:
    """개별 거래 기록"""
    entry_date: str
    exit_date: str
    ticker: str
    direction: PositionType
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    holding_days: int
    signal_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'direction': self.direction.value
        }


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float

    # 수익률
    total_return: float           # 총 수익률 %
    annual_return: float          # 연환산 수익률 %
    benchmark_return: float       # 벤치마크 수익률 %
    alpha: float                  # 초과 수익률 %

    # 리스크
    volatility: float             # 연환산 변동성 %
    max_drawdown: float           # 최대 낙폭 %
    max_drawdown_duration: int    # 최대 낙폭 기간 (일)

    # 리스크 조정 수익
    sharpe_ratio: float           # 샤프 비율
    sortino_ratio: float          # 소르티노 비율
    calmar_ratio: float           # 칼마 비율

    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float               # 승률 %
    avg_win: float                # 평균 수익 %
    avg_loss: float               # 평균 손실 %
    profit_factor: float          # 수익 팩터
    avg_holding_days: float       # 평균 보유 기간

    # 상세 데이터
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_return': round(self.total_return, 2),
            'annual_return': round(self.annual_return, 2),
            'benchmark_return': round(self.benchmark_return, 2),
            'alpha': round(self.alpha, 2),
            'volatility': round(self.volatility, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'trades': [t.to_dict() for t in self.trades],
        }


# ============================================================================
# Strategy Class
# ============================================================================

@dataclass
class Strategy:
    """
    전략 정의

    signal_func: (data: pd.DataFrame, idx: int) -> SignalType
        - data: 전체 데이터프레임
        - idx: 현재 인덱스
        - return: BUY, SELL, HOLD
    """
    name: str
    signal_func: Callable[[pd.DataFrame, int], SignalType]
    ticker: str = "SPY"
    position_size: float = 1.0        # 포지션 크기 (0-1)
    stop_loss: Optional[float] = None  # Stop loss % (예: 0.05 = 5%)
    take_profit: Optional[float] = None  # Take profit %
    max_holding_days: Optional[int] = None  # 최대 보유 기간


# ============================================================================
# Backtester Class
# ============================================================================

class Backtester:
    """
    백테스팅 엔진

    사용법:
        bt = Backtester(strategy, start_date="2023-01-01")
        result = bt.run()
        bt.print_report(result)
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000,
        benchmark: str = "SPY",
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        data: pd.DataFrame = None,  # 외부 데이터 제공 가능
    ):
        self.strategy = strategy
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.commission = commission
        self.slippage = slippage
        self.external_data = data

        self._data: pd.DataFrame = None
        self._benchmark_data: pd.Series = None

    def _load_data(self) -> bool:
        """데이터 로드"""
        print(f"Loading data for {self.strategy.ticker}...")

        if self.external_data is not None:
            self._data = self.external_data
            return True

        try:
            # 전략 자산
            df = yf.download(
                self.strategy.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            if df.empty:
                print(f"No data for {self.strategy.ticker}")
                return False

            # yfinance 멀티인덱스 컬럼 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self._data = df

            # 벤치마크
            if self.benchmark != self.strategy.ticker:
                bench = yf.download(
                    self.benchmark,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                if isinstance(bench.columns, pd.MultiIndex):
                    bench.columns = bench.columns.get_level_values(0)
                self._benchmark_data = bench['Close']
            else:
                self._benchmark_data = df['Close']

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """성과 지표 계산"""
        returns = equity_curve.pct_change().dropna()

        # 총 수익률
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

        # 연환산 수익률
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

        # 변동성 (연환산)
        volatility = returns.std() * np.sqrt(252) * 100

        # 최대 낙폭
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        # MDD 기간
        dd_duration = 0
        max_dd_duration = 0
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Sharpe Ratio (rf = 0 가정)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Sortino Ratio (하방 변동성만)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return / 100) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 거래 통계
        if trades:
            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl <= 0]

            win_rate = len(winning) / len(trades) * 100 if trades else 0
            avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
            avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

            total_profit = sum(t.pnl for t in winning) if winning else 0
            total_loss = abs(sum(t.pnl for t in losing)) if losing else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_holding = np.mean([t.holding_days for t in trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding = 0
            winning = []
            losing = []

        # 벤치마크
        if self._benchmark_data is not None and len(self._benchmark_data) > 1:
            bench_return = (self._benchmark_data.iloc[-1] / self._benchmark_data.iloc[0] - 1) * 100
        else:
            bench_return = 0

        alpha = total_return - bench_return

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'benchmark_return': bench_return,
            'alpha': alpha,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
        }

    def run(self) -> BacktestResult:
        """백테스트 실행"""
        if not self._load_data():
            raise ValueError("Failed to load data")

        print(f"Running backtest: {self.strategy.name}")
        print(f"  Period: {self.start_date} ~ {self.end_date}")
        print(f"  Ticker: {self.strategy.ticker}")

        # 초기화
        capital = self.initial_capital
        position = PositionType.FLAT
        entry_price = 0.0
        entry_date = None
        shares = 0.0
        trades: List[Trade] = []
        equity_history = []

        # 데이터 순회
        for i in range(len(self._data)):
            date = self._data.index[i]
            close = float(self._data['Close'].iloc[i])
            high = float(self._data['High'].iloc[i])
            low = float(self._data['Low'].iloc[i])

            # 현재 포지션 가치
            if position == PositionType.LONG:
                position_value = shares * close
            elif position == PositionType.SHORT:
                position_value = shares * (2 * entry_price - close)  # Short P&L
            else:
                position_value = 0

            current_equity = capital + position_value
            equity_history.append({'date': date, 'equity': current_equity})

            # 포지션이 있을 때 - Exit 조건 확인
            if position != PositionType.FLAT:
                exit_signal = False
                exit_reason = ""

                # Stop Loss
                if self.strategy.stop_loss:
                    if position == PositionType.LONG and low <= entry_price * (1 - self.strategy.stop_loss):
                        exit_signal = True
                        exit_reason = "Stop Loss"
                    elif position == PositionType.SHORT and high >= entry_price * (1 + self.strategy.stop_loss):
                        exit_signal = True
                        exit_reason = "Stop Loss"

                # Take Profit
                if self.strategy.take_profit and not exit_signal:
                    if position == PositionType.LONG and high >= entry_price * (1 + self.strategy.take_profit):
                        exit_signal = True
                        exit_reason = "Take Profit"
                    elif position == PositionType.SHORT and low <= entry_price * (1 - self.strategy.take_profit):
                        exit_signal = True
                        exit_reason = "Take Profit"

                # Max Holding Days
                if self.strategy.max_holding_days and not exit_signal:
                    holding_days = (date - entry_date).days
                    if holding_days >= self.strategy.max_holding_days:
                        exit_signal = True
                        exit_reason = "Max Holding"

                # 시그널 기반 Exit
                if not exit_signal:
                    signal = self.strategy.signal_func(self._data, i)
                    if position == PositionType.LONG and signal == SignalType.SELL:
                        exit_signal = True
                        exit_reason = "Signal Sell"
                    elif position == PositionType.SHORT and signal == SignalType.BUY:
                        exit_signal = True
                        exit_reason = "Signal Cover"

                # Exit 실행
                if exit_signal:
                    exit_price = close * (1 - self.slippage if position == PositionType.LONG else 1 + self.slippage)

                    if position == PositionType.LONG:
                        pnl = (exit_price - entry_price) * shares
                    else:  # SHORT
                        pnl = (entry_price - exit_price) * shares

                    pnl -= abs(exit_price * shares * self.commission)  # 수수료

                    capital += shares * entry_price + pnl  # 원금 + 손익
                    pnl_pct = (pnl / (shares * entry_price)) * 100

                    trade = Trade(
                        entry_date=entry_date.strftime("%Y-%m-%d"),
                        exit_date=date.strftime("%Y-%m-%d"),
                        ticker=self.strategy.ticker,
                        direction=position,
                        entry_price=round(entry_price, 2),
                        exit_price=round(exit_price, 2),
                        shares=round(shares, 4),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        holding_days=(date - entry_date).days,
                        signal_reason=exit_reason,
                    )
                    trades.append(trade)

                    position = PositionType.FLAT
                    shares = 0
                    entry_price = 0

            # 포지션이 없을 때 - Entry 조건 확인
            if position == PositionType.FLAT and i < len(self._data) - 1:
                signal = self.strategy.signal_func(self._data, i)

                if signal == SignalType.BUY:
                    entry_price = close * (1 + self.slippage)
                    invest_amount = capital * self.strategy.position_size
                    shares = invest_amount / entry_price
                    cost = shares * entry_price * self.commission
                    capital -= invest_amount + cost
                    position = PositionType.LONG
                    entry_date = date

                elif signal == SignalType.SELL:
                    entry_price = close * (1 - self.slippage)
                    invest_amount = capital * self.strategy.position_size
                    shares = invest_amount / entry_price
                    cost = shares * entry_price * self.commission
                    capital -= cost  # Short은 증거금만
                    position = PositionType.SHORT
                    entry_date = date

        # 마지막 포지션 청산
        if position != PositionType.FLAT:
            close = float(self._data['Close'].iloc[-1])
            exit_price = close

            if position == PositionType.LONG:
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            capital += shares * entry_price + pnl

            trade = Trade(
                entry_date=entry_date.strftime("%Y-%m-%d"),
                exit_date=self._data.index[-1].strftime("%Y-%m-%d"),
                ticker=self.strategy.ticker,
                direction=position,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                shares=round(shares, 4),
                pnl=round(pnl, 2),
                pnl_pct=round((pnl / (shares * entry_price)) * 100, 2),
                holding_days=(self._data.index[-1] - entry_date).days,
                signal_reason="End of Backtest",
            )
            trades.append(trade)

        # Equity Curve 생성
        equity_df = pd.DataFrame(equity_history)
        equity_curve = equity_df.set_index('date')['equity']

        # Drawdown Curve
        rolling_max = equity_curve.expanding().max()
        drawdown_curve = (equity_curve - rolling_max) / rolling_max * 100

        # 성과 지표 계산
        metrics = self._calculate_metrics(equity_curve, trades)

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            **metrics
        )

    def print_report(self, result: BacktestResult):
        """결과 리포트 출력"""
        print("\n" + "=" * 70)
        print(f"BACKTEST REPORT: {result.strategy_name}")
        print("=" * 70)

        print(f"\n[Period]")
        print(f"  {result.start_date} ~ {result.end_date}")
        print(f"  Initial Capital: ${result.initial_capital:,.0f}")
        print(f"  Final Capital:   ${result.final_capital:,.0f}")

        print(f"\n[Returns]")
        print(f"  Total Return:      {result.total_return:+.2f}%")
        print(f"  Annual Return:     {result.annual_return:+.2f}%")
        print(f"  Benchmark Return:  {result.benchmark_return:+.2f}%")
        print(f"  Alpha:             {result.alpha:+.2f}%")

        print(f"\n[Risk]")
        print(f"  Volatility:        {result.volatility:.2f}%")
        print(f"  Max Drawdown:      {result.max_drawdown:.2f}%")
        print(f"  MDD Duration:      {result.max_drawdown_duration} days")

        print(f"\n[Risk-Adjusted]")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:     {result.sortino_ratio:.2f}")
        print(f"  Calmar Ratio:      {result.calmar_ratio:.2f}")

        print(f"\n[Trade Statistics]")
        print(f"  Total Trades:      {result.total_trades}")
        print(f"  Winning Trades:    {result.winning_trades}")
        print(f"  Losing Trades:     {result.losing_trades}")
        print(f"  Win Rate:          {result.win_rate:.1f}%")
        print(f"  Avg Win:           {result.avg_win:+.2f}%")
        print(f"  Avg Loss:          {result.avg_loss:.2f}%")
        print(f"  Profit Factor:     {result.profit_factor:.2f}")
        print(f"  Avg Holding:       {result.avg_holding_days:.1f} days")

        # 최근 거래 5개
        if result.trades:
            print(f"\n[Recent Trades]")
            for trade in result.trades[-5:]:
                direction = "LONG" if trade.direction == PositionType.LONG else "SHORT"
                pnl_str = f"+{trade.pnl:,.0f}" if trade.pnl > 0 else f"{trade.pnl:,.0f}"
                print(f"  {trade.entry_date} → {trade.exit_date}: {direction} ${pnl_str} ({trade.pnl_pct:+.1f}%)")

        print("\n" + "=" * 70)


# ============================================================================
# Built-in Strategies
# ============================================================================

def create_ma_crossover_strategy(
    short_period: int = 20,
    long_period: int = 50,
    ticker: str = "SPY"
) -> Strategy:
    """이동평균 교차 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < long_period:
            return SignalType.HOLD

        close = data['Close'].values  # numpy array로 변환
        ma_short = float(np.mean(close[idx-short_period+1:idx+1]))
        ma_long = float(np.mean(close[idx-long_period+1:idx+1]))
        ma_short_prev = float(np.mean(close[idx-short_period:idx]))
        ma_long_prev = float(np.mean(close[idx-long_period:idx]))

        # 골든 크로스
        if ma_short > ma_long and ma_short_prev <= ma_long_prev:
            return SignalType.BUY
        # 데드 크로스
        elif ma_short < ma_long and ma_short_prev >= ma_long_prev:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name=f"MA_Crossover_{short_period}_{long_period}",
        signal_func=signal_func,
        ticker=ticker,
        position_size=1.0,
    )


def create_rsi_strategy(
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    ticker: str = "SPY"
) -> Strategy:
    """RSI 역발상 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < period + 1:
            return SignalType.HOLD

        close = data['Close'].values  # numpy array로 변환
        delta = np.diff(close, prepend=close[0])

        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)

        avg_gain = float(np.mean(gain[idx-period+1:idx+1]))
        avg_loss = float(np.mean(loss[idx-period+1:idx+1]))

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 과매도 → 매수
        if rsi < oversold:
            return SignalType.BUY
        # 과매수 → 매도
        elif rsi > overbought:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name=f"RSI_{period}_{int(oversold)}_{int(overbought)}",
        signal_func=signal_func,
        ticker=ticker,
        position_size=1.0,
    )


def create_vix_regime_strategy(ticker: str = "SPY") -> Strategy:
    """VIX 레짐 기반 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        # VIX 데이터가 필요 (별도 로드)
        # 여기서는 단순화하여 변동성 기반
        if idx < 20:
            return SignalType.HOLD

        close = data['Close'].values  # numpy array로 변환
        returns = np.diff(close) / close[:-1]
        vol_20 = float(np.std(returns[max(0, idx-20):idx]) * np.sqrt(252) * 100)

        # 저변동성 → 매수, 고변동성 → 매도
        if vol_20 < 15:  # 저변동성
            return SignalType.BUY
        elif vol_20 > 25:  # 고변동성
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="VIX_Regime",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.8,
        stop_loss=0.05,
    )


def create_fear_greed_contrarian_strategy(ticker: str = "SPY") -> Strategy:
    """Fear & Greed 역발상 전략 (단순화)"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 30:
            return SignalType.HOLD

        close = data['Close'].values  # numpy array로 변환

        # RSI + 최근 변동성으로 Fear/Greed 추정
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = float(np.mean(gain[idx-13:idx+1]))
        avg_loss = float(np.mean(loss[idx-13:idx+1]))
        rsi = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 50.0

        # 30일 수익률
        ret_30d = (close[idx] / close[idx-30] - 1) * 100

        # Extreme Fear (RSI < 30, 30일 수익률 < -5%) → 매수
        if rsi < 30 and ret_30d < -5:
            return SignalType.BUY
        # Extreme Greed (RSI > 70, 30일 수익률 > 10%) → 매도
        elif rsi > 70 and ret_30d > 10:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="Fear_Greed_Contrarian",
        signal_func=signal_func,
        ticker=ticker,
        position_size=1.0,
        take_profit=0.10,
        stop_loss=0.07,
    )


# ============================================================================
# EIMAS Signal Strategies
# ============================================================================

def create_yield_curve_strategy(ticker: str = "SPY") -> Strategy:
    """
    금리 곡선 역전 전략
    - 10Y-2Y 스프레드 대용: TLT/SHY 비율 사용
    - 역전 시 방어적, 정상화 시 공격적
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 60:
            return SignalType.HOLD

        close = data['Close'].values

        # 60일 모멘텀으로 경기 사이클 추정
        mom_60 = (close[idx] / close[idx-60] - 1) * 100
        mom_20 = (close[idx] / close[idx-20] - 1) * 100

        # 변동성
        returns = np.diff(close) / close[:-1]
        vol_20 = float(np.std(returns[max(0, idx-20):idx]) * np.sqrt(252) * 100)

        # 경기 침체 시그널: 모멘텀 하락 + 변동성 상승
        if mom_60 < -5 and mom_20 < 0 and vol_20 > 20:
            return SignalType.SELL  # 방어적
        # 경기 회복 시그널: 모멘텀 상승 + 변동성 정상화
        elif mom_60 > 5 and mom_20 > 0 and vol_20 < 20:
            return SignalType.BUY
        else:
            return SignalType.HOLD

    return Strategy(
        name="Yield_Curve_Proxy",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.8,
        stop_loss=0.08,
    )


def create_copper_gold_strategy(ticker: str = "SPY") -> Strategy:
    """
    Copper/Gold 비율 전략 (경기 선행 지표)
    - 구리 상승/금 하락 = Risk-On → 주식 매수
    - 구리 하락/금 상승 = Risk-Off → 주식 매도
    - 여기서는 모멘텀 기반 대용지표 사용
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 40:
            return SignalType.HOLD

        close = data['Close'].values

        # 20일, 40일 모멘텀
        mom_20 = (close[idx] / close[idx-20] - 1) * 100
        mom_40 = (close[idx] / close[idx-40] - 1) * 100

        # 단기 모멘텀 상승 + 장기 추세 확인 → Risk-On
        if mom_20 > 2 and mom_40 > 0:
            return SignalType.BUY
        # 단기 모멘텀 하락 + 장기 추세 하락 → Risk-Off
        elif mom_20 < -2 and mom_40 < 0:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="Copper_Gold_Proxy",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.8,
        max_holding_days=60,
    )


def create_regime_based_strategy(ticker: str = "SPY") -> Strategy:
    """
    EIMAS 레짐 기반 전략
    - Bull + Low Vol: 공격적 롱
    - Bull + High Vol: 보수적 롱
    - Bear + Low Vol: 관망
    - Bear + High Vol: 방어적 숏
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 50:
            return SignalType.HOLD

        close = data['Close'].values

        # 추세 판단: 50일 이평선 대비 위치 + 20일 모멘텀
        ma_50 = float(np.mean(close[idx-49:idx+1]))
        current_price = close[idx]
        trend_bullish = current_price > ma_50

        mom_20 = (close[idx] / close[idx-20] - 1) * 100
        mom_positive = mom_20 > 0

        # 변동성 판단: 20일 변동성
        returns = np.diff(close) / close[:-1]
        vol_20 = float(np.std(returns[max(0, idx-20):idx]) * np.sqrt(252) * 100)
        low_vol = vol_20 < 18

        # 레짐 판단
        if trend_bullish and mom_positive:
            if low_vol:
                return SignalType.BUY  # Bull + Low Vol: 공격적
            else:
                return SignalType.HOLD  # Bull + High Vol: 관망 (기존 포지션 유지)
        elif not trend_bullish and not mom_positive:
            if not low_vol:
                return SignalType.SELL  # Bear + High Vol: 방어적
            else:
                return SignalType.HOLD  # Bear + Low Vol: 관망
        else:
            return SignalType.HOLD  # Transition

    return Strategy(
        name="EIMAS_Regime",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.7,
        stop_loss=0.06,
        take_profit=0.12,
    )


def create_vix_mean_reversion_strategy(ticker: str = "SPY") -> Strategy:
    """
    VIX Mean Reversion 전략
    - VIX 급등 후 정상화 시 매수
    - VIX 극저점에서 급등 시 매도
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 30:
            return SignalType.HOLD

        close = data['Close'].values

        # VIX 대용: 최근 변동성과 평균 변동성 비교
        returns = np.diff(close) / close[:-1]
        vol_5 = float(np.std(returns[max(0, idx-5):idx]) * np.sqrt(252) * 100)
        vol_20 = float(np.std(returns[max(0, idx-20):idx]) * np.sqrt(252) * 100)
        vol_60 = float(np.std(returns[max(0, idx-60):idx]) * np.sqrt(252) * 100) if idx >= 60 else vol_20

        # VIX 스파이크 후 정상화: 5일 vol이 20일보다 낮고, 20일이 60일보다 낮으면 매수
        if vol_5 < vol_20 * 0.8 and vol_20 < vol_60 * 1.1:
            return SignalType.BUY
        # VIX 급등: 5일 vol이 20일보다 50% 이상 높으면 매도
        elif vol_5 > vol_20 * 1.5:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="VIX_Mean_Reversion",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.8,
        stop_loss=0.05,
        max_holding_days=20,
    )


def create_multi_factor_strategy(ticker: str = "SPY") -> Strategy:
    """
    다중 팩터 전략 (EIMAS 종합)
    - 모멘텀 + 변동성 + 추세 종합
    - 3개 이상 팩터 일치 시 진입
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if idx < 60:
            return SignalType.HOLD

        close = data['Close'].values

        bullish_factors = 0
        bearish_factors = 0

        # Factor 1: 모멘텀 (20일)
        mom_20 = (close[idx] / close[idx-20] - 1) * 100
        if mom_20 > 3:
            bullish_factors += 1
        elif mom_20 < -3:
            bearish_factors += 1

        # Factor 2: 추세 (50일 이평선)
        ma_50 = float(np.mean(close[idx-49:idx+1]))
        if close[idx] > ma_50 * 1.02:
            bullish_factors += 1
        elif close[idx] < ma_50 * 0.98:
            bearish_factors += 1

        # Factor 3: 변동성
        returns = np.diff(close) / close[:-1]
        vol_20 = float(np.std(returns[max(0, idx-20):idx]) * np.sqrt(252) * 100)
        if vol_20 < 15:
            bullish_factors += 1
        elif vol_20 > 25:
            bearish_factors += 1

        # Factor 4: RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = float(np.mean(gain[idx-13:idx+1]))
        avg_loss = float(np.mean(loss[idx-13:idx+1]))
        rsi = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 50.0

        if rsi < 40:  # 과매도 구간 = 반등 기대 = 매수 시그널
            bullish_factors += 1
        elif rsi > 60:  # 과매수 구간 = 조정 기대 = 매도 시그널
            bearish_factors += 1

        # 판정
        if bullish_factors >= 3:
            return SignalType.BUY
        elif bearish_factors >= 3:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="Multi_Factor",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.8,
        stop_loss=0.07,
        take_profit=0.15,
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Backtester Test")
    print("=" * 70)

    # EIMAS 전략만 테스트
    strategies = [
        create_regime_based_strategy("SPY"),
        create_vix_mean_reversion_strategy("SPY"),
        create_multi_factor_strategy("SPY"),
        create_yield_curve_strategy("SPY"),
        create_copper_gold_strategy("SPY"),
    ]

    results = []

    for strategy in strategies:
        print(f"\n{'='*70}")
        bt = Backtester(
            strategy=strategy,
            start_date="2020-01-01",  # 더 긴 기간 (COVID 포함)
            end_date="2024-12-31",
            initial_capital=100000,
        )

        try:
            result = bt.run()
            bt.print_report(result)
            results.append(result)
        except Exception as e:
            print(f"Error running {strategy.name}: {e}")
            import traceback
            traceback.print_exc()

    # 전략 비교
    if results:
        print("\n" + "=" * 70)
        print("EIMAS STRATEGY COMPARISON")
        print("=" * 70)
        print(f"\n{'Strategy':<25} {'Return':>10} {'Annual':>10} {'Sharpe':>8} {'MDD':>8} {'Trades':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.strategy_name:<25} {r.total_return:>+9.1f}% {r.annual_return:>+9.1f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown:>7.1f}% {r.total_trades:>8}")

        # 최고 성과 전략
        best = max(results, key=lambda x: x.sharpe_ratio)
        print(f"\n🏆 Best Sharpe: {best.strategy_name} ({best.sharpe_ratio:.2f})")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
