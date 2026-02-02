#!/usr/bin/env python3
"""
EIMAS Backtesting Engine v2.0
=============================
시그널 기반 전략의 과거 성과 검증

v2.0 변경사항 (2026-01-31):
- 복리 계산 버그 수정 (고정 포지션 사이징)
- Look-ahead bias 방지
- DB 저장 기능 추가
- 기본 기간: 2024-09-01 ~ 현재

사용법:
    from lib.backtester import Backtester, Strategy

    strategy = Strategy(
        name="My_Strategy",
        signal_func=my_signal_function,
        position_size=0.3
    )

    bt = Backtester(strategy)
    result = bt.run()
    bt.print_report(result)
    bt.save_to_db(result)  # DB 저장
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
import json

warnings.filterwarnings('ignore')


# ============================================================================
# Enums & Constants
# ============================================================================

class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionSizingMode(str, Enum):
    """포지션 사이징 모드"""
    FIXED = "fixed"           # 고정 금액 (초기 자본 기준)
    PERCENTAGE = "percentage"  # 현재 자본의 %
    KELLY = "kelly"           # Kelly Criterion


# 기본 수수료/슬리피지
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005   # 0.05%
DEFAULT_SHORT_BORROW_RATE = 0.003  # 연 0.3%

# 기본 기간 (2024년 9월 Fed 금리 인하 시작)
DEFAULT_START_DATE = "2024-09-01"


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
    commission: float = 0.0
    slippage_cost: float = 0.0
    short_cost: float = 0.0
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

    # 비용 분석
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_short_cost: float = 0.0

    # 상세 데이터
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    parameters: Dict = field(default_factory=dict)

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
            'total_commission': round(self.total_commission, 2),
            'total_slippage': round(self.total_slippage, 2),
            'total_short_cost': round(self.total_short_cost, 2),
            'parameters': self.parameters,
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
        - data: 현재까지의 데이터 (look-ahead bias 방지)
        - idx: 현재 인덱스
        - return: BUY, SELL, HOLD
    """
    name: str
    signal_func: Callable[[pd.DataFrame, int], SignalType]
    ticker: str = "SPY"
    position_size: float = 0.3        # 포지션 크기 (0-1)
    sizing_mode: PositionSizingMode = PositionSizingMode.FIXED
    stop_loss: Optional[float] = None  # Stop loss % (예: 0.05 = 5%)
    take_profit: Optional[float] = None  # Take profit %
    max_holding_days: Optional[int] = None  # 최대 보유 기간


# ============================================================================
# Transaction Cost Model
# ============================================================================

class TransactionCostModel:
    """거래 비용 모델"""

    def __init__(
        self,
        commission: float = DEFAULT_COMMISSION,
        base_slippage: float = DEFAULT_SLIPPAGE,
        short_borrow_rate: float = DEFAULT_SHORT_BORROW_RATE
    ):
        self.commission = commission
        self.base_slippage = base_slippage
        self.short_borrow_rate = short_borrow_rate

    def calculate_slippage(
        self,
        trade_value: float,
        avg_volume: float,
        volatility: float = 0.0
    ) -> float:
        """
        거래량 및 변동성 기반 슬리피지 계산

        Args:
            trade_value: 거래 금액
            avg_volume: 평균 거래량 (금액)
            volatility: 현재 변동성 (%)
        """
        if avg_volume == 0:
            return self.base_slippage

        # 거래 비중에 따른 슬리피지
        volume_ratio = trade_value / avg_volume
        if volume_ratio < 0.01:
            volume_factor = 1.0
        elif volume_ratio < 0.05:
            volume_factor = 2.0
        else:
            volume_factor = 5.0

        # 변동성에 따른 슬리피지 증가
        vol_factor = 1.0 + (volatility / 100) if volatility > 20 else 1.0

        return self.base_slippage * volume_factor * vol_factor

    def calculate_short_cost(
        self,
        notional: float,
        days: int,
        hard_to_borrow: bool = False
    ) -> float:
        """숏 비용 계산"""
        rate = self.short_borrow_rate * 10 if hard_to_borrow else self.short_borrow_rate
        return notional * rate * days / 365

    def calculate_commission(self, trade_value: float) -> float:
        """수수료 계산"""
        return abs(trade_value) * self.commission


# ============================================================================
# Backtester Class
# ============================================================================

class Backtester:
    """
    백테스팅 엔진 v2.0

    주요 변경:
    - 고정 포지션 사이징 (복리 버그 수정)
    - Look-ahead bias 방지
    - DB 저장 기능
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000,
        benchmark: str = "SPY",
        cost_model: TransactionCostModel = None,
        data: pd.DataFrame = None,
    ):
        self.strategy = strategy
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.cost_model = cost_model or TransactionCostModel()
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
            df = yf.download(
                self.strategy.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            if df.empty:
                print(f"No data for {self.strategy.ticker}")
                return False

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 거래량 금액 계산 (Market Impact용)
            df['Volume_Value'] = df['Close'] * df['Volume']
            df['Avg_Volume_20'] = df['Volume_Value'].rolling(20).mean()

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

    def _calculate_position_size(self, capital: float, idx: int) -> float:
        """
        포지션 크기 계산

        FIXED 모드: 초기 자본 기준 고정 금액
        PERCENTAGE 모드: 현재 자본 기준 (기존 방식, 복리 효과)
        KELLY 모드: Kelly Criterion 기반
        """
        if self.strategy.sizing_mode == PositionSizingMode.FIXED:
            # 항상 초기 자본 기준으로 고정 (복리 버그 수정)
            return self.initial_capital * self.strategy.position_size

        elif self.strategy.sizing_mode == PositionSizingMode.PERCENTAGE:
            # 현재 자본 기준 (기존 방식 - 주의 필요)
            return capital * self.strategy.position_size

        elif self.strategy.sizing_mode == PositionSizingMode.KELLY:
            # Kelly Criterion (과거 성과 기반)
            # 여기서는 간단히 고정 사이즈의 절반 사용
            return self.initial_capital * self.strategy.position_size * 0.5

        return self.initial_capital * self.strategy.position_size

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
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

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

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return / 100) / downside_std if downside_std > 0 else 0

        # Calmar Ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 거래 통계
        total_commission = 0.0
        total_slippage = 0.0
        total_short_cost = 0.0

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

            total_commission = sum(t.commission for t in trades)
            total_slippage = sum(t.slippage_cost for t in trades)
            total_short_cost = sum(t.short_cost for t in trades)
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
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_short_cost': total_short_cost,
        }

    def run(self) -> BacktestResult:
        """백테스트 실행"""
        if not self._load_data():
            raise ValueError("Failed to load data")

        print(f"Running backtest: {self.strategy.name}")
        print(f"  Period: {self.start_date} ~ {self.end_date}")
        print(f"  Ticker: {self.strategy.ticker}")
        print(f"  Position Size: {self.strategy.position_size * 100:.0f}% ({self.strategy.sizing_mode.value})")

        # 초기화
        capital = self.initial_capital
        position = PositionType.FLAT
        entry_price = 0.0
        entry_date = None
        shares = 0.0
        trades: List[Trade] = []
        equity_history = []
        position_value = 0.0

        # 데이터 순회
        for i in range(len(self._data)):
            date = self._data.index[i]
            close = float(self._data['Close'].iloc[i])
            high = float(self._data['High'].iloc[i])
            low = float(self._data['Low'].iloc[i])
            avg_volume = float(self._data['Avg_Volume_20'].iloc[i]) if pd.notna(self._data['Avg_Volume_20'].iloc[i]) else 0

            # 현재 포지션 가치
            if position == PositionType.LONG:
                position_value = shares * close
            elif position == PositionType.SHORT:
                # 숏: 진입 시점 가치 + 미실현 손익
                unrealized_pnl = (entry_price - close) * shares
                position_value = shares * entry_price + unrealized_pnl
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

                # 시그널 기반 Exit (look-ahead bias 방지: 현재까지 데이터만 전달)
                if not exit_signal:
                    signal = self.strategy.signal_func(self._data.iloc[:i+1], i)
                    if position == PositionType.LONG and signal == SignalType.SELL:
                        exit_signal = True
                        exit_reason = "Signal Sell"
                    elif position == PositionType.SHORT and signal == SignalType.BUY:
                        exit_signal = True
                        exit_reason = "Signal Cover"

                # Exit 실행
                if exit_signal:
                    # 슬리피지 계산
                    trade_value = shares * close
                    slippage = self.cost_model.calculate_slippage(trade_value, avg_volume)
                    exit_price = close * (1 - slippage if position == PositionType.LONG else 1 + slippage)

                    # PnL 계산
                    if position == PositionType.LONG:
                        pnl = (exit_price - entry_price) * shares
                    else:  # SHORT
                        pnl = (entry_price - exit_price) * shares

                    # 비용 계산
                    commission = self.cost_model.calculate_commission(trade_value)
                    slippage_cost = abs(close - exit_price) * shares
                    holding_days = (date - entry_date).days

                    short_cost = 0.0
                    if position == PositionType.SHORT:
                        short_cost = self.cost_model.calculate_short_cost(
                            shares * entry_price, holding_days
                        )

                    total_cost = commission + slippage_cost + short_cost
                    pnl -= total_cost

                    # 자본 업데이트
                    capital += shares * entry_price + pnl
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
                        holding_days=holding_days,
                        commission=round(commission, 2),
                        slippage_cost=round(slippage_cost, 2),
                        short_cost=round(short_cost, 2),
                        signal_reason=exit_reason,
                    )
                    trades.append(trade)

                    position = PositionType.FLAT
                    shares = 0
                    entry_price = 0
                    position_value = 0

            # 포지션이 없을 때 - Entry 조건 확인
            if position == PositionType.FLAT and i < len(self._data) - 1:
                signal = self.strategy.signal_func(self._data.iloc[:i+1], i)

                if signal == SignalType.BUY:
                    # 고정 포지션 사이징 (복리 버그 수정)
                    invest_amount = self._calculate_position_size(capital, i)
                    invest_amount = min(invest_amount, capital * 0.95)  # 최대 95% 투자

                    slippage = self.cost_model.calculate_slippage(invest_amount, avg_volume)
                    entry_price = close * (1 + slippage)
                    shares = invest_amount / entry_price
                    commission = self.cost_model.calculate_commission(invest_amount)

                    capital -= invest_amount + commission
                    position = PositionType.LONG
                    entry_date = date

                elif signal == SignalType.SELL:
                    # Short position (margin 100% 가정 - Long과 동일하게 처리)
                    invest_amount = self._calculate_position_size(capital, i)
                    invest_amount = min(invest_amount, capital * 0.95)

                    slippage = self.cost_model.calculate_slippage(invest_amount, avg_volume)
                    entry_price = close * (1 - slippage)
                    shares = invest_amount / entry_price
                    commission = self.cost_model.calculate_commission(invest_amount)

                    # Short도 Long과 동일하게 자본 차감 (버그 수정)
                    capital -= invest_amount + commission
                    position = PositionType.SHORT
                    entry_date = date

        # 마지막 포지션 청산
        if position != PositionType.FLAT:
            close = float(self._data['Close'].iloc[-1])
            exit_price = close
            holding_days = (self._data.index[-1] - entry_date).days

            if position == PositionType.LONG:
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            commission = self.cost_model.calculate_commission(shares * close)
            short_cost = 0.0
            if position == PositionType.SHORT:
                short_cost = self.cost_model.calculate_short_cost(
                    shares * entry_price, holding_days
                )

            pnl -= commission + short_cost
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
                holding_days=holding_days,
                commission=round(commission, 2),
                short_cost=round(short_cost, 2),
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

        # 파라미터 저장
        parameters = {
            'position_size': self.strategy.position_size,
            'sizing_mode': self.strategy.sizing_mode.value,
            'stop_loss': self.strategy.stop_loss,
            'take_profit': self.strategy.take_profit,
            'max_holding_days': self.strategy.max_holding_days,
            'commission': self.cost_model.commission,
            'slippage': self.cost_model.base_slippage,
        }

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            parameters=parameters,
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

        print(f"\n[Transaction Costs]")
        print(f"  Total Commission:  ${result.total_commission:,.2f}")
        print(f"  Total Slippage:    ${result.total_slippage:,.2f}")
        print(f"  Total Short Cost:  ${result.total_short_cost:,.2f}")

        # 최근 거래 5개
        if result.trades:
            print(f"\n[Recent Trades]")
            for trade in result.trades[-5:]:
                direction = "LONG" if trade.direction == PositionType.LONG else "SHORT"
                pnl_str = f"+{trade.pnl:,.0f}" if trade.pnl > 0 else f"{trade.pnl:,.0f}"
                print(f"  {trade.entry_date} -> {trade.exit_date}: {direction} ${pnl_str} ({trade.pnl_pct:+.1f}%)")

        print("\n" + "=" * 70)

    def save_to_db(self, result: BacktestResult) -> int:
        """결과를 DB에 저장"""
        try:
            from lib.trading_db import TradingDB
        except ImportError:
            print("Warning: trading_db not available, skipping DB save")
            return -1

        db = TradingDB()
        run_id = db.save_backtest_run(result.to_dict())
        print(f"Results saved to DB: run_id={run_id}")
        return run_id

    def save_to_json(self, result: BacktestResult, output_dir: str = None) -> str:
        """결과를 JSON 파일로 저장"""
        from pathlib import Path

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{result.strategy_name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Results saved to: {filepath}")
        return str(filepath)


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
        if len(data) < long_period:
            return SignalType.HOLD

        close = data['Close'].values
        ma_short = float(np.mean(close[-short_period:]))
        ma_long = float(np.mean(close[-long_period:]))

        if len(close) < long_period + 1:
            return SignalType.HOLD

        ma_short_prev = float(np.mean(close[-short_period-1:-1]))
        ma_long_prev = float(np.mean(close[-long_period-1:-1]))

        if ma_short > ma_long and ma_short_prev <= ma_long_prev:
            return SignalType.BUY
        elif ma_short < ma_long and ma_short_prev >= ma_long_prev:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name=f"MA_Crossover_{short_period}_{long_period}",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
    )


def create_regime_based_strategy(ticker: str = "SPY") -> Strategy:
    """
    EIMAS 레짐 기반 전략
    - Bull + Low Vol: 공격적 롱
    - Bear + High Vol: 방어적 (매도/현금)
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if len(data) < 50:
            return SignalType.HOLD

        close = data['Close'].values

        # 추세: 50일 이평선
        ma_50 = float(np.mean(close[-50:]))
        current_price = close[-1]
        trend_bullish = current_price > ma_50

        # 모멘텀: 20일
        mom_20 = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
        mom_positive = mom_20 > 0

        # 변동성: 20일
        returns = np.diff(close[-21:]) / close[-21:-1] if len(close) >= 21 else []
        vol_20 = float(np.std(returns) * np.sqrt(252) * 100) if len(returns) > 0 else 0
        low_vol = vol_20 < 18

        # 레짐 판단
        if trend_bullish and mom_positive:
            if low_vol:
                return SignalType.BUY
            else:
                return SignalType.HOLD
        elif not trend_bullish and not mom_positive:
            if not low_vol:
                return SignalType.SELL
            else:
                return SignalType.HOLD
        else:
            return SignalType.HOLD

    return Strategy(
        name="EIMAS_Regime",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
        stop_loss=0.06,
        take_profit=0.12,
    )


def create_multi_factor_strategy(ticker: str = "SPY") -> Strategy:
    """
    다중 팩터 전략 (모멘텀 + 변동성 + 추세 + RSI)
    3개 이상 팩터 일치 시 진입
    """

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if len(data) < 60:
            return SignalType.HOLD

        close = data['Close'].values

        bullish_factors = 0
        bearish_factors = 0

        # Factor 1: 모멘텀 (20일)
        mom_20 = (close[-1] / close[-20] - 1) * 100
        if mom_20 > 3:
            bullish_factors += 1
        elif mom_20 < -3:
            bearish_factors += 1

        # Factor 2: 추세 (50일 이평선)
        ma_50 = float(np.mean(close[-50:]))
        if close[-1] > ma_50 * 1.02:
            bullish_factors += 1
        elif close[-1] < ma_50 * 0.98:
            bearish_factors += 1

        # Factor 3: 변동성
        returns = np.diff(close[-21:]) / close[-21:-1]
        vol_20 = float(np.std(returns) * np.sqrt(252) * 100)
        if vol_20 < 15:
            bullish_factors += 1
        elif vol_20 > 25:
            bearish_factors += 1

        # Factor 4: RSI
        delta = np.diff(close[-15:])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = float(np.mean(gain))
        avg_loss = float(np.mean(loss))
        rsi = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 50.0

        if rsi < 40:
            bullish_factors += 1
        elif rsi > 60:
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
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
        stop_loss=0.07,
        take_profit=0.15,
    )


def create_vix_mean_reversion_strategy(ticker: str = "SPY") -> Strategy:
    """VIX Mean Reversion 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if len(data) < 60:
            return SignalType.HOLD

        close = data['Close'].values
        returns = np.diff(close) / close[:-1]

        vol_5 = float(np.std(returns[-5:]) * np.sqrt(252) * 100) if len(returns) >= 5 else 0
        vol_20 = float(np.std(returns[-20:]) * np.sqrt(252) * 100) if len(returns) >= 20 else 0
        vol_60 = float(np.std(returns[-60:]) * np.sqrt(252) * 100) if len(returns) >= 60 else vol_20

        if vol_5 < vol_20 * 0.8 and vol_20 < vol_60 * 1.1:
            return SignalType.BUY
        elif vol_5 > vol_20 * 1.5:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="VIX_Mean_Reversion",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
        stop_loss=0.05,
        max_holding_days=20,
    )


def create_yield_curve_strategy(ticker: str = "SPY") -> Strategy:
    """금리 곡선 프록시 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if len(data) < 60:
            return SignalType.HOLD

        close = data['Close'].values

        mom_60 = (close[-1] / close[-60] - 1) * 100
        mom_20 = (close[-1] / close[-20] - 1) * 100

        returns = np.diff(close[-21:]) / close[-21:-1]
        vol_20 = float(np.std(returns) * np.sqrt(252) * 100)

        if mom_60 < -5 and mom_20 < 0 and vol_20 > 20:
            return SignalType.SELL
        elif mom_60 > 5 and mom_20 > 0 and vol_20 < 20:
            return SignalType.BUY
        else:
            return SignalType.HOLD

    return Strategy(
        name="Yield_Curve_Proxy",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
        stop_loss=0.08,
    )


def create_copper_gold_strategy(ticker: str = "SPY") -> Strategy:
    """Copper/Gold 비율 프록시 전략"""

    def signal_func(data: pd.DataFrame, idx: int) -> SignalType:
        if len(data) < 40:
            return SignalType.HOLD

        close = data['Close'].values

        mom_20 = (close[-1] / close[-20] - 1) * 100
        mom_40 = (close[-1] / close[-40] - 1) * 100

        if mom_20 > 2 and mom_40 > 0:
            return SignalType.BUY
        elif mom_20 < -2 and mom_40 < 0:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    return Strategy(
        name="Copper_Gold_Proxy",
        signal_func=signal_func,
        ticker=ticker,
        position_size=0.3,
        sizing_mode=PositionSizingMode.FIXED,
        max_holding_days=60,
    )


# ============================================================================
# Walk-Forward Validation
# ============================================================================

@dataclass
class WalkForwardFold:
    """Walk-Forward 단일 Fold 결과"""
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample_return: float
    in_sample_sharpe: float
    out_sample_return: float
    out_sample_sharpe: float
    degradation_pct: float  # (IS - OOS) / IS * 100


@dataclass
class WalkForwardResult:
    """Walk-Forward 전체 결과"""
    strategy_name: str
    total_folds: int
    folds: List[WalkForwardFold]

    # 집계 통계
    avg_is_sharpe: float
    avg_oos_sharpe: float
    avg_degradation: float
    min_oos_sharpe: float
    max_oos_sharpe: float

    # OOS 합산 수익
    cumulative_oos_return: float

    def is_robust(self) -> bool:
        """전략 견고성 판단: 평균 저하율 < 30%, OOS Sharpe > 0.5"""
        return self.avg_degradation < 30 and self.avg_oos_sharpe > 0.5


class WalkForwardValidator:
    """
    Walk-Forward Validation 엔진

    In-Sample(훈련) 기간에서 전략 평가 후
    Out-of-Sample(테스트) 기간에서 검증

    Usage:
        validator = WalkForwardValidator(
            strategy=create_regime_based_strategy("SPY"),
            start_date="2022-01-01",
            end_date="2025-12-31",
            train_months=12,
            test_months=3
        )
        result = validator.run()
        validator.print_report(result)
    """

    def __init__(
        self,
        strategy: Strategy,
        start_date: str = "2022-01-01",
        end_date: str = None,
        train_months: int = 12,
        test_months: int = 3,
        initial_capital: float = 100000,
    ):
        self.strategy = strategy
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.train_months = train_months
        self.test_months = test_months
        self.initial_capital = initial_capital
        self._data = None

    def _load_data(self) -> bool:
        """전체 기간 데이터 로드"""
        print(f"Loading data for walk-forward: {self.strategy.ticker}")
        try:
            df = yf.download(
                self.strategy.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            if df.empty:
                return False

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Volume_Value'] = df['Close'] * df['Volume']
            df['Avg_Volume_20'] = df['Volume_Value'].rolling(20).mean()
            self._data = df
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def _run_backtest_on_slice(
        self,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """특정 기간에 대해 백테스트 실행"""
        # 해당 기간 데이터 슬라이스
        mask = (self._data.index >= start_date) & (self._data.index <= end_date)
        slice_data = self._data[mask].copy()

        if len(slice_data) < 60:  # 최소 60일 필요
            return None

        bt = Backtester(
            strategy=self.strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            data=slice_data,
        )

        try:
            return bt.run()
        except Exception:
            return None

    def run(self) -> WalkForwardResult:
        """Walk-Forward 실행"""
        if not self._load_data():
            raise ValueError("Failed to load data")

        print(f"\nRunning Walk-Forward Validation: {self.strategy.name}")
        print(f"  Period: {self.start_date} ~ {self.end_date}")
        print(f"  Train: {self.train_months} months, Test: {self.test_months} months")

        folds = []
        fold_number = 0

        # 시작점
        current = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")

        while True:
            # Train 기간
            train_start = current
            train_end = train_start + timedelta(days=self.train_months * 30)

            # Test 기간
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > end:
                break

            fold_number += 1
            print(f"\n  Fold {fold_number}:")
            print(f"    Train: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
            print(f"    Test:  {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

            # In-Sample 백테스트
            is_result = self._run_backtest_on_slice(
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d")
            )

            # Out-of-Sample 백테스트
            oos_result = self._run_backtest_on_slice(
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d")
            )

            if is_result and oos_result:
                # 성과 저하율 계산
                if is_result.sharpe_ratio != 0:
                    degradation = ((is_result.sharpe_ratio - oos_result.sharpe_ratio)
                                   / abs(is_result.sharpe_ratio)) * 100
                else:
                    degradation = 0

                fold = WalkForwardFold(
                    fold_number=fold_number,
                    train_start=train_start.strftime("%Y-%m-%d"),
                    train_end=train_end.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                    in_sample_return=is_result.total_return,
                    in_sample_sharpe=is_result.sharpe_ratio,
                    out_sample_return=oos_result.total_return,
                    out_sample_sharpe=oos_result.sharpe_ratio,
                    degradation_pct=degradation,
                )
                folds.append(fold)

                print(f"    IS: Return={is_result.total_return:+.1f}%, Sharpe={is_result.sharpe_ratio:.2f}")
                print(f"    OOS: Return={oos_result.total_return:+.1f}%, Sharpe={oos_result.sharpe_ratio:.2f}")
                print(f"    Degradation: {degradation:+.1f}%")
            else:
                print(f"    Skipped (insufficient data)")

            # 다음 Fold로 이동 (test_months 만큼)
            current = current + timedelta(days=self.test_months * 30)

        if not folds:
            raise ValueError("No valid folds generated")

        # 집계 통계
        avg_is = np.mean([f.in_sample_sharpe for f in folds])
        avg_oos = np.mean([f.out_sample_sharpe for f in folds])
        avg_deg = np.mean([f.degradation_pct for f in folds])
        min_oos = min(f.out_sample_sharpe for f in folds)
        max_oos = max(f.out_sample_sharpe for f in folds)
        cum_oos = sum(f.out_sample_return for f in folds)

        return WalkForwardResult(
            strategy_name=self.strategy.name,
            total_folds=len(folds),
            folds=folds,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            avg_degradation=avg_deg,
            min_oos_sharpe=min_oos,
            max_oos_sharpe=max_oos,
            cumulative_oos_return=cum_oos,
        )

    def print_report(self, result: WalkForwardResult):
        """Walk-Forward 결과 리포트"""
        print("\n" + "=" * 70)
        print(f"WALK-FORWARD VALIDATION: {result.strategy_name}")
        print("=" * 70)

        print(f"\n[Summary]")
        print(f"  Total Folds:         {result.total_folds}")
        print(f"  Avg IS Sharpe:       {result.avg_is_sharpe:.2f}")
        print(f"  Avg OOS Sharpe:      {result.avg_oos_sharpe:.2f}")
        print(f"  Avg Degradation:     {result.avg_degradation:+.1f}%")
        print(f"  OOS Sharpe Range:    [{result.min_oos_sharpe:.2f}, {result.max_oos_sharpe:.2f}]")
        print(f"  Cumulative OOS:      {result.cumulative_oos_return:+.1f}%")

        robust = result.is_robust()
        print(f"\n[Robustness Check]")
        print(f"  Status: {'PASS' if robust else 'FAIL'}")
        if not robust:
            if result.avg_degradation >= 30:
                print(f"    - High degradation ({result.avg_degradation:.1f}% >= 30%)")
            if result.avg_oos_sharpe <= 0.5:
                print(f"    - Low OOS Sharpe ({result.avg_oos_sharpe:.2f} <= 0.5)")

        print(f"\n[Fold Details]")
        print(f"  {'Fold':>4} {'Train Period':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'Degrad':>8}")
        print("-" * 65)
        for f in result.folds:
            period = f"{f.train_start} ~ {f.train_end[:7]}"
            print(f"  {f.fold_number:>4} {period:<25} {f.in_sample_sharpe:>10.2f} {f.out_sample_sharpe:>11.2f} {f.degradation_pct:>+7.1f}%")

        print("=" * 70)

    def save_to_db(self, result: WalkForwardResult, run_id: int = None) -> int:
        """결과를 DB에 저장"""
        try:
            from lib.trading_db import TradingDB
            db = TradingDB()

            # run_id가 없으면 새로 생성 (빈 백테스트 run)
            if run_id is None:
                run_data = {
                    'strategy_name': result.strategy_name + "_WF",
                    'start_date': result.folds[0].train_start if result.folds else "",
                    'end_date': result.folds[-1].test_end if result.folds else "",
                    'initial_capital': self.initial_capital,
                    'final_capital': self.initial_capital,
                    'total_return': result.cumulative_oos_return,
                    'sharpe_ratio': result.avg_oos_sharpe,
                    'parameters': {
                        'train_months': self.train_months,
                        'test_months': self.test_months,
                        'total_folds': result.total_folds,
                    },
                    'trades': [],
                }
                run_id = db.save_backtest_run(run_data)

            # 각 Fold 결과 저장
            for fold in result.folds:
                db.save_walk_forward_result(
                    run_id=run_id,
                    fold_number=fold.fold_number,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    in_sample_return=fold.in_sample_return,
                    in_sample_sharpe=fold.in_sample_sharpe,
                    out_sample_return=fold.out_sample_return,
                    out_sample_sharpe=fold.out_sample_sharpe,
                )

            print(f"Walk-Forward results saved to DB: run_id={run_id}")
            return run_id

        except Exception as e:
            print(f"DB save failed: {e}")
            return -1


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Backtester v2.0 Test")
    print("=" * 70)
    print(f"Default Period: {DEFAULT_START_DATE} ~ today")
    print("Position Sizing: FIXED (no compounding bug)")
    print("=" * 70)

    # 전략 테스트
    strategies = [
        create_regime_based_strategy("SPY"),
        create_multi_factor_strategy("SPY"),
        create_ma_crossover_strategy(20, 50, "SPY"),
    ]

    results = []

    for strategy in strategies:
        print(f"\n{'='*70}")
        bt = Backtester(
            strategy=strategy,
            initial_capital=100000,
        )

        try:
            result = bt.run()
            bt.print_report(result)
            bt.save_to_json(result)
            results.append(result)
        except Exception as e:
            print(f"Error running {strategy.name}: {e}")
            import traceback
            traceback.print_exc()

    # 전략 비교
    if results:
        print("\n" + "=" * 70)
        print("STRATEGY COMPARISON (Fixed Position 30%)")
        print("=" * 70)
        print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'Trades':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r.strategy_name:<25} {r.total_return:>+9.1f}% {r.sharpe_ratio:>8.2f} {r.max_drawdown:>7.1f}% {r.win_rate:>7.1f}% {r.total_trades:>8}")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
