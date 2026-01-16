# EIMAS 마이크로스트럭처 통합 가이드

> **목표**: 고빈도 시장 미시구조 데이터를 EIMAS에 통합하여 초단기 예측력 강화
> **타겟**: 5분 예측 R² 0.65+, 백테스트 Sharpe > 2.0, Win Rate 65%+

---

## 1. 개요

### 1.1 왜 마이크로스트럭처인가?

```
기존 EIMAS (일간 데이터):
┌────────────────────────────────────────┐
│  FRED/yfinance → LASSO → 30일+ 예측   │
│  R² 0.68~0.76, 장기 레짐 분석          │
└────────────────────────────────────────┘

마이크로스트럭처 확장 (틱/분 데이터):
┌────────────────────────────────────────┐
│  LOB/Trade → OFI/VPIN → 5분 예측      │
│  R² 0.65+, 초단기 시그널 포착          │
│  + 기존 매크로 시그널과 결합           │
└────────────────────────────────────────┘
```

### 1.2 핵심 메트릭

| 메트릭 | 방법론 | 예측 대상 | 목표 성능 |
|--------|--------|----------|----------|
| **OFI** | Cont et al. (2014) | 5-15분 가격 방향 | 방향 정확도 60%+ |
| **VPIN** | Easley et al. (2012) | Intraday crash | 탐지 정확도 71% |
| **Microstructure LASSO** | JMLR (2020) | 1-5분 return | R² 0.65 |
| **Hawkes Process** | Bacry et al. (2015) | Liquidity shock | Lead time 70% |
| **Queue Position** | Cartea et al. (2018) | Execution probability | Fill rate +15% |

---

## 2. 데이터 요구사항

### 2.1 LOB (Limit Order Book) Level 2 데이터

```python
@dataclass
class LOBSnapshot:
    """Level 2 호가창 스냅샷"""
    timestamp: datetime  # 마이크로초 정밀도
    symbol: str

    # Bid side (매수)
    bid_prices: List[float]    # [best, 2nd, 3rd, ..., 10th]
    bid_sizes: List[int]       # 각 레벨 수량

    # Ask side (매도)
    ask_prices: List[float]
    ask_sizes: List[int]

    # 파생 메트릭
    mid_price: float           # (best_bid + best_ask) / 2
    spread: float              # best_ask - best_bid
    spread_bps: float          # spread / mid_price * 10000

# 필요 빈도: 5초마다 (또는 틱마다)
# 레벨: 최소 5 levels, 권장 10 levels
```

### 2.2 Trade (체결) 데이터

```python
@dataclass
class Trade:
    """틱 체결 데이터"""
    timestamp: datetime        # 마이크로초
    symbol: str
    price: float
    size: int
    side: str                  # 'buy' | 'sell' (aggressor)

    # VPIN 계산용
    dollar_volume: float       # price * size
    trade_sign: int            # +1 (buy), -1 (sell)

# 필요 빈도: 모든 체결 (tick-by-tick)
# Trade sign 결정: Lee-Ready algorithm 또는 tick rule
```

### 2.3 Order Events (주문 이벤트)

```python
@dataclass
class OrderEvent:
    """주문 arrival/cancellation 이벤트 (Hawkes용)"""
    timestamp: datetime        # 초 단위
    symbol: str
    event_type: str            # 'new' | 'cancel' | 'modify' | 'fill'
    side: str                  # 'bid' | 'ask'
    price: float
    size: int
    order_id: str              # 주문 추적용

# 필요 빈도: 모든 이벤트 (second-level timestamp)
```

### 2.4 데이터 소스 비교

| 소스 | 데이터 타입 | 비용 | 지연 | 권장 용도 |
|------|------------|------|------|----------|
| **Polygon.io** | LOB L2, Trades | $199/월 | 실시간 | 프로덕션 |
| **Alpaca** | Trades, Quotes | 무료~$99 | 실시간 | 개발/테스트 |
| **IEX Cloud** | Trades, TOPS | $9/월~ | 15분 지연 | 연구 |
| **FirstRate Data** | Historical LOB | 일회성 | - | 백테스트 |
| **Databento** | Full LOB | $500/월~ | 실시간 | 기관급 |

---

## 3. OFI (Order Flow Imbalance)

### 3.1 방법론: Cont et al. (2014)

```
논문: "The Price Impact of Order Book Events"
핵심: Bid/Ask size 변화량의 누적이 단기 가격 방향 예측

OFI = Σ (ΔBid_size - ΔAsk_size)

해석:
- OFI > 0: 매수 압력 증가 → 가격 상승 예상
- OFI < 0: 매도 압력 증가 → 가격 하락 예상
```

### 3.2 구현

```python
class OFICalculator:
    """
    Order Flow Imbalance 계산기

    Cont et al. (2014) 방법론:
    - Best bid/ask size 변화량 추적
    - 누적 불균형으로 가격 방향 예측
    """

    def __init__(self, window_seconds: int = 300):  # 5분 윈도우
        self.window = window_seconds
        self.history: Deque[Tuple[datetime, float]] = deque()
        self.prev_snapshot: Optional[LOBSnapshot] = None

    def update(self, snapshot: LOBSnapshot) -> float:
        """
        새 스냅샷으로 OFI 업데이트

        Returns:
            현재 OFI 값 (누적)
        """
        if self.prev_snapshot is None:
            self.prev_snapshot = snapshot
            return 0.0

        # Bid side 변화
        delta_bid = 0
        if snapshot.bid_prices[0] >= self.prev_snapshot.bid_prices[0]:
            delta_bid = snapshot.bid_sizes[0] - self.prev_snapshot.bid_sizes[0]
        elif snapshot.bid_prices[0] < self.prev_snapshot.bid_prices[0]:
            delta_bid = -self.prev_snapshot.bid_sizes[0]

        # Ask side 변화
        delta_ask = 0
        if snapshot.ask_prices[0] <= self.prev_snapshot.ask_prices[0]:
            delta_ask = snapshot.ask_sizes[0] - self.prev_snapshot.ask_sizes[0]
        elif snapshot.ask_prices[0] > self.prev_snapshot.ask_prices[0]:
            delta_ask = -self.prev_snapshot.ask_sizes[0]

        # OFI = ΔBid - ΔAsk
        ofi_tick = delta_bid - delta_ask

        # 시간 윈도우 내 누적
        now = snapshot.timestamp
        self.history.append((now, ofi_tick))

        # 오래된 데이터 제거
        cutoff = now - timedelta(seconds=self.window)
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # 누적 OFI
        cumulative_ofi = sum(h[1] for h in self.history)

        self.prev_snapshot = snapshot
        return cumulative_ofi

    def get_signal(self, threshold: float = 1000) -> str:
        """
        OFI 기반 시그널

        Args:
            threshold: 시그널 발생 임계값

        Returns:
            'BUY', 'SELL', 'NEUTRAL'
        """
        ofi = sum(h[1] for h in self.history)

        if ofi > threshold:
            return 'BUY'
        elif ofi < -threshold:
            return 'SELL'
        else:
            return 'NEUTRAL'
```

### 3.3 예측 모델

```python
def predict_price_direction(
    ofi: float,
    spread: float,
    depth_imbalance: float,
    horizon_minutes: int = 5
) -> Dict:
    """
    OFI 기반 가격 방향 예측

    Cont et al. (2014) 선형 모델:
    ΔP = β₀ + β₁·OFI + β₂·Spread + β₃·Depth + ε

    Returns:
        {'direction': 'UP'|'DOWN', 'confidence': float, 'expected_move_bps': float}
    """
    # 사전 추정된 계수 (백테스트로 calibration 필요)
    beta = {
        'intercept': 0.0,
        'ofi': 0.00001,        # OFI 1000 → 1bp 이동
        'spread': -0.5,        # 스프레드 확대 → 하락 압력
        'depth': 0.3           # 매수 depth 우위 → 상승
    }

    expected_move = (
        beta['intercept'] +
        beta['ofi'] * ofi +
        beta['spread'] * spread +
        beta['depth'] * depth_imbalance
    )

    return {
        'direction': 'UP' if expected_move > 0 else 'DOWN',
        'confidence': min(abs(expected_move) / 5, 1.0),  # 5bp에서 100% 신뢰도
        'expected_move_bps': expected_move * 10000
    }
```

---

## 4. VPIN (Volume-Synchronized Probability of Informed Trading)

### 4.1 방법론: Easley et al. (2012)

```
논문: "Flow Toxicity and Liquidity in a High Frequency World"
핵심: Volume bucket 단위로 buy/sell 불균형 측정 → 정보 거래자 비율 추정

VPIN = |V_buy - V_sell| / V_total (bucket 평균)

해석:
- VPIN > 0.8: 극심한 정보 비대칭 → Crash 위험
- 2010 Flash Crash 사전 예측 성공
```

### 4.2 구현

```python
class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading

    Easley et al. (2012) 버킷화 기법:
    - 고정 volume bucket으로 시간 정규화
    - Buy/Sell volume 불균형 측정
    - Intraday crash 71% 정확도 탐지
    """

    def __init__(
        self,
        bucket_size: float = 50000,  # $50,000 per bucket
        n_buckets: int = 50          # 50 buckets for rolling VPIN
    ):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0

        self.buckets: Deque[Tuple[float, float]] = deque(maxlen=n_buckets)

    def update(self, trade: Trade) -> Optional[float]:
        """
        새 체결로 VPIN 업데이트

        Args:
            trade: 체결 데이터 (dollar_volume, trade_sign 포함)

        Returns:
            버킷 완성 시 VPIN 값, 아니면 None
        """
        # Trade sign으로 buy/sell 분류
        if trade.trade_sign > 0:
            self.current_bucket_buy += trade.dollar_volume
        else:
            self.current_bucket_sell += trade.dollar_volume

        self.current_bucket_volume += trade.dollar_volume

        # 버킷 완성 체크
        if self.current_bucket_volume >= self.bucket_size:
            # 버킷 저장
            self.buckets.append((
                self.current_bucket_buy,
                self.current_bucket_sell
            ))

            # 리셋
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0
            self.current_bucket_volume = 0.0

            # VPIN 계산
            return self._calculate_vpin()

        return None

    def _calculate_vpin(self) -> float:
        """
        현재 VPIN 계산

        VPIN = Σ|V_buy - V_sell| / Σ(V_buy + V_sell)
        """
        if len(self.buckets) < self.n_buckets:
            return 0.0

        total_imbalance = 0.0
        total_volume = 0.0

        for buy, sell in self.buckets:
            total_imbalance += abs(buy - sell)
            total_volume += buy + sell

        if total_volume == 0:
            return 0.0

        return total_imbalance / total_volume

    def get_current_vpin(self) -> float:
        """현재 VPIN 값 반환"""
        return self._calculate_vpin()

    def is_crash_warning(self, threshold: float = 0.8) -> bool:
        """
        Crash 경고 여부

        Args:
            threshold: VPIN 임계값 (기본 0.8)

        Returns:
            True if crash warning
        """
        return self.get_current_vpin() > threshold


def classify_trade_sign(
    trade_price: float,
    bid: float,
    ask: float,
    prev_price: float
) -> int:
    """
    Lee-Ready Algorithm으로 trade sign 분류

    1. Quote rule: mid보다 위면 buy, 아래면 sell
    2. Tick rule: 이전 가격보다 올랐으면 buy, 내렸으면 sell

    Returns:
        +1 (buy) or -1 (sell)
    """
    mid = (bid + ask) / 2

    # Quote rule
    if trade_price > mid:
        return 1
    elif trade_price < mid:
        return -1
    else:
        # Tick rule (fallback)
        if trade_price > prev_price:
            return 1
        elif trade_price < prev_price:
            return -1
        else:
            return 1  # 기본값
```

### 4.3 Crash 탐지 시스템

```python
class CrashDetector:
    """
    VPIN 기반 Crash 탐지 시스템

    알림 레벨:
    - VPIN > 0.6: ELEVATED (관심)
    - VPIN > 0.7: HIGH (경고)
    - VPIN > 0.8: CRITICAL (즉시 대응)
    """

    def __init__(self, vpin_calculator: VPINCalculator):
        self.vpin = vpin_calculator
        self.alert_history: List[Dict] = []

    def check(self) -> Optional[Dict]:
        """
        Crash 위험 체크

        Returns:
            경고 정보 또는 None
        """
        current_vpin = self.vpin.get_current_vpin()

        if current_vpin > 0.8:
            alert = {
                'level': 'CRITICAL',
                'vpin': current_vpin,
                'message': 'Extreme information asymmetry detected',
                'action': 'REDUCE_EXPOSURE_IMMEDIATELY',
                'timestamp': datetime.now()
            }
        elif current_vpin > 0.7:
            alert = {
                'level': 'HIGH',
                'vpin': current_vpin,
                'message': 'High toxicity flow detected',
                'action': 'TIGHTEN_STOPS',
                'timestamp': datetime.now()
            }
        elif current_vpin > 0.6:
            alert = {
                'level': 'ELEVATED',
                'vpin': current_vpin,
                'message': 'Elevated informed trading activity',
                'action': 'MONITOR_CLOSELY',
                'timestamp': datetime.now()
            }
        else:
            return None

        self.alert_history.append(alert)
        return alert
```

---

## 5. Microstructure LASSO

### 5.1 방법론: JMLR (2020)

```
논문: "High-Dimensional VAR with Microstructure Features"
핵심: 틱 레벨 feature selection으로 1-5분 return 예측

Features:
- OFI (20초 rolling)
- Bid imbalance (top 5 levels)
- Spread (bps)
- Depth imbalance (bid vs ask total depth)
- Trade imbalance (buy vs sell volume)
- Volatility (realized, 1분)
```

### 5.2 Feature Engineering

```python
class MicrostructureFeatures:
    """
    마이크로스트럭처 feature 계산

    20초 rolling window로 실시간 업데이트
    """

    def __init__(self, window_seconds: int = 20):
        self.window = window_seconds
        self.ofi_calc = OFICalculator(window_seconds)
        self.snapshots: Deque[LOBSnapshot] = deque()
        self.trades: Deque[Trade] = deque()

    def update_lob(self, snapshot: LOBSnapshot):
        """LOB 스냅샷 업데이트"""
        now = snapshot.timestamp
        cutoff = now - timedelta(seconds=self.window)

        # 오래된 데이터 제거
        while self.snapshots and self.snapshots[0].timestamp < cutoff:
            self.snapshots.popleft()

        self.snapshots.append(snapshot)
        self.ofi_calc.update(snapshot)

    def update_trade(self, trade: Trade):
        """체결 데이터 업데이트"""
        now = trade.timestamp
        cutoff = now - timedelta(seconds=self.window)

        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()

        self.trades.append(trade)

    def compute_features(self) -> Dict[str, float]:
        """
        현재 윈도우의 모든 feature 계산

        Returns:
            {
                'ofi': float,
                'bid_imbalance': float,
                'spread_bps': float,
                'depth_imbalance': float,
                'trade_imbalance': float,
                'volatility_1m': float,
                'vwap_deviation': float
            }
        """
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]

        # 1. OFI
        ofi = sum(h[1] for h in self.ofi_calc.history)

        # 2. Bid Imbalance (top 5 levels)
        total_bid = sum(latest.bid_sizes[:5])
        total_ask = sum(latest.ask_sizes[:5])
        bid_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1)

        # 3. Spread (bps)
        spread_bps = latest.spread_bps

        # 4. Depth Imbalance (all levels)
        all_bid = sum(latest.bid_sizes)
        all_ask = sum(latest.ask_sizes)
        depth_imbalance = (all_bid - all_ask) / (all_bid + all_ask + 1)

        # 5. Trade Imbalance
        buy_vol = sum(t.size for t in self.trades if t.trade_sign > 0)
        sell_vol = sum(t.size for t in self.trades if t.trade_sign < 0)
        trade_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1)

        # 6. Realized Volatility (1분)
        if len(self.snapshots) > 10:
            prices = [s.mid_price for s in self.snapshots]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(len(returns))
        else:
            volatility = 0.0

        # 7. VWAP Deviation
        if self.trades:
            vwap = sum(t.price * t.size for t in self.trades) / sum(t.size for t in self.trades)
            vwap_deviation = (latest.mid_price - vwap) / vwap * 10000  # bps
        else:
            vwap_deviation = 0.0

        return {
            'ofi': ofi,
            'bid_imbalance': bid_imbalance,
            'spread_bps': spread_bps,
            'depth_imbalance': depth_imbalance,
            'trade_imbalance': trade_imbalance,
            'volatility_1m': volatility,
            'vwap_deviation': vwap_deviation
        }
```

### 5.3 Microstructure LASSO Forecaster

```python
class MicrostructureLASSO:
    """
    마이크로스트럭처 기반 LASSO 예측기

    JMLR (2020) 방법론:
    - 20초 rolling features
    - 1-5분 return 예측
    - 목표 R² 0.65
    """

    def __init__(
        self,
        horizon_minutes: int = 5,
        alpha: float = 0.01
    ):
        self.horizon = horizon_minutes
        self.model = Lasso(alpha=alpha, max_iter=10000)
        self.scaler = StandardScaler()
        self.feature_names = [
            'ofi', 'bid_imbalance', 'spread_bps',
            'depth_imbalance', 'trade_imbalance',
            'volatility_1m', 'vwap_deviation'
        ]
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        모델 학습

        Args:
            X: feature DataFrame (columns = feature_names)
            y: {horizon}분 후 return (bps)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Feature importance 로깅
        coefs = dict(zip(self.feature_names, self.model.coef_))
        logger.info(f"Microstructure LASSO coefficients: {coefs}")

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Return 예측

        Args:
            features: 현재 microstructure features

        Returns:
            {
                'predicted_return_bps': float,
                'direction': 'UP' | 'DOWN',
                'confidence': float
            }
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]

        return {
            'predicted_return_bps': pred,
            'direction': 'UP' if pred > 0 else 'DOWN',
            'confidence': min(abs(pred) / 10, 1.0),  # 10bps에서 100% 신뢰도
            'horizon_minutes': self.horizon
        }

    def get_r_squared(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Out-of-sample R² 계산"""
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
```

---

## 6. Hawkes Process

### 6.1 방법론: Bacry et al. (2015)

```
논문: "Hawkes Processes in Finance"
핵심: 자기-강화 점 과정으로 order arrival intensity 모델링

λ(t) = μ + Σ α·exp(-β(t-tᵢ))

해석:
- 주문이 주문을 부름 (clustering)
- Intensity spike → Liquidity shock 예고
- 70% 정확도로 lead time 예측
```

### 6.2 구현

```python
class HawkesProcess:
    """
    Hawkes Process for Order Flow Modeling

    Bacry et al. (2015):
    - Order arrival/cancellation 이벤트 추적
    - Intensity clustering으로 liquidity shock 예측
    """

    def __init__(
        self,
        mu: float = 1.0,      # 기본 intensity
        alpha: float = 0.5,    # 자기-강화 계수
        beta: float = 1.0      # 감쇠율
    ):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.events: List[datetime] = []

    def add_event(self, timestamp: datetime):
        """이벤트 추가"""
        self.events.append(timestamp)

        # 오래된 이벤트 정리 (1시간 이상)
        cutoff = timestamp - timedelta(hours=1)
        self.events = [e for e in self.events if e > cutoff]

    def intensity(self, t: datetime) -> float:
        """
        현재 시점 intensity 계산

        λ(t) = μ + Σ α·exp(-β(t-tᵢ))
        """
        intensity = self.mu

        for event_time in self.events:
            if event_time < t:
                dt = (t - event_time).total_seconds()
                intensity += self.alpha * np.exp(-self.beta * dt)

        return intensity

    def predict_shock(self, threshold: float = 5.0) -> Dict:
        """
        Liquidity shock 예측

        Args:
            threshold: intensity 임계값 (기본 강도의 5배)

        Returns:
            {
                'current_intensity': float,
                'shock_probability': float,
                'estimated_lead_time_seconds': float
            }
        """
        now = datetime.now()
        current = self.intensity(now)

        # Shock probability (logistic 변환)
        prob = 1 / (1 + np.exp(-(current - threshold)))

        # Lead time 추정 (intensity가 높을수록 짧음)
        if current > self.mu:
            lead_time = 60 / (current - self.mu)  # 초 단위
        else:
            lead_time = float('inf')

        return {
            'current_intensity': current,
            'shock_probability': prob,
            'estimated_lead_time_seconds': min(lead_time, 300)  # 최대 5분
        }


class OrderFlowHawkes:
    """
    양방향 Hawkes (Bid/Ask 분리)
    """

    def __init__(self):
        self.bid_hawkes = HawkesProcess(mu=1.0, alpha=0.4, beta=1.2)
        self.ask_hawkes = HawkesProcess(mu=1.0, alpha=0.4, beta=1.2)

    def add_order_event(self, event: OrderEvent):
        """주문 이벤트 추가"""
        if event.side == 'bid':
            self.bid_hawkes.add_event(event.timestamp)
        else:
            self.ask_hawkes.add_event(event.timestamp)

    def get_imbalance(self) -> float:
        """Intensity 불균형 (bid - ask)"""
        now = datetime.now()
        bid_intensity = self.bid_hawkes.intensity(now)
        ask_intensity = self.ask_hawkes.intensity(now)

        return bid_intensity - ask_intensity

    def predict_direction(self) -> Dict:
        """
        Intensity 기반 방향 예측

        Returns:
            {'direction': 'UP'|'DOWN', 'strength': float}
        """
        imbalance = self.get_imbalance()

        return {
            'direction': 'UP' if imbalance > 0 else 'DOWN',
            'strength': abs(imbalance),
            'bid_intensity': self.bid_hawkes.intensity(datetime.now()),
            'ask_intensity': self.ask_hawkes.intensity(datetime.now())
        }
```

---

## 7. Queue Position Trading

### 7.1 방법론: Cartea et al. (2018)

```
논문: "Algorithmic Trading in Practice"
핵심: Queue position으로 execution probability 추정

Fill Probability ∝ 1 / (Queue Position + 1)

활용:
- 유리한 queue position에서만 주문
- Fill rate 15% 개선
- Adverse selection 회피
```

### 7.2 구현

```python
class QueuePositionTracker:
    """
    Queue Position 추적 및 Execution Probability 추정

    Cartea et al. (2018):
    - Top 5-10 level queue length 모니터링
    - Modification rate 추적
    - Fill rate 최적화
    """

    def __init__(self, levels: int = 10):
        self.levels = levels
        self.queue_history: Deque[Dict] = deque(maxlen=1000)

    def update(self, snapshot: LOBSnapshot):
        """Queue 상태 업데이트"""
        state = {
            'timestamp': snapshot.timestamp,
            'bid_queues': snapshot.bid_sizes[:self.levels],
            'ask_queues': snapshot.ask_sizes[:self.levels],
            'bid_prices': snapshot.bid_prices[:self.levels],
            'ask_prices': snapshot.ask_prices[:self.levels]
        }
        self.queue_history.append(state)

    def estimate_fill_probability(
        self,
        side: str,
        price: float,
        position: int
    ) -> float:
        """
        Execution probability 추정

        Args:
            side: 'bid' or 'ask'
            price: 주문 가격
            position: Queue 내 예상 위치

        Returns:
            Fill probability (0-1)
        """
        if not self.queue_history:
            return 0.5

        latest = self.queue_history[-1]

        # 해당 가격 레벨 찾기
        if side == 'bid':
            prices = latest['bid_prices']
            queues = latest['bid_queues']
        else:
            prices = latest['ask_prices']
            queues = latest['ask_queues']

        # 가격 레벨 확인
        level = None
        for i, p in enumerate(prices):
            if abs(p - price) < 0.001:  # 가격 일치
                level = i
                break

        if level is None:
            return 0.1  # 호가창 밖

        total_queue = queues[level]

        # Fill probability (exponential decay)
        # P(fill) = exp(-position / total_queue)
        if total_queue > 0:
            prob = np.exp(-position / total_queue)
        else:
            prob = 1.0

        return prob

    def get_optimal_price(self, side: str, target_fill_prob: float = 0.5) -> float:
        """
        목표 fill probability 달성을 위한 최적 가격

        Args:
            side: 'bid' or 'ask'
            target_fill_prob: 목표 체결 확률

        Returns:
            최적 주문 가격
        """
        if not self.queue_history:
            return 0.0

        latest = self.queue_history[-1]

        if side == 'bid':
            prices = latest['bid_prices']
            queues = latest['bid_queues']
        else:
            prices = latest['ask_prices']
            queues = latest['ask_queues']

        # 각 레벨의 기대 fill prob 계산
        for level, (price, queue) in enumerate(zip(prices, queues)):
            # 새 주문은 queue 맨 뒤
            position = queue
            prob = np.exp(-position / (queue + 1)) if queue > 0 else 1.0

            if prob >= target_fill_prob:
                return price

        # 목표 달성 불가 → aggressive 가격 제안
        if side == 'bid':
            return prices[0] + 0.01  # best bid + tick
        else:
            return prices[0] - 0.01  # best ask - tick

    def get_modification_rate(self) -> float:
        """Queue modification rate 계산 (변동성 지표)"""
        if len(self.queue_history) < 2:
            return 0.0

        changes = 0
        for i in range(1, len(self.queue_history)):
            prev = self.queue_history[i-1]
            curr = self.queue_history[i]

            for level in range(min(5, self.levels)):
                if prev['bid_queues'][level] != curr['bid_queues'][level]:
                    changes += 1
                if prev['ask_queues'][level] != curr['ask_queues'][level]:
                    changes += 1

        # 초당 변화율
        time_span = (
            self.queue_history[-1]['timestamp'] -
            self.queue_history[0]['timestamp']
        ).total_seconds()

        if time_span > 0:
            return changes / time_span
        return 0.0
```

---

## 8. Critical Path 통합

### 8.1 Path 18: Microstructure Stress

```python
# critical_path_config.py 확장

CRITICAL_PATHS = {
    # ... 기존 paths ...

    18: {
        'name': 'Microstructure Stress',
        'description': 'VPIN>0.8 AND OFI divergence → 초단기 regime change',
        'triggers': [
            {'metric': 'vpin', 'threshold': 0.8, 'direction': '>'},
            {'metric': 'ofi_volatility', 'threshold': 2.0, 'direction': '>'},  # 2σ
            {'metric': 'spread_zscore', 'threshold': 2.0, 'direction': '>'}
        ],
        'linked_paths': [12],  # Path 12: Liquidity Cascade
        'update_frequency': '5min',
        'severity': 'HIGH'
    }
}
```

### 8.2 Path 12 연계 (Liquidity Cascade)

```python
class MicrostructureCriticalPath:
    """
    마이크로스트럭처 기반 Critical Path 분석

    Path 18 (Microstructure Stress)와 Path 12 (Liquidity Cascade) 연계
    """

    def __init__(
        self,
        vpin_calculator: VPINCalculator,
        ofi_calculator: OFICalculator
    ):
        self.vpin = vpin_calculator
        self.ofi = ofi_calculator
        self.ofi_history: Deque[float] = deque(maxlen=60)  # 5분 (5초 간격)

    def update(self, ofi_value: float):
        """5분마다 OFI 업데이트"""
        self.ofi_history.append(ofi_value)

    def check_path_18(self) -> Dict:
        """
        Path 18 (Microstructure Stress) 체크

        조건:
        1. VPIN > 0.8
        2. OFI volatility > 2σ
        3. 5분 내 급격한 변화

        Returns:
            {'triggered': bool, 'severity': str, 'details': Dict}
        """
        current_vpin = self.vpin.get_current_vpin()

        # OFI volatility
        if len(self.ofi_history) > 10:
            ofi_std = np.std(list(self.ofi_history))
            ofi_mean = np.mean(list(self.ofi_history))
            ofi_zscore = abs(self.ofi_history[-1] - ofi_mean) / (ofi_std + 1e-6)
        else:
            ofi_zscore = 0

        # Path 18 트리거 체크
        vpin_triggered = current_vpin > 0.8
        ofi_triggered = ofi_zscore > 2.0

        if vpin_triggered and ofi_triggered:
            return {
                'triggered': True,
                'path': 18,
                'severity': 'CRITICAL',
                'details': {
                    'vpin': current_vpin,
                    'ofi_zscore': ofi_zscore,
                    'linked_path': 12
                },
                'action': 'REDUCE_ALL_POSITIONS',
                'message': 'Microstructure stress detected - potential liquidity cascade'
            }
        elif vpin_triggered or ofi_triggered:
            return {
                'triggered': True,
                'path': 18,
                'severity': 'WARNING',
                'details': {
                    'vpin': current_vpin,
                    'ofi_zscore': ofi_zscore
                },
                'action': 'TIGHTEN_RISK_LIMITS',
                'message': 'Elevated microstructure stress'
            }
        else:
            return {'triggered': False}

    def get_vpin_trend(self, lookback: int = 12) -> str:
        """
        VPIN 트렌드 분석 (5분마다 체크, 1시간 lookback)

        Returns:
            'RISING', 'FALLING', 'STABLE'
        """
        # 구현 필요: VPIN 히스토리 추적
        pass

    def get_ofi_volatility(self) -> float:
        """OFI volatility (표준편차)"""
        if len(self.ofi_history) < 5:
            return 0.0
        return np.std(list(self.ofi_history))
```

---

## 9. EIMAS EconomicNetworkBuilder 확장

### 9.1 Micro Edges 추가

```python
# network_builder.py 확장

class ExtendedEconomicNetworkBuilder(EconomicNetworkBuilder):
    """
    마이크로스트럭처 edge가 추가된 경제 네트워크
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_features = ['ofi', 'vpin', 'spread_bps', 'depth_imbalance']

    def add_micro_edges(
        self,
        micro_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        target: str = 'ret_spy_5min'
    ) -> 'EconomicNetwork':
        """
        마이크로스트럭처 → 매크로 연결 추가

        Args:
            micro_data: 마이크로스트럭처 features (5분 리샘플링)
            macro_data: 기존 FRED/yfinance 데이터
            target: 예측 대상 (5분 SPY 수익률)

        Returns:
            확장된 EconomicNetwork
        """
        # 1. 데이터 병합
        merged = self._merge_micro_macro(micro_data, macro_data)

        # 2. Granger test (micro → target)
        for micro_var in self.micro_features:
            if micro_var in merged.columns:
                p_value = self._granger_test(
                    merged[micro_var],
                    merged[target],
                    maxlag=5
                )

                if p_value < self.significance:
                    self.network.add_edge(
                        source=micro_var,
                        target=target,
                        weight=self._estimate_coefficient(
                            merged[micro_var], merged[target]
                        ),
                        p_value=p_value,
                        edge_type='micro_to_return'
                    )

        # 3. Micro 변수 간 관계
        for i, var1 in enumerate(self.micro_features):
            for var2 in self.micro_features[i+1:]:
                if var1 in merged.columns and var2 in merged.columns:
                    p_value = self._granger_test(
                        merged[var1], merged[var2], maxlag=3
                    )
                    if p_value < self.significance:
                        self.network.add_edge(
                            source=var1, target=var2,
                            weight=self._estimate_coefficient(
                                merged[var1], merged[var2]
                            ),
                            p_value=p_value,
                            edge_type='micro_to_micro'
                        )

        return self.network

    def _merge_micro_macro(
        self,
        micro_data: pd.DataFrame,
        macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        마이크로/매크로 데이터 병합

        마이크로: 5분 간격
        매크로: 일간 → 5분으로 리샘플링 (forward fill)
        """
        # 매크로 데이터 5분 리샘플링
        macro_resampled = macro_data.resample('5min').ffill()

        # 병합
        merged = micro_data.join(macro_resampled, how='inner')

        return merged.dropna()
```

### 9.2 사용 예시

```python
# main2.py에서 사용

async def run_with_microstructure():
    # 1. 매크로 데이터 수집
    collector = UnifiedDataCollectorV2()
    macro_data = await collector.collect()

    # 2. 마이크로 데이터 수집 (별도 수집기)
    micro_collector = MicrostructureCollector(symbol='SPY')
    micro_data = await micro_collector.collect_5min_features()

    # 3. 네트워크 구축 (확장)
    network_builder = ExtendedEconomicNetworkBuilder()
    network = network_builder.build(macro_data)
    network = network_builder.add_micro_edges(micro_data, macro_data)

    # 4. IRF 계산 (micro 포함)
    irf = network_builder.calculate_irf(
        shock_var='vpin',  # VPIN 충격
        response_vars=['ret_spy_5min', 'd_VIX', 'spread_bps']
    )

    return network, irf
```

---

## 10. LASSOForecaster Intraday Horizon

### 10.1 설정 확장

```yaml
# configs/horizon_config.yaml

horizons:
  # 기존 (일간)
  VeryShort:
    max_days: 30
    min_observations: 50

  Short:
    min_days: 31
    max_days: 90
    min_observations: 30

  Long:
    min_days: 180
    min_observations: 20

  # 신규 (분단위)
  intraday:
    enabled: true
    intervals:
      - name: "1min"
        seconds: 60
        min_observations: 1000
      - name: "5min"
        seconds: 300
        min_observations: 500
      - name: "15min"
        seconds: 900
        min_observations: 200

    # days_to_meeting 대체
    use_time_to_event: true  # 다음 뉴스/이벤트까지 시간
```

### 10.2 Intraday LASSO Forecaster

```python
class IntradayLASSOForecaster:
    """
    분 단위 LASSO 예측기

    기존 days_to_meeting 대신:
    - time_to_next_event (분 단위)
    - 마이크로스트럭처 features 사용
    """

    def __init__(self, horizon_minutes: int = 5):
        self.horizon = horizon_minutes
        self.model = LassoCV(cv=TimeSeriesSplit(n_splits=5))
        self.scaler = StandardScaler()

        # 마이크로스트럭처 features
        self.micro_features = [
            'ofi', 'vpin', 'spread_bps', 'depth_imbalance',
            'trade_imbalance', 'volatility_1m'
        ]

        # 매크로 features (5분 리샘플링)
        self.macro_features = [
            'd_VIX_5min', 'Ret_SPY_lag1', 'Ret_QQQ_lag1'
        ]

    def prepare_data(
        self,
        micro_data: pd.DataFrame,
        macro_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터 준비

        Args:
            micro_data: 마이크로스트럭처 features (1분 간격)
            macro_data: 매크로 데이터 (5분 리샘플링)

        Returns:
            X (features), y (target return)
        """
        # 5분 리샘플링
        micro_5min = micro_data.resample('5min').mean()

        # Target: 다음 5분 return
        micro_5min['target'] = (
            micro_5min['mid_price'].shift(-1) / micro_5min['mid_price'] - 1
        ) * 10000  # bps

        # 병합
        data = micro_5min.join(macro_data, how='inner')

        # Feature/Target 분리
        feature_cols = self.micro_features + self.macro_features
        X = data[[c for c in feature_cols if c in data.columns]]
        y = data['target']

        # 결측치 제거
        valid = X.notna().all(axis=1) & y.notna()
        X = X[valid]
        y = y[valid]

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """모델 학습"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # 선택된 변수 로깅
        coefs = pd.Series(self.model.coef_, index=X.columns)
        selected = coefs[coefs != 0]
        logger.info(f"Selected features: {selected.to_dict()}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """성능 메트릭"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        return {
            'r_squared': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'direction_accuracy': np.mean(np.sign(y) == np.sign(y_pred)),
            'n_features_selected': np.sum(self.model.coef_ != 0),
            'optimal_alpha': self.model.alpha_
        }
```

---

## 11. 백테스트 프레임워크

### 11.1 목표 지표

```
Sharpe Ratio: > 2.0
Win Rate: > 65%
Max Drawdown: < 1.5%
Transaction Cost: 0.02% (2bps)
```

### 11.2 구현

```python
class MicrostructureBacktester:
    """
    마이크로스트럭처 전략 백테스터

    2024-2025 SPY 1분 데이터 사용
    """

    def __init__(
        self,
        transaction_cost_bps: float = 2.0,  # 0.02%
        position_size: float = 1.0          # 전체 자본
    ):
        self.tx_cost = transaction_cost_bps / 10000
        self.position_size = position_size
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,  # +1: long, -1: short, 0: flat
        prices: pd.Series
    ) -> Dict:
        """
        백테스트 실행

        Args:
            data: feature 데이터
            signals: 시그널 시리즈
            prices: 가격 시리즈

        Returns:
            백테스트 결과
        """
        equity = 1.0
        position = 0
        entry_price = 0

        for i in range(len(signals)):
            signal = signals.iloc[i]
            price = prices.iloc[i]

            # 포지션 변경
            if signal != position:
                # 기존 포지션 청산
                if position != 0:
                    pnl = (price / entry_price - 1) * position
                    pnl -= self.tx_cost  # 청산 비용
                    equity *= (1 + pnl)

                    self.trades.append({
                        'exit_time': data.index[i],
                        'exit_price': price,
                        'pnl': pnl,
                        'side': 'long' if position > 0 else 'short'
                    })

                # 신규 포지션 진입
                if signal != 0:
                    entry_price = price * (1 + self.tx_cost * np.sign(signal))
                    position = signal

                    self.trades[-1] if self.trades else None
                    # 진입 기록

            self.equity_curve.append(equity)

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """성과 지표 계산"""
        if not self.equity_curve:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Sharpe Ratio (연율화, 1분 데이터 기준)
        # 1년 ≈ 252일 × 6.5시간 × 60분 = 98,280분
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(98280)

        # Win Rate
        trade_pnls = [t['pnl'] for t in self.trades if 'pnl' in t]
        win_rate = np.mean([p > 0 for p in trade_pnls]) if trade_pnls else 0

        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # 총 수익률
        total_return = equity[-1] / equity[0] - 1

        return {
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'num_trades': len(self.trades),
            'avg_trade_pnl': np.mean(trade_pnls) if trade_pnls else 0,
            'profit_factor': (
                sum(p for p in trade_pnls if p > 0) /
                abs(sum(p for p in trade_pnls if p < 0) or 1)
            )
        }

    def plot_equity_curve(self) -> 'Figure':
        """Equity curve 시각화"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        axes[0].plot(self.equity_curve)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity')

        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak

        axes[1].fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown %')

        plt.tight_layout()
        return fig
```

### 11.3 전략 예시

```python
class OFI_VPIN_Strategy:
    """
    OFI + VPIN 복합 전략

    진입 조건:
    - OFI > threshold AND VPIN < 0.6 → LONG
    - OFI < -threshold AND VPIN < 0.6 → SHORT
    - VPIN > 0.8 → EXIT ALL (crash 위험)

    목표: Sharpe > 2.0, Win Rate > 65%
    """

    def __init__(
        self,
        ofi_threshold: float = 1000,
        vpin_safe_threshold: float = 0.6,
        vpin_danger_threshold: float = 0.8
    ):
        self.ofi_threshold = ofi_threshold
        self.vpin_safe = vpin_safe_threshold
        self.vpin_danger = vpin_danger_threshold

    def generate_signals(
        self,
        ofi: pd.Series,
        vpin: pd.Series
    ) -> pd.Series:
        """
        시그널 생성

        Returns:
            pd.Series: +1 (long), -1 (short), 0 (flat)
        """
        signals = pd.Series(0, index=ofi.index)

        # VPIN 안전 구간에서만 진입
        safe_zone = vpin < self.vpin_safe

        # OFI 시그널
        signals[safe_zone & (ofi > self.ofi_threshold)] = 1
        signals[safe_zone & (ofi < -self.ofi_threshold)] = -1

        # VPIN 위험 → 전량 청산
        signals[vpin > self.vpin_danger] = 0

        return signals


def run_backtest_example():
    """백테스트 실행 예시"""

    # 1. 데이터 로드 (2024-2025 SPY 1분)
    data = load_spy_1min_data('2024-01-01', '2025-12-25')

    # 2. Feature 계산
    features = MicrostructureFeatures()
    for _, row in data.iterrows():
        features.update_lob(row['lob'])
        features.update_trade(row['trade'])

    feature_df = pd.DataFrame([
        features.compute_features()
        for _ in range(len(data))
    ], index=data.index)

    # 3. OFI, VPIN 계산
    ofi = feature_df['ofi']
    vpin = feature_df['vpin'] if 'vpin' in feature_df else pd.Series(0, index=ofi.index)

    # 4. 전략 시그널
    strategy = OFI_VPIN_Strategy()
    signals = strategy.generate_signals(ofi, vpin)

    # 5. 백테스트
    backtester = MicrostructureBacktester(transaction_cost_bps=2.0)
    results = backtester.run(feature_df, signals, data['close'])

    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Return: {results['total_return']:.2%}")

    # 검증
    assert results['sharpe_ratio'] > 2.0, "Sharpe < 2.0"
    assert results['win_rate'] > 0.65, "Win rate < 65%"
    assert results['max_drawdown'] < 0.015, "Max DD > 1.5%"

    return results
```

---

## 12. 실시간 데이터 수집

### 12.1 Polygon.io WebSocket

```python
class PolygonMicrostructureCollector:
    """
    Polygon.io WebSocket 기반 실시간 LOB 수집

    5초 주기 LOB update 처리
    """

    def __init__(self, api_key: str, symbols: List[str] = ['SPY']):
        self.api_key = api_key
        self.symbols = symbols
        self.ws_url = "wss://socket.polygon.io/stocks"

        # 데이터 저장소
        self.lob_snapshots: Dict[str, Deque[LOBSnapshot]] = {
            s: deque(maxlen=10000) for s in symbols
        }
        self.trades: Dict[str, Deque[Trade]] = {
            s: deque(maxlen=100000) for s in symbols
        }

        # Feature 계산기
        self.feature_calculators: Dict[str, MicrostructureFeatures] = {
            s: MicrostructureFeatures() for s in symbols
        }

    async def connect(self):
        """WebSocket 연결"""
        async with websockets.connect(self.ws_url) as ws:
            # 인증
            await ws.send(json.dumps({
                "action": "auth",
                "params": self.api_key
            }))

            # 구독
            for symbol in self.symbols:
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "params": f"Q.{symbol},T.{symbol}"  # Quotes + Trades
                }))

            # 메시지 처리
            async for message in ws:
                await self._handle_message(json.loads(message))

    async def _handle_message(self, data: Dict):
        """메시지 처리"""
        for item in data:
            ev = item.get('ev')

            if ev == 'Q':  # Quote (LOB Level 1)
                await self._handle_quote(item)
            elif ev == 'T':  # Trade
                await self._handle_trade(item)

    async def _handle_quote(self, quote: Dict):
        """Quote 처리 → LOB 스냅샷"""
        symbol = quote['sym']
        snapshot = LOBSnapshot(
            timestamp=datetime.fromtimestamp(quote['t'] / 1000),
            symbol=symbol,
            bid_prices=[quote['bp']],
            bid_sizes=[quote['bs']],
            ask_prices=[quote['ap']],
            ask_sizes=[quote['as']],
            mid_price=(quote['bp'] + quote['ap']) / 2,
            spread=quote['ap'] - quote['bp'],
            spread_bps=(quote['ap'] - quote['bp']) / ((quote['bp'] + quote['ap']) / 2) * 10000
        )

        self.lob_snapshots[symbol].append(snapshot)
        self.feature_calculators[symbol].update_lob(snapshot)

    async def _handle_trade(self, trade: Dict):
        """Trade 처리"""
        symbol = trade['sym']
        t = Trade(
            timestamp=datetime.fromtimestamp(trade['t'] / 1000),
            symbol=symbol,
            price=trade['p'],
            size=trade['s'],
            side='buy' if trade.get('c', [0])[0] == 0 else 'sell',  # conditions
            dollar_volume=trade['p'] * trade['s'],
            trade_sign=1 if trade.get('c', [0])[0] == 0 else -1
        )

        self.trades[symbol].append(t)
        self.feature_calculators[symbol].update_trade(t)

    def get_current_features(self, symbol: str) -> Dict:
        """현재 마이크로스트럭처 features"""
        return self.feature_calculators[symbol].compute_features()


async def fetch_realtime_lob(
    collector: PolygonMicrostructureCollector,
    interval_seconds: float = 5.0
) -> AsyncIterator[Dict]:
    """
    5초 주기 LOB feature 스트림

    EIMAS main.py 통합용
    """
    while True:
        for symbol in collector.symbols:
            features = collector.get_current_features(symbol)
            yield {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'features': features
            }

        await asyncio.sleep(interval_seconds)
```

### 12.2 Alpaca WebSocket (대안)

```python
class AlpacaMicrostructureCollector:
    """
    Alpaca WebSocket 기반 실시간 수집

    장점: 무료/저렴, 안정적
    단점: Level 1만 제공 (Level 2는 유료)
    """

    def __init__(self, api_key: str, secret_key: str, symbols: List[str] = ['SPY']):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"  # 무료

    async def connect(self):
        """WebSocket 연결"""
        async with websockets.connect(self.ws_url) as ws:
            # 인증
            await ws.send(json.dumps({
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }))

            # 구독
            await ws.send(json.dumps({
                "action": "subscribe",
                "quotes": self.symbols,
                "trades": self.symbols
            }))

            async for message in ws:
                await self._handle_message(json.loads(message))

    # ... (Polygon과 유사한 처리 로직)
```

### 12.3 EIMAS main.py 통합

```python
# main2.py 확장

class EIMASv2WithMicrostructure(EIMASv2):
    """
    마이크로스트럭처 통합 EIMAS v2
    """

    def __init__(self, *args, polygon_api_key: str = None, **kwargs):
        super().__init__(*args, **kwargs)

        # 마이크로스트럭처 수집기
        if polygon_api_key:
            self.micro_collector = PolygonMicrostructureCollector(
                api_key=polygon_api_key,
                symbols=['SPY', 'QQQ']
            )
        else:
            self.micro_collector = None

        # 마이크로스트럭처 분석기
        self.ofi_calculator = OFICalculator(window_seconds=300)
        self.vpin_calculator = VPINCalculator(bucket_size=50000, n_buckets=50)
        self.micro_lasso = MicrostructureLASSO(horizon_minutes=5)
        self.critical_path = MicrostructureCriticalPath(
            self.vpin_calculator,
            self.ofi_calculator
        )

    async def run_realtime(self):
        """실시간 모드 실행"""
        if not self.micro_collector:
            raise ValueError("Polygon API key required for realtime mode")

        # 백그라운드로 WebSocket 연결
        asyncio.create_task(self.micro_collector.connect())

        # 5초마다 분석
        async for update in fetch_realtime_lob(self.micro_collector):
            # 1. 마이크로스트럭처 분석
            features = update['features']

            # 2. VPIN/OFI 업데이트
            # (실제로는 collector 내부에서 이미 업데이트됨)

            # 3. Critical Path 체크
            path_result = self.critical_path.check_path_18()

            if path_result['triggered']:
                logger.warning(f"Path 18 triggered: {path_result}")
                # 알림 발송 등

            # 4. 예측
            if self.micro_lasso.is_fitted:
                prediction = self.micro_lasso.predict(features)
                logger.info(f"5min prediction: {prediction}")

            # 5. 주기적 리포트
            await self._publish_update({
                'timestamp': update['timestamp'],
                'features': features,
                'path_18': path_result,
                'vpin': self.vpin_calculator.get_current_vpin(),
                'ofi': self.ofi_calculator.get_signal()
            })
```

---

## 13. 구현 로드맵

```
Phase 1: 데이터 인프라 (1주)
├── [ ] Polygon.io 계정 설정
├── [ ] LOB/Trade 스키마 정의
├── [ ] WebSocket 수집기 구현
└── [ ] 히스토리컬 데이터 다운로드 (백테스트용)

Phase 2: 핵심 지표 (1주)
├── [ ] OFICalculator 구현
├── [ ] VPINCalculator 구현
├── [ ] MicrostructureFeatures 구현
└── [ ] 단위 테스트

Phase 3: 예측 모델 (1주)
├── [ ] MicrostructureLASSO 구현
├── [ ] HawkesProcess 구현
├── [ ] QueuePositionTracker 구현
└── [ ] 모델 학습 및 검증

Phase 4: EIMAS 통합 (1주)
├── [ ] Critical Path 18 추가
├── [ ] EconomicNetworkBuilder 확장
├── [ ] main2.py 실시간 모드
└── [ ] 대시보드 마이크로스트럭처 섹션

Phase 5: 백테스트 및 최적화 (1주)
├── [ ] MicrostructureBacktester 구현
├── [ ] OFI_VPIN_Strategy 백테스트
├── [ ] Sharpe > 2.0 달성 검증
└── [ ] 파라미터 최적화
```

---

## 14. 참고 문헌

1. **Cont, R., Kukanov, A., & Stoikov, S. (2014)**. "The Price Impact of Order Book Events." *Journal of Financial Econometrics*.

2. **Easley, D., López de Prado, M., & O'Hara, M. (2012)**. "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies*.

3. **Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015)**. "Hawkes Processes in Finance." *Market Microstructure and Liquidity*.

4. **Cartea, Á., Jaimungal, S., & Penalva, J. (2015)**. "Algorithmic and High-Frequency Trading." *Cambridge University Press*.

5. **JMLR (2020)**. "High-Dimensional Vector Autoregression for Intraday Returns." *Journal of Machine Learning Research*.

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-12-25 | v1.0 | 초기 문서 작성 |
