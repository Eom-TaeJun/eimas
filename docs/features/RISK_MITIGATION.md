# EIMAS Risk Mitigation & Enhancement Plan

---

## 현재 취약점 분석

### 1. Multi-Agent Debate 실패 모드 (치명도: HIGH)

| 문제 | 현재 상태 | 영향 |
|------|----------|------|
| **Inter-Agent Misalignment** | 자연어 기반 의견 교환 | LASSO "4.5%" → Debate "불확실" 왜곡 |
| **Sycophancy** | 신뢰도 가중치 없음 | 약한 에이전트가 강한 의견 추종 |
| **Verification 부재** | Debate 후 검증 없음 | 잘못된 Consensus → DB 오염 |

**현재 코드 위치**: `agents/orchestrator.py`, `core/debate.py`

```python
# 현재: Rule-based consensus (85% 일관성)
# 문제: 일관성 ≠ 정확성
```

### 2. 퀀트 모델 리스크 (치명도: MEDIUM-HIGH)

| 문제 | 현재 상태 | 영향 |
|------|----------|------|
| **Overfitting** | 2024-25 데이터 최적화 | Regime 변화 시 실패 |
| **Data Quality** | yfinance 무료 API | 생존자 편향, 결측 |
| **Point Forecast** | 단일 예측값 | 불확실성 무시 |
| **Execution** | Signal만 생성 | Slippage/비용 무시 |

**현재 코드 위치**: `lib/lasso_model.py`, `agents/forecast_agent.py`

### 3. 운영 리스크 (치명도: MEDIUM)

| 문제 | 현재 상태 | 영향 |
|------|----------|------|
| **Alert Fatigue** | 필터링 없음 | 100+개/일 → 무시 |
| **Black Swan** | Event Framework 한정적 | 지정학 미커버 |
| **Audit Trail** | 부분적 저장 | 결정 근거 추적 어려움 |

---

## 즉시 구현 가능한 개선안

### Phase 1: Debate 강화 (오늘)

#### 1.1 Verification Agent 추가

```
현재: Agent → Debate → Consensus → Output
개선: Agent → Debate → Consensus → Verifier → Output
                                      ↓
                            [REJECT → Re-debate]
```

#### 1.2 Structured Output 강제

```python
# 현재: 자유 텍스트
opinion = "I think the market will go up..."

# 개선: Pydantic Schema
class StructuredOpinion(BaseModel):
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence: float  # 0.0 - 1.0
    time_horizon: int  # days
    key_factors: List[str]
    risks: List[str]
    numerical_target: Optional[float]  # e.g., Fed rate
```

#### 1.3 Agent Diversity

```
현재: Forecast, Analysis, Research, Strategy (4명)

개선:
- Bull Advocate (강제 긍정)
- Bear Advocate (강제 부정)
- Devil's Advocate (반론 전문)
- Fact Checker (사실 검증)
- Synthesizer (종합)
```

---

### Phase 2: Model Robustness (3일)

#### 2.1 Walk-Forward OOS Test

```python
# 매주 실행
def walk_forward_test(data, train_window=252, test_window=5):
    """
    최근 1주 데이터로 LASSO 재학습 후 OOS 성능 측정
    """
    results = []
    for t in rolling_windows:
        train = data[t-train_window:t]
        test = data[t:t+test_window]

        model.fit(train)
        pred = model.predict(test)

        # Overfitting 탐지
        in_sample_r2 = model.score(train)
        out_sample_r2 = model.score(test)

        if in_sample_r2 - out_sample_r2 > 0.3:
            alert("OVERFITTING DETECTED")

        results.append(out_sample_r2)

    return results
```

#### 2.2 Quantile Regression

```python
# 현재: Point forecast
forecast = model.predict(X)  # Fed rate = 4.5%

# 개선: Uncertainty quantiles
from sklearn.linear_model import QuantileRegressor

forecasts = {
    'p10': QuantileRegressor(quantile=0.10).fit(X, y).predict(X_new),
    'p50': QuantileRegressor(quantile=0.50).fit(X, y).predict(X_new),
    'p90': QuantileRegressor(quantile=0.90).fit(X, y).predict(X_new),
}
# Fed rate: 4.2% (10th) - 4.5% (median) - 4.9% (90th)
# 80% CI = [4.2%, 4.9%]
```

#### 2.3 Stress Test Suite

| Scenario | Parameters | 예상 반응 |
|----------|-----------|----------|
| RRP Drain | RRP -$500B | Net Liquidity ↑, Signal: BUY |
| VIX Spike | VIX → 80 | Risk-Off, Signal: SELL |
| Yield Inversion | 10Y-2Y = -100bp | Recession flag |
| TGA Drain | TGA -$300B | Liquidity injection |
| Flash Crash | SPY -10% intraday | Circuit breaker |

```python
def run_stress_test(scenario: str, pipeline):
    """시나리오별 Signal 안정성 측정"""
    synthetic_data = generate_stress_scenario(scenario)
    signals = pipeline.process(synthetic_data)

    return {
        'scenario': scenario,
        'signal_stability': measure_stability(signals),
        'false_alarm_rate': count_false_alarms(signals),
        'response_time': measure_latency(signals),
    }
```

---

### Phase 3: Risk Management (1주)

#### 3.1 Position Sizing (Kelly Criterion)

```python
def kelly_position(signal_confidence: float,
                   win_rate: float,
                   avg_win: float,
                   avg_loss: float) -> float:
    """
    Kelly Criterion with Signal Confidence adjustment

    f* = (p*b - q) / b
    where:
        p = win rate * confidence
        q = 1 - p
        b = avg_win / avg_loss
    """
    p = win_rate * signal_confidence
    q = 1 - p
    b = avg_win / abs(avg_loss)

    kelly = (p * b - q) / b

    # Half-Kelly for safety
    return max(0, min(kelly / 2, 0.25))  # Max 25%
```

#### 3.2 Kill Switch

```python
class RiskController:
    def __init__(self, max_drawdown: float = 0.05):
        self.max_dd = max_drawdown
        self.peak_equity = 0
        self.current_equity = 0
        self.killed = False

    def update(self, equity: float):
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        if drawdown >= self.max_dd:
            self.killed = True
            self.send_alert("KILL SWITCH ACTIVATED: MDD exceeded")
            return "HALT_ALL_SIGNALS"

        return "OK"
```

#### 3.3 Alert Filtering

```python
class AlertFilter:
    """Critical만 전송, 나머지는 로그"""

    RULES = {
        'telegram': {
            'min_confidence': 0.7,
            'regime_match': True,
            'min_importance': 'HIGH',
        },
        'log_only': {
            'min_confidence': 0.0,
        }
    }

    def should_send(self, signal: IntegratedSignal) -> bool:
        if signal.confidence < 0.7:
            return False

        if signal.combined_signal in ['neutral']:
            return False

        if signal.alerts and 'CRITICAL' in signal.alerts:
            return True

        # Regime match check
        if signal.macro_signal != signal.micro_signal:
            return False  # Conflicting signals

        return True
```

#### 3.4 Audit Log

```python
@dataclass
class DecisionAudit:
    """모든 결정 추적 가능"""
    timestamp: datetime
    decision_id: str

    # Input
    market_data: Dict
    fred_data: Dict
    microstructure: Dict

    # Agent Opinions
    agent_opinions: List[AgentOpinion]
    debate_rounds: List[DebateRound]

    # Output
    consensus: Consensus
    final_signal: str
    confidence: float

    # Verification
    verifier_result: Optional[str]
    override_reason: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    def save(self, db: Database):
        db.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?)",
            (self.decision_id, self.timestamp, self.to_json())
        )
```

---

## 구현 우선순위

```
┌─────────────────────────────────────────────────────────────┐
│ Priority Matrix (Impact vs Effort)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HIGH    │ [1] Structured Output  │ [3] Stress Test        │
│  IMPACT  │ [2] Verification Agent │ [4] Quantile Reg       │
│          │                        │                         │
│  ────────┼────────────────────────┼────────────────────────│
│          │                        │                         │
│  LOW     │ [5] Alert Filter       │ [7] Social Sentiment   │
│  IMPACT  │ [6] Kill Switch        │ [8] Options Skew       │
│          │                        │                         │
│          │      LOW EFFORT        │     HIGH EFFORT        │
└─────────────────────────────────────────────────────────────┘
```

### 실행 순서

| 순서 | 항목 | 예상 시간 | 파일 |
|-----|------|----------|------|
| 1 | Structured Output (Pydantic) | 2시간 | `core/schemas.py` |
| 2 | Alert Filter | 1시간 | `lib/alert_manager.py` |
| 3 | Verification Agent | 3시간 | `agents/verifier.py` |
| 4 | Kill Switch | 1시간 | `lib/risk_manager.py` |
| 5 | Walk-Forward Test | 4시간 | `lib/backtest.py` |
| 6 | Quantile Regression | 3시간 | `lib/lasso_model.py` |
| 7 | Stress Test Suite | 4시간 | `tests/stress_test.py` |
| 8 | Audit Log | 2시간 | `core/database.py` |

---

## 성공 지표

| 지표 | 현재 | 목표 |
|-----|------|------|
| Debate Misalignment | ~37% | <15% |
| OOS R² Degradation | Unknown | <0.2 |
| False Alarm Rate | High | <10% |
| Signal Verification | 0% | 100% |
| Audit Coverage | 30% | 100% |

---

## 미해결 리스크 (향후)

1. **Black Swan 지정학**: X/Twitter 실시간 + Perplexity 통합 필요
2. **Execution Risk**: 실제 거래 연동 시 slippage 모델 필요
3. **Model Ensemble**: 단일 LASSO → XGBoost/RF 앙상블 고려
4. **Real-time News**: Bloomberg/Reuters 유료 API 고려

---

*Created: 2025-01-06*
*Status: Planning*
