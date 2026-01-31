# EIMAS Realtime Architecture Analysis

> 작성일: 2026-01-31
> 목적: `--realtime` 기능의 현재 상태 분석 및 개선 방향 정리

---

## 1. 현재 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    EIMAS Pipeline Structure                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Phase 1-3] 정적 분석 (Daily Data)                              │
│  ├─ [1.1] FRED 거시지표 (RRP, TGA, Net Liquidity)                │
│  ├─ [1.2] yfinance 시장 데이터 (SPY, QQQ 등 24개)                 │
│  ├─ [1.3] 크립토/RWA 데이터                                       │
│  ├─ [2.x] 분석 모듈들                                            │
│  │   ├─ RegimeDetector (Bull/Bear/Neutral)                      │
│  │   ├─ CriticalPathAggregator (리스크 점수)                     │
│  │   ├─ BubbleDetector (버블 경고)                               │
│  │   ├─ GC-HRP Portfolio (포트폴리오 최적화)                      │
│  │   └─ DTW/DBSCAN (시계열 유사도/이상치)                         │
│  ├─ [3.x] Multi-Agent Debate                                    │
│  │   ├─ 7개 에이전트 의견 수렴                                    │
│  │   ├─ Claude/GPT 토론                                         │
│  │   └─ 합의 도출                                                │
│  └─► 결과: BULLISH/BEARISH + Confidence + Risk Level            │
│                                                                 │
│  ════════════════════════════════════════════════════════════   │
│                     ↓ 연결 없음 (Disconnected) ↓                  │
│  ════════════════════════════════════════════════════════════   │
│                                                                 │
│  [Phase 4] 실시간 스트리밍 (WebSocket)                            │
│  ├─ Binance WebSocket 연결                                       │
│  │   └─ 심볼: BTCUSDT (기본)                                     │
│  ├─ 수집 지표                                                    │
│  │   ├─ OFI (Order Flow Imbalance): 주문 흐름 불균형              │
│  │   ├─ VPIN: 정보비대칭 확률                                     │
│  │   └─ Depth Ratio: 호가창 불균형                                │
│  ├─ 처리 방식                                                    │
│  │   └─ 단순 수집 → DB 저장                                      │
│  └─► 결과: List[RealtimeSignal] (미활용)                         │
│                                                                 │
│  [Phase 5-7] 저장 및 리포트                                       │
│  ├─ JSON/MD 저장                                                 │
│  ├─ AI 리포트 생성                                               │
│  └─ Whitening & Fact Check                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 현재 문제점

### 2.1 구조적 문제

| 문제 | 설명 | 영향 |
|------|------|------|
| **연결 단절** | Phase 4가 Phase 1-3과 독립적으로 실행 | 실시간 데이터가 분석에 반영 안됨 |
| **피드백 루프 없음** | 실시간 지표 → 레짐 업데이트 경로 없음 | 레짐 전환 감지 불가 |
| **알림 시스템 부재** | VPIN 급등 등 이상 상황 알림 없음 | 실시간 대응 불가 |
| **단일 심볼** | BTCUSDT만 모니터링 | 시장 전체 파악 제한 |

### 2.2 기능적 문제

```python
# 현재 코드 (pipeline/realtime.py)
def on_metrics(metrics):
    signal = RealtimeSignal(
        timestamp=datetime.now().isoformat(),
        symbol=symbols[0],
        ofi=getattr(metrics, 'ofi', 0.0),
        vpin=getattr(metrics, 'vpin', 0.0),
        signal=getattr(metrics, 'signal', 'neutral')  # ← 단순 저장만
    )
    signals_collected.append(signal)  # ← 분석 로직 없음
```

- 수집된 OFI/VPIN을 **해석하는 로직 없음**
- Phase 3 결과(BULLISH/BEARISH)와 **교차 검증 없음**
- 경제학적 맥락(유동성, 레짐) **미적용**

### 2.3 데이터 흐름 문제

```
현재:
  Phase 1-3 ──────────────────► 최종 결과
                                    │
  Phase 4 ──► DB 저장 ──► (미사용)  │
                                    ↓
                              리포트 생성

이상적:
  Phase 1-3 ◄─────────────────► Phase 4
       │                           │
       ▼                           ▼
  거시 레짐 ◄── 실시간 확인 ──► 미시 시그널
       │                           │
       └───────────┬───────────────┘
                   ▼
            통합 의사결정
```

---

## 3. 개선 방향

### 3.1 가능한 목적별 개선안

| 목적 | 설명 | 구현 복잡도 | 가치 |
|------|------|------------|------|
| **A. 실시간 이상탐지** | VPIN > 0.7 → "정보거래자 유입" 경고 | 낮음 | 중간 |
| **B. 시그널 확인** | 거시(BULLISH) + 미시(OFI+) → 진입 확신 | 중간 | 높음 |
| **C. 레짐 전환 감지** | 실시간 지표로 레짐 변화 선행 포착 | 높음 | 매우 높음 |
| **D. 진입/청산 타이밍** | 거시 조건 충족 시 최적 진입점 탐색 | 중간 | 높음 |
| **E. 리스크 모니터링** | 포지션 보유 중 실시간 위험 감시 | 중간 | 높음 |

### 3.2 권장 개선 순서

```
Phase 1: 기본 연결 (1-2일)
├─ Phase 3 결과를 Phase 4에 전달
├─ 레짐에 따른 VPIN/OFI 임계값 차등 적용
└─ 간단한 알림 로직 추가

Phase 2: 시그널 통합 (2-3일)
├─ 거시 시그널 + 미시 시그널 교차 검증
├─ Confidence 조정 로직
└─ 실시간 대시보드 연동

Phase 3: 고급 기능 (1주)
├─ 실시간 레짐 확률 업데이트
├─ 멀티 심볼 모니터링 (BTC, ETH, SOL)
└─ 자동 진입/청산 시그널 생성
```

### 3.3 아키텍처 개선안

```
┌─────────────────────────────────────────────────────────────────┐
│                    개선된 파이프라인 구조                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Phase 1-3] 거시 분석                                           │
│  └─► MacroContext {                                             │
│        regime: "BULL",                                          │
│        confidence: 0.74,                                        │
│        risk_score: 9.6,                                         │
│        liquidity: "ABUNDANT"                                    │
│      }                                                          │
│            │                                                    │
│            ▼                                                    │
│  [Phase 4] 실시간 분석 (MacroContext 주입)                        │
│  ├─ RealtimeAnalyzer(macro_context)                             │
│  │   ├─ 레짐별 임계값 적용                                        │
│  │   │   ├─ BULL: VPIN > 0.8 경고 (높은 임계값)                  │
│  │   │   └─ BEAR: VPIN > 0.6 경고 (낮은 임계값)                  │
│  │   ├─ OFI 방향 vs 레짐 일치 확인                                │
│  │   │   ├─ BULL + OFI+ → "Confirmed"                           │
│  │   │   └─ BULL + OFI- → "Divergence Warning"                  │
│  │   └─ 동적 Confidence 조정                                     │
│  │                                                              │
│  └─► RealtimeSignal {                                           │
│        base_signal: "BULLISH",                                  │
│        realtime_confirmation: true,                             │
│        adjusted_confidence: 0.82,  // 상향 조정                  │
│        alerts: []                                               │
│      }                                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 핵심 구현 포인트

### 4.1 MacroContext 전달

```python
# 개선된 구조
@dataclass
class MacroContext:
    regime: str           # "BULL", "BEAR", "NEUTRAL"
    confidence: float     # 0-1
    risk_score: float     # 0-100
    liquidity_signal: str # "ABUNDANT", "TIGHT", "NEUTRAL"
    primary_risk: str     # 주요 리스크 요인

async def run_realtime_stream(
    duration: int,
    symbols: List[str],
    macro_context: MacroContext  # ← 추가
) -> List[RealtimeSignal]:
    ...
```

### 4.2 레짐 기반 임계값

```python
# 레짐별 VPIN 임계값
VPIN_THRESHOLDS = {
    "BULL": 0.80,   # 상승장에서는 높은 VPIN도 허용
    "NEUTRAL": 0.70,
    "BEAR": 0.60,   # 하락장에서는 낮은 VPIN도 경고
}

def check_vpin_alert(vpin: float, regime: str) -> bool:
    return vpin > VPIN_THRESHOLDS.get(regime, 0.70)
```

### 4.3 시그널 확인 로직

```python
def confirm_signal(macro_signal: str, ofi: float) -> dict:
    """거시 시그널과 실시간 OFI 교차 검증"""

    if macro_signal == "BULLISH":
        if ofi > 0.3:
            return {"status": "CONFIRMED", "adjustment": +0.1}
        elif ofi < -0.3:
            return {"status": "DIVERGENCE", "adjustment": -0.15}

    elif macro_signal == "BEARISH":
        if ofi < -0.3:
            return {"status": "CONFIRMED", "adjustment": +0.1}
        elif ofi > 0.3:
            return {"status": "DIVERGENCE", "adjustment": -0.15}

    return {"status": "NEUTRAL", "adjustment": 0}
```

---

## 5. 다음 단계

- [ ] 목적 결정: 이상탐지 / 시그널확인 / 레짐감지 / 타이밍최적화
- [ ] MacroContext 인터페이스 설계
- [ ] 임계값 파라미터 설정
- [ ] 알림 시스템 구현 (Slack/Discord/Telegram)
- [ ] 실시간 대시보드 연동

---

*마지막 업데이트: 2026-01-31*
