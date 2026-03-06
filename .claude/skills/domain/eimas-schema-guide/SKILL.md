---
name: eimas-schema-guide
description: EIMAS 핵심 데이터 스키마 레퍼런스. pipeline/schemas.py 기준 주요 클래스·필드·import 경로 정리.
user-invocable: false
---

# EIMAS 스키마 레퍼런스

소스: `pipeline/schemas.py` (분석 결과), `core/schemas.py` (에이전트 통신)

## 주요 클래스

| 클래스 | 역할 | import |
|---|---|---|
| `EIMASResult` | 전체 분석 결과 컨테이너 | `from pipeline.schemas import EIMASResult` |
| `FREDSummary` | FRED 거시지표 요약 | `from pipeline.schemas import FREDSummary` |
| `RegimeResult` | 시장 레짐 분류 | `from pipeline.schemas import RegimeResult` |
| `DebateResult` | AI 토론 결과 | `from pipeline.schemas import DebateResult` |
| `PortfolioResult` | 포트폴리오 최적화 결과 | `from pipeline.schemas import PortfolioResult` |
| `CriticalPathResult` | Critical Path 분석 | `from pipeline.schemas import CriticalPathResult` |
| `AgentOpinion` | 에이전트 개별 의견 | `from core.schemas import AgentOpinion` |

## FREDSummary 핵심 필드

```python
fed_funds: float          # 기준금리
treasury_2y / 10y / 30y   # 국채 수익률
spread_10y2y: float       # 장단기 스프레드 (역전 시 음수)
hy_oas: float             # HY 스프레드
cpi_yoy / core_pce_yoy    # 인플레이션
net_liquidity: float      # Fed BS - RRP - TGA
liquidity_regime: str     # "Tightening" / "Normal" / "Easing"
curve_inverted: bool      # 수익률 곡선 역전 여부
```

> FREDSummary는 dataclass — `.get()` 사용 불가, 반드시 `.필드명` 직접 접근

## RegimeResult 핵심 필드

```python
regime: str               # "Bull" / "Neutral" / "Bear"
confidence: float         # 0~1
bull_prob / bear_prob: float
risk_score: float         # 0~100 (낮을수록 안전)
risk_level: str           # "LOW" / "MEDIUM" / "HIGH"
```

## EIMASResult 핵심 필드

```python
timestamp: str
final_recommendation: str        # "BULLISH" / "NEUTRAL" / "BEARISH"
risk_score: float
fred_summary: FREDSummary
regime_result: RegimeResult
debate_result: DebateResult
portfolio_weights: Dict[str, float]
errors: List[str]                # 실행 중 발생한 에러 목록
phase_timings: Dict[str, float]  # Phase별 소요 시간(초)
```

## 직렬화

```python
result.to_dict()      # JSON-serializable dict
result.to_markdown()  # 사람이 읽는 리포트 문자열
```
