---
name: eimas-error-patterns
description: EIMAS 파이프라인 알려진 에러 패턴 카탈로그. $ARGUMENTS로 에러 메시지를 전달하면 패턴 매칭 후 해결책 제시.
argument-hint: "<에러 메시지 또는 에러 키워드>"
user-invocable: false
---

# EIMAS 에러 패턴 카탈로그

## 매칭 방법

`$ARGUMENTS`가 있으면 아래 패턴과 키워드 매칭. 없으면 전체 목록 출력.

---

## Import / 모듈 에러

| 키워드 | 원인 | 해결 |
|---|---|---|
| `networkx ImportError` | `NETWORKX_AVAILABLE` 플래그 미선언 | `lib/causality/builder.py` → `try/except ImportError` 감싸기 |
| `ModuleNotFoundError: arch` | 선택적 의존성 미설치 | `pip install arch` 또는 safe import 추가 |
| `cannot import name 'X' from pipeline` | 스키마 필드명 변경 | `pipeline/schemas.py` 클래스 정의 확인 |

## 데이터 타입 에러

| 키워드 | 원인 | 해결 |
|---|---|---|
| `FREDSummary.get()` / `AttributeError` | FREDSummary는 dataclass, dict 아님 | `.get('key')` → `.key` 속성 직접 접근 |
| `float(Series)` / `ValueError` | Series를 scalar로 캐스팅 | `RegimeDetector.to_scalar()` 확인, `.item()` 사용 |
| `dict vs AgentOpinion` | VerificationAgent 객체 타입 불일치 | `core/schemas.py` `AgentOpinion` 정의 확인 |

## 파이프라인 실행 에러

| 키워드 | 원인 | 해결 |
|---|---|---|
| `LASSO 0 variables` | 알파 범위 너무 넓음 | `LassoCV(alphas=np.logspace(-4, 0, 50))` + top-3 fallback |
| `토론 gridlock` | 에이전트 관점 동질성 | `institutional_bias` 파라미터로 GS/미래에셋/신한 관점 강제 |
| `portfolio bounds violation` | HRP가 cash ETF 미배분 | `cash_min: 0.0`, `commodity_max: 0.20`으로 완화 |
| `critical path ticker 누락` | `_FI_MARKET_TICKERS` 미정의 | `lib/critical_path.py`에 HYG, LQD, XLY, XLP, XLF, SMH, NVDA, EEM 추가 |
| `VIXTermStructureResult attribute` | 필드명 불일치 | `AIReportGenerator` 내 속성 접근 확인 |

## 환경 에러

| 키워드 | 원인 | 해결 |
|---|---|---|
| `SKIP_KOREA_SAVINGS` | Korea 지표 수집 실패 | `SKIP_KOREA_SAVINGS=1 python main.py --full` |
| `API key` / `AuthenticationError` | 환경변수 미설정 | `env | grep -E "ANTHROPIC|OPENAI|FRED|PERPLEXITY"` 확인 |
| `DB not found` | SQLite DB 미생성 | `python main.py --short` 먼저 실행해 DB 초기화 |

## 진단 명령어

```bash
# import 체인 전체 확인
python -c "
from pipeline.schemas import EIMASResult
from pipeline.collectors import collect_fred_data
from agents.orchestrator import MetaOrchestrator
print('All imports OK')
"

# 최근 실행 에러 필드
ls -t outputs/eimas_*.json | head -1 | xargs jq '.errors // empty'

# 환경변수 확인 (값 숨김)
env | grep -E "ANTHROPIC|OPENAI|FRED|PERPLEXITY|GOOGLE" | sed 's/=.*/=<set>/'

# DB 상태
sqlite3 data/trading.db ".tables" 2>/dev/null || echo "DB not found"
```
