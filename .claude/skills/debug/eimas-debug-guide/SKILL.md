---
name: eimas-debug-guide
description: EIMAS 파이프라인 에러 패턴 및 해결책 레퍼런스
user-invocable: false
---

# EIMAS 디버그 가이드

## 알려진 에러 패턴

### AttributeError: FREDSummary에서 `.get()` 호출
- **원인**: FREDSummary는 dataclass, dict 아님
- **위치**: `pipeline/phases/phase45_operational.py`
- **해결**: `.get('key')` → `.key` 속성 직접 접근

### LASSO 0변수 선택
- **원인**: 기본 알파 범위가 너무 넓음
- **위치**: `lib/lasso_model.py`
- **해결**: `LassoCV(alphas=np.logspace(-4, 0, 50))` + top-3 fallback

### 토론 gridlock (모든 에이전트 동의 또는 교착)
- **원인**: 에이전트 관점 다양성 부족
- **위치**: `agents/interpretation_debate.py`
- **해결**: institutional_bias 파라미터로 GS/미래에셋/신한 관점 강제

### networkx ImportError
- **원인**: `NETWORKX_AVAILABLE` 플래그 미선언
- **위치**: `lib/causality/builder.py`
- **해결**: `try/except ImportError`로 networkx import 감싸기

### 포트폴리오 bounds violation
- **원인**: HRP 알고리즘이 cash ETF 미배분
- **해결**: `cash_min: 0.0`, `commodity_max: 0.20`으로 완화

### 필요 티커 누락 (CriticalPath)
- **원인**: `_FI_MARKET_TICKERS` 미정의 티커
- **위치**: `lib/critical_path.py`
- **해결**: HYG, LQD, XLY, XLP, XLF, SMH, NVDA, EEM 추가

## 진단 명령어

```bash
# import 체인 확인
python -c "
from pipeline.schemas import EIMASResult
from pipeline.collectors import collect_fred_data
from agents.orchestrator import MetaOrchestrator
print('All imports OK')
"

# DB 상태 확인
sqlite3 data/trading.db ".tables" 2>/dev/null || echo "DB not found"

# 최근 실행 로그
ls -t outputs/eimas_*.json | head -1 | xargs jq '.errors // empty'

# Phase별 소요 시간 (있는 경우)
ls -t outputs/eimas_*.json | head -1 | xargs jq '.phase_timings // empty'
```

## Phase 1 Korea 지표 스킵

```bash
SKIP_KOREA_SAVINGS=1 python main.py --full
```

## 환경 변수 체크

```bash
env | grep -E "ANTHROPIC|OPENAI|FRED|PERPLEXITY|GOOGLE" | sed 's/=.*/=<set>/'
```
