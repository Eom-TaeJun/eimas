# EIMAS MCP 서버 — 하니스 플러그인 연동 가이드

> EIMAS 분석 결과를 Claude Code 하니스 플러그인에서 MCP 도구로 사용하는 방법

---

## 개요

EIMAS MCP 서버(`mcp_eimas_server.py`)는 EIMAS의 분석 결과를 Claude Code 플러그인에서
MCP 도구로 노출한다. 이를 통해 AI 에이전트가 EIMAS 레짐 판단·시그널·리포트를 직접 참조할 수 있다.

```
Claude Code 하니스 플러그인
       ↓ MCP 호출
mcp_eimas_server.py
  ├── Primary:  EIMAS FastAPI (localhost:8000) ← 실시간
  └── Fallback: outputs/*.json + data/events.db ← 파일 기반
```

---

## 이중 모드 (Dual-mode)

| 상태 | 동작 |
|------|------|
| `python main.py --full` 실행 중 | FastAPI `/api/*` 엔드포인트 실시간 호출 |
| API 꺼져 있음 | `outputs/*.json`, `data/events.db` 직접 읽기 (자동 폴백) |

→ EIMAS가 실행 중이든 아니든 MCP 도구가 항상 동작.

---

## 제공 도구 (9개)

| 도구 | 설명 | 소스 |
|------|------|------|
| `eimas_status` | API 실행 여부, 파일 존재, 최신 분석 타임스탬프 | 파일 시스템 |
| `get_regime` | 현재 레짐 (BULLISH/BEARISH/NEUTRAL), VIX, RSI | API → 파일 |
| `get_regime_history` | 레짐 이력, 전환 패턴, 필터링 가능 | regime_history.json |
| `get_signals` | 에이전트 합의 BUY/SELL/HOLD, conviction | API → 파일 |
| `get_latest_analysis` | 종합 분석 결과 (섹션 선택 가능) | eimas_*.json |
| `get_risk_metrics` | VaR, Sharpe, 변동성, 리스크 레벨 | API → 파일 |
| `get_sector_rotation` | 경기사이클, 오버/언더웨이트 섹터 | API → 파일 |
| `query_events` | events.db 이벤트 조회 (읽기 전용) | SQLite |
| `get_ai_report` | AI 에이전트 리포트 요약 | ai_report_*.json |

---

## MCP 도구 호출 예시

```python
# 1. 상태 확인 (항상 먼저)
mcp__finance-analysis-harness_eimas__eimas_status()

# 2. 현재 레짐
mcp__finance-analysis-harness_eimas__get_regime(ticker="SPY")
# → {"regime": "BULLISH", "confidence": 0.75, "vix": 17.4, "recommendation": "BUY"}

# 3. 최신 종합 분석
mcp__finance-analysis-harness_eimas__get_latest_analysis(section="recommendation")
# → {"final_recommendation": "BULLISH", "confidence": 0.72, "executive_summary": "..."}

# 4. 레짐 이력 (최근 10개, 베어리시만)
mcp__finance-analysis-harness_eimas__get_regime_history(limit=10, regime_filter="BEARISH")

# 5. AI 리포트
mcp__finance-analysis-harness_eimas__get_ai_report(section="executive_summary")
```

---

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install mcp requests
```

### 2. .mcp.json 설정

하니스 플러그인의 `.mcp.json`에 이미 등록되어 있음:

```json
{
  "mcpServers": {
    "eimas": {
      "command": "python",
      "args": ["${CLAUDE_PLUGIN_ROOT}/mcp_servers/mcp_eimas_server.py"],
      "env": {
        "EIMAS_ROOT": "${EIMAS_ROOT:-/home/tj/projects/autoai/eimas}",
        "EIMAS_API_URL": "${EIMAS_API_URL:-http://localhost:8000}"
      }
    }
  }
}
```

### 3. 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EIMAS_ROOT` | `/home/tj/projects/autoai/eimas` | EIMAS 루트 경로 |
| `EIMAS_API_URL` | `http://localhost:8000` | FastAPI 서버 URL |

---

## 워크플로우: 하니스 + EIMAS 통합 분석

```
/analyze "현재 시장 분석"
    ↓
Step 0: EIMAS 선행 컨텍스트 로드
  mcp..eimas..eimas_status()
  mcp..eimas..get_regime()          → 현재 레짐 확인
  mcp..eimas..get_latest_analysis() → EIMAS 추천 확인
    ↓
Step 1: FRED + 시장 데이터 보완 수집 (레짐에 집중된 지표)
    ↓
Step 2~5: data-validator → macro-analyst → signal-interpreter → report-writer
          (EIMAS 레짐을 시작점으로 심화 분석)
```

---

## EIMAS 출력 파일 위치

```
eimas/
├── outputs/
│   ├── eimas_*.json              ← 메인 분석 결과
│   ├── real_analysis_result.json ← 고정 결과 파일
│   ├── regime_history.json       ← 레짐 이력
│   └── ai_report_*.json          ← AI 에이전트 리포트
└── data/
    └── events.db                 ← 이벤트 SQLite DB
```

---

## MCP 서버 파일 위치

```
harness_engineering/example-plugin/
└── mcp_servers/
    └── mcp_eimas_server.py   ← 이 파일
```
