# Harness Engineering — AI 에이전트 인프라 설계 가이드

> 작성일: 2026-02-20
> 기반 프로젝트: `/home/tj/projects/harness_engineering/example-plugin`

---

## 1. 개념: 컨텍스트 엔지니어링 → 하니스 엔지니어링

### 패러다임 전환

```
2023-2024: Context Engineering
  → LLM에게 무엇을 줄 것인가 (프롬프트, 예시, 지식)

2025-2026: Harness Engineering
  → LLM을 둘러싼 전체 시스템을 어떻게 설계할 것인가
```

**하니스 = LLM을 제외한 모든 것**
- Tools (무엇을 할 수 있나)
- State (무엇을 기억하나)
- Guardrails (무엇을 막나)
- Orchestration (어떻게 조율하나)

> "Model = 엔진, Harness = 자동차 전체"

### 왜 중요한가 (2026 현황)

- 82% 중견기업 / 95% PE 펌 → 에이전틱 AI 도입 시작 또는 계획
- 도입 조직 99% → 운영 효율성·생산성 개선 확인
- **단순 코드 생성 → 분석 파이프라인 설계**로 역할 진화
- AI 에이전트 구축 역량 = 2026년 퀀트 핵심 역량

---

## 2. 하니스 구조 5개 레이어

Claude Code 플러그인 기준:

```
하니스 레이어       Claude Code 구현체       역할
──────────────────────────────────────────────────────────
도메인 지식         Skills (SKILL.md)        컨텍스트 자동 주입
외부 도구 연동      MCP Servers (.mcp.json)  API, DB, 데이터 소스
자율 실행           Agents (agents/*.md)     단일 책임 서브프로세스
가드레일            Hooks (hooks/hooks.json) 이벤트 기반 제어
진입점              Commands (commands/*.md) 사용자 슬래시 커맨드
운영 매뉴얼         CLAUDE.md               매 세션 자동 로드
```

---

## 3. 금융 분석 예시 플러그인 전체 구조

```
example-plugin/
├── CLAUDE.md                          ← 운영 매뉴얼 (매 세션 자동 로드)
├── .claude-plugin/plugin.json         ← 플러그인 메타데이터
├── .mcp.json                          ← MCP 서버 등록
│
├── skills/                            ← 도메인 지식 (자동 활성화)
│   ├── macro-economics/SKILL.md       v1.2.0 — 거시경제 분석 표준
│   ├── financial-signals/SKILL.md     v1.1.0 — VIX/스프레드/이상탐지
│   ├── korean-finance/SKILL.md        v1.1.0 — BOK/원달러/KOSPI
│   ├── asset-class-universe/SKILL.md  v1.1.0 — 자산 클래스 프레임워크
│   ├── ai-finance-workflow/SKILL.md   v1.1.0 — AI×금융 워크플로우
│   ├── portfolio-theory/SKILL.md      v1.0.0 — HRP/MVF/팩터
│   ├── market-microstructure/SKILL.md v1.0.0 — 거래량/시장구조
│   └── analysis-standards/SKILL.md   v1.0.0 — 코드/차트 표준
│
├── agents/                            ← 에이전트 (단일 책임)
│   ├── data-validator.md              데이터 품질 검증
│   ├── macro-analyst.md               레짐 판단 + 정책 경로
│   ├── signal-interpreter.md          시그널 일관성 + 이상 탐지
│   ├── quant-coder.md                 코드 실행 + 시각화
│   └── report-writer.md               최종 리포트 저장
│
├── commands/                          ← 슬래시 커맨드
│   ├── analyze.md                     /analyze — 종합 분석 파이프라인
│   ├── regime.md                      /regime — 레짐 스냅샷 (5분)
│   ├── signal-check.md                /signal-check [지표] — 즉시 해석
│   └── report.md                      /report — 세션 결과 저장
│
├── hooks/
│   ├── hooks.json                     이벤트 훅 정의
│   └── scripts/
│       ├── load-market-context.sh     SessionStart: API 키/env 확인
│       └── validate-bash.sh           PreToolUse: 위험 명령어 차단
│
├── mcp_servers/                       ← MCP 서버 실제 구현
│   ├── mcp_fred_server.py             FRED API (거시 데이터)
│   ├── mcp_market_server.py           시장 가격/수급 데이터
│   └── requirements.txt               mcp, yfinance, pykrx
│
└── outputs/                           ← 분석 결과 저장
    ├── context/
    │   ├── SCHEMA.md                  에이전트 핸드오프 스키마
    │   ├── validation_result.json     ← data-validator 출력
    │   ├── regime_snapshot.json       ← macro-analyst 출력
    │   ├── signal_summary.json        ← signal-interpreter 출력
    │   └── chart_paths.json           ← quant-coder 출력
    ├── charts/                        생성된 차트 이미지
    └── reports/                       최종 리포트 (YYYYMMDD 형식)
```

---

## 4. 각 레이어 설계 원칙

### 4.1 Skills — 도메인 지식 자동 주입

```yaml
# SKILL.md 프론트매터
---
name: skill-name
description: |
  이 텍스트가 활성화 트리거.
  작업 컨텍스트와 매칭되면 자동 로드.
version: 1.0.0
---
```

**설계 원칙:**
- `description`은 "언제 쓰는가"를 구체적으로 (트리거 역할)
- 내용은 1-2페이지 분량 (threshold, 판단 기준, 원칙 중심)
- 전체 파일 덤프 금지 → 핵심 수치/원칙만 추출
- 시간 민감 내용은 `🔴 YYYY년 현재 환경 업데이트` 섹션 분리

**Skills에 담을 것 vs 담지 말 것:**
```
담을 것:
  - 판단 임계값 (VIX > 30, HY OAS > 500bp, etc.)
  - 분석 절차 순서
  - 레짐 매트릭스, 의사결정 트리
  - 현재 시장 스냅샷 (정기 업데이트 필요)

담지 말 것:
  - 전체 논문/책 내용
  - 실행 코드 (→ agents로)
  - 특정 데이터 값 (→ MCP로 실시간 조회)
```

### 4.2 MCP Servers — 외부 데이터 연동

```json
// .mcp.json
{
  "mcpServers": {
    "fred-api": {
      "command": "python",
      "args": ["${CLAUDE_PLUGIN_ROOT}/mcp_servers/mcp_fred_server.py"],
      "env": { "FRED_API_KEY": "${FRED_API_KEY}" }
    }
  }
}
```

**도구 호출 패턴:**
```
mcp__{plugin-name}_{server-name}__{tool-name}(args)
예: mcp__finance-analysis-harness_fred-api__fetch_series(series_id="DGS10")
```

**FRED 주요 시계열 ID:**
| 지표 | ID | 설명 |
|------|-----|------|
| 미국 10년물 | DGS10 | 10-Year Treasury |
| 미국 2년물 | DGS2 | 2-Year Treasury |
| Fed Funds Rate | FEDFUNDS | 기준금리 |
| Core PCE | PCEPILFE | Fed 공식 목표 |
| 원달러 환율 | DEXKOUS | KRW/USD |
| M2 | M2SL | 광의통화 |

### 4.3 Agents — 단일 책임 에이전트

```markdown
---
name: agent-name
description: |
  활성화 조건 설명.

  <example>
  Context: 상황 설명
  user: "..."
  assistant: "..."
  </example>

model: inherit
color: blue
tools: ["Read", "Bash", "Write"]
---

에이전트 지시문...
```

**에이전트 파이프라인 (금융 분석):**
```
data-validator → macro-analyst → signal-interpreter → quant-coder → report-writer
       ↓               ↓               ↓                  ↓              ↓
validation_result  regime_snapshot  signal_summary    chart_paths    report_YYYYMMDD
```

**핸드오프 규칙:**
- 각 에이전트는 `outputs/context/*.json`으로 결과 전달
- 다음 에이전트는 시작 전 이전 JSON을 읽음
- 실패 시 다음 단계 진행 금지 + 사용자에게 보고

### 4.4 Hooks — 이벤트 기반 가드레일

```json
{
  "hooks": [
    {
      "event": "PreToolUse",
      "tool": "Bash",
      "type": "command",
      "command": "validate-bash.sh"
    },
    {
      "event": "Stop",
      "type": "prompt",
      "prompt": "완료 체크리스트 확인..."
    }
  ]
}
```

**훅 유형 선택:**
```
command 훅: 결정적 규칙 (rm -rf 차단, 경로 확인)
prompt 훅:  맥락 판단 필요 (완료 여부, 오류 해석)
```

**금융 분석 필수 훅:**
- `SessionStart` → API 키 확인, outputs/ 생성
- `PreToolUse/Bash` → 위험 명령어 차단 (rm -rf, DROP TABLE)
- `Stop` → 분석 완료 여부 체크 (5단계 파이프라인 완료?)
- `PreCompact` → 컨텍스트 압축 전 마일스톤 보존

### 4.5 CLAUDE.md — 운영 매뉴얼

프로젝트 루트의 `CLAUDE.md`는 **매 세션 자동 로드**되는 운영 매뉴얼.

**담을 내용:**
```
1. 프로젝트 목적 (1-2줄)
2. 핵심 원칙 (항상 지켜야 할 규칙)
3. 에이전트 파이프라인 다이어그램
4. MCP 도구 사용법 예시
5. 자주 쓰는 ID/경로/설정값
6. 즉시 경보 조건 (임계값 기반)
7. 파일 경로 규칙
8. 언어 규칙 (한국어/영어 혼용 기준)
```

---

## 5. 지식 파일 변환 가이드

기존 파일 → Skills로 변환하는 방법:

| 파일 형식 | 변환 전략 |
|---------|---------|
| `.md` 프레임워크 | 프론트매터 추가, 코드 블록 → 판단 기준으로 요약 |
| `.py` 분석 코드 | 설계 결정·임계값·사용 패턴 추출 → SKILL.md, 원본은 `references/`에 보관 |
| `.pdf` 논문/리포트 | 핵심 수식·임계값·결론만 추출 (5-10개 항목) |
| `.txt` 노트 | 카테고리 분류 후 해당 SKILL.md에 병합 |

**원칙: SKILL.md = 1-2페이지 요약. 원본 파일은 삭제하지 않고 `references/` 보관.**

---

## 6. 2026 환경 업데이트 (Skills 반영 완료)

### 거시경제 (macro-economics v1.2.0)
- Fed 금리 동결, Kevin Warsh 의장 취임 예정 (2026.5)
- 미국 실효관세율 100년래 최고 수준
- GDP 2026E: +2.2%, 경기침체 확률 30-35%

### 금융 시그널 (financial-signals v1.1.0)
- HY OAS ≈ 300bp = 5년 범위 5퍼센타일 (역사적 최저권)
- VIX = 12퍼센타일 (극도의 낙관/컴플레이선시)
- 10Y-2Y = +65bp (역전 해소, 정상 기울기 복귀)
- **종합 판단: near-perfect 시나리오 pricing → 헤지 비용이 가장 저렴한 구간**

### 한국 금융 (korean-finance v1.1.0)
- BOK 기준금리 2.50% (5연속 동결)
- KRW/USD ≈ 1,470 (16년래 최저권)
- 2026 GDP 1.8% (BOK 전망)

### 자산 클래스 (asset-class-universe v1.1.0)
- GENIUS Act: 구현 규정 2026.7, 발효 2027.1
- 온체인 RWA 공급 $36B+ 돌파
- EM > DM (달러 약세 + 무역 다변화)

### AI×금융 워크플로우 (ai-finance-workflow v1.1.0)
- 에이전틱 AI 82% 중견기업 도입
- LLM이 리서치 퍼널 병목을 "아이디어 발굴 → 평가 속도"로 전환
- 대안 데이터 폭발 (텍스트/특허/위성/신용카드/구인공고)

---

## 7. 즉시 적용 체크리스트

새 프로젝트에 하니스 구조 적용 시:

```
[ ] CLAUDE.md 작성 (운영 매뉴얼, 핵심 원칙)
[ ] Skills 정의 (도메인 지식 → SKILL.md 변환)
[ ] Agents 설계 (단일 책임 원칙, 파이프라인 순서)
[ ] MCP 서버 구현 (config + 실제 Python 코드)
[ ] Hooks 설정 (SessionStart, PreToolUse, Stop 최소한)
[ ] Commands 작성 (사용자 진입점, 파이프라인 오케스트레이션)
[ ] outputs/ 디렉토리 구조 정의
[ ] 에이전트 핸드오프 JSON 스키마 문서화
```

---

## 8. 참고 자료

- 전체 설계 가이드: `/home/tj/projects/harness_engineering/DESIGN_GUIDE.md`
- 지식 변환 가이드: `/home/tj/projects/harness_engineering/KNOWLEDGE_CONVERSION.md`
- 예시 플러그인: `/home/tj/projects/harness_engineering/example-plugin/`
- 원본 프레임워크: `/home/tj/finance/frameworks/`
