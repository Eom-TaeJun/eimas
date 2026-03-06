# EIMAS — 운영 규칙

## 필수 작업 원칙

- 모든 작업 전 `command.md` 먼저 확인 (P0 문서, 단일 기준)
- 새 기능은 반드시 `main.py --xxx` 플래그로 귀속
- 변경 후 최소 검증: `python main.py --help && python -m compileall main.py pipeline/app`

## 실행 모드

| 모드 | 명령 | 설명 |
|------|------|------|
| `--short` | `python main.py --short` | 경량 수집 → 운용 → DB 적재 |
| `--full` | `python main.py --full` | Phase 1~9 전체 분석 |
| `--quick1/2` | `python main.py --quick1` | Quick AI 검증 (KOSPI/SPX) |
| `--profile` | `python main.py --profile us-trader-v1` | 프로파일 실행 |
| `--paper-auto` | `python main.py --paper-auto` | 모의주문 자동 실행 |
| `--backtest` | `python main.py --backtest` | 백테스트 (5년) |
| `--realtime` | `python main.py --realtime -d 60` | 실시간 스트리밍 |

**--short vs --full 라우팅:**
```
--short: Phase 1(경량) → 4(실시간) → 4.5(운용) → 4.6(모의주문) → 5(DB)
         outputs/eimas_*.json 최신 full 결과를 자동 로드해 운용에 활용
--full:  Phase 1~9 전체 (AI 토론 + Multi-LLM 검증 포함)
```

## 파이프라인 구조

```
Phase 1   : 데이터 수집 (FRED, Market, Crypto, Korea)
Phase 2   : 정량 분석 (레짐, 리스크, 포트폴리오, ARK, ETF Flow)
Phase 3   : AI 에이전트 토론 (Full 365d + Reference 90d)
Phase 4   : 실시간 스트리밍 (--realtime 전용)
Phase 4.5 : 운용 의사결정
Phase 4.6 : 모의주문 (--paper-auto 전용)
Phase 5   : 저장 (outputs/eimas_*.json)
Phase 7   : AI 리포트 생성
Phase 8   : Multi-LLM 검증 (--full 전용)
Phase 9   : 아티팩트 Export
```

**핵심 파일:**

| 역할 | 파일 |
|------|------|
| Full 파이프라인 | `pipeline/app/orchestrator_steps.py` → `run_pipeline_phases()` |
| Short 파이프라인 | `pipeline/app/orchestrator_steps.py` → `run_short_pipeline_phases()` |
| 데이터 스키마 | `pipeline/schemas.py` |
| 분석 모듈 | `lib/` (패키지 + shim 구조) |
| API 서버 | `api/main.py` (FastAPI, localhost:8000) |

> `lib/` 내 monolith 파일들은 동명 패키지의 shim으로 전환됨.
> 기존 import 경로 (`from lib.bubble_detector import BubbleDetector`) 그대로 유지.

## 핵심 경제학적 방법론

| 방법론 | 사용처 |
|--------|--------|
| GMM 3-State | 레짐 탐지 (Bull/Neutral/Bear) |
| Bekaert VIX 분해 | VIX = Uncertainty + Risk Appetite |
| Greenwood-Shleifer | 버블 탐지 (2년 100% run-up 기준) |
| Amihud Lambda + VPIN | 미세구조 비유동성 / 독성 주문 |
| MST (Mantegna) + HRP | 포트폴리오 최적화 |
| Granger Causality | 유동성 인과관계 |
| LASSO (L1) | 변수 선택 — Treasury 제외 (Simultaneity 방지) |
| Net Liquidity | Fed BS - RRP - TGA |

**Horizon 분리**: 초단기(≤30일) / 단기(31-90일) / 장기(≥180일) — LASSO 모델에서 분리 필수

## 참고 문서 우선순위

| 우선순위 | 문서 | 용도 |
|----------|------|------|
| **P0** | `command.md` | 실행 명령 단일 기준 |
| **P1** | `REFACTOR_PLAN.md` | 현재 진행 중인 작업 + TODO |
| **P2** | `README.md` | 프로젝트 소개 |

## Claude Code 통합 (.claude/)

### 에이전트

| 에이전트 | 색상 | 역할 |
|----------|------|------|
| `eimas-runner` | blue | 파이프라인 실행·결과 검증 전담 |
| `eimas-diagnostics` | red | 에러 진단·최소 수정 전담 |
| `eimas-analyst` | purple | 결과 해석·투자 코멘트 전담 |

### 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/check` | 최신 실행 결과 요약 |
| `/compare [file1] [file2]` | 두 실행 결과 비교 |
| `/phase [N]` | Phase N 설명 (생략 시 전체) |

### 스킬 구조

```
.claude/skills/
  pipeline/  eimas-run-guide, eimas-phase-guide
  output/    eimas-output-check, eimas-output-compare
  debug/     eimas-debug-guide, eimas-error-patterns
  domain/    eimas-schema-guide
```
