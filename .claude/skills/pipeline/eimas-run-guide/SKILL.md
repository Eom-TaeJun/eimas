---
name: eimas-run-guide
description: EIMAS 파이프라인 실행 모드, 주요 명령어, 파일 경로 레퍼런스
user-invocable: false
---

# EIMAS 실행 가이드

## 실행 모드

| 모드 | 명령어 | 설명 |
|------|--------|------|
| `--short` | `python main.py --short` | 경량 수집 → 운용 → DB 적재 (빠름) |
| `--full` | `python main.py --full` | Phase 1~9 전체 (AI 토론 + Multi-LLM, 느림) |
| `--quick1` | `python main.py --quick1` | Quick AI 검증 (KOSPI) |
| `--quick2` | `python main.py --quick2` | Quick AI 검증 (SPX) |
| `--profile` | `python main.py --profile us-trader-v1` | 프로파일 기반 실행 |
| `--paper-auto` | `python main.py --paper-auto` | 모의주문 자동 실행 |
| `--backtest` | `python main.py --backtest` | 5년 백테스트 |
| `--realtime` | `python main.py --realtime -d 60` | 실시간 스트리밍 (60초 주기) |

## 파이프라인 단계

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

## 핵심 파일 위치

| 역할 | 경로 |
|------|------|
| Full 파이프라인 진입점 | `pipeline/app/orchestrator_steps.py::run_pipeline_phases()` |
| Short 파이프라인 진입점 | `pipeline/app/orchestrator_steps.py::run_short_pipeline_phases()` |
| 데이터 스키마 | `pipeline/schemas.py` — EIMASResult, FREDSummary, RegimeResult |
| 에이전트 코어 | `agents/orchestrator.py` — MetaOrchestrator |
| 분석 모듈 | `lib/` (shim 구조, 기존 import 유지) |
| DB | `core/database.py` — trading.db, events.db, unified_store.db |
| 출력 | `outputs/eimas_*.json` (JSON), `outputs/reports/*.html` (HTML) |

## 빠른 유효성 검사

```bash
# import 확인
python -c "from pipeline.schemas import EIMASResult; print('OK')"

# 최신 결과 확인
jq '.final_recommendation, .risk_score' outputs/eimas_*.json | tail -2

# 도움말
python main.py --help
```
