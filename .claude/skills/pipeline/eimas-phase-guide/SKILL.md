---
name: eimas-phase-guide
description: EIMAS 파이프라인 각 Phase의 역할, 입출력, 핵심 파일 레퍼런스. $ARGUMENTS로 특정 Phase 번호 전달 가능.
argument-hint: "<phase 번호 예: 1, 2, 3, 4.5>"
user-invocable: false
---

# EIMAS Phase 레퍼런스

`$ARGUMENTS`가 있으면 해당 Phase만 출력. 없으면 전체 요약 출력.

진입점: `pipeline/app/orchestrator_steps.py`
- Full: `run_pipeline_phases()`
- Short: `run_short_pipeline_phases()`

---

## Phase 1 — 데이터 수집

**모드**: Full + Short (Short은 경량 수집)
**파일**: `pipeline/phases/phase1_data_collection.py`
**수집 대상**:
- FRED: 금리, 유동성, 인플레이션, 고용
- Market: SPY/QQQ/TLT/GLD/VIX 등 주요 ETF
- Crypto: BTC-USD 등
- Korea: 저축은행 지표 (SKIP_KOREA_SAVINGS=1로 스킵 가능)
**출력**: `FREDSummary`, `MarketData`, `IndicatorsSummary`

---

## Phase 2 — 정량 분석

**모드**: Full 전용
**파일**: `pipeline/phases/phase2_quantitative.py`
**분석 항목**:
- GMM 3-State 레짐 탐지 (Bull/Neutral/Bear)
- 리스크 점수 계산 (0~100)
- HRP 포트폴리오 최적화
- ARK 거래 분석
- ETF Flow 분석
- 버블/마이크로구조 분석
**출력**: `RegimeResult`, `PortfolioResult`, `BubbleRiskMetrics`

---

## Phase 3 — AI 에이전트 토론

**모드**: Full 전용
**파일**: `pipeline/debate.py`, `agents/interpretation_debate.py`
**동작**:
- Full 모드: 365일 데이터로 장기 관점 토론
- Reference 모드: 90일 데이터로 단기 관점 토론
- 합의 임계값: 85% (`configs/default.yaml`)
**출력**: `DebateResult` (consensus, confidence, rounds)

---

## Phase 4 — 실시간 스트리밍

**모드**: `--realtime` 전용
**파일**: `pipeline/phases/phase4_realtime.py`
**동작**: `-d <seconds>` 주기로 시장 데이터 스트리밍

---

## Phase 4.5 — 운용 의사결정

**모드**: Full + Short
**파일**: `pipeline/phases/phase45_operational.py`
**동작**:
- Short 모드: 최신 full JSON 자동 로드 후 운용 판단
- HOLD 조건 / 제약 충족 여부 점검
- `operational_report` 생성
**주의**: FREDSummary 필드는 `.get()` 불가, 속성 직접 접근

---

## Phase 4.6 — 모의주문

**모드**: `--paper-auto` 전용
**파일**: `pipeline/phases/phase46_paper_trading.py`
**동작**: `trade_plan`의 BUY/SELL → LIMIT 주문 등록
**DB**: `data/paper_trading.db` (계좌/주문/체결 원장)

---

## Phase 5 — 저장

**모드**: Full + Short
**출력**: `outputs/eimas_<timestamp>.json`
**스키마**: `EIMASResult.to_dict()`

---

## Phase 7 — AI 리포트 생성

**모드**: Full 전용
**파일**: `pipeline/phases/phase7_report.py`
**출력**: `outputs/reports/*.html`, `*.md`

---

## Phase 8 — Multi-LLM 검증

**모드**: Full 전용
**파일**: `pipeline/phases/phase8_verification.py`
**동작**: Claude + Perplexity 교차 검증

---

## Phase 9 — 아티팩트 Export

**모드**: Full 전용
**출력**: 최종 아티팩트 패키징 (`examples/` 복사 등)
