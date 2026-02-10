# EIMAS TODO (Full Mode Refactor) - 2026-02-06

## Goal
- 기준 실행 경로를 `python main.py --full`로 고정
- 과도하게 결합된 기능을 도메인별로 분리
- `eimas`는 "full orchestration core"로 축소
- 실거래 지향 베이스라인(`us-trader-v1`)을 별도 profile로 운영

## Refactor Rules
- 단일 진입점: `main.py` (`run_integrated_pipeline`)
- 호환성 유지: 기존 import 경로는 shim으로 유지 후 단계적 제거
- 분리 우선순위: "독립 실행 가능 + 외부 API 의존 + 파일 크기 큰 영역" 먼저
- 검증 정책: 작은 변경은 `py_compile + import/function smoke`, full은 milestone/merge에서만 실행
- 구조 기준 문서: `STRUCTURE_REDESIGN_MASTERPLAN.md`
- 비대화 해소 기준 문서: `BLOAT_RESOLUTION_ARCHITECTURE.md`

---

## Track A - Full Mode 안정화 (이번 세션)

### A1. 실행 경로 정리
- [x] `main_integrated.py` 제거 (단일 진입점 `main.py`로 통합)
- [x] `pipeline/runner.py` 제거 (단일 진입점 `main.py`로 통합 완료)
- [x] `run_all_pipeline.sh` 제거 (`python main.py --full` 직접 실행으로 단순화)
- [x] `api/main.py`, `cli/eimas.py`의 legacy 주석/의존 제거
- [x] 단일 run JSON artifact 경로 정책 도입 (`ADV_007`, phase7/8 동일 파일 갱신)
- [x] `api/main.py`를 canonical API 엔트리로 고정, `api/server.py` 제거
- [x] `main.py` 런타임 보조 로직 분리 (`pipeline/app/runtime.py`, `PhaseRuntimeTracker`)
- [x] `main.py` phase 실행 블록 분리 (`pipeline/app/orchestrator_steps.py`)
- [x] 미사용 스크립트 2종 제거 (`scripts/check_gold_data.py`, `scripts/visualize_agents.py`)
- [x] instruction-only 스크립트 제거 (`scripts/setup_scheduler.sh`, cron 예시는 manual 문서로 이전)

---

## Track F - US Trader Baseline (14일 컷오버)

기준 문서: `docs/US_TRADER_BASELINE_20260210.md`

- [x] `--profile us-trader-v1` 추가 (phase-level 정책 적용)
- [x] profile 기반 phase skip audit trail 추가 (`audit_metadata.profile_skips`)
- [x] phase2 sentiment/bubble 분리 스위치 추가 (`skip_bubble`)
- [x] 브로커 우선순위 확정 (`IBKR` 우선)
- [x] IBKR 기준 실행 라우터 + idempotency 키 도입 (`lib/broker_execution.py`)
- [x] 실행 DB에 order_state/idempotency/explainability 컬럼 추가 (`lib/trading_db.py`)
- [x] 멀티자산 주문정책(v1.1) 적용 (자산군별 notional cap/수량 정밀도/비거래 자산 차단)
- [ ] 운영 리스크 파라미터(회전율/집중도/리스크 임계값) 실측 튜닝
- [ ] 연속 실행 운영 시나리오(cron/daemon) + 장애 복구 검증

### A2. 깨진/중복 경로 청소
- [x] `main_integrated` 직접 참조 제거 (활성 코드 경로 기준)
- [x] 구버전 archive/docs 및 docs/archive 제거
- [x] 활성 경로 backup 파일(`*_backup_*`, `.backup_*`) 제거
- [x] `lib/deprecated/` 제거 (레거시 import 의존 정리)
- [x] `archive/` 전체 제거 (구형 코드/문서 정리)
- [x] `pipeline.collection.runner` 등 archive 잔재 참조 제거
- [x] 사용하지 않는 실행 스크립트 목록화 및 1차 정리 (`RUN_SCRIPT_INVENTORY_20260207`, `merge_frontend.sh` 제거)

---

## Track B - 기능 분할 (온체인 방식으로 외부 폴더 분리)

### B0. 완료/진행 상태
- [x] `onchain_intelligence` 1차 분리 완료
- [x] `eimas`와 `onchain_intelligence` 인터페이스 계약(JSON schema) 명시 (`docs/architecture/ONCHAIN_INTELLIGENCE_INTERFACE_CONTRACT_V1.md`, `docs/references/onchain_intelligence_bridge_payload_v1.schema.json`)

### B1. 분리 후보 1: Execution Intelligence (우선)
대상: 운영결정/리밸런싱/제약복구/전술배분/스트레스테스트
- [x] 새 폴더 생성: `/home/tj/projects/autoai/execution_intelligence`
- [x] 1차 이동 완료:
  - `lib/allocation_engine.py` -> `execution_intelligence/models/allocation_engine.py`
  - `lib/rebalancing_policy.py` -> `execution_intelligence/models/rebalancing_policy.py`
  - `pipeline/analyzers.py`가 adapter 경유 import로 전환
- [x] 2차 이동 완료 (adapter 경유 전환):
  - `lib/tactical_allocation.py` -> `execution_intelligence.models.tactical_allocation`
  - `lib/stress_test.py` -> `execution_intelligence.models.stress_test`
  - `phase6_portfolio.py`, `tests/test_portfolio_modules.py` import를 `lib.adapters`로 전환
- [ ] 운영결정 이동 정리:
  - `lib/operational_engine.py`
  - `lib/operational/` (EXIS에 copy 완료, eimas 원본은 아직 유지)
  - 진행: `lib/operational/*`을 EXIS 기준으로 동기화, `execution_backend`는 package-first + monolith fallback으로 전환
  - 진행: `phase45_operational.py`가 `audit_metadata`에 `execution_backend_source`/`execution_backend_fallback_reason` 기록
  - 다음: fallback 관측(`backend_source`, `backend_fallback_reason`) 기준으로 monolith 제거 시점 결정
- [x] EIMAS adapter 작성: 실패 시 HOLD fallback 보장

### B2. 분리 후보 2: Reporting Intelligence
대상: AI 리포트/화이트닝/팩트체크/문서 변환
- [ ] 새 폴더 생성: `/home/tj/projects/autoai/reporting_intelligence`
- [ ] 이동 대상 확정:
  - `pipeline/report.py`
  - `lib/ai_report_generator.py`
  - `lib/whitening_engine.py`
  - `lib/autonomous_agent.py`
  - `lib/json_to_md_converter.py`
  - `lib/json_to_html_converter.py`

### B3. 분리 후보 3: Realtime Intelligence
대상: 바이낸스 스트림/실시간 파이프라인/알림
- [ ] 새 폴더 생성: `/home/tj/projects/autoai/realtime_intelligence`
- [ ] 이동 대상 확정:
  - `pipeline/realtime.py`
  - `lib/binance_stream.py`
  - `lib/realtime_pipeline.py`

---

## Track C - Full Mode 성능/신뢰성

### C1. 성능 예산
- [ ] FULL 총 실행 시간: `249s -> 150s -> 120s`
- [x] Phase 2 분석 병목 1차 캐시 도입 (`1h TTL`, 무인자 고비용 분석 결과 파일 캐시)
- [x] Phase 2 캐시 hit/miss 텔레메트리 (`result.phase2_cache_stats`) 추가
- [x] AI 검증 병렬화 1차: `ValidationAgentManager.validate_all` thread fan-out 적용 (`EIMAS_VALIDATION_AGENT_TIMEOUT_SEC`)
- [x] AI 검증 병렬화 2차: timeout/재시도/백오프 기본 정책 적용 (`EIMAS_VALIDATION_RETRY_COUNT`, `EIMAS_VALIDATION_RETRY_BACKOFF_SEC`)
- [x] AI 검증 병렬화 3차: agent별 정책 차등화 + 실패 유형별 selective retry (`EIMAS_VALIDATION_RETRY_POLICY_OVERRIDES`)
- [x] AI 검증 관측성 보강: `validation_runtime_stats`(agent별 attempts/retries/failure_type) 추가
- [x] 파이프라인 phase 타이밍 텔레메트리 추가 (`result.pipeline_phase_timings`, `result.pipeline_elapsed_sec`)
- [x] Phase 1 컴포넌트 타이밍 텔레메트리 추가 (`audit_metadata.phase1_component_timings`, `audit_metadata.phase1_elapsed_sec`)
- [x] Phase 1 market/crypto 중복 다운로드 제거 (`collect_market_data(..., include_crypto=False)` 적용)
- [x] Extended data 네트워크 fail-fast 스킵 옵션 추가 (`EIMAS_EXTENDED_FAIL_FAST_NETWORK`, `EIMAS_SKIP_EXTENDED_DATA`)
- [x] Institutional frameworks 네트워크 fail-fast/스킵 옵션 추가 (`EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK`, `EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS`)
- [x] Institutional frameworks 컴포넌트 타이밍 텔레메트리 추가 (`audit_metadata.phase2_institutional_components`)
- [x] Adaptive portfolio DB I/O 배치 최적화 (`AdaptiveAgentManager.run_all` 단일 트랜잭션)
- [x] Phase3 debate 런타임 제어 추가 (`--debate-full-lookback`, `--debate-ref-lookback`, `--debate-skip-reference`, quick-mode cap 180/45 + env override)

### C2. 신뢰성
- [x] `sys.path.insert` 1차 축소 (`13 -> 6`, `scripts/_project_bootstrap.py`로 스크립트 경로부트스트랩 통합)
- [x] `sys.path.insert` 2차 축소 (`6 -> 4`, `lib/path_bootstrap.py`로 동적 외부경로 주입 통합)
- [x] `api/main.py`, `cli/eimas.py` module-first + direct-script fallback 가드 적용
- [x] `sys.path.insert` 제거 계획 문서화 (`docs/session_artifacts/SYSPATH_INSERT_REDUCTION_PLAN_20260207.md`)
- [ ] 잔여 `sys.path.insert` 4건 정리 (`api/main.py`, `cli/eimas.py`, `scripts/_project_bootstrap.py`, `lib/path_bootstrap.py`)
- [x] 절대경로 제거 (`/home/tj/projects/autoai/eimas` 하드코딩, 실행 코드 기준)
- [x] `Close`/`close` 컬럼 편차에 대한 backtest/전략배분 로직 내성 강화
- [x] `calculate_strategic_allocation`의 `us_fair_gap`/`korea_fair_gap` 미초기화 예외 수정
- [ ] `pytest` 실행 가능한 테스트 환경 정비

---

## Track D - 문서/운영 프로세스 재설계

- [x] `FULL_EXECUTION_PROCESS.md` 신설
- [x] `README.md`에 full-mode refactor 문서 링크 추가 (`FULL_EXECUTION_PROCESS`, `CURRENT_STATUS`, `TODO`)
- [x] `CURRENT_STATUS.md`를 refactor 진행 기준으로 업데이트
- [x] 구버전 archive 제거 + 문서 참조 정리

---

## 이번 주 실행 순서

1. `A` 완료: 실행 경로/깨진 import 정리 완료
2. `B1` 시작: Execution Intelligence 폴더 생성 + 모듈 이동 리스트 확정
3. `C` 착수: per-change/per-wave 검증 자동화
4. `B2`/`B3`로 확장: 보고서/실시간 분리

---

## 내일 바로 시작 (Restart)

1. 상태 확인:
   - `git status --short`
   - `python3 -m py_compile main.py api/main.py lib/ai_report_generator.py`
2. 계약 확인:
   - `bash scripts/check_execution_contract.sh`
3. 다음 클리닝:
   - `docs/architecture/*`의 legacy 명령/엔트리 참조 정리
   - `sys.path.insert`/절대경로 하드코딩 2차 축소
4. 구조 리팩토링 재개:
   - `execution_intelligence` 운영결정 이동 정리 (`operational_engine`, `operational/`)

---

## Canonical Commands

```bash
# Full mode (기준 실행)
python main.py --full

# Full + realtime
python main.py --full --realtime -d 30

```

---

## Track E - RA SQL Productionization (현업형 구현/어필)

### E1. RA 데이터 모델 표준화 (PostgreSQL)
- [ ] `fi_ra` 스키마 표준 확정 (`company_fundamentals`, `macro_series`, `etf_snapshot`, `research_views`, `trade_recommendations`)
- [ ] 공통 메타 컬럼 추가: `as_of_date`, `source`, `ingested_at`, `revision_tag`, `quality_flag`
- [ ] PK/UK/FK 정책 확정 (ticker+as_of_date, series_id+as_of_date 등)
- [ ] 분석/리포트 조회 인덱스 추가 (`ticker`, `as_of_date`, `regime_label`, `risk_score`)

### E2. ETL/적재 파이프라인 고도화
- [ ] 수집 -> 정제 -> 적재 분리 (`staging` -> `mart`) 구조로 리팩토링
- [ ] upsert 정책 통일 (`ON CONFLICT DO UPDATE`) + 충돌 로그 기록
- [ ] 배치 실행 로그 테이블 추가 (`job_name`, `started_at`, `ended_at`, `row_count`, `status`, `error`)
- [ ] 재부팅/장애 복구 런북과 실제 커맨드 검증 체크리스트 정비

### E3. SQL 검증/품질관리 (Data QA)
- [ ] 일일 DQ 쿼리 세트 작성: 결측, 중복, 이상치, stale-data, 날짜 역행 검사
- [ ] 핵심 지표 스냅샷 검증 (`COUNT`, `DISTINCT`, `MIN/MAX date`) 자동 리포트화
- [ ] 품질 경고를 `quality_flag`로 저장하고 RA 리포트 본문에 경고 배지 표시
- [ ] 소스별 수치 비교(예: 가격/재무치) 허용오차 정책 문서화

### E4. RA 분석/리포트 SQL 레이어
- [ ] RA 조회용 SQL 뷰 생성: `v_ra_macro_regime`, `v_ra_etf_signal`, `v_ra_company_valuation`
- [ ] 리포트 생성 스크립트가 JSON+SQL 뷰를 함께 참조하도록 경로 통합
- [ ] RA 스타일 PDF 표준 목차 고정 (요약/거시/ETF/기업/리스크/결론/부록)
- [ ] 도표 캡션에 데이터 기준일(`as_of_date`)과 소스(`source`) 자동 표기

### E5. 추천 -> 모의주문 -> 백테스트 -> 사후평가
- [ ] `trade_recommendations` 스키마 확장 (thesis, invalidation, target_horizon, confidence)
- [ ] 모의주문 체결모형 보강 (슬리피지/스프레드/거래세션 제약)
- [ ] 백테스트 결과를 추천안 ID 단위로 연결 저장 (hit ratio, MDD, tracking error)
- [ ] 추천 성과 리뷰 SQL 템플릿 작성 (주간/월간 top-bottom attribution)

### E6. 현업/인사 어필용 산출물 패키지
- [ ] "SQL로 구현한 리서치 운영 흐름" 1-page 요약 PDF 자동 생성
- [ ] 자기소개서 첨부용 스크린샷 세트 생성 (ERD, 검증 쿼리, RA PDF 일부)
- [ ] README/intro에 "현재 구현" vs "입사 후 확장" 매핑표 반영
- [ ] 발표용 5분 데모 시나리오 문서화 (입력 -> SQL 적재 -> 리포트 -> 모의주문)
