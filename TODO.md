# EIMAS TODO (Full Mode Refactor) - 2026-02-06

## Goal
- 기준 실행 경로를 `python main.py --full`로 고정
- 과도하게 결합된 기능을 도메인별로 분리
- `eimas`는 "full orchestration core"로 축소

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
- [x] `pipeline/runner.py`를 canonical 파이프라인 위임 방식으로 교체
- [x] `pipeline/runner.py`에 legacy `run_pipeline` alias 복구
- [x] `run_all_pipeline.sh`를 `python main.py --full` 기반으로 교체
- [x] `api/main.py`, `cli/eimas.py`의 legacy 주석/의존 제거
- [x] 단일 run JSON artifact 경로 정책 도입 (`ADV_007`, phase7/8 동일 파일 갱신)
- [x] `api/main.py`를 canonical API 엔트리로 고정, `api/server.py` 제거

### A2. 깨진/중복 경로 청소
- [x] `main_integrated` 직접 참조 제거 (활성 코드 경로 기준)
- [x] 루트 세션성 MD를 `docs/archive/root_legacy/`로 이관
- [x] 활성 경로 backup 파일(`*_backup_*`, `.backup_*`) 제거
- [x] `lib/deprecated/` 제거 (레거시 import 의존 정리)
- [x] `archive/legacy/` 제거 (구형 실행파일 삭제)
- [ ] `pipeline.collection.runner` 등 archive 잔재 참조 제거
- [ ] 사용하지 않는 실행 스크립트 목록화 및 정리

---

## Track B - 기능 분할 (온체인 방식으로 외부 폴더 분리)

### B0. 완료/진행 상태
- [x] `onchain_intelligence` 1차 분리 완료
- [ ] `eimas`와 `onchain_intelligence` 인터페이스 계약(JSON schema) 명시

### B1. 분리 후보 1: Execution Intelligence (우선)
대상: 운영결정/리밸런싱/제약복구/전술배분/스트레스테스트
- [x] 새 폴더 생성: `/home/tj/projects/autoai/execution_intelligence`
- [x] 1차 이동 완료:
  - `lib/allocation_engine.py` -> `execution_intelligence/models/allocation_engine.py`
  - `lib/rebalancing_policy.py` -> `execution_intelligence/models/rebalancing_policy.py`
  - `pipeline/analyzers.py`가 adapter 경유 import로 전환
- [ ] 2차 이동 대기:
  - `lib/tactical_allocation.py`
  - `lib/stress_test.py`
- [ ] 운영결정 이동 정리:
  - `lib/operational_engine.py`
  - `lib/operational/` (EXIS에 copy 완료, eimas 원본은 아직 유지)
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
- [ ] Phase 2 분석 병목: 캐시 도입 (`1h TTL`)
- [ ] AI 검증 병렬화: validation call fan-out

### C2. 신뢰성
- [ ] `sys.path.insert` 제거 계획 수립 (활성 트리 기준 63건)
- [ ] 절대경로 제거 (`/home/tj/projects/autoai/eimas` 하드코딩)
- [x] `Close`/`close` 컬럼 편차에 대한 backtest/전략배분 로직 내성 강화
- [ ] `pytest` 실행 가능한 테스트 환경 정비

---

## Track D - 문서/운영 프로세스 재설계

- [x] `FULL_EXECUTION_PROCESS.md` 신설
- [ ] `README.md`에 full-mode refactor 문서 링크 추가
- [ ] `CURRENT_STATUS.md`를 refactor 진행 기준으로 업데이트

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
   - `python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py`
2. 계약 확인:
   - `bash scripts/check_execution_contract.sh`
3. 다음 클리닝:
   - `docs/architecture/*`의 legacy 명령/엔트리 참조 정리
   - `archive/docs`, `docs/session_artifacts` 보존 기준 확정 후 pruning
4. 구조 리팩토링 재개:
   - `execution_intelligence` 2차 이동 (`tactical_allocation`, `stress_test`)

---

## Canonical Commands

```bash
# Full mode (기준 실행)
python main.py --full

# Full + realtime
python main.py --full --realtime -d 30

# Shell wrapper
./run_all_pipeline.sh
```
