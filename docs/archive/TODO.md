# EIMAS TODO (2026-02-19)

## 🔴 Track B — 도메인 분리

### B1. Execution Intelligence ✅
- [x] `allocation_engine`, `rebalancing_policy`, `tactical_allocation`, `stress_test` → `execution_intelligence/`
- [x] `lib/adapters/` 경유 import 전환

### B2. Reporting Intelligence ✅
- [x] `lib/reports/` 패키지: `AIReportGenerator`, `FinalReportAgent`, `AllocationReportAgent`, `WhiteningEngine`, `convert_json_to_html/md` 통합
- [x] `lib/realtime_intelligence/` 패키지: `BinanceStreamer`, `RealtimePipeline` 통합

### B3. 잔여
- [ ] `reporting_intelligence/` 별도 레포 분리 (장기)
- [ ] `realtime_intelligence/` 별도 레포 분리 (장기)

---

## 🔴 Track C — 성능/신뢰성

- [ ] FULL 실행 시간: 249s → 120s 목표
- [ ] `pytest` 실행 환경 정비
- [ ] 잔여 `sys.path.insert` 4건 정리

---

## 🟡 Track E — RA SQL Productionization

### E1. 데이터 모델
- [ ] `fi_ra` 스키마 표준 확정 (공통 메타: `as_of_date`, `source`, `quality_flag`)
- [ ] PK/UK/FK 정책 확정

### E2. ETL
- [ ] staging → mart 구조 리팩토링
- [ ] upsert 정책 통일 (`ON CONFLICT DO UPDATE`)

### E3. SQL 레이어
- [ ] 뷰 생성: `v_ra_macro_regime`, `v_ra_etf_signal`, `v_ra_company_valuation`
- [ ] 일일 DQ 쿼리 세트 (결측/중복/이상치)

### E4. 산출물
- [ ] 추천 → 모의주문 → 백테스트 → 사후평가 연결 스키마
- [ ] \"SQL로 구현한 리서치 운영 흐름\" 1-page 요약

---

## 🟡 Track F — 운영 리스크

- [ ] 운영 리스크 파라미터 실측 튜닝
- [ ] cron/daemon 연속 실행 + 장애 복구 검증

---

## ✅ 완료

- **Track A**: 단일 진입점 `main.py` 통합
- **Track B (리팩토링)**: monolith 9개 → 패키지 shim, `lib/reports/`, `lib/realtime_intelligence/` 신설
- **Track B (--short)**: `run_short_pipeline_phases()` 신설, full 결과 자동 로드
- **Track F (1차)**: `--profile us-trader-v1`, `--paper-auto`, IBKR 라우터, Backtest DB v2.1
