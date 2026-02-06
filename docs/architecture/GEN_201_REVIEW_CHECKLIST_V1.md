# GEN-201 Review Checklist v1 (2026-02-06)

## 1. Scope
- 대상 작업: `work_orders/GEN-201.md`
- 목표: analyzers shard/facade 분리가 계약을 깨지 않았는지 리뷰

## 2. Review Items

### 2.1 API Compatibility
- [ ] `pipeline/__init__.py`의 export 이름이 기존과 동일
- [ ] `from pipeline import ...` 경로가 정상
- [ ] `from pipeline.analyzers import ...` 경로가 정상

### 2.2 Signature Stability
- [ ] 함수 시그니처 변경 없음
- [ ] async/sync 타입 유지 (`track_events_with_news` async)
- [ ] return payload key 변경 없음

### 2.3 Ownership Correctness
- [ ] 함수가 `ADV_002` ownership matrix대로 shard 배치됨
- [ ] `pipeline/analyzers.py`는 facade 역할만 수행
- [ ] shard 간 순환 import 없음

### 2.4 Code Health
- [ ] dead code/중복 코드 제거 여부 확인
- [ ] `pipeline/analyzers_extension.py` 미수정 확인 (GEN-201 범위)
- [ ] logger/error handling semantics 유지

### 2.5 Validation Evidence
- [ ] py_compile 로그 존재
- [ ] import smoke 로그 존재
- [ ] allocation/rebalancing smoke 로그 존재

## 3. Reject Conditions
- 공개 API break (`ImportError`, 이름 누락)
- payload 계약 위반
- facade 아닌 구현 중복이 남아 drift 증가

## 4. Approve Conditions
- 필수 체크박스 모두 충족
- reject condition 0건
