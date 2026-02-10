# US Trader Baseline Progress - 2026-02-10

## 목적
- 실거래 도움 중심으로 EIMAS를 `us-trader-v1` 구조에 맞춰 재정렬
- 멀티자산 주문 범위를 반영하면서 설명가능성 우선 정책 고정

## 이번 반영 범위
- 파이프라인 프로파일 도입: `legacy`, `us-trader-v1`
- 프로파일 기반 phase skip + `audit_metadata.profile_skips` 기록
- IBKR-first 실행 라우터 + idempotency 주문 중복 방지
- 실행 DB 확장: `broker`, `idempotency_key`, `order_state`, `explainability`
- 멀티자산 주문정책(v1.1):
  - 자산군 분류 (`us_equity`, `us_etf`, `us_bond_etf`, `us_commodity_etf`, `korea_equity`, `crypto_spot`)
  - 자산군별 최소 주문금액/최대 주문비중/수량 정밀도
  - `index`, `futures` 비거래 처리
  - 주문 explainability에 정책 적용값 기록

## 운영 설정
- 브로커: `EIMAS_EXECUTION_BROKER=ibkr`
- 전역 주문 캡: `EIMAS_EXECUTION_MAX_ORDER_NOTIONAL_PCT=0.20`
- 자산군 비활성화: `EIMAS_EXECUTION_DISABLED_ASSET_CLASSES=index,futures`

## 검증
- `python3 -m py_compile`로 변경 파일 컴파일 확인
- idempotency 중복 등록 방지 smoke test 통과

## 남은 작업
- 운영 리스크 파라미터 실측 튜닝(회전율/집중도/리스크 임계값)
- 연속 실행(cron/daemon) 장애 복구 검증
