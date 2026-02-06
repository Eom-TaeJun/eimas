# ADV-004: analyzers_extension Policy v1 (2026-02-06)

## 1. Decision Summary
- `pipeline/analyzers_extension.py`는 **canonical source가 아니다**.
- canonical은 `pipeline/analyzers.py`(및 이후 shard 모듈)로 고정한다.
- `analyzers_extension.py`는 단계적으로 **decommission**한다.

## 2. Evidence
- 런타임 코드에서 direct import 없음:
  - `rg "analyzers_extension"` 결과 기준, 실사용 참조 없음(문서/work_order 제외)
- drift 존재:
  - `pipeline/analyzers_extension.py`는 `AdaptivePortfolioManager`를 import하려 하지만
    현재 `lib.adaptive_agents`와 불일치
  - 실제 import 시 `ImportError` 발생 확인

## 3. Policy

### AD-EXT-01: Canonical Lock
- 이벤트/볼륨/adaptive 함수의 canonical 구현은 `pipeline/analyzers.py`에만 둔다.

### AD-EXT-02: Compatibility Shim
- `pipeline/analyzers_extension.py`는 legacy shim으로 교체:
  - 내부 구현 제거
  - canonical 함수 re-export만 수행
  - module import 시 `DeprecationWarning` 출력

### AD-EXT-03: Archive Relocation
- 현재 extension 본문은 필요 시 `archive/legacy/`로 이동 보관
- active runtime tree에는 중복 구현을 두지 않는다.

## 4. Migration Plan

### Step E1 (Safe)
1. `pipeline/analyzers_extension.py`를 shim으로 교체
2. 다음만 re-export:
   - `analyze_volume_anomalies`
   - `track_events_with_news`
   - `run_adaptive_portfolio`

### Step E2 (Cleanup)
1. legacy 본문 archive 이동
2. docs 및 import graph 재확인

## 5. Validation
- `python3 -m py_compile pipeline/analyzers_extension.py pipeline/analyzers.py`
- `python3 -c "from pipeline.analyzers_extension import run_adaptive_portfolio; print('ok')"`
- `python3 -c "from pipeline.analyzers import run_adaptive_portfolio; print('ok')"`

## 6. Risk & Mitigation
- Risk: hidden consumer가 extension 구 구현에 의존
- Mitigation:
  - shim re-export로 API 표면 유지
  - canonical 함수로 일원화되어 동작 편차 제거

## 7. Work Order Mapping
- `GEN-202`: extension shim화 + legacy body 정리
