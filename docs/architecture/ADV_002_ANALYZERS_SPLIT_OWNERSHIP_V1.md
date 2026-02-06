# ADV-002: Analyzers Split Ownership v1 (2026-02-06)

## 1. Objective
- `pipeline/analyzers.py`(현재 1,473 lines)의 책임을 분할한다.
- 외부 호출 경로(`pipeline.__init__`, `main.py`)는 깨지지 않게 유지한다.
- General Lane이 안전하게 실행할 수 있는 작업 단위로 분해한다.

## 2. Current Risk Snapshot
- 함수 수: `26` (`def 25 + async def 1`)
- 대형 함수:
  - `run_allocation_engine` (113 lines)
  - `analyze_hft_microstructure` (88 lines)
  - `run_ai_validation` (87 lines)
  - `analyze_dtw_similarity` (80 lines)
- 결합도:
  - `main.py`가 analyzer 함수를 직접 다수 import/호출
  - `pipeline/__init__.py`가 analyzer API를 re-export
- 호환성 리스크:
  - `pipeline/analyzers_extension.py`에 중복 구현 존재 (drift 위험)

## 3. Design Decision (Advanced)

### AD-ANL-01: Facade-First Split
- `pipeline/analyzers.py`는 **공개 API facade**로 유지
- 실제 구현은 샤드 모듈로 이동
- facade는 import/re-export + 얇은 wrapper만 유지

Reason:
- `from pipeline import ...` 및 `from pipeline.analyzers import ...` 호환 유지
- 대규모 import 경로 변경 없이 작은 단위 분리 가능

### AD-ANL-02: No Package Flip in v1
- `pipeline/analyzers.py`를 당장 `pipeline/analyzers/` 패키지로 바꾸지 않는다.
- 파일/패키지 동시 전환은 리스크가 높아 v2로 연기.

## 4. Target Internal Layout (v1)
- `pipeline/analyzers.py` (facade, target < 250 lines)
- `pipeline/analyzers_core.py`
- `pipeline/analyzers_advanced.py`
- `pipeline/analyzers_quant.py`
- `pipeline/analyzers_sentiment.py`
- `pipeline/analyzers_governance.py`

## 5. Ownership Matrix

### 5.1 Core Market Analysis -> `pipeline/analyzers_core.py`
- `detect_regime`
- `detect_events`
- `analyze_liquidity`
- `analyze_critical_path`
- `analyze_etf_flow`
- `generate_explanation`

### 5.2 Advanced Thematic/Graph -> `pipeline/analyzers_advanced.py`
- `analyze_genius_act`
- `analyze_theme_etf`
- `analyze_shock_propagation`
- `optimize_portfolio_mst`
- `analyze_volume_anomalies`
- `run_adaptive_portfolio`
- `track_events_with_news` (async)

### 5.3 Quant & Microstructure -> `pipeline/analyzers_quant.py`
- `analyze_hft_microstructure`
- `analyze_volatility_garch`
- `analyze_information_flow`
- `calculate_proof_of_index`
- `enhance_portfolio_with_systemic_similarity`
- `detect_outliers_with_dbscan`
- `analyze_dtw_similarity`
- `analyze_ark_trades`

### 5.4 Sentiment/Bubble -> `pipeline/analyzers_sentiment.py`
- `analyze_bubble_risk`
- `analyze_sentiment`

### 5.5 Governance/Execution -> `pipeline/analyzers_governance.py`
- `run_ai_validation`
- `run_allocation_engine`
- `run_rebalancing_policy`

## 6. Compatibility Rules (Must)
- public function signature 변경 금지
- return payload key 변경 금지
- logging/error handling semantics 유지
- `pipeline/__init__.py` export 목록 유지

## 7. Migration Sequence (General Lane Executable)

### Step A (Low Risk)
1. 새 shard 파일 생성
2. 기존 함수를 shard로 copy
3. `pipeline/analyzers.py`에서 shard import 후 re-export
4. 중복 코드 제거(원본 함수 body 제거, wrapper화)

### Step B (Medium Risk)
1. shard별 공통 헬퍼 분리(필요 최소만)
2. `pipeline/analyzers_extension.py` 중복 함수 정리(삭제 또는 explicit archive 이동)

### Step C (Stability)
1. `pipeline/__init__.py` exports 재검증
2. `main.py` import 경로 영향 없음 확인

## 8. Validation Gate for This Split

### Per-Change
- `py_compile`:
  - `pipeline/analyzers.py`
  - `pipeline/analyzers_*.py`
  - `pipeline/__init__.py`
  - `main.py`
- import smoke:
  - `from pipeline import detect_regime, run_allocation_engine`
  - `from pipeline.analyzers import analyze_hft_microstructure`
- function smoke:
  - `run_allocation_engine` required keys
  - `run_rebalancing_policy` action enum

### Milestone
- full gate는 wave 종료 시 1회 수행 (`python main.py --full`)

## 9. Open Issues (Advanced)
- `pipeline/analyzers_extension.py`의 중복 구현을 canonical로 볼지 폐기할지 결정 필요
- async 함수(`track_events_with_news`)의 호출 경로가 현재 main에서 비활성인 점 정리 필요
- analyzer shard 이후 `main.py` phase helper 분리(ADV-003)와 병행 필요

## 10. Work Orders (Draft IDs)
- `GEN-201`: shard 파일 생성 + 함수 copy + facade re-export
- `GEN-202`: analyzers_extension 중복 정리 + import 영향 정리
- `GEN-203`: per-change validation 스크립트화(`scripts/check_analyzers_split.sh`)
