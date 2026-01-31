# EIMAS 프로젝트 정리 및 구조화 계획
# EIMAS Project Cleanup & Organization Plan

바이브 코딩으로 개발된 EIMAS 프로젝트를 체계적으로 정리하여 협업과 유지보수가 쉬운 구조로 개선합니다.

---

## 📋 분석 요약 (Analysis Summary)

### 현재 프로젝트 현황
| 항목 | 수량 | 비고 |
|------|------|------|
| Python 파일 | 164개 | 핵심 + 테스트 + 문서 |
| 루트 레벨 main 파일 | 3개 | 중복 진입점 |
| lib/ 모듈 | 71개 | 분석기, 수집기 혼재 |
| 아카이브/deprecated | 4개 폴더 | 정리 필요 |
| 스키마 정의 | 2개 파일 | core/ vs pipeline/ 중복 |

### 주요 문제점

1. **중복된 진입점**
   - `main.py` - 현재 사용 중 (371줄)
   - `main_integrated.py` - 리팩토링 버전 (216줄)
   - `main_legacy.py` - 레거시 코드 (1149줄)

2. **스키마 중복**
   - `core/schemas.py` - 611줄 (에이전트 통신용)
   - `pipeline/schemas.py` - 641줄 (결과 저장용, core 임포트)

3. **lib/ 모듈 과밀**
   - 수집기(Collector): 8개 클래스 분산
   - 분석기(Analyzer): 20+ 클래스 분산
   - deprecated/ 폴더에 9개 파일 (정리 필요)

4. **아카이브 폴더 산재**
   - `lib/deprecated/` - 9개 파일
   - `pipeline/archive/` - 레거시 full_pipeline
   - `core/archive/` - debate_framework.py
   - `agents/archive/` - top_down_orchestrator, visualization_agent

---

## User Review Required

> [!IMPORTANT]
> **진입점 통합 결정 필요**
> 현재 3개의 main 파일이 존재합니다. `main.py`를 주 진입점으로 유지하고 나머지를 아카이브할 것을 권장합니다. 다른 방식을 선호하시면 알려주세요.

> [!WARNING]
> **Deprecated 코드 처리**
> `lib/deprecated/` 폴더의 9개 파일을 삭제하거나 별도 아카이브로 이동할 수 있습니다. 삭제 전 사용 여부를 확인하겠습니다.

---

## Proposed Changes

### 1. 진입점 통합 (Entry Point Consolidation)

#### [ARCHIVE] `main_legacy.py`
→ `archive/legacy/main_legacy.py`로 이동

#### [ARCHIVE] `main_integrated.py`
→ `archive/legacy/main_integrated.py`로 이동 (또는 main.py와 병합 후 삭제)

#### [MODIFY] `main.py`
- 독스트링 개선
- 사용하지 않는 import 정리
- 함수별 책임 명확화

---

### 2. lib/ 모듈 구조화 (Library Reorganization)

현재 `lib/` 폴더는 71개 파일이 평면 구조로 배치되어 있습니다. 기능별 하위 디렉토리로 정리합니다.

```diff
lib/
-├── data_collector.py
-├── fred_collector.py
-├── crypto_collector.py
-├── ...
+├── collectors/             # 데이터 수집기
+│   ├── __init__.py
+│   ├── base.py            # BaseCollector 인터페이스
+│   ├── market.py          # DataManager, MarketDataCollector
+│   ├── fred.py            # FREDCollector
+│   ├── crypto.py          # CryptoCollector
+│   └── extended.py        # ExtendedDataCollector
+│
+├── analyzers/              # 분석 엔진
+│   ├── __init__.py
+│   ├── base.py            # BaseAnalyzer 인터페이스
+│   ├── regime.py          # RegimeDetector, GMMRegimeAnalyzer
+│   ├── liquidity.py       # LiquidityAnalyzer
+│   ├── microstructure.py  # MicrostructureAnalyzer, VPIN
+│   ├── sentiment.py       # SentimentAnalyzer
+│   └── causal.py          # GrangerCausalityAnalyzer
+│
+├── reports/                # 리포트 생성
+│   ├── __init__.py
+│   ├── ai_report.py
+│   ├── final_report.py
+│   └── portfolio_report.py
+│
+├── strategies/             # 포트폴리오 전략
+│   ├── __init__.py
+│   ├── adaptive.py
+│   ├── portfolio_optimizer.py
+│   └── risk_manager.py
+│
+├── db/                     # 데이터베이스 인터페이스
+│   ├── __init__.py
+│   ├── trading_db.py
+│   ├── event_db.py
+│   └── unified_store.py
+│
+└── utils/                  # 유틸리티
    ├── __init__.py
    └── json_converter.py
```

---

### 3. 스키마 통합 (Schema Consolidation)

#### [MODIFY] `core/schemas.py`
- 에이전트 통신용 스키마 유지
- 다음 클래스 포함: `AgentRequest`, `AgentResponse`, `AgentOpinion`, `Consensus`, `Conflict`

#### [MODIFY] `pipeline/schemas.py`
- 결과 저장용 스키마 유지
- core.schemas를 임포트하여 재사용
- 다음 클래스 포함: `EIMASResult`, `FREDSummary`, `RegimeResult`, `DebateResult` 등

```python
# pipeline/schemas.py 개선안
from core.schemas import AgentOutputs, DebateResults, VerificationResults  # 재사용

# pipeline 전용 스키마만 정의
@dataclass
class EIMASResult:
    """통합 분석 결과 - 최종 JSON 출력용"""
    ...
```

---

### 4. Archive 폴더 통합 (Archive Consolidation)

현재 4곳에 분산된 아카이브 폴더를 루트의 단일 `archive/` 폴더로 통합합니다.

#### [NEW] `archive/`
```
archive/
├── README.md              # 아카이브 설명
├── legacy/               # 레거시 코드
│   ├── main_legacy.py
│   ├── main_integrated.py
│   └── full_pipeline.py
├── deprecated/           # lib/deprecated 이동
│   └── ...
└── agents/               # agents/archive 이동
    └── ...
```

#### [DELETE] 정리 대상 폴더들
- `lib/deprecated/` → archive/deprecated/로 이동 후 삭제
- `pipeline/archive/` → archive/legacy/로 이동 후 삭제
- `core/archive/` → archive/core/로 이동 후 삭제
- `agents/archive/` → archive/agents/로 이동 후 삭제

---

### 5. 코드 품질 개선 (Code Quality)

#### 5.1 함수/클래스 이름 개선

| 현재 이름 | 제안 이름 | 이유 |
|----------|----------|------|
| `_safe_call` | `safe_invoke_with_warning` | 명확한 동작 표현 |
| `_set_liquidity` | `_apply_liquidity_analysis` | 동작 의미 명확화 |
| `run_full_pipeline` | `execute_complete_analysis_pipeline` | 더 서술적 |

#### 5.2 핵심 모듈 주석 추가

다음 파일들에 영문 + 한글 병기 주석을 추가합니다:

- `main.py` - 파이프라인 흐름도
- `pipeline/__init__.py` - 모듈 익스포트 설명
- `agents/orchestrator.py` - 에이전트 조정 로직
- `lib/critical_path.py` - 리스크 계산 알고리즘

```python
# 예시: main.py 주석 개선
def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False,
    full_mode: bool = False
) -> EIMASResult:
    """
    EIMAS 통합 분석 파이프라인 실행 
    Execute the unified EIMAS analysis pipeline.
    
    Pipeline Flow:
        Phase 1: Data Collection (FRED, Market, Crypto)
        Phase 2: Market Analysis (Regime, Risk, Liquidity)
        Phase 3: AI Agent Debate (Dual Mode)
        Phase 4: Realtime Streaming (Optional)
        Phase 5: Result Storage (JSON, DB)
        Phase 6: Report Generation (AI-powered)
        Phase 7: Validation (Whitening, Fact Check)
        Phase 8: Multi-LLM Validation (--full only)
    
    Args:
        enable_realtime: Enable real-time streaming mode
        realtime_duration: Duration in seconds for streaming
        quick_mode: Skip heavy computations (bubble, DTW)
        generate_report: Generate AI-powered report
        full_mode: Include Multi-LLM validation (API cost)
    
    Returns:
        EIMASResult: Comprehensive analysis results
    """
```

---

### 6. 리팩토링 (Refactoring with SOLID)

#### 6.1 Single Responsibility Principle

| 파일 | 현재 줄 수 | 문제점 | 개선 방안 |
|------|-----------|--------|----------|
| `lib/critical_path.py` | 160,525 | 거대 모듈 | 클래스별 분리 |
| `lib/ai_report_generator.py` | 100,115 | 다중 책임 | 리포트 유형별 분리 |
| `lib/final_report_agent.py` | 72,791 | 리포트 + 분석 혼합 | 관심사 분리 |

#### 6.2 인터페이스 도입 (Open/Closed Principle)

```python
# lib/collectors/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseCollector(ABC):
    """Base interface for all data collectors"""
    
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect data from source"""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate collected data"""
        pass
```

```python
# lib/analyzers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAnalyzer(ABC):
    """Base interface for all analyzers"""
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on input data"""
        pass
    
    @abstractmethod
    def get_summary(self) -> str:
        """Return human-readable summary"""
        pass
```

---

### 7. 문서화 (Documentation)

#### [NEW] `ARCHITECTURE.md`
프로젝트 아키텍처 상세 설명

#### [MODIFY] `README.md`
- 디렉토리 구조 업데이트
- 빠른 시작 가이드 개선
- 기여자 가이드 추가

#### [NEW] `CONTRIBUTING.md`
- 개발 환경 설정
- 코드 스타일 가이드
- PR 절차

#### [NEW] `CHANGELOG.md`
- 이번 정리 작업 기록
- 향후 변경 사항 추적

---

## Verification Plan

### 기존 기능 테스트
```bash
# 1. 기존 테스트 실행
python -m pytest tests/ -v

# 2. 메인 파이프라인 실행 확인
python main.py --quick

# 3. API 서버 테스트
python api/main.py &
curl http://localhost:8000/health
```

### 임포트 검증
```bash
# 리팩토링 후 임포트 확인
python -c "from pipeline import *; print('Pipeline imports OK')"
python -c "from lib.collectors import *; print('Collectors imports OK')"
python -c "from lib.analyzers import *; print('Analyzers imports OK')"
```

### Manual Verification
- 웹 대시보드 정상 작동 확인 (`localhost:3002`)
- JSON 출력 포맷 호환성 확인
- API 엔드포인트 호환성 확인

---

## 실행 순서 (Phased Approach)

### Phase 1: 안전한 정리 (Low Risk)
1. Archive 폴더 통합
2. deprecated 파일 이동
3. README 업데이트

### Phase 2: 구조 개선 (Medium Risk)
1. lib/ 하위 디렉토리 생성
2. 파일 이동 및 임포트 수정
3. __init__.py 업데이트

### Phase 3: 코드 품질 (Lower Priority)
1. 주석 및 독스트링 추가
2. 함수명 개선 (호환성 유지하며 alias 제공)
3. 인터페이스 도입

---

## 예상 결과

| 지표 | Before | After |
|------|--------|-------|
| 루트 레벨 main 파일 | 3개 | 1개 |
| lib/ 직접 파일 | 71개 | ~10개 (하위 디렉토리 정리) |
| 아카이브 폴더 | 4개 분산 | 1개 통합 |
| 문서화 수준 | 부분적 | 체계적 |
