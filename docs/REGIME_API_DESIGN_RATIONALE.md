# EIMAS Regime Change & API 설계 근거

> **작성일**: 2025-12-27
> **버전**: 2.0.0

---

## 1. Stage 2.5: Regime Change Detection

### 1.1 왜 필요한가?

경제/금융 데이터 분석에서 **구조적 변화(Structural Break)**를 무시하면 심각한 오류가 발생한다.

| 문제 | 설명 | 예시 |
|-----|------|------|
| **Spurious Regression** | 레짐 변화 전후 데이터를 섞으면 가짜 상관관계 발생 | 2008년 금융위기 전후 데이터로 회귀분석 시 계수 왜곡 |
| **Parameter Instability** | 모델 파라미터가 시간에 따라 변함 | 저금리 시대 vs 고금리 시대의 주가-금리 관계 역전 |
| **Forecast Failure** | 과거 패턴이 미래에 적용 안 됨 | COVID-19 이후 소비 패턴 변화 |

### 1.2 왜 Stage 2.5인가?

```
Stage 2: Top-Down Analysis (거시 환경 파악)
    ↓
Stage 2.5: Regime Check ← 여기서 "데이터를 어떻게 다룰지" 결정
    ↓
Stage 3: Methodology Selection (분석 방법 선택)
```

**배치 이유**:
1. **Top-Down 이후**: 거시 환경을 먼저 파악해야 레짐 판단의 맥락이 생김
2. **Methodology 이전**: 레짐 변화가 있으면 분석 방법도 달라져야 함
   - 레짐 변화 있음 → 최근 데이터만 사용, 구조변화 모델 적용
   - 레짐 변화 없음 → 전체 데이터 사용, 표준 모델 적용

### 1.3 RegimeContext가 전달하는 정보

```python
@dataclass
class RegimeContext:
    regime_type: RegimeType      # 현재 레짐 (expansion/contraction/crisis 등)
    regime_aware: bool           # 레짐 변화 감지 여부
    context_adjustment: Dict     # 후속 분석에 대한 지시사항
    data_split_date: datetime    # 데이터 분할 기준일
    use_post_regime_only: bool   # 레짐 변화 이후 데이터만 사용 여부
```

**후속 Stage에 미치는 영향**:
- `regime_aware=True` → Core Analysis에서 데이터 필터링
- `data_split_date` 설정 → 해당 날짜 이후 데이터만 분석
- `context_adjustment` → 분석 시 고려사항 전달

### 1.4 경제학적 근거

| 이론 | 저자 | 핵심 아이디어 |
|-----|------|--------------|
| **Structural Break Test** | Chow (1960) | 회귀 계수가 특정 시점에서 변했는지 검정 |
| **Regime Switching Model** | Hamilton (1989) | 경제가 여러 상태(레짐) 사이를 전환한다고 모델링 |
| **CUSUM Test** | Brown et al. (1975) | 누적합으로 파라미터 불안정성 탐지 |

---

## 2. FastAPI 모듈 구조

### 2.1 왜 모듈 분리인가?

**단일 파일 구조의 문제점**:
```python
# ❌ 나쁜 예: 모든 것이 한 파일에
# server.py - 2000줄짜리 파일
@app.post("/analyze") ...
@app.get("/regime") ...
@app.get("/debate") ...
# 모든 Pydantic 모델도 여기에...
```

**모듈 분리의 이점**:

| 이점 | 설명 |
|-----|------|
| **관심사 분리** | 각 도메인(분석, 레짐, 토론)이 독립적으로 관리됨 |
| **테스트 용이성** | 개별 라우터를 독립적으로 테스트 가능 |
| **확장성** | 새 기능 추가 시 새 라우터 파일만 생성 |
| **협업** | 여러 개발자가 충돌 없이 작업 가능 |

### 2.2 파일 구조 설계

```
api/
├── server.py           # 앱 설정, 미들웨어, 라우터 등록
├── models/
│   ├── requests.py     # 입력 검증 (Pydantic)
│   └── responses.py    # 출력 직렬화 (Pydantic)
└── routes/
    ├── analysis.py     # 핵심 분석 로직
    ├── regime.py       # 레짐 관련 조회
    ├── debate.py       # 토론 결과 조회
    └── health.py       # 헬스 체크
```

**설계 원칙**:
1. **models/**: 데이터 구조 정의 (무엇이 들어오고 나가는지)
2. **routes/**: 비즈니스 로직 (어떻게 처리하는지)
3. **server.py**: 인프라 설정 (CORS, 미들웨어 등)

### 2.3 상태 공유 패턴

```python
# 각 라우터 파일
_state: Dict[str, Any] = {}

def set_state(state: dict):
    global _state
    _state = state

# server.py에서 공유 상태 주입
_global_state = {"results": {}, "last_analysis_id": None}
set_analysis_state(_global_state)
set_regime_state(_global_state)
```

**왜 이렇게 했나?**:
- FastAPI는 stateless가 기본이지만, 분석 결과 캐싱이 필요
- 의존성 주입보다 간단한 패턴으로 상태 공유
- 추후 Redis 등으로 쉽게 교체 가능

### 2.4 API 설계 원칙

| 원칙 | 적용 |
|-----|------|
| **RESTful** | 자원 중심 URL (`/api/analyze`, `/api/regime`) |
| **일관성** | 모든 응답이 동일한 구조 (status, data, error) |
| **문서화** | Pydantic 모델로 자동 OpenAPI 생성 |
| **버전관리** | `/api/v1/...` 형태로 확장 가능 |

---

## 3. 엔드포인트 설계 근거

### 3.1 POST /api/analyze

**왜 POST인가?**
- 분석 요청은 서버 상태를 변경함 (결과 캐싱)
- 요청 본문에 복잡한 데이터 전달 필요
- 멱등성이 보장되지 않음 (매번 새 analysis_id 생성)

**요청 파라미터 설계**:
```python
class AnalysisRequest:
    question: str           # 필수: 분석 질문
    data: Optional[Dict]    # 선택: 외부 데이터
    use_mock: bool = False  # 개발/테스트용
    stop_at_level: str      # Top-Down 분석 깊이 조절
    research_goal: str      # 방법론 선택에 영향
    skip_stages: List[str]  # 특정 스테이지 건너뛰기
```

### 3.2 GET /api/regime

**왜 별도 엔드포인트인가?**
- 레짐 정보만 필요한 경우가 많음
- 전체 분석 결과보다 가벼운 응답
- 프론트엔드에서 레짐 상태 표시에 활용

### 3.3 GET /api/debate

**왜 별도 엔드포인트인가?**
- 방법론 토론과 경제학파 해석은 별도 관심사
- UI에서 "왜 이 방법론을 선택했는지" 설명할 때 사용
- consensus_points, divergence_points로 투명성 제공

---

## 4. 향후 확장 계획

### 4.1 WebSocket 스트리밍
```
/ws/analyze → 실시간 진행 상황 전송
Stage 1 완료 → {"stage": "data_collection", "status": "done"}
Stage 2 완료 → {"stage": "top_down", "status": "done"}
...
```

### 4.2 외부 도구 연동
```
POST /api/visualize → v0.dev로 차트 생성
POST /api/research  → Elicit으로 논문 검색
```

### 4.3 배치 분석
```
POST /api/batch/analyze → 여러 질문 동시 분석
GET /api/batch/{batch_id} → 배치 결과 조회
```

---

## 5. 참고 문헌

1. Chow, G. C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear Regressions"
2. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
3. FastAPI Documentation - https://fastapi.tiangolo.com/
4. Pydantic V2 - https://docs.pydantic.dev/

---

*작성: Claude Code*
