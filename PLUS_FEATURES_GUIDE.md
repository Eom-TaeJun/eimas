# EIMAS Plus Features Integration Guide

## 개요

`plus/` 폴더의 기능들을 EIMAS Multi-Agent 시스템에 통합하기 위한 가이드라인과 워크플로우.

---

## 1. 기존 Plus 기능 요약

### 1.1 Dashboard Generator (`dashboard_generator.py`)

**목적**: Market Anomaly 탐지 결과를 인터랙티브 HTML 대시보드로 시각화

**핵심 컴포넌트**:

| 섹션 | 함수 | 설명 |
|------|------|------|
| 자산군별 위험 현황 | `generate_asset_risk_section()` | 주식/채권/원자재/환율/암호화폐별 리스크 카드 |
| 레짐 표시 | `generate_regime_display()` | BULL/BEAR/TRANSITION/CRISIS 레짐 게이지 |
| 암호화폐 패널 | `generate_crypto_panel_html()` | BTC/ETH/SOL/XRP 전용 분석 |
| 스필오버 분석 | `_generate_spillover_section()` | 자산간 전이 효과 시각화 |
| 마르코프 레짐 | `_generate_markov_regime_section()` | 레짐 전환 확률 차트 (Chart.js) |
| 리스크 메트릭 | `_generate_risk_metrics_section()` | 위험조정수익률 (Sharpe, Sortino 등) |
| 매크로 환경 | `_generate_macro_environment_section()` | 거시경제 선행지표 |
| Critical Path | `generate_critical_path_section()` | Granger Causality 기반 전이 경로 |
| LLM 요약 | `_generate_llm_summary_section()` | Claude API 기반 AI 해석 |

**주요 입력 파라미터** (`generate_dashboard()`):
```python
def generate_dashboard(
    signals: List[Dict],           # 이상 신호 목록
    summary: str,                  # 요약 텍스트
    interpretations: List[Dict],   # AI 해석
    news: List[Dict],              # 뉴스 데이터
    regime_data: Dict,             # 레짐 정보
    crypto_panel: Dict,            # 암호화폐 패널
    risk_data: Dict,               # ML 기반 위험 확률
    critical_path_data: Dict,      # Critical Path 분석
    risk_metrics: Dict,            # 위험조정수익률
    macro_indicators: Dict,        # 거시경제 지표
    llm_summary: str               # LLM 요약
) -> str:
```

### 1.2 LASSO Forecasting (PDF 논문 기반)

**목적**: Fed 금리 기대 변화 예측 및 변수 선택

**방법론**:
1. **LASSO (L1 정규화)**: 고차원 변수에서 핵심 변수 선택
2. **Post-LASSO HAC OLS**: Newey-West 표준오차로 통계적 추론
3. **TimeSeriesSplit**: 시계열 교차검증 (5-fold)

**Horizon 분류**:
| Horizon | 일수 | 특성 |
|---------|------|------|
| VeryShort | ≤30일 | 거의 확정된 정보, R² ≈ 0 |
| Short | 31-90일 | 신용시장/인플레이션 기대 중심 |
| Long | ≥180일 | 광범위 거시변수, R² ≈ 0.64 |

**핵심 변수 그룹**:
- Credit: `d_Baa_Yield`, `d_Spread_Baa`, `d_HighYield_Rate`
- Dollar: `Ret_Dollar_Idx`, `d_Dollar_Idx`
- Inflation: `d_Breakeven5Y`
- Risk: `Ret_VIX`, `d_VIX`

**제외 변수**: Treasury 관련 (Simultaneity 문제 방지)

### 1.3 결과물

| 파일 | 형식 | 내용 |
|------|------|------|
| `dashboard_*.html` | HTML | 인터랙티브 대시보드 |
| 논문 PDF | PDF | LASSO 분석 결과 (학술 형식) |
| 방향성 문서 | DOCX | 프로젝트 방향 정리 |

---

## 2. EIMAS 통합 아키텍처

### 2.1 현재 EIMAS 구조
```
eimas/
├── agents/
│   ├── base_agent.py         # BaseAgent 추상 클래스
│   ├── analysis_agent.py     # CriticalPath 분석
│   ├── orchestrator.py       # 워크플로우 조정
│   ├── forecast_agent.py     # [미구현] LASSO 예측
│   └── ...
├── core/
│   ├── config.py             # API 설정
│   ├── schemas.py            # 데이터 스키마
│   └── debate.py             # 토론 프로토콜
└── lib/
    ├── critical_path.py      # CriticalPathAggregator
    └── data_collector.py     # DataManager
```

### 2.2 Plus 기능 통합 위치
```
eimas/
├── agents/
│   ├── forecast_agent.py     # ← LASSO 예측 통합
│   └── visualization_agent.py # ← [신규] 대시보드 생성
├── lib/
│   ├── dashboard_generator.py # ← plus/에서 이동
│   └── lasso_model.py        # ← [신규] LASSO 래핑
└── outputs/
    └── dashboards/           # HTML 출력 저장
```

---

## 3. 통합 워크플로우

### 3.1 Phase 1: ForecastAgent 구현

**파일**: `eimas/agents/forecast_agent.py`

**참고 자료**:
- `plus/` PDF 논문의 LASSO 방법론
- 기존 `forecasting_20251218.py`

**구현 요소**:
```python
class ForecastAgent(BaseAgent):
    """LASSO 기반 Fed 금리 예측 에이전트"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.lasso_model = LassoCV(cv=TimeSeriesSplit(n_splits=5))
        self.scaler = StandardScaler()

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        """
        1. 데이터 전처리 (Treasury 변수 제외)
        2. Horizon별 분리 (VeryShort/Short/Long)
        3. LASSO 학습 + 변수 선택
        4. Post-LASSO OLS (HAC)
        5. ForecastResult 반환
        """
        pass

    async def form_opinion(self, topic: str, context: Dict) -> AgentOpinion:
        """
        토픽별 의견 형성:
        - rate_direction: 금리 방향 (UP/DOWN/HOLD)
        - rate_magnitude: 변화 폭 (bp)
        - forecast_confidence: 예측 신뢰도
        """
        pass

    def _filter_treasury_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simultaneity 문제 방지를 위해 Treasury 변수 제외"""
        treasury_cols = ['d_US10Y', 'd_US2Y', 'd_RealYield10Y', 'd_Term_Spread']
        return df.drop(columns=treasury_cols, errors='ignore')
```

### 3.2 Phase 2: Dashboard 통합

**파일**: `eimas/lib/dashboard_generator.py`

**수정 사항**:
1. `plus/dashboard_generator.py` → `eimas/lib/` 이동
2. EIMAS 스키마와 호환되도록 입력 형식 조정
3. 멀티에이전트 결과 시각화 추가

**새로운 섹션 추가**:
```python
def generate_multi_agent_section(
    agent_opinions: List[AgentOpinion],
    consensus: Consensus,
    conflicts: List[Conflict]
) -> str:
    """멀티에이전트 토론 결과 시각화"""
    pass
```

### 3.3 Phase 3: VisualizationAgent 구현

**파일**: `eimas/agents/visualization_agent.py`

```python
class VisualizationAgent(BaseAgent):
    """대시보드 생성 전용 에이전트"""

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        """
        1. 다른 에이전트 결과 수집
        2. dashboard_generator 호출
        3. HTML 파일 저장
        4. 파일 경로 반환
        """
        from lib.dashboard_generator import generate_dashboard

        html = generate_dashboard(
            signals=request.context.get('signals', []),
            regime_data=request.context.get('regime_data', {}),
            # ... 나머지 파라미터
        )

        output_path = f"outputs/dashboards/dashboard_{timestamp}.html"
        with open(output_path, 'w') as f:
            f.write(html)

        return AgentResponse(result={'dashboard_path': output_path})
```

### 3.4 Phase 4: 전체 파이프라인 통합

**파일**: `eimas/main.py` 수정

```python
async def run_full_pipeline():
    """
    1. DataManager로 데이터 수집
    2. AnalysisAgent로 Critical Path 분석
    3. ForecastAgent로 LASSO 예측
    4. MetaOrchestrator로 토론 및 합의
    5. VisualizationAgent로 대시보드 생성
    """

    # 데이터 수집
    data_manager = DataManager()
    market_data = await data_manager.collect_all()

    # 에이전트 실행
    orchestrator = MetaOrchestrator(config)

    # 토론 주제 자동 감지
    topics = orchestrator.auto_detect_topics(market_data)

    # 멀티에이전트 토론
    result = await orchestrator.run_with_debate(
        request=AgentRequest(context=market_data),
        topics=topics,
        agents=[analysis_agent, forecast_agent, strategy_agent]
    )

    # 대시보드 생성
    dashboard_agent = VisualizationAgent(config)
    dashboard_result = await dashboard_agent.execute(
        AgentRequest(context={
            'signals': result.signals,
            'agent_opinions': result.opinions,
            'consensus': result.consensus,
            **market_data
        })
    )

    return dashboard_result.result['dashboard_path']
```

---

## 4. 데이터 스키마 확장

### 4.1 ForecastResult 확장 (`core/schemas.py`)

```python
@dataclass
class ForecastResult:
    """LASSO 예측 결과"""
    horizon: str                    # VeryShort/Short/Long
    selected_variables: List[str]   # LASSO 선택 변수
    coefficients: Dict[str, float]  # 변수별 계수
    r_squared: float               # 설명력
    predicted_rate_change: float   # 예측 금리 변화 (bp)
    confidence_interval: Tuple[float, float]  # 신뢰구간
    mincer_zarnowitz_beta: float   # 예측 효율성

@dataclass
class LASSODiagnostics:
    """LASSO 진단 결과"""
    lambda_optimal: float
    n_selected: int
    vif_scores: Dict[str, float]   # 다중공선성 검사
    hac_std_errors: Dict[str, float]
```

### 4.2 DashboardConfig 추가

```python
@dataclass
class DashboardConfig:
    """대시보드 설정"""
    theme: str = 'dark'           # dark/light
    include_crypto: bool = True
    include_regime: bool = True
    include_critical_path: bool = True
    include_lasso_results: bool = True
    include_agent_debate: bool = True
    chart_library: str = 'chartjs'  # chartjs/plotly
```

---

## 5. 구현 우선순위

| 순위 | 작업 | 파일 | 예상 복잡도 |
|------|------|------|------------|
| 1 | ForecastAgent 구현 | `agents/forecast_agent.py` | 높음 |
| 2 | LASSO 래핑 라이브러리 | `lib/lasso_model.py` | 중간 |
| 3 | Dashboard Generator 이동 | `lib/dashboard_generator.py` | 낮음 |
| 4 | VisualizationAgent 구현 | `agents/visualization_agent.py` | 중간 |
| 5 | 스키마 확장 | `core/schemas.py` | 낮음 |
| 6 | Main 파이프라인 통합 | `main.py` | 중간 |
| 7 | 통합 테스트 | `tests/test_integration.py` | 높음 |

---

## 6. 주요 고려사항

### 6.1 LASSO 모델 관련

- **Treasury 변수 제외**: Simultaneity bias 방지 필수
- **Horizon 분리**: 각 horizon별 별도 모델 학습
- **다중공선성 주의**: VIF > 10인 변수 그룹은 결합 효과로 해석
- **HAC 표준오차**: Newey-West lag=5 (1주일 거래일)

### 6.2 대시보드 관련

- **Chart.js CDN 의존성**: 오프라인 사용 시 로컬 번들 필요
- **HTML 크기**: 현재 ~185KB, 최적화 고려
- **한글 지원**: 이미 완료 (`lang="ko"`)

### 6.3 에이전트 통합

- **비동기 실행**: `asyncio` 기반 병렬 처리
- **에러 핸들링**: `BaseAgent`의 재시도 로직 활용
- **토론 프로토콜**: 최대 3라운드, 85% 일관성 임계값

---

## 7. 참고 자료

- PDF 논문: "Market Expectations and Structural Changes in Fed Policy"
- 기존 코드: `market_anomaly_detector_v2.2/`
- EIMAS 아키텍처: `CLAUDE.md`

---

## 8. 상세 구현 명세서 (LLM 코드 생성용)

> **이 섹션은 Cursor, Gemini 등 LLM이 코드를 생성할 때 참조하는 명세서입니다.**
> 각 함수/클래스의 정확한 동작을 정의하며, 이 명세를 따라 구현하면 됩니다.

---

### 8.1 `LASSOForecaster` 클래스

**파일 위치**: `eimas/lib/lasso_model.py`

**역할**: LASSO 기반 Fed 금리 예측 모델을 래핑하는 유틸리티 클래스

#### 8.1.1 클래스 구조

```
LASSOForecaster
├── __init__(config: LASSOConfig)
├── fit(X: DataFrame, y: Series, horizon: str) -> LASSOResult
├── predict(X: DataFrame) -> np.ndarray
├── get_selected_variables() -> List[str]
├── get_coefficients() -> Dict[str, float]
├── compute_hac_standard_errors(X: DataFrame, y: Series) -> Dict[str, float]
├── compute_vif_scores(X: DataFrame) -> Dict[str, float]
└── _filter_treasury_variables(df: DataFrame) -> DataFrame
```

#### 8.1.2 `__init__` 메서드

**입력 파라미터**:
| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `config` | `LASSOConfig` | 필수 | LASSO 설정 객체 |

**`LASSOConfig` 구조**:
```python
@dataclass
class LASSOConfig:
    n_splits: int = 5                    # TimeSeriesSplit fold 수
    max_iter: int = 10000                # LASSO 최대 반복
    tol: float = 1e-4                    # 수렴 허용오차
    hac_lag: int = 5                     # Newey-West lag (1주일 거래일)
    excluded_prefixes: List[str] = None  # 제외할 변수 접두사
```

**기본 제외 변수** (Treasury 관련):
```python
DEFAULT_EXCLUDED = [
    'd_US10Y', 'd_US2Y', 'd_RealYield10Y', 'd_Term_Spread',
    'Ret_Treasury_7_10Y', 'Ret_Treasury_1_3Y', 'Ret_Treasury_20Y'
]
```

**초기화 로직**:
1. `LassoCV` 인스턴스 생성 (cv=TimeSeriesSplit(n_splits))
2. `StandardScaler` 인스턴스 생성
3. 내부 상태 변수 초기화 (`_fitted_model`, `_scaler`, `_selected_vars`)

---

#### 8.1.3 `fit` 메서드

**목적**: 주어진 데이터로 LASSO 모델 학습 및 변수 선택

**입력 파라미터**:
| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `X` | `pd.DataFrame` | 설명변수 (N x P), 컬럼명 = 변수명 |
| `y` | `pd.Series` | 종속변수 (`d_Exp_Rate`, 기대금리 일별 변화) |
| `horizon` | `str` | "VeryShort" / "Short" / "Long" |

**출력**:
```python
@dataclass
class LASSOResult:
    horizon: str
    lambda_optimal: float          # CV로 선택된 최적 lambda
    selected_variables: List[str]  # 계수 != 0인 변수 목록
    coefficients: Dict[str, float] # {변수명: 계수}
    r_squared: float              # 설명력 (0~1)
    n_observations: int
    n_selected: int
```

**구현 로직**:
```
1. Treasury 변수 필터링
   - _filter_treasury_variables(X) 호출
   - 결과: X_filtered

2. Horizon별 데이터 분리
   - VeryShort: days_to_meeting <= 30
   - Short: 31 <= days_to_meeting <= 90
   - Long: days_to_meeting >= 180
   - 참고: 91~179일 구간은 분석에서 제외

3. 표준화
   - X_scaled = scaler.fit_transform(X_filtered)
   - 주의: y는 표준화하지 않음 (해석 용이성)

4. LASSO 학습
   - lasso.fit(X_scaled, y)
   - lambda_optimal = lasso.alpha_

5. 변수 선택
   - selected_idx = np.where(lasso.coef_ != 0)[0]
   - selected_vars = X_filtered.columns[selected_idx].tolist()

6. R² 계산
   - y_pred = lasso.predict(X_scaled)
   - r_squared = 1 - (sum((y - y_pred)^2) / sum((y - y.mean())^2))

7. LASSOResult 반환
```

**에러 처리**:
- `X`가 비어있으면: `ValueError("Empty feature matrix")`
- 선택된 변수가 0개면: 경고 로그 출력 후 빈 결과 반환 (VeryShort에서 정상)
- 수렴 실패 시: `max_iter` 증가 후 재시도, 3회 실패 시 예외

---

#### 8.1.4 `compute_hac_standard_errors` 메서드

**목적**: Post-LASSO OLS의 HAC(Newey-West) 표준오차 계산

**입력**:
| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `X` | `pd.DataFrame` | LASSO 선택된 변수만 포함 |
| `y` | `pd.Series` | 종속변수 |

**출력**: `Dict[str, float]` - {변수명: HAC 표준오차}

**구현 로직**:
```
1. OLS 회귀 (statsmodels 사용)
   - model = sm.OLS(y, sm.add_constant(X))
   - results = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lag})

2. 표준오차 추출
   - std_errors = results.bse  # Series
   - return {var: std_errors[var] for var in X.columns}
```

**의존성**: `statsmodels.api`

---

#### 8.1.5 `compute_vif_scores` 메서드

**목적**: 다중공선성 진단 (VIF > 10이면 주의)

**입력**: `X: pd.DataFrame`

**출력**: `Dict[str, float]` - {변수명: VIF 점수}

**구현 로직**:
```
from statsmodels.stats.outliers_influence import variance_inflation_factor

1. 상수항 추가
   X_with_const = sm.add_constant(X)

2. 각 변수별 VIF 계산
   vif_scores = {}
   for i, col in enumerate(X.columns):
       vif_scores[col] = variance_inflation_factor(X_with_const.values, i+1)

3. return vif_scores
```

---

### 8.2 `ForecastAgent` 클래스

**파일 위치**: `eimas/agents/forecast_agent.py`

**역할**: LASSO 예측을 수행하고 멀티에이전트 토론에 참여하는 에이전트

**상속**: `BaseAgent` (from `eimas/agents/base_agent.py`)

#### 8.2.1 클래스 구조

```
ForecastAgent(BaseAgent)
├── __init__(config: AgentConfig)
├── async _execute(request: AgentRequest) -> AgentResponse
├── async form_opinion(topic: str, context: Dict) -> AgentOpinion
├── _prepare_features(market_data: Dict) -> pd.DataFrame
├── _classify_horizon(days_to_meeting: int) -> str
└── _interpret_coefficients(result: LASSOResult) -> str
```

#### 8.2.2 `_execute` 메서드

**목적**: LASSO 분석 실행 및 예측 결과 반환

**입력**: `AgentRequest`
```python
request.context = {
    'market_data': pd.DataFrame,      # 일별 금융/거시 데이터
    'target_meetings': List[Dict],    # FOMC 회의 목록
    'current_date': str,              # 'YYYY-MM-DD'
}
```

**출력**: `AgentResponse`
```python
AgentResponse(
    agent_id='forecast_agent',
    status='success',
    result={
        'forecasts': List[ForecastResult],  # Horizon별 예측
        'diagnostics': LASSODiagnostics,
        'interpretation': str,               # 자연어 해석
    },
    metadata={
        'execution_time': float,
        'data_range': Tuple[str, str],
    }
)
```

**구현 로직**:
```
1. 데이터 준비
   - market_data = request.context['market_data']
   - X = _prepare_features(market_data)
   - y = market_data['d_Exp_Rate']

2. Horizon별 분석 루프
   forecasts = []
   for horizon in ['VeryShort', 'Short', 'Long']:
       # 해당 horizon 데이터 필터링
       mask = _get_horizon_mask(market_data['days_to_meeting'], horizon)
       X_h, y_h = X[mask], y[mask]

       # LASSO 학습
       lasso = LASSOForecaster(config)
       result = lasso.fit(X_h, y_h, horizon)

       # HAC 표준오차 (선택된 변수가 있을 때만)
       if result.selected_variables:
           X_selected = X_h[result.selected_variables]
           hac_errors = lasso.compute_hac_standard_errors(X_selected, y_h)
           vif_scores = lasso.compute_vif_scores(X_selected)
       else:
           hac_errors, vif_scores = {}, {}

       forecasts.append(ForecastResult(
           horizon=horizon,
           selected_variables=result.selected_variables,
           coefficients=result.coefficients,
           r_squared=result.r_squared,
           hac_std_errors=hac_errors,
           vif_scores=vif_scores,
       ))

3. 종합 진단
   diagnostics = LASSODiagnostics(
       lambda_optimal=...,
       total_vars_selected=sum(len(f.selected_variables) for f in forecasts),
       high_vif_warnings=[v for v, s in vif_scores.items() if s > 10],
   )

4. 자연어 해석 생성
   interpretation = _interpret_coefficients(forecasts)

5. AgentResponse 반환
```

---

#### 8.2.3 `form_opinion` 메서드

**목적**: 멀티에이전트 토론에서 특정 토픽에 대한 의견 형성

**입력**:
| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `topic` | `str` | 토론 주제 (아래 목록 참조) |
| `context` | `Dict` | LASSO 분석 결과 포함 |

**지원 토픽**:
| 토픽 | 설명 | 의견 형식 |
|------|------|----------|
| `rate_direction` | 금리 방향 | "UP" / "DOWN" / "HOLD" |
| `rate_magnitude` | 변화 폭 | 숫자 (bp 단위) |
| `forecast_confidence` | 예측 신뢰도 | 0.0 ~ 1.0 |
| `key_drivers` | 핵심 동인 | 변수 목록 |

**출력**: `AgentOpinion`
```python
AgentOpinion(
    agent_id='forecast_agent',
    topic=topic,
    position=str,           # 의견 (예: "DOWN")
    confidence=float,       # 0.0 ~ 1.0
    reasoning=str,          # 근거 설명
    evidence=List[str],     # 지지 증거
    caveats=List[str],      # 주의사항/한계
)
```

**구현 로직 (rate_direction 예시)**:
```
1. Long horizon 결과 추출
   long_result = context['forecasts'][2]  # Long = index 2

2. 핵심 변수 부호 분석
   key_vars = ['d_Spread_Baa', 'Ret_Dollar_Idx', 'd_Breakeven5Y']
   signals = {}
   for var in key_vars:
       if var in long_result.coefficients:
           signals[var] = long_result.coefficients[var]

3. 방향 결정 로직
   # d_Spread_Baa 음(-): 스프레드 확대 → 인하 기대 감소
   # Ret_Dollar_Idx 양(+): 달러 강세 → 인하 기대 감소
   # 최근 변화 * 계수 → 기대 변화 방향

   if weighted_signal > threshold:
       position = "UP"
   elif weighted_signal < -threshold:
       position = "DOWN"
   else:
       position = "HOLD"

4. 신뢰도 계산
   confidence = min(long_result.r_squared, 0.95)  # R²기반, 최대 0.95

5. AgentOpinion 반환
```

---

### 8.3 `VisualizationAgent` 클래스

**파일 위치**: `eimas/agents/visualization_agent.py`

**역할**: 분석 결과를 HTML 대시보드로 시각화

#### 8.3.1 클래스 구조

```
VisualizationAgent(BaseAgent)
├── __init__(config: AgentConfig, dashboard_config: DashboardConfig)
├── async _execute(request: AgentRequest) -> AgentResponse
├── _collect_agent_results(context: Dict) -> Dict
├── _generate_output_path() -> str
└── _save_dashboard(html: str, path: str) -> None
```

#### 8.3.2 `_execute` 메서드

**입력**: `AgentRequest`
```python
request.context = {
    # 기존 시장 데이터
    'signals': List[Dict],
    'regime_data': Dict,
    'risk_metrics': Dict,
    'macro_indicators': Dict,

    # 에이전트 결과 (신규)
    'agent_opinions': List[AgentOpinion],
    'consensus': Consensus,
    'conflicts': List[Conflict],
    'forecast_results': List[ForecastResult],

    # 메타데이터
    'timestamp': str,
    'project_id': str,
}
```

**출력**: `AgentResponse`
```python
AgentResponse(
    agent_id='visualization_agent',
    status='success',
    result={
        'dashboard_path': str,    # 저장된 HTML 경로
        'dashboard_size': int,    # 바이트
        'sections_generated': List[str],
    }
)
```

**구현 로직**:
```
1. 대시보드 설정 로드
   config = self.dashboard_config

2. 기존 대시보드 생성
   from lib.dashboard_generator import generate_dashboard

   base_html = generate_dashboard(
       signals=context['signals'],
       regime_data=context['regime_data'],
       risk_metrics=context['risk_metrics'],
       macro_indicators=context['macro_indicators'],
       # ... 기존 파라미터
   )

3. 멀티에이전트 섹션 추가 (신규)
   if config.include_agent_debate:
       agent_section = generate_multi_agent_section(
           context['agent_opinions'],
           context['consensus'],
           context['conflicts']
       )
       # base_html에 섹션 삽입

4. LASSO 결과 섹션 추가 (신규)
   if config.include_lasso_results:
       lasso_section = generate_lasso_section(
           context['forecast_results']
       )
       # base_html에 섹션 삽입

5. 파일 저장
   output_path = _generate_output_path()
   _save_dashboard(final_html, output_path)

6. AgentResponse 반환
```

---

### 8.4 신규 대시보드 섹션 함수

**파일 위치**: `eimas/lib/dashboard_generator.py` (기존 파일에 추가)

#### 8.4.1 `generate_multi_agent_section`

**목적**: 멀티에이전트 토론 결과를 HTML로 시각화

**입력**:
```python
def generate_multi_agent_section(
    opinions: List[AgentOpinion],
    consensus: Consensus,
    conflicts: List[Conflict]
) -> str:
```

**출력**: HTML 문자열

**생성할 UI 요소**:
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Multi-Agent Analysis                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AnalysisAgent│  │ForecastAgent │  │StrategyAgent│       │
│  │   BEARISH    │  │    HOLD      │  │   CAUTIOUS  │       │
│  │  conf: 0.75  │  │  conf: 0.68  │  │  conf: 0.72 │       │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  📊 Consensus: CAUTIOUS HOLD (Agreement: 78%)              │
│  ⚠️  Conflicts: rate_magnitude (ForecastAgent vs Strategy) │
└─────────────────────────────────────────────────────────────┘
```

**구현 로직**:
```
1. 에이전트 카드 생성
   for opinion in opinions:
       card_html = f"""
       <div class="agent-card" style="border-left: 4px solid {get_color(opinion.position)}">
           <div class="agent-name">{opinion.agent_id}</div>
           <div class="agent-position">{opinion.position}</div>
           <div class="agent-confidence">conf: {opinion.confidence:.2f}</div>
           <div class="agent-reasoning">{opinion.reasoning[:100]}...</div>
       </div>
       """

2. 합의 섹션 생성
   consensus_html = f"""
   <div class="consensus-box">
       <span class="consensus-icon">📊</span>
       <span class="consensus-text">
           Consensus: {consensus.position} (Agreement: {consensus.agreement_score*100:.0f}%)
       </span>
   </div>
   """

3. 충돌 목록 생성
   if conflicts:
       conflicts_html = "<ul class='conflict-list'>"
       for c in conflicts:
           conflicts_html += f"<li>⚠️ {c.topic}: {c.agent_a} vs {c.agent_b}</li>"
       conflicts_html += "</ul>"

4. 전체 섹션 조립 및 반환
```

---

#### 8.4.2 `generate_lasso_section`

**목적**: LASSO 분석 결과를 HTML로 시각화

**입력**:
```python
def generate_lasso_section(
    results: List[ForecastResult]
) -> str:
```

**생성할 UI 요소**:
```
┌─────────────────────────────────────────────────────────────┐
│  📈 LASSO Fed Rate Forecast                                 │
├─────────────────────────────────────────────────────────────┤
│  Horizon      │ R²    │ Selected │ Top Variables           │
│  ─────────────┼───────┼──────────┼─────────────────────────│
│  VeryShort    │ 0.00  │ 1        │ d_Breakeven5Y           │
│  Short        │ 0.37  │ 7        │ d_HighYield_Rate, ...   │
│  Long         │ 0.64  │ 28       │ d_Baa_Yield, ...        │
├─────────────────────────────────────────────────────────────┤
│  [Bar Chart: Top 10 Coefficients - Long Horizon]           │
│  ████████████████████ d_Baa_Yield (+2.09)                  │
│  ██████████████████   d_Spread_Baa (-1.66)                 │
│  ████████████████     Ret_Dollar_Idx (+1.04)               │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

**구현 로직**:
```
1. 요약 테이블 생성
   table_html = "<table class='lasso-summary'>"
   table_html += "<tr><th>Horizon</th><th>R²</th><th>Selected</th><th>Top Variables</th></tr>"
   for result in results:
       top_vars = ', '.join(result.selected_variables[:3])
       table_html += f"""
       <tr>
           <td>{result.horizon}</td>
           <td>{result.r_squared:.2f}</td>
           <td>{len(result.selected_variables)}</td>
           <td>{top_vars}...</td>
       </tr>
       """
   table_html += "</table>"

2. 계수 바 차트 데이터 준비 (Chart.js용)
   long_result = results[2]  # Long horizon
   sorted_coefs = sorted(
       long_result.coefficients.items(),
       key=lambda x: abs(x[1]),
       reverse=True
   )[:10]

   chart_data = {
       'labels': [c[0] for c in sorted_coefs],
       'values': [c[1] for c in sorted_coefs],
       'colors': ['#22c55e' if v > 0 else '#ef4444' for _, v in sorted_coefs]
   }

3. Chart.js 스크립트 생성
   chart_script = f"""
   <script>
   new Chart(document.getElementById('lassoChart'), {{
       type: 'bar',
       data: {{
           labels: {chart_data['labels']},
           datasets: [{{
               data: {chart_data['values']},
               backgroundColor: {chart_data['colors']}
           }}]
       }},
       options: {{
           indexAxis: 'y',
           plugins: {{ legend: {{ display: false }} }}
       }}
   }});
   </script>
   """

4. 전체 섹션 조립 및 반환
```

---

### 8.5 스키마 확장

**파일 위치**: `eimas/core/schemas.py`

#### 8.5.1 추가할 데이터클래스

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime

@dataclass
class ForecastResult:
    """LASSO 예측 결과"""
    horizon: str                              # "VeryShort" / "Short" / "Long"
    selected_variables: List[str]             # LASSO가 선택한 변수 목록
    coefficients: Dict[str, float]            # {변수명: 표준화 계수}
    r_squared: float                          # 결정계수 (0~1)
    n_observations: int                       # 관측치 수
    lambda_optimal: float                     # 최적 정규화 파라미터
    hac_std_errors: Dict[str, float] = field(default_factory=dict)   # HAC 표준오차
    vif_scores: Dict[str, float] = field(default_factory=dict)       # VIF 점수
    predicted_change: Optional[float] = None  # 예측 금리 변화 (bp)
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class LASSODiagnostics:
    """LASSO 진단 정보"""
    total_candidate_vars: int                 # 후보 변수 총 수
    excluded_vars: List[str]                  # 제외된 변수 (Treasury 등)
    high_vif_warnings: List[str]              # VIF > 10인 변수
    convergence_info: Dict[str, bool]         # {horizon: 수렴여부}
    computation_time: float                   # 계산 시간 (초)

@dataclass
class DashboardConfig:
    """대시보드 생성 설정"""
    theme: str = 'dark'                       # 'dark' / 'light'
    language: str = 'ko'                      # 'ko' / 'en'
    include_crypto: bool = True
    include_regime: bool = True
    include_critical_path: bool = True
    include_lasso_results: bool = True
    include_agent_debate: bool = True
    include_risk_metrics: bool = True
    include_macro_indicators: bool = True
    chart_library: str = 'chartjs'            # 'chartjs' / 'plotly'
    max_signals_display: int = 30
    output_dir: str = 'outputs/dashboards'

@dataclass
class HorizonConfig:
    """Horizon 분류 설정"""
    very_short_max: int = 30                  # VeryShort: <= 30일
    short_min: int = 31                       # Short: 31일 이상
    short_max: int = 90                       # Short: 90일 이하
    long_min: int = 180                       # Long: 180일 이상
    # 참고: 91~179일은 분석에서 제외
```

---

### 8.6 테스트 케이스 명세

**파일 위치**: `eimas/tests/test_lasso_forecast.py`

#### 8.6.1 단위 테스트

```python
class TestLASSOForecaster:
    """LASSOForecaster 클래스 테스트"""

    def test_treasury_filter(self):
        """Treasury 변수가 정상적으로 제외되는지 확인"""
        # Given: Treasury 변수 포함된 DataFrame
        df = pd.DataFrame({
            'd_US10Y': [0.1, 0.2],
            'd_Baa_Yield': [0.3, 0.4],
            'Ret_SP500': [0.5, 0.6]
        })
        # When: 필터링 적용
        result = forecaster._filter_treasury_variables(df)
        # Then: Treasury 변수 제외, 나머지 유지
        assert 'd_US10Y' not in result.columns
        assert 'd_Baa_Yield' in result.columns

    def test_horizon_classification(self):
        """Horizon 분류가 정확한지 확인"""
        # VeryShort
        assert _classify_horizon(15) == 'VeryShort'
        assert _classify_horizon(30) == 'VeryShort'
        # Short
        assert _classify_horizon(31) == 'Short'
        assert _classify_horizon(90) == 'Short'
        # Excluded
        assert _classify_horizon(120) is None
        # Long
        assert _classify_horizon(180) == 'Long'
        assert _classify_horizon(365) == 'Long'

    def test_lasso_fit_returns_result(self):
        """LASSO fit이 LASSOResult를 반환하는지 확인"""
        # Given: 샘플 데이터
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randn(100))
        # When: fit 실행
        result = forecaster.fit(X, y, 'Long')
        # Then: LASSOResult 타입, 필수 필드 존재
        assert isinstance(result, LASSOResult)
        assert result.horizon == 'Long'
        assert 0 <= result.r_squared <= 1

    def test_empty_selection_very_short(self):
        """VeryShort horizon에서 변수 선택이 없어도 에러 없이 동작"""
        # Given: 노이즈 데이터 (설명력 없음)
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.randn(50))
        # When: VeryShort fit
        result = forecaster.fit(X, y, 'VeryShort')
        # Then: 빈 선택 허용
        assert result.selected_variables == [] or len(result.selected_variables) <= 1
```

#### 8.6.2 통합 테스트

```python
class TestForecastAgentIntegration:
    """ForecastAgent 통합 테스트"""

    @pytest.fixture
    def sample_market_data(self):
        """테스트용 시장 데이터"""
        return pd.DataFrame({
            'd_Exp_Rate': np.random.randn(500),
            'd_Baa_Yield': np.random.randn(500),
            'd_Spread_Baa': np.random.randn(500),
            'Ret_Dollar_Idx': np.random.randn(500),
            'd_Breakeven5Y': np.random.randn(500),
            'days_to_meeting': np.random.randint(1, 400, 500),
        })

    @pytest.mark.asyncio
    async def test_execute_returns_forecasts(self, sample_market_data):
        """_execute가 Horizon별 예측을 반환하는지 확인"""
        # Given
        agent = ForecastAgent(config)
        request = AgentRequest(context={'market_data': sample_market_data})
        # When
        response = await agent._execute(request)
        # Then
        assert response.status == 'success'
        assert 'forecasts' in response.result
        assert len(response.result['forecasts']) == 3  # VeryShort, Short, Long

    @pytest.mark.asyncio
    async def test_form_opinion_rate_direction(self, sample_market_data):
        """rate_direction 토픽에 대한 의견 형성"""
        # Given
        agent = ForecastAgent(config)
        context = {'forecasts': [...]}  # 미리 계산된 결과
        # When
        opinion = await agent.form_opinion('rate_direction', context)
        # Then
        assert opinion.topic == 'rate_direction'
        assert opinion.position in ['UP', 'DOWN', 'HOLD']
        assert 0 <= opinion.confidence <= 1
```

---

### 8.7 에러 처리 및 로깅

#### 8.7.1 예상 에러 및 처리 방법

| 에러 상황 | 에러 타입 | 처리 방법 |
|----------|----------|----------|
| 데이터 없음 | `ValueError` | 빈 결과 반환 + 경고 로그 |
| LASSO 수렴 실패 | `ConvergenceWarning` | max_iter 증가 후 재시도 |
| 메모리 부족 | `MemoryError` | 데이터 청킹 또는 변수 축소 |
| API 타임아웃 | `TimeoutError` | 지수 백오프 재시도 (최대 3회) |
| 잘못된 horizon | `KeyError` | 기본값 'Long' 사용 + 경고 |

#### 8.7.2 로깅 포맷

```python
import logging

logger = logging.getLogger('eimas.forecast')

# 정보 로그
logger.info(f"LASSO fit completed: horizon={horizon}, R²={r_squared:.4f}, selected={n_selected}")

# 경고 로그
logger.warning(f"High VIF detected: {high_vif_vars}")

# 에러 로그
logger.error(f"LASSO convergence failed after {max_retries} attempts", exc_info=True)
```

---

---

## 9. LLM 코드 생성용 프롬프트

> **사용법**: 아래 프롬프트를 Cursor, Gemini, Claude 등에 복사하여 사용하세요.
> 각 프롬프트는 독립적으로 실행 가능하며, 순서대로 진행하는 것을 권장합니다.

---

### 9.1 LASSOForecaster 클래스 생성

```
당신은 Python 금융 분석 전문가입니다. 다음 요구사항에 따라 LASSOForecaster 클래스를 구현하세요.

## 파일 위치
`eimas/lib/lasso_model.py`

## 요구사항

1. **목적**: LASSO (L1 정규화) 기반 Fed 금리 예측 모델 래퍼 클래스

2. **의존성**:
   - sklearn.linear_model.LassoCV
   - sklearn.preprocessing.StandardScaler
   - sklearn.model_selection.TimeSeriesSplit
   - statsmodels.api (HAC 표준오차용)
   - statsmodels.stats.outliers_influence.variance_inflation_factor

3. **클래스 구조**:
```python
class LASSOForecaster:
    def __init__(self, config: LASSOConfig)
    def fit(self, X: pd.DataFrame, y: pd.Series, horizon: str) -> LASSOResult
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_selected_variables(self) -> List[str]
    def get_coefficients(self) -> Dict[str, float]
    def compute_hac_standard_errors(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]
    def compute_vif_scores(self, X: pd.DataFrame) -> Dict[str, float]
    def _filter_treasury_variables(self, df: pd.DataFrame) -> pd.DataFrame
```

4. **Treasury 제외 변수** (Simultaneity bias 방지):
   - d_US10Y, d_US2Y, d_RealYield10Y, d_Term_Spread
   - Ret_Treasury_7_10Y, Ret_Treasury_1_3Y, Ret_Treasury_20Y

5. **LASSOConfig 데이터클래스**:
   - n_splits: int = 5 (TimeSeriesSplit)
   - max_iter: int = 10000
   - tol: float = 1e-4
   - hac_lag: int = 5 (Newey-West lag)

6. **LASSOResult 데이터클래스**:
   - horizon: str
   - lambda_optimal: float
   - selected_variables: List[str]
   - coefficients: Dict[str, float]
   - r_squared: float
   - n_observations: int
   - n_selected: int

7. **에러 처리**:
   - 빈 DataFrame → ValueError("Empty feature matrix")
   - 수렴 실패 → max_iter 증가 후 재시도 (최대 3회)
   - 선택 변수 0개 → 경고 로그 후 빈 결과 반환

8. **로깅**: logging.getLogger('eimas.lasso') 사용

## 참고
- HAC 표준오차: statsmodels OLS의 cov_type='HAC', cov_kwds={'maxlags': 5}
- VIF > 10이면 다중공선성 경고

코드만 출력하세요. 설명은 주석으로 포함하세요.
```

---

### 9.2 ForecastAgent 클래스 생성

```
당신은 Python 멀티에이전트 시스템 전문가입니다. 다음 요구사항에 따라 ForecastAgent 클래스를 구현하세요.

## 파일 위치
`eimas/agents/forecast_agent.py`

## 컨텍스트
- 기존 BaseAgent 클래스를 상속 (eimas/agents/base_agent.py)
- LASSOForecaster 사용 (eimas/lib/lasso_model.py)
- 스키마는 eimas/core/schemas.py 참조

## 요구사항

1. **목적**: LASSO 기반 Fed 금리 예측을 수행하고 멀티에이전트 토론에 참여

2. **클래스 구조**:
```python
class ForecastAgent(BaseAgent):
    def __init__(self, config: AgentConfig)
    async def _execute(self, request: AgentRequest) -> AgentResponse
    async def form_opinion(self, topic: str, context: Dict) -> AgentOpinion
    def _prepare_features(self, market_data: Dict) -> pd.DataFrame
    def _classify_horizon(self, days_to_meeting: int) -> Optional[str]
    def _interpret_coefficients(self, results: List[LASSOResult]) -> str
```

3. **Horizon 분류**:
   - VeryShort: days_to_meeting <= 30
   - Short: 31 <= days_to_meeting <= 90
   - Long: days_to_meeting >= 180
   - 91~179일: None 반환 (분석 제외)

4. **_execute 로직**:
   a. market_data에서 X, y 추출
   b. 각 horizon별 LASSO 학습
   c. HAC 표준오차, VIF 계산
   d. ForecastResult 리스트 생성
   e. AgentResponse 반환

5. **form_opinion 지원 토픽**:
   | 토픽 | 출력 형식 |
   |------|----------|
   | rate_direction | "UP" / "DOWN" / "HOLD" |
   | rate_magnitude | float (bp 단위) |
   | forecast_confidence | float (0~1) |
   | key_drivers | List[str] |

6. **rate_direction 결정 로직**:
   - Long horizon 결과의 핵심 변수 계수 분석
   - d_Spread_Baa 음(-): 스프레드 확대 → 인하 기대 감소
   - Ret_Dollar_Idx 양(+): 달러 강세 → 인하 기대 감소
   - 가중 신호 > threshold → "UP"
   - 가중 신호 < -threshold → "DOWN"
   - else → "HOLD"

7. **AgentOpinion 필드**:
   - agent_id: 'forecast_agent'
   - topic: str
   - position: str
   - confidence: float (R² 기반, 최대 0.95)
   - reasoning: str (자연어 설명)
   - evidence: List[str]
   - caveats: List[str]

## BaseAgent 인터페이스 참고
```python
class BaseAgent(ABC):
    @abstractmethod
    async def _execute(self, request: AgentRequest) -> AgentResponse: ...
    @abstractmethod
    async def form_opinion(self, topic: str, context: Dict) -> AgentOpinion: ...
```

코드만 출력하세요.
```

---

### 9.3 VisualizationAgent 클래스 생성

```
당신은 Python 데이터 시각화 전문가입니다. 다음 요구사항에 따라 VisualizationAgent 클래스를 구현하세요.

## 파일 위치
`eimas/agents/visualization_agent.py`

## 요구사항

1. **목적**: 분석 결과를 HTML 대시보드로 시각화하는 에이전트

2. **클래스 구조**:
```python
class VisualizationAgent(BaseAgent):
    def __init__(self, config: AgentConfig, dashboard_config: DashboardConfig)
    async def _execute(self, request: AgentRequest) -> AgentResponse
    def _collect_agent_results(self, context: Dict) -> Dict
    def _generate_output_path(self) -> str
    def _save_dashboard(self, html: str, path: str) -> None
```

3. **DashboardConfig 필드**:
   - theme: str = 'dark'
   - language: str = 'ko'
   - include_crypto: bool = True
   - include_regime: bool = True
   - include_critical_path: bool = True
   - include_lasso_results: bool = True
   - include_agent_debate: bool = True
   - chart_library: str = 'chartjs'
   - output_dir: str = 'outputs/dashboards'

4. **request.context 입력 형식**:
```python
{
    'signals': List[Dict],
    'regime_data': Dict,
    'risk_metrics': Dict,
    'macro_indicators': Dict,
    'agent_opinions': List[AgentOpinion],
    'consensus': Consensus,
    'conflicts': List[Conflict],
    'forecast_results': List[ForecastResult],
    'timestamp': str,
    'project_id': str,
}
```

5. **_execute 로직**:
   a. dashboard_generator.generate_dashboard() 호출
   b. config에 따라 추가 섹션 삽입
   c. HTML 파일 저장
   d. AgentResponse 반환

6. **출력 경로 형식**: `{output_dir}/dashboard_{timestamp}_{project_id}.html`

7. **AgentResponse 결과**:
```python
{
    'dashboard_path': str,
    'dashboard_size': int,
    'sections_generated': List[str]
}
```

코드만 출력하세요.
```

---

### 9.4 대시보드 섹션 함수 추가

```
당신은 JavaScript/HTML 시각화 전문가입니다. 기존 dashboard_generator.py에 다음 두 함수를 추가하세요.

## 파일 위치
`eimas/lib/dashboard_generator.py` (기존 파일에 추가)

## 함수 1: generate_multi_agent_section

```python
def generate_multi_agent_section(
    opinions: List[AgentOpinion],
    consensus: Consensus,
    conflicts: List[Conflict]
) -> str:
    """멀티에이전트 토론 결과를 HTML로 시각화"""
```

### UI 레이아웃:
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Multi-Agent Analysis                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AnalysisAgent│  │ForecastAgent │  │StrategyAgent│       │
│  │   BEARISH    │  │    HOLD      │  │   CAUTIOUS  │       │
│  │  conf: 0.75  │  │  conf: 0.68  │  │  conf: 0.72 │       │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  📊 Consensus: CAUTIOUS HOLD (Agreement: 78%)              │
│  ⚠️  Conflicts: rate_magnitude (ForecastAgent vs Strategy) │
└─────────────────────────────────────────────────────────────┘
```

### 스타일 요구사항:
- 다크 테마 (배경: #1a1a2e, 카드: #16213e)
- 포지션별 색상: UP=#22c55e, DOWN=#ef4444, HOLD=#f59e0b
- 에이전트 카드: flexbox 가로 배치
- 한글 지원

---

## 함수 2: generate_lasso_section

```python
def generate_lasso_section(
    results: List[ForecastResult]
) -> str:
    """LASSO 분석 결과를 HTML로 시각화"""
```

### UI 레이아웃:
```
┌─────────────────────────────────────────────────────────────┐
│  📈 LASSO Fed Rate Forecast                                 │
├─────────────────────────────────────────────────────────────┤
│  Horizon      │ R²    │ Selected │ Top Variables           │
│  ─────────────┼───────┼──────────┼─────────────────────────│
│  VeryShort    │ 0.00  │ 1        │ d_Breakeven5Y           │
│  Short        │ 0.37  │ 7        │ d_HighYield_Rate, ...   │
│  Long         │ 0.64  │ 28       │ d_Baa_Yield, ...        │
├─────────────────────────────────────────────────────────────┤
│  [Horizontal Bar Chart: Top 10 Coefficients]               │
│  ████████████████████ d_Baa_Yield (+2.09)                  │
│  ██████████████████   d_Spread_Baa (-1.66)                 │
└─────────────────────────────────────────────────────────────┘
```

### Chart.js 요구사항:
- 수평 막대 차트 (indexAxis: 'y')
- 양수 계수: #22c55e, 음수 계수: #ef4444
- Long horizon 상위 10개 변수 표시
- 범례 숨김

코드만 출력하세요.
```

---

### 9.5 스키마 확장

```
다음 데이터클래스들을 eimas/core/schemas.py에 추가하세요.

## 추가할 클래스

```python
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class ForecastResult:
    """LASSO 예측 결과"""
    horizon: str                              # "VeryShort" / "Short" / "Long"
    selected_variables: List[str]
    coefficients: Dict[str, float]
    r_squared: float
    n_observations: int
    lambda_optimal: float
    hac_std_errors: Dict[str, float] = field(default_factory=dict)
    vif_scores: Dict[str, float] = field(default_factory=dict)
    predicted_change: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class LASSODiagnostics:
    """LASSO 진단 정보"""
    total_candidate_vars: int
    excluded_vars: List[str]
    high_vif_warnings: List[str]
    convergence_info: Dict[str, bool]
    computation_time: float

@dataclass
class DashboardConfig:
    """대시보드 생성 설정"""
    theme: str = 'dark'
    language: str = 'ko'
    include_crypto: bool = True
    include_regime: bool = True
    include_critical_path: bool = True
    include_lasso_results: bool = True
    include_agent_debate: bool = True
    include_risk_metrics: bool = True
    include_macro_indicators: bool = True
    chart_library: str = 'chartjs'
    max_signals_display: int = 30
    output_dir: str = 'outputs/dashboards'

@dataclass
class HorizonConfig:
    """Horizon 분류 설정"""
    very_short_max: int = 30
    short_min: int = 31
    short_max: int = 90
    long_min: int = 180
```

기존 schemas.py 파일 구조를 유지하면서 위 클래스들을 추가하세요.
```

---

### 9.6 통합 테스트 생성

```
eimas/tests/test_lasso_forecast.py에 pytest 기반 테스트를 작성하세요.

## 테스트 케이스

### 1. TestLASSOForecaster (단위 테스트)
- test_treasury_filter: Treasury 변수 제외 확인
- test_horizon_classification: Horizon 분류 정확성
- test_lasso_fit_returns_result: fit 결과 타입 확인
- test_empty_selection_very_short: VeryShort에서 빈 선택 허용
- test_vif_calculation: VIF 계산 정상 동작
- test_hac_standard_errors: HAC 표준오차 계산

### 2. TestForecastAgentIntegration (통합 테스트)
- test_execute_returns_forecasts: 3개 horizon 예측 반환
- test_form_opinion_rate_direction: rate_direction 의견 형성
- test_form_opinion_confidence_bounds: confidence 범위 (0~1)

### 3. Fixtures
- sample_market_data: 500행 테스트 데이터
- forecast_agent: 설정된 ForecastAgent 인스턴스

### 요구사항
- pytest.mark.asyncio 사용
- numpy.random.seed(42) 고정
- 모든 assertion에 명확한 메시지 포함

코드만 출력하세요.
```

---

### 9.7 전체 파이프라인 통합 (main.py 수정)

```
eimas/main.py를 수정하여 전체 파이프라인을 통합하세요.

## 추가할 함수

```python
async def run_full_pipeline(config_path: str = 'configs/default.yaml') -> str:
    """
    전체 EIMAS 파이프라인 실행

    1. DataManager로 데이터 수집
    2. AnalysisAgent로 Critical Path 분석
    3. ForecastAgent로 LASSO 예측
    4. MetaOrchestrator로 토론 및 합의
    5. VisualizationAgent로 대시보드 생성

    Returns:
        생성된 대시보드 파일 경로
    """
```

## 실행 흐름

1. 설정 로드 (YAML)
2. DataManager 초기화 및 데이터 수집
3. 에이전트 초기화:
   - AnalysisAgent
   - ForecastAgent
   - VisualizationAgent
4. MetaOrchestrator로 워크플로우 실행
   - auto_detect_topics()
   - run_with_debate()
5. 대시보드 생성
6. 경로 반환

## CLI 인터페이스

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output-dir', default='outputs/dashboards')
    args = parser.parse_args()

    result = asyncio.run(run_full_pipeline(args.config))
    print(f"Dashboard generated: {result}")
```

기존 main.py 구조를 유지하면서 위 기능을 추가하세요.
```

---

## 10. 구현 현황 및 미완료 작업

> **마지막 업데이트**: 2025-12-25

### 10.1 구현 완료 ✅

| 항목 | 파일 | 상태 | 비고 |
|------|------|------|------|
| LASSOForecaster 클래스 | `lib/lasso_model.py` | ✅ 완료 | LASSO 학습, HAC, VIF 계산 |
| ForecastAgent | `agents/forecast_agent.py` | ✅ 완료 | 3개 horizon 예측 |
| VisualizationAgent | `agents/visualization_agent.py` | ✅ 완료 | 기본 대시보드 생성 |
| UnifiedDataCollector | `lib/data_collector.py` | ✅ 완료 | Yahoo + FRED → Ret_*, d_* 변환 |
| Main Pipeline | `main.py` | ✅ 완료 | 전체 파이프라인 통합 |
| CME 패널 로드 | `main.py` | ✅ 완료 | d_Exp_Rate, days_to_meeting |
| 기본 스키마 | `core/schemas.py` | ✅ 완료 | ForecastResult, DashboardConfig 등 |

### 10.2 미완료 작업 ❌

#### 10.2.1 대시보드 고급 기능 (우선순위: 높음)

| 항목 | 설명 | 예상 작업량 |
|------|------|------------|
| `generate_multi_agent_section()` | 멀티에이전트 토론 결과 시각화 (의견/합의/충돌) | 중간 |
| `generate_lasso_section()` | LASSO 계수 막대 차트, 변수 선택 테이블 | 중간 |
| Spillover 분석 섹션 | 자산간 전이 효과 히트맵 | 높음 |
| Markov 레짐 섹션 | 레짐 전환 확률 차트 | 높음 |
| LLM 요약 섹션 | Claude API 기반 AI 해석 | 중간 |

**프롬프트 (generate_multi_agent_section)**:
```
다음 함수를 lib/dashboard_generator.py에 추가하세요.

def generate_multi_agent_section(
    opinions: List[AgentOpinion],
    consensus: Optional[Consensus],
    conflicts: List[Conflict]
) -> str:
    """
    멀티에이전트 토론 결과를 HTML로 시각화

    UI 레이아웃:
    - 에이전트별 의견 카드 (flexbox 가로 배치)
    - 각 카드에 agent_id, position, confidence 표시
    - 포지션별 색상: UP=#22c55e, DOWN=#ef4444, HOLD=#f59e0b
    - 합의 상태 표시 바
    - 충돌 목록 (있는 경우)

    다크 테마 (배경: #1a1a2e, 카드: #16213e)
    """
```

**프롬프트 (generate_lasso_section)**:
```
다음 함수를 lib/dashboard_generator.py에 추가하세요.

def generate_lasso_section(results: List[ForecastResult]) -> str:
    """
    LASSO 분석 결과를 HTML로 시각화

    UI 레이아웃:
    - Horizon별 요약 테이블 (R², 선택 변수 수, lambda)
    - 수평 막대 차트 (Top 10 계수)
      - Chart.js, indexAxis: 'y'
      - 양수: #22c55e, 음수: #ef4444
    - Long horizon 핵심 변수 강조

    다크 테마
    """
```

---

#### 10.2.2 추가 에이전트 (우선순위: 중간)

| 에이전트 | 역할 | 필요 API |
|----------|------|----------|
| ResearchAgent | 실시간 뉴스/분석 수집 | Perplexity API |
| StrategyAgent | 투자 전략 권고 | Claude API |

**프롬프트 (ResearchAgent)**:
```
agents/research_agent.py에 ResearchAgent 클래스를 구현하세요.

class ResearchAgent(BaseAgent):
    """Perplexity API 기반 실시간 리서치 에이전트"""

    def __init__(self, config: AgentConfig):
        # PERPLEXITY_API_KEY 환경변수 사용

    async def _execute(self, request: AgentRequest) -> AgentResponse:
        # 1. request.context에서 검색 쿼리 추출
        # 2. Perplexity API 호출 (sonar-medium-online 모델)
        # 3. 관련 뉴스/분석 요약 반환

    async def form_opinion(self, topic: str, context: Dict) -> AgentOpinion:
        # 토픽: market_sentiment, breaking_news, analyst_consensus
```

---

#### 10.2.3 대시보드 Generator 이동 (우선순위: 낮음)

`plus/dashboard_generator.py` (154KB, ~1800줄)의 고급 기능을 `lib/dashboard_generator.py`로 통합:

| 함수 | 현재 상태 | 필요 작업 |
|------|----------|----------|
| `generate_asset_risk_section()` | plus/에 존재 | lib/로 이동 |
| `generate_regime_display()` | plus/에 존재 | lib/로 이동 |
| `generate_crypto_panel_html()` | plus/에 존재 | lib/로 이동 |
| `_generate_spillover_section()` | plus/에 존재 | lib/로 이동 |
| `_generate_markov_regime_section()` | plus/에 존재 | lib/로 이동 |
| `_generate_risk_metrics_section()` | plus/에 존재 | lib/로 이동 |
| `_generate_macro_environment_section()` | plus/에 존재 | lib/로 이동 |
| `_generate_llm_summary_section()` | plus/에 존재 | lib/로 이동 |

**작업 방법**:
1. `plus/dashboard_generator.py` 전체를 `lib/dashboard_full.py`로 복사
2. EIMAS 스키마와 호환되도록 입력 파라미터 조정
3. `VisualizationAgent`에서 선택적으로 사용

---

#### 10.2.4 테스트 및 품질 (우선순위: 중간)

| 항목 | 파일 | 설명 |
|------|------|------|
| 단위 테스트 | `tests/test_lasso_model.py` | LASSOForecaster 테스트 |
| 통합 테스트 | `tests/test_pipeline.py` | 전체 파이프라인 E2E |
| 성능 벤치마크 | `tests/benchmark.py` | 실행 시간 측정 |

---

### 10.3 다음 단계 권장 순서

```
1. generate_lasso_section() 구현
   → LASSO 결과가 대시보드에 시각화됨

2. generate_multi_agent_section() 구현
   → 에이전트 토론 결과가 시각화됨

3. plus/dashboard_generator.py → lib/ 이동
   → 고급 대시보드 기능 활성화

4. ResearchAgent 구현
   → 실시간 뉴스/분석 통합

5. StrategyAgent 구현
   → 투자 전략 권고 추가

6. 통합 테스트 작성
   → 안정성 확보
```

---

### 10.4 빠른 테스트 명령어

```bash
# 전체 파이프라인 실행
cd /home/tj/projects/autoai/eimas
python main.py

# LASSO 결과 확인
python -c "
import json
with open('outputs/dashboards/report_*.json') as f:
    data = json.load(f)
for r in data['forecast_results']:
    print(f\"{r['horizon']}: R²={r['r_squared']:.4f}, n={r['n_observations']}\")
"

# 데이터 수집 테스트
python -c "
from lib.data_collector import UnifiedDataCollector
collector = UnifiedDataCollector(start_date='2024-09-01')
df = collector.collect_all()
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
"
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-25 | 초기 가이드라인 작성 |
| 2025-12-25 | 상세 구현 명세서 추가 (섹션 8) |
| 2025-12-25 | LLM 코드 생성용 프롬프트 추가 (섹션 9) |
| 2025-12-25 | 구현 현황 및 미완료 작업 추가 (섹션 10) |
