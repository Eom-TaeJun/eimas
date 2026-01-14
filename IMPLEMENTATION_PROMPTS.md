# EIMAS v2.0 구현 프롬프트 가이드

> 각 프롬프트를 순서대로 실행하여 main2.py를 구현합니다.
> AI 코딩 도구 (Claude, Cursor, Gemini)에 복사-붙여넣기하여 사용하세요.

---

## Phase 1: 데이터 기반

### Prompt 1.1: UnifiedDataCollectorV2 구현

```
TO DO:
eimas/lib/data_collector_v2.py에 UnifiedDataCollectorV2 클래스를 구현하세요.

CONTEXT:
- 기존 파일: eimas/lib/data_collector.py의 UnifiedDataCollector 참고
- 개선점: 이벤트 태깅, 캐싱, 비동기 수집

REQUIREMENTS:
1. 클래스 구조:
   - __init__(mode='batch', event_tagging=True, cache_dir='data/cache')
   - async collect() -> pd.DataFrame
   - _fetch_yahoo_async(), _fetch_fred_async()
   - _transform_variables(): Ret_*, d_* 변환
   - _tag_events(): 이벤트 레짐 태깅
   - _validate_quality(): 결측치, 이상치 검증

2. 수집 대상 (Yahoo Finance):
   - 주가: SPY, QQQ, IWM, DIA
   - 섹터: XLF, XLE, XLK, XLV, XLC, XLI
   - 원자재: GLD, SLV, GDX, USO, UNG, DBA
   - 채권: TLT, IEF, LQD, HYG, TIP
   - 환율: UUP (달러), FXY (엔), FXE (유로)
   - 암호화폐: BTC-USD, ETH-USD
   - 변동성: ^VIX

3. 수집 대상 (FRED):
   - DGS10, DGS2 (국채 금리)
   - BAA10Y, AAA10Y (신용 스프레드)
   - T5YIE, T10YIE (인플레이션 기대)
   - DTWEXBGS (달러 인덱스)

4. 변환 규칙:
   - 주가/원자재/암호화폐 → Ret_{ticker} = log return * 100
   - 금리/스프레드 → d_{name} = diff
   - 파생변수: Term_Spread = DGS10 - DGS2
   - 파생변수: Spread_Baa = BAA10Y (이미 스프레드)
   - 파생변수: Spread_HighYield = HYG 스프레드

5. 캐싱:
   - pickle 형식으로 data/cache/{date}.pkl 저장
   - 당일 캐시 있으면 재사용
   - force_refresh 옵션

OUTPUT:
- eimas/lib/data_collector_v2.py 파일
- 테스트 코드 포함 (if __name__ == "__main__")
```

---

### Prompt 1.2: EventRegistry 구현

```
TO DO:
eimas/lib/event_registry.py에 이벤트 레지스트리 시스템을 구현하세요.

CONTEXT:
- 목적: 구조변화 이벤트를 관리하고 데이터에 태깅
- 용도: 이벤트 전후 데이터를 분리하여 다른 모델로 학습

REQUIREMENTS:
1. Event 데이터 클래스:
   @dataclass
   class Event:
       name: str
       start: datetime
       end: datetime
       description: str
       impact: Dict[str, str]  # {'rate': 'dovish', 'equity': 'bullish'}
       category: str  # 'monetary', 'fiscal', 'geopolitical', 'corporate'

2. EventRegistry 클래스:
   - __init__(): 기본 이벤트 로드
   - add_event(event: Event)
   - get_events_in_range(start, end) -> List[Event]
   - get_regime(date) -> str  # 'normal' 또는 이벤트명
   - load_from_yaml(path: str)
   - save_to_yaml(path: str)

3. 기본 등록 이벤트:
   - fed_pivot_2024: 2024-09-01 ~ 2024-10-31, Fed 금리 인하 시작
   - trump_election_2024: 2024-11-05 ~ 2024-12-31, 트럼프 당선
   - covid_crisis: 2020-02-20 ~ 2020-06-30, 코로나 위기
   - inflation_surge_2022: 2022-01-01 ~ 2022-12-31, 인플레이션 급등
   - svb_crisis_2023: 2023-03-08 ~ 2023-03-31, SVB 은행 위기

4. YAML 형식:
   events:
     - name: fed_pivot_2024
       start: 2024-09-01
       end: 2024-10-31
       description: Fed 금리 인하 시작
       impact:
         rate: dovish
         equity: bullish
       category: monetary

OUTPUT:
- eimas/lib/event_registry.py
- eimas/configs/events.yaml (기본 이벤트)
```

---

## Phase 2: 네트워크 분석

### Prompt 2.1: EconomicNetworkBuilder 구현

```
TO DO:
eimas/lib/network_builder.py에 경제 네트워크 분석 클래스를 구현하세요.

CONTEXT:
- Palantir Ontology 개념 적용
- Node = 경제 변수, Edge = 인과 관계
- Granger Causality + VAR 기반

REQUIREMENTS:
1. 클래스 구조:
   class EconomicNetworkBuilder:
       def __init__(self, significance=0.05, max_lags=5):
       def build(self, data: pd.DataFrame) -> EconomicNetwork:
       def calculate_granger_matrix(self, data) -> np.ndarray:
       def fit_var(self, data) -> VARResults:
       def calculate_irf(self, shock_var, periods=20) -> Dict:

2. EconomicNetwork 클래스:
   class EconomicNetwork:
       nodes: Dict[str, NodeInfo]  # {name: {type, category}}
       edges: List[EdgeInfo]  # {source, target, weight, p_value}

       def add_node(self, name, type, category)
       def add_edge(self, source, target, weight, p_value)
       def get_edge_weight(self, source, target) -> float
       def get_significant_paths(self, target) -> List[Path]
       def to_dict() -> Dict
       def to_networkx() -> nx.DiGraph

3. 변수 분류:
   - monetary: d_US10Y, d_US2Y, d_Exp_Rate, Term_Spread
   - credit: d_Spread_Baa, d_Spread_HighYield
   - equity: Ret_SPY, Ret_QQQ, Ret_IWM
   - commodity: Ret_GLD, Ret_SLV, Ret_USO
   - volatility: d_VIX
   - currency: Ret_UUP, d_Dollar_Idx

4. Granger Causality:
   - statsmodels.tsa.stattools.grangercausalitytests 사용
   - 양방향 테스트 (X→Y, Y→X)
   - p-value < significance면 edge 추가

5. VAR 모델:
   - statsmodels.tsa.api.VAR 사용
   - AIC로 최적 lag 선택 (max_lags 이내)
   - IRF 계산: var_model.irf(periods)

OUTPUT:
- eimas/lib/network_builder.py
- 의존성: statsmodels, networkx
```

---

### Prompt 2.2: IRF 시각화 함수

```
TO DO:
eimas/lib/irf_visualizer.py에 충격반응함수 시각화를 구현하세요.

CONTEXT:
- VAR 모델의 IRF 결과를 시각화
- 경제학적 해석 지원

REQUIREMENTS:
1. 함수:
   def plot_irf(
       irf_result: Dict,
       shock_var: str,
       response_vars: List[str],
       periods: int = 20,
       confidence: float = 0.95
   ) -> plt.Figure:

2. 출력 형식:
   - 서브플롯: 각 response_var별 IRF
   - X축: 기간 (0, 1, 2, ..., periods)
   - Y축: 충격 반응 크기
   - 신뢰구간 음영 표시
   - 0선 점선 표시

3. 추가 함수:
   def plot_cumulative_irf(...):  # 누적 IRF
   def plot_irf_comparison(...):  # 여러 shock 비교
   def export_irf_html(...):  # 대시보드용 HTML

4. 경제학적 주석:
   - 장기 수렴 여부 표시
   - 유의미한 반응 기간 하이라이트
   - M↔P 관계 특별 표시

OUTPUT:
- eimas/lib/irf_visualizer.py
- 의존성: matplotlib, plotly (HTML용)
```

---

## Phase 3: 경제학파 에이전트

### Prompt 3.1: BaseEconomicAgent 추상 클래스

```
TO DO:
eimas/agents/economic_agents/base.py에 경제학파 에이전트 기반 클래스를 구현하세요.

CONTEXT:
- 기존: eimas/agents/base_agent.py
- 확장: 경제학 프레임워크, IRF 해석, 증거 기반 의견

REQUIREMENTS:
1. 클래스 구조:
   class BaseEconomicAgent(ABC):
       school: str  # 'monetarist', 'keynesian', 'austrian'
       framework: str  # 경제학 방정식/이론
       key_variables: List[str]  # 주시 변수

       @abstractmethod
       async def analyze(self, data, network, irf) -> EconomicOpinion

       def interpret_irf(self, irf, var) -> str  # IRF 해석
       def calculate_evidence_score(self, data) -> float
       def get_framework_equation(self) -> str

2. EconomicOpinion 스키마:
   @dataclass
   class EconomicOpinion:
       agent_school: str
       topic: str
       position: str  # BULLISH, BEARISH, NEUTRAL, WARNING
       confidence: float
       evidence: List[Dict]  # {'metric': ..., 'value': ..., 'interpretation': ...}
       framework_reference: str
       irf_interpretation: Optional[str]
       dissent_from_consensus: Optional[str]

3. 공통 메서드:
   def _analyze_trend(self, data, var, window=20) -> float
   def _detect_regime_change(self, data, var) -> bool
   def _compare_to_historical(self, data, var) -> str

OUTPUT:
- eimas/agents/economic_agents/__init__.py
- eimas/agents/economic_agents/base.py
```

---

### Prompt 3.2: MonetaristAgent 구현

```
TO DO:
eimas/agents/economic_agents/monetarist.py를 구현하세요.

CONTEXT:
- 통화주의 관점: M↔P 장기 관계, 통화 중립성
- 핵심 인물: Milton Friedman
- 방정식: MV = PY

REQUIREMENTS:
1. 클래스:
   class MonetaristAgent(BaseEconomicAgent):
       school = 'monetarist'
       framework = 'MV = PY (화폐수량설)'
       key_variables = ['M2', 'CPI', 'd_Exp_Rate', 'Velocity']

2. 분석 로직:
   async def analyze(self, data, network, irf):
       # 1. M2 증가율 계산
       m2_growth = ...

       # 2. CPI와의 상관관계 확인
       m_p_correlation = network.get_edge_weight('M2', 'CPI')

       # 3. IRF에서 통화 충격 효과
       monetary_irf = irf['responses'].get('CPI', [])
       long_run_effect = monetary_irf[-1] if monetary_irf else 0

       # 4. 의견 형성
       if m2_growth > 0.05 and m_p_correlation > 0.3:
           position = "INFLATIONARY"
           reasoning = "통화량 증가 → 물가 상승 예상"
       elif long_run_effect < 0.01:
           position = "NEUTRAL"
           reasoning = "통화 중립성 작동, 실질 효과 제한적"

       return EconomicOpinion(...)

3. 핵심 지표:
   - M2 증가율 (연율화)
   - M2-CPI 상관관계 (12개월 rolling)
   - 화폐 유통속도 (V = PY/M)
   - 장기 IRF 수렴값

4. 증거 형식:
   evidence = [
       {'metric': 'M2 YoY Growth', 'value': '6.2%', 'interpretation': '확장적'},
       {'metric': 'M2-CPI Correlation', 'value': '0.45', 'interpretation': '연결 유지'},
       {'metric': 'Long-run IRF', 'value': '0.82', 'interpretation': '통화 → 물가 전이'}
   ]

OUTPUT:
- eimas/agents/economic_agents/monetarist.py
```

---

### Prompt 3.3: KeynesianAgent 구현

```
TO DO:
eimas/agents/economic_agents/keynesian.py를 구현하세요.

CONTEXT:
- 케인즈주의: 총수요 관리, 승수효과
- 핵심 인물: John Maynard Keynes
- 방정식: Y = C + I + G + NX

REQUIREMENTS:
1. 클래스:
   class KeynesianAgent(BaseEconomicAgent):
       school = 'keynesian'
       framework = 'Y = C + I + G + NX (총수요)'
       key_variables = ['Ret_Consumer', 'Ret_Industrial', 'Gov_Spending', 'NX']

2. 분석 로직:
   async def analyze(self, data, network, irf):
       # 1. 소비 동향 (Consumer Discretionary)
       consumption = self._analyze_trend(data, 'Ret_XLY')

       # 2. 투자 동향 (Industrial)
       investment = self._analyze_trend(data, 'Ret_XLI')

       # 3. 산출 갭 추정
       output_gap = self._estimate_output_gap(data)

       # 4. 승수 효과 (네트워크에서)
       multiplier = 1 / (1 - network.get_edge_weight('C', 'Y'))

       # 5. 의견 형성
       if output_gap < -2:
           position = "STIMULATE"
           reasoning = "산출 갭 음수, 재정/통화 확대 필요"
       elif output_gap > 2:
           position = "TIGHTEN"
           reasoning = "과열 징후, 긴축 필요"

3. 핵심 지표:
   - 소비 모멘텀 (XLY 수익률)
   - 투자 모멘텀 (XLI 수익률)
   - 산출 갭 추정치
   - 재정 승수 (1/(1-MPC))

4. 유동성 함정 감지:
   def _detect_liquidity_trap(self, data):
       # 금리 0 근접 + 소비/투자 부진
       rate_near_zero = data['d_Exp_Rate'].iloc[-1] < 0.5
       demand_weak = consumption < 0 and investment < 0
       return rate_near_zero and demand_weak

OUTPUT:
- eimas/agents/economic_agents/keynesian.py
```

---

### Prompt 3.4: AustrianAgent 구현

```
TO DO:
eimas/agents/economic_agents/austrian.py를 구현하세요.

CONTEXT:
- 오스트리아 학파: 경기 사이클, 신용 팽창, 버블
- 핵심 인물: Mises, Hayek
- 이론: Austrian Business Cycle Theory (ABCT)

REQUIREMENTS:
1. 클래스:
   class AustrianAgent(BaseEconomicAgent):
       school = 'austrian'
       framework = 'Austrian Business Cycle Theory'
       key_variables = ['Credit', 'Ret_GLD', 'Ret_SLV', 'd_VIX', 'Real_Rate']

2. 분석 로직:
   async def analyze(self, data, network, irf):
       # 1. 신용 사이클 분석
       credit_expansion = self._analyze_credit_cycle(data)

       # 2. 버블 징후 탐지
       bubble_score = self._detect_bubble_signs(data)

       # 3. 금/은 가격 (실물 자산 선호)
       precious_metals = self._precious_metals_signal(data)

       # 4. 실질 금리 (저금리 = 자본 오배분)
       real_rate = self._calculate_real_rate(data)

       # 5. 의견 형성
       if bubble_score > 0.7:
           position = "BUBBLE_WARNING"
       elif precious_metals > 0.1:
           position = "INFLATION_HEDGE"
       elif real_rate < -1:
           position = "MALINVESTMENT_RISK"

3. 버블 탐지:
   def _detect_bubble_signs(self, data) -> Dict:
       indicators = {
           'price_earnings': ...,  # P/E 비율
           'credit_gdp': ...,      # 신용/GDP
           'volatility_suppression': ...,  # 낮은 VIX
           'speculation_index': ...,  # 투기 지수
       }
       score = weighted_average(indicators)
       return {'score': score, 'confidence': ...}

4. 핵심 지표:
   - 신용 증가율 vs GDP 증가율
   - 금/은/구리/백금 가격 추세
   - 실질 금리 (명목 - 인플레이션 기대)
   - Shiller P/E (CAPE)

OUTPUT:
- eimas/agents/economic_agents/austrian.py
```

---

### Prompt 3.5: TechnicalAgent (시그널 포착)

```
TO DO:
eimas/agents/economic_agents/technical.py를 구현하세요.

CONTEXT:
- 목적: 시장 시그널 포착 (Net Buy, 체결강도, 이상 패턴)
- 경제 이론보다 데이터 기반

REQUIREMENTS:
1. 클래스:
   class TechnicalAgent(BaseEconomicAgent):
       school = 'technical'
       framework = 'Data-driven Signal Detection'
       key_variables = ['Volume', 'Price', 'Volatility', 'Breadth']

2. 시그널 탐지:
   def detect_signals(self, data) -> List[Signal]:
       signals = []

       # 1. Net Buy Ratio
       if self._net_buy_ratio(data) >= 5:
           signals.append(Signal('STRONG_BUY', confidence=0.9))

       # 2. 거래량 스파이크
       if self._volume_spike(data) > 2:  # 2 표준편차
           signals.append(Signal('VOLUME_ALERT', ...))

       # 3. VIX 급등
       if data['d_VIX'].iloc[-1] > 5:
           signals.append(Signal('VOLATILITY_SPIKE', ...))

       # 4. 섹터 로테이션
       rotation = self._detect_sector_rotation(data)
       if rotation:
           signals.append(Signal('SECTOR_ROTATION', ...))

       return signals

3. Signal 스키마:
   @dataclass
   class Signal:
       type: str  # STRONG_BUY, VOLUME_ALERT, etc.
       confidence: float
       timestamp: datetime
       details: Dict
       actionable: bool

4. 추가 분석:
   - 모멘텀 지표 (RSI, MACD)
   - 시장 폭 (Advance/Decline)
   - 변동성 구조 (VIX term structure)
   - 상관관계 붕괴 탐지

OUTPUT:
- eimas/agents/economic_agents/technical.py
```

---

## Phase 4: 토론 및 통합

### Prompt 4.1: EconomicDebateOrchestrator 구현

```
TO DO:
eimas/agents/economic_debate.py에 경제학파 토론 오케스트레이터를 구현하세요.

CONTEXT:
- 3개 학파 에이전트 + Technical 에이전트
- 증거 기반 토론, LLM 합의 도출

REQUIREMENTS:
1. 클래스 구조:
   class EconomicDebateOrchestrator:
       def __init__(self, agents: Dict[str, BaseEconomicAgent], use_llm=True):
       async def run_debate(self, data, network, irf) -> DebateResult:
       async def reach_consensus(self, opinions: List[EconomicOpinion]) -> Consensus:
       def identify_conflicts(self, opinions) -> List[Conflict]:
       def generate_synthesis(self, consensus, signals) -> str

2. 토론 프로세스:
   async def run_debate(self, data, network, irf):
       # 1. 각 에이전트 의견 수집 (병렬)
       opinions = await asyncio.gather(*[
           agent.analyze(data, network, irf)
           for agent in self.agents.values()
       ])

       # 2. 충돌 식별
       conflicts = self.identify_conflicts(opinions)

       # 3. 증거 비교
       evidence_comparison = self._compare_evidence(opinions)

       # 4. LLM 합의 도출 (선택적)
       if self.use_llm:
           consensus = await self._llm_consensus(opinions, conflicts)
       else:
           consensus = self._rule_based_consensus(opinions)

       # 5. 시그널 통합
       signals = self.agents['technical'].detect_signals(data)

       return DebateResult(opinions, conflicts, consensus, signals)

3. LLM 합의 프롬프트:
   """
   다음 경제학파 에이전트들의 의견을 종합하여 합의를 도출하세요.

   [Monetarist]: {opinion1}
   [Keynesian]: {opinion2}
   [Austrian]: {opinion3}

   충돌 지점: {conflicts}

   각 학파의 증거 강도를 고려하여:
   1. 최종 시장 전망 (BULLISH/BEARISH/NEUTRAL)
   2. 신뢰도 (0-1)
   3. 핵심 근거 3가지
   4. 주의사항/리스크
   """

4. DebateResult 스키마:
   @dataclass
   class DebateResult:
       opinions: List[EconomicOpinion]
       conflicts: List[Conflict]
       consensus: Consensus
       signals: List[Signal]
       synthesis: str
       timestamp: datetime

OUTPUT:
- eimas/agents/economic_debate.py
```

---

### Prompt 4.2: main2.py 통합

```
TO DO:
eimas/main2.py에 전체 파이프라인을 통합하세요.

CONTEXT:
- Phase 1-4의 모든 컴포넌트 사용
- 비동기 실행, CLI 인터페이스

REQUIREMENTS:
1. 클래스 구조:
   class EIMASv2:
       def __init__(self, config_path='configs/default_v2.yaml'):
           self.collector = UnifiedDataCollectorV2(...)
           self.network_builder = EconomicNetworkBuilder(...)
           self.agents = {
               'monetarist': MonetaristAgent(),
               'keynesian': KeynesianAgent(),
               'austrian': AustrianAgent(),
               'technical': TechnicalAgent()
           }
           self.orchestrator = EconomicDebateOrchestrator(self.agents)

       async def run(self, query: str = None) -> Dict:
           # Phase 1: 데이터 수집
           data = await self.collector.collect()

           # Phase 2: 네트워크 구축
           network = self.network_builder.build(data)
           irf = self.network_builder.calculate_irf(data)

           # Phase 3: 토론
           debate_result = await self.orchestrator.run_debate(data, network, irf)

           # Phase 4: 결과 종합
           return self._compile_results(data, network, irf, debate_result)

2. CLI 인터페이스:
   - python main2.py                    # 기본 실행
   - python main2.py --mode realtime    # 실시간 모드
   - python main2.py --no-llm           # LLM 없이 실행
   - python main2.py --export html      # HTML 대시보드 생성

3. 결과 출력:
   {
       'timestamp': ...,
       'data_summary': {...},
       'network': {...},
       'irf': {...},
       'debate': {
           'opinions': [...],
           'conflicts': [...],
           'consensus': {...},
           'signals': [...]
       },
       'recommendations': [...],
       'risk_assessment': {...}
   }

4. 로깅:
   - 각 단계별 진행 상황
   - 에러 처리 및 복구
   - 성능 메트릭

OUTPUT:
- eimas/main2.py
- eimas/configs/default_v2.yaml
```

---

### Prompt 4.3: Dashboard v2 생성기

```
TO DO:
eimas/lib/dashboard_generator_v2.py에 개선된 대시보드 생성기를 구현하세요.

CONTEXT:
- 기존: eimas/lib/dashboard_generator.py
- 개선: 네트워크 시각화, IRF 차트, 에이전트 토론 뷰

REQUIREMENTS:
1. 추가 섹션:
   - 경제 네트워크 그래프 (D3.js/vis.js)
   - IRF 차트 (Plotly)
   - 학파별 의견 카드
   - 충돌/합의 시각화
   - 시그널 타임라인

2. 네트워크 시각화:
   def generate_network_section(self, network: EconomicNetwork) -> str:
       # vis.js 또는 D3.js 사용
       # 노드: 변수 (색상으로 카테고리 구분)
       # 엣지: Granger 인과관계 (굵기 = 강도)
       # 인터랙티브: 클릭 시 상세 정보

3. IRF 차트:
   def generate_irf_section(self, irf: Dict) -> str:
       # Plotly 사용
       # 충격 변수 선택 드롭다운
       # 반응 변수 멀티 선택
       # 신뢰구간 표시

4. 토론 뷰:
   def generate_debate_section(self, debate_result: DebateResult) -> str:
       # 학파별 카드 (색상 구분)
       # Monetarist: 파란색
       # Keynesian: 녹색
       # Austrian: 주황색
       # 합의: 보라색 강조
       # 충돌: 빨간색 표시

5. 반응형 디자인:
   - 모바일 대응
   - 다크/라이트 테마 전환
   - 프린트 친화적

OUTPUT:
- eimas/lib/dashboard_generator_v2.py
- static/js/network.js (네트워크 시각화)
- static/js/irf.js (IRF 차트)
```

---

## 실행 순서

```
1. Prompt 1.1 → UnifiedDataCollectorV2
2. Prompt 1.2 → EventRegistry
3. Prompt 2.1 → EconomicNetworkBuilder
4. Prompt 2.2 → IRF Visualizer
5. Prompt 3.1 → BaseEconomicAgent
6. Prompt 3.2 → MonetaristAgent
7. Prompt 3.3 → KeynesianAgent
8. Prompt 3.4 → AustrianAgent
9. Prompt 3.5 → TechnicalAgent
10. Prompt 4.1 → EconomicDebateOrchestrator
11. Prompt 4.2 → main2.py
12. Prompt 4.3 → Dashboard v2
```

---

## 검증 체크리스트

```
Phase 1 완료 후:
[ ] UnifiedDataCollectorV2.collect() 실행 성공
[ ] 이벤트 태깅 확인 (event_regime 컬럼)
[ ] 캐싱 작동 확인

Phase 2 완료 후:
[ ] Granger 인과성 매트릭스 생성
[ ] VAR 모델 추정 성공
[ ] IRF 계산 및 시각화

Phase 3 완료 후:
[ ] 각 에이전트 analyze() 성공
[ ] EconomicOpinion 형식 준수
[ ] 증거 기반 의견 생성

Phase 4 완료 후:
[ ] 전체 파이프라인 실행
[ ] 대시보드 생성
[ ] CLI 명령어 작동
```

---

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2025-12-25 | v1.0 | 초기 프롬프트 가이드 작성 |
