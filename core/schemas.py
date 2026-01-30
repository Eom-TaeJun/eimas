#!/usr/bin/env python3
"""
Multi-Agent System - Core Schemas
=================================
에이전트 간 통신을 위한 표준화된 데이터 스키마
Standardized data schemas for inter-agent communication.

Schema Organization | 스키마 구성:
    - core/schemas.py (이 파일): 에이전트 통신, 토론, 워크플로우 스키마
    - pipeline/schemas.py: 분석 결과 및 저장용 스키마 (EIMASResult 등)

Key Classes:
    - AgentRequest/Response: Agent communication protocol
    - AgentOpinion/Consensus: Multi-agent debate schemas
    - WorkflowPlan/WorkflowStep: Task orchestration schemas
    - ForecastResult/LASSODiagnostics: Prediction output schemas

Design Principles | 설계 원칙:
    - 에이전트 간 정보 비대칭 최소화 (Information Symmetry)
    - 표준화된 인터페이스로 모듈화 극대화 (Standardization)
    - Type safety로 런타임 에러 최소화 (Type Safety)
    - to_dict() 메서드로 JSON 직렬화 지원 (Serialization)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal, Tuple
from datetime import datetime
from enum import Enum


class AgentRole(str, Enum):
    """에이전트 역할 정의"""
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    FORECAST = "forecast"
    STRATEGY = "strategy"
    VERIFICATION = "verification"


class TaskPriority(str, Enum):
    """작업 우선순위"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OpinionStrength(str, Enum):
    """의견 강도 (Debate용)"""
    STRONG_AGREE = "strong_agree"      # 강하게 동의
    AGREE = "agree"                    # 동의
    NEUTRAL = "neutral"                # 중립
    DISAGREE = "disagree"              # 반대
    STRONG_DISAGREE = "strong_disagree"  # 강하게 반대


class AnalysisMode(str, Enum):
    """
    분석 모드 - 역사적 데이터 활용 방식

    FULL: 2024-2025 데이터를 주요 분석 입력으로 사용
          - 장점: 최신 패턴 반영, 정밀한 예측
          - 단점: Regime 변화 시 overfitting 위험

    REFERENCE: 역사적 데이터를 참고용으로만 사용
          - 장점: Regime 변화에 강건
          - 단점: 최신 패턴 놓칠 수 있음

    경제학적 의미:
    - FULL = In-sample optimization (정밀도 ↑)
    - REFERENCE = Out-of-sample robustness (일반화 ↑)
    """
    FULL = "full"          # 역사적 데이터 full weight (기존 방식)
    REFERENCE = "reference"  # 역사적 데이터 참고만 (낮은 가중치)


@dataclass
class HistoricalDataConfig:
    """
    역사적 데이터 활용 설정

    Attributes:
        mode: 분석 모드 (FULL 또는 REFERENCE)
        historical_weight: 역사적 데이터 가중치 (0.0-1.0)
        realtime_weight: 실시간 데이터 가중치 (0.0-1.0)
        regime_adaptive: Regime 변화 시 가중치 자동 조정
    """
    mode: AnalysisMode = AnalysisMode.FULL
    historical_weight: float = 0.7  # FULL: 0.7, REFERENCE: 0.2
    realtime_weight: float = 0.3    # FULL: 0.3, REFERENCE: 0.8
    regime_adaptive: bool = True     # Regime 변화 감지 시 가중치 조정

    def __post_init__(self):
        """모드에 따라 기본 가중치 설정"""
        if self.mode == AnalysisMode.REFERENCE:
            self.historical_weight = 0.2
            self.realtime_weight = 0.8
        elif self.mode == AnalysisMode.FULL:
            self.historical_weight = 0.7
            self.realtime_weight = 0.3

    def get_combined_weight(self, is_historical: bool) -> float:
        """데이터 소스에 따른 가중치 반환"""
        return self.historical_weight if is_historical else self.realtime_weight


@dataclass
class AgentRequest:
    """
    에이전트에게 전달되는 요청

    Attributes:
        task_id: 작업 고유 ID
        role: 요청 대상 에이전트 역할
        instruction: 작업 지시사항
        context: 추가 컨텍스트 (이전 결과, 데이터 등)
        priority: 우선순위
        deadline: 완료 기한 (초 단위, None이면 무제한)
        metadata: 추가 메타데이터
    """
    task_id: str
    role: AgentRole
    instruction: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[int] = None  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['role'] = self.role.value
        data['priority'] = self.priority.value
        return data


@dataclass
class AgentResponse:
    """
    에이전트가 반환하는 응답

    Attributes:
        task_id: 작업 ID
        agent_role: 응답한 에이전트 역할
        status: 성공/실패 상태
        result: 실행 결과 (딕셔너리)
        confidence: 결과 신뢰도 (0-1)
        reasoning: 추론 과정 설명
        error: 에러 메시지 (실패 시)
        execution_time: 실행 시간 (초)
        timestamp: 타임스탬프
        metadata: 추가 메타데이터
    """
    task_id: str
    agent_role: AgentRole
    status: Literal["success", "failure", "partial"]
    result: Dict[str, Any]
    confidence: float = 1.0  # 0.0 - 1.0
    reasoning: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['agent_role'] = self.agent_role.value
        return data

    def is_success(self) -> bool:
        """성공 여부 확인"""
        return self.status == "success"


@dataclass
class AgentOpinion:
    """
    에이전트의 의견 (Debate용)

    경제학적 의미:
    - 의견 다양성(Diversity of Opinion)이 군집사고 방지
    - 신뢰도(Confidence)가 가중평균 시 가중치로 사용
    - 근거(Evidence)가 투명성과 검증가능성 확보
    """
    agent_role: AgentRole
    topic: str  # 의견 주제 (예: "market_outlook", "primary_risk")
    position: str  # 핵심 주장
    strength: OpinionStrength
    confidence: float  # 0.0 - 1.0
    evidence: List[str]  # 근거 목록
    caveats: List[str] = field(default_factory=list)  # 단서/제약조건
    key_metrics: Dict[str, float] = field(default_factory=dict)  # 정량적 근거
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['agent_role'] = self.agent_role.value
        data['strength'] = self.strength.value
        return data


@dataclass
class Conflict:
    """
    에이전트 간 의견 충돌

    Attributes:
        agents: 충돌하는 에이전트들
        topic: 충돌 주제
        positions: 각 에이전트의 입장
        severity: 충돌 심각도 (0-1)
        resolution_needed: 해결 필요 여부
    """
    agents: List[AgentRole]
    topic: str
    positions: Dict[AgentRole, str]
    severity: float  # 0.0 - 1.0
    resolution_needed: bool = True

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'agents': [a.value for a in self.agents],
            'topic': self.topic,
            'positions': {k.value: v for k, v in self.positions.items()},
            'severity': self.severity,
            'resolution_needed': self.resolution_needed
        }


@dataclass
class Consensus:
    """
    에이전트 간 합의 결과

    경제학적 의미:
    - Nash Equilibrium: 모든 에이전트가 수용 가능한 해
    - Pareto Efficiency: 개선 여지가 없는 최적 상태
    - Weighted Average: 신뢰도 기반 가중평균
    """
    topic: str
    final_position: str
    confidence: float  # 합의 신뢰도 (0-1)
    supporting_agents: List[AgentRole]
    dissenting_agents: List[AgentRole]
    compromises: List[str]  # 타협 내용
    debate_rounds: int = 0  # 거친 토론 라운드 수
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 반대의견 상세 (의견 묵살 방지)
    dissent_details: List[Dict] = field(default_factory=list)
    has_strong_dissent: bool = False  # 강력한 반대의견 존재 여부

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'topic': self.topic,
            'final_position': self.final_position,
            'confidence': self.confidence,
            'supporting_agents': [a.value for a in self.supporting_agents],
            'dissenting_agents': [a.value for a in self.dissenting_agents],
            'compromises': self.compromises,
            'debate_rounds': self.debate_rounds,
            'timestamp': self.timestamp,
            'dissent_details': self.dissent_details,
            'has_strong_dissent': self.has_strong_dissent
        }


@dataclass
class WorkflowStep:
    """워크플로우 단계"""
    step_id: str
    agent_role: AgentRole
    task_description: str
    dependencies: List[str] = field(default_factory=list)  # 선행 step_id 목록
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: Optional[int] = None  # 초 단위

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'step_id': self.step_id,
            'agent_role': self.agent_role.value,
            'task_description': self.task_description,
            'dependencies': self.dependencies,
            'priority': self.priority.value,
            'estimated_duration': self.estimated_duration
        }


@dataclass
class WorkflowPlan:
    """
    Meta-Orchestrator가 생성하는 워크플로우 계획

    경제학적 의미:
    - Critical Path: 최장 경로 = 최소 완료 시간
    - Parallel Execution: 독립적 작업은 병렬 처리로 효율 극대화
    - Resource Allocation: 우선순위 기반 자원 배분
    """
    workflow_id: str
    goal: str
    steps: List[WorkflowStep]
    total_estimated_duration: Optional[int] = None
    critical_path: List[str] = field(default_factory=list)  # step_id 순서
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'workflow_id': self.workflow_id,
            'goal': self.goal,
            'steps': [s.to_dict() for s in self.steps],
            'total_estimated_duration': self.total_estimated_duration,
            'critical_path': self.critical_path,
            'created_at': self.created_at
        }

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """step_id로 단계 조회"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_parallel_steps(self) -> List[List[WorkflowStep]]:
        """
        병렬 실행 가능한 단계들을 그룹으로 반환

        Returns:
            List of groups, where each group can run in parallel
        """
        # 의존성 그래프 기반으로 레벨별로 그룹화
        completed = set()
        groups = []

        while len(completed) < len(self.steps):
            # 현재 실행 가능한 단계들 (모든 의존성이 완료됨)
            ready = []
            for step in self.steps:
                if step.step_id in completed:
                    continue
                if all(dep in completed for dep in step.dependencies):
                    ready.append(step)

            if not ready:
                break  # 순환 의존성 또는 에러

            groups.append(ready)
            completed.update(s.step_id for s in ready)

        return groups


@dataclass
class MarketData:
    """시장 데이터 스키마"""
    ticker: str
    data: Dict[str, Any]  # OHLCV, indicators 등
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EconomicIndicator:
    """경제 지표 스키마"""
    indicator_name: str
    value: float
    unit: str
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ForecastResult:
    """
    LASSO 예측 결과 스키마
    
    Fed 금리 기대 변화 예측을 위한 LASSO 분석 결과를 담는 데이터클래스.
    각 horizon별로 별도의 ForecastResult 인스턴스가 생성됨.
    """
    horizon: str                              # "VeryShort" / "Short" / "Long"
    selected_variables: List[str]             # LASSO가 선택한 변수 목록
    coefficients: Dict[str, float]            # {변수명: 표준화 계수}
    r_squared: float                          # 결정계수 (0~1)
    n_observations: int                       # 관측치 수
    lambda_optimal: float                     # 최적 정규화 파라미터
    hac_std_errors: Dict[str, float] = field(default_factory=dict)   # HAC 표준오차
    vif_scores: Dict[str, float] = field(default_factory=dict)       # VIF 점수
    predicted_change: Optional[float] = None  # 예측 금리 변화 (bp)
    confidence_interval: Optional[Tuple[float, float]] = None  # 신뢰구간 (lower, upper)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def get_top_variables(self, n: int = 5) -> List[Tuple[str, float]]:
        """절대값 기준 상위 n개 변수 반환"""
        sorted_coefs = sorted(
            self.coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_coefs[:n]
    
    def has_high_vif_warnings(self, threshold: float = 10.0) -> List[str]:
        """VIF > threshold인 변수 목록 반환"""
        return [var for var, vif in self.vif_scores.items() if vif > threshold]


@dataclass
class LASSODiagnostics:
    """
    LASSO 진단 정보
    
    LASSO 모델 학습 과정의 진단 정보를 담는 데이터클래스.
    모델 품질 평가 및 디버깅에 활용.
    """
    total_candidate_vars: int                 # 후보 변수 총 수
    excluded_vars: List[str]                  # 제외된 변수 (Treasury 등)
    high_vif_warnings: List[str]              # VIF > 10인 변수
    convergence_info: Dict[str, bool]         # {horizon: 수렴여부}
    computation_time: float                   # 계산 시간 (초)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def is_all_converged(self) -> bool:
        """모든 horizon이 수렴했는지 확인"""
        return all(self.convergence_info.values())


@dataclass
class DashboardConfig:
    """
    대시보드 생성 설정
    
    VisualizationAgent가 대시보드를 생성할 때 사용하는 설정.
    각 섹션의 포함 여부와 스타일을 제어.
    """
    theme: str = 'dark'                       # 'dark' / 'light'
    language: str = 'ko'                      # 'ko' / 'en'
    include_crypto: bool = True               # 암호화폐 패널 포함
    include_regime: bool = True               # 레짐 분석 포함
    include_critical_path: bool = True        # Critical Path 분석 포함
    include_lasso_results: bool = True        # LASSO 결과 포함
    include_agent_debate: bool = True         # 에이전트 토론 결과 포함
    include_risk_metrics: bool = True         # 위험 메트릭 포함
    include_macro_indicators: bool = True     # 거시경제 지표 포함
    chart_library: str = 'chartjs'            # 'chartjs' / 'plotly'
    max_signals_display: int = 30             # 최대 신호 표시 수
    output_dir: str = 'outputs/dashboards'    # 출력 디렉토리

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass
class HorizonConfig:
    """
    Horizon 분류 설정
    
    FOMC 회의까지 남은 일수에 따른 horizon 분류 기준.
    91~179일 구간은 분석에서 제외됨.
    """
    very_short_max: int = 30                  # VeryShort: <= 30일
    short_min: int = 31                       # Short: 31일 이상
    short_max: int = 90                       # Short: 90일 이하
    long_min: int = 180                       # Long: 180일 이상
    # 참고: 91~179일은 분석에서 제외

    def classify(self, days_to_meeting: int) -> Optional[str]:
        """
        FOMC 회의까지 남은 일수를 기준으로 horizon 분류
        
        Args:
            days_to_meeting: FOMC 회의까지 남은 거래일 수
            
        Returns:
            "VeryShort", "Short", "Long", 또는 None (제외 구간)
        """
        if days_to_meeting <= self.very_short_max:
            return "VeryShort"
        elif self.short_min <= days_to_meeting <= self.short_max:
            return "Short"
        elif days_to_meeting >= self.long_min:
            return "Long"
        else:
            return None  # 91~179일은 제외

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass
class StrategyRecommendation:
    """전략 권고 스키마"""
    strategy_type: str  # "LONG", "SHORT", "HEDGE", "NEUTRAL"
    asset_class: str  # "EQUITY", "BOND", "COMMODITY", "FX", "CRYPTO"
    position_size: float  # 0.0 - 1.0 (포트폴리오 비중)
    rationale: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    expected_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass
class AgentOutputs:
    """각 에이전트의 출력 기록"""
    analysis: Dict          # AnalysisAgent 출력
    forecast: Dict          # ForecastAgent 출력
    research: Dict          # ResearchAgent 출력 (NEW)
    strategy: Dict          # StrategyAgent 출력 (NEW)
    interpretation: Dict    # InterpretationDebateAgent 출력 (NEW)
    methodology: Dict       # MethodologyDebateAgent 출력 (NEW)


@dataclass
class DebateResults:
    """Multi-LLM 토론 결과"""
    transcript: List[Dict]              # 전체 대화 기록
    consensus_position: str             # 합의 입장
    consensus_confidence: Tuple[float, float]  # 신뢰도 범위
    dissent_points: List[Dict]          # 불일치 포인트
    model_contributions: Dict[str, str] # 각 모델 기여
    consensus_points: List[str] = field(default_factory=list)


@dataclass
class VerificationResults:
    """검증 결과"""
    overall_reliability: float          # 0-100
    consistency_score: float            # 내부 일관성
    data_alignment_score: float         # 데이터 정합성
    bias_detected: List[str]            # 탐지된 편향
    warnings: List[str]                 # 경고 사항


@dataclass
class EIMASResult:
    timestamp: str

    # ===== Phase 1: Data =====
    fred_summary: Dict
    market_data_count: int
    crypto_data_count: int

    # ===== Phase 2: Analysis =====
    regime: Dict
    risk_score: float
    events_detected: List[Dict]

    # ===== Phase 3: Agent Outputs (NEW) =====
    agent_outputs: AgentOutputs  # 새 데이터클래스

    # ===== Phase 4: Debate (NEW) =====
    debate_results: DebateResults  # 새 데이터클래스

    # ===== Phase 5: Verification (NEW) =====
    verification: VerificationResults  # 새 데이터클래스

    # ===== Final =====
    final_recommendation: str
    confidence: float
    confidence_range: Tuple[float, float]  # NEW: 범위로 표현
    reasoning_chain: List[Dict]  # NEW: 추론 과정 추적
    risk_level: str = "Unknown"


if __name__ == "__main__":
    # 테스트
    print("=== Schema Test ===")

    # AgentRequest 테스트
    req = AgentRequest(
        task_id="task_001",
        role=AgentRole.RESEARCH,
        instruction="Search for Fed policy changes",
        priority=TaskPriority.HIGH
    )
    print(f"\nAgentRequest: {req.to_dict()}")

    # AgentResponse 테스트
    resp = AgentResponse(
        task_id="task_001",
        agent_role=AgentRole.RESEARCH,
        status="success",
        result={"findings": "Fed signals hawkish shift"},
        confidence=0.85
    )
    print(f"\nAgentResponse: {resp.to_dict()}")

    # WorkflowPlan 테스트
    plan = WorkflowPlan(
        workflow_id="wf_001",
        goal="Analyze market anomalies",
        steps=[
            WorkflowStep("step1", AgentRole.RESEARCH, "Search news", dependencies=[]),
            WorkflowStep("step2", AgentRole.ANALYSIS, "Run analysis", dependencies=[]),
            WorkflowStep("step3", AgentRole.FORECAST, "Generate forecast", dependencies=["step2"]),
            WorkflowStep("step4", AgentRole.STRATEGY, "Recommend strategy", dependencies=["step2", "step3"])
        ]
    )

    parallel_groups = plan.get_parallel_steps()
    print(f"\nWorkflowPlan parallel groups:")
    for i, group in enumerate(parallel_groups):
        print(f"  Level {i}: {[s.step_id for s in group]}")
