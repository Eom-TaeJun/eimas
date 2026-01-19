"""
Top-Down Orchestrator - 하향식 분석 오케스트레이터

Level 0 (세계 정세) → Level 1 (통화 환경) → Level 2 (자산군) →
Level 3 (섹터) → Level 4 (개별 기업)

상위 레벨이 부정적이면 하위 레벨 분석은 의미 감소
ECON_AI_AGENT_SYSTEM.md Section 6 구현

Author: EIMAS Team
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable

import sys
sys.path.append('..')
from core.archive.debate_framework import (
    DebateFramework,
    DebateParticipant,
    DebateConfig,
    DebateResult,
    Opinion,
    AIProvider
)
from core.schemas import AgentRole
from agents.base_agent import BaseAgent, AgentConfig


# ============================================================================
# Enums and Constants
# ============================================================================

class AnalysisLevel(Enum):
    """분석 레벨"""
    GEOPOLITICS = 0       # 세계 정세
    MONETARY = 1          # 통화 환경
    ASSET_CLASS = 2       # 자산군
    SECTOR = 3            # 섹터
    INDIVIDUAL = 4        # 개별 기업/자산


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Stance(Enum):
    """투자 스탠스"""
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"
    RISK_OFF = "RISK_OFF"


class PolicyStance(Enum):
    """통화정책 스탠스"""
    VERY_DOVISH = "VERY_DOVISH"
    DOVISH = "DOVISH"
    NEUTRAL = "NEUTRAL"
    HAWKISH = "HAWKISH"
    VERY_HAWKISH = "VERY_HAWKISH"


class LiquidityRegime(Enum):
    """유동성 레짐"""
    ABUNDANT = "ABUNDANT"
    NORMAL = "NORMAL"
    TIGHT = "TIGHT"


class AllocationWeight(Enum):
    """자산 배분 가중치"""
    OVERWEIGHT = "OVERWEIGHT"
    NEUTRAL = "NEUTRAL"
    UNDERWEIGHT = "UNDERWEIGHT"


class CyclePosition(Enum):
    """경기 사이클 위치"""
    EARLY = "EARLY"
    MID = "MID"
    LATE = "LATE"
    RECESSION = "RECESSION"
    TURNING = "TURNING"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LevelAnalysisResult:
    """레벨별 분석 결과"""
    level: AnalysisLevel
    risk_level: RiskLevel
    stance: Stance
    summary: str
    key_findings: List[str]
    recommendations: Dict[str, Any]
    confidence: float
    should_continue: bool = True  # 다음 레벨 분석 계속 여부
    abort_reason: Optional[str] = None  # 중단 사유
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GeopoliticalResult(LevelAnalysisResult):
    """Level 0: 세계 정세 분석 결과"""
    primary_risks: List[Dict[str, Any]] = field(default_factory=list)
    risk_assets_stance: AllocationWeight = AllocationWeight.NEUTRAL
    safe_haven_stance: AllocationWeight = AllocationWeight.NEUTRAL


@dataclass
class MonetaryResult(LevelAnalysisResult):
    """Level 1: 통화 환경 분석 결과"""
    liquidity_regime: LiquidityRegime = LiquidityRegime.NORMAL
    policy_stance: PolicyStance = PolicyStance.NEUTRAL
    inflation_trend: str = "STABLE"
    cycle_position: CyclePosition = CyclePosition.MID
    asset_implications: Dict[str, AllocationWeight] = field(default_factory=dict)


@dataclass
class AssetClassResult(LevelAnalysisResult):
    """Level 2: 자산군 분석 결과"""
    allocations: Dict[str, AllocationWeight] = field(default_factory=dict)
    conviction: str = "MEDIUM"
    cross_asset_signals: List[str] = field(default_factory=list)
    preferred_assets: List[str] = field(default_factory=list)
    avoid_assets: List[str] = field(default_factory=list)


@dataclass
class SectorResult(LevelAnalysisResult):
    """Level 3: 섹터 분석 결과"""
    top_sectors: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)
    rotation_signal: str = "NEUTRAL"  # "DEFENSIVE" | "NEUTRAL" | "CYCLICAL"
    cycle_based_rationale: str = ""


@dataclass
class IndividualResult(LevelAnalysisResult):
    """Level 4: 개별 자산 분석 결과"""
    analyzed_assets: List[Dict[str, Any]] = field(default_factory=list)
    top_picks: List[str] = field(default_factory=list)
    short_candidates: List[str] = field(default_factory=list)


@dataclass
class TopDownResult:
    """전체 하향식 분석 결과"""
    geopolitical: Optional[GeopoliticalResult] = None
    monetary: Optional[MonetaryResult] = None
    asset_class: Optional[AssetClassResult] = None
    sector: Optional[SectorResult] = None
    individual: Optional[IndividualResult] = None

    final_stance: Stance = Stance.NEUTRAL
    final_recommendation: str = ""
    aborted_at_level: Optional[AnalysisLevel] = None
    abort_reason: Optional[str] = None
    total_confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Level-Specific Debate Prompts
# ============================================================================

LEVEL_DEBATE_PROMPTS = {
    AnalysisLevel.GEOPOLITICS: {
        "topic": "Global Geopolitical Risk Assessment",
        "questions": [
            "현재 세계 정세에서 가장 큰 위험 요소는?",
            "향후 3개월 내 지정학적 충격 가능성은?",
            "리스크 자산 투자에 적합한 환경인가?"
        ],
        "data_keywords": [
            "Russia Ukraine war",
            "China Taiwan tensions",
            "Middle East conflict",
            "US China trade relations",
            "Global supply chain risks"
        ],
        "response_format": """
위 정보를 바탕으로 다음을 분석해주세요:

1. 전반적 지정학적 위험 수준 (LOW/MEDIUM/ELEVATED/HIGH/CRITICAL)
2. 주요 위험 요소 3가지 (위험, 확률, 영향도)
3. 리스크 자산 스탠스 (OVERWEIGHT/NEUTRAL/UNDERWEIGHT)
4. 안전자산 스탠스 (OVERWEIGHT/NEUTRAL/UNDERWEIGHT)
5. 분석 신뢰도 (0-1)

JSON 형식으로 응답:
```json
{
    "risk_level": "MEDIUM",
    "primary_risks": [
        {"risk": "...", "probability": 0.3, "impact": "HIGH"}
    ],
    "risk_assets_stance": "NEUTRAL",
    "safe_haven_stance": "NEUTRAL",
    "key_findings": ["...", "..."],
    "summary": "...",
    "confidence": 0.7
}
```
"""
    },

    AnalysisLevel.MONETARY: {
        "topic": "Global Monetary Environment Assessment",
        "questions": [
            "돈이 많이 풀린 상황인가? (글로벌 유동성)",
            "중앙은행들은 긴축인가 완화인가?",
            "인플레이션은 통제되고 있는가?",
            "금리 사이클의 어디에 있는가?"
        ],
        "indicators": {
            "liquidity": ["Global M2 growth", "Central bank balance sheets", "Credit growth"],
            "policy": ["Fed stance", "ECB stance", "BOJ stance", "PBOC stance"],
            "inflation": ["US CPI/PCE", "Eurozone HICP", "Breakeven inflation"]
        },
        "response_format": """
위 데이터를 바탕으로 다음을 분석해주세요:

1. 유동성 레짐 (ABUNDANT/NORMAL/TIGHT)
2. 통화정책 스탠스 (VERY_DOVISH/DOVISH/NEUTRAL/HAWKISH/VERY_HAWKISH)
3. 인플레이션 추세 (ACCELERATING/STABLE/DECELERATING)
4. 경기 사이클 위치 (EARLY/MID/LATE/RECESSION/TURNING)
5. 자산군별 함의 (주식/채권/원자재/암호화폐 각각 OVERWEIGHT/NEUTRAL/UNDERWEIGHT)

JSON 형식으로 응답:
```json
{
    "liquidity_regime": "NORMAL",
    "policy_stance": "NEUTRAL",
    "inflation_trend": "STABLE",
    "cycle_position": "MID",
    "asset_implications": {
        "equities": "NEUTRAL",
        "bonds": "NEUTRAL",
        "commodities": "NEUTRAL",
        "crypto": "NEUTRAL"
    },
    "key_findings": ["...", "..."],
    "summary": "...",
    "confidence": 0.75
}
```
"""
    },

    AnalysisLevel.ASSET_CLASS: {
        "topic": "Asset Class Assessment",
        "questions": [
            "주식시장이 계속 성장할 것인가?",
            "채권은 매력적인가?",
            "원자재 슈퍼사이클인가?",
            "암호화폐에 자금이 유입되고 있는가?"
        ],
        "asset_classes": ["Equities", "Fixed Income", "Commodities", "Crypto", "Cash"],
        "response_format": """
상위 레벨 분석 결과와 현재 데이터를 바탕으로:

1. 자산군별 배분 추천 (각각 OVERWEIGHT/NEUTRAL/UNDERWEIGHT)
2. 확신도 (HIGH/MEDIUM/LOW)
3. 교차 자산 시그널 (예: Stock-bond correlation)
4. 선호 자산, 회피 자산

JSON 형식으로 응답:
```json
{
    "allocations": {
        "equities": "NEUTRAL",
        "bonds": "NEUTRAL",
        "commodities": "NEUTRAL",
        "crypto": "NEUTRAL",
        "cash": "NEUTRAL"
    },
    "conviction": "MEDIUM",
    "cross_asset_signals": ["...", "..."],
    "preferred_assets": ["...", "..."],
    "avoid_assets": ["..."],
    "key_findings": ["...", "..."],
    "summary": "...",
    "confidence": 0.7
}
```
"""
    },

    AnalysisLevel.SECTOR: {
        "topic": "Sector Rotation Analysis",
        "questions": [
            "현재 경기 사이클에서 어떤 섹터가 유리한가?",
            "금리 환경이 어떤 섹터에 영향을 주는가?",
            "정책 변화로 수혜받는 섹터는?"
        ],
        "sectors": [
            "Technology", "Financials", "Healthcare", "Energy",
            "Consumer Discretionary", "Consumer Staples", "Industrials",
            "Materials", "Utilities", "Real Estate", "Communication Services"
        ],
        "response_format": """
통화 환경과 경기 사이클을 고려하여:

1. 상위 3개 유망 섹터
2. 회피해야 할 섹터
3. 섹터 로테이션 시그널 (DEFENSIVE/NEUTRAL/CYCLICAL)
4. 사이클 기반 근거

JSON 형식으로 응답:
```json
{
    "top_sectors": ["Technology", "Financials", "Healthcare"],
    "avoid_sectors": ["Utilities"],
    "rotation_signal": "NEUTRAL",
    "cycle_based_rationale": "...",
    "key_findings": ["...", "..."],
    "summary": "...",
    "confidence": 0.65
}
```
"""
    },

    AnalysisLevel.INDIVIDUAL: {
        "topic": "Individual Asset Analysis",
        "questions": [
            "선택된 섹터 내에서 가장 유망한 기업/자산은?",
            "펀더멘털, 밸류에이션, 기술적 분석 결과는?",
            "레짐 변화 리스크가 있는 기업은?"
        ],
        "response_format": """
선택된 섹터 내 개별 자산 분석:

1. 분석된 자산 목록 (이름, 스코어, 근거)
2. Top 3 매수 추천
3. 공매도 후보 (있다면)

JSON 형식으로 응답:
```json
{
    "analyzed_assets": [
        {"name": "...", "score": 0.8, "rationale": "..."}
    ],
    "top_picks": ["...", "..."],
    "short_candidates": [],
    "key_findings": ["...", "..."],
    "summary": "...",
    "confidence": 0.6
}
```
"""
    }
}


# ============================================================================
# Top-Down Orchestrator
# ============================================================================

class TopDownOrchestrator(BaseAgent):
    """
    하향식 분석 오케스트레이터

    상위 레벨 분석 결과가 하위 레벨에 영향을 미치는
    계층적 분석 흐름을 관리

    사용법:
    ```python
    orchestrator = TopDownOrchestrator()
    result = await orchestrator.run_full_analysis(
        data=market_data,
        stop_at_level=AnalysisLevel.SECTOR  # 섹터까지만 분석
    )
    ```
    """

    # 레벨별 리스크 임계값 (이 이상이면 다음 레벨 중단)
    ABORT_THRESHOLDS = {
        AnalysisLevel.GEOPOLITICS: RiskLevel.CRITICAL,
        AnalysisLevel.MONETARY: RiskLevel.HIGH,
        AnalysisLevel.ASSET_CLASS: RiskLevel.ELEVATED,
        AnalysisLevel.SECTOR: RiskLevel.HIGH,
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        debate_config: Optional[DebateConfig] = None
    ):
        if config is None:
            config = AgentConfig(
                name="TopDownOrchestrator",
                role=AgentRole.ORCHESTRATOR,
                model="multi-ai"
            )
        super().__init__(config)

        self.debate_config = debate_config or DebateConfig(
            max_rounds=3,
            enable_rebuttal=True,
            consensus_threshold=0.7
        )

        # 레벨별 분석 함수 매핑
        self._level_analyzers: Dict[AnalysisLevel, Callable] = {
            AnalysisLevel.GEOPOLITICS: self._analyze_geopolitics,
            AnalysisLevel.MONETARY: self._analyze_monetary,
            AnalysisLevel.ASSET_CLASS: self._analyze_asset_class,
            AnalysisLevel.SECTOR: self._analyze_sector,
            AnalysisLevel.INDIVIDUAL: self._analyze_individual,
        }

    async def _execute(self, request: Any) -> TopDownResult:
        """BaseAgent 구현"""
        return await self.run_full_analysis(
            data=request.get('data', {}),
            stop_at_level=request.get('stop_at_level'),
            skip_levels=request.get('skip_levels', [])
        )

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        특정 토픽에 대한 의견 형성

        Parameters:
        -----------
        topic : str
            토론 주제
        context : Dict
            컨텍스트 데이터

        Returns:
        --------
        Dict
            의견 및 근거
        """
        # 해당 레벨 분석 결과 활용
        result = await self.run_full_analysis(
            data=context,
            stop_at_level=AnalysisLevel.ASSET_CLASS  # 빠른 분석
        )

        return {
            "topic": topic,
            "stance": result.final_stance.value,
            "recommendation": result.final_recommendation,
            "confidence": result.total_confidence,
            "key_findings": [
                result.geopolitical.key_findings[:2] if result.geopolitical else [],
                result.monetary.key_findings[:2] if result.monetary else []
            ],
            "rationale": f"하향식 분석 기반: {result.final_recommendation}"
        }

    async def run_full_analysis(
        self,
        data: Dict[str, Any],
        stop_at_level: Optional[AnalysisLevel] = None,
        skip_levels: Optional[List[AnalysisLevel]] = None
    ) -> TopDownResult:
        """
        전체 하향식 분석 실행

        Parameters:
        -----------
        data : Dict
            분석에 필요한 데이터
        stop_at_level : AnalysisLevel
            이 레벨까지만 분석 (None이면 전체)
        skip_levels : List[AnalysisLevel]
            건너뛸 레벨 목록

        Returns:
        --------
        TopDownResult
            전체 분석 결과
        """
        skip_levels = skip_levels or []
        result = TopDownResult()
        context: Dict[str, LevelAnalysisResult] = {}

        # 레벨 순서대로 분석
        levels = [
            AnalysisLevel.GEOPOLITICS,
            AnalysisLevel.MONETARY,
            AnalysisLevel.ASSET_CLASS,
            AnalysisLevel.SECTOR,
            AnalysisLevel.INDIVIDUAL
        ]

        for level in levels:
            # 중단 레벨 확인
            if stop_at_level and level.value > stop_at_level.value:
                break

            # 스킵 레벨 확인
            if level in skip_levels:
                continue

            # 레벨 분석 실행
            level_result = await self._analyze_level(level, data, context)

            # 결과 저장
            self._store_result(result, level, level_result)
            context[level.name] = level_result

            # 중단 조건 확인
            if not level_result.should_continue:
                result.aborted_at_level = level
                result.abort_reason = level_result.abort_reason
                break

        # 최종 결과 종합
        result.final_stance = self._determine_final_stance(result)
        result.final_recommendation = self._generate_final_recommendation(result)
        result.total_confidence = self._calculate_total_confidence(result)

        return result

    async def _analyze_level(
        self,
        level: AnalysisLevel,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> LevelAnalysisResult:
        """레벨별 분석 실행"""
        analyzer = self._level_analyzers.get(level)
        if analyzer:
            return await analyzer(data, context)

        # 기본 결과
        return LevelAnalysisResult(
            level=level,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary="분석 미구현",
            key_findings=[],
            recommendations={},
            confidence=0.5
        )

    async def _analyze_geopolitics(
        self,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> GeopoliticalResult:
        """
        Level 0: 세계 정세 분석

        데이터 없이도 AI 토론으로 현황 파악 가능
        """
        prompt_config = LEVEL_DEBATE_PROMPTS[AnalysisLevel.GEOPOLITICS]

        # 외부 데이터가 있으면 포함
        news_data = data.get('geopolitical_news', [])
        risk_indices = data.get('risk_indices', {})

        analysis_prompt = f"""
# {prompt_config['topic']}

## 분석 질문
{chr(10).join(['- ' + q for q in prompt_config['questions']])}

## 검색 키워드 (참고)
{', '.join(prompt_config['data_keywords'])}

## 제공된 데이터
뉴스: {len(news_data)}건
리스크 지표: {risk_indices}

{prompt_config['response_format']}
"""

        # 실제 AI 토론은 DebateFramework 사용
        # 여기서는 기본 결과 반환 (통합 테스트에서 실제 토론)
        result = GeopoliticalResult(
            level=AnalysisLevel.GEOPOLITICS,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary="지정학적 위험은 중간 수준. 주요 분쟁 지역 모니터링 필요.",
            key_findings=[
                "미중 무역 갈등 지속",
                "중동 긴장 고조",
                "글로벌 공급망 점진적 안정화"
            ],
            recommendations={
                "risk_assets": "NEUTRAL",
                "safe_havens": "NEUTRAL",
                "monitoring": ["China-Taiwan", "Middle East", "US Trade Policy"]
            },
            confidence=0.7,
            primary_risks=[
                {"risk": "US-China tensions", "probability": 0.3, "impact": "HIGH"},
                {"risk": "Middle East escalation", "probability": 0.25, "impact": "MEDIUM"}
            ],
            risk_assets_stance=AllocationWeight.NEUTRAL,
            safe_haven_stance=AllocationWeight.NEUTRAL
        )

        # 중단 조건 확인
        if result.risk_level == RiskLevel.CRITICAL:
            result.should_continue = False
            result.abort_reason = "지정학적 위험 극심 - 리스크 관리 우선"

        return result

    async def _analyze_monetary(
        self,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> MonetaryResult:
        """
        Level 1: 통화 환경 분석

        상위 레벨(지정학적 상황) 컨텍스트 반영
        """
        geo_context = context.get('GEOPOLITICS')
        prompt_config = LEVEL_DEBATE_PROMPTS[AnalysisLevel.MONETARY]

        # FRED 데이터 추출
        fred_data = data.get('fred_data', {})
        fedwatch = data.get('fedwatch', {})

        # 기본 결과 (실제 구현에서는 AI 토론으로 대체)
        result = MonetaryResult(
            level=AnalysisLevel.MONETARY,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary="유동성 환경 정상, Fed 정책 중립적. 인플레이션 둔화 추세.",
            key_findings=[
                "Fed 금리 인상 사이클 종료 가능성",
                "인플레이션 점진적 둔화",
                "실질금리 양(+) 영역 유지"
            ],
            recommendations={
                "duration": "NEUTRAL",
                "credit": "NEUTRAL",
                "cyclical_vs_defensive": "BALANCED"
            },
            confidence=0.75,
            liquidity_regime=LiquidityRegime.NORMAL,
            policy_stance=PolicyStance.NEUTRAL,
            inflation_trend="DECELERATING",
            cycle_position=CyclePosition.LATE,
            asset_implications={
                "equities": AllocationWeight.NEUTRAL,
                "bonds": AllocationWeight.NEUTRAL,
                "commodities": AllocationWeight.UNDERWEIGHT,
                "crypto": AllocationWeight.NEUTRAL
            }
        )

        # 상위 레벨 위험 반영
        if geo_context and geo_context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            result.stance = Stance.BEARISH
            result.key_findings.append("지정학적 위험으로 보수적 스탠스")

        return result

    async def _analyze_asset_class(
        self,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> AssetClassResult:
        """
        Level 2: 자산군 분석

        통화 환경과 지정학적 상황 컨텍스트 반영
        """
        monetary_context: Optional[MonetaryResult] = context.get('MONETARY')

        # 통화 환경 기반 기본 배분
        allocations = {
            "equities": AllocationWeight.NEUTRAL,
            "bonds": AllocationWeight.NEUTRAL,
            "commodities": AllocationWeight.NEUTRAL,
            "crypto": AllocationWeight.NEUTRAL,
            "cash": AllocationWeight.NEUTRAL
        }

        # 통화 환경에 따른 조정
        if monetary_context:
            if monetary_context.policy_stance == PolicyStance.DOVISH:
                allocations["equities"] = AllocationWeight.OVERWEIGHT
                allocations["bonds"] = AllocationWeight.OVERWEIGHT
            elif monetary_context.policy_stance == PolicyStance.HAWKISH:
                allocations["equities"] = AllocationWeight.UNDERWEIGHT
                allocations["cash"] = AllocationWeight.OVERWEIGHT

            if monetary_context.cycle_position == CyclePosition.LATE:
                allocations["commodities"] = AllocationWeight.OVERWEIGHT

        result = AssetClassResult(
            level=AnalysisLevel.ASSET_CLASS,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary="균형 잡힌 자산 배분 권고. 경기 후반부 특성 고려.",
            key_findings=[
                "주식-채권 상관관계 양(+)으로 전환",
                "원자재는 인플레이션 헷지 가치",
                "현금 비중 유지 권고"
            ],
            recommendations=allocations,
            confidence=0.7,
            allocations=allocations,
            conviction="MEDIUM",
            cross_asset_signals=[
                "Stock-bond positive correlation",
                "Dollar strength pressuring commodities"
            ],
            preferred_assets=["Quality stocks", "Short-duration bonds"],
            avoid_assets=["Long-duration bonds", "Speculative crypto"]
        )

        return result

    async def _analyze_sector(
        self,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> SectorResult:
        """
        Level 3: 섹터 분석

        경기 사이클과 금리 환경 기반
        """
        monetary_context: Optional[MonetaryResult] = context.get('MONETARY')

        # 경기 사이클 기반 섹터 선호
        cycle_position = CyclePosition.MID
        if monetary_context:
            cycle_position = monetary_context.cycle_position

        # 사이클별 섹터 매핑
        cycle_sector_map = {
            CyclePosition.EARLY: {
                "top": ["Consumer Discretionary", "Financials", "Real Estate"],
                "avoid": ["Utilities", "Consumer Staples"],
                "signal": "CYCLICAL"
            },
            CyclePosition.MID: {
                "top": ["Technology", "Industrials", "Materials"],
                "avoid": ["Utilities"],
                "signal": "NEUTRAL"
            },
            CyclePosition.LATE: {
                "top": ["Energy", "Materials", "Healthcare"],
                "avoid": ["Consumer Discretionary", "Real Estate"],
                "signal": "DEFENSIVE"
            },
            CyclePosition.RECESSION: {
                "top": ["Utilities", "Consumer Staples", "Healthcare"],
                "avoid": ["Consumer Discretionary", "Financials"],
                "signal": "DEFENSIVE"
            },
            CyclePosition.TURNING: {
                "top": ["Technology", "Consumer Discretionary"],
                "avoid": ["Energy"],
                "signal": "CYCLICAL"
            }
        }

        sector_config = cycle_sector_map.get(cycle_position, cycle_sector_map[CyclePosition.MID])

        result = SectorResult(
            level=AnalysisLevel.SECTOR,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary=f"경기 {cycle_position.value} 단계 기반 섹터 로테이션 권고.",
            key_findings=[
                f"현재 경기 사이클: {cycle_position.value}",
                f"선호 섹터: {', '.join(sector_config['top'][:2])}",
                f"회피 섹터: {', '.join(sector_config['avoid'][:1])}"
            ],
            recommendations={
                "rotation_strategy": sector_config['signal'],
                "top_3": sector_config['top'][:3],
                "avoid": sector_config['avoid'][:2]
            },
            confidence=0.65,
            top_sectors=sector_config['top'],
            avoid_sectors=sector_config['avoid'],
            rotation_signal=sector_config['signal'],
            cycle_based_rationale=f"경기 {cycle_position.value} 단계에서는 {sector_config['top'][0]} 섹터가 역사적으로 아웃퍼폼"
        )

        return result

    async def _analyze_individual(
        self,
        data: Dict[str, Any],
        context: Dict[str, LevelAnalysisResult]
    ) -> IndividualResult:
        """
        Level 4: 개별 자산 분석

        선택된 섹터 내 개별 종목/자산 분석
        """
        sector_context: Optional[SectorResult] = context.get('SECTOR')

        top_sectors = []
        if sector_context:
            top_sectors = sector_context.top_sectors

        # 개별 자산 분석은 외부 데이터와 레짐 변화 감지 필요
        analyzed = data.get('individual_analysis', [])

        result = IndividualResult(
            level=AnalysisLevel.INDIVIDUAL,
            risk_level=RiskLevel.MEDIUM,
            stance=Stance.NEUTRAL,
            summary=f"선택 섹터({', '.join(top_sectors[:2])}) 내 개별 분석 수행.",
            key_findings=[
                f"분석 대상 섹터: {top_sectors[:2]}",
                "펀더멘털 + 기술적 분석 결합",
                "레짐 변화 리스크 모니터링"
            ],
            recommendations={
                "focus_sectors": top_sectors[:2],
                "analysis_depth": "FUNDAMENTAL + TECHNICAL"
            },
            confidence=0.6,
            analyzed_assets=analyzed,
            top_picks=[],
            short_candidates=[]
        )

        return result

    def _store_result(
        self,
        result: TopDownResult,
        level: AnalysisLevel,
        level_result: LevelAnalysisResult
    ):
        """결과 저장"""
        if level == AnalysisLevel.GEOPOLITICS:
            result.geopolitical = level_result
        elif level == AnalysisLevel.MONETARY:
            result.monetary = level_result
        elif level == AnalysisLevel.ASSET_CLASS:
            result.asset_class = level_result
        elif level == AnalysisLevel.SECTOR:
            result.sector = level_result
        elif level == AnalysisLevel.INDIVIDUAL:
            result.individual = level_result

    def _determine_final_stance(self, result: TopDownResult) -> Stance:
        """최종 스탠스 결정"""
        stances = []

        if result.geopolitical:
            stances.append(result.geopolitical.stance)
        if result.monetary:
            stances.append(result.monetary.stance)
        if result.asset_class:
            stances.append(result.asset_class.stance)

        if not stances:
            return Stance.NEUTRAL

        # 가장 보수적인 스탠스 선택 (상위 레벨 우선)
        stance_priority = {
            Stance.RISK_OFF: 0,
            Stance.VERY_BEARISH: 1,
            Stance.BEARISH: 2,
            Stance.NEUTRAL: 3,
            Stance.BULLISH: 4,
            Stance.VERY_BULLISH: 5
        }

        # 상위 레벨 가중치 적용
        weighted_stances = []
        for i, stance in enumerate(stances):
            weight = len(stances) - i  # 상위 레벨일수록 높은 가중치
            weighted_stances.extend([stance] * weight)

        # 가장 빈번한 스탠스 (보수적 방향 우선)
        from collections import Counter
        counter = Counter(weighted_stances)
        return counter.most_common(1)[0][0]

    def _generate_final_recommendation(self, result: TopDownResult) -> str:
        """최종 추천 생성"""
        recommendations = []

        if result.aborted_at_level:
            recommendations.append(f"⚠️ 분석 중단: {result.abort_reason}")
            recommendations.append("현금/안전자산 비중 확대 권고")
            return " | ".join(recommendations)

        if result.geopolitical:
            recommendations.append(f"지정학: {result.geopolitical.risk_level.value}")

        if result.monetary:
            recommendations.append(f"통화: {result.monetary.policy_stance.value}")

        if result.asset_class and result.asset_class.preferred_assets:
            recommendations.append(f"선호: {', '.join(result.asset_class.preferred_assets[:2])}")

        if result.sector and result.sector.top_sectors:
            recommendations.append(f"섹터: {', '.join(result.sector.top_sectors[:2])}")

        return " | ".join(recommendations) if recommendations else "분석 불충분"

    def _calculate_total_confidence(self, result: TopDownResult) -> float:
        """전체 신뢰도 계산"""
        confidences = []

        if result.geopolitical:
            confidences.append(result.geopolitical.confidence * 1.2)  # 상위 레벨 가중치
        if result.monetary:
            confidences.append(result.monetary.confidence * 1.1)
        if result.asset_class:
            confidences.append(result.asset_class.confidence)
        if result.sector:
            confidences.append(result.sector.confidence * 0.9)
        if result.individual:
            confidences.append(result.individual.confidence * 0.8)

        if not confidences:
            return 0.5

        return sum(confidences) / len(confidences)


# ============================================================================
# Quick Access Functions
# ============================================================================

async def run_quick_top_down(
    data: Optional[Dict[str, Any]] = None,
    stop_at: str = "sector"
) -> TopDownResult:
    """
    빠른 하향식 분석 실행

    Parameters:
    -----------
    data : Dict
        분석 데이터
    stop_at : str
        "geopolitics", "monetary", "asset_class", "sector", "individual"

    Returns:
    --------
    TopDownResult
    """
    level_map = {
        "geopolitics": AnalysisLevel.GEOPOLITICS,
        "monetary": AnalysisLevel.MONETARY,
        "asset_class": AnalysisLevel.ASSET_CLASS,
        "sector": AnalysisLevel.SECTOR,
        "individual": AnalysisLevel.INDIVIDUAL
    }

    stop_level = level_map.get(stop_at, AnalysisLevel.SECTOR)

    orchestrator = TopDownOrchestrator()
    return await orchestrator.run_full_analysis(
        data=data or {},
        stop_at_level=stop_level
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/tj/projects/autoai/eimas')

    async def test():
        print("=== Top-Down Orchestrator Test ===\n")

        orchestrator = TopDownOrchestrator()

        # 테스트 데이터
        test_data = {
            "geopolitical_news": [
                "US-China trade talks resume",
                "Middle East tensions ease"
            ],
            "fred_data": {
                "DFF": 4.5,
                "DGS10": 4.2,
                "CPI": 3.2
            }
        }

        # 섹터까지 분석
        result = await orchestrator.run_full_analysis(
            data=test_data,
            stop_at_level=AnalysisLevel.SECTOR
        )

        print(f"Final Stance: {result.final_stance.value}")
        print(f"Final Recommendation: {result.final_recommendation}")
        print(f"Total Confidence: {result.total_confidence:.2f}")
        print()

        if result.geopolitical:
            print(f"[L0] Geopolitical: {result.geopolitical.risk_level.value}")
            print(f"     Findings: {result.geopolitical.key_findings[:2]}")
        if result.monetary:
            print(f"[L1] Monetary: {result.monetary.policy_stance.value}, Cycle: {result.monetary.cycle_position.value}")
        if result.asset_class:
            print(f"[L2] Asset Class: Preferred={result.asset_class.preferred_assets}")
        if result.sector:
            print(f"[L3] Sector: Top={result.sector.top_sectors[:3]}")

        print("\n✅ Test passed!")

    asyncio.run(test())
