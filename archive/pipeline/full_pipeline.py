"""
Full Pipeline Runner - 전체 파이프라인 통합

데이터 수집 → Top-Down 분석 → 방법론 선택 → 분석 실행 →
경제학파별 해석 → 전략 생성 → 최종 종합

Author: EIMAS Team
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

# Core imports
from core.schemas import AgentRole

# Agent imports
from agents.top_down_orchestrator import (
    TopDownOrchestrator,
    TopDownResult,
    AnalysisLevel,
    RiskLevel,
    Stance
)
from agents.methodology_debate import (
    MethodologyDebateAgent,
    MethodologyDecision,
    MethodologyType,
    ResearchGoal,
    DataSummary
)
from agents.interpretation_debate import (
    InterpretationDebateAgent,
    InterpretationConsensus,
    AnalysisResult,
    EconomicSchool
)
from agents.strategy_agent import (
    StrategyAgent,
    TradeRecommendation
)

# Lib imports
from lib.causal_network import (
    CausalNetworkAnalyzer,
    NetworkAnalysisResult
)

# Regime Change imports
from agents.regime_change import (
    RegimeChangeDetectionPipeline,
    RegimeChangeResult,
    VolumeBreakoutDetector,
    ImpactDuration
)
import pandas as pd


# ============================================================================
# Pipeline Stages
# ============================================================================

class PipelineStage(Enum):
    """파이프라인 단계"""
    DATA_COLLECTION = "data_collection"
    TOP_DOWN_ANALYSIS = "top_down_analysis"
    REGIME_CHECK = "regime_check"  # Stage 2.5: 레짐 변화 감지
    METHODOLOGY_SELECTION = "methodology_selection"
    CORE_ANALYSIS = "core_analysis"
    INTERPRETATION = "interpretation"
    STRATEGY_GENERATION = "strategy_generation"
    SYNTHESIS = "synthesis"


class RegimeType(Enum):
    """레짐 유형"""
    EXPANSION = "expansion"       # 확장기
    CONTRACTION = "contraction"   # 수축기
    TRANSITION = "transition"     # 전환기
    CRISIS = "crisis"            # 위기
    STABLE = "stable"            # 안정 (레짐 변화 없음)


class PipelineStatus(Enum):
    """파이프라인 상태"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StageResult:
    """단계별 결과"""
    stage: PipelineStage
    status: PipelineStatus
    result: Any
    duration_seconds: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeContext:
    """레짐 변화 컨텍스트 (Stage 2.5 결과)"""
    regime_type: RegimeType
    regime_changes: List[RegimeChangeResult]
    regime_aware: bool  # 레짐 변화 감지 여부
    context_adjustment: Dict[str, Any]  # 하위 분석 단계에 전달할 조정 지침
    data_split_date: Optional[datetime] = None  # 데이터 분리 기준일
    use_post_regime_only: bool = False  # 레짐 변화 이후 데이터만 사용 여부
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화용"""
        return {
            "regime_type": self.regime_type.value,
            "regime_aware": self.regime_aware,
            "changes_detected": len(self.regime_changes),
            "context_adjustment": self.context_adjustment,
            "data_split_date": str(self.data_split_date) if self.data_split_date else None,
            "use_post_regime_only": self.use_post_regime_only,
            "confidence": self.confidence
        }


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 분석 범위
    stop_at_level: AnalysisLevel = AnalysisLevel.SECTOR
    skip_stages: List[PipelineStage] = field(default_factory=list)

    # 분석 파라미터
    research_goal: ResearchGoal = ResearchGoal.VARIABLE_SELECTION
    risk_tolerance: str = "moderate"

    # 실행 옵션
    verbose: bool = True
    save_intermediate: bool = True
    max_retries: int = 2


@dataclass
class PipelineResult:
    """전체 파이프라인 결과"""
    # 단계별 결과
    stages: Dict[PipelineStage, StageResult] = field(default_factory=dict)

    # 핵심 결과
    top_down: Optional[TopDownResult] = None
    regime_context: Optional[RegimeContext] = None  # Stage 2.5 결과
    methodology: Optional[MethodologyDecision] = None
    analysis: Optional[Any] = None
    interpretation: Optional[InterpretationConsensus] = None
    strategy: Optional[TradeRecommendation] = None

    # 최종 종합
    final_recommendation: str = ""
    executive_summary: str = ""
    confidence: float = 0.0

    # 메타 정보
    total_duration_seconds: float = 0.0
    status: PipelineStatus = PipelineStatus.NOT_STARTED
    aborted_at: Optional[PipelineStage] = None
    abort_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Mock Data Provider
# ============================================================================

class MockDataProvider:
    """
    테스트용 가상 데이터 제공

    실제 API 연동 전 파이프라인 테스트용
    """

    @staticmethod
    def get_market_data() -> Dict[str, Any]:
        """시장 데이터 (Mock)"""
        return {
            "fred_data": {
                "DFF": 4.50,           # Fed Funds Rate
                "DGS10": 4.25,         # 10Y Treasury
                "DGS2": 4.10,          # 2Y Treasury
                "T10Y2Y": 0.15,        # Yield Curve
                "BAMLH0A0HYM2": 3.50,  # HY Spread
                "UNRATE": 4.1,         # Unemployment
                "CPIAUCSL": 3.2,       # CPI YoY
                "PCEPILFE": 2.8,       # Core PCE
                "VIXCLS": 18.5,        # VIX
            },
            "indicators": {
                "gdp_growth": 2.5,
                "inflation": 3.2,
                "unemployment": 4.1,
                "consumer_sentiment": 68.5,
                "pmi_manufacturing": 49.2,
                "pmi_services": 52.1,
            },
            "prices": {
                "SPY": 480.50,
                "QQQ": 410.25,
                "TLT": 92.30,
                "GLD": 195.80,
                "BTC": 43500.0,
            },
            "volatility": {
                "vix": 18.5,
                "vix_1m": 17.2,
                "vix_3m": 19.1,
                "realized_vol_20d": 15.3,
            }
        }

    @staticmethod
    def get_geopolitical_data() -> Dict[str, Any]:
        """지정학적 데이터 (Mock)"""
        return {
            "geopolitical_news": [
                {"headline": "US-China trade talks continue", "sentiment": "neutral"},
                {"headline": "Middle East tensions ease slightly", "sentiment": "positive"},
                {"headline": "European energy prices stabilize", "sentiment": "positive"},
            ],
            "risk_indices": {
                "geopolitical_risk_index": 125,  # 100 = baseline
                "trade_policy_uncertainty": 110,
                "conflict_probability": 0.15,
            }
        }

    @staticmethod
    def get_fedwatch_data() -> Dict[str, Any]:
        """FedWatch 데이터 (Mock)"""
        return {
            "next_fomc": "2025-01-29",
            "days_to_meeting": 32,
            "current_rate_bp": 450,
            "expected_rate_bp": 450,
            "probabilities": {
                425: 0.15,
                450: 0.70,
                475: 0.15,
            },
            "rate_change_expected": 0,
            "uncertainty_index": 0.35,
        }

    @staticmethod
    def get_sentiment_data() -> Dict[str, Any]:
        """심리 데이터 (Mock)"""
        return {
            "fear_greed_index": 55,  # 0-100, 50=neutral
            "put_call_ratio": 0.85,
            "vix_term_structure": "contango",
            "retail_sentiment": 0.2,  # -1 to 1
            "institutional_flow": 0.1,
        }

    @staticmethod
    def get_data_summary() -> DataSummary:
        """데이터 요약 (방법론 선택용)"""
        return DataSummary(
            n_observations=1000,
            n_variables=50,
            time_range="2020-01-01 to 2024-12-31",
            frequency="daily",
            missing_ratio=0.02,
            stationarity={"DFF": True, "SPY": False, "VIX": True},
            variable_categories={
                "rates": ["DFF", "DGS10", "DGS2"],
                "prices": ["SPY", "QQQ", "TLT"],
                "volatility": ["VIX"]
            }
        )

    @classmethod
    def get_all_data(cls) -> Dict[str, Any]:
        """전체 데이터 통합"""
        return {
            **cls.get_market_data(),
            **cls.get_geopolitical_data(),
            "fedwatch": cls.get_fedwatch_data(),
            "sentiment": cls.get_sentiment_data(),
        }


# ============================================================================
# Full Pipeline Runner
# ============================================================================

class FullPipelineRunner:
    """
    전체 파이프라인 실행기

    데이터 수집 → Top-Down 분석 → 방법론 선택 → 분석 실행 →
    경제학파별 해석 → 전략 생성 → 최종 종합

    사용법:
    ```python
    # Mock 모드 (기본 - API 호출 없음)
    runner = FullPipelineRunner()
    result = await runner.run(...)

    # Real 모드 (실제 AI 토론)
    runner = FullPipelineRunner(use_mock=False)
    result = await runner.run(...)
    ```
    """

    def __init__(self, verbose: bool = True, use_mock: bool = True):
        """
        Parameters:
        -----------
        verbose : bool
            상세 로깅 활성화
        use_mock : bool
            True: Mock 응답 사용 (빠름, API 비용 없음)
            False: 실제 AI 토론 실행 (Claude, OpenAI 등)
        """
        self.verbose = verbose
        self.use_mock = use_mock

        # 에이전트 초기화 (lazy)
        self._top_down: Optional[TopDownOrchestrator] = None
        self._methodology: Optional[MethodologyDebateAgent] = None
        self._interpretation: Optional[InterpretationDebateAgent] = None
        self._strategy: Optional[StrategyAgent] = None
        self._causal: Optional[CausalNetworkAnalyzer] = None
        self._regime_pipeline: Optional[RegimeChangeDetectionPipeline] = None

    @property
    def top_down(self) -> TopDownOrchestrator:
        if self._top_down is None:
            self._top_down = TopDownOrchestrator()
        return self._top_down

    @property
    def methodology(self) -> MethodologyDebateAgent:
        if self._methodology is None:
            self._methodology = MethodologyDebateAgent()
        return self._methodology

    @property
    def interpretation(self) -> InterpretationDebateAgent:
        if self._interpretation is None:
            self._interpretation = InterpretationDebateAgent()
        return self._interpretation

    @property
    def strategy(self) -> StrategyAgent:
        if self._strategy is None:
            self._strategy = StrategyAgent()
        return self._strategy

    @property
    def regime_pipeline(self) -> RegimeChangeDetectionPipeline:
        if self._regime_pipeline is None:
            self._regime_pipeline = RegimeChangeDetectionPipeline(verbose=self.verbose)
        return self._regime_pipeline

    def _log(self, message: str):
        """로깅"""
        if self.verbose:
            print(f"[Pipeline] {message}")

    async def run(
        self,
        research_question: str,
        data: Optional[Dict[str, Any]] = None,
        config: Optional[PipelineConfig] = None
    ) -> PipelineResult:
        """
        전체 파이프라인 실행

        Parameters:
        -----------
        research_question : str
            연구 질문
        data : Dict
            입력 데이터 (None이면 Mock 사용)
        config : PipelineConfig
            파이프라인 설정

        Returns:
        --------
        PipelineResult
            전체 파이프라인 결과
        """
        start_time = datetime.now()
        config = config or PipelineConfig()
        result = PipelineResult()

        self._log(f"Starting pipeline for: {research_question}")
        self._log(f"Config: stop_at={config.stop_at_level.name}, goal={config.research_goal.value}")

        try:
            # Stage 1: Data Collection
            if PipelineStage.DATA_COLLECTION not in config.skip_stages:
                stage_result = await self._run_data_collection(data)
                result.stages[PipelineStage.DATA_COLLECTION] = stage_result
                data = stage_result.result

            # Stage 2: Top-Down Analysis
            if PipelineStage.TOP_DOWN_ANALYSIS not in config.skip_stages:
                stage_result = await self._run_top_down(data, config)
                result.stages[PipelineStage.TOP_DOWN_ANALYSIS] = stage_result
                result.top_down = stage_result.result

                # 상위 레벨 위험 시 중단
                if result.top_down and result.top_down.aborted_at_level:
                    result.aborted_at = PipelineStage.TOP_DOWN_ANALYSIS
                    result.abort_reason = result.top_down.abort_reason
                    result.status = PipelineStatus.ABORTED
                    self._log(f"Aborted at Top-Down: {result.abort_reason}")
                    return self._finalize_result(result, start_time)

            # Stage 2.5: Regime Check
            if PipelineStage.REGIME_CHECK not in config.skip_stages:
                stage_result = await self._run_regime_check(data, result.top_down, config)
                result.stages[PipelineStage.REGIME_CHECK] = stage_result
                result.regime_context = stage_result.result

            # Stage 3: Methodology Selection
            if PipelineStage.METHODOLOGY_SELECTION not in config.skip_stages:
                stage_result = await self._run_methodology_selection(
                    research_question, data, config
                )
                result.stages[PipelineStage.METHODOLOGY_SELECTION] = stage_result
                result.methodology = stage_result.result

            # Stage 4: Core Analysis
            if PipelineStage.CORE_ANALYSIS not in config.skip_stages:
                stage_result = await self._run_core_analysis(
                    data, result.methodology, result.top_down
                )
                result.stages[PipelineStage.CORE_ANALYSIS] = stage_result
                result.analysis = stage_result.result

            # Stage 5: Interpretation
            if PipelineStage.INTERPRETATION not in config.skip_stages:
                stage_result = await self._run_interpretation(
                    research_question, result.analysis, result.methodology
                )
                result.stages[PipelineStage.INTERPRETATION] = stage_result
                result.interpretation = stage_result.result

            # Stage 6: Strategy Generation
            if PipelineStage.STRATEGY_GENERATION not in config.skip_stages:
                stage_result = await self._run_strategy(
                    data, result.top_down, result.interpretation, config
                )
                result.stages[PipelineStage.STRATEGY_GENERATION] = stage_result
                result.strategy = stage_result.result

            # Stage 7: Synthesis
            if PipelineStage.SYNTHESIS not in config.skip_stages:
                stage_result = await self._run_synthesis(
                    research_question, result
                )
                result.stages[PipelineStage.SYNTHESIS] = stage_result
                synthesis = stage_result.result
                result.final_recommendation = synthesis.get('recommendation', '')
                result.executive_summary = synthesis.get('summary', '')

            result.status = PipelineStatus.COMPLETED

        except Exception as e:
            result.status = PipelineStatus.FAILED
            self._log(f"Pipeline failed: {e}")
            raise

        return self._finalize_result(result, start_time)

    async def _run_data_collection(
        self,
        data: Optional[Dict[str, Any]]
    ) -> StageResult:
        """Stage 1: 데이터 수집"""
        start = datetime.now()
        self._log("Stage 1: Data Collection")

        if data is None:
            self._log("  Using mock data")
            data = MockDataProvider.get_all_data()
        else:
            self._log(f"  Using provided data ({len(data)} keys)")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.DATA_COLLECTION,
            status=PipelineStatus.COMPLETED,
            result=data,
            duration_seconds=duration
        )

    async def _run_top_down(
        self,
        data: Dict[str, Any],
        config: PipelineConfig
    ) -> StageResult:
        """Stage 2: Top-Down 분석"""
        start = datetime.now()
        self._log("Stage 2: Top-Down Analysis")

        result = await self.top_down.run_full_analysis(
            data=data,
            stop_at_level=config.stop_at_level
        )

        self._log(f"  Stance: {result.final_stance.value}")
        self._log(f"  Recommendation: {result.final_recommendation[:50]}...")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.TOP_DOWN_ANALYSIS,
            status=PipelineStatus.COMPLETED,
            result=result,
            duration_seconds=duration
        )

    # =========================================================================
    # Stage 2.5: Regime Check
    # =========================================================================

    async def _run_regime_check(
        self,
        data: Dict[str, Any],
        top_down: Optional[TopDownResult],
        config: PipelineConfig
    ) -> StageResult:
        """
        Stage 2.5: 레짐 변화 감지

        거래량 급변 → 뉴스 검색 → 분류 → 영향 평가 → 레짐 결정
        """
        start = datetime.now()
        self._log("Stage 2.5: Regime Check")

        if self.use_mock:
            return await self._run_regime_check_mock(data, top_down, start)
        else:
            return await self._run_regime_check_real(data, top_down, start)

    async def _run_regime_check_mock(
        self,
        data: Dict[str, Any],
        top_down: Optional[TopDownResult],
        start: datetime
    ) -> StageResult:
        """Stage 2.5: Regime Check (Mock)"""
        self._log("  (Mock mode - no actual regime detection)")

        # Mock 레짐 컨텍스트 - 레짐 변화 없음
        regime_context = RegimeContext(
            regime_type=RegimeType.STABLE,
            regime_changes=[],
            regime_aware=False,
            context_adjustment={
                "data_handling": "use_all_data",
                "analysis_note": "No structural break detected (mock)"
            },
            confidence=0.8
        )

        self._log(f"  Regime: {regime_context.regime_type.value}")
        self._log(f"  Confidence: {regime_context.confidence:.0%}")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.REGIME_CHECK,
            status=PipelineStatus.COMPLETED,
            result=regime_context,
            duration_seconds=duration
        )

    async def _run_regime_check_real(
        self,
        data: Dict[str, Any],
        top_down: Optional[TopDownResult],
        start: datetime
    ) -> StageResult:
        """Stage 2.5: Regime Check (Real API - Perplexity/Claude)"""
        self._log("  Analyzing regime changes (Real API)...")

        try:
            # 1. 가격 DataFrame 추출
            price_df = self._extract_price_dataframe(data)

            if price_df.empty:
                self._log("  No price data available for regime detection")
                return self._create_no_data_regime_result(start)

            # 2. 분석 대상 티커 선택
            tickers = self._select_tickers_for_regime(data, top_down)
            self._log(f"  Analyzing tickers: {tickers}")

            # 3. 각 티커에 대해 레짐 변화 감지 실행
            all_regime_changes: List[RegimeChangeResult] = []

            for ticker in tickers:
                ticker_data = self._get_ticker_dataframe(data, ticker)
                if ticker_data is not None and not ticker_data.empty:
                    company_info = self._get_company_info(ticker, data)
                    try:
                        changes = await self.regime_pipeline.run(
                            ticker, ticker_data, company_info
                        )
                        all_regime_changes.extend(changes)
                    except Exception as e:
                        self._log(f"  Warning: Regime check failed for {ticker}: {e}")

            # 4. 레짐 컨텍스트 종합
            regime_context = self._synthesize_regime_context(all_regime_changes)

            self._log(f"  Regime type: {regime_context.regime_type.value}")
            self._log(f"  Changes detected: {len(regime_context.regime_changes)}")
            self._log(f"  Confidence: {regime_context.confidence:.0%}")

        except Exception as e:
            self._log(f"  Regime detection failed: {e}")
            regime_context = RegimeContext(
                regime_type=RegimeType.STABLE,
                regime_changes=[],
                regime_aware=False,
                context_adjustment={"error": str(e)},
                confidence=0.0
            )

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.REGIME_CHECK,
            status=PipelineStatus.COMPLETED,
            result=regime_context,
            duration_seconds=duration
        )

    def _extract_price_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Dict에서 가격 DataFrame 추출"""
        # 직접 DataFrame이 있는 경우
        if 'price_df' in data:
            return data['price_df']

        # market_data Dict가 있는 경우
        if 'market_data' in data:
            market_data = data['market_data']
            if isinstance(market_data, pd.DataFrame):
                return market_data

            # Dict[ticker, DataFrame] 형식인 경우
            if isinstance(market_data, dict):
                closes = {}
                volumes = {}
                for ticker, df in market_data.items():
                    if isinstance(df, pd.DataFrame):
                        if 'Close' in df.columns:
                            closes[ticker] = df['Close']
                        if 'Volume' in df.columns:
                            volumes[ticker] = df['Volume']

                if closes:
                    result = pd.DataFrame(closes)
                    if volumes:
                        vol_df = pd.DataFrame(volumes)
                        vol_df.columns = [f"{c}_Volume" for c in vol_df.columns]
                        result = result.join(vol_df)
                    return result

        return pd.DataFrame()

    def _get_ticker_dataframe(
        self,
        data: Dict[str, Any],
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """특정 티커의 DataFrame 반환"""
        if 'market_data' in data:
            market_data = data['market_data']
            if isinstance(market_data, dict) and ticker in market_data:
                return market_data[ticker]
        return None

    def _get_company_info(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """티커의 회사 정보 반환"""
        if 'company_info' in data and ticker in data['company_info']:
            return data['company_info'][ticker]

        # 기본 회사 정보
        return {
            'name': ticker,
            'ticker': ticker,
            'industry': 'Unknown',
            'market_cap': 0
        }

    def _select_tickers_for_regime(
        self,
        data: Dict[str, Any],
        top_down: Optional[TopDownResult]
    ) -> List[str]:
        """분석 대상 티커 선택"""
        default_tickers = ['SPY', 'QQQ', 'TLT', 'GLD']

        # Top-Down 결과에서 선호 자산 가져오기
        if top_down and hasattr(top_down, 'asset_class') and top_down.asset_class:
            if hasattr(top_down.asset_class, 'preferred_assets'):
                preferred = top_down.asset_class.preferred_assets
                if preferred:
                    return preferred[:5]

        # 사용 가능한 시장 데이터에서 선택
        if 'market_data' in data and isinstance(data['market_data'], dict):
            available = list(data['market_data'].keys())
            return [t for t in default_tickers if t in available][:5] or available[:5]

        return default_tickers

    def _synthesize_regime_context(
        self,
        regime_changes: List[RegimeChangeResult]
    ) -> RegimeContext:
        """레짐 변화 결과를 컨텍스트로 종합"""
        if not regime_changes:
            return RegimeContext(
                regime_type=RegimeType.STABLE,
                regime_changes=[],
                regime_aware=False,
                context_adjustment={
                    "data_handling": "use_all_data",
                    "analysis_note": "No structural breaks detected"
                },
                confidence=0.8
            )

        # 구조적 변화만 필터링
        structural_changes = [r for r in regime_changes if r.is_regime_change]

        if not structural_changes:
            return RegimeContext(
                regime_type=RegimeType.STABLE,
                regime_changes=regime_changes,
                regime_aware=True,
                context_adjustment={
                    "data_handling": "use_all_data",
                    "analysis_note": "Events detected but no structural break"
                },
                confidence=0.7
            )

        # 가장 최근 구조적 변화
        latest = max(
            structural_changes,
            key=lambda r: r.change_date or datetime.min
        )

        # 레짐 유형 결정
        regime_type = RegimeType.TRANSITION
        if latest.after_regime:
            characteristics = latest.after_regime.characteristics
            sentiment = characteristics.get('sentiment', '').upper()
            if sentiment == 'POSITIVE':
                regime_type = RegimeType.EXPANSION
            elif sentiment == 'NEGATIVE':
                regime_type = RegimeType.CONTRACTION

        return RegimeContext(
            regime_type=regime_type,
            regime_changes=structural_changes,
            regime_aware=True,
            context_adjustment={
                "data_handling": "split_at_regime_change",
                "analysis_note": latest.analysis_instruction,
                "focus_period": "post_regime"
            },
            data_split_date=latest.change_date,
            use_post_regime_only=True,
            confidence=0.75
        )

    def _create_no_data_regime_result(self, start: datetime) -> StageResult:
        """데이터 없을 때 기본 결과 반환"""
        regime_context = RegimeContext(
            regime_type=RegimeType.STABLE,
            regime_changes=[],
            regime_aware=False,
            context_adjustment={
                "data_handling": "use_all_data",
                "analysis_note": "No price data for regime detection"
            },
            confidence=0.5
        )

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.REGIME_CHECK,
            status=PipelineStatus.COMPLETED,
            result=regime_context,
            duration_seconds=duration
        )

    async def _run_methodology_selection(
        self,
        research_question: str,
        data: Dict[str, Any],
        config: PipelineConfig
    ) -> StageResult:
        """Stage 3: 방법론 선택"""
        start = datetime.now()

        if self.use_mock:
            return await self._run_methodology_selection_mock(
                research_question, data, config, start
            )
        else:
            return await self._run_methodology_selection_real(
                research_question, data, config, start
            )

    async def _run_methodology_selection_mock(
        self,
        research_question: str,
        data: Dict[str, Any],
        config: PipelineConfig,
        start: datetime
    ) -> StageResult:
        """Stage 3: 방법론 선택 (Mock)"""
        self._log("Stage 3: Methodology Selection (Mock)")

        # Mock 결과 (실제 AI 토론 없이)
        decision = MethodologyDecision(
            selected_methodology=MethodologyType.LASSO,
            components=[MethodologyType.LASSO],
            pipeline=[
                "1. 데이터 전처리 (결측치 처리, 정규화)",
                "2. LASSO 변수 선택 (alpha=0.01)",
                "3. 시계열 교차검증 (5-fold)",
                "4. 모델 학습 및 해석"
            ],
            parameters={"alpha": 0.01, "cv_folds": 5},
            validation="time_series_cv",
            confidence=0.82,
            rationale="LASSO는 변수 선택과 회귀를 동시에 수행하며, sparsity를 통해 해석 가능한 모델을 제공합니다.",
            dissenting_views=["VAR은 다변량 관계 분석에 적합", "Granger는 인과관계 검정에 강점"]
        )

        self._log(f"  Selected: {decision.selected_methodology.value}")
        self._log(f"  Confidence: {decision.confidence:.0%}")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.METHODOLOGY_SELECTION,
            status=PipelineStatus.COMPLETED,
            result=decision,
            duration_seconds=duration
        )

    async def _run_methodology_selection_real(
        self,
        research_question: str,
        data: Dict[str, Any],
        config: PipelineConfig,
        start: datetime
    ) -> StageResult:
        """Stage 3: 방법론 선택 (Real AI Debate)"""
        self._log("Stage 3: Methodology Selection (Real AI Debate)")

        try:
            # 데이터 요약 생성
            data_summary = MockDataProvider.get_data_summary()

            # 실제 AI 토론으로 방법론 선택
            decision = await self.methodology.debate_methodology(
                research_question=research_question,
                research_goal=config.research_goal,
                data_summary=data_summary
            )

            self._log(f"  Selected: {decision.selected_methodology.value}")
            self._log(f"  Confidence: {decision.confidence:.0%}")

        except Exception as e:
            self._log(f"  Real debate failed, falling back to mock: {e}")
            # 실패 시 Mock으로 폴백
            return await self._run_methodology_selection_mock(
                research_question, data, config, start
            )

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.METHODOLOGY_SELECTION,
            status=PipelineStatus.COMPLETED,
            result=decision,
            duration_seconds=duration
        )

    async def _run_core_analysis(
        self,
        data: Dict[str, Any],
        methodology: Optional[MethodologyDecision],
        top_down: Optional[TopDownResult]
    ) -> StageResult:
        """Stage 4: 핵심 분석"""
        start = datetime.now()
        self._log("Stage 4: Core Analysis")

        # 방법론에 따른 분석 실행 (Mock)
        selected = methodology.selected_methodology if methodology else MethodologyType.LASSO

        analysis_result = {
            "methodology": selected.value,
            "key_findings": [
                "Fed 금리가 주식 시장에 -0.42 영향",
                "인플레이션 기대가 채권 수익률에 +0.65 영향",
                "달러 강세가 원자재에 -0.38 영향"
            ],
            "statistics": {
                "r_squared": 0.72,
                "mse": 0.015,
                "n_significant_vars": 8,
                "p_value": 0.001
            },
            "predictions": {
                "fed_rate_3m": 4.50,
                "fed_rate_6m": 4.25,
                "spy_return_1m": 0.02,
            },
            "confidence": 0.75,
            "top_down_context": {
                "stance": top_down.final_stance.value if top_down else "NEUTRAL",
                "cycle": top_down.monetary.cycle_position.value if top_down and top_down.monetary else "MID"
            }
        }

        self._log(f"  R-squared: {analysis_result['statistics']['r_squared']:.2f}")
        self._log(f"  Key vars: {analysis_result['statistics']['n_significant_vars']}")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.CORE_ANALYSIS,
            status=PipelineStatus.COMPLETED,
            result=analysis_result,
            duration_seconds=duration
        )

    async def _run_interpretation(
        self,
        research_question: str,
        analysis: Optional[Dict[str, Any]],
        methodology: Optional[MethodologyDecision]
    ) -> StageResult:
        """Stage 5: 경제학파별 해석"""
        start = datetime.now()

        if self.use_mock:
            return await self._run_interpretation_mock(
                research_question, analysis, methodology, start
            )
        else:
            return await self._run_interpretation_real(
                research_question, analysis, methodology, start
            )

    async def _run_interpretation_mock(
        self,
        research_question: str,
        analysis: Optional[Dict[str, Any]],
        methodology: Optional[MethodologyDecision],
        start: datetime
    ) -> StageResult:
        """Stage 5: 경제학파별 해석 (Mock)"""
        self._log("Stage 5: Interpretation (Mock Multi-School)")

        # Mock 해석 결과 생성
        from agents.interpretation_debate import SchoolInterpretation

        school_interpretations = [
            SchoolInterpretation(
                school=EconomicSchool.MONETARIST,
                interpretation="Fed의 통화정책이 시장의 핵심 동인. 금리 인상 사이클 종료 시 유동성 개선 기대.",
                key_points=["M2 성장률 둔화", "실질금리 양(+) 영역", "인플레이션 통제 진행 중"],
                policy_implications=["긴축 종료 시 리스크 자산 회복", "채권 듀레이션 확대 고려"],
                risk_assessment="인플레이션 재발 시 추가 긴축 리스크",
                confidence=0.75,
                supporting_theory="Quantity Theory of Money"
            ),
            SchoolInterpretation(
                school=EconomicSchool.KEYNESIAN,
                interpretation="총수요 둔화로 경기 하방 리스크. 재정/통화정책 완화 필요성 증가.",
                key_points=["소비자 심리 약화", "투자 둔화", "산출 갭 확대"],
                policy_implications=["재정 확대로 수요 보완", "금리 인하로 투자 촉진"],
                risk_assessment="과도한 긴축 시 경기 침체 우려",
                confidence=0.72,
                supporting_theory="Aggregate Demand Theory"
            ),
            SchoolInterpretation(
                school=EconomicSchool.AUSTRIAN,
                interpretation="장기간 저금리로 형성된 버블 조정 불가피. 구조조정 필요.",
                key_points=["신용 사이클 후반부", "자산 가격 고평가", "부채 수준 과다"],
                policy_implications=["인위적 부양책 지양", "시장 자율 조정 허용"],
                risk_assessment="급격한 조정 시 시스템 리스크",
                confidence=0.68,
                supporting_theory="Austrian Business Cycle Theory"
            ),
            SchoolInterpretation(
                school=EconomicSchool.TECHNICAL,
                interpretation="주요 지수 200일선 위 유지. 상승 추세 지속 중이나 모멘텀 둔화.",
                key_points=["RSI 중립 영역", "거래량 감소", "VIX 안정"],
                policy_implications=["추세 추종 전략 유지", "지지선 이탈 시 방어"],
                risk_assessment="기술적 저항선 돌파 실패 시 조정",
                confidence=0.70,
                supporting_theory="Technical Analysis"
            )
        ]

        consensus = InterpretationConsensus(
            topic=research_question,
            consensus_points=[
                "현재 통화정책 사이클은 후반부에 위치",
                "인플레이션 통제가 진행 중이나 완전하지 않음",
                "경기 둔화 신호가 나타나고 있음"
            ],
            divergence_points=[
                "[Monetarist] 인플레이션이 핵심 리스크 - 긴축 유지 필요",
                "[Keynesian] 경기 침체가 핵심 리스크 - 완화 필요",
                "[Austrian] 구조조정이 불가피 - 개입 자제 필요"
            ],
            school_interpretations=school_interpretations,
            recommended_action="중립적 포지션 유지, 경기 지표 모니터링",
            risk_factors=[
                "[Monetarist] 인플레이션 재발",
                "[Keynesian] 경기 침체",
                "[Austrian] 버블 붕괴",
                "[Technical] 기술적 지지선 붕괴"
            ],
            confidence=0.72,
            summary="경제학파별로 상이한 해석. 통화주의는 인플레이션, 케인즈는 경기 침체를 우려."
        )

        self._log(f"  Schools: {len(consensus.school_interpretations)}")
        self._log(f"  Consensus points: {len(consensus.consensus_points)}")
        self._log(f"  Divergence points: {len(consensus.divergence_points)}")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.INTERPRETATION,
            status=PipelineStatus.COMPLETED,
            result=consensus,
            duration_seconds=duration
        )

    async def _run_interpretation_real(
        self,
        research_question: str,
        analysis: Optional[Dict[str, Any]],
        methodology: Optional[MethodologyDecision],
        start: datetime
    ) -> StageResult:
        """Stage 5: 경제학파별 해석 (Real AI Debate)"""
        self._log("Stage 5: Interpretation (Real AI Debate)")

        try:
            # 분석 결과를 AnalysisResult로 변환
            if analysis:
                analysis_result = AnalysisResult(
                    topic=research_question,
                    methodology=methodology.selected_methodology.value if methodology else "LASSO",
                    key_findings=analysis.get('key_findings', []),
                    statistics=analysis.get('statistics', {}),
                    predictions=analysis.get('predictions', {}),
                    confidence=analysis.get('confidence', 0.7)
                )
            else:
                # Mock 분석 결과
                analysis_result = AnalysisResult(
                    topic=research_question,
                    methodology="LASSO",
                    key_findings=["Fed 금리 인상 사이클 종료 근접"],
                    statistics={"r_squared": 0.72},
                    predictions={"fed_rate_3m": 4.5},
                    confidence=0.7
                )

            # 실제 AI 토론으로 해석
            consensus = await self.interpretation.interpret_results(
                analysis_result=analysis_result,
                additional_context={"research_question": research_question}
            )

            self._log(f"  Schools: {len(consensus.school_interpretations)}")
            self._log(f"  Consensus points: {len(consensus.consensus_points)}")

        except Exception as e:
            self._log(f"  Real debate failed, falling back to mock: {e}")
            # 실패 시 Mock으로 폴백
            return await self._run_interpretation_mock(
                research_question, analysis, methodology, start
            )

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.INTERPRETATION,
            status=PipelineStatus.COMPLETED,
            result=consensus,
            duration_seconds=duration
        )

    async def _run_strategy(
        self,
        data: Dict[str, Any],
        top_down: Optional[TopDownResult],
        interpretation: Optional[InterpretationConsensus],
        config: PipelineConfig
    ) -> StageResult:
        """Stage 6: 전략 생성"""
        start = datetime.now()
        self._log("Stage 6: Strategy Generation")

        # 전략 생성 (간소화된 버전)
        strategy = {
            "overall_stance": top_down.final_stance.value if top_down else "NEUTRAL",
            "risk_level": config.risk_tolerance,
            "recommendations": [
                {
                    "asset": "Equities",
                    "action": "HOLD",
                    "rationale": "중립적 통화 환경, 경기 후반부"
                },
                {
                    "asset": "Bonds",
                    "action": "NEUTRAL",
                    "rationale": "금리 피크 근접, 듀레이션 중립"
                },
                {
                    "asset": "Commodities",
                    "action": "UNDERWEIGHT",
                    "rationale": "달러 강세, 수요 둔화"
                },
                {
                    "asset": "Cash",
                    "action": "OVERWEIGHT",
                    "rationale": "경기 후반부 방어적 포지션"
                }
            ],
            "key_risks": interpretation.risk_factors[:3] if interpretation else [],
            "confidence": 0.7
        }

        self._log(f"  Stance: {strategy['overall_stance']}")
        self._log(f"  Recommendations: {len(strategy['recommendations'])}")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.STRATEGY_GENERATION,
            status=PipelineStatus.COMPLETED,
            result=strategy,
            duration_seconds=duration
        )

    async def _run_synthesis(
        self,
        research_question: str,
        result: PipelineResult
    ) -> StageResult:
        """Stage 7: 최종 종합"""
        start = datetime.now()
        self._log("Stage 7: Synthesis")

        # 결과 종합
        synthesis = self._synthesize_results(research_question, result)

        self._log(f"  Summary: {synthesis['summary'][:50]}...")

        duration = (datetime.now() - start).total_seconds()
        return StageResult(
            stage=PipelineStage.SYNTHESIS,
            status=PipelineStatus.COMPLETED,
            result=synthesis,
            duration_seconds=duration
        )

    def _synthesize_results(
        self,
        research_question: str,
        result: PipelineResult
    ) -> Dict[str, Any]:
        """결과 종합"""
        parts = []

        # Top-Down 결과
        if result.top_down:
            td = result.top_down
            parts.append(f"하향식 분석: {td.final_stance.value} 스탠스")
            if td.geopolitical:
                parts.append(f"지정학적 위험: {td.geopolitical.risk_level.value}")
            if td.monetary:
                parts.append(f"통화환경: {td.monetary.policy_stance.value}")

        # 방법론
        if result.methodology:
            parts.append(f"분석 방법: {result.methodology.selected_methodology.value}")

        # 해석
        if result.interpretation:
            if result.interpretation.consensus_points:
                parts.append(f"학파 합의: {result.interpretation.consensus_points[0][:30]}...")

        # 전략
        strategy = result.stages.get(PipelineStage.STRATEGY_GENERATION)
        if strategy and strategy.result:
            s = strategy.result
            parts.append(f"전략 스탠스: {s['overall_stance']}")

        summary = " | ".join(parts) if parts else "분석 결과 종합 중"

        # 신뢰도 계산
        confidences = []
        if result.top_down:
            confidences.append(result.top_down.total_confidence)
        if result.methodology:
            confidences.append(result.methodology.confidence)
        if result.interpretation:
            confidences.append(result.interpretation.confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return {
            "question": research_question,
            "summary": summary,
            "recommendation": result.top_down.final_recommendation if result.top_down else "",
            "confidence": avg_confidence,
            "components": {
                "top_down": result.top_down is not None,
                "methodology": result.methodology is not None,
                "interpretation": result.interpretation is not None,
                "strategy": PipelineStage.STRATEGY_GENERATION in result.stages
            }
        }

    def _finalize_result(
        self,
        result: PipelineResult,
        start_time: datetime
    ) -> PipelineResult:
        """결과 최종화"""
        result.total_duration_seconds = (datetime.now() - start_time).total_seconds()

        # 신뢰도 계산
        confidences = []
        if result.top_down:
            confidences.append(result.top_down.total_confidence)
        if result.methodology:
            confidences.append(result.methodology.confidence)
        if result.interpretation:
            confidences.append(result.interpretation.confidence)

        result.confidence = sum(confidences) / len(confidences) if confidences else 0.5

        self._log(f"\nPipeline completed in {result.total_duration_seconds:.2f}s")
        self._log(f"Status: {result.status.value}")
        self._log(f"Confidence: {result.confidence:.0%}")

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

async def run_quick_analysis(
    question: str,
    verbose: bool = True,
    use_mock: bool = True
) -> PipelineResult:
    """
    빠른 분석 실행

    Parameters:
    -----------
    question : str
        연구 질문
    verbose : bool
        상세 로깅
    use_mock : bool
        True: Mock 응답 (빠름), False: 실제 AI 토론

    Returns:
    --------
    PipelineResult
    """
    runner = FullPipelineRunner(verbose=verbose, use_mock=use_mock)
    return await runner.run(
        research_question=question,
        config=PipelineConfig(
            stop_at_level=AnalysisLevel.SECTOR,
            research_goal=ResearchGoal.VARIABLE_SELECTION
        )
    )


def print_result_summary(result: PipelineResult):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("PIPELINE RESULT SUMMARY")
    print("=" * 60)

    print(f"\nStatus: {result.status.value}")
    print(f"Duration: {result.total_duration_seconds:.2f}s")
    print(f"Confidence: {result.confidence:.0%}")

    if result.aborted_at:
        print(f"\n⚠️ Aborted at: {result.aborted_at.value}")
        print(f"   Reason: {result.abort_reason}")

    print("\n--- Top-Down Analysis ---")
    if result.top_down:
        print(f"  Stance: {result.top_down.final_stance.value}")
        print(f"  Recommendation: {result.top_down.final_recommendation}")

    print("\n--- Methodology ---")
    if result.methodology:
        print(f"  Selected: {result.methodology.selected_methodology.value}")
        print(f"  Confidence: {result.methodology.confidence:.0%}")

    print("\n--- Interpretation ---")
    if result.interpretation:
        print(f"  Schools: {len(result.interpretation.school_interpretations)}")
        if result.interpretation.consensus_points:
            print(f"  Consensus: {result.interpretation.consensus_points[0][:50]}...")

    print("\n--- Strategy ---")
    strategy_stage = result.stages.get(PipelineStage.STRATEGY_GENERATION)
    if strategy_stage and strategy_stage.result:
        s = strategy_stage.result
        print(f"  Overall: {s['overall_stance']}")
        for rec in s['recommendations'][:3]:
            print(f"    {rec['asset']}: {rec['action']}")

    print("\n--- Executive Summary ---")
    print(f"  {result.executive_summary}")

    print("\n" + "=" * 60)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    async def test():
        print("=== Full Pipeline Test ===\n")

        # 파이프라인 실행
        result = await run_quick_analysis(
            question="Fed 금리 정책이 2025년 시장에 미치는 영향은?"
        )

        # 결과 출력
        print_result_summary(result)

        print("\n✅ Pipeline test completed!")

    asyncio.run(test())
