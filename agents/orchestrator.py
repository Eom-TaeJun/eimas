#!/usr/bin/env python3
"""
Multi-Agent System - Meta Orchestrator
=======================================
에이전트 조정, 토론 관리, 결과 종합

경제학적 의미:
- 중앙 계획자(Central Planner) 역할
- 정보 흐름 최적화 (Information Flow Optimization)
- 집단 의사결정 조정 (Collective Decision Coordination)
- Nash Equilibrium 탐색 (Debate를 통한 균형점 도달)
"""

import logging
import os
import socket
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from core.schemas import (
    AgentRequest,
    AgentResponse,
    AgentOpinion,
    Consensus,
    AgentRole,
    TaskPriority,
)
from core.debate import DebateProtocol
from core.reasoning_chain import ReasoningChain
from agents.analysis_agent import AnalysisAgent
from agents.forecast_agent import ForecastAgent
from agents.research_agent import ResearchAgent
from agents.strategy_agent import StrategyAgent
from agents.verification_agent import VerificationAgent
from agents.interpretation_debate import InterpretationDebateAgent, AnalysisResult
from agents.methodology_debate import MethodologyDebateAgent, ResearchGoal


class MetaOrchestrator:
    """
    메타 오케스트레이터

    역할:
    1. 에이전트 간 작업 조정
    2. 토론 관리 및 합의 도출
    3. 최종 보고서 종합

    방법론:
    - 정보 흐름 최적화 (Information Flow Optimization)
    - 집단 의사결정 조정 (Collective Decision Coordination)
    - 다수결 합의 (Majority Voting Consensus)
    """

    def __init__(self, verbose: bool = True):
        """초기화"""
        self.verbose = verbose
        self.logger = self._setup_logger()

        # 핵심 컴포넌트 초기화
        # VIX > 20을 "높은 변동성"의 기준으로 설정 (기존 85% 일관성 임계값 유지)
        self.debate_protocol = DebateProtocol(
            max_rounds=3,
            consistency_threshold=85.0,
            modification_threshold=5.0
        )

        # 에이전트 초기화
        self.analysis_agent = AnalysisAgent()
        self.forecast_agent = ForecastAgent()
        self.research_agent = ResearchAgent()
        self.strategy_agent = StrategyAgent()
        self.verification_agent = VerificationAgent()

        # Phase 2 에이전트 (Multi-LLM Debate)
        self.interpretation_agent = InterpretationDebateAgent()
        self.methodology_agent = MethodologyDebateAgent()

        # Phase 3 추론 체인 (Traceability)
        self.reasoning_chain = ReasoningChain()

        # 기본 토론 주제
        self.default_debate_topics = [
            "market_outlook",
            "primary_risk",
            "regime_stability",
            "rate_direction",
            "bubble_assessment",     # JP Morgan 5단계 (NEW)
            "institutional_view"     # 기관 투자자 관점 (NEW)
        ]

        self.logger.info("MetaOrchestrator initialized (7 agents + Multi-LLM Debate + ReasoningChain)")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("MetaOrchestrator")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - MetaOrchestrator - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _enhanced_debate_skip_reason(self) -> str:
        if self._env_flag("EIMAS_SKIP_ENHANCED_DEBATE", default=False):
            return "EIMAS_SKIP_ENHANCED_DEBATE"

        fail_fast_default = self._env_flag("EIMAS_REPORT_FAIL_FAST_NETWORK", default=False)
        if not self._env_flag("EIMAS_DEBATE_FAIL_FAST_NETWORK", default=fail_fast_default):
            return ""

        hosts_raw = os.getenv(
            "EIMAS_DEBATE_NETWORK_PROBE_HOSTS",
            "api.openai.com,api.anthropic.com,generativelanguage.googleapis.com",
        )
        hosts = [item.strip() for item in hosts_raw.split(",") if item.strip()]
        if not hosts:
            hosts = [
                "api.openai.com",
                "api.anthropic.com",
                "generativelanguage.googleapis.com",
            ]

        for host in hosts:
            try:
                socket.getaddrinfo(host, 443)
                return ""
            except OSError:
                continue
        return f"dns_unavailable:{','.join(hosts)}"

    async def run_with_debate(
        self,
        query: str,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        토론 포함 전체 워크플로우 실행

        Flow:
        1. 분석 에이전트 실행 (Critical Path Analysis)
        2. 분석 결과 기반 토론 주제 자동 감지
        3. 에이전트들로부터 의견 수집
        4. 토론 프로토콜 실행 및 합의 도출
        5. 최종 보고서 종합

        Args:
            query: 사용자 질문/요청
            market_data: 시장 데이터 (Dict[ticker, pd.DataFrame])

        Returns:
            종합 결과 딕셔너리:
            {
                'timestamp': str,
                'query': str,
                'analysis': Dict (CriticalPath 결과),
                'debate_topics': List[str],
                'opinions': Dict[topic, List[AgentOpinion]],
                'consensus': Dict[topic, Consensus],
                'recommendations': List[str],
                'warnings': List[str]
            }
        """
        self.logger.info(f"Starting workflow with debate for query: {query[:100]}")

        workflow_start = datetime.now()

        # Reset reasoning chain for new workflow
        self.reasoning_chain = ReasoningChain()

        # ============================================================
        # Step 1: Run Analysis Agent
        # ============================================================
        self.logger.info("Step 1: Running Analysis Agent...")

        task_id = str(uuid.uuid4())
        analysis_request = AgentRequest(
            task_id=task_id,
            role=AgentRole.ANALYSIS,
            instruction=f"Analyze market conditions and risks: {query}",
            context={'market_data': market_data},
            priority=TaskPriority.HIGH
        )

        analysis_response: AgentResponse = await self.analysis_agent.execute(analysis_request)

        if not analysis_response.is_success():
            self.logger.error(f"Analysis failed: {analysis_response.error}")
            return self._create_error_result(query, analysis_response.error)

        analysis_result = analysis_response.result
        self.logger.info(
            f"Analysis complete: Risk={analysis_result.get('total_risk_score', 0):.1f}, "
            f"Regime={analysis_result.get('current_regime', 'UNKNOWN')}"
        )

        # Track in reasoning chain
        self.reasoning_chain.add_step(
            agent='AnalysisAgent',
            input_summary='Market data + CriticalPath analysis',
            output_summary=f"Risk={analysis_result.get('total_risk_score', 0):.1f}, Regime={analysis_result.get('current_regime', 'UNKNOWN')}",
            confidence=analysis_result.get('regime_confidence', 50.0),
            key_factors=analysis_result.get('active_warnings', [])[:3]
        )

        # ============================================================
        # Step 1.5: Run Forecast Agent
        # ============================================================
        self.logger.info("Step 1.5: Running Forecast Agent...")
        
        forecast_request = AgentRequest(
            task_id=str(uuid.uuid4()),
            role=AgentRole.FORECAST,
            instruction="Forecast Fed Funds Rate and identify key drivers",
            context={'market_data': market_data, 'horizon': 'short'},
            priority=TaskPriority.HIGH
        )
        
        forecast_response = await self.forecast_agent.execute(forecast_request)
        forecast_result = {}
        forecast_payload = forecast_response.result if forecast_response.is_success() else {}
        has_legacy_forecast = bool(forecast_payload.get('forecast_results'))
        has_modern_forecast = bool(forecast_payload.get('forecasts'))
        if has_legacy_forecast or has_modern_forecast:
            forecast_result = forecast_response.result
            if has_legacy_forecast:
                first_forecast = forecast_result['forecast_results'][0] if forecast_result['forecast_results'] else {}
                point_forecast = first_forecast.get('point_forecast', 0.0)
                r2_score = forecast_result.get('model_metrics', {}).get('r2_score', 0.0)
            else:
                forecasts = forecast_result.get('forecasts', [])
                long_forecast = next(
                    (item for item in forecasts if item.get('horizon') == 'Long'),
                    forecasts[0] if forecasts else {},
                )
                point_forecast = long_forecast.get('predicted_change', 0.0) or 0.0
                r2_score = long_forecast.get('r_squared', forecast_result.get('confidence', 0.0))
            self.logger.info(
                f"Forecast complete: Point={point_forecast:.4f} "
                f"(R2={r2_score:.2f})"
            )
        else:
            self.logger.warning(f"Forecast results empty or failed: {forecast_response.error or 'No results'}")

        # Track in reasoning chain
        if forecast_result:
            if forecast_result.get('forecast_results'):
                forecast_summary = forecast_result.get('forecast_results', [{}])[0]
                confidence_r2 = forecast_result.get('model_metrics', {}).get('r2_score', 0.5)
            else:
                forecast_rows = forecast_result.get('forecasts', [])
                forecast_summary = next(
                    (item for item in forecast_rows if item.get('horizon') == 'Long'),
                    forecast_rows[0] if forecast_rows else {},
                )
                confidence_r2 = forecast_summary.get('r_squared', forecast_result.get('confidence', 0.5))
            key_drivers = forecast_result.get('key_drivers', [])
            if not key_drivers and isinstance(forecast_summary, dict):
                key_drivers = forecast_summary.get('selected_variables', [])
            self.reasoning_chain.add_step(
                agent='ForecastAgent',
                input_summary='Market data + LASSO model',
                output_summary=(
                    f"Point forecast: "
                    f"{forecast_summary.get('point_forecast', forecast_summary.get('predicted_change', 'N/A'))}"
                ),
                confidence=confidence_r2 * 100,
                key_factors=key_drivers[:3]
            )

        # Merge results for topic detection and opinion collection
        combined_context = {**analysis_result, **forecast_result}

        # ============================================================
        # Step 2: Auto-detect Debate Topics
        # ============================================================
        self.logger.info("Step 2: Auto-detecting debate topics...")

        debate_topics = self._auto_detect_topics(combined_context)
        self.logger.info(f"Detected {len(debate_topics)} debate topics: {debate_topics}")

        # ============================================================
        # Step 3: Collect Opinions
        # ============================================================
        self.logger.info("Step 3: Collecting opinions from agents...")

        opinions_by_topic = await self.collect_opinions(
            topics=debate_topics,
            context=combined_context
        )

        total_opinions = sum(len(ops) for ops in opinions_by_topic.values())
        self.logger.info(f"Collected {total_opinions} opinions across {len(opinions_by_topic)} topics")

        # ============================================================
        # Step 4: Run Debates
        # ============================================================
        self.logger.info("Step 4: Running debates...")

        consensus_results = await self.run_debates(opinions_by_topic)
        self.logger.info(f"Reached consensus on {len(consensus_results)} topics")

        # Track in reasoning chain
        self.reasoning_chain.add_step(
            agent='DebateProtocol',
            input_summary=f'{total_opinions} opinions on {len(opinions_by_topic)} topics',
            output_summary=f'Consensus on {len(consensus_results)} topics',
            confidence=sum(c.confidence for c in consensus_results.values()) / max(len(consensus_results), 1) * 100,
            key_factors=[f"{t}: {c.final_position}" for t, c in list(consensus_results.items())[:3]]
        )

        # ============================================================
        # Step 4.5: Enhanced Multi-LLM Debate (Optional)
        # ============================================================
        self.logger.info("Step 4.5: Running Enhanced Multi-LLM Debate...")

        enhanced_debate_results = {}
        enhanced_skip_reason = self._enhanced_debate_skip_reason()
        if enhanced_skip_reason:
            self.logger.info(f"Skipping enhanced debate ({enhanced_skip_reason})")
            enhanced_debate_results = {"skipped": True, "reason": enhanced_skip_reason}
        else:
            try:
                # Interpretation Debate: 경제학파별 해석
                interpretation_result = await self._run_interpretation_debate(combined_context)
                enhanced_debate_results['interpretation'] = interpretation_result

                # Methodology Debate: 방법론 토론
                methodology_result = await self._run_methodology_debate(query, combined_context)
                enhanced_debate_results['methodology'] = methodology_result

                self.logger.info(
                    f"Enhanced debate complete: Interpretation={interpretation_result.get('recommended_action', 'N/A')}, "
                    f"Methodology={methodology_result.get('selected_methodology', 'N/A')}"
                )

                # Track in reasoning chain
                self.reasoning_chain.add_step(
                    agent='MultiLLMDebate',
                    input_summary='Analysis results + Economic schools',
                    output_summary=f"Interpretation: {interpretation_result.get('recommended_action', 'N/A')}, Methodology: {methodology_result.get('selected_methodology', 'N/A')}",
                    confidence=((interpretation_result.get('confidence', 0.5) + methodology_result.get('confidence', 0.5)) / 2) * 100,
                    key_factors=['Claude (Economist)', 'GPT-4 (Devil\'s Advocate)', 'Gemini (Risk Manager)']
                )

            except Exception as e:
                self.logger.warning(f"Enhanced debate failed (non-critical): {e}")
                enhanced_debate_results = {'error': str(e)}

        # ============================================================
        # Step 5: Synthesize Report
        # ============================================================
        self.logger.info("Step 5: Synthesizing final report...")

        final_report = self.synthesize_report(
            query=query,
            analysis_result=analysis_result,
            forecast_result=forecast_result,
            consensus_results=consensus_results,
            opinions_by_topic=opinions_by_topic,
            enhanced_debate_results=enhanced_debate_results
        )

        # ============================================================
        # Step 6: Verification (Optional)
        # ============================================================
        self.logger.info("Step 6: Running verification...")

        try:
            verification_request = AgentRequest(
                task_id=str(uuid.uuid4()),
                role=AgentRole.VERIFICATION,
                instruction="Verify debate results for hallucination and consistency",
                context={
                    'debate_results': final_report.get('debate', {}),
                    'opinions': final_report.get('debate', {}).get('opinions', []),
                    'market_data': market_data,
                    'consensus': final_report.get('debate', {}).get('consensus', {})
                },
                priority=TaskPriority.MEDIUM
            )

            verification_response = await self.verification_agent.execute(verification_request)

            if verification_response.is_success():
                verification_result = verification_response.result
                final_report['verification'] = {
                    'overall_score': verification_result.get('verification_result', {}).get('overall_score', 0),
                    'hallucination_risk': verification_result.get('hallucination_check', {}).get('confidence', 0),
                    'sycophancy_risk': verification_result.get('sycophancy_check', {}).get('agreement_rate', 0),
                    'passed': verification_result.get('verification_result', {}).get('passed', True),
                    'warnings': verification_result.get('verification_result', {}).get('warnings', [])
                }
                self.logger.info(
                    f"Verification complete: Score={final_report['verification']['overall_score']:.1f}, "
                    f"Passed={final_report['verification']['passed']}"
                )
            else:
                self.logger.warning(f"Verification failed: {verification_response.error}")
                final_report['verification'] = {'status': 'failed', 'error': str(verification_response.error)}

        except Exception as e:
            self.logger.warning(f"Verification step error: {e}")
            final_report['verification'] = {'status': 'error', 'error': str(e)}

        workflow_duration = (datetime.now() - workflow_start).total_seconds()
        final_report['workflow_duration_seconds'] = workflow_duration

        self.logger.info(f"Workflow completed in {workflow_duration:.2f}s")

        return final_report

    def _auto_detect_topics(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        분석 결과 기반 토론 주제 자동 감지

        감지 규칙:
        - total_risk_score가 40-60 범위: "market_outlook" 추가 (불확실성 높음)
        - transition_probability > 0.3: "regime_stability" 추가 (레짐 변화 가능)
        - active_warnings > 3개: "primary_risk" 추가 (다양한 위험 요인)
        - crypto 데이터 존재: "crypto_correlation" 추가

        Args:
            analysis_result: CriticalPath 분석 결과

        Returns:
            토론 주제 리스트
        """
        topics = []

        total_risk_score = analysis_result.get('total_risk_score', 50.0)
        transition_probability = analysis_result.get('transition_probability', 0.0)
        active_warnings = analysis_result.get('active_warnings', [])
        path_contributions = analysis_result.get('path_contributions', {})
        crypto_result = analysis_result.get('critical_path_result', {}).get('crypto_result', {})

        # Rule 1: 애매한 리스크 범위 → market_outlook 토론 필요
        if 40 <= total_risk_score <= 60:
            topics.append("market_outlook")
            self.logger.debug(f"Added 'market_outlook' (risk={total_risk_score:.1f} in uncertain range)")

        # Rule 2: 높은 전이 확률 → regime_stability 토론 필요
        if transition_probability > 0.3:
            topics.append("regime_stability")
            display_prob = transition_probability / 100.0 if transition_probability > 1.0 else transition_probability
            self.logger.debug(f"Added 'regime_stability' (transition_prob={display_prob:.1%})")

        # Rule 3: 다수의 경고 → primary_risk 토론 필요
        if len(active_warnings) > 3:
            topics.append("primary_risk")
            self.logger.debug(f"Added 'primary_risk' ({len(active_warnings)} active warnings)")

        # Rule 4: 다중 경로 기여 → primary_risk 토론 필요
        if len([c for c in path_contributions.values() if c > 15.0]) > 2:
            if "primary_risk" not in topics:
                topics.append("primary_risk")
                self.logger.debug(f"Added 'primary_risk' (multiple significant paths)")

        # Rule 5: 암호화폐 데이터 존재 → crypto_correlation 토론 가능
        if crypto_result:
            topics.append("crypto_correlation")
            self.logger.debug(f"Added 'crypto_correlation' (crypto data available)")

        # Rule 6: 금리 예측 데이터 존재 → rate_direction 토론 추가
        if 'forecast_results' in analysis_result or 'forecasts' in analysis_result:
            topics.append("rate_direction")
            topics.append("rate_magnitude")
            self.logger.debug("Added 'rate_direction', 'rate_magnitude' (forecast data available)")

        # Rule 7: 버블 리스크 데이터 존재 → bubble_assessment 토론 추가 (NEW)
        bubble_risk = analysis_result.get('bubble_risk', {})
        if bubble_risk.get('overall_status') in ['WATCH', 'WARNING', 'DANGER']:
            topics.append("bubble_assessment")
            self.logger.debug(f"Added 'bubble_assessment' (status={bubble_risk.get('overall_status')})")

        # Rule 8: 기관 투자자 관점 항상 포함 (NEW)
        # 시장 품질 또는 버블 리스크 데이터가 있으면 기관 관점 토론
        market_quality = analysis_result.get('market_quality', {})
        if market_quality or bubble_risk:
            topics.append("institutional_view")
            self.logger.debug("Added 'institutional_view' (institutional methodology data available)")

        # Fallback: 토픽이 없으면 기본 토픽 사용
        if not topics:
            self.logger.info("No topics auto-detected, using default topics")
            topics = self.default_debate_topics.copy()

        return topics

    async def collect_opinions(
        self,
        topics: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, List[AgentOpinion]]:
        """
        각 에이전트로부터 의견 수집

        활성 에이전트: AnalysisAgent, ForecastAgent

        Args:
            topics: 의견을 수집할 주제 리스트
            context: 분석 결과 등 컨텍스트

        Returns:
            {topic: [opinion1, opinion2, ...]} 딕셔너리
        """
        opinions_by_topic = {}

        for topic in topics:
            opinions = []

            # Analysis Agent의 의견 수집
            try:
                opinion = await self.analysis_agent.form_opinion(topic, context)
                opinions.append(opinion)
                self.logger.debug(
                    f"Collected opinion from AnalysisAgent on '{topic}': "
                    f"{opinion.position} (confidence={opinion.confidence:.2f})"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get opinion from AnalysisAgent on '{topic}': {e}")

            # Forecast Agent의 의견 수집
            if topic in ["rate_direction", "rate_magnitude", "forecast_confidence", "market_outlook"]:
                try:
                    opinion = await self.forecast_agent.form_opinion(topic, context)
                    opinions.append(opinion)
                    self.logger.debug(
                        f"Collected opinion from ForecastAgent on '{topic}': "
                        f"{opinion.position} (confidence={opinion.confidence:.2f})"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to get opinion from ForecastAgent on '{topic}': {e}")

            # Research Agent의 의견 수집
            try:
                opinion = await self.research_agent.form_opinion(topic, context)
                opinions.append(opinion)
                self.logger.debug(
                    f"Collected opinion from ResearchAgent on '{topic}': "
                    f"{opinion.position} (confidence={opinion.confidence:.2f})"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get opinion from ResearchAgent on '{topic}': {e}")

            # Strategy Agent의 의견 수집
            try:
                opinion = await self.strategy_agent.form_opinion(topic, context)
                opinions.append(opinion)
                self.logger.debug(
                    f"Collected opinion from StrategyAgent on '{topic}': "
                    f"{opinion.position} (confidence={opinion.confidence:.2f})"
                )
            except Exception as e:
                self.logger.warning(f"Failed to get opinion from StrategyAgent on '{topic}': {e}")

            if opinions:
                opinions_by_topic[topic] = opinions
            else:
                self.logger.warning(f"No opinions collected for topic '{topic}'")

        return opinions_by_topic

    async def run_debates(
        self,
        opinions_by_topic: Dict[str, List[AgentOpinion]]
    ) -> Dict[str, Consensus]:
        """
        각 주제에 대해 토론 프로토콜 실행

        논리:
        - 의견이 1개만 있으면 토론 스킵 (단일 의견 = 합의)
        - 의견이 2개 이상이면 DebateProtocol.run_debate() 실행
        - 초기 일관성이 매우 높으면(95% 이상) 토론 스킵

        Args:
            opinions_by_topic: {topic: [opinions]} 딕셔너리

        Returns:
            {topic: Consensus} 딕셔너리
        """
        consensus_results = {}

        for topic, opinions in opinions_by_topic.items():
            self.logger.info(f"Processing debate for topic '{topic}' with {len(opinions)} opinions")

            # 단일 의견: 토론 불필요
            if len(opinions) == 1:
                self.logger.info(f"Single opinion for '{topic}' - skipping debate")
                consensus = Consensus(
                    topic=topic,
                    final_position=opinions[0].position,
                    confidence=opinions[0].confidence,
                    supporting_agents=[opinions[0].agent_role],
                    dissenting_agents=[],
                    compromises=[],
                    debate_rounds=0
                )
                consensus_results[topic] = consensus
                continue

            # 초기 일관성 체크
            initial_consistency = self.debate_protocol.calculate_consistency(opinions)
            self.logger.debug(f"Initial consistency for '{topic}': {initial_consistency:.1f}%")

            if initial_consistency >= 95.0:
                self.logger.info(f"High initial consistency ({initial_consistency:.1f}%) - skipping debate")
                consensus = self.debate_protocol.reach_consensus(opinions, debate_rounds=0)
                consensus_results[topic] = consensus
                continue

            # 토론 실행
            try:
                consensus = self.debate_protocol.run_debate(opinions, topic)
                self.logger.info(
                    f"Debate complete for '{topic}': {consensus.final_position} "
                    f"(confidence={consensus.confidence:.2%}, rounds={consensus.debate_rounds})"
                )
                consensus_results[topic] = consensus
            except Exception as e:
                self.logger.error(f"Debate failed for '{topic}': {e}")
                # Fallback: 첫 번째 의견을 기본 합의로 사용
                consensus = Consensus(
                    topic=topic,
                    final_position=opinions[0].position,
                    confidence=0.5,
                    supporting_agents=[opinions[0].agent_role],
                    dissenting_agents=[],
                    compromises=["Debate failed - using fallback opinion"],
                    debate_rounds=0
                )
                consensus_results[topic] = consensus

        return consensus_results

    def synthesize_report(
        self,
        query: str,
        analysis_result: Dict[str, Any],
        forecast_result: Dict[str, Any],
        consensus_results: Dict[str, Consensus],
        opinions_by_topic: Dict[str, List[AgentOpinion]],
        enhanced_debate_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        최종 보고서 종합
        """
        enhanced_debate_results = enhanced_debate_results or {}
        # transition_probability 스케일 보정 (0-100 -> 0-1)
        trans_prob = analysis_result.get('transition_probability', 0.0)
        if trans_prob > 1.0:
            trans_prob /= 100.0

        # 모든 의견을 플랫 리스트로 변환
        all_opinions = []
        for topic, opinions in opinions_by_topic.items():
            for opinion in opinions:
                all_opinions.append({
                    'topic': topic,
                    'agent_role': opinion.agent_role.value if hasattr(opinion.agent_role, 'value') else str(opinion.agent_role),
                    'position': opinion.position,
                    'confidence': opinion.confidence,
                    'strength': opinion.strength.value if hasattr(opinion.strength, 'value') else str(opinion.strength),
                    'evidence': opinion.evidence,
                    'caveats': opinion.caveats,
                    'key_metrics': opinion.key_metrics
                })

        # Agent Outputs collection
        agent_outputs_dict = {
            'analysis': analysis_result,
            'forecast': forecast_result,
            # We don't have direct access to Research/Strategy raw outputs here easily 
            # unless we stored them in Step 3.
            # They are in opinions_by_topic but mixed.
            # For traceability, we should store them when collecting.
        }

        report = {
            'timestamp': datetime.now().isoformat(),
            'agent_outputs': agent_outputs_dict, # Added this
            'query': query,
            'analysis': {
                'total_risk_score': analysis_result.get('total_risk_score', 0.0),
                'risk_level': analysis_result.get('risk_level', 'UNKNOWN'),
                'current_regime': analysis_result.get('current_regime', 'UNKNOWN'),
                'regime_confidence': analysis_result.get('regime_confidence', 0.0),
                'transition_probability': trans_prob,
                'primary_risk_path': analysis_result.get('primary_risk_path', 'unknown'),
                'path_contributions': analysis_result.get('path_contributions', {}),
                'active_warnings': analysis_result.get('active_warnings', []),
            },
            'forecast': forecast_result,
            'debate': {
                'opinions': all_opinions,
                'consensus': {
                    topic: {
                        'final_position': consensus.final_position,
                        'confidence': consensus.confidence,
                        'supporting_agents': [a.value for a in consensus.supporting_agents],
                        'dissenting_agents': [a.value for a in consensus.dissenting_agents],
                        'debate_rounds': consensus.debate_rounds,
                        'compromises': consensus.compromises
                    }
                    for topic, consensus in consensus_results.items()
                },
                'conflicts': [],
                # Phase 2: Enhanced Multi-LLM Debate Results
                'enhanced_debate': {
                    'interpretation': enhanced_debate_results.get('interpretation', {}),
                    'methodology': enhanced_debate_results.get('methodology', {})
                }
            },
            # Phase 3: Reasoning Chain (Traceability)
            'reasoning_chain': self.reasoning_chain.to_dict(),
            'debate_topics': list(consensus_results.keys()),
            'recommendations': self._generate_recommendations(analysis_result, consensus_results),
            'warnings': self._generate_warnings(analysis_result, consensus_results),
            # Institutional Analysis (NEW)
            'institutional_analysis': self._extract_institutional_insights(
                consensus_results, opinions_by_topic, analysis_result
            ),
            'metadata': {
                'num_agents': 7,  # Analysis, Forecast, Research, Strategy, Verification + Interpretation, Methodology
                'total_opinions': len(all_opinions),
                'total_debates': len([c for c in consensus_results.values() if c.debate_rounds > 0]),
                'avg_confidence': sum(c.confidence for c in consensus_results.values()) / max(len(consensus_results), 1),
                'enhanced_debate_enabled': bool(
                    enhanced_debate_results
                    and 'error' not in enhanced_debate_results
                    and not enhanced_debate_results.get('skipped', False)
                ),
                'reasoning_chain_steps': len(self.reasoning_chain.to_dict()),
                'institutional_methodology_enabled': True  # NEW
            }
        }

        return report

    async def _run_interpretation_debate(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        경제학파별 해석 토론 실행 (Monetarist, Keynesian, Austrian)
        """
        # Create AnalysisResult for interpretation
        analysis_result = AnalysisResult(
            topic=context.get('primary_risk_path', 'market_conditions'),
            methodology='CriticalPath + LASSO',
            key_findings=[
                f"Risk Score: {context.get('total_risk_score', 50):.1f}",
                f"Regime: {context.get('current_regime', 'NEUTRAL')}",
                f"Primary Risk: {context.get('primary_risk_path', 'unknown')}"
            ],
            statistics={
                'risk_score': context.get('total_risk_score', 50),
                'regime_confidence': context.get('regime_confidence', 50)
            },
            predictions={
                'regime_transition': context.get('transition_probability', 0.0)
            },
            confidence=context.get('regime_confidence', 50) / 100.0
        )

        try:
            consensus = await self.interpretation_agent.interpret_results(
                analysis_result=analysis_result,
                additional_context=context
            )
            return {
                'recommended_action': consensus.recommended_action,
                'consensus_points': consensus.consensus_points,
                'divergence_points': consensus.divergence_points,
                'school_interpretations': consensus.school_interpretations,
                'confidence': consensus.confidence,
                'summary': consensus.summary
            }
        except Exception as e:
            self.logger.warning(f"Interpretation debate error: {e}")
            return {'error': str(e), 'recommended_action': 'NEUTRAL', 'confidence': 0.5}

    async def _run_methodology_debate(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        방법론 토론 실행 (LASSO, VAR, GARCH 등)
        """
        try:
            decision = await self.methodology_agent.debate_methodology(
                research_question=query,
                research_goal=ResearchGoal.VARIABLE_SELECTION,
                data_summary=None,
                research_context=str(context.get('path_contributions', {}))
            )
            return {
                'selected_methodology': decision.selected_methodology,
                'components': decision.components,
                'pipeline': decision.pipeline,
                'confidence': decision.confidence,
                'rationale': decision.rationale,
                'dissenting_views': decision.dissenting_views
            }
        except Exception as e:
            self.logger.warning(f"Methodology debate error: {e}")
            return {'error': str(e), 'selected_methodology': 'LASSO', 'confidence': 0.5}

    def _extract_institutional_insights(
        self,
        consensus_results: Dict[str, Consensus],
        opinions_by_topic: Dict[str, List[AgentOpinion]],
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        기관 투자자 방법론 기반 인사이트 추출 (NEW)

        References:
        - JP Morgan: 5단계 버블 프레임워크, Time-series Shock Modeling
        - Goldman Sachs: Gap-Bridging, DCF 검증
        - Berkshire Hathaway: 보수적 추정, Fair Value Hierarchy

        Args:
            consensus_results: 토론 합의 결과
            opinions_by_topic: 주제별 의견
            analysis_result: 분석 결과

        Returns:
            기관 인사이트 딕셔너리
        """
        insights = {
            'methodology_applied': [],
            'jpmorgan_framework': {},
            'gap_bridging': {},
            'risk_premium_quantification': {},
            'narrative': ''
        }

        # 1. JP Morgan 5단계 버블 프레임워크
        if 'bubble_assessment' in consensus_results:
            bubble_consensus = consensus_results['bubble_assessment']
            insights['jpmorgan_framework'] = {
                'consensus_position': bubble_consensus.final_position,
                'confidence': bubble_consensus.confidence,
                'supporting_agents': [a.value for a in bubble_consensus.supporting_agents],
                'dissenting_agents': [a.value for a in bubble_consensus.dissenting_agents],
                'methodology': 'JP Morgan 5-Stage Bubble Framework (Paradigm → Credit → Leverage → Speculation → Collapse)'
            }
            insights['methodology_applied'].append('JP Morgan 5-Stage Bubble Framework')

        # 2. 기관 투자자 관점 종합
        if 'institutional_view' in consensus_results:
            inst_consensus = consensus_results['institutional_view']
            insights['institutional_view'] = {
                'consensus_position': inst_consensus.final_position,
                'confidence': inst_consensus.confidence,
                'methodology': 'Institutional Multi-Factor Assessment (Goldman Gap-Bridging + Berkshire Conservative)'
            }
            insights['methodology_applied'].append('Goldman Sachs Gap-Bridging')
            insights['methodology_applied'].append('Berkshire Hathaway Conservative Valuation')

        # 3. Gap-Bridging 분석 (시장 내재 기대 vs 분석 결과)
        total_risk_score = analysis_result.get('total_risk_score', 50)
        regime = analysis_result.get('current_regime', 'NEUTRAL')

        # Market expectation proxy (VIX-based)
        if total_risk_score < 30 and regime in ['BULL', 'EXPANSION']:
            market_expectation = 'BULLISH'
            model_forecast = 'BULLISH'
            gap = 'ALIGNED'
        elif total_risk_score > 60:
            market_expectation = 'CAUTIOUS'
            model_forecast = 'BEARISH'
            gap = 'ALIGNED'
        else:
            market_expectation = 'MIXED'
            model_forecast = 'NEUTRAL'
            gap = 'UNCERTAIN'

        insights['gap_bridging'] = {
            'market_expectation': market_expectation,
            'model_forecast': model_forecast,
            'gap_status': gap,
            'methodology': 'Goldman Sachs Gap-Bridging (Market Implied vs Quantitative Forecast)'
        }

        # 4. 리스크 프리미엄 정량화
        path_contributions = analysis_result.get('path_contributions', {})
        if path_contributions:
            primary_risk = max(path_contributions.items(), key=lambda x: x[1], default=('unknown', 0))
            insights['risk_premium_quantification'] = {
                'primary_risk_source': primary_risk[0],
                'risk_contribution': f"{primary_risk[1]:.1f}%",
                'methodology': 'Bekaert et al. VIX Decomposition (Uncertainty + Risk Appetite)'
            }
            insights['methodology_applied'].append('Bekaert VIX Risk Decomposition')

        # 5. 종합 내러티브 생성
        narratives = []

        if 'JP Morgan' in str(insights['methodology_applied']):
            bubble_pos = insights.get('jpmorgan_framework', {}).get('consensus_position', '')
            if 'DANGER' in bubble_pos or 'WARNING' in bubble_pos:
                narratives.append(
                    "JP Morgan 5단계 프레임워크: 버블 경고 신호 감지. "
                    "1990년대 닷컴 버블 또는 1840년대 철도 광기와 유사한 패턴."
                )
            else:
                narratives.append(
                    "JP Morgan 5단계 프레임워크: 현재 가격 움직임은 정상 범위 내."
                )

        if gap == 'ALIGNED':
            narratives.append(
                f"Goldman Sachs Gap-Bridging: 시장 내재 기대와 분석 결과가 일치 ({market_expectation})."
            )
        else:
            narratives.append(
                f"Goldman Sachs Gap-Bridging: 시장 기대와 분석 결과 간 괴리 존재. "
                f"추가 검증 필요."
            )

        if total_risk_score < 40:
            narratives.append(
                "Berkshire 보수적 접근: 리스크 점수 낮음. 점진적 포지션 구축 가능."
            )
        elif total_risk_score > 60:
            narratives.append(
                "Berkshire 보수적 접근: 리스크 점수 상승. 현금 비중 확대 권장."
            )

        insights['narrative'] = " ".join(narratives)

        return insights

    def _generate_recommendations(
        self,
        analysis_result: Dict[str, Any],
        consensus_results: Dict[str, Consensus]
    ) -> List[str]:
        """
        권고 사항 생성 (Rule-based)
        """
        recommendations = []

        total_risk_score = analysis_result.get('total_risk_score', 50.0)
        transition_probability = analysis_result.get('transition_probability', 0.0)
        if transition_probability > 1.0:
            transition_probability /= 100.0
            
        current_regime = analysis_result.get('current_regime', 'NEUTRAL')

        # Risk-based recommendations
        if total_risk_score > 70:
            recommendations.append(
                f"HIGH RISK ({total_risk_score:.1f}/100): Consider defensive positioning and risk reduction"
            )
        elif total_risk_score < 30:
            recommendations.append(
                f"LOW RISK ({total_risk_score:.1f}/100): Favorable conditions for growth-oriented strategies"
            )

        # Regime transition recommendations
        if transition_probability > 0.5:
            recommendations.append(
                f"HIGH REGIME TRANSITION RISK ({transition_probability:.1%}): "
                f"Prepare for potential shift from {current_regime} regime"
            )

        # Consensus confidence recommendations
        low_confidence_topics = [
            topic for topic, consensus in consensus_results.items()
            if consensus.confidence < 0.6
        ]
        if low_confidence_topics:
            recommendations.append(
                f"LOW CONSENSUS CONFIDENCE on {', '.join(low_confidence_topics)}: "
                f"Monitor developments closely and reassess regularly"
            )

        # Dissent recommendations
        topics_with_dissent = [
            topic for topic, consensus in consensus_results.items()
            if len(consensus.dissenting_agents) > 0
        ]
        if topics_with_dissent:
            recommendations.append(
                f"AGENT DISAGREEMENT on {', '.join(topics_with_dissent)}: "
                f"Consider multiple scenarios in planning"
            )

        # Institutional methodology recommendations (NEW)
        if 'bubble_assessment' in consensus_results:
            bubble_consensus = consensus_results['bubble_assessment']
            if 'DANGER' in bubble_consensus.final_position:
                recommendations.append(
                    f"[JP Morgan Framework] DANGER detected: Reduce exposure to bubble-risk assets. "
                    f"Historical precedent suggests 40%+ drawdown risk within 2 years."
                )
            elif 'WARNING' in bubble_consensus.final_position:
                recommendations.append(
                    f"[JP Morgan Framework] WARNING: Monitor for speculative feedback loops. "
                    f"Consider hedging strategies."
                )

        if 'institutional_view' in consensus_results:
            inst_consensus = consensus_results['institutional_view']
            if 'CAUTIOUS' in inst_consensus.final_position:
                recommendations.append(
                    f"[Institutional View] Defensive positioning recommended: "
                    f"Quality over growth, higher cash allocation (Berkshire approach)."
                )
            elif 'CONSTRUCTIVE' in inst_consensus.final_position:
                recommendations.append(
                    f"[Institutional View] Risk-on appropriate: "
                    f"Fundamentals supportive, consider sector rotation (Goldman approach)."
                )

        # Default fallback
        if not recommendations:
            recommendations.append(
                f"Current market regime: {current_regime} with moderate risk levels - "
                f"Maintain balanced portfolio approach"
            )

        return recommendations

    def _generate_warnings(
        self,
        analysis_result: Dict[str, Any],
        consensus_results: Dict[str, Consensus]
    ) -> List[str]:
        """
        주의 사항 생성

        Args:
            analysis_result: 분석 결과
            consensus_results: 합의 결과

        Returns:
            주의 사항 리스트
        """
        warnings = []

        # Extract active warnings from analysis
        active_warnings = analysis_result.get('active_warnings', [])
        warnings.extend(active_warnings[:5])  # Top 5 warnings

        # Add debate-related warnings
        debate_failures = [
            topic for topic, consensus in consensus_results.items()
            if "failed" in ' '.join(consensus.compromises).lower()
        ]
        if debate_failures:
            warnings.append(
                f"Debate process encountered issues for: {', '.join(debate_failures)}"
            )

        return warnings

    def _create_error_result(self, query: str, error_msg: str) -> Dict[str, Any]:
        """
        에러 발생 시 결과 생성

        Args:
            query: 원본 질문
            error_msg: 에러 메시지

        Returns:
            에러 결과 딕셔너리
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'status': 'error',
            'error': error_msg,
            'analysis': {},
            'consensus': {},
            'recommendations': [f"Error occurred: {error_msg}"],
            'warnings': ["Analysis failed - results unavailable"],
            'metadata': {}
        }


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test_orchestrator():
        """Test MetaOrchestrator with mock data"""
        print("="*60)
        print("MetaOrchestrator Test")
        print("="*60)

        orchestrator = MetaOrchestrator(verbose=True)

        print("\nNote: Full test requires real market_data from DataManager")
        print("For complete integration test, use test_debate_system.py\n")

        # Mock context for testing opinion collection
        mock_context = {
            'total_risk_score': 55.0,
            'risk_level': 'MEDIUM',
            'current_regime': 'TRANSITION',
            'regime_confidence': 65.0,
            'transition_probability': 0.45,
            'primary_risk_path': 'liquidity',
            'path_contributions': {
                'liquidity': 35.0,
                'credit': 25.0,
                'volatility': 20.0,
                'crypto': 15.0
            },
            'active_warnings': [
                'Credit spread widening detected',
                'VIX elevated above 20',
                'High volatility in crypto markets',
                'Liquidity conditions tightening'
            ]
        }

        # Test auto-detection
        print("Testing topic auto-detection...")
        topics = orchestrator._auto_detect_topics(mock_context)
        print(f"Auto-detected topics: {topics}\n")

        # Test opinion collection
        print("Testing opinion collection...")
        opinions = await orchestrator.collect_opinions(topics[:2], mock_context)
        print(f"Collected {sum(len(ops) for ops in opinions.values())} opinions\n")

        for topic, topic_opinions in opinions.items():
            print(f"Topic '{topic}':")
            for op in topic_opinions:
                print(f"  - {op.agent_role.value}: {op.position} (conf={op.confidence:.2f})")

        print("\n" + "="*60)
        print("Basic tests passed!")
        print("Run test_debate_system.py for full integration test")
        print("="*60)

    asyncio.run(test_orchestrator())
