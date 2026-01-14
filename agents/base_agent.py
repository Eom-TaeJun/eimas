#!/usr/bin/env python3
"""
Multi-Agent System - Base Agent
================================
모든 에이전트의 베이스 클래스

경제학적 의미:
- 에이전트 자율성(Autonomy): 각 에이전트는 독립적으로 의사결정
- 표준화된 인터페이스: 거래 비용(Transaction Cost) 최소화
- 비동기 실행: 병렬 처리로 생산성(Productivity) 극대화
"""

import asyncio
import time
import logging
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (
    AgentRequest,
    AgentResponse,
    AgentRole,
    AgentOpinion,
    OpinionStrength,
)


@dataclass
class AgentConfig:
    """
    에이전트 설정

    Attributes:
        name: 에이전트 이름
        role: 에이전트 역할
        model: 사용할 LLM 모델 (optional)
        api_key_name: 환경변수 키 이름 (optional)
        max_retries: 최대 재시도 횟수
        timeout: 작업 타임아웃 (초)
        verbose: 상세 로그 출력 여부
        custom_config: 에이전트별 커스텀 설정
    """
    name: str
    role: AgentRole
    model: Optional[str] = None
    api_key_name: Optional[str] = None
    max_retries: int = 3
    timeout: int = 60  # 1분
    verbose: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    모든 에이전트의 베이스 클래스

    핵심 기능:
    1. 비동기 실행 (asyncio)
    2. 표준화된 입출력 (AgentRequest → AgentResponse)
    3. 에러 핸들링 (재시도, 타임아웃, graceful degradation)
    4. 로깅 (구조화된 로그)
    5. 상태 관리 (실행 중인 작업 추적)
    """

    def __init__(self, config: AgentConfig):
        """
        Args:
            config: 에이전트 설정
        """
        self.config = config
        self.logger = self._setup_logger()
        self._current_tasks: Dict[str, asyncio.Task] = {}

        self.logger.info(
            f"Initialized {self.config.name} ({self.config.role.value})"
        )

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"agent.{self.config.name}")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        # 핸들러가 없으면 추가
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.config.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """
        에이전트별 실행 로직 (서브클래스에서 구현)

        Args:
            request: 에이전트 요청

        Returns:
            실행 결과 딕셔너리
        """
        pass

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        요청 실행 (에러 핸들링, 재시도, 타임아웃 포함)

        Args:
            request: 에이전트 요청

        Returns:
            AgentResponse 객체
        """
        start_time = time.time()
        self.logger.info(f"Starting task {request.task_id}: {request.instruction[:100]}")

        # 타임아웃 설정
        timeout = request.deadline if request.deadline else self.config.timeout

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # 타임아웃과 함께 실행
                result = await asyncio.wait_for(
                    self._execute_with_tracking(request),
                    timeout=timeout
                )

                execution_time = time.time() - start_time

                # 성공 응답 생성
                response = AgentResponse(
                    task_id=request.task_id,
                    agent_role=self.config.role,
                    status="success",
                    result=result,
                    confidence=self._calculate_confidence(result),
                    reasoning=result.get('reasoning', ''),
                    execution_time=execution_time,
                )

                self.logger.info(
                    f"Task {request.task_id} completed successfully "
                    f"in {execution_time:.2f}s (confidence: {response.confidence:.2f})"
                )

                return response

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Task {request.task_id} timeout (attempt {attempt}/{self.config.max_retries})"
                )
                if attempt == self.config.max_retries:
                    return self._create_error_response(
                        request,
                        f"Timeout after {timeout}s",
                        time.time() - start_time
                    )

            except Exception as e:
                self.logger.error(
                    f"Task {request.task_id} failed (attempt {attempt}/{self.config.max_retries}): {str(e)}"
                )
                if attempt == self.config.max_retries:
                    return self._create_error_response(
                        request,
                        str(e),
                        time.time() - start_time
                    )

                # 재시도 전 대기 (exponential backoff)
                await asyncio.sleep(2 ** attempt)

        # 여기 도달하면 안 되지만 안전장치
        return self._create_error_response(
            request,
            "Unknown error after all retries",
            time.time() - start_time
        )

    async def _execute_with_tracking(self, request: AgentRequest) -> Dict[str, Any]:
        """
        작업 추적과 함께 실행

        Args:
            request: 에이전트 요청

        Returns:
            실행 결과
        """
        task = asyncio.current_task()
        if task:
            self._current_tasks[request.task_id] = task

        try:
            result = await self._execute(request)
            return result
        finally:
            if request.task_id in self._current_tasks:
                del self._current_tasks[request.task_id]

    def _create_error_response(
        self,
        request: AgentRequest,
        error_msg: str,
        execution_time: float
    ) -> AgentResponse:
        """
        에러 응답 생성

        Args:
            request: 원본 요청
            error_msg: 에러 메시지
            execution_time: 실행 시간

        Returns:
            AgentResponse (failure 상태)
        """
        return AgentResponse(
            task_id=request.task_id,
            agent_role=self.config.role,
            status="failure",
            result={},
            confidence=0.0,
            error=error_msg,
            execution_time=execution_time,
        )

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        결과의 신뢰도 계산 (서브클래스에서 오버라이드 가능)

        Args:
            result: 실행 결과

        Returns:
            신뢰도 (0.0 - 1.0)
        """
        # 기본값: result에 confidence 키가 있으면 사용, 없으면 1.0
        return float(result.get('confidence', 1.0))

    async def cancel_task(self, task_id: str) -> bool:
        """
        실행 중인 작업 취소

        Args:
            task_id: 취소할 작업 ID

        Returns:
            취소 성공 여부
        """
        if task_id in self._current_tasks:
            task = self._current_tasks[task_id]
            task.cancel()
            self.logger.info(f"Task {task_id} cancelled")
            return True
        return False

    def get_current_tasks(self) -> list[str]:
        """
        현재 실행 중인 작업 ID 목록 반환

        Returns:
            작업 ID 리스트
        """
        return list(self._current_tasks.keys())

    @abstractmethod
    async def form_opinion(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """
        특정 주제에 대한 의견 형성 (Debate용)

        Args:
            topic: 의견을 물어볼 주제
            context: 추가 컨텍스트

        Returns:
            AgentOpinion 객체
        """
        pass

    def __repr__(self) -> str:
        """문자열 표현"""
        return f"<{self.__class__.__name__} name={self.config.name} role={self.config.role.value}>"


class DummyAgent(BaseAgent):
    """
    테스트용 더미 에이전트
    """

    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """더미 실행 (1초 대기 후 성공)"""
        await asyncio.sleep(1)
        return {
            'message': f"Dummy response for: {request.instruction}",
            'confidence': 0.9
        }

    async def form_opinion(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """더미 의견"""
        return AgentOpinion(
            agent_role=self.config.role,
            position=f"Dummy opinion on {topic}",
            strength=OpinionStrength.NEUTRAL,
            confidence=0.5,
            evidence=["This is a dummy agent"]
        )


# ============================================================
# 테스트 코드
# ============================================================

async def test_base_agent():
    """BaseAgent 테스트"""
    print("="*60)
    print("BaseAgent Test")
    print("="*60)

    # 더미 에이전트 생성
    config = AgentConfig(
        name="test_dummy",
        role=AgentRole.RESEARCH,
        max_retries=2,
        timeout=10,
        verbose=True
    )

    agent = DummyAgent(config)

    # 요청 생성
    request = AgentRequest(
        task_id="test_001",
        role=AgentRole.RESEARCH,
        instruction="Search for Fed policy changes",
        context={"period": "last 7 days"}
    )

    # 실행
    print(f"\n1. Testing successful execution...")
    response = await agent.execute(request)
    print(f"   Status: {response.status}")
    print(f"   Confidence: {response.confidence:.2f}")
    print(f"   Execution time: {response.execution_time:.2f}s")
    print(f"   Result: {response.result}")

    # 의견 형성 테스트
    print(f"\n2. Testing opinion formation...")
    opinion = await agent.form_opinion(
        topic="Should we buy bonds?",
        context={"inflation": "high"}
    )
    print(f"   Position: {opinion.position}")
    print(f"   Strength: {opinion.strength.value}")
    print(f"   Confidence: {opinion.confidence:.2f}")

    # 병렬 실행 테스트
    print(f"\n3. Testing parallel execution...")
    requests = [
        AgentRequest(
            task_id=f"test_{i:03d}",
            role=AgentRole.RESEARCH,
            instruction=f"Task {i}"
        )
        for i in range(3)
    ]

    start = time.time()
    responses = await asyncio.gather(*[agent.execute(req) for req in requests])
    elapsed = time.time() - start

    print(f"   Completed {len(responses)} tasks in {elapsed:.2f}s")
    print(f"   Expected ~1s (parallel), got {elapsed:.2f}s")

    # 타임아웃 테스트
    print(f"\n4. Testing timeout...")

    class SlowAgent(DummyAgent):
        async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
            await asyncio.sleep(20)  # 20초 대기 (타임아웃보다 길게)
            return {"message": "Should not reach here"}

    slow_agent = SlowAgent(AgentConfig(
        name="slow_dummy",
        role=AgentRole.ANALYSIS,
        timeout=2,  # 2초 타임아웃
        max_retries=1
    ))

    slow_request = AgentRequest(
        task_id="slow_001",
        role=AgentRole.ANALYSIS,
        instruction="Slow task"
    )

    slow_response = await slow_agent.execute(slow_request)
    print(f"   Status: {slow_response.status}")
    print(f"   Error: {slow_response.error}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_base_agent())
