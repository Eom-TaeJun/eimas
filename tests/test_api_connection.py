"""
API Connection Test - Multi-AI API 연결 테스트

각 AI Provider(Claude, OpenAI, Gemini, Perplexity) 연결 상태 확인
DebateFramework 통합 테스트 포함

Author: EIMAS Team
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Tuple

from core.debate_framework import (
    AIClient,
    AIProvider,
    DebateParticipant,
    get_default_participants
)


# ============================================================================
# Individual API Tests
# ============================================================================

async def test_claude_api() -> Tuple[bool, str]:
    """Claude API 연결 테스트"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return False, "ANTHROPIC_API_KEY not set"

    try:
        participant = DebateParticipant(
            name="Claude",
            provider=AIProvider.CLAUDE,
            model="claude-sonnet-4-20250514"
        )
        client = AIClient(participant)

        response = await client.complete(
            prompt="Say 'API connection successful' in exactly 3 words.",
            system_prompt="You are a helpful assistant. Respond concisely."
        )

        if response and len(response) > 0:
            return True, f"OK - Response: {response[:50]}..."
        return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


async def test_openai_api() -> Tuple[bool, str]:
    """OpenAI API 연결 테스트"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set"

    try:
        participant = DebateParticipant(
            name="OpenAI",
            provider=AIProvider.OPENAI,
            model="gpt-4o-mini"  # 비용 절약을 위해 mini 사용
        )
        client = AIClient(participant)

        response = await client.complete(
            prompt="Say 'API connection successful' in exactly 3 words.",
            system_prompt="You are a helpful assistant. Respond concisely."
        )

        if response and len(response) > 0:
            return True, f"OK - Response: {response[:50]}..."
        return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


async def test_gemini_api() -> Tuple[bool, str]:
    """Gemini API 연결 테스트"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY not set"

    try:
        participant = DebateParticipant(
            name="Gemini",
            provider=AIProvider.GEMINI,
            model="gemini-1.5-flash"  # 비용 절약
        )
        client = AIClient(participant)

        response = await client.complete(
            prompt="Say 'API connection successful' in exactly 3 words.",
            system_prompt="You are a helpful assistant. Respond concisely."
        )

        if response and len(response) > 0:
            return True, f"OK - Response: {response[:50]}..."
        return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


async def test_perplexity_api() -> Tuple[bool, str]:
    """Perplexity API 연결 테스트"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return False, "PERPLEXITY_API_KEY not set"

    try:
        participant = DebateParticipant(
            name="Perplexity",
            provider=AIProvider.PERPLEXITY,
            model="sonar"  # 비용 절약
        )
        client = AIClient(participant)

        response = await client.complete(
            prompt="Say 'API connection successful' in exactly 3 words.",
            system_prompt="You are a helpful assistant. Respond concisely."
        )

        if response and len(response) > 0:
            return True, f"OK - Response: {response[:50]}..."
        return False, "Empty response"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


# ============================================================================
# Integrated Tests
# ============================================================================

async def test_all_apis() -> Dict[str, Tuple[bool, str]]:
    """모든 API 병렬 테스트"""
    print("\n" + "=" * 60)
    print("Testing All AI APIs...")
    print("=" * 60 + "\n")

    results = {}

    # 병렬 실행
    tasks = [
        ("Claude", test_claude_api()),
        ("OpenAI", test_openai_api()),
        ("Gemini", test_gemini_api()),
        ("Perplexity", test_perplexity_api())
    ]

    for name, task in tasks:
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            success, message = await task
            results[name] = (success, message)
            status = "✓" if success else "✗"
            print(f"{status} {message[:60]}")
        except Exception as e:
            results[name] = (False, str(e))
            print(f"✗ Exception: {str(e)[:60]}")

    return results


async def test_simple_debate() -> bool:
    """간단한 토론 테스트 (3개 AI 참여)"""
    print("\n" + "=" * 60)
    print("Testing Simple Multi-AI Debate...")
    print("=" * 60 + "\n")

    # 사용 가능한 API 확인
    api_status = {
        "Claude": bool(os.getenv("ANTHROPIC_API_KEY")),
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Gemini": bool(os.getenv("GOOGLE_API_KEY"))
    }

    available = [name for name, ok in api_status.items() if ok]
    print(f"Available APIs: {available}")

    if len(available) < 2:
        print("Need at least 2 APIs for debate test")
        return False

    # 간단한 토론 - 각 AI에게 같은 질문
    prompt = """
    As an economic analyst, briefly answer:
    What is the most important economic indicator to watch in 2025?

    Respond in this format:
    INDICATOR: [name]
    REASON: [brief reason in 1-2 sentences]
    CONFIDENCE: [0.0-1.0]
    """

    responses = {}

    for name in available[:3]:  # 최대 3개만 테스트
        provider = {
            "Claude": AIProvider.CLAUDE,
            "OpenAI": AIProvider.OPENAI,
            "Gemini": AIProvider.GEMINI
        }[name]

        model = {
            "Claude": "claude-sonnet-4-20250514",
            "OpenAI": "gpt-4o-mini",
            "Gemini": "gemini-1.5-flash"
        }[name]

        participant = DebateParticipant(
            name=name,
            provider=provider,
            model=model,
            system_prompt="You are an economic analyst."
        )

        client = AIClient(participant)

        print(f"\nAsking {name}...", end=" ", flush=True)
        try:
            response = await client.complete(prompt)
            responses[name] = response
            print("Done")
            print(f"  Response preview: {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
            responses[name] = None

    # 결과 분석
    successful = sum(1 for r in responses.values() if r)
    print(f"\n✓ {successful}/{len(responses)} APIs responded successfully")

    return successful >= 2


# ============================================================================
# Environment Check
# ============================================================================

def check_environment() -> Dict[str, bool]:
    """환경변수 확인"""
    print("\n" + "=" * 60)
    print("Checking Environment Variables...")
    print("=" * 60 + "\n")

    env_vars = {
        "ANTHROPIC_API_KEY": "Claude",
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Gemini",
        "PERPLEXITY_API_KEY": "Perplexity"
    }

    results = {}
    for var, name in env_vars.items():
        value = os.getenv(var)
        is_set = bool(value)
        masked = f"{value[:8]}...{value[-4:]}" if value and len(value) > 12 else "[not set]"

        status = "✓" if is_set else "✗"
        print(f"  {status} {var}: {masked}")
        results[name] = is_set

    return results


# ============================================================================
# Main
# ============================================================================

async def main():
    """전체 테스트 실행"""
    print("\n" + "=" * 60)
    print(f"  EIMAS API Connection Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 환경변수 확인
    env_status = check_environment()

    # 2. 개별 API 테스트
    api_results = await test_all_apis()

    # 3. 통합 토론 테스트
    debate_ok = await test_simple_debate()

    # 4. 최종 결과
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60 + "\n")

    print("Environment:")
    for name, ok in env_status.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print("\nAPI Connections:")
    for name, (ok, msg) in api_results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print(f"\nDebate Test: {'✓ Passed' if debate_ok else '✗ Failed'}")

    # 최종 판정
    all_ok = all(ok for ok, _ in api_results.values())
    if all_ok and debate_ok:
        print("\n✓ All tests passed! Ready for production.")
        return 0
    else:
        working = sum(1 for ok, _ in api_results.values() if ok)
        print(f"\n⚠ {working}/4 APIs working. Check failed APIs before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
