"""
Methodology Validation Script
=============================
Claude + Perplexity API를 사용하여 구현 방향성 검증

검증 대상:
1. Stablecoin Collateral Risk Assessment (genius_act_macro.py)
2. MST-based Systemic Risk Identification (graph_clustered_portfolio.py)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import APIConfig


def query_perplexity(question: str) -> str:
    """Perplexity API로 최신 정보 검색"""
    try:
        client = APIConfig.get_client('perplexity')

        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial research assistant. Provide accurate, up-to-date information with academic and industry sources."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=2000,
            temperature=0.1
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"[Perplexity Error] {str(e)}"


def query_claude(question: str, context: str = "") -> str:
    """Claude API로 방법론 검증"""
    try:
        client = APIConfig.get_client('anthropic')

        full_prompt = f"{context}\n\n{question}" if context else question

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )

        return response.content[0].text
    except Exception as e:
        return f"[Claude Error] {str(e)}"


def validate_stablecoin_risk():
    """Task 1: Stablecoin Collateral Risk Assessment 검증"""
    print("\n" + "=" * 70)
    print("VALIDATION 1: Stablecoin Collateral Risk Assessment")
    print("=" * 70)

    # 구현 내용 요약
    implementation = """
    Implementation Summary (genius_act_macro.py):

    1. StablecoinCollateralType Enum:
       - TREASURY_CASH: USDC (Circle) - Treasury & Cash backed
       - MIXED_RESERVE: USDT (Tether) - Mixed collateral
       - CRYPTO_BACKED: DAI - Crypto-collateralized
       - DERIVATIVE_HEDGE: USDe (Ethena) - Uses derivative hedging
       - ALGORITHMIC: Algorithmic stablecoins (high risk)

    2. Risk Scoring Logic:
       - Base risk scores: USDC=15, USDT=35, DAI=40, USDe=50, ALGO=80
       - Interest-paying stablecoins (like USDe) get +15 regulatory penalty
         (SEC securities classification risk)
       - Collateral transparency reduces risk
       - Peg stability affects risk score

    3. Key Differentiation:
       - USDC: Treasury/Cash backing → Low counterparty risk
       - USDe: Derivative hedging + Interest → Higher regulatory risk
    """

    # Perplexity로 최신 정보 검색
    print("\n[1/3] Searching latest stablecoin risk research via Perplexity...")
    perplexity_query = """
    What are the current best practices for assessing stablecoin collateral risk in 2024-2025?
    Specifically:
    1. How do institutions differentiate USDC vs USDe (Ethena) risk?
    2. What regulatory considerations apply to interest-bearing stablecoins?
    3. How is collateral type (treasury vs crypto vs derivatives) typically weighted in risk models?

    Please cite recent academic papers, regulatory guidance, or industry standards.
    """

    perplexity_result = query_perplexity(perplexity_query)
    print("\n[Perplexity Research Result]")
    print("-" * 50)
    print(perplexity_result[:2000] + "..." if len(perplexity_result) > 2000 else perplexity_result)

    # Claude로 방법론 검증
    print("\n[2/3] Validating methodology with Claude...")
    claude_query = """
    Please evaluate the following stablecoin risk assessment methodology.
    Is the approach directionally correct? What improvements would you suggest?

    Rate the approach: [CORRECT / PARTIALLY_CORRECT / NEEDS_REVISION]

    Specific questions:
    1. Is the base risk ordering (USDC < USDT < DAI < USDe) reasonable?
    2. Is the +15 penalty for interest-paying stablecoins justified?
    3. Are there any critical risk factors missing?
    """

    claude_result = query_claude(claude_query, implementation)
    print("\n[Claude Validation Result]")
    print("-" * 50)
    print(claude_result)

    return {
        "task": "Stablecoin Collateral Risk Assessment",
        "perplexity_research": perplexity_result,
        "claude_validation": claude_result,
        "timestamp": datetime.now().isoformat()
    }


def validate_mst_systemic_risk():
    """Task 2: MST-based Systemic Risk Identification 검증"""
    print("\n" + "=" * 70)
    print("VALIDATION 2: MST-based Systemic Risk Identification")
    print("=" * 70)

    # 구현 내용 요약
    implementation = """
    Implementation Summary (graph_clustered_portfolio.py):

    1. Distance Matrix Construction:
       - d = sqrt(2 * (1 - rho)) where rho is correlation
       - rho = 1 → d = 0 (perfect correlation)
       - rho = 0 → d = sqrt(2) ≈ 1.414 (uncorrelated)
       - rho = -1 → d = 2 (inverse correlation)

    2. MST (Minimum Spanning Tree) Construction:
       - Uses NetworkX's minimum_spanning_tree with Prim/Kruskal
       - MST connects all assets with minimum total distance
       - Reference: Mantegna (1999) "Hierarchical structure in financial markets"

    3. Systemic Risk Node Identification:
       - Centrality metrics calculated on MST:
         * Degree Centrality (30% weight)
         * Betweenness Centrality (35% weight) - highest, for shock propagation
         * Closeness Centrality (15% weight)
         * Eigenvector Centrality (20% weight)
       - Top 3 nodes with highest composite score = systemic risk factors

    4. Economic Interpretation:
       - High betweenness = Bridge node, critical for shock transmission
       - High degree = Hub node, connected to many assets
       - These nodes require careful monitoring in portfolio
    """

    # Perplexity로 최신 정보 검색
    print("\n[1/3] Searching latest MST network analysis research via Perplexity...")
    perplexity_query = """
    What are the current best practices for using MST (Minimum Spanning Tree)
    in financial network analysis for systemic risk identification in 2024-2025?

    Specifically:
    1. Is the distance formula d = sqrt(2*(1-rho)) standard in financial network literature?
    2. Which centrality measures are most relevant for identifying systemic risk nodes?
    3. How do practitioners weight different centrality metrics?
    4. What are the limitations of MST-based analysis?

    Please cite Mantegna (1999), Onnela et al., or other key references.
    """

    perplexity_result = query_perplexity(perplexity_query)
    print("\n[Perplexity Research Result]")
    print("-" * 50)
    print(perplexity_result[:2000] + "..." if len(perplexity_result) > 2000 else perplexity_result)

    # Claude로 방법론 검증
    print("\n[2/3] Validating methodology with Claude...")
    claude_query = """
    Please evaluate the following MST-based systemic risk identification methodology.
    Is the approach directionally correct? What improvements would you suggest?

    Rate the approach: [CORRECT / PARTIALLY_CORRECT / NEEDS_REVISION]

    Specific questions:
    1. Is the distance formula d = sqrt(2*(1-rho)) academically sound?
    2. Is the centrality weight distribution (Degree 30%, Betweenness 35%,
       Closeness 15%, Eigenvector 20%) reasonable?
    3. Are there any critical aspects of MST analysis missing?
    4. Is focusing on top 3 nodes sufficient for systemic risk monitoring?
    """

    claude_result = query_claude(claude_query, implementation)
    print("\n[Claude Validation Result]")
    print("-" * 50)
    print(claude_result)

    return {
        "task": "MST-based Systemic Risk Identification",
        "perplexity_research": perplexity_result,
        "claude_validation": claude_result,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """메인 실행"""
    print("=" * 70)
    print("EIMAS Methodology Validation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # API 키 확인
    api_status = APIConfig.validate()
    print("\n[API Status]")
    for api, available in api_status.items():
        status = "✓" if available else "✗"
        print(f"  {api}: {status}")

    if not api_status.get('anthropic') or not api_status.get('perplexity'):
        print("\n[ERROR] Required APIs (anthropic, perplexity) not available!")
        return

    results = []

    # Task 1: Stablecoin Risk
    result1 = validate_stablecoin_risk()
    results.append(result1)

    # Task 2: MST Systemic Risk
    result2 = validate_mst_systemic_risk()
    results.append(result2)

    # 결과 저장
    output_file = f"outputs/methodology_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Validation complete. Results saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
