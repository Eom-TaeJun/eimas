"""
Integration Design Validation
==============================
Microstructure + Bubble Detector 통합 설계 검증

설계안을 Claude와 Perplexity로 검증
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import APIConfig
from datetime import datetime
import json


# =============================================================================
# 통합 설계안
# =============================================================================

DESIGN_PROPOSAL = """
## EIMAS Integration Design: Microstructure + Bubble Detection

### 1. Architecture Overview

Current EIMAS Pipeline (Phase 2):
```
Phase 2.1  → RegimeDetector
Phase 2.1.1 → GMM & Entropy (optional)
Phase 2.2  → EventDetector (liquidity, market events)
Phase 2.3  → LiquidityAnalyzer (Granger Causality)
Phase 2.4  → CriticalPathAggregator (risk score)
Phase 2.5  → ETFFlowAnalyzer
... (2.6-2.10)
```

### 2. Proposed Integration Points

**Option A: Sequential After Event Detection**
```
Phase 2.2    → EventDetector
Phase 2.2.1  → DailyMicrostructureAnalyzer (NEW)
Phase 2.2.2  → BubbleDetector (NEW)
Phase 2.3    → LiquidityAnalyzer
```

**Option B: Parallel with Event Detection**
```
Phase 2.2a → EventDetector
Phase 2.2b → MicrostructureAnalyzer (parallel)
Phase 2.2c → BubbleDetector (parallel, async)
```

**Option C: As Risk Enhancement Layer**
```
Phase 2.4    → CriticalPathAggregator
Phase 2.4.1  → MicrostructureRiskEnhancer (NEW)
Phase 2.4.2  → BubbleRiskOverlay (NEW)
→ Combine all risk signals into unified risk score
```

### 3. EIMASResult Fields Design

**Option A: Separate detailed fields**
```python
@dataclass
class EIMASResult:
    # Existing fields...

    # NEW: Market Microstructure
    microstructure_analysis: Dict = field(default_factory=dict)
    # Contains: {ticker: {liquidity_score, amihud_lambda, roll_spread, vpin, toxicity_level}}

    # NEW: Bubble Detection
    bubble_detection: Dict = field(default_factory=dict)
    # Contains: {ticker: {warning_level, risk_score, runup, volatility_spike, issuance}}

    # NEW: Aggregated Risk Flags
    market_quality_score: float = 0.0  # 0-100, higher = better liquidity
    bubble_risk_tickers: List[str] = field(default_factory=list)  # WARNING/DANGER level tickers
```

**Option B: Summary-focused fields**
```python
@dataclass
class EIMASResult:
    # NEW: Summary fields only
    microstructure_summary: str = ""  # "High liquidity (avg score 85.2)"
    bubble_risk_status: str = "NONE"  # NONE/WATCH/WARNING/DANGER
    bubble_risk_tickers: List[str] = field(default_factory=list)
```

**Option C: Nested Risk Object**
```python
@dataclass
class MarketQualityMetrics:
    avg_liquidity_score: float
    low_liquidity_tickers: List[str]
    high_toxicity_tickers: List[str]

@dataclass
class BubbleRiskMetrics:
    overall_status: str
    warning_tickers: List[Dict]  # [{ticker, level, runup_pct, risk_score}]

@dataclass
class EIMASResult:
    # NEW: Structured risk metrics
    market_quality: MarketQualityMetrics = None
    bubble_risk: BubbleRiskMetrics = None
```

### 4. Report Generation Design

**3. Risk Assessment Section (Enhanced)**
```markdown
## 3. Risk Assessment

- **Risk Score**: 45.2/100
- **Risk Level**: MEDIUM
- **Liquidity Signal**: NEUTRAL

### Market Microstructure Quality
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Avg Liquidity Score | 78.5 | Good |
| High Toxicity Count | 2 tickers | Monitor |
| Illiquid Tickers | COIN, ONDO | Caution |

### Bubble Risk Assessment
⚠️ **BUBBLE WARNING DETECTED**
| Ticker | Level | 2Y Return | Volatility Z | Risk Score |
|--------|-------|-----------|--------------|------------|
| NVDA | WARNING | +1094.6% | +2.3σ | 72.5 |
| SMH | WATCH | +156.2% | +1.5σ | 45.2 |

> **Alert**: NVDA shows bubble characteristics (run-up > 100%, volatility spike).
> Consider reducing position or hedging.
```

### 5. Exception Handling Strategy

```python
# Pattern: Graceful Degradation
try:
    microstructure_results = analyzer.analyze_multiple(market_data)
    result.microstructure_analysis = microstructure_results
except Exception as e:
    logger.warning(f"Microstructure analysis failed: {e}")
    result.microstructure_analysis = {'error': str(e), 'status': 'UNAVAILABLE'}
    # Continue pipeline - don't break for optional analysis
```

### 6. Quick Mode Behavior

```python
if not quick_mode:
    # Run microstructure and bubble detection
    # These are computationally expensive
else:
    # Skip - mark as N/A in report
    result.microstructure_summary = "Skipped (quick mode)"
    result.bubble_risk_status = "N/A"
```

### 7. Key Design Questions

1. Should microstructure metrics influence the final risk score calculation?
2. Should bubble warnings trigger automatic position sizing adjustments in recommendations?
3. Should we cache microstructure results (expensive computation)?
4. How to handle missing data for some tickers?
"""

VALIDATION_QUESTIONS = """
Please evaluate this integration design for a financial analysis system:

1. **Architecture Placement**: Which option (A/B/C) is most appropriate for integrating
   microstructure analysis and bubble detection into an existing analysis pipeline?
   - Option A: Sequential after event detection
   - Option B: Parallel execution
   - Option C: As risk enhancement layer

2. **Data Structure**: Which EIMASResult field design is better for maintainability
   and extensibility?
   - Option A: Separate detailed fields
   - Option B: Summary-focused fields
   - Option C: Nested risk objects

3. **Economic Rationale**: Is it appropriate to combine microstructure metrics
   (Amihud Lambda, Roll Spread, VPIN) with bubble detection (Greenwood-Shleifer)
   in a single risk assessment framework?

4. **Report Integration**: Should bubble warnings be prominently displayed in the
   Risk Assessment section, or should they have their own dedicated section?

5. **Missing considerations**: Are there any critical aspects I'm missing in this
   integration design?

Please provide specific recommendations with economic/technical justification.
"""


def query_perplexity(question: str) -> str:
    """Perplexity로 최신 정보 검색"""
    try:
        client = APIConfig.get_client('perplexity')
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative finance systems architect. Provide specific, actionable recommendations."
                },
                {"role": "user", "content": question}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Perplexity Error] {str(e)}"


def query_claude(question: str, context: str = "") -> str:
    """Claude로 설계 검증"""
    try:
        client = APIConfig.get_client('anthropic')
        full_prompt = f"{context}\n\n{question}" if context else question
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"[Claude Error] {str(e)}"


def main():
    print("=" * 70)
    print("EIMAS Integration Design Validation")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # API 키 확인
    api_status = APIConfig.validate()
    if not api_status.get('anthropic') or not api_status.get('perplexity'):
        print("\n[ERROR] Required APIs not available!")
        return

    # 1. Perplexity로 최신 아키텍처 트렌드 검색
    print("\n[1/3] Researching best practices via Perplexity...")
    perplexity_query = """
    What are the current best practices (2024-2025) for integrating market microstructure
    analysis into quantitative trading systems?

    Specifically:
    1. How should Amihud illiquidity, Roll spread, and VPIN be combined with other risk metrics?
    2. What is the recommended pipeline architecture for real-time vs batch analysis?
    3. How do hedge funds typically integrate bubble detection signals into risk management?

    Please cite academic sources or industry practices.
    """
    perplexity_result = query_perplexity(perplexity_query)
    print("\n[Perplexity Research]")
    print("-" * 50)
    print(perplexity_result[:2000] + "..." if len(perplexity_result) > 2000 else perplexity_result)

    # 2. Claude로 설계안 검증
    print("\n[2/3] Validating design with Claude...")
    claude_result = query_claude(VALIDATION_QUESTIONS, DESIGN_PROPOSAL)
    print("\n[Claude Validation]")
    print("-" * 50)
    print(claude_result)

    # 3. 최종 권고사항 요약
    print("\n[3/3] Generating final recommendation...")
    summary_query = """
    Based on the research and validation above, provide a final recommendation:

    1. Recommended architecture option (A/B/C) and why
    2. Recommended data structure option and why
    3. Top 3 implementation priorities
    4. Any critical warnings or considerations

    Keep it concise (bullet points).
    """
    final_rec = query_claude(summary_query, f"{perplexity_result}\n\n{claude_result}")
    print("\n[Final Recommendation]")
    print("-" * 50)
    print(final_rec)

    # 결과 저장
    output_file = f"outputs/integration_design_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'design_proposal': DESIGN_PROPOSAL,
            'perplexity_research': perplexity_result,
            'claude_validation': claude_result,
            'final_recommendation': final_rec,
            'timestamp': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Validation complete. Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
