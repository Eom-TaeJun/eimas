#!/usr/bin/env python3
"""
EIMAS Debate Agent
==================
Uses Perplexity API to interpret signals and generate debate-style analysis.
Produces dated markdown reports with final conclusions.

Usage:
    from lib.debate_agent import DebateAgent

    agent = DebateAgent()
    report = agent.generate_report(signals, consensus)
"""

import os
import httpx
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Get API key from environment
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


@dataclass
class DebatePosition:
    """A position in the debate"""
    stance: str  # "bullish", "bearish", "neutral"
    argument: str
    confidence: float
    supporting_signals: List[str]


@dataclass
class DebateResult:
    """Result of the debate"""
    bull_case: DebatePosition
    bear_case: DebatePosition
    synthesis: str
    final_recommendation: str
    conviction: float
    key_risks: List[str]
    key_opportunities: List[str]


class DebateAgent:
    """Agent that debates market signals and generates reports"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai"

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Perplexity API synchronously"""
        if not self.api_key:
            return self._mock_response(user_prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"  Warning: Perplexity API error: {e}")
            return self._mock_response(user_prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when API is unavailable"""
        if "bull" in prompt.lower():
            return """Based on the current signals, the bullish case rests on:
1. Low VIX indicating market calm
2. Extreme Fear in sentiment (contrarian opportunity)
3. Bull market regime with low volatility

However, caution is warranted due to yield curve signals."""
        elif "bear" in prompt.lower():
            return """The bearish case is supported by:
1. Critical Path warnings on yield curve inversion
2. Breakevens at elevated levels suggesting inflation concerns
3. Multiple warning signals from risk indicators

Risk management should be prioritized."""
        else:
            return """Synthesis: Mixed signals suggest a cautious approach.
While regime indicators are positive, structural warnings from critical paths
suggest reducing exposure and maintaining hedges. Focus on quality assets
and maintain flexibility for market dislocations."""

    def generate_bull_case(self, signals: List[Dict], consensus: Dict) -> DebatePosition:
        """Generate the bullish argument"""
        print("  Generating bull case...")

        signal_summary = self._format_signals(signals, filter_action="buy")

        system_prompt = """You are a bullish market analyst. Your job is to make
the strongest possible case for why the market will go up based on the given signals.
Be specific and cite the signals that support your view. Keep response concise."""

        user_prompt = f"""Current Market Signals:
{signal_summary}

Overall Consensus: {consensus.get('action', 'N/A')} ({consensus.get('conviction', 0):.0%} conviction)

Make the BULLISH case. What signals support buying? What opportunities exist?
Focus on specific data points and avoid generic statements."""

        response = self._call_api(system_prompt, user_prompt)

        # Extract supporting signals
        buy_signals = [s['reasoning'] for s in signals if s.get('action') == 'buy']

        return DebatePosition(
            stance="bullish",
            argument=response,
            confidence=consensus.get('conviction', 0.5),
            supporting_signals=buy_signals[:3]
        )

    def generate_bear_case(self, signals: List[Dict], consensus: Dict) -> DebatePosition:
        """Generate the bearish argument"""
        print("  Generating bear case...")

        signal_summary = self._format_signals(signals, filter_action="reduce")

        system_prompt = """You are a bearish market analyst. Your job is to make
the strongest possible case for why caution is warranted based on the given signals.
Be specific and cite the signals that support your view. Keep response concise."""

        user_prompt = f"""Current Market Signals:
{signal_summary}

Overall Consensus: {consensus.get('action', 'N/A')} ({consensus.get('conviction', 0):.0%} conviction)

Make the BEARISH case. What signals suggest risk? What should investors be concerned about?
Focus on specific data points and avoid generic statements."""

        response = self._call_api(system_prompt, user_prompt)

        # Extract supporting signals
        risk_signals = [s['reasoning'] for s in signals
                       if s.get('action') in ['reduce', 'sell', 'hedge']]

        return DebatePosition(
            stance="bearish",
            argument=response,
            confidence=consensus.get('conviction', 0.5),
            supporting_signals=risk_signals[:3]
        )

    def synthesize_debate(self, bull: DebatePosition, bear: DebatePosition,
                         signals: List[Dict], consensus: Dict) -> DebateResult:
        """Synthesize bull and bear cases into final recommendation"""
        print("  Synthesizing debate...")

        system_prompt = """You are a senior investment strategist synthesizing
opposing views. Provide a balanced, actionable conclusion. Be specific about
what investors should do and at what conviction level."""

        user_prompt = f"""BULL CASE:
{bull.argument}

BEAR CASE:
{bear.argument}

CURRENT CONSENSUS: {consensus.get('action', 'N/A')} ({consensus.get('conviction', 0):.0%})

Provide:
1. A synthesis that weighs both arguments
2. A clear final recommendation (BUY/HOLD/REDUCE/HEDGE)
3. Key risks to monitor
4. Key opportunities to watch

Be concise and actionable."""

        synthesis = self._call_api(system_prompt, user_prompt)

        # Extract key points
        key_risks = self._extract_risks(signals)
        key_opportunities = self._extract_opportunities(signals)

        return DebateResult(
            bull_case=bull,
            bear_case=bear,
            synthesis=synthesis,
            final_recommendation=consensus.get('action', 'hold').upper(),
            conviction=consensus.get('conviction', 0.5),
            key_risks=key_risks,
            key_opportunities=key_opportunities
        )

    def _format_signals(self, signals: List[Dict], filter_action: str = None) -> str:
        """Format signals into readable text"""
        lines = []
        for s in signals:
            if filter_action and s.get('action') != filter_action:
                continue
            source = s.get('source', 'unknown')
            action = s.get('action', 'N/A')
            conviction = s.get('conviction', 0)
            reasoning = s.get('reasoning', '')
            lines.append(f"- [{source}] {action.upper()} ({conviction:.0%}): {reasoning}")

        return "\n".join(lines) if lines else "No matching signals"

    def _extract_risks(self, signals: List[Dict]) -> List[str]:
        """Extract key risks from signals"""
        risks = []
        for s in signals:
            if s.get('action') in ['reduce', 'sell', 'hedge']:
                reasoning = s.get('reasoning', '')
                if 'CRITICAL' in reasoning.upper():
                    risks.append(reasoning.split(':')[-1].strip() if ':' in reasoning else reasoning)
        return risks[:5]

    def _extract_opportunities(self, signals: List[Dict]) -> List[str]:
        """Extract opportunities from signals"""
        opps = []
        for s in signals:
            if s.get('action') == 'buy':
                reasoning = s.get('reasoning', '')
                opps.append(reasoning)
        return opps[:3]

    def generate_report(self, signals: List[Dict], consensus: Dict) -> str:
        """Generate full debate report as markdown"""
        print("\n" + "=" * 60)
        print("EIMAS Debate Agent")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\nStarting debate analysis...")

        # Run debate
        bull = self.generate_bull_case(signals, consensus)
        bear = self.generate_bear_case(signals, consensus)
        result = self.synthesize_debate(bull, bear, signals, consensus)

        # Generate markdown
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_time = datetime.now().strftime("%H:%M:%S")

        md = f"""# EIMAS Market Analysis Report
## Date: {report_date}
### Generated at {report_time} UTC

---

## Executive Summary

**Final Recommendation:** {result.final_recommendation}
**Conviction Level:** {result.conviction:.0%}

---

## Signal Summary

| Source | Action | Conviction | Reasoning |
|--------|--------|------------|-----------|
"""
        for s in signals:
            source = s.get('source', 'unknown')
            action = s.get('action', 'N/A')
            conviction = s.get('conviction', 0)
            reasoning = s.get('reasoning', '')[:50] + '...' if len(s.get('reasoning', '')) > 50 else s.get('reasoning', '')
            md += f"| {source} | {action.upper()} | {conviction:.0%} | {reasoning} |\n"

        md += f"""

---

## Debate Analysis

### Bull Case (Optimistic View)

{result.bull_case.argument}

**Supporting Signals:**
"""
        for sig in result.bull_case.supporting_signals:
            md += f"- {sig}\n"

        md += f"""

### Bear Case (Cautious View)

{result.bear_case.argument}

**Warning Signals:**
"""
        for sig in result.bear_case.supporting_signals:
            md += f"- {sig}\n"

        md += f"""

---

## Synthesis & Conclusion

{result.synthesis}

---

## Key Risk Factors

"""
        for i, risk in enumerate(result.key_risks, 1):
            md += f"{i}. {risk}\n"

        md += """

## Key Opportunities

"""
        for i, opp in enumerate(result.key_opportunities, 1):
            md += f"{i}. {opp}\n"

        md += f"""

---

## Appendix: Consensus Details

- **Winning Action:** {consensus.get('action', 'N/A').upper()}
- **Action Scores:**
"""
        for action, score in consensus.get('action_scores', {}).items():
            md += f"  - {action}: {score:.2f}\n"

        md += f"""- **Total Signals:** {consensus.get('signal_count', 0)}
- **Primary Reasoning:** {consensus.get('reasoning', 'N/A')}

---

*This report was generated by EIMAS (Economic Intelligence Multi-Agent System)*
*Debate powered by Perplexity AI*
"""

        print("  Report generated successfully")
        return md

    def save_report(self, report: str, output_dir: str = "outputs") -> str:
        """Save report to dated markdown file"""
        import os

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H%M")
        filename = f"market_report_{date_str}_{time_str}.md"
        filepath = os.path.join(output_dir, filename)

        # Write report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"  Report saved to: {filepath}")
        return filepath


# ============================================================================
# Convenience Functions
# ============================================================================

def run_full_pipeline() -> str:
    """Run complete pipeline: signals -> debate -> report"""
    from lib.signal_pipeline import SignalPipeline

    print("=" * 70)
    print("EIMAS Full Pipeline: Signal Collection + Debate + Report")
    print("=" * 70)

    # 1. Collect signals
    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()

    # Convert signals to dict format
    signal_dicts = []
    for s in signals:
        signal_dicts.append({
            'source': s.source.value,
            'action': s.action.value,
            'ticker': s.ticker,
            'conviction': s.conviction,
            'reasoning': s.reasoning,
            'metadata': s.metadata
        })

    # 2. Run debate
    agent = DebateAgent()
    report = agent.generate_report(signal_dicts, consensus)

    # 3. Save report
    filepath = agent.save_report(report)

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nReport saved to: {filepath}")

    return filepath


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    filepath = run_full_pipeline()

    # Print report preview
    print("\n" + "=" * 70)
    print("Report Preview (first 100 lines)")
    print("=" * 70)

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[:100]:
            print(line.rstrip())
