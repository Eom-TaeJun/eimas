
import sys
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ai_report_generator import AIReportGenerator
from lib.whitening_engine import WhiteningEngine
from lib.autonomous_agent import AutonomousFactChecker

logger = logging.getLogger('eimas.pipeline.report')

async def run_report_and_qa(result: Any, market_data: Dict, output_file: str, quick_mode: bool) -> Optional[str]:
    """
    Phase 6 & 7: AI Report and Quality Assurance
    """
    print("\n" + "=" * 50)
    print("PHASE 6: AI REPORT GENERATION")
    print("=" * 50)

    report_path = None
    try:
        generator = AIReportGenerator(verbose=True)
        print("\n[6.1] Generating AI-powered investment report...")
        report = await generator.generate(result.to_dict(), market_data)
        print("\n[6.2] Saving report...")
        report_path = await generator.save_report(report)
        print(f"\n      ✓ AI Report saved: {report_path}")
    except Exception as e:
        print(f"      ✗ AI Report error: {e}")

    if not quick_mode:
        print("\n" + "=" * 50)
        print("PHASE 7: WHITENING & FACT CHECK")
        print("=" * 50)

        # 7.1 Whitening
        print("\n[7.1] Economic whitening analysis...")
        try:
            whitening = WhiteningEngine()
            portfolio_result = {
                'allocation': result.portfolio_weights if result.portfolio_weights else {'SPY': 0.3, 'QQQ': 0.2, 'TLT': 0.15},
                'changes': {}, 'returns': {}
            }
            explanation = whitening.explain_allocation(portfolio_result)
            result.whitening_summary = explanation.summary
            print(f"      ✓ Summary: {explanation.summary[:100]}...")
        except Exception as e:
            print(f"      ✗ Whitening error: {e}")

        # 7.2 Fact Check
        print("\n[7.2] Fact checking AI outputs...")
        try:
            fact_checker = AutonomousFactChecker(use_perplexity=False, verbose=False)
            check_text = f"""
            Current regime is {result.regime.get('regime', 'Unknown')}.
            Risk score is {result.risk_score:.1f} out of 100.
            Recommendation is {result.final_recommendation}.
            """
            check_result = await fact_checker.verify_document(check_text, max_claims=5)
            result.fact_check_grade = check_result['summary']['grade']
            print(f"      ✓ Grade: {result.fact_check_grade}")
        except Exception as e:
            print(f"      ✗ Fact check error: {e}")

    return report_path
