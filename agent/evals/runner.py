#!/usr/bin/env python3
"""
Economic Insight Agent - Eval Runner
=====================================

Run evaluation scenarios and check JSON output validity.

Usage:
    python -m agent.evals.runner                  # Run all scenarios
    python -m agent.evals.runner --scenario S01   # Run specific scenario
    python -m agent.evals.runner --verbose        # Verbose output
"""

import argparse
import sys
import os
from typing import List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.schemas.insight_schema import InsightRequest, EconomicInsightReport
from agent.core.orchestrator import EconomicInsightOrchestrator
from agent.evals.scenarios import SCENARIOS, EvalScenario, get_scenario


@dataclass
class EvalResult:
    """평가 결과"""
    scenario_id: str
    scenario_name: str
    passed: bool
    checks_total: int
    checks_passed: int
    errors: List[str]
    processing_time_ms: int


def check_report(report: EconomicInsightReport, scenario: EvalScenario) -> Tuple[bool, List[str]]:
    """리포트 유효성 검사"""
    errors = []

    # 1. Required top-level keys
    required_keys = ['meta', 'phenomenon', 'causal_graph', 'mechanisms',
                     'hypotheses', 'risk', 'suggested_data', 'next_actions']

    report_dict = report.model_dump()
    for key in required_keys:
        if key not in report_dict or report_dict[key] is None:
            errors.append(f"Missing required key: {key}")

    # 2. Meta validation
    if not report.meta.request_id:
        errors.append("meta.request_id is empty")
    if not report.meta.timestamp:
        errors.append("meta.timestamp is empty")

    # 3. Frame check
    if scenario.expected_frame and report.meta.frame != scenario.expected_frame:
        errors.append(f"Frame mismatch: expected {scenario.expected_frame.value}, got {report.meta.frame.value}")

    # 4. Graph validation
    if len(report.causal_graph.nodes) < scenario.expected_min_nodes:
        errors.append(f"Too few nodes: expected >= {scenario.expected_min_nodes}, got {len(report.causal_graph.nodes)}")

    if len(report.causal_graph.edges) < scenario.expected_min_edges:
        errors.append(f"Too few edges: expected >= {scenario.expected_min_edges}, got {len(report.causal_graph.edges)}")

    # 5. Mechanisms validation
    if len(report.mechanisms) < scenario.expected_min_mechanisms:
        errors.append(f"Too few mechanisms: expected >= {scenario.expected_min_mechanisms}, got {len(report.mechanisms)}")

    for i, mech in enumerate(report.mechanisms):
        if not mech.nodes:
            errors.append(f"Mechanism {i} has no nodes")
        if not mech.narrative:
            errors.append(f"Mechanism {i} has no narrative")

    # 6. Hypotheses validation
    if not report.hypotheses.main:
        errors.append("Missing main hypothesis")
    if not report.hypotheses.main.statement:
        errors.append("Main hypothesis has no statement")

    # 7. Risk section
    # regime_shift_risks can be empty, just check structure

    # 8. Suggested data (should have at least 1)
    if len(report.suggested_data) < 1:
        errors.append("No suggested data provided")

    # 9. Next actions (3-7 required)
    if len(report.next_actions) < 3:
        errors.append(f"Too few next_actions: expected >= 3, got {len(report.next_actions)}")
    if len(report.next_actions) > 7:
        errors.append(f"Too many next_actions: expected <= 7, got {len(report.next_actions)}")

    # 10. Phenomenon keywords check
    for keyword in scenario.expected_keywords_in_phenomenon:
        if keyword.lower() not in report.phenomenon.lower():
            errors.append(f"Keyword '{keyword}' not found in phenomenon: {report.phenomenon}")

    passed = len(errors) == 0
    return passed, errors


def run_scenario(scenario: EvalScenario, verbose: bool = False) -> EvalResult:
    """단일 시나리오 실행"""
    orchestrator = EconomicInsightOrchestrator(debug=verbose)

    request = InsightRequest(
        question=scenario.question,
        frame_hint=scenario.frame,
        context=scenario.context
    )

    try:
        report = orchestrator.run(request)

        passed, errors = check_report(report, scenario)

        checks_total = 10  # Number of check categories
        checks_passed = checks_total - len(errors)

        return EvalResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            passed=passed,
            checks_total=checks_total,
            checks_passed=checks_passed,
            errors=errors,
            processing_time_ms=report.meta.processing_time_ms or 0
        )

    except Exception as e:
        return EvalResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            passed=False,
            checks_total=10,
            checks_passed=0,
            errors=[f"Exception: {str(e)}"],
            processing_time_ms=0
        )


def run_all_scenarios(verbose: bool = False) -> List[EvalResult]:
    """모든 시나리오 실행"""
    results = []

    for scenario in SCENARIOS:
        if verbose:
            print(f"Running {scenario.id}: {scenario.name}...")

        result = run_scenario(scenario, verbose)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status} ({result.checks_passed}/{result.checks_total}) - {result.processing_time_ms}ms")
            if not result.passed:
                for err in result.errors:
                    print(f"    - {err}")

    return results


def print_summary(results: List[EvalResult]):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nScenarios: {passed}/{total} passed")

    # Details
    print("\nDetails:")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.scenario_id}: {r.scenario_name} "
              f"({r.checks_passed}/{r.checks_total}) - {r.processing_time_ms}ms")

    # Failed details
    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailed scenarios:")
        for r in failed:
            print(f"\n  {r.scenario_id}: {r.scenario_name}")
            for err in r.errors:
                print(f"    - {err}")

    # Overall status
    print("\n" + "=" * 60)
    if passed == total:
        print("ALL SCENARIOS PASSED")
    else:
        print(f"FAILED: {total - passed} scenarios need attention")
    print("=" * 60)

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Economic Insight Agent Eval Runner")

    parser.add_argument(
        "-s", "--scenario",
        type=str,
        help="Run specific scenario by ID (e.g., S01)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    if args.scenario:
        scenario = get_scenario(args.scenario)
        if not scenario:
            print(f"Error: Scenario '{args.scenario}' not found", file=sys.stderr)
            print(f"Available: {[s.id for s in SCENARIOS]}", file=sys.stderr)
            sys.exit(1)

        result = run_scenario(scenario, verbose=args.verbose)
        results = [result]
    else:
        results = run_all_scenarios(verbose=args.verbose)

    if args.json:
        import json
        output = [
            {
                "id": r.scenario_id,
                "name": r.scenario_name,
                "passed": r.passed,
                "checks": f"{r.checks_passed}/{r.checks_total}",
                "time_ms": r.processing_time_ms,
                "errors": r.errors
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        all_passed = print_summary(results)
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
