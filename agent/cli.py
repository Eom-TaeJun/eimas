#!/usr/bin/env python3
"""
Economic Insight Agent - CLI
=============================

Run the agent on a JSON request file and print JSON output.

Usage:
    python -m agent.cli request.json
    python -m agent.cli request.json --output report.json
    python -m agent.cli --question "Fed 금리 인상이 시장에 미치는 영향은?"
    python -m agent.cli --with-eimas  # EIMAS 모듈 결과 활용

Examples:
    # 템플릿 기반 분석
    python -m agent.cli --question "스테이블코인 공급 증가가 국채 수요에 미치는 영향은?"

    # JSON 파일 입력
    echo '{"question": "Fed 금리 인상 영향"}' > request.json
    python -m agent.cli request.json

    # EIMAS 통합 (최신 결과 사용)
    python -m agent.cli --with-eimas --question "현재 시장 상황 분석"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agent.schemas.insight_schema import InsightRequest, AnalysisFrame
from agent.core.orchestrator import EconomicInsightOrchestrator


def find_latest_eimas_result() -> Optional[dict]:
    """outputs/ 디렉토리에서 최신 integrated JSON 찾기"""
    outputs_dir = Path(__file__).parent.parent / "outputs"

    if not outputs_dir.exists():
        return None

    json_files = list(outputs_dir.glob("integrated_*.json"))
    if not json_files:
        return None

    latest = max(json_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {latest}: {e}", file=sys.stderr)
        return None


def parse_eimas_to_modules(eimas_result: dict) -> dict:
    """EIMAS 전체 결과를 모듈별 결과로 변환"""
    modules = {}

    # shock_propagation_graph 결과 추출
    if 'shock_propagation' in eimas_result:
        modules['shock_propagation'] = eimas_result['shock_propagation']

    # critical_path 결과
    if 'regime' in eimas_result:
        modules['critical_path'] = {
            'current_regime': eimas_result.get('regime', {}).get('market_regime'),
            'transition_probability': eimas_result.get('regime', {}).get('transition_probability', 0),
            'total_risk_score': eimas_result.get('risk_score', 0),
            'active_warnings': eimas_result.get('warnings', []),
        }

    # genius_act 결과
    if 'genius_act_regime' in eimas_result:
        modules['genius_act'] = {
            'regime': eimas_result.get('genius_act_regime', 'NEUTRAL'),
            'signals': eimas_result.get('genius_act_signals', []),
            'extended_liquidity': eimas_result.get('fred_summary', {})
        }

    # bubble_detector 결과
    if 'bubble_risk' in eimas_result:
        modules['bubble_detector'] = eimas_result['bubble_risk']

    # portfolio 결과
    if 'portfolio_weights' in eimas_result:
        modules['portfolio'] = {
            'weights': eimas_result.get('portfolio_weights', {}),
            'systemic_risk_nodes': eimas_result.get('systemic_risk_nodes', [])
        }

    return modules


def main():
    parser = argparse.ArgumentParser(
        description="Economic Insight Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "request_file",
        nargs="?",
        help="JSON request file path (optional if --question is provided)"
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Analysis question (alternative to request file)"
    )

    parser.add_argument(
        "-f", "--frame",
        choices=["macro", "markets", "crypto", "mixed"],
        help="Analysis frame hint"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--with-eimas",
        action="store_true",
        help="Use latest EIMAS module results from outputs/"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw EIMAS data in output"
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact JSON output (no indentation)"
    )

    args = parser.parse_args()

    # Build request
    if args.request_file:
        try:
            with open(args.request_file, 'r', encoding='utf-8') as f:
                request_data = json.load(f)
            request = InsightRequest(**request_data)
        except FileNotFoundError:
            print(f"Error: File not found: {args.request_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {args.request_file}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to parse request: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.question:
        frame_hint = AnalysisFrame(args.frame) if args.frame else None
        request = InsightRequest(
            question=args.question,
            frame_hint=frame_hint
        )
    else:
        print("Error: Either request_file or --question is required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Run orchestrator
    orchestrator = EconomicInsightOrchestrator(debug=args.debug)

    if args.with_eimas:
        eimas_result = find_latest_eimas_result()
        if eimas_result:
            print(f"Using EIMAS result: {eimas_result.get('timestamp', 'unknown')}", file=sys.stderr)
            modules = parse_eimas_to_modules(eimas_result)
            report = orchestrator.run_with_eimas_results(request, modules)
        else:
            print("Warning: No EIMAS results found, using template-based analysis", file=sys.stderr)
            report = orchestrator.run(request)
    else:
        report = orchestrator.run(request)

    # Output
    indent = None if args.compact else 2
    json_output = report.model_dump_json(indent=indent)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
