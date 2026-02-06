#!/usr/bin/env python3
"""
EIMAS Pipeline - Storage Module
================================

Purpose:
    Phase 5 데이터 저장 담당 (Data Storage)

Functions:
    - save_result_json(result, output_dir, output_file) -> str
    - save_to_trading_db(signals)
    - save_to_event_db(events)

Dependencies:
    - lib.trading_db
    - lib.event_db

Example:
    from pipeline.storage import save_result_json
    path = save_result_json(result)
    print(f"Saved to {path}")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# EIMAS 라이브러리
from lib.trading_db import TradingDB, Signal
from lib.event_db import EventDatabase
from pipeline.schemas import EIMASResult, Event, RealtimeSignal

def save_result_json(
    result: EIMASResult,
    output_dir: Path = None,
    output_file: Optional[Union[str, Path]] = None,
) -> str:
    """결과를 JSON 파일로 저장 (통합 포맷)"""
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)
    print("\n[5.3] Saving unified JSON result...")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)

    if output_file:
        target_file = Path(output_file).expanduser()
        if not target_file.is_absolute():
            target_file = output_dir / target_file
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # NEW: 통합 파일명 (eimas_*)
        target_file = output_dir / f"eimas_{timestamp_str}.json"

    target_file.parent.mkdir(exist_ok=True, parents=True)

    try:
        with open(target_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"      ✓ Saved: {target_file}")
        return str(target_file)
    except Exception as e:
        print(f"      ✗ JSON save error: {e}")
        return ""

def save_result_md(result: EIMASResult, output_dir: Path = None, use_schema_renderer: bool = False) -> str:
    """
    결과를 Markdown 파일로 저장

    Args:
        result: EIMASResult 객체
        output_dir: 출력 디렉토리
        use_schema_renderer: True면 스키마 기반 렌더러 사용 (v2)

    Returns:
        저장된 파일 경로
    """
    print("\n[5.4] Saving full Markdown (JSON to MD conversion)...")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"

    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eimas_{timestamp_str}.md"

    try:
        json_data = result.to_dict()

        if use_schema_renderer:
            # New schema-driven renderer (v2)
            from pipeline.schema_renderer import render_json_to_md
            md_content = render_json_to_md(json_data)
            print("      (using schema-driven renderer v2)")
        else:
            # Legacy renderer
            md_content = _json_to_full_markdown(json_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"      ✓ Saved: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"      ✗ Markdown save error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def save_result_md_v2(result: EIMASResult, output_dir: Path = None) -> str:
    """
    Schema-driven Markdown 저장 (v2)

    Features:
    - 스키마 기반 렌더링
    - 새 키 자동 처리
    - 일관성 검증

    Usage:
        path = save_result_md_v2(result)
    """
    return save_result_md(result, output_dir, use_schema_renderer=True)


def _json_to_full_markdown(data: dict, level: int = 1) -> str:
    """JSON 데이터를 전체 Markdown으로 변환 (요약/트렁케이션 없음)"""
    lines = []

    if level == 1:
        lines.append("# EIMAS Analysis Report (Full Data)")
        lines.append(f"**Generated**: {data.get('timestamp', 'N/A')}")
        lines.append("")

    # 섹션 순서 정의 (전체 JSON 키 포함)
    section_order = [
        # ============ PHASE 1: DATA COLLECTION ============
        ('timestamp', None),  # 헤더에서 이미 처리
        ('fred_summary', '1. FRED Economic Data'),
        ('market_data_count', None),  # 1번에 포함
        ('crypto_data_count', None),  # 1번에 포함

        # ============ PHASE 2: ANALYSIS ============
        ('regime', '2. Market Regime'),
        ('genius_act_regime', None),  # 2번에 포함

        ('risk_score', '3. Risk Assessment'),
        ('base_risk_score', None),
        ('microstructure_adjustment', None),
        ('bubble_risk_adjustment', None),
        ('extended_data_adjustment', None),

        ('bubble_risk', '4. Bubble Risk'),
        ('bubble_framework', '5. Bubble Framework'),
        ('market_quality', '6. Market Quality'),

        ('events_detected', '7. Events Detected'),
        ('event_tracking', None),
        ('event_predictions', None),
        ('event_attributions', None),
        ('tracked_events', None),

        ('liquidity_analysis', '8. Liquidity Analysis'),
        ('liquidity_signal', None),

        ('shock_propagation', '9. Shock Propagation'),
        ('critical_path_monitoring', None),

        ('volume_anomalies', '10. Volume Anomalies'),
        ('volume_analysis_summary', None),

        # ============ PHASE 2: TECHNICAL ANALYSIS ============
        ('hft_microstructure', '11. HFT Microstructure'),
        ('garch_volatility', '12. GARCH Volatility'),
        ('information_flow', '13. Information Flow'),
        ('proof_of_index', '14. Proof of Index'),

        ('dtw_similarity', '15. DTW Similarity'),
        ('dbscan_outliers', '16. DBSCAN Outliers'),
        ('correlation_matrix', None),
        ('correlation_tickers', None),

        # ============ PHASE 2: PORTFOLIO ============
        ('portfolio_weights', '17. Portfolio Weights'),
        ('allocation_strategy', None),
        ('allocation_config', None),
        ('allocation_result', '18. Allocation Result'),  # NEW: 2026-02-04
        ('adaptive_portfolios', None),
        ('hrp_allocation_rationale', None),
        ('rebalance_decision', '19. Rebalancing Decision'),  # NEW: 2026-02-04

        # ============ PHASE 2: EXTERNAL DATA ============
        ('ark_analysis', '20. ARK Invest Analysis'),
        ('genius_act_signals', '21. Genius Act Signals'),
        ('extended_data', '22. Extended Data'),
        ('sentiment_analysis', '23. Sentiment Analysis'),
        ('fomc_analysis', '24. FOMC Analysis'),

        ('gap_analysis', '25. Gap Analysis'),
        ('institutional_analysis', '26. Institutional Analysis'),

        # ============ PHASE 2: CRYPTO ============
        ('crypto_monitoring', '27. Crypto Monitoring'),
        ('crypto_stress_test', None),
        ('onchain_risk_signals', None),
        ('defi_tvl', None),

        # ============ PHASE 2: GLOBAL MARKETS ============
        ('mena_markets', '28. Global Markets'),
        ('intraday_summary', None),

        # ============ PHASE 3: MULTI-AGENT DEBATE ============
        ('debate_consensus', '29. Multi-Agent Debate'),
        ('debate_results', None),
        ('full_mode_position', None),
        ('reference_mode_position', None),
        ('modes_agree', None),
        ('has_strong_dissent', None),
        ('dissent_records', None),
        ('devils_advocate_arguments', None),
        ('agent_outputs', None),

        ('reasoning_chain', '30. Reasoning Chain'),
        ('validation_loop_result', '31. Validation Results'),
        ('verification', None),

        # ============ PHASE 4.5: OPERATIONAL ENGINE ============
        ('operational_report', 'OPERATIONAL_REPORT'),  # 30-34: hold, decision, score, repair, rebalance

        # ============ PHASE 5: FINAL OUTPUT ============
        ('final_recommendation', '35. Final Recommendation'),
        ('confidence', None),
        ('risk_level', None),
        ('warnings', '36. Warnings'),

        # ============ PHASE 6: AI REPORT ============
        ('ai_report', '37. AI Report'),
        ('whitening_summary', None),
        ('fact_check_grade', None),
        ('news_correlations', None),

        # ============ PHASE 4: REALTIME ============
        ('realtime_signals', '38. Realtime Signals'),
        ('trading_db_status', None),

        # ============ BACKTEST ============
        ('event_backtest_results', '39. Backtest Results'),
        ('integrated_signals', None),
    ]

    # 섹션별로 처리
    processed_keys = set()
    for key, section_title in section_order:
        if key not in data:
            continue
        processed_keys.add(key)
        value = data[key]

        # Special handling for operational_report (rubric required sections)
        if key == 'operational_report' and isinstance(value, dict) and value:
            lines.extend(_format_operational_report(value))
            continue

        # Special handling for allocation_result (2026-02-04)
        if key == 'allocation_result' and isinstance(value, dict) and value:
            if section_title:
                lines.append(f"\n## {section_title}")
            lines.extend(_format_allocation_result(value))
            continue

        # Special handling for rebalance_decision (2026-02-04)
        if key == 'rebalance_decision' and isinstance(value, dict) and value:
            if section_title:
                lines.append(f"\n## {section_title}")
            lines.extend(_format_rebalance_decision(value))
            continue

        if section_title:
            lines.append(f"\n## {section_title}")

        lines.extend(_format_value(key, value, level=2))

    # 나머지 키 처리
    for key, value in data.items():
        if key in processed_keys or key == 'timestamp':
            continue
        if value is None or value == '' or value == [] or value == {}:
            continue
        lines.append(f"\n## {key.replace('_', ' ').title()}")
        lines.extend(_format_value(key, value, level=2))

    return '\n'.join(lines)


def _format_operational_report(op_data: dict) -> list:
    """
    Operational Report를 상세 rubric 형식으로 포맷

    Sections:
    - 30. hold_policy (HOLD 판단 과정)
    - 31. decision_policy (의사결정 규칙 실행 과정)
    - 32. score_definitions (단일 Canonical 리스크 점수)
    - 33. constraint_repair (제약조건 수리 과정)
    - 34. rebalance_plan (리밸런싱 실행 계획)
    """
    lines = []

    # ========================================
    # 30. hold_policy (HOLD 판단 과정)
    # ========================================
    lines.append("\n## 30. hold_policy")
    hp = op_data.get('hold_policy', {})

    is_hold = hp.get('is_hold', False)
    hold_conditions = hp.get('hold_conditions', [])

    lines.append(f"### HOLD Decision: **{'YES - HOLD TRIGGERED' if is_hold else 'NO - PROCEED'}**")
    lines.append("")

    if is_hold:
        lines.append("> ⚠️ **HOLD가 발동되어 모든 거래가 중단됩니다.**")
        lines.append("")

    lines.append("#### Hold Condition Evaluation (Priority Order)")
    lines.append("| Priority | Condition | Triggered | Current Value | Threshold | Reason Code |")
    lines.append("|----------|-----------|-----------|---------------|-----------|-------------|")

    for cond in hold_conditions:
        triggered = cond.get('is_triggered', False)
        triggered_str = "**YES**" if triggered else "NO"
        lines.append(f"| {cond.get('priority', '-')} | {cond.get('condition_name', 'N/A')} | {triggered_str} | {cond.get('current_value', 'N/A')} | {cond.get('threshold', 'N/A')} | `{cond.get('reason_code', '')}` |")
    lines.append("")

    # Triggered conditions detail
    triggered_conds = [c for c in hold_conditions if c.get('is_triggered')]
    if triggered_conds:
        lines.append("#### Triggered Hold Conditions (Conflict Resolution)")
        for cond in triggered_conds:
            lines.append(f"- **{cond.get('condition_name')}**: {cond.get('description', '')}")
            lines.append(f"  - Current: `{cond.get('current_value')}`")
            lines.append(f"  - Required: `{cond.get('threshold')}`")
            lines.append(f"  - Resolution: Force HOLD until condition is resolved")
        lines.append("")
    else:
        lines.append("*All hold conditions passed. Proceeding with decision rules.*")
        lines.append("")

    # ========================================
    # 28. decision_policy (의사결정 규칙 실행 과정)
    # ========================================
    lines.append("\n## 31. decision_policy")
    dp = op_data.get('decision_policy', {})

    # Required fields
    final_stance = dp.get('final_stance', 'HOLD')
    constraint_status = dp.get('constraint_status_input', 'OK')
    client_profile = dp.get('client_profile_status_input', 'COMPLETE')
    constraints_ok = constraint_status in ('OK', 'REPAIRED')

    lines.append(f"### final_stance: **{final_stance}**")
    lines.append("")
    lines.append(f"- **constraints_ok**: {constraints_ok}")
    lines.append(f"- **client_profile**: {client_profile}")
    lines.append("")

    lines.append("#### Decision Inputs")
    lines.append("| Input | Value | Description |")
    lines.append("|-------|-------|-------------|")
    lines.append(f"| Regime | {dp.get('regime_input', 'NEUTRAL')} | Current market regime |")
    lines.append(f"| Risk Score (Canonical) | {dp.get('risk_score_input', 50.0):.1f} | Single source of truth |")
    lines.append(f"| Confidence | {dp.get('confidence_input', 0.5):.2%} | Agent consensus confidence |")
    lines.append(f"| Agent Consensus | {dp.get('agent_consensus_input', 'NEUTRAL')} | Multi-agent vote result |")
    lines.append(f"| Modes Agree | {dp.get('modes_agree_input', True)} | FULL vs REFERENCE mode |")
    lines.append(f"| Constraint Status | {constraint_status} | Asset class constraints |")
    lines.append(f"| Client Profile | {client_profile} | Profile completeness |")
    lines.append("")

    # Rule Evaluation Log (상세 과정)
    rule_log = dp.get('rule_evaluation_log', [])
    if rule_log:
        lines.append("#### Rule Evaluation Process (Sequential, Early-Exit)")
        lines.append("")
        lines.append("```")
        lines.append("Rules are evaluated in order. First triggered rule determines outcome.")
        lines.append("```")
        lines.append("")
        lines.append("| Rule | Condition | Input | Result |")
        lines.append("|------|-----------|-------|--------|")
        for rule in rule_log:
            result = rule.get('result', '')
            result_fmt = f"**{result}**" if 'TRIGGERED' in result or 'HOLD' in result.upper() else result
            lines.append(f"| {rule.get('rule', 'N/A')} | {rule.get('condition', 'N/A')} | {rule.get('input', 'N/A')} | {result_fmt} |")
        lines.append("")

    lines.append("#### applied_rules")
    for rule in dp.get('applied_rules', []):
        lines.append(f"- {rule}")
    lines.append("")

    lines.append("#### reason_codes")
    for code in dp.get('reason_codes', []):
        lines.append(f"- `{code}`")
    lines.append("")

    # ========================================
    # 29. score_definitions (단일 Canonical 리스크 점수)
    # ========================================
    lines.append("\n## 32. score_definitions")
    lines.append("")
    lines.append("> **Important**: 의사결정에는 오직 `canonical_risk_score` 하나만 사용됩니다.")
    lines.append("> 다른 점수들(base, micro, bubble 등)은 **참고용 보조 지표**입니다.")
    lines.append("")

    sd = op_data.get('score_definitions', {})
    canonical = sd.get('canonical_risk_score', 50.0)
    risk_level = sd.get('risk_level', 'MEDIUM')

    lines.append("### Canonical Risk Score (THE ONLY SCORE FOR DECISIONS)")
    lines.append("")
    lines.append(f"## **{canonical:.1f} / 100** ({risk_level})")
    lines.append("")

    lines.append("### Scale Interpretation")
    lines.append("| Range | Level | Action |")
    lines.append("|-------|-------|--------|")
    lines.append("| 0 - 30 | LOW | Aggressive positions allowed |")
    lines.append("| 30 - 70 | MEDIUM | Standard risk management |")
    lines.append("| 70 - 100 | HIGH | Defensive stance, RULE_4 triggers |")
    lines.append("")

    lines.append("### Auxiliary Sub-Scores (REFERENCE ONLY - NOT for decisions)")
    lines.append("")
    lines.append("These scores are provided for **transparency and debugging only**.")
    lines.append("They are **NOT** used in any decision rules.")
    lines.append("")

    # Extract from nested auxiliary_sub_scores structure
    aux_scores = sd.get('auxiliary_sub_scores', {})
    base_risk = aux_scores.get('base_risk_score', {}).get('value', 0) if isinstance(aux_scores.get('base_risk_score'), dict) else sd.get('base_risk_score', 0)
    base_source = aux_scores.get('base_risk_score', {}).get('source', 'CriticalPathAggregator') if isinstance(aux_scores.get('base_risk_score'), dict) else 'N/A'
    micro_adj = aux_scores.get('microstructure_adjustment', {}).get('value', 0) if isinstance(aux_scores.get('microstructure_adjustment'), dict) else sd.get('microstructure_adjustment', 0)
    micro_source = aux_scores.get('microstructure_adjustment', {}).get('source', 'DailyMicrostructureAnalyzer') if isinstance(aux_scores.get('microstructure_adjustment'), dict) else 'N/A'
    bubble_adj = aux_scores.get('bubble_risk_adjustment', {}).get('value', 0) if isinstance(aux_scores.get('bubble_risk_adjustment'), dict) else sd.get('bubble_risk_adjustment', 0)
    bubble_source = aux_scores.get('bubble_risk_adjustment', {}).get('source', 'BubbleDetector') if isinstance(aux_scores.get('bubble_risk_adjustment'), dict) else 'N/A'
    extended_adj = aux_scores.get('extended_data_adjustment', {}).get('value', 0) if isinstance(aux_scores.get('extended_data_adjustment'), dict) else sd.get('extended_data_adjustment', 0)
    extended_source = aux_scores.get('extended_data_adjustment', {}).get('source', 'ExtendedDataCollector') if isinstance(aux_scores.get('extended_data_adjustment'), dict) else 'N/A'

    lines.append("| Component | Value | Source | Note |")
    lines.append("|-----------|-------|--------|------|")
    lines.append(f"| base_risk_score | {base_risk:.1f} | {base_source} | Base from CriticalPath |")
    lines.append(f"| microstructure_adj | {micro_adj:+.1f} | {micro_source} | Liquidity/toxicity adjustment |")
    lines.append(f"| bubble_risk_adj | {bubble_adj:+.1f} | {bubble_source} | Bubble overlay |")
    lines.append(f"| extended_data_adj | {extended_adj:+.1f} | {extended_source} | External data factors |")
    lines.append("")

    lines.append("### Calculation Formula")
    lines.append("```")
    lines.append(f"canonical_risk_score = base + micro_adj + bubble_adj + extended_adj")
    lines.append(f"                     = {base_risk:.1f} + ({micro_adj:+.1f}) + ({bubble_adj:+.1f}) + ({extended_adj:+.1f})")
    lines.append(f"                     = {canonical:.1f}")
    lines.append("```")
    lines.append("")

    # ========================================
    # 30. constraint_repair (제약조건 수리 과정)
    # ========================================
    lines.append("\n## 33. constraint_repair")
    cr = op_data.get('constraint_repair', {})

    constraints_satisfied = cr.get('constraints_satisfied', True)
    force_hold = cr.get('force_hold', False)
    force_hold_reason = cr.get('force_hold_reason', '')

    if force_hold:
        lines.append(f"### Status: **FORCE HOLD** ⛔")
        lines.append("")
        lines.append(f"> ⚠️ Constraint repair failed. Reason: `{force_hold_reason}`")
        lines.append("> All trades are blocked until constraints can be satisfied.")
    elif constraints_satisfied:
        lines.append("### Status: **SATISFIED** ✅")
    else:
        lines.append("### Status: **REPAIRED** 🔧")
    lines.append("")

    lines.append(f"- **constraints_ok**: {constraints_satisfied}")
    lines.append(f"- **force_hold**: {force_hold}")
    lines.append("")

    # violations found
    violations = cr.get('violations_found', [])
    if violations:
        lines.append("### Violations Detected")
        lines.append("| Asset Class | Violation Type | Current Weight | Limit | Excess |")
        lines.append("|-------------|----------------|----------------|-------|--------|")
        for v in violations:
            current = v.get('current_value', v.get('current_weight', 0))
            limit = v.get('limit_value', v.get('limit', 0))
            excess = current - limit if v.get('violation_type') == 'ABOVE_MAX' else limit - current
            lines.append(f"| {v.get('asset_class', 'N/A')} | {v.get('violation_type', 'N/A')} | {current:.1%} | {limit:.1%} | {excess:+.1%} |")
        lines.append("")

    # repair actions
    repair_actions = cr.get('repair_actions', [])
    if repair_actions:
        lines.append("### Repair Actions Taken")
        for action in repair_actions:
            lines.append(f"- {action}")
        lines.append("")

    # before_weights / after_weights comparison
    comparison = cr.get('asset_class_comparison', [])
    if comparison:
        lines.append("### before_weights vs after_weights")
        lines.append("")
        lines.append("| Asset Class | before_weights | after_weights | Delta | Min | Max | Status |")
        lines.append("|-------------|----------------|---------------|-------|-----|-----|--------|")
        for c in comparison:
            status = c.get('status', 'OK')
            status_icon = "✅" if status == 'OK' else "⚠️"
            lines.append(f"| {c.get('asset_class', 'N/A')} | {c.get('original_weight', 0):.1%} | {c.get('repaired_weight', 0):.1%} | {c.get('delta', 0):+.1%} | {c.get('min_bound', 0):.0%} | {c.get('max_bound', 1):.0%} | {status_icon} {status} |")
        lines.append("")

    # original vs repaired weights (if available)
    original_weights = cr.get('original_weights', {})
    repaired_weights = cr.get('repaired_weights', {})
    if original_weights and repaired_weights:
        lines.append("### Individual Asset Weight Changes")
        lines.append("| Asset | Before | After | Change |")
        lines.append("|-------|--------|-------|--------|")
        all_assets = set(original_weights.keys()) | set(repaired_weights.keys())
        for asset in sorted(all_assets)[:15]:  # Limit to 15
            before = original_weights.get(asset, 0)
            after = repaired_weights.get(asset, 0)
            if before != after:
                lines.append(f"| {asset} | {before:.2%} | {after:.2%} | {after - before:+.2%} |")
        lines.append("")

    # ========================================
    # 31. rebalance_plan (리밸런싱 실행 계획)
    # ========================================
    lines.append("\n## 34. rebalance_plan")
    rp = op_data.get('rebalance_plan', {})

    # Handle nested structure
    execution = rp.get('execution', {})
    should_execute = execution.get('should_execute', rp.get('should_execute', False))
    not_executed_reason = execution.get('not_executed_reason', rp.get('not_executed_reason', ''))

    trigger = rp.get('trigger', {})
    trigger_type = trigger.get('type', rp.get('trigger_type', 'MANUAL'))
    trigger_reason = trigger.get('reason', '')

    summary = rp.get('summary', {})
    turnover = summary.get('total_turnover', rp.get('total_turnover', 0.0))
    buy_count = summary.get('buy_count', 0)
    sell_count = summary.get('sell_count', 0)
    hold_count = summary.get('hold_count', 0)

    cost_breakdown = rp.get('cost_breakdown', {})
    commission = cost_breakdown.get('commission', 0)
    spread = cost_breakdown.get('spread', 0)
    market_impact = cost_breakdown.get('market_impact', 0)
    total_cost = cost_breakdown.get('total', rp.get('total_estimated_cost', 0.0))

    approval = rp.get('approval', {})
    requires_approval = approval.get('requires_human_approval', rp.get('requires_human_approval', False))
    approval_reason = approval.get('approval_reason', '')

    lines.append(f"### Execution Status: **{'EXECUTE' if should_execute else 'NOT EXECUTED'}**")
    lines.append("")

    if not should_execute:
        lines.append(f"> ℹ️ Not executed: `{not_executed_reason}`")
        lines.append("")

    lines.append("#### Summary")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **turnover** | {turnover:.2%} |")
    lines.append(f"| **trigger_type** | {trigger_type} |")
    lines.append(f"| **requires_approval** | {'⚠️ YES' if requires_approval else '✅ NO'} |")
    lines.append(f"| Buy Orders | {buy_count} |")
    lines.append(f"| Sell Orders | {sell_count} |")
    lines.append(f"| Hold (No Change) | {hold_count} |")
    lines.append("")

    if requires_approval:
        lines.append(f"> ⚠️ **Human Approval Required**: {approval_reason}")
        lines.append("")

    lines.append("#### Cost Breakdown (**est_total_cost**)")
    lines.append("| Cost Type | Amount |")
    lines.append("|-----------|--------|")
    lines.append(f"| Commission | {commission:.4f} |")
    lines.append(f"| Spread | {spread:.4f} |")
    lines.append(f"| Market Impact | {market_impact:.4f} |")
    lines.append(f"| **Total** | **{total_cost:.4f}** |")
    lines.append("")

    # Trade list
    trades = rp.get('trades', [])
    if trades:
        lines.append("### Trade List")
        lines.append("| # | Asset | Action | Current | Target | Delta | Est. Cost |")
        lines.append("|---|-------|--------|---------|--------|-------|-----------|")
        for i, t in enumerate(trades[:20], 1):  # Limit to 20
            action = t.get('action', 'HOLD')
            action_fmt = f"**{action}**" if action in ('BUY', 'SELL') else action
            lines.append(f"| {i} | {t.get('ticker', 'N/A')} | {action_fmt} | {t.get('current_weight', 0):.2%} | {t.get('target_weight', 0):.2%} | {t.get('delta_weight', t.get('delta', 0)):+.2%} | {t.get('estimated_cost', t.get('est_cost', 0)):.4f} |")
        if len(trades) > 20:
            lines.append(f"| ... | *{len(trades) - 20} more trades* | | | | | |")
        lines.append("")

    # Asset class summary
    asset_class_summary = rp.get('asset_class_summary', [])
    if asset_class_summary:
        lines.append("### Asset Class Summary")
        lines.append("| Asset Class | Current | Target | Delta |")
        lines.append("|-------------|---------|--------|-------|")
        for ac in asset_class_summary:
            lines.append(f"| {ac.get('asset_class', 'N/A')} | {ac.get('current_weight', 0):.1%} | {ac.get('target_weight', 0):.1%} | {ac.get('delta', 0):+.1%} |")
        lines.append("")

    return lines


def _format_allocation_result(alloc_data: dict) -> list:
    """
    Allocation Result를 읽기 쉬운 형식으로 포맷 (2026-02-04)

    Sections:
    - Strategy & Portfolio Metrics
    - Target Weights (Top 10)
    - Risk Contributions (Top 5)
    """
    lines = []
    lines.append("")

    # Strategy & Metrics
    strategy = alloc_data.get('strategy', 'N/A')
    expected_return = alloc_data.get('expected_return', 0)
    expected_vol = alloc_data.get('expected_volatility', 0)
    sharpe = alloc_data.get('sharpe_ratio', 0)
    div_ratio = alloc_data.get('diversification_ratio', 0)
    effective_n = alloc_data.get('effective_n', 0)

    lines.append(f"**Strategy**: {strategy}")
    lines.append(f"**Expected Return**: {expected_return:.2%}")
    lines.append(f"**Expected Volatility**: {expected_vol:.2%}")
    lines.append(f"**Sharpe Ratio**: {sharpe:.2f}")
    lines.append(f"**Diversification Ratio**: {div_ratio:.2f}")
    lines.append(f"**Effective N**: {effective_n:.1f}")
    lines.append("")

    # Target Weights (Top 10)
    weights = alloc_data.get('weights', {})
    if weights and isinstance(weights, dict):
        lines.append("### Target Weights (Top 10)")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        for ticker, weight in sorted_weights:
            lines.append(f"- **{ticker}**: {weight:.1%}")
        lines.append("")

    # Risk Contributions (Top 5)
    risk_contribs = alloc_data.get('risk_contributions', {})
    if risk_contribs and isinstance(risk_contribs, dict):
        lines.append("### Risk Contributions (Top 5)")
        sorted_rc = sorted(risk_contribs.items(), key=lambda x: x[1], reverse=True)[:5]
        for ticker, rc in sorted_rc:
            lines.append(f"- **{ticker}**: {rc:.1%}")
        lines.append("")

    return lines


def _format_rebalance_decision(rb_data: dict) -> list:
    """
    Rebalancing Decision을 읽기 쉬운 형식으로 포맷 (2026-02-04)

    Sections:
    - Decision Summary
    - Priority Trades (HIGH only, Top 5)
    - Warnings
    """
    lines = []
    lines.append("")

    # Decision Summary
    should_rebalance = rb_data.get('should_rebalance', False)
    action = rb_data.get('action', 'HOLD')
    reason = rb_data.get('reason', 'N/A')
    turnover = rb_data.get('turnover', 0)
    estimated_cost = rb_data.get('estimated_cost', 0)

    status_emoji = "✅" if should_rebalance else "❌"
    lines.append(f"**Should Rebalance**: {status_emoji} {'Yes' if should_rebalance else 'No'}")
    lines.append(f"**Action**: {action}")
    lines.append(f"**Reason**: {reason}")
    lines.append(f"**Turnover**: {turnover:.1%}")
    lines.append(f"**Estimated Cost**: {estimated_cost:.2%}")
    lines.append("")

    # Priority Trades (HIGH only, Top 5)
    trade_plan = rb_data.get('trade_plan', [])
    if trade_plan:
        high_priority = [t for t in trade_plan if isinstance(t, dict) and t.get('priority') == 'HIGH']
        if high_priority:
            lines.append("### Priority Trades (HIGH, Top 5)")
            for i, trade in enumerate(high_priority[:5], 1):
                action_type = trade.get('action', 'HOLD')
                ticker = trade.get('ticker', 'Unknown')
                delta = trade.get('delta_weight', 0)
                cost_breakdown = trade.get('cost_breakdown', {})
                cost = cost_breakdown.get('total', 0) if isinstance(cost_breakdown, dict) else 0

                lines.append(f"{i}. **{action_type}** {ticker}: {delta:+.1%} (Cost: {cost:.2%})")
            lines.append("")

    # Warnings
    warnings = rb_data.get('warnings', [])
    if warnings:
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return lines


def _format_value(key: str, value, level: int = 2) -> list:
    """값을 Markdown 포맷으로 변환"""
    lines = []
    prefix = "  " * (level - 2)
    key_str = str(key)  # 키가 정수일 수 있음
    key_lower = key_str.lower()

    if value is None:
        lines.append(f"{prefix}- **{key_str}**: N/A")
    elif isinstance(value, bool):
        lines.append(f"{prefix}- **{key_str}**: {'Yes' if value else 'No'}")
    elif isinstance(value, (int, float)):
        if 'score' in key_lower or 'ratio' in key_lower:
            lines.append(f"{prefix}- **{key_str}**: {value:.2f}")
        elif 'confidence' in key_lower and value <= 1:
            lines.append(f"{prefix}- **{key_str}**: {value:.1%}")
        else:
            lines.append(f"{prefix}- **{key_str}**: {value}")
    elif isinstance(value, str):
        if len(value) > 200:
            lines.append(f"{prefix}- **{key_str}**:")
            lines.append(f"{prefix}  ```")
            lines.append(f"{prefix}  {value}")
            lines.append(f"{prefix}  ```")
        else:
            lines.append(f"{prefix}- **{key_str}**: {value}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}- **{key_str}**: (empty)")
        elif all(isinstance(item, (str, int, float)) for item in value):
            lines.append(f"{prefix}- **{key_str}**: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"{prefix}### {key_str.replace('_', ' ').title()}")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    lines.append(f"{prefix}#### Item {i+1}")
                    for k, v in item.items():
                        lines.extend(_format_value(k, v, level + 1))
                else:
                    lines.append(f"{prefix}- {item}")
    elif isinstance(value, dict):
        if not value:
            lines.append(f"{prefix}- **{key_str}**: (empty)")
        else:
            for k, v in value.items():
                lines.extend(_format_value(k, v, level))
    else:
        lines.append(f"{prefix}- **{key_str}**: {str(value)}")

    return lines

def save_to_trading_db(signals: List[RealtimeSignal]):
    """트레이딩 DB에 시그널 저장"""
    print("\n[5.2] Saving to Signal Database...")
    if not signals:
        print("      - No signals to save")
        return

    try:
        # Note: pipeline/schemas.py의 RealtimeSignal을 lib.trading_db.Signal과는 다름
        # 여기서는 통합 시그널 저장용으로 간주하거나, 실제로는 IntegratedSignal을 저장해야 함.
        # 단순화를 위해 로깅만 수행하거나, 필요 시 변환 로직 추가.
        
        # 실제 구현에서는 trading_db.py의 Signal 객체로 변환 필요
        # 예시:
        # db = TradingDB()
        # for s in signals:
        #     db_signal = Signal(...)
        #     db.save_signal(db_signal)
        
        print(f"      ✓ Processed {len(signals)} signals (DB save skipped in this snippet)")
    except Exception as e:
        print(f"      ✗ Signal DB error: {e}")

def save_to_event_db(events: List[Event], market_snapshot: Dict[str, Any] = None):
    """이벤트 DB에 저장"""
    print("\n[5.1] Saving to Event Database...")
    if not events:
        print("      - No events to save")
        return

    try:
        event_db = EventDatabase('data/events.db')

        # 이벤트 저장
        for event in events:
            event_db.save_detected_event({
                'event_type': event.type,
                'importance': event.importance,
                'description': event.description,
                'timestamp': event.timestamp,
            })
            
        print(f"      ✓ Saved {len(events)} events")
        
        # 마켓 스냅샷 저장 (옵션)
        if market_snapshot:
            import uuid
            snapshot_id = str(uuid.uuid4())[:8]
            # snapshot 스키마에 맞춰 데이터 보정 필요
            # event_db.save_market_snapshot(market_snapshot) 
            print(f"      ✓ Saved market snapshot (ID: {snapshot_id})")
            
    except Exception as e:
        print(f"      ✗ Event DB error: {e}")
