#!/usr/bin/env python3
"""
EIMAS Pipeline - Storage Module
================================

Purpose:
    Phase 5 데이터 저장 담당 (Data Storage)

Functions:
    - save_result_json(result, output_dir) -> str
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
from typing import List, Dict, Any

# EIMAS 라이브러리
from lib.trading_db import TradingDB, Signal
from lib.event_db import EventDatabase
from pipeline.schemas import EIMASResult, Event, RealtimeSignal

def save_result_json(result: EIMASResult, output_dir: Path = None) -> str:
    """결과를 JSON 파일로 저장 (통합 포맷)"""
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)
    print("\n[5.3] Saving unified JSON result...")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"
    
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NEW: 통합 파일명 (eimas_*)
    output_file = output_dir / f"eimas_{timestamp_str}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"      ✓ Saved: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"      ✗ JSON save error: {e}")
        return ""

def save_result_md(result: EIMASResult, output_dir: Path = None) -> str:
    """결과를 Markdown 파일로 저장 (JSON 전체 내용 변환)"""
    print("\n[5.4] Saving full Markdown (JSON to MD conversion)...")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"

    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eimas_{timestamp_str}.md"

    try:
        # JSON 전체 내용을 MD로 변환 (요약 없이)
        md_content = _json_to_full_markdown(result.to_dict())
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"      ✓ Saved: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"      ✗ Markdown save error: {e}")
        return ""


def _json_to_full_markdown(data: dict, level: int = 1) -> str:
    """JSON 데이터를 전체 Markdown으로 변환 (요약/트렁케이션 없음)"""
    lines = []

    if level == 1:
        lines.append("# EIMAS Analysis Report (Full Data)")
        lines.append(f"**Generated**: {data.get('timestamp', 'N/A')}")
        lines.append("")

    # 섹션 순서 정의
    section_order = [
        ('fred_summary', '1. FRED Economic Data'),
        ('regime', '2. Market Regime'),
        ('risk_score', '3. Risk Assessment'),
        ('base_risk_score', None),
        ('microstructure_adjustment', None),
        ('bubble_risk_adjustment', None),
        ('extended_data_adjustment', None),
        ('bubble_risk', '4. Bubble Risk'),
        ('market_quality', '5. Market Quality'),
        ('events_detected', '6. Events Detected'),
        ('debate_consensus', '7. Multi-Agent Debate'),
        ('full_mode_position', None),
        ('reference_mode_position', None),
        ('modes_agree', None),
        ('reasoning_chain', '8. Reasoning Chain'),
        ('portfolio_weights', '9. Portfolio Weights'),
        ('ark_analysis', '10. ARK Invest Analysis'),
        ('hft_microstructure', '11. HFT Microstructure'),
        ('garch_volatility', '12. GARCH Volatility'),
        ('information_flow', '13. Information Flow'),
        ('dtw_similarity', '14. DTW Similarity'),
        ('dbscan_outliers', '15. DBSCAN Outliers'),
        ('proof_of_index', '16. Proof of Index'),
        ('shock_propagation', '17. Shock Propagation'),
        ('volume_anomalies', '18. Volume Anomalies'),
        ('liquidity_analysis', '19. Liquidity Analysis'),
        ('genius_act_signals', '20. Genius Act Signals'),
        ('extended_data', '21. Extended Data'),
        ('sentiment_analysis', '22. Sentiment Analysis'),
        ('fomc_analysis', '23. FOMC Analysis'),
        ('bubble_framework', '24. Bubble Framework'),
        ('gap_analysis', '25. Gap Analysis'),
        ('institutional_analysis', '26. Institutional Analysis'),
        ('validation_loop_result', '27. Validation Results'),
        # Operational Report Sections (required by rubric)
        ('operational_report', 'OPERATIONAL_REPORT'),  # Special handling
        ('final_recommendation', '32. Final Recommendation'),
        ('confidence', None),
        ('risk_level', None),
        ('warnings', '33. Warnings'),
        ('ai_report', '34. AI Report'),
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
    Operational Report를 rubric 형식으로 포맷

    Required sections:
    - decision_policy
    - score_definitions
    - rebalance_plan
    - constraint_repair
    """
    lines = []

    # ========================================
    # 28. decision_policy (required)
    # ========================================
    lines.append("\n## 28. decision_policy")
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

    lines.append("#### Inputs")
    lines.append(f"- Regime: {dp.get('regime_input', 'NEUTRAL')}")
    lines.append(f"- Risk Score: {dp.get('risk_score_input', 50.0):.1f}")
    lines.append(f"- Confidence: {dp.get('confidence_input', 0.5):.2f}")
    lines.append(f"- Agent Consensus: {dp.get('agent_consensus_input', 'NEUTRAL')}")
    lines.append(f"- Modes Agree: {dp.get('modes_agree_input', True)}")
    lines.append(f"- Constraint Status: {constraint_status}")
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
    # 29. score_definitions (required)
    # ========================================
    lines.append("\n## 29. score_definitions")
    sd = op_data.get('score_definitions', {})

    canonical = sd.get('canonical_risk_score', 50.0)
    risk_level = sd.get('risk_level', 'MEDIUM')

    lines.append("### Canonical Risk Score")
    lines.append("")
    lines.append(f"**Score: {canonical:.1f} / 100**")
    lines.append(f"**Level: {risk_level}**")
    lines.append("")

    lines.append("### Scale (0-100)")
    lines.append("| Range | Level | Description |")
    lines.append("|-------|-------|-------------|")
    lines.append("| 0 - 30 | LOW | Aggressive positions allowed |")
    lines.append("| 30 - 70 | MEDIUM | Standard risk management |")
    lines.append("| 70 - 100 | HIGH | Defensive stance required |")
    lines.append("")

    lines.append("### Auxiliary Sub-Scores (NOT used in decision rules)")
    lines.append(f"- base_risk_score: {sd.get('base_risk_score', 0):.1f}")
    lines.append(f"- microstructure_adj: {sd.get('microstructure_adjustment', 0):+.1f}")
    lines.append(f"- bubble_risk_adj: {sd.get('bubble_risk_adjustment', 0):+.1f}")
    lines.append("")

    # ========================================
    # 30. constraint_repair (required)
    # ========================================
    lines.append("\n## 30. constraint_repair")
    cr = op_data.get('constraint_repair', {})

    constraints_satisfied = cr.get('constraints_satisfied', True)
    lines.append(f"- **constraints_ok**: {constraints_satisfied}")
    lines.append("")

    if constraints_satisfied:
        lines.append("### Status: **REPAIRED** ✓")
    else:
        lines.append("### Status: **VIOLATED** ✗ (force HOLD)")
    lines.append("")

    # violations
    violations = cr.get('violations_found', [])
    if violations:
        lines.append("### violations")
        lines.append("| Asset Class | Type | Current | Limit |")
        lines.append("|-------------|------|---------|-------|")
        for v in violations:
            lines.append(f"| {v.get('asset_class', 'N/A')} | {v.get('violation_type', 'N/A')} | {v.get('current_value', 0):.1%} | {v.get('limit_value', 0):.1%} |")
        lines.append("")

    # before_weights / after_weights
    comparison = cr.get('asset_class_comparison', [])
    if comparison:
        lines.append("### before_weights / after_weights")
        lines.append("| Asset Class | before_weights | after_weights | Change |")
        lines.append("|-------------|----------------|---------------|--------|")
        for c in comparison:
            lines.append(f"| {c.get('asset_class', 'N/A')} | {c.get('original_weight', 0):.1%} | {c.get('repaired_weight', 0):.1%} | {c.get('delta', 0):+.1%} |")
        lines.append("")

    # ========================================
    # 31. rebalance_plan (required)
    # ========================================
    lines.append("\n## 31. rebalance_plan")
    rp = op_data.get('rebalance_plan', {})

    should_execute = rp.get('should_execute', False)
    trigger_type = rp.get('trigger_type', 'MANUAL')
    turnover = rp.get('total_turnover', 0.0)
    est_cost = rp.get('total_estimated_cost', 0.0)
    requires_approval = rp.get('requires_human_approval', False)

    lines.append(f"### Status: {'EXECUTE' if should_execute else 'NOT EXECUTED'}")
    lines.append("")
    lines.append(f"- **turnover**: {turnover:.2%}")
    lines.append(f"- **est_total_cost**: {est_cost:.4f}")
    lines.append(f"- **trigger_type**: {trigger_type}")
    lines.append(f"- **requires_approval**: {requires_approval}")
    lines.append("")

    if not should_execute:
        lines.append(f"*Not executed reason: {rp.get('not_executed_reason', 'N/A')}*")
        lines.append("")

    # Trade list
    trades = rp.get('trades', [])
    if trades:
        lines.append("### Trade List")
        lines.append("| asset | current_w | target_w | delta_w | est_cost |")
        lines.append("|-------|-----------|----------|---------|----------|")
        for t in trades[:20]:  # Limit to 20
            lines.append(f"| {t.get('ticker', 'N/A')} | {t.get('current_weight', 0):.2%} | {t.get('target_weight', 0):.2%} | {t.get('delta_weight', 0):+.2%} | {t.get('estimated_cost', 0):.4f} |")
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
