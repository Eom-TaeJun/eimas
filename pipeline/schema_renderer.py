#!/usr/bin/env python3
"""
EIMAS Schema-Driven JSON → Markdown Renderer
=============================================

A flexible, schema-driven renderer that converts JSON output to Markdown
without hard-coded per-run formatting.

Features:
- Schema versioning for forward compatibility
- Normalization layer: raw JSON → ReportModel
- Generic renderer: ReportModel → Markdown
- Auto table generation for list-of-objects
- Unknown fields handled gracefully
- Validation checks for consistency

Usage:
    from pipeline.schema_renderer import render_json_to_md
    md_content = render_json_to_md(json_data)

Author: EIMAS Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import json
import re

# =============================================================================
# SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION = "1.0.0"

# =============================================================================
# SECTION REGISTRY - Priority-based ordering
# =============================================================================

@dataclass
class SectionConfig:
    """Configuration for a section in the report."""
    key: str                          # JSON key
    title: str                        # Section title (without ##)
    priority: int                     # Lower = earlier (decision_policy first)
    parent_key: Optional[str] = None  # For nested sections
    is_subsection: bool = False       # Render as ### instead of ##
    render_as_table: bool = False     # Force table rendering
    skip_if_empty: bool = True        # Skip if value is empty/None

# Section registry with priorities
# Priority groups:
# 0-99: Meta/Header
# 100-199: Operational Decision (most important)
# 200-299: Data Collection
# 300-399: Analysis
# 400-499: Technical
# 500-599: Portfolio
# 600-699: External Data
# 700-799: Debate/AI
# 800-899: Final Output
# 900-999: Extras

SECTION_REGISTRY: Dict[str, SectionConfig] = {
    # Meta
    'schema_version': SectionConfig('schema_version', 'Schema Version', 1, skip_if_empty=False),
    'timestamp': SectionConfig('timestamp', 'Timestamp', 2, skip_if_empty=False),

    # Operational Decision (Priority: 100-199)
    'hold_policy': SectionConfig('hold_policy', 'Hold Policy', 100),
    'decision_policy': SectionConfig('decision_policy', 'Decision Policy', 101),
    'score_definitions': SectionConfig('score_definitions', 'Score Definitions', 102),
    'constraint_repair': SectionConfig('constraint_repair', 'Constraint Repair', 103),
    'rebalance_plan': SectionConfig('rebalance_plan', 'Rebalance Plan', 104),
    'signal_hierarchy': SectionConfig('signal_hierarchy', 'Signal Hierarchy', 105),
    'allocation': SectionConfig('allocation', 'Allocation', 106),
    'audit_metadata': SectionConfig('audit_metadata', 'Audit Metadata', 107),

    # Final Recommendation (Priority: 110-119)
    'final_recommendation': SectionConfig('final_recommendation', 'Final Recommendation', 110),
    'confidence': SectionConfig('confidence', 'Confidence', 111),
    'risk_level': SectionConfig('risk_level', 'Risk Level', 112),
    'risk_score': SectionConfig('risk_score', 'Risk Score', 113),

    # Data Collection (Priority: 200-299)
    'fred_summary': SectionConfig('fred_summary', 'FRED Economic Data', 200),
    'market_data_count': SectionConfig('market_data_count', 'Market Data', 201),
    'crypto_data_count': SectionConfig('crypto_data_count', 'Crypto Data', 202),
    'extended_data': SectionConfig('extended_data', 'Extended Data', 203),

    # Regime & Risk (Priority: 300-349)
    'regime': SectionConfig('regime', 'Market Regime', 300),
    'base_risk_score': SectionConfig('base_risk_score', 'Base Risk Score', 310),
    'microstructure_adjustment': SectionConfig('microstructure_adjustment', 'Microstructure Adjustment', 311),
    'bubble_risk_adjustment': SectionConfig('bubble_risk_adjustment', 'Bubble Risk Adjustment', 312),
    'extended_data_adjustment': SectionConfig('extended_data_adjustment', 'Extended Data Adjustment', 313),

    # Risk Analysis (Priority: 350-399)
    'bubble_risk': SectionConfig('bubble_risk', 'Bubble Risk', 350),
    'bubble_framework': SectionConfig('bubble_framework', 'Bubble Framework', 351),
    'market_quality': SectionConfig('market_quality', 'Market Quality', 352),
    'liquidity_analysis': SectionConfig('liquidity_analysis', 'Liquidity Analysis', 360),
    'liquidity_signal': SectionConfig('liquidity_signal', 'Liquidity Signal', 361),

    # Events (Priority: 400-449)
    'events_detected': SectionConfig('events_detected', 'Events Detected', 400),
    'event_tracking': SectionConfig('event_tracking', 'Event Tracking', 401),
    'event_predictions': SectionConfig('event_predictions', 'Event Predictions', 402),
    'tracked_events': SectionConfig('tracked_events', 'Tracked Events', 403),

    # Technical Analysis (Priority: 450-499)
    'hft_microstructure': SectionConfig('hft_microstructure', 'HFT Microstructure', 450),
    'garch_volatility': SectionConfig('garch_volatility', 'GARCH Volatility', 451),
    'information_flow': SectionConfig('information_flow', 'Information Flow', 452),
    'proof_of_index': SectionConfig('proof_of_index', 'Proof of Index', 453),
    'shock_propagation': SectionConfig('shock_propagation', 'Shock Propagation', 454),
    'volume_anomalies': SectionConfig('volume_anomalies', 'Volume Anomalies', 455),
    'dtw_similarity': SectionConfig('dtw_similarity', 'DTW Similarity', 460),
    'dbscan_outliers': SectionConfig('dbscan_outliers', 'DBSCAN Outliers', 461),

    # Portfolio (Priority: 500-549)
    'portfolio_weights': SectionConfig('portfolio_weights', 'Portfolio Weights', 500),
    'allocation_strategy': SectionConfig('allocation_strategy', 'Allocation Strategy', 501),
    'allocation_config': SectionConfig('allocation_config', 'Allocation Config', 502),
    'allocation_result': SectionConfig('allocation_result', 'Allocation Result', 503),
    'adaptive_portfolios': SectionConfig('adaptive_portfolios', 'Adaptive Portfolios', 504),

    # External Data (Priority: 550-599)
    'ark_analysis': SectionConfig('ark_analysis', 'ARK Invest Analysis', 550),
    'genius_act_signals': SectionConfig('genius_act_signals', 'Genius Act Signals', 551),
    'genius_act_regime': SectionConfig('genius_act_regime', 'Genius Act Regime', 552),
    'sentiment_analysis': SectionConfig('sentiment_analysis', 'Sentiment Analysis', 560),
    'fomc_analysis': SectionConfig('fomc_analysis', 'FOMC Analysis', 561),
    'gap_analysis': SectionConfig('gap_analysis', 'Gap Analysis', 570),
    'institutional_analysis': SectionConfig('institutional_analysis', 'Institutional Analysis', 571),

    # Crypto & Global (Priority: 600-649)
    'crypto_monitoring': SectionConfig('crypto_monitoring', 'Crypto Monitoring', 600),
    'crypto_stress_test': SectionConfig('crypto_stress_test', 'Crypto Stress Test', 601),
    'onchain_risk_signals': SectionConfig('onchain_risk_signals', 'On-Chain Risk Signals', 602),
    'defi_tvl': SectionConfig('defi_tvl', 'DeFi TVL', 603),
    'mena_markets': SectionConfig('mena_markets', 'Global Markets', 610),

    # Debate & AI (Priority: 700-799)
    'debate_consensus': SectionConfig('debate_consensus', 'Multi-Agent Debate', 700),
    'debate_results': SectionConfig('debate_results', 'Debate Results', 701),
    'full_mode_position': SectionConfig('full_mode_position', 'Full Mode Position', 710),
    'reference_mode_position': SectionConfig('reference_mode_position', 'Reference Mode Position', 711),
    'modes_agree': SectionConfig('modes_agree', 'Modes Agreement', 712),
    'has_strong_dissent': SectionConfig('has_strong_dissent', 'Strong Dissent', 713),
    'dissent_records': SectionConfig('dissent_records', 'Dissent Records', 720),
    'devils_advocate_arguments': SectionConfig('devils_advocate_arguments', 'Devil\'s Advocate', 721),
    'agent_outputs': SectionConfig('agent_outputs', 'Agent Outputs', 722),
    'reasoning_chain': SectionConfig('reasoning_chain', 'Reasoning Chain', 730),
    'verification': SectionConfig('verification', 'Verification', 740),
    'validation_loop_result': SectionConfig('validation_loop_result', 'Validation Results', 741),

    # AI Report (Priority: 800-849)
    'ai_report': SectionConfig('ai_report', 'AI Report', 800),
    'whitening_summary': SectionConfig('whitening_summary', 'Whitening Summary', 801),
    'fact_check_grade': SectionConfig('fact_check_grade', 'Fact Check Grade', 802),

    # Warnings & Extras (Priority: 850-899)
    'warnings': SectionConfig('warnings', 'Warnings', 850),
    'realtime_signals': SectionConfig('realtime_signals', 'Realtime Signals', 860),
    'integrated_signals': SectionConfig('integrated_signals', 'Integrated Signals', 861),
    'trading_db_status': SectionConfig('trading_db_status', 'Trading DB Status', 870),
    'intraday_summary': SectionConfig('intraday_summary', 'Intraday Summary', 871),
    'event_backtest_results': SectionConfig('event_backtest_results', 'Backtest Results', 880),
    'event_attributions': SectionConfig('event_attributions', 'Event Attributions', 881),
    'news_correlations': SectionConfig('news_correlations', 'News Correlations', 882),
    'correlation_matrix': SectionConfig('correlation_matrix', 'Correlation Matrix', 890),
    'correlation_tickers': SectionConfig('correlation_tickers', 'Correlation Tickers', 891),
    'hrp_allocation_rationale': SectionConfig('hrp_allocation_rationale', 'HRP Rationale', 892),
    'critical_path_monitoring': SectionConfig('critical_path_monitoring', 'Critical Path', 893),
}

# Preferred column ordering for tables
TABLE_COLUMN_ORDER = [
    # Asset identifiers
    'ticker', 'asset', 'symbol', 'name', 'asset_class',
    # Actions
    'action', 'signal', 'direction',
    # Weights
    'current_weight', 'target_weight', 'delta_weight', 'weight', 'delta',
    # Values
    'current_value', 'target_value', 'value', 'score', 'price',
    # Limits
    'min', 'max', 'min_bound', 'max_bound', 'limit',
    # Costs
    'est_cost', 'estimated_cost', 'cost', 'commission', 'spread',
    # Status
    'status', 'result', 'triggered', 'is_triggered',
    # Meta
    'priority', 'source', 'timestamp', 'date',
]

# =============================================================================
# REPORT MODEL - Normalized internal structure
# =============================================================================

@dataclass
class SectionData:
    """A normalized section with metadata."""
    key: str
    title: str
    priority: int
    content: Any
    content_type: str  # 'scalar', 'dict', 'list', 'table'
    is_empty: bool = False

@dataclass
class ReportModel:
    """Normalized report model for rendering."""
    schema_version: str
    timestamp: str
    sections: List[SectionData] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    # Quick access to key values
    final_stance: Optional[str] = None
    canonical_risk_score: Optional[float] = None

    def add_section(self, section: SectionData):
        """Add a section maintaining priority order."""
        self.sections.append(section)

    def sort_sections(self):
        """Sort sections by priority."""
        self.sections.sort(key=lambda s: s.priority)

# =============================================================================
# NORMALIZER - Raw JSON → ReportModel
# =============================================================================

def _is_empty(value: Any) -> bool:
    """Check if a value should be considered empty."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False

def _detect_content_type(value: Any) -> str:
    """Detect the content type for rendering."""
    if value is None:
        return 'scalar'
    if isinstance(value, (str, int, float, bool)):
        return 'scalar'
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], dict):
            return 'table'
        return 'list'
    if isinstance(value, dict):
        return 'dict'
    return 'scalar'

def _extract_nested_operational(data: Dict, key: str) -> Any:
    """Extract from operational_report if exists, otherwise from root."""
    op_report = data.get('operational_report', {})
    if isinstance(op_report, dict) and key in op_report:
        return op_report[key]
    return data.get(key)

def normalize(raw_json: Dict) -> ReportModel:
    """
    Normalize raw JSON into a ReportModel.

    This layer handles:
    - Schema version detection
    - Flattening nested structures (operational_report)
    - Detecting content types
    - Extracting key values for validation
    """
    model = ReportModel(
        schema_version=raw_json.get('schema_version', SCHEMA_VERSION),
        timestamp=raw_json.get('timestamp', datetime.now().isoformat())
    )

    processed_keys: Set[str] = {'schema_version', 'timestamp'}

    # Process operational_report sections first (flatten them)
    op_report = raw_json.get('operational_report', {})
    if isinstance(op_report, dict):
        processed_keys.add('operational_report')
        for op_key in ['hold_policy', 'decision_policy', 'score_definitions',
                       'constraint_repair', 'rebalance_plan', 'signal_hierarchy',
                       'allocation', 'audit_metadata']:
            if op_key in op_report:
                value = op_report[op_key]
                config = SECTION_REGISTRY.get(op_key, SectionConfig(
                    op_key, op_key.replace('_', ' ').title(), 999
                ))
                content_type = _detect_content_type(value)
                model.add_section(SectionData(
                    key=op_key,
                    title=config.title,
                    priority=config.priority,
                    content=value,
                    content_type=content_type,
                    is_empty=_is_empty(value)
                ))

    # Extract key values for validation
    decision_policy = _extract_nested_operational(raw_json, 'decision_policy')
    if isinstance(decision_policy, dict):
        model.final_stance = decision_policy.get('final_stance')

    score_defs = _extract_nested_operational(raw_json, 'score_definitions')
    if isinstance(score_defs, dict):
        model.canonical_risk_score = score_defs.get('canonical_risk_score')

    # Process all other keys
    for key, value in raw_json.items():
        if key in processed_keys:
            continue

        # Skip if already processed via operational_report
        if key in ['hold_policy', 'decision_policy', 'score_definitions',
                   'constraint_repair', 'rebalance_plan', 'signal_hierarchy',
                   'allocation', 'audit_metadata']:
            if key in [s.key for s in model.sections]:
                processed_keys.add(key)
                continue

        processed_keys.add(key)

        # Get config or create default
        config = SECTION_REGISTRY.get(key)
        if config is None:
            # Unknown key - add to extra_fields
            if not _is_empty(value):
                model.extra_fields[key] = value
            continue

        content_type = _detect_content_type(value)
        is_empty = _is_empty(value)

        # Skip empty sections if configured
        if is_empty and config.skip_if_empty:
            continue

        model.add_section(SectionData(
            key=key,
            title=config.title,
            priority=config.priority,
            content=value,
            content_type=content_type,
            is_empty=is_empty
        ))

    # Sort sections by priority
    model.sort_sections()

    return model

# =============================================================================
# VALIDATORS
# =============================================================================

def validate_model(model: ReportModel, raw_json: Dict) -> List[str]:
    """
    Validate the model for consistency.

    Checks:
    - MD stance must equal decision_policy.final_stance
    - risk_score should match canonical_risk_score
    """
    errors = []

    # Check final_recommendation matches decision_policy.final_stance
    final_rec = raw_json.get('final_recommendation')
    if final_rec and model.final_stance:
        if final_rec != model.final_stance:
            errors.append(
                f"MISMATCH: final_recommendation ({final_rec}) != "
                f"decision_policy.final_stance ({model.final_stance})"
            )

    # Check risk_score matches canonical_risk_score
    risk_score = raw_json.get('risk_score')
    if risk_score is not None and model.canonical_risk_score is not None:
        if abs(risk_score - model.canonical_risk_score) > 0.01:
            errors.append(
                f"MISMATCH: risk_score ({risk_score:.2f}) != "
                f"canonical_risk_score ({model.canonical_risk_score:.2f})"
            )

    model.validation_errors = errors
    return errors

# =============================================================================
# RENDERER - ReportModel → Markdown
# =============================================================================

def _format_scalar(value: Any, key: str = '') -> str:
    """Format a scalar value for markdown."""
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        # Smart formatting based on key name
        key_lower = key.lower()
        if 'weight' in key_lower or 'ratio' in key_lower:
            return f'{value:.2%}'
        if 'score' in key_lower:
            return f'{value:.2f}'
        if 'confidence' in key_lower and value <= 1:
            return f'{value:.1%}'
        if 'cost' in key_lower or 'price' in key_lower:
            return f'{value:.4f}'
        if abs(value) < 0.01 and value != 0:
            return f'{value:.4f}'
        return f'{value:.2f}'
    return str(value)

def _build_table(items: List[Dict], preferred_columns: List[str] = None) -> str:
    """
    Auto-generate a markdown table from a list of dicts.

    Uses preferred column ordering if keys match common names.
    Appends extra keys as additional columns.
    """
    if not items:
        return '*No data*\n'

    # Collect all unique keys across all items
    all_keys: Set[str] = set()
    for item in items:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    if not all_keys:
        return '*No data*\n'

    # Order columns: preferred first, then remaining alphabetically
    preferred = preferred_columns or TABLE_COLUMN_ORDER
    ordered_columns = []
    for col in preferred:
        if col in all_keys:
            ordered_columns.append(col)
            all_keys.remove(col)
    ordered_columns.extend(sorted(all_keys))

    # Build header
    header = '| ' + ' | '.join(ordered_columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(ordered_columns)) + ' |'

    # Build rows
    rows = []
    for item in items[:50]:  # Limit to 50 rows
        if isinstance(item, dict):
            row_values = []
            for col in ordered_columns:
                val = item.get(col, '')
                row_values.append(_format_scalar(val, col))
            rows.append('| ' + ' | '.join(row_values) + ' |')
        else:
            rows.append(f'| {item} |' + ' |' * (len(ordered_columns) - 1))

    if len(items) > 50:
        rows.append(f'| ... | *{len(items) - 50} more rows* |' + ' |' * (len(ordered_columns) - 2))

    return '\n'.join([header, separator] + rows) + '\n'

def _render_dict(data: Dict, level: int = 0, max_depth: int = 3) -> str:
    """Render a dict as markdown with proper indentation."""
    if not data:
        return '*Empty*\n'

    if level >= max_depth:
        # Too deep - render as compact JSON
        return f'```json\n{json.dumps(data, indent=2, default=str)[:500]}\n```\n'

    lines = []
    indent = '  ' * level

    for key, value in data.items():
        if _is_empty(value):
            continue

        key_display = key.replace('_', ' ').title() if level == 0 else key

        if isinstance(value, dict):
            lines.append(f'{indent}- **{key_display}**:')
            lines.append(_render_dict(value, level + 1, max_depth))
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                lines.append(f'{indent}- **{key_display}**:')
                lines.append(_build_table(value))
            else:
                items_str = ', '.join(str(v) for v in value[:10])
                if len(value) > 10:
                    items_str += f', ... (+{len(value) - 10} more)'
                lines.append(f'{indent}- **{key_display}**: {items_str}')
        else:
            lines.append(f'{indent}- **{key_display}**: {_format_scalar(value, key)}')

    return '\n'.join(lines) + '\n'

def _render_list(items: List, key: str = '') -> str:
    """Render a list as markdown."""
    if not items:
        return '*No items*\n'

    # Check if it's a list of dicts (render as table)
    if isinstance(items[0], dict):
        return _build_table(items)

    # Simple list
    lines = []
    for i, item in enumerate(items[:20]):
        lines.append(f'- {_format_scalar(item, key)}')

    if len(items) > 20:
        lines.append(f'- ... (+{len(items) - 20} more)')

    return '\n'.join(lines) + '\n'

def _render_section(section: SectionData) -> str:
    """Render a single section to markdown."""
    lines = [f'\n## {section.title}\n']

    if section.is_empty:
        lines.append('*No data*\n')
        return '\n'.join(lines)

    content = section.content

    # Special rendering for known section types
    if section.key == 'decision_policy':
        lines.extend(_render_decision_policy(content))
    elif section.key == 'score_definitions':
        lines.extend(_render_score_definitions(content))
    elif section.key == 'hold_policy':
        lines.extend(_render_hold_policy(content))
    elif section.key == 'constraint_repair':
        lines.extend(_render_constraint_repair(content))
    elif section.key == 'rebalance_plan':
        lines.extend(_render_rebalance_plan(content))
    elif section.content_type == 'table':
        lines.append(_build_table(content))
    elif section.content_type == 'dict':
        lines.append(_render_dict(content))
    elif section.content_type == 'list':
        lines.append(_render_list(content, section.key))
    else:
        lines.append(f'{_format_scalar(content, section.key)}\n')

    return '\n'.join(lines)

# =============================================================================
# SPECIAL SECTION RENDERERS
# =============================================================================

def _render_decision_policy(data: Dict) -> List[str]:
    """Render decision_policy section with rule evaluation log."""
    lines = []

    final_stance = data.get('final_stance', 'N/A')
    lines.append(f'### Final Stance: **{final_stance}**\n')

    # Key inputs
    lines.append('#### Decision Inputs')
    lines.append('| Input | Value |')
    lines.append('|-------|-------|')
    for key in ['regime_input', 'risk_score_input', 'confidence_input',
                'agent_consensus_input', 'modes_agree_input',
                'constraint_status_input', 'client_profile_status_input']:
        if key in data:
            val = data[key]
            lines.append(f'| {key} | {_format_scalar(val, key)} |')
    lines.append('')

    # Rule evaluation log
    rule_log = data.get('rule_evaluation_log', [])
    if rule_log:
        lines.append('#### Rule Evaluation Log')
        lines.append('| Rule | Condition | Input | Result |')
        lines.append('|------|-----------|-------|--------|')
        for rule in rule_log:
            result = rule.get('result', '')
            result_fmt = f'**{result}**' if 'TRIGGERED' in result else result
            lines.append(f"| {rule.get('rule', '')} | {rule.get('condition', '')} | {rule.get('input', '')} | {result_fmt} |")
        lines.append('')

    # Applied rules
    applied = data.get('applied_rules', [])
    if applied:
        lines.append('#### Applied Rules')
        for rule in applied:
            lines.append(f'- {rule}')
        lines.append('')

    # Reason codes
    codes = data.get('reason_codes', [])
    if codes:
        lines.append('#### Reason Codes')
        lines.append(', '.join(f'`{c}`' for c in codes))
        lines.append('')

    return lines

def _render_score_definitions(data: Dict) -> List[str]:
    """Render score_definitions with emphasis on canonical score."""
    lines = []

    canonical = data.get('canonical_risk_score', 0)
    risk_level = data.get('risk_level', 'MEDIUM')

    lines.append('> **Note**: Only `canonical_risk_score` is used for decisions.')
    lines.append('> Other scores are auxiliary reference only.')
    lines.append('')
    lines.append(f'### Canonical Risk Score: **{canonical:.1f} / 100** ({risk_level})')
    lines.append('')

    # Scale
    lines.append('| Range | Level | Action |')
    lines.append('|-------|-------|--------|')
    lines.append('| 0-30 | LOW | Aggressive allowed |')
    lines.append('| 30-70 | MEDIUM | Standard management |')
    lines.append('| 70-100 | HIGH | Defensive required |')
    lines.append('')

    # Auxiliary scores
    aux = data.get('auxiliary_sub_scores', {})
    if aux:
        lines.append('#### Auxiliary Sub-Scores (Reference Only)')
        lines.append('| Component | Value | Source |')
        lines.append('|-----------|-------|--------|')
        for key, val in aux.items():
            if isinstance(val, dict):
                v = val.get('value', 0)
                src = val.get('source', 'N/A')
                lines.append(f'| {key} | {v:+.1f} | {src} |')
            else:
                lines.append(f'| {key} | {val} | - |')
        lines.append('')

    return lines

def _render_hold_policy(data: Dict) -> List[str]:
    """Render hold_policy with condition evaluation."""
    lines = []

    is_hold = data.get('is_hold', False)
    status = '**HOLD TRIGGERED**' if is_hold else '**PROCEED**'
    lines.append(f'### Status: {status}')
    lines.append('')

    conditions = data.get('hold_conditions', [])
    if conditions:
        lines.append('#### Hold Conditions')
        lines.append('| Priority | Condition | Triggered | Current | Threshold |')
        lines.append('|----------|-----------|-----------|---------|-----------|')
        for c in conditions:
            triggered = c.get('is_triggered', False)
            trig_str = '**YES**' if triggered else 'NO'
            lines.append(f"| {c.get('priority', '-')} | {c.get('condition_name', '')} | {trig_str} | {c.get('current_value', '')} | {c.get('threshold', '')} |")
        lines.append('')

    return lines

def _render_constraint_repair(data: Dict) -> List[str]:
    """Render constraint_repair with before/after comparison."""
    lines = []

    satisfied = data.get('constraints_satisfied', True)
    force_hold = data.get('force_hold', False)

    if force_hold:
        lines.append('### Status: **FORCE HOLD**')
        lines.append(f"> Reason: {data.get('force_hold_reason', 'N/A')}")
    elif satisfied:
        lines.append('### Status: **SATISFIED** ✓')
    else:
        lines.append('### Status: **REPAIRED** 🔧')
    lines.append('')

    # Violations
    violations = data.get('violations_found', [])
    if violations:
        lines.append('#### Violations')
        lines.append(_build_table(violations))

    # Comparison
    comparison = data.get('asset_class_comparison', [])
    if comparison:
        lines.append('#### Before/After Weights')
        lines.append(_build_table(comparison))

    # Repair actions
    actions = data.get('repair_actions', [])
    if actions:
        lines.append('#### Repair Actions')
        for a in actions:
            lines.append(f'- {a}')
        lines.append('')

    return lines

def _render_rebalance_plan(data: Dict) -> List[str]:
    """Render rebalance_plan with trade list."""
    lines = []

    execution = data.get('execution', {})
    should_execute = execution.get('should_execute', data.get('should_execute', False))

    status = 'EXECUTE' if should_execute else 'NOT EXECUTED'
    lines.append(f'### Status: **{status}**')

    if not should_execute:
        reason = execution.get('not_executed_reason', data.get('not_executed_reason', ''))
        if reason:
            lines.append(f'> Reason: {reason}')
    lines.append('')

    # Summary
    summary = data.get('summary', {})
    lines.append('#### Summary')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.append(f"| Turnover | {summary.get('total_turnover', 0):.2%} |")
    lines.append(f"| Buy Orders | {summary.get('buy_count', 0)} |")
    lines.append(f"| Sell Orders | {summary.get('sell_count', 0)} |")
    lines.append('')

    # Cost breakdown
    cost = data.get('cost_breakdown', {})
    if cost:
        lines.append('#### Cost Breakdown')
        lines.append('| Type | Amount |')
        lines.append('|------|--------|')
        for k, v in cost.items():
            lines.append(f'| {k} | {v:.4f} |')
        lines.append('')

    # Approval
    approval = data.get('approval', {})
    requires = approval.get('requires_human_approval', False)
    if requires:
        lines.append(f"**⚠️ Human Approval Required**: {approval.get('approval_reason', '')}")
        lines.append('')

    # Trade list
    trades = data.get('trades', [])
    if trades:
        lines.append('#### Trade List')
        lines.append(_build_table(trades))

    return lines

# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_md(model: ReportModel) -> str:
    """
    Render a ReportModel to Markdown.

    This is the main rendering function that produces the final output.
    """
    lines = []

    # Header
    lines.append('# EIMAS Analysis Report')
    lines.append('')
    lines.append(f'**Generated**: {model.timestamp}')
    lines.append(f'**Schema Version**: {model.schema_version}')
    lines.append('')

    # Validation errors
    if model.validation_errors:
        lines.append('## ⚠️ Validation Warnings')
        for err in model.validation_errors:
            lines.append(f'- {err}')
        lines.append('')

    # Render all sections
    section_num = 1
    for section in model.sections:
        if section.is_empty and section.key not in ['schema_version', 'timestamp']:
            continue

        # Add section number to title
        section_content = _render_section(section)
        # Replace first ## with numbered ##
        section_content = section_content.replace(
            f'\n## {section.title}\n',
            f'\n## {section_num}. {section.title}\n',
            1
        )
        lines.append(section_content)
        section_num += 1

    # Extra fields
    if model.extra_fields:
        lines.append(f'\n## {section_num}. Extra Fields')
        lines.append('')
        lines.append('> These fields were not recognized by the schema.')
        lines.append('')
        lines.append('```json')
        lines.append(json.dumps(model.extra_fields, indent=2, default=str)[:2000])
        if len(json.dumps(model.extra_fields, default=str)) > 2000:
            lines.append('... (truncated)')
        lines.append('```')
        lines.append('')

    # Footer
    lines.append('---')
    lines.append(f'*Rendered by EIMAS Schema Renderer v{SCHEMA_VERSION}*')

    return '\n'.join(lines)

def render_json_to_md(raw_json: Dict) -> str:
    """
    Main entry point: Convert raw JSON to Markdown.

    Usage:
        md_content = render_json_to_md(json_data)
    """
    # Normalize
    model = normalize(raw_json)

    # Validate
    validate_model(model, raw_json)

    # Render
    return render_md(model)

# =============================================================================
# CLI / TEST
# =============================================================================

def find_latest_json(output_dir: str = "outputs") -> Optional[str]:
    """Find the latest eimas_*.json file in the output directory."""
    from pathlib import Path
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    json_files = list(output_path.glob("eimas_*.json"))
    if not json_files:
        # Fallback to integrated_*.json
        json_files = list(output_path.glob("integrated_*.json"))

    if not json_files:
        return None

    # Sort by modification time (newest first)
    latest = max(json_files, key=lambda x: x.stat().st_mtime)
    return str(latest)


def render_latest(output_dir: str = "outputs") -> Tuple[str, str]:
    """
    Find and render the latest eimas_*.json file.

    Returns:
        Tuple of (json_path, md_path)
    """
    json_path = find_latest_json(output_dir)
    if not json_path:
        raise FileNotFoundError(f"No eimas_*.json found in {output_dir}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    md_content = render_json_to_md(data)

    # Output path: same name but .schema.md
    from pathlib import Path
    md_path = Path(json_path).with_suffix('.schema.md')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    return json_path, str(md_path)


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # If no arguments, find and render the latest eimas_*.json
    if len(sys.argv) < 2:
        print("=" * 60)
        print("EIMAS Schema-Driven Renderer")
        print("=" * 60)

        try:
            json_path, md_path = render_latest()
            print(f"\n✓ Input:  {json_path}")
            print(f"✓ Output: {md_path}")

            with open(md_path, 'r') as f:
                content = f.read()
            print(f"\n  - Sections: {content.count('## ')}")
            print(f"  - Lines: {len(content.splitlines())}")

            # Show validation status
            with open(json_path, 'r') as f:
                data = json.load(f)
            model = normalize(data)
            errors = validate_model(model, data)
            if errors:
                print(f"\n⚠️ Validation Warnings:")
                for e in errors:
                    print(f"   - {e}")
            else:
                print(f"\n✓ Validation: PASSED")

        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print("\nUsage:")
            print("  python -m pipeline.schema_renderer           # Render latest eimas_*.json")
            print("  python -m pipeline.schema_renderer <file>    # Render specific file")
            sys.exit(1)
    else:
        # Render specific file
        json_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else json_path.with_suffix('.schema.md')

        if not json_path.exists():
            print(f"✗ File not found: {json_path}")
            sys.exit(1)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        md_content = render_json_to_md(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✓ Rendered: {output_path}")
        print(f"  - Sections: {md_content.count('## ')}")
        print(f"  - Lines: {len(md_content.splitlines())}")
